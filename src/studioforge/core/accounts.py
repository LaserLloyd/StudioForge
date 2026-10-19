"""Accounts and scoped API keys.

The one place that knows how credentials are stored and checked, so the gateway
middleware (``api/auth.py``), the control panel's sign-in, the CLI and the MCP
policy all agree:

* **Owner account** -- a username plus an argon2id password hash. The hash is
  *not* in the database: it goes through :class:`studioforge.secret_store.SecretStore`
  (OS keyring, else an owner-only file), so a copied ``registry.sqlite3`` carries
  no login material. The ``accounts`` row holds only the username and clocks.
* **API keys** -- ``sf-<scope letter>-<base62>`` tokens (see :mod:`studioforge.authkeys`).
  Only the SHA-256 is stored. Each key has a scope (``read`` < ``inference`` <
  ``admin``), an optional expiry and requests-per-minute limit, a last-used time
  and a revoked flag.
* **Legacy key** -- an existing ``server.api_key`` is mirrored into the table as
  an ``admin`` key named ``legacy`` the first time it is seen, so it shows up in
  key lists and can be revoked, while the config key keeps working unchanged.

Every mutation rewrites ``keys.snapshot.json`` beside ``config.yaml`` -- the
file the watchdog (which must never open the database) verifies keys against.

Thread-safe: sqlite access goes through :class:`studioforge.db.Database`, the
rate limiter and last-used throttle are guarded by their own lock.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studioforge.authkeys import (
    SCOPES,
    KeyRecord,
    Principal,
    generate_token,
    hash_token,
    looks_like_key,
    write_snapshot,
)
from studioforge.db import Database
from studioforge.errors import AuthError, BadRequestError
from studioforge.logging import get_logger
from studioforge.secret_store import SecretStore
from studioforge.secure_files import write_private

log = get_logger(__name__)

__all__ = ["Account", "AccountStore", "LEGACY_KEY_NAME", "MIN_PASSWORD_LENGTH"]

LEGACY_KEY_NAME = "legacy"
LOCAL_CLI_KEY_NAME = "local-cli"
LOCAL_CLI_KEY_FILE = "cli.key"
#: Written when an account is created or about to be deleted: "an owner existed here". Lets a
#: network-bound server keep requiring a credential after its last account is force-deleted,
#: the way ``had_keys_cached`` does for keys (a bind rule is only checked at start).
OWNER_EVER_FILE = ".had-owner"

#: Long enough that argon2id's cost is the weak point, not the password.
MIN_PASSWORD_LENGTH = 10

#: ``last_used_at`` is written at most this often per key: a busy inference
#: client would otherwise turn every request into a database write.
_TOUCH_INTERVAL_S = 30.0

#: How long "are there any active keys?" is remembered between requests.
_POSTURE_TTL_S = 2.0


@dataclass(frozen=True)
class Account:
    id: str
    username: str
    role: str
    created_at: float
    last_login_at: float | None = None
    disabled: bool = False

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "disabled": self.disabled,
        }


def _row_to_key(row: dict[str, Any]) -> KeyRecord:
    return KeyRecord(
        id=str(row["id"]),
        name=str(row["name"]),
        scope=str(row["scope"]),
        key_hash=str(row["key_hash"]),
        prefix=str(row["prefix"]),
        created_at=float(row["created_at"]),
        expires_at=None if row["expires_at"] is None else float(row["expires_at"]),
        last_used_at=None if row["last_used_at"] is None else float(row["last_used_at"]),
        rate_limit_rpm=None if row["rate_limit_rpm"] is None else int(row["rate_limit_rpm"]),
        revoked_at=None if row["revoked_at"] is None else float(row["revoked_at"]),
    )


def _row_to_account(row: dict[str, Any]) -> Account:
    return Account(
        id=str(row["id"]),
        username=str(row["username"]),
        role=str(row["role"]),
        created_at=float(row["created_at"]),
        last_login_at=None if row["last_login_at"] is None else float(row["last_login_at"]),
        disabled=bool(row["disabled"]),
    )


class AccountStore:
    """Owner accounts and API keys over the registry database."""

    def __init__(
        self,
        db: Database,
        data_dir: Path,
        *,
        secrets: SecretStore | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.db = db
        self.data_dir = Path(data_dir)
        self.secrets = secrets or SecretStore(self.data_dir)
        self._clock = clock
        self._lock = threading.Lock()
        self._touched: dict[str, float] = {}
        self._windows: dict[str, deque[float]] = {}
        self._posture: tuple[float, bool, bool, bool, bool] | None = None

    # ------------------------------------------------------------------
    # Keys: create / list / revoke
    # ------------------------------------------------------------------

    def create_key(
        self,
        name: str,
        scope: str,
        *,
        expires_at: float | None = None,
        expires_in_days: float | None = None,
        rate_limit_rpm: int | None = None,
        created_by: str | None = None,
    ) -> tuple[KeyRecord, str]:
        """Create a key. Returns ``(record, token)``; the token is shown once and not stored."""
        clean = (name or "").strip()
        if not clean:
            raise BadRequestError("A key needs a name.", code="invalid_request", param="name")
        if scope not in SCOPES:
            raise BadRequestError(
                f"Unknown scope {scope!r}; use one of {', '.join(SCOPES)}.",
                code="invalid_request",
                param="scope",
            )
        if rate_limit_rpm is not None and rate_limit_rpm < 1:
            raise BadRequestError(
                "rate_limit_rpm must be at least 1 request per minute (omit it for no limit).",
                code="invalid_request",
                param="rate_limit_rpm",
            )
        now = self._clock()
        if expires_at is None and expires_in_days is not None:
            if expires_in_days <= 0:
                raise BadRequestError(
                    "expires_in_days must be positive.", code="invalid_request", param="expires"
                )
            expires_at = now + expires_in_days * 86400.0
        token = generate_token(scope)
        record = KeyRecord(
            id=uuid.uuid4().hex[:12],
            name=clean,
            scope=scope,
            key_hash=hash_token(token),
            prefix=token[:9],
            created_at=now,
            expires_at=expires_at,
            rate_limit_rpm=rate_limit_rpm,
        )
        self._insert(record, created_by=created_by)
        log.info("api key created", key_id=record.id, name=record.name, scope=record.scope)
        return record, token

    def _insert(self, record: KeyRecord, *, created_by: str | None) -> None:
        self.db.execute_write(
            "INSERT INTO api_keys (id, name, scope, key_hash, prefix, created_at, expires_at, "
            "rate_limit_rpm, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.name,
                record.scope,
                record.key_hash,
                record.prefix,
                record.created_at,
                record.expires_at,
                record.rate_limit_rpm,
                created_by,
            ),
        )
        self._invalidate_posture()
        self.sync_snapshot()

    def list_keys(self, *, include_revoked: bool = True) -> list[KeyRecord]:
        rows = self.db.query("SELECT * FROM api_keys ORDER BY created_at, id")
        keys = [_row_to_key(r) for r in rows]
        return keys if include_revoked else [k for k in keys if not k.revoked]

    def find_key(self, ident: str) -> KeyRecord | None:
        """By id, or by name when that is unambiguous among live keys."""
        keys = self.list_keys()
        for key in keys:
            if key.id == ident:
                return key
        named = [k for k in keys if k.name == ident and not k.revoked]
        return named[0] if len(named) == 1 else None

    def revoke_key(self, ident: str) -> KeyRecord | None:
        key = self.find_key(ident)
        if key is None:
            return None
        if not key.revoked:
            self.db.execute_write(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ?", (self._clock(), key.id)
            )
            self._invalidate_posture()
            self.sync_snapshot()
            log.info("api key revoked", key_id=key.id, name=key.name)
        return self.find_key(key.id)

    def has_active_keys(self) -> bool:
        return any(k.active(self._clock()) for k in self.list_keys())

    def posture(self) -> tuple[bool, bool]:
        """``(any active key, any owner account)``, remembered for a few seconds.

        Asked on every request (is a credential required?), so it must not be a
        query each time. The CLI creates keys straight in the database while the
        server runs, hence a short TTL rather than only invalidating on our own
        writes.
        """
        cached = self._cached_posture()
        return cached[1], cached[2]

    def _cached_posture(self) -> tuple[float, bool, bool, bool, bool]:
        """``(stamp, active key, enabled account, any key ever, any account ever)``."""
        now = self._clock()
        with self._lock:
            cached = self._posture
            if cached is not None and now - cached[0] < _POSTURE_TTL_S:
                return cached
        rows = self.list_keys()
        owner = self.has_owner()
        ever = owner or self._had_owner()
        fresh = (now, any(k.active(now) for k in rows), owner, bool(rows), ever)
        with self._lock:
            self._posture = fresh
        return fresh

    def has_active_keys_cached(self) -> bool:
        return self._cached_posture()[1]

    def has_owner_cached(self) -> bool:
        return self._cached_posture()[2]

    def had_keys_cached(self) -> bool:
        """Whether any key was ever created here (revoked and expired ones count).

        A network-bound server keeps requiring a credential after its last key
        is revoked: the bind rule is only checked at start, and revoking the last
        key must not silently reopen the server to the network.
        """
        return self._cached_posture()[3]

    def had_owner_cached(self) -> bool:
        """Whether an account ever existed here (deleted ones count).

        Same reason as :meth:`had_keys_cached`: deleting the last account must not
        silently reopen a network-bound server whose bind rule is only checked at start.
        """
        return self._cached_posture()[4]

    def _had_owner(self) -> bool:
        return (self.data_dir / OWNER_EVER_FILE).is_file()

    def _mark_had_owner(self) -> None:
        with contextlib.suppress(OSError):
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / OWNER_EVER_FILE).write_text("an account existed", encoding="utf-8")

    def has_active_admin_key(self) -> bool:
        return any(k.active(self._clock()) and k.scope == "admin" for k in self.list_keys())

    def _invalidate_posture(self) -> None:
        with self._lock:
            self._posture = None

    def sync_snapshot(self) -> None:
        """Rewrite ``keys.snapshot.json`` (what the watchdog verifies against)."""
        try:
            keys = self.list_keys()
            write_snapshot(
                self.data_dir,
                [k for k in keys if not k.revoked],
                revoked_legacy=[
                    k.key_hash for k in keys if k.revoked and k.name == LEGACY_KEY_NAME
                ],
            )
        except OSError as exc:
            log.warning("could not write the key snapshot for the watchdog", error=str(exc))

    # ------------------------------------------------------------------
    # Keys: authenticate
    # ------------------------------------------------------------------

    def authenticate(self, token: str | None, *, count: bool = True) -> Principal | None:
        """The principal for a presented token, ``None`` if it is not a stored key.

        Raises :class:`AuthError` for a key that *is* ours but cannot be used:
        revoked, expired, or over its rate limit -- each with its own catalogued
        code, so the client is told which of the three it is instead of "wrong
        key". ``count=False`` re-checks a token that was already admitted (the
        MCP tool layer) without spending the key's rate-limit budget or touching
        ``last_used_at``.
        """
        if not token or not looks_like_key(token):
            return None
        rows = self.db.query("SELECT * FROM api_keys WHERE key_hash = ?", (hash_token(token),))
        if not rows:
            return None
        return self._admit(_row_to_key(rows[0]), count=count)

    def _admit(self, key: KeyRecord, *, count: bool = True) -> Principal:
        now = self._clock()
        if key.revoked:
            raise AuthError(
                f"The API key '{key.name}' was revoked.",
                code="api_key_revoked",
            )
        if key.expired(now):
            raise AuthError(
                f"The API key '{key.name}' expired.",
                code="api_key_expired",
            )
        if count:
            wait = self._rate_check(key, now)
            if wait > 0:
                seconds = max(1, int(wait + 0.999))
                raise AuthError(
                    f"The API key '{key.name}' is limited to {key.rate_limit_rpm} requests per "
                    f"minute. Retry in {seconds}s.",
                    code="key_rate_limited",
                    status_code=429,
                    details={"retry_after_s": seconds, "limit_rpm": key.rate_limit_rpm},
                )
            self._touch(key.id, now)
        return Principal("key", key.id, key.name, key.scope)

    def _rate_check(self, key: KeyRecord, now: float) -> float:
        """Seconds to wait if ``key`` is over its limit, else 0 (and count this request)."""
        limit = key.rate_limit_rpm
        if not limit:
            return 0.0
        with self._lock:
            window = self._windows.setdefault(key.id, deque())
            while window and now - window[0] >= 60.0:
                window.popleft()
            if len(window) >= limit:
                return max(0.0, 60.0 - (now - window[0]))
            window.append(now)
        return 0.0

    def _touch(self, key_id: str, now: float) -> None:
        with self._lock:
            last = self._touched.get(key_id)
            if last is not None and now - last < _TOUCH_INTERVAL_S:
                return
            self._touched[key_id] = now
        try:
            self.db.execute_write(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id)
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping must never fail a request
            log.debug("could not record key use", error=str(exc))

    # ------------------------------------------------------------------
    # The legacy server.api_key
    # ------------------------------------------------------------------

    def adopt_legacy_key(self, api_key: str | None) -> KeyRecord | None:
        """Mirror ``server.api_key`` into the table as the admin key ``legacy``.

        Idempotent. A changed config key revokes the old mirror and adds a new
        one; a cleared key revokes it. A mirror the operator revoked stays
        revoked (the hash is remembered), so ``keys revoke legacy`` really
        disables the config key too.
        """
        live = [k for k in self.list_keys() if k.name == LEGACY_KEY_NAME and not k.revoked]
        if not api_key:
            for key in live:
                self.revoke_key(key.id)
            return None
        digest = hash_token(api_key)
        existing = self.db.query("SELECT * FROM api_keys WHERE key_hash = ?", (digest,))
        if existing:
            return _row_to_key(existing[0])
        for key in live:
            self.revoke_key(key.id)
        record = KeyRecord(
            id=uuid.uuid4().hex[:12],
            name=LEGACY_KEY_NAME,
            scope="admin",
            key_hash=digest,
            prefix=(api_key[:4] + "...")[:9],
            created_at=self._clock(),
        )
        self._insert(record, created_by="migration")
        log.info(
            "server.api_key migrated into the key table as the admin key 'legacy'; the config "
            "key still works but is deprecated (create scoped keys with `studioforge keys create`)"
        )
        return record

    def authenticate_legacy(self, api_key: str) -> Principal | None:
        """The principal for a presented ``server.api_key`` that matched the config.

        ``None`` only when the operator revoked the ``legacy`` mirror.
        """
        record = self.adopt_legacy_key(api_key)
        if record is None:
            return Principal("legacy", "", LEGACY_KEY_NAME, "admin")
        if record.revoked:
            raise AuthError("The legacy server.api_key was revoked.", code="api_key_revoked")
        if record.expired(self._clock()):
            raise AuthError("The legacy server.api_key expired.", code="api_key_expired")
        self._touch(record.id, self._clock())
        return Principal("legacy", record.id, LEGACY_KEY_NAME, "admin")

    # ------------------------------------------------------------------
    # The local CLI key
    # ------------------------------------------------------------------

    @property
    def local_cli_key_path(self) -> Path:
        return self.data_dir / LOCAL_CLI_KEY_FILE

    def local_cli_key(self) -> str:
        """The admin token the CLI on this machine uses. Created on first need.

        A file in the data directory (owner-only): whoever can read it can already
        read the database, so it widens nothing. If the file's key was revoked or
        removed, a fresh one replaces it.
        """
        path = self.local_cli_key_path
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
        if token:
            rows = self.db.query("SELECT * FROM api_keys WHERE key_hash = ?", (hash_token(token),))
            if rows and _row_to_key(rows[0]).active(self._clock()):
                return token
        for key in self.list_keys():
            if key.name == LOCAL_CLI_KEY_NAME and not key.revoked:
                self.revoke_key(key.id)
        _record, fresh = self.create_key(LOCAL_CLI_KEY_NAME, "admin", created_by="cli")
        write_private(path, fresh)
        return fresh

    # ------------------------------------------------------------------
    # Owner accounts
    # ------------------------------------------------------------------

    @staticmethod
    def _hasher() -> Any:
        from argon2 import PasswordHasher

        return PasswordHasher()

    def _hash_name(self, username: str) -> str:
        return f"owner.{username.lower()}.hash"

    def has_owner(self) -> bool:
        rows = self.db.query("SELECT COUNT(*) AS n FROM accounts WHERE disabled = 0")
        return bool(rows and rows[0]["n"])

    def list_accounts(self) -> list[Account]:
        rows = self.db.query("SELECT * FROM accounts ORDER BY created_at")
        return [_row_to_account(r) for r in rows]

    def get_account(self, username: str) -> Account | None:
        rows = self.db.query(
            "SELECT * FROM accounts WHERE lower(username) = lower(?)", ((username or "").strip(),)
        )
        return _row_to_account(rows[0]) if rows else None

    @staticmethod
    def _check_password(password: str) -> None:
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise BadRequestError(
                f"The password must be at least {MIN_PASSWORD_LENGTH} characters.",
                code="invalid_request",
                param="password",
            )

    def create_account(self, username: str, password: str) -> Account:
        """Create an account. The first one is the ``owner``; later ones are ``admin``."""
        name = (username or "").strip()
        if not name or len(name) > 64 or any(c.isspace() or c in "/\\:" for c in name):
            raise BadRequestError(
                "The username must be 1-64 characters with no spaces, slashes or colons.",
                code="invalid_request",
                param="username",
            )
        self._check_password(password)
        if self.get_account(name) is not None:
            raise BadRequestError(
                f"An account named '{name}' already exists. Use `studioforge users passwd` to "
                "change its password.",
                code="invalid_request",
                param="username",
            )
        role = "admin" if self.has_owner() else "owner"
        now = self._clock()
        account_id = uuid.uuid4().hex[:12]
        self.secrets.set(self._hash_name(name), self._hasher().hash(password))
        self.db.execute_write(
            "INSERT INTO accounts (id, username, role, created_at, password_changed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (account_id, name, role, now, now),
        )
        self._mark_had_owner()
        self._invalidate_posture()
        log.info("account created", username=name, role=role)
        return Account(account_id, name, role, now)

    def set_password(self, username: str, password: str) -> Account:
        account = self.get_account(username)
        if account is None:
            raise BadRequestError(
                f"There is no account named '{username}'. List accounts with "
                "`studioforge users list`.",
                code="invalid_request",
                param="username",
            )
        self._check_password(password)
        self.secrets.set(self._hash_name(account.username), self._hasher().hash(password))
        self.db.execute_write(
            "UPDATE accounts SET password_changed_at = ? WHERE id = ?", (self._clock(), account.id)
        )
        log.info("account password changed", username=account.username)
        return account

    def owner_username(self) -> str | None:
        """The owner's username (else the oldest account's), ``None`` when there are none."""
        accounts = self.list_accounts()
        for account in accounts:
            if account.role == "owner":
                return account.username
        return accounts[0].username if accounts else None

    def reset_login(self, username: str, password: str) -> tuple[Account, bool]:
        """Set ``username``'s password, creating the account when it does not exist.

        Returns ``(account, created)``. An existing account is also re-enabled.
        The lockout recovery path (``<data>/RESET_LOGIN``, see
        :mod:`studioforge.login_reset`); the CLI uses ``create_account`` and
        ``set_password`` directly.
        """
        existing = self.get_account(username)
        if existing is None:
            return self.create_account(username, password), True
        account = self.set_password(existing.username, password)
        if existing.disabled:
            self.db.execute_write("UPDATE accounts SET disabled = 0 WHERE id = ?", (account.id,))
            self._invalidate_posture()
        return account, False

    def delete_account(self, username: str, *, force: bool = False) -> Account:
        """Remove an account and its password hash; its panel sessions end with it.

        Refuses, unless ``force``, a deletion that would leave no ENABLED account and
        no active admin key: nothing could sign in, and a network-bound server would
        refuse to start (``auth_required_for_remote_bind``). Even forced, an account
        having existed is remembered (``.had-owner``), so a running network-bound
        server keeps requiring a credential instead of opening up. When the owner is
        removed and enabled accounts remain, the oldest becomes the owner. API keys
        are not touched (they are revoked with ``keys revoke``).
        """
        account = self.get_account(username)
        if account is None:
            raise BadRequestError(
                f"There is no account named '{username}'. List accounts with "
                "`studioforge users list`.",
                code="invalid_request",
                param="username",
            )
        others = [a for a in self.list_accounts() if a.id != account.id]
        enabled_others = [a for a in others if not a.disabled]
        if not enabled_others and not self.has_active_admin_key() and not force:
            raise BadRequestError(
                f"'{account.username}' is the last account that can sign in and no active admin "
                "key exists; deleting it would leave nothing to sign in with (and a "
                "network-bound server would keep refusing everyone). Create another with "
                "`studioforge users add <name>` or an admin key (`studioforge keys create "
                "--scope admin`) first, or pass --force to delete it anyway.",
                code="last_account",
                param="username",
            )
        self._mark_had_owner()
        self.db.execute_write("DELETE FROM accounts WHERE id = ?", (account.id,))
        self.secrets.delete(self._hash_name(account.username))
        if (
            account.role == "owner"
            and enabled_others
            and not any(a.role == "owner" for a in enabled_others)
        ):
            self.db.execute_write(
                "UPDATE accounts SET role = 'owner' WHERE id = ?", (enabled_others[0].id,)
            )
        self._invalidate_posture()
        log.info("account deleted", username=account.username, role=account.role)
        return account

    def verify_password(self, username: str, password: str) -> Account | None:
        """The account when ``password`` is right, else ``None``. Constant-ish time on a miss."""
        account = self.get_account(username)
        hasher = self._hasher()
        stored = self.secrets.get(self._hash_name(account.username)) if account else None
        if account is None or account.disabled or not stored:
            # Burn a hash so "no such user" is not faster than "wrong password".
            with contextlib.suppress(Exception):
                hasher.verify(_dummy_hash(hasher), password or "x")
            return None
        try:
            hasher.verify(stored, password or "")
        except Exception:  # noqa: BLE001 - VerifyMismatch, InvalidHash, ...
            return None
        if hasher.check_needs_rehash(stored):
            self.secrets.set(self._hash_name(account.username), hasher.hash(password))
        self.db.execute_write(
            "UPDATE accounts SET last_login_at = ? WHERE id = ?", (self._clock(), account.id)
        )
        return account

    def password_fingerprint(self, username: str) -> str:
        """A value that changes whenever the password does -- binds panel sessions to it."""
        stored = self.secrets.get(self._hash_name(username)) or ""
        return hash_token(stored)[:16]

    # ------------------------------------------------------------------
    # Posture
    # ------------------------------------------------------------------

    def has_admin_credential(self) -> bool:
        """An owner account or an active admin key exists (the bind rule's test)."""
        return self.has_owner() or self.has_active_admin_key()

    def summary(self) -> dict[str, Any]:
        now = self._clock()
        keys = self.list_keys()
        active = [k for k in keys if k.active(now)]
        return {
            "owner_exists": self.has_owner(),
            "accounts": len(self.list_accounts()),
            "keys_active": len(active),
            "keys_by_scope": {s: sum(1 for k in active if k.scope == s) for s in SCOPES},
            "keys_revoked": sum(1 for k in keys if k.revoked),
            "secret_backend": self.secrets.backend,
        }


_DUMMY: list[str] = []


def _dummy_hash(hasher: Any) -> str:
    """A real argon2id hash of a throwaway string, made once, only ever verified against
    to equalise timing for unknown users."""
    if not _DUMMY:
        _DUMMY.append(hasher.hash("studioforge-timing-equaliser"))
    return _DUMMY[0]
