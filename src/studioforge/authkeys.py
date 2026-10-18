"""API-key primitives shared by the gateway, the control panel and the watchdog.

A stdlib-only leaf, on purpose: the watchdog is a separate process that must not
import the app's machinery (``tests/unit/test_watchdog.py`` enforces it), yet it
has to recognise the same scoped keys as the gateway. So everything both sides
need lives here -- the key format, the hash, the scope ordering, the loopback
test, and the *snapshot* file the gateway keeps beside ``config.yaml`` so the
watchdog can verify a key without ever opening ``registry.sqlite3``.

Key format
----------
``sf-<scope letter>-<base62>``: ``r`` read, ``i`` inference, ``a`` admin, then
43 base62 characters (256 bits). The scope letter is a *hint for humans and
secret scanners*, never trusted: the scope that counts is the one stored beside
the hash. Only the SHA-256 of the whole token is stored, so a stolen database or
snapshot yields nothing usable; the token itself is shown once, at creation.
SHA-256 (not argon2) is right for a 256-bit random secret: there is nothing to
brute-force, and the check runs on every request.

Scopes are ordered: ``read < inference < admin``. "inference+" means inference
or admin.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from studioforge.secure_files import write_private

__all__ = [
    "SCOPES",
    "SCOPE_LETTER",
    "KeyRecord",
    "Principal",
    "SnapshotVerifier",
    "generate_token",
    "hash_token",
    "is_loopback_host",
    "looks_like_key",
    "origin_allowed",
    "scope_at_least",
    "snapshot_path",
    "write_snapshot",
]

Scope = Literal["read", "inference", "admin"]

#: Weakest to strongest.
SCOPES: tuple[str, ...] = ("read", "inference", "admin")
SCOPE_LETTER: dict[str, str] = {"read": "r", "inference": "i", "admin": "a"}
_LETTER_SCOPE = {v: k for k, v in SCOPE_LETTER.items()}

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_TOKEN_LENGTH = 43  # 62**43 > 2**256
KEY_PREFIX = "sf-"

#: File beside ``config.yaml`` that the watchdog reads (see module docstring).
SNAPSHOT_NAME = "keys.snapshot.json"


def scope_at_least(have: str | None, need: str) -> bool:
    """Whether scope ``have`` satisfies the requirement ``need`` (admin >= inference >= read)."""
    if have not in SCOPES or need not in SCOPES:
        return False
    return SCOPES.index(have) >= SCOPES.index(need)


def generate_token(scope: str) -> str:
    """A fresh ``sf-<letter>-<base62>`` token for ``scope``. Shown once, never stored."""
    if scope not in SCOPE_LETTER:
        raise ValueError(f"unknown scope {scope!r}; expected one of {', '.join(SCOPES)}")
    number = int.from_bytes(secrets.token_bytes(32), "big")
    digits: list[str] = []
    while number:
        number, rem = divmod(number, 62)
        digits.append(_BASE62[rem])
    body = "".join(reversed(digits)).rjust(_TOKEN_LENGTH, "0")
    return f"{KEY_PREFIX}{SCOPE_LETTER[scope]}-{body}"


def hash_token(token: str) -> str:
    """SHA-256 hex of the whole token."""
    return hashlib.sha256(token.encode("utf-8", "surrogateescape")).hexdigest()


def looks_like_key(value: str | None) -> bool:
    """Cheap shape test (``sf-r|i|a-...``) used to skip a lookup for obviously other tokens."""
    return (
        bool(value) and str(value).startswith(KEY_PREFIX) and str(value)[3:5] in {"r-", "i-", "a-"}
    )


def scope_hint(token: str) -> str | None:
    """The scope a token's letter claims -- a hint only, the stored scope decides."""
    return _LETTER_SCOPE.get(token[3:4]) if looks_like_key(token) else None


def is_loopback_host(host: str | None) -> bool:
    """Whether a *bind address* only accepts connections from this machine.

    ``0.0.0.0``, ``::`` and the empty string are wildcards (every interface) and
    are not loopback; a hostname other than ``localhost`` is treated as remote
    because it may resolve to anything.
    """
    text = (host or "").strip().strip("[]").lower()
    if text in {"localhost", "ip6-localhost", "ip6-loopback"}:
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def origin_allowed(origin: str | None, allowed: list[str] | tuple[str, ...] = ()) -> bool:
    """Whether a browser ``Origin`` may reach the management (MCP) endpoints.

    No ``Origin`` header means a non-browser client (agents, curl, SDKs): allowed.
    A present one must be a loopback page (``http(s)://127.0.0.1|localhost|[::1]``,
    any port) or exactly one of ``allowed`` (``server.cors_origins``). A wildcard
    ``*`` in ``allowed`` is deliberately **not** honoured here: it means "any page
    may *read* the API", and must not also mean "any page may drive the management
    tools". ``Origin: null`` (sandboxed frames) is refused.

    The point is DNS rebinding and drive-by cross-site pages: both arrive with an
    ``Origin`` naming the attacker's site, which a same-``Host`` comparison would
    wave through (the rebound name *is* the Host), so only an allowlist works.
    """
    if origin is None or not origin.strip():
        return True
    from urllib.parse import urlsplit

    text = origin.strip()
    if text.lower() == "null":
        return False
    wanted = text.rstrip("/").lower()
    if any(wanted == a.strip().rstrip("/").lower() for a in allowed if a.strip() != "*"):
        return True
    try:
        parts = urlsplit(text)
        host = parts.hostname or ""
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and is_loopback_host(host)


@dataclass(frozen=True)
class KeyRecord:
    """One API key as stored: everything except the secret."""

    id: str
    name: str
    scope: str
    key_hash: str
    prefix: str
    created_at: float
    expires_at: float | None = None
    last_used_at: float | None = None
    rate_limit_rpm: int | None = None
    revoked_at: float | None = None

    @property
    def revoked(self) -> bool:
        return self.revoked_at is not None

    def expired(self, now: float | None = None) -> bool:
        return self.expires_at is not None and (now if now is not None else time.time()) >= (
            self.expires_at
        )

    def active(self, now: float | None = None) -> bool:
        return not self.revoked and not self.expired(now)

    def public(self) -> dict[str, Any]:
        """The row as an API/CLI/panel shows it: never the hash."""
        return {
            "id": self.id,
            "name": self.name,
            "scope": self.scope,
            "prefix": self.prefix,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "last_used_at": self.last_used_at,
            "rate_limit_rpm": self.rate_limit_rpm,
            "revoked": self.revoked,
            "expired": self.expired(),
        }


@dataclass(frozen=True)
class Principal:
    """Who a request is, once authenticated.

    ``kind``: ``key`` (a scoped key), ``legacy`` (``server.api_key``), ``owner``
    (a signed-in account), ``pin`` (the deprecated MCP pairing PIN), ``local``
    (an unauthenticated caller on this machine, or an in-process/stdio call) or
    ``anonymous`` (open install, remote caller).
    """

    kind: str
    id: str
    name: str
    scope: str

    @property
    def actor(self) -> str:
        """The string audit lines carry."""
        return f"{self.kind}:{self.id}" if self.id else self.kind

    def allows(self, need: str) -> bool:
        return scope_at_least(self.scope, need)


#: Trusted in-process callers: the stdio MCP server, tests, the panel calling a
#: handler directly. They never crossed a network.
LOCAL_ADMIN = Principal("local", "", "local process", "admin")


# ---------------------------------------------------------------------------
# The snapshot the watchdog reads
# ---------------------------------------------------------------------------


def snapshot_path(data_dir: Path) -> Path:
    return Path(data_dir) / SNAPSHOT_NAME


def write_snapshot(
    data_dir: Path, records: list[KeyRecord], *, revoked_legacy: list[str] | None = None
) -> None:
    """Write the active keys (hash, scope, expiry -- no secrets) for the watchdog.

    Only keys that could still authenticate are written; a revoked key simply
    disappears, so the watchdog needs no revocation logic of its own -- except
    for the legacy ``server.api_key``, which the watchdog compares against
    ``config.yaml`` rather than this file. ``revoked_legacy`` lists the hashes of
    revoked ``legacy`` mirrors so ``keys revoke legacy`` disables the config key
    on the watchdog too.
    """
    now = time.time()
    payload = {
        "version": 1,
        "written_at": now,
        "revoked_legacy": sorted(set(revoked_legacy or [])),
        "keys": [
            {
                "id": r.id,
                "name": r.name,
                "scope": r.scope,
                "hash": r.key_hash,
                "expires_at": r.expires_at,
            }
            for r in records
            if r.active(now)
        ],
    }
    target = snapshot_path(data_dir)
    # Per-process temp name: the CLI and the server may write the snapshot at once.
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    write_private(tmp, json.dumps(payload, indent=1))
    for attempt in range(6):
        try:
            tmp.replace(target)
            return
        except PermissionError:  # Windows: the watchdog has it open for a read
            time.sleep(0.05 * (attempt + 1))
    with contextlib.suppress(OSError):
        tmp.unlink()


class SnapshotVerifier:
    """Verify a presented token against the snapshot file, re-reading on change.

    Used by the watchdog. Cheap: ``stat`` per call, parse only when the mtime
    moved. Never writes (no ``last_used``) and never raises: an unreadable or
    absent snapshot simply verifies nothing, which is the safe direction.
    """

    def __init__(self, data_dir: Path | Any) -> None:
        self._path = snapshot_path(Path(data_dir))
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._by_hash: dict[str, dict[str, Any]] = {}
        self._revoked_legacy: frozenset[str] = frozenset()

    def _refresh(self) -> None:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            self._mtime = None
            self._by_hash = {}
            self._revoked_legacy = frozenset()
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            rows = data.get("keys", [])
            self._by_hash = {str(r["hash"]): r for r in rows if isinstance(r, dict) and "hash" in r}
            revoked = data.get("revoked_legacy") or []
            self._revoked_legacy = frozenset(str(h) for h in revoked if isinstance(h, str))
            self._mtime = mtime
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            self._by_hash = {}
            self._revoked_legacy = frozenset()
            self._mtime = None

    def legacy_revoked(self, api_key: str | None) -> bool:
        """Whether the operator revoked the ``legacy`` mirror of this ``server.api_key``."""
        if not api_key:
            return False
        digest = hash_token(api_key)
        with self._lock:
            self._refresh()
            return digest in self._revoked_legacy

    def has_keys(self) -> bool:
        with self._lock:
            self._refresh()
            return bool(self._by_hash)

    def has_admin(self) -> bool:
        with self._lock:
            self._refresh()
            return any(r.get("scope") == "admin" for r in self._by_hash.values())

    def verify(self, token: str | None) -> Principal | None:
        if not token or not looks_like_key(token):
            return None
        digest = hash_token(token)
        with self._lock:
            self._refresh()
            row = self._by_hash.get(digest)
        if row is None:
            return None
        expires = row.get("expires_at")
        if expires is not None and time.time() >= float(expires):
            return None
        # Re-compare in constant time: the dict lookup above already narrowed it
        # to one stored hash, this just keeps the habit.
        if not hmac.compare_digest(digest, str(row["hash"])):
            return None
        scope = str(row.get("scope", "read"))
        if scope not in SCOPES:
            return None
        return Principal("key", str(row.get("id", "")), str(row.get("name", "")), scope)
