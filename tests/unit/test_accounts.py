"""Accounts, scoped API keys and the key snapshot the watchdog reads.

The claims worth testing: a token is shown once and only its SHA-256 is stored;
scope ordering; expiry, revocation and the per-key rate limit each have their own
catalogued error; ``last_used_at`` is recorded without a write per request; the
legacy ``server.api_key`` is mirrored (and stays revoked when revoked); the owner
password is argon2id and never lands in the database; and the snapshot the
watchdog verifies against carries hashes, never tokens.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from studioforge.authkeys import (
    SnapshotVerifier,
    generate_token,
    hash_token,
    is_loopback_host,
    looks_like_key,
    origin_allowed,
    scope_at_least,
    snapshot_path,
)
from studioforge.core.accounts import LEGACY_KEY_NAME, MIN_PASSWORD_LENGTH, AccountStore
from studioforge.db import Database
from studioforge.errors import AuthError, BadRequestError


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> AccountStore:
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    return AccountStore(db, tmp_path, clock=clock)


# ---------------------------------------------------------------------------
# token format and hashing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("scope", "letter"), [("read", "r"), ("inference", "i"), ("admin", "a")])
def test_token_format_is_sf_letter_base62(scope: str, letter: str) -> None:
    token = generate_token(scope)
    assert token.startswith(f"sf-{letter}-")
    body = token.split("-", 2)[2]
    assert len(body) == 43 and body.isalnum()
    assert looks_like_key(token)
    assert generate_token(scope) != token


def test_unknown_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown scope"):
        generate_token("root")


def test_scope_ordering() -> None:
    assert scope_at_least("admin", "inference") and scope_at_least("inference", "read")
    assert not scope_at_least("read", "inference") and not scope_at_least("inference", "admin")
    assert not scope_at_least("bogus", "read") and not scope_at_least(None, "read")


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("[::1]", True),
        ("localhost", True),
        ("127.0.0.2", True),
        ("0.0.0.0", False),
        ("::", False),
        ("", False),
        ("192.168.1.5", False),
        ("my-box.example", False),
    ],
)
def test_is_loopback_host(host: str, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------


def test_the_token_is_shown_once_and_only_its_hash_is_stored(
    store: AccountStore, tmp_path: Path
) -> None:
    record, token = store.create_key("agent", "inference")
    assert record.key_hash == hash_token(token) and token not in json.dumps(record.public())
    raw = sqlite3.connect(tmp_path / "registry.sqlite3")
    try:
        rows = raw.execute("SELECT * FROM api_keys").fetchall()
    finally:
        raw.close()
    assert token not in repr(rows) and record.key_hash in repr(rows)
    assert record.prefix == token[:9]
    assert "key_hash" not in record.public() and "hash" not in json.dumps(record.public())


def test_authenticate_returns_the_principal_and_scope(store: AccountStore) -> None:
    _record, token = store.create_key("ro", "read")
    principal = store.authenticate(token)
    assert principal is not None
    assert (principal.kind, principal.name, principal.scope) == ("key", "ro", "read")
    assert principal.allows("read") and not principal.allows("inference")
    assert store.authenticate("sf-r-" + "x" * 43) is None
    assert store.authenticate("not-a-key") is None and store.authenticate(None) is None


def test_name_and_scope_are_validated(store: AccountStore) -> None:
    with pytest.raises(BadRequestError):
        store.create_key("  ", "read")
    with pytest.raises(BadRequestError, match="Unknown scope"):
        store.create_key("x", "root")
    with pytest.raises(BadRequestError, match="rate_limit_rpm"):
        store.create_key("x", "read", rate_limit_rpm=0)


def test_an_expired_key_is_refused_with_its_own_code(store: AccountStore, clock: Clock) -> None:
    _record, token = store.create_key("short", "read", expires_in_days=1)
    assert store.authenticate(token) is not None
    clock.now += 2 * 86400
    with pytest.raises(AuthError) as excinfo:
        store.authenticate(token)
    assert excinfo.value.code == "api_key_expired" and excinfo.value.status_code == 401


def test_a_revoked_key_is_refused_with_its_own_code(store: AccountStore) -> None:
    record, token = store.create_key("gone", "admin")
    assert store.revoke_key(record.id) is not None
    with pytest.raises(AuthError) as excinfo:
        store.authenticate(token)
    assert excinfo.value.code == "api_key_revoked"
    assert store.revoke_key(record.id) is not None  # idempotent
    assert store.revoke_key("no-such-key") is None


def test_revoke_by_name_only_when_unambiguous(store: AccountStore) -> None:
    store.create_key("dup", "read")
    store.create_key("dup", "read")
    assert store.find_key("dup") is None
    store.create_key("solo", "read")
    assert store.find_key("solo") is not None


def test_the_per_key_rate_limit_returns_429_with_retry_after(
    store: AccountStore, clock: Clock
) -> None:
    _record, token = store.create_key("slow", "inference", rate_limit_rpm=3)
    for _ in range(3):
        assert store.authenticate(token) is not None
    with pytest.raises(AuthError) as excinfo:
        store.authenticate(token)
    assert excinfo.value.code == "key_rate_limited" and excinfo.value.status_code == 429
    assert excinfo.value.details["limit_rpm"] == 3 and excinfo.value.details["retry_after_s"] >= 1
    clock.now += 61
    assert store.authenticate(token) is not None  # the window slid


def test_a_rate_limited_key_does_not_limit_another(store: AccountStore) -> None:
    _a, slow = store.create_key("slow", "read", rate_limit_rpm=1)
    _b, free = store.create_key("free", "read")
    assert store.authenticate(slow) is not None
    with pytest.raises(AuthError):
        store.authenticate(slow)
    for _ in range(20):
        assert store.authenticate(free) is not None


def test_count_false_does_not_spend_the_budget(store: AccountStore) -> None:
    _record, token = store.create_key("peek", "read", rate_limit_rpm=1)
    for _ in range(5):
        assert store.authenticate(token, count=False) is not None
    assert store.authenticate(token) is not None


def test_last_used_is_recorded_but_not_on_every_request(store: AccountStore, clock: Clock) -> None:
    record, token = store.create_key("seen", "read")
    assert store.find_key(record.id).last_used_at is None  # type: ignore[union-attr]
    store.authenticate(token)
    first = store.find_key(record.id).last_used_at  # type: ignore[union-attr]
    assert first == clock.now
    clock.now += 5
    store.authenticate(token)
    assert store.find_key(record.id).last_used_at == first  # type: ignore[union-attr]
    clock.now += 60
    store.authenticate(token)
    assert store.find_key(record.id).last_used_at == clock.now  # type: ignore[union-attr]


def test_posture_helpers(store: AccountStore) -> None:
    assert not store.has_active_keys() and not store.has_admin_credential()
    store.create_key("r", "read")
    assert store.has_active_keys() and not store.has_active_admin_key()
    assert store.has_active_keys_cached() is True
    store.create_account("owner", "a-long-enough-password")
    assert store.has_admin_credential() and store.has_owner_cached()


# ---------------------------------------------------------------------------
# the legacy server.api_key
# ---------------------------------------------------------------------------


def test_the_legacy_key_is_mirrored_as_an_admin_key_named_legacy(store: AccountStore) -> None:
    record = store.adopt_legacy_key("legacy-secret-value")
    assert record is not None and record.name == LEGACY_KEY_NAME and record.scope == "admin"
    assert record.key_hash == hash_token("legacy-secret-value")
    assert store.adopt_legacy_key("legacy-secret-value").id == record.id  # type: ignore[union-attr]
    assert len(store.list_keys()) == 1
    principal = store.authenticate_legacy("legacy-secret-value")
    assert principal is not None and principal.kind == "legacy" and principal.scope == "admin"


def test_revoking_the_legacy_mirror_stays_revoked_across_restarts(store: AccountStore) -> None:
    record = store.adopt_legacy_key("legacy-secret-value")
    assert record is not None
    store.revoke_key(LEGACY_KEY_NAME)
    again = store.adopt_legacy_key("legacy-secret-value")
    assert again is not None and again.revoked  # not resurrected
    with pytest.raises(AuthError) as excinfo:
        store.authenticate_legacy("legacy-secret-value")
    assert excinfo.value.code == "api_key_revoked"


def test_changing_the_legacy_key_replaces_the_mirror(store: AccountStore) -> None:
    store.adopt_legacy_key("first-legacy-key")
    store.adopt_legacy_key("second-legacy-key")
    live = [k for k in store.list_keys(include_revoked=False) if k.name == LEGACY_KEY_NAME]
    assert len(live) == 1 and live[0].key_hash == hash_token("second-legacy-key")
    store.adopt_legacy_key(None)
    assert not [k for k in store.list_keys(include_revoked=False) if k.name == LEGACY_KEY_NAME]


# ---------------------------------------------------------------------------
# accounts
# ---------------------------------------------------------------------------


def test_owner_password_is_argon2id_and_not_in_the_database(
    store: AccountStore, tmp_path: Path
) -> None:
    account = store.create_account("owner", "correct horse battery")
    assert account.role == "owner" and store.has_owner()
    stored = store.secrets.get("owner.owner.hash")
    assert stored is not None and stored.startswith(
        "$argon2id$"
    )  # scrub-ok: the hash prefix, not a hash
    raw = (tmp_path / "registry.sqlite3").read_bytes()
    assert b"argon2" not in raw and b"correct horse" not in raw
    assert store.verify_password("OWNER", "correct horse battery") is not None  # case-insensitive
    assert store.verify_password("owner", "wrong password!!") is None
    assert store.verify_password("nobody", "correct horse battery") is None


def test_the_second_account_is_not_the_owner(store: AccountStore) -> None:
    store.create_account("owner", "correct horse battery")
    assert store.create_account("second", "another long password").role == "admin"


def test_account_validation(store: AccountStore) -> None:
    with pytest.raises(BadRequestError, match=str(MIN_PASSWORD_LENGTH)):
        store.create_account("owner", "short")
    with pytest.raises(BadRequestError, match="username"):
        store.create_account("has space", "a-long-enough-password")
    store.create_account("owner", "a-long-enough-password")
    with pytest.raises(BadRequestError, match="already exists"):
        store.create_account("Owner", "a-long-enough-password")


def test_changing_the_password_changes_the_session_fingerprint(store: AccountStore) -> None:
    store.create_account("owner", "first password long")
    before = store.password_fingerprint("owner")
    store.set_password("owner", "second password long")
    assert store.password_fingerprint("owner") != before
    assert store.verify_password("owner", "first password long") is None
    assert store.verify_password("owner", "second password long") is not None
    with pytest.raises(BadRequestError, match="no account"):
        store.set_password("ghost", "second password long")


def test_local_cli_key_is_created_once_and_replaced_when_revoked(
    store: AccountStore, tmp_path: Path
) -> None:
    token = store.local_cli_key()
    assert (tmp_path / "cli.key").read_text(encoding="utf-8").strip() == token
    assert store.local_cli_key() == token
    principal = store.authenticate(token)
    assert principal is not None and principal.scope == "admin"
    store.revoke_key("local-cli")
    assert store.local_cli_key() != token


# ---------------------------------------------------------------------------
# the snapshot the watchdog reads
# ---------------------------------------------------------------------------


def test_the_snapshot_has_hashes_not_tokens_and_drops_revoked_keys(
    store: AccountStore, tmp_path: Path
) -> None:
    keep, keep_token = store.create_key("keep", "admin")
    drop, drop_token = store.create_key("drop", "read")
    store.revoke_key(drop.id)
    text = snapshot_path(tmp_path).read_text(encoding="utf-8")
    assert keep_token not in text and drop_token not in text
    assert keep.key_hash in text and drop.key_hash not in text


def test_snapshot_verifier_follows_the_file(store: AccountStore, tmp_path: Path) -> None:
    verifier = SnapshotVerifier(tmp_path)
    _record, token = store.create_key("wd", "admin")
    found = verifier.verify(token)
    assert found is not None and found.scope == "admin" and verifier.has_admin()
    assert verifier.verify("sf-a-" + "z" * 43) is None and verifier.verify(None) is None
    store.revoke_key("wd")
    assert verifier.verify(token) is None and not verifier.has_keys()


def test_snapshot_verifier_honours_expiry_and_survives_a_bad_file(
    store: AccountStore, tmp_path: Path, clock: Clock
) -> None:
    verifier = SnapshotVerifier(tmp_path)
    _record, token = store.create_key("tmp", "read", expires_at=1.0)  # long expired
    assert verifier.verify(token) is None
    snapshot_path(tmp_path).write_text("{ not json", encoding="utf-8")
    assert verifier.verify(token) is None and not verifier.has_keys()
    assert SnapshotVerifier(tmp_path / "missing").verify(token) is None


# ---------------------------------------------------------------------------
# the Origin allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("origin", "allowed", "ok"),
    [
        (None, [], True),
        ("", [], True),
        ("http://127.0.0.1:8080", [], True),
        ("http://localhost:3000", [], True),
        ("https://[::1]:8443", [], True),
        ("http://evil.example", [], False),
        ("http://evil.example", ["*"], False),  # a wildcard never opens MCP
        ("https://app.example", ["https://app.example"], True),
        ("https://app.example/", ["https://app.example"], True),
        ("null", [], False),
        ("http://127.0.0.1.evil.example", [], False),
        ("ftp://127.0.0.1", [], False),
    ],
)
def test_origin_allowlist(origin: str | None, allowed: list[str], ok: bool) -> None:
    assert origin_allowed(origin, allowed) is ok
