"""Users & Keys: the posture card, the tables, and the four mutations.

Every mutation must (1) pass the local-admin rule first, (2) go through the same route
handler the HTTP API uses, (3) leave an audit line with ``surface='panel'``, and (4) never
put a token, a hash or a password anywhere but the one-time dialog.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import GUARD
from studioforge.auditlog import AuditLog
from studioforge.authkeys import hash_token
from studioforge.config import Config
from studioforge.core.accounts import MIN_PASSWORD_LENGTH, AccountStore
from studioforge.db import Database
from studioforge.errors import BadRequestError, StudioForgeError
from studioforge.gui import tabs
from studioforge.gui.app import create_gui_app
from studioforge.gui.tabs import users_keys
from tests.unit.test_gui import _FakeState

PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _fresh_guard() -> Iterator[None]:
    GUARD.reset()
    yield
    GUARD.reset()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.models.dir = tmp_path / "models"
    cfg.models.dir.mkdir(parents=True, exist_ok=True)
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def store(tmp_path: Path) -> AccountStore:
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    return AccountStore(db, tmp_path)


@pytest.fixture
def state(config: Config, store: AccountStore) -> _FakeState:
    fake = _FakeState(config)
    fake.accounts = store  # type: ignore[attr-defined]
    fake.audit = AuditLog(config.logs_dir)  # type: ignore[attr-defined]
    return fake


@pytest.fixture
def ctx(config: Config, state: _FakeState) -> tabs.GuiContext:
    return tabs.GuiContext(config=config, api_state=state)


def audit_lines(state: _FakeState) -> list[dict[str, Any]]:
    return state.audit.tail(50)  # type: ignore[attr-defined,no-any-return]


# ---------------------------------------------------------------------------
# Tables and posture
# ---------------------------------------------------------------------------


def test_key_rows_carry_no_hash_and_no_token(ctx: tabs.GuiContext, store: AccountStore) -> None:
    record, token = store.create_key("agent", "inference", rate_limit_rpm=30, expires_in_days=7)
    rows = users_keys.key_rows(ctx)
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "agent" and row["scope"] == "inference" and row["rpm"] == 30
    assert row["state"] == "active" and row["last_used"] == "never"
    assert row["prefix"].endswith("…") and len(row["prefix"]) < 12
    flat = str(rows)
    assert token not in flat and record.key_hash not in flat and hash_token(token) not in flat


def test_key_state_distinguishes_revoked_expired_and_active(
    ctx: tabs.GuiContext, store: AccountStore
) -> None:
    gone, _ = store.create_key("gone", "read")
    store.create_key("kept", "read")
    store.revoke_key(gone.id)
    states = {r["name"]: r["state"] for r in users_keys.key_rows(ctx)}
    assert states == {"gone": "revoked", "kept": "active"}
    assert users_keys.key_state({"expired": True}) == "expired"


def test_account_rows(ctx: tabs.GuiContext, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_account("bob", PASSWORD)
    rows = users_keys.account_rows(ctx)
    assert [(r["username"], r["role"]) for r in rows] == [("alice", "owner"), ("bob", "admin")]
    assert "password" not in str(rows).lower()


def test_the_posture_matches_what_the_api_reports(
    config: Config, ctx: tabs.GuiContext, store: AccountStore
) -> None:
    from studioforge.api.security_routes import auth_status
    from studioforge.gui.tabs import api_request

    config.server.api_key = "legacy-key-123456"
    config.mcp.pin = "12345678"
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    posture = users_keys.auth_posture(ctx)
    api = asyncio.run(auth_status(api_request(ctx)))
    for key in (
        "auth_required",
        "legacy_api_key_set",
        "mcp_pin_set",
        "owner_exists",
        "accounts",
        "keys_active",
        "keys_by_scope",
        "keys_revoked",
        "secret_backend",
    ):
        assert posture[key] == api[key], key


def test_the_posture_says_deprecated_for_the_legacy_key_and_the_pin() -> None:
    lines = users_keys.posture_lines(
        {
            "auth_required": True,
            "owner_exists": True,
            "accounts": 1,
            "keys_active": 2,
            "keys_by_scope": {"read": 1, "inference": 0, "admin": 1},
            "keys_revoked": 0,
            "secret_backend": "file",
            "legacy_api_key_set": True,
            "mcp_pin_set": True,
            "remote": [{"service": "the gateway", "host": "0.0.0.0"}],
        }
    )
    text = " ".join(line[2] for line in lines)
    assert "Sign-in is required" in text
    assert text.count("deprecated") == 2
    assert "1 read, 1 admin" in text and "file" in text and "0.0.0.0" in text


def test_the_posture_of_an_open_install_warns() -> None:
    lines = users_keys.posture_lines({"auth_required": False, "owner_exists": False})
    assert "No sign-in is required yet" in lines[0][2]
    assert lines[1][1] == "warning"


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


async def test_creating_a_key_returns_the_token_once_and_audits_without_it(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore
) -> None:
    record, token = await users_keys.create_key(
        ctx, "  dispatch ", "inference", expires_in_days=30, rate_limit_rpm=60
    )
    assert record["name"] == "dispatch" and record["scope"] == "inference"
    assert token.startswith("sf-i-")
    assert store.authenticate(token, count=False) is not None  # the token really works
    line = next(e for e in audit_lines(state) if e["action"] == "panel.key.create")
    assert line["surface"] == "panel" and "dispatch (inference)" in line["target"]
    assert token not in str(audit_lines(state))
    assert "token" not in record and "key_hash" not in record


@pytest.mark.parametrize(
    ("name", "scope"),
    [("", "read"), ("   ", "read"), ("x" * 81, "read"), ("ok", "root")],
)
async def test_a_bad_key_request_says_what_is_wrong(
    ctx: tabs.GuiContext, name: str, scope: str
) -> None:
    with pytest.raises(StudioForgeError) as info:
        await users_keys.create_key(ctx, name, scope)
    assert info.value.status_code == 400
    assert "not allowed" in info.value.message


async def test_revoking_a_key_audits_and_an_unknown_key_is_explained(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore
) -> None:
    record, token = store.create_key("old", "read")
    revoked = await users_keys.revoke_key(ctx, record.id)
    assert revoked["revoked"] is True
    with pytest.raises(Exception, match="revoked"):
        store.authenticate(token, count=False)
    assert any(e["action"] == "panel.key.revoke" for e in audit_lines(state))
    with pytest.raises(BadRequestError, match="There is no key"):
        await users_keys.revoke_key(ctx, "nope")


async def test_creating_accounts_checks_the_confirmation_and_the_length(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore
) -> None:
    with pytest.raises(StudioForgeError, match="do not match"):
        await users_keys.create_account(ctx, "alice", PASSWORD, "different")
    with pytest.raises(BadRequestError, match=f"at least {MIN_PASSWORD_LENGTH} characters"):
        await users_keys.create_account(ctx, "alice", "short", "short")
    assert not store.has_owner()
    first = await users_keys.create_account(ctx, "alice", PASSWORD, PASSWORD)
    second = await users_keys.create_account(ctx, "bob", PASSWORD, PASSWORD)
    assert (first["role"], second["role"]) == ("owner", "admin")
    assert store.verify_password("alice", PASSWORD) is not None
    lines = [e for e in audit_lines(state) if e["action"] == "panel.account.create"]
    assert [e["target"] for e in lines] == ["alice (owner)", "bob (admin)"]
    assert PASSWORD not in str(audit_lines(state))


async def test_changing_a_password_replaces_it_and_audits_the_name_only(
    ctx: tabs.GuiContext, state: _FakeState, store: AccountStore
) -> None:
    store.create_account("alice", PASSWORD)
    before = store.password_fingerprint("alice")
    await users_keys.change_password(
        ctx, "alice", "a brand new passphrase", "a brand new passphrase"
    )
    assert store.verify_password("alice", PASSWORD) is None
    assert store.verify_password("alice", "a brand new passphrase") is not None
    assert store.password_fingerprint("alice") != before
    line = next(e for e in audit_lines(state) if e["action"] == "panel.account.password")
    assert line["target"] == "alice"
    assert "passphrase" not in str(audit_lines(state))
    with pytest.raises(StudioForgeError, match="do not match"):
        await users_keys.change_password(ctx, "alice", "one one one one", "two two two two")
    with pytest.raises(BadRequestError, match="no account named"):
        await users_keys.change_password(ctx, "ghost", PASSWORD, PASSWORD)


async def test_every_mutation_is_refused_for_a_remote_viewer_on_an_open_install(
    ctx: tabs.GuiContext, store: AccountStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tabs, "viewer_host", lambda: "192.168.1.50")
    with pytest.raises(StudioForgeError) as info:
        await users_keys.create_key(ctx, "x", "read")
    assert info.value.code == "remote_admin_requires_credential"
    with pytest.raises(StudioForgeError):
        await users_keys.revoke_key(ctx, "x")
    with pytest.raises(StudioForgeError):
        await users_keys.create_account(ctx, "alice", PASSWORD, PASSWORD)
    with pytest.raises(StudioForgeError):
        await users_keys.change_password(ctx, "alice", PASSWORD, PASSWORD)
    assert store.list_keys() == [] and store.list_accounts() == []


def test_the_audit_rows_are_newest_first(ctx: tabs.GuiContext, state: _FakeState) -> None:
    audit: AuditLog = state.audit  # type: ignore[attr-defined]
    for i in range(3):
        audit.record(actor="panel", action=f"a{i}", surface="panel")
    assert [r["action"] for r in users_keys.audit_rows(ctx)] == ["a2", "a1", "a0"]
    assert len(users_keys.audit_rows(ctx, n=2)) == 2


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def test_the_page_shows_names_but_never_a_token_or_a_hash(
    config: Config, state: _FakeState, store: AccountStore
) -> None:
    store.create_account("alice", PASSWORD)
    record, token = store.create_key("ops-robot", "admin")
    with TestClient(create_gui_app(config, api_state=state)) as client:
        text = client.get("/", headers={"Authorization": f"Bearer {token}"}).text
    assert "could not be rendered" not in text
    for needle in ("alice", "ops-robot", "Create an account", "Create a key", "Audit log"):
        assert needle in text, needle
    assert token not in text
    assert record.key_hash not in text and hash_token(token) not in text
    assert PASSWORD not in text


def test_the_page_without_accounts_says_so(config: Config) -> None:
    bare = _FakeState(config)
    with TestClient(create_gui_app(config, api_state=bare)) as client:
        text = client.get("/").text
    assert "Accounts are not available on this server build" in text
    assert "could not be rendered" not in text
