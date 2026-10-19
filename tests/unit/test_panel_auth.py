"""The control panel's sign-in: owner accounts, scoped keys, sessions, lockout, CSRF.

The gate used to open only for ``server.api_key``. It now opens whenever the server
requires a credential at all (``auth_enforced``), takes an owner's username and
password or an *admin* key, and keeps the legacy key and bearer header working. These
tests drive the real ASGI app through ``TestClient`` with a real ``AccountStore``.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import GUARD
from studioforge.auditlog import AuditLog
from studioforge.config import Config
from studioforge.core.accounts import AccountStore
from studioforge.db import Database
from studioforge.gui.app import (
    COOKIE_NAME,
    LOGIN_PATH,
    GuiAuthGate,
    create_gui_app,
    make_session_cookie,
    session_token,
)
from tests.unit.test_gui import _FakeState

PASSWORD = "correct horse battery"
HTML = {"accept": "text/html"}


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
def client(config: Config, state: _FakeState) -> Iterator[TestClient]:
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, **form: str) -> Any:
    return client.post(LOGIN_PATH, data=form, follow_redirects=False)


def _cookie(response: Any) -> str:
    return str(response.cookies[COOKIE_NAME])


# ---------------------------------------------------------------------------
# When the panel is gated
# ---------------------------------------------------------------------------


def test_an_open_install_has_no_gate(client: TestClient) -> None:
    assert client.get("/", follow_redirects=False).status_code == 200


def test_the_gate_closes_as_soon_as_any_key_exists(client: TestClient, store: AccountStore) -> None:
    store.create_key("agent", "read")
    assert client.get("/", headers=HTML, follow_redirects=False).status_code == 303
    refused = client.get("/", follow_redirects=False)
    assert refused.status_code == 401
    assert "sign-in" in refused.json()["error"]["message"]


def test_an_owner_on_a_network_bind_closes_the_gate(
    client: TestClient, store: AccountStore, config: Config
) -> None:
    store.create_account("alice", PASSWORD)
    assert client.get("/", follow_redirects=False).status_code == 200  # loopback: still open
    config.gui.host = "0.0.0.0"  # the config is a live object; the gate re-reads it
    assert client.get("/", headers=HTML, follow_redirects=False).status_code == 303


def test_the_login_page_offers_a_username_and_a_key(client: TestClient) -> None:
    text = client.get(LOGIN_PATH).text
    assert "Sign in to StudioForge" in text
    assert 'name="username"' in text and 'name="credential"' in text
    assert "API key" in text


def test_the_legacy_key_cookie_and_bearer_still_work(client: TestClient, config: Config) -> None:
    config.server.api_key = "legacy-key-123456"
    assert client.get("/", headers=HTML, follow_redirects=False).status_code == 303
    bearer = {"Authorization": "Bearer legacy-key-123456"}
    assert client.get("/", headers=bearer).status_code == 200
    response = _login(client, api_key="legacy-key-123456")
    assert response.status_code == 303
    assert _cookie(response) == session_token("legacy-key-123456")
    assert (
        client.get("/", cookies={COOKIE_NAME: session_token("legacy-key-123456")}).status_code
        == 200
    )
    # ...and the same key works through the new field name too.
    again = _login(client, credential="legacy-key-123456")
    assert _cookie(again) == session_token("legacy-key-123456")


# ---------------------------------------------------------------------------
# Owner sign-in
# ---------------------------------------------------------------------------


def test_owner_signs_in_with_a_password_and_the_session_names_them(
    client: TestClient, store: AccountStore, state: _FakeState
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")  # makes the sign-in required
    response = _login(client, username="alice", credential=PASSWORD)
    assert response.status_code == 303
    cookie = _cookie(response)
    kind, ident, expiry, signature = cookie.split(".")
    assert (kind, ident) == ("owner", "alice")
    assert int(expiry) > time.time() + 29 * 86400 and len(signature) == 64
    page = client.get("/?section=Advanced&page=accounts", cookies={COOKIE_NAME: cookie})
    assert page.status_code == 200
    # viewer_actor() read the actor the gate put on the ASGI scope.
    assert "signed in as: owner:alice" in page.text
    audit = state.audit.tail(10)  # type: ignore[attr-defined]
    assert any(e["action"] == "panel.login" and e["result"] == "ok" for e in audit)


def test_the_session_cookie_flags(client: TestClient, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    header = _login(client, username="alice", credential=PASSWORD).headers["set-cookie"]
    lowered = header.lower()
    assert "httponly" in lowered and "samesite=lax" in lowered and "secure" not in lowered
    assert "max-age=2592000" in lowered


def test_the_cookie_is_secure_when_the_panel_is_https(
    client: TestClient, store: AccountStore, config: Config
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    config.server.tls.mode = "self_signed"
    header = _login(client, username="alice", credential=PASSWORD).headers["set-cookie"]
    assert "secure" in header.lower()


def test_changing_the_password_ends_the_session(client: TestClient, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    cookie = _cookie(_login(client, username="alice", credential=PASSWORD))
    assert client.get("/", cookies={COOKIE_NAME: cookie}).status_code == 200
    store.set_password("alice", "a completely new password")
    assert (
        client.get("/", headers=HTML, cookies={COOKIE_NAME: cookie}, follow_redirects=False)
    ).status_code == 303


def test_a_wrong_password_is_refused_without_leaking_it(
    client: TestClient, store: AccountStore
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    response = _login(client, username="alice", credential="not-the-password")
    assert response.status_code == 401
    assert "not-the-password" not in response.text
    assert COOKIE_NAME not in response.cookies
    unknown = _login(client, username="nobody", credential=PASSWORD)
    assert unknown.status_code == 401  # an unknown user looks the same as a wrong password


def test_a_password_with_no_username_is_not_taken_as_a_key(
    client: TestClient, store: AccountStore
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    assert _login(client, credential=PASSWORD).status_code == 401


# ---------------------------------------------------------------------------
# Key sign-in: admin only
# ---------------------------------------------------------------------------


def test_an_admin_key_opens_the_panel(client: TestClient, store: AccountStore) -> None:
    record, token = store.create_key("ops", "admin")
    response = _login(client, credential=token)
    assert response.status_code == 303
    cookie = _cookie(response)
    assert cookie.startswith(f"key.{record.id}.")
    assert client.get("/", cookies={COOKIE_NAME: cookie}).status_code == 200
    assert client.get("/", headers={"Authorization": f"Bearer {token}"}).status_code == 200


@pytest.mark.parametrize("scope", ["read", "inference"])
def test_a_lesser_key_cannot_open_the_panel(
    client: TestClient, store: AccountStore, scope: str
) -> None:
    _, token = store.create_key("app", scope)
    response = _login(client, credential=token)
    assert response.status_code == 403
    assert "needs the admin scope" in response.text
    assert token not in response.text
    assert COOKIE_NAME not in response.cookies
    bearer = {"Authorization": f"Bearer {token}"}
    assert client.get("/", headers={**HTML, **bearer}, follow_redirects=False).status_code == 303


def test_revoking_the_key_ends_its_sessions(client: TestClient, store: AccountStore) -> None:
    record, token = store.create_key("ops", "admin")
    cookie = _cookie(_login(client, credential=token))
    assert client.get("/", cookies={COOKIE_NAME: cookie}).status_code == 200
    store.revoke_key(record.id)
    # No admin key is left but the revoked one: nothing requires a sign-in any more, so
    # prove the cookie itself is dead by adding another key to keep the gate closed.
    store.create_key("other", "read")
    assert (
        client.get("/", headers=HTML, cookies={COOKIE_NAME: cookie}, follow_redirects=False)
    ).status_code == 303


def test_a_revoked_key_cannot_sign_in_and_says_why(client: TestClient, store: AccountStore) -> None:
    record, token = store.create_key("ops", "admin")
    store.create_key("other", "read")
    store.revoke_key(record.id)
    response = _login(client, credential=token)
    assert response.status_code == 401
    assert "revoked" in response.text


# ---------------------------------------------------------------------------
# Sessions: expiry, forgery, garbage
# ---------------------------------------------------------------------------


def test_an_expired_session_is_refused(
    client: TestClient, store: AccountStore, config: Config
) -> None:
    from studioforge.gui.app import _storage_secret

    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    secret = _storage_secret(config)
    old = make_session_cookie(
        secret, "owner", "alice", store.password_fingerprint("alice"), now=time.time() - 40 * 86400
    )
    assert (
        client.get(
            "/", headers=HTML, cookies={COOKIE_NAME: old}, follow_redirects=False
        ).status_code
        == 303
    )


def test_a_forged_session_is_refused(client: TestClient, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    forged = make_session_cookie("not-the-panel-secret", "owner", "alice", "x" * 16)
    assert (
        client.get(
            "/", headers=HTML, cookies={COOKIE_NAME: forged}, follow_redirects=False
        ).status_code
        == 303
    )


@pytest.mark.parametrize(
    "garbage",
    [
        "owner",
        "owner.alice",
        "owner.alice.123",
        "owner..9999999999.abc",
        "root.alice.9999999999.abc",
        "owner.alice.notanumber.abc",
        "owner.alice.9999999999.",
        "..",
        "key.zzz.9999999999.00",
        "a" * 5000,
    ],
)
def test_garbage_cookies_are_a_plain_sign_out_never_a_500(
    client: TestClient, store: AccountStore, garbage: str
) -> None:
    store.create_key("agent", "read")
    response = client.get("/", headers=HTML, cookies={COOKIE_NAME: garbage}, follow_redirects=False)
    assert response.status_code == 303


def test_non_ascii_cookie_bytes_are_a_sign_out_not_a_crash(
    config: Config, state: _FakeState, store: AccountStore
) -> None:
    """Starlette decodes headers as latin-1; a stale cookie from another app can be anything."""
    from starlette.datastructures import Headers

    from studioforge.gui.app import authorize

    store.create_key("agent", "read")
    raw = f"{COOKIE_NAME}=\xe9.\xe9.\xe9.\xe9".encode("latin-1")
    headers = Headers(raw=[(b"cookie", raw), (b"authorization", b"Bearer " + b"\xe9\xe9")])
    assert authorize(headers, config, store, lambda: "secret") is None


# ---------------------------------------------------------------------------
# Lockout and CSRF
# ---------------------------------------------------------------------------


def test_repeated_wrong_credentials_are_locked_out_with_retry_after(
    client: TestClient, store: AccountStore
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    for _ in range(4):  # three free attempts, then the first lockout
        assert _login(client, username="alice", credential="wrong").status_code == 401
    locked = _login(client, username="alice", credential=PASSWORD)  # even the right one
    assert locked.status_code == 429
    assert int(locked.headers["retry-after"]) >= 1
    assert "Too many" in locked.text and PASSWORD not in locked.text


def test_a_good_sign_in_clears_the_failure_count(client: TestClient, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    for _ in range(2):
        _login(client, username="alice", credential="wrong")
    assert _login(client, username="alice", credential=PASSWORD).status_code == 303
    for _ in range(3):  # a fresh budget of free attempts
        assert _login(client, username="alice", credential="wrong").status_code == 401


def test_a_cross_site_login_post_is_refused(client: TestClient, store: AccountStore) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    response = client.post(
        LOGIN_PATH,
        data={"username": "alice", "credential": PASSWORD},
        headers={"Origin": "http://evil.example", "Host": "192.168.1.50:8080"},
        follow_redirects=False,
    )
    assert response.status_code == 403
    assert COOKIE_NAME not in response.cookies
    same = client.post(
        LOGIN_PATH,
        data={"username": "alice", "credential": PASSWORD},
        headers={"Origin": "http://192.168.1.50:8080", "Host": "192.168.1.50:8080"},
        follow_redirects=False,
    )
    assert same.status_code == 303


def test_the_submitted_credential_is_never_logged(
    client: TestClient, store: AccountStore, capsys: pytest.CaptureFixture[str]
) -> None:
    store.create_account("alice", PASSWORD)
    store.create_key("agent", "read")
    _login(client, username="alice", credential="hunter2-hunter2")
    _login(client, username="alice", credential=PASSWORD)
    captured = capsys.readouterr()
    assert "hunter2-hunter2" not in captured.out + captured.err
    assert PASSWORD not in captured.out + captured.err


# ---------------------------------------------------------------------------
# The gate on the ASGI scope: websocket, and the actor
# ---------------------------------------------------------------------------


async def test_the_gate_closes_an_unauthenticated_websocket_and_stamps_the_actor(
    config: Config, state: _FakeState, store: AccountStore
) -> None:
    _, token = store.create_key("ops", "admin")
    reached: list[dict[str, Any]] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached.append(scope)

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    gate = GuiAuthGate(inner, config, state)
    anonymous = {"type": "websocket", "path": "/_nicegui_ws/socket.io/", "headers": []}
    await gate(anonymous, receive, send)
    assert reached == [] and sent == [{"type": "websocket.close", "code": 1008}]

    signed = {
        "type": "http",
        "path": "/",
        "headers": [(b"authorization", f"Bearer {token}".encode())],
    }
    await gate(signed, receive, send)
    assert reached and reached[0]["state"]["panel_actor"].startswith("key:")


# ---------------------------------------------------------------------------
# viewer_may_change_box and the audit line on a setting change
# ---------------------------------------------------------------------------


def test_past_the_gate_is_admin_even_for_a_remote_viewer(
    config: Config, state: _FakeState, store: AccountStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui import tabs

    ctx = tabs.GuiContext(config=config, api_state=state)
    monkeypatch.setattr(tabs, "viewer_host", lambda: "192.168.1.50")
    assert tabs.viewer_may_change_box(ctx) is False  # open install, remote: the D32 rule
    store.create_key("agent", "read")  # a credential is now required
    assert tabs.viewer_may_change_box(ctx) is True


def test_viewer_actor_defaults_to_panel_outside_a_page() -> None:
    from studioforge.gui.tabs import viewer_actor

    assert viewer_actor() == "panel"


async def test_a_setting_saved_from_the_panel_leaves_an_audit_line_with_keys_only(
    config: Config, state: _FakeState
) -> None:
    from studioforge.gui import tabs

    ctx = tabs.GuiContext(config=config, api_state=state)
    await tabs.apply_config_updates(ctx, {"models.default_ctx": 12345})
    entries = state.audit.tail(5)  # type: ignore[attr-defined]
    line = next(e for e in entries if e["action"] == "panel.config.set")
    assert line["target"] == "models.default_ctx"
    assert line["surface"] == "panel" and line["actor"] == "panel"
    assert "12345" not in str(line)


async def test_a_state_without_an_audit_log_still_saves(config: Config) -> None:
    from studioforge.gui import tabs

    bare = _FakeState(config)
    assert not hasattr(bare, "audit")
    payload = await tabs.apply_config_updates(
        tabs.GuiContext(config=config, api_state=bare), {"models.default_ctx": 4096}
    )
    assert payload["updated"] == ["models.default_ctx"]
