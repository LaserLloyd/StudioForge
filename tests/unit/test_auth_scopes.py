"""Scoped authentication over the real FastAPI app.

``read < inference < admin``, enforced per route by ``api/auth.py:required_scope``;
the legacy ``server.api_key`` and the deprecated ``mcp.pin`` still work; auth is
required exactly when it should be; the bind rule refuses an unsafe start; every
admin mutation lands in the audit log; CORS is closed by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from studioforge.api.app import build_state, create_app
from studioforge.api.auth import (
    auth_enforced,
    check_bind_rule,
    has_admin_credential,
    remote_services,
    required_scope,
)
from studioforge.config import Config
from studioforge.errors import AuthError

LOOPBACK = ("127.0.0.1", 50000)
LAN = ("192.168.1.50", 50000)
PIN = "12345678"  # scrub-ok: fixture
LEGACY = "legacy-key-0123456789"  # scrub-ok: fixture


def make_config(tmp_path: Path, **overrides: Any) -> Config:
    base: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "models": {"dir": tmp_path / "models"},
        "gui": {"enabled": False, "port": 18080},
        "watchdog": {"enabled": False, "port": 11235},
        "server": {"port": 11234},
        "logging": {"level": "ERROR"},
        "mcp": {"pin": PIN},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return Config(**base)


def make_app(tmp_path: Path, **overrides: Any) -> Any:
    config = make_config(tmp_path, **overrides)
    return create_app(config, state=build_state(config), start_background=False)


def keys_for(app: Any) -> dict[str, str]:
    accounts = app.state.accounts
    return {
        scope: accounts.create_key(f"{scope}-key", scope)[1]
        for scope in ("read", "inference", "admin")
    }


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# posture: when is a credential required?
# ---------------------------------------------------------------------------


def test_a_loopback_install_with_no_keys_is_open(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    assert not auth_enforced(app.state.config, app.state.accounts)
    with TestClient(app, client=LOOPBACK) as client:
        assert client.get("/api/status").status_code == 200
        assert client.get("/v1/models").status_code == 200


def test_creating_a_key_turns_authentication_on(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys_for(app)
    with TestClient(app, client=LOOPBACK) as client:
        missing = client.get("/api/status")
        assert missing.status_code == 401
        body = missing.json()["error"]
        assert body["code"] == "invalid_api_key" and "keys create" in body["message"]
        assert client.get("/health").status_code == 200  # liveness stays public


def test_the_legacy_api_key_is_still_an_admin_credential(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": LEGACY})
    with TestClient(app, client=LOOPBACK) as client:
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", headers=bearer(LEGACY)).status_code == 200
        patched = client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers=bearer(LEGACY)
        )
        assert patched.status_code == 200
        names = [k.name for k in app.state.accounts.list_keys()]
        assert "legacy" in names  # mirrored into the key table by create_app


def test_a_network_bind_with_an_owner_account_requires_credentials(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"host": "0.0.0.0"})
    app.state.accounts.create_account("owner", "a-long-enough-password")
    assert auth_enforced(app.state.config, app.state.accounts)
    with TestClient(app, client=LAN) as client:
        assert client.get("/api/status").status_code == 401


# ---------------------------------------------------------------------------
# scopes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "scope"),
    [
        ("GET", "/api/status", "read"),
        ("GET", "/v1/models", "read"),
        ("GET", "/metrics", "read"),
        ("GET", "/docs", "read"),
        ("POST", "/v1/chat/completions", "inference"),
        ("POST", "/v1/messages", "inference"),
        ("POST", "/v1/responses", "inference"),
        ("POST", "/api/v0/chat/completions", "inference"),
        ("POST", "/api/v1/models/load", "inference"),
        ("POST", "/api/v1/models/unload", "inference"),
        ("POST", "/api/v1/models/download", "admin"),
        ("POST", "/api/models/x/load", "inference"),
        ("POST", "/api/models/x/load-recommended", "inference"),
        ("POST", "/api/models/x/unload", "inference"),
        ("POST", "/api/models/x/test", "inference"),
        ("POST", "/api/models/unload-all", "admin"),
        ("POST", "/api/models/scan", "admin"),
        ("DELETE", "/api/models/x", "admin"),
        ("POST", "/api/models/x/pin", "admin"),
        ("PATCH", "/api/config", "admin"),
        ("POST", "/api/restart/server", "admin"),
        ("POST", "/api/engine/install", "admin"),
        ("POST", "/api/update/install", "admin"),
        ("POST", "/api/downloads", "admin"),
        ("POST", "/api/leases", "admin"),
        ("GET", "/api/keys", "admin"),
        ("POST", "/api/keys", "admin"),
        ("GET", "/api/users", "admin"),
        ("GET", "/api/audit", "admin"),
        ("GET", "/api/doctor", "admin"),
        ("POST", "/api/support-bundle", "admin"),
        ("POST", "/api/server/stop", "admin"),
        ("POST", "/api/something/new", "admin"),  # unknown mutations fail closed
    ],
)
def test_required_scope_table(method: str, path: str, scope: str) -> None:
    assert required_scope(method, path, Config()) == scope


def test_deep_health_needs_inference_and_mcp_needs_read() -> None:
    config = Config()
    assert required_scope("GET", "/health", config, deep=True) == "inference"
    assert required_scope("GET", "/health", config) == "read"
    assert required_scope("POST", "/mcp", config) == "read"


def test_read_inference_and_admin_keys_see_exactly_their_routes(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys = keys_for(app)
    with TestClient(app, client=LOOPBACK) as client:
        # reads: any scope
        for scope in keys:
            assert client.get("/api/status", headers=bearer(keys[scope])).status_code == 200
        # residency writes: inference and up
        assert (
            client.post("/api/models/none/load", json={}, headers=bearer(keys["read"])).status_code
            == 403
        )
        loading = client.post("/api/models/none/load", json={}, headers=bearer(keys["inference"]))
        assert loading.status_code == 404  # authorised; the model simply does not exist
        # box changes: admin only
        for scope in ("read", "inference"):
            denied = client.patch(
                "/api/config", json={"models.default_ctx": 4096}, headers=bearer(keys[scope])
            )
            assert denied.status_code == 403
            error = denied.json()["error"]
            assert error["code"] == "permission_denied"
            assert error["studioforge"]["needed"] == "admin" or "admin" in error["message"]
        allowed = client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers=bearer(keys["admin"])
        )
        assert allowed.status_code == 200
        # inventory of who may log in is admin-only even to read
        assert client.get("/api/keys", headers=bearer(keys["inference"])).status_code == 403
        assert client.get("/api/keys", headers=bearer(keys["admin"])).status_code == 200


def test_inference_routes_need_inference_scope(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys = keys_for(app)
    body = {"model": "nope", "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(app, client=LOOPBACK) as client:
        assert (
            client.post("/v1/chat/completions", json=body, headers=bearer(keys["read"])).status_code
            == 403
        )
        reached = client.post("/v1/chat/completions", json=body, headers=bearer(keys["inference"]))
        assert reached.status_code == 404 and reached.json()["error"]["code"] == "model_not_found"
        assert client.get("/v1/models", headers=bearer(keys["read"])).status_code == 200
        assert client.get("/health?deep=true", headers=bearer(keys["read"])).status_code == 403


def test_a_scoped_key_is_enforced_even_when_no_credential_is_required(tmp_path: Path) -> None:
    """Presenting a credential means opting in to its limits."""
    app = make_app(tmp_path)
    app.state.accounts.create_key("ro", "read")  # creates the key -> enforced; use a fresh app
    app2 = make_app(tmp_path / "other")
    _record, token = app2.state.accounts.create_key("ro", "read")
    app2.state.accounts._invalidate_posture()
    with TestClient(app2, client=LOOPBACK) as client:
        denied = client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers=bearer(token)
        )
        assert denied.status_code == 403


def test_revoked_expired_and_rate_limited_keys_have_their_own_errors(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    accounts = app.state.accounts
    revoked, revoked_token = accounts.create_key("old", "admin")
    _fast, fast_token = accounts.create_key("fast", "read", rate_limit_rpm=2)
    accounts.revoke_key(revoked.id)
    with TestClient(app, client=LOOPBACK) as client:
        gone = client.get("/api/status", headers=bearer(revoked_token))
        assert gone.status_code == 401 and gone.json()["error"]["code"] == "api_key_revoked"
        assert client.get("/api/status", headers=bearer(fast_token)).status_code == 200
        assert client.get("/api/status", headers=bearer(fast_token)).status_code == 200
        limited = client.get("/api/status", headers=bearer(fast_token))
        assert limited.status_code == 429
        assert limited.json()["error"]["code"] == "key_rate_limited"
        assert int(limited.headers["retry-after"]) >= 1


def test_the_pin_is_mcp_only_once_authentication_is_enforced(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys_for(app)
    with TestClient(app, client=LOOPBACK) as client:
        off_mcp = client.get("/api/status", headers={"X-MCP-Pin": PIN})
        assert off_mcp.status_code == 401
        patched = client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers=bearer(PIN)
        )
        assert patched.status_code == 401


def test_the_pin_is_admin_on_the_mcp_path(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys_for(app)
    from starlette.requests import Request

    from studioforge.api.auth import check_request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"x-mcp-pin", PIN.encode())],
        "query_string": b"",
        "client": LOOPBACK,
        "app": app,
        "scheme": "http",
        "server": ("testserver", 80),
    }
    request = Request(scope)
    check_request(request, app.state.config)
    assert request.state.principal.kind == "pin" and request.state.principal.scope == "admin"


def test_a_weak_key_cannot_reset_the_pin_lockout(tmp_path: Path) -> None:
    """A read key's success must not forgive wrong PIN guesses (escalation to admin)."""
    from starlette.requests import Request

    from studioforge.api.auth import GUARD, check_request

    app = make_app(tmp_path)
    tokens = keys_for(app)

    def call(method: str, path: str, headers: list[tuple[bytes, bytes]]) -> str | None:
        request = Request(
            {
                "type": "http",
                "method": method,
                "path": path,
                "headers": headers,
                "query_string": b"",
                "client": LAN,
                "app": app,
                "scheme": "http",
                "server": ("testserver", 80),
            }
        )
        try:
            check_request(request, app.state.config)
        except AuthError as exc:
            return exc.code
        return None

    wrong = [(b"x-mcp-pin", b"00000000")]
    WRONG_PIN = {"invalid_api_key", "invalid_mcp_pin"}  # noqa: N806
    read = [(b"authorization", f"Bearer {tokens['read']}".encode())]
    GUARD.reset()
    try:
        for _ in range(3):  # the free attempts
            assert call("POST", "/mcp", wrong) in WRONG_PIN
        assert call("GET", "/api/status", read) is None  # a genuine, weak credential
        assert call("POST", "/mcp", wrong) in WRONG_PIN  # 4th failure arms the lockout
        assert call("POST", "/mcp", wrong) == "too_many_credential_attempts"
        assert call("GET", "/api/status", read) == "too_many_credential_attempts"
    finally:
        GUARD.reset()
    # An admin credential still forgives, as before.
    try:
        for _ in range(3):
            call("POST", "/mcp", wrong)
        admin = [(b"authorization", f"Bearer {tokens['admin']}".encode())]
        assert call("GET", "/api/status", admin) is None
        for _ in range(3):
            assert call("POST", "/mcp", wrong) in WRONG_PIN
    finally:
        GUARD.reset()


# ---------------------------------------------------------------------------
# keys / users / audit over HTTP
# ---------------------------------------------------------------------------


def test_keys_over_http_show_the_token_once(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    _admin_record, admin = app.state.accounts.create_key("root", "admin")
    with TestClient(app, client=LOOPBACK) as client:
        created = client.post(
            "/api/keys",
            json={"name": "agent", "scope": "inference", "rate_limit_rpm": 60},
            headers=bearer(admin),
        ).json()
        token = created["token"]
        assert token.startswith("sf-i-") and created["key"]["scope"] == "inference"
        listing = client.get("/api/keys", headers=bearer(admin)).json()
        assert token not in json.dumps(listing)
        assert any(k["name"] == "agent" and k["rate_limit_rpm"] == 60 for k in listing["keys"])
        # the new key works, then dies when revoked
        assert client.get("/api/status", headers=bearer(token)).status_code == 200
        revoked = client.delete(f"/api/keys/{created['key']['id']}", headers=bearer(admin))
        assert revoked.status_code == 200 and revoked.json()["key"]["revoked"] is True
        assert client.get("/api/status", headers=bearer(token)).status_code == 401
        bad = client.delete("/api/keys/nope", headers=bearer(admin))
        assert bad.status_code == 400


def test_users_over_http_create_an_owner_and_never_echo_a_password(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app, client=LOOPBACK) as client:
        made = client.post(
            "/api/users", json={"username": "owner", "password": "a-long-password-1"}
        )
        assert made.status_code == 200 and made.json()["user"]["role"] == "owner"
        assert "a-long-password-1" not in made.text
        short = client.post("/api/users", json={"username": "x", "password": "short"})
        assert short.status_code == 400
        changed = client.post(
            "/api/users/password", json={"username": "owner", "password": "another-long-pass-2"}
        )
        assert changed.status_code == 200
        assert client.get("/api/users").json()["users"][0]["username"] == "owner"
    assert app.state.accounts.verify_password("owner", "another-long-pass-2") is not None


def test_admin_mutations_are_audited_and_reads_are_not(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    keys = keys_for(app)
    with TestClient(app, client=LOOPBACK) as client:
        client.get("/api/status", headers=bearer(keys["admin"]))
        client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers=bearer(keys["admin"])
        )
        client.patch("/api/config", json={"models.default_ctx": 2048}, headers=bearer(keys["read"]))
        entries = client.get("/api/audit", headers=bearer(keys["admin"])).json()["entries"]
    actions = [(e["action"], e["result"], e["actor"].split(":")[0]) for e in entries]
    assert ("api.PATCH /api/config", "ok", "key") in actions
    assert ("api.PATCH /api/config", "denied", "key") in actions
    assert not any(e["action"].startswith("api.GET") for e in entries)
    ok = next(e for e in entries if e["result"] == "ok")
    assert ok["target"] == "models.default_ctx" and ok["peer"] == "127.0.0.1"
    assert set(ok) >= {"ts", "actor", "action", "target", "result", "surface"}


def test_the_audit_log_is_jsonl_append_only_and_rotates(tmp_path: Path) -> None:
    from studioforge.auditlog import AuditLog

    log = AuditLog(tmp_path / "logs", check_every=1)
    log.record(actor="key:1", action="a", target="t")
    log.record(actor="key:2", action="b", target="u", result="denied", detail="why")
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["action"] for line in lines] == ["a", "b"]
    first = lines[0]
    log.record(actor="key:3", action="c", target="v")
    assert log.path.read_text(encoding="utf-8").splitlines()[0] == first  # never edited
    log._max_bytes = 200  # force a rotation
    for i in range(5):
        log.record(actor="key:9", action=f"spam{i}", target="x" * 40)
    rotated = [p for p in (tmp_path / "logs").iterdir() if p.name.startswith("audit.2")]
    assert rotated, "the audit log should have rotated to audit.<stamp>.jsonl"
    assert log.tail(3)  # the live file still reads


def test_set_config_never_puts_a_value_in_the_audit_log(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    _record, admin = app.state.accounts.create_key("root", "admin")
    secret = "hf_" + "a" * 30
    with TestClient(app, client=LOOPBACK) as client:
        client.patch("/api/config", json={"hf.token": secret}, headers=bearer(admin))
    text = (app.state.config.logs_dir / "audit.jsonl").read_text(encoding="utf-8")
    assert "hf.token" in text and secret not in text


# ---------------------------------------------------------------------------
# CORS and the Origin gate
# ---------------------------------------------------------------------------


def test_cors_is_closed_by_default(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app, client=LOOPBACK) as client:
        preflight = client.options(
            "/api/status",
            headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in {k.lower() for k in preflight.headers}
        plain = client.get("/api/status", headers={"Origin": "http://evil.example"})
        assert "access-control-allow-origin" not in {k.lower() for k in plain.headers}


def test_an_explicit_cors_origin_is_honoured_without_credentials(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"cors_origins": ["https://app.example"]})
    with TestClient(app, client=LOOPBACK) as client:
        reply = client.get("/api/status", headers={"Origin": "https://app.example"})
        assert reply.headers["access-control-allow-origin"] == "https://app.example"
        assert "access-control-allow-credentials" not in {k.lower() for k in reply.headers}


def test_a_foreign_origin_never_reaches_the_mcp_endpoint(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    _record, admin = app.state.accounts.create_key("root", "admin")
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    headers = {**bearer(admin), "Content-Type": "application/json", "Accept": "application/json"}
    with TestClient(app, client=LOOPBACK) as client:
        for origin in ("http://evil.example", "null", "http://127.0.0.1.evil.example"):
            refused = client.post("/mcp", json=body, headers={**headers, "Origin": origin})
            assert refused.status_code == 403, origin
            assert refused.json()["error"]["code"] == "origin_not_allowed"
        for origin in ("http://127.0.0.1:3000", "http://localhost:5173"):
            passed = client.post("/mcp", json=body, headers={**headers, "Origin": origin})
            assert passed.status_code != 403, origin


def test_a_configured_origin_may_reach_mcp_but_a_wildcard_may_not(tmp_path: Path) -> None:
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    app = make_app(tmp_path, server={"cors_origins": ["https://app.example", "*"]})
    _record, admin = app.state.accounts.create_key("root", "admin")
    headers = {**bearer(admin), "Content-Type": "application/json", "Accept": "application/json"}
    with TestClient(app, client=LOOPBACK) as client:
        ok = client.post("/mcp", json=body, headers={**headers, "Origin": "https://app.example"})
        assert ok.status_code != 403
        wild = client.post(
            "/mcp", json=body, headers={**headers, "Origin": "https://other.example"}
        )
        assert wild.status_code == 403


# ---------------------------------------------------------------------------
# the bind rule
# ---------------------------------------------------------------------------


def test_remote_services_lists_only_enabled_network_listeners() -> None:
    config = Config(
        server={"host": "0.0.0.0"},
        gui={"enabled": False, "host": "0.0.0.0"},
        watchdog={"host": "192.168.1.5"},
    )
    assert [name for name, _ in remote_services(config)] == ["the gateway", "the watchdog"]
    assert remote_services(Config()) == []


def test_the_bind_rule_refuses_a_network_bind_with_no_credential(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"host": "0.0.0.0"})
    config, accounts = app.state.config, app.state.accounts
    with pytest.raises(AuthError) as excinfo:
        check_bind_rule(config, accounts)
    error = excinfo.value
    assert error.code == "auth_required_for_remote_bind"
    assert "studioforge setup" in error.message and "keys create" in error.message
    assert "the gateway" in error.message
    # each way to satisfy it
    accounts.create_account("owner", "a-long-enough-password")
    check_bind_rule(config, accounts)
    assert has_admin_credential(config, accounts)


def test_an_admin_key_or_the_legacy_key_satisfies_the_bind_rule(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"host": "0.0.0.0"})
    config, accounts = app.state.config, app.state.accounts
    accounts.create_key("read-only", "read")
    with pytest.raises(AuthError):
        check_bind_rule(config, accounts)  # a read key is not an admin credential
    accounts.create_key("root", "admin")
    check_bind_rule(config, accounts)
    legacy = make_config(tmp_path / "b", server={"host": "0.0.0.0", "api_key": LEGACY})
    check_bind_rule(legacy, None)


def test_a_loopback_bind_never_needs_a_credential(tmp_path: Path) -> None:
    check_bind_rule(make_config(tmp_path), None)


def test_serve_refuses_to_start_on_a_network_bind_without_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.__main__ import app as cli
    from studioforge.core.ports import EXIT_AUTH_REQUIRED

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path, server={"host": "0.0.0.0"})
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    started: list[object] = []
    monkeypatch.setattr("studioforge.__main__._serve", lambda *a, **k: started.append(1))
    result = CliRunner().invoke(cli, ["serve", "--config", str(path), "--no-gui", "--no-watchdog"])
    assert result.exit_code == EXIT_AUTH_REQUIRED
    assert "auth_required_for_remote_bind" in (result.output + str(result.exception)) or (
        "studioforge setup" in result.output
    )
    assert not started


def test_serve_proceeds_once_an_owner_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.__main__ import app as cli
    from studioforge.core.accounts import AccountStore
    from studioforge.db import Database

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path, server={"host": "0.0.0.0"})
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    db = Database(config.db_path)
    db.migrate()
    AccountStore(db, config.data_dir).create_account("owner", "a-long-enough-password")
    db.close()
    started: list[object] = []

    async def fake_serve(*args: object, **kwargs: object) -> int:
        started.append(1)
        return 0

    monkeypatch.setattr("studioforge.__main__._serve", fake_serve)
    result = CliRunner().invoke(cli, ["serve", "--config", str(path), "--no-gui", "--no-watchdog"])
    assert result.exit_code == 0, result.output
    assert started


# ---------------------------------------------------------------------------
# defaults and migration
# ---------------------------------------------------------------------------


def test_all_three_services_default_to_loopback_and_cors_to_empty() -> None:
    config = Config()
    assert config.server.host == config.gui.host == config.watchdog.host == "127.0.0.1"
    assert config.server.cors_origins == [] and config.server.cors_allow_credentials is False
    assert config.server.tls.mode == "off"


def test_an_existing_config_keeps_its_explicit_values(tmp_path: Path) -> None:
    """The breaking change is a new DEFAULT; a config that wrote 0.0.0.0 keeps it."""
    from studioforge.config import load_config

    path = tmp_path / "data" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "server:\n  host: 0.0.0.0\n  cors_origins: ['*']\ngui:\n  host: 0.0.0.0\n"
        "watchdog:\n  host: 0.0.0.0\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.server.host == config.gui.host == config.watchdog.host == "0.0.0.0"
    assert config.server.cors_origins == ["*"]
    assert [name for name, _ in remote_services(config)] == [
        "the gateway",
        "the control panel",
        "the watchdog",
    ]


# ---------------------------------------------------------------------------
# the legacy key through its whole life
# ---------------------------------------------------------------------------


def test_setting_and_rotating_server_api_key_over_http_keeps_the_mirror_in_step(
    tmp_path: Path,
) -> None:
    from studioforge.authkeys import hash_token

    app = make_app(tmp_path)
    accounts = app.state.accounts
    with TestClient(app, client=LOOPBACK) as client:
        # an open loopback install: the local caller may set the key
        assert client.patch("/api/config", json={"server.api_key": LEGACY}).status_code == 200
        live = [k for k in accounts.list_keys(include_revoked=False) if k.name == "legacy"]
        assert len(live) == 1 and live[0].key_hash == hash_token(LEGACY)
        assert client.get("/api/status").status_code == 401  # and now it is required
        rotated = "rotated-key-9876543210"  # scrub-ok: fixture
        assert (
            client.patch(
                "/api/config", json={"server.api_key": rotated}, headers=bearer(LEGACY)
            ).status_code
            == 200
        )
        live = [k for k in accounts.list_keys(include_revoked=False) if k.name == "legacy"]
        assert len(live) == 1 and live[0].key_hash == hash_token(rotated)
        assert client.get("/api/status", headers=bearer(LEGACY)).status_code == 401
        assert client.get("/api/status", headers={"X-API-Key": rotated}).status_code == 200


def test_revoking_legacy_disables_the_config_key_too(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": LEGACY})
    _record, admin = app.state.accounts.create_key("root", "admin")
    with TestClient(app, client=LOOPBACK) as client:
        assert client.get("/api/status", headers=bearer(LEGACY)).status_code == 200
        assert client.delete("/api/keys/legacy", headers=bearer(admin)).status_code == 200
        gone = client.get("/api/status", headers=bearer(LEGACY))
        assert gone.status_code == 401 and gone.json()["error"]["code"] == "api_key_revoked"
        assert client.get("/api/status", headers=bearer(admin)).status_code == 200


def test_auth_status_is_honest_about_the_posture(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": LEGACY, "host": "0.0.0.0"})
    with TestClient(app, client=LAN) as client:
        body = client.get("/api/auth/status", headers=bearer(LEGACY)).json()
    assert body["auth_required"] is True and body["admin_credential_exists"] is True
    assert body["legacy_api_key_set"] is True and body["mcp_pin_set"] is True
    assert body["bind"]["remote"] == [{"service": "the gateway", "host": "0.0.0.0"}]
    assert body["tls"]["mode"] == "off" and "owner_exists" in body
    assert LEGACY not in json.dumps(body) and PIN not in json.dumps(body)


def test_a_running_server_makes_sure_local_helpers_have_a_key(tmp_path: Path) -> None:
    """With authentication on, local helpers need ``cli.key`` before the first 401."""
    config = make_config(tmp_path)
    state = build_state(config)
    state.accounts.create_key("someone", "read")
    app = create_app(config, state=state, start_background=True)
    try:
        key_file = config.data_dir / "cli.key"
        assert key_file.is_file()
        principal = state.accounts.authenticate(key_file.read_text(encoding="utf-8").strip())
        assert principal is not None and principal.scope == "admin"
    finally:
        lock = getattr(app.state, "instance_lock", None)
        if lock is not None:
            lock.release()
    quiet = make_config(tmp_path / "quiet")
    quiet_app = create_app(quiet, state=build_state(quiet), start_background=True)
    try:
        assert not (quiet.data_dir / "cli.key").exists()  # nothing asks for a credential yet
    finally:
        lock = getattr(quiet_app.state, "instance_lock", None)
        if lock is not None:
            lock.release()
