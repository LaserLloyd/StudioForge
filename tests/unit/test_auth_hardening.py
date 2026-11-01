"""Loopback-trust hardening: Host validation, ``server.behind_proxy``, key handling.

Covers the DNS-rebinding guard on the same-machine admin shortcut, the proxy switch
that turns the shortcut off, the legacy ``server.api_key`` floor and redaction, the
MCP policy failing closed off a missing header map, the cached legacy-key adoption,
the bind rule at ``PATCH /api/config`` time, and the ``/logout`` and ``/api/v1`` scope
details.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.api.auth import host_header_trusted, required_scope, trusted_local_request
from studioforge.authkeys import LOCAL_ADMIN
from studioforge.config import (
    MIN_API_KEY_LENGTH,
    Config,
    apply_overrides,
    redact,
)
from studioforge.errors import ConfigError
from studioforge.mcp_policy import PolicyExtension, _is_http_call

LOOPBACK = ("127.0.0.1", 50000)
PIN = "12345678"  # scrub-ok: fixture
GOOD_KEY = "a-sufficiently-long-key-0123"  # scrub-ok: fixture


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


def _req(host: str | None, *, peer: str | None = "127.0.0.1", origin: str | None = None) -> Any:
    headers: dict[str, str] = {}
    if host is not None:
        headers["host"] = host
    if origin is not None:
        headers["origin"] = origin
    return SimpleNamespace(
        headers=headers,
        client=SimpleNamespace(host=peer) if peer else None,
        url=SimpleNamespace(scheme="http"),
    )


# ---------------------------------------------------------------------------
# #4 Host validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("host", ["127.0.0.1:1234", "localhost", "[::1]:8080", None, ""])
def test_loopback_and_missing_hosts_are_this_server(tmp_path: Path, host: str | None) -> None:
    assert host_header_trusted(_req(host), make_config(tmp_path))


def test_a_foreign_host_is_not_this_server(tmp_path: Path) -> None:
    assert not host_header_trusted(_req("evil.test"), make_config(tmp_path))
    assert not host_header_trusted(_req("evil.test:1234"), make_config(tmp_path))


def test_allowed_hosts_and_explicit_cors_origins_extend_the_set(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        server={
            "allowed_hosts": ["box.lan", "other.lan:9000"],
            "cors_origins": ["https://panel.example"],
        },
    )
    assert host_header_trusted(_req("box.lan:1234"), config)
    assert host_header_trusted(_req("other.lan:9000"), config)
    assert not host_header_trusted(_req("other.lan:1"), config)
    assert host_header_trusted(_req("panel.example"), config)


def test_a_wildcard_cors_origin_does_not_trust_any_host(tmp_path: Path) -> None:
    config = make_config(tmp_path, server={"cors_origins": ["*"]})
    assert not host_header_trusted(_req("evil.test"), config)


def test_dns_rebinding_with_a_matching_origin_is_not_a_local_admin(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    rebinding = _req("evil.test:1234", origin="http://evil.test:1234")
    assert not trusted_local_request(rebinding, config)
    assert trusted_local_request(_req("127.0.0.1:1234"), config)
    # An in-process call has no peer at all and stays trusted.
    assert trusted_local_request(_req("evil.test", peer=None), config)


def test_rebinding_cannot_edit_config_or_read_the_pin_over_http(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    evil = {"Host": "evil.test:11234", "Origin": "http://evil.test:11234"}
    with TestClient(app, client=LOOPBACK) as client:
        patched = client.patch("/api/config", json={"models.default_ctx": 4096}, headers=evil)
        assert patched.status_code == 403
        assert patched.json()["error"]["code"] == "remote_admin_requires_credential"
        info = client.get("/api/mcp/info", headers=evil).json()
        assert PIN not in str(info)
        # The same request addressed to loopback still works.
        ok = client.patch(
            "/api/config", json={"models.default_ctx": 4096}, headers={"Host": "127.0.0.1:11234"}
        )
        assert ok.status_code == 200


# ---------------------------------------------------------------------------
# #5 behind_proxy
# ---------------------------------------------------------------------------


def test_behind_proxy_turns_loopback_trust_off(tmp_path: Path) -> None:
    config = make_config(tmp_path, server={"behind_proxy": True})
    assert not trusted_local_request(_req("127.0.0.1:1234"), config)
    assert trusted_local_request(_req("127.0.0.1:1234", peer=None), config)


def test_behind_proxy_refuses_an_uncredentialed_config_edit_from_loopback(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"behind_proxy": True})
    with TestClient(app, client=LOOPBACK) as client:
        denied = client.patch("/api/config", json={"models.default_ctx": 4096})
        assert denied.status_code == 403
        assert PIN not in str(client.get("/api/mcp/info").json())
        # A real credential still opens it.
        _record, token = app.state.accounts.create_key("admin", "admin")
        allowed = client.patch(
            "/api/config",
            json={"models.default_ctx": 4096},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert allowed.status_code == 200


def test_doctor_flags_an_owner_without_keys_on_a_loopback_install(tmp_path: Path) -> None:
    from studioforge.core.doctor import DoctorReport, _check_auth

    app = make_app(tmp_path)
    app.state.accounts.create_account("owner", "a-long-enough-password")
    report = DoctorReport()
    _check_auth(report, app.state.config, app.state.accounts)
    assert "auth.loopback_trust_owner_no_keys" in {f.check_id for f in report.findings}

    proxied = DoctorReport()
    app.state.config.server.behind_proxy = True
    _check_auth(proxied, app.state.config, app.state.accounts)
    assert "auth.loopback_trust_owner_no_keys" not in {f.check_id for f in proxied.findings}


# ---------------------------------------------------------------------------
# #7 legacy key floor and redaction
# ---------------------------------------------------------------------------


def test_a_new_api_key_needs_sixteen_characters(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    with pytest.raises(ConfigError, match=str(MIN_API_KEY_LENGTH)):
        apply_overrides(config, {"server.api_key": "short-key-123"})
    assert apply_overrides(config, {"server.api_key": GOOD_KEY}).server.api_key == GOOD_KEY


def test_an_existing_short_key_still_loads_and_unrelated_edits_pass(tmp_path: Path) -> None:
    config = make_config(tmp_path, server={"api_key": "old-key-12"})
    assert apply_overrides(config, {"models.default_ctx": 4096}).server.api_key == "old-key-12"


def test_redact_is_a_constant_shape_and_never_shows_a_character(tmp_path: Path) -> None:
    masked = redact(GOOD_KEY)
    assert masked is not None
    assert masked.startswith("redacted:") and len(masked) == len("redacted:") + 8
    assert not any(GOOD_KEY[i : i + 3] in masked for i in range(len(GOOD_KEY) - 2))
    assert redact(PIN) == "redacted:********"  # a short secret is never fingerprinted
    assert redact(None) is None and redact("") is None


def test_the_config_route_and_the_legacy_mirror_leak_no_plaintext(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": GOOD_KEY})
    with TestClient(app, client=LOOPBACK) as client:
        body = client.get("/api/config", headers={"Authorization": f"Bearer {GOOD_KEY}"}).json()
        assert body["config"]["server"]["api_key"] == redact(GOOD_KEY)
    legacy = [k for k in app.state.accounts.list_keys() if k.name == "legacy"]
    assert legacy and all(GOOD_KEY[:4] not in k.prefix for k in legacy)


# ---------------------------------------------------------------------------
# #33 MCP policy fails closed
# ---------------------------------------------------------------------------


def _ctx(kind: str | None, *, request: Any = None) -> Any:
    transport = SimpleNamespace(kind=kind, headers=None) if kind else None
    dispatch = SimpleNamespace(transport=transport)
    return SimpleNamespace(
        session=SimpleNamespace(_dispatch_context=dispatch), request=request, params=None
    )


def test_http_is_recognised_from_the_transport_not_from_headers() -> None:
    assert _is_http_call(_ctx("streamable-http"))
    assert _is_http_call(_ctx("jsonrpc", request=object()))
    assert not _is_http_call(_ctx("jsonrpc"))
    assert not _is_http_call(_ctx(None))


@pytest.mark.asyncio
async def test_an_http_call_without_readable_headers_is_denied_not_local_admin() -> None:
    seen: list[Any] = []

    def resolve(_headers: Any) -> Any:  # would be reached only if headers were found
        seen.append(1)
        return LOCAL_ADMIN

    ext = PolicyExtension("gateway", resolve)
    params = SimpleNamespace(name="delete_model", arguments={"model_id": "x"})
    called: list[Any] = []

    async def call_next(_ctx_: Any) -> Any:
        called.append(1)
        return "ran"

    denied = await ext.intercept_tool_call(params, _ctx("streamable-http"), call_next)  # type: ignore[arg-type]
    assert called == [] and seen == []
    assert denied.is_error is True

    allowed = await ext.intercept_tool_call(params, _ctx("jsonrpc"), call_next)  # type: ignore[arg-type]
    assert allowed == "ran" and called == [1]


# ---------------------------------------------------------------------------
# #34 cached legacy adoption
# ---------------------------------------------------------------------------


def test_legacy_adoption_does_not_scan_the_key_table_per_request(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": GOOD_KEY})
    accounts = app.state.accounts
    calls = {"n": 0}
    real = accounts.list_keys

    def counting(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real(*a, **k)

    accounts.list_keys = counting
    accounts.adopt_legacy_key(GOOD_KEY)  # warm
    before = calls["n"]
    for _ in range(25):
        assert accounts.authenticate_legacy(GOOD_KEY) is not None
    assert calls["n"] == before


def test_revoking_the_legacy_mirror_is_seen_immediately(tmp_path: Path) -> None:
    from studioforge.errors import AuthError

    app = make_app(tmp_path, server={"api_key": GOOD_KEY})
    accounts = app.state.accounts
    assert accounts.authenticate_legacy(GOOD_KEY) is not None
    assert accounts.revoke_key("legacy") is not None
    with pytest.raises(AuthError):
        accounts.authenticate_legacy(GOOD_KEY)


# ---------------------------------------------------------------------------
# #35 bind rule at PATCH time
# ---------------------------------------------------------------------------


def test_patching_a_network_bind_without_an_admin_credential_is_refused(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    path = app.state.config.config_path
    with TestClient(app, client=LOOPBACK) as client:
        refused = client.patch("/api/config", json={"server.host": "0.0.0.0"})
        assert refused.status_code == 503
        assert refused.json()["error"]["code"] == "auth_required_for_remote_bind"
        assert app.state.config.server.host == "127.0.0.1"
        assert not path or not path.exists() or "0.0.0.0" not in path.read_text("utf-8")
        # With an admin credential created first, the same edit is accepted.
        _record, token = app.state.accounts.create_key("admin", "admin")
        ok = client.patch(
            "/api/config",
            json={"server.host": "0.0.0.0"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status_code == 200


# ---------------------------------------------------------------------------
# #17 small items
# ---------------------------------------------------------------------------


def test_unknown_native_api_v1_posts_are_admin_not_inference() -> None:
    assert required_scope("POST", "/api/v1/chat") == "inference"
    assert required_scope("POST", "/api/v1/models/load") == "inference"
    assert required_scope("POST", "/api/v1/models/download") == "admin"
    assert required_scope("POST", "/api/v1/something-new") == "admin"
    assert required_scope("POST", "/v1/chat/completions") == "inference"


def test_get_logout_changes_nothing_and_post_signs_out(tmp_path: Path) -> None:
    from studioforge.gui.app import COOKIE_NAME, create_gui_app
    from tests.unit.test_gui import _FakeState

    config = make_config(tmp_path)
    config.ensure_dirs()
    with TestClient(create_gui_app(config, api_state=_FakeState(config))) as client:
        client.cookies.set(COOKIE_NAME, "x")
        page = client.get("/logout", follow_redirects=False)
        assert page.status_code == 200 and "<form" in page.text
        assert "set-cookie" not in page.headers
        out = client.post("/logout", follow_redirects=False)
        assert out.status_code == 303 and out.headers["location"] == "/login"
        assert COOKIE_NAME in out.headers.get("set-cookie", "")
        cross = client.post(
            "/logout",
            headers={"Origin": "http://evil.test", "Host": "127.0.0.1:8080"},
            follow_redirects=False,
        )
        assert cross.status_code == 403
