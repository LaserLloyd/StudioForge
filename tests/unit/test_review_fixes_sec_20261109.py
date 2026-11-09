"""Security review fixes (2026-11-09).

1. ``server.behind_proxy`` / ``server.allowed_hosts`` apply live (PATCH and MCP).
2. A redacted placeholder is never written back as the secret (dotted, section, MCP).
3. Behind a proxy, an owner account makes the panel offer its sign-in.
4. The bind rule and the key floor hold for every config-edit surface and form.
5. The watchdog's ``/health`` detail uses the gateway's "this machine" rule.
7. An ``allowed_hosts`` URL without a port matches any port.

Findings 6 (TLS) and 8 (bundle, owner-only files, legacy prefixes) are tested in
test_tls_san_lock, test_doctor_bundle, test_secure_files_order and test_auth_hardening.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.api.auth import (
    GUARD,
    host_header_trusted,
    panel_sign_in_required,
    trusted_local_request,
)
from studioforge.authkeys import OWNER_EVER_FILE, host_names_this_server
from studioforge.config import (
    MIN_API_KEY_LENGTH,
    REDACTED_PREFIX,
    Config,
    apply_overrides,
    redact_config_dict,
)
from studioforge.errors import AuthError, ConfigError

LOOPBACK = ("127.0.0.1", 50000)
PIN = "12345678"  # scrub-ok: fixture
GOOD_KEY = "a-sufficiently-long-key-0123"  # scrub-ok: fixture
PASSWORD = "correct horse battery"  # scrub-ok: fixture
LOCAL = {"Host": "127.0.0.1:11234"}


@pytest.fixture(autouse=True)
def _fresh_guard() -> Iterator[None]:
    GUARD.reset()
    yield
    GUARD.reset()


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


async def call_tool(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# 1. behind_proxy and allowed_hosts take effect without a restart
# ---------------------------------------------------------------------------


def test_patch_behind_proxy_turns_loopback_trust_off_at_once(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    config = app.state.config
    assert trusted_local_request(_req("127.0.0.1:11234"), config)
    with TestClient(app, client=LOOPBACK) as client:
        changed = client.patch("/api/config", json={"server.behind_proxy": True}, headers=LOCAL)
        assert changed.status_code == 200
        assert changed.json()["restart_required"] == []
        assert config.server.behind_proxy is True
        assert not trusted_local_request(_req("127.0.0.1:11234"), config)
        # The very next box change from loopback now needs a real credential.
        denied = client.patch("/api/config", json={"models.default_ctx": 4096}, headers=LOCAL)
        assert denied.status_code == 403
        assert denied.json()["error"]["code"] == "remote_admin_requires_credential"


def test_patch_allowed_hosts_is_honoured_by_the_next_request(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    config = app.state.config
    named = {"Host": "box.lan:11234"}
    with TestClient(app, client=LOOPBACK) as client:
        before = client.patch("/api/config", json={"models.default_ctx": 4096}, headers=named)
        assert before.status_code == 403  # a Host that is not this server: the rebinding shape
        ok = client.patch("/api/config", json={"server.allowed_hosts": ["box.lan"]}, headers=LOCAL)
        assert ok.status_code == 200 and ok.json()["restart_required"] == []
        assert config.server.allowed_hosts == ["box.lan"]
        assert trusted_local_request(_req("box.lan:11234"), config)
        after = client.patch("/api/config", json={"models.default_ctx": 4096}, headers=named)
        assert after.status_code == 200


async def test_mcp_set_config_applies_behind_proxy_and_allowed_hosts_live(tmp_path: Path) -> None:
    from studioforge.mcp.management import build_management_mcp

    app = make_app(tmp_path)
    server = build_management_mcp(app.state)
    payload = await call_tool(
        server,
        "set_config",
        updates={"server.behind_proxy": True, "server.allowed_hosts": ["box.lan:9000"]},
    )
    assert payload["ok"] is True and payload["restart_required"] == []
    live = app.state.config
    assert live.server.behind_proxy is True and live.server.allowed_hosts == ["box.lan:9000"]
    assert not trusted_local_request(_req("127.0.0.1:11234"), live)


# ---------------------------------------------------------------------------
# 2. Redacted placeholders are refused
# ---------------------------------------------------------------------------


def _with_secrets(tmp_path: Path) -> Config:
    return make_config(tmp_path, server={"api_key": GOOD_KEY}, hf={"token": "hf_" + "x" * 30})


@pytest.mark.parametrize("dotted", ["server.api_key", "mcp.pin", "hf.token"])
def test_a_dotted_placeholder_is_refused_and_names_the_field(tmp_path: Path, dotted: str) -> None:
    config = _with_secrets(tmp_path)
    section, key = dotted.split(".")
    placeholder = redact_config_dict(config.to_yaml_dict())[section][key]
    assert str(placeholder).startswith(REDACTED_PREFIX)
    with pytest.raises(ConfigError) as caught:
        apply_overrides(config, {dotted: placeholder})
    assert dotted in caught.value.message and "placeholder" in caught.value.message
    assert caught.value.param == dotted


@pytest.mark.parametrize("section", ["server", "mcp", "hf"])
def test_sending_a_redacted_section_back_is_refused(tmp_path: Path, section: str) -> None:
    config = _with_secrets(tmp_path)
    dumped = redact_config_dict(config.to_yaml_dict())[section]
    with pytest.raises(ConfigError, match="placeholder"):
        apply_overrides(config, {section: dumped})


def test_real_values_and_clearing_still_work(tmp_path: Path) -> None:
    config = _with_secrets(tmp_path)
    updated = apply_overrides(config, {"server.api_key": "another-real-key-0123", "mcp.pin": None})
    assert updated.server.api_key == "another-real-key-0123" and updated.mcp.pin is None


def test_a_stored_value_that_looks_redacted_does_not_block_other_edits(tmp_path: Path) -> None:
    poisoned = REDACTED_PREFIX + "3b720401"  # what 2.3.0 let a client write back
    config = make_config(tmp_path, server={"api_key": poisoned})
    assert apply_overrides(config, {"models.default_ctx": 4096}).server.api_key == poisoned


def test_patch_refuses_a_placeholder_and_keeps_the_key(tmp_path: Path) -> None:
    app = make_app(tmp_path, server={"api_key": GOOD_KEY})
    auth = {"Authorization": f"Bearer {GOOD_KEY}", **LOCAL}
    with TestClient(app, client=LOOPBACK) as client:
        dumped = client.get("/api/config", headers=auth).json()["config"]
        refused = client.patch(
            "/api/config", json={"server.api_key": dumped["server"]["api_key"]}, headers=auth
        )
        assert refused.status_code == 400
        assert refused.json()["error"]["code"] == "invalid_config"
    assert app.state.config.server.api_key == GOOD_KEY


async def test_mcp_set_config_refuses_a_placeholder(tmp_path: Path) -> None:
    from studioforge.mcp.management import build_management_mcp

    app = make_app(tmp_path)
    server = build_management_mcp(app.state)
    payload = await call_tool(server, "set_config", updates={"mcp.pin": REDACTED_PREFIX + "*" * 8})
    assert payload["ok"] is False
    assert "mcp.pin" in payload["error"]["message"]
    assert app.state.config.mcp.pin == PIN


# ---------------------------------------------------------------------------
# 3. Behind a proxy, an owner account means a panel sign-in
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Any:
    from studioforge.core.accounts import AccountStore
    from studioforge.db import Database

    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    return AccountStore(db, tmp_path / "store")


def test_panel_sign_in_follows_behind_proxy_and_the_owner(tmp_path: Path, store: Any) -> None:
    config = make_config(tmp_path)
    assert not panel_sign_in_required(config, store)
    store.create_account("alice", PASSWORD)
    assert not panel_sign_in_required(config, store), "loopback + owner: still the open panel"
    config.server.behind_proxy = True
    assert panel_sign_in_required(config, store)
    assert not panel_sign_in_required(config, None)


def test_panel_without_an_owner_stays_open_behind_a_proxy(tmp_path: Path, store: Any) -> None:
    config = make_config(tmp_path, server={"behind_proxy": True})
    assert not panel_sign_in_required(config, store)


def test_the_panel_gate_offers_the_owner_sign_in_behind_a_proxy(tmp_path: Path, store: Any) -> None:
    from studioforge.gui.app import COOKIE_NAME, LOGIN_PATH, create_gui_app
    from tests.unit.test_gui import _FakeState

    config = make_config(tmp_path, server={"behind_proxy": True})
    config.ensure_dirs()
    state = _FakeState(config)
    state.accounts = store  # type: ignore[attr-defined]
    html = {"accept": "text/html"}
    with TestClient(create_gui_app(config, api_state=state)) as client:
        assert client.get("/", headers=html, follow_redirects=False).status_code == 200
        store.create_account("alice", PASSWORD)
        gated = client.get("/", headers=html, follow_redirects=False)
        assert gated.status_code == 303 and gated.headers["location"] == LOGIN_PATH
        signed = client.post(
            LOGIN_PATH,
            data={"username": "alice", "credential": PASSWORD},
            follow_redirects=False,
        )
        assert signed.status_code == 303 and COOKIE_NAME in signed.headers.get("set-cookie", "")


def test_a_signed_in_viewer_may_change_the_box_behind_a_proxy(
    tmp_path: Path, store: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui import tabs

    config = make_config(tmp_path, server={"behind_proxy": True})
    ctx = tabs.GuiContext(config=config, api_state=SimpleNamespace(accounts=store))
    monkeypatch.setattr(tabs, "viewer_host", lambda: "127.0.0.1")
    assert not tabs.viewer_may_change_box(ctx), "no owner: nobody can sign in, so read-only"
    store.create_account("alice", PASSWORD)
    # Past the gate, which now demands the owner's sign-in, is the permission.
    assert tabs.viewer_may_change_box(ctx)


def test_the_doctor_names_the_panel_sign_in_not_an_unreachable_session(
    tmp_path: Path, store: Any
) -> None:
    from studioforge.core.doctor import DoctorReport, _check_auth

    store.create_account("alice", PASSWORD)
    report = DoctorReport()
    _check_auth(report, make_config(tmp_path), store)
    finding = next(f for f in report.findings if f.check_id == "auth.loopback_trust_owner_no_keys")
    assert "panel asks for the owner's sign-in" in finding.fix
    assert "owner session" not in finding.fix


# ---------------------------------------------------------------------------
# 4. The bind rule and the key floor, whatever the form or surface
# ---------------------------------------------------------------------------


def _admin(present: bool) -> Any:
    return SimpleNamespace(has_admin_credential=lambda: present)


def test_a_section_form_short_key_is_refused(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    server = dict(config.to_yaml_dict()["server"], api_key="short-key-1")
    with pytest.raises(ConfigError, match=str(MIN_API_KEY_LENGTH)):
        apply_overrides(config, {"server": server})


def test_an_unchanged_old_short_key_survives_a_section_round_trip(tmp_path: Path) -> None:
    config = make_config(tmp_path, server={"api_key": "old-key-12"})
    server = dict(config.to_yaml_dict()["server"], port=4321)
    updated = apply_overrides(config, {"server": server})
    assert updated.server.api_key == "old-key-12" and updated.server.port == 4321


def test_a_section_form_exposure_needs_an_admin_credential(tmp_path: Path) -> None:
    config = make_config(tmp_path, gui={"enabled": True})
    gui = dict(config.to_yaml_dict()["gui"], host="0.0.0.0")  # noqa: S104 - a config value
    with pytest.raises(AuthError) as caught:
        apply_overrides(config, {"gui": gui})
    assert caught.value.code == "auth_required_for_remote_bind"
    assert caught.value.details["exposed"] == [{"service": "the control panel", "host": "0.0.0.0"}]
    with pytest.raises(AuthError):
        apply_overrides(config, {"gui": gui}, accounts=_admin(False))
    assert apply_overrides(config, {"gui": gui}, accounts=_admin(True)).gui.host == "0.0.0.0"
    # A key set in the same edit is a credential too.
    both = apply_overrides(config, {"gui": gui, "server.api_key": GOOD_KEY})
    assert both.gui.host == "0.0.0.0"


def test_enabling_a_service_bound_to_the_network_is_an_exposure(tmp_path: Path) -> None:
    config = make_config(tmp_path, watchdog={"host": "0.0.0.0"})  # noqa: S104 - disabled
    with pytest.raises(AuthError, match="the watchdog"):
        apply_overrides(config, {"watchdog.enabled": True})


def test_an_already_exposed_install_can_still_be_edited(tmp_path: Path) -> None:
    config = make_config(tmp_path, server={"host": "0.0.0.0"})  # noqa: S104
    assert apply_overrides(config, {"models.default_ctx": 4096}).models.default_ctx == 4096
    assert apply_overrides(config, {"server.host": "192.0.2.10"}).server.host == "192.0.2.10"
    assert apply_overrides(config, {"server.host": "127.0.0.1"}).server.host == "127.0.0.1"


async def test_mcp_set_config_holds_the_bind_rule(tmp_path: Path) -> None:
    from studioforge.mcp.management import build_management_mcp

    app = make_app(tmp_path)
    server = build_management_mcp(app.state)
    refused = await call_tool(server, "set_config", updates={"server.host": "0.0.0.0"})
    assert refused["ok"] is False
    assert refused["error"]["code"] == "auth_required_for_remote_bind"
    assert app.state.config.server.host == "127.0.0.1"
    app.state.accounts.create_account("alice", PASSWORD)
    accepted = await call_tool(server, "set_config", updates={"server.host": "0.0.0.0"})
    assert accepted["ok"] is True and accepted["restart_required"] == ["server.host"]


def _watchdog_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path]:
    from studioforge.watchdog.server import Watchdog, build_watchdog_mcp

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path)
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    return build_watchdog_mcp(Watchdog(path)), path


async def test_watchdog_set_config_holds_the_bind_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, path = _watchdog_on(tmp_path, monkeypatch)
    refused = await call_tool(server, "set_config", updates={"server.host": "0.0.0.0"})
    assert refused["ok"] is False and "non-loopback" in refused["error"]["message"]
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["server"]["host"] == "127.0.0.1"
    # The watchdog never opens the database; the account store's marker is its evidence.
    (tmp_path / "data" / OWNER_EVER_FILE).write_text("an account existed", encoding="utf-8")
    accepted = await call_tool(server, "set_config", updates={"server.host": "0.0.0.0"})
    assert accepted["ok"] is True


async def test_watchdog_set_config_holds_the_key_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, path = _watchdog_on(tmp_path, monkeypatch)
    short = await call_tool(server, "set_config", updates={"server.api_key": "short-key-1"})
    assert short["ok"] is False and str(MIN_API_KEY_LENGTH) in short["error"]["message"]
    placeholder = await call_tool(
        server, "set_config", updates={"mcp.pin": REDACTED_PREFIX + "*" * 8}
    )
    assert placeholder["ok"] is False and "placeholder" in placeholder["error"]["message"]
    # The recovery path (a config.yaml that does not validate) keeps the key rules too.
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["models"]["default_ctx"] = -1
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    broken = await call_tool(server, "set_config", updates={"server.api_key": "short-key-1"})
    assert broken["ok"] is False and str(MIN_API_KEY_LENGTH) in broken["error"]["message"]
    fixed = await call_tool(server, "set_config", updates={"models.default_ctx": 4096})
    assert fixed["ok"] is True and fixed["repaired_invalid_config"] is True


def test_the_cli_wizard_exposure_needs_the_owner_first(tmp_path: Path) -> None:
    from studioforge.config import load_config
    from studioforge.setup_wizard import LiveServices

    config = make_config(tmp_path)
    config.ensure_dirs()
    config.save(tmp_path / "data" / "config.yaml")
    config.source_path = tmp_path / "data" / "config.yaml"
    live = LiveServices(config)
    with pytest.raises(AuthError):
        live.apply_exposure("lan")
    assert load_config(tmp_path / "data" / "config.yaml").server.host == "127.0.0.1"
    live.create_owner("alice", PASSWORD)
    assert live.apply_exposure("lan")["hosts"] == "0.0.0.0"


# ---------------------------------------------------------------------------
# 5. The watchdog's /health detail
# ---------------------------------------------------------------------------


class _Watchdog:
    def __init__(self, config: Config) -> None:
        self.config = config

    def load_config(self) -> tuple[Config, str | None]:
        return self.config, None

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "up",
            "summary": "1 child up",
            "children_total": 1,
            "children": [{"pid": 4242}],
        }


async def _inner(scope: dict[str, Any], receive: Any, send: Any) -> None:  # never reached
    raise AssertionError("health is answered by the wrapper")


async def _health(
    app: Any, *, client: tuple[str, int] | None = LOOPBACK, headers: dict[str, str] | None = None
) -> dict[str, Any]:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": "/health",
        "scheme": "http",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
    }
    if client is not None:
        scope["client"] = client
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return dict(json.loads(body))


async def test_watchdog_health_detail_needs_this_server_as_host(tmp_path: Path) -> None:
    from studioforge.watchdog.server import wrap_asgi

    config = make_config(tmp_path, server={"allowed_hosts": ["box.lan"]})
    app = wrap_asgi(_inner, _Watchdog(config), credentials=lambda: (None, PIN))
    assert "children" in await _health(app, headers={"Host": "127.0.0.1:1235"})
    assert "children" in await _health(app, headers={"Host": "box.lan:1235"})
    assert "children" in await _health(app, client=None, headers={"Host": "evil.test"})
    rebinding = await _health(app, headers={"Host": "evil.test:1235"})
    assert rebinding["redacted"] is True and "children" not in rebinding
    cross = await _health(
        app, headers={"Host": "127.0.0.1:1235", "Origin": "http://127.0.0.1:8188"}
    )
    assert cross["redacted"] is True
    # A credential still opens the detail whatever the Host.
    pinned = await _health(app, headers={"Host": "evil.test:1235", "X-MCP-Pin": PIN})
    assert "children" in pinned


async def test_watchdog_health_detail_behind_a_proxy_needs_a_credential(tmp_path: Path) -> None:
    from studioforge.watchdog.server import wrap_asgi

    config = make_config(tmp_path, server={"behind_proxy": True})
    app = wrap_asgi(_inner, _Watchdog(config), credentials=lambda: (None, PIN))
    proxied = await _health(app, headers={"Host": "127.0.0.1:1235"})
    assert proxied["redacted"] is True
    assert "children" in await _health(app, headers={"Host": "127.0.0.1:1235", "X-MCP-Pin": PIN})


# ---------------------------------------------------------------------------
# 7. allowed_hosts URL spellings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "host", "trusted"),
    [
        ("http://box.lan", "box.lan:1234", True),
        ("https://box.lan", "box.lan:8080", True),
        ("http://box.lan:1234", "box.lan:1234", True),
        ("http://box.lan:1234", "box.lan:8080", False),
        ("box.lan:1234", "box.lan:1234", True),
        ("box.lan:1234", "box.lan:80", False),
        ("box.lan", "box.lan:1234", True),
        ("[fd00::5]", "[fd00::5]:1234", True),
        ("[fd00::5]:1234", "[fd00::5]:8080", False),
    ],
)
def test_allowed_hosts_port_comes_from_the_url_not_a_colon(
    tmp_path: Path, entry: str, host: str, trusted: bool
) -> None:
    config = make_config(tmp_path, server={"allowed_hosts": [entry]})
    assert host_header_trusted(_req(host), config) is trusted
    assert host_names_this_server(host, allowed_hosts=[entry]) is trusted
