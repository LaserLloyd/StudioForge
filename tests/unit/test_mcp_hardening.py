"""MCP hardening (gate G4): annotations, scopes, Origin, both protocol revisions, audit.

The SDK's own client drives the real gateway app and the real watchdog wrapper
over httpx's ASGI transport (no sockets), so what is exercised is the deployed
path: HTTP auth middleware -> Origin gate -> streamable-HTTP transport -> the
``tools/call`` interceptor in ``studioforge.mcp_policy``.

Protocol revisions: the handshake-based ``2025-11-25`` (``mode="legacy"``) and the
stateless ``2026-07-28`` (a pinned modern mode, and ``auto`` which probes
``server/discover``) must both negotiate and both enforce the same policy.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp_types.version import HANDSHAKE_PROTOCOL_VERSIONS, MODERN_PROTOCOL_VERSIONS

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.mcp.management import build_management_mcp
from studioforge.mcp_policy import (
    GATEWAY_TOOLS,
    WATCHDOG_TOOLS,
    annotations_for,
    policy_for,
)
from studioforge.watchdog.server import Watchdog, build_watchdog_mcp, create_asgi_app

PIN = "12345678"  # scrub-ok: fixture
URL = "http://127.0.0.1:11234/mcp"

#: The tools that remove, rewrite or reserve things. They need an admin key.
DESTRUCTIVE_GATEWAY = {
    "delete_model",
    "set_config",
    "download_model",
    "reserve_gpus",
    "unload_model",
}
DESTRUCTIVE_WATCHDOG = {
    "restart_server",
    "kill_model",
    "nuke_all_models",
    "reclaim_orphan_engines",
    "rollback_update",
    "set_config",
}
PROTOCOLS = [("legacy", "2025-11-25"), ("2026-07-28", "2026-07-28"), ("auto", "2026-07-28")]


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
    base.update(overrides)
    return Config(**base)


@pytest.fixture
def app(tmp_path: Path) -> Any:
    config = make_config(tmp_path)
    return create_app(config, state=build_state(config), start_background=False)


@contextlib.asynccontextmanager
async def session(asgi: Any, mode: str = "legacy", **headers: str) -> AsyncIterator[Client]:
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=asgi), base_url="http://127.0.0.1:11234", headers=headers
    )
    try:
        async with Client(streamable_http_client(URL, http_client=http), mode=mode) as client:
            yield client
    finally:
        await http.aclose()


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def error_code(result: Any) -> str | None:
    body = result.structured_content or {}
    return (body.get("error") or {}).get("code") if isinstance(body, dict) else None


# ---------------------------------------------------------------------------
# annotations and the policy table
# ---------------------------------------------------------------------------


async def test_every_gateway_tool_is_classified_and_annotated(app: Any) -> None:
    server = build_management_mcp(app.state)
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    names = {t.name for t in tools}
    assert names == set(GATEWAY_TOOLS), (
        "a tool was added or removed without a row in mcp_policy.GATEWAY_TOOLS "
        f"(unclassified: {sorted(names - set(GATEWAY_TOOLS))}, stale: "
        f"{sorted(set(GATEWAY_TOOLS) - names)})"
    )
    for tool in tools:
        annotations = tool.annotations
        assert annotations is not None, tool.name
        assert annotations.read_only_hint is not None
        assert annotations.destructive_hint is not None
        assert annotations.idempotent_hint is not None
    destructive = {t.name for t in tools if t.annotations.destructive_hint}  # type: ignore[union-attr]
    assert destructive == DESTRUCTIVE_GATEWAY
    read_only = {t.name for t in tools if t.annotations.read_only_hint}  # type: ignore[union-attr]
    assert {"list_models", "server_status", "plan_load", "get_config"} <= read_only
    assert not read_only & destructive


async def test_every_watchdog_tool_is_classified_and_annotated(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    server = build_watchdog_mcp(Watchdog(path))
    async with Client(server) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == set(WATCHDOG_TOOLS)
    destructive = {t.name for t in tools if t.annotations.destructive_hint}  # type: ignore[union-attr]
    assert destructive == DESTRUCTIVE_WATCHDOG
    for name in ("health", "tail_logs", "gpu_status", "get_config"):
        assert policy_for("watchdog", name).kind == "read"


def test_an_unclassified_tool_fails_closed() -> None:
    row = policy_for("gateway", "some_future_tool")
    assert (row.kind, row.scope) == ("destructive", "admin")
    hints = annotations_for("gateway", "some_future_tool")
    assert hints.destructive_hint is True and hints.read_only_hint is False


def test_policy_scopes_are_ordered_read_inference_admin() -> None:
    assert policy_for("gateway", "list_models").scope == "read"
    assert policy_for("gateway", "load_model").scope == "inference"
    assert policy_for("gateway", "pin_model").scope == "admin"
    assert policy_for("gateway", "delete_model").scope == "admin"
    assert policy_for("gateway", "load_recommended").admin_arg == "persist"


def test_the_sdk_supports_both_revisions_the_server_must_speak() -> None:
    assert "2025-11-25" in HANDSHAKE_PROTOCOL_VERSIONS
    assert "2026-07-28" in MODERN_PROTOCOL_VERSIONS


# ---------------------------------------------------------------------------
# both protocol revisions, through the gateway
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("mode", "negotiated"), PROTOCOLS)
async def test_the_gateway_speaks_both_protocol_revisions(
    app: Any, mode: str, negotiated: str
) -> None:
    _record, admin = app.state.accounts.create_key("root", "admin")
    async with app.router.lifespan_context(app), session(app, mode, **bearer(admin)) as client:
        assert client.session.protocol_version == negotiated
        listed = await client.list_tools()
        assert {t.name for t in listed.tools} == set(GATEWAY_TOOLS)
        status = await client.call_tool("server_status", {})
        assert status.is_error is False


@pytest.mark.parametrize(("mode", "_negotiated"), PROTOCOLS)
async def test_a_read_key_is_blocked_from_destructive_tools(
    app: Any, mode: str, _negotiated: str
) -> None:
    _r, read = app.state.accounts.create_key("ro", "read")
    async with app.router.lifespan_context(app), session(app, mode, **bearer(read)) as client:
        listing = await client.call_tool("list_models", {})
        assert listing.is_error is False
        for tool, args in (
            ("delete_model", {"model_id": "x", "confirm": True}),
            ("unload_model", {"model_id": "x"}),
            ("set_config", {"updates": {"models.default_ctx": 4096}}),
            ("download_model", {"repo_id": "a/b"}),
            ("reserve_gpus", {"devices": [0], "holder": "h"}),
        ):
            denied = await client.call_tool(tool, args)
            assert denied.is_error is True, tool
            assert error_code(denied) == "permission_denied", tool
            details = denied.structured_content["error"]["details"]
            assert details["needed"] == "admin" and details["scope"] == "read"
        # a state-changing but non-destructive tool needs inference, which read lacks
        load = await client.call_tool("load_model", {"model_id": "x"})
        assert error_code(load) == "permission_denied"
        assert load.structured_content["error"]["details"]["needed"] == "inference"


async def test_an_inference_key_may_load_but_not_destroy(app: Any) -> None:
    _r, key = app.state.accounts.create_key("agent", "inference")
    async with app.router.lifespan_context(app), session(app, "legacy", **bearer(key)) as client:
        load = await client.call_tool("load_model", {"model_id": "no-such-model"})
        assert error_code(load) == "model_not_found"  # authorised; the model does not exist
        denied = await client.call_tool("unload_model", {"model_id": "no-such-model"})
        assert error_code(denied) == "permission_denied"
        pin = await client.call_tool("pin_model", {"model_id": "no-such-model"})
        assert error_code(pin) == "permission_denied"  # pinning outlives the instance: admin
        persist = await client.call_tool(
            "load_recommended", {"model_id": "x", "ctx_size": 4096, "persist": True}
        )
        assert error_code(persist) == "permission_denied"
        plain = await client.call_tool("load_recommended", {"model_id": "x", "ctx_size": 4096})
        assert error_code(plain) != "permission_denied"


@pytest.mark.parametrize(("mode", "_negotiated"), PROTOCOLS)
async def test_an_admin_key_may_call_destructive_tools(
    app: Any, mode: str, _negotiated: str
) -> None:
    _r, admin = app.state.accounts.create_key("root", "admin")
    async with app.router.lifespan_context(app), session(app, mode, **bearer(admin)) as client:
        done = await client.call_tool("unload_model", {"model_id": "not-loaded"})
        assert done.is_error is False and done.structured_content["unloaded"] is False
        changed = await client.call_tool("set_config", {"updates": {"models.default_ctx": 4096}})
        assert changed.structured_content["ok"] is True


async def test_the_pin_is_still_an_admin_credential_on_mcp(app: Any) -> None:
    async with (
        app.router.lifespan_context(app),
        session(app, "legacy", **{"X-MCP-Pin": PIN}) as client,
    ):
        done = await client.call_tool("unload_model", {"model_id": "not-loaded"})
        assert done.is_error is False


async def test_no_credential_is_a_401_once_keys_exist(app: Any) -> None:
    app.state.accounts.create_key("root", "admin")
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://127.0.0.1:11234"
    )
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        async with app.router.lifespan_context(app):
            refused = await http.post(URL, json=body, headers=headers)
            assert refused.status_code == 401
            # the PIN is the only other way in, and it is deprecated but accepted
            assert refused.json()["error"]["code"] == "invalid_mcp_pin"
    finally:
        await http.aclose()


async def test_in_process_calls_are_the_local_process_and_bypass_the_policy(app: Any) -> None:
    server = build_management_mcp(app.state)
    result = await server.call_tool("unload_model", {"model_id": "not-loaded"})
    assert result.is_error is False


async def test_mutating_calls_are_audited_reads_are_not(app: Any) -> None:
    admin_record, admin = app.state.accounts.create_key("root", "admin")
    _r, read = app.state.accounts.create_key("ro", "read")
    secret = "hf_" + "b" * 30
    async with app.router.lifespan_context(app):
        async with session(app, "legacy", **bearer(admin)) as client:
            await client.call_tool("list_models", {})
            await client.call_tool("unload_model", {"model_id": "not-loaded"})
            await client.call_tool("set_config", {"updates": {"hf.token": secret}})
        async with session(app, "legacy", **bearer(read)) as client:
            await client.call_tool("delete_model", {"model_id": "x", "confirm": True})
    entries = app.state.audit.tail(50)
    rows = [(e["action"], e["result"], e["actor"]) for e in entries]
    assert ("mcp.unload_model", "ok", f"key:{admin_record.id}") in rows
    assert any(a == "mcp.set_config" and r == "ok" for a, r, _ in rows)
    assert any(
        a == "mcp.delete_model" and r == "denied" and who.startswith("key:") for a, r, who in rows
    )
    assert not any(a == "mcp.list_models" for a, _, _ in rows)
    text = json.dumps(entries)
    assert secret not in text and "hf.token" in text  # the key name, never the value
    assert all(e["surface"] == "mcp:gateway" for e in entries)


# ---------------------------------------------------------------------------
# Origin validation
# ---------------------------------------------------------------------------


async def test_the_gateway_refuses_a_foreign_origin_even_with_a_valid_key(app: Any) -> None:
    _r, admin = app.state.accounts.create_key("root", "admin")
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=app), base_url="http://127.0.0.1:11234"
    )
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    headers = {**bearer(admin), "Content-Type": "application/json", "Accept": "application/json"}
    try:
        async with app.router.lifespan_context(app):
            bad = await http.post(URL, json=body, headers={**headers, "Origin": "http://evil.test"})
            assert bad.status_code == 403
            assert bad.json()["error"]["code"] == "origin_not_allowed"
            good = await http.post(URL, json=body, headers=headers)
            assert good.status_code != 403
    finally:
        await http.aclose()


# ---------------------------------------------------------------------------
# the watchdog
# ---------------------------------------------------------------------------


@pytest.fixture
def watchdog_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Watchdog, Any, Any, Any]:
    # The watchdog resolves its data directory like the server: SF_DATA_DIR wins.
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path)
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    watchdog = Watchdog(path)
    asgi, server = create_asgi_app(watchdog)
    from studioforge.core.accounts import AccountStore
    from studioforge.db import Database

    db = Database(config.db_path)
    db.migrate()
    return watchdog, asgi, server, AccountStore(db, config.data_dir)


@pytest.mark.parametrize(("mode", "_negotiated"), PROTOCOLS)
async def test_watchdog_scopes_over_both_protocol_revisions(
    watchdog_env: tuple[Watchdog, Any, Any, Any], mode: str, _negotiated: str
) -> None:
    _watchdog, asgi, server, accounts = watchdog_env
    _r, read = accounts.create_key("ro", "read")
    _a, admin = accounts.create_key("root", "admin")
    async with server.session_manager.run():
        async with session(asgi, mode, **bearer(read)) as client:
            health = await client.call_tool("health", {})
            assert error_code(health) is None
            for tool, args in (
                ("nuke_all_models", {"confirm": False}),
                ("restart_server", {"confirm": False}),
                ("reclaim_orphan_engines", {"dry_run": True}),
                ("rollback_update", {"confirm": False}),
                ("kill_model", {"model_name": "nothing"}),
                ("set_config", {"updates": {"models.default_ctx": 4096}}),
            ):
                denied = await client.call_tool(tool, args)
                assert error_code(denied) == "permission_denied", tool
        async with session(asgi, mode, **bearer(admin)) as client:
            preview = await client.call_tool("nuke_all_models", {"confirm": False})
            assert error_code(preview) != "permission_denied" and preview.is_error is False
            preview = await client.call_tool("restart_server", {"confirm": False})
            assert error_code(preview) != "permission_denied"


async def test_watchdog_refuses_a_foreign_origin_and_a_read_key_on_restart(
    watchdog_env: tuple[Watchdog, Any, Any, Any],
) -> None:
    _watchdog, asgi, server, accounts = watchdog_env
    _r, read = accounts.create_key("ro", "read")
    _a, admin = accounts.create_key("root", "admin")
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=asgi), base_url="http://127.0.0.1:11235"
    )
    try:
        async with server.session_manager.run():
            for path in ("/mcp", "/restart"):
                bad = await http.post(
                    path, headers={**bearer(admin), "Origin": "http://evil.test"}, json={}
                )
                assert bad.status_code == 403, path
                assert bad.json()["error"]["code"] == "origin_not_allowed"
            denied = await http.post("/restart", headers=bearer(read))
            assert denied.status_code == 403
            assert denied.json()["error"]["code"] == "permission_denied"
            nobody = await http.post("/restart")
            assert nobody.status_code == 401
            assert (await http.get("/health")).status_code in (200, 503)  # liveness stays open
    finally:
        await http.aclose()


async def test_watchdog_a_read_key_does_not_reset_the_pin_lockout(
    watchdog_env: tuple[Watchdog, Any, Any, Any],
) -> None:
    from studioforge.watchdog import server as watchdog_server

    _watchdog, asgi, _server, accounts = watchdog_env
    _r, read = accounts.create_key("ro", "read")
    guard = watchdog_server._CREDENTIAL_GUARD
    guard.reset()
    http = httpx2.AsyncClient(
        transport=httpx2.ASGITransport(app=asgi, client=("192.168.1.50", 50000)),
        base_url="http://127.0.0.1:11235",
    )
    wrong = {"X-MCP-Pin": "00000000"}
    try:
        for _ in range(3):
            assert (await http.post("/restart", headers=wrong)).status_code == 401
        assert (await http.post("/restart", headers=bearer(read))).status_code == 403  # genuine
        assert (await http.post("/restart", headers=wrong)).status_code == 401  # arms lockout
        assert (await http.post("/restart", headers=wrong)).status_code == 429
    finally:
        guard.reset()
        await http.aclose()


async def test_watchdog_audits_denials_and_the_restart_route(
    watchdog_env: tuple[Watchdog, Any, Any, Any],
) -> None:
    watchdog, asgi, server, accounts = watchdog_env
    _r, read = accounts.create_key("ro", "read")
    async with server.session_manager.run():
        async with session(asgi, "legacy", **bearer(read)) as client:
            await client.call_tool("nuke_all_models", {"confirm": False})
        http = httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=asgi), base_url="http://127.0.0.1:11235"
        )
        try:
            await http.post("/restart", headers=bearer(read))
        finally:
            await http.aclose()
    config, _ = watchdog.load_config()
    lines = [
        json.loads(line)
        for line in (config.logs_dir / "audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    pairs = {(e["action"], e["result"]) for e in lines}
    assert ("mcp.nuke_all_models", "denied") in pairs
    assert ("watchdog.restart", "denied") in pairs
    assert all(e["surface"].startswith(("mcp:", "watchdog")) for e in lines)


async def test_watchdog_write_config_is_owner_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.secure_files import is_owner_only

    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    config = make_config(tmp_path)
    config.ensure_dirs()
    path = config.save(tmp_path / "data" / "config.yaml")
    watchdog = Watchdog(path)
    loaded, _ = watchdog.load_config()
    watchdog.write_config(loaded)
    assert is_owner_only(path) is True
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["server"]["port"] == 11234
