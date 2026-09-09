"""Hardening (D55): the findings of the 2026-09-04 audit, pinned one by one.

The panel's control channel is gated by path, not transport, and every GUI
response refuses to be framed; a model a standing lease holds is not unloaded
by a stranger; the child's launch line is redacted everywhere it is written;
the open reads (logs, VRAM holders) shape their answer by who is asking; a 500
names a reference, not the exception; the watchdog fails CLOSED with no
credential and shows a remote poller the verdict only; ``kill_model`` refuses
to guess; the panel's session secret is random; and the two panel actions that
touch the operator's desktop take the D32 rule.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.config import Config
from studioforge.core import supervisor as supervisor_mod
from studioforge.core.priority import PRIORITY_AGENT
from studioforge.errors import LeaseConflictError, ModelLoadError
from studioforge.gui.app import (
    _FRAME_HEADERS,
    GUI_SECRET_FILE,
    NICEGUI_SOCKET_PREFIX,
    GuiAuthGate,
    _storage_secret,
)
from studioforge.logging import RING_BUFFER
from studioforge.types import InstanceInfo, LoadPlan
from studioforge.watchdog.server import _public_health, wrap_asgi
from tests.unit.test_catalog_routes import MODEL_ID, FakeSupervisor
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export
from tests.unit.test_gateway_lifecycle import make_manager, make_record, placed
from tests.unit.test_lease_vacate import HOLDER_URL, Recorder, _settle

LAN = ("192.168.1.50", 5000)
LOOPBACK = ("127.0.0.1", 50000)
PIN = "87654321"
#: A synthetic Windows profile for the redaction fixtures -- no real user.
PROFILE = "C:\\Users\\operator"  # scrub-ok: fixture


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.models.dir = tmp_path / "models"
    cfg.models.dir.mkdir(parents=True, exist_ok=True)
    cfg.ensure_dirs()
    return cfg


# ---------------------------------------------------------------------------
# A-1: the panel's control channel and the frame headers
# ---------------------------------------------------------------------------


async def _drive_gate(
    config: Config, scope: dict[str, Any], *, inner_extra: list[tuple[bytes, bytes]] | None = None
) -> tuple[bool, list[dict[str, Any]], list[tuple[bytes, bytes]]]:
    """Run ``scope`` through the gate; returns (inner reached, messages sent, inner's headers)."""
    reached = {"inner": False}
    inner_headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"text/html"),
        *(inner_extra or []),
    ]

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        reached["inner"] = True
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": inner_headers})
            await send({"type": "http.response.body", "body": b"<html></html>"})

    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await GuiAuthGate(inner, config)(scope, receive, send)
    return reached["inner"], sent, inner_headers


def _http_scope(
    path: str, *, origin: str | None, host: str = "192.168.1.50:8080"
) -> dict[str, Any]:
    headers = [(b"host", host.encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    return {"type": "http", "method": "GET", "path": path, "headers": headers}


def _response_headers(sent: list[dict[str, Any]]) -> dict[str, bytes]:
    start = next(m for m in sent if m["type"] == "http.response.start")
    return {name.lower(): value for name, value in start["headers"]}


async def test_a_cross_site_page_cannot_drive_the_panel_over_long_polling(config: Config) -> None:
    """The bypass: socket.io's first transport is HTTP polling, not a WebSocket,
    and the old gate keyed on ``type == "websocket"``. The refusal carries no
    Access-Control-Allow-Origin, so the attacking page cannot even read it."""
    config.server.api_key = None
    scope = _http_scope(
        NICEGUI_SOCKET_PREFIX + "socket.io/?EIO=4&transport=polling",
        origin="https://evil.example",
    )
    reached, sent, _ = await _drive_gate(config, scope)
    assert reached is False
    headers = _response_headers(sent)
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 403
    assert b"access-control-allow-origin" not in headers
    body = json.loads(
        b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    )
    assert body["error"]["code"] == "cross_site_control_channel"


async def test_same_origin_polling_and_non_browser_clients_pass(config: Config) -> None:
    config.server.api_key = None
    same = _http_scope(NICEGUI_SOCKET_PREFIX + "socket.io/", origin="http://192.168.1.50:8080")
    reached, _, _ = await _drive_gate(config, same)
    assert reached is True
    no_origin = _http_scope(NICEGUI_SOCKET_PREFIX + "socket.io/", origin=None)
    reached, _, _ = await _drive_gate(config, no_origin)
    assert reached is True, "a non-browser client sends no Origin and is not the threat"


async def test_a_foreign_origin_on_an_ordinary_page_is_not_the_control_channel(
    config: Config,
) -> None:
    """The origin rule is for the channel that presses buttons. A plain page
    fetch with a foreign Origin is answered (the browser's own same-origin
    policy keeps the page from reading it) -- but never framed."""
    config.server.api_key = None
    reached, sent, _ = await _drive_gate(config, _http_scope("/", origin="https://evil.example"))
    assert reached is True
    headers = _response_headers(sent)
    assert headers[b"x-frame-options"] == b"DENY"
    assert headers[b"content-security-policy"] == b"frame-ancestors 'none'"


async def test_every_gui_response_carries_the_frame_headers_once(config: Config) -> None:
    config.server.api_key = None
    reached, sent, inner_headers = await _drive_gate(config, _http_scope("/", origin=None))
    assert reached is True
    start = next(m for m in sent if m["type"] == "http.response.start")
    names = [name.lower() for name, _ in start["headers"]]
    for name, _ in _FRAME_HEADERS:
        assert names.count(name) == 1
    # An app that already set one keeps its own value; nothing is doubled.
    reached, sent, _ = await _drive_gate(
        config, _http_scope("/", origin=None), inner_extra=[(b"X-Frame-Options", b"SAMEORIGIN")]
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    values = [value for name, value in start["headers"] if name.lower() == b"x-frame-options"]
    assert values == [b"SAMEORIGIN"]


# ---------------------------------------------------------------------------
# A-6: the session-signing secret
# ---------------------------------------------------------------------------


def test_the_gui_session_secret_is_random_persisted_and_not_derived_from_the_data_dir(
    config: Config,
) -> None:
    import hashlib

    first = _storage_secret(config)
    assert len(first) >= 64 and re.fullmatch(r"[0-9a-f]+", first)
    derived = hashlib.sha256(f"studioforge-gui::{config.data_dir}".encode()).hexdigest()
    assert first != derived, (
        "the watchdog publishes the data dir; a secret derived from it is public"
    )
    path = config.data_dir / GUI_SECRET_FILE
    assert path.read_text(encoding="utf-8").strip() == first
    assert _storage_secret(config) == first, "sessions survive a restart"
    path.write_text("short", encoding="utf-8")
    assert _storage_secret(config) != "short", "a truncated file is replaced, not trusted"


# ---------------------------------------------------------------------------
# F-3: a model a standing lease holds is not unloaded by a stranger
# ---------------------------------------------------------------------------


async def test_unload_of_a_lease_held_model_is_refused_unless_forced_or_housekeeping() -> None:
    record = make_record("bench/model")
    manager, supervisor = make_manager([record])
    lease = await manager.acquire_lease([1], holder="crucibleforge", model_ids=[record.id])
    supervisor.instances[record.id] = placed(record.id, [1])

    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.unload(record.id)
    err = excinfo.value
    assert err.status_code == 409 and err.code == "lease_conflict"
    assert [row["id"] for row in err.details["leases"]] == [lease.id]
    assert "X-SF-Client" in err.message and "DELETE /api/leases" in err.message
    assert record.id in supervisor.instances, "refused means untouched"

    # Housekeeping -- the TTL sweep, a benchmark's own teardown, the D46
    # restore -- is never guarded: it only ever puts back what it took.
    assert await manager.unload(record.id, deliberate=False) is True
    supervisor.instances[record.id] = placed(record.id, [1])
    # The holder or an admin waives it with force.
    assert await manager.unload(record.id, force=True) is True
    assert record.id not in supervisor.instances


async def test_a_squatter_on_a_leased_card_is_held_too_and_a_free_model_is_not() -> None:
    """The lease names nothing; a resident on its cards is still its business."""
    held = make_record("squatter/model")
    free = make_record("free/model")
    manager, supervisor = make_manager([held, free])
    await manager.acquire_lease([2, 3], holder="crucibleforge")
    supervisor.instances[held.id] = placed(held.id, [3])
    supervisor.instances[free.id] = placed(free.id, [0])

    assert [lease.holder for lease in manager.leases_holding([held.id])] == ["crucibleforge"]
    assert manager.leases_holding([free.id]) == []
    with pytest.raises(LeaseConflictError):
        await manager.unload(held.id)
    assert await manager.unload(free.id) is True, "no lease in the way: open, as before"


async def test_unload_all_is_refused_wholesale_while_any_resident_is_held() -> None:
    held = make_record("held/model")
    free = make_record("free/model")
    manager, supervisor = make_manager([held, free])
    supervisor.instances[free.id] = placed(free.id, [0])
    assert await manager.unload_all() == [free.id], "nothing leased: unload-all is open"

    await manager.acquire_lease([1], holder="benchmark", model_ids=[held.id])
    supervisor.instances[held.id] = placed(held.id, [1])
    supervisor.instances[free.id] = placed(free.id, [0])
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.unload_all()
    assert excinfo.value.code == "lease_conflict"
    assert set(supervisor.instances) == {held.id, free.id}, "a partial unload-all is worse"
    assert sorted(await manager.unload_all(force=True)) == sorted([held.id, free.id])


async def test_the_unload_guard_leaves_the_lease_grants_own_eviction_alone() -> None:
    """D55 composes with D43/D56: the grant evicts an idle resident through the
    supervisor, not through ``unload``, so a lease already standing on other
    cards never turns a fresh grant's eviction into a 409."""
    owner = make_record("owner/model")
    idle = make_record("idle/model")
    manager, supervisor = make_manager([owner, idle])
    standing = await manager.acquire_lease([1], holder="crucibleforge", model_ids=[owner.id])
    supervisor.instances[owner.id] = placed(owner.id, [1])
    supervisor.instances[idle.id] = placed(idle.id, [0])

    granted = await manager.acquire_lease([0], holder="benchmark", priority=PRIORITY_AGENT)
    assert idle.id not in supervisor.instances, "the idle resident was evicted for the grant"
    assert owner.id in supervisor.instances, "the other lease's owner was not touched"
    assert {lease.id for lease in manager.leases.all()} == {standing.id, granted.id}


async def test_the_unload_guard_composes_with_a_vacate_release() -> None:
    """A tenant asked to vacate (D56) releases its lease; from then on its model
    is nobody's to protect and the plain unload goes through."""
    tenant_model = make_record("render/model")
    manager, supervisor = make_manager([tenant_model])
    manager._vacate_sender = Recorder()
    tenant = await manager.acquire_lease(
        [2], holder="clawforge2", model_ids=[tenant_model.id], vacate_url=HOLDER_URL
    )
    supervisor.instances[tenant_model.id] = placed(tenant_model.id, [2])

    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2], holder="crucibleforge", priority=PRIORITY_AGENT)
    assert excinfo.value.code == "lease_vacating"
    with pytest.raises(LeaseConflictError) as still:
        await manager.unload(tenant_model.id)
    assert still.value.code == "lease_conflict", "vacating is not released: still held"
    await _settle()

    manager.release_lease(tenant.id)
    assert await manager.unload(tenant_model.id) is True
    granted = await manager.acquire_lease([2], holder="crucibleforge", priority=PRIORITY_AGENT)
    assert granted.devices == [2]


class _StoppableSupervisor(FakeSupervisor):
    """The catalog fake plus the three calls the unload and log routes make."""

    def __init__(self, instances: list[InstanceInfo], *, log_dir: Path) -> None:
        super().__init__(instances)
        self.log_dir = log_dir

    async def stop(self, model_id: str, **_kwargs: Any) -> None:
        self._instances = [i for i in self._instances if i.model_id != model_id]

    async def stop_all(self, **_kwargs: Any) -> dict[str, BaseException | None]:
        outcome: dict[str, BaseException | None] = {i.model_id: None for i in self._instances}
        self._instances = []
        return outcome

    def log_path(self, model_id: str) -> Path | None:
        return self.log_dir / "models" / f"{model_id.replace('/', '__')}.log"

    def tail_log(self, model_id: str, n: int = 200) -> list[str]:
        return [
            f"=== studioforge launch: llama-server.exe --model {model_id}.gguf --port 41000",
            f"load: reading {self.log_dir / 'models' / 'thing.gguf'}",
            "srv  load_model: loaded",
        ]


def _install(app: Any, instances: list[InstanceInfo]) -> _StoppableSupervisor:
    supervisor = _StoppableSupervisor(
        instances,
        log_dir=Path("C:/Users/operator/Desktop/_UserData/StudioForge/logs"),  # scrub-ok: fixture
    )
    app.state.supervisor = supervisor
    app.state.manager.supervisor = supervisor
    return supervisor


def _resident(model_id: str, devices: list[int]) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        ttl_s=1800,
        plan=LoadPlan(model_id=model_id, devices=devices, ctx_size=8192, ctx_per_slot=8192),
    )


def test_the_open_unload_routes_refuse_a_stranger_but_not_the_holder_or_an_admin(
    app: Any,
) -> None:
    supervisor = _install(app, [])
    local = TestClient(app, client=LOOPBACK)
    lease = local.post(
        "/api/leases",
        json={"devices": [0], "holder": "crucibleforge", "model_ids": [MODEL_ID]},
    ).json()
    assert lease["holder"] == "crucibleforge"
    supervisor._instances = [_resident(MODEL_ID, [0])]

    stranger = TestClient(app, client=LAN)
    refused = stranger.post(f"/api/models/{MODEL_ID}/unload")
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "lease_conflict"
    assert [row["id"] for row in refused.json()["error"]["studioforge"]["leases"]] == [lease["id"]]
    assert stranger.post("/api/models/unload-all").status_code == 409
    assert stranger.post(f"/api/models/{MODEL_ID}/restart").status_code == 409
    assert supervisor.get(MODEL_ID) is not None, "nothing was taken apart"

    # The label alone is NOT the holder (round-2 review): the lease was taken
    # from this machine, so a LAN peer that merely spells the holder's name
    # is still a stranger -- the first cut of D55 let it through.
    spoof = stranger.post(
        f"/api/models/{MODEL_ID}/unload", headers={"X-SF-Client": "crucibleforge-judge"}
    )
    assert spoof.status_code == 409 and supervisor.get(MODEL_ID) is not None
    assert "X-SF-Vacate-Token" in spoof.json()["error"]["message"]

    admin = TestClient(app, client=LOOPBACK)
    assert admin.post("/api/models/unload-all").json()["unloaded"] == [MODEL_ID]


def test_the_holder_waiver_needs_the_registering_peer_or_the_leases_own_token(
    app: Any,
) -> None:
    """The remote holder (CrucibleForge on another box, sending the PIN to
    take the lease) may still unload its own benchmark model from the open
    route -- from the address it leased from, or with the ``vacate_token`` it
    registered. A third host with the same label, and nothing else, may not:
    the family rule makes ``crucibleforge-judge`` one client, not one
    credential."""
    supervisor = _install(app, [])
    app.state.config.mcp.pin = PIN
    benchbox = TestClient(app, client=("10.9.0.2", 5000))
    lease = benchbox.post(
        "/api/leases",
        json={
            "devices": [0],
            "holder": "crucibleforge",
            "model_ids": [MODEL_ID],
            "vacate_url": HOLDER_URL,
            "vacate_token": "cf-secret-0123456789",  # scrub-ok: fixture bearer
        },
        headers={"X-MCP-Pin": PIN},
    ).json()
    assert lease["holder"] == "crucibleforge", lease
    record = app.state.manager.leases.get(lease["id"])
    assert record.holder_peer == "10.9.0.2"

    # A third host with the label: refused, nothing taken apart.
    supervisor._instances = [_resident(MODEL_ID, [0])]
    elsewhere = TestClient(app, client=("10.9.0.3", 5000))
    label_only = {"X-SF-Client": "crucibleforge-judge"}
    assert elsewhere.post(f"/api/models/{MODEL_ID}/unload", headers=label_only).status_code == 409
    assert elsewhere.post("/api/models/unload-all", headers=label_only).status_code == 409
    assert elsewhere.post(f"/api/models/{MODEL_ID}/restart", headers=label_only).status_code == 409
    # ...and no label at all, from the registering peer, is not the holder either.
    assert benchbox.post(f"/api/models/{MODEL_ID}/unload").status_code == 409
    assert supervisor.get(MODEL_ID) is not None

    # The registering peer with the label: the holder.
    ok = benchbox.post(f"/api/models/{MODEL_ID}/unload", headers=label_only)
    assert ok.status_code == 200 and ok.json()["unloaded"] is True

    # A third host presenting the lease's own token: proof enough.
    supervisor._instances = [_resident(MODEL_ID, [0])]
    wrong_token = {**label_only, "X-SF-Vacate-Token": "not-it"}
    assert elsewhere.post(f"/api/models/{MODEL_ID}/unload", headers=wrong_token).status_code == 409
    right_token = {**label_only, "X-SF-Vacate-Token": "cf-secret-0123456789"}  # scrub-ok: fixture
    assert elsewhere.post(f"/api/models/{MODEL_ID}/unload", headers=right_token).status_code == 200

    # The PIN alone is the admin rule, label or not.
    supervisor._instances = [_resident(MODEL_ID, [0])]
    with_pin = elsewhere.post(f"/api/models/{MODEL_ID}/unload", headers={"X-MCP-Pin": PIN})
    assert with_pin.status_code == 200

    # The peer never leaks: not in the book's views, not in the refusal.
    supervisor._instances = [_resident(MODEL_ID, [0])]
    refused = elsewhere.post(f"/api/models/{MODEL_ID}/unload", headers=label_only)
    assert "10.9.0.2" not in refused.text
    assert "10.9.0.2" not in benchbox.get("/api/leases", headers={"X-MCP-Pin": PIN}).text


def test_an_unleased_model_unloads_from_the_lan_exactly_as_before(app: Any) -> None:
    supervisor = _install(app, [_resident(MODEL_ID, [0])])
    stranger = TestClient(app, client=LAN)
    reply = stranger.post(f"/api/models/{MODEL_ID}/unload")
    assert reply.status_code == 200 and reply.json()["unloaded"] is True
    assert supervisor.list() == []


# ---------------------------------------------------------------------------
# F-1a: the launch line is redacted at every point it is written
# ---------------------------------------------------------------------------


async def test_the_child_log_header_and_the_spawn_line_carry_the_redacted_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.test_supervisor import FAKE_CHILD, make_plan, resolver
    from tests.unit.test_supervisor import fake_binary as _fake_binary  # noqa: F401
    from tests.unit.test_supervisor import make_record as sup_record

    config = Config(data_dir=tmp_path / "data")
    config.gateway.child_port_start = 47500
    config.gateway.child_port_end = 47510
    config.gateway.load_timeout_s = 20.0
    config.gateway.health_poll_interval_s = 0.05
    config.ensure_dirs()
    binary = tmp_path / "engine" / "fake_llama_server.py"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(FAKE_CHILD, encoding="utf-8")

    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def _record(self, event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

        debug = info = warning = error = exception = _record

    monkeypatch.setattr(supervisor_mod, "log", _Log())
    supervisor = supervisor_mod.Supervisor(
        config, resolve_binary=resolver(binary), launch_prefix=[sys.executable, "-u"]
    )
    record = sup_record(tmp_path)
    try:
        info = await supervisor.start(record, make_plan())
        log_path = supervisor.log_path(record.id)
        assert log_path is not None
        header = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        assert header.startswith("=== studioforge launch:")
        assert record.path.name in header and binary.name in header
        assert str(record.path) not in header and str(binary) not in header
        assert not any(Path(token).is_absolute() for token in header.split()[3:])
        spawn = next(kwargs for event, kwargs in events if event == "model_spawn")
        assert spawn["argv"] == " ".join(info.launch_args or [])
        assert str(record.path) not in spawn["argv"]
    finally:
        await supervisor.aclose()


async def test_a_failed_launch_reports_the_redacted_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.test_supervisor import make_plan, resolver
    from tests.unit.test_supervisor import make_record as sup_record

    config = Config(data_dir=tmp_path / "data")
    config.gateway.child_port_start = 47520
    config.gateway.child_port_end = 47530
    config.ensure_dirs()
    binary = tmp_path / "engines" / "b1" / "llama-server.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_bytes(b"")

    async def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError(13, "permission denied")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    supervisor = supervisor_mod.Supervisor(config, resolve_binary=resolver(binary))
    record = sup_record(tmp_path)
    try:
        with pytest.raises(ModelLoadError) as excinfo:
            await supervisor.start(record, make_plan())
    finally:
        await supervisor.aclose()
    argv = excinfo.value.details["argv"]
    assert record.path.name in argv
    assert not any(Path(token).is_absolute() for token in argv)


# ---------------------------------------------------------------------------
# F-1b / F-2: the open reads answer by who is asking
# ---------------------------------------------------------------------------


def test_redact_paths_reduces_absolute_paths_and_leaves_routes_alone() -> None:
    from studioforge.api.mgmt_routes import _redact_paths

    line = (
        "loaded C:\\Users\\operator\\models\\Qwen3-30B-Q5_K_M.gguf from "  # scrub-ok: fixture
        "/home/operator/.cache/x.bin via GET /api/models/vendor/Some-Model "  # scrub-ok: fixture
        "(\\\\nas\\share\\m.gguf)"
    )
    out = _redact_paths(line)
    assert "operator" not in out
    assert "Qwen3-30B-Q5_K_M.gguf" in out and "x.bin" in out and "m.gguf" in out
    assert "/api/models/vendor/Some-Model" in out, "a URL path is not a filesystem path"


def test_logs_are_path_redacted_for_a_lan_reader_and_verbatim_for_an_admin(app: Any) -> None:
    import logging

    _install(app, [])
    RING_BUFFER.emit(
        logging.LogRecord(
            "studioforge.test",
            logging.INFO,
            __file__,
            1,
            f"model_spawn model={PROFILE}\\models\\thing.gguf port=41000",
            (),
            None,
        )
    )
    stranger = TestClient(app, client=LAN)
    ring = stranger.get("/api/logs", params={"n": 5}).json()
    assert ring["redacted"] is True
    assert all("operator" not in json.dumps(line) for line in ring["lines"])
    assert any("thing.gguf" in line["message"] for line in ring["lines"])
    child = stranger.get(f"/api/logs/models/{MODEL_ID}").json()
    assert child["redacted"] is True
    assert child["path"] == f"{MODEL_ID.replace('/', '__')}.log"
    assert all("operator" not in line for line in child["lines"])
    assert "srv  load_model: loaded" in child["lines"]
    admin = TestClient(app, client=LOOPBACK)
    ring = admin.get("/api/logs", params={"n": 5}).json()
    assert ring["redacted"] is False
    assert any(PROFILE in line["message"] for line in ring["lines"])
    child = admin.get(f"/api/logs/models/{MODEL_ID}").json()
    assert child["redacted"] is False and "operator" in child["path"]


def test_vram_holders_hide_every_command_line_from_a_lan_reader(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core import vram_holders as vh

    def fake_view(probe: Any, engines_dir: Any, *, own_pids: Any = ()) -> dict[str, Any]:
        return {
            "engines_dir": PROFILE + "\\_UserData\\StudioForge\\engines",
            "orphan_count": 0,
            "holders": [
                {
                    "pid": 11104,
                    "classification": "foreign",
                    "exe": "E:\\SD\\ComfyUI\\venv\\Scripts\\python.exe",
                    "parent_name": "C:\\Windows\\explorer.exe",
                    "parent_cmdline": "docker run -e TOKEN=hunter2 image",
                    "detail": "--alias secret-alias --port 41001",
                    "per_gpu_bytes": {"2": 6_000_000_000},
                }
            ],
        }

    monkeypatch.setattr(vh, "holders_view", fake_view)
    payload = TestClient(app, client=LAN).get("/api/vram/holders").json()
    assert payload["redacted"] is True
    assert payload["engines_dir"] == "engines"
    row = payload["holders"][0]
    assert row["parent_cmdline"] is None and row["detail"] is None
    assert row["exe"] == "python.exe" and row["parent_name"] == "explorer.exe"
    assert row["pid"] == 11104 and row["per_gpu_bytes"] == {"2": 6_000_000_000}
    assert "hunter2" not in json.dumps(payload)
    full = TestClient(app, client=LOOPBACK).get("/api/vram/holders").json()
    assert full["holders"][0]["parent_cmdline"] == "docker run -e TOKEN=hunter2 image"
    assert "redacted" not in full


# ---------------------------------------------------------------------------
# F-8: a 500 names a reference, not the exception
# ---------------------------------------------------------------------------


def test_an_unhandled_error_hands_the_caller_a_reference_and_the_log_the_text(
    app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.api import app as app_mod

    async def boom() -> None:
        where = "C:\\Users\\operator\\registry.sqlite3"  # scrub-ok: fixture
        raise RuntimeError(f"sqlite3 at {where} is locked")

    app.add_api_route("/api/_boom", boom, methods=["GET"])
    logged: list[dict[str, Any]] = []
    monkeypatch.setattr(
        app_mod,
        "log",
        SimpleNamespace(exception=lambda event, **kw: logged.append({"event": event, **kw})),
    )
    reply = TestClient(app, client=LAN, raise_server_exceptions=False).get("/api/_boom")
    assert reply.status_code == 500
    error = reply.json()["error"]
    assert error["code"] == "internal_error"
    assert "operator" not in error["message"] and "sqlite3" not in error["message"]
    ref = error["studioforge"]["ref"]
    assert re.fullmatch(r"[0-9a-f]{8}", ref) and ref in error["message"]
    entry = next(e for e in logged if e["event"] == "unhandled error")
    assert entry["ref"] == ref and "operator" in entry["error"]


# ---------------------------------------------------------------------------
# A-3 / A-4 / A-8: the watchdog
# ---------------------------------------------------------------------------


class _FakeWatchdog:
    def __init__(self) -> None:
        self.restart_calls: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "up",
            "summary": "1 child up",
            "children_total": 1,
            "children_unhealthy": 0,
            "config_path": PROFILE + "\\_UserData\\StudioForge\\config.yaml",
            "children": [{"pid": 4242, "alias": "vendor/Model-Q8_0", "port": 41000}],
            "server": {"status": "ok"},
        }

    async def restart_server(self, **kwargs: Any) -> dict[str, Any]:
        self.restart_calls.append(kwargs)
        return {"ok": True, "method": "kill+respawn", "new_pid": 4242}


async def _inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b'{"reached":"mcp"}'})


async def _call(
    app: Any,
    method: str,
    path: str,
    *,
    client: tuple[str, int] | None,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[str, Any]]:
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": headers or [],
    }
    if client is not None:
        scope["client"] = client
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(raw)


async def test_the_watchdog_fails_closed_with_no_credential_configured() -> None:
    """Clear the PIN on an install with no key and the recovery sidecar used to
    be anonymous on 0.0.0.0:1235. Now: this machine only."""
    watchdog = _FakeWatchdog()
    app = wrap_asgi(_inner, watchdog, credentials=lambda: (None, None))

    for path in ("/restart", "/mcp"):
        status, body = await _call(app, "POST", path, client=LAN)
        assert status == 403, path
        assert body["error"]["code"] == "remote_admin_requires_credential"
    assert watchdog.restart_calls == []

    status, _ = await _call(app, "POST", "/restart", client=LOOPBACK)
    assert status == 200 and watchdog.restart_calls == [{"confirm": True}]
    status, body = await _call(app, "POST", "/mcp", client=None)
    assert status == 200 and body == {"reached": "mcp"}, "an in-process call has no peer"
    status, body = await _call(app, "GET", "/health", client=LAN)
    assert status == 200 and body["status"] == "up", "liveness stays open"


async def test_a_remote_uncredentialed_health_poll_sees_the_verdict_only() -> None:
    app = wrap_asgi(_inner, _FakeWatchdog(), credentials=lambda: (None, PIN))

    status, body = await _call(app, "GET", "/health", client=LAN)
    assert status == 200
    assert body["redacted"] is True and body["status"] == "up" and body["children_total"] == 1
    assert "config_path" not in body and "children" not in body and "server" not in body
    assert "operator" not in json.dumps(body)

    status, body = await _call(
        app, "GET", "/health", client=LAN, headers=[(b"x-mcp-pin", PIN.encode())]
    )
    assert "redacted" not in body and body["config_path"].endswith("config.yaml")
    status, body = await _call(
        app, "GET", "/health", client=LAN, headers=[(b"authorization", b"Bearer " + PIN.encode())]
    )
    assert "redacted" not in body
    status, body = await _call(app, "GET", "/health", client=LOOPBACK)
    assert "redacted" not in body and body["children"][0]["pid"] == 4242


def test_public_health_keeps_only_derived_fields() -> None:
    full = {
        "ok": False,
        "status": "degraded",
        "summary": "x",
        "children_total": 2,
        "children_unhealthy": 1,
        "watchdog_uptime_s": 12.5,
        "restart_in_progress": False,
        "config_path": "C:/somewhere",
        "children": [{}],
    }
    assert _public_health(full) == {
        "ok": False,
        "status": "degraded",
        "summary": "x",
        "children_total": 2,
        "children_unhealthy": 1,
        "watchdog_uptime_s": 12.5,
        "restart_in_progress": False,
        "redacted": True,
    }


async def test_kill_model_refuses_to_guess_between_two_matches() -> None:
    from tests.unit.test_watchdog import CHILD_PORT_START, Harness, _alive, build_mcp, call

    harness = Harness(_tmp_dir("kill-ambiguous"))
    try:
        first = harness.start_child("vendor/Qwen-Alpha-Q8_0", CHILD_PORT_START + 6)
        second = harness.start_child("vendor/Qwen-Beta-Q8_0", CHILD_PORT_START + 7)
        server = build_mcp(harness.watchdog())
        result = await call(server, "kill_model", model_name="qwen")
        assert result["ok"] is False
        assert result["error"]["code"] == "ambiguous_model"
        assert result["error"]["matches"] == ["vendor/Qwen-Alpha-Q8_0", "vendor/Qwen-Beta-Q8_0"]
        assert _alive(first) and _alive(second), "a kill is not guessed at"
        exact = await call(server, "kill_model", model_name="vendor/Qwen-Beta-Q8_0")
        assert exact["ok"] is True and exact["pids"] == [second]
    finally:
        harness.cleanup()


def _tmp_dir(name: str) -> Path:
    import tempfile

    return Path(tempfile.mkdtemp(prefix=f"sf-{name}-"))


# ---------------------------------------------------------------------------
# A-2 / A-8: the panel's Logs tab and the folder openers
# ---------------------------------------------------------------------------


class _Element:
    def __init__(self, seen: list[str], text: str = "") -> None:
        self.seen = seen
        if text:
            seen.append(text)

    def classes(self, *_args: Any, **_kwargs: Any) -> _Element:
        return self

    def __enter__(self) -> _Element:
        return self

    def __exit__(self, *_exc: Any) -> bool:
        return False


def test_the_logs_tab_shows_a_remote_viewer_nothing(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspect

    from studioforge.gui import tabs
    from studioforge.gui.tabs import logs as logs_mod
    from tests.unit.test_gui import _FakeState

    config.server.api_key = None
    seen: list[str] = []
    monkeypatch.setattr(
        logs_mod,
        "ui",
        SimpleNamespace(
            column=lambda: _Element(seen), label=lambda text: _Element(seen, str(text))
        ),
    )
    monkeypatch.setattr(tabs, "viewer_host", lambda: "10.0.0.7")
    state = _FakeState(config)
    state.registry = None
    logs_mod.render(tabs.GuiContext(config=config, api_state=state))
    assert seen and seen[0] == "Logs are not shown to a remote viewer"
    assert any("server.api_key" in text for text in seen)
    # The refresh path checks again: the controls live in a page a websocket
    # can re-enable, and that function is the one that reads the files.
    source = inspect.getsource(logs_mod.render)
    refresh = source[source.index("async def _refresh_once") :]
    assert refresh.lstrip().startswith("async def _refresh_once")
    assert 'require_local_admin(ctx, "reading the server and model logs")' in refresh


def test_opening_a_folder_on_the_servers_desktop_takes_the_d32_rule(
    config: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import os

    from studioforge.gui import tabs
    from studioforge.gui.tabs import setup as setup_mod
    from tests.unit.test_gui import _FakeState

    config.server.api_key = None
    notified: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        setup_mod, "ui", SimpleNamespace(notify=lambda m, **kw: notified.append((str(m), kw)))
    )
    opened: list[str] = []
    monkeypatch.setattr(os, "startfile", lambda p: opened.append(p), raising=False)
    ctx = tabs.GuiContext(config=config, api_state=_FakeState(config))
    target = tmp_path / "opened-from-afar"

    monkeypatch.setattr(tabs, "viewer_host", lambda: "10.0.0.7")
    setup_mod._open_path(ctx, target)
    assert not target.exists() and opened == []
    assert notified[-1][1]["type"] == "negative" and "server.api_key" in notified[-1][0]

    monkeypatch.setattr(tabs, "viewer_host", lambda: "127.0.0.1")
    setup_mod._open_path(ctx, target)
    assert target.is_dir()
    if sys.platform == "win32":
        assert opened == [str(target)]
    assert notified[-1][1]["type"] == "positive"


# ---------------------------------------------------------------------------
# The PIN-gated MCP plane is the operator's: its unload waives the lease guard
# ---------------------------------------------------------------------------


async def test_the_mcp_unload_is_the_operators_and_waives_the_lease_guard() -> None:
    """Reaching an MCP tool at all cleared the D32/PIN gate, so ``unload_model``
    passes ``force=True``; without it the operator's own ``sfctl unload`` of a
    benchmark's model would be refused -- the one caller entitled to do it."""
    from studioforge.mcp.management import build_management_mcp
    from tests.unit.test_mcp import TINY, call

    with_lease = _MCP_STATE()
    if with_lease is None:  # pragma: no cover - the fixture module could not be loaded
        pytest.skip("the MCP state fixture is unavailable")
    state, close = with_lease
    try:
        state.manager.leases.acquire([0], holder="benchmark", model_ids=[TINY])
        server = build_management_mcp(state)
        payload = await call(server, "unload_model", model_id=TINY)
        assert payload["ok"] is True, payload
        assert payload["unloaded"] is False, "nothing was resident; the point is no 409"
    finally:
        close()


def _MCP_STATE() -> tuple[Any, Any] | None:  # noqa: N802 - a fixture-shaped helper
    """The ``test_mcp.state`` fixture, driven by hand so it needs no pytest plumbing."""
    import tempfile

    from tests.unit import test_mcp

    generator = test_mcp.state.__wrapped__(Path(tempfile.mkdtemp(prefix="sf-mcp-d55-")))
    state = next(generator)

    def close() -> None:
        with contextlib.suppress(StopIteration):
            next(generator)

    return state, close
