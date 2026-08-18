"""``POST /api/restart/server`` must actually restart the server.

The incident (2026-08-18 00:00, this box). The endpoint answered
``{"restarting": true, "via": "watchdog"}`` and the server did not restart --
twice over, because *both* paths were broken and each hid the other:

1. **The handoff was refused.** ``_ask_watchdog_to_restart`` sent
   ``Authorization: Bearer <server.api_key>`` and nothing else. On the default
   install ``server.api_key`` is null and the only credential is the MCP pairing
   PIN, which the watchdog accepts and the app never sent. The watchdog's ASGI
   wrapper answered ``401`` *before any watchdog code ran*, which is why
   ``watchdog.log`` had nothing in it at the time and the failure looked silent.
2. **The fallback could not work.** ``_self_restart`` then respawned itself
   detached, and the replacement ran its own port preflight while this process
   still held 1234 and 8080 and its watchdog child still held 1235 -- so the
   replacement exited ``rc 3`` ("startup port conflict") every time, and the old
   process logged "the replacement process exited immediately; staying up".

And the wreckage: ``manager.stop()`` had already latched ``draining``, so a
server that stayed up and went on serving reported ``draining: true`` forever.

See DECISIONS.md D21 for the design these tests pin down.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import typer

from studioforge.__main__ import _console, _log_mcp_banner, _preflight_ports, _spawn_watchdog
from studioforge.api import admin_routes
from studioforge.config import Config
from studioforge.core import ports as ports_module
from studioforge.core.manager import ModelManager
from studioforge.watchdog import server as wd

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Recorder:
    """Stands in for the module logger so branch coverage is assertable."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str) -> Any:
        def emit(event: str, **fields: Any) -> None:
            self.lines.append((level, event, fields))

        return emit

    def __getattr__(self, name: str) -> Any:
        if name in {"debug", "info", "warning", "error", "critical", "exception"}:
            return self._record(name)
        raise AttributeError(name)

    def events(self, level: str | None = None) -> list[str]:
        return [e for lvl, e, _ in self.lines if level is None or lvl == level]

    def fields(self, event: str) -> dict[str, Any]:
        for _lvl, name, fields in self.lines:
            if name == event:
                return fields
        raise AssertionError(f"no log line {event!r} in {self.events()}")


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    def __init__(self, calls: list[tuple[str, dict[str, Any]]], response: _Response) -> None:
        self._calls = calls
        self._response = response

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _Response:
        self._calls.append((url, kwargs))
        return self._response


def _state(*, api_key: str | None = None, pin: str | None = None) -> Any:
    class _Server:
        pass

    class _Mcp:
        pass

    class _Config:
        pass

    class _State:
        pass

    server, mcp, config, state = _Server(), _Mcp(), _Config(), _State()
    server.api_key = api_key  # type: ignore[attr-defined]
    mcp.pin = pin  # type: ignore[attr-defined]
    config.server = server  # type: ignore[attr-defined]
    config.mcp = mcp  # type: ignore[attr-defined]
    state.config = config  # type: ignore[attr-defined]
    return state


async def _noop_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(admin_routes, "log", rec)
    monkeypatch.setattr(admin_routes.asyncio, "sleep", _noop_sleep)
    return rec


# ---------------------------------------------------------------------------
# 1. the credential -- the actual root cause
# ---------------------------------------------------------------------------


class TestTheHandoffCredential:
    async def test_the_mcp_pin_is_sent_when_there_is_no_api_key(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """THE regression. This exact config is what shipped: key null, PIN set."""
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx, "AsyncClient", lambda **_: _FakeClient(calls, _Response(200))
        )
        fell_back: list[bool] = []
        monkeypatch.setattr(
            admin_routes, "_self_restart", lambda _s: _appended(fell_back)  # noqa: ARG005
        )

        state = _state(api_key=None, pin="40021977")
        await admin_routes._ask_watchdog_to_restart(state, "http://127.0.0.1:1235")

        assert calls[0][1]["headers"] == {"Authorization": "Bearer 40021977"}
        assert fell_back == [], "a credentialled handoff must not fall back"
        assert "watchdog accepted the restart handoff" in recorder.events()
        assert state.restart_status["outcome"] == "handed-off"

    async def test_the_api_key_wins_when_both_exist(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx, "AsyncClient", lambda **_: _FakeClient(calls, _Response(200))
        )

        await admin_routes._ask_watchdog_to_restart(
            _state(api_key="k3yk3y", pin="40021977"), "http://wd:1235"
        )

        assert calls[0][1]["headers"] == {"Authorization": "Bearer k3yk3y"}

    async def test_no_credential_at_all_sends_no_header(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """A watchdog with neither credential is open by design; do not invent one."""
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx, "AsyncClient", lambda **_: _FakeClient(calls, _Response(200))
        )

        await admin_routes._ask_watchdog_to_restart(_state(), "http://wd:1235")

        assert calls[0][1]["headers"] == {}

    async def test_a_refusal_names_the_credential_it_used_and_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """The 401 that was logged but not explained. Say what we sent."""
        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx,
            "AsyncClient",
            lambda **_: _FakeClient(calls, _Response(401, "needs a credential")),
        )
        fell_back: list[bool] = []
        monkeypatch.setattr(
            admin_routes, "_self_restart", lambda _s: _appended(fell_back)  # noqa: ARG005
        )

        state = _state(api_key=None, pin=None)
        await admin_routes._ask_watchdog_to_restart(state, "http://wd:1235")

        fields = recorder.fields("watchdog refused the restart handoff; respawning instead")
        assert fields["status"] == 401
        assert fields["credential"] == "none"
        assert fell_back == [True], "the caller was promised a restart; keep trying"
        assert state.restart_status["outcome"] == "watchdog-refused"
        assert "401" in state.restart_status["detail"]

    async def test_a_non_http_failure_still_reaches_the_fallback(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """Only ``httpx.HTTPError`` was caught, so anything else vanished."""

        class _Exploding:
            async def __aenter__(self) -> _Exploding:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def post(self, *a: Any, **k: Any) -> _Response:
                raise RuntimeError("event loop is closed")

        monkeypatch.setattr(admin_routes.httpx, "AsyncClient", lambda **_: _Exploding())
        fell_back: list[bool] = []
        monkeypatch.setattr(
            admin_routes, "_self_restart", lambda _s: _appended(fell_back)  # noqa: ARG005
        )

        state = _state()
        await admin_routes._ask_watchdog_to_restart(state, "http://wd:1235")

        assert fell_back == [True]
        assert state.restart_status["outcome"] == "watchdog-unreachable"


async def _appended(sink: list[bool]) -> None:
    sink.append(True)


# ---------------------------------------------------------------------------
# 2. a restart task that dies must not die quietly
# ---------------------------------------------------------------------------


class TestRestartTaskFailuresAreAudible:
    async def test_a_crashing_restart_task_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """The reply already said "restarting"; this task is the only witness left."""
        import asyncio

        async def boom() -> None:
            raise RuntimeError("no")

        task = admin_routes._spawn_restart_task(boom())
        with contextlib.suppress(RuntimeError):
            await task
        await asyncio.sleep(0)

        assert "the restart task crashed; nothing restarted" in recorder.events("error")

    async def test_a_completed_task_is_not_reported_as_a_failure(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        import asyncio

        async def fine() -> None:
            return None

        await admin_routes._spawn_restart_task(fine())
        await asyncio.sleep(0)

        assert recorder.events("error") == []


# ---------------------------------------------------------------------------
# 3. the self-respawn fallback, and the drain flag it used to strand
# ---------------------------------------------------------------------------


class _FakeManager:
    def __init__(self) -> None:
        self.draining = False
        self.resumed = 0

    async def stop(self) -> None:
        self.draining = True

    async def resume(self) -> None:
        self.draining = False
        self.resumed += 1


class _FakeChild:
    def __init__(self, returncode: int | None) -> None:
        self.returncode = returncode
        self.pid = 4242

    def poll(self) -> int | None:
        return self.returncode


def _restart_state(child: _FakeChild | None, *, spawned: bool = True) -> tuple[Any, _FakeManager]:
    manager = _FakeManager()

    class _State:
        pass

    state = _State()
    state.config = _state().config  # type: ignore[attr-defined]
    state.manager = manager  # type: ignore[attr-defined]
    return state, manager


def _patch_updater(
    monkeypatch: pytest.MonkeyPatch, *, child: _FakeChild | None, spawned: bool = True
) -> None:
    class _FakeUpdater:
        respawn_wait_s = 45.0

        def __init__(self, _config: Any) -> None:
            self._last_child = child

        def _respawn_detached(self) -> bool:
            return spawned

    monkeypatch.setattr("studioforge.core.updater.Updater", _FakeUpdater)


class TestSelfRestartFallback:
    async def test_a_dead_replacement_puts_the_drain_flag_back_down(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """The stuck flag: /health said draining:true for hours on a serving box."""
        _patch_updater(monkeypatch, child=_FakeChild(returncode=3))
        state, manager = _restart_state(None)

        await admin_routes._self_restart(state)

        assert manager.draining is False
        assert manager.resumed == 1

    async def test_a_dead_replacement_says_so_through_the_api(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """The exact 00:00 failure, now readable instead of log-only."""
        _patch_updater(monkeypatch, child=_FakeChild(returncode=3))
        state, _ = _restart_state(None)

        await admin_routes._self_restart(state)

        status = state.restart_status
        assert status["outcome"] == "failed"
        assert status["returncode"] == 3
        assert "restart did not happen" in status["detail"]

    async def test_a_failure_to_spawn_at_all_also_undrains(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        _patch_updater(monkeypatch, child=None, spawned=False)
        state, manager = _restart_state(None)

        await admin_routes._self_restart(state)

        assert manager.resumed == 1
        assert state.restart_status["outcome"] == "failed"

    async def test_a_live_replacement_hands_the_watchdog_over_and_exits(
        self, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """Exiting must not take the supervisor with it (D21)."""
        _patch_updater(monkeypatch, child=_FakeChild(returncode=None))
        state, manager = _restart_state(None)
        signalled: list[int] = []
        monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append(sig))

        await admin_routes._self_restart(state)

        assert state.handing_over is True
        assert signalled, "the process must actually signal itself to shut down"
        assert manager.resumed == 0, "a successful restart must NOT cancel the drain"
        assert state.restart_status["outcome"] == "exiting"


class TestManagerResume:
    """``draining`` has to be able to come back down."""

    async def test_resume_clears_the_flag_and_restarts_the_ttl_sweeper(self) -> None:
        # Built without its dependency graph on purpose: resume() touches three
        # attributes and nothing else, and a real ModelManager needs a database, a
        # registry, a planner and a supervisor to exist.
        manager = ModelManager.__new__(ModelManager)
        manager._draining = True
        manager._ttl_task = None
        swept: list[bool] = []

        async def _fake_loop() -> None:
            swept.append(True)

        manager._ttl_loop = _fake_loop  # type: ignore[method-assign]

        await manager.resume()

        assert manager.draining is False
        assert manager._ttl_task is not None
        await manager._ttl_task
        assert swept == [True], "eviction must start happening again"


# ---------------------------------------------------------------------------
# 4. the replacement is told which pid it replaces
# ---------------------------------------------------------------------------


class TestRespawnHandshake:
    def test_the_replacement_is_told_which_pid_it_replaces(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Without this the child fails its preflight on our ports, every time."""
        from studioforge.core.updater import Updater

        captured: dict[str, Any] = {}

        def _fake_popen(argv: list[str], **kwargs: Any) -> Any:
            captured["argv"] = argv
            captured["env"] = kwargs["env"]
            return _FakeChild(returncode=None)

        monkeypatch.setattr(subprocess, "Popen", _fake_popen)
        config = _config(tmp_path)
        updater = Updater(config)

        assert updater._respawn_detached() is True

        env = captured["env"]
        assert env[ports_module.ENV_RESPAWN_PARENT_PID] == str(os.getpid())
        assert float(env[ports_module.ENV_RESPAWN_WAIT_S]) >= config.server.drain_timeout_s
        assert "serve" in captured["argv"]


# ---------------------------------------------------------------------------
# 5. preflight: wait for the parent, adopt the watchdog
# ---------------------------------------------------------------------------


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@contextlib.contextmanager
def _held_port() -> Iterator[tuple[int, socket.socket]]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield int(sock.getsockname()[1]), sock
    finally:
        with contextlib.suppress(OSError):
            sock.close()


def _config(tmp_path: Path, *, server: int | None = None, watchdog: int | None = None) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": server or _free_port()},
        gui={"enabled": False, "host": "127.0.0.1", "port": _free_port()},
        watchdog={
            "enabled": watchdog is not None,
            "host": "127.0.0.1",
            "port": watchdog or _free_port(),
        },
        logging={"level": "ERROR"},
    )


@contextlib.contextmanager
def _fake_watchdog_http(body: dict[str, Any], status: int = 200) -> Iterator[int]:
    """A real listener that answers ``/health`` the way the watchdog does."""
    payload = json.dumps(body).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _clean_respawn_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        ports_module.ENV_RESPAWN_PARENT_PID,
        ports_module.ENV_RESPAWN_WAIT_S,
        ports_module.ENV_ADOPT_WATCHDOG,
    ):
        monkeypatch.delenv(key, raising=False)


class TestWatchdogAdoption:
    def test_our_own_running_watchdog_is_adoptable(self, tmp_path: Path) -> None:
        config = _config(tmp_path, watchdog=1)
        body = {
            "status": "up",
            "watchdog_uptime_s": 812.4,
            "config_path": str(config.config_path),
        }
        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            presence = ports_module.inspect_running_watchdog(config)

        assert presence.adoptable is True
        assert presence.uptime_s == 812.4

    def test_a_watchdog_reporting_a_down_server_is_still_adoptable(self, tmp_path: Path) -> None:
        """503 is what it says *during* the restart -- the moment we need it."""
        config = _config(tmp_path, watchdog=1)
        body = {
            "status": "down",
            "watchdog_uptime_s": 3.0,
            "config_path": str(config.config_path),
        }
        with _fake_watchdog_http(body, status=503) as port:
            config.watchdog.port = port
            assert ports_module.inspect_running_watchdog(config).adoptable is True

    def test_a_watchdog_for_a_different_config_is_a_conflict(self, tmp_path: Path) -> None:
        """Two StudioForge installs share a box; never adopt the neighbour's."""
        config = _config(tmp_path, watchdog=1)
        body = {
            "status": "up",
            "watchdog_uptime_s": 1.0,
            "config_path": str(tmp_path / "somebody-else" / "config.yaml"),
        }
        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            presence = ports_module.inspect_running_watchdog(config)

        assert presence.adoptable is False
        assert "somebody-else" in presence.reason

    def test_a_stranger_on_the_port_is_a_conflict(self, tmp_path: Path) -> None:
        config = _config(tmp_path, watchdog=1)
        with _fake_watchdog_http({"hello": "i am not a watchdog"}) as port:
            config.watchdog.port = port
            presence = ports_module.inspect_running_watchdog(config)

        assert presence.adoptable is False
        assert "not a StudioForge watchdog" in presence.reason

    def test_a_silent_holder_is_a_conflict(self, tmp_path: Path) -> None:
        with _held_port() as (port, _sock):
            config = _config(tmp_path, watchdog=port)
            presence = ports_module.inspect_running_watchdog(config)

        assert presence.adoptable is False

    def test_a_free_port_is_not_an_adoption(self, tmp_path: Path) -> None:
        config = _config(tmp_path, watchdog=_free_port())
        presence = ports_module.inspect_running_watchdog(config)
        assert presence.adoptable is False
        assert "nothing is listening" in presence.reason

    def test_adoption_can_be_switched_off(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Escape hatch for "the watchdog's own code changed"."""
        monkeypatch.setenv(ports_module.ENV_ADOPT_WATCHDOG, "0")
        config = _config(tmp_path, watchdog=1)
        body = {"status": "up", "watchdog_uptime_s": 1.0, "config_path": str(config.config_path)}
        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            assert ports_module.inspect_running_watchdog(config).adoptable is False

    def test_preflight_adopts_instead_of_refusing_to_start(self, tmp_path: Path) -> None:
        """The bug: a watchdog-driven restart could never pass its own preflight."""
        config = _config(tmp_path, watchdog=1)
        body = {"status": "up", "watchdog_uptime_s": 1.0, "config_path": str(config.config_path)}
        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            assert _preflight_ports(config) is None

    def test_preflight_still_refuses_a_foreign_holder_of_the_watchdog_port(
        self, tmp_path: Path
    ) -> None:
        with _held_port() as (port, _sock):
            config = _config(tmp_path, watchdog=port)
            with pytest.raises(typer.Exit) as excinfo:
                _preflight_ports(config)

        assert excinfo.value.exit_code == 3

    def test_spawn_watchdog_adopts_rather_than_starting_a_second_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two supervisors disagreeing about which process is the server is worse
        than none."""
        config = _config(tmp_path, watchdog=1)
        body = {"status": "up", "watchdog_uptime_s": 1.0, "config_path": str(config.config_path)}

        def _must_not_spawn(*a: Any, **k: Any) -> Any:
            raise AssertionError("a second watchdog was started")

        monkeypatch.setattr(subprocess, "Popen", _must_not_spawn)
        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            assert _spawn_watchdog(config) is None


class TestRespawnPortWait:
    def test_preflight_waits_for_the_parent_to_release_its_ports(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The whole self-respawn path depends on this one behaviour."""
        monkeypatch.setenv(ports_module.ENV_RESPAWN_PARENT_PID, str(os.getpid()))
        monkeypatch.setenv(ports_module.ENV_RESPAWN_WAIT_S, "20")

        with _held_port() as (port, sock):
            config = _config(tmp_path, server=port)
            releaser = threading.Timer(0.75, sock.close)
            releaser.start()
            try:
                started = time.monotonic()
                assert _preflight_ports(config) is None
            finally:
                releaser.cancel()

        assert time.monotonic() - started >= 0.5, "it must actually have waited"

    def test_preflight_gives_up_when_the_ports_never_free(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ports_module.ENV_RESPAWN_PARENT_PID, str(os.getpid()))
        monkeypatch.setenv(ports_module.ENV_RESPAWN_WAIT_S, "0.5")

        with _held_port() as (port, _sock):
            config = _config(tmp_path, server=port)
            with pytest.raises(typer.Exit) as excinfo:
                _preflight_ports(config)

        assert excinfo.value.exit_code == 3

    def test_without_the_handshake_a_busy_port_fails_at_once(self, tmp_path: Path) -> None:
        """An ordinary operator mistake must not sit in a 45-second wait."""
        with _held_port() as (port, _sock):
            config = _config(tmp_path, server=port)
            started = time.monotonic()
            with pytest.raises(typer.Exit):
                _preflight_ports(config)

        assert time.monotonic() - started < 3.0

    def test_wait_for_ports_returns_the_ones_still_held(self) -> None:
        with _held_port() as (port, _sock):
            conflict = ports_module.PortConflict(
                role="server", port=port, host="127.0.0.1", setting="server.port"
            )
            remaining = ports_module.wait_for_ports([conflict], 0.3)

        assert [c.port for c in remaining] == [port]

    def test_a_non_numeric_parent_pid_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ports_module.ENV_RESPAWN_PARENT_PID, "not-a-pid")
        assert ports_module.respawn_parent_pid() is None

    def test_the_handshake_is_consumed_and_not_inherited(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It describes one startup. Inherited, it would make every later port
        conflict wait 45s on a pid that died days ago."""
        monkeypatch.setenv(ports_module.ENV_RESPAWN_PARENT_PID, str(os.getpid()))

        assert _preflight_ports(_config(tmp_path)) is None

        assert ports_module.ENV_RESPAWN_PARENT_PID not in os.environ


# ---------------------------------------------------------------------------
# 5b. the endpoint itself: what it promises, and where the truth lands
# ---------------------------------------------------------------------------


class _FakeRequest:
    def __init__(self, state: Any) -> None:
        class _App:
            pass

        self.app = _App()
        self.app.state = state  # type: ignore[attr-defined]


def _route_state(config: Config) -> Any:
    class _State:
        pass

    state = _State()
    state.config = config  # type: ignore[attr-defined]
    state.manager = _FakeManager()  # type: ignore[attr-defined]
    return state


class TestRestartRoute:
    async def test_it_refuses_without_confirmation(self, tmp_path: Path) -> None:
        from studioforge.errors import BadRequestError

        state = _route_state(_config(tmp_path))
        with pytest.raises(BadRequestError):
            await admin_routes.restart_server(_FakeRequest(state), False)

    async def test_a_reachable_watchdog_gets_the_job_and_the_reply_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        config = _config(tmp_path, watchdog=1)
        config.mcp.pin = "40021977"
        config.server.api_key = None
        body = {"status": "up", "watchdog_uptime_s": 5.0, "config_path": str(config.config_path)}
        handed: list[str] = []
        monkeypatch.setattr(
            admin_routes,
            "_ask_watchdog_to_restart",
            lambda _s, url: _appended_str(handed, url),
        )

        with _fake_watchdog_http(body) as port:
            config.watchdog.port = port
            state = _route_state(config)
            reply = await admin_routes.restart_server(_FakeRequest(state), True)
            await _drain_restart_tasks()

        assert reply["via"] == "watchdog"
        assert reply["credential"] == "mcp_pin", "the reply must say what it will authenticate with"
        assert "/api/restart/status" in reply["verify"]
        assert handed == [f"http://127.0.0.1:{port}"]

    async def test_an_unreachable_watchdog_falls_back_and_says_which_path_it_took(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        config = _config(tmp_path, watchdog=_free_port())
        respawned: list[str] = []
        monkeypatch.setattr(
            admin_routes, "_self_restart", lambda _s: _appended_str(respawned, "self")
        )

        state = _route_state(config)
        reply = await admin_routes.restart_server(_FakeRequest(state), True)
        await _drain_restart_tasks()

        assert reply["via"] == "self-respawn"
        assert "did not answer" in reply["note"]
        assert respawned == ["self"]

    async def test_status_is_never_before_anything_has_been_asked_for(
        self, tmp_path: Path
    ) -> None:
        state = _route_state(_config(tmp_path))
        assert (await admin_routes.restart_status(_FakeRequest(state)))["outcome"] == "never"

    async def test_a_failed_restart_is_readable_from_the_api_afterwards(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorder: _Recorder
    ) -> None:
        """The whole point: the 200 was a promise, and this is where it is kept or broken."""
        _patch_updater(monkeypatch, child=_FakeChild(returncode=3))
        state = _route_state(_config(tmp_path))

        await admin_routes._self_restart(state)
        status = await admin_routes.restart_status(_FakeRequest(state))

        assert status["outcome"] == "failed"
        assert "restart did not happen" in status["detail"]


async def _appended_str(sink: list[str], value: str) -> None:
    sink.append(value)


async def _drain_restart_tasks() -> None:
    import asyncio

    for _ in range(10):
        if not admin_routes._RESTART_TASKS:
            return
        await asyncio.sleep(0)
    await asyncio.gather(*admin_routes._RESTART_TASKS, return_exceptions=True)


# ---------------------------------------------------------------------------
# 6. the watchdog must not kill itself
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, pid: int, children: list[_FakeProc] | None = None) -> None:
        self.pid = pid
        self._children = children or []
        self.killed = False

    def children(self, recursive: bool = False) -> list[_FakeProc]:
        return list(self._children)

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.killed = True


class TestKillProcessTreeExclusion:
    def test_the_watchdog_is_not_killed_by_the_tree_it_lives_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The watchdog is a CHILD of the main server, so it is inside the tree."""
        watchdog_proc = _FakeProc(pid=os.getpid())
        llama = _FakeProc(pid=777)
        main = _FakeProc(pid=100, children=[watchdog_proc, llama])
        waited: list[list[int]] = []

        monkeypatch.setattr(wd.psutil, "Process", lambda pid: main)
        monkeypatch.setattr(
            wd.psutil,
            "wait_procs",
            lambda procs, timeout=0: (waited.append([p.pid for p in procs]), ([], []))[1],
        )

        signalled = wd.kill_process_tree(100, exclude={os.getpid()})

        assert sorted(signalled) == [100, 777]
        assert watchdog_proc.killed is False, "the watchdog killed itself mid-restart"
        assert main.killed is True and llama.killed is True
        assert os.getpid() not in waited[0], "it would have waited for its own death"

    def test_descendants_of_an_excluded_pid_are_still_killable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After one watchdog restart the new server is the watchdog's child, and
        the next restart must still be able to kill it."""
        target = _FakeProc(pid=500)
        monkeypatch.setattr(wd.psutil, "Process", lambda pid: target)
        monkeypatch.setattr(wd.psutil, "wait_procs", lambda procs, timeout=0: ([], []))

        assert wd.kill_process_tree(500, exclude={os.getpid()}) == [500]
        assert target.killed is True

    def test_restart_server_excludes_the_watchdogs_own_pid(self) -> None:
        source = Path(wd.__file__ or "").read_text(encoding="utf-8")
        body = source.split("async def restart_server")[1].split("# -- model kills")[0]
        assert "keep_alive = {os.getpid()}" in body
        assert "exclude=keep_alive" in body


# ---------------------------------------------------------------------------
# 7. a windowless launch must not be fatal
# ---------------------------------------------------------------------------


class _DeadStream:
    """A stream that exists and cannot be written to.

    The realistic windowless failure: a detached console or a closed handle,
    not ``sys.stdout is None``. ``print`` dies on this one.
    """

    def write(self, _text: str) -> int:
        raise ValueError("I/O operation on closed file")

    def flush(self) -> None:
        return None


@pytest.fixture
def main_log(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    """Take the log's own console writes out of the picture.

    Where log records land is ``configure_logging``'s problem (in the real
    process they go through stdlib logging, which swallows write errors by
    design). What these tests are about is the *banner*, which writes to the
    console directly and had nothing catching it.
    """
    rec = _Recorder()
    monkeypatch.setattr("studioforge.__main__.log", rec)
    return rec


class TestConsoleWithoutATerminal:
    def test_print_really_does_die_on_a_dead_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pins the hazard itself, so the guard below is not cargo cult.

        (``sys.stdout is None`` is *currently* survivable -- CPython's ``print``
        and click's ``echo`` both no-op -- so it is deliberately not what this
        asserts.)
        """
        monkeypatch.setattr(sys, "stdout", _DeadStream())
        with pytest.raises(ValueError):
            print("x", flush=True)  # noqa: T201 - demonstrating the failure

    def test_writing_to_a_dead_stream_is_a_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "stdout", _DeadStream())
        _console("this must not raise")

    def test_writing_without_stdout_at_all_is_a_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys, "stdout", None)
        _console("this must not raise either")

    @pytest.mark.parametrize("stdout", [None, "dead"])
    def test_the_startup_banner_survives_a_windowless_launch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main_log: _Recorder, stdout: Any
    ) -> None:
        """It died right after "management MCP mounted", with no traceback."""
        config = _config(tmp_path)
        config.mcp.enabled = True
        config.mcp.advertise = True
        config.mcp.pin_required = True
        config.mcp.pin = "12345678"
        monkeypatch.setattr(sys, "stdout", None if stdout is None else _DeadStream())

        _log_mcp_banner(config)

        assert "mcp endpoint" in main_log.events(), "the PIN must still reach the log"

    @pytest.mark.parametrize("stderr", [None, "dead"])
    def test_a_port_conflict_still_exits_cleanly_without_a_console(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, main_log: _Recorder, stderr: Any
    ) -> None:
        """The conflict message is unprintable; the exit code still has to be right."""
        monkeypatch.setattr(sys, "stderr", None if stderr is None else _DeadStream())
        with _held_port() as (port, _sock):
            config = _config(tmp_path, server=port)
            with pytest.raises(typer.Exit) as excinfo:
                _preflight_ports(config)

        assert excinfo.value.exit_code == 3
        assert "startup port conflict" in main_log.events("error")
