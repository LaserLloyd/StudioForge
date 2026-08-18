"""Process-lifecycle hardening from the WP13 audit.

* a dead stderr (detached console, closed redirect) cannot take the server down
  through a log line: the stream handler detaches instead of raising, and under
  ``pythonw`` (``sys.stderr is None``) no stream handler is installed at all;
* on Linux, llama-server children are launched through a ``PR_SET_PDEATHSIG``
  shim so a ``kill -9`` of the server takes them with it (the POSIX half of D23);
  elsewhere the prefix is empty;
* the orphan sweep re-verifies a pid's identity right before killing it;
* the tray-supervised restart branch requires a *live* tray parent, and the
  supervisor variable is never inherited by the watchdog or its spawns.
"""

from __future__ import annotations

import io
import logging
import sys
from typing import Any

import pytest

from studioforge import logging as sf_logging


@pytest.fixture(autouse=True)
def _restore_root_logging() -> Any:
    """These tests reconfigure the global logging; put it back for the rest."""
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


class _DeadStream(io.TextIOBase):
    def write(self, text: str) -> int:  # type: ignore[override]
        raise ValueError("I/O operation on closed file")

    def flush(self) -> None:
        raise ValueError("I/O operation on closed file")


def test_a_dead_stderr_does_not_raise_out_of_a_log_call() -> None:
    handler = sf_logging._SafeStreamHandler(_DeadStream())
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello", None, None)
    handler.emit(record)  # must not raise
    # Detached for good: later records go nowhere and cost nothing.
    assert handler.stream is None
    handler.emit(record)


def test_configure_logging_installs_no_stream_handler_under_pythonw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr(sys, "stderr", None)
    sf_logging.configure_logging("INFO", log_dir=tmp_path)
    root = logging.getLogger()
    kinds = {type(h).__name__ for h in root.handlers}
    assert "_SafeStreamHandler" not in kinds and "StreamHandler" not in kinds
    assert "RingBufferHandler" in kinds and "FileHandler" in kinds
    # And logging still works (the ring buffer + file take the record).
    sf_logging.get_logger("t").info("under pythonw")


def test_configure_logging_uses_the_safe_handler_with_a_live_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    sf_logging.configure_logging("INFO")
    root = logging.getLogger()
    assert any(isinstance(h, sf_logging._SafeStreamHandler) for h in root.handlers)


def test_pdeathsig_prefix_is_linux_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from studioforge.core import supervisor as sup

    monkeypatch.delenv("SF_NO_PDEATHSIG", raising=False)
    monkeypatch.setattr(sup.sys, "platform", "win32")
    assert sup._pdeathsig_prefix() == []
    monkeypatch.setattr(sup.sys, "platform", "darwin")
    assert sup._pdeathsig_prefix() == []
    monkeypatch.setattr(sup.sys, "platform", "linux")
    prefix = sup._pdeathsig_prefix()
    assert prefix[0] == sys.executable and prefix[1] == "-c"
    assert "prctl(1" in prefix[2] and "execv" in prefix[2]
    monkeypatch.setenv("SF_NO_PDEATHSIG", "1")
    assert sup._pdeathsig_prefix() == []


@pytest.mark.skipif(sys.platform == "win32", reason="os.execv is not exec on Windows")
def test_the_pdeathsig_shim_execs_its_argument(tmp_path: Any) -> None:
    """The shim is plain Python: on POSIX its exec half runs the target
    (prctl is only reachable on Linux; the test stubs the CDLL call)."""
    import subprocess

    from studioforge.core.supervisor import _PDEATHSIG_SHIM

    stub = _PDEATHSIG_SHIM.replace(
        "ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0); ", ""
    )
    result = subprocess.run(
        [sys.executable, "-c", stub, sys.executable, "-c", "print('exec ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "exec ok" in result.stdout


def test_reclaim_skips_a_pid_that_is_gone_before_the_kill(monkeypatch: pytest.MonkeyPatch) -> None:
    from studioforge.core import vram_holders as vh

    entry = vh.EngineProcess(
        pid=424242,
        alias="leaked",
        port=18100,
        exe="/x/engines/b1/llama-server",
        create_time=1.0,
        parent_pid=1,
        parent_alive=False,
        parent_recycled=False,
        classification=vh.CLASS_ORPHAN,
    )
    monkeypatch.setattr(vh, "find_engine_processes", lambda *a, **k: [entry])
    killed: list[int] = []
    monkeypatch.setattr(vh, "kill_process_tree", lambda pid, **k: killed.append(pid))
    monkeypatch.setattr(vh, "process_is_alive", lambda pid, create_time=None: False)
    actions = vh.reclaim_orphans("/x/engines", own_pids=[])
    assert killed == [], "a recycled/vanished pid must never reach kill_process_tree"
    assert actions and actions[0]["killed"] is True
    assert "already gone" in actions[0].get("note", "")


# ---------------------------------------------------------------------------
# D33: the port binds before the slow half of startup runs
# ---------------------------------------------------------------------------


def _slow_boot_app(tmp_path: Any, gate: Any) -> Any:
    """An app whose registry scan blocks on ``gate`` -- a stand-in for a cold
    library scan or a 600 MB engine install."""
    from studioforge.api.app import build_state, create_app
    from studioforge.config import Config

    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    state = build_state(config)

    class _BlockingRegistry:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

        def scan(self) -> Any:
            # Short: an executor thread parked here outlives a cancelled boot
            # task, and the loop's shutdown waits for it.
            gate.wait(timeout=3)
            return self.inner.scan()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

    state.registry = _BlockingRegistry(state.registry)

    class _NoEngine:
        """No network: a fresh data dir would otherwise ask GitHub for b10425."""

        async def ensure_engine(self, **kwargs: Any) -> Any:
            raise RuntimeError("no engine in this test")

        def active(self) -> Any:
            return None

        async def aclose(self) -> None:
            return None

    state.engine_manager = _NoEngine()
    return create_app(config, state=state, start_background=True)


def test_health_answers_and_reports_the_phase_while_boot_is_still_running(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from fastapi.testclient import TestClient

    from studioforge.core import vram_holders as vh

    monkeypatch.setattr(vh, "reclaim_orphans", lambda *a, **k: [])
    gate = threading.Event()
    app = _slow_boot_app(tmp_path, gate)
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        body = http.get("/health").json()
        assert body["status"] == "ok", "liveness is not readiness"
        assert body["boot"]["ready"] is False
        assert body["boot"]["phase"] == "scanning models"
        assert body["can_serve"] is False
        assert "still starting" in body["cannot_serve_reason"]
        gate.set()
        deadline = 30
        while not app.state.boot.ready and deadline > 0:
            import time as _time

            _time.sleep(0.05)
            deadline -= 0.05
        body = http.get("/health").json()
        assert body["boot"]["ready"] is True
        assert body["boot"]["phase"] == "ready"
        assert body["boot"]["error"] is None


def test_a_model_listing_waits_for_the_first_scan(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``/v1/models`` during the boot scan waits for the index instead of
    answering an empty library (bounded; the bound is shortened here)."""
    import threading
    import time as _time

    from fastapi.testclient import TestClient

    from studioforge.api import app as app_module
    from studioforge.core import vram_holders as vh

    monkeypatch.setattr(vh, "reclaim_orphans", lambda *a, **k: [])
    monkeypatch.setattr(app_module, "SCAN_WAIT_S", 5.0)
    gate = threading.Event()
    app = _slow_boot_app(tmp_path, gate)
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        threading.Timer(0.3, gate.set).start()
        started = _time.monotonic()
        response = http.get("/v1/models")
        waited = _time.monotonic() - started
    assert response.status_code == 200
    assert waited >= 0.25, "the listing must have waited for the scan"
    assert app.state.boot.ready


def test_shutdown_during_boot_cancels_it_cleanly(tmp_path: Any) -> None:
    import threading

    from fastapi.testclient import TestClient

    gate = threading.Event()  # never set: the boot is mid-scan at shutdown
    app = _slow_boot_app(tmp_path, gate)
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        assert http.get("/health").status_code == 200
    gate.set()  # release the worker thread the blocking scan is parked on
    boot = app.state.boot
    assert boot.ready, "finish() must run on cancellation so nothing waits forever"
    assert boot.error is not None and "shut down" in boot.error


async def test_the_manager_waits_for_the_boot_gate_before_resolving() -> None:
    import asyncio

    from tests.unit.test_load_retry import StubPlanner, StubSupervisor, make_manager

    manager = make_manager(StubSupervisor(), StubPlanner())
    manager.boot_gate = asyncio.Event()
    manager.config.gateway.load_timeout_s = 5.0

    async def release() -> None:
        await asyncio.sleep(0.2)
        manager.boot_gate.set()

    asyncio.create_task(release())
    started = asyncio.get_running_loop().time()
    instance = await manager.load("test/model")
    assert instance.state == "ready"
    assert asyncio.get_running_loop().time() - started >= 0.15
