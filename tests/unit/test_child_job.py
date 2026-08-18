"""The Windows job object that stops children outliving the supervisor.

The property under test is the one that failed on 2026-08-18 (DECISIONS.md
D23): three ``llama-server`` children survived because their parent's cleanup
was an ``atexit`` hook, and ``atexit`` runs only for a *clean* exit. A job
object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` moves the guarantee into the
kernel, so the test is deliberately crude: assign a real child, close the
handle, and require the child to be gone.

The second property matters just as much: the net must never be load-bearing.
A box where the job cannot be created, or where assignment is refused, still
loads models -- it just loses the safety net, loudly.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core import supervisor as sup_mod
from studioforge.core.supervisor import Supervisor, WindowsChildJob, create_child_job
from studioforge.types import LoadPlan, ModelCapabilities, ModelRecord, ModelSettings

TEST_PORT_START = 19560
TEST_PORT_END = 19580

windows_only = pytest.mark.skipif(os.name != "nt", reason="job objects are a Windows API")


# ---------------------------------------------------------------------------
# The kernel guarantee
# ---------------------------------------------------------------------------


@windows_only
def test_closing_the_job_handle_kills_its_children() -> None:
    """A child in the job dies with the handle, with no cleanup code involved.

    This is the whole point: nothing in Python runs between "handle closed" and
    "child dead", which is why it also holds for a SIGKILL of the supervisor.
    """
    job = WindowsChildJob()
    child = subprocess.Popen(  # noqa: S603 - fixed argv, our own interpreter
        [sys.executable, "-c", "import time; time.sleep(60)"]
    )
    try:
        assert job.assign(child.pid) is True
        job.close()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.05)
        assert child.poll() is not None, "the child outlived the job handle"
    finally:
        job.close()
        if child.poll() is None:  # pragma: no cover - only on failure
            child.kill()
        child.wait(timeout=5)


@windows_only
def test_close_is_idempotent() -> None:
    job = WindowsChildJob()
    job.close()
    job.close()
    assert job.available is False
    # A closed job refuses assignments rather than raising: aclose() runs before
    # the last teardown in some shutdown orders.
    assert job.assign(os.getpid()) is False


def test_create_child_job_degrades_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """No pywin32, no job -- and no exception. The server still has to boot."""
    monkeypatch.setattr(
        sup_mod, "_load_win32", lambda: (_ for _ in ()).throw(ImportError("no pywin32"))
    )
    assert create_child_job() is None


def test_assignment_failure_is_reported_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A job that refuses nesting logs one warning, not one per load."""
    if os.name != "nt":
        pytest.skip("job objects are a Windows API")
    job = WindowsChildJob()
    warnings: list[dict[str, Any]] = []
    monkeypatch.setattr(
        sup_mod.log, "warning", lambda event, **kw: warnings.append({"event": event, **kw})
    )

    class Boom:
        def OpenProcess(self, *_args: Any) -> Any:  # noqa: N802 - win32 spelling
            raise OSError(5, "Access is denied")

    monkeypatch.setattr(sup_mod, "_load_win32", lambda: (job._win32job, Boom()))
    try:
        assert job.assign(os.getpid()) is False
        assert job.assign(os.getpid()) is False
    finally:
        job.close()
    assert len(warnings) == 1
    assert warnings[0]["event"] == "child_job_assign_failed"


# ---------------------------------------------------------------------------
# The net is never load-bearing
# ---------------------------------------------------------------------------

FAKE_CHILD = '''
"""Minimal llama-server stand-in: answers /health and /props, ignores the rest."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = sys.argv[1:]


def opt(name, default=None):
    if name in ARGS:
        index = ARGS.index(name)
        if index + 1 < len(ARGS):
            return ARGS[index + 1]
    return default


PORT = int(opt("--port", "0"))
ALIAS = opt("--alias", "fake")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/props"):
            body = json.dumps({"model_alias": ALIAS}).encode()
        else:
            body = b'{"status": "ok"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = TEST_PORT_START
    cfg.gateway.child_port_end = TEST_PORT_END
    cfg.gateway.load_timeout_s = 30.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.ensure_dirs()
    return cfg


def _record(tmp_path: Path) -> ModelRecord:
    model = tmp_path / "models" / "tiny.gguf"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"GGUF")
    return ModelRecord(
        id="tiny",
        name="tiny",
        kind="chat",
        path=model,
        capabilities=ModelCapabilities(),
        settings=ModelSettings(),
    )


def _resolver(path: Path) -> Callable[[str | None], Path]:
    def resolve(_tag: str | None) -> Path:
        return path

    return resolve


class RefusingJob:
    """A job object that exists but refuses every assignment."""

    def __init__(self) -> None:
        self.attempts: list[int] = []

    @property
    def available(self) -> bool:
        return True

    def assign(self, pid: int) -> bool:
        self.attempts.append(pid)
        return False

    def close(self) -> None:
        return None


async def test_a_refused_assignment_does_not_fail_the_load(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety net must never be the reason a model will not load.

    Assignment is refused by the kernel on a box whose existing job forbids
    nesting. That costs the guarantee, and it is logged -- but a server that
    refused to load models because of it would have traded a rare leak for a
    total outage.
    """
    binary = tmp_path / "engine" / "fake_llama_server.py"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(FAKE_CHILD, encoding="utf-8")

    job = RefusingJob()
    monkeypatch.setattr(sup_mod, "create_child_job", lambda: job)
    supervisor = Supervisor(
        config, resolve_binary=_resolver(binary), launch_prefix=[sys.executable, "-u"]
    )
    try:
        info = await supervisor.start(
            _record(tmp_path), LoadPlan(model_id="tiny", devices=[0], ctx_size=1024)
        )
        assert info.state == "ready"
        assert job.attempts == [info.pid]
        # And the child really is running: a suspended-but-never-resumed child
        # would have failed the health poll instead.
        assert await supervisor.health("tiny") is True
    finally:
        await asyncio.wait_for(supervisor.aclose(), timeout=30)


async def test_children_are_tracked_for_the_orphan_sweep(config: Config, tmp_path: Path) -> None:
    """``child_pids`` is what stops the sweep killing our own live children."""
    binary = tmp_path / "engine" / "fake_llama_server.py"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text(FAKE_CHILD, encoding="utf-8")
    supervisor = Supervisor(
        config, resolve_binary=_resolver(binary), launch_prefix=[sys.executable, "-u"]
    )
    try:
        assert supervisor.child_pids() == set()
        info = await supervisor.start(
            _record(tmp_path), LoadPlan(model_id="tiny", devices=[0], ctx_size=1024)
        )
        assert supervisor.child_pids() == {info.pid}
        await supervisor.stop("tiny", timeout=10.0)
        assert supervisor.child_pids() == set()
    finally:
        await asyncio.wait_for(supervisor.aclose(), timeout=30)
