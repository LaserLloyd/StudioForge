"""An unload must be verified, not assumed.

Incident (production, the system this replaces): *"the unload API returned
success while models stayed resident; we resorted to killing llama-server
processes by hand."* Killing the process tree (which this supervisor already
does) is the right mechanism -- but a kill call that *returned* is not evidence
that the process is gone, and on a GPU-only server a survivor keeps its CUDA
context, i.e. all of its VRAM, forever.

So ``stop()`` re-checks the pid, escalates a survivor, records the VRAM
actually reclaimed, and refuses to report success it cannot back up.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from studioforge.config import Config
from studioforge.core import supervisor as supervisor_module
from studioforge.core.supervisor import Supervisor, process_is_alive
from studioforge.errors import ModelUnloadError
from studioforge.types import GB, GpuInfo, LoadPlan, ModelRecord

TEST_PORT_START = 19500
TEST_PORT_END = 19540

FAKE_CHILD = '''
"""Minimal llama-server stand-in; optionally deaf to SIGTERM."""
import json
import signal
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

if "--fake-ignore-sigterm" in ARGS:
    # POSIX: genuinely ignore the polite signal, so only SIGKILL ends this.
    # Windows: TerminateProcess cannot be caught, so this is a no-op there and
    # the test degenerates to the ordinary path (still a valid assertion).
    for name in ("SIGTERM", "SIGINT", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, signal.SIG_IGN)
            except (ValueError, OSError):
                pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/props":
            self._send(200, {"model_alias": ALIAS})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def log_message(self, fmt, *args):
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False


Server(("127.0.0.1", PORT), Handler).serve_forever()
'''


class VramProbe:
    """A GPU that gives memory back after the first reading.

    The first ``list_gpus`` call is the supervisor's "before" measurement and
    the second is its "after", which is exactly the sequence a real unload
    produces as the driver tears the CUDA context down.
    """

    backend = "fake"

    def __init__(self, before: int = 20 * GB, after: int = 8 * GB) -> None:
        self._readings = [before, after]
        self.used = before

    def available(self) -> bool:
        return True

    def list_gpus(self) -> list[GpuInfo]:
        self.used = self._readings.pop(0) if self._readings else self.used
        return [
            GpuInfo(
                index=0,
                name="FakeGPU0",
                total_bytes=32 * GB,
                free_bytes=32 * GB - self.used,
                used_bytes=self.used,
            )
        ]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return self.list_gpus()[0] if index == 0 else None

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return (13, 3)

    def compute_processes(self) -> list[object]:
        return []

    def shutdown(self) -> None:
        return None


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = TEST_PORT_START
    cfg.gateway.child_port_end = TEST_PORT_END
    cfg.gateway.load_timeout_s = 20.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.gateway.max_restarts = 0
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "engine" / "fake_llama_server.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CHILD, encoding="utf-8")
    return path


def resolver(path: Path) -> Callable[[str | None], Path]:
    def resolve(tag: str | None) -> Path:
        return path

    return resolve


def make_supervisor(config: Config, binary: Path, probe: object | None = None) -> Supervisor:
    return Supervisor(
        config,
        resolve_binary=resolver(binary),
        launch_prefix=[sys.executable, "-u"],
        probe=probe,  # type: ignore[arg-type]
    )


def make_record(tmp_path: Path, model_id: str = "test/model", extra_flags: str = "") -> ModelRecord:
    path = tmp_path / "models" / "model.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    record = ModelRecord(id=model_id, name=model_id, path=path)
    record.settings.extra_flags = extra_flags
    return record


def plan(model_id: str = "test/model") -> LoadPlan:
    return LoadPlan(model_id=model_id, devices=[0], ctx_size=2048)


# ---------------------------------------------------------------------------


async def test_stop_verifies_the_pid_is_gone(
    config: Config, fake_binary: Path, tmp_path: Path
) -> None:
    """ "the unload API returned success while models stayed resident"."""
    supervisor = make_supervisor(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, plan())
    pid = info.pid
    assert pid is not None

    await supervisor.stop(record.id)

    report = supervisor.unload_report(record.id)
    assert report is not None
    assert report.pid_gone is True
    assert report.pid == pid
    assert process_is_alive(pid) is False


async def test_a_child_that_ignores_sigterm_is_still_verified_dead(
    config: Config, fake_binary: Path, tmp_path: Path
) -> None:
    """The polite signal is not the guarantee; the verification is."""
    supervisor = make_supervisor(config, fake_binary)
    record = make_record(tmp_path, extra_flags="--fake-ignore-sigterm")
    info = await supervisor.start(record, plan())
    pid = info.pid
    assert pid is not None

    await supervisor.stop(record.id, timeout=1.0)

    report = supervisor.unload_report(record.id)
    assert report is not None and report.pid_gone is True
    assert process_is_alive(pid) is False, "a SIGTERM-deaf child survived an 'unload'"
    assert supervisor.get(record.id) is None


async def test_reclaimed_vram_is_measured_not_assumed(
    config: Config, fake_binary: Path, tmp_path: Path
) -> None:
    supervisor = make_supervisor(config, fake_binary, probe=VramProbe(20 * GB, 8 * GB))
    record = make_record(tmp_path)
    await supervisor.start(record, plan())

    await supervisor.stop(record.id)

    report = supervisor.unload_report(record.id)
    assert report is not None
    assert report.vram_before_bytes == 20 * GB
    assert report.vram_reclaimed_bytes == 12 * GB


async def test_a_survivor_is_reported_instead_of_returning_success(
    config: Config, fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode that motivated this: success reported, VRAM still held.

    ``process_is_alive`` is forced to keep saying "alive" so the unverifiable
    case is exercised without leaving a real orphan behind.
    """
    supervisor = make_supervisor(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, plan())
    pid = info.pid

    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)

    with pytest.raises(ModelUnloadError) as excinfo:
        await supervisor.stop(record.id, timeout=1.0)

    assert str(pid) in excinfo.value.message
    report = supervisor.unload_report(record.id)
    assert report is not None
    assert report.pid_gone is False
    assert report.escalated is True, "a survivor must be escalated to a forced kill"
    # Still in the table: hiding a process that holds VRAM is how it stays lost.
    assert supervisor.get(record.id) is not None
    assert supervisor.get(record.id).state == "failed"  # type: ignore[union-attr]

    monkeypatch.undo()
    with contextlib.suppress(Exception):
        if pid is not None:
            psutil.Process(pid).kill()


async def test_kill_reports_false_when_it_cannot_be_verified(
    config: Config, fake_binary: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = make_supervisor(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, plan())
    pid = info.pid

    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)
    assert await supervisor.kill(record.id) is False

    monkeypatch.undo()
    with contextlib.suppress(Exception):
        if pid is not None:
            psutil.Process(pid).kill()


def test_pid_reuse_is_not_mistaken_for_survival() -> None:
    """A recycled pid must not look like our model still running."""
    own = psutil.Process()

    assert process_is_alive(own.pid, create_time=own.create_time()) is True
    assert process_is_alive(own.pid, create_time=own.create_time() - 3600) is False
