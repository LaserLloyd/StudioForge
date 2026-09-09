"""An unload that failed is reported as one -- and a slow one is not called failed.

The regression audit of 2026-09-09 (§2, §4.2) found ``Supervisor.stop_all``
gathering every ``stop()`` with ``return_exceptions=True`` and discarding the
list unread, so ``POST /api/models/unload-all`` answered 200 with a cheerful
count for children still alive on the GPUs; it found four fail-open branches
in the identity check and a dead stderr pump that said nothing; and it found
the unload chain giving a 20+ GB CUDA context ~35 s to tear down before
raising ``ModelUnloadError`` for a process that died seconds later. These
tests pin the repairs on the supervisor side; the manager and route halves
are the orchestrator's (``lane_d2_requests.md``).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from studioforge.config import Config
from studioforge.core import supervisor as supervisor_module
from studioforge.core.supervisor import Supervisor, _Instance
from studioforge.errors import ModelUnloadError
from studioforge.types import GB, GpuInfo, InstanceInfo
from tests.unit.test_supervisor import FAKE_CHILD, make_plan, make_record, resolver

# See test_gpu_only_policy for the map of every test file's child range.
TEST_PORT_START = 19541
TEST_PORT_END = 19599


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


def fake_sup(config: Config, binary: Path, probe: object | None = None) -> Supervisor:
    return Supervisor(
        config,
        resolve_binary=resolver(binary),
        launch_prefix=[sys.executable, "-u"],
        probe=probe,  # type: ignore[arg-type]
    )


class _Recorder:
    """Keeps every structured log call by level; swallows nothing important."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str) -> Callable[..., None]:
        def emit(event: str, **fields: Any) -> None:
            self.events.append((level, event, fields))

        return emit

    def __getattr__(self, name: str) -> Callable[..., None]:
        return self._record(name)

    def of(self, level: str, event: str) -> list[dict[str, Any]]:
        return [f for lvl, e, f in self.events if lvl == level and e == event]


class SettlingProbe:
    """A GPU whose memory comes back one reading after the process dies."""

    backend = "fake"

    def __init__(self) -> None:
        self.readings = [20 * GB, 8 * GB]
        self.used = 20 * GB

    def available(self) -> bool:
        return True

    def list_gpus(self) -> list[GpuInfo]:
        self.used = self.readings.pop(0) if self.readings else self.used
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


def _kill_quietly(pid: int | None) -> None:
    with contextlib.suppress(Exception):
        if pid is not None:
            psutil.Process(pid).kill()


# ---------------------------------------------------------------------------
# stop_all
# ---------------------------------------------------------------------------


async def test_stop_all_reports_every_child_by_name(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    await supervisor.start(make_record(tmp_path, "model-a"), make_plan("model-a"))
    await supervisor.start(make_record(tmp_path, "model-b"), make_plan("model-b"))
    try:
        results = await supervisor.stop_all()
    finally:
        await supervisor.aclose()
    assert results == {"model-a": None, "model-b": None}
    assert supervisor.list() == []


async def test_stop_all_surfaces_a_survivor_instead_of_swallowing_it(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``return_exceptions=True`` used to be the last anyone heard of a failed
    unload. Now the survivor comes back by name, logged at ERROR, still in the
    table with state ``failed`` -- so unload-all can refuse to say 200."""
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    supervisor = fake_sup(config, fake_binary)
    good = await supervisor.start(make_record(tmp_path, "model-a"), make_plan("model-a"))
    bad = await supervisor.start(make_record(tmp_path, "model-b"), make_plan("model-b"))
    good_pid, bad_pid = good.pid, bad.pid
    assert bad_pid is not None

    real_alive = supervisor_module.process_is_alive

    def alive_only_for_b(pid: int, **kwargs: Any) -> bool:
        return True if pid == bad_pid else real_alive(pid, **kwargs)

    monkeypatch.setattr(supervisor_module, "process_is_alive", alive_only_for_b)
    try:
        results = await supervisor.stop_all(timeout=1.0)
        monkeypatch.setattr(supervisor_module, "process_is_alive", real_alive)

        assert results["model-a"] is None
        failure = results["model-b"]
        assert isinstance(failure, ModelUnloadError)
        assert failure.details["model_id"] == "model-b" and str(bad_pid) in failure.message
        logged = recorder.of("error", "model_unload_failed")
        assert [f["model_id"] for f in logged] == ["model-b"]
        assert logged[0]["pid"] == bad_pid
        # The survivor is not hidden: still listed, and honest about its state.
        survivor = supervisor.get("model-b")
        assert survivor is not None and survivor.state == "failed"
        assert supervisor.get("model-a") is None
    finally:
        monkeypatch.setattr(supervisor_module, "process_is_alive", real_alive)
        for pid in (good_pid, bad_pid):
            _kill_quietly(pid)
        # With the liveness check honest again, the second sweep takes the
        # survivor down for real -- which is also the aclose() path proving it
        # copes with a failed instance in the table.
        await supervisor.aclose()
    assert supervisor.get("model-b") is None


async def test_stop_all_still_never_raises_so_shutdown_reaches_the_job_close(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    supervisor = fake_sup(config, fake_binary)
    info = await supervisor.start(make_record(tmp_path), make_plan())
    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)
    try:
        results = await supervisor.stop_all(timeout=1.0)  # must not raise
        assert isinstance(results[info.model_id], ModelUnloadError)
        await supervisor.aclose()  # nor this
    finally:
        monkeypatch.undo()
        _kill_quietly(info.pid)


# ---------------------------------------------------------------------------
# The settle: patience for a slow driver teardown (F1), a beat before the
# VRAM sample (F6)
# ---------------------------------------------------------------------------


def _pending_instance(loop: asyncio.AbstractEventLoop, *, done: bool = False) -> Any:
    future: asyncio.Future[int] = loop.create_future()
    if done:
        future.set_result(0)
    return SimpleNamespace(record=SimpleNamespace(id="m"), create_time=None, wait_task=future)


async def test_a_child_that_dies_during_the_settle_is_a_verified_unload(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every signal has been sent; the driver is still tearing the context
    down. That is an unload in progress, not a failed one."""
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    monkeypatch.setattr(supervisor_module, "UNLOAD_SETTLE_S", 2.0)
    monkeypatch.setattr(supervisor_module, "UNLOAD_POLL_S", 0.01)
    calls = {"n": 0}

    def dying(pid: int, **_kw: Any) -> bool:
        calls["n"] += 1
        return calls["n"] <= 3  # alive, alive, alive, gone

    monkeypatch.setattr(supervisor_module, "process_is_alive", dying)
    supervisor = Supervisor(config, resolve_binary=resolver(tmp_path / "x"))
    inst = _pending_instance(asyncio.get_running_loop())

    assert await supervisor._linger(inst, 4242) is False  # noqa: SLF001 - the unit under test
    assert calls["n"] == 4
    settled = recorder.of("info", "unload_settled")
    assert settled and settled[0]["pid"] == 4242


async def test_the_settle_gives_up_at_its_deadline(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisor_module, "UNLOAD_SETTLE_S", 0.05)
    monkeypatch.setattr(supervisor_module, "UNLOAD_POLL_S", 0.01)
    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)
    supervisor = Supervisor(config, resolve_binary=resolver(tmp_path / "x"))
    inst = _pending_instance(asyncio.get_running_loop())
    started = asyncio.get_running_loop().time()
    assert await supervisor._linger(inst, 4242) is True  # noqa: SLF001
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_the_settle_stops_early_once_the_os_has_reported_the_exit(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A done wait_task with a liveness check that still says "alive" is not
    a teardown in progress; waiting a minute would only delay the honest
    answer (and every test that forces "alive" to exercise it)."""
    monkeypatch.setattr(supervisor_module, "UNLOAD_SETTLE_S", 30.0)
    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)
    supervisor = Supervisor(config, resolve_binary=resolver(tmp_path / "x"))
    inst = _pending_instance(asyncio.get_running_loop(), done=True)
    started = asyncio.get_running_loop().time()
    assert await supervisor._linger(inst, 4242) is True  # noqa: SLF001
    assert asyncio.get_running_loop().time() - started < 1.0


async def test_a_survivor_is_still_escalated_before_the_settle(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order kept: forced tree kill first, patience after."""
    order: list[str] = []
    monkeypatch.setattr(supervisor_module, "UNLOAD_SETTLE_S", 0.05)
    monkeypatch.setattr(supervisor_module, "UNLOAD_POLL_S", 0.01)
    supervisor = fake_sup(config, fake_binary)
    info = await supervisor.start(make_record(tmp_path), make_plan())
    real_kill = supervisor_module.kill_process_tree

    def kill(pid: int, **kw: Any) -> None:
        order.append("kill")
        real_kill(pid, **kw)

    async def linger(inst: Any, pid: int) -> bool:
        order.append("linger")
        return True

    monkeypatch.setattr(supervisor_module, "kill_process_tree", kill)
    monkeypatch.setattr(supervisor_module, "process_is_alive", lambda *a, **k: True)
    monkeypatch.setattr(supervisor, "_linger", linger)
    try:
        with pytest.raises(ModelUnloadError):
            await supervisor.stop(info.model_id, timeout=1.0)
        # Read the evidence now: aclose() below sweeps the survivor again with
        # the liveness check honest, and that second, verified unload rightly
        # replaces this report.
        report = supervisor.unload_report(info.model_id)
    finally:
        monkeypatch.undo()
        _kill_quietly(info.pid)
        await supervisor.aclose()
    assert order[-2:] == ["kill", "linger"]
    assert report is not None and report.escalated is True and report.pid_gone is False


async def test_vram_is_sampled_after_a_settle_when_there_is_a_probe(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settles: list[float] = []

    async def settle(self: Supervisor) -> None:
        settles.append(self.vram_settle_s)

    monkeypatch.setattr(Supervisor, "_settle_vram", settle)
    supervisor = fake_sup(config, fake_binary, probe=SettlingProbe())
    record = make_record(tmp_path)
    await supervisor.start(record, make_plan())
    await supervisor.stop(record.id)
    await supervisor.aclose()

    assert settles == [supervisor_module.VRAM_SETTLE_S]
    assert supervisor.vram_settle_s == supervisor_module.VRAM_SETTLE_S == 1.2
    report = supervisor.unload_report(record.id)
    assert report is not None and report.vram_reclaimed_bytes == 12 * GB


async def test_no_probe_means_no_settle(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settles: list[float] = []

    async def settle(self: Supervisor) -> None:
        settles.append(self.vram_settle_s)

    monkeypatch.setattr(Supervisor, "_settle_vram", settle)
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    await supervisor.start(record, make_plan())
    await supervisor.stop(record.id)
    await supervisor.aclose()
    assert settles == []


# ---------------------------------------------------------------------------
# The fail-open identity check and the dead pump say so
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, body: Any = None, *, raw: bool = False) -> None:
        self.status_code = status_code
        self._body = body
        self._raw = raw

    def json(self) -> Any:
        if self._raw:
            raise ValueError("not json")
        return self._body


class _PropsClient:
    def __init__(self, response: _Response | Exception) -> None:
        self._response = response

    async def get(self, url: str, timeout: float = 0) -> _Response:  # noqa: ASYNC109 - httpx shape
        if isinstance(self._response, Exception):
            raise self._response
        return self._response

    async def aclose(self) -> None:
        return None


def _instance(tmp_path: Path, supervisor: Supervisor) -> _Instance:
    record = make_record(tmp_path)
    inst = _Instance(
        info=InstanceInfo(model_id=record.id, state="loading", port=TEST_PORT_START),
        record=record,
        plan=make_plan(),
        port=TEST_PORT_START,
        engine_tag=None,
        draft=None,
        adapters=(),
        log_path=tmp_path / "m.log",
    )
    inst.argv = ["llama-server", "--alias", record.id]
    return inst


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (OSError("connection refused"), "unreachable"),
        (_Response(503), "HTTP 503"),
        (_Response(200, raw=True), "non-JSON"),
        (_Response(200, ["not", "a", "dict"]), "JSON list"),
        (_Response(200, {"no_alias": True}), "no model_alias"),
        (_Response(200, {"model_alias": 7}), "is a int"),
    ],
)
async def test_every_fail_open_branch_of_the_identity_check_is_logged(
    config: Config,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: _Response | Exception,
    reason: str,
) -> None:
    """The fail-open is deliberate (a hard dependency on /props is not the
    goal); silence about it was not. A squatter that answers /health and not
    /props was adopted as a successful load with no line saying the check
    never ran."""
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    supervisor = Supervisor(
        config,
        resolve_binary=resolver(tmp_path / "x"),
        client=_PropsClient(response),  # type: ignore[arg-type]
    )
    inst = _instance(tmp_path, supervisor)
    assert await supervisor._confirm_identity(inst) is True  # noqa: SLF001 - fail-open kept
    unchecked = recorder.of("warning", "child_identity_unchecked")
    assert len(unchecked) == 1
    assert reason in unchecked[0]["reason"]
    assert unchecked[0]["model_id"] == inst.record.id and unchecked[0]["port"] == inst.port


async def test_a_matching_alias_is_confirmed_without_a_warning(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    supervisor = Supervisor(
        config,
        resolve_binary=resolver(tmp_path / "x"),
        client=_PropsClient(_Response(200, {"model_alias": "qwen2.5-7b"})),  # type: ignore[arg-type]
    )
    inst = _instance(tmp_path, supervisor)
    assert await supervisor._confirm_identity(inst) is True  # noqa: SLF001
    assert recorder.of("warning", "child_identity_unchecked") == []
    # ...and a foreign alias is still the hard refusal it always was.
    supervisor2 = Supervisor(
        config,
        resolve_binary=resolver(tmp_path / "x"),
        client=_PropsClient(_Response(200, {"model_alias": "somebody-else"})),  # type: ignore[arg-type]
    )
    assert await supervisor2._confirm_identity(inst) is False  # noqa: SLF001
    assert recorder.of("error", "child_port_conflict")


async def test_a_dead_output_pump_says_which_stream_died_and_why(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the moment the pump dies ``stderr_ring`` stops filling, and the
    next failure message reads "No output captured" -- on precisely the child
    being debugged. The log line is the only clue that this is why."""
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    supervisor = Supervisor(config, resolve_binary=resolver(tmp_path / "x"))
    inst = _instance(tmp_path, supervisor)

    class Broken:
        async def readline(self) -> bytes:
            raise RuntimeError("transport closed under us")

    await supervisor._pump(inst, Broken(), stderr=True)  # type: ignore[arg-type]  # noqa: SLF001
    ended = recorder.of("warning", "child_output_pump_ended")
    assert len(ended) == 1
    assert ended[0]["stream"] == "stderr" and "transport closed" in ended[0]["error"]
    assert ended[0]["model_id"] == inst.record.id


# ---------------------------------------------------------------------------
# The job object logs every unprotected child, not only the first
# ---------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="job objects are a Windows API")
def test_job_assignment_failures_are_logged_per_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """The old once-ever latch logged at boot and then launched every later
    model unprotected in silence -- the D23 guarantee quietly absent."""
    job = supervisor_module.WindowsChildJob()
    warnings: list[dict[str, Any]] = []
    monkeypatch.setattr(
        supervisor_module.log,
        "warning",
        lambda event, **kw: warnings.append({"event": event, **kw}),
    )

    class Boom:
        def OpenProcess(self, *_args: Any) -> Any:  # noqa: N802 - win32 spelling
            raise OSError(5, "Access is denied")

    monkeypatch.setattr(supervisor_module, "_load_win32", lambda: (job._win32job, Boom()))  # noqa: SLF001
    try:
        assert job.assign(1111) is False
        assert job.assign(1111) is False  # same child again: one line
        assert job.assign(2222) is False  # a new child: its own line
    finally:
        job.close()
    assert [w["pid"] for w in warnings] == [1111, 2222]
    assert all(w["event"] == "child_job_assign_failed" for w in warnings)
