"""Every child that ends outside a deliberate stop logs ``model_exited``, exactly once (D64, CR-7).

Raised by the ClawChat V14 audit (2026-09-12): four llama-server crash dumps in
the window, one matching ``exit_code=3221226505`` (``0xC0000409``,
STATUS_STACK_BUFFER_OVERRUN) to the minute -- and two of the four with no
``model_exited`` line at all. ``model_exited`` was logged in exactly one place,
the watcher of a child that had become ready. A child that died while loading, a
child torn down for never becoming healthy, and any child whose watcher had
itself failed ended with no exit record.

These run the fake llama-server from ``test_supervisor.py`` (a real child
process on the test port range) and pin: a crash while serving, a crash while
loading, a health-timeout teardown, a failed relaunch and repeated crashes each
give one line per process; a failed OS wait still reports the exit, as
``exit_code_unavailable``; a watcher that fails reports the child it lost; a
deliberate stop logs none; and Windows' NTSTATUS codes are named.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

from studioforge.config import Config
from studioforge.core import supervisor as supervisor_module
from studioforge.core.supervisor import describe_exit_code
from studioforge.errors import ModelLoadError
from studioforge.types import ModelSettings
from tests.unit.test_supervisor import (  # noqa: F401 - the fixtures
    config,
    fake_binary,
    fake_sup,
    make_plan,
    make_record,
    wait_for,
)


class Lines:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def exits(self) -> list[dict[str, Any]]:
        return [fields for _, event, fields in self.lines if event == "model_exited"]

    def events(self) -> list[str]:
        return [event for _, event, _ in self.lines]


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> Lines:
    lines = Lines()

    def level(name: str) -> Any:
        return lambda event, **fields: lines.lines.append((name, event, fields))

    monkeypatch.setattr(
        supervisor_module,
        "log",
        SimpleNamespace(
            **{n: level(n) for n in ("debug", "info", "warning", "error", "exception")}
        ),
    )
    return lines


def _kill(pid: int) -> None:
    with contextlib.suppress(psutil.Error):
        psutil.Process(pid).kill()


# ---------------------------------------------------------------------------
# Every way a child ends
# ---------------------------------------------------------------------------


async def test_a_crash_while_serving_logs_one_exit_with_its_code(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    marker = tmp_path / "crashed-once.marker"
    supervisor = fake_sup(config, fake_binary)
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            extra_flags=f"--fake-crash-after 1.0 --fake-crash-once {marker.as_posix()}"
        ),
    )
    try:
        info = await supervisor.start(record, make_plan())
        first_pid = info.pid
        assert await wait_for(lambda: info.restarts >= 1 and info.state == "ready", timeout=20.0)
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert len(exits) == 1, exits
    assert exits[0]["pid"] == first_pid
    assert exits[0]["exit_code"] == 9
    assert exits[0]["phase"] == "running"


async def test_a_child_that_dies_while_loading_logs_its_exit(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    """The likeliest shape of the two silent 2026-09 crashes: dead before /health said ok."""
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-exit-code 3"))
    try:
        with pytest.raises(ModelLoadError):
            await supervisor.start(record, make_plan())
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert len(exits) == 1, exits
    assert exits[0]["exit_code"] == 3
    assert exits[0]["phase"] == "startup"
    assert "cause" not in exits[0], "it crashed; nobody tore it down"


async def test_a_child_torn_down_for_never_becoming_healthy_logs_its_exit_and_why(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    config.gateway.load_timeout_s = 1.0
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-unhealthy"))
    try:
        with pytest.raises(ModelLoadError):
            await supervisor.start(record, make_plan())
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert len(exits) == 1, exits
    assert exits[0]["phase"] == "startup"
    assert "did not become healthy" in exits[0]["cause"]


async def test_repeated_crashes_log_one_exit_per_process(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    config.gateway.max_restarts = 2
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-crash-after 1.0"))
    try:
        info = await supervisor.start(record, make_plan())
        assert await wait_for(lambda: info.state == "failed", timeout=40.0)
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert len(exits) == 3, exits
    assert len({e["pid"] for e in exits}) == 3
    assert all(e["exit_code"] == 9 for e in exits)
    assert "model_restart_limit" in recorded.events()


async def test_a_relaunch_that_hangs_logs_the_exit_of_the_child_it_kills(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    config.gateway.load_timeout_s = 2.0
    config.gateway.max_restarts = 1
    once = tmp_path / "crashed-once.marker"
    crashed = tmp_path / "crash-happened.marker"
    supervisor = fake_sup(config, fake_binary)
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            extra_flags=(
                f"--fake-crash-after 1.0 --fake-crash-once {once.as_posix()} "
                f"--fake-touch-on-crash {crashed.as_posix()} "
                f"--fake-unhealthy-if {crashed.as_posix()}"
            )
        ),
    )
    try:
        info = await supervisor.start(record, make_plan())
        assert await wait_for(lambda: info.state == "failed", timeout=30.0)
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert [e["phase"] for e in exits] == ["running", "restart"], exits
    assert "failed relaunch" in exits[1]["cause"]
    assert exits[0]["pid"] != exits[1]["pid"]


async def test_a_failed_os_wait_still_reports_the_exit_as_unavailable(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The process object is no help: the child is watched by pid, and the line says so."""
    config.gateway.max_restarts = 0
    monkeypatch.setattr(supervisor_module, "EXIT_POLL_S", 0.05)
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    try:
        info = await supervisor.start(record, make_plan())
        inst = supervisor._instances[record.id]
        pid = info.pid
        assert pid is not None and inst.watcher is not None

        inst.watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await inst.watcher

        async def broken_wait() -> int:
            raise RuntimeError("transport closed under wait()")

        real_proc = inst.proc
        inst.wait_task = asyncio.create_task(broken_wait())
        inst.proc = SimpleNamespace(pid=pid, returncode=None)  # type: ignore[assignment]
        inst.watcher = asyncio.create_task(supervisor._watch(inst))
        await asyncio.sleep(0.2)
        _kill(pid)

        assert await wait_for(lambda: bool(recorded.exits()), timeout=15.0)
        inst.proc = real_proc
    finally:
        await supervisor.aclose()

    exits = recorded.exits()
    assert len(exits) == 1, exits
    assert exits[0]["pid"] == pid
    assert exits[0]["exit_code"] is None
    assert exits[0]["exit_code_unavailable"] is True
    assert "model_wait_failed" in recorded.events()


async def test_a_watcher_that_fails_reports_the_child_it_lost(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)

    def exploding_failure_message(*_a: Any, **_k: Any) -> str:
        raise RuntimeError("bug in the watcher")

    try:
        info = await supervisor.start(record, make_plan())
        monkeypatch.setattr(supervisor, "_failure_message", exploding_failure_message)
        assert info.pid is not None
        _kill(info.pid)
        assert await wait_for(lambda: info.state == "failed", timeout=15.0)
    finally:
        await supervisor.aclose()

    assert "model_watch_failed" in recorded.events()
    exits = recorded.exits()
    assert len(exits) == 1, exits
    assert "supervision failed" in exits[0]["cause"]
    assert exits[0]["exit_code"] is not None


async def test_a_deliberate_stop_logs_no_exit(
    config: Config,  # noqa: F811 - the imported fixture, by design
    tmp_path: Path,
    fake_binary: Path,  # noqa: F811 - the imported fixture, by design
    recorded: Lines,
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    try:
        await supervisor.start(record, make_plan())
        await supervisor.stop(record.id)
        await asyncio.sleep(0.3)
    finally:
        await supervisor.aclose()
    assert recorded.exits() == []
    assert "model_stopped" in recorded.events()


# ---------------------------------------------------------------------------
# Reading the code
# ---------------------------------------------------------------------------


def test_the_windows_stack_buffer_overrun_code_is_named() -> None:
    """The 09-08 07:00 dump: 3221226505 is 0xC0000409, and a log line should say so."""
    assert describe_exit_code(3221226505) == {
        "exit_code": 3221226505,
        "exit_code_hex": "0xC0000409",
        "exit_status": "STATUS_STACK_BUFFER_OVERRUN",
    }


def test_a_posix_signal_and_an_unknown_code_are_described() -> None:
    assert describe_exit_code(-15)["signal"] == "SIGTERM"  # a name every platform has
    assert describe_exit_code(1) == {"exit_code": 1}
    assert describe_exit_code(None) == {"exit_code": None, "exit_code_unavailable": True}
