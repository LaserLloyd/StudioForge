"""One instance per data directory, and the app path that depends on it.

These tests exist because of the 2026-08-18 incident (DECISIONS.md D24): a
second ``create_app`` against the live data directory resumed the live download
queue and interleaved its bytes into a ``.part`` the live server was already
writing. The lock is the thing that makes that impossible; the app tests are the
thing that makes sure the lock is actually consulted before any worker starts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from studioforge.api.app import create_app
from studioforge.config import Config
from studioforge.core.instance_lock import INSTANCE_LOCK_NAME, InstanceLock

# ---------------------------------------------------------------------------
# The lock itself
# ---------------------------------------------------------------------------


def test_two_locks_on_one_directory_cannot_both_be_held(tmp_path: Path) -> None:
    first = InstanceLock(tmp_path)
    second = InstanceLock(tmp_path)
    try:
        assert first.acquire() is True
        assert second.acquire() is False, (
            "two processes acquired the same data directory: this is exactly the "
            "condition that corrupted a 19 GB download on 2026-08-18"
        )
        assert first.held is True
        assert second.held is False
    finally:
        first.release()
        second.release()


def test_the_lock_is_released_and_can_be_retaken(tmp_path: Path) -> None:
    first = InstanceLock(tmp_path)
    assert first.acquire() is True
    first.release()
    assert first.held is False

    second = InstanceLock(tmp_path)
    try:
        assert second.acquire() is True
    finally:
        second.release()


def test_acquiring_twice_from_the_same_object_is_a_no_op(tmp_path: Path) -> None:
    lock = InstanceLock(tmp_path)
    try:
        assert lock.acquire() is True
        assert lock.acquire() is True
    finally:
        lock.release()


def test_holder_names_the_pid_while_the_lock_is_held(tmp_path: Path) -> None:
    """The payload must stay readable: on Windows a locked byte range is not.

    If the lock byte sat at offset zero this read would fail with a sharing
    violation and ``/health`` could not name the process to stop.
    """
    lock = InstanceLock(tmp_path)
    try:
        assert lock.acquire() is True
        holder = lock.holder()
        assert holder is not None
        assert holder["pid"] == os.getpid()
        assert holder["data_dir"] == str(tmp_path)
        assert holder["started_at"] > 0
        # Readable by a plain reader too, not only through our own handle.
        raw = json.loads((tmp_path / INSTANCE_LOCK_NAME).read_text())
        assert raw["pid"] == os.getpid()
    finally:
        lock.release()


def test_a_stale_lock_file_from_a_crash_is_taken_over(tmp_path: Path) -> None:
    """A crash leaves the file but not the lock; the next start must proceed."""
    stale = tmp_path / INSTANCE_LOCK_NAME
    stale.write_text(
        json.dumps(
            {"pid": 999_999, "create_time": 1.0, "started_at": 1.0, "data_dir": str(tmp_path)}
        )
    )
    lock = InstanceLock(tmp_path)
    try:
        assert lock.holder() is None, "a dead pid is not a holder"
        assert lock.acquire() is True
        assert (lock.holder() or {})["pid"] == os.getpid()
    finally:
        lock.release()


def test_a_recycled_pid_is_not_mistaken_for_the_holder(tmp_path: Path) -> None:
    """Our own pid with somebody else's creation time is somebody else."""
    (tmp_path / INSTANCE_LOCK_NAME).write_text(
        json.dumps({"pid": os.getpid(), "create_time": 1.0, "started_at": 1.0})
    )
    assert InstanceLock(tmp_path).holder() is None


@pytest.mark.parametrize("body", ["", "   ", "not json", '["a list"]', '{"pid": "not an int"}'])
def test_an_unreadable_lock_file_is_not_a_holder(tmp_path: Path, body: str) -> None:
    (tmp_path / INSTANCE_LOCK_NAME).write_text(body)
    lock = InstanceLock(tmp_path)
    assert lock.holder() is None
    try:
        assert lock.acquire() is True
    finally:
        lock.release()


def test_holder_is_none_when_there_is_no_lock_file(tmp_path: Path) -> None:
    assert InstanceLock(tmp_path / "never-created").holder() is None


def test_the_lock_creates_its_data_directory(tmp_path: Path) -> None:
    target = tmp_path / "data" / "nested"
    lock = InstanceLock(target)
    try:
        assert lock.acquire() is True
        assert (target / INSTANCE_LOCK_NAME).is_file()
    finally:
        lock.release()


def test_non_blocking_acquire_returns_immediately(tmp_path: Path) -> None:
    import time

    holder = InstanceLock(tmp_path)
    other = InstanceLock(tmp_path)
    try:
        assert holder.acquire() is True
        started = time.monotonic()
        assert other.acquire() is False
        assert time.monotonic() - started < 1.0, "a non-blocking acquire waited"
    finally:
        holder.release()
        other.release()


def test_blocking_acquire_gives_up_at_the_timeout(tmp_path: Path) -> None:
    holder = InstanceLock(tmp_path)
    other = InstanceLock(tmp_path)
    try:
        assert holder.acquire() is True
        assert other.acquire(blocking=True, timeout_s=0.2, poll_s=0.05) is False
    finally:
        holder.release()
        other.release()


def test_context_manager_releases(tmp_path: Path) -> None:
    with InstanceLock(tmp_path) as lock:
        assert lock.held is True
    after = InstanceLock(tmp_path)
    try:
        assert after.acquire() is True
    finally:
        after.release()


# ---------------------------------------------------------------------------
# The app path
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", models={"dir": tmp_path / "models"})


class _RecordingDownloader:
    """Stands in for the Downloader so "did it resume?" is directly observable."""

    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.disabled_reason: str | None = None

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    def disable_transfers(self, reason: str) -> None:
        self.disabled_reason = reason


def _neutered_state(config: Config, downloader: Any) -> Any:
    """A state object whose background pieces do nothing but count calls."""
    from studioforge.api.app import build_state

    state = build_state(config)
    state.downloader = downloader

    class _Manager:
        draining = False
        started = 0

        async def start(self) -> None:
            type(self).started += 1

        async def stop(self) -> None:
            return None

    class _EngineManager:
        async def ensure_engine(self) -> Any:
            raise RuntimeError("no engine in this test")

        async def aclose(self) -> None:
            return None

    class _Registry:
        def scan(self) -> Any:
            raise RuntimeError("no scan in this test")

    state.manager = _Manager()
    state.engine_manager = _EngineManager()
    state.registry = _Registry()
    return state


def test_a_secondary_instance_does_not_start_the_downloader(config: Config) -> None:
    """The incident, reduced to one assertion.

    A second app against a data directory somebody else owns must not resume the
    download queue -- that is what made a second writer appear inside a live
    ``.part``. ``TestClient`` as a context manager is what runs the lifespan, so
    this really does exercise the startup path rather than only the wiring.
    """
    from fastapi.testclient import TestClient

    config.ensure_dirs()
    owner = InstanceLock(config.data_dir)
    assert owner.acquire() is True
    downloader = _RecordingDownloader()
    try:
        app = create_app(config, state=_neutered_state(config, downloader), start_background=True)
        assert app.state.instance_role == "secondary"
        with TestClient(app) as http:
            body = http.get("/health").json()
        assert body["instance"] == "secondary"
        assert body["instance_holder_pid"] == os.getpid()
        assert downloader.started == 0, (
            "a secondary instance resumed the download queue: two writers, one .part"
        )
        assert app.state.manager.started == 0, "a secondary instance started the TTL sweeper"
        # And it may not *start* one from the API/MCP/GUI either: enqueue is
        # refused with the holder's pid, not merely un-resumed.
        assert downloader.disabled_reason is not None
        assert str(os.getpid()) in downloader.disabled_reason
    finally:
        owner.release()


def test_the_primary_instance_starts_the_downloader_and_frees_the_lock(config: Config) -> None:
    from fastapi.testclient import TestClient

    downloader = _RecordingDownloader()
    app = create_app(config, state=_neutered_state(config, downloader), start_background=True)
    assert app.state.instance_role == "primary"
    with TestClient(app) as http:
        assert http.get("/health").json()["instance"] == "primary"
    assert downloader.started == 1

    # Shutdown released it, so the next process can take over cleanly.
    after = InstanceLock(config.data_dir)
    try:
        assert after.acquire() is True
    finally:
        after.release()


def test_a_read_only_app_never_claims_the_directory(config: Config) -> None:
    """``start_background=False`` composes the graph and takes nothing.

    The stdio MCP server and every unit test build the same object graph. If
    that act took the lock, running one of them would silently demote the real
    server to secondary -- turning a safety feature into an outage.
    """
    app = create_app(config, start_background=False)
    assert app.state.instance_role == "primary"
    assert app.state.instance_lock is None
    lock = InstanceLock(config.data_dir)
    try:
        assert lock.acquire() is True, "building an app locked the data directory"
    finally:
        lock.release()


def test_shutdown_closes_the_supervisor(config: Config) -> None:
    """``Supervisor.aclose()`` is what releases the Windows job object (D23);
    the lifespan never called it before (WP17 F11), so a survivor of
    ``stop_all`` was only ever reaped by the interpreter-exit safety net."""
    from fastapi.testclient import TestClient

    downloader = _RecordingDownloader()
    state = _neutered_state(config, downloader)

    class _Supervisor:
        closed = 0

        def list(self) -> list[Any]:
            return []

        def child_pids(self) -> set[int]:
            return set()

        async def aclose(self) -> None:
            type(self).closed += 1

    state.supervisor = _Supervisor()
    app = create_app(config, state=state, start_background=True)
    with TestClient(app) as http:
        assert http.get("/health").status_code == 200
    assert _Supervisor.closed == 1, "the supervisor must be closed exactly once at shutdown"
