"""Free-space reporting for the Download tab.

``shutil.disk_usage`` is monkeypatched throughout: the real answer depends on
whichever volume the test runner happens to live on, and a threshold test that
passes on a 4 TB rig and fails on a full CI box tests nothing.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from studioforge.core.diskspace import (
    LOW_FREE_BYTES,
    LOW_FREE_FRACTION,
    clear_cache,
    disk_report,
)

GIB = 1024**3
TIB = 1024**4

EXPECTED_KEYS = {
    "path",
    "drive",
    "total_bytes",
    "free_bytes",
    "queued_bytes",
    "free_after_queue_bytes",
    "low",
    "error",
}


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """The 2 s memo is process-global; a leaked entry would cross tests."""
    clear_cache()
    yield
    clear_cache()


def fake_usage(
    monkeypatch: pytest.MonkeyPatch, *, total: int, free: int, seen: list[Path] | None = None
) -> None:
    def _usage(path: Any) -> tuple[int, int, int]:
        if seen is not None:
            seen.append(Path(path))
        return (total, total - free, free)

    monkeypatch.setattr(shutil, "disk_usage", _usage)


# ---------------------------------------------------------------------------
# Shape and arithmetic
# ---------------------------------------------------------------------------


def test_report_has_exactly_the_documented_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    assert set(disk_report(tmp_path, 0)) == EXPECTED_KEYS


def test_the_queue_is_subtracted_from_free(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    report = disk_report(tmp_path, 120 * GIB)
    assert report["free_bytes"] == 500 * GIB
    assert report["queued_bytes"] == 120 * GIB
    assert report["free_after_queue_bytes"] == 380 * GIB
    assert report["low"] is False


def test_an_overrunning_queue_reports_a_negative_remainder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed on purpose: "40 GiB short" and "nothing left" are different facts."""
    fake_usage(monkeypatch, total=1 * TIB, free=30 * GIB)
    report = disk_report(tmp_path, 70 * GIB)
    assert report["free_after_queue_bytes"] == -40 * GIB
    assert report["low"] is True


def test_a_negative_or_missing_queue_is_clamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    assert disk_report(tmp_path, -5)["queued_bytes"] == 0
    assert disk_report(tmp_path, 0)["free_after_queue_bytes"] == 500 * GIB


def test_the_reported_path_is_the_directory_that_was_asked_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    target = tmp_path / "models"
    target.mkdir()
    assert Path(disk_report(target, 0)["path"]) == target.resolve()


def test_the_drive_label_is_short_enough_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``E:`` on Windows, a mount point on POSIX -- never the full path."""
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    drive = disk_report(tmp_path, 0)["drive"]
    assert drive
    assert len(drive) < len(str(tmp_path))


# ---------------------------------------------------------------------------
# The "low" threshold
# ---------------------------------------------------------------------------


def test_low_below_the_fixed_floor_on_a_small_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 100 GiB volume: 5% is 5 GiB, so the 10 GiB floor is what bites."""
    fake_usage(monkeypatch, total=100 * GIB, free=9 * GIB)
    assert disk_report(tmp_path, 0)["low"] is True
    clear_cache()
    fake_usage(monkeypatch, total=100 * GIB, free=11 * GIB)
    assert disk_report(tmp_path, 0)["low"] is False


def test_low_below_five_percent_on_a_large_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4 TB library drive with 50 GiB left is in trouble the floor would miss."""
    fake_usage(monkeypatch, total=4 * TIB, free=50 * GIB)
    report = disk_report(tmp_path, 0)
    assert report["free_after_queue_bytes"] > LOW_FREE_BYTES
    assert report["low"] is True
    assert report["free_after_queue_bytes"] < int(4 * TIB * LOW_FREE_FRACTION)


def test_low_is_measured_after_the_queue_not_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: 200 GiB free is fine until 195 GiB of it is spoken for."""
    fake_usage(monkeypatch, total=1 * TIB, free=200 * GIB)
    assert disk_report(tmp_path, 0)["low"] is False
    assert disk_report(tmp_path, 195 * GIB)["low"] is True


def test_the_boundary_is_not_low(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_usage(monkeypatch, total=100 * GIB, free=LOW_FREE_BYTES)
    assert disk_report(tmp_path, 0)["low"] is False


# ---------------------------------------------------------------------------
# Missing directories
# ---------------------------------------------------------------------------


def test_a_missing_models_dir_walks_up_to_an_existing_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing has been downloaded yet, so ``models.dir`` does not exist.

    The volume underneath it does, and that is the question being asked.
    """
    seen: list[Path] = []
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB, seen=seen)
    missing = tmp_path / "models" / "not" / "created" / "yet"
    report = disk_report(missing, 0)
    assert report["total_bytes"] == 1 * TIB
    assert seen == [tmp_path.resolve()]
    # The path reported is still the one the user configured, not the ancestor
    # the measurement was taken on.
    assert Path(report["path"]) == missing.resolve()


def test_an_unmapped_drive_reports_the_reason_and_is_not_called_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "I could not tell" must not render as "you are about to run out"."""

    def _boom(_path: Any) -> tuple[int, int, int]:
        raise OSError("[WinError 21] The device is not ready")

    monkeypatch.setattr(shutil, "disk_usage", _boom)
    report = disk_report(tmp_path, 40 * GIB)
    assert report["total_bytes"] == 0
    assert report["free_bytes"] == 0
    assert report["free_after_queue_bytes"] == 0
    assert report["low"] is False
    assert "not ready" in str(report["error"])
    # The queue figure came from us, not from the volume, so it survives.
    assert report["queued_bytes"] == 40 * GIB


def test_a_failure_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A drive that comes back must be reported as back on the next poll."""
    calls: list[int] = []

    def _boom(_path: Any) -> tuple[int, int, int]:
        calls.append(1)
        raise OSError("gone")

    monkeypatch.setattr(shutil, "disk_usage", _boom)
    disk_report(tmp_path, 0)
    disk_report(tmp_path, 0)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_repeated_reports_cost_one_syscall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The GUI polls twice a second for hours; the volume is asked once per 2 s."""
    seen: list[Path] = []
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB, seen=seen)
    for _ in range(20):
        disk_report(tmp_path, 0)
    assert len(seen) == 1


def test_the_cache_expires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from studioforge.core import diskspace

    seen: list[Path] = []
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB, seen=seen)
    clock = [1000.0]
    monkeypatch.setattr(diskspace.time, "monotonic", lambda: clock[0])
    disk_report(tmp_path, 0)
    clock[0] += diskspace.DISK_USAGE_TTL_S + 0.1
    disk_report(tmp_path, 0)
    assert len(seen) == 2


def test_clear_cache_forces_a_fresh_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    assert disk_report(tmp_path, 0)["free_bytes"] == 500 * GIB
    clear_cache()
    fake_usage(monkeypatch, total=1 * TIB, free=9 * GIB)
    assert disk_report(tmp_path, 0)["free_bytes"] == 9 * GIB


def test_two_directories_on_different_paths_are_cached_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB, seen=seen)
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    disk_report(first, 0)
    disk_report(second, 0)
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# The downloader side of the figure
# ---------------------------------------------------------------------------


def test_queued_remaining_bytes_counts_only_what_is_still_coming() -> None:
    """Remainders, not totals -- and nothing a human has to restart by hand."""
    from studioforge.core.downloader import Downloader, _FileState

    downloader = Downloader.__new__(Downloader)  # no DB, no HTTP client needed
    states = {
        "a": _FileState(
            id="a",
            group_id="g",
            repo_id="r",
            filename="a.gguf",
            dest=Path("a.gguf"),
            status="running",
            total_bytes=10 * GIB,
            downloaded_bytes=4 * GIB,
        ),
        "b": _FileState(
            id="b",
            group_id="g",
            repo_id="r",
            filename="b.gguf",
            dest=Path("b.gguf"),
            status="queued",
            total_bytes=20 * GIB,
        ),
        "c": _FileState(
            id="c",
            group_id="g",
            repo_id="r",
            filename="c.gguf",
            dest=Path("c.gguf"),
            status="paused",
            total_bytes=5 * GIB,
            downloaded_bytes=1 * GIB,
        ),
        "d": _FileState(
            id="d",
            group_id="g",
            repo_id="r",
            filename="d.gguf",
            dest=Path("d.gguf"),
            status="completed",
            total_bytes=30 * GIB,
            downloaded_bytes=30 * GIB,
        ),
        "e": _FileState(
            id="e",
            group_id="g",
            repo_id="r",
            filename="e.gguf",
            dest=Path("e.gguf"),
            status="failed",
            total_bytes=40 * GIB,
        ),
        "f": _FileState(
            id="f",
            group_id="g",
            repo_id="r",
            filename="f.gguf",
            dest=Path("f.gguf"),
            status="canceled",
            total_bytes=50 * GIB,
        ),
    }
    downloader._files = states  # noqa: SLF001 - the point of the test
    assert downloader.queued_remaining_bytes() == (6 + 20 + 4) * GIB


def test_queued_remaining_bytes_is_zero_with_nothing_queued() -> None:
    from studioforge.core.downloader import Downloader

    downloader = Downloader.__new__(Downloader)
    downloader._files = {}  # noqa: SLF001
    assert downloader.queued_remaining_bytes() == 0


def test_an_over_delivered_file_cannot_go_negative() -> None:
    """A .part longer than the declared size (the 2026-08-18 shape) reads as 0."""
    from studioforge.core.downloader import Downloader, _FileState

    downloader = Downloader.__new__(Downloader)
    downloader._files = {  # noqa: SLF001
        "a": _FileState(
            id="a",
            group_id="g",
            repo_id="r",
            filename="a.gguf",
            dest=Path("a.gguf"),
            status="running",
            total_bytes=19 * GIB,
            downloaded_bytes=22 * GIB,
        )
    }
    assert downloader.queued_remaining_bytes() == 0
