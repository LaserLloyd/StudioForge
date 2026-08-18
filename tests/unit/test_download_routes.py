"""``GET /api/downloads``: the queue, its retry state, and the disk under it.

Two things this route got wrong and now must not get wrong again:

* it hand-built its payload and so silently dropped the retry/resume fields
  WP11 added to :class:`~studioforge.core.downloader.DownloadProgress`, leaving
  an API client unable to tell a backing-off download from a stalled one;
* it could not answer "will this even fit on the disk", which is the question
  the GUI is now asking it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.diskspace import clear_cache
from studioforge.core.downloader import DownloadProgress
from studioforge.errors import ConfigError

GIB = 1024**3
TIB = 1024**4


def make_progress(
    download_id: str = "d1",
    *,
    status: str = "running",
    attempt: int = 0,
    next_retry_at: float | None = None,
    last_error: str | None = None,
    part_bytes: int = 0,
) -> DownloadProgress:
    return DownloadProgress(
        id=download_id,
        group_id="g1",
        repo_id="unsloth/Qwen3.8-27B-GGUF",
        filename=f"{download_id}.gguf",
        status=status,  # type: ignore[arg-type]
        downloaded_bytes=4 * GIB,
        total_bytes=20 * GIB,
        speed_bps=50_000_000.0,
        eta_s=320.0,
        error=None,
        attempt=attempt,
        next_retry_at=next_retry_at,
        last_error=last_error,
        part_bytes=part_bytes,
    )


class FakeDownloader:
    """Only the four methods the route touches."""

    def __init__(
        self,
        models_dir: Path | None,
        progress: list[DownloadProgress],
        queued: int = 0,
    ) -> None:
        self._dir = models_dir
        self._progress = progress
        self._queued = queued

    def all(self) -> list[DownloadProgress]:
        return list(self._progress)

    def active(self) -> list[DownloadProgress]:
        return [p for p in self._progress if p.status in ("queued", "running")]

    def models_dir(self) -> Path:
        if self._dir is None:
            raise ConfigError("no model directory configured", param="models.dir")
        return self._dir

    def queued_remaining_bytes(self) -> int:
        return self._queued


@pytest.fixture(autouse=True)
def _fresh_cache() -> Any:
    clear_cache()
    yield
    clear_cache()


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    return create_app(config, state=build_state(config), start_background=False)


def fake_usage(monkeypatch: pytest.MonkeyPatch, *, total: int, free: int) -> None:
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: (total, total - free, free))


# ---------------------------------------------------------------------------
# The disk block
# ---------------------------------------------------------------------------


def test_downloads_carries_a_disk_block(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    app.state.downloader = FakeDownloader(tmp_path / "models", [make_progress()], 120 * GIB)
    with TestClient(app) as http:
        body = http.get("/api/downloads").json()

    disk = body["disk"]
    assert disk["free_bytes"] == 500 * GIB
    assert disk["queued_bytes"] == 120 * GIB
    assert disk["free_after_queue_bytes"] == 380 * GIB
    assert disk["low"] is False
    assert disk["drive"]


def test_the_disk_block_reports_low_after_the_queue(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=50 * GIB)
    app.state.downloader = FakeDownloader(tmp_path / "models", [make_progress()], 45 * GIB)
    with TestClient(app) as http:
        disk = http.get("/api/downloads").json()["disk"]
    assert disk["low"] is True


def test_the_list_shape_is_unchanged(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``disk`` is additive: the CLI and GUI read ``downloads``/``active``."""
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    running = make_progress("d1", status="running")
    done = make_progress("d2", status="completed")
    app.state.downloader = FakeDownloader(tmp_path / "models", [running, done])
    with TestClient(app) as http:
        body = http.get("/api/downloads").json()

    assert set(body) == {"downloads", "active", "disk"}
    assert [d["id"] for d in body["downloads"]] == ["d1", "d2"]
    assert [d["id"] for d in body["active"]] == ["d1"]


def test_an_unmeasurable_volume_sends_null_not_zeroes(app: Any) -> None:
    """No ``models.dir`` yet. A zeroed block would render as a full disk."""
    app.state.downloader = FakeDownloader(None, [make_progress()])
    with TestClient(app) as http:
        body = http.get("/api/downloads").json()
    assert body["disk"] is None
    assert body["downloads"]


# ---------------------------------------------------------------------------
# The WP11 retry keys
# ---------------------------------------------------------------------------


def test_the_retry_fields_reach_the_wire(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    backing_off = make_progress(
        "d1",
        attempt=2,
        next_retry_at=9_999_999_999.0,
        last_error="ReadTimeout: the CDN hung up",
    )
    app.state.downloader = FakeDownloader(tmp_path / "models", [backing_off])
    with TestClient(app) as http:
        row = http.get("/api/downloads").json()["downloads"][0]

    assert row["attempt"] == 2
    assert row["max_attempts"] >= 1
    assert row["next_retry_at"] == 9_999_999_999.0
    assert row["retry_in_s"] > 0
    assert "CDN hung up" in row["last_error"]


def test_a_failed_row_says_what_resume_would_keep(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    failed = make_progress("d1", status="failed", part_bytes=19 * GIB)
    app.state.downloader = FakeDownloader(tmp_path / "models", [failed])
    with TestClient(app) as http:
        row = http.get("/api/downloads").json()["downloads"][0]
    assert row["part_bytes"] == 19 * GIB


def test_the_payload_does_not_drift_from_the_progress_object(
    app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The drift guard.

    ``_progress_payload`` is hand-built, which is how the retry fields went
    missing from the API for a whole work package while the GUI (which calls
    ``to_dict()``) had them. Pinning the two key sets together means the next
    field added to one is a failing test rather than a silent omission.
    """
    fake_usage(monkeypatch, total=1 * TIB, free=500 * GIB)
    progress = make_progress()
    app.state.downloader = FakeDownloader(tmp_path / "models", [progress])
    with TestClient(app) as http:
        row = http.get("/api/downloads").json()["downloads"][0]
    assert set(row) == set(progress.to_dict())
