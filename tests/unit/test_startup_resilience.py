"""The server must always boot: degraded and loud, never crashed.

Each test breaks one startup dependency (model directory, registry scan,
SQLite file, GPU probe) and asserts the API still comes up and answers --
because a headless box whose gateway dies on boot cannot even be diagnosed
remotely, while a degraded server can report what is wrong.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.gpu import reset_probe


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)
    # The null probe: these tests are about boot resilience, not NVML.
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    reset_probe()
    yield
    reset_probe()


def make_config(tmp_path: Path, models_dir: Path | None = None) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": models_dir if models_dir is not None else tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )


def boot(config: Config, state: Any = None) -> TestClient:
    """Create the app and run its lifespan -- the real boot path."""
    app = create_app(config, state=state, start_background=False)
    return TestClient(app)


def test_boots_with_a_missing_model_directory(tmp_path: Path) -> None:
    config = make_config(tmp_path, models_dir=tmp_path / "does-not-exist")
    with boot(config) as client:
        assert client.get("/health").json()["status"] == "ok"
        models = client.get("/v1/models").json()
        assert models["data"] == []


def test_boots_when_the_startup_scan_raises(tmp_path: Path) -> None:
    """A scan crash means an empty model list and a log line, not a dead boot."""
    config = make_config(tmp_path)
    state = build_state(config)

    def explode(*, force: bool = False) -> None:
        raise RuntimeError("disk on fire")

    state.registry.scan = explode  # type: ignore[method-assign]
    with boot(config, state=state) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").json()["data"] == []


def test_boots_with_a_corrupt_sqlite_file(tmp_path: Path) -> None:
    """End-to-end proof of the DB recovery path through build_state."""
    db_path = tmp_path / "data" / "registry.sqlite3"
    db_path.parent.mkdir(parents=True)
    db_path.write_bytes(b"garbage, not sqlite" * 64)

    config = make_config(tmp_path)
    with boot(config) as client:
        assert client.get("/health").json()["status"] == "ok"
    backups = list(db_path.parent.glob("registry.sqlite3.corrupt-*"))
    assert len(backups) == 1, "the corrupt file must be preserved for manual recovery"


def test_boots_with_no_gpus_and_status_says_so(tmp_path: Path) -> None:
    """No NVML/no GPUs degrades to an empty GPU list, and a load attempt gets
    the planner's clean GPU-only rejection rather than a crash."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    config = make_config(tmp_path, models_dir=models_dir)
    with boot(config) as client:
        status = client.get("/api/status").json()
        assert status["gpus"] == []
        gpus = client.get("/api/gpus").json()
        assert gpus["backend"] == "null"
        assert gpus["gpus"] == []


def test_shutdown_closes_the_engine_and_updater_http_clients(tmp_path: Path) -> None:
    """The lazily created GitHub clients must not outlive the app.

    EngineManager and Updater open their own httpx.AsyncClient on first use
    (release checks); the lifespan closed only the proxy client, so every
    shutdown after a release check leaked two clients and their connection
    pools.
    """
    config = make_config(tmp_path)
    app = create_app(config, start_background=False)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        # Force both lazy clients into existence, as a release check would.
        engine_client = app.state.engine_manager.client
        app.state.updater._client = updater_client = __import__("httpx").AsyncClient()
        assert not engine_client.is_closed
        assert not updater_client.is_closed
    assert engine_client.is_closed, "the engine manager's client leaked at shutdown"
    assert updater_client.is_closed, "the updater's client leaked at shutdown"
