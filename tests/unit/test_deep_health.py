"""A shallow health check lies; the deep probe has to actually generate.

Incident (production, the system this replaces): *"the control channel reported
'connected' and GET /models returned 200 for hours while every long stream died
mid-generation."* Short streams completed, so every dashboard was green. The
fix that finally caught it was a data-channel pre-flight -- a real streamed
completion -- because a 200 from a status endpoint proves nothing about whether
generation works.

These tests pin the properties that make the probe worth having:

* it streams, and a stream that stops without its terminator is a FAILURE;
* it cannot pass when there is nothing loaded (``no_models_loaded``);
* it never raises and never hangs, because a health check that takes the server
  down with it is worse than no health check.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.gpu import reset_probe
from studioforge.core.health import deep_health, probe_model
from studioforge.types import GgufMeta, InstanceInfo, ModelRecord

BASE = "http://127.0.0.1:18100"


def sse(*chunks: str) -> list[str]:
    """SSE lines for content deltas, without a terminator."""
    return ["data: " + json.dumps({"choices": [{"delta": {"content": chunk}}]}) for chunk in chunks]


DONE = "data: [DONE]"


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


class FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200, delay: float = 0.0) -> None:
        self.status_code = status_code
        self._lines = lines
        self._delay = delay

    async def __aenter__(self) -> FakeStreamResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def aiter_lines(self) -> Any:
        for line in self._lines:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield line

    async def aread(self) -> bytes:
        return b'{"error": {"message": "no"}}'


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload


class FakeClient:
    """Stands in for the shared ``httpx.AsyncClient``."""

    def __init__(
        self,
        *,
        lines: list[str] | None = None,
        status_code: int = 200,
        delay: float = 0.0,
        raises: Exception | None = None,
        embedding: dict[str, Any] | None = None,
    ) -> None:
        self.lines = lines or []
        self.status_code = status_code
        self.delay = delay
        self.raises = raises
        self.embedding = embedding
        self.stream_calls: list[dict[str, Any]] = []

    def stream(self, method: str, url: str, **kwargs: Any) -> FakeStreamResponse:
        if self.raises is not None:
            raise self.raises
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return FakeStreamResponse(self.lines, self.status_code, self.delay)

    async def post(self, url: str, **kwargs: Any) -> FakeResponse:
        return FakeResponse(self.embedding or {}, self.status_code)


class FakeSupervisor:
    def __init__(self, instances: dict[str, str]) -> None:
        #: model_id -> state
        self._instances = instances

    def base_url(self, model_id: str) -> str | None:
        return BASE if model_id in self._instances else None

    def is_ready(self, model_id: str) -> bool:
        return self._instances.get(model_id) == "ready"

    def list(self) -> list[InstanceInfo]:
        return [
            InstanceInfo(model_id=mid, state=state, port=18100)  # type: ignore[arg-type]
            for mid, state in self._instances.items()
        ]


# ---------------------------------------------------------------------------
# probe_model
# ---------------------------------------------------------------------------


async def test_a_healthy_stream_passes() -> None:
    client = FakeClient(lines=[*sse("o", "k", "!"), DONE])

    result = await probe_model("m", BASE, client)  # type: ignore[arg-type]

    assert result.ok is True
    assert result.tokens == 3
    assert result.first_token_ms is not None
    assert result.error is None


async def test_the_probe_actually_streams() -> None:
    """It must be a streamed completion, deterministic and small.

    A non-streamed request would not exercise the path that broke: the
    incident's short (non-streamed-feeling) calls all worked.
    """
    client = FakeClient(lines=[*sse("ok"), DONE])

    await probe_model("m", BASE, client, max_tokens=8)  # type: ignore[arg-type]

    payload = client.stream_calls[0]["json"]
    assert payload["stream"] is True
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 8


async def test_a_stream_cut_mid_generation_fails() -> None:
    """THE incident: tokens arrive, then the stream dies without terminating.

    A check that only asserted "some tokens arrived" would have called this
    healthy for hours, which is exactly what happened in production.
    """
    client = FakeClient(lines=sse("th", "in", "king"))  # no [DONE]

    result = await probe_model("m", BASE, client)  # type: ignore[arg-type]

    assert result.ok is False
    assert result.tokens == 3
    assert "[DONE]" in (result.error or "")


async def test_a_stream_with_no_content_fails() -> None:
    client = FakeClient(lines=[DONE])

    result = await probe_model("m", BASE, client)  # type: ignore[arg-type]

    assert result.ok is False
    assert result.tokens == 0
    assert "generated nothing" in (result.error or "")


async def test_a_non_200_fails_without_raising() -> None:
    client = FakeClient(lines=[], status_code=503)

    result = await probe_model("m", BASE, client)  # type: ignore[arg-type]

    assert result.ok is False
    assert "503" in (result.error or "")


async def test_a_dead_connection_is_recorded_not_raised() -> None:
    """A failure is data. A probe that raises just relocates the outage."""
    client = FakeClient(raises=OSError("connection refused"))

    result = await probe_model("m", BASE, client)  # type: ignore[arg-type]

    assert result.ok is False
    assert "connection refused" in (result.error or "")


async def test_a_hung_stream_hits_the_hard_timeout() -> None:
    """A probe must never wedge: a model that stops emitting is bounded."""
    client = FakeClient(lines=[*sse("a"), *sse("b"), DONE], delay=5.0)

    result = await probe_model("m", BASE, client, timeout_s=0.1)  # type: ignore[arg-type]

    assert result.ok is False
    assert "timed out" in (result.error or "")
    assert result.duration_ms is not None and result.duration_ms < 3000


async def test_an_embedding_model_is_probed_as_an_embedding() -> None:
    client = FakeClient(embedding={"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    result = await probe_model("e", BASE, client, kind="embedding")  # type: ignore[arg-type]

    assert result.ok is True
    assert result.embedding_dims == 3


# ---------------------------------------------------------------------------
# deep_health aggregation
# ---------------------------------------------------------------------------


async def test_nothing_loaded_is_not_a_pass() -> None:
    """A probe that cannot fail is worse than none: say so instead."""
    result = await deep_health(FakeSupervisor({}), FakeClient())  # type: ignore[arg-type]

    assert result.status == "no_models_loaded"
    assert result.checked == 0


async def test_all_healthy_is_ok() -> None:
    supervisor = FakeSupervisor({"a": "ready", "b": "ready"})
    client = FakeClient(lines=[*sse("ok"), DONE])

    result = await deep_health(supervisor, client)  # type: ignore[arg-type]

    assert result.status == "ok"
    assert result.checked == 2
    assert [m.model_id for m in result.models] == ["a", "b"]


async def test_one_failure_degrades_the_whole_verdict() -> None:
    supervisor = FakeSupervisor({"a": "ready"})
    client = FakeClient(lines=sse("cut"))  # no terminator

    result = await deep_health(supervisor, client)  # type: ignore[arg-type]

    assert result.status == "degraded"
    assert result.models[0].ok is False


async def test_a_loading_model_is_not_probed() -> None:
    """Only *ready* instances are probed; a loading one is not a failure."""
    supervisor = FakeSupervisor({"a": "loading"})

    result = await deep_health(supervisor, FakeClient())  # type: ignore[arg-type]

    assert result.status == "no_models_loaded"


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    reset_probe()
    yield
    reset_probe()


class FakeRegistry:
    def __init__(self, records: dict[str, ModelRecord]) -> None:
        self._records = records

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def resolve(self, name: str) -> ModelRecord | None:
        return self._records.get(name)

    def known_ids(self) -> list[str]:
        return sorted(self._records)

    def scan(self, *, force: bool = False) -> Any:  # pragma: no cover - lifespan only
        raise RuntimeError("not used")


def make_app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    return create_app(config, state=build_state(config), start_background=False)


def make_record(model_id: str = "vendor/thing") -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path=Path("/models/thing.gguf"),
        meta=GgufMeta(architecture="llama", n_layer=8),
    )


def test_shallow_health_stays_shallow_and_free(tmp_path: Path) -> None:
    """The watchdog and load balancers poll this constantly: no inference."""
    app = make_app(tmp_path)
    with TestClient(app) as http:
        app.state.supervisor = FakeSupervisor({"vendor/thing": "ready"})
        app.state.client = FakeClient(lines=[*sse("ok"), DONE])

        body = http.get("/health").json()

    assert body["status"] == "ok"
    assert "probe" not in body
    assert app.state.client.stream_calls == []


def test_deep_health_reports_a_broken_stream_as_degraded(tmp_path: Path) -> None:
    """The endpoint the incident needed: 200 everywhere, generation broken."""
    app = make_app(tmp_path)
    with TestClient(app) as http:
        app.state.supervisor = FakeSupervisor({"vendor/thing": "ready"})
        app.state.client = FakeClient(lines=sse("half a repl"))  # cut mid-stream

        body = http.get("/health?deep=true").json()

    assert body["status"] == "degraded"
    assert body["probe"]["models"][0]["ok"] is False
    assert body["probe"]["checked"] == 1


def test_deep_health_with_nothing_loaded_says_so(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as http:
        app.state.supervisor = FakeSupervisor({})
        app.state.client = FakeClient()

        body = http.get("/health?deep=true").json()

    assert body["status"] == "no_models_loaded"
    assert body["probe"]["checked"] == 0


def test_single_model_probe_endpoint(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    record = make_record()
    with TestClient(app) as http:
        app.state.registry = FakeRegistry({record.id: record})
        app.state.supervisor = FakeSupervisor({record.id: "ready"})
        app.state.client = FakeClient(lines=[*sse("o", "k"), DONE])

        response = http.post(f"/api/models/{record.id}/probe")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["model_id"] == record.id
    assert body["models"][0]["tokens"] == 2


def test_probing_an_unloaded_model_does_not_load_it(tmp_path: Path) -> None:
    """A monitoring poll must never trigger minutes of VRAM allocation."""
    app = make_app(tmp_path)
    record = make_record()
    with TestClient(app) as http:
        app.state.registry = FakeRegistry({record.id: record})
        app.state.supervisor = FakeSupervisor({})
        app.state.client = FakeClient()

        body = http.post(f"/api/models/{record.id}/probe").json()

    assert body["status"] == "degraded"
    assert "not loaded" in body["models"][0]["error"]


def test_probing_an_unknown_model_is_a_404(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app) as http:
        app.state.registry = FakeRegistry({})
        app.state.supervisor = FakeSupervisor({})
        app.state.client = FakeClient()

        response = http.post("/api/models/nope/probe")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"
