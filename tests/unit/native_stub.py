"""An in-process stub llama-server child and a ready-resident app, for the WP5a suites.

No process, no port, no engine: the stub is an ``httpx.MockTransport`` that
answers the child endpoints deterministically and records every request, and the
app is the real FastAPI application with the registry and supervisor swapped for
in-memory ones whose ``base_url`` points at the stub. Everything between the
route and the child -- selector resolution, JIT, in-flight counting, forwarding,
headers -- is the real code.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.types import (
    GgufMeta,
    InFlightRequest,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelKind,
    ModelRecord,
)
from tests.unit.test_catalog_routes import FakeProbe, FakeRegistry

CHILD = "http://stub-child.invalid"


class _Chunks(httpx.AsyncByteStream):
    """A response body delivered as a real stream (a ``content=`` body is pre-read by httpx)."""

    def __init__(self, *chunks: bytes) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> Any:
        for chunk in self._chunks:
            yield chunk


def sse(*chunks: bytes, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status, headers={"content-type": "text/event-stream"}, stream=_Chunks(*chunks)
    )


#: The llama-server /props shape, trimmed to what the tests read.
PROPS = {
    "default_generation_settings": {"id": 0, "n_ctx": 8192, "params": {"n_predict": -1}},
    "total_slots": 2,
    "model_path": "/models/thing.gguf",
    "build_info": "b12110-stub",
}


def record(
    model_id: str,
    *,
    kind: ModelKind = "chat",
    param_count: int = 8_000_000_000,
    vision: bool = False,
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id.rsplit("/", 1)[-1],
        path=Path(f"/models/{model_id.replace('/', '_')}.gguf"),
        kind=kind,
        size_bytes=4_000_000_000,
        capabilities=ModelCapabilities(vision=vision, embedding=kind == "embedding"),
        meta=GgufMeta(
            architecture="llama",
            n_ctx_train=32768,
            n_vocab=32000,
            n_embd=4096,
            param_count=param_count,
        ),
    )


def instance(
    model_id: str,
    *,
    state: str = "ready",
    engine: str | None = "b12110",
    ctx: int = 8192,
    parallel: int = 2,
) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state=state,  # type: ignore[arg-type]
        resolved_engine_tag=engine,
        launch_args=["llama-server", "-c", str(ctx)],
        plan=LoadPlan(
            model_id=model_id,
            devices=[0],
            ctx_size=ctx,
            parallel=parallel,
            ctx_per_slot=ctx,
            max_parallel=parallel,
        ),
    )


class StubChild:
    """The child endpoints, recorded. ``routes`` overrides an answer per ``(method, path)``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.routes: dict[tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def install(self, monkeypatch: Any) -> StubChild:
        """Make the app's shared ``httpx.AsyncClient`` (built in its lifespan) talk to this stub."""
        real = httpx.AsyncClient
        handler = self.handle

        def factory(**kwargs: Any) -> httpx.AsyncClient:
            kwargs.pop("limits", None)
            return real(transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr("studioforge.api.app.httpx.AsyncClient", factory)
        return self

    def last(self) -> httpx.Request:
        return self.requests[-1]

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content or b"{}")

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        override = self.routes.get((request.method, request.url.path))
        if override is not None:
            return override(request)
        path = request.url.path
        if request.method == "GET" and path == "/props":
            return httpx.Response(200, json=PROPS)
        if request.method == "GET" and path == "/slots":
            return httpx.Response(200, json=[{"id": 0, "is_processing": False}])
        body = json.loads(request.content or b"{}") if request.content else {}
        if path == "/tokenize":
            return httpx.Response(200, json={"tokens": [1, 2, 3]})
        if path == "/detokenize":
            return httpx.Response(200, json={"content": "abc"})
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<s>hi"})
        if path in {"/completion", "/infill"}:
            if body.get("stream"):
                return sse(
                    b'data: {"content":"he","stop":false}\n\n',
                    b'data: {"content":"llo","stop":true}\n\n',
                )
            return httpx.Response(200, json={"content": "hello", "model": body.get("model")})
        if path == "/embedding":
            return httpx.Response(200, json=[{"index": 0, "embedding": [[0.1, 0.2]]}])
        if path in {"/reranking", "/rerank"}:
            return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})
        if path == "/v1/chat/completions":
            if body.get("stream"):
                return sse(
                    b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"
                )
            return httpx.Response(
                200,
                json={
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                },
            )
        if path in {"/v1/completions", "/v1/embeddings", "/v1/rerank"}:
            return httpx.Response(200, json={"object": "list", "data": []})
        return httpx.Response(404, json={"error": {"code": 404, "message": "stub: no such path"}})


class Supervisor:
    """What the routes read from the supervisor, over a plain list of instances."""

    def __init__(self, instances: list[InstanceInfo]) -> None:
        self.instances = instances
        self.started: list[str] = []
        self.ended: list[str] = []
        self.in_flight_seen: list[int] = []

    def list(self) -> list[InstanceInfo]:
        return list(self.instances)

    def get(self, model_id: str) -> InstanceInfo | None:
        return next((i for i in self.instances if i.model_id == model_id), None)

    def is_ready(self, model_id: str) -> bool:
        found = self.get(model_id)
        return found is not None and found.state == "ready"

    def base_url(self, model_id: str) -> str | None:
        return CHILD if self.get(model_id) is not None else None

    def mark_request_start(self, model_id: str, *, client: str | None = None) -> str:
        found = self.get(model_id)
        if found is not None:
            found.active_requests += 1
            found.in_flight.append(InFlightRequest(id=f"r{len(self.started)}", started_at=0.0))
            self.in_flight_seen.append(found.active_requests)
        self.started.append(model_id)
        return f"r{len(self.started) - 1}"

    def mark_request_end(self, model_id: str, **_kwargs: Any) -> None:
        found = self.get(model_id)
        if found is not None:
            found.active_requests = max(0, found.active_requests - 1)
            found.in_flight.clear()
        self.ended.append(model_id)

    def tail_log(self, model_id: str, n: int = 200) -> list[str]:
        return []


class Registry(FakeRegistry):
    """``FakeRegistry`` plus ``touch`` (called by ``ensure_loaded``)."""

    def touch(self, model_id: str) -> None:
        return None


def build_app(
    tmp_path: Path,
    records: list[ModelRecord],
    instances: list[InstanceInfo],
    child: StubChild,  # noqa: ARG001 - installed by the fixture; kept so call sites read as a unit
    **config_overrides: Any,
) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234, **config_overrides.pop("server", {})},
        models={"dir": tmp_path / "models", **config_overrides.pop("models", {})},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
        **config_overrides,
    )
    app = create_app(config, state=build_state(config), start_background=False)
    registry = Registry(records)
    supervisor = Supervisor(instances)
    app.state.registry = registry
    app.state.probe = FakeProbe()
    app.state.planner.probe = FakeProbe()
    app.state.supervisor = supervisor
    app.state.manager.registry = registry
    app.state.manager.supervisor = supervisor
    return app
