"""Shared harness for the compatibility-surface tests (W2).

The pattern is the one ``test_request_priority.py`` and
``test_model_alias_loaded.py`` established: the real FastAPI app (real
middleware, real routers, real manager) with a fake registry and supervisor,
and the engine itself stubbed at the two seams where the gateway talks to
llama-server -- ``_forward`` (non-streamed) and ``_stream_upstream``
(streamed). Nothing here starts a process, binds a port beyond the in-process
``TestClient``, or touches a GPU.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.types import GB, GgufMeta, InstanceInfo, ModelCapabilities, ModelRecord
from tests.unit.test_catalog_routes import (
    MODEL_ID,
    FakeProbe,
    FakeRegistry,
    FakeSupervisor,
    make_plan,
    make_record,
)

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "lmstudio"
EMBED_ID = "vendor/embed-Q8_0"
VISION_ID = "vendor/seer-Q4_K_M"


def load_fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class Registry(FakeRegistry):
    """``FakeRegistry`` plus the ``touch`` every JIT path calls."""

    def touch(self, model_id: str) -> None:
        return None


class Supervisor(FakeSupervisor):
    """``FakeSupervisor`` plus what the chat path asks of a real one."""

    def base_url(self, model_id: str) -> str | None:
        return "http://127.0.0.1:1/fake"

    def mark_request_start(self, model_id: str, *, client: str | None = None) -> str:
        return "req-1"

    def mark_request_end(self, model_id: str, **_kwargs: Any) -> None:
        return None

    def remove(self, model_id: str) -> None:
        self._instances = [i for i in self._instances if i.model_id != model_id]


def make_embedding() -> ModelRecord:
    return ModelRecord(
        id=EMBED_ID,
        name="embed",
        kind="embedding",
        path=Path("/models/embed.gguf"),
        size_bytes=GB,
        quant="Q8_0",
        architecture="bert",
        capabilities=ModelCapabilities(embedding=True),
        meta=GgufMeta(
            architecture="bert", n_ctx_train=2048, param_count=137_000_000, n_layer=12, n_embd=768
        ),
    )


def make_vision() -> ModelRecord:
    record = make_record(VISION_ID)
    return record.model_copy(update={"capabilities": ModelCapabilities(vision=True, tools=True)})


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
    built = create_app(config, state=build_state(config), start_background=False)
    built.state.registry = Registry([make_record(), make_embedding(), make_vision()])
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    # The chat model resident and ready, so ensure_loaded returns without
    # planning: these tests are about the wire formats, not about a load.
    supervisor = Supervisor(
        [InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())]
    )
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


def instance_of(app: Any, model_id: str = MODEL_ID) -> InstanceInfo:
    found = app.state.supervisor.get(model_id)
    assert found is not None
    return found


def openai_reply(
    content: str | None = "Hello there.",
    *,
    finish: str = "stop",
    tool_calls: list[dict[str, Any]] | None = None,
    usage: tuple[int, int] = (11, 5),
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }


class Upstream:
    """What the gateway sent to llama-server, and what the stub answers."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.reply: dict[str, Any] = openai_reply()
        self.chunks: list[bytes] = []

    @property
    def last(self) -> dict[str, Any]:
        return self.payloads[-1]


def sse(*frames: Any) -> list[bytes]:
    """OpenAI SSE bytes: dict frames as ``data:`` JSON, ``"[DONE]"``, ``": x"`` comments."""
    out: list[bytes] = []
    for frame in frames:
        if frame == "[DONE]":
            out.append(b"data: [DONE]\n\n")
        elif isinstance(frame, str):
            out.append(f"{frame}\n\n".encode())
        else:
            out.append(f"data: {json.dumps(frame)}\n\n".encode())
    return out


def chunk(
    delta: dict[str, Any] | None = None,
    *,
    finish: str | None = None,
    usage: tuple[int, int] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish, **extra}],
    }
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    return body


@pytest.fixture()
def upstream(monkeypatch: pytest.MonkeyPatch) -> Upstream:
    stub = Upstream()

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any], **_kwargs: Any
    ) -> JSONResponse:
        stub.payloads.append(json.loads(json.dumps(payload)))
        return JSONResponse(stub.reply)

    async def fake_stream_upstream(
        state: Any, record: Any, url: str, payload: dict[str, Any], started: float, **_kw: Any
    ) -> AsyncIterator[bytes]:
        stub.payloads.append(json.loads(json.dumps(payload)))
        for part in stub.chunks:
            yield part

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    monkeypatch.setattr(openai_routes, "_stream_upstream", fake_stream_upstream)
    return stub


def assert_shape(actual: Any, expected: Any, path: str = "$") -> None:
    """``actual`` carries every key of ``expected`` with a compatible type.

    Extra keys in ``actual`` are fine (the surface is additive by design);
    ``null`` in the fixture means "any value"; ints and floats are one kind;
    only the first element of a list is compared.
    """
    if expected is None:
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {actual!r}"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} is missing (has {sorted(actual)})"
            assert_shape(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected an array, got {actual!r}"
        if expected and actual:
            assert_shape(actual[0], expected[0], f"{path}[0]")
    elif isinstance(expected, bool):
        assert isinstance(actual, bool), f"{path}: expected bool, got {actual!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), (
            f"{path}: expected a number, got {actual!r}"
        )
    else:
        assert isinstance(actual, type(expected)), f"{path}: expected {type(expected).__name__}"


@pytest.fixture()
def http(app: Any) -> Any:
    """ONE client (one lifespan) per test, as a caller on this machine.

    The app's MCP session manager can run its lifespan only once, so a test
    that needs several requests shares this client rather than opening more.
    """
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        yield client
