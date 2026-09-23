"""An upstream failure names its exception even when ``str(exc)`` is empty (D69 §7).

Several httpx exceptions stringify to ``""``. On 2026-09-20 six ``stream failed
error=`` lines were all the log kept of six cut streams, and four
``request failed`` lines on 09-16/17 read ``failed: . Recent llama-server
output``. Both sites now carry ``ClassName: message``, or the class name alone.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from studioforge.api import openai_routes
from studioforge.config import Config
from studioforge.errors import UpstreamError
from studioforge.types import GB, GgufMeta, ModelRecord


class _Recorder:
    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


class _Supervisor:
    def __init__(self) -> None:
        self.active = 0

    def mark_request_start(self, model_id: str, *, client: str | None = None) -> str | None:
        self.active += 1
        return None

    def mark_request_end(
        self,
        model_id: str,
        *,
        tokens_per_second: float | None = None,
        request_id: str | None = None,
    ) -> None:
        self.active -= 1

    def base_url(self, model_id: str) -> str:
        return "http://127.0.0.1:1"

    def tail_log(self, model_id: str, n: int = 200) -> list[str]:
        return ["srv  update_slots: all slots are idle"]


class _BreaksAfterOneChunk:
    status_code = 200

    async def __aenter__(self) -> _BreaksAfterOneChunk:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def aiter_raw(self) -> Any:
        yield b'data: {"delta":1}\n\n'
        await asyncio.sleep(0)
        # What a child that died mid-generation looks like from httpx: no text.
        raise httpx.ReadError("")

    async def aread(self) -> bytes:
        return b""


class _State:
    def __init__(self, client: Any) -> None:
        self.supervisor = _Supervisor()
        self.client = client
        self.config = Config(data_dir="/tmp/sf-upstream-text")


def _record() -> ModelRecord:
    return ModelRecord(
        id="test/model",
        name="test/model",
        path="/models/test.gguf",
        size_bytes=GB,
        meta=GgufMeta(architecture="llama", n_layer=8, n_head=8, n_head_kv=8, n_embd=512),
    )


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    monkeypatch.setattr(openai_routes, "log", rec)
    return rec


def test_exc_text_names_the_class_when_the_message_is_empty() -> None:
    assert openai_routes._exc_text(httpx.ReadError("")) == "ReadError"
    assert openai_routes._exc_text(httpx.RemoteProtocolError("  ")) == "RemoteProtocolError"
    assert (
        openai_routes._exc_text(httpx.ConnectError("connection refused"))
        == "ConnectError: connection refused"
    )


async def test_a_stream_cut_mid_generation_logs_and_frames_the_exception_class(
    recorder: _Recorder,
) -> None:
    state = _State(type("C", (), {"stream": lambda self, *a, **k: _BreaksAfterOneChunk()})())
    chunks = [
        chunk
        async for chunk in openai_routes._stream_upstream(
            state, _record(), "http://x/v1/chat/completions", {}, 0.0
        )
    ]

    (event, fields) = recorder.warnings[0]
    assert event == "stream failed"
    assert fields["error"] == "ReadError"
    error_frames = [c for c in chunks if c.startswith(b"data: {") and b'"error"' in c]
    assert len(error_frames) == 1
    message = json.loads(error_frames[0][len(b"data: ") :])["error"]["message"]
    assert "failed mid-stream: ReadError." in message
    assert "mid-stream: ." not in message
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert state.supervisor.active == 0


async def test_a_non_streamed_failure_names_the_exception_class() -> None:
    class _Client:
        async def post(self, *_a: object, **_kw: object) -> Any:
            raise httpx.RemoteProtocolError("")

    state = _State(_Client())
    with pytest.raises(UpstreamError) as caught:
        await openai_routes._forward(state, _record(), "/v1/chat/completions", {})

    message = caught.value.message
    assert "failed: RemoteProtocolError. Recent llama-server output" in message
    assert "failed: ." not in message
    assert state.supervisor.active == 0
