"""A JIT load that fails after the SSE keep-alives began ends with an error frame (D64, CR-4).

Raised by the ClawChat V14 audit (2026-09-12): 162 ``jit load failed
mid-stream`` log lines between 08-17 and 09-11, and a client that saw a chat
turn stop mid-stream with a permanent spinner. A streaming request loads inside
the SSE body, so by the time the load fails the keep-alive comments have
committed a 200 and no status is left to carry the error.

The frame itself predates the report (the initial import emitted one; D53 added
the details) -- what a client could not rely on was its shape: ``type`` was
always ``server_error``, ``param`` was absent, and ``retry_after_s`` lived only
inside the vendor block. What these pin:

* the stream ends with ``data: {"error": ...}`` and then ``data: [DONE]``, after
  at least one keep-alive has already gone out;
* the frame is the HTTP envelope of the same error (``message``, ``type``,
  ``code``, ``param``, ``studioforge``), plus a top-level ``retry_after_s`` when
  the error knows the wait and none when it does not;
* an unexpected exception still ends in a frame, with a reference, never its
  text;
* over HTTP the whole thing is a 200 ``text/event-stream``.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.errors import InsufficientVramError, ModelLoadError
from tests.unit.test_catalog_routes import MODEL_ID, app, make_record  # noqa: F401 - the fixture


def _state(failure: BaseException, *, delay_s: float = 0.05) -> Any:
    async def ensure_loaded(model_id: str, **_: Any) -> None:
        await asyncio.sleep(delay_s)
        raise failure

    return SimpleNamespace(
        manager=SimpleNamespace(ensure_loaded=ensure_loaded),
        config=SimpleNamespace(gateway=SimpleNamespace(stream_keepalive_interval_s=0.01)),
    )


async def _frames(state: Any) -> list[bytes]:
    return [
        chunk async for chunk in openai_routes._stream_with_jit_load(state, make_record(), {}, None)
    ]


def _error(frame: bytes) -> dict[str, Any]:
    assert frame.startswith(b"data: ") and frame.endswith(b"\n\n"), frame
    body = json.loads(frame[len(b"data: ") :])
    assert set(body) == {"error"}
    return dict(body["error"])


async def test_a_load_that_fails_after_the_keepalives_ends_with_an_error_frame_then_done() -> None:
    leased = InsufficientVramError(
        "Cannot load 'vendor/thing-Q4_K_M' entirely in VRAM: CUDA [1] is leased",
        code="gpu_leased",
        details={"retry_after_s": 60, "lease": {"id": "4398b3afda14", "holder": "clawforge2"}},
    )
    frames = await _frames(_state(leased))

    assert frames[0].startswith(b": loading "), "the 200 was already committed by a keep-alive"
    assert frames[-1] == b"data: [DONE]\n\n"
    error = _error(frames[-2])
    assert error == leased.to_payload()["error"] | {"retry_after_s": 60}
    assert error["code"] == "gpu_leased"
    assert error["type"] == "server_error"
    assert error["param"] is None
    assert error["studioforge"]["lease"]["id"] == "4398b3afda14"


async def test_a_failure_with_no_wait_carries_no_retry_after() -> None:
    """ "Retry later" is bad advice when nothing is going to change (D36)."""
    failed = ModelLoadError("llama-server for 'vendor/thing-Q4_K_M' exited with code 1")
    frames = await _frames(_state(failed))
    error = _error(frames[-2])
    assert frames[-1] == b"data: [DONE]\n\n"
    assert "retry_after_s" not in error
    assert error["code"] == (failed.code or "model_load_failed")
    assert error["message"] == failed.message


async def test_an_error_without_a_code_gets_model_load_failed() -> None:
    uncoded = ModelLoadError("no code on this one")
    uncoded.code = None
    frames = await _frames(_state(uncoded))
    assert _error(frames[-2])["code"] == "model_load_failed"


async def test_an_unexpected_exception_ends_in_a_frame_with_a_reference_not_its_text() -> None:
    """D55: an exception's text is for the operator's log, never the caller."""
    frames = await _frames(_state(RuntimeError("operator-only detail: row 7 of the registry")))
    error = _error(frames[-2])
    assert frames[-1] == b"data: [DONE]\n\n"
    assert error["code"] == "model_load_failed"
    assert "operator-only" not in error["message"]
    assert error["studioforge"]["ref"] in error["message"]


def test_over_http_the_failed_load_is_a_200_event_stream_that_ends_in_the_frame(
    app: Any,  # noqa: F811 - the imported fixture, by design
) -> None:
    async def ensure_loaded(model_id: str, **_: Any) -> None:
        await asyncio.sleep(0.05)
        raise InsufficientVramError(
            "Cannot load it", code="gpu_leased", details={"retry_after_s": 30}
        )

    app.state.config.gateway.stream_keepalive_interval_s = 0.01
    app.state.manager.ensure_loaded = ensure_loaded
    app.state.manager.admission_check = lambda *a, **k: None
    app.state.manager.lease_check = lambda *a, **k: None

    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [e for e in response.text.split("\n\n") if e]
    assert events[0].startswith(": loading ")
    assert events[-1] == "data: [DONE]"
    error = json.loads(events[-2][len("data: ") :])["error"]
    assert error["code"] == "gpu_leased"
    assert error["retry_after_s"] == 30
    assert error["studioforge"]["retry_after_s"] == 30
