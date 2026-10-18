"""The SSE reader and the chunk -> event normaliser the two translators share."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from studioforge.api.compat_stream import chat_events, iter_frames, sse_event
from tests.unit.compat.conftest import chunk, sse


async def _feed(parts: list[bytes]) -> AsyncIterator[bytes]:
    for part in parts:
        yield part


async def _frames(parts: list[bytes]) -> list[tuple[str, object]]:
    return [
        (f.kind, f.data if f.kind == "data" else f.text) async for f in iter_frames(_feed(parts))
    ]


def test_sse_event_framing() -> None:
    assert sse_event("ping", {"type": "ping"}) == b'event: ping\ndata: {"type":"ping"}\n\n'
    assert sse_event(None, {"a": 1}) == b'data: {"a":1}\n\n'


async def test_frames_across_arbitrary_boundaries_and_crlf() -> None:
    raw = b"".join(sse(chunk({"content": "a"}), ": hold on", "[DONE]")).replace(b"\n", b"\r\n")
    for size in (1, 3, 11, len(raw)):
        parts = [raw[i : i + size] for i in range(0, len(raw), size)]
        kinds = [k for k, _ in await _frames(parts)]
        assert kinds == ["data", "comment", "done"], size


async def test_a_trailing_event_without_a_blank_line_is_still_delivered() -> None:
    frames = await _frames([b'data: {"x": 1}'])
    assert frames == [("data", {"x": 1})]


async def test_non_json_and_non_object_data_are_not_fatal() -> None:
    frames = await _frames([b"data: not json\n\n", b"data: [1, 2]\n\n", b'data: {"ok": true}\n\n'])
    assert frames[-1] == ("data", {"ok": True})
    assert frames[0][0] == "comment"


async def test_chat_events_normalises_text_reasoning_tools_usage_and_finish() -> None:
    parts = sse(
        chunk({"reasoning_content": "think"}),
        chunk({"content": "hi"}),
        chunk(
            {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{"}}]}
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}),
        chunk({"tool_calls": [{"index": 1, "function": {"name": "g", "arguments": ""}}]}),
        chunk({}, finish="tool_calls", usage=(4, 6)),
        "[DONE]",
    )
    got = [
        (e.kind, e.text or e.name or e.reason or e.index) async for e in chat_events(_feed(parts))
    ]
    assert [k for k, _ in got] == [
        "reasoning",
        "text",
        "tool_start",
        "tool_args",
        "tool_args",
        "tool_start",
        "usage",
        "finish",
        "done",
    ]
    starts = [e async for e in chat_events(_feed(parts)) if e.kind == "tool_start"]
    assert [(s.index, s.call_id, s.name) for s in starts][0] == (0, "c", "f")
    assert starts[1].call_id.startswith("call_"), "a missing id is minted, never empty"


async def test_error_frames_and_sse_comments() -> None:
    parts = [b": loading\n\n", b'data: {"error": {"message": "x"}}\n\n']
    kinds = [e.kind async for e in chat_events(_feed(parts))]
    assert kinds == ["ping", "error"]


async def test_closing_the_consumer_closes_the_upstream_generator() -> None:
    closed: list[bool] = []

    async def upstream() -> AsyncIterator[bytes]:
        try:
            for i in range(100):
                yield f"data: {json.dumps(chunk({'content': str(i)}))}\n\n".encode()
        finally:
            closed.append(True)

    stream = chat_events(upstream())
    first = await stream.__anext__()
    assert first.kind == "text"
    await stream.aclose()
    assert closed == [True]
