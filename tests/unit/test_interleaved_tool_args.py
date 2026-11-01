"""#32: interleaved streamed tool arguments are buffered per index, never dropped."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from studioforge.api.anthropic_routes import anthropic_stream
from studioforge.api.compat_stream import ChatEvent
from studioforge.api.responses_routes import response_stream


async def _events() -> AsyncIterator[ChatEvent]:
    yield ChatEvent("tool_start", index=0, call_id="call_a", name="alpha")
    yield ChatEvent("tool_args", index=0, text='{"x":')
    yield ChatEvent("tool_start", index=1, call_id="call_b", name="beta")
    yield ChatEvent("tool_args", index=1, text='{"y":')
    yield ChatEvent("tool_args", index=0, text="1}")
    yield ChatEvent("tool_args", index=1, text="2}")
    yield ChatEvent("finish", reason="tool_calls")
    yield ChatEvent("done")


def _parse(frames: list[bytes]) -> list[dict]:
    out = []
    for frame in frames:
        for line in frame.decode().splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


async def test_anthropic_buffers_interleaved_calls() -> None:
    frames = [f async for f in anthropic_stream(_events(), model="m", context={})]
    data = _parse(frames)
    starts = [d for d in data if d["type"] == "content_block_start"]
    assert [s["content_block"]["name"] for s in starts] == ["alpha", "beta"]
    args: dict[int, str] = {}
    for d in data:
        if d["type"] == "content_block_delta":
            args[d["index"]] = args.get(d["index"], "") + d["delta"]["partial_json"]
    assert json.loads(args[0]) == {"x": 1}
    assert json.loads(args[1]) == {"y": 2}


async def test_responses_buffers_interleaved_calls() -> None:
    frames = [f async for f in response_stream(_events(), {}, model="m")]
    data = _parse(frames)
    done = [d["item"] for d in data if d["type"] == "response.output_item.done"]
    assert [i["name"] for i in done] == ["alpha", "beta"]
    assert [json.loads(i["arguments"]) for i in done] == [{"x": 1}, {"y": 2}]
