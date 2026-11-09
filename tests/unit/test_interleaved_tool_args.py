"""#32: interleaved streamed tool arguments are buffered per index, never dropped.

...and only interleaved ones: sequential calls (each call's arguments complete
before the next one starts, the normal llama-server shape) stream live, one
argument fragment per frame, instead of being held back to the end.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable

import pytest

from studioforge.api.anthropic_routes import anthropic_stream
from studioforge.api.compat_stream import ChatEvent, arguments_complete
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


def _sequential() -> list[ChatEvent]:
    return [
        ChatEvent("tool_start", index=0, call_id="call_a", name="alpha"),
        ChatEvent("tool_args", index=0, text='{"x":'),
        ChatEvent("tool_args", index=0, text="1}"),
        ChatEvent("tool_start", index=1, call_id="call_b", name="beta"),
        ChatEvent("tool_args", index=1, text='{"y":'),
        ChatEvent("tool_args", index=1, text="2}"),
        ChatEvent("finish", reason="tool_calls"),
        ChatEvent("done"),
    ]


def _headers_first() -> list[ChatEvent]:
    """Every call opened before any arguments arrive, then a late third call."""
    return [
        ChatEvent("tool_start", index=0, call_id="call_a", name="alpha"),
        ChatEvent("tool_start", index=1, call_id="call_b", name="beta"),
        ChatEvent("tool_args", index=0, text='{"x":1}'),
        ChatEvent("tool_args", index=1, text='{"y":'),
        ChatEvent("tool_start", index=2, call_id="call_c", name="gamma"),
        ChatEvent("tool_args", index=1, text="2}"),
        ChatEvent("tool_args", index=2, text='{"z":3}'),
        ChatEvent("finish", reason="tool_calls"),
        ChatEvent("done"),
    ]


async def _replay(events: Iterable[ChatEvent]) -> AsyncIterator[ChatEvent]:
    for event in events:
        yield event


def _parse(frames: list[bytes]) -> list[dict]:
    out = []
    for frame in frames:
        for line in frame.decode().splitlines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


async def _anthropic(events: AsyncIterator[ChatEvent]) -> list[dict]:
    return _parse([f async for f in anthropic_stream(events, model="m", context={})])


async def _responses(events: AsyncIterator[ChatEvent]) -> list[dict]:
    return _parse([f async for f in response_stream(events, {}, model="m")])


def _anthropic_calls(data: list[dict]) -> list[tuple[str, list[str]]]:
    """``(name, [argument fragments, one per delta frame])`` per tool block, in order."""
    names: dict[int, str] = {}
    fragments: dict[int, list[str]] = {}
    for d in data:
        if d["type"] == "content_block_start" and d["content_block"]["type"] == "tool_use":
            names[d["index"]] = d["content_block"]["name"]
            fragments[d["index"]] = []
        elif d["type"] == "content_block_delta" and d["index"] in fragments:
            fragments[d["index"]].append(d["delta"]["partial_json"])
    return [(names[i], fragments[i]) for i in sorted(names)]


def _responses_calls(data: list[dict]) -> list[tuple[str, list[str], str]]:
    """``(name, [delta fragments], final arguments)`` per function_call item, in order."""
    fragments: dict[str, list[str]] = {}
    out = []
    for d in data:
        if d["type"] == "response.function_call_arguments.delta":
            fragments.setdefault(d["item_id"], []).append(d["delta"])
        elif d["type"] == "response.output_item.done" and d["item"]["type"] == "function_call":
            item = d["item"]
            out.append((item["name"], fragments.get(item["id"], []), item["arguments"]))
    return out


# ---------------------------------------------------------------------------
# interleaved: buffered, never dropped (the 2.3.0 fix)
# ---------------------------------------------------------------------------


async def test_anthropic_buffers_interleaved_calls() -> None:
    data = await _anthropic(_events())
    starts = [d for d in data if d["type"] == "content_block_start"]
    assert [s["content_block"]["name"] for s in starts] == ["alpha", "beta"]
    args: dict[int, str] = {}
    for d in data:
        if d["type"] == "content_block_delta":
            args[d["index"]] = args.get(d["index"], "") + d["delta"]["partial_json"]
    assert json.loads(args[0]) == {"x": 1}
    assert json.loads(args[1]) == {"y": 2}


async def test_responses_buffers_interleaved_calls() -> None:
    data = await _responses(_events())
    done = [d["item"] for d in data if d["type"] == "response.output_item.done"]
    assert [i["name"] for i in done] == ["alpha", "beta"]
    assert [json.loads(i["arguments"]) for i in done] == [{"x": 1}, {"y": 2}]


async def test_anthropic_headers_first_calls_keep_their_order_and_arguments() -> None:
    calls = _anthropic_calls(await _anthropic(_replay(_headers_first())))
    assert [name for name, _ in calls] == ["alpha", "beta", "gamma"]
    assert [json.loads("".join(parts)) for _, parts in calls] == [{"x": 1}, {"y": 2}, {"z": 3}]


async def test_responses_headers_first_calls_keep_their_order_and_arguments() -> None:
    calls = _responses_calls(await _responses(_replay(_headers_first())))
    assert [name for name, _, _ in calls] == ["alpha", "beta", "gamma"]
    assert [json.loads(args) for _, _, args in calls] == [{"x": 1}, {"y": 2}, {"z": 3}]


# ---------------------------------------------------------------------------
# sequential: every call streams live
# ---------------------------------------------------------------------------


async def test_anthropic_streams_sequential_calls_live() -> None:
    calls = _anthropic_calls(await _anthropic(_replay(_sequential())))
    # One delta frame per upstream fragment: the second call was not held back
    # and replayed as one blob at the end.
    assert calls == [("alpha", ['{"x":', "1}"]), ("beta", ['{"y":', "2}"])]


async def test_responses_streams_sequential_calls_live() -> None:
    calls = _responses_calls(await _responses(_replay(_sequential())))
    assert calls == [
        ("alpha", ['{"x":', "1}"], '{"x":1}'),
        ("beta", ['{"y":', "2}"], '{"y":2}'),
    ]


async def test_the_second_sequential_call_opens_before_its_arguments_arrive() -> None:
    """Live means live: the block for call 1 is on the wire before its first fragment."""
    seen: list[str] = []

    async def source() -> AsyncIterator[ChatEvent]:
        for event in _sequential():
            seen.append(f"{event.kind}:{event.index}")
            yield event

    async for frame in anthropic_stream(source(), model="m", context={}):
        if b'"name":"beta"' in frame:
            assert seen[-1] == "tool_start:1"  # nothing after the start was read yet
            break
    else:  # pragma: no cover - the assertion above is the test
        pytest.fail("beta's tool_use block never started")


# ---------------------------------------------------------------------------
# the decision itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "complete"),
    [
        ("", False),
        ("   ", False),
        ('{"x":', False),
        ('{"x":1', False),
        ('{"x":1}', True),
        ("{}", True),
        (' {"x": [1, 2]} ', True),
    ],
)
def test_arguments_complete(text: str, complete: bool) -> None:
    assert arguments_complete(text) is complete
