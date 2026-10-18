"""Shared plumbing for the translated front doors (``/v1/messages``, ``/v1/responses``).

Both translate an OpenAI chat-completions exchange into another vendor's wire
format, and both need the same two things:

* **Server-sent-event framing.** :func:`sse_event` writes a named event;
  :func:`iter_frames` reads the upstream bytes back into frames. The reader
  buffers, because the gateway forwards llama-server's bytes verbatim
  (``_stream_upstream``) and a TCP chunk boundary can fall anywhere -- inside a
  line, between the two newlines that end an event.
* **A vendor-neutral view of a chat stream.** :func:`chat_events` turns OpenAI
  ``chat.completion.chunk`` frames into a handful of :class:`ChatEvent`
  records (text, reasoning, tool-call start/arguments, finish, usage, error),
  so each translator is a small state machine over events and none of them
  re-implements chunk parsing. The upstream's keep-alive comments
  (``: loading ...``, ``: prefilling ...``) become ``ping`` events, which both
  target protocols have a native frame for.
"""

from __future__ import annotations

import codecs
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any


def as_dict(value: Any) -> dict[str, Any]:
    """``value`` if it is a JSON object, else ``{}`` (untrusted upstream shapes)."""
    return value if isinstance(value, dict) else {}


@asynccontextmanager
async def closing(iterator: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Close an async generator deterministically when the ``with`` block ends.

    ``contextlib.aclosing`` wants an ``AsyncGenerator``; the chat path hands
    around ``AsyncIterator`` annotations. A client that hangs up closes the
    outermost generator, and this carries that close down to the one that holds
    the request slot and the httpx stream instead of waiting for GC.
    """
    try:
        yield iterator
    finally:
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()


def sse_event(event: str | None, data: Any) -> bytes:
    """One SSE frame. ``event`` may be ``None`` for a bare ``data:`` line."""
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {json.dumps(data, separators=(',', ':'), default=str)}\n\n".encode()


@dataclass
class Frame:
    """One parsed upstream SSE frame."""

    kind: str  # "comment" | "data" | "done"
    data: dict[str, Any] | None = None
    text: str = ""


async def iter_frames(chunks: AsyncIterator[bytes]) -> AsyncIterator[Frame]:
    """Re-frame a byte stream into SSE events (CRLF and LF both accepted).

    Closing this generator closes ``chunks`` too, so a client that hangs up
    reaches the upstream stream (which holds the request slot) at once.
    """
    buffer = ""
    # Incremental: a multi-byte character split across chunks is held until its
    # last byte arrives, and an invalid byte becomes U+FFFD at once. (Slicing at
    # UnicodeDecodeError.start held EVERYTHING after an invalid byte -- the
    # stream stalled, and the held bytes were dropped at the end.)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    async with closing(chunks):
        async for chunk in chunks:
            if not chunk:
                continue
            text = decoder.decode(chunk)
            # Normalise on the whole buffer: a CRLF split across two reads
            # leaves a lone CR here until its LF arrives.
            buffer = (buffer + text).replace("\r\n", "\n")
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                frame = _parse_block(block)
                if frame is not None:
                    yield frame
    buffer = (buffer + decoder.decode(b"", final=True)).replace("\r\n", "\n")
    while "\n\n" in buffer:
        block, buffer = buffer.split("\n\n", 1)
        frame = _parse_block(block)
        if frame is not None:
            yield frame
    if buffer.strip():
        frame = _parse_block(buffer)
        if frame is not None:
            yield frame


def _parse_block(block: str) -> Frame | None:
    data_lines: list[str] = []
    comments: list[str] = []
    for line in block.split("\n"):
        if line.startswith(":"):
            comments.append(line[1:].strip())
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if not data_lines:
        return Frame("comment", text=" ".join(comments)) if comments else None
    payload = "\n".join(data_lines).strip()
    if payload == "[DONE]":
        return Frame("done")
    try:
        parsed = json.loads(payload)
    except ValueError:
        return Frame("comment", text=payload)
    return Frame("data", data=parsed) if isinstance(parsed, dict) else None


@dataclass
class ChatEvent:
    """A vendor-neutral event of one chat completion stream."""

    kind: str  # ping text reasoning tool_start tool_args finish usage error done
    text: str = ""
    index: int = 0
    call_id: str = ""
    name: str = ""
    reason: str | None = None
    stopping_word: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    error: dict[str, Any] = field(default_factory=dict)


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:24]


async def chat_events(upstream: AsyncIterator[bytes]) -> AsyncIterator[ChatEvent]:
    """OpenAI chat-completion chunks -> :class:`ChatEvent` records."""
    started_tools: set[int] = set()
    # aclosing: when the consumer is closed (client hung up) the close must
    # reach the upstream generator NOW, not at garbage collection -- it holds
    # the request slot and the httpx stream.
    async with closing(iter_frames(upstream)) as frames:
        async for event in _chat_events(frames, started_tools):
            yield event


async def _chat_events(
    frames: AsyncIterator[Frame], started_tools: set[int]
) -> AsyncIterator[ChatEvent]:
    async for frame in frames:
        if frame.kind == "comment":
            yield ChatEvent("ping", text=frame.text)
            continue
        if frame.kind == "done":
            yield ChatEvent("done")
            return
        data = frame.data or {}
        if isinstance(data.get("error"), dict):
            yield ChatEvent("error", error=data["error"])
            continue
        usage = data.get("usage")
        if isinstance(usage, dict):
            details = usage.get("prompt_tokens_details")
            cached = details.get("cached_tokens") if isinstance(details, dict) else None
            yield ChatEvent(
                "usage",
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                cached_tokens=_int_or_none(cached),
            )
        for choice in data.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if isinstance(delta, dict):
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(reasoning, str) and reasoning:
                    yield ChatEvent("reasoning", text=reasoning)
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield ChatEvent("text", text=content)
                for call in delta.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    index = call.get("index")
                    index = index if isinstance(index, int) else 0
                    function = call.get("function")
                    function = function if isinstance(function, dict) else {}
                    if index not in started_tools:
                        started_tools.add(index)
                        yield ChatEvent(
                            "tool_start",
                            index=index,
                            call_id=str(call.get("id") or new_call_id()),
                            name=str(function.get("name") or ""),
                        )
                    arguments = function.get("arguments")
                    if isinstance(arguments, str) and arguments:
                        yield ChatEvent("tool_args", index=index, text=arguments)
            reason = choice.get("finish_reason")
            if reason:
                word = choice.get("stopping_word") or data.get("stopping_word")
                yield ChatEvent(
                    "finish",
                    reason=str(reason),
                    stopping_word=word if isinstance(word, str) and word else None,
                )


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def parse_tool_arguments(raw: Any) -> Any:
    """A tool call's ``arguments`` string as a JSON value (``{}`` when unparseable)."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except ValueError:
        return {}
