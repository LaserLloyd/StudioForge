"""Anthropic-compatible ``POST /v1/messages``.

LM Studio added this surface in 0.4.1, and tools written for it (Claude Code
pointed at a local base URL, the ``anthropic`` SDK) expect it on the same port
as the OpenAI one. It is a *translation* layer: an Anthropic Messages request
becomes an OpenAI chat-completions body, runs through the very same front half
as ``/v1/chat/completions`` (:func:`~studioforge.api.openai_routes.prepare_chat`
-- model resolution, vision normalisation, tool validation, tiering,
attribution, ``ttl``), and the reply is translated back. Nothing here loads a
model or talks to llama-server itself.

What maps, and what does not
----------------------------
* ``system`` (string or text blocks), ``messages`` with ``text`` / ``image``
  (base64 or URL) / ``tool_use`` / ``tool_result`` blocks, ``tools`` +
  ``tool_choice``, ``max_tokens``, ``temperature``, ``top_p``, ``top_k``,
  ``stop_sequences``, ``stream``.
* ``thinking`` blocks in the history are dropped (they cannot be replayed to a
  local template); the reply carries a ``thinking`` block only when the request
  set ``thinking: {"type": "enabled"}`` *and* the engine split its reasoning
  out (``reasoning_content``).
* Additive, StudioForge-only top-level fields, consumed and never forwarded,
  exactly as on the OpenAI route: ``ttl`` and ``priority``. ``X-SF-Client`` is
  honoured too.
* Not supported, refused with a 400 naming the parameter rather than quietly
  ignored: documents/search results as content blocks, server-side tools
  (``web_search``, ``bash`` ...), and anything else that would change the answer.
* ``stop_sequence`` is reported only when the engine says which word stopped
  generation (``stopping_word``); otherwise a stop-sequence stop reads
  ``end_turn``. ``usage`` is the engine's own token counts.

Errors leave in Anthropic's envelope
(``{"type": "error", "error": {"type": ..., "message": ...}}``) with
StudioForge's stable ``code`` beside ``type``. Authentication failures come
from the shared middleware and keep the OpenAI envelope; both SDKs read
``error.message`` from either.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from studioforge.api.compat_stream import (
    ChatEvent,
    as_dict,
    chat_events,
    closing,
    parse_tool_arguments,
    sse_event,
)
from studioforge.api.openai_routes import (
    SSE_HEADERS,
    _json_body,
    admit_stream,
    chat_json,
    chat_stream,
    prepare_chat,
)
from studioforge.errors import BadRequestError, StudioForgeError

router = APIRouter()

ROUTE = "/v1/messages"

#: HTTP status -> Anthropic error ``type``.
_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    503: "overloaded_error",
    529: "overloaded_error",
}


def error_type_for(status: int) -> str:
    if status in _ERROR_TYPES:
        return _ERROR_TYPES[status]
    return "invalid_request_error" if 400 <= status < 500 else "api_error"


def error_body(message: str, status: int, code: str | None = None) -> dict[str, Any]:
    """Anthropic's ``{"type": "error", "error": {type, message}}`` plus the catalog keys.

    ``code``, ``fix`` and ``docs`` are additive: the Anthropic SDKs read ``type``
    and ``message`` and ignore the rest, and every other StudioForge surface
    carries the same three keys for a catalogued code (``error_catalog``).
    """
    from studioforge.error_catalog import envelope_extras

    error: dict[str, Any] = {"type": error_type_for(status), "message": message}
    if code:
        error["code"] = code
        error.update(envelope_extras(code))
    return {"type": "error", "error": error}


# ---------------------------------------------------------------------------
# Request: Anthropic -> OpenAI chat
# ---------------------------------------------------------------------------


def _bad(message: str, param: str, code: str | None = None) -> BadRequestError:
    return BadRequestError(message, param=param, code=code)


def _text_of(content: Any, param: str) -> str:
    """Flatten a ``system`` / tool-result value to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad("expected a string or a list of text blocks", param)
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
        else:
            raise _bad(
                "only text blocks are supported here",
                param,
                code="unsupported_parameter",
            )
    return "\n\n".join(parts)


def _image_part(block: dict[str, Any], param: str) -> dict[str, Any]:
    source = block.get("source")
    if not isinstance(source, dict):
        raise _bad("image block needs a 'source' object", param)
    kind = source.get("type")
    if kind == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data")
        if not isinstance(data, str) or not data:
            raise _bad("image source.data is required", param)
        url = f"data:{media};base64,{data}"
    elif kind == "url":
        link = source.get("url")
        if not isinstance(link, str) or not link:
            raise _bad("image source.url is required", param)
        url = link
    else:
        raise _bad(
            f"image source type {kind!r} is not supported (use base64 or url)",
            param,
            code="unsupported_parameter",
        )
    return {"type": "image_url", "image_url": {"url": url}}


def _user_messages(content: Any, index: int) -> list[dict[str, Any]]:
    param = f"messages[{index}].content"
    if isinstance(content, str):
        return [{"role": "user", "content": content}]
    if not isinstance(content, list):
        raise _bad("content must be a string or an array of blocks", param)
    tool_messages: list[dict[str, Any]] = []
    spill_images: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    for position, block in enumerate(content):
        where = f"{param}[{position}]"
        if isinstance(block, str):
            parts.append({"type": "text", "text": block})
            continue
        if not isinstance(block, dict):
            raise _bad("every content block must be an object", where)
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": str(block.get("text") or "")})
        elif kind == "image":
            parts.append(_image_part(block, where))
        elif kind == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str) or not call_id:
                raise _bad("tool_result.tool_use_id is required", where)
            result = block.get("content")
            texts: list[str] = []
            if isinstance(result, list):
                for inner in result:
                    if isinstance(inner, dict) and inner.get("type") == "image":
                        spill_images.append(_image_part(inner, where))
                    else:
                        texts.append(_text_of([inner], where))
            else:
                texts.append(_text_of(result, where))
            text = "\n\n".join(t for t in texts if t)
            if block.get("is_error") and not text.lower().startswith("error"):
                text = f"Error: {text}" if text else "Error"
            tool_messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise _bad(
                f"content block type {kind!r} is not supported",
                where,
                code="unsupported_parameter",
            )
    out = list(tool_messages)
    tail = parts + spill_images
    if tail:
        if all(p["type"] == "text" for p in tail):
            out.append({"role": "user", "content": "\n".join(p["text"] for p in tail)})
        else:
            out.append({"role": "user", "content": tail})
    return out


def _assistant_message(content: Any, index: int) -> dict[str, Any]:
    param = f"messages[{index}].content"
    if isinstance(content, str):
        return {"role": "assistant", "content": content}
    if not isinstance(content, list):
        raise _bad("content must be a string or an array of blocks", param)
    texts: list[str] = []
    calls: list[dict[str, Any]] = []
    for position, block in enumerate(content):
        where = f"{param}[{position}]"
        if isinstance(block, str):
            texts.append(block)
            continue
        if not isinstance(block, dict):
            raise _bad("every content block must be an object", where)
        kind = block.get("type")
        if kind == "text":
            texts.append(str(block.get("text") or ""))
        elif kind == "tool_use":
            call_id, name = block.get("id"), block.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str) or not name:
                raise _bad("tool_use needs 'id' and 'name'", where)
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        elif kind in ("thinking", "redacted_thinking"):
            continue
        else:
            raise _bad(
                f"assistant content block type {kind!r} is not supported",
                where,
                code="unsupported_parameter",
            )
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts)}
    if calls:
        message["tool_calls"] = calls
    return message


def _tools(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise _bad("'tools' must be an array", "tools")
    out: list[dict[str, Any]] = []
    for index, tool in enumerate(raw):
        where = f"tools[{index}]"
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise _bad("every tool needs a 'name'", where)
        kind = tool.get("type")
        if kind not in (None, "custom"):
            raise _bad(
                f"server-side tool type {kind!r} is not supported; only client tools "
                "(name + input_schema) are",
                where,
                code="unsupported_parameter",
            )
        schema = tool.get("input_schema")
        function: dict[str, Any] = {
            "name": tool["name"],
            "parameters": schema if isinstance(schema, dict) else {"type": "object"},
        }
        if tool.get("description"):
            function["description"] = str(tool["description"])
        out.append({"type": "function", "function": function})
    return out


def _tool_choice(choice: Any, payload: dict[str, Any]) -> None:
    if choice is None:
        return
    if not isinstance(choice, dict):
        raise _bad("'tool_choice' must be an object", "tool_choice")
    kind = choice.get("type")
    if kind == "auto":
        payload["tool_choice"] = "auto"
    elif kind == "any":
        payload["tool_choice"] = "required"
    elif kind == "none":
        payload["tool_choice"] = "none"
    elif kind == "tool":
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise _bad("tool_choice.name is required when type is 'tool'", "tool_choice")
        payload["tool_choice"] = {"type": "function", "function": {"name": name}}
    else:
        raise _bad(f"unknown tool_choice type {kind!r}", "tool_choice")
    if choice.get("disable_parallel_tool_use") is True:
        payload["parallel_tool_calls"] = False


def anthropic_to_chat(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate a Messages request. Returns ``(chat_body, context)``.

    ``context`` carries what the reply translation needs back: the requested
    ``stop_sequences`` and whether ``thinking`` was enabled.
    """
    max_tokens = body.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise _bad("max_tokens: a positive integer is required", "max_tokens")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _bad("messages: at least one message is required", "messages")

    chat_messages: list[dict[str, Any]] = []
    system = _text_of(body.get("system"), "system")
    if system:
        chat_messages.append({"role": "system", "content": system})
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise _bad("every message must be an object", f"messages[{index}]")
        role = message.get("role")
        if role == "user":
            chat_messages.extend(_user_messages(message.get("content"), index))
        elif role == "assistant":
            chat_messages.append(_assistant_message(message.get("content"), index))
        else:
            raise _bad("role must be 'user' or 'assistant'", f"messages[{index}].role")

    payload: dict[str, Any] = {
        "model": body.get("model"),
        "messages": chat_messages,
        "max_tokens": max_tokens,
    }
    for key in ("temperature", "top_p", "top_k", "ttl", "priority"):
        if key in body:
            payload[key] = body[key]
    stops = body.get("stop_sequences")
    if stops is not None:
        if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
            raise _bad("stop_sequences must be an array of strings", "stop_sequences")
        if stops:
            payload["stop"] = list(stops)
    if "tools" in body and body["tools"] is not None:
        tools = _tools(body["tools"])
        if tools:
            payload["tools"] = tools
            _tool_choice(body.get("tool_choice"), payload)
    if body.get("stream"):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    thinking = body.get("thinking")
    context = {
        "stop_sequences": list(stops or []),
        "thinking": isinstance(thinking, dict) and thinking.get("type") == "enabled",
    }
    return payload, context


# ---------------------------------------------------------------------------
# Response: OpenAI chat -> Anthropic
# ---------------------------------------------------------------------------


def stop_reason(
    finish: str | None,
    *,
    has_tool: bool,
    stopping_word: str | None = None,
    stop_sequences: list[str] | None = None,
) -> tuple[str, str | None]:
    """``(stop_reason, stop_sequence)`` for an OpenAI ``finish_reason``."""
    if has_tool or finish == "tool_calls":
        return "tool_use", None
    if finish == "length":
        return "max_tokens", None
    if stopping_word and stopping_word in (stop_sequences or []):
        return "stop_sequence", stopping_word
    return "end_turn", None


def new_message_id() -> str:
    return "msg_" + uuid.uuid4().hex[:24]


def chat_to_anthropic(
    data: dict[str, Any], *, model: str, context: dict[str, Any], message_id: str | None = None
) -> dict[str, Any]:
    """An OpenAI chat-completion response as an Anthropic ``message``."""
    choices = data.get("choices")
    choice = as_dict(choices[0] if isinstance(choices, list) and choices else None)
    message = as_dict(choice.get("message"))

    blocks: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if context.get("thinking") and isinstance(reasoning, str) and reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning, "signature": ""})
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    tool_calls = [c for c in (message.get("tool_calls") or []) if isinstance(c, dict)]
    for call in tool_calls:
        function = as_dict(call.get("function"))
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id") or "toolu_" + uuid.uuid4().hex[:24]),
                "name": str(function.get("name") or ""),
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})

    word = choice.get("stopping_word") or data.get("stopping_word")
    reason, sequence = stop_reason(
        choice.get("finish_reason"),
        has_tool=bool(tool_calls),
        stopping_word=word if isinstance(word, str) else None,
        stop_sequences=context.get("stop_sequences"),
    )
    usage = as_dict(data.get("usage"))
    return {
        "id": message_id or new_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": reason,
        "stop_sequence": sequence,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


async def anthropic_stream(
    events: AsyncIterator[ChatEvent],
    *,
    model: str,
    context: dict[str, Any],
    message_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Chat events -> Anthropic Messages SSE (``message_start`` ... ``message_stop``)."""
    mid = message_id or new_message_id()
    yield sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": mid,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    index = -1
    open_kind: str | None = None  # "text" | "thinking" | "tool"
    open_tool: int | None = None  # the OpenAI tool index of the open tool block
    saw_tool = False
    finish: str | None = None
    word: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    deltas = 0

    def stop_block() -> list[bytes]:
        nonlocal open_kind, open_tool
        if open_kind is None:
            return []
        frame = sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        open_kind = None
        open_tool = None
        return [frame]

    def start_block(kind: str, block: dict[str, Any]) -> list[bytes]:
        nonlocal index, open_kind
        frames = stop_block()
        index += 1
        open_kind = kind
        frames.append(
            sse_event(
                "content_block_start",
                {"type": "content_block_start", "index": index, "content_block": block},
            )
        )
        return frames

    def delta(kind: str, **fields: Any) -> bytes:
        return sse_event(
            "content_block_delta",
            {"type": "content_block_delta", "index": index, "delta": {"type": kind, **fields}},
        )

    async with closing(events) as stream:
        async for event in stream:
            if event.kind == "ping":
                yield sse_event("ping", {"type": "ping"})
            elif event.kind == "reasoning":
                if not context.get("thinking"):
                    continue
                deltas += 1
                if open_kind != "thinking":
                    for frame in start_block(
                        "thinking", {"type": "thinking", "thinking": "", "signature": ""}
                    ):
                        yield frame
                yield delta("thinking_delta", thinking=event.text)
            elif event.kind == "text":
                deltas += 1
                if open_kind != "text":
                    for frame in start_block("text", {"type": "text", "text": ""}):
                        yield frame
                yield delta("text_delta", text=event.text)
            elif event.kind == "tool_start":
                saw_tool = True
                for frame in start_block(
                    "tool",
                    {"type": "tool_use", "id": event.call_id, "name": event.name, "input": {}},
                ):
                    yield frame
                open_tool = event.index
            elif event.kind == "tool_args":
                deltas += 1
                # Calls arrive one after another from llama-server; a fragment
                # for a call that is no longer the open block has nowhere to go.
                if open_kind == "tool" and open_tool == event.index:
                    yield delta("input_json_delta", partial_json=event.text)
            elif event.kind == "finish":
                finish, word = event.reason, event.stopping_word
            elif event.kind == "usage":
                prompt_tokens = event.prompt_tokens
                completion_tokens = event.completion_tokens
            elif event.kind == "error":
                err = event.error
                code = err.get("code") if isinstance(err.get("code"), str) else None
                yield sse_event(
                    "error",
                    error_body(str(err.get("message") or "upstream error"), 502, code),
                )
                return
            elif event.kind == "done":
                break

    for frame in stop_block():
        yield frame
    reason, sequence = stop_reason(
        finish,
        has_tool=saw_tool,
        stopping_word=word,
        stop_sequences=context.get("stop_sequences"),
    )
    yield sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": sequence},
            "usage": {
                "input_tokens": prompt_tokens or 0,
                "output_tokens": completion_tokens if completion_tokens is not None else deltas,
            },
        },
    )
    yield sse_event("message_stop", {"type": "message_stop"})


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


def _error_response(exc: StudioForgeError) -> JSONResponse:
    from studioforge.api.app import log_rejection

    log_rejection(exc)
    headers: dict[str, str] = {}
    if exc.status_code == 503:
        headers["Retry-After"] = str(exc.details.get("retry_after_s", 5))
    elif exc.details.get("retry_after_s"):
        headers["Retry-After"] = str(int(exc.details["retry_after_s"]))
    body = error_body(exc.message, exc.status_code, exc.code)
    if exc.details:
        body["error"]["studioforge"] = exc.details
    return JSONResponse(body, status_code=exc.status_code, headers=headers)


@router.post("/v1/messages")
async def create_message(request: Request) -> Any:
    """Anthropic Messages API on top of the shared chat path."""
    try:
        body = await _json_body(request)
        chat_body, context = anthropic_to_chat(body)
        prepared = await prepare_chat(request, chat_body)
        model = prepared.record.id
        if prepared.payload.get("stream"):
            admit_stream(prepared)
            return StreamingResponse(
                anthropic_stream(
                    chat_events(chat_stream(prepared, route=ROUTE)),
                    model=model,
                    context=context,
                ),
                media_type="text/event-stream",
                headers=SSE_HEADERS,
            )
        data = await chat_json(prepared, route=ROUTE)
        return JSONResponse(chat_to_anthropic(data, model=model, context=context))
    except StudioForgeError as exc:
        return _error_response(exc)
