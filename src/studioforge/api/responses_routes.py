"""OpenAI Responses API: ``POST /v1/responses`` (a faithful, stateless subset).

LM Studio has served ``/v1/responses`` since 0.3.29, and agent frameworks that
moved to it (the OpenAI Agents SDK, Codex-style CLIs, OpenClaw's
``openai-responses`` provider) expect it next to chat completions. Like
``/v1/messages`` this is a translation layer over the shared chat path
(:func:`~studioforge.api.openai_routes.prepare_chat`), so model resolution,
vision normalisation, tool validation, tiering, attribution and the per-request
``ttl``/``priority`` behave exactly as on ``/v1/chat/completions``.

Supported
---------
``model``, ``input`` (string, or a list of ``message`` / ``function_call`` /
``function_call_output`` items; ``input_text`` / ``output_text`` / ``input_image``
parts), ``instructions``, ``tools`` (type ``function``), ``tool_choice``,
``parallel_tool_calls``, ``temperature``, ``top_p``, ``max_output_tokens``,
``text.format`` (``text``, ``json_object``, ``json_schema``), ``stream``. The
reply is a ``response`` object whose ``output[]`` holds a ``message`` item with
``output_text`` content and/or ``function_call`` items, plus ``usage``.

Streaming emits the events the official SDKs consume: ``response.created``,
``response.in_progress``, ``response.output_item.added``,
``response.content_part.added``, ``response.output_text.delta`` / ``.done``,
``response.function_call_arguments.delta`` / ``.done``,
``response.content_part.done``, ``response.output_item.done`` and
``response.completed`` (``response.incomplete`` when the token budget ran out).

Not supported (a 400 naming the parameter, never a quiet no-op)
---------------------------------------------------------------
Statefulness: ``previous_response_id``, ``item_reference`` input items and
``background: true`` need a response store this server does not keep;
built-in tools (web search, file search, MCP, computer use, code interpreter)
and ``input_file`` parts. ``store``, ``metadata``, ``user``, ``include``,
``reasoning`` and ``truncation`` are accepted and ignored; ``store`` is
echoed as ``false`` because nothing is stored. Reasoning items are not emitted.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from studioforge.api.compat_stream import ChatEvent, as_dict, chat_events, closing, sse_event
from studioforge.api.openai_routes import (
    SSE_HEADERS,
    _json_body,
    admit_stream,
    chat_json,
    chat_stream,
    prepare_chat,
)
from studioforge.errors import BadRequestError

router = APIRouter()

ROUTE = "/v1/responses"


def _bad(message: str, param: str, code: str | None = None) -> BadRequestError:
    return BadRequestError(message, param=param, code=code)


def _unsupported(message: str, param: str) -> BadRequestError:
    return _bad(message, param, code="unsupported_parameter")


# ---------------------------------------------------------------------------
# Request: Responses -> chat completions
# ---------------------------------------------------------------------------


def _content_parts(content: Any, param: str) -> str | list[dict[str, Any]]:
    """Message ``content`` as a chat string, or a parts list when it has images."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise _bad("content must be a string or an array of parts", param)
    parts: list[dict[str, Any]] = []
    for position, part in enumerate(content):
        where = f"{param}[{position}]"
        if isinstance(part, str):
            parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            raise _bad("every content part must be an object", where)
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": str(part.get("text") or "")})
        elif kind == "input_image":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            if not isinstance(url, str) or not url:
                raise _unsupported(
                    "input_image needs an 'image_url' (file_id images are not supported)", where
                )
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif kind == "refusal":
            parts.append({"type": "text", "text": str(part.get("refusal") or "")})
        else:
            raise _unsupported(f"content part type {kind!r} is not supported", where)
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def _output_text(value: Any, param: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    parts = _content_parts(value, param)
    if isinstance(parts, str):
        return parts
    return "\n".join(p["text"] for p in parts if p["type"] == "text")


def _items_to_messages(items: Any) -> list[dict[str, Any]]:
    if isinstance(items, str):
        return [{"role": "user", "content": items}]
    if not isinstance(items, list) or not items:
        raise _bad("'input' must be a string or a non-empty array of items", "input")
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        where = f"input[{index}]"
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise _bad("every input item must be an object", where)
        kind = item.get("type") or ("message" if "role" in item else None)
        if kind == "message":
            role = item.get("role")
            if role == "developer":
                role = "system"
            if role not in ("user", "assistant", "system"):
                raise _bad("message role must be user, assistant, system or developer", where)
            content = _content_parts(item.get("content"), f"{where}.content")
            if role != "user" and not isinstance(content, str):
                content = "\n".join(p["text"] for p in content if p["type"] == "text")
            messages.append({"role": role, "content": content})
        elif kind == "function_call":
            call_id, name = item.get("call_id"), item.get("name")
            if not isinstance(call_id, str) or not isinstance(name, str) or not name:
                raise _bad("function_call needs 'call_id' and 'name'", where)
            call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": str(item.get("arguments") or "{}")},
            }
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
        elif kind == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise _bad("function_call_output needs 'call_id'", where)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _output_text(item.get("output"), f"{where}.output"),
                }
            )
        elif kind == "reasoning":
            continue
        elif kind == "item_reference":
            raise _unsupported(
                "item_reference needs stored responses; this server is stateless", where
            )
        else:
            raise _unsupported(f"input item type {kind!r} is not supported", where)
    return messages


def _tools(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise _bad("'tools' must be an array", "tools")
    out: list[dict[str, Any]] = []
    for index, tool in enumerate(raw):
        where = f"tools[{index}]"
        if not isinstance(tool, dict):
            raise _bad("every tool must be an object", where)
        if tool.get("type") != "function":
            raise _unsupported(
                f"tool type {tool.get('type')!r} is not supported; only 'function' tools are",
                where,
            )
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise _bad("function tools need a 'name'", where)
        function: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("parameters")
            if isinstance(tool.get("parameters"), dict)
            else {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = str(tool["description"])
        if isinstance(tool.get("strict"), bool):
            function["strict"] = tool["strict"]
        out.append({"type": "function", "function": function})
    return out


def _response_format(text: Any) -> dict[str, Any] | None:
    if text is None:
        return None
    if not isinstance(text, dict):
        raise _bad("'text' must be an object", "text")
    fmt = text.get("format")
    if fmt is None:
        return None
    if not isinstance(fmt, dict):
        raise _bad("text.format must be an object", "text")
    kind = fmt.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return {"type": "json_object"}
    if kind == "json_schema":
        schema = fmt.get("schema")
        if not isinstance(schema, dict):
            raise _bad("text.format.schema is required for json_schema", "text")
        spec: dict[str, Any] = {"name": fmt.get("name") or "response", "schema": schema}
        if isinstance(fmt.get("strict"), bool):
            spec["strict"] = fmt["strict"]
        return {"type": "json_schema", "json_schema": spec}
    raise _bad(f"text.format.type {kind!r} is not supported", "text")


def responses_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Translate a Responses request into an OpenAI chat-completions body."""
    if body.get("previous_response_id"):
        raise _unsupported(
            "previous_response_id needs stored responses; this server is stateless -- "
            "send the whole conversation in 'input'",
            "previous_response_id",
        )
    if body.get("background"):
        raise _unsupported("background responses are not supported", "background")
    if "input" not in body:
        raise _bad("'input' is required", "input")

    messages: list[dict[str, Any]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.append({"role": "system", "content": instructions})
    messages.extend(_items_to_messages(body["input"]))

    payload: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    for source, target in (
        ("max_output_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("ttl", "ttl"),
        ("priority", "priority"),
    ):
        if source in body and body[source] is not None:
            payload[target] = body[source]
    if body.get("tools"):
        payload["tools"] = _tools(body["tools"])
        choice = body.get("tool_choice")
        if isinstance(choice, str):
            payload["tool_choice"] = choice
        elif isinstance(choice, dict):
            if choice.get("type") != "function" or not choice.get("name"):
                raise _unsupported(
                    "only {'type':'function','name':...} tool_choice objects", "tool_choice"
                )
            payload["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        if isinstance(body.get("parallel_tool_calls"), bool):
            payload["parallel_tool_calls"] = body["parallel_tool_calls"]
    fmt = _response_format(body.get("text"))
    if fmt is not None:
        payload["response_format"] = fmt
    if body.get("stream"):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


# ---------------------------------------------------------------------------
# Response objects
# ---------------------------------------------------------------------------


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _usage(prompt: int | None, completion: int | None, cached: int | None = None) -> dict[str, Any]:
    prompt_n, completion_n = int(prompt or 0), int(completion or 0)
    return {
        "input_tokens": prompt_n,
        "input_tokens_details": {"cached_tokens": int(cached or 0)},
        "output_tokens": completion_n,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": prompt_n + completion_n,
    }


def _skeleton(
    request_body: dict[str, Any],
    *,
    response_id: str,
    model: str,
    created_at: int,
    status: str,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    incomplete: str | None = None,
) -> dict[str, Any]:
    """The ``response`` object, echoing the request's knobs the way OpenAI does."""
    text_format = {"type": "text"}
    text = request_body.get("text")
    if isinstance(text, dict) and isinstance(text.get("format"), dict):
        text_format = text["format"]
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": None,
        "incomplete_details": {"reason": incomplete} if incomplete else None,
        "instructions": request_body.get("instructions"),
        "max_output_tokens": request_body.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": request_body.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": {"effort": None, "summary": None},
        "store": False,
        "temperature": request_body.get("temperature", 1.0),
        "text": {"format": text_format},
        "tool_choice": request_body.get("tool_choice", "auto"),
        "tools": request_body.get("tools") or [],
        "top_p": request_body.get("top_p", 1.0),
        "truncation": "disabled",
        "usage": usage,
        "user": request_body.get("user"),
        "metadata": request_body.get("metadata") or {},
    }


def _message_item(text: str, *, item_id: str | None = None, done: bool = True) -> dict[str, Any]:
    return {
        "id": item_id or _id("msg"),
        "type": "message",
        "status": "completed" if done else "in_progress",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}] if done else [],
    }


def _call_item(
    call_id: str, name: str, arguments: str, *, item_id: str | None = None, done: bool = True
) -> dict[str, Any]:
    return {
        "id": item_id or _id("fc"),
        "type": "function_call",
        "status": "completed" if done else "in_progress",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    }


def chat_to_response(
    data: dict[str, Any], request_body: dict[str, Any], *, model: str
) -> dict[str, Any]:
    """An OpenAI chat-completion response as a Responses ``response`` object."""
    choices = data.get("choices")
    choice = as_dict(choices[0] if isinstance(choices, list) and choices else None)
    message = as_dict(choice.get("message"))
    output: list[dict[str, Any]] = []
    content = message.get("content")
    calls = [c for c in (message.get("tool_calls") or []) if isinstance(c, dict)]
    if (isinstance(content, str) and content) or not calls:
        output.append(_message_item(content if isinstance(content, str) else ""))
    for call in calls:
        function = as_dict(call.get("function"))
        arguments = function.get("arguments")
        output.append(
            _call_item(
                str(call.get("id") or _id("call")),
                str(function.get("name") or ""),
                arguments if isinstance(arguments, str) else "{}",
            )
        )
    usage = as_dict(data.get("usage"))
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    truncated = choice.get("finish_reason") == "length"
    return _skeleton(
        request_body,
        response_id=_id("resp"),
        model=model,
        created_at=int(time.time()),
        status="incomplete" if truncated else "completed",
        output=output,
        usage=_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"), cached),
        incomplete="max_output_tokens" if truncated else None,
    )


async def response_stream(
    events: AsyncIterator[ChatEvent], request_body: dict[str, Any], *, model: str
) -> AsyncIterator[bytes]:
    """Chat events -> Responses SSE (``response.created`` ... ``response.completed``)."""
    response_id = _id("resp")
    created_at = int(time.time())
    seq = 0

    def emit(kind: str, **fields: Any) -> bytes:
        nonlocal seq
        frame = sse_event(kind, {"type": kind, "sequence_number": seq, **fields})
        seq += 1
        return frame

    def snapshot(
        status: str,
        output: list[dict[str, Any]],
        usage: dict[str, Any] | None,
        incomplete: str | None = None,
    ) -> dict[str, Any]:
        return _skeleton(
            request_body,
            response_id=response_id,
            model=model,
            created_at=created_at,
            status=status,
            output=output,
            usage=usage,
            incomplete=incomplete,
        )

    yield emit("response.created", response=snapshot("in_progress", [], None))
    yield emit("response.in_progress", response=snapshot("in_progress", [], None))

    done_items: list[dict[str, Any]] = []
    # The open output item: kind, ids, accumulated text / arguments.
    current: dict[str, Any] | None = None
    tool_index: int | None = None
    finish: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None

    def close_item() -> list[bytes]:
        nonlocal current
        if current is None:
            return []
        frames: list[bytes] = []
        out_index = len(done_items)
        if current["kind"] == "message":
            text = current["text"]
            part = {"type": "output_text", "text": text, "annotations": []}
            frames.append(
                emit(
                    "response.output_text.done",
                    item_id=current["id"],
                    output_index=out_index,
                    content_index=0,
                    text=text,
                    logprobs=[],
                )
            )
            frames.append(
                emit(
                    "response.content_part.done",
                    item_id=current["id"],
                    output_index=out_index,
                    content_index=0,
                    part=part,
                )
            )
            item = _message_item(text, item_id=current["id"])
        else:
            frames.append(
                emit(
                    "response.function_call_arguments.done",
                    item_id=current["id"],
                    output_index=out_index,
                    name=current["name"],
                    arguments=current["text"],
                )
            )
            item = _call_item(
                current["call_id"], current["name"], current["text"], item_id=current["id"]
            )
        frames.append(emit("response.output_item.done", output_index=out_index, item=item))
        done_items.append(item)
        current = None
        return frames

    def open_message() -> list[bytes]:
        nonlocal current
        frames = close_item()
        current = {"kind": "message", "id": _id("msg"), "text": ""}
        out_index = len(done_items)
        frames.append(
            emit(
                "response.output_item.added",
                output_index=out_index,
                item=_message_item("", item_id=current["id"], done=False),
            )
        )
        frames.append(
            emit(
                "response.content_part.added",
                item_id=current["id"],
                output_index=out_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": []},
            )
        )
        return frames

    async with closing(events) as stream:
        async for event in stream:
            if event.kind == "ping":
                yield b": keep-alive\n\n"
            elif event.kind == "text":
                if current is None or current["kind"] != "message":
                    for frame in open_message():
                        yield frame
                assert current is not None
                current["text"] += event.text
                yield emit(
                    "response.output_text.delta",
                    item_id=current["id"],
                    output_index=len(done_items),
                    content_index=0,
                    delta=event.text,
                    logprobs=[],
                )
            elif event.kind == "tool_start":
                frames = close_item()
                current = {
                    "kind": "call",
                    "id": _id("fc"),
                    "call_id": event.call_id,
                    "name": event.name,
                    "text": "",
                }
                tool_index = event.index
                frames.append(
                    emit(
                        "response.output_item.added",
                        output_index=len(done_items),
                        item=_call_item(
                            event.call_id, event.name, "", item_id=current["id"], done=False
                        ),
                    )
                )
                for frame in frames:
                    yield frame
            elif event.kind == "tool_args":
                if current is not None and current["kind"] == "call" and tool_index == event.index:
                    current["text"] += event.text
                    yield emit(
                        "response.function_call_arguments.delta",
                        item_id=current["id"],
                        output_index=len(done_items),
                        delta=event.text,
                    )
            elif event.kind == "finish":
                finish = event.reason
            elif event.kind == "usage":
                prompt_tokens = event.prompt_tokens
                completion_tokens = event.completion_tokens
                cached_tokens = event.cached_tokens
            elif event.kind == "error":
                err = event.error
                yield emit(
                    "error",
                    code=err.get("code") if isinstance(err.get("code"), str) else "upstream_error",
                    message=str(err.get("message") or "upstream error"),
                    param=None,
                )
                return
            elif event.kind == "done":
                break

    for frame in close_item():
        yield frame
    if not done_items:
        for frame in open_message():
            yield frame
        for frame in close_item():
            yield frame
    truncated = finish == "length"
    final = snapshot(
        "incomplete" if truncated else "completed",
        done_items,
        _usage(prompt_tokens, completion_tokens, cached_tokens),
        "max_output_tokens" if truncated else None,
    )
    yield emit("response.incomplete" if truncated else "response.completed", response=final)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/v1/responses")
async def create_response(request: Request) -> Any:
    """OpenAI Responses API on top of the shared chat path."""
    body = await _json_body(request)
    chat_body = responses_to_chat(body)
    prepared = await prepare_chat(request, chat_body)
    model = prepared.record.id
    if prepared.payload.get("stream"):
        admit_stream(prepared)
        return StreamingResponse(
            response_stream(chat_events(chat_stream(prepared, route=ROUTE)), body, model=model),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )
    data = await chat_json(prepared, route=ROUTE)
    return JSONResponse(chat_to_response(data, body, model=model))
