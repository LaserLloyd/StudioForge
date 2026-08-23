"""OpenAI-compatible endpoints.

This is the contract OpenClaw depends on, so behaviour here is deliberately
LM Studio-shaped:

* ``GET /v1/models`` lists every **downloaded** model, loaded or not.
* Naming an unloaded model just-in-time loads it; concurrent requests during
  the load queue rather than error.
* Streaming is passed through byte-for-byte, including the ``data: [DONE]``
  sentinel, because clients parse that framing strictly.

Route handlers stay thin: resolution/loading lives in
:class:`~studioforge.core.manager.ModelManager`, image normalization in
:mod:`studioforge.api.vision`, and the byte plumbing in :func:`_stream_upstream`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from studioforge.api.vision import prepare_messages
from studioforge.errors import (
    BadRequestError,
    ModelNotFoundError,
    StudioForgeError,
    UpstreamError,
)
from studioforge.logging import get_logger
from studioforge.types import ModelRecord

log = get_logger(__name__)

router = APIRouter()

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    # Nginx/tailscale-serve style proxies buffer SSE unless told not to; a
    # buffered stream looks exactly like a hung model to the client.
    "X-Accel-Buffering": "no",
}


def _app_state(request: Request) -> Any:
    return request.app.state


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


async def _wait_for_scan(state: Any) -> None:
    """The middleware already holds requests for the boot scan (D33); this is
    the same wait for a caller that invokes the handler in-process."""
    from studioforge.api.app import SCAN_WAIT_S, wait_for_boot

    await wait_for_boot(state, timeout_s=SCAN_WAIT_S, scan_only=True)


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    """Every downloaded model, loaded or not -- LM Studio parity."""
    state = _app_state(request)
    # A client that connects the moment the port answers (OpenClaw reconnects
    # on a timer) must not read an empty library because the boot scan is
    # still running: wait for the index, bounded (D33).
    await _wait_for_scan(state)
    data = state.registry.openai_list()
    # `state` and `loaded_context_length` are additive fields that strict
    # OpenAI clients ignore. Putting them here makes the most-used endpoint
    # authoritative about what is resident, instead of forcing clients to a
    # second endpoint to answer "is anything loaded?".
    for entry in data:
        _decorate_openai_entry(state, entry)
    return JSONResponse({"object": "list", "data": data})


def _decorate_openai_entry(state: Any, entry: dict[str, Any]) -> None:
    """Overlay what is resident onto one ``openai_dict`` entry, in place.

    One implementation for the list and the single-model endpoint: the same
    model must not read ``loaded`` from ``GET /v1/models`` and have no
    ``state`` at all from ``GET /v1/models/{id}``.
    """
    instance = state.supervisor.get(_serving_id(state, entry["id"]))
    loaded = instance is not None and instance.state == "ready"
    # "loading" is neither: a client that reads not-loaded and issues a
    # second load, or MCP list_models saying loading while this said
    # not-loaded, was the confusion. LM Studio's own vocabulary is
    # loaded/not-loaded; the third value is additive.
    if loaded:
        entry["state"] = "loaded"
    elif instance is not None and instance.state == "loading":
        entry["state"] = "loading"
    else:
        entry["state"] = "not-loaded"
    if loaded and instance is not None and instance.plan is not None:
        plan = instance.plan
        entry["loaded_context_length"] = plan.ctx_size
        # Concurrency, in the vendor block where a strict OpenAI client
        # ignores it. `loaded_context_length` alone is ambiguous once a
        # model runs multiple slots: --ctx-size is the TOTAL across slots
        # (D4), so a client that reads it as "what one conversation gets"
        # is right only at parallel 1. ctx_per_slot says which one it is,
        # and max_parallel says how many streams the load was planned for
        # -- the number a client should match its own concurrency to.
        entry["studioforge"]["ctx_per_slot"] = plan.ctx_per_slot or plan.ctx_size
        entry["studioforge"]["max_parallel"] = plan.max_parallel
        entry["studioforge"]["parallel"] = plan.parallel
        entry["studioforge"]["parallel_limited_by"] = plan.parallel_limited_by
    entry["studioforge"]["state"] = entry["state"]


def _serving_id(state: Any, model_id: str) -> str:
    """The instance key that serves ``model_id``.

    Differs from ``model_id`` only for preset-only virtual models, which share
    their base's instance; reporting those as loaded whenever the base is
    loaded is what makes `state` truthful for them.
    """
    record = state.registry.get(model_id)
    if record is None:
        return model_id
    return str(state.manager.serving_record(record).id)


@router.get("/v1/models/{model_id:path}")
async def retrieve_model(model_id: str, request: Request) -> JSONResponse:
    state = _app_state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    entry = record.openai_dict()
    _decorate_openai_entry(state, entry)
    return JSONResponse(entry)


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Chat completions, with every check done *before* the first byte.

    Ordering matters. Resolution and validation run against the registry
    record, which needs no load, so a bad request still gets a real 4xx status
    instead of an error frame buried in a 200 SSE stream -- clients handle the
    former and routinely mishandle the latter. Only the load itself is deferred
    into the stream, where it is covered by keep-alives.
    """
    state = _app_state(request)
    body = await _json_body(request)
    model_name = _require_model(body, state.config)

    # Resolve WITHOUT loading, so validation can happen up front.
    record = _resolve_or_404(state, model_name)
    payload = dict(body)

    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise BadRequestError("'messages' must be a non-empty array", param="messages")
    if any(not isinstance(message, dict) for message in messages):
        # Otherwise the first `.get` inside prepare_messages raises
        # AttributeError and a plainly malformed request comes back as a 500.
        raise BadRequestError(
            "every element of 'messages' must be an object with 'role' and 'content'",
            param="messages",
        )

    prepared, _stats = await prepare_messages(
        messages, record=record, config=state.config, client=state.client
    )
    payload["messages"] = prepared
    payload["model"] = record.id

    _validate_tools(payload)
    _validate_response_format(payload)
    _normalize_sampler_aliases(payload)
    ttl_override = _pop_ttl(payload)
    # After alias normalization, so a client's `repetition_penalty` counts as
    # an explicit choice that blocks the preset's repeat_penalty default.
    if record.preset is not None:
        record.preset.apply_to_payload(payload, chat=True)
    # A preset-only virtual model is served by its base's instance; everything
    # from here down (load, URL, request accounting) keys off `serving`.
    serving = state.manager.serving_record(record)

    if payload.get("stream"):
        # Load inside the stream so a multi-minute cold start is covered by
        # keep-alive comments rather than silence.
        return StreamingResponse(
            _stream_with_jit_load(state, serving, payload, ttl_override),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    await state.manager.ensure_loaded(serving.id, source="jit:/v1/chat/completions")
    _apply_ttl_override(state, serving.id, ttl_override)
    return await _forward(state, serving, "/v1/chat/completions", payload)


def _resolve_or_404(state: Any, name: str) -> ModelRecord:
    record = state.registry.resolve(name)
    if record is None:
        raise ModelNotFoundError(name, known=state.registry.known_ids())
    return record


async def _stream_with_jit_load(
    state: Any, record: ModelRecord, payload: dict[str, Any], ttl_override: int | None
) -> AsyncIterator[bytes]:
    """Emit SSE keep-alives while a model loads, then proxy the real stream.

    A cold load of a large model takes minutes. Without this the socket is
    silent for that whole time and a client's read timeout fires on a load that
    is progressing perfectly well -- one of the most common ways a local LLM
    server looks broken when it is not. ``:`` comment lines are valid SSE that
    every compliant parser ignores.
    """
    loader = asyncio.ensure_future(
        state.manager.ensure_loaded(record.id, source="jit:/v1/chat/completions")
    )
    # If the client disconnects mid-load this generator is closed while the task
    # is still running. The load is deliberately NOT cancelled -- the model will
    # finish loading and then idle-unload on its TTL, which is much cheaper than
    # abandoning a half-started child process. But an unobserved task exception
    # would surface as an "exception was never retrieved" warning at GC time, so
    # attach a consumer for that case.
    loader.add_done_callback(_consume_load_exception)
    interval = state.config.gateway.stream_keepalive_interval_s
    waited = 0.0
    try:
        while True:
            done, _ = await asyncio.wait({loader}, timeout=interval)
            if done:
                break
            waited += interval
            yield f": loading {record.id} ({waited:.0f}s)\n\n".encode()
        await loader
    except StudioForgeError as exc:
        log.warning("jit load failed mid-stream", model_id=record.id, error=exc.message)
        yield _sse_error(exc.message, code=exc.code or "model_load_failed")
        yield b"data: [DONE]\n\n"
        return
    except Exception as exc:  # pragma: no cover - defensive
        log.exception("unexpected jit load failure", model_id=record.id)
        yield _sse_error(f"failed to load '{record.id}': {exc}", code="model_load_failed")
        yield b"data: [DONE]\n\n"
        return

    _apply_ttl_override(state, record.id, ttl_override)
    base = state.supervisor.base_url(record.id)
    if base is None:
        yield _sse_error(f"model '{record.id}' is not serving", code="upstream_error")
        yield b"data: [DONE]\n\n"
        return

    started = time.perf_counter()
    async for chunk in _stream_upstream(
        state, record, f"{base}/v1/chat/completions", payload, started
    ):
        yield chunk


def _consume_load_exception(task: asyncio.Task[Any]) -> None:
    """Observe a detached load task's exception so it is not reported at GC."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning("background load failed after client disconnect", error=str(exc))


@router.post("/v1/completions")
async def completions(request: Request) -> Any:
    state = _app_state(request)
    body = await _json_body(request)
    model_name = _require_model(body, state.config)
    record = _resolve_or_404(state, model_name)
    payload = dict(body)
    payload["model"] = record.id
    if "prompt" not in payload:
        raise BadRequestError("'prompt' is required", param="prompt")
    _normalize_sampler_aliases(payload)
    ttl_override = _pop_ttl(payload)
    if record.preset is not None:
        # Sampler defaults only: there are no messages to carry a system prompt.
        record.preset.apply_to_payload(payload, chat=False)
    serving = state.manager.serving_record(record)
    await state.manager.ensure_loaded(serving.id, source="jit:/v1/completions")
    _apply_ttl_override(state, serving.id, ttl_override)
    return await _forward(state, serving, "/v1/completions", payload)


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Any:
    state = _app_state(request)
    body = await _json_body(request)
    model_name = _require_model(body, state.config)
    # Resolve without loading: a guaranteed-400 request must never trigger a
    # multi-minute load that could also evict resident models first.
    record = _resolve_or_404(state, model_name)
    if record.kind != "embedding" and not record.capabilities.embedding:
        # llama-server only exposes /v1/embeddings on an instance launched with
        # --embedding, so this would otherwise fail confusingly upstream.
        raise BadRequestError(
            f"Model '{record.id}' is not an embedding model. Embedding models are "
            f"launched with a dedicated instance; pick a model whose capabilities "
            f"include 'embedding'.",
            code="not_an_embedding_model",
            param="model",
        )
    payload = dict(body)
    payload["model"] = record.id
    if "input" not in payload:
        raise BadRequestError("'input' is required", param="input")
    serving = state.manager.serving_record(record)
    await state.manager.ensure_loaded(serving.id, source="jit:/v1/embeddings")
    return await _forward(state, serving, "/v1/embeddings", payload)


@router.post("/v1/rerank")
async def rerank(request: Request) -> Any:
    state = _app_state(request)
    body = await _json_body(request)
    record = _resolve_or_404(state, _require_model(body, state.config))
    payload = dict(body)
    payload["model"] = record.id
    serving = state.manager.serving_record(record)
    await state.manager.ensure_loaded(serving.id, source="jit:/v1/rerank")
    return await _forward(state, serving, "/v1/rerank", payload)


@router.post("/v1/tokenize")
async def tokenize(request: Request) -> Any:
    state = _app_state(request)
    body = await _json_body(request)
    record = _resolve_or_404(state, _require_model(body, state.config))
    serving = state.manager.serving_record(record)
    await state.manager.ensure_loaded(serving.id, source="jit:/v1/tokenize")
    return await _forward(state, serving, "/tokenize", dict(body))


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise BadRequestError(f"request body is not valid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise BadRequestError("request body must be a JSON object")
    return body


#: Names clients send meaning "whatever this server serves". LM Studio clients
#: fall back to the literal "local-model", so 404-ing it breaks them needlessly.
DEFAULT_MODEL_ALIASES = frozenset({"local-model", "default", "auto", "current"})


def _require_model(body: dict[str, Any], config: Any = None) -> str:
    """Resolve the request's model name, honouring the configured default.

    A missing ``model``, or one of :data:`DEFAULT_MODEL_ALIASES`, resolves to
    ``models.default_model`` when it is set. That is what makes "just point
    OpenClaw at it and go" work without the client naming a model.
    """
    model = body.get("model")
    name = model.strip() if isinstance(model, str) else ""
    default = getattr(getattr(config, "models", None), "default_model", None)

    if not name or name.lower() in DEFAULT_MODEL_ALIASES:
        if default:
            return str(default)
        if not name:
            raise BadRequestError(
                "'model' is required. Set models.default_model in config.yaml to "
                "serve a model when the client does not name one.",
                param="model",
            )
        raise BadRequestError(
            f"'{name}' means 'use the server default', but models.default_model "
            f"is not set in config.yaml.",
            param="model",
            code="no_default_model",
        )
    return name


def _normalize_sampler_aliases(payload: dict[str, Any]) -> None:
    """Accept both spellings of the repetition penalty.

    llama.cpp calls it ``repeat_penalty``; HuggingFace-flavoured clients send
    ``repetition_penalty``. LM Studio accepts only the former and *silently
    ignores* the latter, which makes a sampler look broken with no error at
    all. Translating instead of ignoring is strictly better.
    """
    if "repetition_penalty" in payload and "repeat_penalty" not in payload:
        payload["repeat_penalty"] = payload.pop("repetition_penalty")


def _pop_ttl(payload: dict[str, Any]) -> int | None:
    """Take LM Studio's request-level ``ttl`` out of the upstream payload.

    LM Studio lets a client attach a TTL to a chat request so JIT-loaded models
    self-evict. llama-server does not know the field, so it is consumed here
    and applied to our own idle timer instead of being forwarded.

    Anything that is not a positive number of seconds is *no override*. In
    particular ``0`` -- and a negative or sub-second value, which ``int``
    rounds to it -- is the wire form of **pinned** everywhere (the sweeper
    never idle-unloads it, the planner excludes it from every eviction ladder,
    a lease refuses it), and pinning is a box change behind the D32 gate. A
    request may shorten or lengthen the idle timer; it may not pin, which is
    the mirror of D41 item 4: it may not unpin either. Returning ``None``
    rather than clamping keeps to what the docs promise ("the idle timer
    resets to it") instead of inventing a remote "unload as soon as idle".
    """
    if "ttl" not in payload:
        return None
    raw = payload.pop("ttl")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    seconds = int(raw)
    return seconds if seconds > 0 else None


def _apply_ttl_override(state: Any, model_id: str, ttl_s: int | None) -> None:
    # `ttl_s <= 0` is refused here as well as in `_pop_ttl`: a direct caller
    # must not be able to write the pinned wire value onto an instance either.
    if ttl_s is None or ttl_s <= 0:
        return
    instance = state.supervisor.get(model_id)
    if instance is None:
        return
    if instance.ttl_s == 0:
        # ttl_s == 0 is the wire representation of *pinned* everywhere (the
        # sweeper and the eviction planner both read it off the instance), so
        # honouring a request-level ttl here would let any client silently
        # unpin a model the owner pinned. The request still works; only its
        # idle-timer wish is ignored.
        return
    instance.ttl_s = ttl_s


def _validate_tools(payload: dict[str, Any]) -> None:
    """Fail fast on malformed tool definitions rather than deep in the engine."""
    tools = payload.get("tools")
    if tools is None:
        return
    if not isinstance(tools, list):
        raise BadRequestError("'tools' must be an array", param="tools")
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise BadRequestError(
                f"tools[{index}] must be an object with type='function'", param="tools"
            )
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            raise BadRequestError(f"tools[{index}].function.name is required", param="tools")
    choice = payload.get("tool_choice")
    if choice is not None and not isinstance(choice, (str, dict)):
        raise BadRequestError("'tool_choice' must be a string or object", param="tool_choice")


def _validate_response_format(payload: dict[str, Any]) -> None:
    """Validate ``response_format`` and make ``json_object`` actually binding.

    llama.cpp only compiles a grammar for ``json_schema``; it treats
    ``json_object`` as a hint, so a small model happily answers with a
    markdown-fenced `````json`` block that ``json.loads`` rejects.
    OpenAI's contract is that ``json_object`` *guarantees* parseable JSON, so we
    upgrade it to a permissive object schema, which llama.cpp does enforce with
    a grammar. Verified against the real engine: without this, the contract test
    for ``json_object`` fails.
    """
    fmt = payload.get("response_format")
    if fmt is None:
        return
    if not isinstance(fmt, dict):
        raise BadRequestError("'response_format' must be an object", param="response_format")
    kind = fmt.get("type")
    if kind not in {"text", "json_object", "json_schema"}:
        raise BadRequestError(
            "response_format.type must be one of 'text', 'json_object', 'json_schema'",
            param="response_format",
        )
    if kind == "json_schema":
        spec = fmt.get("json_schema")
        if not isinstance(spec, dict) or "schema" not in spec:
            raise BadRequestError(
                "response_format.json_schema must be an object containing 'schema'",
                param="response_format",
            )
    elif kind == "json_object":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "json_object", "schema": {"type": "object"}},
        }


# ---------------------------------------------------------------------------
# Upstream forwarding
# ---------------------------------------------------------------------------


async def _forward(state: Any, record: ModelRecord, path: str, payload: dict[str, Any]) -> Any:
    base = state.supervisor.base_url(record.id)
    if base is None:
        raise UpstreamError(f"model '{record.id}' is not serving")
    url = f"{base}{path}"
    stream = bool(payload.get("stream"))

    started = time.perf_counter()

    if stream:
        # The generator increments AND decrements the request counter itself.
        # Incrementing here would leak the slot whenever the client disconnects
        # before the response body is first iterated -- StreamingResponse does
        # not start the generator until then.
        return StreamingResponse(
            _stream_upstream(state, record, url, payload, started),
            media_type="text/event-stream",
            headers=SSE_HEADERS,
        )

    # One `finally` for the whole non-streamed exchange: only httpx errors used
    # to decrement, so a cancellation (client gone, shutdown) or any exception
    # outside the httpx hierarchy left active_requests stuck at 1 -- which
    # blocks TTL unload and eviction for that model until a restart. The
    # streaming path has had this discipline for a while; this is its twin.
    state.supervisor.mark_request_start(record.id)
    tps: float | None = None
    try:
        try:
            response = await state.client.post(
                url,
                json=payload,
                timeout=httpx.Timeout(state.config.server.request_timeout_s, connect=10.0),
            )
        except httpx.ReadTimeout as exc:
            raise UpstreamError(
                f"llama-server for '{record.id}' did not answer within "
                f"{state.config.server.request_timeout_s:.0f} s (server.request_timeout_s). "
                "A long generation should stream (stream: true) so no single read waits "
                "for the whole reply.",
                code="upstream_timeout",
                status_code=504,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(
                f"llama-server for '{record.id}' failed: {exc}. {_stderr_hint(state, record.id)}"
            ) from exc

        elapsed = time.perf_counter() - started
        tps = _tokens_per_second(response, elapsed)
        if response.status_code >= 400:
            raise UpstreamError(
                _upstream_message(response, record.id, state),
                status_code=response.status_code if response.status_code < 500 else 502,
            )
        data = response.json()
    finally:
        state.supervisor.mark_request_end(record.id, tokens_per_second=tps)

    if state.config.gateway.merge_reasoning_into_content:
        _merge_reasoning(data, record.id)
    return JSONResponse(data, status_code=response.status_code)


def _merge_reasoning(data: Any, model_id: str) -> None:
    """Never return an empty ``content`` when reasoning text exists.

    ``reasoning_content`` is not part of the OpenAI chat-completions schema, so
    a client reading ``choices[0].message.content`` sees "" and concludes the
    model said nothing. That is the normal case for a thinking model whose
    token budget was spent reasoning. The default launch flag
    (``--reasoning-format none``) prevents the split in the first place; this is
    the safety net for anyone who switches it to ``deepseek``.
    """
    if not isinstance(data, dict):
        return
    for choice in data.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content:
            continue
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            message["content"] = reasoning
            # Mirror both spellings: older clients read reasoning_content, the
            # newer OpenAI-style spec reads `reasoning`.
            message.setdefault("reasoning_content", reasoning)
            message.setdefault("reasoning", reasoning)
            log.info(
                "merged reasoning into empty content",
                model_id=model_id,
                chars=len(reasoning),
            )


async def _stream_upstream(
    state: Any, record: ModelRecord, url: str, payload: dict[str, Any], started: float
) -> AsyncIterator[bytes]:
    """Pass SSE through verbatim, and never leave the stream unterminated.

    Clients treat a stream that stops without ``data: [DONE]`` as a hang, so
    any upstream failure is converted into an error event followed by the
    sentinel.
    """
    completion_tokens = 0
    sent_done = False
    closing = False
    #: The pending read of the first upstream chunk while keep-alives run. Held
    #: at function scope so the finally can cancel it if the client disconnects
    #: mid-prefill -- an orphaned read task would hold the httpx stream open.
    first: asyncio.Task[bytes] | None = None
    state.supervisor.mark_request_start(record.id)
    try:
        async with state.client.stream(
            "POST", url, json=payload, timeout=state.config.server.request_timeout_s
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                message = _upstream_message(response, record.id, state)
                yield _sse_error(message, code="upstream_error")
                yield b"data: [DONE]\n\n"
                sent_done = True
                return
            stream = response.aiter_raw()
            # Keep the socket warm until the FIRST byte. On a busy server that
            # first byte is the end of prefill, and prefill grows with
            # concurrency (eight cold slots re-processing a shared prompt can be
            # tens of seconds); a silent socket that whole time trips a client's
            # read timeout, and a retrying client piles more prefill onto an
            # already-saturated batch -- the self-amplifying failure behind
            # "trouble with 8 concurrent". A ``:`` comment is valid SSE every
            # parser ignores. Only until the first chunk: after that the stream
            # is flowing, so there is no per-chunk timer overhead.
            interval = state.config.gateway.stream_keepalive_interval_s
            first = asyncio.ensure_future(stream.__anext__())  # type: ignore[arg-type]
            waited = 0.0
            while True:
                done, _ = await asyncio.wait({first}, timeout=interval)
                if done:
                    break
                # ``wait`` did NOT cancel the read; the same task is still
                # pending and we re-wait on it. Cancelling it here would abort
                # the httpx read mid-flight and corrupt the stream.
                waited += interval
                yield f": prefilling {record.id} ({waited:.0f}s)\n\n".encode()
            try:
                first_chunk = first.result()
            except StopAsyncIteration:
                first_chunk = None
            if first_chunk:
                if b"[DONE]" in first_chunk:
                    sent_done = True
                completion_tokens += first_chunk.count(b'"delta"')
                yield first_chunk
            if not sent_done:
                async for chunk in stream:
                    if not chunk:
                        continue
                    if b"[DONE]" in chunk:
                        sent_done = True
                    completion_tokens += chunk.count(b'"delta"')
                    yield chunk
    except GeneratorExit:
        # The client went away. Nothing may be yielded during this unwind --
        # doing so raises RuntimeError("async generator ignored GeneratorExit")
        # out of aclose(), which is noise at best and can mask the real reason
        # the connection ended.
        closing = True
        raise
    except httpx.HTTPError as exc:
        log.warning("stream failed", model_id=record.id, error=str(exc))
        yield _sse_error(
            f"llama-server for '{record.id}' failed mid-stream: {exc}. "
            f"{_stderr_hint(state, record.id)}",
            code="upstream_error",
        )
    finally:
        # Cancel a first-chunk read still pending when the client vanished
        # mid-prefill, so it does not hold the httpx stream open after us.
        if first is not None and not first.done():
            first.cancel()
        # Release the request slot FIRST. When a client disconnects, this
        # generator is closed with GeneratorExit, and yielding during that
        # unwind raises RuntimeError("async generator ignored GeneratorExit") --
        # which would skip everything after the yield. Decrementing after the
        # sentinel therefore leaked active_requests on every disconnect, and a
        # stuck counter permanently blocks both TTL-unload and eviction for that
        # model: a one-client hang-up would pin VRAM until restart.
        elapsed = time.perf_counter() - started
        tps = round(completion_tokens / elapsed, 2) if elapsed > 0 and completion_tokens else None
        state.supervisor.mark_request_end(record.id, tokens_per_second=tps)
        # Only terminate the stream if there is still a client to terminate it
        # for. Yielding during a GeneratorExit unwind raises RuntimeError.
        if not sent_done and not closing:
            yield b"data: [DONE]\n\n"


def _sse_error(message: str, *, code: str) -> bytes:
    payload = {"error": {"message": message, "type": "server_error", "code": code}}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _upstream_message(response: httpx.Response, model_id: str, state: Any) -> str:
    detail = ""
    try:
        data = response.json()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                detail = str(error.get("message") or "")
            elif isinstance(error, str):
                detail = error
    except Exception:
        detail = (response.text or "")[:500]
    base = f"llama-server for '{model_id}' returned HTTP {response.status_code}"
    if detail:
        base += f": {detail}"
    return base


def _stderr_hint(state: Any, model_id: str) -> str:
    """Last stderr lines from the child -- the difference between a usable
    error and a mystery."""
    try:
        lines = state.supervisor.tail_log(model_id, 8)
    except Exception:
        return ""
    if not lines:
        return ""
    tail = " | ".join(line.strip() for line in lines if line.strip())
    return f"Recent llama-server output: {tail[:800]}"


def _tokens_per_second(response: httpx.Response, elapsed: float) -> float | None:
    if elapsed <= 0:
        return None
    try:
        data = response.json()
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None
    tokens = usage.get("completion_tokens")
    if not isinstance(tokens, int) or tokens <= 0:
        return None
    return round(tokens / elapsed, 2)


# ---------------------------------------------------------------------------
# LM Studio's /api/v0 surface
# ---------------------------------------------------------------------------
#
# LM Studio ships a second, richer API alongside the OpenAI one, and clients
# written against it reach for /api/v0/models to learn each model's *state*.
# StudioForge also puts `state` on /v1/models (see list_models) so the standard
# endpoint is authoritative on its own -- but a drop-in replacement that omitted
# /api/v0 entirely would still break those clients, so it is implemented here.


def _v0_model_entry(state: Any, record: ModelRecord) -> dict[str, Any]:
    instance = state.supervisor.get(state.manager.serving_record(record).id)
    loaded = instance is not None and instance.state == "ready"
    entry: dict[str, Any] = {
        "id": record.id,
        "object": "model",
        "type": _v0_type(record),
        "publisher": record.publisher or "local",
        "arch": record.architecture,
        "compatibility_type": "gguf",
        "quantization": record.quant,
        "state": "loaded" if loaded else "not-loaded",
        "max_context_length": record.meta.n_ctx_train if record.meta else None,
        "capabilities": [k for k, v in record.capabilities.model_dump().items() if v],
    }
    if loaded and instance is not None and instance.plan is not None:
        entry["loaded_context_length"] = instance.plan.ctx_size
    return entry


def _v0_type(record: ModelRecord) -> str:
    """LM Studio's model ``type``.

    Delegates to the catalog so the two surfaces cannot drift: an agent that
    reads ``type: "vlm"`` from ``/api/v0/models`` and ``type`` from
    ``/api/catalog`` must get the same word for the same model.
    """
    from studioforge.core.catalog import model_type

    return model_type(record)


@router.get("/api/v0/models")
async def v0_list_models(request: Request) -> JSONResponse:
    """LM Studio's model list, including per-model load state."""
    state = _app_state(request)
    return JSONResponse(
        {
            "object": "list",
            "data": [_v0_model_entry(state, r) for r in state.registry.all()],
        }
    )


@router.get("/api/v0/models/{model_id:path}")
async def v0_retrieve_model(model_id: str, request: Request) -> JSONResponse:
    state = _app_state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    return JSONResponse(_v0_model_entry(state, record))


@router.post("/api/v0/chat/completions")
async def v0_chat_completions(request: Request) -> Any:
    return await chat_completions(request)


@router.post("/api/v0/completions")
async def v0_completions(request: Request) -> Any:
    return await completions(request)


@router.post("/api/v0/embeddings")
async def v0_embeddings(request: Request) -> Any:
    return await embeddings(request)
