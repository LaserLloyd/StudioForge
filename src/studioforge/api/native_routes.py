"""StudioForge as a clean superset of stock ``llama-server`` (D90).

A client written for ``llama-server`` -- the router vocabulary and the native
per-model endpoints -- works against StudioForge unchanged; a client that knows
StudioForge gets the extras. This module is the root-level surface:

* **Router vocabulary** (``llama-server`` b12110 ``tools/server/README.md``):
  ``GET /models`` (``?reload=1`` rescans), ``POST /models/load`` and
  ``POST /models/unload`` (``{"model": ...}`` -> ``{"success": true}``), and
  ``?autoload=true|false`` per request. ``POST /models/load?dryRun=true`` is the
  canonical dry run: the same decision as the real load, nothing started.
* **Native routes**: ``/props``, ``/slots`` (GET), ``/tokenize``, ``/detokenize``,
  ``/apply-template``, ``/completion``, ``/embedding``, ``/infill`` and
  ``/reranking`` (POST). The child's answer is forwarded byte for byte; the only
  body StudioForge touches is ``/props``, where it ADDS a ``studioforge`` block.
* **KServe v2 probes**: ``/v2/health/live``, ``/v2/health/ready``,
  ``/v2/models/{id}/ready``.

**Which model.** Exactly llama-server's router rule: a POST names it in the JSON
body (``model``), a GET in ``?model=``; ``X-SF-Model`` is accepted on both. With no
selector, the one resident model that can serve the route is used; with none or
several it is a 400 ``model_required`` listing the choices. (On a box that sets
``models.default_model`` the default is used instead of the 400.) A named model
that is not resident is loaded just in time on a POST, exactly as on ``/v1/*``; a
GET never loads (a poll must not start a multi-gigabyte load) unless it says
``?autoload=true``. ``?autoload=false`` on a POST refuses the load with 404
``model_not_loaded``.

Errors on these routes keep llama-server's / OpenAI's shape (``error{code,
message, type}``); problem+json negotiation is for the management routes only.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from studioforge.api.auth import require_admin_action
from studioforge.api.identity import autoload_requested, enforce_autoload, note_serving
from studioforge.api.openai_routes import (
    SSE_HEADERS,
    _exc_text,
    _json_body,
    _note_client,
    _require_model,
    _resolve_loaded_alias,
    _resolve_or_404,
    _serves_want,
    _stderr_hint,
    _wait_for_scan,
    architecture_modalities,
)
from studioforge.core.attribution import attributed_source, client_of
from studioforge.errors import BadRequestError, ModelNotFoundError, UpstreamError
from studioforge.logging import get_logger
from studioforge.types import ModelRecord

log = get_logger(__name__)

router = APIRouter()

#: Query parameters that select or steer StudioForge, not the child.
_OWN_QUERY = frozenset({"model", "autoload"})


def _state(request: Request) -> Any:
    return request.app.state


# ---------------------------------------------------------------------------
# The model selector
# ---------------------------------------------------------------------------


def _selector(request: Request, body: dict[str, Any] | None) -> str | None:
    """The model the caller named: body ``model``, else ``?model=``, else ``X-SF-Model``."""
    candidates = (
        body.get("model") if body else None,
        request.query_params.get("model"),
        request.headers.get("x-sf-model"),
    )
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resident_candidates(state: Any, want: str | None) -> list[str]:
    """Ready instances that can serve a ``want`` request (any kind when ``None``)."""
    found: list[str] = []
    for instance in state.supervisor.list():
        if instance.state != "ready":
            continue
        record = state.registry.get(instance.model_id)
        if record is None:
            continue
        if want is None or _serves_want(record, want):
            found.append(instance.model_id)
    return found


def _model_for_no_selector(state: Any, want: str | None) -> str:
    """The one resident model, else ``models.default_model``, else 400 ``model_required``."""
    candidates = _resident_candidates(state, want)
    if len(candidates) == 1:
        return candidates[0]
    default = getattr(getattr(state.config, "models", None), "default_model", None)
    if default:
        return str(default)
    ways = (
        'the "model" field of the JSON body (POST routes), the ?model= query parameter '
        "(GET routes), or the X-SF-Model header"
    )
    if candidates:
        message = (
            f"No model was named and {len(candidates)} loaded models could serve this "
            f"request ({', '.join(sorted(candidates))}). Name one with {ways}."
        )
    else:
        message = (
            f"No model was named and none is loaded that can serve this request. Name one "
            f"with {ways}; POST routes load it just in time."
        )
    raise BadRequestError(
        message,
        param="model",
        code="model_required",
        details={"resident": sorted(candidates), "resident_count": len(candidates)},
    )


async def _select(
    request: Request,
    state: Any,
    body: dict[str, Any] | None,
    *,
    want: str | None,
    load: bool,
    route: str,
) -> tuple[ModelRecord, ModelRecord, str | None]:
    """Resolve the model for one native request and make it ready.

    Returns ``(record, serving, client)``: the record the caller named, the
    record whose child answers (a preset-only virtual model is served by its
    base), and the attribution label. ``load`` is the route's default for just-
    in-time loading (``True`` for POSTs, ``False`` for GET polls);
    ``?autoload=`` overrides it per request.
    """
    name = _selector(request, body)
    if name is None:
        model_name = _model_for_no_selector(state, want)
    else:
        model_name = _resolve_loaded_alias(
            state,
            _require_model({"model": name}, state.config),
            want=want or "chat",
        )
    record = _resolve_or_404(state, model_name)
    if want == "embedding" and record.kind != "embedding" and not record.capabilities.embedding:
        raise BadRequestError(
            f"Model '{record.id}' is not an embedding model. Embedding models are launched "
            f"with a dedicated instance; pick a model whose capabilities include 'embedding'.",
            code="not_an_embedding_model",
            param="model",
        )
    # D91: a base id from a caller holding its own instance of it is routed there.
    from studioforge.api.holder import routed_serving

    lease_value = body.get("lease_id") if body else None
    serving = routed_serving(
        state, request, record, lease_ids=[lease_value] if isinstance(lease_value, str) else []
    )
    note_serving(serving.id)
    _note_client(state, request, serving.id)
    client = client_of(request)
    if autoload_requested(request, default=load):
        await state.manager.ensure_loaded(
            serving.id, source=attributed_source(f"jit:{route}", client)
        )
    else:
        enforce_autoload(request, state, serving.id, default=False)
    return record, serving, client


# ---------------------------------------------------------------------------
# Forwarding (the child's answer, unchanged)
# ---------------------------------------------------------------------------


def _passthrough(response: httpx.Response) -> Response:
    """The child's status and body exactly as it sent them."""
    content_type = response.headers.get("content-type") or "application/json"
    return Response(
        content=response.content,
        status_code=response.status_code,
        media_type=None,
        headers={"content-type": content_type},
    )


def _upstream_failure(state: Any, serving: ModelRecord, exc: httpx.HTTPError) -> UpstreamError:
    if isinstance(exc, httpx.ReadTimeout):
        return UpstreamError(
            f"llama-server for '{serving.id}' did not answer within "
            f"{state.config.server.request_timeout_s:.0f} s (server.request_timeout_s). "
            "A long generation should stream (stream: true).",
            code="upstream_timeout",
            status_code=504,
        )
    return UpstreamError(
        f"llama-server for '{serving.id}' failed: {_exc_text(exc)}. "
        f"{_stderr_hint(state, serving.id)}"
    )


def _child_url(state: Any, serving: ModelRecord, path: str) -> str:
    base = state.supervisor.base_url(serving.id)
    if base is None:
        raise UpstreamError(f"model '{serving.id}' is not serving")
    return f"{base}{path}"


async def _forward_post(
    state: Any,
    serving: ModelRecord,
    path: str,
    payload: dict[str, Any],
    *,
    client: str | None,
) -> Response:
    """POST ``payload`` to the child's ``path``; stream it back when ``stream`` is true.

    The request is counted in flight for the whole exchange (TTL unload and the
    eviction ladder must not stop a model mid-answer), and the child's status
    and body are returned unchanged -- its own error JSON included, because a
    llama-server client parses llama-server's errors.
    """
    url = _child_url(state, serving, path)
    if payload.get("stream"):
        return await _forward_stream(state, serving, url, payload, client=client)
    request_id = state.supervisor.mark_request_start(serving.id, client=client)
    try:
        try:
            response = await state.client.post(
                url,
                json=payload,
                timeout=httpx.Timeout(state.config.server.request_timeout_s, connect=10.0),
            )
        except httpx.HTTPError as exc:
            raise _upstream_failure(state, serving, exc) from exc
    finally:
        state.supervisor.mark_request_end(serving.id, request_id=request_id)
    return _passthrough(response)


async def _forward_stream(
    state: Any,
    serving: ModelRecord,
    url: str,
    payload: dict[str, Any],
    *,
    client: str | None,
) -> Response:
    """A streamed native answer, bytes untouched (no ``[DONE]`` is added).

    The upstream is opened first so a child refusal (a 400 for a bad grammar) is
    a real status the client can read; the request is counted in flight inside the
    generator, which is the only place a disconnect is guaranteed to reach.
    """
    manager = state.client.stream(
        "POST", url, json=payload, timeout=state.config.server.request_timeout_s
    )
    try:
        upstream = await manager.__aenter__()
    except httpx.HTTPError as exc:
        raise _upstream_failure(state, serving, exc) from exc
    content_type = upstream.headers.get("content-type") or "text/event-stream"
    if upstream.status_code >= 400:
        try:
            body = await upstream.aread()
        finally:
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)
        return Response(
            content=body,
            status_code=upstream.status_code,
            headers={"content-type": content_type},
        )

    async def relay() -> Any:
        request_id = state.supervisor.mark_request_start(serving.id, client=client)
        try:
            async for chunk in upstream.aiter_raw():
                if chunk:
                    yield chunk
        except httpx.HTTPError as exc:
            # Nothing can be added to a stream that has started; the truncated
            # body is the signal, the log has the reason.
            log.warning("native stream failed", model_id=serving.id, error=_exc_text(exc))
        finally:
            state.supervisor.mark_request_end(serving.id, request_id=request_id)
            with contextlib.suppress(Exception):
                await manager.__aexit__(None, None, None)

    return StreamingResponse(
        relay(),
        status_code=upstream.status_code,
        media_type=content_type,
        headers=SSE_HEADERS,
    )


async def _native_post(request: Request, path: str, *, want: str | None, apply_preset: bool) -> Any:
    state = _state(request)
    body = await _json_body(request)
    record, serving, client = await _select(request, state, body, want=want, load=True, route=path)
    payload = dict(body)
    if "model" in payload:
        # The id that answers: an alias ("loaded", a default name) must not be
        # echoed back to the caller as the model.
        payload["model"] = record.id
    if apply_preset and record.preset is not None:
        record.preset.apply_to_payload(payload, chat=False)
    return await _forward_post(state, serving, path, payload, client=client)


@router.post("/tokenize")
async def native_tokenize(request: Request) -> Any:
    """llama-server ``POST /tokenize`` (``content``, ``add_special``, ``with_pieces``)."""
    return await _native_post(request, "/tokenize", want=None, apply_preset=False)


@router.post("/detokenize")
async def native_detokenize(request: Request) -> Any:
    """llama-server ``POST /detokenize`` (``tokens``)."""
    return await _native_post(request, "/detokenize", want=None, apply_preset=False)


@router.post("/apply-template")
async def native_apply_template(request: Request) -> Any:
    """llama-server ``POST /apply-template`` (``messages`` -> ``prompt``), no inference."""
    return await _native_post(request, "/apply-template", want="chat", apply_preset=False)


@router.post("/completion")
async def native_completion(request: Request) -> Any:
    """llama-server ``POST /completion`` (``prompt``, every native sampler field, ``stream``).

    The model's preset sampler defaults fill absent fields, as on ``/v1/completions``.
    """
    return await _native_post(request, "/completion", want="chat", apply_preset=True)


@router.post("/infill")
async def native_infill(request: Request) -> Any:
    """llama-server ``POST /infill``: prefix / suffix plus every ``/completion`` option."""
    return await _native_post(request, "/infill", want="chat", apply_preset=True)


@router.post("/embedding")
async def native_embedding(request: Request) -> Any:
    """llama-server ``POST /embedding`` (``content``): the non-OpenAI embedding shape."""
    return await _native_post(request, "/embedding", want="embedding", apply_preset=False)


@router.post("/reranking")
async def native_reranking(request: Request) -> Any:
    """llama-server ``POST /reranking`` (``query``, ``documents``, ``top_n``)."""
    return await _native_post(request, "/reranking", want="rerank", apply_preset=False)


@router.post("/rerank")
async def native_rerank_alias(request: Request) -> Any:
    """llama-server's alias of ``/reranking``."""
    return await _native_post(request, "/rerank", want="rerank", apply_preset=False)


# ---------------------------------------------------------------------------
# /props and /slots
# ---------------------------------------------------------------------------


def _child_params(request: Request) -> list[tuple[str, str]]:
    return [(k, v) for k, v in request.query_params.multi_items() if k not in _OWN_QUERY]


async def _proxy_get(
    request: Request, path: str
) -> tuple[httpx.Response, ModelRecord, ModelRecord]:
    state = _state(request)
    record, serving, _client = await _select(
        request, state, None, want=None, load=False, route=path
    )
    url = _child_url(state, serving, path)
    try:
        response = await state.client.get(
            url,
            params=_child_params(request),
            timeout=httpx.Timeout(min(60.0, state.config.server.request_timeout_s), connect=10.0),
        )
    except httpx.HTTPError as exc:
        raise _upstream_failure(state, serving, exc) from exc
    return response, record, serving


def _props_block(state: Any, record: ModelRecord, serving: ModelRecord) -> dict[str, Any]:
    """What StudioForge knows about the instance behind ``/props``, from its own books."""
    instance = state.supervisor.get(serving.id)
    block: dict[str, Any] = {
        "instance_id": serving.id,
        "model_id": record.id,
        "base_model_id": serving.base_model_id,
    }
    if instance is None:
        return block
    plan = instance.plan
    if plan is not None:
        block.update(
            {
                "devices": list(plan.devices),
                "split_mode": plan.split_mode,
                "tensor_split": plan.tensor_split,
                "parallel": plan.parallel,
                "ctx_per_slot": plan.ctx_per_slot or plan.ctx_size,
                "kv_cache_type": plan.kv_cache_type,
                "kv_cache_type_v": plan.kv_cache_type_v,
                "flash_attn": plan.flash_attn,
            }
        )
    effective = instance.effective
    if effective is not None:
        block["batch_size"] = effective.batch_size
        block["ubatch_size"] = effective.ubatch_size
    engine_tag = instance.resolved_engine_tag or instance.engine_tag
    block["engine"] = {"tag": engine_tag}
    lease_id = getattr(instance, "lease_id", None)
    if lease_id:
        block["lease_id"] = lease_id
    return block


@router.get("/props")
async def native_props(request: Request) -> Response:
    """llama-server ``GET /props``, plus a ``studioforge`` block.

    ``/props?model=<id>`` (or the one resident model). The child's JSON is passed
    through untouched except for the added ``studioforge`` key: instance id,
    placement, slots, KV types and engine tag.
    """
    state = _state(request)
    response, record, serving = await _proxy_get(request, "/props")
    if response.status_code != 200:
        return _passthrough(response)
    try:
        data = response.json()
    except ValueError:
        return _passthrough(response)
    if not isinstance(data, dict):
        return _passthrough(response)
    data["studioforge"] = _props_block(state, record, serving)
    return JSONResponse(data)


@router.post("/props")
async def native_props_post(request: Request) -> Response:
    """Refused: llama-server lets ``POST /props`` change global properties; here
    a model's settings are the single source of truth."""
    raise HTTPException(
        status_code=405,
        detail=(
            "POST /props is not supported: StudioForge's per-model settings are the source "
            "of truth. Change launch settings with PATCH /api/models/{id}/settings, then "
            "reload the model."
        ),
        headers={"Allow": "GET"},
    )


@router.get("/slots")
async def native_slots(request: Request) -> Response:
    """llama-server ``GET /slots`` for the selected model, the child's answer unchanged."""
    response, _record, _serving = await _proxy_get(request, "/slots")
    return _passthrough(response)


# ---------------------------------------------------------------------------
# Router vocabulary: GET /models, POST /models/load, POST /models/unload
# ---------------------------------------------------------------------------


def _router_entry(state: Any, record: ModelRecord, arch_memo: dict[Any, Any]) -> dict[str, Any]:
    serving = state.manager.serving_record(record)
    instance = state.supervisor.get(serving.id)
    status: dict[str, Any]
    if instance is not None and instance.state == "ready":
        status = {"value": "loaded"}
    elif instance is not None and instance.state == "loading":
        status = {"value": "loading"}
    else:
        status = {"value": "unloaded"}
    args = instance.launch_args if instance is not None else None
    if status["value"] != "unloaded" and args:
        status["args"] = list(args)
    failure: dict[str, Any] | None = None
    if status["value"] == "unloaded":
        try:
            failure = state.manager.load_support_fields(record, memo=arch_memo).get(
                "last_load_failure"
            )
        except Exception:  # noqa: BLE001 - a listing never fails on a verdict
            failure = None
    if failure is not None:
        status["failed"] = True
        if failure.get("exit_code") is not None:
            status["exit_code"] = failure["exit_code"]
    plan = instance.plan if instance is not None else None
    entry: dict[str, Any] = {
        "id": record.id,
        "object": "model",
        "path": str(record.path),
        "status": status,
        "architecture": architecture_modalities(record),
        "studioforge": {
            "kind": record.kind,
            "arch": record.architecture,
            "state": {"loaded": "loaded", "loading": "loading"}.get(status["value"], "not-loaded"),
            "is_virtual": record.is_virtual,
            "base_model_id": record.base_model_id,
            "size_bytes": record.size_bytes,
            "n_ctx_train": record.meta.n_ctx_train if record.meta else None,
            "loaded_context_length": plan.ctx_size if plan is not None else None,
        },
    }
    if record.mmproj_path is not None:
        entry["sidecars"] = ["mmproj"]
    if failure is not None:
        entry["studioforge"]["last_load_failure"] = dict(failure)
    return entry


@router.get("/models")
async def router_list_models(
    request: Request,
    reload: str | None = Query(None, description="1 / true rescans the library first."),
) -> JSONResponse:
    """llama-server router ``GET /models``: every model with ``status.value``.

    ``status.value`` is ``unloaded``, ``loading`` or ``loaded`` (``failed: true``
    on an unloaded model whose last launch died); ``architecture`` is the
    ``input_modalities`` / ``output_modalities`` pair; ``studioforge`` carries the
    rest. ``?reload=1`` rescans the model library first (admin, like
    ``POST /api/models/scan``).
    """
    state = _state(request)
    await _wait_for_scan(state)
    if (reload or "").strip().lower() in {"1", "true", "yes", "on"}:
        require_admin_action(request, state.config, "rescanning the model library")
        await asyncio.to_thread(state.registry.scan, force=False)
    arch_memo: dict[Any, Any] = {}
    # D91: a lease's instances are models a client may name, so they are listed too.
    try:
        records = state.registry.all(include_ephemeral=True)
    except TypeError:  # a registry stand-in without lease instances
        records = state.registry.all()
    data = [_router_entry(state, record, arch_memo) for record in records]
    return JSONResponse({"object": "list", "data": data})


class ModelsLoadBody(BaseModel):
    """``POST /models/load``: llama-server's ``{"model"}`` plus StudioForge's extras.

    ``context_length`` is tokens PER SLOT (LM Studio's option name; the same
    number as ``ctx_size`` on ``/api/models/{id}/load-recommended``). Unknown
    keys are ignored, as llama-server ignores them.
    """

    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    context_length: int | None = None
    ctx_size: int | None = None
    devices: list[int] | None = None
    allowed_devices: list[int] | None = None
    parallel: int | None = None
    min_slots: int | None = None
    max_slots: int | None = None
    kv_cache_type: str | None = None
    kv_cache_type_v: str | None = None
    kv_unified: bool | None = None
    kv_min: str | None = None
    prefer_mode: str | None = None
    priority: int | None = None
    lease_id: str | None = None
    persist: bool = False
    force: bool = False


class ModelsUnloadBody(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str | None = None
    #: D91 (8a): cut another client's in-flight requests on a lease-held model,
    #: or (with an admin credential) take another holder's exact instance down.
    force: bool = False


def _require_named(model: str | None) -> str:
    if not model or not model.strip():
        raise BadRequestError(
            'The "model" field is required: {"model": "<id>"}. GET /models lists the ids.',
            param="model",
            code="model_required",
        )
    return model.strip()


@router.post("/models/load")
async def router_load_model(
    request: Request,
    body: ModelsLoadBody,
    dry_run: bool = Query(False, alias="dryRun", description="Plan only; start nothing."),
) -> dict[str, Any]:
    """llama-server router ``POST /models/load`` -> ``{"success": true, ...}``.

    Two decisions behind one name, picked by what the body carries:

    * ``context_length`` alone (optionally with ``min_slots``, ``max_slots``,
      ``kv_unified``, ``kv_min``, ``prefer_mode``, ``allowed_devices``,
      ``priority``, ``persist``) is the strict walk of
      ``/api/models/{id}/load-recommended``: exactly that context per slot or a
      structured refusal.
    * ``devices``, ``parallel``, ``kv_cache_type(_v)`` or ``force`` is the explicit
      load of ``/api/models/{id}/load``; ``context_length`` is then its ``ctx_size``.
    * Neither: the model's own settings, like a just-in-time load.

    ``?dryRun=true`` runs the same decision without starting anything
    (:meth:`ModelManager.load_dry_run`, routed by the same
    :meth:`~ModelManager.load_route_for` as the real load) and returns the plan
    body with ``dry_run: true`` and ``dry_run_of`` (``load`` or
    ``load_recommended``) added. ``lease_id`` (body or ``?lease_id=``) binds the
    load to a lease the caller holds, as on ``/api/models/{id}/load`` (D87). The
    reply to a real load is the instance (as ``/load`` answers) with
    ``success: true`` in front.
    """
    state = _state(request)
    name = _require_named(body.model)
    record = _resolve_or_404(state, name)
    ctx = body.context_length if body.context_length is not None else body.ctx_size
    explicit = any(
        value is not None
        for value in (body.devices, body.parallel, body.kv_cache_type, body.kv_cache_type_v)
    ) or bool(body.force)
    walk_only = {
        "min_slots": body.min_slots,
        "max_slots": body.max_slots,
        "kv_unified": body.kv_unified,
        "kv_min": body.kv_min,
        "prefer_mode": body.prefer_mode,
        "persist": body.persist or None,
    }
    named_walk_only = sorted(k for k, v in walk_only.items() if v is not None)
    if explicit and named_walk_only:
        raise BadRequestError(
            f"{', '.join(named_walk_only)} belong to the strict context walk "
            "(context_length alone) and cannot be combined with devices, parallel, "
            "kv_cache_type, kv_cache_type_v or force.",
            code="invalid_request",
        )
    if named_walk_only and ctx is None:
        raise BadRequestError(
            f"{', '.join(named_walk_only)} need context_length (tokens per slot).",
            param="context_length",
            code="invalid_request",
        )
    # Which load this body is -- the one rule the real load and its dry run
    # share (D88): a body that places the model itself (or forces), or names no
    # context, is the explicit load; a window alone is the strict walk.
    route = state.manager.load_route_for(
        context_length=ctx,
        devices=body.devices,
        parallel=body.parallel,
        kv_cache_type=body.kv_cache_type,
        kv_cache_type_v=body.kv_cache_type_v,
        force=body.force,
    )
    if body.devices is not None and body.allowed_devices is not None:
        raise BadRequestError(
            "'devices' and 'allowed_devices' are mutually exclusive: devices says use exactly "
            "these cards, allowed_devices says choose among these.",
            param="allowed_devices",
            code="invalid_request",
        )

    from studioforge.api.mgmt_routes import _bound_lease_id, request_caller

    # lease_id (D87, B.6): body or ?lease_id=, ownership-checked as on
    # /api/models/{id}/load (D80); an unknown id is the manager's 404.
    lease_id = _bound_lease_id(state, request, body.lease_id, request.query_params.get("lease_id"))

    if dry_run:
        return await state.manager.load_dry_run(
            record.id,
            context_length=ctx,
            devices=body.devices,
            parallel=body.parallel,
            kv_cache_type=body.kv_cache_type,
            kv_cache_type_v=body.kv_cache_type_v,
            allowed_devices=body.allowed_devices,
            priority=body.priority,
            min_slots=body.min_slots,
            max_slots=body.max_slots,
            kv_unified=body.kv_unified,
            prefer_modes=[body.prefer_mode] if body.prefer_mode else None,
            kv_min=body.kv_min,
            force=body.force,
            # Only when sent: a dry run without a lease forwards exactly what it did.
            **({"lease_id": lease_id} if lease_id else {}),
        )

    if body.persist:
        # The path-based middleware cannot see a body field, and this one writes
        # settings that outlive the instance.
        require_admin_action(
            request,
            state.config,
            "persisting the resolved load profile to this model's saved settings",
        )
    source = attributed_source("api:/models/load", client_of(request))
    extra: dict[str, Any] = {"lease_id": lease_id} if lease_id else {}
    # D91: whose instance a base id means, and the copy a second holder gets.
    extra["caller"] = request_caller(state, request)
    if route == "load":
        instance = await state.manager.load(
            record.id,
            ctx_size=ctx,
            kv_cache_type=body.kv_cache_type,
            kv_cache_type_v=body.kv_cache_type_v,
            parallel=body.parallel,
            devices=body.devices,
            allowed_devices=body.allowed_devices,
            force=body.force,
            source=source,
            priority=body.priority,
            **extra,
        )
    else:
        assert ctx is not None  # load_route_for: a window-only body names one
        instance = await state.manager.load_recommended(
            record.id,
            int(ctx),
            prefer_modes=[body.prefer_mode] if body.prefer_mode else None,
            kv_min=body.kv_min,
            max_slots=body.max_slots,
            allowed_devices=body.allowed_devices,
            persist=body.persist,
            source=source,
            priority=body.priority,
            min_slots=body.min_slots,
            kv_unified=body.kv_unified,
            **extra,
        )
    return {"success": True, **instance.model_dump(mode="json")}


@router.post("/models/unload")
async def router_unload_model(request: Request, body: ModelsUnloadBody) -> dict[str, Any]:
    """llama-server router ``POST /models/unload`` -> ``{"success": true}``.

    Idempotent: a model that was not loaded answers ``success: true`` with
    ``unloaded: false``. A model held by a GPU lease is refused unless the caller
    may override it, exactly as ``POST /api/models/{id}/unload``.
    """
    from studioforge.api.mgmt_routes import unload_for_request

    state = _state(request)
    name = _require_named(body.model)
    record = _resolve_or_404(state, name)
    # D91 (8a): the caller's own instance for a base id; force only when sent.
    target, unloaded = await unload_for_request(
        state, request, record.id, force=bool(getattr(body, "force", False))
    )
    return {"success": True, "model": target, "unloaded": bool(unloaded)}


# ---------------------------------------------------------------------------
# KServe v2 probes
# ---------------------------------------------------------------------------


@router.get("/v2/health/live")
async def v2_live() -> dict[str, Any]:
    """KServe v2 server liveness: this process answers."""
    return {"live": True}


@router.get("/v2/health/ready")
async def v2_ready(request: Request) -> JSONResponse:
    """KServe v2 server readiness: boot finished and not draining (503 otherwise)."""
    state = _state(request)
    boot = getattr(state, "boot", None)
    if boot is not None and not boot.ready:
        return JSONResponse(
            {"ready": False, "reason": f"still starting ({boot.phase})"}, status_code=503
        )
    if getattr(state.manager, "draining", False):
        return JSONResponse({"ready": False, "reason": "draining"}, status_code=503)
    return JSONResponse({"ready": True})


@router.get("/v2/models/{model_id:path}/ready")
async def v2_model_ready(model_id: str, request: Request) -> JSONResponse:
    """KServe v2 model readiness: 200 when the model is resident and serving, else 503."""
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    serving = state.manager.serving_record(record)
    instance = state.supervisor.get(serving.id)
    if instance is not None and instance.state == "ready":
        return JSONResponse({"name": record.id, "ready": True})
    state_name = instance.state if instance is not None else "unloaded"
    return JSONResponse({"name": record.id, "ready": False, "state": state_name}, status_code=503)
