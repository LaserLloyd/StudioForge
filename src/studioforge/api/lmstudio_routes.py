"""LM Studio's native v1 REST API (``/api/v1``), served on the OpenAI port.

LM Studio 0.4 added a second, richer API next to its OpenAI-compatible one, and
OpenClaw preloads its models through ``POST /api/v1/models/load`` rather than
relying on just-in-time loading. A drop-in replacement therefore has to answer
this surface too, with LM Studio's own field names:

* ``GET  /api/v1/models`` -- every downloaded model with ``key``,
  ``loaded_instances`` and the rest of LM Studio's per-model fields.
* ``POST /api/v1/models/load`` / ``unload`` -- onto the **same manager
  methods** the management API (``POST /api/models/{id}/load``) and the MCP
  ``load_model`` tool use. No planner logic lives here: a load this route
  cannot fit is the planner's honest 507, with its numbers.
* ``POST /api/v1/models/download`` and ``GET /api/v1/models/download/status``
  -- onto the download queue behind ``/api/downloads``. A job is a StudioForge
  download *group* (all the files of one quant), and ``job_id`` is its id.
* ``POST /api/v1/chat`` -- **501 ``not_implemented``**. LM Studio's chat is a
  stateful, MCP-aware endpoint; this server is stateless. The body says what to
  use instead (``/v1/chat/completions``, ``/v1/messages``, ``/v1/responses``).

Field provenance: names and shapes follow LM Studio's published 0.4 REST
documentation (``tests/fixtures/lmstudio/README.md`` says which fixtures are
derived from it rather than captured from a running LM Studio). Where
StudioForge has a truthful value LM Studio does not name, the field is
additive: ``state``, ``loaded_context_length``, and the ``arch`` / ``data``
aliases that the older ``/api/v0`` and OpenAI shapes taught clients to read.

A StudioForge model has at most one instance, so an ``instance_id`` is the
serving model's id; a persona (preset-only virtual model) is served by its
base's instance and reports that id.

``ttl`` and ``priority`` are accepted on load exactly as on chat completions
(same parsing, same tier ceiling, never a pin) -- see
:func:`~studioforge.api.openai_routes._pop_ttl`.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from studioforge.api.mgmt_routes import _peer_host, may_unload_lease_held, start_download
from studioforge.api.openai_routes import (
    _app_state,
    _apply_ttl_override,
    _json_body,
    _pop_priority,
    _pop_ttl,
    _require_model,
    _resolve_or_404,
    _wait_for_scan,
)
from studioforge.core.attribution import attributed_source, client_of
from studioforge.core.catalog import model_type
from studioforge.errors import BadRequestError, ModelNotFoundError, StudioForgeError
from studioforge.types import InstanceInfo, ModelRecord

router = APIRouter()

#: LM Studio's enum is llm | embedding | decision; a reranker has no LM Studio
#: word, so it keeps its own (additive) one rather than posing as an LLM.
_TYPE_MAP = {"llm": "llm", "vlm": "llm", "embeddings": "embedding", "rerank": "rerank"}


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------


def _params_string(count: int | None) -> str | None:
    """``8_000_000_000`` -> ``"8B"``; ``270_000_000`` -> ``"270M"``."""
    if not count or count <= 0:
        return None
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if count >= size:
            value = count / size
            text = f"{value:.1f}".rstrip("0").rstrip(".") if value < 100 else f"{value:.0f}"
            return f"{text}{unit}"
    return str(count)


def _ready_instance(state: Any, record: ModelRecord) -> InstanceInfo | None:
    instance = state.supervisor.get(state.manager.serving_record(record).id)
    return instance if instance is not None and instance.state == "ready" else None


def _model_entry(state: Any, record: ModelRecord) -> dict[str, Any]:
    serving = state.manager.serving_record(record)
    instance = state.supervisor.get(serving.id)
    ready = instance is not None and instance.state == "ready"
    loaded: list[dict[str, Any]] = []
    entry_state = "not-loaded"
    if instance is not None and instance.state == "loading":
        entry_state = "loading"
    loaded_ctx: int | None = None
    if ready and instance is not None:
        entry_state = "loaded"
        plan = instance.plan
        loaded_ctx = plan.ctx_size if plan is not None else None
        config: dict[str, Any] = {"offload_kv_cache_to_gpu": True}
        if plan is not None:
            config["context_length"] = plan.ctx_size
            config["parallel"] = plan.parallel
        loaded.append({"id": serving.id, "config": config, "context_length": loaded_ctx})
    meta = record.meta
    return {
        "type": _TYPE_MAP.get(model_type(record), "llm"),
        "publisher": record.publisher or "local",
        "key": record.id,
        "display_name": record.name,
        "architecture": record.architecture,
        "arch": record.architecture,
        "quantization": {"name": record.quant, "bits_per_weight": None},
        "size_bytes": record.size_bytes,
        "params_string": _params_string(meta.param_count if meta else None),
        "loaded_instances": loaded,
        "max_context_length": meta.n_ctx_train if meta else None,
        "format": "gguf",
        "capabilities": {
            "vision": bool(record.capabilities.vision),
            "trained_for_tool_use": bool(record.capabilities.tools),
        },
        # Additive (not LM Studio names): the load state in the vocabulary of
        # /v1/models and /api/v0/models, and the loaded window.
        "state": entry_state,
        "loaded_context_length": loaded_ctx,
    }


@router.get("/api/v1/models")
async def v1_list_models(request: Request) -> JSONResponse:
    """Every downloaded model, loaded or not, in LM Studio's v1 shape."""
    state = _app_state(request)
    await _wait_for_scan(state)
    models = [_model_entry(state, record) for record in state.registry.all()]
    # `models` is LM Studio's key; `data`/`object` mirror the OpenAI list shape
    # for clients that read this endpoint the way they read /v1/models.
    return JSONResponse({"models": models, "object": "list", "data": models})


# ---------------------------------------------------------------------------
# POST /api/v1/models/load, /unload
# ---------------------------------------------------------------------------


def _optional_int(body: dict[str, Any], key: str) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BadRequestError(f"'{key}' must be a positive integer", param=key)
    return value


@router.post("/api/v1/models/load")
async def v1_load_model(request: Request) -> JSONResponse:
    """Load a model, LM Studio style: ``{model, context_length?, ttl?, ...}``.

    ``context_length`` is the per-conversation window (StudioForge's
    ``ctx_size``). LM Studio's engine tuning knobs -- ``eval_batch_size``,
    ``flash_attention``, ``num_experts`` -- have no per-request equivalent
    here (the planner owns them) and are reported under ``ignored`` instead of
    failing the preload. ``offload_kv_cache_to_gpu: false`` is refused: this
    server is GPU-only.
    """
    state = _app_state(request)
    body = await _json_body(request)
    model = _require_model(body, state.config)
    context_length = _optional_int(body, "context_length")
    parallel = _optional_int(body, "parallel")
    if body.get("offload_kv_cache_to_gpu") is False:
        raise BadRequestError(
            "offload_kv_cache_to_gpu=false asks for a CPU-resident KV cache; this server is "
            "GPU-only and never spills to the CPU.",
            param="offload_kv_cache_to_gpu",
            code="unsupported_parameter",
        )
    ttl = _pop_ttl(body)
    priority = _pop_priority(body)
    record = _resolve_or_404(state, model)
    ignored = [k for k in ("eval_batch_size", "flash_attention", "num_experts") if k in body]

    started = time.perf_counter()
    instance = await state.manager.load(
        record.id,
        ctx_size=context_length,
        parallel=parallel,
        source=attributed_source("api:/api/v1/models/load", client_of(request)),
        priority=priority,
    )
    elapsed = time.perf_counter() - started
    _apply_ttl_override(state, instance.model_id, ttl)

    plan = instance.plan
    actual = plan.ctx_size if plan is not None else None
    config: dict[str, Any] = {"offload_kv_cache_to_gpu": True}
    if plan is not None:
        config["context_length"] = plan.ctx_size
        config["parallel"] = plan.parallel
    result: dict[str, Any] = {
        "type": _TYPE_MAP.get(model_type(record), "llm"),
        "instance_id": instance.model_id,
        "model": record.id,
        "context_length": actual,
        "load_time_seconds": round(elapsed, 3),
        "status": "loaded",
    }
    if body.get("echo_load_config") is True:
        result["load_config"] = config
    if ignored:
        result["ignored"] = ignored
    if context_length is not None and actual is not None and actual != context_length:
        result["warnings"] = [
            f"'{record.id}' is already loaded with context_length {actual}; "
            f"unload it first to change the window to {context_length}."
        ]
    return JSONResponse(result)


@router.post("/api/v1/models/unload")
async def v1_unload_model(request: Request) -> JSONResponse:
    """Unload an instance by ``instance_id`` (or, additively, by ``model``)."""
    state = _app_state(request)
    body = await _json_body(request)
    name = body.get("instance_id") or body.get("model")
    if not isinstance(name, str) or not name.strip():
        raise BadRequestError("'instance_id' is required", param="instance_id")
    name = name.strip()

    instance_id = next((i.model_id for i in state.supervisor.list() if i.model_id == name), None)
    if instance_id is None:
        record = state.registry.resolve(name)
        if record is None:
            raise ModelNotFoundError(name, known=state.registry.known_ids())
        instance_id = state.manager.serving_record(record).id
        if state.supervisor.get(instance_id) is None:
            raise BadRequestError(
                f"'{name}' has no loaded instance.",
                param="instance_id",
                code="instance_not_found",
                status_code=404,
            )
    await state.manager.unload(
        instance_id,
        force=may_unload_lease_held(request, state, [instance_id]),
        source="rest",
        client=client_of(request),
        peer=_peer_host(request),
    )
    return JSONResponse({"instance_id": instance_id})


# ---------------------------------------------------------------------------
# POST /api/v1/models/download, GET /api/v1/models/download/status[/{job_id}]
# ---------------------------------------------------------------------------

#: LM Studio's job states; StudioForge's per-file ones fold onto them.
_JOB_STATES = {
    "queued": "downloading",
    "running": "downloading",
    "paused": "paused",
    "completed": "completed",
    "failed": "failed",
    "canceled": "failed",
}


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _jobs(state: Any) -> dict[str, dict[str, Any]]:
    jobs = getattr(state, "v1_download_jobs", None)
    if not isinstance(jobs, dict):
        jobs = {}
        state.v1_download_jobs = jobs
    return jobs


_HF_PREFIXES = ("https://huggingface.co/", "http://huggingface.co/", "hf.co/", "huggingface.co/")


def _parse_download_target(body: dict[str, Any]) -> tuple[str, str | None]:
    """``(hf repo id, quant)`` from LM Studio's ``model`` (+ ``quantization``)."""
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise BadRequestError("'model' is required", param="model")
    text = model.strip()
    quant = body.get("quantization")
    if quant is not None and (not isinstance(quant, str) or not quant.strip()):
        raise BadRequestError("'quantization' must be a string", param="quantization")
    for prefix in _HF_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    else:
        if "://" in text:
            raise BadRequestError(
                "only Hugging Face links and 'owner/repo' ids can be downloaded",
                param="model",
                code="unsupported_parameter",
            )
    text = text.split("?", 1)[0].split("#", 1)[0]
    if ":" in text:  # hf.co/owner/repo:Q4_K_M
        text, _, tag = text.partition(":")
        quant = quant or tag
    parts = [p for p in text.split("/") if p]
    if len(parts) < 2:
        raise BadRequestError(
            f"'{model}' is not a Hugging Face 'owner/repo' id or link. LM Studio's own model "
            "catalog is not available here.",
            param="model",
            code="unsupported_parameter",
        )
    return f"{parts[0]}/{parts[1]}", (quant.strip() if isinstance(quant, str) else None)


def _job_view(downloader: Any, group_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    items = downloader.group(group_id)
    status = _JOB_STATES.get(str(downloader.group_status(group_id)), "downloading")
    total = sum(int(p.total_bytes) for p in items)
    done = sum(int(p.downloaded_bytes) for p in items)
    view: dict[str, Any] = {
        "job_id": group_id,
        "status": status,
        "total_size_bytes": total,
        "downloaded_bytes": done,
        "started_at": _iso(meta.get("started_at")),
    }
    if status == "downloading":
        speed = sum(float(p.speed_bps) for p in items if p.status == "running")
        view["bytes_per_second"] = round(speed)
        if speed > 0 and total > done:
            view["estimated_completion"] = _iso(time.time() + (total - done) / speed)
    elif status == "completed":
        meta.setdefault("completed_at", time.time())
        view["completed_at"] = _iso(meta["completed_at"])
    else:
        errors = [p.error for p in items if p.error]
        if errors:
            view["error"] = errors[0]
    return view


def _downloader(state: Any) -> Any:
    from studioforge.api.mgmt_routes import _require_downloader

    return _require_downloader(state)


@router.post("/api/v1/models/download")
async def v1_download_model(request: Request) -> JSONResponse:
    """Queue a download. ``model`` is a Hugging Face ``owner/repo`` id or link.

    Behind the same admin gate as ``POST /api/downloads`` (it writes to the
    box's disk): on an install without ``server.api_key`` it needs a caller on
    this machine or the pairing PIN.
    """
    state = _app_state(request)
    body = await _json_body(request)
    repo_id, quant = _parse_download_target(body)
    downloader = _downloader(state)
    queued = await start_download(
        request, repo_id=repo_id, quant=quant, include_mmproj=True, force=False
    )
    group_id = queued["group_id"]
    if str(downloader.group_status(group_id)) == "completed":
        return JSONResponse({"status": "already_downloaded", "model": repo_id})
    jobs = _jobs(state)
    meta = jobs.setdefault(group_id, {"started_at": time.time(), "repo_id": repo_id})
    view = _job_view(downloader, group_id, meta)
    view["total_size_bytes"] = queued.get("total_bytes") or view["total_size_bytes"]
    view["model"] = repo_id
    view["quantization"] = queued.get("quant")
    return JSONResponse(view)


def _status_of(state: Any, job_id: str) -> dict[str, Any]:
    downloader = _downloader(state)
    if not downloader.group(job_id):
        raise BadRequestError(
            f"no download job '{job_id}'",
            param="job_id",
            code="download_not_found",
            status_code=404,
        )
    meta = _jobs(state).setdefault(job_id, {"started_at": None})
    return _job_view(downloader, job_id, meta)


@router.get("/api/v1/models/download/status/{job_id}")
async def v1_download_status(job_id: str, request: Request) -> JSONResponse:
    """One job's progress, in LM Studio's shape."""
    return JSONResponse(_status_of(_app_state(request), job_id))


@router.get("/api/v1/models/download/status")
async def v1_download_status_all(request: Request) -> JSONResponse:
    """Every job (additive: LM Studio's endpoint is per ``job_id``)."""
    state = _app_state(request)
    downloader = _downloader(state)
    group_ids: list[str] = []
    for progress in downloader.all():
        if progress.group_id not in group_ids:
            group_ids.append(progress.group_id)
    jobs = [_status_of(state, gid) for gid in group_ids]
    return JSONResponse({"jobs": jobs})


# ---------------------------------------------------------------------------
# POST /api/v1/chat
# ---------------------------------------------------------------------------


class NotImplementedApiError(StudioForgeError):
    """HTTP 501 in the shared error envelope."""

    status_code = 501
    error_type = "not_implemented_error"
    code = "not_implemented"


@router.post("/api/v1/chat")
async def v1_chat(request: Request) -> JSONResponse:
    """LM Studio's stateful chat -- not implemented, and says so."""
    raise NotImplementedApiError(
        "Stateful /api/v1/chat is not supported; use /v1/chat/completions "
        "(or /v1/messages, /v1/responses)",
        details={
            "alternatives": ["/v1/chat/completions", "/v1/messages", "/v1/responses"],
            "reason": "this server is stateless: send the whole conversation each turn",
        },
    )
