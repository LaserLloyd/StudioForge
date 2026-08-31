"""Management API (mounted under ``/api``).

Consumed by the GUI, ``sfctl`` and the management-plane MCP server, so all three
control surfaces share one implementation. Inference is deliberately absent
here -- it stays on the OpenAI endpoints.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Body, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from studioforge import __version__
from studioforge.api.auth import (
    PIN_WITHHELD_NOTE,
    may_reveal_pin,
    redact_config_dict,
    require_admin_action,
)
from studioforge.config import RESTART_REQUIRED_KEYS, apply_overrides
from studioforge.core import parallel_bench
from studioforge.core.benchmark import (
    DEFAULT_CTX_SIZE,
    DEFAULT_MAX_TOKENS,
    Benchmarker,
    BenchmarkJob,
    BenchmarkJobs,
    available_modes,
)
from studioforge.core.diskspace import disk_report
from studioforge.core.leases import lease_view
from studioforge.core.manager import validate_load_args
from studioforge.core.model_gate import (
    GateRequirement,
    gate_answer,
    parse_min_params,
    parse_tags,
)
from studioforge.core.parallel_bench import (
    DEFAULT_MAX_TOKENS as PARALLEL_MAX_TOKENS,
)
from studioforge.core.parallel_bench import (
    DEFAULT_PROMPT_TOKENS,
    DEFAULT_STREAMS,
    ParallelBenchmarker,
)
from studioforge.errors import (
    BadRequestError,
    ModelBusyError,
    ModelNotFoundError,
    StudioForgeError,
)
from studioforge.logging import RING_BUFFER, get_logger
from studioforge.types import AdapterAttachment, ModelSettings, VirtualPreset

log = get_logger(__name__)

router = APIRouter()

#: How many finished benchmark jobs stay pollable in memory. Completed
#: reports live in SQLite, so evicting an old job loses nothing durable.
BENCHMARK_JOB_HISTORY = 20


def _state(request: Request) -> Any:
    return request.app.state


# ---------------------------------------------------------------------------
# Status / health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {
        "status": "ok",
        "version": __version__,
        "uptime_s": round(time.time() - state.started_at, 1),
        "loaded_models": [i.model_id for i in state.supervisor.list()],
        "busy": state.manager.busy_snapshot(),
        "draining": state.manager.draining,
    }


@router.get("/status")
async def status(request: Request) -> dict[str, Any]:
    """Live server snapshot: GPUs, residents, queue, downloads, VRAM holders.

    Two keys on every ``loaded[]`` row are worth naming because nothing else
    reports them per instance: ``priority`` is the tier that load was actually
    made at (1 active chat, 2 dispatched agent, 3 background) -- the hard
    answer, not the soft prediction ``GET /api/models`` gives for an unloaded
    model -- and ``speculative`` is what drafting resolved to, where ``null``
    means the launch has not decided yet and ``{"type": "none", "reason": ...}``
    means it was considered and declined.

    ``busy.priority_hold`` alongside says whether a chat/agent load is
    currently holding worse-tier traffic off.
    """
    state = _state(request)
    engine = state.engine_manager.active()
    downloads = 0
    if state.downloader is not None:
        downloads = len(state.downloader.active())
    payload = state.manager.status(engine=engine, active_downloads=downloads)
    data = payload.model_dump(mode="json")
    # A running benchmark was invisible here -- REST clients had to learn of
    # it from a 503 on their next POST. Present while one runs, null after.
    running = _benchmark_jobs(state).running()
    data["benchmark"] = (
        None
        if running is None
        else {
            "job_id": running.job_id,
            "model_id": running.model_id,
            "mode": running.mode,
            "phase": running.phase,
            "fraction": running.fraction,
        }
    )
    # Who used inference in the trailing hour, busiest first (X-SF-Client
    # label or peer IP). The open :1234 trade makes this the whole defence
    # until a key is set: the next mystery client names itself here.
    data["clients"] = state.manager.clients_snapshot()
    _attach_child_metrics(state, data)
    # Names every VRAM holder, says which GPU each one's memory is actually on
    # (``device_bytes``/``per_gpu_bytes``), and collapses desktop noise. Off the
    # event loop: it walks the process table. See DECISIONS.md D23 and D39.
    await run_in_threadpool(_attach_vram_attribution, state, data)
    return data


def _attach_vram_attribution(state: Any, data: dict[str, Any]) -> None:
    """Annotate the VRAM holders in ``/api/status``; never fail the endpoint."""
    from studioforge.core.vram_holders import annotate_status_payload

    try:
        annotate_status_payload(
            data,
            engines_dir=state.config.engines_dir,
            own_pids=_own_child_pids(state),
        )
    except Exception as exc:  # noqa: BLE001 - status must answer even when this cannot
        log.debug("vram attribution failed", error=str(exc))


def _own_child_pids(state: Any) -> set[int]:
    """Pids of the llama-server children this process owns."""
    supervisor = getattr(state, "supervisor", None)
    if supervisor is None:
        return set()
    getter = getattr(supervisor, "child_pids", None)
    if getter is not None:
        try:
            return set(getter())
        except Exception:  # noqa: BLE001 - pragma: no cover - defensive
            return set()
    return {i.pid for i in supervisor.list() if i.pid is not None}


def _attach_child_metrics(state: Any, data: dict[str, Any]) -> None:
    """Add each loaded model's live llama-server gauges to ``/api/status``.

    ``requests_deferred`` is the one that matters operationally: llama.cpp
    queues a request that arrives when every slot is busy rather than refusing
    it (which is the right behaviour and is why there is no 429 here), so a
    server that looks healthy and feels slow shows it *only* in this counter.
    Surfacing it next to ``max_parallel`` lets a client see that it is asking
    for more concurrency than the load was planned for.

    Read from the collector's cache, never scraped inline -- the GUI polls this
    endpoint continuously.
    """
    from studioforge.core import throughput

    snapshot = state.manager.metrics_snapshot()
    for entry in data.get("loaded", []):
        gauges = snapshot.get(entry.get("model_id"))
        plan = entry.get("plan") or {}
        entry["requests_deferred"] = (
            gauges.get(throughput.METRIC_REQUESTS_DEFERRED) if gauges else None
        )
        entry["requests_processing"] = (
            gauges.get(throughput.METRIC_REQUESTS_PROCESSING) if gauges else None
        )
        entry["busy_slots_per_decode"] = (
            gauges.get(throughput.METRIC_BUSY_SLOTS) if gauges else None
        )
        entry["metrics_sampled_at"] = gauges.get("sampled_at") if gauges else None
        entry["max_parallel"] = plan.get("max_parallel")
        entry["ctx_per_slot"] = plan.get("ctx_per_slot") or plan.get("ctx_size")


@router.get("/gpus")
async def gpus(request: Request) -> dict[str, Any]:
    state = _state(request)
    from studioforge.core.gpu import system_ram

    total_ram, used_ram = system_ram()
    return {
        "backend": state.probe.backend,
        "driver_version": state.probe.driver_version(),
        "cuda_driver_version": state.probe.cuda_driver_version(),
        "gpus": [g.model_dump(mode="json") for g in state.probe.list_gpus()],
        "system_ram_total_bytes": total_ram,
        "system_ram_used_bytes": used_ram,
    }


@router.get("/vram/holders")
async def vram_holders(request: Request) -> dict[str, Any]:
    """Every process holding VRAM, with who launched it and whether it is safe to kill.

    Answers the question ``/api/gpus`` cannot: "10 GiB is gone and nothing is
    loaded -- who has it?". Each holder carries ``classification``:

    * ``ours`` -- a child of this server. Unload the model.
    * ``child-of-live-process`` -- someone else's running llama-server (a test
      run, a second instance). Reported, never killed automatically; the parent
      is named so you can go and stop it.
    * ``orphan`` -- our engine binary with a dead parent. Pure leak; reclaim it.
    * ``other-instance`` -- a ``llama-server`` from a different install, named
      by its own ``--alias``/``--port`` in ``detail``. Never killed from here.
    * ``foreign`` -- not one of our binaries at all (browser, compositor,
      ComfyUI).

    Desktop noise is summarised in ``desktop_processes_count`` instead of being
    listed. ``used_bytes_source`` says where each byte figure came from: NVML
    cannot measure per-process VRAM on Windows, so ``pdh`` numbers are
    per-process totals across adapters and must not be summed across GPUs.

    ``per_gpu_bytes`` splits that total per CUDA ordinal (D39) --
    ``{"0": 16660000000, "1": 15550000000}`` -- and ``gpu_indices`` lists only
    the devices holding at least 256 MiB. ``gpu_indices_source`` says which
    question was answered: ``pdh`` means "this is where the memory is",
    ``nvml-context`` means "these are the devices the process has a CUDA
    context on", which llama.cpp opens on every visible card.
    """
    state = _state(request)
    from studioforge.core.vram_holders import holders_view

    return await run_in_threadpool(
        holders_view,
        state.probe,
        state.config.engines_dir,
        own_pids=_own_child_pids(state),
    )


@router.post("/vram/reclaim")
async def vram_reclaim(request: Request, dry_run: bool = Body(False, embed=True)) -> dict[str, Any]:
    """Kill orphaned llama-server processes and free their VRAM.

    Only ``orphan`` holders are touched: our own engine binary, under our own
    engines directory, with a parent that no longer exists. Nothing else on the
    box launches binaries from there, so such a process cannot belong to anyone
    else and nothing is waiting on it.

    A ``child-of-live-process`` is deliberately never killed here, however much
    VRAM it holds -- it belongs to something that is still running, and taking
    its model away is not recovery. Use ``dry_run: true`` to see what would go.
    """
    state = _state(request)
    from studioforge.core.vram_holders import reclaim_orphans

    actions = await run_in_threadpool(
        reclaim_orphans,
        state.config.engines_dir,
        own_pids=_own_child_pids(state),
        dry_run=dry_run,
    )
    return {
        "dry_run": dry_run,
        "orphans_found": len(actions),
        "killed": sum(1 for action in actions if action.get("killed")),
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# GPU leases (D43)
# ---------------------------------------------------------------------------


@router.get("/leases")
async def list_leases(request: Request) -> dict[str, Any]:
    """Every standing GPU lease: which cards, held by whom, for which models."""
    state = _state(request)
    leases = [lease_view(lease) for lease in state.manager.leases.all()]
    return {"leases": leases, "count": len(leases)}


@router.post("/leases")
async def create_lease(
    request: Request,
    devices: list[int] = Body(...),
    model_ids: list[str] | None = Body(None),
    holder: str = Body("api"),
    reason: str = Body(""),
    idle_ttl_s: float | None = Body(3600.0),
    force: bool = Body(False),
) -> dict[str, Any]:
    """Give CUDA ``devices`` to ``model_ids`` -- or, with none, to something outside this server.

    While the lease stands nothing else is planned onto those cards; the named
    models are loaded onto exactly them, sized for as many slots as their
    context allows, in the split mode their own benchmark measured fastest
    there. Idle residents on the cards are unloaded; a model mid-request
    refuses the call (503, ``retry_after_s``); a pinned idle resident refuses
    unless ``force``. ``idle_ttl_s`` (default 3600) is how long the lease
    survives without activity before the server releases it; ``null`` holds
    it until ``DELETE``.
    """
    state = _state(request)
    lease = await state.manager.acquire_lease(
        devices,
        holder=holder,
        model_ids=model_ids or [],
        reason=reason,
        idle_ttl_s=idle_ttl_s,
        force=force,
    )
    return lease_view(lease)


@router.delete("/leases/{lease_id}")
async def release_lease(lease_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    return {"released": lease_view(state.manager.release_lease(lease_id))}


@router.post("/leases/{lease_id}/touch")
async def touch_lease(lease_id: str, request: Request) -> dict[str, Any]:
    """Restart the idle clock -- for a holder outside this server that is still busy."""
    state = _state(request)
    return lease_view(state.manager.touch_lease(lease_id))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@router.get("/models")
async def list_models(request: Request) -> dict[str, Any]:
    """The whole library, each row's saved record plus what it is doing now.

    Additive per-row keys on top of the stored ``ModelRecord``: ``loaded`` /
    ``state`` / ``port`` (is there a child, and where), ``ttl_remaining_s`` and
    ``effective_ttl_s`` (0 means pinned), ``active_requests`` and
    ``last_tokens_per_second``, and ``priority`` -- the tier this model would
    load at right now (D46/D48): the running instance if there is one, else the
    in-memory memo of what clients have been asking for, else the saved
    ``settings.priority``, else background.

    ``priority`` and ``effective_ttl_s`` share a caveat: both are **soft
    answers** about the *next* load, not promises. A request may still name its
    own tier, and a tier that outranks the standing one wins.
    """
    state = _state(request)
    from studioforge.api.app import wait_for_boot

    await wait_for_boot(state, timeout_s=60.0, scan_only=True)  # D33: not an empty list mid-boot
    loaded = {i.model_id: i for i in state.supervisor.list()}
    models = []
    for record in state.registry.all():
        # Preset-only virtual models serve from their base's instance; report
        # them loaded whenever the base is.
        serving = state.manager.serving_record(record)
        instance = loaded.get(serving.id)
        models.append(
            {
                **record.model_dump(mode="json"),
                "loaded": instance is not None,
                "state": instance.state if instance else "stopped",
                "port": instance.port if instance else None,
                "ttl_remaining_s": instance.ttl_remaining_s if instance else None,
                "effective_ttl_s": state.manager.ttl_for(record),
                # The soft "what tier would the next load use" answer -- see
                # the docstring. The serving record this loop already resolved
                # is passed in, so the accessor does not repeat that registry
                # lookup on the way to the same answer.
                "priority": state.manager.effective_priority_for(serving),
                "active_requests": instance.active_requests if instance else 0,
                "last_tokens_per_second": (instance.last_tokens_per_second if instance else None),
            }
        )
    return {"models": models, "count": len(models)}


@router.get("/catalog")
async def catalog(
    request: Request,
    refresh: bool = Query(False, description="Rebuild instead of serving the short cache"),
    model: str | None = Query(None, description="Restrict to one model id or alias"),
    compact: bool = Query(False, description="Keep only each model's recommended row"),
) -> dict[str, Any]:
    """Every model with its loading options, newest download first.

    One call answers "what can I run, at what context, how fast, and how many
    at once" for the whole library. Each option row carries a ``load_args``
    object that ``POST /api/models/{id}/load`` (and the MCP ``load_model``
    tool) accept verbatim.

    Built off the event loop: a full catalog walks the planner once per model
    per context tier, which is pure arithmetic but enough of it to stall a
    stream if it ran inline.
    """
    state = _state(request)
    return await run_in_threadpool(
        state.manager.catalog, model=model, compact=compact, refresh=refresh
    )


# Deliberately "/model-gate" and NOT "/models/gate": every other per-model route
# is "/models/{model_id:path}/...", and a :path converter is greedy enough that a
# literal sibling under the same prefix is a standing invitation to shadowing
# bugs the day someone adds a model whose id starts with "gate/". A hyphenated
# sibling of /models cannot collide with anything.
@router.get("/model-gate")
async def model_gate(
    request: Request,
    min_params: str | None = Query(
        None, description="Size bar in billions: 20, '20b', '500m', '0.5b'"
    ),
    vision: bool = Query(False, description="Require image input"),
    audio: bool = Query(False, description="Require audio input"),
    tools: bool = Query(False, description="Require tool calling"),
    thinking: bool = Query(False, description="Require a reasoning/thinking mode"),
    uncensored: bool = Query(False, description="Require an uncensored/abliterated model"),
    tags: str | None = Query(
        None, description="Comma-separated extra tags, e.g. 'coding,roleplay,vietnamese'"
    ),
) -> dict[str, Any]:
    """Is a loaded model above this bar, and which one should I send work to? (D52)

    The routing question an agent asks *before* it decides between three
    options: send this task to the local ``/v1`` API, spend 40 seconds loading
    something bigger, or pay a cloud provider. The operator usually has
    something loaded and most loaded models can do most jobs, so "is what is
    already resident good enough" is the cheap question that avoids the other
    two -- but only if it can be asked in one call.

    The answer therefore carries the **model id**, not just a yes: a caller told
    "yes, something qualifies" would still have to fetch ``/v1/models`` and
    guess which of three resident models was meant. ``model`` goes straight into
    the next ``/v1/chat/completions`` request. ``instances`` shows the working
    for every resident model, and on a "no" ``reason`` names the gap and
    ``hint`` points at the fix.

    ``vision``/``audio``/``tools``/``thinking``/``uncensored`` are sugar for the
    ``tags`` list -- every one of them lands in the same requirement set -- and
    ``tags`` itself is free-form, so a caller can ask for something nobody
    anticipated and get a truthful "yes" or "unknown" without a server change.

    Read-only, and therefore **not** behind the D32 admin gate: that gate is
    ``auth.is_admin_mutation``, which returns False for any verb outside
    ``{POST, PUT, PATCH, DELETE}``, so a GET is open by construction and this
    route needs no entry in any admin list. That is correct here -- routing
    decisions are exactly what LAN clients are supposed to make, and this
    endpoint reveals no more than ``GET /api/models`` already does.
    """
    state = _state(request)
    try:
        min_params_b = parse_min_params(min_params)
    except ValueError as exc:
        raise BadRequestError(str(exc), param="min_params") from exc

    sugar = {
        "vision": vision,
        "audio": audio,
        "tools": tools,
        "thinking": thinking,
        "uncensored": uncensored,
    }
    try:
        requested = parse_tags(tags, extra=[name for name, on in sugar.items() if on])
    except ValueError as exc:
        raise BadRequestError(str(exc), param="tags") from exc

    return await gate_answer(
        GateRequirement(min_params_b=min_params_b, tags=requested),
        registry=state.registry,
        supervisor=state.supervisor,
        introspect=state.manager.introspect,
    )


@router.post("/models/scan")
async def scan_models(request: Request, force: bool = Query(False)) -> dict[str, Any]:
    state = _state(request)
    # Off the event loop: scanning parses GGUF headers and stats every file in
    # the library, which would otherwise stall in-flight requests and streams.
    result = await asyncio.to_thread(state.registry.scan, force=force)
    return {
        "added": result.added,
        "removed": result.removed,
        "unchanged": result.unchanged,
        # Models kept from the previous scan because this one could not parse
        # them but the file is still there. A transient read error must not make
        # a model disappear from a client's list.
        "stale": list(getattr(result, "stale", []) or []),
        "errors": [{"path": p, "error": e} for p, e in result.errors],
        "duration_s": round(result.duration_s, 3),
    }


@router.get("/models/{model_id:path}/introspect")
async def introspect(model_id: str, request: Request) -> dict[str, Any]:
    """Real running settings from llama-server's own /props and /slots."""
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    return await state.manager.introspect(record.id)


@router.get("/models/{model_id:path}/plan")
async def plan(
    model_id: str,
    request: Request,
    ctx_size: int | None = Query(None),
    kv_cache_type: str | None = Query(None),
    parallel: int | None = Query(None),
) -> dict[str, Any]:
    """Live fit verdict, so the GUI can show it before the user clicks Load."""
    state = _state(request)
    return state.manager.plan_preview(
        model_id, ctx_size=ctx_size, kv_cache_type=kv_cache_type, parallel=parallel
    )


@router.get("/models/{model_id:path}/options")
async def model_options(
    model_id: str, request: Request, refresh: bool = Query(False)
) -> dict[str, Any]:
    """Every loading option for one model -- the MCP ``model_options`` table, over REST.

    Read-only capacity math (context/KV matrices, per-placement fit, the
    ``load_args`` recipes), previously reachable only through the MCP plane --
    so a planning agent without MCP credentials had to *estimate* the very
    numbers this server computes exactly. GET, ungated, like every other
    read surface.
    """
    state = _state(request)
    catalog = await run_in_threadpool(
        state.manager.catalog, model=model_id, compact=False, refresh=refresh
    )
    entries = catalog["models"]
    if not entries:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    return {"catalog_hint": catalog["catalog_hint"], "model": entries[0]}


@router.get("/models/{model_id:path}/profiles")
async def placement_profiles(model_id: str, request: Request) -> dict[str, Any]:
    """Best achievable load per hardware mode, so an agent can pick one.

    Saves OpenClaw from probing context sizes by trial and error: one call says
    what this model can do on the 5090 pair, on the 3090 pair, and on the whole
    rig, with the KV cache type each choice implies.
    """
    state = _state(request)
    return await run_in_threadpool(state.manager.placement_profiles, model_id)


@router.post("/models/{model_id:path}/load")
async def load_model(
    model_id: str,
    request: Request,
    ctx_size: int | None = Body(None),
    kv_cache_type: str | None = Body(None),
    kv_cache_type_v: str | None = Body(None),
    parallel: int | None = Body(None),
    devices: list[int] | None = Body(None),
    force: bool = Body(False),
    priority: int | None = Body(None),
) -> dict[str, Any]:
    """Load one model, optionally onto named GPUs.

    ``devices`` is a one-shot placement for this load only (D36) -- the
    catalog's per-hardware-mode ``load_args`` carry one -- and never touches the
    model's saved settings. A CUDA index this box does not have is a 400 naming
    the parameter, not a planner refusal that reads like a VRAM problem.

    ``priority`` is the load's tier (D46): 1 the active chat model, 2 a
    dispatched agent, 3 (or omitted) background. A tier-1/2 load takes the
    fastest placement -- displacing idle worse-tier models from the cards it
    picks and reloading the recently active ones afterwards where they fit --
    jumps the load queue, and holds new worse-tier traffic off (503 +
    Retry-After) while it loads. The tier is remembered per model, so a JIT
    reload keeps it -- and since D48 a saved ``settings.priority`` is the floor
    that survives a restart.

    The mirror image: while a better-tier load holds the server, this route
    answers 503 with the code ``priority_hold`` (not the generic busy code),
    ``Retry-After`` and ``details.priority_hold`` naming the model being loaded
    and its tier. It is a wait, not a failure -- honour the header.
    """
    state = _state(request)
    instance = await state.manager.load(
        model_id,
        ctx_size=ctx_size,
        kv_cache_type=kv_cache_type,
        kv_cache_type_v=kv_cache_type_v,
        parallel=parallel,
        devices=devices,
        force=force,
        source="api:/api/models/{id}/load",
        priority=priority,
    )
    return instance.model_dump(mode="json")


@router.post("/models/{model_id:path}/load-recommended")
async def load_recommended(
    model_id: str,
    request: Request,
    ctx_size: int = Body(..., embed=True),
    prefer_mode: str | None = Body(None),
    kv_min: str | None = Body(
        None,
        description=(
            "Lowest KV cache quality to accept: 'f16', 'q8_0' or 'q4_0'. The window is "
            "refused rather than reached with a worse cache than this."
        ),
    ),
    priority: int | None = Body(
        None,
        description="Load tier: 1 active chat, 2 dispatched agent, 3 (or omitted) background.",
    ),
    max_slots: int | None = Body(
        None,
        description=(
            "Ceiling on the slot count this load may choose (>= 1). Omitted, the "
            "estimator's own recommendation stands."
        ),
    ),
    persist: bool = Body(
        False,
        description=(
            "Write the resolved context, KV cache types, slot count and tier into the "
            "model's saved settings. Admin: needs server.api_key, a local caller or the "
            "MCP PIN."
        ),
    ),
) -> dict[str, Any]:
    """Load at exactly ``ctx_size`` per slot; the server picks everything else.

    Name the model and the context you need. This walks the hardware modes in
    headline order, asks for that exact context per slot under the quality-first
    KV rule with the recommended slot count, and loads the first placement that
    fits -- evicting only idle models, never a busy one.

    **Strict about context, unlike every other load path.** A window that does
    not fit is a structured 507 naming, per mode, the largest context that would
    work and what is in the way, with ``retry_after_s`` when the cause is a
    model that is serving right now. Above the model's trained window it is a
    400 with the number that would be accepted.

    ``kv_min`` is the lowest KV cache quality this call will accept -- ``f16``,
    ``q8_0`` or ``q4_0``. It floors the quality-first ladder rather than
    steering it: a window that needs a worse cache than the floor is refused
    with the same structured 507, never quietly quantized to reach the number.
    Omitted, the whole ladder is available (and a q4_0 K cache is still never
    chosen automatically).

    ``max_slots`` caps the slot count for this load only. The estimator's
    answer is what a placement *could* sustain; a caller that knows it has
    three bots wanting one window each does not want the other five slots' KV
    cache priced into the fit. Must be >= 1 (400 naming the parameter
    otherwise). The cap applies before the descent loop, so the winning plan is
    really planned at the capped count rather than planned larger and launched
    smaller.

    ``persist`` writes the *resolved* profile -- context per slot, both KV
    cache types, the slot count and the tier -- into the model's saved settings
    once the load succeeds, so the next plain load (a JIT reload included)
    reproduces it. The trade: it freezes the KV ladder and the slot estimator
    for this model, which then stop responding to a changed machine; null those
    fields with ``PATCH /api/models/{id}/settings`` to hand the decision back.
    Devices are deliberately **not** persisted -- a placement is a one-shot
    load argument (D36). It is refused for a preset-only virtual model (the
    write would land on the base model and every persona sharing it -- call it
    on the base id instead) and while the model is being benchmarked; both are
    checked before the walk begins, so the 400 arrives with nothing loaded.
    Afterwards the write is best-effort: anything that refuses it once the load
    has succeeded -- a benchmark that started *during* the load, or the settings
    validation rejecting the resolved row (a resident whose slot count predates
    a ``max_parallel_cap`` lowered under it) -- leaves the load standing and
    skips only the write, with a warning in the server log.

    ``priority`` is the load's tier (D46): 1 the active chat model, 2 a
    dispatched agent, 3 (or omitted) background.

    **Auth.** The route itself is ungated (residency is open, LM Studio
    parity), but ``persist: true`` writes settings -- the same box change
    ``PUT /settings`` is gated for (D32) -- so that field alone needs
    ``server.api_key``, a caller on this machine, or the MCP pairing PIN. The
    refusal is a 403 ``remote_admin_requires_credential`` and the same request
    without ``persist`` still loads.

    A load held off by a higher-tier load in flight is a 503 with code
    ``priority_hold``, ``Retry-After`` and ``details.priority_hold`` naming the
    holder -- distinct from the other 503s this route can return, so honour the
    header rather than retrying immediately.
    """
    state = _state(request)
    if persist:
        # The path-based D32 middleware cannot see a body field, and this one
        # writes settings that outlive the instance.
        require_admin_action(
            request,
            state.config,
            "persisting the resolved load profile to this model's saved settings",
        )
    instance = await state.manager.load_recommended(
        model_id,
        int(ctx_size),
        prefer_modes=[prefer_mode] if prefer_mode else None,
        kv_min=kv_min,
        max_slots=max_slots,
        persist=persist,
        source="api:/api/models/{id}/load-recommended",
        priority=priority,
    )
    return instance.model_dump(mode="json")


@router.post("/models/{model_id:path}/unload")
async def unload_model(model_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    unloaded = await state.manager.unload(model_id)
    return {"model_id": model_id, "unloaded": unloaded}


@router.post("/models/{model_id:path}/test")
async def test_model(
    model_id: str,
    request: Request,
    prompt: str | None = Body(None),
    keep_loaded: bool = Body(False),
) -> dict[str, Any]:
    """One canned request through the model, on an otherwise idle server (D36).

    A cold model is loaded small (one slot, the default context) and unloaded
    again unless ``keep_loaded``. Refused with a 503 and ``retry_after_s`` while
    anything is serving, loading or benchmarking, and while another test runs.
    """
    state = _state(request)
    return await state.manager.test_model(model_id, prompt, keep_loaded=keep_loaded)


@router.get("/models/{model_id:path}/settings")
async def get_settings(model_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    return record.settings.model_dump(mode="json")


@router.put("/models/{model_id:path}/settings")
async def put_settings(
    model_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """Save per-model settings, validating expert flags against the engine.

    Validation happens here rather than at load time so an unknown flag is a
    save-time error the user sees immediately, not a mystery crash later.

    **This is a full-object REPLACE**: a field the payload omits is reset to
    its default, silently. That has already cost a deliberate
    ``reasoning_format`` once -- a client that means "change one field" wants
    ``PATCH`` below, and should only PUT a payload it first GETs and merges.

    ``priority`` saved here is the **standing default** a restart falls back to
    (D48); a tier this server session already remembers from an explicit load
    still outranks it until the next restart or the next explicitly tiered load.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    try:
        settings = ModelSettings.model_validate(payload)
    except Exception as exc:
        raise BadRequestError(f"invalid settings: {exc}") from exc
    return await _validated_settings_save(state, record, settings)


@router.patch("/models/{model_id:path}/settings")
async def patch_settings(
    model_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """Merge-patch per-model settings (RFC 7386): change ONLY the named fields.

    A present key is set, an explicit ``null`` clears that field, and an
    absent key is left byte-identical -- so ``{"ctx_size": 131072}`` touches
    nothing else. This exists because the full-object PUT punishes the natural
    call shape: sending only the field you mean to change silently nulls every
    other field, and the loss surfaces later as a different symptom (thinking
    leaking into ``content``, a model sprawling across cards). A key that is
    not a settings field is a 400 naming it, not an ignored no-op -- a typoed
    field that "worked" is the same failure class this verb exists to close.

    ``priority`` saved here is the **standing default** a restart falls back to
    (D48); a tier this server session already remembers from an explicit load
    still outranks it until the next restart or the next explicitly tiered load.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    current = record.settings.model_dump(mode="json")
    unknown = sorted(set(payload) - set(current))
    if unknown:
        raise BadRequestError(
            f"unknown settings field(s): {', '.join(unknown)}",
            param=unknown[0],
        )
    try:
        settings = ModelSettings.model_validate({**current, **payload})
    except Exception as exc:
        raise BadRequestError(f"invalid settings: {exc}") from exc
    return await _validated_settings_save(state, record, settings)


async def _validated_settings_save(state: Any, record: Any, settings: Any) -> dict[str, Any]:
    """The shared tail of PUT and PATCH: guards, expert-flag checks, save."""
    # A benchmark rewrites this record's settings per mode and restores them at
    # the end, so a save landing mid-run reaches SQLite and is then silently
    # reverted in memory -- the change appears to have worked and has not.
    benchmarker = getattr(state, "benchmarker", None)
    if benchmarker is not None and getattr(benchmarker, "benchmarking", None) == record.id:
        raise BadRequestError(
            f"{record.id} is being benchmarked right now; the run rewrites its "
            "settings and would overwrite this save. Wait for it to finish, or "
            "cancel it, then save again.",
            code="model_benchmarking",
            param="model",
        )

    if settings.extra_flags.strip():
        tag = settings.engine_tag or state.config.engine.pinned_tag
        errors = await state.engine_manager.validate_extra_flags(tag, settings.extra_flags)
        if errors:
            raise BadRequestError("extra_flags rejected: " + "; ".join(errors), param="extra_flags")

    if settings.draft_model_id:
        problems = _check_draft_compatibility(state, record, settings.draft_model_id)
        if problems:
            raise BadRequestError("; ".join(problems), param="draft_model_id")

    updated = state.registry.save_settings(record.id, settings)
    # A pin or ttl change must bite now, not at the next load.
    state.manager.refresh_ttl(updated.id)
    return updated.settings.model_dump(mode="json")


def _check_draft_compatibility(state: Any, record: Any, draft_id: str) -> list[str]:
    """Block obviously incompatible draft pairings; warn on uncertain ones.

    Speculative decoding requires the draft and target to share a tokenizer:
    a mismatched vocab produces garbage, and llama.cpp will not always refuse.
    """
    draft = state.registry.resolve(draft_id)
    if draft is None:
        return [f"draft model '{draft_id}' is not in the registry"]
    if draft.id == record.id:
        return ["a model cannot be its own draft model"]
    problems: list[str] = []
    target_meta, draft_meta = record.meta, draft.meta
    if target_meta is None or draft_meta is None:
        return problems
    if target_meta.n_vocab and draft_meta.n_vocab and target_meta.n_vocab != draft_meta.n_vocab:
        problems.append(
            f"vocab size mismatch: target '{record.id}' has {target_meta.n_vocab} tokens, "
            f"draft '{draft.id}' has {draft_meta.n_vocab}. Speculative decoding requires a "
            f"shared vocabulary."
        )
    elif (
        target_meta.tokenizer_model
        and draft_meta.tokenizer_model
        and target_meta.tokenizer_model != draft_meta.tokenizer_model
    ):
        log.warning(
            "draft tokenizer differs from target; pairing may be unreliable",
            model_id=record.id,
            draft_model_id=draft.id,
            target=target_meta.tokenizer_model,
            draft=draft_meta.tokenizer_model,
        )
    return problems


@router.post("/models/{model_id:path}/pin")
async def pin_model(
    model_id: str, request: Request, pinned: bool = Body(True, embed=True)
) -> dict[str, Any]:
    state = _state(request)
    updated, effective_ttl = state.manager.set_pinned(model_id, pinned)
    return {
        "model_id": updated.id,
        "pinned": updated.settings.pinned,
        "effective_ttl_s": effective_ttl,
    }


def _instance_holding(state: Any, model_id: str) -> str | None:
    """Id of any resident instance whose weights are *model_id*'s, else None.

    Walks every loaded instance and resolves it back to the record actually
    served, so a virtual model riding a base (or a base whose persona is
    loaded) is reported rather than missed.
    """
    if state.supervisor.get(model_id) is not None:
        return str(model_id)
    # ``list()`` -- the supervisor has no ``all()``; the old call raised
    # AttributeError, so DELETE on any model that was not itself loaded came
    # back as a 500 (and the guard never ran for the persona-on-base case it
    # was written for).
    for instance in state.supervisor.list():
        candidate = state.registry.resolve(instance.model_id)
        if candidate is None:
            continue
        try:
            serving = state.manager.serving_record(candidate)
        except Exception:  # noqa: BLE001 - display/guard only
            serving = candidate
        if str(serving.id) == str(model_id) or str(candidate.id) == str(model_id):
            return str(instance.model_id)
    return None


@router.delete("/models/{model_id:path}")
async def delete_model(
    model_id: str,
    request: Request,
    delete_files: bool = Query(False),
    confirm: bool = Query(False),
) -> dict[str, Any]:
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    if delete_files and not confirm:
        raise BadRequestError(
            "deleting files requires confirm=true", param="confirm", code="confirmation_required"
        )
    # A virtual model with launch-time overrides gets its OWN instance that
    # opens the BASE model's GGUF. Checking only this id let a base be deleted
    # -- files and all -- while a child still had them open under a different
    # id. Refuse if anything resident resolves to these files.
    holder = _instance_holding(state, record.id)
    if holder is not None:
        raise BadRequestError(
            f"model '{record.id}' is loaded"
            + (f" (serving as '{holder}')" if holder != record.id else "")
            + "; unload it before deleting",
            code="model_loaded",
        )
    removed = state.registry.delete_model(record.id, delete_files=delete_files)
    return {"model_id": record.id, "removed": [str(p) for p in removed]}


# ---------------------------------------------------------------------------
# Adapters and virtual models
# ---------------------------------------------------------------------------


@router.get("/adapters")
async def list_adapters(request: Request) -> dict[str, Any]:
    state = _state(request)
    return {"adapters": [a.model_dump(mode="json") for a in state.registry.adapters()]}


@router.post("/adapters/scan")
async def scan_adapters(request: Request) -> dict[str, Any]:
    state = _state(request)
    # Off the event loop for the same reason as /models/scan: this re-walks and
    # re-parses the whole library, which would stall in-flight streams.
    found = await asyncio.to_thread(state.registry.scan_adapters)
    return {"adapters": [a.model_dump(mode="json") for a in found], "count": len(found)}


@router.delete("/adapters/{adapter_id:path}")
async def delete_adapter(
    adapter_id: str,
    request: Request,
    delete_file: bool = Query(False),
    confirm: bool = Query(False),
) -> dict[str, Any]:
    state = _state(request)
    if delete_file and not confirm:
        raise BadRequestError("deleting files requires confirm=true", param="confirm")
    state.registry.delete_adapter(adapter_id, delete_file=delete_file)
    return {"adapter_id": adapter_id, "deleted": True}


#: Request-time preset fields accepted at the top level of the create payload.
_PRESET_FIELDS = (
    "system_prompt",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "max_tokens",
)


@router.post("/virtual-models")
async def create_virtual_model(
    request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """A named base + adapters + preset combination, selectable through the
    OpenAI API -- the Modelfile/persona experience.

    The OpenAI protocol has no LoRA, system-prompt or preset parameter, so a
    virtual model id is how a client picks all of them: by model name, with no
    client changes. ``system_prompt`` and the sampler defaults are applied per
    request by the gateway, so any number of such presets over one base share
    a single llama-server instance. ``ctx_size``/``kv_cache_type`` are
    launch-time overrides: they are stored as the virtual model's settings
    (the planner honours those) and cost the model its own instance.
    """
    state = _state(request)
    model_id = payload.get("id")
    base = payload.get("base_model_id")
    if not model_id or not base:
        raise BadRequestError("'id' and 'base_model_id' are required")
    attachments = []
    for item in payload.get("adapters") or []:
        try:
            attachments.append(AdapterAttachment.model_validate(item))
        except Exception as exc:
            raise BadRequestError(f"invalid adapter attachment: {exc}") from exc

    try:
        preset = VirtualPreset.model_validate({k: payload.get(k) for k in _PRESET_FIELDS})
    except Exception as exc:
        raise BadRequestError(f"invalid preset: {exc}", param="preset") from exc

    record = state.registry.create_virtual_model(
        id=str(model_id),
        base_model_id=str(base),
        name=payload.get("name"),
        adapters=attachments,
        preset=None if preset.is_empty() else preset,
    )

    overrides = {k: payload[k] for k in ("ctx_size", "kv_cache_type") if payload.get(k) is not None}
    if overrides:
        settings = record.settings.model_copy(update=overrides)
        record = state.registry.save_settings(record.id, settings)
    return record.model_dump(mode="json")


@router.delete("/virtual-models/{model_id:path}")
async def delete_virtual_model(model_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    state.registry.delete_virtual_model(model_id)
    return {"model_id": model_id, "deleted": True}


@router.post("/models/{model_id:path}/adapter-scales")
async def set_adapter_scales(
    model_id: str, request: Request, payload: dict[str, Any] = Body(...)
) -> dict[str, Any]:
    """Change LoRA scales on a running instance without a reload.

    llama-server exposes runtime adapter scaling, so a scale change on an
    already-loaded model is a POST rather than a multi-second reload.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    scales = payload.get("scales")
    if not isinstance(scales, list):
        raise BadRequestError("'scales' must be a list of {id, scale} objects", param="scales")
    ok = await state.supervisor.set_lora_scales(record.id, scales)
    return {"model_id": record.id, "applied": ok, "reload_required": not ok}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


#: What an activation does *not* do, said in the response because this is the
#: single most-reported surprise about switching engines: a running
#: llama-server child holds its own binary open and keeps it until it is
#: replaced. Names the route that does the replacing so the answer to "and now
#: what?" is in the same payload.
ENGINE_ACTIVATE_NOTE = (
    "Loaded models are not disturbed: each llama-server child keeps the build it was "
    "launched with until it is reloaded. Call POST /api/restart/backend (Dashboard -> "
    "Restart engines) to move the resident models onto the newly activated build."
)


def _extra_flags_text(value: Any) -> str:
    """One flag string from either accepted shape (D49-11).

    Callers disagree about how a flag list travels: the GUI holds one editable
    string, while ``sfctl`` and most scripts hold an argv-shaped list. Refusing
    the list form produced a raw pydantic dump that never named the shape it
    wanted, so both are accepted here and a list is joined with spaces -- which
    is exactly how the saved ``settings.extra_flags`` string is split again.

    Raises:
        BadRequestError: 400 naming both accepted shapes.
    """
    shapes = (
        "'extra_flags' must be a string of flags (\"--foo 1 --bar\") or a list of "
        'strings (["--foo", "1", "--bar"])'
    )
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Named per element, because "got list" to someone who did send a list
        # is the same unhelpful answer the raw pydantic dump gave.
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise BadRequestError(
                    f"{shapes}; item {index} is {type(item).__name__}, not a string",
                    param="extra_flags",
                )
        return " ".join(value)
    raise BadRequestError(f"{shapes}; got {type(value).__name__}", param="extra_flags")


async def _flag_revalidation_sweep(state: Any, tag: str) -> list[dict[str, Any]]:
    """Every saved ``extra_flags`` re-checked against ``tag`` (D49-6).

    Saved flags are validated once, at save time, against whatever engine was
    current then -- and llama.cpp renames and removes flags between builds while
    llama-server ignores what it does not recognise. So a switch can silently
    drop a setting the operator believes is in force (the D2 failure). The sweep
    never blocks the switch: a stale flag is a warning, which is precisely why
    it has to be a loud one.
    """
    return list(await state.engine_manager.revalidate_extra_flags(tag, state.registry.all()))


@router.get("/engine")
async def engine_status(request: Request) -> dict[str, Any]:
    """The engine inventory: what is pinned, what is live, what is installed.

    ``drift`` is non-null when ``engine.pinned_tag`` and the actually-active
    build disagree -- ``active.json`` wins at load time, so a pin edited on its
    own (via ``PATCH /api/config``) changes nothing until something activates
    it, and before D49 that disagreement was a boot-time log line and nothing
    else. ``install_progress`` is the latest snapshot from an in-flight
    ``POST /api/engine/install`` (tag, phase, fraction, done/error) or ``null``
    when no install has run in this process; a ~600 MB download is otherwise a
    blind wait for the caller.
    """
    state = _state(request)
    active = state.engine_manager.active()
    pinned = state.config.engine.pinned_tag
    active_tag = active.tag if active else None
    drift = {"pinned": pinned, "active": active_tag} if active_tag != pinned else None
    return {
        "pinned_tag": pinned,
        "active": active.model_dump(mode="json") if active else None,
        "installed": [e.model_dump(mode="json") for e in state.engine_manager.installed()],
        "drift": drift,
        # getattr, not attribute access: an engine manager that has never run an
        # install in this process reports nothing rather than making the status
        # card -- polled by the Dashboard on a timer -- fail.
        "install_progress": getattr(state.engine_manager, "install_progress", None),
    }


@router.get("/engine/releases")
async def engine_releases(request: Request, limit: int = Query(20)) -> dict[str, Any]:
    """Installable llama.cpp builds, newest first.

    Upstream publishes build releases with GitHub's ``prerelease`` flag set, so
    the answer is filtered on asset eligibility rather than on that flag (D49-1);
    an empty list means no build carried an asset for this OS/arch/CUDA variant,
    not that no build exists.
    """
    state = _state(request)
    return {"releases": await state.engine_manager.list_releases(limit)}


@router.post("/engine/install")
async def engine_install(
    request: Request,
    tag: str = Body(...),
    force: bool = Body(False),
    activate: bool = Body(False),
) -> dict[str, Any]:
    """Download and unpack a llama.cpp build. Does not switch to it.

    **Behaviour change (D49-4).** Installing used to activate the new build as
    its last step, unconditionally -- which is how a failed smoke test could
    leave ``active.json`` pointing at the build that had just failed while the
    CLI printed "keeping <current>". Install and activate are now separate
    decisions: this route unpacks the binary and leaves the live engine exactly
    where it was. Pass ``activate: true`` for the old behaviour, or -- better --
    call ``POST /api/engine/activate`` afterwards, which additionally pins the
    tag in ``engine.pinned_tag`` (so it survives a restart) and revalidates
    saved expert flags against the new build.

    ``force`` re-downloads a build that is already present. Poll ``GET
    /api/engine`` for ``install_progress`` while this runs; the response itself
    arrives only on completion.
    """
    state = _state(request)
    info = await state.engine_manager.install(tag, force=force, activate=activate)
    return info.model_dump(mode="json")


@router.post("/engine/activate")
async def engine_activate(request: Request, tag: str = Body(..., embed=True)) -> dict[str, Any]:
    """Make an installed build the live engine, and pin it (D49-5).

    Three things, because doing fewer of them is what produced pin/active drift:
    ``active.json`` is rewritten (that is what new loads read),
    ``engine.pinned_tag`` is set and saved (that is what survives a restart, and
    what a rollback edits), and every saved ``extra_flags`` string is
    revalidated against the new build.

    ``offenders`` is ``[{model_id, errors}]`` for models whose saved flags no
    longer parse -- a warning, never a refusal: llama-server ignores flags it
    does not know, so the switch would otherwise take effect with a setting
    silently dropped. ``previous`` is the tag that was live before, for an
    undo. ``note`` is :data:`ENGINE_ACTIVATE_NOTE`.

    The tag must already be installed (``POST /api/engine/install`` first);
    activating a missing build is an error rather than an implicit download.
    """
    state = _state(request)
    before = state.engine_manager.active()
    previous = before.tag if before else None

    await state.engine_manager.activate(tag)

    # Same persistence path as PATCH /api/config: the dotted-path override so
    # the value is validated by the config model, save() to disk, then the
    # section swapped onto the live config object -- which every component
    # shares by reference, so the new pin is in force without a restart.
    updated = apply_overrides(state.config, {"engine.pinned_tag": tag})
    updated.save()
    state.config.engine = updated.engine

    offenders = await _flag_revalidation_sweep(state, tag)
    log.info(
        "engine activated",
        tag=tag,
        previous=previous,
        offenders=[entry.get("model_id") for entry in offenders],
    )
    return {
        "tag": tag,
        "previous": previous,
        "offenders": offenders,
        "note": ENGINE_ACTIVATE_NOTE,
    }


@router.post("/engine/smoke-test")
async def engine_smoke_test(request: Request, tag: str = Body(..., embed=True)) -> dict[str, Any]:
    """Load a tiny model on the GPU with this build and report whether it ran.

    A real micro-load, not a ``--version`` check: it is the only thing that
    catches a driver too old for the CUDA build. It costs a few seconds of GPU,
    which is why the route is D32-gated like the rest of ``/api/engine/``.
    """
    state = _state(request)
    ok, detail = await state.engine_manager.smoke_test(tag)
    return {"tag": tag, "ok": ok, "detail": detail}


@router.post("/engine/validate-flags")
async def engine_validate_flags(
    request: Request, extra_flags: Any = Body(...), tag: str | None = Body(None)
) -> dict[str, Any]:
    """Check expert flags against a build's own ``--help`` before saving them.

    ``extra_flags`` takes either shape: a single string (``"--foo 1 --bar"``) or
    a list of strings (``["--foo", "1", "--bar"]``, joined with spaces). Anything
    else is a 400 naming both (D49-11). ``tag`` defaults to the pinned build.

    Open without a credential even on an install with no ``server.api_key``:
    unlike its neighbours under ``/api/engine/`` this changes nothing on the box
    -- it reads one binary's help text -- and pre-checking a flag is exactly what
    a remote operator should be able to do before ``PUT /api/models/{id}/settings``
    (which is gated) makes it stick. The first validation run for a tag execs
    that build's ``--help`` and caches the accepted flags in
    ``engines/<tag>/flags.txt``, so it is slower than the runs after it.
    """
    state = _state(request)
    effective = tag or state.config.engine.pinned_tag
    errors = await state.engine_manager.validate_extra_flags(
        effective, _extra_flags_text(extra_flags)
    )
    return {"tag": effective, "ok": not errors, "errors": errors}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@router.get("/config")
async def get_config(request: Request) -> dict[str, Any]:
    """Config with every credential redacted to a short fingerprint.

    ``server.api_key``, ``mcp.pin`` and ``hf.token`` all go through
    :func:`~studioforge.api.auth.redact_config_dict`. The PIN matters most here:
    on the shipped default (``server.api_key`` unset) this route needs no
    credential at all, so returning the PIN in full handed the MCP control
    plane to any caller who could reach the port.
    """
    state = _state(request)
    data = state.config.to_yaml_dict()
    redact_config_dict(data)
    return {
        "config": data,
        "config_path": str(state.config.config_path),
        "restart_required_keys": sorted(RESTART_REQUIRED_KEYS),
    }


@router.patch("/config")
async def set_config(request: Request, updates: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Apply dotted-path updates, validate, persist, and report restart needs.

    One key here does less than it looks: ``engine.pinned_tag`` records which
    build *should* be live, but ``active.json`` is what a load actually reads, so
    setting the pin alone leaves the running engine untouched and shows up as
    ``drift`` on ``GET /api/engine``. Use ``POST /api/engine/activate`` to switch
    engines -- it does both, and revalidates saved expert flags (D49-5).
    """
    state = _state(request)
    updated = apply_overrides(state.config, updates)
    updated.save()
    changed = sorted(updates)
    needs_restart = [k for k in changed if k in RESTART_REQUIRED_KEYS]

    # Apply live where it is safe to do so: the running config object is shared
    # by reference with every component, so mutating the sub-models in place is
    # what makes a change take effect without a restart.
    for section in (
        "models",
        "planner",
        "gateway",
        "hf",
        "logging",
        "update",
        "engine",
        # `mcp` was missing, so PATCH {"mcp.pin_required": false} answered
        # {"updated": [...], "restart_required": []} while check_request kept
        # reading the OLD mcp config and enforcing the old PIN. The caller was
        # told a change was live that had not happened. (mcp.path/enabled are
        # restart-required -- they decide where the app mounts.)
        "mcp",
    ):
        setattr(state.config, section, getattr(updated, section))
    state.config.server.api_key = updated.server.api_key
    state.config.server.cors_origins = updated.server.cors_origins
    state.config.server.drain_timeout_s = updated.server.drain_timeout_s
    state.config.server.request_timeout_s = updated.server.request_timeout_s

    log.info("config updated", keys=changed, restart_required=needs_restart)
    return {"updated": changed, "restart_required": needs_restart}


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


@router.get("/logs")
async def get_logs(
    request: Request,
    n: int = Query(200, le=5000),
    level: str | None = Query(None),
) -> dict[str, Any]:
    return {"lines": RING_BUFFER.tail(n, level)}


@router.get("/logs/models/{model_id:path}")
async def get_model_log(
    model_id: str, request: Request, n: int = Query(200, le=5000)
) -> dict[str, Any]:
    """Per-model llama-server stderr, which is where load failures explain
    themselves."""
    state = _state(request)
    record = state.registry.resolve(model_id)
    resolved = record.id if record else model_id
    # tail_log reads the whole log file (it can be tens of MB after a long
    # session); keep that off the event loop.
    lines = await asyncio.to_thread(state.supervisor.tail_log, resolved, n)
    return {
        "model_id": resolved,
        "path": str(state.supervisor.log_path(resolved) or ""),
        "lines": lines,
    }


# ---------------------------------------------------------------------------
# Shutdown / restart helpers used by the watchdog and CLI
# ---------------------------------------------------------------------------


@router.post("/models/unload-all")
async def unload_all(request: Request) -> dict[str, Any]:
    """Unload every resident model, freeing all VRAM."""
    state = _state(request)
    unloaded = await state.manager.unload_all()
    log.info("unloaded all models", count=len(unloaded))
    return {"unloaded": unloaded, "count": len(unloaded)}


@router.get("/version")
async def version() -> dict[str, Any]:
    return {"version": __version__}


@router.get("/openclaw-setup")
async def openclaw_setup(request: Request) -> JSONResponse:
    """Everything needed to point OpenClaw at this server.

    Three blocks, not two: ``inference`` (the OpenAI-compatible env), ``mcp``
    (the stdio server registration) and ``companion_config`` (what ``sfctl``
    needs locally).

    ``next_steps`` rides along because the reader is often the agent itself:
    knowing the URL is not the same as knowing which of the merged tools to call
    first, and the answer has been stable since the catalog landed.
    """
    from studioforge.core.netinfo import reachable_urls

    state = _state(request)
    config = state.config
    # Give an address the other machine can actually use. `server.host` is a
    # BIND address and is normally 0.0.0.0, so echoing it produced a placeholder
    # the user had to substitute by hand -- and the Tailscale address is the one
    # that keeps working across network changes, so it goes first.
    endpoints = reachable_urls(config.server.port, host=config.server.host)
    host = endpoints[0]["ip"] if endpoints else "127.0.0.1"
    key = config.server.api_key or "not-required"
    # Same rule as /api/mcp/info: the PIN only goes out when a credential was
    # needed to get here, or when the caller is on this machine. See
    # studioforge.api.auth.may_reveal_pin.
    reveal = may_reveal_pin(request, config)
    return JSONResponse(
        {
            "inference": {
                "OPENAI_BASE_URL": f"http://{host}:{config.server.port}/v1",
                "OPENAI_API_KEY": key,
            },
            # Two spellings, because clients disagree. OpenClaw's key is
            # `mcp.servers` (docs.openclaw.ai/cli/mcp); Claude Code, Cline,
            # LibreChat and most others take a top-level `mcpServers` map.
            # Handing back only one of them is how a paste-this-in snippet ends
            # up silently registering nothing.
            "mcp": {
                "openclaw_cli": "openclaw mcp add studioforge --command sfctl --arg mcp",
                "openclaw_config": {
                    "mcp": {
                        "servers": {
                            "studioforge": {"command": "sfctl", "args": ["mcp"]},
                        }
                    }
                },
                "generic_config": {
                    "mcpServers": {
                        "studioforge": {"command": "sfctl", "args": ["mcp"]},
                    }
                },
            },
            "companion_config": {
                "server.url": f"http://{host}:{config.server.port}",
                "server.api_key": key,
            },
            "endpoints": endpoints,
            "mcp_pin": (config.mcp.pin if config.mcp.pin_required else None) if reveal else None,
            "mcp_pin_note": (
                None
                if reveal or not (config.mcp.pin_required and config.mcp.pin)
                else PIN_WITHHELD_NOTE
            ),
            "next_steps": [
                "`sfctl mcp` merges the app's 20 management tools with the "
                "watchdog's 10 recovery tools into one stdio tool list (30).",
                "Start with list_models: it returns the catalog newest-download-first, "
                "one options row per context size, exactly one marked recommended.",
                "Pass that row's load_args verbatim to load_model, then send prompts "
                f"to http://{host}:{config.server.port}/v1/chat/completions.",
                "New model: search_models -> repo_details(repo_id) -> download_model.",
                "VRAM missing: server_status names every holder; reclaim_orphan_engines "
                "(watchdog) kills leaked engine processes and nothing else.",
            ],
        }
    )


# ---------------------------------------------------------------------------
# Self-update
# ---------------------------------------------------------------------------


@router.get("/update")
async def update_status(request: Request, check: bool = Query(False)) -> dict[str, Any]:
    """Current vs latest version. ``check=true`` queries GitHub."""
    state = _state(request)
    updater = state.updater
    if not check:
        return updater.status_sync().to_dict()
    return (await updater.check()).to_dict()


@router.get("/update/releases")
async def update_releases(request: Request, limit: int = Query(20)) -> dict[str, Any]:
    state = _state(request)
    releases = await state.updater.list_releases(limit)
    return {
        "releases": [
            {
                "tag": r.tag,
                "version": r.version,
                "name": r.name,
                "published_at": r.published_at,
                "prerelease": r.prerelease,
                "asset_name": r.asset_name,
                "asset_size": r.asset_size,
                "has_checksum": r.checksum_url is not None,
                "newer": r.is_newer_than_current,
            }
            for r in releases
        ]
    }


async def _drain_for_update(state: Any) -> None:
    """Settle in-flight work before an update switches the release.

    Refusing to update while a download is mid-flight (rather than waiting it
    out) is deliberate: a model download can run for an hour, and silently
    killing it would lose the bytes AND leave a half-written file in the user's
    model library.
    """
    if state.downloader is not None:
        active = state.downloader.active()
        if active:
            raise BadRequestError(
                f"{len(active)} download(s) are in progress; pause or cancel them before updating",
                code="downloads_active",
            )
    deadline = time.time() + state.config.server.drain_timeout_s
    while time.time() < deadline:
        in_flight = sum(i.active_requests for i in state.supervisor.list())
        if in_flight == 0:
            return
        await asyncio.sleep(0.25)
    remaining = sum(i.active_requests for i in state.supervisor.list())
    if remaining:
        log.warning("update draining timed out", in_flight=remaining)


@router.post("/update/install")
async def update_install(
    request: Request,
    tag: str | None = Body(None),
    confirm: bool = Body(False),
    restart: bool = Body(True),
) -> dict[str, Any]:
    """Install a release, then health-check and auto-rollback on failure."""
    state = _state(request)
    if not confirm:
        raise BadRequestError(
            "installing an update restarts the server; pass confirm=true",
            param="confirm",
            code="confirmation_required",
        )
    return await state.updater.install(tag, drain=lambda: _drain_for_update(state), restart=restart)


@router.post("/update/rollback")
async def update_rollback(
    request: Request, confirm: bool = Body(False, embed=True)
) -> dict[str, Any]:
    state = _state(request)
    if not confirm:
        raise BadRequestError(
            "rolling back restarts the server; pass confirm=true",
            param="confirm",
            code="confirmation_required",
        )
    return await state.updater.rollback()


# ---------------------------------------------------------------------------
# HuggingFace search + downloads
# ---------------------------------------------------------------------------


def _require_downloader(state: Any) -> Any:
    if state.downloader is None:
        raise BadRequestError(
            "downloads are not available on this server", code="downloads_unavailable"
        )
    return state.downloader


#: Largest ``limit`` for which ``GET /api/hf/search?with_context=1`` is allowed.
#: One header read is a few MB of range requests; twenty of them would be a
#: couple of hundred MB per keystroke-driven search, which is why the search
#: route does not do it by default. Opening one repo (``/hf/repo/{id}``) always
#: does, because that is one read for a user who has already chosen.
CONTEXT_SEARCH_LIMIT = 5


@router.get("/hf/search")
async def hf_search(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(20, le=100),
    author: str | None = Query(None),
    sort: str = Query("downloads"),
    newer_than_days: int | None = Query(None, ge=1, le=3650),
    date_field: str = Query("updated"),
    with_context: bool = Query(False),
) -> dict[str, Any]:
    """Search HuggingFace for GGUF repos, with per-quant fit verdicts.

    ``sort`` and ``date_field`` are validated in :mod:`studioforge.core.hf_search`
    rather than by an ``enum`` annotation here, so the HTTP route, the GUI and
    the MCP tool all reject the same values with the same message -- and so the
    allowed set lives next to the HF names it maps to.

    ``with_context=1`` adds the per-quant context matrix, which costs one remote
    GGUF header read *per repo*. It is refused above ``limit``
    :data:`CONTEXT_SEARCH_LIMIT` rather than silently made slow: a 20-result
    search would pull a couple of hundred megabytes of headers.

    The response echoes the options back under ``sort_options`` /
    ``date_field_options``. The reader is often an LLM driving this endpoint
    with no schema in hand, and a discoverable menu is the difference between
    one wrong guess and three.
    """
    from studioforge.core.hf_search import DATE_FIELDS, SORT_KEYS, HfSearch

    state = _state(request)
    if with_context and limit > CONTEXT_SEARCH_LIMIT:
        raise BadRequestError(
            f"with_context reads one GGUF header per repo, so it is limited to "
            f"limit<={CONTEXT_SEARCH_LIMIT} (got {limit}). Search without it, then call "
            f"GET /api/hf/repo/{{repo_id}} for the repo you care about.",
            param="with_context",
        )
    search = HfSearch(state.config)
    try:
        repos = await search.search(
            q,
            limit=limit,
            author=author,
            sort=sort,
            newer_than_days=newer_than_days,
            date_field=date_field,
        )
        truncated = search.last_search_truncated
    finally:
        await search.aclose()
    return {
        "repos": [await _repo_payload(state, repo, with_context=with_context) for repo in repos],
        "sort": sort,
        "newer_than_days": newer_than_days,
        "date_field": date_field,
        "sort_options": list(SORT_KEYS),
        "date_field_options": list(DATE_FIELDS),
        # True means "the window holds more than this"; see HfSearch's page cap.
        "truncated": truncated,
    }


@router.get("/hf/repo/{repo_id:path}")
async def hf_repo(repo_id: str, request: Request) -> dict[str, Any]:
    """One repo's quants, each with a fit verdict and a context matrix.

    Unlike search this always reads the model's GGUF header (once per repo, from
    the smallest quant -- every quant shares the geometry), because the user has
    stopped browsing and is choosing. That read is what turns ``fit.approximate``
    off and fills in ``context_fit``.
    """
    from studioforge.core.hf_search import HfSearch

    state = _state(request)
    search = HfSearch(state.config)
    try:
        repo = await search.repo_info(repo_id)
    finally:
        await search.aclose()
    return await _repo_payload(state, repo, with_context=True)


async def _repo_payload(state: Any, repo: Any, *, with_context: bool = False) -> dict[str, Any]:
    """Repo summary plus a fit verdict (and optionally a context matrix) per quant.

    The verdict is attached here rather than in the GUI so the CLI and MCP get
    the same steering -- "won't fit" is the single most useful thing to know
    before spending an hour downloading.

    With ``with_context`` the model's real KV geometry is fetched once for the
    whole repo (from a sibling quant already in the registry if there is one,
    otherwise over HTTP range requests) and then reused for every quant: it is
    passed to ``fit_verdict`` as ``arch_hint``, which is what makes
    ``fit.approximate`` False, and it drives ``context_fit``. A failed header
    read degrades to today's bounded estimate with the reason attached; it never
    fails the request, because "could not read the header" is not a reason to
    stop the user browsing a repo.
    """
    from studioforge.core.downloader import fit_verdict
    from studioforge.core.hf_meta import ArchMeta, idle_planner, repo_arch_meta

    options = repo.logical_models()
    arch = ArchMeta()
    # Built once for the whole repo: constructing it enumerates the GPUs, and
    # every quant of a repo is answered against the same idle inventory.
    idle = None
    if with_context and options:
        arch = await repo_arch_meta(state.config, repo, registry=state.registry)
        idle = idle_planner(state.planner)

    entries = []
    for option in options:
        verdict: dict[str, Any] = {}
        try:
            verdict = fit_verdict(
                option, planner=state.planner, siblings=options, arch_hint=arch.meta
            )
        except Exception as exc:  # pragma: no cover - never block browsing
            log.debug("fit verdict failed", repo=repo.repo_id, error=str(exc))
        entry: dict[str, Any] = {
            "quant": option.quant,
            "total_bytes": option.total_bytes,
            "files": [f.filename for f in option.files],
            "mmproj": option.mmproj.filename if option.mmproj else None,
            "group_id": option.group_id,
            "fit": verdict,
        }
        if with_context:
            entry["context_fit"] = _context_fit(idle, option, arch)
        entries.append(entry)
    return {
        "repo_id": repo.repo_id,
        "publisher": repo.publisher,
        "name": repo.name,
        # HF's "downloads" is the trailing-30-day count, not an all-time total.
        # Named plainly here because that is the key clients already read; the
        # meaning is documented on hf_search.SORT_KEYS.
        "downloads": repo.downloads,
        "likes": repo.likes,
        # None whenever HF was not asked to sort by it -- absent, never zero.
        "trending_score": repo.trending_score,
        "gated": repo.gated,
        "private": repo.private,
        "last_modified": repo.last_modified,
        "created_at": repo.created_at,
        # Both ages are sent rather than one "age" keyed off date_field, so a
        # client can render "updated 3d ago (created 2y ago)" without a second
        # request and without knowing which field the search was filtered on.
        "updated_days_ago": repo.updated_days_ago,
        "created_days_ago": repo.created_days_ago,
        "quants": entries,
    }


def _context_fit(planner: Any, option: Any, arch: Any) -> dict[str, Any]:
    """The context matrix for one quant, or an empty dict if it cannot be built.

    ``LogicalDownload.total_bytes`` already includes the projector, so the two
    are separated again here: the planner charges an mmproj its own compute
    buffer, and folding it into the weights would under-count a vision model by
    a few hundred MB on every row.
    """
    from studioforge.core.hf_meta import context_matrix

    if planner is None:  # pragma: no cover - guarded by the caller
        return {}
    mmproj_bytes = option.mmproj.size_bytes if option.mmproj else 0
    weights = max(0, int(option.total_bytes) - int(mmproj_bytes))
    try:
        return context_matrix(
            arch.meta,
            weights,
            planner=planner,
            mmproj_bytes=mmproj_bytes,
            model_id=f"hf:{option.repo_id}#{option.quant}",
            source=arch.source,
            unavailable=arch.unavailable,
        )
    except Exception as exc:  # pragma: no cover - never block browsing
        log.debug("context matrix failed", repo=option.repo_id, error=str(exc))
        return {}


@router.get("/downloads")
async def list_downloads(request: Request) -> dict[str, Any]:
    """The queue, plus how much room is left where it is landing.

    ``disk`` rides along on the list rather than living on its own endpoint
    because every consumer that renders a queue wants both, and asking twice
    would let the two answers disagree by a poll interval. It is ``None`` when
    the volume cannot be measured (no ``models.dir`` yet, an unmapped drive) --
    never an invented zero, which a client would render as a full disk.
    """
    downloader = _require_downloader(_state(request))
    try:
        disk: dict[str, Any] | None = disk_report(
            downloader.models_dir(), downloader.queued_remaining_bytes()
        )
    except Exception as exc:  # noqa: BLE001 - a missing figure must not fail the list
        log.debug("downloads.disk_unavailable", error=str(exc))
        disk = None
    return {
        "downloads": [_progress_payload(p) for p in downloader.all()],
        "active": [_progress_payload(p) for p in downloader.active()],
        "disk": disk,
    }


def _progress_payload(progress: Any) -> dict[str, Any]:
    """One file's transfer as JSON.

    Built by hand rather than from ``DownloadProgress.to_dict()`` so the wire
    shape is a deliberate choice, which means the retry fields below have to be
    added here too: without them an API client sees a ``running`` download that
    has not moved for thirty seconds and no way to learn it is mid-backoff, and
    a ``failed`` one with no way to learn what Resume would keep.
    """
    return {
        "id": progress.id,
        "group_id": progress.group_id,
        "repo_id": progress.repo_id,
        "filename": progress.filename,
        "status": progress.status,
        "downloaded_bytes": progress.downloaded_bytes,
        "total_bytes": progress.total_bytes,
        "percent": progress.percent,
        "speed_bps": progress.speed_bps,
        "eta_s": progress.eta_s,
        "error": progress.error,
        "attempt": progress.attempt,
        "max_attempts": progress.max_attempts,
        "next_retry_at": progress.next_retry_at,
        "retry_in_s": progress.retry_in_s,
        "last_error": progress.last_error,
        "part_bytes": progress.part_bytes,
    }


@router.post("/downloads")
async def start_download(
    request: Request,
    repo_id: str = Body(...),
    quant: str | None = Body(None),
    include_mmproj: bool = Body(True),
    force: bool = Body(False),
) -> dict[str, Any]:
    """Queue a download, picking the quant when one is not named."""
    from studioforge.core.downloader import resolve_download_choice

    state = _state(request)
    downloader = _require_downloader(state)
    chosen = await resolve_download_choice(state.config, state.planner, repo_id, quant)
    group_id = await downloader.enqueue(chosen, include_mmproj=include_mmproj, force=force)
    return {
        "group_id": group_id,
        "repo_id": repo_id,
        "quant": chosen.quant,
        "total_bytes": chosen.total_bytes,
        "files": [f.filename for f in chosen.files],
        "mmproj": chosen.mmproj.filename if chosen.mmproj else None,
        "downloads": [_progress_payload(p) for p in downloader.group(group_id)],
    }


@router.post("/downloads/{group_id}/pause")
async def pause_download(group_id: str, request: Request) -> dict[str, Any]:
    downloader = _require_downloader(_state(request))
    await downloader.pause(group_id)
    return {"group_id": group_id, "status": downloader.group_status(group_id)}


@router.post("/downloads/{group_id}/resume")
async def resume_download(group_id: str, request: Request) -> dict[str, Any]:
    downloader = _require_downloader(_state(request))
    await downloader.resume(group_id)
    return {"group_id": group_id, "status": downloader.group_status(group_id)}


@router.delete("/downloads/{group_id}")
async def cancel_download(
    group_id: str, request: Request, delete_partial: bool = Query(True)
) -> dict[str, Any]:
    downloader = _require_downloader(_state(request))
    await downloader.cancel(group_id, delete_partial=delete_partial)
    return {"group_id": group_id, "canceled": True}


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def _benchmarker(state: Any) -> Benchmarker:
    """The process-wide benchmarker, created on first use.

    Lazily attached to ``app.state`` rather than built in the app factory so a
    server that never benchmarks pays nothing, and so the object -- which owns
    the "only one run at a time" lock -- is genuinely a singleton.
    """
    benchmarker = getattr(state, "benchmarker", None)
    if benchmarker is None:
        benchmarker = Benchmarker(state.manager, probe=state.probe)
        state.benchmarker = benchmarker
    return benchmarker


def _benchmark_jobs(state: Any) -> BenchmarkJobs:
    jobs = getattr(state, "benchmark_jobs", None)
    if jobs is None:
        jobs = BenchmarkJobs(limit=BENCHMARK_JOB_HISTORY)
        state.benchmark_jobs = jobs
    return jobs


@router.get("/evictions")
async def evictions(request: Request, since: float | None = Query(None)) -> dict[str, Any]:
    """Recent eviction events, newest first: who was unloaded, by what, and why.

    "Why did that model disappear?" used to need log archaeology once the
    evicting plan scrolled out of /api/status -- and it is the first question
    asked whenever a companion's model goes missing mid-conversation. Each
    event carries ``{ts, evicted, evicted_by, reason, freed_bytes, priority}``
    with ``reason`` in {plan, oom-retry, ttl, lease, removed}. In-memory ring:
    survives as long as the process, which is when the question is asked.
    """
    events = _state(request).manager.evictions(since)
    return {"evictions": list(reversed(events)), "count": len(events)}


def _require_job(state: Any, job_id: str) -> BenchmarkJob:
    job = _benchmark_jobs(state).get(job_id)
    if job is None:
        raise BadRequestError(
            f"unknown benchmark job {job_id!r}", code="job_not_found", status_code=404
        )
    return job


def _gpu_summary(gpu: Any) -> dict[str, Any]:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "total_bytes": gpu.total_bytes,
        "compute_capability": (
            list(gpu.compute_capability) if gpu.compute_capability is not None else None
        ),
    }


@router.get("/benchmark/modes")
async def benchmark_modes(request: Request) -> dict[str, Any]:
    """Every GPU placement this machine can benchmark, model-independent."""
    state = _state(request)
    gpus = state.probe.list_gpus()
    return {
        "modes": [mode.to_dict() for mode in available_modes(gpus)],
        "gpus": [_gpu_summary(gpu) for gpu in gpus],
    }


@router.get("/models/{model_id:path}/benchmark/modes")
async def model_benchmark_modes(
    model_id: str,
    request: Request,
    ctx_size: int = Query(DEFAULT_CTX_SIZE, ge=1),
) -> dict[str, Any]:
    """Modes for one model, each carrying the planner's applicability verdict.

    Inapplicable modes are listed with the rejection reason rather than hidden,
    so the GUI can grey out a placement *and* say why it is unavailable.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    entries = _benchmarker(state).modes_for(record, ctx_size=ctx_size)
    return {
        "model_id": record.id,
        "modes": [
            {**mode.to_dict(), "applicable": applicable, "skipped_reason": reason}
            for mode, applicable, reason in entries
        ],
    }


def _positive_int(value: Any, default: int, param: str) -> int:
    if value is None:
        return default
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise BadRequestError(f"{param!r} must be an integer", param=param) from exc
    if number <= 0:
        raise BadRequestError(f"{param!r} must be greater than zero", param=param)
    return number


@router.post("/models/{model_id:path}/benchmark")
async def start_benchmark(
    model_id: str, request: Request, payload: dict[str, Any] | None = Body(None)
) -> JSONResponse:
    """Start a benchmark in the background and return a job id.

    A background job rather than a synchronous response because a four-mode run
    on a 17 GiB model is minutes of loading and generating -- far past any
    sensible HTTP timeout. Clients poll ``/api/benchmark/jobs/{job_id}``.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())

    data = payload or {}
    ctx_size = _positive_int(data.get("ctx_size"), DEFAULT_CTX_SIZE, "ctx_size")
    max_tokens = _positive_int(data.get("max_tokens"), DEFAULT_MAX_TOKENS, "max_tokens")
    prompt = data.get("prompt")
    if prompt is not None and not isinstance(prompt, str):
        raise BadRequestError("'prompt' must be a string or null", param="prompt")
    requested = data.get("modes")
    if requested is not None and not isinstance(requested, list):
        raise BadRequestError("'modes' must be a list of mode keys or null", param="modes")

    benchmarker = _benchmarker(state)
    if benchmarker.busy:
        raise ModelBusyError(
            "a benchmark is already running; benchmarks are serialized because "
            "concurrent runs would compete for the VRAM being measured",
            code="benchmark_busy",
        )

    available = [mode.key for mode, _, _ in benchmarker.modes_for(record, ctx_size=ctx_size)]
    if requested:
        wanted = [str(key) for key in requested]
        unknown = [key for key in wanted if key not in available]
        if unknown:
            raise BadRequestError(
                "unknown benchmark mode(s): "
                + ", ".join(unknown)
                + "; available: "
                + (", ".join(available) or "none"),
                param="modes",
            )
        selected = [key for key in available if key in set(wanted)]
    else:
        selected = available
    if not selected:
        raise BadRequestError(
            "no GPU placements are available to benchmark on this machine",
            code="no_benchmark_modes",
        )

    job = _benchmark_jobs(state).create(record.id, selected)
    job.task = asyncio.create_task(
        _run_benchmark_job(
            state,
            benchmarker,
            record,
            job,
            ctx_size=ctx_size,
            max_tokens=max_tokens,
            prompt=prompt,
        ),
        name=f"sf-benchmark-{job.job_id}",
    )
    return JSONResponse(
        {"job_id": job.job_id, "model_id": record.id, "modes": selected}, status_code=202
    )


async def _run_benchmark_job(
    state: Any,
    benchmarker: Benchmarker,
    record: Any,
    job: BenchmarkJob,
    *,
    ctx_size: int,
    max_tokens: int,
    prompt: str | None,
) -> None:
    """Drive one job to a terminal state; never lets an exception escape."""
    try:
        report = await benchmarker.run(
            record,
            modes=job.modes,
            ctx_size=ctx_size,
            max_tokens=max_tokens,
            prompt=prompt,
            on_progress=job.on_progress,
            cancel_event=job.cancel_event,
        )
    except asyncio.CancelledError:
        job.state = "canceled"
        job.error = "benchmark canceled"
        raise
    except StudioForgeError as exc:
        job.state = "failed"
        job.error = exc.message
        log.warning("benchmark job failed", job_id=job.job_id, error=exc.message)
    except Exception as exc:
        job.state = "failed"
        job.error = str(exc)
        log.error("benchmark job failed", job_id=job.job_id, error=str(exc))
    else:
        job.report = report.to_dict()
        job.phase = "done"
        job.fraction = 1.0
        job.state = "canceled" if job.cancel_event.is_set() else "completed"
        if job.state == "completed":
            # Off the event loop: sqlite writes block, and the report is a few
            # kilobytes of JSON.
            try:
                await asyncio.to_thread(state.db.save_benchmark, record.id, job.report)
            except Exception as exc:  # pragma: no cover - history is not critical
                log.warning(
                    "could not persist benchmark report", model_id=record.id, error=str(exc)
                )


@router.get("/benchmark/jobs")
async def benchmark_jobs(request: Request) -> dict[str, Any]:
    """Every known benchmark job, newest first.

    Without this the job table was write-only -- ``job_id`` or nothing. A
    second client, or the same client after a crash, could not discover a
    running benchmark except by getting ``503 benchmark_busy`` on its next
    POST; the playbook literally said "poll the job_id you started".
    """
    jobs = _benchmark_jobs(_state(request)).all()
    return {"jobs": [job.to_dict() for job in reversed(jobs)], "count": len(jobs)}


@router.get("/benchmark/jobs/{job_id}")
async def benchmark_job(job_id: str, request: Request) -> dict[str, Any]:
    return _require_job(_state(request), job_id).to_dict()


@router.delete("/benchmark/jobs/{job_id}")
async def cancel_benchmark_job(job_id: str, request: Request) -> dict[str, Any]:
    """Ask a running job to stop.

    The stop happens at a mode boundary rather than mid-load: killing a
    half-started llama-server would leak a child process and leave the model's
    device override unrestored, which is exactly the surprise this subsystem
    must not create.
    """
    job = _require_job(_state(request), job_id)
    if job.state != "running":
        return {**job.to_dict(), "canceled": False}
    job.cancel_event.set()
    return {**job.to_dict(), "canceled": True}


@router.get("/models/{model_id:path}/benchmarks")
async def list_model_benchmarks(
    model_id: str, request: Request, limit: int = Query(10, ge=1, le=100)
) -> dict[str, Any]:
    """Persisted benchmark history for one model, newest first."""
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    rows = await asyncio.to_thread(state.db.list_benchmarks, record.id, limit)
    return {
        "model_id": record.id,
        "benchmarks": [{"id": row["id"], "ts": row["ts"], "report": row["report"]} for row in rows],
    }


# ---------------------------------------------------------------------------
# Parallel benchmark (WP19 / D37)
# ---------------------------------------------------------------------------


def _parallel_benchmarker(state: Any) -> ParallelBenchmarker:
    """The process-wide parallel benchmarker, created on first use.

    Shares the placement benchmarker's lock rather than owning one: two
    measurement runs at once compete for exactly the resource each is
    measuring, and a concurrency sweep is no less a benchmark than a placement
    sweep.
    """
    # The factory lives in core so the MCP tool reaches the SAME object: two
    # runners would be two locks, which is no lock. `_benchmarker` is called
    # first so the placement benchmarker is this route's usual singleton rather
    # than one the factory happens to create.
    _benchmarker(state)
    return parallel_bench.for_state(state)


@router.post("/models/{model_id:path}/benchmark-parallel")
async def start_parallel_benchmark(
    model_id: str, request: Request, payload: dict[str, Any] | None = Body(None)
) -> JSONResponse:
    """Measure how many parallel slots this model is worth running (WP19).

    Loads the model once at the chosen placement with as many slots as it holds,
    fires 1 / 2 / 4 / 8 concurrent completions, records the curve, and leaves the
    rig as it found it. A background job for the same reason the placement
    benchmark is one -- minutes of loading and generating past any HTTP timeout
    -- and pollable through the same ``/api/benchmark/jobs/{job_id}``.

    Body: ``mode`` (a hardware-mode key such as ``dual_5090``), ``devices``,
    ``ctx_size``, ``kv_cache_type``, ``streams``, ``prompt_tokens``,
    ``max_tokens``. All optional; the default is the placement's own optimal on
    the planner's choice of cards, which is the load the catalog recommends and
    therefore the one worth measuring.
    """
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())

    data = payload or {}
    mode = data.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise BadRequestError("'mode' must be a hardware-mode key or null", param="mode")
    devices = data.get("devices")
    if devices is not None and not (
        isinstance(devices, list) and all(isinstance(d, int) for d in devices)
    ):
        raise BadRequestError("'devices' must be a list of CUDA indices or null", param="devices")
    streams = data.get("streams")
    if streams is not None and not (
        isinstance(streams, list) and all(isinstance(n, int) and n >= 1 for n in streams)
    ):
        raise BadRequestError(
            "'streams' must be a list of positive concurrency levels or null", param="streams"
        )
    ctx_size = data.get("ctx_size")
    if ctx_size is not None:
        ctx_size = _positive_int(ctx_size, 0, "ctx_size")
    prompt_tokens = _positive_int(data.get("prompt_tokens"), DEFAULT_PROMPT_TOKENS, "prompt_tokens")
    max_tokens = _positive_int(data.get("max_tokens"), PARALLEL_MAX_TOKENS, "max_tokens")
    # The same validation an ordinary load gets, BEFORE the 202: a bad cache
    # type or a CUDA index this box does not have would otherwise be accepted
    # here and fail minutes later inside the job, where the caller has to go
    # and read it off /api/benchmark/jobs.
    validate_load_args(
        ctx_size=ctx_size,
        parallel=None,
        kv_cache_type=data.get("kv_cache_type"),
        kv_cache_type_v=data.get("kv_cache_type_v"),
        devices=devices,
        known_devices=state.manager._known_devices(),
    )

    runner = _parallel_benchmarker(state)
    # Refused here, as a 503 with retry_after_s, rather than accepted as a job
    # that fails on its first line: the MCP tool answers the same way, and a
    # caller polling a job for "the server was busy" learns it minutes late.
    runner.refuse_if_busy()
    levels = [int(n) for n in (streams or DEFAULT_STREAMS)]
    job = _benchmark_jobs(state).create(record.id, [str(n) for n in sorted(set(levels))])
    job.task = asyncio.create_task(
        _run_parallel_job(
            state,
            runner,
            record,
            job,
            mode=mode,
            devices=devices,
            ctx_size=ctx_size,
            kv_cache_type=data.get("kv_cache_type"),
            kv_cache_type_v=data.get("kv_cache_type_v"),
            streams=levels,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        ),
        name=f"sf-parallel-{job.job_id}",
    )
    return JSONResponse(
        {"job_id": job.job_id, "model_id": record.id, "streams": sorted(set(levels)), "mode": mode},
        status_code=202,
    )


async def _run_parallel_job(
    state: Any,
    runner: ParallelBenchmarker,
    record: Any,
    job: BenchmarkJob,
    **kwargs: Any,
) -> None:
    """Drive one parallel sweep to a terminal state; never lets one escape.

    Deliberately the same job table and the same terminal-state handling as the
    placement benchmark: a client that already polls ``/api/benchmark/jobs`` has
    nothing new to learn, and a second job mechanism would be a second set of
    ways to leak a running task.
    """
    try:
        report = await runner.run(
            record, on_progress=job.on_progress, cancel_event=job.cancel_event, **kwargs
        )
    except asyncio.CancelledError:
        job.state = "canceled"
        job.error = "parallel benchmark canceled"
        raise
    except StudioForgeError as exc:
        job.state = "failed"
        job.error = exc.message
        log.warning("parallel benchmark job failed", job_id=job.job_id, error=exc.message)
    except Exception as exc:
        job.state = "failed"
        job.error = str(exc)
        log.error("parallel benchmark job failed", job_id=job.job_id, error=str(exc))
    else:
        job.report = report.to_dict()
        job.phase = "done"
        job.fraction = 1.0
        job.state = "canceled" if job.cancel_event.is_set() else "completed"


@router.get("/models/{model_id:path}/parallel-observations")
async def list_parallel_observations(
    model_id: str, request: Request, limit: int = Query(64, ge=1, le=500)
) -> dict[str, Any]:
    """The measured slot sweeps behind this model's ``recommended_parallel``."""
    state = _state(request)
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    # Through the manager, which answers [] for a data directory that predates
    # the table (or a test's db=None) instead of a 500.
    rows = await asyncio.to_thread(state.manager.parallel_observations, record.id, limit=limit)
    return {"model_id": record.id, "observations": rows}
