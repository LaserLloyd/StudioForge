"""The model catalog: one table an LLM can pick a load out of.

The problem this solves. An agent that wants to run a model has to answer four
questions before it can call ``load_model``: which model, at what context, with
how many slots, and will that even fit right now. Answering them today means a
``/v1/models`` call, a ``/profiles`` call per model, arithmetic about KV caches,
and a guess about speed. The catalog answers all four in one call, per model,
as a table of *rows the agent can hand back verbatim*.

Shape of the answer::

    {
      "catalog_hint": "...one paragraph explaining the columns...",
      "models": [                       # NEWEST DOWNLOAD FIRST
        {
          "id": ..., "summary": ..., "downloaded_at": ...,
          "quantization": ..., "architecture": ..., "capabilities": [...],
          "attention_kind": "iswa",     # why its long windows are cheap
          "options": [                  # one row per context tier
            {"ctx_per_slot": 65536, "fits": true, "devices": [0, 1],
             "max_parallel": 4, "est_gen_tps": 58.0,
             "est_gen_tps_full_ctx": 41.0, "recommended": true,
             "load_args": {"model_id": ..., "ctx_size": 65536, ...}}
          ]
        }
      ]
    }

Three design rules, all of them about the consumer being a language model:

**Newest first, always.** The user works from the last thing they downloaded,
so ``downloaded_at`` (the newest mtime across a model's GGUF shards -- a
multi-part download finishes on its *last* file) is the sort key and it sorts
descending. This is the one ordering the catalog guarantees.

**Nothing is left as an exercise.** Every row carries ``load_args``, which is
exactly the argument object the ``load_model`` tool accepts. An agent that has
chosen a row is done choosing: it passes the object through. No field in it
needs to be computed, converted or looked up.

**The planner is the only authority on placement.** The rows do not
re-implement fitting -- they call :meth:`Planner.plan_load` once per context
tier, so exclusions, reservations, quant affinity, per-model device overrides
and *current free VRAM* are all respected by construction. Each row is computed
twice: once against the machine as it is now (``fits``/``devices``) and once
against every GPU idle (``if_gpus_idle``), so the agent can tell "impossible"
apart from "possible after unloading something".
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from studioforge.core import throughput
from studioforge.core.planner import (
    Planner,
    attention_kind,
    is_moe,
    kv_read_bytes_per_slot,
)
from studioforge.logging import get_logger
from studioforge.types import GB, MB, GpuInfo, LoadPlan, ModelRecord

log = get_logger(__name__)

#: Context tiers the catalog reports, ascending. Powers of two from 16k to 1M:
#: every one of them is a size a client actually asks for, and the doublings
#: are where the KV cache cost visibly changes the answer. Tiers above the
#: model's ``n_ctx_train`` are dropped -- serving past the trained window needs
#: RoPE scaling and quietly degrades quality (D14), so offering it as a menu
#: item would be offering a trap.
CTX_TIERS: tuple[int, ...] = (16384, 32768, 65536, 131072, 262144, 524288, 1048576)

#: Paragraph returned at the top of every catalog. Written for a model, not a
#: human: it is the difference between an agent that picks a row and an agent
#: that asks the user which context size it should use.
CATALOG_HINT = (
    "Each model lists loading options, one row per context size; models are "
    "sorted by download date, newest first. Per row: ctx_per_slot is the "
    "context EACH concurrent conversation gets; fits says whether it loads on "
    "the VRAM free right now; devices are the CUDA indices it would use; "
    "kv_cache_type is the KV quantization picked to reach that context; "
    "vram_mb is the whole load at this row's max_parallel; max_parallel is "
    "how many conversations the placement sustains, and parallel_limited_by "
    "names what caps it ('vram', 'knee' = where extra slots stop buying "
    "throughput, or 'cap'). Speeds are tokens/second: est_gen_tps is ONE "
    "stream with ~8k tokens of context in the window (a typical turn); "
    "est_gen_tps_full_ctx is that same stream with the window nearly full -- "
    "generation slows as context fills, so the truth is between the two; "
    "est_gen_tps_batched is the aggregate across all slots; est_prompt_tps is "
    "prompt ingestion. measured_gen_tps / measured_prompt_tps are real "
    "observations of this exact placement, and confidence is 'measured', "
    "'calibrated' (corrected by a learned factor -- calibration.basis says "
    "from what: 'model+devices', 'model', 'peers' = other models of the same "
    "density on the same hardware, or 'none') or 'estimated' (nominal "
    "hardware numbers: an order of magnitude, not a promise). The model's "
    "attention_kind explains its context prices: 'full' keeps every token on "
    "every layer, while 'iswa' and 'hybrid' keep only a fraction of the "
    "context in KV, which is why their huge windows stay cheap and fast. "
    "if_gpus_idle repeats the verdict as it would be with every GPU free, so "
    "a row that does not fit now may still be reachable by unloading "
    "something. Exactly one row is recommended:true -- the highest context "
    "that fits at or above this server's default context floor, preferring "
    "one that also sustains two slots; a second slot is never bought by "
    "dropping under the floor, because an agent transcript that does not fit "
    "is a failed task. Take that row unless you need more context, then pass "
    "its load_args object verbatim to the load_model tool."
)

#: How long a built catalog stays servable before it is rebuilt. Short, because
#: `fits` is a statement about free VRAM *right now* and a stale yes is worse
#: than a slow no.
CACHE_TTL_S = 20.0


# ---------------------------------------------------------------------------
# Probe wrappers
# ---------------------------------------------------------------------------


class _SnapshotProbe:
    """A GPU probe frozen at one instant.

    Two reasons, both load-bearing. Consistency: a catalog built from a live
    probe would ask NVML for free VRAM once per model per tier, so an early row
    and a late row could disagree about the same card and no single load would
    match the table. Cost: that is hundreds of NVML round-trips per build, and
    the per-process enumeration behind a rejection is far more expensive still.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.backend = getattr(inner, "backend", "unknown")
        self._gpus: list[GpuInfo] = list(inner.list_gpus())
        self._processes: list[Any] | None = None

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [g.model_copy(deep=True) for g in self._gpus]

    def get_gpu(self, index: int) -> GpuInfo | None:
        for gpu in self._gpus:
            if gpu.index == index:
                return gpu.model_copy(deep=True)
        return None

    def compute_processes(self) -> list[Any]:
        if self._processes is None:
            getter = getattr(self._inner, "compute_processes", None)
            try:
                self._processes = list(getter()) if getter is not None else []
            except Exception:  # noqa: BLE001 - attribution must never break a build
                self._processes = []
        return [p.model_copy(deep=True) if hasattr(p, "model_copy") else p for p in self._processes]

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


class _IdleProbe(_SnapshotProbe):
    """The same GPUs, reported as if nothing were loaded on them.

    Free VRAM becomes total VRAM; ``planner.headroom_fraction``,
    ``reserved_mb`` and ``excluded_devices`` still apply, because those describe
    memory that is *never* ours regardless of what is loaded. This is what
    powers the ``if_gpus_idle`` column: "what this rig could do for this model
    if you cleared it", which is precisely the question an agent needs answered
    before deciding whether unloading something is worth it.
    """

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self._gpus = [
            g.model_copy(update={"free_bytes": g.total_bytes, "used_bytes": 0}) for g in self._gpus
        ]

    def compute_processes(self) -> list[Any]:
        return []


# ---------------------------------------------------------------------------
# Small projections
# ---------------------------------------------------------------------------


def model_type(record: ModelRecord) -> str:
    """LM Studio's ``type`` for a model: llm / vlm / embeddings / rerank.

    The single implementation behind both ``/api/v0/models`` and the catalog,
    so the two can never disagree about whether something is a vision model.
    """
    if record.kind == "embedding":
        return "embeddings"
    if record.kind == "rerank":
        return "rerank"
    return "vlm" if record.capabilities.vision else "llm"


def capability_list(record: ModelRecord) -> list[str]:
    """Capability names that are true, in the same spelling ``/v1/models`` uses."""
    return [name for name, on in record.capabilities.model_dump().items() if on]


#: Settings that change what a load *is*, as opposed to how it is served.
#: Reported as ``settings_pinned`` so an agent can see why a model ignored the
#: ctx_size it asked for before it concludes the server is broken.
_PINNED_SETTING_FIELDS: tuple[str, ...] = (
    "ctx_size",
    "kv_cache_type",
    "kv_cache_type_v",
    "parallel",
    "max_parallel_cap",
    "kv_unified",
    "device_override",
    "draft_model_id",
    "engine_tag",
    "ttl_s",
)


def pinned_settings(record: ModelRecord) -> dict[str, Any]:
    """The per-model settings that override catalog defaults, if any."""
    out: dict[str, Any] = {}
    for field in _PINNED_SETTING_FIELDS:
        value = getattr(record.settings, field, None)
        if value is not None:
            out[field] = value
    if record.settings.pinned:
        out["pinned"] = True
    return out


def _params_b(meta: Any, weights_bytes: int) -> tuple[float | None, float | None]:
    """``(total_B, active_B)`` in billions, or ``(None, None)``."""
    total = throughput.total_params(meta, weights_bytes)
    if total <= 0:
        return None, None
    active = throughput.active_params(meta, weights_bytes)
    return round(total / 1e9, 1), round(active / 1e9, 1)


#: How an attention kind is spelled in a one-line summary. ``"full"`` and
#: ``"unknown"`` are deliberately absent: the first is the default nobody needs
#: told, and the second is an admission, not a property. Only the two kinds that
#: change what a context number *costs* earn a place in a forty-line list.
_ATTENTION_TAGS: dict[str, str] = {"iswa": "iSWA", "hybrid": "hybrid"}


def summarize(
    record: ModelRecord,
    params_total: float | None,
    params_active: float | None,
    attention: str | None = None,
) -> str:
    """One line: what this model is, in the order a chooser cares about.

    Family, size, quantization, attention shape, what it can do, how much disk
    it takes. Short enough that a list of forty of them is still readable in a
    context window.

    ``attention`` is :func:`~studioforge.core.planner.attention_kind`, and it is
    on this line rather than only in the entry because it is the single fact
    that makes the rest of the table legible: an iSWA or hybrid model reaches
    262k tokens for a fraction of a full-attention model's KV cache, so "why is
    this 31B offering four slots at 262k and that one offering one" has an
    answer visible in the summary itself.
    """
    meta = record.meta
    parts: list[str] = [record.architecture or "unknown"]
    if params_total is not None:
        if params_active is not None and params_active < params_total:
            parts.append(f"{params_total:g}B-A{params_active:g}B MoE")
        else:
            parts.append(f"{params_total:g}B")
    parts.append(record.quant or "unknown")
    tag = _ATTENTION_TAGS.get(attention or "")
    if tag:
        parts.append(tag)
    caps = [c for c in ("tools", "thinking", "vision") if getattr(record.capabilities, c, False)]
    if caps:
        parts.append("+".join(caps))
    elif record.kind != "chat":
        parts.append(record.kind)
    parts.append(f"{record.size_bytes / GB:.1f} GB")
    trained = int(getattr(meta, "n_ctx_train", 0) or 0) if meta else 0
    if trained:
        parts.append(f"{trained} ctx train")
    return " | ".join(parts)


def ctx_tiers_for(record: ModelRecord, tiers: Sequence[int] = CTX_TIERS) -> list[int]:
    """Context tiers worth reporting for this model, ascending.

    The standard ladder capped at the trained window, plus the model's own
    pinned ``ctx_size`` when it has one -- a pinned value is what a load will
    actually use, so leaving it off the table would mean the table never
    describes the load that happens.
    """
    trained = int(getattr(record.meta, "n_ctx_train", 0) or 0) if record.meta else 0
    usable = [t for t in tiers if trained <= 0 or t <= trained]
    if not usable and trained > 0:
        # A short-window model (n_ctx_train below the smallest tier) still
        # deserves a row: its own window is the only meaningful option.
        usable = [trained]
    pinned = record.settings.ctx_size
    if pinned and pinned not in usable:
        usable.append(int(pinned))
    return sorted(set(usable))


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------


def _capacity_for(planner: Planner, devices: Sequence[int], *, forced: bool) -> int:
    gpu_map = {g.index: g for g in planner.probe.list_gpus()}
    return sum(planner.usable_bytes(gpu_map[d], forced=forced) for d in devices if d in gpu_map)


def slots_for_plan(planner: Planner, record: ModelRecord, plan: LoadPlan) -> tuple[int, str, int]:
    """``(slots, bound, vram_bytes)`` for this placement.

    One call to :meth:`Planner.size_slots` -- **the same function a real load
    goes through**. The catalog used to carry its own copy of that arithmetic
    (an analytic bound, then a walk-down to verify it), and a copy is exactly
    how a table starts advertising four slots where a load would settle for
    two. The walk lives in the planner now; this function's whole remaining job
    is to work out the capacity to walk against and to unpack the result.

    Computed at all -- rather than read off ``plan.max_parallel`` -- because the
    plan only carries an *estimated* slot count when ``models.default_parallel``
    is ``"auto"``. With an explicit integer configured (a legitimate deployment
    choice, and what this rig runs) every plan reports one slot, and the
    catalog's whole concurrency column would collapse to 1. The question the
    catalog asks is "what could this placement sustain", which is well defined
    regardless of the current policy.

    The VRAM figure comes back **at the chosen slot count**, so ``vram_mb``
    describes the load ``load_args`` would actually produce rather than the one
    the planner happened to size while checking the fit.
    """
    forced = bool(record.settings.device_override)
    capacity = _capacity_for(planner, plan.devices, forced=forced)
    estimate, slots, _max_parallel, bound = planner.size_slots(
        record,
        ctx=plan.ctx_size,
        kv_k=plan.kv_cache_type,
        kv_v=plan.kv_cache_type_v,
        devices=plan.devices,
        capacity_bytes=capacity,
        base_estimate=plan.estimate,
    )
    return slots, bound, estimate.total_bytes


def _plan_at(
    planner: Planner,
    record: ModelRecord,
    ctx: int,
    *,
    loaded: Sequence[Any] = (),
) -> Any:
    """One planner call for one tier, with eviction off.

    ``allow_evict=False`` is what makes ``fits`` mean "loads right now without
    disturbing anything". The eviction case is not guessed at here -- it is
    answered exactly by the ``if_gpus_idle`` column, which is the same question
    asked of an empty machine.
    """
    try:
        return planner.plan_load(record, ctx_size=ctx, loaded=loaded, allow_evict=False)
    except Exception as exc:  # noqa: BLE001 - one bad model must not empty the catalog
        log.debug("catalog plan failed", model_id=record.id, ctx=ctx, error=str(exc))
        return None


def estimate_speed(
    planner: Planner,
    record: ModelRecord,
    plan: LoadPlan,
    slots: int,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Speed for one placement, quoted at **two** fills of the same window.

    Generation gets slower as the window fills, because every decode step
    re-reads the KV cache. One number therefore cannot describe a row honestly:
    quoting the empty window flatters a 262k option that nobody will experience
    at 262k, and quoting the full one condemns it for a cost only its last token
    pays. So the row carries both ends:

    * ``gen_tps`` at :data:`throughput.REFERENCE_FILL_TOKENS` (8k, or the row's
      own window when that is smaller) -- one ordinary turn, which is what a
      user compares against a number they have seen elsewhere.
    * ``gen_tps_full_ctx`` at the row's whole ``ctx_per_slot`` -- the same
      stream with the window nearly full, the pessimistic end.

    Both come from :func:`planner.kv_read_bytes_per_slot`, which knows the
    model's per-layer geometry. Multiplying a uniform bytes-per-token by the
    fill (what this did before) charged a Gemma-4 31B 258 GB of KV reads per
    token at 262k and reported 1.9 tok/s for a model that measures 39.4.

    ``prompt_tps`` and ``gen_tps_batched`` are quoted at the reference fill
    too: prefill does not depend on the fill at all, and the batched aggregate
    is read alongside ``gen_tps``, so pinning them to the same instant keeps the
    row internally consistent.
    """
    gpu_map = {g.index: g for g in planner.probe.list_gpus()}
    split = plan.per_gpu_bytes or {
        d: plan.estimate.total_bytes // max(1, len(plan.devices)) for d in plan.devices
    }

    def kv_read(ctx_fill: int) -> int:
        if record.meta is None:
            return 0
        return kv_read_bytes_per_slot(
            record.meta,
            kv_k=plan.kv_cache_type,
            kv_v=plan.kv_cache_type_v,
            ctx_fill=ctx_fill,
        )

    def at(ctx_fill: int) -> dict[str, Any]:
        return throughput.estimate(
            record.meta,
            plan.estimate.weights_bytes,
            split,
            kv_read_bytes_per_slot=kv_read(ctx_fill),
            parallel=slots,
            gpus=gpu_map,
            efficiency=float(calibration.get("gen", 1.0)),
            prompt_efficiency=float(calibration.get("prompt", 1.0)),
            knee=slots,
        )

    ctx = int(plan.ctx_size)
    reference_fill = min(ctx, throughput.REFERENCE_FILL_TOKENS)
    speed = at(reference_fill)
    # A window at or below the reference fill *is* the full-context case; a
    # second identical call would only cost arithmetic.
    speed["gen_tps_full_ctx"] = speed["gen_tps"] if ctx <= reference_fill else at(ctx)["gen_tps"]
    return speed


#: ``devices -> calibration``: the factor to apply to one placement's estimate.
#: A function rather than a value because the tiers in
#: :func:`throughput.calibrate` are device-specific, and one model's rows are
#: routinely placed differently at 16k and at 262k.
Calibrator = Callable[[Sequence[int]], Mapping[str, Any]]


def _idle_variant(
    idle_planner: Planner, record: ModelRecord, ctx: int, calibrate_for: Calibrator
) -> dict[str, Any]:
    """The same tier, judged against every GPU idle."""
    plan = _plan_at(idle_planner, record, ctx)
    if not isinstance(plan, LoadPlan):
        return {"fits": False}
    slots, bound, vram = slots_for_plan(idle_planner, record, plan)
    speed = estimate_speed(idle_planner, record, plan, slots, calibrate_for(plan.devices))
    return {
        "fits": True,
        "devices": list(plan.devices),
        "kv_cache_type": plan.kv_cache_type,
        "vram_mb": round(vram / MB),
        "max_parallel": slots,
        "parallel_limited_by": bound,
        "est_gen_tps": speed["gen_tps"],
        "est_gen_tps_full_ctx": speed["gen_tps_full_ctx"],
        "est_gen_tps_batched": speed["gen_tps_batched"],
    }


def _option_row(
    planner: Planner,
    idle_planner: Planner,
    record: ModelRecord,
    ctx: int,
    *,
    loaded: Sequence[Any],
    observations: Sequence[Mapping[str, Any]],
    calibrate_for: Calibrator,
) -> dict[str, Any]:
    """One (model, context) row of the catalog."""
    idle = _idle_variant(idle_planner, record, ctx, calibrate_for)
    plan = _plan_at(planner, record, ctx, loaded=loaded)

    if not isinstance(plan, LoadPlan):
        reason = getattr(plan, "reason", None) or "does not fit in the VRAM free right now"
        return {
            "ctx_per_slot": ctx,
            "fits": False,
            "reason": reason,
            "devices": [],
            "kv_cache_type": None,
            "vram_mb": round(getattr(plan, "required_bytes", 0) / MB) if plan else None,
            "max_parallel": 0,
            "parallel_limited_by": "vram",
            "est_prompt_tps": None,
            "est_gen_tps": None,
            "est_gen_tps_full_ctx": None,
            "est_gen_tps_batched": None,
            "measured_gen_tps": None,
            "measured_prompt_tps": None,
            "confidence": "estimated",
            "if_gpus_idle": idle,
            "load_args": None,
            "recommended": False,
        }

    slots, bound, vram = slots_for_plan(planner, record, plan)
    calibration = calibrate_for(plan.devices)
    speed = estimate_speed(planner, record, plan, slots, calibration)
    measured = throughput.measured_for(
        observations, devices=plan.devices, ctx_size=ctx, parallel=slots
    )
    return {
        "ctx_per_slot": ctx,
        "fits": True,
        "devices": list(plan.devices),
        "kv_cache_type": plan.kv_cache_type,
        "vram_mb": round(vram / MB),
        "max_parallel": slots,
        "parallel_limited_by": bound,
        "est_prompt_tps": speed["prompt_tps"],
        "est_gen_tps": speed["gen_tps"],
        "est_gen_tps_full_ctx": speed["gen_tps_full_ctx"],
        "est_gen_tps_batched": speed["gen_tps_batched"],
        "measured_gen_tps": measured["gen_tps"],
        "measured_prompt_tps": measured["prompt_tps"],
        "confidence": throughput.confidence_for(measured, calibration),
        "if_gpus_idle": idle,
        # Exactly what the MCP `load_model` tool accepts. Pass it through
        # unchanged; every value here is already the value that tool wants.
        "load_args": {
            "model_id": record.id,
            "ctx_size": ctx,
            "parallel": slots,
            "kv_cache_type": plan.kv_cache_type,
        },
        "recommended": False,
    }


def recommendation_floor(config: Any, record: ModelRecord) -> int:
    """The smallest context this server considers a usable default (D14).

    ``models.default_ctx`` is the floor the planner's own context ladder never
    walks below, so the catalog must not recommend below it either -- the two
    surfaces disagreeing about what "enough context" means is how an agent ends
    up loading a window the server itself would have refused to settle for.
    A thinking model spends its budget reasoning before it answers, so its floor
    is raised to ``models.thinking_default_ctx`` for exactly the reason that
    setting exists.
    """
    models = getattr(config, "models", None)
    floor = int(getattr(models, "default_ctx", 0) or 0)
    if record.capabilities.thinking:
        floor = max(floor, int(getattr(models, "thinking_default_ctx", 0) or 0))
    return floor


def _best(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: int(r["ctx_per_slot"]))


def _live_slots(row: Mapping[str, Any]) -> int:
    return int(row.get("max_parallel") or 0)


def _idle_slots(row: Mapping[str, Any]) -> int:
    return int((row.get("if_gpus_idle") or {}).get("max_parallel") or 0)


def _preferred(
    rows: Sequence[dict[str, Any]],
    *,
    chat_class: bool,
    floor: int,
    slots_of: Callable[[Mapping[str, Any]], int],
) -> tuple[dict[str, Any], str] | None:
    """The three-way preference, applied to one pool of already-usable rows."""
    above = [r for r in rows if int(r["ctx_per_slot"]) >= floor]
    if above:
        if chat_class:
            concurrent = [r for r in above if slots_of(r) >= 2]
            if concurrent:
                return _best(concurrent), "highest ctx >= floor with max_parallel >= 2"
        return _best(above), "highest ctx that fits >= floor"
    if rows:
        return _best(rows), "highest ctx that fits (below floor)"
    return None


def mark_recommended(rows: list[dict[str, Any]], *, chat_class: bool, floor: int = 0) -> str | None:
    """Mark exactly one row ``recommended`` and say which rule chose it.

    ``floor`` is :func:`recommendation_floor` -- the context below which this
    server does not consider a model usefully loaded. **The floor outranks the
    second slot**, which is the one thing this rule got wrong before: it would
    take 16384 tokens with two slots over 32768 with one, and 16k is where an
    OpenClaw agent's tool transcript stops fitting. A queued second
    conversation is a latency problem; a window that cannot hold the task is a
    failed task, and D14 already decided which of those the server optimises
    for. (The old rule picked 16k for the resident 122B for precisely this
    reason, off a knee that was itself wrong -- see D22.)

    In order:

    1. **Chat-class models** (anything an agent talks to): the highest context
       **at or above the floor** that also sustains at least two slots. One slot
       means every concurrent request queues behind the one before it, so above
       the floor the second conversation is worth a context doubling. Non-chat
       models (embeddings, rerankers) skip this: they are called in bursts and a
       single slot is not a bottleneck in the same way.
    2. The highest context at or above the floor that fits at all.
    3. If nothing reaches the floor, the highest context that fits -- said
       plainly, with ``"(below floor)"`` in the basis, because a small window is
       still better than no recommendation.
    4. If nothing fits right now, the same three-way preference applied to the
       ``if_gpus_idle`` column, so the agent is told "unload something" rather
       than "impossible". The basis is ``"if_gpus_idle"``.

    Returns the basis string, or ``None`` when there is nothing to recommend.
    """
    fitting = [r for r in rows if r.get("fits")]
    chosen = _preferred(fitting, chat_class=chat_class, floor=floor, slots_of=_live_slots)
    if chosen is not None:
        chosen[0]["recommended"] = True
        return chosen[1]

    reachable = [r for r in rows if (r.get("if_gpus_idle") or {}).get("fits")]
    chosen = _preferred(reachable, chat_class=chat_class, floor=floor, slots_of=_idle_slots)
    if chosen is not None:
        chosen[0]["recommended"] = True
        return "if_gpus_idle"
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_catalog(
    *,
    registry: Any,
    planner: Planner,
    supervisor: Any = None,
    db: Any = None,
    model: str | None = None,
    compact: bool = False,
    ctx_tiers: Sequence[int] = CTX_TIERS,
    now: float | None = None,
) -> dict[str, Any]:
    """Build the whole catalog: every model, newest download first.

    Args:
        registry: the model registry (``all()`` / ``resolve()``).
        planner: the live planner. Its probe is snapshotted, so every row in
            one catalog describes the same instant.
        supervisor: optional; supplies each model's loaded state and plan.
        db: optional; supplies measured throughput for the speed columns.
        model: build only this model (id or alias).
        compact: drop every non-recommended option row. The recommended row is
            what an agent needs 90% of the time, and the full table is roughly
            seven times the tokens.
        ctx_tiers: override the standard ladder (tests, mostly).
        now: override the clock (tests).

    Never raises for a single bad model: a record whose GGUF metadata could not
    be parsed comes back with an empty ``options`` list and an ``unavailable``
    note, because dropping it silently would make it invisible to the only
    interface that could tell the user it is broken.
    """
    started = time.perf_counter()
    generated_at = now if now is not None else time.time()

    records: list[ModelRecord]
    if model:
        found = registry.resolve(model)
        records = [found] if found is not None else []
    else:
        records = list(registry.all())

    # One VRAM snapshot for the whole build (see _SnapshotProbe).
    live_probe = _SnapshotProbe(planner.probe)
    live = Planner(planner.config, live_probe, log_plans=False)
    idle = Planner(planner.config, _IdleProbe(live_probe), log_plans=False)

    loaded_map: dict[str, Any] = {}
    if supervisor is not None:
        try:
            loaded_map = {i.model_id: i for i in supervisor.list()}
        except Exception:  # noqa: BLE001 - the catalog must survive a sick supervisor
            loaded_map = {}
    loaded_list = list(loaded_map.values())

    gpus = live_probe.list_gpus()
    class_label = throughput.gpu_class(gpus)
    peer_moe = _density_map(registry)

    entries: list[dict[str, Any]] = []
    for record in records:
        entries.append(
            _model_entry(
                record,
                live=live,
                idle=idle,
                loaded=loaded_list,
                instance=loaded_map.get(record.id),
                db=db,
                class_label=class_label,
                peer_moe=peer_moe,
                ctx_tiers=ctx_tiers,
                compact=compact,
            )
        )

    # THE ordering guarantee: most recently downloaded first.
    entries.sort(key=lambda e: e.get("downloaded_at_ts") or 0.0, reverse=True)
    for entry in entries:
        entry.pop("downloaded_at_ts", None)

    return {
        "catalog_hint": CATALOG_HINT,
        "generated_at": _iso(generated_at),
        "build_ms": round((time.perf_counter() - started) * 1000),
        "compact": compact,
        "gpu_class": class_label,
        "gpus": [
            {
                "index": g.index,
                "name": g.name,
                "free_gib": round(g.free_bytes / GB, 2),
                "total_gib": round(g.total_bytes / GB, 2),
            }
            for g in gpus
        ],
        "ctx_tiers": list(ctx_tiers),
        "count": len(entries),
        "models": entries,
    }


def _model_entry(
    record: ModelRecord,
    *,
    live: Planner,
    idle: Planner,
    loaded: Sequence[Any],
    instance: Any,
    db: Any,
    class_label: str,
    peer_moe: Mapping[str, bool],
    ctx_tiers: Sequence[int],
    compact: bool,
) -> dict[str, Any]:
    meta = record.meta
    weights = int(getattr(meta, "tensor_bytes", 0) or 0) or int(record.size_bytes)
    params_total, params_active = _params_b(meta, weights)
    downloaded_ts = float(record.mtime or record.added_at or 0.0)
    moe = is_moe(meta) if meta is not None else False
    attention = attention_kind(meta) if meta is not None else None

    observations: list[Mapping[str, Any]] = []
    if db is not None:
        try:
            observations = list(db.throughput_observations(record.id, limit=100))
        except Exception as exc:  # noqa: BLE001 - measurements are a bonus, not a dependency
            log.debug("catalog observations unavailable", model_id=record.id, error=str(exc))

    all_observations = observations
    if db is not None and not observations:
        # No history for this model: the peer tier needs everyone's.
        try:
            all_observations = list(db.throughput_observations(limit=200))
        except Exception:  # noqa: BLE001
            all_observations = []

    # Calibrated per device set, memoised. The tiers in throughput.calibrate are
    # device-specific and one model's rows are routinely placed differently at
    # 16k and at 262k, so a single per-model factor would attribute a four-way
    # split's measurements to a single-GPU row. Two or three distinct sets per
    # model in practice, so the cache makes this a handful of calls.
    _cache: dict[tuple[int, ...], Mapping[str, Any]] = {}

    def calibrate_for(devices: Sequence[int]) -> Mapping[str, Any]:
        key = tuple(sorted(int(d) for d in devices))
        hit = _cache.get(key)
        if hit is None:
            hit = throughput.calibrate(
                all_observations,
                model_id=record.id,
                devices=key,
                gpu_class_label=class_label,
                is_moe=moe,
                peer_moe=peer_moe,
            )
            _cache[key] = hit
        return hit

    entry: dict[str, Any] = {
        "id": record.id,
        "summary": summarize(record, params_total, params_active, attention),
        "downloaded_at": _iso(downloaded_ts),
        "downloaded_at_ts": downloaded_ts,
        "size_gb": round(record.size_bytes / GB, 2),
        "quantization": record.quant,
        "architecture": record.architecture,
        "params_total_b": params_total,
        "params_active_b": params_active,
        "is_moe": moe,
        # "full" | "iswa" | "hybrid" | "unknown": why this model's context tiers
        # cost what they do. An iSWA or hybrid model keeps only a fraction of the
        # window in KV, which is what lets a 31B offer several slots at 262k.
        "attention_kind": attention,
        "n_ctx_train": int(getattr(meta, "n_ctx_train", 0) or 0) if meta else None,
        "type": model_type(record),
        "kind": record.kind,
        "capabilities": capability_list(record),
        "has_mmproj": record.mmproj_path is not None,
        "publisher": record.publisher,
        # Same vocabulary as /v1/models and /api/v0/models, deliberately.
        "state": "loaded" if instance is not None else "not-loaded",
        "port": getattr(instance, "port", None) if instance is not None else None,
        "loaded_plan": loaded_plan_of(instance),
        "settings_pinned": pinned_settings(record),
        "calibration": _calibration_block(calibrate_for(())),
        "options": [],
    }

    if meta is None:
        entry["unavailable"] = (
            "no parsed GGUF metadata for this model, so no loading options can be "
            "computed; run a model scan or check the file"
        )
        return entry

    rows = [
        _option_row(
            live,
            idle,
            record,
            ctx,
            loaded=loaded,
            observations=observations,
            calibrate_for=calibrate_for,
        )
        for ctx in ctx_tiers_for(record, ctx_tiers)
    ]
    basis = mark_recommended(
        rows,
        chat_class=record.kind == "chat",
        floor=recommendation_floor(live.config, record),
    )
    entry["recommended_basis"] = basis
    # The entry-level block describes the factor the recommended row was quoted
    # with -- the row a caller is being told to take. Reporting a model-wide
    # average instead would name a number no row in the table actually used.
    chosen = next((r for r in rows if r.get("recommended")), None)
    if chosen is not None:
        placed = chosen.get("devices") or (chosen.get("if_gpus_idle") or {}).get("devices") or ()
        entry["calibration"] = _calibration_block(calibrate_for(placed))
    entry["options"] = rows
    return compact_entry(entry) if compact else entry


def _calibration_block(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """The learned factor, as the catalog reports it.

    ``basis`` is one of ``"model+devices"``, ``"model"``, ``"peers"`` or
    ``"none"`` -- the tier the number came from, so a caller can tell "this
    model, measured here" from "other models of the same density on this
    hardware" without a second call.
    """
    return {
        "basis": calibration.get("basis"),
        "gen_factor": round(float(calibration.get("gen", 1.0)), 3),
        "samples": calibration.get("samples", 0),
    }


def _density_map(registry: Any) -> dict[str, bool]:
    """``{model_id: is_moe}`` for the whole library, for the peer calibration tier.

    A sparse-MoE decode and a dense decode miss the roofline by different
    factors, so a peer pool that mixes them teaches every dense model on the box
    the resident MoE's number -- which is exactly what happened here (D22: 84
    MoE rows at 0.411 against 3 dense rows, and every Gemma-4 row read 0.411).
    Built once per catalog build off metadata already in memory.
    """
    try:
        records = list(registry.all())
    except Exception:  # noqa: BLE001 - calibration is a bonus, never a dependency
        return {}
    return {r.id: is_moe(r.meta) for r in records if r.meta is not None}


def loaded_plan_of(instance: Any) -> dict[str, Any] | None:
    """The plan a resident model is actually running under, or ``None``.

    Kept out of the ``state`` field so ``state`` can stay the plain
    ``"loaded"``/``"not-loaded"`` string that ``/api/v0/models`` and
    ``/v1/models`` already use. One word meaning one thing across every
    surface is worth more than nesting the detail inside it.
    """
    if instance is None:
        return None
    plan = getattr(instance, "plan", None)
    if plan is None:
        return None
    return {
        "ctx_per_slot": plan.ctx_per_slot or plan.ctx_size,
        "parallel": plan.parallel,
        "kv_cache_type": plan.kv_cache_type,
        "devices": list(plan.devices),
        "max_parallel": plan.max_parallel,
        "parallel_limited_by": plan.parallel_limited_by,
    }


def compact_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Strip an entry to its recommended row and drop uninformative fields.

    Everything removed is null, empty, or identical to a sibling, so a client
    reading the compact view never sees a *different* answer -- only a shorter
    one, where an absent key means exactly what the omitted value said. That
    matters because the cost is real: a forty-model library at full detail is
    tens of thousands of tokens charged to an agent that asked a one-line
    question.
    """
    out = {
        k: v
        for k, v in entry.items()
        if v is not None and k not in {"calibration", "settings_pinned"}
    }
    if entry.get("calibration", {}).get("basis") not in (None, "none"):
        out["calibration"] = entry["calibration"]
    if entry.get("settings_pinned"):
        out["settings_pinned"] = entry["settings_pinned"]
    if entry.get("params_active_b") == entry.get("params_total_b"):
        out.pop("params_active_b", None)
    out["options"] = [compact_row(r) for r in entry.get("options", []) if r.get("recommended")]
    return out


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """One option row with its nulls and its duplicated idle verdict removed.

    The idle verdict counts as a duplicate when it names the **same set** of
    devices, not the same list. The planner orders devices by its own candidate
    preference, so a busy rig placing a model on ``[1, 0]`` and an idle one
    placing it on ``[0, 1]`` describe the same placement -- and comparing the
    lists shipped that duplicate on every row where the free-VRAM ordering
    happened to differ, which is most of them on a rig with something loaded.
    """
    out = {k: v for k, v in row.items() if v is not None}
    idle = row.get("if_gpus_idle") or {}
    if (
        row.get("fits")
        and idle.get("fits")
        and sorted(idle.get("devices") or []) == sorted(row.get("devices") or [])
    ):
        # The idle machine would place it identically: saying so twice is noise.
        out.pop("if_gpus_idle", None)
    return out


def _iso(ts: float) -> str | None:
    """UTC ISO-8601, or ``None`` for a missing timestamp."""
    if not ts:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
