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
          "attention_kind": "iswa",     # why its long windows are cheap
          "recommended": {              # THE default load: best pair of cards
            "mode": "dual_5090", "label": "2x RTX 5090", "devices": [0, 1],
            "ctx_per_slot": 131072, "kv_cache_type": "f16", "max_parallel": 3,
            "fits_now": true, "would_evict": [],
            "load_args": {"model_id": ..., "ctx_size": 131072,
                          "parallel": 3, "kv_cache_type": "f16",
                          "devices": [0, 1]}},
          "placements": [...],          # the same, per hardware mode
          "options": [                  # the drill-down, one row per ctx tier
            {"ctx_per_slot": 65536, "fits": true, "devices": [0, 1],
             "max_parallel": 4, "est_gen_tps": 58.0, "best_now": true,
             "load_args": {...}}
          ]
        }
      ]
    }

Four design rules, all of them about the consumer being a language model:

**Newest first, always.** The user works from the last thing they downloaded,
so ``downloaded_at`` (the newest mtime across a model's GGUF shards -- a
multi-part download finishes on its *last* file) is the sort key and it sorts
descending. This is the one ordering the catalog guarantees.

**Nothing is left as an exercise.** Every row carries ``load_args``, which is
exactly the argument object the ``load_model`` tool accepts. An agent that has
chosen a row is done choosing: it passes the object through. No field in it
needs to be computed, converted or looked up.

**The headline answer is a placement, not a context size.** The question a
caller has is "which GPUs should this model get", so ``recommended`` names a
set of cards and the settings that are optimal on them **with those cards
idle** -- the user's "assume you can fill them both". What is in the way right
now travels beside it as ``fits_now`` / ``would_evict``, which is the half a
caller can act on. ``placements`` answers the same question for every other
mode this box has; ``options`` remains the per-context drill-down, with
``best_now`` flagging the row that would load on the machine as it stands.

**The planner is the only authority on placement.** The rows do not
re-implement fitting -- they call :meth:`Planner.plan_load` once per context
tier, so exclusions, reservations, quant affinity, per-model device overrides
and *current free VRAM* are all respected by construction. Each ``options`` row
is computed twice: once against the machine as it is now (``fits``/``devices``)
and once against every GPU idle (``if_gpus_idle``), so the agent can tell
"impossible" apart from "possible after unloading something". A model that is
already loaded is judged against a machine credited with its own footprint,
because its rows describe reloading it (D36).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

from studioforge.core import placements as placements_mod
from studioforge.core import throughput
from studioforge.core.kv_sensitivity import (
    allows_q8,
    kv_quality_label,
    kv_quality_rank,
    sensitivity_for,
)
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
    "Models are sorted by download date, newest first. START AT recommended: "
    "the default load for that model -- the optimal settings on this rig's best "
    "pair of GPUs, computed as if those cards were free -- whose load_args go "
    "verbatim to load_model, devices included. Never edit or recompute a "
    "load_args object. placements answers the same question for every hardware "
    "mode of this box, each with its own optimal and a ranking of fastest / "
    "largest_context / cheapest; the compact list gives their settings and "
    "devices but only recommended carries load_args, so call model_options for "
    "the mode you pick. An optimal is computed on IDLE cards, so fits_now says "
    "whether it loads right now, would_evict names (or counts) what is in the "
    "way, and fits_now_ctx is the largest context that does fit on that mode "
    "this second. Settings are chosen QUALITY FIRST: the best KV cache that "
    "reaches this server's context floor, then the largest context at that "
    "quality, then whatever slots fit -- a 4-bit K cache is never chosen "
    "automatically, and a doubled window does not justify a quantized one "
    "(planner.preference 'throughput' restores the older largest-window rule). "
    "options is the per-context drill-down for when you need a different "
    "window, with best_now on the row that fits the machine as it stands. Per "
    "row: ctx_per_slot is the context EACH concurrent conversation gets; fits "
    "is against the VRAM free right now; devices are CUDA indices; vram_mb is "
    "the whole load at this row's max_parallel; parallel_limited_by names what "
    "caps the slot count ('vram', 'knee' = where extra slots stop buying "
    "throughput, or 'cap'); if_gpus_idle repeats the verdict with every GPU "
    "free. max_parallel is how many slots FIT and recommended_parallel how many "
    "are worth running -- what load_args asks for; basis 'measured' means a "
    "parallel benchmark swept it. Speeds are tokens/second: est_gen_tps is ONE "
    "stream at ~8k of "
    "context (a typical turn), est_gen_tps_full_ctx the same stream with the "
    "window nearly full -- the truth is between them -- est_gen_tps_batched the "
    "aggregate across slots, est_prompt_tps ingestion. confidence is 'measured' "
    "(this exact placement was observed), 'calibrated' (a learned factor; "
    "calibration.basis says from what) or 'estimated' (nominal hardware "
    "numbers: an order of magnitude, not a promise). attention_kind explains a "
    "model's context prices: 'iswa' and 'hybrid' keep only a fraction of the "
    "window in KV, which is why their huge contexts stay cheap."
)

#: How long a built catalog stays servable before it is rebuilt. Short, because
#: `fits` is a statement about free VRAM *right now* and a stale yes is worse
#: than a slow no.
CACHE_TTL_S = 20.0

#: Above this, one INFO line naming the cost. The build is pure arithmetic and
#: runs off the event loop behind a 20-second cache, so it is not on any hot
#: path -- but it now plans every model at every context tier on every hardware
#: mode, and that product grows with the library. A number that is drifting
#: upward should be visible in the log before somebody notices it as a stall.
SLOW_BUILD_MS = 1000


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


class CreditedProbe(_SnapshotProbe):
    """The same GPUs, with one resident child's own VRAM handed back as free.

    A loaded model's catalog rows describe *reloading it*, and a reload frees
    the allocation it currently holds before the replacement takes any. Judging
    those rows against a machine that still contains the model is judging them
    against memory the row itself would release, and the answer is visibly
    wrong: on 2026-08-19 the resident 17.4 GB Gemma-4 31B -- running at 262144
    on three cards -- was told by its own catalog that it fitted only at 262144
    with a **q4_0** KV cache spread over three GPUs, because its 17.4 GB of
    weights and 20 GB of KV were counted as somebody else's.

    The credit is :meth:`Planner.instance_footprint`, exactly what D30's
    ``reload_of`` credits back for a forced reload, so the catalog's promise
    and the load's own arithmetic cannot drift apart. Other models' rows are
    computed against the uncredited probe, because *their* load does not free
    anything.
    """

    def __init__(self, inner: Any, footprint: Mapping[int, int]) -> None:
        super().__init__(inner)
        self._gpus = [
            g.model_copy(
                update={
                    "free_bytes": min(g.total_bytes, g.free_bytes + int(footprint.get(g.index, 0))),
                    "used_bytes": max(0, g.used_bytes - int(footprint.get(g.index, 0))),
                }
            )
            for g in self._gpus
        ]


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


def plan_at(
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
    plan = plan_at(idle_planner, record, ctx)
    if not isinstance(plan, LoadPlan):
        return {"fits": False}
    slots, bound, vram = slots_for_plan(idle_planner, record, plan)
    speed = estimate_speed(idle_planner, record, plan, slots, calibrate_for(plan.devices))
    return {
        "fits": True,
        "devices": list(plan.devices),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "vram_mb": round(vram / MB),
        "max_parallel": slots,
        "parallel_limited_by": bound,
        "est_gen_tps": speed["gen_tps"],
        "est_gen_tps_full_ctx": speed["gen_tps_full_ctx"],
        "est_gen_tps_batched": speed["gen_tps_batched"],
    }


def recommended_slots(
    record: ModelRecord,
    plan: LoadPlan,
    max_parallel: int,
    *,
    observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """``recommended_parallel`` for one placement: value, basis and why.

    One helper because three surfaces need the same answer -- the per-context
    ``options`` table, every ``placements[*].optimal``, and
    :meth:`ModelManager.load_recommended` -- and the D36 lesson about
    ``choose_row`` is that a second copy of a rule is a second answer.

    ``max_parallel`` is what the placement can hold (VRAM, plus D17's estimated
    knee, plus the cap). What comes back is how many of those slots are *worth*
    running: the measured knee when a parallel benchmark has swept this model on
    these devices at this context, else the estimate, and ``basis`` says which.
    """
    from studioforge.core import parallel as parallel_mod

    rows = parallel_mod.observations_for(
        observations,
        devices=plan.devices,
        ctx_per_slot=int(plan.ctx_size),
        kv_cache_type=plan.kv_cache_type,
        kv_cache_type_v=plan.kv_cache_type_v,
    )
    return parallel_mod.recommended_parallel(
        record.meta,
        weights_bytes=plan.estimate.weights_bytes or int(record.size_bytes),
        ctx_per_slot=int(plan.ctx_size),
        kv_cache_type=plan.kv_cache_type,
        kv_cache_type_v=plan.kv_cache_type_v,
        max_parallel=int(max_parallel),
        observations=rows,
    )


def load_args_for(record: ModelRecord, plan: LoadPlan, slots: int) -> dict[str, Any]:
    """The exact argument object ``load_model`` accepts for this placement.

    ``slots`` is ``recommended_parallel``, not ``max_parallel``: the recipe a
    caller passes verbatim should ask for the number of slots that is worth
    running, not the number the VRAM would tolerate. The larger figure stays
    visible on the row beside it, because "this placement could hold eight" is
    a real fact about the hardware and an operator with eight agents may want
    it -- it is just not the default.

    ``kv_cache_type_v`` appears **only when it differs from K**: the tool
    defaults V to K, so emitting an equal pair would cost every caller tokens
    to be told the same thing twice. It does differ on the ladder's third rung
    (``q8_0`` K with a ``q4_0`` V), which is the rung that exists precisely
    because K and V are not equally sensitive to quantization.
    """
    args: dict[str, Any] = {
        "model_id": record.id,
        "ctx_size": int(plan.ctx_size),
        "parallel": int(slots),
        "kv_cache_type": plan.kv_cache_type,
    }
    if plan.kv_cache_type_v != plan.kv_cache_type:
        args["kv_cache_type_v"] = plan.kv_cache_type_v
    return args


def _option_row(
    planner: Planner,
    idle_planner: Planner,
    record: ModelRecord,
    ctx: int,
    *,
    loaded: Sequence[Any],
    observations: Sequence[Mapping[str, Any]],
    parallel_observations: Sequence[Mapping[str, Any]] = (),
    calibrate_for: Calibrator,
) -> dict[str, Any]:
    """One (model, context) row of the catalog."""
    idle = _idle_variant(idle_planner, record, ctx, calibrate_for)
    plan = plan_at(planner, record, ctx, loaded=loaded)

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
            "recommended_parallel": 0,
            "recommended_parallel_basis": None,
            "est_prompt_tps": None,
            "est_gen_tps": None,
            "est_gen_tps_full_ctx": None,
            "est_gen_tps_batched": None,
            "measured_gen_tps": None,
            "measured_prompt_tps": None,
            "confidence": "estimated",
            "if_gpus_idle": idle,
            "load_args": None,
            "best_now": False,
        }

    slots, bound, vram = slots_for_plan(planner, record, plan)
    recommended = recommended_slots(record, plan, slots, observations=parallel_observations)
    calibration = calibrate_for(plan.devices)
    speed = estimate_speed(planner, record, plan, slots, calibration)
    measured = throughput.measured_for(
        observations, devices=plan.devices, ctx_size=ctx, parallel=recommended["value"]
    )
    return {
        "ctx_per_slot": ctx,
        "fits": True,
        "devices": list(plan.devices),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "vram_mb": round(vram / MB),
        "max_parallel": slots,
        "parallel_limited_by": bound,
        # How many slots are worth running, versus how many fit (WP19/D37).
        "recommended_parallel": recommended["value"],
        "recommended_parallel_basis": recommended["basis"],
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
        "load_args": load_args_for(record, plan, recommended["value"]),
        "best_now": False,
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


class _RowView(NamedTuple):
    """The four facts a chooser compares, read off whichever half of a row holds them."""

    ctx: int
    kv_k: str
    kv_v: str
    slots: int


#: A row carries two verdicts -- the live one at the top level and the idle one
#: under ``if_gpus_idle`` -- and the chooser must be able to rank either without
#: knowing which. Passing an accessor rather than duplicating the rule is what
#: keeps "highest ctx at the best KV quality" a single implementation.
RowView = Callable[[Mapping[str, Any]], _RowView]


def live_view(row: Mapping[str, Any]) -> _RowView:
    kv_k = str(row.get("kv_cache_type") or "")
    return _RowView(
        int(row.get("ctx_per_slot") or 0),
        kv_k,
        str(row.get("kv_cache_type_v") or kv_k),
        int(row.get("max_parallel") or 0),
    )


def idle_view(row: Mapping[str, Any]) -> _RowView:
    idle = row.get("if_gpus_idle") or {}
    kv_k = str(idle.get("kv_cache_type") or "")
    return _RowView(
        int(row.get("ctx_per_slot") or 0),
        kv_k,
        str(idle.get("kv_cache_type_v") or kv_k),
        int(idle.get("max_parallel") or 0),
    )


def _throughput_preferred(
    rows: Sequence[dict[str, Any]],
    *,
    chat_class: bool,
    floor: int,
    view: RowView,
) -> tuple[dict[str, Any], str] | None:
    """D20's rule: the largest window, preferring one that also serves two slots.

    Kept, and reachable through ``planner.preference: "throughput"``, because
    it is the right answer for a host whose job is to serve many short
    conversations: there, a doubled window nobody fills is worth less than a
    second slot, and the KV cache quality that buys the window is a cost the
    operator has decided to pay. It is no longer the *default* (D36).
    """
    above = [r for r in rows if view(r).ctx >= floor]
    if above:
        if chat_class:
            concurrent = [r for r in above if view(r).slots >= 2]
            if concurrent:
                return _best(concurrent), "highest ctx >= floor with max_parallel >= 2"
        return _best(above), "highest ctx that fits >= floor"
    if rows:
        return _best(rows), "highest ctx that fits (below floor)"
    return None


def _quality_preferred(
    rows: Sequence[dict[str, Any]],
    *,
    floor: int,
    meta: Any,
    view: RowView,
) -> tuple[dict[str, Any], str] | None:
    """Quality first, then context, then slots (D36) -- the default rule.

    The order is the user's, stated plainly: *"quality is more important than
    speed"*. So:

    1. **The best KV cache quality that reaches the floor** at one slot or
       more. Rows are ranked by
       :func:`~studioforge.core.kv_sensitivity.kv_quality_rank`, which puts
       ``f16`` ahead of ``q8_0`` ahead of ``q8_0`` K + ``q4_0`` V. Quantizing
       to reach a *bigger* window is not a trade this rule makes; quantizing to
       reach the floor at all is, because the alternative is not serving the
       model.
    2. **The highest context at that quality** (already capped at
       ``n_ctx_train`` by :func:`ctx_tiers_for`).
    3. **Whatever slots that placement sustains** -- reported, never bought.
       A slot count is a latency property; a KV cache type is a correctness
       one, and D14/D22 already settled that the floor outranks the second slot.

    **One exception, and it is measured.** A family whose KV cache is *known*
    to survive quantization (:mod:`studioforge.core.kv_sensitivity`: Qwen 3.6
    measures KL 0.024 at ``q8_0``, inside sampler noise) may take ``q8_0`` when
    it buys at least a **doubling** of the window. A sensitive family (Gemma
    3/4 at KL 0.108 dense / 0.377 MoE, and every unmeasured architecture) never
    does.

    Below the floor the same order applies to whatever does fit, with
    ``"(below floor)"`` said out loud -- a small window beats no answer.
    """
    usable = [r for r in rows if view(r).slots >= 1]
    above = [r for r in usable if view(r).ctx >= floor]
    pool = above or usable
    if not pool:
        return None

    by_rank: dict[int, list[dict[str, Any]]] = {}
    for row in pool:
        seen = view(row)
        by_rank.setdefault(kv_quality_rank(seen.kv_k, seen.kv_v), []).append(row)
    rank = min(by_rank)

    if rank == 0 and 1 in by_rank:
        f16_ctx = max(view(r).ctx for r in by_rank[0])
        q8_ctx = max(view(r).ctx for r in by_rank[1])
        if q8_ctx >= 2 * f16_ctx and allows_q8(
            meta, f16_reaches_floor=bool(above), buys_doubling=True
        ):
            rank = 1

    chosen = max(by_rank[rank], key=lambda r: (view(r).ctx, view(r).slots))
    seen = view(chosen)
    basis = (
        f"{kv_quality_label(seen.kv_k, seen.kv_v)} KV, highest ctx "
        f"{seen.ctx}, {seen.slots} slot{'' if seen.slots == 1 else 's'}"
    )
    if not above:
        basis += " (below floor)"
    return chosen, basis


def choose_row(
    rows: Sequence[dict[str, Any]],
    *,
    chat_class: bool,
    floor: int,
    meta: Any = None,
    preference: str = "quality",
    view: RowView = live_view,
) -> tuple[dict[str, Any], str] | None:
    """The ONE chooser, used by the per-context table and by every placement.

    Both surfaces answer the same question -- "of these rows, which is the one
    to load?" -- and answering it twice is how ``/profiles`` came to recommend
    a 262144-token q4_0 row while the catalog recommended something else for
    the same model on the same hardware.
    """
    if preference == "throughput":
        return _throughput_preferred(rows, chat_class=chat_class, floor=floor, view=view)
    return _quality_preferred(rows, floor=floor, meta=meta, view=view)


def mark_best_now(
    rows: list[dict[str, Any]],
    *,
    chat_class: bool,
    floor: int = 0,
    meta: Any = None,
    preference: str = "quality",
) -> str | None:
    """Flag the one row of the per-context table that is best **right now**.

    The entry-level ``recommended`` is a *placement* (D36): the best this model
    can do on the rig's headline hardware mode, computed as if that hardware
    were free, because "which GPUs should I give this model" is the question a
    caller actually has. This flag is the drill-down's own answer to a narrower
    question -- given the machine exactly as it is this second, which context
    row would load without disturbing anything -- and it is what the compact
    view keeps.

    Falls back to the ``if_gpus_idle`` column when nothing fits now, so the
    answer is "unload something" rather than "impossible"; the basis then says
    ``if_gpus_idle``.
    """
    fitting = [r for r in rows if r.get("fits")]
    chosen = choose_row(
        fitting,
        chat_class=chat_class,
        floor=floor,
        meta=meta,
        preference=preference,
        view=live_view,
    )
    if chosen is not None:
        chosen[0]["best_now"] = True
        return chosen[1]

    reachable = [r for r in rows if (r.get("if_gpus_idle") or {}).get("fits")]
    chosen = choose_row(
        reachable,
        chat_class=chat_class,
        floor=floor,
        meta=meta,
        preference=preference,
        view=idle_view,
    )
    if chosen is not None:
        chosen[0]["best_now"] = True
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
    # Derived once: the mode list is a property of the box, not of a model, and
    # deriving it per model would ask the probe thirty-three times for an answer
    # that cannot have changed inside one snapshot. Excluded cards are not part
    # of any mode (D19): a mode is planned through a device override, which is
    # the one thing that beats an exclusion.
    modes = placements_mod.hardware_modes(gpus, excluded=planner.config.planner.excluded_devices)

    entries: list[dict[str, Any]] = []
    for record in records:
        entries.append(
            _model_entry(
                record,
                live=live,
                idle=idle,
                live_probe=live_probe,
                loaded=loaded_list,
                instance=loaded_map.get(record.id),
                db=db,
                class_label=class_label,
                peer_moe=peer_moe,
                ctx_tiers=ctx_tiers,
                compact=compact,
                modes=modes,
            )
        )

    # THE ordering guarantee: most recently downloaded first.
    entries.sort(key=lambda e: e.get("downloaded_at_ts") or 0.0, reverse=True)
    for entry in entries:
        entry.pop("downloaded_at_ts", None)

    build_ms = round((time.perf_counter() - started) * 1000)
    if build_ms > SLOW_BUILD_MS:
        # Not an error -- the result is cached for CACHE_TTL_S and built off the
        # event loop -- but a build that keeps growing is how a "cheap" call
        # becomes a stall, and the number belongs in the log before it does.
        log.info(
            "catalog build is slow",
            build_ms=build_ms,
            models=len(entries),
            modes=len(modes),
            detail="each model is planned per context tier and per hardware mode",
        )

    return {
        "catalog_hint": CATALOG_HINT,
        "generated_at": _iso(generated_at),
        "build_ms": build_ms,
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
    live_probe: Any,
    loaded: Sequence[Any],
    instance: Any,
    db: Any,
    class_label: str,
    peer_moe: Mapping[str, bool],
    ctx_tiers: Sequence[int],
    compact: bool,
    modes: Sequence[placements_mod.HardwareMode],
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

    # Parallel sweeps are per model and rare (one run writes four rows), so the
    # whole history for this model is a handful of rows and there is no peer
    # tier: a slot knee belongs to one model's weights-to-KV ratio on one set of
    # cards, and borrowing another model's would be inventing a measurement.
    parallel_observations: list[Mapping[str, Any]] = []
    if db is not None:
        try:
            parallel_observations = list(db.parallel_observations(record.id, limit=64))
        except Exception as exc:  # noqa: BLE001 - measurements are a bonus, not a dependency
            log.debug(
                "catalog parallel observations unavailable", model_id=record.id, error=str(exc)
            )

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

    # A resident model's rows describe RELOADING it, so its own allocation is
    # credited back and it is not one of the obstacles (D36 / see
    # :class:`CreditedProbe`). Every other model's rows see the machine as it
    # is, because their load frees nothing.
    entry_live, entry_loaded = _live_view_for(record, live, live_probe, loaded, instance)

    rows = [
        _option_row(
            entry_live,
            idle,
            record,
            ctx,
            loaded=entry_loaded,
            observations=observations,
            parallel_observations=parallel_observations,
            calibrate_for=calibrate_for,
        )
        for ctx in ctx_tiers_for(record, ctx_tiers)
    ]
    floor = recommendation_floor(entry_live.config, record)
    preference = str(getattr(entry_live.config.planner, "preference", "quality"))
    basis = mark_best_now(
        rows,
        chat_class=record.kind == "chat",
        floor=floor,
        meta=meta,
        preference=preference,
    )
    entry["best_now_basis"] = basis
    # The entry-level block describes the factor the chosen row was quoted
    # with -- the row a caller is being told to take. Reporting a model-wide
    # average instead would name a number no row in the table actually used.
    chosen = next((r for r in rows if r.get("best_now")), None)
    if chosen is not None:
        placed = chosen.get("devices") or (chosen.get("if_gpus_idle") or {}).get("devices") or ()
        entry["calibration"] = _calibration_block(calibrate_for(placed))
    entry["options"] = rows

    # -- what this model does on each set of cards, and the default load ----
    entry["placements"] = placements_mod.placement_report(
        record,
        planner=live,
        live_planner=entry_live,
        probe=live_probe,
        loaded=entry_loaded,
        ctx_tiers=ctx_tiers,
        floor=floor,
        preference=preference,
        calibrate_for=calibrate_for,
        modes=modes,
        parallel_observations=parallel_observations,
    )
    _apply_recommendation(entry, record)
    notes = quality_notes(record)
    if notes:
        entry["quality_notes"] = notes
    return compact_entry(entry) if compact else entry


def _apply_recommendation(entry: dict[str, Any], record: ModelRecord) -> None:
    """Promote the headline placement to the entry's ``recommended`` load.

    ``placements[0]`` is the two best cards of this box, computed as if they
    were empty -- "assume you can fill them both". That is the load a caller
    should take by default, so it is stated once at the top of the entry with
    its own ``fits_now`` / ``would_evict`` beside it rather than left for the
    caller to dig out of a list. A mode that cannot hold the model at all is
    skipped, so ``recommended`` is the best mode that *can*.
    """
    usable = [p for p in entry.get("placements", []) if p.get("optimal")]
    if not usable:
        entry["recommended"] = None
        entry["recommended_basis"] = None
        return
    head = usable[0]
    optimal = head["optimal"]
    entry["recommended"] = {
        "mode": head["mode"],
        "label": head["label"],
        "devices": list(head["devices"]),
        "ctx_per_slot": optimal["ctx_per_slot"],
        "kv_cache_type": optimal["kv_cache_type"],
        "kv_cache_type_v": optimal["kv_cache_type_v"],
        "max_parallel": optimal["max_parallel"],
        "recommended_parallel": optimal.get("recommended_parallel", optimal["max_parallel"]),
        "recommended_parallel_basis": optimal.get("recommended_parallel_basis"),
        "est_gen_tps": optimal["est_gen_tps"],
        "est_gen_tps_full_ctx": optimal["est_gen_tps_full_ctx"],
        "vram_mb": optimal["vram_mb"],
        "fits_now": head["fits_now"],
        "would_evict": list(head["would_evict"]),
        "load_args": optimal["load_args"],
    }
    entry["recommended_basis"] = f"{head['label']}: {head['basis']}"


def quality_notes(record: ModelRecord) -> list[str]:
    """Warnings about *saved settings* that cap this model's quality.

    Only one so far, and it is aimed at a real pair of records on this rig: the
    two Gemma-4 QAT models pin ``kv_cache_type: q8_0``, and Gemma is precisely
    the family that measurably minds (KL 0.108 dense / 0.377 MoE against f16 --
    :mod:`studioforge.core.kv_sensitivity`). The planner honours an explicit
    setting verbatim and must go on doing so, so the only correct action here
    is to *say* what it costs and where to clear it. The placements carry an
    ``if_unpinned`` block beside the pinned one so the size of the choice is
    visible rather than asserted.
    """
    pinned = record.settings.kv_cache_type
    if pinned is None or pinned in ("auto", "f16"):
        return []
    sensitivity = sensitivity_for(record.meta)
    if not sensitivity.sensitive:
        return []
    measured = (
        f" ({sensitivity.family}: KL {sensitivity.kl_q8} at q8_0 vs f16)"
        if sensitivity.kl_q8 is not None
        else ""
    )
    return [
        f"pinned kv_cache_type {pinned} caps quality on a KV-sensitive family"
        f"{measured}; clear it in Models -> settings to let the planner use f16"
    ]


def _live_view_for(
    record: ModelRecord,
    live: Planner,
    live_probe: Any,
    loaded: Sequence[Any],
    instance: Any,
) -> tuple[Planner, list[Any]]:
    """``(planner, other_loaded)`` this model's live rows should be judged against.

    For a model that is **not** loaded this is the shared live planner and the
    whole loaded list. For a model that **is** loaded it is a planner over a
    :class:`CreditedProbe` -- its own footprint handed back as free VRAM --
    with itself removed from the obstacles, which is precisely the view D30's
    ``reload_of`` takes when the same reload is actually performed. Both halves
    matter: the credited probe is what ``fits`` is decided against, and it is
    also what :func:`slots_for_plan` measures capacity against, so the slot
    column is credited too rather than only the fit verdict.
    """
    if instance is None:
        return live, list(loaded)
    others = [i for i in loaded if i.model_id != record.id]
    footprint = Planner.instance_footprint(instance)
    if not footprint:
        # A child with no plan (an adopted instance, a fake in a test): there is
        # nothing to credit, and inventing a figure would be worse than none.
        return live, others
    credited = Planner(live.config, CreditedProbe(live_probe, footprint), log_plans=False)
    return credited, others


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
    """Strip an entry to its headline answers and drop uninformative fields.

    Everything removed is null, empty, or identical to a sibling, so a client
    reading the compact view never sees a *different* answer -- only a shorter
    one, where an absent key means exactly what the omitted value said. That
    matters because the cost is real: a forty-model library at full detail is
    tens of thousands of tokens charged to an agent that asked a one-line
    question.

    What survives is the two things a caller acts on: ``recommended`` (the
    default load, on the best pair of cards) and ``placements`` (the same
    question answered for every other set of cards), plus the single option row
    that is best on this machine right now.
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
    out["options"] = [compact_row(r) for r in entry.get("options", []) if r.get("best_now")]
    if entry.get("placements"):
        out["placements"] = [compact_placement(p) for p in entry["placements"]]
    if isinstance(out.get("recommended"), dict):
        out["recommended"] = _compact_recommended(out["recommended"])
    return out


def _compact_recommended(recommended: Mapping[str, Any]) -> dict[str, Any]:
    """The headline load with its two uninformative concurrency fields dropped.

    Same rule as everywhere else in the compact view: a key goes when its value
    is null, empty, or identical to a sibling. ``recommended_parallel`` equals
    ``max_parallel`` until a parallel benchmark has run (D17's knee is already
    folded into ``max_parallel``), and its basis is ``"estimated"`` until then
    -- so on a rig that has measured nothing this pays two keys per model to say
    what the row already said. ``load_args.parallel`` carries the number a
    caller acts on either way, so nothing is lost.
    """
    out = dict(recommended)
    if out.get("recommended_parallel") == out.get("max_parallel"):
        out.pop("recommended_parallel", None)
    if out.get("recommended_parallel_basis") != "measured":
        out.pop("recommended_parallel_basis", None)
    return out


def compact_placement(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One hardware mode, at the size a list of forty models can afford.

    ``would_evict`` collapses from the model ids to a count -- "two models are
    in the way" is the decision, and *which* two is a ``model_options`` call
    away. ``est_prompt_tps``, ``parallel_limited_by``, ``fits_now_ctx`` and the
    ``if_unpinned`` comparison go for the same reason: they inform a choice
    between modes that this view has already framed.

    **``load_args`` goes too, and only here.** The catalog's rule is that a
    chosen row needs no assembly, and the rule is kept where it is used: the
    entry's ``recommended`` -- the mode a caller takes by default -- carries a
    complete, verbatim-passable object. Repeating one per mode costs about 40%
    of a compact entry to describe four loads of which at most one happens, on
    a payload a caller asks for twenty-five models at a time. The alternatives
    keep ``mode``, ``devices`` and their settings, and ``model_options`` returns
    their ``load_args`` in full for the one mode that is actually chosen.
    """
    optimal = entry.get("optimal")
    out: dict[str, Any] = {
        "mode": entry["mode"],
        "label": entry["label"],
        "devices": list(entry["devices"]),
        "fits_now": bool(entry.get("fits_now")),
        "would_evict": len(entry.get("would_evict") or []),
    }
    if entry.get("ranking"):
        out["ranking"] = list(entry["ranking"])
    if optimal is None:
        out["optimal"] = None
        return out
    out["optimal"] = {
        "ctx_per_slot": optimal["ctx_per_slot"],
        "kv_cache_type": optimal["kv_cache_type"],
        "max_parallel": optimal["max_parallel"],
        "est_gen_tps": optimal["est_gen_tps"],
        "est_gen_tps_full_ctx": optimal["est_gen_tps_full_ctx"],
        "vram_mb": optimal["vram_mb"],
    }
    if optimal["kv_cache_type_v"] != optimal["kv_cache_type"]:
        out["optimal"]["kv_cache_type_v"] = optimal["kv_cache_type_v"]
    # Same rule as kv_cache_type_v: say it only when it says something. With no
    # parallel benchmark the recommendation lands on max_parallel (D17's knee is
    # already folded into that number), and repeating it per mode per model
    # would be paying tokens to be told the same thing twice. When a run has
    # been recorded the two diverge, and then the basis is worth the bytes too.
    if optimal.get("recommended_parallel") not in (None, optimal["max_parallel"]):
        out["optimal"]["recommended_parallel"] = optimal["recommended_parallel"]
    if optimal.get("recommended_parallel_basis") == "measured":
        out["optimal"]["recommended_parallel_basis"] = "measured"
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
