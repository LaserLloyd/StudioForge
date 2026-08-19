"""Hardware modes: what this model can do on each *set of cards* in this box.

The question a caller actually has is not "what does this model do at 65536
tokens" but **"which GPUs should I give it, and what do I get"** -- the user's
words were "these should be the optimal settings to run on either the 5090s,
the 3090s, a single 5090, or all". Three things were wrong with the answer this
server used to give.

*The modes were hard-coded.* ``ModelManager.PLACEMENT_MODES`` listed
``(0, 1)``, ``(2, 3)`` and "all" as literals, so a box with different hardware
got labels that were simply false, and the single-5090 mode the user asks about
did not exist at all. :func:`hardware_modes` derives them from the inventory.

*They were judged against a busy machine.* ``/profiles`` planned each mode
against live free VRAM, so "what can this model do on the two 5090s" was
answered as "...given whatever is on them this second", which is not the
question. Every mode here is computed against **that mode's cards, idle** --
"assume you can fill them both" -- and what stands in the way *right now* is
reported separately as ``fits_now`` / ``would_evict``, which is the actionable
half.

*And they used a second recommendation rule.* ``/profiles`` asked the planner
for the largest context that fits, so it answered ``262144`` on a q4_0 cache
while ``/api/catalog`` answered something else for the same model on the same
hardware. Both now call :func:`studioforge.core.catalog.choose_row`.

Mode order is deliberate and is the user's priority: the two best cards first
(the fastest thing this box can do while leaving the rest free), then the two
second-best, then everything, then one card. ``placements[0]`` is therefore the
default a caller should take, and it is what the catalog reports as the
model's ``recommended`` load.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studioforge.core import throughput
from studioforge.core.planner import Planner
from studioforge.logging import get_logger
from studioforge.types import GpuInfo, LoadPlan, ModelRecord

log = get_logger(__name__)


@dataclass(frozen=True)
class HardwareMode:
    """One set of cards worth asking about, named the way a person would."""

    #: Stable identifier: ``single_5090``, ``dual_5090``, ``dual_3090``,
    #: ``all_gpus``. Derived from the card's performance class rather than from
    #: CUDA indices, which move when a card is added or the driver reorders
    #: them -- the same reasoning D22 applied to calibration's ``gpu_class``.
    key: str
    #: ``"2x RTX 5090"`` / ``"all 4 GPUs (2x RTX 5090 + 2x RTX 3090)"``.
    label: str
    devices: tuple[int, ...]


def _class_of(gpu: GpuInfo) -> throughput.GpuPerf:
    return throughput.gpu_perf_for(gpu.name, int(gpu.total_bytes or 0))


def _slug(label: str) -> str:
    """``"RTX 5090"`` -> ``"5090"``; ``"unknown GPU"`` -> ``"unknown"``.

    The distinguishing token is the one carrying digits: every consumer card in
    the performance table is named by its number, and "rtx" tells a caller
    nothing that "5090" does not.
    """
    for token in reversed(label.split()):
        if any(ch.isdigit() for ch in token):
            return token.lower()
    return label.split()[0].lower() if label.split() else "gpu"


def _mix(gpus: Sequence[GpuInfo]) -> str:
    counts: dict[str, int] = {}
    for gpu in gpus:
        name = _class_of(gpu).label
        counts[name] = counts.get(name, 0) + 1
    return " + ".join(f"{n}x {label}" for label, n in counts.items())


def hardware_modes(gpus: Sequence[GpuInfo]) -> list[HardwareMode]:
    """The sets of cards worth reporting on, best-and-fewest first.

    Order: ``dual_<best class>``, ``dual_<second class>``, ``all_gpus``,
    ``single_<best class>``. The pair of best cards leads because it is what
    the user asks for by default -- the fastest placement that still leaves the
    rest of the rig free for a second model -- and because a pair is where a
    31B's context stops being cramped.

    Deduplicated on the device set, keeping the first (and therefore
    best-named) description: on a two-card box "2x RTX 5090" and "all 2 GPUs"
    are the same placement, and the first says more.
    """
    usable = list(gpus)
    if not usable:
        return []

    by_class: dict[str, list[GpuInfo]] = {}
    for gpu in sorted(usable, key=lambda g: g.index):
        by_class.setdefault(_class_of(gpu).label, []).append(gpu)
    ranked = sorted(
        by_class.items(),
        key=lambda kv: (-_class_of(kv[1][0]).bw_bytes_per_s, kv[0]),
    )

    candidates: list[HardwareMode] = []
    for label, members in ranked[:2]:
        if len(members) >= 2:
            pair = members[:2]
            candidates.append(
                HardwareMode(
                    key=f"dual_{_slug(label)}",
                    label=f"2x {label}",
                    devices=tuple(g.index for g in pair),
                )
            )
    if len(usable) >= 2:
        every = sorted(usable, key=lambda g: g.index)
        candidates.append(
            HardwareMode(
                key="all_gpus",
                label=f"all {len(every)} GPUs ({_mix(every)})",
                devices=tuple(g.index for g in every),
            )
        )
    best_label, best_members = ranked[0]
    candidates.append(
        HardwareMode(
            key=f"single_{_slug(best_label)}",
            label=f"1x {best_label}",
            devices=(best_members[0].index,),
        )
    )

    seen: set[tuple[int, ...]] = set()
    modes: list[HardwareMode] = []
    for mode in candidates:
        if mode.devices in seen:
            continue
        seen.add(mode.devices)
        modes.append(mode)
    return modes


class _ModeProbe:
    """The mode's cards, reported idle; every other card hidden.

    Idle because the question is "what can this model do on these cards", not
    "...given what is on them this second" -- ``planner.headroom_fraction``,
    ``reserved_mb`` and ``excluded_devices`` still apply, because those describe
    memory that is never ours whatever is loaded. Hidden rather than merely
    unpreferred because a mode that could quietly spill onto a third card would
    not be the mode it claims to be.
    """

    backend = "mode"

    def __init__(self, inner: Any, devices: Sequence[int]) -> None:
        wanted = {int(d) for d in devices}
        self._gpus = [
            g.model_copy(update={"free_bytes": g.total_bytes, "used_bytes": 0})
            for g in inner.list_gpus()
            if g.index in wanted
        ]

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [g.model_copy(deep=True) for g in self._gpus]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return next((g.model_copy(deep=True) for g in self._gpus if g.index == index), None)

    def compute_processes(self) -> list[Any]:
        return []

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


def forced_onto(record: ModelRecord, devices: Sequence[int]) -> ModelRecord:
    """A copy of the record pinned to ``devices``.

    A **copy**: the old ``/profiles`` rewrote ``record.settings`` in place and
    restored it in a ``finally``, which is a saved-settings corruption waiting
    for an exception -- and did in fact race the benchmark, which does the same
    thing to the same object.
    """
    return record.model_copy(
        update={
            "settings": record.settings.model_copy(
                update={"device_override": [int(d) for d in devices]}
            )
        }
    )


def unpinned_kv(record: ModelRecord) -> ModelRecord:
    """A copy of the record with any pinned KV cache type cleared.

    Used to show what a model would reach if its saved ``kv_cache_type`` were
    not capping it -- the two Gemma-4 QAT records on this rig pin ``q8_0``, and
    Gemma is the family that measurably minds (see
    :mod:`studioforge.core.kv_sensitivity`). The saved settings are never
    changed here; the point is to show the user the size of the choice.
    """
    return record.model_copy(
        update={
            "settings": record.settings.model_copy(
                update={"kv_cache_type": None, "kv_cache_type_v": None}
            )
        }
    )


def _mode_row(
    planner: Planner,
    record: ModelRecord,
    ctx: int,
    calibrate_for: Any,
    parallel_observations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """One context tier on one mode's idle cards, or ``None`` if it does not fit."""
    from studioforge.core import catalog as catalog_mod

    plan = catalog_mod.plan_at(planner, record, ctx)
    if not isinstance(plan, LoadPlan):
        return None
    slots, bound, vram = catalog_mod.slots_for_plan(planner, record, plan)
    recommended = catalog_mod.recommended_slots(
        record, plan, slots, observations=parallel_observations
    )
    speed = catalog_mod.estimate_speed(planner, record, plan, slots, calibrate_for(plan.devices))
    row: dict[str, Any] = {
        "ctx_per_slot": int(plan.ctx_size),
        "fits": True,
        "devices": list(plan.devices),
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "vram_mb": round(vram / (1024**2)),
        "max_parallel": slots,
        "parallel_limited_by": bound,
        # How many of those slots are worth running (WP19/D37). ``max_parallel``
        # is capacity plus D17's estimated knee; this is the number a load
        # should ask for, and it is the number ``load_args`` carries.
        "recommended_parallel": recommended["value"],
        "recommended_parallel_basis": recommended["basis"],
        "recommended_parallel_detail": recommended["detail"],
        "est_gen_tps": speed["gen_tps"],
        "est_gen_tps_full_ctx": speed["gen_tps_full_ctx"],
        "est_prompt_tps": speed["prompt_tps"],
        "load_args": catalog_mod.load_args_for(record, plan, recommended["value"]),
    }
    row["load_args"]["devices"] = list(plan.devices)
    return row


def _optimal_for(
    record: ModelRecord,
    mode: HardwareMode,
    *,
    planner: Planner,
    probe: Any,
    ctx_tiers: Sequence[int],
    floor: int,
    preference: str,
    calibrate_for: Any,
    parallel_observations: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any] | None, str | None]:
    """The best row this mode can reach with its cards idle, and why it won."""
    from studioforge.core import catalog as catalog_mod

    mode_planner = Planner(planner.config, _ModeProbe(probe, mode.devices), log_plans=False)
    pinned = forced_onto(record, mode.devices)
    rows = [
        row
        for ctx in catalog_mod.ctx_tiers_for(record, ctx_tiers)
        if (row := _mode_row(mode_planner, pinned, ctx, calibrate_for, parallel_observations))
        is not None
    ]
    if not rows:
        return None, None
    chosen = catalog_mod.choose_row(
        rows,
        chat_class=record.kind == "chat",
        floor=floor,
        meta=record.meta,
        preference=preference,
    )
    if chosen is None:
        return None, None
    return chosen[0], chosen[1]


def placement_report(
    record: ModelRecord,
    *,
    planner: Planner,
    live_planner: Planner | None = None,
    probe: Any = None,
    loaded: Sequence[Any] = (),
    ctx_tiers: Sequence[int] = (),
    floor: int = 0,
    preference: str = "quality",
    calibrate_for: Any = None,
    modes: Sequence[HardwareMode] | None = None,
    parallel_observations: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """One entry per hardware mode: the optimal load, and what blocks it now.

    Args:
        record: the model.
        planner: supplies ``config`` and, when ``probe`` is omitted, the GPU
            inventory the modes are derived from.
        live_planner: the planner ``fits_now`` / ``would_evict`` are decided
            against -- the machine as it really is, credited for this model's
            own footprint when it is the one already loaded (D36). Defaults to
            ``planner``.
        probe: the snapshot every mode's idle view is built from.
        loaded: running instances, for the eviction question.
        ctx_tiers: the context ladder to consider.
        floor: :func:`studioforge.core.catalog.recommendation_floor`.
        preference: ``"quality"`` or ``"throughput"`` (``planner.preference``).
        calibrate_for: ``devices -> calibration`` (the catalog's memoised one).
        parallel_observations: rows from ``db.parallel_observations`` for this
            model. Each mode picks out the ones taken on **its** cards at the
            context it chose (:func:`studioforge.core.parallel.observations_for`)
            and turns them into ``optimal.recommended_parallel``; without any,
            that number is D17's estimated knee and says so.

    Each entry carries ``optimal`` (idle, this mode's cards), ``fits_now``
    (would ``optimal.load_args`` load right now without disturbing anything),
    ``would_evict`` (the model ids it would have to stop if allowed to),
    ``fits_now_ctx`` (the largest tier that does fit right now on this mode, or
    ``None``) and ``ranking`` -- ``fastest`` / ``largest_context`` /
    ``cheapest``, so an agent can pick on one axis without re-deriving it.
    """
    from studioforge.core import catalog as catalog_mod

    probe = probe if probe is not None else planner.probe
    live = live_planner if live_planner is not None else planner
    if calibrate_for is None:

        def calibrate_for(_devices: Sequence[int]) -> Mapping[str, Any]:
            return {}

    tiers = tuple(ctx_tiers) or catalog_mod.CTX_TIERS
    all_modes = list(modes) if modes is not None else hardware_modes(probe.list_gpus())
    pinned_kv = record.settings.kv_cache_type is not None

    entries: list[dict[str, Any]] = []
    for mode in all_modes:
        optimal, basis = _optimal_for(
            record,
            mode,
            planner=planner,
            probe=probe,
            ctx_tiers=tiers,
            floor=floor,
            preference=preference,
            calibrate_for=calibrate_for,
            parallel_observations=parallel_observations,
        )
        entry: dict[str, Any] = {
            "mode": mode.key,
            "label": mode.label,
            "devices": list(mode.devices),
            "optimal": optimal,
            "basis": basis,
            "fits_now": False,
            "would_evict": [],
            "fits_now_ctx": None,
        }
        if optimal is None:
            entry["reason"] = f"this model does not fit on {mode.label} even with those cards empty"
            entries.append(entry)
            continue

        live_record = forced_onto(record, mode.devices)
        entry["fits_now"] = _plan_fits(live, live_record, optimal, loaded=loaded, allow_evict=False)
        if entry["fits_now"]:
            # The optimal itself loads, so the largest context that fits is at
            # least this one; walking the ladder again to find out whether some
            # lower-quality rung reaches further would cost seven more plans per
            # mode per model to refine a number nobody is blocked on.
            entry["fits_now_ctx"] = int(optimal["ctx_per_slot"])
        else:
            entry["would_evict"] = _eviction_price(live, live_record, optimal, loaded=loaded)
            entry["fits_now_ctx"] = _largest_now(
                live, live_record, tiers, loaded=loaded, record=record
            )

        if pinned_kv:
            free_optimal, free_basis = _optimal_for(
                unpinned_kv(record),
                mode,
                planner=planner,
                probe=probe,
                ctx_tiers=tiers,
                floor=floor,
                preference=preference,
                calibrate_for=calibrate_for,
            )
            if (
                free_optimal is not None
                and free_optimal["kv_cache_type"] != optimal["kv_cache_type"]
            ):
                entry["if_unpinned"] = {**free_optimal, "basis": free_basis}
        entries.append(entry)

    _rank(entries)
    return entries


def _plan_fits(
    planner: Planner,
    record: ModelRecord,
    optimal: Mapping[str, Any],
    *,
    loaded: Sequence[Any],
    allow_evict: bool,
) -> bool:
    return isinstance(_plan_optimal(planner, record, optimal, loaded, allow_evict), LoadPlan)


def _plan_optimal(
    planner: Planner,
    record: ModelRecord,
    optimal: Mapping[str, Any],
    loaded: Sequence[Any],
    allow_evict: bool,
) -> Any:
    """Plan exactly what ``optimal.load_args`` asks for -- the same slots included.

    Quoting a ``fits_now`` for a smaller load than the one ``load_args`` would
    submit is how a table comes to say yes to a call that then fails. Since
    WP19 the slot count in ``load_args`` is ``recommended_parallel`` rather than
    ``max_parallel``, so this follows it; falling back to ``max_parallel`` keeps
    a hand-built ``optimal`` in a test meaning what it used to.
    """
    try:
        return planner.plan_load(
            record,
            ctx_size=int(optimal["ctx_per_slot"]),
            kv_cache_type=optimal["kv_cache_type"],
            kv_cache_type_v=optimal.get("kv_cache_type_v") or optimal["kv_cache_type"],
            parallel=int(optimal.get("recommended_parallel") or optimal["max_parallel"]),
            loaded=loaded,
            allow_evict=allow_evict,
        )
    except Exception as exc:  # noqa: BLE001 - one bad mode must not empty the report
        log.debug("placement plan failed", model_id=record.id, error=str(exc))
        return None


def _eviction_price(
    planner: Planner,
    record: ModelRecord,
    optimal: Mapping[str, Any],
    *,
    loaded: Sequence[Any],
) -> list[str]:
    """Model ids this placement would stop if it were allowed to evict.

    May be empty even when ``fits_now`` is false: something the planner is not
    allowed to touch (a pinned model, a busy one -- D36's busy rule) can be
    what is in the way, and an empty list beside ``fits_now: false`` says
    exactly that.
    """
    plan = _plan_optimal(planner, record, optimal, loaded, True)
    return list(plan.evict_model_ids) if isinstance(plan, LoadPlan) else []


def _largest_now(
    planner: Planner,
    live_record: ModelRecord,
    tiers: Sequence[int],
    *,
    loaded: Sequence[Any],
    record: ModelRecord,
) -> int | None:
    """The biggest context tier that loads on this mode right now, or ``None``."""
    from studioforge.core import catalog as catalog_mod

    for ctx in reversed(catalog_mod.ctx_tiers_for(record, tiers)):
        plan = catalog_mod.plan_at(planner, live_record, ctx, loaded=loaded)
        if isinstance(plan, LoadPlan):
            return int(plan.ctx_size)
    return None


def _rank(entries: list[dict[str, Any]]) -> None:
    """Label each mode ``fastest`` / ``largest_context`` / ``cheapest``.

    ``cheapest`` is the fewest cards that still reach the largest context, ties
    broken towards the *slower* class -- the point of the label is "use this one
    and leave the good hardware free for something else".
    """
    usable = [e for e in entries if e.get("optimal")]
    for entry in entries:
        entry["ranking"] = []
    if not usable:
        return

    fastest = max(usable, key=lambda e: float(e["optimal"]["est_gen_tps"] or 0.0))
    fastest["ranking"].append("fastest")

    widest = max(int(e["optimal"]["ctx_per_slot"]) for e in usable)
    for entry in usable:
        if int(entry["optimal"]["ctx_per_slot"]) == widest:
            entry["ranking"].append("largest_context")

    at_widest = [e for e in usable if int(e["optimal"]["ctx_per_slot"]) == widest]
    cheapest = min(
        at_widest,
        key=lambda e: (len(e["devices"]), float(e["optimal"]["est_gen_tps"] or 0.0)),
    )
    cheapest["ranking"].append("cheapest")
