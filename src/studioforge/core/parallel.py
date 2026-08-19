"""How many parallel slots are *worth* running, measured where possible.

D17 gave every placement a ``max_parallel``: the smaller of a VRAM bound (the
KV cache is the one term that scales with slots -- ``--ctx-size`` is the total
across them, D4) and a *knee* bound derived from memory bandwidth. That knee is
arithmetic, and D17 said so out loud: ``CTX_FILL_FRACTION`` of 0.5 and the MoE
derate of 0.5 are "deliberate approximations in the safe direction, not
measurements -- the real curves want a benchmark this rig has not run".

This module is the other half. :func:`recommended_parallel` answers "how many
slots should this load actually ask for", from a **measured** sweep when one
exists and from the knee estimate when it does not, and says which of the two it
did.

The physics the rule encodes
----------------------------

Batch-1 decode is memory-bandwidth bound. Every step streams the active weights
through the memory system once regardless of how many sequences are in the
batch, and then reads each busy slot's KV cache on top. So while weight traffic
dominates, N busy slots amortise one weight read across N tokens and the
*aggregate* rate rises close to linearly; per-slot rate falls slowly. Once the N
slots' KV reads match the weight read, another slot buys nothing and costs VRAM
and latency -- that crossover is D17's knee,
``active_weights / (ctx_fill * kv_bytes_per_token)``.

Published measurements of the same effect: llama.cpp's own batching discussion
(https://github.com/ggml-org/llama.cpp/discussions/4130) explains why the
aggregate scales; a system-design writeup measured roughly **3.8x aggregate**
over sequential decode from slot batching on a single T4
(https://markaicode.com/architecture/llamacpp-system-design-architecture-1158/).
``llama-server``'s ``--parallel`` / ``--cont-batching`` flags and the
``/metrics`` counter ``llamacpp:n_busy_slots_per_decode`` that proves batching
actually happened are documented at
https://manpages.debian.org/testing/llama.cpp-tools/llama-server.1.en.html, and
the batch/ubatch dimension at
https://www.promptsicle.com/tips/boosting-llama-server-performance-with-batch-settings/.
Every one of those says the useful slot count is workload- and
hardware-specific, which is why this module prefers a measurement to a formula.

A MoE complicates it: expert fan-out grows with batch size (at batch N roughly
``min(N * n_expert_used, n_expert)`` experts are touched), so weight traffic
stops being flat and the knee arrives sooner. D17 derates it by half. That is
still a guess, and it is a guess a measurement replaces.

**Measured here, and the reason this module exists** (D37, 2026-08-19, a
Qwen2.5-1.5B on one RTX 3090 at 8192 tokens per slot): 1 / 2 / 4 / 8 slots gave
per-stream 302.8 / 225.3 / 134.5 / 83.3 t/s against aggregate 302.8 / 425.3 /
436.0 / 576.9. The **bandwidth** knee -- what the estimate finds -- is at 8: the
aggregate is still climbing there. The **useful** knee is at 2, because by 8
slots a single conversation is running at 27% of its solo speed. The two are not
the same quantity, and no amount of refining the arithmetic would have found the
second one.

**Quality is not involved.** A slot count changes nothing about the answer a
model gives; what each conversation gets is ``ctx_per_slot``, and that is per
slot by construction (D4). "Recommended parallel" is a throughput-and-latency
decision only, which is why it can be measured with a benchmark and why it never
overrides the quality-first KV choice (D36).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from studioforge.core.planner import (
    CTX_FILL_FRACTION,
    MOE_KNEE_DERATE,
    active_weight_bytes,
    is_moe,
    kv_read_bytes_per_slot,
)

#: Concurrency levels a parallel benchmark sweeps, and therefore the values a
#: measured recommendation can take. Doubling rather than 1..8 because the
#: quantity being located is a knee on a curve that changes by factors, and
#: because every extra level costs a full generation pass on a busy rig.
PARALLEL_LEVELS: tuple[int, ...] = (1, 2, 4, 8)

#: A slot count is only recommended while one conversation still runs at this
#: fraction of its solo speed. Below that the aggregate has been bought with
#: latency a user feels turn by turn, and this server's stated priority is the
#: experience of the conversation, not the tokens-per-second of the box.
PER_STREAM_FLOOR = 0.65

#: ...and only while doubling the slots still buys this much aggregate. 1.15 is
#: deliberately well under the 2.0 of perfect scaling and well over measurement
#: noise: a level that adds under 15% is the plateau, and calling the plateau
#: "worth it" is how a recommendation ends up at the cap for every model.
AGGREGATE_GAIN = 1.15


def _level_rows(observations: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    """``{n_streams: row}`` for the newest run in ``observations``.

    "Newest run" is a group, not a row: one sweep writes four rows and the
    recommendation compares them against each other, so mixing the top of one
    run with the bottom of an older one at a different context would compare
    two different machines. ``run_id`` is the group key; rows written before it
    existed (or by a caller that did not set one) fall back to "every row
    here", which is what a hand-assembled table in a test means.
    """
    rows = [r for r in observations if r.get("n_streams")]
    if not rows:
        return {}
    newest = max(rows, key=lambda r: float(r.get("ts") or 0.0))
    run_id = newest.get("run_id")
    if run_id:
        rows = [r for r in rows if r.get("run_id") == run_id]
    by_level: dict[int, Mapping[str, Any]] = {}
    for row in sorted(rows, key=lambda r: float(r.get("ts") or 0.0)):
        by_level[int(row["n_streams"])] = row
    return by_level


def _slots(n: int) -> str:
    return f"{n} slot" if n == 1 else f"{n} slots"


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if number > 0 else None


def measured_recommendation(
    observations: Sequence[Mapping[str, Any]], *, cap: int = 0
) -> tuple[int, str] | None:
    """Apply the rule to one measured sweep: ``(slots, why)`` or ``None``.

    The rule, in one sentence: **the largest level whose per-stream rate is
    still at least** :data:`PER_STREAM_FLOOR` **of the solo rate and whose
    aggregate is at least** :data:`AGGREGATE_GAIN` **times the level below it.**

    Walked upward, stopping at the first level that fails, rather than scanning
    for "the largest level that happens to satisfy both". The quantity being
    located is a knee, and past the knee the aggregate flattens -- so a level
    above a failure can still pass the ratio test against its depressed
    predecessor and would promote a slot count the run has already shown to be
    useless. Stopping at the knee is the claim the benchmark actually supports.

    ``None`` when there is no level 1 to compare against: without the solo rate
    the per-stream floor has no meaning, and a rule that silently drops half of
    itself is worse than an estimate that admits what it is.
    """
    levels = _level_rows(observations)
    solo = levels.get(1)
    if solo is None:
        return None
    base = _number(solo, "per_stream_tps")
    base_aggregate = _number(solo, "aggregate_tps")
    if base is None or base_aggregate is None:
        return None

    limit = int(cap) if cap and cap > 0 else max(PARALLEL_LEVELS)
    best = 1
    reason = f"{_slots(1)}: no higher level was measured"
    previous_aggregate = base_aggregate
    for level in PARALLEL_LEVELS[1:]:
        if level > limit:
            reason = f"{_slots(best)}: {limit} is the most this placement can hold"
            break
        row = levels.get(level)
        if row is None:
            break
        per_stream = _number(row, "per_stream_tps")
        aggregate = _number(row, "aggregate_tps")
        if per_stream is None or aggregate is None:
            break
        if per_stream < PER_STREAM_FLOOR * base:
            reason = (
                f"{_slots(best)}: at {level} each stream drops to "
                f"{per_stream / base:.0%} of its solo speed "
                f"(floor {PER_STREAM_FLOOR:.0%})"
            )
            break
        if aggregate < AGGREGATE_GAIN * previous_aggregate:
            reason = (
                f"{_slots(best)}: going from {best} to {level} adds only "
                f"{aggregate / previous_aggregate - 1:+.0%} aggregate "
                f"(needs +{AGGREGATE_GAIN - 1:.0%})"
            )
            break
        best = level
        previous_aggregate = aggregate
        reason = (
            f"{_slots(best)}: aggregate {aggregate:.1f} t/s "
            f"({aggregate / base_aggregate:.1f}x one slot), each stream still "
            f"{per_stream / base:.0%} of solo"
        )
    return best, reason


def knee_estimate(
    meta: Any,
    *,
    weights_bytes: int,
    ctx_per_slot: int,
    kv_cache_type: str,
    kv_cache_type_v: str | None = None,
) -> int | None:
    """D17's knee, in slots, or ``None`` when the geometry cannot support it.

    ``active_weights / kv_read_per_slot`` at :data:`CTX_FILL_FRACTION` of the
    window, halved for a MoE. Exactly the arithmetic
    :func:`studioforge.core.planner.max_parallel_for` uses for its knee half --
    deliberately the same function's worth of reasoning rather than a second
    copy of it, because a table that disagrees with the planner about
    concurrency is how ``max_parallel: 4`` came to sit beside a load that
    settled for two.
    """
    if meta is None or ctx_per_slot <= 0:
        return None
    kv_v = kv_cache_type_v or kv_cache_type
    read = kv_read_bytes_per_slot(
        meta,
        kv_k=kv_cache_type,
        kv_v=kv_v,
        ctx_fill=max(1, int(ctx_per_slot * CTX_FILL_FRACTION)),
    )
    if read <= 0:
        return None
    active = active_weight_bytes(meta, weights_bytes)
    if active <= 0:
        return None
    knee = active / read
    if is_moe(meta):
        knee *= MOE_KNEE_DERATE
    return max(1, int(round(knee)))


def recommended_parallel(
    meta: Any,
    *,
    weights_bytes: int,
    ctx_per_slot: int,
    kv_cache_type: str,
    kv_cache_type_v: str | None = None,
    max_parallel: int,
    observations: Sequence[Mapping[str, Any]] = (),
    is_moe_model: bool | None = None,
) -> dict[str, Any]:
    """How many slots this placement should actually ask for.

    Returns ``{"value": int, "basis": "measured"|"estimated", "detail": str}``.

    ``max_parallel`` is the ceiling in both branches: it is what the placement
    can hold, and a recommendation above it is a load that will not start.
    Under it, the two branches are:

    * **measured** -- :func:`measured_recommendation` over the newest parallel
      benchmark on these devices at this context. A measurement at a *different*
      context is not used: the knee moves with KV bytes per slot, so a sweep at
      8192 says nothing about the same model at 131072. The caller narrows the
      rows it passes in (see :func:`observations_for`).
    * **estimated** -- :func:`knee_estimate`, clamped into ``[1, max_parallel]``.

    A note about the estimated branch that is worth saying out loud rather than
    leaving for someone to discover: because D17 already folds the same knee
    into ``max_parallel``, the estimate normally lands **on** ``max_parallel``
    rather than below it. That is not a bug and it is not redundancy -- it is
    the honest statement that with no measurement there is nothing to say beyond
    what the planner already said. The field earns its keep the moment a run
    exists, because a measured value can and does come in lower.
    """
    ceiling = max(1, int(max_parallel))
    measured = measured_recommendation(observations, cap=ceiling)
    if measured is not None:
        value, why = measured
        return {"value": min(ceiling, max(1, value)), "basis": "measured", "detail": why}

    knee = knee_estimate(
        meta,
        weights_bytes=weights_bytes,
        ctx_per_slot=ctx_per_slot,
        kv_cache_type=kv_cache_type,
        kv_cache_type_v=kv_cache_type_v,
    )
    if knee is None:
        return {
            "value": 1,
            "basis": "estimated",
            "detail": (
                "1 slot: this model's per-layer KV geometry could not be read, "
                "so the bandwidth knee cannot be estimated"
            ),
        }
    value = min(ceiling, max(1, knee))
    moe = is_moe(meta) if is_moe_model is None else bool(is_moe_model)
    detail = (
        f"{_slots(value)}: estimated bandwidth knee "
        f"({knee}) at half a {ctx_per_slot}-token window"
        + (", halved because experts fan out with batch size" if moe else "")
        + (f", capped by the {ceiling} this placement holds" if knee > ceiling else "")
        + " -- run a parallel benchmark to measure it"
    )
    return {"value": value, "basis": "estimated", "detail": detail}


def observations_for(
    observations: Sequence[Mapping[str, Any]],
    *,
    devices: Sequence[int],
    ctx_per_slot: int,
    kv_cache_type: str | None = None,
    kv_cache_type_v: str | None = None,
) -> list[Mapping[str, Any]]:
    """The measured rows that describe *this* placement, or an empty list.

    Same devices, the same context per slot, **and the same KV cache types**.
    All three matter for the same reason: the knee is set by how many KV bytes
    each busy slot reads per step, and a sweep on one 5090 does not describe
    two, a sweep at 8192 does not describe 131072, and a sweep on an f16 cache
    does not describe a q8_0 one (half the bytes per token, so the knee moves).
    Being strict here is what lets ``basis: "measured"`` be read literally --
    the same standard :func:`throughput.confidence_for` holds itself to.

    ``kv_cache_type`` / ``kv_cache_type_v`` are compared only when the caller
    names them; a row that does not record its cache type cannot match a named
    one (the benchmark always records both).
    """
    key = ",".join(str(int(d)) for d in sorted(devices))
    want_k = str(kv_cache_type) if kv_cache_type else None
    want_v = str(kv_cache_type_v or kv_cache_type) if (kv_cache_type_v or kv_cache_type) else None

    def same_kv(row: Mapping[str, Any]) -> bool:
        if want_k is None:
            return True
        row_k = row.get("kv_cache_type")
        row_v = row.get("kv_cache_type_v") or row_k
        return str(row_k) == want_k and str(row_v) == want_v

    return [
        row
        for row in observations
        if str(row.get("devices") or "") == key
        and int(row.get("ctx_per_slot") or 0) == int(ctx_per_slot)
        and same_kv(row)
    ]
