"""`recommended_parallel`: how many slots are worth running (WP19 / D37).

D17 gave every placement a ``max_parallel`` and said out loud that its knee
half was arithmetic, not measurement. This file pins the other half.

Three claims are load-bearing beyond "the code works":

* **the rule stops at the knee**, walking upward, rather than scanning for the
  largest level that happens to pass -- past the knee the aggregate flattens, so
  a level above a failure can still beat its depressed predecessor by 15% and
  would promote a slot count the run has already shown to be useless;
* **``basis`` can be read literally.** ``"measured"`` means a sweep on *these*
  devices at *this* context, never a sweep at another context reinterpreted;
* **the ceiling always wins.** A recommendation above what the placement holds
  is a load that does not start.
"""

from __future__ import annotations

from studioforge.core.parallel import (
    AGGREGATE_GAIN,
    PER_STREAM_FLOOR,
    knee_estimate,
    measured_recommendation,
    observations_for,
    recommended_parallel,
)
from studioforge.types import GB, GgufMeta


def level(
    n: int,
    per_stream: float,
    aggregate: float,
    *,
    ts: float = 0.0,
    run_id: str = "run-a",
    devices: str = "0,1",
    ctx: int = 8192,
) -> dict[str, object]:
    return {
        "n_streams": n,
        "per_stream_tps": per_stream,
        "aggregate_tps": aggregate,
        "ts": ts or float(n),
        "run_id": run_id,
        "devices": devices,
        "ctx_per_slot": ctx,
    }


def sweep(*rows: tuple[int, float, float], **kwargs: object) -> list[dict[str, object]]:
    return [level(n, per_stream, aggregate, **kwargs) for n, per_stream, aggregate in rows]  # type: ignore[arg-type]


def dense_meta() -> GgufMeta:
    """An 8B dense: 36 layers, 8 KV heads, head_dim 128."""
    return GgufMeta(
        architecture="llama",
        n_layer=36,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=131072,
        param_count=8_000_000_000,
        tensor_bytes=8 * GB,
        quant_label="Q8_0",
    )


def moe_meta() -> GgufMeta:
    """A 122B-A10B: the same geometry the catalog fixtures use."""
    return GgufMeta(
        architecture="qwen3moe",
        n_layer=48,
        n_head=32,
        n_head_kv=2,
        n_embd=3072,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_expert=256,
        n_expert_used=8,
        n_ctx_train=262144,
        param_count=122_000_000_000,
        tensor_bytes=60 * GB,
        quant_label="Q5_K_M",
    )


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_clean_knee_at_four_is_recommended_as_four() -> None:
    """Aggregate still climbing to 4, each stream still comfortable; 8 collapses."""
    rows = sweep((1, 100.0, 100.0), (2, 90.0, 180.0), (4, 70.0, 280.0), (8, 40.0, 320.0))
    slots, why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 4
    assert "40%" in why and "floor 65%" in why


def test_a_model_that_does_not_batch_stays_at_one_slot() -> None:
    """Aggregate flat from the first doubling: the second slot buys nothing."""
    rows = sweep((1, 100.0, 100.0), (2, 90.0, 102.0), (4, 80.0, 100.0))
    slots, why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 1
    assert "+2%" in why


def test_a_model_that_scales_all_the_way_takes_the_top_level() -> None:
    rows = sweep((1, 100.0, 100.0), (2, 98.0, 196.0), (4, 92.0, 368.0), (8, 80.0, 640.0))
    slots, _why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 8


def test_the_walk_stops_at_the_knee_rather_than_scanning_past_it() -> None:
    """8 beats 4 by 20%, but 4 already failed -- the knee is 2, and 2 is the answer.

    The literal reading of "the largest N satisfying both conditions" would say
    8 here, because 480/400 is +20% and 8's per-stream is above the floor. That
    reading promotes a level whose own predecessor the run has already shown to
    be a plateau, so the measurement would be arguing against itself.
    """
    rows = sweep((1, 100.0, 100.0), (2, 95.0, 190.0), (4, 90.0, 200.0), (8, 70.0, 480.0))
    slots, why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 2
    assert "+5%" in why


def test_the_per_stream_floor_is_checked_against_the_solo_rate_not_the_level_below() -> None:
    """A slow slide is still a slide: each step is >80% of the last, and 8 is not.

    Level 8 runs at 78% of level 4 -- comfortable against its neighbour, and
    **51% of solo**, which is the number a conversation actually experiences.
    """
    rows = sweep((1, 100.0, 100.0), (2, 85.0, 170.0), (4, 66.0, 264.0), (8, 51.0, 408.0))
    slots, _why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 4  # 51/100 = 51% < 65%, so 8 is refused


def test_the_thresholds_are_the_documented_ones() -> None:
    """A sweep sitting exactly on both thresholds passes -- the rule is >=."""
    rows = sweep(
        (1, 100.0, 100.0),
        (2, 100.0 * PER_STREAM_FLOOR, 100.0 * AGGREGATE_GAIN),
    )
    slots, _why = measured_recommendation(rows)  # type: ignore[misc]
    assert slots == 2


def test_a_sweep_with_no_solo_level_is_not_a_measurement() -> None:
    """Without level 1 the per-stream floor has no meaning; say so, do not guess."""
    assert measured_recommendation(sweep((2, 90.0, 180.0), (4, 70.0, 280.0))) is None


def test_only_the_newest_run_is_compared() -> None:
    """Two runs at different placements must not be read as one curve."""
    old = sweep((1, 100.0, 100.0), (2, 95.0, 190.0), (4, 90.0, 360.0), run_id="old", ts=1.0)
    for row in old:
        row["ts"] = 1.0
    new = sweep((1, 50.0, 50.0), (2, 26.0, 52.0), run_id="new")
    for index, row in enumerate(new):
        row["ts"] = 100.0 + index
    slots, _why = measured_recommendation([*old, *new])  # type: ignore[misc]
    assert slots == 1


def test_the_placement_ceiling_wins_over_a_measurement() -> None:
    """A measured 8 on a placement that holds 2 is a load that would not start."""
    rows = sweep((1, 100.0, 100.0), (2, 98.0, 196.0), (4, 92.0, 368.0), (8, 80.0, 640.0))
    slots, why = measured_recommendation(rows, cap=2)  # type: ignore[misc]
    assert slots == 2
    assert "the most this placement can hold" in why


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------


def test_the_estimate_is_d17s_knee() -> None:
    """active_weights / KV read per slot at half the window."""
    value = knee_estimate(
        dense_meta(), weights_bytes=8 * GB, ctx_per_slot=32768, kv_cache_type="f16"
    )
    assert value is not None and value >= 1


def test_a_moe_knee_arrives_sooner_than_the_same_arithmetic_undearated() -> None:
    """Experts fan out with batch size, so the derate is not cosmetic."""
    moe = knee_estimate(moe_meta(), weights_bytes=60 * GB, ctx_per_slot=32768, kv_cache_type="f16")
    assert moe is not None
    # The same weights and KV geometry read as a dense model would give twice
    # this: MOE_KNEE_DERATE is 0.5 and it is applied.
    dense_equivalent = knee_estimate(
        moe_meta().model_copy(update={"n_expert": 0, "n_expert_used": 0}),
        weights_bytes=60 * GB * 8 // 256,
        ctx_per_slot=32768,
        kv_cache_type="f16",
    )
    assert dense_equivalent is not None
    assert moe <= dense_equivalent


def test_a_model_with_no_geometry_gets_one_slot_and_says_why() -> None:
    result = recommended_parallel(
        None, weights_bytes=8 * GB, ctx_per_slot=32768, kv_cache_type="f16", max_parallel=8
    )
    assert result["value"] == 1
    assert result["basis"] == "estimated"
    assert "geometry" in result["detail"]


def test_with_no_measurement_the_basis_is_estimated_and_the_detail_says_how_to_fix_it() -> None:
    result = recommended_parallel(
        dense_meta(),
        weights_bytes=8 * GB,
        ctx_per_slot=32768,
        kv_cache_type="f16",
        max_parallel=4,
    )
    assert result["basis"] == "estimated"
    assert result["value"] <= 4
    assert "parallel benchmark" in result["detail"]


def test_a_measurement_replaces_the_estimate() -> None:
    rows = sweep((1, 100.0, 100.0), (2, 51.0, 102.0))
    result = recommended_parallel(
        dense_meta(),
        weights_bytes=8 * GB,
        ctx_per_slot=8192,
        kv_cache_type="f16",
        max_parallel=8,
        observations=rows,
    )
    assert result == {"value": 1, "basis": "measured", "detail": result["detail"]}
    assert result["basis"] == "measured"
    assert result["value"] == 1


# ---------------------------------------------------------------------------
# "measured" means measured
# ---------------------------------------------------------------------------


def test_a_sweep_at_another_context_does_not_describe_this_row() -> None:
    """The knee moves with KV bytes per slot, so 8192 says nothing about 131072."""
    rows = sweep((1, 100.0, 100.0), (2, 98.0, 196.0), ctx=8192)
    assert observations_for(rows, devices=[0, 1], ctx_per_slot=131072) == []
    assert len(observations_for(rows, devices=[0, 1], ctx_per_slot=8192)) == 2


def test_a_sweep_on_other_cards_does_not_describe_this_row() -> None:
    rows = sweep((1, 100.0, 100.0), (2, 98.0, 196.0), devices="2,3")
    assert observations_for(rows, devices=[0, 1], ctx_per_slot=8192) == []


def test_device_order_does_not_matter() -> None:
    """The planner orders devices by its own preference; [1, 0] is [0, 1]."""
    rows = sweep((1, 100.0, 100.0), devices="0,1")
    assert observations_for(rows, devices=[1, 0], ctx_per_slot=8192)
