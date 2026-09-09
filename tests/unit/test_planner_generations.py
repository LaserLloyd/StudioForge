"""Same-generation placements before mixed ones (2026-09-09 review).

The candidate order sorts cards by compute capability and then by free VRAM,
so on the reference rig a 5090+3090 pair sorted ahead of the 3090 pair on both
keys -- and a layer split across the two generations runs at the 3090's pace
plus a sync hop per token, measured here at roughly half a same-generation
pair's speed. ``planner.mixed_generation_split`` orders the walk.
"""

from __future__ import annotations

from studioforge.core.planner import Planner
from studioforge.types import GB, LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_planner import (
    StubProbe,
    gpu,
    make_config,
    make_meta,
    make_record,
    rig_5090x2_3090x2,
)


def _model(gib: int):
    """A model of ``gib`` GiB of weights. 34 GiB is the interesting size on
    the reference rig: too big for one 5090 (~27.8 GiB usable after headroom),
    small enough for the 3090 pair (~42 GiB usable) once the compute buffers
    and the two CUDA contexts are charged."""
    return make_record(meta=make_meta(tensor_bytes=gib * GB), size_bytes=gib * GB)


def _rig_where_the_5090_pair_is_full() -> StubProbe:
    """One 5090 nearly full, one free, both 3090s free: a 34 GiB model fits on
    5090+3090 (mixed) and on the 3090 pair (same generation), not on either
    5090 alone and not on the 5090 pair."""
    return StubProbe(
        [
            gpu(0, 31.84, 2.0, (12, 0)),
            gpu(1, 31.84, 31.0, (12, 0)),
            gpu(2, 24.0, 23.5, (8, 6)),
            gpu(3, 24.0, 23.5, (8, 6)),
        ]
    )


def test_the_same_generation_pair_is_tried_before_a_mixed_one() -> None:
    planner = Planner(make_config(), _rig_where_the_5090_pair_is_full())
    plan = planner.plan_load(_model(34), ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [2, 3], "the 3090 pair, not 5090+3090"
    assert not any("mixed-generation" in n for n in plan.notes)


def test_the_old_order_is_still_available_and_says_when_it_mixed() -> None:
    planner = Planner(make_config(mixed_generation_split="any"), _rig_where_the_5090_pair_is_full())
    plan = planner.plan_load(_model(34), ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1, 2], "compute capability then free VRAM: 5090+3090 first"
    assert any("mixed-generation split" in n for n in plan.notes)


def test_a_mixed_split_is_the_fallback_when_nothing_else_fits() -> None:
    probe = StubProbe(
        [
            gpu(0, 31.84, 31.0, (12, 0)),
            gpu(1, 31.84, 2.0, (12, 0)),
            gpu(2, 24.0, 23.5, (8, 6)),
            gpu(3, 24.0, 5.0, (8, 6)),
        ]
    )
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(_model(34), ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0, 2]
    note = next(n for n in plan.notes if "mixed-generation split" in n)
    assert "half the speed" in note
    assert "no same-generation placement fit" in note


def test_never_refuses_a_mixed_split_and_says_so() -> None:
    probe = StubProbe(
        [
            gpu(0, 31.84, 31.0, (12, 0)),
            gpu(1, 31.84, 2.0, (12, 0)),
            gpu(2, 24.0, 23.5, (8, 6)),
            gpu(3, 24.0, 5.0, (8, 6)),
        ]
    )
    planner = Planner(make_config(mixed_generation_split="never"), probe)
    result = planner.plan_load(_model(34), ctx_size=4096)
    assert isinstance(result, LoadRejected)
    assert any("mixed-generation splits were not considered" in n for n in result.notes)


def test_never_still_places_a_model_that_needs_the_whole_same_generation_pair() -> None:
    planner = Planner(make_config(mixed_generation_split="never"), rig_5090x2_3090x2())
    plan = planner.plan_load(_model(40), ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0, 1]


def test_a_device_override_is_never_second_guessed_by_the_policy() -> None:
    planner = Planner(make_config(mixed_generation_split="never"), rig_5090x2_3090x2())
    record = make_record(
        meta=make_meta(tensor_bytes=40 * GB),
        size_bytes=40 * GB,
        settings=ModelSettings(device_override=[1, 2]),
    )
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1, 2]


def test_the_plan_carries_the_allowed_devices_bound_it_was_chosen_within() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(settings=ModelSettings(allowed_devices=[2, 3]))
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.allowed_devices == [2, 3]
    assert set(plan.devices) <= {2, 3}

    unbounded = planner.plan_load(make_record(), ctx_size=4096)
    assert isinstance(unbounded, LoadPlan)
    assert unbounded.allowed_devices is None


def test_the_refusal_names_the_shortfall_and_the_largest_term() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=100 * GB), size_bytes=100 * GB)
    result = planner.plan_load(record, ctx_size=4096)
    assert isinstance(result, LoadRejected)
    message = result.message()
    assert "short by" in message
    assert "largest term is the model weights" in message
    term = result.largest_term()
    assert term is not None and term[0] == "model weights"
