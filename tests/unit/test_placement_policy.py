"""The 2026-08-26 placement policies: min_ctx, allowed_devices, placement_tier.

Three answers to the same complaint -- "the load worked and the result was
quietly wrong": a fallback model serving a window a long session cannot live
in, a big model sprawling onto cards its owner never meant for it, and a
mixed-generation split running at half speed with nothing saying so.
"""

from __future__ import annotations

from studioforge.core.planner import Planner
from studioforge.types import GB, LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_planner import StubProbe, gpu, make_config, make_meta, make_record


def test_min_ctx_refuses_below_the_floor_instead_of_serving_less() -> None:
    """A 61k window that "works" per turn and then shreds a 51k-prompt session
    through compaction is worse than a structured refusal."""
    probe = StubProbe([gpu(0, 32.0, 10.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB, n_ctx_train=262144),
        settings=ModelSettings(min_ctx=131072),
    )

    result = planner.plan_load(record, loaded=[])
    if isinstance(result, LoadPlan):
        assert int(result.ctx_per_slot or result.ctx_size) >= 131072
    else:
        assert any("min_ctx" in s for s in result.suggestions)


def test_min_ctx_names_itself_in_the_refusal() -> None:
    """Without the line, "not even the floor fits" reads as a VRAM problem."""
    probe = StubProbe([gpu(0, 32.0, 9.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB, n_ctx_train=262144),
        settings=ModelSettings(min_ctx=262144),
    )
    result = planner.plan_load(record, loaded=[])
    assert isinstance(result, LoadRejected)
    assert any("min_ctx" in s for s in result.suggestions)


def test_an_explicit_ctx_ask_outranks_min_ctx() -> None:
    """D14: an explicit value is honoured verbatim, the policy floor is not."""
    probe = StubProbe([gpu(0, 32.0, 20.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB, n_ctx_train=262144),
        settings=ModelSettings(min_ctx=131072),
    )
    plan = planner.plan_load(record, ctx_size=8192, loaded=[])
    assert isinstance(plan, LoadPlan)
    assert plan.ctx_size == 8192


def test_allowed_devices_bounds_the_planners_choice() -> None:
    """Softer than device_override: a set to choose within, never outside."""
    probe = StubProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB),
        settings=ModelSettings(allowed_devices=[1]),
    )
    plan = planner.plan_load(record, ctx_size=4096, loaded=[])
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1]
    assert any("allowed_devices" in n for n in plan.notes)


def test_allowed_devices_matching_nothing_is_a_named_refusal() -> None:
    probe = StubProbe([gpu(0, 32.0, 30.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB),
        settings=ModelSettings(allowed_devices=[7]),
    )
    result = planner.plan_load(record, ctx_size=4096, loaded=[])
    assert isinstance(result, LoadRejected)
    assert "allowed_devices" in result.reason


def test_a_mixed_generation_split_is_graded_and_named() -> None:
    """Measured ~half generation speed on this rig's 5090+3090 splits; the
    plan must say so instead of letting the slowness surface as a different
    complaint later."""
    probe = StubProbe([gpu(0, 32.0, 14.0, (12, 0)), gpu(1, 24.0, 14.0, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=20 * GB), size_bytes=20 * GB)

    plan = planner.plan_load(record, ctx_size=4096, loaded=[])
    assert isinstance(plan, LoadPlan)
    assert sorted(plan.devices) == [0, 1]
    assert plan.placement_tier == "degraded"
    assert any("mixed-generation split" in n for n in plan.notes)


def test_a_single_card_placement_is_optimal() -> None:
    probe = StubProbe([gpu(0, 32.0, 30.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=8 * GB))
    plan = planner.plan_load(record, ctx_size=4096, loaded=[])
    assert isinstance(plan, LoadPlan)
    assert plan.placement_tier == "optimal"


# ---------------------------------------------------------------------------
# Per-model prefer_single_gpu (D48)
# ---------------------------------------------------------------------------


def two_matched_5090s() -> Planner:
    """Two identical cards, either of which holds the model on its own."""
    config = make_config(headroom_fraction=0.0)
    assert config.planner.prefer_single_gpu is True, "the shipped global policy"
    return Planner(config, StubProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 32.0, 30.0, (12, 0))]))


def test_a_per_model_false_overrides_a_true_global() -> None:
    """One model that is happier split must not cost every other model on the
    box the single-card rule -- before D48, ``planner.prefer_single_gpu`` was
    the only place to say it and it applied to the whole library."""
    planner = two_matched_5090s()
    record = make_record(
        meta=make_meta(tensor_bytes=8 * GB),
        settings=ModelSettings(prefer_single_gpu=False),
    )
    plan = planner.plan_load(record, ctx_size=4096, loaded=[])
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0, 1]


def test_none_inherits_the_global_policy() -> None:
    """The default, and the reason an upgrade changes nothing: every existing
    model carries ``None``."""
    planner = two_matched_5090s()
    inherited = planner.plan_load(
        make_record(meta=make_meta(tensor_bytes=8 * GB)), ctx_size=4096, loaded=[]
    )
    assert isinstance(inherited, LoadPlan)
    assert inherited.devices == [0]
    assert make_record().settings.prefer_single_gpu is None


def test_a_per_model_true_holds_the_single_card_when_the_global_is_off() -> None:
    """The override runs both ways: an operator who turned the global policy
    off for the box can still keep one model on one card."""
    config = make_config(headroom_fraction=0.0, prefer_single_gpu=False)
    planner = Planner(config, StubProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 32.0, 30.0, (12, 0))]))
    meta = make_meta(tensor_bytes=8 * GB)

    spread = planner.plan_load(make_record(meta=meta), ctx_size=4096, loaded=[])
    assert isinstance(spread, LoadPlan)
    assert spread.devices == [0, 1]

    kept = planner.plan_load(
        make_record(meta=meta, settings=ModelSettings(prefer_single_gpu=True)),
        ctx_size=4096,
        loaded=[],
    )
    assert isinstance(kept, LoadPlan)
    assert kept.devices == [0]
