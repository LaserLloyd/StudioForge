"""Parallel-slot estimator, its config surface, and the calibration guards.

The estimator's job is to answer "how many conversations can this placement
actually serve at once?" with arithmetic rather than with a constant. Two
independent bounds decide it -- VRAM (KV cache is per slot) and the
bandwidth knee (decode reads the active weights once per step regardless of
batch size) -- and these tests pin both, using the shapes measured on the
reference rig so a formula change that breaks a real model fails here first.

Numbers come from DECISIONS.md D17 and the audit that produced it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from studioforge.config import Config
from studioforge.core.engine import EngineManager
from studioforge.core.planner import (
    CALIBRATION_MIN_ROWS,
    MAX_PARALLEL_CAP,
    OBSERVATION_NOTE_PER_PID,
    OVERHEAD_FRACTION_MAX,
    OVERHEAD_FRACTION_MIN,
    Planner,
    active_weight_bytes,
    calibrated_overhead_fraction,
    clean_observations,
    is_moe,
    kv_bytes_per_token,
    max_parallel_for,
    parallel_options,
)
from studioforge.types import GgufMeta, LoadPlan, ModelSettings
from tests.unit.test_planner import (
    StubProbe,
    gpu,
    make_config,
    make_meta,
    make_record,
    rig_5090x2_3090x2,
)

KIB = 1024
GB_DEC = 1_000_000_000  # decimal GB, matching how VRAM budgets are quoted


# ---------------------------------------------------------------------------
# Model shapes measured on the reference rig
# ---------------------------------------------------------------------------


def meta_122b_a10b() -> GgufMeta:
    """Qwen3.5-122B-A10B: 48 layers, 2 KV heads, head_dim 256 -> 96 KiB/token."""
    return GgufMeta(
        architecture="qwen3moe",
        n_layer=48,
        n_head=64,
        n_head_kv=2,
        n_embd=8192,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_expert=128,
        n_expert_used=10,
        n_ctx_train=262144,
        quant_label="Q5_K_M",
    )


def meta_27b_dense() -> GgufMeta:
    """Qwen3.8-27B: 64 layers, 8 KV heads, head_dim 128 -> 256 KiB/token."""
    return GgufMeta(
        architecture="qwen3",
        n_layer=64,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=262144,
        quant_label="Q4_K_M",
    )


def meta_8b_dense() -> GgufMeta:
    """An ordinary 8B: 36 layers, 8 KV heads, head_dim 128 -> 144 KiB/token."""
    return GgufMeta(
        architecture="llama",
        n_layer=36,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=131072,
        quant_label="Q8_0",
    )


# ---------------------------------------------------------------------------
# kv_bytes_per_token
# ---------------------------------------------------------------------------


def test_kv_bytes_per_token_122b_a10b_is_96_kib() -> None:
    assert kv_bytes_per_token(meta_122b_a10b(), "f16", "f16") == 96 * KIB


def test_kv_bytes_per_token_27b_is_256_kib() -> None:
    assert kv_bytes_per_token(meta_27b_dense(), "f16", "f16") == 256 * KIB


def test_kv_bytes_per_token_8b_is_144_kib() -> None:
    assert kv_bytes_per_token(meta_8b_dense(), "f16", "f16") == 144 * KIB


def test_kv_bytes_per_token_tracks_the_cache_type() -> None:
    """q8_0 stores 32 values in 34 bytes, so it is 34/64 of f16, not half."""
    f16 = kv_bytes_per_token(meta_122b_a10b(), "f16", "f16")
    q8 = kv_bytes_per_token(meta_122b_a10b(), "q8_0", "q8_0")
    assert q8 == pytest.approx(f16 * (34 / 32) / 2, rel=1e-9)


def test_kv_bytes_per_token_is_zero_without_usable_metadata() -> None:
    """Zero means "cannot estimate", and every caller reads it that way."""
    assert kv_bytes_per_token(GgufMeta(), "f16", "f16") == 0


# ---------------------------------------------------------------------------
# active weights / MoE detection
# ---------------------------------------------------------------------------


def test_active_weights_of_a_dense_model_are_all_of_them() -> None:
    assert active_weight_bytes(meta_8b_dense(), 8 * GB_DEC) == 8 * GB_DEC
    assert not is_moe(meta_8b_dense())


def test_active_weights_of_a_moe_are_the_routed_share() -> None:
    meta = meta_122b_a10b()  # 10 of 128 experts
    assert is_moe(meta)
    assert active_weight_bytes(meta, 128 * GB_DEC) == 10 * GB_DEC


# ---------------------------------------------------------------------------
# max_parallel_for -- the two bounds
# ---------------------------------------------------------------------------


def test_122b_a10b_reaches_four_slots_at_32k_q8_and_is_knee_bound() -> None:
    """The rig's resident agent model: 4 slots at 32k/q8_0, limited by the knee.

    KV is only 51 KiB/token at q8_0, so VRAM would allow a fifth slot. What
    stops it is bandwidth: 10 of 128 experts is ~7 GB read per decode step,
    and by four half-full 32k slots the KV traffic has caught up with it.
    """
    meta = meta_122b_a10b()
    weights = int(82.9 * 1024**3)  # Q5_K_M on disk
    slots, bound = max_parallel_for(
        kv_budget_bytes=9 * GB_DEC,
        kv_per_token=kv_bytes_per_token(meta, "q8_0", "q8_0"),
        ctx_per_slot=32768,
        active_weight_bytes=active_weight_bytes(meta, weights),
        is_moe=is_moe(meta),
    )
    assert (slots, bound) == (4, "knee")


def test_8b_dense_reaches_four_slots_at_32k_f16() -> None:
    """144 KiB/token, ~20 GB of KV budget left on one 5090 after a Q8_0 8B."""
    slots, bound = max_parallel_for(
        kv_budget_bytes=20 * GB_DEC,
        kv_per_token=kv_bytes_per_token(meta_8b_dense(), "f16", "f16"),
        ctx_per_slot=32768,
        active_weight_bytes=int(8.5 * GB_DEC),
        is_moe=False,
    )
    assert slots == 4
    assert bound == "vram"


def test_27b_gets_one_slot_on_one_card_and_four_across_two() -> None:
    """The case that justifies letting a split outrank a single GPU.

    256 KiB/token means a single 32k slot costs a full 8 GiB of KV. One 5090
    has room for exactly one conversation; two have room for the four the
    bandwidth knee allows.
    """
    meta = meta_27b_dense()
    per_token = kv_bytes_per_token(meta, "f16", "f16")
    weights = 16 * GB_DEC

    single = max_parallel_for(
        kv_budget_bytes=12 * GB_DEC,
        kv_per_token=per_token,
        ctx_per_slot=32768,
        active_weight_bytes=weights,
    )
    assert single == (1, "vram")

    pair = max_parallel_for(
        kv_budget_bytes=43 * GB_DEC,
        kv_per_token=per_token,
        ctx_per_slot=32768,
        active_weight_bytes=weights,
    )
    assert pair == (4, "knee")


def test_moe_knee_arrives_sooner_than_the_dense_one() -> None:
    """Experts fan out with batch size, so a MoE's weight traffic is not flat."""
    common = {
        "kv_budget_bytes": 1_000 * GB_DEC,  # deliberately not the binding term
        "kv_per_token": 96 * KIB,
        "ctx_per_slot": 32768,
        "active_weight_bytes": 8 * GB_DEC,
    }
    dense_slots, _ = max_parallel_for(**common, is_moe=False)
    moe_slots, _ = max_parallel_for(**common, is_moe=True)
    assert moe_slots < dense_slots


def test_the_cap_is_reported_as_the_bound() -> None:
    slots, bound = max_parallel_for(
        kv_budget_bytes=10_000 * GB_DEC,
        kv_per_token=8 * KIB,
        ctx_per_slot=4096,
        active_weight_bytes=400 * GB_DEC,
    )
    assert (slots, bound) == (MAX_PARALLEL_CAP, "cap")


def test_a_per_model_cap_lowers_the_answer() -> None:
    slots, bound = max_parallel_for(
        kv_budget_bytes=10_000 * GB_DEC,
        kv_per_token=8 * KIB,
        ctx_per_slot=4096,
        active_weight_bytes=400 * GB_DEC,
        cap=2,
    )
    assert (slots, bound) == (2, "cap")


def test_never_returns_fewer_than_one_slot() -> None:
    """One slot is what a load already does; the estimator can only ever add."""
    slots, bound = max_parallel_for(
        kv_budget_bytes=0,
        kv_per_token=256 * KIB,
        ctx_per_slot=131072,
        active_weight_bytes=16 * GB_DEC,
    )
    assert slots == 1
    assert bound == "vram"


def test_unusable_metadata_reports_one_slot_and_says_it_does_not_know() -> None:
    slots, bound = max_parallel_for(
        kv_budget_bytes=20 * GB_DEC,
        kv_per_token=0,
        ctx_per_slot=32768,
        active_weight_bytes=8 * GB_DEC,
    )
    assert (slots, bound) == (1, "unknown")


def test_measured_kv_read_bytes_move_the_knee_and_nothing_else() -> None:
    """The knee's KV term is now the bytes a slot really reads, when known.

    ``ctx_used * kv_per_token`` assumes every layer re-reads the whole
    transcript on every step. A Gemma-4 reads 1024 tokens on five layers in six,
    so the old term was ~20x too large and pinned the knee near 1. See
    ``tests/unit/test_kv_geometry.py`` for the per-layer numbers; this pins that
    the knob exists here and leaves the VRAM bound alone.
    """
    common = {
        "kv_budget_bytes": 1_000 * GB_DEC,
        "kv_per_token": 256 * KIB,
        "ctx_per_slot": 32768,
        "active_weight_bytes": 30 * GB_DEC,
    }
    naive, _ = max_parallel_for(**common)
    measured, _ = max_parallel_for(**common, kv_read_bytes_per_slot=1 * GB_DEC)
    assert measured > naive

    # A tiny read budget cannot buy slots the VRAM bound does not have.
    starved, bound = max_parallel_for(
        **{**common, "kv_budget_bytes": 8 * GB_DEC}, kv_read_bytes_per_slot=1
    )
    assert (starved, bound) == (1, "vram")


# ---------------------------------------------------------------------------
# parallel_options -- the catalog's table
# ---------------------------------------------------------------------------


def test_parallel_options_covers_every_tier_and_kv_type() -> None:
    rows = parallel_options(
        meta_27b_dense(),
        16 * GB_DEC,
        43 * GB_DEC,
        ctx_tiers=[16384, 32768],
        kv_types=["f16", "q8_0"],
    )
    assert len(rows) == 4
    assert {(r["ctx_per_slot"], r["kv_cache_type"]) for r in rows} == {
        (16384, "f16"),
        (16384, "q8_0"),
        (32768, "f16"),
        (32768, "q8_0"),
    }
    for row in rows:
        assert row["max_parallel"] >= 1
        assert row["kv_bytes"] == row["kv_bytes_per_slot"] * row["max_parallel"]
        assert row["parallel_limited_by"] in {"vram", "knee", "cap", "unknown"}


def test_parallel_options_cheaper_kv_never_buys_fewer_slots() -> None:
    rows = {
        r["kv_cache_type"]: r
        for r in parallel_options(
            meta_27b_dense(),
            16 * GB_DEC,
            20 * GB_DEC,
            ctx_tiers=[32768],
            kv_types=["f16", "q8_0", "q4_0"],
        )
    }
    assert rows["q8_0"]["max_parallel"] >= rows["f16"]["max_parallel"]
    assert rows["q4_0"]["max_parallel"] >= rows["q8_0"]["max_parallel"]


# ---------------------------------------------------------------------------
# Wiring into a real plan
# ---------------------------------------------------------------------------


def auto_config(**overrides: object) -> Config:
    config = make_config()
    config.models.default_parallel = "auto"
    for key, value in overrides.items():
        setattr(config.models, key, value)
    return config


def test_an_explicit_parallel_is_honoured_verbatim_under_auto() -> None:
    """D14: an explicit value wins, and switches the estimator off entirely."""
    planner = Planner(auto_config(), rig_5090x2_3090x2())
    plan = planner.plan_load(make_record(), ctx_size=8192, parallel=3)
    assert isinstance(plan, LoadPlan)
    assert plan.parallel == 3
    assert plan.parallel_limited_by == "explicit"
    assert plan.max_parallel == 3


def test_an_integer_default_parallel_still_pins_the_slot_count() -> None:
    """``default_parallel: 1`` must behave exactly as it did before D17."""
    config = make_config()
    config.models.default_parallel = 1
    planner = Planner(config, rig_5090x2_3090x2())
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.parallel == 1
    assert plan.parallel_limited_by == "explicit"


def test_auto_sizes_more_than_one_slot_for_a_small_model() -> None:
    planner = Planner(auto_config(), rig_5090x2_3090x2())
    plan = planner.plan_load(
        make_record(meta=make_meta(tensor_bytes=4 * 1024**3)), ctx_size=8192
    )
    assert isinstance(plan, LoadPlan)
    assert plan.parallel > 1
    assert plan.parallel_limited_by in {"vram", "knee", "cap"}


def test_the_plan_carries_the_fields_the_api_needs() -> None:
    planner = Planner(auto_config(), rig_5090x2_3090x2())
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.ctx_per_slot == plan.ctx_size == 8192
    assert plan.kv_bytes_per_token > 0
    assert plan.max_parallel >= plan.parallel
    # --ctx-size is the TOTAL across slots (D4); the child gets this number.
    assert plan.ctx_total == plan.ctx_size * plan.parallel


def test_a_per_model_cap_limits_the_automatic_slot_count() -> None:
    planner = Planner(auto_config(), rig_5090x2_3090x2())
    record = make_record(
        meta=make_meta(tensor_bytes=4 * 1024**3),
        settings=ModelSettings(max_parallel_cap=1),
    )
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.parallel == 1


def test_auto_never_turns_a_working_load_into_a_rejection() -> None:
    """A model that fits at one slot must still fit when slots are automatic."""
    probe = StubProbe([gpu(0, 32.0, 16.0, (12, 0))])
    pinned = make_config()
    pinned.models.default_parallel = 1
    fixed_plan = Planner(pinned, probe).plan_load(make_record(), ctx_size=8192)
    assert isinstance(fixed_plan, LoadPlan)

    auto_plan = Planner(auto_config(), probe).plan_load(make_record(), ctx_size=8192)
    assert isinstance(auto_plan, LoadPlan)
    assert auto_plan.parallel >= fixed_plan.parallel


def test_a_split_outranks_a_single_card_only_when_it_breaks_a_one_slot_cage() -> None:
    """prefer_single_gpu yields exactly when a second card buys concurrency.

    A 27B at 32k costs 8 GiB of KV per slot: one 5090 serves one conversation,
    two serve several. Anything less starved keeps the single-GPU placement.
    """
    probe = StubProbe([gpu(0, 31.84, 31.0, (12, 0)), gpu(1, 31.84, 31.0, (12, 0))])
    planner = Planner(auto_config(), probe)
    record = make_record(
        meta=meta_27b_dense().model_copy(update={"tensor_bytes": 16 * 1024**3}),
        size_bytes=16 * 1024**3,
    )
    plan = planner.plan_load(record, ctx_size=32768)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0, 1]
    assert plan.parallel >= 2
    assert any("sustains" in n for n in plan.notes)


def test_a_split_never_drags_a_model_onto_a_slower_card_for_slots() -> None:
    """A split runs at its slowest member's pace: 5090 + 3090 is not a bargain."""
    probe = StubProbe([gpu(0, 31.84, 31.0, (12, 0)), gpu(2, 24.0, 23.5, (8, 6))])
    planner = Planner(auto_config(), probe)
    record = make_record(
        meta=meta_27b_dense().model_copy(update={"tensor_bytes": 16 * 1024**3}),
        size_bytes=16 * 1024**3,
    )
    plan = planner.plan_load(record, ctx_size=32768)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


def test_default_parallel_accepts_auto_and_positive_ints() -> None:
    assert Config(models={"default_parallel": "auto"}).models.default_parallel == "auto"
    assert Config(models={"default_parallel": 4}).models.default_parallel == 4


def test_default_parallel_rejects_nonsense() -> None:
    with pytest.raises(ValidationError):
        Config(models={"default_parallel": 0})
    with pytest.raises(ValidationError):
        Config(models={"default_parallel": "sometimes"})


def test_auto_survives_a_yaml_round_trip(tmp_path: Path) -> None:
    """``save()`` writes "auto" as a string, and loading it back keeps it."""
    from studioforge.config import load_config

    config = Config(data_dir=tmp_path)
    config.models.default_parallel = "auto"
    target = tmp_path / "config.yaml"
    config.save(target)
    assert load_config(target).models.default_parallel == "auto"


def test_a_config_carrying_the_removed_max_loaded_key_still_loads() -> None:
    """P8: the key is gone, but an existing config.yaml must not break."""
    config = Config(models={"max_loaded": 4, "default_ctx": 32768})
    assert not hasattr(config.models, "max_loaded")
    assert config.models.default_ctx == 32768


def test_excluded_devices_and_reserved_mb_are_validated() -> None:
    assert Config(planner={"excluded_devices": [3, 3, 1]}).planner.excluded_devices == [1, 3]
    assert Config(planner={"reserved_mb": {"3": 8192}}).planner.reserved_mb == {3: 8192}
    with pytest.raises(ValidationError):
        Config(planner={"excluded_devices": [-1]})
    with pytest.raises(ValidationError):
        Config(planner={"reserved_mb": {3: -1}})


def test_the_defaults_reserve_nothing() -> None:
    """Reserving VRAM is a policy decision; the shipped default must not make one."""
    config = Config()
    assert config.planner.excluded_devices == []
    assert config.planner.reserved_mb == {}


# ---------------------------------------------------------------------------
# Engine pinned-tag drift (P6)
# ---------------------------------------------------------------------------


def engine_manager(tmp_path: Path, *, pinned: str, active: str | None) -> EngineManager:
    config = Config(data_dir=tmp_path)
    config.engine.pinned_tag = pinned
    config.engines_dir.mkdir(parents=True, exist_ok=True)
    if active is not None:
        (config.engines_dir / "active.json").write_text(
            json.dumps({"tag": active}), encoding="utf-8"
        )
    return EngineManager(config)


def test_pinned_tag_drift_is_reported(tmp_path: Path) -> None:
    """active.json wins, so a config that disagrees is silently misleading."""
    mgr = engine_manager(tmp_path, pinned="b10441", active="b10425")
    assert mgr.check_pinned_tag() == "b10425"


def test_matching_tags_are_silent(tmp_path: Path) -> None:
    mgr = engine_manager(tmp_path, pinned="b10425", active="b10425")
    assert mgr.check_pinned_tag() is None


def test_a_missing_active_json_is_not_drift(tmp_path: Path) -> None:
    """Nothing has been activated yet; the pinned tag is the whole truth."""
    mgr = engine_manager(tmp_path, pinned="b10425", active=None)
    assert mgr.check_pinned_tag() is None


def test_the_shipped_config_pins_the_engine_that_actually_runs() -> None:
    """The code default must not drift from the engine every load uses."""
    assert Config().engine.pinned_tag == "b10425"


# ---------------------------------------------------------------------------
# Calibration guards (P3/P4)
# ---------------------------------------------------------------------------


def observation(*, note: str | None, shortfall: float = 0.5) -> dict[str, object]:
    weights = 10 * GB_DEC
    predicted = 12 * GB_DEC
    return {
        "model_id": "test/model",
        "predicted_bytes": predicted,
        "actual_bytes": int(predicted + weights * shortfall),
        "weights_bytes": weights,
        "ok": True,
        "note": note,
    }


def test_only_per_pid_observations_are_trusted() -> None:
    rows = [observation(note=None) for _ in range(10)]
    rows += [observation(note=OBSERVATION_NOTE_PER_PID) for _ in range(3)]
    assert len(clean_observations(rows)) == 3


def test_too_little_clean_data_changes_nothing() -> None:
    rows = [observation(note=OBSERVATION_NOTE_PER_PID) for _ in range(CALIBRATION_MIN_ROWS - 1)]
    assert calibrated_overhead_fraction(rows, current=0.06) is None


def test_contaminated_history_alone_changes_nothing() -> None:
    """The pre-fix rows are whole-device sums; they would peg the factor high."""
    rows = [observation(note=None, shortfall=3.0) for _ in range(200)]
    assert calibrated_overhead_fraction(rows, current=0.06) is None


def test_a_wild_suggestion_is_clamped_to_the_ceiling() -> None:
    rows = [observation(note=OBSERVATION_NOTE_PER_PID, shortfall=5.0) for _ in range(20)]
    tuned = calibrated_overhead_fraction(rows, current=0.06)
    assert tuned == OVERHEAD_FRACTION_MAX


def test_a_modest_shortfall_raises_the_factor_within_the_clamp() -> None:
    rows = [observation(note=OBSERVATION_NOTE_PER_PID, shortfall=0.02) for _ in range(20)]
    tuned = calibrated_overhead_fraction(rows, current=0.06)
    assert tuned is not None
    assert OVERHEAD_FRACTION_MIN <= tuned <= OVERHEAD_FRACTION_MAX
    assert tuned > 0.06


def test_calibrate_applies_the_value_to_the_live_config() -> None:
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2())
    rows = [observation(note=OBSERVATION_NOTE_PER_PID, shortfall=5.0) for _ in range(20)]
    assert planner.calibrate(rows) == OVERHEAD_FRACTION_MAX
    assert config.planner.compute_overhead_fraction == OVERHEAD_FRACTION_MAX
    # Idempotent: a second pass with the same data must not keep ratcheting.
    assert planner.calibrate(rows) is None


def test_calibrate_leaves_the_config_alone_without_clean_data() -> None:
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2())
    before = config.planner.compute_overhead_fraction
    assert planner.calibrate([observation(note=None) for _ in range(50)]) is None
    assert config.planner.compute_overhead_fraction == before


def recording_manager(probe: object) -> tuple[object, list[dict[str, object]]]:
    from studioforge.core.manager import ModelManager

    seen: list[dict[str, object]] = []
    planner = Planner(make_config(), probe, observation_sink=seen.append)  # type: ignore[arg-type]
    manager = ModelManager(
        make_config(),
        registry=None,  # type: ignore[arg-type]
        planner=planner,
        supervisor=None,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    return manager, seen


def test_the_observation_measures_our_child_not_the_whole_card() -> None:
    """P3: whole-device ``used_bytes`` answered a question nobody asked.

    The card here holds 28 GiB: 9 of ours, 19 belonging to something else on
    the box. Only the 9 is evidence about our estimate. The historical rows
    recorded 28 and produced a median actual/predicted ratio of 2.97 against a
    documented 0.81-1.23.
    """
    from studioforge.core.gpu import FakeGpuProbe
    from studioforge.types import GB, GpuInfo, InstanceInfo, VramEstimate, VramProcess

    probe = FakeGpuProbe(
        [
            GpuInfo(
                index=0,
                name="GPU0",
                total_bytes=32 * GB,
                free_bytes=4 * GB,
                used_bytes=28 * GB,
                compute_capability=(12, 0),
            )
        ]
    )
    probe.set_processes(
        [
            VramProcess(gpu_index=0, pid=4242, name="llama-server.exe", used_bytes=9 * GB),
            VramProcess(gpu_index=0, pid=99, name="ComfyUI.exe", used_bytes=19 * GB),
            # Another of our children, on a device this plan does not use.
            VramProcess(gpu_index=1, pid=7777, name="llama-server.exe", used_bytes=5 * GB),
        ]
    )
    manager, seen = recording_manager(probe)
    plan = LoadPlan(
        model_id="test/model",
        devices=[0],
        ctx_size=8192,
        estimate=VramEstimate(weights_bytes=8 * GB),
    )
    instance = InstanceInfo(model_id="test/model", state="ready", pid=4242, plan=plan)

    manager._record_actual_vram(make_record(), plan, instance)  # type: ignore[attr-defined]

    assert len(seen) == 1
    assert seen[0]["actual_bytes"] == 9 * GB
    assert seen[0]["note"] == OBSERVATION_NOTE_PER_PID


def test_no_observation_is_recorded_when_nvml_cannot_attribute() -> None:
    """No data beats data that quietly means something else."""
    from studioforge.core.gpu import FakeGpuProbe
    from studioforge.types import GB, GpuInfo, InstanceInfo

    probe = FakeGpuProbe(
        [
            GpuInfo(
                index=0,
                name="GPU0",
                total_bytes=32 * GB,
                free_bytes=4 * GB,
                used_bytes=28 * GB,
                compute_capability=(12, 0),
            )
        ]
    )
    probe.set_processes([])  # containers / WSL / MIG
    manager, seen = recording_manager(probe)
    plan = LoadPlan(model_id="test/model", devices=[0], ctx_size=8192)
    instance = InstanceInfo(model_id="test/model", state="ready", pid=4242, plan=plan)

    manager._record_actual_vram(make_record(), plan, instance)  # type: ignore[attr-defined]

    assert seen == []
