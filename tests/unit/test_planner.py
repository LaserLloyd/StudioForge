"""Unit tests for the VRAM planner.

Uses a self-contained fake probe rather than importing one, so these tests pin
the planner's behaviour independently of the GPU module's implementation.
"""

from __future__ import annotations

import pytest

from studioforge.config import Config
from studioforge.core.planner import (
    Planner,
    PlannerError,
    estimate_kv_bytes,
    max_ctx_for_budget,
    suggest_overhead_fraction,
)
from studioforge.types import (
    GB,
    MB,
    AdapterRecord,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
    VramEstimate,
)


class StubProbe:
    """Minimal GpuProbe: a fixed GPU list with mutable free memory."""

    backend = "fake"

    def __init__(self, gpus: list[GpuInfo]) -> None:
        self._gpus = {g.index: g.model_copy(deep=True) for g in gpus}

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [self._gpus[i].model_copy(deep=True) for i in sorted(self._gpus)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        gpu = self._gpus.get(index)
        return gpu.model_copy(deep=True) if gpu else None

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return (13, 3)

    def shutdown(self) -> None:
        return None

    def set_free(self, index: int, free_bytes: int) -> None:
        gpu = self._gpus[index]
        gpu.free_bytes = free_bytes
        gpu.used_bytes = gpu.total_bytes - free_bytes


def gpu(index: int, total_gib: float, free_gib: float, cc: tuple[int, int]) -> GpuInfo:
    total = int(total_gib * GB)
    free = int(free_gib * GB)
    return GpuInfo(
        index=index,
        name=f"FakeGPU{index}",
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free,
        compute_capability=cc,
    )


def rig_5090x2_3090x2(free_gib: float = 31.0) -> StubProbe:
    """The real target box: 2x RTX 5090 (cc 12.0) + 2x RTX 3090 (cc 8.6)."""
    return StubProbe(
        [
            gpu(0, 31.84, free_gib, (12, 0)),
            gpu(1, 31.84, free_gib, (12, 0)),
            gpu(2, 24.0, min(free_gib, 23.5), (8, 6)),
            gpu(3, 24.0, min(free_gib, 23.5), (8, 6)),
        ]
    )


def make_meta(
    *,
    n_layer: int = 32,
    n_head: int = 32,
    n_head_kv: int = 8,
    n_embd: int = 4096,
    tensor_bytes: int = 8 * GB,
    n_ctx_train: int = 32768,
    key_length: int = 0,
    value_length: int = 0,
) -> GgufMeta:
    return GgufMeta(
        architecture="llama",
        n_layer=n_layer,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_embd=n_embd,
        n_ctx_train=n_ctx_train,
        n_embd_head_k=key_length,
        n_embd_head_v=value_length,
        tensor_bytes=tensor_bytes,
        quant_label="Q4_K_M",
    )


def make_record(
    model_id: str = "test/model",
    *,
    meta: GgufMeta | None = None,
    settings: ModelSettings | None = None,
    vision: bool = False,
    mmproj_path=None,
    mmproj_bytes: int = 0,
    size_bytes: int = 8 * GB,
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path="/models/test/model.gguf",
        size_bytes=size_bytes,
        meta=meta if meta is not None else make_meta(),
        settings=settings or ModelSettings(),
        capabilities=ModelCapabilities(vision=vision),
        mmproj_path=mmproj_path,
        mmproj_bytes=mmproj_bytes,
    )


def make_config(**planner_overrides) -> Config:
    config = Config(data_dir="/tmp/sf-test")
    for key, value in planner_overrides.items():
        setattr(config.planner, key, value)
    return config


# ---------------------------------------------------------------------------
# KV cache math
# ---------------------------------------------------------------------------


def test_kv_bytes_exact_gqa_math() -> None:
    """32 layers, 8 KV heads, head_dim 128, ctx 8192, f16 -> exactly 1 GiB."""
    kv = estimate_kv_bytes(
        n_layer=32,
        n_head_kv=8,
        head_dim_k=128,
        head_dim_v=128,
        ctx_total=8192,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    assert kv == 1 * GB


def test_kv_bytes_gqa_is_eight_times_smaller_than_mha() -> None:
    gqa = estimate_kv_bytes(
        n_layer=32,
        n_head_kv=8,
        head_dim_k=128,
        head_dim_v=128,
        ctx_total=8192,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    mha = estimate_kv_bytes(
        n_layer=32,
        n_head_kv=64,
        head_dim_k=128,
        head_dim_v=128,
        ctx_total=8192,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    assert mha == gqa * 8


def test_kv_quantized_cache_is_cheaper() -> None:
    args = {
        "n_layer": 32,
        "n_head_kv": 8,
        "head_dim_k": 128,
        "head_dim_v": 128,
        "ctx_total": 8192,
    }
    f16 = estimate_kv_bytes(**args, kv_type_k="f16", kv_type_v="f16")
    q8 = estimate_kv_bytes(**args, kv_type_k="q8_0", kv_type_v="q8_0")
    q4 = estimate_kv_bytes(**args, kv_type_k="q4_0", kv_type_v="q4_0")
    assert q8 == pytest.approx(f16 * (34 / 32) / 2, rel=1e-6)
    assert q4 < q8 < f16


def test_kv_asymmetric_head_dims() -> None:
    """Different K/V head dims must be sized separately, not symmetrically."""
    kv = estimate_kv_bytes(
        n_layer=10,
        n_head_kv=4,
        head_dim_k=256,
        head_dim_v=128,
        ctx_total=1000,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    expected = 10 * 1000 * (4 * 256 * 2 + 4 * 128 * 2)
    assert kv == expected


def test_unknown_kv_type_assumes_f16_not_zero() -> None:
    """Under-estimating the KV cache is what causes OOM, so degrade upward."""
    kv = estimate_kv_bytes(
        n_layer=8,
        n_head_kv=8,
        head_dim_k=64,
        head_dim_v=64,
        ctx_total=512,
        kv_type_k="q3_k_nonsense",
        kv_type_v="q3_k_nonsense",
    )
    assert kv == 8 * 512 * (8 * 64 * 2 + 8 * 64 * 2)


def test_max_ctx_for_budget_roundtrips() -> None:
    args = {
        "n_layer": 32,
        "n_head_kv": 8,
        "head_dim_k": 128,
        "head_dim_v": 128,
        "kv_type_k": "f16",
        "kv_type_v": "f16",
    }
    ctx = max_ctx_for_budget(budget_bytes=1 * GB, **args)
    assert ctx == 8192
    assert estimate_kv_bytes(**args, ctx_total=ctx) <= 1 * GB


def test_max_ctx_for_budget_divides_by_slots() -> None:
    """With 4 slots the per-slot context is a quarter of the total (D4)."""
    args = {
        "n_layer": 32,
        "n_head_kv": 8,
        "head_dim_k": 128,
        "head_dim_v": 128,
        "kv_type_k": "f16",
        "kv_type_v": "f16",
    }
    assert max_ctx_for_budget(budget_bytes=1 * GB, parallel=4, **args) == 2048


def test_max_ctx_for_budget_zero_budget() -> None:
    assert (
        max_ctx_for_budget(
            budget_bytes=0,
            n_layer=32,
            n_head_kv=8,
            head_dim_k=128,
            head_dim_v=128,
            kv_type_k="f16",
            kv_type_v="f16",
        )
        == 0
    )


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------


def test_estimate_full_breakdown() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record()
    est = planner.estimate(
        record,
        ctx_size=8192,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    assert est.weights_bytes == 8 * GB
    assert est.kv_bytes == 1 * GB
    assert est.compute_bytes == int(8 * GB * 0.06)  # fraction beats the 400MB floor
    assert est.cuda_context_bytes == 300 * MB
    assert est.total_bytes == (
        est.weights_bytes + est.kv_bytes + est.compute_bytes + est.cuda_context_bytes
    )


def test_estimate_compute_floor_applies_to_tiny_models() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=500 * MB))
    est = planner.estimate(
        record, ctx_size=4096, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert est.compute_bytes == 400 * MB  # floor, not 6% of 500MB


def test_estimate_ctx_total_is_ctx_times_parallel() -> None:
    """DECISIONS.md D4: --ctx-size is the TOTAL budget across slots."""
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record()
    one = planner.estimate(
        record, ctx_size=4096, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    four = planner.estimate(
        record, ctx_size=4096, parallel=4, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert four.kv_bytes == one.kv_bytes * 4


def test_estimate_cuda_context_scales_with_device_count() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record()
    two = planner.estimate(
        record,
        ctx_size=4096,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=2,
    )
    assert two.cuda_context_bytes == 2 * 300 * MB


def test_estimate_counts_mmproj(tmp_path) -> None:
    mmproj = tmp_path / "mmproj-F16.gguf"
    mmproj.write_bytes(b"\0" * (700 * MB))
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(vision=True, mmproj_path=mmproj)
    est = planner.estimate(
        record, ctx_size=8192, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert est.mmproj_bytes == 700 * MB
    assert est.mmproj_compute_bytes == 512 * MB


def test_estimate_counts_adapters_and_draft() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record()
    draft = make_record("test/draft", meta=make_meta(n_layer=8, tensor_bytes=500 * MB))
    adapters = [AdapterRecord(id="a", name="a", path="/a.gguf", size_bytes=120 * MB)]
    est = planner.estimate(
        record,
        ctx_size=8192,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        draft=draft,
        adapters=adapters,
    )
    assert est.adapter_bytes == 120 * MB
    assert est.draft_weights_bytes == 500 * MB
    assert est.draft_kv_bytes > 0


def test_estimate_requires_metadata() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record()
    record.meta = None
    with pytest.raises(PlannerError):
        planner.estimate(
            record, ctx_size=4096, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
        )


def test_key_length_override_used_for_head_dim() -> None:
    """An explicit attention.key_length must win over n_embd/n_head."""
    meta = make_meta(n_embd=4096, n_head=32, key_length=256, value_length=256)
    assert meta.head_dim_k == 256
    planner = Planner(make_config(), rig_5090x2_3090x2())
    est = planner.estimate(
        make_record(meta=meta),
        ctx_size=8192,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
    )
    assert est.kv_bytes == 2 * GB  # twice the 128-head-dim case


# ---------------------------------------------------------------------------
# Placement policy
# ---------------------------------------------------------------------------


def test_prefers_single_fastest_gpu() -> None:
    """Fits on one card -> one card, no tensor split, and the fastest one."""
    planner = Planner(make_config(), rig_5090x2_3090x2())
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]
    assert plan.fits_single_gpu
    assert plan.tensor_split is None
    assert plan.split_mode == "none"
    assert plan.main_gpu == 0


def test_prefers_higher_compute_capability_over_more_free_vram() -> None:
    """A 3090 with more free VRAM must still lose to a 5090 that fits."""
    # The ~9.8 GiB estimate fits on both cards; the 3090 has twice the room.
    probe = StubProbe(
        [
            gpu(0, 31.84, 14.0, (12, 0)),  # 5090, enough room (10.8 GiB usable)
            gpu(2, 24.0, 23.5, (8, 6)),  # 3090, much more room (21.1 GiB usable)
        ]
    )
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]


def test_falls_back_to_second_gpu_when_first_is_full() -> None:
    probe = rig_5090x2_3090x2()
    probe.set_free(0, int(2 * GB))
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1]


def test_splits_across_two_gpus_when_no_single_card_fits() -> None:
    """40 GiB model on 32 GiB cards -> narrowest viable split."""
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=40 * GB), size_bytes=40 * GB)
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert len(plan.devices) == 2
    assert plan.split_mode == "layer"
    assert plan.tensor_split is not None
    assert sum(plan.tensor_split) == pytest.approx(1.0, abs=0.01)
    assert any("split across 2 GPUs" in n for n in plan.notes)


def test_tensor_split_is_proportional_to_usable_vram() -> None:
    probe = StubProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 32.0, 15.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=40 * GB), size_bytes=40 * GB)
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert plan.tensor_split is not None
    # 30:15 usable -> roughly 2:1
    assert plan.tensor_split[0] / plan.tensor_split[1] == pytest.approx(2.0, rel=0.05)


def test_uses_four_gpus_only_when_needed() -> None:
    """~86 GiB needed; the three roomiest cards offer ~77 GiB usable, so all 4."""
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=80 * GB), size_bytes=80 * GB)
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert len(plan.devices) == 4


def test_oversized_model_is_rejected_not_split_forever() -> None:
    """Bigger than the whole box -> a clean rejection with the real numbers."""
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=100 * GB), size_bytes=100 * GB)
    result = planner.plan_load(record, ctx_size=4096)
    assert isinstance(result, LoadRejected)
    assert result.required_bytes > result.available_bytes
    assert any("smaller quantization" in s for s in result.suggestions)


def test_headroom_is_respected() -> None:
    """A model that fits in raw free VRAM but not inside the headroom is refused."""
    probe = StubProbe([gpu(0, 32.0, 10.0, (12, 0))])
    config = make_config(headroom_fraction=0.10)
    planner = Planner(config, probe)
    # usable = 10 GiB free - 3.2 GiB headroom = 6.8 GiB
    record = make_record(meta=make_meta(tensor_bytes=8 * GB), size_bytes=8 * GB)
    result = planner.plan_load(record, ctx_size=2048)
    assert isinstance(result, LoadRejected)
    # With no headroom the same load succeeds.
    planner_nohead = Planner(make_config(headroom_fraction=0.0), probe)
    assert isinstance(planner_nohead.plan_load(record, ctx_size=2048), LoadPlan)


def test_usable_bytes_headroom_is_fraction_of_total() -> None:
    planner = Planner(make_config(headroom_fraction=0.10), rig_5090x2_3090x2())
    g = gpu(0, 32.0, 20.0, (12, 0))
    assert planner.usable_bytes(g) == int(20 * GB) - int(32 * GB * 0.10)


def test_no_gpus_is_a_gpu_only_rejection() -> None:
    planner = Planner(make_config(), StubProbe([]))
    result = planner.plan_load(make_record())
    assert isinstance(result, LoadRejected)
    assert "GPU-only" in result.reason
    assert "never" in result.reason


# ---------------------------------------------------------------------------
# Device override
# ---------------------------------------------------------------------------


def test_device_override_is_honoured() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(settings=ModelSettings(device_override=[2, 3]))
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [2, 3]


def test_device_override_unknown_gpu_rejected() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(settings=ModelSettings(device_override=[7]))
    result = planner.plan_load(record)
    assert isinstance(result, LoadRejected)
    assert "do not exist" in result.reason
    assert any("device override" in s for s in result.suggestions)


def test_device_override_too_small_suggests_clearing_it() -> None:
    probe = rig_5090x2_3090x2()
    probe.set_free(2, int(1 * GB))
    planner = Planner(make_config(), probe)
    record = make_record(settings=ModelSettings(device_override=[2]))
    result = planner.plan_load(record, ctx_size=8192)
    assert isinstance(result, LoadRejected)
    assert any("device override" in s for s in result.suggestions)


# ---------------------------------------------------------------------------
# Rejection quality
# ---------------------------------------------------------------------------


def test_rejection_computes_a_context_that_would_fit() -> None:
    probe = StubProbe([gpu(0, 32.0, 12.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    result = planner.plan_load(record, ctx_size=131072)
    assert isinstance(result, LoadRejected)
    assert result.max_ctx_that_fits is not None
    assert 0 < result.max_ctx_that_fits < 131072
    assert any("reduce context" in s for s in result.suggestions)
    # And that suggested context must actually fit.
    replan = planner.plan_load(record, ctx_size=result.max_ctx_that_fits)
    assert isinstance(replan, LoadPlan)


def test_rejection_suggests_smaller_quant_when_weights_alone_too_big() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=200 * GB), size_bytes=200 * GB)
    record.publisher = "bartowski"
    record.repo = "Behemoth-123B-GGUF"
    result = planner.plan_load(record, ctx_size=4096)
    assert isinstance(result, LoadRejected)
    assert any("smaller quantization" in s for s in result.suggestions)
    assert any("bartowski/Behemoth-123B-GGUF" in s for s in result.suggestions)


def test_rejection_suggests_q8_kv_when_kv_dominates() -> None:
    probe = StubProbe([gpu(0, 32.0, 20.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    # Small weights, enormous context -> KV is the problem.
    record = make_record(meta=make_meta(tensor_bytes=2 * GB), size_bytes=2 * GB)
    # Pin f16 explicitly: with kv_cache_type="auto" the planner would simply
    # pick a smaller cache and succeed, which is the new intended behaviour --
    # what this test guards is that when a cache type IS pinned and the KV is
    # what does not fit, the rejection says so and names the cheaper cache.
    result = planner.plan_load(record, ctx_size=262144, kv_cache_type="f16")
    assert isinstance(result, LoadRejected)
    assert any("q8_0" in s for s in result.suggestions)


def test_auto_kv_downgrades_instead_of_rejecting() -> None:
    """The other half of the same situation: "auto" should just fit it."""
    probe = StubProbe([gpu(0, 32.0, 20.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=2 * GB), size_bytes=2 * GB)

    result = planner.plan_load(record, ctx_size=262144, kv_cache_type="auto")

    assert isinstance(result, LoadPlan)
    assert result.ctx_size == 262144, "an explicit context must still be exact"
    assert result.kv_cache_type in ("q8_0", "q4_0")
    # Never a type the prebuilt CUDA engines cannot run under flash attention.
    assert result.kv_cache_type not in ("q5_1", "q5_0", "q4_1")


def test_rejection_says_dropping_the_draft_would_help() -> None:
    probe = StubProbe([gpu(0, 32.0, 11.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    draft = make_record("test/draft", meta=make_meta(n_layer=8, tensor_bytes=3 * GB))
    result = planner.plan_load(record, ctx_size=4096, draft=draft)
    assert isinstance(result, LoadRejected)
    assert any("remove the draft model" in s for s in result.suggestions)
    assert any("test/draft" in s for s in result.suggestions)


def test_rejection_message_and_numbers_present() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(meta=make_meta(tensor_bytes=500 * GB), size_bytes=500 * GB)
    result = planner.plan_load(record)
    assert isinstance(result, LoadRejected)
    assert result.required_bytes > result.available_bytes > 0
    assert result.per_gpu_free
    msg = result.message()
    assert "Cannot load" in msg and "GiB" in msg and "Suggestions" in msg
    assert result.suggestions  # never empty


def test_rejection_without_metadata_still_reports_a_size() -> None:
    planner = Planner(make_config(), StubProbe([gpu(0, 8.0, 1.0, (8, 6))]))
    record = make_record(size_bytes=40 * GB)
    record.meta = None
    result = planner.plan_load(record)
    assert isinstance(result, LoadRejected)
    assert result.estimate.weights_bytes == 40 * GB


# ---------------------------------------------------------------------------
# Memory guard / eviction
# ---------------------------------------------------------------------------


def loaded_instance(
    model_id: str,
    *,
    device: int,
    bytes_held: int,
    ttl_s: int | None = 1800,
    active: int = 0,
    last_activity: float = 100.0,
) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18100,
        ttl_s=ttl_s,
        active_requests=active,
        last_activity_at=last_activity,
        plan=LoadPlan(
            model_id=model_id,
            devices=[device],
            per_gpu_bytes={device: bytes_held},
            estimate=VramEstimate(weights_bytes=bytes_held),
        ),
    )


def test_evicts_lru_unpinned_model_to_make_room() -> None:
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    loaded = [
        loaded_instance("old/model", device=0, bytes_held=int(10 * GB), last_activity=10.0),
        loaded_instance("new/model", device=0, bytes_held=int(10 * GB), last_activity=999.0),
    ]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    plan = planner.plan_load(record, ctx_size=4096, loaded=loaded)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["old/model"]  # LRU only, not both
    assert any("evicting least-recently-used" in n for n in plan.notes)


def test_never_evicts_pinned_models() -> None:
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    loaded = [loaded_instance("pinned/model", device=0, bytes_held=int(20 * GB), ttl_s=0)]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    result = planner.plan_load(record, ctx_size=4096, loaded=loaded)
    assert isinstance(result, LoadRejected)
    assert any("unpin" in s and "pinned/model" in s for s in result.suggestions)


def test_never_evicts_a_model_serving_a_request() -> None:
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    loaded = [loaded_instance("busy/model", device=0, bytes_held=int(20 * GB), active=3)]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    result = planner.plan_load(record, ctx_size=4096, loaded=loaded)
    assert isinstance(result, LoadRejected)


def test_reject_policy_does_not_evict_and_says_so() -> None:
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="reject"), probe)
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    result = planner.plan_load(record, ctx_size=4096, loaded=loaded)
    assert isinstance(result, LoadRejected)
    assert any("on_insufficient" in s for s in result.suggestions)


def test_allow_evict_argument_overrides_config_policy() -> None:
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="reject"), probe)
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)
    plan = planner.plan_load(record, ctx_size=4096, loaded=loaded, allow_evict=True)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["idle/model"]


# ---------------------------------------------------------------------------
# Vision awareness
# ---------------------------------------------------------------------------


def test_vision_context_budget_warning(tmp_path) -> None:
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"\0" * (10 * MB))
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(vision=True, mmproj_path=mmproj)
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert any("tokens/image" in n for n in plan.notes)


def test_vision_no_warning_with_generous_context(tmp_path) -> None:
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"\0" * (10 * MB))
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(vision=True, mmproj_path=mmproj)
    plan = planner.plan_load(record, ctx_size=32768)
    assert isinstance(plan, LoadPlan)
    assert not any("tokens/image" in n for n in plan.notes)


def test_image_tokens_from_mmproj_metadata(tmp_path) -> None:
    """Patch counts from mmproj metadata beat the conservative default."""
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"\0" * (10 * MB))
    meta = make_meta()
    meta.vision_image_size = 896
    meta.vision_patch_size = 14
    planner = Planner(make_config(), rig_5090x2_3090x2())
    record = make_record(
        meta=meta, vision=True, mmproj_path="/models/test/mmproj.gguf", mmproj_bytes=10 * MB
    )
    assert planner._image_tokens(record) == (896 // 14) ** 2  # 4096


def test_non_vision_model_gets_no_mmproj_charge() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2())
    est = planner.estimate(
        make_record(),
        ctx_size=8192,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
    )
    assert est.mmproj_bytes == 0
    assert est.mmproj_compute_bytes == 0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_observe_forwards_to_sink() -> None:
    captured: list[dict[str, object]] = []
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_sink=captured.append)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    planner.observe(model_id="test/model", plan=plan, actual_bytes=11 * GB)
    assert len(captured) == 1
    assert captured[0]["model_id"] == "test/model"
    assert captured[0]["actual_bytes"] == 11 * GB
    assert captured[0]["ok"] == 1


def test_suggest_overhead_fraction_needs_data() -> None:
    assert suggest_overhead_fraction([], current=0.06) == 0.06
    one_sample = [{"predicted_bytes": 1, "actual_bytes": 2}]
    assert suggest_overhead_fraction(one_sample, current=0.06) == 0.06


def test_suggest_overhead_fraction_covers_worst_case() -> None:
    """Uses the worst under-estimate, not the average: the factor prevents OOM."""
    obs = [
        {"predicted_bytes": 10 * GB, "actual_bytes": 10 * GB + 100 * MB, "weights_bytes": 8 * GB},
        {"predicted_bytes": 10 * GB, "actual_bytes": 10 * GB + 800 * MB, "weights_bytes": 8 * GB},
        {"predicted_bytes": 10 * GB, "actual_bytes": 10 * GB + 200 * MB, "weights_bytes": 8 * GB},
    ]
    suggested = suggest_overhead_fraction(obs, current=0.06)
    assert suggested > 0.06
    # worst shortfall 800MB / 8GB = 0.0977 -> 0.06 + 0.0977 ~= 0.158
    assert 0.15 <= suggested <= 0.17


def test_suggest_overhead_fraction_ignores_over_estimates() -> None:
    obs = [
        {"predicted_bytes": 10 * GB, "actual_bytes": 9 * GB, "weights_bytes": 8 * GB}
        for _ in range(5)
    ]
    assert suggest_overhead_fraction(obs, current=0.06) == 0.06


def test_suggest_overhead_fraction_is_capped() -> None:
    obs = [
        {"predicted_bytes": 1 * GB, "actual_bytes": 50 * GB, "weights_bytes": 1 * GB}
        for _ in range(5)
    ]
    assert suggest_overhead_fraction(obs, current=0.06) == 0.5


# ---------------------------------------------------------------------------
# Quantization / hardware affinity (FP4 on Blackwell)
# ---------------------------------------------------------------------------


def nvfp4_record(size_gib: float = 18.0) -> ModelRecord:
    meta = make_meta(tensor_bytes=int(size_gib * GB))
    meta.quant_label = "NVFP4"
    record = make_record(meta=meta, size_bytes=int(size_gib * GB))
    record.quant = "NVFP4"
    return record


def test_fp4_prefers_blackwell_when_both_fit() -> None:
    """NVFP4 gets native FP4 acceleration only on sm_120, so steer it there."""
    probe = StubProbe([gpu(2, 24.0, 23.5, (8, 6)), gpu(0, 31.84, 31.0, (12, 0))])
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(nvfp4_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]


def test_fp4_falls_back_to_ampere_rather_than_failing() -> None:
    """The 3090s stay fully usable: 'prefer' must never become a refusal."""
    probe = StubProbe([gpu(0, 31.84, 2.0, (12, 0)), gpu(2, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(nvfp4_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [2]
    assert any("without native acceleration" in n for n in plan.notes), plan.notes


def test_fp4_require_mode_refuses_ampere() -> None:
    from studioforge.config import QuantAffinity

    probe = StubProbe([gpu(2, 24.0, 23.5, (8, 6)), gpu(3, 24.0, 23.5, (8, 6))])
    config = make_config()
    config.planner.quant_affinity = {
        "NVFP4": QuantAffinity(min_compute_capability="12.0", mode="require")
    }
    planner = Planner(config, probe)
    result = planner.plan_load(nvfp4_record(), ctx_size=8192)
    assert isinstance(result, LoadRejected)
    assert any("compute capability 12.0" in s for s in result.suggestions)
    assert any("'prefer'" in s for s in result.suggestions)


def test_fp4_require_mode_still_uses_blackwell() -> None:
    from studioforge.config import QuantAffinity

    probe = rig_5090x2_3090x2()
    config = make_config()
    config.planner.quant_affinity = {
        "NVFP4": QuantAffinity(min_compute_capability="12.0", mode="require")
    }
    planner = Planner(config, probe)
    plan = planner.plan_load(nvfp4_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]
    assert all(d in (0, 1) for d in plan.devices)


def test_non_fp4_model_is_unaffected_by_affinity() -> None:
    """A plain Q4_K_M must be free to use the 3090s with no note."""
    probe = StubProbe([gpu(2, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [2]
    assert not any("native acceleration" in n for n in plan.notes)


def test_affinity_matches_mxfp4_too() -> None:
    meta = make_meta(tensor_bytes=10 * GB)
    meta.quant_label = "MXFP4"
    record = make_record(meta=meta, size_bytes=10 * GB)
    record.quant = "MXFP4"
    probe = StubProbe([gpu(2, 24.0, 23.5, (8, 6)), gpu(0, 31.84, 31.0, (12, 0))])
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]


def test_affinity_can_be_disabled_entirely() -> None:
    probe = StubProbe([gpu(2, 24.0, 23.5, (8, 6)), gpu(0, 31.84, 31.0, (12, 0))])
    config = make_config()
    config.planner.quant_affinity = {}
    planner = Planner(config, probe)
    plan = planner.plan_load(nvfp4_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    # With no affinity table, ordering is pure compute-capability preference,
    # which still puts the 5090 first -- but no affinity note is attached.
    assert not any("native acceleration" in n for n in plan.notes)


def test_device_override_beats_affinity() -> None:
    """An explicit device override is the user's decision and wins."""
    probe = rig_5090x2_3090x2()
    record = nvfp4_record()
    record.settings = ModelSettings(device_override=[3])
    planner = Planner(make_config(), probe)
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [3]


# ---------------------------------------------------------------------------
# Re-plan after eviction (DECISIONS.md D16)
# ---------------------------------------------------------------------------


def test_eviction_replans_the_ladder_instead_of_landing_on_the_floor() -> None:
    """Having decided to evict, take the best context the freed VRAM allows.

    The 12:03 defect: 79,832 MB available, 19,423 MB reclaimable. The floor
    rung decided the eviction and then loaded AT the floor, ignoring that the
    combined 99,255 MB reached a far larger window for the identical cost.
    """
    probe = StubProbe([gpu(0, 32.0, 4.5, (12, 0))])
    config = make_config(headroom_fraction=0.0, on_insufficient="evict")
    config.models.default_ctx = 8192
    config.models.target_ctx = 65536
    planner = Planner(config, probe)
    # 20 GiB held by an idle model, 4.5 GiB free. Not even the 8192 floor fits
    # in what is free, so eviction is genuinely forced -- and once it is, the
    # freed 20 GiB reaches the top of the ladder rather than the bottom.
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=4 * GB, n_ctx_train=131072))

    plan = planner.plan_load(record, loaded=loaded)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["idle/model"]
    # The point of the fix: the aim, not the 8192 floor.
    assert plan.ctx_size == 65536
    # And full-quality KV, because the freed VRAM affords it.
    assert plan.kv_cache_type == "f16"
    assert any("re-planned after eviction" in n for n in plan.notes)


def test_a_roomier_context_never_causes_an_eviction() -> None:
    """D14's invariant survives D16: eviction is decided at the FLOOR only.

    A model that fits at the floor without evicting must not evict merely to
    reach a larger window -- that would make someone else's model the price of
    our nicety.
    """
    probe = StubProbe([gpu(0, 32.0, 12.0, (12, 0))])
    config = make_config(headroom_fraction=0.0, on_insufficient="evict")
    config.models.default_ctx = 8192
    config.models.target_ctx = 131072
    planner = Planner(config, probe)
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=4 * GB, n_ctx_train=131072))

    plan = planner.plan_load(record, loaded=loaded)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == []


def test_replan_after_eviction_honours_an_explicit_context_verbatim() -> None:
    """An explicit ctx_size is a one-rung ladder; both passes must respect it."""
    probe = StubProbe([gpu(0, 32.0, 4.0, (12, 0))])
    config = make_config(headroom_fraction=0.0, on_insufficient="evict")
    config.models.default_ctx = 8192
    config.models.target_ctx = 131072
    planner = Planner(config, probe)
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=9 * GB, n_ctx_train=131072))

    plan = planner.plan_load(record, ctx_size=4096, loaded=loaded)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["idle/model"]
    assert plan.ctx_size == 4096  # never a helpful upgrade


# ---------------------------------------------------------------------------
# Excluded devices / reserved VRAM (multi-tenant GPU sharing)
# ---------------------------------------------------------------------------


def test_excluded_device_is_never_chosen_automatically() -> None:
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0)), gpu(1, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    config.planner.excluded_devices = [0]
    planner = Planner(config, probe)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1]


def test_excluding_every_device_rejects_and_names_the_knob() -> None:
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    config.planner.excluded_devices = [0]
    planner = Planner(config, probe)
    result = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(result, LoadRejected)
    assert any("excluded_devices" in s for s in result.suggestions)


def test_device_override_beats_an_exclusion() -> None:
    """Explicit beats policy: the user naming a device is the user deciding."""
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0)), gpu(1, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    config.planner.excluded_devices = [0]
    planner = Planner(config, probe)
    record = make_record(settings=ModelSettings(device_override=[0]))
    plan = planner.plan_load(record, ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]
    assert any("excluded_devices" in n for n in plan.notes)


def test_reserved_mb_is_subtracted_from_usable_capacity() -> None:
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    planner = Planner(config, probe)
    card = probe.list_gpus()[0]
    before = planner.usable_bytes(card)

    config.planner.reserved_mb = {0: 8192}
    assert planner.usable_bytes(card) == before - 8192 * MB


def test_reserved_mb_applies_even_to_a_forced_placement() -> None:
    """A reservation describes a NEIGHBOUR's memory, not our placement policy."""
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    config.planner.reserved_mb = {0: 8192}
    config.planner.excluded_devices = [0]
    planner = Planner(config, probe)
    card = probe.list_gpus()[0]
    # Exclusion is ours, so forcing ignores it; the reservation still applies.
    assert planner.usable_bytes(card) == 0
    assert planner.usable_bytes(card, forced=True) == card.free_bytes - 8192 * MB


def test_reserved_mb_can_make_a_load_not_fit() -> None:
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0))])
    config = make_config(headroom_fraction=0.0)
    config.planner.reserved_mb = {0: 28 * 1024}
    planner = Planner(config, probe)
    result = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(result, LoadRejected)
    assert any("reserved_mb" in s for s in result.suggestions)


# ---------------------------------------------------------------------------
# D40: where the bytes land -- the output layer, and the micro-batch
# ---------------------------------------------------------------------------


def _meta_with_vocab(*, tensor_bytes: int, n_embd: int = 4096, n_vocab: int = 151936) -> GgufMeta:
    meta = make_meta(tensor_bytes=tensor_bytes, n_embd=n_embd)
    return meta.model_copy(update={"n_vocab": n_vocab, "param_count": tensor_bytes * 8 // 5})


def test_output_layer_bytes_is_vocab_times_embd_at_the_file_density() -> None:
    from studioforge.core.planner import OUTPUT_LAYER_DEFAULT_BPW, output_layer_bytes

    meta = _meta_with_vocab(tensor_bytes=40 * GB)  # 5 bits per weight declared
    assert output_layer_bytes(meta) == int(151936 * 4096 * 5 / 8)
    # Without a declared parameter count the default density applies.
    undeclared = make_meta(tensor_bytes=40 * GB).model_copy(update={"n_vocab": 151936})
    assert output_layer_bytes(undeclared) == int(151936 * 4096 * OUTPUT_LAYER_DEFAULT_BPW / 8)
    # No vocabulary in the metadata: nothing to charge, nothing guessed.
    assert output_layer_bytes(make_meta()) == 0
    assert output_layer_bytes(None) == 0


def test_split_charges_the_output_layer_to_the_last_device() -> None:
    """llama.cpp puts the output layer on the last device of the list (D40).

    Measured on the scratch rig: a 0.5,0.5 split of two small models landed
    104-110 MiB more on the last card, independent of -ub; the live rig's 27B
    planned at 0.5079,0.4921 landed 0.76 GiB MORE on the card the split gave
    less to. So the plan charges it there and tilts the split away from it.
    """
    from studioforge.core.planner import output_layer_bytes

    probe = StubProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 32.0, 30.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    meta = _meta_with_vocab(tensor_bytes=40 * GB)
    record = make_record(meta=meta, size_bytes=40 * GB)
    plan = planner.plan_load(record, ctx_size=4096)
    assert isinstance(plan, LoadPlan)
    assert len(plan.devices) == 2
    first, last = plan.devices
    shift = output_layer_bytes(meta)
    assert shift > 0
    # Equal cards: the split leans toward the FIRST card (fewer blocks on the
    # one that also holds the output layer)...
    assert plan.tensor_split is not None
    assert plan.tensor_split[0] > plan.tensor_split[1]
    # ...and the last card's share is its block share PLUS the output layer,
    # so it still ends up the heavier of the two.
    body = plan.estimate.total_bytes - shift
    assert plan.per_gpu_bytes[last] == pytest.approx(body * plan.tensor_split[1] + shift, rel=0.01)
    assert plan.per_gpu_bytes[first] == pytest.approx(body * plan.tensor_split[0], rel=0.01)
    assert plan.per_gpu_bytes[last] > plan.per_gpu_bytes[first]
    assert sum(plan.per_gpu_bytes.values()) == pytest.approx(plan.estimate.total_bytes, abs=4)


def test_the_output_layer_shift_can_refuse_a_last_card_that_is_too_tight() -> None:
    """Totals that add up are not enough: the last card must hold its layer too."""
    from studioforge.core.planner import output_layer_bytes

    meta = _meta_with_vocab(tensor_bytes=40 * GB)
    shift = output_layer_bytes(meta)
    need = 40 * GB + 400 * MB  # weights + compute floor, roughly; ctx is tiny here
    # Two cards that together hold the total with a hair to spare, but the
    # second one cannot hold its proportional share PLUS the output layer.
    free0 = need // 2 + shift
    free1 = need // 2 - shift // 2
    probe = StubProbe([gpu(0, 32.0, free0 / GB, (12, 0)), gpu(1, 32.0, free1 / GB, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, cuda_context_mb=0), probe)
    record = make_record(meta=meta, size_bytes=40 * GB)
    result = planner.plan_load(record, ctx_size=256, kv_cache_type="f16")
    # Either it refuses, or every card is within its capacity with the shift applied.
    if isinstance(result, LoadPlan):
        caps = {g.index: planner.usable_bytes(g) for g in probe.list_gpus()}
        for dev, want in result.per_gpu_bytes.items():
            assert want <= caps[dev]
        assert result.per_gpu_bytes[result.devices[-1]] >= shift
    else:
        assert isinstance(result, LoadRejected)


def test_ubatch_scratch_grows_per_token_per_embd_per_device() -> None:
    from studioforge.core.planner import (
        DEFAULT_UBATCH,
        UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD,
        ubatch_scratch_bytes,
    )

    meta = make_meta(n_embd=1536)
    assert ubatch_scratch_bytes(meta, ubatch=DEFAULT_UBATCH) == 0
    assert ubatch_scratch_bytes(meta, ubatch=256) == 0
    one = ubatch_scratch_bytes(meta, ubatch=2048)
    assert one == 1536 * 1536 * UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD
    # Every device of a split holds the whole micro-batch.
    assert ubatch_scratch_bytes(meta, ubatch=2048, n_devices=2) == 2 * one
    # Measured on the 1.5B (n_embd 1536): +198 MiB single, +270 MiB per card of
    # a pair, for -ub 512 -> 2048. The charge must cover both.
    assert one >= 270 * MB


def test_estimate_is_ubatch_aware_with_the_supervisor_precedence() -> None:
    """``-ub`` reaches the estimate: per-model, else engine.ubatch_size, else 512."""
    from studioforge.core.planner import DEFAULT_UBATCH

    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2())
    record = make_record(meta=make_meta(n_embd=4096))
    base = planner.estimate(
        record, ctx_size=8192, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert planner.ubatch_for(record) == DEFAULT_UBATCH

    config.engine.ubatch_size = 2048
    assert planner.ubatch_for(record) == 2048
    global_raised = planner.estimate(
        record, ctx_size=8192, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert global_raised.compute_bytes > base.compute_bytes
    assert global_raised.total_bytes - base.total_bytes == 1536 * 4096 * 128

    pinned = make_record(meta=make_meta(n_embd=4096), settings=ModelSettings(ubatch_size=1024))
    assert planner.ubatch_for(pinned) == 1024
    per_model = planner.estimate(
        pinned, ctx_size=8192, parallel=1, kv_cache_type="f16", kv_cache_type_v="f16"
    )
    assert base.compute_bytes < per_model.compute_bytes < global_raised.compute_bytes

    # An explicit argument wins over both, and the default costs nothing extra.
    explicit = planner.estimate(
        pinned,
        ctx_size=8192,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        ubatch=DEFAULT_UBATCH,
    )
    assert explicit.compute_bytes == base.compute_bytes
