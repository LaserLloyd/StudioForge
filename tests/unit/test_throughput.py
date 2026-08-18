"""Speed estimator, metrics parsing and calibration.

The estimator turns hardware constants into tokens/second. Those constants are
nominal, so what these tests pin is not "the number is right" -- nothing here
can know that -- but that the *arithmetic* is right and lands in a defensible
band, that the shape of the answer degrades honestly when inputs are missing,
and that a real measurement always wins over a projection.

The metrics fixtures are the real exposition output of llama.cpp b10425, taken
from the running child on the reference rig.
"""

from __future__ import annotations

import pytest

from studioforge.core import throughput
from studioforge.core.throughput import (
    ESTIMATOR_VERSION,
    GPU_PERF_UNKNOWN,
    active_params,
    bits_per_weight,
    calibrate,
    confidence_for,
    estimate,
    gpu_class,
    gpu_perf_for,
    measured_for,
    parse_metrics,
    sample_between,
    total_params,
)
from studioforge.types import GB, GpuInfo

GB_DEC = 1_000_000_000
MB = 1024 * 1024


# ---------------------------------------------------------------------------
# Model + hardware fixtures (shapes measured on the reference rig)
# ---------------------------------------------------------------------------


class Meta:
    """Duck-typed GgufMeta stand-in; the estimator only reads attributes."""

    def __init__(
        self,
        *,
        param_count: int = 0,
        n_expert: int = 0,
        n_expert_used: int = 0,
        quant_label: str = "Q4_K_M",
        **dims: int,
    ) -> None:
        self.param_count = param_count
        self.n_expert = n_expert
        self.n_expert_used = n_expert_used
        self.quant_label = quant_label
        # Architecture dimensions default to "unknown", so a test that does not
        # set them exercises the fallback path rather than the trunk model.
        self.n_layer = 0
        self.n_embd = 0
        self.n_head = 0
        self.n_head_kv = 0
        self.head_dim_k = 0
        self.head_dim_v = 0
        self.n_vocab = 0
        for name, value in dims.items():
            setattr(self, name, value)


def meta_8b() -> Meta:
    return Meta(param_count=8_000_000_000, quant_label="Q8_0")


def meta_122b_a10b() -> Meta:
    """Qwen3.5-122B-A10B: 10 of 128 experts routed per token."""
    return Meta(
        param_count=122_000_000_000,
        n_expert=128,
        n_expert_used=10,
        quant_label="Q5_K_M",
    )


def gpu(index: int, name: str, total_gib: float) -> GpuInfo:
    total = int(total_gib * GB)
    return GpuInfo(index=index, name=name, total_bytes=total, free_bytes=total, used_bytes=0)


def rig() -> dict[int, GpuInfo]:
    return {
        0: gpu(0, "NVIDIA GeForce RTX 5090", 31.84),
        1: gpu(1, "NVIDIA GeForce RTX 5090", 31.84),
        2: gpu(2, "NVIDIA GeForce RTX 3090", 24.0),
        3: gpu(3, "NVIDIA GeForce RTX 3090", 24.0),
    }


# ---------------------------------------------------------------------------
# GPU table lookup
# ---------------------------------------------------------------------------


def test_gpu_matched_by_name_fragment() -> None:
    assert gpu_perf_for("NVIDIA GeForce RTX 5090").label == "RTX 5090"
    assert gpu_perf_for("NVIDIA GeForce RTX 3090 Ti").label == "RTX 3090"
    assert gpu_perf_for("NVIDIA GeForce RTX 4090").label == "RTX 4090"


def test_5090_has_roughly_double_the_bandwidth_of_a_3090() -> None:
    """The ratio is what decides a mixed split's speed, so pin it."""
    fast = gpu_perf_for("RTX 5090").bw_bytes_per_s
    slow = gpu_perf_for("RTX 3090").bw_bytes_per_s
    assert 1.8 < fast / slow < 2.0


def test_unknown_gpu_falls_back_to_vram_size() -> None:
    assert gpu_perf_for("Some Future Card", int(32 * GB)).label == "RTX 5090"
    assert gpu_perf_for("Some Future Card", int(24 * GB)).label == "RTX 3090"


def test_unknown_gpu_with_no_size_is_pessimistic() -> None:
    """An unknown card assumed fast makes every number on the page a lie."""
    perf = gpu_perf_for(None, 0)
    assert perf is GPU_PERF_UNKNOWN
    assert perf.bw_bytes_per_s < gpu_perf_for("RTX 3090").bw_bytes_per_s


def test_gpu_class_labels_a_mixed_rig() -> None:
    assert gpu_class(list(rig().values())) == "RTX 3090x2+RTX 5090x2"


def test_gpu_class_of_a_single_card() -> None:
    assert gpu_class([gpu(0, "NVIDIA GeForce RTX 5090", 31.84)]) == "RTX 5090x1"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def test_declared_parameter_count_wins() -> None:
    assert total_params(meta_8b(), 1) == 8_000_000_000


def test_parameter_count_derived_from_bytes_when_absent() -> None:
    """8.5 bits per weight for Q8_0: 8.5 GB of file is about 8B parameters."""
    meta = Meta(quant_label="Q8_0")
    derived = total_params(meta, int(8.5e9))
    assert 7.5e9 < derived < 8.5e9


def test_unknown_quant_label_uses_the_documented_default() -> None:
    assert bits_per_weight(Meta(quant_label="SOMETHING_NEW")) == 5.0


def test_quant_label_matches_a_known_prefix() -> None:
    assert bits_per_weight(Meta(quant_label="Q4_K_M-imat")) == pytest.approx(4.85)


def test_active_params_of_a_moe_is_the_routed_share() -> None:
    """10 of 128 experts on 122B total -> about 9.5B active."""
    active = active_params(meta_122b_a10b(), 1)
    assert 9e9 < active < 10e9


def test_active_params_of_a_dense_model_is_all_of_them() -> None:
    assert active_params(meta_8b(), 1) == 8_000_000_000


def test_active_params_and_active_bytes_agree_by_construction() -> None:
    """The FLOP term and the bandwidth term must mean the same thing."""
    from studioforge.core.throughput import active_bytes

    meta = meta_122b_a10b()
    param_share = active_params(meta, 89 * GB) / total_params(meta, 89 * GB)
    byte_share = active_bytes(meta, 89 * GB) / (89 * GB)
    assert param_share == pytest.approx(byte_share, rel=1e-3)


# ---------------------------------------------------------------------------
# The MoE dense trunk -- shapes from the reference rig's resident model
# ---------------------------------------------------------------------------


def meta_122b_full() -> Meta:
    """The real GGUF shape: 256 experts, 8 routed, 48 layers, n_embd 3072.

    Note the metadata says 8-of-256 (3.1%) while the model is *named* A10B
    (~8%). That difference is the whole reason the dense trunk is modelled.
    """
    meta = Meta(n_expert=256, n_expert_used=8, quant_label="Q5_K_M")
    meta.n_layer = 48
    meta.n_embd = 3072
    meta.n_head = 32
    meta.n_head_kv = 2
    meta.head_dim_k = 256
    meta.head_dim_v = 256
    meta.n_vocab = 248320
    meta.param_count = 122_241_853_278
    return meta


def test_dense_trunk_is_attention_plus_the_output_embedding() -> None:
    """Derived from metadata, not fitted: ~3.3B for the reference 122B."""
    from studioforge.core.throughput import dense_trunk_params

    trunk = dense_trunk_params(meta_122b_full())
    assert 3.0e9 < trunk < 3.7e9


def test_dense_trunk_is_zero_without_architecture_dimensions() -> None:
    """Zero means "cannot model this", and active_params falls back on it."""
    from studioforge.core.throughput import dense_trunk_params

    assert dense_trunk_params(Meta()) == 0


def test_a_sparse_moe_activates_far_more_than_the_routed_share() -> None:
    """8-of-256 is 3.1% of the file, but attention and lm_head are not routed.

    The planner's flat share says 3.8B; charging the trunk in full says 7.1B,
    against a model whose own name claims ~10B. Under-counting here would mean
    advertising a model as 2.6x faster than it is.
    """
    from studioforge.core.planner import active_weight_bytes

    meta = meta_122b_full()
    weights = 86_944_518_144
    flat = active_weight_bytes(meta, weights)
    modelled = active_params(meta, weights)
    assert flat / 1e9 == pytest.approx(2.7, abs=0.2)
    assert 6.5e9 < modelled < 7.6e9


def test_the_dense_trunk_model_is_still_conservative() -> None:
    """It lands under the stated A10B, so the estimate errs slow, not fast."""
    assert active_params(meta_122b_full(), 86_944_518_144) < 10.1e9


def test_a_dense_model_has_no_trunk_correction() -> None:
    meta = meta_8b()
    meta.n_layer, meta.n_embd, meta.n_head = 32, 4096, 32
    meta.head_dim_k = meta.head_dim_v = 128
    meta.n_vocab = 128256
    assert active_params(meta, int(8.5e9)) == 8_000_000_000


# ---------------------------------------------------------------------------
# estimate() -- worked numbers
# ---------------------------------------------------------------------------


#: KV a slot re-reads per decode step, at the two fills the old tests used.
#: The v2 signature takes bytes-per-step directly (the planner derives it
#: per-layer), so these are the old `ctx_fill * 144 KiB/token` products spelled
#: out rather than a formula the estimator no longer owns.
KV_READ_1K = 1024 * 144 * 1024
KV_READ_16K = 16384 * 144 * 1024


def est8b(
    split: dict[int, int] | None = None,
    *,
    kv_read: int = KV_READ_16K,
    parallel: int = 1,
    **kwargs: object,
) -> dict:
    """The 8B dense reference call, so a signature change edits one place."""
    return estimate(
        meta_8b(),
        int(8.5e9),
        {0: int(9 * GB)} if split is None else split,
        kv_read_bytes_per_slot=kv_read,
        parallel=parallel,
        gpus=rig(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_8b_dense_on_one_5090_lands_in_the_sanity_band() -> None:
    """8.5 GB over 1792 GB/s at eff 0.75 is 6.3 ms, plus 1.5 ms of floor.

    Measured at a short context so the weight term dominates and the band is
    testing the thing it names. The band is wide on purpose: it checks that the
    arithmetic is bandwidth-bound and roughly right, not that the number is true
    of any particular kernel. It sits lower than v1's 155 tok/s because
    T_TOKEN_OVERHEAD_S is now charged -- 1.5 ms is a fifth of this model's whole
    token budget.
    """
    assert 100 <= est8b(kv_read=KV_READ_1K)["gen_tps"] <= 160


def test_a_long_context_measurably_slows_decode() -> None:
    """Attention re-reads the KV cache per token, so context is not free.

    At 16k tokens of KV an 8B model reads 2.4 GB of cache beside 8.5 GB of
    weights -- a fifth of the traffic, and the reason a long conversation feels
    slower than a fresh one.
    """
    fresh = est8b(kv_read=KV_READ_1K)
    long_ctx = est8b(kv_read=KV_READ_16K)
    assert long_ctx["gen_tps"] < fresh["gen_tps"]
    assert 0.7 < long_ctx["gen_tps"] / fresh["gen_tps"] < 0.95


def test_the_same_model_is_slower_on_a_3090() -> None:
    """Bandwidth is most of the story for decode, so the ratio tracks it.

    Not all of it: the 1.5 ms latency floor is the same on both cards, so the
    ratio is pulled below the 1.9x the bandwidth table alone would give. That
    compression is real -- it is why a small model on a fast card is not as much
    faster than the same model on a slow one as the spec sheets imply.
    """
    fast = est8b({0: int(9 * GB)})
    slow = est8b({2: int(9 * GB)})
    assert slow["gen_tps"] < fast["gen_tps"]
    assert 1.6 < fast["gen_tps"] / slow["gen_tps"] < 2.0


# ---------------------------------------------------------------------------
# Measured anchors -- the two placements the reference rig actually ran
# ---------------------------------------------------------------------------


def meta_gemma4_31b() -> Meta:
    """Dark-Scarlett-v2.0-31B-Q8_0: gemma4 dense, 60 layers, iSWA attention."""
    return Meta(
        quant_label="Q8_0",
        n_layer=60,
        n_embd=5376,
        n_head=32,
        n_head_kv=4,
        head_dim_k=512,
        head_dim_v=512,
        n_vocab=262144,
    )


#: 30.4 GiB of Q8_0 weights, split evenly over CUDA 0 and 1 (both RTX 5090).
GEMMA4_WEIGHTS = int(30.4 * GB)
GEMMA4_SPLIT = {0: GEMMA4_WEIGHTS // 2, 1: GEMMA4_WEIGHTS - GEMMA4_WEIGHTS // 2}

#: Measured on the live child at ctx 262144 / f16 KV / 1 slot (spec table).
MEASURED_GEMMA4_GEN_TPS = 39.4
MEASURED_GEMMA4_PROMPT_TPS = 2053.0


def test_gemma4_31b_on_two_5090s_matches_the_measurement() -> None:
    """The dense anchor: 39.4 tok/s measured, and the physics has to reach it.

    This is the row v1 got catastrophically wrong -- not because the roofline
    was bad but because it charged a uniform 1.9 MB/token KV read for an iSWA
    model, which at 262k context is 258 GB per token and reported 1.9 tok/s.
    With the real per-step KV read (a few hundred MB) the untuned dense estimate
    lands within a tenth of the measurement, which is what "the dense physics is
    fine, the MoE derate must not leak onto it" means in practice.
    """
    result = estimate(
        meta_gemma4_31b(),
        GEMMA4_WEIGHTS,
        GEMMA4_SPLIT,
        kv_read_bytes_per_slot=300 * MB,
        parallel=1,
        gpus=rig(),
    )
    assert result["gen_tps"] == pytest.approx(MEASURED_GEMMA4_GEN_TPS, rel=0.25)
    # Prefill stays a band: the FLOP roofline is nominal and the harmonic split
    # term is deliberately conservative, so it under-promises against 2053.
    assert 800 <= result["prompt_tps"] <= 3000
    assert result["basis"]["is_moe"] is False
    # Two devices, dense: 0.75 - 1 * 0.05, no MoE multiplier.
    assert result["basis"]["eff_decode"] == pytest.approx(0.70)


def meta_qwen35moe_122b() -> Meta:
    """Qwen3.5-122B-A10B Q5_K_M: qwen35moe, 48 layers, 256 experts / 8 routed."""
    return meta_122b_full()


#: The measured placement: 81 GiB over 5090, 5090, 3090, 3090 at ~29/28/22/22 %.
QWEN35MOE_WEIGHTS = 86_944_518_144
QWEN35MOE_SPLIT = {
    0: int(QWEN35MOE_WEIGHTS * 0.29),
    1: int(QWEN35MOE_WEIGHTS * 0.28),
    2: int(QWEN35MOE_WEIGHTS * 0.22),
    3: int(QWEN35MOE_WEIGHTS * 0.22),
}

#: Measured on the live child at ctx 262144 / q4_0 KV / 1 slot (spec table).
MEASURED_QWEN35MOE_GEN_TPS = 37.3
MEASURED_QWEN35MOE_PROMPT_TPS = 869.0


def test_qwen35moe_122b_across_four_gpus_matches_the_measurement() -> None:
    """The MoE anchor: 37.3 tok/s measured against a 4x-optimistic v1.

    The remaining gap is calibration's, not the formula's -- but the formula has
    to get close enough that a per-model factor is a *correction* rather than
    the whole answer. Two asymmetric guards, because the two directions are not
    equally bad: never more than 2x optimistic (over-promising is what D20 named
    as the failure), and never more than 20% below the measurement (a formula
    that under-promises this badly would make every MoE look unusable).
    """
    result = estimate(
        meta_qwen35moe_122b(),
        QWEN35MOE_WEIGHTS,
        QWEN35MOE_SPLIT,
        kv_read_bytes_per_slot=350 * MB,
        parallel=1,
        gpus=rig(),
    )
    assert 30 <= result["gen_tps"] <= 75
    assert result["gen_tps"] <= 2.0 * MEASURED_QWEN35MOE_GEN_TPS
    assert result["gen_tps"] >= 0.8 * MEASURED_QWEN35MOE_GEN_TPS
    assert 500 <= result["prompt_tps"] <= 2000
    assert result["basis"]["is_moe"] is True
    # Four devices, MoE: (0.75 - 3 * 0.10) * 0.45.
    assert result["basis"]["eff_decode"] == pytest.approx(0.2025)


def test_a_small_dense_model_hits_the_latency_floor() -> None:
    """0.92 GB of weights is 0.5 ms of bandwidth and 1.5 ms of everything else.

    v1 had no per-token latency term at all, so it claimed 927 tok/s for a
    Qwen2.5-1.5B (and 2063 for SmolVLM) -- numbers llama.cpp has never produced
    for any model on any hardware. Below about 5 GB the floor, not the bus, is
    what a token costs, and the estimate has to say so.
    """
    meta = Meta(
        quant_label="Q4_K_M",
        n_layer=28,
        n_embd=1536,
        n_head=12,
        n_head_kv=2,
        head_dim_k=128,
        head_dim_v=128,
        n_vocab=151936,
    )
    weights = int(0.92 * GB)
    result = estimate(
        meta,
        weights,
        {0: weights},
        kv_read_bytes_per_slot=50 * MB,
        parallel=1,
        gpus=rig(),
    )
    assert result["gen_tps"] < 700
    # And the floor is the majority of the token, not a rounding error.
    assert result["basis"]["t_overhead_s"] / result["basis"]["t_token_s"] > 0.5


def test_prefill_across_devices_is_harmonic_not_a_weighted_mean() -> None:
    """A pipeline runs at the sum of its stages' times, not their average rate.

    Half the layers on a 3090 means the 5090 sits idle for that half. The old
    share-weighted mean of FLOPS said (209 + 71) / 2 = 140 TFLOPS; the pipeline
    really delivers 1 / (0.5/209 + 0.5/71) = 106. Over-stating a mixed split's
    prefill is what made the planner's prefer_single_gpu exceptions look free.
    """
    meta = meta_gemma4_31b()
    half = GEMMA4_WEIGHTS // 2
    mixed = estimate(
        meta,
        GEMMA4_WEIGHTS,
        {0: half, 2: GEMMA4_WEIGHTS - half},
        kv_read_bytes_per_slot=300 * MB,
        parallel=1,
        gpus=rig(),
    )
    params = mixed["basis"]["active_params"]
    weighted_mean_flops = 0.5 * 209e12 + 0.5 * 71e12
    would_have_said = throughput.PROMPT_EFFICIENCY * weighted_mean_flops / (2.0 * params)
    assert mixed["prompt_tps"] < would_have_said
    # Not a rounding difference: the slow half dominates the pipeline.
    assert mixed["prompt_tps"] < 0.85 * would_have_said


#: The resident model's real placement: all four GPUs, ctx 8192, parallel 1.
RESIDENT_122B_SPLIT = {0: 25 * GB, 1: 25 * GB, 2: 18 * GB, 3: 18 * GB}
RESIDENT_122B_WEIGHTS = 86_944_518_144
RESIDENT_122B_KV_READ = 4096 * 96 * 1024

#: What the live child actually reports there, over 254,420 generated tokens
#: (llamacpp:tokens_predicted_total / llamacpp:tokens_predicted_seconds_total).
MEASURED_122B_GEN_TPS = 37.05
MEASURED_122B_PROMPT_TPS = 1148.99


def est122b(**kwargs: object) -> dict:
    return estimate(
        meta_122b_full(),
        RESIDENT_122B_WEIGHTS,
        RESIDENT_122B_SPLIT,
        kv_read_bytes_per_slot=RESIDENT_122B_KV_READ,
        parallel=1,
        gpus=rig(),
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_122b_estimate_is_optimistic_by_a_known_factor() -> None:
    """Pins the measured gap, so a formula change that moves it is visible.

    D20 recorded this band at 0.20-0.35 (measured / estimated) for the v1
    formula -- a 4x over-promise the document called out as calibration's
    problem. v2 charges the MoE kernel derate and the harsher four-way split
    derate directly, so the same placement now lands within about 30% of the
    measurement and the leftover really is per-model noise. If this test starts
    failing, the estimator changed -- check whether it got better or worse
    against D20's number before widening the band.
    """
    ratio = MEASURED_122B_GEN_TPS / est122b()["gen_tps"]
    assert 0.6 <= ratio <= 1.0


def test_the_122b_prefill_estimate_lands_on_the_measurement() -> None:
    """1149 tok/s measured; the harmonic split plus the MoE prefill derate
    reproduce it without a calibration factor, which is the point of both."""
    assert est122b()["prompt_tps"] == pytest.approx(MEASURED_122B_PROMPT_TPS, rel=0.25)


def test_calibration_pulls_the_122b_estimate_onto_the_measurement() -> None:
    """One measured ratio is all it takes for the number to become useful."""
    raw = est122b()
    factor = MEASURED_122B_GEN_TPS / raw["gen_tps"]
    calibrated = est122b(efficiency=factor)
    assert calibrated["gen_tps"] == pytest.approx(MEASURED_122B_GEN_TPS, abs=0.5)


def test_efficiency_drops_five_points_per_extra_device() -> None:
    single = est8b({0: int(9 * GB)})
    pair = est8b({0: int(4.5 * GB), 1: int(4.5 * GB)})
    assert single["basis"]["eff_decode"] == pytest.approx(0.75)
    assert pair["basis"]["eff_decode"] == pytest.approx(0.70)


def test_a_moe_pays_twice_the_split_penalty_and_a_kernel_derate() -> None:
    """Two independent MoE effects, and the basis has to show both.

    The kernel derate applies on one GPU too (MUL_MAT_ID gathers small scattered
    experts wherever it runs); the doubled per-device term is the routing-aware
    pipeline stall on top of it. Conflating them into one number would make a
    single-GPU MoE look fine and a split one unexplainable.
    """
    meta = meta_122b_full()
    one = estimate(
        meta, 40 * GB, {0: 40 * GB}, kv_read_bytes_per_slot=100 * MB, parallel=1, gpus=rig()
    )
    two = estimate(
        meta,
        40 * GB,
        {0: 20 * GB, 1: 20 * GB},
        kv_read_bytes_per_slot=100 * MB,
        parallel=1,
        gpus=rig(),
    )
    assert one["basis"]["eff_decode"] == pytest.approx(0.75 * 0.45)
    assert two["basis"]["eff_decode"] == pytest.approx((0.75 - 0.10) * 0.45)


def test_a_split_across_generations_runs_at_the_slower_cards_pace() -> None:
    """Half the layers on a 3090 costs far more than half the speed."""
    both_fast = est8b({0: int(4.25 * GB), 1: int(4.25 * GB)})
    mixed = est8b({0: int(4.25 * GB), 2: int(4.25 * GB)})
    assert mixed["gen_tps"] < both_fast["gen_tps"]


def test_batched_throughput_grows_then_saturates() -> None:
    """More slots buy less and less: that flattening IS the knee."""
    one = est8b(parallel=1)
    four = est8b(parallel=4)
    eight = est8b(parallel=8)
    assert four["gen_tps_batched"] > one["gen_tps_batched"]
    assert eight["gen_tps_batched"] > four["gen_tps_batched"]
    # Sub-linear: eight slots do not buy eight times one slot.
    assert eight["gen_tps_batched"] < 8 * one["gen_tps"]


def test_batched_is_never_below_a_single_stream() -> None:
    result = est8b(parallel=1)
    assert result["gen_tps_batched"] >= result["gen_tps"]


def test_the_per_token_overhead_is_paid_once_per_step_not_once_per_slot() -> None:
    """Amortising the fixed cost over a batch is most of why batching helps.

    A step at N slots pays the sampling/launch floor once, so a model whose
    token is mostly floor gains more from a second slot than the KV arithmetic
    alone predicts.
    """
    one = est8b(kv_read=0, parallel=1)
    two = est8b(kv_read=0, parallel=2)
    # With no KV traffic at all, t_step is identical to t_token, so two slots
    # are exactly twice one -- the floor is inside the step, not per token.
    assert two["gen_tps_batched"] == pytest.approx(2 * one["gen_tps"], rel=1e-3)


def test_the_knee_caps_the_advertised_batch() -> None:
    """Past the knee the catalog must not advertise slots the planner refuses."""
    capped = est8b(parallel=8, knee=2)
    uncapped = est8b(parallel=8)
    assert capped["basis"]["slots_used"] == 2
    assert capped["gen_tps_batched"] < uncapped["gen_tps_batched"]


def test_prompt_throughput_is_flop_bound_not_bandwidth_bound() -> None:
    """A 5090's FLOP advantage over a 3090 is bigger than its bandwidth one."""
    fast = est8b({0: int(9 * GB)})
    slow = est8b({2: int(9 * GB)})
    prompt_ratio = fast["prompt_tps"] / slow["prompt_tps"]
    gen_ratio = fast["gen_tps"] / slow["gen_tps"]
    assert prompt_ratio > gen_ratio


def test_prompt_throughput_of_a_moe_beats_a_dense_model_of_the_same_size() -> None:
    """Only the routed experts do arithmetic, so prefill is far cheaper.

    Less cheap than v1 claimed: MOE_PROMPT_EFFICIENCY takes 60% of the advantage
    back, because routing shreds the prefill batch into per-expert slivers. A
    12.8x parameter advantage becomes about 5x of real throughput.
    """
    moe = estimate(
        meta_122b_a10b(),
        89 * GB,
        {0: int(30 * GB)},
        kv_read_bytes_per_slot=RESIDENT_122B_KV_READ,
        parallel=1,
        gpus=rig(),
    )
    dense = estimate(
        Meta(param_count=122_000_000_000, quant_label="Q5_K_M"),
        89 * GB,
        {0: int(30 * GB)},
        kv_read_bytes_per_slot=RESIDENT_122B_KV_READ,
        parallel=1,
        gpus=rig(),
    )
    assert moe["prompt_tps"] > dense["prompt_tps"] * 4


def test_no_placement_returns_nulls_not_zeros() -> None:
    """ "Cannot estimate" and "estimated to be zero" are different answers."""
    result = est8b({})
    assert result["gen_tps"] is None
    assert result["prompt_tps"] is None
    assert "unavailable" in result["basis"]


def test_the_unavailable_basis_carries_the_same_keys_as_a_real_one() -> None:
    """A consumer must never have to branch on which shape it got."""
    real = est8b()
    unavailable = est8b({})
    assert set(real["basis"]) <= set(unavailable["basis"])
    assert unavailable["basis"]["estimator_version"] == ESTIMATOR_VERSION


def test_the_basis_reports_the_estimator_version_and_its_inputs() -> None:
    """Every recorded observation is stamped from here, so it has to be here."""
    basis = est8b(kv_read=KV_READ_1K)["basis"]
    assert basis["estimator_version"] == ESTIMATOR_VERSION
    assert basis["kv_read_bytes_per_slot"] == KV_READ_1K
    assert basis["t_overhead_s"] == throughput.T_TOKEN_OVERHEAD_S
    assert basis["t_token_s"] > basis["t_overhead_s"]


def test_unknown_parameter_count_still_gives_a_decode_estimate() -> None:
    """Decode needs bytes, not parameters, so it survives thin metadata."""
    result = estimate(
        Meta(quant_label="Q4_K_M"),
        int(5e9),
        {0: int(6 * GB)},
        kv_read_bytes_per_slot=8192 * 144 * 1024,
        parallel=1,
    )
    assert result["gen_tps"] is not None


def test_zero_kv_read_does_not_divide_by_zero() -> None:
    result = est8b(kv_read=0)
    assert result["gen_tps"] is not None


def test_the_reference_fill_is_below_every_context_tier() -> None:
    """est_gen_tps is quoted at a fill; that fill must fit the smallest row."""
    assert throughput.REFERENCE_FILL_TOKENS <= 16384


# ---------------------------------------------------------------------------
# Metrics parsing -- real b10425 exposition output
# ---------------------------------------------------------------------------

LIVE_METRICS = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 71091
# HELP llamacpp:prompt_seconds_total Total time spent processing prompts
# TYPE llamacpp:prompt_seconds_total counter
llamacpp:prompt_seconds_total 60.0428
# TYPE llamacpp:tokens_predicted_total counter
llamacpp:tokens_predicted_total 254420
llamacpp:tokens_predicted_seconds_total 6983.54
llamacpp:n_decode_total 254526
llamacpp:prompt_tokens_seconds 1148.99
llamacpp:predicted_tokens_seconds 37.0467
llamacpp:requests_processing 0
llamacpp:requests_deferred 0
llamacpp:n_busy_slots_per_decode 1
"""


def test_parse_real_exposition_output() -> None:
    parsed = parse_metrics(LIVE_METRICS)
    assert parsed["llamacpp:tokens_predicted_total"] == 254420
    assert parsed["llamacpp:predicted_tokens_seconds"] == pytest.approx(37.0467)
    assert parsed["llamacpp:requests_deferred"] == 0
    # Comment lines contribute nothing.
    assert not any(k.startswith("#") for k in parsed)


def test_parse_strips_prometheus_labels() -> None:
    """llama.cpp emits none today; a release that adds them must not break us."""
    parsed = parse_metrics('llamacpp:n_decode_total{model="x"} 42')
    assert parsed["llamacpp:n_decode_total"] == 42


def test_parse_survives_garbage_and_emptiness() -> None:
    assert parse_metrics(None) == {}
    assert parse_metrics("") == {}
    assert parse_metrics("not a metric line\nllamacpp:x NaN\nllamacpp:y abc") == {}


def test_parse_drops_non_finite_values() -> None:
    assert "llamacpp:x" not in parse_metrics("llamacpp:x inf")


# ---------------------------------------------------------------------------
# sample_between -- deltas, not lifetime averages
# ---------------------------------------------------------------------------


def _counters(decodes: float, gen_tok: float, gen_s: float, pp_tok: float, pp_s: float) -> dict:
    return {
        throughput.METRIC_DECODE_TOTAL: decodes,
        throughput.METRIC_PREDICTED_TOKENS: gen_tok,
        throughput.METRIC_PREDICTED_SECONDS: gen_s,
        throughput.METRIC_PROMPT_TOKENS: pp_tok,
        throughput.METRIC_PROMPT_SECONDS: pp_s,
        throughput.METRIC_BUSY_SLOTS: 2.0,
        throughput.METRIC_REQUESTS_DEFERRED: 1.0,
    }


def test_sample_measures_the_window_not_the_lifetime() -> None:
    """A lifetime gauge cannot tell "fast now" from "was fast an hour ago"."""
    before = _counters(1000, 10_000, 1000.0, 5_000, 10.0)
    after = _counters(1200, 10_400, 1005.0, 5_600, 10.5)
    sample = sample_between(before, after, elapsed_s=120.0)
    assert sample is not None
    assert sample["gen_tps"] == pytest.approx(80.0)  # 400 tokens / 5 s
    assert sample["prompt_tps"] == pytest.approx(1200.0)  # 600 tokens / 0.5 s
    assert sample["n_busy_slots"] == 2.0
    assert sample["requests_deferred"] == 1.0
    assert sample["sample_s"] == 120.0


def test_an_idle_window_records_nothing() -> None:
    """A zero would poison the median calibration is built on."""
    counters = _counters(1000, 10_000, 1000.0, 5_000, 10.0)
    assert sample_between(counters, counters, elapsed_s=120.0) is None


def test_a_counter_reset_records_nothing() -> None:
    """The child restarted; the "delta" is meaningless."""
    before = _counters(9000, 90_000, 9000.0, 50_000, 100.0)
    after = _counters(10, 100, 1.0, 50, 0.1)
    assert sample_between(before, after, elapsed_s=120.0) is None


def test_missing_metrics_record_nothing() -> None:
    assert sample_between({}, _counters(10, 10, 1, 10, 1), elapsed_s=60.0) is None
    assert sample_between(_counters(10, 10, 1, 10, 1), {}, elapsed_s=60.0) is None


def test_decodes_without_timing_records_nothing() -> None:
    before = {throughput.METRIC_DECODE_TOTAL: 100.0}
    after = {throughput.METRIC_DECODE_TOTAL: 200.0}
    assert sample_between(before, after, elapsed_s=60.0) is None


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def obs(
    *,
    model_id: str = "m",
    devices: str = "0",
    gpu_class_label: str = "RTX 5090x1",
    gen: float = 50.0,
    est_gen: float = 100.0,
    prompt: float = 500.0,
    est_prompt: float = 1000.0,
    ctx: int = 32768,
    parallel: int = 1,
    estimator_version: int | None = ESTIMATOR_VERSION,
) -> dict:
    return {
        "model_id": model_id,
        "devices": devices,
        "gpu_class": gpu_class_label,
        "gen_tps": gen,
        "est_gen_tps": est_gen,
        "prompt_tps": prompt,
        "est_prompt_tps": est_prompt,
        "ctx_size": ctx,
        "parallel": parallel,
        "estimator_version": estimator_version,
    }


def test_calibration_is_the_median_ratio_for_this_model_on_these_devices() -> None:
    rows = [obs(gen=40.0), obs(gen=50.0), obs(gen=60.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["gen"] == pytest.approx(0.5)
    assert result["basis"] == "model+devices"
    assert result["samples"] == 3
    assert result["models"] == 1


def test_calibration_uses_the_median_so_one_outlier_cannot_move_it() -> None:
    """A sample taken while ComfyUI hammered the same card is not the answer."""
    rows = [obs(gen=50.0), obs(gen=50.0), obs(gen=15.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["gen"] == pytest.approx(0.5)


def test_the_model_tier_sits_between_the_exact_placement_and_the_peers() -> None:
    """A placement change moves the number far less than a model change does.

    Without this tier a model measured only on ``0,1`` got nothing at all when
    the catalog asked about ``0``, and fell straight through to whatever the
    neighbours happened to be doing.
    """
    rows = [obs(devices="0,1", gen=30.0), obs(devices="0,1", gen=30.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "model"
    assert result["gen"] == pytest.approx(0.3)
    assert result["models"] == 1


def test_the_exact_placement_beats_the_model_tier() -> None:
    rows = [
        obs(devices="0", gen=50.0),
        obs(devices="0", gen=50.0),
        obs(devices="0,1", gen=30.0),
        obs(devices="0,1", gen=30.0),
    ]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "model+devices"
    assert result["gen"] == pytest.approx(0.5)


def test_the_peer_tier_is_a_median_of_per_model_medians() -> None:
    """One chatty model must not outvote the rest of the rig.

    The live registry is exactly this shape: 84 rows for the resident MoE and 3
    for a dense model measured once. A raw pool median *is* the chatty model's
    number, which is how a sparse-MoE derate ended up applied to every dense
    model on the box.
    """
    rows = [obs(model_id="chatty", gen=30.0) for _ in range(20)]
    rows += [obs(model_id="quiet", gen=90.0) for _ in range(2)]
    result = calibrate(
        rows,
        model_id="new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={"chatty": False, "quiet": False},
    )
    assert result["basis"] == "peers"
    assert result["gen"] == pytest.approx(0.6)  # median(0.3, 0.9), not ~0.3
    assert result["models"] == 2
    assert result["samples"] == 22


def test_a_peer_of_the_wrong_density_does_not_calibrate_this_model() -> None:
    """The MoE derate must not leak onto a dense model, or the reverse."""
    rows = [obs(model_id="moe", gen=20.0) for _ in range(4)]
    result = calibrate(
        rows,
        model_id="dense-new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={"moe": True},
    )
    assert result["basis"] == "none"


def test_a_peer_on_a_different_number_of_devices_does_not_count() -> None:
    """One card and four cards are different machines as far as decode cares."""
    rows = [
        obs(model_id="peer", devices="0,1,2,3", gpu_class_label="RTX 5090x1", gen=20.0)
        for _ in range(4)
    ]
    result = calibrate(
        rows,
        model_id="new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={"peer": False},
    )
    assert result["basis"] == "none"


def test_a_peer_with_an_unknown_density_is_not_evidence() -> None:
    rows = [obs(model_id="mystery", gen=30.0) for _ in range(4)]
    result = calibrate(
        rows,
        model_id="new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={},
    )
    assert result["basis"] == "none"


def test_a_peer_needs_its_own_minimum_before_it_votes() -> None:
    rows = [obs(model_id="one-shot", gen=30.0)]
    rows += [obs(model_id="settled", gen=90.0) for _ in range(3)]
    result = calibrate(
        rows,
        model_id="new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={"one-shot": False, "settled": False},
    )
    assert result["basis"] == "peers"
    assert result["gen"] == pytest.approx(0.9)
    assert result["models"] == 1


def test_rows_from_an_older_estimator_are_ignored() -> None:
    """A ratio corrects ONE formula; v1's would teach v2 a dead mistake."""
    rows = [obs(gen=30.0, estimator_version=1), obs(gen=30.0, estimator_version=1)]
    assert calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")["basis"] == (
        "none"
    )


def test_rows_with_no_estimator_version_are_ignored() -> None:
    """Everything written before migration 004 carries NULL."""
    rows = [obs(gen=30.0, estimator_version=None), obs(gen=30.0, estimator_version=None)]
    assert calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")["basis"] == (
        "none"
    )


def test_current_version_rows_survive_beside_stale_ones() -> None:
    """A mixed table must calibrate off the fresh half, not refuse outright."""
    rows = [obs(gen=90.0, estimator_version=1) for _ in range(10)]
    rows += [obs(gen=50.0), obs(gen=50.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "model+devices"
    assert result["gen"] == pytest.approx(0.5)
    assert result["samples"] == 2


def test_calibration_returns_neutral_with_no_data() -> None:
    result = calibrate([], model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result == {"gen": 1.0, "prompt": 1.0, "basis": "none", "samples": 0, "models": 0}


def test_one_sample_is_not_enough_to_calibrate() -> None:
    result = calibrate([obs()], model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "none"


def test_absurd_ratios_are_discarded_not_applied() -> None:
    """A 50x ratio means the two numbers describe different things."""
    rows = [obs(gen=5000.0), obs(gen=5000.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "none"


def test_the_clamp_still_applies_in_the_peer_tier() -> None:
    """The 9.2 ratio the rig recorded for Dark-Scarlett must not reach anyone."""
    rows = [obs(model_id="peer", gen=920.0, est_gen=100.0) for _ in range(4)]
    result = calibrate(
        rows,
        model_id="new",
        devices=[0],
        gpu_class_label="RTX 5090x1",
        is_moe=False,
        peer_moe={"peer": False},
    )
    assert result["basis"] == "none"


def test_a_different_model_on_the_same_devices_does_not_calibrate_this_one() -> None:
    """Without a gpu_class to fall back to there is nothing left to say."""
    rows = [obs(model_id="other"), obs(model_id="other")]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label=None)
    assert result["basis"] == "none"


def test_zero_measurements_are_ignored() -> None:
    rows = [obs(gen=0.0), obs(gen=0.0), obs(gen=50.0)]
    result = calibrate(rows, model_id="m", devices=[0], gpu_class_label="RTX 5090x1")
    assert result["basis"] == "none"


# ---------------------------------------------------------------------------
# measured_for / confidence
# ---------------------------------------------------------------------------


def test_measured_prefers_the_exact_context() -> None:
    rows = [obs(ctx=32768, gen=40.0), obs(ctx=131072, gen=20.0)]
    found = measured_for(rows, devices=[0], ctx_size=131072)
    assert found["gen_tps"] == 20.0
    assert found["exact"] is True


def test_measured_falls_back_to_the_same_devices_at_another_context() -> None:
    """Still evidence about the hardware -- but not about this row."""
    rows = [obs(ctx=32768, gen=40.0)]
    found = measured_for(rows, devices=[0], ctx_size=131072)
    assert found["gen_tps"] == 40.0
    assert found["exact"] is False


def test_measured_is_null_for_a_placement_never_observed() -> None:
    found = measured_for([obs(devices="0")], devices=[2, 3], ctx_size=32768)
    assert found["gen_tps"] is None


def test_confidence_says_measured_only_for_an_exact_observation() -> None:
    assert confidence_for({"exact": True, "gen_tps": 40.0}, {"basis": "none"}) == "measured"
    assert confidence_for({"exact": False, "gen_tps": 40.0}, {"basis": "gpu_class"}) == "calibrated"
    assert confidence_for({"exact": None, "gen_tps": None}, {"basis": "none"}) == "estimated"
