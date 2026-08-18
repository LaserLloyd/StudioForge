"""Per-layer KV geometry: the four shapes, and the numbers they must produce.

The planner used to know two things about a KV cache -- "uniform" and "iSWA" --
and half this library is neither. Qwen3.5/3.6/3.8 cache only every fourth layer
and keep a fixed recurrent state on the rest; the planner charged all 65 layers
full context, a straight 4x that spread a 27B over four GPUs it did not need.
The iSWA half was sized correctly for VRAM but its *per-token* figure was still
the uniform one, so every downstream consumer -- slot counts, the bandwidth
knee, throughput -- divided or multiplied by a number 24x too big.

Anchors are measured GGUF shapes from the reference rig (see SPEC/D22), so a
formula change that breaks a real model fails here first.
"""

from __future__ import annotations

import pytest

from studioforge.core.planner import (
    MAX_PARALLEL_CAP,
    KvLayer,
    Planner,
    attention_kind,
    effective_kv_bytes_per_token,
    estimate_kv_bytes,
    estimate_kv_bytes_iswa,
    kv_alloc_bytes,
    kv_bytes_per_token,
    kv_layers,
    kv_read_bytes_per_slot,
    max_parallel_for,
    recurrent_state_bytes_per_slot,
)
from studioforge.types import GB, GgufMeta
from tests.unit.test_planner import (
    StubProbe,
    gpu,
    make_config,
    make_meta,
    make_record,
)

GB_DEC = 1_000_000_000
MB_DEC = 1_000_000
KIB = 1024
GIB = 1024**3


# ---------------------------------------------------------------------------
# Model shapes, dumped from the real GGUF headers on this rig
# ---------------------------------------------------------------------------


def meta_gemma4_31b(**overrides: object) -> GgufMeta:
    """Gemma-4 31B (``gemma4``): 60 layers, 5 sliding-window per full one.

    ``attention.head_count_kv`` is a per-layer array -- 16 heads on the window
    layers, 4 on the global ones -- and the window layers use half the head
    dimension (256 against 512).
    """
    pattern = [True, True, True, True, True, False] * 10
    heads = [16, 16, 16, 16, 16, 4] * 10
    return GgufMeta(
        architecture="gemma4",
        n_layer=60,
        n_head=32,
        n_head_kv=16,
        n_embd=5376,
        n_embd_head_k=512,
        n_embd_head_v=512,
        n_ctx_train=262144,
        tensor_bytes=int(30.4 * GB_DEC),
        quant_label="Q8_0",
        extra={
            "swa_window": 1024,
            "swa_pattern": pattern,
            "swa_key_length": 256,
            "swa_value_length": 256,
            "head_count_kv_values": heads,
        },
    ).model_copy(update=dict(overrides))


def meta_qwen35_27b(**overrides: object) -> GgufMeta:
    """Qwen3.5-27B (``qwen35``): 65 blocks, 1 of them an MTP head.

    ``full_attention_interval 4`` -- layers 3, 7, ... 63 hold a KV cache and the
    other 49 are Gated-DeltaNet recurrent layers. One of those 49 is the
    ``nextn_predict_layers`` head, which holds neither kind of state.
    """
    return GgufMeta(
        architecture="qwen35",
        n_layer=65,
        n_head=24,
        n_head_kv=4,
        n_embd=5120,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_ctx_train=262144,
        tensor_bytes=int(19.5 * GB_DEC),
        quant_label="Q5_K_M",
        extra={
            "full_attention_interval": 4,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 6144,
            "ssm_state_size": 128,
            "ssm_group_count": 16,
            "ssm_time_step_rank": 48,
            "nextn_predict_layers": 1,
        },
    ).model_copy(update=dict(overrides))


def meta_qwen35moe_122b() -> GgufMeta:
    """Qwen3.5-122B-A10B (``qwen35moe``): 48 blocks, no MTP head, 256/8 experts."""
    return GgufMeta(
        architecture="qwen35moe",
        n_layer=48,
        n_head=32,
        n_head_kv=2,
        n_embd=3072,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_expert=256,
        n_expert_used=8,
        n_ctx_train=262144,
        tensor_bytes=int(81 * GB_DEC),
        quant_label="Q5_K_M",
        extra={
            "full_attention_interval": 4,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 8192,
            "ssm_state_size": 128,
            "ssm_group_count": 16,
            "ssm_time_step_rank": 64,
        },
    )


def meta_uniform() -> GgufMeta:
    """An ordinary dense model: 32 layers, 8 KV heads, head_dim 128."""
    return GgufMeta(
        architecture="llama",
        n_layer=32,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=131072,
        tensor_bytes=8 * GB,
    )


# ---------------------------------------------------------------------------
# kv_layers: the four shapes
# ---------------------------------------------------------------------------


def kind_counts(meta: GgufMeta) -> dict[str, int]:
    counts: dict[str, int] = {}
    for layer in kv_layers(meta):
        counts[layer.kind] = counts.get(layer.kind, 0) + 1
    return counts


def test_a_uniform_model_is_all_full_attention_layers() -> None:
    layers = kv_layers(meta_uniform())
    assert len(layers) == 32
    assert layers == [KvLayer("full", 8, 128, 128)] * 32


def test_gemma4_is_five_window_layers_per_global_one() -> None:
    """And the window layers are cheaper twice over: fewer heads, half the dim."""
    assert kind_counts(meta_gemma4_31b()) == {"swa": 50, "full": 10}
    layers = kv_layers(meta_gemma4_31b())
    assert layers[0] == KvLayer("swa", 16, 256, 256, 1024)
    assert layers[5] == KvLayer("full", 4, 512, 512)


def test_qwen35_caches_only_every_fourth_layer() -> None:
    """(il + 1) % full_attention_interval == 0, and nothing else has a cache."""
    layers = kv_layers(meta_qwen35_27b())
    assert kind_counts(meta_qwen35_27b()) == {"full": 16, "none": 49}
    assert [i for i, layer in enumerate(layers) if layer.kind == "full"] == list(range(3, 64, 4))


def test_a_zero_in_the_head_count_array_means_that_layer_has_no_cache() -> None:
    """Gemma-3n / LFM2 / Nemotron-H: the scalar collapse (max) hides these."""
    meta = GgufMeta(
        architecture="gemma3n",
        n_layer=6,
        n_head=8,
        n_head_kv=4,
        n_embd_head_k=128,
        n_embd_head_v=128,
        extra={"head_count_kv_values": [4, 4, 0, 4, 0, 4]},
    )
    assert kind_counts(meta) == {"full": 4, "none": 2}


def test_unusable_metadata_yields_no_layers_rather_than_a_free_model() -> None:
    """Empty means "cannot estimate". A caller that reads it as 0 bytes lies."""
    assert kv_layers(GgufMeta()) == []
    assert kv_alloc_bytes(GgufMeta(), ctx_total=8192, kv_k="f16", kv_v="f16") == 0
    assert effective_kv_bytes_per_token(GgufMeta(), kv_k="f16", kv_v="f16", ctx_per_slot=8192) == 0
    assert kv_read_bytes_per_slot(GgufMeta(), kv_k="f16", kv_v="f16", ctx_fill=8192) == 0


def test_attention_kind_labels_the_four_shapes() -> None:
    assert attention_kind(meta_uniform()) == "full"
    assert attention_kind(meta_gemma4_31b()) == "iswa"
    assert attention_kind(meta_qwen35_27b()) == "hybrid"
    assert attention_kind(meta_qwen35moe_122b()) == "hybrid"
    assert attention_kind(GgufMeta()) == "unknown"


# ---------------------------------------------------------------------------
# kv_alloc_bytes reproduces what it replaced
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx_total", [16384, 65536, 262144])
@pytest.mark.parametrize("parallel", [1, 2, 8])
@pytest.mark.parametrize("kv_type", ["f16", "q8_0", "q4_0"])
def test_iswa_allocation_is_byte_for_byte_what_it_always_was(
    ctx_total: int, parallel: int, kv_type: str
) -> None:
    """D15's arithmetic was validated against real loads; it must not move.

    ``kv_alloc_bytes`` is the single entry point now, but for an iSWA model it
    has to agree with :func:`estimate_kv_bytes_iswa` exactly -- not within a
    percent. Under-estimating a KV cache is an OOM at load.
    """
    meta = meta_gemma4_31b()
    legacy = estimate_kv_bytes_iswa(
        meta=meta,
        ctx_total=ctx_total,
        kv_type_k=kv_type,
        kv_type_v=kv_type,
        parallel=parallel,
    )
    assert legacy is not None
    assert (
        kv_alloc_bytes(
            meta, ctx_total=ctx_total, kv_k=kv_type, kv_v=kv_type, parallel=parallel
        )
        == legacy
    )


@pytest.mark.parametrize("ctx_total", [8192, 65536])
@pytest.mark.parametrize("kv_type", ["f16", "q8_0"])
def test_uniform_allocation_is_byte_for_byte_what_it_always_was(
    ctx_total: int, kv_type: str
) -> None:
    meta = meta_uniform()
    legacy = estimate_kv_bytes(
        n_layer=meta.n_layer,
        n_head_kv=meta.n_head_kv,
        head_dim_k=meta.head_dim_k,
        head_dim_v=meta.head_dim_v,
        ctx_total=ctx_total,
        kv_type_k=kv_type,
        kv_type_v=kv_type,
    )
    assert kv_alloc_bytes(meta, ctx_total=ctx_total, kv_k=kv_type, kv_v=kv_type) == legacy


def test_a_non_iswa_model_still_reports_none_from_the_legacy_helper() -> None:
    """Callers built their uniform fallback on that signal."""
    assert (
        estimate_kv_bytes_iswa(
            meta=meta_uniform(), ctx_total=8192, kv_type_k="f16", kv_type_v="f16"
        )
        is None
    )


# ---------------------------------------------------------------------------
# The hybrid case: the 4x over-charge this work exists to fix
# ---------------------------------------------------------------------------


def test_qwen35_27b_at_16k_costs_one_gib_of_kv_plus_its_recurrent_state() -> None:
    """The anchor: 16 cached layers, not 65.

    16 attention layers x 16384 tokens x 4 KV heads x (256 + 256) dims x 2 B is
    exactly 1 GiB. The planner used to charge all 65 layers -- 4.3 GiB -- which
    is what put this model on four GPUs at 262k and forced the 122B onto a q4_0
    cache.
    """
    meta = meta_qwen35_27b()
    cache = 16 * 16384 * 4 * (256 + 256) * 2
    assert cache == 1 * GIB

    state = recurrent_state_bytes_per_slot(meta)
    total = kv_alloc_bytes(meta, ctx_total=16384, kv_k="f16", kv_v="f16", parallel=1)
    assert total == cache + state

    uniform = 65 * 16384 * 4 * (256 + 256) * 2
    assert total < uniform / 3


def test_the_recurrent_state_is_157_mb_per_slot_and_excludes_the_mtp_head() -> None:
    """48 Gated-DeltaNet layers x ((3 x (6144 + 4096)) + 128 x 6144) x 4 B.

    The 65th block is an MTP head: llama.cpp does not run it when decoding, so
    it holds neither a KV cache nor a recurrent state. Counting it would charge
    every slot one layer too many.
    """
    meta = meta_qwen35_27b()
    per_layer = ((4 - 1) * (6144 + 2 * 16 * 128) + 128 * 6144) * 4
    assert recurrent_state_bytes_per_slot(meta) == 48 * per_layer
    assert recurrent_state_bytes_per_slot(meta) == pytest.approx(157 * MB_DEC, rel=0.01)

    # Without the nextn key the head would be counted as a recurrent layer.
    no_mtp = meta_qwen35_27b(extra={**meta.extra, "nextn_predict_layers": 0})
    assert recurrent_state_bytes_per_slot(no_mtp) == 49 * per_layer


def test_the_122b_recurrent_state_is_156_mb_per_slot() -> None:
    """36 recurrent layers of 48; no MTP head on this one."""
    assert recurrent_state_bytes_per_slot(meta_qwen35moe_122b()) == pytest.approx(
        156 * MB_DEC, rel=0.01
    )


def test_the_recurrent_state_scales_with_slots_and_not_with_context() -> None:
    """That is the whole point of the architecture -- and it still costs VRAM."""
    meta = meta_qwen35_27b()
    state = recurrent_state_bytes_per_slot(meta)
    one = kv_alloc_bytes(meta, ctx_total=16384, kv_k="f16", kv_v="f16", parallel=1)
    four = kv_alloc_bytes(meta, ctx_total=16384 * 4, kv_k="f16", kv_v="f16", parallel=4)
    assert four == (one - state) * 4 + state * 4


def test_a_model_without_ssm_keys_has_no_recurrent_state() -> None:
    assert recurrent_state_bytes_per_slot(meta_uniform()) == 0
    assert recurrent_state_bytes_per_slot(meta_gemma4_31b()) == 0


# ---------------------------------------------------------------------------
# effective_kv_bytes_per_token
# ---------------------------------------------------------------------------


def test_the_effective_per_token_cost_is_unchanged_for_a_uniform_model() -> None:
    meta = meta_uniform()
    for kv_type in ("f16", "q8_0", "q4_0"):
        assert effective_kv_bytes_per_token(
            meta, kv_k=kv_type, kv_v=kv_type, ctx_per_slot=16384
        ) == kv_bytes_per_token(meta, kv_type, kv_type)


def test_gemma4_costs_a_fraction_of_its_uniform_per_token_figure() -> None:
    """1.9 MB/token was the number every slot count divided by; ~155 KB is real."""
    meta = meta_gemma4_31b()
    uniform = kv_bytes_per_token(meta, "f16", "f16")
    assert uniform == 1920 * KIB
    effective = effective_kv_bytes_per_token(meta, kv_k="f16", kv_v="f16", ctx_per_slot=16384)
    assert effective == pytest.approx(155 * KIB, rel=0.05)
    assert effective < uniform / 12


def test_the_effective_cost_falls_as_the_context_grows() -> None:
    """A 1024-token window costs the same at 262k as at 16k, so the average drops.

    This is why the function takes a context instead of being a constant of the
    model -- a caller that caches one value per model gets the wrong answer for
    every tier but the one it sampled.
    """
    meta = meta_gemma4_31b()
    at_16k = effective_kv_bytes_per_token(meta, kv_k="f16", kv_v="f16", ctx_per_slot=16384)
    at_262k = effective_kv_bytes_per_token(meta, kv_k="f16", kv_v="f16", ctx_per_slot=262144)
    assert at_262k < at_16k
    assert at_262k == pytest.approx(85 * KIB, rel=0.05)


# ---------------------------------------------------------------------------
# kv_read_bytes_per_slot -- what decode actually pays
# ---------------------------------------------------------------------------


def test_a_window_layer_reads_its_window_however_long_the_transcript_is() -> None:
    """Gemma-4 at a 131k fill: ~11.6 GB per step, not the uniform 258 GB.

    The uniform figure produced ``est_gen_tps: 1.9`` for a 31B on two 5090s that
    measures 39.4. Ten global layers read the whole transcript; the other fifty
    read 1024 tokens each, forever.
    """
    meta = meta_gemma4_31b()
    read = kv_read_bytes_per_slot(meta, kv_k="f16", kv_v="f16", ctx_fill=131072)

    globals_bytes = 10 * 131072 * 4 * (512 + 512) * 2
    windows_bytes = 50 * 1024 * 16 * (256 + 256) * 2
    assert read == globals_bytes + windows_bytes

    uniform = 131072 * kv_bytes_per_token(meta, "f16", "f16")
    assert read < uniform / 20


def test_a_uniform_model_reads_exactly_the_fill_it_holds() -> None:
    meta = meta_uniform()
    assert kv_read_bytes_per_slot(meta, kv_k="f16", kv_v="f16", ctx_fill=8192) == (
        8192 * kv_bytes_per_token(meta, "f16", "f16")
    )


def test_a_hybrid_model_reads_its_recurrent_state_once_per_step() -> None:
    meta = meta_qwen35_27b()
    read = kv_read_bytes_per_slot(meta, kv_k="f16", kv_v="f16", ctx_fill=8192)
    cached = 16 * 8192 * 4 * (256 + 256) * 2
    assert read == cached + recurrent_state_bytes_per_slot(meta)


# ---------------------------------------------------------------------------
# The knee, on real read bytes
# ---------------------------------------------------------------------------


def test_the_knee_uses_the_real_read_bytes_when_it_is_given_them() -> None:
    """Same model, same budget: the uniform KV traffic pins the knee at 2."""
    meta = meta_gemma4_31b()
    weights = int(30.4 * GB_DEC)
    common = {
        "kv_budget_bytes": 1_000 * GB_DEC,  # deliberately not the binding term
        "kv_per_token": kv_bytes_per_token(meta, "f16", "f16"),
        "ctx_per_slot": 16384,
        "active_weight_bytes": weights,
    }
    naive, naive_bound = max_parallel_for(**common)
    assert (naive, naive_bound) == (2, "knee")

    real, real_bound = max_parallel_for(
        **common,
        kv_read_bytes_per_slot=kv_read_bytes_per_slot(meta, kv_k="f16", kv_v="f16", ctx_fill=8192),
    )
    assert real > naive
    assert real_bound in {"knee", "cap"}


def test_the_vram_bound_of_max_parallel_for_is_unchanged() -> None:
    """Only the knee term learned anything; other callers must see no drift."""
    with_read = max_parallel_for(
        kv_budget_bytes=8 * GB_DEC,
        kv_per_token=256 * KIB,
        ctx_per_slot=32768,
        active_weight_bytes=1 * GB_DEC,
        kv_read_bytes_per_slot=1,  # a knee of a billion; VRAM must still win
    )
    assert with_read == (1, "vram")


# ---------------------------------------------------------------------------
# max_slots_by_vram -- the exact walk
# ---------------------------------------------------------------------------


def planner_with(free_gib: float = 40.0) -> Planner:
    config = make_config()
    config.models.default_parallel = "auto"
    return Planner(config, StubProbe([gpu(0, 48.0, free_gib, (12, 0))]), log_plans=False)


def kv_budget_capacity(planner: Planner, record, *, ctx: int, budget: int) -> int:
    """Capacity that leaves exactly ``budget`` bytes for KV at one slot."""
    base = planner.estimate(
        record,
        ctx_size=ctx,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    return base.total_bytes - base.kv_bytes + budget


def test_gemma4_with_34_gb_of_kv_budget_at_16k_gets_more_than_one_slot() -> None:
    """The bug: 34 GB free and every Gemma-4 row said ``max_parallel: 1 (vram)``.

    ``kv_budget // (16384 x 1.9 MB/token)`` is 1. The real cache is 2.6 GB for
    the first slot and ~2.1 GB for each one after it, so the honest answer is
    the cap.
    """
    planner = planner_with()
    record = make_record(meta=meta_gemma4_31b(), size_bytes=int(30.4 * GB_DEC))
    capacity = kv_budget_capacity(planner, record, ctx=16384, budget=34 * GB_DEC)

    slots, estimate = planner.max_slots_by_vram(
        record,
        ctx=16384,
        kv_k="f16",
        kv_v="f16",
        n_devices=1,
        capacity_bytes=capacity,
        cap=MAX_PARALLEL_CAP,
    )
    assert slots > 1
    assert estimate.total_bytes <= capacity


def test_the_walk_never_returns_fewer_than_one_slot() -> None:
    """One slot is what a load does today; auto can only ever add (D17)."""
    planner = planner_with()
    record = make_record(meta=meta_gemma4_31b(), size_bytes=int(30.4 * GB_DEC))
    slots, estimate = planner.max_slots_by_vram(
        record,
        ctx=262144,
        kv_k="f16",
        kv_v="f16",
        n_devices=1,
        capacity_bytes=0,
        cap=MAX_PARALLEL_CAP,
    )
    assert slots == 1
    assert estimate.total_bytes > 0


def test_the_walk_returns_the_estimate_at_the_slot_count_it_chose() -> None:
    planner = planner_with()
    record = make_record(meta=make_meta(tensor_bytes=4 * GB), size_bytes=4 * GB)
    capacity = kv_budget_capacity(planner, record, ctx=8192, budget=6 * GB_DEC)
    slots, estimate = planner.max_slots_by_vram(
        record,
        ctx=8192,
        kv_k="f16",
        kv_v="f16",
        n_devices=1,
        capacity_bytes=capacity,
        cap=MAX_PARALLEL_CAP,
    )
    direct = planner.estimate(
        record,
        ctx_size=8192,
        parallel=slots,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    assert estimate.total_bytes == direct.total_bytes


def test_a_per_model_cap_bounds_the_walk() -> None:
    planner = planner_with()
    record = make_record(meta=make_meta(tensor_bytes=2 * GB), size_bytes=2 * GB)
    slots, _ = planner.max_slots_by_vram(
        record,
        ctx=4096,
        kv_k="f16",
        kv_v="f16",
        n_devices=1,
        capacity_bytes=40 * GB,
        cap=3,
    )
    assert slots == 3


# ---------------------------------------------------------------------------
# size_slots -- the two bounds together
# ---------------------------------------------------------------------------


def size_slots_for(planner: Planner, record, *, ctx: int, capacity: int):
    base = planner.estimate(
        record,
        ctx_size=ctx,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    return planner.size_slots(
        record,
        ctx=ctx,
        kv_k="f16",
        kv_v="f16",
        devices=[0],
        capacity_bytes=capacity,
        base_estimate=base,
    )


def test_size_slots_gives_gemma4_real_concurrency() -> None:
    planner = planner_with()
    record = make_record(meta=meta_gemma4_31b(), size_bytes=int(30.4 * GB_DEC))
    capacity = kv_budget_capacity(planner, record, ctx=16384, budget=34 * GB_DEC)
    estimate, slots, max_parallel, bound = size_slots_for(
        planner, record, ctx=16384, capacity=capacity
    )
    assert slots > 1
    assert max_parallel == slots
    assert bound in {"vram", "knee", "cap"}
    assert estimate.total_bytes <= capacity


def test_size_slots_reports_one_slot_and_says_it_does_not_know_without_metadata() -> None:
    """A weights-only estimate would let the walk return the cap: an invention."""
    planner = planner_with()
    record = make_record(meta=GgufMeta(), size_bytes=2 * GB)
    _, slots, max_parallel, bound = size_slots_for(planner, record, ctx=8192, capacity=40 * GB)
    assert (slots, max_parallel, bound) == (1, 1, "unknown")


def test_size_slots_estimate_matches_the_slot_count_it_returns() -> None:
    """The plan's VRAM figure must describe the load it actually describes."""
    planner = planner_with()
    record = make_record(meta=meta_qwen35_27b(), size_bytes=int(19.5 * GB_DEC))
    capacity = kv_budget_capacity(planner, record, ctx=32768, budget=12 * GB_DEC)
    estimate, slots, _, _ = size_slots_for(planner, record, ctx=32768, capacity=capacity)
    direct = planner.estimate(
        record,
        ctx_size=32768,
        parallel=slots,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    assert estimate.total_bytes == direct.total_bytes


def test_size_parallel_is_a_delegate_to_size_slots() -> None:
    planner = planner_with()
    record = make_record(meta=meta_gemma4_31b(), size_bytes=int(30.4 * GB_DEC))
    capacity = kv_budget_capacity(planner, record, ctx=16384, budget=34 * GB_DEC)
    base = planner.estimate(
        record,
        ctx_size=16384,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=1,
    )
    legacy = planner._size_parallel(  # type: ignore[attr-defined]
        record,
        ctx=16384,
        kv_k="f16",
        kv_v="f16",
        devices=[0],
        draft=None,
        draft_ctx=None,
        adapters=(),
        base_estimate=base,
        total_capacity=capacity,
    )
    assert legacy[1:] == size_slots_for(planner, record, ctx=16384, capacity=capacity)[1:]


# ---------------------------------------------------------------------------
# The plan carries the effective figure
# ---------------------------------------------------------------------------


def test_the_plan_carries_the_effective_per_token_cost_not_the_uniform_one() -> None:
    from studioforge.types import LoadPlan

    config = make_config()
    config.models.default_parallel = "auto"
    planner = Planner(config, StubProbe([gpu(0, 48.0, 44.0, (12, 0))]), log_plans=False)
    record = make_record(meta=meta_gemma4_31b(), size_bytes=int(30.4 * GB_DEC))
    plan = planner.plan_load(record, ctx_size=16384)
    assert isinstance(plan, LoadPlan)
    assert plan.kv_bytes_per_token == effective_kv_bytes_per_token(
        record.meta, kv_k=plan.kv_cache_type, kv_v=plan.kv_cache_type_v, ctx_per_slot=16384
    )
    assert plan.kv_bytes_per_token < kv_bytes_per_token(
        record.meta, plan.kv_cache_type, plan.kv_cache_type_v
    )
