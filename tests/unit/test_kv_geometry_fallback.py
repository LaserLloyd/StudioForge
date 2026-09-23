"""Unknown KV geometry is never planned as a free KV cache (D69 §11).

``kv_layers`` returns ``[]`` when a GGUF lacks a head dimension or every head
count. ``kv_alloc_bytes`` summed that into zero, ``Planner.estimate`` added the
zero, and the slot sizer answered one slot "unknown": the model was planned
from its weights alone and "fit". No library model hits this today; a new
architecture or a truncated conversion would.

Now a model with a layer count and a width is charged a pessimistic fallback
(every layer full multi-head attention, ``n_embd`` wide, at f16), and a model
with neither is refused with a reason that says so.
"""

from __future__ import annotations

from studioforge.core.planner import (
    FALLBACK_KV_CACHE_TYPE,
    Planner,
    fallback_kv_layers,
    kv_alloc_bytes,
    kv_geometry_unknown,
    kv_layers,
)
from studioforge.types import GB, GgufMeta, LoadPlan, LoadRejected
from tests.unit.test_planner import make_config, make_meta, make_record, rig_5090x2_3090x2


def _no_heads() -> GgufMeta:
    """Layer count and width, but no head count and no head dimension."""
    return GgufMeta(
        architecture="future-arch",
        n_layer=32,
        n_embd=4096,
        n_head=0,
        n_head_kv=0,
        n_ctx_train=32768,
        tensor_bytes=8 * GB,
        quant_label="Q4_K_M",
    )


def _nothing_known() -> GgufMeta:
    """Heads, but no width, no head dimension and no layer count."""
    return GgufMeta(
        architecture="future-arch",
        n_layer=0,
        n_embd=0,
        n_head=0,
        n_head_kv=8,
        tensor_bytes=8 * GB,
        quant_label="Q4_K_M",
    )


def test_partial_geometry_is_charged_a_pessimistic_fallback_at_f16() -> None:
    meta = _no_heads()
    assert kv_layers(meta) == []
    layers = fallback_kv_layers(meta)
    assert len(layers) == 32 and all(layer.kind == "full" for layer in layers)
    assert not kv_geometry_unknown(meta)

    width = 4096  # n_embd: K and V each, per token, per layer -- no GQA assumed
    expected = 32 * 8192 * (width * 2 + width * 2)  # f16 = 2 bytes per element
    assert FALLBACK_KV_CACHE_TYPE == "f16"
    assert kv_alloc_bytes(meta, ctx_total=8192, kv_k="f16", kv_v="f16") == expected
    # A quantized cache is not believed for a guessed geometry.
    assert kv_alloc_bytes(meta, ctx_total=8192, kv_k="q4_0", kv_v="q4_0") == expected


def test_a_known_geometry_never_takes_the_fallback() -> None:
    meta = make_meta()
    assert kv_layers(meta)
    uniform = 32 * 8192 * (8 * 128 * 2 + 8 * 128 * 2)
    assert kv_alloc_bytes(meta, ctx_total=8192, kv_k="f16", kv_v="f16") == uniform
    # ...and a real quantized cache is still honoured there.
    assert kv_alloc_bytes(meta, ctx_total=8192, kv_k="q8_0", kv_v="q8_0") < uniform


def test_a_fallback_model_is_planned_with_its_kv_cache_never_zero() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    record = make_record(meta=_no_heads())

    plan = planner.plan_load(record, ctx_size=8192, parallel=1)

    assert isinstance(plan, LoadPlan)
    assert plan.estimate.kv_bytes >= 32 * 8192 * 4096 * 4
    fit = planner.fits_on(record, devices=[0], ctx_size=8192)
    assert fit is not None and fit.kv_bytes > 0


def test_nothing_known_is_refused_naming_the_missing_keys() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    record = make_record(meta=_nothing_known(), size_bytes=8 * GB)
    assert kv_geometry_unknown(record.meta)

    result = planner.plan_load(record, ctx_size=4096)

    assert isinstance(result, LoadRejected)
    assert result.reason.startswith("KV geometry unknown")
    for key in ("block_count", "embedding_length", "attention.key_length"):
        assert key in result.reason
    assert result.estimate.weights_bytes == 8 * GB
    assert result.estimate.kv_bytes == 0
    assert result.suggestions
    message = result.message()
    assert "KV geometry unknown" in message and "0.00 GiB" not in message
    # The read-only fit question agrees with the load.
    assert planner.fits_on(record, devices=[0, 1], ctx_size=4096) is None


def test_a_record_without_metadata_is_refused_the_same_way() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    record = make_record(size_bytes=2 * GB)
    record.meta = None

    result = planner.plan_load(record, ctx_size=4096)

    assert isinstance(result, LoadRejected)
    assert "no parsed GGUF metadata" in result.reason
    assert result.estimate.weights_bytes == 2 * GB
