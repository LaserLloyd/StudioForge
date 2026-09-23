"""Keys captured for a later KV sizing change, with the sizing itself unchanged (D69 §13).

laguna (``Laguna-S-2.1``) declares ``attention.sliding_window = 512`` with no
``sliding_window_pattern``, a per-layer ``attention.head_count`` array and
``leading_dense_block_count``; MLA models (DeepSeek2/3, Kimi, GLM-DSA) declare
``kv_lora_rank`` and friends. The parser dropped all of them, so nothing could
be checked against upstream later. They are now kept in ``GgufMeta.extra``;
what llama.cpp does with them is unverified, so the planner still charges the
uniform full-attention cache for these models, byte for byte as before.
"""

from __future__ import annotations

from pathlib import Path

from studioforge.core import gguf
from studioforge.core.gguf import ARRAY, STRING, UINT32, read_meta
from studioforge.core.planner import kv_alloc_bytes, kv_layers
from studioforge.types import GgufMeta
from tests.unit.test_gguf import Arr, KvEntry, llm_kv, write_gguf

NEW_KEYS = (
    "sliding_window",
    "head_count_values",
    "head_count_len",
    "leading_dense_block_count",
    "kv_lora_rank",
    "q_lora_rank",
    "key_length_mla",
    "value_length_mla",
    "rope_dimension_count",
)


def _without_new_keys(meta: GgufMeta) -> GgufMeta:
    extra = {k: v for k, v in meta.extra.items() if k not in NEW_KEYS}
    return meta.model_copy(update={"extra": extra})


def _laguna(tmp_path: Path) -> GgufMeta:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "laguna"),
        ("laguna.block_count", UINT32, 6),
        ("laguna.embedding_length", UINT32, 2048),
        ("laguna.attention.head_count", ARRAY, Arr(UINT32, [48, 72, 48, 72, 48, 72])),
        ("laguna.attention.head_count_kv", UINT32, 8),
        ("laguna.attention.key_length", UINT32, 128),
        ("laguna.attention.value_length", UINT32, 128),
        ("laguna.attention.sliding_window", UINT32, 512),
        ("laguna.leading_dense_block_count", UINT32, 1),
        ("laguna.expert_count", UINT32, 256),
    ]
    return read_meta(write_gguf(tmp_path / "laguna.gguf", kv))


def test_the_meta_format_version_was_bumped_so_cached_rows_reparse() -> None:
    assert gguf.META_FORMAT_VERSION == 3


def test_laguna_keeps_its_window_head_counts_and_dense_prefix(tmp_path: Path) -> None:
    meta = _laguna(tmp_path)

    assert meta.extra["sliding_window"] == 512
    assert meta.extra["head_count_values"] == [48, 72, 48, 72, 48, 72]
    assert meta.extra["head_count_len"] == 6
    assert meta.extra["leading_dense_block_count"] == 1
    assert meta.n_head == 72  # the scalar collapse is unchanged: the max
    # No pattern, so the planner's iSWA keys stay absent...
    assert "swa_window" not in meta.extra and "swa_pattern" not in meta.extra
    # ...and the sizing rule is exactly what it was: six uniform full layers.
    layers = kv_layers(meta)
    assert [layer.kind for layer in layers] == ["full"] * 6
    for kv_type in ("f16", "q8_0"):
        assert kv_alloc_bytes(meta, ctx_total=131072, kv_k=kv_type, kv_v=kv_type) == (
            kv_alloc_bytes(_without_new_keys(meta), ctx_total=131072, kv_k=kv_type, kv_v=kv_type)
        )
    assert kv_alloc_bytes(meta, ctx_total=8192, kv_k="f16", kv_v="f16") == (
        6 * 8192 * (8 * 128 * 2 + 8 * 128 * 2)
    )


def test_mla_keys_and_their_rope_dimension_are_kept(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "deepseek2"),
        ("deepseek2.block_count", UINT32, 4),
        ("deepseek2.embedding_length", UINT32, 7168),
        ("deepseek2.attention.head_count", UINT32, 128),
        ("deepseek2.attention.head_count_kv", UINT32, 1),
        ("deepseek2.attention.key_length", UINT32, 576),
        ("deepseek2.attention.value_length", UINT32, 512),
        ("deepseek2.attention.kv_lora_rank", UINT32, 512),
        ("deepseek2.attention.q_lora_rank", UINT32, 1536),
        ("deepseek2.attention.key_length_mla", UINT32, 192),
        ("deepseek2.attention.value_length_mla", UINT32, 128),
        ("deepseek2.rope.dimension_count", UINT32, 64),
        ("deepseek2.leading_dense_block_count", UINT32, 3),
    ]
    meta = read_meta(write_gguf(tmp_path / "mla.gguf", kv))

    assert meta.extra["kv_lora_rank"] == 512
    assert meta.extra["q_lora_rank"] == 1536
    assert meta.extra["key_length_mla"] == 192
    assert meta.extra["value_length_mla"] == 128
    assert meta.extra["rope_dimension_count"] == 64
    assert meta.extra["leading_dense_block_count"] == 3
    assert kv_alloc_bytes(meta, ctx_total=16384, kv_k="f16", kv_v="f16") == (
        kv_alloc_bytes(_without_new_keys(meta), ctx_total=16384, kv_k="f16", kv_v="f16")
    )


def test_an_iswa_model_keeps_both_the_planner_keys_and_the_raw_window(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        *llm_kv("gemma3", block_count=6, embedding_length=512),
        ("gemma3.attention.sliding_window", UINT32, 1024),
        ("gemma3.attention.sliding_window_pattern", ARRAY, Arr(1, [1, 1, 1, 1, 1, 0])),
    ]
    meta = read_meta(write_gguf(tmp_path / "iswa.gguf", kv))

    assert meta.extra["swa_window"] == 1024 and meta.extra["sliding_window"] == 1024
    assert meta.extra["swa_pattern"] == [True, True, True, True, True, False]
    assert kv_alloc_bytes(meta, ctx_total=65536, kv_k="f16", kv_v="f16") == (
        kv_alloc_bytes(_without_new_keys(meta), ctx_total=65536, kv_k="f16", kv_v="f16")
    )


def test_an_ordinary_model_gains_none_of_the_new_keys(tmp_path: Path) -> None:
    """A plain llama carries rope.dimension_count; without MLA it is not kept."""
    kv = [*llm_kv(), ("llama.rope.dimension_count", UINT32, 16)]
    meta = read_meta(write_gguf(tmp_path / "plain.gguf", kv))
    for key in NEW_KEYS:
        assert key not in meta.extra, key
