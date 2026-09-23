"""A MoE's compute term is sized from the weights that run (D71).

The planner charged every model ``compute_overhead_fraction`` of its *whole*
weight size for graph buffers. For a mixture-of-experts that fraction was of
bytes that never run together: the live Qwen3.5-122B-A10B (94.6 % routed
experts) was charged 12.4 GB of compute and measured under 1 GB, Hy-MT2-30B-A3B
2.45 GB against ~0.2-0.7 GB. A MoE's compute term now takes the fraction of
the dense trunk plus the routed share of the experts, keeps the floor per
device (each card has its own compute buffer) and charges each card's copy of
the attention mask. Dense models are untouched, byte for byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studioforge.config import Config
from studioforge.core import gguf
from studioforge.core.gguf import (
    STRING,
    UINT32,
    expert_tensor_bytes,
    meta_from_gguf,
    read_gguf,
    read_meta,
    shard_paths_for,
)
from studioforge.core.planner import (
    DEFAULT_UBATCH,
    MB,
    Planner,
    compute_basis_bytes,
    kq_mask_bytes,
)
from studioforge.types import GgufMeta, ModelRecord, ModelSettings
from tests.unit.test_gguf import KvEntry, TensorSpec, expected_bytes, write_gguf

F16 = 1  # ggml type id


# ---------------------------------------------------------------------------
# GGUF: the parser counts the routed experts
# ---------------------------------------------------------------------------


def _moe_kv(n_expert: int = 8, n_used: int = 2) -> list[KvEntry]:
    return [
        ("general.architecture", STRING, "qwen3moe"),
        ("qwen3moe.block_count", UINT32, 1),
        ("qwen3moe.embedding_length", UINT32, 64),
        ("qwen3moe.attention.head_count", UINT32, 4),
        ("qwen3moe.attention.head_count_kv", UINT32, 2),
        ("qwen3moe.context_length", UINT32, 4096),
        ("qwen3moe.expert_count", UINT32, n_expert),
        ("qwen3moe.expert_used_count", UINT32, n_used),
    ]


_TRUNK: list[TensorSpec] = [
    ("token_embd.weight", (64, 256), F16),
    ("blk.0.attn_q.weight", (64, 64), F16),
    ("blk.0.ffn_gate_inp.weight", (64, 8), F16),
    # A shared expert runs on every token: trunk, not routed.
    ("blk.0.ffn_gate_shexp.weight", (64, 32), F16),
]
_EXPERTS: list[TensorSpec] = [
    ("blk.0.ffn_gate_exps.weight", (64, 32, 8), F16),
    ("blk.0.ffn_up_exps.weight", (64, 32, 8), F16),
    ("blk.0.ffn_down_exps.weight", (32, 64, 8), F16),
]


def _bytes(specs: list[TensorSpec]) -> int:
    return sum(expected_bytes(dims, ggml_type) for _name, dims, ggml_type in specs)


def test_the_parser_counts_routed_experts_and_not_the_shared_one(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "moe.gguf", _moe_kv(), _TRUNK + _EXPERTS)
    meta = read_meta(path)
    assert meta.extra["expert_tensor_bytes"] == _bytes(_EXPERTS)
    assert meta.tensor_bytes == _bytes(_TRUNK) + _bytes(_EXPERTS)
    assert expert_tensor_bytes(read_gguf(path).tensors) == _bytes(_EXPERTS)


def test_a_dense_model_records_no_expert_bytes(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("llama.block_count", UINT32, 1),
        ("llama.embedding_length", UINT32, 64),
        ("llama.attention.head_count", UINT32, 4),
    ]
    meta = read_meta(write_gguf(tmp_path / "dense.gguf", kv, _TRUNK))
    assert "expert_tensor_bytes" not in meta.extra


def test_a_split_model_sums_its_experts_across_every_shard(tmp_path: Path) -> None:
    first = write_gguf(tmp_path / "moe-00001-of-00002.gguf", _moe_kv(), _TRUNK + _EXPERTS[:1])
    write_gguf(tmp_path / "moe-00002-of-00002.gguf", _moe_kv(), _EXPERTS[1:])
    meta = read_meta(first, shard_paths=shard_paths_for(first))
    assert meta.extra["expert_tensor_bytes"] == _bytes(_EXPERTS)


def test_a_split_model_with_a_missing_shard_keeps_the_conservative_charge(
    tmp_path: Path,
) -> None:
    first = write_gguf(tmp_path / "moe-00001-of-00002.gguf", _moe_kv(), _TRUNK + _EXPERTS[:1])
    missing = tmp_path / "moe-00002-of-00002.gguf"
    meta = read_meta(first, shard_paths=[first, missing])
    # Half an expert count would make the trunk look bigger than it is; none
    # at all falls back to charging the whole weights.
    assert "expert_tensor_bytes" not in meta.extra


def test_a_header_read_without_the_tensor_table_has_no_count(tmp_path: Path) -> None:
    """The remote (HuggingFace) reader parses with ``load_tensors=False``."""
    path = write_gguf(tmp_path / "moe.gguf", _moe_kv(), _TRUNK + _EXPERTS)
    parsed = read_gguf(path, load_tensors=False)
    meta = meta_from_gguf(parsed, path=path, tensor_bytes=123_456, local=False)
    assert "expert_tensor_bytes" not in meta.extra


def test_the_meta_format_version_moved_so_every_model_recounts() -> None:
    assert gguf.META_FORMAT_VERSION == 4


# ---------------------------------------------------------------------------
# The basis and the mask
# ---------------------------------------------------------------------------


def _meta(**over: Any) -> GgufMeta:
    values: dict[str, Any] = {
        "architecture": "qwen35moe",
        "n_layer": 48,
        "n_embd": 3072,
        "n_head": 32,
        "n_head_kv": 2,
        "n_ctx_train": 262144,
        "n_vocab": 248320,
        "n_embd_head_k": 256,
        "n_embd_head_v": 256,
        "n_expert": 256,
        "n_expert_used": 8,
        "tensor_bytes": 82917 * MB,
        "extra": {"expert_tensor_bytes": 78480 * MB},
    }
    values.update(over)
    return GgufMeta(**values)


def test_the_basis_is_the_trunk_plus_the_routed_share() -> None:
    meta = _meta()
    weights = 82917 * MB
    trunk = (82917 - 78480) * MB
    assert compute_basis_bytes(meta, weights) == trunk + 78480 * MB * 8 // 256


def test_a_dense_model_basis_is_its_whole_weights() -> None:
    meta = _meta(n_expert=0, n_expert_used=0, extra={})
    assert compute_basis_bytes(meta, 20 * 1024 * MB) == 20 * 1024 * MB


def test_a_moe_without_a_count_keeps_the_whole_weights() -> None:
    """No count (remote header, incomplete shards): the old, conservative charge."""
    assert compute_basis_bytes(_meta(extra={}), 1000 * MB) == 1000 * MB


def test_nonsense_counts_keep_the_whole_weights() -> None:
    weights = 1000 * MB
    assert compute_basis_bytes(_meta(extra={"expert_tensor_bytes": 2000 * MB}), weights) == weights
    assert compute_basis_bytes(_meta(extra={"expert_tensor_bytes": True}), weights) == weights
    assert compute_basis_bytes(_meta(extra={"expert_tensor_bytes": "900"}), weights) == weights
    assert compute_basis_bytes(_meta(n_expert_used=256), weights) == weights  # all experts run


def test_the_mask_is_f16_cells_by_micro_batch() -> None:
    assert kq_mask_bytes(262144, 512) == 262144 * 512 * 2  # 256 MiB per device
    assert kq_mask_bytes(16384, 0) == 16384 * DEFAULT_UBATCH * 2
    assert kq_mask_bytes(16384, 1024) == 2 * kq_mask_bytes(16384, 512)


# ---------------------------------------------------------------------------
# The planner's estimate
# ---------------------------------------------------------------------------


class _NoGpus:
    def list_gpus(self) -> list[Any]:
        return []


def _planner() -> Planner:
    config = Config()
    config.planner.compute_overhead_fraction = 0.15
    config.planner.compute_overhead_floor_mb = 400
    return Planner(config, _NoGpus(), log_plans=False)  # type: ignore[arg-type]


def _record(meta: GgufMeta, **settings: Any) -> ModelRecord:
    return ModelRecord(
        id="pub/repo/model",
        name="model",
        path=Path("/models/model.gguf"),
        size_bytes=meta.tensor_bytes,
        architecture=meta.architecture,
        meta=meta,
        settings=ModelSettings(**settings),
    )


def _compute(record: ModelRecord, *, ctx: int, parallel: int = 1, n_devices: int = 1) -> int:
    return (
        _planner()
        .estimate(
            record,
            ctx_size=ctx,
            parallel=parallel,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            n_devices=n_devices,
            ubatch=512,
        )
        .compute_bytes
    )


def test_the_122b_is_charged_what_it_measured_not_twelve_gigabytes() -> None:
    """The live 122B-A10B at 16k x 4 on four cards: 12,438 MiB before, ~1.6 GiB now."""
    record = _record(_meta())
    basis = compute_basis_bytes(record.meta, 82917 * MB)
    compute = _compute(record, ctx=16384, parallel=4, n_devices=4)
    expected = max(400 * MB * 4, int(basis * 0.15)) + kq_mask_bytes(16384, 512) * 4
    assert compute == expected
    assert 1500 * MB < compute < 1800 * MB
    assert compute < int(82917 * MB * 0.15) // 7  # the phantom is gone


def test_the_floor_is_per_device_for_a_moe() -> None:
    small = _record(
        _meta(
            tensor_bytes=16353 * MB,
            extra={"expert_tensor_bytes": 15348 * MB},
            n_expert=128,
            n_expert_used=8,
            n_embd=2048,
            n_vocab=120832,
            n_head_kv=4,
            n_embd_head_k=128,
            n_embd_head_v=128,
        )
    )
    one = _compute(small, ctx=16384, n_devices=1)
    three = _compute(small, ctx=16384, n_devices=3)
    mask = kq_mask_bytes(16384, 512)
    assert one == 400 * MB + mask
    assert three == 3 * 400 * MB + 3 * mask


def test_the_mask_follows_the_context_for_a_moe() -> None:
    record = _record(_meta())
    short = _compute(record, ctx=16384, n_devices=4)
    long = _compute(record, ctx=262144, n_devices=4)
    assert long - short == (kq_mask_bytes(262144, 512) - kq_mask_bytes(16384, 512)) * 4


def test_a_unified_kv_masks_every_slot_s_cells() -> None:
    split = _compute(_record(_meta()), ctx=16384, parallel=4, n_devices=1)
    unified = _compute(_record(_meta(), kv_unified=True), ctx=16384, parallel=4, n_devices=1)
    assert unified - split == kq_mask_bytes(16384 * 4, 512) - kq_mask_bytes(16384, 512)


def test_a_dense_model_is_charged_exactly_as_before() -> None:
    """D71 changes nothing for dense models: no per-device floor, no mask."""
    dense = _record(_meta(n_expert=0, n_expert_used=0, extra={}, tensor_bytes=21642 * MB))
    for n_devices in (1, 2, 4):
        for ctx in (8192, 262144):
            assert _compute(dense, ctx=ctx, n_devices=n_devices) == max(
                400 * MB, int(21642 * MB * 0.15)
            )


def test_a_moe_without_an_expert_count_is_charged_exactly_as_before() -> None:
    uncounted = _record(_meta(extra={}))
    assert _compute(uncounted, ctx=262144, n_devices=4) == int(82917 * MB * 0.15)
