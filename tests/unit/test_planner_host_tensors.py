"""Host-resident tensors are not charged to a GPU (D73).

llama.cpp keeps the per-layer token-embedding table (``per_layer_token_embd``,
the PLE of Gemma-3n/E4B and Qwen3.8-Flash-Next) in host memory whatever ``-ngl``
says: it is an input-layer GET_ROWS tensor, placed on the CPU like
``token_embd``. The planner charged it to the GPUs as weights and as compute
basis, which for a 177B Qwen3.8-Flash-Next is 27-36 GiB of phantom VRAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from studioforge.config import Config
from studioforge.core import gguf
from studioforge.core.gguf import (
    STRING,
    UINT32,
    host_tensor_bytes,
    meta_from_gguf,
    read_gguf,
    read_meta,
    shard_paths_for,
)
from studioforge.core.hf_meta import _throwaway_record
from studioforge.core.planner import MB, Planner, compute_basis_bytes, device_weight_bytes
from studioforge.types import GgufMeta, ModelRecord, ModelSettings
from tests.unit.test_gguf import KvEntry, TensorSpec, expected_bytes, write_gguf

F16 = 1  # ggml type id


def _kv() -> list[KvEntry]:
    return [
        ("general.architecture", STRING, "qwen4exp"),
        ("qwen4exp.block_count", UINT32, 1),
        ("qwen4exp.embedding_length", UINT32, 64),
        ("qwen4exp.attention.head_count", UINT32, 4),
        ("qwen4exp.attention.head_count_kv", UINT32, 2),
        ("qwen4exp.context_length", UINT32, 4096),
        ("qwen4exp.expert_count", UINT32, 8),
        ("qwen4exp.expert_used_count", UINT32, 2),
    ]


_PLE: list[TensorSpec] = [("per_layer_token_embd.weight", (16, 4096), F16)]
_BODY: list[TensorSpec] = [
    # token_embd stays charged on purpose: a tied model duplicates it on the GPU.
    ("token_embd.weight", (64, 256), F16),
    ("blk.0.attn_q.weight", (64, 64), F16),
    ("blk.0.ffn_gate_exps.weight", (64, 32, 8), F16),
]


def _bytes(specs: list[TensorSpec]) -> int:
    return sum(expected_bytes(dims, ggml_type) for _name, dims, ggml_type in specs)


# ---------------------------------------------------------------------------
# GGUF: the parser counts the host-resident table
# ---------------------------------------------------------------------------


def test_the_parser_counts_the_ple_table_and_not_token_embd(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "ple.gguf", _kv(), _PLE + _BODY)
    meta = read_meta(path)
    assert meta.extra["host_tensor_bytes"] == _bytes(_PLE)
    assert meta.tensor_bytes == _bytes(_PLE) + _bytes(_BODY)
    assert host_tensor_bytes(read_gguf(path).tensors) == _bytes(_PLE)


def test_a_model_without_a_ple_records_nothing(tmp_path: Path) -> None:
    meta = read_meta(write_gguf(tmp_path / "plain.gguf", _kv(), _BODY))
    assert "host_tensor_bytes" not in meta.extra


def test_a_split_model_counts_the_table_in_whichever_shard_holds_it(tmp_path: Path) -> None:
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", _kv(), _BODY)
    write_gguf(tmp_path / "m-00002-of-00002.gguf", _kv(), _PLE)
    meta = read_meta(first, shard_paths=shard_paths_for(first))
    assert meta.extra["host_tensor_bytes"] == _bytes(_PLE)


def test_a_split_model_with_a_missing_shard_charges_everything(tmp_path: Path) -> None:
    first = write_gguf(tmp_path / "m-00001-of-00002.gguf", _kv(), _BODY + _PLE)
    meta = read_meta(first, shard_paths=[first, tmp_path / "m-00002-of-00002.gguf"])
    assert "host_tensor_bytes" not in meta.extra


def test_a_header_read_without_the_tensor_table_has_no_count(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "ple.gguf", _kv(), _PLE + _BODY)
    parsed = read_gguf(path, load_tensors=False)
    meta = meta_from_gguf(parsed, path=path, tensor_bytes=123_456, local=False)
    assert "host_tensor_bytes" not in meta.extra


def test_the_meta_format_version_moved_so_every_model_recounts() -> None:
    assert gguf.META_FORMAT_VERSION >= 6


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

GIB = 1024 * MB

#: The real i1-Q4_K_S of Qwen3.8-Flash-Next-Uncensored: 104.08 GiB of tensors,
#: 32.78 GiB of them the Q5_0 PLE table, 68.26 GiB routed experts.
Q4KS_TOTAL = int(104.08 * GIB)
Q4KS_PLE = int(32.78 * GIB)
Q4KS_EXPERTS = int(68.26 * GIB)


def _meta(**over: Any) -> GgufMeta:
    values: dict[str, Any] = {
        "architecture": "qwen4exp",
        "n_layer": 48,
        "n_embd": 2560,
        "n_head": 24,
        "n_head_kv": 2,
        "n_ctx_train": 262144,
        "n_vocab": 248320,
        "n_embd_head_k": 256,
        "n_embd_head_v": 256,
        "n_expert": 512,
        "n_expert_used": 10,
        "tensor_bytes": Q4KS_TOTAL,
        "extra": {
            "expert_tensor_bytes": Q4KS_EXPERTS,
            "host_tensor_bytes": Q4KS_PLE,
            "full_attention_interval": 4,
        },
    }
    values.update(over)
    return GgufMeta(**values)


def test_device_weights_are_the_file_minus_the_table() -> None:
    assert device_weight_bytes(_meta(), Q4KS_TOTAL) == Q4KS_TOTAL - Q4KS_PLE


def test_nonsense_counts_charge_the_whole_file() -> None:
    w = 1000 * MB
    assert device_weight_bytes(_meta(extra={}), w) == w
    assert device_weight_bytes(_meta(extra={"host_tensor_bytes": True}), w) == w
    assert device_weight_bytes(_meta(extra={"host_tensor_bytes": "5"}), w) == w
    assert device_weight_bytes(_meta(extra={"host_tensor_bytes": w}), w) == w  # nothing left
    assert device_weight_bytes(_meta(extra={"host_tensor_bytes": -1}), w) == w
    assert device_weight_bytes(None, w) == w


class _NoGpus:
    def list_gpus(self) -> list[Any]:
        return []


def _estimate(meta: GgufMeta) -> Any:
    config = Config()
    config.planner.compute_overhead_fraction = 0.15
    config.planner.compute_overhead_floor_mb = 400
    planner = Planner(config, _NoGpus(), log_plans=False)  # type: ignore[arg-type]
    record = ModelRecord(
        id="pub/repo/model",
        name="model",
        path=Path("/models/model.gguf"),
        size_bytes=meta.tensor_bytes,
        architecture=meta.architecture,
        meta=meta,
        settings=ModelSettings(),
    )
    return planner.estimate(
        record,
        ctx_size=200_000,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=3,
        ubatch=512,
    )


def test_the_estimate_charges_neither_weights_nor_compute_for_the_table() -> None:
    with_table = _estimate(_meta())
    extra = dict(_meta().extra)
    del extra["host_tensor_bytes"]
    without = _estimate(_meta(extra=extra))
    assert with_table.weights_bytes == Q4KS_TOTAL - Q4KS_PLE
    assert without.weights_bytes == Q4KS_TOTAL
    # The table was in the MoE's trunk, so it inflated the compute basis too.
    assert compute_basis_bytes(_meta(), Q4KS_TOTAL - Q4KS_PLE) < compute_basis_bytes(
        _meta(extra=extra), Q4KS_TOTAL
    )
    assert with_table.compute_bytes < without.compute_bytes
    assert with_table.kv_bytes == without.kv_bytes
    # ~33 GiB of weights plus ~5 GiB of compute fraction off a 3-card plan.
    assert without.total_bytes - with_table.total_bytes > 37 * GIB


def test_a_picker_estimate_for_another_quant_drops_the_siblings_table() -> None:
    """The sibling's table is the sibling's quant of it: drop, never guess."""
    record = _throwaway_record("pub/repo/Q5_K_M", _meta(), int(124.89 * GIB), 0)
    assert record.meta is not None
    assert "host_tensor_bytes" not in record.meta.extra
    assert record.meta.extra["expert_tensor_bytes"] == Q4KS_EXPERTS
    same = _throwaway_record("pub/repo/Q4_K_S", _meta(), Q4KS_TOTAL, 0)
    assert same.meta is not None
    assert same.meta.extra["host_tensor_bytes"] == Q4KS_PLE
