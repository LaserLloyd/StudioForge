"""Tests for the pure-Python GGUF reader.

Two halves:

* **Synthetic** -- tiny GGUF files assembled byte-by-byte in ``tmp_path``. These
  own the format contract (every value type, array truncation, tensor byte
  math, malformed input) because they can express cases no real file does.
* **Real library** -- every ``*.gguf`` under ``E:\\LLM\\Models``, skipped when
  that drive is absent. These catch what synthetic tests structurally cannot:
  metadata conventions that quantisers actually emit.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from studioforge.core.gguf import (
    ARRAY,
    BOOL,
    DEFAULT_ALIGNMENT,
    FLOAT32,
    FLOAT64,
    GGUF_MAGIC,
    INT8,
    INT16,
    INT32,
    INT64,
    KNOWN_QUANT_LABELS,
    STRING,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    GgufError,
    TensorInfo,
    is_gguf,
    looks_like_auxiliary_gguf,
    looks_like_mmproj,
    missing_shard_names,
    quant_label_from,
    quant_label_from_filename,
    read_gguf,
    read_meta,
    shard_paths_for,
)

# ---------------------------------------------------------------------------
# Synthetic GGUF writer
# ---------------------------------------------------------------------------

_PACK: dict[int, str] = {
    UINT8: "<B",
    INT8: "<b",
    UINT16: "<H",
    INT16: "<h",
    UINT32: "<I",
    INT32: "<i",
    FLOAT32: "<f",
    UINT64: "<Q",
    INT64: "<q",
    FLOAT64: "<d",
}

# Mirrors of the reader's tables, spelled out here on purpose: if a size in the
# implementation is edited by accident, these tests must not follow it.
_BLOCKS: dict[int, tuple[int, int]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    8: (32, 34),  # Q8_0
    12: (256, 144),  # Q4_K
    14: (256, 210),  # Q6_K
    21: (256, 110),  # IQ3_S
    30: (1, 2),  # BF16
    39: (32, 17),  # MXFP4
}
_UNKNOWN_FALLBACK = (256, 144)


@dataclass
class Arr:
    """An ARRAY metadata value: element type plus elements."""

    elem_type: int
    values: list[Any]


KvEntry = tuple[str, int, Any]
TensorSpec = tuple[str, tuple[int, ...], int]


def _enc_str(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _enc_scalar(value_type: int, value: Any) -> bytes:
    if value_type == STRING:
        return _enc_str(str(value))
    if value_type == BOOL:
        return struct.pack("<B", 1 if value else 0)
    return struct.pack(_PACK[value_type], value)


def _enc_value(value_type: int, value: Any) -> bytes:
    if value_type != ARRAY:
        return _enc_scalar(value_type, value)
    assert isinstance(value, Arr)
    out = struct.pack("<I", value.elem_type) + struct.pack("<Q", len(value.values))
    for item in value.values:
        out += _enc_scalar(value.elem_type, item)
    return out


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def expected_bytes(dims: tuple[int, ...], ggml_type: int) -> int:
    """Independent re-derivation of the on-disk size of one tensor."""
    n_elements = 1
    for dim in dims:
        n_elements *= dim
    block, size = _BLOCKS.get(ggml_type, _UNKNOWN_FALLBACK)
    return -(-n_elements // block) * size


def write_gguf(
    path: Path,
    kv: list[KvEntry],
    tensors: list[TensorSpec] | None = None,
    *,
    version: int = 3,
    alignment: int = DEFAULT_ALIGNMENT,
    declare_alignment: bool = False,
    write_data: bool = True,
    truncate_to: int | None = None,
    magic: bytes = GGUF_MAGIC,
) -> Path:
    """Serialise a complete, valid-by-construction GGUF file."""
    tensors = tensors or []
    entries = list(kv)
    if declare_alignment:
        entries.insert(0, ("general.alignment", UINT32, alignment))

    body = bytearray()
    body += magic
    body += struct.pack("<I", version)
    body += struct.pack("<Q", len(tensors))
    body += struct.pack("<Q", len(entries))
    for key, value_type, value in entries:
        body += _enc_str(key)
        body += struct.pack("<I", value_type)
        body += _enc_value(value_type, value)

    # Tensor offsets are relative to the data section and each tensor starts on
    # an alignment boundary, exactly as llama.cpp's writer does it.
    data_cursor = 0
    total_data = 0
    for name, dims, ggml_type in tensors:
        body += _enc_str(name)
        body += struct.pack("<I", len(dims))
        for dim in dims:
            body += struct.pack("<Q", dim)
        body += struct.pack("<I", ggml_type)
        body += struct.pack("<Q", data_cursor)
        size = expected_bytes(dims, ggml_type)
        data_cursor = _align_up(data_cursor + size, alignment)
        total_data = data_cursor

    body += b"\x00" * (_align_up(len(body), alignment) - len(body))
    if write_data:
        body += b"\x00" * total_data

    payload = bytes(body) if truncate_to is None else bytes(body)[:truncate_to]
    path.write_bytes(payload)
    return path


def llm_kv(arch: str = "llama", **over: Any) -> list[KvEntry]:
    """A minimal but realistic set of text-model metadata keys."""
    values: dict[str, Any] = {
        "block_count": 4,
        "embedding_length": 128,
        "attention.head_count": 8,
        "context_length": 4096,
    }
    values.update(over)
    kv: list[KvEntry] = [("general.architecture", STRING, arch)]
    for key, value in values.items():
        if value is None:
            continue
        if key.startswith("general.") or key.startswith("tokenizer."):
            continue
        if isinstance(value, float):
            kv.append((f"{arch}.{key}", FLOAT32, value))
        elif isinstance(value, Arr):
            kv.append((f"{arch}.{key}", ARRAY, value))
        else:
            kv.append((f"{arch}.{key}", UINT32, value))
    for key, value in over.items():
        if key.startswith("general.") or key.startswith("tokenizer."):
            if isinstance(value, str):
                kv.append((key, STRING, value))
            elif isinstance(value, Arr):
                kv.append((key, ARRAY, value))
            elif isinstance(value, float):
                kv.append((key, FLOAT32, value))
            else:
                kv.append((key, UINT32, value))
    return kv


# ---------------------------------------------------------------------------
# Header / basic parsing
# ---------------------------------------------------------------------------


def test_parse_v3_minimal(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "m.gguf",
        llm_kv(),
        [("blk.0.attn_q.weight", (128, 128), 0)],
    )
    gguf = read_gguf(path)
    assert gguf.version == 3
    assert gguf.tensor_count == 1
    assert gguf.alignment == DEFAULT_ALIGNMENT
    assert gguf.kv["general.architecture"] == "llama"
    assert gguf.kv["llama.block_count"] == 4
    assert gguf.tensors[0].name == "blk.0.attn_q.weight"
    assert gguf.tensors[0].dims == (128, 128)
    assert gguf.tensors[0].n_bytes == 128 * 128 * 4
    assert gguf.total_tensor_bytes == 128 * 128 * 4
    assert gguf.data_offset % DEFAULT_ALIGNMENT == 0
    assert gguf.data_offset < path.stat().st_size


def test_parse_v2_is_supported(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "v2.gguf", llm_kv(), version=2)
    assert read_gguf(path).version == 2


def test_custom_alignment_shifts_data_offset(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "a.gguf",
        llm_kv(),
        [("t", (32,), 0)],
        alignment=1024,
        declare_alignment=True,
    )
    gguf = read_gguf(path)
    assert gguf.alignment == 1024
    assert gguf.data_offset % 1024 == 0


def test_non_power_of_two_alignment_rejected(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "bad.gguf", [("general.alignment", UINT32, 48), *llm_kv()])
    with pytest.raises(GgufError, match="power of two"):
        read_gguf(path)


def test_load_tensors_false_skips_table(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "m.gguf", llm_kv(), [("t", (128, 128), 0)])
    gguf = read_gguf(path, load_tensors=False)
    assert gguf.tensor_count == 1  # header count still reported
    assert gguf.tensors == []
    assert gguf.total_tensor_bytes == 0
    assert gguf.data_offset == 0


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


def test_all_scalar_value_types(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("t.u8", UINT8, 200),
        ("t.i8", INT8, -100),
        ("t.u16", UINT16, 60000),
        ("t.i16", INT16, -30000),
        ("t.u32", UINT32, 4000000000),
        ("t.i32", INT32, -2000000),
        ("t.f32", FLOAT32, 0.5),
        ("t.bool_true", BOOL, True),
        ("t.bool_false", BOOL, False),
        ("t.str", STRING, "hello \u00e9\u4e16"),
        ("t.u64", UINT64, 2**40),
        ("t.i64", INT64, -(2**40)),
        ("t.f64", FLOAT64, 1.0 / 3.0),
    ]
    gguf = read_gguf(write_gguf(tmp_path / "types.gguf", kv))
    assert gguf.kv["t.u8"] == 200
    assert gguf.kv["t.i8"] == -100
    assert gguf.kv["t.u16"] == 60000
    assert gguf.kv["t.i16"] == -30000
    assert gguf.kv["t.u32"] == 4000000000
    assert gguf.kv["t.i32"] == -2000000
    assert gguf.kv["t.f32"] == pytest.approx(0.5)
    assert gguf.kv["t.bool_true"] is True
    assert gguf.kv["t.bool_false"] is False
    assert gguf.kv["t.str"] == "hello \u00e9\u4e16"
    assert gguf.kv["t.u64"] == 2**40
    assert gguf.kv["t.i64"] == -(2**40)
    assert gguf.kv["t.f64"] == pytest.approx(1.0 / 3.0)


def test_short_arrays_are_plain_lists(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("t.ints", ARRAY, Arr(INT32, [1, 2, 3])),
        ("t.strs", ARRAY, Arr(STRING, ["a", "b"])),
        ("t.floats", ARRAY, Arr(FLOAT32, [1.5, 2.5])),
        ("t.empty", ARRAY, Arr(UINT32, [])),
    ]
    gguf = read_gguf(write_gguf(tmp_path / "arr.gguf", kv))
    assert gguf.kv["t.ints"] == [1, 2, 3]
    assert gguf.kv["t.strs"] == ["a", "b"]
    assert gguf.kv["t.floats"] == pytest.approx([1.5, 2.5])
    assert gguf.kv["t.empty"] == []


def test_long_numeric_array_truncated_but_counted(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("t.big", ARRAY, Arr(INT32, list(range(200)))),
        ("after", STRING, "still-parsed"),
    ]
    gguf = read_gguf(write_gguf(tmp_path / "big.gguf", kv), max_array_len=8)
    big = gguf.kv["t.big"]
    assert big["__array__"] is True
    assert big["type"] == INT32
    assert big["len"] == 200
    assert big["sample"] == list(range(8))
    # Small numeric arrays keep their full values: per-layer GQA lives here.
    assert big["values"] == list(range(200))
    # Parsing must resume exactly after the skipped payload.
    assert gguf.kv["after"] == "still-parsed"


def test_long_string_array_is_skipped_not_materialised(tmp_path: Path) -> None:
    tokens = [f"tok{i}" for i in range(5000)]
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("tokenizer.ggml.tokens", ARRAY, Arr(STRING, tokens)),
        ("tokenizer.ggml.model", STRING, "gpt2"),
    ]
    gguf = read_gguf(write_gguf(tmp_path / "vocab.gguf", kv))
    entry = gguf.kv["tokenizer.ggml.tokens"]
    assert entry["len"] == 5000
    assert len(entry["sample"]) == 64
    assert entry["sample"][0] == "tok0"
    assert "values" not in entry  # never materialised
    assert gguf.kv["tokenizer.ggml.model"] == "gpt2"


def test_bulk_numeric_arrays_are_never_materialised(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("tokenizer.ggml.scores", ARRAY, Arr(FLOAT32, [0.0] * 300)),
        ("tokenizer.ggml.token_type", ARRAY, Arr(INT32, [1] * 300)),
    ]
    gguf = read_gguf(write_gguf(tmp_path / "bulk.gguf", kv))
    for key in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
        entry = gguf.kv[key]
        assert entry["len"] == 300
        assert "values" not in entry


def test_huge_string_array_spanning_read_window(tmp_path: Path) -> None:
    """The fast string-skip loop must handle payloads larger than one window."""
    tokens = [f"token-{i:06d}-{'x' * 110}" for i in range(40000)]
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("tokenizer.ggml.tokens", ARRAY, Arr(STRING, tokens)),
        ("sentinel", STRING, "end"),
    ]
    path = write_gguf(tmp_path / "huge.gguf", kv, [("t", (32,), 0)])
    assert path.stat().st_size > (4 << 20)  # forces a window refill mid-array
    gguf = read_gguf(path)
    assert gguf.kv["tokenizer.ggml.tokens"]["len"] == 40000
    assert gguf.kv["sentinel"] == "end"
    assert gguf.tensors[0].dims == (32,)


# ---------------------------------------------------------------------------
# Tensor byte math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ggml_type", "dims", "want"),
    [
        (0, (256, 4), 256 * 4 * 4),  # F32
        (1, (256, 4), 256 * 4 * 2),  # F16
        (30, (256, 4), 256 * 4 * 2),  # BF16
        (2, (256, 4), 1024 // 32 * 18),  # Q4_0
        (8, (256, 4), 1024 // 32 * 34),  # Q8_0
        (12, (256, 4), 1024 // 256 * 144),  # Q4_K
        (14, (256, 4), 1024 // 256 * 210),  # Q6_K
        (21, (256, 4), 1024 // 256 * 110),  # IQ3_S
        (39, (256, 4), 1024 // 32 * 17),  # MXFP4
        (12, (128, 8, 2), 2048 // 256 * 144),  # 3-D tensor
    ],
)
def test_tensor_byte_math(tmp_path: Path, ggml_type: int, dims: tuple[int, ...], want: int) -> None:
    path = write_gguf(tmp_path / f"t{ggml_type}.gguf", llm_kv(), [("w", dims, ggml_type)])
    gguf = read_gguf(path)
    assert gguf.tensors[0].n_bytes == want
    assert gguf.total_tensor_bytes == want


def test_unknown_ggml_type_falls_back_and_is_reported(tmp_path: Path) -> None:
    path = write_gguf(
        tmp_path / "future.gguf",
        [*llm_kv(), ("general.file_type", UINT32, 15)],
        [("blk.0.ffn_down.weight", (256, 8), 250)],
    )
    gguf = read_gguf(path)
    assert gguf.tensors[0].n_bytes == 2048 // 256 * 144  # conservative estimate
    assert gguf.unknown_ggml_types == (250,)
    meta = read_meta(path)
    assert meta.extra["unknown_ggml_types"] == [250]
    assert meta.tensor_bytes == 2048 // 256 * 144


def test_tensor_info_helpers(tmp_path: Path) -> None:
    info = TensorInfo(name="w", dims=(4, 8), ggml_type=12, offset=0, n_bytes=99)
    assert info.n_elements == 32
    assert info.type_name == "Q4_K"
    assert TensorInfo(name="w", dims=(1,), ggml_type=250, offset=0, n_bytes=1).type_name == (
        "TYPE_250"
    )


# ---------------------------------------------------------------------------
# Malformed input
# ---------------------------------------------------------------------------


def test_garbage_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"NOPE" + os.urandom(512))
    with pytest.raises(GgufError, match="not a GGUF file"):
        read_gguf(path)
    assert is_gguf(path) is False


def test_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.gguf"
    path.write_bytes(b"")
    with pytest.raises(GgufError):
        read_gguf(path)


def test_truncated_file_raises(tmp_path: Path) -> None:
    full = write_gguf(tmp_path / "full.gguf", llm_kv(), [("blk.0.attn_q.weight", (128, 128), 0)])
    data = full.read_bytes()
    for cut in (8, 20, 40, 100):
        path = tmp_path / f"cut{cut}.gguf"
        path.write_bytes(data[:cut])
        with pytest.raises(GgufError):
            read_gguf(path)


@pytest.mark.parametrize("trim", [8, 200])
def test_truncated_mid_tensor_table_raises(tmp_path: Path, trim: int) -> None:
    """A file cut short of its own data section must be rejected.

    ``trim=8`` leaves the tensor table intact but puts the declared data offset
    past EOF; ``trim=200`` cuts into the table itself. Both are what a failed
    download looks like.
    """
    path = write_gguf(tmp_path / "t.gguf", llm_kv(), [("a", (64, 64), 0), ("b", (64, 64), 0)])
    data_offset = read_gguf(path).data_offset
    path.write_bytes(path.read_bytes()[: data_offset - trim])
    with pytest.raises(GgufError):
        read_gguf(path)


def test_absurd_counts_rejected(tmp_path: Path) -> None:
    path = tmp_path / "liar.gguf"
    path.write_bytes(
        GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 2**60) + struct.pack("<Q", 1)
    )
    with pytest.raises(GgufError, match="implausible tensor count"):
        read_gguf(path)

    path2 = tmp_path / "liar2.gguf"
    path2.write_bytes(
        GGUF_MAGIC + struct.pack("<I", 3) + struct.pack("<Q", 1) + struct.pack("<Q", 2**60)
    )
    with pytest.raises(GgufError, match="implausible metadata count"):
        read_gguf(path2)


def test_v1_and_big_endian_rejected(tmp_path: Path) -> None:
    v1 = tmp_path / "v1.gguf"
    v1.write_bytes(GGUF_MAGIC + struct.pack("<I", 1) + struct.pack("<I", 0) * 2)
    with pytest.raises(GgufError, match="v1 is not supported"):
        read_gguf(v1)

    swapped = write_gguf(tmp_path / "be.gguf", llm_kv(), magic=b"FUGG")
    with pytest.raises(GgufError, match="big-endian"):
        read_gguf(swapped)


def test_unknown_value_type_raises(tmp_path: Path) -> None:
    path = tmp_path / "badtype.gguf"
    body = bytearray(GGUF_MAGIC + struct.pack("<I", 3))
    body += struct.pack("<Q", 0) + struct.pack("<Q", 1)
    body += _enc_str("weird")
    body += struct.pack("<I", 99)
    path.write_bytes(bytes(body))
    with pytest.raises(GgufError, match="unknown GGUF value type"):
        read_gguf(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(GgufError, match="cannot read"):
        read_gguf(tmp_path / "nope.gguf")
    assert is_gguf(tmp_path / "nope.gguf") is False


# ---------------------------------------------------------------------------
# read_meta mapping
# ---------------------------------------------------------------------------


def test_meta_basic_mapping(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("general.type", STRING, "model"),
        ("general.file_type", UINT32, 15),
        ("general.parameter_count", UINT64, 7_000_000_000),
        ("llama.block_count", UINT32, 32),
        ("llama.embedding_length", UINT32, 4096),
        ("llama.attention.head_count", UINT32, 32),
        ("llama.attention.head_count_kv", UINT32, 8),
        ("llama.context_length", UINT32, 32768),
        ("llama.rope.freq_base", FLOAT32, 500000.0),
        ("llama.expert_count", UINT32, 8),
        ("llama.expert_used_count", UINT32, 2),
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.tokens", ARRAY, Arr(STRING, [f"t{i}" for i in range(1000)])),
        ("tokenizer.chat_template", STRING, "{% for m in messages %}{{ m }}{% endfor %}"),
    ]
    meta = read_meta(write_gguf(tmp_path / "m.gguf", kv, [("blk.0.w", (4096, 4096), 12)]))
    assert meta.architecture == "llama"
    assert (meta.n_layer, meta.n_embd, meta.n_head, meta.n_head_kv) == (32, 4096, 32, 8)
    assert meta.n_ctx_train == 32768
    assert meta.n_vocab == 1000
    assert meta.n_embd_head_k == 128  # 4096 / 32
    assert meta.n_embd_head_v == 128
    assert meta.rope_freq_base == pytest.approx(500000.0)
    assert (meta.n_expert, meta.n_expert_used) == (8, 2)
    assert meta.file_type == 15
    assert meta.quant_label == "Q4_K_M"
    assert meta.param_count == 7_000_000_000
    assert meta.tokenizer_model == "gpt2"
    assert meta.chat_template is not None
    assert meta.tensor_bytes == expected_bytes((4096, 4096), 12)
    assert meta.is_mmproj is False
    assert meta.is_adapter is False
    assert meta.has_vision_tensors is False
    assert meta.extra["gguf_version"] == 3


def test_gqa_defaults_to_mha_when_head_count_kv_absent(tmp_path: Path) -> None:
    kv = llm_kv("llama", block_count=8, embedding_length=512, **{"attention.head_count": 16})
    meta = read_meta(write_gguf(tmp_path / "mha.gguf", kv))
    assert meta.n_head == 16
    # Absent head_count_kv means MHA; a 0 here would zero out the KV estimate.
    assert meta.n_head_kv == 16


def test_key_length_and_value_length_override_head_dim(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "gemma3"),
        ("gemma3.block_count", UINT32, 62),
        ("gemma3.embedding_length", UINT32, 5376),
        ("gemma3.attention.head_count", UINT32, 32),
        ("gemma3.attention.head_count_kv", UINT32, 16),
        ("gemma3.attention.key_length", UINT32, 256),
        ("gemma3.attention.value_length", UINT32, 128),
    ]
    meta = read_meta(write_gguf(tmp_path / "g.gguf", kv))
    assert meta.n_embd_head_k == 256  # not 5376/32 == 168
    assert meta.n_embd_head_v == 128
    assert meta.head_dim_k == 256
    assert meta.head_dim_v == 128


def test_value_length_defaults_to_key_length(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "gemma3"),
        ("gemma3.embedding_length", UINT32, 5376),
        ("gemma3.attention.head_count", UINT32, 32),
        ("gemma3.attention.key_length", UINT32, 256),
    ]
    meta = read_meta(write_gguf(tmp_path / "g.gguf", kv))
    assert meta.n_embd_head_k == 256
    assert meta.n_embd_head_v == 256


def test_per_layer_head_count_kv_uses_max(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "gemma3n"),
        ("gemma3n.block_count", UINT32, 6),
        ("gemma3n.embedding_length", UINT32, 512),
        ("gemma3n.attention.head_count", UINT32, 8),
        ("gemma3n.attention.head_count_kv", ARRAY, Arr(UINT32, [1, 2, 4, 4, 2, 1])),
    ]
    meta = read_meta(write_gguf(tmp_path / "pl.gguf", kv))
    assert meta.n_head_kv == 4  # max, conservative for KV sizing
    assert meta.extra["head_count_kv_per_layer"] is True
    assert meta.extra["head_count_kv_len"] == 6


def test_per_layer_head_counts_are_kept_even_without_a_sliding_window(
    tmp_path: Path,
) -> None:
    """The values, not just the max: a zero entry means that layer has no cache.

    They used to be recorded only under the iSWA branch, so a Gemma-3n / LFM2 /
    Nemotron-H layer stack collapsed to its maximum and every layer was charged
    the most expensive one's KV.
    """
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "gemma3n"),
        ("gemma3n.block_count", UINT32, 6),
        ("gemma3n.embedding_length", UINT32, 512),
        ("gemma3n.attention.head_count", UINT32, 8),
        ("gemma3n.attention.head_count_kv", ARRAY, Arr(UINT32, [4, 4, 0, 4, 0, 4])),
    ]
    meta = read_meta(write_gguf(tmp_path / "pl0.gguf", kv))
    assert meta.extra["head_count_kv_values"] == [4, 4, 0, 4, 0, 4]
    assert meta.n_head_kv == 4


def test_hybrid_attention_and_ssm_keys_are_captured(tmp_path: Path) -> None:
    """Qwen3.5/3.6/3.8: without these the planner charges 4x the real KV cache.

    ``full_attention_interval`` says only every 4th layer caches anything; the
    ``ssm.*`` keys size the fixed per-sequence state the other layers hold; and
    ``nextn_predict_layers`` marks a trailing MTP head that holds neither.
    """
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "qwen35"),
        ("qwen35.block_count", UINT32, 65),
        ("qwen35.embedding_length", UINT32, 5120),
        ("qwen35.attention.head_count", UINT32, 24),
        ("qwen35.attention.head_count_kv", UINT32, 4),
        ("qwen35.attention.key_length", UINT32, 256),
        ("qwen35.attention.value_length", UINT32, 256),
        ("qwen35.full_attention_interval", UINT32, 4),
        ("qwen35.ssm.conv_kernel", UINT32, 4),
        ("qwen35.ssm.inner_size", UINT32, 6144),
        ("qwen35.ssm.state_size", UINT32, 128),
        ("qwen35.ssm.group_count", UINT32, 16),
        ("qwen35.ssm.time_step_rank", UINT32, 48),
        ("qwen35.nextn_predict_layers", UINT32, 1),
    ]
    meta = read_meta(write_gguf(tmp_path / "hybrid.gguf", kv))
    assert meta.extra["full_attention_interval"] == 4
    assert meta.extra["ssm_conv_kernel"] == 4
    assert meta.extra["ssm_inner_size"] == 6144
    assert meta.extra["ssm_state_size"] == 128
    assert meta.extra["ssm_group_count"] == 16
    assert meta.extra["ssm_time_step_rank"] == 48
    assert meta.extra["nextn_predict_layers"] == 1


def test_an_ordinary_model_gains_no_hybrid_keys(tmp_path: Path) -> None:
    """Absent keys stay absent: the planner reads their presence as the signal."""
    meta = read_meta(write_gguf(tmp_path / "plain.gguf", llm_kv()))
    for key in (
        "full_attention_interval",
        "ssm_conv_kernel",
        "ssm_inner_size",
        "nextn_predict_layers",
        "head_count_kv_values",
    ):
        assert key not in meta.extra


def test_vocab_size_fallback(tmp_path: Path) -> None:
    kv = [*llm_kv(), ("llama.vocab_size", UINT32, 32000)]
    assert read_meta(write_gguf(tmp_path / "v.gguf", kv)).n_vocab == 32000


def test_unknown_architecture_is_tolerated(tmp_path: Path) -> None:
    meta = read_meta(write_gguf(tmp_path / "x.gguf", [("general.name", STRING, "mystery")]))
    assert meta.architecture == "unknown"
    assert meta.n_layer == 0
    assert meta.quant_label == "unknown"


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------


def test_mmproj_detection_from_clip_keys(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "clip"),
        ("general.file_type", UINT32, 1),
        ("clip.has_vision_encoder", BOOL, True),
        ("clip.vision.image_size", UINT32, 896),
        ("clip.vision.patch_size", UINT32, 14),
        ("clip.vision.block_count", UINT32, 27),
    ]
    tensors: list[TensorSpec] = [
        ("mm.input_projection.weight", (1152, 2560), 1),
        ("v.blk.0.attn_q.weight", (1152, 1152), 1),
    ]
    meta = read_meta(write_gguf(tmp_path / "mmproj-test.gguf", kv, tensors))
    assert meta.is_mmproj is True
    assert meta.has_vision_tensors is True
    assert meta.vision_image_size == 896
    assert meta.vision_patch_size == 14
    assert meta.vision_n_patch == (896 // 14) ** 2 == 4096
    assert meta.quant_label == "F16"


def test_mmproj_detection_from_filename_only(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "mmproj-F32.gguf", [("general.name", STRING, "proj")])
    assert looks_like_mmproj(path) is True
    assert read_meta(path).is_mmproj is True


def test_single_file_multimodal_is_not_a_projector(tmp_path: Path) -> None:
    kv = llm_kv("qwen2vl")
    tensors: list[TensorSpec] = [
        ("token_embd.weight", (128, 1000), 0),
        ("blk.0.attn_q.weight", (128, 128), 0),
        ("v.blk.0.attn_q.weight", (128, 128), 0),
    ]
    meta = read_meta(write_gguf(tmp_path / "vlm.gguf", kv, tensors))
    assert meta.has_vision_tensors is True
    assert meta.is_mmproj is False  # it is a loadable model, not a projector
    assert meta.extra["single_file_multimodal"] is True


def test_adapter_detection(tmp_path: Path) -> None:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, "llama"),
        ("general.type", STRING, "adapter"),
        ("adapter.type", STRING, "lora"),
        ("adapter.lora.alpha", FLOAT32, 32.0),
        ("llama.block_count", UINT32, 32),
    ]
    tensors: list[TensorSpec] = [
        ("blk.0.attn_q.weight.lora_a", (4096, 16), 0),
        ("blk.0.attn_q.weight.lora_b", (16, 4096), 0),
    ]
    meta = read_meta(write_gguf(tmp_path / "lora.gguf", kv, tensors))
    assert meta.is_adapter is True
    assert meta.extra["adapter_alpha"] == pytest.approx(32.0)
    assert meta.extra["adapter_type"] == "lora"
    assert meta.extra["adapter_rank"] == 16


def test_adapter_detection_from_tensor_names_only(tmp_path: Path) -> None:
    tensors: list[TensorSpec] = [("blk.5.ffn_up.weight.lora_a", (2048, 8), 0)]
    meta = read_meta(write_gguf(tmp_path / "l2.gguf", llm_kv(), tensors))
    assert meta.is_adapter is True
    assert meta.extra["adapter_rank"] == 8


# ---------------------------------------------------------------------------
# Sharding
# ---------------------------------------------------------------------------


def _write_shard(tmp_path: Path, index: int, total: int, tensors: list[TensorSpec]) -> Path:
    name = f"big-model-{index:05d}-of-{total:05d}.gguf"
    kv = llm_kv() if index == 1 else [("split.no", UINT32, index - 1)]
    return write_gguf(tmp_path / name, kv, tensors)


def test_shard_paths_for_non_sharded(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "solo.gguf", llm_kv())
    assert shard_paths_for(path) == [path]
    assert missing_shard_names(path) == []


def test_shard_discovery_and_total_bytes(tmp_path: Path) -> None:
    first = _write_shard(tmp_path, 1, 2, [("blk.0.w", (256, 16), 12)])
    second = _write_shard(tmp_path, 2, 2, [("blk.1.w", (256, 32), 12)])
    shards = shard_paths_for(first)
    assert shards == [first, second]
    assert shard_paths_for(second) == [first, second]

    want = expected_bytes((256, 16), 12) + expected_bytes((256, 32), 12)
    meta = read_meta(first, shard_paths=shards)
    assert meta.tensor_bytes == want
    assert meta.extra["shard_count"] == 2
    assert "missing_shards" not in meta.extra

    # Without shard_paths only the given file is counted -- the caller opts in.
    assert read_meta(first).tensor_bytes == expected_bytes((256, 16), 12)


def test_missing_intermediate_shard_is_reported_not_fatal(tmp_path: Path) -> None:
    first = _write_shard(tmp_path, 1, 3, [("blk.0.w", (256, 16), 12)])
    third = _write_shard(tmp_path, 3, 3, [("blk.2.w", (256, 16), 12)])
    assert shard_paths_for(first) == [first, third]
    assert missing_shard_names(first) == ["big-model-00002-of-00003.gguf"]

    meta = read_meta(first, shard_paths=[first, tmp_path / "big-model-00002-of-00003.gguf", third])
    assert meta.tensor_bytes == 2 * expected_bytes((256, 16), 12)
    assert meta.extra["missing_shards"] == ["big-model-00002-of-00003.gguf"]
    assert meta.extra["shard_count"] == 2

    # Even without explicit shard_paths the gap is surfaced.
    assert read_meta(first).extra["missing_shards"] == ["big-model-00002-of-00003.gguf"]


def test_shard_paths_for_missing_siblings_never_returns_empty(tmp_path: Path) -> None:
    lonely = tmp_path / "ghost-00002-of-00004.gguf"
    assert shard_paths_for(lonely) == [lonely]


# ---------------------------------------------------------------------------
# Quant labelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_type", "want"),
    [
        (0, "F32"),
        (1, "F16"),
        (7, "Q8_0"),
        (15, "Q4_K_M"),
        (17, "Q5_K_M"),
        (18, "Q6_K"),
        (27, "IQ3_M"),
        (30, "IQ4_XS"),
        (32, "BF16"),
        (38, "MXFP4"),
    ],
)
def test_quant_label_from_file_type(file_type: int, want: str) -> None:
    assert quant_label_from({}, [], file_type) == want


def test_quant_label_from_tensors_when_file_type_unknown() -> None:
    tensors = [
        TensorInfo(name="blk.0.w", dims=(4096, 4096), ggml_type=12, offset=0, n_bytes=9_000_000),
        TensorInfo(name="blk.1.w", dims=(4096, 4096), ggml_type=14, offset=0, n_bytes=1_000_000),
        TensorInfo(name="norm", dims=(4096,), ggml_type=0, offset=0, n_bytes=16384),
    ]
    # ftype 999 is not in the table -> dominant 2-D weight type wins.
    assert quant_label_from({}, tensors, 999) == "Q4_K"
    assert quant_label_from({}, tensors, None) == "Q4_K"


# ===========================================================================
# Auxiliary GGUFs: files in a repo that are not models
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "size", "want"),
    [
        # unsloth ships the draft modules in their own directory.
        ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 2_297_478_020, True),
        ("MTP/mtp-Qwen3.8-27B-Q8_0.gguf", None, True),
        # ...and the same module flat, leading with the mtp token.
        ("mtp-Qwen3.8-27B-Q4_0.gguf", 2_297_478_020, True),
        ("mtp_Qwen3.8-27B-Q4_0.gguf", None, True),
        # A calibration matrix is never loadable, wherever it sits.
        ("imatrix_unsloth.gguf", 944_892_805, True),
        ("Qwen3.8-27B-imatrix.gguf", None, True),
        # THE FALSE-POSITIVE GUARD: a full model with an MTP head is published
        # with -MTP- in the middle of its name, at model size. Dropping it
        # would hide a real 20 GiB model from the picker.
        ("Qwen3.8-27B-NVFP4-MTP-Q6_K.gguf", 21_000_000_000, False),
        ("Qwen3.8-27B-NVFP4-MTP-Q6_K.gguf", None, False),
        ("Gemma4-31B-QAT-Uncensored-Balanced-MTP-Q4_K_M.gguf", 18_000_000_000, False),
        # A middle mtp token IS auxiliary once a size says it cannot be a model.
        ("Qwen3.8-27B-mtp-draft.gguf", 300_000_000, True),
        # ...but with no size at all, keep it: a stray row is cosmetic, a
        # missing model is not.
        ("Qwen3.8-27B-mtp-draft.gguf", None, False),
        # "mtp" inside a longer word is not a token and must never match.
        ("Qwen3.8-27B-mtpx-Q4_K_M.gguf", 100_000, False),
        ("Promtpheus-7B-Q4_K_M.gguf", 100_000, False),
        # Ordinary quants and projectors are untouched.
        ("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", 14_250_000_000, False),
        ("mmproj-F32.gguf", 1_000_000, False),
    ],
)
def test_looks_like_auxiliary_gguf(name: str, size: int | None, want: bool) -> None:
    assert looks_like_auxiliary_gguf(name, size_bytes=size) is want


def test_auxiliary_matching_is_case_and_separator_insensitive() -> None:
    assert looks_like_auxiliary_gguf("mtp/MTP-Model-Q4_0.gguf")
    assert looks_like_auxiliary_gguf("IMATRIX_unsloth.GGUF")
    assert looks_like_auxiliary_gguf(Path("MTP") / "mtp-x-Q4_0.gguf")


def test_a_windows_style_mtp_path_is_still_auxiliary() -> None:
    """HF hands back POSIX paths, but a Path on Windows renders backslashes."""
    assert looks_like_auxiliary_gguf(r"MTP\mtp-Qwen3.8-27B-Q4_0.gguf")


def test_quant_label_reads_file_type_from_kv() -> None:
    assert quant_label_from({"general.file_type": 18}, [], None) == "Q6_K"


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("model-Q4_K_M.gguf", "Q4_K_M"),
        ("thing.i1-IQ3_XXS.gguf", "IQ3_XXS"),
        ("gemma-4-31B-heretic-NVFP4.gguf", "NVFP4"),
        ("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "Q4_K_XL"),
        # unsloth's dynamic mixes: the UD- prefix is deliberately not part of
        # the label (the download picker matches on the bare quant), but the
        # label itself must resolve instead of falling through to "unknown".
        ("Qwen3.8-27B-UD-Q2_K_XL.gguf", "Q2_K_XL"),
        ("Qwen3.8-27B-UD-Q6_K_M.gguf", "Q6_K_M"),
        ("Qwen3.8-27B-UD-Q8_K_L.gguf", "Q8_K_L"),
        ("24_10_Mistrial_Celeste-12B-V1.6.Q8_0NSFW.gguf", "Q8_0"),
        ("mmproj-model-BF16.gguf", "BF16"),
        ("Behemoth-123B-IQ3_M-00001-of-00002.gguf", "IQ3_M"),
        ("qwen3-embedding-8b-q4_k_m.gguf", "Q4_K_M"),
        ("Qwen2.5-0.5B-Instruct.gguf", None),
        ("Qwen3-VL-Embedding-8B.gguf", None),
    ],
)
def test_quant_label_from_filename(name: str, want: str | None) -> None:
    assert quant_label_from_filename(Path("/models") / name) == want


def test_quant_label_falls_back_to_filename(tmp_path: Path) -> None:
    # No file_type, and only 1-D F32 tensors, so neither of the first two
    # sources can decide -- the filename is all that is left.
    path = write_gguf(tmp_path / "mystery-IQ2_M.gguf", llm_kv(), [("norm", (128,), 0)])
    assert read_meta(path).quant_label == "IQ2_M"


def test_filename_refines_coarse_file_type(tmp_path: Path) -> None:
    """unsloth's UD-* mixes declare a coarse ftype but name the real recipe."""
    kv = [*llm_kv(), ("general.file_type", UINT32, 2)]  # ftype 2 == Q4_0
    path = write_gguf(tmp_path / "gemma-UD-Q4_K_XL.gguf", kv, [("blk.0.w", (256, 8), 12)])
    assert read_meta(path).quant_label == "Q4_K_XL"


def test_vaguer_filename_does_not_override_file_type(tmp_path: Path) -> None:
    kv = [*llm_kv(), ("general.file_type", UINT32, 15)]  # Q4_K_M
    path = write_gguf(tmp_path / "model-Q4_0.gguf", kv, [("blk.0.w", (256, 8), 12)])
    assert read_meta(path).quant_label == "Q4_K_M"


def test_quant_label_unknown_when_undeterminable(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "plain.gguf", llm_kv(), [("norm", (128,), 0)])
    assert read_meta(path).quant_label == "unknown"


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def test_is_gguf(tmp_path: Path) -> None:
    good = write_gguf(tmp_path / "g.gguf", llm_kv())
    assert is_gguf(good) is True
    bad = tmp_path / "b.gguf"
    bad.write_bytes(b"GGUX1234")
    assert is_gguf(bad) is False
    assert is_gguf(tmp_path) is False  # a directory is not a GGUF


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("mmproj-F32.gguf", True),
        ("model.mmproj-f16.gguf", True),
        ("Model-mmproj-BF16.gguf", True),
        ("gemma-4-31B-it-QAT-Q4_0.gguf", False),
    ],
)
def test_looks_like_mmproj(name: str, want: bool) -> None:
    assert looks_like_mmproj(Path("/models") / name) is want


def test_known_quant_labels_cover_ftype_table() -> None:
    from studioforge.core.gguf import LLAMA_FTYPE_LABELS

    unknown = set(LLAMA_FTYPE_LABELS.values()) - KNOWN_QUANT_LABELS
    assert unknown == set()


# ===========================================================================
# Real library
# ===========================================================================


def _library_root() -> Path | None:
    """The GGUF library to parse for real, or ``None`` to skip.

    ``SF_TEST_MODELS_DIR`` first, else whatever the app itself would detect
    (LM Studio's ``downloadsFolder`` and friends). Never a hard-coded absolute
    path: that is right on exactly one machine, and wrong everywhere else.
    """
    env = os.environ.get("SF_TEST_MODELS_DIR", "").strip()
    if env:
        return Path(env)
    from studioforge.config import detect_model_dir

    return detect_model_dir()


MODELS_ROOT = _library_root() or Path("<no-model-library-detected>")
TINY = (
    MODELS_ROOT
    / "lmstudio-community"
    / "Qwen2.5-0.5B-Instruct-GGUF"
    / ("Qwen2.5-0.5B-Instruct-Q8_0.gguf")
)
VISION_DIR = MODELS_ROOT / "lmstudio-community" / "gemma-4-31B-it-QAT-GGUF"
VISION_MAIN = VISION_DIR / "gemma-4-31B-it-QAT-Q4_0.gguf"
VISION_MMPROJ = VISION_DIR / "mmproj-gemma-4-31B-it-QAT-BF16.gguf"
EMBEDDING = (
    MODELS_ROOT
    / "endyjasmi"
    / "Qwen3-Embedding-8B-Q4_K_M-GGUF"
    / ("qwen3-embedding-8b-q4_k_m.gguf")
)
SHARDED_DIR = MODELS_ROOT / "bartowski" / "TheDrummer_Behemoth-X-123B-v2.1-GGUF"
SHARDED_FIRST = SHARDED_DIR / "TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf"

needs_library = pytest.mark.skipif(
    not MODELS_ROOT.is_dir(), reason=f"model library {MODELS_ROOT} not present"
)


def _first_shard_only(path: Path) -> bool:
    """Non-first shards carry no metadata, so they cannot be asserted on."""
    shards = shard_paths_for(path)
    return len(shards) == 1 or shards[0] == path


@needs_library
def test_real_library_inventory() -> None:
    files = sorted(MODELS_ROOT.rglob("*.gguf"))
    assert files, "library directory contains no GGUF files"

    rows: list[str] = []
    header = (
        f"{'file':<52} {'arch':<10} {'quant':<8} {'ft':>4} {'L':>4} {'H':>4} "
        f"{'KVH':>4} {'ctx':>8} {'vocab':>7} {'GiB':>7} {'ms':>6}  flags"
    )
    problems: list[str] = []
    for path in files:
        started = time.perf_counter()
        meta = read_meta(path, shard_paths=shard_paths_for(path))
        elapsed_ms = (time.perf_counter() - started) * 1000

        flags = [
            name
            for name, on in (
                ("mmproj", meta.is_mmproj),
                ("vision", meta.has_vision_tensors),
                ("adapter", meta.is_adapter),
                ("moe", bool(meta.n_expert)),
                ("tools", meta.supports_tools),
            )
            if on
        ]
        if meta.extra.get("unknown_ggml_types"):
            flags.append(f"unknown_types={meta.extra['unknown_ggml_types']}")
        if meta.extra.get("missing_shards"):
            flags.append("MISSING_SHARDS")
        rows.append(
            f"{path.name[:52]:<52} {meta.architecture[:10]:<10} {meta.quant_label:<8} "
            f"{'-' if meta.file_type is None else meta.file_type:>4} {meta.n_layer:>4} "
            f"{meta.n_head:>4} {meta.n_head_kv:>4} {meta.n_ctx_train:>8} {meta.n_vocab:>7} "
            f"{meta.tensor_bytes / 2**30:>7.2f} {elapsed_ms:>6.1f}  {','.join(flags)}"
        )

        # Every file must yield a label a human would recognise.
        if meta.quant_label not in KNOWN_QUANT_LABELS:
            problems.append(f"{path.name}: implausible quant label {meta.quant_label!r}")
        if looks_like_mmproj(path) and not meta.is_mmproj:
            problems.append(f"{path.name}: mmproj file not detected as one")
        if meta.is_mmproj or meta.is_adapter or not _first_shard_only(path):
            continue
        # A real text model must expose the two numbers the KV planner needs.
        if meta.n_layer <= 0:
            problems.append(f"{path.name}: n_layer == 0")
        if meta.n_head_kv <= 0:
            problems.append(f"{path.name}: n_head_kv == 0")
        if meta.n_ctx_train <= 0:
            problems.append(f"{path.name}: n_ctx_train == 0")

    print(f"\n{header}\n{'-' * len(header)}")
    print("\n".join(rows))
    print(f"\n{len(files)} GGUF files parsed from {MODELS_ROOT}")
    assert problems == [], "\n".join(problems)


@needs_library
def test_real_library_tensor_bytes_are_consistent_with_file_size() -> None:
    """tensor_bytes must be within the file, and account for nearly all of it.

    This is the cheapest end-to-end check that the ggml block-size table is
    right: if a size were wrong, the sum would drift far from the real file.
    """
    checked = 0
    candidates = 0
    for path in sorted(MODELS_ROOT.rglob("*.gguf")):
        if not _first_shard_only(path) or len(shard_paths_for(path)) > 1:
            continue
        candidates += 1
        gguf = read_gguf(path)
        size = path.stat().st_size
        assert gguf.data_offset < size
        payload = size - gguf.data_offset
        assert gguf.total_tensor_bytes <= payload
        assert gguf.total_tensor_bytes > payload * 0.98, path.name
        checked += 1
    # `> 10` encoded the size of one developer's library: on a box with seven
    # single-shard models every per-file invariant above passed and the test
    # still failed. The real invariant is that every eligible file was
    # checked, and that there was something to check at all.
    assert checked == candidates
    assert checked > 0, f"no single-shard GGUF files under {MODELS_ROOT}"


@needs_library
@pytest.mark.skipif(not TINY.is_file(), reason="tiny fixture model missing")
def test_real_tiny_model() -> None:
    gguf = read_gguf(TINY)
    meta = read_meta(TINY)
    assert meta.architecture == "qwen2"
    assert meta.n_layer == 24
    assert meta.n_head == 14
    assert meta.n_head_kv == 2  # GQA
    assert meta.n_embd == 896
    assert meta.n_embd_head_k == 64
    assert meta.n_vocab == 151936
    assert meta.quant_label == "Q8_0"
    assert meta.file_type == 7
    assert meta.tokenizer_model == "gpt2"
    assert meta.chat_template
    assert meta.is_mmproj is False
    assert meta.is_adapter is False
    # The 151936-token vocabulary must never be materialised as a list.
    tokens = gguf.kv["tokenizer.ggml.tokens"]
    assert isinstance(tokens, dict)
    assert tokens["len"] == 151936
    assert len(tokens["sample"]) == 64


@needs_library
@pytest.mark.skipif(not VISION_MMPROJ.is_file(), reason="vision fixture pair missing")
def test_real_vision_pair() -> None:
    main = read_meta(VISION_MAIN)
    assert main.is_mmproj is False
    assert main.architecture == "gemma4"
    assert main.n_head_kv == 16
    assert main.n_embd_head_k == 512  # explicit attention.key_length
    assert main.extra.get("head_count_kv_per_layer") is True

    proj = read_meta(VISION_MMPROJ)
    assert proj.is_mmproj is True
    assert proj.has_vision_tensors is True
    assert proj.vision_image_size is not None
    assert proj.vision_patch_size is not None
    assert proj.vision_n_patch == (proj.vision_image_size // proj.vision_patch_size) ** 2
    assert proj.quant_label == "BF16"
    assert proj.tensor_bytes > 0


@needs_library
@pytest.mark.skipif(not EMBEDDING.is_file(), reason="embedding fixture missing")
def test_real_embedding_model() -> None:
    meta = read_meta(EMBEDDING)
    assert meta.architecture == "qwen3"
    assert meta.quant_label == "Q4_K_M"
    assert meta.n_layer > 0
    assert meta.n_head_kv > 0
    assert meta.is_mmproj is False


@needs_library
@pytest.mark.skipif(not SHARDED_FIRST.is_file(), reason="sharded fixture missing")
def test_real_sharded_model_sums_all_shards() -> None:
    shards = shard_paths_for(SHARDED_FIRST)
    assert len(shards) == 2
    assert [p.name for p in shards] == sorted(p.name for p in shards)

    single = read_meta(SHARDED_FIRST)
    both = read_meta(SHARDED_FIRST, shard_paths=shards)
    assert both.extra["shard_count"] == 2
    assert "missing_shards" not in both.extra
    assert both.tensor_bytes > single.tensor_bytes

    # The weight total must be close to the combined on-disk size of both
    # shards; counting only shard 1 would understate it by ~15 GB.
    on_disk = sum(p.stat().st_size for p in shards)
    assert both.tensor_bytes <= on_disk
    assert both.tensor_bytes > on_disk * 0.98
    assert both.quant_label == "IQ3_M"
    assert both.n_layer == 88

    # Shard 2 alone carries no model metadata but still sums correctly.
    from_second = read_meta(shards[1], shard_paths=shards)
    assert from_second.tensor_bytes == both.tensor_bytes
    assert from_second.n_layer == 0


@needs_library
def test_real_large_file_metadata_read_is_fast() -> None:
    """Header-only reads must not scale with file size.

    A reader that materialised the tokenizer arrays (or, worse, touched tensor
    data) would blow this budget immediately; the whole 39-model library is
    rescanned on demand, so per-file cost is what makes that viable.
    """
    candidates = sorted(
        (p for p in MODELS_ROOT.rglob("*.gguf") if p.stat().st_size > 20 * 2**30),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    if not candidates:
        pytest.skip("no file larger than 20 GiB in the library")
    biggest = candidates[0]
    size_gib = biggest.stat().st_size / 2**30

    read_meta(biggest)  # warm the OS cache for the header pages
    started = time.perf_counter()
    meta = read_meta(biggest)
    elapsed = time.perf_counter() - started
    print(f"\nread_meta({biggest.name}, {size_gib:.1f} GiB) took {elapsed * 1000:.0f} ms")
    assert meta.n_layer > 0
    assert elapsed < 1.0
