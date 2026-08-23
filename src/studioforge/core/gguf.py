"""Pure-Python GGUF header/metadata reader.

Why hand-rolled instead of the ``gguf`` pip package: that package pulls in
numpy and, worse, memory-maps the whole file to expose tensor data. This
module only ever touches the header, the metadata key/value block and the
tensor-info table -- never a byte of tensor data -- so a 100 GB model is read
in a few milliseconds' worth of bounded, sequential I/O plus seeks. Scanning a
library of 40 models at startup has to be effectively free, and it has to work
with zero binary dependencies.

Binary layout (GGUF v2/v3, always little-endian)::

    char[4]  magic  == "GGUF"
    uint32   version                       (2 or 3; v1 used uint32 counts)
    uint64   tensor_count
    uint64   metadata_kv_count
    kv[metadata_kv_count]:
        uint64 key_len, char[key_len] key  (utf-8, not NUL terminated)
        uint32 value_type                  (see GGUFValueType below)
        value                              (type-dependent, see _read_value)
    tensor_info[tensor_count]:
        uint64 name_len, char[name_len] name
        uint32 n_dims
        uint64 dims[n_dims]                (row-major reversed vs. torch)
        uint32 ggml_type
        uint64 offset                      (relative to the data section start)
    padding to general.alignment (default 32)
    tensor data                            <-- never read here

The one performance trap is ``tokenizer.ggml.tokens`` / ``.merges`` /
``.scores`` / ``.token_type``: those arrays hold 100k-500k entries. Building
Python lists for them costs hundreds of milliseconds and tens of megabytes per
model, so long arrays are skipped by seeking past the payload while still
recording their element count (which is where ``n_vocab`` comes from).
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Final

from studioforge.types import GgufMeta

__all__ = [
    "GGUF_MAGIC",
    "META_FORMAT_VERSION",
    "GgufError",
    "GgufFile",
    "TensorInfo",
    "is_gguf",
    "looks_like_mmproj",
    "meta_from_gguf",
    "quant_label_from",
    "read_gguf",
    "read_meta",
    "shard_paths_for",
]

GGUF_MAGIC: Final = b"GGUF"
GGUF_MAGIC_SWAPPED: Final = b"FUGG"  # big-endian writers emit the magic reversed

#: Version of the *parsed* shape :func:`read_meta` produces. The registry caches
#: parsed metadata in SQLite keyed on (path, mtime, size), which is exactly
#: right for detecting a changed file and useless for detecting a changed
#: parser: a model registered before this module learned to read
#: ``full_attention_interval`` would keep serving metadata without it forever,
#: and the planner would keep charging Qwen3.5 four times its real KV with no
#: sign anything was wrong. Bump this whenever ``read_meta`` starts extracting
#: something the planner reads, and every already-registered model re-parses on
#: the next ordinary scan (a few ms each -- only the header is touched).
#:
#: 1 -> 2: full_attention_interval, ssm.*, nextn_predict_layers, and
#:         head_count_kv_values for non-iSWA per-layer arrays.
META_FORMAT_VERSION: Final = 2

DEFAULT_ALIGNMENT: Final = 32

#: Read buffer for the underlying file object. The metadata block of a big
#: model is a few MB (mostly the tokenizer), so a 4 MiB buffer usually means
#: one or two syscalls for the whole parse.
_READ_CHUNK: Final = 4 << 20

# Sanity limits. A garbage/truncated file happily decodes as "3.4e18 tensors";
# refusing absurd counts up front turns that into a clean GgufError instead of
# a MemoryError or a multi-hour loop.
_MAX_TENSORS: Final = 1 << 22  # 4M tensors
_MAX_KV: Final = 1 << 20
_MAX_STRING_BYTES: Final = 512 << 20
_MAX_DIMS: Final = 8

#: Numeric arrays at most this long are materialised in full even when they
#: exceed ``max_array_len``. Some architectures store *per-layer* values (e.g.
#: ``attention.head_count_kv`` on Gemma-3n / Jamba); truncating those to a
#: sample would silently mis-read the model, and a few thousand ints is cheap.
_FULL_NUMERIC_ARRAY_LEN: Final = 4096

#: Keys whose payload is always skipped wholesale when long: they are pure bulk
#: and nothing in StudioForge needs their contents, only their length.
_BULK_ARRAY_KEYS: Final = frozenset(
    {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    }
)


class GgufError(Exception):
    """The file is not a readable/parsable GGUF file."""


# ---------------------------------------------------------------------------
# GGUF value types
# ---------------------------------------------------------------------------

UINT8: Final = 0
INT8: Final = 1
UINT16: Final = 2
INT16: Final = 3
UINT32: Final = 4
INT32: Final = 5
FLOAT32: Final = 6
BOOL: Final = 7
STRING: Final = 8
ARRAY: Final = 9
UINT64: Final = 10
INT64: Final = 11
FLOAT64: Final = 12

#: Byte width of every fixed-size (non-string, non-array) value type.
_SCALAR_WIDTH: Final[dict[int, int]] = {
    UINT8: 1,
    INT8: 1,
    UINT16: 2,
    INT16: 2,
    UINT32: 4,
    INT32: 4,
    FLOAT32: 4,
    BOOL: 1,
    UINT64: 8,
    INT64: 8,
    FLOAT64: 8,
}

_NUMERIC_TYPES: Final = frozenset(_SCALAR_WIDTH) - {BOOL}


# ---------------------------------------------------------------------------
# GGML tensor types
# ---------------------------------------------------------------------------

#: ``ggml_type -> (block_size_in_elements, bytes_per_block)``.
#:
#: Quantised tensors are stored as blocks: ``block_size`` consecutive elements
#: share a scale (and possibly a min), so the on-disk size of a tensor is
#: ``prod(dims) / block_size * type_size``. These constants come from
#: ggml.c's ``type_traits`` table and are the reason a Q4_K_M model is ~4.5
#: bits/weight rather than a round 4.
GGML_TYPE_SIZES: Final[dict[int, tuple[int, int]]] = {
    0: (1, 4),  # F32
    1: (1, 2),  # F16
    2: (32, 18),  # Q4_0
    3: (32, 20),  # Q4_1
    6: (32, 22),  # Q5_0
    7: (32, 24),  # Q5_1
    8: (32, 34),  # Q8_0
    9: (32, 36),  # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),  # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),  # I8
    25: (1, 2),  # I16
    26: (1, 4),  # I32
    27: (1, 8),  # I64
    28: (1, 8),  # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),  # BF16
    34: (256, 54),  # TQ1_0
    35: (256, 66),  # TQ2_0
    39: (32, 17),  # MXFP4
}

GGML_TYPE_NAMES: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
}

#: Conservative stand-in for a ggml type this build has never heard of. New
#: quant types land in llama.cpp every few weeks; crashing the whole library
#: scan over one unknown id would be unacceptable, so unknown tensors are
#: costed at 4.5 bits/element (the Q4_K block geometry) and the id is reported
#: in ``GgufMeta.extra["unknown_ggml_types"]`` so it can be added here.
UNKNOWN_TYPE_SIZE: Final[tuple[int, int]] = (256, 144)


# ---------------------------------------------------------------------------
# llama.cpp file-type (quantisation preset) table
# ---------------------------------------------------------------------------

#: ``general.file_type`` -> label, mirroring ``enum llama_ftype`` in
#: llama.cpp/include/llama.h. Gaps are types that were removed upstream
#: (Q4_2/Q4_3, the Q4_0_N_M repacks); they must stay gaps so the numbering of
#: everything after them lines up.
LLAMA_FTYPE_LABELS: Final[dict[int, str]] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1",  # MOSTLY_Q4_1_SOME_F16 (removed upstream)
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4",
}

#: Quant labels accepted when parsing a filename. Used as a whitelist so that
#: "Qwen3" or "Q8_0NSFW" cannot invent a bogus label.
KNOWN_QUANT_LABELS: Final[frozenset[str]] = frozenset(
    {
        "F32",
        "F16",
        "BF16",
        "NVFP4",
        "MXFP4",
        "TQ1_0",
        "TQ2_0",
        "Q2_K",
        "Q2_K_S",
        "Q2_K_L",
        "Q2_K_XL",
        "Q3_K",
        "Q3_K_S",
        "Q3_K_M",
        "Q3_K_L",
        "Q3_K_XL",
        "Q4_0",
        "Q4_1",
        "Q4_K",
        "Q4_K_S",
        "Q4_K_M",
        "Q4_K_L",
        "Q4_K_XL",
        "Q5_0",
        "Q5_1",
        "Q5_K",
        "Q5_K_S",
        "Q5_K_M",
        "Q5_K_L",
        "Q5_K_XL",
        "Q6_K",
        "Q6_K_L",
        "Q6_K_M",
        "Q6_K_XL",
        "Q8_0",
        "Q8_K",
        "Q8_K_L",
        "Q8_K_XL",
        "IQ1_S",
        "IQ1_M",
        "IQ2_XXS",
        "IQ2_XS",
        "IQ2_S",
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_XS",
        "IQ3_S",
        "IQ3_M",
        "IQ4_NL",
        "IQ4_XS",
    }
)

# Candidate quant token in a filename: an optional I prefix, Q + digit, then
# any number of ``_SUFFIX`` groups drawn from the small suffix alphabet, or one
# of the non-Q formats. Deliberately has no trailing boundary assertion so that
# "...Q8_0NSFW.gguf" still yields the longest valid prefix "Q8_0"; the
# whitelist above rejects nonsense matches.
_QUANT_TOKEN_RE: Final = re.compile(
    r"(?<![A-Z0-9])(?:I?Q\d(?:_(?:0|1|K|S|M|L|XS|XXS|XL|NL))*|F32|F16|BF16|NVFP4|MXFP4|TQ\d_\d)"
)

# "<base>-00001-of-00005.gguf" -- llama.cpp's own split naming.
_SHARD_RE: Final = re.compile(r"^(?P<base>.*)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)

_VISION_TENSOR_PREFIXES: Final = ("v.", "mm.", "model.vision", "vision_model.", "multi_modal")
_LORA_TENSOR_MARKERS: Final = (".lora_a", ".lora_b")


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorInfo:
    """One entry of the tensor-info table (no tensor data attached)."""

    name: str
    dims: tuple[int, ...]
    ggml_type: int
    offset: int
    n_bytes: int

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"TYPE_{self.ggml_type}")

    @property
    def n_elements(self) -> int:
        total = 1
        for dim in self.dims:
            total *= dim
        return total


@dataclass
class GgufFile:
    """Everything this reader extracts from one GGUF file."""

    path: Path
    version: int
    tensor_count: int
    alignment: int
    kv: dict[str, Any]
    tensors: list[TensorInfo]
    data_offset: int
    total_tensor_bytes: int
    # Not part of the requested surface but needed to propagate the "new quant
    # type, sizes guessed" signal up into GgufMeta.extra.
    unknown_ggml_types: tuple[int, ...] = ()

    @property
    def file_size(self) -> int:
        try:
            return self.path.stat().st_size
        except OSError:
            return 0


# ---------------------------------------------------------------------------
# Buffered little-endian reader
# ---------------------------------------------------------------------------


_U16 = struct.Struct("<H")
_I16 = struct.Struct("<h")
_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_U64 = struct.Struct("<Q")
_I64 = struct.Struct("<q")
_F32 = struct.Struct("<f")
_F64 = struct.Struct("<d")


class _Reader:
    """Sliding-window reader over a binary stream.

    The metadata block has no length prefix, so it must be parsed
    incrementally. Parsing straight out of a ``bytes`` window (rather than
    calling ``fh.read(8)`` per field) matters: a 262k-token vocabulary means
    ~500k length prefixes, and the per-call overhead of the file object
    dominates at that count.
    """

    __slots__ = ("_base", "_buf", "_fh", "_pos")

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh
        self._buf: bytes = b""
        self._pos = 0  # cursor inside _buf
        self._base = 0  # file offset of _buf[0]

    @property
    def offset(self) -> int:
        """Absolute file offset of the cursor."""
        return self._base + self._pos

    def _fill(self, need: int) -> None:
        """Guarantee ``need`` bytes are available at the cursor."""
        if self._pos:
            # Drop the consumed prefix so the window does not grow unbounded.
            self._buf = self._buf[self._pos :]
            self._base += self._pos
            self._pos = 0
        while len(self._buf) < need:
            chunk = self._fh.read(max(_READ_CHUNK, need - len(self._buf)))
            if not chunk:
                raise GgufError(
                    f"unexpected end of file at offset {self.offset} (wanted {need} more bytes)"
                )
            self._buf += chunk

    def take(self, n: int) -> bytes:
        if n < 0:
            raise GgufError(f"negative read length {n}")
        end = self._pos + n
        if end > len(self._buf):
            self._fill(n)
            end = n
        data = self._buf[self._pos : end]
        self._pos = end
        return data

    def skip(self, n: int) -> None:
        """Advance the cursor, seeking the underlying file for large jumps."""
        if n < 0:
            raise GgufError(f"negative skip length {n}")
        end = self._pos + n
        if end <= len(self._buf):
            self._pos = end
            return
        target = self.offset + n
        self._fh.seek(target)
        self._buf = b""
        self._pos = 0
        self._base = target

    # --- scalars -------------------------------------------------------
    def u8(self) -> int:
        return self.take(1)[0]

    def i8(self) -> int:
        return int.from_bytes(self.take(1), "little", signed=True)

    def u16(self) -> int:
        return int(_U16.unpack(self.take(2))[0])

    def i16(self) -> int:
        return int(_I16.unpack(self.take(2))[0])

    def u32(self) -> int:
        return int(_U32.unpack(self.take(4))[0])

    def i32(self) -> int:
        return int(_I32.unpack(self.take(4))[0])

    def u64(self) -> int:
        return int(_U64.unpack(self.take(8))[0])

    def i64(self) -> int:
        return int(_I64.unpack(self.take(8))[0])

    def f32(self) -> float:
        return float(_F32.unpack(self.take(4))[0])

    def f64(self) -> float:
        return float(_F64.unpack(self.take(8))[0])

    def boolean(self) -> bool:
        return self.take(1)[0] != 0

    def string(self) -> str:
        length = self.u64()
        if length > _MAX_STRING_BYTES:
            raise GgufError(f"implausible string length {length} at offset {self.offset}")
        # errors="replace": a mis-encoded tokenizer entry must not abort a scan.
        return self.take(length).decode("utf-8", errors="replace")

    def skip_string(self) -> int:
        """Skip one string, returning its byte length."""
        length = self.u64()
        if length > _MAX_STRING_BYTES:
            raise GgufError(f"implausible string length {length} at offset {self.offset}")
        self.skip(length)
        return length

    def skip_strings(self, count: int) -> None:
        """Skip ``count`` length-prefixed strings.

        Hot loop: a 262k-entry vocabulary plus its merge table means half a
        million length prefixes to step over. Everything the inner loop touches
        is a local, and ``unpack_from`` reads the prefix without slicing a
        temporary ``bytes`` -- that is the difference between ~50 ms and ~250 ms
        per model, times a whole library.
        """
        remaining = count
        unpack_from = _U64.unpack_from
        while remaining > 0:
            buf = self._buf
            pos = self._pos
            end = len(buf)
            while remaining > 0:
                nxt = pos + 8
                if nxt > end:
                    break
                length: int = unpack_from(buf, pos)[0]
                nxt += length
                if nxt > end:
                    break
                pos = nxt
                remaining -= 1
            self._pos = pos
            if remaining <= 0:
                return
            # The next entry straddles the window edge: fall back to the
            # buffered path for exactly one string, then resume the fast loop.
            self.skip_string()
            remaining -= 1


_SCALAR_READERS: Final[dict[int, Callable[[_Reader], Any]]] = {
    UINT8: _Reader.u8,
    INT8: _Reader.i8,
    UINT16: _Reader.u16,
    INT16: _Reader.i16,
    UINT32: _Reader.u32,
    INT32: _Reader.i32,
    FLOAT32: _Reader.f32,
    BOOL: _Reader.boolean,
    UINT64: _Reader.u64,
    INT64: _Reader.i64,
    FLOAT64: _Reader.f64,
    STRING: _Reader.string,
}


# ---------------------------------------------------------------------------
# Metadata value parsing
# ---------------------------------------------------------------------------


def _read_value(reader: _Reader, value_type: int, *, key: str, max_array_len: int) -> Any:
    if value_type == ARRAY:
        return _read_array(reader, key=key, max_array_len=max_array_len)
    read = _SCALAR_READERS.get(value_type)
    if read is None:
        raise GgufError(f"unknown GGUF value type {value_type} for key {key!r}")
    return read(reader)


def _read_array(reader: _Reader, *, key: str, max_array_len: int) -> Any:
    """Read (or skip past) one ARRAY value.

    Short arrays become plain lists. Long ones become a descriptor
    ``{"__array__": True, "type", "len", "sample"}`` -- with a ``values`` key
    added when the array was short enough to materialise anyway -- because the
    tokenizer arrays would otherwise dominate both runtime and memory.
    """
    elem_type = reader.u32()
    count = reader.u64()
    if elem_type == ARRAY:
        raise GgufError(f"nested arrays are not valid GGUF (key {key!r})")

    if elem_type == STRING:
        return _read_string_array(
            reader, key=key, elem_type=elem_type, count=count, max_array_len=max_array_len
        )

    width = _SCALAR_WIDTH.get(elem_type)
    if width is None:
        raise GgufError(f"unknown GGUF array element type {elem_type} for key {key!r}")

    read = _SCALAR_READERS[elem_type]
    if count <= max_array_len:
        return [read(reader) for _ in range(count)]

    full = (
        elem_type in _NUMERIC_TYPES
        and count <= _FULL_NUMERIC_ARRAY_LEN
        and key not in _BULK_ARRAY_KEYS
    )
    take = count if full else min(count, max_array_len)
    head = [read(reader) for _ in range(take)]
    reader.skip((count - take) * width)
    descriptor: dict[str, Any] = {
        "__array__": True,
        "type": elem_type,
        "len": count,
        "sample": head[:max_array_len],
    }
    if full:
        descriptor["values"] = head
    return descriptor


def _read_string_array(
    reader: _Reader, *, key: str, elem_type: int, count: int, max_array_len: int
) -> Any:
    """Read a string array, skipping the tail of long ones.

    Strings are length-prefixed rather than fixed width, so the tail cannot be
    jumped over in one seek: every length prefix has to be walked. Walking is
    still ~10x cheaper than decoding and retaining 262k str objects.
    """
    if count <= max_array_len:
        return [reader.string() for _ in range(count)]
    head = [reader.string() for _ in range(max_array_len)]
    reader.skip_strings(count - max_array_len)
    return {
        "__array__": True,
        "type": elem_type,
        "len": count,
        "sample": head,
    }


# ---------------------------------------------------------------------------
# Reading a whole file
# ---------------------------------------------------------------------------


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _tensor_n_bytes(dims: Sequence[int], ggml_type: int, unknown: set[int]) -> int:
    """On-disk byte size of one tensor."""
    n_elements = 1
    for dim in dims:
        n_elements *= dim
    try:
        block_size, type_size = GGML_TYPE_SIZES[ggml_type]
    except KeyError:
        unknown.add(ggml_type)
        block_size, type_size = UNKNOWN_TYPE_SIZE
    # Ceiling division: a well-formed file always divides evenly, but rounding
    # up keeps a hand-written/odd file's estimate conservative rather than short.
    n_blocks = -(-n_elements // block_size)
    return n_blocks * type_size


def _read_header(reader: _Reader, path: Path) -> tuple[int, int, int]:
    magic = reader.take(4)
    if magic != GGUF_MAGIC:
        if magic == GGUF_MAGIC_SWAPPED:
            raise GgufError(f"{path}: big-endian GGUF files are not supported")
        raise GgufError(f"{path}: not a GGUF file (magic {magic!r})")
    version = reader.u32()
    if version == 1:
        # v1 used uint32 counts and uint32 tensor dims throughout. No such file
        # has been produced since 2023 and llama.cpp itself dropped support, so
        # fail loudly rather than carry an untestable code path.
        raise GgufError(f"{path}: GGUF v1 is not supported (re-quantise to v3)")
    if version < 1:
        raise GgufError(f"{path}: implausible GGUF version {version}")
    tensor_count = reader.u64()
    kv_count = reader.u64()
    if tensor_count > _MAX_TENSORS:
        raise GgufError(f"{path}: implausible tensor count {tensor_count}")
    if kv_count > _MAX_KV:
        raise GgufError(f"{path}: implausible metadata count {kv_count}")
    return version, tensor_count, kv_count


def _resolve_alignment(kv: dict[str, Any], path: Path) -> int:
    raw = kv.get("general.alignment", DEFAULT_ALIGNMENT)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw <= 0:
        raise GgufError(f"{path}: invalid general.alignment {raw!r}")
    if raw & (raw - 1):
        raise GgufError(f"{path}: general.alignment {raw} is not a power of two")
    return raw


def read_gguf(path: Path, *, load_tensors: bool = True, max_array_len: int = 64) -> GgufFile:
    """Parse the header, metadata KV block and (optionally) tensor-info table.

    With ``load_tensors=False`` the tensor-info table is not read, which makes
    the call a single small read; ``tensors``/``total_tensor_bytes`` come back
    empty and ``data_offset`` is 0 because it cannot be known without walking
    the table.
    """
    path = Path(path)
    try:
        with path.open("rb", buffering=_READ_CHUNK) as fh:
            return _read_stream(
                fh, path, load_tensors=load_tensors, max_array_len=max(1, max_array_len)
            )
    except OSError as exc:
        raise GgufError(f"cannot read GGUF file {path}: {exc}") from exc


def _read_stream(fh: BinaryIO, path: Path, *, load_tensors: bool, max_array_len: int) -> GgufFile:
    reader = _Reader(fh)
    version, tensor_count, kv_count = _read_header(reader, path)

    kv: dict[str, Any] = {}
    for _ in range(kv_count):
        key = reader.string()
        value_type = reader.u32()
        kv[key] = _read_value(reader, value_type, key=key, max_array_len=max_array_len)

    alignment = _resolve_alignment(kv, path)

    if not load_tensors:
        return GgufFile(
            path=path,
            version=version,
            tensor_count=tensor_count,
            alignment=alignment,
            kv=kv,
            tensors=[],
            data_offset=0,
            total_tensor_bytes=0,
        )

    unknown: set[int] = set()
    tensors: list[TensorInfo] = []
    total = 0
    for _ in range(tensor_count):
        name = reader.string()
        n_dims = reader.u32()
        if n_dims > _MAX_DIMS:
            raise GgufError(f"{path}: tensor {name!r} claims {n_dims} dimensions")
        dims = tuple(reader.u64() for _ in range(n_dims))
        ggml_type = reader.u32()
        offset = reader.u64()
        n_bytes = _tensor_n_bytes(dims, ggml_type, unknown)
        total += n_bytes
        tensors.append(
            TensorInfo(name=name, dims=dims, ggml_type=ggml_type, offset=offset, n_bytes=n_bytes)
        )

    data_offset = _align_up(reader.offset, alignment)
    # A truncated file can parse "successfully" if the damage is past the point
    # we read, so cross-check the one invariant we can: the data section must
    # start inside the file.
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size and data_offset > size:
        raise GgufError(
            f"{path}: tensor data offset {data_offset} is past end of file ({size} bytes)"
        )

    return GgufFile(
        path=path,
        version=version,
        tensor_count=tensor_count,
        alignment=alignment,
        kv=kv,
        tensors=tensors,
        data_offset=data_offset,
        total_tensor_bytes=total,
        unknown_ggml_types=tuple(sorted(unknown)),
    )


def is_gguf(path: Path) -> bool:
    """Cheap magic-byte check; False for anything unreadable."""
    try:
        with Path(path).open("rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def looks_like_mmproj(path: Path) -> bool:
    """Filename-only heuristic for a vision projector file (no I/O)."""
    return "mmproj" in Path(path).name.lower()


#: A speculative-decoding draft module is small by construction -- it is a few
#: layers, not a model. Used only to break the ambiguous case in
#: :func:`looks_like_auxiliary_gguf`; well above the ~0.8 GiB modules seen in
#: the wild and far below any real quant of a model worth drafting for.
DRAFT_MODULE_MAX_BYTES: Final = 1_610_612_736  # 1.5 GiB


def _name_tokens(stem: str) -> list[str]:
    """Filename stem split on the separators publishers actually use."""
    return [tok for tok in re.split(r"[-_.\s]+", stem.lower()) if tok]


def looks_like_auxiliary_gguf(path: Path | str, *, size_bytes: int | None = None) -> bool:
    """True for a ``.gguf`` that is *not* a loadable model (no I/O).

    Repos increasingly ship GGUFs that are not models: unsloth publishes MTP
    speculative-decoding draft modules under ``MTP/`` and an ``imatrix_*.gguf``
    calibration file beside the real quants. Both parse as quants by filename,
    so without this they became their own rows in the Download tab -- rows that
    could not be downloaded anyway, because a ``MTP/`` path separator is
    refused by ``safe_filename``.

    **Matching is on tokens and path segments, never substrings**, because
    ``mtp`` is also a legitimate part of a *model* name: a model with an MTP
    head is commonly published as ``Qwen3.8-27B-NVFP4-MTP-Q6_K.gguf``, and that
    is a real 20 GiB model that must stay selectable. What separates the two is
    *position*, not presence -- a draft module lives in an ``MTP/`` directory or
    leads with an ``mtp-`` token, whereas a full model carries ``-MTP-`` in the
    middle of its name. The middle case is genuinely ambiguous by name alone,
    so it is only treated as auxiliary when a known size says it is too small
    to be a model (:data:`DRAFT_MODULE_MAX_BYTES`); with no size, the file is
    kept. Keeping a stray draft module is a cosmetic bug; dropping somebody's
    model is a real one.
    """
    pure = PurePosixPath(str(path).replace("\\", "/"))
    # An "MTP/" directory is unambiguous: that is where the draft modules go.
    if any(part.lower() == "mtp" for part in pure.parts[:-1]):
        return True

    stem = pure.name
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    tokens = _name_tokens(stem)
    if not tokens:
        return False

    # A calibration matrix is never loadable, wherever it sits in the name.
    if "imatrix" in tokens:
        return True

    if tokens[0] == "mtp":
        return True

    # Ambiguous position: "-MTP-" in the middle of a name is how a full model
    # with an MTP head is published, so only a known, tiny size settles it.
    return (
        "mtp" in tokens[1:] and size_bytes is not None and 0 < size_bytes <= DRAFT_MODULE_MAX_BYTES
    )


def shard_paths_for(path: Path) -> list[Path]:
    """All sibling shards of a multi-part GGUF, in index order.

    llama.cpp names splits ``<base>-00001-of-00003.gguf`` and puts the full
    metadata only in the *first* shard, so callers must read shard 1 for
    metadata but every shard for the weight total. A non-sharded path comes
    back as a one-element list.
    """
    path = Path(path)
    match = _SHARD_RE.match(path.name)
    if match is None:
        return [path]
    base = match.group("base")
    total = int(match.group("total"))
    found: list[Path] = []
    for index in range(1, total + 1):
        candidate = path.with_name(f"{base}-{index:05d}-of-{total:05d}.gguf")
        if candidate.is_file():
            found.append(candidate)
    if not found:
        # The path we were handed exists as far as the caller is concerned;
        # never return an empty list, that would look like "no files".
        return [path]
    return found


def missing_shard_names(path: Path) -> list[str]:
    """Names of shards implied by ``path``'s filename that are not on disk."""
    path = Path(path)
    match = _SHARD_RE.match(path.name)
    if match is None:
        return []
    base = match.group("base")
    total = int(match.group("total"))
    missing: list[str] = []
    for index in range(1, total + 1):
        name = f"{base}-{index:05d}-of-{total:05d}.gguf"
        if not path.with_name(name).is_file():
            missing.append(name)
    return missing


# ---------------------------------------------------------------------------
# KV accessors
# ---------------------------------------------------------------------------


def _is_array_descriptor(value: Any) -> bool:
    return isinstance(value, dict) and value.get("__array__") is True


def array_len(value: Any) -> int | None:
    """Element count of an array value, whether materialised or skipped."""
    if _is_array_descriptor(value):
        length = value.get("len")
        return int(length) if isinstance(length, int) else None
    if isinstance(value, list):
        return len(value)
    return None


def _array_items(value: Any) -> list[Any]:
    if _is_array_descriptor(value):
        items = value.get("values") or value.get("sample") or []
        return list(items) if isinstance(items, list) else []
    if isinstance(value, list):
        return list(value)
    return []


def _as_int(value: Any) -> int | None:
    """Coerce a KV value to int; arrays collapse to their max.

    Per-layer arrays appear for ``attention.head_count_kv`` in some
    architectures. The maximum is the right collapse for KV-cache sizing: it
    over-estimates rather than under-estimates memory.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    items = [int(v) for v in _array_items(value) if isinstance(v, (int, float))]
    if items:
        return max(items)
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    items = [float(v) for v in _array_items(value) if isinstance(v, (int, float))]
    if items:
        return max(items)
    return None


def _as_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


# ---------------------------------------------------------------------------
# Quantisation labelling
# ---------------------------------------------------------------------------


def quant_label_from(
    kv: dict[str, Any],
    tensors: list[TensorInfo],
    file_type: int | None,
    *,
    path: Path | None = None,
) -> str:
    """Best available human label for the quantisation of a file.

    Three sources in decreasing reliability: the ``general.file_type`` preset
    (the only one that can distinguish Q4_K_M from Q4_K_S), the dominant ggml
    type of the big 2-D weight tensors, and finally the filename -- which is
    the only source for community formats llama.cpp has no ftype for (NVFP4,
    unsloth's UD-* mixes). ``path`` is keyword-only so the documented
    positional signature stays intact for callers that have no path.
    """
    if file_type is None:
        file_type = _as_int(kv.get("general.file_type"))
    if file_type is not None:
        label = LLAMA_FTYPE_LABELS.get(file_type)
        if label is not None:
            from_name = quant_label_from_filename(path) if path is not None else None
            if from_name is not None and _refines(from_name, label):
                return from_name
            return label

    dominant = _dominant_weight_type(tensors)
    if dominant is not None:
        return dominant

    from_name = quant_label_from_filename(path) if path is not None else None
    return from_name or "unknown"


def _quant_family(label: str) -> str:
    """Coarse bucket of a quant label: ``Q4_K_M`` and ``Q4_0`` are both ``Q4``."""
    match = re.match(r"(I?Q\d)", label)
    return match.group(1) if match else label


def _refines(from_name: str, ftype_label: str) -> bool:
    """Whether a filename label is a strictly more specific same-family label.

    Custom mixes (unsloth's ``UD-Q4_K_XL``) have to pick *some* existing ftype
    id to write, and they pick a coarse one: the real
    ``gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`` reports ftype 2 (Q4_0) while
    actually being a K-quant mix. Requiring the same family and a longer label
    keeps that refinement while ignoring a filename that is merely vaguer than
    the ftype (a ``...Q4_0.gguf`` name on a genuine Q4_K_M file).
    """
    if from_name == ftype_label:
        return False
    if _quant_family(from_name) != _quant_family(ftype_label):
        return False
    return len(from_name) > len(ftype_label)


def _dominant_weight_type(tensors: Sequence[TensorInfo]) -> str | None:
    """Most common ggml type by bytes among the large 2-D weight tensors.

    Restricted to 2-D tensors because norms/biases are 1-D and always F32, and
    weighted by bytes because a handful of F32 outliers (token embeddings kept
    at high precision) must not outvote every attention/FFN matrix.
    """
    by_type: dict[int, int] = {}
    for tensor in tensors:
        if len(tensor.dims) != 2 or tensor.n_elements < 4096:
            continue
        by_type[tensor.ggml_type] = by_type.get(tensor.ggml_type, 0) + tensor.n_bytes
    if not by_type:
        return None
    best = max(by_type.items(), key=lambda item: item[1])[0]
    return GGML_TYPE_NAMES.get(best)


def quant_label_from_filename(path: Path | None) -> str | None:
    """Longest whitelisted quant token in the filename, or None."""
    if path is None:
        return None
    stem = Path(path).name.upper()
    if stem.endswith(".GGUF"):
        stem = stem[: -len(".GGUF")]
    # Shard suffixes would otherwise leave "-00001-OF-00002" in the way.
    stem = re.sub(r"-\d{5}-OF-\d{5}$", "", stem)
    candidates: list[str] = [
        str(match) for match in _QUANT_TOKEN_RE.findall(stem) if match in KNOWN_QUANT_LABELS
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


# ---------------------------------------------------------------------------
# GgufMeta assembly
# ---------------------------------------------------------------------------


@dataclass
class _Caps:
    """Detected capability flags, collected before GgufMeta is built."""

    is_mmproj: bool = False
    has_vision_tensors: bool = False
    is_adapter: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _detect_caps(path: Path, gguf: GgufFile, arch: str) -> _Caps:
    kv = gguf.kv
    caps = _Caps()
    has_clip_keys = any(key.startswith("clip.") for key in kv)
    vision_tensors = [t for t in gguf.tensors if t.name.startswith(_VISION_TENSOR_PREFIXES)]
    text_tensors = any(t.name.startswith(("blk.", "token_embd", "output")) for t in gguf.tensors)

    caps.has_vision_tensors = bool(vision_tensors) or has_clip_keys
    caps.is_mmproj = (
        has_clip_keys or arch == "clip" or bool(vision_tensors) or looks_like_mmproj(path)
    )
    # A single-file multimodal model carries both vision *and* text blocks; it
    # is a loadable model, not a projector, and calling it a projector would
    # make the registry pair it with itself.
    if caps.is_mmproj and text_tensors and not looks_like_mmproj(path):
        caps.is_mmproj = False
        caps.extra["single_file_multimodal"] = True

    lora_tensors = [
        t for t in gguf.tensors if any(marker in t.name.lower() for marker in _LORA_TENSOR_MARKERS)
    ]
    caps.is_adapter = (
        _as_str(kv.get("general.type")) == "adapter" or "adapter.type" in kv or bool(lora_tensors)
    )
    if caps.is_adapter:
        alpha = _as_float(kv.get("adapter.lora.alpha"))
        if alpha is not None:
            caps.extra["adapter_alpha"] = alpha
        adapter_type = _as_str(kv.get("adapter.type"))
        if adapter_type:
            caps.extra["adapter_type"] = adapter_type
        rank = _infer_lora_rank(lora_tensors)
        if rank is not None:
            caps.extra["adapter_rank"] = rank
    return caps


def _infer_lora_rank(lora_tensors: Sequence[TensorInfo]) -> int | None:
    """Rank of a LoRA = the short axis of its A/B factors."""
    for tensor in lora_tensors:
        if len(tensor.dims) == 2:
            return min(tensor.dims)
    return None


def _vision_geometry(kv: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    image_size = _as_int(kv.get("clip.vision.image_size"))
    patch_size = _as_int(kv.get("clip.vision.patch_size"))
    n_patch: int | None = None
    if image_size and patch_size:
        n_patch = (image_size // patch_size) ** 2
    return image_size, patch_size, n_patch


def _sum_shard_bytes(
    path: Path, main: GgufFile, shard_paths: Sequence[Path], extra: dict[str, Any]
) -> int:
    """Total tensor bytes across every shard of a split model.

    A 2-shard 123B model whose weight total only counted shard 1 would look
    like it fits in 40 GB of VRAM when it needs 55 GB, so this is load-bearing
    for the planner rather than cosmetic.
    """
    total = 0
    counted: list[str] = []
    missing: list[str] = []
    for candidate in shard_paths:
        shard = Path(candidate)
        if _same_file(shard, path):
            total += main.total_tensor_bytes
            counted.append(shard.name)
            continue
        if not shard.is_file():
            missing.append(shard.name)
            continue
        total += read_gguf(shard).total_tensor_bytes
        counted.append(shard.name)
    if missing:
        extra["missing_shards"] = missing
    extra["shard_count"] = len(counted)
    return total


def _same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def read_meta(path: Path, *, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
    """Read one GGUF file into the planner-facing :class:`GgufMeta`.

    ``shard_paths`` is explicit rather than auto-discovered so that callers
    that already enumerated a model's files (the registry) do not pay for a
    second directory scan; pass ``shard_paths_for(path)`` to opt in.
    """
    path = Path(path)
    gguf = read_gguf(path)
    return meta_from_gguf(gguf, path=path, shard_paths=shard_paths)


def meta_from_gguf(
    gguf: GgufFile,
    *,
    path: Path,
    shard_paths: Sequence[Path] | None = None,
    tensor_bytes: int = 0,
    local: bool = True,
) -> GgufMeta:
    """Turn a parsed :class:`GgufFile` into the planner-facing :class:`GgufMeta`.

    Split out of :func:`read_meta` so that "kv -> GgufMeta" has exactly ONE
    implementation. The second caller is
    :func:`studioforge.core.hf_meta.remote_meta`, which parses a GGUF header
    over HTTP range requests before the file exists on disk; a second copy of
    this mapping would drift the moment either side learned a new key (the
    hybrid ``full_attention_interval`` extraction below is precisely the kind
    of thing that would have been missed remotely), and the whole point of
    reading the remote header is that its answer matches what the local file
    will say once it lands.

    ``tensor_bytes`` overrides the summed tensor-table total. It exists for the
    remote reader, which parses with ``load_tensors=False`` (the tensor table of
    a big model is megabytes of names nobody needs) and knows the file size from
    HuggingFace instead. Zero means "use what the table said".

    ``local=False`` says the file is not on this disk. It suppresses the
    filesystem probe for sibling shards, which would otherwise report every
    shard of a split model as missing -- true, and useless, when nothing has
    been downloaded yet.
    """
    path = Path(path)
    kv = gguf.kv

    arch = _as_str(kv.get("general.architecture")) or "unknown"
    prefix = f"{arch}."

    n_layer = _as_int(kv.get(f"{prefix}block_count")) or 0
    n_embd = _as_int(kv.get(f"{prefix}embedding_length")) or 0
    n_head = _as_int(kv.get(f"{prefix}attention.head_count")) or 0
    n_ctx_train = _as_int(kv.get(f"{prefix}context_length")) or 0

    extra: dict[str, Any] = {
        "gguf_version": gguf.version,
        "tensor_count": gguf.tensor_count,
        "alignment": gguf.alignment,
        "data_offset": gguf.data_offset,
    }

    # GQA: absent head_count_kv means multi-head attention, i.e. n_head_kv ==
    # n_head. Defaulting it to 0 instead would make the KV cache estimate zero.
    raw_head_kv: Any = kv.get(f"{prefix}attention.head_count_kv")
    n_head_kv = _as_int(raw_head_kv)
    head_kv_len = array_len(raw_head_kv)
    if head_kv_len is not None:
        # Per-layer GQA (Gemma-3n style). Max is conservative for sizing.
        extra["head_count_kv_per_layer"] = True
        extra["head_count_kv_len"] = head_kv_len
        if _is_array_descriptor(raw_head_kv) and "values" not in dict(raw_head_kv):
            extra["head_count_kv_truncated"] = True
        # The values themselves, whenever they exist -- not only for iSWA
        # models. A layer with zero KV heads has no cache at all (Gemma-3n,
        # LFM2, Nemotron-H), and the scalar collapse below hides that: it
        # reports the maximum, which is right for a worst case and wrong for
        # every per-layer sum the planner now does.
        per_layer_heads = _array_items(raw_head_kv)
        if per_layer_heads:
            extra["head_count_kv_values"] = [int(x) for x in per_layer_heads]
    if n_head_kv is None:
        n_head_kv = n_head

    n_embd_head_k = _as_int(kv.get(f"{prefix}attention.key_length"))
    if n_embd_head_k is None:
        n_embd_head_k = n_embd // n_head if n_head else 0
    n_embd_head_v = _as_int(kv.get(f"{prefix}attention.value_length"))
    if n_embd_head_v is None:
        n_embd_head_v = n_embd_head_k

    # Interleaved sliding-window attention (Gemma 3/4 "iSWA"): most layers keep
    # only a short window of KV, and only every Nth layer holds the full
    # context. Sizing every layer at full context over-estimates the cache by
    # an order of magnitude on these architectures, which then refuses contexts
    # that fit with room to spare. Captured here; consumed by the planner.
    swa_window = _as_int(kv.get(f"{prefix}attention.sliding_window"))
    swa_pattern = _array_items(kv.get(f"{prefix}attention.sliding_window_pattern"))
    if swa_window and swa_pattern:
        extra["swa_window"] = int(swa_window)
        # True marks a sliding-window layer, False a full-attention one.
        extra["swa_pattern"] = [bool(x) for x in swa_pattern]
        extra["swa_key_length"] = (
            _as_int(kv.get(f"{prefix}attention.key_length_swa")) or n_embd_head_k
        )
        extra["swa_value_length"] = (
            _as_int(kv.get(f"{prefix}attention.value_length_swa")) or n_embd_head_v
        )

    # Hybrid attention/recurrent stacks (qwen3next, qwen35, qwen35moe): only
    # every Nth layer has a KV cache; the others are Gated-DeltaNet layers whose
    # state is a fixed size per *sequence* rather than per token. Without these
    # keys the planner charges full KV for all of them -- a straight 4x, which
    # placed a 27B on four GPUs and forced a q4_0 cache on the 122B at 262k.
    # ``time_step_rank`` sizes weights rather than state and is captured only so
    # the catalog can describe the model.
    interval = _as_int(kv.get(f"{prefix}full_attention_interval"))
    if interval:
        extra["full_attention_interval"] = int(interval)
    for suffix, name in (
        ("ssm.conv_kernel", "ssm_conv_kernel"),
        ("ssm.inner_size", "ssm_inner_size"),
        ("ssm.state_size", "ssm_state_size"),
        ("ssm.group_count", "ssm_group_count"),
        ("ssm.time_step_rank", "ssm_time_step_rank"),
    ):
        value = _as_int(kv.get(f"{prefix}{suffix}"))
        if value is not None:
            extra[name] = int(value)

    # Multi-token-prediction heads are counted in ``block_count`` but are not
    # run during ordinary decoding, so they hold neither a KV cache nor a
    # recurrent state. Counting them as recurrent layers over-charges every
    # slot by one layer's worth of state.
    nextn = _as_int(kv.get(f"{prefix}nextn_predict_layers"))
    if nextn:
        extra["nextn_predict_layers"] = int(nextn)

    n_vocab = array_len(kv.get("tokenizer.ggml.tokens"))
    if n_vocab is None:
        n_vocab = _as_int(kv.get(f"{prefix}vocab_size")) or 0

    chat_template = _as_str(kv.get("tokenizer.chat_template")) or _as_str(
        kv.get("tokenizer.chat_template.default")
    )

    file_type = _as_int(kv.get("general.file_type"))
    caps = _detect_caps(path, gguf, arch)
    extra.update(caps.extra)
    if gguf.unknown_ggml_types:
        extra["unknown_ggml_types"] = list(gguf.unknown_ggml_types)

    if shard_paths:
        tensor_bytes = _sum_shard_bytes(path, gguf, shard_paths, extra)
    elif tensor_bytes <= 0:
        tensor_bytes = gguf.total_tensor_bytes
    if local and not shard_paths:
        implied_missing = missing_shard_names(path)
        if implied_missing:
            extra["missing_shards"] = implied_missing

    image_size, patch_size, n_patch = _vision_geometry(kv)

    return GgufMeta(
        architecture=arch,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_ctx_train=n_ctx_train,
        n_vocab=n_vocab,
        n_embd_head_k=n_embd_head_k,
        n_embd_head_v=n_embd_head_v,
        rope_freq_base=_as_float(kv.get(f"{prefix}rope.freq_base")) or 0.0,
        n_expert=_as_int(kv.get(f"{prefix}expert_count")) or 0,
        n_expert_used=_as_int(kv.get(f"{prefix}expert_used_count")) or 0,
        file_type=file_type,
        quant_label=quant_label_from(kv, gguf.tensors, file_type, path=path),
        param_count=_as_int(kv.get("general.parameter_count")),
        tensor_bytes=tensor_bytes,
        tokenizer_model=_as_str(kv.get("tokenizer.ggml.model")) or "",
        chat_template=chat_template,
        has_vision_tensors=caps.has_vision_tensors,
        is_mmproj=caps.is_mmproj,
        is_adapter=caps.is_adapter,
        vision_n_patch=n_patch,
        vision_image_size=image_size,
        vision_patch_size=patch_size,
        extra=extra,
    )
