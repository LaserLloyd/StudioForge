"""VRAM planner and memory guard.

**GPU-only, always.** ``n_gpu_layers`` is unconditionally "all layers"; the
planner's job is not deciding *how much* goes to GPU but *whether it fits and
where*. There is no CPU-offload path anywhere in this module (or the codebase)
and none should be added -- a partially-offloaded model is exactly the silent
degradation this project exists to avoid.

The estimate is deliberately explicit about being an estimate: every load
records predicted-vs-actual through :meth:`Planner.observe`, and
:func:`suggest_overhead_fraction` turns that history into a tuned fudge factor.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from studioforge.config import Config, FlashAttn, KvCacheType, QuantAffinity, SplitMode
from studioforge.core.gpu import vram_processes
from studioforge.core.kv_sensitivity import KV_QUALITY_LADDER
from studioforge.logging import get_logger
from studioforge.types import (
    MB,
    AdapterRecord,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    LoadRejected,
    ModelRecord,
    PlanResult,
    VramEstimate,
    VramProcess,
)

if TYPE_CHECKING:
    from studioforge.core.gpu import GpuProbe

log = get_logger(__name__)

# Bytes per element for each KV cache type, straight from ggml's block layouts.
# The quantized entries are fractional because they store a whole block of 32
# elements in a fixed number of bytes (e.g. q8_0 = 34 bytes per 32 values).
KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}

# Ordered cheapest-context-first, for the "try a smaller KV cache" suggestion.
#: q5_1/q5_0/q4_1 are deliberately absent: the prebuilt CUDA engines are not
#: built with GGML_CUDA_FA_ALL_QUANTS, so flash attention cannot run those KV
#: types. With --flash-attn on, llama.cpp silently places attention on the CPU
#: -- which would break the GPU-only promise without failing loudly. Only types
#: this engine can actually run in VRAM belong here.
KV_DOWNGRADE_ORDER: tuple[KvCacheType, ...] = ("f16", "q8_0", "q4_0")

# Context sizes we are willing to suggest as a fallback, descending.
_CTX_LADDER = (
    262144,
    131072,
    98304,
    65536,
    49152,
    32768,
    24576,
    16384,
    12288,
    8192,
    6144,
    4096,
    3072,
    2048,
    1024,
    512,
)


#: Context preferred for a *thinking* model that has no explicit ``ctx_size``.
#: A reasoning model spends its budget thinking before it answers, so the
#: ordinary 8192 default truncates the chain of thought and the visible answer
#: with it -- the failure looks like a model that stops mid-sentence, not like
#: a context that is too small. Always clamped to the model's trained window
#: and to what actually fits; overridden by any explicit per-model ``ctx_size``.
#: Overridable via ``models.thinking_default_ctx``; this is the fallback for
#: a config object that predates the field.
THINKING_DEFAULT_CTX = 32768


class PlannerError(Exception):
    """Raised only for programmer error (e.g. a model with no metadata)."""


def kv_bytes_per_element(kv_type: str) -> float:
    try:
        return KV_BYTES_PER_ELEMENT[kv_type]
    except KeyError:
        # Unknown/new cache type: assume f16 rather than guessing low, because
        # under-estimating the KV cache is what produces an OOM at load.
        log.warning("unknown kv cache type, assuming f16", kv_cache_type=kv_type)
        return 2.0


#: llama.cpp sizes the sliding-window cache as
#: ``GGML_PAD(min(full_ctx, n_swa * n_seq_max + n_ubatch), 256)``
#: (llama-kv-cache-iswa.cpp). A flat multiplier was wrong in the dangerous
#: direction: at window=1024 it gave 1280 cells against a real 1536, and it
#: ignored ``parallel`` entirely -- 3.6x under at 4 slots. Under-estimating KV
#: is an OOM at load, so this mirrors the real formula instead.
SWA_CELL_ALIGN = 256
DEFAULT_UBATCH = 512


# ---------------------------------------------------------------------------
# Per-layer KV geometry
# ---------------------------------------------------------------------------
#
# "Every layer holds the full context" is true of a plain llama and of nothing
# else on this rig. Four shapes have to be told apart, and getting it wrong is
# never harmless: it either refuses contexts that fit (iSWA, 10-40x over) or
# advertises concurrency that does not exist. The per-layer list below is the
# one place that knowledge lives; every KV number in the codebase is a fold
# over it.


@dataclass(frozen=True)
class KvLayer:
    """What one transformer block costs in cache, in llama.cpp's terms.

    ``kind`` is the only field callers branch on:

    * ``"full"`` -- ordinary attention, ``ctx_total`` cells.
    * ``"swa"`` -- sliding window, ``GGML_PAD(min(ctx, window*slots+ubatch), 256)``
      cells at (usually) half the head dimension.
    * ``"none"`` -- no KV cache at all: a Gated-DeltaNet/SSM recurrent layer, a
      layer whose per-layer ``head_count_kv`` entry is zero, or an MTP head.
      Its state, if it has one, is counted by
      :func:`recurrent_state_bytes_per_slot` because it is per *sequence*
      rather than per token.
    """

    kind: Literal["full", "swa", "none"]
    n_head_kv: int
    head_k: int
    head_v: int
    window: int = 0  # swa only


#: The shared singleton for a layer that caches nothing. Frozen, so one
#: instance can be repeated across a whole layer list without copying.
_NO_KV = KvLayer(kind="none", n_head_kv=0, head_k=0, head_v=0)


def kv_layers(meta: Any) -> list[KvLayer]:
    """Per-layer KV geometry for one model, in block order.

    Four architectures share this function because each breaks the uniform
    assumption differently, and all four are in this library:

    * **iSWA** (Gemma 3/4): ``attention.sliding_window_pattern`` marks five
      sliding-window layers per full-attention one, and the window layers use
      ``key_length_swa``/``value_length_swa`` -- half the head dimension.
    * **hybrid** (``qwen3next``/``qwen35``/``qwen35moe``): layer ``il`` has a KV
      cache only when ``(il + 1) % full_attention_interval == 0``; the rest are
      Gated-DeltaNet recurrent layers holding a fixed per-sequence state. The
      planner used to charge full KV for every layer, a straight 4x over --
      enough to spread a 27B across four GPUs it did not need and to force the
      122B onto a q4_0 cache at 262k.
    * **per-layer GQA arrays** (Gemma-3n, LFM2, Nemotron-H): an entry of zero
      means that layer has no cache.
    * everything else: ``n_layer`` uniform full-attention layers.

    Returns ``[]`` when the metadata cannot support a per-layer answer (no
    layer count, no head dimension, no head count anywhere). Callers must read
    that as "cannot estimate" and never as "free" -- summing an empty list into
    a zero-byte KV cache would report a fit for a model nobody has measured.
    """
    extra = getattr(meta, "extra", None) or {}
    pattern = [bool(x) for x in (extra.get("swa_pattern") or [])]
    window = int(extra.get("swa_window") or 0)

    n_layer = int(getattr(meta, "n_layer", 0) or 0)
    if n_layer <= 0:
        # A GGUF with a sliding-window pattern but no block_count still has a
        # usable layer count: the pattern is one entry per layer.
        n_layer = len(pattern)
    head_k = int(getattr(meta, "head_dim_k", 0) or 0)
    head_v = int(getattr(meta, "head_dim_v", 0) or 0) or head_k
    if n_layer <= 0 or head_k <= 0:
        return []

    per_layer = [int(x) for x in (extra.get("head_count_kv_values") or [])]
    default_heads = int(getattr(meta, "n_head_kv", 0) or 0) or int(getattr(meta, "n_head", 0) or 0)
    if not per_layer and default_heads <= 0:
        return []

    def heads_at(index: int) -> int:
        return per_layer[index % len(per_layer)] if per_layer else default_heads

    if pattern and window > 0:
        swa_k = int(extra.get("swa_key_length") or head_k)
        swa_v = int(extra.get("swa_value_length") or head_v)
        layers: list[KvLayer] = []
        for il in range(n_layer):
            n_kv = heads_at(il)
            if n_kv <= 0:
                layers.append(_NO_KV)
            elif pattern[il % len(pattern)]:
                layers.append(KvLayer("swa", n_kv, swa_k, swa_v, window))
            else:
                layers.append(KvLayer("full", n_kv, head_k, head_v))
        return layers

    interval = int(extra.get("full_attention_interval") or 0)
    if interval > 0:
        # Position decides, including for the trailing ``nextn_predict_layers``
        # MTP head: llama.cpp gives it no KV cache either way, and treating it
        # by position is what keeps this a one-line rule instead of a table of
        # per-architecture exceptions.
        hybrid: list[KvLayer] = []
        for il in range(n_layer):
            n_kv = heads_at(il)
            if n_kv <= 0 or (il + 1) % interval != 0:
                hybrid.append(_NO_KV)
            else:
                hybrid.append(KvLayer("full", n_kv, head_k, head_v))
        return hybrid

    uniform: list[KvLayer] = []
    for il in range(n_layer):
        n_kv = heads_at(il)
        uniform.append(KvLayer("full", n_kv, head_k, head_v) if n_kv > 0 else _NO_KV)
    return uniform


def attention_kind(meta: Any) -> str:
    """``"full"`` | ``"iswa"`` | ``"hybrid"`` | ``"unknown"`` -- a catalog column.

    Derived from :func:`kv_layers` rather than from ``general.architecture``,
    because the architecture string is not a reliable predictor: ``gemma4``
    covers both a dense iSWA model and a MoE one, and a new hybrid arch name
    lands upstream every few weeks. ``"unknown"`` means the metadata could not
    support a per-layer answer at all, which is a reason to distrust every KV
    number for that model -- not a reason to assume the cheap case.
    """
    layers = kv_layers(meta)
    if not layers:
        return "unknown"
    kinds = {layer.kind for layer in layers}
    if "swa" in kinds:
        return "iswa"
    if "none" in kinds:
        return "hybrid"
    return "full"


def recurrent_state_bytes_per_slot(meta: Any) -> int:
    """Bytes of Gated-DeltaNet/SSM state one sequence needs, all layers summed.

    llama.cpp's recurrent memory (llama-memory-recurrent.cpp) allocates two f32
    buffers per recurrent layer per sequence::

        n_embd_r = (ssm_d_conv - 1) * (ssm_d_inner + 2 * ssm_n_group * ssm_d_state)
        n_embd_s = ssm_d_state * ssm_d_inner

    Unlike a KV cache this does not grow with context -- that is the whole point
    of the architecture -- but it does grow with ``n_seq_max``, so it is charged
    per slot. 157 MB for the Qwen3.5-27B and 156 MB for the 122B-A10B: trivial
    next to the 65 GB of KV those models were being over-charged, and large
    enough that dropping it would under-estimate an 8-slot load by more than a
    gigabyte. Returns 0 when the model has no SSM keys, i.e. for everything that
    is not hybrid.

    ``nextn_predict_layers`` are excluded from the count. An MTP head is neither
    an attention block nor a recurrent mixer and llama.cpp does not run it
    during ordinary decoding, so it holds neither kind of state.
    :func:`kv_layers` still lists it as ``none`` (it has no KV either), which is
    why the recurrent count is derived here rather than read off that list.
    ``ssm.time_step_rank`` is deliberately unused: it sizes weights, not state.
    """
    return _recurrent_state_bytes(meta, kv_layers(meta))


def _recurrent_state_bytes(meta: Any, layers: Sequence[KvLayer]) -> int:
    """:func:`recurrent_state_bytes_per_slot` with the layer list already built.

    Every caller of the public function has just built that list, and building
    it twice per estimate is pure waste in the middle of a ladder walk.
    """
    extra = getattr(meta, "extra", None) or {}
    d_conv = int(extra.get("ssm_conv_kernel") or 0)
    d_inner = int(extra.get("ssm_inner_size") or 0)
    d_state = int(extra.get("ssm_state_size") or 0)
    n_group = max(0, int(extra.get("ssm_group_count") or 0))
    if min(d_conv, d_inner, d_state) <= 0 or not layers:
        return 0
    nextn = max(0, int(extra.get("nextn_predict_layers") or 0))
    n_recurrent = sum(1 for layer in layers if layer.kind == "none") - nextn
    if n_recurrent <= 0:
        return 0
    n_embd_r = (d_conv - 1) * (d_inner + 2 * n_group * d_state)
    n_embd_s = d_state * d_inner
    return n_recurrent * (n_embd_r + n_embd_s) * 4


def _alloc_bytes(
    layers: Sequence[KvLayer],
    *,
    ctx_total: int,
    kv_type_k: str,
    kv_type_v: str,
    parallel: int,
    ubatch: int,
) -> int:
    """Sum the KV cells llama.cpp really allocates over a layer list.

    The single implementation of the cell arithmetic. Every public KV-size
    function folds over this, so the uniform, iSWA and hybrid answers cannot
    drift apart -- which is what happened when the planner carried two of them.
    """
    bytes_k = kv_bytes_per_element(kv_type_k)
    bytes_v = kv_bytes_per_element(kv_type_v)
    total = 0.0
    for layer in layers:
        if layer.kind == "none":
            continue
        if layer.kind == "swa":
            # GGML_PAD(min(full, n_swa * n_seq_max + n_ubatch), 256)
            cells = min(ctx_total, layer.window * max(1, parallel) + max(0, ubatch))
            depth = -(-int(cells) // SWA_CELL_ALIGN) * SWA_CELL_ALIGN
        else:
            depth = ctx_total
        total += depth * layer.n_head_kv * (layer.head_k * bytes_k + layer.head_v * bytes_v)
    return int(total)


def kv_alloc_bytes(
    meta: Any,
    *,
    ctx_total: int,
    kv_k: str,
    kv_v: str,
    parallel: int = 1,
    ubatch: int = DEFAULT_UBATCH,
) -> int:
    """VRAM llama.cpp reserves for cache + recurrent state, for any shape.

    Full layers take ``ctx_total`` cells, sliding-window layers
    ``GGML_PAD(min(ctx_total, window*parallel + ubatch), 256)``, no-KV layers
    nothing, plus :func:`recurrent_state_bytes_per_slot` once per slot. This is
    the number :meth:`Planner.estimate` charges, and it replaces the
    uniform/iSWA pair the planner used to pick between -- a pair that had no
    branch at all for a hybrid model and so charged Qwen3.5 four times over.

    ``ctx_total`` is what reaches ``--ctx-size``: the total shared across slots
    (D4), not the per-slot window. Returns 0 when the geometry is unknown,
    which callers must treat as "cannot estimate".
    """
    if ctx_total <= 0:
        return 0
    layers = kv_layers(meta)
    if not layers:
        return 0
    cache = _alloc_bytes(
        layers,
        ctx_total=ctx_total,
        kv_type_k=kv_k,
        kv_type_v=kv_v,
        parallel=parallel,
        ubatch=ubatch,
    )
    return cache + _recurrent_state_bytes(meta, layers) * max(1, parallel)


def kv_read_bytes_per_slot(meta: Any, *, kv_k: str, kv_v: str, ctx_fill: int) -> int:
    """Bytes one slot reads from cache per decode step, holding ``ctx_fill`` tokens.

    Decode is memory-bandwidth bound, so this -- not the allocation -- is what
    belongs in a throughput estimate and in the concurrency knee. The two
    numbers differ by two orders of magnitude on the models that matter:

    * a **full** layer reads everything it has cached, ``ctx_fill`` tokens;
    * a **swa** layer reads at most its window, 1024 tokens, however long the
      conversation is;
    * a **none** layer reads its recurrent state once -- a fixed cost that does
      not grow with the transcript at all.

    Gemma-4 31B at a 131k fill reads ~1.9 GB per step. Charging it the uniform
    ``131072 x 1.9 MB/token = 258 GB`` is what produced ``est_gen_tps: 1.9`` for
    a 31B on two 5090s that really runs at 39.
    """
    if ctx_fill <= 0:
        return 0
    layers = kv_layers(meta)
    if not layers:
        return 0
    bytes_k = kv_bytes_per_element(kv_k)
    bytes_v = kv_bytes_per_element(kv_v)
    total = 0.0
    for layer in layers:
        if layer.kind == "none":
            continue
        depth = min(ctx_fill, layer.window) if layer.kind == "swa" else ctx_fill
        total += depth * layer.n_head_kv * (layer.head_k * bytes_k + layer.head_v * bytes_v)
    return int(total) + _recurrent_state_bytes(meta, layers)


def effective_kv_bytes_per_token(meta: Any, *, kv_k: str, kv_v: str, ctx_per_slot: int) -> int:
    """Allocated KV bytes per token of context, averaged over the real geometry.

    ``kv_alloc_bytes(ctx_total=ctx_per_slot, parallel=1) // ctx_per_slot``: the
    honest version of :func:`kv_bytes_per_token` for a model whose layers do not
    all cost the same. Identical to it for a uniform model; ~80-155 KB/token
    rather than 1.9 MB for Gemma-4 31B, and a quarter of the uniform figure for
    the Qwen3.5 hybrids.

    It depends on ``ctx_per_slot`` because the cheap layers do not scale with
    context: a sliding window costs the same 1024 cells at 16k as at 262k, so
    the average per-token cost *falls* as the window grows. That is why this
    takes a context instead of being a constant of the model.
    """
    if ctx_per_slot <= 0:
        return 0
    return (
        kv_alloc_bytes(meta, ctx_total=ctx_per_slot, kv_k=kv_k, kv_v=kv_v, parallel=1)
        // ctx_per_slot
    )


def estimate_kv_bytes(
    *,
    n_layer: int,
    n_head_kv: int,
    head_dim_k: int,
    head_dim_v: int,
    ctx_total: int,
    kv_type_k: str,
    kv_type_v: str,
) -> int:
    """KV cache size for a full context, uniform layers.

    K and V are sized separately: some architectures use different key/value
    head dimensions (Gemma's ``attention.key_length``, DeepSeek MLA), and
    llama.cpp allows a different quantization for each side. Treating them as
    one symmetric term mis-sizes both.

    ``ctx_total`` is the value passed to ``--ctx-size``, i.e. the *total*
    context shared across slots -- see DECISIONS.md D4.

    Kept for callers that have loose numbers rather than a ``GgufMeta`` (the
    draft-model term, the downloader's what-if sizing). Anything holding real
    metadata wants :func:`kv_alloc_bytes`, which knows about the layers that do
    not cost this.
    """
    if min(n_layer, n_head_kv, head_dim_k, ctx_total) <= 0:
        return 0
    return _alloc_bytes(
        [KvLayer("full", n_head_kv, head_dim_k, head_dim_v)] * n_layer,
        ctx_total=ctx_total,
        kv_type_k=kv_type_k,
        kv_type_v=kv_type_v,
        parallel=1,
        ubatch=0,
    )


def estimate_kv_bytes_iswa(
    *,
    meta: Any,
    ctx_total: int,
    kv_type_k: str,
    kv_type_v: str,
    parallel: int = 1,
    ubatch: int = DEFAULT_UBATCH,
) -> int | None:
    """KV size for interleaved sliding-window attention, or None if N/A.

    Gemma 3/4 keep only a short window of KV on most layers and the full
    context on every Nth layer. Sizing every layer at full context -- correct
    for ordinary attention -- over-estimates these models by 10-40x and refuses
    contexts that fit easily (D15).

    Now a thin wrapper over :func:`kv_layers`: same arithmetic, byte for byte,
    reached through the one per-layer description every other KV function uses.
    ``None`` still means "not an iSWA model", which is the signal callers built
    their uniform fallback on.
    """
    extra = getattr(meta, "extra", None) or {}
    if not extra.get("swa_pattern") or not extra.get("swa_window") or ctx_total <= 0:
        return None
    layers = kv_layers(meta)
    if not layers:
        return None
    return _alloc_bytes(
        layers,
        ctx_total=ctx_total,
        kv_type_k=kv_type_k,
        kv_type_v=kv_type_v,
        parallel=parallel,
        ubatch=ubatch,
    )


def max_ctx_for_budget(
    *,
    budget_bytes: int,
    n_layer: int,
    n_head_kv: int,
    head_dim_k: int,
    head_dim_v: int,
    kv_type_k: str,
    kv_type_v: str,
    parallel: int = 1,
) -> int:
    """Largest per-slot context whose KV cache fits in ``budget_bytes``."""
    if budget_bytes <= 0 or min(n_layer, n_head_kv, head_dim_k) <= 0:
        return 0
    per_token = n_layer * (
        n_head_kv * head_dim_k * kv_bytes_per_element(kv_type_k)
        + n_head_kv * head_dim_v * kv_bytes_per_element(kv_type_v)
    )
    if per_token <= 0:
        return 0
    total = int(budget_bytes // per_token)
    return max(0, total // max(1, parallel))


def max_ctx_for_budget_geometry(
    meta: Any,
    *,
    budget_bytes: int,
    kv_k: str,
    kv_v: str,
    parallel: int = 1,
) -> int:
    """Largest ladder context per slot whose *real* allocation fits ``budget_bytes``.

    The per-layer answer to the question :func:`max_ctx_for_budget` answers
    uniformly. Walks :data:`_CTX_LADDER` from the top and asks
    :func:`kv_alloc_bytes` -- the same arithmetic a load is charged -- at each
    rung, so the number a refusal offers is one the next load will accept. For
    an iSWA or hybrid model the uniform figure under-offers by 4-40x (D22: the
    window layers do not grow with context), which sent users to a 4k window
    when 65k fit. Returns 0 when nothing on the ladder fits or the geometry is
    unknown; callers fall back to the uniform figure for the latter.
    """
    if budget_bytes <= 0 or not kv_layers(meta):
        return 0
    slots = max(1, int(parallel))
    for ctx in _CTX_LADDER:
        needed = kv_alloc_bytes(meta, ctx_total=ctx * slots, kv_k=kv_k, kv_v=kv_v, parallel=slots)
        if 0 < needed <= budget_bytes:
            return ctx
    return 0


def _round_ctx_down(ctx: int) -> int:
    """Snap to a friendly context size from the ladder (multiples users expect)."""
    for candidate in _CTX_LADDER:
        if candidate <= ctx:
            return candidate
    return 0


# ---------------------------------------------------------------------------
# Parallel slots (DECISIONS.md D17)
# ---------------------------------------------------------------------------

#: Hard ceiling on the slot count the estimator will ever propose. Past this
#: the per-slot context each conversation gets is small enough that a long
#: agent transcript no longer fits, and llama.cpp's per-slot context
#: checkpoints (32 each, unmodelled here) start to matter. A user who really
#: wants more sets ``parallel`` explicitly, which is never capped.
MAX_PARALLEL_CAP = 8

#: How full a slot's context is assumed to be when sizing the knee. Slots
#: rarely sit at their maximum; assuming they do would put the knee at a
#: pessimistic 1-2 slots for every model. Half is the deliberate middle.
CTX_FILL_FRACTION = 0.5

#: MoE knee derate. Experts fan out with batch size -- at batch N roughly
#: min(N * n_expert_used, n_expert) experts are touched per step -- so a MoE's
#: weight traffic grows with N until N ~ n_expert/n_expert_used instead of
#: staying flat like a dense model's. The knee therefore arrives sooner than
#: the dense formula says. 0.5 is a deliberate approximation, not a
#: measurement: it is the safe direction (fewer slots), and the real curve
#: needs a benchmark this rig has not run yet.
MOE_KNEE_DERATE = 0.5


def kv_bytes_per_token(meta: Any, kv_k: str, kv_v: str) -> int:
    """Bytes of KV cache one token of context costs, K and V together.

    The atom every concurrency and context calculation is built from:
    ``n_layer * n_head_kv * (head_dim_k * bpe(k) + head_dim_v * bpe(v))``.
    Returns 0 when the metadata cannot support the calculation, which callers
    read as "cannot estimate" rather than as "free".

    This is the *uniform* per-token cost, and it is wrong for half this
    library: an iSWA model (Gemma 3/4) pays it on one layer in six, and a
    hybrid (Qwen3.5) on one in four. It over-estimates, which is safe for a
    context ladder and *not* safe for anything downstream -- a slot count
    divided by it collapsed to 1 with 34 GB free, and a throughput estimate
    multiplied by it claimed 258 GB of KV traffic per token. Use
    :func:`effective_kv_bytes_per_token` unless the caller genuinely wants the
    uniform figure.
    """
    n_layer = int(getattr(meta, "n_layer", 0) or 0)
    n_head_kv = int(getattr(meta, "n_head_kv", 0) or 0) or int(getattr(meta, "n_head", 0) or 0)
    head_k = int(getattr(meta, "head_dim_k", 0) or 0)
    head_v = int(getattr(meta, "head_dim_v", 0) or 0) or head_k
    if min(n_layer, n_head_kv, head_k) <= 0:
        return 0
    return int(
        n_layer
        * n_head_kv
        * (head_k * kv_bytes_per_element(kv_k) + head_v * kv_bytes_per_element(kv_v))
    )


def is_moe(meta: Any) -> bool:
    """Whether this is a mixture-of-experts model, per its GGUF metadata."""
    n_expert = int(getattr(meta, "n_expert", 0) or 0)
    n_used = int(getattr(meta, "n_expert_used", 0) or 0)
    return n_expert > 1 and n_used > 0


def active_weight_bytes(meta: Any, weights_bytes: int) -> int:
    """Weight bytes actually read per decoded token.

    Dense: all of them. MoE: only the routed experts, approximated as
    ``weights * n_expert_used / n_expert``. That approximation charges the
    shared/dense tensors (attention, embeddings, the shared expert if there is
    one) at the same discount as the routed experts, so it runs a little low
    for a model with a large dense trunk. Low here means a *nearer* knee and
    so fewer slots, which is the direction that cannot hurt.
    """
    weights = max(0, int(weights_bytes))
    n_expert = int(getattr(meta, "n_expert", 0) or 0)
    n_used = int(getattr(meta, "n_expert_used", 0) or 0)
    if n_expert > 1 and 0 < n_used < n_expert:
        return max(1, int(weights * n_used / n_expert))
    return weights


def max_parallel_for(
    *,
    kv_budget_bytes: int,
    kv_per_token: int,
    ctx_per_slot: int,
    active_weight_bytes: int,
    is_moe: bool = False,
    cap: int = MAX_PARALLEL_CAP,
    kv_read_bytes_per_slot: int | None = None,
) -> tuple[int, str]:
    """How many slots this placement can sustain, and what limits it.

    Two independent bounds, and the smaller wins:

    * **VRAM** -- ``--ctx-size`` is the total across slots (D4), so N slots at
      ``ctx_per_slot`` cost N times the KV cache. ``kv_budget_bytes`` is what
      is left for KV after the weights, compute buffers and CUDA contexts are
      paid for, so ``by_vram = kv_budget // (ctx_per_slot * kv_per_token)``.
    * **Knee** -- decode is memory-bandwidth bound. Each step reads the active
      weights once regardless of batch size, plus every busy slot's KV. Once
      the KV traffic matches the weight traffic, another slot stops buying
      throughput and only costs latency and VRAM. That crossover is
      ``active_weight_bytes / kv_read_per_slot``, derated for MoE.

    ``kv_read_bytes_per_slot`` is that KV traffic, measured properly by
    :func:`kv_read_bytes_per_slot`. Without it the knee falls back to
    ``ctx_per_slot * CTX_FILL_FRACTION * kv_per_token``, which assumes every
    layer re-reads the whole transcript -- 135x too much for Gemma-4, and the
    reason a 31B was told it could serve one conversation.

    Returns ``(slots, bound)`` where bound is ``"vram"``, ``"knee"``, ``"cap"``
    or ``"unknown"``. Never returns less than 1: one slot is what a load
    already does today, so this can only ever *add* concurrency, never turn a
    working load into a rejection.
    """
    cap = max(1, int(cap))
    if kv_per_token <= 0 or ctx_per_slot <= 0:
        # No usable metadata. One slot is what happens today; say so honestly
        # rather than inventing a bound.
        return 1, "unknown"

    per_slot_bytes = ctx_per_slot * kv_per_token
    by_vram = int(max(0, kv_budget_bytes) // per_slot_bytes)

    if kv_read_bytes_per_slot and kv_read_bytes_per_slot > 0:
        read_per_slot = float(kv_read_bytes_per_slot)
    else:
        read_per_slot = max(1.0, ctx_per_slot * CTX_FILL_FRACTION) * kv_per_token
    if active_weight_bytes > 0:
        by_knee_exact = active_weight_bytes / read_per_slot
        if is_moe:
            by_knee_exact *= MOE_KNEE_DERATE
        by_knee = int(round(by_knee_exact))
    else:
        by_knee = cap

    chosen = min(by_vram, by_knee)
    if chosen >= cap:
        return cap, "cap"
    # A tie reports "vram": when both bounds land on the same number, the one
    # the user can actually act on (free memory, or lower the context) is the
    # more useful thing to name.
    bound = "vram" if by_vram <= by_knee else "knee"
    return max(1, chosen), bound


def parallel_options(
    meta: Any,
    weights_bytes: int,
    kv_budget_bytes: int,
    ctx_tiers: Sequence[int],
    kv_types: Sequence[str],
    *,
    cap: int = MAX_PARALLEL_CAP,
) -> list[dict[str, Any]]:
    """The (context, KV type) -> slots table, as pure arithmetic.

    Built for the model catalog: one row per ``(ctx_per_slot, kv_cache_type)``
    pair carrying the slot count, which bound produced it, and the KV bytes
    involved. No hardware is touched and nothing is loaded -- the caller
    supplies ``kv_budget_bytes`` for whatever placement it is asking about.
    """
    active = active_weight_bytes(meta, weights_bytes)
    moe = is_moe(meta)
    rows: list[dict[str, Any]] = []
    for ctx in ctx_tiers:
        for kv_type in kv_types:
            per_token = kv_bytes_per_token(meta, kv_type, kv_type)
            slots, bound = max_parallel_for(
                kv_budget_bytes=kv_budget_bytes,
                kv_per_token=per_token,
                ctx_per_slot=int(ctx),
                active_weight_bytes=active,
                is_moe=moe,
                cap=cap,
            )
            rows.append(
                {
                    "ctx_per_slot": int(ctx),
                    "kv_cache_type": kv_type,
                    "kv_bytes_per_token": per_token,
                    "kv_bytes_per_slot": per_token * int(ctx),
                    "kv_bytes": per_token * int(ctx) * slots,
                    "max_parallel": slots,
                    "parallel_limited_by": bound,
                }
            )
    return rows


class Planner:
    """Turns a model + requested settings into a placement decision."""

    def __init__(
        self,
        config: Config,
        probe: GpuProbe,
        *,
        observation_sink: Callable[[dict[str, object]], None] | None = None,
        log_plans: bool = True,
    ) -> None:
        self.config = config
        self.probe = probe
        self._observation_sink = observation_sink
        #: Whether an accepted or refused plan is worth an INFO line. False for
        #: a planner that exists only to *ask* questions -- the catalog runs one
        #: plan per model per context tier per hardware state, which at INFO
        #: would be several hundred lines describing loads nobody requested.
        #: D16 removed exactly this class of spam; a new surface must not
        #: reintroduce it. The lines are still emitted at DEBUG.
        self._log_plans = log_plans

    # -- VRAM accounting -------------------------------------------------

    def estimate(
        self,
        record: ModelRecord,
        *,
        ctx_size: int,
        parallel: int,
        kv_cache_type: KvCacheType,
        kv_cache_type_v: KvCacheType,
        n_devices: int = 1,
        draft: ModelRecord | None = None,
        draft_ctx_size: int | None = None,
        adapters: Sequence[AdapterRecord] = (),
    ) -> VramEstimate:
        """Project VRAM for one load.

        Terms, all of which are real allocations on the device:

        * weights -- summed tensor bytes across every shard
        * KV cache + recurrent state -- sized per layer on the *total* context
          (``ctx_size * parallel``); see :func:`kv_alloc_bytes`
        * compute/graph buffers -- scratch that scales with model width
        * mmproj weights + the image-encoding buffer, for vision models
        * adapter weights
        * draft model weights + its own KV cache
        * a per-GPU CUDA context/cuBLAS workspace charge
        """
        meta = record.meta
        if meta is None:
            raise PlannerError(
                f"model '{record.id}' has no parsed GGUF metadata; cannot plan a load"
            )

        planner_cfg = self.config.planner
        ctx_total = max(1, ctx_size) * max(1, parallel)

        weights = int(meta.tensor_bytes) or int(record.size_bytes)
        # One call for all four shapes: uniform, iSWA, hybrid, per-layer array.
        # Picking between two formulas here is what let the hybrid case fall
        # through the gap and be charged as if it were uniform.
        kv = kv_alloc_bytes(
            meta,
            ctx_total=ctx_total,
            kv_k=kv_cache_type,
            kv_v=kv_cache_type_v,
            parallel=max(1, parallel),
        )

        # Graph/activation buffers track model width and batch size far more than
        # depth. A fraction of the weight size is a crude but stable proxy, with a
        # floor so tiny models still get room for their scratch buffers.
        compute = max(
            planner_cfg.compute_overhead_floor_mb * MB,
            int(weights * planner_cfg.compute_overhead_fraction),
        )

        mmproj_bytes = 0
        mmproj_compute = 0
        if record.capabilities.vision and record.mmproj_path is not None:
            mmproj_bytes = int(record.mmproj_bytes or 0)
            if mmproj_bytes <= 0:
                # Fallback for records built outside a scan; the registry
                # normally fills mmproj_bytes in.
                try:
                    mmproj_bytes = record.mmproj_path.stat().st_size
                except OSError:
                    mmproj_bytes = 0
            mmproj_compute = planner_cfg.mmproj_compute_mb * MB

        adapter_bytes = 0
        for adapter in adapters:
            adapter_bytes += int(adapter.size_bytes or 0)

        draft_weights = 0
        draft_kv = 0
        if draft is not None and draft.meta is not None:
            draft_weights = int(draft.meta.tensor_bytes) or int(draft.size_bytes)
            # The draft shares the target's context window in practice; sizing its
            # KV on the same total is the conservative choice.
            draft_ctx_total = (draft_ctx_size or ctx_size) * max(1, parallel)
            draft_kv = estimate_kv_bytes(
                n_layer=draft.meta.n_layer,
                n_head_kv=draft.meta.n_head_kv or draft.meta.n_head,
                head_dim_k=draft.meta.head_dim_k,
                head_dim_v=draft.meta.head_dim_v,
                ctx_total=draft_ctx_total,
                kv_type_k=kv_cache_type,
                kv_type_v=kv_cache_type_v,
            )

        cuda_context = planner_cfg.cuda_context_mb * MB * max(1, n_devices)

        return VramEstimate(
            weights_bytes=weights,
            kv_bytes=kv,
            compute_bytes=compute,
            mmproj_bytes=mmproj_bytes,
            mmproj_compute_bytes=mmproj_compute,
            adapter_bytes=adapter_bytes,
            draft_weights_bytes=draft_weights,
            draft_kv_bytes=draft_kv,
            cuda_context_bytes=cuda_context,
        )

    # -- capacity ---------------------------------------------------------

    def usable_bytes(self, gpu: GpuInfo, *, forced: bool = False) -> int:
        """Free VRAM minus the configured headroom, exclusions and reservations.

        Headroom is a fraction of *total* rather than of free, so the guard
        means the same thing whether the card is empty or nearly full.

        ``planner.reserved_mb`` is subtracted on top, and always -- it describes
        memory a *neighbour* process needs (ComfyUI, a training job), which does
        not stop being true because someone forced a placement.

        ``planner.excluded_devices`` is different: it is our own placement
        policy, so ``forced=True`` (a per-model ``device_override``) ignores it.
        An explicit choice by the user outranks a default written in config.
        """
        planner_cfg = self.config.planner
        if not forced and gpu.index in planner_cfg.excluded_devices:
            return 0
        headroom = int(gpu.total_bytes * planner_cfg.headroom_fraction)
        reserved = int(planner_cfg.reserved_mb.get(gpu.index, 0)) * MB
        return max(0, gpu.free_bytes - headroom - reserved)

    def _gpu_map(self) -> dict[int, GpuInfo]:
        return {gpu.index: gpu for gpu in self.probe.list_gpus()}

    def _candidate_order(self, gpus: Sequence[GpuInfo]) -> list[int]:
        """GPUs best-first: highest compute capability, then most free VRAM.

        Excluded devices are dropped entirely rather than sorted last: they are
        not a worse option, they are not an option. A ``device_override`` that
        names one still reaches :meth:`_plan_on_devices` directly.
        """
        excluded = set(self.config.planner.excluded_devices)
        return [
            gpu.index
            for gpu in sorted(
                gpus,
                key=lambda g: (
                    -(g.compute_capability or (0, 0))[0],
                    -(g.compute_capability or (0, 0))[1],
                    -self.usable_bytes(g),
                    g.index,
                ),
            )
            if gpu.index not in excluded
        ]

    # -- quantization/hardware affinity -----------------------------------

    def _affinity_for(self, record: ModelRecord) -> QuantAffinity | None:
        """Hardware affinity for this model's quantization family, if any.

        FP4 quants are the motivating case: NVFP4 runs everywhere but only gets
        native tensor-core acceleration on Blackwell, where it is ~6x faster
        than on Ampere (vs ~3x for a plain Q4_0). See
        :func:`studioforge.config.default_quant_affinity` for the measurements.
        """
        table = self.config.planner.quant_affinity
        if not table:
            return None
        quant = (record.quant or "").upper()
        meta_quant = (record.meta.quant_label if record.meta else "").upper()
        for family, affinity in table.items():
            key = family.upper()
            if key and (key in quant or key in meta_quant):
                return affinity
        return None

    def _eligible_for(self, gpu: GpuInfo, affinity: QuantAffinity) -> bool:
        cc = gpu.compute_capability
        if cc is None:
            # Unknown capability: treat as ineligible for a *required* affinity
            # (fail loudly rather than launch something that may not run) but
            # allow it as a fallback when the affinity is only a preference.
            return False
        return cc >= affinity.min_cc_tuple

    def _device_pools(
        self,
        order: list[int],
        gpu_map: dict[int, GpuInfo],
        affinity: QuantAffinity | None,
    ) -> list[tuple[list[int], str | None]]:
        """Candidate device pools, most-preferred first.

        With no affinity there is one pool: every GPU. With a ``prefer``
        affinity, capable GPUs are tried first and the full set second, so a
        model still loads on slower hardware rather than failing. With
        ``require``, only capable GPUs are ever offered.
        """
        if affinity is None:
            return [(order, None)]

        eligible = [i for i in order if self._eligible_for(gpu_map[i], affinity)]
        if affinity.mode == "require":
            if not eligible:
                return []
            return [
                (eligible, f"restricted to compute capability >= {affinity.min_compute_capability}")
            ]

        pools: list[tuple[list[int], str | None]] = []
        if eligible:
            pools.append((eligible, None))
        if len(eligible) != len(order):
            pools.append(
                (
                    order,
                    f"placed on a GPU below compute capability "
                    f"{affinity.min_compute_capability}: this quantization runs but "
                    f"without native acceleration, so expect notably slower prompt "
                    f"processing",
                )
            )
        return pools

    # -- the main entry point ---------------------------------------------

    def plan_load(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None = None,
        kv_cache_type: KvCacheType | None = None,
        kv_cache_type_v: KvCacheType | None = None,
        parallel: int | None = None,
        loaded: Sequence[InstanceInfo] = (),
        draft: ModelRecord | None = None,
        adapters: Sequence[AdapterRecord] = (),
        allow_evict: bool | None = None,
        reload_of: str | None = None,
    ) -> PlanResult:
        """Decide where (and whether) a model can be loaded.

        Returns a :class:`LoadPlan` on success or a :class:`LoadRejected`
        carrying the numbers and concrete next steps. Because there is no CPU
        fallback, a rejection is terminal for the requested settings -- so it
        must always be actionable.

        When the model is a *thinking* model and nobody asked for a specific
        context, a larger default is tried first (see
        :attr:`THINKING_DEFAULT_CTX`) and the ordinary default is used if it
        does not fit. Preferring a bigger window must never turn a load that
        would have worked into a rejection.

        ``reload_of`` names a model whose running instance is about to be
        replaced by this plan (a forced reload, D30). The plan is made *as if
        that child were already gone*: its planned footprint is credited back
        to the GPUs it sits on, it is neither an eviction candidate nor a
        pinned obstacle, and its pid still counts as ours in the VRAM-holder
        attribution. The returned plan lists it first in
        ``evict_model_ids`` so the caller stops it on the way to spawning --
        which means a refusal leaves the resident child running, instead of
        the old unload-then-plan order that left the model unloaded whenever
        the reload was refused.
        """
        if reload_of is None:
            return self._plan_load(
                record,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                kv_cache_type_v=kv_cache_type_v,
                parallel=parallel,
                loaded=loaded,
                draft=draft,
                adapters=adapters,
                allow_evict=allow_evict,
            )
        resident = next((i for i in loaded if i.model_id == reload_of), None)
        others = [i for i in loaded if i.model_id != reload_of]
        gpus_view = self._gpus_as_if_gone(resident) if resident is not None else None
        own = [resident.pid] if resident is not None and resident.pid is not None else []
        result = self._plan_load(
            record,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            kv_cache_type_v=kv_cache_type_v,
            parallel=parallel,
            loaded=others,
            draft=draft,
            adapters=adapters,
            allow_evict=allow_evict,
            gpus=gpus_view,
            extra_own_pids=own,
        )
        if isinstance(result, LoadPlan) and resident is not None:
            if reload_of not in result.evict_model_ids:
                result.evict_model_ids.insert(0, reload_of)
            credited = sum(self.instance_footprint(resident).values())
            result.notes.append(
                f"forced reload: planned as if the running instance of {reload_of} "
                f"were already unloaded ({round(credited / MB)} MB credited back)"
            )
        return result

    def _gpus_as_if_gone(self, resident: InstanceInfo) -> list[GpuInfo]:
        """The live GPU list with ``resident``'s planned footprint credited as free.

        The footprint is the instance's own plan (:meth:`instance_footprint`),
        the same figure the eviction ladder credits for a victim; the truth is
        whatever the driver releases when the child exits, and a plan made on
        the estimate meets the same one-retry OOM path a post-eviction plan does.
        """
        gpus = list(self.probe.list_gpus())
        footprint = self.instance_footprint(resident)
        if not footprint:
            return gpus
        view: list[GpuInfo] = []
        for gpu in gpus:
            credit = int(footprint.get(gpu.index, 0))
            if credit <= 0:
                view.append(gpu)
                continue
            view.append(
                gpu.model_copy(
                    update={
                        "free_bytes": min(gpu.total_bytes, gpu.free_bytes + credit),
                        "used_bytes": max(0, gpu.used_bytes - credit),
                    }
                )
            )
        return view

    def _plan_load(
        self,
        record: ModelRecord,
        *,
        ctx_size: int | None,
        kv_cache_type: KvCacheType | None,
        kv_cache_type_v: KvCacheType | None,
        parallel: int | None,
        loaded: Sequence[InstanceInfo],
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        allow_evict: bool | None,
        gpus: Sequence[GpuInfo] | None = None,
        extra_own_pids: Sequence[int] = (),
    ) -> PlanResult:
        """:meth:`plan_load` proper; ``gpus``/``extra_own_pids`` are the reload view."""
        settings = record.settings
        defaults = self.config.models

        requested_ctx = ctx_size or settings.ctx_size
        slots, auto_parallel = self._resolve_parallel(record, parallel)
        kv_k: KvCacheType = (
            kv_cache_type or settings.kv_cache_type or defaults.default_kv_cache_type
        )
        kv_v: KvCacheType = kv_cache_type_v or settings.kv_cache_type_v or kv_k
        evict_allowed = (
            allow_evict
            if allow_evict is not None
            else self.config.planner.on_insufficient == "evict"
        )

        ladder = self._context_ladder(record, requested_ctx)
        aim = ladder[0]
        floor = ladder[-1]
        thinking = bool(record.capabilities.thinking)
        # Context first, then KV quality. At each rung the best-quality cache
        # that still fits is used, so a model only pays for quantization when
        # the window it wants genuinely cannot be afforded any other way.
        kv_options = self._kv_options(kv_k, kv_v)
        tried: list[str] = []

        # Pass 1: the whole ladder with eviction DISABLED, floor included. A
        # roomier context is a nicety and must never be the reason another
        # model gets unloaded (D14) -- so nothing here may evict, not even the
        # floor. Aiming high therefore cannot turn a load that would have
        # worked into a rejection.
        for ctx in ladder:
            for cand_k, cand_v in kv_options:
                tried.append(f"{ctx}/{cand_k}")
                attempt = self._plan_at_ctx(
                    record,
                    ctx=ctx,
                    slots=slots,
                    kv_k=cand_k,
                    kv_v=cand_v,
                    loaded=loaded,
                    draft=draft,
                    adapters=adapters,
                    evict_allowed=False,
                    auto_parallel=auto_parallel,
                    terminal=False,
                    gpus=gpus,
                    extra_own_pids=extra_own_pids,
                    extra_notes=self._rung_notes(
                        ctx, aim, floor, thinking=thinking, kv_k=cand_k, kv_options=kv_options
                    ),
                )
                if isinstance(attempt, LoadPlan):
                    self._log_plan(record, attempt, tried)
                    return attempt

        if not evict_allowed:
            return self._terminal_rejection(
                record,
                ctx=floor,
                slots=slots,
                kv_k=kv_options[0][0],
                kv_v=kv_options[0][1],
                loaded=loaded,
                draft=draft,
                adapters=adapters,
                evict_allowed=False,
                auto_parallel=auto_parallel,
                aim=aim,
                floor=floor,
                tried=tried,
                gpus=gpus,
                extra_own_pids=extra_own_pids,
            )

        # Pass 2 (DECISIONS.md D16): even the floor does not fit, so eviction is
        # now DECIDED -- the choice is no longer "evict or not" but "having
        # evicted, what context do we get?". Re-walk the same ladder against
        # available + reclaimable and take the highest rung that fits. Loading
        # at the 8192 floor after freeing 19 GB, when 65536 would have fitted
        # for the identical cost, is the defect this pass exists to fix.
        reclaimable = self._reclaimable_bytes(loaded)
        for ctx in ladder:
            for cand_k, cand_v in kv_options:
                tried.append(f"{ctx}/{cand_k}+evict")
                attempt = self._plan_at_ctx(
                    record,
                    ctx=ctx,
                    slots=slots,
                    kv_k=cand_k,
                    kv_v=cand_v,
                    loaded=loaded,
                    draft=draft,
                    adapters=adapters,
                    evict_allowed=True,
                    auto_parallel=auto_parallel,
                    terminal=False,
                    gpus=gpus,
                    extra_own_pids=extra_own_pids,
                    extra_notes=self._rung_notes(
                        ctx, aim, floor, thinking=thinking, kv_k=cand_k, kv_options=kv_options
                    ),
                )
                if isinstance(attempt, LoadPlan):
                    victims = ", ".join(attempt.evict_model_ids) or "nothing"
                    freed_mb = round(reclaimable / MB)
                    attempt.notes.append(
                        f"re-planned after eviction: freeing {victims} released "
                        f"{freed_mb} MB, which reaches {attempt.ctx_size} tokens "
                        f"rather than the {floor} floor"
                    )
                    log.info(
                        "re-planned after eviction",
                        model_id=record.id,
                        evicting=attempt.evict_model_ids,
                        frees_mb=freed_mb,
                        ctx=attempt.ctx_size,
                        kv=attempt.kv_cache_type,
                        parallel=attempt.parallel,
                        floor_ctx=floor,
                        devices=attempt.devices,
                        detail=(
                            f"evicting {victims} frees {freed_mb} MB -> re-planned "
                            f"ctx={attempt.ctx_size} kv={attempt.kv_cache_type}"
                        ),
                    )
                    self._log_plan(record, attempt, tried)
                    return attempt

        return self._terminal_rejection(
            record,
            ctx=floor,
            slots=slots,
            kv_k=kv_options[0][0],
            kv_v=kv_options[0][1],
            loaded=loaded,
            draft=draft,
            adapters=adapters,
            evict_allowed=True,
            auto_parallel=auto_parallel,
            aim=aim,
            floor=floor,
            tried=tried,
            gpus=gpus,
            extra_own_pids=extra_own_pids,
        )

    # -- plan_load helpers -------------------------------------------------

    def _resolve_parallel(self, record: ModelRecord, parallel: int | None) -> tuple[int, bool]:
        """``(slots, auto)`` for this load.

        An explicit slot count from anywhere -- the request, the per-model
        settings, or an integer ``models.default_parallel`` -- is used verbatim
        and switches the estimator off entirely (the D14 "explicit value is
        honoured" invariant). Only ``models.default_parallel: auto`` hands the
        decision to :meth:`Planner.size_slots`, and even then the fit is decided
        at the requested count *before* any slot is added and the walk floors at
        one slot -- so auto can never reject a load that a single slot would
        have allowed.
        """
        explicit = parallel or record.settings.parallel
        if explicit:
            return int(explicit), False
        configured = self.config.models.default_parallel
        if isinstance(configured, str):  # "auto"
            return 1, True
        return max(1, int(configured)), False

    def _parallel_cap(self, record: ModelRecord) -> int:
        per_model = record.settings.max_parallel_cap
        if per_model is not None and per_model >= 1:
            return min(MAX_PARALLEL_CAP, int(per_model))
        return MAX_PARALLEL_CAP

    def _rung_notes(
        self,
        ctx: int,
        aim: int,
        floor: int,
        *,
        thinking: bool,
        kv_k: KvCacheType,
        kv_options: Sequence[tuple[KvCacheType, KvCacheType]],
    ) -> list[str]:
        notes: list[str] = []
        if ctx != floor:
            notes.append(self._ctx_note(ctx, aim, floor, thinking=thinking))
        elif aim > floor:
            notes.append(
                f"wanted up to {aim} tokens of context but only {floor} fits in the "
                f"VRAM available right now"
            )
        if len(kv_options) > 1 and kv_k != KV_QUALITY_LADDER[0][0]:
            notes.append(
                f"KV cache set to {kv_k} automatically: the best-quality cache "
                f"that reaches {ctx} tokens here."
            )
        return notes

    def _reclaimable_bytes(self, loaded: Sequence[InstanceInfo]) -> int:
        """VRAM the planner is allowed to take back by evicting idle models."""
        total = 0
        for instance in self._evictable(loaded):
            total += sum(self.instance_footprint(instance).values())
        return total

    def _terminal_rejection(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        loaded: Sequence[InstanceInfo],
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        evict_allowed: bool,
        auto_parallel: bool,
        aim: int,
        floor: int,
        tried: Sequence[str],
        gpus: Sequence[GpuInfo] | None = None,
        extra_own_pids: Sequence[int] = (),
    ) -> PlanResult:
        """Recompute the floor rung as the *terminal* refusal, and log it.

        The ladder walk rejects a dozen rungs on the way down; those are
        working notes, not answers, so they are computed cheaply and logged at
        DEBUG. Exactly one refusal reaches the caller, and that one pays for
        the full diagnosis -- the VRAM-holder enumeration in particular, which
        costs an NVML process walk per call and is the actual answer to "why
        did this stop working" on a shared box.
        """
        result = self._plan_at_ctx(
            record,
            ctx=ctx,
            slots=slots,
            kv_k=kv_k,
            kv_v=kv_v,
            loaded=loaded,
            draft=draft,
            adapters=adapters,
            evict_allowed=evict_allowed,
            auto_parallel=auto_parallel,
            terminal=True,
            gpus=gpus,
            extra_own_pids=extra_own_pids,
            extra_notes=(
                [
                    f"wanted up to {aim} tokens of context but not even the {floor} "
                    f"floor fits in the VRAM available right now"
                ]
                if aim > floor
                else []
            ),
        )
        if isinstance(result, LoadPlan):  # pragma: no cover - VRAM freed mid-walk
            # Something released memory between the ladder walk and this
            # recomputation. Take it: refusing a load that now fits, because an
            # earlier arithmetic pass said otherwise, would be a lie about the
            # present state of the machine.
            self._log_plan(record, result, tried)
            return result
        emit = log.info if self._log_plans else log.debug
        emit(
            "load rejected",
            model_id=record.id,
            required_mb=round(result.required_bytes / MB),
            available_mb=round(result.available_bytes / MB),
            ctx=ctx,
            rungs_tried=len(tried),
            rungs=list(tried),
            evict_allowed=evict_allowed,
            breakdown=result.estimate.breakdown_mb(),
        )
        return result

    def _log_plan(self, record: ModelRecord, plan: LoadPlan, tried: Sequence[str]) -> None:
        """One INFO line per plan: what was tried, and what was chosen.

        Replaces the per-rung ``load rejected`` spam (fifteen INFO lines for one
        ordinary ladder walk, none of them the answer). The rejected rungs are
        still in the record -- as one ``rungs`` field on the line that matters.
        """
        emit = log.info if self._log_plans else log.debug
        emit(
            "load planned",
            model_id=record.id,
            chosen=f"ctx={plan.ctx_size} kv={plan.kv_cache_type} parallel={plan.parallel}",
            devices=plan.devices,
            rungs_tried=len(tried),
            rungs=list(tried),
            max_parallel=plan.max_parallel,
            parallel_limited_by=plan.parallel_limited_by,
            evicting=plan.evict_model_ids,
            estimate_mb=round(plan.estimate.total_bytes / MB),
        )

    def _kv_options(
        self, kv_k: KvCacheType, kv_v: KvCacheType
    ) -> list[tuple[KvCacheType, KvCacheType]]:
        """KV cache types to try, best quality first.

        An explicit type is used verbatim -- one option, no substitution. Only
        ``"auto"`` fans out, and it fans out over
        :data:`~studioforge.core.kv_sensitivity.KV_QUALITY_LADDER`::

            f16/f16  ->  q8_0/q8_0  ->  q8_0 K + q4_0 V

        The symmetric ``q4_0/q4_0`` rung this used to end on is **gone** (D36).
        K and V are not equally sensitive: a q4_0 **K** cache alone drops
        Qwen2.5-7B to 11.7% token agreement with its f16 self, while a q4_0
        **V** cache alone is nearly free (llama.cpp #23470). Fanning out over
        matched pairs therefore skipped the one useful cheap rung and offered
        the one rung that ruins the model -- and because a q4_0 cache reaches
        the biggest window, that was the rung the catalog then recommended. A
        q4_0 K cache is still reachable by setting it explicitly, which this
        method honours verbatim like every other explicit value.
        """
        if kv_k != "auto" and kv_v != "auto":
            return [(kv_k, kv_v)]
        return list(KV_QUALITY_LADDER)

    def _ctx_note(self, ctx: int, aim: int, floor: int, *, thinking: bool) -> str:
        why = (
            "this is a thinking model and reasons before answering, so a small "
            "window truncates the chain of thought and the answer with it"
            if thinking
            else "agent workloads carry long tool transcripts"
        )
        return (
            f"context set to {ctx} rather than the {floor} floor because {why}. "
            f"The aim is {aim}; the largest size that fits in VRAM is used. "
            f"Set a per-model ctx_size to pin an exact value."
        )

    def _context_ladder(self, record: ModelRecord, requested_ctx: int | None) -> list[int]:
        """Candidate contexts, largest first, ending at the floor.

        An explicit request is honoured exactly -- asking for 4096 must give
        4096, not a silent upgrade. Otherwise the ladder starts at
        ``models.target_ctx`` (clamped to the model's trained window, because
        going past it needs RoPE scaling and quietly degrades quality) and
        halves down to ``models.default_ctx``.
        """
        defaults = self.config.models
        floor = max(1, int(defaults.default_ctx))
        trained_window = record.meta.n_ctx_train if record.meta else 0
        if trained_window and trained_window > 0:
            # Hardware tuning raises default_ctx on a big rig, which could push
            # the FLOOR past a small model's trained window and launch it
            # beyond what it was trained for without anyone asking.
            floor = min(floor, int(trained_window))
        if requested_ctx:
            return [int(requested_ctx)]

        aim = int(getattr(defaults, "target_ctx", 0) or 0)
        if record.capabilities.thinking:
            aim = max(aim, int(getattr(defaults, "thinking_default_ctx", 0) or 0))
        trained = record.meta.n_ctx_train if record.meta else 0
        if trained > 0:
            aim = min(aim, trained) if aim else trained
        if aim <= floor:
            return [floor]

        rungs: list[int] = []
        ctx = aim
        while ctx > floor:
            rungs.append(ctx)
            ctx //= 2
        rungs.append(floor)
        return rungs

    def _thinking_ctx(self, record: ModelRecord) -> int | None:
        """The elevated default context for a thinking model, or ``None``.

        ``None`` whenever the boost would be pointless or wrong: a non-thinking
        model, a global default that is already at least as large, or a model
        whose trained window is smaller than the boost (in which case the
        clamped value is used, and only if it still beats the default).
        """
        if not record.capabilities.thinking:
            return None
        wanted = int(
            getattr(self.config.models, "thinking_default_ctx", THINKING_DEFAULT_CTX)
            or THINKING_DEFAULT_CTX
        )
        trained = record.meta.n_ctx_train if record.meta else 0
        if trained > 0:
            wanted = min(wanted, trained)
        if wanted <= self.config.models.default_ctx:
            return None
        return wanted

    def _plan_at_ctx(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        loaded: Sequence[InstanceInfo],
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        evict_allowed: bool,
        extra_notes: Sequence[str] = (),
        auto_parallel: bool = False,
        terminal: bool = True,
        gpus: Sequence[GpuInfo] | None = None,
        extra_own_pids: Sequence[int] = (),
    ) -> PlanResult:
        """Plan a load at one specific context size.

        ``gpus`` is the live probe unless a caller hands in a view (the forced
        reload credits the resident child's footprint back, see
        :meth:`plan_load`); ``extra_own_pids`` are pids that count as ours in
        the holder attribution beyond those in ``loaded``.
        """
        settings = record.settings

        gpus = list(gpus) if gpus is not None else self.probe.list_gpus()
        if not gpus:
            estimate = self._safe_estimate(
                record, ctx, slots, kv_k, kv_v, 1, draft, settings.draft_ctx_size, adapters
            )
            return LoadRejected(
                model_id=record.id,
                reason=(
                    "No CUDA GPUs were detected. This server is GPU-only: it never "
                    "falls back to CPU inference."
                ),
                estimate=estimate,
                required_bytes=estimate.total_bytes,
                available_bytes=0,
                suggestions=[
                    "check that the NVIDIA driver is installed and nvidia-smi works",
                    "confirm the engine build matches your GPU architecture",
                ],
            )

        notes: list[str] = list(extra_notes)
        notes.extend(self._trained_context_notes(record, ctx))
        notes.extend(self._context_budget_notes(record, ctx, slots))

        # Honour an explicit device override exactly -- the user asked for it.
        if settings.device_override:
            forced_notes = [*notes, "device placement forced by per-model device_override"]
            clashing = sorted(
                set(settings.device_override) & set(self.config.planner.excluded_devices)
            )
            if clashing:
                # Explicit beats policy, loudly. Silently honouring the override
                # would hide a real contradiction (someone reserved CUDA3 for
                # ComfyUI and then pinned a model to it); silently honouring the
                # exclusion would ignore what the user actually asked for.
                log.warning(
                    "device_override overrides planner.excluded_devices",
                    model_id=record.id,
                    devices=list(settings.device_override),
                    excluded=clashing,
                    detail=(
                        "an explicit per-model device_override outranks the exclusion "
                        "list; clear one of the two if this was not intended"
                    ),
                )
                forced_notes.append(
                    f"device_override uses CUDA {clashing}, which planner."
                    f"excluded_devices reserves for other software: the explicit "
                    f"override wins"
                )
            return self._plan_on_devices(
                record,
                devices=list(settings.device_override),
                gpus=gpus,
                ctx=ctx,
                slots=slots,
                kv_k=kv_k,
                kv_v=kv_v,
                draft=draft,
                adapters=adapters,
                loaded=loaded,
                evict_allowed=evict_allowed,
                notes=forced_notes,
                forced=True,
                auto_parallel=auto_parallel,
                terminal=terminal,
                extra_own_pids=extra_own_pids,
            )

        order = self._candidate_order(gpus)
        gpu_map = {gpu.index: gpu for gpu in gpus}
        affinity = self._affinity_for(record)
        pools = self._device_pools(order, gpu_map, affinity)

        for pool, pool_note in pools:
            # Policy: the cheapest placement that fits. A single GPU avoids
            # tensor-split overhead and all cross-GPU traffic, so every
            # single-GPU option is tried (best card first) before any split.
            if self.config.planner.prefer_single_gpu:
                for index in pool:
                    result = self._try_devices(
                        record,
                        [index],
                        gpu_map,
                        ctx,
                        slots,
                        kv_k,
                        kv_v,
                        draft,
                        settings.draft_ctx_size,
                        adapters,
                        extra_free={},
                        auto_parallel=auto_parallel,
                    )
                    if result is None:
                        continue
                    # ...unless spreading buys real concurrency. A split costs
                    # cross-device traffic; doubling the slot count is worth
                    # that, a marginal gain is not.
                    wider = self._wider_split_for_parallel(
                        record,
                        result,
                        pool,
                        gpu_map,
                        ctx,
                        slots,
                        kv_k,
                        kv_v,
                        draft,
                        adapters,
                        auto_parallel=auto_parallel,
                    )
                    chosen = wider if wider is not None else result
                    chosen.notes.extend(notes)
                    if pool_note:
                        chosen.notes.append(pool_note)
                    return chosen

            # Then multi-GPU splits, narrowest first: two cards beat four
            # because each added device adds a CUDA context and more
            # cross-device traffic.
            for width in range(2, len(pool) + 1):
                for combo in _combinations(pool, width):
                    result = self._try_devices(
                        record,
                        list(combo),
                        gpu_map,
                        ctx,
                        slots,
                        kv_k,
                        kv_v,
                        draft,
                        settings.draft_ctx_size,
                        adapters,
                        extra_free={},
                        auto_parallel=auto_parallel,
                    )
                    if result is not None:
                        result.notes.extend(notes)
                        result.notes.append(
                            f"split across {width} GPUs: did not fit on any single device"
                        )
                        if pool_note:
                            result.notes.append(pool_note)
                        return result

        # Nothing fits as-is. Try again with LRU unpinned models evicted.
        if evict_allowed:
            evictable = self._evictable(loaded)
            if evictable:
                freed: dict[int, int] = {}
                evicted: list[str] = []
                for instance in evictable:
                    evicted.append(instance.model_id)
                    for dev, amount in self.instance_footprint(instance).items():
                        freed[dev] = freed.get(dev, 0) + amount
                    for pool, pool_note in pools:
                        for devices in (
                            *([idx] for idx in pool),
                            *(
                                list(combo)
                                for width in range(2, len(pool) + 1)
                                for combo in _combinations(pool, width)
                            ),
                        ):
                            result = self._try_devices(
                                record,
                                devices,
                                gpu_map,
                                ctx,
                                slots,
                                kv_k,
                                kv_v,
                                draft,
                                settings.draft_ctx_size,
                                adapters,
                                extra_free=freed,
                                auto_parallel=auto_parallel,
                            )
                            if result is not None:
                                result.evict_model_ids = list(evicted)
                                result.notes.extend(notes)
                                result.notes.append(
                                    "evicting least-recently-used unpinned models: "
                                    + ", ".join(evicted)
                                )
                                if pool_note:
                                    result.notes.append(pool_note)
                                return result

        return self._reject(
            record,
            gpus=gpus,
            ctx=ctx,
            slots=slots,
            kv_k=kv_k,
            kv_v=kv_v,
            draft=draft,
            adapters=adapters,
            loaded=loaded,
            evict_allowed=evict_allowed,
            notes=notes,
            affinity=affinity,
            terminal=terminal,
            extra_own_pids=extra_own_pids,
        )

    def _wider_split_for_parallel(
        self,
        record: ModelRecord,
        single: LoadPlan,
        pool: Sequence[int],
        gpu_map: dict[int, GpuInfo],
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        *,
        auto_parallel: bool,
    ) -> LoadPlan | None:
        """A split placement worth taking over ``single``, or ``None``.

        ``prefer_single_gpu`` is right almost always: a split pays cross-device
        traffic on every token and buys nothing back. It stops being right in
        exactly one case -- a model whose KV cache is so large that one card
        serves a *single* conversation, where a second card is the difference
        between one client and four. Three conditions, all necessary:

        * the single-GPU placement is genuinely starved (one slot). If one card
          already sustains two, the split is paying a real per-token cost for a
          marginal gain.
        * the split at least doubles the slot count.
        * every added device is at least as capable as the one the single-GPU
          placement chose. A split runs at its slowest member's pace, so
          dragging a 5090-resident model onto a 3090 to buy slots trades
          latency everyone feels for throughput only a batch would notice.

        Only consulted when the slot count is automatic. With an explicit
        ``parallel`` there is nothing to buy: the count is fixed, so the
        cheapest placement that fits it stays the right answer.
        """
        if not auto_parallel or single.parallel != 1:
            return None
        base_cc = gpu_map[single.devices[0]].compute_capability or (0, 0)
        for width in range(2, len(pool) + 1):
            for combo in _combinations(pool, width):
                if any((gpu_map[d].compute_capability or (0, 0)) < base_cc for d in combo):
                    continue
                candidate = self._try_devices(
                    record,
                    list(combo),
                    gpu_map,
                    ctx,
                    slots,
                    kv_k,
                    kv_v,
                    draft,
                    record.settings.draft_ctx_size,
                    adapters,
                    extra_free={},
                    auto_parallel=True,
                )
                if candidate is None or candidate.parallel < 2:
                    continue
                candidate.notes.append(
                    f"split across {width} GPUs rather than a single card: it "
                    f"sustains {candidate.parallel} concurrent slots against "
                    f"{single.parallel} on CUDA{single.devices[0]}"
                )
                return candidate
        return None

    # -- placement helpers ------------------------------------------------

    def _safe_estimate(
        self,
        record: ModelRecord,
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        n_devices: int,
        draft: ModelRecord | None,
        draft_ctx: int | None,
        adapters: Sequence[AdapterRecord],
    ) -> VramEstimate:
        try:
            return self.estimate(
                record,
                ctx_size=ctx,
                parallel=slots,
                kv_cache_type=kv_k,
                kv_cache_type_v=kv_v,
                n_devices=n_devices,
                draft=draft,
                draft_ctx_size=draft_ctx,
                adapters=adapters,
            )
        except PlannerError:
            # No metadata: fall back to the on-disk size so the rejection still
            # carries a number rather than zeros.
            return VramEstimate(weights_bytes=int(record.size_bytes))

    def _try_devices(
        self,
        record: ModelRecord,
        devices: list[int],
        gpu_map: dict[int, GpuInfo],
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        draft: ModelRecord | None,
        draft_ctx: int | None,
        adapters: Sequence[AdapterRecord],
        *,
        extra_free: dict[int, int],
        forced: bool = False,
        auto_parallel: bool = False,
    ) -> LoadPlan | None:
        """Return a plan if the model fits on exactly ``devices``, else None."""
        gpus = [gpu_map[d] for d in devices if d in gpu_map]
        if len(gpus) != len(devices):
            return None

        estimate = self._safe_estimate(
            record, ctx, slots, kv_k, kv_v, len(devices), draft, draft_ctx, adapters
        )

        capacities = {
            gpu.index: self.usable_bytes(gpu, forced=forced) + extra_free.get(gpu.index, 0)
            for gpu in gpus
        }
        total_capacity = sum(capacities.values())
        # Fit is decided at the requested slot count FIRST. Sizing concurrency
        # before knowing the model fits at all would let the estimator turn a
        # working single-slot load into a rejection, which auto must never do.
        if total_capacity < estimate.total_bytes:
            return None

        # The *effective* cost: what a token of context really adds on this
        # model's layer geometry, not the uniform figure. It reaches clients as
        # LoadPlan.kv_bytes_per_token and is what the catalog divides by.
        per_token = (
            effective_kv_bytes_per_token(record.meta, kv_k=kv_k, kv_v=kv_v, ctx_per_slot=ctx)
            if record.meta
            else 0
        )
        chosen_slots = slots
        bound = "explicit"
        max_parallel = slots
        if auto_parallel:
            estimate, chosen_slots, max_parallel, bound = self._size_parallel(
                record,
                ctx=ctx,
                kv_k=kv_k,
                kv_v=kv_v,
                devices=devices,
                draft=draft,
                draft_ctx=draft_ctx,
                adapters=adapters,
                base_estimate=estimate,
                total_capacity=total_capacity,
            )

        total_needed = estimate.total_bytes
        shared: dict[str, Any] = {
            "model_id": record.id,
            "ctx_size": ctx,
            "parallel": chosen_slots,
            "kv_cache_type": kv_k,
            "kv_cache_type_v": kv_v,
            "flash_attn": self._flash_attn_for(record),
            "estimate": estimate,
            "max_parallel": max_parallel,
            "parallel_limited_by": bound,
            "ctx_per_slot": ctx,
            "kv_bytes_per_token": per_token,
        }

        if len(devices) == 1:
            index = devices[0]
            return LoadPlan(
                devices=[index],
                tensor_split=None,
                split_mode="none",
                main_gpu=index,
                per_gpu_bytes={index: total_needed},
                notes=[f"single-GPU placement on CUDA{index} (no tensor-split overhead)"],
                **shared,
            )

        # Proportional split by usable capacity. Every device must actually be
        # able to hold its share, otherwise the split is not viable even though
        # the totals add up.
        split = [capacities[d] / total_capacity for d in devices]
        per_gpu = {d: int(total_needed * frac) for d, frac in zip(devices, split, strict=True)}
        for dev, want in per_gpu.items():
            if want > capacities[dev]:
                return None

        return LoadPlan(
            devices=list(devices),
            tensor_split=[round(f, 4) for f in split],
            split_mode=self._split_mode_for(record),
            main_gpu=devices[0],
            per_gpu_bytes=per_gpu,
            notes=[],
            **shared,
        )

    def fits_on(
        self,
        record: ModelRecord,
        *,
        devices: Sequence[int],
        ctx_size: int,
        parallel: int = 1,
        kv_cache_type: KvCacheType = "f16",
        kv_cache_type_v: KvCacheType | None = None,
        forced: bool = False,
    ) -> VramEstimate | None:
        """ "Would this model fit on exactly these GPUs?" -- the estimate, or None.

        A read-only question, asked by surfaces that are *not* loading anything:
        the pre-download context matrix (:mod:`studioforge.core.hf_meta`) asks it
        a few dozen times per repo to build its tier table.

        It exists so those surfaces cannot re-implement the fit rule and drift
        from it. Everything that decides a placement lives in
        :meth:`_try_devices` -- the per-layer KV geometry, the per-device CUDA
        context charge, and in particular the proportional-split viability check
        that stops a 60 GB model being declared fine on 32+32 when one card can
        only hold its share. A picker that promised a context the loader then
        refuses is worse than one that says nothing, because the user finds out
        after the download.

        No eviction, no auto-parallel, no fallback to another device set: this
        answers about the device set it was handed, at the slot count it was
        handed. ``None`` means "not at these settings", never "never".
        """
        return_plan = self._try_devices(
            record,
            list(devices),
            self._gpu_map(),
            max(1, int(ctx_size)),
            max(1, int(parallel)),
            kv_cache_type,
            kv_cache_type_v or kv_cache_type,
            None,
            None,
            (),
            extra_free={},
            forced=forced,
            auto_parallel=False,
        )
        return return_plan.estimate if return_plan is not None else None

    def max_slots_by_vram(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        n_devices: int,
        capacity_bytes: int,
        cap: int,
        draft: ModelRecord | None = None,
        draft_ctx: int | None = None,
        adapters: Sequence[AdapterRecord] = (),
    ) -> tuple[int, VramEstimate]:
        """Largest slot count in ``[1, cap]`` that really fits, and its estimate.

        The exact VRAM bound, replacing ``kv_budget // (ctx * kv_per_token)``.
        That quotient assumed the KV cache scales linearly with slots, which no
        interesting model obeys: an iSWA model's window layers grow with
        ``n_swa * n_seq_max`` and its global layers not at all, and a hybrid
        model's recurrent state is flat per slot. Walking down from the cap and
        asking :meth:`estimate` -- the same function a load asks -- costs at
        most ``cap`` (8) pure-arithmetic calls and cannot disagree with itself.

        Never returns less than 1. One slot is what a load does today, so this
        can only ever add concurrency (D17), never turn a working load into a
        rejection; the one-slot estimate comes back with it so the caller always
        has a number to report.
        """
        cap = max(1, int(cap))
        floor_estimate = self._safe_estimate(
            record, ctx, 1, kv_k, kv_v, n_devices, draft, draft_ctx, adapters
        )
        for slots in range(cap, 1, -1):
            candidate = self._safe_estimate(
                record, ctx, slots, kv_k, kv_v, n_devices, draft, draft_ctx, adapters
            )
            if candidate.total_bytes <= capacity_bytes:
                return slots, candidate
        return 1, floor_estimate

    def size_slots(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        devices: Sequence[int],
        capacity_bytes: int,
        base_estimate: VramEstimate,
        draft: ModelRecord | None = None,
        draft_ctx: int | None = None,
        adapters: Sequence[AdapterRecord] = (),
    ) -> tuple[VramEstimate, int, int, str]:
        """Pick the slot count for a placement already known to fit as requested.

        Returns ``(estimate_at_slots, slots, max_parallel, bound)`` with bound
        in ``{"vram", "knee", "cap", "unknown"}``. Still the two D17 bounds with
        the smaller winning, but both halves were wrong for half the library:

        * **VRAM** is now :meth:`max_slots_by_vram`, an exact walk down from the
          cap. The analytic quotient divided by a *uniform* per-token cost --
          1.9 MB/token for Gemma-4 31B against an effective ~80 KB -- so every
          row of the catalog said ``max_parallel: 1 (vram)`` with 34 GB free.
        * **Knee** still compares weight traffic against KV traffic per decode
          step, but the KV side is the bytes a slot really reads
          (:func:`kv_read_bytes_per_slot`): global layers plus 1024-token
          windows, not the whole transcript on every layer.

        The knee is evaluated at ``ctx * CTX_FILL_FRACTION`` because slots
        rarely sit at their maximum; assuming they do puts the knee at one or
        two slots for every model on the box.
        """
        meta = record.meta
        n_devices = len(devices) or 1
        per_token = (
            effective_kv_bytes_per_token(meta, kv_k=kv_k, kv_v=kv_v, ctx_per_slot=ctx)
            if meta is not None
            else 0
        )
        if per_token <= 0:
            # No usable geometry. The walk would happily return the cap off a
            # weights-only estimate, which would be an invention rather than an
            # estimate; one slot is what happens today, so say that.
            return base_estimate, 1, 1, "unknown"

        cap = self._parallel_cap(record)
        by_vram, estimate = self.max_slots_by_vram(
            record,
            ctx=ctx,
            kv_k=kv_k,
            kv_v=kv_v,
            n_devices=n_devices,
            capacity_bytes=capacity_bytes,
            cap=cap,
            draft=draft,
            draft_ctx=draft_ctx,
            adapters=adapters,
        )
        fixed = base_estimate.total_bytes - base_estimate.kv_bytes
        wanted, bound = max_parallel_for(
            kv_budget_bytes=max(0, capacity_bytes - fixed),
            kv_per_token=per_token,
            ctx_per_slot=ctx,
            active_weight_bytes=active_weight_bytes(meta, base_estimate.weights_bytes),
            is_moe=is_moe(meta),
            cap=cap,
            kv_read_bytes_per_slot=kv_read_bytes_per_slot(
                meta, kv_k=kv_k, kv_v=kv_v, ctx_fill=int(ctx * CTX_FILL_FRACTION)
            ),
        )
        slots = max(1, min(by_vram, wanted))
        if slots < wanted:
            # The exact walk beat the analytic bound, so VRAM is what actually
            # limits this placement whatever the arithmetic said.
            bound = "vram"
        elif slots != by_vram:
            estimate = self._safe_estimate(
                record, ctx, slots, kv_k, kv_v, n_devices, draft, draft_ctx, adapters
            )
        return estimate, slots, slots, bound

    def _size_parallel(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        devices: Sequence[int],
        draft: ModelRecord | None,
        draft_ctx: int | None,
        adapters: Sequence[AdapterRecord],
        base_estimate: VramEstimate,
        total_capacity: int,
    ) -> tuple[VramEstimate, int, int, str]:
        """Internal alias for :meth:`size_slots`, which is the public surface."""
        return self.size_slots(
            record,
            ctx=ctx,
            kv_k=kv_k,
            kv_v=kv_v,
            devices=devices,
            capacity_bytes=total_capacity,
            base_estimate=base_estimate,
            draft=draft,
            draft_ctx=draft_ctx,
            adapters=adapters,
        )

    def _plan_on_devices(
        self,
        record: ModelRecord,
        *,
        devices: list[int],
        gpus: Sequence[GpuInfo],
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        loaded: Sequence[InstanceInfo],
        evict_allowed: bool,
        notes: list[str],
        forced: bool,
        auto_parallel: bool = False,
        terminal: bool = True,
        extra_own_pids: Sequence[int] = (),
    ) -> PlanResult:
        gpu_map = {gpu.index: gpu for gpu in gpus}
        unknown = [d for d in devices if d not in gpu_map]
        if unknown:
            estimate = self._safe_estimate(
                record, ctx, slots, kv_k, kv_v, len(devices), draft, None, adapters
            )
            return LoadRejected(
                model_id=record.id,
                reason=(
                    f"device_override names GPU(s) {unknown} that do not exist "
                    f"(available: {sorted(gpu_map)})"
                ),
                estimate=estimate,
                required_bytes=estimate.total_bytes,
                available_bytes=0,
                suggestions=["clear the per-model device override to use automatic placement"],
            )

        result = self._try_devices(
            record,
            devices,
            gpu_map,
            ctx,
            slots,
            kv_k,
            kv_v,
            draft,
            record.settings.draft_ctx_size,
            adapters,
            extra_free={},
            forced=forced,
            auto_parallel=auto_parallel,
        )
        if result is not None:
            result.notes.extend(notes)
            return result

        if evict_allowed:
            freed: dict[int, int] = {}
            evicted: list[str] = []
            for instance in self._evictable(loaded):
                evicted.append(instance.model_id)
                for dev, amount in self.instance_footprint(instance).items():
                    freed[dev] = freed.get(dev, 0) + amount
                result = self._try_devices(
                    record,
                    devices,
                    gpu_map,
                    ctx,
                    slots,
                    kv_k,
                    kv_v,
                    draft,
                    record.settings.draft_ctx_size,
                    adapters,
                    extra_free=freed,
                    forced=forced,
                    auto_parallel=auto_parallel,
                )
                if result is not None:
                    result.evict_model_ids = list(evicted)
                    result.notes.extend(notes)
                    return result

        rejection = self._reject(
            record,
            gpus=[gpu_map[d] for d in devices],
            ctx=ctx,
            slots=slots,
            kv_k=kv_k,
            kv_v=kv_v,
            draft=draft,
            adapters=adapters,
            loaded=loaded,
            evict_allowed=evict_allowed,
            notes=notes,
            terminal=terminal,
            forced=forced,
            extra_own_pids=extra_own_pids,
        )
        if forced:
            rejection.suggestions.append(
                "clear the per-model device override so the planner can use other GPUs"
            )
        return rejection

    def _reject(
        self,
        record: ModelRecord,
        *,
        gpus: Sequence[GpuInfo],
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        draft: ModelRecord | None,
        adapters: Sequence[AdapterRecord],
        loaded: Sequence[InstanceInfo],
        evict_allowed: bool,
        notes: list[str],
        affinity: QuantAffinity | None = None,
        terminal: bool = True,
        forced: bool = False,
        extra_own_pids: Sequence[int] = (),
    ) -> LoadRejected:
        estimate = self._safe_estimate(
            record,
            ctx,
            slots,
            kv_k,
            kv_v,
            len(gpus) or 1,
            draft,
            record.settings.draft_ctx_size,
            adapters,
        )
        per_gpu_free = {gpu.index: self.usable_bytes(gpu, forced=forced) for gpu in gpus}
        available = sum(per_gpu_free.values())
        best_single = max(per_gpu_free.values()) if per_gpu_free else 0

        suggestions: list[str] = []
        meta = record.meta

        # 0. Fewer slots at the SAME window. An explicit `parallel` -- usually a
        #    catalog row's load_args, computed against the VRAM free at that
        #    instant -- that no longer fits is more often a slot problem than a
        #    context problem, and the recommendation rule (D22) is that the
        #    window outranks the second slot. So this is offered first, and it
        #    is the exact walk a load runs (max_slots_by_vram), not arithmetic.
        max_parallel: int | None = None
        if slots > 1 and meta is not None:
            # Two placements, the way a load would try them: the best single
            # card, then the whole usable pool (each device costs a CUDA
            # context, so the pool is not simply "more room").
            usable_gpus = [g for g in gpus if per_gpu_free.get(g.index, 0) > 0]
            placements = [(1, best_single)]
            if len(usable_gpus) > 1:
                placements.append((len(usable_gpus), available))
            for n_devices, capacity in placements:
                fewer, at_fewer = self.max_slots_by_vram(
                    record,
                    ctx=ctx,
                    kv_k=kv_k,
                    kv_v=kv_v,
                    n_devices=n_devices,
                    capacity_bytes=capacity,
                    cap=slots - 1,
                    draft=draft,
                    draft_ctx=record.settings.draft_ctx_size,
                    adapters=adapters,
                )
                if at_fewer.total_bytes <= capacity and fewer > (max_parallel or 0):
                    max_parallel = fewer
            if max_parallel is not None:
                suggestions.append(
                    f"reduce parallel from {slots} to {max_parallel}: {max_parallel} slot(s) at "
                    f"{ctx} tokens fit in the VRAM usable right now (a catalog row's parallel is "
                    f"the most that placement sustained when the row was built, not a "
                    f"requirement)"
                )

        # 1. A smaller context that would actually fit. Per-layer geometry
        #    first (the number a load is really charged, D22); the uniform
        #    formula only when the metadata cannot support a per-layer answer.
        max_ctx: int | None = None
        if meta is not None:
            fixed = estimate.total_bytes - estimate.kv_bytes
            budget = available - fixed
            raw = max_ctx_for_budget_geometry(
                meta, budget_bytes=budget, kv_k=kv_k, kv_v=kv_v, parallel=slots
            )
            if raw <= 0 and not kv_layers(meta):
                raw = _round_ctx_down(
                    max_ctx_for_budget(
                        budget_bytes=budget,
                        n_layer=meta.n_layer,
                        n_head_kv=meta.n_head_kv or meta.n_head,
                        head_dim_k=meta.head_dim_k,
                        head_dim_v=meta.head_dim_v,
                        kv_type_k=kv_k,
                        kv_type_v=kv_v,
                        parallel=slots,
                    )
                )
            max_ctx = raw or None
            if max_ctx and max_ctx < ctx:
                suggestions.append(
                    f"reduce context from {ctx} to {max_ctx} tokens (KV cache is "
                    f"{estimate.kv_bytes / (1024**3):.2f} GiB at {ctx})"
                )

        # 2. A cheaper KV cache type, when the KV term is what is hurting.
        if estimate.kv_bytes > estimate.weights_bytes * 0.25 and kv_k == "f16":
            suggestions.append(
                "set KV cache type to q8_0 (roughly halves KV cache VRAM for a small quality cost)"
            )

        # 3. Weights alone too big -> only a smaller quant can help.
        if estimate.weights_bytes > available:
            repo = f"{record.publisher}/{record.repo}" if record.repo else record.id
            suggestions.append(
                f"the weights alone need {estimate.weights_bytes / (1024**3):.2f} GiB -- "
                f"download a smaller quantization of {repo}"
            )

        # 4. Drop the draft model if that is what tipped it over.
        draft_total = estimate.draft_weights_bytes + estimate.draft_kv_bytes
        if draft_total > 0 and (estimate.total_bytes - draft_total) <= available:
            suggestions.append(
                f"remove the draft model '{draft.id if draft else '?'}' -- without it "
                f"({draft_total / (1024**3):.2f} GiB) this load would fit"
            )

        # 5. Pinned models are holding VRAM the planner is not allowed to reclaim.
        pinned = [i.model_id for i in loaded if self._is_pinned(i)]
        if pinned:
            suggestions.append(f"unpin or unload pinned model(s): {', '.join(pinned)}")
        elif not evict_allowed and loaded:
            suggestions.append(
                "planner.on_insufficient is 'reject'; set it to 'evict' to auto-unload idle models"
            )

        reason_parts = [
            f"largest single GPU offers {best_single / (1024**3):.2f} GiB usable "
            f"(headroom {self.config.planner.headroom_fraction:.0%} reserved)"
        ]
        if len(gpus) > 1:
            reason_parts.append(
                f"and {available / (1024**3):.2f} GiB across {len(gpus)} GPUs combined"
            )
        # 6. A 'require' affinity may be what excluded otherwise-usable GPUs.
        if affinity is not None and affinity.mode == "require":
            capable = [g.index for g in gpus if self._eligible_for(g, affinity)]
            if not capable:
                suggestions.insert(
                    0,
                    f"no GPU meets the compute capability {affinity.min_compute_capability} "
                    f"required for {record.quant}; set planner.quant_affinity."
                    f"{record.quant}.mode to 'prefer' to allow slower hardware",
                )

        # 7. Who actually took the VRAM. On a box that also runs ComfyUI or a
        # training job, "not enough VRAM" without attribution is unactionable:
        # the numbers are right and the user still cannot tell what to close.
        # Silently empty when NVML cannot enumerate processes (containers/WSL).
        # Only for the refusal that actually reaches the caller: this walks
        # every process on every GPU, and a ladder walk asks for a dozen
        # rejections it will throw away.
        holders = self._vram_holders(loaded, extra_own_pids) if terminal else []
        foreign = [h for h in holders if not h.is_ours and h.used_bytes > 0]
        if foreign:
            top = sorted(foreign, key=lambda h: -h.used_bytes)[:4]
            suggestions.append(
                "VRAM is held by other processes: " + "; ".join(h.describe() for h in top)
            )

        # 8. Our own policy may be what is holding the memory back.
        excluded = sorted(set(self.config.planner.excluded_devices) & {g.index for g in gpus})
        if excluded and not forced:
            suggestions.append(
                f"planner.excluded_devices reserves CUDA {excluded} for other software; "
                f"remove an index there to let the planner use it"
            )
        reserved = {
            index: mb
            for index, mb in self.config.planner.reserved_mb.items()
            if mb > 0 and index in {g.index for g in gpus}
        }
        if reserved:
            suggestions.append(
                "planner.reserved_mb holds back "
                + ", ".join(f"{mb} MiB on CUDA{index}" for index, mb in sorted(reserved.items()))
            )

        if not suggestions:
            suggestions.append("free VRAM on the box, or choose a smaller model")

        rejected = LoadRejected(
            model_id=record.id,
            reason=", ".join(reason_parts) + ".",
            estimate=estimate,
            required_bytes=estimate.total_bytes,
            available_bytes=available,
            per_gpu_free=per_gpu_free,
            max_ctx_that_fits=max_ctx,
            max_parallel_that_fits=max_parallel,
            suggestions=suggestions,
            notes=list(notes),
            vram_holders=holders,
        )
        # A rung the ladder walked past is not news: fifteen INFO lines per
        # ordinary load, none of which was the outcome, buried the one that
        # was. The terminal refusal is logged by the caller, with the ladder
        # summary attached.
        log.debug(
            "load rejected",
            model_id=record.id,
            required_mb=round(estimate.total_bytes / MB),
            available_mb=round(available / MB),
            ctx=ctx,
            kv=kv_k,
            terminal=terminal,
            breakdown=estimate.breakdown_mb(),
        )
        for note in notes:
            log.debug("planner note", model_id=record.id, note=note)
        return rejected

    def _vram_holders(
        self, loaded: Sequence[InstanceInfo], extra_own_pids: Sequence[int] = ()
    ) -> list[VramProcess]:
        """Processes holding VRAM, with our own children marked as ours.

        The pids of our llama-server children come from the instance table, so
        a rejection can say "2.1 GiB held by python.exe (pid 1234)" about a
        foreign process without blaming the models we loaded on purpose.
        ``extra_own_pids`` covers a child deliberately left out of ``loaded``
        (the resident instance of a forced reload).
        """
        own = [i.pid for i in loaded if i.pid is not None] + list(extra_own_pids)
        return vram_processes(self.probe, own_pids=own)

    # -- eviction ---------------------------------------------------------

    @staticmethod
    def _is_pinned(instance: InstanceInfo) -> bool:
        """A TTL of 0 means pinned; that is the wire representation everywhere."""
        return instance.ttl_s == 0

    def _evictable(self, loaded: Sequence[InstanceInfo]) -> list[InstanceInfo]:
        """Unpinned, idle instances, least-recently-used first.

        Instances with in-flight requests are never evicted -- killing a child
        mid-stream would fail a live request to save a future one.
        """
        candidates = [
            i
            for i in loaded
            if not self._is_pinned(i) and i.active_requests == 0 and i.state == "ready"
        ]
        return sorted(candidates, key=lambda i: i.last_activity_at or i.started_at or 0.0)

    @staticmethod
    def instance_footprint(instance: InstanceInfo) -> dict[int, int]:
        """Per-GPU bytes an instance is believed to hold (from its own plan).

        Public because three surfaces need the same figure and a second
        definition of "how much is this child holding" is how two of them start
        disagreeing: the eviction ladder credits it for a victim, ``reload_of``
        (D30) credits it for the child being replaced, and the catalog credits
        it to a *loaded* model when computing that model's own rows (D36) --
        otherwise every row of a resident model is judged against VRAM the row
        itself would release.
        """
        if instance.plan is None:
            return {}
        return dict(instance.plan.per_gpu_bytes)

    # -- misc -------------------------------------------------------------

    def _flash_attn_for(self, record: ModelRecord) -> FlashAttn:
        return record.settings.flash_attn or self.config.models.default_flash_attn

    def _split_mode_for(self, record: ModelRecord) -> SplitMode:
        return record.settings.split_mode or "layer"

    def _trained_context_notes(self, record: ModelRecord, ctx: int) -> list[str]:
        """Warn when the requested context exceeds the model's trained window.

        Asking for 256K on a 128K model is not an error and is not clamped
        here: with RoPE scaling it is a legitimate (if lossy) thing to want, and
        silently reducing it would be its own kind of lie. What is *not*
        acceptable is for it to be invisible -- the failure mode is a user who
        believes they have 256K, gets whatever the engine decides, and sees only
        degraded output. So the mismatch is stated, with the knob that makes it
        real (``rope_freq_scale``, already a per-model setting).
        """
        meta = record.meta
        if meta is None or meta.n_ctx_train <= 0 or ctx <= meta.n_ctx_train:
            return []
        return [
            f"requested context {ctx} exceeds this model's trained context window of "
            f"{meta.n_ctx_train}: going beyond it needs RoPE scaling (set the per-model "
            f"rope_freq_scale, e.g. {meta.n_ctx_train / ctx:.3f}, or rope_scaling) -- "
            f"without it quality degrades past {meta.n_ctx_train} tokens"
        ]

    def _context_budget_notes(self, record: ModelRecord, ctx: int, slots: int) -> list[str]:
        """Warn when a vision model's context cannot hold the images it will get.

        Images do not add VRAM beyond the KV cache already reserved for
        ``ctx_size`` -- llama.cpp allocates the full KV up front. What they do
        is *consume* that context, hundreds to thousands of tokens each. An
        agent workload that pastes screenshots into an 8k window will fail on
        context, not on memory, so this is a budgeting warning rather than a
        VRAM term.
        """
        notes: list[str] = []
        if not record.capabilities.vision:
            return notes
        per_image = self._image_tokens(record)
        max_images = self.config.gateway.max_images_per_request
        needed = per_image * max_images
        if needed + 2048 > ctx:
            notes.append(
                f"vision model: {per_image} tokens/image x {max_images} max images = "
                f"{needed} tokens, which leaves little of the {ctx}-token context for "
                f"text; consider raising context"
            )
        return notes

    def _image_tokens(self, record: ModelRecord) -> int:
        """Context tokens one image consumes, from mmproj metadata when available."""
        default = self.config.planner.image_tokens_default
        meta = record.meta
        if meta is None:
            return default
        patches = meta.vision_n_patch
        if patches:
            return int(patches)
        if meta.vision_image_size and meta.vision_patch_size:
            side = meta.vision_image_size // max(1, meta.vision_patch_size)
            return max(1, side * side)
        return default

    # -- calibration ------------------------------------------------------

    def calibrate(self, observations: Sequence[dict[str, object]]) -> float | None:
        """Apply a tuned ``compute_overhead_fraction`` from load history.

        Returns the new value, or ``None`` when nothing changed. Mutates the
        live config in memory only -- config.yaml is never rewritten, so a
        calibration that turns out badly is undone by a restart rather than by
        editing a file. Called once at startup; see
        :func:`calibrated_overhead_fraction` for the guards.
        """
        current = float(self.config.planner.compute_overhead_fraction)
        tuned = calibrated_overhead_fraction(observations, current=current)
        if tuned is None:
            return None
        self.config.planner.compute_overhead_fraction = tuned
        log.info(
            "calibrated compute overhead fraction from load history",
            previous=round(current, 4),
            tuned=round(tuned, 4),
            clamp=[OVERHEAD_FRACTION_MIN, OVERHEAD_FRACTION_MAX],
            rows=len(clean_observations(observations)),
        )
        return tuned

    def observe(
        self,
        *,
        model_id: str,
        plan: LoadPlan,
        actual_bytes: int,
        ok: bool = True,
        note: str | None = None,
    ) -> None:
        """Record predicted-vs-actual VRAM so the fudge factor can be tuned.

        Logged at INFO with the ratio, and forwarded to the observation sink
        (the SQLite table) when one is wired in.
        """
        predicted = plan.estimate.total_bytes
        ratio = (actual_bytes / predicted) if predicted else 0.0
        log.info(
            "load observation",
            model_id=model_id,
            predicted_mb=round(predicted / MB),
            actual_mb=round(actual_bytes / MB),
            ratio=round(ratio, 3),
            ctx=plan.ctx_size,
            devices=plan.devices,
        )
        if self._observation_sink is not None:
            self._observation_sink(
                {
                    "model_id": model_id,
                    "ctx_size": plan.ctx_size,
                    "parallel": plan.parallel,
                    "kv_cache_type": plan.kv_cache_type,
                    "devices": ",".join(str(d) for d in plan.devices),
                    "predicted_bytes": predicted,
                    "actual_bytes": actual_bytes,
                    "weights_bytes": plan.estimate.weights_bytes,
                    "ok": 1 if ok else 0,
                    "note": note,
                }
            )


#: Written into ``load_observations.note`` when ``actual_bytes`` is our own
#: child's VRAM, attributed per pid. Rows without it summed whole-device
#: ``used_bytes`` and are contaminated by every other process on the card --
#: see docs/LIMITATIONS.md. Calibration reads only the marked rows.
OBSERVATION_NOTE_PER_PID = "per_pid"

#: Bounds the auto-calibrated ``compute_overhead_fraction`` may move between.
#: Below the floor the compute term stops covering real graph buffers on small
#: models; above the ceiling it eats enough VRAM to refuse loads that fit. A
#: calibration loop with no clamp is a way to make the planner slowly wrong
#: with no one noticing, so the clamp is the point, not a detail.
OVERHEAD_FRACTION_MIN = 0.03
OVERHEAD_FRACTION_MAX = 0.15

#: Clean observations needed before the factor is touched at all. A handful of
#: loads of one model is not a calibration, it is that model.
CALIBRATION_MIN_ROWS = 5


def clean_observations(
    observations: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Observations whose ``actual_bytes`` is really *our child's* VRAM.

    Everything recorded before the per-pid fix summed whole-device usage, so a
    desktop compositor or a ComfyUI run inflated it -- over 540 historical rows
    the median actual/predicted ratio is 2.97 and p90 is 12.0. Feeding those to
    the calibrator would ratchet the overhead fraction to its ceiling and start
    refusing loads that fit. The marker is the only thing separating them.
    """
    return [
        row
        for row in observations
        if str(row.get("note") or "") == OBSERVATION_NOTE_PER_PID and row.get("ok", True)
    ]


def calibrated_overhead_fraction(
    observations: Sequence[dict[str, object]], *, current: float
) -> float | None:
    """A clamped ``compute_overhead_fraction`` to adopt, or ``None``.

    ``None`` means "leave it alone": too little clean data, or the suggestion
    is what is already configured. The clamp is applied after
    :func:`suggest_overhead_fraction`, so a wild suggestion from a handful of
    strange loads costs at most the difference to the nearest bound.
    """
    clean = clean_observations(observations)
    if len(clean) < CALIBRATION_MIN_ROWS:
        return None
    suggested = suggest_overhead_fraction(clean, current=current)
    clamped = min(OVERHEAD_FRACTION_MAX, max(OVERHEAD_FRACTION_MIN, suggested))
    if abs(clamped - current) < 1e-9:
        return None
    return clamped


def suggest_overhead_fraction(
    observations: Sequence[dict[str, object]], *, current: float
) -> float:
    """Suggest a tuned ``compute_overhead_fraction`` from load history.

    Uses the worst (highest) actual/predicted ratio rather than the mean: the
    factor exists to prevent OOM, so it must cover the bad case, not the
    typical one. Returns ``current`` unchanged when there is too little data.
    """
    ratios: list[float] = []
    for row in observations:
        predicted = row.get("predicted_bytes")
        actual = row.get("actual_bytes")
        weights = row.get("weights_bytes")
        if not isinstance(predicted, (int, float)) or not isinstance(actual, (int, float)):
            continue
        if not isinstance(weights, (int, float)) or weights <= 0 or predicted <= 0:
            continue
        shortfall = float(actual) - float(predicted)
        if shortfall <= 0:
            continue
        # Attribute the whole shortfall to the compute term and express it as
        # the extra fraction-of-weights it would have taken to cover it.
        ratios.append(shortfall / float(weights))
    if len(ratios) < 3:
        return current
    needed = current + max(ratios)
    return float(min(0.5, math.ceil(needed * 200) / 200))  # round up to 0.5%


def _combinations(items: Sequence[int], width: int) -> list[tuple[int, ...]]:
    """Ordered combinations, preserving the caller's preference order."""
    return list(itertools.combinations(items, width))
