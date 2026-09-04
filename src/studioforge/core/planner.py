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

import contextlib
import itertools
import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from studioforge.config import Config, FlashAttn, KvCacheType, QuantAffinity, SplitMode
from studioforge.core.gpu import vram_processes
from studioforge.core.kv_sensitivity import KV_QUALITY_LADDER
from studioforge.core.leases import LeaseBook, lease_view
from studioforge.core.priority import PRIORITY_BACKGROUND
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

#: How long a caller is told to wait when a *busy* model is what blocked a
#: load. Long enough that an ordinary agent turn has finished, short enough that
#: a retry loop is not a stall. It is advice attached to a refusal, never a
#: sleep this process takes.
BUSY_RETRY_AFTER_S = 15.0

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

#: Slot count above which the automatic ``-ub`` raise applies. The same "many
#: slots" line the batch size uses (supervisor.BATCH_SIZE_MANY_SLOTS): a
#: single-stream or lightly-concurrent load keeps the engine's 512 and stays
#: byte-identical, while a heavy concurrent prefill gets the larger micro-batch
#: that D38 §5 measured as faster.
UBATCH_MANY_SLOTS_THRESHOLD = 4


def effective_ubatch(
    *,
    settings_ubatch: int | None,
    engine_ubatch: int | None,
    engine_ubatch_many_slots: int | None,
    slots: int,
) -> int | None:
    """The ``-ub`` for a launch, or ``None`` to keep the engine default (512).

    One precedence, shared by the planner (which charges its VRAM, D40) and the
    supervisor (which emits it), so the micro-batch the estimate assumes is
    always the one the child runs with:

    1. an explicit per-model ``ubatch_size`` -- the operator chose it;
    2. an explicit ``engine.ubatch_size`` -- the rig-wide choice;
    3. ``engine.ubatch_many_slots`` when the launch runs more than
       :data:`UBATCH_MANY_SLOTS_THRESHOLD` slots -- the automatic raise;
    4. otherwise ``None`` (the flag is omitted; the engine keeps 512).
    """
    if settings_ubatch is not None:
        return int(settings_ubatch)
    if engine_ubatch is not None:
        return int(engine_ubatch)
    if engine_ubatch_many_slots is not None and slots > UBATCH_MANY_SLOTS_THRESHOLD:
        return int(engine_ubatch_many_slots)
    return None


#: Compute-buffer growth per extra micro-batch token, per unit of ``n_embd``,
#: per device. The activation scratch every device keeps for a forward pass
#: scales with ``n_ubatch * n_embd``; measured on the scratch rig (D40, two
#: Qwen2.5 sizes, ``-ub`` 512 -> 2048): 76-94 bytes per token per ``n_embd``
#: on a single card, 113-126 on each card of a two-way layer split (each holds
#: the full micro-batch plus the cross-device copy buffers). 128 covers the
#: split case with a little room, and errs toward refusing rather than OOM.
UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD = 128

#: Bits per weight assumed for the output projection when the GGUF does not
#: declare ``general.parameter_count``. Quantizers keep the embedding/output
#: tensors at Q6_K/Q8_0 even in a Q4 file, so this is deliberately above the
#: body's typical density.
OUTPUT_LAYER_DEFAULT_BPW = 6.5


def output_layer_bytes(meta: Any) -> int:
    """Weight bytes of the output projection (``lm_head``), or 0 when unknown.

    llama.cpp assigns the output layer -- and the scratch that goes with it --
    to the **last** device of a ``--split-mode layer`` placement, on top of
    that device's proportional share of the blocks (``llama_model::load_tensors``
    puts ``dev_output`` on the device holding layer ``n_layer``, which the
    split arithmetic always makes the last one). Measured here (D40): the last
    device of a 0.5,0.5 split held 104 MiB (0.5B) and 110 MiB (1.5B) more than
    the first, independent of ``-ub``; on the live rig a 27B planned at
    ``0.5079,0.4921`` landed 0.76 GiB *more* on the card the split gave less to.
    The planner charges this figure to the last device so that card is not the
    one that OOMs on a tight fit.

    Derived from ``n_vocab * n_embd`` at the file's average bytes per weight
    when ``general.parameter_count`` is declared, else at
    :data:`OUTPUT_LAYER_DEFAULT_BPW`. A model with tied embeddings still pays
    it: llama.cpp duplicates ``token_embd`` onto the output device.
    """
    n_vocab = int(getattr(meta, "n_vocab", 0) or 0)
    n_embd = int(getattr(meta, "n_embd", 0) or 0)
    if n_vocab <= 0 or n_embd <= 0:
        return 0
    weights = int(getattr(meta, "tensor_bytes", 0) or 0)
    params = int(getattr(meta, "param_count", 0) or 0)
    bpw = (weights * 8 / params) if weights > 0 and params > 0 else OUTPUT_LAYER_DEFAULT_BPW
    return int(n_vocab * n_embd * bpw / 8)


def ubatch_scratch_bytes(meta: Any, *, ubatch: int, n_devices: int = 1) -> int:
    """Extra compute-buffer bytes a micro-batch above the engine's 512 costs.

    Zero at or below :data:`DEFAULT_UBATCH`; grows linearly above it by
    :data:`UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD` per token per ``n_embd``
    **on every device** of the placement, because each device runs its layers
    over the whole micro-batch. This is what makes ``-ub`` safe to raise: the
    planner's compute term used to be calibrated against 512 only (D38).
    """
    n_embd = int(getattr(meta, "n_embd", 0) or 0)
    extra_tokens = max(0, int(ubatch) - DEFAULT_UBATCH)
    if n_embd <= 0 or extra_tokens <= 0:
        return 0
    return extra_tokens * UBATCH_SCRATCH_BYTES_PER_TOKEN_PER_EMBD * n_embd * max(1, n_devices)


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
#: wants more sets ``parallel`` explicitly, which this module never clamps.
#: Not to be confused with the per-model ``settings.max_parallel_cap``: since
#: D48 that one is a hard ceiling the manager enforces, refusing an explicit
#: ``parallel`` above it with a 400 rather than silently clamping it (D14 --
#: explicit is honoured verbatim or refused loudly, never quietly rewritten).
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
        observation_lookup: Callable[..., dict[str, Any] | None] | None = None,
        log_plans: bool = True,
        leases: LeaseBook | None = None,
    ) -> None:
        self.config = config
        self.probe = probe
        self._observation_sink = observation_sink
        #: The other half of the loop the sink opens (D51): "what did this
        #: exact configuration of this model really weigh last time?". Without
        #: it the planner writes observations nothing ever reads back, which is
        #: what shipped until 2026-08-30. ``None`` in every planner that has no
        #: database behind it (tests, the catalog's throwaway instances) and
        #: the whole feature is then inert.
        self._observation_lookup = observation_lookup
        #: Per-``plan_load`` lookup cache; ``None`` outside a planning pass.
        #: See :class:`_ObservationMemo`. Surfaces that ask the planner
        #: hypothetical questions without loading anything -- ``fits_on`` for
        #: the pre-download context matrix, the catalog's per-tier sweep --
        #: run with it unset and stay formula-only: they ask about models that
        #: have often never been loaded here at all, and they ask dozens of
        #: times per repo, which is the one shape this cache cannot absorb.
        self._obs_memo: _ObservationMemo | None = None
        #: Standing GPU leases (D43). A card leased to someone other than the
        #: model being planned is absent from every placement; ``None`` (tests,
        #: the catalog's throwaway planners) means no leases exist.
        self.leases = leases
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
        ubatch: int | None = None,
    ) -> VramEstimate:
        """Project VRAM for one load.

        Terms, all of which are real allocations on the device:

        * weights -- summed tensor bytes across every shard
        * KV cache + recurrent state -- sized per layer on the *total* context
          (``ctx_size * parallel``); see :func:`kv_alloc_bytes`
        * compute/graph buffers -- scratch that scales with model width, plus
          the growth a micro-batch above the engine's 512 costs on every
          device (:func:`ubatch_scratch_bytes`, D40)
        * mmproj weights + the image-encoding buffer, for vision models
        * adapter weights
        * draft model weights + its own KV cache
        * a per-GPU CUDA context/cuBLAS workspace charge

        ``ubatch`` is the ``-ub`` the child will be launched with; omitted, it
        is resolved from the record's settings and ``engine.ubatch_size`` (the
        same precedence the supervisor uses), so every caller that does not
        know about micro-batches still gets the right answer.
        """
        meta = record.meta
        if meta is None:
            raise PlannerError(
                f"model '{record.id}' has no parsed GGUF metadata; cannot plan a load"
            )

        planner_cfg = self.config.planner
        ctx_total = max(1, ctx_size) * max(1, parallel)
        micro_batch = (
            int(ubatch) if ubatch is not None else self.ubatch_for(record, max(1, parallel))
        )

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
            ubatch=micro_batch,
        )

        # Graph/activation buffers track model width and batch size far more than
        # depth. A fraction of the weight size is a crude but stable proxy, with a
        # floor so tiny models still get room for their scratch buffers. Above the
        # engine's default micro-batch the scratch grows with -ub on every device,
        # and that growth is charged explicitly rather than folded into the
        # fraction (which is calibrated against loads at the default).
        compute = max(
            planner_cfg.compute_overhead_floor_mb * MB,
            int(weights * planner_cfg.compute_overhead_fraction),
        ) + ubatch_scratch_bytes(meta, ubatch=micro_batch, n_devices=n_devices)

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
        evict_busy: bool = False,
        source: str | None = None,
        parallel_auto: bool = False,
        priority: int = PRIORITY_BACKGROUND,
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

        Cards leased to someone other than this model (D43) are absent from
        every placement, and a ``device_override`` naming one is refused with
        the lease named. ``parallel_auto`` -- set for a leased model -- treats
        an integer ``models.default_parallel`` as ``auto``: the cards are that
        model's alone, so the estimator sizes its slots.

        ``priority`` is the load's tier (D46). A chat or agent load (1 or 2)
        plans *as if every idle lower-tier resident were already gone* -- the
        same credit ``reload_of`` gets -- so it takes the placement an idle
        box would have given it instead of squeezing in beside background
        models; the lower-tier residents actually sitting on the chosen cards
        land in ``evict_model_ids``, and the ones elsewhere stay loaded. It
        also makes eviction first-class for this load regardless of
        ``planner.on_insufficient`` (an explicit ``allow_evict`` still wins),
        and restricts every eviction ladder to equal-or-lower tiers.
        """
        blocked = self._leased_away(record)
        forced = set(record.settings.device_override or ())
        if forced & blocked:
            return self._leased_rejection(record, sorted(forced & blocked))

        evict_allowed = (
            allow_evict
            if allow_evict is not None
            else (priority < PRIORITY_BACKGROUND or self.config.planner.on_insufficient == "evict")
        )

        resident = (
            next((i for i in loaded if i.model_id == reload_of), None)
            if reload_of is not None
            else None
        )
        others = [i for i in loaded if resident is None or i.model_id != reload_of]

        # D46 preemption: the lower-tier idle residents this load may displace
        # for the fastest placement. Strictly lower tiers only -- an equal-tier
        # resident is displaced by the ordinary ladder when the fit demands it,
        # never just for a nicer split.
        preempt: list[InstanceInfo] = []
        if priority < PRIORITY_BACKGROUND and evict_allowed:
            # Idle-only, always -- even under force. Preemption displaces for a
            # *nicer placement*; D36 allows interrupting a stream only when the
            # fit genuinely demands it, and that path is the eviction ladder
            # below, which still honours ``evict_busy``.
            preempt = self._evictable(others, include_busy=False, for_priority=priority + 1)
        preempt_ids = {i.model_id for i in preempt}

        credited = ([resident] if resident is not None else []) + preempt
        gpus_view = self._gpus_as_if_gone(credited) if credited else None
        if blocked:
            live = gpus_view if gpus_view is not None else list(self.probe.list_gpus())
            gpus_view = [g for g in live if g.index not in blocked]
        allowed = record.settings.allowed_devices
        if allowed is not None and not forced:
            # The soft cousin of device_override: a SET the planner may choose
            # within, so a big model with otherwise-null settings cannot
            # sprawl onto cards its owner never meant for it. An explicit
            # device_override outranks it, the same way it outranks
            # planner.excluded_devices (explicit beats policy).
            allowed_set = {int(d) for d in allowed}
            live = gpus_view if gpus_view is not None else list(self.probe.list_gpus())
            gpus_view = [g for g in live if g.index in allowed_set]
            if not gpus_view:
                return self._note_leases(
                    LoadRejected(
                        model_id=record.id,
                        reason=(
                            f"the model's allowed_devices {sorted(allowed_set)} matches no "
                            f"usable GPU right now (leases or planner.excluded_devices may "
                            f"have taken them)"
                        ),
                        suggestions=[
                            "widen or clear the model's allowed_devices setting, or free "
                            "one of the cards it names"
                        ],
                        # Only when no lease is involved: _note_leases stamps
                        # ``gpu_leased`` over this when one of the cards
                        # allowed_devices names is leased away, because that is
                        # the more specific -- and waitable -- cause.
                        reason_code="allowed_devices_unavailable",
                    ),
                    blocked,
                    allowed=allowed_set,
                )
        own = [i.pid for i in credited if i.pid is not None]
        with self._observation_pass():
            result = self._plan_load(
                record,
                ctx_size=ctx_size,
                kv_cache_type=kv_cache_type,
                kv_cache_type_v=kv_cache_type_v,
                parallel=parallel,
                loaded=[i for i in others if i.model_id not in preempt_ids],
                draft=draft,
                adapters=adapters,
                allow_evict=evict_allowed,
                gpus=gpus_view,
                extra_own_pids=own,
                evict_busy=evict_busy,
                source=source,
                parallel_auto=parallel_auto,
                priority=priority,
            )
        if isinstance(result, LoadPlan):
            if reload_of is not None and resident is not None:
                if reload_of not in result.evict_model_ids:
                    result.evict_model_ids.insert(0, reload_of)
                credit = sum(self.instance_footprint(resident).values())
                result.notes.append(
                    f"forced reload: planned as if the running instance of {reload_of} "
                    f"were already unloaded ({round(credit / MB)} MB credited back)"
                )
            if preempt:
                chosen = set(result.devices)
                displaced = [
                    i
                    for i in preempt
                    if any(
                        dev in chosen and amount > 0
                        for dev, amount in self.instance_footprint(i).items()
                    )
                ]
                for victim in displaced:
                    if victim.model_id not in result.evict_model_ids:
                        result.evict_model_ids.append(victim.model_id)
                if displaced:
                    result.notes.append(
                        f"priority {priority} load: displacing lower-priority "
                        + ", ".join(i.model_id for i in displaced)
                        + f" from CUDA {sorted(chosen)} for the fastest placement; "
                        f"recently active ones are reloaded afterwards where they fit"
                    )
            if allowed is not None and not forced:
                result.notes.append(
                    f"placement restricted to CUDA {sorted({int(d) for d in allowed})} "
                    f"by the model's allowed_devices setting"
                )
            self._grade_placement(result, gpus_view)
        # The cards this load could have used, for the gpu_leased verdict: a
        # device_override IS that set (it outranks allowed_devices and
        # excluded_devices alike, and a clash was already refused above), so a
        # forced placement that simply does not fit is never blamed on a
        # lease standing on cards it was never going to touch.
        if forced:
            usable: set[int] | None = forced
        elif allowed is not None:
            usable = {int(d) for d in allowed}
        else:
            usable = None
        return self._note_leases(result, blocked, allowed=usable)

    def _grade_placement(self, plan: LoadPlan, gpus_view: Sequence[GpuInfo] | None) -> None:
        """Stamp ``placement_tier`` and say so when a split mixes generations.

        Measured on this rig: a 5090+3090 split generates at roughly half the
        pace of a same-generation placement, and nothing used to warn -- the
        load "worked" and the slowness surfaced as a different complaint.
        """
        gpus = list(gpus_view) if gpus_view is not None else list(self.probe.list_gpus())
        caps: dict[int, tuple[int, int]] = {
            g.index: g.compute_capability for g in gpus if g.compute_capability is not None
        }
        majors = {caps[d][0] for d in plan.devices if d in caps}
        if len(plan.devices) > 1 and len(majors) > 1:
            plan.placement_tier = "degraded"
            named = ", ".join(
                f"CUDA{d} sm_{caps[d][0]}{caps[d][1]}" for d in plan.devices if d in caps
            )
            plan.notes.append(
                f"mixed-generation split ({named}): expect roughly the slower "
                f"card's generation speed -- about half, measured on this rig"
            )
        else:
            plan.placement_tier = "optimal"

    # -- GPU leases (D43) --------------------------------------------------

    def _leased_away(self, record: ModelRecord) -> frozenset[int]:
        """Devices a standing lease holds for someone other than ``record``."""
        if self.leases is None:
            return frozenset()
        return frozenset(self.leases.blocked_for(record.id))

    def _gpus_without(self, blocked: frozenset[int]) -> list[GpuInfo] | None:
        """The live GPU list minus leased-away cards; ``None`` (= live) when none are."""
        if not blocked:
            return None
        return [g for g in self.probe.list_gpus() if g.index not in blocked]

    def _lease_lines(self, blocked: frozenset[int]) -> list[str]:
        """One actionable line per lease standing in the way."""
        if self.leases is None or not blocked:
            return []
        lines: list[str] = []
        for lease in self.leases.all():
            held = sorted(set(lease.devices) & blocked)
            if not held:
                continue
            who = lease.holder
            if lease.model_ids:
                who += f" for {', '.join(lease.model_ids)}"
            if lease.reason:
                who += f" ({lease.reason})"
            ends = (
                f"it is released automatically once idle for {int(lease.idle_ttl_s)} s"
                if lease.idle_ttl_s is not None
                else "it stands until released"
            )
            lines.append(
                f"CUDA {held} is leased to {who} -- lease {lease.id}; {ends}, or now via "
                f"DELETE /api/leases/{lease.id} (the release_gpus tool)"
            )
        return lines

    def _blocking_leases(self, blocked: frozenset[int]) -> list[dict[str, Any]]:
        """The ``lease_view`` of every lease holding one of ``blocked``.

        Attached to a refusal so a client backs off against a fact -- an
        ``expires_at`` and a ``retry_after_s`` -- rather than substring-matching
        the word "leased" out of the prose (D53).
        """
        if self.leases is None or not blocked:
            return []
        return [lease_view(lease) for lease in self.leases.all() if set(lease.devices) & blocked]

    def _lease_candidates(self, allowed: set[int] | frozenset[int] | None) -> set[int]:
        """The cards this load could have used if no lease stood.

        Live cards, narrowed by the model's ``allowed_devices`` when it has one
        and by ``planner.excluded_devices`` when it does not (an explicit
        allow-list outranks the policy exclusion, exactly as it does in
        placement). Empty when the probe cannot be asked -- which makes the
        caller stamp nothing, the right answer when we do not know.
        """
        try:
            live = {g.index for g in self.probe.list_gpus()}
        except Exception:  # noqa: BLE001 - a sick probe must not break a refusal
            return set()
        if allowed is not None:
            return live & {int(d) for d in allowed}
        return live - set(self.config.planner.excluded_devices)

    def _note_leases(
        self,
        result: PlanResult,
        blocked: frozenset[int],
        *,
        allowed: set[int] | frozenset[int] | None = None,
    ) -> PlanResult:
        """Say which cards a lease took away, on a plan and on a refusal alike.

        On a refusal it also stamps the machine-readable half -- ``gpu_leased``
        plus the lease records -- but **only when the leases took every card
        this load could have used**. A lease standing on some other card while
        a model genuinely does not fit is context, not cause, and calling that
        refusal ``gpu_leased`` would send a client away to wait for a release
        that will not change the answer.
        """
        if not blocked:
            return result
        lines = self._lease_lines(blocked)
        if isinstance(result, LoadRejected):
            result.suggestions.extend(lines)
            candidates = self._lease_candidates(allowed)
            if candidates and candidates <= blocked:
                result.leases = self._blocking_leases(frozenset(candidates))
                # A lease outranks whatever coarser code got there first: it is
                # the specific, waitable cause, and the only one with a clock.
                result.reason_code = "gpu_leased"
        else:
            result.notes.extend(lines)
        return result

    def _leased_rejection(self, record: ModelRecord, clash: list[int]) -> LoadRejected:
        """A forced placement onto someone else's leased card is refused, not honoured."""
        rejection = LoadRejected(
            model_id=record.id,
            reason=(
                f"the requested placement names CUDA {clash}, which is leased to someone "
                f"else; a lease is not a default to override, it is a promise to its holder"
            ),
            suggestions=[
                *self._lease_lines(frozenset(clash)),
                "load without the device override and let the planner place it elsewhere",
            ],
            reason_code="gpu_leased",
            leases=self._blocking_leases(frozenset(clash)),
        )
        log.info(
            "load rejected: device leased to another holder",
            model_id=record.id,
            devices=clash,
        )
        return rejection

    def _gpus_as_if_gone(self, residents: Sequence[InstanceInfo]) -> list[GpuInfo]:
        """The live GPU list with the planned footprints credited as free.

        The footprint is each instance's own plan (:meth:`instance_footprint`),
        the same figure the eviction ladder credits for a victim; the truth is
        whatever the driver releases when the child exits, and a plan made on
        the estimate meets the same one-retry OOM path a post-eviction plan
        does. One instance for a forced reload (D30); the reload target plus
        every preemptable lower-tier resident for a priority load (D46).
        """
        gpus = list(self.probe.list_gpus())
        footprint: dict[int, int] = {}
        for resident in residents:
            for dev, amount in self.instance_footprint(resident).items():
                footprint[dev] = footprint.get(dev, 0) + amount
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
        allow_evict: bool,
        gpus: Sequence[GpuInfo] | None = None,
        extra_own_pids: Sequence[int] = (),
        evict_busy: bool = False,
        source: str | None = None,
        parallel_auto: bool = False,
        priority: int = PRIORITY_BACKGROUND,
    ) -> PlanResult:
        """:meth:`plan_load` proper; ``gpus``/``extra_own_pids`` are the reload view."""
        settings = record.settings
        defaults = self.config.models

        requested_ctx = ctx_size or settings.ctx_size
        slots, auto_parallel = self._resolve_parallel(record, parallel, force_auto=parallel_auto)
        kv_k: KvCacheType = (
            kv_cache_type or settings.kv_cache_type or defaults.default_kv_cache_type
        )
        kv_v: KvCacheType = kv_cache_type_v or settings.kv_cache_type_v or kv_k
        # Resolved by plan_load -- the ONE place the default (config policy,
        # plus D46's tier clause) lives; a second lookalike resolution here
        # would silently diverge on the priority rule.
        evict_allowed = allow_evict

        ladder = self._context_ladder(record, requested_ctx)
        min_ctx = settings.min_ctx if ctx_size is None else None
        if min_ctx:
            # The floor a fallback model must not serve below: a window that
            # "works" per turn and then shreds a long session through
            # compaction is worse than a refusal the client can route around.
            # Only when the request named no ctx_size -- an explicit ask is
            # honoured verbatim (D14).
            ladder = [c for c in ladder if c >= int(min_ctx)] or [int(min_ctx)]
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
                    evict_busy=evict_busy,
                    priority=priority,
                    extra_notes=self._rung_notes(
                        ctx, aim, floor, thinking=thinking, kv_k=cand_k, kv_options=kv_options
                    ),
                )
                if isinstance(attempt, LoadPlan):
                    self._log_plan(record, attempt, tried, source=source)
                    return attempt

        if not evict_allowed:
            return self._min_ctx_noted(
                min_ctx,
                self._terminal_rejection(
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
                    evict_busy=evict_busy,
                    source=source,
                    priority=priority,
                ),
            )

        # Pass 2 (DECISIONS.md D16): even the floor does not fit, so eviction is
        # now DECIDED -- the choice is no longer "evict or not" but "having
        # evicted, what context do we get?". Re-walk the same ladder against
        # available + reclaimable and take the highest rung that fits. Loading
        # at the 8192 floor after freeing 19 GB, when 65536 would have fitted
        # for the identical cost, is the defect this pass exists to fix.
        reclaimable = self._reclaimable_bytes(loaded, evict_busy=evict_busy, for_priority=priority)
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
                    evict_busy=evict_busy,
                    priority=priority,
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
                    self._log_plan(record, attempt, tried, source=source)
                    return attempt

        return self._min_ctx_noted(
            min_ctx,
            self._terminal_rejection(
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
                evict_busy=evict_busy,
                source=source,
                priority=priority,
            ),
        )

    @staticmethod
    def _min_ctx_noted(min_ctx: int | None, result: PlanResult) -> PlanResult:
        """Name ``settings.min_ctx`` in a refusal it caused.

        Without the line, "not even the floor fits" reads as a VRAM problem
        when the actual obstacle is the model's own configured floor.
        """
        if min_ctx and isinstance(result, LoadRejected):
            result.suggestions.append(
                f"the model's settings.min_ctx = {int(min_ctx)} refuses any smaller "
                f"window; lower or clear it to let the context ladder step down, or "
                f"free VRAM until {int(min_ctx)} fits"
            )
        return result

    # -- plan_load helpers -------------------------------------------------

    def _resolve_parallel(
        self, record: ModelRecord, parallel: int | None, *, force_auto: bool = False
    ) -> tuple[int, bool]:
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
        if force_auto:
            # A leased model (D43): the cards are its alone, so an integer
            # models.default_parallel -- a rig-wide caution -- does not apply.
            return 1, True
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

    def _reclaimable_bytes(
        self,
        loaded: Sequence[InstanceInfo],
        *,
        evict_busy: bool = False,
        for_priority: int | None = None,
    ) -> int:
        """VRAM the planner is allowed to take back by evicting idle models."""
        total = 0
        for instance in self._evictable(loaded, include_busy=evict_busy, for_priority=for_priority):
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
        evict_busy: bool = False,
        source: str | None = None,
        priority: int = PRIORITY_BACKGROUND,
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
            priority=priority,
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

    def _log_plan(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        tried: Sequence[str],
        source: str | None = None,
    ) -> None:
        """One INFO line per plan: what was tried, and what was chosen.

        Replaces the per-rung ``load rejected`` spam (fifteen INFO lines for one
        ordinary ladder walk, none of them the answer). The rejected rungs are
        still in the record -- as one ``rungs`` field on the line that matters.
        """
        emit = log.info if self._log_plans else log.debug
        emit(
            "load planned",
            model_id=record.id,
            # WHO asked. The 2026-08-19 log review could not tell an OpenClaw
            # load from the GUI's from a JIT one, on a box several clients
            # share; "a 262144-token model appeared on three GPUs" with no
            # requester is not a diagnosable event.
            source=source,
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
        evict_busy: bool = False,
        priority: int = PRIORITY_BACKGROUND,
    ) -> PlanResult:
        """Plan a load at one specific context size.

        ``gpus`` is the live probe unless a caller hands in a view (the forced
        reload credits the resident child's footprint back, see
        :meth:`plan_load`); ``extra_own_pids`` are pids that count as ours in
        the holder attribution beyond those in ``loaded``.

        ``prefer_single_gpu`` (per-model setting, falling back to
        ``planner.prefer_single_gpu``) governs the primary placement walk only.
        The eviction-fallback round below still tries single-card placements
        before splits regardless of the flag: there the alternative is refusing
        the load outright, so the cheap option is always worth a try.
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
                evict_busy=evict_busy,
                priority=priority,
            )

        order = self._candidate_order(gpus)
        gpu_map = {gpu.index: gpu for gpu in gpus}
        affinity = self._affinity_for(record)
        pools = self._device_pools(order, gpu_map, affinity)

        # Per-model override of the global policy; None = inherit (D48).
        per_model_single = settings.prefer_single_gpu
        prefer_single = (
            per_model_single
            if per_model_single is not None
            else self.config.planner.prefer_single_gpu
        )

        for pool, pool_note in pools:
            # Policy: the cheapest placement that fits. A single GPU avoids
            # tensor-split overhead and all cross-GPU traffic, so every
            # single-GPU option is tried (best card first) before any split.
            if prefer_single:
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

        # Nothing fits as-is. Try again with unpinned models evicted, worst
        # tier first (D46) and least-recently-used within one; a tier the
        # asking load does not outrank or match is never a candidate.
        if evict_allowed:
            evictable = self._evictable(loaded, include_busy=evict_busy, for_priority=priority)
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
                                    "evicting unpinned models, lowest priority tier "
                                    "first and least-recently-used within one: "
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
            evict_busy=evict_busy,
            priority=priority,
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

    @contextlib.contextmanager
    def _observation_pass(self) -> Iterator[None]:
        """Give one planning pass its own observed-footprint lookup cache (D51).

        Entered once per :meth:`plan_load`, around the whole ladder, so every
        rung of one load shares one set of answers and the next load asks the
        database again. Saved and restored rather than simply cleared: a
        planner is not guaranteed to be planning one load at a time, and a
        nested or concurrent pass must not leave the outer one without a cache.
        """
        previous = self._obs_memo
        self._obs_memo = _ObservationMemo()
        try:
            yield
        finally:
            self._obs_memo = previous

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
        """The formula estimate, corrected by what this configuration measured (D51).

        This is where the correction lands because this is the one funnel every
        candidate estimate flows through -- the fit check in
        :meth:`_try_devices`, the slot walk in :meth:`max_slots_by_vram`, the
        forced placement in :meth:`_plan_on_devices`, and the numbers a
        :class:`LoadRejected` reports. Correcting anywhere further downstream
        would leave those disagreeing with each other: a plan sized against a
        measured footprint but refused against a formula one, or the reverse.

        It is applied BEFORE any fit or refusal decision and before the
        per-device split is taken, so each card's share is a share of the
        corrected total rather than of a number the last load disproved.

        The metadata-less fallback is deliberately left alone: an on-disk file
        size is not a formula, so there is no band to clamp a measurement
        against and nothing the correction could honestly say about it.
        """
        try:
            estimate = self.estimate(
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
        correction = self._observed_correction(
            record,
            ctx=ctx,
            slots=slots,
            kv_k=kv_k,
            kv_v=kv_v,
            n_devices=n_devices,
            formula_bytes=estimate.total_bytes,
        )
        if correction is None:
            return estimate
        return scaled_estimate(estimate, correction.factor)

    def _observed_correction(
        self,
        record: ModelRecord,
        *,
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        n_devices: int,
        formula_bytes: int,
    ) -> ObservedCorrection | None:
        """Look up what this configuration last really weighed, and by how much
        the formula should move because of it (D51).

        Inert -- returning ``None`` without touching the database -- unless all
        three are true: ``planner.observed_correction`` is on, a lookup is
        wired in, and this call is inside a planning pass (see
        :attr:`_obs_memo`). Every result is memoized for the rest of the pass,
        including "there is nothing", because the ladder asks the same question
        once per placement per context rung.

        A lookup that raises disables itself for the rest of the pass after one
        warning. The correction is an accuracy improvement; a broken database
        must never be able to turn it into a refused load.
        """
        memo = self._obs_memo
        if memo is None or memo.failed:
            return None
        if self._observation_lookup is None:
            return None
        if not self.config.planner.observed_correction:
            return None
        key = (record.id, int(ctx), int(slots), str(kv_k), str(kv_v), int(n_devices))
        if key in memo.rows:
            return memo.rows[key]
        try:
            row = self._observation_lookup(
                record.id,
                ctx_size=int(ctx),
                parallel=int(slots),
                kv_cache_type=kv_k,
                kv_cache_type_v=kv_v,
                device_count=int(n_devices),
            )
        except Exception as exc:  # noqa: BLE001 -- any failure means "no history"
            memo.failed = True
            log.warning(
                "observed-footprint lookup failed; planning from the formula alone",
                model_id=record.id,
                error=str(exc),
                detail=(
                    "a database that cannot answer must cost accuracy, never a load: "
                    "the correction is skipped for the rest of this planning pass"
                ),
            )
            return None
        correction: ObservedCorrection | None = None
        actual = row.get("actual_bytes") if row is not None else None
        if isinstance(actual, (int, float)) and actual > 0:
            correction = observed_correction(
                formula_bytes=formula_bytes, observed_bytes=int(actual)
            )
        memo.rows[key] = correction
        if correction is not None:
            # DEBUG, not INFO: the ladder applies this once per placement per
            # context rung and only one of those becomes a load. The note on
            # the chosen plan is the line a human is meant to see.
            log.debug(
                "estimate corrected from the last load of this configuration",
                model_id=record.id,
                ctx=ctx,
                parallel=slots,
                devices=n_devices,
                formula_mb=round(correction.formula_bytes / MB),
                observed_mb=round(correction.observed_bytes / MB),
                corrected_mb=round(correction.formula_bytes * correction.factor / MB),
                factor=round(correction.factor, 3),
                clamped=correction.clamped,
            )
        return correction

    def _memoized_correction(
        self,
        model_id: str,
        *,
        ctx: int,
        slots: int,
        kv_k: KvCacheType,
        kv_v: KvCacheType,
        n_devices: int,
    ) -> ObservedCorrection | None:
        """The correction already spent on this configuration's estimate, for the note.

        Read-only on purpose. :meth:`_safe_estimate` has already decided and
        cached; re-deriving it here would mean recomputing a factor against an
        estimate that is itself already corrected, which is how a correction
        turns into a compounding one.
        """
        memo = self._obs_memo
        if memo is None:
            return None
        return memo.rows.get((model_id, int(ctx), int(slots), str(kv_k), str(kv_v), int(n_devices)))

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

        # Whatever history moved this estimate travels with the plan (D51). A
        # placement whose numbers came from a measurement rather than from the
        # formula has to say so, or the `load planned` line and the API
        # response describe a calculation nobody can reproduce. Asked at
        # ``chosen_slots``, which is what the estimate above was finally sized
        # at once auto-parallel had its say.
        # ``chosen_slots`` first, then the count originally asked for: when the
        # slot sizer bails early (no usable KV geometry) it hands back the
        # estimate it was given, which was corrected at ``slots``. Falling back
        # keeps the note describing the estimate actually in ``shared`` rather
        # than going quiet about a correction that was really applied.
        correction: ObservedCorrection | None = None
        for candidate_slots in dict.fromkeys((chosen_slots, slots)):
            correction = self._memoized_correction(
                record.id,
                ctx=ctx,
                slots=candidate_slots,
                kv_k=kv_k,
                kv_v=kv_v,
                n_devices=len(devices),
            )
            if correction is not None:
                break
        obs_notes = [correction.note] if correction is not None else []

        if len(devices) == 1:
            index = devices[0]
            return LoadPlan(
                devices=[index],
                tensor_split=None,
                split_mode="none",
                main_gpu=index,
                per_gpu_bytes={index: total_needed},
                notes=[
                    f"single-GPU placement on CUDA{index} (no tensor-split overhead)",
                    *obs_notes,
                ],
                **shared,
            )

        # Proportional split by usable capacity. Every device must actually be
        # able to hold its share, otherwise the split is not viable even though
        # the totals add up.
        #
        # The output layer is not split: llama.cpp puts it on the LAST device of
        # the list, beyond that device's share of the blocks (D40). So the last
        # device's capacity is reduced by that layer before the fractions are
        # taken -- which both charges it where it really lands and tilts the
        # tensor-split a little away from that card -- and the bytes are added
        # back onto it in per_gpu_bytes. Measured 0.5,0.5 splits here landed
        # 104-110 MiB more on the last card of two small models; the live rig
        # saw 0.76 GiB on a 27B.
        output_shift = min(output_layer_bytes(record.meta), total_needed)
        last = devices[-1]
        effective = dict(capacities)
        effective[last] = max(0, effective[last] - output_shift)
        effective_total = sum(effective.values())
        body = total_needed - output_shift
        if effective_total <= 0 or body > effective_total:
            return None
        split = [effective[d] / effective_total for d in devices]
        per_gpu = {d: int(body * frac) for d, frac in zip(devices, split, strict=True)}
        per_gpu[last] += output_shift
        for dev, want in per_gpu.items():
            if want > capacities[dev]:
                return None

        return LoadPlan(
            devices=list(devices),
            tensor_split=[round(f, 4) for f in split],
            split_mode=self._split_mode_for(record),
            main_gpu=devices[0],
            per_gpu_bytes=per_gpu,
            notes=list(obs_notes),
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
        evict_busy: bool = False,
        auto_parallel: bool = False,
        terminal: bool = True,
        extra_own_pids: Sequence[int] = (),
        priority: int = PRIORITY_BACKGROUND,
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
            for instance in self._evictable(loaded, include_busy=evict_busy, for_priority=priority):
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
            evict_busy=evict_busy,
            priority=priority,
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
        evict_busy: bool = False,
        priority: int = PRIORITY_BACKGROUND,
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
        # 5b. Higher-tier models this load may not displace (D46). Without the
        # line, an empty eviction list beside "does not fit" reads as "pinned
        # or busy", which would be the wrong diagnosis here.
        outranked = [i for i in loaded if not self._is_pinned(i) and i.priority < priority]
        if outranked:
            held = ", ".join(f"{i.model_id} (priority {i.priority})" for i in outranked)
            suggestions.append(
                f"VRAM is held by higher-priority models this load does not outrank: "
                f"{held}; load with a better priority tier, or unload them"
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

        # 9. The box may be BUSY rather than full. A model serving a request is
        # not an eviction candidate (D36) -- killing a child mid-stream fails a
        # live request to save a future one -- so a refusal that would have
        # succeeded against an idle machine must say so, and say it is worth
        # waiting. "Not enough VRAM" for a condition that clears itself in
        # seconds is the least actionable message this planner can produce.
        busy: list[dict[str, Any]] = []
        retry_after: float | None = None
        if evict_allowed and not evict_busy:
            blocked = [
                i
                for i in self.busy_instances(loaded)
                if not self._is_pinned(i) and i.model_id != record.id
            ]
            if blocked:
                busy = [
                    {"model_id": i.model_id, "active_requests": i.active_requests} for i in blocked
                ]
                retry_after = BUSY_RETRY_AFTER_S
                names = ", ".join(
                    f"{i.model_id} ({i.active_requests} in flight)"
                    if i.active_requests
                    else f"{i.model_id} (still loading)"
                    for i in blocked
                )
                suggestions.insert(
                    0,
                    f"wait ~{BUSY_RETRY_AFTER_S:.0f}s: {names} would free the VRAM but is "
                    f"serving right now and is never evicted mid-request; pass force=true "
                    f"to evict it anyway",
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
            busy_models=busy,
            retry_after_s=retry_after,
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

    def _evictable(
        self,
        loaded: Sequence[InstanceInfo],
        *,
        include_busy: bool = False,
        for_priority: int | None = None,
    ) -> list[InstanceInfo]:
        """Unpinned, idle instances, worst tier first, least-recently-used within one.

        Instances with in-flight requests are never evicted -- killing a child
        mid-stream fails a live request to save a future one, which is the
        wrong trade on a server several agents share. Nor is one that is still
        ``loading``: it has not taken the VRAM its plan promises, so evicting it
        frees a figure that does not exist yet (the same reasoning D29 used to
        serialise loads in the first place).

        ``include_busy`` is the ONE override, reached only by an explicit
        ``force=True`` from a human-driven caller (D36). A JIT load never sets
        it: an inference request that arrives mid-stream for somebody else must
        queue or be refused, never win by killing the stream.

        ``for_priority`` is the asking load's tier (D46): only instances at the
        same or a lower tier (an equal or higher number) are candidates, so a
        background load can never displace the chat model to make room for
        itself. ``None`` applies no tier rule -- the informational surfaces
        that predate tiers.
        """
        candidates = [
            i
            for i in loaded
            if not self._is_pinned(i)
            and i.state == "ready"
            and (include_busy or i.active_requests == 0)
            and (for_priority is None or i.priority >= for_priority)
        ]
        return sorted(
            candidates,
            key=lambda i: (-i.priority, i.last_activity_at or i.started_at or 0.0),
        )

    @staticmethod
    def busy_instances(loaded: Sequence[InstanceInfo]) -> list[InstanceInfo]:
        """Instances a load must not disturb: serving a request, or still loading."""
        return [i for i in loaded if i.active_requests > 0 or i.state == "loading"]

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

    def ubatch_for(self, record: ModelRecord, slots: int = 1) -> int:
        """The ``-ub`` a launch of ``record`` at ``slots`` gets, as a real number.

        Shares :func:`effective_ubatch` with the supervisor, so the VRAM the
        planner charges for the micro-batch is the micro-batch the child really
        runs with -- including the automatic many-slots raise (D38 §5 / D40),
        which is why ``slots`` matters: the estimate for an eight-slot rung must
        include the bigger compute buffer that rung will actually allocate.
        ``None`` from the shared policy means "engine default", i.e. 512.
        """
        ub = effective_ubatch(
            settings_ubatch=record.settings.ubatch_size,
            engine_ubatch=getattr(self.config.engine, "ubatch_size", None),
            engine_ubatch_many_slots=getattr(self.config.engine, "ubatch_many_slots", None),
            slots=slots,
        )
        return ub if ub is not None else DEFAULT_UBATCH

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
        per_gpu_actual: Mapping[int, int] | None = None,
    ) -> None:
        """Record predicted-vs-actual VRAM so the fudge factor can be tuned.

        Logged at INFO with the ratio, and forwarded to the observation sink
        (the SQLite table) when one is wired in.

        ``per_gpu_actual`` is what the child really holds on each of the plan's
        devices (D39's per-adapter measurement). It is stored beside the plan's
        own ``per_gpu_bytes`` (D40) and compared here: a device that ended up
        holding more than :data:`PER_DEVICE_OVERRUN_WARN` times its planned
        share is a WARNING naming the card and both numbers, because that is
        the delta a tight placement OOMs on -- and it was invisible while the
        observation was one total.
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
            per_device_mb=(
                {str(d): round(b / MB) for d, b in sorted(per_gpu_actual.items())}
                if per_gpu_actual
                else None
            ),
        )
        overruns = per_device_overruns(plan, per_gpu_actual)
        if overruns:
            log.warning(
                "a device holds more than its planned share",
                model_id=model_id,
                devices=plan.devices,
                overruns={
                    f"CUDA{d}": {"planned_mb": round(p / MB), "actual_mb": round(a / MB)}
                    for d, p, a in overruns
                },
                detail=(
                    "llama.cpp places the output layer on the last device of the list; "
                    "the planner now charges it there (D40), so a persistent overrun "
                    "means the charge is too small for this model"
                ),
            )
        if self._observation_sink is not None:
            self._observation_sink(
                {
                    "model_id": model_id,
                    "ctx_size": plan.ctx_size,
                    "parallel": plan.parallel,
                    "kv_cache_type": plan.kv_cache_type,
                    # The V half too (migration 007). Calibration never needed
                    # it -- it averages ratios across every model. Reading ONE
                    # row back as an estimate (D51) does: K and V are set
                    # independently and a quantized V cache is roughly half the
                    # V bytes, so a row that cannot name its V type cannot
                    # safely describe the placement that asks for it.
                    "kv_cache_type_v": plan.kv_cache_type_v,
                    "devices": ",".join(str(d) for d in plan.devices),
                    "predicted_bytes": predicted,
                    "actual_bytes": actual_bytes,
                    "weights_bytes": plan.estimate.weights_bytes,
                    "ok": 1 if ok else 0,
                    "note": note,
                    "per_gpu_planned": json.dumps(
                        {str(d): int(b) for d, b in sorted(plan.per_gpu_bytes.items())}
                    ),
                    "per_gpu_actual": (
                        json.dumps({str(d): int(b) for d, b in sorted(per_gpu_actual.items())})
                        if per_gpu_actual
                        else None
                    ),
                }
            )


#: A device holding more than this multiple of its planned share is worth a
#: WARNING in the load observation (D40). 15%: below the 10% headroom plus the
#: CUDA context charge the planner already keeps per card, a smaller overrun
#: is noise; above it the card was genuinely planned too tight.
PER_DEVICE_OVERRUN_WARN = 1.15


def per_device_overruns(
    plan: LoadPlan, per_gpu_actual: Mapping[int, int] | None
) -> list[tuple[int, int, int]]:
    """``(device, planned_bytes, actual_bytes)`` for every device over the bar.

    Only devices the plan named and measured are compared; a card with no
    planned share (``0``) is skipped rather than reported as infinitely over.
    """
    if not per_gpu_actual:
        return []
    out: list[tuple[int, int, int]] = []
    for device in plan.devices:
        planned = int(plan.per_gpu_bytes.get(device, 0) or 0)
        actual = int(per_gpu_actual.get(device, 0) or 0)
        if planned <= 0 or actual <= 0:
            continue
        if actual > planned * PER_DEVICE_OVERRUN_WARN:
            out.append((device, planned, actual))
    return out


#: --- observed-footprint correction (D51) --------------------------------
#:
#: What a measured child is multiplied by before the planner will spend it as
#: an estimate. The observation is one sample of a placement the planner is
#: about to make again but not identically: tensor-split proportions are
#: recomputed from live free VRAM on every load, so the same model on the same
#: two cards can land a little differently, and the compute buffers move with
#: the batch the child happens to be serving when it is measured. 10% is the
#: price of keeping the error direction pointed at "refuse", which costs a load
#: that would have fit, rather than at "OOM", which costs a child mid-request.
OBS_SAFETY = 1.10

#: The floor the correction may pull the estimate down to, as a fraction of the
#: formula's own answer. A row that is a fluke -- or contaminated in some way
#: the note check has not yet learned to spot -- can talk the estimate down by
#: at most 40%. The live Gemma-4-E4B case lands at 0.674, comfortably inside
#: it; anything claiming a model is less than 60% of its computed size is more
#: likely a measurement that missed an allocation than a discovery.
OBS_BAND_MIN = 0.60

#: And the ceiling. A load that measured far ABOVE its estimate is real
#: information -- Dark-Scarlett-27B came in at 1.042 -- and the correction
#: raises the estimate for it, which is the direction that prevents OOM. But
#: past 30% over, the formula is not slightly stale, it is missing a term, and
#: quietly padding every plan of that model would hide the bug instead of
#: fixing it. The clamp is the point, not a detail: exactly as with
#: :data:`OVERHEAD_FRACTION_MIN`, a feedback loop with no bound is a way to
#: make the planner slowly wrong with nobody noticing.
OBS_BAND_MAX = 1.30


@dataclass(frozen=True)
class ObservedCorrection:
    """A measured footprint the planner has decided to trust, and by how much.

    ``factor`` is what every term of the formula estimate is multiplied by;
    ``note`` is the sentence the chosen plan carries into the ``load planned``
    log and the API response, because a plan whose numbers were silently
    overridden by history is a plan nobody can debug.
    """

    factor: float
    observed_bytes: int
    formula_bytes: int
    clamped: bool
    note: str


def observed_correction(*, formula_bytes: int, observed_bytes: int) -> ObservedCorrection | None:
    """Trust a measured footprint over the formula, within a band (D51).

    ``corrected = clamp(observed * OBS_SAFETY, formula * OBS_BAND_MIN,
    formula * OBS_BAND_MAX)``, and the returned ``factor`` is
    ``corrected / formula``.

    The formula is a good general estimator and a poor specific one. It is
    built from GGUF geometry that every architecture reports slightly its own
    way, so it is systematically wrong per *model family* while being roughly
    right across the library -- and one global ``compute_overhead_fraction``
    (:func:`calibrated_overhead_fraction`) cannot be two signs at once. The
    live rig on 2026-08-30 had both signs open simultaneously:

    * Dark-Scarlett-27B: formula 35742 MB, measured 37258 MB. Corrected to
      40984 MB, a factor of 1.147 -- MORE conservative, which is this model's
      real risk direction, since it was already over its estimate.
    * Gemma-4-E4B: formula 13377 MB, measured 8202 MB. Corrected to 9022 MB, a
      factor of 0.674, releasing ~4.3 GB of phantom. The formula does not model
      per-layer embeddings or the KV sharing this architecture uses, and no
      amount of global calibration was going to teach it: the planner predicted
      13377 again twelve seconds after that observation landed.

    A 5 GB phantom on a 24 GB 3090 is not a rounding error -- it refuses a
    co-residency that fits, or pushes the model onto a mixed-generation split
    this project's own measurements put at half the speed.

    Returns ``None`` when there is nothing to do: a non-positive formula total
    (no band to clamp against), a non-positive measurement, or a correction
    that rounds to no change at all.
    """
    if formula_bytes <= 0 or observed_bytes <= 0:
        return None
    trusted = observed_bytes * OBS_SAFETY
    low = formula_bytes * OBS_BAND_MIN
    high = formula_bytes * OBS_BAND_MAX
    corrected = min(high, max(low, trusted))
    clamped = corrected != trusted
    factor = corrected / formula_bytes
    if abs(factor - 1.0) < 0.005:
        return None
    note = (
        f"estimate corrected x{factor:.2f} from the last load of this exact "
        f"configuration (observed {round(observed_bytes / MB)} MB against a formula "
        f"estimate of {round(formula_bytes / MB)} MB)"
    )
    if clamped:
        note += (
            f"; clamped to the {OBS_BAND_MIN:.2f}-{OBS_BAND_MAX:.2f} band around the "
            f"formula, so the measurement was only partly trusted"
        )
    return ObservedCorrection(
        factor=factor,
        observed_bytes=int(observed_bytes),
        formula_bytes=int(formula_bytes),
        clamped=clamped,
        note=note,
    )


def scaled_estimate(estimate: VramEstimate, factor: float) -> VramEstimate:
    """Every term of ``estimate`` multiplied by ``factor``.

    ``VramEstimate.total_bytes`` is a derived sum, not a field, so a correction
    to the total has to be spent across the terms. It is spread uniformly
    rather than dumped into ``compute_bytes`` -- the term that exists to absorb
    unmodelled overhead -- for one blunt reason: the Gemma-4-E4B correction is
    -4.3 GB and ``compute_bytes`` is a fraction of that, so the fudge term
    physically cannot hold it. Uniform scaling also keeps the arithmetic
    downstream self-consistent: ``_reject`` computes its "largest context that
    would fit" as ``total - kv``, and a total corrected without its KV term
    would make that subtraction describe a placement that does not exist.

    The cost is that ``weights_bytes`` and ``kv_bytes`` on a corrected estimate
    are shares of a measured whole rather than the formula's own answers for
    those terms. That is honest -- the measurement cannot say which term was
    wrong, only that their sum was -- but it is why the per-device split in
    :meth:`Planner._try_devices` keeps charging the output layer at its real
    size and takes the corrected total as the body to divide.
    """
    return VramEstimate(
        **{
            field_name: max(0, int(round(value * factor)))
            for field_name, value in estimate.model_dump().items()
        }
    )


@dataclass
class _ObservationMemo:
    """One ``plan_load`` call's worth of observed-footprint lookups.

    The context ladder plans the same model at several contexts, each against
    up to every placement on the box, and each of those asks
    :meth:`Planner._safe_estimate` again -- several dozen calls for one load.
    Without this, each would be a SQLite round trip for an answer that cannot
    have changed inside a single planning pass.

    ``model_id`` is part of the key even though the memo is per-call, so that
    two ``plan_load`` calls racing on one Planner can only ever share a cached
    row, never cross-attribute one model's measurement to another.

    ``failed`` latches the first exception out of the lookup for the rest of
    the call. A database that cannot answer must cost one warning and nothing
    else -- never a refused load, and never the same warning once per rung.
    """

    rows: dict[tuple[str, int, int, str, str, int], ObservedCorrection | None] = field(
        default_factory=dict
    )
    failed: bool = False


#: Written into ``load_observations.note`` when ``actual_bytes`` is our own
#: child's VRAM, attributed per pid. Rows without it summed whole-device
#: ``used_bytes`` and are contaminated by every other process on the card --
#: see docs/LIMITATIONS.md.
#:
#: Superseded by :data:`OBSERVATION_NOTE_PER_PID_DEVICE`: on Windows these rows
#: summed the PDH per-process total once per NVML (gpu, pid) row, so a two-GPU
#: load recorded twice its footprint and a four-GPU load four times (D40). The
#: calibrator no longer reads them; the marker is kept so the rows can still be
#: told apart from the device-total ones that came before.
OBSERVATION_NOTE_PER_PID = "per_pid"

#: The marker calibration reads (D40): ``actual_bytes`` is our child's own
#: VRAM on the plan's devices, measured per device -- PDH per adapter joined to
#: CUDA ordinals on Windows (D39), NVML per process per GPU on Linux -- and
#: never the same total counted once per card.
OBSERVATION_NOTE_PER_PID_DEVICE = "per_pid_v2"

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
    the median actual/predicted ratio is 2.97 and p90 is 12.0. The ``per_pid``
    rows that replaced them were right on Linux and wrong on Windows, where the
    per-process total was counted once per card (29 live rows, every multi-GPU
    load at a ratio of its device count; D40). Feeding either to the
    calibrator ratchets the overhead fraction to its ceiling and starts
    refusing loads that fit. The marker is the only thing separating them.
    """
    return [
        row
        for row in observations
        if str(row.get("note") or "") == OBSERVATION_NOTE_PER_PID_DEVICE and row.get("ok", True)
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
