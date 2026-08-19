"""Speed estimator and throughput observation store.

Two halves, deliberately in one module because neither is useful alone.

**The estimator** (:func:`estimate`) answers "how fast will this placement be?"
from first principles, before anything is loaded. Decode is memory-bandwidth
bound: every generated token reads the active weights plus every busy slot's KV
cache, so the time per token is the sum over devices of ``bytes_on_device /
bandwidth_of_device``. Prompt processing is FLOP bound instead -- it ingests
many tokens per pass, so the weights are read once and amortised, and the
limit is arithmetic throughput.

**The observation store** (:func:`parse_metrics`, :func:`sample_between`,
:func:`calibrate`) answers the same question from measurement. llama-server
exposes Prometheus counters on ``/metrics``; sampling them twice gives the real
tokens/second between the samples. The ratio ``measured / estimated`` is a
per-model, per-placement efficiency factor that is folded back into future
estimates.

**The GPU numbers below are NOMINAL vendor figures, not measurements.** They
are the *only* hardware constants in this file, they all live in one table, and
calibration is what makes them right. A nominal peak bandwidth is never reached
by a real kernel, and the gap is not a constant -- it depends on the model's
shape, the quantization, the split, and llama.cpp's kernels. So the estimate is
labelled ``"estimated"`` until an observation exists, ``"calibrated"`` once one
does, and ``"measured"`` when the number being reported *is* the observation.
Treat an uncalibrated number as an order of magnitude, not a promise.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studioforge.core.planner import (
    MAX_PARALLEL_CAP,
    active_weight_bytes,
    is_moe,
    max_parallel_for,
)
from studioforge.logging import get_logger

log = get_logger(__name__)

GIB = 1024**3


# ---------------------------------------------------------------------------
# Hardware constants -- NOMINAL, in one place, corrected by calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuPerf:
    """Nominal peak numbers for one GPU model.

    ``bw_bytes_per_s`` is the vendor's memory bandwidth and ``flops_fp16`` the
    dense fp16/bf16 tensor-core rate without sparsity. Neither is achievable in
    practice; the efficiency factors below (and calibration on top of them) are
    what turn them into a usable estimate.
    """

    label: str
    bw_bytes_per_s: float
    flops_fp16: float


#: Name fragment -> nominal performance. Matched case-insensitively as a
#: substring of ``GpuInfo.name`` ("NVIDIA GeForce RTX 5090" contains "5090").
#: NOMINAL VENDOR FIGURES -- none of these was measured on this rig.
GPU_PERF_TABLE: tuple[tuple[str, GpuPerf], ...] = (
    ("5090", GpuPerf("RTX 5090", 1792e9, 209e12)),
    ("4090", GpuPerf("RTX 4090", 1008e9, 165e12)),
    ("3090", GpuPerf("RTX 3090", 936e9, 71e12)),
)

#: Last-resort profile for a GPU whose name matches nothing in the table. It is
#: deliberately pessimistic: an unknown card that turns out to be fast makes the
#: catalog look conservative, whereas an unknown card assumed fast makes every
#: number on the page a lie.
GPU_PERF_UNKNOWN = GpuPerf("unknown GPU", 600e9, 40e12)

#: VRAM-size fallback, largest first, for an unnamed/renamed card. A 32 GiB
#: consumer card on this class of machine is a 5090; a 24 GiB one is a 3090.
GPU_PERF_BY_VRAM: tuple[tuple[float, GpuPerf], ...] = (
    (30.0, GPU_PERF_TABLE[0][1]),
    (23.0, GPU_PERF_TABLE[2][1]),
)

#: Fraction of nominal memory bandwidth a single-GPU decode actually reaches.
#: Real GEMV kernels on a quantized model do not saturate the bus.
DECODE_EFFICIENCY_SINGLE = 0.75

#: Subtracted from :data:`DECODE_EFFICIENCY_SINGLE` per *additional* device.
#: ``--split-mode layer`` pipelines: each device runs its own layers in turn and
#: hands activations across PCIe, so a split pays synchronisation on every token
#: and only one device is busy at a time.
DECODE_EFFICIENCY_PER_EXTRA_DEVICE = 0.05

#: Fraction of nominal fp16 FLOPs prompt processing reaches. Prefill is a
#: batched GEMM and gets much closer to peak than decode does to bandwidth
#: peak, but dequantization, attention and the KV write all cost real time.
PROMPT_EFFICIENCY = 0.35

#: The context fill ``est_gen_tps`` is quoted at: "a typical turn", not the full
#: window. Decode speed depends on how much KV a slot is actually holding, so a
#: single number needs a stated fill or it means nothing. Quoting the *full*
#: context would advertise the worst case a model ever reaches and make every
#: long-context row look broken; quoting zero would advertise a speed no real
#: conversation sees. 8192 is roughly one agent turn with tool output on this
#: rig, it is below every context tier the catalog offers (so the quote is never
#: above the row's own window), and the catalog still reports the full-window
#: figure separately as ``est_gen_tps_full_ctx``.
REFERENCE_FILL_TOKENS = 8192

#: Per-token latency the roofline does not model: sampling, the grammar/logit
#: pass, the HTTP and slot bookkeeping in llama-server, and the CUDA graph
#: launch for a batch of one. It is a *floor*, not a bandwidth term -- it does
#: not shrink when the model does.
#:
#: Evidence: without it the bandwidth roofline claims 927 tok/s for a 0.92 GB
#: Qwen2.5-1.5B on one 5090 and 2063 tok/s for SmolVLM, numbers llama.cpp has
#: never produced for any model on any hardware; b10425's own single-stream
#: decode on a tiny model tops out in the 300-500 tok/s region, i.e. 2-3 ms per
#: token of which the weights explain well under one. 1.5 ms is the middle of
#: the 1-2 ms llama-server spends per token before it touches a weight.
#:
#: Direction of error: too small. A floor that is under-stated leaves small
#: models looking slightly fast, which is the same direction the rest of this
#: module already errs; over-stating it would drag the *large* models (where
#: t_token is 20-30 ms) visibly slow for no physical reason.
T_TOKEN_OVERHEAD_S = 1.5e-3

#: Multiplier on :data:`DECODE_EFFICIENCY_SINGLE` for a mixture-of-experts
#: decode. A dense decode streams a few large contiguous tensors per layer; a
#: MoE decode issues ``MUL_MAT_ID`` and gathers ``n_expert_used`` *small*,
#: scattered expert matrices per layer. That is a launch- and occupancy-bound
#: pattern, not a bandwidth-bound one -- the bus is idle between gathers.
#:
#: Evidence: D20 recorded the reference 122B-A10B at 37.05 tok/s measured
#: against 143.5 tok/s estimated (ratio 0.26) on a formula whose only derate was
#: the four-device term, and the live registry now holds 84 rows for it at a
#: median measured/estimated of 0.411. Splitting that gap between "MoE kernels"
#: and "four-way PCIe split" (the constant below) lands the MoE share near 0.45.
#:
#: Direction of error: this is the one constant that makes MoE rows *slower*.
#: Set too low and a MoE looks worse than it is (safe: the catalog under-promises
#: and calibration corrects upward); set too high and the catalog repeats the
#: 4x-optimistic claim D20 already named as the failure to avoid.
MOE_DECODE_EFFICIENCY = 0.45

#: Per-extra-device decode derate for MoE, replacing
#: :data:`DECODE_EFFICIENCY_PER_EXTRA_DEVICE` (0.05, dense). A layer split pays
#: one synchronisation per device per token for *any* model; a MoE additionally
#: routes to experts that may live on a device the previous layer did not touch,
#: so the pipeline stalls harder and the activations crossing PCIe are less
#: predictable. Twice the dense penalty is a deliberate approximation, not a
#: measurement: it is the smallest step that separates the two behaviours, and
#: it errs slow, which for a speed claim is the safe side.
DECODE_EFFICIENCY_PER_EXTRA_DEVICE_MOE = 0.10

#: Multiplier on :data:`PROMPT_EFFICIENCY` for a MoE *prefill*. Prefill is a
#: batched GEMM, but a MoE's batch is shredded by routing: the tokens in one
#: ubatch scatter across ``n_expert`` experts, so each expert sees a fraction of
#: the batch and the GEMM degrades toward many skinny matmuls plus the gather /
#: scatter around them.
#:
#: Evidence: the reference 122B-A10B measures 869 tok/s prefill on the four-way
#: split; the plain FLOP roofline at PROMPT_EFFICIENCY alone says ~2800 for the
#: same placement, and D20 recorded the same shape (1149 measured vs 3752
#: estimated, ratio 0.31) before the harmonic split term existed. 0.4 lands the
#: estimate ~1.3x above the measurement, which is the smallest honest gap the
#: two anchors support.
#:
#: Direction of error: slightly optimistic, matched to the decode side so that a
#: single calibration factor per model does not have to pull the two terms in
#: opposite directions.
MOE_PROMPT_EFFICIENCY = 0.4

#: Seconds of cross-device synchronisation one layer costs under
#: ``--split-mode tensor``. Tensor parallelism shards every weight matrix *and*
#: the KV across the devices, so each device does 1/N of the work -- and then
#: every layer has to all-reduce its partial results twice (after the attention
#: projection and after the MLP). On NVLink that is nearly free; on this rig the
#: cards talk over PCIe and it is not.
#:
#: 60 us is an approximation, not a measurement of this rig's bus: two small
#: all-reduces at PCIe 4.0 x16 latency. It is the term that decides where tensor
#: mode stops paying, and it is deliberately large enough to keep small models
#: on the layer split -- which is what the rig actually measured. Qwen2.5-1.5B
#: Q4_K_M on 2x RTX 3090, 8k context: layer 344 tok/s, tensor 294 tok/s, single
#: card 353 tok/s (DECISIONS.md D38). Calibration corrects the rest.
T_TENSOR_SYNC_S = 60e-6

#: Bumped whenever :func:`estimate` changes shape. Every observation records the
#: version that produced its ``est_*`` columns, and :func:`calibrate` reads only
#: rows from the *current* version.
#:
#: Without this, the 84 rows the live rig recorded against the v1 formula (which
#: charged a uniform iSWA KV read and had no MoE derate) would keep teaching the
#: v2 estimate a correction for a mistake v2 no longer makes -- a calibration
#: loop learning the difference between two dead formulas. A version column is
#: cheaper and more honest than deleting history: the rows stay readable by
#: :func:`measured_for`, because a *measurement* does not expire when our
#: arithmetic changes.
ESTIMATOR_VERSION = 2

#: Bits per stored weight by quantization label, for turning file bytes into a
#: parameter count when the GGUF does not carry ``general.parameter_count``.
#: Values are the real k-quant mixes rounded to two decimals.
BITS_PER_WEIGHT: dict[str, float] = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.69,
    "Q5_K_S": 5.52,
    "Q5_1": 6.0,
    "Q5_0": 5.5,
    "Q4_K_M": 4.85,
    "Q4_K_S": 4.58,
    "Q4_1": 5.0,
    "Q4_0": 4.5,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.5,
    "Q3_K_M": 3.91,
    "Q3_K_S": 3.5,
    "Q2_K": 3.35,
    "NVFP4": 4.5,
    "MXFP4": 4.25,
}

#: Assumed bits per weight when the label is unknown. Sits between Q4 and Q5,
#: which is where most of a real library lives.
BITS_PER_WEIGHT_DEFAULT = 5.0


def gpu_perf_for(name: str | None, total_bytes: int = 0) -> GpuPerf:
    """Nominal performance for a GPU, by name then by VRAM size."""
    haystack = (name or "").lower()
    for fragment, perf in GPU_PERF_TABLE:
        if fragment in haystack:
            return perf
    gib = total_bytes / GIB if total_bytes else 0.0
    for threshold, perf in GPU_PERF_BY_VRAM:
        if gib >= threshold:
            return perf
    return GPU_PERF_UNKNOWN


def gpu_class(gpus: Sequence[Any]) -> str:
    """A stable label for a *set* of GPUs, e.g. ``"RTX 5090x2+RTX 3090x2"``.

    Recorded alongside every observation so calibration can fall back from
    "this model on these exact devices" to "comparable models on this class of
    hardware" (:func:`calibrate`'s peer tier). Keying the fallback off device
    *indices* would break the moment a card is moved or CUDA reorders them; the
    class label survives that.

    Note the label describes the *whole rig*, not the placement, so the peer
    tier matches on it **and** on device count -- a model on one 5090 and a
    model spread over all four cards share this label and share nothing else.
    """
    counts: dict[str, int] = {}
    for gpu in gpus:
        perf = gpu_perf_for(getattr(gpu, "name", None), int(getattr(gpu, "total_bytes", 0) or 0))
        counts[perf.label] = counts.get(perf.label, 0) + 1
    return "+".join(f"{label}x{n}" for label, n in sorted(counts.items())) or "unknown"


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------


def bits_per_weight(meta: Any) -> float:
    """Bits each stored weight occupies, from the quant label."""
    label = str(getattr(meta, "quant_label", "") or "").upper()
    if label in BITS_PER_WEIGHT:
        return BITS_PER_WEIGHT[label]
    # "Q4_K_M-something" or a vendor suffix: match the longest known prefix.
    for known, bits in sorted(BITS_PER_WEIGHT.items(), key=lambda kv: -len(kv[0])):
        if label.startswith(known):
            return bits
    return BITS_PER_WEIGHT_DEFAULT


def total_params(meta: Any, weights_bytes: int) -> int:
    """Total parameter count, from metadata when present else from file size.

    ``general.parameter_count`` is authoritative and most modern GGUFs carry
    it. When it is absent the count is derived from the stored bytes and the
    quantization's bits-per-weight, which lands within a few percent -- good
    enough for a FLOP-bound estimate whose efficiency factor is itself an
    approximation.
    """
    declared = int(getattr(meta, "param_count", 0) or 0)
    if declared > 0:
        return declared
    bpw = bits_per_weight(meta)
    if weights_bytes <= 0 or bpw <= 0:
        return 0
    return int(weights_bytes * 8 / bpw)


def dense_trunk_params(meta: Any) -> int:
    """Parameters every token uses regardless of expert routing.

    Attention (Q/K/V/O projections across every layer) plus the output
    embedding, all derived from metadata the GGUF already carries::

        attn    = n_layer * n_embd * (n_head*head_k + n_head_kv*head_k
                                      + n_head_kv*head_v + n_head*head_v)
        lm_head = n_vocab * n_embd

    The *input* embedding is deliberately absent: it is a row lookup, so it
    costs one row of bandwidth per token rather than the whole table. The
    output projection is a real matmul against the whole vocabulary and is
    charged in full.

    Returns 0 when the shape cannot be derived, which callers read as "cannot
    model this" and fall back on.
    """
    n_layer = int(getattr(meta, "n_layer", 0) or 0)
    n_embd = int(getattr(meta, "n_embd", 0) or 0)
    n_head = int(getattr(meta, "n_head", 0) or 0)
    n_head_kv = int(getattr(meta, "n_head_kv", 0) or 0) or n_head
    head_k = int(getattr(meta, "head_dim_k", 0) or 0)
    head_v = int(getattr(meta, "head_dim_v", 0) or 0) or head_k
    n_vocab = int(getattr(meta, "n_vocab", 0) or 0)
    if min(n_layer, n_embd, n_head, head_k) <= 0:
        return 0
    attn = (
        n_layer
        * n_embd
        * (n_head * head_k + n_head_kv * head_k + n_head_kv * head_v + n_head * head_v)
    )
    return attn + max(0, n_vocab) * n_embd


def active_params(meta: Any, weights_bytes: int) -> int:
    """Parameters actually multiplied per token.

    Dense: all of them. **MoE: the dense trunk in full plus the routed share of
    the experts** -- which is deliberately *not*
    :func:`studioforge.core.planner.active_weight_bytes`, and the difference
    matters.

    The planner charges the whole file at ``n_expert_used / n_expert``,
    including attention and embeddings, which are not routed. For a sparsely
    routed model that under-counts badly. Measured on the reference rig's
    ``Qwen3.5-122B-A10B`` (256 experts, 8 used, 48 layers, n_embd 3072):

    | method | active |
    | --- | --- |
    | flat routed share (the planner's) | 3.8B |
    | dense trunk + routed experts (here) | 7.1B |
    | what the model's own name claims | ~10B |

    The planner's version is *right for the planner*: it feeds D17's knee,
    where under-counting means proposing fewer slots, which cannot cause an
    OOM. Here it would mean claiming a model is 2.6x faster than it is, which
    is the unsafe direction -- so this module models the trunk instead. The
    remaining gap to the stated 10B is a shared expert and dense FFN layers
    that GGUF metadata does not describe; it still errs slow-and-safe.

    Falls back to the flat routed share when the architecture dimensions are
    missing.
    """
    total = total_params(meta, weights_bytes)
    if total <= 0:
        return 0
    n_expert = int(getattr(meta, "n_expert", 0) or 0)
    n_used = int(getattr(meta, "n_expert_used", 0) or 0)
    if not (n_expert > 1 and 0 < n_used < n_expert):
        return total

    dense = dense_trunk_params(meta)
    if dense <= 0 or dense >= total:
        return max(1, int(total * n_used / n_expert))
    experts = total - dense
    return max(1, int(dense + experts * n_used / n_expert))


def active_bytes(meta: Any, weights_bytes: int) -> int:
    """Weight bytes read per decoded token, with the MoE trunk charged in full.

    The bandwidth counterpart of :func:`active_params`, kept in step with it by
    construction (same ratio applied to the file size) so the decode term and
    the prefill term can never disagree about what "active" means.
    """
    total = total_params(meta, weights_bytes)
    if total <= 0 or weights_bytes <= 0:
        # No parameter count to reason about; the planner's flat share is the
        # only estimate available.
        return active_weight_bytes(meta, weights_bytes)
    return max(1, int(weights_bytes * active_params(meta, weights_bytes) / total))


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


def estimate(
    meta: Any,
    weights_bytes: int,
    devices_split: Mapping[int, int],
    *,
    kv_read_bytes_per_slot: int,
    parallel: int,
    gpus: Mapping[int, Any] | None = None,
    efficiency: float = 1.0,
    prompt_efficiency: float = 1.0,
    knee: int | None = None,
    split_mode: str = "layer",
) -> dict[str, Any]:
    """Projected prompt and generation speed for one placement.

    Args:
        meta: the model's :class:`~studioforge.types.GgufMeta`.
        weights_bytes: total stored weight bytes (all shards).
        devices_split: ``{cuda_index: bytes placed on that device}``. The
            planner's ``LoadPlan.per_gpu_bytes`` is exactly this. Only the
            *proportions* matter; a layer split places the same fraction of the
            active weights as of the total.
        kv_read_bytes_per_slot: bytes ONE slot reads from its KV cache per
            decode step, from
            :func:`studioforge.core.planner.kv_read_bytes_per_slot`. Keyword-only
            and *not* ``ctx_fill * kv_bytes_per_token``: an iSWA model re-reads
            only its window on five layers in six, and a hybrid model's recurrent
            layers re-read a fixed state rather than a growing cache. Charging
            the uniform product instead is what made a 31B Gemma-4 read 258 GB
            per token at 262k context and report 1.9 tok/s.
        parallel: slots to report the batched aggregate for.
        gpus: ``{cuda_index: GpuInfo}`` so each device's bandwidth can be
            looked up. Missing entries fall back to the unknown-GPU profile.
        efficiency: calibration multiplier for decode (``measured/estimated``).
        prompt_efficiency: the same for prefill.
        knee: slot count past which extra slots stop buying throughput. Taken
            from :func:`~studioforge.core.planner.max_parallel_for` when the
            caller has already computed it.
        split_mode: how the placement shards the model. ``"layer"`` (and
            ``"none"``/``"row"``) keeps the pipeline model below, where the
            per-device times ADD. ``"tensor"`` runs the devices in parallel, so
            the *slowest* device sets the pace and every layer pays
            :data:`T_TENSOR_SYNC_S` of cross-device synchronisation on top.

    Returns a dict with ``prompt_tps``, ``gen_tps``, ``gen_tps_batched`` and a
    ``basis`` block carrying every intermediate value, so a number that looks
    wrong can be taken apart without re-deriving it.

    **Decode.** One token reads the active weights once plus the busy slot's KV,
    and then pays a fixed latency no roofline models. Per device the time is
    ``bytes_there / bandwidth_there``; a layer split runs devices in sequence,
    so the times *add*::

        t_weights = Σ_dev (active_bytes * share_dev / BW_dev) / eff_decode
        t_kv      = Σ_dev (kv_read_bytes_per_slot * share_dev / BW_dev)
        t_token   = t_weights + t_kv + T_TOKEN_OVERHEAD_S
        gen_tps   = calibration / t_token

    ``eff_decode`` is ``0.75 - per_extra * (n_devices - 1)``, with ``per_extra``
    0.05 dense / 0.10 MoE, then multiplied by :data:`MOE_DECODE_EFFICIENCY` for
    a MoE. It divides *only* the weight term: the weight read is the scattered,
    kernel-bound one (a MoE's especially), while the KV re-read is a long
    contiguous stream that lands much closer to peak. Folding the efficiency
    into both would make the estimate wrong in the same direction twice at long
    context, which is where the catalog is read most.

    **Batched decode.** At N slots the weights are still read once per step but
    N slots' KV is read, and the fixed overhead is paid once per *step* rather
    than once per token -- which is most of why batching helps at all::

        t_step(N)       = t_weights + N * t_kv + T_TOKEN_OVERHEAD_S
        gen_tps_batched = calibration * N / t_step(N)

    which is ``N x gen_tps`` while KV traffic is small and flattens as it
    catches up -- the same crossover ``max_parallel_for`` calls the knee. N is
    clamped to the knee so the catalog never advertises throughput past the
    point the planner would stop adding slots.

    **Prefill.** FLOP bound: two FLOPs per active parameter per token. The
    per-device *times* add, exactly as decode's do, because a layer split is a
    pipeline and a single prompt pass walks the devices in turn::

        t_prompt   = Σ_dev (2 * active_params * share_dev / FLOPS_dev) / eff_prompt
        prompt_tps = calibration / t_prompt

    That is the harmonic (share-weighted) mean of the devices' FLOPS, not the
    arithmetic one this used to compute. The arithmetic mean says a 5090 paired
    with a 3090 runs at 140 TFLOPS; the pipeline actually runs at 106, because
    the fast card sits idle while the slow one works. Over-stating a mixed
    split's prefill is what made the planner's ``prefer_single_gpu`` exceptions
    look free.

    **Tensor split.** Under ``--split-mode tensor`` every device holds a slice of
    every weight matrix and of the KV, so they work *at the same time* and the
    per-device times take a ``max`` instead of a sum -- but each layer then pays
    two cross-device all-reduces::

        t_weights = max_dev(active_bytes * share_dev / BW_dev) / eff_decode
        t_kv      = max_dev(kv_read_bytes_per_slot * share_dev / BW_dev)
        t_token   = t_weights + t_kv + n_layer * T_TENSOR_SYNC_S + T_TOKEN_OVERHEAD_S

    The sync term is why this is not free money: it is fixed per layer while the
    halved weight read shrinks with the model, so tensor mode only pays above a
    crossover in model size. Measured on this rig, a 1.5B on two 3090s came out
    *slower* than the layer split and slower than one card (D38), which is what
    the constant is calibrated to reproduce.

    **Prefill is deliberately modelled the same for both modes.** Tensor mode
    parallelises the GEMMs, so a roofline would predict a prefill win -- and the
    rig measured a 57% prefill *loss* on the same 1.5B. Claiming the win would be
    a fabrication and claiming the loss would be an unmeasured constant, so
    prefill keeps the pipeline arithmetic and the benchmark's layer-vs-tensor
    modes are the honest way to know.
    """
    split = {int(k): max(0, int(v)) for k, v in devices_split.items() if int(v) > 0}
    total_split = sum(split.values())
    n_devices = max(1, len(split))
    gpu_map = dict(gpus or {})
    moe = is_moe(meta)

    # NOT planner.active_weight_bytes: see active_params() for why a speed
    # estimate must charge a MoE's dense trunk in full where the planner's
    # slot-count bound may safely under-count it.
    active = active_bytes(meta, weights_bytes)
    params_active = active_params(meta, weights_bytes)
    kv_per_slot = max(0, int(kv_read_bytes_per_slot))

    per_extra = (
        DECODE_EFFICIENCY_PER_EXTRA_DEVICE_MOE if moe else DECODE_EFFICIENCY_PER_EXTRA_DEVICE
    )
    eff_decode = max(0.05, DECODE_EFFICIENCY_SINGLE - per_extra * (n_devices - 1))
    if moe:
        eff_decode *= MOE_DECODE_EFFICIENCY
    eff_prompt = PROMPT_EFFICIENCY * (MOE_PROMPT_EFFICIENCY if moe else 1.0)

    tensor_parallel = split_mode == "tensor" and len(split) > 1
    weight_times: list[float] = []
    kv_times: list[float] = []
    t_prompt_raw = 0.0
    perf_by_device: dict[int, str] = {}
    for index, bytes_here in split.items():
        share = bytes_here / total_split if total_split else 1.0 / n_devices
        perf = gpu_perf_for(
            getattr(gpu_map.get(index), "name", None),
            int(getattr(gpu_map.get(index), "total_bytes", 0) or 0),
        )
        perf_by_device[index] = perf.label
        weight_times.append((active * share) / perf.bw_bytes_per_s)
        kv_times.append((kv_per_slot * share) / perf.bw_bytes_per_s)
        # Prefill uses the pipeline sum for BOTH split modes -- see the
        # docstring: the rig measured tensor prefill *slower*, so a parallel
        # roofline here would advertise a win that does not exist.
        t_prompt_raw += (2.0 * params_active * share) / perf.flops_fp16

    if not split:
        # No placement to reason about. Say so rather than dividing by zero.
        return _unknown_result(active, params_active, eff_decode, eff_prompt)

    # Devices run in sequence under a layer split and together under a tensor
    # split, so the times add in one case and the slowest device sets the pace
    # in the other.
    reduce = max if tensor_parallel else sum
    t_weights_raw = float(reduce(weight_times))
    t_kv = float(reduce(kv_times))
    t_sync = (
        max(0, int(getattr(meta, "n_layer", 0) or 0)) * T_TENSOR_SYNC_S if tensor_parallel else 0.0
    )

    t_weights = t_weights_raw / eff_decode
    t_token = t_weights + t_kv + t_sync + T_TOKEN_OVERHEAD_S
    if t_token <= 0:  # pragma: no cover - the overhead floor makes this unreachable
        return _unknown_result(active, params_active, eff_decode, eff_prompt)

    gen_tps = max(0.0, efficiency) / t_token

    slots = max(1, int(parallel))
    if knee is not None and knee >= 1:
        slots = min(slots, int(knee))
    t_step = t_weights + slots * t_kv + t_sync + T_TOKEN_OVERHEAD_S
    gen_tps_batched = (max(0.0, efficiency) * slots / t_step) if t_step > 0 else 0.0
    # A batch can never be slower in aggregate than a single stream.
    gen_tps_batched = max(gen_tps_batched, gen_tps)

    if params_active > 0 and t_prompt_raw > 0:
        prompt_tps = max(0.0, prompt_efficiency) * eff_prompt / t_prompt_raw
    else:
        prompt_tps = 0.0

    return {
        "prompt_tps": round(prompt_tps, 1) if prompt_tps > 0 else None,
        "gen_tps": round(gen_tps, 1) if gen_tps > 0 else None,
        "gen_tps_batched": round(gen_tps_batched, 1) if gen_tps_batched > 0 else None,
        "basis": {
            "devices": sorted(split),
            "gpu_labels": [perf_by_device[i] for i in sorted(split)],
            "active_weight_bytes": active,
            "active_params": params_active,
            "kv_read_bytes_per_slot": kv_per_slot,
            "eff_decode": round(eff_decode, 4),
            "eff_prompt": round(eff_prompt, 4),
            "decode_calibration": round(efficiency, 3),
            "prompt_calibration": round(prompt_efficiency, 3),
            "t_token_s": t_token,
            "t_overhead_s": T_TOKEN_OVERHEAD_S,
            "t_tensor_sync_s": t_sync,
            "split_mode": split_mode,
            "slots_used": slots,
            "is_moe": moe,
            "estimator_version": ESTIMATOR_VERSION,
        },
    }


def _unknown_result(
    active: int, params_active: int, eff_decode: float, eff_prompt: float
) -> dict[str, Any]:
    """The "cannot estimate" shape: nulls, not zeros, and the reason.

    Same ``basis`` keys as a real result, so a consumer never has to branch on
    which shape it got before reading a field.
    """
    return {
        "prompt_tps": None,
        "gen_tps": None,
        "gen_tps_batched": None,
        "basis": {
            "devices": [],
            "gpu_labels": [],
            "active_weight_bytes": active,
            "active_params": params_active,
            "kv_read_bytes_per_slot": 0,
            "eff_decode": round(eff_decode, 4),
            "eff_prompt": round(eff_prompt, 4),
            "decode_calibration": 1.0,
            "prompt_calibration": 1.0,
            "t_token_s": 0.0,
            "t_overhead_s": T_TOKEN_OVERHEAD_S,
            "t_tensor_sync_s": 0.0,
            "split_mode": "layer",
            "slots_used": 0,
            "is_moe": False,
            "estimator_version": ESTIMATOR_VERSION,
            "unavailable": "no device placement to estimate from",
        },
    }


def knee_for(
    meta: Any,
    weights_bytes: int,
    kv_budget_bytes: int,
    kv_per_token: int,
    ctx_per_slot: int,
    *,
    cap: int = MAX_PARALLEL_CAP,
) -> tuple[int, str]:
    """Thin pass-through to the planner's bound, so both agree by construction."""
    return max_parallel_for(
        kv_budget_bytes=kv_budget_bytes,
        kv_per_token=kv_per_token,
        ctx_per_slot=ctx_per_slot,
        active_weight_bytes=active_weight_bytes(meta, weights_bytes),
        is_moe=is_moe(meta),
        cap=cap,
    )


# ---------------------------------------------------------------------------
# Metrics scraping
# ---------------------------------------------------------------------------

#: The llama-server counters the collector reads. Counters (monotonic totals)
#: are preferred over the ``*_seconds`` gauges because a gauge averages over
#: the child's whole lifetime -- it cannot tell "fast now" from "was fast an
#: hour ago". Two samples of a counter give the rate *between* them.
METRIC_PROMPT_TOKENS = "llamacpp:prompt_tokens_total"
METRIC_PROMPT_SECONDS = "llamacpp:prompt_seconds_total"
METRIC_PREDICTED_TOKENS = "llamacpp:tokens_predicted_total"
METRIC_PREDICTED_SECONDS = "llamacpp:tokens_predicted_seconds_total"
METRIC_DECODE_TOTAL = "llamacpp:n_decode_total"
METRIC_BUSY_SLOTS = "llamacpp:n_busy_slots_per_decode"
METRIC_REQUESTS_DEFERRED = "llamacpp:requests_deferred"
METRIC_REQUESTS_PROCESSING = "llamacpp:requests_processing"

#: Gauges reported straight through (no delta), for ``/api/status``.
GAUGE_METRICS: tuple[str, ...] = (
    METRIC_BUSY_SLOTS,
    METRIC_REQUESTS_DEFERRED,
    METRIC_REQUESTS_PROCESSING,
)


def parse_metrics(text: str | None) -> dict[str, float]:
    """Parse a Prometheus text exposition into ``{metric_name: value}``.

    Deliberately tiny and total: comments are skipped, labels are stripped, and
    anything unparseable is dropped rather than raised. A metrics endpoint that
    changes shape must degrade to "no data", never to an exception on a
    background timer.
    """
    out: dict[str, float] = {}
    if not text:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        if not name or not value:
            continue
        # Strip Prometheus labels: `name{a="b"} 1.0`. llama.cpp emits none
        # today, but a future release adding them must not silently stop the
        # collector.
        brace = name.find("{")
        if brace != -1:
            name = name[:brace]
        try:
            parsed = float(value.strip())
        except ValueError:
            continue
        if math.isfinite(parsed):
            out[name] = parsed
    return out


def sample_between(
    previous: Mapping[str, float],
    current: Mapping[str, float],
    *,
    elapsed_s: float,
) -> dict[str, Any] | None:
    """Tokens/second between two ``/metrics`` scrapes, or ``None``.

    Returns ``None`` -- meaning "nothing worth recording" -- when the child did
    no work between the samples, when a counter went backwards (the child
    restarted and reset its counters), or when the metrics are missing
    entirely. Recording a zero in any of those cases would poison the median
    that calibration is built on.
    """
    if not previous or not current:
        return None

    decodes = current.get(METRIC_DECODE_TOTAL, 0.0) - previous.get(METRIC_DECODE_TOTAL, 0.0)
    if decodes <= 0:
        return None  # idle, or the counters reset

    gen_tokens = current.get(METRIC_PREDICTED_TOKENS, 0.0) - previous.get(
        METRIC_PREDICTED_TOKENS, 0.0
    )
    gen_seconds = current.get(METRIC_PREDICTED_SECONDS, 0.0) - previous.get(
        METRIC_PREDICTED_SECONDS, 0.0
    )
    prompt_tokens = current.get(METRIC_PROMPT_TOKENS, 0.0) - previous.get(METRIC_PROMPT_TOKENS, 0.0)
    prompt_seconds = current.get(METRIC_PROMPT_SECONDS, 0.0) - previous.get(
        METRIC_PROMPT_SECONDS, 0.0
    )

    gen_tps = gen_tokens / gen_seconds if gen_seconds > 0 and gen_tokens > 0 else None
    prompt_tps = (
        prompt_tokens / prompt_seconds if prompt_seconds > 0 and prompt_tokens > 0 else None
    )
    if gen_tps is None and prompt_tps is None:
        return None

    return {
        "prompt_tps": round(prompt_tps, 2) if prompt_tps else None,
        "gen_tps": round(gen_tps, 2) if gen_tps else None,
        "n_busy_slots": current.get(METRIC_BUSY_SLOTS),
        "requests_deferred": current.get(METRIC_REQUESTS_DEFERRED),
        "sample_s": round(max(0.0, elapsed_s), 2),
        "decodes": int(decodes),
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

#: Below this many matching observations the median is noise, so the next
#: fallback is used instead. Two is enough to reject a single freak sample
#: without waiting hours for data on a rarely-used model.
CALIBRATION_MIN_ROWS = 2

#: Clamp on the learned factor. A ratio outside this range means the estimate
#: and the measurement are describing different things (a mis-parsed metric, a
#: model that was swapped underneath the id), and applying it would make the
#: catalog confidently wrong rather than roughly right. Same reasoning as the
#: VRAM calibration clamp in D18.
CALIBRATION_MIN = 0.1
CALIBRATION_MAX = 3.0


def _ratios(rows: Sequence[Mapping[str, Any]], measured: str, predicted: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        got = row.get(measured)
        want = row.get(predicted)
        if not isinstance(got, int | float) or not isinstance(want, int | float):
            continue
        if got <= 0 or want <= 0:
            continue
        ratio = float(got) / float(want)
        if CALIBRATION_MIN <= ratio <= CALIBRATION_MAX:
            out.append(ratio)
    return out


def _device_count(row: Mapping[str, Any]) -> int:
    """How many devices a row's ``devices`` string names ("0,1,2,3" -> 4)."""
    return len([part for part in str(row.get("devices") or "").split(",") if part.strip()])


def calibrate(
    observations: Sequence[Mapping[str, Any]],
    *,
    model_id: str | None = None,
    devices: Sequence[int] | None = None,
    gpu_class_label: str | None = None,
    is_moe: bool | None = None,
    peer_moe: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Learn ``measured / estimated`` factors from recorded observations.

    **Only rows recorded by the current :data:`ESTIMATOR_VERSION` contribute a
    ratio.** A ratio is a correction to one specific formula; applying one
    learned against a formula that has since been fixed teaches the estimator
    the difference between two dead arithmetics. Older rows (and rows with no
    version at all, written before the column existed) are ignored here and
    still read by :func:`measured_for` -- a measurement does not expire when our
    arithmetic changes.

    Four tiers, most specific first, because the same model behaves differently
    on a different placement and a different model behaves differently on the
    same one:

    1. ``"model+devices"`` -- this model on this exact device set.
    2. ``"model"`` -- this model anywhere. A placement change moves the number,
       but far less than a *model* change does, so this sits above the peer
       tier. It is new in v2: without it, a model measured only on ``0,1`` got
       nothing at all when the catalog asked about ``0``, and fell through to
       whatever the neighbours were doing.
    3. ``"peers"`` -- *other* models on this hardware class with the same
       density (dense vs MoE, via ``peer_moe[model_id]``) and the same device
       *count*. Taken as the **median of per-model medians**, each contributing
       model needing its own :data:`CALIBRATION_MIN_ROWS`. Never a raw pool
       median: the live rig holds 84 rows for one MoE and 3 for one dense model,
       so a pooled median *is* the MoE's number, and applying a sparse-MoE
       derate to a dense model is exactly the cross-contamination that made
       every Gemma-4 row read 0.411.
    4. ``"none"`` -- 1.0, and the caller reports ``"estimated"``.

    The **median** rather than the mean throughout: one sample taken while
    another process was hammering the same GPU is an outlier, and a mean lets it
    move the answer for every model on the box.

    Args:
        is_moe: whether the model being calibrated is a MoE. ``None`` means
            "unknown", which disables the density filter rather than guessing.
        peer_moe: ``{model_id: is_moe}`` for the models in ``observations``. A
            peer missing from the map is skipped when a density is asked for:
            an unknown density is not evidence.

    Returns ``{"gen", "prompt", "basis", "samples", "models"}`` where ``models``
    is the number of distinct models that contributed.
    """
    rows = [row for row in observations if row.get("estimator_version") == ESTIMATOR_VERSION]
    device_key = ",".join(str(d) for d in sorted(devices)) if devices is not None else None
    want_devices = len(devices) if devices is not None else None

    mine = [row for row in rows if model_id is None or row.get("model_id") == model_id]

    exact = (
        mine
        if device_key is None
        else [row for row in mine if str(row.get("devices") or "") == device_key]
    )
    for pool, basis in ((exact, "model+devices"), (mine, "model")):
        gen = _ratios(pool, "gen_tps", "est_gen_tps")
        if len(gen) < CALIBRATION_MIN_ROWS:
            continue
        prompt = _ratios(pool, "prompt_tps", "est_prompt_tps")
        return {
            "gen": statistics.median(gen),
            "prompt": statistics.median(prompt) if len(prompt) >= CALIBRATION_MIN_ROWS else 1.0,
            "basis": basis,
            "samples": len(gen),
            "models": 1,
        }

    if gpu_class_label:
        by_model: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            peer_id = str(row.get("model_id") or "")
            if not peer_id or peer_id == model_id:
                continue
            if row.get("gpu_class") != gpu_class_label:
                continue
            if want_devices is not None and _device_count(row) != want_devices:
                continue
            if is_moe is not None:
                peer_flag = (peer_moe or {}).get(peer_id)
                if peer_flag is None or bool(peer_flag) != bool(is_moe):
                    continue
            by_model.setdefault(peer_id, []).append(row)

        gen_medians: list[float] = []
        prompt_medians: list[float] = []
        samples = 0
        for peer_id in sorted(by_model):
            peer_rows = by_model[peer_id]
            gen = _ratios(peer_rows, "gen_tps", "est_gen_tps")
            if len(gen) < CALIBRATION_MIN_ROWS:
                continue
            gen_medians.append(statistics.median(gen))
            samples += len(gen)
            prompt = _ratios(peer_rows, "prompt_tps", "est_prompt_tps")
            if len(prompt) >= CALIBRATION_MIN_ROWS:
                prompt_medians.append(statistics.median(prompt))
        if gen_medians:
            return {
                "gen": statistics.median(gen_medians),
                "prompt": statistics.median(prompt_medians) if prompt_medians else 1.0,
                "basis": "peers",
                "samples": samples,
                "models": len(gen_medians),
            }

    return {"gen": 1.0, "prompt": 1.0, "basis": "none", "samples": 0, "models": 0}


def measured_for(
    observations: Sequence[Mapping[str, Any]],
    *,
    devices: Sequence[int],
    ctx_size: int,
    parallel: int | None = None,
) -> dict[str, float | None]:
    """The best measured numbers for one catalog row, or nulls.

    Exact ``(devices, ctx_size)`` first, then the same devices at any context.
    A measurement taken at a different context is still evidence about the
    hardware, but it is not evidence about *this row*, so it is only used when
    nothing better exists and the caller downgrades the confidence with it.
    """
    device_key = ",".join(str(d) for d in sorted(devices))
    same_devices = [row for row in observations if str(row.get("devices") or "") == device_key]
    exact = [row for row in same_devices if int(row.get("ctx_size") or 0) == int(ctx_size)]
    if parallel is not None:
        narrowed = [row for row in exact if int(row.get("parallel") or 0) == int(parallel)]
        if narrowed:
            exact = narrowed
    pool = exact or same_devices
    if not pool:
        return {"gen_tps": None, "prompt_tps": None, "exact": None}

    gen = [float(r["gen_tps"]) for r in pool if isinstance(r.get("gen_tps"), int | float)]
    prompt = [float(r["prompt_tps"]) for r in pool if isinstance(r.get("prompt_tps"), int | float)]
    return {
        "gen_tps": round(statistics.median(gen), 1) if gen else None,
        "prompt_tps": round(statistics.median(prompt), 1) if prompt else None,
        "exact": bool(exact),
    }


def confidence_for(measured: Mapping[str, Any], calibration: Mapping[str, Any]) -> str:
    """``"measured"`` | ``"calibrated"`` | ``"estimated"`` for one row.

    ``"measured"`` is reserved for a row whose exact placement and context were
    actually observed -- an agent reading the catalog should be able to trust
    that word literally.
    """
    if measured.get("exact") and measured.get("gen_tps"):
        return "measured"
    if calibration.get("basis") not in (None, "none"):
        return "calibrated"
    return "estimated"
