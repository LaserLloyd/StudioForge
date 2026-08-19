"""How much quality a KV cache quantization costs, per model family.

The planner and the catalog both have to answer "is a smaller KV cache worth
the context it buys?", and until now neither of them looked at quality at all:
the auto ladder walked ``f16 -> q8_0 -> q4_0`` and the catalog recommended
whatever reached the largest window, so a 262144-token row on a **q4_0** cache
outranked a 131072-token row on f16. That is the wrong trade for an agent host
-- a doubled window that answers slightly worse is a worse server -- and it is
wrong by measurable amounts that differ by family.

**The two asymmetries that decide the policy.**

*K is the sensitive cache, V is not.* llama.cpp discussion #23470 measures each
side alone: with a q4_0 **K** cache and an f16 V cache, Qwen2.5-7B reproduces
only **11.7%** of its f16 tokens; with q4_0 **V** and f16 K the output is
essentially unchanged, and the matched ``q8_0/q8_0`` pair sits at a KL
divergence of **0.0018**. So the useful third rung is ``q8_0`` K with ``q4_0``
V -- not the symmetric ``q4_0/q4_0`` this code used to offer. **A q4_0 K cache
is never chosen automatically anywhere in StudioForge**; it remains reachable
only by a user setting it explicitly, which is the same rule the rest of the
planner applies to explicit values.

*Families differ, by a factor of ten.* The localbench KV-quantization benchmark
(``localbench.substack.com/p/kv-cache-quantization-benchmark``; KL divergence
over top-40 logprobs, ~250k tokens, against a BF16 GGUF with an f16 KV cache)
measures, at ``q8_0`` and ``q4_0``:

| Family | q8_0 KV | q4_0 KV | Verdict |
| --- | --- | --- | --- |
| Gemma-4 31B dense (``gemma4``) | 0.108 | 0.524 | sensitive |
| Gemma-4 26B-A4B MoE (``gemma4``) | 0.377 | 1.088 | sensitive |
| Qwen 3.6 (``qwen35`` / ``qwen35moe``) | 0.024 | 0.039 | tolerant |

Gemma's iSWA layout is the plausible reason: only every sixth layer holds the
full window, so each surviving KV element carries far more of the model's
memory of the conversation and quantizing it costs proportionally more.

**The policy that follows**, applied by :func:`allows_q8`:

* *sensitive* family: ``q8_0`` only when ``f16`` cannot reach the context floor
  on that hardware. Quality first; a bigger window is not a reason.
* *tolerant* family: ``q8_0`` is also allowed when it buys at least one full
  doubling of context beyond the floor -- 0.024 KL is inside the noise of a
  sampler, and 2x the window is a real capability.
* *unknown* family: treated as sensitive. The numbers above are three
  measurements on two families, and assuming a model behaves like the tolerant
  one because nobody measured it is exactly the direction that produces a
  quietly worse server.

Nothing here is a VRAM figure; :mod:`studioforge.core.planner` owns those. This
module only ranks *quality*, so a change in one cannot silently move the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from studioforge.types import KvCacheType

#: KV (K, V) pairs the planner may choose **automatically**, best quality
#: first. ``q4_0`` never appears in the K position: see the module docstring.
#: An explicit per-model or per-request ``kv_cache_type`` bypasses this list
#: entirely, because an explicit value is honoured verbatim everywhere else in
#: the planner and a quality policy is not a reason to start ignoring one.
KV_QUALITY_LADDER: tuple[tuple[KvCacheType, KvCacheType], ...] = (
    ("f16", "f16"),
    ("q8_0", "q8_0"),
    ("q8_0", "q4_0"),
)


def kv_quality_rank(kv_k: str, kv_v: str) -> int:
    """Position of a (K, V) pair in :data:`KV_QUALITY_LADDER`; lower is better.

    A pair outside the ladder -- an explicit ``q4_0`` K, an ``f32`` cache --
    ranks after everything in it, so a comparison never silently promotes a
    combination this module has no measurement for.
    """
    pair = (kv_k, kv_v)
    for index, candidate in enumerate(KV_QUALITY_LADDER):
        if candidate == pair:
            return index
    return len(KV_QUALITY_LADDER)


def kv_quality_label(kv_k: str, kv_v: str) -> str:
    """``"f16"`` / ``"q8_0"`` / ``"q8_0 K + q4_0 V"`` -- how a basis string says it."""
    if kv_k == kv_v:
        return str(kv_k)
    return f"{kv_k} K + {kv_v} V"


@dataclass(frozen=True)
class KvSensitivity:
    """What quantizing this family's KV cache is known to cost."""

    #: Architecture family this describes, as the GGUF spells it.
    family: str
    #: True when ``q8_0`` is only acceptable as a last resort (see the module
    #: docstring). Unknown families are sensitive by construction.
    sensitive: bool
    #: KL divergence at ``q8_0`` KV against an f16 reference, or ``None`` when
    #: nothing was measured for this family.
    kl_q8: float | None = None
    #: The same at ``q4_0`` KV.
    kl_q4: float | None = None
    #: Why, in one sentence, for a docstring or a GUI tooltip.
    note: str = ""


#: Measured families, keyed by the ``general.architecture`` prefix the GGUF
#: carries. Matched as a prefix so ``gemma4`` covers both the dense and the MoE
#: Gemma-4 (D22 established that the architecture string does not distinguish
#: them, and here it does not need to: both measure sensitive).
FAMILY_SENSITIVITY: tuple[tuple[str, KvSensitivity], ...] = (
    (
        "gemma",
        KvSensitivity(
            family="gemma",
            sensitive=True,
            kl_q8=0.108,
            kl_q4=0.524,
            note=(
                "Gemma 3/4 lose measurable quality to a quantized KV cache "
                "(KL 0.108 dense / 0.377 MoE at q8_0, 0.524 / 1.088 at q4_0), "
                "plausibly because iSWA keeps only every sixth layer at full "
                "context so each retained element carries more of the state"
            ),
        ),
    ),
    (
        "qwen35",
        KvSensitivity(
            family="qwen35",
            sensitive=False,
            kl_q8=0.024,
            kl_q4=0.039,
            note=(
                "Qwen 3.6 tolerates a q8_0 KV cache (KL 0.024, inside sampler "
                "noise), so q8_0 is worth taking when it buys a context doubling"
            ),
        ),
    ),
)

#: What an unmeasured family gets. Sensitive, deliberately: three measurements
#: on two families do not describe a library of forty models, and guessing
#: "tolerant" is the guess whose failure mode is a server that quietly answers
#: worse.
UNKNOWN_SENSITIVITY = KvSensitivity(
    family="unknown",
    sensitive=True,
    note=(
        "no KV-quantization measurement exists for this architecture, so it is "
        "treated as sensitive: f16 unless f16 cannot reach the context floor"
    ),
)


def sensitivity_for(meta: Any) -> KvSensitivity:
    """The family's KV sensitivity, from GGUF metadata (or an architecture string).

    Accepts a :class:`~studioforge.types.GgufMeta`, anything with an
    ``architecture`` attribute, a plain string, or ``None`` -- every catalog
    surface has a slightly different handle on the same fact, and making each
    of them unwrap it would be four chances to unwrap it differently.
    """
    if meta is None:
        return UNKNOWN_SENSITIVITY
    arch = meta if isinstance(meta, str) else getattr(meta, "architecture", None)
    name = str(arch or "").lower()
    if not name:
        return UNKNOWN_SENSITIVITY
    for prefix, entry in FAMILY_SENSITIVITY:
        if name.startswith(prefix):
            return entry
    return UNKNOWN_SENSITIVITY


def allows_q8(meta: Any, *, f16_reaches_floor: bool, buys_doubling: bool) -> bool:
    """Whether a ``q8_0`` cache may be chosen *automatically* for this model.

    Args:
        meta: the model's GGUF metadata (or architecture name).
        f16_reaches_floor: whether an ``f16`` cache reaches the context floor on
            the hardware being considered. When it does not, quantizing is the
            difference between serving the model and not serving it, and every
            family allows it.
        buys_doubling: whether the ``q8_0`` option reaches at least twice the
            context the ``f16`` option does. Only a tolerant family may spend
            quality on that.
    """
    if not f16_reaches_floor:
        return True
    return (not sensitivity_for(meta).sensitive) and buys_doubling
