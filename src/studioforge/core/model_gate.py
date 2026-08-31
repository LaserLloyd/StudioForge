"""D52: the local-model gate -- "is what is already loaded good enough for my job?"

This is **not** :mod:`studioforge.core.capabilities`. That module answers "what
can this *box* run" -- which llama.cpp build is installed, which architectures
and quantizations it knows, whether a newer engine exists. This one answers a
much smaller, much more frequently asked question about the *current residency*:
given a bar ("at least 20B", "must do vision", "must be uncensored"), is a model
that clears it loaded **right now**, and if so what id do I put in my request?

The two are deliberately separate because they have different consumers and
different lifetimes: engine capability changes when the operator installs a
build, residency changes minute to minute.

Why the answer carries a model id
---------------------------------
The caller is an agent -- an OpenClaw skill, a ClawChat bot -- standing at a
three-way fork: send this work to the local ``/v1`` API, spend 40 seconds
loading something bigger, or pay a cloud provider. A bare yes/no cannot be acted
on: knowing "yes, something big enough is loaded" without knowing *which* one
still costs the caller a second round trip to ``GET /v1/models`` plus the
guesswork of picking between three resident models. So the gate returns the id
to use, its size, and what it can do. A yes you cannot act on is useless.

Unknown is not a pass
---------------------
Every judgement here fails closed. The gate's promise is "at least this big" /
"this capability is present"; a model we cannot *size* cannot make the first
promise, and a tag we cannot *verify* cannot make the second. So an unsized
model fails a ``min_params`` bar and an unverifiable tag fails a tag bar, both
with a ``why_not`` that says which of the two happened ("params 4.0B < 20.0B" is
a different operational problem from "cannot verify 'uncensored'"). The cost of
a false negative is one unnecessary load; the cost of a false positive is an
agent sending vision work to a text-only model and getting garbage back.

What this module reuses rather than re-derives
----------------------------------------------
* Parameter counts come from :mod:`studioforge.core.throughput` -- its
  ``BITS_PER_WEIGHT`` table, ``bits_per_weight()`` prefix matching and
  ``active_params()`` MoE trunk model. A second bits-per-weight table in this
  package would be a second thing to keep calibrated, and the estimator's is
  already measured. (Note its unknown-quant default is 5.0 bits, "between Q4 and
  Q5, which is where most of a real library lives", not the 4.85 a Q4_K_M-shaped
  guess would use. The difference is ~3% on a number the operator's own framing
  called "approx".)
* Modalities come from :func:`modalities_from`, which is also what
  ``gui.state.modalities_text`` renders on the Dashboard, so the gate and the
  screen can never disagree about whether a running child accepts images.
* ``general.parameter_count`` is *already* parsed into ``GgufMeta.param_count``
  by :mod:`studioforge.core.gguf`, so the exact-count path costs nothing.

Future work: ``general.size_label`` (the "27B"/"8x7B" string many conversions
embed) is **not** currently retained on ``GgufMeta``. Adding it would mean
bumping ``META_FORMAT_VERSION`` and rescanning the whole library, which is not
worth it while the name parser below covers every model on this rig.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from studioforge.core import throughput
from studioforge.logging import get_logger
from studioforge.types import ModelRecord

log = get_logger(__name__)

#: A per-tag judgement. ``"unknown"`` is a first-class answer, not an error: it
#: means no local source could speak to this tag, which is materially different
#: from a source that answered "no". Requirements pass on ``"yes"`` only.
Verdict = Literal["yes", "no", "unknown"]

#: What a caller sees when the gate cannot size a model at all.
UNKNOWN_SOURCE = "unknown"


# ---------------------------------------------------------------------------
# The bar
# ---------------------------------------------------------------------------


#: Accepted spellings for ``min_params``, quoted back in the 400 so a caller who
#: guessed wrong is told the answer rather than left to experiment.
_MIN_PARAMS_SHAPES = (
    "a number in billions (20, 1.5), "
    "or a string with a unit ('20b', '500m', '0.5b', '7000m'), "
    "or omitted for no size bar"
)

_MIN_PARAMS_RE = re.compile(r"^(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[bm])?$", re.IGNORECASE)


def parse_min_params(value: str | int | float | None) -> float | None:
    """Normalise a size bar to **billions of parameters**.

    Billions is the unit of this entire domain -- nobody says "I need a
    7,000,000,000 parameter model", they say 7B -- so a bare number is read as
    billions and *never* as a raw count. ``parse_min_params(20)`` is 20.0
    billion, not 20 parameters. This is the one piece of implicit behaviour in
    the module and it is why it is documented this loudly.

    Accepts ``None`` (no bar), a number, or a string with an optional ``b``/``m``
    suffix; ``"500m"`` is 0.5 and ``"7000m"`` is 7.0, because a caller who thinks
    in millions should not have to convert. Surrounding and internal whitespace
    is tolerated ("20 B").

    Raises:
        ValueError: naming the accepted shapes. The REST layer turns this into
            the project's standard 400 rather than a 500, so a typo in an
            agent's config is a legible error and not an outage.
    """
    if value is None:
        return None
    # bool is an int subclass, and `min_params=True` is a caller mistake (almost
    # certainly a boolean flag sent to the wrong parameter), not a 1B bar.
    if isinstance(value, bool):
        raise ValueError(f"min_params must be {_MIN_PARAMS_SHAPES}; got a boolean")
    if isinstance(value, int | float):
        number = float(value)
        if number < 0:
            raise ValueError(f"min_params cannot be negative; expected {_MIN_PARAMS_SHAPES}")
        # 0 is the natural programmatic spelling of "no bar" (a config default,
        # a slider at rest). Treating it as a literal bar produced the absurd
        # refusal "cannot prove >= 0.0B" for a model whose size is unknown --
        # every model has at least zero parameters, so a zero bar IS no bar.
        return number or None

    text = str(value).strip()
    if not text:
        return None
    match = _MIN_PARAMS_RE.match(text.replace(" ", ""))
    if match is None:
        raise ValueError(f"cannot read min_params {value!r}; expected {_MIN_PARAMS_SHAPES}")
    number = float(match.group("num"))
    unit = (match.group("unit") or "b").lower()
    scaled = number / 1000.0 if unit == "m" else number
    return scaled or None  # "0", "0b", "0m": same no-bar reading as numeric zero


#: Tags the REST/MCP layers expose as named booleans. They are sugar: every one
#: of them lands in the same ``tags`` set the free-form list feeds, so the
#: evaluator has exactly one code path to reason about.
SUGAR_TAGS: tuple[str, ...] = ("vision", "audio", "tools", "thinking", "uncensored")

#: A tag longer than this, or carrying control characters, is not a tag anybody
#: meant to send -- it is a mangled query string. Everything else is allowed
#: through: the generic matcher (below) is what makes new tags free, and
#: rejecting unfamiliar spellings would defeat that.
_MAX_TAG_LEN = 64


def parse_tags(raw: str | None = None, *, extra: Iterable[str] = ()) -> frozenset[str]:
    """Normalise a comma-separated tag list plus any sugar flags into one set.

    Lowercased, stripped, de-duplicated. Empty elements are dropped **silently**
    -- ``"vision,,"`` is a trailing comma, not an error, and 400ing on it would
    punish string interpolation for no safety gain.

    Raises:
        ValueError: only for a value that cannot be a tag at all (too long, or
            containing control characters), which means the query string was
            mangled rather than that the caller wanted something exotic.
    """
    out: set[str] = set()
    for candidate in [*(raw or "").split(","), *extra]:
        tag = str(candidate).strip().lower()
        if not tag:
            continue
        if len(tag) > _MAX_TAG_LEN or any(ch < " " or ch == "\x7f" for ch in tag):
            raise ValueError(f"tag {candidate!r} is not a usable tag name")
        out.add(tag)
    return frozenset(out)


@dataclass(frozen=True)
class GateRequirement:
    """The bar a caller wants cleared, as one immutable value.

    ``tags`` holds *both* the canonical capabilities (vision, tools, thinking,
    audio) and free-form ones (uncensored, coding, roleplay, or anything the
    caller invents). Collapsing them into a single set is what keeps the
    evaluator open to new tags without a code change per tag -- see
    :func:`tag_verdict`.
    """

    min_params_b: float | None = None
    tags: frozenset[str] = field(default_factory=frozenset)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


#: A parameter-count token inside a model id.
#:
#: This deliberately uses boundary lookarounds rather than splitting the id on
#: ``[-_/ .]``, because a literal split on ``.`` destroys the very case it has
#: to get right: ``SmolLM2-1.7B`` would split into ``1`` and ``7B`` and report a
#: 7B model as passing a 5B bar. The lookarounds treat ``.`` as a separator
#: *except* between digits, which is exactly the intent "whole tokens only" was
#: reaching for.
#:
#: Groups:
#:   ``mult``   -- the ``8`` of ``8x7B`` (a MoE spelled as expert count x size)
#:   ``prefix`` -- ``A`` for the active/routed count of an MoE (``A10B``),
#:                 ``E`` for an "effective" size (Gemma 3n's ``E4B``)
#:   ``num``    -- the number itself, decimals included
#:
#: What it must NOT match is the whole reason for the lookarounds: quantization
#: labels (``Q8_0``, ``Q4_K_M``), engine builds (``b10689``), tensor types
#: (``BF16``, ``f16``), vendor suffixes (``hb16``) and version tags (``v2.0``)
#: all live in these ids and all contain digits.
_SIZE_TOKEN = re.compile(
    r"(?<![0-9A-Za-z.])"  # start of a token (a bare `.` is a separator...
    r"(?:(?P<mult>\d{1,3})\s*[xX]\s*)?"
    r"(?P<prefix>[AaEe])?"
    r"(?P<num>\d+(?:\.\d+)?)"  # ...but not one inside a number)
    r"[Bb]"
    r"(?![0-9A-Za-z.])"  # end of a token
)


def parse_size_tokens(model_id: str) -> tuple[float | None, float | None]:
    """``(total_B, active_B)`` read out of a model id, or ``(None, None)``.

    Model ids on this rig are HuggingFace repo paths with the size in the name,
    which makes the name a more reliable size source than any file arithmetic --
    and unlike a GGUF header read it costs nothing. ``122B`` plus ``A10B`` in
    one id is an MoE and yields both numbers; ``8x7B`` multiplies out to 56.

    The **first** total token wins rather than the largest. Ids routinely repeat
    themselves (``.../Dark-Scarlett-27B-v2.0-GGUF/Dark-Scarlett-27B-...``) so in
    practice they agree; where they do not, the leftmost is the one a human
    reads as the model's own size, and preferring the largest would bias the
    error towards *overstating* a model -- the unsafe direction for a gate whose
    promise is "at least this big".
    """
    total: float | None = None
    active: float | None = None
    for match in _SIZE_TOKEN.finditer(model_id or ""):
        number = float(match.group("num"))
        prefix = (match.group("prefix") or "").lower()
        if prefix == "a":
            if active is None:
                active = number
            continue
        # No prefix, or `E` for an effective size: both describe the whole model.
        multiplier = int(match.group("mult") or 1)
        if total is None:
            total = number * multiplier
    return total, active


class _QuantCarrier:
    """Minimal stand-in exposing ``quant_label`` for :func:`throughput.bits_per_weight`.

    Needed because the quant label lives on ``meta.quant_label`` for a scanned
    record but on ``record.quant`` when the metadata could not be parsed, and
    the shared bits-per-weight lookup (with its longest-prefix matching for
    vendor suffixes) reads only the former.
    """

    __slots__ = ("quant_label",)

    def __init__(self, label: str) -> None:
        self.quant_label = label


def _record_facts(record: ModelRecord | str | None) -> tuple[str, Any, int, str]:
    """``(model_id, meta, weight_bytes, quant_label)`` from a record or a bare id."""
    if record is None:
        return "", None, 0, ""
    if isinstance(record, str):
        return record, None, 0, ""
    meta = getattr(record, "meta", None)
    # tensor_bytes excludes the GGUF header and any padding, so it is the truer
    # divisor when present; size_bytes is the fallback. Same choice catalog.py
    # makes, kept in step deliberately.
    weights = int(getattr(meta, "tensor_bytes", 0) or 0) or int(getattr(record, "size_bytes", 0))
    return str(record.id), meta, weights, str(getattr(record, "quant", "") or "")


def _moe_active_b(meta: Any, weights_bytes: int, total_count: int) -> float | None:
    """Routed (per-token) parameters in billions, or ``None`` for a dense model.

    ``None`` rather than "same as total" for dense models, so that
    ``active_params_b`` means one thing across all three sources: *this is an
    MoE and here is its active count*. The name parser can only ever say that
    (an ``A10B`` token), so the metadata and estimate paths say it the same way.
    """
    if meta is None or total_count <= 0:
        return None
    n_expert = int(getattr(meta, "n_expert", 0) or 0)
    n_used = int(getattr(meta, "n_expert_used", 0) or 0)
    if not (n_expert > 1 and 0 < n_used < n_expert):
        return None
    active = throughput.active_params(meta, weights_bytes)
    if active <= 0 or active >= total_count:
        return None
    return round(active / 1e9, 1)


def approx_params_b(
    record: ModelRecord | str | None, live_params: int | None = None
) -> tuple[float | None, float | None, str]:
    """``(total_B, active_B, source)`` -- how big is this model, and how do we know?

    Three layered sources, first hit wins, each labelled so a caller can tell an
    exact count from an educated guess:

    ``"metadata"``
        An exact count: ``live_params`` (a raw count a running child reported)
        or ``general.parameter_count``, which :mod:`studioforge.core.gguf`
        already parses into ``GgufMeta.param_count``.
    ``"name"``
        :func:`parse_size_tokens` on the model id. Covers every model on this
        rig, including the ones whose GGUF omits the count.
    ``"estimated"``
        File bytes divided by the quantization's bits-per-weight, via
        :mod:`studioforge.core.throughput`'s measured table. Deliberately
        approximate -- the operator's framing for this whole feature was "approx
        parameters" -- and lands within a few percent.

    ``record`` accepts a bare model id string as well as a record: a supervisor
    instance whose registry entry has gone missing (a file deleted under a
    running child) must still be sizeable from its name rather than silently
    becoming unknown, which would fail every bar.

    Returns ``(None, None, "unknown")`` when nothing can speak to the size.
    """
    model_id, meta, weights, quant = _record_facts(record)

    declared = int(live_params or 0)
    if declared <= 0 and meta is not None:
        declared = int(getattr(meta, "param_count", 0) or 0)
    if declared > 0:
        return (
            round(declared / 1e9, 1),
            _moe_active_b(meta, weights, declared),
            "metadata",
        )

    total, active = parse_size_tokens(model_id)
    if total is not None or active is not None:
        return (
            round(total, 1) if total is not None else None,
            round(active, 1) if active is not None else None,
            "name",
        )

    if weights > 0:
        label = str(getattr(meta, "quant_label", "") or "") or quant
        bits = throughput.bits_per_weight(_QuantCarrier(label))
        if bits > 0:
            count = int(weights * 8 / bits)
            return (
                round(count / 1e9, 1),
                _moe_active_b(meta, weights, count),
                "estimated",
            )

    return None, None, UNKNOWN_SOURCE


# ---------------------------------------------------------------------------
# Modalities and tags
# ---------------------------------------------------------------------------


def modalities_from(introspection: Mapping[str, Any] | None) -> list[str] | None:
    """What a *running* child says it accepts, or ``None`` if it did not say.

    The distinction matters to the gate in a way it does not to the Dashboard:
    ``None`` ("the child never answered") and ``[]`` ("the child answered, and
    it is text only") are the same pixel on screen but opposite verdicts here --
    the first is ``"unknown"`` and fails a bar, the second is a clean ``"no"``.

    ``gui.state.modalities_text`` renders this same function's output, so the
    gate and the Dashboard cannot drift apart on what a model can do.
    """
    if not introspection:
        return None
    actual = introspection.get("actual")
    modalities = actual.get("modalities") if isinstance(actual, Mapping) else None
    if isinstance(modalities, Mapping):
        return sorted(str(k) for k, v in modalities.items() if v)
    if isinstance(modalities, Sequence) and not isinstance(modalities, str):
        return [str(m) for m in modalities]
    return None


#: Tags answered from ``ModelCapabilities`` -- derived from the file at scan
#: time, so they are an answer even when no child is running.
_CAPABILITY_FIELD_TAGS: dict[str, str] = {
    "vision": "vision",
    "tools": "tools",
    "thinking": "thinking",
}

#: Tags only a running child can speak to. No GGUF field and no
#: ``ModelCapabilities`` member describes them, so with no live answer they are
#: ``"unknown"`` -- and unknown fails the bar.
_LIVE_ONLY_TAGS: frozenset[str] = frozenset({"audio", "video"})

#: Name tokens and GGUF ``general.tags`` entries that *prove* a curated tag.
#: Absence proves nothing (see :func:`tag_verdict`), which is why every one of
#: these resolves to ``"yes"`` or ``"unknown"`` and never to ``"no"``.
_NAME_TAG_HINTS: dict[str, frozenset[str]] = {
    "uncensored": frozenset({"uncensored", "uncensor", "abliterated", "ablated"}),
    "coding": frozenset({"coder", "code"}),
    "roleplay": frozenset({"rp", "roleplay"}),
}
_FILE_TAG_HINTS: dict[str, frozenset[str]] = {
    "uncensored": frozenset({"uncensored", "abliterated"}),
    # Deliberately EMPTY (post-review fix, D52): ``general.tags`` is the model
    # card's tag soup, and cards routinely claim aspirations -- a
    # creative-writing merge in this very library carries ``coding``, ``math``
    # and ``stem`` in its card tags. A card tag is a claim; a name token
    # ("-Coder-") is an identity. Identity tags (uncensored, roleplay) stay
    # file-matchable because tagging a card "uncensored" is the author
    # declaring intent about the finetune itself, which is the fact asked for.
    "coding": frozenset(),
    "roleplay": frozenset({"roleplay", "roleplaying", "rp"}),
}

#: Everything the gate can answer without opening a file. Anything outside this
#: set may need the lazy ``general.tags`` read, so it is what decides whether
#: the gate touches the disk at all.
_NO_FILE_READ_TAGS: frozenset[str] = (
    frozenset(_CAPABILITY_FIELD_TAGS) | _LIVE_ONLY_TAGS | frozenset({"embedding"})
)

_TOKEN_SPLIT = re.compile(r"[-_/ .]+")


def name_tokens(model_id: str) -> frozenset[str]:
    """Lowercased whole tokens of a model id, for tag matching."""
    return frozenset(t for t in _TOKEN_SPLIT.split((model_id or "").lower()) if t)


#: Header-read cache keyed by ``(path, mtime)``. A gate call is a routing
#: primitive an agent may hit before every task, and re-parsing a GGUF header
#: per call would turn a cheap question into disk I/O. Keying on mtime means a
#: replaced file re-reads itself without any invalidation logic.
_FILE_TAGS_CACHE: dict[tuple[str, float], frozenset[str]] = {}


def gguf_tags(path: Path | None) -> frozenset[str]:
    """``general.tags`` from a GGUF header, lazily and never fatally.

    Why read the file at gate time instead of storing this at scan time: HF repo
    tags are not retained by the downloader, and ``general.tags`` is not on
    ``GgufMeta`` -- putting it there means bumping ``META_FORMAT_VERSION`` and
    rescanning the whole library, a minutes-long stall for a field only this
    feature wants. A header-only read (``load_tensors=False``) costs tens of
    milliseconds cold on a large header (measured 35-90 ms here, microseconds
    cached), paid once per model per file version -- the ``(path, mtime)`` cache
    key -- and only when a requested tag actually needs the file.

    Any failure -- missing file, truncated header, a key that is not a string
    array -- returns an empty set. A gate that 500s because a model file is
    being replaced would be worse than useless to the agent that depends on it.
    """
    if path is None:
        return frozenset()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return frozenset()

    key = (str(path), mtime)
    cached = _FILE_TAGS_CACHE.get(key)
    if cached is not None:
        return cached

    tags: frozenset[str] = frozenset()
    try:
        from studioforge.core.gguf import read_gguf

        raw = read_gguf(path, load_tensors=False).kv.get("general.tags")
        if isinstance(raw, str):
            tags = frozenset({raw.strip().lower()}) - {""}
        elif isinstance(raw, Sequence):
            tags = frozenset(str(t).strip().lower() for t in raw) - {""}
    except Exception as exc:  # noqa: BLE001 - tags are a bonus, never a dependency
        log.debug("model_gate: could not read general.tags from %s: %s", path, exc)

    _FILE_TAGS_CACHE[key] = tags
    return tags


def tag_verdict(
    tag: str,
    *,
    record: ModelRecord | None,
    model_id: str,
    modalities: list[str] | None,
    file_tags: frozenset[str],
) -> Verdict:
    """Can this candidate do ``tag``? ``"yes"`` / ``"no"`` / ``"unknown"``.

    Sources are tried in order of authority and the first one that *can* answer
    wins. The three shapes of tag:

    **Capability tags** (vision/tools/thinking) come from ``ModelCapabilities``,
    which was derived from the file at scan time -- so ``False`` there is a real
    ``"no"``, not an absence of information. Vision additionally accepts a live
    child's ``modalities`` as a second yes-path, because a model can be served
    with a projector the record predates.

    **Live-only tags** (audio/video) have no local field at all, so they are
    ``"unknown"`` unless a running child listed them.

    **Everything else** goes to the generic matcher: the literal tag as a whole
    name token of the model id, or in the file's ``general.tags``. This is what
    makes the feature extensible -- a caller can ask for ``tags=vietnamese`` or
    ``tags=medical`` and get a useful answer with **zero code change here**, and
    new curated tags only ever need an entry in the hint tables.

    The asymmetry that matters: a curated or generic tag that is *not* found is
    ``"unknown"``, never ``"no"``. A model id without "uncensored" in it is not
    thereby proven censored -- most model names simply do not describe their
    alignment. Since unknown fails a requirement, the gate still refuses to
    claim what it cannot show; it just says *why* honestly ("cannot verify
    'uncensored'" rather than a confident and unfounded "no uncensored").
    """
    caps = getattr(record, "capabilities", None) if record is not None else None
    live = set(modalities or ())

    field_name = _CAPABILITY_FIELD_TAGS.get(tag)
    if field_name is not None:
        # Vision is the one capability a RUNNING child reports directly, and
        # when it answers, its answer outranks the scan (post-review fix, D52):
        # the record says what the *file* can do, the child says what this
        # *process* was launched able to do -- and a projector deleted after the
        # scan, or a multimodal launched without `--mmproj`, is exactly the
        # record-yes/live-no divergence that would send an image to a child that
        # cannot read it. The gate routes to the process, so the process wins.
        if tag == "vision" and modalities is not None:
            return "yes" if tag in live else "no"
        if caps is not None and bool(getattr(caps, field_name, False)):
            return "yes"
        if tag in live:
            return "yes"
        if caps is not None:
            return "no"
        # No record at all: nothing derived this from the file, so the absence
        # of a live answer leaves us genuinely ignorant rather than negative.
        return "no" if modalities is not None else "unknown"

    if tag in _LIVE_ONLY_TAGS:
        if modalities is None:
            return "unknown"
        return "yes" if tag in live else "no"

    tokens = name_tokens(model_id)
    hints = _NAME_TAG_HINTS.get(tag, frozenset({tag}))
    if tokens & hints:
        return "yes"
    if file_tags & _FILE_TAG_HINTS.get(tag, frozenset({tag})):
        return "yes"
    return "unknown"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def _why_not(
    *,
    params_b: float | None,
    min_params_b: float | None,
    verdicts: Mapping[str, Verdict],
) -> list[str]:
    """Short, operator-readable reasons this candidate missed the bar."""
    reasons: list[str] = []
    if min_params_b is not None:
        if params_b is None:
            reasons.append(f"unknown size, cannot prove >= {min_params_b:.1f}B")
        elif params_b < min_params_b:
            reasons.append(f"params {params_b:.1f}B < {min_params_b:.1f}B")
    for tag in sorted(verdicts):
        verdict = verdicts[tag]
        if verdict == "no":
            reasons.append(f"no {tag}")
        elif verdict == "unknown":
            reasons.append(f"cannot verify {tag!r}")
    return reasons


async def _introspect_safely(
    introspect: Callable[[str], Awaitable[dict[str, Any] | None]], model_id: str
) -> dict[str, Any] | None:
    """Live state for one child, or ``None`` -- never an exception.

    The manager's ``introspect`` talks HTTP to a child that may be mid-restart,
    wedged, or gone. A gate that propagated that would fail the *routing*
    decision over a *reporting* problem, when the registry fallback below can
    still answer most of the question.
    """
    try:
        result = await introspect(model_id)
    except Exception as exc:  # noqa: BLE001 - a silent child is a fact, not a fault
        log.debug("model_gate: introspect(%s) failed: %s", model_id, exc)
        return None
    return result if isinstance(result, dict) else None


def _resolve(registry: Any, model_id: str) -> ModelRecord | None:
    """The registry entry for a loaded model, or ``None`` if it has none."""
    try:
        resolver = getattr(registry, "resolve", None) or getattr(registry, "get", None)
        record = resolver(model_id) if resolver is not None else None
    except Exception as exc:  # noqa: BLE001 - a missing record must not fail the gate
        log.debug("model_gate: registry lookup for %s failed: %s", model_id, exc)
        return None
    return record if isinstance(record, ModelRecord) else None


_EMBEDDING_REASON = (
    "'embedding' is not a gate tag: embedding models cannot chat, so they are "
    "never gate candidates -- call /v1/embeddings with the embedding model directly"
)


async def gate_answer(
    requirement: GateRequirement,
    *,
    registry: Any,
    supervisor: Any,
    introspect: Callable[[str], Awaitable[dict[str, Any] | None]],
) -> dict[str, Any]:
    """Answer "is a loaded model above this bar, and which one should I use?" (D52).

    Candidates are the supervisor's ``ready`` instances minus embedding-kind
    records: an embedding model is loaded and healthy and completely unable to
    serve a chat request, so counting it would produce a "yes" that breaks the
    caller's very next call.

    Among candidates that clear every requirement the **largest** wins (an
    unsized model sorts last), ties broken by most recent activity -- the bigger
    model is the better answer to "is something good enough loaded", and recency
    breaks the tie towards the one whose weights and prompt cache are warm.
    """
    if "embedding" in requirement.tags:
        return _refusal(requirement, _EMBEDDING_REASON, hint=None)

    # The one call left that could take the gate down, guarded for the same
    # reason introspection and the header read are: on the REST surface an
    # exception here is a 500 while the MCP _guard would wrap it, and "the two
    # surfaces answer identically" plus "a gate must never be the thing that
    # breaks" are both part of the contract (post-review fix, D52).
    try:
        listed = list(supervisor.list())
    except Exception as exc:  # noqa: BLE001 - degrade to a refusal, never a 500
        log.warning("model_gate: supervisor.list() failed: %s", exc)
        return _refusal(
            requirement, "the loaded-model table could not be read; retry shortly", hint=None
        )

    ready = [i for i in listed if getattr(i, "state", None) == "ready"]
    records = {i.model_id: _resolve(registry, i.model_id) for i in ready}
    candidates = [
        i for i in ready if getattr(records.get(i.model_id), "kind", "chat") != "embedding"
    ]
    excluded_embeddings = len(ready) - len(candidates)

    if not candidates:
        reason = (
            "only an embedding model is loaded, and an embedding model cannot chat"
            if excluded_embeddings
            else "nothing is loaded"
        )
        return _refusal(requirement, reason, hint=_HINT)

    introspections = await asyncio.gather(
        *(_introspect_safely(introspect, i.model_id) for i in candidates)
    )

    # Only touch the disk when a requested tag actually needs the file, and only
    # for candidates that have one. Most gate calls ask about size and vision
    # and never open anything.
    wants_file_tags = bool(requirement.tags - _NO_FILE_READ_TAGS)
    paths = [getattr(records.get(i.model_id), "path", None) for i in candidates]
    if wants_file_tags:
        file_tag_sets = await asyncio.gather(*(asyncio.to_thread(gguf_tags, p) for p in paths))
    else:
        file_tag_sets = [frozenset()] * len(candidates)

    rows: list[dict[str, Any]] = []
    scored: list[tuple[float, float, dict[str, Any], Any, ModelRecord | None]] = []
    for instance, introspection, file_tags in zip(
        candidates, introspections, file_tag_sets, strict=True
    ):
        record = records.get(instance.model_id)
        modalities = modalities_from(introspection)
        total_b, active_b, source = approx_params_b(record or instance.model_id)
        verdicts = {
            tag: tag_verdict(
                tag,
                record=record,
                model_id=instance.model_id,
                modalities=modalities,
                file_tags=file_tags,
            )
            for tag in requirement.tags
        }
        why_not = _why_not(
            params_b=total_b, min_params_b=requirement.min_params_b, verdicts=verdicts
        )
        row: dict[str, Any] = {
            "model": instance.model_id,
            "params_b": total_b,
            "active_params_b": active_b,
            "params_source": source,
            "modalities": list(modalities or ()),
            "tags": dict(verdicts),
            "meets": not why_not,
            "why_not": why_not,
        }
        rows.append(row)
        if not why_not:
            scored.append(
                (
                    total_b if total_b is not None else -1.0,
                    float(getattr(instance, "last_activity_at", None) or 0.0),
                    row,
                    modalities,
                    record,
                )
            )

    if not scored:
        return _refusal(requirement, _gap_sentence(requirement, rows), hint=_HINT, instances=rows)

    _, _, best, best_modalities, best_record = max(scored, key=lambda s: (s[0], s[1]))
    capabilities = best_record.capabilities.model_dump() if best_record is not None else None
    return {
        "ok": True,
        "answer": "yes",
        "model": best["model"],
        "params_b": best["params_b"],
        "active_params_b": best["active_params_b"],
        "params_source": best["params_source"],
        "modalities": list(best_modalities or ()),
        "capabilities": capabilities,
        "checked": _checked(requirement),
        "instances": rows,
        "reason": None,
        "hint": None,
    }


_HINT = (
    "Load one with POST /api/models/{id}/load-recommended (or the MCP "
    "load_recommended tool, or the GUI's Models tab) before sending this work locally."
)


def _checked(requirement: GateRequirement) -> dict[str, Any]:
    return {
        "min_params_b": requirement.min_params_b,
        "tags": sorted(requirement.tags),
    }


def _refusal(
    requirement: GateRequirement,
    reason: str,
    *,
    hint: str | None,
    instances: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """A "no", with every winner-describing field emptied.

    All five winner fields go ``None``/``[]`` together, so a client never has to
    wonder whether a populated ``params_b`` next to ``model: null`` describes
    something it could use. There is no winner; there is nothing to describe.
    """
    return {
        "ok": False,
        "answer": "no",
        "model": None,
        "params_b": None,
        "active_params_b": None,
        "params_source": None,
        "modalities": [],
        "capabilities": None,
        "checked": _checked(requirement),
        "instances": instances or [],
        "reason": reason,
        "hint": hint,
    }


def _gap_sentence(requirement: GateRequirement, rows: Sequence[Mapping[str, Any]]) -> str:
    """One sentence naming the gap, chosen from what actually blocked.

    A blanket modality/tag miss is stated first because it is the crisper fact:
    if *nothing* loaded does vision, saying so is more actionable than comparing
    sizes. When a tag is satisfied by somebody, the size comparison below is
    computed over only the models that satisfied it, so the number quoted is the
    largest model the caller could actually have used.
    """
    for tag in sorted(requirement.tags):
        verdicts = [str(r["tags"].get(tag)) for r in rows]
        if "yes" in verdicts:
            continue
        if all(v == "unknown" for v in verdicts):
            return f"no loaded model can be verified as {tag!r}"
        return f"no loaded model reports {tag}"

    if requirement.min_params_b is None:
        return "no loaded model meets the requirement"

    eligible = [r for r in rows if all(str(r["tags"].get(t)) == "yes" for t in requirement.tags)]
    sizes = [r["params_b"] for r in (eligible or rows) if r["params_b"] is not None]
    bar = requirement.min_params_b
    if not sizes:
        return f"no loaded model reports a size, so none can be proven to meet the {bar:g}B bar"
    return f"largest loaded model is {max(sizes):g}B, below the {bar:g}B bar"
