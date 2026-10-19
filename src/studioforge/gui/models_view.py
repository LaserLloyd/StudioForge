"""Pure helpers for the Models page and Test Chat's curl button.

Everything here derives a value from data the backend already produced
(``ModelRecord``, ``InstanceInfo``, a ``catalog()`` entry, a ``plan_preview()`` dict, a GPU
list). Nothing here loads a model, calls the planner or touches a widget, so the page logic
that matters -- what state a row shows, whether a model "fits", what the fit bars say -- is
testable without NiceGUI.
"""

from __future__ import annotations

import json
import re
import shlex
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import parse_qs, unquote, urlparse

from studioforge.gui import nav
from studioforge.gui import state as st

# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

_GGUF_SUFFIX = re.compile(r"\.gguf$", re.IGNORECASE)
_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_GGUF_WORD = re.compile(r"[-_. ]gguf$", re.IGNORECASE)


def friendly_name(record: Any) -> str:
    """A readable name for a model row; the exact id stays beside it.

    ``record.name`` when the registry gave one that differs from the id (a virtual model's
    title, a repo's display name). Otherwise the id with its ``.gguf``/shard suffix, a
    trailing ``-GGUF`` and a trailing quant label removed, underscores read as spaces.
    Never empty: it falls back to the id.
    """
    model_id = str(getattr(record, "id", "") or "")
    name = str(getattr(record, "name", "") or "")
    if name and name != model_id:
        return name
    text = model_id.rsplit("/", 1)[-1]
    text = _GGUF_SUFFIX.sub("", text)
    text = _SHARD_SUFFIX.sub("", text)
    quant = str(getattr(record, "quant", "") or "")
    if quant and quant.lower() != "unknown":
        text = re.sub(rf"[-_. ]{re.escape(quant)}$", "", text, flags=re.IGNORECASE)
    text = _GGUF_WORD.sub("", text)
    text = text.replace("_", " ").strip(" -")
    return text or model_id


def family_label(record: Any) -> str:
    """The architecture family as a column value (``qwen3`` -> ``qwen3``), or a dash."""
    arch = str(getattr(record, "architecture", "") or "")
    return arch if arch and arch.lower() != "unknown" else st.UNKNOWN


def quant_colour(quant: str | None) -> str:
    """Badge colour by bit width: lighter quants are cheaper, so they read as calmer."""
    text = (quant or "").upper()
    match = re.search(r"(\d+)", text)
    if text.startswith(("F16", "BF16", "F32")):
        return "deep-orange"
    if text.startswith(("MXFP", "NVFP", "FP4")):
        return "teal"
    if not match:
        return "grey"
    bits = int(match.group(1))
    if bits <= 3:
        return "blue-grey"
    if bits == 4:
        return "primary"
    if bits <= 6:
        return "secondary"
    return "accent"


def fmt_ctx(tokens: Any) -> str:
    """``131072`` -> ``128k``; ``8192`` -> ``8k``; anything unreadable -> a dash."""
    try:
        value = int(tokens)
    except (TypeError, ValueError):
        return st.UNKNOWN
    if value >= 1_048_576 and value % 1_048_576 == 0:
        return f"{value // 1_048_576}M"
    if value >= 1024:
        return f"{round(value / 1024)}k"
    return str(value)


# ---------------------------------------------------------------------------
# "Fits on" (from the catalog's recommended placement -- the planner's own answer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitHint:
    text: str
    colour: str  # a theme colour name: positive / warning / negative / grey
    tooltip: str = ""


def fits_hint(entry: Mapping[str, Any] | None) -> FitHint:
    """The "Fits on" cell for one ``manager.catalog()`` entry.

    The catalog is the planner's recommendation computed as if the best cards were free,
    with ``fits_now`` / ``would_evict`` beside it for the machine as it stands. This only
    words that answer; it never recomputes it.
    """
    if not entry:
        return FitHint(st.UNKNOWN, "grey", "The fit estimate is not available yet.")
    if entry.get("arch_supported") is False:
        return FitHint(
            "unsupported",
            "negative",
            str(entry.get("arch_note") or "The active engine cannot load this architecture."),
        )
    rec = entry.get("recommended")
    if not isinstance(rec, Mapping):
        return FitHint(
            "does not fit",
            "negative",
            str(entry.get("recommended_basis") or "No set of GPUs on this box can hold it."),
        )
    label = str(rec.get("label") or "GPUs")
    ctx = fmt_ctx(rec.get("ctx_per_slot"))
    text = f"{label} · {ctx}"
    evict = [str(v) for v in (rec.get("would_evict") or [])]
    if rec.get("fits_now"):
        return FitHint(text, "positive", f"Fits right now on {label} at {ctx} context.")
    if evict:
        return FitHint(
            text,
            "warning",
            f"Fits on {label} after unloading {', '.join(evict)}.",
        )
    return FitHint(
        text,
        "warning",
        f"Fits on {label} when idle; something outside StudioForge holds VRAM right now.",
    )


def catalog_by_id(catalog: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    """``{model id: entry}`` from a ``manager.catalog()`` payload."""
    if not isinstance(catalog, Mapping):
        return {}
    return {
        str(entry.get("id")): entry
        for entry in (catalog.get("models") or [])
        if isinstance(entry, Mapping) and entry.get("id")
    }


# ---------------------------------------------------------------------------
# What the Load form starts from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadForm:
    """The drawer's Load fields before the person touches them, and where they came from."""

    ctx_size: int | None
    kv_cache_type: str
    kv_cache_type_v: str
    parallel: int | None
    devices: tuple[int, ...]
    source: str  # "running" | "saved" | "recommended" | "defaults"


def load_form_defaults(
    *,
    running_plan: Any,
    settings: Any,
    recommended_args: Mapping[str, Any] | None,
    default_ctx: int | None,
) -> LoadForm:
    """Pre-fill the Load form: what is running now, else the model's saved settings, else the
    planner's recommendation, else the server's defaults -- field by field.

    A running model reloads onto what it runs on (so "reload" changes nothing it was not
    asked to); a model whose settings were saved keeps them (the drawer must not override
    what "All settings" stored); otherwise the catalog's ``load_args`` are used verbatim.
    """
    rec = dict(recommended_args or {})
    if running_plan is not None:
        return LoadForm(
            ctx_size=int(getattr(running_plan, "ctx_per_slot", 0) or running_plan.ctx_size),
            kv_cache_type=str(getattr(running_plan, "kv_cache_type", "") or ""),
            kv_cache_type_v=str(getattr(running_plan, "kv_cache_type_v", "") or ""),
            parallel=int(getattr(running_plan, "parallel", 0) or 0) or None,
            devices=tuple(int(d) for d in (getattr(running_plan, "devices", None) or ())),
            source="running",
        )
    saved_ctx = getattr(settings, "ctx_size", None)
    saved_kv = getattr(settings, "kv_cache_type", None)
    saved_kv_v = getattr(settings, "kv_cache_type_v", None)
    saved_parallel = getattr(settings, "parallel", None)
    saved_devices = getattr(settings, "device_override", None)
    any_saved = any(
        v not in (None, [], "")
        for v in (saved_ctx, saved_kv, saved_kv_v, saved_parallel, saved_devices)
    )
    devices = saved_devices or rec.get("devices") or ()
    return LoadForm(
        ctx_size=int(saved_ctx or rec.get("ctx_size") or default_ctx or 0) or None,
        kv_cache_type=str(saved_kv or rec.get("kv_cache_type") or ""),
        kv_cache_type_v=str(saved_kv_v or rec.get("kv_cache_type_v") or ""),
        parallel=int(saved_parallel or rec.get("parallel") or 0) or None,
        devices=tuple(int(d) for d in devices),
        source="saved" if any_saved else ("recommended" if rec else "defaults"),
    )


def evictions_for(preview: Mapping[str, Any] | None, own_ids: Iterable[str]) -> list[str]:
    """The models a load would have to unload, never counting the model being (re)loaded.

    A reload frees the running copy first, but the dry run still counts it as resident, so
    it can name the very model being reloaded as something to evict.
    """
    own = {str(i) for i in own_ids}
    return [str(m) for m in ((preview or {}).get("evict_model_ids") or []) if str(m) not in own]


# ---------------------------------------------------------------------------
# Row state
# ---------------------------------------------------------------------------

STATE_IDLE: Final = "idle"
STATE_LOADING: Final = "loading"
STATE_LOADED: Final = "loaded"
STATE_ERROR: Final = "error"

_STATE_COLOUR: Final[Mapping[str, str]] = {
    STATE_IDLE: "grey",
    STATE_LOADING: "warning",
    STATE_LOADED: "positive",
    STATE_ERROR: "negative",
}


@dataclass(frozen=True)
class LoadError:
    """What went wrong on the last load of one model: the catalog code and a sentence."""

    code: str
    message: str
    at: float = 0.0


@dataclass(frozen=True)
class RowState:
    primary: str
    colour: str
    detail: str = ""
    badges: tuple[tuple[str, str, str], ...] = ()  # (text, colour, tooltip)
    error: LoadError | None = None
    progress_s: float | None = None  # seconds spent loading, when loading

    @property
    def label(self) -> str:
        return self.primary


def row_state(
    record: Any,
    instance: Any,
    *,
    pinned: bool,
    leased: bool,
    error: LoadError | None = None,
    pending_since: float | None = None,
    arch_refusal: str | None = None,
    now: float | None = None,
) -> RowState:
    """One model's state for the table: idle, loading, loaded or error, plus badges.

    ``instance`` is the supervisor's view of the child serving this record (``None`` when
    idle); ``pending_since`` is when this panel started a load that has not returned yet,
    which covers the gap before the supervisor lists the child. A load error is only
    shown while the model is not running -- a model that is up has nothing to explain.
    """
    clock = time.time() if now is None else now
    badges: list[tuple[str, str, str]] = []
    if pinned:
        badges.append(("pinned", "accent", "Kept loaded; loaded again at start-up."))
    if leased:
        badges.append(("leased", "secondary", "On GPUs held by a standing lease."))
    if arch_refusal:
        badges.append((st.UNSUPPORTED_ARCH_BADGE, "negative", arch_refusal))
    state = str(getattr(instance, "state", "") or "")
    if state == "loading" or (instance is None and pending_since is not None):
        started = pending_since if pending_since is not None else None
        return RowState(
            STATE_LOADING,
            _STATE_COLOUR[STATE_LOADING],
            "starting the engine and reading the weights",
            tuple(badges),
            None,
            max(0.0, clock - started) if started is not None else None,
        )
    if instance is not None and state == "ready":
        active = int(getattr(instance, "active_requests", 0) or 0)
        detail = f"busy · {active} request{'s' if active != 1 else ''}" if active else "idle"
        speed = getattr(instance, "last_tokens_per_second", None)
        if speed:
            detail += f" · {speed} tok/s"
        return RowState(STATE_LOADED, _STATE_COLOUR[STATE_LOADED], detail, tuple(badges))
    if error is not None or state == "failed":
        failure = error
        if failure is None:
            failure = LoadError(
                "load_failed", str(getattr(instance, "last_error", "") or "The model failed.")
            )
        return RowState(
            STATE_ERROR, _STATE_COLOUR[STATE_ERROR], failure.code, tuple(badges), failure
        )
    return RowState(STATE_IDLE, _STATE_COLOUR[STATE_IDLE], "", tuple(badges))


def leased_model_ids(leases: Iterable[Any], instances: Iterable[Any]) -> set[str]:
    """Models a standing lease names, or that are resident on leased GPUs."""
    ids: set[str] = set()
    leased_devices: set[int] = set()
    for lease in leases:
        ids.update(str(m) for m in (getattr(lease, "model_ids", None) or []))
        leased_devices.update(int(d) for d in (getattr(lease, "devices", None) or []))
    if leased_devices:
        for instance in instances:
            plan = getattr(instance, "plan", None)
            devices = {int(d) for d in (getattr(plan, "devices", None) or [])}
            if devices & leased_devices:
                ids.add(str(instance.model_id))
    return ids


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_rows(
    records: Sequence[Any],
    *,
    needle: str | None,
    chip: str,
    loaded_ids: set[str],
) -> list[Any]:
    """Apply the search box and the Loaded chip.

    The search is the library's existing text filter (:func:`state.filter_models`: id, quant,
    architecture, kind, capabilities, ``pinned``) widened with the friendly name, so
    "qwen 32b" finds ``Qwen3-32B-Q4_K_M``. The Downloads chip shows the queue instead of
    the table, so here it behaves like All.
    """
    rows = list(records)
    if chip == nav.FILTER_LOADED:
        rows = [r for r in rows if r.id in loaded_ids]
    text = (needle or "").strip().lower()
    if not text:
        return rows
    narrowed = {r.id for r in st.filter_models(rows, text)}
    terms = text.split()
    out = []
    for record in rows:
        if record.id in narrowed:
            out.append(record)
            continue
        haystack = f"{friendly_name(record)} {record.id} {record.quant} {record.architecture}"
        if terms and all(term in haystack.lower() for term in terms):
            out.append(record)
    return out


# ---------------------------------------------------------------------------
# VRAM fit bars (drawn before the Load button is pressed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitBar:
    """One GPU's bar: what is used now, what this load would add, how it ends up."""

    index: int
    name: str
    total_bytes: int
    used_bytes: int
    add_bytes: int
    targeted: bool  # this load places weights on the GPU

    @property
    def after_bytes(self) -> int:
        return self.used_bytes + self.add_bytes

    @property
    def over(self) -> bool:
        return self.total_bytes > 0 and self.after_bytes > self.total_bytes

    @property
    def used_fraction(self) -> float:
        return _fraction(self.used_bytes, self.total_bytes)

    @property
    def add_fraction(self) -> float:
        room = max(0.0, 1.0 - self.used_fraction)
        return min(room, _fraction(self.add_bytes, self.total_bytes))

    @property
    def text(self) -> str:
        after = st.format_gib(self.after_bytes, precision=1)
        base = f"{after} of {st.format_gib(self.total_bytes, precision=1)}"
        if self.add_bytes:
            return f"{base} (+{st.format_gib(self.add_bytes, precision=1)})"
        return f"{base} used" if self.used_bytes else "free"


def _fraction(part: float, whole: float) -> float:
    return 0.0 if not whole else max(0.0, min(1.0, float(part) / float(whole)))


def fit_bars(
    preview: Mapping[str, Any] | None,
    gpus: Sequence[Any],
    *,
    credit: Mapping[int, int] | None = None,
) -> list[FitBar]:
    """Per-GPU bars from a ``plan_preview()`` dict and the probe's GPU list.

    An accepted plan carries ``per_gpu_bytes`` (what this load adds to each card); a refusal
    carries none, so the bars show the cards as they stand with nothing added. Cards the
    load does not touch still get a bar, because "what else is on the 3090s" is half of the
    answer.

    ``credit`` is what the model being (re)loaded already holds on each card: a reload
    replaces that footprint, so it is taken off the "used" side rather than counted twice.
    """
    added: dict[int, int] = {}
    if isinstance(preview, Mapping) and preview.get("fits"):
        for index, size in st.fit_verdict(preview).per_gpu:
            added[int(index)] = int(size)
    bars = []
    for gpu in gpus:
        index = int(getattr(gpu, "index", 0))
        bars.append(
            FitBar(
                index=index,
                name=str(getattr(gpu, "name", f"GPU{index}")),
                total_bytes=int(getattr(gpu, "total_bytes", 0) or 0),
                used_bytes=max(
                    0, int(getattr(gpu, "used_bytes", 0) or 0) - int((credit or {}).get(index, 0))
                ),
                add_bytes=added.get(index, 0),
                targeted=index in added,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# Add a model by repo id or URL
# ---------------------------------------------------------------------------

_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_HF_HOSTS: Final = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})


def parse_hf_reference(text: str | None) -> tuple[str | None, str | None]:
    """``(repo_id, error)`` from what someone pasted into the add-model box.

    Accepts ``owner/repo``, a ``huggingface.co``/``hf.co`` URL (the model page, ``/tree/...``,
    ``/blob/...`` or ``/resolve/...`` forms), and the ``lmstudio://open_from_hf?model=`` link
    the Hugging Face button produces. Anything else is an ``error`` sentence, never a guess:
    the value ends up in a Hugging Face API path.
    """
    raw = (text or "").strip()
    if not raw:
        return None, "Paste a Hugging Face repo id (owner/repo) or a link to the model page."
    candidate = raw
    if "://" in raw:
        parsed = urlparse(raw)
        if parsed.scheme in ("lmstudio", "studioforge"):
            values = parse_qs(parsed.query).get("model") or []
            candidate = unquote(values[0]) if values else ""
        elif parsed.scheme in ("http", "https") and (parsed.hostname or "") in _HF_HOSTS:
            parts = [unquote(p) for p in parsed.path.split("/") if p]
            if parts and parts[0] in {"models", "datasets", "spaces"}:
                parts = parts[1:] if parts[0] == "models" else []
            candidate = "/".join(parts[:2])
        else:
            return None, "That link is not a Hugging Face model page."
    if ".." in candidate or not _REPO.match(candidate):
        return None, f"{raw!r} is not a Hugging Face repo id; expected owner/repo."
    return candidate, None


# ---------------------------------------------------------------------------
# Test Chat: copy as curl
# ---------------------------------------------------------------------------

KEY_ENV: Final = "$STUDIOFORGE_API_KEY"


def chat_curl(
    *,
    base_url: str,
    model_id: str,
    messages: Sequence[Mapping[str, Any]],
    sampler: Mapping[str, Any] | None = None,
    stream: bool = True,
    needs_key: bool = False,
) -> str:
    """A ``curl`` command that repeats the current chat request against the gateway.

    The address is the gateway's own OpenAI base URL, never the child's loopback port the
    panel talks to. Images are replaced by a placeholder (a data URL can be megabytes), and
    the API key is never embedded: when the server requires one the header reads the
    ``$STUDIOFORGE_API_KEY`` environment variable. Single quotes are escaped the POSIX way,
    so the command pastes into bash, zsh and Git Bash; on Windows PowerShell use ``curl.exe``
    with the same arguments.
    """
    body: dict[str, Any] = {"model": model_id, "messages": _plain_messages(messages)}
    for key, value in (sampler or {}).items():
        if key in {"model", "messages", "stream_options", "return_progress"}:
            continue
        body[key] = value
    body["stream"] = bool(stream)
    payload = json.dumps(body, ensure_ascii=False, indent=2)
    lines = [
        f"curl {'-N ' if stream else ''}{shlex.quote(base_url.rstrip('/') + '/chat/completions')}",
        "  -H 'Content-Type: application/json'",
    ]
    if needs_key:
        lines.append(f'  -H "Authorization: Bearer {KEY_ENV}"')
    lines.append(f"  -d {shlex.quote(payload)}")
    return " \\\n".join(lines)


def _plain_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "image_url":
                    parts.append({"type": "image_url", "image_url": {"url": "<image omitted>"}})
                else:
                    parts.append(dict(part) if isinstance(part, Mapping) else part)
            content = parts
        out.append({**{k: v for k, v in message.items() if k != "content"}, "content": content})
    return out


__all__ = [
    "KEY_ENV",
    "STATE_ERROR",
    "STATE_IDLE",
    "STATE_LOADED",
    "STATE_LOADING",
    "FitBar",
    "FitHint",
    "LoadError",
    "LoadForm",
    "RowState",
    "catalog_by_id",
    "chat_curl",
    "family_label",
    "filter_rows",
    "fit_bars",
    "fits_hint",
    "fmt_ctx",
    "evictions_for",
    "friendly_name",
    "leased_model_ids",
    "load_form_defaults",
    "parse_hf_reference",
    "quant_colour",
    "row_state",
]
