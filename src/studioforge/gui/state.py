"""Pure presentation logic for the web GUI.

Everything in this module is a plain function of its arguments: no NiceGUI
imports, no I/O, no global state. That is deliberate -- the interesting logic of
a control panel is *derivation and formatting* (does this fit? what does "Auto"
mean? did the user actually change the API key?), and none of that is testable
once it is tangled into element callbacks. The tab modules are therefore
presentation only and call into here.

Two invariants are load-bearing and are covered by tests:

* **``None`` round-trips as ``None``.** A blank/"Auto" form field must stay
  ``None`` all the way into :class:`~studioforge.types.ModelSettings`, because
  ``None`` means "ask the planner at load time, against real free VRAM". Baking
  a concrete number in at save time would defeat the entire planner.
* **A redacted secret is never sent back.** ``GET /api/config`` returns
  ``"abcd...yz"`` for the API key; posting that string back would overwrite the
  real key with the placeholder. :func:`masked_secret_changed` is the guard.
"""

from __future__ import annotations

import contextlib
import re
import time
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Final

from pydantic import BaseModel

from studioforge.api.auth import redact
from studioforge.config import RESTART_REQUIRED_KEYS
from studioforge.types import GpuInfo, InstanceInfo, ModelRecord, ModelSettings, VirtualPreset

#: Shown wherever a value is genuinely unknown, rather than zero.
UNKNOWN: Final = "—"

_KIB: Final = 1024.0
_BYTE_UNITS: Final = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")

#: Fields that are plain ``bool`` (not ``bool | None``) in ModelSettings, so an
#: absent form value means ``False`` rather than "inherit the default".
_NON_OPTIONAL_BOOL: Final = frozenset({"pinned", "mlock"})

#: Text fields where the empty string is a real value, not "unset".
_KEEP_EMPTY_TEXT: Final = frozenset({"extra_flags"})

#: Comma-separated integer lists rendered as a text field.
_DEVICE_LIST_FIELDS: Final = frozenset({"device_override", "draft_device_override"})

#: Mirrors :func:`studioforge.api.auth.redact`'s output shape
#: (``"abcd...yz"``). Anything matching it is treated as an untouched
#: placeholder rather than a new secret.
_REDACTION_RE: Final = re.compile(r"^\S{1,8}\.\.\.\S{1,4}$")

#: The planner's FP4 note (DECISIONS D9) is the one plan note users must not
#: miss, so it is detected and promoted to a warning rather than buried.
FP4_NOTE_MARKER: Final = "without native acceleration"


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_bytes(value: float | int | None, *, precision: int = 1) -> str:
    """Binary-prefixed size, e.g. ``"17.99 GiB"``.

    Binary units because that is what VRAM and GGUF sizes are quoted in
    everywhere else in this system; mixing GB and GiB in one UI makes the
    planner's numbers look wrong.
    """
    if value is None:
        return UNKNOWN
    number = float(value)
    negative = number < 0
    number = abs(number)
    if number < _KIB:
        text = f"{int(number)} B"
        return f"-{text}" if negative else text
    for unit in _BYTE_UNITS[1:]:
        number /= _KIB
        if number < _KIB or unit == _BYTE_UNITS[-1]:
            text = f"{number:.{precision}f} {unit}"
            return f"-{text}" if negative else text
    return UNKNOWN  # pragma: no cover - unreachable, the loop always returns


def format_gib(value: float | int | None, *, precision: int = 2) -> str:
    """Size in GiB regardless of magnitude, for columns that must line up."""
    if value is None:
        return UNKNOWN
    return f"{float(value) / (_KIB**3):.{precision}f} GiB"


def format_mib(value: float | int | None, *, precision: int = 0) -> str:
    if value is None:
        return UNKNOWN
    return f"{float(value) / (_KIB**2):.{precision}f} MiB"


def format_duration(seconds: float | int | None) -> str:
    """Compact human duration: ``"45s"``, ``"5m 03s"``, ``"2h 05m"``."""
    if seconds is None:
        return UNKNOWN
    total = int(max(0.0, float(seconds)))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours:02d}h"


def format_rate(bytes_per_second: float | int | None) -> str:
    """Transfer rate for download rows.

    Binary-prefixed and labelled as such: the value is divided by 1024^2, and
    calling that "MB/s" would overstate the true decimal rate by ~5% -- this
    module's own rule is that binary maths gets a binary label.
    """
    if bytes_per_second is None:
        return UNKNOWN
    return f"{float(bytes_per_second) / (_KIB**2):.1f} MiB/s"


def format_eta(seconds: float | int | None) -> str:
    if seconds is None:
        return UNKNOWN
    return format_duration(seconds)


def format_percent(fraction: float | int | None, *, precision: int = 0) -> str:
    if fraction is None:
        return UNKNOWN
    return f"{float(fraction) * 100:.{precision}f}%"


def progress_fraction(done: float | int | None, total: float | int | None) -> float:
    """Clamped 0..1 fraction that never divides by zero."""
    if not total or float(total) <= 0 or done is None:
        return 0.0
    return max(0.0, min(1.0, float(done) / float(total)))


def ttl_text(remaining_s: float | int | None, *, pinned: bool = False) -> str:
    """TTL countdown label.

    Pinned wins over any number: a pinned model has no countdown at all, and
    showing one would suggest it is about to disappear.
    """
    if pinned:
        return "pinned"
    if remaining_s is None:
        return "no TTL"
    return format_duration(remaining_s)


def instance_ttl_text(instance: InstanceInfo | None, *, pinned: bool = False) -> str:
    if instance is None:
        return UNKNOWN
    return ttl_text(instance.ttl_remaining_s, pinned=pinned)


def format_timestamp(epoch: float | int | None) -> str:
    if epoch is None:
        return UNKNOWN
    return time.strftime("%H:%M:%S", time.localtime(float(epoch)))


def format_datetime(epoch: float | int | None) -> str:
    """Absolute local date and time, for tooltips behind a relative label."""
    if not epoch:
        return UNKNOWN
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(epoch)))


def format_when(epoch: float | int | None, *, now: float | None = None) -> str:
    """Human "when", coarsening with age: ``"2 hours ago"`` … ``"18 Jul 2025"``.

    The Models table is sorted newest-first by default, so the top of the list is
    the part that must answer "did that download finish?" at a glance -- and a
    raw ``2026-08-14 17:32`` does not. Resolution is deliberately dropped as
    things get older: an exact minute matters for something that arrived while
    you were watching, and stops mattering entirely a month later.

    Day boundaries are *calendar* boundaries, not multiples of 24 hours, because
    "yesterday" means the previous date to a reader, not "25 hours ago".
    """
    if not epoch:
        return UNKNOWN
    stamp_epoch = float(epoch)
    current = time.time() if now is None else float(now)
    delta = current - stamp_epoch
    stamp = time.localtime(stamp_epoch)
    today = time.localtime(current)

    if 0 <= delta < 60:
        return "just now"
    if 0 <= delta < 3600:
        minutes = int(delta // 60)
        return f"{minutes} minute{'' if minutes == 1 else 's'} ago"
    if (stamp.tm_year, stamp.tm_yday) == (today.tm_year, today.tm_yday):
        if delta < 0:
            # A file stamped in the future (clock skew, a restored archive) is
            # still "today"; claiming "-3 hours ago" would just look broken.
            return time.strftime("%H:%M", stamp)
        hours = int(delta // 3600)
        return f"{hours} hour{'' if hours == 1 else 's'} ago"
    yesterday = time.localtime(current - 86400.0)
    if (stamp.tm_year, stamp.tm_yday) == (yesterday.tm_year, yesterday.tm_yday):
        return "yesterday"
    # %-d is not portable (it is a glibc extension), so the day is formatted by
    # hand -- this code runs on Windows.
    month = time.strftime("%b", stamp)
    if stamp.tm_year == today.tm_year:
        return f"{stamp.tm_mday} {month}"
    return f"{stamp.tm_mday} {month} {stamp.tm_year}"


def model_added_at(record: ModelRecord) -> float:
    """When this model *arrived*, as a sortable epoch.

    ``mtime`` is the newest mtime across every file of the logical model, so a
    multi-part download is dated by the shard that finished last rather than by
    shard 1. ``added_at`` is now derived from it and kept stable across rescans,
    and is the fallback for a record (a virtual model) with no files of its own.
    """
    return float(record.mtime or record.added_at or 0.0)


#: A message that is already a fully rendered structlog console line -- it
#: starts with its own ISO timestamp and carries its own level and logger.
_RENDERED_LOG_LINE_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def log_line_text(entry: Mapping[str, Any]) -> str:
    """One :data:`studioforge.logging.RING_BUFFER` record as a display line.

    structlog hands the stdlib handler an already-rendered console line
    (timestamp, level, event, logger, key=value pairs); prefixing that with a
    second timestamp and level printed every line's metadata twice. A message
    that clearly carries its own header is therefore shown as-is, and the
    prefix is only added for plain-stdlib records that need one.
    """
    message = str(entry.get("message") or entry.get("msg") or "")
    if _RENDERED_LOG_LINE_RE.match(message):
        return message.rstrip()
    stamp = format_timestamp(entry.get("ts"))
    level = str(entry.get("level") or "INFO")
    logger = str(entry.get("logger") or "")
    prefix = f"{stamp} {level:<8}"
    if logger:
        prefix = f"{prefix} {logger}"
    return f"{prefix}  {message}".rstrip()


# ---------------------------------------------------------------------------
# Model table derivations
# ---------------------------------------------------------------------------


def capability_badges(record: ModelRecord) -> list[str]:
    """Short capability labels for the Models table.

    Order is fixed so a row of badges is scannable down a column rather than
    reshuffling per model. Virtual models say *what kind* of virtual they are
    -- ``persona`` (a request-time preset, D13), ``LoRA`` (an adapter set), or
    plain ``virtual`` (a bare alias) -- because those three have very different
    VRAM consequences and lumping them together hides that.
    """
    badges: list[str] = []
    capabilities = record.capabilities
    if capabilities.vision:
        badges.append("vision")
    if capabilities.tools:
        badges.append("tools")
    if capabilities.thinking:
        badges.append("thinking")
    if capabilities.embedding:
        badges.append("embedding")
    if capabilities.multi_part:
        badges.append("multi-part")
    if record.is_virtual:
        if record.preset is not None:
            badges.append("persona")
        if record.settings.adapters:
            badges.append("LoRA")
        if record.preset is None and not record.settings.adapters:
            badges.append("virtual")
    return badges


@dataclass(frozen=True)
class CapabilityIcon:
    """One capability, as a coloured glyph with the sentence it stands for.

    An icon on its own is a rebus, not information -- nobody guesses that a
    brain means "this model thinks out loud in the reply". So the tooltip is
    part of the data, not an optional extra, and every icon has one.
    """

    key: str
    label: str
    icon: str
    colour: str
    tooltip: str


#: Fixed display order, so a column of icons is scannable down the table rather
#: than reshuffling per model.
CAPABILITY_ICONS: Final[tuple[CapabilityIcon, ...]] = (
    CapabilityIcon(
        "vision",
        "vision",
        "visibility",
        "cyan",
        "Vision — accepts images. This model has an mmproj projector, so you can paste or "
        "attach a screenshot in the Chat tab and it will actually be looked at.",
    ),
    CapabilityIcon(
        "tools",
        "tools",
        "handyman",
        "orange",
        "Tools — the chat template handles tool/function calling, so an agent client can "
        "give this model tools to call.",
    ),
    CapabilityIcon(
        "thinking",
        "thinking",
        "psychology",
        "purple",
        "Thinking — a reasoning model. By default its thoughts appear INLINE in the reply "
        "(reasoning_format = none, DECISIONS D12), which is what keeps ordinary OpenAI "
        "clients from showing an empty message. Switch to 'deepseek' in the model's "
        "settings only if your client reads the reasoning_content field.",
    ),
    CapabilityIcon(
        "embedding",
        "embedding",
        "scatter_plot",
        "teal",
        "Embeddings — a vector model. Use /v1/embeddings; it has no chat endpoint, which "
        "is why it is not offered in the Chat tab.",
    ),
    CapabilityIcon(
        "multi_part",
        "multi-part",
        "layers",
        "blue-grey",
        "Multi-part — the weights are split across several GGUF shard files, all of which "
        "are loaded together as one model.",
    ),
)


def capability_icons(record: ModelRecord) -> list[CapabilityIcon]:
    """The coloured icons for one model's capabilities, in fixed order.

    A model with no capabilities yields an empty list on purpose: a "none"
    placeholder in every row would add a column of noise to say nothing.
    """
    capabilities = record.capabilities
    return [item for item in CAPABILITY_ICONS if bool(getattr(capabilities, item.key, False))]


def capability_signature(record: ModelRecord) -> str:
    """Stable key for "same feature set", used to group the type/features sort.

    Built from :data:`CAPABILITY_ICONS`'s fixed order so two models with the
    same capabilities always produce the same string and therefore sort
    adjacently -- which is the entire point of sorting by features.
    """
    return "+".join(item.key for item in capability_icons(record))


def model_status_label(instance: InstanceInfo | None) -> str:
    """``loaded`` / ``failed`` / ``loading`` / ``unloaded`` for one model."""
    if instance is None:
        return "unloaded"
    if instance.state == "ready":
        return "loaded"
    if instance.state == "stopped":
        return "unloaded"
    return instance.state


STATUS_COLOURS: Final[Mapping[str, str]] = {
    "loaded": "positive",
    "loading": "warning",
    "unloading": "warning",
    "failed": "negative",
    "unloaded": "grey",
}


def status_colour(label: str) -> str:
    return STATUS_COLOURS.get(label, "grey")


# ---------------------------------------------------------------------------
# Model table sorting / filtering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SortColumn:
    """One clickable column header of the Models table."""

    key: str
    label: str
    #: Direction the *first* click on this header applies. Sizes and dates are
    #: asked about biggest/newest-first; names are asked about A-Z. Making every
    #: column start ascending would mean two clicks to get the obvious answer.
    descending_first: bool = False
    tooltip: str = ""


#: Display order of the table's columns, which is also their order in the
#: header row. Every entry is sortable by clicking it.
MODEL_COLUMNS: Final[tuple[SortColumn, ...]] = (
    SortColumn("name", "Name", False, "Model id — what a client asks for. A→Z."),
    SortColumn(
        "date",
        "Downloaded",
        True,
        "When the model's files arrived, taken from the newest file of the set so a "
        "multi-part download is dated by the shard that finished last.",
    ),
    SortColumn("size", "Size", True, "Total size of the weights on disk."),
    SortColumn("quant", "Quant", False, "Quantisation label, e.g. Q4_K_M."),
    SortColumn("architecture", "Arch", False, "Model architecture reported by the GGUF."),
    SortColumn(
        "type",
        "Features",
        False,
        "Kind (chat / embedding / rerank) first, then the capability set — so vision "
        "models sit with vision models and thinking models with thinking models.",
    ),
    SortColumn("recent", "Last used", True, "When this model last served a request."),
    SortColumn("loaded", "Status", True, "Loaded models first."),
)

MODEL_COLUMN_BY_KEY: Final[Mapping[str, SortColumn]] = {c.key: c for c in MODEL_COLUMNS}

#: What a browser with no stored preference gets. Newest-first, because "the one
#: I just downloaded" is overwhelmingly the model being looked for.
DEFAULT_SORT_KEY: Final = "date"

#: Where an unknown/stale key degrades to. Deliberately *not* the default: a
#: broken preference should land somewhere boringly predictable.
FALLBACK_SORT_KEY: Final = "name"


def _sort_value(record: ModelRecord, key: str, loaded_ids: frozenset[str] | set[str]) -> Any:
    if key == "size":
        return float(record.size_bytes)
    if key == "date":
        return model_added_at(record)
    if key == "quant":
        return record.quant.lower()
    if key == "architecture":
        return record.architecture.lower()
    if key == "type":
        # Kind first, capability set second: a NUL joiner keeps the two fields
        # from bleeding into each other on comparison.
        return f"{record.kind}\x00{capability_signature(record)}"
    if key == "recent":
        return float(record.last_used_at or 0.0)
    if key == "loaded":
        return 1.0 if record.id in loaded_ids else 0.0
    return record.id.lower()


def sort_models(
    records: Sequence[ModelRecord],
    key: str | None,
    descending: bool | None = None,
    *,
    loaded_ids: frozenset[str] | set[str] = frozenset(),
) -> list[ModelRecord]:
    """Sort the model table by one column.

    ``descending=None`` means "this column's natural first-click direction"
    (:attr:`SortColumn.descending_first`), which is what a fresh click on a
    header uses; an explicit ``True``/``False`` is the toggled state.

    Ties always break on the id, ascending, in both directions -- Python's sort
    is stable, so pre-sorting by id and then sorting by the column value gives a
    deterministic order that never reshuffles equal rows between refreshes. That
    matters more than it sounds: the table repaints on a timer, and rows that
    swap places under the cursor make it unusable.

    An unknown ``key`` (a stale value remembered from an older build) degrades
    to the name sort rather than raising -- a stored preference must never be
    able to break the table.
    """
    column = MODEL_COLUMN_BY_KEY.get(str(key or ""), MODEL_COLUMN_BY_KEY[FALLBACK_SORT_KEY])
    reverse = column.descending_first if descending is None else bool(descending)
    by_id = sorted(records, key=lambda r: r.id.lower())
    return sorted(by_id, key=lambda r: _sort_value(r, column.key, loaded_ids), reverse=reverse)


def next_sort(current_key: str, current_descending: bool, clicked_key: str) -> tuple[str, bool]:
    """Header-click state machine: new column, or flip the current one.

    A first click on a different column uses that column's natural direction;
    clicking the active column again reverses it. Clicking an unknown key is a
    no-op rather than an error, so a stale element id cannot wedge the table.
    """
    column = MODEL_COLUMN_BY_KEY.get(clicked_key)
    if column is None:
        return (current_key, current_descending)
    if clicked_key != current_key:
        return (clicked_key, column.descending_first)
    return (clicked_key, not current_descending)


def sort_indicator(column_key: str, active_key: str, descending: bool) -> str:
    """Material icon name for a header's arrow, or ``""`` when it is not active."""
    if column_key != active_key:
        return ""
    return "arrow_downward" if descending else "arrow_upward"


#: How each column's two directions read in plain words. "Descending" is
#: precise and useless: what a user wants to know is whether clicking again
#: gives them the biggest, the newest, or Z-to-A.
_SORT_DIRECTION_WORDS: Final[Mapping[str, tuple[str, str]]] = {
    "size": ("smallest first", "largest first"),
    "date": ("oldest first", "newest first"),
    "recent": ("least recently used first", "most recently used first"),
    "loaded": ("unloaded first", "loaded first"),
}


def sort_direction_text(column_key: str, descending: bool) -> str:
    """Plain-words description of a column's current sort direction."""
    ascending_text, descending_text = _SORT_DIRECTION_WORDS.get(column_key, ("A→Z", "Z→A"))
    return descending_text if descending else ascending_text


def stored_sort_key(value: Any) -> str:
    """Validate a remembered sort key, degrading to the default (newest first)."""
    text = str(value or "")
    return text if text in MODEL_COLUMN_BY_KEY else DEFAULT_SORT_KEY


def stored_sort_descending(value: Any, key: str) -> bool:
    """Validate a remembered direction, degrading to the column's natural one."""
    if isinstance(value, bool):
        return value
    column = MODEL_COLUMN_BY_KEY.get(key)
    return column.descending_first if column is not None else False


def filter_models(records: Sequence[ModelRecord], needle: str | None) -> list[ModelRecord]:
    """Substring filter over the fields a user actually scans for.

    Matches the id, quant, architecture, kind, the capability badges and the
    ``pinned`` badge, so ``vision``, ``nvfp4``, ``qwen`` or ``pinned`` all
    narrow a 30-model library usefully -- an id-only match would make "show me
    the vision models" impossible.
    """
    if not needle or not needle.strip():
        return list(records)
    lowered = needle.strip().lower()

    def matches(record: ModelRecord) -> bool:
        haystack = (
            record.id,
            record.quant,
            record.architecture,
            record.kind,
            *capability_badges(record),
            *(("pinned",) if record.settings.pinned else ()),
        )
        return any(lowered in text.lower() for text in haystack)

    return [r for r in records if matches(r)]


# ---------------------------------------------------------------------------
# Virtual models / presets (DECISIONS D13)
# ---------------------------------------------------------------------------

#: Baseline for "does this virtual model change the launch command?".
_DEFAULT_MODEL_SETTINGS: Final = ModelSettings()

#: Preset fields rendered in the persona dialog, in display order.
PRESET_FORM_FIELDS: Final = (
    "system_prompt",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repeat_penalty",
    "max_tokens",
)

#: Preset fields that are integers (NiceGUI number inputs hand back floats).
_PRESET_INT_FIELDS: Final = frozenset({"top_k", "max_tokens"})


def has_launch_overrides(settings: ModelSettings) -> bool:
    """Whether these settings (adapters aside) change the child's argv.

    This is the half of the sharing rule the persona dialog needs separately:
    adapters are chosen *in* the dialog, but a ctx/kv/anything override was set
    beforehand in the settings dialog and would silently cost a dedicated
    instance if the UI did not surface it.
    """
    return settings.model_copy(update={"adapters": []}) != _DEFAULT_MODEL_SETTINGS


def shares_base_instance(record: ModelRecord) -> bool:
    """Mirror of ``ModelManager.serving_record``'s sharing rule (D13).

    A preset-only virtual model rides its base's instance; any
    :class:`ModelSettings` delta (adapters included) means its own child.
    """
    if not record.is_virtual or record.base_model_id is None:
        return False
    return record.settings == _DEFAULT_MODEL_SETTINGS


def virtual_instance_note(
    base_id: str, *, has_adapters: bool, has_overrides: bool
) -> tuple[bool, str]:
    """(shares, explanation) for the persona dialog's VRAM-cost indicator.

    Shown *before* the user commits, because the difference between "free" and
    "another full copy of a 30B model" is exactly the thing to know before
    clicking Create, not after the load fails or evicts something.
    """
    if has_adapters or has_overrides:
        causes: list[str] = []
        if has_adapters:
            causes.append("LoRA adapters")
        if has_overrides:
            causes.append("launch-time setting overrides (ctx size, KV type, …)")
        return (
            False,
            f"Needs its own llama-server instance: {' and '.join(causes)} change the launch "
            f"command line. Loading this model costs the full VRAM of another copy of "
            f"{base_id}.",
        )
    return (
        True,
        f"Shares {base_id}'s running instance. The system prompt and sampler defaults are "
        "applied per request by the gateway, so this persona costs no extra VRAM — any "
        "number of personas can ride one loaded base.",
    )


def virtual_base_line(record: ModelRecord) -> str | None:
    """One-line origin note for a virtual model's table row, else ``None``."""
    if not record.is_virtual:
        return None
    base = record.base_model_id or UNKNOWN
    if shares_base_instance(record):
        return f"preset over {base} — shares its instance (no extra VRAM)"
    return f"over {base} — needs its own instance (adapters or setting overrides)"


def form_from_preset(preset: VirtualPreset | None) -> dict[str, Any]:
    """Flat form dict for the persona dialog; ``None`` prompt renders as ``""``."""
    source = preset if preset is not None else VirtualPreset()
    form: dict[str, Any] = {name: getattr(source, name) for name in PRESET_FORM_FIELDS}
    if form["system_prompt"] is None:
        form["system_prompt"] = ""
    return form


def preset_from_form(form: Mapping[str, Any]) -> VirtualPreset | None:
    """Validate the persona form into a preset; all-blank collapses to ``None``.

    Blank means "no default" for every field -- an absent sampler default lets
    the client (or the model's own settings) decide, exactly like the settings
    dialog's ``None`` round-trip invariant.
    """
    payload: dict[str, Any] = {}
    for name in PRESET_FORM_FIELDS:
        value = form.get(name)
        if isinstance(value, str):
            value = value.strip() or None
        if value is not None and name in _PRESET_INT_FIELDS:
            value = int(float(value))
        payload[name] = value
    preset = VirtualPreset.model_validate(payload)
    return None if preset.is_empty() else preset


def preset_summary_lines(preset: VirtualPreset | None) -> list[str]:
    """Compact display of what a preset does, for the table row and dialog."""
    if preset is None:
        return []
    lines: list[str] = []
    prompt = (preset.system_prompt or "").strip()
    if prompt:
        shown = prompt if len(prompt) <= 80 else prompt[:77] + "…"
        lines.append(f"system prompt: {shown}")
    defaults = [
        f"{name}={getattr(preset, name)}"
        for name in PRESET_FORM_FIELDS
        if name != "system_prompt" and getattr(preset, name) is not None
    ]
    if defaults:
        lines.append("defaults: " + " · ".join(defaults) + " (an explicit client value wins)")
    return lines


def actual_ctx_text(introspection: Mapping[str, Any] | None) -> str:
    """Context llama-server *actually* reports, not what we asked for.

    The distinction matters: the engine can clamp the request, and a dashboard
    that echoes the request back would hide that.
    """
    if not introspection or not introspection.get("loaded"):
        return UNKNOWN
    actual = introspection.get("actual") or {}
    value = actual.get("n_ctx") if isinstance(actual, Mapping) else None
    if value is None:
        return UNKNOWN
    return f"{int(value)}"


def slots_text(slots: Sequence[Mapping[str, Any]] | None) -> str:
    """``busy/idle`` counts from llama-server's ``/slots``."""
    if not slots:
        return UNKNOWN
    busy = 0
    for slot in slots:
        state = slot.get("state")
        is_processing = slot.get("is_processing")
        if is_processing is True or (isinstance(state, int) and state != 0):
            busy += 1
    return f"{busy} busy / {len(slots) - busy} idle"


# ---------------------------------------------------------------------------
# Live activity (LM Studio parity)
# ---------------------------------------------------------------------------

#: Activity state -> Quasar colour, so "generating" reads differently from
#: "ingesting a prompt" at a glance.
ACTIVITY_COLOURS: Final[Mapping[str, str]] = {
    "generating": "positive",
    "processing_prompt": "warning",
    "idle": "grey",
}


def _activity(introspection: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The ``activity`` block from ``manager.introspect()``, or an empty one."""
    if not introspection:
        return {}
    activity = introspection.get("activity")
    return activity if isinstance(activity, Mapping) else {}


def activity_label(introspection: Mapping[str, Any] | None) -> str:
    """LM Studio's own phrasing for what the model is doing right now.

    The string is produced by ``manager.slot_activity`` -- "Processing prompt
    40/100 (40%)", "Generating - 37 tokens", "Idle" -- and used verbatim so the
    Dashboard says the same thing LM Studio would.
    """
    activity = _activity(introspection)
    label = activity.get("label")
    if isinstance(label, str) and label:
        return label
    return UNKNOWN if not activity else "Idle"


def activity_state(introspection: Mapping[str, Any] | None) -> str:
    state = _activity(introspection).get("state")
    return str(state) if isinstance(state, str) else "idle"


def activity_colour(introspection: Mapping[str, Any] | None) -> str:
    return ACTIVITY_COLOURS.get(activity_state(introspection), "grey")


def activity_slots_text(introspection: Mapping[str, Any] | None) -> str:
    """``busy/idle`` from the derived activity, falling back to ``/slots``."""
    activity = _activity(introspection)
    if activity and "busy" in activity:
        return f"{int(activity.get('busy') or 0)} busy / {int(activity.get('idle') or 0)} idle"
    slots = (introspection or {}).get("slots")
    return slots_text(slots if isinstance(slots, Sequence) else None)


def tokens_generated(introspection: Mapping[str, Any] | None) -> int:
    value = _activity(introspection).get("tokens_generated")
    return int(value) if isinstance(value, int) else 0


def is_speculative(introspection: Mapping[str, Any] | None) -> bool:
    """Whether drafting is actually armed on this instance.

    Read from the per-slot ``speculative`` flag (via ``introspect()['actual']``),
    not from ``/props`` -- the props value describes per-request defaults and
    stays "none" even when a draft model is loaded and drafting.
    """
    if not introspection:
        return False
    actual = introspection.get("actual")
    if isinstance(actual, Mapping) and actual.get("speculative"):
        return True
    return any(bool(slot.get("speculative")) for slot in activity_slot_rows(introspection))


def modalities_text(introspection: Mapping[str, Any] | None) -> str:
    """What the running server says it accepts (e.g. vision), as it says it."""
    if not introspection:
        return UNKNOWN
    actual = introspection.get("actual")
    modalities = actual.get("modalities") if isinstance(actual, Mapping) else None
    if isinstance(modalities, Mapping):
        enabled = sorted(k for k, v in modalities.items() if v)
        return ", ".join(enabled) if enabled else "text only"
    if isinstance(modalities, Sequence) and not isinstance(modalities, str):
        return ", ".join(str(m) for m in modalities) or "text only"
    return UNKNOWN


def build_info_text(introspection: Mapping[str, Any] | None) -> str:
    if not introspection:
        return UNKNOWN
    actual = introspection.get("actual")
    info = actual.get("build_info") if isinstance(actual, Mapping) else None
    return str(info) if info else UNKNOWN


def activity_slot_rows(introspection: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    slots = _activity(introspection).get("slots")
    if not isinstance(slots, Sequence):
        return []
    return [slot for slot in slots if isinstance(slot, Mapping)]


def prompt_cache_text(slot: Mapping[str, Any]) -> str:
    """Prompt-cache hit count for one slot.

    Surfaced prominently because prompt-cache reuse is the headline latency
    feature for agent workloads, and "is it actually reusing?" is otherwise
    invisible.
    """
    cached = slot.get("prompt_tokens_cached")
    total = slot.get("prompt_tokens")
    if not isinstance(cached, int):
        return "cache n/a"
    if isinstance(total, int) and total > 0:
        return f"cache hit {cached}/{total} ({int(100 * cached / total)}%)"
    return f"cache hit {cached}"


def slot_line(slot: Mapping[str, Any]) -> str:
    """One per-slot line for the Dashboard's expanded view."""
    parts = [f"slot {slot.get('id', '?')}", str(slot.get("label") or "Idle")]
    n_ctx = slot.get("n_ctx")
    if isinstance(n_ctx, int):
        parts.append(f"ctx {n_ctx}")
    parts.append(prompt_cache_text(slot))
    generated = slot.get("tokens_generated")
    if isinstance(generated, int) and generated:
        parts.append(f"{generated} generated")
    if slot.get("speculative"):
        parts.append("draft")
    task = slot.get("task_id")
    if isinstance(task, int) and task >= 0:
        parts.append(f"task {task}")
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# Unload / restart controls
# ---------------------------------------------------------------------------

#: The two "restart" buttons mean genuinely different things and are one
#: mis-click apart, so each says plainly what it takes down. Conflating them is
#: the failure mode worth designing against: an operator who wanted to pick up a
#: new engine build must not lose the API, and one who wanted a clean process
#: must not be told the engines were merely reloaded.
RESTART_ENGINES_HELP: Final = (
    "Reloads every loaded model's llama-server child while StudioForge itself stays up. "
    "This is what actually moves running models onto a newly installed engine build — a "
    "child keeps the engine it was launched with. In-flight generations are dropped; the "
    "API, the download queue and this panel are not interrupted."
)

RESTART_SERVER_WARNING: Final = (
    "This restarts the whole StudioForge process. Every model is unloaded, in-flight "
    "requests are dropped, and the API will be unavailable for a few seconds — any client "
    "talking to the gateway right now will see a connection error. This page will "
    "reconnect on its own once the server is back."
)


def unload_all_prompt(model_ids: Sequence[str]) -> str:
    """Confirmation text for Unload all, naming what is about to be dropped."""
    names = [str(m) for m in model_ids]
    if not names:
        return "Nothing is loaded, so there is nothing to unload."
    listed = ", ".join(names)
    return (
        f"Unload {len(names)} resident model(s) and free their VRAM?\n{listed}\n\n"
        "Each will reload on its next request, which costs the load time again. "
        "Pinned models are unloaded too, and stay down until they are loaded or "
        "pinned again — an explicit unload outranks the pin."
    )


def restart_backend_note(payload: Mapping[str, Any] | None) -> str:
    """Outcome line for ``POST /api/restart/backend``.

    Failures are named individually rather than folded into a count: "3 of 4
    restarted" leaves the operator hunting for which one did not.
    """
    if not payload:
        return "nothing to restart — no models are loaded"
    restarted = [str(m) for m in (payload.get("restarted") or [])]
    failed = payload.get("failed") or []
    if not restarted and not failed:
        return "nothing to restart — no models are loaded"
    parts = [f"restarted {len(restarted)} engine(s): {', '.join(restarted)}"] if restarted else []
    for item in failed:
        if isinstance(item, Mapping):
            parts.append(f"FAILED {item.get('model_id')}: {item.get('error')}")
    return " · ".join(parts)


def restart_server_note(payload: Mapping[str, Any] | None) -> str:
    """Outcome line for ``POST /api/restart/server``, naming *how* it restarts."""
    via = str((payload or {}).get("via") or "")
    if via == "tray":
        return (
            "Restarting: the server is exiting and the tray that launched it brings it back. "
            "The API is down for a few seconds; this page will reconnect by itself."
        )
    if via == "watchdog":
        return (
            "Restarting via the watchdog sidecar. The API is down for a few seconds; "
            "this page will reconnect by itself."
        )
    if via:
        return (
            "The watchdog was not reachable, so the server is respawning itself detached. "
            "This page will reconnect by itself."
        )
    return "Restart requested."


# ---------------------------------------------------------------------------
# Test results / speculative decoding
# ---------------------------------------------------------------------------


def acceptance_rate_text(result: Mapping[str, Any]) -> str:
    """Draft acceptance rate as a percentage, or why there isn't one.

    A badly matched draft model makes generation *slower*, and the acceptance
    rate is the only number that makes that visible instead of mysterious.
    """
    if not result.get("speculative_used"):
        return "speculative decoding not used"
    rate = result.get("draft_acceptance_rate")
    drafted = result.get("draft_tokens")
    accepted = result.get("draft_tokens_accepted")
    if rate is None:
        return f"drafted {drafted} tokens (acceptance unknown)"
    verdict = "good" if float(rate) >= 0.6 else "poor — consider dropping the draft model"
    return f"draft acceptance {float(rate) * 100:.1f}% ({accepted}/{drafted} tokens) — {verdict}"


def test_result_lines(result: Mapping[str, Any]) -> list[str]:
    """Everything worth showing after a Test, in display order."""
    if result.get("embedding_dims") is not None:
        return [
            f"embedding of {result['embedding_dims']} dimensions",
            f"latency {result.get('latency_s')}s",
        ]
    tps = result.get("tokens_per_second")
    lines = [
        f"latency {result.get('latency_s')}s · {result.get('completion_tokens', 0)} tokens · "
        f"{tps if tps is not None else UNKNOWN} tok/s",
    ]
    engine_tps = result.get("engine_predicted_per_second")
    engine_pp = result.get("engine_prompt_per_second")
    if engine_tps is not None or engine_pp is not None:
        lines.append(
            f"engine timings: generation {_round_or_dash(engine_tps)} tok/s · "
            f"prompt {_round_or_dash(engine_pp)} tok/s"
        )
    lines.append(acceptance_rate_text(result))
    return lines


def _round_or_dash(value: Any) -> str:
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return UNKNOWN


def ab_comparison_lines(
    with_draft: Mapping[str, Any], without_draft: Mapping[str, Any]
) -> list[str]:
    """A/B summary of the same test run with and without speculative decoding."""
    a = with_draft.get("tokens_per_second")
    b = without_draft.get("tokens_per_second")
    lines = [
        f"with draft:    {_round_or_dash(a)} tok/s",
        f"without draft: {_round_or_dash(b)} tok/s",
        acceptance_rate_text(with_draft),
    ]
    try:
        ratio = float(a) / float(b)  # type: ignore[arg-type]
    except (TypeError, ValueError, ZeroDivisionError):
        return lines
    if ratio >= 1.05:
        lines.append(f"speculative decoding is {ratio:.2f}x faster here — keep the draft model")
    elif ratio <= 0.95:
        lines.append(f"speculative decoding is {1 / ratio:.2f}x SLOWER here — drop the draft model")
    else:
        lines.append("no meaningful difference; the draft model is costing VRAM for nothing")
    return lines


def fp4_plan_note(instance: InstanceInfo | None) -> str | None:
    """The planner's quant-affinity note for a running instance, if any.

    Surfaced on the Dashboard so an FP4 quant that landed on an Ampere card is
    *visible* rather than a mystery slowdown. Informative, never alarming --
    DECISIONS D9 measured this configuration working correctly, just with
    slower prompt processing than a Blackwell card would give.
    """
    if instance is None or instance.plan is None:
        return None
    return next((n for n in instance.plan.notes if FP4_NOTE_MARKER in n), None)


def poll_failure_note(error: str | BaseException) -> str:
    """Stale-data marker for a polled panel whose refresh just failed.

    The panel keeps showing its last successful data with this next to it,
    rather than replacing everything with an error card on each tick -- a
    transient poll failure should read as "stale", not as "gone".
    """
    return f"live refresh failed ({error}) — showing the last good data; retrying"


def device_text(instance: InstanceInfo | None) -> str:
    if instance is None or instance.plan is None:
        return UNKNOWN
    devices = instance.plan.devices
    if not devices:
        return UNKNOWN
    label = ", ".join(f"GPU{index}" for index in devices)
    if len(devices) > 1:
        label += f" ({instance.plan.split_mode})"
    return label


def vram_fraction(gpu: GpuInfo) -> float:
    return progress_fraction(gpu.used_bytes, gpu.total_bytes)


def gpu_headline(gpu: GpuInfo) -> str:
    return f"GPU{gpu.index} · {gpu.name}"


def gpu_detail_lines(gpu: GpuInfo) -> list[str]:
    """Everything shown under a dashboard GPU gauge, NVML-absence tolerant."""
    lines = [
        f"{format_gib(gpu.used_bytes)} used of {format_gib(gpu.total_bytes)}"
        f"  ({format_gib(gpu.free_bytes)} free)",
    ]
    util = (
        f"utilisation {gpu.utilization_pct:.0f}%"
        if gpu.utilization_pct is not None
        else "utilisation unavailable"
    )
    temp = f"{gpu.temperature_c:.0f} °C" if gpu.temperature_c is not None else "temp n/a"
    lines.append(f"{util}  ·  {temp}  ·  cc {gpu.cc_str}")
    return lines


def ram_text(total_bytes: int | None, used_bytes: int | None) -> str:
    if not total_bytes:
        return "system RAM unavailable"
    return f"System RAM {format_gib(used_bytes)} used of {format_gib(total_bytes)}"


# ---------------------------------------------------------------------------
# VRAM holders (DECISIONS.md D23, D39)
#
# The panel these feed answers one question -- "who has my VRAM" -- and it has
# to answer it for a holder that is NOT ours, because that was the incident: a
# stray pytest run's llama-server children held 25 GiB and every surface we had
# said "llama-server.exe, 0 bytes, not ours".
# ---------------------------------------------------------------------------

#: Mirrors ``vram_holders.DEVICE_PLACEMENT_MIN_BYTES``. Repeated rather than
#: imported: this module is pure presentation and importing ``core.vram_holders``
#: would drag psutil and the supervisor into it. Below this a device entry is a
#: llama.cpp CUDA context (~0.22 GiB on a 3090, ~0.43 GiB on a 5090), not a
#: placement, and printing it turns a two-GPU model into a four-GPU one.
_DEVICE_MIN_BYTES: Final = 512 * 1024 * 1024


def vram_holder_origin(holder: Mapping[str, Any]) -> str:
    """Who owns this holder, in the words an operator needs.

    The parent is named for a foreign child because that is the actionable
    fact: "child of pytest.exe (pid 41288)" tells you what to close, whereas
    "not ours" tells you nothing at all. Same reason a llama-server from
    another install is named by its alias and port rather than filed under
    "foreign": those two words are what you type to find and stop it.
    """
    classification = str(holder.get("classification") or "")
    if classification == "ours" or holder.get("is_ours"):
        return "ours"
    if classification == "orphan":
        return "ORPHAN"
    if classification == "child-of-live-process":
        parent = holder.get("parent_name") or "an unknown process"
        pid = holder.get("parent_pid")
        return f"child of {parent}" + (f" (pid {pid})" if pid else "")
    if classification == "other-instance":
        bits = []
        if holder.get("alias"):
            bits.append(f"alias {holder['alias']}")
        if holder.get("port"):
            bits.append(f"port {holder['port']}")
        return "another install" + (f" ({', '.join(bits)})" if bits else "")
    return "foreign"


def vram_holder_devices(holder: Mapping[str, Any]) -> str:
    """The device column: where this holder's memory actually is.

    ``per_gpu_bytes`` is a measurement (PDH per adapter, joined to CUDA
    ordinals through the adapter LUID -- D39), so it is preferred and quoted
    with its numbers: ``CUDA0 15.5 GiB, CUDA1 14.5 GiB``. Entries below
    :data:`_DEVICE_MIN_BYTES` are dropped, because llama.cpp opens a CUDA
    context on *every* visible device and those contexts were what made a
    two-GPU model read as ``CUDA0,1,2,3``.

    Without a measurement the only thing left is that same context list, and it
    is printed in the old bare form -- ``CUDA0,1`` -- so the two cases do not
    look alike.
    """
    per_gpu = holder.get("per_gpu_bytes") or {}
    if per_gpu:
        parts = []
        for key in sorted(per_gpu, key=lambda item: int(item)):
            used = int(per_gpu[key] or 0)
            if used < _DEVICE_MIN_BYTES:
                continue
            index = int(key)
            label = f"CUDA{index}" if index >= 0 else "other adapter"
            parts.append(f"{label} {format_gib(used, precision=1)}")
        if parts:
            return ", ".join(parts)
    gpus = holder.get("gpu_indices") or []
    if gpus:
        return "CUDA" + ",".join(str(index) for index in gpus)
    return ""


def vram_holder_line(holder: Mapping[str, Any]) -> str:
    """One holder as a single monospaced row."""
    name = str(holder.get("name") or "unknown")
    pid = holder.get("pid")
    size = (
        format_gib(holder.get("used_bytes"))
        if int(holder.get("used_bytes") or 0) > 0
        else f"size {UNKNOWN}"
    )
    parts = [f"{name} (pid {pid})", size, vram_holder_origin(holder)]
    devices = vram_holder_devices(holder)
    if devices:
        parts.insert(2, devices)
    alias = holder.get("alias")
    # The origin already carries the alias for another install; repeating it
    # would push the useful end of the row off the panel.
    if alias and str(holder.get("classification") or "") != "other-instance":
        parts.append(f"alias {alias}")
    return "  ·  ".join(parts)


def vram_holder_tooltip(holder: Mapping[str, Any]) -> str:
    """The whole row plus the facts the truncated line cannot fit.

    ``detail`` (which install a stray llama-server belongs to) and the exact
    per-device bytes are the actionable end of a holder, and they are also the
    end that falls off a narrow panel.
    """
    parts = [vram_holder_line(holder)]
    detail = holder.get("detail")
    if detail:
        parts.append(str(detail))
    per_gpu = holder.get("per_gpu_bytes") or {}
    if per_gpu:
        measured = ", ".join(
            f"CUDA{key} {format_bytes(int(value))}"
            if int(key) >= 0
            else f"other adapter {format_bytes(int(value))}"
            for key, value in sorted(per_gpu.items(), key=lambda item: int(item[0]))
        )
        parts.append(f"measured per device: {measured}")
    elif holder.get("gpu_indices"):
        parts.append(
            "devices are CUDA contexts, not measured placement — per-GPU attribution "
            "is unavailable here"
        )
    return "\n".join(parts)


def vram_holder_is_reclaimable(holder: Mapping[str, Any]) -> bool:
    """Only an orphan may be killed from the panel. See D23."""
    return str(holder.get("classification") or "") == "orphan"


def vram_holders_note(view: Mapping[str, Any] | None) -> str:
    """The one-line summary under the holder rows.

    Says something in every state, including the good one: an empty panel with
    no explanation reads as "the panel is broken", and "nothing but desktop
    apps hold VRAM" is a real, reassuring answer.
    """
    if not view:
        return "VRAM holders unavailable."
    holders = list(view.get("holders") or [])
    desktop = int(view.get("desktop_processes_count") or 0)
    desktop_bytes = int(view.get("desktop_processes_bytes") or 0)
    tail = ""
    if desktop:
        tail = f" {desktop} desktop process(es) hold {format_gib(desktop_bytes)} between them."
    if not holders:
        return ("Nothing but desktop applications is holding VRAM." + tail).strip()
    orphans = int(view.get("orphan_count") or 0)
    lead = f"{len(holders)} holder(s)."
    if orphans:
        lead = (
            f"{len(holders)} holder(s), {orphans} orphaned — leaked llama-server "
            "processes whose parent is gone."
        )
    if str(view.get("per_process_bytes")) == "unavailable":
        tail += " Per-process VRAM is unavailable on this box, so sizes read as unknown."
    elif str(view.get("per_gpu_bytes_source")) == "nvml-context":
        # Saying this outright matters: a context list looks exactly like a
        # placement, and reading one as the other is how a 30 GiB model appears
        # to be spread over four cards (D39).
        tail += (
            " Per-GPU attribution is unavailable, so devices are the ones each "
            "process has a CUDA context on."
        )
    return (lead + tail).strip()


# ---------------------------------------------------------------------------
# Fit verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitVerdict:
    """Rendered form of :meth:`ModelManager.plan_preview`'s two shapes."""

    fits: bool
    headline: str
    detail_lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    per_gpu: list[tuple[int, int]] = field(default_factory=list)
    fp4_warning: str | None = None

    @property
    def colour(self) -> str:
        return "positive" if self.fits else "negative"

    def as_text(self) -> str:
        parts = [self.headline, *self.detail_lines]
        parts += [f"note: {n}" for n in self.notes]
        parts += [f"try: {s}" for s in self.suggestions]
        return "\n".join(parts)


def _int_keyed(mapping: Any) -> list[tuple[int, int]]:
    """Normalise a ``{device: bytes}`` map whose keys may have been JSON-ified."""
    if not isinstance(mapping, Mapping):
        return []
    out: list[tuple[int, int]] = []
    for key, value in mapping.items():
        try:
            out.append((int(key), int(value)))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def fit_verdict(preview: Mapping[str, Any]) -> FitVerdict:
    """Turn a ``plan_preview()`` dict into something a human can act on.

    Handles both shapes the manager returns -- the accepted plan and the
    rejection -- because the settings dialog shows this live while the user is
    still dragging sliders, so it flips between them constantly.
    """
    notes = [str(n) for n in (preview.get("notes") or [])]
    fp4 = next((n for n in notes if FP4_NOTE_MARKER in n), None)

    if preview.get("fits"):
        devices = [int(d) for d in (preview.get("devices") or [])]
        if len(devices) == 1:
            headline = f"Fits on GPU{devices[0]}"
        elif devices:
            joined = ", ".join(f"GPU{d}" for d in devices)
            headline = f"Fits across {joined} (split: {preview.get('split_mode', 'layer')})"
        else:
            headline = "Fits"
        ctx = int(preview.get("ctx_size") or 0)
        parallel = int(preview.get("parallel") or 1)
        estimate = preview.get("estimate_mb") or {}
        total_mb = estimate.get("total") if isinstance(estimate, Mapping) else None
        details: list[str] = [
            f"context {ctx} tokens x {parallel} slot(s) = {ctx * parallel} total",
            f"KV cache {preview.get('kv_cache_type', 'f16')}"
            f" · flash-attn {preview.get('flash_attn', 'auto')}",
        ]
        if total_mb is not None:
            details.append(f"projected VRAM {float(total_mb) / 1024:.2f} GiB")
        evict = [str(e) for e in (preview.get("evict_model_ids") or [])]
        if evict:
            details.append("would evict: " + ", ".join(evict))
        return FitVerdict(
            fits=True,
            headline=headline,
            detail_lines=details,
            notes=notes,
            per_gpu=_int_keyed(preview.get("per_gpu_bytes")),
            fp4_warning=fp4,
        )

    details = []
    required = preview.get("required_bytes")
    available = preview.get("available_bytes")
    if required is not None or available is not None:
        details.append(f"needs {format_gib(required)}, {format_gib(available)} usable")
    reason = preview.get("reason")
    if reason:
        details.append(str(reason))
    max_ctx = preview.get("max_ctx_that_fits")
    if max_ctx:
        details.append(f"largest context that would fit: {int(max_ctx)} tokens")
    return FitVerdict(
        fits=False,
        headline="Will not fit in VRAM",
        detail_lines=details,
        notes=notes,
        suggestions=[str(s) for s in (preview.get("suggestions") or [])],
        per_gpu=_int_keyed(preview.get("per_gpu_free")),
        fp4_warning=fp4,
    )


def fit_verdict_text(preview: Mapping[str, Any]) -> str:
    """One-string form of :func:`fit_verdict`, for tooltips and tests."""
    return fit_verdict(preview).as_text()


# ---------------------------------------------------------------------------
# Optimal settings per hardware mode (D36)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacementLine:
    """One hardware mode, as the Models tab shows it.

    The settings half (``summary``) is what this model can do on those cards
    **with them idle**, and the availability half (``availability``) is what
    stands in the way right now. They are separate strings because they answer
    separate questions and change on different timescales: the first is a
    property of the hardware, the second of the last ten seconds.
    """

    mode: str
    label: str
    devices: list[int]
    summary: str
    availability: str
    fits_now: bool
    would_evict: list[str]
    load_args: dict[str, Any]

    @property
    def colour(self) -> str:
        if self.fits_now:
            return "positive"
        return "warning" if self.would_evict else "negative"

    @property
    def text(self) -> str:
        return f"{self.label}: {self.summary} · {self.availability}"


def _tps(value: Any) -> str:
    try:
        return f"~{float(value):.0f} t/s"
    except (TypeError, ValueError):
        return "speed unknown"


def placement_availability(entry: Mapping[str, Any]) -> str:
    """ "fits now" / "needs 2 unloads" / "does not fit" for one mode."""
    if entry.get("optimal") is None:
        return "too small for this model"
    if entry.get("fits_now"):
        return "fits now"
    victims = list(entry.get("would_evict") or [])
    if victims:
        plural = "" if len(victims) == 1 else "s"
        return f"needs {len(victims)} unload{plural} ({', '.join(victims)})"
    return "does not fit right now"


def placement_lines(profiles: Mapping[str, Any]) -> list[PlacementLine]:
    """Render ``ModelManager.placement_profiles`` for the settings dialog.

    Pure, so the wording is testable without a browser -- the Models tab is the
    one surface where a user compares "what would this do on the 3090s" against
    "what would it do on everything", and a line that reads well is the whole
    feature.
    """
    lines: list[PlacementLine] = []
    for entry in profiles.get("modes") or profiles.get("profiles") or []:
        optimal = entry.get("optimal")
        if optimal is None:
            summary = "does not fit on these cards even when empty"
        else:
            slots = int(optimal.get("max_parallel") or 1)
            kv = str(optimal.get("kv_cache_type") or "?")
            if optimal.get("kv_cache_type_v") and optimal["kv_cache_type_v"] != kv:
                kv = f"{kv}/{optimal['kv_cache_type_v']}"
            summary = (
                f"{optimal.get('ctx_per_slot')} ctx · {kv} · "
                f"{slots} slot{'' if slots == 1 else 's'} · "
                f"{_tps(optimal.get('est_gen_tps'))} "
                f"({_tps(optimal.get('est_gen_tps_full_ctx')).removesuffix(' t/s')} full)"
            )
        lines.append(
            PlacementLine(
                mode=str(entry.get("mode") or ""),
                label=str(entry.get("label") or entry.get("mode") or ""),
                devices=[int(d) for d in (entry.get("devices") or [])],
                summary=summary,
                availability=placement_availability(entry),
                fits_now=bool(entry.get("fits_now")),
                would_evict=[str(v) for v in (entry.get("would_evict") or [])],
                load_args=dict((optimal or {}).get("load_args") or {}),
            )
        )
    return lines


def placement_headline(profiles: Mapping[str, Any]) -> str:
    """One sentence naming the default placement, for the dialog header."""
    lines = [line for line in placement_lines(profiles) if line.load_args]
    if not lines:
        return "No hardware mode on this box can hold this model."
    best = lines[0]
    return f"Recommended: {best.text}"


# ---------------------------------------------------------------------------
# Slot counts and "load at exactly this context" (WP19 / D37)
# ---------------------------------------------------------------------------


def parallel_summary(entry: Mapping[str, Any]) -> str:
    """ "2 of 7 slots (measured)" for one placement, or ``""`` when nothing was learned.

    Two cases are worth a line and one is not.

    *Worth it:* the recommendation is **below** the ceiling (a real trade-off the
    user should see), or it was **measured** at all -- a sweep that confirms the
    estimate is still a fact about this placement, and "measured" beside the
    ceiling is how the user knows the button has already been pressed.

    *Not worth it:* an estimate that equals the ceiling, which is every
    placement on a rig that has never benchmarked anything. D17's knee is
    already folded into ``max_parallel``, so that line would print one number
    twice on every row of the dialog.
    """
    optimal = entry.get("optimal") or {}
    ceiling = int(optimal.get("max_parallel") or 0)
    recommended = optimal.get("recommended_parallel")
    if not ceiling or recommended is None:
        return ""
    basis = str(optimal.get("recommended_parallel_basis") or "estimated")
    if int(recommended) == ceiling and basis != "measured":
        return ""
    return f"{int(recommended)} of {ceiling} slots ({basis})"


@dataclass(frozen=True)
class CtxButton:
    """One "load at this exact context" button in the Models tab."""

    label: str
    ctx_size: int
    enabled: bool
    tooltip: str


#: The four windows the Models tab offers as buttons. The same list the MCP
#: ``load_recommended`` docstring names, so a user and an agent asking for "256k"
#: mean the same load.
CTX_BUTTONS: Final[tuple[tuple[str, int], ...]] = (
    ("64k", 65536),
    ("128k", 131072),
    ("256k", 262144),
    ("512k", 524288),
)


def ctx_buttons(profiles: Mapping[str, Any]) -> list[CtxButton]:
    """The four context buttons, each enabled or greyed with the reason why.

    Greyed rather than hidden: "why can this model not do 256k" is the question
    the button is there to answer, and a control that vanishes answers nothing.
    Two reasons it can be off, and they are different -- past the model's trained
    window is permanent and about the model, while "no placement reaches it"
    is about this box and may change when something unloads.
    """
    trained = int(profiles.get("n_ctx_train") or 0)
    reachable = max(
        (
            int((entry.get("optimal") or {}).get("ctx_per_slot") or 0)
            for entry in profiles.get("modes") or profiles.get("profiles") or []
        ),
        default=0,
    )
    out: list[CtxButton] = []
    for label, ctx in CTX_BUTTONS:
        if trained and ctx > trained:
            out.append(
                CtxButton(
                    label,
                    ctx,
                    False,
                    f"this model was trained to {trained} tokens; asking for more "
                    f"needs RoPE scaling and degrades quality",
                )
            )
        elif reachable and ctx > reachable:
            out.append(
                CtxButton(
                    label,
                    ctx,
                    False,
                    f"no set of cards on this box reaches {ctx} tokens for this "
                    f"model even when idle (the best is {reachable})",
                )
            )
        else:
            out.append(
                CtxButton(
                    label,
                    ctx,
                    True,
                    f"load at exactly {ctx} tokens per conversation; the server "
                    f"picks the GPUs, the KV cache and the slot count",
                )
            )
    return out


def parallel_level_lines(report: Mapping[str, Any]) -> list[str]:
    """The measured curve as one aligned line per concurrency level."""
    lines = [f"{'N':>2}  {'per-stream':>10}  {'aggregate':>9}  {'p95':>7}  {'batch':>5}"]
    for level in report.get("levels") or []:
        lines.append(
            f"{int(level.get('n_streams') or 0):>2}  "
            f"{_tps(level.get('per_stream_tps')):>10}  "
            f"{_tps(level.get('aggregate_tps')):>9}  "
            f"{_secs(level.get('p95_latency_s')):>7}  "
            f"{_batch(level.get('achieved_batch')):>5}"
        )
    return lines


def parallel_verdict(report: Mapping[str, Any]) -> str:
    """One sentence naming the recommendation and why it landed there."""
    detail = str(report.get("recommended_parallel_detail") or "")
    basis = str(report.get("recommended_parallel_basis") or "estimated")
    return f"Recommended: {detail}" if detail else f"Recommended parallel basis: {basis}"


def _secs(value: Any) -> str:
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "-"


def _batch(value: Any) -> str:
    try:
        return f"{float(value):.1f}x"
    except (TypeError, ValueError):
        return "-"


def per_gpu_projection_lines(verdict: FitVerdict) -> list[str]:
    """``GPU0  12.34 GiB`` lines for the projected/free per-GPU map."""
    label = "projected" if verdict.fits else "free"
    return [f"GPU{index}: {format_gib(value)} {label}" for index, value in verdict.per_gpu]


# ---------------------------------------------------------------------------
# Deep links (HuggingFace "use this model" button)
# ---------------------------------------------------------------------------

#: Query ``tab=`` value -> the tab to open. Aliases are accepted because the
#: value comes from a URL a user (or another app) can type, not from us.
DEEP_LINK_TABS: Final[Mapping[str, str]] = {
    "dashboard": "Dashboard",
    "setup": "Setup",
    "models": "Models",
    "model": "Models",
    "download": "Download",
    "downloads": "Download",
    "chat": "Chat",
    "server": "Server",
    "settings": "Server",
    "logs": "Logs",
    "log": "Logs",
}

#: ``owner/repo`` and nothing else. Strict on purpose: this value is
#: interpolated into a HuggingFace API *path*, so a permissive parse would let
#: ``../..`` in a query string walk to another endpoint.
_REPO_ID_RE: Final = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")

#: Model ids are library-relative paths, so slashes and dots are legitimate --
#: but never a traversal segment or a control character.
_MODEL_ID_RE: Final = re.compile(r"^[^\x00-\x1f<>\"'\\]{1,300}$")


def _first_value(query: Mapping[str, Any], name: str) -> str | None:
    """One query value, tolerating both plain dicts and multi-value mappings."""
    if name not in query:
        return None
    value = query[name]
    if isinstance(value, str):
        text = value
    elif isinstance(value, Sequence) and value:
        text = str(value[0])
    else:
        text = str(value)
    text = text.strip()
    return text or None


def deep_link_params(query: Mapping[str, Any] | None) -> dict[str, str | None]:
    """Normalise the landing-page query string into an intent.

    This is what turns HuggingFace's "download to LM Studio" button into
    StudioForge's quant picker: the protocol handler resolves the deep link to
    ``/?tab=download&repo=owner/repo[&quant=Q4_K_M]`` and this reads it back.

    Every field is optional and every malformed field degrades to ``None`` with
    a human-readable ``error``, because a plain ``GET /`` with no query string
    must still land on the Dashboard exactly as before.
    """
    blank: dict[str, str | None] = {
        "tab": None,
        "repo": None,
        "quant": None,
        "model": None,
        "error": None,
    }
    if not query:
        return blank

    result = dict(blank)
    raw_tab = _first_value(query, "tab")
    if raw_tab:
        tab = DEEP_LINK_TABS.get(raw_tab.lower())
        if tab is None:
            result["error"] = f"unknown tab {raw_tab!r} in the link; showing the dashboard"
        result["tab"] = tab

    raw_repo = _first_value(query, "repo") or _first_value(query, "model_repo")
    if raw_repo:
        if _REPO_ID_RE.match(raw_repo):
            result["repo"] = raw_repo
        else:
            result["error"] = (
                f"{raw_repo!r} is not a valid HuggingFace repository id; expected 'owner/repo'"
            )

    result["quant"] = _first_value(query, "quant")

    raw_model = _first_value(query, "model")
    if raw_model and raw_model != raw_repo:
        if _MODEL_ID_RE.match(raw_model) and ".." not in raw_model:
            result["model"] = raw_model
        else:
            result["error"] = f"{raw_model!r} is not a usable model id"

    # ``?tab=download&model=owner/repo`` is the same intent spelled differently;
    # honour it rather than sending the user to a Models tab with no such model.
    model = result["model"]
    if result["tab"] == "Download" and not result["repo"] and model and _REPO_ID_RE.match(model):
        result["repo"], result["model"] = model, None

    # A link that names a target but not a (valid) tab still knows where it
    # wants to go, and landing there is where any error message is shown.
    if result["tab"] is None:
        if result["repo"]:
            result["tab"] = "Download"
        elif result["model"]:
            result["tab"] = "Models"
    return result


def initial_tab(params: Mapping[str, Any] | None, *, default: str = "Dashboard") -> str:
    """Which tab the page opens on. Unchanged (Dashboard) without a deep link."""
    if not params:
        return default
    tab = params.get("tab")
    return str(tab) if tab in DEEP_LINK_TABS.values() else default


def normalise_quant(quant: str | None) -> str:
    """Quant labels for comparison: case- and separator-insensitive.

    ``q4_k_m``, ``Q4_K_M`` and ``Q4-K-M`` all name the same file; a deep link
    written by hand should still highlight the right row.
    """
    if not quant:
        return ""
    return re.sub(r"[^a-z0-9]", "", quant.strip().lower())


def quant_matches(left: str | None, right: str | None) -> bool:
    normalised = normalise_quant(left)
    return bool(normalised) and normalised == normalise_quant(right)


def deep_link_headline(repo_id: str, quant: str | None) -> str:
    """Title for the landing card, naming the quant when the link picked one."""
    if quant:
        return f"{repo_id} — {quant}"
    return repo_id


# ---------------------------------------------------------------------------
# Download queue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadGroupRow:
    """One *logical* download, aggregated from its per-file progress rows.

    The downloader tracks files, but a sharded model plus its projector is one
    thing the user asked for and one thing they want a single progress bar for.
    Showing five bars for a five-part model is noise, and worse, none of them
    answers "is my model ready?".
    """

    group_id: str
    label: str
    status: str
    done_bytes: int
    total_bytes: int
    speed_bps: float
    eta_s: float | None
    error: str | None
    file_count: int

    @property
    def fraction(self) -> float:
        return progress_fraction(self.done_bytes, self.total_bytes)

    @property
    def detail(self) -> str:
        parts = [
            format_percent(self.fraction),
            f"{format_bytes(self.done_bytes)} of {format_bytes(self.total_bytes)}",
            format_rate(self.speed_bps),
            f"ETA {format_eta(self.eta_s)}",
        ]
        if self.file_count > 1:
            parts.append(f"{self.file_count} files")
        return " · ".join(parts)


#: Worst-first, mirroring ``Downloader.group_status``. Only used when the live
#: downloader is not available to ask; it is the authority when it is.
_DOWNLOAD_STATUS_ORDER: Final = ("failed", "running", "queued", "paused", "canceled", "completed")


def _fallback_group_status(statuses: Sequence[str]) -> str:
    for candidate in _DOWNLOAD_STATUS_ORDER:
        if candidate in statuses:
            return candidate
    return "queued"


def group_download_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    status_for: Any = None,
) -> list[DownloadGroupRow]:
    """Collapse ``DownloadProgress.to_dict()`` rows into one row per group.

    ``status_for`` should be ``Downloader.group_status`` so the displayed status
    comes from the component that owns the rule (a group is only complete when
    every file is -- a model missing shard 3 of 5 is not usable).
    """
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or row.get("id") or "?")
        grouped.setdefault(group_id, []).append(row)

    out: list[DownloadGroupRow] = []
    for group_id, files in grouped.items():
        statuses = [str(f.get("status") or "queued") for f in files]
        status = _fallback_group_status(statuses)
        if status_for is not None:
            # Suppressed on purpose: the authoritative status is nice to have,
            # but a progress row must still render without it.
            with contextlib.suppress(Exception):
                status = str(status_for(group_id))
        etas = [float(f["eta_s"]) for f in files if isinstance(f.get("eta_s"), int | float)]
        errors = [str(f["error"]) for f in files if f.get("error")]
        repo = next((str(f.get("repo_id")) for f in files if f.get("repo_id")), group_id)
        out.append(
            DownloadGroupRow(
                group_id=group_id,
                label=repo,
                status=status,
                done_bytes=sum(int(f.get("downloaded_bytes") or 0) for f in files),
                total_bytes=sum(int(f.get("total_bytes") or 0) for f in files),
                speed_bps=sum(float(f.get("speed_bps") or 0.0) for f in files),
                eta_s=max(etas) if etas else None,
                error=errors[0] if errors else None,
                file_count=len(files),
            )
        )
    return sorted(out, key=lambda row: row.group_id)


DOWNLOAD_STATUS_COLOURS: Final[Mapping[str, str]] = {
    "running": "primary",
    "queued": "grey",
    "paused": "warning",
    "completed": "positive",
    "failed": "negative",
    "canceled": "grey",
}


def download_status_colour(status: str) -> str:
    return DOWNLOAD_STATUS_COLOURS.get(status, "grey")


#: Which controls make sense per download status. A completed download offers
#: nothing: pausing it is meaningless and a "Cancel (deletes the partial
#: file)" button next to a finished 40 GiB model reads as "delete my model"
#: even though the downloader would actually refuse.
_DOWNLOAD_ACTIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "queued": ("pause", "cancel"),
    "running": ("pause", "cancel"),
    "paused": ("resume", "cancel"),
    "failed": ("resume", "cancel"),
    "canceled": ("resume",),
    "completed": (),
}


def download_actions(status: str) -> list[str]:
    """The row controls applicable to one download group's status.

    An unknown status (a future downloader state) gets every control rather
    than none, because a stuck download with no buttons is unrecoverable from
    the GUI.
    """
    return list(_DOWNLOAD_ACTIONS.get(status, ("pause", "resume", "cancel")))


def download_fit_verdict(
    total_bytes: int | None,
    gpus: Sequence[GpuInfo],
    *,
    headroom_fraction: float = 0.10,
) -> str:
    """First-paint fit steer for the Download tab's quant picker.

    Deliberately crude -- weights only, no KV, no compute buffers, no CUDA
    context -- because the point is to steer a *download* choice before any file
    exists, not to replace the planner. It is labelled as an estimate for
    exactly that reason.

    It is now a **fallback**. The picker reads the model's GGUF header a beat
    after the rows appear, and once it has, :func:`fit_badge_from_context` gives
    the planner's own answer and replaces the badge. The two disagree on real
    hardware: a 27.9 GiB Q8_0 "fits one GPU" by this arithmetic while the
    planner cannot place it on a 32 GiB card at any context, because weights
    are not the whole cost. Kept because it is the only answer available in the
    window before the header lands, and because that window is not always short
    (a gated repo never gets a header at all).
    """
    if not total_bytes:
        return "size unknown"
    if not gpus:
        return "no GPU detected"
    factor = max(0.0, 1.0 - headroom_fraction)
    usable = [int(g.free_bytes * factor) for g in gpus]
    best = max(usable)
    if total_bytes <= best:
        index = usable.index(best)
        return f"fits one GPU (GPU{gpus[index].index}, {format_gib(best)} usable)"
    if total_bytes <= sum(usable):
        return f"needs multiple GPUs ({format_gib(sum(usable))} usable in total)"
    return f"will not fit ({format_gib(total_bytes)} vs {format_gib(sum(usable))} usable)"


#: The three badge texts, kept byte-identical to the openings of
#: :func:`download_fit_verdict` so a row does not appear to change its verdict
#: when the header arrives and only the *source* of the answer changed.
_FIT_BADGES: Final[Mapping[str, tuple[str, str]]] = {
    "single": ("fits one GPU", "positive"),
    "multi": ("needs multiple GPUs", "warning"),
    "none": ("will not fit", "negative"),
}


def fit_badge_from_context(context_fit: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """``(label, colour)`` from the planner's own placement answers.

    ``context_fit`` is ``hf_meta.context_matrix``'s output, whose
    ``placements[*].weights_fit`` is the planner deciding whether the weights,
    the compute buffers and the CUDA context fit a given set of devices -- the
    same ``_try_devices`` a real load runs. That is a strictly better answer
    than :func:`download_fit_verdict`'s file-size comparison, which is how a row
    could read ``fits one GPU`` beside a context line of ``1x5090: --``.

    Returns ``None`` when there is nothing better to say, and the caller keeps
    the weights-only badge: no matrix, no placements (a box with no usable GPU),
    or no ``source`` -- an approximate matrix got its numbers from the bounded
    pre-download allowance, which is not the planner reading a header and so is
    no more authoritative than the badge it would be replacing.
    """
    if not context_fit or not context_fit.get("source"):
        return None
    placements = context_fit.get("placements") or []
    if not placements:
        return None
    single = next((p for p in placements if p.get("key") == "single_best"), None)
    if single is not None and single.get("weights_fit"):
        return _FIT_BADGES["single"]
    if any(p.get("weights_fit") for p in placements):
        return _FIT_BADGES["multi"]
    return _FIT_BADGES["none"]


# ---------------------------------------------------------------------------
# Disk space
# ---------------------------------------------------------------------------


def disk_line(report: Mapping[str, Any] | None) -> str:
    """One line of ``core.diskspace.disk_report``, for the top of the queue.

    Reads as a sentence about the future rather than the present, because the
    present is not the question: what the user wants to know before pressing
    Download is what the disk looks like *after* everything already queued has
    landed. The queued clause is dropped when nothing is queued -- "0 B queued
    → ~412 GiB after downloads" is arithmetic theatre.

    Binary units throughout, like every other size in this GUI: a "38 GB
    queued" line next to a "35.4 GiB of 41.2 GiB" progress row describing the
    same bytes looks like a bug.
    """
    if not report:
        return ""
    if int(report.get("total_bytes") or 0) <= 0:
        reason = str(report.get("error") or "")
        return f"Disk: unavailable{f' ({reason})' if reason else ''}"
    free = int(report.get("free_bytes") or 0)
    queued = int(report.get("queued_bytes") or 0)
    after = int(report.get("free_after_queue_bytes") or 0)
    line = f"Disk: {format_bytes(free)} free on {report.get('drive') or '?'}"
    if queued > 0:
        tail = (
            f"~{format_bytes(after)} after downloads"
            if after >= 0
            else f"{format_bytes(-after)} SHORT"
        )
        line += f" · {format_bytes(queued)} queued → {tail}"
    return line


def disk_is_low(report: Mapping[str, Any] | None) -> bool:
    """Whether the disk line should be shouted rather than muttered."""
    return bool(report and report.get("low"))


def disk_would_overflow(report: Mapping[str, Any] | None, needed_bytes: int | None) -> bool:
    """Would downloading *needed_bytes* more not fit in what the queue leaves?

    ``False`` whenever the volume could not be measured. A "not enough disk"
    warning the user cannot verify -- on a row they can see is 12 GiB against a
    drive they know has room -- teaches them to ignore the one that is real.
    """
    if not report or not needed_bytes:
        return False
    if int(report.get("total_bytes") or 0) <= 0:
        return False
    return int(needed_bytes) > int(report.get("free_after_queue_bytes") or 0)


# ---------------------------------------------------------------------------
# Context / slots
# ---------------------------------------------------------------------------


def total_ctx_tokens(ctx: int | None, parallel: int | None, *, default_ctx: int = 8192) -> int:
    effective_ctx = int(ctx) if ctx else default_ctx
    slots = max(1, int(parallel or 1))
    return effective_ctx * slots


def per_slot_ctx_hint(ctx: int | None, parallel: int | None, *, default_ctx: int = 8192) -> str:
    """Explain the ctx/parallel interaction -- the most misread knob here.

    llama-server's ``--ctx-size`` is the TOTAL budget shared by all slots
    (DECISIONS D4), so StudioForge multiplies the per-conversation number the
    user typed by the slot count when it launches. Both numbers are named
    because users otherwise assume one of the two wrong things: that raising
    ``parallel`` is free, or that it quarters the context they asked for.
    """
    slots = max(1, int(parallel or 1))
    if slots == 1:
        return ""
    effective_ctx = int(ctx) if ctx else default_ctx
    total = effective_ctx * slots
    return (
        f"Context is split across slots: llama-server's --ctx-size is the TOTAL budget, so "
        f"StudioForge launches with {effective_ctx} x {slots} = {total} tokens and each of the "
        f"{slots} slots gets the {effective_ctx} tokens you asked for. "
        f"{slots} slots therefore cost {slots}x the KV cache."
    )


# ---------------------------------------------------------------------------
# Settings form <-> ModelSettings
# ---------------------------------------------------------------------------


def parse_device_list(value: Any) -> list[int] | None:
    """``"0, 1"`` / ``[0, 1]`` -> ``[0, 1]``; blank -> ``None`` (i.e. "Auto")."""
    if value is None:
        return None
    if isinstance(value, str):
        tokens = [t for t in re.split(r"[,\s]+", value.strip()) if t]
    elif isinstance(value, Sequence):
        tokens = [str(v) for v in value]
    else:
        tokens = [str(value)]
    devices: list[int] = []
    for token in tokens:
        try:
            devices.append(int(token))
        except ValueError as exc:
            raise ValueError(f"device list must be integers, got {token!r}") from exc
    return devices or None


def format_device_list(devices: Sequence[int] | None) -> str:
    if not devices:
        return ""
    return ",".join(str(int(d)) for d in devices)


def form_from_settings(settings: ModelSettings) -> dict[str, Any]:
    """Flat form dict mirroring :class:`ModelSettings` field-for-field.

    ``None`` becomes ``""`` only for the fields rendered as text/select inputs
    (an empty input is how the UI spells "Auto"); numeric and tri-state fields
    keep ``None`` so a NiceGUI number/toggle can bind straight to them.
    """
    form: dict[str, Any] = {}
    for name in ModelSettings.model_fields:
        value = getattr(settings, name)
        if name == "adapters":
            form[name] = [attachment.model_dump(mode="python") for attachment in value]
        elif name in _DEVICE_LIST_FIELDS:
            form[name] = format_device_list(value)
        elif value is None and _is_text_field(name):
            form[name] = ""
        else:
            form[name] = value
    return form


def _is_text_field(name: str) -> bool:
    """Whether a settings field is rendered as a text/select input.

    Those are the fields where ``None`` must become ``""`` for the widget (an
    empty select is how the UI spells "Auto"). Determined from the annotation
    rather than a hand-maintained list, so a new ``Literal[...]`` field added to
    :class:`ModelSettings` is handled without touching this module.
    """
    return _has_str_member(ModelSettings.model_fields[name].annotation)


def _has_str_member(annotation: Any) -> bool:
    if annotation is str:
        return True
    origin = typing.get_origin(annotation)
    if origin is typing.Literal:
        return any(isinstance(arg, str) for arg in typing.get_args(annotation))
    if origin in (typing.Union, types.UnionType):
        return any(_has_str_member(arg) for arg in typing.get_args(annotation))
    return False


def settings_from_form(form: Mapping[str, Any]) -> ModelSettings:
    """Validate a form dict into :class:`ModelSettings`.

    Blank means ``None`` means "inherit / let the planner decide"; the one
    exception is ``extra_flags``, where the empty string is the real value.
    """
    payload: dict[str, Any] = {}
    for name in ModelSettings.model_fields:
        if name not in form:
            continue
        value = form[name]
        if name == "adapters":
            payload[name] = _adapters_from_form(value)
            continue
        if name in _DEVICE_LIST_FIELDS:
            payload[name] = parse_device_list(value)
            continue
        if name in _KEEP_EMPTY_TEXT:
            payload[name] = "" if value is None else str(value).strip()
            continue
        if isinstance(value, str):
            value = value.strip() or None
        if value is None and name in _NON_OPTIONAL_BOOL:
            payload[name] = False
            continue
        payload[name] = value
    return ModelSettings.model_validate(payload)


def _adapters_from_form(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, Mapping):
            out.append({"adapter_id": item.get("adapter_id"), "scale": item.get("scale", 1.0)})
    return out


#: Explains DECISIONS D12 in the settings dialog. Measured, not theoretical:
#: with llama.cpp's own default a reasoning model returns an EMPTY ``content``
#: and puts the text in ``reasoning_content``, which standard OpenAI clients
#: (OpenClaw included) never read -- so the reply looks blank.
REASONING_FORMAT_HELP: Final = (
    "Leave this on 'none' unless you know your client reads reasoning_content. "
    "'none' keeps a thinking model's output in the standard message.content field. "
    "'deepseek' moves the thoughts into a separate reasoning_content field, which is "
    "NOT part of the OpenAI schema — most clients ignore it and show an empty reply "
    "(measured: content length 0 vs 323 for the same prompt)."
)


def reasoning_format_hint(value: str | None, default: str) -> str:
    effective = value or default
    if effective == "none":
        return f"reasoning_format = {effective} (inline in message.content — safe for any client)"
    return (
        f"reasoning_format = {effective}: thoughts go to message.reasoning_content, which "
        "most OpenAI clients ignore. Expect an apparently empty reply unless your client "
        "reads that field."
    )


def cache_reuse_hint(value: int | None, default: int) -> str:
    """Explain what an untouched (``None``) prompt-cache-reuse field will do.

    Prompt-cache reuse is the single biggest real-world latency win for agent
    workloads -- OpenClaw re-sends near-identical long prompts constantly -- so
    the default is ON and the UI says so rather than leaving a bare number.
    """
    effective = default if value is None else value
    state = "ON" if effective > 0 else "OFF"
    inherited = " (inherited default)" if value is None else ""
    return (
        f"Prompt-cache reuse is {state} at {effective} tokens{inherited}. "
        "This is the biggest real-world latency win for agent workloads: it lets "
        "llama-server reuse the cached prefix of a near-identical prompt instead of "
        "reprocessing it."
    )


# ---------------------------------------------------------------------------
# Draft model plausibility
# ---------------------------------------------------------------------------


def _arch_family(record: ModelRecord) -> str:
    arch = (record.architecture or "unknown").strip().lower()
    return arch.split("-")[0] if arch else "unknown"


def _vocab_of(record: ModelRecord) -> int | None:
    meta = record.meta
    if meta is None or not meta.n_vocab:
        return None
    return int(meta.n_vocab)


def plausible_draft_models(
    records: Sequence[ModelRecord], target: ModelRecord
) -> list[ModelRecord]:
    """Candidate draft models for speculative decoding against ``target``.

    Speculative decoding requires a *shared tokenizer*: a mismatched vocab
    produces garbage rather than an error, so a known-different vocab size is a
    hard exclusion. Where the vocab is unknown on either side we fall back to
    the architecture family and let :func:`draft_uncertainty_note` say the
    pairing is a guess. A draft that is not smaller than its target can never
    pay off, so those are excluded too.
    """
    target_vocab = _vocab_of(target)
    target_family = _arch_family(target)
    candidates: list[ModelRecord] = []
    for record in records:
        if record.id == target.id:
            continue
        if record.kind != target.kind:
            continue
        if record.is_virtual:
            continue
        vocab = _vocab_of(record)
        if target_vocab is not None and vocab is not None:
            if vocab != target_vocab:
                continue
        elif _arch_family(record) != target_family:
            # No vocab to compare, and not even the same family: too likely to
            # be a silent mismatch to offer.
            continue
        if record.size_bytes and target.size_bytes and record.size_bytes >= target.size_bytes:
            continue
        candidates.append(record)
    return sorted(candidates, key=lambda r: (r.size_bytes, r.id))


def draft_uncertainty_note(target: ModelRecord, draft: ModelRecord | None) -> str | None:
    """Warning text when a draft pairing cannot be verified, else ``None``."""
    if draft is None:
        return None
    target_vocab, draft_vocab = _vocab_of(target), _vocab_of(draft)
    if target_vocab is not None and draft_vocab is not None:
        if target_vocab != draft_vocab:
            return (
                f"vocab mismatch: {target.id} has {target_vocab} tokens, "
                f"{draft.id} has {draft_vocab}. Speculative decoding needs a shared vocabulary."
            )
        target_tok = (target.meta.tokenizer_model if target.meta else "") or ""
        draft_tok = (draft.meta.tokenizer_model if draft.meta else "") or ""
        if target_tok and draft_tok and target_tok != draft_tok:
            return (
                f"vocab sizes match but the tokenizer differs ({target_tok} vs {draft_tok}); "
                "verify output quality before relying on this pairing."
            )
        return None
    return (
        "vocab size is unknown for one of these models, so tokenizer compatibility could not "
        "be verified. If speculative decoding produces garbage, this pairing is why."
    )


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def masked_secret(value: str | None) -> str:
    """Display form of a secret: never the secret itself."""
    return redact(value) or ""


def masked_secret_changed(old_masked: str | None, new_value: str | None) -> bool:
    """Whether a masked field holds a genuinely new secret worth sending.

    The bug this exists to prevent: the config endpoint hands the GUI
    ``"abcd...yz"`` instead of the real key, and a naive "save everything on
    the form" would PATCH that placeholder back, silently replacing the real
    API key with nine literal characters and locking every client out. So a
    value is only sent when it differs from the placeholder *and* does not
    itself look like one.
    """
    if new_value is None:
        return False
    new = new_value.strip()
    if not new:
        return False
    if new == (old_masked or "").strip():
        return False
    return not (new == "***" or _REDACTION_RE.match(new))


# ---------------------------------------------------------------------------
# Config form description (Server tab)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigField:
    """One editable config key on the Server tab."""

    key: str
    label: str
    kind: str  # "text" | "int" | "float" | "bool" | "secret" | "select" | "list"
    help: str = ""
    options: tuple[str, ...] = ()

    @property
    def restart_required(self) -> bool:
        return self.key in RESTART_REQUIRED_KEYS


CONFIG_FIELDS: Final[tuple[ConfigField, ...]] = (
    ConfigField("server.host", "Bind address", "text", "0.0.0.0 exposes it on the tailnet"),
    ConfigField("server.port", "Gateway port", "int", "LM Studio's default is 1234"),
    ConfigField("server.api_key", "API key", "secret", "Blank disables auth (LAN/tailnet trust)"),
    ConfigField("server.cors_origins", "CORS origins", "list", "Comma separated; * allows all"),
    ConfigField("models.dir", "Model directory", "text", "Primary GGUF library root"),
    ConfigField("models.default_ctx", "Default context", "int", "Used when a model says Auto"),
    ConfigField("models.default_ttl_s", "Default TTL (s)", "int", "0 means never idle-unload"),
    ConfigField(
        "models.default_cache_reuse",
        "Default prompt-cache reuse",
        "int",
        "Biggest latency win for agent workloads; keep it above 0",
    ),
    ConfigField(
        "planner.headroom_fraction",
        "VRAM headroom",
        "float",
        "Fraction of each GPU held back from the planner",
    ),
    ConfigField(
        "planner.on_insufficient",
        "When VRAM is short",
        "select",
        "evict = unload LRU unpinned models; reject = refuse the load",
        ("evict", "reject"),
    ),
    ConfigField(
        "planner.preference",
        "Optimise loads for",
        "select",
        "quality = best KV cache first; throughput = biggest window first",
        ("quality", "throughput"),
    ),
    ConfigField("hf.token", "HuggingFace token", "secret", "Needed for gated repos"),
    ConfigField("gui.port", "GUI port", "int", "This panel's own port"),
)

SECRET_CONFIG_KEYS: Final = frozenset(f.key for f in CONFIG_FIELDS if f.kind == "secret")


def config_value(config_dict: Mapping[str, Any], dotted: str) -> Any:
    """Read a dotted key out of ``GET /api/config``'s nested payload."""
    cursor: Any = config_dict
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def restart_required_keys(changed: Sequence[str]) -> list[str]:
    return sorted(key for key in changed if key in RESTART_REQUIRED_KEYS)


def coerce_config_value(kind: str, raw: Any) -> Any:
    """Turn a form value into the type the config model expects."""
    if kind == "int":
        return None if raw in (None, "") else int(raw)
    if kind == "float":
        return None if raw in (None, "") else float(raw)
    if kind == "bool":
        return bool(raw)
    if kind == "list":
        if isinstance(raw, str):
            return [part.strip() for part in raw.split(",") if part.strip()]
        return list(raw or [])
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


# ---------------------------------------------------------------------------
# First-run / setup readiness
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupItem:
    """One line of the Server tab's readiness checklist."""

    name: str
    ok: bool
    detail: str
    action: str = ""

    @property
    def icon(self) -> str:
        return "check_circle" if self.ok else "error"

    @property
    def colour(self) -> str:
        return "positive" if self.ok else "warning"


def setup_status(
    *,
    model_dir: Any,
    model_count: int,
    gpu_count: int,
    engine_tag: str | None,
    pinned_tag: str,
) -> list[SetupItem]:
    """Everything that must be true before the server can serve anything.

    Exists so a first run does not require hunting through tabs: each unmet
    condition names the one action that fixes it, and the Server tab renders a
    button for exactly that action.
    """
    return [
        SetupItem(
            "Model directory",
            bool(model_dir),
            str(model_dir) if model_dir else "not set — point models.dir at your GGUF library",
        ),
        SetupItem(
            "Models indexed",
            model_count > 0,
            f"{model_count} model(s) in the registry"
            if model_count
            else "no models indexed yet — run a scan",
            action="" if model_count else "scan",
        ),
        SetupItem(
            "GPUs detected",
            gpu_count > 0,
            f"{gpu_count} GPU(s) visible to NVML"
            if gpu_count
            else "no GPUs detected — this server is GPU-only, so nothing can load",
        ),
        SetupItem(
            "llama.cpp engine",
            bool(engine_tag),
            f"active: {engine_tag}" if engine_tag else f"not installed — install {pinned_tag}",
            action="" if engine_tag else "install-engine",
        ),
    ]


def setup_is_ready(items: Sequence[SetupItem]) -> bool:
    return all(item.ok for item in items)


def quant_affinity_summary(affinity: Mapping[str, Any] | None) -> list[str]:
    """One line per quant family in ``planner.quant_affinity``."""
    if not affinity:
        return ["no quant affinity configured"]
    lines: list[str] = []
    for family, spec in sorted(affinity.items()):
        if not isinstance(spec, Mapping):
            continue
        mode = spec.get("mode", "prefer")
        minimum = spec.get("min_compute_capability", "0.0")
        verb = "requires" if mode == "require" else "prefers"
        lines.append(f"{family}: {verb} compute capability >= {minimum}")
    return lines or ["no quant affinity configured"]


#: DECISIONS D9, stated positively on purpose. FP4 was *measured* running fine
#: on Ampere; the affinity only steers placement toward the faster card. Nothing
#: in this system is Blackwell-only, and presenting it that way would make users
#: think half their hardware was unusable.
QUANT_AFFINITY_NOTE: Final = (
    "'prefer' steers placement toward the faster card; it never excludes a GPU. "
    "FP4 quants were measured loading and generating correctly on the RTX 3090s "
    "(sm_86) — they get native tensor-core acceleration only on Blackwell, but every "
    "GPU here is fully usable for every model, FP4 included. 'require' is the only "
    "mode that excludes hardware, and a per-model device override outranks both."
)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def chat_model_records(records: Sequence[ModelRecord]) -> list[ModelRecord]:
    """The records the Chat tab may offer: chat-kind only.

    An embedding model has no chat endpoint, so offering it would fail at send
    time with an opaque error; excluding it here (and explaining the exclusion
    via :func:`hidden_chat_models_note`) is the prevented-not-broken path.
    """
    return [r for r in records if r.kind == "chat"]


def hidden_chat_models_note(records: Sequence[ModelRecord]) -> str | None:
    """Why some library models are missing from the Chat picker, or ``None``.

    Without this line a user who just downloaded an embedding model searches
    the dropdown, finds nothing, and concludes the scan failed -- the model is
    deliberately absent and the UI should say so.
    """
    hidden = [r for r in records if r.kind != "chat"]
    if not hidden:
        return None
    kinds = ", ".join(sorted({str(r.kind) for r in hidden}))
    return (
        f"{len(hidden)} model(s) not shown ({kinds}): only chat models can be used here. "
        "An embedding model has no chat endpoint — use /v1/embeddings instead."
    )


@dataclass(frozen=True)
class ChatTarget:
    """Which model the Chat tab will actually talk to, and why.

    The "use the loaded model" switch is a convenience over a real ambiguity --
    there can be nothing loaded, exactly one thing loaded, or several -- and each
    of those needs a different visible answer. Resolving it here means the tab
    never has to guess, and the awkward cases (nothing loaded; two models
    loaded; a loaded embedding model that cannot chat) are testable.
    """

    #: The model the next send goes to, or ``None`` when there is nothing to send to.
    model_id: str | None
    #: Whether the switch is *effectively* on. Never ``True`` when it is disabled.
    use_loaded: bool
    #: Whether the switch itself can be operated.
    switch_disabled: bool
    #: Why the switch cannot be used, or ``None`` when it can.
    disabled_reason: str | None
    #: Whether the manual model picker is greyed out.
    picker_disabled: bool
    #: The loaded model the switch targets, if any.
    loaded_id: str | None
    #: Other loaded chat models that the switch is *not* targeting.
    other_loaded: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """What is shown next to the switch."""
        if self.loaded_id is None:
            return self.disabled_reason or "no model is loaded"
        if self.other_loaded:
            return f"{self.loaded_id}  (most recently used of {1 + len(self.other_loaded)} loaded)"
        return self.loaded_id

    @property
    def note(self) -> str:
        """Extra line shown when the choice was not obvious, else ``""``."""
        if self.loaded_id is None or not self.other_loaded:
            return ""
        return (
            f"Also loaded: {', '.join(self.other_loaded)}. The switch follows the most "
            "recently used model, so it can change under you as other clients make "
            "requests — turn it off to pin the choice."
        )


def _instance_recency(instance: InstanceInfo) -> float:
    return float(instance.last_activity_at or instance.started_at or 0.0)


def chat_target(
    records: Sequence[ModelRecord],
    instances: Sequence[InstanceInfo],
    *,
    use_loaded: bool,
    manual_choice: str | None,
) -> ChatTarget:
    """Resolve the Chat tab's target model.

    Rules, each of which exists because the silent alternative is worse:

    * Nothing loaded -> the switch is **disabled with a reason**, not quietly
      on and pointing at nothing.
    * Something loaded but nothing that can chat (an embedding model) -> also
      disabled, and it says which model and why, rather than offering a target
      that would fail at send time.
    * Several loaded -> the most recently used one is chosen **and named**,
      with the others listed, so the choice is visible rather than arbitrary.
    * Switch off -> ``manual_choice`` is handed straight back, so turning the
      switch off restores what the user had picked instead of resetting the
      picker to the first model in the library.
    """
    chat_ids = {r.id for r in records if r.kind == "chat"}
    ready = [i for i in instances if i.state == "ready"]
    candidates = sorted(
        (i for i in ready if i.model_id in chat_ids),
        key=lambda i: (-_instance_recency(i), i.model_id.lower()),
    )

    if not candidates:
        if not ready:
            reason = "no model is loaded"
        else:
            names = ", ".join(sorted(i.model_id for i in ready))
            reason = (
                f"no chat model is loaded ({names} cannot chat — an embedding model has "
                "no chat endpoint)"
            )
        return ChatTarget(
            model_id=manual_choice or None,
            use_loaded=False,
            switch_disabled=True,
            disabled_reason=reason,
            picker_disabled=False,
            loaded_id=None,
        )

    loaded_id = candidates[0].model_id
    others = tuple(i.model_id for i in candidates[1:])
    if use_loaded:
        return ChatTarget(
            model_id=loaded_id,
            use_loaded=True,
            switch_disabled=False,
            disabled_reason=None,
            picker_disabled=True,
            loaded_id=loaded_id,
            other_loaded=others,
        )
    return ChatTarget(
        model_id=manual_choice or None,
        use_loaded=False,
        switch_disabled=False,
        disabled_reason=None,
        picker_disabled=False,
        loaded_id=loaded_id,
        other_loaded=others,
    )


def vision_attach_reason(record: ModelRecord | None) -> str | None:
    """Why image attachment is disabled, or ``None`` when it is available."""
    if record is None:
        return "Select a model first."
    if not record.capabilities.vision:
        return (
            f"{record.id} has no vision projector (mmproj), so it cannot accept images. "
            "Pick a model with the 'vision' badge."
        )
    return None


def build_chat_content(text: str, images: Sequence[str]) -> Any:
    """OpenAI message content: a bare string, or a content array with images.

    Images travel as base64 ``data:`` URLs so the GUI never needs a publicly
    reachable URL for a pasted screenshot -- which it could not have behind a
    tailnet anyway.
    """
    if not images:
        return text
    parts: list[dict[str, Any]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for url in images:
        parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def tokens_per_second(tokens: int, elapsed_s: float) -> float | None:
    if elapsed_s <= 0 or tokens <= 0:
        return None
    return round(tokens / elapsed_s, 2)


def number_value(raw: Any, default: float) -> float:
    """A numeric form field's value, defaulting only when genuinely unset.

    ``value or default`` is the bug this replaces: it silently turns an explicit
    ``0`` into the default, so a user who set temperature to 0 (greedy decoding,
    a perfectly normal choice) was served temperature 0.7 with no indication
    anywhere. Only ``None``, the empty string and unparseable input fall back.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

#: Shown when the benchmark subsystem is not wired into this build. It is an
#: optional subsystem and the Models tab must render perfectly without it, so
#: this is a normal state with an explanation, never an error card.
BENCHMARK_UNAVAILABLE_NOTE: Final = (
    "Benchmarking is not available in this build. Nothing else on this tab is "
    "affected — this panel appears only when the benchmark subsystem is present."
)


@dataclass(frozen=True)
class BenchmarkModeRow:
    """One selectable GPU placement to benchmark (1x 5090, 2x 3090, all, …)."""

    key: str
    label: str
    devices: tuple[int, ...] = ()
    gpu_name: str = ""
    applicable: bool = True
    skipped_reason: str | None = None

    @property
    def detail(self) -> str:
        devices = ", ".join(f"GPU{d}" for d in self.devices) if self.devices else "every GPU"
        return f"{devices}{f' · {self.gpu_name}' if self.gpu_name else ''}"

    @property
    def tooltip(self) -> str:
        """Never bare: a checkbox that cannot be ticked must say why."""
        if self.applicable:
            return f"Benchmark this model on {self.detail}."
        reason = self.skipped_reason or "this model does not fit in this configuration"
        return f"Cannot benchmark on {self.detail}: {reason}"


def _maybe_devices(value: Any) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return ()
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe_int(value: Any) -> int | None:
    number = _maybe_float(value)
    return None if number is None else int(number)


def benchmark_modes(payload: Mapping[str, Any] | None) -> list[BenchmarkModeRow]:
    """Rows for the mode picker, from either ``modes`` endpoint.

    The per-model endpoint adds ``applicable``/``skipped_reason``; the global one
    does not, and its modes are all applicable by definition. Both are read
    through this one function so the picker cannot drift between them.
    """
    modes = (payload or {}).get("modes")
    if not isinstance(modes, Sequence) or isinstance(modes, str):
        return []
    rows: list[BenchmarkModeRow] = []
    for item in modes:
        if not isinstance(item, Mapping):
            continue
        key = str(item.get("key") or "")
        if not key:
            continue
        reason = item.get("skipped_reason")
        rows.append(
            BenchmarkModeRow(
                key=key,
                label=str(item.get("label") or key),
                devices=_maybe_devices(item.get("devices")),
                gpu_name=str(item.get("gpu_name") or ""),
                applicable=bool(item.get("applicable", True)),
                skipped_reason=str(reason) if reason else None,
            )
        )
    return rows


def default_selected_modes(rows: Sequence[BenchmarkModeRow]) -> list[str]:
    """Pre-ticked modes: every applicable one, in the backend's order."""
    return [row.key for row in rows if row.applicable]


def benchmark_start_disabled_reason(selected: Sequence[str]) -> str | None:
    if not selected:
        return "Pick at least one mode to benchmark."
    return None


def benchmark_job_state(job: Mapping[str, Any] | None) -> str:
    return str((job or {}).get("state") or "unknown")


#: Job states after which nothing is running and nothing can be cancelled.
BENCHMARK_TERMINAL_STATES: Final = frozenset(
    {"succeeded", "completed", "done", "failed", "cancelled", "canceled"}
)


def benchmark_job_finished(job: Mapping[str, Any] | None) -> bool:
    """Whether a job has reached a terminal state.

    Drives two things in the dialog: the poll timer stops, and the Cancel
    button disappears -- a Cancel that outlives the job it would cancel reads
    as a control that does nothing.
    """
    return benchmark_job_state(job) in BENCHMARK_TERMINAL_STATES


def benchmark_progress_fraction(job: Mapping[str, Any] | None) -> float:
    """0..1 for the progress bar, tolerating a job that has not started yet."""
    progress = (job or {}).get("progress")
    if not isinstance(progress, Mapping):
        return 0.0
    fraction = progress.get("fraction")
    if isinstance(fraction, int | float) and not isinstance(fraction, bool):
        return max(0.0, min(1.0, float(fraction)))
    return progress_fraction(progress.get("completed"), progress.get("total"))


def benchmark_progress_text(job: Mapping[str, Any] | None) -> str:
    """What is running, not just how far along it is.

    A benchmark that reloads a 30B model onto four different device sets takes
    minutes; a bare percentage gives no way to tell a slow load from a wedged
    one, so the mode and phase are named as they go past.
    """
    state = benchmark_job_state(job)
    progress = (job or {}).get("progress")
    if not isinstance(progress, Mapping):
        return state
    parts: list[str] = []
    completed, total = _maybe_int(progress.get("completed")), _maybe_int(progress.get("total"))
    if completed is not None and total:
        parts.append(f"{completed} of {total}")
    mode = progress.get("mode")
    if mode:
        parts.append(str(mode))
    phase = progress.get("phase")
    if phase:
        parts.append(str(phase))
    parts.append(format_percent(benchmark_progress_fraction(job)))
    return " · ".join(parts)


@dataclass(frozen=True)
class BenchmarkResultRow:
    """One mode's measured numbers, ready for the comparison table."""

    mode: str
    label: str
    devices: tuple[int, ...] = ()
    applicable: bool = True
    skipped_reason: str | None = None
    load_time_s: float | None = None
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    prompt_tps: float | None = None
    generated_tokens: int | None = None
    generation_tps: float | None = None
    error: str | None = None

    @property
    def ran(self) -> bool:
        """Whether this row carries numbers, rather than a reason it has none."""
        return self.error is None and self.applicable and self.generation_tps is not None

    @property
    def status_text(self) -> str:
        """Why there are no numbers, or ``""`` when there are."""
        if self.error:
            return f"failed: {self.error}"
        if not self.applicable:
            return f"skipped: {self.skipped_reason or 'not applicable on this hardware'}"
        if self.generation_tps is None:
            return "no result"
        return ""


def benchmark_result_rows(report: Mapping[str, Any] | None) -> list[BenchmarkResultRow]:
    """Per-mode rows from a finished report, in the order the backend returned."""
    if not report:
        return []
    results = report.get("results")
    if not isinstance(results, Sequence) or isinstance(results, str):
        return []
    rows: list[BenchmarkResultRow] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        mode = str(item.get("mode") or "")
        if not mode:
            continue
        reason = item.get("skipped_reason")
        error = item.get("error")
        rows.append(
            BenchmarkResultRow(
                mode=mode,
                label=str(item.get("label") or mode),
                devices=_maybe_devices(item.get("devices")),
                applicable=bool(item.get("applicable", True)),
                skipped_reason=str(reason) if reason else None,
                load_time_s=_maybe_float(item.get("load_time_s")),
                ttft_s=_maybe_float(item.get("ttft_s")),
                prompt_tokens=_maybe_int(item.get("prompt_tokens")),
                prompt_tps=_maybe_float(item.get("prompt_tps")),
                generated_tokens=_maybe_int(item.get("generated_tokens")),
                generation_tps=_maybe_float(item.get("generation_tps")),
                error=str(error) if error else None,
            )
        )
    return rows


def fastest_modes(rows: Sequence[BenchmarkResultRow]) -> tuple[str | None, str | None]:
    """``(fastest generation mode, fastest prompt mode)``.

    Two winners, not one, because they are routinely different: adding a second
    GPU usually helps prompt processing and hurts generation, since a layer
    split adds per-token cross-device traffic. A single "fastest" would hide
    exactly the trade-off this table exists to show.
    """
    ran = [row for row in rows if row.ran]

    def best(attribute: str) -> str | None:
        scored = [
            (value, row.mode)
            for row, value in ((row, getattr(row, attribute)) for row in ran)
            if isinstance(value, int | float) and not isinstance(value, bool)
        ]
        if not scored:
            return None
        # Ties keep the backend's own order rather than sorting by mode name.
        return max(scored, key=lambda pair: float(pair[0]))[1]

    return (best("generation_tps"), best("prompt_tps"))


def report_best_modes(
    report: Mapping[str, Any] | None, rows: Sequence[BenchmarkResultRow]
) -> tuple[str | None, str | None]:
    """The winners to highlight: the report's own, or derived from the rows.

    The backend computes and persists ``best_generation_mode`` /
    ``best_prompt_mode``; using them keeps the highlighted row identical to what
    was stored, instead of the panel quietly disagreeing with its own history
    over a tie-break. :func:`fastest_modes` is the fallback for a report that
    predates those fields.
    """
    payload = report or {}
    generation = payload.get("best_generation_mode")
    prompt = payload.get("best_prompt_mode")
    if generation or prompt:
        return (
            str(generation) if generation else None,
            str(prompt) if prompt else None,
        )
    return fastest_modes(rows)


def benchmark_report_notes(report: Mapping[str, Any] | None) -> list[str]:
    notes = (report or {}).get("notes")
    if not isinstance(notes, Sequence) or isinstance(notes, str):
        return []
    return [str(note) for note in notes]


def benchmark_speedup_text(rows: Sequence[BenchmarkResultRow]) -> str:
    """One sentence on what the best generation mode actually buys, or ``""``."""
    ran = [row for row in rows if row.ran and row.generation_tps]
    if len(ran) < 2:
        return ""
    ordered = sorted(ran, key=lambda r: float(r.generation_tps or 0.0), reverse=True)
    best, worst = ordered[0], ordered[-1]
    fast, slow = float(best.generation_tps or 0.0), float(worst.generation_tps or 0.0)
    if slow <= 0:
        return ""
    ratio = fast / slow
    if ratio < 1.05:
        return (
            f"Every mode generates within 5% of the others ({slow:.1f}–{fast:.1f} tok/s); "
            "placement is not what limits this model."
        )
    return (
        f"{best.label} generates {ratio:.2f}x faster than {worst.label} "
        f"({fast:.1f} vs {slow:.1f} tok/s)."
    )


def benchmark_history_label(entry: Mapping[str, Any] | None, *, now: float | None = None) -> str:
    """Heading for one stored run in the history list."""
    if not entry:
        return UNKNOWN
    when = format_when(_maybe_float(entry.get("ts")), now=now)
    report = entry.get("report")
    count = len(benchmark_result_rows(report if isinstance(report, Mapping) else None))
    return f"{when} · {count} mode(s)"


# ---------------------------------------------------------------------------
# Backend capabilities: "what kinds of model can I actually run?"
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def sizing_headline(report: Mapping[str, Any] | None) -> str:
    """The answer to the user's actual question, in one line.

    Leads with how many of *their* models fit where, because "142 architectures
    supported" does not tell anyone whether the 30 GB file they just downloaded
    will load.
    """
    sizing = _mapping(_mapping(_mapping(report).get("library")).get("sizing"))
    if not sizing:
        return "Model sizing is unavailable (no GPU readings)."
    one = _maybe_int(sizing.get("fits_one_gpu")) or 0
    split = _maybe_int(sizing.get("needs_multiple_gpus")) or 0
    too_big = _maybe_int(sizing.get("too_big")) or 0
    total = one + split + too_big
    return f"{one} of your {total} models fit on one GPU · {split} need a split · {too_big} too big"


def sizing_note(report: Mapping[str, Any] | None) -> str:
    """The backend's own caveat about that headline, verbatim."""
    sizing = _mapping(_mapping(_mapping(report).get("library")).get("sizing"))
    return str(sizing.get("note") or "")


def too_big_models(report: Mapping[str, Any] | None) -> list[str]:
    sizing = _mapping(_mapping(_mapping(report).get("library")).get("sizing"))
    models = sizing.get("too_big_models")
    if not isinstance(models, Sequence) or isinstance(models, str):
        return []
    return [str(m) for m in models]


def engine_summary_lines(report: Mapping[str, Any] | None) -> list[str]:
    """Engine identity and reach: tag, variant, build string, how much it knows."""
    engine = _mapping(_mapping(report).get("engine"))
    if not engine:
        return ["No engine installed — install one from the Engine section above."]
    tag = str(engine.get("tag") or UNKNOWN)
    variant = str(engine.get("variant") or UNKNOWN)
    lines = [f"engine {tag} ({variant})"]
    version = str(engine.get("version_string") or "")
    if version:
        lines.append(version)
    architectures = _maybe_int(engine.get("architecture_count")) or len(
        _sequence(engine.get("architectures"))
    )
    quants = len(_sequence(engine.get("quant_types")))
    lines.append(f"supports {architectures} architectures · {quants} quantizations")
    lines.append(
        "smoke-tested on this machine"
        if engine.get("smoke_tested")
        else "not smoke-tested yet — run the smoke test in the Engine section"
    )
    return lines


def _sequence(value: Any) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return list(value)


def capability_source_caveat(report: Mapping[str, Any] | None) -> str | None:
    """Warning when the supported-architecture list is not from this engine.

    A bundled snapshot can silently disagree with the engine that is actually
    installed, which would turn this panel from an answer into a guess. So the
    provenance is shown rather than smoothed over.
    """
    engine = _mapping(_mapping(report).get("engine"))
    if not engine:
        return None
    source = str(engine.get("capability_source") or "")
    detail = str(engine.get("capability_source_detail") or "")
    if source == "snapshot":
        return (
            "This list comes from a bundled snapshot, not from the installed engine's own "
            "source, so it may be out of date for the build actually running. " + (detail or "")
        ).strip()
    return None


def hardware_summary_lines(report: Mapping[str, Any] | None) -> list[str]:
    hardware = _mapping(_mapping(report).get("hardware"))
    if not hardware:
        return ["No GPUs detected — this server is GPU-only, so nothing can load."]
    gpus = _sequence(hardware.get("gpus"))
    lines: list[str] = []
    for gpu in gpus:
        item = _mapping(gpu)
        lines.append(
            f"GPU{item.get('index')} · {item.get('name')} · "
            f"{format_gib(_maybe_float(item.get('total_bytes')))} · "
            f"sm_{item.get('sm_arch') or UNKNOWN}"
        )
    lines.append(
        f"{format_gib(_maybe_float(hardware.get('total_vram_bytes')))} total VRAM · "
        f"{format_gib(_maybe_float(hardware.get('usable_total_bytes')))} usable after headroom · "
        f"largest single GPU {format_gib(_maybe_float(hardware.get('usable_largest_bytes')))}"
    )
    driver = hardware.get("driver_version")
    if driver:
        lines.append(f"driver {driver} · CUDA {hardware.get('cuda_driver_version') or UNKNOWN}")
    return lines


@dataclass(frozen=True)
class Chip:
    """A ``label ×count`` pill, used for architecture/quant breakdowns."""

    label: str
    count: int
    tooltip: str = ""

    @property
    def text(self) -> str:
        return f"{self.label} ×{self.count}"


def _chips(counts: Any, *, tooltip: str = "") -> list[Chip]:
    mapping = _mapping(counts)
    rows = [
        Chip(str(name), _maybe_int(count) or 0, tooltip)
        for name, count in mapping.items()
        if (_maybe_int(count) or 0) > 0
    ]
    # Commonest first, then alphabetically: the long tail of one-off
    # architectures is not what anyone is looking for.
    return sorted(rows, key=lambda chip: (-chip.count, chip.label.lower()))


def architecture_chips(report: Mapping[str, Any] | None) -> list[Chip]:
    library = _mapping(_mapping(report).get("library"))
    return _chips(library.get("by_architecture"), tooltip="models of this architecture")


def quant_chips(report: Mapping[str, Any] | None) -> list[Chip]:
    library = _mapping(_mapping(report).get("library"))
    return _chips(library.get("by_quant"), tooltip="models at this quantisation")


def library_capability_chips(report: Mapping[str, Any] | None) -> list[Chip]:
    """Capability counts, labelled with the same words as the table's icons."""
    library = _mapping(_mapping(report).get("library"))
    counts = _mapping(library.get("capabilities"))
    by_key = {item.key: item for item in CAPABILITY_ICONS}
    chips: list[Chip] = []
    for key, item in by_key.items():
        count = _maybe_int(counts.get(key)) or 0
        if count:
            chips.append(Chip(item.label, count, item.tooltip))
    return chips


def unsupported_models(report: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``(model_id, architecture)`` for models this engine cannot load.

    The single most consequential thing on the panel: these are not slow or
    suboptimal, they will not start at all.
    """
    library = _mapping(_mapping(report).get("library"))
    out: list[tuple[str, str]] = []
    for item in _sequence(library.get("unsupported_by_engine")):
        entry = _mapping(item)
        model_id = str(entry.get("model_id") or "")
        if model_id:
            out.append((model_id, str(entry.get("architecture") or UNKNOWN)))
    return sorted(out)


def unsupported_warning(report: Mapping[str, Any] | None) -> str | None:
    rows = unsupported_models(report)
    if not rows:
        return None
    return (
        f"{len(rows)} model(s) use an architecture this engine does not implement and "
        "cannot be loaded. Updating the engine is the usual fix."
    )


def feature_rows(report: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """``(feature, explanation)`` pairs, in the backend's order."""
    features = _mapping(_mapping(report).get("features"))
    return [(str(name), str(text)) for name, text in features.items()]


def quant_hardware_notes(report: Mapping[str, Any] | None) -> list[tuple[str, str]]:
    """Per-quant hardware notes (NVFP4/MXFP4), which are informative, not warnings.

    DECISIONS D9: FP4 quants were *measured* loading and generating correctly on
    the Ampere cards. Presenting these as "unsupported" would make half the rig
    look unusable, which is both wrong and discouraging.
    """
    notes = _mapping(_mapping(report).get("quant_hardware_notes"))
    return [(str(name), str(text)) for name, text in sorted(notes.items())]


def engine_update_line(update: Mapping[str, Any] | None) -> str:
    """One line for the update row, covering all four states it can be in.

    The variant is named when the check knows it, because "b10488 (cuda-13.3)"
    and "b10488 (source)" are the difference between a two-minute download and a
    half-hour local CUDA compile, and the button looks identical either way.
    """
    payload = _mapping(update)
    if not payload or not payload.get("checked"):
        return "Update check not run yet."
    error = payload.get("error")
    if error:
        return f"Could not check for a newer engine: {error}"
    current = str(payload.get("current") or UNKNOWN)
    latest = payload.get("latest")
    if not latest:
        return f"Engine {current}. No installable release was found on GitHub."
    if payload.get("update_available"):
        variant = payload.get("latest_variant")
        suffix = f" ({variant})" if variant else ""
        return f"Engine {current} — {latest}{suffix} is available."
    return f"Engine {current} is the latest release."


def engine_update_available(update: Mapping[str, Any] | None) -> bool:
    return bool(_mapping(update).get("update_available"))


#: Shown next to the update button. Installing an engine does not touch running
#: children -- they keep the build they were launched with until reloaded --
#: and saying so is what stops "I updated and nothing changed" reports.
ENGINE_UPDATE_NOTE: Final = (
    "Installing a new engine does not disturb running models: each llama-server child "
    "keeps the build it was launched with. Use 'Restart engines' on the Dashboard to "
    "move the loaded models onto the new build."
)


def filter_architectures(names: Sequence[str], needle: str | None) -> list[str]:
    """Substring filter for the (long) supported-architecture list."""
    ordered = sorted({str(name) for name in names})
    if not needle or not needle.strip():
        return ordered
    lowered = needle.strip().lower()
    return [name for name in ordered if lowered in name.lower()]


def supported_architectures(report: Mapping[str, Any] | None) -> list[str]:
    engine = _mapping(_mapping(report).get("engine"))
    return [str(name) for name in _sequence(engine.get("architectures"))]


def supported_quant_types(report: Mapping[str, Any] | None) -> list[str]:
    engine = _mapping(_mapping(report).get("engine"))
    return [str(name) for name in _sequence(engine.get("quant_types"))]


# ---------------------------------------------------------------------------
# Setup tab: the first-run checklist
#
# Everything below is a pure function of primitives -- counts, flags, paths --
# so the tab can be a thin renderer and the *rules* (what is required, what is
# merely nice to have, what wording each state gets) are testable without a
# browser, a GPU or a config file. See DECISIONS.md D26.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SetupCheck:
    """One line of the Setup tab's first-run checklist.

    ``required`` is the load-bearing field. A required check that fails means
    this server cannot serve inference at all; an optional one means something
    the user may well not want (a HuggingFace token, autostart) is absent. They
    are rendered differently and only the required ones gate "ready", because a
    checklist that shouts about an unset optional key teaches people to ignore
    the ones that matter.
    """

    key: str
    name: str
    ok: bool
    detail: str
    required: bool = True
    action: str = ""
    action_label: str = ""
    help: str = ""

    @property
    def icon(self) -> str:
        if self.ok:
            return "check_circle"
        return "error" if self.required else "info"

    @property
    def colour(self) -> str:
        if self.ok:
            return "positive"
        return "warning" if self.required else "grey"

    @property
    def status_text(self) -> str:
        if self.ok:
            return "ok"
        return "required" if self.required else "optional"


def _host_is_loopback(host: str | None) -> bool:
    """``127.0.0.1``, ``::1``, ``localhost`` -- anything only this machine can reach."""
    import ipaddress

    text = str(host or "").strip().strip("[]").lower()
    if text in {"localhost", ""}:
        return text == "localhost"
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def first_run_checks(
    *,
    data_dir: Any,
    data_dir_writable: bool,
    models_dir: Any,
    models_dir_exists: bool,
    gguf_count: int,
    indexed_count: int,
    gpu_count: int,
    engine_tag: str | None,
    driver_version: str | None = None,
    cuda_driver: tuple[int, int] | None = None,
    excluded_devices: Sequence[int] = (),
    engine_smoke_tested: bool = False,
    pinned_tag: str = "",
    api_port: int = 0,
    api_reachable: bool = True,
    api_port_detail: str = "",
    mcp_pin_set: bool = False,
    mcp_pin_required: bool = True,
    hf_token_set: bool = False,
    autostart_enabled: bool = False,
    autostart_mechanism: str = "",
    bind_host: str = "0.0.0.0",
    api_key_set: bool = False,
    gui_host: str | None = None,
    watchdog_host: str | None = None,
    boot_phase: str | None = None,
) -> list[SetupCheck]:
    """Everything a fresh checkout has to get right, in the order it matters.

    The required set is exactly "can this box load a model and answer a
    request": a writable data dir, a model library that has been indexed, GPUs,
    an engine, and a listening port. Everything else -- a HuggingFace token,
    autostart, and the MCP PIN when pairing is not enforced -- is optional and
    says so, because a checklist that shouts about an unset optional key is a
    checklist people learn to ignore.
    """
    checks: list[SetupCheck] = []

    checks.append(
        SetupCheck(
            key="data_dir",
            name="Data directory",
            ok=bool(data_dir_writable),
            detail=(
                f"{data_dir} is writable"
                if data_dir_writable
                else f"{data_dir} is not writable — config, registry and logs all live here"
            ),
            action="" if data_dir_writable else "open-data-dir",
            action_label="Open data dir",
            help="config.yaml, registry.sqlite3, engines/ and logs/ all live in one place.",
        )
    )

    library_ok = bool(models_dir) and models_dir_exists and gguf_count > 0
    if not models_dir:
        library_detail = "not set — point models.dir at your GGUF library"
    elif not models_dir_exists:
        library_detail = f"{models_dir} does not exist"
    elif gguf_count <= 0:
        library_detail = f"{models_dir} contains no .gguf files"
    else:
        library_detail = f"{gguf_count} GGUF file(s) under {models_dir}"
    checks.append(
        SetupCheck(
            key="models_dir",
            name="Model library",
            ok=library_ok,
            detail=library_detail,
            action="" if library_ok else "detect-library",
            action_label="Detect LM Studio library",
            help="models.dir is scanned in place; nothing is ever copied or moved.",
        )
    )

    checks.append(
        SetupCheck(
            key="models_indexed",
            name="Models indexed",
            ok=indexed_count > 0,
            detail=(
                f"{indexed_count} model(s) in the registry"
                if indexed_count
                else "nothing indexed yet — run a scan"
            ),
            action="" if indexed_count else "scan",
            action_label="Rescan now",
            help="The registry is what /v1/models, the catalog and the planner read.",
        )
    )

    if gpu_count:
        gpu_detail = f"{gpu_count} GPU(s) visible to NVML"
        if driver_version:
            gpu_detail += f" · driver {driver_version}"
        if cuda_driver:
            gpu_detail += f" · driver CUDA {cuda_driver[0]}.{cuda_driver[1]}"
        excluded = sorted({int(index) for index in excluded_devices})
        if excluded:
            gpu_detail += " · excluded: CUDA" + ",".join(str(index) for index in excluded)
    else:
        gpu_detail = "no GPUs detected — this server is GPU-only, so nothing can load"
    checks.append(
        SetupCheck(
            key="gpus",
            name="GPUs",
            ok=gpu_count > 0,
            detail=gpu_detail,
            action="" if gpu_count else "reprobe",
            action_label="Re-probe",
            help="Devices are numbered by CUDA ordinal, which is what every plan refers to.",
        )
    )

    engine_ok = bool(engine_tag)
    booting_engine = bool(boot_phase) and "engine" in str(boot_phase)
    if not engine_tag and booting_engine:
        # The boot is installing it right now (D33): show that instead of a
        # second Install button that would race the first.
        engine_detail = f"{boot_phase} — the first run does this once; re-check in a minute"
    elif not engine_tag:
        engine_detail = f"not installed — install {pinned_tag or 'the pinned build'}"
    elif engine_smoke_tested:
        engine_detail = f"active: {engine_tag} (smoke-tested)"
    else:
        engine_detail = f"active: {engine_tag} — never smoke-tested on this box"
    checks.append(
        SetupCheck(
            key="engine",
            name="llama.cpp engine",
            ok=engine_ok,
            detail=engine_detail,
            action="" if (engine_ok or booting_engine) else "install-engine",
            action_label=f"Install engine {pinned_tag}" if pinned_tag else "Install engine",
            help="Engines are versioned artifacts under engines/<tag>/, not whatever is on PATH.",
        )
    )

    checks.append(
        SetupCheck(
            key="api_port",
            name="Gateway port",
            ok=bool(api_reachable),
            detail=api_port_detail or f"port {api_port}",
            help="LM Studio uses the same port by default; only one of them can hold it.",
        )
    )

    # All three listeners, not just the gateway: the control panel and the
    # watchdog's recovery surface share the credential and are just as much
    # "the server" to anyone on the LAN. server.host 127.0.0.1 with gui.host
    # 0.0.0.0 used to read green while the panel was wide open.
    binds = {"server.host": bind_host}
    if gui_host is not None:
        binds["gui.host"] = gui_host
    if watchdog_host is not None:
        binds["watchdog.host"] = watchdog_host
    open_binds = [f"{name} {host}" for name, host in binds.items() if not _host_is_loopback(host)]
    exposed = bool(open_binds)
    network_ok = (not exposed) or bool(api_key_set)
    checks.append(
        SetupCheck(
            key="network",
            name="Network exposure",
            ok=network_ok,
            # Required-when-it-matters: a loopback bind or a key makes it green;
            # 0.0.0.0 with no key is a real gap, not a preference (WP17 F4).
            required=exposed,
            detail=(
                f"bound to {bind_host} (this machine only)"
                if not exposed
                else (
                    f"reachable from the network ({', '.join(open_binds)}), protected by "
                    "server.api_key"
                    if api_key_set
                    else (
                        f"reachable from the whole network ({', '.join(open_binds)}) with NO "
                        "API key: anyone on the LAN can load, unload and delete models, "
                        "change settings and restart the server (the control panel and the "
                        "watchdog included). The MCP PIN guards only /mcp."
                    )
                )
            ),
            action="" if network_ok else "set-api-key",
            action_label="Set API key",
            help=(
                "server.host 127.0.0.1 keeps it private to this box; 0.0.0.0 serves the LAN "
                "and Tailscale and then needs server.api_key (OpenClaw sends it as a Bearer "
                "token)."
            ),
        )
    )

    checks.append(
        SetupCheck(
            key="mcp_pin",
            name="MCP pairing PIN",
            ok=bool(mcp_pin_set) or not mcp_pin_required,
            required=bool(mcp_pin_required),
            detail=(
                "set — agents pair with it"
                if mcp_pin_set
                else (
                    "not set, and mcp.pin_required is on: the MCP endpoint cannot be paired"
                    if mcp_pin_required
                    else "not required (mcp.pin_required is off); the API key is the credential"
                )
            ),
            action="" if mcp_pin_set else "generate-pin",
            action_label="Generate PIN",
            help="A short pairing code for the MCP path only — never the inference credential.",
        )
    )

    checks.append(
        SetupCheck(
            key="hf_token",
            name="HuggingFace token",
            ok=bool(hf_token_set),
            required=False,
            detail=(
                "set — gated repositories are reachable"
                if hf_token_set
                else "not set — public repositories still download fine"
            ),
            help="Only needed for gated or private repositories.",
        )
    )

    checks.append(
        SetupCheck(
            key="autostart",
            name="Start at login",
            ok=bool(autostart_enabled),
            required=False,
            detail=(f"enabled via {autostart_mechanism}" if autostart_enabled else "not enabled"),
            action="" if autostart_enabled else "enable-autostart",
            action_label="Enable autostart",
            help="A per-user Startup entry on Windows, a systemd --user unit on Linux.",
        )
    )

    return checks


#: Required checks that are about a SAFE install rather than a WORKING one.
#: They still gate "ready" (a LAN-open admin surface is not a finished setup),
#: but the headline must not claim they stop models from loading.
SAFETY_CHECK_KEYS: frozenset[str] = frozenset({"network"})


def checklist_is_ready(checks: Sequence[SetupCheck]) -> bool:
    """Whether every **required** check passes. Optional ones never gate."""
    return all(check.ok for check in checks if check.required)


def checklist_headline(checks: Sequence[SetupCheck]) -> str:
    """The one line at the top of the Setup tab."""
    if not checks:
        return "Nothing to check."
    outstanding = [check for check in checks if check.required and not check.ok]
    optional = [check for check in checks if not check.required and not check.ok]
    # Two kinds of "required": what stops a model from loading at all, and
    # what must be fixed for a *safe* install (network exposure). Saying
    # "before this server can load a model" about the second one is untrue and
    # teaches the reader that the headline exaggerates.
    blocking = [check for check in outstanding if check.key not in SAFETY_CHECK_KEYS]
    safety = [check for check in outstanding if check.key in SAFETY_CHECK_KEYS]
    if blocking:
        names = ", ".join(check.name for check in blocking)
        extra = f" Also fix: {', '.join(c.name for c in safety)}." if safety else ""
        head = f"{len(blocking)} thing(s) to fix before this server can load a model"
        return f"{head}: {names}.{extra}"
    if safety:
        names = ", ".join(check.name for check in safety)
        return f"Ready to serve, but fix before exposing it: {names}."
    tail = f" {len(optional)} optional item(s) left." if optional else ""
    return f"Ready to serve — every required check passes.{tail}"


def checklist_actions(checks: Sequence[SetupCheck]) -> list[tuple[str, str]]:
    """``(action, label)`` for every unmet check that offers one."""
    return [(c.action, c.action_label) for c in checks if not c.ok and c.action]


# ---------------------------------------------------------------------------
# Setup tab: model library detection
# ---------------------------------------------------------------------------


def lmstudio_candidate_lines(candidates: Sequence[Mapping[str, Any]]) -> list[str]:
    """One line per probed LM Studio location, in the order they were probed.

    Shows the ones that did *not* match too. "Detect found nothing" is an
    unhelpful answer when the reason is that the ``downloadsFolder`` recorded
    in ``~/.lmstudio/settings.json`` points at a drive that is no longer
    mounted.
    """
    lines: list[str] = []
    for entry in candidates:
        path = str(entry.get("path") or "")
        if not path:
            continue
        if not entry.get("exists"):
            lines.append(f"{path} — does not exist")
            continue
        count = int(entry.get("gguf_count") or 0)
        lines.append(f"{path} — {count} GGUF file(s)" if count else f"{path} — no GGUF files")
    return lines


def lmstudio_detection_note(detected: Any, current: Any) -> str:
    """What the detect button found, relative to what is configured now."""
    if not detected:
        return (
            "No LM Studio library found. Any directory of .gguf files works — "
            "type or paste its path above."
        )
    if current and str(detected) == str(current):
        return f"Detected {detected}, which is already the configured library."
    return f"Detected {detected}. Save to point models.dir at it."


def models_dir_status_line(
    models_dir: Any,
    *,
    exists: bool,
    gguf_count: int,
    disk: Mapping[str, Any] | None = None,
) -> str:
    """Validation line under ``models.dir``: exists, how many GGUFs, free space."""
    if not models_dir:
        return "models.dir is not set — nothing can be indexed until it is."
    if not exists:
        return f"{models_dir} does not exist yet. It is created on the first download."
    body = f"{models_dir} — {gguf_count} GGUF file(s)"
    line = disk_line(disk)
    return f"{body} · {line}" if line else body


# ---------------------------------------------------------------------------
# Setup tab: GPUs
# ---------------------------------------------------------------------------


#: Why the numbers in every plan are CUDA ordinals, and why they can move.
PLANNER_PREFERENCE_NOTE = (
    "quality (default): pick the best KV cache that still reaches the context floor, then the "
    "largest context at that quality. A 4-bit K cache is never chosen automatically, and a "
    "doubled window is not traded for a quantized one — Gemma-4 measures a KL divergence of "
    "0.108 at q8_0 against f16. throughput: the older rule — the largest window at or above the "
    "floor, preferring one that also serves two conversations."
)

DEVICE_RECOGNITION_NOTE: Final = (
    "GPUs are enumerated by NVML and referred to everywhere by their CUDA ordinal "
    "(CUDA0, CUDA1, …) — that is what excluded_devices, reserved_mb and every "
    "per-model device override mean. Ordinals are assigned by the driver and are not "
    "stable across a hardware change: adding, removing or re-slotting a card can "
    "renumber the rest, and CUDA_VISIBLE_DEVICES in the environment renumbers them "
    "for this process only. After any of those, re-probe and re-check the exclusions "
    "below — they are indices, not names."
)


@dataclass(frozen=True)
class GpuSetupRow:
    """One GPU as the Setup tab's table shows it."""

    index: int
    name: str
    total_bytes: int
    free_bytes: int
    compute_capability: str
    excluded: bool = False
    reserved_mb: int = 0
    holders: str = ""

    @property
    def vram_text(self) -> str:
        return f"{format_gib(self.free_bytes)} free of {format_gib(self.total_bytes)}"

    def summary(self) -> str:
        parts = [f"CUDA{self.index}", self.name, self.vram_text, f"cc {self.compute_capability}"]
        if self.excluded:
            parts.append("EXCLUDED")
        if self.reserved_mb:
            parts.append(f"{self.reserved_mb} MiB reserved")
        if self.holders:
            parts.append(self.holders)
        return "  ·  ".join(parts)


def gpu_setup_rows(
    gpus: Sequence[GpuInfo],
    *,
    excluded_devices: Sequence[int] = (),
    reserved_mb: Mapping[int, int] | None = None,
    holders: Sequence[Mapping[str, Any]] = (),
) -> list[GpuSetupRow]:
    """The live GPU table, joined with the planner's per-device policy."""
    excluded = {int(index) for index in excluded_devices}
    reserved = {int(k): int(v) for k, v in (reserved_mb or {}).items()}
    per_gpu: dict[int, list[Mapping[str, Any]]] = {}
    for holder in holders:
        for index in holder.get("gpu_indices") or []:
            per_gpu.setdefault(int(index), []).append(holder)
    return [
        GpuSetupRow(
            index=gpu.index,
            name=gpu.name,
            total_bytes=gpu.total_bytes,
            free_bytes=gpu.free_bytes,
            compute_capability=gpu.cc_str,
            excluded=gpu.index in excluded,
            reserved_mb=reserved.get(gpu.index, 0),
            holders=_holder_summary(per_gpu.get(gpu.index, ())),
        )
        for gpu in gpus
    ]


def _holder_summary(holders: Sequence[Mapping[str, Any]]) -> str:
    """ "2 process(es) holding 5.10 GiB", or nothing at all.

    A holder row may carry ``device_bytes`` -- what the process holds on THIS
    card (D39's per-adapter measurement) -- and when it does, that is the
    figure summed. ``used_bytes`` is a per-process total across every card, so
    summing it per GPU double-counted a model split over two of them (WP21's
    follow-up); it remains the fallback when no per-device figure exists.
    """
    if not holders:
        return ""
    total = 0
    for h in holders:
        device = h.get("device_bytes")
        total += int(device) if device is not None else int(h.get("used_bytes") or 0)
    if total <= 0:
        return f"{len(holders)} process(es) holding VRAM (size unavailable)"
    return f"{len(holders)} process(es) holding {format_gib(total)}"


def parse_reserved_mb(raw: Any) -> int:
    """A per-GPU reservation from a number widget: 0 rather than ``None``."""
    if raw in (None, ""):
        return 0
    try:
        return max(0, int(float(raw)))
    except (TypeError, ValueError):
        return 0


def reserved_mb_map(entries: Mapping[int, Any]) -> dict[int, int]:
    """Only the non-zero reservations; zero means "none", not "0 MiB reserved"."""
    out: dict[int, int] = {}
    for index, raw in entries.items():
        amount = parse_reserved_mb(raw)
        if amount > 0:
            out[int(index)] = amount
    return out


def excluded_devices_list(flags: Mapping[int, Any]) -> list[int]:
    """Sorted, de-duplicated CUDA indices from a map of per-GPU toggles."""
    return sorted({int(index) for index, flag in flags.items() if bool(flag)})


def device_policy_note(excluded: Sequence[int], reserved: Mapping[int, int] | None) -> str:
    """Plain-language summary of the two multi-tenant knobs (DECISIONS D19)."""
    parts: list[str] = []
    if excluded:
        parts.append(
            "never placed on: " + ", ".join(f"CUDA{index}" for index in sorted(set(excluded)))
        )
    for index, amount in sorted((reserved or {}).items()):
        if amount:
            parts.append(f"{amount} MiB held back on CUDA{index}")
    if not parts:
        return (
            "No device policy set: the planner may use every GPU, minus the global headroom. "
            "Exclude a card to leave it entirely to a neighbour (ComfyUI, a training job), or "
            "reserve MiB on it to leave the neighbour room while still using the rest."
        )
    return "; ".join(parts) + ". A per-model device override still wins, with a warning."


# ---------------------------------------------------------------------------
# Setup tab: engine
# ---------------------------------------------------------------------------


def cuda_variant_note(driver_cuda: tuple[int, int] | None, variant: str = "auto") -> str:
    """Why a particular CUDA build, backed by the driver's own number.

    The correctness rule runs one way only: a build compiled against CUDA X.Y
    runs on a driver advertising **>= X.Y** and never below it, so the highest
    eligible build wins. On Blackwell that has to be a 13.x build -- the 12.4
    archives carry no sm_120 kernels at all (DECISIONS D2/D3).
    """
    driver = f"{driver_cuda[0]}.{driver_cuda[1]}" if driver_cuda else UNKNOWN
    chosen = (variant or "auto").strip() or "auto"
    if chosen.lower() == "auto":
        body = f"auto: the highest CUDA build this driver can run (driver CUDA {driver})."
    else:
        body = f"pinned to CUDA {chosen}; this driver advertises CUDA {driver}."
    return (
        f"{body} A build made against CUDA X.Y needs a driver advertising X.Y or newer — "
        "CUDA compatibility runs forward, not back. Blackwell (sm_120) cards need the 13.x "
        "builds; the 12.4 archives carry no sm_120 kernels at all."
    )


def engine_install_rows(installed: Sequence[Mapping[str, Any]]) -> list[str]:
    """``★ b10425 (cuda-13.3) · smoke tested`` lines for the installed engines."""
    rows: list[str] = []
    for info in installed:
        marker = "★" if info.get("active") else "·"
        tested = "smoke tested" if info.get("smoke_tested") else "not smoke tested"
        rows.append(f"{marker} {info.get('tag')} ({info.get('variant')}) · {tested}")
    return rows


# ---------------------------------------------------------------------------
# Setup tab: network and access
# ---------------------------------------------------------------------------


def reachable_lines(endpoints: Sequence[Mapping[str, Any]]) -> list[str]:
    """``Tailscale  http://…:1234`` lines from ``netinfo.reachable_urls``."""
    return [f"{entry.get('label') or entry.get('kind')}  {entry.get('url')}" for entry in endpoints]


def bind_note(host: str) -> str:
    """What a bind address actually exposes, said plainly."""
    value = (host or "").strip()
    if value in {"0.0.0.0", "::", ""}:
        return (
            "0.0.0.0 listens on every interface — LAN and tailnet included. Set an API key if "
            "this machine is on a network you do not control."
        )
    if value in {"127.0.0.1", "localhost", "::1"}:
        return (
            "127.0.0.1 is this machine only: nothing on the LAN or the tailnet can reach it, "
            "an agent on another box included."
        )
    return f"Bound to {value} only; the server answers on no other address."


def port_conflict_note(port: int, *, lmstudio_default: int = 1234) -> str:
    """The one port collision every new install hits."""
    if int(port) == int(lmstudio_default):
        return (
            f"Port {port} is LM Studio's default too. They can share a model library on disk "
            "but not the port — quit LM Studio, or change this."
        )
    return (
        f"Port {port}. LM Studio-compatible clients default to {lmstudio_default}, so anything "
        "pointed at this box has to be told the new port."
    )


def secret_state_text(value: str | None, *, unset_note: str = "not set") -> str:
    """Masked display for a secret, or why its absence is fine."""
    if not value:
        return unset_note
    return masked_secret(value) or "***"


#: What a masked secret looks like inside a rendered snippet.
SNIPPET_MASK: Final = "••••••••"


def mask_secrets(text: str, secrets: Sequence[str | None], *, mask: str = SNIPPET_MASK) -> str:
    """Blank every secret out of a snippet before it is put on screen.

    The OpenClaw snippets are built by the management route and legitimately
    contain the API key and the pairing PIN -- that is what makes them
    paste-ready. Rendering them into a page that may be on a shared screen is a
    different thing from copying them to a clipboard, so the display is masked
    and the copy button still copies the real text.

    Longest first, so a secret that contains another one cannot be half-masked.
    """
    out = text
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        out = out.replace(secret, mask)
    return out


def redacted_config(config: Any) -> dict[str, Any]:
    """``config.to_yaml_dict()`` with every :data:`SECRET_KEYS` value masked.

    The single source of the values every config form is drawn from, so no
    surface can accidentally render a real credential into a page -- and so the
    "did this field really change" guard always compares against the same
    placeholder shape.
    """
    data: dict[str, Any] = config.to_yaml_dict()
    for dotted in SECRET_KEYS:
        section, _, leaf = dotted.partition(".")
        holder = data.get(section)
        if isinstance(holder, dict) and holder.get(leaf):
            holder[leaf] = redact(str(holder[leaf]))
    return data


# ---------------------------------------------------------------------------
# Setup tab: where things live
# ---------------------------------------------------------------------------


def data_dir_source(env_value: str | None, checkout_dir: Any) -> str:
    """Which of D25's three rules produced the data directory in force."""
    if env_value:
        return f"SF_DATA_DIR={env_value}"
    if checkout_dir:
        return f"source checkout ({checkout_dir})"
    return "platform data directory (no SF_DATA_DIR, not a source checkout)"


def where_things_live(config: Any, *, source: str = "") -> list[tuple[str, str]]:
    """The read-only "where is everything" table. Paths only, never secrets."""
    rows: list[tuple[str, str]] = [
        ("Data directory", str(config.data_dir)),
        ("Config file", str(config.config_path)),
        ("Engines", str(config.engines_dir)),
        ("Logs", str(config.logs_dir)),
        ("Registry database", str(config.db_path)),
        ("Downloads (in progress)", str(config.downloads_dir)),
        ("Model library", str(config.models.dir) if config.models.dir else "not set"),
    ]
    extra = [str(path) for path in config.models.extra_dirs]
    if extra:
        rows.append(("Extra model directories", ", ".join(extra)))
    if source:
        rows.append(("Data directory chosen by", source))
    return rows


# ---------------------------------------------------------------------------
# Setup tab: field metadata generated from the pydantic model
#
# The "Advanced" section is generated rather than hand-listed so that "every
# setting is reachable from the GUI" stays true when the config model grows: a
# new scalar key appears the moment it is declared, with no second edit here.
# Anything needing a real widget (secrets, the per-GPU maps, the quant-affinity
# table) is excluded and has its own section above. See DECISIONS.md D26.
# ---------------------------------------------------------------------------


#: Every key holding a credential, wherever it is rendered. Displayed masked,
#: only sent when :func:`masked_secret_changed` says it really changed, and
#: never logged -- ``config._register_secrets`` puts all three in the redactor.
SECRET_KEYS: Final = frozenset({"server.api_key", "hf.token", "mcp.pin"})

#: Keys whose *type* rules out a generated widget: two CUDA-index mappings and
#: the per-family quant-affinity table. Each has a purpose-built row control,
#: and each is excluded from the generated Advanced section so there is exactly
#: one control per key. ``excluded_devices`` is a plain ``list[int]`` and would
#: render as a text box, but a row of per-GPU checkboxes next to the live VRAM
#: figures is the only form in which it is actually checkable.
CUSTOM_WIDGET_KEYS: Final = frozenset(
    {
        "planner.excluded_devices",
        "planner.reserved_mb",
        "planner.quant_affinity",
    }
)


@dataclass(frozen=True)
class ConfigFieldSpec:
    """One config key, described well enough to render an input for it."""

    key: str
    section: str
    name: str
    kind: str
    options: tuple[str, ...] = ()
    default: Any = None
    help: str = ""

    @property
    def restart_required(self) -> bool:
        return self.key in RESTART_REQUIRED_KEYS

    @property
    def is_secret(self) -> bool:
        return self.key in SECRET_KEYS

    @property
    def label(self) -> str:
        return self.name.replace("_", " ")

    @property
    def summary(self) -> str:
        """The one-line explanation shown under a field."""
        default = "none" if self.default is None else str(self.default)
        base = self.help or f"{self.kind}, default {default}"
        if self.restart_required:
            return f"{base} · takes effect after a restart"
        return base


def _spec_kind(annotation: Any) -> tuple[str, tuple[str, ...]]:
    """Map a pydantic annotation onto a widget kind and its choices.

    Returns ``("unsupported", ())`` for anything with no honest scalar
    rendering (nested models, mappings). The caller drops those rather than
    guessing: a dict rendered as a text box is a data-loss bug waiting to
    happen, and it is exactly why ``planner.reserved_mb`` has its own row
    widget instead.
    """
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        # ``PositiveInt | None`` reaches pydantic as
        # ``Optional[Annotated[int, Gt(0)]]``: pydantic strips a TOP-level
        # Annotated into ``FieldInfo.metadata`` but leaves one nested inside a
        # Union alone, and the bare-scalar checks below never saw the ``int``.
        # The field was then "unsupported" and silently missing from the form
        # -- which is how ``engine.ubatch_size`` came to be typed ``int | None``
        # with its bound in a validator (WP20). The constraint still belongs to
        # the config model, which validates what was typed; the widget only
        # needs the scalar underneath.
        return _spec_kind(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        return "select", tuple(str(arg) for arg in typing.get_args(annotation))
    if origin in (types.UnionType, typing.Union):
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return _spec_kind(args[0])
        # A union of a scalar and a literal (``models.default_parallel`` is
        # ``int | Literal["auto"]``) has no single widget, and free text
        # round-trips both halves -- the config model validates whichever was
        # typed and rejects the rest.
        return "text", ()
    if origin in (list, Sequence):
        return "list", ()
    if origin is not None:
        # A parametrised generic that is not a list -- ``dict[int, int]``,
        # ``dict[str, QuantAffinity]``. Deliberately no fallback rendering.
        return "unsupported", ()
    if annotation is bool:
        return "bool", ()
    if annotation is int:
        return "int", ()
    if annotation is float:
        return "float", ()
    if annotation is str:
        return "text", ()
    if isinstance(annotation, type) and issubclass(annotation, PurePath):
        return "text", ()
    return "unsupported", ()


def config_field_specs(
    config_model: Any = None, *, help_text: Mapping[str, str] | None = None
) -> list[ConfigFieldSpec]:
    """Describe every scalar key of the config model, section by section.

    Generated from the pydantic model itself so it cannot drift. ``data_dir``
    and ``source_path`` are skipped: the first is derived from the environment
    (D25) and editing it here would fight whatever set ``SF_DATA_DIR``, and the
    second is bookkeeping that never belongs in a form.
    """
    from studioforge.config import Config as _Config

    model = config_model or _Config
    helps = dict(CONFIG_FIELD_HELP if help_text is None else help_text)
    specs: list[ConfigFieldSpec] = []
    for section_name, section_field in model.model_fields.items():
        if section_name in {"source_path", "data_dir"}:
            continue
        annotation = section_field.annotation
        if not (isinstance(annotation, type) and issubclass(annotation, BaseModel)):
            continue
        for name, field_info in annotation.model_fields.items():
            key = f"{section_name}.{field_info.alias or name}"
            kind, options = _spec_kind(field_info.annotation)
            if kind == "unsupported":
                continue
            default: Any = field_info.get_default(call_default_factory=True)
            if isinstance(default, PurePath):
                default = str(default)
            specs.append(
                ConfigFieldSpec(
                    key=key,
                    section=section_name,
                    name=field_info.alias or name,
                    kind="secret" if key in SECRET_KEYS else kind,
                    options=options,
                    default=default,
                    help=helps.get(key, ""),
                )
            )
    return specs


def config_sections(specs: Sequence[ConfigFieldSpec]) -> list[str]:
    """Top-level section names, in the order the config model declares them."""
    seen: list[str] = []
    for spec in specs:
        if spec.section not in seen:
            seen.append(spec.section)
    return seen


def spec_by_key(specs: Sequence[ConfigFieldSpec]) -> dict[str, ConfigFieldSpec]:
    return {spec.key: spec for spec in specs}


def advanced_field_specs(
    specs: Sequence[ConfigFieldSpec], covered: Sequence[str] = ()
) -> list[ConfigFieldSpec]:
    """The generated Advanced section: everything with no widget of its own.

    Three exclusion rules, each a rule rather than a list, so they keep holding
    as the config grows: keys already given a purpose-built control above (one
    control per key, always), secrets (they need the masked widget and the
    "did it really change" guard), and keys whose type has no honest scalar
    rendering -- :func:`config_field_specs` has already dropped those.
    """
    taken = set(covered) | CUSTOM_WIDGET_KEYS
    return [
        spec
        for spec in specs
        if spec.key not in taken and not spec.is_secret and spec.kind != "unsupported"
    ]


def spec_display_value(payload: Mapping[str, Any], spec: ConfigFieldSpec) -> Any:
    """The value to put in a widget, taken from the redacted config payload."""
    value = config_value(payload, spec.key)
    if spec.kind == "list":
        return ", ".join(str(item) for item in (value or []))
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "secret":
        return "" if value is None else str(value)
    if spec.kind in {"int", "float"}:
        return value
    return "" if value is None else str(value)


def spec_form_value(spec: ConfigFieldSpec, raw: Any) -> Any:
    """Coerce a widget value back to what ``apply_overrides`` expects.

    An integer typed into the free-text field that ``int | Literal["auto"]``
    needs is converted here, because YAML round-tripping ``"4"`` as a string
    would then fail the config model's own validator.
    """
    if spec.kind in {"int", "float", "bool", "list"}:
        return coerce_config_value(spec.kind, raw)
    if spec.kind == "select":
        return None if raw in (None, "") else str(raw)
    text = coerce_config_value("text", raw)
    if isinstance(text, str) and text.lstrip("-").isdigit():
        return int(text)
    return text


def config_updates_from_form(
    specs: Sequence[ConfigFieldSpec],
    payload: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Only the keys that genuinely changed, with secrets guarded.

    A secret is included only when :func:`masked_secret_changed` says the field
    holds something other than the placeholder it was rendered with -- the same
    guard the Server tab uses, for the same reason: posting ``"abcd...yz"``
    back would overwrite the real credential with nine literal characters and
    lock every client out.
    """
    updates: dict[str, Any] = {}
    for spec in specs:
        if spec.key not in values:
            continue
        raw = values[spec.key]
        current = config_value(payload, spec.key)
        if spec.is_secret:
            if masked_secret_changed(
                None if current is None else str(current), None if raw is None else str(raw)
            ):
                updates[spec.key] = str(raw).strip()
            continue
        coerced = spec_form_value(spec, raw)
        if spec.kind == "list":
            current = list(current or [])
        elif spec.kind in {"int", "float"} and current is not None and coerced is not None:
            current = int(current) if spec.kind == "int" else float(current)
        if coerced != current:
            updates[spec.key] = coerced
    return updates


def save_result_text(payload: Mapping[str, Any] | None) -> str:
    """What ``PATCH /api/config`` did, including the restart it still needs."""
    if not payload:
        return "nothing changed"
    changed = [str(key) for key in payload.get("updated") or []]
    if not changed:
        return "nothing changed"
    restart = [str(key) for key in payload.get("restart_required") or []]
    message = "saved: " + ", ".join(sorted(changed))
    if restart:
        message += " — restart required for: " + ", ".join(sorted(restart))
    return message


#: Hand-written one-liners for the keys a first-run user has to think about.
#: Anything without an entry falls back to "<kind>, default <x>", which is why
#: a newly added key is never *missing* an explanation, only a good one.
CONFIG_FIELD_HELP: Final[Mapping[str, str]] = {
    "server.host": "Bind address. 0.0.0.0 exposes the gateway on the LAN and the tailnet.",
    "server.port": "Gateway port. 1234 is LM Studio's, so OpenAI clients need no change.",
    "server.api_key": "Inference + panel credential. Blank disables auth (LAN/tailnet trust).",
    "server.cors_origins": "Comma separated; * allows every browser origin.",
    "gui.port": "This control panel's own port.",
    "watchdog.port": "The recovery sidecar's port. It is a separate process.",
    "mcp.pin": "Short pairing code for the MCP path. Rotate it if it leaks.",
    "mcp.pin_required": "Off falls back to the API key alone for MCP.",
    "models.dir": "Primary GGUF library root. Scanned in place; nothing is copied or moved.",
    "models.default_ctx": "Context FLOOR. The planner never drops below this to buy a slot.",
    "models.target_ctx": (
        "Context every load AIMS for; the planner halves down from here to what fits (D14)."
    ),
    "models.thinking_default_ctx": (
        "Floor for reasoning models, which spend their budget thinking before answering."
    ),
    "models.default_parallel": (
        "'auto' sizes conversation slots per model and placement (D17); an integer is honoured."
    ),
    "models.ctx_per_slot_default": "Per-slot context the slot-count estimator assumes.",
    "models.default_kv_cache_type": (
        "'auto' keeps full-quality KV where it is affordable and quantizes only where it is not."
    ),
    "models.default_ttl_s": "Idle unload timer, in seconds. 0 means never idle-unload.",
    "models.auto_load_pinned": (
        "Load pinned models at startup and keep them resident: one that goes down "
        "is reloaded automatically (D41)."
    ),
    "models.default_model": "Served when a request omits 'model', or names local-model/default.",
    "models.preload_default_model": "Load that default at startup rather than on first use.",
    "models.default_flash_attn": "'on' everywhere from Ampere up; a large KV-bandwidth win.",
    "models.default_cache_reuse": "Prompt-cache reuse: the biggest agent-workload latency win.",
    "models.default_reasoning_format": (
        "'none' keeps a thinking model's output in message.content instead of emptying it."
    ),
    "engine.pinned_tag": "The llama.cpp release this install uses unless a newer one is activated.",
    "engine.cuda_variant": "'auto' picks the highest CUDA build this driver can run.",
    "engine.keep_versions": "How many old engine directories to keep when pruning.",
    "engine.allow_source_build": "Fall back to building llama.cpp when no prebuilt asset fits.",
    "planner.headroom_fraction": "Fraction of EVERY GPU held back from the planner.",
    "planner.on_insufficient": "evict = unload LRU unpinned models; reject = refuse the load.",
    "planner.preference": (
        "quality = best KV cache that reaches the context floor, then the biggest window at "
        "that quality; throughput = the biggest window, preferring one that serves two slots."
    ),
    "planner.compute_overhead_fraction": "Calibrated allowance for compute and graph buffers.",
    "hf.token": "Needed only for gated or private HuggingFace repositories.",
    "hf.max_concurrent_downloads": "Parallel downloads. Raising it rarely helps a single link.",
    "hf.chunk_bytes": "Download chunk size, in bytes.",
    "logging.level": "DEBUG/INFO/WARNING/ERROR for the whole process.",
    "logging.json": "Structured JSON log lines instead of the human-readable renderer.",
}
