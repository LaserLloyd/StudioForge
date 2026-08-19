"""Engine manager: versioned ``llama-server`` artifacts under ``engines/<tag>/``.

The engine is *never* "whatever ``llama-server`` is on PATH". Every build is a
pinned, downloaded, smoke-tested artifact in its own directory, because
llama.cpp releases rename flags and occasionally break older GGUFs (see
DECISIONS.md D2/D3 -- ``b10425`` removed the entire ``--draft*`` surface). A
versioned directory plus a per-model engine pin means one bad release can be
rolled back by editing one config key instead of reinstalling anything.

Responsibilities:

* discovery -- list upstream releases and their assets (GitHub API);
* selection -- pick the right prebuilt archive for this box's driver/arch;
* install -- download, verify, extract (flattening nested archives), fetch the
  matching CUDA runtime bundle, smoke test;
* inventory -- what is installed, which tag is active, pruning old versions;
* verification -- ``--version`` / ``--list-devices`` / a real micro-load;
* flag surface -- the ``--help`` flag set used to validate the expert-tier
  "extra flags" box at save time rather than at load time.

This module deliberately does not touch the SQLite registry: the active tag is
persisted as ``engines/active.json`` so the engine layer stays usable during
first-run bootstrap, before any database exists.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import platform
import re
import shlex
import shutil
import socket
import stat
import subprocess
import time
import zipfile
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx
import psutil

from studioforge.config import Config
from studioforge.errors import StudioForgeError
from studioforge.logging import get_logger
from studioforge.types import EngineInfo, GpuInfo

if TYPE_CHECKING:  # pragma: no cover - import cycle / concurrently authored module
    from studioforge.core.gpu import GpuProbe

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BIN_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
GITHUB_API = "https://api.github.com"
ACTIVE_FILE = "active.json"
META_FILE = "engine.json"
FLAGS_FILE = "flags.txt"
HELP_FILE = "help.txt"

ProgressFn = Callable[[str, float], None]

#: Flags the manager owns. A user override here does not merely change a
#: default, it breaks the system: ``--port``/``--host`` would make the gateway
#: proxy to nothing, ``--model``/``--alias`` would desync the registry, and
#: ``--n-gpu-layers`` would silently defeat the GPU-only policy by spilling
#: layers to CPU. Maps every alias to the canonical name for error messages.
MANAGED_FLAGS: dict[str, str] = {
    "-m": "--model",
    "--model": "--model",
    "-mu": "--model-url",
    "--model-url": "--model-url",
    "--port": "--port",
    "--host": "--host",
    "-a": "--alias",
    "--alias": "--alias",
    "-ngl": "--n-gpu-layers",
    "--gpu-layers": "--n-gpu-layers",
    "--n-gpu-layers": "--n-gpu-layers",
    # b10425's --fit (default ON) "adjusts unset arguments to fit in device
    # memory". Allowing it back in via extra flags would reintroduce a silent
    # partial-offload path, which the GPU-only policy forbids: the planner owns
    # placement, and an over-commit must fail loudly rather than be shrunk.
    "-fit": "--fit",
    "--fit": "--fit",
    "-fitt": "--fit-target",
    "--fit-target": "--fit-target",
    "-fitc": "--fit-ctx",
    "--fit-ctx": "--fit-ctx",
}

#: Removed-flag hints used as a *fallback* when the engine's own ``--help``
#: output does not identify the flag as removed. ``b10425`` renamed the whole
#: speculative-decoding surface; the old spellings are either gone or accepted
#: and ignored, which looks exactly like speculative decoding doing nothing.
REMOVED_FLAG_HINTS: dict[str, str] = {
    "--draft": "--spec-draft-n-max",
    "--draft-n": "--spec-draft-n-max",
    "--draft-max": "--spec-draft-n-max",
    "--draft-min": "--spec-draft-n-min",
    "--draft-n-min": "--spec-draft-n-min",
    "--cache-type-k-draft": "--spec-draft-type-k",
    "--cache-type-v-draft": "--spec-draft-type-v",
    "--n-gpu-layers-draft": "--spec-draft-ngl",
}

_REMOVED_PHRASE = "the argument has been removed"

#: Rejected anywhere in an extra-flags token: these only mean something to a
#: shell, and their presence signals either a copy-pasted shell snippet or an
#: injection attempt. We exec the argv directly, so they can never expand --
#: rejecting them early turns a silently-wrong flag value into a save-time error.
_SHELL_METACHARS = frozenset("|&;<>$`\n\r()")

#: Flags whose *values* legitimately contain ``;``, ``|``, ``(`` and ``)``:
#: sampler chains and tensor-override regexes. Only the truly shell-only
#: characters stay banned for these.
_RELAXED_VALUE_FLAGS = frozenset(
    {
        "--samplers",
        "--sampling-seq",
        "--dry-sequence-breaker",
        "--override-tensor",
        "-ot",
        "--override-tensor-draft",
        "-otd",
        "--spec-draft-override-tensor",
        "--override-kv",
        "--tensor-split",
        "-ts",
        "--grammar",
        "--chat-template",
        "--logit-bias",
        "-l",
    }
)
_RELAXED_METACHARS = frozenset("$`\n\r&<>")

_FLAG_START_RE = re.compile(r"^-{1,2}[A-Za-z0-9]")
_FLAG_TOKEN_RE = re.compile(r"^-{1,2}[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LONG_FLAG_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9_-]*")
_REPLACEMENT_RE = re.compile(r"use (?:the respective )?(--[A-Za-z0-9*][A-Za-z0-9*_-]*)")
_PERCENT_RE = re.compile(r"\[\s*(\d{1,3})%\]")
_BUILD_RE = re.compile(r"build\s+(\d+)")
_DEVICE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*\d*)\s*:\s*(.+)$")
_CUDA_VER_RE = re.compile(r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)\.(\d+)")

#: ``llama-<tag>-bin-<os>[-<backend>[-<ver>]]-<arch>.zip``
_ASSET_RE = re.compile(
    r"^llama-(?P<tag>[A-Za-z0-9._]+)-bin-(?P<os>win|ubuntu|linux|macos)-(?P<rest>.+)\.zip$"
)
_CUDART_RE = re.compile(
    r"^cudart-llama-bin-(?P<os>win|ubuntu|linux)-cuda-(?P<ver>[0-9.]+)-(?P<arch>x64|arm64)\.zip$"
)

#: The only tag scheme this manager can act on. llama.cpp's ordinary build
#: releases are ``bNNNN`` and *everything downstream assumes it*:
#: ``engine.pinned_tag``, the ``llama-bNNNN-bin-...`` asset-name parser,
#: ``engines/<tag>/``, ``active.json`` and the ``git clone --branch <tag>``
#: source-build path. A tag outside the scheme is not an engine we can install.
ENGINE_TAG_RE = re.compile(r"^b\d+$")
#: What may name an engine directory at all: one path component, no
#: separators, no drive letters. Looser than ENGINE_TAG_RE on purpose (a local
#: source build may carry a descriptive tag); strict about escaping engines/.
_SAFE_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

#: Same scheme, plus the ``-local`` suffix :meth:`build_from_source` appends, so
#: a source-built active tag still compares numerically against upstream.
_BUILD_NUMBER_RE = re.compile(r"^b(\d+)(?:-local)?$")

_ARCH_ALIASES = {
    "amd64": "x64",
    "x86_64": "x64",
    "x64": "x64",
    "aarch64": "arm64",
    "arm64": "arm64",
}

#: Preferred tiny model for the micro-load half of the smoke test, expressed
#: RELATIVE to each configured model directory (LM Studio's ``publisher/repo/``
#: layout). An absolute path here would only ever be correct on one machine.
_PREFERRED_TINY_MODELS = (
    Path("lmstudio-community") / "Qwen2.5-0.5B-Instruct-GGUF" / "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
)
_TINY_MODEL_MAX_BYTES = 2 * 1024 * 1024 * 1024


class EngineError(StudioForgeError):
    """An engine could not be resolved, installed, or verified.

    Subclasses :class:`StudioForgeError` (itself an ``Exception``) so a failure
    surfaced through the HTTP API still renders as the OpenAI error envelope.
    """

    status_code = 500
    error_type = "server_error"
    code = "engine_error"


# ---------------------------------------------------------------------------
# Asset model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineAsset:
    """One downloadable release archive."""

    tag: str
    name: str
    url: str
    size_bytes: int
    variant: str  # "cuda-13.3", "cuda-12.4", "cpu", "vulkan", "rocm-7.14", ...
    needs_cudart: bool
    cudart_url: str | None
    os_token: str = "win"
    arch: str = "x64"

    @property
    def is_cuda(self) -> bool:
        return self.variant == "cuda" or self.variant.startswith("cuda-")

    @property
    def cuda_version(self) -> tuple[int, int] | None:
        """Parsed CUDA toolkit version, or ``None`` for unversioned/non-CUDA."""
        if not self.variant.startswith("cuda-"):
            return None
        return _parse_version(self.variant[len("cuda-") :])


def _parse_version(text: str) -> tuple[int, int] | None:
    parts = text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None
    return major, minor


def _norm_arch(token: str) -> str:
    return _ARCH_ALIASES.get(token.lower(), token.lower())


def build_number(tag: str | None) -> int | None:
    """The integer behind a ``bNNNN`` (or ``bNNNN-local``) tag, else ``None``.

    Update checks must compare builds *numerically*. ``latest != current`` --
    which is what every call site used to do -- calls a downgrade an update:
    ``b10441`` is installed on this box while ``b10425`` is active, so a rollback
    would have been advertised as something to install. It also mis-orders
    lexically once the build number gains a digit (``"b9999" > "b10000"``).
    """
    if not tag:
        return None
    match = _BUILD_NUMBER_RE.match(tag)
    return int(match.group(1)) if match else None


def parse_asset_name(name: str) -> tuple[str, str, str, str] | None:
    """Split a release asset filename into ``(tag, os, variant, arch)``.

    Asset naming drifts between platforms and eras (Windows carries an explicit
    ``cuda-13.3``, Linux of this era is just ``ubuntu-cuda``), so selection is
    driven by pattern-matching the *actual* asset list rather than by
    constructing filenames and hoping they exist.
    """
    match = _ASSET_RE.match(name)
    if match is None:
        return None
    rest = match.group("rest")
    parts = rest.split("-")
    if parts and _norm_arch(parts[-1]) in {"x64", "arm64"}:
        arch = _norm_arch(parts[-1])
        backend = parts[:-1]
    else:
        arch = "x64"
        backend = parts
    variant = "-".join(backend) if backend else "cpu"
    return match.group("tag"), match.group("os"), variant, arch


# ---------------------------------------------------------------------------
# --help parsing
# ---------------------------------------------------------------------------


def _split_help_entry(line: str) -> tuple[list[str], str]:
    """Split one ``--help`` entry header into its flag aliases + description.

    llama.cpp lays entries out as ``-x, --long, --alias VALUE  description``,
    where the flag list can overflow the description column. Consuming leading
    comma/space separated *flag-shaped* tokens and treating the first
    non-flag-shaped token as the start of the description handles both that
    overflow and value placeholders like ``{none,layer,row}``.
    """
    flags: list[str] = []
    pos = 0
    length = len(line)
    while pos < length:
        while pos < length and line[pos] in " \t,":
            pos += 1
        start = pos
        while pos < length and not line[pos].isspace() and line[pos] != ",":
            pos += 1
        token = line[start:pos]
        if not token:
            break
        if _FLAG_TOKEN_RE.match(token):
            flags.append(token)
        else:
            return flags, line[start:].strip()
    return flags, ""


def parse_help_entries(help_text: str) -> list[tuple[list[str], str]]:
    """Parse ``llama-server --help`` into ``(flag aliases, description)`` pairs."""
    entries: list[tuple[list[str], str]] = []
    flags: list[str] | None = None
    desc: list[str] = []

    def flush() -> None:
        nonlocal flags, desc
        if flags:
            entries.append((flags, " ".join(desc).strip()))
        flags = None
        desc = []

    for raw in help_text.splitlines():
        if _FLAG_START_RE.match(raw):
            flush()
            flags, first = _split_help_entry(raw)
            desc = [first] if first else []
        elif not raw.strip():
            continue
        elif flags is not None and raw.startswith("  "):
            desc.append(raw.strip())
        else:
            flush()
    flush()
    return entries


def removed_flags_from_help(help_text: str) -> dict[str, str | None]:
    """Flags the engine itself reports as removed, mapped to its suggestion.

    Detected from the help text (``"the argument has been removed"``) rather
    than from a hardcoded table, so a future release that retires another flag
    is caught without a code change.
    """
    removed: dict[str, str | None] = {}
    for flags, desc in parse_help_entries(help_text):
        if _REMOVED_PHRASE not in desc.lower():
            continue
        match = _REPLACEMENT_RE.search(desc)
        replacement = match.group(1) if match else None
        for flag in flags:
            removed[flag] = replacement
    return removed


def flags_from_help(help_text: str) -> set[str]:
    """Every flag spelling the engine declares, long and short."""
    found: set[str] = set()
    for flags, _desc in parse_help_entries(help_text):
        found.update(flags)
    # Safety net: long flags only mentioned inside a description (e.g. the
    # replacement named by a removed flag) are still real flags.
    for token in _LONG_FLAG_RE.findall(help_text):
        found.add(token.rstrip("-"))
    found.discard("--")
    return found


# ---------------------------------------------------------------------------
# Engine feature detection (which of the newer knobs this build actually has)
# ---------------------------------------------------------------------------

#: Where :func:`parse_engine_features` gets cached, next to the binary.
FEATURES_FILE = "features.json"

_BRACED_RE = re.compile(r"\{([^}]+)\}")
_BRACKETED_RE = re.compile(r"\[([^\]]+)\]")
_DEFAULT_RE = re.compile(r"\(default:\s*([^,)]+)")
_ENUM_TOKEN_RE = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")


@dataclass(frozen=True)
class EngineFeatures:
    """What one engine build advertises, read from its own ``--help``.

    **Why this exists at all.** D2 recorded the incident this guards against:
    ``b10425`` renamed the whole speculative-decoding surface and *accepted the
    old spellings while ignoring them*. A flag that is silently ignored looks
    exactly like the feature not helping, which is the most expensive kind of
    wrong. So StudioForge never passes a flag on faith: every optional flag in
    :mod:`studioforge.core.supervisor` is gated on the active engine declaring
    it, and the *values* of the enum flags (``--split-mode``,
    ``--spec-type``, ``--flash-attn``) are read from the same output rather
    than hardcoded -- upstream adds speculative types faster than this file
    could track them.

    :attr:`known` is False for the "we could not read the help" case. That is
    deliberately not the same as "the engine has nothing": an unknown engine
    keeps the pre-WP20 launch surface (no new flags) instead of guessing.
    """

    tag: str = ""
    known: bool = False
    flags: frozenset[str] = frozenset()
    #: Values ``-sm/--split-mode`` accepts, e.g. ``("none", "layer", "row", "tensor")``.
    split_modes: tuple[str, ...] = ()
    #: Values ``--spec-type`` accepts, e.g. ``("none", "draft-simple", "draft-mtp", ...)``.
    spec_types: tuple[str, ...] = ()
    #: Values ``-fa/--flash-attn`` accepts, e.g. ``("on", "off", "auto")``.
    flash_attn_values: tuple[str, ...] = ()
    backend_sampling: bool = False
    kv_unified: bool = False
    #: The engine's own words for when unified KV is on, e.g. "enabled if
    #: number of slots is auto". Recorded verbatim because the semantics moved
    #: between releases and D38 pins them to a measurement, not to a memory.
    kv_unified_default: str = ""
    cache_ram: bool = False
    cache_ram_default_mib: int | None = None
    ctx_checkpoints: bool = False
    ctx_checkpoints_default: int | None = None
    fit: bool = False
    reasoning_budget: bool = False
    spec_draft_n_max_default: int | None = None

    @classmethod
    def unknown(cls, tag: str = "") -> EngineFeatures:
        """The "could not read this engine's help" value. Advertises nothing."""
        return cls(tag=tag, known=False)

    def has(self, flag: str) -> bool:
        return self.known and flag in self.flags

    def supports_spec(self, spec_type: str) -> bool:
        """Whether every comma-separated member of ``spec_type`` is offered."""
        if not self.known or "--spec-type" not in self.flags:
            return False
        wanted = [part.strip() for part in spec_type.split(",") if part.strip()]
        return bool(wanted) and all(part in self.spec_types for part in wanted)

    def supports_split(self, mode: str) -> bool:
        return self.known and mode in self.split_modes

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "known": self.known,
            "split_modes": list(self.split_modes),
            "spec_types": list(self.spec_types),
            "flash_attn_values": list(self.flash_attn_values),
            "backend_sampling": self.backend_sampling,
            "kv_unified": self.kv_unified,
            "kv_unified_default": self.kv_unified_default,
            "cache_ram": self.cache_ram,
            "cache_ram_default_mib": self.cache_ram_default_mib,
            "ctx_checkpoints": self.ctx_checkpoints,
            "ctx_checkpoints_default": self.ctx_checkpoints_default,
            "fit": self.fit,
            "reasoning_budget": self.reasoning_budget,
            "spec_draft_n_max_default": self.spec_draft_n_max_default,
            "flags": sorted(self.flags),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EngineFeatures:
        def _tuple(key: str) -> tuple[str, ...]:
            value = data.get(key)
            return tuple(str(v) for v in value) if isinstance(value, list) else ()

        def _int(key: str) -> int | None:
            value = data.get(key)
            return int(value) if isinstance(value, int) else None

        flags = data.get("flags")
        return cls(
            tag=str(data.get("tag") or ""),
            known=bool(data.get("known")),
            flags=frozenset(str(f) for f in flags) if isinstance(flags, list) else frozenset(),
            split_modes=_tuple("split_modes"),
            spec_types=_tuple("spec_types"),
            flash_attn_values=_tuple("flash_attn_values"),
            backend_sampling=bool(data.get("backend_sampling")),
            kv_unified=bool(data.get("kv_unified")),
            kv_unified_default=str(data.get("kv_unified_default") or ""),
            cache_ram=bool(data.get("cache_ram")),
            cache_ram_default_mib=_int("cache_ram_default_mib"),
            ctx_checkpoints=bool(data.get("ctx_checkpoints")),
            ctx_checkpoints_default=_int("ctx_checkpoints_default"),
            fit=bool(data.get("fit")),
            reasoning_budget=bool(data.get("reasoning_budget")),
            spec_draft_n_max_default=_int("spec_draft_n_max_default"),
        )


def _entry_descriptions(help_text: str) -> dict[str, str]:
    """``{flag spelling: description}`` for every entry, aliases included."""
    table: dict[str, str] = {}
    for flags, desc in parse_help_entries(help_text):
        for flag in flags:
            table[flag] = desc
    return table


def _enum_values(desc: str, pattern: re.Pattern[str], separator: str) -> tuple[str, ...]:
    """Values from a ``{a,b,c}`` / ``[a|b|c]`` placeholder in a description."""
    match = pattern.search(desc)
    if match is None:
        return ()
    return tuple(
        token
        for token in (part.strip() for part in match.group(1).split(separator))
        if _ENUM_TOKEN_RE.match(token)
    )


def _default_int(desc: str) -> int | None:
    match = _DEFAULT_RE.search(desc)
    if match is None:
        return None
    try:
        return int(match.group(1).strip())
    except ValueError:
        return None


def _default_text(desc: str) -> str:
    match = _DEFAULT_RE.search(desc)
    return match.group(1).strip() if match else ""


def parse_engine_features(help_text: str, tag: str = "") -> EngineFeatures:
    """Read one engine's optional-feature surface out of its ``--help``.

    Pure and total: anything it cannot find is simply absent, because the
    consumer's rule is "never pass a flag the engine did not advertise" and a
    missing entry must therefore mean "do not pass it", never "assume yes".
    """
    entries = _entry_descriptions(help_text)
    flags = flags_from_help(help_text) - set(removed_flags_from_help(help_text))

    split_desc = entries.get("--split-mode", "")
    split_modes = _enum_values(split_desc, _BRACED_RE, ",")

    # ``--spec-type`` lists its values bare (no braces), so the enumeration is
    # the first whitespace-delimited token of the description.
    spec_desc = entries.get("--spec-type", "")
    first = spec_desc.split(" ", 1)[0] if spec_desc else ""
    spec_types = tuple(
        token
        for token in (part.strip() for part in first.split(","))
        if _ENUM_TOKEN_RE.match(token)
    )

    kv_desc = entries.get("--kv-unified", "")
    return EngineFeatures(
        tag=tag,
        known=bool(flags),
        flags=frozenset(flags),
        split_modes=split_modes,
        spec_types=spec_types,
        flash_attn_values=_enum_values(entries.get("--flash-attn", ""), _BRACKETED_RE, "|"),
        backend_sampling="--backend-sampling" in flags,
        kv_unified="--kv-unified" in flags and "--no-kv-unified" in flags,
        kv_unified_default=_default_text(kv_desc),
        cache_ram="--cache-ram" in flags,
        cache_ram_default_mib=_default_int(entries.get("--cache-ram", "")),
        ctx_checkpoints="--ctx-checkpoints" in flags,
        ctx_checkpoints_default=_default_int(entries.get("--ctx-checkpoints", "")),
        fit="--fit" in flags,
        reasoning_budget="--reasoning-budget" in flags,
        spec_draft_n_max_default=_default_int(entries.get("--spec-draft-n-max", "")),
    )


def read_features_file(directory: Path, tag: str = "") -> EngineFeatures | None:
    """The cached ``features.json`` for an engine directory, if it is readable."""
    path = directory / FEATURES_FILE
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("known"):
        return None
    features = EngineFeatures.from_dict(data)
    return features if not tag or features.tag == tag else None


def write_features_file(directory: Path, features: EngineFeatures) -> None:
    """Cache ``features`` next to the binary. Best effort; never raises."""
    with contextlib.suppress(OSError):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / FEATURES_FILE).write_text(
            json.dumps(features.to_dict(), indent=2), encoding="utf-8"
        )


def cached_engine_features(directory: Path, tag: str = "") -> EngineFeatures:
    """Features from the files already on disk. Never spawns a process.

    ``features.json`` -> ``help.txt`` -> "unknown". For every caller that must
    not block: a report route, a GUI panel, anything on an event loop.
    """
    cached = read_features_file(directory, tag)
    if cached is not None:
        return cached
    help_path = directory / HELP_FILE
    if help_path.is_file():
        with contextlib.suppress(OSError):
            text = help_path.read_text(encoding="utf-8")
            if text.strip():
                features = parse_engine_features(text, tag)
                if features.known:
                    write_features_file(directory, features)
                return features
    return EngineFeatures.unknown(tag)


def probe_engine_features(binary: Path, tag: str = "") -> EngineFeatures:
    """Features for the build at ``binary``, reading (or filling) the caches.

    Synchronous on purpose: the supervisor calls it through
    ``asyncio.to_thread`` on the load path, where an ``await`` into the engine
    manager would be a new dependency between two modules that deliberately
    know nothing about each other. Falls back through ``features.json`` ->
    ``help.txt`` -> actually running ``--help``, and returns
    :meth:`EngineFeatures.unknown` rather than raising if all three fail: a
    model must still load on a box where the help cannot be read.

    **The subprocess only ever runs a file actually named like the engine
    binary.** Anything else that a ``resolve_binary`` hook returns -- a test
    stub, a wrapper script -- is read from disk or reported unknown, never
    executed. Executing whatever was handed to us as a side effect of *building
    a command line* is a side effect nobody asked for: it cost two flaky test
    failures here, where launching the fake ``fake_llama_server.py`` with
    ``--help`` started a real HTTP server on the port the next test wanted.
    """
    directory = binary.parent
    from_disk = cached_engine_features(directory, tag)
    if from_disk.known:
        return from_disk

    if not binary.is_file() or binary.name.lower() != BIN_NAME.lower():
        return EngineFeatures.unknown(tag)
    try:
        completed = subprocess.run(  # noqa: S603 - our own pinned engine binary
            [str(binary), "--help"],
            capture_output=True,
            cwd=str(directory),
            timeout=120,
            check=False,
            **_spawn_kwargs(),
        )
    except (OSError, subprocess.SubprocessError):
        return EngineFeatures.unknown(tag)
    text = completed.stdout.decode("utf-8", "replace") + completed.stderr.decode("utf-8", "replace")
    if not text.strip():
        return EngineFeatures.unknown(tag)
    with contextlib.suppress(OSError):
        (directory / HELP_FILE).write_text(text, encoding="utf-8")

    features = parse_engine_features(text, tag)
    if features.known:
        write_features_file(directory, features)
    return features


# ---------------------------------------------------------------------------
# Archive extraction
# ---------------------------------------------------------------------------


def _common_prefix(names: Sequence[str]) -> str:
    """Longest directory prefix shared by every archive member.

    Windows assets are flat; Linux assets nest ``<name>/build/bin/``. Stripping
    the whole shared prefix (not just one level) lands the binary directly in
    ``engines/<tag>/`` either way, which is what every other code path assumes.
    """
    files = [n.replace("\\", "/") for n in names if not n.endswith("/")]
    if len(files) < 2:
        return ""
    split = [f.split("/")[:-1] for f in files]
    prefix = split[0]
    for parts in split[1:]:
        keep = 0
        for a, b in zip(prefix, parts, strict=False):
            if a != b:
                break
            keep += 1
        prefix = prefix[:keep]
        if not prefix:
            return ""
    return "/".join(prefix) + "/" if prefix else ""


def extract_engine_zip(zip_path: Path, dest: Path) -> Path | None:
    """Extract ``zip_path`` into ``dest``, flattening a shared top-level dir.

    Returns the extracted server binary, if the archive contained one. Members
    with absolute paths or ``..`` components are skipped: a release archive
    should never contain them, and honouring one would write outside ``dest``.
    """
    dest.mkdir(parents=True, exist_ok=True)
    binary: Path | None = None
    with zipfile.ZipFile(zip_path) as archive:
        prefix = _common_prefix(archive.namelist())
        for member in archive.infolist():
            name = member.filename.replace("\\", "/")
            if prefix and name.startswith(prefix):
                name = name[len(prefix) :]
            if not name or name.endswith("/"):
                continue
            parts = [p for p in name.split("/") if p not in ("", ".")]
            if any(p == ".." for p in parts) or Path(name).is_absolute():
                log.warning("engine.zip.skip_unsafe_member", member=member.filename)
                continue
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            if target.name == BIN_NAME:
                binary = target
    return binary


def _make_executable(directory: Path) -> None:
    if os.name == "nt":
        return
    for path in directory.rglob("*"):
        if path.is_file() and (path.suffix in ("", ".sh") or path.name.startswith("llama")):
            with contextlib.suppress(OSError):
                path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn_kwargs() -> dict[str, Any]:
    """Detach the child into its own group so we can kill the whole tree."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_process_tree(pid: int, *, timeout: float = 5.0) -> None:
    """Terminate ``pid`` and every descendant.

    A bare ``terminate()`` on ``llama-server`` can leave helper children alive
    still holding VRAM, which then makes the next load fail its VRAM check for
    no visible reason. Always kill the tree.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    try:
        victims = parent.children(recursive=True)
    except psutil.Error:
        victims = []
    victims.append(parent)
    for proc in victims:
        with contextlib.suppress(psutil.Error):
            proc.terminate()
    _gone, alive = psutil.wait_procs(victims, timeout=timeout)
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    with contextlib.suppress(psutil.Error):
        psutil.wait_procs(alive, timeout=timeout)


async def _drain(stream: asyncio.StreamReader | None, sink: deque[str]) -> None:
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except (ValueError, OSError):  # pragma: no cover - oversized/closed pipe
            return
        if not line:
            return
        sink.append(line.decode("utf-8", "replace").rstrip())


@dataclass
class _SmokeResult:
    ok: bool
    detail: str
    version_ok: bool = False
    version_string: str | None = None
    model_used: Path | None = None
    no_model: bool = False


class _ProbeLike(Protocol):  # pragma: no cover - structural typing only
    def gpus(self) -> Sequence[GpuInfo]: ...


def _emit(progress: ProgressFn | None, phase: str, fraction: float) -> None:
    if progress is None:
        return
    with contextlib.suppress(Exception):  # a broken GUI callback must not fail an install
        progress(phase, max(0.0, min(1.0, fraction)))


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class EngineManager:
    """Owns ``<data_dir>/engines/``: discovery, install, verification, pruning."""

    def __init__(
        self,
        config: Config,
        *,
        probe: GpuProbe | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self._probe: Any = probe
        self._client = client
        self._owns_client = client is None
        self._flag_cache: dict[str, set[str]] = {}
        self._removed_cache: dict[str, dict[str, str | None]] = {}
        self._help_cache: dict[str, str] = {}
        self._features_cache: dict[str, EngineFeatures] = {}
        #: One lock per tag around install/build. Two installs of one tag at
        #: once -- the Setup tab's Install button clicked while boot is already
        #: installing, or clicked twice -- would both stream into the same
        #: ``downloads/<asset>.zip.part`` and both extract into the same
        #: directory; the second finishes with a corrupt archive or a
        #: half-overwritten engine. Serialised, the second caller finds the
        #: engine present and passes through ``already_present``.
        self._install_locks: dict[str, asyncio.Lock] = {}
        self.os_token = self._detect_os_token()
        self.arch_token = _norm_arch(platform.machine() or "x86_64")

    def _install_lock(self, tag: str) -> asyncio.Lock:
        lock = self._install_locks.get(tag)
        if lock is None:
            lock = asyncio.Lock()
            self._install_locks[tag] = lock
        return lock

    # -- lifecycle ------------------------------------------------------

    @staticmethod
    def _detect_os_token() -> str:
        if os.name == "nt":
            return "win"
        if platform.system() == "Darwin":
            return "macos"
        return "ubuntu"

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=300.0),
                headers=self._api_headers(),
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _api_headers() -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "studioforge",
        }
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # -- paths ----------------------------------------------------------

    @property
    def engines_dir(self) -> Path:
        return self.config.engines_dir

    def engine_dir(self, tag: str) -> Path:
        # A tag arrives from config, a per-model setting or the install route's
        # body and becomes a directory name under engines/. Anything that is
        # not one plain path component ("../..", "b1/../../x", "") is refused
        # before it can name a directory outside the engines tree.
        if not _SAFE_TAG_RE.fullmatch(tag or "") or tag.strip(".") == "":
            raise EngineError(
                f"invalid engine tag {tag!r}: expected a llama.cpp build tag such as "
                "'b10425' (letters, digits, '.', '_' or '-' only)"
            )
        return self.engines_dir / tag

    @property
    def vendor_dir(self) -> Path:
        """Pre-existing llama.cpp checkout used to skip a fresh clone."""
        env = os.environ.get("SF_VENDOR_LLAMA_CPP")
        if env:
            return Path(env)
        return self.config.data_dir.parent / "vendor" / "llama.cpp"

    # ------------------------------------------------------------------
    # Discovery / selection
    # ------------------------------------------------------------------

    async def list_releases(
        self, limit: int = 30, *, include_prerelease: bool = False
    ) -> list[str]:
        """Newest ``limit`` *installable* release tags, newest build first.

        Filtered, not raw. On 2026-08-18 llama.cpp published a **prerelease**
        tagged ``v0.1.2`` carrying no prebuilt Windows CUDA archives at all,
        sitting at the top of the release list above the ordinary ``b10488`` /
        ``b10486`` builds. Returning it unfiltered made the Server tab offer
        "Engine b10425 -- v0.1.2 is available" behind a button that could only
        ever fail: ``no GPU-capable llama-server build ... available variants:
        <none>``. Three filters, each guarding a different assumption:

        * ``draft`` entries are not published at all and their assets 404;
        * ``prerelease`` entries are excluded unless ``include_prerelease``,
          because upstream uses them for things that are not build releases;
        * the tag must match :data:`ENGINE_TAG_RE`. This one is **unconditional**
          -- ``include_prerelease=True`` widens the release *kind*, never the tag
          scheme -- because ``pinned_tag``, the ``llama-bNNNN-bin-...`` asset
          parser, ``engines/<tag>/``, ``active.json`` and the source-build
          ``git clone --branch <tag>`` all assume ``bNNNN``. A ``vX.Y.Z`` tag is
          not an engine this manager can install under any flag.

        Sorted by build number descending rather than trusting the API's order,
        and over-fetched (``per_page >= 2*limit``) so filtering out prereleases
        cannot return fewer tags than asked for.
        """
        url = f"{GITHUB_API}/repos/{self.config.engine.repo}/releases"
        per_page = min(100, max(limit * 2, 20))
        try:
            resp = await self.client.get(url, params={"per_page": per_page})
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"could not list llama.cpp releases: {exc}") from exc
        if not isinstance(payload, list):
            raise EngineError("unexpected response listing llama.cpp releases")

        tags: list[str] = []
        skipped: list[dict[str, str]] = []
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            tag = entry.get("tag_name")
            if not isinstance(tag, str) or not tag:
                continue
            reason = _release_skip_reason(entry, tag, include_prerelease=include_prerelease)
            if reason is not None:
                skipped.append({"tag": tag, "reason": reason})
                continue
            tags.append(tag)
        if skipped:
            # Once, at debug: this is normal upstream traffic, not an incident,
            # but "why is b10488 not offered" needs an answer somewhere.
            log.debug("engine.releases.filtered", skipped=skipped, kept=len(tags))
        tags.sort(key=lambda t: build_number(t) or 0, reverse=True)
        return tags[:limit]

    async def check_update(self, *, limit: int = 5, probe_assets: int = 3) -> dict[str, Any]:
        """Is there a newer engine this box can actually install?

        Replaces the ``latest = releases[0]; latest != current`` check that was
        copy-pasted into the GUI, the ``/capabilities`` route and the CLI. That
        shape was wrong twice over: it offered whatever tag sorted first even
        when the release had no asset for this driver (the ``v0.1.2`` prerelease
        of 2026-08-18), and ``!=`` calls a *downgrade* an update -- ``b10441`` is
        installed here while ``b10425`` is active.

        So ``latest`` is the newest tag that has a **selectable** asset for this
        box, verified by actually reading its asset list. Only the newest
        ``probe_assets`` tags are checked, one GitHub call each: on release day a
        build can be tagged minutes before its Windows zips finish uploading, so
        the newest tag legitimately having no asset yet is a normal state that
        must not hide the perfectly good build underneath it.

        Returns ``checked``/``current``/``latest``/``update_available``/
        ``recent``/``latest_variant``/``skipped``. ``current`` is the **active**
        tag (``active.json`` wins over ``engine.pinned_tag`` -- see
        :meth:`check_pinned_tag`), falling back to the pin. A per-tag asset
        failure lands in ``skipped`` and never raises; only a failure to list
        releases at all raises :class:`EngineError`, so a caller can show it.
        """
        current = self._read_active() or self.config.engine.pinned_tag
        releases = await self.list_releases(limit)
        skipped: list[dict[str, str]] = []
        latest: str | None = None
        latest_variant: str | None = None

        gpus = self._gpus()
        driver = self._cuda_driver_version()
        for tag in releases[: max(1, probe_assets)]:
            try:
                assets = await self.list_assets(tag)
                asset = self.select_asset(assets, gpus=gpus, cuda_driver=driver)
            except EngineError as exc:
                skipped.append({"tag": tag, "reason": str(exc)})
                continue
            if asset is None:
                variants = sorted({a.variant for a in assets}) or ["<none>"]
                skipped.append(
                    {
                        "tag": tag,
                        "reason": (
                            f"no asset for {self.os_token}/{self.arch_token} is compatible "
                            f"with this driver (CUDA {_fmt_version(driver)}); "
                            f"available variants: {', '.join(variants)}"
                        ),
                    }
                )
                continue
            latest = tag
            latest_variant = asset.variant
            break

        if latest is None and releases and self.config.engine.allow_source_build:
            # Nothing prebuilt fits, but a source build is exactly the documented
            # fallback for that (D2/D3), and it clones by tag -- so the newest
            # release is installable after all, just slowly.
            latest = releases[0]
            latest_variant = "source"

        current_n = build_number(current)
        latest_n = build_number(latest)
        update_available = latest_n is not None and (current_n is None or latest_n > current_n)
        return {
            "checked": True,
            "current": current,
            "latest": latest,
            "update_available": update_available,
            "recent": list(releases),
            "latest_variant": latest_variant,
            "skipped": skipped,
        }

    async def list_assets(self, tag: str) -> list[EngineAsset]:
        """Release archives for ``tag``, parsed into :class:`EngineAsset`."""
        url = f"{GITHUB_API}/repos/{self.config.engine.repo}/releases/tags/{tag}"
        try:
            resp = await self.client.get(url)
            if resp.status_code == 404:
                raise EngineError(f"llama.cpp release '{tag}' does not exist upstream")
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            raise EngineError(f"could not fetch llama.cpp release '{tag}': {exc}") from exc
        raw = payload.get("assets") if isinstance(payload, dict) else None
        if not isinstance(raw, list):
            raise EngineError(f"release '{tag}' returned no asset list")
        return build_assets(tag, raw)

    def select_asset(
        self,
        assets: Sequence[EngineAsset],
        *,
        gpus: Sequence[GpuInfo],
        cuda_driver: tuple[int, int] | None,
    ) -> EngineAsset | None:
        """Pick the best prebuilt archive for this box, or ``None``.

        Eligibility rule (the one correctness rule that matters here): an asset
        built against CUDA *X.Y* runs only on a driver whose supported CUDA
        version is **>= X.Y**. CUDA has forward -- not backward -- minor-version
        compatibility, so a driver advertising CUDA 13.0 happily runs a 12.4
        build but cannot load a 13.3 one. We therefore keep assets where
        ``asset_cuda <= driver_cuda`` and take the highest of those.

        Returning ``None`` when nothing is eligible is deliberate. A CPU or
        Vulkan archive would "work" in the sense of starting, and then quietly
        violate the project's core GPU-only policy -- every VRAM estimate,
        eviction decision and rejection message assumes all layers are on the
        GPU. Better to hand ``ensure_engine`` a ``None`` so it can fall back to
        a source build against the actual compute capabilities.
        """
        matching = [
            a
            for a in assets
            if _os_matches(a.os_token, self.os_token) and a.arch == self.arch_token
        ]
        if not matching:
            return None

        requested = self.config.engine.cuda_variant.strip()
        if requested and requested.lower() != "auto":
            return self._select_explicit(matching, requested)

        # AMD box: ROCm is the only GPU-only option upstream ships for it, and a
        # CUDA build would be selected purely because it exists.
        if gpus and not any(_looks_nvidia(g.name) for g in gpus):
            rocm = [a for a in matching if a.variant.startswith("rocm")]
            if rocm:
                return max(rocm, key=lambda a: _parse_version(a.variant[5:]) or (0, 0))
            return None

        cuda = [a for a in matching if a.is_cuda]
        if cuda:
            eligible = [a for a in cuda if _cuda_eligible(a.cuda_version, cuda_driver)]
            if eligible:
                best = max(eligible, key=lambda a: a.cuda_version or (0, 0))
                log.debug(
                    "engine.select_asset",
                    chosen=best.name,
                    driver_cuda=cuda_driver,
                    considered=[a.variant for a in cuda],
                )
                return best
            log.info(
                "engine.select_asset.no_eligible_cuda",
                driver_cuda=cuda_driver,
                available=[a.variant for a in cuda],
            )

        return None

    def _select_explicit(self, matching: Sequence[EngineAsset], requested: str) -> EngineAsset:
        want = requested if not requested[0].isdigit() else f"cuda-{requested}"
        if want == "cpu":
            raise EngineError(
                "engine.cuda_variant='cpu' is not supported: StudioForge is GPU-only and "
                "every VRAM estimate assumes all layers are offloaded"
            )
        for asset in matching:
            if asset.variant == want:
                return asset
        available = sorted({a.variant for a in matching})
        raise EngineError(
            f"engine.cuda_variant={requested!r} matches no asset for "
            f"{self.os_token}/{self.arch_token}; available variants: {', '.join(available)}"
        )

    # ------------------------------------------------------------------
    # Local inventory
    # ------------------------------------------------------------------

    def installed(self) -> list[EngineInfo]:
        """Every engine directory that actually contains a server binary."""
        root = self.engines_dir
        if not root.is_dir():
            return []
        active_tag = self._read_active()
        found: list[EngineInfo] = []
        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or entry.name.startswith(("src-", ".", "_")):
                continue
            binary = find_server_binary(entry)
            if binary is None:
                continue
            found.append(self._info_for(entry, binary, active_tag))
        found.sort(key=lambda info: info.installed_at)
        return found

    def _info_for(self, directory: Path, binary: Path, active_tag: str | None) -> EngineInfo:
        meta: dict[str, Any] = {}
        meta_path = directory / META_FILE
        if meta_path.is_file():
            with contextlib.suppress(OSError, ValueError):
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    meta = loaded
        flags: list[str] = []
        flags_path = directory / FLAGS_FILE
        if flags_path.is_file():
            with contextlib.suppress(OSError):
                flags = sorted(
                    line.strip()
                    for line in flags_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        installed_at = meta.get("installed_at")
        if not isinstance(installed_at, int | float):
            with contextlib.suppress(OSError):
                installed_at = directory.stat().st_mtime
        build_log = meta.get("build_log")
        variant = str(meta.get("variant") or "")
        if not variant or variant == "unknown":
            variant = guess_variant(directory)
        return EngineInfo(
            tag=directory.name,
            path=directory,
            server_binary=binary,
            variant=variant,
            version_string=meta.get("version_string"),
            smoke_tested=bool(meta.get("smoke_tested", False)),
            smoke_tested_at=meta.get("smoke_tested_at"),
            active=directory.name == active_tag,
            installed_at=float(installed_at or time.time()),
            flags=flags,
            build_log=Path(build_log) if isinstance(build_log, str) else None,
        )

    def get(self, tag: str) -> EngineInfo | None:
        directory = self.engine_dir(tag)
        if not directory.is_dir():
            return None
        binary = find_server_binary(directory)
        if binary is None:
            return None
        return self._info_for(directory, binary, self._read_active())

    def active(self) -> EngineInfo | None:
        """The engine new loads should use.

        Falls back to the configured pinned tag, then to the newest install, so
        a missing/stale ``active.json`` degrades instead of breaking startup.
        """
        tag = self._read_active()
        if tag:
            info = self.get(tag)
            if info is not None:
                return info
        info = self.get(self.config.engine.pinned_tag)
        if info is not None:
            return info
        found = self.installed()
        return found[-1] if found else None

    def server_binary(self, tag: str | None = None) -> Path:
        """Absolute path to a specific engine's ``llama-server``."""
        if tag is None:
            info = self.active()
            if info is None:
                raise EngineError(
                    "no llama-server engine is installed, so nothing can be loaded. Install "
                    f"{self.config.engine.pinned_tag} from the Setup tab (llama.cpp engine -> "
                    "Install), or run `studioforge engine --update` "
                    f"(expected under {self.engines_dir})"
                )
            return info.server_binary
        directory = self.engine_dir(tag)
        binary = find_server_binary(directory)
        if binary is None:
            raise EngineError(
                f"engine '{tag}' is not installed (looked in {directory}). This model pins "
                f"engine_tag={tag!r}: install that build from the Setup tab, or clear the "
                f"model's engine_tag to use the active engine"
            )
        return binary

    def prune(self, keep: int | None = None) -> list[str]:
        """Delete old engine directories, newest ``keep`` retained.

        The active tag and the configured pinned tag are never removed however
        old they are -- pruning the engine currently in use, or the one config
        says to use, would break the next load rather than save disk.
        """
        limit = self.config.engine.keep_versions if keep is None else keep
        limit = max(0, limit)
        protected = {self.config.engine.pinned_tag}
        active_tag = self._read_active()
        if active_tag:
            protected.add(active_tag)
        newest_first = sorted(self.installed(), key=lambda i: i.installed_at, reverse=True)
        removed: list[str] = []
        for index, info in enumerate(newest_first):
            if index < limit or info.tag in protected:
                continue
            try:
                shutil.rmtree(info.path)
            except OSError as exc:  # pragma: no cover - locked files on Windows
                log.warning("engine.prune.failed", tag=info.tag, error=str(exc))
                continue
            self._flag_cache.pop(info.tag, None)
            self._removed_cache.pop(info.tag, None)
            self._help_cache.pop(info.tag, None)
            removed.append(info.tag)
        if removed:
            log.info("engine.pruned", tags=removed, keep=limit)
        return removed

    # -- active marker --------------------------------------------------

    def _read_active(self) -> str | None:
        path = self.engines_dir / ACTIVE_FILE
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        tag = data.get("tag") if isinstance(data, dict) else None
        return tag if isinstance(tag, str) else None

    def check_pinned_tag(self) -> str | None:
        """Warn when ``engine.pinned_tag`` disagrees with ``active.json``.

        ``active.json`` wins -- :meth:`active` reads it first, so every load
        runs whatever it names. That is correct (a rollback has to survive a
        config that still points at the bad build) and it is also invisible:
        config.yaml said ``b10441`` for weeks while every child launched
        ``b10425``, so anyone reading the config to work out which engine was
        running got the wrong answer, including us.

        Returns the active tag when the two disagree, else ``None``. Called at
        startup rather than on every :meth:`active` lookup, because a warning
        that repeats on a hot path stops being read.
        """
        active_tag = self._read_active()
        pinned = self.config.engine.pinned_tag
        if not active_tag or active_tag == pinned:
            return None
        log.warning(
            "engine_tag_drift",
            pinned_tag=pinned,
            active_tag=active_tag,
            detail=(
                f"config.yaml pins engine {pinned} but engines/active.json says "
                f"{active_tag}, and active.json wins -- every load runs {active_tag}. "
                f"Set engine.pinned_tag to {active_tag} if the rollback was "
                f"deliberate, or re-activate {pinned} if it was not."
            ),
        )
        return active_tag

    def set_active(self, tag: str) -> None:
        """Persist the active tag as ``engines/active.json``.

        Not the SQLite registry: the engine must be resolvable during first-run
        bootstrap, before a database exists, and by the watchdog process.
        """
        self.engines_dir.mkdir(parents=True, exist_ok=True)
        path = self.engines_dir / ACTIVE_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"tag": tag, "updated_at": time.time()}, indent=2), encoding="utf-8"
        )
        tmp.replace(path)

    def _write_meta(self, info: EngineInfo) -> None:
        payload = {
            "tag": info.tag,
            "variant": info.variant,
            "version_string": info.version_string,
            "smoke_tested": info.smoke_tested,
            "smoke_tested_at": info.smoke_tested_at,
            "installed_at": info.installed_at,
            "server_binary": str(info.server_binary),
            "build_log": str(info.build_log) if info.build_log else None,
        }
        with contextlib.suppress(OSError):
            (info.path / META_FILE).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    async def install(
        self,
        tag: str,
        *,
        progress: ProgressFn | None = None,
        force: bool = False,
    ) -> EngineInfo:
        """Download, extract and verify the best prebuilt engine for ``tag``.

        Serialised per tag (see ``_install_locks``): a second concurrent call
        waits, then finds the engine present and returns it.
        """
        async with self._install_lock(tag):
            return await self._install_locked(tag, progress=progress, force=force)

    async def _install_locked(
        self,
        tag: str,
        *,
        progress: ProgressFn | None,
        force: bool,
    ) -> EngineInfo:
        dest = self.engine_dir(tag)
        existing = find_server_binary(dest)
        if existing is not None and not force:
            smoke = await self._smoke(tag, None)
            if smoke.ok or smoke.no_model:
                log.info("engine.install.already_present", tag=tag, smoke_ok=smoke.ok)
                _emit(progress, "done", 1.0)
                return self._finalize(dest, existing, "prebuilt", smoke, activate=True)
            log.warning("engine.install.reinstall_after_failed_smoke", tag=tag, detail=smoke.detail)

        assets = await self.list_assets(tag)
        asset = self.select_asset(
            assets, gpus=self._gpus(), cuda_driver=self._cuda_driver_version()
        )
        if asset is None:
            variants = sorted({a.variant for a in assets}) or ["<none>"]
            raise EngineError(
                f"no GPU-capable llama-server build for {self.os_token}/{self.arch_token} at "
                f"{tag} is compatible with this driver "
                f"(CUDA {_fmt_version(self._cuda_driver_version())}); "
                f"available variants: {', '.join(variants)}"
            )

        self.config.downloads_dir.mkdir(parents=True, exist_ok=True)
        archive = self.config.downloads_dir / asset.name
        await self._download(asset.url, archive, asset.size_bytes, "download", progress)
        _verify_zip(archive)

        if force and dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        _emit(progress, "extract", 0.0)
        binary = extract_engine_zip(archive, dest)
        _emit(progress, "extract", 1.0)

        if asset.needs_cudart and asset.cudart_url:
            cudart = self.config.downloads_dir / Path(asset.cudart_url).name
            await self._download(asset.cudart_url, cudart, 0, "cudart", progress)
            _verify_zip(cudart)
            extract_engine_zip(cudart, dest)

        _make_executable(dest)
        if binary is None:
            binary = find_server_binary(dest)
        if binary is None:
            raise EngineError(f"{asset.name} contained no {BIN_NAME}")

        _emit(progress, "smoke", 0.0)
        smoke = await self._smoke(tag, None)
        _emit(progress, "smoke", 1.0)
        if not smoke.ok and not smoke.no_model:
            raise EngineError(
                f"engine {tag} ({asset.variant}) failed its smoke test: {smoke.detail}"
            )
        if smoke.no_model:
            log.warning("engine.install.smoke_partial", tag=tag, detail=smoke.detail)
        _emit(progress, "done", 1.0)
        return self._finalize(dest, binary, asset.variant, smoke, activate=True)

    def _finalize(
        self,
        dest: Path,
        binary: Path,
        variant: str,
        smoke: _SmokeResult,
        *,
        activate: bool,
        build_log: Path | None = None,
    ) -> EngineInfo:
        info = EngineInfo(
            tag=dest.name,
            path=dest,
            server_binary=binary,
            variant=variant,
            version_string=smoke.version_string,
            smoke_tested=smoke.ok,
            smoke_tested_at=time.time() if smoke.ok else None,
            active=activate,
            build_log=build_log,
        )
        self._write_meta(info)
        if activate:
            self.set_active(info.tag)
        return info

    async def _download(
        self,
        url: str,
        target: Path,
        expected_bytes: int,
        phase: str,
        progress: ProgressFn | None,
    ) -> Path:
        """Stream ``url`` to ``target``, reusing a complete previous download."""
        if _reuse_completed_download(target, expected_bytes):
            log.info("engine.download.cached", name=target.name, bytes=expected_bytes)
            _emit(progress, phase, 1.0)
            return target

        tmp = target.with_suffix(target.suffix + ".part")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        done = 0
        try:
            async with self.client.stream("GET", url) as resp:
                resp.raise_for_status()
                header = resp.headers.get("content-length")
                total = int(header) if header and header.isdigit() else expected_bytes
                with tmp.open("wb") as handle:
                    async for chunk in resp.aiter_bytes(1024 * 1024):
                        handle.write(chunk)
                        done += len(chunk)
                        if total:
                            _emit(progress, phase, done / total)
        except httpx.HTTPError as exc:
            tmp.unlink(missing_ok=True)
            raise EngineError(f"download of {url} failed: {exc}") from exc
        if expected_bytes and done != expected_bytes:
            tmp.unlink(missing_ok=True)
            raise EngineError(
                f"download of {target.name} is {done} bytes, expected {expected_bytes}"
            )
        tmp.replace(target)
        _emit(progress, phase, 1.0)
        return target

    # ------------------------------------------------------------------
    # Source build
    # ------------------------------------------------------------------

    def cuda_arches(self, gpus: Sequence[GpuInfo] | None = None) -> list[str]:
        """CMake ``CMAKE_CUDA_ARCHITECTURES`` values for the detected GPUs.

        Deduped and sorted numerically, so a mixed 2x5090 (sm_120) + 2x3090
        (sm_86) box yields ``["86", "120"]`` -- string sorting would emit
        ``120;86`` and produce a build missing the newer arch's fast paths.
        """
        pool = list(gpus) if gpus is not None else self._gpus()
        arches = {gpu.sm_arch for gpu in pool if gpu.sm_arch}
        return sorted(arches, key=lambda a: int(a) if a.isdigit() else 0)

    async def build_from_source(
        self,
        tag: str,
        *,
        arches: Sequence[str],
        progress: ProgressFn | None = None,
    ) -> EngineInfo:
        """Build ``llama-server`` from source for exactly this box's arches.

        The fallback path for a GPU too new for any prebuilt archive (or a
        driver too old for the newest CUDA build). Installs to
        ``engines/<tag>-local/`` so it never collides with a prebuilt ``<tag>``.
        """
        arch_list = (
            self.cuda_arches([])
            if not arches
            else sorted({a for a in arches if a}, key=lambda a: int(a) if a.isdigit() else 0)
        )
        if not arch_list:
            raise EngineError(
                "cannot build llama.cpp from source: no CUDA compute capability was "
                "detected, so CMAKE_CUDA_ARCHITECTURES would be empty"
            )

        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.config.logs_dir / f"engine-build-{tag}.log"
        with contextlib.suppress(OSError):
            log_path.write_text("", encoding="utf-8")

        src = await self._prepare_source(tag, log_path, progress)
        build_dir = src / "build-studioforge"
        dest = self.engine_dir(f"{tag}-local")

        configure = [
            "cmake",
            "-S",
            str(src),
            "-B",
            str(build_dir),
            "-DGGML_CUDA=ON",
            f"-DCMAKE_CUDA_ARCHITECTURES={';'.join(arch_list)}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_BUILD_SERVER=ON",
        ]
        code = await self._run_logged(
            configure, cwd=src, log_path=log_path, phase="configure", progress=progress
        )
        if code != 0:
            raise EngineError(f"cmake configure failed (exit {code}); see {log_path}")

        build = [
            "cmake",
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "llama-server",
            "--parallel",
            str(os.cpu_count() or 4),
        ]
        code = await self._run_logged(
            build, cwd=src, log_path=log_path, phase="build", progress=progress
        )
        if code != 0:
            raise EngineError(f"cmake build failed (exit {code}); see {log_path}")

        binary = self._install_built_binaries(build_dir, dest)
        _make_executable(dest)
        smoke = await self._smoke(dest.name, None)
        if not smoke.ok and not smoke.no_model:
            raise EngineError(
                f"source-built engine {dest.name} failed its smoke test: {smoke.detail}"
            )
        return self._finalize(
            dest, binary, "source-local", smoke, activate=True, build_log=log_path
        )

    async def _prepare_source(self, tag: str, log_path: Path, progress: ProgressFn | None) -> Path:
        vendor = self.vendor_dir
        if vendor.is_dir() and self._git_at_tag(vendor, tag):
            log.info("engine.build.reuse_vendor", path=str(vendor), tag=tag)
            return vendor
        src = self.engines_dir / f"src-{tag}"
        if not (src / "CMakeLists.txt").is_file():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
            clone = [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                tag,
                f"https://github.com/{self.config.engine.repo}.git",
                str(src),
            ]
            code = await self._run_logged(
                clone, cwd=None, log_path=log_path, phase="clone", progress=progress
            )
            if code != 0:
                raise EngineError(f"git clone of {tag} failed (exit {code}); see {log_path}")
        return src

    @staticmethod
    def _git_at_tag(repo: Path, tag: str) -> bool:
        try:
            result = subprocess.run(  # noqa: S603 - fixed argv, no shell
                ["git", "-C", str(repo), "describe", "--tags", "--exact-match", "HEAD"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and result.stdout.strip() == tag

    def _install_built_binaries(self, build_dir: Path, dest: Path) -> Path:
        """Copy the produced binaries/libraries out of the CMake build tree."""
        candidates = [build_dir / "bin" / "Release", build_dir / "bin", build_dir / "Release"]
        source = next((c for c in candidates if c.is_dir()), None)
        if source is None:
            raise EngineError(f"build produced no bin directory under {build_dir}")
        dest.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)
            elif item.is_dir():
                shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        binary = find_server_binary(dest)
        if binary is None:
            raise EngineError(f"build produced no {BIN_NAME} in {source}")
        return binary

    async def _run_logged(
        self,
        cmd: Sequence[str],
        *,
        cwd: Path | None,
        log_path: Path,
        phase: str,
        progress: ProgressFn | None,
    ) -> int:
        """Run ``cmd``, tee-ing output to ``log_path`` and parsing ``[ NN%]``."""
        log.info("engine.build.run", phase=phase, cmd=list(cmd))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **_spawn_kwargs(),
        )
        with log_path.open("a", encoding="utf-8", errors="replace") as sink:
            sink.write(f"\n$ {' '.join(cmd)}\n")
            assert proc.stdout is not None
            while True:
                raw = await proc.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").rstrip()
                sink.write(line + "\n")
                match = _PERCENT_RE.search(line)
                if match:
                    _emit(progress, phase, int(match.group(1)) / 100.0)
        return await proc.wait()

    # ------------------------------------------------------------------
    # First-run resolution
    # ------------------------------------------------------------------

    async def ensure_engine(self, *, progress: ProgressFn | None = None) -> EngineInfo:
        """Guarantee a usable engine exists, installing or building if needed.

        **An installed engine is reused, never reinstalled, from here** (D27).
        This runs at every boot, before the API port is bound, so it has to
        be fast and it has to be deterministic. It used to run the full smoke
        test -- ``--version`` plus a real GPU micro-load -- and treat a failed
        micro-load as "the install is bad": it then went to GitHub, called
        :meth:`install` on the *same* tag, which ran the same micro-load
        again, failed again, and re-downloaded a 600 MB archive over a working
        engine, all with the port unbound. Every one of the ways that
        micro-load fails at boot -- every GPU full because ComfyUI is
        training, a corrupt or half-downloaded tiny model, a driver too old
        for the build -- is a condition a reinstall of the same archive cannot
        change, so the reinstall bought minutes of dead air and nothing else.

        The rule now: ``--version`` must run (a half-extracted zip or missing
        DLLs fail here, and *that* is a broken install worth replacing). The
        micro-load runs only for a build that has never passed one; if it
        fails, the engine is still activated with one WARNING carrying the
        detail, and the first real load reports the real error with the
        child's stderr tail. Reinstalling is an explicit act (Setup tab,
        ``engine --update``, ``install(force=True)``), not a boot side effect.
        """
        self.check_pinned_tag()
        tag = self.config.engine.pinned_tag
        for candidate in (tag, f"{tag}-local"):
            info = self.get(candidate)
            if info is None:
                continue
            smoke = await self._boot_check(candidate, info)
            if not smoke.version_ok:
                log.warning(
                    "engine.ensure.broken_install",
                    tag=candidate,
                    detail=smoke.detail,
                    action="reinstalling: the binary does not even run --version",
                )
                continue
            self.set_active(candidate)
            info.active = True
            if smoke.ok:
                info.smoke_tested = True
                info.smoke_tested_at = time.time()
            info.version_string = smoke.version_string or info.version_string
            self._write_meta(info)
            self._warn_driver_too_old(info)
            # Warm engines/<tag>/features.json while we are still before the
            # port bind. The supervisor reads that file synchronously on the
            # load path; without it the first load would either pay for a
            # ``--help`` run or fall back to "advertises nothing" and quietly
            # drop every optional flag (see EngineFeatures).
            with contextlib.suppress(Exception):
                features = await self.engine_features(candidate)
                log.info(
                    "engine.features",
                    tag=candidate,
                    known=features.known,
                    split_modes=list(features.split_modes),
                    spec_types=list(features.spec_types),
                )
            if smoke.ok or smoke.no_model or info.smoke_tested:
                log.info("engine.ensure.reused", tag=candidate, detail=smoke.detail)
            else:
                log.warning(
                    "engine.ensure.smoke_failed",
                    tag=candidate,
                    detail=smoke.detail,
                    action=(
                        "kept as the active engine: the binary runs, and a reinstall of the "
                        "same build cannot fix a failed micro-load. The first real load will "
                        "report the actual error; `studioforge engine --smoke-test` re-runs "
                        "the check by hand"
                    ),
                )
            return info

        assets: list[EngineAsset] = []
        error: str | None = None
        try:
            assets = await self.list_assets(tag)
        except EngineError as exc:
            error = str(exc)
        asset = (
            self.select_asset(assets, gpus=self._gpus(), cuda_driver=self._cuda_driver_version())
            if assets
            else None
        )
        if asset is not None:
            return await self.install(tag, progress=progress)

        if self.config.engine.allow_source_build:
            log.info("engine.ensure.source_build", tag=tag)
            return await self.build_from_source(tag, arches=self.cuda_arches(), progress=progress)

        gpus = self._gpus()
        detected = ", ".join(f"{g.name} (sm_{g.sm_arch or '?'})" for g in gpus) or "no GPUs"
        variants = sorted({a.variant for a in assets}) or ["<none>"]
        raise EngineError(
            f"no usable llama-server engine for {tag}: detected {detected} with driver CUDA "
            f"{_fmt_version(self._cuda_driver_version())} on {self.os_token}/{self.arch_token}; "
            f"upstream offered {', '.join(variants)}; engine.allow_source_build is disabled"
            + (f" ({error})" if error else "")
        )

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    async def version_string(self, tag: str) -> str | None:
        """The engine's own version line, e.g. ``version: ... (build 10425, ...)``."""
        code, text = await self._capture(self.server_binary(tag), ["--version"])
        if code != 0 and not text.strip():
            return None
        found = _version_line(text)
        if found is not None:
            return found
        return next((line.strip() for line in text.splitlines() if line.strip()), None)

    async def list_devices(self, tag: str) -> list[str]:
        """Devices the engine can actually see (``--list-devices``)."""
        code, text = await self._capture(self.server_binary(tag), ["--list-devices"])
        devices: list[str] = []
        seen_header = False
        for line in text.splitlines():
            if "available devices" in line.lower():
                seen_header = True
                continue
            if not seen_header:
                continue
            match = _DEVICE_RE.match(line)
            if match and line.startswith((" ", "\t")):
                devices.append(line.strip())
        if not devices and code != 0:
            log.warning("engine.list_devices.failed", tag=tag, exit_code=code)
        return devices

    async def smoke_test(self, tag: str, *, tiny_model: Path | None = None) -> tuple[bool, str]:
        """Verify an engine really runs: ``--version`` plus a real micro-load.

        A version check alone proves only that the DLLs resolve. The micro-load
        proves CUDA initialises, the arch is supported by this build, and the
        HTTP server reaches ``/health: ok`` -- which is the actual contract the
        gateway depends on. On failure the detail carries the tail of the
        child's stderr, without which a failure report is undiagnosable.
        """
        result = await self._smoke(tag, tiny_model)
        return result.ok, result.detail

    async def _boot_check(self, tag: str, info: EngineInfo) -> _SmokeResult:
        """The startup verification of an installed engine (see :meth:`ensure_engine`).

        ``--version`` always: it is cheap, touches no GPU, and is what fails
        for a genuinely broken install (missing DLLs, a half-extracted
        archive). The GPU micro-load only for a build that has never passed
        one -- a build that passed on this box is trusted, because the
        micro-load fails at boot for reasons that have nothing to do with the
        install (every GPU full, a corrupt tiny model, a driver change) and
        each of those is reported far more usefully by the first real load.
        """
        binary = info.server_binary
        code, text = await self._capture(binary, ["--version"])
        version = _version_line(text) if code == 0 else None
        if version is None:
            tail = "\n".join(text.splitlines()[-40:])
            return _SmokeResult(
                False,
                f"'{binary.name} --version' exited {code} without a parsable version line. "
                f"Output tail:\n{tail}",
            )
        if info.smoke_tested:
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.localtime(info.smoke_tested_at))
                if info.smoke_tested_at
                else "earlier"
            )
            return _SmokeResult(
                True,
                f"engine {tag}: {version}; micro-load passed {when}, not repeated at boot",
                version_ok=True,
                version_string=version,
            )
        return await self._smoke(tag, None)

    def _warn_driver_too_old(self, info: EngineInfo) -> str | None:
        """One WARNING when the driver cannot run the installed CUDA build.

        A driver downgrade after install (or an install copied from another
        box) is otherwise diagnosed only by the first load, as an opaque
        ``cuda error`` in a stderr tail. The comparison is the same one
        :meth:`select_asset` uses to *choose* a build (``_cuda_eligible``): a
        ``cuda-13.3`` binary needs a driver advertising CUDA 13.3 or newer.
        Returns the warning text, or ``None`` when the driver is fine.
        """
        needed = (
            _parse_version(info.variant[len("cuda-") :])
            if info.variant.startswith("cuda-")
            else None
        )
        driver = self._cuda_driver_version()
        if needed is None or driver is None or _cuda_eligible(needed, driver):
            return None
        detail = (
            f"engine {info.tag} is a {info.variant} build but this driver only "
            f"advertises CUDA {_fmt_version(driver)}; loads will fail to initialise "
            f"CUDA. Update the NVIDIA driver, or set engine.cuda_variant to a build "
            f"this driver can run and reinstall the engine (Setup tab, or "
            f"`studioforge engine --update`)."
        )
        log.warning(
            "engine.driver_too_old",
            tag=info.tag,
            variant=info.variant,
            driver_cuda=_fmt_version(driver),
            detail=detail,
        )
        return detail

    async def _smoke(self, tag: str, tiny_model: Path | None) -> _SmokeResult:
        try:
            binary = self.server_binary(tag)
        except EngineError as exc:
            return _SmokeResult(False, str(exc))

        code, text = await self._capture(binary, ["--version"])
        version = _version_line(text)
        if code != 0 or version is None:
            tail = "\n".join(text.splitlines()[-40:])
            return _SmokeResult(
                False,
                f"'{binary.name} --version' exited {code} without a parsable version line. "
                f"Output tail:\n{tail}",
            )

        model = tiny_model or self.find_tiny_model()
        if model is None:
            return _SmokeResult(
                False,
                f"engine {tag} reports '{version}' but no GGUF model was available for the "
                "micro-load, so only the --version check ran (this is NOT a full pass)",
                version_ok=True,
                version_string=version,
                no_model=True,
            )

        ok, detail = await self._micro_load(binary, model)
        if ok:
            return _SmokeResult(
                True,
                f"engine {tag}: {version}; micro-load of {model.name} reached /health ok",
                version_ok=True,
                version_string=version,
                model_used=model,
            )
        return _SmokeResult(
            False,
            f"engine {tag}: {version}; micro-load of {model.name} failed: {detail}",
            version_ok=True,
            version_string=version,
            model_used=model,
        )

    async def _micro_load(self, binary: Path, model: Path) -> tuple[bool, str]:
        port = _free_port()
        cmd = [
            str(binary),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            "512",
            "--n-gpu-layers",
            "999",
            # Same GPU-only invariant as a real launch (DECISIONS.md D11).
            "--fit",
            "off",
            "--no-webui",
        ]
        timeout = max(10.0, float(self.config.engine.smoke_test_timeout_s))
        stderr_tail: deque[str] = deque(maxlen=200)
        stdout_tail: deque[str] = deque(maxlen=200)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(binary.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_spawn_kwargs(),
            )
        except OSError as exc:
            return False, f"could not launch {binary}: {exc}"

        readers = [
            asyncio.create_task(_drain(proc.stdout, stdout_tail)),
            asyncio.create_task(_drain(proc.stderr, stderr_tail)),
        ]
        healthy = False
        reason = "timed out"
        deadline = time.monotonic() + timeout
        try:
            async with httpx.AsyncClient(timeout=5.0) as probe:
                while time.monotonic() < deadline:
                    if proc.returncode is not None:
                        reason = f"process exited with code {proc.returncode} before /health ok"
                        break
                    try:
                        resp = await probe.get(f"http://127.0.0.1:{port}/health")
                        if resp.status_code == 200 and resp.json().get("status") == "ok":
                            healthy = True
                            break
                    except (httpx.HTTPError, ValueError):
                        pass
                    await asyncio.sleep(0.5)
                else:
                    reason = f"never reported /health ok within {timeout:.0f}s"
        finally:
            try:
                if proc.returncode is None:
                    await asyncio.to_thread(kill_process_tree, proc.pid)
                with contextlib.suppress(asyncio.TimeoutError, ProcessLookupError):
                    await asyncio.wait_for(proc.wait(), timeout=15)
                for task in readers:
                    task.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
            except asyncio.CancelledError:
                # Cancellation can land on the cleanup's own first await (a
                # client aborting the smoke test, Ctrl-C during startup). The
                # child was spawned in its own process group, so abandoning it
                # here would leave a llama-server resident with a full CUDA
                # context -- permanently leaked VRAM. Finish the kill
                # synchronously; a short block on a dying path beats the leak.
                if proc.returncode is None:
                    with contextlib.suppress(Exception):
                        kill_process_tree(proc.pid, timeout=5.0)
                for task in readers:
                    task.cancel()
                raise

        if healthy:
            return True, "ok"
        tail = list(stderr_tail)[-40:] or list(stdout_tail)[-40:]
        return False, f"{reason}. Last stderr lines:\n" + "\n".join(tail)

    def find_tiny_model(self) -> Path | None:
        """A small GGUF suitable for a micro-load, or ``None``.

        Prefers a known 0.5B model, else the smallest plain GGUF in the library
        under ~2 GiB. mmproj/adapter files are skipped: they cannot be loaded
        with ``--model`` at all, so picking one would fail the smoke test for a
        reason that has nothing to do with the engine.
        """
        for directory in self.config.model_dirs():
            for relative in _PREFERRED_TINY_MODELS:
                candidate = directory / relative
                if candidate.is_file():
                    return candidate
        best: tuple[int, Path] | None = None
        for directory in self.config.model_dirs():
            if not directory.is_dir():
                continue
            try:
                entries = list(directory.rglob("*.gguf"))
            except OSError:  # pragma: no cover - unreadable share
                continue
            for path in entries:
                name = path.name.lower()
                if "mmproj" in name or "lora" in name or "adapter" in name:
                    continue
                if "-00002-of-" in name or "-of-0" in name and "-00001-of-" not in name:
                    continue
                try:
                    size = path.stat().st_size
                except OSError:
                    continue
                if size == 0 or size > _TINY_MODEL_MAX_BYTES:
                    continue
                if best is None or size < best[0]:
                    best = (size, path)
        return best[1] if best else None

    async def _capture(self, binary: Path, args: Sequence[str]) -> tuple[int, str]:
        """Run the engine with ``args`` and return ``(exit code, stdout+stderr)``."""
        try:
            proc = await asyncio.create_subprocess_exec(
                str(binary),
                *args,
                cwd=str(binary.parent),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **_spawn_kwargs(),
            )
        except OSError as exc:
            return 127, f"could not launch {binary}: {exc}"
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
        except TimeoutError:
            await asyncio.to_thread(kill_process_tree, proc.pid)
            return 124, f"'{binary.name} {' '.join(args)}' timed out"
        text = out.decode("utf-8", "replace") + err.decode("utf-8", "replace")
        return proc.returncode if proc.returncode is not None else 0, text

    # ------------------------------------------------------------------
    # Flag surface
    # ------------------------------------------------------------------

    async def help_text(self, tag: str) -> str:
        """The engine's ``--help`` output, cached in memory and on disk."""
        cached = self._help_cache.get(tag)
        if cached is not None:
            return cached
        directory = self.engine_dir(tag)
        path = directory / HELP_FILE
        if path.is_file():
            with contextlib.suppress(OSError):
                text = path.read_text(encoding="utf-8")
                if text.strip():
                    self._help_cache[tag] = text
                    return text
        _code, text = await self._capture(self.server_binary(tag), ["--help"])
        if not text.strip():
            raise EngineError(f"engine '{tag}' produced no --help output")
        self._help_cache[tag] = text
        with contextlib.suppress(OSError):
            directory.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return text

    async def supported_flags(self, tag: str) -> set[str]:
        """Flags this engine actually honours.

        Flags the engine declares but reports as removed are excluded: they are
        parsed and ignored at runtime, so treating them as supported is exactly
        the failure mode this validation exists to prevent. Cached to
        ``engines/<tag>/flags.txt`` so a restart does not re-run the binary.
        """
        cached = self._flag_cache.get(tag)
        if cached is not None:
            return set(cached)

        directory = self.engine_dir(tag)
        flags_path = directory / FLAGS_FILE
        if flags_path.is_file():
            with contextlib.suppress(OSError):
                lines = {
                    line.strip()
                    for line in flags_path.read_text(encoding="utf-8").splitlines()
                    if line.strip().startswith("-")
                }
                if lines:
                    self._flag_cache[tag] = lines
                    return set(lines)

        text = await self.help_text(tag)
        removed = removed_flags_from_help(text)
        flags = flags_from_help(text) - set(removed)
        self._flag_cache[tag] = flags
        self._removed_cache[tag] = self._merge_removed(removed, flags_from_help(text))
        with contextlib.suppress(OSError):
            directory.mkdir(parents=True, exist_ok=True)
            flags_path.write_text("\n".join(sorted(flags)) + "\n", encoding="utf-8")
        return set(flags)

    @staticmethod
    def _merge_removed(
        detected: dict[str, str | None], declared: set[str]
    ) -> dict[str, str | None]:
        """Combine engine-reported removals with the hardcoded fallback table.

        A hardcoded entry is only applied when the engine does not declare the
        flag at all. In ``b10425`` several old spellings survive as *live
        aliases* (``--cache-type-k-draft`` -> ``--spec-draft-type-k``); those
        still work, so rejecting them would be a false positive.
        """
        merged: dict[str, str | None] = {}
        for flag, replacement in detected.items():
            merged[flag] = replacement or REMOVED_FLAG_HINTS.get(flag)
        for flag, replacement in REMOVED_FLAG_HINTS.items():
            if flag not in merged and flag not in declared:
                merged[flag] = replacement
        return merged

    async def engine_features(self, tag: str | None = None) -> EngineFeatures:
        """The optional-feature surface of ``tag``, cached on disk and in memory.

        The cache file lives next to the binary (``engines/<tag>/features.json``)
        so the supervisor can read it synchronously on the load path without
        importing this manager -- see :func:`probe_engine_features`.
        """
        resolved = tag or self.config.engine.pinned_tag
        cached = self._features_cache.get(resolved)
        if cached is not None:
            return cached
        directory = self.engine_dir(resolved)
        on_disk = read_features_file(directory, resolved)
        if on_disk is not None:
            self._features_cache[resolved] = on_disk
            return on_disk
        try:
            text = await self.help_text(resolved)
        except EngineError as exc:
            log.warning("engine.features.unavailable", tag=resolved, error=str(exc))
            return EngineFeatures.unknown(resolved)
        features = parse_engine_features(text, resolved)
        if features.known:
            write_features_file(directory, features)
        self._features_cache[resolved] = features
        return features

    async def removed_flags(self, tag: str) -> dict[str, str | None]:
        """Flags this engine no longer honours, mapped to their replacement."""
        cached = self._removed_cache.get(tag)
        if cached is not None:
            return dict(cached)
        text = await self.help_text(tag)
        merged = self._merge_removed(removed_flags_from_help(text), flags_from_help(text))
        self._removed_cache[tag] = merged
        return dict(merged)

    async def validate_extra_flags(self, tag: str, extra: str) -> list[str]:
        """Validate the expert tier's free-text flag string. ``[]`` means ok.

        Validated at *save* time against the pinned engine's own ``--help``, so
        a typo or a renamed flag is a form error instead of a load failure --
        or worse, a flag llama-server accepts and ignores.
        """
        errors: list[str] = []
        if not extra.strip():
            return errors
        if "\n" in extra or "\r" in extra:
            errors.append("extra flags must be a single line")
            extra = extra.replace("\n", " ").replace("\r", " ")

        try:
            tokens = shlex.split(extra, posix=(os.name != "nt"))
        except ValueError as exc:
            return [*errors, f"could not parse extra flags ({exc}); check your quoting"]

        try:
            supported = await self.supported_flags(tag)
            removed = await self.removed_flags(tag)
        except EngineError as exc:
            return [*errors, f"cannot validate flags: {exc}"]

        previous = ""
        for token in tokens:
            base = token.split("=", 1)[0]
            relaxed = previous in _RELAXED_VALUE_FLAGS or base in _RELAXED_VALUE_FLAGS
            banned = _RELAXED_METACHARS if relaxed else _SHELL_METACHARS
            bad = sorted(set(token) & banned)
            if bad:
                errors.append(
                    f"illegal shell character(s) {' '.join(repr(c) for c in bad)} in "
                    f"{token!r}: extra flags are passed straight to llama-server, not to a shell"
                )
            previous = base

            if not _FLAG_START_RE.match(base) or not re.match(r"^-{1,2}[A-Za-z]", base):
                continue  # a value (including negative numbers like -1)
            if base in MANAGED_FLAGS:
                canonical = MANAGED_FLAGS[base]
                alias = "" if canonical == base else f" ({canonical})"
                errors.append(
                    f"'{base}'{alias} is managed by StudioForge and cannot be set in extra "
                    "flags: the manager assigns the model, alias, port, host and full GPU "
                    "offload for every instance"
                )
            elif base in removed:
                hint = removed[base]
                suffix = f"; try {hint}" if hint else ""
                errors.append(
                    f"unknown flag '{base}' for engine {tag} (removed in this release{suffix})"
                )
            elif base not in supported:
                errors.append(f"unknown flag '{base}' for engine {tag}")
        return errors

    # ------------------------------------------------------------------
    # GPU facts (lazy: core.gpu is optional at import time)
    # ------------------------------------------------------------------

    def _gpus(self) -> list[GpuInfo]:
        """Detected GPUs, degrading to ``[]`` rather than failing.

        ``core.gpu`` is imported lazily so this module stays importable (and
        testable) without it, and an injected probe short-circuits detection
        entirely.
        """
        probe = self._probe
        if probe is None:
            try:
                from studioforge.core.gpu import get_probe
            except Exception:  # pragma: no cover - module may not exist yet
                return _nvidia_smi_gpus()
            with contextlib.suppress(Exception):
                probe = get_probe()
                self._probe = probe
        if probe is None:  # pragma: no cover - construction failed
            return _nvidia_smi_gpus()
        for attr in ("list_gpus", "gpus", "snapshot", "detect"):
            method = getattr(probe, attr, None)
            if callable(method):
                try:
                    result = method()
                except Exception:  # pragma: no cover - probe failure is non-fatal
                    continue
                gpus = [g for g in _as_iterable(result) if isinstance(g, GpuInfo)]
                if gpus:
                    return gpus
        return _nvidia_smi_gpus()

    def _cuda_driver_version(self) -> tuple[int, int] | None:
        """Highest CUDA version this driver can run, e.g. ``(13, 3)``."""
        probe = self._probe
        if probe is not None:
            for attr in ("cuda_driver_version", "driver_cuda_version", "cuda_version"):
                value = getattr(probe, attr, None)
                if value is None:
                    continue
                try:
                    resolved = value() if callable(value) else value
                except Exception:  # pragma: no cover
                    continue
                parsed = _coerce_version(resolved)
                if parsed is not None:
                    return parsed
        return _detect_cuda_driver_version()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _as_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, list | tuple | set):
        return value
    gpus = getattr(value, "gpus", None)
    if isinstance(gpus, list | tuple):
        return gpus
    return ()


def _coerce_version(value: Any) -> tuple[int, int] | None:
    if isinstance(value, tuple) and len(value) >= 2:
        with contextlib.suppress(TypeError, ValueError):
            return int(value[0]), int(value[1])
    if isinstance(value, str):
        return _parse_version(value)
    if isinstance(value, int):
        # pynvml style: 13030 -> (13, 3)
        return value // 1000, (value % 1000) // 10
    return None


def _release_skip_reason(
    entry: dict[str, Any], tag: str, *, include_prerelease: bool
) -> str | None:
    """Why ``entry`` is not an installable engine release, or ``None``.

    Split out so the reason is a string both the debug log and a test can read,
    rather than a boolean that says a tag was dropped without saying why.
    """
    if entry.get("draft") is True:
        return "draft release (unpublished; its assets 404)"
    if entry.get("prerelease") is True and not include_prerelease:
        return "prerelease"
    if not ENGINE_TAG_RE.match(tag):
        return "tag is not a bNNNN llama.cpp build release"
    return None


def _os_matches(asset_os: str, host_os: str) -> bool:
    if asset_os == host_os:
        return True
    return {asset_os, host_os} <= {"ubuntu", "linux"}


def _looks_nvidia(name: str) -> bool:
    lowered = name.lower()
    return "nvidia" in lowered or "geforce" in lowered or "rtx" in lowered or "tesla" in lowered


def _cuda_eligible(asset_cuda: tuple[int, int] | None, driver: tuple[int, int] | None) -> bool:
    """Whether a CUDA build can run on this driver.

    ``asset_cuda <= driver`` -- and not the reverse. CUDA is forward compatible
    within a major version: a driver reporting 13.0 runs binaries built against
    12.4 or 13.0, but a 13.3 binary needs a 13.3-capable driver. Inverting this
    comparison yields an engine that fails to initialise CUDA at load time with
    an opaque error, which is why it is spelled out here.
    """
    if asset_cuda is None:
        # Unversioned CUDA asset (Linux naming): cannot be checked, so allow it
        # and let the smoke test be the gate.
        return True
    if driver is None:
        return True
    return asset_cuda <= driver


def _fmt_version(version: tuple[int, int] | None) -> str:
    return "unknown" if version is None else f"{version[0]}.{version[1]}"


def _version_line(text: str) -> str | None:
    """The ``version: ... (build NNNN, ...)`` line out of ``--version`` output."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("version:") or _BUILD_RE.search(stripped):
            return stripped
    return None


def _reuse_completed_download(target: Path, expected_bytes: int) -> bool:
    """Whether an earlier download of ``target`` is complete and intact.

    Resume-by-verify: the API gives a size, so a byte-exact file that also opens
    as a zip is trusted. A truncated or corrupt one is deleted rather than
    reused, which is the difference between "re-download 500 MB" and "extract
    garbage and fail the smoke test for no visible reason".
    """
    if not (target.is_file() and expected_bytes):
        return False
    if target.stat().st_size != expected_bytes:
        return False
    try:
        _verify_zip(target)
    except EngineError:
        target.unlink(missing_ok=True)
        return False
    return True


def _verify_zip(path: Path) -> None:
    if not zipfile.is_zipfile(path):
        raise EngineError(f"{path.name} is not a valid zip archive")
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
    except (zipfile.BadZipFile, OSError) as exc:
        raise EngineError(f"{path.name} could not be opened: {exc}") from exc
    if bad is not None:
        raise EngineError(f"{path.name} is corrupt (bad member {bad})")


def guess_variant(directory: Path) -> str:
    """Infer a backend from the shipped ggml libraries.

    Engine directories placed by hand (or by an older version) have no
    ``engine.json``; reporting "unknown" in the GUI for a perfectly good CUDA
    build is worse than reading it off the DLLs that are right there.
    """
    for variant, names in (
        ("cuda", ("ggml-cuda.dll", "libggml-cuda.so")),
        ("rocm", ("ggml-hip.dll", "libggml-hip.so")),
        ("vulkan", ("ggml-vulkan.dll", "libggml-vulkan.so")),
    ):
        if any((directory / name).is_file() for name in names):
            return variant
    return "unknown"


def find_server_binary(directory: Path) -> Path | None:
    """Locate ``llama-server`` in an engine directory, however it is nested."""
    direct = directory / BIN_NAME
    if direct.is_file():
        return direct
    if not directory.is_dir():
        return None
    for candidate in ("bin", "build/bin", "build/bin/Release"):
        nested = directory / candidate / BIN_NAME
        if nested.is_file():
            return nested
    with contextlib.suppress(OSError):
        return next(iter(sorted(directory.rglob(BIN_NAME))), None)
    return None


def build_assets(tag: str, raw_assets: Sequence[Any]) -> list[EngineAsset]:
    """Turn a GitHub asset payload into :class:`EngineAsset` records.

    CUDA runtime bundles ship as separate ``cudart-*`` archives on Windows; they
    are matched to their engine archive here so ``install`` can fetch both
    without a second API round-trip (a CUDA engine without cudart fails to
    start with a missing-DLL error that looks nothing like the real cause).
    """
    cudart: dict[tuple[str, str, str], str] = {}
    for entry in raw_assets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        url = entry.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        match = _CUDART_RE.match(name)
        if match:
            cudart[(match.group("os"), match.group("ver"), _norm_arch(match.group("arch")))] = url

    assets: list[EngineAsset] = []
    for entry in raw_assets:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        url = entry.get("browser_download_url")
        if not isinstance(name, str) or not isinstance(url, str):
            continue
        parsed = parse_asset_name(name)
        if parsed is None:
            continue
        asset_tag, asset_os, variant, arch = parsed
        cudart_url: str | None = None
        if variant.startswith("cuda-"):
            cudart_url = cudart.get((asset_os, variant[len("cuda-") :], arch))
        size = entry.get("size")
        assets.append(
            EngineAsset(
                tag=asset_tag or tag,
                name=name,
                url=url,
                size_bytes=int(size) if isinstance(size, int) else 0,
                variant=variant,
                needs_cudart=cudart_url is not None,
                cudart_url=cudart_url,
                os_token=asset_os,
                arch=arch,
            )
        )
    return assets


def _nvidia_smi_gpus() -> list[GpuInfo]:
    """Last-resort GPU enumeration when ``core.gpu`` is unavailable."""
    query = "index,name,memory.total,memory.free,compute_cap"
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            index = int(parts[0])
            total = int(float(parts[2])) * 1024 * 1024
            free = int(float(parts[3])) * 1024 * 1024
        except ValueError:
            continue
        gpus.append(
            GpuInfo(
                index=index,
                name=parts[1],
                total_bytes=total,
                free_bytes=free,
                used_bytes=max(0, total - free),
                compute_capability=_parse_version(parts[4]),
            )
        )
    return gpus


def _detect_cuda_driver_version() -> tuple[int, int] | None:
    """Max CUDA version the installed driver supports."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            raw = int(pynvml.nvmlSystemGetCudaDriverVersion_v2())
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        if raw > 0:
            return raw // 1000, (raw % 1000) // 10
    except Exception:  # pragma: no cover - nvml missing or no driver
        pass
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["nvidia-smi"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = _CUDA_VER_RE.search(result.stdout)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None
