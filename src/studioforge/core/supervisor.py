"""Supervision of ``llama-server`` child processes.

One loaded model == one supervised child process on a private loopback port.
This module owns everything about that child's life: the exact argv it is
launched with, its port, its log file, its readiness, its crash/restart policy
and -- most importantly -- making sure it is *really* dead when we say it is.

Four rules are load-bearing enough to be stated up front, because getting any
of them wrong fails silently rather than loudly:

* **A child can never outlive this process.** On Windows every child is placed
  in a job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel
  kills it when our last handle closes -- which happens however we die,
  including a ``SIGKILL``/Task Manager kill that never runs ``atexit``. See
  :class:`WindowsChildJob` and DECISIONS.md D23.
* **``--n-gpu-layers`` is always 999.** StudioForge is GPU-only by design; a
  model that does not fit is rejected by the planner, never quietly split onto
  the CPU. There is deliberately no code path here that computes a layer count.
* **``--ctx-size`` is the TOTAL context shared across ``--parallel`` slots**, so
  we pass ``ctx_size * parallel``. Verified against b10425: ``--ctx-size 4096``
  with 4 slots reports ``n_ctx: 4096`` and ``total_slots: 4``, i.e. 1024 tokens
  per conversation. Without the multiplication a user who asks for 8192 gets a
  quarter of it.
* **Speculative decoding needs ``--spec-type``.** b10425 renamed every drafting
  flag and defaults ``--spec-type`` to ``none``; the old names are *accepted and
  ignored* ("the argument has been removed"), so a wrong spelling here looks
  like speculative decoding simply not helping.
* **No optional flag is passed on faith.** Every flag below the mandatory core
  is gated on the *active engine* advertising it
  (:class:`studioforge.core.engine.EngineFeatures`, read from that build's own
  ``--help``). An engine whose help cannot be read advertises nothing, and the
  launch falls back to the flag surface that predates this gating rather than
  guessing. See DECISIONS.md D38.
"""

from __future__ import annotations

import asyncio
import atexit
import builtins
import contextlib
import os
import shlex
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, TYPE_CHECKING, Any

import httpx
import psutil

from studioforge.config import (
    CACHE_RAM_MIN_GRANT_MIB,
    Config,
    grant_cache_ram_mib,
    resolve_cache_ram_mb,
)
from studioforge.core.engine import (
    EngineFeatures,
    child_environment,
    policy_family,
    probe_engine_features,
    refuse_policy_flags,
)
from studioforge.core.planner import attention_kind, effective_ubatch, is_moe
from studioforge.errors import ModelLoadError, ModelUnloadError
from studioforge.logging import get_logger
from studioforge.types import (
    AdapterRecord,
    EffectiveLaunch,
    InstanceInfo,
    LoadPlan,
    ModelRecord,
    ModelSettings,
)

if TYPE_CHECKING:
    from studioforge.core.gpu import GpuProbe

log = get_logger(__name__)

#: Children always bind loopback only; the gateway is the sole public surface.
CHILD_HOST = "127.0.0.1"

#: GPU-only means *all* layers, unconditionally. Not a tunable.
ALL_GPU_LAYERS = "999"

#: Draft depth, emitted explicitly so the GUI's displayed command line matches
#: what the child actually runs. **3, not 16**: b10425's ``--help`` says
#: "number of tokens to draft for speculative decoding (default: 3)", and the
#: value used to be 16 here under a comment claiming it *was* the engine
#: default. Measured on Qwen3.8-27B Q5_K_S with its own MTP head (one RTX 3090,
#: 8k context, four unseen prompts): n_max 3 gave 50.7 tok/s at 53% draft
#: acceptance, n_max 4 gave 47.5 tok/s at 45%. Deeper drafts are past the knee
#: -- every rejected token was verified for nothing. See DECISIONS.md D38.
DEFAULT_SPEC_DRAFT_N_MAX = 3

#: StudioForge's own sentinel for ``ModelSettings.spec_type``.
SPEC_AUTO = "auto"
#: ``--spec-type`` value that disables drafting entirely.
SPEC_TYPE_NONE = "none"
#: ``--spec-type`` value that enables draft-model speculative decoding.
SPEC_TYPE_DRAFT = "draft-simple"
#: ``--spec-type`` value that uses the model's own multi-token-prediction heads
#: (GGUF ``nextn_predict_layers >= 1``). No draft model, no extra VRAM.
SPEC_TYPE_MTP = "draft-mtp"
#: Draftless n-gram speculation. llama.cpp recommends it for output that
#: repeats itself: reasoning models re-treading their own thoughts, code
#: iteration, MoE models. ~16 MiB of host state.
SPEC_TYPE_NGRAM = "ngram-mod"

#: ``--split-mode`` value that shards weights *and* KV across GPUs in parallel.
#: EXPERIMENTAL upstream, and gated hard -- see :func:`tensor_split_blockers`.
SPLIT_MODE_TENSOR = "tensor"
#: The pipelined default: layers are dealt out to devices and run in sequence.
SPLIT_MODE_LAYER = "layer"

#: ``--batch-size`` used once a model serves more than four slots. The engine
#: default (2048) is a *shared* logical batch, so many slots ingesting prompts
#: at once queue behind each other. ``--ubatch-size`` stays at its 512 default
#: -- that one is a VRAM term the planner models (planner.DEFAULT_UBATCH).
BATCH_SIZE_MANY_SLOTS = 4096

#: Above this many slots, ``auto`` speculative decoding turns itself off.
#: Speculation spends spare GPU compute to shorten a single stream: it drafts
#: tokens and verifies them in one pass, a big win at one slot because decode
#: is memory-bound there (the weights are read regardless, so the extra tokens
#: are nearly free). Every added slot fills that spare compute with real work,
#: and once the batch saturates the GPU the drafted-then-rejected tokens are
#: pure waste that slows *every* concurrent request. The crossover is model-
#: and hardware-specific, but it is well below the point of a heavy agent or
#: benchmark load; four is the same "many slots" line the batch size uses, and
#: a model that genuinely wants speculation at high concurrency can still set
#: ``spec_type`` explicitly. This gates ``auto`` only.
SPEC_AUTO_MAX_SLOTS = 4
#: llama.cpp's own default logical batch. A micro-batch is clamped to the
#: logical batch (``n_ubatch = min(n_batch, n_ubatch)``), so an ``-ub`` above
#: this needs ``-b`` raised with it or it is silently not what was asked for.
ENGINE_DEFAULT_BATCH_SIZE = 2048

#: ``--slot-prompt-similarity`` for a multi-slot launch. The 0.10 default makes
#: slot reuse almost accidental; 0.3 keeps an agent's near-identical prompts
#: landing on the slot that already has the prefix cached.
SLOT_PROMPT_SIMILARITY_MULTI = 0.3

#: llama.cpp's own defaults for the knobs a launch may leave unsaid, from
#: ``common/common.h`` at b10689 (``slot_prompt_similarity`` 0.10 :694,
#: ``n_ctx_checkpoints`` 32 :629, ``checkpoint_min_step`` 8192 :631,
#: ``cache_ram_mib`` 8192 :632, ``n_ubatch`` 512). Used by
#: :func:`effective_launch` when the engine's help could not be read; a known
#: engine's parsed defaults win.
ENGINE_DEFAULT_UBATCH_SIZE = 512
ENGINE_DEFAULT_SLOT_PROMPT_SIMILARITY = 0.10
ENGINE_DEFAULT_CTX_CHECKPOINTS = 32
ENGINE_DEFAULT_CHECKPOINT_MIN_STEP = 8192
ENGINE_DEFAULT_CACHE_RAM_MIB = 8192

#: Flags whose VALUE must never appear on a status surface. None of them is
#: emitted by :meth:`Supervisor.build_command`, but ``extra_flags`` is free
#: text and could carry any of them. ``--hf-token`` (``-hft``) is the Hugging
#: Face credential llama-server accepts on the command line (b10689
#: ``common/arg.cpp``); it is a secret exactly like the API key.
_SECRET_VALUE_FLAGS = frozenset(
    {"--api-key", "--api-key-file", "--ssl-key-file", "--hf-token", "-hft"}
)

#: Every spelling llama.cpp accepts for the flags :func:`effective_launch`
#: reads (b10689 ``common/arg.cpp``). Value flags consume the next token;
#: switch pairs set a boolean. Aliases matter because ``extra_flags`` is
#: written by hand and ``-nocb`` is as legal as ``--no-cont-batching``.
#:
#: The second block is the GPU-only policy's own vocabulary: what the child
#: was really told about layers, fit, devices and the KV cache, so a launch
#: that landed half on the CPU is *reported* as such and not only refused.
_VALUE_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "ctx_total": ("-c", "--ctx-size"),
    "parallel": ("-np", "--parallel"),
    "batch_size": ("-b", "--batch-size"),
    "ubatch_size": ("-ub", "--ubatch-size"),
    "cache_reuse": ("--cache-reuse",),
    "cache_ram_mib": ("-cram", "--cache-ram"),
    "slot_prompt_similarity": ("-sps", "--slot-prompt-similarity"),
    "ctx_checkpoints": ("-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"),
    "checkpoint_min_step": ("-cms", "--checkpoint-min-step"),
    "spec_type": ("--spec-type",),
    "flash_attn": ("-fa", "--flash-attn"),
    # --- GPU-only policy ---------------------------------------------------
    "n_gpu_layers": ("-ngl", "--gpu-layers", "--n-gpu-layers"),
    "fit": ("-fit", "--fit"),
    "device": ("-dev", "--device"),
    "split_mode": ("-sm", "--split-mode"),
    "load_mode": ("-lm", "--load-mode"),
    "spec_draft_ngl": ("--spec-draft-ngl", "-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"),
    "spec_draft_device": ("--spec-draft-device", "-devd", "--device-draft"),
    "mmproj_device": ("-mmdev", "--mmproj-device"),
}
_SWITCH_ALIASES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "cont_batching": (("-cb", "--cont-batching"), ("-nocb", "--no-cont-batching")),
    "kv_unified": (("-kvu", "--kv-unified"), ("-no-kvu", "--no-kv-unified")),
    "cache_prompt": (("--cache-prompt",), ("--no-cache-prompt",)),
    "cache_idle_slots": (("--cache-idle-slots",), ("--no-cache-idle-slots",)),
    # --- GPU-only policy ---------------------------------------------------
    "kv_offload": (("-kvo", "--kv-offload"), ("-nkvo", "--no-kv-offload")),
    "mmproj_offload": (("--mmproj-offload",), ("--no-mmproj-offload",)),
    "mlock": (("--mlock",), ()),
    "mmap": (("--mmap",), ("--no-mmap",)),
}

#: ``--n-gpu-layers`` / ``--spec-draft-ngl`` values that mean "everything".
#: ``999`` is what StudioForge passes; ``all`` is b10689's own word for it.
_ALL_LAYERS_VALUES = frozenset({ALL_GPU_LAYERS, "all"})
#: ``--load-mode`` values that pin the model in host RAM. Composed by
#: StudioForge itself for ``settings.mlock`` (see ``_load_mode_args``); a
#: policy violation only when nothing asked for them.
_MLOCK_LOAD_MODES = frozenset({"mlock", "mmap+mlock"})


def _is_absolute_path(token: str) -> bool:
    return PureWindowsPath(token).is_absolute() or PurePosixPath(token).is_absolute()


def _basename(token: str) -> str:
    return PureWindowsPath(token).name or PurePosixPath(token).name or token


def redact_argv(argv: Sequence[str]) -> list[str]:
    """The argv as it may appear on a status surface (D54).

    Every absolute path -- the binary, ``--model``, ``--mmproj``, ``--lora``,
    ``--chat-template-file``, anything from ``extra_flags`` -- is reduced to
    its basename, and the value after a key-carrying flag is replaced by
    ``"<redacted>"``.

    **This is now the only form the argv is ever written in (D55).** The
    docstring used to say the full command line stayed in the child's log file,
    "which is not served over HTTP" -- it was, by ``GET /api/logs/models/{id}``
    and, through the app ring buffer, by ``GET /api/logs``, both of which had no
    credential in front of them on an open install. Rather than gate two reads
    that are genuinely useful, the argv is redacted at every point it is
    recorded: the child log header, the ``model_spawn`` log line, and the
    ``ModelLoadError`` details. A secret an operator put in ``extra_flags`` is
    covered too, which the value-registration log scrubber never was.
    """
    out: list[str] = []
    redact_next = False
    for token in argv:
        if redact_next:
            out.append("<redacted>")
            redact_next = False
            continue
        if token in _SECRET_VALUE_FLAGS:
            out.append(token)
            redact_next = True
            continue
        flag, sep, _value = token.partition("=")
        if sep and flag in _SECRET_VALUE_FLAGS:
            out.append(f"{flag}=<redacted>")
            continue
        out.append(_basename(token) if _is_absolute_path(token) else token)
    return out


def _parse_launch_argv(argv: Sequence[str]) -> tuple[dict[str, str], dict[str, bool]]:
    """``({value flag: last value}, {switch: last state})`` from an argv.

    Last occurrence wins for both kinds, which is how llama.cpp reads a
    repeated option and the reason ``extra_flags`` goes last in
    :meth:`Supervisor.build_command`.
    """
    value_of = {alias: key for key, aliases in _VALUE_FLAG_ALIASES.items() for alias in aliases}
    switch_of: dict[str, tuple[str, bool]] = {}
    for key, (on, off) in _SWITCH_ALIASES.items():
        switch_of.update(dict.fromkeys(on, (key, True)))
        switch_of.update(dict.fromkeys(off, (key, False)))

    values: dict[str, str] = {}
    switches: dict[str, bool] = {}
    tokens = list(argv[1:])  # argv[0] is the binary
    i = 0
    while i < len(tokens):
        token = tokens[i]
        # ``--flag=value`` is read too. llama.cpp itself does not accept the
        # spelling, but the policy report must not be blind to a token just
        # because the engine would be -- a refusal that reads the argv the
        # way the attacker hopes it does is no refusal.
        flag, sep, inline = token.partition("=")
        if sep and flag in value_of:
            values[value_of[flag]] = inline
            i += 1
            continue
        if token in value_of and i + 1 < len(tokens):
            values[value_of[token]] = tokens[i + 1]
            i += 2
            continue
        if token in switch_of:
            key, state = switch_of[token]
            switches[key] = state
        i += 1
    return values, switches


def launch_policy_violations(
    argv: Sequence[str], settings: ModelSettings | None = None
) -> list[str]:
    """Every token of a FINAL argv that contradicts the GPU-only policy.

    Two kinds, named the way they were typed so the report is actionable:
    every token of a forbidden (``offload``) family that is present at all
    (``--cpu-moe``, ``--n-cpu-ffn 48``, ``-ot .*=CPU``), and every managed
    flag whose *last* value -- llama.cpp's last-wins rule -- contradicts what
    StudioForge always passes: ``--n-gpu-layers`` not ``999``/``all``,
    ``--fit`` not ``off``, ``--device none``, ``--no-kv-offload``, the same
    for the draft model and the vision projector, and a ``--load-mode`` that
    pins the model in host RAM when ``settings.mlock`` did not ask for it.

    Pure, and the same function on both sides of the fence:
    :meth:`Supervisor.build_command` refuses a launch whose argv returns
    anything here, and :func:`effective_launch` reports the list on every
    instance -- so on every launch StudioForge composes itself this is empty,
    and non-empty is a bug report (:attr:`EffectiveLaunch.policy_violations`).
    """
    found: list[str] = []
    tokens = list(argv[1:])
    i = 0
    while i < len(tokens):
        token = tokens[i]
        family = policy_family(token) if token.startswith("-") else None
        if family is not None and family.kind == "offload":
            base, sep, _inline = token.partition("=")
            if family.takes_value and not sep and i + 1 < len(tokens):
                found.append(f"{base} {tokens[i + 1]}")
                i += 2
                continue
            found.append(token)
        i += 1

    values, switches = _parse_launch_argv(argv)

    def _contradicts(key: str, flag: str, allowed: Callable[[str], bool]) -> None:
        value = values.get(key)
        if value is not None and not allowed(value):
            found.append(f"{flag} {value}")

    _contradicts("n_gpu_layers", "--n-gpu-layers", lambda v: v in _ALL_LAYERS_VALUES)
    _contradicts("fit", "--fit", lambda v: v == "off")
    _contradicts("device", "--device", lambda v: v.strip().lower() != "none")
    _contradicts("spec_draft_ngl", "--spec-draft-ngl", lambda v: v in _ALL_LAYERS_VALUES)
    _contradicts("spec_draft_device", "--spec-draft-device", lambda v: v.strip().lower() != "none")
    _contradicts("mmproj_device", "--mmproj-device", lambda v: v.strip().lower() != "none")
    if switches.get("kv_offload") is False:
        found.append("--no-kv-offload")
    if switches.get("mmproj_offload") is False:
        found.append("--no-mmproj-offload")
    load_mode = values.get("load_mode")
    asked_mlock = settings is not None and bool(settings.mlock)
    if load_mode in _MLOCK_LOAD_MODES and not asked_mlock:
        found.append(f"--load-mode {load_mode}")
    return found


def _int_or(value: str | None, fallback: int) -> int:
    try:
        return int(value) if value is not None else fallback
    except ValueError:
        return fallback


def _float_or(value: str | None, fallback: float) -> float:
    try:
        return float(value) if value is not None else fallback
    except ValueError:
        return fallback


def effective_launch(
    argv: Sequence[str],
    features: EngineFeatures,
    plan: LoadPlan,
    settings: ModelSettings | None = None,
) -> EffectiveLaunch:
    """What the child really runs with: the argv, then the engine's defaults.

    Pure. ``features`` supplies the defaults a known engine printed in its
    help; an unknown engine (help unreadable) falls back to the b10689
    ``common.h`` constants and marks them ``engine_default`` too -- the same
    "advertises nothing" fallback D38 uses, applied to reporting rather than
    to emission. ``settings`` is only consulted to name what the child
    *cannot* see (:attr:`EffectiveLaunch.inert`).
    """
    values, switches = _parse_launch_argv(argv)
    sources: dict[str, str] = {}

    def src(field: str, from_argv: bool) -> None:
        sources[field] = "argv" if from_argv else "engine_default"

    parallel = _int_or(values.get("parallel"), plan.parallel)
    if parallel < 1:  # llama.cpp's -1 = auto; StudioForge always passes a count
        parallel = max(1, plan.parallel)
    src("parallel", "parallel" in values)
    ctx_total = _int_or(values.get("ctx_total"), plan.ctx_size * max(1, plan.parallel))
    src("ctx_total", "ctx_total" in values)
    ctx_per_slot = ctx_total // max(1, parallel)

    cache_prompt = switches.get("cache_prompt", features.cache_prompt_default)
    src("cache_prompt", "cache_prompt" in switches)
    cache_reuse = _int_or(values.get("cache_reuse"), 0)
    src("cache_reuse", "cache_reuse" in values)

    cache_ram_mib: int | None
    if "cache_ram_mib" in values:
        cache_ram_mib = _int_or(values["cache_ram_mib"], ENGINE_DEFAULT_CACHE_RAM_MIB)
    elif features.cache_ram:
        cache_ram_mib = (
            features.cache_ram_default_mib
            if features.cache_ram_default_mib is not None
            else ENGINE_DEFAULT_CACHE_RAM_MIB
        )
    elif features.known:
        cache_ram_mib = None  # this build has no host cache at all
    else:
        cache_ram_mib = ENGINE_DEFAULT_CACHE_RAM_MIB
    src("cache_ram_mib", "cache_ram_mib" in values)

    # The engine turns idle-slot snapshots off itself when there is no host
    # cache to put them in (b10689 server-context.cpp:1420-1423).
    has_idle = features.cache_idle_slots or not features.known
    cache_idle_slots = switches.get("cache_idle_slots", has_idle)
    if not cache_ram_mib:
        cache_idle_slots = False
    src("cache_idle_slots", "cache_idle_slots" in switches)

    cont_batching = switches.get("cont_batching", features.cont_batching_default)
    src("cont_batching", "cont_batching" in switches)
    # The engine's own default is "enabled if the slot count is auto", and
    # StudioForge always passes an explicit count, so unsaid means partitioned.
    kv_unified = switches.get("kv_unified", False)
    src("kv_unified", "kv_unified" in switches)

    similarity_default = (
        features.slot_prompt_similarity_default
        if features.slot_prompt_similarity_default is not None
        else ENGINE_DEFAULT_SLOT_PROMPT_SIMILARITY
    )
    slot_prompt_similarity = _float_or(values.get("slot_prompt_similarity"), similarity_default)
    src("slot_prompt_similarity", "slot_prompt_similarity" in values)

    batch_size = _int_or(values.get("batch_size"), ENGINE_DEFAULT_BATCH_SIZE)
    src("batch_size", "batch_size" in values)
    # n_ubatch = min(n_batch, n_ubatch) inside llama.cpp: report the clamp.
    ubatch_size = min(batch_size, _int_or(values.get("ubatch_size"), ENGINE_DEFAULT_UBATCH_SIZE))
    src("ubatch_size", "ubatch_size" in values)

    ctx_checkpoints: int | None
    if "ctx_checkpoints" in values:
        ctx_checkpoints = _int_or(values["ctx_checkpoints"], ENGINE_DEFAULT_CTX_CHECKPOINTS)
    elif features.ctx_checkpoints or not features.known:
        ctx_checkpoints = (
            features.ctx_checkpoints_default
            if features.ctx_checkpoints_default is not None
            else ENGINE_DEFAULT_CTX_CHECKPOINTS
        )
    else:
        ctx_checkpoints = None
    src("ctx_checkpoints", "ctx_checkpoints" in values)

    checkpoint_min_step: int | None
    if "checkpoint_min_step" in values:
        checkpoint_min_step = _int_or(
            values["checkpoint_min_step"], ENGINE_DEFAULT_CHECKPOINT_MIN_STEP
        )
    elif features.has("--checkpoint-min-step") or not features.known:
        checkpoint_min_step = (
            features.checkpoint_min_step_default
            if features.checkpoint_min_step_default is not None
            else ENGINE_DEFAULT_CHECKPOINT_MIN_STEP
        )
    else:
        checkpoint_min_step = None
    src("checkpoint_min_step", "checkpoint_min_step" in values)

    spec_type = values.get("spec_type") or SPEC_TYPE_NONE
    src("spec_type", "spec_type" in values)
    flash_attn = values.get("flash_attn") or "auto"
    src("flash_attn", "flash_attn" in values)

    # The GPU-only policy, as launched. StudioForge always passes the first
    # two, so "engine_default" on either means the argv builder was bypassed.
    n_gpu_layers = values.get("n_gpu_layers")
    src("n_gpu_layers", "n_gpu_layers" in values)
    fit = values.get("fit")
    src("fit", "fit" in values)
    device = values.get("device")
    src("device", "device" in values)
    split_mode = values.get("split_mode")
    src("split_mode", "split_mode" in values)
    load_mode = values.get("load_mode")
    src("load_mode", "load_mode" in values)
    kv_offload = switches.get("kv_offload", True)
    src("kv_offload", "kv_offload" in switches)
    policy_violations = launch_policy_violations(argv, settings)

    inert: list[str] = []
    if settings is not None and settings.cont_batching is False and cont_batching:
        inert.append("cont_batching")
    # mlock is honoured by the deprecated switch or by a load mode that locks;
    # anything else means the engine advertised neither (see
    # Supervisor._load_mode_args) and the setting is a wish.
    if (
        settings is not None
        and settings.mlock
        and not switches.get("mlock")
        and load_mode not in _MLOCK_LOAD_MODES
    ):
        inert.append("mlock")
    if (
        settings is not None
        and settings.no_mmap
        and switches.get("mmap", True) is not False
        and load_mode not in ("none", "mlock")
    ):
        inert.append("no_mmap")

    if cache_prompt:
        cache_bits = [f"reuse {cache_reuse}" if cache_reuse else "chunk reuse off"]
        if cache_ram_mib is None:
            cache_bits.append("no host cache")
        elif cache_ram_mib == 0:
            cache_bits.append("host cache off")
        elif cache_ram_mib < 0:
            cache_bits.append("host cache unlimited")
        else:
            cache_bits.append(f"host {cache_ram_mib} MiB")
        cache_bits.append(
            f"routing {_fmt_float(slot_prompt_similarity)}"
            if slot_prompt_similarity
            else "no routing"
        )
        cache_text = "prefix cache on (" + ", ".join(cache_bits) + ")"
    else:
        cache_text = "prefix cache OFF"
    slots_word = "slot" if parallel == 1 else "slots"
    parts = [
        cache_text,
        "continuous batching " + ("on" if cont_batching else "OFF"),
        f"{parallel} {slots_word} x {ctx_per_slot}",
        "unified KV" if kv_unified else "partitioned KV",
        f"spec {spec_type}",
    ]
    if inert:
        parts.append("inert: " + ", ".join(inert))
    # Last, always: the one word an operator scans a status line for.
    if policy_violations:
        parts.append("POLICY VIOLATION: " + ", ".join(policy_violations))
    else:
        parts.append("GPU-only")

    return EffectiveLaunch(
        cache_prompt=cache_prompt,
        cache_reuse=cache_reuse,
        cache_ram_mib=cache_ram_mib,
        cache_idle_slots=cache_idle_slots,
        cont_batching=cont_batching,
        kv_unified=kv_unified,
        slot_prompt_similarity=slot_prompt_similarity,
        parallel=parallel,
        ctx_per_slot=ctx_per_slot,
        ctx_total=ctx_total,
        batch_size=batch_size,
        ubatch_size=ubatch_size,
        ctx_checkpoints=ctx_checkpoints,
        checkpoint_min_step=checkpoint_min_step,
        spec_type=spec_type,
        flash_attn=flash_attn,
        n_gpu_layers=n_gpu_layers,
        fit=fit,
        device=device,
        split_mode=split_mode,
        load_mode=load_mode,
        kv_offload=kv_offload,
        policy_violations=policy_violations,
        sources=sources,
        inert=inert,
        summary=", ".join(parts),
    )


#: Lines of child output kept in memory per instance for error reporting.
STDERR_RING_SIZE = 200

#: How many of those lines are quoted in a load failure message.
ERROR_TAIL_LINES = 30

_HTTP_TIMEOUT = 5.0

#: How long a killed child may take to actually exit before its unload is
#: declared unverified (audit 2026-09-09, F1). On Windows ``terminate()`` IS
#: ``TerminateProcess``, so the whole wait is the CUDA driver tearing down a
#: 20+ GB context -- tens of seconds for a 27B at 256k -- and the old chain
#: gave it ~35 s in total before raising ``ModelUnloadError`` for a process
#: that died seconds later. The wait is patience, not grace: every signal has
#: already been sent by the time it starts (see ``_linger``).
UNLOAD_SETTLE_S = 60.0
#: Poll interval inside that wait.
UNLOAD_POLL_S = 0.5
#: Pause between a verified exit and the "after" VRAM sample (F6): the driver
#: hands memory back a beat after the process is gone, so a sample taken on
#: the same tick under-reports ``vram_reclaimed_mb`` -- and a plan computed on
#: it sees VRAM still in use on an empty card. Same value as the watchdog's
#: ``VRAM_SETTLE_S``; the manager awaits :attr:`Supervisor.vram_settle_s`
#: before its OOM-retry re-plan and its lease handover for the same reason.
VRAM_SETTLE_S = 1.2

# ---------------------------------------------------------------------------
# Interpreter-exit safety net
# ---------------------------------------------------------------------------

_TRACKED_PIDS: set[int] = set()
_ATEXIT_REGISTERED = False


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_kill_tracked_pids)
        _ATEXIT_REGISTERED = True


def _kill_tracked_pids() -> None:
    """Last-resort cleanup: an abrupt exit must not orphan a VRAM holder."""
    for pid in list(_TRACKED_PIDS):
        with contextlib.suppress(Exception):
            kill_process_tree(pid, timeout=2.0, force=True)
    _TRACKED_PIDS.clear()


# ---------------------------------------------------------------------------
# Kernel-enforced child lifetime (Windows job object)
# ---------------------------------------------------------------------------

#: ``CreateProcess`` flag: the child exists but its initial thread is suspended
#: until someone resumes it. Not exported by :mod:`subprocess`, so spelled out.
CREATE_SUSPENDED = 0x00000004

#: ``OpenProcess`` rights needed to move a process into a job object.
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


def _load_win32() -> tuple[Any, Any]:
    """Import the pywin32 job-object bindings.

    A module-level function purely so tests can monkeypatch it to raise
    (simulating a box without pywin32) or to return fakes.
    """
    import win32api
    import win32job

    return win32job, win32api


class WindowsChildJob:
    """A job object whose members die when this process does.

    **The failure this exists to prevent** (DECISIONS.md D23): on 2026-08-18
    three ``llama-server`` children holding ~25 GiB of VRAM were found running
    with "everything stopped". Their parent was a ``pytest`` process, and they
    only exited because that parent exited *cleanly*. The ``atexit`` net above
    is exactly that -- an ``atexit`` net: it does not run for a ``SIGKILL``, a
    Task Manager "End task", a hard power-cycle of the interpreter, or a
    segfault. On a GPU-only server every one of those cases leaks VRAM that
    nothing on the box knows how to attribute.

    A job object moves the guarantee into the kernel.
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` terminates every process still in
    the job the moment its **last handle** closes, and the handle closes when
    this process ends however it ends. The job is anonymous (no name) so two
    supervisors in two processes never share one.

    Nested jobs are legal on Windows 8+, which matters here: the serve process
    routinely already lives inside somebody else's job (the tray launcher, a
    terminal, a CI runner). Where nesting is refused anyway --
    ``ERROR_ACCESS_DENIED`` from ``AssignProcessToJobObject`` -- the failure is
    logged once at WARNING and the load continues. A safety net that refuses to
    be hung must never be the reason a model will not load.
    """

    def __init__(self) -> None:
        win32job, _ = _load_win32()
        self._win32job = win32job
        # Anonymous: a named job would be shared with any other process that
        # guessed the name, and closing it there would kill our children.
        self._handle: Any = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            self._handle, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(
            self._handle, win32job.JobObjectExtendedLimitInformation, info
        )
        self._closed = False
        #: Pids whose failed assignment has been logged. Per child, not per
        #: process: the old once-ever latch meant a box where nesting is
        #: refused logged one warning at boot and then launched every later
        #: model unprotected in silence -- the D23 guarantee quietly absent
        #: (audit 2026-09-09 §2.10).
        self._warned_pids: set[int] = set()

    @property
    def available(self) -> bool:
        return not self._closed and self._handle is not None

    def assign(self, pid: int) -> bool:
        """Put ``pid`` in the job. Never raises; ``False`` means "unprotected".

        Every failure is logged, once per child: the warning names the pid
        that is now unprotected, which is the thing an operator has to go and
        find after a hard kill.
        """
        if not self.available:
            return False
        try:
            win32job, win32api = self._win32job, _load_win32()[1]
            handle = win32api.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            try:
                win32job.AssignProcessToJobObject(self._handle, handle)
            finally:
                with contextlib.suppress(Exception):
                    handle.Close()
        except Exception as exc:  # noqa: BLE001 - the net must not break the load
            if pid not in self._warned_pids:
                self._warned_pids.add(pid)
                log.warning(
                    "child_job_assign_failed",
                    pid=pid,
                    error=str(exc),
                    detail=(
                        "this llama-server child is not protected by a job object; a hard "
                        "kill of this process would leave it holding VRAM. Usually means "
                        "the process is already in a job that refuses nesting (pre-Windows "
                        "8, or a job without BREAKAWAY_OK)."
                    ),
                )
            return False
        return True

    def close(self) -> None:
        """Close the job handle, killing anything still in it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._handle.Close()
        self._handle = None


def create_child_job() -> WindowsChildJob | None:
    """A job object for this process's children, or ``None`` where impossible.

    ``None`` on every non-Windows platform (POSIX gets the same guarantee from
    process groups plus the ``atexit`` net, and a missing job object is not a
    reason to fail) and on a Windows box without pywin32.
    """
    if os.name != "nt":
        return None
    try:
        return WindowsChildJob()
    except Exception as exc:  # noqa: BLE001 - degrade, never refuse to serve
        log.warning(
            "child_job_unavailable",
            error=str(exc),
            detail=(
                "could not create a Windows job object; llama-server children will "
                "not be killed automatically if this process is hard-killed"
            ),
        )
        return None


def process_create_time(pid: int) -> float | None:
    """Creation timestamp of ``pid``, or ``None`` if it cannot be read.

    Captured at spawn so a later liveness check cannot be fooled by pid reuse
    -- on a busy box the OS can hand our dead child's pid to something else,
    and mistaking that for "the model is still running" would be a false alarm
    in exactly the code whose job is to be trusted.
    """
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, ValueError):  # pragma: no cover - race with exit
        return None


#: How often a child whose OS wait failed is re-checked by pid (D64).
EXIT_POLL_S = 2.0


def _why(exc: BaseException) -> str:
    """A short cause for a log field: the class, and the first line of a StudioForge message."""
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message:
        return f"{type(exc).__name__}: {message.splitlines()[0][:200]}"
    return type(exc).__name__


#: Windows NTSTATUS values a crashing llama-server actually exits with, by name
#: (D64, CR-7). ``asyncio`` reports them as large positive integers --
#: ``3221226505`` is ``0xC0000409`` -- which nobody recognises in a log line.
WINDOWS_EXIT_STATUS: dict[int, str] = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000001D: "STATUS_ILLEGAL_INSTRUCTION",
    0xC0000094: "STATUS_INTEGER_DIVIDE_BY_ZERO",
    0xC00000FD: "STATUS_STACK_OVERFLOW",
    0xC000013A: "STATUS_CONTROL_C_EXIT",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def describe_exit_code(code: int | None) -> dict[str, Any]:
    """The ``model_exited`` fields for an exit code, readable on both platforms (D64).

    ``None`` is said out loud (``exit_code_unavailable: true``) rather than
    omitted. A negative code is a POSIX signal (``signal: "SIGKILL"``); a code at
    or above ``0xC0000000`` is a Windows NTSTATUS (``exit_code_hex`` and, when
    known, ``exit_status``).
    """
    if code is None:
        return {"exit_code": None, "exit_code_unavailable": True}
    fields: dict[str, Any] = {"exit_code": code}
    if code < 0:
        import signal

        with contextlib.suppress(ValueError):
            fields["signal"] = signal.Signals(-code).name
    elif code >= 0xC0000000:
        fields["exit_code_hex"] = f"0x{code & 0xFFFFFFFF:08X}"
        name = WINDOWS_EXIT_STATUS.get(code & 0xFFFFFFFF)
        if name is not None:
            fields["exit_status"] = name
    return fields


def process_is_alive(pid: int, *, create_time: float | None = None) -> bool:
    """Whether ``pid`` is a live (non-zombie) process, honouring ``create_time``."""
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            actual = proc.create_time()
            # More than a second apart means the pid was recycled: our process
            # is gone and this is a stranger wearing its number.
            if abs(actual - create_time) > 1.0:
                return False
        return bool(proc.is_running())
    except (psutil.Error, ValueError):
        return False


@dataclass(slots=True)
class UnloadReport:
    """Evidence that an unload actually happened.

    Kept per model and exposed through :meth:`Supervisor.unload_report` so the
    claim "unloaded" is backed by something checkable rather than by the fact
    that a kill call returned.
    """

    model_id: str
    pid: int | None
    pid_gone: bool
    escalated: bool = False
    vram_before_bytes: int = 0
    vram_after_bytes: int = 0
    at: float = 0.0

    @property
    def vram_reclaimed_bytes(self) -> int:
        return max(0, self.vram_before_bytes - self.vram_after_bytes)


def kill_process_tree(pid: int, *, timeout: float = 15.0, force: bool = False) -> None:
    """Terminate ``pid`` and every descendant, escalating to SIGKILL.

    Killing only the direct child is not enough: llama-server can spawn helpers,
    and any survivor keeps its CUDA context -- which on a GPU-only server means
    permanently leaked VRAM and a model that can never be loaded again without a
    reboot. So we enumerate the whole tree with psutil, signal it, wait, then
    hard-kill whatever is left.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return
    try:
        procs: list[psutil.Process] = parent.children(recursive=True)
    except psutil.Error:
        procs = []
    procs.append(parent)

    for proc in procs:
        with contextlib.suppress(psutil.Error):
            if force:
                proc.kill()
            else:
                proc.terminate()
    _, alive = psutil.wait_procs(procs, timeout=max(0.0, timeout))
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    if alive:
        psutil.wait_procs(alive, timeout=5.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: float, places: int = 4) -> str:
    """Format a float for the command line: fixed precision, no noise digits."""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def safe_log_name(model_id: str) -> str:
    """Turn a model id into a single filesystem-safe log file name."""
    cleaned = []
    for char in model_id:
        cleaned.append(char if (char.isalnum() or char in "-_.") else "_")
    name = "".join(cleaned).strip("._") or "model"
    return name[:120]


def _port_is_bindable(port: int, host: str = CHILD_HOST) -> bool:
    """True when nothing else currently holds ``port``.

    Bookkeeping alone is not enough -- another process (a stale llama-server, a
    dev server) can own a port inside our range -- so we actually try to bind.
    ``SO_EXCLUSIVEADDRUSE`` is essential on Windows: a listener that set
    ``SO_REUSEADDR`` (llama-server and most servers do) otherwise lets a second
    plain bind succeed, so the probe would report a busy port as free and we
    would end up talking to the wrong process.

    On POSIX the probe must set ``SO_REUSEADDR`` for the mirror-image reason.
    A child that has just been unloaded leaves its served connections in
    ``TIME_WAIT`` for around a minute, and a *plain* bind to that port fails
    for as long as they last -- so the probe called a port busy that the next
    llama-server, which sets ``SO_REUSEADDR`` itself, would have taken
    happily. The port was skipped for a minute after every unload, and on a
    narrow ``child_port_start..end`` range that is how "No free port in the
    llama-server child range" arrives on a box with free ports. The two
    options are opposites by design: on Windows the danger is a false *free*,
    on Linux a false *busy*, and the probe has to model what the child will
    actually do on each.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            with contextlib.suppress(OSError, AttributeError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            with contextlib.suppress(OSError, AttributeError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


# ---------------------------------------------------------------------------
# Launch-time feature resolution (DECISIONS.md D38)
# ---------------------------------------------------------------------------

#: KV cache types ``--split-mode tensor`` is documented to work with. Upstream's
#: multi-GPU doc says quantized KV is "not implemented" for tensor mode. b10425
#: does *not* enforce it -- a scratch load with ``-sm tensor --cache-type-k q8_0``
#: started and answered correctly -- so this list is policy, not a crash guard:
#: a quality-first server does not run a combination its own engine documents as
#: unimplemented just because it happens not to fall over on a 1.5B.
TENSOR_SPLIT_KV_TYPES = frozenset({"f32", "f16", "bf16"})


@dataclass(frozen=True, slots=True)
class ResolvedFeatures:
    """What the optional-feature knobs resolved to for one launch.

    Kept as a value rather than mutated onto the plan so the same resolution can
    be computed for a *preview* (the GUI's "command line" panel) without any
    side effect, and applied once, in one place, when the child is really
    spawned.
    """

    spec_type: str = SPEC_TYPE_NONE
    spec_draft_n_max: int | None = None
    spec_reason: str = ""
    split_mode: str = SPLIT_MODE_LAYER
    split_mode_reason: str | None = None

    @property
    def drafting(self) -> bool:
        return self.spec_type != SPEC_TYPE_NONE

    def speculative_dict(self, *, draft_model_id: str | None = None) -> dict[str, Any]:
        """The block carried on :class:`~studioforge.types.LoadPlan` / InstanceInfo."""
        return {
            "type": self.spec_type,
            "draft_n_max": self.spec_draft_n_max,
            "draft_model_id": draft_model_id,
            "reason": self.spec_reason,
        }


def _nextn_heads(record: ModelRecord) -> int:
    """``nextn_predict_layers`` from the GGUF, or 0.

    Captured by the scanner into ``meta.extra``; it is the *only* honest signal
    that a model has multi-token-prediction heads. Repository names lie about it
    both ways -- a library "...-MTP-GGUF" on this box carries no such key at all.
    """
    meta = record.meta
    extra = getattr(meta, "extra", None) if meta is not None else None
    if not isinstance(extra, dict):
        return 0
    try:
        return int(extra.get("nextn_predict_layers") or 0)
    except (TypeError, ValueError):
        return 0


def resolve_spec_type(
    record: ModelRecord, features: EngineFeatures, *, has_draft: bool, slots: int = 1
) -> tuple[str, str]:
    """Pick the ``--spec-type`` for this model. Returns ``(type, reason)``.

    Speculative decoding is *distribution-preserving*: the draft proposes, the
    full model verifies, and rejected tokens are resampled from the true
    distribution. It is therefore the rare feature that is pure speed with no
    quality cost, which is why ``auto`` is allowed to turn it on by itself --
    but only *at low concurrency*. ``slots`` is the launch's parallel slot count
    (D17); above :data:`SPEC_AUTO_MAX_SLOTS`, ``auto`` returns ``none``, because
    speculation trades spare compute for latency and a saturated multi-slot
    batch has none to spare (see the constant). An explicit ``spec_type`` is
    still honoured at any slot count -- the caller chose it.

    Raises :class:`ModelLoadError` when an explicitly configured type is not one
    the active engine offers -- b10425 accepts unknown values on some flags and
    ignores them, and "speculation configured but silently off" is precisely the
    failure this whole module refuses to allow (D2).
    """
    requested = (record.settings.spec_type or SPEC_AUTO).strip()

    if requested != SPEC_AUTO:
        if requested == SPEC_TYPE_NONE:
            return SPEC_TYPE_NONE, "turned off for this model"
        if not features.known:
            log.warning(
                "spec_type_unverified",
                model_id=record.id,
                spec_type=requested,
                detail=(
                    "the active engine's --help could not be read, so this explicitly "
                    "configured --spec-type is passed without being checked against it"
                ),
            )
            return requested, "set on this model (engine feature list unavailable)"
        if not features.supports_spec(requested):
            offered = ", ".join(features.spec_types) or "none"
            raise ModelLoadError(
                f"'{record.id}' asks for --spec-type {requested}, which engine "
                f"{features.tag or '(active)'} does not offer. It offers: {offered}. "
                "Change the model's spec_type, or pin an engine that has it.",
                details={"model_id": record.id, "spec_type": requested, "offered": offered},
            )
        return requested, "set on this model"

    # --- auto ---------------------------------------------------------
    if slots > SPEC_AUTO_MAX_SLOTS:
        # A saturated multi-slot batch has no spare compute for drafting to
        # spend, so speculation there slows every concurrent request down
        # rather than speeding one up. Off before any of the what-to-draft-from
        # checks below, since none of them changes this.
        return (
            SPEC_TYPE_NONE,
            f"{slots} slots: speculation is a single-stream win and hurts a saturated batch",
        )
    if not features.known:
        # Unknown engine: keep exactly the pre-gating behaviour, no guesses.
        if has_draft:
            return SPEC_TYPE_DRAFT, "a draft model is attached"
        return SPEC_TYPE_NONE, "engine feature list unavailable"

    heads = _nextn_heads(record)
    if heads >= 1 and features.supports_spec(SPEC_TYPE_MTP):
        return SPEC_TYPE_MTP, f"the model carries {heads} multi-token-prediction head(s)"
    if has_draft and features.supports_spec(SPEC_TYPE_DRAFT):
        return SPEC_TYPE_DRAFT, "a draft model is attached"
    thinking = bool(record.capabilities.thinking)
    moe = is_moe(record.meta)
    if (thinking or moe) and features.supports_spec(SPEC_TYPE_NGRAM):
        why = "a thinking model" if thinking else "a mixture-of-experts model"
        return SPEC_TYPE_NGRAM, f"{why}; its output repeats itself often enough to draft from"
    return SPEC_TYPE_NONE, "nothing to draft from: no MTP heads, no draft model"


def spec_draft_n_max_for(record: ModelRecord, spec_type: str) -> int | None:
    """``--spec-draft-n-max`` for ``spec_type``, or ``None`` to omit the flag.

    The n-gram types do not read it (they have ``--spec-ngram-*-n-max`` instead),
    so emitting it there would be a flag that looks like it is doing something.
    """
    if not any(part.strip().startswith("draft-") for part in spec_type.split(",")):
        return None
    explicit = record.settings.spec_draft_n_max
    return int(explicit) if explicit is not None else DEFAULT_SPEC_DRAFT_N_MAX


def tensor_split_model_blockers(record: ModelRecord) -> list[str]:
    """Reasons *this model* cannot use ``--split-mode tensor``, model-only.

    Split out from :func:`tensor_split_blockers` so the benchmark can decide
    which modes to even offer without a plan or an engine in hand.
    """
    reasons: list[str] = []
    meta = record.meta
    if meta is None:
        return ["the model's GGUF metadata could not be read, so it cannot be proven dense"]
    if is_moe(meta):
        reasons.append("it is a mixture-of-experts model (llama.cpp refuses MoE in tensor mode)")
    kind = attention_kind(meta)
    if kind == "hybrid":
        reasons.append(
            "it is a hybrid/state-space model (recurrent layers are not supported in tensor mode)"
        )
    elif kind == "unknown":
        reasons.append("its attention layout could not be determined from the GGUF")
    return reasons


def tensor_split_blockers(
    record: ModelRecord, plan: LoadPlan, features: EngineFeatures
) -> list[str]:
    """Every reason ``--split-mode tensor`` cannot be used for this launch.

    Empty means it can. Each entry is a sentence the user can act on, because
    the alternative -- letting the child start and die -- costs a model load and
    produces a stack trace instead of a suggestion. Two of these are hard errors
    in b10425 and were reproduced on the rig: flash attention off exits with
    ``SPLIT_MODE_TENSOR requires flash_attn to be enabled``, and a single device
    makes the mode meaningless.
    """
    reasons: list[str] = []
    if len(plan.devices) < 2:
        reasons.append("the placement uses a single GPU, where tensor mode does nothing")
    if not features.known:
        reasons.append("the active engine's feature list could not be read")
    elif not features.supports_split(SPLIT_MODE_TENSOR):
        offered = ", ".join(features.split_modes) or "unknown"
        reasons.append(
            f"engine {features.tag or '(active)'} has no --split-mode tensor (it offers: {offered})"
        )
    if plan.flash_attn != "on":
        reasons.append(
            f"tensor mode requires flash attention on, and this plan uses '{plan.flash_attn}'"
        )
    quantized = sorted(
        {plan.kv_cache_type, plan.kv_cache_type_v} - TENSOR_SPLIT_KV_TYPES,
    )
    if quantized:
        reasons.append(
            f"tensor mode needs an unquantized KV cache and this plan uses {', '.join(quantized)} "
            "(lower the context, or set kv_cache_type f16, to get one)"
        )
    reasons.extend(tensor_split_model_blockers(record))
    return reasons


def resolve_split_mode(
    record: ModelRecord, plan: LoadPlan, features: EngineFeatures
) -> tuple[str, str | None]:
    """Resolve ``plan.split_mode`` for the launch. Returns ``(mode, reason)``.

    ``auto`` downgrades to ``layer`` with a logged reason; an explicit
    ``tensor`` that cannot run is refused, because a user who typed "tensor" and
    silently got "layer" would go on to benchmark the wrong thing.
    """
    requested = plan.split_mode
    if len(plan.devices) < 2:
        # One device: the engine wants 'none', and that is what _placement_args
        # has always emitted. Nothing to resolve.
        return "none", None
    if requested not in (SPLIT_MODE_TENSOR, SPEC_AUTO):
        return requested, None

    blockers = tensor_split_blockers(record, plan, features)
    if not blockers:
        return SPLIT_MODE_TENSOR, None
    detail = "; ".join(blockers)
    if requested == SPLIT_MODE_TENSOR:
        raise ModelLoadError(
            f"'{record.id}' asks for --split-mode tensor, which cannot be used here: {detail}.",
            details={"model_id": record.id, "blockers": blockers},
        )
    reason = f"split_mode auto chose layer over tensor: {detail}"
    log.info("split_mode_downgraded", model_id=record.id, detail=detail)
    return SPLIT_MODE_LAYER, reason


def resolve_launch_features(
    record: ModelRecord,
    plan: LoadPlan,
    features: EngineFeatures,
    *,
    has_draft: bool,
) -> ResolvedFeatures:
    """Resolve every ``auto``/gated knob for one launch. Pure; may raise."""
    spec_type, spec_reason = resolve_spec_type(
        record, features, has_draft=has_draft, slots=max(1, plan.parallel)
    )
    split_mode, split_reason = resolve_split_mode(record, plan, features)
    return ResolvedFeatures(
        spec_type=spec_type,
        spec_draft_n_max=spec_draft_n_max_for(record, spec_type),
        spec_reason=spec_reason,
        split_mode=split_mode,
        split_mode_reason=split_reason,
    )


class _Instance:
    """Mutable supervisor-side state for one child process."""

    def __init__(
        self,
        *,
        info: InstanceInfo,
        record: ModelRecord,
        plan: LoadPlan,
        port: int,
        engine_tag: str | None,
        draft: ModelRecord | None,
        adapters: Sequence[tuple[AdapterRecord, float]],
        log_path: Path,
    ) -> None:
        self.info = info
        self.record = record
        self.plan = plan
        self.port = port
        self.engine_tag = engine_tag
        self.draft = draft
        self.adapters = list(adapters)
        self.log_path = log_path
        self.proc: asyncio.subprocess.Process | None = None
        self.wait_task: asyncio.Task[int] | None = None
        self.pumps: list[asyncio.Task[None]] = []
        self.watcher: asyncio.Task[None] | None = None
        self.stderr_ring: deque[str] = deque(maxlen=STDERR_RING_SIZE)
        self.argv: list[str] = []
        # Explicit intent flag: a deliberate stop() must never look like a
        # crash. Guessing from the exit code cannot work -- a terminated
        # process and a crashed one report the same thing on Windows.
        self.stopping = False
        # Alias reported by a *foreign* server found on our port, if any.
        self.port_conflict: str | None = None
        # Captured at spawn so an unload check cannot be fooled by pid reuse.
        self.create_time: float | None = None
        # The pid whose end has been logged as ``model_exited``: every exit is
        # reported by whichever path notices it first, and exactly once (D64).
        self.exit_logged_pid: int | None = None
        self._log_file: IO[str] | None = None

    # --- logging -------------------------------------------------------

    def open_log(self) -> None:
        if self._log_file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("a", encoding="utf-8", errors="replace")

    def write_log(self, line: str) -> None:
        if self._log_file is None:
            return
        with contextlib.suppress(ValueError, OSError):
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None

    def stderr_tail(self, n: int = ERROR_TAIL_LINES) -> list[str]:
        return list(self.stderr_ring)[-n:]


#: A tiny self-exec'ing shim for Linux: ask the kernel to SIGKILL the child
#: when its parent thread dies, then exec the real binary. This is the POSIX
#: counterpart of the Windows job object (D23): ``kill -9`` of the server, an
#: OOM-kill, a crashed interpreter -- the llama-server children go with it
#: instead of holding VRAM until the next boot's orphan sweep. Done as a
#: separate interpreter rather than ``preexec_fn`` because ``preexec_fn`` runs
#: after ``fork()`` in *this* multi-threaded process (structlog locks, httpx
#: pools, the asyncio child watcher) and is documented unsafe there; the shim
#: is single-threaded, does two syscalls and execs. ``prctl`` is Linux-only,
#: so on other POSIX systems the prefix is empty and behaviour is unchanged.
#: Printed by the shim when the exec fails, and matched on the way back out.
#: Without it a missing or unlaunchable engine binary is reported as a Python
#: traceback with exit code 1: the shim turns what used to be an ``OSError``
#: out of ``Popen`` in *this* process into a child that starts fine and then
#: dies, so the "Could not launch llama-server" path was unreachable on Linux
#: and the operator got `FileNotFoundError` from a stack they do not own.
PDEATHSIG_EXEC_FAILED = "studioforge-pdeathsig: exec failed: "

#: Exit code the shim uses for that case. 127 is the shell's convention for
#: "command not found", which is what this is.
PDEATHSIG_EXEC_FAILED_CODE = 127

_PDEATHSIG_SHIM = (
    "import ctypes, os, signal, sys; "
    "ctypes.CDLL(None, use_errno=True).prctl(1, int(signal.SIGKILL), 0, 0, 0); "
    "\ntry:\n"
    "    os.execv(sys.argv[1], sys.argv[1:])\n"
    "except OSError as exc:\n"
    f"    sys.stderr.write({PDEATHSIG_EXEC_FAILED!r} + str(exc) + chr(10))\n"
    f"    raise SystemExit({PDEATHSIG_EXEC_FAILED_CODE})\n"
)


def _pdeathsig_prefix() -> list[str]:
    """``[python, -c, shim]`` on Linux, ``[]`` elsewhere. See ``_PDEATHSIG_SHIM``."""
    if sys.platform != "linux" or os.environ.get("SF_NO_PDEATHSIG"):
        return []
    return [sys.executable, "-c", _PDEATHSIG_SHIM]


class Supervisor:
    """Owns every ``llama-server`` child process.

    ``resolve_binary`` is injected rather than imported so this module never
    depends on the engine manager's internals: anything that maps an optional
    engine tag to a server binary works, including a test stub.
    """

    def __init__(
        self,
        config: Config,
        *,
        resolve_binary: Callable[[str | None], Path],
        client: httpx.AsyncClient | None = None,
        launch_prefix: Sequence[str] = (),
        probe: GpuProbe | None = None,
        engine_features: Callable[[str | None], EngineFeatures] | None = None,
        resolve_engine_tag: Callable[[str | None], str | None] | None = None,
    ) -> None:
        self._config = config
        self._resolve_binary = resolve_binary
        # Third callable of the same injection family as resolve_binary and
        # engine_features: maps a REQUESTED tag (None = "the active engine") to
        # the tag that request actually lands on. It has to be separate from
        # resolve_binary because that one returns a Path and a path is not an
        # answer -- D50 needs the engine manager's own idea of the build, from
        # the same EngineInfo the binary came out of, so that "this child is
        # already on b10689" is a fact rather than a string parsed off a
        # directory name. The default does parse the directory name, because a
        # supervisor built without an engine manager (the GUI command preview,
        # every test) should still degrade to a plausible tag rather than None.
        self._resolve_engine_tag = resolve_engine_tag or self._engine_tag_from_binary
        # Monotonic count of deliberate starts. See InstanceInfo.spawn_seq: it
        # is what lets a queued forced reload tell "the child I was going to
        # restart" from "a child somebody restarted while I waited".
        self._spawn_seq = 0
        # Same injection story as resolve_binary: a callable, not the engine
        # manager, so this module keeps knowing nothing about it. The default
        # reads engines/<tag>/features.json (written at boot), falling back to
        # help.txt and finally to running --help once.
        self._resolve_features = engine_features or self._probe_features
        self._features: dict[str, EngineFeatures] = {}
        # Optional purely so a supervisor can be built without hardware; when
        # present it is used to log the VRAM an unload actually reclaimed.
        self._probe = probe
        self._client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._owns_client = client is None
        # Interpreter/wrapper args placed before the engine binary. Real
        # deployments leave this empty; tests use it to run a fake child.
        self._launch_prefix = list(launch_prefix)
        self._instances: dict[str, _Instance] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # The holder set the last cache_ram_pool_oversubscribed WARNING was
        # about, or None while the pool is not over-committed (D64, CR-6).
        self._cache_ram_oversub_warned: tuple[int, frozenset[tuple[str, int]]] | None = None
        self._ports_in_use: set[int] = set()
        self._unload_reports: dict[str, UnloadReport] = {}
        # Kernel-level "children die with me" net (Windows only). Held for the
        # supervisor's whole life: the guarantee IS the open handle.
        self._job = create_child_job()
        _register_atexit()

    # ------------------------------------------------------------------
    # Engine features
    # ------------------------------------------------------------------

    def _probe_features(self, tag: str | None) -> EngineFeatures:
        """Default feature lookup: read (or fill) the cache next to the binary."""
        try:
            binary = self._resolve_binary(tag)
        except Exception as exc:  # noqa: BLE001 - a missing engine fails at spawn, loudly
            log.warning("engine_features_unresolved", engine_tag=tag, error=str(exc))
            return EngineFeatures.unknown(tag or "")
        return probe_engine_features(binary, tag or "")

    async def features_for(self, tag: str | None) -> EngineFeatures:
        """The active engine's advertised feature set, memoised per tag.

        Awaited on the load path so the (rare) fallback that actually runs
        ``llama-server --help`` cannot block the event loop.
        """
        key = tag or ""
        cached = self._features.get(key)
        if cached is not None:
            return cached
        features = await asyncio.to_thread(self._resolve_features, tag)
        self._features[key] = features
        return features

    def _engine_tag_from_binary(self, tag: str | None) -> str | None:
        """Default ``resolve_engine_tag``: the build directory's own name.

        Engines live at ``engines/<tag>/llama-server.exe``, so the parent
        directory *is* the tag. Only a fallback -- the injected resolver reads
        it off the ``EngineInfo`` the binary was chosen from, which is the same
        answer without the layout assumption -- but a plausible tag beats
        ``None`` for a supervisor built without an engine manager, and ``None``
        is what every D50 comparison reads as "cannot prove this is current".
        """
        return self._resolve_binary(tag).parent.name or None

    def resolved_engine_tag(self, tag: str | None) -> str | None:
        """Which build a launch asking for ``tag`` would actually use.

        ``None`` when the engine cannot be identified -- nothing is installed,
        the pinned tag is missing, the resolver threw. Never raises: this is
        bookkeeping, and a load must not fail because we could not name the
        build it is about to run. The spawn itself fails loudly a few lines
        later if the binary is genuinely unresolvable.
        """
        try:
            resolved = self._resolve_engine_tag(tag)
        except Exception as exc:  # noqa: BLE001 - see the docstring
            log.warning("engine_tag_unresolved", engine_tag=tag, error=str(exc))
            return None
        return str(resolved) if resolved else None

    def active_engine_tag(self) -> str | None:
        """The build a load that pins nothing would launch with, or ``None``.

        The one question "activate + reload" has to answer to be idempotent:
        a resident whose :attr:`InstanceInfo.resolved_engine_tag` already equals
        this needs no reload at all (D50).
        """
        return self.resolved_engine_tag(None)

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def build_command(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        *,
        port: int,
        engine_tag: str | None = None,
        draft: ModelRecord | None = None,
        adapters: Sequence[tuple[AdapterRecord, float]] = (),
        features: EngineFeatures | None = None,
        resolved: ResolvedFeatures | None = None,
        cache_ram_mib: int | None = None,
    ) -> list[str]:
        """Build the full argv for one ``llama-server`` child.

        Ordering is deliberate: our own flags first, the user's expert
        ``extra_flags`` after them, so a deliberate override actually wins
        (llama.cpp takes the last occurrence of a repeated option) -- and the
        two flags the GPU-only policy rests on, ``--n-gpu-layers 999`` and
        ``--fit off``, come LAST of all, after ``extra_flags``, so last-wins
        can never be turned against them.

        The policy is enforced here as well as at save time (audit 2026-09-09
        §3): ``extra_flags`` is refused if it names a managed or CPU-offload
        family (:data:`studioforge.core.engine.POLICY_FAMILIES`), and the
        *final* argv is checked with :func:`launch_policy_violations` before
        it is returned. Save-time validation alone left a row written under
        an older engine, or straight into SQLite, applied unchecked forever.
        Raises :class:`ModelLoadError` naming the offending token; nothing is
        ever silently dropped, because a flag that vanished is the failure
        mode this module exists to prevent.

        See the module docstring for why ``--n-gpu-layers`` is hardcoded, why
        ``--ctx-size`` is multiplied by ``parallel``, and why ``--spec-type`` is
        mandatory for drafting.

        ``features`` is what the active engine advertises; omitted (as by the
        GUI's command-line preview and by tests) it defaults to "advertises
        nothing", which yields the pre-gating flag surface -- never a guess.
        ``resolved`` lets :meth:`_spawn` pass in a resolution it has already
        recorded on the plan instead of computing it twice. ``cache_ram_mib``
        is there for the same reason and matters more: under ``"auto"`` the
        ``--cache-ram`` grant depends on what the other live children hold, so
        :meth:`_spawn` computes it once, stamps it on the instance and passes it
        here -- the number in the argv and the number ``GET /api/models`` shows
        are then the same number, not two computations that can disagree.
        Omitted (preview, tests) it is computed from the current instance table.
        """
        settings = record.settings
        binary = self._resolve_binary(engine_tag or settings.engine_tag)
        engine = features if features is not None else EngineFeatures.unknown(engine_tag or "")
        decided = resolved if resolved is not None else self.resolve(record, plan, engine, draft)

        argv: list[str] = [
            str(binary),
            "--model",
            str(record.path),
            "--alias",
            record.id,
            "--host",
            CHILD_HOST,
            "--port",
            str(port),
            # TOTAL context across all slots -- see module docstring.
            "--ctx-size",
            str(plan.ctx_size * plan.parallel),
            "--parallel",
            str(plan.parallel),
        ]

        argv += self._placement_args(plan, decided)
        argv += ["--cache-type-k", plan.kv_cache_type, "--cache-type-v", plan.kv_cache_type_v]
        # b10425: --flash-attn takes a value (on|off|auto), it is not a switch.
        argv += ["--flash-attn", plan.flash_attn]
        # We proxy everything; the bundled web UI is dead weight, while the
        # introspection endpoints are what the Dashboard and planner feed on.
        argv += ["--no-webui", "--props", "--slots", "--metrics"]

        argv += self._optional_args(record, plan, engine, cache_ram_mib=cache_ram_mib)
        argv += self._concurrency_args(record, plan, engine)

        if record.kind == "embedding":
            argv.append("--embedding")
        elif record.mmproj_path is not None:
            # Vision projector. Never for embedding models -- llama-server
            # rejects the combination.
            argv += ["--mmproj", str(record.mmproj_path)]

        for adapter, scale in adapters:
            if abs(scale - 1.0) < 1e-9:
                argv += ["--lora", str(adapter.path)]
            else:
                argv += ["--lora-scaled", str(adapter.path), _fmt_float(scale)]

        argv += self._spec_args(record, plan, draft, decided, engine)

        if settings.extra_flags.strip():
            # posix=False on Windows so backslash paths survive verbatim.
            extra = shlex.split(settings.extra_flags, posix=(os.name != "nt"))
            # Enforcement point two of the GPU-only policy. The same table the
            # save-time validator uses, applied to what is about to launch:
            # this is the check that survives a stale row.
            refused = refuse_policy_flags(extra)
            if refused:
                raise ModelLoadError(
                    f"'{record.id}' cannot be launched: its saved extra_flags carry "
                    f"{len(refused)} flag(s) StudioForge refuses -- " + "; ".join(refused),
                    details={
                        "model_id": record.id,
                        "refused": refused,
                        "extra_flags": redact_argv(extra),
                    },
                )
            argv += extra

        # GPU-only: never conditional, never computed -- and LAST, after the
        # user's flags, so llama.cpp's last-wins rule can only ever land on
        # these two. ``--fit`` (default ON since upstream PR #16653, merged
        # Dec 2025) "adjusts unset arguments to fit in device memory"; the
        # planner already decided placement and context against live free
        # VRAM, so engine-side auto-adjustment is at best redundant and at
        # worst a silent partial-offload path -- exactly the degradation the
        # GPU-only policy exists to prevent. Off, and a genuine over-commit
        # fails loudly instead (D11). Verified accepted by b10425.
        argv += ["--n-gpu-layers", ALL_GPU_LAYERS, "--fit", "off"]

        # The final argv, read the way the child will read it. Anything that
        # got past the token check above -- there is no such path today, and
        # this is what keeps it that way -- is refused with its name.
        violations = launch_policy_violations(argv, settings)
        if violations:
            raise ModelLoadError(
                f"'{record.id}' cannot be launched: the final command line contradicts the "
                f"GPU-only policy ({', '.join(violations)})",
                details={
                    "model_id": record.id,
                    "policy_violations": violations,
                    "argv": redact_argv(argv),
                },
            )
        return argv

    def resolve(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        features: EngineFeatures,
        draft: ModelRecord | None = None,
    ) -> ResolvedFeatures:
        """Resolve the ``auto``/gated knobs for a launch. See D38."""
        return resolve_launch_features(record, plan, features, has_draft=draft is not None)

    def _placement_args(self, plan: LoadPlan, decided: ResolvedFeatures) -> list[str]:
        """Device selection, split mode and main GPU.

        ``--main-gpu`` indexes the *filtered* device list produced by
        ``--device``, not the physical CUDA ordinal: passing ``--device CUDA1
        --main-gpu 1`` makes llama.cpp reject the load with an out-of-range main
        GPU. So the physical ordinal in the plan is translated to its position.

        The split mode comes from ``decided``, not from the plan: the planner
        cannot judge ``tensor`` because it does not know what the engine offers
        (see :func:`resolve_split_mode`).
        """
        devices = list(plan.devices) or [0]
        device_arg = ",".join(f"CUDA{i}" for i in devices)
        if len(devices) == 1:
            return ["--device", device_arg, "--main-gpu", "0", "--split-mode", "none"]

        args = ["--device", device_arg]
        if plan.tensor_split:
            args += ["--tensor-split", ",".join(_fmt_float(v) for v in plan.tensor_split)]
        args += ["--split-mode", decided.split_mode]
        # --main-gpu is only read for split modes 'none' and 'row', but passing
        # it under tensor mode is harmless and keeps the argv shape stable.
        main = devices.index(plan.main_gpu) if plan.main_gpu in devices else 0
        args += ["--main-gpu", str(main)]
        return args

    def _cache_ram_grant(self, *, exclude: str | None = None) -> int | None:
        """MiB for this child's ``--cache-ram``, or ``None`` to pass nothing.

        Two settings, two scopes (D50). An explicit ``engine.cache_ram_mb``
        integer is the operator's own number and every child gets it verbatim,
        including ``0`` (off) and ``-1`` (no limit) -- D14 says an explicit
        value is honoured, and the total is then theirs to reason about.

        ``"auto"`` is a machine-wide POOL, and this is where it is divided. The
        old reading handed the full pool to every child, so the comment
        promising the cache "can never be the reason the machine starts
        swapping" was only true for one resident: four of them on this rig
        promised 4 x 32 GiB of a 128 GiB box. Each spawn now gets what the other
        LIVE children were not granted, floored so that the last one in still
        has a cache at all. ``exclude`` drops the arriving child's own row --
        :meth:`start` inserts it before :meth:`_spawn` runs, and a child must
        not be asked to share with itself.

        Unlimited grants (``-1``) are counted as zero rather than as a negative
        share: you cannot subtract "all of it" from a pool, and the operator who
        typed ``-1`` opted out of the accounting anyway.
        """
        configured = self._config.engine.cache_ram_mb
        resolved = resolve_cache_ram_mb(configured)
        if configured != "auto" or resolved is None:
            return resolved
        holders = frozenset(
            (model_id, int(inst.info.cache_ram_mib or 0))
            for model_id, inst in self._instances.items()
            if model_id != exclude and (inst.info.cache_ram_mib or 0) > 0
        )
        held = sum(mib for _, mib in holders)
        grant = grant_cache_ram_mib(resolved, held)
        if held + grant > resolved:
            # The floor won, which means the pool is now an intention rather
            # than a bound. Say so with the numbers: this is the one line that
            # tells an operator staring at a swapping box that the automatic
            # cache is over-committed and the fix is a smaller explicit value.
            #
            # Once per holder set, not once per spawn (D64, CR-6). D50 accepts
            # that the first resident onto an empty pool takes all of it, so
            # with one long-lived chat model holding the pool EVERY other load
            # meets the floor -- 700 identical WARNINGs in 13 days on the rig,
            # which is noise that hides the one worth reading. The next WARNING
            # comes when who holds the pool (or the pool) changes, or after the
            # pool stopped being over-committed; the repeats stay at DEBUG.
            key = (int(resolved), holders)
            fields = {
                "pool_mib": resolved,
                "held_mib": held,
                "grant_mib": grant,
                "floor_mib": CACHE_RAM_MIN_GRANT_MIB,
                "residents": len(self._instances) - (1 if exclude in self._instances else 0),
                "holders": sorted(f"{model_id}={mib}" for model_id, mib in holders),
            }
            if key != self._cache_ram_oversub_warned:
                self._cache_ram_oversub_warned = key
                log.warning("cache_ram_pool_oversubscribed", **fields)
            else:
                log.debug("cache_ram_pool_oversubscribed", repeat=True, **fields)
        else:
            self._cache_ram_oversub_warned = None
        return grant

    def _optional_args(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        features: EngineFeatures,
        *,
        cache_ram_mib: int | None = None,
    ) -> list[str]:
        settings = record.settings
        args: list[str] = []

        if settings.batch_size is not None:
            args += ["--batch-size", str(settings.batch_size)]
        ubatch = self.ubatch_for(record, max(1, plan.parallel))
        if ubatch is not None:
            args += ["--ubatch-size", str(ubatch)]
        if settings.threads is not None:
            args += ["--threads", str(settings.threads)]
        if settings.threads_batch is not None:
            args += ["--threads-batch", str(settings.threads_batch)]
        if settings.cont_batching is True:
            args.append("--cont-batching")
        elif settings.cont_batching is False:
            # Continuous batching is the engine's DEFAULT, so "off" needs a
            # flag of its own -- and until D54 nothing emitted one, which left
            # the GUI's tri-state toggle with an "off" position that did
            # nothing. Same D17 rule that dropped --defrag-thold: a switch the
            # child cannot see must not look honoured, so where the engine has
            # no --no-cont-batching the load says so instead of staying quiet.
            if features.has("--no-cont-batching"):
                args.append("--no-cont-batching")
            else:
                log.warning(
                    "setting_inert",
                    model_id=record.id,
                    setting="cont_batching",
                    value=False,
                    flag="--no-cont-batching",
                    engine_known=features.known,
                    detail=(
                        "the engine does not advertise --no-cont-batching, so "
                        "continuous batching stays at the engine default (on); "
                        "the instance's `effective` block lists the setting as inert"
                    ),
                )

        # Prompt-cache reuse is ON by default: OpenClaw re-sends near-identical
        # long agent prompts constantly, and reusing the cached prefix is the
        # single biggest real-world latency win available here.
        cache_reuse = (
            settings.cache_reuse
            if settings.cache_reuse is not None
            else self._config.models.default_cache_reuse
        )
        if cache_reuse and cache_reuse > 0:
            args += ["--cache-reuse", str(cache_reuse)]

        # Host-RAM prompt cache. The other half of the same win as
        # --cache-reuse: reuse recovers a prefix that is still in the slot,
        # --cache-ram keeps a prefix that has been evicted from one, in system
        # memory, so the next request that carries it does not re-prefill.
        # Costs no VRAM (measured identical at 8192 and 32768 MiB) and cannot
        # change a token, so it is on by default under the quality-first rule.
        # It does cost host RAM, though, which is why the automatic setting is
        # a shared pool rather than a per-child allowance -- see
        # _cache_ram_grant. The precomputed value wins so that the argv and the
        # instance's recorded grant cannot drift apart.
        if features.cache_ram:
            cache_ram = cache_ram_mib if cache_ram_mib is not None else self._cache_ram_grant()
            if cache_ram is not None:
                args += ["--cache-ram", str(cache_ram)]

        # GPU-side sampling. Marked EXPERIMENTAL by b10425 and silently
        # downgraded ("backend sampling not supported with SPLIT_MODE_TENSOR;
        # using CPU") under tensor split, so it stays opt-in.
        if self._config.engine.backend_sampling and features.backend_sampling:
            args.append("--backend-sampling")

        # Reasoning/thinking models: llama.cpp defaults --reasoning-format to
        # 'auto', which routes thoughts into message.reasoning_content and can
        # leave message.content EMPTY. Verified against DeepSeek-R1-0528-Qwen3-8B
        # on b10425: content len 0, reasoning_content len 316. Any OpenAI client
        # (OpenClaw included) then sees an empty reply. 'none' keeps the thoughts
        # inline in content, which also keeps SSE pass-through correct. See
        # DECISIONS.md D12.
        reasoning_format = (
            settings.reasoning_format
            if settings.reasoning_format is not None
            else self._config.models.default_reasoning_format
        )
        if reasoning_format:
            args += ["--reasoning-format", reasoning_format]
        if settings.reasoning is not None:
            args += ["--reasoning", settings.reasoning]
        if settings.reasoning_budget is not None:
            args += ["--reasoning-budget", str(settings.reasoning_budget)]

        # Per-model chat-template override. The GGUF's own template is used by
        # default and is right almost always -- but "almost" is the problem: a
        # baked-in template the engine cannot compile (Jinja `raise_exception`
        # is the known case) turns certain request shapes into a 400 with no
        # way out. This flag is that way out, and it is never set implicitly.
        if settings.chat_template_file is not None:
            args += ["--chat-template-file", str(settings.chat_template_file)]

        if settings.no_context_shift:
            args.append("--no-context-shift")
        # --defrag-thold is deliberately NOT emitted: b10425 marks it
        # deprecated, and passing a deprecated flag to keep a stored setting
        # "working" is how a value quietly stops meaning anything. The field
        # survives on ModelSettings only so old rows and the GUI form keep
        # loading; see types.ModelSettings.defrag_thold.
        args += self._load_mode_args(record, features)

        if settings.rope_freq_base is not None:
            args += ["--rope-freq-base", _fmt_float(settings.rope_freq_base)]
        if settings.rope_freq_scale is not None:
            args += ["--rope-freq-scale", _fmt_float(settings.rope_freq_scale)]
        if settings.rope_scaling is not None:
            args += ["--rope-scaling", settings.rope_scaling]

        # Sampler defaults are only emitted when explicitly set; otherwise the
        # engine's own defaults apply and per-request values stay in charge.
        if settings.temperature is not None:
            args += ["--temp", _fmt_float(settings.temperature)]
        if settings.top_p is not None:
            args += ["--top-p", _fmt_float(settings.top_p)]
        if settings.top_k is not None:
            args += ["--top-k", str(settings.top_k)]
        if settings.min_p is not None:
            args += ["--min-p", _fmt_float(settings.min_p)]
        if settings.repeat_penalty is not None:
            args += ["--repeat-penalty", _fmt_float(settings.repeat_penalty)]

        return args

    def _load_mode_args(self, record: ModelRecord, features: EngineFeatures) -> list[str]:
        """``settings.mlock`` / ``settings.no_mmap`` in the active engine's spelling.

        b10689 marks ``--mlock`` and ``--mmap``/``--no-mmap`` "DEPRECATED in
        favor of ``--load-mode``" -- still accepted today, gone some release
        soon, and the removed-flag detector keys on a different phrase, so
        nothing would warn before every load with either setting hard-failed
        on an unknown argument (audit 2026-09-09, F7). So, D38's rule:

        * an engine that advertises ``--load-mode`` gets the one flag. The
          old pair mapped onto its modes, faithfully: ``--mlock`` alone kept
          the default mmap and locked the mapping, which is ``mmap+mlock``;
          ``--no-mmap`` alone is ``none``; both together read the file into
          locked memory, which is ``mlock``;
        * an engine that advertises the deprecated pair gets the pair;
        * a known engine with neither logs ``setting_inert`` per setting and
          passes nothing -- a switch the child cannot see must not look
          honoured; :func:`effective_launch` lists it as inert;
        * an unknown engine (help unreadable) keeps the pre-gating surface.
        """
        settings = record.settings
        if not (settings.mlock or settings.no_mmap):
            return []
        if features.has("--load-mode"):
            if settings.mlock and settings.no_mmap:
                mode = "mlock"
            elif settings.mlock:
                mode = "mmap+mlock"
            else:
                mode = "none"
            return ["--load-mode", mode]
        args: list[str] = []
        for enabled, flag, setting in (
            (settings.mlock, "--mlock", "mlock"),
            (settings.no_mmap, "--no-mmap", "no_mmap"),
        ):
            if not enabled:
                continue
            if not features.known or features.has(flag):
                args.append(flag)
                continue
            log.warning(
                "setting_inert",
                model_id=record.id,
                setting=setting,
                value=True,
                flag=flag,
                engine_known=features.known,
                detail=(
                    f"the engine advertises neither {flag} nor --load-mode, so the "
                    "setting cannot be passed; the instance's `effective` block lists "
                    "it as inert"
                ),
            )
        return args

    def ubatch_for(self, record: ModelRecord, slots: int = 1) -> int | None:
        """The ``-ub`` this launch passes, or ``None`` to keep the engine's 512.

        Shares :func:`effective_ubatch` with ``Planner.ubatch_for`` so the VRAM
        the planner charged for the micro-batch is the micro-batch the child
        runs -- including the automatic many-slots raise (D38 §5), which is why
        ``slots`` is threaded in from ``plan.parallel``.
        """
        return effective_ubatch(
            settings_ubatch=record.settings.ubatch_size,
            engine_ubatch=self._config.engine.ubatch_size,
            engine_ubatch_many_slots=self._config.engine.ubatch_many_slots,
            slots=slots,
        )

    def _concurrency_args(
        self, record: ModelRecord, plan: LoadPlan, features: EngineFeatures
    ) -> list[str]:
        """Flags that only make sense once a model serves more than one slot.

        Emitted from the *plan*, not from settings, because the slot count can
        be chosen by the planner (DECISIONS.md D17) and these have to follow
        whatever it decided rather than whatever was saved.

        None of them is on at one slot, which is what keeps the default launch
        byte-identical to before the estimator existed.
        """
        settings = record.settings
        slots = max(1, plan.parallel)
        args: list[str] = []

        # The default 2048-token logical batch is shared across slots, so past
        # roughly four of them prompt ingestion starts serialising for no
        # reason. Only when the user has not pinned a batch size: llama.cpp
        # takes the last occurrence of a repeated flag, so appending here would
        # otherwise silently overrule an explicit setting.
        #
        # The logical batch must also be at least the micro-batch: llama.cpp
        # clamps ``n_ubatch`` to ``n_batch``, so ``-ub 4096`` against the
        # default ``-b 2048`` would silently run at 2048 -- a flag that looks
        # like it did something. An explicit batch_size is still honoured
        # verbatim (the user chose both numbers); the automatic one is raised
        # to cover the micro-batch.
        auto_batch = BATCH_SIZE_MANY_SLOTS if slots > 4 else ENGINE_DEFAULT_BATCH_SIZE
        ubatch = self.ubatch_for(record, slots) or 0
        if settings.batch_size is None and (slots > 4 or ubatch > auto_batch):
            args += ["--batch-size", str(max(auto_batch, ubatch))]

        # Slot affinity. With --cache-reuse on, routing a request to the slot
        # that already holds a similar prefix is what makes the prompt cache
        # pay off; the 0.10 default is loose enough to scatter an agent's
        # near-identical prompts across slots and lose the cache each time.
        if slots > 1:
            args += ["--slot-prompt-similarity", _fmt_float(SLOT_PROMPT_SIMILARITY_MULTI)]

        # KV pool shape across slots. Measured on b10425 with the 0.5B, two
        # slots and --ctx-size 16384 (DECISIONS.md D38):
        #
        #   nothing passed  -> kv_unified='false', n_ctx_slot 8192; a 12k-token
        #                      request is refused up front with a 400 naming the
        #                      limit. `ctx_per_slot` is a real guarantee.
        #   --kv-unified    -> kv_unified='true',  n_ctx_slot 16384, SAME VRAM
        #                      (997 vs 1005 MiB). A lone request reaches the
        #                      whole pool -- but two 12k requests at once both
        #                      died mid-generation with a 500 "Context size has
        #                      been exceeded".
        #
        # An agent host would rather be told "no" before the work starts than be
        # 500ed halfway through one of four concurrent agents, so the partitioned
        # pool stays the default and `--no-kv-unified` is passed *explicitly*:
        # the engine's own default is "enabled if the slot count is auto", and a
        # guarantee that depends on a flag we happen to pass is not a guarantee.
        if settings.kv_unified:
            args.append("--kv-unified")
        elif slots > 1 and features.has("--no-kv-unified"):
            args.append("--no-kv-unified")

        return args

    def _spec_args(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        draft: ModelRecord | None,
        decided: ResolvedFeatures,
        features: EngineFeatures,
    ) -> list[str]:
        """Speculative decoding flags, b10425 spelling.

        ``--spec-type`` defaults to ``none``; without it a draft model is loaded
        (costing VRAM) and then never used, which is invisible except as
        "speculation did not help". The old ``--draft*`` names are accepted and
        ignored by b10425, so they must never appear here.

        The draft-model flags are emitted only for a type that actually uses
        one: ``draft-mtp`` reads the base model's own heads and ``ngram-*`` read
        the generated text, so pointing either at a ``--spec-draft-model`` would
        load a second model for nothing.
        """
        settings = record.settings
        if not decided.drafting:
            return []
        if features.known and not features.has("--spec-type"):  # pragma: no cover - ancient build
            return []

        args: list[str] = ["--spec-type", decided.spec_type]
        if decided.spec_draft_n_max is not None:
            args += ["--spec-draft-n-max", str(decided.spec_draft_n_max)]
        if settings.spec_draft_n_min is not None:
            args += ["--spec-draft-n-min", str(settings.spec_draft_n_min)]
        if settings.spec_draft_p_min is not None:
            args += ["--spec-draft-p-min", _fmt_float(settings.spec_draft_p_min)]

        if draft is None:
            return args

        args += ["--spec-draft-model", str(draft.path)]
        # The draft model is GPU-only too; a CPU-resident draft is slower than
        # not drafting at all.
        args += ["--spec-draft-ngl", ALL_GPU_LAYERS]
        draft_devices = settings.draft_device_override or plan.devices or [0]
        args += ["--spec-draft-device", ",".join(f"CUDA{i}" for i in draft_devices)]
        # "auto" is OUR sentinel, resolved by the planner; llama-server has no
        # such value and would refuse to start. The draft cache type is not
        # planned, so pass it through only when it is a real type.
        if draft.settings.kv_cache_type not in (None, "auto"):
            args += ["--spec-draft-type-k", str(draft.settings.kv_cache_type)]
        if draft.settings.kv_cache_type_v not in (None, "auto"):
            args += ["--spec-draft-type-v", str(draft.settings.kv_cache_type_v)]
        return args

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------

    def _allocate_port(self, preferred: int | None = None) -> int:
        gateway = self._config.gateway
        candidates: Iterable[int] = range(gateway.child_port_start, gateway.child_port_end + 1)
        if preferred is not None:
            candidates = [preferred, *candidates]
        for port in candidates:
            if port in self._ports_in_use:
                continue
            if not _port_is_bindable(port):
                continue
            self._ports_in_use.add(port)
            return port
        raise ModelLoadError(
            "No free port in the llama-server child range "
            f"{gateway.child_port_start}-{gateway.child_port_end}; "
            "widen gateway.child_port_start/child_port_end or unload models."
        )

    def _release_port(self, port: int | None) -> None:
        if port is not None:
            self._ports_in_use.discard(port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _lock(self, model_id: str) -> asyncio.Lock:
        lock = self._locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[model_id] = lock
        return lock

    def _prune_lock(self, model_id: str) -> None:
        """Drop the per-model lock once its instance is gone and nobody holds it.

        Keeps the lock table from growing one entry per model id ever started.
        A waiter woken between our ``locked()`` check and its own resume would
        find its lock discarded and a later caller on a fresh one -- benign
        here, because every operation re-checks ``self._instances`` after
        acquiring, so two locks can never produce two children for one id.
        """
        lock = self._locks.get(model_id)
        if lock is not None and not lock.locked() and model_id not in self._instances:
            del self._locks[model_id]

    async def start(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        *,
        engine_tag: str | None = None,
        draft: ModelRecord | None = None,
        adapters: Sequence[tuple[AdapterRecord, float]] = (),
        source: str | None = None,
        priority: int = 3,
    ) -> InstanceInfo:
        """Launch (or return) the child for ``record``; returns once /health is ok.

        ``source`` names the caller ("mcp:load_model", "jit:/v1/chat/completions",
        "gui"...). It is stamped on the instance and carried into the spawn and
        ready log lines so a model appearing on the GPUs can be traced back to
        whoever asked for it (D36). ``priority`` is the load's tier (D46),
        stamped the same way; the crash-relaunch loop reuses this instance's
        info, so the tier survives a restart.

        Every child that gets here also gets a ``spawn_seq`` (D50). It counts
        *deliberate* starts, not processes -- the crash-relaunch loop reuses
        this same info and leaves the number alone -- because its one reader is
        a forced reload asking "did somebody already do the restart I queued
        for?", and a crash is not somebody doing it.
        """
        async with self._lock(record.id):
            existing = self._instances.get(record.id)
            if existing is not None and existing.info.state in ("ready", "loading"):
                # A concurrent caller already started this model.
                return existing.info

            tag = engine_tag or record.settings.engine_tag
            port = self._allocate_port()
            self._spawn_seq += 1
            info = InstanceInfo(
                model_id=record.id,
                state="loading",
                port=port,
                engine_tag=tag,
                spawn_seq=self._spawn_seq,
                plan=plan,
                ttl_s=record.settings.ttl_s,
                loaded_by=source,
                priority=priority,
                log_path=self._log_path_for(record.id),
            )
            inst = _Instance(
                info=info,
                record=record,
                plan=plan,
                port=port,
                engine_tag=tag,
                draft=draft,
                adapters=adapters,
                log_path=self._log_path_for(record.id),
            )
            self._instances[record.id] = inst
            try:
                await self._spawn(inst)
                await self._await_ready(inst)
            except BaseException as exc:
                inst.stopping = True
                inst.info.state = "failed"
                if isinstance(exc, ModelLoadError):
                    inst.info.last_error = exc.message
                await self._teardown(inst, timeout=5.0, force=True)
                # A child that crashed during startup was logged by
                # _await_ready; one torn down here (a health timeout, a port
                # squatter, a cancelled load) ends now, and says so (D64).
                self._log_child_exit(
                    inst, phase="startup", cause=f"torn down after a failed start ({_why(exc)})"
                )
                # Only live children stay in the table; the diagnostics travel
                # with the raised ModelLoadError (stderr tail + argv).
                self._instances.pop(record.id, None)
                self._release_port(port)
                raise

            inst.info.state = "ready"
            inst.info.started_at = time.time()
            inst.info.last_activity_at = time.time()
            inst.watcher = asyncio.create_task(self._watch(inst), name=f"sf-watch-{record.id}")
            # ``engine_tag`` here is the build this child is RUNNING, not the
            # build it asked for. Until D50 it was the request, which is None
            # for almost every load -- so the 2026-08-30 log said
            # `engine_tag=None` on both halves of a double reload and no reader
            # could tell that the second round was pure churn. The request is
            # only worth a field of its own when it was an actual per-model pin.
            extra: dict[str, Any] = {"pinned_engine_tag": tag} if tag else {}
            log.info(
                "model_ready",
                model_id=record.id,
                port=port,
                pid=inst.info.pid,
                engine_tag=inst.info.resolved_engine_tag,
                spawn_seq=inst.info.spawn_seq,
                source=source,
                **extra,
            )
            return inst.info

    async def _spawn(self, inst: _Instance) -> None:
        """Create the child process and start pumping its output."""
        features = await self.features_for(inst.engine_tag)
        # Which build this child will actually be, recorded before the argv is
        # even built (D50): `inst.engine_tag` is the request, and a request of
        # None -- what almost every load carries -- names nothing at all.
        inst.info.resolved_engine_tag = self.resolved_engine_tag(inst.engine_tag)
        # The host prompt cache comes out of a shared pool under "auto", so the
        # grant depends on what the other live children hold and has to be
        # computed here, once, rather than independently by whoever asks. Only
        # when the engine actually offers the flag: a grant nobody spends is a
        # phantom holding down everybody else's share.
        inst.info.cache_ram_mib = (
            self._cache_ram_grant(exclude=inst.record.id) if features.cache_ram else None
        )
        decided = self.resolve(inst.record, inst.plan, features, inst.draft)
        self._record_resolution(inst, decided)
        argv = self.build_command(
            inst.record,
            inst.plan,
            port=inst.port,
            engine_tag=inst.engine_tag,
            draft=inst.draft,
            adapters=inst.adapters,
            features=features,
            resolved=decided,
            cache_ram_mib=inst.info.cache_ram_mib,
        )
        inst.argv = argv
        # What the child will REALLY run with, readable on every instance
        # view (D54). Parsed from the final argv rather than re-derived from
        # settings, so extra_flags and engine defaults are in the answer too.
        inst.info.effective = effective_launch(argv, features, inst.plan, inst.record.settings)
        inst.info.launch_args = redact_argv(argv)
        if inst.info.effective.policy_violations:
            # build_command refuses these, so this cannot fire today; it is
            # the report's own tripwire for the day the two checks disagree.
            log.warning(
                "gpu_only_policy_violation",
                model_id=inst.record.id,
                violations=list(inst.info.effective.policy_violations),
                detail="the child is being launched with flags that contradict the GPU-only policy",
            )
        # The shim must be the OUTERMOST element, ahead of the launch prefix
        # as well as the engine argv: its body is
        # `os.execv(sys.argv[1], sys.argv[1:])`, so whatever follows it is
        # taken as a complete command line to exec. With the prefix first, a
        # prefix of `[python, -u]` swallowed `-c <shim>` as its own script
        # argument and then tried to compile the interpreter binary --
        # "SyntaxError: source code cannot contain null bytes" out of an ELF
        # header. PDEATHSIG survives execve, so wrapping the prefix too keeps
        # the guarantee for every layer. Production never noticed because
        # `_launch_prefix` is empty there; every launch-prefix test on Linux
        # did (26 of them).
        full_argv = [*_pdeathsig_prefix(), *self._launch_prefix, *argv]
        inst.open_log()
        # Redacted, not raw: this file is served by GET /api/logs/models/{id}
        # (D55). The header still names every flag and every basename, which is
        # what makes a failed launch diagnosable; what it no longer names is the
        # operator's username, the disk layout and anything key-shaped that
        # reached extra_flags.
        inst.write_log(f"=== studioforge launch: {' '.join(redact_argv(full_argv))}")

        kwargs: dict[str, Any] = {}
        # Start suspended when there is a job to put the child in, so the window
        # between "process exists" and "process is in the job" contains no
        # executed instruction at all. Assigning right after Popen would leave a
        # sub-millisecond gap in which a hard kill of this process could strand a
        # child -- vanishingly unlikely, but the gap is avoidable and this is the
        # code whose entire job is to make the guarantee unconditional. A child
        # that is never resumed has allocated nothing, so the worst case of this
        # path is strictly better than the worst case of the other one.
        suspended = os.name == "nt" and self._job is not None and self._job.available
        if os.name == "nt":
            # New process group so a console Ctrl+C does not race our own
            # orderly shutdown, and so the whole group can be signalled.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            if suspended:
                kwargs["creationflags"] |= CREATE_SUSPENDED
        else:
            kwargs["start_new_session"] = True

        # Run from the engine directory: that is the Windows DLL search rule,
        # and the CUDA/ggml DLLs ship alongside llama-server. It does nothing
        # for Linux -- ld.so never searches the working directory -- so there
        # the .so files beside the binary are found through its $ORIGIN
        # RUNPATH instead: upstream's ubuntu archives carry one, and the
        # source build (engine.build_from_source) bakes the same one in.
        engine_dir = Path(argv[0]).parent
        # Enforcement point three of the GPU-only policy: the environment.
        # b10689 reads an ``LLAMA_ARG_*`` variable for nearly every flag, and
        # for a flag StudioForge never emits -- which is every CPU-offload
        # flag -- the variable is wholly unopposed by the argv. The child
        # gets our environment minus that surface (engine.child_environment),
        # and the names stripped are logged so an operator who set one on
        # purpose learns why it did nothing.
        env, stripped = child_environment()
        if stripped:
            log.warning(
                "child_env_stripped",
                model_id=inst.record.id,
                names=stripped,
                detail=(
                    "llama-server would read these as launch flags; the GPU-only policy "
                    "is enforced on the argv alone, so they are not passed to the child"
                ),
            )
        log.info(
            "model_spawn",
            model_id=inst.record.id,
            port=inst.port,
            source=inst.info.loaded_by,
            # The ring buffer behind GET /api/logs. Same reason as the child
            # log header above (D55); ``inst.info.launch_args`` is the very
            # same redaction, already computed.
            argv=" ".join(inst.info.launch_args or redact_argv(argv)),
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(engine_dir) if engine_dir.is_dir() else None,
                env=env,
                **kwargs,
            )
        except OSError as exc:
            raise ModelLoadError(
                f"Could not launch llama-server for '{inst.record.id}': {exc}",
                details={"argv": redact_argv(argv)},
            ) from exc

        if self._job is not None:
            # Best-effort by design: assign() logs and returns False rather than
            # raising, because an unprotected child still serves inference.
            self._job.assign(proc.pid)
        if suspended:
            self._resume(inst, proc)

        inst.proc = proc
        inst.info.pid = proc.pid
        inst.info.state = "loading"
        inst.create_time = process_create_time(proc.pid)
        _TRACKED_PIDS.add(proc.pid)
        inst.pumps = []
        if proc.stdout is not None:
            inst.pumps.append(asyncio.create_task(self._pump(inst, proc.stdout, stderr=False)))
        if proc.stderr is not None:
            inst.pumps.append(asyncio.create_task(self._pump(inst, proc.stderr, stderr=True)))
        inst.wait_task = asyncio.create_task(proc.wait(), name=f"sf-wait-{inst.record.id}")

    @staticmethod
    def _record_resolution(inst: _Instance, decided: ResolvedFeatures) -> None:
        """Publish the resolved knobs on the plan and the instance.

        The plan object is the one carried by ``InstanceInfo.plan``, so writing
        the *resolved* split mode back onto it is what makes every surface that
        renders a placement -- the catalog, the Dashboard, ``GET /api/models`` --
        show what the child is really doing rather than what was asked for.
        """
        speculative = decided.speculative_dict(
            draft_model_id=inst.draft.id if inst.draft is not None else None
        )
        inst.plan.speculative = speculative
        inst.info.speculative = speculative
        if len(inst.plan.devices) > 1:
            inst.plan.split_mode = decided.split_mode  # type: ignore[assignment]
        inst.plan.split_mode_reason = decided.split_mode_reason
        if decided.split_mode_reason and decided.split_mode_reason not in inst.plan.notes:
            inst.plan.notes.append(decided.split_mode_reason)

    @staticmethod
    def _resume(inst: _Instance, proc: asyncio.subprocess.Process) -> None:
        """Start the suspended child, or kill it and fail the load loudly.

        The one new failure mode ``CREATE_SUSPENDED`` introduces is a child that
        never runs. It must not be survivable-but-invisible: a suspended child
        holds a port and answers nothing, so the load would fail later as an
        unexplained health timeout. Kill it here and say why.
        """
        try:
            psutil.Process(proc.pid).resume()
        except Exception as exc:  # noqa: BLE001 - reported as a load failure below
            with contextlib.suppress(Exception):
                kill_process_tree(proc.pid, timeout=2.0, force=True)
            raise ModelLoadError(
                f"llama-server for '{inst.record.id}' was created suspended (so it could "
                f"be placed in this process's job object) and could not be resumed: {exc}",
                details={"pid": proc.pid, "argv": inst.argv},
            ) from exc

    async def _pump(self, inst: _Instance, stream: asyncio.StreamReader, *, stderr: bool) -> None:
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            except Exception as exc:  # noqa: BLE001 - transport teardown races
                # Said out loud: from here on ``stderr_ring`` stops filling,
                # so a later failure message degrades to "No output captured"
                # on precisely the child being debugged (audit 2026-09-09
                # §2.11). The line says which stream, and why.
                log.warning(
                    "child_output_pump_ended",
                    model_id=inst.record.id,
                    stream="stderr" if stderr else "stdout",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if stderr:
                inst.stderr_ring.append(line)
            inst.write_log(line)

    async def _await_ready(self, inst: _Instance) -> None:
        """Poll ``GET /health`` until ok, the process dies, or we time out."""
        gateway = self._config.gateway
        deadline = time.monotonic() + gateway.load_timeout_s
        interval = max(0.05, gateway.health_poll_interval_s)
        url = f"http://{CHILD_HOST}:{inst.port}/health"

        while True:
            code = inst.proc.returncode if inst.proc is not None else None
            if code is not None:
                # Fail fast: waiting out a 10 minute timeout for a process that
                # already exited buries the actual error.
                await self._drain_pumps(inst)
                # A child that dies while loading is a child that exited, and
                # it used to leave no model_exited line at all -- only the load
                # failure, several frames up. Two of four crash dumps in the
                # 2026-09 window had nothing else (D64, CR-7).
                self._log_child_exit(inst, phase="startup", code=code)
                # The shim could not exec the engine at all. That is a launch
                # failure, not a crashed engine, and it must read like the
                # `OSError` from `create_subprocess_exec` it replaced --
                # otherwise a mistyped engine path surfaces as somebody else's
                # Python traceback.
                for line in inst.stderr_tail():
                    if PDEATHSIG_EXEC_FAILED in line:
                        reason = line.split(PDEATHSIG_EXEC_FAILED, 1)[1].strip()
                        raise ModelLoadError(
                            f"Could not launch llama-server for '{inst.record.id}': {reason}",
                            details={"argv": inst.argv},
                        )
                raise ModelLoadError(
                    self._failure_message(inst, f"exited with code {code} during startup"),
                    details={
                        "exit_code": code,
                        "stderr": inst.stderr_tail(),
                        "argv": inst.argv,
                    },
                )
            try:
                resp = await self._client.get(url, timeout=_HTTP_TIMEOUT)
                if resp.status_code == 200 and await self._confirm_identity(inst):
                    return
            except (httpx.HTTPError, OSError):
                pass

            if time.monotonic() >= deadline:
                raise ModelLoadError(
                    self._failure_message(
                        inst, f"did not become healthy within {gateway.load_timeout_s:g}s"
                    ),
                    details={
                        "stderr": inst.stderr_tail(),
                        "argv": inst.argv,
                        "port_conflict": inst.port_conflict,
                    },
                )
            await asyncio.sleep(interval)

    @staticmethod
    def _expected_alias(inst: _Instance) -> str:
        """The alias we actually launched with (extra_flags may override ours)."""
        alias = inst.record.id
        for index, token in enumerate(inst.argv[:-1]):
            if token == "--alias":
                alias = inst.argv[index + 1]
        return alias

    async def _confirm_identity(self, inst: _Instance) -> bool:
        """Check that the healthy server on our port is really *our* child.

        A stale llama-server (or anything else) squatting on the port answers
        ``/health`` perfectly happily, and adopting it would mean proxying
        requests to a process we do not control, with the wrong model and the
        wrong context size -- a failure that looks like a mystery bug rather
        than a port clash. ``--alias`` is echoed by ``/props``, so it doubles as
        an identity check. When ``/props`` cannot be read we do not block the
        load; the goal is catching an impostor, not adding a hard dependency.

        Every fail-open branch is logged at WARNING with its reason (audit
        2026-09-09 §2.4): the fail-open is deliberate, but a squatter that
        answers ``/health`` and not ``/props`` used to be adopted as a
        successful load with no line anywhere saying the check never ran.
        """
        base = f"http://{CHILD_HOST}:{inst.port}"

        def fail_open(reason: str) -> bool:
            log.warning(
                "child_identity_unchecked",
                model_id=inst.record.id,
                port=inst.port,
                reason=reason,
                detail="/props could not confirm the alias; the child is adopted unverified",
            )
            return True

        try:
            resp = await self._client.get(f"{base}/props", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError) as exc:
            return fail_open(f"/props unreachable: {type(exc).__name__}: {exc}")
        if resp.status_code != 200:
            return fail_open(f"/props answered HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            return fail_open("/props answered non-JSON")
        if not isinstance(data, dict):
            return fail_open(f"/props answered a JSON {type(data).__name__}, not an object")
        alias = data.get("model_alias")
        expected = self._expected_alias(inst)
        if isinstance(alias, str) and alias != expected:
            inst.port_conflict = alias
            log.error(
                "child_port_conflict",
                model_id=inst.record.id,
                port=inst.port,
                foreign_alias=alias,
            )
            return False
        if not isinstance(alias, str):
            return fail_open(
                "/props carries no model_alias string"
                if alias is None
                else f"/props model_alias is a {type(alias).__name__}"
            )
        inst.port_conflict = None
        return True

    def _failure_message(self, inst: _Instance, what: str) -> str:
        tail = inst.stderr_tail()
        text = f"llama-server for '{inst.record.id}' {what}."
        if inst.port_conflict is not None:
            text += (
                f" Port {inst.port} is held by another server "
                f"(alias '{inst.port_conflict}'), not by this child."
            )
        if tail:
            text += " Last output:\n" + "\n".join(tail)
        else:
            text += f" No output captured; see {inst.log_path}."
        return text

    async def _drain_pumps(
        self,
        inst: _Instance,
        timeout: float = 2.0,  # noqa: ASYNC109 - drain deadline
    ) -> None:
        if not inst.pumps:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*inst.pumps, return_exceptions=True), timeout=timeout
            )

    def _log_child_exit(
        self,
        inst: _Instance,
        *,
        phase: str,
        code: int | None = None,
        cause: str | None = None,
    ) -> None:
        """Log ``model_exited`` for the instance's current process, once (D64, CR-7).

        Every path that notices a child is gone calls this -- the watcher, the
        startup health poll, a failed start's teardown, a failed relaunch, a
        watcher that itself failed -- and the first one wins: the pid is
        remembered, so the same exit is never reported twice and no exit is
        reported by nobody. A deliberate ``stop``/``kill`` is not an exit in
        this sense and logs ``model_stopped``/``model_killed`` instead.

        ``code`` defaults to the process's own ``returncode``; when neither is
        known the line says ``exit_code_unavailable`` rather than leaving the
        field out. ``phase`` is ``startup`` (never became ready), ``running``
        (was serving) or ``restart`` (a crash-relaunch that did not come up).
        """
        proc = inst.proc
        pid = proc.pid if proc is not None else inst.info.pid
        if pid is None or inst.exit_logged_pid == pid:
            return
        inst.exit_logged_pid = pid
        if code is None and proc is not None:
            code = proc.returncode
        fields: dict[str, Any] = {
            "model_id": inst.record.id,
            "pid": pid,
            "phase": phase,
            "restarts": inst.info.restarts,
            **describe_exit_code(code),
        }
        if cause is not None:
            fields["cause"] = cause
        log.warning("model_exited", **fields)

    async def _watch(self, inst: _Instance) -> None:
        """Restart the child on unexpected exit, with exponential backoff.

        A watcher that fails is itself a way for a child to vanish unreported:
        the task dies, its exception is never retrieved, and a crash after that
        is silence. So a failure here is logged with its traceback, and if the
        child is gone by then its exit is logged too and the instance marked
        failed; a child still running is left alone rather than orphaned into a
        second launch (D64, CR-7).
        """
        try:
            await self._watch_loop(inst)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception(
                "model_watch_failed", model_id=inst.record.id, error=f"{type(exc).__name__}: {exc}"
            )
            if inst.stopping:
                return
            proc = inst.proc
            if proc is None or proc.returncode is not None:
                self._log_child_exit(
                    inst, phase="running", cause=f"supervision failed: {type(exc).__name__}"
                )
                inst.info.state = "failed"
                inst.info.last_error = (
                    f"llama-server for '{inst.record.id}' is no longer supervised: "
                    f"{type(exc).__name__}: {exc}"
                )

    async def _wait_for_exit(self, inst: _Instance) -> int | None:
        """The child's exit code, or ``None`` when the OS wait itself failed.

        ``proc.wait()`` is the normal answer. If that task raises -- a transport
        torn down under it -- the child is watched by pid instead, so its end is
        still noticed and reported as ``exit_code_unavailable`` rather than
        never (D64, CR-7).
        """
        wait_task = inst.wait_task
        assert wait_task is not None
        try:
            # Shielded so cancelling the watcher (a deliberate stop) does not
            # cancel the underlying wait() that stop() itself needs to observe.
            return await asyncio.shield(wait_task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - fall back to watching the pid
            log.warning(
                "model_wait_failed",
                model_id=inst.record.id,
                pid=inst.info.pid,
                error=f"{type(exc).__name__}: {exc}",
                detail="watching the pid instead; the exit code will not be known",
            )
        pid = inst.info.pid
        while not inst.stopping:
            proc = inst.proc
            if proc is not None and proc.returncode is not None:
                return proc.returncode
            if pid is None or not await asyncio.to_thread(
                process_is_alive, pid, create_time=inst.create_time
            ):
                return None
            await asyncio.sleep(EXIT_POLL_S)
        return None

    async def _watch_loop(self, inst: _Instance) -> None:
        gateway = self._config.gateway
        while True:
            if inst.stopping or inst.wait_task is None:
                return
            phase = "running" if inst.info.state == "ready" else "restart"
            code = await self._wait_for_exit(inst)
            if inst.stopping:
                return

            await self._drain_pumps(inst)
            described = f"exited with code {code}" if code is not None else "exited"
            inst.info.last_error = self._failure_message(inst, described)
            _TRACKED_PIDS.discard(inst.info.pid or -1)
            self._log_child_exit(inst, phase=phase, code=code)

            if inst.info.restarts >= gateway.max_restarts:
                inst.info.state = "failed"
                log.error(
                    "model_restart_limit",
                    model_id=inst.record.id,
                    max_restarts=gateway.max_restarts,
                )
                return

            attempt = inst.info.restarts
            inst.info.restarts += 1
            inst.info.state = "loading"
            delay = gateway.restart_backoff_s * (2**attempt)
            await asyncio.sleep(delay)
            if inst.stopping:
                return

            # Reuse the same port when it is still ours and still free, so
            # anything caching the base URL keeps working across a restart.
            self._release_port(inst.port)
            try:
                inst.port = self._allocate_port(preferred=inst.port)
            except ModelLoadError as exc:
                inst.info.state = "failed"
                inst.info.last_error = exc.message
                return
            inst.info.port = inst.port

            try:
                await self._spawn(inst)
                if inst.stopping:
                    # stop() landed while create_subprocess_exec was in flight:
                    # the process exists but the teardown that stop() ran saw
                    # no proc to kill. Take it down here, on the far side of
                    # the await, or it outlives its watcher with the port and
                    # (on Linux, where it was not created suspended) the VRAM.
                    await self._teardown(inst, timeout=0.0, force=True)
                    return
                await self._await_ready(inst)
            except ModelLoadError as exc:
                inst.info.last_error = exc.message
                # A failed relaunch can leave a live child behind: _await_ready
                # times out on a hung load with the process still running.
                # start() tears that case down; without the same here the hung
                # llama-server would outlive its watcher, silently keeping its
                # port and -- far worse -- its VRAM. Teardown is a no-op for a
                # child that already exited.
                await self._teardown(inst, timeout=5.0, force=True)
                # One that crashed was logged by _await_ready; one killed here
                # for hanging ends now (D64, CR-7).
                self._log_child_exit(
                    inst, phase="restart", cause=f"torn down after a failed relaunch ({_why(exc)})"
                )
                if inst.info.restarts >= gateway.max_restarts:
                    inst.info.state = "failed"
                    return
                continue

            inst.info.state = "ready"
            inst.info.started_at = time.time()
            log.info(
                "model_restarted",
                model_id=inst.record.id,
                port=inst.port,
                restarts=inst.info.restarts,
            )

    async def _teardown(
        self,
        inst: _Instance,
        *,
        timeout: float,  # noqa: ASYNC109 - psutil.wait_procs grace period
        force: bool,
    ) -> None:
        """Cancel supervision tasks and make sure the process tree is gone."""
        if inst.watcher is not None and inst.watcher is not asyncio.current_task():
            inst.watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await inst.watcher
            inst.watcher = None

        proc = inst.proc
        if proc is not None and proc.returncode is None:
            await asyncio.to_thread(kill_process_tree, proc.pid, timeout=timeout, force=force)
        if inst.wait_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(inst.wait_task), timeout=5.0)
        await self._drain_pumps(inst)
        for pump in inst.pumps:
            pump.cancel()
        inst.pumps = []
        if inst.info.pid is not None:
            _TRACKED_PIDS.discard(inst.info.pid)
        inst.close_log()

    def _vram_used_bytes(self, devices: Sequence[int]) -> int:
        """Used VRAM across ``devices`` right now, or 0 without a probe."""
        if self._probe is None:
            return 0
        try:
            gpus = {gpu.index: gpu for gpu in self._probe.list_gpus()}
        except Exception:  # pragma: no cover - a probe must not break an unload
            return 0
        wanted = list(devices) or sorted(gpus)
        return sum(gpus[d].used_bytes for d in wanted if d in gpus)

    async def _verify_unloaded(self, inst: _Instance, pid: int | None, before: int) -> UnloadReport:
        """Confirm the child is really gone, and log the VRAM it gave back.

        A kill call that returned is not proof: the whole reason this project
        supervises processes itself is that an unload which *reports* success
        while the model stays resident leaves VRAM permanently spoken for. So
        the pid is re-checked, a survivor is escalated to an unconditional
        tree-kill, and the before/after VRAM numbers are recorded so "unloaded
        13.4 GiB" can be verified instead of assumed.
        """
        escalated = False
        alive = pid is not None and await asyncio.to_thread(
            process_is_alive, pid, create_time=inst.create_time
        )
        if alive and pid is not None:
            escalated = True
            log.warning(
                "unload_survivor",
                model_id=inst.record.id,
                pid=pid,
                detail="process still alive after teardown; escalating to a forced tree kill",
            )
            await asyncio.to_thread(kill_process_tree, pid, timeout=5.0, force=True)
            alive = await self._linger(inst, pid)

        if not alive and self._probe is not None:
            # The sample is what the settle is for; without a probe there is
            # nothing to sample and nothing to wait for.
            await self._settle_vram()
        report = UnloadReport(
            model_id=inst.record.id,
            pid=pid,
            pid_gone=not alive,
            escalated=escalated,
            vram_before_bytes=before,
            vram_after_bytes=self._vram_used_bytes(inst.plan.devices),
            at=time.time(),
        )
        self._unload_reports[inst.record.id] = report
        if report.pid_gone:
            log.info(
                "model_unload_verified",
                model_id=inst.record.id,
                pid=pid,
                escalated=escalated,
                vram_reclaimed_mb=round(report.vram_reclaimed_bytes / (1024 * 1024)),
            )
        else:
            log.error(
                "model_unload_unverified",
                model_id=inst.record.id,
                pid=pid,
                detail="process survived SIGTERM and SIGKILL; its VRAM is still held",
            )
        return report

    async def _linger(self, inst: _Instance, pid: int) -> bool:
        """Whether ``pid`` is still alive after up to :data:`UNLOAD_SETTLE_S`.

        Called only after every signal has been sent -- SIGTERM, SIGKILL, and
        the escalated tree kill -- so this is patience for a process that is
        already dying: on Windows a 20+ GB CUDA context takes the driver tens
        of seconds to tear down after ``TerminateProcess``, and declaring the
        unload failed in the middle of that (the old ~35 s chain, audit F1)
        was a spurious ``ModelUnloadError`` on an unload that was working.

        Polls every :data:`UNLOAD_POLL_S`. Stops early when the OS has already
        reported the exit (``wait_task`` done) yet the liveness check still
        says "alive": that is not a teardown in progress, it is a pid the
        check cannot account for, and waiting would only delay the honest
        answer.
        """
        deadline = time.monotonic() + UNLOAD_SETTLE_S
        started = time.monotonic()
        while True:
            alive = await asyncio.to_thread(process_is_alive, pid, create_time=inst.create_time)
            if not alive:
                waited = time.monotonic() - started
                if waited >= UNLOAD_POLL_S:
                    log.info(
                        "unload_settled",
                        model_id=inst.record.id,
                        pid=pid,
                        waited_s=round(waited, 1),
                        detail="the child exited after the forced kill; a slow driver teardown",
                    )
                return False
            wait_task = inst.wait_task
            if wait_task is not None and wait_task.done():
                return True
            if time.monotonic() >= deadline:
                return True
            await asyncio.sleep(UNLOAD_POLL_S)

    @property
    def vram_settle_s(self) -> float:
        """How long freed VRAM takes to show up in a probe (:data:`VRAM_SETTLE_S`).

        Exposed so the manager can await the same settle before the plans it
        computes right after an unload -- the OOM-retry re-plan and the lease
        handover -- instead of reading a card the driver has not handed back.
        """
        return VRAM_SETTLE_S

    async def _settle_vram(self) -> None:
        await asyncio.sleep(self.vram_settle_s)

    def unload_report(self, model_id: str) -> UnloadReport | None:
        """Evidence from the last unload of ``model_id``, if there was one."""
        return self._unload_reports.get(model_id)

    async def stop(
        self,
        model_id: str,
        *,
        timeout: float = 15.0,  # noqa: ASYNC109 - SIGTERM grace
    ) -> None:
        """Terminate the child gracefully, and verify that it actually died.

        Raises :class:`~studioforge.errors.ModelUnloadError` when the process
        outlives both signals: returning normally there would report freed VRAM
        that is still held, and every subsequent plan would be computed against
        a lie.
        """
        async with self._lock(model_id):
            inst = self._instances.get(model_id)
            if inst is not None:
                inst.stopping = True
                inst.info.state = "unloading"
                pid = inst.info.pid
                before = self._vram_used_bytes(inst.plan.devices)
                await self._teardown(inst, timeout=timeout, force=False)
                report = await self._verify_unloaded(inst, pid, before)
                if not report.pid_gone:
                    # Keep the instance in the table: it is still real, still
                    # holding VRAM, and hiding it would make the leak invisible.
                    inst.info.state = "failed"
                    inst.info.last_error = (
                        f"unload could not be verified: pid {pid} is still running"
                    )
                    raise ModelUnloadError(
                        f"Unloaded '{model_id}' but its llama-server process (pid {pid}) is "
                        "still alive, so its VRAM has not been reclaimed. Kill it manually "
                        "before loading anything else.",
                        details={"pid": pid, "model_id": model_id},
                    )
                inst.info.state = "stopped"
                inst.info.pid = None
                self._instances.pop(model_id, None)
                self._release_port(inst.port)
                log.info(
                    "model_stopped",
                    model_id=model_id,
                    vram_reclaimed_mb=round(report.vram_reclaimed_bytes / (1024 * 1024)),
                )
        self._prune_lock(model_id)

    async def stop_all(
        self,
        *,
        timeout: float = 15.0,  # noqa: ASYNC109 - per-child SIGTERM grace
    ) -> dict[str, BaseException | None]:
        """Stop every child, concurrently; report each one's outcome by name.

        Returns ``{model_id: None}`` for a verified unload and
        ``{model_id: exception}`` for one that failed -- a
        :class:`~studioforge.errors.ModelUnloadError` for a child that
        outlived every signal, still in the instance table and still holding
        its VRAM. Every failure is logged at ERROR here with the model id.

        Never raises for a failed unload: the two callers want different
        things from one. Process shutdown (``aclose``, the manager's drain)
        must carry on to the job-object close that is its real safety net,
        while ``POST /api/models/unload-all`` must turn any survivor into a
        500 -- so the manager's ``unload_all`` inspects this dict and raises
        an aggregate ``ModelUnloadError`` naming every survivor. Until it did,
        ``return_exceptions=True`` here discarded the result unread, and the
        route answered 200 with a cheerful count for children still alive on
        the GPUs (audit 2026-09-09 §2.1-2.3, headline finding 1).
        """
        ids = list(self._instances)
        results = await asyncio.gather(
            *(self.stop(model_id, timeout=timeout) for model_id in ids),
            return_exceptions=True,
        )
        outcome: dict[str, BaseException | None] = {}
        for model_id, result in zip(ids, results, strict=True):
            if not isinstance(result, BaseException):
                outcome[model_id] = None
                continue
            outcome[model_id] = result
            if isinstance(result, ModelUnloadError):
                log.error(
                    "model_unload_failed",
                    model_id=model_id,
                    pid=result.details.get("pid"),
                    error=result.message,
                    detail="still in the instance table, state failed; its VRAM is still held",
                )
            else:
                log.error(
                    "model_unload_failed",
                    model_id=model_id,
                    error=f"{type(result).__name__}: {result}",
                    detail="stop() raised something other than ModelUnloadError",
                )
        return outcome

    async def kill(self, model_id: str) -> bool:
        """Hard-kill the child immediately, without draining requests.

        Returns True only when the process is *verified* gone -- a survivor
        reports False and stays in the instance table rather than being
        forgotten while it still holds VRAM.
        """
        killed = False
        async with self._lock(model_id):
            inst = self._instances.get(model_id)
            if inst is not None:
                inst.stopping = True
                inst.info.state = "unloading"
                pid = inst.info.pid
                before = self._vram_used_bytes(inst.plan.devices)
                await self._teardown(inst, timeout=0.0, force=True)
                report = await self._verify_unloaded(inst, pid, before)
                if not report.pid_gone:
                    inst.info.state = "failed"
                    inst.info.last_error = f"kill could not be verified: pid {pid} is still running"
                else:
                    inst.info.state = "stopped"
                    inst.info.pid = None
                    self._instances.pop(model_id, None)
                    self._release_port(inst.port)
                    log.warning("model_killed", model_id=model_id)
                    killed = True
        self._prune_lock(model_id)
        return killed

    async def aclose(self) -> None:
        await self.stop_all(timeout=5.0)
        if self._owns_client:
            await self._client.aclose()
        # Closing the job kills anything still in it. Deliberately last, and
        # deliberately after stop_all: by here every child should already be
        # gone, so this only catches a survivor -- which is the point.
        if self._job is not None:
            self._job.close()

    def child_pids(self) -> set[int]:
        """Pids of the children this supervisor currently owns.

        Used by the orphan sweep to tell "ours" from "someone else's": a
        llama-server under our engines dir that this set does not contain is
        either another live process's child or a leak. See
        :mod:`studioforge.core.vram_holders`.
        """
        return {inst.info.pid for inst in self._instances.values() if inst.info.pid is not None}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> InstanceInfo | None:
        inst = self._instances.get(model_id)
        return inst.info if inst is not None else None

    def list(self) -> builtins.list[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]

    def is_ready(self, model_id: str) -> bool:
        inst = self._instances.get(model_id)
        return inst is not None and inst.info.state == "ready"

    def base_url(self, model_id: str) -> str | None:
        inst = self._instances.get(model_id)
        if inst is None or inst.port is None:
            return None
        return f"http://{CHILD_HOST}:{inst.port}"

    def _log_path_for(self, model_id: str) -> Path:
        return self._config.model_logs_dir / f"{safe_log_name(model_id)}.log"

    def log_path(self, model_id: str) -> Path | None:
        inst = self._instances.get(model_id)
        if inst is not None:
            return inst.log_path
        path = self._log_path_for(model_id)
        return path if path.is_file() else None

    def tail_log(self, model_id: str, n: int = 200) -> builtins.list[str]:
        path = self.log_path(model_id)
        if path is None or not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return [line.rstrip("\n") for line in deque(handle, maxlen=n)]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Child HTTP proxies (never raise: the Dashboard polls these constantly)
    # ------------------------------------------------------------------

    async def _get_json(self, model_id: str, path: str) -> Any:
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return None
        try:
            resp = await self._client.get(f"{base}{path}", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def props(self, model_id: str) -> dict[str, Any] | None:
        data = await self._get_json(model_id, "/props")
        return data if isinstance(data, dict) else None

    async def slots(self, model_id: str) -> builtins.list[dict[str, Any]] | None:
        data = await self._get_json(model_id, "/slots")
        return data if isinstance(data, list) else None

    async def metrics(self, model_id: str) -> str | None:
        """Raw Prometheus text from the child's ``/metrics``, or ``None``.

        Enabled by the ``--metrics`` flag every child is launched with. Returns
        text rather than a parsed structure because parsing belongs to
        :func:`studioforge.core.throughput.parse_metrics`, which is unit-tested
        against real exposition output; the supervisor's job here is only to
        fetch it without ever raising -- this is called from a background timer
        that must not die because a child is mid-restart.
        """
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return None
        try:
            resp = await self._client.get(f"{base}/metrics", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return None
        if resp.status_code != 200:
            return None
        return resp.text

    async def health(self, model_id: str) -> bool:
        base = self.base_url(model_id)
        if base is None:
            return False
        try:
            resp = await self._client.get(f"{base}/health", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return False
        return resp.status_code == 200

    async def set_lora_scales(self, model_id: str, scales: builtins.list[dict[str, Any]]) -> bool:
        """Hot-adjust LoRA scales via ``POST /lora-adapters``."""
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return False
        try:
            resp = await self._client.post(
                f"{base}/lora-adapters", json=scales, timeout=_HTTP_TIMEOUT
            )
        except (httpx.HTTPError, OSError):
            return False
        return resp.status_code == 200

    # ------------------------------------------------------------------
    # Activity accounting (feeds TTL unloading and the Dashboard)
    # ------------------------------------------------------------------

    def mark_request_start(self, model_id: str) -> None:
        inst = self._instances.get(model_id)
        if inst is None:
            return
        inst.info.active_requests += 1
        inst.info.total_requests += 1
        inst.info.last_activity_at = time.time()

    def mark_request_end(self, model_id: str, *, tokens_per_second: float | None = None) -> None:
        inst = self._instances.get(model_id)
        if inst is None:
            return
        inst.info.active_requests = max(0, inst.info.active_requests - 1)
        inst.info.last_activity_at = time.time()
        if tokens_per_second is not None:
            inst.info.last_tokens_per_second = tokens_per_second
