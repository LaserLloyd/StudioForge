"""Configuration: a single config.yaml plus a few SF_-prefixed env variables.

Layering (lowest priority first): field defaults -> ``SF_*`` environment ->
config.yaml. The file wins over the environment for every key it names because
``save()`` writes every key: an env override that outranked the file would be
silently undone by the first Setup-tab edit and reappear on the next restart,
which is worse than either order alone. The environment therefore only decides
things the file does not carry -- ``SF_DATA_DIR`` (D25) and ``SF_CONFIG`` --
plus a first-run default for anything not yet written.

The resolved config is written back on ``save()`` so the GUI/MCP ``set_config``
surfaces and hand edits round-trip through the same file. The file lives
*outside* the release directories so self-update never clobbers it, and
``data_dir`` itself is never written into it (see :meth:`Config.to_yaml_dict`).
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from studioforge.errors import ConfigError

#: "auto" is resolved by the planner per model: it takes the best-quality KV
#: cache that still reaches the chosen context on the hardware available.
KvCacheType = Literal["auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
#: ``auto`` is StudioForge's own sentinel, resolved at launch by the supervisor:
#: ``tensor`` when every gate passes (engine offers it, >= 2 devices, dense
#: non-hybrid model, flash attention on, unquantized KV) and ``layer`` with a
#: logged reason otherwise. ``layer`` remains the default because tensor mode is
#: EXPERIMENTAL upstream and measured *slower* than layer on this PCIe rig
#: (DECISIONS.md D38). ``row`` is accepted by the flag parser but the CUDA
#: backend rejects it at load time ("device CUDAn does not support split
#: buffers") -- see docs/LIMITATIONS.md.
SplitMode = Literal["none", "layer", "row", "tensor", "auto"]
FlashAttn = Literal["on", "off", "auto"]

#: Ceiling for the automatic host-RAM prompt cache, in MiB. 32 GiB on a 128 GiB
#: box: big enough to hold many agent prefixes, small enough that the cache can
#: never be the reason the machine starts swapping.
CACHE_RAM_AUTO_MAX_MIB = 32768
#: Fraction of system RAM the automatic setting will claim.
CACHE_RAM_AUTO_FRACTION = 0.25
#: What ``cache_ram_mb: "auto"`` falls back to when system RAM cannot be read.
CACHE_RAM_AUTO_FALLBACK_MIB = 8192

APP_NAME = "studioforge"


def _checkout_data_dir() -> Path | None:
    """``<repo>/data`` when this package is running from a source checkout.

    A checkout is recognised by a ``pyproject.toml`` two levels above this
    module (``<repo>/src/studioforge/config.py``). An installed wheel lives in
    site-packages, where that file does not exist, so a packaged install still
    gets the platform directory D7 describes -- and an editable checkout keeps
    its data next to itself, where ``.gitignore`` covers it and where the
    launchers, the justfile and the docs all agree it is.
    """
    root = Path(__file__).resolve().parents[2]
    if (root / "pyproject.toml").is_file() and (root / "src" / APP_NAME).is_dir():
        return root / "data"
    return None


def default_data_dir() -> Path:
    """Where the data directory lives, in one sentence.

    ``SF_DATA_DIR`` if it is set; else ``<repo>/data`` when running from a
    source checkout; else the platform data directory
    (``%LOCALAPPDATA%\\studioforge`` / ``~/.local/share/studioforge``).

    Holds engines/, logs/, registry.sqlite3 and config.yaml -- everything that
    must survive an app self-update, which is why it is never *inside* a
    release directory (DECISIONS.md D7).
    """
    env = os.environ.get("SF_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    checkout = _checkout_data_dir()
    if checkout is not None:
        return checkout
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME")
    base_path = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base_path / APP_NAME


def lmstudio_model_dir_candidates() -> list[Path]:
    """Probe order for an existing LM Studio model library.

    LM Studio records a relocated library in ``settings.json``
    (``downloadsFolder``), which is checked first -- users with models on a
    second drive are the common case, and the default path is then empty.
    """
    candidates: list[Path] = []
    home = Path.home()
    settings = home / ".lmstudio" / "settings.json"
    if settings.is_file():
        try:
            data = json.loads(settings.read_text(encoding="utf-8"))
            folder = data.get("downloadsFolder")
            if isinstance(folder, str) and folder.strip():
                candidates.append(Path(folder))
        except Exception:
            pass
    candidates.append(home / ".lmstudio" / "models")
    candidates.append(home / ".cache" / "lm-studio" / "models")
    if os.name == "nt":
        profile = os.environ.get("USERPROFILE")
        if profile:
            candidates.append(Path(profile) / ".lmstudio" / "models")
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.expanduser()
        except Exception:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def detect_model_dir() -> Path | None:
    """First candidate directory that exists and contains at least one GGUF."""
    for candidate in lmstudio_model_dir_candidates():
        if not candidate.is_dir():
            continue
        try:
            next(candidate.rglob("*.gguf"))
        except StopIteration:
            continue
        except OSError:
            continue
        return candidate
    # Fall back to an existing-but-empty LM Studio dir so new downloads land
    # where LM Studio will also see them.
    for candidate in lmstudio_model_dir_candidates():
        if candidate.is_dir():
            return candidate
    return None


#: A TCP port. ``0`` is excluded on purpose: "pick any free port" is meaningless
#: for a server whose clients are configured with the number.
Port = Annotated[int, Field(ge=1, le=65535)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: Port = 1234  # LM Studio's default port: OpenClaw only changes the host
    # >= 6 chars so the log-redaction processor will register it: shorter
    # values are ignored there, and an unredactable key can surface inside a
    # logged command line or an httpx error repr.
    api_key: str | None = Field(default=None, min_length=6)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False
    drain_timeout_s: NonNegativeFloat = 30.0
    request_timeout_s: PositiveFloat = 900.0

    @model_validator(mode="after")
    def _no_credentialed_wildcard_cors(self) -> ServerConfig:
        """``["*"]`` + ``allow_credentials`` is the one CORS pair browsers treat
        as "any website may make credentialed requests here". The shipped default
        (wildcard, credentials off) is safe; flipping credentials on without
        narrowing the origins would silently open the API to every page the
        operator visits. Refuse the pair at load time (WP17 review, open item 1).
        """
        if self.cors_allow_credentials and any(o.strip() == "*" for o in self.cors_origins):
            raise ValueError(
                "server.cors_allow_credentials=true requires explicit server.cors_origins "
                "(no '*'): a credentialed wildcard lets any website call this API as you"
            )
        return self


class GuiConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: Port = 8080
    refresh_interval_s: PositiveFloat = 2.0


class WatchdogConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: Port = 1235
    poll_interval_s: PositiveFloat = 10.0
    health_timeout_s: PositiveFloat = 5.0
    wedged_after_failures: PositiveInt = 3


#: config.yaml is rewritten while the watchdog may have it open for reading.
_SAVE_RETRIES = 6
_SAVE_RETRY_DELAY_S = 0.05


class ModelsConfig(BaseModel):
    dir: Path | None = None
    extra_dirs: list[Path] = Field(default_factory=list)
    default_ctx: PositiveInt = 8192
    default_ttl_s: NonNegativeInt = 1800  # 30 min; 0 = pinned/never unload
    # "auto": the planner picks per model rather than forcing one type on the
    # whole library. At long context the KV cache dwarfs the weights, so the
    # right trade differs per model -- a 27B reaches native 262144 on f16 and
    # needs no quantization at all, while a 31B dense only reaches 65536 even
    # at q8_0. Auto keeps full-quality KV wherever it is affordable.
    default_kv_cache_type: KvCacheType = "auto"
    # "auto": the planner sizes the slot count per model and per placement (see
    # DECISIONS.md D17) instead of forcing every load to one conversation. An
    # explicit integer -- here, per model, or per request -- is still honoured
    # verbatim (the D14 "explicit value is honoured" invariant), and 1 keeps the
    # old behaviour exactly.
    default_parallel: int | Literal["auto"] = "auto"
    #: Per-slot context the parallel estimator assumes when nothing else says.
    #: Only used to *size* the slot count and to build the catalog's
    #: loading-options table; it never overrides a planned or explicit ctx_size.
    ctx_per_slot_default: PositiveInt = 32768
    # "on" rather than "auto": flash attention is a large KV-bandwidth and
    # memory win and is supported on every GPU from Ampere (sm_80) up, which
    # is all of this rig. "auto" defers the same decision to the engine and
    # has been observed to decline it. tune_for_hardware drops this back to
    # "auto" if a pre-Ampere card is ever detected.
    default_flash_attn: FlashAttn = "on"
    default_cache_reuse: NonNegativeInt = 256  # prompt-cache reuse: the OpenClaw latency win
    # "none" keeps a reasoning model's thoughts inline in message.content.
    # llama.cpp's default ("auto") splits them into reasoning_content and leaves
    # content empty, which reads as an empty reply to every OpenAI client.
    default_reasoning_format: Literal["none", "deepseek", "deepseek-legacy"] = "none"
    # Reasoning models spend their budget thinking before answering, so the
    # ordinary default truncates the chain of thought AND the visible answer.
    # Applied only when a thinking model has no explicit ctx_size, clamped to
    # the model's trained window and to what actually fits.
    thinking_default_ctx: PositiveInt = 32768
    # What every model AIMS for when nobody asks for a specific size. The
    # planner walks down from here -- halving -- to the largest window that
    # actually fits in VRAM, never below ``default_ctx``. Agent workloads
    # (OpenClaw) carry long tool transcripts and run out of room at 8k, so the
    # aim is high and the step-down is what keeps big models loadable.
    target_ctx: PositiveInt = 1048576
    auto_load_pinned: bool = True
    # Served when a request omits "model", or names one of DEFAULT_MODEL_ALIASES
    # ("local-model", "default", "auto"). LM Studio clients send "local-model"
    # as a fallback, so 404-ing it breaks them for no good reason.
    default_model: str | None = None
    # Load the default model at startup so the first request is not also the
    # first (slow) load.
    preload_default_model: bool = False

    # NOTE: ``max_loaded`` was removed. It was declared here and never read by
    # anything, so it read as a working cap while VRAM was the only real limit
    # -- a config key that silently does nothing is worse than no key at all.
    # An existing config.yaml carrying it still loads: pydantic ignores unknown
    # keys on this model, so no migration is needed.

    @field_validator("default_parallel")
    @classmethod
    def _sane_parallel(cls, v: int | str) -> int | str:
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("models.default_parallel must be a positive int or 'auto'")
            return v
        if v < 1:
            raise ValueError("models.default_parallel must be >= 1")
        return v

    @field_validator("dir")
    @classmethod
    def _dir_is_not_a_file(cls, v: Path | None) -> Path | None:
        # A missing directory is fine (it may be created, or be on a drive not
        # mounted yet); a path that exists and is a *file* can never be a
        # library, and scanning it silently found nothing.
        if v is not None and Path(v).is_file():
            raise ValueError(f"models.dir points at a file, not a directory: {v}")
        return v


class EngineConfig(BaseModel):
    pinned_tag: str = "b10425"
    cuda_variant: str = "auto"  # e.g. "12.4" / "13.3" / "auto"
    keep_versions: PositiveInt = 3
    allow_source_build: bool = True
    repo: str = "ggml-org/llama.cpp"
    smoke_test_timeout_s: PositiveFloat = 180.0

    #: ``--cache-ram``: host-RAM prompt cache, in MiB. Costs no VRAM (measured:
    #: identical VRAM at 8192 and 32768) and is quality-neutral -- it only lets
    #: the engine keep evicted prompt prefixes in system memory instead of
    #: recomputing them, which is exactly the OpenClaw pattern of re-sending a
    #: long, near-identical agent prompt. ``"auto"`` = 25% of system RAM capped
    #: at 32 GiB; ``0`` disables it; ``-1`` is the engine's "no limit".
    cache_ram_mb: int | Literal["auto"] = "auto"
    #: ``-ub/--ubatch-size`` default for every model that does not set its own.
    #: ``None`` leaves the engine's 512. Raising it buys prompt-processing
    #: speed for compute-buffer VRAM -- measured on a 1.5B on one RTX 3090 with
    #: a 5166-token prompt: 512 -> 15232 tok/s at 1492 MiB, 1024 -> 17307 tok/s
    #: at 1562 MiB, 2048 -> 18061 tok/s at 1702 MiB. The planner charges that
    #: growth per device (``planner.ubatch_scratch_bytes``, D40), so raising it
    #: costs context rather than risking an OOM; it stays unset because it is a
    #: VRAM-for-prefill trade the operator should make knowingly, not a default.
    #:
    #: Spelled ``int | None`` rather than ``PositiveInt | None`` deliberately:
    #: an *optional* Annotated int survives into the pydantic annotation as
    #: ``Optional[Annotated[int, Gt(0)]]``, which the Setup tab's field-spec
    #: generator classifies as unsupported and silently drops from the form.
    #: The bound lives in the validator below instead.
    ubatch_size: int | None = None
    #: ``-bs/--backend-sampling``: run sampling on the GPU. Faster, but marked
    #: EXPERIMENTAL by b10425 and silently downgraded to CPU sampling under
    #: ``--split-mode tensor``. Off by default under the quality-first rule --
    #: only features that are lossless *and* not experimental ship enabled.
    backend_sampling: bool = False

    @field_validator("ubatch_size")
    @classmethod
    def _positive_ubatch(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("engine.ubatch_size must be >= 1 (null = the engine's own 512)")
        return v

    @field_validator("cache_ram_mb")
    @classmethod
    def _sane_cache_ram(cls, v: int | str) -> int | str:
        if isinstance(v, str):
            if v != "auto":
                raise ValueError("engine.cache_ram_mb must be an int (MiB), -1, 0, or 'auto'")
            return v
        if v < -1:
            raise ValueError("engine.cache_ram_mb must be >= -1 (-1 = no limit, 0 = disabled)")
        return v


def resolve_cache_ram_mb(value: int | str) -> int | None:
    """Turn ``engine.cache_ram_mb`` into the number that reaches ``--cache-ram``.

    ``None`` means "pass nothing" -- reserved for a value we cannot resolve, so
    the engine keeps its own default rather than being handed a guess.
    """
    if isinstance(value, int):
        return value
    try:
        import psutil

        total_mib = int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:  # pragma: no cover - psutil is a hard dep, but never fail here
        return CACHE_RAM_AUTO_FALLBACK_MIB
    if total_mib <= 0:  # pragma: no cover - defensive
        return CACHE_RAM_AUTO_FALLBACK_MIB
    return max(1024, min(CACHE_RAM_AUTO_MAX_MIB, int(total_mib * CACHE_RAM_AUTO_FRACTION)))


class QuantAffinity(BaseModel):
    """Hardware affinity for a quantization family.

    ``mode="prefer"`` steers placement toward capable GPUs but still allows the
    others; ``mode="require"`` refuses to place the model anywhere else.
    """

    min_compute_capability: str = "0.0"
    mode: Literal["prefer", "require"] = "prefer"

    @property
    def min_cc_tuple(self) -> tuple[int, int]:
        try:
            major, _, minor = self.min_compute_capability.partition(".")
            return (int(major), int(minor or 0))
        except ValueError:
            return (0, 0)


def default_quant_affinity() -> dict[str, QuantAffinity]:
    """Measured defaults for FP4 quant families.

    Benchmarked on the reference rig (RTX 5090 sm_120 vs RTX 3090 sm_86,
    gemma-4-31B, llama.cpp b10425), prompt processing at 512 tokens::

        NVFP4 : 5992 tok/s on sm_120  vs   987 tok/s on sm_86   (6.1x)
        Q4_0  : 4109 tok/s on sm_120  vs  1367 tok/s on sm_86   (3.0x)

    NVFP4 gets roughly double the relative Blackwell speedup, i.e. real native
    FP4 tensor-core acceleration. Crucially it still *runs* on Ampere -- so this
    is a preference, not a hardware requirement. It is worth steering, though:
    on Ampere the NVFP4 build is ~28% SLOWER at prompt processing than a plain
    Q4_0 of the same model, so an FP4 file placed on a 3090 is a bad trade
    unless nothing else is free.
    """
    return {
        "NVFP4": QuantAffinity(min_compute_capability="12.0", mode="prefer"),
        "MXFP4": QuantAffinity(min_compute_capability="12.0", mode="prefer"),
    }


class PlannerConfig(BaseModel):
    headroom_fraction: float = 0.10
    on_insufficient: Literal["evict", "reject"] = "evict"
    # Compute/graph buffers scale with batch and model width; calibrated against
    # observed loads and refined by predicted-vs-actual logging. See DECISIONS.md.
    compute_overhead_fraction: NonNegativeFloat = 0.06
    compute_overhead_floor_mb: NonNegativeInt = 400
    cuda_context_mb: NonNegativeInt = 300  # per-GPU CUDA context + cuBLAS workspace
    image_tokens_default: PositiveInt = 1024
    mmproj_compute_mb: NonNegativeInt = 512
    prefer_single_gpu: bool = True
    #: What the catalog optimises for when it names one load per model (D36).
    #:
    #: ``"quality"`` (the default): the best KV cache quality that reaches the
    #: context floor, then the largest context at that quality, then whatever
    #: slots still fit. ``"throughput"`` is D20's original rule -- the largest
    #: context at or above the floor, preferring one that also sustains two
    #: slots. The two disagree exactly where the user cares: on this rig the
    #: quality rule takes 131072 tokens with an f16 cache over 262144 with a
    #: quantized one, and a Gemma-4 measures a KL divergence of 0.108 between
    #: those two caches (see core/kv_sensitivity.py).
    preference: Literal["quality", "throughput"] = "quality"
    # Quant family -> hardware affinity. Keys are matched case-insensitively
    # against the model's quant label. See default_quant_affinity().
    quant_affinity: dict[str, QuantAffinity] = Field(default_factory=default_quant_affinity)

    # --- multi-tenant GPU sharing --------------------------------------
    #: CUDA indices the planner may never place a model on, for a box that also
    #: runs something else on the GPU (ComfyUI, a training job). ``headroom_fraction``
    #: cannot express this: it is a percentage of every card, so reserving one
    #: whole card for a neighbour meant starving all of them. Empty by default --
    #: this is a policy decision, and the shipped default must not make one.
    #: A per-model ``device_override`` still wins (explicit beats policy), with a
    #: warning, because the user naming a device is the user deciding.
    excluded_devices: list[int] = Field(default_factory=list)
    #: CUDA index -> MiB held back on that card, subtracted inside
    #: :meth:`Planner.usable_bytes` alongside the headroom. The softer half of
    #: the same knob: "leave ComfyUI 8 GB on CUDA3" rather than "never touch
    #: CUDA3". Unlike exclusion this applies to forced placements too -- a
    #: reservation is about the *neighbour's* memory, not about our policy.
    reserved_mb: dict[int, int] = Field(default_factory=dict)

    @field_validator("headroom_fraction")
    @classmethod
    def _check_headroom(cls, v: float) -> float:
        if not 0.0 <= v < 0.9:
            raise ValueError("headroom_fraction must be in [0.0, 0.9)")
        return v

    @field_validator("excluded_devices")
    @classmethod
    def _check_excluded(cls, v: list[int]) -> list[int]:
        for index in v:
            if index < 0:
                raise ValueError("planner.excluded_devices entries must be >= 0")
        # Deduplicated and ordered so the value is stable through a save/load
        # round-trip and so a doubled entry cannot read as "extra excluded".
        return sorted(set(v))

    @field_validator("reserved_mb")
    @classmethod
    def _check_reserved(cls, v: dict[int, int]) -> dict[int, int]:
        for index, amount in v.items():
            if index < 0:
                raise ValueError("planner.reserved_mb keys must be CUDA indices >= 0")
            if amount < 0:
                raise ValueError(f"planner.reserved_mb[{index}] must be >= 0 MiB (got {amount})")
        return v


class McpConfig(BaseModel):
    """Model Context Protocol surface.

    The PIN is a **pairing code for the MCP endpoint only**, deliberately
    separate from ``server.api_key``: it is short enough to read off a startup
    banner and type into a client, it is printed in the log on purpose, and it
    grants access to the management tools rather than to inference. The API key
    remains the stronger credential and still works everywhere.
    """

    enabled: bool = True
    path: str = "/mcp"
    #: Auto-generated on first run when empty. Set to null / pin_required=False
    #: to fall back to the API key alone.
    pin: str | None = None
    pin_required: bool = True
    #: Print the reachable MCP URLs (and the PIN) in the startup banner.
    advertise: bool = True

    @field_validator("path")
    @classmethod
    def _rooted_path(cls, v: str) -> str:
        # Starlette asserts a route path starts with "/"; a bare "mcp" made
        # the mount fail, which the app swallowed as "management MCP not
        # mounted" -- so MCP was silently absent, the PIN was never enforced,
        # and the banner advertised a URL that 404s. Normalise instead.
        cleaned = (v or "").strip()
        if not cleaned or cleaned == "/":
            raise ValueError("mcp.path must be a non-root path such as '/mcp'")
        if not cleaned.startswith("/"):
            cleaned = "/" + cleaned
        return cleaned.rstrip("/") or "/mcp"

    @field_validator("pin")
    @classmethod
    def _sane_pin(cls, v: str | None) -> str | None:
        if v is None:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if len(cleaned) < 6:
            raise ValueError("mcp.pin must be at least 6 characters")
        return cleaned


# ---------------------------------------------------------------------------
# Credential redaction (a leaf: the watchdog may import this module, never api.*)
# ---------------------------------------------------------------------------


def redact(value: str | None) -> str | None:
    """Short prefix of a key, for logs and GUI display."""
    if not value:
        return None
    return f"{value[:4]}...{value[-2:]}" if len(value) > 8 else "***"


#: Every dotted config path whose value is a credential, and must therefore
#: never leave the process in full through a config-dumping surface.
#:
#: ``mcp.pin`` belongs here and was missing from all four dump sites (the
#: ``/api/config`` route, the management-MCP ``get_config`` tool, the watchdog's
#: ``get_config`` tool and ``studioforge config``). Each had grown its own copy
#: of the same two-key redaction, so a third secret had to be remembered four
#: times and was remembered zero. On the shipped default -- ``server.api_key``
#: unset, so the whole API is open -- that turned ``GET /api/config`` into a
#: free read of the PIN that is the ONLY credential on the MCP control plane.
SECRET_CONFIG_PATHS: tuple[tuple[str, str], ...] = (
    ("server", "api_key"),
    ("mcp", "pin"),
    ("hf", "token"),
)


def redact_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Replace every credential in a ``Config.to_yaml_dict()`` with a fingerprint.

    Mutates and returns ``data``. Driven by :data:`SECRET_CONFIG_PATHS` so a new
    secret is declared once rather than in every surface that dumps config.
    """
    for section_name, key in SECRET_CONFIG_PATHS:
        section = data.get(section_name)
        if isinstance(section, dict) and section.get(key):
            section[key] = redact(str(section[key]))
    return data


def generate_pin(length: int = 8) -> str:
    """A readable pairing PIN.

    Digits only, and generated with ``secrets`` rather than ``random``: it is a
    credential, even if a short-lived, LAN-scoped one. Eight digits is 10^8 --
    weak against an unthrottled remote attacker, which is exactly why it guards
    only the MCP path and why the API key remains available for stronger auth.
    """
    import secrets

    return "".join(secrets.choice("0123456789") for _ in range(length))


class GatewayConfig(BaseModel):
    child_port_start: Port = 18100
    child_port_end: Port = 18200
    load_timeout_s: PositiveFloat = 600.0
    health_poll_interval_s: PositiveFloat = 0.5
    max_restarts: NonNegativeInt = 3
    restart_backoff_s: NonNegativeFloat = 2.0
    max_images_per_request: NonNegativeInt = 8
    max_image_bytes: PositiveInt = 20 * 1024 * 1024
    # Vision image_url fetches are the one outbound request a CALLER chooses,
    # so by default the server refuses loopback/private/link-local targets: on
    # an open install that made it an unauthenticated probe for anything else
    # on the LAN or tailnet. Turn on only if you genuinely serve images from
    # another box on your own network.
    allow_private_image_hosts: bool = False
    image_fetch_timeout_s: PositiveFloat = 20.0
    max_image_dim: PositiveInt = 2048
    ttl_sweep_interval_s: PositiveFloat = 15.0
    # While a streaming request waits on a JIT load, emit an SSE comment this
    # often so the client's read timeout cannot fire on a load that is
    # progressing. Large models take minutes to page in from disk.
    stream_keepalive_interval_s: PositiveFloat = 5.0
    # Merge a reasoning-only reply into `content` rather than returning "".
    merge_reasoning_into_content: bool = True
    # Deep health probe: a real streamed completion per loaded model. Small and
    # bounded, because a check that can hang is worse than no check.
    deep_probe_timeout_s: PositiveFloat = 20.0
    deep_probe_max_tokens: PositiveInt = 8


class HfConfig(BaseModel):
    token: str | None = None
    cache_dir: Path | None = None
    max_concurrent_downloads: PositiveInt = 2
    chunk_bytes: PositiveInt = 8 * 1024 * 1024


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    # Aliased so config.yaml keeps the natural key `json` while the attribute
    # name avoids shadowing BaseModel.json.
    model_config = ConfigDict(populate_by_name=True)

    level: LogLevel = "INFO"
    json_logs: bool = Field(default=False, alias="json")

    @field_validator("level", mode="before")
    @classmethod
    def _upper_level(cls, v: Any) -> Any:
        # "info" and "warn" are what people type; a typo used to be accepted
        # and silently downgraded to INFO by the logging setup.
        if isinstance(v, str):
            upper = v.strip().upper()
            return "WARNING" if upper == "WARN" else upper
        return v


#: The placeholder the app shipped with before it had a public home. A config
#: that still carries it (every ``config.yaml`` written by 0.1.0/0.2.0 does) is
#: treated exactly like ``repo: null``: not configured, no network call.
UPDATE_REPO_PLACEHOLDER = "studioforge/studioforge"


class UpdateConfig(BaseModel):
    #: ``owner/name`` of the GitHub repository that publishes StudioForge
    #: releases, or null when there is none yet. Unset means the self-update
    #: check reports "not configured" instead of asking GitHub for a repo that
    #: does not exist (a 404 every 24 h against the anonymous rate limit).
    repo: str | None = None
    channel: Literal["stable", "prerelease"] = "stable"
    auto_check: bool = True
    check_interval_h: PositiveFloat = 24.0
    health_check_timeout_s: PositiveFloat = 60.0

    @property
    def configured_repo(self) -> str | None:
        """The repo to ask for releases, or ``None`` when self-update is off."""
        value = (self.repo or "").strip()
        if not value or value == UPDATE_REPO_PLACEHOLDER or "/" not in value:
            return None
        return value


class Config(BaseSettings):
    """Root configuration object."""

    model_config = SettingsConfigDict(
        env_prefix="SF_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_assignment=True,
    )

    data_dir: Path = Field(default_factory=default_data_dir)
    server: ServerConfig = Field(default_factory=ServerConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    hf: HfConfig = Field(default_factory=HfConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    update: UpdateConfig = Field(default_factory=UpdateConfig)

    # Set when loaded from disk so save() knows where to write back.
    source_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def _validate_ports(self) -> Config:
        ports = {
            "server.port": self.server.port,
            "gui.port": self.gui.port,
            "watchdog.port": self.watchdog.port,
        }
        seen: dict[int, str] = {}
        for name, port in ports.items():
            if port in seen:
                raise ValueError(f"{name} ({port}) collides with {seen[port]}")
            seen[port] = name
        if self.gateway.child_port_start >= self.gateway.child_port_end:
            raise ValueError("gateway.child_port_start must be < gateway.child_port_end")
        for name, port in ports.items():
            if self.gateway.child_port_start <= port <= self.gateway.child_port_end:
                raise ValueError(
                    f"{name} ({port}) falls inside the llama-server child port range "
                    f"{self.gateway.child_port_start}-{self.gateway.child_port_end}"
                )
        return self

    # --- derived paths -------------------------------------------------

    @property
    def engines_dir(self) -> Path:
        return self.data_dir / "engines"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def model_logs_dir(self) -> Path:
        return self.logs_dir / "models"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "registry.sqlite3"

    @property
    def downloads_dir(self) -> Path:
        return self.data_dir / "downloads"

    @property
    def releases_dir(self) -> Path:
        return self.data_dir / "releases"

    @property
    def config_path(self) -> Path:
        return self.source_path or (self.data_dir / "config.yaml")

    def model_dirs(self) -> list[Path]:
        """All directories to scan for models, primary first."""
        dirs: list[Path] = []
        if self.models.dir is not None:
            dirs.append(Path(self.models.dir))
        dirs.extend(Path(d) for d in self.models.extra_dirs)
        return dirs

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.engines_dir,
            self.logs_dir,
            self.model_logs_dir,
            self.downloads_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    # --- persistence ---------------------------------------------------

    #: Keys that never go into config.yaml. ``data_dir`` is *derived* --
    #: ``SF_DATA_DIR``, else ``<repo>/data``, else the platform dir (D25) --
    #: and a value written into the file outranked all three on the next load:
    #: copying an old install's ``config.yaml`` into a fresh checkout silently
    #: pointed the whole install back at the old data directory, and
    #: ``local-env.bat`` stopped working the moment Setup saved anything.
    UNPERSISTED_KEYS: ClassVar[frozenset[str]] = frozenset({"source_path", "data_dir"})

    def to_yaml_dict(self) -> dict[str, Any]:
        # by_alias so LoggingConfig.json_logs round-trips as the `json` key.
        return self.model_dump(mode="json", exclude=set(self.UNPERSISTED_KEYS), by_alias=True)

    def save(self, path: Path | None = None) -> Path:
        """Write the config atomically, tolerating a concurrent reader.

        The rename is atomic on POSIX, but on Windows ``os.replace`` fails with
        ``PermissionError`` (WinError 5/32) while ANOTHER process has the
        target open -- and the watchdog re-reads ``config.yaml`` on every
        request it authenticates, so that window is hit in normal operation.
        Observed live: a settings change 500ed and left a stale ``.tmp``
        alongside a half-applied in-memory config. Retry briefly, then fail
        loudly rather than silently leaving the file unwritten.
        """
        target = path or self.config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(self.to_yaml_dict(), sort_keys=False, default_flow_style=False)
        tmp = target.with_suffix(target.suffix + ".tmp")
        # fsync before the rename: without it a power loss in the rename window
        # can leave a zero-length config.yaml, which loads as "{}" and silently
        # reverts every setting to its default (WP17 R5). The previous file is
        # kept as .bak so a truncated one is recoverable by hand.
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.is_file():
            with contextlib.suppress(OSError):
                backup = target.with_suffix(target.suffix + ".bak")
                backup.write_bytes(target.read_bytes())
        last: OSError | None = None
        for attempt in range(_SAVE_RETRIES):
            try:
                tmp.replace(target)
                return target
            except PermissionError as exc:  # Windows: reader holds the target
                last = exc
                time.sleep(_SAVE_RETRY_DELAY_S * (attempt + 1))
        # Do not leave a stale .tmp behind to confuse the next writer.
        with contextlib.suppress(OSError):  # pragma: no cover - best effort
            tmp.unlink(missing_ok=True)
        raise ConfigError(
            f"could not write {target}: {last}. Another process (usually the "
            "watchdog) holds it open; retry in a moment."
        ) from last


def tune_for_hardware(config: Config, gpus: Sequence[Any]) -> list[str]:
    """Adjust first-run defaults to the detected GPUs. Returns what changed.

    Deliberately conservative: only settings where the hardware genuinely
    determines a better default are touched, and each one is justified. Guessing
    at batch sizes or thread counts would look like tuning while actually just
    adding variance -- llama.cpp's own defaults are well chosen there, and the
    planner does not model batch size precisely enough to spend VRAM on it.

    Mutates ``config`` in place; the caller persists it.
    """
    changes: list[str] = []
    if not gpus:
        return changes

    caps = [g.compute_capability for g in gpus if getattr(g, "compute_capability", None)]
    totals = [int(getattr(g, "total_bytes", 0) or 0) for g in gpus]
    smallest_gib = (min(totals) / (1024**3)) if totals else 0.0

    # Context: the KV cache is what a bigger default actually costs, and it
    # scales with the model, so key off the SMALLEST card -- that is the one a
    # single-GPU placement has to fit inside.
    if smallest_gib >= 24:
        target_ctx = 16384
    elif smallest_gib >= 12:
        target_ctx = 8192
    elif smallest_gib > 0:
        target_ctx = 4096
    else:
        target_ctx = config.models.default_ctx
    if target_ctx != config.models.default_ctx:
        config.models.default_ctx = target_ctx
        changes.append(
            f"models.default_ctx={target_ctx} (floor; smallest GPU {smallest_gib:.0f} GiB)"
        )

    # Flash attention is a clear win and fully supported from Ampere (sm_80)
    # onward; leaving it on "auto" just defers the same decision to the engine.
    if caps and all(cap >= (8, 0) for cap in caps) and config.models.default_flash_attn != "on":
        config.models.default_flash_attn = "on"
        changes.append("models.default_flash_attn=on (all GPUs are sm_80 or newer)")

    # FP4 affinity is only meaningful when there is Blackwell hardware to prefer.
    has_blackwell = any(cap >= (12, 0) for cap in caps)
    if not has_blackwell and config.planner.quant_affinity:
        config.planner.quant_affinity = {}
        changes.append("planner.quant_affinity cleared (no Blackwell GPU detected)")

    return changes


def find_config_path(explicit: Path | str | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser()
    env = os.environ.get("SF_CONFIG")
    if env:
        return Path(env).expanduser()
    return default_data_dir() / "config.yaml"


def resolve_data_dir(config_path: Path, *, explicit: bool) -> Path:
    """The data directory for a load of ``config_path`` (D25, in one place).

    ``SF_DATA_DIR`` wins. Otherwise a config file that was *named* -- by
    ``--config`` or ``SF_CONFIG`` -- lives in its data directory (the file is
    always ``<data_dir>/config.yaml``, that is how every process that spawns
    another passes the location: the tray, the watchdog, autostart), so its
    parent is the data dir. Only an unnamed load falls through to the checkout
    / platform default. What is never consulted is a ``data_dir`` key inside
    the file (see :meth:`Config.to_yaml_dict`).
    """
    env = os.environ.get("SF_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    if explicit:
        return config_path.expanduser().resolve().parent
    return default_data_dir()


def load_config(path: Path | str | None = None, *, create: bool = False) -> Config:
    """Load config.yaml, applying env overrides on top.

    With ``create=True`` a missing file is written with detected defaults
    (first-run bootstrap), including the auto-detected LM Studio model dir.
    """
    config_path = find_config_path(path)
    explicit = path is not None or bool(os.environ.get("SF_CONFIG"))
    data_dir = resolve_data_dir(config_path, explicit=explicit)
    raw: dict[str, Any] = {}
    file_present = config_path.is_file()
    if file_present:
        try:
            text = config_path.read_text(encoding="utf-8")
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file {config_path} is not valid YAML: {exc}") from exc
        if loaded is None:
            # An empty (0-byte or comment-only) file. This is what a crash in
            # the old non-fsync save window left behind, and it used to load
            # as "{}" -- every setting silently back at its default, forever,
            # while the file's presence stopped first-run bootstrap from ever
            # rewriting it. Treat it as missing: say so, and let ``create``
            # regenerate it (a .bak from the last good save sits beside it).
            _log().warning(
                "config.yaml is empty; treating it as missing and using defaults",
                path=str(config_path),
                hint="a config.yaml.bak from the last successful save may be next to it",
            )
            loaded = {}
            file_present = False
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {config_path} must contain a YAML mapping")
        raw = loaded

    # ``data_dir`` in the file is ignored (D25: SF_DATA_DIR, else the directory
    # the named config file lives in, else <repo>/data / the platform dir). It
    # got there from an older build's save(); a value that differs from the
    # resolved one is worth one line, not a silent relocation of the install.
    stray = raw.pop("data_dir", None)
    if stray is not None:
        with contextlib.suppress(Exception):
            if Path(str(stray)).expanduser().resolve() != data_dir:
                _log().warning(
                    "config.yaml names a data_dir; ignored -- the data directory is set by "
                    "SF_DATA_DIR or by where config.yaml lives, never by a key inside it",
                    in_file=str(stray),
                    using=str(data_dir),
                )

    _warn_unknown_keys(raw, config_path)

    try:
        config = Config(data_dir=data_dir, **raw)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {config_path}: {exc}") from exc

    config.source_path = config_path if file_present else None

    if config.models.dir is None:
        detected = detect_model_dir()
        if detected is not None:
            config.models.dir = detected

    minted_pin = False
    if config.mcp.pin_required and not config.mcp.pin:
        config.mcp.pin = generate_pin()
        minted_pin = True

    if create and not file_present:
        config.source_path = config_path
        config.ensure_dirs()
        # First run: adapt to the detected hardware so the box works well without
        # the user tuning anything. Import lazily -- config must stay importable
        # on a machine with no NVML at all.
        try:
            from studioforge.core.gpu import get_probe

            applied = tune_for_hardware(config, get_probe().list_gpus())
            if applied:
                from studioforge.logging import get_logger

                get_logger(__name__).info("tuned defaults for detected hardware", changes=applied)
        except Exception:
            pass
        config.save(config_path)
        minted_pin = False  # already written as part of the new file

    if minted_pin and config_path.is_file():
        # Persist a freshly minted PIN, or it would be regenerated on every
        # start -- and a pairing code that changes each restart silently breaks
        # every client that already paired with it.
        try:
            config.source_path = config_path
            config.save(config_path)
        except OSError as exc:
            from studioforge.logging import get_logger

            get_logger(__name__).warning(
                "could not persist the generated MCP pin; it will change on restart",
                error=str(exc),
            )

    _register_secrets(config)
    return config


def _log() -> Any:
    from studioforge.logging import get_logger

    return get_logger(__name__)


def _warn_unknown_keys(raw: dict[str, Any], config_path: Path) -> None:
    """One WARNING per key in the file that no config model declares.

    Unknown keys are ignored rather than fatal -- a file from a newer build, or
    a removed key like ``models.max_loaded``, must not stop the server -- but
    silently ignoring them is how a typo (``server.api_kye``) becomes "the key
    does not work and nothing says why".
    """
    unknown: list[str] = []
    top_fields = set(Config.model_fields)
    for key, value in raw.items():
        if key not in top_fields:
            unknown.append(key)
            continue
        section_type = Config.model_fields[key].annotation
        fields = getattr(section_type, "model_fields", None)
        if not isinstance(value, dict) or not isinstance(fields, dict):
            continue
        accepted = set(fields)
        for info in fields.values():
            if info.alias:
                accepted.add(info.alias)
        unknown.extend(f"{key}.{sub}" for sub in value if sub not in accepted)
    if unknown:
        _log().warning(
            "config.yaml has keys this build does not know; they are ignored",
            path=str(config_path),
            keys=sorted(unknown),
        )


def _register_secrets(config: Config) -> None:
    from studioforge.logging import register_secret

    register_secret(config.server.api_key)
    register_secret(config.hf.token)
    # The MCP pairing PIN is a credential too: it grants the management tools.
    # It is deliberately printed in the startup banner, but a PIN embedded in
    # a longer logged string (a ?pin= URL, an error message) must still scrub.
    register_secret(config.mcp.pin)


def apply_overrides(config: Config, updates: dict[str, Any]) -> Config:
    """Apply dotted-path updates (``"models.default_ctx": 4096``) and validate.

    Returns a NEW validated Config; the caller decides whether to persist. Used
    by both the GUI/MCP ``set_config`` and the watchdog's on-disk editor, so
    validation lives in exactly one place.
    """
    data = config.to_yaml_dict()
    for dotted, value in updates.items():
        if dotted.split(".")[0] in Config.UNPERSISTED_KEYS:
            raise ConfigError(
                f"{dotted} is not stored in config.yaml: the data directory is set by "
                "SF_DATA_DIR (local-env.bat next to the launchers, or the shell/systemd "
                "environment), see README 'Data directory'",
                param=dotted,
            )
        parts = dotted.split(".")
        cursor: Any = data
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor:
                raise ConfigError(f"unknown config key: {dotted}")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"unknown config key: {dotted}")
        cursor[leaf] = value
    try:
        # data_dir is not in the dump (never persisted); carry it over or the
        # new object -- and its save() -- lands in the default data directory.
        updated = Config(data_dir=config.data_dir, **data)
    except Exception as exc:
        raise ConfigError(f"invalid config update: {exc}") from exc
    updated.source_path = config.source_path
    _register_secrets(updated)
    return updated


# Keys that only take effect after a restart; the GUI shows an indicator.
RESTART_REQUIRED_KEYS = frozenset(
    {
        "server.host",
        "server.port",
        "gui.enabled",
        "gui.host",
        "gui.port",
        "watchdog.enabled",
        "watchdog.host",
        "watchdog.port",
        "data_dir",
        "gateway.child_port_start",
        "gateway.child_port_end",
        # CORS origins are captured when the middleware is constructed, and the
        # MCP path decides where the app mounts -- neither can change in place.
        "server.cors_origins",
        "server.cors_allow_credentials",
        "mcp.path",
        "mcp.enabled",
    }
)
