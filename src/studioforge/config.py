"""Configuration: a single config.yaml plus SF_-prefixed env overrides.

Layering (lowest priority first): field defaults -> config.yaml -> environment.
The resolved config is written back on ``save()`` so the GUI/MCP ``set_config``
surfaces and hand edits round-trip through the same file. The file lives
*outside* the release directories so self-update never clobbers it.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from studioforge.errors import ConfigError

#: "auto" is resolved by the planner per model: it takes the best-quality KV
#: cache that still reaches the chosen context on the hardware available.
KvCacheType = Literal["auto", "f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]
SplitMode = Literal["none", "layer", "row", "tensor"]
FlashAttn = Literal["on", "off", "auto"]

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


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 1234  # LM Studio's default port: OpenClaw only changes the host
    # >= 6 chars so the log-redaction processor will register it: shorter
    # values are ignored there, and an unredactable key can surface inside a
    # logged command line or an httpx error repr.
    api_key: str | None = Field(default=None, min_length=6)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False
    drain_timeout_s: float = 30.0
    request_timeout_s: float = 900.0


class GuiConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    refresh_interval_s: float = 2.0


class WatchdogConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 1235
    poll_interval_s: float = 10.0
    health_timeout_s: float = 5.0
    wedged_after_failures: int = 3


#: config.yaml is rewritten while the watchdog may have it open for reading.
_SAVE_RETRIES = 6
_SAVE_RETRY_DELAY_S = 0.05


class ModelsConfig(BaseModel):
    dir: Path | None = None
    extra_dirs: list[Path] = Field(default_factory=list)
    default_ctx: int = 8192
    default_ttl_s: int = 1800  # 30 min; 0 = pinned/never unload
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
    ctx_per_slot_default: int = 32768
    # "on" rather than "auto": flash attention is a large KV-bandwidth and
    # memory win and is supported on every GPU from Ampere (sm_80) up, which
    # is all of this rig. "auto" defers the same decision to the engine and
    # has been observed to decline it. tune_for_hardware drops this back to
    # "auto" if a pre-Ampere card is ever detected.
    default_flash_attn: FlashAttn = "on"
    default_cache_reuse: int = 256  # prompt-cache reuse: the big OpenClaw latency win
    # "none" keeps a reasoning model's thoughts inline in message.content.
    # llama.cpp's default ("auto") splits them into reasoning_content and leaves
    # content empty, which reads as an empty reply to every OpenAI client.
    default_reasoning_format: Literal["none", "deepseek", "deepseek-legacy"] = "none"
    # Reasoning models spend their budget thinking before answering, so the
    # ordinary default truncates the chain of thought AND the visible answer.
    # Applied only when a thinking model has no explicit ctx_size, clamped to
    # the model's trained window and to what actually fits.
    thinking_default_ctx: int = 32768
    # What every model AIMS for when nobody asks for a specific size. The
    # planner walks down from here -- halving -- to the largest window that
    # actually fits in VRAM, never below ``default_ctx``. Agent workloads
    # (OpenClaw) carry long tool transcripts and run out of room at 8k, so the
    # aim is high and the step-down is what keeps big models loadable.
    target_ctx: int = 1048576
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

    @field_validator("ctx_per_slot_default")
    @classmethod
    def _positive_ctx_per_slot(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("models.ctx_per_slot_default must be positive")
        return v


class EngineConfig(BaseModel):
    pinned_tag: str = "b10425"
    cuda_variant: str = "auto"  # e.g. "12.4" / "13.3" / "auto"
    keep_versions: int = 3
    allow_source_build: bool = True
    repo: str = "ggml-org/llama.cpp"
    smoke_test_timeout_s: float = 180.0


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
    compute_overhead_fraction: float = 0.06
    compute_overhead_floor_mb: int = 400
    cuda_context_mb: int = 300  # per-GPU CUDA context + cuBLAS workspace
    image_tokens_default: int = 1024
    mmproj_compute_mb: int = 512
    prefer_single_gpu: bool = True
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
                raise ValueError(
                    f"planner.reserved_mb[{index}] must be >= 0 MiB (got {amount})"
                )
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
    child_port_start: int = 18100
    child_port_end: int = 18200
    load_timeout_s: float = 600.0
    health_poll_interval_s: float = 0.5
    max_restarts: int = 3
    restart_backoff_s: float = 2.0
    max_images_per_request: int = 8
    max_image_bytes: int = 20 * 1024 * 1024
    # Vision image_url fetches are the one outbound request a CALLER chooses,
    # so by default the server refuses loopback/private/link-local targets: on
    # an open install that made it an unauthenticated probe for anything else
    # on the LAN or tailnet. Turn on only if you genuinely serve images from
    # another box on your own network.
    allow_private_image_hosts: bool = False
    image_fetch_timeout_s: float = 20.0
    max_image_dim: int = 2048
    ttl_sweep_interval_s: float = 15.0
    # While a streaming request waits on a JIT load, emit an SSE comment this
    # often so the client's read timeout cannot fire on a load that is
    # progressing. Large models take minutes to page in from disk.
    stream_keepalive_interval_s: float = 5.0
    # Merge a reasoning-only reply into `content` rather than returning "".
    merge_reasoning_into_content: bool = True
    # Deep health probe: a real streamed completion per loaded model. Small and
    # bounded, because a check that can hang is worse than no check.
    deep_probe_timeout_s: float = 20.0
    deep_probe_max_tokens: int = 8


class HfConfig(BaseModel):
    token: str | None = None
    cache_dir: Path | None = None
    max_concurrent_downloads: int = 2
    chunk_bytes: int = 8 * 1024 * 1024


class LoggingConfig(BaseModel):
    # Aliased so config.yaml keeps the natural key `json` while the attribute
    # name avoids shadowing BaseModel.json.
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=False, alias="json")


class UpdateConfig(BaseModel):
    repo: str = "studioforge/studioforge"
    channel: Literal["stable", "prerelease"] = "stable"
    auto_check: bool = True
    check_interval_h: float = 24.0
    health_check_timeout_s: float = 60.0


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

    def to_yaml_dict(self) -> dict[str, Any]:
        # by_alias so LoggingConfig.json_logs round-trips as the `json` key.
        return self.model_dump(mode="json", exclude={"source_path"}, by_alias=True)

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
        tmp.write_text(payload, encoding="utf-8")
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


def load_config(path: Path | str | None = None, *, create: bool = False) -> Config:
    """Load config.yaml, applying env overrides on top.

    With ``create=True`` a missing file is written with detected defaults
    (first-run bootstrap), including the auto-detected LM Studio model dir.
    """
    config_path = find_config_path(path)
    raw: dict[str, Any] = {}
    if config_path.is_file():
        try:
            loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"config file {config_path} is not valid YAML: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file {config_path} must contain a YAML mapping")
        raw = loaded

    try:
        config = Config(**raw)
    except Exception as exc:
        raise ConfigError(f"invalid configuration in {config_path}: {exc}") from exc

    config.source_path = config_path if config_path.is_file() else None

    if config.models.dir is None:
        detected = detect_model_dir()
        if detected is not None:
            config.models.dir = detected

    minted_pin = False
    if config.mcp.pin_required and not config.mcp.pin:
        config.mcp.pin = generate_pin()
        minted_pin = True

    if create and not config_path.is_file():
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
        updated = Config(**data)
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
        "mcp.path",
        "mcp.enabled",
    }
)
