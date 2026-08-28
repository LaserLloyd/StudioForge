"""What this box can actually run.

Answers "what backend am I on, what kinds of model does it support, what does my
hardware allow, and is there a newer one?" -- a question that is otherwise only
answerable by reading llama.cpp release notes and doing VRAM arithmetic by hand.

The architecture and quantization lists are **extracted from llama.cpp's own
source at the pinned tag**, not hand-maintained. A hand-written list would be
wrong within a release or two: `b10425` alone knows 142 architectures, and new
ones land constantly. Extraction happens against a checkout when one is
available and otherwise falls back to a snapshot shipped with the package, and
the report always says which of the two it used -- a stale list presented as
authoritative would be worse than no list.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from studioforge.config import Config
from studioforge.logging import get_logger
from studioforge.types import GB, GpuInfo, ModelRecord

log = get_logger(__name__)

#: Quant families that run everywhere but are meaningfully faster on newer
#: hardware. Measured, not assumed -- see DECISIONS.md D9.
QUANT_HARDWARE_NOTES: dict[str, str] = {
    "NVFP4": (
        "runs on any supported GPU, but only Blackwell (sm_120+) has native FP4 "
        "tensor cores; measured ~6x faster prefill there vs ~3x for Q4_0"
    ),
    "MXFP4": "FP4 variant; same Blackwell acceleration caveat as NVFP4",
}

#: Feature -> the llama.cpp mechanism behind it, so the panel is explanatory
#: rather than a list of ticks.
FEATURE_NOTES: dict[str, str] = {
    "vision": "multimodal projector (mmproj) loaded alongside the model",
    "embeddings": "dedicated instance launched with --embedding",
    "reranking": "rerank endpoint, for models built for it",
    "tools": "function calling, when the model's chat template handles tools",
    "thinking": "reasoning models; thoughts stay inline in content by default",
    "lora": "GGUF LoRA adapters, with runtime scale changes where supported",
    "speculative": (
        "speculative decoding, chosen per model: the model's own MTP heads "
        "(--spec-type draft-mtp) when it has them, a draft model "
        "(draft-simple) when one is attached, n-gram drafting (ngram-mod) for "
        "thinking and MoE models. Distribution-preserving: speed, not a trade"
    ),
    "multi_part": "sharded GGUFs (-00001-of-0000N) treated as one model",
    "prompt_cache": (
        "prompt-cache reuse (--cache-reuse) plus a host-RAM cache (--cache-ram) "
        "for prefixes that left the slot -- the big agent-workload win"
    ),
    "tensor_split": (
        "opt-in tensor parallelism (--split-mode tensor): weights and KV shard "
        "across GPUs. EXPERIMENTAL upstream and measured SLOWER than the layer "
        "split on this PCIe rig, so it is something to benchmark, not a default"
    ),
}

#: The optional-feature keys the Setup tab's Engine card shows, in the order it
#: shows them, with the one-line explanation each needs. Lives here rather than
#: in the GUI so the CLI's ``studioforge capabilities`` and the card cannot
#: disagree about what a feature is.
ENGINE_FEATURE_LABELS: tuple[tuple[str, str, str], ...] = (
    ("split_modes", "Split modes", "how a multi-GPU placement shards the model"),
    ("spec_types", "Speculative types", "drafting strategies --spec-type accepts"),
    ("flash_attn_values", "Flash attention", "values -fa accepts; StudioForge passes 'on'"),
    ("backend_sampling", "GPU sampling", "--backend-sampling (experimental, opt-in)"),
    ("cache_ram", "Host prompt cache", "--cache-ram, host RAM, no VRAM, on by default"),
    ("kv_unified", "Unified KV switch", "--kv-unified / --no-kv-unified"),
    ("ctx_checkpoints", "Context checkpoints", "--ctx-checkpoints, per slot"),
    ("fit", "Engine auto-fit", "--fit; StudioForge always passes 'off' (D11)"),
)


def engine_feature_rows(features: Mapping[str, Any]) -> list[dict[str, str]]:
    """``[{"name", "value", "note"}]`` for the Engine card and the CLI.

    Renders "not advertised" rather than "off" for anything a build does not
    declare: those are different facts, and conflating them is how a missing
    feature reads as a disabled one.
    """
    rows: list[dict[str, str]] = []
    known = bool(features.get("known"))
    for key, name, note in ENGINE_FEATURE_LABELS:
        raw = features.get(key)
        if not known:
            value = "unknown"
        elif isinstance(raw, list):
            value = ", ".join(str(item) for item in raw) or "none"
        elif isinstance(raw, bool):
            value = "yes" if raw else "not advertised"
        else:
            value = str(raw) if raw else "not advertised"
        rows.append({"name": name, "value": value, "note": note})
    return rows


@dataclass
class EngineCapabilities:
    """What the pinned engine build supports."""

    tag: str
    variant: str
    version_string: str | None
    smoke_tested: bool
    installed_at: float | None
    architectures: list[str]
    quant_types: list[str]
    ggml_types: list[str]
    source: str  # "checkout" | "snapshot"
    source_detail: str
    #: The tag the architecture list actually describes: the snapshot's
    #: ``source_tag``, or the running tag for a checkout (a checkout is
    #: resolved from the pinned tag, so it is the build's own source tree).
    #: Empty when unknown, which counts as "does not describe this engine".
    source_tag: str = ""
    #: What this *build* advertises, read from its own ``--help``
    #: (:class:`studioforge.core.engine.EngineFeatures`). Distinct from the
    #: architecture/quant lists above, which come from llama.cpp's source at the
    #: pinned tag: those say what the project supports, this says what the
    #: binary on this disk will actually accept.
    features: dict[str, Any] = field(default_factory=dict)

    def supports_architecture(self, arch: str) -> bool:
        return arch.lower() in {a.lower() for a in self.architectures}

    @property
    def describes_active_engine(self) -> bool:
        """Whether the architecture list is this build's, or somebody else's.

        The shipped snapshot is pinned to one tag (``b10425``) and the engine
        moves independently of it, so "this engine does not support X" was a
        claim about a build the user may not be running -- stated as fact, in
        the one place that decides whether a model gets flagged unrunnable. A
        checkout is resolved from the running tag, so it counts; a snapshot only
        counts when its ``source_tag`` matches. Anything else is advisory
        (D49-8).
        """
        return self.source == "checkout" or (bool(self.source_tag) and self.source_tag == self.tag)


@dataclass
class HardwareCapabilities:
    gpus: list[dict[str, Any]]
    total_vram_bytes: int
    largest_gpu_bytes: int
    usable_largest_bytes: int
    usable_total_bytes: int
    blackwell_present: bool
    driver_version: str | None
    cuda_driver_version: str | None


@dataclass
class CapabilityReport:
    engine: EngineCapabilities
    hardware: HardwareCapabilities
    features: dict[str, str]
    library: dict[str, Any]
    update: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": {
                "tag": self.engine.tag,
                "variant": self.engine.variant,
                "version_string": self.engine.version_string,
                "smoke_tested": self.engine.smoke_tested,
                "installed_at": self.engine.installed_at,
                "architecture_count": len(self.engine.architectures),
                "architectures": self.engine.architectures,
                "quant_types": self.engine.quant_types,
                "ggml_types": self.engine.ggml_types,
                "capability_source": self.engine.source,
                "capability_source_detail": self.engine.source_detail,
                "capability_source_tag": self.engine.source_tag,
                "capability_describes_engine": self.engine.describes_active_engine,
                "features": self.engine.features,
                "feature_rows": engine_feature_rows(self.engine.features),
            },
            "hardware": {
                "gpus": self.hardware.gpus,
                "total_vram_bytes": self.hardware.total_vram_bytes,
                "largest_gpu_bytes": self.hardware.largest_gpu_bytes,
                "usable_largest_bytes": self.hardware.usable_largest_bytes,
                "usable_total_bytes": self.hardware.usable_total_bytes,
                "blackwell_present": self.hardware.blackwell_present,
                "driver_version": self.hardware.driver_version,
                "cuda_driver_version": self.hardware.cuda_driver_version,
            },
            "features": self.features,
            "quant_hardware_notes": QUANT_HARDWARE_NOTES,
            "library": self.library,
            "update": self.update,
        }


# ---------------------------------------------------------------------------
# Engine capability extraction
# ---------------------------------------------------------------------------

_ARCH_RE = re.compile(r'\{\s*LLM_ARCH_[A-Z0-9_]+,\s*"([a-z0-9._-]+)"\s*\}')
_FTYPE_RE = re.compile(r"LLAMA_FTYPE_MOSTLY_([A-Z0-9_]+)\s*=")
_GGML_TYPE_RE = re.compile(r'\.type_name\s*=\s*"([a-z0-9_]+)"')


def snapshot_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "engine_capabilities.json"


def checkout_candidates(config: Config) -> list[Path]:
    """Where a llama.cpp source tree might be, most specific first."""
    import os

    candidates: list[Path] = []
    env = os.environ.get("SF_VENDOR_LLAMA_CPP")
    if env:
        candidates.append(Path(env))
    # A source-build engine keeps its checkout next to the installed binaries.
    candidates.append(config.engines_dir / f"src-{config.engine.pinned_tag}")
    # The dev layout: <project>/vendor/llama.cpp alongside <project>/studioforge.
    candidates.append(config.data_dir.parent / "vendor" / "llama.cpp")
    candidates.append(Path.cwd().parent / "vendor" / "llama.cpp")
    return candidates


def extract_from_checkout(root: Path) -> dict[str, list[str]] | None:
    """Pull the arch/quant tables out of a llama.cpp source tree."""
    arch_file = root / "src" / "llama-arch.cpp"
    header = root / "include" / "llama.h"
    if not arch_file.is_file() or not header.is_file():
        return None
    try:
        arch_text = arch_file.read_text(encoding="utf-8", errors="replace")
        header_text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    architectures = sorted({m for m in _ARCH_RE.findall(arch_text) if m != "unknown"})
    file_types = sorted(set(_FTYPE_RE.findall(header_text)))
    ggml_types: list[str] = []
    ggml_file = root / "ggml" / "src" / "ggml.c"
    if ggml_file.is_file():
        try:
            ggml_types = sorted(
                set(_GGML_TYPE_RE.findall(ggml_file.read_text(encoding="utf-8", errors="replace")))
            )
        except OSError:
            ggml_types = []
    if not architectures:
        return None
    return {
        "architectures": architectures,
        "file_types": file_types,
        "ggml_types": ggml_types,
    }


def load_snapshot() -> dict[str, list[str]]:
    path = snapshot_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("engine capability snapshot unreadable", error=str(exc))
        return {"architectures": [], "file_types": [], "ggml_types": []}
    return {
        "architectures": list(data.get("architectures") or []),
        "file_types": list(data.get("file_types") or []),
        "ggml_types": list(data.get("ggml_types") or []),
        "source_tag": data.get("source_tag", ""),
    }


def _quant_labels(file_types: list[str]) -> list[str]:
    """The quantizations a user would recognise, from the ftype enum."""
    skip = {"ALL_F32", "F16", "BF16", "COPY", "GUESSED"}
    return sorted(t for t in file_types if t not in skip)


def detect_engine_features(config: Config, tag: str) -> dict[str, Any]:
    """The pinned build's advertised feature surface, from the on-disk cache.

    Deliberately the *cached* reader: this runs inside an async route and a GUI
    refresh, and a cold cache would otherwise spawn ``llama-server --help``
    there. Boot warms the cache (``EngineManager.ensure_engine``), so a miss
    here means the engine is not installed -- which is worth reporting as
    "unknown", not worth blocking a page for.
    """
    from studioforge.core.engine import cached_engine_features

    try:
        return cached_engine_features(config.engines_dir / tag, tag).to_dict()
    except Exception as exc:  # noqa: BLE001 - a report must never fail on this
        log.warning("engine feature detection failed", tag=tag, error=str(exc))
        return {"tag": tag, "known": False}


def engine_capabilities(config: Config, engine_manager: Any = None) -> EngineCapabilities:
    tag = config.engine.pinned_tag
    variant = "unknown"
    version_string = None
    smoke_tested = False
    installed_at = None
    if engine_manager is not None:
        try:
            info = engine_manager.get(tag) or engine_manager.active()
        except Exception:
            info = None
        if info is not None:
            tag = info.tag
            variant = info.variant
            version_string = info.version_string
            smoke_tested = info.smoke_tested
            installed_at = info.installed_at

    features = detect_engine_features(config, tag)

    for root in checkout_candidates(config):
        extracted = extract_from_checkout(root)
        if extracted is not None:
            return EngineCapabilities(
                tag=tag,
                variant=variant,
                version_string=version_string,
                smoke_tested=smoke_tested,
                installed_at=installed_at,
                architectures=extracted["architectures"],
                quant_types=_quant_labels(extracted["file_types"]),
                ggml_types=extracted["ggml_types"],
                source="checkout",
                source_detail=str(root),
                source_tag=tag,
                features=features,
            )

    snapshot = load_snapshot()
    snap_tag = str(snapshot.get("source_tag") or "")
    detail = f"bundled snapshot for {snap_tag or 'an unknown tag'}"
    if snap_tag != tag:
        # Say so plainly: the running engine is not the one the list came from.
        # ``scripts/refresh_engine_capabilities.py`` regenerates the snapshot.
        detail += f"; the active engine is {tag}, so this list may be out of date"
    return EngineCapabilities(
        tag=tag,
        variant=variant,
        version_string=version_string,
        smoke_tested=smoke_tested,
        installed_at=installed_at,
        architectures=list(snapshot["architectures"]),
        quant_types=_quant_labels(list(snapshot["file_types"])),
        ggml_types=list(snapshot["ggml_types"]),
        source="snapshot",
        source_detail=detail,
        source_tag=snap_tag,
        features=features,
    )


# ---------------------------------------------------------------------------
# Hardware + library
# ---------------------------------------------------------------------------


def hardware_capabilities(
    config: Config, gpus: list[GpuInfo], probe: Any = None
) -> HardwareCapabilities:
    headroom = config.planner.headroom_fraction
    usable = [max(0, g.free_bytes - int(g.total_bytes * headroom)) for g in gpus]
    driver = cuda = None
    if probe is not None:
        try:
            driver = probe.driver_version()
            version = probe.cuda_driver_version()
            cuda = f"{version[0]}.{version[1]}" if version else None
        except Exception:
            pass
    return HardwareCapabilities(
        gpus=[
            {
                "index": g.index,
                "name": g.name,
                "total_bytes": g.total_bytes,
                "free_bytes": g.free_bytes,
                "compute_capability": g.cc_str,
                "sm_arch": g.sm_arch,
            }
            for g in gpus
        ],
        total_vram_bytes=sum(g.total_bytes for g in gpus),
        largest_gpu_bytes=max((g.total_bytes for g in gpus), default=0),
        usable_largest_bytes=max(usable, default=0),
        usable_total_bytes=sum(usable),
        blackwell_present=any((g.compute_capability or (0, 0)) >= (12, 0) for g in gpus),
        driver_version=driver,
        cuda_driver_version=cuda,
    )


def library_summary(records: list[ModelRecord], engine: EngineCapabilities) -> dict[str, Any]:
    """What is in the library, grouped the way a user thinks about it.

    Also flags any architecture the engine does not know -- that is the one case
    where a model in the library genuinely cannot be run, and it is invisible
    until the load fails.

    **The verdict is only given when the architecture list actually describes
    the running build** (D49-8). The list ships as a snapshot pinned to one tag
    and the engine moves on its own, so a model using an architecture added
    *after* the snapshot was taken was being reported as unsupported by an
    engine that supports it perfectly well -- a hard "cannot run this" derived
    from a list about a different build. When the list does not describe the
    active engine those models go to ``unknown_to_architecture_list`` instead,
    and ``architecture_list_describes_engine`` says which of the two happened so
    a caller can word it honestly.
    """
    by_arch: dict[str, int] = {}
    by_quant: dict[str, int] = {}
    caps = {"vision": 0, "tools": 0, "thinking": 0, "embedding": 0, "multi_part": 0}
    unsupported: list[dict[str, str]] = []
    advisory: list[dict[str, str]] = []
    authoritative = engine.describes_active_engine

    for record in records:
        by_arch[record.architecture] = by_arch.get(record.architecture, 0) + 1
        by_quant[record.quant] = by_quant.get(record.quant, 0) + 1
        for key in caps:
            if getattr(record.capabilities, key, False):
                caps[key] += 1
        if engine.architectures and not engine.supports_architecture(record.architecture):
            row = {"model_id": record.id, "architecture": record.architecture}
            (unsupported if authoritative else advisory).append(row)

    return {
        "model_count": len(records),
        "total_bytes": sum(r.size_bytes for r in records),
        "by_architecture": dict(sorted(by_arch.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_quant": dict(sorted(by_quant.items(), key=lambda kv: (-kv[1], kv[0]))),
        "capabilities": caps,
        "unsupported_by_engine": unsupported,
        "unknown_to_architecture_list": advisory,
        "architecture_list_describes_engine": authoritative,
        "architecture_list_source": engine.source,
        "architecture_list_detail": engine.source_detail,
    }


def size_verdicts(records: list[ModelRecord], hardware: HardwareCapabilities) -> dict[str, Any]:
    """How much of the library this hardware can actually load.

    Weights only -- a real fit check needs the planner and a context length.
    This is the honest coarse answer to "what can this box handle?", and says so.
    """
    single = hardware.usable_largest_bytes
    total = hardware.usable_total_bytes
    fits_one = [r for r in records if r.size_bytes <= single]
    fits_split = [r for r in records if single < r.size_bytes <= total]
    too_big = [r for r in records if r.size_bytes > total]
    return {
        "fits_one_gpu": len(fits_one),
        "needs_multiple_gpus": len(fits_split),
        "too_big": len(too_big),
        "too_big_models": [
            {"model_id": r.id, "size_bytes": r.size_bytes}
            for r in sorted(too_big, key=lambda r: r.size_bytes)[:10]
        ],
        "largest_runnable_bytes": total,
        "note": (
            "weights only, at current free VRAM; the KV cache for your context "
            "length is extra. Use the per-model fit check for an exact verdict."
        ),
    }


def build_report(
    config: Config,
    *,
    gpus: list[GpuInfo],
    records: list[ModelRecord],
    engine_manager: Any = None,
    probe: Any = None,
) -> CapabilityReport:
    engine = engine_capabilities(config, engine_manager)
    hardware = hardware_capabilities(config, gpus, probe)
    library = library_summary(records, engine)
    library["sizing"] = size_verdicts(records, hardware)
    return CapabilityReport(
        engine=engine, hardware=hardware, features=dict(FEATURE_NOTES), library=library
    )


def format_bytes(value: int) -> str:
    return f"{value / GB:.1f} GiB"
