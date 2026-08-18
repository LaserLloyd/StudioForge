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
    "speculative": "draft-model decoding (--spec-type draft-simple)",
    "multi_part": "sharded GGUFs (-00001-of-0000N) treated as one model",
    "prompt_cache": "prompt-cache reuse (--cache-reuse), the big agent-workload win",
}


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

    def supports_architecture(self, arch: str) -> bool:
        return arch.lower() in {a.lower() for a in self.architectures}


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
            )

    snapshot = load_snapshot()
    snap_tag = str(snapshot.get("source_tag") or "")
    detail = f"bundled snapshot for {snap_tag or 'an unknown tag'}"
    if snap_tag and snap_tag != tag:
        # Say so plainly: the running engine is not the one the list came from.
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

    Also flags any architecture the pinned engine does not know -- that is the
    one case where a model in the library genuinely cannot be run, and it is
    invisible until the load fails.
    """
    by_arch: dict[str, int] = {}
    by_quant: dict[str, int] = {}
    caps = {"vision": 0, "tools": 0, "thinking": 0, "embedding": 0, "multi_part": 0}
    unsupported: list[dict[str, str]] = []

    for record in records:
        by_arch[record.architecture] = by_arch.get(record.architecture, 0) + 1
        by_quant[record.quant] = by_quant.get(record.quant, 0) + 1
        for key in caps:
            if getattr(record.capabilities, key, False):
                caps[key] += 1
        if engine.architectures and not engine.supports_architecture(record.architecture):
            unsupported.append({"model_id": record.id, "architecture": record.architecture})

    return {
        "model_count": len(records),
        "total_bytes": sum(r.size_bytes for r in records),
        "by_architecture": dict(sorted(by_arch.items(), key=lambda kv: (-kv[1], kv[0]))),
        "by_quant": dict(sorted(by_quant.items(), key=lambda kv: (-kv[1], kv[0]))),
        "capabilities": caps,
        "unsupported_by_engine": unsupported,
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
