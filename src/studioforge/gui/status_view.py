"""Pure derivations for the Status tab: the health chip and what sits on each GPU.

Presentation only, like the rest of ``gui``: it reads what the backend already reports
(``BootStatus.snapshot()``, ``InstanceInfo``, ``GpuLease``, ``GpuInfo``) and words it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from studioforge.gui import state as st


@dataclass(frozen=True)
class Health:
    label: str
    colour: str
    detail: str


def health_chip(
    boot: Mapping[str, Any] | None,
    *,
    draining: bool = False,
    instances: Sequence[Any] = (),
    gpus_seen: bool = True,
    manager_present: bool = True,
) -> Health:
    """One word for "is it fine", with the reason when it is not.

    Order matters: a boot that has not finished or failed outranks everything (nothing else
    is trustworthy yet), a drain is deliberate, a failed child is a fault worth naming, and a
    box with no GPU probe still serves the API but cannot place anything.
    """
    if boot:
        if boot.get("error"):
            return Health("Boot failed", "negative", str(boot["error"]))
        if not boot.get("ready", True):
            phase = str(boot.get("phase") or "starting")
            return Health("Starting", "warning", f"boot phase: {phase}")
    if not manager_present:
        return Health("Degraded", "warning", "the model manager is not running")
    if draining:
        return Health("Draining", "warning", "finishing requests before a restart")
    failed = [str(getattr(i, "model_id", "?")) for i in instances if _state(i) == "failed"]
    if failed:
        return Health("Attention", "negative", "failed to load: " + ", ".join(failed))
    if not gpus_seen:
        return Health("No GPU", "warning", "no GPU is reported; models cannot be placed")
    return Health("Healthy", "positive", "ready to serve")


def _state(instance: Any) -> str:
    return str(getattr(instance, "state", "") or "")


@dataclass(frozen=True)
class GpuOccupant:
    model_id: str
    bytes: int | None
    state: str


def gpu_occupants(index: int, instances: Iterable[Any]) -> list[GpuOccupant]:
    """The loaded models whose plan puts weights on GPU ``index`` (bytes when known)."""
    out: list[GpuOccupant] = []
    for instance in instances:
        plan = getattr(instance, "plan", None)
        devices = [int(d) for d in (getattr(plan, "devices", None) or [])]
        if index not in devices:
            continue
        per_gpu = getattr(plan, "per_gpu_bytes", None) or {}
        raw = per_gpu.get(index)
        out.append(GpuOccupant(str(instance.model_id), int(raw) if raw else None, _state(instance)))
    return out


def other_programs_bytes(gpu: Any, occupants: Sequence[GpuOccupant]) -> int | None:
    """VRAM used on the card that none of our children account for, or ``None`` if unknown.

    Only meaningful when every occupant reported its size; otherwise the difference would
    blame another program for what is simply a missing number.
    """
    if any(o.bytes is None for o in occupants):
        return None
    ours = sum(o.bytes or 0 for o in occupants)
    extra = int(getattr(gpu, "used_bytes", 0) or 0) - ours
    return extra if extra > 0 else 0


def leases_on_gpu(index: int, leases: Iterable[Any]) -> list[Any]:
    return [lease for lease in leases if index in (getattr(lease, "devices", None) or [])]


def gpu_occupant_lines(gpu: Any, instances: Sequence[Any], leases: Sequence[Any]) -> list[str]:
    """The "what is on it" lines under one GPU's bar."""
    index = int(getattr(gpu, "index", 0))
    lines: list[str] = []
    occupants = gpu_occupants(index, instances)
    for occupant in occupants:
        size = f" · {st.format_gib(occupant.bytes, precision=1)}" if occupant.bytes else ""
        state = "" if occupant.state in ("", "ready") else f" ({occupant.state})"
        lines.append(f"{occupant.model_id}{size}{state}")
    other = other_programs_bytes(gpu, occupants)
    if other and other >= 256 * 1024 * 1024:
        lines.append(f"other programs · {st.format_gib(other, precision=1)}")
    for lease in leases_on_gpu(index, leases):
        lines.append(f"lease · {getattr(lease, 'holder', '?')}")
    if not lines:
        lines.append("nothing loaded")
    return lines


def total_active_requests(instances: Iterable[Any]) -> int:
    return sum(int(getattr(i, "active_requests", 0) or 0) for i in instances)


def busy_label(instance: Any) -> tuple[str, str]:
    """``(Busy|Idle|Loading|Failed, colour)`` for a loaded-model row."""
    state = _state(instance)
    if state == "loading":
        return ("Loading", "warning")
    if state == "failed":
        return ("Failed", "negative")
    if int(getattr(instance, "active_requests", 0) or 0) > 0:
        return ("Busy", "primary")
    return ("Idle", "grey")


__all__ = [
    "GpuOccupant",
    "Health",
    "busy_label",
    "gpu_occupant_lines",
    "gpu_occupants",
    "health_chip",
    "leases_on_gpu",
    "other_programs_bytes",
    "total_active_requests",
]
