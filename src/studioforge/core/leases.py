"""GPU leases: a claim on specific cards that the planner honours (D43).

A lease says "these CUDA devices belong to *this* for now". While it stands,
only the models it names may be planned onto those cards -- everything else
(a JIT load, the pin reconciler, the rebalancer, a benchmark of some other
model) sees the cards as absent and places elsewhere or is refused with the
lease named. A lease with no models holds the cards for something outside
this server entirely (a ComfyUI run, a training job).

The book is in-memory on purpose: a lease describes a live situation -- a
benchmark in progress, a model someone wants warm for the afternoon -- and a
restart is a clean slate. What keeps it honest is the idle TTL: the sweep
releases a lease nobody has touched for that long, so a crashed benchmark or
a forgotten reservation cannot hold a card forever.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterable
from typing import Any

from studioforge.errors import BadRequestError, LeaseConflictError, LeaseNotFoundError
from studioforge.types import GpuLease

#: How long a lease survives with no activity before the sweep releases it.
DEFAULT_IDLE_TTL_S = 3600.0


class LeaseBook:
    """Every standing lease, keyed by id. Single-threaded: lives on the event loop."""

    def __init__(self) -> None:
        self._leases: dict[str, GpuLease] = {}

    def __len__(self) -> int:
        return len(self._leases)

    def all(self) -> list[GpuLease]:
        return sorted(self._leases.values(), key=lambda lease: lease.created_at)

    def get(self, lease_id: str) -> GpuLease | None:
        return self._leases.get(lease_id)

    def conflicts(self, devices: Iterable[int]) -> list[GpuLease]:
        """Leases holding any of ``devices``."""
        wanted = set(devices)
        return [lease for lease in self.all() if wanted & set(lease.devices)]

    def acquire(
        self,
        devices: Iterable[int],
        *,
        holder: str,
        model_ids: Iterable[str] = (),
        reason: str = "",
        idle_ttl_s: float | None = DEFAULT_IDLE_TTL_S,
        now: float | None = None,
    ) -> GpuLease:
        """Record a lease, or raise :class:`LeaseConflictError` if a card is taken."""
        wanted = sorted({int(d) for d in devices})
        if not wanted:
            raise BadRequestError("a lease must name at least one CUDA device", param="devices")
        if idle_ttl_s is not None and idle_ttl_s <= 0:
            raise BadRequestError(
                "idle_ttl_s must be positive, or null for 'until released'", param="idle_ttl_s"
            )
        clash = self.conflicts(wanted)
        if clash:
            names = "; ".join(
                f"lease {lease.id} ({lease.holder}) holds "
                f"CUDA {sorted(set(lease.devices) & set(wanted))}"
                for lease in clash
            )
            raise LeaseConflictError(
                f"CUDA {wanted} is already leased: {names}. Release it first, or wait for it "
                f"to idle out.",
                param="devices",
                details={"leases": [lease_view(lease) for lease in clash]},
            )
        stamp = time.time() if now is None else now
        lease = GpuLease(
            id=uuid.uuid4().hex[:12],
            devices=wanted,
            holder=holder,
            model_ids=list(dict.fromkeys(model_ids)),
            reason=reason,
            created_at=stamp,
            last_activity_at=stamp,
            idle_ttl_s=idle_ttl_s,
        )
        self._leases[lease.id] = lease
        return lease

    def release(self, lease_id: str) -> GpuLease:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            raise LeaseNotFoundError(
                f"no lease '{lease_id}'; GET /api/leases (or server_status.leases) lists the "
                f"standing ones",
                param="lease_id",
            )
        return lease

    def touch(self, lease_id: str, *, at: float | None = None) -> GpuLease:
        """Refresh a lease's activity clock -- never backwards."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"no lease '{lease_id}'", param="lease_id")
        stamp = time.time() if at is None else at
        if stamp > lease.last_activity_at:
            lease.last_activity_at = stamp
        return lease

    def blocked_for(self, model_id: str | None) -> frozenset[int]:
        """Devices ``model_id`` may NOT be planned onto: every lease that does not name it."""
        blocked: set[int] = set()
        for lease in self._leases.values():
            if model_id is None or model_id not in lease.model_ids:
                blocked.update(lease.devices)
        return frozenset(blocked)

    def for_model(self, model_id: str) -> GpuLease | None:
        """The lease that names ``model_id``, if any (a model holds at most one)."""
        for lease in self.all():
            if model_id in lease.model_ids:
                return lease
        return None

    def expired(self, now: float | None = None) -> list[GpuLease]:
        stamp = time.time() if now is None else now
        return [
            lease
            for lease in self.all()
            if lease.idle_ttl_s is not None and stamp - lease.last_activity_at >= lease.idle_ttl_s
        ]


def lease_view(lease: GpuLease) -> dict[str, Any]:
    """The API/MCP projection: the stored fields plus the derived clocks."""
    return {
        **lease.model_dump(mode="json"),
        "idle_s": round(lease.idle_s),
        "expires_at": lease.expires_at,
    }
