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

#: Idle seconds past which a lease is reported ``"idle"`` rather than
#: ``"active"``. A held-but-quiet lease is the shape a crashed holder leaves
#: behind, and until now nothing said so out loud: a consumer saw the same
#: record for a benchmark generating flat out and for one whose process died
#: an hour ago.
LEASE_IDLE_AFTER_S = 300.0
#: How close to the sweep's release counts as ``"expiring"``.
LEASE_EXPIRING_WITHIN_S = 300.0
#: Cap on the retry advice. The honest full answer is ``expires_at``; a client
#: that sleeps two hours on one number is a client that never notices an early
#: release, so the advice is "come back and re-ask", not "wait it out".
LEASE_RETRY_CAP_S = 300.0
#: Retry advice for a lease with ``idle_ttl_s = None`` -- held until released.
#: There is no clock to count down, so rather than invent one (or say nothing,
#: which reads as "never retry") the advice is a short re-ask interval.
LEASE_OPEN_ENDED_RETRY_S = 60.0

#: ``holder_family`` -> a coarse description of the work holding the cards.
#: **Descriptive only, and deliberately not policy**: the book stays strictly
#: first-come-first-served (D43) and nothing anywhere preempts on this value.
#: It exists because "stand down entirely" (a benchmark owns four cards for an
#: hour) and "wait ninety seconds" (someone is rendering one picture) are
#: opposite answers that a holder string could not tell apart.
LEASE_KINDS: dict[str, str] = {
    "crucibleforge": "benchmark",
    "benchmark": "benchmark",
    "clawforge": "render",
    "clawforge2": "render",
    "comfyui": "render",
    "openclaw": "agent",
    "agent": "agent",
}


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


def holder_family(holder: str) -> str:
    """The stable half of a holder name.

    CrucibleForge leases as ``crucibleforge`` for the run and
    ``crucibleforge-judge`` for the judge phase, so a consumer doing exact
    holder matching saw NO lease at all for the whole judging window -- a real
    outage class on the client side. The family is everything before the first
    ``-``: one rule, no registry, and both phases answer ``crucibleforge``.

    **Exposure only.** :meth:`LeaseBook.blocked_for` keys on ``model_ids`` and
    never on the holder, and nothing in this module or the manager matches on
    the family. It is published so a *client* can match ``crucibleforge*``
    without exact-string fragility; wiring it into the book would turn a
    display convenience into placement policy.
    """
    return (holder or "").split("-", 1)[0].strip().lower() or (holder or "")


def lease_kind(holder: str) -> str:
    """``benchmark`` | ``render`` | ``agent`` | ``other`` for a holder name.

    Derived from :func:`holder_family` in this one place so the REST view, the
    MCP view and a 507's lease records cannot drift into three answers. See
    :data:`LEASE_KINDS`: descriptive, never enforced.
    """
    return LEASE_KINDS.get(holder_family(holder), "other")


def lease_state(lease: GpuLease, now: float | None = None) -> str:
    """``active`` | ``idle`` | ``expiring`` -- expiry outranks idleness.

    A lease inside :data:`LEASE_EXPIRING_WITHIN_S` of the sweep is reported
    ``expiring`` even if it is being touched, because that is the fact a
    waiting client needs; a lease with no TTL is never ``expiring``.

    ``now`` is read once and both comparisons use it, so a caller can inject a
    clock and get a consistent answer rather than one derived half from the
    argument and half from :func:`time.time`.
    """
    stamp = time.time() if now is None else now
    expires = lease.expires_at
    if expires is not None and expires - stamp <= LEASE_EXPIRING_WITHIN_S:
        return "expiring"
    idle = max(0.0, stamp - lease.last_activity_at)
    return "idle" if idle >= LEASE_IDLE_AFTER_S else "active"


def lease_view(lease: GpuLease) -> dict[str, Any]:
    """The API/MCP projection: the stored fields plus the derived clocks.

    ``idle_s``/``expires_at`` are ``@property`` on :class:`GpuLease`, so a bare
    ``model_dump`` drops them -- every consumer of a lease record goes through
    here (D53) rather than dumping the model, or it gets a record that cannot
    answer "is this lease still alive?".
    """
    now = time.time()
    expires = lease.expires_at
    return {
        **lease.model_dump(mode="json"),
        "idle_s": round(lease.idle_s),
        "expires_at": expires,
        "state": lease_state(lease, now),
        "holder_family": holder_family(lease.holder),
        "kind": lease_kind(lease.holder),
        # Not the full wait: capped, because an early release is common and a
        # client asleep for two hours would never see it.
        "retry_after_s": (
            None if expires is None else max(1, int(min(expires - now, LEASE_RETRY_CAP_S)))
        ),
    }
