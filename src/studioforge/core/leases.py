"""GPU leases: a claim on specific cards that the planner honours (D43).

A lease says "these CUDA devices belong to *this* for now". While it stands,
only the models it names may be planned onto those cards -- everything else
(a JIT load, the pin reconciler, the rebalancer, a benchmark of some other
model) sees the cards as absent and places elsewhere or is refused with the
lease named. A lease with no models holds the cards for something outside
this server entirely (a ComfyUI run, a training job).

The book lives in memory and is mirrored to the registry (D61). It was
in-memory only until 2026-09-10, on the theory that a lease describes a live
situation and a restart is a clean slate; that morning a restart emptied the
book and two tenants' loads landed on the card an image-generation tenant had
leased for ComfyUI, and every re-ask since answered ``503 busy`` because a lease never
interrupts a stream (D36). Now every acquire, release and vacate mark is
written through, touches are written on a coalesced clock, and a restart
restores the standing leases with their clocks -- a holder that did
everything right keeps its card. What keeps the book honest is still the
idle TTL: the sweep releases a lease nobody has touched for that long, a
row idle past it is dropped at restore rather than re-entered, and this
server's own benchmark leases -- whose holder cannot outlive the process --
are never restored at all. So a crashed benchmark, a forgotten reservation
or a holder that died while the server was down cannot hold a card forever.

Vacating (D56): a lease carries the D46 class of its claim, and a holder may
register a ``vacate_url``. When a strictly better class asks for the cards,
the server POSTs one vacate request to that URL, marks the lease
``vacating`` and answers 409 ``lease_vacating`` with a re-ask interval. The
holder releases -- or does not, and the deadline turns the answer back into
today's plain ``lease_conflict``. The book never takes a lease away.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import ipaddress
import re
import socket
import ssl
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from studioforge.core.priority import PRIORITY_BACKGROUND, PRIORITY_LEVELS
from studioforge.errors import (
    BadRequestError,
    LeaseConflictError,
    LeaseNotFoundError,
    StudioForgeError,
)
from studioforge.logging import get_logger
from studioforge.types import GpuLease, LeaseInstance

log = get_logger(__name__)

#: How long a lease survives with no activity before the sweep releases it.
DEFAULT_IDLE_TTL_S = 3600.0

# -- persistence (D61) --------------------------------------------------------

#: Touches are coalesced: a keep-alive every second must not become a disk
#: write every second, so a touch is mirrored only when this long has passed
#: since the row was last written. Acquire, release and the vacate marks are
#: always written. The cost of the gap is bounded by it: a restore can see a
#: ``last_activity_at`` at most this much older than the holder's last touch.
TOUCH_PERSIST_INTERVAL_S = 5.0

#: Holder families whose leases belong to THIS process: the in-process
#: benchmarker leases as ``benchmark`` and ``benchmark:parallel``. Such a
#: holder cannot outlive a restart, so its rows are dropped at restore rather
#: than held for an hour of idle TTL on behalf of a run that no longer
#: exists. ``crucibleforge`` is a separate process and is NOT here, whatever
#: :data:`LEASE_KINDS` calls it.
OWN_HOLDER_FAMILIES: frozenset[str] = frozenset({"benchmark"})

# -- vacate (D56) -------------------------------------------------------------

#: Header carrying the holder's own token back to it, so a receiver can tell a
#: real vacate request from anything else that finds its URL.
VACATE_TOKEN_HEADER = "X-SF-Vacate-Token"
#: Always sent: the one thing a receiver needs to correlate.
VACATE_LEASE_HEADER = "X-SF-Lease-Id"
VACATE_URL_MAX_CHARS = 512
VACATE_TOKEN_MAX_CHARS = 512
#: How long the holder has to release before the ask degrades to a plain 409.
DEFAULT_VACATE_TIMEOUT_S = 180.0
#: The re-ask interval handed to a requester while the holder is vacating.
DEFAULT_VACATE_RETRY_AFTER_S = 15.0
#: Per-request HTTP timeout for the vacate POST.
DEFAULT_VACATE_CALLBACK_TIMEOUT_S = 30.0
_VACATE_SCHEMES = ("http", "https")
_VACATE_DEFAULT_PORTS = {"http": 80, "https": 443}
#: Printable ASCII, no whitespace: anything else httpx/h11 refuse -- and quote
#: back in the exception text, which is how a token ends up in a log line.
_VACATE_TOKEN_RE = re.compile(r"[\x21-\x7e]+")

#: Idle seconds past which a lease is reported ``"idle"`` rather than
#: ``"active"``. A held-but-quiet lease is the shape a crashed holder leaves
#: behind, and until now nothing said so out loud: a consumer saw the same
#: record for a benchmark generating flat out and for one whose process died
#: an hour ago. A CAP: a short lease goes idle at :data:`LEASE_IDLE_FRACTION`
#: of its TTL instead (see :func:`lease_idle_after_s`).
LEASE_IDLE_AFTER_S = 300.0
#: How close to the sweep's release counts as ``"expiring"``. Also a cap: a
#: short lease is ``expiring`` for its last :data:`LEASE_EXPIRING_FRACTION`
#: only (see :func:`lease_expiring_within_s`). Absolute, this made a fresh
#: 300 s lease ``expiring`` from its first second and a 600 s render lease
#: ``expiring`` for half its life -- seen live on 2026-09-04 -- which turned
#: the one word a waiting client reads into noise.
LEASE_EXPIRING_WITHIN_S = 300.0
#: The share of a lease's TTL after which it is ``idle`` (cap: the absolute).
LEASE_IDLE_FRACTION = 0.5
#: The share of a lease's TTL that counts as ``expiring`` (cap: the absolute).
LEASE_EXPIRING_FRACTION = 0.25
#: Cap on the retry advice. The honest full answer is ``expires_at``; a client
#: that sleeps two hours on one number is a client that never notices an early
#: release, so the advice is "come back and re-ask", not "wait it out".
LEASE_RETRY_CAP_S = 300.0
#: Retry advice for a lease with ``idle_ttl_s = None`` -- held until released.
#: There is no clock to count down, so rather than invent one (or say nothing,
#: which reads as "never retry") the advice is a short re-ask interval.
LEASE_OPEN_ENDED_RETRY_S = 60.0

# -- the canonical shape (B.0 standards-first, D87) ---------------------------

#: Kubernetes ``coordination.k8s.io/v1`` Lease field names are the canonical
#: lease shape: ``holder_identity``, ``lease_duration_seconds``,
#: ``acquire_time``, ``renew_time``, ``lease_transitions``, plus an
#: ``allocation.devices[]`` echo in DRA's vocabulary. Times stay Unix epoch
#: seconds -- every other clock this API publishes is one, and the legacy twin
#: must carry the EQUAL value -- rather than k8s MicroTime strings. Each
#: StudioForge name below stays on every lease record as a deprecated alias
#: with the same value for the transition window (``deprecated_fields`` in the
#: envelope says which), mapped to the canonical name a client moves to.
LEASE_FIELD_ALIASES: dict[str, str] = {
    "holder": "holder_identity",
    "idle_ttl_s": "lease_duration_seconds",
    "created_at": "acquire_time",
    "last_activity_at": "renew_time",
    "lease_id": "id",
    "expires_in_s": "expires_at",
}
#: The alias names, in a stable order, for the ``deprecated_fields`` envelope key.
LEASE_DEPRECATED_FIELDS: tuple[str, ...] = tuple(LEASE_FIELD_ALIASES)

#: When the StudioForge-specific lease surfaces were deprecated (this release),
#: as the RFC 9745 ``Deprecation`` structured-field date ``@<epoch>``. No
#: ``Sunset`` yet: that is set per surface once the clients report "migrated"
#: (lead decision F-10).
ALIASES_DEPRECATED_AT = 1794182400  # 2026-11-09T00:00:00Z
#: Where a client reads what replaces a deprecated surface (F-10).
MIGRATION_DOC = "docs/MIGRATION.md"


def deprecation_headers() -> dict[str, str]:
    """``Deprecation`` + ``Link: rel="deprecation"`` for a response from a deprecated alias.

    Only on a response whose RESOURCE is the deprecated one (``POST
    /api/leases/{id}/touch``, superseded by ``PATCH /api/leases/{id}``); a
    canonical route that merely still carries alias FIELDS says so in its body
    (``deprecated_fields``) instead, because RFC 9745 deprecates the resource,
    and these routes are not deprecated.
    """
    return {
        "Deprecation": f"@{ALIASES_DEPRECATED_AT}",
        "Link": f'<{MIGRATION_DOC}>; rel="deprecation"',
    }


#: ``holder_family`` -> a coarse description of the work holding the cards.
#: **Descriptive only, and deliberately not policy**: the book stays strictly
#: first-come-first-served (D43) and nothing anywhere preempts on this value.
#: It exists because "stand down entirely" (a benchmark owns four cards for an
#: hour) and "wait ninety seconds" (someone is rendering one picture) are
#: opposite answers that a holder string could not tell apart.
#:
#: These are the built-in families. An operator whose own programs lease under
#: other names adds them with ``leases.holder_kinds`` in config.yaml, which
#: :func:`configure_holder_kinds` merges over this table.
LEASE_KINDS: dict[str, str] = {
    "crucibleforge": "benchmark",
    "benchmark": "benchmark",
    "imagegen": "render",
    "comfyui": "render",
    "openclaw": "agent",
    "agent": "agent",
}

#: The values :data:`LEASE_KINDS` and ``leases.holder_kinds`` may map a family to.
LEASE_KIND_VALUES: frozenset[str] = frozenset({"benchmark", "render", "agent", "other"})

#: ``leases.holder_kinds`` from config.yaml, set by :func:`configure_holder_kinds`.
#: Consulted before :data:`LEASE_KINDS`, so an operator can also re-describe a
#: built-in family.
_CONFIGURED_KINDS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# The mirror (D61)
# ---------------------------------------------------------------------------


class LeaseStore(Protocol):
    """Where the book mirrors itself. ``load`` answers oldest first.

    A store may raise: the book catches, warns once per failure kind and
    carries on, because the in-memory book is authoritative and a disk
    hiccup must never refuse or lose a lease.
    """

    def put(self, lease: GpuLease) -> None: ...

    def delete(self, lease_id: str) -> None: ...

    def load(self) -> list[GpuLease]: ...


class LeaseRows(Protocol):
    """The slice of :class:`studioforge.db.Database` the store adapter needs."""

    def save_lease(self, row: dict[str, Any]) -> None: ...

    def delete_lease(self, lease_id: str) -> None: ...

    def list_leases(self) -> list[dict[str, Any]]: ...


def lease_row(lease: GpuLease) -> dict[str, Any]:
    """The stored shape of a lease: the standing fields, token and peer included.

    The vacate window (``vacate_requested_at`` .. ``vacate_delivery_status``)
    and ``restored_at`` are deliberately absent: a restart ends any ask in
    flight, and the next better-class ask re-runs D56 from scratch.
    """
    return {
        "id": lease.id,
        "devices": list(lease.devices),
        "holder": lease.holder,
        "model_ids": list(lease.model_ids),
        "reason": lease.reason,
        "created_at": lease.created_at,
        "last_activity_at": lease.last_activity_at,
        "idle_ttl_s": lease.idle_ttl_s,
        "priority": lease.priority,
        "vacate_url": lease.vacate_url,
        "vacate_token": lease.vacate_token,
        "holder_peer": lease.holder_peer,
        "holder_key": lease.holder_key,
        # D91: the instances the lease created, spec and last-loaded shape.
        "instances": [spec.model_dump(mode="json") for spec in lease.instances],
        "updated_at": time.time(),
    }


def lease_from_row(row: Mapping[str, Any]) -> GpuLease:
    """A lease from its stored shape; the model validates what the row claims."""
    idle_ttl = row.get("idle_ttl_s")
    return GpuLease(
        id=str(row["id"]),
        devices=[int(d) for d in row["devices"]],
        holder=str(row["holder"]),
        model_ids=[str(m) for m in row.get("model_ids") or []],
        reason=str(row.get("reason") or ""),
        created_at=float(row["created_at"]),
        last_activity_at=float(row["last_activity_at"]),
        idle_ttl_s=None if idle_ttl is None else float(idle_ttl),
        priority=int(row.get("priority") or PRIORITY_BACKGROUND),
        vacate_url=row.get("vacate_url") or None,
        vacate_token=row.get("vacate_token") or None,
        holder_peer=row.get("holder_peer") or None,
        holder_key=row.get("holder_key") or None,
        instances=_instances_from_row(row),
    )


def _instances_from_row(row: Mapping[str, Any]) -> list[LeaseInstance]:
    """The stored instance specs of a lease row (D91); an unreadable one is skipped, said once."""
    out: list[LeaseInstance] = []
    for raw in row.get("instances") or []:
        try:
            out.append(LeaseInstance.model_validate(raw))
        except Exception as exc:  # noqa: BLE001 - one bad spec must not lose the lease
            log.warning(
                "lease instance not restored: unreadable spec",
                lease_id=row.get("id"),
                error=type(exc).__name__,
            )
    return out


class DatabaseLeaseStore:
    """:class:`LeaseStore` over the registry's ``gpu_leases`` table."""

    def __init__(self, db: LeaseRows) -> None:
        self._db = db

    def put(self, lease: GpuLease) -> None:
        self._db.save_lease(lease_row(lease))

    def delete(self, lease_id: str) -> None:
        self._db.delete_lease(lease_id)

    def load(self) -> list[GpuLease]:
        out: list[GpuLease] = []
        for row in self._db.list_leases():
            try:
                out.append(lease_from_row(row))
            except Exception as exc:  # noqa: BLE001 - one bad row must not lose the rest
                log.warning(
                    "gpu lease row skipped at restore: unreadable",
                    lease_id=row.get("id") if isinstance(row, Mapping) else None,
                    error=type(exc).__name__,
                )
        return out


def lease_store_for(db: Any) -> DatabaseLeaseStore | None:
    """A store over ``db`` when it has the lease table's methods, else ``None``.

    Duck-typed so a manager built over a stub database -- most unit tests --
    keeps an unmirrored book rather than warning on every acquire.
    """
    if db is None:
        return None
    needed = ("save_lease", "delete_lease", "list_leases")
    if all(callable(getattr(db, name, None)) for name in needed):
        return DatabaseLeaseStore(db)
    return None


class LeaseBook:
    """Every standing lease, keyed by id. Single-threaded: lives on the event loop.

    With a :class:`LeaseStore` the book writes through (D61): every acquire,
    release and vacate mark at once, touches on the
    :data:`TOUCH_PERSIST_INTERVAL_S` clock. Without one it is the pre-D61
    in-memory book to the byte.
    """

    def __init__(self, store: LeaseStore | None = None) -> None:
        self._leases: dict[str, GpuLease] = {}
        self._store = store
        #: lease id -> the clock value the row was last written at, for the
        #: touch coalescing. Absent for a row whose last write failed, so the
        #: next touch retries it.
        self._written_at: dict[str, float] = {}
        #: Failure kinds (``op:ExceptionName``) already warned about during the
        #: current run of failures; cleared by the next success so a store
        #: that comes back and fails again is heard again.
        self._store_warned: set[str] = set()

    @property
    def store(self) -> LeaseStore | None:
        return self._store

    def bind_store(self, store: LeaseStore) -> None:
        """Attach the mirror to a book built without one.

        The app builds the book before the manager -- the planner shares it
        -- and the manager is the one holding the database, so the binding
        happens there rather than at construction. Nothing standing is
        written retroactively: the book is empty when this is called at
        boot, and :meth:`restore` follows.
        """
        self._store = store

    def _persist(self, lease: GpuLease, *, at: float) -> None:
        if self._store is None:
            return
        try:
            self._store.put(lease)
        except Exception as exc:  # noqa: BLE001 - the book is authoritative; disk is a mirror
            self._store_failed("put", exc)
            return
        self._written_at[lease.id] = at
        self._store_warned.clear()

    def _forget(self, lease_id: str) -> None:
        self._written_at.pop(lease_id, None)
        if self._store is None:
            return
        try:
            self._store.delete(lease_id)
        except Exception as exc:  # noqa: BLE001 - see _persist
            self._store_failed("delete", exc)
            return
        self._store_warned.clear()

    def _store_failed(self, op: str, exc: BaseException) -> None:
        kind = f"{op}:{type(exc).__name__}"
        if kind in self._store_warned:
            log.debug("gpu lease store failed again", op=op, error=repr(exc))
            return
        self._store_warned.add(kind)
        log.warning(
            "gpu lease store failed; the in-memory book stays authoritative and this lease "
            "will not survive a restart until a later write succeeds",
            op=op,
            error=repr(exc),
        )

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
        priority: int = PRIORITY_BACKGROUND,
        vacate_url: str | None = None,
        vacate_token: str | None = None,
        holder_peer: str | None = None,
        holder_key: str | None = None,
        now: float | None = None,
    ) -> GpuLease:
        """Record a lease, or raise :class:`LeaseConflictError` if a card is taken.

        The conflict is always the plain ``lease_conflict`` here: whether a
        better class may *ask* the holder to leave is the manager's call
        (:meth:`ModelManager.acquire_lease`), made before this is reached.
        ``holder_peer`` is the address the lease was registered from over
        HTTP (D55) -- stored, never shown.
        """
        wanted = sorted({int(d) for d in devices})
        if not wanted:
            raise BadRequestError("a lease must name at least one CUDA device", param="devices")
        if idle_ttl_s is not None and idle_ttl_s <= 0:
            raise BadRequestError(
                "idle_ttl_s must be positive, or null for 'until released'", param="idle_ttl_s"
            )
        if isinstance(priority, bool) or priority not in PRIORITY_LEVELS:
            raise BadRequestError(
                "priority must be 1 (active chat), 2 (dispatched agent) or 3 (background)",
                param="priority",
            )
        url = validate_vacate_url(vacate_url) if vacate_url is not None else None
        token = validate_vacate_token(vacate_token, has_url=url is not None)
        clash = self.conflicts(wanted)
        if clash:
            raise self.conflict_error(wanted, clash)
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
            priority=priority,
            vacate_url=url,
            vacate_token=token,
            holder_peer=(holder_peer or "").strip() or None,
            holder_key=(holder_key or "").strip() or None,
        )
        self._leases[lease.id] = lease
        self._persist(lease, at=stamp)
        return lease

    @staticmethod
    def conflict_error(
        wanted: list[int], clash: list[GpuLease], *, now: float | None = None
    ) -> LeaseConflictError:
        """Today's 409, phrased by the book, with the wait in the header's place.

        ``details.retry_after_s`` is the shortest of the clashing rows' own
        advice (open-ended rows count :data:`LEASE_OPEN_ENDED_RETRY_S`), so the
        API layer emits ``Retry-After`` for a lease conflict the way it already
        does for a leased 507 -- a client that only reads headers was told
        nothing before.
        """
        stamp = time.time() if now is None else now
        names = "; ".join(
            f"lease {lease.id} ({lease.holder}) holds "
            f"CUDA {sorted(set(lease.devices) & set(wanted))}"
            for lease in clash
        )
        rows = [lease_view(lease, now=stamp) for lease in clash]
        waits = [
            row["retry_after_s"] if row["retry_after_s"] is not None else LEASE_OPEN_ENDED_RETRY_S
            for row in rows
        ]
        # ``code`` is the specific refusal (D87): ``error.code`` stays the
        # ``lease_conflict`` every client already branches on, and this one is
        # sharpened by the manager to ``lease_outranked`` when the requester's
        # class could not have asked the holder to leave.
        details: dict[str, Any] = {
            "code": "lease_conflict",
            "leases": rows,
            "retry_after_s": max(1, int(min(waits))),
        }
        timed_out = [
            lease
            for lease in clash
            if lease.vacate_deadline is not None and stamp >= lease.vacate_deadline
        ]
        if timed_out:
            # A vacate was asked and the window closed without a release: say so,
            # so the requester can tell "never asked" from "asked, and refused"
            # -- and "asked, and the holder never heard" (``undeliverable``: the
            # POST failed and the window was collapsed on the spot), which is
            # the case where waiting for a release is waiting for nothing.
            undeliverable = [lease.id for lease in timed_out if lease.vacate_delivery == "failed"]
            details["vacate"] = {
                "state": "undeliverable" if len(undeliverable) == len(timed_out) else "timed_out",
                "leases": [lease.id for lease in timed_out],
                "undeliverable": undeliverable,
                "reask_at": min(
                    stamp_at
                    for stamp_at in (_vacate_reask_at(lease) for lease in timed_out)
                    if stamp_at is not None
                ),
            }
        return LeaseConflictError(
            f"CUDA {wanted} is already leased: {names}. Release it first, or wait for it "
            f"to idle out.",
            param="devices",
            details=details,
        )

    def mark_vacating(
        self, lease_id: str, *, requested_by: str, deadline_s: float, now: float | None = None
    ) -> GpuLease:
        """Open a vacate window on a lease: stamped, never sent from here."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"no lease '{lease_id}'", param="lease_id")
        stamp = time.time() if now is None else now
        lease.vacate_requested_at = stamp
        lease.vacate_requested_by = requested_by
        lease.vacate_deadline = stamp + deadline_s
        # A quiet period equal to the window follows a lapse: bounded on both
        # sides, so a requester that keeps re-asking costs the holder one POST
        # per two windows, and a holder that restarted mid-window (and
        # re-adopted its lease) is asked again rather than never.
        lease.vacate_reask_at = stamp + 2 * deadline_s
        lease.vacate_delivery = "pending"
        lease.vacate_delivery_status = None
        self._persist(lease, at=stamp)
        return lease

    def mark_vacate_delivery(
        self,
        lease_id: str,
        *,
        delivered: bool,
        status: str,
        quiet_s: float,
        now: float | None = None,
    ) -> GpuLease | None:
        """Record what became of the vacate POST; a failure ends the window now.

        ``None`` when the lease is already gone (released or swept while the
        POST was in flight -- the protocol worked, there is nothing to mark).
        On a failure the deadline is pulled in to ``now``: the asker's next
        re-ask is answered ``lease_conflict`` with ``vacate.state:
        "undeliverable"`` instead of ``lease_vacating`` for the rest of a
        window the holder never heard about. The holder may be asked again
        after ``quiet_s`` (the window length) -- a dead holder costs one
        failed connect per window while someone keeps asking, and a holder
        that was merely restarting gets the next ask on the same clock a
        lapse would have given it.
        """
        lease = self._leases.get(lease_id)
        if lease is None or lease.vacate_deadline is None:
            return None
        stamp = time.time() if now is None else now
        lease.vacate_delivery_status = status
        if delivered:
            lease.vacate_delivery = "delivered"
            self._persist(lease, at=stamp)
            return lease
        lease.vacate_delivery = "failed"
        if stamp < lease.vacate_deadline:
            lease.vacate_deadline = stamp
        lease.vacate_reask_at = stamp + max(0.0, quiet_s)
        self._persist(lease, at=stamp)
        return lease

    def release(self, lease_id: str) -> GpuLease:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            raise LeaseNotFoundError(
                f"no lease '{lease_id}'; GET /api/leases (or server_status.leases) lists the "
                f"standing ones",
                param="lease_id",
            )
        self._forget(lease.id)
        return lease

    def touch(self, lease_id: str, *, at: float | None = None) -> GpuLease:
        """Refresh a lease's activity clock -- never backwards.

        Mirrored on the :data:`TOUCH_PERSIST_INTERVAL_S` clock: a keep-alive
        every second is one disk write per five, and a clock that did not
        move is no write at all.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"no lease '{lease_id}'", param="lease_id")
        stamp = time.time() if at is None else at
        if stamp > lease.last_activity_at:
            lease.last_activity_at = stamp
            written = self._written_at.get(lease.id)
            if written is None or stamp - written >= TOUCH_PERSIST_INTERVAL_S:
                self._persist(lease, at=stamp)
        return lease

    def bind_models(
        self, lease_id: str, model_ids: Iterable[str], *, now: float | None = None
    ) -> list[str]:
        """Name more models in a standing lease; returns the ids that were NEW to it (D87).

        How a load bound with ``lease_id`` (B.6) or a resident copy the grant
        spared (A-1) becomes an owner: :meth:`blocked_for` then lets it onto
        the lease's cards, the planner's ladders treat it as the holder's, and
        the release unloads it. Written through like an acquire. The caller
        has already refused a model another lease names -- a model belongs to
        at most one lease (:meth:`for_model`).
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"no lease '{lease_id}'", param="lease_id")
        added = [m for m in dict.fromkeys(model_ids) if m and m not in lease.model_ids]
        if added:
            lease.model_ids = [*lease.model_ids, *added]
            self._persist(lease, at=time.time() if now is None else now)
        return added

    @contextlib.contextmanager
    def provisionally_named(self, lease_id: str | None, model_id: str) -> Iterator[None]:
        """Count ``model_id`` as named by ``lease_id`` for a synchronous dry run only (D87).

        In memory, never written to the store, and undone on the way out: the
        dry run of a ``lease_id`` load must plan the way the bound load would,
        and a decision with no ``await`` inside cannot be observed half-way by
        anything else on the loop. ``None`` (or an unknown lease, or a model it
        already names) changes nothing.
        """
        lease = self._leases.get(lease_id) if lease_id is not None else None
        if lease is None or model_id in lease.model_ids:
            yield
            return
        lease.model_ids = [*lease.model_ids, model_id]
        try:
            yield
        finally:
            lease.model_ids = [m for m in lease.model_ids if m != model_id]

    def unbind_models(self, lease_id: str, model_ids: Iterable[str]) -> None:
        """Take models back out of a lease -- the undo of a bind whose load failed."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return
        drop = set(model_ids)
        kept = [m for m in lease.model_ids if m not in drop]
        if kept != lease.model_ids:
            lease.model_ids = kept
            self._persist(lease, at=time.time())

    # -- instances (D91) ---------------------------------------------------

    def attach_instances(
        self, lease_id: str, instances: Iterable[LeaseInstance], *, now: float | None = None
    ) -> list[LeaseInstance]:
        """Give a standing lease these instances; each id is named in ``model_ids`` too (D91).

        Written through like an acquire, so a restart brings them back. The
        caller has already refused an id that collides with anything standing.
        """
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"no lease '{lease_id}'", param="lease_id")
        instances = list(instances)
        for spec in instances:
            for other in self._leases.values():
                if other.id != lease_id and any(i.id == spec.id for i in other.instances):
                    # D91: an instance id belongs to one lease, ever at once.
                    raise StudioForgeError(
                        f"the instance id '{spec.id}' is held by lease {other.id} "
                        f"({other.holder}); an instance belongs to one lease",
                        code="instance_id_conflict",
                        status_code=409,
                        param="instances",
                        details={"id": spec.id, "lease_id": other.id},
                    )
        added = [spec for spec in instances if spec.id not in {i.id for i in lease.instances}]
        if added:
            lease.instances = [*lease.instances, *added]
            ids = [spec.id for spec in added if spec.id not in lease.model_ids]
            lease.model_ids = [*lease.model_ids, *ids]
            self._persist(lease, at=time.time() if now is None else now)
        return added

    def detach_instances(self, lease_id: str, instance_ids: Iterable[str]) -> list[LeaseInstance]:
        """Take instances out of a lease (and out of its ``model_ids``); returns what went."""
        lease = self._leases.get(lease_id)
        if lease is None:
            return []
        drop = set(instance_ids)
        gone = [spec for spec in lease.instances if spec.id in drop]
        if gone:
            lease.instances = [spec for spec in lease.instances if spec.id not in drop]
            lease.model_ids = [m for m in lease.model_ids if m not in drop]
            self._persist(lease, at=time.time())
        return gone

    def note_instance_loaded(self, instance_id: str, shape: Mapping[str, Any]) -> None:
        """Remember the shape an instance ACTUALLY loaded with, for the restore (D91)."""
        found = self.instance(instance_id)
        if found is None:
            return
        lease, spec = found
        shape = dict(shape)
        if spec.loaded_spec == shape:
            return
        spec.loaded_spec = shape
        self._persist(lease, at=time.time())

    def instance(self, instance_id: str) -> tuple[GpuLease, LeaseInstance] | None:
        """The lease holding instance ``instance_id`` and its spec, if one does."""
        for lease in self._leases.values():
            for spec in lease.instances:
                if spec.id == instance_id:
                    return lease, spec
        return None

    def instances(self) -> list[tuple[GpuLease, LeaseInstance]]:
        """Every standing instance with its lease, oldest lease first."""
        return [(lease, spec) for lease in self.all() for spec in lease.instances]

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

    def restore(self, now: float | None = None) -> list[GpuLease]:
        """Re-enter the leases the store holds; the ones a restart should keep (D61).

        Oldest first, and for each row: a lease idle past its ``idle_ttl_s``
        is dropped and deleted from the store -- the sweep would have released
        it had the server stayed up; one held by this server's own benchmark
        (:data:`OWN_HOLDER_FAMILIES`) is dropped the same way, because its
        holder did not survive the restart; one whose cards a lease already
        standing in the book holds is dropped too, that lease having been
        granted by the running server. Everything else comes back with its
        clocks, its ``vacate_url``/``vacate_token`` and its ``holder_peer``,
        the vacate window cleared (a new better-class ask re-runs D56 from
        scratch) and ``restored_at`` stamped. Returns the restored leases.

        A store that cannot be read is a warning and an empty answer, never
        a boot failure: the server serves without its leases the way it did
        before D61, and says so.
        """
        if self._store is None:
            return []
        stamp = time.time() if now is None else now
        try:
            rows = self._store.load()
        except Exception as exc:  # noqa: BLE001 - a mirror that cannot be read is not a boot failure
            self._store_failed("load", exc)
            return []
        restored: list[GpuLease] = []
        for lease in sorted(rows, key=lambda row: (row.created_at, row.id)):
            idle = max(0.0, stamp - lease.last_activity_at)
            if lease.id in self._leases:
                continue  # already standing: nothing to do
            if lease.idle_ttl_s is not None and idle >= lease.idle_ttl_s:
                log.info(
                    "gpu lease not restored: idle past its ttl while the server was down",
                    lease_id=lease.id,
                    holder=lease.holder,
                    devices=list(lease.devices),
                    idle_s=round(idle),
                    idle_ttl_s=lease.idle_ttl_s,
                )
                self._forget(lease.id)
                continue
            if holder_family(lease.holder) in OWN_HOLDER_FAMILIES:
                log.info(
                    "gpu lease not restored: held by this server's own benchmark, which did "
                    "not survive the restart",
                    lease_id=lease.id,
                    holder=lease.holder,
                    devices=list(lease.devices),
                )
                self._forget(lease.id)
                continue
            clash = self.conflicts(lease.devices)
            if clash:
                log.warning(
                    "gpu lease not restored: its cards are held by a lease granted since start",
                    lease_id=lease.id,
                    holder=lease.holder,
                    devices=list(lease.devices),
                    held_by=[other.id for other in clash],
                )
                self._forget(lease.id)
                continue
            lease.vacate_requested_at = None
            lease.vacate_requested_by = None
            lease.vacate_deadline = None
            lease.vacate_reask_at = None
            lease.vacate_delivery = None
            lease.vacate_delivery_status = None
            lease.restored_at = stamp
            self._leases[lease.id] = lease
            # The row on disk is current as of its last write; the coalescing
            # clock starts from the restore so the first keep-alive is not an
            # immediate rewrite of a row that just came back.
            self._written_at[lease.id] = stamp
            restored.append(lease)
        return restored


def holder_family(holder: str) -> str:
    """The stable half of a holder name.

    CrucibleForge leases as ``crucibleforge`` for the run and
    ``crucibleforge-judge`` for the judge phase, so a consumer doing exact
    holder matching saw NO lease at all for the whole judging window -- a real
    outage class on the client side. The family is everything before the first
    ``-`` or ``:``: one rule, no registry, both phases answer ``crucibleforge``
    and this server's own ``benchmark:parallel`` answers ``benchmark`` (it
    read as ``other`` while the rule split on ``-`` alone, which inverted the
    advice a client derives from ``kind``).

    **Exposure only.** :meth:`LeaseBook.blocked_for` keys on ``model_ids`` and
    never on the holder, and nothing in this module or the manager matches on
    the family. It is published so a *client* can match ``crucibleforge*``
    without exact-string fragility; wiring it into the book would turn a
    display convenience into placement policy.
    """
    return re.split(r"[-:]", holder or "", maxsplit=1)[0].strip().lower() or (holder or "")


def lease_kind(holder: str) -> str:
    """``benchmark`` | ``render`` | ``agent`` | ``other`` for a holder name.

    Derived from :func:`holder_family` in this one place so the REST view, the
    MCP view and a 507's lease records cannot drift into three answers. See
    :data:`LEASE_KINDS`: descriptive, never enforced. ``leases.holder_kinds``
    from config.yaml is consulted first.
    """
    family = holder_family(holder)
    return _CONFIGURED_KINDS.get(family) or LEASE_KINDS.get(family, "other")


def configure_holder_kinds(mapping: Mapping[str, str] | None) -> None:
    """Install ``leases.holder_kinds`` (family -> kind) over the built-in table.

    Called whenever the config is built or saved. Keys are lower-cased the way
    :func:`holder_family` lower-cases a holder; values must be one of
    :data:`LEASE_KIND_VALUES` (the config model already refuses anything else).
    ``None`` or ``{}`` restores the built-in table alone.
    """
    _CONFIGURED_KINDS.clear()
    for family, kind in (mapping or {}).items():
        if kind in LEASE_KIND_VALUES:
            _CONFIGURED_KINDS[str(family).strip().lower()] = kind


def lease_idle_after_s(lease: GpuLease) -> float:
    """Idle seconds after which this lease reads ``idle``: relative to its TTL.

    ``min(LEASE_IDLE_AFTER_S, idle_ttl_s * LEASE_IDLE_FRACTION)``: 300 s for
    anything an hour long, half the TTL for a short one, 300 s for a lease
    with no TTL. The three bands then exist for every TTL, in order.
    """
    if lease.idle_ttl_s is None:
        return LEASE_IDLE_AFTER_S
    return min(LEASE_IDLE_AFTER_S, float(lease.idle_ttl_s) * LEASE_IDLE_FRACTION)


def lease_expiring_within_s(lease: GpuLease) -> float:
    """Seconds before the sweep during which this lease reads ``expiring``.

    ``min(LEASE_EXPIRING_WITHIN_S, idle_ttl_s * LEASE_EXPIRING_FRACTION)``:
    the last five minutes of a long lease, the last quarter of a short one.
    Absolute, the threshold made a fresh 300 s lease ``expiring`` at birth
    and a 600 s one ``expiring`` for half its life.
    """
    if lease.idle_ttl_s is None:
        return LEASE_EXPIRING_WITHIN_S
    return min(LEASE_EXPIRING_WITHIN_S, float(lease.idle_ttl_s) * LEASE_EXPIRING_FRACTION)


def lease_state(lease: GpuLease, now: float | None = None) -> str:
    """``vacating`` | ``expiring`` | ``idle`` | ``active``, in that order of precedence.

    A lease inside an open vacate window (D56) is ``vacating`` whatever its
    clocks say: that it has been asked to leave is the fact a waiting client
    needs most. Next, a lease inside :func:`lease_expiring_within_s` of the
    sweep is ``expiring`` even if it is being touched; a lease with no TTL is
    never ``expiring``. Then ``idle`` past :func:`lease_idle_after_s`. Both
    thresholds are relative to the lease's own TTL (capped at the absolutes),
    so a 300 s lease is ``active`` for its first half, ``idle`` for the next
    quarter and ``expiring`` for the last -- not ``expiring`` from birth.

    ``now`` is read once and every comparison uses it, so a caller can inject
    a clock and get a consistent answer rather than one derived half from the
    argument and half from :func:`time.time`.
    """
    stamp = time.time() if now is None else now
    if lease.vacating(stamp):
        return "vacating"
    expires = lease.expires_at
    if expires is not None and expires - stamp <= lease_expiring_within_s(lease):
        return "expiring"
    idle = max(0.0, stamp - lease.last_activity_at)
    return "idle" if idle >= lease_idle_after_s(lease) else "active"


def lease_view(
    lease: GpuLease,
    *,
    now: float | None = None,
    reveal_vacate_url: bool = False,
    gpu_uuids: Mapping[int, str | None] | None = None,
    live: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The API/MCP projection: the stored fields plus the derived clocks.

    ``instances`` (D91) are rendered as reply rows (:func:`~studioforge.core.
    instances.instance_row`): ``live`` maps an instance id to its running
    :class:`~studioforge.types.InstanceInfo` (or ``None``), and with it each
    row carries ``state`` and the shape the child actually runs; without it
    (a caller with no supervisor at hand) ``state`` is ``null``.

    ``idle_s``/``expires_at`` are ``@property`` on :class:`GpuLease`, so a bare
    ``model_dump`` drops them -- every consumer of a lease record goes through
    here (D53) rather than dumping the model, or it gets a record that cannot
    answer "is this lease still alive?".

    Three fields never travel by default: ``vacate_token`` (D56) and
    ``holder_peer`` (D55), like ``holder_key``, are excluded at the model and asserted absent here
    anyway, and ``vacate_url`` is replaced by ``vacate_registered`` -- a
    holder's private endpoint is not something every LAN reader of
    ``GET /api/leases`` needs. Pass ``reveal_vacate_url`` only for a caller
    the D32 gate already trusts with the box.

    The canonical Kubernetes Lease names ride beside the StudioForge ones with
    equal values (D87, :data:`LEASE_FIELD_ALIASES`): ``holder_identity``,
    ``lease_duration_seconds``, ``acquire_time``, ``renew_time`` (epoch
    seconds), ``lease_transitions`` (always 0 -- the book never hands a lease
    to another holder) and ``allocation.devices[{index, uuid}]``. ``uuid`` is
    filled from ``gpu_uuids`` when the caller has the probe's answer, else
    ``null``. ``lease_id`` and ``expires_in_s`` are the legacy spellings some
    clients read.
    """
    stamp = time.time() if now is None else now
    expires = lease.expires_at
    data = lease.model_dump(mode="json")
    data.pop("vacate_token", None)
    data.pop("holder_peer", None)
    data.pop("holder_key", None)
    url = data.pop("vacate_url", None)
    if reveal_vacate_url:
        data["vacate_url"] = url
    uuids = gpu_uuids or {}
    from studioforge.core.instances import instance_row

    data["instances"] = [
        instance_row(
            lease,
            spec,
            live.get(spec.id) if live is not None else None,
            live_known=live is not None,
        )
        for spec in lease.instances
    ]
    return {
        **data,
        "vacate_registered": bool(url),
        "idle_s": round(max(0.0, stamp - lease.last_activity_at)),
        "expires_at": expires,
        "state": lease_state(lease, stamp),
        "holder_family": holder_family(lease.holder),
        "kind": lease_kind(lease.holder),
        # Not the full wait: capped, because an early release is common and a
        # client asleep for two hours would never see it.
        "retry_after_s": (
            None if expires is None else max(1, int(min(expires - stamp, LEASE_RETRY_CAP_S)))
        ),
        # -- the canonical k8s Lease names (D87), equal to their aliases ------
        "holder_identity": lease.holder,
        "lease_duration_seconds": lease.idle_ttl_s,
        "acquire_time": lease.created_at,
        "renew_time": lease.last_activity_at,
        "lease_transitions": 0,
        "allocation": {
            "devices": [{"index": int(d), "uuid": uuids.get(int(d))} for d in lease.devices]
        },
        # -- legacy spellings clients read (deprecated aliases) ---------------
        "lease_id": lease.id,
        "expires_in_s": None if expires is None else round(max(0.0, expires - stamp)),
    }


def gpu_uuid_map(probe: Any) -> dict[int, str | None]:
    """``{index: uuid}`` from a probe's cards, for ``allocation.devices[].uuid``.

    Duck-typed: a card that publishes no ``uuid`` (today's ``GpuInfo``, a stub
    probe) maps to ``None``, and a probe that cannot answer yields ``{}`` --
    never an error on a read of the lease book.
    """
    try:
        cards = list(probe.list_gpus()) if probe is not None else []
    except Exception:  # noqa: BLE001 - a sick probe must not break a lease listing
        return {}
    out: dict[int, str | None] = {}
    for card in cards:
        uuid_value = getattr(card, "uuid", None)
        out[int(card.index)] = str(uuid_value) if uuid_value else None
    return out


# ---------------------------------------------------------------------------
# Vacate (D56): validation and the one outbound request
# ---------------------------------------------------------------------------


def _vacate_reask_at(lease: GpuLease) -> float | None:
    """When a holder whose vacate window closed may be asked again.

    The book stamps it (:meth:`LeaseBook.mark_vacating`, a quiet period equal
    to the window after the deadline; :meth:`LeaseBook.mark_vacate_delivery`,
    a full window after a failed delivery collapsed the deadline). The
    fallback derives it from the clocks for a record stamped before the field
    existed.
    """
    if lease.vacate_deadline is None:
        return None
    if lease.vacate_reask_at is not None:
        return lease.vacate_reask_at
    window = lease.vacate_deadline - (lease.vacate_requested_at or lease.vacate_deadline)
    return lease.vacate_deadline + max(0.0, window)


def vacate_reaskable(lease: GpuLease, now: float | None = None) -> bool:
    """A vacate may be SENT to this lease now: never asked, or the lapse is over."""
    stamp = time.time() if now is None else now
    if lease.vacate_deadline is None:
        return True
    if stamp < lease.vacate_deadline:
        return False  # in flight: dedupe
    reask = _vacate_reask_at(lease)
    return reask is None or stamp >= reask


def validate_vacate_token(token: str | None, *, has_url: bool) -> str | None:
    """A holder-minted bearer: printable ASCII, no whitespace, bounded.

    Refused rather than sent as-is because a header value h11 rejects is
    quoted back in the exception text, and that text is one log line away
    from being the token itself.
    """
    if token is None:
        return None
    if not has_url:
        raise BadRequestError(
            "vacate_token needs a vacate_url to be sent to; give both or neither",
            param="vacate_token",
        )
    if not isinstance(token, str) or not token:
        raise BadRequestError("vacate_token must be a non-empty string", param="vacate_token")
    if len(token) > VACATE_TOKEN_MAX_CHARS:
        raise BadRequestError(
            f"vacate_token is longer than {VACATE_TOKEN_MAX_CHARS} characters", param="vacate_token"
        )
    if _VACATE_TOKEN_RE.fullmatch(token) is None:
        raise BadRequestError(
            "vacate_token must be printable ASCII with no whitespace (it travels in a header)",
            param="vacate_token",
        )
    return token


def validate_vacate_url(url: str) -> str:
    """The shape half of the check: http(s), a host, a sane port, no credentials.

    Whether the target is this server's OWN listener is the manager's half
    (:func:`vacate_url_targets_self`), because only it knows the ports.
    """
    text = (url or "").strip() if isinstance(url, str) else ""
    if not text:
        raise BadRequestError(
            "vacate_url is empty; omit it, or give the http(s) URL this server may POST a "
            "vacate request to",
            param="vacate_url",
        )
    if len(text) > VACATE_URL_MAX_CHARS:
        raise BadRequestError(
            f"vacate_url is longer than {VACATE_URL_MAX_CHARS} characters", param="vacate_url"
        )
    if any(ch.isspace() for ch in text):
        raise BadRequestError("vacate_url contains whitespace", param="vacate_url")
    try:
        parts = urlsplit(text)
    except ValueError as exc:
        raise BadRequestError(
            f"vacate_url could not be parsed ({exc})", param="vacate_url"
        ) from exc
    scheme = (parts.scheme or "").lower()
    if scheme not in _VACATE_SCHEMES:
        raise BadRequestError(
            f"vacate_url must be an http:// or https:// URL, got "
            f"{('scheme ' + scheme) if scheme else 'no scheme'}",
            param="vacate_url",
        )
    if not parts.hostname:
        raise BadRequestError("vacate_url has no host", param="vacate_url")
    try:
        parts.port  # noqa: B018 - the property raises on an unusable port
    except ValueError as exc:
        raise BadRequestError(
            "vacate_url has an unusable port (1-65535)", param="vacate_url"
        ) from exc
    if parts.username or parts.password:
        raise BadRequestError(
            "vacate_url must not embed credentials; use vacate_token", param="vacate_url"
        )
    return text


def vacate_url_target(url: str) -> tuple[str, int]:
    """``(hostname, port)`` of a validated vacate URL, default port filled in."""
    parts = urlsplit(url)
    scheme = (parts.scheme or "http").lower()
    return str(parts.hostname or ""), int(parts.port or _VACATE_DEFAULT_PORTS.get(scheme, 0))


def _literal_ip(text: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """``text`` as an address, including the exotic IPv4 spellings.

    ``ipaddress`` takes dotted-quad and IPv6 only; ``127.1``, ``0x7f000001``
    and ``2130706433`` -- every one of which ``connect()`` routes to loopback
    -- slip past it. ``inet_aton`` accepts them all, so it is the fallback.
    ``None`` for a DNS name or junk.
    """
    raw = (text or "").strip().strip("[]")
    if not raw:
        return None
    try:
        return ipaddress.ip_address(raw)
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(raw)
    except (OSError, ValueError):
        return None
    return ipaddress.ip_address(socket.inet_ntoa(packed))


def _address_is_self(text: str, own_addresses: Iterable[str]) -> bool:
    ip = _literal_ip(text)
    if ip is None:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        ip = mapped
    if ip.is_loopback or ip.is_unspecified:
        return True
    return str(ip) in set(own_addresses)


async def vacate_url_targets_self(
    url: str,
    *,
    own_ports: Iterable[int],
    own_addresses: Iterable[str] = (),
    resolve_timeout_s: float = 2.0,
) -> bool:
    """Whether ``url`` lands on one of THIS server's listeners.

    A vacate request is POSTed from this box with no credential of its own,
    and the D32 gate trusts a loopback peer -- so a ``vacate_url`` aimed at
    ``/api/restart/...`` on our own port would be a remote restart button for
    anyone allowed to register a lease. Only our ports are refused: a holder
    on the same box (an image gateway on its own port) is the expected deployment.

    Literals and ``localhost`` are decided without I/O. A name is resolved
    through the loop's executor with a short cap; a name that cannot be
    resolved is ALLOWED (split-horizon DNS, a resolver that is down), because
    the thing being defended against always resolves.
    """
    host, port = vacate_url_target(url)
    if port not in set(own_ports):
        return False
    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    if _address_is_self(host, own_addresses):
        return True
    if _literal_ip(host) is not None:
        return False  # a literal that is not ours
    try:
        loop = asyncio.get_running_loop()
        infos = await asyncio.wait_for(loop.getaddrinfo(host, port), resolve_timeout_s)
    except (TimeoutError, OSError, ValueError):
        return False
    return any(_address_is_self(str(info[4][0]), own_addresses) for info in infos)


@functools.cache
def _ssl_context() -> ssl.SSLContext | None:
    """The TLS trust store, built once per process.

    ``httpx.AsyncClient()`` builds a fresh ``SSLContext`` and loads the CA
    bundle -- ~250 ms of *synchronous* work -- per client. Built once here,
    off the loop by the caller, and handed to every client since.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - a broken CA bundle must not stop vacates
        try:
            return ssl.create_default_context()
        except Exception:
            return None


def vacate_request_body(
    lease: GpuLease, *, requester: str, requester_priority: int, deadline_s: float
) -> dict[str, Any]:
    """The wire body of a vacate request -- one place, so the docs cannot drift."""
    return {
        "lease_id": lease.id,
        "devices": list(lease.devices),
        "requester": requester,
        "requester_priority": requester_priority,
        "deadline_s": int(deadline_s),
        "deadline_at": lease.vacate_deadline,
    }


async def send_vacate_request(
    lease: GpuLease,
    body: dict[str, Any],
    *,
    timeout_s: float = DEFAULT_VACATE_CALLBACK_TIMEOUT_S,
    transport: httpx.AsyncBaseTransport | None = None,
) -> tuple[bool, str]:
    """POST one vacate request to the lease's URL. Never raises (except cancellation).

    Returns ``(delivered, status)`` where ``status`` is the HTTP code or the
    exception's CLASS NAME -- never ``str(exc)``, which for a header h11
    disliked is the header. Redirects are not followed: a 3xx is "the holder
    moved" and is logged as undelivered rather than chased to wherever the
    Location header points, token and all.
    """
    if not lease.vacate_url:
        return False, "no_url"
    headers = {"Content-Type": "application/json", VACATE_LEASE_HEADER: lease.id}
    if lease.vacate_token:
        headers[VACATE_TOKEN_HEADER] = lease.vacate_token
    kwargs: dict[str, Any] = {
        "timeout": timeout_s,
        "follow_redirects": False,
        # The one request that carries a holder's secret: honouring HTTP(S)_PROXY
        # would hand that header to whatever the environment named.
        "trust_env": False,
    }
    if transport is not None:
        kwargs["transport"] = transport
    else:
        context = await asyncio.to_thread(_ssl_context)
        if context is not None:
            kwargs["verify"] = context
    try:
        async with httpx.AsyncClient(**kwargs) as client:
            response = await client.post(lease.vacate_url, json=body, headers=headers)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a vacate is a courtesy, never a failure here
        return False, type(exc).__name__
    status = int(response.status_code)
    return 200 <= status < 300, str(status)


# ---------------------------------------------------------------------------
# Auto placement (D78): "a card", chosen the way the planner would
# ---------------------------------------------------------------------------

GIB = 1024**3

#: Why a card was or was not chosen; every ``considered[]`` row carries one.
VERDICT_CHOSEN = "chosen"
VERDICT_CANDIDATE = "candidate"
VERDICT_AVOIDED = "avoided"
VERDICT_LEASED = "leased"
VERDICT_BUSY = "busy"
VERDICT_PINNED = "pinned"
VERDICT_BETTER_TIER = "better_tier_resident"
VERDICT_TOO_LITTLE_FREE = "too_little_free"
#: ``planner.excluded_devices`` names the card: no model is ever planned there,
#: so an automatic lease (which judges a card the way a placement would) skips it.
VERDICT_EXCLUDED = "excluded"
#: A model the lease will name may not run on the card: its saved
#: ``device_override`` or ``allowed_devices`` leaves the card out (A-14).
VERDICT_NOT_ALLOWED = "not_allowed"


def card_class_key(gpu: Any) -> str:
    """A portable "these cards are alike" key: architecture + rounded memory (D87).

    The card's own ``class_key`` when the probe publishes one (B.3); otherwise
    the same shape built here -- ``sm_<major><minor>`` from the compute
    capability (or the AMD ``gfx_target``) and the total memory in whole GiB,
    e.g. ``sm_120-32g``. Never a marketing name and never an index, so two
    identical cards on any box group together and a 48 GB workstation card
    does not pass for a 32 GB one.
    """
    published = getattr(gpu, "class_key", None)
    if published:
        return str(published)
    capability = getattr(gpu, "compute_capability", None)
    if capability:
        arch = f"sm_{int(capability[0])}{int(capability[1])}"
    else:
        arch = str(getattr(gpu, "gfx_target", None) or "unknown")
    total = int(getattr(gpu, "total_bytes", 0) or 0)
    return f"{arch}-{round(total / GIB)}g"


class AutoPlacement:
    """What ``devices: "auto"`` decided, and why, card by card.

    ``devices`` is the chosen card set (one card, or ``count`` cards of one
    class), or ``None`` when nothing fit; ``considered`` has one row per card
    the box has, in the order they were ranked (the chosen cards first), each
    with a ``verdict`` from the ``VERDICT_*`` set, its ``class_key`` and the
    numbers the verdict rests on. A tenant that is refused can show its user
    exactly this list.
    """

    __slots__ = ("considered", "count", "devices")

    def __init__(
        self, devices: list[int] | None, considered: list[dict[str, Any]], *, count: int = 1
    ) -> None:
        self.devices = devices
        self.considered = considered
        self.count = count

    @property
    def refused(self) -> bool:
        return self.devices is None

    def verdicts(self) -> set[str]:
        return {str(row["verdict"]) for row in self.considered}

    def view(self) -> dict[str, Any]:
        return {
            "mode": "auto",
            "devices": self.devices,
            "count": self.count,
            "considered": list(self.considered),
        }


def choose_auto_devices(
    gpus: Iterable[Any],
    residents: Iterable[Any],
    book: LeaseBook,
    *,
    priority: int = PRIORITY_BACKGROUND,
    owners: Iterable[str] = (),
    force: bool = False,
    min_free_gib: float | None = None,
    prefer: Iterable[int] = (),
    avoid: Iterable[int] = (),
    excluded: Iterable[int] = (),
    allowed: Iterable[int] | None = None,
    allowed_reason: str | None = None,
    count: int = 1,
) -> AutoPlacement:
    """Pick the card(s) for an exclusive lease, the way the grant itself would judge them (D78).

    The rules are the grant's own (:meth:`ModelManager.acquire_lease`), applied
    per card before anything is touched:

    * a card in ``avoid`` is out;
    * a card in ``excluded`` (``planner.excluded_devices``) is out, and so is
      one outside ``allowed`` -- the cards the lease's own models may run on
      (their saved ``device_override`` / ``allowed_devices``, D87/A-14),
      ``allowed_reason`` saying which setting;
    * a card any standing lease holds is out -- auto placement never asks a
      holder to vacate (D56 is for a claim on *named* cards; an ask for a card
      the asker did not need is noise);
    * a card with a resident mid-request or still loading is out (D36);
    * a card with a pinned idle resident is out unless ``force``;
    * a card with an idle resident of a strictly better tier than ``priority``
      is out unless ``force`` (D46, as D56 item 6 applies it);
    * ``min_free_gib`` is measured against what the card would have once its
      evictable idle residents were unloaded: the probe's free bytes plus the
      planner's own per-card estimate for each of them.

    What is left is ranked: fewest evictions first (an empty card over one whose
    idle resident would have to reload later), then the caller's ``prefer``
    order, then the most free VRAM, then the lowest index. ``owners`` are the
    models the lease will name; a resident that IS an owner counts for nothing
    against its card.

    ``count`` (default 1) asks for that many cards of ONE class
    (:func:`card_class_key`) -- a model split over two cards paces at the
    slower one, so an automatic pair is never a mixed pair; name the cards to
    mix. The class whose best ``count`` cards rank best (same keys, summed)
    wins. No class with ``count`` candidates is a refusal like any other.
    """
    owner_ids = set(owners)
    prefer_rank = {int(d): i for i, d in enumerate(prefer)}
    avoid_set = {int(d) for d in avoid}
    excluded_set = {int(d) for d in excluded}
    allowed_set = None if allowed is None else {int(d) for d in allowed}
    by_device: dict[int, list[Any]] = {}
    for instance in residents:
        plan = getattr(instance, "plan", None)
        if plan is None or getattr(instance, "model_id", None) in owner_ids:
            continue
        for device in plan.devices:
            by_device.setdefault(int(device), []).append(instance)
    rows: list[dict[str, Any]] = []
    for gpu in gpus:
        device = int(gpu.index)
        here = by_device.get(device, [])
        free_now = int(getattr(gpu, "free_bytes", 0))
        row: dict[str, Any] = {
            "device": device,
            "name": getattr(gpu, "name", None),
            "class_key": card_class_key(gpu),
            "free_gib": round(free_now / GIB, 2),
            "reclaimable_gib": 0.0,
            "residents": [i.model_id for i in here],
            "verdict": VERDICT_CANDIDATE,
            "detail": None,
        }
        rows.append(row)
        if device in avoid_set:
            row["verdict"] = VERDICT_AVOIDED
            row["detail"] = "named in avoid"
            continue
        if device in excluded_set:
            row["verdict"] = VERDICT_EXCLUDED
            row["detail"] = "planner.excluded_devices names it"
            continue
        if allowed_set is not None and device not in allowed_set:
            row["verdict"] = VERDICT_NOT_ALLOWED
            row["detail"] = allowed_reason or "the lease's models may not run on it"
            continue
        clash = book.conflicts([device])
        if clash:
            row["verdict"] = VERDICT_LEASED
            row["detail"] = "; ".join(f"lease {lease.id} ({lease.holder})" for lease in clash)
            row["leases"] = [lease.id for lease in clash]
            continue
        busy = [i for i in here if i.state == "loading" or i.active_requests > 0]
        if busy:
            row["verdict"] = VERDICT_BUSY
            row["detail"] = ", ".join(f"{i.model_id} ({i.active_requests} in flight)" for i in busy)
            continue
        pinned = [i for i in here if i.ttl_s == 0]
        if pinned and not force:
            row["verdict"] = VERDICT_PINNED
            row["detail"] = ", ".join(i.model_id for i in pinned)
            continue
        better = [i for i in here if i.priority < priority]
        if better and not force:
            row["verdict"] = VERDICT_BETTER_TIER
            row["detail"] = ", ".join(f"{i.model_id} (priority {i.priority})" for i in better)
            continue
        reclaimable = 0
        for instance in here:
            per_gpu = getattr(instance.plan, "per_gpu_bytes", None) or {}
            reclaimable += int(per_gpu.get(device) or per_gpu.get(str(device)) or 0)
        row["reclaimable_gib"] = round(reclaimable / GIB, 2)
        projected = free_now + reclaimable
        row["projected_free_gib"] = round(projected / GIB, 2)
        if min_free_gib is not None and projected < float(min_free_gib) * GIB:
            row["verdict"] = VERDICT_TOO_LITTLE_FREE
            row["detail"] = (
                f"{projected / GIB:.1f} GiB would be free, {float(min_free_gib):.1f} asked"
            )
            continue
        row["evictions"] = len(here)

    def rank(row: dict[str, Any]) -> tuple[int, int, float, int]:
        return (
            int(row.get("evictions", 0)),
            prefer_rank.get(int(row["device"]), len(prefer_rank)),
            -float(row.get("projected_free_gib", row["free_gib"])),
            int(row["device"]),
        )

    want = max(1, int(count))
    candidates = sorted((row for row in rows if row["verdict"] == VERDICT_CANDIDATE), key=rank)
    rejected = [row for row in rows if row["verdict"] != VERDICT_CANDIDATE]
    picked: list[dict[str, Any]] = []
    if want == 1:
        picked = candidates[:1]
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            groups.setdefault(str(row["class_key"]), []).append(row)
        options = [group[:want] for group in groups.values() if len(group) >= want]
        if options:
            # Summed per-card keys: the pair that costs the fewest evictions,
            # follows the caller's preference, frees the most and sits lowest.
            picked = min(
                options,
                key=lambda group: (
                    sum(rank(row)[0] for row in group),
                    sum(rank(row)[1] for row in group),
                    sum(rank(row)[2] for row in group),
                    min(int(row["device"]) for row in group),
                ),
            )
    chosen: list[int] | None = None
    if picked:
        for row in picked:
            row["verdict"] = VERDICT_CHOSEN
        chosen = sorted(int(row["device"]) for row in picked)
    ordered = picked + [row for row in candidates if row not in picked]
    return AutoPlacement(chosen, ordered + rejected, count=want)
