"""The tenant registry (D77): who shares this rig's cards, and whether each is up.

StudioForge has been the arbiter of the GPUs since D43 -- leases, D56's vacate
protocol, D61's persistence -- but every program that leased a card here was
anonymous to it beyond a holder string. The registry gives each one a row:
declared by the operator in ``tenants.registry`` (config), or announced by the
program itself through ``POST /api/tenants/register`` at boot, or both merged
by ``name``. Each row carries the tenant's family (the ``holder_family`` its
leases use), its kind, its version, its capabilities, a health URL this server
polls, and the leases it currently holds.

What the registry is NOT (D79): a supervisor. StudioForge never starts or
stops a tenant process. A ``module`` -- an optional add-on such as a speech
server -- may be ASKED to warm through its own ``warm_url``, and that is the
whole of the control this server has over it. Each tenant keeps its own repo,
port, launcher and update path; the registry only knows where to look.

Every URL stored here is one this server will later call, so every one of them
passes :mod:`studioforge.netguard` (loopback, the tailnet range, or an
operator-listed host; names never resolved). The poller never follows a
redirect, never honours a proxy environment, and gives each probe two seconds.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from studioforge.config import TENANT_NAME_PATTERN, TenantEntry, TenantsConfig
from studioforge.core.leases import LeaseBook, holder_family, lease_view
from studioforge.errors import BadRequestError, TenantNotFoundError, TenantUnreachableError
from studioforge.logging import get_logger
from studioforge.netguard import TenantUrlError, url_target, validate_tenant_url
from studioforge.types import GpuLease

log = get_logger(__name__)

#: The protocol this registry and ``docs/TENANTS.md`` describe. A tenant may
#: send ``protocol`` in its registration; a version this server does not speak
#: is refused rather than guessed at.
PROTOCOL_VERSION = "vt.1"
SUPPORTED_PROTOCOLS: frozenset[str] = frozenset({PROTOCOL_VERSION})
#: HTTP timeout for the one warm POST. Long: a warm is a model load.
WARM_TIMEOUT_S = 30.0
#: Bounds on what a tenant may register: a flood of capabilities or a novel is
#: not a registration.
MAX_CAPABILITIES = 32
VERSION_MAX_CHARS = 64
NOTES_MAX_CHARS = 400
_CAPABILITY_RE = re.compile(r"^[a-z0-9][a-z0-9._:/-]{0,63}$")
_NAME_RE = re.compile(TENANT_NAME_PATTERN)
_VERSION_RE = re.compile(r"^[\x21-\x7e]+$")
#: Keys lifted from a JSON health body into ``health.summary``. Whatever else a
#: tenant returns stays with the tenant: the summary is a glance, not a mirror.
_SUMMARY_KEYS = ("status", "state", "ok", "ready", "version", "loaded", "can_serve", "device")
HEALTH_STATES = ("unpolled", "healthy", "unhealthy", "unreachable")


@dataclass
class TenantHealth:
    """The last health probe of one tenant."""

    #: ``unpolled`` (no URL, or not asked yet), ``healthy`` (any 2xx),
    #: ``unhealthy`` (answered, not 2xx -- a redirect counts), ``unreachable``.
    state: str = "unpolled"
    checked_at: float | None = None
    status_code: int | None = None
    latency_ms: float | None = None
    #: The exception's CLASS NAME, never its text (D6: a message can quote a URL).
    error: str | None = None
    summary: dict[str, Any] | None = None

    def view(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "checked_at": self.checked_at,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "summary": self.summary,
        }


@dataclass
class TenantRecord:
    """One tenant as the registry knows it: the operator's half and the program's."""

    name: str
    family: str
    kind: str = "tenant"
    #: ``config`` (declared in ``tenants.registry``) or ``registered`` (announced
    #: itself). A config row that also registered stays ``config``: the
    #: operator's declaration is the one that survives a restart.
    source: str = "registered"
    health_url: str | None = None
    status_url: str | None = None
    warm_url: str | None = None
    vacate_url: str | None = None
    default_priority: int = 3
    notes: str = ""
    version: str | None = None
    capabilities: list[str] = field(default_factory=list)
    #: Whether the program itself has registered this process lifetime.
    registered: bool = False
    registered_at: float | None = None
    deregistered_at: float | None = None
    #: The later of the last registration and the last healthy probe.
    last_seen: float | None = None
    #: The address the registration came from. Stored like a lease's
    #: ``holder_peer`` (D55) and shown to nobody.
    peer: str | None = None
    #: The scoped key that registered (``key:<id>``), when one did: the only
    #: non-admin key that may refresh or deregister this row (2.0.1). Kept in
    #: memory like the registration itself, which a restart clears.
    owner: str | None = None
    health: TenantHealth = field(default_factory=TenantHealth)

    @classmethod
    def from_entry(cls, entry: TenantEntry) -> TenantRecord:
        return cls(
            name=entry.name,
            family=entry.effective_family,
            kind=entry.kind,
            source="config",
            health_url=entry.health_url,
            status_url=entry.status_url,
            warm_url=entry.warm_url,
            default_priority=entry.default_priority,
            notes=entry.notes,
        )


def _target(url: str | None) -> str | None:
    if not url:
        return None
    host, port = url_target(url)
    return f"{host}:{port}"


def tenant_view(
    record: TenantRecord,
    *,
    leases: Iterable[GpuLease] = (),
    reveal_urls: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    """The API/MCP projection of a tenant.

    URLs travel only to a caller the D32 gate trusts with the box
    (``reveal_urls``); everyone else gets ``*_registered`` booleans and the
    ``host:port`` of the health endpoint -- the same rule ``lease_view`` applies
    to ``vacate_url``. ``peer`` never travels. ``leases`` is this tenant's slice
    of the book: every lease whose ``holder_family`` is the tenant's family.
    """
    stamp = time.time() if now is None else now
    rows = [lease_view(lease, now=stamp) for lease in leases]
    devices = sorted({int(d) for lease in leases for d in lease.devices})
    data: dict[str, Any] = {
        "name": record.name,
        "family": record.family,
        "kind": record.kind,
        "source": record.source,
        "protocol": PROTOCOL_VERSION,
        "registered": record.registered,
        "version": record.version,
        "capabilities": list(record.capabilities),
        "default_priority": record.default_priority,
        "notes": record.notes,
        "health": record.health.view(),
        "health_target": _target(record.health_url),
        "health_registered": bool(record.health_url),
        "status_registered": bool(record.status_url),
        "warm_registered": bool(record.warm_url),
        "vacate_registered": bool(record.vacate_url),
        "registered_at": record.registered_at,
        "deregistered_at": record.deregistered_at,
        "last_seen": record.last_seen,
        "leases": rows,
        "devices": devices,
        "lease_count": len(rows),
    }
    if reveal_urls:
        data["health_url"] = record.health_url
        data["status_url"] = record.status_url
        data["warm_url"] = record.warm_url
        data["vacate_url"] = record.vacate_url
    return data


def _summary_of(body: Any) -> dict[str, Any] | None:
    if not isinstance(body, Mapping):
        return None
    out: dict[str, Any] = {}
    for key in _SUMMARY_KEYS:
        if key in body:
            value = body[key]
            if isinstance(value, bool | int | float | str) or value is None:
                out[key] = value if not isinstance(value, str) else value[:80]
    return out or None


class TenantRegistry:
    """Every known tenant, keyed by name. Lives on the event loop.

    Built from :class:`TenantsConfig` at boot; :meth:`register` and
    :meth:`deregister` are the runtime half. :meth:`start` runs the health
    poller; :meth:`poll_once` is the same sweep for a test or a one-off.
    """

    def __init__(
        self,
        config: TenantsConfig | None = None,
        *,
        leases: LeaseBook | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config or TenantsConfig()
        self._leases = leases
        self._transport = transport
        self._records: dict[str, TenantRecord] = {}
        self._task: asyncio.Task[None] | None = None
        self._client: httpx.AsyncClient | None = None
        for entry in self._config.registry:
            self._records[entry.name] = TenantRecord.from_entry(entry)

    # -- reading --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[TenantRecord]:
        return sorted(self._records.values(), key=lambda record: record.name)

    def get(self, name: str) -> TenantRecord | None:
        return self._records.get(name)

    def require(self, name: str) -> TenantRecord:
        record = self._records.get(name)
        if record is None:
            raise TenantNotFoundError(
                f"no tenant {name!r}; GET /api/tenants lists the known ones", param="name"
            )
        return record

    def leases_of(self, record: TenantRecord) -> list[GpuLease]:
        if self._leases is None:
            return []
        return [
            lease for lease in self._leases.all() if holder_family(lease.holder) == record.family
        ]

    def view(
        self, record: TenantRecord, *, reveal_urls: bool = False, now: float | None = None
    ) -> dict[str, Any]:
        return tenant_view(record, leases=self.leases_of(record), reveal_urls=reveal_urls, now=now)

    def views(self, *, reveal_urls: bool = False, now: float | None = None) -> list[dict[str, Any]]:
        stamp = time.time() if now is None else now
        return [self.view(record, reveal_urls=reveal_urls, now=stamp) for record in self.all()]

    # -- the runtime half -------------------------------------------------------

    def _url(self, value: Any, *, field: str) -> str | None:
        if value is None:
            return None
        try:
            return validate_tenant_url(value, field=field, allowed_hosts=self._config.allowed_hosts)
        except TenantUrlError as exc:
            raise BadRequestError(str(exc), code="tenant_url_refused", param=field) from exc

    def register(
        self,
        payload: Mapping[str, Any],
        *,
        peer: str | None = None,
        owner: str | None = None,
        now: float | None = None,
    ) -> TenantRecord:
        """Enter or refresh a tenant from its own announcement.

        A name the config declares keeps the operator's ``kind``, ``family``,
        ``notes`` and ``default_priority`` (the program may not promote itself
        to a module); its URLs are taken from the registration when sent, else
        kept. A new name gets whatever the registration says, validated.
        Re-registering is the normal boot path and simply refreshes the row.
        """
        stamp = time.time() if now is None else now
        name = str(payload.get("name") or "").strip().lower()
        if not _NAME_RE.fullmatch(name):
            raise BadRequestError(
                "name must be lower-case: a letter or digit, then up to 63 of "
                "letters, digits, '.', '-' or '_'",
                param="name",
            )
        protocol = payload.get("protocol")
        if protocol is not None and str(protocol) not in SUPPORTED_PROTOCOLS:
            raise BadRequestError(
                f"protocol {protocol!r} is not one this server speaks "
                f"({', '.join(sorted(SUPPORTED_PROTOCOLS))})",
                code="tenant_protocol_unsupported",
                param="protocol",
            )
        family_raw = payload.get("family")
        family = str(family_raw).strip().lower() if family_raw else name
        if not _NAME_RE.fullmatch(family):
            raise BadRequestError("family must be spelled like a name", param="family")
        kind = str(payload.get("kind") or "tenant")
        if kind not in ("module", "tenant"):
            raise BadRequestError("kind must be 'module' or 'tenant'", param="kind")
        version = payload.get("version")
        if version is not None:
            version = str(version).strip()
            if not version or len(version) > VERSION_MAX_CHARS or not _VERSION_RE.match(version):
                raise BadRequestError(
                    f"version must be printable ASCII without spaces, at most "
                    f"{VERSION_MAX_CHARS} characters",
                    param="version",
                )
        priority = payload.get("default_priority", 3)
        if isinstance(priority, bool) or priority not in (1, 2, 3):
            raise BadRequestError(
                "default_priority must be 1 (active chat), 2 (dispatched agent) or 3 (background)",
                param="default_priority",
            )
        caps_raw = payload.get("capabilities") or []
        if not isinstance(caps_raw, list) or len(caps_raw) > MAX_CAPABILITIES:
            raise BadRequestError(
                f"capabilities must be a list of at most {MAX_CAPABILITIES} short tags",
                param="capabilities",
            )
        capabilities: list[str] = []
        for cap in caps_raw:
            text = str(cap).strip().lower()
            if not _CAPABILITY_RE.fullmatch(text):
                raise BadRequestError(
                    f"capability {text!r} is not a short lower-case tag", param="capabilities"
                )
            if text not in capabilities:
                capabilities.append(text)
        notes = str(payload.get("notes") or "")[:NOTES_MAX_CHARS]
        health_url = self._url(payload.get("health_url"), field="health_url")
        status_url = self._url(payload.get("status_url"), field="status_url")
        warm_url = self._url(payload.get("warm_url"), field="warm_url")
        vacate_url = self._url(payload.get("vacate_url"), field="vacate_url")
        if warm_url and kind != "module":
            raise BadRequestError("warm_url is only meaningful for kind 'module'", param="warm_url")

        record = self._records.get(name)
        if record is None:
            record = TenantRecord(name=name, family=family, kind=kind, source="registered")
            self._records[name] = record
            record.default_priority = int(priority)
            record.notes = notes
            record.health_url = health_url
            record.status_url = status_url
            record.warm_url = warm_url
        elif record.source == "config":
            # The operator's declaration wins on identity; the program may fill
            # in or refresh where it can be reached.
            if health_url:
                record.health_url = health_url
            if status_url:
                record.status_url = status_url
            if warm_url and record.kind == "module":
                record.warm_url = warm_url
        else:
            record.family = family
            record.kind = kind
            record.default_priority = int(priority)
            record.notes = notes or record.notes
            record.health_url = health_url
            record.status_url = status_url
            record.warm_url = warm_url
        record.version = version
        record.capabilities = capabilities
        record.vacate_url = vacate_url
        record.registered = True
        record.registered_at = stamp
        record.deregistered_at = None
        record.last_seen = stamp
        record.peer = (peer or "").strip() or None
        record.owner = (owner or "").strip() or None
        if record.health_url is None:
            record.health = TenantHealth()
        log.info(
            "tenant registered",
            tenant=record.name,
            family=record.family,
            kind=record.kind,
            source=record.source,
            version=record.version,
            health_target=_target(record.health_url),
            vacate_registered=bool(record.vacate_url),
            capabilities=len(record.capabilities),
        )
        return record

    def deregister(self, name: str, *, now: float | None = None) -> TenantRecord:
        """The program says goodbye. A config row stays, marked unregistered."""
        stamp = time.time() if now is None else now
        record = self.require(name)
        if record.source == "config":
            record.registered = False
            record.deregistered_at = stamp
            record.version = None
            record.capabilities = []
            record.vacate_url = None
            record.peer = None
            record.owner = None
        else:
            del self._records[name]
            record.registered = False
            record.deregistered_at = stamp
        log.info("tenant deregistered", tenant=record.name, source=record.source)
        return record

    # -- health -----------------------------------------------------------------

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict[str, Any] = {
                "timeout": float(self._config.health_timeout_s),
                # A tenant URL is a tenant's own listener: a 3xx is "it moved"
                # and is reported as such, never chased; and no proxy
                # environment may redirect a loopback probe anywhere.
                "follow_redirects": False,
                "trust_env": False,
                "headers": {"X-SF-Client": "studioforge-tenants"},
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def _probe(self, record: TenantRecord, *, now: float | None = None) -> TenantHealth:
        url = record.health_url
        if not url:
            return record.health
        before = time.perf_counter()
        stamp = time.time() if now is None else now
        health = TenantHealth(checked_at=stamp)
        try:
            response = await self._http().get(url, timeout=float(self._config.health_timeout_s))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a probe never raises past here
            health.state = "unreachable"
            health.error = type(exc).__name__
        else:
            health.latency_ms = round((time.perf_counter() - before) * 1000.0, 1)
            health.status_code = int(response.status_code)
            health.state = "healthy" if response.is_success else "unhealthy"
            body: Any = None
            with contextlib.suppress(ValueError):
                body = response.json()
            health.summary = _summary_of(body)
            if health.state == "healthy":
                record.last_seen = max(record.last_seen or 0.0, stamp)
        previous = record.health.state
        record.health = health
        if previous != health.state and not (previous == "unpolled" and health.state == "healthy"):
            log.info(
                "tenant health changed",
                tenant=record.name,
                was=previous,
                now=health.state,
                status=health.status_code,
                error=health.error,
                target=_target(url),
            )
        return health

    async def poll_once(self, *, now: float | None = None) -> list[TenantRecord]:
        """Probe every tenant with a health URL, concurrently. Never raises."""
        targets = [record for record in self.all() if record.health_url]
        if not targets:
            return []
        await asyncio.gather(*(self._probe(record, now=now) for record in targets))
        return targets

    async def _loop(self) -> None:
        interval = float(self._config.health_interval_s)
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the poller must never die
                log.exception("tenant health sweep failed", error=repr(exc))
            await asyncio.sleep(interval)

    def start(self) -> None:
        """Run the health poller until :meth:`stop`. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._loop(), name="studioforge-tenant-health")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.aclose()

    # -- the one thing this server may ask of a module --------------------------

    async def warm(
        self, name: str, *, requested_by: str = "studioforge", timeout_s: float = WARM_TIMEOUT_S
    ) -> dict[str, Any]:
        """Relay one POST to a module's ``warm_url``; never start anything (D79).

        The module decides what warming means (load its models, open its
        session) and answers; this returns its status and a bounded summary of
        its JSON body. A non-2xx or a transport failure is
        :class:`TenantUnreachableError` naming the class of failure.
        """
        record = self.require(name)
        if record.kind != "module" or not record.warm_url:
            raise BadRequestError(
                f"tenant {name!r} cannot be warmed from here: "
                + ("it is not a module" if record.kind != "module" else "it has no warm_url"),
                code="tenant_not_warmable",
                param="name",
            )
        body = {"requested_by": requested_by, "protocol": PROTOCOL_VERSION}
        target = _target(record.warm_url)
        try:
            response = await self._http().post(record.warm_url, json=body, timeout=timeout_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - reported, never raised raw
            log.warning(
                "tenant warm not delivered", tenant=name, target=target, error=type(exc).__name__
            )
            raise TenantUnreachableError(
                f"tenant {name!r} did not answer its warm_url ({type(exc).__name__})",
                details={"tenant": name, "error": type(exc).__name__},
            ) from exc
        status = int(response.status_code)
        parsed: Any = None
        with contextlib.suppress(ValueError):
            parsed = response.json()
        summary = _summary_of(parsed)
        if not response.is_success:
            log.warning("tenant warm refused", tenant=name, target=target, status=status)
            raise TenantUnreachableError(
                f"tenant {name!r} answered {status} to the warm request",
                details={"tenant": name, "status_code": status, "summary": summary},
            )
        log.info("tenant warm requested", tenant=name, target=target, status=status)
        return {"tenant": name, "ok": True, "status_code": status, "summary": summary}
