"""The lease and registry client of the VRAM Tenant Protocol (vt.1).

Every call is bounded by ``timeout_s`` and answers with a VALUE, never an
exception, because a tenant's hot path (a render about to start, a voice
session opening) must not die on StudioForge being slow. The four outcomes of
:meth:`TenantClient.acquire`:

* :class:`Granted` -- hold the lease; keep it touched; release it when done.
* :class:`Refused` -- StudioForge said no and said why: ``code`` is its
  machine code (``lease_conflict``, ``lease_vacating``, ``model_busy``,
  ``insufficient_vram``, ...), ``retry_after_s`` its advice, ``considered``
  the per-card verdicts an automatic placement produced.
* :class:`Unreachable` -- no answer, or not a StudioForge. The reference
  behaviour here is SpeechForge S16: load unleased only onto a card that is
  CLEARLY free (NVML shows the room and no ``llama-server`` holds memory
  there), otherwise refuse and cool down. Never treat silence as a grant.
* :class:`Unsupported` -- a StudioForge too old for what was asked (no lease
  API at all, or no ``devices: "auto"``); fall back to explicit devices.

The vacate token travels in the lease body and in the callback header and
nowhere else: not in logs, not in a status page, not in ``repr``.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

log = logging.getLogger("studioforge_tenant")

PROTOCOL_VERSION = "vt.1"
#: The D46 classes. A tenant picks one and says why in its own decision log.
PRIORITY_CHAT = 1
PRIORITY_AGENT = 2
PRIORITY_BACKGROUND = 3
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_IDLE_TTL_S = 3600.0
#: ``touch_loop``'s default cadence; StudioForge coalesces the disk writes.
DEFAULT_TOUCH_INTERVAL_S = 30.0


@dataclass(frozen=True)
class Granted:
    lease_id: str
    devices: list[int]
    view: dict[str, Any] = field(default_factory=dict)
    #: ``{"mode": "auto", "devices": [...], "considered": [...]}`` for an
    #: automatic placement; ``None`` when devices were named.
    placement: dict[str, Any] | None = None


@dataclass(frozen=True)
class Refused:
    status: int
    code: str
    message: str
    retry_after_s: float | None
    #: Per-card verdicts (``device``, ``verdict``, ``detail``, ``free_gib``...)
    #: when the server considered cards for an automatic placement.
    considered: list[dict[str, Any]] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        """Whether waiting ``retry_after_s`` and asking again can change the answer."""
        return self.code in ("lease_vacating", "model_busy", "priority_hold", "benchmark_busy")


@dataclass(frozen=True)
class Unreachable:
    #: An exception CLASS name or ``"http <status>"``; never a message.
    reason: str


@dataclass(frozen=True)
class Unsupported:
    #: What the server lacks: ``"lease_api"`` or ``"auto_placement"``.
    missing: str
    reason: str


Outcome = Granted | Refused | Unreachable | Unsupported


def _json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _error(body: dict[str, Any]) -> dict[str, Any]:
    err = body.get("error")
    return err if isinstance(err, dict) else {}


def _details(body: dict[str, Any]) -> dict[str, Any]:
    details = _error(body).get("studioforge")
    return details if isinstance(details, dict) else {}


def _retry_after(response: httpx.Response, body: dict[str, Any]) -> float | None:
    header = response.headers.get("retry-after")
    if header:
        try:
            return max(1.0, float(header))
        except ValueError:
            pass
    for source in (_details(body), _error(body), body):
        value = source.get("retry_after_s")
        if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
            return float(value)
    return None


def _public(view: dict[str, Any]) -> dict[str, Any]:
    """The lease fields worth keeping in a tenant's own status: never the URL or token."""
    keep = (
        "id",
        "devices",
        "holder",
        "priority",
        "idle_ttl_s",
        "state",
        "expires_at",
        "retry_after_s",
        "vacate_registered",
    )
    return {key: view.get(key) for key in keep if key in view}


class TenantClient:
    """One tenant's handle on a StudioForge: leases, registration, keep-alive.

    Construct once per process. ``holder`` is the name every lease carries
    (and the registry ``name`` by default); make it your ``holder_family``
    (lower-case, no suffix -- ``speechforge``, ``clawforge2``) so
    StudioForge's views group your leases under you. ``priority`` is the D46
    class of your claims. ``vacate_url``/``vacate_token`` register where a
    better class may ask you to leave (D56); mount :func:`vacate_asgi_route`
    there with :meth:`token_matches`.
    """

    def __init__(
        self,
        url: str,
        holder: str,
        priority: int = PRIORITY_BACKGROUND,
        *,
        vacate_url: str | None = None,
        vacate_token: str | None = None,
        api_key: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        name: str | None = None,
        family: str | None = None,
        kind: str = "tenant",
        version: str | None = None,
        health_url: str | None = None,
        status_url: str | None = None,
        warm_url: str | None = None,
        capabilities: Iterable[str] = (),
        reason: str = "",
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if vacate_url and not vacate_token:
            raise ValueError("a vacate_url needs a vacate_token to be trusted at the other end")
        if priority not in (PRIORITY_CHAT, PRIORITY_AGENT, PRIORITY_BACKGROUND):
            raise ValueError("priority must be 1, 2 or 3")
        self.base_url = url.rstrip("/")
        self.holder = holder
        self.priority = int(priority)
        self.vacate_url = vacate_url
        self._vacate_token = vacate_token
        self.timeout_s = float(timeout_s)
        self.name = (name or holder).strip().lower()
        self.family = (family or self.name).strip().lower()
        self.kind = kind
        self.version = version
        self.health_url = health_url
        self.status_url = status_url
        self.warm_url = warm_url
        self.capabilities = [str(c) for c in capabilities]
        self.reason = reason
        headers = {"X-SF-Client": holder}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        kwargs: dict[str, Any] = {
            "base_url": self.base_url,
            "timeout": self.timeout_s,
            "headers": headers,
            # The lease body carries our vacate token: never via an env proxy,
            # never to wherever a redirect points.
            "trust_env": False,
            "follow_redirects": False,
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._http = httpx.AsyncClient(**kwargs)
        self._features: set[str] | None = None

    def __repr__(self) -> str:
        return (
            f"TenantClient(url={self.base_url!r}, holder={self.holder!r}, priority={self.priority})"
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> TenantClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # -- the vacate token ------------------------------------------------------

    def token_matches(self, presented: str | None) -> bool:
        """Constant-time comparison of a presented ``X-SF-Vacate-Token``."""
        if not presented or not self._vacate_token:
            return False
        return hmac.compare_digest(str(presented).encode(), self._vacate_token.encode())

    # -- feature gating -----------------------------------------------------------

    async def features(self, *, refresh: bool = False) -> set[str]:
        """The ``implemented.features`` of ``GET /api/capabilities`` (D64), cached.

        Empty when the server is unreachable or predates D64.
        """
        if self._features is not None and not refresh:
            return set(self._features)
        try:
            response = await self._http.get("/api/capabilities")
        except httpx.HTTPError:
            return set()
        if not response.is_success:
            return set()
        implemented = _json(response).get("implemented")
        names = implemented.get("features") if isinstance(implemented, dict) else None
        self._features = {str(n) for n in names} if isinstance(names, list) else set()
        return set(self._features)

    # -- leases -----------------------------------------------------------------------

    async def acquire(
        self,
        devices: str | Sequence[int] = "auto",
        *,
        min_free_gib: float | None = None,
        prefer: Sequence[int] | None = None,
        avoid: Sequence[int] | None = None,
        idle_ttl_s: float | None = DEFAULT_IDLE_TTL_S,
        reason: str | None = None,
        model_ids: Sequence[str] = (),
        priority: int | None = None,
    ) -> Outcome:
        """Take an exclusive lease on ``devices`` (``"auto"`` lets StudioForge pick).

        Never ``force``: a pinned or deliberately placed model is the
        operator's decision (ClawForge2's rule, SpeechForge S3).
        """
        auto = isinstance(devices, str)
        body: dict[str, Any] = {
            "holder": self.holder,
            "reason": reason if reason is not None else self.reason,
            "idle_ttl_s": idle_ttl_s,
            "priority": self.priority if priority is None else int(priority),
            "force": False,
            "exclusive": True,
        }
        if auto:
            body["devices"] = "auto"
            if min_free_gib is not None:
                body["min_free_gib"] = float(min_free_gib)
            if prefer:
                body["prefer"] = [int(d) for d in prefer]
            if avoid:
                body["avoid"] = [int(d) for d in avoid]
        else:
            body["devices"] = [int(d) for d in devices]
        if model_ids:
            body["model_ids"] = list(model_ids)
        if self.vacate_url:
            body["vacate_url"] = self.vacate_url
            body["vacate_token"] = self._vacate_token
        try:
            response = await self._http.post("/api/leases", json=body)
        except httpx.HTTPError as exc:
            return Unreachable(type(exc).__name__)
        data = _json(response)
        if response.is_success:
            lease_id = str(data.get("id") or "")
            granted = data.get("devices")
            if not lease_id or not isinstance(granted, list):
                return Unreachable("grant without an id")
            placement = data.get("placement") if isinstance(data.get("placement"), dict) else None
            return Granted(lease_id, [int(d) for d in granted], _public(data), placement)
        if response.status_code in (404, 405):
            # No lease API (pre-D43), or not a StudioForge at all.
            return Unreachable(f"http {response.status_code}")
        if response.status_code == 422 and auto:
            # A pre-D78 server validates ``devices`` as a list of ints.
            return Unsupported("auto_placement", 'this StudioForge has no devices "auto"')
        err = _error(data)
        details = _details(data)
        considered = details.get("considered")
        return Refused(
            status=int(response.status_code),
            code=str(err.get("code") or f"http_{response.status_code}"),
            message=str(err.get("message") or "")[:400],
            retry_after_s=_retry_after(response, data),
            considered=list(considered) if isinstance(considered, list) else [],
            details=details,
        )

    async def touch(self, lease_id: str) -> bool | None:
        """``True`` kept alive, ``False`` the lease is gone (404), ``None`` unknown."""
        try:
            response = await self._http.post(f"/api/leases/{lease_id}/touch")
        except httpx.HTTPError:
            return None
        if response.status_code == 404:
            return False
        return True if response.is_success else None

    async def release(self, lease_id: str) -> bool:
        """Give the card back. A 404 is success: the lease is gone, which is the point."""
        try:
            response = await self._http.delete(f"/api/leases/{lease_id}")
        except httpx.HTTPError as exc:
            log.warning(
                "lease release failed (%s); StudioForge's idle TTL will reclaim it",
                type(exc).__name__,
            )
            return False
        if response.status_code == 404:
            return True
        if not response.is_success:
            log.warning(
                "lease release answered %s; StudioForge's idle TTL will reclaim it",
                response.status_code,
            )
        return response.is_success

    async def ours(self) -> list[dict[str, Any]]:
        """Every standing lease whose ``holder_family`` is ours (``[]`` when unreachable)."""
        try:
            response = await self._http.get("/api/leases")
        except httpx.HTTPError:
            return []
        if not response.is_success:
            return []
        rows = _json(response).get("leases") or []
        out: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            family = row.get("holder_family") or str(row.get("holder") or "").split("-")[0]
            if str(family).lower() == self.family or row.get("holder") == self.holder:
                out.append(row)
        return out

    async def release_stale(self) -> int:
        """At boot this process holds nothing, so every lease under our family is stale.

        D61 persists leases across a StudioForge restart and a crash of ours
        leaves one standing for up to its idle TTL; releasing them first is
        what keeps a card from being held by a ghost.
        """
        released = 0
        for row in await self.ours():
            lease_id = str(row.get("id") or "")
            if lease_id and await self.release(lease_id):
                released += 1
                log.info("released stale lease %s on CUDA %s", lease_id, row.get("devices"))
        return released

    async def touch_loop(
        self,
        lease_id: str,
        *,
        interval_s: float = DEFAULT_TOUCH_INTERVAL_S,
        stop: asyncio.Event | None = None,
        while_active: Callable[[], bool] | None = None,
        on_lost: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        """Keep a lease alive until ``stop`` is set or the lease is gone.

        ``while_active`` (optional) says whether the tenant did work since the
        last tick: a lease idle in both places should be allowed to expire, so
        a tick with no activity sends nothing. ``on_lost`` is called once when
        a touch answers 404 -- the lease was released or swept under us -- and
        the loop ends. Transport errors are not fatal: the next tick retries.
        """
        while True:
            if stop is not None and stop.is_set():
                return
            if while_active is None or while_active():
                alive = await self.touch(lease_id)
                if alive is False:
                    if on_lost is not None:
                        result = on_lost()
                        if asyncio.iscoroutine(result):
                            await result
                    return
            if stop is None:
                await asyncio.sleep(interval_s)
                continue
            try:
                await asyncio.wait_for(stop.wait(), interval_s)
                return
            except TimeoutError:
                continue

    # -- the registry (D77) -------------------------------------------------------

    async def register(self) -> dict[str, Any] | Unreachable | Unsupported | Refused:
        """Announce this tenant: ``POST /api/tenants/register``. Idempotent; send at every boot."""
        body: dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "kind": self.kind,
            "protocol": PROTOCOL_VERSION,
            "default_priority": self.priority,
            "capabilities": self.capabilities,
        }
        for key in ("version", "health_url", "status_url", "warm_url", "vacate_url"):
            value = getattr(self, key)
            if value:
                body[key] = value
        try:
            response = await self._http.post("/api/tenants/register", json=body)
        except httpx.HTTPError as exc:
            return Unreachable(type(exc).__name__)
        data = _json(response)
        if response.is_success:
            return data
        if response.status_code in (404, 405):
            return Unsupported("tenant_registry", "this StudioForge has no tenant registry (D77)")
        err = _error(data)
        return Refused(
            status=int(response.status_code),
            code=str(err.get("code") or f"http_{response.status_code}"),
            message=str(err.get("message") or "")[:400],
            retry_after_s=_retry_after(response, data),
            details=_details(data),
        )

    async def deregister(self) -> bool:
        """``DELETE /api/tenants/<name>``. A 404 is success: nothing to forget."""
        try:
            response = await self._http.delete(f"/api/tenants/{self.name}")
        except httpx.HTTPError:
            return False
        return response.is_success or response.status_code == 404
