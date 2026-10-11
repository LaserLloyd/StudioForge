"""The D56 vacate callback, as an ASGI app a tenant mounts on its own server.

StudioForge POSTs one request to the ``vacate_url`` a lease registered when a
strictly better class wants the cards: ``{lease_id, devices, requester,
requester_priority, deadline_s, deadline_at}`` with the lease's own token in
``X-SF-Vacate-Token`` and its id in ``X-SF-Lease-Id``. The holder then has
``deadline_s`` (StudioForge's ``leases.vacate_timeout_s``, 180 s by default) to
drain, free and ``DELETE`` its lease; nothing is taken from it.

This route does the part every tenant gets wrong the same way: only ``POST``,
only from loopback (StudioForge and the tenant share the box, and the token
would otherwise be one open port away from anyone), a constant-time token
compare, a bounded body, and an answer that leaks nothing. What to do about
the request -- stop taking new work, finish the piece in flight, free, release
-- is ``on_vacate``'s, and it is awaited (bounded by StudioForge's own
callback timeout, 30 s) so a tenant that frees synchronously still answers
2xx inside the window.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

VACATE_TOKEN_HEADER = "X-SF-Vacate-Token"
VACATE_LEASE_HEADER = "X-SF-Lease-Id"
#: A vacate body is a handful of fields; anything bigger is not one.
MAX_BODY_BYTES = 64 * 1024

TokenMatcher = Callable[[str | None], bool]
VacateHandler = Callable[["VacateRequest"], Awaitable[Any] | Any]
Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
AsgiApp = Callable[[Scope, Receive, Send], Awaitable[None]]


@dataclass(frozen=True)
class VacateRequest:
    """What StudioForge asked, parsed and bounded."""

    lease_id: str
    devices: list[int]
    requester: str
    requester_priority: int | None
    deadline_s: float | None
    deadline_at: float | None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_body(cls, body: dict[str, Any], *, header_lease_id: str | None) -> VacateRequest:
        devices_raw = body.get("devices")
        devices = (
            [int(d) for d in devices_raw if isinstance(d, int) and not isinstance(d, bool)]
            if isinstance(devices_raw, list)
            else []
        )
        priority = body.get("requester_priority")
        deadline_s = body.get("deadline_s")
        deadline_at = body.get("deadline_at")
        return cls(
            lease_id=str(body.get("lease_id") or header_lease_id or ""),
            devices=devices,
            requester=str(body.get("requester") or "")[:120],
            requester_priority=int(priority) if isinstance(priority, int) else None,
            deadline_s=float(deadline_s) if isinstance(deadline_s, int | float) else None,
            deadline_at=float(deadline_at) if isinstance(deadline_at, int | float) else None,
            raw=dict(body),
        )


def constant_time_matcher(token: str) -> TokenMatcher:
    """A :data:`TokenMatcher` over one fixed token, in constant time."""
    expected = token.encode()

    def match(presented: str | None) -> bool:
        return bool(presented) and hmac.compare_digest(str(presented).encode(), expected)

    return match


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    text = str(host).strip().strip("[]")
    if text.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    return (mapped or ip).is_loopback


def vacate_asgi_route(
    token_matcher: TokenMatcher,
    on_vacate: VacateHandler,
    *,
    loopback_only: bool = True,
) -> AsgiApp:
    """An ASGI application implementing the D56 callback.

    Mount it where the lease's ``vacate_url`` points (Starlette/FastAPI:
    ``app.mount("/vacate", vacate_asgi_route(client.token_matches, drain))``).
    Answers: ``405`` for anything but POST, ``403`` for a non-loopback peer
    (with ``loopback_only``), ``401`` for a missing or wrong token, ``400`` for
    a body that is not a JSON object, ``413`` for one over
    :data:`MAX_BODY_BYTES`, ``500`` if ``on_vacate`` raised (the text stays in
    the tenant's log), and ``202 {"accepted": true, "lease_id": ...}`` once
    ``on_vacate`` returned. The token is never echoed.
    """

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            return

        async def reply(status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                        (b"cache-control", b"no-store"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        if str(scope.get("method", "")).upper() != "POST":
            await reply(405, {"error": "method_not_allowed"})
            return
        client = scope.get("client")
        peer = client[0] if isinstance(client, list | tuple) and client else None
        if loopback_only and not _is_loopback(peer):
            await reply(403, {"error": "loopback_only"})
            return
        headers: dict[str, str] = {}
        for name, value in scope.get("headers") or []:
            headers[bytes(name).decode("latin-1").lower()] = bytes(value).decode("latin-1")
        if not token_matcher(headers.get(VACATE_TOKEN_HEADER.lower())):
            await reply(401, {"error": "bad_vacate_token"})
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            chunk = message.get("body") or b""
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                await reply(413, {"error": "body_too_large"})
                return
            chunks.append(chunk)
            if not message.get("more_body"):
                break
        try:
            parsed = json.loads(b"".join(chunks) or b"{}")
        except ValueError:
            await reply(400, {"error": "body_not_json"})
            return
        if not isinstance(parsed, dict):
            await reply(400, {"error": "body_not_an_object"})
            return
        lease_header = headers.get(VACATE_LEASE_HEADER.lower())
        request = VacateRequest.from_body(parsed, header_lease_id=lease_header)
        try:
            result = on_vacate(request)
            if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                await result
        except Exception:  # noqa: BLE001 - the handler's failure is the tenant's to log
            await reply(500, {"error": "vacate_handler_failed", "lease_id": request.lease_id})
            return
        await reply(202, {"accepted": True, "lease_id": request.lease_id})

    return app
