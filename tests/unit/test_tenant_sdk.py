"""``packages/studioforge-tenant``: the tenant SDK against ``httpx.MockTransport``.

Imported off ``sys.path`` like the companion, never installed: the package is
pure Python over httpx and the point of these tests is the wire shape -- what
a tenant sends, how each server answer becomes an outcome value, that the
vacate token travels only where it must, and that the ASGI vacate route does
the D56 callback's checks in the right order.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SDK_SRC = REPO_ROOT / "packages" / "studioforge-tenant" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from studioforge_tenant import (  # noqa: E402
    VACATE_LEASE_HEADER,
    VACATE_TOKEN_HEADER,
    Granted,
    Refused,
    TenantClient,
    Unreachable,
    Unsupported,
    VacateRequest,
    constant_time_matcher,
    vacate_asgi_route,
)

TOKEN = "vt_3a9c0f1e5d7b2a4c6e8f0a1b2c3d4e5f"  # scrub-ok: test fixture bearer
VACATE_URL = "http://127.0.0.1:1236/v1/speech/vacate"


class Server:
    """A scripted StudioForge: one canned response per (method, path)."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.script: dict[tuple[str, str], Any] = {}

    def transport(self) -> httpx.MockTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            self.calls.append(request)
            answer = self.script.get((request.method, request.url.path))
            if isinstance(answer, Exception):
                raise answer
            if answer is None:
                return httpx.Response(404, json={"error": {"code": "not_found"}})
            return answer

        return httpx.MockTransport(handle)

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.calls[index].content or b"{}")


def client(server: Server, **kwargs: Any) -> TenantClient:
    return TenantClient(
        "http://127.0.0.1:1234/",
        "speechforge",
        2,
        vacate_url=VACATE_URL,
        vacate_token=TOKEN,
        api_key="sf-test-key-123456",
        version="0.9.1",
        health_url="http://127.0.0.1:1236/health",
        capabilities=["stt", "tts"],
        transport=server.transport(),
        **kwargs,
    )


def lease_row(**extra: Any) -> dict[str, Any]:
    return {
        "id": "ab12cd34ef56",
        "devices": [3],
        "holder": "speechforge",
        "holder_family": "speechforge",
        "priority": 2,
        "idle_ttl_s": 1200.0,
        "state": "active",
        "expires_at": 1.7e9,
        "vacate_registered": True,
        "vacate_url": VACATE_URL,
        **extra,
    }


# ---------------------------------------------------------------------------
# acquire
# ---------------------------------------------------------------------------


async def test_acquire_auto_sends_the_vt1_body_and_returns_granted_with_the_placement() -> None:
    server = Server()
    server.script[("POST", "/api/leases")] = httpx.Response(
        200,
        json={
            **lease_row(),
            "placement": {"mode": "auto", "devices": [3], "considered": [{"device": 3}]},
        },
    )
    async with client(server) as tenant:
        outcome = await tenant.acquire(
            "auto", min_free_gib=8.0, prefer=[3, 2], avoid=[0], idle_ttl_s=1200, reason="voice"
        )
    assert isinstance(outcome, Granted)
    assert outcome.lease_id == "ab12cd34ef56" and outcome.devices == [3]
    assert outcome.placement == {"mode": "auto", "devices": [3], "considered": [{"device": 3}]}
    assert "vacate_url" not in outcome.view and "vacate_token" not in outcome.view
    body = server.body()
    assert body == {
        "holder": "speechforge",
        "reason": "voice",
        "idle_ttl_s": 1200,
        "priority": 2,
        "force": False,
        "exclusive": True,
        "devices": "auto",
        "min_free_gib": 8.0,
        "prefer": [3, 2],
        "avoid": [0],
        "vacate_url": VACATE_URL,
        "vacate_token": TOKEN,
    }
    request = server.calls[-1]
    assert request.headers["x-sf-client"] == "speechforge"
    assert request.headers["authorization"].startswith("Bearer ")


async def test_acquire_explicit_devices_never_forces_and_never_sends_steering() -> None:
    server = Server()
    server.script[("POST", "/api/leases")] = httpx.Response(200, json=lease_row())
    async with client(server) as tenant:
        outcome = await tenant.acquire([3], min_free_gib=8.0, prefer=[2])
    assert isinstance(outcome, Granted) and outcome.placement is None
    body = server.body()
    assert body["devices"] == [3] and body["force"] is False
    assert "min_free_gib" not in body and "prefer" not in body


async def test_acquire_turns_each_refusal_into_a_value_with_its_advice() -> None:
    server = Server()
    server.script[("POST", "/api/leases")] = httpx.Response(
        409,
        headers={"Retry-After": "15"},
        json={
            "error": {
                "code": "lease_vacating",
                "message": "asked to leave",
                "studioforge": {
                    "retry_after_s": 15,
                    "considered": [{"device": 3, "verdict": "leased"}],
                },
            }
        },
    )
    async with client(server) as tenant:
        outcome = await tenant.acquire("auto")
    assert isinstance(outcome, Refused)
    assert outcome.code == "lease_vacating" and outcome.retry_after_s == 15.0
    assert outcome.retryable and outcome.considered == [{"device": 3, "verdict": "leased"}]

    server.script[("POST", "/api/leases")] = httpx.Response(
        503, json={"error": {"code": "model_busy", "studioforge": {"retry_after_s": 5}}}
    )
    async with client(server) as tenant:
        busy = await tenant.acquire([3])
    assert isinstance(busy, Refused) and busy.retry_after_s == 5.0 and busy.retryable

    server.script[("POST", "/api/leases")] = httpx.Response(
        409, json={"error": {"code": "lease_conflict", "message": "held"}}
    )
    async with client(server) as tenant:
        conflict = await tenant.acquire([3])
    assert isinstance(conflict, Refused) and not conflict.retryable


async def test_acquire_reports_unreachable_and_unsupported_distinctly() -> None:
    server = Server()
    server.script[("POST", "/api/leases")] = httpx.ConnectError("refused")
    async with client(server) as tenant:
        down = await tenant.acquire("auto")
    assert down == Unreachable("ConnectError")

    server.script[("POST", "/api/leases")] = httpx.Response(404, text="not here")
    async with client(server) as tenant:
        missing = await tenant.acquire([3])
    assert missing == Unreachable("http 404")

    server.script[("POST", "/api/leases")] = httpx.Response(422, json={"detail": "list expected"})
    async with client(server) as tenant:
        old = await tenant.acquire("auto")
    assert isinstance(old, Unsupported) and old.missing == "auto_placement"

    server.script[("POST", "/api/leases")] = httpx.Response(
        200, json={"devices": [3], "holder": "speechforge"}
    )
    async with client(server) as tenant:
        odd = await tenant.acquire([3])
    assert odd == Unreachable("grant without an id")


# ---------------------------------------------------------------------------
# touch / release / release_stale / touch_loop
# ---------------------------------------------------------------------------


async def test_touch_release_and_stale_release_by_family() -> None:
    server = Server()
    server.script[("POST", "/api/leases/ab12cd34ef56/touch")] = httpx.Response(
        200, json=lease_row()
    )
    server.script[("DELETE", "/api/leases/ab12cd34ef56")] = httpx.Response(200, json={})
    server.script[("DELETE", "/api/leases/gone")] = httpx.Response(404, json={})
    server.script[("GET", "/api/leases")] = httpx.Response(
        200,
        json={
            "leases": [
                lease_row(),
                lease_row(id="gone", holder="speechforge-canary", holder_family="speechforge"),
                lease_row(id="theirs", holder="clawforge2", holder_family="clawforge2"),
            ]
        },
    )
    async with client(server) as tenant:
        assert await tenant.touch("ab12cd34ef56") is True
        assert await tenant.touch("nope") is False
        assert await tenant.release("ab12cd34ef56") is True
        assert await tenant.release("gone") is True, "already gone is the outcome we wanted"
        ours = await tenant.ours()
        assert [row["id"] for row in ours] == ["ab12cd34ef56", "gone"]
        assert await tenant.release_stale() == 2
    deletes = [c.url.path for c in server.calls if c.method == "DELETE"]
    assert "/api/leases/theirs" not in deletes, "never another family's lease"


async def test_touch_loop_touches_on_its_cadence_and_ends_when_the_lease_is_gone() -> None:
    server = Server()
    answers = iter(
        [httpx.Response(200, json={}), httpx.Response(200, json={}), httpx.Response(404)]
    )

    def handle(request: httpx.Request) -> httpx.Response:
        server.calls.append(request)
        return next(answers)

    lost: list[bool] = []

    async def on_lost() -> None:
        lost.append(True)

    tenant = TenantClient(
        "http://127.0.0.1:1234", "speechforge", transport=httpx.MockTransport(handle)
    )
    await asyncio.wait_for(
        tenant.touch_loop("ab12cd34ef56", interval_s=0.001, on_lost=on_lost), timeout=5.0
    )
    assert len(server.calls) == 3 and lost == [True]
    await tenant.aclose()


async def test_touch_loop_skips_a_quiet_tick_and_stops_on_the_event() -> None:
    server = Server()
    server.script[("POST", "/api/leases/x/touch")] = httpx.Response(200, json={})
    stop = asyncio.Event()
    active = {"value": False}
    tenant = TenantClient("http://127.0.0.1:1234", "speechforge", transport=server.transport())
    task = asyncio.create_task(
        tenant.touch_loop("x", interval_s=0.001, stop=stop, while_active=lambda: active["value"])
    )
    await asyncio.sleep(0.02)
    assert server.calls == [], "no activity, no touch -- let an idle lease expire"
    active["value"] = True
    await asyncio.sleep(0.02)
    assert len(server.calls) >= 1
    stop.set()
    await asyncio.wait_for(task, timeout=2.0)
    await tenant.aclose()


# ---------------------------------------------------------------------------
# register / deregister / features
# ---------------------------------------------------------------------------


async def test_register_sends_the_vt1_announcement_and_deregister_forgets() -> None:
    server = Server()
    server.script[("POST", "/api/tenants/register")] = httpx.Response(
        200, json={"name": "speechforge", "registered": True}
    )
    server.script[("DELETE", "/api/tenants/speechforge")] = httpx.Response(200, json={})
    async with client(server, kind="module", warm_url="http://127.0.0.1:1236/warm") as tenant:
        result = await tenant.register()
        assert result == {"name": "speechforge", "registered": True}
        assert await tenant.deregister() is True
    body = server.body(0)
    assert body == {
        "name": "speechforge",
        "family": "speechforge",
        "kind": "module",
        "protocol": "vt.1",
        "default_priority": 2,
        "capabilities": ["stt", "tts"],
        "version": "0.9.1",
        "health_url": "http://127.0.0.1:1236/health",
        "warm_url": "http://127.0.0.1:1236/warm",
        "vacate_url": VACATE_URL,
    }
    assert "vacate_token" not in body, "the token rides on the lease, never on the registration"


async def test_register_against_an_older_server_is_unsupported_not_an_error() -> None:
    server = Server()
    async with client(server) as tenant:
        result = await tenant.register()
    assert isinstance(result, Unsupported) and result.missing == "tenant_registry"
    server.script[("POST", "/api/tenants/register")] = httpx.Response(
        400, json={"error": {"code": "tenant_url_refused", "message": "no", "param": "health_url"}}
    )
    async with client(server) as tenant:
        refused = await tenant.register()
    assert isinstance(refused, Refused) and refused.code == "tenant_url_refused"


async def test_features_reads_the_capabilities_block_once() -> None:
    server = Server()
    server.script[("GET", "/api/capabilities")] = httpx.Response(
        200, json={"implemented": {"features": ["lease_auto_placement", "tenant_registry"]}}
    )
    async with client(server) as tenant:
        assert "lease_auto_placement" in await tenant.features()
        assert "tenant_registry" in await tenant.features()
    assert len([c for c in server.calls if c.url.path == "/api/capabilities"]) == 1


def test_the_client_refuses_a_vacate_url_without_a_token_and_hides_the_token() -> None:
    with pytest.raises(ValueError):
        TenantClient("http://127.0.0.1:1234", "x", vacate_url=VACATE_URL)
    tenant = TenantClient("http://127.0.0.1:1234", "x", vacate_url=VACATE_URL, vacate_token=TOKEN)
    assert TOKEN not in repr(tenant)
    assert tenant.token_matches(TOKEN) and not tenant.token_matches("nope")
    assert not tenant.token_matches(None)


# ---------------------------------------------------------------------------
# The vacate route
# ---------------------------------------------------------------------------


async def _call(
    app: Any,
    *,
    method: str = "POST",
    client_host: str | None = "127.0.0.1",
    headers: dict[str, str] | None = None,
    body: bytes = b"",
) -> tuple[int, dict[str, Any]]:
    scope: dict[str, Any] = {
        "type": "http",
        "method": method,
        "path": "/vacate",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": (client_host, 5000) if client_host else None,
    }
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    payload = json.loads(next(m for m in sent if m["type"] == "http.response.body")["body"])
    return int(start["status"]), payload


def _vacate_body() -> bytes:
    return json.dumps(
        {
            "lease_id": "ab12cd34ef56",
            "devices": [3],
            "requester": "openclaw-main",
            "requester_priority": 1,
            "deadline_s": 180,
            "deadline_at": 1.7e9,
        }
    ).encode()


async def test_the_vacate_route_checks_method_peer_token_and_body_in_that_order() -> None:
    seen: list[VacateRequest] = []

    async def on_vacate(request: VacateRequest) -> None:
        seen.append(request)

    app = vacate_asgi_route(constant_time_matcher(TOKEN), on_vacate)
    good = {VACATE_TOKEN_HEADER: TOKEN, VACATE_LEASE_HEADER: "ab12cd34ef56"}
    assert (await _call(app, method="GET", headers=good))[0] == 405
    assert (await _call(app, client_host="192.168.1.50", headers=good))[0] == 403
    assert (await _call(app, headers={}))[0] == 401
    assert (await _call(app, headers={VACATE_TOKEN_HEADER: "wrong"}))[0] == 401
    assert (await _call(app, headers=good, body=b"not json"))[0] == 400
    assert (await _call(app, headers=good, body=b"[1, 2]"))[0] == 400
    assert (await _call(app, headers=good, body=b"x" * (64 * 1024 + 1)))[0] == 413
    assert seen == [], "nothing reached the handler before every check passed"

    status, payload = await _call(app, headers=good, body=_vacate_body())
    assert status == 202 and payload == {"accepted": True, "lease_id": "ab12cd34ef56"}
    (request,) = seen
    assert request.devices == [3] and request.requester == "openclaw-main"
    assert request.requester_priority == 1 and request.deadline_s == 180.0
    assert TOKEN not in json.dumps(payload)


async def test_the_vacate_route_accepts_ipv6_loopback_and_a_sync_handler_and_hides_failures() -> (
    None
):
    calls: list[str] = []

    def on_vacate(request: VacateRequest) -> None:
        calls.append(request.lease_id)
        if request.lease_id == "boom":
            raise RuntimeError("the tenant's own bug, with a path in it")

    app = vacate_asgi_route(constant_time_matcher(TOKEN), on_vacate)
    good = {VACATE_TOKEN_HEADER: TOKEN}
    status, _ = await _call(app, client_host="::1", headers=good, body=_vacate_body())
    assert status == 202
    status, payload = await _call(
        app, headers=good, body=json.dumps({"lease_id": "boom", "devices": [1]}).encode()
    )
    assert status == 500 and payload == {"error": "vacate_handler_failed", "lease_id": "boom"}
    assert "path" not in json.dumps(payload)
    relaxed = vacate_asgi_route(constant_time_matcher(TOKEN), on_vacate, loopback_only=False)
    status, _ = await _call(relaxed, client_host="100.64.0.9", headers=good, body=_vacate_body())
    assert status == 202
    assert calls == ["ab12cd34ef56", "boom", "ab12cd34ef56"]


def test_the_sdk_depends_on_httpx_alone_and_never_imports_studioforge() -> None:
    import tomllib

    meta = tomllib.loads((SDK_SRC.parent / "pyproject.toml").read_text(encoding="utf-8"))
    assert meta["project"]["dependencies"] == ["httpx>=0.27"]
    for path in SDK_SRC.rglob("*.py"):
        assert "import studioforge\n" not in path.read_text(encoding="utf-8")
        assert "from studioforge." not in path.read_text(encoding="utf-8")
