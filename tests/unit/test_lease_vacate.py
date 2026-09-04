"""Vacate cards (D56): a better class may ASK a tenant to leave, never make it.

Pinned here, layer by layer: the model never leaks its token; the book keeps
the default byte-identical and phrases both 409s with a wait; the manager
sends exactly one vacate per lease per window and answers ``lease_vacating``
until the holder releases or the window lapses; ``force`` still overrides no
standing lease; the REST/MCP surfaces carry the new fields and the header; and
this server's own benchmarks lease at class 2.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from studioforge.core import leases as leases_mod
from studioforge.core import manager as manager_mod
from studioforge.core.benchmark import Benchmarker, benchmark_lease_priority
from studioforge.core.leases import (
    VACATE_LEASE_HEADER,
    VACATE_TOKEN_HEADER,
    LeaseBook,
    holder_family,
    lease_kind,
    lease_state,
    lease_view,
    send_vacate_request,
    vacate_url_targets_self,
    validate_vacate_token,
    validate_vacate_url,
)
from studioforge.core.priority import PRIORITY_AGENT, PRIORITY_BACKGROUND, PRIORITY_CHAT
from studioforge.errors import BadRequestError, LeaseConflictError
from studioforge.logging import _redact
from studioforge.mcp.management import build_management_mcp
from studioforge.types import GpuLease, ServerStatus
from tests.unit.test_benchmark import (
    FakeEngine,
    StubManager,
    StubProbe,
    content_chunk,
    reference_rig,
    sse,
    timings_chunk,
)
from tests.unit.test_benchmark import (
    engine as engine,  # noqa: PLC0414 - fixture re-export
)
from tests.unit.test_benchmark import (
    make_record as bench_record,
)
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export
from tests.unit.test_gateway_lifecycle import make_manager, make_record, placed
from tests.unit.test_mcp import TINY, call
from tests.unit.test_mcp import state as state  # noqa: PLC0414 - fixture re-export

TOKEN = "vt_9f3c1b7e2a5d4c6b8e1f0a2b3c4d5e6f"  # scrub-ok: test fixture bearer
HOLDER_URL = "http://127.0.0.1:8700/ui/api/vacate"


class Recorder:
    """A vacate sender that records instead of POSTing."""

    def __init__(self, *, delivered: bool = True) -> None:
        self.calls: list[tuple[GpuLease, dict[str, Any], float]] = []
        self.delivered = delivered

    async def __call__(
        self, lease: GpuLease, body: dict[str, Any], *, timeout_s: float
    ) -> tuple[bool, str]:
        self.calls.append((lease, dict(body), timeout_s))
        return self.delivered, "200" if self.delivered else "ConnectError"


def _rig() -> tuple[Any, Any, Recorder]:
    manager, supervisor = make_manager([make_record("some/model")])
    recorder = Recorder()
    manager._vacate_sender = recorder
    return manager, supervisor, recorder


async def _settle() -> None:
    """Let a spawned vacate task run to completion."""
    for _ in range(3):
        await asyncio.sleep(0)


class LogRecorder:
    """Every event the manager logs, with its kwargs, whatever structlog is configured to.

    ``structlog.testing.capture_logs`` sees nothing once an earlier test has
    configured logging with a cached logger, so the sweep records at the
    module's own ``log`` instead.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def _record(self, event: str, **kwargs: Any) -> None:
        self.events.append((event, kwargs))

    debug = info = warning = error = exception = _record

    def dump(self) -> str:
        return json.dumps(self.events, default=str)


# ---------------------------------------------------------------------------
# The model and the book: defaults, the token, the two 409s
# ---------------------------------------------------------------------------


def test_priority_default_is_byte_identical_to_a_pre_d56_lease() -> None:
    book = LeaseBook()
    lease = book.acquire([0], holder="api")
    assert lease.priority == PRIORITY_BACKGROUND
    assert lease.vacate_url is None and lease.vacate_token is None
    assert lease.vacate_deadline is None and not lease.vacating()

    view = lease_view(lease)
    assert view["priority"] == 3
    assert view["vacate_registered"] is False
    assert view["state"] == "active"
    assert view["vacate_deadline"] is None and view["vacate_requested_by"] is None
    assert "vacate_url" not in view and "vacate_token" not in view

    with pytest.raises(BadRequestError) as excinfo:
        book.acquire([1], holder="api", priority=0)
    assert excinfo.value.param == "priority"


def test_the_token_is_never_serialised_or_shown() -> None:
    book = LeaseBook()
    lease = book.acquire([0], holder="clawforge2", vacate_url=HOLDER_URL, vacate_token=TOKEN)
    assert lease.vacate_token == TOKEN, "stored -- it has to be sent back to the holder"

    assert TOKEN not in json.dumps(lease.model_dump(mode="json"))
    assert TOKEN not in lease.model_dump_json()
    assert TOKEN not in json.dumps(lease_view(lease))
    assert TOKEN not in json.dumps(lease_view(lease, reveal_vacate_url=True))
    assert TOKEN not in repr(lease) and TOKEN not in str(lease)
    status = ServerStatus(version="t", uptime_s=1.0, gpus=[], leases=[lease])
    assert TOKEN not in status.model_dump_json()
    # D6: the key is redacted wherever it turns up in a log event.
    event = {"vacate_token": TOKEN, "nested": {"X-SF-Vacate-Token": TOKEN}}
    scrubbed = _redact(None, "info", event)
    assert TOKEN not in json.dumps(scrubbed)


def test_the_url_is_shown_only_on_request() -> None:
    book = LeaseBook()
    lease = book.acquire([0], holder="clawforge2", vacate_url=HOLDER_URL)
    plain = lease_view(lease)
    assert plain["vacate_registered"] is True and "vacate_url" not in plain
    assert lease_view(lease, reveal_vacate_url=True)["vacate_url"] == HOLDER_URL


def test_lease_conflict_now_carries_a_top_level_retry_after() -> None:
    """Q3 #11: the 409 knew the wait and said it only per row."""
    book = LeaseBook()
    now = time.time()
    book.acquire([0], holder="crucibleforge", idle_ttl_s=7200.0, now=now)
    book.acquire([1], holder="api", idle_ttl_s=None, now=now)
    with pytest.raises(LeaseConflictError) as excinfo:
        book.acquire([0, 1], holder="api")
    details = excinfo.value.details
    assert excinfo.value.code == "lease_conflict"
    assert details["retry_after_s"] == leases_mod.LEASE_OPEN_ENDED_RETRY_S, (
        "the shortest of the rows' own advice; an open-ended row counts the re-ask interval"
    )
    assert "vacate" not in details, "nobody was ever asked"


def test_holder_family_splits_on_a_colon_too() -> None:
    """SF-2: this server's own parallel benchmark read as ``other``."""
    assert holder_family("benchmark:parallel") == "benchmark"
    assert lease_kind("benchmark:parallel") == "benchmark"
    assert holder_family("crucibleforge-judge") == "crucibleforge"
    assert holder_family("benchmark") == "benchmark"
    assert holder_family("clawforge2") == "clawforge2"


def test_vacating_outranks_every_other_state() -> None:
    book = LeaseBook()
    start = 1_000_000.0
    lease = book.acquire([0], holder="clawforge2", idle_ttl_s=60.0, now=start)
    assert lease_state(lease, start + 59) == "expiring"
    book.mark_vacating(lease.id, requested_by="benchmark", deadline_s=180.0, now=start + 59)
    assert lease_state(lease, start + 59) == "vacating"
    assert lease_state(lease, start + 59 + 181) != "vacating", "the window lapsed"


# ---------------------------------------------------------------------------
# URL and token validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "ftp://holder/vacate",
        "holder/vacate",
        "http:///vacate",
        "http://holder:99999/vacate",
        "http://user:pw@holder/vacate",
        "http://holder/va cate",
        "http://" + "h" * 600,
    ],
)
def test_vacate_url_shape_is_validated(bad: str) -> None:
    with pytest.raises(BadRequestError) as excinfo:
        validate_vacate_url(bad)
    assert excinfo.value.param == "vacate_url"


def test_vacate_token_is_validated_and_needs_a_url() -> None:
    assert validate_vacate_token(None, has_url=False) is None
    assert validate_vacate_token(TOKEN, has_url=True) == TOKEN
    for bad, has_url in ((TOKEN, False), ("", True), ("has space", True), ("té", True)):
        with pytest.raises(BadRequestError) as excinfo:
            validate_vacate_token(bad, has_url=has_url)
        assert excinfo.value.param == "vacate_token"


async def test_a_vacate_url_onto_our_own_listener_is_refused() -> None:
    own = {1234, 1235}
    for url in (
        "http://127.0.0.1:1234/api/restart/server",
        "http://127.1:1234/x",
        "http://localhost:1235/x",
        "http://LOCALHOST.:1234/x",
        "http://[::1]:1234/x",
        "http://[::ffff:127.0.0.1]:1234/x",
        "http://0.0.0.0:1234/x",
        "http://192.168.1.20:1234/x",
    ):
        assert await vacate_url_targets_self(url, own_ports=own, own_addresses=["192.168.1.20"]), (
            url
        )
    # Another port on this box is the expected deployment (ClawForge2 on :8700).
    assert not await vacate_url_targets_self(HOLDER_URL, own_ports=own)
    # A stranger's literal on our port is not us.
    assert not await vacate_url_targets_self("http://192.168.1.21:1234/x", own_ports=own)


async def test_a_name_that_resolves_to_us_is_refused_and_an_unresolvable_one_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()

    async def to_loopback(host: str, port: int, **_: Any) -> list[Any]:
        return [(2, 1, 6, "", ("127.0.0.1", port))]

    monkeypatch.setattr(loop, "getaddrinfo", to_loopback)
    assert await vacate_url_targets_self("http://rig.example:1234/x", own_ports={1234})

    async def nowhere(host: str, port: int, **_: Any) -> list[Any]:
        raise OSError("no such host")

    monkeypatch.setattr(loop, "getaddrinfo", nowhere)
    assert not await vacate_url_targets_self("http://rig.example:1234/x", own_ports={1234})


async def test_the_manager_refuses_a_self_target_with_a_400_naming_the_param() -> None:
    manager, _supervisor, recorder = _rig()
    with pytest.raises(BadRequestError) as excinfo:
        await manager.acquire_lease(
            [0], holder="x", vacate_url="http://127.0.0.1:1234/api/restart/server"
        )
    assert excinfo.value.param == "vacate_url" and "own listener" in excinfo.value.message
    assert len(manager.leases) == 0 and recorder.calls == []


# ---------------------------------------------------------------------------
# The grant: ask, dedupe, release, lapse
# ---------------------------------------------------------------------------


async def test_a_better_class_sends_one_vacate_and_gets_lease_vacating() -> None:
    manager, _supervisor, recorder = _rig()
    tenant = await manager.acquire_lease(
        [2], holder="clawforge2", vacate_url=HOLDER_URL, vacate_token=TOKEN
    )
    assert tenant.priority == 3

    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([0, 1, 2, 3], holder="crucibleforge", priority=PRIORITY_AGENT)
    err = excinfo.value
    assert err.status_code == 409 and err.code == "lease_vacating"
    assert err.details["retry_after_s"] == 15
    assert err.details["vacate"]["state"] == "vacating"
    assert err.details["vacate"]["leases"] == [tenant.id]
    assert err.details["vacate"]["requested"] == [tenant.id]
    assert err.details["vacate"]["requester"] == "crucibleforge"
    assert err.details["leases"][0]["state"] == "vacating"
    assert err.details["leases"][0]["vacate_requested_by"] == "crucibleforge"
    assert TOKEN not in json.dumps(err.to_payload())
    assert tenant.vacating() and tenant.vacate_deadline is not None
    assert tenant.vacate_deadline - time.time() == pytest.approx(180.0, abs=2.0)
    assert len(manager.leases) == 1, "the tenant's lease stands; nothing was taken"

    await _settle()
    assert len(recorder.calls) == 1
    lease, body, timeout_s = recorder.calls[0]
    assert lease is tenant and timeout_s == 10.0
    assert body == {
        "lease_id": tenant.id,
        "devices": [2],
        "requester": "crucibleforge",
        "requester_priority": 2,
        "deadline_s": 180,
        "deadline_at": tenant.vacate_deadline,
    }

    # The requester re-asks inside the window: same answer, no second POST --
    # from the same asker or a different better-class one.
    for asker in ("crucibleforge", "crucibleforge-judge"):
        with pytest.raises(LeaseConflictError) as again:
            await manager.acquire_lease([2], holder=asker, priority=PRIORITY_AGENT)
        assert again.value.code == "lease_vacating"
        assert again.value.details["vacate"]["requested"] == []
    await _settle()
    assert len(recorder.calls) == 1

    # The holder releases: the next ask is granted and evicts nothing new.
    manager.release_lease(tenant.id)
    granted = await manager.acquire_lease([0, 1, 2, 3], holder="crucibleforge", priority=2)
    assert granted.devices == [0, 1, 2, 3] and granted.priority == 2
    assert [lease.id for lease in manager.leases.all()] == [granted.id]


async def test_the_sweep_releasing_the_tenant_also_ends_the_vacate() -> None:
    manager, _supervisor, recorder = _rig()
    tenant = await manager.acquire_lease(
        [2], holder="clawforge2", vacate_url=HOLDER_URL, idle_ttl_s=60.0
    )
    with pytest.raises(LeaseConflictError):
        await manager.acquire_lease([2], holder="benchmark", priority=PRIORITY_AGENT)
    tenant.last_activity_at = time.time() - 120
    manager._expire_leases()
    assert manager.leases.get(tenant.id) is None
    granted = await manager.acquire_lease([2], holder="benchmark", priority=PRIORITY_AGENT)
    assert granted.devices == [2]
    await _settle()
    assert len(recorder.calls) == 1


async def test_a_lapsed_window_degrades_to_todays_conflict_then_may_be_asked_again() -> None:
    manager, _supervisor, recorder = _rig()
    tenant = await manager.acquire_lease([2], holder="clawforge2", vacate_url=HOLDER_URL)
    with pytest.raises(LeaseConflictError):
        await manager.acquire_lease([2], holder="crucibleforge", priority=PRIORITY_AGENT)
    await _settle()
    assert len(recorder.calls) == 1

    # The deadline passes and the holder never released.
    now = time.time()
    tenant.vacate_requested_at = now - 181
    tenant.vacate_deadline = now - 1
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2], holder="crucibleforge", priority=PRIORITY_AGENT)
    err = excinfo.value
    assert err.code == "lease_conflict", "today's answer, to the byte of its code"
    assert err.details["retry_after_s"] >= 1
    assert err.details["vacate"]["state"] == "timed_out"
    assert err.details["vacate"]["leases"] == [tenant.id]
    assert err.details["vacate"]["reask_at"] == pytest.approx(now - 1 + 180, abs=2.0)
    assert err.details["leases"][0]["state"] != "vacating"
    assert manager.leases.get(tenant.id) is tenant, "the lease stays"
    await _settle()
    assert len(recorder.calls) == 1, "not nagged inside the quiet period"

    # A quiet period as long as the window it ignored, then one more ask.
    tenant.vacate_requested_at = now - 400
    tenant.vacate_deadline = now - 220
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2], holder="crucibleforge", priority=PRIORITY_AGENT)
    assert excinfo.value.code == "lease_vacating"
    await _settle()
    assert len(recorder.calls) == 2


async def test_an_equal_or_worse_class_gets_the_plain_conflict_and_no_post() -> None:
    manager, _supervisor, recorder = _rig()
    tenant = await manager.acquire_lease([2], holder="clawforge2", vacate_url=HOLDER_URL)

    for priority in (None, PRIORITY_BACKGROUND):
        with pytest.raises(LeaseConflictError) as excinfo:
            await manager.acquire_lease([2], holder="api", priority=priority)
        assert excinfo.value.code == "lease_conflict"
        assert "vacate" not in excinfo.value.details
        assert excinfo.value.details["retry_after_s"] >= 1

    # A class-2 tenant is not asked by a class-3 asker, and not by a class-2 one.
    manager.release_lease(tenant.id)
    await manager.acquire_lease(
        [2], holder="agent-x", vacate_url=HOLDER_URL, priority=PRIORITY_AGENT
    )
    for priority in (PRIORITY_BACKGROUND, PRIORITY_AGENT):
        with pytest.raises(LeaseConflictError) as excinfo:
            await manager.acquire_lease([2], holder="api", priority=priority)
        assert excinfo.value.code == "lease_conflict"
    # ...but the chat class may.
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2], holder="chat", priority=PRIORITY_CHAT)
    assert excinfo.value.code == "lease_vacating"
    await _settle()
    assert len(recorder.calls) == 1


async def test_a_holder_without_a_url_is_never_asked_even_by_a_better_class() -> None:
    """A CrucibleForge run or this server's own benchmark cannot be pre-empted."""
    manager, _supervisor, recorder = _rig()
    await manager.acquire_lease([0, 1], holder="benchmark:parallel", priority=PRIORITY_AGENT)
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([0], holder="chat", priority=PRIORITY_CHAT)
    assert excinfo.value.code == "lease_conflict"
    assert "vacate" not in excinfo.value.details
    await _settle()
    assert recorder.calls == []


async def test_a_mixed_clash_asks_nobody() -> None:
    """One lease in the way that cannot be asked makes the whole ask pointless."""
    manager, _supervisor, recorder = _rig()
    await manager.acquire_lease([2], holder="clawforge2", vacate_url=HOLDER_URL)
    await manager.acquire_lease([3], holder="crucibleforge", priority=PRIORITY_AGENT)
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2, 3], holder="chat", priority=PRIORITY_CHAT)
    assert excinfo.value.code == "lease_conflict"
    await _settle()
    assert recorder.calls == []


async def test_force_never_overrides_a_standing_lease_vacating_or_not() -> None:
    manager, _supervisor, recorder = _rig()
    tenant = await manager.acquire_lease([2], holder="clawforge2", vacate_url=HOLDER_URL)
    with pytest.raises(LeaseConflictError) as first:
        await manager.acquire_lease([2], holder="api", force=True)
    assert first.value.code == "lease_conflict"
    with pytest.raises(LeaseConflictError) as second:
        await manager.acquire_lease([2], holder="crucibleforge", priority=2, force=True)
    assert second.value.code == "lease_vacating"
    assert manager.leases.get(tenant.id) is tenant
    await _settle()
    assert len(recorder.calls) == 1


async def test_a_dead_holder_costs_the_window_and_nothing_else(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager, _supervisor, _ = _rig()
    recorder = Recorder(delivered=False)
    manager._vacate_sender = recorder
    logged = LogRecorder()
    monkeypatch.setattr(manager_mod, "log", logged)
    tenant = await manager.acquire_lease([2], holder="clawforge2", vacate_url=HOLDER_URL)
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([2], holder="crucibleforge", priority=2)
    await _settle()
    assert excinfo.value.code == "lease_vacating"
    assert tenant.vacating(), "the window runs regardless; it degrades at the deadline"
    assert any("vacate not delivered" in event for event, _ in logged.events)
    assert len(recorder.calls) == 1


async def test_the_token_never_reaches_a_log_line(monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _supervisor, recorder = _rig()
    logged = LogRecorder()
    monkeypatch.setattr(manager_mod, "log", logged)
    tenant = await manager.acquire_lease(
        [2],
        holder="clawforge2",
        vacate_url=HOLDER_URL + "?secret=no",
        vacate_token=TOKEN,
    )
    with pytest.raises(LeaseConflictError):
        await manager.acquire_lease([2], holder="crucibleforge", priority=2)
    await _settle()
    manager.release_lease(tenant.id)
    text = logged.dump()
    assert any(event == "gpu lease acquired" for event, _ in logged.events)
    assert TOKEN not in text
    assert "?secret=no" not in text, "the URL's path and query are the holder's business"
    assert "127.0.0.1:8700" in text, "the host:port is logged, so the operator can see who"
    assert len(recorder.calls) == 1


async def test_a_class_2_lease_evicts_an_idle_equal_tier_resident_but_not_a_better_one() -> None:
    """D46 on the lease plane: a claim may displace only equal-or-worse tiers.

    The default class 3 keeps the pre-D56 rule exactly: any tier-1/2 idle
    resident refuses. A class-2 claim (this server's benchmarks) treats an
    idle tier-2 resident as displaceable and an idle tier-1 one as a standing
    claim, exactly as a tier-2 *load* would.
    """
    agent = make_record("agent/model")
    chat = make_record("chat/model")
    manager, supervisor = make_manager([agent, chat])
    supervisor.instances[agent.id] = placed(agent.id, [0])
    supervisor.instances[agent.id].priority = PRIORITY_AGENT

    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([0], holder="api")
    assert excinfo.value.details["priority_models"] == [agent.id]
    assert excinfo.value.details["priority"] == 3

    lease = await manager.acquire_lease([0], holder="benchmark", priority=PRIORITY_AGENT)
    assert agent.id not in supervisor.instances
    manager.release_lease(lease.id)

    supervisor.instances[chat.id] = placed(chat.id, [0])
    supervisor.instances[chat.id].priority = PRIORITY_CHAT
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([0], holder="benchmark", priority=PRIORITY_AGENT)
    assert excinfo.value.details["priority_models"] == [chat.id]
    assert chat.id in supervisor.instances
    forced = await manager.acquire_lease([0], holder="benchmark", priority=2, force=True)
    assert chat.id not in supervisor.instances and forced.priority == 2


# ---------------------------------------------------------------------------
# The one outbound request
# ---------------------------------------------------------------------------


async def test_send_vacate_request_posts_the_body_and_the_token_and_follows_no_redirect() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/moved"):
            return httpx.Response(302, headers={"Location": "http://127.0.0.1:8700/elsewhere"})
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    lease = GpuLease(
        id="abc123",
        devices=[2],
        holder="clawforge2",
        created_at=1.0,
        last_activity_at=1.0,
        vacate_url=HOLDER_URL,
        vacate_token=TOKEN,
    )
    body = {"lease_id": "abc123", "requester": "benchmark", "requester_priority": 2}
    assert await send_vacate_request(lease, body, transport=transport) == (True, "200")
    request = seen[0]
    assert request.method == "POST"
    assert request.headers[VACATE_TOKEN_HEADER] == TOKEN
    assert request.headers[VACATE_LEASE_HEADER] == "abc123"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == body

    moved = lease.model_copy(update={"vacate_url": HOLDER_URL + "/moved"})
    assert await send_vacate_request(moved, body, transport=transport) == (False, "302")
    assert len(seen) == 2, "a 3xx is logged as undelivered, never chased"

    no_token = lease.model_copy(update={"vacate_token": None})
    await send_vacate_request(no_token, body, transport=transport)
    assert VACATE_TOKEN_HEADER not in seen[-1].headers


async def test_send_vacate_request_reports_a_transport_error_by_class_name_only() -> None:
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused, and by the way the header was {TOKEN}")

    lease = GpuLease(
        id="abc123",
        devices=[2],
        holder="clawforge2",
        created_at=1.0,
        last_activity_at=1.0,
        vacate_url=HOLDER_URL,
        vacate_token=TOKEN,
    )
    delivered, status = await send_vacate_request(lease, {}, transport=httpx.MockTransport(explode))
    assert (delivered, status) == (False, "ConnectError")
    assert await send_vacate_request(lease.model_copy(update={"vacate_url": None}), {}) == (
        False,
        "no_url",
    )


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------


def test_post_leases_accepts_the_new_fields_and_both_409s_carry_retry_after(app: Any) -> None:
    recorder = Recorder()
    app.state.manager._vacate_sender = recorder
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        created = http.post(
            "/api/leases",
            json={
                "devices": [0],
                "holder": "clawforge2",
                "vacate_url": HOLDER_URL,
                "vacate_token": TOKEN,
            },
        )
        assert created.status_code == 200, created.text
        lease = created.json()
        assert lease["priority"] == 3 and lease["vacate_registered"] is True
        assert lease["vacate_url"] == HOLDER_URL, "the registrant sees what it stored"
        assert TOKEN not in created.text

        plain = http.post("/api/leases", json={"devices": [0]})
        assert plain.status_code == 409
        assert plain.json()["error"]["code"] == "lease_conflict"
        assert plain.headers["Retry-After"].isdigit(), "Q3 #11"

        asked = http.post("/api/leases", json={"devices": [0], "priority": 2, "holder": "cf"})
        assert asked.status_code == 409, asked.text
        envelope = asked.json()["error"]
        assert envelope["code"] == "lease_vacating"
        assert envelope["studioforge"]["retry_after_s"] == 15
        assert envelope["studioforge"]["vacate"]["state"] == "vacating"
        assert asked.headers["Retry-After"] == "15"
        assert TOKEN not in asked.text

        listed = http.get("/api/leases").json()["leases"][0]
        assert listed["state"] == "vacating" and listed["vacate_url"] == HOLDER_URL
        assert TOKEN not in json.dumps(listed)

        bad = http.post("/api/leases", json={"devices": [1], "priority": 7})
        assert bad.status_code == 400 and bad.json()["error"]["param"] == "priority"
        own = http.post(
            "/api/leases",
            json={"devices": [1], "vacate_url": "http://127.0.0.1:1234/api/restart/server"},
        )
        assert own.status_code == 400 and own.json()["error"]["param"] == "vacate_url"

        released = http.delete(f"/api/leases/{lease['id']}")
        assert released.status_code == 200
        granted = http.post("/api/leases", json={"devices": [0], "priority": 2, "holder": "cf"})
        assert granted.status_code == 200, granted.text
        assert granted.json()["priority"] == 2 and granted.json()["holder"] == "cf"

    assert len(recorder.calls) == 1


def test_a_remote_reader_sees_registered_but_not_the_url(app: Any) -> None:
    app.state.manager.leases.acquire([0], holder="clawforge2", vacate_url=HOLDER_URL)
    with TestClient(app, client=("10.0.0.7", 50000)) as http:
        row = http.get("/api/leases").json()["leases"][0]
        assert row["vacate_registered"] is True
        assert "vacate_url" not in row and "vacate_token" not in row
        status_row = http.get("/api/status").json()["leases"][0]
        assert "vacate_url" not in status_row and "vacate_token" not in status_row


def test_post_leases_holder_defaults_to_x_sf_client(app: Any) -> None:
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        # The fixture's fake rig has one card, so each claim is released first.
        named = http.post("/api/leases", json={"devices": [0]}, headers={"X-SF-Client": "cf2"})
        assert named.status_code == 200, named.text
        assert named.json()["holder"] == "cf2"
        http.delete(f"/api/leases/{named.json()['id']}")
        anonymous = http.post("/api/leases", json={"devices": [0]})
        assert anonymous.json()["holder"] == "api", "byte-identical without the header"
        http.delete(f"/api/leases/{anonymous.json()['id']}")
        explicit = http.post(
            "/api/leases", json={"devices": [0], "holder": "x"}, headers={"X-SF-Client": "cf2"}
        )
        assert explicit.json()["holder"] == "x", "the body wins"


# ---------------------------------------------------------------------------
# MCP
# ---------------------------------------------------------------------------


async def test_reserve_gpus_takes_holder_priority_and_the_vacate_fields(state: Any) -> None:
    server = build_management_mcp(state)
    schema = next(t for t in await server.list_tools() if t.name == "reserve_gpus").input_schema
    for name in ("holder", "priority", "vacate_url", "vacate_token"):
        assert name in schema["properties"], name

    reserved = await call(
        server,
        "reserve_gpus",
        devices=[0],
        model_id=TINY,
        holder="crucibleforge",
        priority=2,
        vacate_url=HOLDER_URL,
        vacate_token=TOKEN,
    )
    assert reserved["ok"] is True, reserved
    lease = reserved["lease"]
    assert lease["holder"] == "crucibleforge" and lease["priority"] == 2
    assert lease["vacate_registered"] is True and lease["vacate_url"] == HOLDER_URL
    assert TOKEN not in json.dumps(reserved)

    status = await call(server, "server_status")
    assert status["leases"][0]["priority"] == 2
    assert "vacate_url" not in status["leases"][0]

    default = await call(server, "reserve_gpus", devices=[1])
    assert default["lease"]["holder"] == "mcp" and default["lease"]["priority"] == 3

    own = await call(server, "reserve_gpus", devices=[1], vacate_url="http://localhost:1234/api/x")
    assert own["ok"] is False and own["error"]["param"] == "vacate_url"


# ---------------------------------------------------------------------------
# This server's own benchmarks
# ---------------------------------------------------------------------------


def test_benchmark_lease_priority_reads_the_config_and_defaults_to_agent() -> None:
    from types import SimpleNamespace

    from studioforge.config import Config

    config = Config(data_dir="/tmp/sf-vacate")
    assert config.benchmark.lease_priority == 2
    assert benchmark_lease_priority(SimpleNamespace(config=config)) == 2
    config.benchmark.lease_priority = 3
    assert benchmark_lease_priority(SimpleNamespace(config=config)) == 3
    assert benchmark_lease_priority(SimpleNamespace()) == PRIORITY_AGENT
    with pytest.raises(ValueError):
        Config(data_dir="/tmp/sf-vacate", benchmark={"lease_priority": 4})


async def test_the_placement_benchmark_leases_at_class_2(engine: FakeEngine) -> None:
    probe = StubProbe(reference_rig())
    record = bench_record(device_override=[3], ctx_size=2048)
    manager = StubManager(record, probe)
    engine.scripts = [sse(content_chunk(), timings_chunk(prompt_tps=2000.0, generation_tps=120.0))]

    await Benchmarker(manager, probe=probe).run(
        record, modes=["rtx-5090-x1"], ctx_size=1024, max_tokens=32
    )

    assert manager.leased and manager.leased[0]["holder"] == "benchmark"
    assert manager.leased[0]["priority"] == 2
