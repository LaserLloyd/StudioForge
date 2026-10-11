"""``devices: "auto"`` on a lease (D78): the card is chosen the way the grant judges one.

Table-driven over fake GPU states and resident tiers: never a busy card, never
a leased one (and never a vacate ask -- that is for a claim on named cards),
never a pinned or better-tier idle resident without ``force``; then fewest
evictions, the caller's ``prefer`` order, most free VRAM, lowest index.
``min_free_gib`` is measured after the evictable idle residents are gone. A
refusal carries ``considered`` under the code a tenant already knows; explicit
devices behave exactly as before, and the steering fields are refused beside
them.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.core.gpu import FakeGpuProbe
from studioforge.core.leases import (
    VERDICT_AVOIDED,
    VERDICT_BETTER_TIER,
    VERDICT_BUSY,
    VERDICT_CANDIDATE,
    VERDICT_CHOSEN,
    VERDICT_LEASED,
    VERDICT_PINNED,
    VERDICT_TOO_LITTLE_FREE,
    LeaseBook,
    choose_auto_devices,
)
from studioforge.core.priority import PRIORITY_AGENT, PRIORITY_BACKGROUND, PRIORITY_CHAT
from studioforge.errors import (
    BadRequestError,
    InsufficientVramError,
    LeaseConflictError,
    ModelBusyError,
    NotSupportedError,
)
from studioforge.types import GpuInfo
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export
from tests.unit.test_gateway_lifecycle import make_manager, make_record, placed

GIB = 1024**3
LOOPBACK = ("127.0.0.1", 50000)


def gpu(index: int, total_gib: float, free_gib: float) -> GpuInfo:
    return GpuInfo(
        index=index,
        name=f"GPU {index}",
        total_bytes=int(total_gib * GIB),
        free_bytes=int(free_gib * GIB),
        used_bytes=int((total_gib - free_gib) * GIB),
    )


def rig(*free: float) -> list[GpuInfo]:
    """Four cards: two 32 GiB, two 24 GiB, with the given free GiB each."""
    totals = (32.0, 32.0, 24.0, 24.0)
    return [gpu(i, totals[i], free[i]) for i in range(len(free))]


def resident(
    model_id: str,
    devices: list[int],
    *,
    priority: int = PRIORITY_BACKGROUND,
    busy: int = 0,
    pinned: bool = False,
    bytes_per_card: float = 8.0,
) -> Any:
    instance = placed(model_id, devices)
    instance.priority = priority
    instance.active_requests = busy
    if pinned:
        instance.ttl_s = 0
    assert instance.plan is not None
    instance.plan.per_gpu_bytes = {d: int(bytes_per_card * GIB) for d in devices}
    return instance


def verdicts(placement: Any) -> dict[int, str]:
    return {int(row["device"]): str(row["verdict"]) for row in placement.considered}


# ---------------------------------------------------------------------------
# The pure decision
# ---------------------------------------------------------------------------


def test_an_empty_rig_picks_the_card_with_the_most_free_vram() -> None:
    placement = choose_auto_devices(rig(30.0, 31.0, 23.0, 23.5), [], LeaseBook())
    assert placement.devices == [1]
    assert verdicts(placement) == {
        1: VERDICT_CHOSEN,
        0: VERDICT_CANDIDATE,
        2: VERDICT_CANDIDATE,
        3: VERDICT_CANDIDATE,
    }
    assert placement.considered[0]["device"] == 1, "the chosen card is first"


def test_an_empty_card_beats_a_fuller_card_with_an_idle_resident() -> None:
    """Evicting an idle resident costs someone a reload later; an empty card costs nobody."""
    gpus = rig(20.0, 10.0, 23.0, 23.0)
    residents = [resident("bg/model", [1], bytes_per_card=20.0)]
    placement = choose_auto_devices(gpus, residents, LeaseBook())
    assert placement.devices == [2], "an empty 3090 beats a 5090 that would need an eviction"
    assert placement.devices != [3], "ties on free VRAM go to the lowest index"
    row = next(r for r in placement.considered if r["device"] == 1)
    assert row["verdict"] == VERDICT_CANDIDATE and row["evictions"] == 1
    assert row["projected_free_gib"] == 30.0, "the idle resident's estimate counts as reclaimable"


def test_prefer_orders_within_the_same_eviction_class_and_avoid_rules_out() -> None:
    placement = choose_auto_devices(rig(31.0, 31.0, 23.0, 23.5), [], LeaseBook(), prefer=[3, 2])
    assert placement.devices == [3], "prefer outranks free VRAM"
    placement = choose_auto_devices(
        rig(31.0, 31.0, 23.0, 23.5), [], LeaseBook(), prefer=[3], avoid=[3]
    )
    assert placement.devices == [0] and verdicts(placement)[3] == VERDICT_AVOIDED


def test_prefer_does_not_outrank_an_empty_card() -> None:
    residents = [resident("bg/model", [3])]
    placement = choose_auto_devices(rig(31.0, 31.0, 23.0, 23.5), residents, LeaseBook(), prefer=[3])
    assert placement.devices == [0], "an eviction is worse than a lost preference"


def test_leased_busy_pinned_and_better_tier_cards_are_out() -> None:
    book = LeaseBook()
    book.acquire([0], holder="clawforge2")
    residents = [
        resident("busy/model", [1], busy=1),
        resident("pinned/model", [2], pinned=True),
        resident("chat/model", [3], priority=PRIORITY_CHAT),
    ]
    placement = choose_auto_devices(
        rig(31.0, 31.0, 23.0, 23.5), residents, book, priority=PRIORITY_AGENT
    )
    assert placement.refused and placement.devices is None
    assert verdicts(placement) == {
        0: VERDICT_LEASED,
        1: VERDICT_BUSY,
        2: VERDICT_PINNED,
        3: VERDICT_BETTER_TIER,
    }
    leased = next(r for r in placement.considered if r["device"] == 0)
    assert leased["leases"] == [book.all()[0].id]


def test_force_admits_pinned_and_better_tier_cards_but_never_busy_or_leased() -> None:
    book = LeaseBook()
    book.acquire([0], holder="clawforge2")
    residents = [
        resident("busy/model", [1], busy=1),
        resident("pinned/model", [2], pinned=True),
        resident("chat/model", [3], priority=PRIORITY_CHAT),
    ]
    placement = choose_auto_devices(
        rig(31.0, 31.0, 23.0, 23.5), residents, book, priority=PRIORITY_AGENT, force=True
    )
    assert placement.devices == [3] or placement.devices == [2]
    assert verdicts(placement)[0] == VERDICT_LEASED and verdicts(placement)[1] == VERDICT_BUSY


def test_the_claims_class_is_what_an_idle_resident_is_measured_against() -> None:
    """D56 item 6: a class-2 claim may displace an idle tier-2 or tier-3 resident, never tier 1."""
    residents = [
        resident("agent/model", [0], priority=PRIORITY_AGENT),
        resident("chat/model", [1], priority=PRIORITY_CHAT),
    ]
    at_two = choose_auto_devices(rig(31.0, 31.0), residents, LeaseBook(), priority=PRIORITY_AGENT)
    assert at_two.devices == [0] and verdicts(at_two)[1] == VERDICT_BETTER_TIER
    at_three = choose_auto_devices(
        rig(31.0, 31.0), residents, LeaseBook(), priority=PRIORITY_BACKGROUND
    )
    assert at_three.refused and set(verdicts(at_three).values()) == {VERDICT_BETTER_TIER}
    at_one = choose_auto_devices(rig(31.0, 31.0), residents, LeaseBook(), priority=PRIORITY_CHAT)
    assert at_one.devices == [0]
    assert verdicts(at_one)[1] == VERDICT_CANDIDATE, (
        "an equal class is displaceable, exactly as the grant on named cards treats it"
    )


def test_min_free_gib_is_measured_after_evictable_residents_are_gone() -> None:
    gpus = rig(4.0, 2.0, 1.0, 1.0)
    residents = [resident("bg/model", [1], bytes_per_card=20.0)]
    placement = choose_auto_devices(gpus, residents, LeaseBook(), min_free_gib=8.0)
    assert placement.devices == [1], "2 GiB free now, 22 GiB once the idle resident is unloaded"
    refused = choose_auto_devices(gpus, [], LeaseBook(), min_free_gib=8.0)
    assert refused.refused and set(verdicts(refused).values()) == {VERDICT_TOO_LITTLE_FREE}


def test_an_owner_resident_counts_for_nothing_against_its_card() -> None:
    residents = [resident("mine/model", [0], busy=3)]
    placement = choose_auto_devices(rig(31.0, 10.0), residents, LeaseBook(), owners=["mine/model"])
    assert placement.devices == [0]


# ---------------------------------------------------------------------------
# The manager: the refusal codes, and the grant on the chosen card
# ---------------------------------------------------------------------------


def _manager(gpus: list[GpuInfo]) -> tuple[Any, Any]:
    manager, supervisor = make_manager([make_record("some/model")])
    from studioforge.core.planner import Planner

    manager.planner = Planner(manager.config, FakeGpuProbe(gpus), leases=manager.leases)
    return manager, supervisor


async def test_auto_grants_on_the_chosen_card_and_evicts_only_there() -> None:
    manager, supervisor = _manager(rig(31.0, 31.0, 23.0, 23.5))
    supervisor.instances["bg/model"] = resident("bg/model", [1])
    lease, placement = await manager.acquire_lease_auto(holder="speechforge", priority=2)
    assert lease.devices == [0] and placement.devices == [0]
    assert "bg/model" in supervisor.instances, "an unchosen card's resident is untouched"
    assert manager.leases.all() == [lease]
    second = await manager.acquire_lease("auto", holder="clawforge2")
    assert second.devices == [3], "the leased card is skipped; the 3090 with more free VRAM wins"


async def test_auto_refusals_carry_the_known_codes_and_the_considered_rows() -> None:
    manager, supervisor = _manager(rig(31.0, 31.0))
    supervisor.instances["busy/model"] = resident("busy/model", [0], busy=1)
    supervisor.instances["bg/model"] = resident("bg/model", [1])
    manager.leases.acquire([1], holder="clawforge2")
    with pytest.raises(ModelBusyError) as busy:
        await manager.acquire_lease("auto", holder="speechforge")
    assert busy.value.details["retry_after_s"] and busy.value.details["placement"] == "auto"
    assert {row["device"] for row in busy.value.details["considered"]} == {0, 1}

    supervisor.instances.pop("busy/model")
    manager.leases.acquire([0], holder="benchmark")
    with pytest.raises(LeaseConflictError) as leased:
        await manager.acquire_lease("auto", holder="speechforge", priority=1)
    assert leased.value.code == "lease_conflict", "auto placement asks nobody to vacate"
    assert "retry_after_s" in leased.value.details and "considered" in leased.value.details
    assert all(lease.vacate_deadline is None for lease in manager.leases.all())


async def test_auto_refusal_for_pins_and_tiers_points_at_force_and_min_free_at_vram() -> None:
    manager, supervisor = _manager(rig(31.0, 31.0))
    supervisor.instances["pinned/model"] = resident("pinned/model", [0], pinned=True)
    supervisor.instances["chat/model"] = resident("chat/model", [1], priority=PRIORITY_CHAT)
    with pytest.raises(LeaseConflictError) as held:
        await manager.acquire_lease("auto", holder="speechforge", priority=2)
    assert held.value.param == "force" and held.value.details["priority"] == 2
    supervisor.instances.clear()
    with pytest.raises(InsufficientVramError) as short:
        await manager.acquire_lease(None, holder="speechforge", min_free_gib=40.0)
    assert short.value.code == "insufficient_vram" and short.value.param == "min_free_gib"
    with pytest.raises(BadRequestError) as avoided:
        await manager.acquire_lease("auto", holder="speechforge", avoid=[0, 1])
    assert avoided.value.param == "avoid"


async def test_auto_validates_its_steering_and_explicit_devices_refuse_it() -> None:
    manager, _ = _manager(rig(31.0, 31.0))
    with pytest.raises(BadRequestError):
        await manager.acquire_lease("sometimes", holder="x")
    with pytest.raises(BadRequestError) as unknown:
        await manager.acquire_lease("auto", holder="x", prefer=[7])
    assert unknown.value.param == "prefer"
    with pytest.raises(BadRequestError):
        await manager.acquire_lease("auto", holder="x", min_free_gib=-1)
    with pytest.raises(BadRequestError) as explicit:
        await manager.acquire_lease([0], holder="x", prefer=[1])
    assert explicit.value.param == "devices"
    lease = await manager.acquire_lease([1], holder="x")
    assert lease.devices == [1], "explicit devices are unchanged"


async def test_auto_without_a_working_probe_says_so_instead_of_guessing() -> None:
    manager, _ = make_manager([make_record("some/model")])

    class Sick:
        def list_gpus(self) -> list[GpuInfo]:
            raise RuntimeError("nvml down")

    from types import SimpleNamespace

    manager.planner = SimpleNamespace(probe=Sick())
    with pytest.raises(BadRequestError) as excinfo:
        await manager.acquire_lease("auto", holder="x")
    assert excinfo.value.code == "placement_unavailable"


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _four_cards(app: Any) -> None:
    """The catalog fixture has one GPU; these routes need a rig to choose from."""
    probe = FakeGpuProbe(rig(31.0, 31.0, 23.0, 23.5))
    app.state.probe = probe
    app.state.planner.probe = probe
    app.state.manager.planner.probe = probe


def test_the_route_takes_auto_and_returns_the_placement(app: Any) -> None:
    _four_cards(app)
    local = TestClient(app, client=LOOPBACK)
    reply = local.post(
        "/api/leases",
        json={"devices": "auto", "holder": "speechforge", "priority": 2, "idle_ttl_s": 600},
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["devices"] == [0] and body["placement"]["mode"] == "auto"
    assert body["placement"]["considered"][0]["verdict"] == VERDICT_CHOSEN
    omitted = local.post("/api/leases", json={"holder": "clawforge2"})
    assert omitted.status_code == 200 and omitted.json()["devices"] == [1]
    explicit = local.post("/api/leases", json={"devices": [1], "holder": "again"})
    assert explicit.status_code == 409 and "placement" not in explicit.json()


def test_the_route_refuses_a_share_lease_and_a_bad_devices_word(app: Any) -> None:
    local = TestClient(app, client=LOOPBACK)
    share = local.post("/api/leases", json={"devices": "auto", "exclusive": False})
    assert share.status_code == 400
    assert share.json()["error"]["code"] == NotSupportedError.code
    assert "share" in share.json()["error"]["message"]
    word = local.post("/api/leases", json={"devices": "any"})
    assert word.status_code == 400 and word.json()["error"]["param"] == "devices"


def test_a_route_refusal_carries_considered_and_retry_after(app: Any) -> None:
    _four_cards(app)
    local = TestClient(app, client=LOOPBACK)
    for device, holder in ((0, "a"), (1, "b"), (2, "c"), (3, "d")):
        reply = local.post("/api/leases", json={"devices": [device], "holder": holder})
        assert reply.status_code == 200, reply.text
    refused = local.post("/api/leases", json={"devices": "auto", "holder": "e"})
    assert refused.status_code == 409
    details = refused.json()["error"]["studioforge"]
    assert {row["verdict"] for row in details["considered"]} == {VERDICT_LEASED}
    assert refused.headers.get("retry-after") is not None
