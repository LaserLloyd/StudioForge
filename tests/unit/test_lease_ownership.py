"""Lease ownership and the load path (D87, work package WP2).

The defects this pins, from the 2026-11-09 multi-instance review (A-1 ... A-22):

* A-1  a lease naming a base evicted the base's own resident copy (``base@b``);
* A-2  ``load_recommended`` treated the caller's own lease as a block for others
       only, so the holder's model landed on the headline mode;
* A-9  an outsider load's eviction ladder could stop a lease's model, and the
       cumulative ladder stopped victims that freed nothing on the target cards;
* A-10 any load in flight refused every lease grant, wherever it was headed;
* A-11 a grant was documented as loading the model -- it reserves;
* A-14 ``devices: "auto"`` ignored ``planner.excluded_devices`` and the owners'
       saved placement, and could only ever pick one card;
* A-15 the grant evicted silently (only ``GET /api/evictions`` knew);
* A-21 MCP ``reserve_gpus`` lacked auto/steering/``model_ids`` and owned nothing;
* A-22 a release or expiry left the lease's models resident (lead decision F-2).

Plus B.6 (``lease_id`` on the load routes), the Kubernetes Lease field names
(B.0) with every legacy key kept, and the machine-readable refusal details the
B.0 fragile-item table asks for -- each beside the exact prose clients parse.
No GPU, no engine, no port: fake probes, stub supervisors, in-process MCP and
``TestClient``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.core.gpu import FakeGpuProbe
from studioforge.core.leases import (
    LEASE_DEPRECATED_FIELDS,
    LEASE_FIELD_ALIASES,
    VERDICT_CHOSEN,
    VERDICT_EXCLUDED,
    VERDICT_NOT_ALLOWED,
    LeaseBook,
    card_class_key,
    choose_auto_devices,
    holder_family,
    lease_kind,
    lease_view,
)
from studioforge.core.planner import Planner
from studioforge.core.priority import PRIORITY_AGENT, PRIORITY_CHAT
from studioforge.errors import (
    BadRequestError,
    InsufficientVramError,
    LeaseConflictError,
    LeaseNotFoundError,
    ModelBusyError,
)
from studioforge.mcp.management import build_management_mcp
from studioforge.types import GB, GpuInfo, InstanceInfo, LoadPlan, LoadRejected, VramEstimate
from tests.unit.test_catalog_routes import MODEL_ID as ROUTE_MODEL
from tests.unit.test_catalog_routes import app as app  # noqa: PLC0414 - fixture re-export
from tests.unit.test_gateway_lifecycle import make_manager, make_record, placed
from tests.unit.test_load_recommended import MODEL
from tests.unit.test_load_recommended import make_manager as walk_manager
from tests.unit.test_mcp import TINY, call
from tests.unit.test_mcp import state as state  # noqa: PLC0414 - fixture re-export
from tests.unit.test_planner import make_config

GIB = 1024**3
LOOPBACK = ("127.0.0.1", 50000)


def copy_record(base_id: str, tag: str = "b", **settings: Any) -> Any:
    """A virtual ``<base>@<tag>`` with launch settings of its own (its own server)."""
    return make_record(f"{base_id}@{tag}", **settings).model_copy(
        update={"is_virtual": True, "base_model_id": base_id}
    )


def resident(
    model_id: str,
    devices: list[int],
    *,
    gib: float = 8.0,
    busy: int = 0,
    pinned: bool = False,
    priority: int = 3,
    idle_s: float = 400.0,
) -> InstanceInfo:
    instance = placed(model_id, devices, idle_s=idle_s)
    instance.active_requests = busy
    instance.priority = priority
    if pinned:
        instance.ttl_s = 0
    assert instance.plan is not None
    instance.plan.per_gpu_bytes = {d: int(gib * GIB) for d in devices}
    return instance


def four_cards(*free: float) -> list[GpuInfo]:
    """Two 32 GiB cc 12.0 cards and two 24 GiB cc 8.6 cards: two classes, two of each."""
    spec = ((32.0, (12, 0)), (32.0, (12, 0)), (24.0, (8, 6)), (24.0, (8, 6)))
    out = []
    for index, gib_free in enumerate(free):
        total, cc = spec[index]
        out.append(
            GpuInfo(
                index=index,
                name=f"GPU {index}",
                total_bytes=int(total * GIB),
                free_bytes=int(gib_free * GIB),
                used_bytes=int((total - gib_free) * GIB),
                compute_capability=cc,
            )
        )
    return out


def with_probe(manager: Any, gpus: list[GpuInfo]) -> Any:
    manager.planner = Planner(manager.config, FakeGpuProbe(gpus), leases=manager.leases)
    return manager


# ---------------------------------------------------------------------------
# A-1: a lease for the base keeps the base's resident copy
# ---------------------------------------------------------------------------


async def test_a_lease_naming_the_base_keeps_its_resident_copy_and_adopts_it() -> None:
    """The reproduction seed (test_fable_lease.py D5), inverted: the copy stays."""
    base = make_record("pub/base")
    copy = copy_record("pub/base", device_override=[2, 3])
    stranger = make_record("other/model")
    manager, supervisor = make_manager([base, copy, stranger])
    supervisor.instances[copy.id] = resident(copy.id, [2, 3])
    supervisor.instances[stranger.id] = resident(stranger.id, [3])

    grant = await manager.grant_lease([2, 3], holder="judge-b", model_ids=[base.id], reason="x")

    assert copy.id in supervisor.instances, "the holder's own copy was evicted (A-1)"
    assert grant.adopted == [copy.id]
    assert grant.lease.model_ids == [base.id, copy.id]
    assert stranger.id not in supervisor.instances
    assert [row["model_id"] for row in grant.evicted] == [stranger.id]
    assert manager.leases.blocked_for(copy.id) == frozenset(), "the copy may reload here"


async def test_a_copy_another_lease_names_is_spared_but_not_adopted() -> None:
    base = make_record("pub/base")
    copy = copy_record("pub/base", device_override=[1, 2])
    manager, supervisor = make_manager([base, copy])
    supervisor.instances[copy.id] = resident(copy.id, [1, 2])
    theirs = manager.leases.acquire([1], holder="judge-a", model_ids=[copy.id])

    grant = await manager.grant_lease([2, 3], holder="judge-b", model_ids=[base.id])

    assert copy.id in supervisor.instances
    assert grant.adopted == [] and grant.lease.model_ids == [base.id]
    assert manager.leases.for_model(copy.id) is theirs, "a model belongs to at most one lease"


async def test_a_lease_naming_a_copy_does_not_spare_the_base_or_a_sibling() -> None:
    """Only copies OF a named base are the holder's: the copy needs the cards."""
    base = make_record("pub/base")
    copy_b = copy_record("pub/base", "b", device_override=[2])
    copy_c = copy_record("pub/base", "c", device_override=[3])
    manager, supervisor = make_manager([base, copy_b, copy_c])
    supervisor.instances[base.id] = resident(base.id, [2])
    supervisor.instances[copy_c.id] = resident(copy_c.id, [3])

    grant = await manager.grant_lease([2, 3], holder="judge", model_ids=[copy_b.id])

    assert {row["model_id"] for row in grant.evicted} == {base.id, copy_c.id}
    assert grant.adopted == []


# ---------------------------------------------------------------------------
# A-2: the holder's load_recommended lands on the lease's cards
# ---------------------------------------------------------------------------


def _leased_walk(devices: list[int], **kwargs: Any) -> tuple[Any, Any, Any]:
    manager, supervisor = walk_manager(**kwargs)
    book = LeaseBook()
    manager.leases = book
    manager.planner.leases = book
    lease = book.acquire(devices, holder="bench", model_ids=[MODEL], reason="bench")
    return manager, supervisor, lease


async def test_the_holders_load_recommended_lands_on_its_leased_cards() -> None:
    """The reproduction seed (test_fable_lease.py D4), inverted."""
    manager, supervisor, lease = _leased_walk([2, 3])
    instance = await manager.load_recommended(MODEL, 32768, priority=1, max_slots=1)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3], "the holder's model went elsewhere (A-2)"
    notes = " ".join(instance.plan.notes)
    assert f"lease {lease.id}" in notes and "saves settings.device_override" not in notes
    assert supervisor.starts == 1


async def test_the_dry_run_agrees_and_names_the_lease() -> None:
    manager, _supervisor, lease = _leased_walk([2, 3])
    plan = await manager.plan_recommended(MODEL, 32768, priority=1, max_slots=1)
    assert plan["fits"] is True, plan
    assert sorted(plan["devices"]) == [2, 3]
    assert plan["lease_id"] == lease.id and plan["lease_devices"] == [2, 3]
    assert plan["device_override"] is None, "a lease is not a saved device_override"


async def test_a_lease_bound_walk_refuses_a_prefer_mode_off_its_cards() -> None:
    manager, _supervisor, lease = _leased_walk([2, 3])
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 32768, prefer_modes=["dual_5090"])
    assert excinfo.value.param == "prefer_modes"
    assert lease.id in excinfo.value.message
    with pytest.raises(BadRequestError) as bound:
        await manager.load_recommended(MODEL, 32768, allowed_devices=[0, 1])
    assert bound.value.param == "allowed_devices" and bound.value.details["lease_id"] == lease.id


async def test_a_lease_bound_refusal_is_told_as_the_leases_not_as_an_override() -> None:
    manager, _supervisor, lease = _leased_walk([3], n_ctx_train=262144)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144, kv_min="f16", min_slots=4)
    message = excinfo.value.message
    assert f"the cards lease {lease.id} (bench) holds for it" in message
    assert "device_override" not in excinfo.value.details
    assert excinfo.value.details["lease_devices"] == [3]


# ---------------------------------------------------------------------------
# B.6: lease_id on the load paths
# ---------------------------------------------------------------------------


async def test_load_with_lease_id_binds_the_model_and_lands_on_the_lease() -> None:
    manager, supervisor = walk_manager()
    lease = manager.leases.acquire([2, 3], holder="bench")
    manager.planner.leases = manager.leases
    instance = await manager.load(MODEL, ctx_size=8192, lease_id=lease.id, source="x")
    assert instance.plan is not None and sorted(instance.plan.devices) == [2, 3]
    assert lease.model_ids == [MODEL]
    with pytest.raises(BadRequestError) as outside:
        await manager.load(MODEL, devices=[0], lease_id=lease.id)
    assert outside.value.param == "devices"


async def test_load_recommended_with_lease_id_binds_and_a_failure_unbinds() -> None:
    manager, _supervisor = walk_manager()
    manager.planner.leases = manager.leases
    lease = manager.leases.acquire([3], holder="bench")
    with pytest.raises(InsufficientVramError):
        await manager.load_recommended(MODEL, 131072, kv_min="f16", min_slots=8, lease_id=lease.id)
    assert lease.model_ids == [], "a failed bound load takes the binding back out"
    instance = await manager.load_recommended(MODEL, 8192, max_slots=1, lease_id=lease.id)
    assert instance.plan is not None and instance.plan.devices == [3]
    assert lease.model_ids == [MODEL]


async def test_lease_id_refuses_an_unknown_lease_and_a_model_another_lease_names() -> None:
    manager, _supervisor = walk_manager()
    manager.planner.leases = manager.leases
    with pytest.raises(LeaseNotFoundError):
        await manager.load(MODEL, lease_id="nope")
    manager.leases.acquire([0], holder="a", model_ids=[MODEL])
    other = manager.leases.acquire([2], holder="b")
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.load_recommended(MODEL, 8192, lease_id=other.id)
    assert excinfo.value.code == "lease_conflict"
    assert excinfo.value.details["code"] == "lease_conflict"


async def test_a_lease_id_dry_run_plans_on_the_lease_and_binds_nothing() -> None:
    manager, _supervisor = walk_manager()
    manager.planner.leases = manager.leases
    lease = manager.leases.acquire([2, 3], holder="bench")
    plan = await manager.plan_recommended(MODEL, 32768, max_slots=1, lease_id=lease.id)
    assert plan["fits"] is True and sorted(plan["devices"]) == [2, 3]
    assert lease.model_ids == [], "a dry run binds nothing"


def test_the_route_binds_only_the_holders_lease(app: Any, monkeypatch: Any) -> None:
    """``load?lease_id=`` is refused for another key's lease (D80), before any load."""
    accounts = app.state.accounts
    mine_record, mine = accounts.create_key("bench", "inference")
    _r, theirs = accounts.create_key("other", "inference")
    lease = app.state.manager.leases.acquire(
        [0], holder="bench", holder_key=f"key:{mine_record.id}"
    )
    seen: list[dict[str, Any]] = []

    async def fake_load(name: str, *args: Any, **kwargs: Any) -> Any:
        seen.append({"name": name, **kwargs})
        return placed(name, [0])

    monkeypatch.setattr(app.state.manager, "load", fake_load)
    monkeypatch.setattr(app.state.manager, "load_recommended", fake_load)
    with TestClient(app, client=LOOPBACK) as http:
        denied = http.post(
            f"/api/models/{ROUTE_MODEL}/load?lease_id={lease.id}",
            json={},
            headers={"Authorization": f"Bearer {theirs}"},
        )
        assert denied.status_code == 403, denied.text
        assert denied.json()["error"]["code"] == "permission_denied"
        assert seen == []
        ok = http.post(
            f"/api/models/{ROUTE_MODEL}/load-recommended",
            json={"ctx_size": 8192, "lease_id": lease.id},
            headers={"Authorization": f"Bearer {mine}"},
        )
        assert ok.status_code == 200, ok.text
        assert seen[0]["lease_id"] == lease.id
        assert seen[0]["source"].startswith(f"api:/api/leases/{lease.id}")
        clash = http.post(
            f"/api/models/{ROUTE_MODEL}/load?lease_id=abc",
            json={"lease_id": "xyz"},
            headers={"Authorization": f"Bearer {mine}"},
        )
        assert clash.status_code == 400 and clash.json()["error"]["param"] == "lease_id"


# ---------------------------------------------------------------------------
# A-9: the planner's ladders and lease owners
# ---------------------------------------------------------------------------


def _planner(book: LeaseBook, *free: float) -> Planner:
    from tests.unit.test_planner import StubProbe, gpu

    cards = [(31.84, (12, 0)), (31.84, (12, 0)), (24.0, (8, 6)), (24.0, (8, 6))]
    probe = StubProbe([gpu(i, cards[i][0], f, cards[i][1]) for i, f in enumerate(free)])
    return Planner(make_config(), probe, leases=book, log_plans=False)


def test_an_outsider_load_never_evicts_a_lease_owner_wherever_it_sits() -> None:
    book = LeaseBook()
    book.acquire([3], holder="bench", model_ids=["owner/model"])
    planner = _planner(book, 2.0, 2.0, 2.0, 2.0)
    owner = resident("owner/model", [0], gib=29.0, idle_s=9000.0)  # least recently used
    bystander = resident("bystander/model", [1], gib=29.0, idle_s=100.0)

    plan = planner.plan_load(
        make_record("outsider/model"), loaded=[owner, bystander], allow_evict=True
    )

    assert isinstance(plan, LoadPlan), plan
    assert plan.evict_model_ids == ["bystander/model"], plan.evict_model_ids
    alone = planner.plan_load(make_record("outsider/model"), loaded=[owner], allow_evict=True)
    assert isinstance(alone, LoadRejected), "with only the owner to evict, the answer is no"


def test_a_lease_mate_may_be_displaced_by_the_holders_own_load() -> None:
    book = LeaseBook()
    book.acquire([3], holder="bench", model_ids=["owner/model", "mine/model"])
    planner = _planner(book, 2.0, 2.0, 2.0, 2.0)
    owner = resident("owner/model", [0], gib=29.0)
    plan = planner.plan_load(make_record("mine/model"), loaded=[owner], allow_evict=True)
    assert isinstance(plan, LoadPlan) and plan.evict_model_ids == ["owner/model"]


def test_a_forced_placement_never_evicts_a_resident_off_its_cards() -> None:
    planner = _planner(LeaseBook(), 2.0, 2.0, 2.0, 2.0)
    elsewhere = resident("far/model", [0], gib=29.0, idle_s=9000.0)
    here = resident("near/model", [2], gib=21.0, idle_s=100.0)
    plan = planner.plan_load(
        make_record("mine/model", device_override=[2]),
        loaded=[elsewhere, here],
        allow_evict=True,
    )
    assert isinstance(plan, LoadPlan), plan
    assert plan.evict_model_ids == ["near/model"], "a victim off the cards frees nothing (A-9)"


def test_the_automatic_ladder_evicts_only_what_sits_on_the_chosen_cards() -> None:
    planner = _planner(LeaseBook(), 2.0, 2.0, 2.0, 2.0)
    small = resident("small/model", [0], gib=1.0, idle_s=9000.0)  # LRU, frees too little
    big = resident("big/model", [1], gib=29.0, idle_s=100.0)
    plan = planner.plan_load(make_record("mine/model"), loaded=[small, big], allow_evict=True)
    assert isinstance(plan, LoadPlan), plan
    assert plan.devices == [1] and plan.evict_model_ids == ["big/model"]


# ---------------------------------------------------------------------------
# A-15: the grant says what it evicted; the 409/503 details
# ---------------------------------------------------------------------------


async def test_the_grant_reply_lists_what_it_evicted() -> None:
    manager, supervisor = make_manager([make_record("bg/model")])
    supervisor.instances["bg/model"] = resident("bg/model", [0], priority=3)
    grant = await manager.grant_lease([0], holder="imagegen")
    assert grant.evicted == [
        {"model_id": "bg/model", "reason": "lease", "devices": [0], "priority": 3}
    ]
    assert [e["evicted"] for e in manager.evictions()] == ["bg/model"]


async def test_each_409_keeps_its_code_and_words_and_names_the_specific_case() -> None:
    """CrucibleForge branches on error.code AND on prose; both stay, the detail is added."""
    manager, supervisor = make_manager([make_record("pinned/model"), make_record("chat/model")])
    supervisor.instances["pinned/model"] = resident("pinned/model", [0], pinned=True)
    supervisor.instances["chat/model"] = resident("chat/model", [1], priority=PRIORITY_CHAT)

    with pytest.raises(LeaseConflictError) as pinned:
        await manager.acquire_lease([0], holder="crucibleforge")
    assert pinned.value.code == "lease_conflict"
    assert "pinned model(s) " in pinned.value.message and "pass force=true" in pinned.value.message
    assert pinned.value.details["code"] == "resident_pinned"
    assert pinned.value.to_payload()["error"]["studioforge"]["code"] == "resident_pinned"

    with pytest.raises(LeaseConflictError) as tiered:
        await manager.acquire_lease([1], holder="crucibleforge")
    assert tiered.value.code == "lease_conflict"
    assert "higher-priority model" in tiered.value.message
    assert "does not outrank" in tiered.value.message
    assert tiered.value.details["code"] == "resident_outranks"

    supervisor.instances.clear()
    manager.leases.acquire([2], holder="imagegen", priority=PRIORITY_AGENT)
    with pytest.raises(LeaseConflictError) as outranked:
        await manager.acquire_lease([2], holder="crucibleforge")
    assert outranked.value.code == "lease_conflict" and "already leased" in outranked.value.message
    assert outranked.value.details["code"] == "lease_outranked"

    manager.leases.acquire([3], holder="speechforge")  # class 3, no vacate_url
    with pytest.raises(LeaseConflictError) as plain:
        await manager.acquire_lease([3], holder="crucibleforge", priority=PRIORITY_AGENT)
    assert plain.value.code == "lease_conflict" and "already leased" in plain.value.message
    assert plain.value.details["code"] == "lease_conflict"


async def test_a_busy_refusal_carries_busy_models_with_in_flight() -> None:
    manager, supervisor = make_manager([make_record("busy/model")])
    supervisor.instances["busy/model"] = resident("busy/model", [0], busy=2)
    with pytest.raises(ModelBusyError) as excinfo:
        await manager.acquire_lease([0], holder="x")
    assert "busy/model (2 in flight) is serving" in excinfo.value.message
    assert excinfo.value.details["busy_models"] == [
        {"model_id": "busy/model", "active_requests": 2, "in_flight": 2}
    ]


def test_a_507_carries_requested_parallel_and_the_lease_reason() -> None:
    manager, _supervisor = make_manager([make_record()])
    shortfall = LoadRejected(
        model_id="m",
        reason="no room",
        estimate=VramEstimate(kv_bytes=40 * GB),
        required_bytes=40 * GB,
        available_bytes=30 * GB,
        max_parallel_that_fits=5,
        suggestions=["reduce parallel from 6 to 5: 5 slot(s) at 8192 tokens fit"],
    )
    error = manager._vram_error(shortfall)
    assert error.code == "insufficient_vram"
    assert error.details["requested_parallel"] == 6
    assert error.details["max_parallel_that_fits"] == 5
    assert "reduce parallel from 6 to 5" in error.message
    leased = LoadRejected(
        model_id="m",
        reason="the requested placement names CUDA [1], which is leased to someone else; a "
        "lease is not a default to override, it is a promise to its holder",
        reason_code="gpu_leased",
    )
    error = manager._vram_error(leased)
    assert error.code == "gpu_leased" and error.details["reason"] == "lease_not_default"
    assert "leased" in error.message and error.details["requested_parallel"] is None


# ---------------------------------------------------------------------------
# A-10: a load in flight elsewhere does not block a grant
# ---------------------------------------------------------------------------


async def test_a_grant_proceeds_beside_a_load_planned_onto_other_cards() -> None:
    manager, _supervisor = make_manager([make_record()])
    manager._loading.add("loading/model")
    manager._loading_devices["loading/model"] = frozenset({0})
    lease = await manager.acquire_lease([2, 3], holder="bench")
    assert lease.devices == [2, 3]
    manager._loading.add("unplanned/model")  # at the gate, no plan yet: reads the book later
    second = await manager.acquire_lease([1], holder="other")
    assert second.devices == [1]


async def test_a_real_load_records_its_planned_cards_while_in_flight() -> None:
    manager, supervisor = walk_manager()
    seen: list[dict[str, frozenset[int]]] = []
    original = supervisor.start

    async def spying_start(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(manager._loading_devices))
        return await original(*args, **kwargs)

    supervisor.start = spying_start  # type: ignore[method-assign]
    instance = await manager.load(MODEL, devices=[2, 3])
    assert instance.plan is not None
    assert seen == [{MODEL: frozenset({2, 3})}]
    assert manager._loading_devices == {}, "cleared once the load settles"


# ---------------------------------------------------------------------------
# A-22 / F-2: release and expiry unload what the lease owns
# ---------------------------------------------------------------------------


async def test_release_unloads_the_leases_idle_models_and_keeps_the_rest() -> None:
    names = ["a/model", "busy/model", "pinned/model", "far/model", "kept/model"]
    manager, supervisor = make_manager([make_record(n) for n in names])
    lease = manager.leases.acquire([0, 1], holder="bench", model_ids=names[:4])
    supervisor.instances["a/model"] = resident("a/model", [0])
    supervisor.instances["busy/model"] = resident("busy/model", [1], busy=1)
    supervisor.instances["pinned/model"] = resident("pinned/model", [1], pinned=True)
    supervisor.instances["far/model"] = resident("far/model", [3])
    supervisor.instances["kept/model"] = resident("kept/model", [0])  # not named

    result = await manager.release_and_unload(lease.id)

    assert result.unloaded == ["a/model"]
    assert {row["model_id"]: row["reason"] for row in result.kept} == {
        "busy/model": "busy",
        "pinned/model": "pinned",
        "far/model": "elsewhere",
    }
    assert set(supervisor.instances) == {"busy/model", "pinned/model", "far/model", "kept/model"}
    assert manager.leases.get(lease.id) is None
    assert manager.evictions()[-1]["reason"] == "lease_released"


async def test_release_with_unload_false_keeps_everything_resident() -> None:
    manager, supervisor = make_manager([make_record("a/model")])
    lease = manager.leases.acquire([0], holder="bench", model_ids=["a/model"])
    supervisor.instances["a/model"] = resident("a/model", [0])
    result = await manager.release_and_unload(lease.id, unload_owned=False)
    assert result.unloaded == [] and "a/model" in supervisor.instances
    assert result.kept == [
        {"model_id": "a/model", "reason": "not_requested", "detail": "not_requested"}
    ]


async def test_expiry_unloads_the_leases_models_but_never_a_benchmarks() -> None:
    manager, supervisor = make_manager([make_record("a/model"), make_record("b/model")])
    old = time.time() - 10_000
    mine = manager.leases.acquire(
        [0], holder="bench", model_ids=["a/model"], idle_ttl_s=60, now=old
    )
    ours = manager.leases.acquire(
        [1], holder="benchmark:parallel", model_ids=["b/model"], idle_ttl_s=60, now=old
    )
    supervisor.instances["a/model"] = resident("a/model", [0], idle_s=9000.0)
    supervisor.instances["b/model"] = resident("b/model", [1], idle_s=9000.0)

    manager._expire_leases()
    await asyncio.gather(*list(manager._lease_unload_tasks))

    assert manager.leases.get(mine.id) is None and manager.leases.get(ours.id) is None
    assert "a/model" not in supervisor.instances
    assert "b/model" in supervisor.instances, "this server's own benchmark leases are unchanged"


def test_expiry_without_a_loop_releases_and_unloads_nothing() -> None:
    manager, supervisor = make_manager([make_record("a/model")])
    old = time.time() - 10_000
    manager.leases.acquire([0], holder="bench", model_ids=["a/model"], idle_ttl_s=60, now=old)
    supervisor.instances["a/model"] = resident("a/model", [0], idle_s=9000.0)
    manager._expire_leases()
    assert len(manager.leases) == 0 and "a/model" in supervisor.instances


# ---------------------------------------------------------------------------
# A-14: automatic placement -- exclusions, owners' placement, count
# ---------------------------------------------------------------------------


def test_class_key_groups_alike_cards_and_never_by_name() -> None:
    cards = four_cards(30.0, 30.0, 23.0, 23.0)
    keys = [card_class_key(card) for card in cards]
    assert keys == ["sm_120-32g", "sm_120-32g", "sm_86-24g", "sm_86-24g"]
    big = GpuInfo(
        index=0,
        name="RTX 5090",
        total_bytes=48 * GIB,
        free_bytes=48 * GIB,
        used_bytes=0,
        compute_capability=(12, 0),
    )
    assert card_class_key(big) == "sm_120-48g", "a 48 GB card is not a 32 GB one"


def test_auto_count_two_picks_two_cards_of_one_class() -> None:
    placement = choose_auto_devices(four_cards(30.0, 10.0, 23.0, 23.5), [], LeaseBook(), count=2)
    assert placement.devices == [2, 3], "the 3090 pair frees more than 5090s at 30+10"
    assert placement.count == 2
    chosen = [row for row in placement.considered if row["verdict"] == VERDICT_CHOSEN]
    assert {row["class_key"] for row in chosen} == {"sm_86-24g"}


def test_auto_skips_excluded_cards_and_cards_the_owners_may_not_use() -> None:
    placement = choose_auto_devices(
        four_cards(30.0, 31.0, 23.0, 23.0),
        [],
        LeaseBook(),
        excluded=[1],
        allowed=[0, 2],
        allowed_reason="'m' saves allowed_devices [0, 2]",
    )
    verdicts = {row["device"]: row["verdict"] for row in placement.considered}
    assert placement.devices == [0]
    assert verdicts[1] == VERDICT_EXCLUDED and verdicts[3] == VERDICT_NOT_ALLOWED


async def test_the_manager_applies_exclusions_owner_placement_and_count() -> None:
    owner = make_record("owner/model", allowed_devices=[2, 3])
    manager, _supervisor = make_manager([owner])
    with_probe(manager, four_cards(31.0, 31.0, 23.0, 23.0))
    manager.config.planner.excluded_devices = [0]
    lease, placement = await manager.acquire_lease_auto(holder="x")
    assert lease.devices == [1], "planner.excluded_devices is honoured (A-14)"
    manager.leases.release(lease.id)

    grant = await manager.grant_lease("auto", holder="x", model_ids=[owner.id], count=2)
    assert grant.lease.devices == [2, 3] and grant.placement is not None
    manager.leases.release(grant.lease.id)

    with pytest.raises(BadRequestError) as too_many:
        await manager.acquire_lease("auto", holder="x", count=5)
    assert too_many.value.param == "count"
    with pytest.raises(BadRequestError) as explicit:
        await manager.acquire_lease([0, 1], holder="x", count=3)
    assert explicit.value.param == "count"


async def test_auto_count_refuses_a_mixed_pair_with_the_classes_named() -> None:
    manager, _supervisor = make_manager([make_record()])
    with_probe(manager, four_cards(31.0, 31.0, 23.0, 23.0))
    manager.leases.acquire([1], holder="a")
    manager.leases.acquire([3], holder="b")
    with pytest.raises(LeaseConflictError):
        # leased cards are what stood in the way: the book's own code
        await manager.acquire_lease("auto", holder="x", count=2)
    manager.leases.release(manager.leases.all()[0].id)
    manager.leases.release(manager.leases.all()[0].id)
    manager.config.planner.excluded_devices = [1, 3]
    with pytest.raises(BadRequestError) as mixed:
        await manager.acquire_lease("auto", holder="x", count=2)
    assert mixed.value.code == "placement_unavailable"
    assert "sm_120-32g" in mixed.value.message and "sm_86-24g" in mixed.value.message


# ---------------------------------------------------------------------------
# The canonical (Kubernetes Lease) shape, every legacy key kept
# ---------------------------------------------------------------------------


def test_a_lease_record_carries_the_k8s_names_beside_equal_legacy_ones() -> None:
    book = LeaseBook()
    lease = book.acquire([2, 3], holder="crucibleforge-judge-b", model_ids=["m"], idle_ttl_s=7200)
    view = lease_view(lease, gpu_uuids={2: "GPU-abc", 3: None})
    assert view["holder_identity"] == view["holder"] == "crucibleforge-judge-b"
    assert view["lease_duration_seconds"] == view["idle_ttl_s"] == 7200
    assert view["acquire_time"] == view["created_at"]
    assert view["renew_time"] == view["last_activity_at"]
    assert view["lease_transitions"] == 0
    assert view["allocation"] == {
        "devices": [{"index": 2, "uuid": "GPU-abc"}, {"index": 3, "uuid": None}]
    }
    assert view["lease_id"] == view["id"] == lease.id
    assert 7100 <= view["expires_in_s"] <= 7200
    for legacy, canonical in LEASE_FIELD_ALIASES.items():
        assert legacy in view and canonical in view, (legacy, canonical)
    assert set(LEASE_DEPRECATED_FIELDS) == set(LEASE_FIELD_ALIASES)


def test_the_holder_classification_contract_is_unchanged() -> None:
    """DisPatch, InfoForge, TransForge, story-*, jobforge read these; pinned (B.0)."""
    for holder in ("crucibleforge", "crucibleforge-judge", "crucibleforge-judge-a"):
        assert holder_family(holder) == "crucibleforge" and lease_kind(holder) == "benchmark"
    assert lease_kind("imagegen") == "render"
    assert lease_kind("speechforge") == "other"


def test_the_lease_routes_speak_both_shapes(app: Any) -> None:
    """CrucibleForge / an image gateway / SpeechForge / makobench-operator / DisPatch shapes, pinned."""
    with TestClient(app, client=LOOPBACK) as http:
        created = http.post(
            "/api/leases",
            json={
                "devices": [0],
                "model_ids": [ROUTE_MODEL],
                "holder": "crucibleforge",
                "reason": "crucibleforge benchmark: thing",
                "idle_ttl_s": 7200.0,
                "force": True,
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        lease_id = body["lease_id"]
        # CrucibleForge reads lease_id | id | lease.lease_id | lease.id; SpeechForge `id`.
        assert body["id"] == lease_id == body["lease"]["id"] == body["lease"]["lease_id"]
        assert body["evicted"] == [] and body["adopted"] == []
        assert body["holder_identity"] == body["holder"] == "crucibleforge"
        assert set(LEASE_DEPRECATED_FIELDS) <= set(body["deprecated_fields"])
        assert "Deprecation" not in created.headers, "the create route itself is canonical"

        listed = http.get("/api/leases").json()
        (row,) = listed["leases"]
        for key in (
            "id",
            "lease_id",
            "holder",
            "holder_family",
            "kind",
            "state",
            "devices",
            "model_ids",
            "priority",
            "reason",
            "expires_at",
            "expires_in_s",
            "retry_after_s",
            "created_at",
            "last_activity_at",
            "idle_ttl_s",
            "idle_s",
            "vacate_registered",
        ):
            assert key in row, key
        assert row["kind"] == "benchmark" and row["holder_family"] == "crucibleforge"
        assert listed["deprecated_fields"] == list(LEASE_DEPRECATED_FIELDS)

        touched = http.post(f"/api/leases/{lease_id}/touch")
        assert touched.status_code == 200
        assert touched.headers["Deprecation"].startswith("@")
        assert 'rel="deprecation"' in touched.headers["Link"]

        before = touched.json()["renew_time"]
        time.sleep(0.01)
        renewed = http.patch(f"/api/leases/{lease_id}", json={"renew_time": "2000-01-01T00:00Z"})
        assert renewed.status_code == 200, renewed.text
        assert renewed.json()["renew_time"] > before, "the server stamps its own clock"
        assert "Deprecation" not in renewed.headers
        bad = http.patch(f"/api/leases/{lease_id}", json={"renew_time": 1, "devices": [1]})
        assert bad.status_code == 400 and bad.json()["error"]["param"] == "devices"
        empty = http.patch(f"/api/leases/{lease_id}", json={})
        assert empty.status_code == 400

        released = http.delete(f"/api/leases/{lease_id}")
        assert released.status_code == 200
        assert released.json()["released"]["id"] == lease_id
        assert released.json()["unloaded"] == [] and released.json()["kept"] == []
        assert http.delete(f"/api/leases/{lease_id}").status_code == 404, "404 = already gone"


def test_patch_renew_is_tenant_protocol_scope() -> None:
    from studioforge.api.auth import required_scope
    from studioforge.config import Config

    assert required_scope("PATCH", "/api/leases/abc123", Config()) == "inference"


def test_a_lease_refusal_over_http_keeps_the_envelope_and_adds_the_code(app: Any) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        first = http.post("/api/leases", json={"devices": [0], "holder": "imagegen"})
        assert first.status_code == 200
        refused = http.post("/api/leases", json={"devices": [0], "holder": "crucibleforge"})
        assert refused.status_code == 409
        error = refused.json()["error"]
        assert error["code"] == "lease_conflict" and "already leased" in error["message"]
        assert error["studioforge"]["code"] == "lease_outranked"
        assert refused.headers.get("retry-after") is not None


# ---------------------------------------------------------------------------
# A-21: MCP reserve_gpus parity, renew/touch, release unload
# ---------------------------------------------------------------------------


async def test_mcp_reserve_auto_with_steering_and_renew_round_trip(state: Any) -> None:
    server = build_management_mcp(state)
    reserved = await call(
        server, "reserve_gpus", devices="auto", min_free_gib=25.0, model_ids=[TINY], holder="mb"
    )
    assert reserved["ok"] is True, reserved
    lease = reserved["lease"]
    assert lease["devices"] == [0], "only the 32 GiB card has 25 GiB free"
    assert reserved["placement"]["mode"] == "auto" and reserved["evicted"] == []
    assert lease["model_ids"] == [TINY]

    before = lease["renew_time"]
    await asyncio.sleep(0.01)
    renewed = await call(server, "renew_lease", lease_id=lease["id"])
    assert renewed["ok"] is True and renewed["lease"]["renew_time"] > before
    touched = await call(server, "touch_lease", lease_id=lease["id"])
    assert touched["ok"] is True and touched["lease"]["id"] == lease["id"]

    released = await call(server, "release_gpus", lease_id=lease["id"], unload=False)
    assert released["ok"] is True and released["unloaded"] == []
    missing = await call(server, "renew_lease", lease_id=lease["id"])
    assert missing["ok"] is False


async def test_mcp_reserve_keeps_its_old_call_shape(state: Any) -> None:
    """OpenClaw's recipe: reserve_gpus(devices=[0], model_id=...) -> lease.id."""
    server = build_management_mcp(state)
    schema = next(t for t in await server.list_tools() if t.name == "reserve_gpus").input_schema
    for name in ("devices", "model_id", "reason", "idle_ttl_s", "force", "holder", "model_ids"):
        assert name in schema["properties"], name
    reserved = await call(server, "reserve_gpus", devices=[1], model_id=TINY)
    assert reserved["ok"] is True and reserved["lease"]["devices"] == [1]
    assert reserved["lease"]["model_ids"] == [TINY]
