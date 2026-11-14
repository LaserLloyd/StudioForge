"""Several instances of one model under leases, by default (D91, work package WP4).

The incident this pins (2026-11-14 02:10): holder A served a model on CUDA [2, 3]
with three streams in flight; holder B leased [0, 1] naming the same id, and
B's unload -- passed as a forced unload because the two shared one inference
key -- cut A's streams; A's next request reloaded the id on the default ladder.

Here: instance ids (``<model>:<n>`` from 2 with gaps reused, ``<model>@<tag>``,
collisions refused), parity with the base, the launch keys, the all-or-nothing
grant, the second holder's own copy, routing by holder identity, unload
targeting and the in-flight refusal, the lifecycle (release, expiry, the
``instance_released`` / ``instance_not_found`` errors), the restart restore with
the exact stored shape, the lease row, and the MCP round trip. No GPU, no port:
the walk manager's real planner over four fake cards, a stub supervisor.
The end-to-end proof on stub staging is ``test_lease_instances_staging.py``.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core.capabilities import SERVER_FEATURES, implemented_report
from studioforge.core.instances import (
    Caller,
    claims,
    names_other_holder,
    next_free_number,
    normalise_launch,
    numbered_id,
    parity_view,
    parse_instance_requests,
    parse_numbered_id,
)
from studioforge.core.leases import LeaseBook, lease_from_row, lease_row, lease_view
from studioforge.errors import BadRequestError, LeaseConflictError, StudioForgeError
from studioforge.mcp.management import build_management_mcp
from studioforge.types import GpuLease, InFlightRequest, InstanceInfo, ModelSettings
from tests.unit.test_load_recommended import MODEL
from tests.unit.test_load_recommended import make_manager as walk_manager
from tests.unit.test_load_retry import StubSupervisor
from tests.unit.test_mcp import TINY, call
from tests.unit.test_mcp import state as state  # noqa: PLC0414 - fixture re-export

KEY = "key:shared-bench-key"
A = "benchrun-tput-d"
B = "benchrun-tput-p"


def manager_with(**kwargs: Any) -> tuple[Any, StubSupervisor]:
    manager, supervisor = walk_manager(**kwargs)
    manager.planner.leases = manager.leases
    return manager, supervisor


def streams(instance: InstanceInfo, client: str, count: int = 3) -> None:
    """``count`` requests in flight on ``instance``, from ``client``."""
    now = time.time()
    instance.active_requests = count
    instance.in_flight = [
        InFlightRequest(id=f"req-{client}-{i}", started_at=now, client=client) for i in range(count)
    ]


class MemoryStore:
    """A lease store that round-trips through the stored row shape (D61), in memory."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def put(self, lease: GpuLease) -> None:
        self.rows[lease.id] = lease_row(lease)

    def delete(self, lease_id: str) -> None:
        self.rows.pop(lease_id, None)

    def load(self) -> list[GpuLease]:
        return [lease_from_row(row) for row in self.rows.values()]


# ---------------------------------------------------------------------------
# Ids
# ---------------------------------------------------------------------------


def test_the_id_grammar() -> None:
    assert numbered_id("pub/x", 2) == "pub/x:2"
    assert parse_numbered_id("pub/x:12") == ("pub/x", 12)
    assert parse_numbered_id("pub/x@b") is None and parse_numbered_id("pub/x") is None
    taken = {"pub/x:2", "pub/x:3", "pub/x:5"}
    assert next_free_number("pub/x", taken.__contains__) == 4
    assert next_free_number("pub/x", set().__contains__) == 2


async def test_numbered_ids_start_at_two_and_are_never_reused_while_running() -> None:
    """A released ``X:2`` stays a 404 for a late request: its number is not handed out again."""
    manager, _ = manager_with()
    first = await manager.grant_lease(
        [0], holder="a", instances=[{"model": MODEL, "count": 2, "load": False}]
    )
    assert [row["id"] for row in first.instances] == [f"{MODEL}:2", f"{MODEL}:3"]
    second = await manager.grant_lease([1], holder="b", instances=[{"model": MODEL, "load": False}])
    assert [row["id"] for row in second.instances] == [f"{MODEL}:4"]
    await manager.release_and_unload(first.lease.id)
    third = await manager.grant_lease([2], holder="c", instances=[{"model": MODEL, "load": False}])
    assert [row["id"] for row in third.instances] == [f"{MODEL}:5"], "numbers are monotonic"
    late = manager.instance_gone_error(f"{MODEL}:2")
    assert late is not None and late.code == "instance_not_found"
    assert late.details["lease_id"] == first.lease.id, "the late request reaches no new holder"
    # A tag, asked for explicitly, may be reused once its lease ended.
    tagged = await manager.grant_lease(
        [3], holder="d", instances=[{"model": MODEL, "tag": "run", "load": False}]
    )
    await manager.release_and_unload(tagged.lease.id)
    again = await manager.grant_lease(
        [3], holder="d", instances=[{"model": MODEL, "tag": "run", "load": False}]
    )
    assert [row["id"] for row in again.instances] == [f"{MODEL}@run"]


async def test_a_tag_names_the_instance_and_a_taken_id_is_refused() -> None:
    manager, _ = manager_with()
    tagged = await manager.grant_lease(
        [0], holder="a", instances=[{"model": MODEL, "tag": "mb-a", "load": False}]
    )
    assert [row["id"] for row in tagged.instances] == [f"{MODEL}@mb-a"]
    with pytest.raises(StudioForgeError) as clash:
        await manager.grant_lease(
            [1], holder="b", instances=[{"model": MODEL, "tag": "mb-a", "load": False}]
        )
    assert clash.value.code == "instance_id_conflict" and clash.value.status_code == 409
    assert len(manager.leases) == 1, "a refused grant leaves no lease behind"

    # The judge's persistent ``...@b`` keeps its name: a tag cannot take it.
    base = manager.registry.get(MODEL)
    persistent = base.model_copy(
        update={"id": f"{MODEL}@b", "is_virtual": True, "base_model_id": MODEL}
    )
    manager.registry._records[persistent.id] = persistent  # noqa: SLF001
    with pytest.raises(StudioForgeError) as judge:
        await manager.grant_lease(
            [1], holder="b", instances=[{"model": MODEL, "tag": "b", "load": False}]
        )
    assert judge.value.code == "instance_id_conflict"
    assert manager.registry.get(f"{MODEL}@b") is persistent

    counted = await manager.grant_lease(
        [2], holder="c", instances=[{"model": MODEL, "tag": "t", "count": 2, "load": False}]
    )
    assert [row["id"] for row in counted.instances] == [f"{MODEL}@t1", f"{MODEL}@t2"]


async def test_an_instance_of_an_instance_and_gpus_off_the_lease_are_refused() -> None:
    manager, _ = manager_with()
    grant = await manager.grant_lease([0], holder="a", instances=[{"model": MODEL, "load": False}])
    with pytest.raises(BadRequestError):
        await manager.grant_lease(
            [1], holder="b", instances=[{"model": grant.instances[0]["id"], "load": False}]
        )
    with pytest.raises(BadRequestError) as outside:
        await manager.grant_lease(
            [1], holder="b", instances=[{"model": MODEL, "gpus": [1, 2], "load": False}]
        )
    assert outside.value.param == "instances[0].gpus"
    assert len(manager.leases) == 1


# ---------------------------------------------------------------------------
# Parity and the launch keys
# ---------------------------------------------------------------------------


async def test_the_copy_is_the_base_but_for_launch_and_placement() -> None:
    settings = ModelSettings(
        ctx_size=65536,
        kv_cache_type="f16",
        temperature=0.6,
        top_k=20,
        reasoning_format="deepseek",
        extra_flags="--no-warmup",
        engine_tag="b12110",
        device_override=[0],
        allowed_devices=[0, 1],
        main_gpu=0,
        ttl_s=900,
        pinned=True,
    )
    manager, _ = manager_with(settings=settings)
    launch = {"ctx_size": 16384, "parallel": 2, "kv_cache_type": "q8_0", "kv_cache_type_v": "q8_0"}
    grant = await manager.grant_lease(
        [2, 3], holder="judge", instances=[{"model": MODEL, "launch": launch, "load": False}]
    )
    copy = manager.registry.get(grant.instances[0]["id"])
    base = manager.registry.get(MODEL)
    assert parity_view(copy.settings) == parity_view(base.settings), "parity is the server's"
    assert copy.settings.ctx_size == 16384 and copy.settings.parallel == 2
    assert copy.settings.kv_cache_type == "q8_0" and copy.settings.kv_cache_type_v == "q8_0"
    assert copy.settings.device_override == [2, 3], "placement follows the instance's cards"
    assert copy.settings.allowed_devices is None and copy.settings.main_gpu is None
    assert copy.settings.pinned is False, "an instance lives and dies with its lease"
    assert copy.path == base.path and copy.ephemeral and copy.base_model_id == MODEL
    assert manager.serving_record(copy) is copy, "an instance always has its own child"


def test_a_launch_key_outside_the_six_is_a_400_with_a_fix() -> None:
    with pytest.raises(BadRequestError) as refused:
        normalise_launch({"ctx_size": 8192, "temperature": 0.2})
    assert refused.value.code == "launch_key_not_allowed"
    assert "copied from the base" in refused.value.message
    assert refused.value.to_payload()["error"]["fix"], "the catalog's fix text rides along"
    assert normalise_launch({"context_length": 4096}) == {"ctx_size": 4096}
    with pytest.raises(BadRequestError):
        normalise_launch({"min_slots": 4, "max_slots": 2})
    with pytest.raises(BadRequestError):
        normalise_launch({"parallel": 6, "max_slots": 4})
    with pytest.raises(BadRequestError):
        parse_instance_requests([{"model": "x", "launhc": {}}])


async def test_a_bad_launch_value_is_refused_before_anything_is_made() -> None:
    manager, supervisor = manager_with()
    with pytest.raises(BadRequestError) as refused:
        await manager.grant_lease(
            [0], holder="a", instances=[{"model": MODEL, "launch": {"kv_cache_type": "q9_9"}}]
        )
    assert refused.value.param == "kv_cache_type"
    with pytest.raises(BadRequestError) as unsized:
        await manager.grant_lease(
            [0], holder="a", instances=[{"model": MODEL, "launch": {"max_slots": 4}}]
        )
    assert unsized.value.param == "instances[0].launch.ctx_size"
    assert len(manager.leases) == 0 and supervisor.starts == 0


# ---------------------------------------------------------------------------
# The grant: loads, reports the real shape, all or nothing
# ---------------------------------------------------------------------------


async def test_the_grant_loads_each_instance_and_reports_what_actually_loaded() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [2, 3],
        holder="bench",
        instances=[{"model": MODEL, "launch": {"ctx_size": 8192, "parallel": 2}}],
    )
    (row,) = grant.instances
    assert row["id"] == row["model"] == f"{MODEL}:2"
    assert row["state"] == "loaded" and row["devices"] == [2, 3]
    assert row["ctx_per_slot"] == 8192 and row["parallel"] == 2
    assert row["kv_cache_type"] and row["kv_cache_type_v"]
    assert row["lease_id"] == grant.lease.id and row["holder"] == "bench"
    instance = supervisor.get(f"{MODEL}:2")
    assert instance is not None and sorted(instance.plan.devices) == [2, 3]
    assert instance.loaded_by.startswith(f"api:/api/leases/{grant.lease.id}")
    assert grant.lease.model_ids == [f"{MODEL}:2"]
    assert grant.lease.instances[0].loaded_spec["ctx_per_slot"] == 8192, "the shape is stored"
    view = lease_view(grant.lease, live=manager.live_instances(grant.lease))
    assert view["instances"][0]["state"] == "loaded"


async def test_slot_bounds_walk_the_instances_own_cards() -> None:
    manager, _ = manager_with()
    grant = await manager.grant_lease(
        [0, 1],
        holder="bench",
        instances=[{"model": MODEL, "launch": {"ctx_size": 16384, "min_slots": 2, "max_slots": 3}}],
    )
    (row,) = grant.instances
    assert row["devices"] == [0, 1] and row["ctx_per_slot"] == 16384
    assert 2 <= row["parallel"] <= 3


async def test_one_failed_instance_rolls_the_whole_grant_back() -> None:
    manager, supervisor = manager_with()
    real_start = supervisor.start

    async def start(record: Any, plan: Any, **kwargs: Any) -> InstanceInfo:
        if record.id.endswith(":3"):
            from studioforge.errors import ModelLoadError

            raise ModelLoadError(f"llama-server for '{record.id}' exited with code 1.")
        return await real_start(record, plan, **kwargs)

    supervisor.start = start  # type: ignore[method-assign]
    with pytest.raises(StudioForgeError) as failed:
        await manager.grant_lease(
            [0, 1, 2, 3],
            holder="bench",
            instances=[
                {"model": MODEL, "gpus": [0, 1], "launch": {"ctx_size": 4096}},
                {"model": MODEL, "gpus": [2, 3], "launch": {"ctx_size": 4096}},
            ],
        )
    error = failed.value
    assert error.code == "model_load_failed"
    details = error.details
    assert details["lease_rolled_back"] is True
    states = {row["id"]: row["state"] for row in details["instances"]}
    assert states == {f"{MODEL}:2": "rolled_back", f"{MODEL}:3": "failed"}
    failure = next(row for row in details["instances"] if row["state"] == "failed")
    assert failure["code"] == "model_load_failed" and failure["reason"]
    assert len(manager.leases) == 0, "the lease was released"
    assert manager.registry.get(f"{MODEL}:2") is None and manager.registry.get(f"{MODEL}:3") is None
    assert supervisor.get(f"{MODEL}:2") is None, "the instance already up was unloaded"
    assert f"{MODEL}:2" in supervisor.stopped


# ---------------------------------------------------------------------------
# A second holder never replaces the first (the 02:10 scenario)
# ---------------------------------------------------------------------------


async def test_the_0210_scenario_the_second_holder_gets_its_own_instance() -> None:
    manager, supervisor = manager_with()
    lease_a = await manager.grant_lease([2, 3], holder=A, model_ids=[MODEL], holder_key=KEY)
    serving_a = await manager.load(MODEL, lease_id=lease_a.lease.id)
    assert sorted(serving_a.plan.devices) == [2, 3]
    streams(serving_a, A)
    starts = supervisor.starts

    # B leases [0, 1] naming the same id, the OLD shape, with the SAME key.
    lease_b = await manager.grant_lease([0, 1], holder=B, model_ids=[MODEL], holder_key=KEY)
    assert lease_b.lease.model_ids == [f"{MODEL}:2"], "B never names A's instance"
    (row,) = lease_b.instances
    assert row["origin"] == "second_holder" and row["state"] == "unloaded"
    assert row["devices"] == [0, 1]
    assert supervisor.get(MODEL) is serving_a and serving_a.active_requests == 3
    assert supervisor.starts == starts, "a grant reserves; nothing loaded or reloaded"

    # B's requests for the base id go to B's own instance -- by its label, or by
    # its lease -- and load it on B's cards; A's stay on A's.
    b_label = Caller(client=B, key=KEY)
    routed = manager.route_for(manager.registry.get(MODEL), b_label)
    assert routed.id == f"{MODEL}:2"
    record, instance = await manager.ensure_loaded(routed.id)
    assert sorted(instance.plan.devices) == [0, 1]
    assert manager.route_for(manager.registry.get(MODEL), Caller(client=A, key=KEY)).id == MODEL

    # B's unload of the base id targets only B's instance.
    assert manager.unload_target(MODEL, b_label) == f"{MODEL}:2"
    assert await manager.unload(MODEL, caller=b_label, lease_ok=True) is True
    assert supervisor.get(f"{MODEL}:2") is None
    assert supervisor.get(MODEL) is serving_a and serving_a.active_requests == 3
    assert MODEL not in supervisor.stopped, "A's streams were never cut"


async def test_a_lease_naming_a_model_serving_other_clients_elsewhere_gets_a_copy() -> None:
    manager, supervisor = manager_with()
    resident = await manager.load(MODEL, devices=[2, 3])
    streams(resident, "clawchat")
    grant = await manager.grant_lease([0, 1], holder="bench", model_ids=[MODEL])
    assert grant.lease.model_ids == [f"{MODEL}:2"]
    assert supervisor.get(MODEL) is resident
    # The same model idle and unleased elsewhere is today's behaviour: named.
    manager2, supervisor2 = manager_with()
    await manager2.load(MODEL, devices=[2, 3])
    plain = await manager2.grant_lease([0, 1], holder="bench", model_ids=[MODEL])
    assert plain.lease.model_ids == [MODEL] and plain.instances == []


async def test_a_grant_never_evicts_another_leases_model_from_its_cards() -> None:
    manager, supervisor = manager_with()
    lease_a = manager.leases.acquire([3], holder=A, model_ids=[MODEL])
    resident = await manager.load(MODEL, devices=[2])
    assert manager.leases.for_model(MODEL) is lease_a
    with pytest.raises(LeaseConflictError) as refused:
        await manager.grant_lease([2], holder=B)
    assert refused.value.code == "instance_held"
    assert refused.value.details["lease_id"] == lease_a.id
    assert supervisor.get(MODEL) is resident


async def test_an_unbound_forced_reload_of_another_holders_model_is_refused() -> None:
    manager, supervisor = manager_with()
    lease_a = await manager.grant_lease([2, 3], holder=A, model_ids=[MODEL], holder_key=KEY)
    resident = await manager.load(MODEL, lease_id=lease_a.lease.id)
    stranger = Caller(client="someone-else", key="key:other")
    with pytest.raises(LeaseConflictError) as refused:
        await manager.load(MODEL, force=True, caller=stranger)
    assert refused.value.code == "instance_held"
    with pytest.raises(LeaseConflictError):
        await manager.load_recommended(MODEL, 4096, caller=stranger)
    assert supervisor.get(MODEL) is resident
    admin = Caller(admin=True)
    reloaded = await manager.load(MODEL, force=True, caller=admin)
    assert reloaded is not resident, "an admin's explicit force is the old behaviour"


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


async def test_routing_exact_ids_own_instances_and_everyone_else() -> None:
    manager, _ = manager_with()
    mine = await manager.grant_lease(
        [0, 1], holder=A, holder_key="key:a", instances=[{"model": MODEL, "load": False}]
    )
    theirs = await manager.grant_lease(
        [2, 3], holder=B, holder_key="key:b", instances=[{"model": MODEL, "load": False}]
    )
    base = manager.registry.get(MODEL)
    a_copy, b_copy = mine.instances[0]["id"], theirs.instances[0]["id"]

    exact = manager.registry.get(b_copy)
    assert manager.route_for(exact, Caller(client=A)) is exact, "an exact id routes exactly"
    assert manager.route_for(base, Caller(client=A)).id == a_copy, "by X-SF-Client"
    assert manager.route_for(base, Caller(key="key:b")).id == b_copy, "by the lease's key"
    assert manager.route_for(base, Caller(lease_ids=(mine.lease.id,))).id == a_copy
    assert manager.route_for(base, Caller(client="third", key="key:c")) is base
    assert manager.route_for(base, None) is base
    # One key behind both leases and no label: ambiguous, so the base as before.
    manager.leases.get(theirs.lease.id).holder_key = "key:a"  # type: ignore[union-attr]
    assert manager.route_for(base, Caller(key="key:a")) is base


async def test_a_holder_with_two_copies_is_routed_to_the_least_busy() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1, 2, 3],
        holder=A,
        instances=[
            {"model": MODEL, "gpus": [0, 1], "launch": {"ctx_size": 4096}},
            {"model": MODEL, "gpus": [2, 3], "launch": {"ctx_size": 4096}},
        ],
    )
    first, second = (row["id"] for row in grant.instances)
    streams(supervisor.get(first), A, count=2)
    assert manager.route_for(manager.registry.get(MODEL), Caller(client=A)).id == second


# ---------------------------------------------------------------------------
# Unload targeting and the in-flight refusal (8, 8a)
# ---------------------------------------------------------------------------


async def test_another_holders_instance_needs_its_exact_id_admin_and_force() -> None:
    manager, supervisor = manager_with()
    mine = await manager.grant_lease(
        [0, 1], holder=A, holder_key=KEY, instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}]
    )
    await manager.grant_lease(
        [2, 3], holder=B, holder_key=KEY, instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}]
    )
    a_copy = mine.instances[0]["id"]
    streams(supervisor.get(a_copy), A)
    b_caller = Caller(client=B, key=KEY)
    with pytest.raises(LeaseConflictError) as held:
        await manager.unload(a_copy, caller=b_caller, lease_ok=True)
    assert held.value.code == "instance_held" and held.value.details["holder"] == A
    with pytest.raises(LeaseConflictError):
        await manager.unload(a_copy, caller=Caller(admin=True), lease_ok=True)
    assert supervisor.get(a_copy).active_requests == 3, "A's streams intact"
    assert await manager.unload(a_copy, caller=Caller(admin=True), lease_ok=True, force=True)
    assert supervisor.get(a_copy) is None


async def test_a_holders_unload_never_cuts_another_clients_requests_without_force() -> None:
    manager, supervisor = manager_with()
    lease = await manager.grant_lease([2, 3], holder=A, model_ids=[MODEL], holder_key=KEY)
    resident = await manager.load(MODEL, lease_id=lease.lease.id)
    streams(resident, "a-worker")
    me = Caller(client=A, key=KEY)
    with pytest.raises(StudioForgeError) as busy:
        await manager.unload(MODEL, caller=me, lease_ok=True)
    assert busy.value.code == "instance_in_use" and busy.value.status_code == 409
    assert busy.value.details["in_flight_clients"] == ["a-worker"]
    assert supervisor.get(MODEL) is resident
    # The holder's own requests are its own business; force says "cut them".
    streams(resident, A)
    assert await manager.unload(MODEL, caller=me, lease_ok=True) is True
    resident = await manager.load(MODEL, lease_id=lease.lease.id)
    streams(resident, "a-worker")
    assert await manager.unload(MODEL, caller=me, lease_ok=True, force=True) is True


async def test_two_copies_of_one_holder_make_a_base_id_unload_ambiguous() -> None:
    manager, _ = manager_with()
    await manager.grant_lease(
        [0, 1, 2, 3],
        holder=A,
        instances=[
            {"model": MODEL, "gpus": [0, 1], "load": False},
            {"model": MODEL, "gpus": [2, 3], "load": False},
        ],
    )
    with pytest.raises(StudioForgeError) as ambiguous:
        manager.unload_target(MODEL, Caller(client=A))
    assert ambiguous.value.code == "instance_ambiguous"
    assert ambiguous.value.details["instances"] == [f"{MODEL}:2", f"{MODEL}:3"]


def test_holder_identity_beats_a_shared_key() -> None:
    a = GpuLease(id="la", devices=[2, 3], holder=A, created_at=1, last_activity_at=1)
    b = GpuLease(id="lb", devices=[0, 1], holder=B, created_at=1, last_activity_at=1)
    a.holder_key = b.holder_key = KEY
    standing = [a, b]
    as_b = Caller(client=B, key=KEY)
    assert names_other_holder(as_b, a, standing) and not claims(as_b, a, standing)
    assert claims(as_b, b, standing)
    assert claims(Caller(key=KEY), a, standing) and claims(Caller(key=KEY), b, standing)
    assert names_other_holder(Caller(lease_ids=("lb",), key=KEY), a, standing)
    assert names_other_holder(Caller(client="clawchat", key=KEY), a, standing)
    # A family label that is no standing holder's is not a claim to be someone else.
    judge = GpuLease(
        id="lj", devices=[0], holder="crucibleforge-judge", created_at=1, last_activity_at=1
    )
    assert not names_other_holder(Caller(client="crucibleforge"), judge, [judge])


# ---------------------------------------------------------------------------
# Lifecycle: release, expiry, the two errors
# ---------------------------------------------------------------------------


async def test_release_ends_busy_instances_and_a_later_request_is_404_never_a_load() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1], holder=A, instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}]
    )
    copy = grant.instances[0]["id"]
    streams(supervisor.get(copy), A)
    result = await manager.release_and_unload(grant.lease.id, unload_owned=False)
    assert result.instances_ended == [copy] and copy in result.unloaded
    assert supervisor.get(copy) is None, "an instance ends with its lease, busy or not"
    assert manager.registry.get(copy) is None
    released = manager.instance_released_error(copy)
    assert released is not None and released.code == "instance_released"
    assert released.status_code == 410
    gone = manager.instance_gone_error(copy)
    assert gone is not None and gone.code == "instance_not_found" and gone.status_code == 404
    assert gone.details["lease_id"] == grant.lease.id
    never = manager.instance_gone_error(f"{MODEL}:9")
    assert never is not None and never.code == "instance_not_found"
    assert manager.instance_gone_error("unrelated/model:2") is None
    starts = supervisor.starts
    with pytest.raises(Exception):  # noqa: B017 - the registry no longer knows the id
        await manager.ensure_loaded(copy)
    assert supervisor.starts == starts, "no load was attempted"


async def test_expiry_ends_the_instances_too() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1],
        holder=A,
        idle_ttl_s=60.0,
        instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}],
    )
    copy = grant.instances[0]["id"]
    instance = supervisor.get(copy)
    instance.last_activity_at = instance.started_at = time.time() - 3600
    grant.lease.last_activity_at = time.time() - 3600
    manager._expire_leases()  # noqa: SLF001
    assert len(manager.leases) == 0 and manager.registry.get(copy) is None
    for task in list(manager._lease_unload_tasks):  # noqa: SLF001
        await task
    assert supervisor.get(copy) is None
    assert manager.released_instance(copy)["why"] == "expired"


async def test_the_sync_release_path_ends_instances_as_well() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1], holder=A, instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}]
    )
    copy = grant.instances[0]["id"]
    manager.release_lease(grant.lease.id)
    assert manager.registry.get(copy) is None
    for task in list(manager._lease_unload_tasks):  # noqa: SLF001
        await task
    assert supervisor.get(copy) is None


class _BrokenStream:
    status_code = 200

    async def __aenter__(self) -> Any:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def aiter_raw(self) -> Any:
        raise httpx.ReadError("connection closed")
        yield b""  # pragma: no cover - makes this an async generator

    async def aread(self) -> bytes:
        return b""


class _BrokenClient:
    def stream(self, *_args: object, **_kwargs: object) -> _BrokenStream:
        return _BrokenStream()

    async def post(self, *_args: object, **_kwargs: object) -> Any:
        raise httpx.RemoteProtocolError("server disconnected")


async def test_a_request_in_flight_when_its_instance_ends_gets_instance_released() -> None:
    from studioforge.api import openai_routes
    from tests.unit.test_gateway_lifecycle import CountingSupervisor

    manager, _ = manager_with()
    grant = await manager.grant_lease([0, 1], holder=A, instances=[{"model": MODEL, "load": False}])
    record = manager.registry.get(grant.instances[0]["id"])
    await manager.release_and_unload(grant.lease.id)
    supervisor = CountingSupervisor()
    state = SimpleNamespace(
        supervisor=supervisor,
        client=_BrokenClient(),
        config=Config(data_dir="/tmp/sf-wp4"),
        manager=manager,
    )
    frames = [
        chunk
        async for chunk in openai_routes._stream_upstream(  # noqa: SLF001
            state, record, "http://127.0.0.1:1/v1/chat/completions", {"stream": True}, 0.0
        )
    ]
    text = b"".join(frames).decode()
    assert '"code": "instance_released"' in text and text.rstrip().endswith("data: [DONE]")
    with pytest.raises(StudioForgeError) as plain:
        await openai_routes._forward(state, record, "/v1/chat/completions", {})  # noqa: SLF001
    assert plain.value.code == "instance_released" and plain.value.status_code == 410


# ---------------------------------------------------------------------------
# Restart: restored unloaded, loaded with exactly the stored shape
# ---------------------------------------------------------------------------


async def test_a_restart_restores_instances_unloaded_and_loads_the_stored_shape() -> None:
    store = MemoryStore()
    manager, supervisor = manager_with()
    manager.leases = LeaseBook(store=store)
    manager.planner.leases = manager.leases
    grant = await manager.grant_lease(
        [2, 3],
        holder=A,
        instances=[
            {"model": MODEL, "launch": {"ctx_size": 12288, "parallel": 3, "kv_cache_type": "q8_0"}}
        ],
    )
    copy = grant.instances[0]["id"]
    shape = grant.lease.instances[0].loaded_spec
    assert shape["ctx_per_slot"] == 12288 and shape["parallel"] == 3
    assert store.rows[grant.lease.id]["instances"][0]["loaded_spec"] == shape

    # A new process: an empty supervisor, the same store.
    reborn, reborn_supervisor = manager_with()
    reborn.leases = LeaseBook(store=store)
    reborn.planner.leases = reborn.leases
    reborn._restore_leases()  # noqa: SLF001
    restored = reborn.registry.get(copy)
    assert restored is not None and restored.ephemeral and restored.lease_id == grant.lease.id
    assert reborn_supervisor.get(copy) is None, "restored unloaded"
    _record, instance = await reborn.ensure_loaded(copy)
    plan = instance.plan
    assert sorted(plan.devices) == shape["devices"]
    assert plan.ctx_per_slot == 12288 and plan.parallel == 3
    assert plan.kv_cache_type == "q8_0", "exactly the stored shape, never the ladder"


def test_the_lease_row_carries_the_instances() -> None:
    lease = GpuLease(id="l1", devices=[0, 1], holder=A, created_at=1.0, last_activity_at=1.0)
    from studioforge.types import LeaseInstance

    lease.instances = [
        LeaseInstance(
            id=f"{MODEL}:2",
            base_model_id=MODEL,
            devices=[0, 1],
            launch={"ctx_size": 4096},
            loaded_spec={"devices": [0, 1], "ctx_per_slot": 4096, "parallel": 2},
        )
    ]
    back = lease_from_row(lease_row(lease))
    assert back.instances == lease.instances
    unreadable = {**lease_row(lease), "instances": [{"id": "x"}]}
    assert lease_from_row(unreadable).instances == [], "a bad spec is skipped, not fatal"


def test_the_database_round_trips_instances(tmp_path: Any) -> None:
    from studioforge.db import SCHEMA_VERSION, Database
    from studioforge.types import LeaseInstance

    assert SCHEMA_VERSION >= 12
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    try:
        lease = GpuLease(id="l1", devices=[0], holder=A, created_at=1.0, last_activity_at=1.0)
        lease.instances = [LeaseInstance(id=f"{MODEL}@t", base_model_id=MODEL, devices=[0])]
        db.save_lease(lease_row(lease))
        (row,) = db.list_leases()
        assert lease_from_row(row).instances == lease.instances
    finally:
        db.close()


# ---------------------------------------------------------------------------
# MCP and capabilities
# ---------------------------------------------------------------------------


async def test_mcp_reserve_gpus_takes_instances_round_trip(state: Any) -> None:
    server = build_management_mcp(state)
    schema = next(t for t in await server.list_tools() if t.name == "reserve_gpus").input_schema
    assert "instances" in schema["properties"]
    reserved = await call(
        server,
        "reserve_gpus",
        devices=[0],
        holder="mb",
        instances=[{"model": TINY, "tag": "mb-a", "launch": {"ctx_size": 4096}, "load": False}],
    )
    assert reserved["ok"] is True, reserved
    (row,) = reserved["instances"]
    assert row["model"] == f"{TINY}@mb-a" and row["state"] == "unloaded"
    assert reserved["lease"]["instances"][0]["id"] == f"{TINY}@mb-a"
    listed = await call(server, "gpu_leases")
    assert listed["leases"][0]["instances"][0]["holder"] == "mb"
    bad = await call(
        server, "reserve_gpus", devices=[1], instances=[{"model": TINY, "launch": {"top_k": 1}}]
    )
    assert bad["ok"] is False and bad["error"]["code"] == "launch_key_not_allowed"
    released = await call(server, "release_gpus", lease_id=reserved["lease"]["id"])
    assert released["ok"] is True and released["instances_ended"] == [f"{TINY}@mb-a"]


def test_the_capability_features_name_d91() -> None:
    assert SERVER_FEATURES["lease_instances"] == "D91"
    assert SERVER_FEATURES["multi_instance_default"] == "D91"
    report = implemented_report()
    assert "lease_instances" in report["features"] and report["latest_decision"] == "D91"


def test_the_panel_lease_line_names_the_instances() -> None:
    from studioforge.gui import state as st
    from studioforge.types import LeaseInstance

    lease = GpuLease(id="l1", devices=[0, 1], holder=A, created_at=1.0, last_activity_at=1.0)
    lease.instances = [LeaseInstance(id=f"{MODEL}:2", base_model_id=MODEL, devices=[0, 1])]
    lease.model_ids = [f"{MODEL}:2"]
    line = st.lease_line(lease)
    assert f"instances: {MODEL}:2 [0, 1]" in line and "by " + A in line


def test_the_mcp_recipes_teach_two_copies_of_one_model() -> None:
    from studioforge.mcp.management import INSTRUCTIONS

    assert "two copies of one model" in INSTRUCTIONS
    assert 'instances=[{"model": "<id>"}]' in INSTRUCTIONS


async def test_an_instance_has_no_settings_of_its_own(state: Any) -> None:
    """Parity is the server's: a settings save, a delete or a persist on a copy is refused."""
    grant = await state.manager.grant_lease(
        [0], holder="mb", instances=[{"model": TINY, "load": False}]
    )
    copy = grant.instances[0]["id"]
    with pytest.raises(BadRequestError) as saved:
        state.registry.save_settings(copy, ModelSettings(ctx_size=4096))
    assert saved.value.code == "unsupported"
    with pytest.raises(BadRequestError) as deleted:
        state.registry.delete_virtual_model(copy)
    assert deleted.value.code == "unsupported"
    assert state.db.get_model_settings(copy) is None, "no settings row was written"
    with pytest.raises(BadRequestError) as persisted:
        await state.manager.load_recommended(copy, 4096, persist=True)
    assert persisted.value.param == "persist"
    assert state.registry.get(copy) is not None, "a refused persist drops nothing"


# ---------------------------------------------------------------------------
# Review fixes (2026-11-14): one regression test per finding
# ---------------------------------------------------------------------------


def test_review1_a_same_family_label_never_borrows_the_key() -> None:
    """B with NO lease of its own and A's key: its label still says it is not A."""
    from studioforge.core.instances import label_vouches, other_clients

    a = GpuLease(id="la", devices=[2, 3], holder=A, created_at=1, last_activity_at=1)
    a.holder_key = KEY
    as_b = Caller(client=B, key=KEY)
    assert names_other_holder(as_b, a, [a]), "same family, different holder: someone else"
    assert not claims(as_b, a, [a]), "the shared key is not identity"
    # Exactly the family root still vouches (crucibleforge for crucibleforge-judge).
    judge = GpuLease(
        id="lj", devices=[0], holder="crucibleforge-judge", created_at=1, last_activity_at=1
    )
    judge.holder_key = KEY
    assert label_vouches("crucibleforge", judge)
    assert claims(Caller(client="crucibleforge", key=KEY), judge, [judge])
    assert not label_vouches("crucibleforge-judge-b", judge)
    # A peer address never makes an unlabeled request the caller's own.
    instance = InstanceInfo(model_id=MODEL, state="ready")
    streams(instance, "127.0.0.1")
    assert other_clients(instance, Caller(key=KEY, peer="127.0.0.1")) == ["127.0.0.1"]
    assert other_clients(instance, Caller(client=B, peer="127.0.0.1")) == ["127.0.0.1"]
    streams(instance, B)
    assert other_clients(instance, Caller(client=B, peer="127.0.0.1")) == []


async def test_review1_b_without_a_lease_cannot_cut_as_unlabeled_streams() -> None:
    """The key-only caller that resolves to A's model meets 409 instance_in_use."""
    manager, supervisor = manager_with()
    lease = await manager.grant_lease([2, 3], holder=A, model_ids=[MODEL], holder_key=KEY)
    resident = await manager.load(MODEL, lease_id=lease.lease.id)
    streams(resident, "127.0.0.1")  # the holder's requests, unlabeled: the peer is all they carry
    key_only = Caller(key=KEY, peer="127.0.0.1")
    with pytest.raises(StudioForgeError) as busy:
        await manager.unload(MODEL, caller=key_only, lease_ok=True)
    assert busy.value.code == "instance_in_use"
    assert supervisor.get(MODEL) is resident and resident.active_requests == 3


async def test_review2_concurrent_grants_never_share_an_instance_id() -> None:
    """Two grants planned before either made its copy (an eviction awaits between)."""
    import asyncio

    manager, supervisor = manager_with()
    other = manager.registry.get(MODEL).model_copy(update={"id": "pub/other", "name": "pub/other"})
    manager.registry._records[other.id] = other  # noqa: SLF001
    await manager.load(other.id, devices=[0, 1])
    await manager.load(MODEL, devices=[2, 3])

    async def slow_settle() -> None:
        await asyncio.sleep(0.05)

    manager._settle_vram = slow_settle  # type: ignore[method-assign]  # noqa: SLF001
    first, second = await asyncio.gather(
        manager.grant_lease([0, 1], holder="a", instances=[{"model": MODEL, "load": False}]),
        manager.grant_lease([2, 3], holder="b", instances=[{"model": MODEL, "load": False}]),
    )
    ids = {first.instances[0]["id"], second.instances[0]["id"]}
    assert ids == {f"{MODEL}:2", f"{MODEL}:3"}, "each grant has its own id"
    for grant in (first, second):
        copy = grant.instances[0]["id"]
        assert manager.registry.get(copy).lease_id == grant.lease.id
        assert [spec.id for spec in grant.lease.instances] == [copy]
    assert supervisor.get(other.id) is None and supervisor.get(MODEL) is None, "both evicted"


async def test_review2_the_book_refuses_another_leases_instance_id() -> None:
    from studioforge.types import LeaseInstance

    manager, _ = manager_with()
    grant = await manager.grant_lease([0], holder="a", instances=[{"model": MODEL, "load": False}])
    copy = grant.instances[0]["id"]
    rival = manager.leases.acquire([1], holder="b")
    with pytest.raises(StudioForgeError) as booked:
        manager.leases.attach_instances(
            rival.id, [LeaseInstance(id=copy, base_model_id=MODEL, devices=[1])]
        )
    assert booked.value.code == "instance_id_conflict"
    assert manager.leases.instance(copy)[0].id == grant.lease.id


async def test_review2_a_tag_taken_between_plan_and_make_is_a_409(state: Any) -> None:
    """The tag re-check inside the make, and the real registry's refusal to replace."""
    from studioforge.core.instances import ephemeral_record

    record = state.registry.get(TINY)
    clash = ephemeral_record(
        record, instance_id=f"{TINY}@t", lease_id="other", devices=[0], launch={}
    )
    state.registry.add_ephemeral(clash)
    stranger = ephemeral_record(
        record, instance_id=f"{TINY}@t", lease_id="mine", devices=[1], launch={}
    )
    with pytest.raises(BadRequestError) as refused:
        state.registry.add_ephemeral(stranger)
    assert refused.value.code == "instance_id_conflict"
    lease = state.manager.leases.acquire([1], holder="b")
    with pytest.raises(StudioForgeError) as made:
        state.manager._make_instance(  # noqa: SLF001
            lease,
            record,
            instance_id=f"{TINY}@t",
            devices=[1],
            launch={},
            tag="t",
            load=False,
            origin="requested",
        )
    assert made.value.code == "instance_id_conflict"
    assert state.registry.get(f"{TINY}@t").lease_id == "other"


async def test_review3_a_forced_bound_load_into_another_holders_lease_is_refused() -> None:
    """Defence in depth behind the route's 403: force on a lease's copy needs its holder."""
    manager, supervisor = manager_with()
    lease_a = await manager.grant_lease(
        [2, 3],
        holder=A,
        holder_key=KEY,
        instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}],
    )
    copy = lease_a.instances[0]["id"]
    streams(supervisor.get(copy), A)
    with pytest.raises(LeaseConflictError) as refused:
        await manager.load(
            MODEL, lease_id=lease_a.lease.id, force=True, caller=Caller(client=B, key=KEY)
        )
    assert refused.value.code == "instance_held"
    assert supervisor.get(copy).active_requests == 3


async def test_review5_another_holders_copy_cannot_be_reshaped_without_admin() -> None:
    manager, _ = manager_with()
    lease_a = await manager.grant_lease(
        [2, 3],
        holder=A,
        holder_key=KEY,
        instances=[{"model": MODEL, "launch": {"ctx_size": 16384}}],
    )
    copy = lease_a.instances[0]["id"]
    stored = dict(lease_a.lease.instances[0].loaded_spec)
    as_b = Caller(client=B, key=KEY)
    with pytest.raises(LeaseConflictError) as shaped:
        await manager.load(copy, ctx_size=8192, caller=as_b)
    assert shaped.value.code == "instance_held"
    with pytest.raises(LeaseConflictError):
        await manager.load_recommended(copy, 2048, caller=as_b)
    assert lease_a.lease.instances[0].loaded_spec == stored
    # An admin may reshape it -- and the holder's restore spec stays the holder's.
    reshaped = await manager.load(copy, ctx_size=8192, force=True, caller=Caller(admin=True))
    assert reshaped.plan.ctx_per_slot == 8192
    assert lease_a.lease.instances[0].loaded_spec == stored, "only the owner's loads are stored"
    # The holder's own reshape is stored.
    await manager.load(copy, ctx_size=4096, force=True, caller=Caller(client=A, key=KEY))
    assert lease_a.lease.instances[0].loaded_spec["ctx_per_slot"] == 4096


async def test_review6_a_keyed_bound_load_without_a_label_gets_its_own_copy() -> None:
    """The bound lease is the caller's identity: the other lease is another holder's."""
    manager, supervisor = manager_with()
    lease_a = await manager.grant_lease([2, 3], holder=A, model_ids=[MODEL], holder_key=KEY)
    resident = await manager.load(MODEL, lease_id=lease_a.lease.id)
    lease_b = await manager.grant_lease([0, 1], holder=B, holder_key=KEY)
    key_only = Caller(lease_ids=(lease_b.lease.id,), key=KEY)
    instance = await manager.load(MODEL, lease_id=lease_b.lease.id, ctx_size=4096, caller=key_only)
    assert instance.model_id == f"{MODEL}:2" and sorted(instance.plan.devices) == [0, 1]
    assert supervisor.get(MODEL) is resident


async def test_review7_a_load_that_outlived_its_lease_is_stopped_and_410() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease([0, 1], holder=A, instances=[{"model": MODEL, "load": False}])
    copy = grant.instances[0]["id"]
    real_start = supervisor.start

    async def start_while_released(record: Any, plan: Any, **kwargs: Any) -> InstanceInfo:
        # The lease ends while this load waits for its child.
        await manager.release_and_unload(grant.lease.id)
        return await real_start(record, plan, **kwargs)

    supervisor.start = start_while_released  # type: ignore[method-assign]
    with pytest.raises(StudioForgeError) as gone:
        await manager.load(copy, ctx_size=4096)
    assert gone.value.code == "instance_released" and gone.value.status_code == 410
    assert supervisor.get(copy) is None, "the orphan child was stopped"


def test_review8_a_torn_instances_column_keeps_the_lease(tmp_path: Any) -> None:
    from studioforge.db import Database

    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    try:
        lease = GpuLease(id="l1", devices=[0], holder=A, created_at=1.0, last_activity_at=1.0)
        db.save_lease(lease_row(lease))
        connection = db.connect()
        connection.execute("UPDATE gpu_leases SET instances_json = '{not json' WHERE id = 'l1'")
        connection.commit()
        (row,) = db.list_leases()
        assert row["id"] == "l1" and row["instances"] == []
        assert lease_from_row(row).devices == [0]
    finally:
        db.close()


async def test_review_restored_numbers_stay_below_new_ones() -> None:
    """After a restart, a new copy numbers above every restored id."""
    store = MemoryStore()
    manager, _ = manager_with()
    manager.leases = LeaseBook(store=store)
    manager.planner.leases = manager.leases
    await manager.grant_lease(
        [0], holder=A, instances=[{"model": MODEL, "count": 3, "load": False}]
    )
    reborn, _ = manager_with()
    reborn.leases = LeaseBook(store=store)
    reborn.planner.leases = reborn.leases
    reborn._restore_leases()  # noqa: SLF001
    fresh = await reborn.grant_lease([1], holder=B, instances=[{"model": MODEL, "load": False}])
    assert fresh.instances[0]["id"] == f"{MODEL}:5"


async def test_review4_mcp_load_by_a_non_admin_key_cannot_take_another_holders_copy(
    state: Any,
) -> None:
    from studioforge import mcp_policy
    from studioforge.authkeys import Principal

    grant = await state.manager.grant_lease(
        [0], holder="mb-a", holder_key="key:ka", instances=[{"model": TINY, "load": False}]
    )
    copy = grant.instances[0]["id"]
    server = build_management_mcp(state)
    token = mcp_policy._CALLER.set(Principal("key", "kb", "bench-b", "inference"))  # noqa: SLF001
    try:
        refused = await call(server, "load_model", model_id=copy, force=True)
        shaped = await call(server, "load_recommended", model_id=copy, ctx_size=4096)
    finally:
        mcp_policy._CALLER.reset(token)  # noqa: SLF001
    assert refused["ok"] is False and refused["error"]["code"] == "instance_held", refused
    assert shaped["ok"] is False and shaped["error"]["code"] == "instance_held", shaped


# ---------------------------------------------------------------------------
# Re-review fixes (2026-11-14): R1-R3
# ---------------------------------------------------------------------------


def test_r1_the_label_a_lease_was_taken_with_vouches_for_its_holder() -> None:
    """A holder leasing as ``openclaw`` that labels its requests ``openclaw-main``."""
    from studioforge.core.instances import label_vouches

    lease = GpuLease(id="lo", devices=[0], holder="openclaw", created_at=1.0, last_activity_at=1.0)
    lease.holder_key = KEY
    lease.holder_label = "openclaw-main"
    me = Caller(client="openclaw-main", key=KEY)
    assert label_vouches("openclaw-main", lease)
    assert claims(me, lease, [lease]) and not names_other_holder(me, lease, [lease])
    # Persisted with the lease; an older row (no label recorded) keeps today's rule.
    back = lease_from_row(lease_row(lease))
    assert back.holder_label == "openclaw-main"
    old = {**lease_row(lease)}
    old.pop("holder_label")
    legacy = lease_from_row(old)
    assert legacy.holder_label is None
    assert names_other_holder(me, legacy, [legacy]), "an unrecorded label does not vouch"
    assert claims(Caller(key=KEY), legacy, [legacy]), "the key alone still works as before"
    assert "holder_label" not in lease_view(lease), "never shown, like holder_peer"


async def test_r1_its_own_key_and_label_route_and_unload_its_own_copy() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1],
        holder="openclaw",
        holder_key=KEY,
        holder_label="openclaw-main",
        instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}],
    )
    copy = grant.instances[0]["id"]
    me = Caller(client="openclaw-main", key=KEY)
    assert manager.route_for(manager.registry.get(MODEL), me).id == copy
    assert manager.unload_target(MODEL, me) == copy
    assert await manager.unload(MODEL, caller=me, lease_ok=True) is True
    assert supervisor.get(copy) is None


def test_r2_an_exact_label_wins_over_the_family_root() -> None:
    cf = GpuLease(id="lc", devices=[0], holder="crucibleforge", created_at=1, last_activity_at=1)
    judge = GpuLease(
        id="lj", devices=[1], holder="crucibleforge-judge", created_at=1, last_activity_at=1
    )
    cf.holder_key = judge.holder_key = KEY
    both = [cf, judge]
    run = Caller(client="crucibleforge", key=KEY)
    assert claims(run, cf, both) and not claims(run, judge, both)
    assert not names_other_holder(run, cf, both) and names_other_holder(run, judge, both)
    # With only the judge's lease standing, the family root still vouches (D55).
    assert claims(run, judge, [judge]) and not names_other_holder(run, judge, [judge])


async def test_r2_routing_and_unload_follow_the_exact_match() -> None:
    manager, _ = manager_with()
    run = await manager.grant_lease(
        [0, 1],
        holder="crucibleforge",
        holder_key=KEY,
        instances=[{"model": MODEL, "load": False}],
    )
    await manager.grant_lease(
        [2, 3],
        holder="crucibleforge-judge",
        holder_key=KEY,
        instances=[{"model": MODEL, "load": False}],
    )
    caller = Caller(client="crucibleforge", key=KEY)
    mine = run.instances[0]["id"]
    assert manager.route_for(manager.registry.get(MODEL), caller).id == mine
    assert manager.unload_target(MODEL, caller) == mine, "not instance_ambiguous"


async def test_r3_a_forced_reload_never_cuts_requests_someone_else_labelled() -> None:
    manager, supervisor = manager_with()
    grant = await manager.grant_lease(
        [0, 1],
        holder=A,
        holder_key=KEY,
        instances=[{"model": MODEL, "launch": {"ctx_size": 4096}}],
    )
    copy = grant.instances[0]["id"]
    me = Caller(client=A, key=KEY)
    streams(supervisor.get(copy), "chat-app")
    with pytest.raises(StudioForgeError) as busy:
        await manager.load(copy, ctx_size=4096, force=True, caller=me)
    assert busy.value.code == "instance_in_use"
    assert busy.value.details["in_flight_clients"] == ["chat-app"]
    with pytest.raises(StudioForgeError):
        await manager.load(MODEL, lease_id=grant.lease.id, force=True, caller=me)
    # Its own labelled requests, or unlabeled ones (a peer address), do not stop it.
    streams(supervisor.get(copy), A)
    reloaded = await manager.load(copy, ctx_size=4096, force=True, caller=me)
    streams(reloaded, "127.0.0.1")
    reloaded = await manager.load(copy, ctx_size=4096, force=True, caller=me)
    streams(reloaded, "chat-app")
    assert await manager.load(copy, ctx_size=4096, force=True, caller=Caller(admin=True))


async def test_r3_mcp_a_shared_key_must_name_the_copys_own_lease(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge import mcp_policy
    from studioforge.authkeys import Principal

    grant = await state.manager.grant_lease(
        [0], holder="mb-a", holder_key="key:ks", instances=[{"model": TINY, "load": False}]
    )
    copy = grant.instances[0]["id"]
    server = build_management_mcp(state)
    busy = InstanceInfo(model_id=copy, state="ready")
    streams(busy, "mb-a")
    real_get = state.supervisor.get
    monkeypatch.setattr(
        state.supervisor, "get", lambda model_id: busy if model_id == copy else real_get(model_id)
    )
    # The SAME key as the lease's: on MCP it still does not say which holder calls.
    token = mcp_policy._CALLER.set(Principal("key", "ks", "bench", "inference"))  # noqa: SLF001
    try:
        unnamed = await call(server, "load_model", model_id=copy, force=True)
        shaped = await call(server, "load_recommended", model_id=copy, ctx_size=4096)
        named = await call(server, "load_model", model_id=copy, force=True, lease_id=grant.lease.id)
    finally:
        mcp_policy._CALLER.reset(token)  # noqa: SLF001
    assert unnamed["ok"] is False and unnamed["error"]["code"] == "instance_held", unnamed
    assert shaped["ok"] is False and shaped["error"]["code"] == "instance_held", shaped
    assert named["ok"] is False and named["error"]["code"] == "instance_in_use", named


# ---------------------------------------------------------------------------
# Final review (2026-11-14): N1, N2
# ---------------------------------------------------------------------------


def test_n1_the_recorded_label_is_stored_and_restored_in_the_compared_form() -> None:
    book = LeaseBook()
    lease = book.acquire([0], holder="openclaw", holder_label="OpenClaw  (main)")
    assert lease.holder_label == "OpenClaw main", "the form caller_of compares in"
    raw = {**lease_row(lease), "holder_label": "OpenClaw  (main)" + "x" * 80}
    restored = lease_from_row(raw)
    assert restored.holder_label is not None and len(restored.holder_label) == 64
    assert restored.holder_label.startswith("OpenClaw main")


def _mcp_principal(key_id: str) -> Any:
    from studioforge.authkeys import Principal

    return Principal("key", key_id, "tenant", "inference")


async def test_n2_mcp_a_tenants_own_base_model_lease_loads_as_before(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Old model_ids shape, no copy anywhere: the tenant's key is its identity, as pre-D91."""
    from studioforge import mcp_policy
    from studioforge.types import LoadPlan

    state.manager.leases.acquire([0], holder="tenant", model_ids=[TINY], holder_key="key:kt")
    resident = InstanceInfo(
        model_id=TINY,
        state="ready",
        plan=LoadPlan(model_id=TINY, devices=[0], ctx_size=4096, ctx_per_slot=4096, parallel=1),
    )
    real_get = state.supervisor.get
    monkeypatch.setattr(
        state.supervisor,
        "get",
        lambda model_id: resident if model_id == TINY else real_get(model_id),
    )
    loads: list[str] = []

    async def fake_load(name: str, *_args: Any, **_kwargs: Any) -> InstanceInfo:
        loads.append(name)
        return resident

    monkeypatch.setattr(state.manager, "_load_body", fake_load)
    monkeypatch.setattr(state.manager, "_load_recommended_body", fake_load)
    server = build_management_mcp(state)
    token = mcp_policy._CALLER.set(_mcp_principal("kt"))  # noqa: SLF001
    try:
        walked = await call(server, "load_recommended", model_id=TINY, ctx_size=16384)
        forced = await call(server, "load_model", model_id=TINY, force=True)
    finally:
        mcp_policy._CALLER.reset(token)  # noqa: SLF001
    assert walked["ok"] is True, walked
    assert forced["ok"] is True, forced
    assert loads == [TINY, TINY]


async def test_n2_mcp_a_base_id_that_resolves_to_a_copy_keeps_the_copy_rule(
    state: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge import mcp_policy

    grant = await state.manager.grant_lease(
        [0], holder="tenant", holder_key="key:kt", instances=[{"model": TINY, "load": False}]
    )
    copy = grant.instances[0]["id"]
    loads: list[str] = []

    async def fake_load(name: str, *_args: Any, **_kwargs: Any) -> InstanceInfo:
        loads.append(name)
        return InstanceInfo(model_id=name, state="ready")

    monkeypatch.setattr(state.manager, "_load_body", fake_load)
    server = build_management_mcp(state)
    token = mcp_policy._CALLER.set(_mcp_principal("kt"))  # noqa: SLF001
    try:
        unnamed = await call(server, "load_model", model_id=TINY, force=True)
        named = await call(server, "load_model", model_id=TINY, force=True, lease_id=grant.lease.id)
    finally:
        mcp_policy._CALLER.reset(token)  # noqa: SLF001
    assert unnamed["ok"] is False and unnamed["error"]["code"] == "instance_held", unnamed
    assert named["ok"] is True, named
    assert loads == [copy]
