"""Load-priority tiers (D46): candidacy, preemption, the hold, and the restore.

Three tiers -- 1 the active chat model, 2 a dispatched agent, 3 (or
unspecified) background. The promises under test:

* a load may displace only equal-or-worse tiers, and background is displaced
  first (a background load can never touch the chat model);
* a tier-1/2 load takes the *fastest* placement, displacing idle worse-tier
  residents from the cards it picks even when a lesser placement would have
  fitted beside them;
* while a tier-1/2 load is in flight, new traffic for worse-tier models is
  refused with a 503 + Retry-After, and the load jumps the D29 gate queue;
* displaced models that were recently active are reloaded afterwards where
  they still fit -- with no eviction licence of their own -- and idle ones
  stay unloaded.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core.manager import ModelManager, _RestoreEntry
from studioforge.core.planner import Planner
from studioforge.core.priority import (
    PRIORITY_AGENT,
    PRIORITY_BACKGROUND,
    PRIORITY_CHAT,
    PriorityLock,
    normalise_priority,
)
from studioforge.errors import BadRequestError, ModelBusyError
from studioforge.types import GB, InstanceInfo, LoadPlan, LoadRejected
from tests.unit.test_load_retry import (
    StubPlanner,
    StubProbe,
    StubRegistry,
    StubSupervisor,
    make_record,
)
from tests.unit.test_planner import StubProbe as PlannerProbe
from tests.unit.test_planner import (
    gpu,
    loaded_instance,
    make_config,
    make_meta,
)
from tests.unit.test_planner import make_record as make_planner_record

# ---------------------------------------------------------------------------
# normalise_priority
# ---------------------------------------------------------------------------


def test_normalise_accepts_the_three_tiers_and_none() -> None:
    assert normalise_priority(None) is None
    assert normalise_priority(1) == PRIORITY_CHAT
    assert normalise_priority(2) == PRIORITY_AGENT
    assert normalise_priority(3) == PRIORITY_BACKGROUND


@pytest.mark.parametrize("bad", [0, 4, -1, "1", 1.0, True, False])
def test_normalise_refuses_anything_else_naming_the_parameter(bad: Any) -> None:
    """``True`` especially: it is an ``int`` in Python and would silently
    become the chat tier, which is the one tier that must never be reached by
    accident."""
    with pytest.raises(BadRequestError) as excinfo:
        normalise_priority(bad)
    assert excinfo.value.param == "priority"


# ---------------------------------------------------------------------------
# PriorityLock
# ---------------------------------------------------------------------------


async def _spin(rounds: int = 10) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


async def test_the_lock_serves_the_best_tier_first_fifo_within_one() -> None:
    lock = PriorityLock()
    await lock.acquire()
    order: list[str] = []

    async def waiter(name: str, priority: int) -> None:
        await lock.acquire(priority)
        order.append(name)
        lock.release()

    # Arrival order: two background, then a chat and an agent load.
    tasks = [asyncio.create_task(waiter("bg-a", 3))]
    await _spin()
    tasks.append(asyncio.create_task(waiter("bg-b", 3)))
    await _spin()
    tasks.append(asyncio.create_task(waiter("chat", 1)))
    await _spin()
    tasks.append(asyncio.create_task(waiter("agent", 2)))
    await _spin()

    lock.release()
    await asyncio.gather(*tasks)
    assert order == ["chat", "agent", "bg-a", "bg-b"]
    assert lock.locked() is False


async def test_a_cancelled_waiter_does_not_wedge_the_lock() -> None:
    lock = PriorityLock()
    await lock.acquire()
    waiter = asyncio.create_task(lock.acquire(1))
    await _spin()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    lock.release()
    assert lock.locked() is False
    await lock.acquire()  # still usable
    lock.release()


# ---------------------------------------------------------------------------
# Planner: tier candidacy and ordering
# ---------------------------------------------------------------------------


def tiered_instance(
    model_id: str,
    *,
    device: int,
    bytes_held: int,
    priority: int,
    last_activity: float = 100.0,
    active: int = 0,
    ttl_s: int | None = 1800,
) -> InstanceInfo:
    inst = loaded_instance(
        model_id,
        device=device,
        bytes_held=bytes_held,
        last_activity=last_activity,
        active=active,
        ttl_s=ttl_s,
    )
    inst.priority = priority
    return inst


def test_evictable_filters_and_orders_by_tier_then_lru() -> None:
    planner = Planner(make_config(), StubProbe())
    chat = tiered_instance("chat", device=0, bytes_held=GB, priority=1, last_activity=1.0)
    agent = tiered_instance("agent", device=0, bytes_held=GB, priority=2, last_activity=50.0)
    bg = tiered_instance("bg", device=0, bytes_held=GB, priority=3, last_activity=999.0)
    loaded = [chat, bg, agent]

    # Worst tier first, LRU within one; the asking tier bounds candidacy.
    assert [i.model_id for i in planner._evictable(loaded, for_priority=1)] == [
        "bg",
        "agent",
        "chat",
    ]
    assert [i.model_id for i in planner._evictable(loaded, for_priority=2)] == ["bg", "agent"]
    assert [i.model_id for i in planner._evictable(loaded, for_priority=3)] == ["bg"]
    # The pre-D46 informational surfaces apply no tier rule.
    assert len(planner._evictable(loaded)) == 3


def test_a_background_load_cannot_displace_the_chat_model() -> None:
    """Priority 3 takes a backseat to everything, including when evicting."""
    probe = PlannerProbe([gpu(0, 32.0, 4.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    chat = tiered_instance("chat/model", device=0, bytes_held=int(20 * GB), priority=1)
    record = make_planner_record(meta=make_meta(tensor_bytes=9 * GB), size_bytes=9 * GB)

    rejected = planner.plan_load(record, ctx_size=4096, loaded=[chat], priority=3)
    assert isinstance(rejected, LoadRejected)

    # An agent load may not touch it either...
    rejected = planner.plan_load(record, ctx_size=4096, loaded=[chat], priority=2)
    assert isinstance(rejected, LoadRejected)

    # ...but another chat-tier load may (an idle equal tier is fair game).
    plan = planner.plan_load(record, ctx_size=4096, loaded=[chat], priority=1)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == ["chat/model"]


def test_a_chat_load_takes_the_fast_card_and_displaces_background() -> None:
    """The max-T/S rule: a tier-1 load plans as if idle lower tiers were gone.

    The 5090 holds an idle background model; the 3090 is free. A background
    load squeezes onto the 3090 without touching anyone. The chat load takes
    the 5090 -- the background model on it is displaced, and the plan says so.
    """
    probe = PlannerProbe([gpu(0, 32.0, 10.0, (12, 0)), gpu(1, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    bg = tiered_instance(
        "bg/model", device=0, bytes_held=int(20 * GB), priority=3, last_activity=time.time()
    )
    record = make_planner_record(meta=make_meta(tensor_bytes=12 * GB), size_bytes=12 * GB)

    background_plan = planner.plan_load(record, ctx_size=4096, loaded=[bg], priority=3)
    assert isinstance(background_plan, LoadPlan)
    assert background_plan.devices == [1]
    assert background_plan.evict_model_ids == []

    chat_plan = planner.plan_load(record, ctx_size=4096, loaded=[bg], priority=1)
    assert isinstance(chat_plan, LoadPlan)
    assert chat_plan.devices == [0]
    assert chat_plan.evict_model_ids == ["bg/model"]
    assert any("displacing lower-priority" in n for n in chat_plan.notes)


def test_preemption_leaves_models_off_the_chosen_cards_alone() -> None:
    """Displacement is per card, not per tier: a background model on a card
    the chat load does not want stays loaded."""
    probe = PlannerProbe([gpu(0, 32.0, 30.0, (12, 0)), gpu(1, 24.0, 3.5, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    bg = tiered_instance("bg/model", device=1, bytes_held=int(20 * GB), priority=3)
    record = make_planner_record(meta=make_meta(tensor_bytes=12 * GB), size_bytes=12 * GB)

    plan = planner.plan_load(record, ctx_size=4096, loaded=[bg], priority=1)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]
    assert plan.evict_model_ids == []


def test_preemption_never_displaces_a_busy_model_even_under_force() -> None:
    """Preemption is for a nicer placement; only the eviction ladder, under a
    genuine shortfall, may ever touch a busy model (D36). A forced tier-1
    load that fits elsewhere leaves the mid-stream background model alone."""
    probe = PlannerProbe([gpu(0, 32.0, 10.0, (12, 0)), gpu(1, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    busy_bg = tiered_instance("bg/model", device=0, bytes_held=int(20 * GB), priority=3, active=1)
    record = make_planner_record(meta=make_meta(tensor_bytes=12 * GB), size_bytes=12 * GB)

    plan = planner.plan_load(record, ctx_size=4096, loaded=[busy_bg], priority=1, evict_busy=True)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1]
    assert plan.evict_model_ids == []


def test_a_no_evict_priority_load_preempts_nothing() -> None:
    """The restore pass reloads with ``allow_evict=False``; that licence must
    also switch the preemption credit off, or a restore could displace in turn."""
    probe = PlannerProbe([gpu(0, 32.0, 10.0, (12, 0)), gpu(1, 24.0, 23.5, (8, 6))])
    planner = Planner(make_config(headroom_fraction=0.0, on_insufficient="evict"), probe)
    bg = tiered_instance("bg/model", device=0, bytes_held=int(20 * GB), priority=3)
    record = make_planner_record(meta=make_meta(tensor_bytes=12 * GB), size_bytes=12 * GB)

    plan = planner.plan_load(record, ctx_size=4096, loaded=[bg], priority=1, allow_evict=False)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [1]
    assert plan.evict_model_ids == []


# ---------------------------------------------------------------------------
# Manager: memo, validation, the hold, the gate, the restore
# ---------------------------------------------------------------------------


def multi_manager(
    supervisor: Any, planner: Any, ids: tuple[str, ...] = ("test/model",)
) -> ModelManager:
    records = {mid: make_record(mid) for mid in ids}
    return ModelManager(
        Config(data_dir="/tmp/sf-priority"),
        registry=StubRegistry(records),  # type: ignore[arg-type]
        planner=planner,  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )


async def test_an_explicit_tier_is_remembered_for_the_jit_reload() -> None:
    """The chat model stays the chat model across a TTL unload and its JIT
    reload -- the tier is per model, not per call."""
    supervisor = StubSupervisor()
    manager = multi_manager(supervisor, StubPlanner())

    instance = await manager.load("test/model", priority=2)
    assert instance.priority == 2

    await manager.unload("test/model")
    _, reloaded = await manager.ensure_loaded("test/model")
    assert reloaded.priority == 2


async def test_an_unspecified_priority_is_background() -> None:
    supervisor = StubSupervisor()
    manager = multi_manager(supervisor, StubPlanner())
    instance = await manager.load("test/model")
    assert instance.priority == PRIORITY_BACKGROUND
    assert manager.busy_snapshot()["priority_hold"] is None


async def test_an_invalid_priority_is_a_400() -> None:
    manager = multi_manager(StubSupervisor(), StubPlanner())
    with pytest.raises(BadRequestError):
        await manager.load("test/model", priority=0)


async def test_promoting_a_running_model_re_tiers_the_live_instance() -> None:
    """``load(priority=1)`` on a ready model must reach the instance the
    planner's candidacy reads -- a memo the instance contradicts is a silent
    no-op, and the "chat" model would stay evictable as background."""
    supervisor = StubSupervisor()
    manager = multi_manager(supervisor, StubPlanner())
    instance = await manager.load("test/model")
    assert instance.priority == PRIORITY_BACKGROUND

    promoted = await manager.load("test/model", priority=1)
    assert promoted is instance
    assert promoted.priority == 1
    assert manager._effective_priority("test/model") == 1


async def test_a_refused_load_does_not_rewrite_the_remembered_tier() -> None:
    """A load the hold 503s must not re-tier the model as a side effect."""
    manager = multi_manager(StubSupervisor(), StubPlanner(), ids=("chat/model", "test/model"))
    manager._model_priority["test/model"] = 1
    manager._priority_holds["chat/model"] = 1
    with pytest.raises(ModelBusyError):
        await manager.load("test/model", priority=3)
    assert manager._model_priority["test/model"] == 1


class GatedSupervisor(StubSupervisor):
    """start() blocks until released, so a load can be held in flight."""

    def __init__(self) -> None:
        super().__init__()
        self.proceed = asyncio.Event()
        self.entered = asyncio.Event()
        self.order: list[str] = []

    async def start(self, record: Any, plan: Any, **kwargs: Any) -> InstanceInfo:
        self.order.append(record.id)
        self.entered.set()
        await self.proceed.wait()
        return await super().start(record, plan, **kwargs)


async def test_a_chat_load_in_flight_holds_worse_tier_traffic_off() -> None:
    supervisor = GatedSupervisor()
    manager = multi_manager(supervisor, StubPlanner(), ids=("chat/model", "bg/model"))

    chat_load = asyncio.create_task(manager.load("chat/model", priority=1))
    await supervisor.entered.wait()

    hold = manager.busy_snapshot()["priority_hold"]
    assert hold == {"model_id": "chat/model", "priority": 1}

    # New inference for a background model is refused, with the reason.
    with pytest.raises(ModelBusyError) as excinfo:
        await manager.ensure_loaded("bg/model")
    assert excinfo.value.details["priority_hold"] == {"model_id": "chat/model", "priority": 1}
    assert excinfo.value.details["retry_after_s"] > 0

    # A new background load is refused the same way...
    with pytest.raises(ModelBusyError):
        await manager.load("bg/model")

    # ...and the hold lifts the moment the chat model is serving.
    supervisor.proceed.set()
    await chat_load
    assert manager.busy_snapshot()["priority_hold"] is None
    _, instance = await manager.ensure_loaded("bg/model")
    assert instance.model_id == "bg/model"


async def test_a_chat_load_jumps_the_gate_queue() -> None:
    """D29 still serialises; D46 reorders the queue: the chat load queued
    *after* a background load starts first once the gate frees."""
    supervisor = StubSupervisor()
    manager = multi_manager(supervisor, StubPlanner(), ids=("chat/model", "bg/model"))

    await manager._load_gate.acquire()  # somebody's load is mid-flight
    bg_load = asyncio.create_task(manager.load("bg/model"))
    await _spin()
    chat_load = asyncio.create_task(manager.load("chat/model", priority=1))
    await _spin()

    manager._load_gate.release()
    await asyncio.gather(bg_load, chat_load)
    assert list(supervisor.instances) == ["chat/model", "bg/model"]


class QueuePlanner(StubPlanner):
    """Returns queued results first, then StubPlanner's default plan."""

    def __init__(self, results: list[Any]) -> None:
        super().__init__()
        self.queued = list(results)

    def plan_load(self, record: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.seen.append(record)
        self.kwargs.append(kwargs)
        if self.queued:
            return self.queued.pop(0)
        return LoadPlan(model_id=record.id, devices=[0], ctx_size=8192)


def resident_victim(model_id: str, *, last_activity: float, priority: int = 3) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18200,
        ttl_s=1800,
        last_activity_at=last_activity,
        priority=priority,
        plan=LoadPlan(
            model_id=model_id,
            devices=[0],
            ctx_size=4096,
            ctx_per_slot=4096,
            kv_cache_type="q8_0",
            kv_cache_type_v="q8_0",
            parallel=2,
            per_gpu_bytes={0: int(4 * GB)},
        ),
    )


async def test_recently_active_victims_are_restored_and_idle_ones_are_not() -> None:
    now = time.time()
    supervisor = StubSupervisor()
    supervisor.instances["active/victim"] = resident_victim("active/victim", last_activity=now)
    supervisor.instances["idle/victim"] = resident_victim("idle/victim", last_activity=now - 3600.0)
    planner = QueuePlanner(
        [
            LoadPlan(
                model_id="chat/model",
                devices=[0],
                ctx_size=8192,
                evict_model_ids=["active/victim", "idle/victim"],
            )
        ]
    )
    manager = multi_manager(supervisor, planner, ids=("chat/model", "active/victim", "idle/victim"))

    await manager.load("chat/model", priority=1)
    assert supervisor.stopped == ["active/victim", "idle/victim"]
    assert manager._restore_task is not None
    await manager._restore_task

    # The active victim is back, with the settings tuple it ran with before,
    # under a no-evict plan; the idle one stays unloaded.
    restored = supervisor.get("active/victim")
    assert restored is not None
    assert supervisor.get("idle/victim") is None
    restore_kwargs = planner.kwargs[-1]
    assert restore_kwargs["ctx_size"] == 4096
    assert restore_kwargs["parallel"] == 2
    assert restore_kwargs["allow_evict"] is False
    assert restore_kwargs["priority"] == 3


async def test_a_victim_that_no_longer_fits_stays_unloaded() -> None:
    now = time.time()
    supervisor = StubSupervisor()
    supervisor.instances["active/victim"] = resident_victim("active/victim", last_activity=now)
    planner = QueuePlanner(
        [
            LoadPlan(
                model_id="chat/model",
                devices=[0],
                ctx_size=8192,
                evict_model_ids=["active/victim"],
            ),
            LoadRejected(model_id="active/victim", reason="no room left", required_bytes=1),
        ]
    )
    manager = multi_manager(supervisor, planner, ids=("chat/model", "active/victim"))

    await manager.load("chat/model", priority=1)
    assert manager._restore_task is not None
    await manager._restore_task

    assert supervisor.get("active/victim") is None
    assert manager._restore_entries == {}


async def test_a_background_load_restores_nothing() -> None:
    """Background-vs-background eviction is the pre-D46 LRU world: no restore."""
    now = time.time()
    supervisor = StubSupervisor()
    supervisor.instances["active/victim"] = resident_victim("active/victim", last_activity=now)
    planner = QueuePlanner(
        [
            LoadPlan(
                model_id="other/model",
                devices=[0],
                ctx_size=8192,
                evict_model_ids=["active/victim"],
            )
        ]
    )
    manager = multi_manager(supervisor, planner, ids=("other/model", "active/victim"))

    await manager.load("other/model")
    assert manager._restore_entries == {}
    assert manager._restore_task is None
    assert supervisor.get("active/victim") is None


async def test_the_transient_oom_retry_stops_the_replans_own_victims() -> None:
    """The retry's replan may carry victims of its own -- routinely, for a
    tier load whose preemption credit lists lower-tier residents. Launching
    without stopping them spawns the child into VRAM the plan only imagined
    free; and both retry victims get the same restore promise (D46)."""
    now = time.time()
    supervisor = StubSupervisor(
        fail_times=1,
        stderr=["CUDA error: out of memory", "failed to allocate buffer"],
    )
    supervisor.instances["bg/a"] = resident_victim("bg/a", last_activity=now)
    supervisor.instances["bg/b"] = resident_victim("bg/b", last_activity=now)
    planner = QueuePlanner(
        [
            LoadPlan(model_id="chat/model", devices=[0], ctx_size=8192),
            LoadPlan(model_id="chat/model", devices=[0], ctx_size=8192, evict_model_ids=["bg/b"]),
        ]
    )
    manager = multi_manager(supervisor, planner, ids=("chat/model", "bg/a", "bg/b"))

    await manager.load("chat/model", priority=1)
    # The first victim came from the LRU retry eviction, the second from the
    # replan's own evict list; both were stopped before the second spawn.
    assert supervisor.stopped == ["bg/a", "bg/b"]
    assert supervisor.get("chat/model") is not None
    assert manager._restore_task is not None
    await manager._restore_task
    assert supervisor.get("bg/a") is not None
    assert supervisor.get("bg/b") is not None


async def test_a_deliberate_unload_cancels_a_pending_restore() -> None:
    manager = multi_manager(StubSupervisor(), StubPlanner())
    manager._restore_entries["test/model"] = _RestoreEntry(
        model_id="test/model",
        priority=3,
        ctx_size=4096,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        parallel=1,
        evicted_at=time.time(),
    )
    await manager.unload("test/model")
    assert manager._restore_entries == {}
