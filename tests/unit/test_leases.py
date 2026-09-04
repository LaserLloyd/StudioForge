"""GPU leases (D43): a card that belongs to one model until released or idle.

Three layers, each pinned separately: the book's bookkeeping, the planner's
refusal to place anyone else on a leased card (and the owner's right to it,
with auto-sized slots), and the manager's grant -- which unloads idle
neighbours, never a serving one, and a pinned one only when forced.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from studioforge.core.leases import (
    LEASE_IDLE_AFTER_S,
    LEASE_RETRY_CAP_S,
    LeaseBook,
    holder_family,
    lease_kind,
    lease_state,
    lease_view,
)
from studioforge.core.planner import Planner
from studioforge.errors import (
    BadRequestError,
    LeaseConflictError,
    LeaseNotFoundError,
    ModelBusyError,
)
from studioforge.types import LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_gateway_lifecycle import (
    CountingSupervisor,
    make_manager,
    make_record,
    placed,
)
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

# ---------------------------------------------------------------------------
# The book
# ---------------------------------------------------------------------------


def test_acquire_release_round_trip_and_views() -> None:
    book = LeaseBook()
    lease = book.acquire([1, 0], holder="api", model_ids=["a/model", "a/model"], reason="warm")

    assert lease.devices == [0, 1], "devices are sorted and deduplicated"
    assert lease.model_ids == ["a/model"]
    assert book.get(lease.id) is lease
    assert len(book) == 1
    view = lease_view(lease)
    assert view["id"] == lease.id and view["idle_s"] == 0 and view["expires_at"] is not None

    released = book.release(lease.id)
    assert released is lease and len(book) == 0
    with pytest.raises(LeaseNotFoundError):
        book.release(lease.id)


def test_a_card_is_leased_once() -> None:
    book = LeaseBook()
    first = book.acquire([0], holder="benchmark")
    with pytest.raises(LeaseConflictError) as excinfo:
        book.acquire([0, 1], holder="api")
    assert first.id in excinfo.value.message
    assert book.acquire([1], holder="api").devices == [1], "a free card is still free"


def test_acquire_validates_its_arguments() -> None:
    book = LeaseBook()
    with pytest.raises(BadRequestError):
        book.acquire([], holder="api")
    with pytest.raises(BadRequestError):
        book.acquire([0], holder="api", idle_ttl_s=0)


def test_blocked_for_and_for_model() -> None:
    book = LeaseBook()
    book.acquire([0, 1], holder="api", model_ids=["mine"])
    book.acquire([3], holder="api")  # held for an outsider: nobody may use it

    assert book.blocked_for("mine") == frozenset({3})
    assert book.blocked_for("other") == frozenset({0, 1, 3})
    assert book.blocked_for(None) == frozenset({0, 1, 3})
    assert book.for_model("mine") is not None and book.for_model("other") is None


def test_touch_never_moves_the_clock_backwards_and_expiry_respects_the_ttl() -> None:
    book = LeaseBook()
    start = 1_000_000.0
    lease = book.acquire([2], holder="api", idle_ttl_s=60.0, now=start)
    forever = book.acquire([3], holder="api", idle_ttl_s=None, now=start)

    book.touch(lease.id, at=start - 10)
    assert lease.last_activity_at == start
    book.touch(lease.id, at=start + 30)
    assert lease.last_activity_at == start + 30

    assert book.expired(now=start + 80) == []
    assert [x.id for x in book.expired(now=start + 91)] == [lease.id]
    assert forever.id not in [x.id for x in book.expired(now=start + 10**6)]


# ---------------------------------------------------------------------------
# The derived half of a lease record (D53)
# ---------------------------------------------------------------------------


def test_holder_family_collapses_a_phase_suffix() -> None:
    """The outage this exists for: CrucibleForge leases under two names.

    A consumer matching the holder exactly saw NO lease at all for the whole
    judge phase, because the run holds as ``crucibleforge`` and the judge as
    ``crucibleforge-judge``.
    """
    assert holder_family("crucibleforge") == "crucibleforge"
    assert holder_family("crucibleforge-judge") == "crucibleforge"
    assert holder_family("CrucibleForge-Judge") == "crucibleforge"
    assert holder_family("api") == "api"
    assert holder_family("") == ""


def test_lease_kind_is_descriptive_and_never_policy() -> None:
    book = LeaseBook()
    bench = book.acquire([0], holder="crucibleforge-judge", model_ids=["mine"])
    render = book.acquire([1], holder="clawforge2")
    stranger = book.acquire([2], holder="somebody")

    assert lease_kind(bench.holder) == "benchmark"
    assert lease_kind(render.holder) == "render"
    assert lease_kind(stranger.holder) == "other"
    assert lease_view(bench)["kind"] == "benchmark"

    # The point of "descriptive": placement is unchanged by any of it. Only
    # ``model_ids`` decides who may use a card.
    assert book.blocked_for("mine") == frozenset({1, 2})
    assert book.blocked_for("other") == frozenset({0, 1, 2})


def test_lease_state_moves_through_active_idle_and_expiring() -> None:
    book = LeaseBook()
    start = 1_000_000.0
    lease = book.acquire([0], holder="api", idle_ttl_s=3600.0, now=start)
    forever = book.acquire([1], holder="api", idle_ttl_s=None, now=start)

    assert lease_state(lease, start) == "active"
    assert lease_state(lease, start + LEASE_IDLE_AFTER_S + 1) == "idle"
    # Expiry outranks idleness: a lease minutes from the sweep is "expiring"
    # whatever its idle clock says.
    assert lease_state(lease, start + 3600.0 - 10) == "expiring"

    # No TTL, no expiry -- so it can go idle but never expiring.
    assert lease_state(forever, start) == "active"
    assert lease_state(forever, start + 10**6) == "idle"


def test_lease_view_retry_advice_is_capped_and_absent_without_a_ttl() -> None:
    book = LeaseBook()
    now = time.time()
    long_lease = book.acquire([0], holder="crucibleforge", idle_ttl_s=7200.0, now=now)
    short = book.acquire([1], holder="api", idle_ttl_s=30.0, now=now)
    forever = book.acquire([2], holder="api", idle_ttl_s=None, now=now)

    # Two hours to run, but the advice is a re-ask interval: an early release
    # is common and a client asleep for two hours would never see one.
    assert lease_view(long_lease)["retry_after_s"] == LEASE_RETRY_CAP_S
    assert 1 <= lease_view(short)["retry_after_s"] <= 30
    assert lease_view(forever)["retry_after_s"] is None
    assert lease_view(forever)["expires_at"] is None


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


def make_planner(book: LeaseBook, **planner_overrides: Any) -> Planner:
    config = make_config(**planner_overrides)
    return Planner(config, rig_5090x2_3090x2(), leases=book, log_plans=False)


def test_planner_never_places_onto_a_card_leased_to_someone_else() -> None:
    book = LeaseBook()
    lease = book.acquire([0, 1], holder="benchmark", model_ids=["other/model"])
    planner = make_planner(book)

    plan = planner.plan_load(make_record("mine/model"))

    assert isinstance(plan, LoadPlan), plan
    assert set(plan.devices) <= {2, 3}, plan.devices
    assert any(lease.id in note for note in plan.notes), "the plan says what it could not use"


def test_the_lease_owner_keeps_its_cards_and_everyone_else_is_told_why() -> None:
    book = LeaseBook()
    lease = book.acquire([0, 1, 2, 3], holder="api", model_ids=["mine/model"])
    planner = make_planner(book)

    mine = planner.plan_load(make_record("mine/model"))
    assert isinstance(mine, LoadPlan), mine

    theirs = planner.plan_load(make_record("other/model"))
    assert isinstance(theirs, LoadRejected), theirs
    assert any(lease.id in line for line in theirs.suggestions)


def test_a_forced_placement_onto_a_leased_card_is_refused_not_honoured() -> None:
    book = LeaseBook()
    lease = book.acquire([0], holder="benchmark", model_ids=["other/model"])
    planner = make_planner(book)
    record = make_record("mine/model", device_override=[0])

    result = planner.plan_load(record)

    assert isinstance(result, LoadRejected), result
    assert "leased" in result.reason
    assert any(lease.id in line for line in result.suggestions)


# ---------------------------------------------------------------------------
# A lease refusal is a promise, not a shortfall (D53)
# ---------------------------------------------------------------------------


def test_a_lease_refusal_carries_a_code_and_the_lease_not_a_row_of_zeros() -> None:
    """The three lease-shaped refusals all answer ``gpu_leased``.

    And none of them claims "needs 0.00 GiB, 0.00 GiB usable" any more: a lease
    refusal never got as far as arithmetic, and printing the zeros anyway read
    as a sizing bug in the one message whose whole job is saying whose fault it
    is not.
    """
    # 1. a forced placement onto someone else's card
    book = LeaseBook()
    lease = book.acquire([0], holder="crucibleforge", model_ids=["other/model"])
    forced = make_planner(book).plan_load(make_record("mine/model", device_override=[0]))

    # 2. leases removed every usable card
    all_cards = LeaseBook()
    everything = all_cards.acquire([0, 1, 2, 3], holder="crucibleforge", model_ids=["other/model"])
    starved = make_planner(all_cards).plan_load(make_record("mine/model"))

    # 3. allowed_devices matches only leased cards
    narrow = LeaseBook()
    theirs = narrow.acquire([0, 1], holder="crucibleforge", model_ids=["other/model"])
    restricted = make_planner(narrow).plan_load(make_record("mine/model", allowed_devices=[0, 1]))

    for result, held in ((forced, lease), (starved, everything), (restricted, theirs)):
        assert isinstance(result, LoadRejected), result
        assert result.reason_code == "gpu_leased"
        assert [entry["id"] for entry in result.leases] == [held.id]
        assert result.leases[0]["holder_family"] == "crucibleforge"
        assert result.leases[0]["retry_after_s"] is not None
        assert result.message().startswith("Cannot load 'mine/model' entirely in VRAM:")

    # The two refusals that never reached the estimator no longer invent a row
    # of zeros. (The third DID estimate -- with every card leased away there is
    # genuinely 0.00 GiB usable, and saying so is honest, not a bug.)
    for silent in (forced, restricted):
        assert "GiB usable" not in silent.message(), silent.message()

    # The word every existing client substring-matches is still there.
    assert "leased" in forced.message()


def test_allowed_devices_refused_for_a_non_lease_reason_says_so() -> None:
    """Not every empty allowed_devices is a lease, and the codes differ."""
    book = LeaseBook()
    book.acquire([3], holder="crucibleforge", model_ids=["other/model"])
    planner = make_planner(book)
    # CUDA 7 does not exist on this rig, so no lease can be why it is missing.
    result = planner.plan_load(make_record("mine/model", allowed_devices=[7]))

    assert isinstance(result, LoadRejected), result
    assert result.reason_code == "allowed_devices_unavailable"
    assert result.leases == []


def test_a_lease_on_a_card_this_load_never_wanted_is_context_not_cause() -> None:
    """The false positive worth pinning.

    A model that genuinely does not fit on the cards it CAN use must not be
    told ``gpu_leased`` because some unrelated lease stands elsewhere -- that
    sends a client away to wait for a release that will not change the answer.
    """
    book = LeaseBook()
    book.acquire([3], holder="crucibleforge", model_ids=["other/model"])
    planner = make_planner(book)
    result = planner.plan_load(make_record("mine/model", ctx_size=100_000_000))

    assert isinstance(result, LoadRejected), result
    assert result.reason_code is None
    assert result.leases == []
    # The prose still names the lease -- it is useful context, just not the code.
    assert any("leased" in line for line in result.suggestions)


def test_an_ordinary_shortfall_keeps_its_numbers_and_has_no_code() -> None:
    planner = make_planner(LeaseBook())
    huge = make_record("mine/model", ctx_size=100_000_000)
    result = planner.plan_load(huge)

    assert isinstance(result, LoadRejected), result
    assert result.reason_code is None
    assert result.leases == []
    assert "GiB usable" in result.message(), "a real shortfall still shows the arithmetic"


def test_a_forced_placement_that_does_not_fit_is_not_blamed_on_leases_elsewhere() -> None:
    """The cards a device_override names ARE the cards this load could use.

    The override outranks ``planner.excluded_devices`` and ``allowed_devices``
    alike, so the gpu_leased verdict must be taken over the forced set and
    nothing else. Before this was pinned, an override onto an excluded card
    computed the candidates as "live minus excluded", found every one of THOSE
    leased away, and told the client to wait for a release that would never
    make a too-big model fit on the card it was pinned to.
    """
    book = LeaseBook()
    book.acquire([0, 1, 3], holder="crucibleforge", model_ids=["other/model"])
    planner = make_planner(book, excluded_devices=[2])
    forced = make_record("mine/model", device_override=[2], ctx_size=100_000_000)

    result = planner.plan_load(forced)

    assert isinstance(result, LoadRejected), result
    assert result.reason_code is None, result.reason_code
    assert result.leases == []
    # A forced placement onto the leased cards themselves is still the clash.
    clash = planner.plan_load(make_record("mine/model", device_override=[0, 2]))
    assert isinstance(clash, LoadRejected) and clash.reason_code == "gpu_leased"


def test_parallel_auto_sizes_slots_even_when_the_default_is_one() -> None:
    """A leased model (D43) ignores an integer models.default_parallel."""
    planner = make_planner(LeaseBook())
    planner.config.models.default_parallel = 1
    record = make_record("mine/model")

    ordinary = planner.plan_load(record)
    leased = planner.plan_load(record, parallel_auto=True)

    assert isinstance(ordinary, LoadPlan) and isinstance(leased, LoadPlan)
    assert ordinary.parallel == 1 and ordinary.parallel_limited_by == "explicit"
    assert leased.parallel_limited_by != "explicit"
    assert leased.parallel >= 1


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


async def test_acquire_lease_unloads_idle_neighbours_and_spares_the_owner() -> None:
    owner = make_record("owner/model")
    neighbour = make_record("idle/neighbour")
    bystander = make_record("far/away")
    manager, supervisor = make_manager([owner, neighbour, bystander])
    supervisor.instances[owner.id] = placed(owner.id, [0])
    supervisor.instances[neighbour.id] = placed(neighbour.id, [0, 1])
    supervisor.instances[bystander.id] = placed(bystander.id, [3])

    lease = await manager.acquire_lease([0, 1], holder="api", model_ids=[owner.id], reason="warm")

    assert lease.model_ids == [owner.id] and lease.devices == [0, 1]
    assert owner.id in supervisor.instances, "the owner stays"
    assert neighbour.id not in supervisor.instances, "the idle neighbour on the cards goes"
    assert bystander.id in supervisor.instances, "a model elsewhere is untouched"
    assert manager.leases.all() == [lease]


async def test_acquire_lease_never_interrupts_a_stream() -> None:
    busy = make_record("busy/model")
    manager, supervisor = make_manager([busy])
    supervisor.instances[busy.id] = placed(busy.id, [0])
    supervisor.instances[busy.id].active_requests = 1

    with pytest.raises(ModelBusyError) as excinfo:
        await manager.acquire_lease([0], holder="api")
    assert "retry_after_s" in excinfo.value.details
    assert busy.id in supervisor.instances
    assert len(manager.leases) == 0


async def test_acquire_lease_refuses_a_pinned_resident_unless_forced() -> None:
    pinned = make_record("pinned/model", pinned=True)
    manager, supervisor = make_manager([pinned])
    supervisor.instances[pinned.id] = placed(pinned.id, [0])
    supervisor.instances[pinned.id].ttl_s = 0

    with pytest.raises(LeaseConflictError):
        await manager.acquire_lease([0], holder="api")
    assert pinned.id in supervisor.instances

    lease = await manager.acquire_lease([0], holder="api", force=True)
    assert pinned.id not in supervisor.instances
    assert manager._pinned_needing_load() == [pinned.id], "the reconciler will bring it back"
    assert lease.devices == [0]


async def test_acquire_lease_waits_for_a_load_in_flight_and_names_conflicts() -> None:
    manager, _supervisor = make_manager([make_record()])
    manager._loading.add("some/model")
    with pytest.raises(ModelBusyError):
        await manager.acquire_lease([0], holder="api")
    manager._loading.clear()

    first = await manager.acquire_lease([0], holder="api")
    with pytest.raises(LeaseConflictError) as excinfo:
        await manager.acquire_lease([0, 1], holder="api")
    assert first.id in excinfo.value.message

    with pytest.raises(LeaseNotFoundError):
        manager.release_lease("nope")
    assert manager.release_lease(first.id).id == first.id


class YieldingSupervisor(CountingSupervisor):
    """A stop that takes a turn of the loop, as a real child teardown does."""

    def __init__(self) -> None:
        super().__init__()
        self.during_stop: list[Any] = []
        self.on_stop: Any = None
        self.fail_stop = False

    async def stop(self, model_id: str, **_kwargs: Any) -> None:
        await asyncio.sleep(0)
        if self.on_stop is not None:
            self.during_stop.append(self.on_stop())
        if self.fail_stop:
            raise RuntimeError("the child would not die")
        self.instances.pop(model_id, None)


def _lease_rig(*records: Any) -> tuple[Any, YieldingSupervisor]:
    manager, _ = make_manager(list(records))
    supervisor = YieldingSupervisor()
    manager.supervisor = supervisor  # type: ignore[assignment]
    return manager, supervisor


async def test_a_load_mid_eviction_sees_the_cards_blocked() -> None:
    """The book entry used to be written only after the awaited evictions, so a
    load planning during a teardown saw the cards free and landed on them."""
    owner = make_record("owner/model")
    neighbour = make_record("idle/neighbour")
    manager, supervisor = _lease_rig(owner, neighbour)
    supervisor.instances[neighbour.id] = placed(neighbour.id, [0, 1])
    supervisor.on_stop = lambda: manager.leases.blocked_for("jit/intruder")

    lease = await manager.acquire_lease([0, 1], holder="api", model_ids=[owner.id])

    assert supervisor.during_stop == [frozenset({0, 1})], "blocked from the first eviction on"
    assert manager.leases.all() == [lease]


async def test_two_overlapping_grants_the_loser_evicts_nothing() -> None:
    victim_a = make_record("victim/a")
    victim_b = make_record("victim/b")
    manager, supervisor = _lease_rig(victim_a, victim_b)
    supervisor.instances[victim_a.id] = placed(victim_a.id, [0])
    supervisor.instances[victim_b.id] = placed(victim_b.id, [1])

    results = await asyncio.gather(
        manager.acquire_lease([0], holder="first"),
        manager.acquire_lease([0, 1], holder="second"),
        return_exceptions=True,
    )

    conflicts = [r for r in results if isinstance(r, LeaseConflictError)]
    assert len(conflicts) == 1, results
    assert victim_b.id in supervisor.instances, "the loser unloaded nothing for a lease it lost"
    assert len(manager.leases) == 1


async def test_a_failed_eviction_leaves_no_lease_behind() -> None:
    victim = make_record("victim/model")
    manager, supervisor = _lease_rig(victim)
    supervisor.instances[victim.id] = placed(victim.id, [0])
    supervisor.fail_stop = True

    with pytest.raises(RuntimeError):
        await manager.acquire_lease([0], holder="api")

    assert len(manager.leases) == 0, "the caller was never handed a lease to release"


async def test_a_victim_that_became_busy_during_the_grant_refuses_and_releases() -> None:
    """Idle at the scan is not idle at the stop: a later victim can pick up a
    request while an earlier one is torn down, and supervisor.stop has no guard."""
    first = make_record("victim/first")
    second = make_record("victim/second")
    manager, supervisor = _lease_rig(first, second)
    supervisor.instances[first.id] = placed(first.id, [0])
    supervisor.instances[second.id] = placed(second.id, [1])
    supervisor.on_stop = lambda: supervisor.mark_request_start(second.id)

    with pytest.raises(ModelBusyError):
        await manager.acquire_lease([0, 1], holder="api")

    assert second.id in supervisor.instances, "a stream is never interrupted (D36)"
    assert len(manager.leases) == 0


def test_expire_leases_counts_owner_activity_and_releases_the_idle() -> None:
    owner = make_record("owner/model")
    manager, supervisor = make_manager([owner])
    now = time.time()
    active = manager.leases.acquire(
        [0], holder="api", model_ids=[owner.id], idle_ttl_s=60.0, now=now - 600
    )
    forgotten = manager.leases.acquire([3], holder="api", idle_ttl_s=60.0, now=now - 600)
    supervisor.instances[owner.id] = placed(owner.id, [0], idle_s=5.0)

    manager._expire_leases()

    assert manager.leases.get(active.id) is not None, "the owner answered 5s ago"
    assert manager.leases.get(forgotten.id) is None, "nobody touched it for ten minutes"


def test_lease_profile_forces_the_cards_and_auto_slots() -> None:
    record = make_record("mine/model")
    manager, _supervisor = make_manager([record])
    manager.leases.acquire([2, 3], holder="api", model_ids=[record.id])

    steered, parallel_auto = manager._apply_lease_profile(record)

    assert steered.settings.device_override == [2, 3]
    assert parallel_auto is True
    assert record.settings.device_override is None, "the persisted settings are untouched"

    explicit = make_record("mine/model", parallel=2, device_override=[3])
    same, parallel_auto = manager._apply_lease_profile(explicit)
    assert same.settings.device_override == [3], "an explicit override wins"
    assert parallel_auto is False

    unleased, parallel_auto = manager._apply_lease_profile(make_record("other/model"))
    assert parallel_auto is False and unleased.settings.device_override is None


class StubDb:
    def __init__(self, report: dict[str, Any] | None) -> None:
        self.report = report

    def latest_benchmark(self, model_id: str) -> dict[str, Any] | None:
        return None if self.report is None else {"model_id": model_id, "report": self.report}


def test_lease_profile_takes_the_measured_split_mode_only_for_those_cards() -> None:
    """Tensor split is never assumed (D38): only a benchmark on these cards may pick it."""
    record = make_record("mine/model")
    manager, _supervisor = make_manager([record])
    manager.leases.acquire([2, 3], holder="api", model_ids=[record.id])
    manager.db = StubDb(
        {
            "best_generation_mode": "dual_3090-tensor",
            "results": [
                {"mode": "dual_3090", "devices": [2, 3], "split_mode": "layer"},
                {
                    "mode": "dual_3090-tensor",
                    "devices": [2, 3],
                    "split_mode": "tensor",
                    "ubatch": 1024,
                },
            ],
        }
    )

    steered, _ = manager._apply_lease_profile(record)
    assert steered.settings.split_mode == "tensor"
    assert steered.settings.ubatch_size == 1024

    manager.db = StubDb(
        {
            "best_generation_mode": "dual_5090-tensor",
            "results": [{"mode": "dual_5090-tensor", "devices": [0, 1], "split_mode": "tensor"}],
        }
    )
    other_cards, _ = manager._apply_lease_profile(record)
    assert other_cards.settings.split_mode is None, "measured on different cards: not applied"

    manager.db = None
    none, _ = manager._apply_lease_profile(record)
    assert none.settings.split_mode is None


def test_pinned_record_helper_accepts_settings_kwargs() -> None:
    """Sanity anchor: make_record forwards ModelSettings fields."""
    assert make_record("x", pinned=True).settings == ModelSettings(pinned=True)


# ---------------------------------------------------------------------------
# manager.lease_check: the streaming twin of admission_check (D53)
# ---------------------------------------------------------------------------


def test_lease_check_refuses_only_when_the_answer_is_certain() -> None:
    """Conservative on purpose: it must never invent a refusal.

    It runs before the 200 on a streaming request, where a wrong "no" is an
    outage and a missed "no" merely falls through to the in-stream backstop
    that has always been there.
    """
    from studioforge.errors import InsufficientVramError

    forced = make_record("forced/model", device_override=[0])
    free = make_record("free/model")
    manager, supervisor = make_manager([forced, free])
    manager.leases.acquire([0], holder="crucibleforge", model_ids=["someone/else"])

    # Pinned to a card somebody else holds, and not loaded: certain.
    with pytest.raises(InsufficientVramError) as excinfo:
        manager.lease_check("forced/model")
    assert excinfo.value.code == "gpu_leased"
    assert excinfo.value.details["lease"]["holder"] == "crucibleforge"

    # No override, and this stub manager has no probe to enumerate cards with:
    # unknown is not a refusal.
    manager.lease_check("free/model")

    # Already serving: no load is needed, so no card has to be found. Refusing
    # here would take a working model away from its clients over a lease that
    # never touched it.
    supervisor.instances["forced/model"] = placed("forced/model", [0])
    manager.lease_check("forced/model")


def test_lease_check_is_silent_when_nothing_is_leased() -> None:
    record = make_record("mine/model", device_override=[0])
    manager, _supervisor = make_manager([record])
    manager.lease_check("mine/model")
    manager.lease_check("never/heard/of/it")
