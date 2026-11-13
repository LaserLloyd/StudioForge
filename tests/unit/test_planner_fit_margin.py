"""The walk prices what the load is charged, says how much room is left, and steps down (D88).

The 2026-11-09 case, reproduced here on fake cards: two 24 GiB cards, a 33.8 GB
Q8 judge at 16384 tokens per slot with a q8_0 cache. The formula said six slots
fit. The last measurement of exactly that load sat a little UNDER its estimate,
and D51's 10% band on the bytes above the weights put six slots 0.03 GiB over
the pair's usable VRAM. The walk (``load_recommended`` / ``plan-recommended``)
was built without the D51 lookup, so it chose six on the formula; the real
load, planned by the main planner, applied the correction and refused; and
nothing stepped down to the five that fit (A-3, A-8, B.5).

What these pin:

* the walk reads the same history as the load, so they agree (five), and the
  dry run SHOWS the near miss: ``alternatives`` with six refused by ~0.03 GiB;
* when the load still refuses (a measurement lands between the walk and the
  load), ``load_recommended`` re-walks once with one fewer slot, and says so;
* the fit response's margin arithmetic and the ``alternatives`` ordering;
* slot sizing counts the eviction credit the fit was decided with (A-13);
* the ``?dryRun=true`` plumbing routes exactly as the real load would.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studioforge.core import catalog as catalog_mod
from studioforge.core.manager import ModelManager
from studioforge.core.planner import OBS_SAFETY, OBSERVATION_NOTE_PER_PID_DEVICE, Planner
from studioforge.db import Database
from studioforge.errors import BadRequestError, InsufficientVramError
from studioforge.types import GB, LoadPlan, ModelRecord, ModelSettings
from tests.unit.test_load_recommended import MODEL, Registry, make_manager, serving
from tests.unit.test_load_retry import StubSupervisor
from tests.unit.test_planner import StubProbe, gpu, make_config, make_meta, make_record

JUDGE = "pub/judge-32b-q8"
JUDGE_BYTES = int(33.8e9)  # "33.8 GB", decimal, as the file listing says
CTX = 16384
PAIR = "sm_86-24g+sm_86-24g"
#: Each 24 GiB card has 23.1 GiB free (a desktop on it), so the pair offers
#: 2 x (23.1 - 2.4 headroom) = 41.4 GiB usable.
FREE_GIB = 23.1
MISS = int(0.03 * GB)


def judge_record() -> ModelRecord:
    meta = make_meta(
        n_layer=64,
        n_head=32,
        n_head_kv=4,
        n_embd=5120,
        tensor_bytes=JUDGE_BYTES,
        n_ctx_train=131072,
        key_length=128,
        value_length=128,
    )
    return make_record(
        JUDGE,
        meta=meta,
        size_bytes=JUDGE_BYTES,
        settings=ModelSettings(kv_cache_type="q8_0", kv_cache_type_v="q8_0"),
    )


def card_pair() -> StubProbe:
    return StubProbe([gpu(0, 24.0, FREE_GIB, (8, 6)), gpu(1, 24.0, FREE_GIB, (8, 6))])


class ManagerDb:
    """What the manager asks its database for, over a real one."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def parallel_observations(self, model_id=None, *, devices=None, limit=200):  # noqa: ANN001, ANN201
        return []

    def record_load_observation(self, **row: Any) -> None:
        self.database.record_load_observation(**row)


@pytest.fixture()
def database(tmp_path: Path) -> Database:
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    yield db
    db.close()


def judge_manager(database: Database) -> tuple[ModelManager, StubSupervisor, Planner]:
    """The main planner wired to a real observation table, as app.py wires it."""
    config = make_config()
    planner = Planner(
        config,
        card_pair(),
        observation_sink=lambda row: database.record_load_observation(**row),
        observation_lookup=database.matching_observation,
        log_plans=False,
    )
    supervisor = StubSupervisor()
    manager = ModelManager(
        config,
        registry=Registry([judge_record()]),  # type: ignore[arg-type]
        planner=planner,
        supervisor=supervisor,  # type: ignore[arg-type]
        db=ManagerDb(database),  # type: ignore[arg-type]
    )
    return manager, supervisor, planner


def formula_at(planner: Planner, slots: int) -> Any:
    return planner.estimate(
        judge_record(),
        ctx_size=CTX,
        parallel=slots,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        n_devices=2,
    )


def usable(planner: Planner) -> int:
    return sum(planner.usable_bytes(g) for g in planner.probe.list_gpus())


def record_six_slot_measurement(database: Database, planner: Planner) -> int:
    """The measurement that makes six slots miss by 0.03 GiB once D51's band is on it."""
    formula = formula_at(planner, 6)
    target = usable(planner) + MISS
    observed = int(formula.weights_bytes + (target - formula.weights_bytes) / OBS_SAFETY) + 1
    database.record_load_observation(
        model_id=JUDGE,
        ctx_size=CTX,
        parallel=6,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        devices="0,1",
        predicted_bytes=formula.total_bytes,
        actual_bytes=observed,
        weights_bytes=formula.weights_bytes,
        ok=True,
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
        class_key=PAIR,
        split_mode="layer",
        gguf_size=JUDGE_BYTES,
    )
    return observed


def near(value: int, expected: int, *, within: float = 0.002) -> bool:
    return abs(int(value) - int(expected)) <= within * GB


# ---------------------------------------------------------------------------
# The real case
# ---------------------------------------------------------------------------


def test_the_fixture_is_the_real_case(database: Database) -> None:
    """Formula: six fit, seven do not. Measured: a little under the formula.
    Charged: weights + 1.1 x the rest = 0.03 GiB over the pair."""
    _manager, _supervisor, planner = judge_manager(database)
    room = usable(planner)
    six, seven = formula_at(planner, 6), formula_at(planner, 7)
    assert six.total_bytes <= room < seven.total_bytes
    observed = record_six_slot_measurement(database, planner)
    assert observed < six.total_bytes, "the child held less than its estimate"
    charged = six.weights_bytes + (observed - six.weights_bytes) * OBS_SAFETY
    assert near(int(charged) - room, MISS)


async def test_before_the_measurement_the_formula_chooses_six(database: Database) -> None:
    manager, _supervisor, _planner = judge_manager(database)
    dry = await manager.plan_recommended(JUDGE, CTX)
    assert dry["fits"] is True and dry["parallel"] == 6
    assert dry["observed_applied"] is False
    assert dry["formula_bytes"] == dry["needs_bytes"]


async def test_the_walk_and_the_load_agree_on_five_and_the_near_miss_is_visible(
    database: Database,
) -> None:
    manager, supervisor, planner = judge_manager(database)
    record_six_slot_measurement(database, planner)

    dry = await manager.plan_recommended(JUDGE, CTX)
    assert dry["fits"] is True
    assert dry["parallel"] == 5 and dry["devices"] == [0, 1]
    assert dry["max_parallel_that_fits"] == 5
    six = next(a for a in dry["alternatives"] if a["parallel"] == 6)
    assert six["fits"] is False and six["observed_applied"] is True
    assert six["margin_bytes"] < 0 and near(-six["margin_bytes"], MISS)
    five = next(a for a in dry["alternatives"] if a["chosen"])
    assert five["parallel"] == 5 and five["margin_bytes"] == dry["margin_bytes"] > 0

    instance = await manager.load_recommended(JUDGE, CTX)
    assert instance.plan is not None
    assert instance.plan.parallel == dry["parallel"] == 5
    assert instance.plan.devices == dry["devices"]
    assert instance.plan.estimate.total_bytes == dry["needs_bytes"]
    assert supervisor.starts == 1
    assert not [n for n in instance.plan.notes if "stepped down" in n]


async def test_a_refusal_at_six_steps_down_to_five_with_the_miss_in_the_notes(
    database: Database,
) -> None:
    """The race the walk cannot see: the row lands between the walk and the load."""
    manager, supervisor, planner = judge_manager(database)
    original = manager._decide_recommended
    walks: list[int | None] = []

    def walk_then_measure(prep: Any, **kwargs: Any) -> Any:
        decision = original(prep, **kwargs)
        walks.append(kwargs.get("max_slots"))
        if len(walks) == 1:
            assert decision.winner is not None and decision.winner["parallel"] == 6
            record_six_slot_measurement(database, planner)
        return decision

    manager._decide_recommended = walk_then_measure  # type: ignore[method-assign]
    instance = await manager.load_recommended(JUDGE, CTX)

    assert walks == [None, 5], "one re-walk, capped at one fewer slot"
    assert instance.plan is not None and instance.plan.parallel == 5
    assert supervisor.starts == 1, "the refused load never started a child"
    note = next(n for n in instance.plan.notes if "stepped down from 6 to 5" in n)
    assert "refused by 0.03 GiB" in note
    assert "GiB usable" in note


async def test_a_floor_that_cannot_step_down_is_a_507_with_the_numbers(
    database: Database,
) -> None:
    manager, supervisor, planner = judge_manager(database)
    original = manager._decide_recommended

    def walk_then_measure(prep: Any, **kwargs: Any) -> Any:
        decision = original(prep, **kwargs)
        record_six_slot_measurement(database, planner)
        return decision

    manager._decide_recommended = walk_then_measure  # type: ignore[method-assign]
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(JUDGE, CTX, min_slots=6)
    details = excinfo.value.details
    assert excinfo.value.code == "insufficient_vram"
    assert details["requested_parallel"] == 6
    assert details["refused_load"]["margin_bytes"] < 0
    assert near(-details["refused_load"]["margin_bytes"], MISS)
    assert details["margin_bytes"] == details["refused_load"]["margin_bytes"]
    # The planner's own phrase and numbers are still there, verbatim.
    assert "reduce parallel from 6 to 5" in excinfo.value.message
    assert details["max_parallel_that_fits"] == 5
    assert "min_slots" in excinfo.value.message
    assert supervisor.starts == 0


async def test_a_measurement_above_the_formula_moves_walk_and_load_together(
    database: Database,
) -> None:
    """The plainer shape: the child at six slots held MORE than the formula said."""
    manager, supervisor, planner = judge_manager(database)
    formula = formula_at(planner, 6)
    database.record_load_observation(
        model_id=JUDGE,
        ctx_size=CTX,
        parallel=6,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        devices="0,1",
        predicted_bytes=formula.total_bytes,
        actual_bytes=formula.total_bytes + int(0.5 * GB),
        weights_bytes=formula.weights_bytes,
        ok=True,
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
        class_key=PAIR,
        split_mode="layer",
        gguf_size=JUDGE_BYTES,
    )
    dry = await manager.plan_recommended(JUDGE, CTX)
    six = next(a for a in dry["alternatives"] if a["parallel"] == 6)
    assert six["fits"] is False and six["needs_bytes"] > formula.total_bytes
    instance = await manager.load_recommended(JUDGE, CTX)
    assert instance.plan is not None
    assert dry["fits"] is True and instance.plan.parallel == dry["parallel"] == 5
    assert supervisor.starts == 1


async def test_the_dry_run_and_the_real_call_refuse_a_floor_alike(database: Database) -> None:
    """``plan_recommended.fits`` equals the real load's verdict (E.2)."""
    manager, supervisor, planner = judge_manager(database)
    record_six_slot_measurement(database, planner)

    dry = await manager.plan_recommended(JUDGE, CTX, min_slots=6)
    assert dry["fits"] is False and dry["code"] == "insufficient_vram"
    assert dry["max_parallel_that_fits"] == 5
    assert dry["margin_bytes"] < 0 and near(-dry["margin_bytes"], MISS)
    assert dry["observed_applied"] is True
    assert [a["parallel"] for a in dry["alternatives"]] == [5]

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(JUDGE, CTX, min_slots=6)
    assert excinfo.value.details["max_parallel_that_fits"] == 5
    assert excinfo.value.details["margin_bytes"] == dry["margin_bytes"]
    assert supervisor.starts == 0


# ---------------------------------------------------------------------------
# Margin arithmetic and the alternatives table
# ---------------------------------------------------------------------------


async def test_the_margin_is_usable_minus_needs_everywhere(database: Database) -> None:
    manager, _supervisor, planner = judge_manager(database)
    record_six_slot_measurement(database, planner)
    dry = await manager.plan_recommended(JUDGE, CTX)

    assert dry["usable_bytes"] == usable(planner)
    assert dry["margin_bytes"] == dry["usable_bytes"] - dry["needs_bytes"]
    assert sum(dry["usable_bytes_per_gpu"].values()) == dry["usable_bytes"]
    per_card = dry["margin_bytes_per_gpu"]
    assert set(per_card) == {0, 1}
    assert abs(sum(per_card.values()) - dry["margin_bytes"]) <= 2, "integer split rounding"
    assert dry["estimate_bytes"]["total"] == dry["needs_bytes"]
    assert dry["headroom_fraction"] == pytest.approx(0.10)
    for alt in dry["alternatives"]:
        assert alt["margin_bytes"] == alt["usable_bytes"] - alt["needs_bytes"]
        assert alt["fits"] is (alt["margin_bytes"] >= 0)


async def test_alternatives_run_most_slots_first_and_stop_after_the_first_miss(
    database: Database,
) -> None:
    manager, _supervisor, planner = judge_manager(database)
    record_six_slot_measurement(database, planner)
    dry = await manager.plan_recommended(JUDGE, CTX)
    counts = [a["parallel"] for a in dry["alternatives"]]
    assert counts == sorted(counts, reverse=True) == list(range(6, 0, -1))
    assert [a["parallel"] for a in dry["alternatives"] if a["chosen"]] == [dry["parallel"]]
    assert [a["parallel"] for a in dry["alternatives"] if not a["fits"]] == [6]


async def test_max_slots_bounds_the_table(database: Database) -> None:
    manager, _supervisor, _planner = judge_manager(database)
    dry = await manager.plan_recommended(JUDGE, CTX, max_slots=3)
    assert dry["parallel"] == 3
    assert [a["parallel"] for a in dry["alternatives"]] == [3, 2, 1]
    assert dry["max_parallel_that_fits"] == 3


async def test_every_mode_summary_carries_its_margin(database: Database) -> None:
    manager, _supervisor, _planner = judge_manager(database)
    dry = await manager.plan_recommended(JUDGE, CTX)
    for mode in dry["modes"]:
        assert "margin_bytes" in mode and "parallel" in mode


async def test_a_refusal_always_names_max_parallel_that_fits_and_the_miss() -> None:
    """``max_parallel_that_fits`` is present on every refusal now (None: not even one)."""
    manager, _supervisor = make_manager(free_gib=6.0, n_ctx_train=1048576)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 1048576)
    details = excinfo.value.details
    assert "max_parallel_that_fits" in details and details["max_parallel_that_fits"] is None
    assert details["margin_bytes"] == details["usable_bytes"] - details["needs_bytes"] < 0
    assert all("margin_bytes" in mode for mode in details["modes"])
    dry = await manager.plan_recommended(MODEL, 1048576)
    assert dry["margin_bytes"] == details["margin_bytes"]


# ---------------------------------------------------------------------------
# Slot sizing with eviction credit (A-13)
# ---------------------------------------------------------------------------


async def test_slot_sizing_counts_the_room_an_eviction_frees() -> None:
    """Every card holds an idle model; the walk evicts it. The slots must be
    sized against the room that frees, not the 2 GiB free before it leaves."""
    idle = serving("pub/other", requests=0, devices=[0, 1, 2, 3])
    idle.ttl_s = 1800
    manager, _supervisor = make_manager(free_gib=2.0, loaded=[idle])
    dry = await manager.plan_recommended(MODEL, 16384)
    assert dry["fits"] is True and dry["evict_model_ids"] == ["pub/other"]
    assert dry["parallel"] > 1
    assert dry["usable_bytes"] > 40 * GB, "the credit is in the capacity the fit reports"

    # The same plan sized the pre-D88 way -- the live probe alone -- is one slot.
    record = manager.registry.resolve(MODEL)
    pinned = record.model_copy(
        update={"settings": record.settings.model_copy(update={"device_override": [0, 1]})}
    )
    plan = manager.planner.plan_load(pinned, ctx_size=16384, parallel=1, loaded=[idle])
    assert isinstance(plan, LoadPlan) and plan.evict_model_ids == ["pub/other"]
    with_credit, _bound, _vram = catalog_mod.slots_for_plan(manager.planner, record, plan)
    live_only = plan.model_copy(update={"usable_bytes_per_gpu": {}})
    without, _bound, _vram = catalog_mod.slots_for_plan(manager.planner, record, live_only)
    assert without == 1 < with_credit
    # An explicit credit restores it for a plan that recorded no capacities.
    credit = {d: int(29 * GB) for d in (0, 1)}
    restored, _bound, _vram = catalog_mod.slots_for_plan(
        manager.planner, record, live_only, extra_free=credit
    )
    assert restored > 1

    instance = await manager.load_recommended(MODEL, 16384)
    assert instance.plan is not None and instance.plan.parallel == dry["parallel"]


async def test_a_priority_load_sizes_slots_with_its_preemption_credit() -> None:
    idle = serving("pub/other", requests=0, devices=[0, 1, 2, 3])
    idle.ttl_s = 1800
    manager, _supervisor = make_manager(free_gib=2.0, loaded=[idle])
    dry = await manager.plan_recommended(MODEL, 16384, priority=1)
    assert dry["fits"] is True and "pub/other" in dry["evict_model_ids"]
    assert dry["parallel"] > 1


# ---------------------------------------------------------------------------
# The walk's planners and the dry-run plumbing
# ---------------------------------------------------------------------------


def test_every_walk_planner_reads_the_main_planners_history(database: Database) -> None:
    manager, _supervisor, planner = judge_manager(database)
    walk = manager._walk_planner(planner.probe)
    assert walk is not planner
    assert walk.observation_lookup is planner.observation_lookup
    assert walk.observation_sink is planner.observation_sink
    sibling = planner.sibling(planner.probe)
    assert sibling.observation_lookup is planner.observation_lookup


def test_profiles_price_modes_with_the_correction(database: Database) -> None:
    """``/profiles`` (manager.py's other walk) reads the history too."""
    manager, _supervisor, planner = judge_manager(database)
    seen: list[str] = []
    real = planner.observation_lookup
    assert real is not None

    def spy(model_id: str, **key: Any) -> Any:
        seen.append(model_id)
        return real(model_id, **key)

    planner._observation_lookup = spy
    manager.placement_profiles(JUDGE)
    assert JUDGE in seen


@pytest.mark.parametrize(
    "body, route",
    [
        pytest.param({"context_length": CTX}, "load_recommended", id="window only"),
        pytest.param({"context_length": CTX, "devices": [0, 1]}, "load", id="devices"),
        pytest.param({"context_length": CTX, "parallel": 6}, "load", id="parallel"),
        pytest.param({"context_length": CTX, "kv_cache_type": "q8_0"}, "load", id="kv"),
        pytest.param({"context_length": None}, "load", id="no window"),
    ],
)
def test_the_dry_run_routes_as_the_load_would(body: dict[str, Any], route: str) -> None:
    assert ModelManager.load_route_for(**body) == route


def test_force_is_an_explicit_load_in_both() -> None:
    """``force`` is the explicit load's reload licence; the walk never takes it."""
    assert ModelManager.load_route_for(context_length=CTX, force=True) == "load"


async def test_a_lease_bound_explicit_dry_run_plans_on_the_lease_and_binds_nothing(
    database: Database,
) -> None:
    """The dry run of ``load(lease_id=...)`` (D87): the lease's cards, the same refusals."""
    manager, supervisor, _planner = judge_manager(database)
    lease = manager.leases.acquire([1], holder="judge-a")
    dry = await manager.load_dry_run(JUDGE, context_length=4096, parallel=1, lease_id=lease.id)
    assert dry["dry_run_of"] == "load"
    assert dry["lease_id"] == lease.id and dry["lease_devices"] == [1]
    assert JUDGE not in manager.leases.get(lease.id).model_ids, "a dry run binds nothing"
    assert supervisor.starts == 0
    with pytest.raises(BadRequestError):
        # devices outside the lease: the 400 the bound load answers.
        await manager.load_dry_run(
            JUDGE, context_length=4096, devices=[0], parallel=1, lease_id=lease.id
        )


async def test_the_dry_run_of_an_explicit_load_is_that_loads_verdict(database: Database) -> None:
    manager, supervisor, planner = judge_manager(database)
    record_six_slot_measurement(database, planner)

    walk = await manager.load_dry_run(JUDGE, context_length=CTX)
    assert walk["dry_run"] is True and walk["dry_run_of"] == "load_recommended"
    assert walk["parallel"] == 5 and walk["margin_bytes"] > 0

    explicit = await manager.load_dry_run(JUDGE, context_length=CTX, devices=[0, 1], parallel=6)
    assert explicit["dry_run_of"] == "load" and explicit["fits"] is False
    assert explicit["observed_applied"] is True
    assert near(-explicit["margin_bytes"], MISS)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load(JUDGE, ctx_size=CTX, devices=[0, 1], parallel=6)
    assert excinfo.value.details["required_bytes"] == explicit["needs_bytes"]
    assert excinfo.value.details["available_bytes"] == explicit["usable_bytes"]
    assert supervisor.starts == 0

    fits = await manager.load_dry_run(JUDGE, context_length=CTX, devices=[0, 1], parallel=5)
    assert fits["fits"] is True
    assert fits["margin_bytes"] == fits["usable_bytes"] - fits["needs_bytes"] > 0
