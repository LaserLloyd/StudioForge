"""`load_recommended(model, ctx)`: the model and the window, nothing else.

The user's sentence: *"so it can simply specify the model and context needed,
and the server works the rest, or returns an error if it can't load the
requested context for some reason"*.

What these pin:

* **the context is the request.** This is the one load path that does not walk
  D14's halving ladder. An agent that asked for 262144 because its transcript
  is 200k long is not helped by silently getting 131072 and finding out
  mid-conversation, so a window that does not fit is a refusal, not a downgrade;
* **the mode walk is headline-first**, and only reaches for eviction after
  every mode has been tried without it -- a roomier placement must never be the
  reason somebody else's model is stopped when a later mode would have done;
* **the refusal is actionable in one read**: per mode, the largest context that
  would work and what is in the way, with ``retry_after_s`` only when the cause
  is transient.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.core import manager as manager_module
from studioforge.core.manager import ModelManager
from studioforge.core.planner import BUSY_RETRY_AFTER_S, Planner
from studioforge.core.priority import PRIORITY_BACKGROUND, PRIORITY_CHAT
from studioforge.errors import BadRequestError, InsufficientVramError, ModelBusyError
from studioforge.types import GB, InstanceInfo, LoadPlan, ModelSettings, VirtualPreset
from tests.unit.test_catalog import dense_meta, record
from tests.unit.test_load_retry import StubSupervisor
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

MODEL = "pub/dense-8b"


class Registry:
    def __init__(self, records: list[Any]) -> None:
        self._records = {r.id: r for r in records}
        #: Every ``save_settings`` call, newest last -- the persist tests below
        #: assert on what was written, not just that something was.
        self.saved: list[tuple[str, ModelSettings]] = []

    def resolve(self, name: str) -> Any:
        return self._records.get(name)

    def get(self, model_id: str) -> Any:
        return self._records.get(model_id)

    def get_adapter(self, adapter_id: str) -> None:
        return None

    def known_ids(self) -> list[str]:
        return sorted(self._records)

    def all(self) -> list[Any]:
        return list(self._records.values())

    def touch(self, model_id: str) -> None:
        return None

    def save_settings(self, model_id: str, settings: ModelSettings) -> Any:
        # The real registry re-validates here because this is reachable from
        # the HTTP API, and since D48 it refuses a row whose ``parallel`` is
        # above the model's ``max_parallel_cap``. Mirrored (only that check) so
        # the persist tests below meet the same refusal the server would.
        cap = settings.max_parallel_cap
        if settings.parallel is not None and cap is not None and settings.parallel > cap:
            raise BadRequestError(
                f"parallel ({settings.parallel}) is above this model's max_parallel_cap ({cap}).",
                param="settings",
            )
        self.saved.append((model_id, settings))
        record = self._records[model_id]
        record.settings = settings
        return record


class ParallelDb:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []

    def parallel_observations(self, model_id=None, *, devices=None, limit=200):  # noqa: ANN001, ANN201
        return list(self.rows)

    def record_load_observation(self, **_: Any) -> None:
        return None


def lopsided_rig() -> Any:
    """5090s nearly full, 3090s empty -- so the mode walk has a real choice."""
    from tests.unit.test_planner import StubProbe, gpu

    return StubProbe(
        [
            gpu(0, 31.84, 2.0, (12, 0)),
            gpu(1, 31.84, 2.0, (12, 0)),
            gpu(2, 24.0, 23.5, (8, 6)),
            gpu(3, 24.0, 23.5, (8, 6)),
        ]
    )


def make_manager(
    *,
    free_gib: float = 31.0,
    probe: Any = None,
    n_ctx_train: int = 131072,
    settings: ModelSettings | None = None,
    loaded: list[InstanceInfo] | None = None,
    db: ParallelDb | None = None,
) -> tuple[ModelManager, StubSupervisor]:
    rec = record(
        MODEL,
        dense_meta(n_ctx_train),
        mtime=1.0,
        size_bytes=8 * GB,
        settings=settings,
    )
    config = make_config()
    planner = Planner(
        config, probe if probe is not None else rig_5090x2_3090x2(free_gib), log_plans=False
    )
    supervisor = StubSupervisor()
    for instance in loaded or []:
        supervisor.instances[instance.model_id] = instance
    manager = ModelManager(
        config,
        registry=Registry([rec]),  # type: ignore[arg-type]
        planner=planner,
        supervisor=supervisor,  # type: ignore[arg-type]
        db=db,  # type: ignore[arg-type]
    )
    return manager, supervisor


def serving(model_id: str, *, requests: int = 1, devices: list[int] | None = None) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18101,
        ttl_s=1800,
        started_at=1.0,
        last_activity_at=1.0,
        active_requests=requests,
        plan=LoadPlan(
            model_id=model_id,
            devices=devices or [0, 1],
            ctx_size=32768,
            per_gpu_bytes={d: int(29 * GB) for d in (devices or [0, 1])},
        ),
    )


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


async def test_the_context_asked_for_is_the_context_loaded() -> None:
    manager, supervisor = make_manager()
    instance = await manager.load_recommended(MODEL, 65536, source="mcp:load_recommended")
    assert instance.plan is not None
    assert instance.plan.ctx_size == 65536
    assert instance.loaded_by == "mcp:load_recommended"
    assert supervisor.starts == 1


async def test_the_headline_placement_is_taken_first() -> None:
    """`dual_5090` leads, because that is what the user asks for by default."""
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [0, 1]


async def test_a_preferred_mode_is_honoured() -> None:
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(MODEL, 32768, prefer_modes=["dual_3090"])
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]


async def test_an_unknown_preferred_mode_names_the_ones_this_box_has() -> None:
    manager, _supervisor = make_manager()
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 32768, prefer_modes=["dual_4090"])
    assert excinfo.value.param == "prefer_modes"
    assert "dual_5090" in excinfo.value.message


async def test_the_quality_first_kv_rule_still_applies() -> None:
    """A 4-bit K cache is never chosen for you, on this path as on every other."""
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(MODEL, 65536)
    assert instance.plan is not None
    assert instance.plan.kv_cache_type == "f16"


async def test_the_slot_count_is_the_recommendation_not_the_ceiling() -> None:
    """A measured sweep saying "one slot" is obeyed by this path too."""
    sweep = [
        {
            "n_streams": n,
            "per_stream_tps": per_stream,
            "aggregate_tps": aggregate,
            "ts": float(n),
            "run_id": "run-a",
            "devices": "0,1",
            "ctx_per_slot": 16384,
            "kv_cache_type": "f16",
            "kv_cache_type_v": "f16",
        }
        for n, per_stream, aggregate in [(1, 100.0, 100.0), (2, 90.0, 102.0)]
    ]
    manager, _supervisor = make_manager(db=ParallelDb(sweep))
    instance = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"])
    assert instance.plan is not None
    assert instance.plan.parallel == 1


# ---------------------------------------------------------------------------
# Strictness
# ---------------------------------------------------------------------------


async def test_a_context_above_the_trained_window_is_refused_with_the_number() -> None:
    """Serving past n_ctx_train needs RoPE scaling and quietly degrades quality."""
    manager, supervisor = make_manager(n_ctx_train=32768)
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 131072)
    assert excinfo.value.param == "ctx_size"
    assert "32768" in excinfo.value.message
    assert excinfo.value.details["n_ctx_train"] == 32768
    assert supervisor.starts == 0


async def test_the_context_is_never_halved_silently() -> None:
    """The whole point: a window that does not fit is a refusal, not a downgrade."""
    manager, supervisor = make_manager(free_gib=2.0, n_ctx_train=1048576)
    with pytest.raises(InsufficientVramError):
        await manager.load_recommended(MODEL, 1048576)
    assert supervisor.starts == 0


async def test_a_zero_or_negative_context_is_a_400_before_anything_is_planned() -> None:
    manager, _supervisor = make_manager()
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 0)
    assert excinfo.value.param == "ctx_size"


# ---------------------------------------------------------------------------
# Eviction: idle only, and only after every mode has been tried without it
# ---------------------------------------------------------------------------


async def test_a_mode_that_fits_now_is_preferred_over_one_that_needs_an_eviction() -> None:
    """A better set of cards must not cost somebody else their model.

    The 5090s are full of an idle `pub/other` and the 3090s are empty, so the
    headline mode could take the load by evicting and the second mode can take
    it for free. The walk runs every mode with eviction OFF before it runs any
    of them with eviction on, so the free one wins -- D14's "a roomier
    placement is a nicety and must never be the reason somebody else's model is
    unloaded", extended from context rungs to hardware modes.
    """
    idle = serving("pub/other", requests=0, devices=[0, 1])
    idle.ttl_s = 1800
    manager, _supervisor = make_manager(probe=lopsided_rig(), loaded=[idle])
    instance = await manager.load_recommended(MODEL, 16384)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]
    assert instance.plan.evict_model_ids == []


async def test_an_idle_model_is_evicted_when_no_mode_fits_without_it() -> None:
    """...but eviction of an IDLE model is still on the table as the last step."""
    idle = serving("pub/other", requests=0, devices=[0, 1, 2, 3])
    idle.ttl_s = 1800
    manager, _supervisor = make_manager(free_gib=2.0, loaded=[idle])
    instance = await manager.load_recommended(MODEL, 16384)
    assert instance.plan is not None
    assert "pub/other" in instance.plan.evict_model_ids


async def test_a_busy_model_is_never_evicted_and_the_refusal_says_so() -> None:
    busy = serving("pub/busy", requests=3, devices=[0, 1])
    manager, supervisor = make_manager(free_gib=1.0, loaded=[busy], n_ctx_train=262144)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144)
    details = excinfo.value.details
    assert details["busy_models"] == [{"model_id": "pub/busy", "active_requests": 3}]
    assert details["retry_after_s"] == BUSY_RETRY_AFTER_S
    assert supervisor.starts == 0


async def test_an_ordinary_full_box_carries_no_retry_advice() -> None:
    """ "Try again later" is bad advice when nothing is going to change."""
    manager, _supervisor = make_manager(free_gib=1.0, n_ctx_train=262144)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144)
    assert excinfo.value.details["retry_after_s"] is None
    assert excinfo.value.details["busy_models"] == []


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


async def test_the_refusal_names_every_mode_and_what_it_could_have_done() -> None:
    manager, _supervisor = make_manager(free_gib=6.0, n_ctx_train=1048576)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 1048576)
    details = excinfo.value.details
    modes = {entry["mode"]: entry for entry in details["modes"]}
    assert set(modes) == {"dual_5090", "dual_3090", "all_gpus", "single_5090"}
    for entry in modes.values():
        assert entry["reason"]
        assert entry["devices"]
    assert details["requested_ctx"] == 1048576
    assert details["n_ctx_train"] == 1048576


async def test_the_refusal_offers_the_largest_context_that_would_work() -> None:
    """One read has to be enough to know what to ask for next."""
    manager, _supervisor = make_manager(free_gib=6.0, n_ctx_train=1048576)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 1048576)
    details = excinfo.value.details
    largest = details["largest_ctx_that_fits"]
    assert largest and largest < 1048576
    assert str(largest) in " ".join(details["suggestions"])
    # ...and that number really does load.
    instance = await manager.load_recommended(MODEL, int(largest))
    assert instance.plan is not None
    assert instance.plan.ctx_size == int(largest)


# ---------------------------------------------------------------------------
# kv_min
# ---------------------------------------------------------------------------


async def test_kv_min_refuses_a_placement_that_only_reaches_the_window_by_quantizing() -> None:
    """ "Give me 262144, but not at the cost of the cache" is a sayable thing."""
    manager, _supervisor = make_manager(free_gib=17.0, n_ctx_train=262144)
    plain = await manager.load_recommended(MODEL, 262144, prefer_modes=["dual_5090"])
    assert plain.plan is not None
    assert plain.plan.kv_cache_type != "f16", "fixture must need a quantized cache here"

    # A fresh box for the second call: the stub probe does not lose free VRAM
    # when a model loads, so a resident model's credited footprint (below)
    # would otherwise hand this call phantom room.
    manager, _supervisor = make_manager(free_gib=17.0, n_ctx_train=262144)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144, prefer_modes=["dual_5090"], kv_min="f16")
    reasons = [entry["reason"] or "" for entry in excinfo.value.details["modes"]]
    assert any("below the f16 minimum" in reason for reason in reasons)


async def test_kv_min_walks_on_to_a_mode_that_can_afford_the_cache() -> None:
    """A minimum is a constraint on the answer, not a reason to give up early."""
    manager, _supervisor = make_manager(free_gib=17.0, n_ctx_train=262144)
    instance = await manager.load_recommended(MODEL, 262144, kv_min="f16")
    assert instance.plan is not None
    assert instance.plan.kv_cache_type == "f16"
    assert instance.plan.ctx_size == 262144
    assert len(instance.plan.devices) > 2, "it had to reach for more cards to afford f16"


# ---------------------------------------------------------------------------
# The model is already resident (WP22)
# ---------------------------------------------------------------------------


def resident_self(
    *, ctx: int, requests: int = 0, devices: list[int] | None = None, parallel: int = 1
) -> InstanceInfo:
    devs = devices or [0, 1]
    return InstanceInfo(
        model_id=MODEL,
        state="ready",
        port=18101,
        ttl_s=1800,
        started_at=1.0,
        last_activity_at=1.0,
        active_requests=requests,
        loaded_by="mcp:load_model",
        plan=LoadPlan(
            model_id=MODEL,
            devices=devs,
            ctx_size=ctx,
            ctx_per_slot=ctx,
            parallel=parallel,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            per_gpu_bytes={d: int(29 * GB) for d in devs},
        ),
    )


async def test_a_resident_model_that_is_serving_is_not_reloaded_under_its_clients() -> None:
    """A load never interrupts a stream (D36) -- this one included.

    Before: ``load_recommended`` always forced a reload, so agent B asking for
    a bigger window killed agent A's stream on the same model.
    """
    manager, supervisor = make_manager(loaded=[resident_self(ctx=32768, requests=2)])
    with pytest.raises(ModelBusyError) as excinfo:
        await manager.load_recommended(MODEL, 65536)
    assert excinfo.value.details["retry_after_s"] == BUSY_RETRY_AFTER_S
    assert excinfo.value.details["busy"]["busy_models"] == [
        {"model_id": MODEL, "active_requests": 2}
    ]
    assert supervisor.starts == 0
    assert supervisor.stopped == []


async def test_a_resident_model_already_at_that_context_is_returned_as_is() -> None:
    """Reloading to arrive exactly where we are costs a cold start for nothing."""
    manager, supervisor = make_manager(loaded=[resident_self(ctx=65536)])
    instance = await manager.load_recommended(MODEL, 65536)
    assert instance.plan is not None
    assert instance.plan.ctx_per_slot == 65536
    assert supervisor.starts == 0
    assert supervisor.stopped == []


async def test_the_resident_early_return_re_tiers_the_instance_it_hands_back() -> None:
    """``load_recommended(priority=1)`` on a model that is already at that
    context must re-tier it exactly as ``load(priority=1)`` does on its ready
    instance -- the planner's candidacy and the effective-tier reads all live
    on the instance, so a tier that stopped at the memo was a silent no-op on
    this one route. The TTL is restamped with it, so the persist below prices
    the tier that was just asked for rather than the stale one.
    """
    manager, supervisor = make_manager(loaded=[resident_self(ctx=65536)])
    manager.config.models.ttl_by_priority = {1: 3600, 3: 900}
    resident = supervisor.instances[MODEL]
    assert resident.priority == PRIORITY_BACKGROUND

    instance = await manager.load_recommended(MODEL, 65536, priority=PRIORITY_CHAT)

    assert instance is resident
    assert supervisor.starts == 0, "still the early return -- no cold start to re-tier"
    assert instance.priority == PRIORITY_CHAT
    assert instance.ttl_s == 3600, "the tier's TTL, not the one the old tier priced"


async def test_the_resident_early_return_respects_the_slot_ceiling() -> None:
    """An eight-slot resident is not the answer to "load this capped at two
    slots" -- and with persist it would have stored the eight. An over-cap
    resident falls through to a real reload, which is what max_slots asked
    for."""
    manager, supervisor = make_manager(loaded=[resident_self(ctx=16384, parallel=8)])

    instance = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"], max_slots=2)

    assert supervisor.starts == 1, "the resident was over the ceiling, so it was reloaded"
    assert instance.plan is not None
    assert instance.plan.parallel <= 2


async def test_a_resident_model_is_credited_with_its_own_footprint_for_the_walk() -> None:
    """The walk must see the machine the reload will see (D30's credit).

    The model holds 29 GiB on each 5090 and the cards show 2 GiB free. Without
    the credit, 131072 "does not fit" and the call refused a window the reload
    one line later would have fitted. With it, the reload is planned on the
    same cards and the old instance is the one stopped.
    """
    manager, supervisor = make_manager(
        probe=lopsided_rig(), n_ctx_train=262144, loaded=[resident_self(ctx=32768)]
    )
    instance = await manager.load_recommended(MODEL, 131072, prefer_modes=["dual_5090"])
    assert instance.plan is not None
    assert instance.plan.ctx_per_slot == 131072
    assert sorted(instance.plan.devices) == [0, 1]
    assert supervisor.starts == 1
    assert supervisor.stopped == [MODEL]


async def test_the_real_load_never_evicts_a_busy_model_even_though_it_is_forced() -> None:
    """``force=True`` is the reload half only; ``evict_busy`` is passed False.

    Pinned at the manager boundary: whatever the mode walk decided, the plan
    that actually runs is made with ``evict_busy=False``.
    """
    seen: list[dict[str, Any]] = []
    manager, _supervisor = make_manager()
    real_plan_load = manager.planner.plan_load

    def spy(record: Any, **kwargs: Any) -> Any:
        seen.append(dict(kwargs))
        return real_plan_load(record, **kwargs)

    manager.planner.plan_load = spy  # type: ignore[method-assign]
    await manager.load_recommended(MODEL, 32768)
    assert seen, "the final load plans through the manager's planner"
    assert all(call.get("evict_busy") is False for call in seen)


async def test_a_model_without_metadata_is_a_structured_refusal_not_a_traceback() -> None:
    """No GGUF metadata -> no trained window, no KV geometry -> a 502 that says so."""
    from studioforge.errors import ModelLoadError

    rec = record(MODEL, None, mtime=1.0, size_bytes=8 * GB)
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(), log_plans=False)
    manager = ModelManager(
        config,
        registry=Registry([rec]),  # type: ignore[arg-type]
        planner=planner,
        supervisor=StubSupervisor(),  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    with pytest.raises(ModelLoadError) as excinfo:
        await manager.load_recommended(MODEL, 65536)
    assert excinfo.value.status_code == 502
    assert "metadata" in excinfo.value.message


async def test_an_excluded_card_is_never_chosen_even_though_the_walk_pins_devices() -> None:
    """Found live on a scratch server (WP22): planner.excluded_devices [0, 1] and
    load_recommended put the model on CUDA 0,1 -- the mode walk plans through a
    device_override, and an override beats an exclusion. The modes themselves
    must not contain the card."""
    manager, _supervisor = make_manager()
    manager.config.planner.excluded_devices = [0, 1]
    instance = await manager.load_recommended(MODEL, 16384)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"])
    assert excinfo.value.param == "prefer_modes"


async def test_every_card_excluded_is_a_structured_refusal() -> None:
    manager, _supervisor = make_manager()
    manager.config.planner.excluded_devices = [0, 1, 2, 3]
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 16384)
    assert any("excluded_devices" in s for s in excinfo.value.details["suggestions"])


# ---------------------------------------------------------------------------
# max_slots: a per-call ceiling on what the estimator may choose (D48)
# ---------------------------------------------------------------------------


async def test_max_slots_caps_the_slot_count_this_call_may_choose() -> None:
    """The estimator answers what a placement *could* sustain. A caller that
    knows it has one bot does not want the other slots' KV cache priced into
    the fit, and the cap is applied before the descent loop verifies it -- so
    the winning plan really is planned at the capped count rather than planned
    larger and launched smaller."""
    manager, _supervisor = make_manager()
    free = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"])
    assert free.plan is not None
    assert free.plan.parallel > 1, "fixture must leave room for more than one slot"

    # A fresh box for the second call, or the resident-at-that-context early
    # return hands back the plan the first call made.
    manager, _supervisor = make_manager()
    capped = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"], max_slots=1)
    assert capped.plan is not None
    assert capped.plan.parallel == 1


async def test_a_max_slots_below_one_is_a_400_before_anything_is_planned() -> None:
    manager, supervisor = make_manager()
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 16384, max_slots=0)
    assert excinfo.value.param == "max_slots"
    assert supervisor.starts == 0


async def test_max_slots_above_what_fits_changes_nothing() -> None:
    """A ceiling is a ceiling: it may never talk the planner *up* into slots
    the placement cannot hold."""
    manager, _supervisor = make_manager()
    uncapped = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"])
    manager, _supervisor = make_manager()
    generous = await manager.load_recommended(
        MODEL, 16384, prefer_modes=["dual_5090"], max_slots=999
    )
    assert uncapped.plan is not None and generous.plan is not None
    assert generous.plan.parallel == uncapped.plan.parallel


# ---------------------------------------------------------------------------
# persist: freeze what actually launched into the model's settings (D48)
# ---------------------------------------------------------------------------


PERSONA = "pub/dense-8b-persona"


def preset_manager() -> tuple[ModelManager, StubSupervisor]:
    """A preset-only virtual model over ``MODEL``, which shares its instance."""
    base = record(MODEL, dense_meta(131072), mtime=1.0, size_bytes=8 * GB)
    persona = base.model_copy(
        update={
            "id": PERSONA,
            "name": "persona",
            "is_virtual": True,
            "base_model_id": MODEL,
            "preset": VirtualPreset(temperature=0.4),
            "settings": ModelSettings(),
        }
    )
    config = make_config()
    supervisor = StubSupervisor()
    manager = ModelManager(
        config,
        registry=Registry([base, persona]),  # type: ignore[arg-type]
        planner=Planner(config, rig_5090x2_3090x2(31.0), log_plans=False),
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )
    return manager, supervisor


async def test_persist_writes_the_profile_that_actually_launched() -> None:
    """What is written is ``instance.plan`` -- what the load really did -- not
    what the mode walk hoped for; the two differ whenever the planner had to
    step down on the way in."""
    manager, _supervisor = make_manager()
    refreshed: list[str] = []
    real_refresh = manager.refresh_ttl

    def spy(model_id: str) -> int | None:
        refreshed.append(model_id)
        return real_refresh(model_id)

    manager.refresh_ttl = spy  # type: ignore[method-assign]

    instance = await manager.load_recommended(
        MODEL, 16384, prefer_modes=["dual_5090"], persist=True
    )
    assert instance.plan is not None
    plan = instance.plan

    assert [model_id for model_id, _ in manager.registry.saved] == [MODEL]
    saved = manager.registry.saved[-1][1]
    # settings.ctx_size is PER SLOT while a plan's ctx_size can be the total;
    # writing the total would multiply a multi-slot model's ask on its next load.
    assert saved.ctx_size == 16384
    assert saved.ctx_size == int(plan.ctx_per_slot or plan.ctx_size)
    assert saved.kv_cache_type == plan.kv_cache_type
    assert saved.kv_cache_type_v == plan.kv_cache_type_v
    assert saved.parallel == plan.parallel
    assert saved.priority is None, "no tier was asked for, so none was invented"
    # ...and the write bites the resident instance, not only the next load.
    assert refreshed == [MODEL]


async def test_persist_records_the_tier_the_caller_explicitly_asked_for() -> None:
    manager, _supervisor = make_manager()
    await manager.load_recommended(MODEL, 16384, priority=PRIORITY_CHAT, persist=True)
    assert manager.registry.saved[-1][1].priority == PRIORITY_CHAT


async def test_persist_without_a_tier_leaves_the_saved_one_exactly_as_it_was() -> None:
    """Only an explicit ask is written. The RESOLVED tier can come from the
    in-memory memo -- what some other client is doing right now -- or be the
    implicit background default materialised into the row as though somebody
    had chosen it; persisting a context profile must not do either.
    """
    manager, _supervisor = make_manager(settings=ModelSettings(priority=PRIORITY_CHAT))
    # A memo from somebody else's load, which the resolver would otherwise win.
    manager._model_priority[MODEL] = PRIORITY_BACKGROUND

    await manager.load_recommended(MODEL, 16384, persist=True)

    saved = manager.registry.saved[-1][1]
    assert saved.ctx_size == 16384, "the profile is still written"
    assert saved.priority == PRIORITY_CHAT, "the row's own tier, untouched"


async def test_persist_leaves_the_placement_out_of_the_saved_settings() -> None:
    """A placement is a one-shot load argument (D36) -- the cards that happened
    to be free this minute are not a standing property of the model, and
    freezing them would strand it the next time the box changes."""
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(
        MODEL, 16384, prefer_modes=["dual_3090"], persist=True
    )
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]
    saved = manager.registry.saved[-1][1]
    assert saved.device_override is None
    assert saved.allowed_devices is None


async def test_a_load_without_persist_writes_nothing() -> None:
    manager, _supervisor = make_manager()
    await manager.load_recommended(MODEL, 16384)
    assert manager.registry.saved == []


async def test_persist_still_writes_when_the_model_is_already_at_that_context() -> None:
    """That it was already resident at the profile is not a reason to skip the
    write, or persist would be a no-op precisely when it is run twice."""
    manager, supervisor = make_manager(loaded=[resident_self(ctx=65536)])
    await manager.load_recommended(MODEL, 65536, persist=True)
    assert supervisor.starts == 0, "the early return: no reload to arrive where we already are"
    assert manager.registry.saved[-1][1].ctx_size == 65536


async def test_persist_is_refused_while_the_model_is_being_benchmarked() -> None:
    """A benchmark rewrites these very fields per mode and restores them at the
    end, so a write landing mid-run reaches SQLite and is then reverted in
    memory -- the same clobber hazard the settings routes guard."""
    manager, supervisor = make_manager()
    manager.benchmarker = SimpleNamespace(benchmarking=MODEL)

    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 16384, persist=True)

    assert excinfo.value.param == "persist"
    assert excinfo.value.code == "model_benchmarking"
    assert supervisor.starts == 0, "checked before the walk, so nothing was loaded"
    assert manager.registry.saved == []


class RecordingLog:
    """Stands in for the manager module's structlog logger.

    Neither ``caplog`` nor ``structlog.testing.capture_logs`` holds across test
    orders here: the unit suite leaves structlog unconfigured (so nothing is
    bound to stdlib logging for ``caplog`` to see), and once another test file
    has called ``configure_logging`` the loggers are cached and root's handlers
    -- pytest's capture handler included -- have been replaced. Swapping the
    module logger is the one capture that works either way.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


async def test_a_save_refused_after_the_load_succeeded_is_skipped_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reachable version of this: a model already resident at the requested
    context, on a plan whose slot count predates a ``max_parallel_cap`` lowered
    afterwards. The early return hands that plan back, and the settings
    validation will not store ``parallel`` above the cap.

    Persist is bookkeeping *after* a load that already succeeded -- the model
    IS loaded -- so the refusal is a warning and a skipped write, the same
    answer as a benchmark that starts mid-load. Raising here reported failure
    for a call that had done its job.
    """
    manager, supervisor = make_manager(
        settings=ModelSettings(max_parallel_cap=1),
        loaded=[resident_self(ctx=65536, parallel=4)],
    )

    recorder = RecordingLog()
    monkeypatch.setattr(manager_module, "log", recorder)

    instance = await manager.load_recommended(MODEL, 65536, persist=True)

    assert instance.plan is not None
    assert instance.plan.parallel == 4, "the resident is returned as it stands"
    assert supervisor.starts == 0
    assert manager.registry.saved == [], "the write was refused, so nothing was stored"
    # ...but not silently.
    assert recorder.warnings, "a skipped persist must leave a trail"
    assert any(fields.get("model_id") == MODEL for _event, fields in recorder.warnings)
    assert any(
        "max_parallel_cap" in str(fields.get("error", "")) for _event, fields in recorder.warnings
    )


async def test_a_record_deleted_during_the_walk_is_a_skip_not_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``save_settings`` raises more than the cap refusal: the model can be
    deleted out from under a walk that takes minutes (``ModelNotFoundError``),
    and the storage layer underneath it can raise anything at all. The helper
    is best-effort by contract, so *any* save-time failure is a logged skip --
    catching only ``BadRequestError`` turned a load that had already succeeded
    into an error the caller could do nothing about.
    """
    from studioforge.errors import ModelNotFoundError

    manager, supervisor = make_manager()

    def gone(model_id: str, settings: ModelSettings) -> Any:
        raise ModelNotFoundError(model_id, known=[])

    monkeypatch.setattr(manager.registry, "save_settings", gone)
    recorder = RecordingLog()
    monkeypatch.setattr(manager_module, "log", recorder)

    instance = await manager.load_recommended(MODEL, 16384, persist=True)

    assert instance.state == "ready", "the load itself succeeded and is reported as such"
    assert supervisor.starts == 1
    assert any(fields.get("model_id") == MODEL for _event, fields in recorder.warnings)


async def test_a_raw_storage_failure_during_persist_is_a_skip_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DB layer below the registry raises plain ``sqlite3`` errors, which
    are not ``StudioForgeError`` at all. Same contract, same answer."""
    manager, supervisor = make_manager()

    def boom(model_id: str, settings: ModelSettings) -> Any:
        raise RuntimeError("database is locked")

    monkeypatch.setattr(manager.registry, "save_settings", boom)
    recorder = RecordingLog()
    monkeypatch.setattr(manager_module, "log", recorder)

    instance = await manager.load_recommended(MODEL, 16384, persist=True)

    assert instance.state == "ready"
    assert any(
        "database is locked" in str(fields.get("error", "")) for _event, fields in recorder.warnings
    )


async def test_persist_on_a_preset_refuses_and_names_the_base_model() -> None:
    """A preset-only virtual model serves on its BASE's instance, so the
    resolved profile describes the base -- persisting it here would rewrite the
    base's settings and every other persona sharing it."""
    manager, supervisor = preset_manager()

    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(PERSONA, 16384, persist=True)

    assert excinfo.value.param == "persist"
    assert MODEL in excinfo.value.message
    assert excinfo.value.details == {
        "requested_model_id": PERSONA,
        "serving_model_id": MODEL,
    }
    assert supervisor.starts == 0
    assert manager.registry.saved == []

    # The same call without persist still loads, on the base's instance.
    instance = await manager.load_recommended(PERSONA, 16384)
    assert instance.model_id == MODEL
