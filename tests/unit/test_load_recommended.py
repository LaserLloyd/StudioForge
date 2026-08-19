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

from typing import Any

import pytest

from studioforge.core.manager import ModelManager
from studioforge.core.planner import BUSY_RETRY_AFTER_S, Planner
from studioforge.errors import BadRequestError, InsufficientVramError
from studioforge.types import GB, InstanceInfo, LoadPlan, ModelSettings
from tests.unit.test_catalog import dense_meta, record
from tests.unit.test_load_retry import StubSupervisor
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

MODEL = "pub/dense-8b"


class Registry:
    def __init__(self, records: list[Any]) -> None:
        self._records = {r.id: r for r in records}

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
