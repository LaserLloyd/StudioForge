"""Two slots over one KV pool: ``min_slots`` and ``kv_unified`` on load-recommended (D72).

The ask that made these: a companion-chat client wants its chat and its
background requests (titles, memory, summaries) on different slots, so a
background request can never push the chat's cached prompt out -- and wants
that second slot for about the price of none. ``--kv-unified`` gives exactly
that: one pool of ``--ctx-size`` cells shared by every slot, each slot allowed
the whole of it.

What these pin:

* **pricing** -- a pool of ``ctx`` at N slots is charged ONE pool of ``ctx``
  cells plus what really scales with the slot count: each extra slot's
  sliding window (``window * N + ubatch`` cells, as llama.cpp sizes a unified
  SWA cache -- ~0.8 GiB at f16 for a Gemma-4-shaped 31B) and each extra
  slot's recurrent state; a MoE's attention mask spans the pool, which is
  D38's measured 997 -> 1005 MiB;
* **the walk** -- ``min_slots`` is a floor the fit is judged at, a refusal
  names ``max_ctx_that_fits`` and ``max_parallel_that_fits``, and the pool
  loads where partitioned slots of the same window cannot;
* **the shortcut** -- a resident is "already exactly that" only at or above
  the floor and in the pool shape the caller named, if it named one;
* **dry-run parity**, the **launch** (``--ctx-size ctx --parallel N
  --kv-unified --no-cache-idle-slots``), the ``effective`` report, and every
  path that replays a plan (restore, rebalance, OOM retry);
* **D51** -- a pool's measurement and a partitioned one never stand in for
  each other;
* the surfaces: REST, MCP, ``/v1/models``, sfctl.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.config import Config
from studioforge.core.manager import ModelManager, _RestoreEntry, reload_settings
from studioforge.core.planner import (
    OBSERVATION_KV_UNIFIED_KEY,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    Planner,
    kq_mask_bytes,
    recurrent_state_bytes_per_slot,
)
from studioforge.core.priority import PRIORITY_CHAT
from studioforge.core.supervisor import Supervisor, effective_launch
from studioforge.db import OBSERVATION_KV_UNIFIED_KEY as DB_KV_UNIFIED_KEY
from studioforge.db import Database
from studioforge.errors import BadRequestError, InsufficientVramError
from studioforge.types import GB, MB, InstanceInfo, LoadPlan, ModelSettings
from tests.unit.test_catalog import dense_meta, hybrid_meta, iswa_meta, record
from tests.unit.test_catalog_routes import MODEL_ID, app, loaded  # noqa: F401 - the fixture
from tests.unit.test_catalog_routes import make_plan as routes_plan
from tests.unit.test_load_recommended import MODEL, ParallelDb, make_manager
from tests.unit.test_load_retry import OOM_STDERR, StubPlanner, StubSupervisor, resident
from tests.unit.test_load_retry import make_manager as retry_manager
from tests.unit.test_planner import make_config, rig_5090x2_3090x2
from tests.unit.test_planner_moe_compute import _meta as moe_meta
from tests.unit.test_planner_moe_compute import _record as moe_record
from tests.unit.test_planner_observed import db, lookup_key, observation  # noqa: F401
from tests.unit.test_supervisor import make_binary, resolver
from tests.unit.test_supervisor_features import B10425

#: The Gemma-4-shaped fixture's sliding-window cost per cell, all window
#: layers: 50 of its 60 layers slide, each holding 16 KV heads of 256 (K) +
#: 256 (V) at f16.
ISWA_SWA_BYTES_PER_CELL = 50 * 16 * (256 + 256) * 2


def _estimate(
    rec: Any, *, ctx: int, parallel: int, unified: bool = False, n_devices: int = 1
) -> Any:
    return Planner(make_config(), rig_5090x2_3090x2()).estimate(
        rec,
        ctx_size=ctx,
        parallel=parallel,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=n_devices,
        ubatch=512,
        kv_unified=unified,
    )


def _dense() -> Any:
    return record("pub/dense-8b", dense_meta(262144), mtime=1.0, size_bytes=8 * GB)


# ---------------------------------------------------------------------------
# Pricing: one pool, plus what really scales with the slots
# ---------------------------------------------------------------------------


def test_a_dense_pool_of_two_slots_costs_what_one_slot_does() -> None:
    """Full attention holds ``--ctx-size`` cells whatever the slot count: the
    second slot of a pool is free, where a partitioned second slot doubles the
    cache."""
    rec = _dense()
    one = _estimate(rec, ctx=65536, parallel=1)
    pool = _estimate(rec, ctx=65536, parallel=2, unified=True)
    split = _estimate(rec, ctx=65536, parallel=2)
    assert pool.kv_bytes == one.kv_bytes
    assert pool.total_bytes == one.total_bytes
    assert split.kv_bytes == 2 * one.kv_bytes


def test_the_second_slot_of_a_sliding_window_pool_costs_its_window() -> None:
    """llama.cpp sizes a unified SWA cache GGML_PAD(window * n_seq + ubatch):
    2,560 cells at two slots against 1,536 at one -- 1,024 more cells over the
    50 window layers, 800 MiB at f16 for the Gemma-4-shaped 31B. Derived from
    the model's own geometry, not a constant."""
    rec = record("pub/gemma4-31b", iswa_meta(), mtime=1.0, size_bytes=30 * GB)
    one = _estimate(rec, ctx=200000, parallel=1)
    pool = _estimate(rec, ctx=200000, parallel=2, unified=True)
    assert pool.kv_bytes - one.kv_bytes == 1024 * ISWA_SWA_BYTES_PER_CELL == 800 * MB


def test_a_partitioned_pair_of_the_same_window_pays_the_full_layers_twice() -> None:
    rec = record("pub/gemma4-31b", iswa_meta(), mtime=1.0, size_bytes=30 * GB)
    pool = _estimate(rec, ctx=200000, parallel=2, unified=True)
    split = _estimate(rec, ctx=200000, parallel=2)
    # Ten full-attention layers, 4 KV heads of 512 + 512, f16: one more pool.
    full_layers_one_window = 10 * 4 * (512 + 512) * 2 * 200000
    assert split.kv_bytes - pool.kv_bytes == full_layers_one_window


def test_the_second_slot_of_a_hybrid_pool_costs_its_recurrent_state() -> None:
    rec = record("pub/qwen35-27b", hybrid_meta(), mtime=1.0, size_bytes=20 * GB)
    one = _estimate(rec, ctx=131072, parallel=1)
    pool = _estimate(rec, ctx=131072, parallel=2, unified=True)
    assert pool.kv_bytes - one.kv_bytes == recurrent_state_bytes_per_slot(rec.meta) > 0


def test_a_pool_mask_is_the_difference_d38_measured() -> None:
    """D38: --parallel 2 --ctx-size 16384 measured 997 MiB partitioned and 1005
    unified -- the same cells, and a mask spanning 16384 instead of 8192 per
    stream: (16384 - 8192) x 512 x 2 bytes = 8 MiB. The MoE compute term
    (D71) charges exactly that."""
    rec = moe_record(moe_meta())
    split = _estimate(rec, ctx=8192, parallel=2)  # --ctx-size 16384, partitioned
    pool = _estimate(rec, ctx=16384, parallel=2, unified=True)  # --ctx-size 16384, unified
    assert pool.kv_bytes == split.kv_bytes
    assert pool.compute_bytes - split.compute_bytes == 8 * MB
    assert kq_mask_bytes(16384, 512) - kq_mask_bytes(8192, 512) == 8 * MB


def test_a_pool_holds_more_slots_than_its_partitioned_twin() -> None:
    """The exact VRAM walk prices a pool as a pool: with room for one 64k
    window and a little more, partitioned slots stop at one and a pool of the
    dense model's reaches the cap."""
    rec = _dense()
    planner = Planner(make_config(), rig_5090x2_3090x2())
    capacity = _estimate(rec, ctx=65536, parallel=1).total_bytes + 2 * GB
    split, _ = planner.max_slots_by_vram(
        rec, ctx=65536, kv_k="f16", kv_v="f16", n_devices=1, capacity_bytes=capacity, cap=8
    )
    pool, _ = planner.max_slots_by_vram(
        rec,
        ctx=65536,
        kv_k="f16",
        kv_v="f16",
        n_devices=1,
        capacity_bytes=capacity,
        cap=8,
        kv_unified=True,
    )
    assert split == 1
    assert pool == 8


def test_without_the_flag_every_estimate_is_the_one_it_was() -> None:
    """Byte-identical for every caller that does not ask (the per-model
    ``kv_unified`` switch keeps its older meaning: ctx per slot, one pool)."""
    rec = _dense()
    planner = Planner(make_config(), rig_5090x2_3090x2())
    for parallel in (1, 2, 4):
        default = planner.estimate(
            rec, ctx_size=32768, parallel=parallel, kv_cache_type="f16", kv_cache_type_v="f16"
        )
        explicit = planner.estimate(
            rec,
            ctx_size=32768,
            parallel=parallel,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            kv_unified=False,
        )
        assert default == explicit
        assert default.kv_bytes == 36 * 8 * (128 + 128) * 2 * 32768 * parallel


# ---------------------------------------------------------------------------
# The walk: min_slots is a floor, kv_unified a pool
# ---------------------------------------------------------------------------


#: Free GiB per card that leaves ~20 GiB usable across the 5090 pair: one
#: 131072-token window of the dense 8B fits (on a q8_0 cache), a partitioned
#: pair does not fit on any rung of the KV ladder -- not even q8_0 K + q4_0 V
#: -- and a two-slot pool costs exactly the one window.
ONE_WINDOW_FREE_GIB = 13.2


async def test_two_slots_over_one_pool_load_where_a_partitioned_pair_cannot() -> None:
    """The ClawChat ask -- min_slots 2, max_slots 2, kv_unified -- on cards
    that hold one 131072 window: the pool loads at exactly two slots, and the
    same floor partitioned is an honest 507 naming what would fit."""
    manager, supervisor = make_manager(free_gib=ONE_WINDOW_FREE_GIB, n_ctx_train=262144)

    with pytest.raises(InsufficientVramError) as refused:
        await manager.load_recommended(
            MODEL, 131072, min_slots=2, max_slots=2, prefer_modes=["dual_5090"]
        )
    details = refused.value.details
    assert supervisor.starts == 0
    assert details["max_parallel_that_fits"] == 1, "one partitioned slot does fit"
    assert 0 < details["max_ctx_that_fits"] < 131072, "the window two slots would get"
    assert details["min_slots"] == 2 and details["kv_unified"] is False
    assert "with at least 2 slots" in refused.value.message
    assert any("ask for fewer slots" in s for s in details["suggestions"])

    instance = await manager.load_recommended(
        MODEL, 131072, min_slots=2, max_slots=2, kv_unified=True, prefer_modes=["dual_5090"]
    )
    plan = instance.plan
    assert plan is not None
    assert plan.parallel == 2
    assert plan.kv_unified is True
    assert plan.ctx_size == plan.ctx_per_slot == plan.ctx_total == 131072
    assert supervisor.starts == 1


async def test_min_slots_without_a_pool_is_partitioned_slots_of_ctx_each() -> None:
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(
        MODEL, 32768, min_slots=2, max_slots=2, prefer_modes=["dual_5090"]
    )
    plan = instance.plan
    assert plan is not None
    assert plan.parallel == 2 and plan.kv_unified is False
    assert plan.ctx_per_slot == 32768 and plan.ctx_total == 65536


async def test_min_slots_raises_a_measured_one_slot_recommendation() -> None:
    """The estimator's pick stands unless it is below the floor; a sweep that
    said "one slot is worth running" does not beat a caller that needs two."""
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
    measured = await manager.plan_recommended(MODEL, 16384, prefer_modes=["dual_5090"])
    assert measured["parallel"] == 1, "the sweep's own answer, untouched without a floor"
    raised = await manager.load_recommended(MODEL, 16384, prefer_modes=["dual_5090"], min_slots=2)
    assert raised.plan is not None and raised.plan.parallel == 2


async def test_a_pool_without_a_floor_takes_the_estimators_count() -> None:
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(
        MODEL, 32768, kv_unified=True, prefer_modes=["dual_5090"]
    )
    plan = instance.plan
    assert plan is not None and plan.kv_unified is True
    assert plan.parallel >= 1
    assert plan.ctx_total == 32768


async def test_without_the_new_fields_the_load_is_the_one_it_was() -> None:
    manager, _supervisor = make_manager()
    instance = await manager.load_recommended(MODEL, 32768, prefer_modes=["dual_5090"])
    plan = instance.plan
    assert plan is not None
    assert plan.kv_unified is False
    assert plan.ctx_total == 32768 * plan.parallel
    dry = await manager.plan_recommended(MODEL, 32768, prefer_modes=["dual_5090"])
    assert dry["min_slots"] is None
    assert dry["kv_unified"] is False


# ---------------------------------------------------------------------------
# Dry-run parity
# ---------------------------------------------------------------------------


async def test_the_dry_run_names_the_pool_the_real_call_then_loads() -> None:
    manager, supervisor = make_manager(free_gib=20.0, n_ctx_train=262144)
    ask: dict[str, Any] = {
        "min_slots": 2,
        "max_slots": 2,
        "kv_unified": True,
        "priority": PRIORITY_CHAT,
    }
    dry = await manager.plan_recommended(MODEL, 131072, **ask)
    assert supervisor.starts == 0
    assert dry["fits"] is True and dry["already_loaded"] is False
    assert dry["kv_unified"] is True
    assert dry["parallel"] == 2
    assert dry["ctx_size"] == dry["ctx_per_slot"] == dry["ctx_total"] == 131072
    assert dry["min_slots"] == 2
    instance = await manager.load_recommended(MODEL, 131072, **ask)
    plan = instance.plan
    assert plan is not None
    assert sorted(plan.devices) == sorted(dry["devices"])
    assert plan.parallel == dry["parallel"]
    assert plan.kv_unified is dry["kv_unified"]
    assert plan.ctx_total == dry["ctx_total"]
    assert (plan.kv_cache_type, plan.kv_cache_type_v) == (
        dry["kv_cache_type"],
        dry["kv_cache_type_v"],
    )


async def test_the_dry_run_refuses_a_floor_exactly_as_the_real_call_does() -> None:
    manager, _supervisor = make_manager(free_gib=ONE_WINDOW_FREE_GIB, n_ctx_train=262144)
    ask: dict[str, Any] = {"min_slots": 2, "max_slots": 2, "prefer_modes": ["dual_5090"]}
    dry = await manager.plan_recommended(MODEL, 131072, **ask)
    with pytest.raises(InsufficientVramError) as real:
        await manager.load_recommended(MODEL, 131072, **ask)
    assert dry["fits"] is False
    assert dry["status_code"] == real.value.status_code == 507
    assert dry["message"] == real.value.message
    assert dry["max_ctx_that_fits"] == real.value.details["max_ctx_that_fits"]
    assert dry["max_parallel_that_fits"] == real.value.details["max_parallel_that_fits"] == 1
    assert dry["modes"][0]["max_parallel_that_fits"] == 1


# ---------------------------------------------------------------------------
# The "already exactly that" shortcut
# ---------------------------------------------------------------------------


def resident_pool(*, ctx: int, parallel: int, unified: bool, requests: int = 0) -> InstanceInfo:
    return InstanceInfo(
        model_id=MODEL,
        state="ready",
        port=18101,
        ttl_s=1800,
        started_at=1.0,
        last_activity_at=1.0,
        active_requests=requests,
        loaded_by="api:/api/models/{id}/load-recommended",
        plan=LoadPlan(
            model_id=MODEL,
            devices=[0, 1],
            ctx_size=ctx,
            ctx_per_slot=ctx,
            parallel=parallel,
            kv_unified=unified,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            per_gpu_bytes={0: int(12 * GB), 1: int(12 * GB)},
        ),
    )


async def test_a_resident_pool_is_exactly_the_pool_asked_for_even_while_serving() -> None:
    """ClawChat asks before every send; a pool serving the other slot's
    request is still exactly what it asked for, and nothing is interrupted."""
    manager, supervisor = make_manager(
        loaded=[resident_pool(ctx=65536, parallel=2, unified=True, requests=1)]
    )
    instance = await manager.load_recommended(
        MODEL, 65536, min_slots=2, max_slots=2, kv_unified=True
    )
    assert instance.active_requests == 1, "the same, still-serving instance"
    assert supervisor.starts == 0 and supervisor.stopped == []


async def test_a_one_slot_resident_is_not_exactly_an_ask_for_two() -> None:
    manager, supervisor = make_manager(loaded=[resident_pool(ctx=65536, parallel=1, unified=False)])
    instance = await manager.load_recommended(
        MODEL, 65536, min_slots=2, max_slots=2, kv_unified=True, prefer_modes=["dual_5090"]
    )
    assert supervisor.starts == 1, "reloaded into the pool"
    assert supervisor.stopped == [MODEL]
    assert instance.plan is not None and instance.plan.parallel == 2 and instance.plan.kv_unified


async def test_a_partitioned_pair_is_not_exactly_a_pool_nor_the_reverse() -> None:
    manager, supervisor = make_manager(loaded=[resident_pool(ctx=32768, parallel=2, unified=False)])
    await manager.load_recommended(
        MODEL, 32768, min_slots=2, max_slots=2, kv_unified=True, prefer_modes=["dual_5090"]
    )
    assert supervisor.starts == 1

    manager, supervisor = make_manager(loaded=[resident_pool(ctx=32768, parallel=2, unified=True)])
    instance = await manager.load_recommended(
        MODEL, 32768, min_slots=2, max_slots=2, kv_unified=False, prefer_modes=["dual_5090"]
    )
    assert supervisor.starts == 1
    assert instance.plan is not None and instance.plan.kv_unified is False


async def test_a_caller_that_names_no_shape_takes_the_resident_pool_as_it_is() -> None:
    """Every pre-D72 caller keeps its answer: a pool gives each conversation
    the window it asks for, and reshaping it for a caller with no opinion would
    only fight the client that asked for it."""
    manager, supervisor = make_manager(loaded=[resident_pool(ctx=65536, parallel=2, unified=True)])
    instance = await manager.load_recommended(MODEL, 65536)
    assert supervisor.starts == 0
    assert instance.plan is not None and instance.plan.kv_unified is True


async def test_persist_from_a_caller_handed_a_resident_pool_writes_nothing() -> None:
    """The saved settings cannot hold a pool (ctx_size is per slot there); a
    caller that named no shape and was handed one back must not store it."""
    manager, _supervisor = make_manager(loaded=[resident_pool(ctx=65536, parallel=2, unified=True)])
    await manager.load_recommended(MODEL, 65536, persist=True)
    assert manager.registry.saved == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The 400s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, param",
    [
        pytest.param({"min_slots": 0}, "min_slots", id="zero slots"),
        pytest.param({"min_slots": 3, "max_slots": 2}, "min_slots", id="floor above ceiling"),
        pytest.param({"min_slots": True}, "min_slots", id="a bool is not a count"),
        pytest.param({"kv_unified": "yes"}, "kv_unified", id="not a bool"),
    ],
)
async def test_a_bad_ask_is_the_same_400_on_the_call_and_its_dry_run(
    kwargs: dict[str, Any], param: str
) -> None:
    manager, supervisor = make_manager()
    for call in (manager.load_recommended, manager.plan_recommended):
        with pytest.raises(BadRequestError) as excinfo:
            await call(MODEL, 32768, **kwargs)
        assert excinfo.value.param == param
    assert supervisor.starts == 0


async def test_a_pool_cannot_be_persisted() -> None:
    """The saved settings keep ctx_size per slot: written back, a two-slot pool
    would come back as two windows the next time the model loads plainly."""
    manager, supervisor = make_manager()
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 32768, kv_unified=True, persist=True)
    assert excinfo.value.param == "persist"
    assert supervisor.starts == 0
    assert manager.registry.saved == []  # type: ignore[attr-defined]


async def test_a_floor_above_the_models_own_cap_is_a_400() -> None:
    manager, supervisor = make_manager(settings=ModelSettings(max_parallel_cap=1))
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 32768, min_slots=2)
    assert excinfo.value.param == "min_slots"
    assert excinfo.value.details["max_parallel_cap"] == 1
    assert supervisor.starts == 0


# ---------------------------------------------------------------------------
# The launch and what it reports
# ---------------------------------------------------------------------------


def _pool_plan(ctx: int = 131072, parallel: int = 2, unified: bool = True) -> LoadPlan:
    return LoadPlan(
        model_id="qwen2.5-7b",
        devices=[0],
        ctx_size=ctx,
        ctx_per_slot=ctx,
        parallel=parallel,
        kv_unified=unified,
    )


def _launch(
    tmp_path: Path, plan: LoadPlan, *, settings: ModelSettings | None = None, features: Any = B10425
) -> list[str]:
    from tests.unit.test_supervisor import make_record as sup_record

    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    supervisor = Supervisor(config, resolve_binary=resolver(make_binary(tmp_path)))
    return supervisor.build_command(
        sup_record(tmp_path, settings=settings), plan, port=18100, features=features
    )


def _value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_a_pool_launches_at_its_own_size_with_the_idle_slots_kept(tmp_path: Path) -> None:
    argv = _launch(tmp_path, _pool_plan())
    assert _value_after(argv, "--ctx-size") == "131072", "the pool, not the pool times two"
    assert _value_after(argv, "--parallel") == "2"
    assert "--kv-unified" in argv
    assert "--no-kv-unified" not in argv
    # With a unified cache the idle-slot snapshot CLEARS the idle chat slot
    # whenever the other slot starts a task (llama.cpp TAG_IDLE_SLOT_CLEAR).
    assert "--no-cache-idle-slots" in argv


def test_a_partitioned_launch_is_unchanged(tmp_path: Path) -> None:
    argv = _launch(tmp_path, _pool_plan(ctx=32768, unified=False))
    assert _value_after(argv, "--ctx-size") == "65536"
    assert "--no-kv-unified" in argv
    assert "--kv-unified" not in argv and "--no-cache-idle-slots" not in argv


def test_the_idle_slot_flag_is_passed_only_where_the_engine_has_it(tmp_path: Path) -> None:
    older = dataclasses.replace(B10425, flags=B10425.flags - {"--no-cache-idle-slots"})
    argv = _launch(tmp_path, _pool_plan(), features=older)
    assert "--kv-unified" in argv and "--no-cache-idle-slots" not in argv


def test_a_one_slot_pool_needs_no_idle_slot_flag(tmp_path: Path) -> None:
    argv = _launch(tmp_path, _pool_plan(parallel=1))
    assert "--kv-unified" in argv and "--no-cache-idle-slots" not in argv


def test_the_older_per_model_switch_keeps_its_meaning_and_gains_the_idle_slot_flag(
    tmp_path: Path,
) -> None:
    """settings.kv_unified: ctx_size per slot, pool = ctx x slots -- and as a
    unified multi-slot launch it must not clear its idle slots either."""
    argv = _launch(
        tmp_path, _pool_plan(ctx=16384, unified=False), settings=ModelSettings(kv_unified=True)
    )
    assert _value_after(argv, "--ctx-size") == "32768"
    assert "--kv-unified" in argv and "--no-cache-idle-slots" in argv


def test_effective_reports_the_whole_pool_per_slot_and_the_idle_slots_kept(
    tmp_path: Path,
) -> None:
    plan = _pool_plan()
    eff = effective_launch(_launch(tmp_path, plan), B10425, plan)
    assert eff.kv_unified is True
    assert eff.parallel == 2
    assert eff.ctx_per_slot == eff.ctx_total == 131072
    assert eff.cache_idle_slots is False and eff.sources["cache_idle_slots"] == "argv"
    assert "2 slots x 131072" in eff.summary
    assert "unified KV (idle slots kept)" in eff.summary


def test_extra_flags_can_put_the_idle_slot_snapshot_back_and_the_report_says_so(
    tmp_path: Path,
) -> None:
    plan = _pool_plan()
    settings = ModelSettings(extra_flags="--cache-idle-slots")
    eff = effective_launch(_launch(tmp_path, plan, settings=settings), B10425, plan, settings)
    assert eff.cache_idle_slots is True
    assert "unified KV (idle slots cleared)" in eff.summary


def test_a_pool_plan_totals_its_pool() -> None:
    assert _pool_plan(ctx=131072, parallel=2).ctx_total == 131072
    assert _pool_plan(ctx=131072, parallel=2, unified=False).ctx_total == 262144


# ---------------------------------------------------------------------------
# Every replay of a plan keeps the pool
# ---------------------------------------------------------------------------


def test_reload_settings_carries_the_pool_and_nothing_new_otherwise() -> None:
    assert reload_settings(_pool_plan()) == {
        "ctx_size": 131072,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "parallel": 2,
        "kv_unified": True,
    }
    assert reload_settings(_pool_plan(unified=False)) == {
        "ctx_size": 131072,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "parallel": 2,
    }


async def test_a_displaced_pool_is_restored_as_a_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """D46: a priority load displaces a recently-active model and reloads it
    afterwards with its old shape; a pool restored partitioned would ask for
    its window once per slot."""
    import time as time_module

    victim = resident_pool(ctx=65536, parallel=2, unified=True)
    victim.last_activity_at = time_module.time()
    manager, _supervisor = make_manager(loaded=[victim])
    manager._note_displaced(manager.registry.resolve(MODEL), MODEL, priority=PRIORITY_CHAT)
    entry = manager._restore_entries[MODEL]
    assert isinstance(entry, _RestoreEntry) and entry.kv_unified is True

    seen: dict[str, Any] = {}

    async def fake_load(model_id: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return None

    manager.supervisor.instances.pop(MODEL)  # type: ignore[attr-defined]
    monkeypatch.setattr(manager, "load", fake_load)
    await manager._restore_evicted()
    assert seen["kv_unified"] is True
    assert seen["ctx_size"] == 65536 and seen["parallel"] == 2


async def test_an_oom_retry_replans_the_pool_not_partitioned_slots() -> None:
    supervisor = StubSupervisor(fail_times=1, stderr=OOM_STDERR)
    supervisor.instances["victim/model"] = resident("victim/model")
    planner = StubPlanner()
    manager = retry_manager(supervisor, planner)

    await manager.load("test/model", ctx_size=65536, parallel=2, kv_unified=True)

    assert supervisor.starts == 2
    assert [call["kv_unified"] for call in planner.kwargs] == [True, True]


async def test_a_forced_reload_asking_for_a_pool_never_folds_onto_another() -> None:
    """D50 folds a forced reload only when it names no shape; a pool is one."""
    manager, _supervisor = make_manager()
    instance = resident_pool(ctx=65536, parallel=2, unified=True)
    instance.spawn_seq = 5
    instance.resolved_engine_tag = "b10425"
    manager._active_engine_tag = lambda: "b10425"  # type: ignore[method-assign]
    assert manager._reload_already_done(
        instance,
        4,
        ctx_size=None,
        kv_cache_type=None,
        kv_cache_type_v=None,
        parallel=None,
        devices=None,
    )
    assert not manager._reload_already_done(
        instance,
        4,
        ctx_size=None,
        kv_cache_type=None,
        kv_cache_type_v=None,
        parallel=None,
        devices=None,
        kv_unified=True,
    )


# ---------------------------------------------------------------------------
# D51: a pool's measurement is its own
# ---------------------------------------------------------------------------


def test_the_pool_marker_is_the_same_word_on_both_sides() -> None:
    assert DB_KV_UNIFIED_KEY == OBSERVATION_KV_UNIFIED_KEY


def test_an_observed_pool_is_marked_and_a_partitioned_load_is_not() -> None:
    rows: list[dict[str, Any]] = []
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_sink=rows.append)
    for unified in (True, False):
        plan = _pool_plan(unified=unified).model_copy(update={"per_gpu_bytes": {0: 20 * GB}})
        planner.observe(
            model_id="m1",
            plan=plan,
            actual_bytes=19 * GB,
            note=OBSERVATION_NOTE_PER_PID_DEVICE,
        )
    pool_row, split_row = (json.loads(r["per_gpu_planned"]) for r in rows)
    assert pool_row[OBSERVATION_KV_UNIFIED_KEY] is True
    assert OBSERVATION_KV_UNIFIED_KEY not in split_row, "absent, so older rows read the same"
    assert planner.last_observation("m1")["kv_unified"] is False


def test_a_pool_row_and_a_partitioned_row_never_stand_in_for_each_other(db: Database) -> None:  # noqa: F811 - the imported fixture
    pool_row = observation(
        actual_bytes=9 * GB, per_gpu_planned=json.dumps({"0": 1, "kv_unified": True})
    )
    db.record_load_observation(**pool_row)
    assert db.matching_observation("m1", **lookup_key()) is None, "not for a partitioned plan"
    found = db.matching_observation("m1", **lookup_key(), kv_unified=True)
    assert found is not None and found["actual_bytes"] == 9 * GB

    db.record_load_observation(**observation(ts=1.0, actual_bytes=5 * GB))  # a legacy row
    assert db.matching_observation("m1", **lookup_key())["actual_bytes"] == 5 * GB
    assert db.matching_observation("m1", **lookup_key(), kv_unified=True)["actual_bytes"] == 9 * GB


def test_the_lookup_is_asked_about_a_pool_only_for_a_pool() -> None:
    """Every partitioned lookup keeps the call shape it always had."""
    seen: list[dict[str, Any]] = []

    def strict_lookup(
        model_id: str,
        *,
        ctx_size: int,
        parallel: int,
        kv_cache_type: str,
        kv_cache_type_v: str,
        device_count: int,
        **extra: Any,
    ) -> None:
        seen.append(dict(extra))

    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=strict_lookup)
    rec = _dense()
    for unified in (False, True):
        planner.plan_load(rec, ctx_size=32768, parallel=2, kv_cache_type="f16", kv_unified=unified)
    assert {} in seen and {"kv_unified": True} in seen
    assert all(extra in ({}, {"kv_unified": True}) for extra in seen)


# ---------------------------------------------------------------------------
# The surfaces
# ---------------------------------------------------------------------------


def test_load_recommended_forwards_the_floor_and_the_pool(app: Any) -> None:  # noqa: F811 - the imported fixture
    calls: list[dict[str, Any]] = []

    async def capture(model_id: str, ctx_size: int, **kwargs: Any) -> InstanceInfo:
        calls.append(kwargs)
        return InstanceInfo(model_id=model_id, state="ready", plan=routes_plan())

    app.state.manager.load_recommended = capture
    with TestClient(app) as http:
        asked = http.post(
            f"/api/models/{MODEL_ID}/load-recommended",
            json={"ctx_size": 65536, "min_slots": 2, "max_slots": 2, "kv_unified": True},
        )
        silent = http.post(f"/api/models/{MODEL_ID}/load-recommended", json={"ctx_size": 65536})
    assert asked.status_code == 200, asked.text
    assert silent.status_code == 200, silent.text
    assert calls[0]["min_slots"] == 2 and calls[0]["kv_unified"] is True
    assert calls[0]["max_slots"] == 2
    # A caller that names neither is a pre-D72 caller: no opinion, not "false".
    assert calls[1]["min_slots"] is None and calls[1]["kv_unified"] is None


def test_v1_models_says_whether_the_slots_share_a_pool(app: Any) -> None:  # noqa: F811 - the imported fixture
    """Beside ``parallel`` and ``ctx_per_slot``: with it a client can tell two
    slots with their own caches from one slot everybody shares."""
    pool = routes_plan().model_copy(update={"parallel": 2, "kv_unified": True})
    with TestClient(app) as http:
        loaded(app, InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=pool))
        from_plan = http.get("/v1/models").json()["data"][0]["studioforge"]
        partitioned = InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=routes_plan())
        loaded(app, partitioned)
        split = http.get("/v1/models").json()["data"][0]["studioforge"]
    assert from_plan["kv_unified"] is True and from_plan["parallel"] == 2
    assert from_plan["ctx_per_slot"] == 16384
    assert split["kv_unified"] is False


def test_the_manager_takes_the_fields_with_no_opinion_as_the_default() -> None:
    import inspect

    for method in (ModelManager.load_recommended, ModelManager.plan_recommended):
        params = inspect.signature(method).parameters
        assert params["min_slots"].default is None
        assert params["kv_unified"].default is None


def test_a_client_can_gate_on_the_pool_before_asking_for_one() -> None:
    """A server without D72 answers 200 and ignores the two body fields, so a
    400 never tells a client it asked too early -- the feature name does."""
    from studioforge.core import capabilities

    report = capabilities.implemented_report()
    assert report["feature_decisions"]["shared_kv_pool"] == "D72"
    assert report["feature_decisions"]["context_checkpoint_settings"] == "D72"
    assert "D72" in report["decisions"]
