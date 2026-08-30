"""D50: one reload implementation, engine-aware, and a duplicate that folds.

Incident (rig log, 2026-08-30): the GUI's "activate + reload" button fired twice
118 ms apart -- one double-click. Each run force-reloaded every resident, so both
models were reloaded twice back to back (~31 s of churn) and a 37 GB model was
killed 3.6 s after it finished becoming ready. Three separate gaps let that
happen, and all three are pinned here:

* Nothing recorded which engine a child was actually running, so no layer could
  say "this one is already on b10689, skip it" -- see ``test_supervisor.py`` for
  the stamping itself and here for what reads it.
* The reload loop existed twice (the GUI's copy and the route's) with different
  failure semantics. Now it exists once, on the manager.
* A forced reload that queued behind an *identical* forced reload restarted a
  child that was seconds old, because it had no way to notice.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.api.admin_routes import restart_backend
from studioforge.config import Config
from studioforge.core import manager as manager_module
from studioforge.core.manager import ModelManager
from studioforge.errors import ModelLoadError
from studioforge.types import InstanceInfo, LoadPlan, ModelRecord
from tests.unit.test_load_retry import (
    StubPlanner,
    StubProbe,
    StubRegistry,
    StubSupervisor,
    make_record,
)

ACTIVE = "b10689"
STALE = "b10425"
MODEL = "a/model"
OTHER = "b/model"


class EngineSupervisor(StubSupervisor):
    """A StubSupervisor that knows the two facts D50 added to an instance.

    Namely: which engine build a child is actually running, and a launch number
    that increases every time a child is deliberately started. Both come from the
    real supervisor at spawn; here they are stamped by hand so the reload logic
    can be driven without a process.
    """

    def __init__(
        self,
        *,
        active_engine: str | None = ACTIVE,
        spawn_engine: str | None = None,
        fail_on: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.active_engine = active_engine
        #: What a NEW child comes up running. Defaults to the active build,
        #: which is what really happens; a test that wants engine drift sets it.
        self.spawn_engine = spawn_engine if spawn_engine is not None else active_engine
        self.fail_on = fail_on
        self.seq = 0
        #: Set by a test to hold every spawn until it is released, so a second
        #: caller can be parked on the per-model lock deterministically.
        self.release: asyncio.Event | None = None
        self.entered = asyncio.Event()

    def active_engine_tag(self) -> str | None:
        return self.active_engine

    def add(
        self,
        model_id: str,
        *,
        engine_tag: str | None = ACTIVE,
        state: str = "ready",
    ) -> InstanceInfo:
        """Put a child in the table without starting one."""
        self.seq += 1
        info = InstanceInfo(
            model_id=model_id,
            state=state,  # type: ignore[arg-type]
            port=18100 + self.seq,
            ttl_s=1800,
            started_at=1.0,
            last_activity_at=1.0,
            plan=LoadPlan(model_id=model_id, devices=[0], ctx_size=8192),
            resolved_engine_tag=engine_tag,
            spawn_seq=self.seq,
        )
        self.instances[model_id] = info
        return info

    async def start(self, record: ModelRecord, plan: LoadPlan, **kwargs: Any) -> InstanceInfo:
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if record.id in self.fail_on:
            self.starts += 1
            raise ModelLoadError(
                f"llama-server for '{record.id}' exited with code 1 during startup.",
                details={"stderr": ["boom"], "argv": ["llama-server"]},
            )
        info = await super().start(record, plan, **kwargs)
        self.seq += 1
        info.spawn_seq = self.seq
        info.resolved_engine_tag = self.spawn_engine
        return info


def make_manager(
    supervisor: EngineSupervisor, *, model_ids: tuple[str, ...] = (MODEL, OTHER)
) -> ModelManager:
    records = {model_id: make_record(model_id) for model_id in model_ids}
    return ModelManager(
        Config(data_dir="/tmp/sf-d50"),
        registry=StubRegistry(records),  # type: ignore[arg-type]
        planner=StubPlanner(probe=StubProbe()),  # type: ignore[arg-type]
        supervisor=supervisor,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )


class RecordingLog:
    """Swaps the manager module's structlog logger; see test_load_recommended."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def warning(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def names(self) -> list[str]:
        return [event for event, _ in self.events]

    def fields(self, event: str) -> dict[str, Any]:
        return next(fields for name, fields in self.events if name == event)

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


# ---------------------------------------------------------------------------
# reload_resident: the one implementation
# ---------------------------------------------------------------------------


async def test_a_plain_reload_restarts_every_resident() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL)
    supervisor.add(OTHER)
    manager = make_manager(supervisor)

    result = await manager.reload_resident(source="test")

    assert sorted(result["restarted"]) == [MODEL, OTHER]
    assert result["count"] == 2
    assert result["skipped"] == []
    assert result["failed"] == []
    assert result["active_engine"] == ACTIVE
    assert supervisor.starts == 2


async def test_only_stale_engine_skips_the_children_already_on_the_active_build() -> None:
    """The whole point: activating an engine twice must cost nothing the second
    time. The 2026-08-30 double-click reloaded both models a second time because
    nothing could tell they were already current."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    supervisor.add(OTHER, engine_tag=ACTIVE)
    manager = make_manager(supervisor)

    result = await manager.reload_resident(only_stale_engine=True, source="gui:activate")

    assert result["restarted"] == [MODEL], "only the child on the superseded build"
    assert result["count"] == 1
    assert result["skipped"] == [
        {"model_id": OTHER, "engine_tag": ACTIVE, "reason": "already on the active engine"}
    ]
    assert supervisor.starts == 1


async def test_running_the_same_activation_twice_is_a_no_op_the_second_time() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    supervisor.add(OTHER, engine_tag=STALE)
    manager = make_manager(supervisor)

    first = await manager.reload_resident(only_stale_engine=True, source="gui:activate")
    second = await manager.reload_resident(only_stale_engine=True, source="gui:activate")

    assert first["count"] == 2
    assert second["count"] == 0, "the second click restarted nothing"
    assert [s["model_id"] for s in second["skipped"]] == [MODEL, OTHER]
    assert supervisor.starts == 2, "two children were reloaded in total, not four"


async def test_a_child_still_loading_is_left_alone() -> None:
    """Its spawn resolves the active engine on the way through, so it is already
    becoming what a reload would make it -- and force-reloading a load in flight
    is how one click turns into two cold loads of the same weights."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE, state="loading")
    manager = make_manager(supervisor)

    result = await manager.reload_resident(source="test")

    assert result["restarted"] == []
    assert result["skipped"] == [{"model_id": MODEL, "engine_tag": STALE, "reason": "loading"}]
    assert supervisor.starts == 0


async def test_one_model_that_cannot_be_reloaded_does_not_strand_the_others() -> None:
    supervisor = EngineSupervisor(fail_on=(MODEL,))
    supervisor.add(MODEL)
    supervisor.add(OTHER)
    manager = make_manager(supervisor)

    result = await manager.reload_resident(source="test")

    assert result["restarted"] == [OTHER]
    assert result["count"] == 1
    assert [f["model_id"] for f in result["failed"]] == [MODEL]
    assert "exited with code 1" in result["failed"][0]["error"]


async def test_an_unknown_active_engine_reloads_everything_rather_than_skipping() -> None:
    """Not being able to prove a child is current is a reason to reload it: a
    redundant reload costs half a minute, a missed one leaves a superseded
    build serving."""
    supervisor = EngineSupervisor(active_engine=None)
    supervisor.add(MODEL, engine_tag=None)
    manager = make_manager(supervisor)

    result = await manager.reload_resident(only_stale_engine=True, source="test")

    assert result["restarted"] == [MODEL]
    assert result["active_engine"] is None


async def test_a_skip_says_why_in_the_log() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=ACTIVE)
    manager = make_manager(supervisor)
    recorder = RecordingLog()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_module, "log", recorder)
        await manager.reload_resident(only_stale_engine=True, source="gui:activate")

    fields = recorder.fields("reload_resident_skipped")
    assert fields["model_id"] == MODEL
    assert fields["engine_tag"] == ACTIVE
    assert fields["active_engine"] == ACTIVE
    assert fields["source"] == "gui:activate"


# ---------------------------------------------------------------------------
# POST /api/restart/backend still speaks the tray's language
# ---------------------------------------------------------------------------


def fake_request(manager: ModelManager, supervisor: EngineSupervisor) -> Any:
    state = SimpleNamespace(manager=manager, supervisor=supervisor)
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def test_the_restart_route_keeps_the_keys_the_tray_reads() -> None:
    """``tray_app`` reports ``count`` and lists ``failed``; both must survive."""
    supervisor = EngineSupervisor(fail_on=(OTHER,))
    supervisor.add(MODEL)
    supervisor.add(OTHER)
    manager = make_manager(supervisor)

    payload = await restart_backend(fake_request(manager, supervisor))

    assert payload["restarted"] == [MODEL]
    assert payload["count"] == 1
    assert [f["model_id"] for f in payload["failed"]] == [OTHER]
    assert payload["skipped"] == []
    assert payload["active_engine"] == ACTIVE


async def test_the_restart_route_can_ask_for_the_stale_children_only() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    supervisor.add(OTHER, engine_tag=ACTIVE)
    manager = make_manager(supervisor)

    payload = await restart_backend(fake_request(manager, supervisor), only_stale_engine=True)

    assert payload["restarted"] == [MODEL]
    assert [s["model_id"] for s in payload["skipped"]] == [OTHER]


async def test_calling_the_route_as_a_function_still_restarts_everything() -> None:
    """The GUI's Dashboard button calls this handler directly, so the unfilled
    ``Body`` default arrives as FastAPI's own FieldInfo object -- which is
    truthy. Reading it as a bool would have silently turned "Restart engines"
    into "restart only the stale ones"."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=ACTIVE)
    manager = make_manager(supervisor)

    payload = await restart_backend(fake_request(manager, supervisor))

    assert payload["restarted"] == [MODEL], "a current child is still recycled"
    assert payload["skipped"] == []


# ---------------------------------------------------------------------------
# Coalescing a duplicate forced reload
# ---------------------------------------------------------------------------


async def double_click(
    manager: ModelManager, supervisor: EngineSupervisor, **second: Any
) -> tuple[InstanceInfo, InstanceInfo]:
    """Two forced reloads of one model, the second issued while the first runs.

    That is the shape of the incident: the second request reads the resident's
    launch number, then parks on the per-model lock while the first replaces the
    child underneath it.
    """
    supervisor.release = asyncio.Event()
    first = asyncio.create_task(manager.load(MODEL, force=True, source="gui:activate"))
    await supervisor.entered.wait()
    later = asyncio.create_task(manager.load(MODEL, force=True, source="gui:activate", **second))
    await asyncio.sleep(0.05)
    supervisor.release.set()
    return await first, await later


async def test_a_double_clicked_reload_folds_into_the_one_that_already_ran() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)
    recorder = RecordingLog()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(manager_module, "log", recorder)
        first, second = await double_click(manager, supervisor)

    assert supervisor.starts == 1, "the second click restarted a child seconds old"
    assert second is first, "the folded call hands back the live instance"
    assert "forced_reload_coalesced" in recorder.names()
    fields = recorder.fields("forced_reload_coalesced")
    assert fields["model_id"] == MODEL
    assert fields["source"] == "gui:activate"
    assert fields["spawn_seq"] > fields["spawn_seq_queued_behind"]


async def test_an_explicit_context_is_never_folded_away() -> None:
    """A different shape is not a duplicate of anything."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)

    await double_click(manager, supervisor, ctx_size=65536)

    assert supervisor.starts == 2


async def test_an_explicit_placement_is_never_folded_away() -> None:
    """Which is what keeps a fold from cancelling a D42 rebalance or a D46
    restore -- both of them pass ``devices``."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)

    await double_click(manager, supervisor, devices=[1])

    assert supervisor.starts == 2


async def test_an_explicit_slot_count_is_never_folded_away() -> None:
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)

    await double_click(manager, supervisor, parallel=4)

    assert supervisor.starts == 2


async def test_an_engine_adoption_is_never_folded_away() -> None:
    """The one reload that must always run. A child that came up on a build
    which is no longer active still has to be moved, however recently it was
    started -- skipping it leaves a superseded engine serving, which is a worse
    version of the bug this whole decision exists to fix."""
    supervisor = EngineSupervisor(active_engine=ACTIVE, spawn_engine=STALE)
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)

    await double_click(manager, supervisor)

    assert supervisor.starts == 2


async def test_an_explicit_tier_is_still_applied_when_the_reload_folds() -> None:
    """The reload folds; the tier the caller stated does not. Otherwise
    double-clicking a chat load would leave the model evictable as background."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=STALE)
    manager = make_manager(supervisor)

    _first, second = await double_click(manager, supervisor, priority=1)

    assert supervisor.starts == 1, "still folded"
    assert second.priority == 1, "the re-stamp reached the live instance (D46)"


async def test_two_deliberate_reloads_in_sequence_both_run() -> None:
    """The fold is not a rate limiter. A restart issued after the previous one
    finished is compared against a launch number that has stopped moving, so it
    always runs."""
    supervisor = EngineSupervisor()
    supervisor.add(MODEL, engine_tag=ACTIVE)
    manager = make_manager(supervisor)

    await manager.load(MODEL, force=True, source="api:restart")
    await manager.load(MODEL, force=True, source="api:restart")

    assert supervisor.starts == 2


async def test_a_forced_load_of_a_failed_child_is_never_folded() -> None:
    """``spawn_seq`` moved, but a child that died has not been reloaded."""
    supervisor = EngineSupervisor()
    instance = supervisor.add(MODEL, engine_tag=ACTIVE)
    manager = make_manager(supervisor)
    instance.state = "failed"
    instance.spawn_seq = 99

    assert (
        manager._reload_already_done(
            instance,
            0,
            ctx_size=None,
            kv_cache_type=None,
            kv_cache_type_v=None,
            parallel=None,
            devices=None,
        )
        is False
    )


async def test_an_unidentifiable_engine_never_counts_as_a_match() -> None:
    """``None`` means "could not be established", not "the same as yours"."""
    supervisor = EngineSupervisor(active_engine=None)
    instance = supervisor.add(MODEL, engine_tag=None)
    manager = make_manager(supervisor)
    instance.spawn_seq = 99

    assert (
        manager._reload_already_done(
            instance,
            0,
            ctx_size=None,
            kv_cache_type=None,
            kv_cache_type_v=None,
            parallel=None,
            devices=None,
        )
        is False
    )
