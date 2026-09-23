"""A rebalance launch failure relaunches the previous placement (D70, item 10).

On 2026-09-15 12:24:24-46 the D42 rebalancer moved JAX-XORTRON to [3, 2]: it
stopped the resident, the relaunch died at startup (exit 0xC0000409), the
no-evict rule declined to retry by evicting a bystander, and the log said
WARNING "rebalance failed; the model keeps its placement". It did not: the
resident was already stopped, and the model was down until its next JIT
request. D30 plans before it unloads, which covers a refusal but not a crash.

Now: when the model is gone after the failure, the previous plan is relaunched
on the previous devices with ``allow_evict=False``; when that fails too the
model is down and the log and the eviction book say so (``rebalance-failed``);
when the model is still resident the old message stands, because it is true.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core import manager as manager_module
from studioforge.errors import ModelLoadError
from studioforge.types import InstanceInfo
from tests.unit.test_gateway_lifecycle import (
    StubPlanner,
    candidate_plan,
    placed,
    rebalance_rig,
)
from tests.unit.test_unload_attribution import RecordingLog


@pytest.fixture()
def log(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    recording = RecordingLog()
    monkeypatch.setattr(manager_module, "log", recording)
    return recording


def _launch_crash(model_id: str) -> ModelLoadError:
    return ModelLoadError(
        f"llama-server for '{model_id}' exited with code 3221226505 during startup.",
        details={"stderr": ["CUDA error: shared object initialization failed"]},
    )


async def _run_rebalance(manager: Any) -> None:
    manager._maybe_rebalance()
    assert manager._rebalance_task is not None, "the move was not scheduled"
    await manager._rebalance_task


async def test_a_launch_crash_after_the_stop_relaunches_the_previous_placement(
    log: RecordingLog,
) -> None:
    """The 2026-09-15 shape, completed safely: the move dies, the old plan comes back."""
    manager, supervisor, mover = rebalance_rig()
    manager.planner = StubPlanner(candidate_plan(mover.id, [2, 3]))
    previous = supervisor.instances[mover.id]
    previous.priority = 2
    calls: list[dict[str, Any]] = []

    async def fake_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        calls.append({"model_id": model_id, **kwargs})
        if kwargs["source"] == "rebalance":
            # D30: the resident was stopped for the reload, then the launch died.
            supervisor.instances.pop(model_id, None)
            raise _launch_crash(model_id)
        restored = placed(model_id, kwargs["devices"])
        supervisor.instances[model_id] = restored
        return restored

    manager.load = fake_load  # type: ignore[method-assign]
    await _run_rebalance(manager)

    assert [call["source"] for call in calls] == ["rebalance", "rebalance-restore"]
    restore = calls[1]
    assert restore["devices"] == [1, 3], "the PREVIOUS devices, not the move's"
    assert restore["allow_evict"] is False, "no licence to evict a bystander to come back"
    assert restore["evict_busy"] is False
    assert restore["ctx_size"] == 32768 and restore["parallel"] == 1
    assert restore["kv_cache_type"] == "f16"
    assert restore["priority"] == 2, "the tier the model had"
    assert restore["hold_traffic"] is False
    assert restore["enforce_parallel_cap"] is False
    assert "force" not in restore and "require_resident" not in restore
    assert mover.id in supervisor.instances, "the model is serving again"
    assert manager.evictions() == [], "not an eviction: the model came back"

    events = [(level, event) for level, event, _ in log.events]
    assert ("warning", "rebalance launch failed; relaunching the previous placement") in events
    assert ("info", "rebalance rolled back; the model is back on its previous placement") in events
    assert not any("keeps its placement" in event for _, event in events), (
        "the old message was false here and must not appear"
    )
    warned = next(f for _, e, f in log.events if e.startswith("rebalance launch failed"))
    assert warned["devices"] == [1, 3]
    assert warned["error"].startswith("ModelLoadError: ")


async def test_when_the_rollback_fails_too_the_truth_is_recorded(log: RecordingLog) -> None:
    manager, supervisor, mover = rebalance_rig()
    manager.planner = StubPlanner(candidate_plan(mover.id, [2, 3]))
    calls: list[str] = []

    async def fake_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        calls.append(kwargs["source"])
        supervisor.instances.pop(model_id, None)
        raise _launch_crash(model_id)

    manager.load = fake_load  # type: ignore[method-assign]
    await _run_rebalance(manager)  # never raises out of the sweep

    assert calls == ["rebalance", "rebalance-restore"]
    assert mover.id not in supervisor.instances
    (event,) = manager.evictions()
    assert event["evicted"] == mover.id
    assert event["reason"] == "rebalance-failed"
    assert event["evicted_by"] == "rebalance"
    errors = [(e, f) for level, e, f in log.events if level == "error"]
    assert len(errors) == 1
    event_name, fields = errors[0]
    assert "the model is DOWN until its next load" in event_name
    assert fields["model_id"] == mover.id and fields["devices"] == [1, 3]
    assert not any("keeps its placement" in e for _, e, _ in log.events)
    assert mover.id in manager._rebalance_last, "the cooldown stands: no retry loop"


async def test_a_refusal_before_the_stop_keeps_the_old_message_because_it_is_true(
    log: RecordingLog,
) -> None:
    manager, supervisor, mover = rebalance_rig()
    manager.planner = StubPlanner(candidate_plan(mover.id, [2, 3]))
    calls: list[str] = []

    async def fake_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        calls.append(kwargs["source"])
        raise RuntimeError("the world moved between preview and gate")

    manager.load = fake_load  # type: ignore[method-assign]
    await _run_rebalance(manager)

    assert calls == ["rebalance"], "still resident: nothing to relaunch"
    assert mover.id in supervisor.instances
    assert manager.evictions() == []
    (level, event, fields) = next(e for e in log.events if "rebalance failed" in e[1])
    assert (level, event) == ("warning", "rebalance failed; the model keeps its placement")
    assert fields["error"] == "RuntimeError: the world moved between preview and gate"


async def test_a_vanished_model_with_no_previous_plan_is_reported_down(
    log: RecordingLog,
) -> None:
    """Defensive: nothing to relaunch from is still an honest record, not a crash."""
    manager, supervisor, mover = rebalance_rig()
    manager.planner = StubPlanner(candidate_plan(mover.id, [2, 3]))

    async def fake_load(model_id: str, **kwargs: Any) -> InstanceInfo:
        # The captured ``before`` is this very object: its plan is gone by the
        # time the restore asks for it.
        supervisor.instances[model_id].plan = None
        supervisor.instances.pop(model_id, None)
        raise _launch_crash(model_id)

    manager.load = fake_load  # type: ignore[method-assign]
    await _run_rebalance(manager)

    (event,) = manager.evictions()
    assert event["reason"] == "rebalance-failed"
    assert any(
        level == "error" and "no previous placement to relaunch" in e for level, e, _ in log.events
    )
