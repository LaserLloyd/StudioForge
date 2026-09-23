"""A preview planner (``log_plans=False``) says nothing at INFO (D69 §8).

The catalog, placements and fit previews build throwaway planners and plan
every model against hypothetical victims. Two lines ignored the flag:
``re-planned after eviction`` (560 of 1585 log lines on 2026-09-20, naming
evictions that never happened) and ``load rejected: device leased to another
holder`` (a saved override on a leased card, once per catalog build). A real
load (``log_plans=True``) still logs both at INFO.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core import planner as planner_module
from studioforge.core.leases import LeaseBook
from studioforge.core.planner import Planner
from studioforge.types import GB, LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_planner import (
    StubProbe,
    gpu,
    loaded_instance,
    make_config,
    make_meta,
    make_record,
    rig_5090x2_3090x2,
)


class _LevelRecorder:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def _at(self, level: str) -> Any:
        return lambda event, *_a, **_kw: self.lines.append((level, event))

    def __getattr__(self, name: str) -> Any:
        return self._at(name)

    def levels_of(self, event: str) -> list[str]:
        return [level for level, seen in self.lines if seen == event]


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> _LevelRecorder:
    rec = _LevelRecorder()
    monkeypatch.setattr(planner_module, "log", rec)
    return rec


def _eviction_replan(*, log_plans: bool) -> LoadPlan:
    probe = StubProbe([gpu(0, 32.0, 4.5, (12, 0))])
    config = make_config(headroom_fraction=0.0, on_insufficient="evict")
    config.models.default_ctx = 8192
    config.models.target_ctx = 65536
    planner = Planner(config, probe, log_plans=log_plans)
    loaded = [loaded_instance("idle/model", device=0, bytes_held=int(20 * GB))]
    record = make_record(meta=make_meta(tensor_bytes=4 * GB, n_ctx_train=131072))
    plan = planner.plan_load(record, loaded=loaded)
    assert isinstance(plan, LoadPlan)
    assert any("re-planned after eviction" in note for note in plan.notes)
    return plan


@pytest.mark.parametrize(("log_plans", "level"), [(False, "debug"), (True, "info")])
def test_re_planned_after_eviction_follows_the_log_plans_flag(
    recorder: _LevelRecorder, log_plans: bool, level: str
) -> None:
    _eviction_replan(log_plans=log_plans)
    assert recorder.levels_of("re-planned after eviction") == [level]
    assert recorder.levels_of("load planned") == [level]


@pytest.mark.parametrize(("log_plans", "level"), [(False, "debug"), (True, "info")])
def test_a_saved_override_on_a_leased_card_follows_the_log_plans_flag(
    recorder: _LevelRecorder, log_plans: bool, level: str
) -> None:
    book = LeaseBook()
    book.acquire([0], holder="clawforge2")
    planner = Planner(make_config(), rig_5090x2_3090x2(), leases=book, log_plans=log_plans)
    record = make_record(settings=ModelSettings(device_override=[0]))

    result = planner.plan_load(record, ctx_size=4096)

    assert isinstance(result, LoadRejected) and result.reason_code == "gpu_leased"
    assert recorder.levels_of("load rejected: device leased to another holder") == [level]
