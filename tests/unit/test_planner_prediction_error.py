"""A load whose measured VRAM misses the plan by more than 5% is a WARNING,
and the last observation per model is readable without a database (2026-09-09
review: "prediction error >5% is a bug", so it must be visible as one).
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core import planner as planner_module
from studioforge.core.planner import PREDICTION_ERROR_WARN_PCT, Planner
from studioforge.types import GB, LoadPlan, VramEstimate
from tests.unit.test_planner import make_config, rig_5090x2_3090x2


class RecordingLog:
    """The unit suite leaves structlog unconfigured, so the module logger is
    swapped for a recorder (the pattern test_planner_observed.py uses)."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []
        self.infos: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self.infos.append((event, fields))

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


def _plan(model_id: str = "test/model") -> LoadPlan:
    return LoadPlan(
        model_id=model_id,
        devices=[0, 1],
        ctx_size=262144,
        ctx_per_slot=262144,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        estimate=VramEstimate(weights_bytes=22 * GB, kv_bytes=18 * GB, compute_bytes=2 * GB),
        per_gpu_bytes={0: 20 * GB, 1: 22 * GB},
    )


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> RecordingLog:
    log = RecordingLog()
    monkeypatch.setattr(planner_module, "log", log)
    return log


def test_a_miss_over_the_bar_is_a_warning_naming_both_numbers(recorder: RecordingLog) -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    plan = _plan()
    predicted = plan.estimate.total_bytes
    planner.observe(model_id="test/model", plan=plan, actual_bytes=int(predicted * 1.10))

    events = [event for event, _ in recorder.warnings]
    assert "vram prediction error exceeds the bar" in events
    fields = next(f for e, f in recorder.warnings if e == "vram prediction error exceeds the bar")
    assert fields["error_pct"] == pytest.approx(10.0, abs=0.2)
    assert fields["bar_pct"] == PREDICTION_ERROR_WARN_PCT
    assert fields["predicted_mb"] and fields["actual_mb"] > fields["predicted_mb"]
    assert "breakdown_mb" in fields

    last = planner.last_observation("test/model")
    assert last is not None
    assert last["within_bar"] is False
    assert last["error_pct"] == pytest.approx(10.0, abs=0.2)
    assert last["devices"] == [0, 1]


def test_a_miss_inside_the_bar_is_not_a_warning(recorder: RecordingLog) -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    plan = _plan()
    predicted = plan.estimate.total_bytes
    planner.observe(model_id="test/model", plan=plan, actual_bytes=int(predicted * 1.02))

    assert not [e for e, _ in recorder.warnings if e == "vram prediction error exceeds the bar"]
    last = planner.last_observation("test/model")
    assert last is not None and last["within_bar"] is True
    assert last["error_pct"] == pytest.approx(2.0, abs=0.2)


def test_an_under_estimate_counts_the_same_as_an_over_estimate(recorder: RecordingLog) -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    plan = _plan()
    predicted = plan.estimate.total_bytes
    planner.observe(model_id="test/model", plan=plan, actual_bytes=int(predicted * 0.80))

    assert [e for e, _ in recorder.warnings if e == "vram prediction error exceeds the bar"]
    last = planner.last_observation("test/model")
    assert last is not None and last["within_bar"] is False and last["error_pct"] < 0


def test_last_observation_is_none_until_a_load_was_measured() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    assert planner.last_observation("never/loaded") is None
