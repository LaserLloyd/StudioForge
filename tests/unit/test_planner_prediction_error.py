"""A load whose measured VRAM misses the plan by more than 5% is a WARNING,
and the last observation per model is readable without a database (2026-09-09
review: "prediction error >5% is a bug", so it must be visible as one).

The error is the FORMULA's (D63): a plan D51 sized from the last measurement
plus ``OBS_SAFETY`` lands about 9% under its own total on every repeat load by
construction, and that band is reported as the plan's *margin*, never as the
planner's error and never as a warning.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from studioforge.core import planner as planner_module
from studioforge.core.planner import (
    _APPLIED_CORRECTIONS_CAP,
    CORRECTION_NOTE_PREFIX,
    OBS_SAFETY,
    OBSERVATION_FORMULA_KEY,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    PREDICTION_ERROR_WARN_PCT,
    AppliedCorrection,
    Planner,
    _correction_key,
    formula_terms,
    observed_correction,
    scaled_estimate,
)
from studioforge.types import GB, MB, LoadPlan, VramEstimate
from tests.unit.test_planner import make_config, make_record, rig_5090x2_3090x2
from tests.unit.test_planner_observed import StubLookup, key_of, plan_at


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


# ---------------------------------------------------------------------------
# D63: the planner's error is the formula's error
# ---------------------------------------------------------------------------


def corrected_planner(
    *, ratio: float = 0.613, sink: list[dict[str, Any]] | None = None
) -> tuple[Planner, LoadPlan, VramEstimate]:
    """A planner whose plan D51 sized from a stub measurement of ``ratio``
    times the formula: the planner, the corrected plan, and the formula
    estimate the correction replaced."""
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2(), log_plans=False))
    measured = int(baseline.estimate.total_bytes * ratio)
    planner = Planner(
        make_config(),
        rig_5090x2_3090x2(),
        observation_lookup=StubLookup(actual_bytes=measured, key=key_of(baseline)),
        observation_sink=sink.append if sink is not None else None,
        log_plans=False,
    )
    plan = plan_at(planner)
    assert plan.estimate.total_bytes == pytest.approx(measured * OBS_SAFETY, rel=1e-3)
    return planner, plan, baseline.estimate


def test_a_corrected_plan_landing_in_its_band_is_not_a_warning(recorder: RecordingLog) -> None:
    """D51 plans a repeat load at the last measurement times OBS_SAFETY, so the
    child lands ~9% under the plan by construction. Twelve of twelve repeat
    loads on the reference rig were warned at exactly -9.1% for it."""
    planner, plan, formula = corrected_planner()
    actual = round(plan.estimate.total_bytes / OBS_SAFETY)  # exactly what D51 planned from
    planner.observe(model_id=plan.model_id, plan=plan, actual_bytes=actual)

    assert recorder.warnings == []
    last = planner.last_observation(plan.model_id)
    assert last is not None
    assert last["corrected"] is True
    assert last["planned_bytes"] == plan.estimate.total_bytes
    assert last["predicted_bytes"] == formula.total_bytes
    band = -(1 - 1 / OBS_SAFETY) * 100
    assert last["margin_pct"] == pytest.approx(band, abs=0.05)
    formula_error = (actual - formula.total_bytes) / formula.total_bytes * 100
    assert last["error_pct"] == pytest.approx(formula_error, abs=0.05)
    assert last["within_bar"] is False  # the formula's miss, reported honestly
    factor = plan.estimate.total_bytes / formula.total_bytes
    assert last["correction_factor"] == pytest.approx(factor, abs=1e-3)
    info = next(f for e, f in recorder.infos if e == "load observation")
    assert info["corrected"] is True
    assert info["margin_pct"] == pytest.approx(band, abs=0.1)
    assert info["error_pct"] == pytest.approx(formula_error, abs=0.1)
    assert info["predicted_mb"] == round(formula.total_bytes / MB)
    assert info["planned_mb"] == round(plan.estimate.total_bytes / MB)


def test_a_child_over_the_corrected_total_by_the_bar_is_a_warning(recorder: RecordingLog) -> None:
    """The direction the band exists for: the earlier measurement did not
    describe this placement, or the footprint grew."""
    planner, plan, formula = corrected_planner()
    planned = plan.estimate.total_bytes
    planner.observe(model_id=plan.model_id, plan=plan, actual_bytes=int(planned * 1.07))

    assert [e for e, _ in recorder.warnings] == [
        "measured footprint exceeds the corrected estimate"
    ]
    fields = recorder.warnings[0][1]
    assert fields["margin_pct"] == pytest.approx(7.0, abs=0.1)
    assert fields["planned_mb"] == round(planned / MB)
    assert fields["formula_mb"] == round(formula.total_bytes / MB)
    last = planner.last_observation(plan.model_id)
    assert last is not None and last["margin_pct"] == pytest.approx(7.0, abs=0.1)


def test_a_child_inside_the_bar_above_the_corrected_total_is_not(recorder: RecordingLog) -> None:
    planner, plan, _ = corrected_planner()
    planner.observe(
        model_id=plan.model_id, plan=plan, actual_bytes=int(plan.estimate.total_bytes * 1.03)
    )
    assert recorder.warnings == []


@pytest.mark.parametrize("sign", [1, -1], ids=["under", "over"])
def test_an_uncorrected_plan_seven_percent_off_is_a_warning(
    recorder: RecordingLog, sign: int
) -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    plan = _plan()
    total = plan.estimate.total_bytes
    planner.observe(model_id="test/model", plan=plan, actual_bytes=int(total * (1 + sign * 0.07)))

    fields = next(f for e, f in recorder.warnings if e == "vram prediction error exceeds the bar")
    assert fields["error_pct"] == pytest.approx(sign * 7.0, abs=0.1)
    last = planner.last_observation("test/model")
    assert last is not None
    assert last["corrected"] is False
    assert last["margin_pct"] is None and last["correction_factor"] is None
    assert last["predicted_bytes"] == last["planned_bytes"] == total


def test_the_stored_row_of_a_corrected_plan_states_the_formula_and_keeps_the_note() -> None:
    seen: list[dict[str, Any]] = []
    planner, plan, formula = corrected_planner(sink=seen)
    actual = round(plan.estimate.total_bytes / OBS_SAFETY)
    planner.observe(
        model_id=plan.model_id, plan=plan, actual_bytes=actual, note=OBSERVATION_NOTE_PER_PID_DEVICE
    )

    (row,) = seen
    # The provenance marker is the caller's, untouched: D51's match rule keys on it.
    assert row["note"] == OBSERVATION_NOTE_PER_PID_DEVICE
    # predicted/weights are the FORMULA's, not the corrected total's.
    assert row["predicted_bytes"] == formula.total_bytes
    assert row["weights_bytes"] == formula.weights_bytes
    assert row["actual_bytes"] == actual
    planned = json.loads(row["per_gpu_planned"])
    shares = {str(d): b for d, b in plan.per_gpu_bytes.items()}
    assert {k: v for k, v in planned.items() if k.isdigit()} == shares
    fraction = planner.config.planner.compute_overhead_fraction
    block = planned[OBSERVATION_FORMULA_KEY]
    assert block["total_bytes"] == formula.total_bytes
    assert block["weights_bytes"] == formula.weights_bytes
    assert block["compute_bytes"] == formula.compute_bytes
    assert block["planned_bytes"] == plan.estimate.total_bytes
    assert block["overhead_fraction"] == fraction
    assert block["corrected"] is True
    assert block["factor"] == pytest.approx(
        plan.estimate.total_bytes / formula.total_bytes, abs=1e-3
    )
    assert formula_terms(row) == (formula.total_bytes, formula.weights_bytes, fraction)


def test_the_stored_row_of_an_uncorrected_plan_is_its_own_formula() -> None:
    seen: list[dict[str, Any]] = []
    planner = Planner(
        make_config(), rig_5090x2_3090x2(), observation_sink=seen.append, log_plans=False
    )
    plan = _plan()
    total = plan.estimate.total_bytes
    planner.observe(
        model_id="test/model", plan=plan, actual_bytes=total, note=OBSERVATION_NOTE_PER_PID_DEVICE
    )
    (row,) = seen
    assert row["predicted_bytes"] == total
    assert row["weights_bytes"] == plan.estimate.weights_bytes
    block = json.loads(row["per_gpu_planned"])[OBSERVATION_FORMULA_KEY]
    assert block["corrected"] is False and block["factor"] is None
    assert block["planned_bytes"] == block["total_bytes"] == total


def test_a_correction_this_process_cannot_account_for_is_never_read_as_the_formula(
    recorder: RecordingLog,
) -> None:
    """A plan adopted after a restart carries D51's note but not the formula it
    replaced. It is observed as corrected with the formula unknown, and stored
    as a legacy row, rather than with its corrected total dressed up as the
    formula's."""
    seen: list[dict[str, Any]] = []
    planner = Planner(
        make_config(), rig_5090x2_3090x2(), observation_sink=seen.append, log_plans=False
    )
    plan = _plan()
    plan.notes.append(f"{CORRECTION_NOTE_PREFIX}0.67 from the last load of this configuration")
    total = plan.estimate.total_bytes
    planner.observe(
        model_id="test/model",
        plan=plan,
        actual_bytes=round(total / OBS_SAFETY),
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
    )

    assert recorder.warnings == []
    last = planner.last_observation("test/model")
    assert last is not None
    assert last["corrected"] is True
    assert last["predicted_bytes"] is None
    assert last["error_pct"] is None and last["within_bar"] is None
    assert last["margin_pct"] == pytest.approx(-(1 - 1 / OBS_SAFETY) * 100, abs=0.05)
    (row,) = seen
    assert row["predicted_bytes"] == total  # what it reserved: a legacy row
    assert OBSERVATION_FORMULA_KEY not in json.loads(row["per_gpu_planned"])
    assert formula_terms(row) is None
    # ...and the child holding more than even that total is still the warning it was.
    planner.observe(model_id="test/model", plan=plan, actual_bytes=int(total * 1.08))
    assert [e for e, _ in recorder.warnings] == [
        "measured footprint exceeds the corrected estimate"
    ]


def test_a_plan_whose_numbers_are_not_the_remembered_ones_borrows_no_formula() -> None:
    """The remembered correction is verified term by term against the plan's
    estimate before its formula is reported: a plan that carries the note but
    other numbers is corrected with the formula unknown."""
    planner, plan, _ = corrected_planner()
    other = VramEstimate(weights_bytes=plan.estimate.weights_bytes + GB)
    altered = plan.model_copy(update={"estimate": other})
    planner.observe(model_id=plan.model_id, plan=altered, actual_bytes=other.total_bytes)
    last = planner.last_observation(plan.model_id)
    assert last is not None
    assert last["corrected"] is True
    assert last["predicted_bytes"] is None and last["planned_bytes"] == other.total_bytes


def test_an_auto_parallel_corrected_plan_still_knows_its_formula() -> None:
    """The slot sizer re-derives the estimate at the count it settles on; the
    correction spent at that count is the one the observation reads back."""
    config = make_config()
    config.models.default_parallel = "auto"
    baseline = Planner(config, rig_5090x2_3090x2(), log_plans=False).plan_load(
        make_record(), ctx_size=8192
    )
    assert isinstance(baseline, LoadPlan)
    lookup = StubLookup(actual_bytes=int(baseline.estimate.total_bytes * 0.7))
    planner = Planner(config, rig_5090x2_3090x2(), observation_lookup=lookup, log_plans=False)
    plan = planner.plan_load(make_record(), ctx_size=8192)
    assert isinstance(plan, LoadPlan)
    assert [n for n in plan.notes if n.startswith(CORRECTION_NOTE_PREFIX)]

    planner.observe(
        model_id=plan.model_id,
        plan=plan,
        actual_bytes=round(plan.estimate.total_bytes / OBS_SAFETY),
    )
    last = planner.last_observation(plan.model_id)
    assert last is not None
    assert last["corrected"] is True
    assert last["predicted_bytes"] is not None
    assert last["predicted_bytes"] != last["planned_bytes"]
    assert last["correction_factor"] == pytest.approx(
        last["planned_bytes"] / last["predicted_bytes"], abs=1e-3
    )


def test_the_remembered_corrections_are_bounded_oldest_first() -> None:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    correction = observed_correction(formula_bytes=10 * GB, observed_bytes=8 * GB)
    assert correction is not None
    formula = VramEstimate(weights_bytes=10 * GB)
    applied = AppliedCorrection(
        correction=correction,
        formula=formula,
        corrected=scaled_estimate(formula, correction.factor),
    )
    for i in range(_APPLIED_CORRECTIONS_CAP + 5):
        planner._remember_applied(_correction_key(f"m{i}", 8192, 1, "f16", "f16", 1), applied)
    remembered = planner._applied_corrections
    assert len(remembered) == _APPLIED_CORRECTIONS_CAP
    assert _correction_key("m0", 8192, 1, "f16", "f16", 1) not in remembered
    assert _correction_key("m4", 8192, 1, "f16", "f16", 1) not in remembered
    assert _correction_key("m5", 8192, 1, "f16", "f16", 1) in remembered
    newest = _correction_key(f"m{_APPLIED_CORRECTIONS_CAP + 4}", 8192, 1, "f16", "f16", 1)
    assert newest in remembered
