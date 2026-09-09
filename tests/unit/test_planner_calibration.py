"""The overhead fraction can come down (D63).

``compute_overhead_fraction`` is one global knob, tuned once per process from
load history. Until D63 the calibrator could only raise it: a row whose child
held LESS than predicted was silently ignored, and after D51 such a row could
not even say whether "predicted" was the formula or a total corrected from an
earlier measurement. Rows written by ``Planner.observe`` now carry the formula
they were planned from (``per_gpu_planned["formula"]``), and those rows can
lower the fraction: worst case first, with a margin, in bounded steps, and
never on the say-so of a legacy row.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from studioforge.core import planner as planner_module
from studioforge.core.planner import (
    CALIBRATION_MAX_STEP_DOWN,
    CALIBRATION_MIN_ROWS,
    OBSERVATION_FORMULA_KEY,
    OBSERVATION_NOTE_PER_PID,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    OVERHEAD_FRACTION_MAX,
    OVERHEAD_FRACTION_MIN,
    Planner,
    calibrated_overhead_fraction,
    clean_observations,
    formula_observations,
    formula_terms,
    suggest_overhead_fraction,
)
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

GB = 1_000_000_000
MIB = 2**20
WEIGHTS = 20 * GB


def formula_row(
    *,
    delta: float,
    fraction: float = 0.06,
    weights: int = WEIGHTS,
    note: str | None = OBSERVATION_NOTE_PER_PID_DEVICE,
    ok: bool = True,
    corrected: bool = False,
) -> dict[str, Any]:
    """A D63 row: the formula, at ``fraction``, predicted ``total`` and the
    child held ``total + delta * weights``. ``delta`` is therefore the amount
    the row moves the knob by, sign included."""
    total = int(weights * (1 + fraction)) + 2 * GB
    planned = int(total * 0.91) if corrected else total
    return {
        "model_id": "test/model",
        "predicted_bytes": total,
        "actual_bytes": round(total + delta * weights),
        "weights_bytes": weights,
        "ok": ok,
        "note": note,
        "per_gpu_planned": json.dumps(
            {
                "0": planned,
                OBSERVATION_FORMULA_KEY: {
                    "total_bytes": total,
                    "weights_bytes": weights,
                    "compute_bytes": int(weights * fraction),
                    "planned_bytes": planned,
                    "overhead_fraction": fraction,
                    "corrected": corrected,
                    "factor": round(planned / total, 6) if corrected else None,
                },
            }
        ),
    }


def legacy_row(*, shortfall: float, weights: int = WEIGHTS) -> dict[str, Any]:
    """A pre-D63 clean row: ``predicted_bytes`` may be a corrected total and
    the row cannot say. ``shortfall`` is ``(actual - predicted) / weights``."""
    predicted = int(weights * 1.2)
    return {
        "model_id": "test/model",
        "predicted_bytes": predicted,
        "actual_bytes": round(predicted + shortfall * weights),
        "weights_bytes": weights,
        "ok": True,
        "note": OBSERVATION_NOTE_PER_PID_DEVICE,
        "per_gpu_planned": json.dumps({"0": predicted}),
    }


class RecordingLog:
    def __init__(self) -> None:
        self.infos: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **fields: Any) -> None:
        self.infos.append((event, fields))

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


# ---------------------------------------------------------------------------
# Reading the rows
# ---------------------------------------------------------------------------


def test_formula_terms_reads_the_block_as_text_or_as_a_mapping() -> None:
    row = formula_row(delta=0.0, fraction=0.08)
    total = row["predicted_bytes"]
    assert formula_terms(row) == (total, WEIGHTS, 0.08)
    decoded = dict(row, per_gpu_planned=json.loads(row["per_gpu_planned"]))
    assert formula_terms(decoded) == (total, WEIGHTS, 0.08)


def block(**terms: Any) -> str:
    return json.dumps({OBSERVATION_FORMULA_KEY: terms})


@pytest.mark.parametrize(
    "per_gpu_planned",
    [
        None,
        "",
        "not json",
        json.dumps({"0": 1}),
        json.dumps({OBSERVATION_FORMULA_KEY: "yes"}),
        block(total_bytes=1, weights_bytes=1),
        block(total_bytes=True, weights_bytes=1, overhead_fraction=0.06),
        block(total_bytes=0, weights_bytes=1, overhead_fraction=0.06),
        block(total_bytes=5, weights_bytes=0, overhead_fraction=0.06),
        block(total_bytes=5, weights_bytes=1, overhead_fraction=-0.1),
    ],
)
def test_a_row_that_cannot_state_its_formula_is_a_legacy_row(per_gpu_planned: Any) -> None:
    row = dict(legacy_row(shortfall=0.0), per_gpu_planned=per_gpu_planned)
    assert formula_terms(row) is None


def test_formula_rows_are_the_clean_rows_that_also_state_their_formula() -> None:
    rows = [
        formula_row(delta=-0.01),
        formula_row(delta=-0.01, note=OBSERVATION_NOTE_PER_PID),  # D40: not clean
        formula_row(delta=-0.01, note=None),
        formula_row(delta=-0.01, ok=False),
        legacy_row(shortfall=-0.01),
    ]
    assert len(clean_observations(rows)) == 2
    assert formula_observations(rows) == [rows[0]]


# ---------------------------------------------------------------------------
# Down
# ---------------------------------------------------------------------------


def test_all_formula_rows_over_estimating_lowers_to_the_worst_plus_the_margin() -> None:
    """Five rows at 0.15 that all came in under the formula. The tightest of
    them needed 0.11, so the knob comes down to 0.11 + 2% = 0.13: the worst
    case governs, not the loosest row and not the mean."""
    rows = [formula_row(delta=d, fraction=0.15) for d in (-0.04, -0.06, -0.08, -0.10, -0.12)]
    assert suggest_overhead_fraction(rows, current=0.15) == pytest.approx(0.13)
    assert calibrated_overhead_fraction(rows, current=0.15) == pytest.approx(0.13)


def test_the_suggestion_is_rounded_up_to_half_a_percent() -> None:
    rows = [formula_row(delta=-0.043, fraction=0.15) for _ in range(CALIBRATION_MIN_ROWS)]
    # need 0.107, plus the margin 0.127 -> up to 0.13
    assert suggest_overhead_fraction(rows, current=0.15) == pytest.approx(0.13)
    exact = [formula_row(delta=-0.025, fraction=0.06) for _ in range(CALIBRATION_MIN_ROWS)]
    # need 0.035, plus the margin is exactly 0.055: a boundary is not bumped by float noise
    assert suggest_overhead_fraction(exact, current=0.06) == pytest.approx(0.055)


def test_a_move_down_is_bounded_per_calibration_but_a_move_up_is_not() -> None:
    rows = [formula_row(delta=-0.10, fraction=0.15) for _ in range(CALIBRATION_MIN_ROWS)]
    assert suggest_overhead_fraction(rows, current=0.15) == pytest.approx(0.07)
    assert calibrated_overhead_fraction(rows, current=0.15) == pytest.approx(
        0.15 - CALIBRATION_MAX_STEP_DOWN
    )
    # Going up covers the worst case at once: that is the direction that prevents an OOM.
    short = [formula_row(delta=+0.10, fraction=0.06) for _ in range(CALIBRATION_MIN_ROWS)]
    assert calibrated_overhead_fraction(short, current=0.06) == pytest.approx(OVERHEAD_FRACTION_MAX)


def test_a_move_down_stops_at_the_floor() -> None:
    rows = [formula_row(delta=-0.06, fraction=0.05) for _ in range(CALIBRATION_MIN_ROWS)]
    assert calibrated_overhead_fraction(rows, current=0.05) == pytest.approx(OVERHEAD_FRACTION_MIN)


def test_fewer_than_the_minimum_formula_rows_never_lowers() -> None:
    rows = [formula_row(delta=-0.10, fraction=0.15) for _ in range(CALIBRATION_MIN_ROWS - 1)]
    assert suggest_overhead_fraction(rows, current=0.15) == 0.15
    # Legacy rows that came in under their total do not make up the count: they
    # cannot say whether "under" was the formula or D51's band.
    under = [legacy_row(shortfall=-0.2) for _ in range(20)]
    assert calibrated_overhead_fraction(rows + under, current=0.15) is None


def test_the_worst_case_wins_over_any_number_of_easy_rows() -> None:
    easy = [formula_row(delta=-0.10, fraction=0.06) for _ in range(20)]
    # One row the formula only just met: its need is 0.06, so the knob stays.
    met = easy + [formula_row(delta=0.0, fraction=0.06)]
    assert suggest_overhead_fraction(met, current=0.06) == 0.06
    # One row the formula MISSED: not enough misses to raise, and no lowering either.
    missed = easy + [formula_row(delta=+0.01, fraction=0.06)]
    assert calibrated_overhead_fraction(missed, current=0.06) is None
    # Three misses: raised to cover the worst of them, exactly as before D63.
    three = easy + [formula_row(delta=+0.02, fraction=0.06)] * 3
    assert suggest_overhead_fraction(three, current=0.06) == pytest.approx(0.08)


def test_a_legacy_shortfall_keeps_the_knob_where_it_is() -> None:
    rows = [formula_row(delta=-0.10, fraction=0.06) for _ in range(20)]
    assert calibrated_overhead_fraction(rows, current=0.06) == pytest.approx(OVERHEAD_FRACTION_MIN)
    assert calibrated_overhead_fraction(rows + [legacy_row(shortfall=0.005)], current=0.06) is None


def test_legacy_rows_raise_exactly_as_before() -> None:
    rows = [legacy_row(shortfall=s) for s in (0.01, 0.02, 0.03)]
    assert suggest_overhead_fraction(rows, current=0.06) == pytest.approx(0.09)
    assert suggest_overhead_fraction(rows[:2], current=0.06) == 0.06
    # ...and they count together with the formula's misses toward the three.
    mixed = rows[:2] + [formula_row(delta=+0.05, fraction=0.06)]
    assert suggest_overhead_fraction(mixed, current=0.06) == pytest.approx(0.11)


def test_a_rows_need_is_absolute_so_the_same_rows_answer_the_same_on_every_boot() -> None:
    """Rows recorded at 0.15 that each came in 9% of weights under the formula
    needed 0.06. Read at any current value they still say 0.06 (+ the margin)."""
    rows = [formula_row(delta=-0.09, fraction=0.15) for _ in range(CALIBRATION_MIN_ROWS)]
    assert suggest_overhead_fraction(rows, current=0.15) == pytest.approx(0.08)
    assert suggest_overhead_fraction(rows, current=0.10) == pytest.approx(0.08)
    assert suggest_overhead_fraction(rows, current=0.08) == 0.08
    assert suggest_overhead_fraction(rows, current=0.06) == 0.06  # met: not short, not loose
    assert suggest_overhead_fraction(rows, current=0.04) == pytest.approx(0.06)  # short at 0.04


def test_calibrate_moves_the_live_config_down_logs_it_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = RecordingLog()
    monkeypatch.setattr(planner_module, "log", log)
    config = make_config(compute_overhead_fraction=0.15)
    planner = Planner(config, rig_5090x2_3090x2())
    rows = [formula_row(delta=-0.10, fraction=0.15) for _ in range(CALIBRATION_MIN_ROWS)]

    assert planner.calibrate(rows) == pytest.approx(0.12)
    assert config.planner.compute_overhead_fraction == pytest.approx(0.12)
    fields = next(
        f for e, f in log.infos if e == "calibrated compute overhead fraction from load history"
    )
    assert fields["previous"] == 0.15 and fields["tuned"] == 0.12
    assert fields["direction"] == "down"
    assert fields["rows"] == CALIBRATION_MIN_ROWS and fields["formula_rows"] == CALIBRATION_MIN_ROWS
    # Each calibration is one step; at the target the same rows are a no-op.
    assert planner.calibrate(rows) == pytest.approx(0.09)
    assert planner.calibrate(rows) == pytest.approx(0.07)
    assert planner.calibrate(rows) is None


def test_the_reference_rig_on_2026_09_10() -> None:
    """Twelve repeat loads at actual/predicted 0.909 -- D51's band -- with the
    knob at its 0.15 ceiling. As legacy rows they say nothing either way. As
    D63 rows they carry the formula's own miss; and on this rig the formula
    under-estimates Dark-Scarlett-27B at 262k/q8_0 by 7% of its weights even
    at 0.15 (35742 MB predicted, 37260 held), so the knob cannot come down
    while such a load is in the window: one global fraction cannot be two
    signs at once (D51)."""
    band = [legacy_row(shortfall=0.0) for _ in range(12)]
    for row in band:
        row["actual_bytes"] = int(row["predicted_bytes"] * 0.909)
    assert calibrated_overhead_fraction(band, current=OVERHEAD_FRACTION_MAX) is None

    dark_scarlett = formula_row(delta=+0.0701, fraction=0.15, weights=21642 * MIB)
    loose = [formula_row(delta=-0.187, fraction=0.15, weights=26496 * MIB) for _ in range(12)]
    assert (
        calibrated_overhead_fraction(loose + [dark_scarlett], current=OVERHEAD_FRACTION_MAX) is None
    )
    assert calibrated_overhead_fraction(loose, current=OVERHEAD_FRACTION_MAX) == pytest.approx(
        OVERHEAD_FRACTION_MAX - CALIBRATION_MAX_STEP_DOWN
    )
