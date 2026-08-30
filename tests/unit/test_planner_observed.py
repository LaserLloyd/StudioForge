"""The observed-footprint correction (D51): the planner reading its own history.

The planner has recorded predicted-vs-actual VRAM per load since D18 and, until
now, nothing read those rows back at plan time -- the only feedback loop was one
global ``compute_overhead_fraction`` calibrated once at startup. A single scalar
cannot fix two signs at once, and the live rig had both open on 2026-08-30:
Dark-Scarlett-27B measured 4% ABOVE its estimate while Gemma-4-E4B measured
8202 MB against a predicted 13377, and the planner predicted 13377 again twelve
seconds after that observation landed.

These tests pin the two halves that make the feature safe rather than merely
useful: the clamp band (a fluke row can move the estimate, never replace it),
and the match rule (a contaminated pre-D40 row must be *unable* to be spent,
not merely unlikely to be). The fake probe and record helpers are imported from
``test_planner`` so the two files cannot drift on what "the rig" means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studioforge.core import planner as planner_module
from studioforge.core.planner import (
    OBS_BAND_MAX,
    OBS_BAND_MIN,
    OBS_SAFETY,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    Planner,
    observed_correction,
    scaled_estimate,
)
from studioforge.db import OBSERVATION_NOTE_TRUSTED, Database
from studioforge.types import GB, MB, LoadPlan, ModelSettings, VramEstimate
from tests.unit.test_planner import make_config, make_record, rig_5090x2_3090x2

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class StubLookup:
    """Stands in for ``Database.matching_observation``.

    Answers only for the exact key it was built with, the way the real lookup
    does, so a test that changes context or device count is really testing the
    match rule and not a stub that says yes to everything.
    """

    def __init__(
        self,
        *,
        actual_bytes: int | None = None,
        key: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.actual_bytes = actual_bytes
        self.key = key
        self.error = error
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def __call__(self, model_id: str, **fields: Any) -> dict[str, Any] | None:
        self.calls.append(
            (
                model_id,
                (
                    fields["ctx_size"],
                    fields["parallel"],
                    fields["kv_cache_type"],
                    fields["kv_cache_type_v"],
                    fields["device_count"],
                ),
            )
        )
        if self.error is not None:
            raise self.error
        if self.actual_bytes is None:
            return None
        if self.key is not None and any(fields.get(k) != v for k, v in self.key.items()):
            return None
        return {
            "model_id": model_id,
            "actual_bytes": self.actual_bytes,
            "ok": True,
            "note": OBSERVATION_NOTE_PER_PID_DEVICE,
        }


class RecordingLog:
    """Captures warnings off the planner module logger.

    Same reason as ``test_load_recommended.RecordingLog``: the unit suite
    leaves structlog unconfigured, so neither ``caplog`` nor
    ``structlog.testing.capture_logs`` holds across test orders. Swapping the
    module logger is the capture that works either way.
    """

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def __getattr__(self, _name: str) -> Any:
        return lambda *_a, **_kw: None


def plan_at(planner: Planner, record: Any = None, **kwargs: Any) -> LoadPlan:
    """One accepted plan at a pinned context and slot count."""
    kwargs.setdefault("ctx_size", 8192)
    kwargs.setdefault("parallel", 1)
    plan = planner.plan_load(record if record is not None else make_record(), **kwargs)
    assert isinstance(plan, LoadPlan), plan
    return plan


def key_of(plan: LoadPlan) -> dict[str, Any]:
    """The match key the planner will ask a lookup for, taken off a real plan."""
    return {
        "ctx_size": plan.ctx_size,
        "parallel": plan.parallel,
        "kv_cache_type": plan.kv_cache_type,
        "kv_cache_type_v": plan.kv_cache_type_v,
        "device_count": len(plan.devices),
    }


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


def observation(**overrides: Any) -> dict[str, Any]:
    """A clean, matchable per-device observation row."""
    row: dict[str, Any] = {
        "model_id": "m1",
        "ctx_size": 8192,
        "parallel": 1,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "devices": "0,1",
        "predicted_bytes": 13377 * MB,
        "actual_bytes": 8202 * MB,
        "weights_bytes": 7 * GB,
        "ok": True,
        "note": OBSERVATION_NOTE_TRUSTED,
    }
    row.update(overrides)
    return row


def lookup_key(**overrides: Any) -> dict[str, Any]:
    key: dict[str, Any] = {
        "ctx_size": 8192,
        "parallel": 1,
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "device_count": 2,
    }
    key.update(overrides)
    return key


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------


def test_the_two_live_cases_from_the_log() -> None:
    """The numbers this feature was written from, both signs, neither clamped."""
    # Dark-Scarlett-27B: measured ABOVE its estimate, so the correction is the
    # conservative direction -- exactly the one that prevents an OOM.
    scarlett = observed_correction(formula_bytes=35742 * MB, observed_bytes=37258 * MB)
    assert scarlett is not None and not scarlett.clamped
    assert round(scarlett.factor, 3) == 1.147
    assert round(scarlett.formula_bytes * scarlett.factor / MB) == 40984

    # Gemma-4-E4B: 5 GB of phantom on a card that has 24, released.
    gemma = observed_correction(formula_bytes=13377 * MB, observed_bytes=8202 * MB)
    assert gemma is not None and not gemma.clamped
    assert round(gemma.factor, 3) == 0.674
    assert round(gemma.formula_bytes * gemma.factor / MB) == 9022


def test_a_wild_row_is_clamped_at_both_edges() -> None:
    """A poisoned or fluke measurement moves the estimate; it cannot replace it."""
    formula = 20 * GB

    # The pre-D40 contamination shape: a whole-device total, ~3x the truth.
    high = observed_correction(formula_bytes=formula, observed_bytes=60 * GB)
    assert high is not None and high.clamped
    assert high.factor == pytest.approx(OBS_BAND_MAX)
    assert "clamped" in high.note

    # And its mirror: a measurement that missed most of the allocation.
    low = observed_correction(formula_bytes=formula, observed_bytes=1 * GB)
    assert low is not None and low.clamped
    assert low.factor == pytest.approx(OBS_BAND_MIN)
    assert "clamped" in low.note


def test_the_safety_margin_is_applied_before_the_clamp() -> None:
    """A measurement is trusted as itself plus 10%, not as itself."""
    correction = observed_correction(formula_bytes=10 * GB, observed_bytes=9 * GB)
    assert correction is not None
    assert correction.factor == pytest.approx(0.9 * OBS_SAFETY)


def test_nothing_to_correct_returns_none() -> None:
    assert observed_correction(formula_bytes=0, observed_bytes=8 * GB) is None
    assert observed_correction(formula_bytes=10 * GB, observed_bytes=0) is None
    # Within half a percent of the formula: a note about a change nobody can
    # see is noise on every plan of that model.
    near = int(10 * GB / OBS_SAFETY)
    assert observed_correction(formula_bytes=10 * GB, observed_bytes=near) is None


def test_scaling_moves_every_term_so_the_total_is_the_corrected_one() -> None:
    """``total_bytes`` is a derived sum, so a corrected total has to be spent
    across the terms -- and the terms have to stay internally consistent,
    because ``_reject`` computes its max-context suggestion as ``total - kv``."""
    estimate = VramEstimate(
        weights_bytes=8 * GB,
        kv_bytes=2 * GB,
        compute_bytes=512 * MB,
        cuda_context_bytes=300 * MB,
    )
    scaled = scaled_estimate(estimate, 0.5)
    assert scaled.total_bytes == pytest.approx(estimate.total_bytes * 0.5, rel=1e-6)
    assert scaled.weights_bytes == 4 * GB
    assert scaled.kv_bytes == 1 * GB
    assert scaled.total_bytes - scaled.kv_bytes == pytest.approx(
        (estimate.total_bytes - estimate.kv_bytes) * 0.5, rel=1e-6
    )


# ---------------------------------------------------------------------------
# Applying it to a plan
# ---------------------------------------------------------------------------


def test_the_correction_reaches_the_plan_and_its_note() -> None:
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2()))
    formula = baseline.estimate.total_bytes

    # The live Gemma ratio, against whatever this synthetic model computes to.
    measured = int(formula * 0.613)
    lookup = StubLookup(actual_bytes=measured, key=key_of(baseline))
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup)
    corrected = plan_at(planner)

    assert corrected.estimate.total_bytes == pytest.approx(measured * OBS_SAFETY, rel=1e-3)
    assert corrected.estimate.total_bytes < formula
    # Single-GPU placement: the card's share IS the corrected total.
    assert corrected.per_gpu_bytes[corrected.devices[0]] == corrected.estimate.total_bytes
    note = next(n for n in corrected.notes if "estimate corrected" in n)
    assert f"{round(measured / MB)} MB" in note
    assert f"{round(formula / MB)} MB" in note


def test_a_split_placements_per_card_shares_scale_with_the_total() -> None:
    """Per-card fit checks have to be asked against the corrected number too,
    or a plan is sized against a measurement and refused against a formula."""
    forced = make_record(settings=ModelSettings(device_override=[0, 1]))
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2()), forced)
    assert baseline.devices == [0, 1]
    formula = baseline.estimate.total_bytes

    measured = int(formula * 0.613)
    lookup = StubLookup(actual_bytes=measured, key=key_of(baseline))
    corrected = plan_at(
        Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup), forced
    )

    total = corrected.estimate.total_bytes
    assert total == pytest.approx(measured * OBS_SAFETY, rel=1e-3)
    assert sum(corrected.per_gpu_bytes.values()) == pytest.approx(total, rel=1e-3)
    factor = total / formula
    for device, share in corrected.per_gpu_bytes.items():
        assert share == pytest.approx(baseline.per_gpu_bytes[device] * factor, rel=1e-3)


def test_an_over_estimate_is_corrected_upwards() -> None:
    """The direction that prevents an OOM, not just the one that frees VRAM."""
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2()))
    formula = baseline.estimate.total_bytes
    lookup = StubLookup(actual_bytes=int(formula * 1.042), key=key_of(baseline))
    corrected = plan_at(Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup))
    assert corrected.estimate.total_bytes > formula


@pytest.mark.parametrize(
    "planner_kwargs, config_kwargs",
    [
        pytest.param({}, {}, id="no lookup wired"),
        pytest.param(
            {"observation_lookup": StubLookup(actual_bytes=1 * GB)},
            {"observed_correction": False},
            id="flag off",
        ),
        pytest.param(
            {"observation_lookup": StubLookup(actual_bytes=None)}, {}, id="no matching row"
        ),
    ],
)
def test_inert_without_a_lookup_a_flag_or_a_row(
    planner_kwargs: dict[str, Any], config_kwargs: dict[str, Any]
) -> None:
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2()))
    plan = plan_at(Planner(make_config(**config_kwargs), rig_5090x2_3090x2(), **planner_kwargs))
    assert plan.estimate.total_bytes == baseline.estimate.total_bytes
    assert not [n for n in plan.notes if "estimate corrected" in n]


def test_the_flag_off_never_touches_the_database() -> None:
    """Inert means inert: no lookup, not a lookup whose answer is discarded."""
    lookup = StubLookup(actual_bytes=1 * GB)
    planner = Planner(
        make_config(observed_correction=False),
        rig_5090x2_3090x2(),
        observation_lookup=lookup,
    )
    plan_at(planner)
    assert lookup.calls == []


def test_a_broken_lookup_costs_one_warning_and_no_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A database that cannot answer must cost accuracy, never a refused load."""
    baseline = plan_at(Planner(make_config(), rig_5090x2_3090x2()))
    recorder = RecordingLog()
    monkeypatch.setattr(planner_module, "log", recorder)

    lookup = StubLookup(error=RuntimeError("database is locked"))
    plan = plan_at(Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup))

    assert plan.estimate.total_bytes == baseline.estimate.total_bytes
    assert not [n for n in plan.notes if "estimate corrected" in n]
    failures = [w for w in recorder.warnings if "observed-footprint lookup failed" in w[0]]
    assert len(failures) == 1, "latched after the first raise, not once per rung"
    assert failures[0][1]["error"] == "database is locked"
    # And it stopped asking for the rest of the pass.
    assert len(lookup.calls) == 1


def test_the_memo_asks_each_configuration_once_per_plan_load() -> None:
    """The ladder walks contexts against every placement; the answer cannot
    change inside one pass, so neither can the number of round trips."""
    lookup = StubLookup(actual_bytes=None)
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup)
    planner.plan_load(make_record(), ctx_size=8192, parallel=1)

    assert lookup.calls, "the ladder asked at least once"
    assert len(lookup.calls) == len(set(lookup.calls)), "a repeated key was re-queried"

    # A second load is a new pass and asks again -- the row may have changed.
    before = len(lookup.calls)
    planner.plan_load(make_record(), ctx_size=8192, parallel=1)
    assert len(lookup.calls) > before


def test_auto_parallel_still_sizes_and_still_carries_the_note() -> None:
    """The slot sizer walks its own estimates through the same funnel, so the
    correction has to survive a plan whose final slot count is not the one the
    caller asked for."""
    config = make_config()
    config.models.default_parallel = "auto"
    baseline = Planner(config, rig_5090x2_3090x2()).plan_load(make_record(), ctx_size=8192)
    assert isinstance(baseline, LoadPlan)
    assert baseline.parallel >= 1

    lookup = StubLookup(actual_bytes=int(baseline.estimate.total_bytes * 0.7))
    corrected = Planner(config, rig_5090x2_3090x2(), observation_lookup=lookup).plan_load(
        make_record(), ctx_size=8192
    )
    assert isinstance(corrected, LoadPlan)
    assert corrected.parallel >= baseline.parallel, "a smaller estimate cannot buy fewer slots"
    assert [n for n in corrected.notes if "estimate corrected" in n]


def test_hypothetical_questions_stay_formula_only() -> None:
    """``fits_on`` serves the pre-download matrix -- dozens of calls per repo
    about models that have never been loaded here. It is outside a planning
    pass and must not become a database query per tier."""
    lookup = StubLookup(actual_bytes=1 * GB)
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_lookup=lookup)
    assert planner.fits_on(make_record(), devices=[0], ctx_size=8192) is not None
    assert lookup.calls == []


# ---------------------------------------------------------------------------
# The match rule (Database.matching_observation)
# ---------------------------------------------------------------------------


def test_the_note_constant_agrees_with_the_planners() -> None:
    """``db`` duplicates the marker as a literal to stay a leaf module. This is
    the test that keeps the duplicate honest."""
    assert OBSERVATION_NOTE_TRUSTED == OBSERVATION_NOTE_PER_PID_DEVICE


def test_the_newest_clean_row_wins(db: Database) -> None:
    db.record_load_observation(**observation(ts=100.0, actual_bytes=5 * GB))
    db.record_load_observation(**observation(ts=200.0, actual_bytes=9 * GB))
    row = db.matching_observation("m1", **lookup_key())
    assert row is not None and row["actual_bytes"] == 9 * GB


@pytest.mark.parametrize(
    "row_overrides, why",
    [
        pytest.param({"ok": False}, "a failed load measured nothing useful", id="ok=0"),
        pytest.param(
            {"note": "per_pid"},
            "D40: on Windows this summed one per-process total once per card",
            id="superseded note",
        ),
        pytest.param(
            {"note": None},
            "pre-per-pid: whole-device usage, median ratio 2.97",
            id="device-total row",
        ),
        pytest.param({"actual_bytes": 0}, "nothing was measured", id="zero measurement"),
    ],
)
def test_a_dirty_row_can_never_be_spent(
    db: Database, row_overrides: dict[str, Any], why: str
) -> None:
    db.record_load_observation(**observation(**row_overrides))
    assert db.matching_observation("m1", **lookup_key()) is None, why


@pytest.mark.parametrize(
    "lookup_overrides",
    [
        pytest.param({"ctx_size": 16384}, id="context"),
        pytest.param({"parallel": 2}, id="slots"),
        pytest.param({"kv_cache_type": "q8_0"}, id="K cache"),
        pytest.param({"kv_cache_type_v": "q8_0"}, id="V cache"),
        pytest.param({"device_count": 1}, id="device count"),
        pytest.param({"device_count": 4}, id="device count, wider"),
    ],
)
def test_a_different_configuration_is_not_a_match(
    db: Database, lookup_overrides: dict[str, Any]
) -> None:
    db.record_load_observation(**observation())
    assert db.matching_observation("m1", **lookup_key(**lookup_overrides)) is None


def test_another_model_is_not_a_match(db: Database) -> None:
    db.record_load_observation(**observation())
    assert db.matching_observation("m2", **lookup_key()) is None


def test_the_device_list_may_move_but_the_count_may_not(db: Database) -> None:
    """D42 relocates a model between same-count device sets and the tensor split
    is recomputed from live free VRAM every load, so matching the exact list
    would make this feature almost never fire."""
    db.record_load_observation(**observation(devices="2,3"))
    assert db.matching_observation("m1", **lookup_key(device_count=2)) is not None
    assert db.matching_observation("m1", **lookup_key(device_count=3)) is None


def test_a_pre_007_row_matches_only_a_symmetric_cache(db: Database) -> None:
    """NULL V is "this row cannot say", not "it was the same as K"."""
    db.record_load_observation(**observation(kv_cache_type="f16", kv_cache_type_v=None))
    assert db.matching_observation("m1", **lookup_key(kv_cache_type_v="f16")) is not None
    assert db.matching_observation("m1", **lookup_key(kv_cache_type_v=None)) is not None
    assert db.matching_observation("m1", **lookup_key(kv_cache_type_v="q8_0")) is None


def test_migration_007_stores_the_v_cache_type(db: Database) -> None:
    db.record_load_observation(**observation(kv_cache_type="f16", kv_cache_type_v="q8_0"))
    assert db.load_observations("m1")[0]["kv_cache_type_v"] == "q8_0"


def test_observe_records_the_v_cache_type() -> None:
    """The sink stores what the lookup will later have to match on."""
    captured: list[dict[str, object]] = []
    planner = Planner(make_config(), rig_5090x2_3090x2(), observation_sink=captured.append)
    plan = plan_at(planner, kv_cache_type="f16", kv_cache_type_v="q8_0")
    planner.observe(model_id=plan.model_id, plan=plan, actual_bytes=9 * GB)
    assert captured[0]["kv_cache_type_v"] == "q8_0"
    assert captured[0]["kv_cache_type"] == "f16"


def test_a_real_load_round_trips_through_the_database(db: Database) -> None:
    """End to end on the real sink and the real lookup: observe a load, then
    plan the same one again and find the measurement spent on it."""
    planner = Planner(
        make_config(),
        rig_5090x2_3090x2(),
        observation_sink=lambda row: db.record_load_observation(**row),
        observation_lookup=db.matching_observation,
    )
    first = plan_at(planner)
    formula = first.estimate.total_bytes
    assert not [n for n in first.notes if "estimate corrected" in n]

    measured = int(formula * 0.613)
    planner.observe(
        model_id=first.model_id,
        plan=first,
        actual_bytes=measured,
        note=OBSERVATION_NOTE_PER_PID_DEVICE,
    )

    second = plan_at(planner)
    assert second.estimate.total_bytes == pytest.approx(measured * OBS_SAFETY, rel=1e-3)
    assert [n for n in second.notes if "estimate corrected" in n]
