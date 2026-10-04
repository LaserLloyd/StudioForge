"""D76 in the planner: D51's margin skips the weights, and a refusal walks down.

The 2026-10-04 case. The 177B chat model (i1-Q4_K_S, 71.3 GiB of GPU weights
after D73) on CUDA 0, 1 and 2 at 2 slots sharing a 122880-token pool measured
79.2 GiB -- above the 77.65 GiB the three cards offer at the 10% headroom, but
well inside the 86.42 GiB they had free. D51 then planned the repeat at
``79.2 * 1.10`` = 87.1 GiB: 7.1 GiB of "safety" on top of weights that are the
file's own tensor bytes and cannot grow, so the load that had just run could
never be planned again. And the refusal's ``max_ctx_that_fits`` was computed
from that corrected estimate's fixed cost (10% over the weights alone was 78.4
GiB), so it was ``None``; ClawChat's one re-ask at "the largest window that
fits" could not fire and the turn fell through to a JIT load.

What these pin:

* the margin applies to the measured bytes above the CURRENT formula's
  weights (never a stored row's -- pre-D73 rows carry the host-resident
  table), the whole-total rule survives for a measurement below the weights,
  and the band still clamps;
* a corrected estimate keeps its weights term exact and spreads the rest;
* a refusal names the largest ladder window below the one refused that the
  planner would accept -- corrected where a measurement of that window exists,
  the formula elsewhere -- and that window then plans.

The scenario model is 64 GiB of weights on the three-card set of the test rig
(76.73 GiB usable), at 2 slots sharing one pool: the same proportions.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core.planner import (
    OBS_BAND_MAX,
    OBS_SAFETY,
    OBSERVATION_NOTE_PER_PID_DEVICE,
    Planner,
    corrected_estimate,
    observed_correction,
    scaled_estimate,
)
from studioforge.db import Database
from studioforge.types import GB, MB, LoadPlan, LoadRejected, ModelSettings, VramEstimate
from tests.unit.test_planner import make_config, make_meta, make_record, rig_5090x2_3090x2

GIB = GB
THREE = [0, 1, 2]


# ---------------------------------------------------------------------------
# The arithmetic, on the incident's numbers
# ---------------------------------------------------------------------------


def test_the_margin_is_spent_above_the_weights_only() -> None:
    """71.3 GiB of weights, 79.2 measured: 80.0 GiB, not 87.1."""
    weights = int(71.3 * GIB)
    observed = int(79.2 * GIB)
    formula = int(79.2 / 1.032 * GIB)  # measured ~= formula x 1.032 on the rig

    d76 = observed_correction(formula_bytes=formula, observed_bytes=observed, weights_bytes=weights)
    assert d76 is not None and not d76.clamped
    assert d76.weights_bytes == weights
    assert d76.corrected_bytes / GIB == pytest.approx(71.3 + 7.9 * OBS_SAFETY, abs=0.01)
    assert d76.corrected_bytes / GIB == pytest.approx(80.0, abs=0.05)
    assert "weights, which are exact" in d76.note

    # The rule D51 shipped with, kept for a caller that does not know the
    # weights: the whole measurement carries the margin.
    whole = observed_correction(formula_bytes=formula, observed_bytes=observed)
    assert whole is not None and whole.weights_bytes == 0
    assert whole.corrected_bytes / GIB == pytest.approx(79.2 * OBS_SAFETY, abs=0.01)
    assert whole.corrected_bytes / GIB == pytest.approx(87.1, abs=0.05)
    # ...which put the weights term alone past the three cards' 77.65 GiB.
    assert 71.3 * whole.factor > 77.65


def test_a_measurement_below_the_weights_keeps_the_whole_total_rule() -> None:
    """A child holding less than the formula's own weights has disproved that
    term (the Gemma-4-E4B shape), so it cannot be held exact."""
    correction = observed_correction(
        formula_bytes=10 * GIB, observed_bytes=7 * GIB, weights_bytes=8 * GIB
    )
    assert correction is not None
    assert correction.weights_bytes == 0
    assert correction.factor == pytest.approx(0.7 * OBS_SAFETY)
    assert "below the formula's" in correction.note


def test_the_band_still_clamps_the_weights_rule() -> None:
    """A contaminated row (a whole-device total, ~3x) is still only partly trusted."""
    correction = observed_correction(
        formula_bytes=20 * GIB, observed_bytes=60 * GIB, weights_bytes=15 * GIB
    )
    assert correction is not None and correction.clamped
    assert correction.factor == pytest.approx(OBS_BAND_MAX)
    assert "clamped" in correction.note


def test_a_measurement_the_weights_rule_lands_on_the_formula_is_no_correction() -> None:
    """The no-op tolerance applies to the new total exactly as to the old one."""
    weights, formula = 8 * GIB, 10 * GIB
    on_the_formula = weights + int((formula - weights) / OBS_SAFETY)
    assert (
        observed_correction(
            formula_bytes=formula, observed_bytes=on_the_formula, weights_bytes=weights
        )
        is None
    )


def test_a_corrected_estimate_keeps_the_weights_and_moves_the_rest() -> None:
    estimate = VramEstimate(
        weights_bytes=64 * GIB,
        kv_bytes=4 * GIB,
        compute_bytes=int(3.84 * GIB),
        cuda_context_bytes=900 * MB,
    )
    observed = int(estimate.total_bytes * 1.032)
    correction = observed_correction(
        formula_bytes=estimate.total_bytes,
        observed_bytes=observed,
        weights_bytes=estimate.weights_bytes,
    )
    assert correction is not None

    corrected = corrected_estimate(estimate, correction)

    assert corrected.weights_bytes == estimate.weights_bytes
    target = estimate.weights_bytes + (observed - estimate.weights_bytes) * OBS_SAFETY
    assert corrected.total_bytes == pytest.approx(target, abs=8)
    # The rest moved together, by one factor: the KV/compute proportion holds.
    assert corrected.kv_bytes / corrected.compute_bytes == pytest.approx(
        estimate.kv_bytes / estimate.compute_bytes, rel=1e-6
    )
    assert corrected.kv_bytes > estimate.kv_bytes


def test_a_whole_total_correction_still_scales_every_term() -> None:
    estimate = VramEstimate(weights_bytes=8 * GIB, kv_bytes=2 * GIB, compute_bytes=512 * MB)
    correction = observed_correction(
        formula_bytes=estimate.total_bytes, observed_bytes=6 * GIB, weights_bytes=8 * GIB
    )
    assert correction is not None and correction.weights_bytes == 0
    assert corrected_estimate(estimate, correction) == scaled_estimate(estimate, correction.factor)


# ---------------------------------------------------------------------------
# The scenario: a 64 GiB model pinned to three cards
# ---------------------------------------------------------------------------


def big_record() -> Any:
    return make_record(
        meta=make_meta(tensor_bytes=64 * GIB, n_ctx_train=262144),
        size_bytes=64 * GIB,
        settings=ModelSettings(device_override=THREE),
    )


def usable(planner: Planner) -> int:
    return sum(
        planner.usable_bytes(g, forced=True) for g in planner.probe.list_gpus() if g.index in THREE
    )


def formula_at(ctx: int) -> VramEstimate:
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    return planner.estimate(
        big_record(),
        ctx_size=ctx,
        parallel=2,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=3,
        kv_unified=True,
    )


class RowsByContext:
    """A lookup holding measurements of the pool at given contexts on 3 cards.

    ``weights_bytes`` rides on every row the way the real table stores it, so
    a test can put a pre-D73 figure there and prove nobody reads it.
    """

    def __init__(self, rows: dict[int, int], *, row_weights: int | None = None) -> None:
        self.rows = rows
        self.row_weights = row_weights
        self.asked: list[int] = []

    def __call__(self, model_id: str, **key: Any) -> dict[str, Any] | None:
        self.asked.append(int(key["ctx_size"]))
        if key["device_count"] != 3 or key["parallel"] != 2 or not key.get("kv_unified"):
            return None
        actual = self.rows.get(int(key["ctx_size"]))
        if actual is None:
            return None
        row: dict[str, Any] = {
            "model_id": model_id,
            "actual_bytes": actual,
            "ok": True,
            "note": OBSERVATION_NOTE_PER_PID_DEVICE,
        }
        if self.row_weights is not None:
            row["weights_bytes"] = self.row_weights
        return row


def plan(planner: Planner, ctx: int) -> Any:
    return planner.plan_load(
        big_record(), ctx_size=ctx, parallel=2, kv_cache_type="f16", kv_unified=True
    )


def test_the_scenario_is_the_incidents_shape() -> None:
    """Formula inside usable, the measured child above it, raw free above that."""
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    room = usable(planner)
    formula = formula_at(32768)
    assert formula.weights_bytes == 64 * GIB
    assert formula.total_bytes < room
    assert formula.total_bytes * 1.032 / room > 1 / OBS_SAFETY, "measured above 91% of usable"
    assert isinstance(plan(planner, 32768), LoadPlan)


def test_a_load_measured_above_91pct_of_usable_plans_again() -> None:
    """The bug: a repeat of a load that had just run was refused forever."""
    formula = formula_at(32768)
    measured = int(formula.total_bytes * 1.032)
    planner = Planner(
        make_config(),
        rig_5090x2_3090x2(),
        observation_lookup=RowsByContext({32768: measured}),
        log_plans=False,
    )
    room = usable(planner)
    assert measured * OBS_SAFETY > room, "the whole-total rule refuses this repeat"

    repeat = plan(planner, 32768)

    assert isinstance(repeat, LoadPlan), repeat
    assert repeat.devices == THREE
    assert repeat.estimate.weights_bytes == formula.weights_bytes
    expected = formula.weights_bytes + (measured - formula.weights_bytes) * OBS_SAFETY
    assert repeat.estimate.total_bytes == pytest.approx(expected, abs=1024)
    assert repeat.estimate.total_bytes <= room
    # The split divides the corrected total, and every card can hold its share.
    assert sum(repeat.per_gpu_bytes.values()) == pytest.approx(
        repeat.estimate.total_bytes, rel=1e-3
    )
    for gpu in planner.probe.list_gpus():
        if gpu.index in THREE:
            assert repeat.per_gpu_bytes[gpu.index] <= planner.usable_bytes(gpu, forced=True)
    note = next(n for n in repeat.notes if n.startswith("estimate corrected x"))
    assert "weights, which are exact" in note


def test_a_pre_d73_rows_weights_figure_is_never_read(tmp_path: Any) -> None:
    """Rows written before D73 count the host-resident table in
    ``weights_bytes``: 32.8 GiB on the 177B that no GPU ever held. Read as the
    weights, a measurement BELOW them would fall back to the whole-total rule
    and re-create the refusal. The correction uses the current formula's."""
    formula = formula_at(32768)
    measured = int(formula.total_bytes * 1.032)
    db = Database(tmp_path / "registry.sqlite3")
    db.migrate()
    try:
        db.record_load_observation(
            model_id="test/model",
            ctx_size=32768,
            parallel=2,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            devices="0,1,2",
            predicted_bytes=formula.total_bytes + int(32.8 * GIB),
            actual_bytes=measured,
            weights_bytes=formula.weights_bytes + int(32.8 * GIB),
            ok=True,
            note=OBSERVATION_NOTE_PER_PID_DEVICE,
            per_gpu_planned='{"kv_unified": true}',
        )
        planner = Planner(
            make_config(), rig_5090x2_3090x2(), observation_lookup=db.matching_observation
        )
        repeat = plan(planner, 32768)
    finally:
        db.close()

    assert isinstance(repeat, LoadPlan), repeat
    assert [n for n in repeat.notes if n.startswith("estimate corrected x")]
    expected = formula.weights_bytes + (measured - formula.weights_bytes) * OBS_SAFETY
    assert repeat.estimate.total_bytes == pytest.approx(expected, abs=1024)
    assert repeat.estimate.weights_bytes == formula.weights_bytes


# ---------------------------------------------------------------------------
# The refusal walk (max_ctx_that_fits)
# ---------------------------------------------------------------------------


def test_a_refusal_after_a_correction_still_names_a_window_that_plans() -> None:
    """The cascade: 49152 measured above what three cards hold at the headroom,
    so it is refused -- and the refusal names 32768, which then loads."""
    formula = formula_at(49152)
    measured = int(formula.total_bytes * 1.032)
    lookup = RowsByContext({49152: measured})
    planner = Planner(
        make_config(), rig_5090x2_3090x2(), observation_lookup=lookup, log_plans=False
    )
    room = usable(planner)

    refused = plan(planner, 49152)

    assert isinstance(refused, LoadRejected), refused
    assert refused.required_bytes > room
    # What the subtraction at the refused window used to give: under the
    # whole-total rule the fixed cost alone was past the cards, so None.
    old = scaled_estimate(formula, measured * OBS_SAFETY / formula.total_bytes)
    assert old.total_bytes - old.kv_bytes > room
    assert refused.max_ctx_that_fits == 32768
    assert any("reduce context from 49152 to 32768" in s for s in refused.suggestions)
    assert isinstance(plan(planner, refused.max_ctx_that_fits), LoadPlan)


def test_the_walk_skips_a_window_whose_own_measurement_does_not_fit() -> None:
    """A rung is judged on what a load there is really charged: a measurement
    of that rung too big for the cards rules it out, and the walk goes on."""
    rows = {ctx: int(formula_at(ctx).total_bytes * 1.06) for ctx in (49152, 32768)}
    planner = Planner(
        make_config(),
        rig_5090x2_3090x2(),
        observation_lookup=RowsByContext(rows),
        log_plans=False,
    )
    assert not isinstance(plan(planner, 32768), LoadPlan), "the rung below is ruled out too"

    refused = plan(planner, 49152)

    assert isinstance(refused, LoadRejected)
    assert refused.max_ctx_that_fits == 24576
    assert isinstance(plan(planner, 24576), LoadPlan)


def test_the_walk_never_names_a_window_at_or_above_the_refused_one() -> None:
    """A measurement pins ONE window; the formula alone may still fit a larger
    rung (65536 here, by a few MB). A refusal answers "what smaller window
    loads", and ClawChat's re-ask only fires for a window below its own."""
    formula_65k = formula_at(65536)
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    assert formula_65k.total_bytes <= usable(planner), "the larger rung fits on the formula"

    rows = {40000: int(formula_at(40000).total_bytes * 1.06)}
    planner = Planner(
        make_config(), rig_5090x2_3090x2(), observation_lookup=RowsByContext(rows), log_plans=False
    )
    refused = plan(planner, 40000)

    assert isinstance(refused, LoadRejected)
    assert refused.max_ctx_that_fits == 32768


def test_the_walk_consults_the_table_only_until_a_rung_fits() -> None:
    """One lookup per rung the walk actually judges, and it stops at the first
    fit -- the refused window and the rung under it, nothing else."""
    formula = formula_at(49152)
    lookup = RowsByContext({49152: int(formula.total_bytes * 1.032)})
    planner = Planner(
        make_config(), rig_5090x2_3090x2(), observation_lookup=lookup, log_plans=False
    )

    refused = plan(planner, 49152)

    assert isinstance(refused, LoadRejected)
    assert set(lookup.asked) == {49152, 32768}


def test_a_refusal_with_no_window_that_fits_still_says_none() -> None:
    """The walk never invents a window: a model too big for the cards at the
    smallest rung is refused with ``max_ctx_that_fits`` unset."""
    huge = make_record(
        meta=make_meta(tensor_bytes=90 * GIB, n_ctx_train=262144),
        size_bytes=90 * GIB,
        settings=ModelSettings(device_override=THREE),
    )
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    refused = planner.plan_load(huge, ctx_size=8192, parallel=1, kv_cache_type="f16")
    assert isinstance(refused, LoadRejected)
    assert refused.max_ctx_that_fits is None
