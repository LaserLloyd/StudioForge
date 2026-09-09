"""A per-request ``allowed_devices``: choose among these, do not use exactly these (D59).

The incident behind it: ComfyUI holds one card on this rig, and when no lease
stands on it the planner and the D42 rebalancer both -- correctly, by their own
rules -- treat it as free and place models there, starving the renders. A real
line from the log::

    rebalanced model devices=[3, 2] reason='shares GPU[0] with another
    resident; [2, 3] is free of it'

The lease is the primary fix. This is the tenant's own half of it: a way to say
"anywhere but that card" on a single load, for the windows where no lease can be
held, without reaching for ``devices`` -- which is a *forced* placement (use
exactly these, all of them, more than one becomes a split) and therefore takes
the placement decision away from the component that makes it best.

What every test below is guarding:

* the two parameters stay different things, and cannot be confused for one
  another by sending both;
* a one-shot **narrows**. It is not a way for any client that can reach the
  port to talk its way onto a card an operator setting, an exclusion or another
  tenant's lease has closed;
* nothing it says is ever written to the registry;
* omitting it changes nothing at all.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core.leases import LeaseBook
from studioforge.core.manager import ModelManager, validate_load_args
from studioforge.core.planner import Planner
from studioforge.errors import BadRequestError, InsufficientVramError
from studioforge.types import GB, LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_catalog import dense_meta, record
from tests.unit.test_load_recommended import MODEL, ParallelDb, Registry, serving
from tests.unit.test_load_retry import StubSupervisor
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

# ---------------------------------------------------------------------------
# Harness -- the four-card rig, a real planner, a supervisor that only counts
# ---------------------------------------------------------------------------


def make_manager(
    *,
    settings: ModelSettings | None = None,
    leases: LeaseBook | None = None,
    excluded_devices: list[int] | None = None,
    loaded: list[Any] | None = None,
    free_gib: float = 31.0,
) -> tuple[ModelManager, StubSupervisor, Registry]:
    rec = record(
        MODEL,
        dense_meta(131072),
        mtime=1.0,
        size_bytes=8 * GB,
        settings=settings,
    )
    config = make_config(
        **({"excluded_devices": excluded_devices} if excluded_devices is not None else {})
    )
    planner = Planner(config, rig_5090x2_3090x2(free_gib), leases=leases, log_plans=False)
    supervisor = StubSupervisor()
    for instance in loaded or []:
        supervisor.instances[instance.model_id] = instance
    registry = Registry([rec])
    manager = ModelManager(
        config,
        registry=registry,  # type: ignore[arg-type]
        planner=planner,
        supervisor=supervisor,  # type: ignore[arg-type]
        db=ParallelDb(),  # type: ignore[arg-type]
    )
    return manager, supervisor, registry


# ---------------------------------------------------------------------------
# Omitted, nothing moved
# ---------------------------------------------------------------------------


async def test_omitting_allowed_devices_places_exactly_where_it_placed_before() -> None:
    """The no-op guarantee, asserted rather than assumed.

    A new load argument earns its place only if a body that does not carry it
    means precisely what it meant yesterday -- the headline pair, chosen by the
    planner, with no restriction note on the plan.
    """
    # Two identical boxes, so the second load is a cold one too and its plan is
    # comparable note for note -- a forced reload on one manager would add its
    # own credit note and hide exactly the difference this test looks for.
    plain = await make_manager()[0].load(MODEL, ctx_size=32768)
    explicit_null = await make_manager()[0].load(MODEL, ctx_size=32768, allowed_devices=None)

    assert plain.plan is not None and explicit_null.plan is not None
    assert plain.plan.devices == [0]
    assert not any("allowed_devices" in note for note in plain.plan.notes)
    assert explicit_null.plan.devices == plain.plan.devices
    assert explicit_null.plan.notes == plain.plan.notes
    assert explicit_null.plan.ctx_size == plain.plan.ctx_size
    assert explicit_null.plan.parallel == plain.plan.parallel


# ---------------------------------------------------------------------------
# It narrows the candidate set; the planner still chooses inside it
# ---------------------------------------------------------------------------


async def test_a_one_shot_narrows_the_set_and_the_planner_still_chooses_within_it() -> None:
    """The point of the parameter: a bound, not a placement.

    ``devices=[2, 3]`` would have forced both cards and a split. This names
    three and lets the planner rank them, which is the decision it makes better
    than the caller -- the caller only knew which card to stay off.
    """
    manager, _supervisor, _registry = make_manager()

    instance = await manager.load(MODEL, ctx_size=32768, allowed_devices=[1, 2, 3])

    assert instance.plan is not None
    assert set(instance.plan.devices) <= {1, 2, 3}, instance.plan.devices
    assert 0 not in instance.plan.devices
    assert any("allowed_devices" in note for note in instance.plan.notes)


async def test_a_one_shot_is_a_choice_not_a_split_across_everything_named() -> None:
    """The distinction from ``devices`` in one assertion.

    Three cards named, one card used: naming a set must never become a
    three-way tensor split, which is exactly what ``devices=[1, 2, 3]`` means
    and why it was the wrong lever to hand a tenant.
    """
    manager, _supervisor, _registry = make_manager()

    instance = await manager.load(MODEL, ctx_size=8192, allowed_devices=[1, 2, 3])

    assert instance.plan is not None
    assert len(instance.plan.devices) == 1, instance.plan.devices


async def test_the_one_shot_is_never_written_to_the_registry() -> None:
    """One-shot means one shot (D36's rule, D59's parameter).

    A bound that persisted itself would quietly become an operator setting
    nobody chose, and the next JIT load of the model would inherit a
    restriction written by whichever client happened to load it last.
    """
    manager, _supervisor, registry = make_manager()

    await manager.load(MODEL, ctx_size=32768, allowed_devices=[2, 3])

    assert registry.saved == [], "a one-shot load argument wrote to the registry"
    assert registry.resolve(MODEL).settings.allowed_devices is None


async def test_the_next_load_without_it_is_unbounded_again() -> None:
    """The observable half of "never persisted": the placement comes back."""
    manager, _supervisor, _registry = make_manager()

    bounded = await manager.load(MODEL, ctx_size=32768, allowed_devices=[2, 3])
    assert bounded.plan is not None
    assert set(bounded.plan.devices) <= {2, 3}

    after = await manager.load(MODEL, ctx_size=32768, force=True)
    assert after.plan is not None
    assert after.plan.devices == [0], "the bound outlived its load"


# ---------------------------------------------------------------------------
# Precedence: narrow, never widen
# ---------------------------------------------------------------------------


async def test_a_one_shot_cannot_widen_a_persisted_allowed_devices() -> None:
    """The escalation this rule exists to refuse.

    The model's saved settings say "cards 2 and 3 only". A request naming all
    four must not become permission to use 0 and 1: a per-request argument that
    could out-vote an operator setting would make every such setting advisory
    to anyone who can reach the port.
    """
    manager, _supervisor, _registry = make_manager(settings=ModelSettings(allowed_devices=[2, 3]))

    instance = await manager.load(MODEL, ctx_size=32768, allowed_devices=[0, 1, 2, 3])

    assert instance.plan is not None
    assert set(instance.plan.devices) <= {2, 3}, instance.plan.devices


async def test_a_one_shot_disjoint_from_the_persisted_set_is_a_400_not_a_guess() -> None:
    """Nothing is left to choose among, and both numbers are in the message.

    Neither silent answer is defensible -- honouring the request overrules the
    operator, honouring the setting places the load on the card the caller
    asked it to avoid -- so the contradiction is stated instead of resolved.
    """
    manager, _supervisor, _registry = make_manager(settings=ModelSettings(allowed_devices=[2, 3]))

    with pytest.raises(BadRequestError) as excinfo:
        await manager.load(MODEL, ctx_size=32768, allowed_devices=[0, 1])

    assert excinfo.value.param == "allowed_devices"
    assert excinfo.value.details["settings_allowed_devices"] == [2, 3]
    assert excinfo.value.details["allowed_devices"] == [0, 1]
    assert "narrows" in excinfo.value.message


async def test_a_device_override_still_outranks_and_a_contradiction_is_refused() -> None:
    """``device_override`` beats every allow-list -- so a clash cannot be quiet.

    The dangerous shape is the second one: the override names the exact card
    the request excluded, and the pre-D59 planner would have honoured the
    override without a word, which is the starved-renders bug arriving through
    the feature written to prevent it.
    """
    manager, _supervisor, _registry = make_manager(settings=ModelSettings(device_override=[2]))

    # Compatible: the override lies inside the set, so it simply wins.
    agreed = await manager.load(MODEL, ctx_size=32768, allowed_devices=[1, 2, 3])
    assert agreed.plan is not None
    assert agreed.plan.devices == [2]

    with pytest.raises(BadRequestError) as excinfo:
        await manager.load(MODEL, ctx_size=32768, allowed_devices=[1, 3], force=True)

    assert excinfo.value.param == "allowed_devices"
    assert excinfo.value.details["device_override"] == [2]
    assert excinfo.value.details["excluded_by_request"] == [2]


async def test_a_one_shot_does_not_unlock_an_excluded_card() -> None:
    """``planner.excluded_devices`` reserves cards for other software (D19).

    Only a ``device_override`` beats it. A one-shot must not, or the exclusion
    becomes a suggestion -- and the software it reserves the card for is
    usually the very thing this parameter exists to protect.
    """
    manager, supervisor, _registry = make_manager(excluded_devices=[2, 3])

    with pytest.raises(InsufficientVramError):
        await manager.load(MODEL, ctx_size=32768, allowed_devices=[2, 3])
    assert supervisor.starts == 0, "an exclusion was talked out of by a load argument"

    # And the exclusion does not poison the rest of the set: a bound that also
    # names a permitted card still loads, on that card.
    instance = await manager.load(MODEL, ctx_size=32768, allowed_devices=[1, 2, 3])
    assert instance.plan is not None
    assert set(instance.plan.devices) == {1}, instance.plan.devices


async def test_a_one_shot_naming_only_leased_cards_is_an_honest_507() -> None:
    """A lease outranks the request, and the refusal says which lease (D43/D53).

    The client needs to know this is waitable and whose it is -- not a bare
    "does not fit", which reads as a sizing problem and sends it retrying with
    a smaller context that will fail the same way.
    """
    book = LeaseBook()
    lease = book.acquire([2, 3], holder="crucibleforge", model_ids=["other/model"])
    manager, _supervisor, _registry = make_manager(leases=book)

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load(MODEL, ctx_size=32768, allowed_devices=[2, 3])

    assert excinfo.value.code == "gpu_leased"
    assert [entry["id"] for entry in excinfo.value.details["leases"]] == [lease.id]
    assert excinfo.value.details["retry_after_s"] is not None


async def test_a_set_where_nothing_fits_refuses_with_the_reason_not_a_crash() -> None:
    """The set is real, the cards are free, the model is simply too big for them.

    That stays the ordinary shortfall: ``insufficient_vram`` with the
    arithmetic, no lease blamed. And the arithmetic is done over the bounded
    set -- two GPUs, not the box's four -- which is the observable proof that
    the bound reached the estimator rather than being applied as an
    afterthought to a plan made over every card.
    """
    manager, _supervisor, _registry = make_manager()

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load(MODEL, ctx_size=8_000_000, allowed_devices=[2, 3])

    assert excinfo.value.code == "insufficient_vram"
    assert excinfo.value.details.get("leases") in (None, [])
    assert "across 2 GPUs" in excinfo.value.message, excinfo.value.message


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_an_empty_allowed_devices_is_refused_rather_than_ignored() -> None:
    """The most dangerous silent no-op available here.

    A caller that computes "every card except the one ComfyUI has" on a
    single-card box gets ``[]``. Read as "no constraint", the load lands on
    exactly that card -- the failure this parameter exists to prevent, produced
    by the parameter meant to prevent it.
    """
    with pytest.raises(BadRequestError) as excinfo:
        validate_load_args(
            ctx_size=None, parallel=None, kv_cache_type=None, allowed_devices=[], known_devices=[0]
        )
    assert excinfo.value.param == "allowed_devices"
    assert "at least one" in excinfo.value.message


def test_allowed_devices_rejects_a_shape_no_load_could_use() -> None:
    """Non-list, non-int, negative and unknown -- each a 400 naming the parameter.

    Named rather than generic because the alternative is a planner refusal
    several frames deeper that reads like a VRAM problem (the same reason
    ``devices`` is checked against the probe's list).
    """
    for bad in ("0,1", 3, {"0": True}):
        with pytest.raises(BadRequestError) as excinfo:
            validate_load_args(
                ctx_size=None, parallel=None, kv_cache_type=None, allowed_devices=bad
            )
        assert excinfo.value.param == "allowed_devices"

    for bad_member in ([0, "1"], [0, 1.5], [0, True], [0, None]):
        with pytest.raises(BadRequestError) as excinfo:
            validate_load_args(
                ctx_size=None, parallel=None, kv_cache_type=None, allowed_devices=bad_member
            )
        assert excinfo.value.param == "allowed_devices"

    with pytest.raises(BadRequestError) as excinfo:
        validate_load_args(ctx_size=None, parallel=None, kv_cache_type=None, allowed_devices=[-1])
    assert ">= 0" in excinfo.value.message

    with pytest.raises(BadRequestError) as excinfo:
        validate_load_args(
            ctx_size=None,
            parallel=None,
            kv_cache_type=None,
            allowed_devices=[7],
            known_devices=[0, 1, 2, 3],
        )
    assert "does not have" in excinfo.value.message


def test_a_repeated_index_is_collapsed_because_a_set_is_a_set() -> None:
    """Where ``allowed_devices`` and ``devices`` are deliberately different.

    ``devices=[1, 1]`` is refused: a repeat in a forced placement is a
    meaningless tensor split. A repeat in a set the planner chooses within is
    unambiguous, so refusing it would be pedantry that breaks a caller
    concatenating two lists.
    """
    validate_load_args(
        ctx_size=None,
        parallel=None,
        kv_cache_type=None,
        allowed_devices=[1, 1, 2],
        known_devices=[0, 1, 2, 3],
    )
    with pytest.raises(BadRequestError):
        validate_load_args(
            ctx_size=None,
            parallel=None,
            kv_cache_type=None,
            devices=[1, 1],
            known_devices=[0, 1, 2, 3],
        )


def test_devices_and_allowed_devices_together_are_refused() -> None:
    """Both set means the allow-list decides nothing, so it is stated.

    Silently ignoring one of two placement arguments is how a caller ends up
    believing a card is protected when it is not.
    """
    with pytest.raises(BadRequestError) as excinfo:
        validate_load_args(
            ctx_size=None,
            parallel=None,
            kv_cache_type=None,
            devices=[0],
            allowed_devices=[1, 2],
            known_devices=[0, 1, 2, 3],
        )
    assert excinfo.value.param == "allowed_devices"
    assert "forced placement" in excinfo.value.message


# ---------------------------------------------------------------------------
# load-recommended: the same bound, applied where the modes are built
# ---------------------------------------------------------------------------


async def test_load_recommended_walks_only_the_modes_over_the_permitted_cards() -> None:
    """The headline mode is the 5090 pair; the bound has to beat it.

    Each mode is planned as a ``device_override`` copy of the record, and an
    override is precisely what outranks an allow-list -- so a mode left in the
    list is a card this load can still land on. WP22 learned that about
    ``excluded_devices``; the bound is applied in the same place for the same
    reason.
    """
    manager, _supervisor, _registry = make_manager()

    instance = await manager.load_recommended(MODEL, 32768, allowed_devices=[2, 3])

    assert instance.plan is not None
    assert set(instance.plan.devices) <= {2, 3}, instance.plan.devices


async def test_load_recommended_persists_the_profile_without_the_bound() -> None:
    """``persist`` writes what was resolved; the bound is not part of it.

    Devices are deliberately not persisted (D36), and a bound on devices is the
    same kind of thing -- otherwise one client's momentary restriction becomes
    the model's standing configuration.
    """
    manager, _supervisor, registry = make_manager()

    await manager.load_recommended(MODEL, 32768, allowed_devices=[2, 3], persist=True)

    assert registry.saved, "persist=true wrote nothing at all"
    _model_id, saved = registry.saved[-1]
    assert saved.allowed_devices is None
    assert saved.device_override is None


async def test_load_recommended_will_not_hand_back_a_resident_outside_the_bound() -> None:
    """The shortcut that would have defeated the whole parameter.

    A model already resident at exactly the requested context returns
    immediately -- and used to do so without looking at where it was sitting,
    so the fastest path through the function was also the one that ignored the
    restriction. It must reload instead.
    """
    resident = serving(MODEL, requests=0, devices=[0, 1])
    resident.plan.ctx_per_slot = 32768  # type: ignore[union-attr]
    manager, supervisor, _registry = make_manager(loaded=[resident])

    instance = await manager.load_recommended(MODEL, 32768, allowed_devices=[2, 3])

    assert supervisor.starts == 1, "the out-of-bounds resident was handed straight back"
    assert instance.plan is not None
    assert set(instance.plan.devices) <= {2, 3}


async def test_load_recommended_refuses_when_the_bound_leaves_no_card() -> None:
    """An exclusion and a bound can between them leave nothing, and it says so.

    "No usable GPU was found" alone would send the operator to `nvidia-smi` on
    a rig whose cards are all healthy.
    """
    manager, _supervisor, _registry = make_manager(excluded_devices=[2, 3])

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 32768, allowed_devices=[2, 3])

    assert "allowed_devices" in excinfo.value.message
    assert any("allowed_devices" in s for s in excinfo.value.details["suggestions"])


async def test_load_recommended_names_the_narrowed_modes_when_prefer_mode_is_unknown() -> None:
    """A mode that exists on the box but not inside the bound is not silently ignored."""
    manager, _supervisor, _registry = make_manager()

    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(
            MODEL, 32768, prefer_modes=["dual_5090"], allowed_devices=[2, 3]
        )

    assert excinfo.value.param == "prefer_modes"
    assert "allowed_devices" in excinfo.value.message


async def test_load_recommended_without_the_bound_is_unchanged() -> None:
    """The no-op guarantee on the second route, asserted the same way."""
    manager, _supervisor, _registry = make_manager()

    instance = await manager.load_recommended(MODEL, 32768)

    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [0, 1]


# ---------------------------------------------------------------------------
# The planner's own messages no longer claim a saved setting was the cause
# ---------------------------------------------------------------------------


def test_the_restriction_note_does_not_blame_a_setting_the_caller_never_wrote() -> None:
    """Both sources land on ``settings.allowed_devices``; only one is a setting.

    The note used to read "by the model's allowed_devices setting", which for a
    one-shot sends the reader to edit a row that says nothing.
    """
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False)
    rec = record(MODEL, dense_meta(131072), mtime=1.0, size_bytes=8 * GB)
    bounded = rec.model_copy(
        update={"settings": rec.settings.model_copy(update={"allowed_devices": [2, 3]})}
    )

    plan = planner.plan_load(bounded, ctx_size=8192)

    assert isinstance(plan, LoadPlan), plan
    note = next(n for n in plan.notes if "allowed_devices" in n)
    assert "the model's" not in note

    empty = rec.model_copy(
        update={"settings": rec.settings.model_copy(update={"allowed_devices": [7]})}
    )
    refused = planner.plan_load(empty, ctx_size=8192)
    assert isinstance(refused, LoadRejected), refused
    assert refused.reason_code == "allowed_devices_unavailable"
    assert "the model's" not in refused.reason
