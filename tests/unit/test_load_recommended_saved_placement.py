"""load-recommended honours the model's saved placement the way ``/load`` does (D76).

The incident: the owner pinned the 177B chat model to CUDA 0, 1 and 2 with a
saved ``device_override`` to keep CUDA 3 for ComfyUI. ``/load`` and every
on-demand load honoured it (D36/D59). ``load-recommended`` -- the one path
ClawChat loads through -- did not: every hardware mode is planned as a
``device_override`` copy of the record, so the saved override was replaced by
``dual_5090``, ``dual_3090``, ``all_gpus`` and ``single_5090`` in turn, and a
saved ``allowed_devices`` was not consulted at all unless the request carried a
bound of its own. Nothing could keep this model off CUDA 3 on that path.

What these pin:

* a saved ``device_override`` is the only placement the walk offers -- in the
  real call, in the dry run and over MCP -- and the response says why;
* a saved ``allowed_devices`` bounds every mode, is intersected with a
  request's bound, and never leaves an empty or duplicate mode;
* a request bound that excludes a card the saved override names is the same
  400 as on ``/load``;
* the shortcut does not hand back a resident standing off the saved placement;
* leases and ``planner.excluded_devices`` behave as before.
"""

from __future__ import annotations

import pytest

from studioforge.core.leases import LeaseBook
from studioforge.errors import BadRequestError, InsufficientVramError
from studioforge.types import ModelSettings
from tests.unit.test_allowed_devices_one_shot import make_manager
from tests.unit.test_load_recommended import MODEL, resident_self

THREE_CARDS = [0, 1, 2]


def _mode_devices(dry: dict) -> list[tuple[int, ...]]:
    return [tuple(mode["devices"]) for mode in dry["modes"]]


# ---------------------------------------------------------------------------
# A saved device_override is the placement
# ---------------------------------------------------------------------------


async def test_without_a_saved_placement_the_walk_still_leads_with_the_5090_pair() -> None:
    """The control: nothing saved, nothing changed (the headline mode wins)."""
    manager, _supervisor, _registry = make_manager()
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [0, 1]
    assert not any("saves settings" in note for note in instance.plan.notes)


async def test_a_saved_device_override_is_the_only_placement_the_walk_offers() -> None:
    """The owner's three cards, exactly -- not the 5090 pair the walk leads
    with, and never the four-card all_gpus that includes ComfyUI's card."""
    manager, supervisor, registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS)
    )

    instance = await manager.load_recommended(MODEL, 32768, min_slots=2, max_slots=2)

    assert supervisor.starts == 1
    assert instance.plan is not None
    assert instance.plan.devices == THREE_CARDS
    note = next(n for n in instance.plan.notes if "saves settings.device_override" in n)
    assert "[0, 1, 2]" in note and "Clear the override" in note
    # Honoured, not rewritten: the saved row is exactly what it was.
    assert registry.saved == []
    assert registry.resolve(MODEL).settings.device_override == THREE_CARDS


async def test_the_dry_run_reports_the_saved_placement_the_real_call_takes() -> None:
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS)
    )

    dry = await manager.plan_recommended(MODEL, 32768, min_slots=2, max_slots=2)

    assert supervisor.starts == 0
    assert dry["fits"] is True
    assert dry["mode"] == "device_override"
    assert dry["devices"] == THREE_CARDS
    assert dry["device_override"] == THREE_CARDS
    assert _mode_devices(dry) == [tuple(THREE_CARDS)], "one placement walked, not four"
    assert any("saves settings.device_override" in n for n in dry["notes"])

    instance = await manager.load_recommended(MODEL, 32768, min_slots=2, max_slots=2)
    assert instance.plan is not None and instance.plan.devices == dry["devices"]


async def test_the_saved_order_is_kept_because_the_engine_splits_in_list_order() -> None:
    """llama.cpp places the output layer on the LAST device and splits in list
    order, so ``[2, 0, 1]`` is not ``[0, 1, 2]`` -- ``/load`` keeps the saved
    order, and so does the walk."""
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=[2, 0, 1])
    )
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None
    assert instance.plan.devices == [2, 0, 1]


async def test_a_saved_override_that_is_a_hardware_mode_keeps_that_modes_name() -> None:
    """``[2, 3]`` is the 3090 pair: ``prefer_mode="dual_3090"`` still matches."""
    manager, _supervisor, _registry = make_manager(settings=ModelSettings(device_override=[2, 3]))

    dry = await manager.plan_recommended(MODEL, 32768)
    assert dry["mode"] == "dual_3090"
    assert "saved device_override" in dry["label"]

    instance = await manager.load_recommended(MODEL, 32768, prefer_modes=["dual_3090"])
    assert instance.plan is not None and instance.plan.devices == [2, 3]


async def test_a_prefer_mode_the_saved_override_rules_out_is_a_400_naming_it() -> None:
    """Contradicting the model's own placement is stated, not resolved silently
    in either direction (D59's rule for contradictions)."""
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS)
    )

    for call in (manager.load_recommended, manager.plan_recommended):
        with pytest.raises(BadRequestError) as excinfo:
            await call(MODEL, 32768, prefer_modes=["dual_5090"])
        assert excinfo.value.param == "prefer_modes"
        assert excinfo.value.details["device_override"] == THREE_CARDS
        assert "device_override" in excinfo.value.message
    assert supervisor.starts == 0


async def test_a_request_bound_excluding_an_override_card_stays_a_400_as_on_load() -> None:
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS)
    )

    for call in (manager.load_recommended, manager.plan_recommended):
        with pytest.raises(BadRequestError) as excinfo:
            await call(MODEL, 32768, allowed_devices=[0, 1, 3])
        assert excinfo.value.param == "allowed_devices"
        assert excinfo.value.details["device_override"] == THREE_CARDS
        assert excinfo.value.details["excluded_by_request"] == [2]
    assert supervisor.starts == 0


async def test_a_request_bound_that_covers_the_override_takes_the_override() -> None:
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS)
    )
    instance = await manager.load_recommended(MODEL, 32768, allowed_devices=[0, 1, 2, 3])
    assert instance.plan is not None
    assert instance.plan.devices == THREE_CARDS


async def test_a_refusal_on_the_saved_placement_names_it_and_the_setting() -> None:
    """A model that does not fit its owner's three cards is refused as THAT --
    not as "any placement of this box", which would send the caller hunting
    for a mode it was never offered."""
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS), free_gib=4.0
    )

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 131072, min_slots=2, max_slots=2)

    error = excinfo.value
    assert supervisor.starts == 0
    assert "the placement its saved device_override names" in error.message
    assert error.details["device_override"] == THREE_CARDS
    assert [m["devices"] for m in error.details["modes"]] == [THREE_CARDS]
    assert any("saves settings.device_override" in s for s in error.details["suggestions"])

    dry = await manager.plan_recommended(MODEL, 131072, min_slots=2, max_slots=2)
    assert dry["fits"] is False and dry["message"] == error.message
    assert any("saves settings.device_override" in n for n in dry["notes"])


async def test_a_resident_off_the_saved_placement_is_moved_not_handed_back() -> None:
    """The shortcut answered "already loaded" without asking where. A resident
    that predates the saved override and stands on CUDA 3 is reloaded onto the
    owner's cards; one already on them is handed back untouched."""
    sprawled = resident_self(ctx=32768, devices=[0, 1, 2, 3])
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS), loaded=[sprawled]
    )
    instance = await manager.load_recommended(MODEL, 32768)
    assert supervisor.starts == 1
    assert instance.plan is not None and instance.plan.devices == THREE_CARDS

    placed = resident_self(ctx=32768, devices=THREE_CARDS)
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS), loaded=[placed]
    )
    same = await manager.load_recommended(MODEL, 32768)
    assert supervisor.starts == 0, "a resident on the saved placement is already that"
    assert same is placed


async def test_a_lease_on_an_override_card_still_refuses_and_names_the_override() -> None:
    """Leases are unchanged (D43/D64): the walk never lands on a leased card,
    and because the override is the owner's own, the advice is to clear it."""
    book = LeaseBook()
    book.acquire([2], holder="clawforge2", model_ids=["comfy/thing"])
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=THREE_CARDS), leases=book
    )

    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 32768)

    assert supervisor.starts == 0
    assert excinfo.value.code == "gpu_leased"
    assert excinfo.value.details["device_override"] == THREE_CARDS


async def test_a_saved_override_naming_an_excluded_card_wins_as_it_does_on_load() -> None:
    """``planner.excluded_devices`` is policy; a saved override is the owner's
    explicit placement and outranks it on ``/load`` (D19). The walk agrees with
    /load rather than inventing a third answer -- and the plan says so."""
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=[2, 3]), excluded_devices=[3]
    )
    via_load = await manager.load(MODEL, ctx_size=32768)
    assert via_load.plan is not None and via_load.plan.devices == [2, 3]

    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(device_override=[2, 3]), excluded_devices=[3]
    )
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None and instance.plan.devices == [2, 3]
    assert any("excluded_devices" in note for note in instance.plan.notes)


# ---------------------------------------------------------------------------
# A saved allowed_devices bounds every mode
# ---------------------------------------------------------------------------


async def test_a_saved_allowed_devices_narrows_every_mode() -> None:
    """The ClawChat shape: all_gpus over the permitted cards is three cards,
    not four, and no mode reaches CUDA 3."""
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(allowed_devices=THREE_CARDS)
    )

    dry = await manager.plan_recommended(MODEL, 32768)

    assert dry["allowed_devices"] == THREE_CARDS
    assert dry["device_override"] is None
    walked = _mode_devices(dry)
    assert all(3 not in devices for devices in walked), walked
    assert (0, 1, 2) in walked, "all_gpus over the permitted cards"
    assert any("saves settings.allowed_devices" in n for n in dry["notes"])

    instance = await manager.load_recommended(MODEL, 32768, prefer_modes=["all_gpus"])
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == THREE_CARDS


async def test_a_saved_allowed_devices_is_intersected_with_a_request_bound() -> None:
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(allowed_devices=THREE_CARDS)
    )

    dry = await manager.plan_recommended(MODEL, 32768, allowed_devices=[1, 2, 3])

    assert dry["allowed_devices"] == [1, 2]
    assert all(set(devices) <= {1, 2} for devices in _mode_devices(dry))
    instance = await manager.load_recommended(MODEL, 32768, allowed_devices=[1, 2, 3])
    assert instance.plan is not None and set(instance.plan.devices) <= {1, 2}


async def test_a_saved_allowed_devices_disjoint_from_the_request_is_a_400() -> None:
    manager, supervisor, _registry = make_manager(settings=ModelSettings(allowed_devices=[2, 3]))
    with pytest.raises(BadRequestError) as excinfo:
        await manager.load_recommended(MODEL, 32768, allowed_devices=[0, 1])
    assert excinfo.value.param == "allowed_devices"
    assert excinfo.value.details["settings_allowed_devices"] == [2, 3]
    assert supervisor.starts == 0


async def test_a_one_card_bound_leaves_one_mode_never_an_empty_or_repeated_one() -> None:
    """Modes are built FROM the permitted cards, so narrowing cannot leave a
    "dual" mode of one card, an empty mode, or two modes over the same set."""
    manager, _supervisor, _registry = make_manager(settings=ModelSettings(allowed_devices=[2]))

    dry = await manager.plan_recommended(MODEL, 16384)

    assert [(m["mode"], m["devices"]) for m in dry["modes"]] == [("single_3090", [2])]
    walked = _mode_devices(dry)
    assert len(walked) == len(set(walked))


async def test_excluded_devices_still_narrow_a_saved_bound() -> None:
    """A saved bound does not unlock a card the planner excludes (D19/D59)."""
    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(allowed_devices=[2, 3]), excluded_devices=[3]
    )
    dry = await manager.plan_recommended(MODEL, 16384)
    assert _mode_devices(dry) == [(2,)]

    manager, _supervisor, _registry = make_manager(
        settings=ModelSettings(allowed_devices=[3]), excluded_devices=[3]
    )
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 16384)
    assert "the model's saved allowed_devices" in excinfo.value.message


async def test_a_resident_outside_a_saved_bound_is_moved_not_handed_back() -> None:
    sprawled = resident_self(ctx=32768, devices=[2, 3])
    manager, supervisor, _registry = make_manager(
        settings=ModelSettings(allowed_devices=[0, 1]), loaded=[sprawled]
    )
    instance = await manager.load_recommended(MODEL, 32768)
    assert supervisor.starts == 1
    assert instance.plan is not None and set(instance.plan.devices) <= {0, 1}
