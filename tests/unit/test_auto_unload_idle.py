"""One auto-unload timer for every model whose duration nobody stated (D75).

The owner runs the cards all day and wants anything nobody asked to keep
loaded gone after ten minutes idle -- chat and agent tiers included, which the
shipped tier map keeps for fifteen. ``models.auto_unload_idle_s`` is that
timer: off by default (the ladder is unchanged), and when set it outranks the
tier map and ``default_ttl_s`` for any model with no ``settings.ttl_s`` and no
pin. A duration somebody did state -- per model, or on a request -- wins.
"""

from __future__ import annotations

import pytest

from studioforge.api import openai_routes
from studioforge.config import Config, ModelsConfig
from tests.unit.test_gateway_lifecycle import make_record
from tests.unit.test_request_ttl_cap import SHIPPED_TIERS, state_for

AUTO = 600


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_off_by_default_the_tier_ladder_is_unchanged(tier: int) -> None:
    assert ModelsConfig().auto_unload_idle_s is None
    _state, instance = state_for(make_record(), priority=tier)
    assert instance.ttl_s == SHIPPED_TIERS[tier]


@pytest.mark.parametrize("tier", [1, 2, 3])
def test_on_every_unstated_model_gets_the_one_timer(tier: int) -> None:
    """The chat tier's fifteen minutes becomes ten; so does every other tier."""
    state, instance = state_for(make_record(), priority=tier, auto_unload_idle_s=AUTO)
    assert instance.ttl_s == AUTO
    assert state.manager.ttl_for(make_record(), priority=tier) == AUTO


def test_it_outranks_default_ttl_s_too() -> None:
    _state, instance = state_for(
        make_record(), priority=3, ttl_by_priority={}, default_ttl_s=1800, auto_unload_idle_s=AUTO
    )
    assert instance.ttl_s == AUTO


def test_a_per_model_duration_overrides_it() -> None:
    _state, instance = state_for(make_record(ttl_s=7200), priority=1, auto_unload_idle_s=AUTO)
    assert instance.ttl_s == 7200


def test_a_pin_overrides_it() -> None:
    _state, instance = state_for(make_record(pinned=True), priority=1, auto_unload_idle_s=AUTO)
    assert instance.ttl_s == 0


def test_a_request_duration_overrides_it_in_both_directions() -> None:
    """No tier ceiling under D75: a stated duration is honoured as stated."""
    record = make_record()
    state, instance = state_for(record, priority=3, auto_unload_idle_s=AUTO)
    assert state.manager.request_ttl_cap(record.id) is None
    openai_routes._apply_ttl_override(state, record.id, 3600)
    assert instance.ttl_s == 3600
    openai_routes._apply_ttl_override(state, record.id, 120)
    assert instance.ttl_s == 120


def test_a_retier_keeps_a_stated_request_duration() -> None:
    record = make_record()
    state, instance = state_for(record, priority=3, auto_unload_idle_s=AUTO)
    openai_routes._apply_ttl_override(state, record.id, 3600)
    state.manager._retier_resident(record, instance, 1)
    assert instance.priority == 1
    assert instance.ttl_s == 3600


def test_zero_is_refused_use_unset_or_a_pin() -> None:
    with pytest.raises(ValueError):
        ModelsConfig(auto_unload_idle_s=0)
    assert Config(models=ModelsConfig(auto_unload_idle_s=AUTO)).models.auto_unload_idle_s == AUTO
