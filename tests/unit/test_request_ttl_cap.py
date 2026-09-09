"""A request may shorten its model's idle timer, never out-live its tier (D61).

``/v1/chat/completions`` and ``/v1/completions`` accept LM Studio's ``ttl``
and write it onto the instance's idle timer. Until D61 only a pin was
protected, so on 2026-09-10 a background turn sending ``ttl: 3600`` with no
priority left a tier-3 JIT load resident for an hour against the policy's
ten minutes. Pinned here: the tier map's price is the ceiling, a request may
go below it, a per-model ``ttl_s`` and a pin are untouched, an empty or
partial map is the old behaviour, and the cap is logged at DEBUG with the
four facts a log review needs.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.api import openai_routes
from studioforge.types import InstanceInfo, ModelRecord
from tests.unit.test_gateway_lifecycle import loaded, make_manager, make_record

SHIPPED_TIERS = {1: 900, 2: 900, 3: 600}


class _Recorder:
    """Keeps the DEBUG lines; swallows the rest."""

    def __init__(self) -> None:
        self.debugs: list[tuple[str, dict[str, Any]]] = []

    def debug(self, event: str, **fields: Any) -> None:
        self.debugs.append((event, fields))

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


def state_for(
    record: ModelRecord, *, priority: int = 3, **models_cfg: Any
) -> tuple[Any, InstanceInfo]:
    """A route ``state`` the way the JIT path leaves it: loaded, tiered, TTL stamped."""
    models_cfg.setdefault("ttl_by_priority", dict(SHIPPED_TIERS))
    models_cfg.setdefault("default_ttl_s", 600)
    manager, supervisor = make_manager([record], **models_cfg)
    instance = loaded(record.id, None)
    instance.priority = priority
    supervisor.instances[record.id] = instance
    manager.apply_effective_ttl(record, instance)  # what the load stamps (D60)
    return SimpleNamespace(supervisor=supervisor, manager=manager), instance


def test_a_background_request_cannot_out_live_its_tier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live case: ``ttl: 3600`` with no priority on a tier-3 JIT load."""
    recorder = _Recorder()
    monkeypatch.setattr(openai_routes, "log", recorder)
    record = make_record("chat/v13-model")
    state, instance = state_for(record, priority=3)
    assert instance.ttl_s == 600, "the load stamped the tier's price"

    openai_routes._apply_ttl_override(state, record.id, 3600)

    assert instance.ttl_s == 600, "an hour asked, the policy's ten minutes applied"
    assert recorder.debugs == [
        (
            "request ttl capped at its tier's idle timeout",
            {"model_id": record.id, "tier": 3, "asked": 3600, "applied": 600},
        )
    ]


def test_a_request_may_shorten_the_timer_below_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(openai_routes, "log", recorder)
    record = make_record()
    state, instance = state_for(record, priority=3)

    openai_routes._apply_ttl_override(state, record.id, 300)

    assert instance.ttl_s == 300
    assert recorder.debugs == [], "nothing was capped, nothing is logged"


def test_asking_exactly_the_cap_is_not_a_cap() -> None:
    record = make_record()
    state, instance = state_for(record, priority=3)
    openai_routes._apply_ttl_override(state, record.id, 600)
    assert instance.ttl_s == 600


@pytest.mark.parametrize(("tier", "cap"), [(1, 900), (2, 900), (3, 600)])
def test_every_shipped_tier_is_its_own_ceiling(tier: int, cap: int) -> None:
    record = make_record()
    state, instance = state_for(record, priority=tier)
    openai_routes._apply_ttl_override(state, record.id, 3600)
    assert instance.ttl_s == cap
    assert state.manager.request_ttl_cap(record.id) == cap


def test_a_model_with_its_own_ttl_keeps_the_request() -> None:
    """A per-model ``ttl_s`` is the owner's opinion; the tier does not cap it."""
    record = make_record(ttl_s=7200)
    state, instance = state_for(record, priority=3)
    assert instance.ttl_s == 7200

    openai_routes._apply_ttl_override(state, record.id, 3600)

    assert instance.ttl_s == 3600
    assert state.manager.request_ttl_cap(record.id) is None


def test_a_pinned_model_is_untouched_with_or_without_a_manager() -> None:
    record = make_record(pinned=True)
    state, instance = state_for(record, priority=3)
    assert instance.ttl_s == 0

    openai_routes._apply_ttl_override(state, record.id, 60)
    openai_routes._apply_ttl_override(SimpleNamespace(supervisor=state.supervisor), record.id, 60)

    assert instance.ttl_s == 0
    assert state.manager.request_ttl_cap(record.id) is None


def test_an_empty_tier_map_is_the_old_behaviour() -> None:
    record = make_record()
    state, instance = state_for(record, priority=3, ttl_by_priority={}, default_ttl_s=1800)
    assert instance.ttl_s == 1800

    openai_routes._apply_ttl_override(state, record.id, 3600)

    assert instance.ttl_s == 3600, "no tier price, no ceiling: the request's number stands"
    assert state.manager.request_ttl_cap(record.id) is None


def test_a_tier_the_map_does_not_price_is_uncapped() -> None:
    record = make_record()
    state, instance = state_for(record, priority=3, ttl_by_priority={1: 900}, default_ttl_s=600)
    assert instance.ttl_s == 600, "the default applied at load"

    openai_routes._apply_ttl_override(state, record.id, 3600)

    assert instance.ttl_s == 3600, "the default is a fallback, not a ceiling"


def test_a_tier_priced_zero_is_protected_on_the_write_side_not_capped() -> None:
    """``0`` means never idle-unload: the instance is stamped 0 and the D41 guard holds."""
    record = make_record()
    state, instance = state_for(record, priority=3, ttl_by_priority={3: 0})
    assert instance.ttl_s == 0

    openai_routes._apply_ttl_override(state, record.id, 3600)

    assert instance.ttl_s == 0
    assert state.manager.request_ttl_cap(record.id) is None, "never-idle-unload is not a ceiling"


def test_request_ttl_cap_is_none_for_an_unknown_or_unloaded_model() -> None:
    record = make_record()
    state, _instance = state_for(record, priority=3)
    assert state.manager.request_ttl_cap("nobody/knows") is None
    state.supervisor.instances.clear()
    assert state.manager.request_ttl_cap(record.id) is None


def test_a_state_without_a_manager_is_the_uncapped_path() -> None:
    """The stream tests' FakeState has no manager: the pre-D61 write, unchanged."""
    record = make_record()
    state, instance = state_for(record, priority=3)
    bare = SimpleNamespace(supervisor=state.supervisor)

    openai_routes._apply_ttl_override(bare, record.id, 3600)

    assert instance.ttl_s == 3600
