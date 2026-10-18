"""``ttl`` and ``priority`` mean the same thing on every inference front door.

``/v1/chat/completions`` consumes LM Studio's per-request ``ttl`` (and the
``priority`` tier) and applies it to the instance's idle timer, capped at the
tier's ceiling (D61) and never able to pin or unpin (D41). ``/v1/messages``,
``/v1/responses`` and ``POST /api/v1/models/load`` must do exactly the same --
they share ``prepare_chat`` / ``_pop_ttl`` / ``_apply_ttl_override`` -- and this
file holds them to it: for each ttl value the instance's resulting timer is
identical on all four, and equals what the chat route alone produces.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from studioforge.types import InstanceInfo
from tests.unit.compat.conftest import Upstream, instance_of
from tests.unit.test_catalog_routes import MODEL_ID

ROUTES = {
    "chat": ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    "messages": (
        "/v1/messages",
        {"max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
    ),
    "responses": ("/v1/responses", {"input": "hi"}),
    "load": ("/api/v1/models/load", {}),
}


def _arm_load(app: Any) -> None:
    async def load(model_id: str, **_kwargs: Any) -> InstanceInfo:
        return instance_of(app, model_id)

    app.state.manager.load = load


def _ttl_after(
    app: Any, http: Any, route: str, ttl: Any, *, start: int | None = None
) -> int | None:
    path, base = ROUTES[route]
    instance = instance_of(app)
    instance.ttl_s = start
    response = http.post(path, json={"model": MODEL_ID, **base, "ttl": ttl})
    assert response.status_code == 200, response.text
    return instance.ttl_s


@pytest.mark.parametrize("ttl", [120, 600, 3600, 10**7, 0, -5, 1.9, "soon", True, None])
def test_the_resulting_idle_timer_is_identical_on_every_route(
    app: Any, http: Any, upstream: Upstream, ttl: Any
) -> None:
    _arm_load(app)
    results = {route: _ttl_after(app, http, route, ttl) for route in ROUTES}
    assert len(set(results.values())) == 1, results


def test_a_shorter_ttl_is_honoured_on_every_route(app: Any, http: Any, upstream: Upstream) -> None:
    _arm_load(app)
    for route in ROUTES:
        assert _ttl_after(app, http, route, 120) == 120, route


def test_a_longer_ttl_is_capped_at_the_tier_ceiling_on_every_route(
    app: Any, http: Any, upstream: Upstream
) -> None:
    _arm_load(app)
    cap = app.state.manager.request_ttl_cap(MODEL_ID)
    assert cap is not None, "the shipped tier map prices a background load"
    for route in ROUTES:
        assert _ttl_after(app, http, route, 10**7) == cap, route


def test_zero_never_pins_and_a_pinned_model_is_never_unpinned(
    app: Any, http: Any, upstream: Upstream
) -> None:
    _arm_load(app)
    for route in ROUTES:
        assert _ttl_after(app, http, route, 0, start=None) is None, f"{route}: 0 must not pin"
        assert _ttl_after(app, http, route, 60, start=0) == 0, f"{route}: a pin must survive"


def test_ttl_and_priority_never_reach_the_engine(app: Any, http: Any, upstream: Upstream) -> None:
    _arm_load(app)
    for route in ("chat", "messages", "responses"):
        path, base = ROUTES[route]
        response = http.post(path, json={"model": MODEL_ID, **base, "ttl": 60, "priority": 2})
        assert response.status_code == 200, response.text
        assert "ttl" not in upstream.last and "priority" not in upstream.last, route


@pytest.mark.parametrize("route", ["messages", "responses", "load"])
def test_a_bad_priority_is_a_400_naming_it_on_every_route(
    app: Any, http: Any, upstream: Upstream, route: str
) -> None:
    _arm_load(app)
    path, base = ROUTES[route]
    response = http.post(path, json={"model": MODEL_ID, **base, "priority": 7})
    assert response.status_code == 400
    # /v1/messages answers in Anthropic's envelope, the others in OpenAI's.
    assert "priority" in json.dumps(response.json())
