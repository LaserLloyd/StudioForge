"""``GET /api/models/{id}/plan-recommended``: the dry run of load-recommended (D64, CR-1).

Raised by the ClawChat V14 audit (2026-09-12): ``/plan`` is a dry run of
``/load``, a different algorithm, so a client that wanted to show a human "this
fits at 128k and will evict X" before a 40-90 s commitment had to guess or just
load. The same evening CR-9 showed what two paths answering one question two
ways costs: the mode walk and the real load disagreed about a lease and chat
was down for seven minutes.

What these pin:

* the dry run and the real call reach the same decision -- placement, context,
  KV types, slots, evictions -- including the leased-card case of CR-9, and the
  same refusal (code, message, per-mode reasons, the numbers);
* the dry run changes nothing: no load, no eviction, no hold, no tier memo, no
  settings write, no lease;
* bad input is the same 400 on both;
* the route exists, answers 200 either way, and is in the OpenAPI document.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.core.leases import LeaseBook
from studioforge.core.priority import PRIORITY_CHAT
from studioforge.errors import BadRequestError, InsufficientVramError, ModelBusyError
from tests.unit.test_catalog_routes import MODEL_ID, app  # noqa: F401 - the fixture
from tests.unit.test_load_recommended import (
    MODEL,
    lopsided_rig,
    make_manager,
    resident_self,
    serving,
)
from tests.unit.test_load_recommended_leases import leased_manager, tight_3090s


def _agrees(dry: dict[str, Any], instance: Any) -> None:
    plan = instance.plan
    assert plan is not None
    assert dry["fits"] is True
    assert sorted(dry["devices"]) == sorted(plan.devices)
    assert dry["ctx_size"] == int(plan.ctx_per_slot or plan.ctx_size)
    assert dry["kv_cache_type"] == plan.kv_cache_type
    assert dry["kv_cache_type_v"] == plan.kv_cache_type_v
    assert dry["parallel"] == plan.parallel
    assert sorted(dry["evict_model_ids"]) == sorted(
        m for m in plan.evict_model_ids if m != instance.model_id
    )


# ---------------------------------------------------------------------------
# The same decision
# ---------------------------------------------------------------------------


async def test_the_dry_run_names_the_placement_the_real_call_then_takes() -> None:
    manager, supervisor = make_manager()
    dry = await manager.plan_recommended(MODEL, 65536)
    assert supervisor.starts == 0
    assert dry["dry_run"] is True
    assert dry["mode"] == "dual_5090"
    instance = await manager.load_recommended(MODEL, 65536)
    _agrees(dry, instance)


async def test_the_dry_run_agrees_when_the_walk_has_a_real_choice_and_an_idle_neighbour() -> None:
    """A mode that fits now is preferred to one that needs an eviction -- in both."""
    idle = serving("pub/other", requests=0, devices=[0, 1])
    manager, _supervisor = make_manager(probe=lopsided_rig(), loaded=[idle])
    dry = await manager.plan_recommended(MODEL, 16384)
    instance = await manager.load_recommended(MODEL, 16384)
    _agrees(dry, instance)
    assert dry["mode"] == "dual_3090"


async def test_the_dry_run_agrees_with_the_real_call_around_a_leased_card() -> None:
    """CR-9's incident, previewed: the dry run takes the 3090 pair and says why."""
    manager, supervisor, _book, lease = leased_manager()
    dry = await manager.plan_recommended(MODEL, 32768, priority=PRIORITY_CHAT, max_slots=1)
    assert dry["mode"] == "dual_3090"
    assert dry["lease_skipped_modes"] == ["dual_5090"]
    assert any(lease.id in note for note in dry["notes"])
    assert any("passed over dual_5090" in note for note in dry["notes"])
    assert supervisor.starts == 0

    instance = await manager.load_recommended(MODEL, 32768, priority=PRIORITY_CHAT, max_slots=1)
    _agrees(dry, instance)
    assert 1 not in instance.plan.devices


async def test_the_dry_run_returns_the_refusal_the_real_call_raises_around_a_lease() -> None:
    manager, supervisor, _book, lease = leased_manager(probe=tight_3090s(), n_ctx_train=262144)
    dry = await manager.plan_recommended(MODEL, 262144, kv_min="f16", max_slots=1)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144, kv_min="f16", max_slots=1)
    real = excinfo.value

    assert dry["fits"] is False
    assert dry["status_code"] == real.status_code == 507
    assert dry["code"] == real.code == "gpu_leased"
    assert dry["message"] == real.message
    assert dry["error"]["code"] == "gpu_leased"
    assert dry["error"]["studioforge"]["lease"]["id"] == lease.id == real.details["lease"]["id"]
    assert [m["reason"] for m in dry["modes"]] == [m["reason"] for m in real.details["modes"]]
    assert dry["retry_after_s"]
    assert "device override" not in dry["message"]
    assert supervisor.starts == 0


async def test_the_dry_run_refusal_carries_the_shortfall_the_largest_term_and_the_context() -> None:
    manager, _supervisor = make_manager(free_gib=6.0, n_ctx_train=1048576)
    dry = await manager.plan_recommended(MODEL, 1048576)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 1048576)
    real = excinfo.value
    assert dry["fits"] is False
    assert dry["code"] == real.code == "insufficient_vram"
    assert dry["shortfall_bytes"] == real.details["shortfall_bytes"]
    assert dry["shortfall_bytes"] and dry["shortfall_bytes"] > 0
    assert dry["largest_term"] == real.details["largest_term"]
    assert dry["largest_term"]["term"]
    assert dry["max_ctx_that_fits"] == real.details["largest_ctx_that_fits"]
    assert dry["max_ctx_that_fits"] and dry["max_ctx_that_fits"] < 1048576
    assert dry["retry_after_s"] is None


async def test_a_resident_already_at_that_context_is_previewed_as_already_loaded() -> None:
    manager, supervisor = make_manager(loaded=[resident_self(ctx=65536)])
    dry = await manager.plan_recommended(MODEL, 65536)
    assert dry["fits"] is True
    assert dry["already_loaded"] is True
    assert dry["evict_model_ids"] == []
    returned = await manager.load_recommended(MODEL, 65536)
    assert supervisor.starts == 0
    assert sorted(dry["devices"]) == sorted(returned.plan.devices)


async def test_a_serving_resident_is_previewed_as_the_503_the_real_call_raises() -> None:
    manager, _supervisor = make_manager(loaded=[resident_self(ctx=32768, requests=2)])
    dry = await manager.plan_recommended(MODEL, 65536)
    with pytest.raises(ModelBusyError) as excinfo:
        await manager.load_recommended(MODEL, 65536)
    assert dry["fits"] is False
    assert dry["status_code"] == excinfo.value.status_code == 503
    assert dry["code"] == excinfo.value.code
    assert dry["retry_after_s"] == excinfo.value.details["retry_after_s"]


async def test_bad_input_is_the_same_400_on_the_dry_run_and_the_real_call() -> None:
    manager, _supervisor = make_manager(n_ctx_train=32768)
    for kwargs in (
        {"ctx_size": 0},
        {"ctx_size": 131072},
        {"ctx_size": 16384, "max_slots": 0},
        {"ctx_size": 16384, "allowed_devices": []},
        {"ctx_size": 16384, "prefer_modes": ["dual_4090"]},
    ):
        ctx = kwargs.pop("ctx_size")
        with pytest.raises(BadRequestError) as dry_error:
            await manager.plan_recommended(MODEL, ctx, **kwargs)
        with pytest.raises(BadRequestError) as real_error:
            await manager.load_recommended(MODEL, ctx, **kwargs)
        assert dry_error.value.param == real_error.value.param
        assert dry_error.value.message == real_error.value.message


# ---------------------------------------------------------------------------
# No side effects
# ---------------------------------------------------------------------------


async def test_the_dry_run_loads_evicts_holds_remembers_and_writes_nothing() -> None:
    """A tier-1 dry run that would displace an idle neighbour leaves it exactly where it is."""
    idle = serving("pub/other", requests=0, devices=[0, 1, 2, 3])
    manager, supervisor = make_manager(free_gib=2.0, loaded=[idle])
    book = LeaseBook()
    manager.leases = book
    manager.planner.leases = book
    before = manager.busy_snapshot()

    dry = await manager.plan_recommended(MODEL, 16384, priority=PRIORITY_CHAT)

    assert dry["fits"] is True
    assert "pub/other" in dry["evict_model_ids"]
    assert supervisor.starts == 0
    assert supervisor.stopped == []
    assert supervisor.get("pub/other") is idle
    assert manager._priority_holds == {}
    assert MODEL not in manager._model_priority
    assert manager.registry.saved == []
    assert len(book) == 0
    assert manager.busy_snapshot() == before


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_route_is_a_get_that_answers_the_dry_run_and_loads_nothing(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        response = http.get(
            f"/api/models/{MODEL_ID}/plan-recommended",
            params={"ctx_size": 16384, "max_slots": 1, "priority": 2, "allowed_devices": [0]},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dry_run"] is True
    assert body["model_id"] == MODEL_ID
    assert body["priority"] == 2
    assert body["allowed_devices"] == [0]
    assert "fits" in body and "modes" in body and "notes" in body
    assert app.state.supervisor.get(MODEL_ID) is None, "nothing was loaded"


def test_the_route_forwards_every_load_recommended_input(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    seen: dict[str, Any] = {}

    async def capture(model_id: str, ctx_size: int, **kwargs: Any) -> dict[str, Any]:
        seen.update(model_id=model_id, ctx_size=ctx_size, **kwargs)
        return {"dry_run": True, "fits": True}

    app.state.manager.plan_recommended = capture
    with TestClient(app) as http:
        response = http.get(
            f"/api/models/{MODEL_ID}/plan-recommended",
            params={
                "ctx_size": 131072,
                "kv_min": "q8_0",
                "max_slots": 3,
                "prefer_mode": "dual_3090",
                "allowed_devices": [2, 3],
                "priority": 1,
            },
        )
    assert response.status_code == 200, response.text
    assert seen == {
        "model_id": MODEL_ID,
        "ctx_size": 131072,
        "prefer_modes": ["dual_3090"],
        "kv_min": "q8_0",
        "max_slots": 3,
        "allowed_devices": [2, 3],
        "priority": 1,
    }


def test_the_route_refuses_bad_input_in_the_openai_shape_like_load_recommended(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        unknown = http.get(
            f"/api/models/{MODEL_ID}/plan-recommended",
            params={"ctx_size": 16384, "allowed_devices": [7]},
        )
        missing = http.get(f"/api/models/{MODEL_ID}/plan-recommended")
    assert unknown.status_code == 400, unknown.text
    assert unknown.json()["error"]["param"] == "allowed_devices"
    assert missing.status_code == 400, "ctx_size is required, as on load-recommended"


def test_the_route_is_in_the_openapi_document(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        spec = http.get("/openapi.json").json()
    operation = spec["paths"]["/api/models/{model_id}/plan-recommended"]["get"]
    names = {p["name"] for p in operation["parameters"]}
    wanted = {"ctx_size", "kv_min", "max_slots", "prefer_mode", "allowed_devices", "priority"}
    assert wanted <= names
    assert "dry run" in operation["description"]
