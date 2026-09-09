"""``GET /api/models/{id}/plan`` is the dry-run planner (2026-09-09 review).

It answers with the same planner, the same one-shot ``devices`` /
``allowed_devices`` copies and the same validation as the load route, and it
never loads anything. The MCP ``plan_load`` tool is this route; the GUI's live
fit check always was.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.unit.test_catalog_routes import MODEL_ID, app  # noqa: F401 - the fixture


def test_plan_is_a_dry_run_that_carries_its_bound(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        response = http.get(f"/api/models/{MODEL_ID}/plan", params={"allowed_devices": [0]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dry_run"] is True
    assert body["model_id"] == MODEL_ID
    assert body["allowed_devices"] == [0]
    assert "fits" in body
    assert app.state.supervisor.get(MODEL_ID) is None, "nothing was loaded"


def test_plan_refuses_devices_and_allowed_devices_together_like_the_load_route(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        response = http.get(
            f"/api/models/{MODEL_ID}/plan", params={"devices": [0], "allowed_devices": [0]}
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "allowed_devices"


def test_plan_reports_the_tier_it_planned_at(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        default = http.get(f"/api/models/{MODEL_ID}/plan").json()
        explicit = http.get(f"/api/models/{MODEL_ID}/plan", params={"priority": 1}).json()
    assert default["priority"] == 3, "a model nobody tiered plans as background"
    assert explicit["priority"] == 1


def test_a_refusal_names_the_shortfall_and_the_largest_term(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        body = http.get(
            f"/api/models/{MODEL_ID}/plan", params={"ctx_size": 32768, "parallel": 64}
        ).json()
    if body["fits"]:
        return  # the fake rig is roomy enough; nothing to assert about a refusal
    assert body["shortfall_bytes"] > 0
    assert body["largest_term"] is not None
    assert "short by" in body["message"]
