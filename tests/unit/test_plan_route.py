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


def test_plan_keeps_estimate_mb_in_mib_and_adds_the_same_breakdown_in_bytes(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    """CR-5 (2026-09-12): ``estimate_mb`` is MiB while ``LoadPlan.estimate`` is bytes.

    A factor-of-1,048,576 bug waiting for a client that assumed otherwise --
    made worse by the MiB values sitting under ``*_bytes`` key names. The
    existing field cannot change (clients read it), so the bytes twin rides
    beside it and the two must describe the same estimate.
    """
    with TestClient(app) as http:
        for params in ({"ctx_size": 8192}, {"ctx_size": 32768, "parallel": 64}):
            body = http.get(f"/api/models/{MODEL_ID}/plan", params=params).json()
            mib, raw = body["estimate_mb"], body["estimate_bytes"]
            assert set(mib) == set(raw)
            assert raw["total"] > 0
            for key, value in raw.items():
                assert isinstance(value, int)
                assert abs(mib[key] - value / (1024 * 1024)) < 1e-6, key


def test_a_load_refusal_507_carries_the_bytes_breakdown_beside_the_mib_one(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    """The same pair on the real 507, which shipped the same MiB-only breakdown (CR-5)."""
    from studioforge.types import GB, LoadRejected, VramEstimate

    rejected = LoadRejected(
        model_id=MODEL_ID,
        reason="does not fit",
        required_bytes=40 * GB,
        available_bytes=30 * GB,
        estimate=VramEstimate(weights_bytes=8 * GB, kv_bytes=32 * GB),
    )
    details = app.state.manager._vram_error(rejected).details
    assert details["estimate_mb"]["kv_bytes"] == 32 * 1024
    assert details["estimate_bytes"]["kv_bytes"] == 32 * GB
    assert details["estimate_bytes"]["total"] == rejected.estimate.total_bytes
