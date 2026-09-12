"""``GET /api/capabilities`` says which decisions and features this build has (D64, CR-2).

Raised by the ClawChat V14 audit (2026-09-12): the live server answered
``/api/version`` with ``1.26-09-04-3`` while it was serving D61-D63 from a
commit past that tag, so a client wanting D61's request-TTL cap or D62's stable
channel could only match route text in ``openapi.json``. The version is not
bumped between releases on purpose (RELEASING.md); the capability list is what a
client gates on instead.

What these pin: the decision list is exactly the headings of ``DECISIONS.md``
(so appending a decision without bumping ``LATEST_DECISION`` fails here); every
named feature points at a decision the build includes; the route carries the
block and every pre-existing top-level key is unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from studioforge.core import capabilities
from tests.unit.test_catalog_routes import app  # noqa: F401 - the fixture

DECISIONS = Path(__file__).resolve().parents[2] / "DECISIONS.md"


def _headings() -> list[str]:
    text = DECISIONS.read_text(encoding="utf-8")
    return [f"D{n}" for n in re.findall(r"(?m)^## D(\d+)\b", text)]


def test_the_decision_list_is_exactly_the_decisions_file() -> None:
    """Appending D65 without bumping LATEST_DECISION is the failure this exists to catch."""
    headings = _headings()
    assert headings, "DECISIONS.md headings could not be read"
    assert sorted(set(headings), key=lambda d: int(d[1:])) == list(
        capabilities.IMPLEMENTED_DECISIONS
    ), (
        f"DECISIONS.md ends at {headings[-1]} but capabilities.LATEST_DECISION is "
        f"D{capabilities.LATEST_DECISION}; bump it in the same commit (CONTRIBUTING.md)"
    )


def test_every_named_feature_points_at_a_decision_this_build_includes() -> None:
    for feature, decision in capabilities.SERVER_FEATURES.items():
        assert re.fullmatch(r"[a-z][a-z0-9_]*", feature), feature
        assert decision in capabilities.IMPLEMENTED_DECISIONS, (feature, decision)


def test_the_features_this_round_added_are_named() -> None:
    features = capabilities.implemented_report()["features"]
    for name in ("plan_recommended", "sse_error_frame", "load_recommended_lease_aware"):
        assert name in features


def test_the_route_carries_the_block_and_keeps_every_existing_key(app: Any) -> None:  # noqa: F811 - the imported fixture, by design
    with TestClient(app) as http:
        body = http.get("/api/capabilities").json()
    assert {"engine", "hardware", "features", "library", "update"} <= set(body)
    implemented = body["implemented"]
    assert implemented["latest_decision"] == f"D{capabilities.LATEST_DECISION}"
    assert "D61" in implemented["decisions"] and "D64" in implemented["decisions"]
    assert "plan_recommended" in implemented["features"]
    assert implemented["feature_decisions"]["plan_recommended"] == "D64"
