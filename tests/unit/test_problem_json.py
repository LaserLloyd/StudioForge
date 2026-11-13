"""RFC 9457 problem+json on management routes, by ``Accept`` negotiation only (D90, F-9).

The default body does not change: management routes keep the OpenAI-shaped
``{"error": {...}}`` (plus a ``Deprecation`` marker). A caller that sends
``Accept: application/problem+json`` gets a problem document that ALSO embeds the
legacy ``error{}`` object, so an early adopter that still reads ``error.code``
keeps working. Inference routes and the llama-server natives never negotiate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.problem import (
    PROBLEM_CONTENT_TYPE,
    accepts_problem,
    is_management_path,
    problem_from_envelope,
)
from studioforge.errors import ModelBusyError
from tests.unit.native_stub import StubChild, build_app, instance, record
from tests.unit.test_catalog_routes import MODEL_ID as CATALOG_MODEL
from tests.unit.test_catalog_routes import (
    app as catalog_app,  # noqa: F401 - the imported fixture, by design
)

CHAT = "vendor/chat-Q4_K_M"
PROBLEM = {"Accept": PROBLEM_CONTENT_TYPE}


@pytest.fixture()
def child(monkeypatch: pytest.MonkeyPatch) -> StubChild:
    return StubChild().install(monkeypatch)


@pytest.fixture()
def app(tmp_path: Path, child: StubChild) -> Any:
    return build_app(tmp_path, [record(CHAT)], [instance(CHAT)], child)


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        ("application/problem+json", True),
        ("Application/Problem+JSON", True),
        ("application/json, application/problem+json;q=0.9", True),
        ("application/problem+json; charset=utf-8", True),
        ("application/problem+json;q=0", False),
        ("application/json", False),
        ("*/*", False),
        ("application/*+json", False),
        ("", False),
        (None, False),
    ],
)
def test_accepts_problem(accept: str | None, expected: bool) -> None:
    assert accepts_problem(accept) is expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/api/models", True),
        ("/api/leases/abc", True),
        ("/models", True),
        ("/models/load", True),
        ("/models/unload", True),
        ("/api/v0/models", False),
        ("/api/v1/models/load", False),
        ("/v1/chat/completions", False),
        ("/v1/messages", False),
        ("/tokenize", False),
        ("/props", False),
        ("/completion", False),
        ("/health", False),
        ("/v2/health/ready", False),
        ("/mcp", False),
    ],
)
def test_which_paths_are_management_routes(path: str, expected: bool) -> None:
    assert is_management_path(path) is expected


def test_a_management_error_negotiates_a_problem_document(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/models/nope/none/plan", headers=PROBLEM)
    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    body = response.json()
    assert body["type"] == "urn:studioforge:error:model_not_found"
    assert body["title"] == "Model not found"
    assert body["status"] == 404
    assert body["code"] == "model_not_found"
    assert "does not exist" in body["detail"]
    assert body["fix"]
    assert body["param"] == "model"
    assert body["docs"] == "errors/model_not_found"
    # instance is the request id, and it is the id the header carries.
    assert body["instance"] == f"urn:studioforge:request:{response.headers['x-request-id']}"


def test_the_problem_document_embeds_the_legacy_envelope(app: Any) -> None:
    with TestClient(app) as http:
        plain = http.get("/api/models/nope/none/plan")
        problem = http.get("/api/models/nope/none/plan", headers=PROBLEM)
    assert problem.json()["error"] == plain.json()["error"]
    for key in ("message", "type", "code", "param"):
        assert key in problem.json()["error"]


def test_without_the_accept_header_the_body_is_unchanged_and_marked(app: Any) -> None:
    with TestClient(app) as http:
        plain = http.get("/api/models/nope/none/plan")
        json_only = http.get("/api/models/nope/none/plan", headers={"Accept": "application/json"})
    for response in (plain, json_only):
        assert response.status_code == 404
        assert response.headers["content-type"] == "application/json"
        assert set(response.json()) == {"error"}
        assert response.headers["deprecation"].startswith("@")
        assert "MIGRATION.md" in response.headers["link"]


def test_q_zero_does_not_opt_in(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get(
            "/api/models/nope/none/plan", headers={"Accept": "application/problem+json;q=0"}
        )
    assert response.headers["content-type"] == "application/json"


def test_busy_errors_carry_retry_after_as_an_extension_header_and_studioforge_block(
    app: Any,
) -> None:
    async def busy(name: str, **_kwargs: Any) -> Any:
        raise ModelBusyError(
            "a load is in flight",
            details={"retry_after_s": 7, "busy_models": [{"model_id": CHAT, "in_flight": 2}]},
        )

    app.state.manager.load = busy
    with TestClient(app) as http:
        response = http.post("/models/load", json={"model": CHAT}, headers=PROBLEM)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"
    body = response.json()
    assert body["retry_after_s"] == 7
    assert body["studioforge"]["busy_models"] == [{"model_id": CHAT, "in_flight": 2}]
    assert body["error"]["studioforge"]["retry_after_s"] == 7
    assert body["code"] == "model_busy"


def test_a_validation_error_negotiates_too(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/models/load", json={"model": CHAT, "context_length": "many"}, headers=PROBLEM
        )
    assert response.status_code == 400
    assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
    assert response.json()["code"] == "invalid_request"
    assert response.json()["type"] == "urn:studioforge:error:invalid_request"


def test_an_unknown_api_path_negotiates_with_the_framework_codes(app: Any) -> None:
    with TestClient(app) as http:
        missing = http.get("/api/definitely-not-here", headers=PROBLEM)
        wrong_verb = http.put("/api/models", headers=PROBLEM)
    assert missing.status_code == 404
    assert missing.headers["content-type"] == PROBLEM_CONTENT_TYPE
    assert missing.json()["code"] == "not_found"
    assert wrong_verb.status_code == 405
    assert wrong_verb.json()["code"] == "method_not_allowed"


def test_an_authentication_refusal_negotiates(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(CHAT)], [], child, server={"api_key": "admin-key-0123456789"})
    with TestClient(app) as http:
        problem = http.get("/api/models", headers=PROBLEM)
        plain = http.get("/api/models")
    assert problem.status_code == plain.status_code == 401
    assert problem.headers["content-type"] == PROBLEM_CONTENT_TYPE
    assert problem.json()["code"] == "invalid_api_key"
    assert plain.headers["content-type"] == "application/json"
    assert plain.json()["error"]["code"] == "invalid_api_key"


# ---------------------------------------------------------------------------
# Never on the inference routes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        (
            "POST",
            "/v1/chat/completions",
            {"model": "nope/none", "messages": [{"role": "user", "content": "x"}]},
        ),
        ("GET", "/v1/models/nope/none", None),
        ("POST", "/tokenize", {"content": "x", "model": "nope/none"}),
        ("GET", "/props?model=nope/none", None),
        ("POST", "/completion", {"prompt": "x", "model": "nope/none"}),
        ("GET", "/definitely-not-here", None),
        ("GET", "/api/v1/models/nope/none", None),
    ],
)
def test_inference_and_native_routes_keep_their_own_error_shape(
    app: Any, method: str, path: str, body: dict[str, Any] | None
) -> None:
    with TestClient(app) as http:
        response = http.request(method, path, json=body, headers=PROBLEM)
    assert response.status_code >= 400
    assert response.headers["content-type"] == "application/json"
    assert set(response.json()) == {"error"}
    assert "deprecation" not in response.headers


# ---------------------------------------------------------------------------
# The document builder
# ---------------------------------------------------------------------------


def test_problem_from_envelope_for_an_uncatalogued_error_is_about_blank() -> None:
    body = problem_from_envelope(
        {
            "error": {
                "message": "teapot",
                "type": "invalid_request_error",
                "code": None,
                "param": None,
            }
        },
        418,
        "rid",
    )
    assert body["type"] == "about:blank"
    assert body["title"] == "I'm a Teapot"
    assert body["status"] == 418
    assert body["detail"] == "teapot"
    assert body["instance"] == "urn:studioforge:request:rid"
    assert "fix" not in body and "retry_after_s" not in body and "studioforge" not in body


def test_problem_from_envelope_lifts_details_and_suggestions() -> None:
    envelope = {
        "error": {
            "message": "no room",
            "type": "server_error",
            "code": "insufficient_vram",
            "param": None,
            "studioforge": {
                "suggestions": ["lower ctx"],
                "retry_after_s": 3,
                "max_ctx_that_fits": 4096,
            },
        }
    }
    body = problem_from_envelope(envelope, 507, "rid")
    assert body["suggestions"] == ["lower ctx"]
    assert body["retry_after_s"] == 3
    assert body["studioforge"]["max_ctx_that_fits"] == 4096
    assert body["error"] == envelope["error"]


# ---------------------------------------------------------------------------
# context_exceeds_training (the code on the 400 that used to have none)
# ---------------------------------------------------------------------------


def test_a_context_past_the_trained_window_has_a_code_and_the_number(catalog_app: Any) -> None:  # noqa: F811
    with TestClient(catalog_app) as http:
        canonical = http.post(
            "/models/load", json={"model": CATALOG_MODEL, "context_length": 4_000_000}
        )
        alias = http.post(
            f"/api/models/{CATALOG_MODEL}/load-recommended", json={"ctx_size": 4_000_000}
        )
        problem = http.post(
            "/models/load",
            json={"model": CATALOG_MODEL, "context_length": 4_000_000},
            headers=PROBLEM,
        )
    for response in (canonical, alias):
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "context_exceeds_training"
        assert error["param"] == "ctx_size"
        assert error["studioforge"]["n_ctx_train"] == 32768
        assert error["studioforge"]["requested_ctx"] == 4_000_000
        # The prose the consumers match is untouched.
        assert "is trained to 32768 tokens" in error["message"]
        assert error["fix"]
    assert problem.json()["type"] == "urn:studioforge:error:context_exceeds_training"
    assert problem.json()["studioforge"]["n_ctx_train"] == 32768
