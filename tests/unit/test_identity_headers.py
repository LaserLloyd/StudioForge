"""Who answered: Server / X-SF-Model / X-SF-Engine / X-Request-Id, and the alias markers (D90).

Every ``/v1/*`` response -- streamed ones included -- says which server, which
model and which engine build answered, and carries a request id (the client's own
when it sent a sane one). ``server.identity_headers: false`` removes all four.
Deprecated aliases carry ``Deprecation`` and ``Link`` (no ``Sunset`` yet).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
import structlog
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from studioforge import __version__
from studioforge.api.identity import (
    DEPRECATION_EPOCH,
    IdentityMiddleware,
    clean_request_id,
    is_deprecated_alias,
    register_deprecated_alias,
)
from studioforge.config import Config
from tests.unit.native_stub import StubChild, build_app, instance, record, sse

CHAT = "vendor/chat-Q4_K_M"
EMBED = "vendor/embed-Q8_0"
MESSAGES = [{"role": "user", "content": "hi"}]


@pytest.fixture()
def child(monkeypatch: pytest.MonkeyPatch) -> StubChild:
    return StubChild().install(monkeypatch)


@pytest.fixture()
def app(tmp_path: Path, child: StubChild) -> Any:
    return build_app(
        tmp_path,
        [record(CHAT), record(EMBED, kind="embedding")],
        [instance(CHAT, engine="b12110"), instance(EMBED, engine="b12110")],
        child,
    )


def assert_identity(response: httpx.Response, model: str | None) -> None:
    assert response.headers["server"] == f"StudioForge/{__version__}"
    assert response.headers["x-request-id"]
    if model is not None:
        assert response.headers["x-sf-model"] == model
        assert response.headers["x-sf-engine"] == "b12110"


# ---------------------------------------------------------------------------
# The four headers on /v1/*
# ---------------------------------------------------------------------------


def test_chat_completions_carries_all_four_headers(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": CHAT, "messages": MESSAGES})
    assert response.status_code == 200
    assert_identity(response, CHAT)


def test_a_streamed_chat_completion_carries_them_too(app: Any) -> None:
    with (
        TestClient(app) as http,
        http.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": CHAT, "messages": MESSAGES, "stream": True},
        ) as response,
    ):
        body = b"".join(response.iter_bytes())
        headers = response.headers
    assert b"[DONE]" in body
    assert headers["server"] == f"StudioForge/{__version__}"
    assert headers["x-sf-model"] == CHAT
    assert headers["x-sf-engine"] == "b12110"
    assert headers["x-request-id"]


def test_the_other_v1_routes_carry_them(app: Any) -> None:
    with TestClient(app) as http:
        completions = http.post("/v1/completions", json={"model": CHAT, "prompt": "x"})
        embeddings = http.post("/v1/embeddings", json={"model": EMBED, "input": "x"})
        tokenize = http.post("/v1/tokenize", json={"model": CHAT, "content": "x"})
    assert_identity(completions, CHAT)
    assert_identity(embeddings, EMBED)
    assert_identity(tokenize, CHAT)


def test_the_loaded_alias_reports_the_model_that_actually_answered(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": "loaded", "messages": MESSAGES})
    assert response.headers["x-sf-model"] == CHAT


def test_a_preset_only_virtual_model_reports_its_serving_base(
    tmp_path: Path, child: StubChild
) -> None:
    from studioforge.types import VirtualPreset

    base = record(CHAT)
    persona = record("vendor/persona").model_copy(
        update={
            "is_virtual": True,
            "base_model_id": CHAT,
            "preset": VirtualPreset(system_prompt="be brief"),
            "path": base.path,
        }
    )
    app = build_app(tmp_path, [base, persona], [instance(CHAT)], child)
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions", json={"model": "vendor/persona", "messages": MESSAGES}
        )
    assert response.status_code == 200, response.text
    assert response.headers["x-sf-model"] == CHAT


def test_the_engine_header_is_the_childs_own_build(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(CHAT)], [instance(CHAT, engine="b99999")], child)
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": CHAT, "messages": MESSAGES})
    assert response.headers["x-sf-engine"] == "b99999"


def test_a_clients_request_id_is_echoed(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": CHAT, "messages": MESSAGES},
            headers={"X-Request-Id": "bench-run-42/req_7"},
        )
        streamed = http.post(
            "/v1/chat/completions",
            json={"model": CHAT, "messages": MESSAGES, "stream": True},
            headers={"X-Request-Id": "bench-run-42/req_8"},
        )
    assert response.headers["x-request-id"] == "bench-run-42/req_7"
    assert streamed.headers["x-request-id"] == "bench-run-42/req_8"


def test_a_fresh_id_is_made_when_none_or_a_bad_one_is_sent(app: Any) -> None:
    with TestClient(app) as http:
        none = http.get("/v1/models")
        bad = http.get("/v1/models", headers={"X-Request-Id": 'has space and "quotes"'})
        long = http.get("/v1/models", headers={"X-Request-Id": "a" * 500})
        again = http.get("/v1/models")
    ids = [r.headers["x-request-id"] for r in (none, bad, long, again)]
    assert all(len(i) == 32 for i in ids)
    assert len(set(ids)) == 4


def test_error_responses_carry_server_and_request_id_but_no_model(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions", json={"model": "nope/none", "messages": MESSAGES}
        )
    assert response.status_code == 404
    assert response.headers["server"] == f"StudioForge/{__version__}"
    assert response.headers["x-request-id"]
    assert "x-sf-model" not in response.headers
    assert "x-sf-engine" not in response.headers


def test_authentication_refusals_carry_them_too(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(CHAT)], [], child, server={"api_key": "admin-key-0123456789"})
    with TestClient(app) as http:
        response = http.get("/v1/models")
    assert response.status_code == 401
    assert response.headers["server"] == f"StudioForge/{__version__}"
    assert response.headers["x-request-id"]


def test_the_natives_and_probes_carry_them(app: Any) -> None:
    with TestClient(app) as http:
        tokenize = http.post("/tokenize", json={"content": "x", "model": CHAT})
        health = http.get("/health")
    assert_identity(tokenize, CHAT)
    assert health.headers["server"] == f"StudioForge/{__version__}"
    assert "x-sf-model" not in health.headers


def test_there_is_exactly_one_server_header(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/v1/models")
    assert len(response.headers.get_list("server")) == 1


# ---------------------------------------------------------------------------
# server.identity_headers: false
# ---------------------------------------------------------------------------


def test_the_flag_defaults_on_and_turns_all_four_off(tmp_path: Path, child: StubChild) -> None:
    assert Config(data_dir=tmp_path).server.identity_headers is True
    app = build_app(
        tmp_path, [record(CHAT)], [instance(CHAT)], child, server={"identity_headers": False}
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": CHAT, "messages": MESSAGES},
            headers={"X-Request-Id": "keep-me"},
        )
    assert response.status_code == 200
    for name in ("x-sf-model", "x-sf-engine", "x-request-id"):
        assert name not in response.headers
    assert not response.headers.get("server", "").startswith("StudioForge")


def test_the_flag_is_live(app: Any) -> None:
    with TestClient(app) as http:
        assert "x-request-id" in http.get("/v1/models").headers
        app.state.config.server.identity_headers = False
        assert "x-request-id" not in http.get("/v1/models").headers


# ---------------------------------------------------------------------------
# The request id reaches the log context
# ---------------------------------------------------------------------------


def test_the_request_id_is_bound_into_the_log_context_for_the_request() -> None:
    seen: dict[str, Any] = {}
    api = FastAPI()
    api.add_middleware(IdentityMiddleware, config=Config(data_dir=Path()))

    @api.get("/probe")
    async def probe(request: Request) -> dict[str, str]:
        seen["log"] = structlog.contextvars.get_contextvars().get("request_id")
        seen["state"] = request.scope["state"]["request_id"]
        return {"ok": "yes"}

    with TestClient(api) as http:
        response = http.get("/probe", headers={"X-Request-Id": "trace-1"})
    assert response.headers["x-request-id"] == "trace-1"
    assert seen == {"log": "trace-1", "state": "trace-1"}
    # Nothing leaks past the request.
    assert "request_id" not in structlog.contextvars.get_contextvars()


def test_clean_request_id_accepts_tokens_and_refuses_the_rest() -> None:
    assert clean_request_id("abc-123_DEF.4:5/6=7+8") == "abc-123_DEF.4:5/6=7+8"
    assert clean_request_id("  padded  ") == "padded"
    for bad in (None, "", "   ", "two words", "a\r\nb", "x" * 129, 'q"uote', "ünï"):
        assert clean_request_id(bad) is None


# ---------------------------------------------------------------------------
# Deprecated aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/models/vendor/chat-Q4_K_M/load"),
        ("POST", "/api/models/vendor/chat-Q4_K_M/load-recommended"),
        ("POST", "/api/models/vendor/chat-Q4_K_M/unload"),
        ("GET", "/api/models/vendor/chat-Q4_K_M/plan"),
        ("GET", "/api/models/vendor/chat-Q4_K_M/plan-recommended"),
        ("POST", "/api/v1/models/load"),
        ("POST", "/api/v1/models/unload"),
    ],
)
def test_the_old_load_and_plan_routes_are_recognised_as_aliases(method: str, path: str) -> None:
    assert is_deprecated_alias(method, path)


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/models/load"),
        ("POST", "/models/unload"),
        ("GET", "/models"),
        ("POST", "/api/models/unload-all"),
        ("PATCH", "/api/models/vendor/chat-Q4_K_M/settings"),
        ("GET", "/api/models/vendor/chat-Q4_K_M/options"),
        ("POST", "/v1/chat/completions"),
    ],
)
def test_the_canonical_routes_and_the_rest_are_not(method: str, path: str) -> None:
    assert not is_deprecated_alias(method, path)


def test_an_alias_response_carries_deprecation_and_a_link_but_no_sunset(app: Any) -> None:
    async def unload(name: str, **_kwargs: Any) -> bool:
        return True

    app.state.manager.unload = unload
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        alias = http.post(f"/api/models/{CHAT}/unload")
        canonical = http.post("/models/unload", json={"model": CHAT})
        other = http.get("/api/models")
    assert alias.status_code == 200
    assert alias.headers["deprecation"] == f"@{DEPRECATION_EPOCH}"
    assert 'rel="deprecation"' in alias.headers["link"]
    assert "MIGRATION.md" in alias.headers["link"]
    assert "sunset" not in alias.headers
    assert "deprecation" not in canonical.headers
    assert "deprecation" not in other.headers


def test_another_package_can_register_its_alias() -> None:
    assert not is_deprecated_alias("POST", "/api/leases/abc/touch-wp5a-test")
    register_deprecated_alias("POST", r"/api/leases/[^/]+/touch-wp5a-test")
    assert is_deprecated_alias("post", "/api/leases/abc/touch-wp5a-test")


# ---------------------------------------------------------------------------
# Tool-call parse failures get a code, the engine's words untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Failed to parse tool call arguments as JSON",
        "invalid tool call",
        "unable to generate parser for this template",
    ],
)
def test_a_tool_call_parse_failure_gets_its_own_code(
    app: Any, child: StubChild, phrase: str
) -> None:
    child.routes[("POST", "/v1/chat/completions")] = lambda _r: httpx.Response(
        500, json={"error": {"code": 500, "message": phrase, "type": "server_error"}}
    )
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": CHAT, "messages": MESSAGES})
    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "tool_call_parse_failed"
    assert phrase in error["message"]
    assert error["fix"]


def test_a_streamed_tool_call_failure_carries_the_code_in_the_error_frame(
    app: Any, child: StubChild
) -> None:
    child.routes[("POST", "/v1/chat/completions")] = lambda _r: sse(
        b'{"error":{"message":"Failed to parse tool call"}}', status=500
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions", json={"model": CHAT, "messages": MESSAGES, "stream": True}
        )
    assert b"tool_call_parse_failed" in response.content
    assert b"Failed to parse tool call" in response.content
    assert response.content.rstrip().endswith(b"data: [DONE]")


def test_other_upstream_errors_keep_the_generic_code(app: Any, child: StubChild) -> None:
    child.routes[("POST", "/v1/chat/completions")] = lambda _r: httpx.Response(
        500, json={"error": {"message": "the engine fell over"}}
    )
    with TestClient(app) as http:
        response = http.post("/v1/chat/completions", json={"model": CHAT, "messages": MESSAGES})
    assert response.json()["error"]["code"] == "upstream_error"
