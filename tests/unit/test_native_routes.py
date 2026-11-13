"""llama-server's native per-model routes on StudioForge (D90).

``/tokenize``, ``/detokenize``, ``/apply-template``, ``/completion``, ``/infill``,
``/embedding``, ``/reranking`` (POST) and ``/props``, ``/slots`` (GET) forward to
the selected child and hand its answer back unchanged. The child here is an
in-process stub (``tests/unit/native_stub.py``): the real routes, selector rules,
JIT hook, in-flight accounting and forwarding run against a recorded
``httpx.MockTransport``. No process, no port, no engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from studioforge.api.auth import required_scope
from tests.unit.native_stub import (
    PROPS,
    StubChild,
    build_app,
    instance,
    record,
)

CHAT = "vendor/chat-Q4_K_M"
OTHER = "vendor/other-Q4_K_M"
EMBED = "vendor/embed-Q8_0"
RERANK = "vendor/rerank-Q8_0"


@pytest.fixture()
def child(monkeypatch: pytest.MonkeyPatch) -> StubChild:
    return StubChild().install(monkeypatch)


@pytest.fixture()
def app(tmp_path: Path, child: StubChild) -> Any:
    """One chat model resident (and idle): the stock single-model box."""
    return build_app(tmp_path, [record(CHAT), record(OTHER)], [instance(CHAT)], child)


def jit_loader(app: Any) -> list[str]:
    """Replace ``ensure_loaded`` with a recorder that makes the model resident."""
    called: list[str] = []

    async def fake_ensure_loaded(name: str, **_kwargs: Any) -> tuple[Any, Any]:
        called.append(name)
        if app.state.supervisor.get(name) is None:
            app.state.supervisor.instances.append(instance(name))
        return app.state.registry.get(name), app.state.supervisor.get(name)

    app.state.manager.ensure_loaded = fake_ensure_loaded
    return called


# ---------------------------------------------------------------------------
# Every POST native forwards, with the model selector in the body
# ---------------------------------------------------------------------------

POSTS = [
    ("/tokenize", {"content": "hello"}, {"tokens": [1, 2, 3]}, CHAT),
    ("/detokenize", {"tokens": [1, 2, 3]}, {"content": "abc"}, CHAT),
    (
        "/apply-template",
        {"messages": [{"role": "user", "content": "hi"}]},
        {"prompt": "<s>hi"},
        CHAT,
    ),
    ("/completion", {"prompt": "once upon"}, {"content": "hello", "model": CHAT}, CHAT),
    (
        "/infill",
        {"input_prefix": "def f(", "input_suffix": "):"},
        {"content": "hello", "model": CHAT},
        CHAT,
    ),
    ("/embedding", {"content": "hello"}, [{"index": 0, "embedding": [[0.1, 0.2]]}], EMBED),
    (
        "/reranking",
        {"query": "q", "documents": ["a"]},
        {"results": [{"index": 0, "relevance_score": 1.0}]},
        RERANK,
    ),
    (
        "/rerank",
        {"query": "q", "documents": ["a"]},
        {"results": [{"index": 0, "relevance_score": 1.0}]},
        RERANK,
    ),
]


@pytest.mark.parametrize(("path", "body", "expected", "model"), POSTS)
def test_every_post_native_forwards_with_the_body_selector(
    tmp_path: Path,
    child: StubChild,
    path: str,
    body: dict[str, Any],
    expected: Any,
    model: str,
) -> None:
    # Several residents of different kinds: only the body selector can decide.
    app = build_app(
        tmp_path,
        [
            record(CHAT),
            record(OTHER),
            record(EMBED, kind="embedding"),
            record(RERANK, kind="rerank"),
        ],
        [instance(CHAT), instance(OTHER), instance(EMBED), instance(RERANK)],
        child,
    )
    payload = {**body, "model": model}
    with TestClient(app) as http:
        response = http.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert child.last().url.path == path
    assert child.last().method == "POST"
    assert response.headers["x-sf-model"] == model
    # The caller's own fields reach the child untouched.
    sent = child.body()
    for key, value in body.items():
        assert sent[key] == value


def test_the_childs_json_is_returned_byte_for_byte(app: Any, child: StubChild) -> None:
    raw = b'{"tokens":[5,6,7],  "extra" : {"a":1}}'
    child.routes[("POST", "/tokenize")] = lambda _r: httpx.Response(
        200, content=raw, headers={"content-type": "application/json"}
    )
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x"})
    assert response.content == raw


def test_a_child_error_keeps_its_status_and_body(app: Any, child: StubChild) -> None:
    err = {
        "error": {
            "code": 400,
            "message": "Failed to parse grammar",
            "type": "invalid_request_error",
        }
    }
    child.routes[("POST", "/completion")] = lambda _r: httpx.Response(400, json=err)
    with TestClient(app) as http:
        response = http.post("/completion", json={"prompt": "x", "grammar": "broken"})
    assert response.status_code == 400
    assert response.json() == err


def test_an_unreachable_child_is_a_502_upstream_error(app: Any, child: StubChild) -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    child.routes[("POST", "/tokenize")] = boom
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x"})
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_error"


def test_a_non_json_body_is_a_400(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/tokenize", content=b"not json", headers={"content-type": "application/json"}
        )
    assert response.status_code == 400


def test_the_request_is_counted_in_flight_and_released(app: Any) -> None:
    with TestClient(app) as http:
        http.post("/completion", json={"prompt": "x"})
    supervisor = app.state.supervisor
    assert supervisor.started == [CHAT]
    assert supervisor.ended == [CHAT]
    assert supervisor.in_flight_seen == [1]
    assert supervisor.get(CHAT).active_requests == 0


def test_a_streamed_completion_is_relayed_without_a_done_sentinel(
    app: Any, child: StubChild
) -> None:
    with TestClient(app) as http:
        response = http.post("/completion", json={"prompt": "x", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == (
        b'data: {"content":"he","stop":false}\n\ndata: {"content":"llo","stop":true}\n\n'
    )
    assert response.headers["x-sf-model"] == CHAT
    assert app.state.supervisor.started == [CHAT]
    assert app.state.supervisor.ended == [CHAT]


def test_a_stream_refused_by_the_child_is_a_plain_error_status(app: Any, child: StubChild) -> None:
    err = {"error": {"code": 400, "message": "bad", "type": "invalid_request_error"}}
    child.routes[("POST", "/completion")] = lambda _r: httpx.Response(400, json=err)
    with TestClient(app) as http:
        response = http.post("/completion", json={"prompt": "x", "stream": True})
    assert response.status_code == 400
    assert response.json() == err
    # Nothing was counted for a stream that never started.
    assert app.state.supervisor.started == []


# ---------------------------------------------------------------------------
# GET natives: /props and /slots
# ---------------------------------------------------------------------------


def test_props_proxies_the_child_and_adds_only_a_studioforge_block(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/props", params={"model": CHAT})
    assert response.status_code == 200
    body = response.json()
    block = body.pop("studioforge")
    assert body == PROPS
    assert block["instance_id"] == CHAT
    assert block["devices"] == [0]
    assert block["parallel"] == 2
    assert block["ctx_per_slot"] == 8192
    assert block["kv_cache_type"] == "f16"
    assert block["engine"] == {"tag": "b12110"}
    assert response.headers["x-sf-model"] == CHAT
    assert response.headers["x-sf-engine"] == "b12110"


def test_slots_proxies_the_child_unchanged_and_drops_the_selector(
    app: Any, child: StubChild
) -> None:
    with TestClient(app) as http:
        response = http.get("/slots", params={"model": CHAT, "fail_on_no_slot": "1"})
    assert response.status_code == 200
    assert response.json() == [{"id": 0, "is_processing": False}]
    query = dict(child.last().url.params)
    assert query == {"fail_on_no_slot": "1"}


def test_props_post_is_refused_with_the_fix(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/props", json={"x": 1})
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert "PATCH /api/models/{id}/settings" in response.json()["error"]["message"]


def test_props_error_from_the_child_is_passed_through(app: Any, child: StubChild) -> None:
    child.routes[("GET", "/props")] = lambda _r: httpx.Response(501, json={"error": {"code": 501}})
    with TestClient(app) as http:
        response = http.get("/props", params={"model": CHAT})
    assert response.status_code == 501
    assert response.json() == {"error": {"code": 501}}


# ---------------------------------------------------------------------------
# The selector rules
# ---------------------------------------------------------------------------


def test_no_selector_with_one_resident_uses_that_model(app: Any, child: StubChild) -> None:
    with TestClient(app) as http:
        post = http.post("/tokenize", json={"content": "x"})
        get = http.get("/props")
    assert post.status_code == 200
    assert post.headers["x-sf-model"] == CHAT
    assert get.status_code == 200
    assert get.json()["studioforge"]["instance_id"] == CHAT


def test_no_selector_with_two_residents_is_a_400_model_required(
    tmp_path: Path, child: StubChild
) -> None:
    app = build_app(
        tmp_path, [record(CHAT), record(OTHER)], [instance(CHAT), instance(OTHER)], child
    )
    with TestClient(app) as http:
        for response in (
            http.post("/tokenize", json={"content": "x"}),
            http.get("/props"),
            http.get("/slots"),
        ):
            assert response.status_code == 400
            error = response.json()["error"]
            assert error["code"] == "model_required"
            assert error["param"] == "model"
            assert sorted(error["studioforge"]["resident"]) == [CHAT, OTHER]
            assert "X-SF-Model" in error["message"]
            assert error["fix"]
    assert child.requests == []


def test_no_selector_with_nothing_resident_is_a_400_model_required(
    tmp_path: Path, child: StubChild
) -> None:
    app = build_app(tmp_path, [record(CHAT)], [], child)
    with TestClient(app) as http:
        response = http.post("/completion", json={"prompt": "x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_required"
    assert response.json()["error"]["studioforge"]["resident"] == []


def test_the_route_kind_narrows_the_residents(tmp_path: Path, child: StubChild) -> None:
    """A chat model and an embedder resident: /completion without a name is the chat one."""
    app = build_app(
        tmp_path,
        [record(CHAT), record(EMBED, kind="embedding")],
        [instance(CHAT), instance(EMBED)],
        child,
    )
    with TestClient(app) as http:
        completion = http.post("/completion", json={"prompt": "x"})
        embedding = http.post("/embedding", json={"content": "x"})
        tokenize = http.post("/tokenize", json={"content": "x"})
    assert completion.headers["x-sf-model"] == CHAT
    assert embedding.headers["x-sf-model"] == EMBED
    # Any kind may tokenize: two residents, no selector, so no single answer.
    assert tokenize.status_code == 400


def test_a_configured_default_model_stands_in_when_the_choice_is_ambiguous(
    tmp_path: Path, child: StubChild
) -> None:
    app = build_app(
        tmp_path,
        [record(CHAT), record(OTHER)],
        [instance(CHAT), instance(OTHER)],
        child,
        models={"default_model": OTHER},
    )
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x"})
    assert response.status_code == 200
    assert response.headers["x-sf-model"] == OTHER


def test_the_selector_may_be_the_query_or_the_header(tmp_path: Path, child: StubChild) -> None:
    app = build_app(
        tmp_path, [record(CHAT), record(OTHER)], [instance(CHAT), instance(OTHER)], child
    )
    with TestClient(app) as http:
        by_query = http.post("/tokenize", params={"model": OTHER}, json={"content": "x"})
        by_header = http.post("/tokenize", headers={"X-SF-Model": OTHER}, json={"content": "x"})
        both = http.post(
            "/tokenize",
            params={"model": CHAT},
            headers={"X-SF-Model": OTHER},
            json={"content": "x"},
        )
        body_wins = http.post(
            "/tokenize", params={"model": CHAT}, json={"content": "x", "model": OTHER}
        )
    assert by_query.headers["x-sf-model"] == OTHER
    assert by_header.headers["x-sf-model"] == OTHER
    # Query outranks header, body outranks both.
    assert both.headers["x-sf-model"] == CHAT
    assert body_wins.headers["x-sf-model"] == OTHER


def test_an_unknown_model_is_a_404_model_not_found(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x", "model": "nope/none"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_the_loaded_alias_picks_the_largest_resident(tmp_path: Path, child: StubChild) -> None:
    app = build_app(
        tmp_path,
        [record(CHAT, param_count=3_000_000_000), record(OTHER, param_count=30_000_000_000)],
        [instance(CHAT), instance(OTHER)],
        child,
    )
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x", "model": "loaded"})
    assert response.headers["x-sf-model"] == OTHER
    # The alias is not echoed to the child as a model name.
    assert child.body()["model"] == OTHER


def test_embedding_refuses_a_chat_model(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/embedding", json={"content": "x", "model": CHAT})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "not_an_embedding_model"


# ---------------------------------------------------------------------------
# Just-in-time loading and ?autoload
# ---------------------------------------------------------------------------


def test_a_post_for_a_named_unloaded_model_loads_it_just_in_time(app: Any) -> None:
    called = jit_loader(app)
    with TestClient(app) as http:
        response = http.post("/tokenize", json={"content": "x", "model": OTHER})
    assert response.status_code == 200
    assert called == [OTHER]
    assert response.headers["x-sf-model"] == OTHER


def test_autoload_false_refuses_the_jit_with_404_model_not_loaded(
    app: Any, child: StubChild
) -> None:
    called = jit_loader(app)
    with TestClient(app) as http:
        refused = http.post(
            "/tokenize", params={"autoload": "false"}, json={"content": "x", "model": OTHER}
        )
        allowed = http.post(
            "/tokenize", params={"autoload": "false"}, json={"content": "x", "model": CHAT}
        )
    assert refused.status_code == 404
    assert refused.json()["error"]["code"] == "model_not_loaded"
    assert refused.json()["error"]["param"] == "model"
    assert called == []
    # A model that is already serving passes.
    assert allowed.status_code == 200


def test_a_get_never_loads_unless_asked(app: Any) -> None:
    called = jit_loader(app)
    with TestClient(app) as http:
        polled = http.get("/props", params={"model": OTHER})
        asked = http.get("/props", params={"model": OTHER, "autoload": "true"})
    assert polled.status_code == 404
    assert polled.json()["error"]["code"] == "model_not_loaded"
    assert asked.status_code == 200
    assert called == [OTHER]


def test_autoload_false_on_the_openai_routes(app: Any) -> None:
    called = jit_loader(app)
    messages = [{"role": "user", "content": "hi"}]
    with TestClient(app) as http:
        chat = http.post(
            "/v1/chat/completions",
            params={"autoload": "false"},
            json={"model": OTHER, "messages": messages},
        )
        stream = http.post(
            "/v1/chat/completions",
            params={"autoload": "0"},
            json={"model": OTHER, "messages": messages, "stream": True},
        )
        completions = http.post(
            "/v1/completions", params={"autoload": "false"}, json={"model": OTHER, "prompt": "x"}
        )
        resident = http.post(
            "/v1/chat/completions",
            params={"autoload": "false"},
            json={"model": CHAT, "messages": messages},
        )
    for refused in (chat, stream, completions):
        assert refused.status_code == 404
        assert refused.json()["error"]["code"] == "model_not_loaded"
    assert OTHER not in called
    assert resident.status_code == 200


# ---------------------------------------------------------------------------
# Scopes and the unknown-path envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "/tokenize",
        "/detokenize",
        "/apply-template",
        "/completion",
        "/infill",
        "/embedding",
        "/reranking",
        "/rerank",
        "/models/load",
        "/models/unload",
    ],
)
def test_posting_a_native_needs_an_inference_key_not_an_admin_one(path: str) -> None:
    assert required_scope("POST", path) == "inference"


def test_reading_the_natives_needs_a_read_key() -> None:
    for path in ("/props", "/slots", "/models", "/v2/models/x/ready"):
        assert required_scope("GET", path) == "read"


def test_an_inference_key_can_use_the_natives_and_a_read_key_cannot(
    tmp_path: Path, child: StubChild
) -> None:
    app = build_app(
        tmp_path,
        [record(CHAT)],
        [instance(CHAT)],
        child,
        server={"api_key": "admin-key-0123456789"},
    )
    with TestClient(app) as http:
        no_key = http.post("/tokenize", json={"content": "x"})
        ok = http.post(
            "/tokenize",
            json={"content": "x"},
            headers={"Authorization": "Bearer admin-key-0123456789"},
        )
    assert no_key.status_code == 401
    assert ok.status_code == 200


def test_an_unknown_root_path_is_still_a_404_in_the_envelope(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/definitely/not/here")
        completions = http.post("/completions", json={"prompt": "x"})
    for r in (response, completions):
        assert r.status_code in {404, 405}
        error = r.json()["error"]
        assert error["code"] in {"not_found", "method_not_allowed"}
        assert error["type"] == "invalid_request_error"
    assert response.json()["error"]["code"] == "not_found"


def test_the_stub_child_saw_what_the_caller_sent(app: Any, child: StubChild) -> None:
    with TestClient(app) as http:
        http.post("/apply-template", json={"messages": [{"role": "user", "content": "hi"}]})
    assert json.loads(child.last().content) == {"messages": [{"role": "user", "content": "hi"}]}
