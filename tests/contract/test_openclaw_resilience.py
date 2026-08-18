"""Behaviours OpenClaw depends on that go beyond raw OpenAI parity.

Each test here corresponds to a documented way local LLM servers appear broken:
a cold load that looks like a hang, an empty reply from a thinking model, a
silently-ignored sampler, a client that did not name a model.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
import pytest

from tests.contract.conftest import ServerHandle, requires_engine, requires_models

pytestmark = [requires_engine, requires_models]

TINY = "Qwen2.5-0.5B-Instruct-Q8_0"


def resolve_or_skip(server: ServerHandle, needle: str) -> str:
    model = server.resolve_model(needle)
    if model is None:
        pytest.skip(f"model matching {needle!r} is not present")
    return model


# ---------------------------------------------------------------------------
# Default model / model aliases
# ---------------------------------------------------------------------------


@pytest.fixture
def with_default_model(live_server: ServerHandle, chat_model: str):
    """Point models.default_model at the tiny model for the duration."""
    config = live_server.app.state.config
    previous = config.models.default_model
    config.models.default_model = chat_model
    try:
        yield chat_model
    finally:
        config.models.default_model = previous


def test_request_without_model_uses_the_default(raw: httpx.Client, with_default_model: str) -> None:
    """A client that never names a model still gets served."""
    response = raw.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 16},
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"]


@pytest.mark.parametrize("alias", ["local-model", "default", "auto", "current"])
def test_default_model_aliases_resolve(
    raw: httpx.Client, with_default_model: str, alias: str
) -> None:
    """LM Studio clients send the literal 'local-model' as a fallback."""
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": alias,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200, response.text


def test_missing_model_without_a_default_is_a_clear_400(raw: httpx.Client) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8},
    )
    assert response.status_code == 400
    assert "models.default_model" in response.json()["error"]["message"]


def test_local_model_without_a_default_explains_itself(raw: httpx.Client) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": "local-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "no_default_model"


# ---------------------------------------------------------------------------
# Cold-start keep-alives
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
def test_streaming_cold_start_is_never_silent(raw: httpx.Client, live_server: ServerHandle) -> None:
    """A cold streaming load must not be a silent socket.

    The tiny model loads in ~2s, so keep-alives may legitimately not fire. What
    must always hold is that bytes begin flowing without a long silence and the
    stream terminates correctly -- that is the property a client's read timeout
    actually depends on.
    """
    model = resolve_or_skip(live_server, TINY)
    raw.post(f"/api/models/{model}/unload")
    # A short interval makes the behaviour observable even on a fast load.
    live_server.app.state.config.gateway.stream_keepalive_interval_s = 0.25

    first_byte_at: float | None = None
    started = time.perf_counter()
    saw_keepalive = False
    parts: list[str] = []
    with raw.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count to five."}],
            "max_tokens": 48,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        for text in response.iter_text():
            if not text:
                continue
            if first_byte_at is None:
                first_byte_at = time.perf_counter() - started
            if text.lstrip().startswith(":"):
                saw_keepalive = True
            parts.append(text)

    body = "".join(parts)
    assert body.rstrip().endswith("data: [DONE]")
    assert first_byte_at is not None
    assert saw_keepalive or first_byte_at < 30.0, (
        f"no keep-alive and first byte took {first_byte_at:.1f}s"
    )


def test_keepalive_comments_do_not_break_the_openai_client(
    client: Any, live_server: ServerHandle, raw: httpx.Client
) -> None:
    """SSE comment lines must be invisible to a strict parser."""
    model = resolve_or_skip(live_server, TINY)
    raw.post(f"/api/models/{model}/unload")
    live_server.app.state.config.gateway.stream_keepalive_interval_s = 0.25
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello."}],
        max_tokens=32,
        stream=True,
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.strip()


@pytest.mark.timeout(300)
def test_validation_happens_before_the_stream_starts(
    raw: httpx.Client, live_server: ServerHandle
) -> None:
    """A bad streaming request gets a real 4xx, not an error inside a 200."""
    model = resolve_or_skip(live_server, TINY)
    raw.post(f"/api/models/{model}/unload")
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        },
                    ],
                }
            ],
            "stream": True,
        },
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["code"] == "model_not_multimodal"


# ---------------------------------------------------------------------------
# Sampler aliases and request-level ttl
# ---------------------------------------------------------------------------


def test_repetition_penalty_alias_is_accepted(raw: httpx.Client, chat_model: str) -> None:
    """LM Studio silently ignores this spelling; we translate it."""
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
            "repetition_penalty": 1.1,
        },
    )
    assert response.status_code == 200, response.text


def test_request_level_ttl_sets_the_idle_timer(raw: httpx.Client, chat_model: str) -> None:
    """LM Studio lets a client attach a ttl to a chat request."""
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 8,
            "ttl": 1234,
        },
    )
    assert response.status_code == 200, response.text
    loaded = {i["model_id"]: i for i in raw.get("/api/status").json()["loaded"]}
    assert loaded[chat_model]["ttl_s"] == 1234


def test_bad_ttl_type_is_ignored_not_fatal(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "ttl": "not-a-number",
        },
    )
    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Reasoning models never return an empty reply
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_reasoning_model_returns_non_empty_content(
    raw: httpx.Client, client: Any, live_server: ServerHandle
) -> None:
    """The measured failure: content len 0 with the text in reasoning_content."""
    model = live_server.resolve_model("DeepSeek-R1-0528-Qwen3-8B")
    if model is None:
        pytest.skip("no reasoning model in the library")
    plan = raw.get(f"/api/models/{model}/plan", params={"ctx_size": 4096}).json()
    if not plan.get("fits"):
        pytest.skip(f"reasoning model does not fit right now: {plan.get('message')}")

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17*3? Answer briefly."}],
        max_tokens=150,
        temperature=0.0,
    )
    content = completion.choices[0].message.content or ""
    assert content.strip(), (
        "reasoning model returned empty content -- check reasoning_format is 'none'"
    )


# ---------------------------------------------------------------------------
# Errors are never HTML, never a 200
# ---------------------------------------------------------------------------


def test_unknown_route_is_json_404_not_html(raw: httpx.Client) -> None:
    """LM Studio returns 200 for unrouted paths; that trap cost real debugging."""
    response = raw.get("/definitely/not/a/route")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert set(response.json()["error"]) >= {"message", "type", "code", "param"}


def test_wrong_method_is_json_not_html(raw: httpx.Client) -> None:
    response = raw.get("/v1/chat/completions")
    assert response.status_code in {404, 405}
    assert response.headers["content-type"].startswith("application/json")
    assert "error" in response.json()


def test_v1_models_reports_state(raw: httpx.Client, chat_model: str) -> None:
    """/v1/models must answer 'what is loaded?' without a second endpoint."""
    raw.post(f"/api/models/{chat_model}/unload")
    entry = next(m for m in raw.get("/v1/models").json()["data"] if m["id"] == chat_model)
    assert entry["state"] == "not-loaded"
    assert "loaded_context_length" not in entry

    raw.post(f"/api/models/{chat_model}/load", json={"ctx_size": 2048})
    entry = next(m for m in raw.get("/v1/models").json()["data"] if m["id"] == chat_model)
    assert entry["state"] == "loaded"
    assert entry["loaded_context_length"] == 2048


def test_insufficient_vram_carries_actionable_suggestions(
    raw: httpx.Client, live_server: ServerHandle
) -> None:
    """A 507 is terminal for those settings, so it must say what to do instead."""
    model = live_server.resolve_model("Qwen3.5-122B-A10B-heretic-v2")
    if model is None:
        pytest.skip("no oversized model in the library")
    response = raw.post(f"/api/models/{model}/load", json={"ctx_size": 262144, "parallel": 8})
    assert response.status_code == 507, response.text
    error = response.json()["error"]
    assert error["code"] == "insufficient_vram"
    details = error["studioforge"]
    assert details["suggestions"], "a VRAM rejection with no suggestions is useless"
    assert details["required_bytes"] > details["available_bytes"]
    assert details["estimate_mb"]["total"] > 0


def test_error_bodies_are_never_html(raw: httpx.Client) -> None:
    for path, payload in [
        ("/v1/chat/completions", {"model": "nope", "messages": [{"role": "user", "content": "x"}]}),
        ("/v1/embeddings", {"model": "nope", "input": "x"}),
        ("/api/models/nope/load", {}),
    ]:
        response = raw.post(path, json=payload)
        assert response.status_code >= 400
        assert "html" not in response.headers.get("content-type", "").lower()
        json.loads(response.text)


# ---------------------------------------------------------------------------
# Management routes with a single scalar body param
# ---------------------------------------------------------------------------


def test_pin_accepts_a_json_object_body(raw: httpx.Client, chat_model: str) -> None:
    """FastAPI treats a lone Body param as the whole body without embed=True.

    Without it, {"pinned": true} was rejected as "not a valid boolean" -- the
    documented call shape simply did not work.
    """
    try:
        raw.post(f"/api/models/{chat_model}/load", json={"ctx_size": 2048})
        response = raw.post(f"/api/models/{chat_model}/pin", json={"pinned": True})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["pinned"] is True
        # 0 is the wire representation of "pinned" the planner checks.
        assert body["effective_ttl_s"] == 0

        loaded = {i["model_id"]: i for i in raw.get("/api/status").json()["loaded"]}
        assert loaded[chat_model]["ttl_s"] == 0, "a pin must reach the running instance"
    finally:
        raw.post(f"/api/models/{chat_model}/pin", json={"pinned": False})


def test_unpin_restores_the_default_ttl(raw: httpx.Client, chat_model: str) -> None:
    raw.post(f"/api/models/{chat_model}/load", json={"ctx_size": 2048})
    raw.post(f"/api/models/{chat_model}/pin", json={"pinned": True})
    response = raw.post(f"/api/models/{chat_model}/pin", json={"pinned": False})
    assert response.status_code == 200
    assert response.json()["effective_ttl_s"] > 0
    loaded = {i["model_id"]: i for i in raw.get("/api/status").json()["loaded"]}
    assert loaded[chat_model]["ttl_s"] > 0


def test_rollback_requires_confirm_via_object_body(raw: httpx.Client) -> None:
    response = raw.post("/api/update/rollback", json={"confirm": False})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "confirmation_required"


def test_loaded_instances_carry_the_effective_ttl(raw: httpx.Client, chat_model: str) -> None:
    """models.default_ttl_s must reach the instance, or nothing ever unloads."""
    raw.post(f"/api/models/{chat_model}/load", json={"ctx_size": 2048})
    loaded = {i["model_id"]: i for i in raw.get("/api/status").json()["loaded"]}
    instance = loaded[chat_model]
    assert instance["ttl_s"] is not None, "instance ttl_s was None -> TTL sweeper is a no-op"
    assert instance["ttl_s"] > 0
