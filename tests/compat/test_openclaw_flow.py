"""G3: the OpenClaw-style flow and the native v1 surface, against a running instance.

``GET /api/v1/models`` (key present) -> ``POST /api/v1/models/load`` -> chat with
that key; the openai-responses flow; per-request ``ttl`` honoured on every front
door; and the refusals (507 with numbers, 501 chat, JSON 404).
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.compat


def test_native_models_then_load_then_chat(
    http: httpx.Client, openai_client: Any, chat_model: str
) -> None:
    listing = http.get("/api/v1/models")
    assert listing.status_code == 200, listing.text
    models = {m["key"]: m for m in listing.json()["models"]}
    assert chat_model in models, "the model key OpenClaw preloads must be listed"
    entry = models[chat_model]
    for field in ("type", "publisher", "key", "loaded_instances", "max_context_length"):
        assert field in entry, field
    assert "capabilities" in entry

    loaded = http.post("/api/v1/models/load", json={"model": entry["key"], "context_length": 8192})
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["status"] == "loaded"
    instance_id = loaded.json()["instance_id"]

    after = {m["key"]: m for m in http.get("/api/v1/models").json()["models"]}
    assert [i["id"] for i in after[chat_model]["loaded_instances"]] == [instance_id]

    reply = openai_client.chat.completions.create(
        model=entry["key"], messages=[{"role": "user", "content": "ping"}], max_tokens=16
    )
    assert reply.choices

    unloaded = http.post("/api/v1/models/unload", json={"instance_id": instance_id})
    assert unloaded.status_code == 200
    assert unloaded.json()["instance_id"] == instance_id


def test_openai_responses_flow_after_native_load(
    http: httpx.Client, openai_client: Any, chat_model: str
) -> None:
    assert http.post("/api/v1/models/load", json={"model": chat_model}).status_code == 200
    response = openai_client.responses.create(model=chat_model, input="ping", max_output_tokens=16)
    assert response.status in {"completed", "incomplete"}


def test_per_request_ttl_is_honoured_on_every_front_door(
    http: httpx.Client, openai_client: Any, anthropic_client: Any, chat_model: str
) -> None:
    def remaining() -> float:
        rows = http.get("/api/models").json()
        rows = rows["models"] if isinstance(rows, dict) else rows
        row = next(r for r in rows if r["id"] == chat_model)
        return float(row["ttl_remaining_s"])

    http.post("/api/v1/models/load", json={"model": chat_model, "ttl": 90})
    assert 0 < remaining() <= 90

    openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=4,
        extra_body={"ttl": 60},
    )
    assert 0 < remaining() <= 60

    anthropic_client.messages.create(
        model=chat_model,
        max_tokens=4,
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"ttl": 45},
    )
    assert 0 < remaining() <= 45

    openai_client.responses.create(
        model=chat_model, input="hi", max_output_tokens=16, extra_body={"ttl": 30}
    )
    assert 0 < remaining() <= 30


def test_stateful_chat_is_a_clear_501(http: httpx.Client, chat_model: str) -> None:
    response = http.post("/api/v1/chat", json={"model": chat_model, "input": "hi"})
    assert response.status_code == 501
    assert response.json()["error"]["code"] == "not_implemented"


def test_unknown_routes_are_json_404(http: httpx.Client) -> None:
    response = http.get("/api/v1/unknown")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert "error" in response.json()


def test_a_model_that_cannot_fit_is_a_507_with_numbers(
    http: httpx.Client, oversize_model: str
) -> None:
    """Never a CPU spill (STRICT_GPU on): the refusal carries the planner's numbers."""
    response = http.post("/api/v1/models/load", json={"model": oversize_model})
    assert response.status_code == 507, response.text
    error = response.json()["error"]
    assert error["code"] == "insufficient_vram"
    assert error["studioforge"]


@pytest.mark.skipif(
    not os.environ.get("COMPAT_LMSTUDIO_URL"),
    reason="COMPAT_LMSTUDIO_URL is not set (no real LM Studio to diff against)",
)
def test_models_shape_matches_a_real_lm_studio(http: httpx.Client) -> None:
    real = httpx.get(os.environ["COMPAT_LMSTUDIO_URL"].rstrip("/") + "/api/v1/models", timeout=30)
    assert real.status_code == 200
    theirs = real.json()["models"][0]
    ours = http.get("/api/v1/models").json()["models"][0]
    missing = sorted(set(theirs) - set(ours))
    assert not missing, f"LM Studio emits fields we do not: {missing}"
