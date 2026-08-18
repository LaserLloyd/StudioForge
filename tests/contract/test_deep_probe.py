"""The deep health probe against a REAL llama-server.

The unit tests pin the probe's logic against a fake transport. This suite pins
the half that a fake cannot: that the probe's SSE parsing matches what
``llama-server`` actually emits. The incident it exists for (*"the control
channel reported 'connected' and GET /models returned 200 for hours while every
long stream died mid-generation"*) is precisely a case where the shallow signal
and the real one disagreed, so a probe that mis-parses the real framing would
reproduce the bug it is meant to catch -- with more confidence attached.
"""

from __future__ import annotations

import httpx
import pytest

from tests.contract.conftest import requires_engine, requires_models

pytestmark = [requires_engine, requires_models]


@pytest.fixture
def loaded_model(raw: httpx.Client, chat_model: str) -> str:
    response = raw.post(f"/api/models/{chat_model}/load", timeout=600.0)
    assert response.status_code == 200, response.text
    return chat_model


def test_shallow_health_says_nothing_about_generation(
    raw: httpx.Client, loaded_model: str
) -> None:
    """The endpoint that lied: it is still fast, still shallow, still 200."""
    body = raw.get("/health").json()

    assert body["status"] == "ok"
    assert "probe" not in body
    assert loaded_model in body["loaded_models"]


def test_deep_health_really_generates(raw: httpx.Client, loaded_model: str) -> None:
    """A real streamed completion, parsed from real llama-server SSE frames."""
    body = raw.get("/health", params={"deep": "true"}, timeout=120.0).json()

    assert body["status"] == "ok", body
    entry = next(m for m in body["probe"]["models"] if m["model_id"] == loaded_model)
    assert entry["ok"] is True
    assert entry["tokens"] > 0, "no tokens arrived: the probe cannot prove generation works"
    assert entry["first_token_ms"] is not None
    assert entry["error"] is None


def test_single_model_probe_endpoint(raw: httpx.Client, loaded_model: str) -> None:
    response = raw.post(f"/api/models/{loaded_model}/probe", timeout=120.0)

    body = response.json()
    assert response.status_code == 200, response.text
    assert body["status"] == "ok"
    assert body["models"][0]["tokens"] > 0


def test_probe_is_cheap_enough_to_schedule(raw: httpx.Client, loaded_model: str) -> None:
    """It has to be runnable on a timer, so it must stay small and bounded."""
    body = raw.post(f"/api/models/{loaded_model}/probe", timeout=120.0).json()

    assert body["duration_ms"] < 20_000
    assert body["models"][0]["tokens"] <= 8


def test_an_unloaded_model_is_reported_honestly(raw: httpx.Client, chat_model: str) -> None:
    """A probe that cannot fail is worse than none: never pass by default."""
    raw.post(f"/api/models/{chat_model}/unload", timeout=120.0)

    body = raw.post(f"/api/models/{chat_model}/probe", timeout=120.0).json()

    assert body["status"] == "degraded"
    assert body["models"][0]["ok"] is False
    assert "not loaded" in body["models"][0]["error"]
