"""The per-request load tier on the OpenAI inference path (D48 / C7).

``priority`` is an additive JSON body field on POST /v1/chat/completions and
POST /v1/completions, beside LM Studio's ``ttl``. Three promises, exercised
through the real app because all three are wiring:

* it is **consumed here and never forwarded** -- llama-server does not know the
  field, which some builds answer with an unknown-parameter error and others
  ignore silently;
* junk is a **400 naming the parameter**, not the ignore-junk treatment ``ttl``
  gets: a mistyped tier silently demotes a chat turn to background and it then
  meets 503s nothing downstream could attribute back to the typo;
* it gates **this request's own admission** -- a tier-1 turn passes a hold that
  refuses a background turn for the same model -- and it never re-tiers an
  instance somebody else's traffic is already sharing.

The upstream is stubbed: what these assert is the payload the gateway *would*
have sent, and the status code the client sees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.priority import PRIORITY_AGENT, PRIORITY_CHAT
from studioforge.types import InstanceInfo
from tests.unit.test_catalog_routes import (
    MODEL_ID,
    FakeProbe,
    FakeRegistry,
    FakeSupervisor,
    make_plan,
    make_record,
)

MESSAGES = [{"role": "user", "content": "hello"}]


class Registry(FakeRegistry):
    """``FakeRegistry`` plus the ``touch`` every JIT path calls."""

    def touch(self, model_id: str) -> None:
        return None


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    built = create_app(config, state=build_state(config), start_background=False)
    built.state.registry = Registry([make_record()])
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    # Resident and ready, so ensure_loaded returns without planning anything:
    # these tests are about the field, not about a load.
    supervisor = FakeSupervisor(
        [InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())]
    )
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


@pytest.fixture()
def upstream(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every payload the gateway hands to llama-server, newest last."""
    sent: list[dict[str, Any]] = []

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        sent.append(dict(payload))
        return {"id": "cmpl-test", "object": "chat.completion", "choices": []}

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    return sent


# ---------------------------------------------------------------------------
# Consumed, never forwarded
# ---------------------------------------------------------------------------


def test_a_chat_requests_priority_never_reaches_llama_server(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "priority": 1, "ttl": 60},
        )
    assert response.status_code == 200, response.text
    assert len(upstream) == 1
    assert "priority" not in upstream[0]
    assert "ttl" not in upstream[0], "the pre-D48 sibling field, still consumed"


def test_a_completions_requests_priority_never_reaches_llama_server(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/completions",
            json={"model": MODEL_ID, "prompt": "hello", "priority": 2},
        )
    assert response.status_code == 200, response.text
    assert "priority" not in upstream[0]
    assert upstream[0]["prompt"] == "hello"


# ---------------------------------------------------------------------------
# Junk is refused, not ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 4, -1, "1", True])
def test_a_junk_priority_is_a_400_naming_the_parameter(
    app: Any, upstream: list[dict[str, Any]], bad: Any
) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "priority": bad},
        )
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "priority"
    assert upstream == [], "refused before anything was forwarded"


# ---------------------------------------------------------------------------
# The tier gates this request, and only this request
# ---------------------------------------------------------------------------


def test_a_hold_refuses_a_background_turn_and_lets_a_chat_turn_through(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """The point of the field: while the chat model is loading, a bot's
    background traffic drains off the cards, and a person's turn does not."""
    app.state.manager._priority_holds["chat/model"] = PRIORITY_CHAT

    with TestClient(app) as http:
        held = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "priority": 3},
        )
        assert held.status_code == 503, held.text
        error = held.json()["error"]
        assert error["code"] == "priority_hold"
        assert error["studioforge"]["priority_hold"] == {
            "model_id": "chat/model",
            "priority": PRIORITY_CHAT,
        }
        assert upstream == []

        allowed = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "priority": 1},
        )
        assert allowed.status_code == 200, allowed.text
    assert len(upstream) == 1


def test_an_unpriced_turn_is_held_on_the_models_own_tier(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """Every pre-D48 client sends no tier and is admitted on the model's
    standing one, exactly as before the field existed."""
    app.state.manager._priority_holds["agent/model"] = PRIORITY_AGENT

    with TestClient(app) as http:
        held = http.post("/v1/chat/completions", json={"model": MODEL_ID, "messages": MESSAGES})
        assert held.status_code == 503, held.text
        assert held.json()["error"]["code"] == "priority_hold"
    assert upstream == []


def test_a_request_tier_does_not_re_tier_the_instance_it_shares(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """One bot naming ``priority: 1`` must not promote the child a person's
    chat is already served by -- the tier a resident carries stays what loaded
    it, and nothing is remembered per model either."""
    instance = app.state.supervisor.get(MODEL_ID)
    assert instance is not None and instance.priority == 3

    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "priority": 1},
        )

    assert response.status_code == 200, response.text
    assert instance.priority == 3
    assert app.state.manager._model_priority == {}


# ---------------------------------------------------------------------------
# A lease refusal reaches a streaming client as a real 507 (D53)
# ---------------------------------------------------------------------------


def test_a_streaming_request_meets_a_lease_as_a_507_not_an_sse_frame(
    app: Any, upstream: list[dict[str, Any]]
) -> None:
    """The refusal has to arrive BEFORE the 200 or nobody sees it.

    A streaming request loads inside the SSE body, so a lease refusal used to
    be an error frame inside an HTTP 200 -- invisible to exactly the client a
    lease refusal is written for, which branches on the status code.
    """
    app.state.supervisor = FakeSupervisor([])
    app.state.manager.supervisor = app.state.supervisor
    lease = app.state.manager.leases.acquire(
        [0], holder="crucibleforge-judge", reason="8-stream benchmark"
    )

    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "stream": True},
        )

    assert response.status_code == 507, response.text
    error = response.json()["error"]
    assert error["code"] == "gpu_leased"
    assert "leased" in error["message"], "the word existing clients match on survives"
    assert "0.00 GiB" not in error["message"]
    detail = error["studioforge"]
    assert detail["lease"]["id"] == lease.id
    assert detail["lease"]["holder_family"] == "crucibleforge"
    # The wait is a fact the server knows, so it goes in the header every HTTP
    # client already understands -- 507 never carried one before.
    assert response.headers["Retry-After"] == str(int(detail["retry_after_s"]))
    assert upstream == []


def test_a_model_already_serving_is_never_refused_by_the_lease_precheck(
    app: Any, upstream: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precheck gates a *load*, not a request.

    A lease taking the last card says nothing about a model already resident on
    it -- refusing its traffic would turn a display improvement into an outage.
    """

    async def fake_stream(*_args: Any, **_kwargs: Any) -> Any:
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(openai_routes, "_stream_with_jit_load", fake_stream)
    app.state.manager.leases.acquire([0], holder="crucibleforge", reason="benchmark")

    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "stream": True},
        )

    assert response.status_code == 200, response.text
