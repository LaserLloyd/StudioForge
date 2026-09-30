"""``id_slot`` reaches llama-server untouched (D72).

A client that splits one model into a chat slot and a utility slot pins every
request with ``id_slot`` -- llama-server's own field, read from the request
body on ``/v1/chat/completions`` and ``/v1/completions`` (b11037
server-context.cpp: ``task.id_slot = json_value(data, "id_slot", -1)``). The
gateway consumes exactly two body fields of its own (``ttl`` and
``priority``) and forwards the rest; these pin that ``id_slot`` stays in "the
rest", on both the plain and the streaming path, so a later change to the
payload handling cannot quietly send every request to whichever slot the
engine picks.

The upstream is stubbed: what these assert is the payload the gateway hands
to the child.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api import openai_routes
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
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
    def touch(self, model_id: str) -> None:
        return None


class Supervisor(FakeSupervisor):
    """A ready resident, plus the ``base_url`` only the streaming path asks for."""

    def base_url(self, model_id: str) -> str | None:
        return "http://127.0.0.1:1/fake"


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
    two_slots = make_plan().model_copy(update={"parallel": 2, "kv_unified": True})
    supervisor = Supervisor(
        [InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=two_slots)]
    )
    built.state.supervisor = supervisor
    built.state.manager.supervisor = supervisor
    return built


@pytest.fixture()
def sent(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Every payload handed to the child, plain or streamed, newest last."""
    payloads: list[dict[str, Any]] = []

    async def fake_forward(
        state: Any, record: Any, path: str, payload: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        payloads.append(dict(payload))
        return {"id": "cmpl-test", "object": "chat.completion", "choices": []}

    async def fake_stream_upstream(
        state: Any,
        record: Any,
        url: str,
        payload: dict[str, Any],
        started: float,
        **_kwargs: Any,
    ):
        payloads.append(dict(payload))
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(openai_routes, "_forward", fake_forward)
    monkeypatch.setattr(openai_routes, "_stream_upstream", fake_stream_upstream)
    return payloads


@pytest.mark.parametrize("stream", [False, True], ids=["plain", "streamed"])
def test_a_chat_request_keeps_its_id_slot(
    app: Any, sent: list[dict[str, Any]], stream: bool
) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": MESSAGES,
                "id_slot": 1,
                "priority": 1,
                "stream": stream,
            },
        )
    assert response.status_code == 200, response.text
    assert len(sent) == 1
    assert sent[0]["id_slot"] == 1, "the slot the client pinned, as it sent it"
    assert "priority" not in sent[0], "the gateway's own field is still consumed"


def test_slot_zero_is_a_slot_not_a_missing_value(app: Any, sent: list[dict[str, Any]]) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/chat/completions",
            json={"model": MODEL_ID, "messages": MESSAGES, "id_slot": 0},
        )
    assert response.status_code == 200, response.text
    assert sent[0]["id_slot"] == 0


def test_a_completions_request_keeps_its_id_slot(app: Any, sent: list[dict[str, Any]]) -> None:
    with TestClient(app) as http:
        response = http.post(
            "/v1/completions",
            json={"model": MODEL_ID, "prompt": "hello", "id_slot": 1, "ttl": 60},
        )
    assert response.status_code == 200, response.text
    assert sent[0]["id_slot"] == 1
    assert "ttl" not in sent[0]


def test_a_request_without_id_slot_is_not_given_one(app: Any, sent: list[dict[str, Any]]) -> None:
    """The engine's own routing decides then; the gateway invents nothing."""
    with TestClient(app) as http:
        http.post("/v1/chat/completions", json={"model": MODEL_ID, "messages": MESSAGES})
    assert "id_slot" not in sent[0]
