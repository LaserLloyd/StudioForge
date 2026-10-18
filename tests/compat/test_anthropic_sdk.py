"""G3: the ``anthropic`` SDK against a running instance (``POST /v1/messages``)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytestmark = pytest.mark.compat

WEATHER = {
    "name": "get_weather",
    "description": "Current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}


def test_messages_create(anthropic_client: Any, chat_model: str) -> None:
    message = anthropic_client.messages.create(
        model=chat_model, max_tokens=64, messages=[{"role": "user", "content": "Say hi."}]
    )
    assert message.role == "assistant"
    assert message.stop_reason in {"end_turn", "max_tokens", "tool_use", "stop_sequence"}
    assert message.usage.output_tokens > 0


def test_messages_stream(anthropic_client: Any, chat_model: str) -> None:
    with anthropic_client.messages.stream(
        model=chat_model, max_tokens=64, messages=[{"role": "user", "content": "Count to three."}]
    ) as stream:
        text = "".join(stream.text_stream)
        final = stream.get_final_message()
    assert text.strip()
    assert final.usage.output_tokens > 0


def test_tool_use_round_trip(anthropic_client: Any, chat_model: str) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "What is the weather in Oslo?"}]
    first = anthropic_client.messages.create(
        model=chat_model,
        max_tokens=256,
        tools=[WEATHER],
        tool_choice={"type": "any"},
        messages=messages,
    )
    assert first.stop_reason == "tool_use"
    block = next(b for b in first.content if b.type == "tool_use")
    assert block.name == "get_weather"
    assert "city" in block.input
    messages.append({"role": "assistant", "content": [b.model_dump() for b in first.content]})
    messages.append(
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": block.id, "content": "4 C, rain"}],
        }
    )
    second = anthropic_client.messages.create(
        model=chat_model, max_tokens=128, tools=[WEATHER], messages=messages
    )
    assert any(b.type == "text" and b.text for b in second.content)


def test_x_api_key_and_bearer_both_authenticate(
    base_url: str, api_key: str, chat_model: str
) -> None:
    body = {"model": chat_model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    for headers in ({"x-api-key": api_key}, {"Authorization": f"Bearer {api_key}"}):
        response = httpx.post(f"{base_url}/v1/messages", json=body, headers=headers, timeout=600)
        assert response.status_code == 200, response.text
