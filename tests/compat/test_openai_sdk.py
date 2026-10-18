"""G3: the ``openai`` SDK against a running instance."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import pytest

pytestmark = pytest.mark.compat

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_models_list(openai_client: Any, chat_model: str) -> None:
    ids = [m.id for m in openai_client.models.list().data]
    assert chat_model in ids


def test_chat_non_stream(openai_client: Any, chat_model: str) -> None:
    reply = openai_client.chat.completions.create(
        model=chat_model, messages=[{"role": "user", "content": "Say hi."}], max_tokens=64
    )
    assert reply.choices[0].message.content or reply.choices[0].message.tool_calls
    assert reply.usage.completion_tokens > 0


def test_chat_stream(openai_client: Any, chat_model: str) -> None:
    stream = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Count to three."}],
        max_tokens=64,
        stream=True,
    )
    pieces = [c.choices[0].delta.content or "" for c in stream if c.choices]
    assert "".join(pieces).strip()


def test_tool_call_round_trip(openai_client: Any, chat_model: str) -> None:
    messages: list[dict[str, Any]] = [{"role": "user", "content": "What is the weather in Oslo?"}]
    first = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=[WEATHER], tool_choice="required", max_tokens=256
    )
    call = first.choices[0].message.tool_calls[0]
    assert call.function.name == "get_weather"
    assert "city" in json.loads(call.function.arguments)
    messages.append(first.choices[0].message.model_dump(exclude_none=True))
    messages.append({"role": "tool", "tool_call_id": call.id, "content": "4 C, light rain"})
    second = openai_client.chat.completions.create(
        model=chat_model, messages=messages, tools=[WEATHER], max_tokens=128
    )
    assert second.choices[0].message.content


def test_json_schema(openai_client: Any, chat_model: str) -> None:
    reply = openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Give a city and a country."}],
        max_tokens=128,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "place",
                "schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
                    "required": ["city", "country"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
    )
    parsed = json.loads(reply.choices[0].message.content)
    assert set(parsed) == {"city", "country"}


def test_embeddings(openai_client: Any, embed_model: str) -> None:
    result = openai_client.embeddings.create(model=embed_model, input=["alpha", "beta"])
    assert len(result.data) == 2
    assert len(result.data[0].embedding) > 8


def test_completions(openai_client: Any, chat_model: str) -> None:
    result = openai_client.completions.create(model=chat_model, prompt="The sky is", max_tokens=8)
    assert result.choices[0].text


def test_responses_non_stream_and_stream(openai_client: Any, chat_model: str) -> None:
    once = openai_client.responses.create(model=chat_model, input="Say hi.", max_output_tokens=64)
    assert once.output_text
    with openai_client.responses.stream(
        model=chat_model, input="Say hi.", max_output_tokens=64
    ) as stream:
        deltas = [e.delta for e in stream if e.type == "response.output_text.delta"]
        final = stream.get_final_response()
    assert "".join(deltas) == final.output_text


def test_vision(openai_client: Any, vision_model: str) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    reply = openai_client.chat.completions.create(
        model=vision_model,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour is this image? One word."},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
    assert "red" in (reply.choices[0].message.content or "").lower()
