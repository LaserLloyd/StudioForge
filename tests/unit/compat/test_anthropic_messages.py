"""Anthropic-compatible ``POST /v1/messages`` -- W2.

Translation in both directions (pure functions), the route through the real app
with the engine stubbed, the SSE event sequence (including awkward chunk
boundaries), the tool-use round trip, error envelopes, and both key headers.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.anthropic_routes import anthropic_to_chat, chat_to_anthropic, stop_reason
from studioforge.errors import BadRequestError
from tests.unit.compat.conftest import (
    VISION_ID,
    Upstream,
    assert_shape,
    chunk,
    load_fixture,
    openai_reply,
    sse,
)
from tests.unit.test_catalog_routes import MODEL_ID

MESSAGE = {"model": MODEL_ID, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}


def events_of(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse an Anthropic SSE body into ``(event, data)`` pairs."""
    out: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = next(line[6:] for line in lines if line.startswith("data: "))
        parsed = json.loads(data)
        assert parsed["type"] == name, "the SSE event name must equal the data's type"
        out.append((name, parsed))
    return out


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def test_the_documented_request_translates_to_a_chat_body() -> None:
    chat, context = anthropic_to_chat(load_fixture("anthropic_messages_request.json"))
    assert chat["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Say hello."},
    ]
    assert chat["max_tokens"] == 256
    assert chat["temperature"] == 0.2
    assert chat["stop"] == ["END"]
    assert "stream" not in chat
    assert context == {"stop_sequences": ["END"], "thinking": False}


def test_system_as_blocks_and_sampler_fields() -> None:
    chat, _ = anthropic_to_chat(
        {
            **MESSAGE,
            "system": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}],
            "top_p": 0.9,
            "top_k": 40,
            "stream": True,
        }
    )
    assert chat["messages"][0] == {"role": "system", "content": "A\n\nB"}
    assert (chat["top_p"], chat["top_k"]) == (0.9, 40)
    assert chat["stream"] is True
    assert chat["stream_options"] == {"include_usage": True}


def test_the_tool_round_trip_request_becomes_openai_tool_messages() -> None:
    chat, _ = anthropic_to_chat(load_fixture("anthropic_tool_request.json"))
    assert chat["tools"] == [
        {
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
    ]
    assert chat["tool_choice"] == "auto"
    assert chat["messages"] == [
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "toolu_01",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
                }
            ],
        },
        # tool results come first, then the user's own text -- the order every
        # OpenAI-style chat template expects.
        {"role": "tool", "tool_call_id": "toolu_01", "content": "4 C, light rain"},
        {"role": "user", "content": "Summarize."},
    ]


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "t"}, {"type": "function", "function": {"name": "t"}}),
    ],
)
def test_tool_choice_mapping(choice: dict[str, Any], expected: Any) -> None:
    body = {**MESSAGE, "tools": [{"name": "t", "input_schema": {"type": "object"}}]}
    chat, _ = anthropic_to_chat({**body, "tool_choice": choice})
    assert chat["tool_choice"] == expected


def test_disable_parallel_tool_use_maps_to_parallel_tool_calls_false() -> None:
    body = {**MESSAGE, "tools": [{"name": "t", "input_schema": {"type": "object"}}]}
    chat, _ = anthropic_to_chat(
        {**body, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}}
    )
    assert chat["parallel_tool_calls"] is False


def test_tool_result_with_blocks_errors_and_images() -> None:
    chat, _ = anthropic_to_chat(
        {
            **MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "is_error": True,
                            "content": [
                                {"type": "text", "text": "disk full"},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": "AAAA",
                                    },
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    )
    assert chat["messages"][0] == {
        "role": "tool",
        "tool_call_id": "t1",
        "content": "Error: disk full",
    }
    # An image cannot ride in a tool message; it follows as a user turn.
    assert chat["messages"][1]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    ]


def test_image_blocks_become_image_urls() -> None:
    chat, _ = anthropic_to_chat(
        {
            **MESSAGE,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": "QUJD",
                            },
                        },
                        {"type": "image", "source": {"type": "url", "url": "https://x.test/a.png"}},
                    ],
                }
            ],
        }
    )
    assert chat["messages"][0]["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]


def test_thinking_blocks_in_history_are_dropped() -> None:
    chat, _ = anthropic_to_chat(
        {
            **MESSAGE,
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "hm", "signature": "s"},
                        {"type": "text", "text": "a"},
                    ],
                },
                {"role": "user", "content": "q2"},
            ],
        }
    )
    assert chat["messages"][1] == {"role": "assistant", "content": "a"}


@pytest.mark.parametrize(
    ("patch", "param"),
    [
        ({"max_tokens": None}, "max_tokens"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"messages": []}, "messages"),
        ({"messages": [{"role": "system", "content": "x"}]}, "messages[0].role"),
        ({"messages": [{"role": "user", "content": 5}]}, "messages[0].content"),
        (
            {"messages": [{"role": "user", "content": [{"type": "document", "source": {}}]}]},
            "messages[0].content[0]",
        ),
        (
            {"messages": [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]},
            "messages[0].content[0]",
        ),
        ({"tools": [{"type": "web_search_20250305", "name": "web_search"}]}, "tools[0]"),
        ({"tools": [{"input_schema": {}}]}, "tools[0]"),
        ({"stop_sequences": "END"}, "stop_sequences"),
        ({"system": [{"type": "image"}]}, "system"),
        (
            {
                "tools": [{"name": "t", "input_schema": {}}],
                "tool_choice": {"type": "tool"},
            },
            "tool_choice",
        ),
    ],
)
def test_invalid_requests_name_the_offending_parameter(patch: dict[str, Any], param: str) -> None:
    body = copy.deepcopy(MESSAGE)
    body.update(patch)
    if patch.get("max_tokens", 1) is None:
        body.pop("max_tokens")
    with pytest.raises(BadRequestError) as caught:
        anthropic_to_chat(body)
    assert caught.value.param == param


# ---------------------------------------------------------------------------
# Response translation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("finish", "tool", "word", "expected"),
    [
        ("stop", False, None, ("end_turn", None)),
        ("length", False, None, ("max_tokens", None)),
        ("tool_calls", True, None, ("tool_use", None)),
        ("stop", True, None, ("tool_use", None)),
        ("stop", False, "END", ("stop_sequence", "END")),
        ("stop", False, "OTHER", ("end_turn", None)),
        (None, False, None, ("end_turn", None)),
    ],
)
def test_stop_reason_mapping(
    finish: str | None, tool: bool, word: str | None, expected: tuple[str, str | None]
) -> None:
    assert (
        stop_reason(finish, has_tool=tool, stopping_word=word, stop_sequences=["END"]) == expected
    )


def test_chat_to_anthropic_text_usage_and_tool_use() -> None:
    plain = chat_to_anthropic(openai_reply(), model=MODEL_ID, context={})
    assert plain["content"] == [{"type": "text", "text": "Hello there."}]
    assert plain["usage"] == {"input_tokens": 11, "output_tokens": 5}
    assert plain["stop_reason"] == "end_turn"
    assert plain["role"] == "assistant"
    assert plain["id"].startswith("msg_")

    calls = [
        {
            "id": "call_9",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
        }
    ]
    used = chat_to_anthropic(
        openai_reply("Checking.", finish="tool_calls", tool_calls=calls), model=MODEL_ID, context={}
    )
    assert used["stop_reason"] == "tool_use"
    assert used["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "call_9", "name": "get_weather", "input": {"city": "Oslo"}},
    ]
    broken = chat_to_anthropic(
        openai_reply(
            None,
            finish="tool_calls",
            tool_calls=[{**calls[0], "function": {"name": "f", "arguments": "{not json"}}],
        ),
        model=MODEL_ID,
        context={},
    )
    assert broken["content"][0]["input"] == {}


def test_thinking_block_only_when_requested() -> None:
    reply = openai_reply("answer")
    reply["choices"][0]["message"]["reasoning_content"] = "let me think"
    off = chat_to_anthropic(reply, model="m", context={"thinking": False})
    on = chat_to_anthropic(reply, model="m", context={"thinking": True})
    assert [b["type"] for b in off["content"]] == ["text"]
    assert [b["type"] for b in on["content"]] == ["thinking", "text"]
    assert on["content"][0]["thinking"] == "let me think"


def test_an_empty_reply_still_has_one_content_block() -> None:
    out = chat_to_anthropic(openai_reply(""), model="m", context={})
    assert out["content"] == [{"type": "text", "text": ""}]


# ---------------------------------------------------------------------------
# The route, non-streamed
# ---------------------------------------------------------------------------


def test_messages_round_trip_matches_the_fixture_shape(app: Any, upstream: Upstream) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/messages", json=load_fixture("anthropic_messages_request.json"))
    assert response.status_code == 200, response.text
    assert_shape(response.json(), load_fixture("anthropic_messages_response.json"))
    assert response.json()["content"][0]["text"] == "Hello there."
    sent = upstream.last
    assert sent["model"] == MODEL_ID
    assert sent["messages"][0] == {"role": "system", "content": "You are terse."}
    assert sent["max_tokens"] == 256
    assert sent["stop"] == ["END"]
    assert "stream_options" not in sent


def test_the_tool_use_round_trip_through_the_route(app: Any, upstream: Upstream) -> None:
    """assistant tool_use block (turn 1 reply) -> user tool_result block (turn 2 request)."""
    calls = [
        {
            "id": "call_7",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Oslo"}'},
        }
    ]
    upstream.reply = openai_reply(None, finish="tool_calls", tool_calls=calls)
    tools = [
        {
            "name": "get_weather",
            "description": "Weather.",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    with TestClient(app) as http:
        first = http.post(
            "/v1/messages",
            json={
                **MESSAGE,
                "tools": tools,
                "messages": [{"role": "user", "content": "Weather in Oslo?"}],
            },
        )
        assert first.status_code == 200, first.text
        turn1 = first.json()
        assert turn1["stop_reason"] == "tool_use"
        tool_use = turn1["content"][0]
        assert tool_use == {
            "type": "tool_use",
            "id": "call_7",
            "name": "get_weather",
            "input": {"city": "Oslo"},
        }
        assert upstream.last["tools"][0]["function"]["name"] == "get_weather"

        upstream.reply = openai_reply("It is 4 C and raining in Oslo.")
        second = http.post(
            "/v1/messages",
            json={
                **MESSAGE,
                "tools": tools,
                "messages": [
                    {"role": "user", "content": "Weather in Oslo?"},
                    {"role": "assistant", "content": turn1["content"]},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use["id"],
                                "content": "4 C, light rain",
                            }
                        ],
                    },
                ],
            },
        )
    assert second.status_code == 200, second.text
    assert second.json()["stop_reason"] == "end_turn"
    assert second.json()["content"][0]["text"] == "It is 4 C and raining in Oslo."
    replay = upstream.last["messages"]
    assert replay[1]["tool_calls"][0]["id"] == "call_7"
    assert json.loads(replay[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Oslo"}
    assert replay[2] == {"role": "tool", "tool_call_id": "call_7", "content": "4 C, light rain"}


def test_a_length_stop_is_max_tokens(app: Any, upstream: Upstream) -> None:
    upstream.reply = openai_reply("trunc", finish="length")
    with TestClient(app) as http:
        body = http.post("/v1/messages", json=MESSAGE).json()
    assert body["stop_reason"] == "max_tokens"


def test_ttl_and_priority_are_consumed_not_forwarded(app: Any, upstream: Upstream) -> None:
    with TestClient(app) as http:
        response = http.post("/v1/messages", json={**MESSAGE, "ttl": 90, "priority": 1})
    assert response.status_code == 200, response.text
    assert "ttl" not in upstream.last
    assert "priority" not in upstream.last


def test_errors_use_the_anthropic_envelope_with_the_stable_code(
    app: Any, upstream: Upstream
) -> None:
    with TestClient(app) as http:
        missing = http.post(
            "/v1/messages", json={"model": MODEL_ID, "messages": MESSAGE["messages"]}
        )
        unknown = http.post("/v1/messages", json={**MESSAGE, "model": "nope/none"})
        bad_json = http.post("/v1/messages", content="not json")
        no_vision = http.post(
            "/v1/messages",
            json={
                **MESSAGE,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AA",
                                },
                            }
                        ],
                    }
                ],
            },
        )
    assert missing.status_code == 400
    assert missing.json()["type"] == "error"
    assert missing.json()["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in missing.json()["error"]["message"]
    assert unknown.status_code == 404
    assert unknown.json()["error"]["type"] == "not_found_error"
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert bad_json.status_code == 400
    assert no_vision.status_code == 400
    assert no_vision.json()["error"]["code"] == "model_not_multimodal"
    assert upstream.payloads == []


def test_a_busy_model_is_a_529_style_overloaded_error_with_retry_after(
    app: Any, upstream: Upstream
) -> None:
    from studioforge.errors import ModelBusyError

    def busy(*_a: Any, **_k: Any) -> None:
        raise ModelBusyError("loading", details={"retry_after_s": 7})

    app.state.manager.admission_check = busy
    with TestClient(app) as http:
        response = http.post("/v1/messages", json={**MESSAGE, "stream": True})
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "7"
    assert response.json()["error"]["type"] == "overloaded_error"


def test_both_key_headers_are_honoured(app: Any, upstream: Upstream) -> None:
    app.state.config.server.api_key = "sk-test"
    with TestClient(app) as http:
        anonymous = http.post("/v1/messages", json=MESSAGE)
        wrong = http.post("/v1/messages", json=MESSAGE, headers={"x-api-key": "nope"})
        by_x_api_key = http.post("/v1/messages", json=MESSAGE, headers={"x-api-key": "sk-test"})
        by_bearer = http.post(
            "/v1/messages", json=MESSAGE, headers={"Authorization": "Bearer sk-test"}
        )
        sdk_style = http.post(
            "/v1/messages",
            json=MESSAGE,
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
        )
    assert anonymous.status_code == 401
    assert wrong.status_code == 401
    assert by_x_api_key.status_code == 200
    assert by_bearer.status_code == 200
    assert sdk_style.status_code == 200


def test_a_vision_model_accepts_image_blocks(app: Any, upstream: Upstream) -> None:
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode()
    app.state.supervisor._instances.append(
        app.state.supervisor.get(MODEL_ID).model_copy(update={"model_id": VISION_ID})
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/messages",
            json={
                "model": VISION_ID,
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "colour?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": data,
                                },
                            },
                        ],
                    }
                ],
            },
        )
    assert response.status_code == 200, response.text
    parts = upstream.last["messages"][0]["content"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _text_stream() -> list[bytes]:
    return sse(
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop"),
        chunk(usage=(11, 2)) | {"choices": []},
        "[DONE]",
    )


def _post_stream(
    app: Any, body: dict[str, Any] | None = None, *, http: TestClient | None = None
) -> tuple[int, str]:
    payload = {**MESSAGE, "stream": True, **(body or {})}
    if http is not None:
        response = http.post("/v1/messages", json=payload)
    else:
        with TestClient(app) as client:
            response = client.post("/v1/messages", json=payload)
    return response.status_code, response.text


def test_text_stream_event_sequence(app: Any, upstream: Upstream) -> None:
    upstream.chunks = _text_stream()
    status, text = _post_stream(app)
    assert status == 200
    events = events_of(text)
    assert [name for name, _ in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    start = events[0][1]["message"]
    assert start["role"] == "assistant" and start["content"] == [] and start["stop_reason"] is None
    assert events[1][1] == {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    deltas = [d["delta"]["text"] for n, d in events if n == "content_block_delta"]
    assert deltas == ["Hel", "lo"]
    final = events[5][1]
    assert final["delta"] == {"stop_reason": "end_turn", "stop_sequence": None}
    assert final["usage"] == {"input_tokens": 11, "output_tokens": 2}
    assert upstream.last["stream"] is True
    assert upstream.last["stream_options"] == {"include_usage": True}


def test_stream_survives_arbitrary_chunk_boundaries(app: Any, upstream: Upstream) -> None:
    whole = b"".join(_text_stream())
    reference = None
    with TestClient(app) as http:
        for size in (1, 2, 3, 7, 50):
            upstream.chunks = [whole[i : i + size] for i in range(0, len(whole), size)]
            _, text = _post_stream(app, http=http)
            names = [(n, d.get("delta")) for n, d in events_of(text) if n != "message_start"]
            if reference is None:
                reference = names
            assert names == reference, f"chunk size {size} changed the translation"


def test_stream_with_a_multibyte_character_split_across_chunks(
    app: Any, upstream: Upstream
) -> None:
    whole = b"".join(sse(chunk({"content": "café \U0001f600"}), chunk({}, finish="stop"), "[DONE]"))
    upstream.chunks = [bytes([b]) for b in whole]
    _, text = _post_stream(app)
    deltas = [d["delta"]["text"] for n, d in events_of(text) if n == "content_block_delta"]
    assert "".join(deltas) == "café \U0001f600"


def test_tool_use_stream(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"role": "assistant", "content": "Checking. "}),
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "Oslo"}'}}]}),
        chunk({}, finish="tool_calls", usage=(20, 9)),
        "[DONE]",
    )
    status, text = _post_stream(
        app, {"tools": [{"name": "get_weather", "input_schema": {"type": "object"}}]}
    )
    assert status == 200
    events = events_of(text)
    names = [n for n, _ in events]
    assert names == [
        "message_start",
        "content_block_start",  # text
        "content_block_delta",
        "content_block_stop",
        "content_block_start",  # tool_use
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    tool_start = events[4][1]
    assert tool_start["index"] == 1
    assert tool_start["content_block"] == {
        "type": "tool_use",
        "id": "call_1",
        "name": "get_weather",
        "input": {},
    }
    fragments = [
        d["delta"]["partial_json"]
        for n, d in events
        if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    ]
    assert json.loads("".join(fragments)) == {"city": "Oslo"}
    delta = events[-2][1]
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"] == {"input_tokens": 20, "output_tokens": 9}


def test_length_stop_and_usage_fallback_when_the_engine_reports_none(
    app: Any, upstream: Upstream
) -> None:
    upstream.chunks = sse(
        chunk({"content": "a"}), chunk({"content": "b"}), chunk({}, finish="length"), "[DONE]"
    )
    _, text = _post_stream(app)
    final = events_of(text)[-2][1]
    assert final["delta"]["stop_reason"] == "max_tokens"
    assert final["usage"]["output_tokens"] == 2, "falls back to the number of deltas seen"


def test_stop_sequence_reported_when_the_engine_names_the_word(
    app: Any, upstream: Upstream
) -> None:
    upstream.chunks = sse(
        chunk({"content": "ok"}), chunk({}, finish="stop", stopping_word="END"), "[DONE]"
    )
    _, text = _post_stream(app, {"stop_sequences": ["END"]})
    assert events_of(text)[-2][1]["delta"] == {
        "stop_reason": "stop_sequence",
        "stop_sequence": "END",
    }


def test_keepalive_comments_become_pings(app: Any, upstream: Upstream) -> None:
    upstream.chunks = [b": loading vendor/thing (2s)\n\n", *_text_stream()]
    _, text = _post_stream(app)
    names = [n for n, _ in events_of(text)]
    assert names[:2] == ["message_start", "ping"]


def test_thinking_stream_only_when_enabled(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"reasoning_content": "hmm "}),
        chunk({"reasoning_content": "ok"}),
        chunk({"content": "Answer"}),
        chunk({}, finish="stop"),
        "[DONE]",
    )
    with TestClient(app) as http:
        _, off = _post_stream(app, http=http)
        thinking = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
        _, on = _post_stream(app, thinking, http=http)
    kinds_off = [
        d["content_block"]["type"] for n, d in events_of(off) if n == "content_block_start"
    ]
    assert kinds_off == ["text"]
    events = events_of(on)
    kinds_on = [d["content_block"]["type"] for n, d in events if n == "content_block_start"]
    assert kinds_on == ["thinking", "text"]
    thoughts = [
        d["delta"]["thinking"]
        for n, d in events
        if d.get("delta", {}).get("type") == "thinking_delta"
    ]
    assert "".join(thoughts) == "hmm ok"


def test_a_mid_stream_engine_error_becomes_an_error_event(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"content": "par"}),
        {
            "error": {
                "message": "engine fell over",
                "type": "server_error",
                "code": "upstream_error",
            }
        },
        "[DONE]",
    )
    _, text = _post_stream(app)
    events = events_of(text)
    assert events[-1][0] == "error"
    assert events[-1][1]["error"]["message"] == "engine fell over"
    assert events[-1][1]["error"]["code"] == "upstream_error"
    assert "message_stop" not in [n for n, _ in events]


def test_the_stream_is_served_as_sse(app: Any, upstream: Upstream) -> None:
    upstream.chunks = _text_stream()
    with TestClient(app) as http:
        response = http.post("/v1/messages", json={**MESSAGE, "stream": True})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
