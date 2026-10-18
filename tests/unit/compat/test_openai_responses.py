"""OpenAI Responses API ``POST /v1/responses`` -- W2 (stateless subset)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.responses_routes import chat_to_response, responses_to_chat
from studioforge.errors import BadRequestError
from tests.unit.compat.conftest import (
    Upstream,
    assert_shape,
    chunk,
    load_fixture,
    openai_reply,
    sse,
)
from tests.unit.test_catalog_routes import MODEL_ID

BODY = {"model": MODEL_ID, "input": "hi"}


def events_of(body: str) -> list[dict[str, Any]]:
    out = []
    for block in body.strip().split("\n\n"):
        if block.startswith(":"):
            continue
        lines = block.split("\n")
        name = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        assert data["type"] == name
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def test_string_input_and_instructions() -> None:
    chat = responses_to_chat({**BODY, "instructions": "Be brief.", "max_output_tokens": 50})
    assert chat["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hi"},
    ]
    assert chat["max_tokens"] == 50


def test_the_fixture_with_function_call_items_translates() -> None:
    chat = responses_to_chat(load_fixture("responses_request.json"))
    assert chat["messages"] == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Oslo?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "4 C, light rain"},
    ]
    assert chat["tools"][0]["function"]["name"] == "get_weather"


def test_roles_parts_and_images() -> None:
    chat = responses_to_chat(
        {
            "model": "m",
            "input": [
                {"role": "developer", "content": "rules"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "earlier"}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
                    ],
                },
                {"type": "reasoning", "summary": []},
            ],
        }
    )
    assert chat["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "earlier"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        },
    ]


def test_consecutive_function_calls_share_one_assistant_message() -> None:
    chat = responses_to_chat(
        {
            "model": "m",
            "input": [
                {"role": "user", "content": "go"},
                {"type": "function_call", "call_id": "a", "name": "f", "arguments": "{}"},
                {"type": "function_call", "call_id": "b", "name": "g", "arguments": "{}"},
            ],
        }
    )
    assert [m["role"] for m in chat["messages"]] == ["user", "assistant"]
    assert [c["id"] for c in chat["messages"][1]["tool_calls"]] == ["a", "b"]


def test_text_format_and_tool_choice_mapping() -> None:
    chat = responses_to_chat(
        {
            **BODY,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "n",
                    "schema": {"type": "object"},
                    "strict": True,
                }
            },
            "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
            "tool_choice": {"type": "function", "name": "f"},
            "parallel_tool_calls": False,
        }
    )
    assert chat["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "n", "schema": {"type": "object"}, "strict": True},
    }
    assert chat["tool_choice"] == {"type": "function", "function": {"name": "f"}}
    assert chat["parallel_tool_calls"] is False
    assert responses_to_chat({**BODY, "text": {"format": {"type": "json_object"}}})[
        "response_format"
    ] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("patch", "param", "code"),
    [
        ({"previous_response_id": "resp_1"}, "previous_response_id", "unsupported_parameter"),
        ({"background": True}, "background", "unsupported_parameter"),
        ({"tools": [{"type": "web_search"}]}, "tools[0]", "unsupported_parameter"),
        ({"tools": [{"type": "function"}]}, "tools[0]", None),
        ({"input": [{"type": "item_reference", "id": "x"}]}, "input[0]", "unsupported_parameter"),
        (
            {"input": [{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}]},
            "input[0].content[0]",
            "unsupported_parameter",
        ),
        ({"input": []}, "input", None),
        ({"input": [{"type": "function_call_output"}]}, "input[0]", None),
        ({"text": {"format": {"type": "weird"}}}, "text", None),
    ],
)
def test_refusals_name_the_parameter(patch: dict[str, Any], param: str, code: str | None) -> None:
    with pytest.raises(BadRequestError) as caught:
        responses_to_chat({**BODY, **patch})
    assert caught.value.param == param
    if code:
        assert caught.value.code == code


def test_input_is_required() -> None:
    with pytest.raises(BadRequestError) as caught:
        responses_to_chat({"model": "m"})
    assert caught.value.param == "input"


# ---------------------------------------------------------------------------
# Response translation + route
# ---------------------------------------------------------------------------


def test_chat_to_response_message_and_usage() -> None:
    out = chat_to_response(openai_reply(), BODY, model=MODEL_ID)
    assert out["object"] == "response" and out["status"] == "completed"
    assert out["output"][0]["type"] == "message"
    assert out["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "Hello there.",
        "annotations": [],
    }
    assert out["usage"]["input_tokens"] == 11
    assert out["usage"]["output_tokens"] == 5
    assert out["usage"]["total_tokens"] == 16


def test_chat_to_response_function_calls_and_truncation() -> None:
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"a":1}'}}]
    out = chat_to_response(
        openai_reply(None, finish="tool_calls", tool_calls=calls), BODY, model="m"
    )
    assert [i["type"] for i in out["output"]] == ["function_call"]
    assert out["output"][0]["call_id"] == "c1"
    assert out["output"][0]["arguments"] == '{"a":1}'
    cut = chat_to_response(openai_reply("x", finish="length"), BODY, model="m")
    assert cut["status"] == "incomplete"
    assert cut["incomplete_details"] == {"reason": "max_output_tokens"}


def test_route_non_stream_matches_the_fixture_shape(app: Any, upstream: Upstream) -> None:
    body = {
        "model": MODEL_ID,
        "instructions": "You are terse.",
        "input": "Hi",
        "max_output_tokens": 128,
    }
    with TestClient(app) as http:
        response = http.post("/v1/responses", json=body)
    assert response.status_code == 200, response.text
    assert_shape(response.json(), load_fixture("responses_response.json"))
    assert response.json()["model"] == MODEL_ID
    assert upstream.last["messages"][0] == {"role": "system", "content": "You are terse."}
    assert upstream.last["max_tokens"] == 128


def test_function_call_round_trip_through_the_route(app: Any, upstream: Upstream) -> None:
    calls = [
        {
            "id": "call_5",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
        }
    ]
    upstream.reply = openai_reply(None, finish="tool_calls", tool_calls=calls)
    tool = {"type": "function", "name": "get_weather", "parameters": {"type": "object"}}
    with TestClient(app) as http:
        first = http.post(
            "/v1/responses", json={**BODY, "input": "weather?", "tools": [tool]}
        ).json()
        item = next(i for i in first["output"] if i["type"] == "function_call")
        assert item["name"] == "get_weather" and item["call_id"] == "call_5"
        upstream.reply = openai_reply("Rainy.")
        second = http.post(
            "/v1/responses",
            json={
                **BODY,
                "tools": [tool],
                "input": [
                    {"role": "user", "content": "weather?"},
                    {k: item[k] for k in ("type", "call_id", "name", "arguments")},
                    {"type": "function_call_output", "call_id": "call_5", "output": "4 C"},
                ],
            },
        )
    assert second.status_code == 200, second.text
    assert second.json()["output"][0]["content"][0]["text"] == "Rainy."
    assert upstream.last["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "call_5",
        "content": "4 C",
    }


def test_errors_stay_in_the_openai_envelope(app: Any, upstream: Upstream) -> None:
    with TestClient(app) as http:
        unsupported = http.post("/v1/responses", json={**BODY, "previous_response_id": "r"})
        unknown = http.post("/v1/responses", json={**BODY, "model": "nope/none"})
    assert unsupported.status_code == 400
    assert unsupported.json()["error"]["code"] == "unsupported_parameter"
    assert unsupported.json()["error"]["param"] == "previous_response_id"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"


def test_bearer_and_x_api_key_both_authenticate(app: Any, upstream: Upstream) -> None:
    app.state.config.server.api_key = "sk-test"
    with TestClient(app) as http:
        assert http.post("/v1/responses", json=BODY).status_code == 401
        ok = http.post("/v1/responses", json=BODY, headers={"Authorization": "Bearer sk-test"})
        alt = http.post("/v1/responses", json=BODY, headers={"x-api-key": "sk-test"})
    assert ok.status_code == 200 and alt.status_code == 200


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_text_stream_events(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop", usage=(9, 2)),
        "[DONE]",
    )
    with TestClient(app) as http:
        response = http.post("/v1/responses", json={**BODY, "stream": True})
    assert response.status_code == 200
    events = events_of(response.text)
    assert [e["type"] for e in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [e["sequence_number"] for e in events] == list(range(len(events)))
    assert [e["delta"] for e in events if e["type"] == "response.output_text.delta"] == [
        "Hel",
        "lo",
    ]
    done = next(e for e in events if e["type"] == "response.output_text.done")
    assert done["text"] == "Hello"
    created = events[0]["response"]
    assert created["status"] == "in_progress" and created["output"] == []
    final = events[-1]["response"]
    assert final["status"] == "completed"
    assert final["output"][0]["content"][0]["text"] == "Hello"
    assert final["usage"]["input_tokens"] == 9 and final["usage"]["output_tokens"] == 2
    item_ids = {e["item_id"] for e in events if "item_id" in e}
    assert item_ids == {final["output"][0]["id"]}


def test_function_call_stream_events(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": ""},
                    }
                ]
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"a":'}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}),
        chunk({}, finish="tool_calls"),
        "[DONE]",
    )
    with TestClient(app) as http:
        response = http.post(
            "/v1/responses",
            json={
                **BODY,
                "stream": True,
                "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}],
            },
        )
    events = events_of(response.text)
    kinds = [e["type"] for e in events]
    assert kinds == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    added = events[2]["item"]
    assert added["type"] == "function_call" and added["call_id"] == "call_1"
    assert events[5]["arguments"] == '{"a":1}'
    assert events[-1]["response"]["output"][0]["arguments"] == '{"a":1}'


def test_text_then_call_gets_two_output_items(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"content": "Let me check."}),
        chunk(
            {"tool_calls": [{"index": 0, "id": "c", "function": {"name": "f", "arguments": "{}"}}]}
        ),
        chunk({}, finish="tool_calls"),
        "[DONE]",
    )
    with TestClient(app) as http:
        response = http.post("/v1/responses", json={**BODY, "stream": True})
    final = events_of(response.text)[-1]["response"]
    assert [i["type"] for i in final["output"]] == ["message", "function_call"]
    added = [e for e in events_of(response.text) if e["type"] == "response.output_item.added"]
    assert [e["output_index"] for e in added] == [0, 1]


def test_empty_and_truncated_streams(app: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(chunk({}, finish="stop"), "[DONE]")
    with TestClient(app) as http:
        empty = events_of(http.post("/v1/responses", json={**BODY, "stream": True}).text)
        upstream.chunks = sse(chunk({"content": "x"}), chunk({}, finish="length"), "[DONE]")
        cut = events_of(http.post("/v1/responses", json={**BODY, "stream": True}).text)
    assert empty[-1]["type"] == "response.completed"
    assert empty[-1]["response"]["output"][0]["content"][0]["text"] == ""
    assert cut[-1]["type"] == "response.incomplete"
    assert cut[-1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}


def test_stream_error_and_keepalive(app: Any, upstream: Upstream) -> None:
    upstream.chunks = [
        b": loading m (2s)\n\n",
        *sse(
            chunk({"content": "a"}),
            {"error": {"message": "boom", "type": "server_error", "code": "upstream_error"}},
            "[DONE]",
        ),
    ]
    with TestClient(app) as http:
        response = http.post("/v1/responses", json={**BODY, "stream": True})
    assert ": keep-alive" in response.text
    last = events_of(response.text)[-1]
    assert last["type"] == "error" and last["code"] == "upstream_error"
    assert last["message"] == "boom"
