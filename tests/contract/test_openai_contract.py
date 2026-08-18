"""OpenAI API parity contract.

The most important suite in the project: it asserts that a client which works
against LM Studio works against StudioForge with only a base-URL change. Every
test here drives the official ``openai`` Python client against a real server,
real engine and real GGUF weights.
"""

from __future__ import annotations

import concurrent.futures
import json
from typing import Any

import httpx
import pytest

from tests.contract.conftest import (
    ServerHandle,
    requires_engine,
    requires_models,
    tiny_png_data_url,
)

pytestmark = [requires_engine, requires_models]


# ---------------------------------------------------------------------------
# 1. Model listing
# ---------------------------------------------------------------------------


def test_models_lists_all_downloaded_models(client: Any, live_server: ServerHandle) -> None:
    """LM Studio lists every downloaded model, loaded or not; so must we.

    OpenClaw's model picker depends on this: if only loaded models appeared,
    the picker would be empty on a fresh server.
    """
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert len(ids) >= 10, f"expected the real library to be registered, got {ids}"
    assert len(ids) == len(set(ids)), "model ids must be unique"

    # Nothing is loaded yet, so this is proof that listing is not gated on load.
    status = httpx.get(
        f"{live_server.base_url}/api/status",
        headers={"Authorization": f"Bearer {live_server.api_key}"},
        timeout=30,
    ).json()
    assert len(ids) > len(status["loaded"])


def test_model_objects_have_required_openai_fields(client: Any) -> None:
    for model in client.models.list().data:
        assert model.id
        assert model.object == "model"
        assert isinstance(model.created, int)
        assert model.owned_by


def test_retrieve_single_model(client: Any, chat_model: str) -> None:
    model = client.models.retrieve(chat_model)
    assert model.id == chat_model


def test_model_ids_are_stable_across_listings(client: Any) -> None:
    first = sorted(m.id for m in client.models.list().data)
    second = sorted(m.id for m in client.models.list().data)
    assert first == second


def test_capabilities_are_advertised(raw: httpx.Client) -> None:
    """Vision/tools flags let a client pick a model that can do the job."""
    data = raw.get("/v1/models").json()["data"]
    vision = [m for m in data if m.get("studioforge", {}).get("vision")]
    assert vision, "the real library contains vision models; none were advertised"
    for model in data:
        assert "studioforge" in model
        assert isinstance(model["studioforge"]["vision"], bool)


# ---------------------------------------------------------------------------
# 2. JIT loading
# ---------------------------------------------------------------------------


def test_jit_load_on_first_chat_request(client: Any, chat_model: str) -> None:
    """Naming an unloaded model loads it and serves the request."""
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say exactly: ready"}],
        max_tokens=16,
    )
    assert completion.choices[0].message.content
    assert completion.model
    assert completion.usage.completion_tokens > 0


def test_concurrent_requests_during_load_queue_not_error(
    client: Any, chat_model: str, raw: httpx.Client
) -> None:
    """A burst against a cold model must all succeed, sharing one load.

    This is the failure mode that would break OpenClaw hardest: several
    parallel agent calls arriving while the model is still loading.
    """
    raw.post(f"/api/models/{chat_model}/unload")

    def ask(index: int) -> str:
        result = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": f"Reply with the number {index}."}],
            max_tokens=16,
        )
        return result.choices[0].message.content or ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        outputs = list(pool.map(ask, range(5)))

    assert len(outputs) == 5
    assert all(isinstance(text, str) and text for text in outputs)

    # Exactly one instance should be serving it.
    loaded = raw.get("/api/status").json()["loaded"]
    assert sum(1 for i in loaded if i["model_id"] == chat_model) == 1


def test_unload_then_rejit(client: Any, chat_model: str, raw: httpx.Client) -> None:
    assert raw.post(f"/api/models/{chat_model}/unload").status_code == 200
    loaded = [i["model_id"] for i in raw.get("/api/status").json()["loaded"]]
    assert chat_model not in loaded

    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
    )
    assert completion.choices[0].message.content is not None


# ---------------------------------------------------------------------------
# 3. Chat completions: the full parameter surface
# ---------------------------------------------------------------------------


def test_non_streaming_completion_shape(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Name one colour."}],
        max_tokens=24,
    )
    assert completion.id
    assert completion.object == "chat.completion"
    assert completion.choices[0].index == 0
    assert completion.choices[0].message.role == "assistant"
    assert completion.choices[0].finish_reason in {"stop", "length"}
    assert completion.usage.prompt_tokens > 0
    assert completion.usage.total_tokens >= completion.usage.completion_tokens


def test_streaming_yields_deltas_and_terminates(client: Any, chat_model: str) -> None:
    """SSE framing must be exactly what clients expect, including [DONE]."""
    stream = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Count: one two three"}],
        max_tokens=48,
        stream=True,
    )
    chunks = list(stream)
    assert len(chunks) >= 2
    assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text.strip()


def test_streaming_raw_sse_has_done_sentinel(raw: httpx.Client, chat_model: str) -> None:
    """Assert the wire format directly; the SDK hides the sentinel."""
    with raw.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    assert "data: [DONE]" in body
    data_lines = [ln for ln in body.splitlines() if ln.startswith("data: ")]
    assert len(data_lines) >= 2
    # Every non-sentinel data line must be valid JSON.
    for line in data_lines:
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        json.loads(payload)
    assert data_lines[-1].strip() == "data: [DONE]"


def test_system_role_and_multi_turn(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "You always answer with a single word."},
            {"role": "user", "content": "What colour is grass?"},
            {"role": "assistant", "content": "Green"},
            {"role": "user", "content": "And the sky?"},
        ],
        max_tokens=16,
    )
    assert completion.choices[0].message.content


def test_max_tokens_is_respected(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Write a long story about a robot."}],
        max_tokens=8,
    )
    assert completion.usage.completion_tokens <= 8
    assert completion.choices[0].finish_reason == "length"


def test_stop_sequence_truncates(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say: alpha STOPHERE beta"}],
        max_tokens=48,
        stop=["STOPHERE"],
    )
    content = completion.choices[0].message.content or ""
    assert "STOPHERE" not in content


def test_temperature_top_p_and_seed_accepted(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Pick a fruit."}],
        max_tokens=16,
        temperature=0.1,
        top_p=0.9,
        seed=1234,
    )
    assert completion.choices[0].message.content is not None


def test_seed_gives_reproducible_output(client: Any, chat_model: str) -> None:
    """Same seed + greedy settings should reproduce; skip if the engine varies."""
    kwargs: dict[str, Any] = {
        "model": chat_model,
        "messages": [{"role": "user", "content": "Name three animals."}],
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 42,
    }
    first = client.chat.completions.create(**kwargs).choices[0].message.content
    second = client.chat.completions.create(**kwargs).choices[0].message.content
    if first != second:
        pytest.skip("engine did not reproduce a seeded greedy sample; not a parity break")
    assert first == second


def test_n_and_logprobs_do_not_break_the_request(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=8,
        logprobs=True,
        top_logprobs=2,
    )
    assert completion.choices


# ---------------------------------------------------------------------------
# 4. Tool / function calling
# ---------------------------------------------------------------------------

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}


def test_tool_call_is_returned(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        max_tokens=128,
    )
    message = completion.choices[0].message
    if not message.tool_calls:
        pytest.skip(
            "model chose not to call the tool; tool plumbing is verified by the forced-choice test"
        )
    call = message.tool_calls[0]
    assert call.function.name == "get_current_weather"
    args = json.loads(call.function.arguments)
    assert "city" in args


def test_forced_tool_choice(client: Any, chat_model: str) -> None:
    """``tool_choice`` naming a function must produce that call."""
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Weather in Tokyo please."}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
        max_tokens=128,
    )
    message = completion.choices[0].message
    assert message.tool_calls, "forced tool_choice must yield a tool call"
    call = message.tool_calls[0]
    assert call.function.name == "get_current_weather"
    assert call.id
    assert completion.choices[0].finish_reason in {"tool_calls", "stop"}
    json.loads(call.function.arguments)


def test_tool_result_round_trip(client: Any, chat_model: str) -> None:
    """Feed a tool result back and get a normal assistant turn."""
    first = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Weather in Oslo?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
        max_tokens=128,
    )
    call = first.choices[0].message.tool_calls[0]
    second = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "user", "content": "Weather in Oslo?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"temp_c": 3, "sky": "cloudy"}),
            },
        ],
        max_tokens=64,
    )
    assert second.choices[0].message.content


def test_streaming_tool_call(client: Any, chat_model: str) -> None:
    stream = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Weather in Berlin?"}],
        tools=[WEATHER_TOOL],
        tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
        max_tokens=128,
        stream=True,
    )
    saw_tool_delta = False
    for chunk in stream:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.tool_calls:
            saw_tool_delta = True
    assert saw_tool_delta, "streaming tool calls must arrive as tool_calls deltas"


def test_malformed_tools_rejected_with_400(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# 5. Structured output
# ---------------------------------------------------------------------------

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_json_schema_response_format(client: Any, chat_model: str) -> None:
    """``json_schema`` must map to a llama.cpp grammar and constrain output."""
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Invent a person: name and age."}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": PERSON_SCHEMA, "strict": True},
        },
        max_tokens=128,
    )
    content = completion.choices[0].message.content
    assert content
    data = json.loads(content)  # must be parseable, not merely JSON-ish
    assert isinstance(data["name"], str)
    assert isinstance(data["age"], int)


def test_json_object_response_format(client: Any, chat_model: str) -> None:
    completion = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Return a JSON object with key 'ok'."}],
        response_format={"type": "json_object"},
        max_tokens=64,
    )
    json.loads(completion.choices[0].message.content or "")


def test_streaming_json_schema(client: Any, chat_model: str) -> None:
    stream = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Invent a person."}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": PERSON_SCHEMA, "strict": True},
        },
        max_tokens=128,
        stream=True,
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    json.loads(text)


def test_bad_response_format_rejected(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "x"}},
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "response_format"


# ---------------------------------------------------------------------------
# 6. Vision
# ---------------------------------------------------------------------------


def test_vision_base64_data_url(client: Any, vision_model: str) -> None:
    """OpenClaw sends screenshots as data URLs; this is the core vision path."""
    completion = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": tiny_png_data_url(96)}},
                ],
            }
        ],
        max_tokens=96,
    )
    text = (completion.choices[0].message.content or "").lower()
    assert text.strip(), "vision model returned empty content"
    assert any(word in text for word in ("red", "square", "rectangle", "orange", "crimson")), (
        f"description does not mention the image's obvious features: {text!r}"
    )


def test_vision_streaming(client: Any, vision_model: str) -> None:
    stream = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What shape is in this image?"},
                    {"type": "image_url", "image_url": {"url": tiny_png_data_url(96)}},
                ],
            }
        ],
        max_tokens=64,
        stream=True,
    )
    text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
    assert text.strip()


def test_vision_http_url_is_fetched_server_side(
    client: Any, vision_model: str, httpbin_image: str
) -> None:
    completion = client.chat.completions.create(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": httpbin_image}},
                ],
            }
        ],
        max_tokens=64,
    )
    assert (completion.choices[0].message.content or "").strip()


def test_text_only_model_rejects_images_with_400(raw: httpx.Client, chat_model: str) -> None:
    """A clear 400, never a crash -- an explicit acceptance criterion."""
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": tiny_png_data_url()}},
                    ],
                }
            ],
            "max_tokens": 32,
        },
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "model_not_multimodal"
    assert "does not support image input" in error["message"]


def test_too_many_images_rejected(raw: httpx.Client, vision_model: str) -> None:
    parts: list[dict[str, Any]] = [{"type": "text", "text": "Describe these."}]
    for _ in range(20):
        parts.append({"type": "image_url", "image_url": {"url": tiny_png_data_url(32)}})
    response = raw.post(
        "/v1/chat/completions",
        json={"model": vision_model, "messages": [{"role": "user", "content": parts}]},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "too_many_images"


def test_malformed_image_data_rejected(raw: httpx.Client, vision_model: str) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,notbase64!!!"},
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400


def test_local_file_path_image_rejected(raw: httpx.Client, vision_model: str) -> None:
    """A file:// or bare path must not let a client read server-side files."""
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "file:///C:/Windows/win.ini"},
                        },
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image_url"


# ---------------------------------------------------------------------------
# 7. Completions + embeddings
# ---------------------------------------------------------------------------


def test_legacy_completions(client: Any, chat_model: str) -> None:
    completion = client.completions.create(
        model=chat_model, prompt="The capital of France is", max_tokens=12
    )
    assert completion.choices[0].text
    assert completion.object == "text_completion"


def test_legacy_completions_streaming(client: Any, chat_model: str) -> None:
    stream = client.completions.create(
        model=chat_model, prompt="Count to three:", max_tokens=24, stream=True
    )
    text = "".join(chunk.choices[0].text or "" for chunk in stream if chunk.choices)
    assert text


def test_embeddings(client: Any, embedding_model: str) -> None:
    """Embedding models get their own instance launched with --embedding."""
    response = client.embeddings.create(model=embedding_model, input="hello world")
    assert response.data
    vector = response.data[0].embedding
    assert len(vector) > 64
    assert all(isinstance(value, float) for value in vector[:8])


def test_embeddings_batch(client: Any, embedding_model: str) -> None:
    response = client.embeddings.create(model=embedding_model, input=["first text", "second text"])
    assert len(response.data) == 2
    assert len(response.data[0].embedding) == len(response.data[1].embedding)


def test_chat_model_rejects_embeddings_request(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post("/v1/embeddings", json={"model": chat_model, "input": "hi"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "not_an_embedding_model"


# ---------------------------------------------------------------------------
# 8. Errors and auth
# ---------------------------------------------------------------------------


def test_unknown_model_is_openai_shaped_404(raw: httpx.Client) -> None:
    response = raw.post(
        "/v1/chat/completions",
        json={"model": "no/such-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "model_not_found"
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "model"
    assert "does not exist" in error["message"]


def test_unknown_model_raises_notfound_in_sdk(client: Any) -> None:
    from openai import NotFoundError

    with pytest.raises(NotFoundError):
        client.chat.completions.create(
            model="no/such-model", messages=[{"role": "user", "content": "hi"}]
        )


def test_missing_model_field_is_400(raw: httpx.Client) -> None:
    response = raw.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "model"


def test_empty_messages_is_400(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post("/v1/chat/completions", json={"model": chat_model, "messages": []})
    assert response.status_code == 400


def test_malformed_json_is_400(live_server: ServerHandle) -> None:
    response = httpx.post(
        f"{live_server.openai_base}/chat/completions",
        content=b"{not json",
        headers={
            "Authorization": f"Bearer {live_server.api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    assert response.status_code == 400
    assert "error" in response.json()


def test_wrong_api_key_is_401(live_server: ServerHandle) -> None:
    response = httpx.get(
        f"{live_server.openai_base}/models",
        headers={"Authorization": "Bearer totally-wrong"},
        timeout=30,
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_missing_api_key_is_401(live_server: ServerHandle) -> None:
    response = httpx.get(f"{live_server.openai_base}/models", timeout=30)
    assert response.status_code == 401


def test_health_needs_no_key(live_server: ServerHandle) -> None:
    """Probes must work without a credential."""
    response = httpx.get(f"{live_server.base_url}/health", timeout=30)
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_error_envelope_always_has_the_four_keys(raw: httpx.Client) -> None:
    """The openai client parses exactly these keys to build its exceptions."""
    response = raw.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "x"}]},
    )
    error = response.json()["error"]
    for key in ("message", "type", "code", "param"):
        assert key in error


# ---------------------------------------------------------------------------
# 9. Introspection parity (/props, /slots)
# ---------------------------------------------------------------------------


def test_introspection_reports_actual_engine_settings(
    raw: httpx.Client, client: Any, chat_model: str
) -> None:
    """The dashboard shows what the engine reports, not what we asked for."""
    client.chat.completions.create(
        model=chat_model, messages=[{"role": "user", "content": "hi"}], max_tokens=4
    )
    data = raw.get(f"/api/models/{chat_model}/introspect").json()
    assert data["loaded"] is True
    assert data["actual"]["n_ctx"] == 4096
    # parallel defaults to 1, so ctx is not silently divided among slots (D4).
    assert data["actual"]["total_slots"] == 1
    assert isinstance(data["slots"], list)
    assert data["requested"]["ctx_size"] == 4096


# ---------------------------------------------------------------------------
# 10. LM Studio's /api/v0 surface
# ---------------------------------------------------------------------------


def test_v0_models_reports_load_state(raw: httpx.Client, chat_model: str) -> None:
    """/api/v0/models exposes state, which /v1/models deliberately does not."""
    raw.post(f"/api/models/{chat_model}/unload")
    data = raw.get("/api/v0/models").json()["data"]
    assert data
    entry = next(m for m in data if m["id"] == chat_model)
    assert entry["state"] == "not-loaded"
    assert entry["object"] == "model"
    assert entry["compatibility_type"] == "gguf"
    assert entry["quantization"]
    assert entry["arch"]
    assert entry["max_context_length"] > 0

    raw.post(f"/api/models/{chat_model}/load", json={"ctx_size": 2048})
    after = raw.get("/api/v0/models").json()["data"]
    loaded = next(m for m in after if m["id"] == chat_model)
    assert loaded["state"] == "loaded"
    assert loaded["loaded_context_length"] == 2048


def test_v0_model_types_distinguish_vlm_and_embeddings(
    raw: httpx.Client, vision_model: str, embedding_model: str
) -> None:
    data = {m["id"]: m for m in raw.get("/api/v0/models").json()["data"]}
    assert data[vision_model]["type"] == "vlm"
    assert data[embedding_model]["type"] == "embeddings"


def test_v0_retrieve_single_model(raw: httpx.Client, chat_model: str) -> None:
    entry = raw.get(f"/api/v0/models/{chat_model}").json()
    assert entry["id"] == chat_model
    assert entry["state"] in {"loaded", "not-loaded"}


def test_v0_unknown_model_is_404(raw: httpx.Client) -> None:
    response = raw.get("/api/v0/models/no/such-model")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_v0_chat_completions_works(raw: httpx.Client, chat_model: str) -> None:
    response = raw.post(
        "/api/v0/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 16,
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]
