"""The official ``anthropic`` and ``openai`` SDKs against the in-process app.

The wire-format unit tests prove the JSON; these prove that the real client
libraries (whose stream accumulators are strict about event order and field
names) accept it. Transport is ``httpx.ASGITransport`` straight into the
FastAPI app -- no socket, no port -- with the engine stubbed as in the rest of
this package. Skipped when an SDK is not installed (``openai`` is a dev
dependency; ``anthropic`` is optional here and required by ``tests/compat``).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from tests.unit.compat.conftest import Upstream, chunk, openai_reply, sse
from tests.unit.test_catalog_routes import MODEL_ID

anthropic = pytest.importorskip("anthropic")
openai = pytest.importorskip("openai")

BASE = "http://testserver"


def _anthropic_http_client(app: Any) -> Any:
    """An in-process async client of the flavour THIS anthropic SDK accepts.

    anthropic 1.x moved from ``httpx`` to ``httpx2`` and refuses the former's
    client objects; older releases want ``httpx``. Pick by what the SDK itself
    guards against.
    """
    if hasattr(anthropic._base_client, "_reject_httpx_object"):
        httpx2 = pytest.importorskip("httpx2")
        return httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app))


@pytest.fixture()
def transport(app: Any) -> httpx.ASGITransport:
    # No lifespan runs here, so there is no boot status to wait on.
    return httpx.ASGITransport(app=app)


@pytest.fixture()
def claude(app: Any) -> Any:
    return anthropic.AsyncAnthropic(
        base_url=BASE,
        api_key="sk-test",
        http_client=_anthropic_http_client(app),
        max_retries=0,
    )


@pytest.fixture()
def oai(transport: httpx.ASGITransport) -> Any:
    return openai.AsyncOpenAI(
        base_url=f"{BASE}/v1",
        api_key="sk-test",
        http_client=httpx.AsyncClient(transport=transport),
        max_retries=0,
    )


async def test_anthropic_create(claude: Any, upstream: Upstream) -> None:
    message = await claude.messages.create(
        model=MODEL_ID,
        max_tokens=64,
        system="Be terse.",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert message.content[0].text == "Hello there."
    assert message.stop_reason == "end_turn"
    assert (message.usage.input_tokens, message.usage.output_tokens) == (11, 5)
    assert upstream.last["messages"][0] == {"role": "system", "content": "Be terse."}


async def test_anthropic_stream(claude: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop"),
        chunk(usage=(11, 2)) | {"choices": []},
        "[DONE]",
    )
    async with claude.messages.stream(
        model=MODEL_ID, max_tokens=64, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        text = "".join([piece async for piece in stream.text_stream])
        final = await stream.get_final_message()
    assert text == "Hello"
    assert final.content[0].text == "Hello"
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 2


async def test_anthropic_tool_use_round_trip(claude: Any, upstream: Upstream) -> None:
    tool = {
        "name": "get_weather",
        "description": "Weather.",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    upstream.chunks = sse(
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
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Oslo"}'}}]}),
        chunk({}, finish="tool_calls", usage=(30, 12)),
        "[DONE]",
    )
    async with claude.messages.stream(
        model=MODEL_ID,
        max_tokens=64,
        tools=[tool],
        messages=[{"role": "user", "content": "Weather in Oslo?"}],
    ) as stream:
        first = await stream.get_final_message()
    assert first.stop_reason == "tool_use"
    block = first.content[0]
    assert (block.type, block.name, block.input) == ("tool_use", "get_weather", {"city": "Oslo"})

    upstream.reply = openai_reply("Rainy, 4 C.")
    second = await claude.messages.create(
        model=MODEL_ID,
        max_tokens=64,
        tools=[tool],
        messages=[
            {"role": "user", "content": "Weather in Oslo?"},
            {"role": "assistant", "content": [block.model_dump()]},
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": "4 C rain"}
                ],
            },
        ],
    )
    assert second.content[0].text == "Rainy, 4 C."
    assert upstream.last["messages"][-1]["role"] == "tool"


async def test_anthropic_errors_and_the_x_api_key_header(app: Any, claude: Any) -> None:
    with pytest.raises(anthropic.NotFoundError) as caught:
        await claude.messages.create(
            model="nope/none", max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )
    assert "does not exist" in str(caught.value)

    app.state.config.server.api_key = "sk-real"
    with pytest.raises(anthropic.AuthenticationError):
        await claude.messages.create(
            model=MODEL_ID, max_tokens=8, messages=[{"role": "user", "content": "x"}]
        )


async def test_anthropic_accepts_the_right_key_in_x_api_key(app: Any, upstream: Upstream) -> None:
    app.state.config.server.api_key = "sk-real"
    client = anthropic.AsyncAnthropic(
        base_url=BASE,
        api_key="sk-real",
        http_client=_anthropic_http_client(app),
        max_retries=0,
    )
    message = await client.messages.create(
        model=MODEL_ID, max_tokens=8, messages=[{"role": "user", "content": "x"}]
    )
    assert message.content[0].text == "Hello there."


async def test_openai_responses_create(oai: Any, upstream: Upstream) -> None:
    response = await oai.responses.create(model=MODEL_ID, input="hi", instructions="Be terse.")
    assert response.output_text == "Hello there."
    assert response.status == "completed"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (11, 5)


async def test_openai_responses_stream(oai: Any, upstream: Upstream) -> None:
    upstream.chunks = sse(
        chunk({"content": "Hel"}),
        chunk({"content": "lo"}),
        chunk({}, finish="stop", usage=(9, 2)),
        "[DONE]",
    )
    deltas: list[str] = []
    async with oai.responses.stream(model=MODEL_ID, input="hi") as stream:
        async for event in stream:
            if event.type == "response.output_text.delta":
                deltas.append(event.delta)
        final = await stream.get_final_response()
    assert deltas == ["Hel", "lo"]
    assert final.output_text == "Hello"
    assert final.usage.output_tokens == 2


async def test_openai_responses_function_call_round_trip(oai: Any, upstream: Upstream) -> None:
    calls = [
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Oslo"}'},
        }
    ]
    upstream.reply = openai_reply(None, finish="tool_calls", tool_calls=calls)
    tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Weather.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        "strict": False,
    }
    first = await oai.responses.create(model=MODEL_ID, input="weather?", tools=[tool])
    call = next(item for item in first.output if item.type == "function_call")
    assert (call.name, call.call_id, call.arguments) == ("get_weather", "call_3", '{"city":"Oslo"}')

    upstream.reply = openai_reply("Rainy.")
    second = await oai.responses.create(
        model=MODEL_ID,
        tools=[tool],
        input=[
            {"role": "user", "content": "weather?"},
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments,
            },
            {"type": "function_call_output", "call_id": call.call_id, "output": "4 C"},
        ],
    )
    assert second.output_text == "Rainy."


async def test_openai_chat_completions_still_work_through_the_shared_path(
    oai: Any, upstream: Upstream
) -> None:
    reply = await oai.chat.completions.create(
        model=MODEL_ID, messages=[{"role": "user", "content": "hi"}]
    )
    assert reply.choices[0].message.content == "Hello there."
