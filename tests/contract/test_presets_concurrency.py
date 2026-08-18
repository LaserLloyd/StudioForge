"""Contract tests for virtual-model presets and concurrent-load robustness.

Presets: two virtual models over one base must share ONE llama-server
instance (the efficiency win that makes personas cheap), and their sampler
defaults must be observable on the wire (``usage.completion_tokens`` proves
``max_tokens`` was applied) without ever overriding an explicit request value.

Concurrency: a burst of mixed streaming/non-streaming requests against one
loaded model must all succeed and must return the request counter to exactly
zero -- a stuck counter permanently blocks TTL-unload and eviction, which is
the VRAM-pinning failure class test_gateway_lifecycle.py guards at unit level.
"""

from __future__ import annotations

import concurrent.futures
import json
import time
from typing import Any

import httpx
import pytest

from tests.contract.conftest import ServerHandle, requires_engine, requires_models

pytestmark = [requires_engine, requires_models]


def _instances(raw: httpx.Client) -> list[dict[str, Any]]:
    response = raw.get("/api/status")
    response.raise_for_status()
    return list(response.json()["loaded"])


def _delete_virtual(raw: httpx.Client, virtual_id: str) -> None:
    raw.delete(f"/api/virtual-models/{virtual_id}")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


def test_two_presets_over_one_base_share_one_instance(
    raw: httpx.Client, chat_model: str
) -> None:
    """The whole point: N personas over one base cost one llama-server."""
    writer, coder = "sf-test/preset-writer", "sf-test/preset-coder"
    for virtual_id, prompt in ((writer, "You write prose."), (coder, "You write code.")):
        _delete_virtual(raw, virtual_id)
        response = raw.post(
            "/api/virtual-models",
            json={
                "id": virtual_id,
                "base_model_id": chat_model,
                "system_prompt": prompt,
                "temperature": 0.2,
            },
        )
        assert response.status_code == 200, response.text

    try:
        for virtual_id in (writer, coder):
            response = raw.post(
                "/v1/chat/completions",
                json={
                    "model": virtual_id,
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 8,
                },
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"]

        loaded = _instances(raw)
        chat_instances = [i for i in loaded if i["model_id"] in (writer, coder, chat_model)]
        assert [i["model_id"] for i in chat_instances] == [chat_model], (
            f"expected exactly one shared instance under {chat_model!r}, "
            f"got {[i['model_id'] for i in loaded]}"
        )

        # Both personas report as loaded, because their base is.
        models = {e["id"]: e for e in raw.get("/v1/models").json()["data"]}
        assert models[writer]["state"] == "loaded"
        assert models[coder]["state"] == "loaded"
    finally:
        _delete_virtual(raw, writer)
        _delete_virtual(raw, coder)


def test_preset_max_tokens_applies_only_when_the_request_omits_it(
    raw: httpx.Client, chat_model: str
) -> None:
    """usage.completion_tokens makes the sampler default wire-observable."""
    virtual_id = "sf-test/preset-terse"
    _delete_virtual(raw, virtual_id)
    response = raw.post(
        "/api/virtual-models",
        json={"id": virtual_id, "base_model_id": chat_model, "max_tokens": 2},
    )
    assert response.status_code == 200, response.text

    try:
        ask = {"role": "user", "content": "Tell me a very long story about the sea."}

        omitted = raw.post(
            "/v1/chat/completions", json={"model": virtual_id, "messages": [ask]}
        )
        assert omitted.status_code == 200, omitted.text
        assert omitted.json()["usage"]["completion_tokens"] == 2, (
            "the preset's max_tokens=2 must bind when the request omits the field"
        )

        explicit = raw.post(
            "/v1/chat/completions",
            json={"model": virtual_id, "messages": [ask], "max_tokens": 12},
        )
        assert explicit.status_code == 200, explicit.text
        assert explicit.json()["usage"]["completion_tokens"] == 12, (
            "an explicit request value must beat the preset default"
        )
    finally:
        _delete_virtual(raw, virtual_id)


def test_preset_virtual_model_streams(raw: httpx.Client, chat_model: str) -> None:
    """JIT-loadable and streamable by its own name, like any model."""
    virtual_id = "sf-test/preset-streamer"
    _delete_virtual(raw, virtual_id)
    response = raw.post(
        "/api/virtual-models",
        json={"id": virtual_id, "base_model_id": chat_model, "temperature": 0.1},
    )
    assert response.status_code == 200, response.text
    try:
        with raw.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": virtual_id,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8,
                "stream": True,
            },
        ) as stream:
            assert stream.status_code == 200
            body = b"".join(stream.iter_raw())
        assert b"data: [DONE]" in body
        assert b'"error"' not in body
    finally:
        _delete_virtual(raw, virtual_id)


# ---------------------------------------------------------------------------
# Concurrency hammer
# ---------------------------------------------------------------------------


def _one_request(base_url: str, api_key: str, model: str, *, stream: bool) -> str | None:
    """Returns None on success, else a short failure description."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Count to three."}],
        "max_tokens": 8,
        "stream": stream,
    }
    try:
        with httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600.0,
        ) as client:
            if not stream:
                response = client.post("/v1/chat/completions", json=payload)
                if response.status_code != 200:
                    return f"non-stream HTTP {response.status_code}"
                if not response.json()["choices"][0]["message"]["content"]:
                    return "non-stream empty content"
                return None
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    return f"stream HTTP {response.status_code}"
                body = b"".join(response.iter_raw())
            if b"data: [DONE]" not in body:
                return "stream missing [DONE]"
            for line in body.split(b"\n"):
                is_frame = line.startswith(b"data: ") and line != b"data: [DONE]"
                if is_frame and "error" in json.loads(line[6:]):
                    return f"stream error frame: {line[6:120]!r}"
            return None
    except Exception as exc:  # noqa: BLE001 - the failure IS the assertion
        return f"{type(exc).__name__}: {exc}"


def test_twenty_concurrent_mixed_requests_all_succeed_and_settle_to_zero(
    raw: httpx.Client, live_server: ServerHandle, chat_model: str
) -> None:
    # Warm the model first so the burst tests serving, not twenty JIT loads.
    warm = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 2,
        },
    )
    assert warm.status_code == 200, warm.text

    total = 20
    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as pool:
        futures = [
            pool.submit(
                _one_request,
                live_server.base_url,
                live_server.api_key,
                chat_model,
                stream=(index % 2 == 0),
            )
            for index in range(total)
        ]
        failures = [f.result(timeout=600) for f in futures]

    failures = [f for f in failures if f is not None]
    assert not failures, f"{len(failures)}/{total} requests failed: {failures[:5]}"

    # The counter must return to exactly zero -- anything else pins VRAM.
    deadline = time.time() + 15
    counts: dict[str, int] = {}
    while time.time() < deadline:
        counts = {i["model_id"]: i["active_requests"] for i in _instances(raw)}
        if all(v == 0 for v in counts.values()):
            break
        time.sleep(0.25)
    assert counts.get(chat_model) == 0, f"active_requests stuck: {counts}"
    assert all(v == 0 for v in counts.values()), f"active_requests stuck: {counts}"


def test_hammer_does_not_deadlock_subsequent_requests(
    raw: httpx.Client, chat_model: str
) -> None:
    """A plain request straight after the burst must answer promptly."""
    started = time.time()
    response = raw.post(
        "/v1/chat/completions",
        json={
            "model": chat_model,
            "messages": [{"role": "user", "content": "One word."}],
            "max_tokens": 4,
        },
    )
    assert response.status_code == 200
    assert time.time() - started < 60, "post-burst request took implausibly long"


@pytest.fixture(autouse=True, scope="module")
def _cleanup_virtuals(live_server: ServerHandle) -> Any:
    """Belt and braces: never leave test virtual models behind in the dev DB."""
    yield
    with httpx.Client(
        base_url=live_server.base_url,
        headers={"Authorization": f"Bearer {live_server.api_key}"},
        timeout=30.0,
    ) as client:
        for entry in client.get("/v1/models").json().get("data", []):
            if str(entry["id"]).startswith("sf-test/"):
                client.delete(f"/api/virtual-models/{entry['id']}")
