"""A whole staging instance in-process: ``create_app`` + stub children + fake probe (D85).

This is the E.0 smoke: synthetic cards, ``engine.stub``, a real child process
(the stub llama-server) per load, and the real gateway in front of it. It
proves the pieces work together -- load, chat (plain and streaming), the child's
own ``/props``, free memory moving on the synthetic cards, a D51 observation
row, unload, and a JIT reload -- without touching a GPU or a live port (the
child ports are the reserved 19420-19460 range).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studioforge import build
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import gpu as gpu_module
from tests.unit.test_mcp import fake_meta

GIB = 1024**3
FAKE_GPUS = json.dumps(
    [
        {"index": 0, "name": "RTX 5090", "total_bytes": 32 * GIB, "compute_capability": [12, 0]},
        {"index": 1, "name": "RTX 3090", "total_bytes": 24 * GIB, "compute_capability": [8, 6]},
    ]
)
MESSAGES = [{"role": "user", "content": "say something"}]


@pytest.fixture
def staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, object]]:
    monkeypatch.setenv("SF_FAKE_GPUS", FAKE_GPUS)
    monkeypatch.delenv("SF_GPU_PROBE", raising=False)
    gpu_module.reset_probe()
    models = tmp_path / "models" / "pub"
    models.mkdir(parents=True)
    (models / "Tiny-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * (4 * 1024 * 1024))

    cfg = Config(data_dir=tmp_path / "data")
    cfg.engine.stub = True
    cfg.gpu.backend = "fake"
    cfg.models.dir = tmp_path / "models"
    cfg.gateway.child_port_start = 19440
    cfg.gateway.child_port_end = 19460
    cfg.gateway.load_timeout_s = 30.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.ensure_dirs()

    state = build_state(cfg)
    state.registry._meta_reader = fake_meta  # noqa: SLF001 - filename-driven GGUF metadata
    app = create_app(cfg, state=state)
    try:
        with TestClient(app) as client:
            yield client, state
    finally:
        build.mark_stub_engine(False)
        build.build_id.cache_clear()  # type: ignore[attr-defined]
        gpu_module.reset_probe()


def free_by_card(client: TestClient) -> dict[int, int]:
    return {g["index"]: g["free_bytes"] for g in client.get("/api/gpus").json()["gpus"]}


def test_a_staging_instance_serves_every_step_of_a_load(
    staging: tuple[TestClient, object],
) -> None:
    client, state = staging

    health = client.get("/health").json()
    assert health["build"].endswith("+stub")

    gpus = client.get("/api/gpus").json()
    assert gpus["backend"] == "fake"
    before = free_by_card(client)
    assert before == {0: 32 * GIB, 1: 24 * GIB}

    listed = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in listed] == ["pub/Tiny-Q4_K_M"]
    model_id = listed[0]["id"]

    # Load through the management route: the stub child comes up and is ready.
    loaded = client.post(f"/api/models/{model_id}/load", json={"ctx_size": 4096})
    assert loaded.status_code == 200, loaded.text
    body = loaded.json()
    assert body["state"] == "ready"
    assert body["engine_tag"] in (None, "stub") or body["resolved_engine_tag"] == "stub"

    # Free memory moved on the card(s) it was placed on, and only there.
    after = free_by_card(client)
    placed = body["plan"]["devices"]
    assert placed
    for index, free in after.items():
        if index in placed:
            assert free < before[index]
        else:
            assert free == before[index]

    # Chat through the gateway, plain: deterministic in prompt and seed.
    payload = {"model": model_id, "messages": MESSAGES, "max_tokens": 8, "seed": 4}
    first = client.post("/v1/chat/completions", json=payload)
    assert first.status_code == 200, first.text
    again = client.post("/v1/chat/completions", json=payload)
    text = first.json()["choices"][0]["message"]["content"]
    assert text == again.json()["choices"][0]["message"]["content"]
    assert len(text.split()) == 8
    assert first.json()["usage"]["completion_tokens"] == 8

    # ... and streaming.
    streamed = client.post("/v1/chat/completions", json={**payload, "stream": True})
    events = [
        line[len("data: ") :] for line in streamed.text.splitlines() if line.startswith("data: ")
    ]
    assert events[-1] == "[DONE]"
    pieces = [
        json.loads(e)["choices"][0]["delta"].get("content") or ""
        for e in events[:-1]
        if json.loads(e)["choices"]
    ]
    assert "".join(pieces) == text

    # The child's own /props, through the manager's introspection.
    props = client.get(f"/api/models/{model_id}/introspect").json()
    assert "stub" in json.dumps(props)
    assert model_id in json.dumps(props)

    # D51: a stub load writes the predicted-vs-actual row a real one does.
    rows = state.db.load_observations(model_id)  # type: ignore[attr-defined]
    assert rows, "no load observation was recorded for the stub load"
    assert rows[0]["actual_bytes"] and rows[0]["actual_bytes"] > 0

    # Unload gives the memory back.
    unloaded = client.post(f"/api/models/{model_id}/unload")
    assert unloaded.status_code == 200, unloaded.text
    assert free_by_card(client) == before

    # A request for an unloaded model JIT-loads it again.
    jit = client.post("/v1/chat/completions", json=payload)
    assert jit.status_code == 200, jit.text
    assert jit.json()["choices"][0]["message"]["content"] == text
    assert any(free < before[index] for index, free in free_by_card(client).items())
