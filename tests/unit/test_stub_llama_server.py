"""The stub llama-server (D85): every endpoint, determinism, metrics, the CLI shims.

The stub is what a no-GPU staging instance launches for every load and what the
supervisor tests use as their child, so these pin the contract both rely on:
the real argv is parsed, the endpoints a client or the gateway touches answer
the way llama-server answers them, text is a pure function of prompt and seed,
and ``/metrics`` counters advance by what was served.

Servers here run in-process on an ephemeral port (``port 0``); only the CLI
shim and fault tests start the module as a subprocess, and those never listen.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from studioforge.core.engine import _BUILD_RE, _version_line, parse_engine_features
from studioforge.testing import stub_llama_server as stub

STUB_FILE = Path(stub.__file__).resolve()

Serve = Callable[..., httpx.Client]


@pytest.fixture
def serve() -> Iterator[Serve]:
    """Start stubs on ephemeral ports; stop them all at teardown."""
    running: list[tuple[stub.StubServer, threading.Thread, httpx.Client]] = []

    def start(*argv: str) -> httpx.Client:
        cfg = stub.StubConfig.from_argv(["--port", "0", *argv])
        server = stub.make_server(cfg)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address[0], server.server_address[1]
        client = httpx.Client(base_url=f"http://{host}:{port}", timeout=20.0)
        running.append((server, thread, client))
        return client

    yield start
    for server, thread, client in running:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - our own stub, fixed argv
        [sys.executable, str(STUB_FILE), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        env=env,
    )


def sse_events(response: httpx.Response) -> list[str]:
    return [
        line[len("data: ") :] for line in response.text.splitlines() if line.startswith("data: ")
    ]


# ---------------------------------------------------------------------------
# Health, props, slots: what the supervisor reads
# ---------------------------------------------------------------------------


def test_health_is_503_for_the_load_window_then_ok(serve: Serve) -> None:
    client = serve("--stub-load-ms", "600")
    first = client.get("/health")
    assert first.status_code == 503
    assert first.json()["error"]["message"] == "Loading model"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if client.get("/health").status_code == 200:
            break
        time.sleep(0.05)
    assert client.get("/health").json() == {"status": "ok"}


def test_health_is_immediately_ok_by_default(serve: Serve) -> None:
    assert serve().get("/health").json() == {"status": "ok"}


def test_props_echoes_the_launch(serve: Serve, tmp_path: Path) -> None:
    template = tmp_path / "t.jinja"
    template.write_text("{{ stub template }}", encoding="utf-8")
    client = serve(
        "--model", "/m/q.gguf", "--alias", "q", "--ctx-size", "8192", "--parallel", "4",
        "--chat-template-file", str(template), "--stub-build", "7", "--metrics",
    )  # fmt: skip
    props = client.get("/props").json()
    assert props["model_alias"] == "q"
    assert props["model_path"] == "/m/q.gguf"
    assert props["total_slots"] == 4
    # llama-server reports the per-slot context: total / slots.
    assert props["default_generation_settings"]["n_ctx"] == 2048
    assert props["chat_template"] == "{{ stub template }}"
    assert props["build_info"] == "stub-b7"
    assert props["endpoint_metrics"] is True


def test_props_chat_template_falls_back_to_a_fixed_string(serve: Serve) -> None:
    assert serve().get("/props").json()["chat_template"] == "{{ messages }}"


def test_argv_parsing_covers_long_short_and_equals_forms(serve: Serve) -> None:
    client = serve(
        "-m", "x.gguf", "-a", "al", "-c", "4096", "-np", "2", "-ngl", "999", "-sm", "layer",
        "-ts", "1,1", "-dev", "CUDA0,CUDA1", "-ctk", "q8_0", "-ctv", "q4_0", "-b", "2048",
        "-ub", "512", "-fa", "on", "--reasoning-format=none", "--some-flag-the-stub-ignores",
        "--another", "value",
    )  # fmt: skip
    launch = client.get("/props").json()["stub"]["launch"]
    assert launch == {
        "model": "x.gguf",
        "alias": "al",
        "ctx_size": 4096,
        "parallel": 2,
        "n_gpu_layers": "999",
        "device": "CUDA0,CUDA1",
        "tensor_split": "1,1",
        "split_mode": "layer",
        "cache_type_k": "q8_0",
        "cache_type_v": "q4_0",
        "batch_size": 2048,
        "ubatch_size": 512,
        "flash_attn": "on",
        "reasoning_format": "none",
        "chat_template_file": None,
        "metrics": False,
    }


def test_stub_vram_is_echoed(serve: Serve) -> None:
    assert serve("--stub-vram-mib", "2048").get("/props").json()["stub"]["vram_mib"] == 2048


def test_slots_are_idle_and_sized_per_parallel(serve: Serve) -> None:
    slots = serve("--ctx-size", "6000", "--parallel", "3").get("/slots").json()
    assert [s["id"] for s in slots] == [0, 1, 2]
    assert all(s["is_processing"] is False and s["n_ctx"] == 2000 for s in slots)


def test_a_slot_is_processing_while_a_paced_completion_runs(serve: Serve) -> None:
    client = serve("--parallel", "2", "--stub-tps", "40")
    done = threading.Event()

    def call() -> None:
        client.post("/completion", json={"prompt": "hello", "n_predict": 12})
        done.set()

    threading.Thread(target=call, daemon=True).start()
    seen = False
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not done.is_set():
        slots = client.get("/slots").json()
        if any(s["is_processing"] for s in slots):
            seen = True
            break
        time.sleep(0.02)
    assert seen
    assert done.wait(10)
    assert not any(s["is_processing"] for s in client.get("/slots").json())


def test_v1_models_lists_the_alias(serve: Serve) -> None:
    data = serve("--alias", "zed").get("/v1/models").json()
    assert [m["id"] for m in data["data"]] == ["zed"]


def test_lora_adapters(serve: Serve) -> None:
    client = serve()
    assert client.get("/lora-adapters").json() == []
    assert client.post("/lora-adapters", json=[{"id": 0, "scale": 0.5}]).json() == {"success": True}


def test_unknown_route_is_404_in_llama_servers_shape(serve: Serve) -> None:
    response = serve().get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_bad_json_is_400(serve: Serve) -> None:
    response = serve().post("/completion", content=b"{not json")
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Tokenizer and template
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", "a", "hello world", "café ☃ \U0001f600", "x" * 101])
def test_tokenize_detokenize_round_trip(serve: Serve, text: str) -> None:
    client = serve()
    tokens = client.post("/tokenize", json={"content": text}).json()["tokens"]
    assert len(tokens) == math.ceil(len(text.encode()) / 4)
    assert client.post("/detokenize", json={"tokens": tokens}).json() == {"content": text}


def test_tokenize_with_pieces(serve: Serve) -> None:
    tokens = serve().post("/tokenize", json={"content": "abcdefgh", "with_pieces": True}).json()
    assert [t["piece"] for t in tokens["tokens"]] == ["abcd", "efgh"]


def test_detokenize_rejects_a_non_list(serve: Serve) -> None:
    assert serve().post("/detokenize", json={"tokens": "x"}).status_code == 400


def test_apply_template(serve: Serve) -> None:
    prompt = (
        serve()
        .post(
            "/apply-template",
            json={
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ]
            },
        )
        .json()["prompt"]
    )
    assert prompt == "<|system|>\nbe brief\n<|user|>\nhi\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Completions: determinism, shapes, streaming
# ---------------------------------------------------------------------------


def test_completion_is_deterministic_in_prompt_and_seed(serve: Serve) -> None:
    client = serve()

    def text(prompt: str, seed: int | None) -> str:
        body: dict[str, object] = {"prompt": prompt, "n_predict": 20}
        if seed is not None:
            body["seed"] = seed
        return client.post("/completion", json=body).json()["content"]

    assert text("alpha", 1) == text("alpha", 1)
    assert text("alpha", 1) != text("alpha", 2)
    assert text("alpha", 1) != text("beta", 1)
    assert text("alpha", None) == text("alpha", 0) == text("alpha", -1)


def test_completion_shape(serve: Serve) -> None:
    body = (
        serve("--alias", "m", "--ctx-size", "2048")
        .post("/completion", json={"prompt": "hello world", "n_predict": 9})
        .json()
    )
    assert body["stop"] is True
    assert body["tokens_predicted"] == 9
    assert body["tokens_evaluated"] == math.ceil(len("hello world") / 4)
    assert body["model"] == "m"
    assert body["stop_type"] == "limit"
    assert body["timings"]["predicted_n"] == 9
    assert len(body["content"].split()) == 9


def test_completion_accepts_token_ids_as_prompt(serve: Serve) -> None:
    client = serve()
    ids = client.post("/tokenize", json={"content": "same words"}).json()["tokens"]
    by_ids = client.post("/completion", json={"prompt": ids, "n_predict": 5}).json()["content"]
    by_text = client.post("/completion", json={"prompt": "same words", "n_predict": 5}).json()
    assert by_ids == by_text["content"]


def test_completion_stream_matches_the_non_stream_text(serve: Serve) -> None:
    client = serve()
    plain = client.post("/completion", json={"prompt": "p", "n_predict": 8, "seed": 3}).json()
    streamed = client.post(
        "/completion", json={"prompt": "p", "n_predict": 8, "seed": 3, "stream": True}
    )
    events = [json.loads(e) for e in sse_events(streamed)]
    assert "".join(e["content"] for e in events) == plain["content"]
    assert events[-1]["stop"] is True
    assert events[-1]["timings"]["predicted_n"] == 8


def test_openai_completions_shape_and_usage(serve: Serve) -> None:
    body = (
        serve("--alias", "m")
        .post("/v1/completions", json={"prompt": "abcd" * 5, "max_tokens": 6})
        .json()
    )
    assert body["object"] == "text_completion"
    assert body["model"] == "m"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
    assert body["timings"]["prompt_n"] == 5


def test_openai_completions_stream_with_usage(serve: Serve) -> None:
    response = serve().post(
        "/v1/completions",
        json={
            "prompt": "x",
            "max_tokens": 4,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[-1]["usage"]["completion_tokens"] == 4
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"


MESSAGES = [{"role": "user", "content": "say something"}]


def test_chat_completion_shape(serve: Serve) -> None:
    body = (
        serve("--alias", "m")
        .post("/v1/chat/completions", json={"model": "m", "messages": MESSAGES, "max_tokens": 10})
        .json()
    )
    choice = body["choices"][0]
    assert body["object"] == "chat.completion"
    assert body["system_fingerprint"] == "stub-b0"
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"].split()) == 10
    assert choice["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 10
    assert body["usage"]["prompt_tokens"] + 10 == body["usage"]["total_tokens"]
    assert body["timings"]["predicted_n"] == 10


def test_chat_without_a_limit_ends_with_stop(serve: Serve) -> None:
    body = serve().post("/v1/chat/completions", json={"messages": MESSAGES}).json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 24


def test_chat_is_deterministic_in_the_messages_and_seed(serve: Serve) -> None:
    client = serve()

    def content(messages: list[dict[str, str]], seed: int) -> str:
        reply = client.post("/v1/chat/completions", json={"messages": messages, "seed": seed})
        return reply.json()["choices"][0]["message"]["content"]

    other = [{"role": "user", "content": "something else"}]
    assert content(MESSAGES, 5) == content(MESSAGES, 5)
    assert content(MESSAGES, 5) != content(MESSAGES, 6)
    assert content(MESSAGES, 5) != content(other, 5)


def test_chat_stream_equals_non_stream_and_ends_with_done(serve: Serve) -> None:
    client = serve()
    plain = client.post(
        "/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 7, "seed": 9}
    ).json()["choices"][0]["message"]["content"]
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": MESSAGES, "max_tokens": 7, "seed": 9, "stream": True,
            "stream_options": {"include_usage": True},
        },
    )  # fmt: skip
    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == plain
    assert chunks[-1]["usage"]["completion_tokens"] == 7
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finals[-1]["choices"][0]["finish_reason"] == "length"


def test_chat_requires_messages(serve: Serve) -> None:
    assert serve().post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_infill_is_deterministic(serve: Serve) -> None:
    client = serve()
    body = {"input_prefix": "def f(", "input_suffix": "):\n    pass", "n_predict": 6}
    first = client.post("/infill", json=body).json()
    assert first["tokens_predicted"] == 6
    assert first["content"] == client.post("/infill", json=body).json()["content"]


def test_a_prompt_longer_than_the_slot_is_the_engines_400(serve: Serve) -> None:
    client = serve("--ctx-size", "64", "--parallel", "1")
    response = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "w" * 400}]}
    )
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "exceed_context_size_error"
    assert error["n_ctx"] == 64
    assert error["n_prompt_tokens"] > 64


def test_generation_is_capped_to_the_room_left_in_the_slot(serve: Serve) -> None:
    client = serve("--ctx-size", "64")
    body = client.post("/completion", json={"prompt": "abcd" * 10, "n_predict": 500}).json()
    assert body["tokens_predicted"] == 64 - 10


def test_pacing_slows_generation_to_the_requested_rate(serve: Serve) -> None:
    client = serve("--stub-tps", "50")
    started = time.monotonic()
    body = client.post("/completion", json={"prompt": "p", "n_predict": 15}).json()
    assert time.monotonic() - started >= 0.25
    assert body["timings"]["predicted_per_second"] == 50.0


# ---------------------------------------------------------------------------
# Embeddings and reranking
# ---------------------------------------------------------------------------


def test_embeddings_are_fixed_dimension_unit_vectors(serve: Serve) -> None:
    client = serve("--stub-embed-dim", "16", "--alias", "e")
    body = client.post("/v1/embeddings", json={"input": ["one", "two", "one"]}).json()
    vectors = [row["embedding"] for row in body["data"]]
    assert [row["index"] for row in body["data"]] == [0, 1, 2]
    assert all(len(v) == 16 for v in vectors)
    assert vectors[0] == vectors[2] != vectors[1]
    assert math.isclose(sum(x * x for x in vectors[0]), 1.0, rel_tol=1e-3)
    assert body["model"] == "e"
    assert body["usage"]["prompt_tokens"] == body["usage"]["total_tokens"] == 3


def test_embedding_native_shape_and_string_input(serve: Serve) -> None:
    client = serve()
    native = client.post("/embedding", json={"content": "hello"}).json()
    assert native[0]["index"] == 0
    assert len(native[0]["embedding"][0]) == 64
    openai = client.post("/v1/embeddings", json={"input": "hello"}).json()
    assert openai["data"][0]["embedding"] == native[0]["embedding"][0]


def test_embeddings_reject_a_missing_input(serve: Serve) -> None:
    assert serve().post("/v1/embeddings", json={}).status_code == 400


def test_reranking_orders_by_overlap(serve: Serve) -> None:
    body = (
        serve()
        .post(
            "/reranking",
            json={
                "query": "red fox",
                "documents": ["a blue whale", "the red fox jumps", "red"],
                "top_n": 2,
            },
        )
        .json()
    )
    assert [r["index"] for r in body["results"]] == [1, 2]
    assert body["results"][0]["relevance_score"] > body["results"][1]["relevance_score"]


def test_reranking_needs_query_and_documents(serve: Serve) -> None:
    assert serve().post("/reranking", json={"query": "x"}).status_code == 400


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split()
            out[name.removeprefix("llamacpp:")] = float(value)
    return out


def test_metrics_need_the_flag_like_the_real_engine(serve: Serve) -> None:
    response = serve().get("/metrics")
    assert response.status_code == 501
    assert "--metrics" in response.json()["error"]["message"]


def test_metrics_counters_advance_with_each_served_request(serve: Serve) -> None:
    client = serve("--metrics", "--parallel", "2")
    before = parse_metrics(client.get("/metrics").text)
    assert before["prompt_tokens_total"] == before["tokens_predicted_total"] == 0
    first = client.post(
        "/v1/chat/completions", json={"messages": MESSAGES, "max_tokens": 10}
    ).json()
    mid = parse_metrics(client.get("/metrics").text)
    assert mid["prompt_tokens_total"] == first["usage"]["prompt_tokens"]
    assert mid["tokens_predicted_total"] == 10
    assert mid["n_decode_total"] == 10
    assert mid["prompt_seconds_total"] > 0
    assert mid["tokens_predicted_seconds_total"] > 0
    client.post("/completion", json={"prompt": "abcdefgh", "n_predict": 5})
    after = parse_metrics(client.get("/metrics").text)
    assert after["tokens_predicted_total"] == 15
    assert after["prompt_tokens_total"] == mid["prompt_tokens_total"] + 2
    assert after["n_decode_total"] == 15
    assert after["n_busy_slots_per_decode"] == pytest.approx(1.0)
    assert after["requests_processing"] == after["requests_deferred"] == 0
    assert after["predicted_tokens_seconds"] > 0


def test_metrics_text_is_prometheus_exposition(serve: Serve) -> None:
    response = serve("--metrics").get("/metrics")
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE llamacpp:tokens_predicted_total counter" in response.text
    assert "# TYPE llamacpp:requests_processing gauge" in response.text


# ---------------------------------------------------------------------------
# CLI shims: what engine verification runs
# ---------------------------------------------------------------------------


def test_version_has_a_version_line_and_a_build_line() -> None:
    result = run_cli("--version")
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == "version: 0 (stub)"
    assert any(_BUILD_RE.search(line) for line in result.stdout.splitlines())
    assert _version_line(result.stdout) == "version: 0 (stub)"


def test_version_reports_the_configured_build() -> None:
    assert run_cli("--stub-build", "12", "--version").stdout.startswith("version: 12 (stub)")


def test_help_lists_the_flags_feature_detection_reads() -> None:
    result = run_cli("--help")
    assert result.returncode == 0
    features = parse_engine_features(result.stdout, "stub")
    assert features.known
    for flag in (
        "--ctx-size",
        "--parallel",
        "--tensor-split",
        "--metrics",
        "--cache-ram",
        "--device",
    ):
        assert features.has(flag), flag
    assert features.split_modes == ("none", "layer", "row", "tensor")
    assert features.flash_attn_values == ("on", "off", "auto")
    assert features.cache_ram and features.cache_ram_default_mib == 8192
    assert features.kv_unified
    assert features.supports_spec("draft-mtp")
    assert result.stdout == stub.help_text()


def test_help_covers_every_flag_the_supervisor_can_emit() -> None:
    """A flag build_command can send that the help omits would be dropped by feature gating."""
    import re

    supervisor_source = (
        Path(__file__).resolve().parents[2] / "src/studioforge/core/supervisor.py"
    ).read_text(encoding="utf-8")
    emitted = set(re.findall(r'"(--[a-z][a-z0-9-]*)"', supervisor_source))
    advertised = parse_engine_features(stub.help_text(), "stub").flags
    # Policy refusals and flags the supervisor only ever reads or strips.
    refused = {"--cpu-moe", "--n-cpu-moe", "--override-tensor", "--load-mode", "--no-kv-unified"}
    missing = sorted(f for f in emitted - advertised - refused if not f.startswith("--no-"))
    assert missing == [], missing


def test_list_devices_reads_the_fake_cards() -> None:
    import os

    env = dict(os.environ)
    env["SF_FAKE_GPUS"] = json.dumps(
        [
            {"index": 0, "name": "RTX 5090", "total_bytes": 32 * 1024**3},
            {
                "index": 1,
                "name": "RTX 3090",
                "total_bytes": 24 * 1024**3,
                "free_bytes": 20 * 1024**3,
            },
        ]
    )
    result = run_cli("--list-devices", env=env)
    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "Available devices:",
        "  CUDA0: RTX 5090 (32768 MiB, 32768 MiB free)",
        "  CUDA1: RTX 3090 (24576 MiB, 20480 MiB free)",
    ]


def test_list_devices_without_fake_cards_lists_none() -> None:
    import os

    env = {k: v for k, v in os.environ.items() if k != "SF_FAKE_GPUS"}
    assert run_cli("--list-devices", env=env).stdout == "Available devices:\n"


def test_exit_code_fault_flag() -> None:
    result = run_cli("--port", "1", "--fake-exit-code", "3")
    assert result.returncode == 3
    assert "CUDA error: out of memory" in result.stderr
    assert "fake-llama-server starting" in result.stderr


@contextmanager
def _env(**values: str) -> Iterator[None]:
    import os

    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def test_env_supplies_the_tuning_defaults_and_the_flag_wins() -> None:
    with _env(SF_STUB_TPS="25", SF_STUB_LOAD_MS="300"):
        cfg = stub.StubConfig.from_argv([])
        assert (cfg.tps, cfg.load_ms) == (25.0, 300.0)
        cfg = stub.StubConfig.from_argv(["--stub-tps", "5", "--stub-load-ms", "0"])
        assert (cfg.tps, cfg.load_ms) == (5.0, 0.0)


def test_parallel_auto_gets_a_handful_of_slots() -> None:
    assert stub.StubConfig.from_argv(["--parallel", "-1"]).parallel == 4
