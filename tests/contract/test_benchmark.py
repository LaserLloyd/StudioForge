"""Contract tests for the benchmarking API against a REAL engine and models.

These run a real ``llama-server`` child on real GPUs: the whole claim of this
subsystem is that its numbers come from the engine rather than from our
arithmetic, and only an end-to-end run can verify that. The tiny 0.5B chat
model with a small context and a short generation keeps the suite fast.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from tests.contract.conftest import ServerHandle, requires_engine, requires_models

pytestmark = [requires_engine, requires_models]

#: Small enough that a mode completes in a few seconds on any of these cards.
CTX_SIZE = 1024
MAX_TOKENS = 24
SHORT_PROMPT = "Describe a benchmark harness in two sentences."


def poll_job(raw: httpx.Client, job_id: str, *, timeout_s: float = 900.0) -> dict[str, Any]:
    """Poll a benchmark job until it leaves the ``running`` state."""
    deadline = time.time() + timeout_s
    payload: dict[str, Any] = {}
    while time.time() < deadline:
        response = raw.get(f"/api/benchmark/jobs/{job_id}")
        assert response.status_code == 200, response.text
        payload = response.json()
        if payload["state"] != "running":
            return payload
        time.sleep(1.0)
    pytest.fail(f"benchmark job {job_id} did not finish within {timeout_s}s: {payload}")


def start_job(raw: httpx.Client, model: str, body: dict[str, Any]) -> dict[str, Any]:
    response = raw.post(f"/api/models/{model}/benchmark", json=body)
    assert response.status_code == 202, response.text
    return dict(response.json())


def device_override(raw: httpx.Client, model: str) -> Any:
    response = raw.get(f"/api/models/{model}/settings")
    assert response.status_code == 200, response.text
    return response.json()["device_override"]


# ---------------------------------------------------------------------------
# Mode discovery
# ---------------------------------------------------------------------------


def test_benchmark_modes_match_the_real_hardware(raw: httpx.Client) -> None:
    response = raw.get("/api/benchmark/modes")
    assert response.status_code == 200, response.text
    payload = response.json()

    gpus = payload["gpus"]
    if not gpus:
        pytest.skip("no CUDA GPUs visible to this process")
    for entry in gpus:
        assert set(entry) == {"index", "name", "total_bytes", "compute_capability"}
        assert entry["total_bytes"] > 0

    modes = payload["modes"]
    assert modes, "a machine with GPUs must offer at least one benchmark mode"
    for mode in modes:
        assert set(mode) == {"key", "label", "devices", "gpu_name"}
        assert mode["devices"], mode
        assert all(index in {g["index"] for g in gpus} for index in mode["devices"])
        assert mode["key"] == mode["key"].lower()

    keys = [mode["key"] for mode in modes]
    assert len(keys) == len(set(keys))

    # The reference rig (2x RTX 5090 + 2x RTX 3090) produces exactly five modes.
    names = sorted({g["name"] for g in gpus})
    if len(gpus) == 4 and names == ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 5090"]:
        assert keys == ["rtx-5090-x1", "rtx-5090-x2", "rtx-3090-x1", "rtx-3090-x2", "all"]
        assert [m["devices"] for m in modes] == [[0], [0, 1], [2], [2, 3], [0, 1, 2, 3]]


def test_model_modes_carry_planner_verdicts(raw: httpx.Client, chat_model: str) -> None:
    response = raw.get(
        f"/api/models/{chat_model}/benchmark/modes", params={"ctx_size": CTX_SIZE}
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["model_id"] == chat_model

    modes = payload["modes"]
    assert modes
    for mode in modes:
        assert set(mode) == {"key", "label", "devices", "gpu_name", "applicable", "skipped_reason"}
        assert isinstance(mode["applicable"], bool)
        if mode["applicable"]:
            assert mode["skipped_reason"] is None
        else:
            assert mode["skipped_reason"]
    # A 0.5B Q8_0 model fits anywhere.
    assert all(mode["applicable"] for mode in modes)


def test_oversized_model_reports_reasons_rather_than_erroring(
    raw: httpx.Client, live_server: ServerHandle
) -> None:
    """A model too big for a single card must be *reported*, not rejected."""
    listing = raw.get("/api/models")
    assert listing.status_code == 200, listing.text
    gpus = raw.get("/api/benchmark/modes").json()["gpus"]
    if not gpus:
        pytest.skip("no CUDA GPUs visible to this process")
    smallest = min(g["total_bytes"] for g in gpus)

    oversized = next(
        (
            entry["id"]
            for entry in listing.json()["models"]
            if entry.get("size_bytes", 0) > smallest * 1.2
        ),
        None,
    )
    if oversized is None:
        pytest.skip("no model in the library is larger than the smallest GPU")

    response = raw.get(f"/api/models/{oversized}/benchmark/modes", params={"ctx_size": CTX_SIZE})
    assert response.status_code == 200, response.text
    modes = response.json()["modes"]
    skipped = [mode for mode in modes if not mode["applicable"]]
    assert skipped, f"expected some placements to be too small for {oversized}"
    for mode in skipped:
        # The reason is actionable prose from the planner, not a bare boolean.
        assert isinstance(mode["skipped_reason"], str)
        assert len(mode["skipped_reason"]) > 10


# ---------------------------------------------------------------------------
# Running a real benchmark
# ---------------------------------------------------------------------------


@pytest.mark.timeout(1200)
def test_single_gpu_benchmark_runs_and_restores_state(
    raw: httpx.Client, chat_model: str
) -> None:
    modes = raw.get(f"/api/models/{chat_model}/benchmark/modes").json()["modes"]
    first = next(mode for mode in modes if mode["applicable"] and len(mode["devices"]) == 1)
    before = device_override(raw, chat_model)

    started = start_job(
        raw,
        chat_model,
        {
            "modes": [first["key"]],
            "ctx_size": CTX_SIZE,
            "max_tokens": MAX_TOKENS,
            "prompt": SHORT_PROMPT,
        },
    )
    assert started["model_id"] == chat_model
    assert started["modes"] == [first["key"]]
    assert started["job_id"]

    job = poll_job(raw, started["job_id"])
    assert job["state"] == "completed", job
    assert job["error"] is None
    assert job["progress"]["fraction"] == 1.0

    report = job["report"]
    assert report["model_id"] == chat_model
    assert report["ctx_size"] == CTX_SIZE
    assert report["max_tokens"] == MAX_TOKENS
    assert report["prompt_chars"] == len(SHORT_PROMPT)
    assert report["finished_at"] is not None

    result = report["results"][0]
    assert result["mode"] == first["key"]
    assert result["error"] is None, result
    assert result["applicable"] is True
    assert result["load_time_s"] is not None and result["load_time_s"] > 0
    assert result["generation_tps"] is not None and result["generation_tps"] > 0
    assert result["generated_tokens"] is not None and result["generated_tokens"] > 0
    assert result["prompt_tokens"] is not None and result["prompt_tokens"] > 0
    assert result["ttft_s"] is not None and result["ttft_s"] > 0
    assert report["best_generation_mode"] == first["key"]
    assert report["best_prompt_mode"] == first["key"]

    # The benchmark must not leave the model pinned to whichever GPU it used.
    assert device_override(raw, chat_model) == before

    # And the finished report is durable.
    history = raw.get(f"/api/models/{chat_model}/benchmarks", params={"limit": 5})
    assert history.status_code == 200, history.text
    rows = history.json()["benchmarks"]
    assert rows
    assert rows[0]["report"]["results"][0]["mode"] == first["key"]
    assert rows[0]["id"] > 0
    assert rows[0]["ts"] > 0


@pytest.mark.timeout(1200)
def test_cancel_stops_the_job_and_restores_state(raw: httpx.Client, chat_model: str) -> None:
    before = device_override(raw, chat_model)
    started = start_job(
        raw,
        chat_model,
        {"ctx_size": CTX_SIZE, "max_tokens": MAX_TOKENS, "prompt": SHORT_PROMPT},
    )
    job_id = started["job_id"]
    assert len(started["modes"]) >= 1

    cancel = raw.delete(f"/api/benchmark/jobs/{job_id}")
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["canceled"] is True

    job = poll_job(raw, job_id)
    assert job["state"] == "canceled", job
    # A cancel stops at a mode boundary, so at most the modes already started
    # produced results -- never all of them.
    if job["report"] is not None:
        assert len(job["report"]["results"]) < max(2, len(started["modes"]))
    assert device_override(raw, chat_model) == before

    # Cancelling an already-finished job is a no-op, not an error.
    again = raw.delete(f"/api/benchmark/jobs/{job_id}")
    assert again.status_code == 200
    assert again.json()["canceled"] is False


def test_unknown_job_and_unknown_mode_are_clean_errors(
    raw: httpx.Client, chat_model: str
) -> None:
    missing = raw.get("/api/benchmark/jobs/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "job_not_found"

    bad_mode = raw.post(
        f"/api/models/{chat_model}/benchmark", json={"modes": ["rtx-9999-x9"]}
    )
    assert bad_mode.status_code == 400
    assert "unknown benchmark mode" in bad_mode.json()["error"]["message"]

    bad_model = raw.get("/api/models/nope%2Fmissing/benchmark/modes")
    assert bad_model.status_code == 404
