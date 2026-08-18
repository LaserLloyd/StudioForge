"""Speculative decoding and LoRA adapter contract tests.

The important assertion here is not "it did not crash" but that drafting is
*actually enabled*. `b10425` renamed the whole spec-draft flag surface and
accepts the old spellings while ignoring them, so a mis-built command line
produces a server that runs perfectly and never drafts.

Finding the trustworthy signal took measurement: `/props` reports
`default_generation_settings.params["speculative.types"] == "none"` even with a
draft model loaded and demonstrably drafting, because that field describes
per-request sampling defaults. The two signals that ARE truthful are the
per-slot `speculative` boolean in `/slots`, and `draft_n` /
`draft_n_accepted` in a completion's `timings`.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import pytest

from tests.contract.conftest import ServerHandle, requires_engine, requires_models

pytestmark = [requires_engine, requires_models]

# A genuinely compatible pair: both qwen2, tokenizer "gpt2", vocab 151936.
# Deliberately the small pair (940 MiB target + 506 MiB draft) so this suite
# stays runnable -- a 16 GiB target would make it a benchmark, not a test.
TARGET_NEEDLE = "Qwen2.5-1.5B-Instruct-Q4_K_M"
DRAFT_NEEDLE = "Qwen2.5-0.5B-Instruct-Q8_0"
# A different family entirely (gemma4, vocab 262144) -- must be refused.
INCOMPATIBLE_DRAFT_NEEDLE = "gemma-4-31B-it-QAT-Q4_0"


def resolve_or_skip(server: ServerHandle, needle: str) -> str:
    model = server.resolve_model(needle)
    if model is None:
        pytest.skip(f"model matching {needle!r} is not in the library")
    return model


@pytest.fixture
def restore_settings(raw: httpx.Client):
    """Save and restore per-model settings so tests do not leak state."""
    saved: list[tuple[str, dict[str, Any]]] = []

    def remember(model_id: str) -> None:
        response = raw.get(f"/api/models/{model_id}/settings")
        response.raise_for_status()
        saved.append((model_id, response.json()))

    yield remember

    for model_id, settings in reversed(saved):
        raw.put(f"/api/models/{model_id}/settings", json=settings)
        raw.post(f"/api/models/{model_id}/unload")


# ---------------------------------------------------------------------------
# Compatibility gate
# ---------------------------------------------------------------------------


def test_incompatible_draft_is_refused_with_reason(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    """A vocab mismatch produces garbage output, so block it at config time."""
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    bad_draft = resolve_or_skip(live_server, INCOMPATIBLE_DRAFT_NEEDLE)
    restore_settings(target)

    response = raw.put(f"/api/models/{target}/settings", json={"draft_model_id": bad_draft})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["param"] == "draft_model_id"
    assert "vocab size mismatch" in error["message"]
    assert "151936" in error["message"] and "262144" in error["message"]


def test_model_cannot_be_its_own_draft(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    restore_settings(target)
    response = raw.put(f"/api/models/{target}/settings", json={"draft_model_id": target})
    assert response.status_code == 400
    assert "its own draft" in response.json()["error"]["message"]


def test_unknown_draft_is_refused(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    restore_settings(target)
    response = raw.put(f"/api/models/{target}/settings", json={"draft_model_id": "no/such-draft"})
    assert response.status_code == 400
    assert "not in the registry" in response.json()["error"]["message"]


def test_compatible_draft_is_accepted(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    draft = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(target)
    response = raw.put(f"/api/models/{target}/settings", json={"draft_model_id": draft})
    assert response.status_code == 200
    assert response.json()["draft_model_id"] == draft


# ---------------------------------------------------------------------------
# Planner accounts for the draft
# ---------------------------------------------------------------------------


def test_plan_includes_draft_weights(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    """Draft weights + its KV are part of the load plan, not a surprise."""
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    draft = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(target)

    before = raw.get(f"/api/models/{target}/plan", params={"ctx_size": 4096}).json()
    raw.put(f"/api/models/{target}/settings", json={"draft_model_id": draft})
    after = raw.get(f"/api/models/{target}/plan", params={"ctx_size": 4096}).json()

    assert after["estimate_mb"]["draft_weights_bytes"] > 0
    assert after["estimate_mb"]["total"] > before["estimate_mb"]["total"]
    # The draft is ~500 MiB on disk; the estimate should reflect that order.
    assert after["estimate_mb"]["draft_weights_bytes"] > 300


# ---------------------------------------------------------------------------
# Live drafting: the assertion that matters
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_speculative_decoding_is_actually_enabled(
    raw: httpx.Client, client: Any, live_server: ServerHandle, restore_settings: Any
) -> None:
    """llama-server must report an armed speculative slot, and actually draft.

    If `--spec-type draft-simple` were missing, llama-server would load the
    draft model and never use it -- indistinguishable from success without this
    check.
    """
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    draft = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(target)

    raw.post(f"/api/models/{target}/unload")
    response = raw.put(
        f"/api/models/{target}/settings",
        json={"draft_model_id": draft, "ctx_size": 4096},
    )
    assert response.status_code == 200

    plan = raw.get(f"/api/models/{target}/plan", params={"ctx_size": 4096}).json()
    if not plan.get("fits"):
        pytest.skip(f"target+draft does not fit right now: {plan.get('message')}")

    load = raw.post(f"/api/models/{target}/load", json={"ctx_size": 4096})
    assert load.status_code == 200, load.text

    introspect = raw.get(f"/api/models/{target}/introspect").json()
    assert introspect["loaded"] is True
    assert introspect["actual"].get("speculative") is True, (
        "llama-server does not report a speculative slot; --spec-type is "
        "probably missing or the draft pair was rejected at load"
    )

    completion = client.chat.completions.create(
        model=target,
        messages=[{"role": "user", "content": "Count from one to ten in words."}],
        max_tokens=96,
        temperature=0.0,
    )
    assert completion.choices[0].message.content

    # And prove drafting actually ran for a request, with an acceptance rate.
    test = raw.post(f"/api/models/{target}/test", json={"prompt": "Count to twenty."})
    assert test.status_code == 200, test.text
    stats = test.json()
    assert stats["speculative_used"] is True, stats
    assert stats["draft_tokens"] > 0
    assert 0.0 <= stats["draft_acceptance_rate"] <= 1.0
    print(
        f"\n  draft acceptance: {stats['draft_tokens_accepted']}/"
        f"{stats['draft_tokens']} = {stats['draft_acceptance_rate']:.1%}"
        f"  ({stats['tokens_per_second']} tok/s)"
    )


@pytest.mark.timeout(1800)
def test_draft_ab_comparison_reports_throughput(
    raw: httpx.Client, client: Any, live_server: ServerHandle, restore_settings: Any
) -> None:
    """A/B the same prompt with and without a draft and print tok/s.

    A bad pairing can be SLOWER than no draft at all, so the numbers are the
    point -- this is the data the GUI's Test action surfaces.
    """
    target = resolve_or_skip(live_server, TARGET_NEEDLE)
    draft = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(target)

    prompt = (
        "Write a single paragraph explaining why prompt caching helps agent workloads. Be concrete."
    )

    def measure(label: str) -> float:
        raw.post(f"/api/models/{target}/unload")
        load = raw.post(f"/api/models/{target}/load", json={"ctx_size": 4096})
        if load.status_code != 200:
            pytest.skip(f"{label}: could not load ({load.text[:200]})")
        # Warm the instance so the first-token cost is not counted as throughput.
        client.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=8,
        )
        started = time.perf_counter()
        completion = client.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.0,
        )
        elapsed = time.perf_counter() - started
        tokens = completion.usage.completion_tokens
        tps = tokens / elapsed if elapsed > 0 else 0.0
        print(f"  {label:<18} {tokens:4d} tok in {elapsed:6.2f}s = {tps:6.2f} tok/s")
        return tps

    print("\nSpeculative decoding A/B:")
    raw.put(f"/api/models/{target}/settings", json={"draft_model_id": None})
    baseline = measure("no draft")
    raw.put(f"/api/models/{target}/settings", json={"draft_model_id": draft})
    with_draft = measure("with draft")

    assert baseline > 0 and with_draft > 0
    ratio = with_draft / baseline
    print(f"  ratio: {ratio:.2f}x  ({'faster' if ratio > 1 else 'SLOWER'} with draft)")
    # Deliberately not asserting a speedup: whether drafting wins depends on the
    # pair and the prompt, and asserting it would make this a flaky benchmark.
    # The value is the printed comparison plus proof both configurations serve.


# ---------------------------------------------------------------------------
# LoRA adapters and virtual models
# ---------------------------------------------------------------------------


def test_adapters_endpoint_lists_registry_adapters(raw: httpx.Client) -> None:
    response = raw.get("/api/adapters")
    assert response.status_code == 200
    assert isinstance(response.json()["adapters"], list)


def test_virtual_model_requires_existing_base(raw: httpx.Client) -> None:
    response = raw.post(
        "/api/virtual-models",
        json={"id": "test/virtual", "base_model_id": "no/such-base", "adapters": []},
    )
    assert response.status_code in {400, 404}


def test_virtual_model_requires_id_and_base(raw: httpx.Client) -> None:
    response = raw.post("/api/virtual-models", json={"id": "only-id"})
    assert response.status_code == 400
    assert "required" in response.json()["error"]["message"]


def test_virtual_model_appears_in_v1_models(raw: httpx.Client, live_server: ServerHandle) -> None:
    """A virtual model is how a client selects an adapter set -- by name.

    With no adapters it is still a valid alias of the base, which is enough to
    prove the OpenAI-surface plumbing without needing a GGUF LoRA on disk.
    """
    base = resolve_or_skip(live_server, DRAFT_NEEDLE)  # the tiny model
    virtual_id = "studioforge-test/virtual-alias"
    raw.delete(f"/api/virtual-models/{virtual_id}")
    try:
        created = raw.post(
            "/api/virtual-models",
            json={"id": virtual_id, "base_model_id": base, "adapters": []},
        )
        assert created.status_code == 200, created.text
        payload = created.json()
        assert payload["is_virtual"] is True
        assert payload["base_model_id"] == base

        listed = {m["id"] for m in raw.get("/v1/models").json()["data"]}
        assert virtual_id in listed

        detail = raw.get(f"/v1/models/{virtual_id}").json()
        assert detail["studioforge"]["is_virtual"] is True
    finally:
        raw.delete(f"/api/virtual-models/{virtual_id}")


def test_virtual_model_id_cannot_collide_with_a_real_model(
    raw: httpx.Client, live_server: ServerHandle
) -> None:
    base = resolve_or_skip(live_server, DRAFT_NEEDLE)
    response = raw.post(
        "/api/virtual-models",
        json={"id": base, "base_model_id": base, "adapters": []},
    )
    assert response.status_code == 400


def test_adapter_scale_endpoint_reports_reload_requirement(
    raw: httpx.Client, live_server: ServerHandle
) -> None:
    """Scale changes prefer llama-server's runtime endpoint over a reload."""
    model = resolve_or_skip(live_server, DRAFT_NEEDLE)
    raw.post(f"/api/models/{model}/load", json={"ctx_size": 2048})
    response = raw.post(f"/api/models/{model}/adapter-scales", json={"scales": []})
    assert response.status_code == 200
    body = response.json()
    assert "applied" in body and "reload_required" in body


def test_extra_flags_validation_rejects_removed_draft_flag(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    """The removed --draft-max must be a save-time error, not a silent no-op."""
    model = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(model)
    response = raw.put(f"/api/models/{model}/settings", json={"extra_flags": "--draft-max 4"})
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "--spec-draft-n-max" in message


def test_extra_flags_validation_rejects_manager_owned_flags(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    """--fit would reintroduce a silent partial-offload path (DECISIONS D11)."""
    model = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(model)
    for flag in ("--port 9999", "--n-gpu-layers 10", "--fit on"):
        response = raw.put(f"/api/models/{model}/settings", json={"extra_flags": flag})
        assert response.status_code == 400, f"{flag} should have been refused"
        assert "managed by StudioForge" in response.json()["error"]["message"]


def test_extra_flags_accepts_a_real_flag(
    raw: httpx.Client, live_server: ServerHandle, restore_settings: Any
) -> None:
    model = resolve_or_skip(live_server, DRAFT_NEEDLE)
    restore_settings(model)
    response = raw.put(f"/api/models/{model}/settings", json={"extra_flags": "--top-k 20"})
    assert response.status_code == 200
