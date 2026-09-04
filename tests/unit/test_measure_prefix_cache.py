"""The pure half of ``scripts/measure_prefix_cache.py`` (D54).

The script talks to a live rig, so only its network-free helpers are tested:
argument parsing, the refusal logic against fixture JSON, prompt construction,
row extraction, phase arithmetic and the report. It is imported by path
because it deliberately has no package -- it must run against a server this
build does not control.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "measure_prefix_cache.py"


@pytest.fixture(scope="module")
def mpc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("measure_prefix_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL = "vendor/Dark-Scarlett-27B-Q5_K_M"


def status_with(
    model: str = MODEL, state: str = "ready", parallel: int = 3, **extra: object
) -> dict:
    row: dict = {
        "model_id": model,
        "state": state,
        "port": 18100,
        "plan": {"parallel": parallel, "ctx_size": 131072},
        "effective": {"parallel": parallel, "summary": "prefix cache on (reuse 256)"},
    }
    row.update(extra)
    return {"loaded": [row], "benchmark": None, "leases": []}


IDLE_HEALTH = {"busy": {"active_requests": 0}}


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------


def test_parse_args_defaults_match_the_brief(mpc: ModuleType) -> None:
    args = mpc.parse_args(["--model", MODEL])
    assert args.base == "http://127.0.0.1:1234"
    assert args.prefix_tokens == 8000 and args.tail_tokens == 1200
    assert args.requests == 7 and args.concurrency == 3 and args.pairs == 2
    assert args.max_tokens == 64
    assert args.diverge == "user-boundary"
    assert args.warm_first is False
    assert args.client_label == "measure_prefix_cache"
    assert args.yes is False and args.json_out is None


def test_parse_args_accepts_yes_and_force_as_the_same_override(mpc: ModuleType) -> None:
    assert mpc.parse_args(["--model", MODEL, "--yes"]).yes is True
    assert mpc.parse_args(["--model", MODEL, "--force"]).yes is True
    assert mpc.parse_args(["--model", MODEL, "--diverge", "mid-message"]).diverge == "mid-message"


def test_parse_args_requires_a_model_and_sane_counts(mpc: ModuleType) -> None:
    with pytest.raises(SystemExit):
        mpc.parse_args([])
    with pytest.raises(SystemExit):
        mpc.parse_args(["--model", MODEL, "--concurrency", "0"])
    with pytest.raises(SystemExit):
        mpc.parse_args(["--model", MODEL, "--diverge", "somewhere"])


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------


def test_an_idle_rig_with_the_model_ready_is_not_refused(mpc: ModuleType) -> None:
    assert mpc.refusal_reason(status_with(), [], IDLE_HEALTH, MODEL, 3) is None


def test_a_foreign_lease_refuses_the_run(mpc: ModuleType) -> None:
    leases = [{"id": "l1", "holder": "crucibleforge", "devices": [0, 1, 2, 3]}]
    reason = mpc.refusal_reason(status_with(), leases, IDLE_HEALTH, MODEL, 3)
    assert reason is not None and "crucibleforge" in reason and "--yes" in reason
    # The /api/leases envelope shape is accepted too.
    assert mpc.refusal_reason(status_with(), {"leases": leases}, IDLE_HEALTH, MODEL, 3)


def test_active_requests_refuse_the_run(mpc: ModuleType) -> None:
    busy = {"busy": {"active_requests": 8}}
    reason = mpc.refusal_reason(status_with(), [], busy, MODEL, 3)
    assert reason is not None and "8 request" in reason


def test_a_running_benchmark_refuses_the_run(mpc: ModuleType) -> None:
    status = status_with()
    status["benchmark"] = {"mode": "parallel", "model_id": "other"}
    reason = mpc.refusal_reason(status, [], IDLE_HEALTH, MODEL, 3)
    assert reason is not None and "benchmark" in reason


def test_yes_overrides_only_the_idle_checks(mpc: ModuleType) -> None:
    leases = [{"id": "l1", "holder": "someone", "devices": [0]}]
    busy = {"busy": {"active_requests": 2}}
    assert mpc.refusal_reason(status_with(), leases, busy, MODEL, 3, yes=True) is None
    # Not loaded: never overridden -- the script must not cause a load.
    cold = {"loaded": [], "benchmark": None}
    reason = mpc.refusal_reason(cold, [], IDLE_HEALTH, MODEL, 3, yes=True)
    assert reason is not None and "not loaded" in reason and "load_recommended" in reason
    # Loading, not ready: the same.
    loading = mpc.refusal_reason(status_with(state="loading"), [], IDLE_HEALTH, MODEL, 3, yes=True)
    assert loading is not None and "state=loading" in loading


def test_concurrency_above_the_instance_slots_is_refused_even_with_yes(mpc: ModuleType) -> None:
    reason = mpc.refusal_reason(status_with(parallel=3), [], IDLE_HEALTH, MODEL, 4, yes=True)
    assert reason is not None and "parallel=3" in reason
    # The slot count comes from `effective` first, the plan when that is absent.
    pre_d54 = status_with(parallel=2, effective=None)
    assert mpc.instance_parallel(pre_d54["loaded"][0]) == 2
    assert mpc.refusal_reason(pre_d54, [], IDLE_HEALTH, MODEL, 2) is None


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def test_prose_is_deterministic_and_never_repeats_a_sentence(mpc: ModuleType) -> None:
    a = mpc.build_prose(400, seed=7)
    assert a == mpc.build_prose(400, seed=7)
    assert a != mpc.build_prose(400, seed=8)
    sentences = [s for s in a.split(". ") if s]
    assert len(sentences) == len(set(sentences)), "a repeated sentence is a cache hit in disguise"
    assert len(a.split()) >= 400


def test_user_boundary_puts_the_difference_in_its_own_last_user_message(mpc: ModuleType) -> None:
    messages = mpc.build_messages("BIBLE", "the tail", 3)
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    assert messages[1]["content"].endswith("BIBLE")
    assert messages[2]["content"] == "Chapter 3: the tail"
    # The shared part is byte-identical across requests.
    other = mpc.build_messages("BIBLE", "another tail", 4)
    assert other[:2] == messages[:2]


def test_mid_message_buries_the_difference_inside_the_shared_message(mpc: ModuleType) -> None:
    messages = mpc.build_messages("BIBLE", "the tail", 3, diverge="mid-message")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "BIBLE" in messages[1]["content"] and "Chapter 3: the tail" in messages[1]["content"]


# ---------------------------------------------------------------------------
# rows, phases, report
# ---------------------------------------------------------------------------


def test_parse_metrics_keeps_only_the_counters_of_interest(mpc: ModuleType) -> None:
    text = (
        "# HELP x\nllamacpp:prompt_tokens_total 100\nllamacpp:prompt_tokens_cached_total 900\n"
        "llamacpp:n_decode_total 40\nllamacpp:tokens_predicted_total 120\nllamacpp:other 1\n"
    )
    parsed = mpc.parse_metrics(text)
    assert parsed == {
        mpc.METRIC_PROMPT: 100.0,
        mpc.METRIC_PROMPT_CACHED: 900.0,
        mpc.METRIC_DECODE: 40.0,
        mpc.METRIC_PREDICTED: 120.0,
    }


def test_row_from_response_reads_the_truthful_signals(mpc: ModuleType) -> None:
    body = {
        "usage": {"prompt_tokens": 9800, "prompt_tokens_details": {"cached_tokens": 8400}},
        "timings": {
            "cache_n": 8400,
            "prompt_n": 1400,
            "prompt_ms": 812.5,
            "predicted_n": 64,
            "predicted_per_second": 48.2,
        },
    }
    row = mpc.row_from_response(4, "concurrent#1", 2.3456, body)
    assert row["prompt_tokens"] == 9800 and row["cached_tokens"] == 8400
    assert row["cache_n"] == 8400 and row["prompt_n"] == 1400
    assert row["predicted_n"] == 64 and row["wall_s"] == 2.346
    assert row["draft_n"] is None and row["error"] is None
    errored = mpc.row_from_response(1, "serial#1", 0.1, {"error": {"code": "x"}})
    assert errored["error"] == {"code": "x"} and errored["prompt_tokens"] is None


def test_summarize_phase_reports_the_no_cross_slot_sharing_arithmetic(mpc: ModuleType) -> None:
    """R=7 requests, N=3 slots, P=9800, s=0.86: the first three prefill cold,
    the rest reuse ~8400 -- total prefill ~3.6 P, hit ratio ~0.49 of the
    tokens seen, and sum prompt_tokens is 68,600 regardless."""
    rows = []
    for i in range(1, 8):
        cold = i <= 3
        rows.append(
            mpc.row_from_response(
                i,
                "concurrent#1",
                5.0,
                {
                    "usage": {"prompt_tokens": 9800},
                    "timings": {
                        "cache_n": 0 if cold else 8400,
                        "prompt_n": 9800 if cold else 1400,
                        "predicted_n": 64,
                    },
                },
            )
        )
    before = {mpc.METRIC_DECODE: 1000.0, mpc.METRIC_PROMPT_CACHED: 0.0}
    after = {mpc.METRIC_DECODE: 1000.0 + 448 / 2.8, mpc.METRIC_PROMPT_CACHED: 33600.0}
    summary = mpc.summarize_phase(rows, before, after, wall_s=20.0)
    assert summary["sum_prompt_tokens"] == 68600
    assert summary["sum_prompt_n"] == 3 * 9800 + 4 * 1400  # 35000 ~ 3.6 P
    assert summary["sum_cache_n"] == 4 * 8400
    assert summary["hit_ratio"] == pytest.approx(33600 / 68600, abs=1e-4)
    assert summary["sum_predicted_n"] == 448
    assert summary["aggregate_gen_tps"] == pytest.approx(22.4)
    assert summary["achieved_batch"] == pytest.approx(2.8, abs=1e-3)
    assert summary["child_cached_delta"] == 33600
    assert summary["errors"] == 0


def test_summarize_phase_without_the_cached_counter_says_so(mpc: ModuleType) -> None:
    rows = [mpc.row_from_response(1, "serial#1", 1.0, {"usage": {"prompt_tokens": 5}})]
    summary = mpc.summarize_phase(rows, {}, {}, wall_s=0.0)
    assert summary["child_cached_delta"] is None, "no cached counter on this engine"
    assert summary["hit_ratio"] is None, "nothing seen yet"
    assert summary["achieved_batch"] is None, "no decode delta"
    assert summary["aggregate_gen_tps"] is None


def test_the_report_prints_the_effective_block_and_the_expected_shape(mpc: ModuleType) -> None:
    rows = [
        mpc.row_from_response(
            1, "serial#1", 1.0, {"usage": {"prompt_tokens": 100}, "timings": {"cache_n": 0}}
        )
    ]
    summary = mpc.summarize_phase(rows, {}, {}, wall_s=1.0)
    text = mpc.format_report(
        MODEL,
        {"summary": "prefix cache on (reuse 256, host 32603 MiB, routing 0.3)"},
        [("serial#1", summary, rows), ("concurrent#1", summary, rows)],
        concurrency=3,
        diverge="user-boundary",
    )
    assert "effective: prefix cache on (reuse 256" in text
    assert "first request cache_n 0" in text
    assert "the first 3 requests cache_n 0 (no cross-slot sharing)" in text
    assert "never usage.prompt_tokens" in text
    cliff = mpc.format_report(MODEL, None, [], concurrency=3, diverge="mid-message")
    assert "pre-D54" in cliff
    # The mid-message shape is named per phase, not buried in the footer.
    assert "checkpoint" in mpc.expected_shape("serial#1", 3, "mid-message")


def test_exit_codes_are_the_documented_ones(mpc: ModuleType) -> None:
    assert (mpc.EXIT_OK, mpc.EXIT_REFUSED, mpc.EXIT_REQUEST_ERROR) == (0, 2, 3)
