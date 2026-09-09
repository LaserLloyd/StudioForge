"""Pure-function tests for bench_stats. No network, no StudioForge imports."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import bench_stats as bs  # noqa: E402

MIB = 1024 * 1024


def _run(length: int, run: int, prefill: float, decode: float, **extra: object) -> dict:
    row = {
        "prompt_length": length,
        "run": run,
        "warmup": run == 0,
        "prefill_tps": prefill,
        "decode_tps": decode,
        "ttft_s": 1.0 + run,
        "wall_s": 10.0 + run,
        "prompt_n": length - 100,
        "peak_vram_bytes": {"0": 1000 + run, "1": 2000 + run},
    }
    row.update(extra)
    return row


def _summary(
    prefill: float, decode: float, lengths: tuple[int, ...] = (1024, 131072, 262144)
) -> dict:
    runs = [_run(n, i, prefill, decode) for n in lengths for i in range(6)]
    return bs.summarize_runs(runs)


def test_median_and_p95_nearest_rank() -> None:
    assert bs.median([]) is None
    assert bs.median([3, 1, 2]) == 2
    assert bs.median([4, 1, 3, 2]) == 2.5
    assert bs.p95([]) is None
    assert bs.p95([5, 1, 4, 2, 3]) == 5  # ceil(0.95*5)=5 -> 5th of sorted
    assert bs.p95(list(range(1, 21))) == 19  # ceil(19.0)=19 -> 19th value
    assert bs.p95([7]) == 7


def test_delta_pct() -> None:
    assert bs.delta_pct(110, 100) == pytest.approx(10.0)
    assert bs.delta_pct(90, 100) == pytest.approx(-10.0)
    assert bs.delta_pct(None, 100) is None
    assert bs.delta_pct(5, 0) is None


def test_summarize_runs_discards_warmup_and_errors() -> None:
    runs = [_run(1024, i, 100 + i, 20 + i) for i in range(6)]
    runs.append(_run(1024, 6, 9999, 9999, error="boom"))
    runs.append(_run(1024, 7, None, 30, cache_hit_suspected=True))
    summary = bs.summarize_runs(runs)
    entry = summary["1024"]
    assert entry["runs_measured"] == 6 and entry["runs_total"] == 8
    assert entry["prefill_tps"]["n"] == 5  # run 0 (warm-up), the error and the None dropped
    assert entry["prefill_tps"]["median"] == 103
    assert entry["decode_tps"]["n"] == 6 and entry["decode_tps"]["median"] == 23.5
    assert entry["peak_vram_bytes"] == {"0": 1007, "1": 2007}
    assert entry["cache_hit_suspected_runs"] == 1


def test_holder_on_devices_prefers_per_gpu_bytes() -> None:
    ctx_only = {
        "per_gpu_bytes": {"0": 430 * MIB, "1": 430 * MIB, "2": 20_000 * MIB},
        "used_bytes": 21_000 * MIB,
    }
    assert bs.holder_on_devices(ctx_only, [0, 1]) == (False, "per_gpu_bytes")
    assert bs.holder_on_devices(ctx_only, [2]) == (True, "per_gpu_bytes")
    nvml = {
        "gpu_indices": [0, 1, 2, 3],
        "gpu_indices_source": "nvml-context",
        "used_bytes": 3000 * MIB,
    }
    assert bs.holder_on_devices(nvml, [1]) == (True, "nvml-context")
    small = {"gpu_indices": [0], "used_bytes": 100 * MIB}
    assert bs.holder_on_devices(small, [0])[0] is False
    assert bs.holder_on_devices({"used_bytes": 900 * MIB}, [0]) == (True, "unattributed")
    assert bs.holder_on_devices({"used_bytes": 10 * MIB}, [0]) == (False, "unattributed")


def test_classify_validity() -> None:
    payload = {
        "holders": [
            {
                "pid": 11,
                "name": "llama-server.exe",
                "classification": "ours",
                "used_bytes": 40_000 * MIB,
                "per_gpu_bytes": {"0": 20_000 * MIB, "1": 20_000 * MIB},
            },
            {
                "pid": 22,
                "name": "llama-server.exe",
                "classification": "ours",
                "used_bytes": 6000 * MIB,
                "per_gpu_bytes": {"0": 400 * MIB, "2": 6000 * MIB},
            },
        ],
        "desktop_processes_count": 17,
        "desktop_processes_bytes": 900 * MIB,
        "per_gpu_bytes_source": "pdh",
    }
    ok = bs.classify_validity(payload, [0, 1], own_pids=[11])
    assert ok["valid"] is True and ok["own"][0]["pid"] == 11 and ok["offending"] == []
    assert ok["desktop_processes_count"] == 17
    payload["holders"].append(
        {
            "pid": 33,
            "name": "python.exe",
            "classification": "foreign",
            "used_bytes": 3000 * MIB,
            "per_gpu_bytes": {"1": 3000 * MIB},
        }
    )
    bad = bs.classify_validity(payload, [0, 1], own_pids=[11])
    assert bad["valid"] is False and [h["pid"] for h in bad["offending"]] == [33]
    assert bad["offending"][0]["attribution"] == "per_gpu_bytes"
    assert bs.classify_validity({"holders": []}, [0, 1], own_pids=[])["valid"] is True


def test_verdict_rule() -> None:
    old = _summary(prefill=1000.0, decode=20.0)
    assert bs.verdict(_summary(1000.0, 22.5), old)["verdict"] == "major-bar-met"  # decode +12.5%
    assert bs.verdict(_summary(1120.0, 20.0), old)["verdict"] == "major-bar-met"  # prefill +12%
    assert (
        bs.verdict(_summary(970.0, 22.5), old)["verdict"] == "regression"
    )  # -3% prefill blocks it
    assert bs.verdict(_summary(1010.0, 20.4), old)["verdict"] == "keep"  # small win is kept
    assert bs.verdict(_summary(990.0, 19.8), old)["verdict"] == "neutral"  # within tolerance
    assert bs.verdict(_summary(1000.0, 19.0), old)["verdict"] == "regression"  # decode -5%
    smoke = _summary(1000.0, 20.0, lengths=(1024,))
    assert bs.verdict(smoke, old)["verdict"] == "no-verdict"
    tagged = bs.verdict(_summary(1000.0, 22.5), old, new_valid=False)
    assert tagged["verdict"] == "invalid:major-bar-met" and tagged["line"].startswith("INVALID")
    line = bs.verdict(_summary(1000.0, 22.5), old)["line"]
    assert "262144" in line and "131072" in line
    # The prefill headline is the WORST bucket >= 128k: a +15% at 256k does not
    # hide a -3% at 128k.
    mixed = _summary(1150.0, 20.0)
    mixed["131072"]["prefill_tps"]["median"] = 970.0
    graded = bs.verdict(mixed, old)
    assert graded["verdict"] == "regression"
    assert graded["prefill_deltas_pct"]["131072"] == pytest.approx(-3.0)
    assert graded["prefill_deltas_pct"]["262144"] == pytest.approx(15.0)


def test_build_prompt_is_deterministic_and_prefix_stable() -> None:
    short = bs.build_prompt(200, seed=42)
    long = bs.build_prompt(400, seed=42)
    assert short == bs.build_prompt(200, seed=42)
    assert long.startswith(
        short[: len(short) - 3]
    )  # same stream, only the tail punctuation may differ
    assert len(long.split()) == 400
    assert bs.build_prompt(50, seed=1) != bs.build_prompt(50, seed=2)
    assert bs.build_prompt(10, seed=3, header="[run 1] ").startswith("[run 1] ")
    assert "\n\n" in bs.build_prompt(2000, seed=7)


def test_prompt_sizing_helpers() -> None:
    assert bs.estimate_words(1300, 1.3) == 1000
    assert bs.clamp_prompt_length(262144, 262144, 256) == 262144 - 256 - 64
    assert bs.clamp_prompt_length(1024, 262144, 256) == 1024
    assert bs.clamp_prompt_length(1024, 100, 256) == 1
    assert bs.parse_int_list("0,1") == [0, 1]
    assert bs.parse_int_list("1024, 32768") == [1024, 32768]
    with pytest.raises(ValueError):
        bs.parse_int_list("")


def test_result_filename_and_plan_tuple() -> None:
    assert (
        bs.result_filename("3afb47ef7ff5557a2cfb3cdaad3632998c58629f", None, False)
        == "3afb47ef7ff5.json"
    )
    assert (
        bs.result_filename("3afb47ef7ff5557a", "flash attn/on", True)
        == "3afb47ef7ff5-flash-attn-on-smoke.json"
    )
    assert bs.result_filename("", None, False) == "nogit.json"
    plan = {
        "ctx_size": 262144,
        "parallel": 1,
        "devices": [1, 0],
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
    }
    assert bs.plan_tuple(plan) == (262144, 1, (0, 1), "f16", "f16")
    assert bs.plan_tuple(None) is None
