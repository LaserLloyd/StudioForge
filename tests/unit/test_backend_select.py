"""Backend preference, the benchmark-result cache, and the (gated) llama-bench runner.

``select_backend`` is pure; the cache is a JSON file in a tmp dir; the runner is
exercised only through an injected fake -- nothing here spawns ``llama-bench``
or touches a GPU.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from studioforge import __main__ as cli
from studioforge.config import Config
from studioforge.core import backend_bench
from studioforge.core.backend_bench import (
    BenchParseError,
    find_bench_binary,
    llama_bench_command,
    parse_llama_bench_json,
    run_backend_benchmark,
)
from studioforge.core.backend_select import (
    PREFERENCES,
    BackendBenchCache,
    BackendBenchResult,
    select_backend,
    select_backend_per_gpu,
)
from studioforge.core.engine import EngineError
from studioforge.core.gpu import reset_probe

FIXTURES = Path(__file__).parent / "fixtures" / "hw"


def r(backend: str, pp: float, tg: float, *, gpu: str = "gfx1151", tag: str = "b1", cls: str = ""):
    return BackendBenchResult(
        backend=backend, gpu=gpu, engine_tag=tag, pp512=pp, tg128=tg, model_class=cls
    )


# Strix-Halo-shaped: HIP wins prompt processing, Vulkan wins generation.
HIP = r("rocm", pp=1000.0, tg=40.0)
VULKAN = r("vulkan", pp=600.0, tg=50.0)


# ---------------------------------------------------------------------------
# select_backend
# ---------------------------------------------------------------------------


def test_preferences_are_the_documented_three() -> None:
    assert PREFERENCES == ("balanced", "fastest_replies", "fastest_long_prompts")
    assert Config().engine.preference == "balanced"


def test_fastest_replies_follows_generation() -> None:
    choice = select_backend([HIP, VULKAN], "fastest_replies")
    assert choice is not None and choice.backend == "vulkan"
    assert choice.scores == {"rocm": 40.0, "vulkan": 50.0}


def test_fastest_long_prompts_follows_prompt_processing() -> None:
    choice = select_backend([HIP, VULKAN], "fastest_long_prompts")
    assert choice is not None and choice.backend == "rocm"


def test_balanced_is_the_geometric_mean_of_relative_speeds() -> None:
    choice = select_backend([HIP, VULKAN], "balanced")
    assert choice is not None
    # hip: sqrt(1.0 * 0.8) = 0.894; vulkan: sqrt(0.6 * 1.0) = 0.775
    assert choice.backend == "rocm"
    assert choice.scores["rocm"] == pytest.approx(0.8944, abs=1e-3)
    assert choice.scores["vulkan"] == pytest.approx(0.7746, abs=1e-3)
    # default preference is balanced
    assert select_backend([HIP, VULKAN]).backend == "rocm"  # type: ignore[union-attr]
    # move the numbers and the balanced winner moves
    assert select_backend([HIP, r("vulkan", 900.0, 55.0)], "balanced").backend == "vulkan"  # type: ignore[union-attr]


def test_single_backend_wins_by_default_and_says_so() -> None:
    choice = select_backend([HIP], "fastest_replies")
    assert choice is not None and choice.backend == "rocm"
    assert "only backend measured" in choice.reason


def test_no_usable_data_selects_nothing() -> None:
    assert select_backend([]) is None
    junk = [r("cuda", 0.0, 10.0), r("vulkan", float("nan"), 5.0), r("rocm", -1.0, -1.0)]
    assert select_backend(junk) is None


def test_unusable_rows_are_ignored_not_fatal() -> None:
    choice = select_backend([r("cuda", 0.0, 0.0), VULKAN], "balanced")
    assert choice is not None and choice.backend == "vulkan"


def test_ties_prefer_the_vendor_native_backend() -> None:
    tied = [r("vulkan", 500.0, 50.0), r("cuda", 500.0, 50.0), r("rocm", 500.0, 50.0)]
    for pref in PREFERENCES:
        assert select_backend(tied, pref).backend == "cuda"  # type: ignore[union-attr]
    assert select_backend([r("vulkan", 500.0, 50.0), r("rocm", 500.0, 50.0)]).backend == "rocm"  # type: ignore[union-attr]


def test_a_backend_measured_on_two_engine_tags_uses_its_best_per_axis() -> None:
    older = r("rocm", 900.0, 38.0, tag="b1")
    newer = r("rocm", 950.0, 36.0, tag="b2")
    object.__setattr__(older, "measured_at", 1.0)
    object.__setattr__(newer, "measured_at", 2.0)
    choice = select_backend([older, newer, VULKAN], "fastest_replies")
    assert choice is not None and choice.backend == "vulkan"  # 50 > max(38, 36)
    rocm = select_backend([older, newer], "fastest_long_prompts")
    assert rocm is not None and rocm.result.pp512 == 950.0 and rocm.result.tg128 == 38.0


def test_unknown_preference_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown engine preference"):
        select_backend([HIP], "fastest")


def test_select_backend_per_gpu_groups_by_device_and_model_class() -> None:
    rows = [
        r("rocm", 1000.0, 40.0, gpu="gfx1151"),
        r("vulkan", 600.0, 50.0, gpu="gfx1151"),
        r("cuda", 4000.0, 150.0, gpu="RTX 5090"),
        r("vulkan", 3000.0, 160.0, gpu="RTX 5090"),
        r("rocm", 1.0, 1.0, gpu="gfx1151", cls="other-model"),
    ]
    picks = select_backend_per_gpu(rows, "fastest_replies")
    assert {gpu: pick.backend for gpu, pick in picks.items()} == {
        "gfx1151": "vulkan",
        "RTX 5090": "vulkan",
    }
    other = select_backend_per_gpu(rows, "balanced", model_class="other-model")
    assert list(other) == ["gfx1151"]


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def test_cache_round_trip_and_atomic_file(tmp_path: Path) -> None:
    cache = BackendBenchCache.for_engines_dir(tmp_path / "engines")
    assert len(cache) == 0
    cache.put(HIP)
    cache.put(VULKAN)
    assert (tmp_path / "engines" / "backend-bench.json").is_file()
    assert not list((tmp_path / "engines").glob("*.tmp"))

    reread = BackendBenchCache.for_engines_dir(tmp_path / "engines")
    assert len(reread) == 2
    assert reread.choose("gfx1151", "fastest_replies").backend == "vulkan"  # type: ignore[union-attr]
    assert reread.choose("other-gpu") is None
    stamped = reread.results(gpu="gfx1151")
    assert all(item.measured_at > 0 for item in stamped)  # put() stamps the time


def test_cache_replaces_the_same_key_and_filters_by_tag_and_age(tmp_path: Path) -> None:
    cache = BackendBenchCache(tmp_path / "c.json")
    cache.put(r("rocm", 1.0, 1.0, tag="b1"))
    cache.put(r("rocm", 1000.0, 40.0, tag="b1"))  # same key: replaced
    cache.put(r("rocm", 2000.0, 80.0, tag="b2"))
    assert len(cache) == 2
    assert [x.pp512 for x in cache.results(engine_tag="b1")] == [1000.0]
    # a new engine tag invalidates by omission: asking about b3 finds nothing
    assert cache.results(engine_tag="b3") == []
    old = BackendBenchResult("vulkan", "gfx1151", "b1", 5.0, 5.0, measured_at=time.time() - 10_000)
    cache.put(old)
    assert old not in cache.results(max_age_s=3600)
    assert old in cache.results()


def test_a_corrupt_cache_file_reads_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    for content in ("not json", '{"results": 5}', '{"results": [1, {"backend": "x"}]}', ""):
        path.write_text(content, encoding="utf-8")
        assert len(BackendBenchCache(path)) == 0
    missing = BackendBenchCache(tmp_path / "does-not-exist.json")
    assert len(missing) == 0
    missing.put(HIP)  # and it can still be written
    assert len(BackendBenchCache(tmp_path / "does-not-exist.json")) == 1


# ---------------------------------------------------------------------------
# llama-bench: argv + parser + the confirm gate
# ---------------------------------------------------------------------------


def test_llama_bench_command_is_strictly_gpu_resident(tmp_path: Path) -> None:
    argv = llama_bench_command(tmp_path / "llama-bench", tmp_path / "m.gguf")
    assert argv[argv.index("-ngl") + 1] == "999"
    assert argv[argv.index("-p") + 1] == "512" and argv[argv.index("-n") + 1] == "128"
    assert argv[argv.index("-o") + 1] == "json"
    assert "--device" not in argv
    with_device = llama_bench_command(
        tmp_path / "llama-bench", tmp_path / "m.gguf", device="Vulkan0", repetitions=0
    )
    assert with_device[with_device.index("--device") + 1] == "Vulkan0"
    assert with_device[with_device.index("-r") + 1] == "1"  # clamped


def test_parse_llama_bench_json_fixture() -> None:
    text = (FIXTURES / "llama_bench_pp512_tg128.json").read_text(encoding="utf-8")
    assert parse_llama_bench_json("ggml_vulkan: found 1 device\n" + text) == (800.25, 40.0)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no json",
        "[not json",
        "[]",
        '[{"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0}]',  # no tg row
        '[{"n_prompt": 0, "n_gen": 128, "avg_ts": 0}]',
        '{"n_prompt": 512}',
    ],
)
def test_parse_llama_bench_json_rejects_incomplete_output(text: str) -> None:
    with pytest.raises(BenchParseError):
        parse_llama_bench_json(text)


def test_find_bench_binary(tmp_path: Path) -> None:
    server = tmp_path / "llama-server"
    server.write_bytes(b"x")
    assert find_bench_binary(server) is None  # a source build ships none
    (tmp_path / backend_bench.BENCH_NAME).write_bytes(b"x")
    assert find_bench_binary(server) == tmp_path / backend_bench.BENCH_NAME


def test_the_runner_refuses_without_explicit_confirmation(tmp_path: Path) -> None:
    called: list[Sequence[str]] = []

    async def runner(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
        called.append(argv)
        return 0, ""

    with pytest.raises(EngineError) as excinfo:
        asyncio.run(
            run_backend_benchmark(
                tmp_path / "llama-bench",
                tmp_path / "m.gguf",
                backend="rocm",
                gpu="gfx1151",
                engine_tag="b1",
                runner=runner,
            )
        )
    assert excinfo.value.code == "engine_benchmark_unconfirmed"
    assert called == []


def test_the_runner_measures_through_the_injected_runner(tmp_path: Path) -> None:
    text = (FIXTURES / "llama_bench_pp512_tg128.json").read_text(encoding="utf-8")
    seen: list[Sequence[str]] = []

    async def runner(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
        seen.append(argv)
        return 0, text

    result = asyncio.run(
        run_backend_benchmark(
            tmp_path / "llama-bench",
            tmp_path / "m.gguf",
            backend="rocm",
            gpu="gfx1151",
            engine_tag="b10425",
            model_class="8b-q4",
            confirm=True,
            runner=runner,
        )
    )
    assert (result.backend, result.gpu, result.pp512, result.tg128) == (
        "rocm",
        "gfx1151",
        800.25,
        40.0,
    )
    assert result.model_class == "8b-q4" and result.measured_at > 0
    assert seen and seen[0][seen[0].index("-ngl") + 1] == "999"


@pytest.mark.parametrize(
    ("code", "out", "message"),
    [(124, "", "did not finish"), (1, "", "exited 1"), (0, "garbage", "no JSON array")],
)
def test_the_runner_reports_failures_as_engine_errors(
    code: int, out: str, message: str, tmp_path: Path
) -> None:
    async def runner(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
        return code, out

    with pytest.raises(EngineError, match=message):
        asyncio.run(
            run_backend_benchmark(
                tmp_path / "b",
                tmp_path / "m",
                backend="cuda",
                gpu="x",
                engine_tag="t",
                confirm=True,
                runner=runner,
            )
        )


# ---------------------------------------------------------------------------
# CLI: engine list / select / benchmark
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config file whose data dir is a tmp dir, with no GPU probe and no log files."""
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    monkeypatch.setattr(cli, "configure_logging", lambda *a, **k: None)
    reset_probe()
    monkeypatch.setenv("SF_DATA_DIR", str(tmp_path / "data"))
    path = tmp_path / "config.yaml"
    path.write_text("{}\n", encoding="utf-8")
    yield path
    reset_probe()


def _install_engine(config_path: Path, **meta: object) -> Path:
    config = Config(data_dir=config_path.parent / "data")
    directory = config.engines_dir / "b10425"
    directory.mkdir(parents=True)
    (directory / cli_bin_name()).write_bytes(b"x")
    payload = {"tag": "b10425", "variant": "cuda-13.3", "installed_at": 1_790_000_000.0, **meta}
    (directory / "engine.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def cli_bin_name() -> str:
    from studioforge.core.engine import BIN_NAME

    return BIN_NAME


def test_engine_list_prints_backend_verified_and_smoke(cli_config: Path) -> None:
    _install_engine(
        cli_config,
        backend="cuda",
        verified=False,
        verified_by="none: the release publishes no checksum for this archive",
        smoke_result="passed",
        smoke_tested=True,
        warnings=["llama-b10425-bin-win-cuda-13.3-x64.zip was installed WITHOUT checksum"],
    )
    result = CliRunner().invoke(
        cli.app, ["engine", "list", "--config", str(cli_config)], catch_exceptions=False
    )
    assert result.exit_code == 0
    header, row, warning = result.output.strip().splitlines()[:3]
    assert "BACKEND" in header and "VERIFIED" in header and "SMOKE" in header
    assert "b10425" in row and "cuda" in row and "NO" in row and "passed" in row
    assert warning.strip().startswith("!") and "WITHOUT checksum" in warning


def test_engine_list_json_and_empty(cli_config: Path) -> None:
    empty = CliRunner().invoke(
        cli.app, ["engine", "list", "--config", str(cli_config)], catch_exceptions=False
    )
    assert "no engines installed" in empty.output
    _install_engine(cli_config, backend="cuda", verified=True, verified_by="github-digest")
    result = CliRunner().invoke(
        cli.app, ["engine", "list", "--json", "-c", str(cli_config)], catch_exceptions=False
    )
    (record,) = json.loads(result.output)
    assert record["tag"] == "b10425" and record["verified"] is True
    assert record["backend"] == "cuda" and record["smoke_result"] == "not_run"


def test_engine_list_marks_legacy_and_source_builds(cli_config: Path) -> None:
    _install_engine(cli_config)  # engine.json without any verification fields
    out = CliRunner().invoke(cli.app, ["engine", "list", "-c", str(cli_config)]).output
    assert "unknown" in out
    config = Config(data_dir=cli_config.parent / "data")
    built = config.engines_dir / "b10425-local"
    built.mkdir()
    (built / cli_bin_name()).write_bytes(b"x")
    (built / "engine.json").write_text(
        json.dumps(
            {
                "variant": "source-local",
                "backend": "rocm",
                "verified": False,
                "verified_by": "source-build (rocm) from git tag b10425",
                "installed_at": 1_790_000_100.0,
            }
        ),
        encoding="utf-8",
    )
    out = CliRunner().invoke(cli.app, ["engine", "list", "-c", str(cli_config)]).output
    row = next(line for line in out.splitlines() if "b10425-local" in line)
    assert "rocm" in row and "source" in row


def test_the_original_flag_interface_still_works(cli_config: Path) -> None:
    """`studioforge engine --list` etc. predate the subcommands and must not break."""
    result = CliRunner().invoke(cli.app, ["engine", "--help"])
    assert result.exit_code == 0
    for flag in ("--install", "--activate", "--smoke-test", "--check", "--update", "--list"):
        assert flag in result.output
    for sub in ("list", "install", "update", "select", "benchmark"):
        assert sub in result.output


def test_engine_select_reports_the_cached_recommendation(cli_config: Path) -> None:
    none = CliRunner().invoke(cli.app, ["engine", "select", "-c", str(cli_config)])
    assert "no benchmark results cached" in none.output
    cache = BackendBenchCache.for_engines_dir(
        Config(data_dir=cli_config.parent / "data").engines_dir
    )
    cache.put(HIP)
    cache.put(VULKAN)
    default = CliRunner().invoke(cli.app, ["engine", "select", "-c", str(cli_config)])
    assert "gfx1151: balanced: rocm" in default.output
    replies = CliRunner().invoke(
        cli.app, ["engine", "select", "-c", str(cli_config), "--preference", "fastest_replies"]
    )
    assert "fastest_replies: vulkan" in replies.output
    bad = CliRunner().invoke(
        cli.app, ["engine", "select", "-c", str(cli_config), "--preference", "nope"]
    )
    assert bad.exit_code == 2


def test_engine_benchmark_refuses_without_yes_and_never_reaches_the_runner(
    cli_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: object, **k: object) -> None:  # pragma: no cover - must not run
        raise AssertionError("the benchmark ran without --yes")

    monkeypatch.setattr(backend_bench, "run_backend_benchmark", boom)
    result = CliRunner().invoke(
        cli.app, ["engine", "benchmark", "--model", "x.gguf", "-c", str(cli_config)]
    )
    assert result.exit_code == 2
    assert "--yes" in result.output
