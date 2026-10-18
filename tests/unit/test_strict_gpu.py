"""STRICT_GPU: ``engine.strict_gpu`` and the per-model ``allow_partial_cpu_offload``.

The contract (docs/ENGINES.md, "STRICT_GPU"):

* with the shipped defaults every CPU-offload flag stays refused -- at save
  time and at launch -- and ``--n-gpu-layers 999 --fit off`` are always the
  last tokens of the argv;
* ``engine.strict_gpu: false`` alone changes nothing; the model must also opt
  in with ``allow_partial_cpu_offload``, and even then only ``--n-cpu-moe`` and
  ``--no-kv-offload`` become legal;
* ``-ngl 999`` and ``--fit off`` are present in EVERY spawn argv, whatever the
  two switches say.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from studioforge.config import Config
from studioforge.core import engine as engine_module
from studioforge.core.engine import (
    PARTIAL_OFFLOAD_SPELLINGS,
    POLICY_FAMILIES,
    EngineManager,
    partial_offload_allowance,
    policy_refusal,
    refuse_policy_flags,
)
from studioforge.core.supervisor import (
    ALL_GPU_LAYERS,
    Supervisor,
    effective_launch,
    launch_policy_violations,
)
from studioforge.errors import ModelLoadError
from studioforge.types import ModelSettings
from tests.unit.test_gpu_only_policy import HELP_EXCERPT
from tests.unit.test_supervisor import (
    make_binary,
    make_plan,
    make_record,
    resolver,
    value_after,
)

TAG = "b10689"

#: Every offload-kind spelling plus the managed KV-offload family: the full
#: set that must stay refused under the shipped defaults.
OFFLOAD_SPELLINGS = sorted(
    {sp for fam in POLICY_FAMILIES if fam.kind == "offload" for sp in fam.spellings}
    | {"--no-kv-offload", "-nkvo", "--kv-offload", "-kvo"}
)
#: What the opt-in may legalise.
LEGALISABLE = {"--n-cpu-moe", "-ncmoe", "--no-kv-offload", "-nkvo"}
STILL_REFUSED = [s for s in OFFLOAD_SPELLINGS if s not in LEGALISABLE]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = 19461
    cfg.gateway.child_port_end = 19499
    cfg.ensure_dirs()
    return cfg


def build(config: Config, tmp_path: Path, extra: str, *, partial: bool = False) -> list[str]:
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path, settings=ModelSettings(extra_flags=extra, allow_partial_cpu_offload=partial)
    )
    sup = Supervisor(config, resolve_binary=resolver(binary))
    return sup.build_command(record, make_plan(), port=18100)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_defaults_are_strict_and_opt_in_is_off() -> None:
    assert Config().engine.strict_gpu is True
    assert ModelSettings().allow_partial_cpu_offload is False
    assert partial_offload_allowance(strict_gpu=True, allow_partial_cpu_offload=True) == frozenset()
    assert (
        partial_offload_allowance(strict_gpu=False, allow_partial_cpu_offload=False) == frozenset()
    )
    assert partial_offload_allowance(strict_gpu=False, allow_partial_cpu_offload=True) == (
        PARTIAL_OFFLOAD_SPELLINGS
    )


@pytest.mark.parametrize("flag", OFFLOAD_SPELLINGS)
def test_every_offload_flag_is_refused_by_default(flag: str) -> None:
    assert policy_refusal(flag) is not None
    assert refuse_policy_flags([flag]) != []


@pytest.mark.parametrize("flag", OFFLOAD_SPELLINGS)
def test_every_offload_flag_is_refused_at_launch_under_defaults(
    flag: str, config: Config, tmp_path: Path
) -> None:
    with pytest.raises(ModelLoadError):
        build(config, tmp_path, f"{flag} 3" if flag.startswith("--n-") else flag)


@pytest.mark.parametrize("flag", sorted(LEGALISABLE))
def test_opt_in_alone_does_not_help_while_strict_gpu_is_on(
    flag: str, config: Config, tmp_path: Path
) -> None:
    """The per-model flag is inert unless the machine-wide switch is also off."""
    assert config.engine.strict_gpu is True
    with pytest.raises(ModelLoadError):
        build(config, tmp_path, f"{flag} 3" if "cpu-moe" in flag else flag, partial=True)


@pytest.mark.parametrize("flag", sorted(LEGALISABLE))
def test_strict_gpu_off_alone_does_not_help_without_the_model_opt_in(
    flag: str, config: Config, tmp_path: Path
) -> None:
    config.engine.strict_gpu = False
    with pytest.raises(ModelLoadError):
        build(config, tmp_path, f"{flag} 3" if "cpu-moe" in flag else flag, partial=False)


# ---------------------------------------------------------------------------
# -ngl 999 / --fit off, always
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("extra", ["", "--threads 8", "--n-cpu-moe 6", "--no-kv-offload"])
def test_ngl_999_and_fit_off_are_in_every_argv_that_launches(
    strict: bool, partial: bool, extra: str, config: Config, tmp_path: Path
) -> None:
    config.engine.strict_gpu = strict
    offloads = extra in ("--n-cpu-moe 6", "--no-kv-offload")
    legal = (not strict) and partial
    if offloads and not legal:
        # A refused launch has no argv at all, which is stricter still.
        with pytest.raises(ModelLoadError):
            build(config, tmp_path, extra, partial=partial)
        return
    argv = build(config, tmp_path, extra, partial=partial)  # must launch
    assert argv[-4:] == ["--n-gpu-layers", ALL_GPU_LAYERS, "--fit", "off"]
    assert value_after(argv, "--n-gpu-layers") == "999"
    assert value_after(argv, "--fit") == "off"


# ---------------------------------------------------------------------------
# The opt-in: exactly two families
# ---------------------------------------------------------------------------


def test_partial_offload_flags_launch_when_both_switches_are_set(
    config: Config, tmp_path: Path
) -> None:
    config.engine.strict_gpu = False
    argv = build(config, tmp_path, "--n-cpu-moe 6 --no-kv-offload", partial=True)
    assert "--n-cpu-moe" in argv and "--no-kv-offload" in argv
    # ...and the two invariants still come last, after the user's flags.
    assert argv[-4:] == ["--n-gpu-layers", "999", "--fit", "off"]
    assert argv.index("--no-kv-offload") < argv.index("--n-gpu-layers")


def test_short_spellings_are_covered_by_the_opt_in(config: Config, tmp_path: Path) -> None:
    config.engine.strict_gpu = False
    argv = build(config, tmp_path, "-ncmoe 4 -nkvo", partial=True)
    assert "-ncmoe" in argv and "-nkvo" in argv


@pytest.mark.parametrize("flag", STILL_REFUSED)
def test_everything_else_stays_refused_even_with_both_switches_set(
    flag: str, config: Config, tmp_path: Path
) -> None:
    config.engine.strict_gpu = False
    with pytest.raises(ModelLoadError):
        build(config, tmp_path, f"{flag} x" if flag.startswith("--n-") else flag, partial=True)


@pytest.mark.parametrize("override", ["-ngl 12", "--n-gpu-layers 12", "--fit on", "-fit on"])
def test_ngl_and_fit_cannot_be_overridden_even_with_both_switches_set(
    override: str, config: Config, tmp_path: Path
) -> None:
    config.engine.strict_gpu = False
    with pytest.raises(ModelLoadError):
        build(config, tmp_path, override, partial=True)


def test_final_argv_check_honours_the_allowance_only_for_the_two_families() -> None:
    base = ["x", "--n-gpu-layers", "999", "--fit", "off"]
    both = ModelSettings(allow_partial_cpu_offload=True)
    assert launch_policy_violations([*base, "--n-cpu-moe", "6"], both) == ["--n-cpu-moe 6"]
    assert launch_policy_violations([*base, "--n-cpu-moe", "6"], both, strict_gpu=False) == []
    assert launch_policy_violations([*base, "--no-kv-offload"], both, strict_gpu=False) == []
    assert launch_policy_violations([*base, "--cpu-moe"], both, strict_gpu=False) == ["--cpu-moe"]
    # the opt-in never excuses a changed -ngl
    assert launch_policy_violations(
        ["x", "--n-gpu-layers", "20", "--fit", "off", "-ncmoe", "3"], both, strict_gpu=False
    ) == ["--n-gpu-layers 20"]


def test_effective_launch_reports_a_violation_unless_both_switches_allow_it(
    config: Config, tmp_path: Path
) -> None:
    from studioforge.core.engine import EngineFeatures

    argv = [*make_argv_base(), "--n-cpu-moe", "6"]
    plan = make_plan()
    both = ModelSettings(allow_partial_cpu_offload=True)
    assert effective_launch(argv, EngineFeatures.unknown(), plan, both).policy_violations == [
        "--n-cpu-moe 6"
    ]
    allowed = effective_launch(argv, EngineFeatures.unknown(), plan, both, strict_gpu=False)
    assert allowed.policy_violations == []


def make_argv_base() -> list[str]:
    return ["x", "--n-gpu-layers", "999", "--fit", "off"]


# ---------------------------------------------------------------------------
# Save-time validation follows the same rule
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(config: Config) -> EngineManager:
    mgr = EngineManager(config)
    directory = mgr.engine_dir(TAG)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / engine_module.HELP_FILE).write_text(HELP_EXCERPT, encoding="utf-8")
    return mgr


def test_save_time_validation_refuses_offload_by_default(manager: EngineManager) -> None:
    errors = asyncio.run(manager.validate_extra_flags(TAG, "--n-cpu-moe 4"))
    assert errors and "GPU-only" in errors[0]
    errors = asyncio.run(
        manager.validate_extra_flags(TAG, "--n-cpu-moe 4", allow_partial_cpu_offload=True)
    )
    assert errors and "GPU-only" in errors[0]  # strict_gpu is still on


def test_save_time_validation_accepts_the_two_families_only_when_both_switches_allow(
    manager: EngineManager,
) -> None:
    manager.config.engine.strict_gpu = False
    ok = asyncio.run(
        manager.validate_extra_flags(
            TAG, "--n-cpu-moe 4 --no-kv-offload", allow_partial_cpu_offload=True
        )
    )
    assert ok == []
    assert asyncio.run(manager.validate_extra_flags(TAG, "--n-cpu-moe 4")) != []
    still = asyncio.run(
        manager.validate_extra_flags(TAG, "--cpu-moe", allow_partial_cpu_offload=True)
    )
    assert still and "GPU-only" in still[0]
