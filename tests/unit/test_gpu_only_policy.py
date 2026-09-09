"""The GPU-only policy is one table, enforced twice, and reported per launch.

The regression audit of 2026-09-09 (§3) found the policy in two lists that
disagreed: ``CPU_OFFLOAD_FLAGS`` in the supervisor, which nothing in ``src``
consulted, and ``MANAGED_FLAGS`` in the engine manager, which named no
CPU-offload flag at all. The save-time validator was an *existence* check
against the engine's own ``--help`` -- and every offload flag exists in every
engine's help -- so ``--device none``, ``--cpu-moe``, ``--n-cpu-ffn 48``,
``-ot .*=CPU`` and ``--no-kv-offload`` all validated clean, were appended last
where llama.cpp's last-wins rule made them beat our own flags, and the child
inherited every ``LLAMA_ARG_*`` variable on top. These tests pin the repair:

* one table (``engine.POLICY_FAMILIES``) with every b10689 spelling;
* refused at save time (``validate_extra_flags``) BEFORE the existence check;
* refused again at launch (``Supervisor.build_command``) over the saved flags
  and over the final argv, so a stale row cannot launch;
* ``--n-gpu-layers 999 --fit off`` are the LAST tokens of every argv;
* the child's environment is stripped of ``LLAMA_ARG_*``;
* ``effective_launch`` says ``GPU-only`` or names the violation;
* ``--mlock``/``--no-mmap`` map onto ``--load-mode`` where the engine has it.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core import engine as engine_module
from studioforge.core import supervisor as supervisor_module
from studioforge.core.engine import (
    POLICY_FAMILIES,
    POLICY_FLAGS,
    REMOVED_FLAG_HINTS,
    EngineFeatures,
    EngineManager,
    child_environment,
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
from tests.unit.test_supervisor import (
    FAKE_CHILD,
    make_binary,
    make_plan,
    make_record,
    resolver,
    value_after,
)

# Away from the app's ports, the production child range and every other
# test file's range (test_supervisor 19420-19460, test_unload_verification
# 19500-19540, test_unload_reporting 19541-19599, test_watchdog 19600+).
TEST_PORT_START = 19461
TEST_PORT_END = 19499

TAG = "b10689"

#: Entries from the b10689 ``--help`` (the build on the rig on 2026-09-09):
#: the flag columns verbatim, in the engine's own layout so the help parser
#: reads them the way it reads the real thing, the descriptions shortened.
#: Enough of the surface to prove the point: every offload flag EXISTS here,
#: so a refusal cannot be the existence check.
HELP_EXCERPT = """\
----- common params -----

-h,    --help, --usage                  print usage and exit
-t,    --threads N                      number of CPU threads (default: -1)
                                        (env: LLAMA_ARG_THREADS)
-C,    --cpu-mask M                     CPU affinity mask (default: "")
-Cr,   --cpu-range lo-hi                range of CPUs for affinity
--cpu-strict <0|1>                      use strict CPU placement (default: 0)
-c,    --ctx-size N                     size of the prompt context (default: 0)
                                        (env: LLAMA_ARG_CTX_SIZE)
-kvo,  --kv-offload, -nkvo, --no-kv-offload
                                        whether to enable KV cache offloading (default: enabled)
                                        (env: LLAMA_ARG_KV_OFFLOAD)
--no-host                               bypass host buffer allowing extra buffers to be used
                                        (env: LLAMA_ARG_NO_HOST)
--rpc SERVERS                           comma-separated list of RPC servers (host:port)
                                        (env: LLAMA_ARG_RPC)
--mlock                                 DEPRECATED in favor of `--load-mode`: keep model in RAM
                                        (env: LLAMA_ARG_MLOCK)
--mmap, --no-mmap                       DEPRECATED in favor of `--load-mode`: memory-map model
                                        (env: LLAMA_ARG_MMAP)
-lm,   --load-mode MODE                 model loading mode (default: auto)
                                        - auto: mmap, unless a device does not support it
                                        - none: no special loading mode
                                        - mlock: force system to keep model in RAM
                                        (env: LLAMA_ARG_LOAD_MODE)
--tensor-read-lazy MODE                 on-demand reading of certain tensors (default: auto)
                                        (env: LLAMA_ARG_TENSOR_READ_LAZY)
--numa TYPE                             attempt optimizations that help on some NUMA systems
                                        (env: LLAMA_ARG_NUMA)
-dev,  --device <dev1,dev2,..>          devices to use for offloading (none = don't offload)
                                        (env: LLAMA_ARG_DEVICE)
-ot,   --override-tensor <tensor name pattern>=<buffer type>,...
                                        override tensor buffer type
                                        (env: LLAMA_ARG_OVERRIDE_TENSOR)
-cmoe, --cpu-moe                        keep all Mixture of Experts (MoE) weights in the CPU
                                        (env: LLAMA_ARG_CPU_MOE)
-ncmoe, --n-cpu-moe N                   keep the MoE weights of the first N layers in the CPU
                                        (env: LLAMA_ARG_N_CPU_MOE)
-ncffn, --n-cpu-ffn N                   keep the dense FFN weights of the first N layers in the CPU
                                        (env: LLAMA_ARG_N_CPU_FFN)
-ngl,  --gpu-layers, --n-gpu-layers N   max. number of layers to store in VRAM (default: auto)
                                        (env: LLAMA_ARG_N_GPU_LAYERS)
-sm,   --split-mode {none,layer,row,tensor}
                                        how to split the model across multiple GPUs, one of:
                                        (env: LLAMA_ARG_SPLIT_MODE)
-ts,   --tensor-split N0,N1,N2,...      fraction of the model to offload to each GPU
                                        (env: LLAMA_ARG_TENSOR_SPLIT)
-fit,  --fit [on|off]                   whether to adjust unset arguments to fit (default: 'on')
                                        (env: LLAMA_ARG_FIT)
--op-offload, --no-op-offload           whether to offload host tensor operations (default: true)
-m,    --model FNAME                    model path to load
                                        (env: LLAMA_ARG_MODEL)
--top-k N                               top-k sampling (default: 40, 0 = disabled)
                                        (env: LLAMA_ARG_TOP_K)
--spec-draft-override-tensor, -otd, --override-tensor-draft <tensor name pattern>=<buffer type>
                                        override tensor buffer type for draft model
--spec-draft-cpu-moe, -cmoed, --cpu-moe-draft
                                        keep all MoE weights in the CPU for the draft model
                                        (env: LLAMA_ARG_SPEC_DRAFT_CPU_MOE)
--spec-draft-n-cpu-moe, --spec-draft-ncmoe, -ncmoed, --n-cpu-moe-draft N
                                        keep the MoE weights of the first N layers in the CPU
                                        (env: LLAMA_ARG_SPEC_DRAFT_N_CPU_MOE)
--spec-draft-device, -devd, --device-draft <dev1,dev2,..>
                                        devices to use for offloading the draft model
--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N
                                        max. number of draft model layers to store in VRAM
                                        (env: LLAMA_ARG_N_GPU_LAYERS_DRAFT)
--draft, --draft-n, --draft-max N       the argument has been removed. use --spec-draft-n-max or
                                        --spec-ngram-mod-n-max
                                        (env: LLAMA_ARG_DRAFT_MAX)
-mmdev, --mmproj-device DEVICE          device to use for multimodal projector (default: auto)
                                        (env: MTMD_BACKEND_DEVICE)
--mmproj-offload, --no-mmproj-offload   whether to enable GPU offloading for the projector
                                        (env: LLAMA_ARG_MMPROJ_OFFLOAD)
-to,   --timeout N                      server read/write timeout in seconds (default: 3600)
                                        (env: LLAMA_ARG_TIMEOUT)
"""

#: The audit's own list of what got through, one spelling each, plus the two
#: draft spellings and the two flags the old design had no concept of at all.
HOSTILE_EXTRA_FLAGS: list[tuple[str, str]] = [
    ("--device none", "--device"),
    ("--cpu-moe", "--cpu-moe"),
    ("--n-cpu-ffn 48", "--n-cpu-ffn"),
    ('-ot ".*=CPU"', "--override-tensor"),
    ("--no-kv-offload", "--kv-offload"),
    ("--load-mode mlock", "--load-mode"),
    ("-ngld 0", "--spec-draft-ngl"),
    ("--n-gpu-layers-draft 0", "--spec-draft-ngl"),
    ("-ncmoe 12", "--n-cpu-moe"),
    ("--rpc 10.0.0.9:50052", "--rpc"),
    ("--no-op-offload", "--op-offload"),
    ("--tensor-split 0,0,1,0", "--tensor-split"),
    ("-ngl 10", "--n-gpu-layers"),
    ("--fit on", "--fit"),
]

#: Thread placement is not offload. These must keep validating.
ALLOWED_EXTRA_FLAGS = ["--threads 8", "--cpu-mask ff", "--cpu-range 0-7", "--cpu-strict 1"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = TEST_PORT_START
    cfg.gateway.child_port_end = TEST_PORT_END
    cfg.gateway.load_timeout_s = 20.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.gateway.max_restarts = 0
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def manager(config: Config) -> EngineManager:
    """An engine manager whose ``b10689`` is nothing but its help text."""
    mgr = EngineManager(config)
    directory = mgr.engine_dir(TAG)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / engine_module.HELP_FILE).write_text(HELP_EXCERPT, encoding="utf-8")
    return mgr


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "engine" / "fake_llama_server.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CHILD, encoding="utf-8")
    return path


def sup(config: Config, binary: Path) -> Supervisor:
    return Supervisor(config, resolve_binary=resolver(binary))


def known_engine(*flags: str) -> EngineFeatures:
    """A *known* engine advertising exactly ``flags`` (plus the ones every
    launch needs), so the D38 gate is exercised rather than the unknown
    fallback."""
    return dataclasses.replace(
        EngineFeatures.unknown("b10689"),
        known=True,
        flags=frozenset({"--fit", "--spec-type", "--flash-attn", *flags}),
    )


class _Recorder:
    """A stand-in for the module logger that keeps warnings and swallows the rest."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "canonical", "kind"),
    [
        ("-m", "--model", "managed"),
        ("-mu", "--model-url", "managed"),
        ("--port", "--port", "managed"),
        ("--host", "--host", "managed"),
        ("-a", "--alias", "managed"),
        ("-ngl", "--n-gpu-layers", "managed"),
        ("--gpu-layers", "--n-gpu-layers", "managed"),
        ("-fit", "--fit", "managed"),
        ("-fitt", "--fit-target", "managed"),
        ("-fitc", "--fit-ctx", "managed"),
        ("-dev", "--device", "managed"),
        ("-sm", "--split-mode", "managed"),
        ("-ts", "--tensor-split", "managed"),
        ("-mg", "--main-gpu", "managed"),
        ("-c", "--ctx-size", "managed"),
        ("-np", "--parallel", "managed"),
        ("-lm", "--load-mode", "managed"),
        ("--mlock", "--mlock", "managed"),
        ("--no-mmap", "--mmap", "managed"),
        ("-dio", "--direct-io", "managed"),
        ("-kvo", "--kv-offload", "managed"),
        ("-nkvo", "--kv-offload", "managed"),
        ("--no-kv-offload", "--kv-offload", "managed"),
        ("-md", "--spec-draft-model", "managed"),
        ("--model-draft", "--spec-draft-model", "managed"),
        ("-ngld", "--spec-draft-ngl", "managed"),
        ("--gpu-layers-draft", "--spec-draft-ngl", "managed"),
        ("--n-gpu-layers-draft", "--spec-draft-ngl", "managed"),
        ("-devd", "--spec-draft-device", "managed"),
        ("--device-draft", "--spec-draft-device", "managed"),
        ("-mmdev", "--mmproj-device", "managed"),
        ("--no-mmproj-offload", "--mmproj-offload", "managed"),
        ("--spec-type", "--spec-type", "managed"),
        ("-ctk", "--cache-type-k", "managed"),
        ("-ctv", "--cache-type-v", "managed"),
        ("-fa", "--flash-attn", "managed"),
        ("-ot", "--override-tensor", "offload"),
        ("-otd", "--override-tensor-draft", "offload"),
        ("--spec-draft-override-tensor", "--override-tensor-draft", "offload"),
        ("-cmoe", "--cpu-moe", "offload"),
        ("-ncmoe", "--n-cpu-moe", "offload"),
        ("-ncffn", "--n-cpu-ffn", "offload"),
        ("-cmoed", "--spec-draft-cpu-moe", "offload"),
        ("--cpu-moe-draft", "--spec-draft-cpu-moe", "offload"),
        ("-ncmoed", "--spec-draft-n-cpu-moe", "offload"),
        ("--n-cpu-moe-draft", "--spec-draft-n-cpu-moe", "offload"),
        ("--spec-draft-ncmoe", "--spec-draft-n-cpu-moe", "offload"),
        ("--op-offload", "--op-offload", "offload"),
        ("--no-op-offload", "--op-offload", "offload"),
        ("--no-host", "--no-host", "offload"),
        ("--tensor-read-lazy", "--tensor-read-lazy", "offload"),
        ("--lazy-mode", "--tensor-read-lazy", "offload"),
        ("--rpc", "--rpc", "offload"),
    ],
)
def test_every_b10689_spelling_is_in_the_one_table(
    spelling: str, canonical: str, kind: str
) -> None:
    family = POLICY_FLAGS[spelling]
    assert family.canonical == canonical
    assert family.kind == kind


def test_thread_affinity_flags_are_not_offload_and_stay_allowed() -> None:
    for spelling in ("-t", "--threads", "-C", "--cpu-mask", "-Cr", "--cpu-range", "--cpu-strict"):
        assert spelling not in POLICY_FLAGS, spelling
        assert policy_refusal(spelling) is None
    assert "--numa" not in POLICY_FLAGS
    assert refuse_policy_flags(["--threads", "8", "--cpu-mask", "ff", "--numa", "distribute"]) == []


def test_the_old_lists_are_gone_and_the_hint_is_fixed() -> None:
    """``CPU_OFFLOAD_FLAGS`` had no caller; ``MANAGED_FLAGS`` named no offload
    flag; ``--n-gpu-layers-draft`` was listed as retired while b10689 honours
    it as an alias of ``--spec-draft-ngl`` -- and overrides our 999 with it."""
    assert not hasattr(supervisor_module, "CPU_OFFLOAD_FLAGS")
    assert not hasattr(engine_module, "MANAGED_FLAGS")
    assert "--n-gpu-layers-draft" not in REMOVED_FLAG_HINTS
    assert POLICY_FLAGS["--n-gpu-layers-draft"].canonical == "--spec-draft-ngl"
    # The tensor-override carve-out that relaxed the shell-metachar ban for
    # exactly the family the policy forbids is gone with it.
    relaxed = engine_module._RELAXED_VALUE_FLAGS  # noqa: SLF001 - the carve-out under test
    assert not (
        {"-ot", "--override-tensor", "-otd", "--override-tensor-draft", "-ts", "--tensor-split"}
        & relaxed
    )
    assert "--samplers" in relaxed, "sampler chains still legitimately carry ';'"


def test_a_refusal_names_the_family_and_why() -> None:
    assert policy_refusal("--n-cpu-ffn") == (
        "'--n-cpu-ffn' is refused: StudioForge is GPU-only; "
        "--n-cpu-ffn would keep dense FFN weights on the CPU"
    )
    ncffn = policy_refusal("-ncffn")
    assert ncffn is not None and ncffn.startswith("'-ncffn' (--n-cpu-ffn) is refused")
    device = policy_refusal("--device=none")
    assert device is not None and "managed by StudioForge" in device and "--device" in device
    assert policy_refusal("--top-k") is None
    # Every family's message says which side of the fence it is on.
    for family in POLICY_FAMILIES:
        for spelling in family.spellings:
            text = policy_refusal(spelling)
            assert text is not None and family.canonical in text, spelling
            assert ("GPU-only" in text) is (family.kind == "offload"), spelling


# ---------------------------------------------------------------------------
# Enforcement point one: save time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("extra", "canonical"), HOSTILE_EXTRA_FLAGS)
async def test_hostile_extra_flags_are_refused_at_validation_for_the_policy_not_for_existence(
    manager: EngineManager, extra: str, canonical: str
) -> None:
    errors = await manager.validate_extra_flags(TAG, extra)
    assert len(errors) == 1, errors
    assert canonical in errors[0]
    assert "unknown flag" not in errors[0], "refused by the policy, not by the help"
    assert "managed by StudioForge" in errors[0] or "GPU-only" in errors[0]


@pytest.mark.parametrize("extra", ALLOWED_EXTRA_FLAGS)
async def test_thread_placement_still_validates(manager: EngineManager, extra: str) -> None:
    assert await manager.validate_extra_flags(TAG, extra) == []


async def test_validation_keeps_its_other_branches(manager: EngineManager) -> None:
    assert await manager.validate_extra_flags(TAG, "--top-k 20 --timeout 900") == []
    assert await manager.validate_extra_flags(TAG, "--top-k -1") == []
    removed = await manager.validate_extra_flags(TAG, "--draft-max 4")
    assert removed and "removed in this release" in removed[0]
    unknown = await manager.validate_extra_flags(TAG, "--not-a-flag")
    assert unknown and "unknown flag" in unknown[0]


async def test_the_override_tensor_value_no_longer_gets_the_relaxed_metachar_pass(
    manager: EngineManager,
) -> None:
    """The regex value used to be waved through with ``(``/``)``/``;`` allowed;
    now the flag itself is refused and its value gets the ordinary ban."""
    errors = await manager.validate_extra_flags(TAG, '-ot "blk\\.(1|2)\\.ffn=CPU"')
    assert any("--override-tensor" in e for e in errors)
    assert any("illegal shell character" in e for e in errors)


# ---------------------------------------------------------------------------
# Enforcement point two: launch time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("extra", "canonical"), HOSTILE_EXTRA_FLAGS)
def test_hostile_saved_extra_flags_are_refused_at_build_time(
    config: Config, tmp_path: Path, extra: str, canonical: str
) -> None:
    """A row written under an older engine, or straight into SQLite, never had
    the validator run over it. The argv builder does not care how it got there."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags=extra))
    with pytest.raises(ModelLoadError) as excinfo:
        sup(config, binary).build_command(record, make_plan(), port=18100)
    assert canonical in excinfo.value.message
    assert record.id in excinfo.value.message
    refused = excinfo.value.details["refused"]
    assert len(refused) == 1 and canonical in refused[0]


def test_the_build_time_check_does_not_depend_on_the_engine_being_known(
    config: Config, tmp_path: Path
) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--cpu-moe"))
    supervisor = sup(config, binary)
    for features in (None, EngineFeatures.unknown(), known_engine("--cpu-moe")):
        with pytest.raises(ModelLoadError):
            supervisor.build_command(record, make_plan(), port=18100, features=features)


def test_a_refused_launch_never_drops_the_token_silently(config: Config, tmp_path: Path) -> None:
    """The alternative repair -- strip the flag and launch anyway -- is the
    D2 failure mode: a setting that looks in force and is not."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--timeout 900 -ncffn 4"))
    with pytest.raises(ModelLoadError) as excinfo:
        sup(config, binary).build_command(record, make_plan(), port=18100)
    assert "--n-cpu-ffn" in excinfo.value.message
    # The harmless neighbour is reported in the redacted flag list, not lost.
    assert "--timeout" in excinfo.value.details["extra_flags"]


def test_the_two_policy_flags_are_the_last_tokens_of_every_argv(
    config: Config, tmp_path: Path
) -> None:
    """llama.cpp takes the last occurrence of a repeated option. Both flags
    the policy rests on come after the user's flags, once, so last-wins can
    only ever land on them."""
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    plain = make_record(tmp_path)
    with_extra = make_record(
        tmp_path, settings=ModelSettings(extra_flags="--cache-reuse 4096 --timeout 900")
    )
    for record in (plain, with_extra):
        for plan in (make_plan(), make_plan(devices=[0, 1], tensor_split=[0.5, 0.5])):
            argv = supervisor.build_command(
                record, plan, port=18100, draft=make_record(tmp_path, "t")
            )
            assert argv[-4:] == ["--n-gpu-layers", ALL_GPU_LAYERS, "--fit", "off"]
            assert argv.count("--n-gpu-layers") == 1 and argv.count("--fit") == 1
            assert value_after(argv, "--spec-draft-ngl") == ALL_GPU_LAYERS
    argv = supervisor.build_command(with_extra, make_plan(), port=18100)
    assert argv[-8:-4] == ["--cache-reuse", "4096", "--timeout", "900"]


def test_a_clean_argv_has_no_violations_and_a_hostile_one_names_each() -> None:
    clean = [
        "llama-server",
        "--model",
        "m.gguf",
        "--device",
        "CUDA0,CUDA1",
        "--split-mode",
        "layer",
        "--kv-offload",
        "--spec-draft-ngl",
        "999",
        "--spec-draft-device",
        "CUDA0",
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
    ]
    assert launch_policy_violations(clean) == []
    hostile = [
        "llama-server",
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
        "--device",
        "CUDA0",
        # ...and then, last-wins:
        "--device",
        "none",
        "--cpu-moe",
        "-ncffn",
        "48",
        "-ot",
        ".*=CPU",
        "--no-kv-offload",
        "--load-mode",
        "mlock",
        "-ngld",
        "0",
        "--n-gpu-layers",
        "auto",
        "--fit=on",
        "--spec-draft-device",
        "none",
        "--no-mmproj-offload",
        "--mmproj-device",
        "none",
        "--rpc",
        "10.0.0.9:50052",
    ]
    violations = launch_policy_violations(hostile)
    assert set(violations) == {
        "--cpu-moe",
        "-ncffn 48",
        "-ot .*=CPU",
        "--rpc 10.0.0.9:50052",
        "--n-gpu-layers auto",
        "--fit on",
        "--device none",
        "--spec-draft-ngl 0",
        "--spec-draft-device none",
        "--mmproj-device none",
        "--no-kv-offload",
        "--no-mmproj-offload",
        "--load-mode mlock",
    }
    # 'all' is b10689's own word for every layer, and an explicit kv-offload is fine.
    assert launch_policy_violations(["x", "-ngl", "all", "-kvo", "--spec-draft-ngl", "all"]) == []


def test_the_final_argv_check_is_a_second_net_behind_the_token_check(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the token check were ever loosened, the final-argv check still
    refuses: simulate by letting the token check pass and watching the
    argv check catch the same row."""
    monkeypatch.setattr(supervisor_module, "refuse_policy_flags", lambda tokens: [])
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--device none --cpu-moe"))
    with pytest.raises(ModelLoadError) as excinfo:
        sup(config, binary).build_command(record, make_plan(), port=18100)
    assert excinfo.value.details["policy_violations"] == ["--cpu-moe", "--device none"]
    assert "GPU-only policy" in excinfo.value.message


# ---------------------------------------------------------------------------
# Enforcement point three: the environment
# ---------------------------------------------------------------------------


def test_child_environment_strips_the_llama_arg_surface_and_keeps_the_rest() -> None:
    parent = {
        "PATH": "/usr/bin",
        "SYSTEMROOT": "C:/Windows",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "GGML_CUDA_FORCE_MMQ": "1",
        "SF_DATA_DIR": "/tmp/sf",
        "LLAMA_ARG_N_CPU_MOE": "4",
        "llama_arg_device": "none",  # Windows names are case-insensitive
        "LLAMA_ARG_FIT": "on",
        "LLAMA_API_KEY": "sk-secret",
        "LLAMA_LOG_FILE": "/tmp/x.log",
        "MTMD_BACKEND_DEVICE": "none",
    }
    env, stripped = child_environment(parent)
    assert set(env) == {
        "PATH",
        "SYSTEMROOT",
        "CUDA_VISIBLE_DEVICES",
        "GGML_CUDA_FORCE_MMQ",
        "SF_DATA_DIR",
    }
    assert stripped == sorted(
        [
            "LLAMA_ARG_N_CPU_MOE",
            "llama_arg_device",
            "LLAMA_ARG_FIT",
            "LLAMA_API_KEY",
            "LLAMA_LOG_FILE",
            "MTMD_BACKEND_DEVICE",
        ]
    )
    # The default source is the real environment, untouched.
    env, _ = child_environment()
    assert env["PATH"] == os.environ["PATH"]


async def test_a_spawned_child_does_not_inherit_llama_arg_variables(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``LLAMA_ARG_N_CPU_MOE`` in the supervisor's environment is the one
    offload path no argv check can see. It must be absent from the child's,
    and its removal logged."""
    monkeypatch.setenv("LLAMA_ARG_N_CPU_MOE", "4")
    monkeypatch.setenv("LLAMA_ARG_DEVICE", "none")
    seen: dict[str, Any] = {}
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen["env"] = kwargs.get("env")
        return await real(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spy)
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)

    supervisor = Supervisor(
        config, resolve_binary=resolver(fake_binary), launch_prefix=[sys.executable, "-u"]
    )
    record = make_record(tmp_path)
    try:
        info = await supervisor.start(record, make_plan())
        assert info.state == "ready"
    finally:
        await supervisor.aclose()

    env = seen["env"]
    assert env is not None, "the child was spawned with the inherited environment"
    assert "LLAMA_ARG_N_CPU_MOE" not in env and "LLAMA_ARG_DEVICE" not in env
    assert "PATH" in env
    stripped = [f for e, f in recorder.warnings if e == "child_env_stripped"]
    assert stripped and {"LLAMA_ARG_N_CPU_MOE", "LLAMA_ARG_DEVICE"} <= set(stripped[0]["names"])
    assert stripped[0]["model_id"] == record.id


async def test_the_engine_smoke_test_spawns_with_the_same_sanitised_environment(
    manager: EngineManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMA_ARG_CPU_MOE", "1")
    seen: dict[str, Any] = {}

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        seen["env"] = kwargs.get("env")
        raise OSError("not today")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", refuse)
    ok, detail = await manager._micro_load(Path("fake-llama-server.exe"), Path("tiny.gguf"))  # noqa: SLF001
    assert ok is False and "could not launch" in detail
    assert seen["env"] is not None and "LLAMA_ARG_CPU_MOE" not in seen["env"]
    assert "PATH" in seen["env"]


# ---------------------------------------------------------------------------
# Reporting: effective_launch (D54)
# ---------------------------------------------------------------------------


def test_effective_launch_reports_gpu_only_for_a_launch_studioforge_composed(
    config: Config, tmp_path: Path
) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--timeout 900 -t 8"))
    plan = make_plan(devices=[0, 1], tensor_split=[0.5, 0.5])
    argv = sup(config, binary).build_command(record, plan, port=18100)
    eff = effective_launch(argv, EngineFeatures.unknown(), plan, record.settings)
    assert eff.gpu_only is True and eff.policy_violations == []
    assert eff.summary.endswith("GPU-only")
    assert eff.n_gpu_layers == "999" and eff.sources["n_gpu_layers"] == "argv"
    assert eff.fit == "off" and eff.sources["fit"] == "argv"
    assert eff.device == "CUDA0,CUDA1" and eff.split_mode == "layer"
    assert eff.kv_offload is True and eff.sources["kv_offload"] == "engine_default"
    assert eff.load_mode is None
    compact = eff.compact()
    assert compact["gpu_only"] is True and compact["policy_violations"] == []


def test_effective_launch_names_every_violation_in_a_hostile_argv() -> None:
    plan = make_plan(parallel=1)
    argv = [
        "llama-server",
        "-c",
        "8192",
        "-np",
        "1",
        "--n-gpu-layers",
        "999",
        "--fit",
        "off",
        "--device",
        "none",
        "--cpu-moe",
        "-nkvo",
        "-lm",
        "mlock",
        "-ngl",
        "20",
    ]
    eff = effective_launch(argv, EngineFeatures.unknown(), plan)
    assert eff.gpu_only is False
    assert eff.policy_violations == [
        "--cpu-moe",
        "--n-gpu-layers 20",
        "--device none",
        "--no-kv-offload",
        "--load-mode mlock",
    ]
    assert eff.n_gpu_layers == "20" and eff.device == "none" and eff.kv_offload is False
    assert eff.load_mode == "mlock"
    assert eff.summary.endswith(
        "POLICY VIOLATION: --cpu-moe, --n-gpu-layers 20, --device none, --no-kv-offload, "
        "--load-mode mlock"
    )
    assert eff.compact()["gpu_only"] is False


async def test_a_launch_with_violations_is_logged_at_warning_at_spawn(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_command refuses first, so this tripwire only fires if the two
    checks ever disagree: force that by handing _spawn a lenient builder."""
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    supervisor = Supervisor(
        config, resolve_binary=resolver(fake_binary), launch_prefix=[sys.executable, "-u"]
    )
    real_build = supervisor.build_command

    def lenient(*args: Any, **kwargs: Any) -> list[str]:
        return [*real_build(*args, **kwargs), "--device", "none"]

    monkeypatch.setattr(supervisor, "build_command", lenient)
    record = make_record(tmp_path)
    try:
        info = await supervisor.start(record, make_plan())
        assert info.effective is not None and info.effective.policy_violations == ["--device none"]
    finally:
        await supervisor.aclose()
    tripped = [f for e, f in recorder.warnings if e == "gpu_only_policy_violation"]
    assert tripped and tripped[0]["violations"] == ["--device none"]


# ---------------------------------------------------------------------------
# --mlock / --no-mmap on an engine that deprecated them (F7)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "mode"),
    [
        (ModelSettings(mlock=True), "mmap+mlock"),
        (ModelSettings(no_mmap=True), "none"),
        (ModelSettings(mlock=True, no_mmap=True), "mlock"),
    ],
)
def test_an_engine_with_load_mode_gets_the_one_flag_faithfully_mapped(
    config: Config, tmp_path: Path, settings: ModelSettings, mode: str
) -> None:
    """``--mlock`` alone kept the default mmap and locked the mapping; that
    is ``mmap+mlock``, not ``mlock`` (which reads the file into locked memory
    with no mapping at all). The launch must mean what the setting meant."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=settings)
    features = known_engine("--load-mode", "--mlock", "--no-mmap")
    argv = sup(config, binary).build_command(record, make_plan(), port=18100, features=features)
    assert value_after(argv, "--load-mode") == mode
    assert "--mlock" not in argv and "--no-mmap" not in argv
    eff = effective_launch(argv, features, make_plan(), settings)
    assert eff.load_mode == mode
    assert eff.gpu_only is True, "a load mode StudioForge composed is not a violation"
    assert eff.inert == []


def test_an_engine_with_only_the_deprecated_pair_still_gets_the_pair(
    config: Config, tmp_path: Path
) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(mlock=True, no_mmap=True))
    features = known_engine("--mlock", "--no-mmap")
    argv = sup(config, binary).build_command(record, make_plan(), port=18100, features=features)
    assert "--mlock" in argv and "--no-mmap" in argv and "--load-mode" not in argv
    eff = effective_launch(argv, features, make_plan(), record.settings)
    assert eff.inert == [] and eff.gpu_only is True


def test_an_engine_with_neither_logs_setting_inert_and_passes_nothing(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(supervisor_module, "log", recorder)
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(mlock=True, no_mmap=True))
    features = known_engine()
    argv = sup(config, binary).build_command(record, make_plan(), port=18100, features=features)
    assert "--mlock" not in argv and "--no-mmap" not in argv and "--load-mode" not in argv
    inert = [f for e, f in recorder.warnings if e == "setting_inert"]
    assert sorted(f["setting"] for f in inert) == ["mlock", "no_mmap"]
    assert all(f["model_id"] == record.id and f["engine_known"] is True for f in inert)
    eff = effective_launch(argv, features, make_plan(), record.settings)
    assert eff.inert == ["mlock", "no_mmap"]
    assert "inert: mlock, no_mmap" in eff.summary


def test_an_unknown_engine_keeps_the_pre_gating_pair(config: Config, tmp_path: Path) -> None:
    """D38's fallback: an engine whose help could not be read gets the flag
    surface that predates the gating, never a guess."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(mlock=True, no_mmap=True))
    argv = sup(config, binary).build_command(record, make_plan(), port=18100)
    assert "--mlock" in argv and "--no-mmap" in argv and "--load-mode" not in argv


def test_a_load_mode_nobody_asked_for_is_a_violation() -> None:
    """The same ``--load-mode mlock`` is StudioForge's own when settings.mlock
    composed it and somebody else's when it did not."""
    plan = make_plan()
    argv = ["llama-server", "-ngl", "999", "--fit", "off", "--load-mode", "mmap+mlock"]
    assert effective_launch(argv, EngineFeatures.unknown(), plan).policy_violations == [
        "--load-mode mmap+mlock"
    ]
    assert (
        effective_launch(
            argv, EngineFeatures.unknown(), plan, ModelSettings(mlock=True)
        ).policy_violations
        == []
    )
