"""Unit tests for the llama-server child supervisor.

Two layers:

* **Command building** -- pure, no processes. This is where the b10425 flag
  surface is pinned: a wrong spelling there does not fail loudly (llama-server
  prints "the argument has been removed" and carries on), so the argv is
  asserted exactly.
* **Lifecycle** -- driven against a fake child written into ``tmp_path``. The
  fake ignores unknown flags, so the real builder output can be handed to it
  unchanged; ``launch_prefix`` supplies the Python interpreter.

One live test exercises the real ``llama-server.exe`` when it is installed.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import psutil
import pytest

from studioforge.config import Config
from studioforge.core.supervisor import (
    ALL_GPU_LAYERS,
    CPU_OFFLOAD_FLAGS,
    Supervisor,
    safe_log_name,
)
from studioforge.errors import ModelLoadError
from studioforge.types import (
    AdapterRecord,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
)

# Child port range used by these tests; kept away from the app's own ports and
# from the production default range so a running StudioForge cannot interfere.
TEST_PORT_START = 19420
TEST_PORT_END = 19460

# The single live test at the bottom of this file needs the real engine and a
# real (tiny) model. Both are located the way the app locates them -- the
# engine under SF_DATA_DIR (else <repo>/data), the model under
# SF_TEST_MODELS_DIR -- and the test skips when either is missing, so a fresh
# checkout never touches a GPU.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENGINE_TAG = "b10425"
_ENGINE_BIN = "llama-server.exe" if os.name == "nt" else "llama-server"
_TINY_MODEL_RELPATH = (
    Path("lmstudio-community") / "Qwen2.5-0.5B-Instruct-GGUF" / "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
)


def _find_real_engine() -> Path | None:
    env = os.environ.get("SF_DATA_DIR")
    for root in ([Path(env)] if env else []) + [_REPO_ROOT / "data"]:
        candidate = root / "engines" / _ENGINE_TAG / _ENGINE_BIN
        if candidate.is_file():
            return candidate
    return None


def _find_real_model() -> Path | None:
    env = os.environ.get("SF_TEST_MODELS_DIR", "").strip()
    if not env:
        return None
    candidate = Path(env) / _TINY_MODEL_RELPATH
    return candidate if candidate.is_file() else None


REAL_ENGINE = _find_real_engine()
REAL_MODEL = _find_real_model()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.gateway.child_port_start = TEST_PORT_START
    cfg.gateway.child_port_end = TEST_PORT_END
    cfg.gateway.load_timeout_s = 20.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.gateway.max_restarts = 2
    cfg.gateway.restart_backoff_s = 0.05
    cfg.ensure_dirs()
    return cfg


def make_binary(tmp_path: Path, name: str = "llama-server.exe") -> Path:
    path = tmp_path / "engines" / "b10425" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")
    return path


def resolver(path: Path) -> Callable[[str | None], Path]:
    def resolve(tag: str | None) -> Path:
        return path

    return resolve


def make_record(
    tmp_path: Path,
    model_id: str = "qwen2.5-7b",
    *,
    kind: str = "chat",
    settings: ModelSettings | None = None,
    mmproj: bool = False,
) -> ModelRecord:
    model_path = tmp_path / "models" / f"{model_id}.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF")
    mmproj_path = None
    if mmproj:
        mmproj_path = tmp_path / "models" / f"mmproj-{model_id}.gguf"
        mmproj_path.write_bytes(b"GGUF")
    return ModelRecord(
        id=model_id,
        name=model_id,
        kind=kind,  # type: ignore[arg-type]
        path=model_path,
        mmproj_path=mmproj_path,
        capabilities=ModelCapabilities(vision=mmproj, embedding=kind == "embedding"),
        settings=settings or ModelSettings(),
    )


def make_plan(model_id: str = "qwen2.5-7b", **kwargs: object) -> LoadPlan:
    params: dict[str, object] = {"model_id": model_id, "devices": [0], "ctx_size": 8192}
    params.update(kwargs)
    return LoadPlan(**params)  # type: ignore[arg-type]


def sup(config: Config, binary: Path, **kwargs: object) -> Supervisor:
    return Supervisor(config, resolve_binary=resolver(binary), **kwargs)  # type: ignore[arg-type]


def value_after(argv: Sequence[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def test_minimal_chat_command_is_exact(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path)
    plan = make_plan(ctx_size=8192, parallel=1)
    argv = sup(config, binary).build_command(record, plan, port=18100)

    assert argv == [
        str(binary),
        "--model",
        str(record.path),
        "--alias",
        "qwen2.5-7b",
        "--host",
        "127.0.0.1",
        "--port",
        "18100",
        "--n-gpu-layers",
        "999",
        "--ctx-size",
        "8192",
        "--parallel",
        "1",
        "--device",
        "CUDA0",
        "--main-gpu",
        "0",
        "--split-mode",
        "none",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--flash-attn",
        "auto",
        "--no-webui",
        "--props",
        "--slots",
        "--metrics",
        # b10425's --fit defaults to ON and would auto-adjust unset arguments to
        # fit device memory. The planner already made that decision, so the
        # engine must not second-guess it (GPU-only: fail loudly, never shrink).
        "--fit",
        "off",
        "--cache-reuse",
        "256",
        # llama.cpp's default ('auto') routes a reasoning model's thoughts into
        # reasoning_content and leaves content EMPTY -- an empty reply to every
        # OpenAI client. 'none' keeps them inline. See DECISIONS.md D12.
        "--reasoning-format",
        "none",
    ]


def test_reasoning_format_defaults_to_none(config: Config, tmp_path: Path) -> None:
    """Verified against DeepSeek-R1: 'auto' yields content len 0."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(make_record(tmp_path), make_plan(), port=18100)
    assert value_after(argv, "--reasoning-format") == "none"


def test_reasoning_settings_are_overridable(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path,
        settings=ModelSettings(reasoning_format="deepseek", reasoning="off", reasoning_budget=256),
    )
    argv = sup(config, binary).build_command(record, make_plan(), port=18100)
    assert value_after(argv, "--reasoning-format") == "deepseek"
    assert value_after(argv, "--reasoning") == "off"
    assert value_after(argv, "--reasoning-budget") == "256"


def test_engine_side_autofit_is_always_disabled(config: Config, tmp_path: Path) -> None:
    """--fit off on every launch, so no silent partial-offload path exists."""
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    for plan in (
        make_plan(devices=[0]),
        make_plan(devices=[0, 1], tensor_split=[0.5, 0.5]),
        make_plan(devices=[2], ctx_size=131072, parallel=4),
    ):
        argv = supervisor.build_command(make_record(tmp_path), plan, port=18100)
        assert value_after(argv, "--fit") == "off"


def test_n_gpu_layers_is_always_999(config: Config, tmp_path: Path) -> None:
    """GPU-only: there is no code path that computes a partial layer count."""
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    variants = [
        make_plan(devices=[0]),
        make_plan(devices=[0, 1], tensor_split=[0.5, 0.5]),
        make_plan(devices=[2], ctx_size=131072, parallel=8),
    ]
    for plan in variants:
        argv = supervisor.build_command(make_record(tmp_path), plan, port=18100)
        assert value_after(argv, "--n-gpu-layers") == ALL_GPU_LAYERS == "999"


def test_ctx_size_is_multiplied_by_parallel(config: Config, tmp_path: Path) -> None:
    """--ctx-size is the TOTAL budget shared by slots, so ctx x parallel."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(ctx_size=4096, parallel=4), port=18100
    )
    assert value_after(argv, "--ctx-size") == "16384"
    assert value_after(argv, "--parallel") == "4"


def test_single_gpu_placement(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(devices=[1], main_gpu=1), port=18100
    )
    assert value_after(argv, "--device") == "CUDA1"
    assert value_after(argv, "--split-mode") == "none"
    # --main-gpu indexes the filtered --device list, so a single device is 0.
    assert value_after(argv, "--main-gpu") == "0"
    assert "--tensor-split" not in argv


def test_two_gpu_split_placement(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    plan = make_plan(devices=[0, 1], tensor_split=[0.6, 0.4], split_mode="layer", main_gpu=1)
    argv = sup(config, binary).build_command(make_record(tmp_path), plan, port=18100)
    assert value_after(argv, "--device") == "CUDA0,CUDA1"
    assert value_after(argv, "--tensor-split") == "0.6,0.4"
    assert value_after(argv, "--split-mode") == "layer"
    assert value_after(argv, "--main-gpu") == "1"


def test_multi_gpu_main_gpu_is_position_not_ordinal(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    plan = make_plan(devices=[2, 3], tensor_split=[0.5, 0.5], main_gpu=3)
    argv = sup(config, binary).build_command(make_record(tmp_path), plan, port=18100)
    assert value_after(argv, "--device") == "CUDA2,CUDA3"
    assert value_after(argv, "--main-gpu") == "1"


def test_tensor_split_is_four_decimal_places(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    plan = make_plan(devices=[0, 1, 2], tensor_split=[0.333333, 0.333333, 0.333334])
    argv = sup(config, binary).build_command(make_record(tmp_path), plan, port=18100)
    assert value_after(argv, "--tensor-split") == "0.3333,0.3333,0.3333"


@pytest.mark.parametrize("value", ["on", "off", "auto"])
def test_flash_attn_takes_a_value(config: Config, tmp_path: Path, value: str) -> None:
    """b10425: --flash-attn is on|off|auto, never a bare boolean."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(flash_attn=value), port=18100
    )
    assert value_after(argv, "--flash-attn") == value
    index = argv.index("--flash-attn")
    assert not argv[index + 1].startswith("--")


def test_kv_cache_types_from_plan(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    plan = make_plan(kv_cache_type="q8_0", kv_cache_type_v="q5_1")
    argv = sup(config, binary).build_command(make_record(tmp_path), plan, port=18100)
    assert value_after(argv, "--cache-type-k") == "q8_0"
    assert value_after(argv, "--cache-type-v") == "q5_1"


def test_cache_reuse_default_is_applied(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(make_record(tmp_path), make_plan(), port=18100)
    assert value_after(argv, "--cache-reuse") == str(config.models.default_cache_reuse)


def test_cache_reuse_override_and_disable(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    record = make_record(tmp_path, settings=ModelSettings(cache_reuse=1024))
    argv = supervisor.build_command(record, make_plan(), port=18100)
    assert value_after(argv, "--cache-reuse") == "1024"

    off = make_record(tmp_path, settings=ModelSettings(cache_reuse=0))
    argv = supervisor.build_command(off, make_plan(), port=18100)
    assert "--cache-reuse" not in argv


def test_bare_boolean_flags_only_when_true(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    plain = supervisor.build_command(make_record(tmp_path), make_plan(), port=18100)
    for flag in ("--cont-batching", "--mlock", "--no-mmap", "--no-context-shift"):
        assert flag not in plain

    record = make_record(
        tmp_path,
        settings=ModelSettings(cont_batching=True, mlock=True, no_mmap=True, no_context_shift=True),
    )
    argv = supervisor.build_command(record, make_plan(), port=18100)
    for flag in ("--cont-batching", "--mlock", "--no-mmap", "--no-context-shift"):
        assert flag in argv
        # bare flags: the next token must be another flag or nothing
        index = argv.index(flag)
        assert index == len(argv) - 1 or argv[index + 1].startswith("--")


def test_advanced_numeric_settings(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            batch_size=2048,
            ubatch_size=512,
            threads=16,
            threads_batch=32,
            rope_freq_base=1000000.0,
            rope_freq_scale=0.5,
            rope_scaling="linear",
        ),
    )
    argv = sup(config, binary).build_command(record, make_plan(), port=18100)
    assert value_after(argv, "--batch-size") == "2048"
    assert value_after(argv, "--ubatch-size") == "512"
    assert value_after(argv, "--threads") == "16"
    assert value_after(argv, "--threads-batch") == "32"
    assert value_after(argv, "--rope-freq-base") == "1000000"
    assert value_after(argv, "--rope-freq-scale") == "0.5"
    assert value_after(argv, "--rope-scaling") == "linear"


def test_single_slot_launch_carries_no_concurrency_flags(config: Config, tmp_path: Path) -> None:
    """One slot must launch byte-identically to before the estimator existed."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=1), port=18100
    )
    assert "--slot-prompt-similarity" not in argv
    assert "--kv-unified" not in argv
    assert "--batch-size" not in argv


def test_multi_slot_launch_sets_slot_prompt_similarity(config: Config, tmp_path: Path) -> None:
    """Slot affinity only matters once there is more than one slot to pick."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=2), port=18100
    )
    assert value_after(argv, "--slot-prompt-similarity") == "0.3"
    # Still the engine default batch: 2 slots do not congest a 2048 batch.
    assert "--batch-size" not in argv


def test_many_slots_raise_the_batch_size(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=8), port=18100
    )
    assert value_after(argv, "--batch-size") == "4096"
    # --ubatch-size is a VRAM term the planner models; it must stay untouched.
    assert "--ubatch-size" not in argv


def test_explicit_batch_size_wins_over_the_many_slot_default(
    config: Config, tmp_path: Path
) -> None:
    """An explicit setting is never quietly overruled by a slot-count heuristic."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(batch_size=1024))
    argv = sup(config, binary).build_command(record, make_plan(parallel=8), port=18100)
    assert argv.count("--batch-size") == 1
    assert value_after(argv, "--batch-size") == "1024"


def test_kv_unified_is_opt_in_per_model(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    off = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=4), port=18100
    )
    assert "--kv-unified" not in off

    record = make_record(tmp_path, settings=ModelSettings(kv_unified=True))
    on = sup(config, binary).build_command(record, make_plan(parallel=4), port=18100)
    assert "--kv-unified" in on


def test_deprecated_defrag_thold_is_never_emitted(config: Config, tmp_path: Path) -> None:
    """``--defrag-thold`` is deprecated in b10425 and must not reach the child.

    The stored setting still loads -- old rows and the GUI's settings form both
    read it -- but passing a deprecated flag to keep a saved value looking
    honoured is how a setting quietly stops meaning anything.
    """
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(defrag_thold=0.1))
    argv = sup(config, binary).build_command(record, make_plan(), port=18100)
    assert "--defrag-thold" not in argv


def test_sampler_defaults_only_when_set(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    plain = supervisor.build_command(make_record(tmp_path), make_plan(), port=18100)
    for flag in ("--temp", "--top-p", "--top-k", "--min-p", "--repeat-penalty"):
        assert flag not in plain

    record = make_record(
        tmp_path,
        settings=ModelSettings(
            temperature=0.7, top_p=0.95, top_k=40, min_p=0.05, repeat_penalty=1.1
        ),
    )
    argv = supervisor.build_command(record, make_plan(), port=18100)
    assert value_after(argv, "--temp") == "0.7"
    assert value_after(argv, "--top-p") == "0.95"
    assert value_after(argv, "--top-k") == "40"
    assert value_after(argv, "--min-p") == "0.05"
    assert value_after(argv, "--repeat-penalty") == "1.1"


def test_embedding_model_gets_embedding_and_no_mmproj(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, "nomic-embed", kind="embedding", mmproj=True)
    argv = sup(config, binary).build_command(record, make_plan("nomic-embed"), port=18100)
    assert "--embedding" in argv
    assert "--mmproj" not in argv


def test_vision_model_gets_mmproj(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, "qwen2-vl", mmproj=True)
    argv = sup(config, binary).build_command(record, make_plan("qwen2-vl"), port=18100)
    assert value_after(argv, "--mmproj") == str(record.mmproj_path)
    assert "--embedding" not in argv


def test_adapters_scale_one_vs_scaled(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    full = AdapterRecord(id="a1", name="a1", path=tmp_path / "a1.gguf")
    partial = AdapterRecord(id="a2", name="a2", path=tmp_path / "a2.gguf")
    argv = sup(config, binary).build_command(
        make_record(tmp_path),
        make_plan(),
        port=18100,
        adapters=[(full, 1.0), (partial, 0.7)],
    )
    assert argv[argv.index("--lora") : argv.index("--lora") + 2] == [
        "--lora",
        str(full.path),
    ]
    index = argv.index("--lora-scaled")
    assert argv[index : index + 3] == ["--lora-scaled", str(partial.path), "0.7"]


def test_draft_model_requires_spec_type(config: Config, tmp_path: Path) -> None:
    """--spec-type draft-simple is what actually enables drafting in b10425."""
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path,
        settings=ModelSettings(spec_draft_n_max=8, spec_draft_n_min=2, spec_draft_p_min=0.75),
    )
    draft = make_record(tmp_path, "qwen2.5-0.5b")
    argv = sup(config, binary).build_command(
        record, make_plan(devices=[0, 1], tensor_split=[0.5, 0.5]), port=18100, draft=draft
    )
    assert value_after(argv, "--spec-type") == "draft-simple"
    assert value_after(argv, "--spec-draft-model") == str(draft.path)
    assert value_after(argv, "--spec-draft-n-max") == "8"
    assert value_after(argv, "--spec-draft-n-min") == "2"
    assert value_after(argv, "--spec-draft-p-min") == "0.75"
    assert value_after(argv, "--spec-draft-ngl") == "999"
    assert value_after(argv, "--spec-draft-device") == "CUDA0,CUDA1"
    # The renamed-away flags are accepted-and-ignored by b10425: never emit them.
    for removed in ("--draft", "--draft-max", "--draft-min", "--n-gpu-layers-draft"):
        assert removed not in argv


def test_draft_defaults_n_max_and_no_draft_flags_without_draft(
    config: Config, tmp_path: Path
) -> None:
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    argv = supervisor.build_command(
        make_record(tmp_path), make_plan(), port=18100, draft=make_record(tmp_path, "tiny")
    )
    assert value_after(argv, "--spec-draft-n-max") == "16"

    plain = supervisor.build_command(make_record(tmp_path), make_plan(), port=18100)
    assert not [flag for flag in plain if flag.startswith("--spec")]


def test_draft_kv_types_come_from_the_draft_record(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    draft = make_record(
        tmp_path,
        "tiny",
        settings=ModelSettings(kv_cache_type="q8_0", kv_cache_type_v="q8_0"),
    )
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, draft=draft
    )
    assert value_after(argv, "--spec-draft-type-k") == "q8_0"
    assert value_after(argv, "--spec-draft-type-v") == "q8_0"


def test_extra_flags_are_last_and_shell_split(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path, settings=ModelSettings(extra_flags="--cache-reuse 4096 --timeout 900")
    )
    argv = sup(config, binary).build_command(record, make_plan(), port=18100)
    assert argv[-4:] == ["--cache-reuse", "4096", "--timeout", "900"]
    # Our own --cache-reuse is still present but earlier, so the override wins.
    assert argv.count("--cache-reuse") == 2
    assert argv.index("--cache-reuse") < len(argv) - 4


def test_extra_flags_quoting(config: Config, tmp_path: Path) -> None:
    record = make_record(
        tmp_path, settings=ModelSettings(extra_flags='--chat-template-file "my template.jinja"')
    )
    argv = sup(config, make_binary(tmp_path)).build_command(record, make_plan(), port=18100)
    if os.name == "nt":
        # posix=False keeps quotes (and backslash paths) intact on Windows.
        assert argv[-2:] == ["--chat-template-file", '"my template.jinja"']
    else:
        assert argv[-2:] == ["--chat-template-file", "my template.jinja"]


def test_builder_never_emits_cpu_offload_or_partial_gpu(config: Config, tmp_path: Path) -> None:
    """Sweep every builder branch: GPU-only must be unreachable to break."""
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    adapter = AdapterRecord(id="a", name="a", path=tmp_path / "a.gguf")
    settings = ModelSettings(
        batch_size=1024,
        cont_batching=True,
        mlock=True,
        no_mmap=True,
        no_context_shift=True,
        cache_reuse=512,
        temperature=0.6,
        spec_draft_n_max=4,
        extra_flags="--timeout 600",
    )
    candidates = [
        (make_record(tmp_path, mmproj=True, settings=settings), make_plan(devices=[0])),
        (
            make_record(tmp_path, "emb", kind="embedding", settings=settings),
            make_plan("emb", devices=[1]),
        ),
        (
            make_record(tmp_path, mmproj=True, settings=settings),
            make_plan(devices=[0, 1, 2, 3], tensor_split=[0.25] * 4, split_mode="row"),
        ),
    ]
    for record, plan in candidates:
        for draft in (None, make_record(tmp_path, "tiny")):
            argv = supervisor.build_command(
                record, plan, port=18100, draft=draft, adapters=[(adapter, 0.5)]
            )
            for index, token in enumerate(argv):
                assert token not in CPU_OFFLOAD_FLAGS, token
                if token in ("--n-gpu-layers", "-ngl", "--gpu-layers", "--spec-draft-ngl"):
                    assert argv[index + 1] == "999"


def test_safe_log_name_strips_path_separators() -> None:
    assert safe_log_name("publisher/model:Q4_K_M") == "publisher_model_Q4_K_M"
    assert safe_log_name(r"a\b") == "a_b"
    assert "/" not in safe_log_name("x/y/z")


def test_engine_tag_is_passed_to_the_resolver(config: Config, tmp_path: Path) -> None:
    seen: list[str | None] = []
    binary = make_binary(tmp_path)

    def resolve(tag: str | None) -> Path:
        seen.append(tag)
        return binary

    supervisor = Supervisor(config, resolve_binary=resolve)
    record = make_record(tmp_path, settings=ModelSettings(engine_tag="b9999"))
    supervisor.build_command(record, make_plan(), port=18100)
    supervisor.build_command(record, make_plan(), port=18100, engine_tag="b10425")
    supervisor.build_command(make_record(tmp_path), make_plan(), port=18100)
    assert seen == ["b9999", "b10425", None]


# ---------------------------------------------------------------------------
# Fake child, for lifecycle tests
# ---------------------------------------------------------------------------

FAKE_CHILD = '''
"""A stand-in for llama-server: ignores unknown flags, serves the endpoints."""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = sys.argv[1:]


def opt(name, default=None):
    if name in ARGS:
        index = ARGS.index(name)
        if index + 1 < len(ARGS):
            return ARGS[index + 1]
    return default


PORT = int(opt("--port", "0"))
N_CTX = int(opt("--ctx-size", "4096"))
PARALLEL = int(opt("--parallel", "1"))
ALIAS = opt("--alias", "fake")
MODEL = opt("--model", "fake.gguf")
EXIT_CODE = opt("--fake-exit-code")
UNHEALTHY = "--fake-unhealthy" in ARGS
UNHEALTHY_IF = opt("--fake-unhealthy-if")
CRASH_AFTER = opt("--fake-crash-after")
CRASH_ONCE = opt("--fake-crash-once")
TOUCH_ON_CRASH = opt("--fake-touch-on-crash")
NO_LISTEN = "--fake-no-listen" in ARGS

sys.stderr.write("fake-llama-server starting on port %d\\n" % PORT)
sys.stderr.flush()

if EXIT_CODE is not None:
    sys.stderr.write("CUDA error: out of memory\\n")
    sys.stderr.write("fatal: failed to load model\\n")
    sys.stderr.flush()
    sys.exit(int(EXIT_CODE))

should_crash = CRASH_AFTER is not None
if should_crash and CRASH_ONCE:
    if os.path.exists(CRASH_ONCE):
        should_crash = False
    else:
        with open(CRASH_ONCE, "w") as handle:
            handle.write("1")

if should_crash:
    def _crash():
        time.sleep(float(CRASH_AFTER))
        sys.stderr.write("fake: simulated crash\\n")
        sys.stderr.flush()
        if TOUCH_ON_CRASH:
            with open(TOUCH_ON_CRASH, "w") as handle:
                handle.write("1")
        os._exit(9)

    threading.Thread(target=_crash, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if UNHEALTHY or (UNHEALTHY_IF and os.path.exists(UNHEALTHY_IF)):
                self._send(503, {"error": {"message": "Loading model"}})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/props":
            self._send(
                200,
                {
                    "default_generation_settings": {"n_ctx": N_CTX // PARALLEL,
                                                    "params": {"n_predict": -1}},
                    "total_slots": PARALLEL,
                    "model_alias": ALIAS,
                    "model_path": MODEL,
                    "chat_template": "{{ messages }}",
                },
            )
        elif self.path == "/slots":
            self._send(200, [{"id": i, "is_processing": False} for i in range(PARALLEL)])
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.path == "/lora-adapters":
            self._send(200, {"success": True})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def log_message(self, fmt, *args):
        sys.stderr.write("fake: " + (fmt % args) + "\\n")
        sys.stderr.flush()


if NO_LISTEN:
    # Alive but deaf: used to test that a foreign listener on our port is not
    # mistaken for this child.
    while True:
        time.sleep(0.1)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # No SO_REUSEADDR: a port clash must fail loudly instead of silently
    # sharing the port with another listener (which Windows allows).
    allow_reuse_address = False


server = Server(("127.0.0.1", PORT), Handler)
server.serve_forever()
'''


@pytest.fixture
def fake_binary(tmp_path: Path) -> Path:
    path = tmp_path / "engine" / "fake_llama_server.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FAKE_CHILD, encoding="utf-8")
    return path


def fake_sup(config: Config, fake_binary: Path) -> Supervisor:
    return Supervisor(
        config,
        resolve_binary=resolver(fake_binary),
        launch_prefix=[sys.executable, "-u"],
    )


async def wait_for(
    predicate: Callable[[], bool],
    timeout: float = 20.0,  # noqa: ASYNC109 - poll deadline
) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_start_ready_and_introspection(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    try:
        info = await supervisor.start(record, make_plan(ctx_size=2048, parallel=2))
        assert info.state == "ready"
        assert info.port == TEST_PORT_START
        assert info.pid is not None
        assert supervisor.is_ready(record.id)
        assert supervisor.get(record.id) is info
        assert [i.model_id for i in supervisor.list()] == [record.id]
        assert supervisor.base_url(record.id) == f"http://127.0.0.1:{TEST_PORT_START}"
        assert await supervisor.health(record.id) is True

        props = await supervisor.props(record.id)
        assert props is not None
        # The fake echoes ctx-size/parallel back the way llama-server does.
        assert props["total_slots"] == 2
        assert props["default_generation_settings"]["n_ctx"] == 2048
        assert props["model_alias"] == record.id

        slots = await supervisor.slots(record.id)
        assert isinstance(slots, list)
        assert len(slots) == 2

        assert await supervisor.set_lora_scales(record.id, [{"id": 0, "scale": 0.5}]) is True

        assert supervisor.log_path(record.id) is not None
        assert any("fake-llama-server starting" in line for line in supervisor.tail_log(record.id))
    finally:
        await supervisor.aclose()

    assert supervisor.get(record.id) is None
    assert await supervisor.props(record.id) is None
    assert await supervisor.slots(record.id) is None
    assert await supervisor.health(record.id) is False


async def test_stop_kills_tree_and_frees_port(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, make_plan())
    pid = info.pid
    assert pid is not None
    port = info.port

    await supervisor.stop(record.id)
    assert supervisor.get(record.id) is None
    assert not psutil.pid_exists(pid) or not psutil.Process(pid).is_running()

    # The port must be immediately reusable by the next load.
    info2 = await supervisor.start(record, make_plan())
    assert info2.port == port
    await supervisor.aclose()


async def test_kill_is_immediate(config: Config, tmp_path: Path, fake_binary: Path) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, make_plan())
    pid = info.pid
    assert await supervisor.kill(record.id) is True
    assert await supervisor.kill(record.id) is False
    assert pid is not None
    assert not psutil.pid_exists(pid) or not psutil.Process(pid).is_running()
    await supervisor.aclose()


async def test_stop_all_stops_everything(config: Config, tmp_path: Path, fake_binary: Path) -> None:
    supervisor = fake_sup(config, fake_binary)
    first = make_record(tmp_path, "model-a")
    second = make_record(tmp_path, "model-b")
    await supervisor.start(first, make_plan("model-a"))
    await supervisor.start(second, make_plan("model-b"))
    assert len(supervisor.list()) == 2
    await supervisor.stop_all()
    assert supervisor.list() == []
    await supervisor.aclose()


async def test_child_exiting_immediately_reports_stderr(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-exit-code 3"))
    with pytest.raises(ModelLoadError) as excinfo:
        await supervisor.start(record, make_plan())
    message = excinfo.value.message
    assert "CUDA error: out of memory" in message
    assert "exited with code 3" in message
    assert excinfo.value.details["exit_code"] == 3
    assert any("fatal" in line for line in excinfo.value.details["stderr"])
    # A failed start leaves nothing behind, and the port is released.
    assert supervisor.get(record.id) is None
    assert TEST_PORT_START not in supervisor._ports_in_use
    await supervisor.aclose()


async def test_child_never_healthy_times_out(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    config.gateway.load_timeout_s = 1.0
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-unhealthy"))
    with pytest.raises(ModelLoadError) as excinfo:
        await supervisor.start(record, make_plan())
    assert "did not become healthy" in excinfo.value.message
    assert supervisor.get(record.id) is None
    await supervisor.aclose()


async def test_crash_triggers_restart_with_backoff(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    marker = tmp_path / "crashed-once.marker"
    supervisor = fake_sup(config, fake_binary)
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            extra_flags=f"--fake-crash-after 1.5 --fake-crash-once {marker.as_posix()}"
        ),
    )
    info = await supervisor.start(record, make_plan())
    first_pid = info.pid

    assert await wait_for(lambda: info.restarts >= 1, timeout=15.0)
    assert await wait_for(lambda: info.state == "ready", timeout=15.0)
    assert info.restarts == 1
    assert info.pid != first_pid
    assert marker.is_file()
    assert await supervisor.health(record.id) is True
    assert info.last_error is not None and "exited with code" in info.last_error
    await supervisor.aclose()


async def test_repeated_crash_stops_after_max_restarts(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    config.gateway.max_restarts = 2
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-crash-after 1.5"))
    info = await supervisor.start(record, make_plan())

    assert await wait_for(lambda: info.state == "failed", timeout=30.0)
    assert info.restarts == 2
    assert not supervisor.is_ready(record.id)
    await supervisor.aclose()


async def test_a_hung_relaunch_does_not_leak_a_live_child(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    """A restart whose load hangs must kill the hung child before giving up.

    First launch: healthy, then it crashes (writing a marker file). The
    relaunched child finds the marker and reports /health 503 forever, so the
    watcher's ``_await_ready`` times out with the process still alive. Once the
    watcher gives up, that process must be dead -- a survivor here is a
    llama-server holding VRAM with nothing supervising it, which on a GPU-only
    server is the worst possible leak. ``start()`` already tears this case
    down; the restart path must too.
    """
    config.gateway.load_timeout_s = 2.0
    config.gateway.max_restarts = 1
    config.gateway.restart_backoff_s = 0.05
    once = tmp_path / "crashed-once.marker"
    crashed = tmp_path / "crash-happened.marker"
    supervisor = fake_sup(config, fake_binary)
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            # Run 1: healthy, crashes at 1.5s and touches `crashed`.
            # Run 2 (the relaunch): sees `crashed` -> /health 503 forever.
            extra_flags=(
                f"--fake-crash-after 1.5 --fake-crash-once {once.as_posix()} "
                f"--fake-touch-on-crash {crashed.as_posix()} "
                f"--fake-unhealthy-if {crashed.as_posix()}"
            )
        ),
    )
    info = await supervisor.start(record, make_plan())
    try:
        assert await wait_for(lambda: info.state == "failed", timeout=30.0)
        pid = info.pid
        assert pid is not None
        gone = await wait_for(
            lambda: not psutil.pid_exists(pid) or not psutil.Process(pid).is_running(),
            timeout=10.0,
        )
        assert gone, (
            f"relaunched child pid {pid} is still alive after the watcher gave up: "
            "a hung llama-server was abandoned with its VRAM"
        )
    finally:
        await supervisor.aclose()


async def test_deliberate_stop_does_not_restart(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, make_plan())
    await supervisor.stop(record.id)
    assert info.restarts == 0
    # Give a hypothetical restart plenty of time to (not) happen.
    await asyncio.sleep(0.6)
    assert supervisor.get(record.id) is None
    assert info.restarts == 0
    assert not psutil.pid_exists(info.pid or -1) or info.state == "stopped"
    await supervisor.aclose()


async def test_kill_does_not_restart(config: Config, tmp_path: Path, fake_binary: Path) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    info = await supervisor.start(record, make_plan())
    await supervisor.kill(record.id)
    await asyncio.sleep(0.6)
    assert info.restarts == 0
    assert supervisor.list() == []
    await supervisor.aclose()


async def test_port_allocation_skips_a_bound_port(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", TEST_PORT_START))
    blocker.listen(1)
    supervisor = fake_sup(config, fake_binary)
    try:
        info = await supervisor.start(make_record(tmp_path), make_plan())
        assert info.port == TEST_PORT_START + 1
    finally:
        await supervisor.aclose()
        blocker.close()


async def test_foreign_listener_on_the_port_is_not_adopted(
    config: Config, tmp_path: Path, fake_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale server answering /health must not be mistaken for our child."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Impostor(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            payload = (
                {"status": "ok"}
                if self.path == "/health"
                else {"model_alias": "someone-elses-model", "total_slots": 4}
            )
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args: object) -> None:
            return

    impostor = ThreadingHTTPServer(("127.0.0.1", TEST_PORT_START), Impostor)
    impostor.daemon_threads = True
    threading.Thread(target=impostor.serve_forever, daemon=True).start()

    # Force the allocator onto the squatted port; our child stays alive but deaf.
    monkeypatch.setattr(
        "studioforge.core.supervisor._port_is_bindable", lambda port, host="127.0.0.1": True
    )
    config.gateway.load_timeout_s = 2.0
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(extra_flags="--fake-no-listen"))
    try:
        with pytest.raises(ModelLoadError) as excinfo:
            await supervisor.start(record, make_plan())
        assert "someone-elses-model" in excinfo.value.message
        assert excinfo.value.details["port_conflict"] == "someone-elses-model"
    finally:
        await supervisor.aclose()
        impostor.shutdown()
        impostor.server_close()


async def test_port_exhaustion_names_the_range(config: Config, tmp_path: Path) -> None:
    config.gateway.child_port_start = TEST_PORT_START
    config.gateway.child_port_end = TEST_PORT_START + 1
    supervisor = sup(config, make_binary(tmp_path))
    supervisor._ports_in_use.update({TEST_PORT_START, TEST_PORT_START + 1})
    with pytest.raises(ModelLoadError) as excinfo:
        supervisor._allocate_port()
    assert f"{TEST_PORT_START}-{TEST_PORT_START + 1}" in excinfo.value.message


async def test_two_concurrent_starts_spawn_one_child(
    config: Config, tmp_path: Path, fake_binary: Path
) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path)
    plan = make_plan()
    first, second = await asyncio.gather(
        supervisor.start(record, plan), supervisor.start(record, plan)
    )
    assert first is second
    assert len(supervisor.list()) == 1
    assert len(supervisor._ports_in_use) == 1
    await supervisor.aclose()


async def test_request_accounting(config: Config, tmp_path: Path, fake_binary: Path) -> None:
    supervisor = fake_sup(config, fake_binary)
    record = make_record(tmp_path, settings=ModelSettings(ttl_s=60))
    info = await supervisor.start(record, make_plan())
    supervisor.mark_request_start(record.id)
    supervisor.mark_request_start(record.id)
    assert info.active_requests == 2
    assert info.total_requests == 2
    supervisor.mark_request_end(record.id, tokens_per_second=42.5)
    assert info.active_requests == 1
    assert info.last_tokens_per_second == 42.5
    supervisor.mark_request_end(record.id)
    supervisor.mark_request_end(record.id)  # never goes negative
    assert info.active_requests == 0
    assert info.ttl_s == 60
    # Unknown model ids are a no-op rather than an error.
    supervisor.mark_request_start("nope")
    supervisor.mark_request_end("nope")
    await supervisor.aclose()


async def test_unready_model_proxies_return_none(config: Config, tmp_path: Path) -> None:
    supervisor = sup(config, make_binary(tmp_path))
    assert await supervisor.props("ghost") is None
    assert await supervisor.slots("ghost") is None
    assert await supervisor.health("ghost") is False
    assert await supervisor.set_lora_scales("ghost", []) is False
    assert supervisor.base_url("ghost") is None
    assert supervisor.tail_log("ghost") == []
    assert supervisor.log_path("ghost") is None
    assert await supervisor.kill("ghost") is False
    await supervisor.stop("ghost")
    await supervisor.aclose()


async def test_unlaunchable_binary_raises_model_load_error(config: Config, tmp_path: Path) -> None:
    missing = tmp_path / "engine" / "does-not-exist.exe"
    supervisor = Supervisor(config, resolve_binary=resolver(missing))
    with pytest.raises(ModelLoadError) as excinfo:
        await supervisor.start(make_record(tmp_path), make_plan())
    assert "Could not launch llama-server" in excinfo.value.message
    assert supervisor.get("qwen2.5-7b") is None
    await supervisor.aclose()


# ---------------------------------------------------------------------------
# Live test against the real engine
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    REAL_ENGINE is None or REAL_MODEL is None,
    reason=(
        "real llama-server engine or Qwen2.5-0.5B model not available; install an "
        "engine into SF_DATA_DIR and point SF_TEST_MODELS_DIR at your GGUF library"
    ),
)
@pytest.mark.timeout(240)
async def test_live_real_llama_server(config: Config, tmp_path: Path) -> None:
    """Proves the ctx x parallel semantics against the real binary."""
    config.gateway.load_timeout_s = 90.0
    config.gateway.health_poll_interval_s = 0.5
    supervisor = Supervisor(config, resolve_binary=resolver(REAL_ENGINE))
    record = ModelRecord(id="qwen2.5-0.5b-live", name="Qwen2.5 0.5B", path=REAL_MODEL)
    plan = make_plan("qwen2.5-0.5b-live", devices=[0], ctx_size=2048, parallel=1)

    info = await supervisor.start(record, plan)
    assert info.state == "ready"
    pid = info.pid
    assert pid is not None

    props = await supervisor.props(record.id)
    assert props is not None, supervisor.tail_log(record.id, 40)
    assert props["default_generation_settings"]["n_ctx"] == 2048
    assert props["total_slots"] == 1

    slots = await supervisor.slots(record.id)
    assert isinstance(slots, list)
    assert len(slots) == 1

    await supervisor.stop(record.id)
    assert not psutil.pid_exists(pid) or not psutil.Process(pid).is_running()
    assert supervisor.get(record.id) is None
    await supervisor.aclose()


# ---------------------------------------------------------------------------
# Per-model lock table hygiene
# ---------------------------------------------------------------------------


async def test_stop_prunes_the_per_model_lock(config: Config, tmp_path: Path) -> None:
    """The lock table must not keep an entry per model id ever touched.

    Bounded for a static library, a slow leak once virtual models can be
    created and deleted through the API.
    """
    supervisor = sup(config, make_binary(tmp_path))
    await supervisor.stop("some/model")  # not running: still must not leave a lock
    assert supervisor._locks == {}

    assert await supervisor.kill("other/model") is False  # not running
    assert "other/model" not in supervisor._locks
