"""Context checkpoints as first-class settings, and what an iSWA launch really does (D72).

A model whose cache cannot be rolled back -- a sliding-window ("iswa") or
hybrid (recurrent) one -- can only resume a prompt that changed part-way
through from a context checkpoint. llama.cpp makes one at every user-message
start it needs, ALWAYS at the last user message and at the end of every
prompt, so the engine's 8192-token spacing already covers a chat's next turn;
what the default does not bound is host RAM -- each checkpoint is a copy of
the slot's sliding window, ~800 MiB at f16 for a Gemma-4-shaped 31B, 32 per
slot. So:

* ``ctx_checkpoints`` / ``checkpoint_min_step`` are per-model settings (``0``
  meaning off / no minimum), ``models.auto_ctx_checkpoints`` (8) and
  ``models.auto_checkpoint_min_step`` (null: the engine's spacing) are the
  automatic default for iswa/hybrid models only, a per-model value always
  wins and ``extra_flags`` still override last;
* ``effective`` reports each value with where it came from;
* a flag the engine does not advertise is never passed (D38), and a saved
  value it cannot take is named inert;
* the default ``--cache-reuse`` is not passed to an iSWA model, which the
  engine refuses to shift anyway ("cache_reuse is not supported by this
  context"), so ``effective`` stops claiming reuse 256;
* ``slots_debug`` starts the child with the engine's slot-debug variables.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from studioforge.config import Config
from studioforge.core import supervisor as supervisor_module
from studioforge.core.supervisor import (
    SLOTS_DEBUG_ENV,
    CheckpointChoice,
    Supervisor,
    effective_launch,
    resolve_checkpoints,
    slots_debug_env,
)
from studioforge.errors import ModelLoadError
from studioforge.types import GgufMeta, LoadPlan, ModelRecord, ModelSettings
from tests.unit.test_catalog import dense_meta, hybrid_meta, iswa_meta
from tests.unit.test_supervisor import make_binary, resolver
from tests.unit.test_supervisor_features import B10425, UNKNOWN


def _config(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path / "data")
    # The supervisor tests' own port range, away from anything live.
    config.gateway.child_port_start = 19480
    config.gateway.child_port_end = 19489
    config.ensure_dirs()
    return config


def _record(
    tmp_path: Path, meta: GgufMeta | None, settings: ModelSettings | None = None
) -> ModelRecord:
    path = tmp_path / "models" / "model.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return ModelRecord(
        id="pub/model",
        name="model",
        path=path,
        meta=meta,
        settings=settings or ModelSettings(),
    )


def _plan() -> LoadPlan:
    return LoadPlan(model_id="pub/model", devices=[0], ctx_size=65536, ctx_per_slot=65536)


def _launch(
    tmp_path: Path,
    meta: GgufMeta | None,
    settings: ModelSettings | None = None,
    *,
    features: Any = B10425,
    config: Config | None = None,
) -> tuple[list[str], Any]:
    config = config or _config(tmp_path)
    record = _record(tmp_path, meta, settings)
    supervisor = Supervisor(config, resolve_binary=resolver(make_binary(tmp_path)))
    argv = supervisor.build_command(record, _plan(), port=18100, features=features)
    kind = supervisor_module.attention_kind(meta) if meta is not None else None
    eff = effective_launch(
        argv,
        features,
        _plan(),
        record.settings,
        checkpoints=resolve_checkpoints(record, config.models),
        attention=kind,
    )
    return argv, eff


def _last(argv: list[str], flag: str) -> str | None:
    positions = [i for i, token in enumerate(argv) if token == flag]
    return argv[positions[-1] + 1] if positions else None


# ---------------------------------------------------------------------------
# The automatic default: a count for caches that cannot roll back
# ---------------------------------------------------------------------------


def test_a_sliding_window_model_keeps_eight_checkpoints_at_the_engines_spacing(
    tmp_path: Path,
) -> None:
    argv, eff = _launch(tmp_path, iswa_meta())
    assert _last(argv, "--ctx-checkpoints") == "8"
    assert "--checkpoint-min-step" not in argv, "the engine's 8192 is kept"
    assert eff.ctx_checkpoints == 8
    assert eff.checkpoint_min_step == B10425.checkpoint_min_step_default == 8192
    assert eff.checkpoint_sources == {
        "ctx_checkpoints": "auto",
        "checkpoint_min_step": "engine_default",
    }
    assert "checkpoints 8 per slot" in eff.summary


def test_a_hybrid_model_gets_the_same_cap(tmp_path: Path) -> None:
    argv, eff = _launch(tmp_path, hybrid_meta())
    assert _last(argv, "--ctx-checkpoints") == "8"
    assert eff.checkpoint_sources["ctx_checkpoints"] == "auto"


@pytest.mark.parametrize("meta", [dense_meta(), None], ids=["full attention", "no metadata"])
def test_a_model_that_rolls_back_for_free_gets_no_checkpoint_flag(
    tmp_path: Path, meta: GgufMeta | None
) -> None:
    """llama.cpp never checkpoints a full-attention cache; a launch that says
    nothing keeps that launch's argv exactly as it was."""
    argv, eff = _launch(tmp_path, meta)
    assert "--ctx-checkpoints" not in argv and "--checkpoint-min-step" not in argv
    assert eff.checkpoint_sources == {
        "ctx_checkpoints": "engine_default",
        "checkpoint_min_step": "engine_default",
    }
    assert "checkpoints" not in eff.summary


def test_the_automatic_values_are_the_operators_to_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.models.auto_ctx_checkpoints = 16
    config.models.auto_checkpoint_min_step = 1024
    argv, eff = _launch(tmp_path, iswa_meta(), config=config)
    assert _last(argv, "--ctx-checkpoints") == "16"
    assert _last(argv, "--checkpoint-min-step") == "1024"
    assert eff.checkpoint_sources == {"ctx_checkpoints": "auto", "checkpoint_min_step": "auto"}
    assert "checkpoints 16 per slot, 1024-token spacing" in eff.summary

    config.models.auto_ctx_checkpoints = None
    config.models.auto_checkpoint_min_step = None
    argv, _eff = _launch(tmp_path, iswa_meta(), config=config)
    assert "--ctx-checkpoints" not in argv, "null hands the count back to the engine"


def test_the_shipped_defaults_are_eight_and_the_engines_spacing() -> None:
    models = Config(data_dir="/tmp/sf-test").models
    assert models.auto_ctx_checkpoints == 8
    assert models.auto_checkpoint_min_step is None


# ---------------------------------------------------------------------------
# Explicit always wins; extra_flags still win last
# ---------------------------------------------------------------------------


def test_a_per_model_value_wins_even_where_the_automatic_default_would_not_apply(
    tmp_path: Path,
) -> None:
    settings = ModelSettings(ctx_checkpoints=4, checkpoint_min_step=2048)
    argv, eff = _launch(tmp_path, dense_meta(), settings)
    assert _last(argv, "--ctx-checkpoints") == "4"
    assert _last(argv, "--checkpoint-min-step") == "2048"
    assert eff.checkpoint_sources == {"ctx_checkpoints": "model", "checkpoint_min_step": "model"}


def test_zero_turns_checkpoints_off_for_a_model_that_would_get_them(tmp_path: Path) -> None:
    argv, eff = _launch(tmp_path, iswa_meta(), ModelSettings(ctx_checkpoints=0))
    assert _last(argv, "--ctx-checkpoints") == "0"
    assert eff.ctx_checkpoints == 0
    assert eff.checkpoint_sources["ctx_checkpoints"] == "model"


def test_extra_flags_still_override_last_and_the_report_says_who_won(tmp_path: Path) -> None:
    settings = ModelSettings(extra_flags="--ctx-checkpoints 3")
    argv, eff = _launch(tmp_path, iswa_meta(), settings)
    assert argv.count("--ctx-checkpoints") == 2
    assert _last(argv, "--ctx-checkpoints") == "3"
    assert eff.ctx_checkpoints == 3
    assert eff.checkpoint_sources["ctx_checkpoints"] == "extra_flags"


def test_a_negative_value_is_refused_when_it_is_saved() -> None:
    with pytest.raises(ValidationError):
        ModelSettings(ctx_checkpoints=-1)
    with pytest.raises(ValidationError):
        ModelSettings(checkpoint_min_step=-8)


def test_the_resolution_is_pure_and_names_its_sources(tmp_path: Path) -> None:
    models = Config(data_dir=tmp_path / "data").models
    assert resolve_checkpoints(_record(tmp_path, iswa_meta()), models) == CheckpointChoice(
        ctx_checkpoints=8, ctx_checkpoints_source="auto"
    )
    assert resolve_checkpoints(_record(tmp_path, dense_meta()), models) == CheckpointChoice()
    explicit = _record(tmp_path, dense_meta(), ModelSettings(checkpoint_min_step=0))
    assert resolve_checkpoints(explicit, models) == CheckpointChoice(
        checkpoint_min_step=0, checkpoint_min_step_source="model"
    )


# ---------------------------------------------------------------------------
# D38: only what the engine advertises
# ---------------------------------------------------------------------------


def test_an_engine_without_the_flags_gets_none_and_a_saved_value_is_named_inert(
    tmp_path: Path,
) -> None:
    older = dataclasses.replace(
        B10425,
        flags=B10425.flags - {"--ctx-checkpoints", "--checkpoint-min-step", "-ctxcp", "-cms"},
        ctx_checkpoints=False,
    )
    argv, eff = _launch(tmp_path, iswa_meta(), features=older)
    assert "--ctx-checkpoints" not in argv, "the automatic default is simply not passed"
    assert "ctx_checkpoints" not in eff.inert

    argv, eff = _launch(tmp_path, iswa_meta(), ModelSettings(ctx_checkpoints=4), features=older)
    assert "--ctx-checkpoints" not in argv
    assert "ctx_checkpoints" in eff.inert, "a value somebody saved must not look honoured"


def test_an_engine_whose_help_could_not_be_read_gets_no_new_flag(tmp_path: Path) -> None:
    argv, _eff = _launch(tmp_path, iswa_meta(), ModelSettings(ctx_checkpoints=4), features=UNKNOWN)
    assert "--ctx-checkpoints" not in argv


# ---------------------------------------------------------------------------
# --cache-reuse on a sliding-window cache
# ---------------------------------------------------------------------------


def test_the_default_chunk_reuse_is_not_passed_to_a_sliding_window_model(
    tmp_path: Path,
) -> None:
    """llama.cpp: "cache_reuse is not supported by this context, it will be
    disabled" -- the report now says what the child does."""
    argv, eff = _launch(tmp_path, iswa_meta())
    assert "--cache-reuse" not in argv
    assert eff.cache_reuse == 0
    assert "chunk reuse off" in eff.summary
    assert "cache_reuse" not in eff.inert


@pytest.mark.parametrize("meta", [dense_meta(), hybrid_meta()], ids=["dense", "hybrid"])
def test_models_that_can_shift_keep_the_default_chunk_reuse(tmp_path: Path, meta: GgufMeta) -> None:
    argv, eff = _launch(tmp_path, meta)
    assert _last(argv, "--cache-reuse") == "256"
    assert eff.cache_reuse == 256


def test_an_explicit_chunk_reuse_on_a_sliding_window_model_is_passed_and_named_inert(
    tmp_path: Path,
) -> None:
    argv, eff = _launch(tmp_path, iswa_meta(), ModelSettings(cache_reuse=512))
    assert _last(argv, "--cache-reuse") == "512", "an explicit value is honoured verbatim"
    assert eff.cache_reuse == 0, "and reported as what the engine does with it"
    assert "cache_reuse" in eff.inert


# ---------------------------------------------------------------------------
# slots_debug: the engine's own divergence report, per model
# ---------------------------------------------------------------------------


def test_slots_debug_is_off_unless_asked_for() -> None:
    assert slots_debug_env(ModelSettings()) == {}
    assert slots_debug_env(ModelSettings(slots_debug=True)) == {
        "LLAMA_SERVER_SLOTS_DEBUG": "1",
        "LLAMA_SERVER_SLOTS_N_DIFF": "32",
    }
    assert SLOTS_DEBUG_ENV["LLAMA_SERVER_SLOTS_DEBUG"] == "1"


def test_the_report_warns_that_prompt_text_goes_to_the_log() -> None:
    plan = _plan()
    argv = ["llama-server", "--ctx-size", "65536", "--parallel", "1"]
    on = effective_launch(argv, B10425, plan, slots_debug=True)
    off = effective_launch(argv, B10425, plan)
    assert on.slots_debug is True and "SLOTS DEBUG on" in on.summary
    assert off.slots_debug is False and "SLOTS DEBUG" not in off.summary


@pytest.mark.parametrize("asked", [True, False])
async def test_the_child_environment_carries_the_debug_variables_only_when_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asked: bool
) -> None:
    """Captured at the spawn and then refused, so nothing is ever launched."""
    config = _config(tmp_path)
    supervisor = Supervisor(config, resolve_binary=resolver(make_binary(tmp_path)))
    supervisor._features[""] = B10425  # noqa: SLF001 - skip reading a help text
    seen: dict[str, Any] = {}

    async def refuse(*argv: str, **kwargs: Any) -> Any:
        seen["env"] = dict(kwargs.get("env") or {})
        raise OSError("not launched by this test")

    monkeypatch.setattr(supervisor_module.asyncio, "create_subprocess_exec", refuse)
    record = _record(tmp_path, iswa_meta(), ModelSettings(slots_debug=asked))
    try:
        with pytest.raises(ModelLoadError):
            await supervisor.start(record, _plan())
    finally:
        await supervisor.aclose()
    env = seen["env"]
    for name, value in SLOTS_DEBUG_ENV.items():
        assert (env.get(name) == value) is asked
