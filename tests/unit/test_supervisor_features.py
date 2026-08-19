"""WP20: the launch flags that depend on what the engine advertises.

Three things are pinned here, all of them argv-level and all of them pure:

* **Speculative decoding resolution.** ``spec_type: "auto"`` has to pick the
  right strategy per model, and every value has to be checked against the
  engine's own ``--spec-type`` list -- b10425 accepts renamed flags and ignores
  them (D2), so "configured but silently off" is the failure mode.
* **Tensor split gating.** ``--split-mode tensor`` is EXPERIMENTAL upstream and
  has hard prerequisites; llama.cpp enforces one of them by exiting
  ("SPLIT_MODE_TENSOR requires flash_attn to be enabled"). Refusing with a
  sentence beats a dead child and a stack trace.
* **The new quality-neutral flags** (``--cache-ram``, ``-ub``,
  ``--no-kv-unified``, ``--backend-sampling``) appear only when the active
  engine declares them.

The engine feature set comes from the same verbatim b10425 help excerpt
``test_engine_features`` uses, so these assertions are about the real flag
surface rather than about a fixture's idea of it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from studioforge.config import Config
from studioforge.core.engine import EngineFeatures, parse_engine_features
from studioforge.core.supervisor import (
    DEFAULT_SPEC_DRAFT_N_MAX,
    Supervisor,
    _Instance,
    resolve_spec_type,
    tensor_split_blockers,
)
from studioforge.errors import ModelLoadError
from studioforge.types import (
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
)

HELP_EXCERPT = (Path(__file__).parent / "data" / "b10425_help_excerpt.txt").read_text(
    encoding="utf-8"
)
B10425 = parse_engine_features(HELP_EXCERPT, "b10425")
UNKNOWN = EngineFeatures.unknown("b10425")


def engine_without(*spec_types: str) -> EngineFeatures:
    """b10425 minus some ``--spec-type`` values: an older or newer build."""
    keep = tuple(t for t in B10425.spec_types if t not in spec_types)
    return dataclasses.replace(B10425, spec_types=keep)


def engine_without_tensor() -> EngineFeatures:
    modes = tuple(m for m in B10425.split_modes if m != "tensor")
    return dataclasses.replace(B10425, split_modes=modes)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    return cfg


def make_binary(tmp_path: Path) -> Path:
    path = tmp_path / "engines" / "b10425" / "llama-server.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub", encoding="utf-8")
    return path


def resolver(path: Path) -> Callable[[str | None], Path]:
    def resolve(_tag: str | None) -> Path:
        return path

    return resolve


def sup(config: Config, binary: Path) -> Supervisor:
    return Supervisor(config, resolve_binary=resolver(binary))


def dense_meta(**kwargs: object) -> GgufMeta:
    """An ordinary dense, full-attention model: the tensor-split-eligible shape."""
    params: dict[str, object] = {
        "architecture": "qwen2",
        "n_layer": 32,
        "n_embd": 4096,
        "n_head": 32,
        "n_head_kv": 8,
    }
    params.update(kwargs)
    return GgufMeta(**params)  # type: ignore[arg-type]


def make_record(
    tmp_path: Path,
    model_id: str = "m",
    *,
    settings: ModelSettings | None = None,
    meta: GgufMeta | None = None,
    thinking: bool = False,
) -> ModelRecord:
    path = tmp_path / "models" / f"{model_id}.gguf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF")
    return ModelRecord(
        id=model_id,
        name=model_id,
        path=path,
        meta=meta if meta is not None else dense_meta(),
        capabilities=ModelCapabilities(thinking=thinking),
        settings=settings or ModelSettings(),
    )


def make_plan(**kwargs: object) -> LoadPlan:
    params: dict[str, object] = {
        "model_id": "m",
        "devices": [0, 1],
        "ctx_size": 8192,
        "flash_attn": "on",
        "kv_cache_type": "f16",
        "kv_cache_type_v": "f16",
        "split_mode": "layer",
    }
    params.update(kwargs)
    return LoadPlan(**params)  # type: ignore[arg-type]


def value_after(argv: Sequence[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# ---------------------------------------------------------------------------
# spec_type resolution matrix
# ---------------------------------------------------------------------------


def test_auto_picks_mtp_when_the_model_has_its_own_heads(tmp_path: Path) -> None:
    """``nextn_predict_layers`` is the only honest MTP signal -- a repo called
    "...-MTP-GGUF" in this library carries no such key at all."""
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    spec, reason = resolve_spec_type(record, B10425, has_draft=False)
    assert spec == "draft-mtp"
    assert "multi-token-prediction head" in reason


def test_auto_prefers_mtp_over_an_attached_draft_model(tmp_path: Path) -> None:
    """MTP needs no second model in VRAM, so it wins where both are possible."""
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 2}))
    assert resolve_spec_type(record, B10425, has_draft=True)[0] == "draft-mtp"


def test_auto_falls_back_to_draft_simple_when_a_draft_is_attached(tmp_path: Path) -> None:
    record = make_record(tmp_path)
    spec, reason = resolve_spec_type(record, B10425, has_draft=True)
    assert spec == "draft-simple"
    assert "draft model" in reason


def test_auto_picks_ngram_mod_for_a_thinking_model(tmp_path: Path) -> None:
    """Measured free on unseen prose (+0.4%) and a large win on output that
    repeats itself, which is what a reasoning model's chain of thought does."""
    record = make_record(tmp_path, thinking=True)
    spec, reason = resolve_spec_type(record, B10425, has_draft=False)
    assert spec == "ngram-mod"
    assert "thinking" in reason


def test_auto_picks_ngram_mod_for_a_moe_model(tmp_path: Path) -> None:
    record = make_record(tmp_path, meta=dense_meta(n_expert=128, n_expert_used=8))
    assert resolve_spec_type(record, B10425, has_draft=False)[0] == "ngram-mod"


def test_auto_is_none_for_a_plain_dense_model(tmp_path: Path) -> None:
    spec, reason = resolve_spec_type(make_record(tmp_path), B10425, has_draft=False)
    assert spec == "none"
    assert "nothing to draft from" in reason


def test_auto_skips_a_type_this_engine_does_not_offer(tmp_path: Path) -> None:
    """An engine without draft-mtp must not be handed draft-mtp; the model
    still gets whatever else it qualifies for."""
    record = make_record(
        tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}), thinking=True
    )
    assert resolve_spec_type(record, engine_without("draft-mtp"), has_draft=False)[0] == "ngram-mod"
    assert (
        resolve_spec_type(record, engine_without("draft-mtp", "ngram-mod"), has_draft=False)[0]
        == "none"
    )


def test_an_unknown_engine_keeps_the_pre_gating_behaviour(tmp_path: Path) -> None:
    """No help text means no guesses: a draft model still drafts (that is what
    the code did before this gating existed), and nothing new is invented."""
    mtp = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    assert resolve_spec_type(mtp, UNKNOWN, has_draft=False)[0] == "none"
    assert resolve_spec_type(mtp, UNKNOWN, has_draft=True)[0] == "draft-simple"


def test_an_explicit_type_the_engine_lacks_is_refused_not_ignored(tmp_path: Path) -> None:
    record = make_record(tmp_path, settings=ModelSettings(spec_type="draft-eagle3"))
    with pytest.raises(ModelLoadError) as excinfo:
        resolve_spec_type(record, engine_without("draft-eagle3"), has_draft=False)
    message = str(excinfo.value)
    assert "draft-eagle3" in message
    assert "draft-simple" in message, "the error must list what IS offered"


def test_an_explicit_type_is_honoured_verbatim(tmp_path: Path) -> None:
    record = make_record(tmp_path, settings=ModelSettings(spec_type="ngram-simple"))
    assert resolve_spec_type(record, B10425, has_draft=False)[0] == "ngram-simple"


def test_spec_type_none_turns_drafting_off_even_with_a_draft_model(tmp_path: Path) -> None:
    record = make_record(tmp_path, settings=ModelSettings(spec_type="none"))
    assert resolve_spec_type(record, B10425, has_draft=True)[0] == "none"


def test_a_stored_null_spec_type_reads_as_auto() -> None:
    """Settings rows written before WP20 carry ``null`` here; they must hydrate
    to the same sentinel as a freshly built object, or a model would resolve
    differently after a restart."""
    assert ModelSettings.model_validate({"spec_type": None}).spec_type == "auto"
    assert ModelSettings().spec_type == "auto"


# ---------------------------------------------------------------------------
# spec_type -> argv
# ---------------------------------------------------------------------------


def test_mtp_emits_no_draft_model_flags(config: Config, tmp_path: Path) -> None:
    """draft-mtp reads the base model's own heads; a --spec-draft-model there
    would load a second model into VRAM for nothing."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    argv = sup(config, binary).build_command(
        record, make_plan(devices=[0]), port=18100, features=B10425
    )
    assert value_after(argv, "--spec-type") == "draft-mtp"
    assert value_after(argv, "--spec-draft-n-max") == str(DEFAULT_SPEC_DRAFT_N_MAX)
    assert "--spec-draft-model" not in argv
    assert "--spec-draft-device" not in argv


def test_ngram_mod_emits_no_draft_depth(config: Config, tmp_path: Path) -> None:
    """The n-gram types read --spec-ngram-*-n-max, not --spec-draft-n-max, so
    emitting the latter would be a flag that looks like it does something."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, thinking=True)
    argv = sup(config, binary).build_command(
        record, make_plan(devices=[0]), port=18100, features=B10425
    )
    assert value_after(argv, "--spec-type") == "ngram-mod"
    assert "--spec-draft-n-max" not in argv


def test_no_spec_flags_at_all_for_a_plain_model(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(devices=[0]), port=18100, features=B10425
    )
    assert not [flag for flag in argv if flag.startswith("--spec")]


# ---------------------------------------------------------------------------
# Tensor split gating
# ---------------------------------------------------------------------------


def test_tensor_split_is_used_when_every_gate_passes(config: Config, tmp_path: Path) -> None:
    """The planner copies ``settings.split_mode`` onto the plan; the supervisor
    resolves it from there, because a request-level override travels the same
    way."""
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(split_mode="tensor"))
    plan = make_plan(split_mode="tensor")
    argv = sup(config, binary).build_command(record, plan, port=18100, features=B10425)
    assert value_after(argv, "--split-mode") == "tensor"


def test_layer_split_is_still_the_default(config: Config, tmp_path: Path) -> None:
    """Tensor mode measured *slower* than layer on this PCIe rig, and upstream
    calls it experimental. Nothing opts into it by itself."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425
    )
    assert value_after(argv, "--split-mode") == "layer"


@pytest.mark.parametrize(
    ("plan_kwargs", "record_kwargs", "fragment"),
    [
        ({"flash_attn": "auto"}, {}, "flash attention"),
        ({"kv_cache_type": "q8_0", "kv_cache_type_v": "q8_0"}, {}, "unquantized KV"),
        ({}, {"meta": dense_meta(n_expert=128, n_expert_used=8)}, "mixture-of-experts"),
        ({}, {"meta": dense_meta(extra={"full_attention_interval": 4})}, "hybrid"),
        ({}, {"meta": GgufMeta()}, "attention layout"),
    ],
)
def test_tensor_split_blockers_name_the_reason(
    tmp_path: Path,
    plan_kwargs: dict[str, object],
    record_kwargs: dict[str, object],
    fragment: str,
) -> None:
    record = make_record(tmp_path, **record_kwargs)  # type: ignore[arg-type]
    blockers = tensor_split_blockers(record, make_plan(**plan_kwargs), B10425)
    assert any(fragment in reason for reason in blockers), blockers


def test_a_single_device_placement_blocks_tensor_mode(tmp_path: Path) -> None:
    blockers = tensor_split_blockers(make_record(tmp_path), make_plan(devices=[0]), B10425)
    assert any("single GPU" in reason for reason in blockers)


def test_an_engine_without_tensor_mode_blocks_it(tmp_path: Path) -> None:
    blockers = tensor_split_blockers(make_record(tmp_path), make_plan(), engine_without_tensor())
    assert any("--split-mode tensor" in reason for reason in blockers)


def test_explicit_tensor_is_refused_with_the_reason(config: Config, tmp_path: Path) -> None:
    """A user who typed "tensor" and silently got "layer" would go on to
    benchmark the wrong thing, so an ineligible explicit request is an error."""
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path,
        settings=ModelSettings(split_mode="tensor"),
        meta=dense_meta(n_expert=128, n_expert_used=8),
    )
    with pytest.raises(ModelLoadError) as excinfo:
        sup(config, binary).build_command(
            record, make_plan(split_mode="tensor"), port=18100, features=B10425
        )
    assert "mixture-of-experts" in str(excinfo.value)


def test_auto_downgrades_to_layer_and_says_why(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(
        tmp_path,
        settings=ModelSettings(split_mode="auto"),
        meta=dense_meta(extra={"full_attention_interval": 4}),
    )
    supervisor = sup(config, binary)
    plan = make_plan(split_mode="auto")
    decided = supervisor.resolve(record, plan, B10425)
    assert decided.split_mode == "layer"
    assert decided.split_mode_reason is not None
    assert "hybrid" in decided.split_mode_reason
    argv = supervisor.build_command(record, plan, port=18100, features=B10425, resolved=decided)
    assert value_after(argv, "--split-mode") == "layer"


def test_auto_reaches_tensor_when_it_can(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(split_mode="auto"))
    decided = sup(config, binary).resolve(record, make_plan(split_mode="auto"), B10425)
    assert decided.split_mode == "tensor"
    assert decided.split_mode_reason is None


def test_a_single_gpu_plan_still_says_split_mode_none(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(split_mode="auto"))
    argv = sup(config, binary).build_command(
        record, make_plan(devices=[0], split_mode="auto"), port=18100, features=B10425
    )
    assert value_after(argv, "--split-mode") == "none"


# ---------------------------------------------------------------------------
# The quality-neutral flags
# ---------------------------------------------------------------------------


def test_cache_ram_is_on_by_default_and_bounded(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425
    )
    value = int(value_after(argv, "--cache-ram"))
    assert 1024 <= value <= 32768


def test_cache_ram_honours_an_explicit_value(config: Config, tmp_path: Path) -> None:
    config.engine.cache_ram_mb = 4096
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425
    )
    assert value_after(argv, "--cache-ram") == "4096"


def test_no_new_flags_reach_an_engine_that_does_not_advertise_them(
    config: Config, tmp_path: Path
) -> None:
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=4), port=18100, features=UNKNOWN
    )
    for flag in ("--cache-ram", "--no-kv-unified", "--backend-sampling"):
        assert flag not in argv


def test_ubatch_comes_from_config_and_the_model_wins(config: Config, tmp_path: Path) -> None:
    config.engine.ubatch_size = 1024
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425
    )
    assert value_after(argv, "--ubatch-size") == "1024"

    pinned = make_record(tmp_path, settings=ModelSettings(ubatch_size=2048))
    argv = sup(config, binary).build_command(pinned, make_plan(), port=18100, features=B10425)
    assert value_after(argv, "--ubatch-size") == "2048"
    assert argv.count("--ubatch-size") == 1


def test_ubatch_is_absent_by_default(config: Config, tmp_path: Path) -> None:
    """The planner's compute-buffer estimate is calibrated against the engine's
    own 512, so nothing raises it without the operator saying so."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425
    )
    assert "--ubatch-size" not in argv


def test_backend_sampling_is_opt_in(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record, plan = make_record(tmp_path), make_plan()
    assert "--backend-sampling" not in sup(config, binary).build_command(
        record, plan, port=18100, features=B10425
    )
    config.engine.backend_sampling = True
    assert "--backend-sampling" in sup(config, binary).build_command(
        record, plan, port=18100, features=B10425
    )


# ---------------------------------------------------------------------------
# Unified KV
# ---------------------------------------------------------------------------


def test_multi_slot_launches_pin_the_partitioned_kv_pool(config: Config, tmp_path: Path) -> None:
    """Measured (D38): with the pool partitioned a too-long request is refused
    up front with a 400 naming the limit, so ``ctx_per_slot`` is a promise.
    Unified, two concurrent long requests both died with a 500 mid-generation.
    The engine's default depends on the slot count, so we say it out loud."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=4), port=18100, features=B10425
    )
    assert "--no-kv-unified" in argv
    assert "--kv-unified" not in argv


def test_a_single_slot_launch_says_nothing_about_the_kv_pool(
    config: Config, tmp_path: Path
) -> None:
    """At one slot the two shapes are identical, and keeping the flag off there
    keeps the default launch byte-identical to before the estimator (D17)."""
    binary = make_binary(tmp_path)
    argv = sup(config, binary).build_command(
        make_record(tmp_path), make_plan(parallel=1), port=18100, features=B10425
    )
    assert "--no-kv-unified" not in argv


def test_kv_unified_opt_in_still_wins(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    record = make_record(tmp_path, settings=ModelSettings(kv_unified=True))
    argv = sup(config, binary).build_command(
        record, make_plan(parallel=4), port=18100, features=B10425
    )
    assert "--kv-unified" in argv
    assert "--no-kv-unified" not in argv


# ---------------------------------------------------------------------------
# What the rest of the app gets to render
# ---------------------------------------------------------------------------


def test_the_resolution_is_published_on_the_plan_and_the_instance(
    config: Config, tmp_path: Path
) -> None:
    """``InstanceInfo.plan`` IS the plan object, so writing the resolved values
    back onto it is what makes the catalog, the Dashboard and /api/models show
    what the child is really doing rather than what was asked for."""
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    plan = make_plan(devices=[0])
    inst = _Instance(
        info=InstanceInfo(model_id="m", state="loading", plan=plan),
        record=record,
        plan=plan,
        port=18100,
        engine_tag="b10425",
        draft=None,
        adapters=(),
        log_path=tmp_path / "m.log",
    )
    supervisor._record_resolution(inst, supervisor.resolve(record, plan, B10425))
    assert inst.info.speculative == {
        "type": "draft-mtp",
        "draft_n_max": DEFAULT_SPEC_DRAFT_N_MAX,
        "draft_model_id": None,
        "reason": "the model carries 1 multi-token-prediction head(s)",
    }
    assert inst.plan.speculative == inst.info.speculative


def test_a_downgraded_split_mode_lands_in_the_plan_notes(config: Config, tmp_path: Path) -> None:
    binary = make_binary(tmp_path)
    supervisor = sup(config, binary)
    record = make_record(
        tmp_path,
        settings=ModelSettings(split_mode="auto"),
        meta=dense_meta(n_expert=128, n_expert_used=8),
    )
    plan = make_plan(split_mode="auto")
    inst = _Instance(
        info=InstanceInfo(model_id="m", state="loading", plan=plan),
        record=record,
        plan=plan,
        port=18100,
        engine_tag="b10425",
        draft=None,
        adapters=(),
        log_path=tmp_path / "m.log",
    )
    supervisor._record_resolution(inst, supervisor.resolve(record, plan, B10425))
    assert plan.split_mode == "layer"
    assert plan.split_mode_reason is not None
    assert any("mixture-of-experts" in note for note in plan.notes)
