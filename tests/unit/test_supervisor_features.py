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
from types import SimpleNamespace

import pytest

from studioforge.config import CACHE_RAM_MIN_GRANT_MIB, Config, grant_cache_ram_mib
from studioforge.core import supervisor as supervisor_module
from studioforge.core.engine import EngineFeatures, parse_engine_features
from studioforge.core.supervisor import (
    DEFAULT_SPEC_DRAFT_N_MAX,
    ENGINE_DEFAULT_CACHE_RAM_MIB,
    ENGINE_DEFAULT_CHECKPOINT_MIN_STEP,
    ENGINE_DEFAULT_CTX_CHECKPOINTS,
    ENGINE_DEFAULT_SLOT_PROMPT_SIMILARITY,
    ENGINE_DEFAULT_UBATCH_SIZE,
    Supervisor,
    _Instance,
    effective_launch,
    redact_argv,
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


def test_auto_disables_speculation_above_the_slot_threshold(tmp_path: Path) -> None:
    """Speculation is a single-stream win; a saturated multi-slot batch has no
    spare compute for it, so auto turns it off past SPEC_AUTO_MAX_SLOTS -- even
    for an MTP model that would otherwise draft (the observed 8-slot case)."""
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    assert resolve_spec_type(record, B10425, has_draft=False, slots=4)[0] == "draft-mtp"
    spec, reason = resolve_spec_type(record, B10425, has_draft=False, slots=8)
    assert spec == "none"
    assert "8 slots" in reason


def test_auto_slot_gate_also_silences_ngram_and_draft(tmp_path: Path) -> None:
    """The gate is before every what-to-draft-from check, so a thinking model
    (ngram) and an attached draft model both go quiet at high concurrency too."""
    thinking = make_record(tmp_path, thinking=True)
    assert resolve_spec_type(thinking, B10425, has_draft=False, slots=8)[0] == "none"
    plain = make_record(tmp_path)
    assert resolve_spec_type(plain, B10425, has_draft=True, slots=8)[0] == "none"


def test_an_explicit_spec_type_survives_high_concurrency(tmp_path: Path) -> None:
    """The slot gate is an auto-only default; a caller who set spec_type meant
    it (a benchmark measuring speculation at 8 slots, say)."""
    record = make_record(tmp_path, settings=ModelSettings(spec_type="draft-mtp"))
    assert resolve_spec_type(record, B10425, has_draft=False, slots=8)[0] == "draft-mtp"


def test_effective_ubatch_precedence() -> None:
    """The one policy the planner and supervisor share. Per-model wins, then the
    rig-wide setting, then the many-slots raise, then None (engine default)."""
    from studioforge.core.planner import effective_ubatch

    def ub(**kw: object) -> int | None:
        base = {
            "settings_ubatch": None,
            "engine_ubatch": None,
            "engine_ubatch_many_slots": 1024,
            "slots": 8,
        }
        base.update(kw)
        return effective_ubatch(**base)  # type: ignore[arg-type]

    assert ub(settings_ubatch=4096) == 4096, "per-model wins over everything"
    assert ub(engine_ubatch=768) == 768, "rig-wide setting wins over the raise"
    assert ub() == 1024, "the many-slots raise applies above the threshold"
    assert ub(slots=4) is None, "at the threshold the engine keeps 512"
    assert ub(slots=1) is None, "a single stream keeps 512"
    assert ub(engine_ubatch_many_slots=None) is None, "raise off => None even at 8 slots"


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


# ---------------------------------------------------------------------------
# --cache-ram is a machine-wide pool under "auto" (D50)
# ---------------------------------------------------------------------------


def hold_cache_ram(supervisor: Supervisor, tmp_path: Path, **grants: int) -> None:
    """Put live children in the table holding the given grants, in MiB."""
    for model_id, granted in grants.items():
        plan = make_plan()
        supervisor._instances[model_id] = _Instance(
            info=InstanceInfo(model_id=model_id, state="ready", plan=plan, cache_ram_mib=granted),
            record=make_record(tmp_path, model_id),
            plan=plan,
            port=18100,
            engine_tag="b10425",
            draft=None,
            adapters=(),
            log_path=tmp_path / f"{model_id}.log",
        )


def test_the_automatic_grant_is_what_the_other_children_are_not_holding(
    config: Config, tmp_path: Path
) -> None:
    """Before D50 every child was handed the whole 25%-of-RAM allowance, so four
    residents promised four times it and the "this can never make the box swap"
    comment was only true for one of them."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    assert pool is not None

    hold_cache_ram(supervisor, tmp_path, first=pool // 2)

    assert supervisor._cache_ram_grant() == pool - pool // 2


def test_the_grants_of_the_live_children_add_up_to_the_pool(config: Config, tmp_path: Path) -> None:
    """To the pool, not to a multiple of it. Before D50 each of these children
    was handed the whole allowance."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    assert pool is not None and pool > 4 * CACHE_RAM_MIN_GRANT_MIB

    share = pool // 4
    hold_cache_ram(supervisor, tmp_path, m0=share, m1=share)
    grant = supervisor._cache_ram_grant()

    assert grant == pool - 2 * share
    assert 2 * share + grant <= pool, "three residents, one pool's worth of RAM"


def test_a_starved_grant_is_floored_rather_than_switched_off(
    config: Config, tmp_path: Path
) -> None:
    """``--cache-ram 0`` does not mean "a small cache", it turns the host prompt
    cache off. A starved cache still recovers some evicted prefixes."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    assert pool is not None

    hold_cache_ram(supervisor, tmp_path, hog=pool)

    assert supervisor._cache_ram_grant() == CACHE_RAM_MIN_GRANT_MIB


def test_the_floor_pushing_past_the_pool_is_said_out_loud(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the floor the pool is an intention rather than a bound, and the
    operator staring at a swapping box needs the numbers to see that."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    assert pool is not None
    hold_cache_ram(supervisor, tmp_path, hog=pool)

    warnings: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        supervisor_module,
        "log",
        SimpleNamespace(
            warning=lambda event, **fields: warnings.append((event, fields)),
            info=lambda *_a, **_kw: None,
            debug=lambda *_a, **_kw: None,
        ),
    )
    supervisor._cache_ram_grant()

    assert [event for event, _ in warnings] == ["cache_ram_pool_oversubscribed"]
    fields = warnings[0][1]
    assert fields["pool_mib"] == pool
    assert fields["held_mib"] == pool
    assert fields["grant_mib"] == CACHE_RAM_MIN_GRANT_MIB


def test_an_arriving_child_does_not_have_to_share_with_itself(
    config: Config, tmp_path: Path
) -> None:
    """``start`` inserts the instance before ``_spawn`` runs, so its own row is
    in the table by the time the grant is computed. A reload re-grants correctly
    for the same reason from the other side: the outgoing child is stopped, and
    out of the table, before the replacement spawns."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    assert pool is not None
    hold_cache_ram(supervisor, tmp_path, mine=pool // 2)

    assert supervisor._cache_ram_grant(exclude="mine") == pool


def test_an_explicit_value_stays_per_child_verbatim(config: Config, tmp_path: Path) -> None:
    """The operator named a number; every child gets that number (D14). Only the
    automatic setting ever claimed the total was bounded."""
    config.engine.cache_ram_mb = 8192
    supervisor = sup(config, make_binary(tmp_path))
    hold_cache_ram(supervisor, tmp_path, first=8192, second=8192, third=8192)

    assert supervisor._cache_ram_grant() == 8192


@pytest.mark.parametrize("value", [0, -1])
def test_the_disabling_values_are_passed_through_untouched(
    config: Config, tmp_path: Path, value: int
) -> None:
    config.engine.cache_ram_mb = value
    supervisor = sup(config, make_binary(tmp_path))
    assert supervisor._cache_ram_grant() == value


def test_an_unlimited_grant_is_not_counted_as_a_negative_share(
    config: Config, tmp_path: Path
) -> None:
    """You cannot subtract "all of it" from a pool, and nobody may end up with
    MORE than the pool because somebody else holds -1."""
    supervisor = sup(config, make_binary(tmp_path))
    pool = supervisor._cache_ram_grant()
    hold_cache_ram(supervisor, tmp_path, unlimited=-1)

    assert supervisor._cache_ram_grant() == pool


def test_the_pool_arithmetic_needs_no_supervisor() -> None:
    assert grant_cache_ram_mib(32768, 0) == 32768
    assert grant_cache_ram_mib(32768, 8192) == 24576
    assert grant_cache_ram_mib(32768, 32768) == CACHE_RAM_MIN_GRANT_MIB
    assert grant_cache_ram_mib(32768, 999999) == CACHE_RAM_MIN_GRANT_MIB
    assert grant_cache_ram_mib(32768, -5000) == 32768


def test_the_argv_carries_the_grant_the_instance_records(config: Config, tmp_path: Path) -> None:
    """One computation, two readers: a second one could disagree with the first
    and then GET /api/models would be reporting a number nothing was launched
    with."""
    supervisor = sup(config, make_binary(tmp_path))
    argv = supervisor.build_command(
        make_record(tmp_path), make_plan(), port=18100, features=B10425, cache_ram_mib=1234
    )
    assert value_after(argv, "--cache-ram") == "1234"


# ---------------------------------------------------------------------------
# Which build a launch actually lands on (D50)
# ---------------------------------------------------------------------------


def test_the_resolved_engine_tag_comes_from_the_injected_resolver(
    config: Config, tmp_path: Path
) -> None:
    """The engine manager's own answer, taken off the EngineInfo the binary was
    chosen from -- not a string parsed out of a path."""
    supervisor = Supervisor(
        config,
        resolve_binary=resolver(make_binary(tmp_path)),
        resolve_engine_tag=lambda tag: tag or "b10689",
    )
    assert supervisor.active_engine_tag() == "b10689"
    assert supervisor.resolved_engine_tag("b10425") == "b10425"


def test_without_a_resolver_the_build_directory_names_the_engine(
    config: Config, tmp_path: Path
) -> None:
    """A plausible tag beats None for a supervisor built without an engine
    manager -- the GUI's command preview, every test."""
    assert sup(config, make_binary(tmp_path)).active_engine_tag() == "b10425"


def test_an_unresolvable_engine_answers_none_rather_than_raising(config: Config) -> None:
    """Naming the build is bookkeeping. A load must not fail because of it, and
    None reads as "cannot prove this child is current", which reloads."""

    def explode(_tag: str | None) -> Path:
        raise RuntimeError("no engine is installed")

    supervisor = Supervisor(config, resolve_binary=explode)
    assert supervisor.active_engine_tag() is None


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


def test_a_micro_batch_above_the_logical_batch_raises_the_batch(
    config: Config, tmp_path: Path
) -> None:
    """llama.cpp clamps ``n_ubatch`` to ``n_batch``: ``-ub 4096`` against the
    default ``-b 2048`` silently runs at 2048 (D40). The automatic batch is
    raised to cover it; an explicit ``batch_size`` is the user's and stays."""
    from studioforge.core.supervisor import BATCH_SIZE_MANY_SLOTS

    binary = make_binary(tmp_path)
    big = make_record(tmp_path, settings=ModelSettings(ubatch_size=4096))
    argv = sup(config, binary).build_command(big, make_plan(), port=18100, features=B10425)
    assert value_after(argv, "--ubatch-size") == "4096"
    assert value_after(argv, "--batch-size") == "4096"
    assert argv.count("--batch-size") == 1

    # Below the default logical batch nothing is added.
    small = make_record(tmp_path, settings=ModelSettings(ubatch_size=1024))
    argv = sup(config, binary).build_command(small, make_plan(), port=18100, features=B10425)
    assert "--batch-size" not in argv

    # Many slots already raise the batch; a micro-batch above THAT raises it further.
    argv = sup(config, binary).build_command(
        make_record(tmp_path, settings=ModelSettings(ubatch_size=8192)),
        make_plan(parallel=6),
        port=18100,
        features=B10425,
    )
    assert argv.count("--batch-size") == 1
    assert int(value_after(argv, "--batch-size")) == max(BATCH_SIZE_MANY_SLOTS, 8192)

    # The user's own batch_size is honoured verbatim, even below the micro-batch.
    pinned = make_record(tmp_path, settings=ModelSettings(ubatch_size=4096, batch_size=1024))
    argv = sup(config, binary).build_command(pinned, make_plan(), port=18100, features=B10425)
    assert argv.count("--batch-size") == 1
    assert value_after(argv, "--batch-size") == "1024"


# ---------------------------------------------------------------------------
# What the child was REALLY launched with (D54)
# ---------------------------------------------------------------------------


def engine_without_flags(*flags: str) -> EngineFeatures:
    """b10425 minus some flags: an older build, still *known*."""
    return dataclasses.replace(B10425, flags=frozenset(B10425.flags - set(flags)))


def test_effective_launch_reads_engine_defaults_when_nothing_is_passed(
    config: Config, tmp_path: Path
) -> None:
    """A one-slot launch passes none of the cache switches. The report must
    still say what the engine does with them -- "not passed" is not "off"."""
    config.engine.cache_ram_mb = 0  # explicit off: --cache-ram 0 IS passed
    argv = sup(config, make_binary(tmp_path)).build_command(
        make_record(tmp_path), make_plan(parallel=1), port=18100, features=B10425
    )
    eff = effective_launch(argv, B10425, make_plan(parallel=1))
    assert eff.cache_prompt is True and eff.sources["cache_prompt"] == "engine_default"
    assert eff.cont_batching is True and eff.sources["cont_batching"] == "engine_default"
    assert eff.kv_unified is False and eff.sources["kv_unified"] == "engine_default"
    assert eff.slot_prompt_similarity == pytest.approx(ENGINE_DEFAULT_SLOT_PROMPT_SIMILARITY)
    assert eff.sources["slot_prompt_similarity"] == "engine_default"
    assert eff.ubatch_size == ENGINE_DEFAULT_UBATCH_SIZE
    assert eff.ctx_checkpoints == ENGINE_DEFAULT_CTX_CHECKPOINTS
    assert eff.checkpoint_min_step == ENGINE_DEFAULT_CHECKPOINT_MIN_STEP
    assert eff.parallel == 1 and eff.ctx_per_slot == 8192 and eff.ctx_total == 8192
    assert eff.spec_type == "none"
    assert eff.inert == []


def test_effective_launch_shows_cache_reuse_on_at_256_when_the_setting_is_null(
    config: Config, tmp_path: Path
) -> None:
    """SPEC A1's premise: ``cache_reuse: null`` was read as "off". The child
    is launched with ``--cache-reuse 256 --cache-ram <pool>
    --slot-prompt-similarity 0.3``, and the report says so, from the argv."""
    record = make_record(tmp_path)
    assert record.settings.cache_reuse is None
    assert record.settings.cont_batching is None
    plan = make_plan(parallel=3)
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, plan, port=18100, features=B10425, cache_ram_mib=32603
    )
    eff = effective_launch(argv, B10425, plan, record.settings)
    assert eff.cache_reuse == 256 and eff.sources["cache_reuse"] == "argv"
    assert eff.cache_ram_mib == 32603 and eff.sources["cache_ram_mib"] == "argv"
    assert eff.cache_idle_slots is True
    assert eff.slot_prompt_similarity == pytest.approx(0.3)
    assert eff.cont_batching is True
    assert eff.kv_unified is False and eff.sources["kv_unified"] == "argv"  # --no-kv-unified
    assert eff.parallel == 3 and eff.ctx_per_slot == 8192 and eff.ctx_total == 24576
    assert eff.summary.startswith("prefix cache on (reuse 256, host 32603 MiB, routing 0.3)")
    assert "continuous batching on" in eff.summary
    assert "3 slots x 8192" in eff.summary
    assert "partitioned KV" in eff.summary


def test_effective_launch_takes_the_last_occurrence_so_extra_flags_win(
    config: Config, tmp_path: Path
) -> None:
    """llama.cpp reads the last occurrence of a repeated flag; extra_flags go
    last on purpose. The report must read the argv the same way, aliases
    included -- ``-nocb`` is what a human types."""
    record = make_record(
        tmp_path, settings=ModelSettings(extra_flags="--cache-reuse 64 -nocb -sps 0.9 -kvu")
    )
    plan = make_plan(parallel=2)
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, plan, port=18100, features=B10425
    )
    assert argv.count("--cache-reuse") == 2
    eff = effective_launch(argv, B10425, plan, record.settings)
    assert eff.cache_reuse == 64
    assert eff.cont_batching is False and eff.sources["cont_batching"] == "argv"
    assert eff.slot_prompt_similarity == pytest.approx(0.9)
    assert eff.kv_unified is True, "-kvu after --no-kv-unified wins"
    assert "continuous batching OFF" in eff.summary
    assert "unified KV" in eff.summary


def test_effective_cont_batching_is_true_by_default_and_false_only_with_the_no_flag() -> None:
    plan = make_plan(parallel=1)
    base = ["llama-server", "--ctx-size", "8192", "--parallel", "1"]
    assert effective_launch(base, B10425, plan).cont_batching is True
    assert effective_launch([*base, "--cont-batching"], B10425, plan).cont_batching is True
    assert effective_launch([*base, "--no-cont-batching"], B10425, plan).cont_batching is False
    assert effective_launch([*base, "-nocb"], B10425, plan).cont_batching is False
    # An unknown engine falls back to the common.h default, and says so.
    unknown = effective_launch(base, UNKNOWN, plan)
    assert unknown.cont_batching is True and unknown.sources["cont_batching"] == "engine_default"


def test_cont_batching_false_now_emits_the_no_flag_when_the_engine_has_it(
    config: Config, tmp_path: Path
) -> None:
    """Before D54 ``cont_batching: false`` emitted nothing at all -- the GUI's
    tri-state toggle had an "off" position that did nothing."""
    record = make_record(tmp_path, settings=ModelSettings(cont_batching=False))
    plan = make_plan(parallel=1)
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, plan, port=18100, features=B10425
    )
    assert "--no-cont-batching" in argv
    assert "--cont-batching" not in argv
    eff = effective_launch(argv, B10425, plan, record.settings)
    assert eff.cont_batching is False and eff.inert == []

    on = make_record(tmp_path, settings=ModelSettings(cont_batching=True))
    argv = sup(config, make_binary(tmp_path)).build_command(on, plan, port=18100, features=B10425)
    assert "--cont-batching" in argv and "--no-cont-batching" not in argv


def test_cont_batching_false_emits_nothing_on_an_engine_without_the_no_flag(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch the child cannot see must not look honoured (D17's
    --defrag-thold rule): no flag, a ``setting_inert`` warning naming it, and
    the report lists it as inert while showing the engine default."""
    warnings: list[tuple[str, dict[str, object]]] = []

    class _Recorder:
        def warning(self, event: str, **fields: object) -> None:
            warnings.append((event, fields))

        def __getattr__(self, _name: str) -> Callable[..., None]:
            return lambda *_a, **_k: None

    # The structlog sink is configured by whichever test ran first, so the
    # event is asserted on the logger call itself rather than on stdout.
    monkeypatch.setattr(supervisor_module, "log", _Recorder())
    older = engine_without_flags("--no-cont-batching", "-nocb")
    record = make_record(tmp_path, settings=ModelSettings(cont_batching=False))
    plan = make_plan(parallel=1)
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, plan, port=18100, features=older
    )
    assert "--no-cont-batching" not in argv and "--cont-batching" not in argv
    (event, fields) = next((w for w in warnings if w[0] == "setting_inert"), (None, {}))
    assert event == "setting_inert", warnings
    assert fields["setting"] == "cont_batching" and fields["flag"] == "--no-cont-batching"
    assert fields["model_id"] == record.id and fields["engine_known"] is True
    eff = effective_launch(argv, older, plan, record.settings)
    assert eff.cont_batching is True, "what the child actually does"
    assert eff.inert == ["cont_batching"]
    assert "inert: cont_batching" in eff.summary


def test_effective_cache_idle_slots_is_off_when_cache_ram_is_zero() -> None:
    """The engine disables idle-slot snapshots itself without a host cache to
    put them in (b10689 server-context.cpp:1420-1423)."""
    plan = make_plan(parallel=2)
    base = ["llama-server", "--ctx-size", "16384", "--parallel", "2"]
    assert effective_launch([*base, "--cache-ram", "8192"], B10425, plan).cache_idle_slots is True
    off = effective_launch([*base, "--cache-ram", "0"], B10425, plan)
    assert off.cache_idle_slots is False and off.cache_ram_mib == 0
    assert "host cache off" in off.summary
    # No --cache-ram passed: the engine's own default pool applies.
    silent = effective_launch(base, B10425, plan)
    assert silent.cache_ram_mib == 8192 and silent.sources["cache_ram_mib"] == "engine_default"
    # A known engine with no host cache at all reports None, not a number.
    no_cram = dataclasses.replace(B10425, cache_ram=False, cache_idle_slots=False)
    none = effective_launch(base, no_cram, plan)
    assert none.cache_ram_mib is None and none.cache_idle_slots is False
    assert "no host cache" in none.summary
    # Unknown engine: the common.h constant, marked as a default.
    assert effective_launch(base, UNKNOWN, plan).cache_ram_mib == ENGINE_DEFAULT_CACHE_RAM_MIB


def test_effective_ubatch_is_clamped_to_the_logical_batch_like_the_engine_does() -> None:
    plan = make_plan(parallel=1)
    eff = effective_launch(
        ["llama-server", "-c", "8192", "-np", "1", "-b", "1024", "-ub", "4096"], B10425, plan
    )
    assert eff.batch_size == 1024 and eff.ubatch_size == 1024


def test_effective_launch_resolves_spec_type_and_flash_attn_from_the_argv(
    config: Config, tmp_path: Path
) -> None:
    record = make_record(tmp_path, meta=dense_meta(extra={"nextn_predict_layers": 1}))
    plan = make_plan(parallel=1)
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, plan, port=18100, features=B10425
    )
    eff = effective_launch(argv, B10425, plan, record.settings)
    assert eff.spec_type == "draft-mtp" and eff.sources["spec_type"] == "argv"
    assert eff.flash_attn == "on"
    assert "spec draft-mtp" in eff.summary


def test_launch_args_redact_the_api_key_flags(config: Config, tmp_path: Path) -> None:
    """Nothing StudioForge emits carries a secret, but extra_flags is free text
    and llama-server accepts --api-key; a status surface must never echo it.
    Absolute paths go too -- a model row is not the place to map a disk."""
    record = make_record(
        tmp_path,
        settings=ModelSettings(
            extra_flags="--api-key sk-live-secret --api-key-file /etc/keys --ssl-key-file C:/k.pem"
        ),
    )
    argv = sup(config, make_binary(tmp_path)).build_command(
        record, make_plan(), port=18100, features=B10425
    )
    shown = redact_argv(argv)
    joined = " ".join(shown)
    assert "sk-live-secret" not in joined
    assert "/etc/keys" not in joined and "k.pem" not in joined
    assert shown.count("<redacted>") == 3
    assert shown[0] == "llama-server.exe", "the binary by basename only"
    assert shown[shown.index("--model") + 1] == "m.gguf"
    assert not any(Path(token).is_absolute() for token in shown)
    assert not any("\\" in token or (len(token) > 1 and token[1] == ":") for token in shown)
    # The flags themselves stay, so the row still says an api key is in force.
    assert "--api-key" in shown and "--ssl-key-file" in shown
    # The one thing every StudioForge argv carries is exactly as built.
    assert shown[shown.index("--ctx-size") + 1] == "8192"


def test_redact_argv_handles_the_equals_spelling_and_relative_tokens() -> None:
    argv = ["llama-server", "--api-key=abc", "--lora", "./adapters/a.gguf", "-fa", "on"]
    shown = redact_argv(argv)
    assert shown == [
        "llama-server",
        "--api-key=<redacted>",
        "--lora",
        "./adapters/a.gguf",
        "-fa",
        "on",
    ]


def test_redact_argv_hides_the_hugging_face_token_and_a_spaced_windows_path() -> None:
    """``--hf-token`` / ``-hft`` is a credential llama-server takes on the
    command line (b10689 common/arg.cpp); ``extra_flags`` could carry it. And
    a Windows path with spaces and a user name in it must leave only the file
    name behind, since the argv is served on ``/api/status``."""
    argv = [
        r"C:\Users\Some Person\AppData\Local\SF\engines\b10689\llama-server.exe",  # scrub-ok: fake
        "--model",
        r"D:\LLM Models\Some Person\Dark-Scarlett-27B.gguf",
        "--hf-token",
        "hf_secret_value",
        "-hft",
        "hf_other_secret",
        "--hf-token=hf_third",
        "--chat-template-file",
        "/home/someone/templates/chatml.jinja",  # scrub-ok: invented fixture, redacted below
    ]
    shown = redact_argv(argv)
    assert shown == [
        "llama-server.exe",
        "--model",
        "Dark-Scarlett-27B.gguf",
        "--hf-token",
        "<redacted>",
        "-hft",
        "<redacted>",
        "--hf-token=<redacted>",
        "--chat-template-file",
        "chatml.jinja",
    ]
    joined = " ".join(shown)
    assert "Some Person" not in joined and "someone" not in joined and "hf_" not in joined
