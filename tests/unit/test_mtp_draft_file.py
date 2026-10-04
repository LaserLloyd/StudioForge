"""An MTP-only draft head attached by file (D74).

unsloth and others publish a model's multi-token-prediction block as its own
GGUF (``MTP/mtp-<model>-Q8_0.gguf``: the MTP block, ``token_embd`` and
``output``, no trunk) for quants made without their heads. llama.cpp loads one
as ``--spec-draft-model`` under ``--spec-type draft-mtp`` and gives it a KV
cache for the MTP block alone. StudioForge had no way to attach one: drafts
were registry models (the scanner skips these files on purpose), ``draft-mtp``
assumed in-model heads, and the planner priced the head's KV as every one of
its header's ``block_count`` layers (~20 GB at 200k for ~0.4 GB of cache).
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core.gguf import (
    STRING,
    UINT32,
    is_mtp_draft,
    is_mtp_only_tensors,
    meta_from_gguf,
    read_gguf,
    read_meta,
    validate_mtp_draft_file,
)
from studioforge.core.manager import ModelManager
from studioforge.core.planner import MB, Planner, estimate_kv_bytes
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor, resolve_spec_type
from studioforge.db import Database
from studioforge.errors import BadRequestError, ModelLoadError
from studioforge.types import GgufMeta, ModelRecord, ModelSettings
from tests.unit.test_gguf import KvEntry, TensorSpec, write_gguf
from tests.unit.test_supervisor_features import (
    B10425,
    UNKNOWN,
    engine_without,
    make_binary,
    make_plan,
    make_record,
    resolver,
    value_after,
)

F16 = 1
ARCH = "qwen4exp"


def _kv(
    *, blocks: int = 3, nextn: int | None = 1, arch: str = ARCH, n_embd: int = 64
) -> list[KvEntry]:
    kv: list[KvEntry] = [
        ("general.architecture", STRING, arch),
        (f"{arch}.block_count", UINT32, blocks),
        (f"{arch}.embedding_length", UINT32, n_embd),
        (f"{arch}.attention.head_count", UINT32, 4),
        (f"{arch}.attention.head_count_kv", UINT32, 2),
        (f"{arch}.context_length", UINT32, 4096),
    ]
    if nextn is not None:
        kv.append((f"{arch}.nextn_predict_layers", UINT32, nextn))
    return kv


#: A 2-block trunk + 1 MTP block model's head file: block 2 only.
_HEAD: list[TensorSpec] = [
    ("token_embd.weight", (64, 256), F16),
    ("output.weight", (64, 256), F16),
    ("blk.2.nextn.eh_proj.weight", (128, 64), F16),
    ("blk.2.attn_q.weight", (64, 64), F16),
]
_TRUNK: list[TensorSpec] = [("blk.0.attn_q.weight", (64, 64), F16)]


def _head(tmp_path: Path, name: str = "mtp-model-Q8_0.gguf", **kv: Any) -> Path:
    (tmp_path / "MTP").mkdir(parents=True, exist_ok=True)
    return write_gguf(tmp_path / "MTP" / name, _kv(**kv), _HEAD)


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# The parser recognises a head file
# ---------------------------------------------------------------------------


def test_a_head_file_is_flagged_mtp_only(tmp_path: Path) -> None:
    meta = read_meta(_head(tmp_path))
    assert meta.extra["nextn_predict_layers"] == 1
    assert meta.extra["mtp_only"] is True
    assert is_mtp_draft(meta)


def test_a_file_with_a_trunk_is_not(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "full.gguf", _kv(), _TRUNK + _HEAD)
    assert "mtp_only" not in read_meta(path).extra


def test_a_header_only_read_never_claims_it(tmp_path: Path) -> None:
    path = _head(tmp_path)
    meta = meta_from_gguf(read_gguf(path, load_tensors=False), path=path, local=False)
    assert "mtp_only" not in meta.extra


def test_the_tensor_rule() -> None:
    def t(name: str) -> Any:
        return SimpleNamespace(name=name)

    head = [t("token_embd.weight"), t("blk.48.nextn.eh_proj.weight")]
    assert is_mtp_only_tensors(head, n_layer=49, nextn=1)
    assert not is_mtp_only_tensors([*head, t("blk.47.attn_q.weight")], n_layer=49, nextn=1)
    assert not is_mtp_only_tensors([t("token_embd.weight")], n_layer=49, nextn=1)  # no block
    assert not is_mtp_only_tensors(head, n_layer=49, nextn=0)


# ---------------------------------------------------------------------------
# Save-time validation
# ---------------------------------------------------------------------------


def _base(**over: Any) -> GgufMeta:
    values: dict[str, Any] = {"architecture": ARCH, "n_layer": 3, "n_embd": 64}
    values.update(over)
    return GgufMeta(**values)


def test_a_matching_head_validates(tmp_path: Path) -> None:
    path = _head(tmp_path)
    assert validate_mtp_draft_file(path, _base()) == path
    assert validate_mtp_draft_file(None, _base()) is None


@pytest.mark.parametrize(
    ("make", "base", "needle"),
    [
        (lambda p: p / "missing.gguf", _base(), "does not exist"),
        (lambda p: write_gguf(p / "x.gguf", _kv(nextn=None), _HEAD), _base(), "no multi-token"),
        (lambda p: write_gguf(p / "x.gguf", _kv(), _TRUNK + _HEAD), _base(), "trunk"),
        (lambda p: _head(p), _base(architecture="qwen35moe"), "'qwen4exp'"),
        (lambda p: _head(p), _base(n_embd=128), "embedding width"),
        (lambda p: _head(p), None, "no parsed GGUF metadata"),
    ],
)
def test_mismatches_are_named(tmp_path: Path, make: Any, base: Any, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        validate_mtp_draft_file(make(tmp_path), base)


def test_not_a_gguf_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    with pytest.raises(ValueError, match="not a readable GGUF"):
        validate_mtp_draft_file(junk, _base())


MODEL_REL = "publisher/repo-GGUF/model-Q4_K_S.gguf"
MODEL_ID = "publisher/repo-GGUF/model-Q4_K_S"


@pytest.fixture()
def registry(tmp_path: Path) -> Registry:
    library = tmp_path / "models"
    path = library / MODEL_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 4096)
    db = Database(tmp_path / "data" / "registry.sqlite3")
    db.migrate()
    config = Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=library))

    def reader(_path: Path, shard_paths: Any = None) -> GgufMeta:
        return _base(tensor_bytes=4096, quant_label="Q4_K_S")

    reg = Registry(config, db, meta_reader=reader)
    reg.scan()
    return reg


def test_saving_a_head_round_trips(registry: Registry, tmp_path: Path) -> None:
    head = _head(tmp_path)
    registry.save_settings(MODEL_ID, ModelSettings(mtp_draft_file=head))
    assert registry.get_settings(MODEL_ID).mtp_draft_file == head


def test_saving_a_bad_head_is_a_400_naming_the_field(registry: Registry, tmp_path: Path) -> None:
    wrong = write_gguf(tmp_path / "x.gguf", _kv(), _TRUNK + _HEAD)
    with pytest.raises(BadRequestError) as exc:
        registry.save_settings(MODEL_ID, ModelSettings(mtp_draft_file=wrong))
    assert exc.value.param == "mtp_draft_file"
    assert registry.get_settings(MODEL_ID).mtp_draft_file is None


def test_a_head_and_a_draft_model_are_exclusive(registry: Registry, tmp_path: Path) -> None:
    with pytest.raises(BadRequestError, match="exclusive"):
        registry.save_settings(
            MODEL_ID, ModelSettings(mtp_draft_file=_head(tmp_path), draft_model_id="other")
        )


def test_a_head_deleted_after_saving_does_not_brick_other_edits(
    registry: Registry, tmp_path: Path
) -> None:
    head = _head(tmp_path)
    registry.save_settings(MODEL_ID, ModelSettings(mtp_draft_file=head))
    head.unlink()
    registry.save_settings(MODEL_ID, ModelSettings(mtp_draft_file=head, ctx_size=8192))
    assert registry.get_settings(MODEL_ID).ctx_size == 8192


# ---------------------------------------------------------------------------
# spec_type resolution and the argv
# ---------------------------------------------------------------------------


def _moe_record(tmp_path: Path, **settings: Any) -> ModelRecord:
    meta = GgufMeta(
        architecture=ARCH, n_layer=3, n_embd=64, n_head=4, n_head_kv=2, n_expert=8, n_expert_used=2
    )
    return make_record(tmp_path, meta=meta, settings=ModelSettings(**settings))


def test_auto_picks_draft_mtp_for_an_attached_head(tmp_path: Path) -> None:
    spec, reason = resolve_spec_type(_moe_record(tmp_path), B10425, has_draft=True, mtp_draft=True)
    assert spec == "draft-mtp"
    assert "mtp_draft_file" in reason


def test_an_engine_without_draft_mtp_never_runs_the_head_as_a_plain_draft(tmp_path: Path) -> None:
    spec, _ = resolve_spec_type(
        _moe_record(tmp_path), engine_without("draft-mtp"), has_draft=True, mtp_draft=True
    )
    assert spec != "draft-simple"
    assert spec == "ngram-mod"  # the MoE fallback, drafting from the text
    unknown, _ = resolve_spec_type(_moe_record(tmp_path), UNKNOWN, has_draft=True, mtp_draft=True)
    assert unknown == "none"


def test_an_explicit_other_draft_type_is_refused(tmp_path: Path) -> None:
    record = _moe_record(tmp_path, spec_type="draft-simple")
    with pytest.raises(ModelLoadError, match="only runs under draft-mtp"):
        resolve_spec_type(record, B10425, has_draft=True, mtp_draft=True)
    ok = _moe_record(tmp_path, spec_type="draft-mtp")
    assert resolve_spec_type(ok, B10425, has_draft=True, mtp_draft=True)[0] == "draft-mtp"


def _draft_record(path: Path) -> ModelRecord:
    meta = read_meta(path)
    return ModelRecord(id=f"mtp-draft:{path.name}", name=path.stem, path=path, meta=meta)


def test_the_head_is_the_spec_draft_model_of_a_draft_mtp_launch(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    head = _head(tmp_path)
    argv = Supervisor(config, resolve_binary=resolver(make_binary(tmp_path))).build_command(
        _moe_record(tmp_path),
        make_plan(devices=[0, 1]),
        port=18100,
        features=B10425,
        draft=_draft_record(head),
    )
    assert value_after(argv, "--spec-type") == "draft-mtp"
    assert value_after(argv, "--spec-draft-model") == str(head)
    assert value_after(argv, "--spec-draft-device") == "CUDA0,CUDA1"


def test_no_head_flags_when_the_engine_cannot_run_it(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    argv = Supervisor(config, resolve_binary=resolver(make_binary(tmp_path))).build_command(
        _moe_record(tmp_path),
        make_plan(devices=[0]),
        port=18100,
        features=engine_without("draft-mtp"),
        draft=_draft_record(_head(tmp_path)),
    )
    assert "--spec-draft-model" not in argv


# ---------------------------------------------------------------------------
# The planner prices the head's own cache, not its header's block count
# ---------------------------------------------------------------------------


class _NoGpus:
    def list_gpus(self) -> list[Any]:
        return []


def test_the_head_kv_is_its_mtp_block_alone(tmp_path: Path) -> None:
    """The real head says block_count 49; llama.cpp caches only block 48."""
    head_meta = GgufMeta(
        architecture=ARCH,
        n_layer=49,
        n_embd=2560,
        n_head=24,
        n_head_kv=2,
        n_embd_head_k=256,
        n_embd_head_v=256,
        tensor_bytes=int(3.84 * 1024) * MB,
        extra={"nextn_predict_layers": 1, "mtp_only": True},
    )
    head = ModelRecord(id="mtp-draft:h", name="h", path=tmp_path / "h.gguf", meta=head_meta)
    base = GgufMeta(
        architecture=ARCH,
        n_layer=49,
        n_embd=2560,
        n_head=24,
        n_head_kv=2,
        n_embd_head_k=256,
        n_embd_head_v=256,
        tensor_bytes=70 * 1024 * MB,
    )
    record = ModelRecord(id="m", name="m", path=tmp_path / "m.gguf", meta=base)
    planner = Planner(Config(), _NoGpus(), log_plans=False)  # type: ignore[arg-type]
    est = planner.estimate(
        record,
        ctx_size=200_000,
        parallel=1,
        kv_cache_type="f16",
        kv_cache_type_v="f16",
        n_devices=4,
        draft=head,
        ubatch=512,
    )
    one_layer = estimate_kv_bytes(
        n_layer=1,
        n_head_kv=2,
        head_dim_k=256,
        head_dim_v=256,
        ctx_total=200_000,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    assert est.draft_kv_bytes == one_layer
    assert est.draft_kv_bytes < 500 * MB  # was ~20 GB: 49 layers
    assert est.draft_weights_bytes == int(3.84 * 1024) * MB


# ---------------------------------------------------------------------------
# The manager builds the draft record from the file
# ---------------------------------------------------------------------------


def _manager_stub() -> Any:
    ns = SimpleNamespace(_mtp_draft_cache={}, registry=None)
    ns._mtp_draft_record = functools.partial(ModelManager._mtp_draft_record, ns)
    return ns


def test_draft_for_builds_an_unlisted_record_from_the_file(tmp_path: Path) -> None:
    head = _head(tmp_path)
    stub = _manager_stub()
    record = _moe_record(tmp_path, mtp_draft_file=head)
    draft = ModelManager._draft_for(stub, record)
    assert draft is not None
    assert draft.id == f"mtp-draft:{head.name}"
    assert draft.path == head
    assert is_mtp_draft(draft.meta)
    assert ModelManager._draft_for(stub, record) is draft  # memoised


def test_a_head_that_went_missing_loads_without_it(tmp_path: Path) -> None:
    record = _moe_record(tmp_path, mtp_draft_file=tmp_path / "gone.gguf")
    assert ModelManager._draft_for(_manager_stub(), record) is None


def test_a_file_that_stopped_being_a_head_loads_without_it(tmp_path: Path) -> None:
    full = write_gguf(tmp_path / "full.gguf", _kv(), _TRUNK + _HEAD)
    record = _moe_record(tmp_path, mtp_draft_file=full)
    assert ModelManager._draft_for(_manager_stub(), record) is None
