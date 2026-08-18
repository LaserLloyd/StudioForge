"""Unit tests for studioforge.core.registry.

The synthetic tree built by :func:`library` mirrors every awkward case found in
a real LM Studio-style model directory: nested ``publisher/repo/file.gguf``,
loose root-level files (one with spaces in the name), all four mmproj naming
conventions plus an unpairable one, a complete and an incomplete multi-part
model, embedding/rerank/vision/tools models, a LoRA adapter, a file that fails
to parse, an alias collision, and non-model clutter.

GGUF *bytes* are never needed here because the metadata reader is injected --
that is the registry's only injection point, and it keeps this suite
independent of the real parser. A separate live test at the bottom exercises
the real parser against whatever GGUF library this machine actually has
(``SF_TEST_MODELS_DIR``, else the auto-detected LM Studio one); it skips when
there is none.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core import gguf
from studioforge.core import registry as registry_module
from studioforge.core.registry import Registry, ScanResult, _cache_key, _mmproj_core
from studioforge.db import Database
from studioforge.errors import BadRequestError, ModelNotFoundError
from studioforge.types import AdapterAttachment, GgufMeta, ModelSettings

GIB = 1024**3

# ---------------------------------------------------------------------------
# Synthetic library
# ---------------------------------------------------------------------------

#: Relative paths of every file created in the fake library. ``.gguf`` entries
#: become models/mmprojs/adapters; everything else is clutter that must be
#: ignored.
TREE: tuple[str, ...] = (
    # --- plain nested model (the tiny one clients address by short name) ---
    "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    # --- mmproj style (a): prefix, name otherwise identical -----------------
    "exact/TinyVision-GGUF/TinyVision.gguf",
    "exact/TinyVision-GGUF/mmproj-TinyVision.gguf",
    # --- mmproj style (b1): prefix token, quant tail differs ----------------
    "lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0.gguf",
    "lmstudio-community/gemma-4-31B-it-QAT-GGUF/mmproj-gemma-4-31B-it-QAT-BF16.gguf",
    # --- mmproj style (b2): suffix token, dot-separated ---------------------
    "mradermacher/Qwen2.5-VL-7B-Abliterated-Caption-it-GGUF"
    "/Qwen2.5-VL-7B-Abliterated-Caption-it.Q4_K_S.gguf",
    "mradermacher/Qwen2.5-VL-7B-Abliterated-Caption-it-GGUF"
    "/Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-f16.gguf",
    # --- mmproj style (b3): infix token -------------------------------------
    "llmfan46/G4-MeroMero-31B-uncensored-heretic-GGUF/G4-MeroMero-31B-uncensored-heretic-Q6_K.gguf",
    "llmfan46/G4-MeroMero-31B-uncensored-heretic-GGUF"
    "/G4-MeroMero-31B-uncensored-heretic-mmproj-BF16.gguf",
    # --- mmproj style (c): generic name, sole candidate in the directory ----
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
    "unsloth/gemma-4-26B-A4B-it-qat-GGUF/mmproj-F32.gguf",
    # --- mmproj style (d): generic name, two bases -> ambiguous, no pairing -
    "ambig/Two-Bases-GGUF/alpha-thing-Q4_K_M.gguf",
    "ambig/Two-Bases-GGUF/beta-thing-Q4_K_M.gguf",
    "ambig/Two-Bases-GGUF/mmproj-F32.gguf",
    # --- vision without an mmproj (single-file multimodal) ------------------
    "single/VL-Single-GGUF/VL-Single-Q4_K_M.gguf",
    # --- multi-part, complete ----------------------------------------------
    "bartowski/TheDrummer_Behemoth-X-123B-v2.1-GGUF"
    "/TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf",
    "bartowski/TheDrummer_Behemoth-X-123B-v2.1-GGUF"
    "/TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00002-of-00002.gguf",
    # --- multi-part, middle shard missing (00002 of 00003 absent) ----------
    "broken/Broken-Split-GGUF/Broken-Split-Q4_K_M-00001-of-00003.gguf",
    "broken/Broken-Split-GGUF/Broken-Split-Q4_K_M-00003-of-00003.gguf",
    # --- a file the parser rejects -----------------------------------------
    "broken/Corrupt-GGUF/corrupt-me-Q4_K_M.gguf",
    # --- embedding + rerank -------------------------------------------------
    "endyjasmi/Qwen3-Embedding-8B-Q4_K_M-GGUF/qwen3-embedding-8b-q4_k_m.gguf",
    "someone/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B-Q8_0.gguf",
    # --- tools-capable chat template ---------------------------------------
    "toolco/Toolformer-8B-GGUF/Toolformer-8B-Q5_K_M.gguf",
    # --- loose root files, one of them with spaces --------------------------
    "24_10_Mistrial_Celeste-12B-V1.6.Q8_0NSFW.gguf",
    "24_10_Pygmalion or Mistral_cydonia-22b-v1-q6_k.gguf",
    # --- LoRA adapter under LORAs/ -----------------------------------------
    "LORAs/Zeta-17b-GGUF/Zeta-17b-lora-rank32-F16.gguf",
    # --- two models whose bare stems collide -------------------------------
    "collide-a/repo-one-GGUF/samename-Q4_K_M.gguf",
    "collide-b/repo-two-GGUF/samename-Q4_K_M.gguf",
    # --- clutter that must be ignored --------------------------------------
    "README.md",
    "config.yaml",
    "place-your-models-here.txt",
    "_Models run.xlsx",
    "_how to exl2.docx",
    "_Mistrial_Celeste.url",
    "Ko_MOE8x8.kcpps",
    "Templates/ollama_modelfile_template/Modelfile.txt",
    "_Documents-non-models/notes.txt",
    ".cache/huggingface/should-not-be-seen.gguf",
)

QWEN_TINY = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0"
GEMMA_31B = "lmstudio-community/gemma-4-31B-it-QAT-GGUF/gemma-4-31B-it-QAT-Q4_0"
QWEN_VL = (
    "mradermacher/Qwen2.5-VL-7B-Abliterated-Caption-it-GGUF"
    "/Qwen2.5-VL-7B-Abliterated-Caption-it.Q4_K_S"
)
MEROMERO = (
    "llmfan46/G4-MeroMero-31B-uncensored-heretic-GGUF/G4-MeroMero-31B-uncensored-heretic-Q6_K"
)
UNSLOTH = "unsloth/gemma-4-26B-A4B-it-qat-GGUF/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL"
TINY_VISION = "exact/TinyVision-GGUF/TinyVision"
BEHEMOTH = "bartowski/TheDrummer_Behemoth-X-123B-v2.1-GGUF/TheDrummer_Behemoth-X-123B-v2.1-IQ3_M"
BROKEN_SPLIT = "broken/Broken-Split-GGUF/Broken-Split-Q4_K_M"
CORRUPT = "broken/Corrupt-GGUF/corrupt-me-Q4_K_M"
EMBEDDING = "endyjasmi/Qwen3-Embedding-8B-Q4_K_M-GGUF/qwen3-embedding-8b-q4_k_m"
RERANK = "someone/Qwen3-Reranker-4B-GGUF/Qwen3-Reranker-4B-Q8_0"
TOOLS = "toolco/Toolformer-8B-GGUF/Toolformer-8B-Q5_K_M"
LOOSE = "24_10_Mistrial_Celeste-12B-V1.6.Q8_0NSFW"
LOOSE_SPACES = "24_10_Pygmalion or Mistral_cydonia-22b-v1-q6_k"
LORA = "LORAs/Zeta-17b-GGUF/Zeta-17b-lora-rank32-F16"

#: Per-file synthetic sizes, so size summing is verifiable. Deliberately tiny
#: and distinct -- real shard sizes would mean writing tens of GiB, and the
#: arithmetic being tested is identical either way.
SHARD_ONE_SIZE = 3000
SHARD_TWO_SIZE = 2000
SIZES: dict[str, int] = {
    "TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf": SHARD_ONE_SIZE,
    "TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00002-of-00002.gguf": SHARD_TWO_SIZE,
}
DEFAULT_SIZE = 4096

TOOL_TEMPLATE = (
    "{% for m in messages %}{{ m.content }}{% endfor %}{% if tools %}{{ tools }}{% endif %}"  # noqa: E501
)
PLAIN_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


class FakeMetaReader:
    """Filename-driven stand-in for :func:`gguf.read_meta`.

    Records every call so cache hits/misses are directly observable, and
    raises :class:`gguf.GgufError` for the deliberately corrupt fixture.
    """

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.shard_args: dict[str, int] = {}

    @property
    def count(self) -> int:
        return len(self.calls)

    def reset(self) -> None:
        self.calls.clear()
        self.shard_args.clear()

    def __call__(self, path: Path, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
        self.calls.append(path)
        self.shard_args[path.name] = len(shard_paths) if shard_paths else 0
        name = path.name.lower()
        if "corrupt" in name:
            raise gguf.GgufError(f"not a GGUF file: {path.name}")
        if "lora" in name:
            return GgufMeta(
                architecture="llama",
                n_layer=32,
                is_adapter=True,
                quant_label="F16",
                extra={"adapter_rank": 32, "base_model": "meta-llama/Llama-3-17B"},
            )
        if "mmproj" in name:
            return GgufMeta(
                architecture="clip",
                is_mmproj=True,
                has_vision_tensors=True,
                quant_label="BF16",
                vision_image_size=896,
                vision_patch_size=14,
            )
        meta = GgufMeta(
            architecture="qwen3" if "qwen" in name else "gemma3",
            n_layer=32,
            n_embd=4096,
            n_head=32,
            n_head_kv=8,
            n_ctx_train=131072,
            n_vocab=151936,
            quant_label=_quant_from_name(path.name),
            tensor_bytes=DEFAULT_SIZE,
            chat_template=TOOL_TEMPLATE if "toolformer" in name else PLAIN_TEMPLATE,
            has_vision_tensors="vl-single" in name,
        )
        if "embedding" in name:
            meta = meta.model_copy(update={"extra": {"embedding": True}})
        return meta


def _quant_from_name(name: str) -> str:
    stem = name[: -len(".gguf")]
    for token in reversed(stem.replace(".", "-").split("-")):
        if token and (token[0] in "qQiI" or token.upper() in {"F16", "BF16", "F32"}):
            return token.upper()
    return "unknown"


def _size_for(name: str) -> int:
    return SIZES.get(name, DEFAULT_SIZE)


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's ``SF_*`` environment out of these tests."""
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for rel in TREE:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * _size_for(path.name))
    return root


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


def make_config(library: Path, tmp_path: Path, *, extra_dirs: list[Path] | None = None) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        models=ModelsConfig(dir=library, extra_dirs=extra_dirs or []),
    )


@pytest.fixture()
def reader() -> FakeMetaReader:
    return FakeMetaReader()


@pytest.fixture()
def reg(library: Path, tmp_path: Path, db: Database, reader: FakeMetaReader) -> Registry:
    return Registry(make_config(library, tmp_path), db, meta_reader=reader)


@pytest.fixture()
def scanned(reg: Registry) -> Registry:
    reg.scan()
    return reg


class LogRecorder:
    """Minimal structlog-shaped recorder so log assertions need no config."""

    def __init__(self) -> None:
        self.entries: list[tuple[str, str, dict[str, Any]]] = []

    def _record(self, level: str, event: str, **kw: Any) -> None:
        self.entries.append((level, event, kw))

    def debug(self, event: str, **kw: Any) -> None:
        self._record("debug", event, **kw)

    def info(self, event: str, **kw: Any) -> None:
        self._record("info", event, **kw)

    def warning(self, event: str, **kw: Any) -> None:
        self._record("warning", event, **kw)

    def error(self, event: str, **kw: Any) -> None:
        self._record("error", event, **kw)

    def of(self, event: str, level: str | None = None) -> list[dict[str, Any]]:
        return [
            kw for lvl, ev, kw in self.entries if ev == event and (level is None or lvl == level)
        ]


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> LogRecorder:
    rec = LogRecorder()
    monkeypatch.setattr(registry_module, "log", rec)
    return rec


# ---------------------------------------------------------------------------
# ID derivation
# ---------------------------------------------------------------------------


def test_scan_returns_scan_result(scanned: Registry) -> None:
    result = scanned.scan()
    assert isinstance(result, ScanResult)
    assert result.duration_s >= 0.0
    assert result.added == []  # second scan adds nothing
    assert result.unchanged == len(scanned.all())


def test_ids_are_relative_posix_paths(scanned: Registry) -> None:
    ids = set(scanned.known_ids())
    assert QWEN_TINY in ids
    assert GEMMA_31B in ids
    assert all("\\" not in model_id for model_id in ids)
    assert all(not model_id.lower().endswith(".gguf") for model_id in ids)


def test_loose_root_file_id_is_bare_stem(scanned: Registry) -> None:
    assert LOOSE in scanned.known_ids()
    record = scanned.get(LOOSE)
    assert record is not None
    assert record.publisher is None
    assert record.repo is None


def test_filename_with_spaces_is_indexed(scanned: Registry) -> None:
    record = scanned.get(LOOSE_SPACES)
    assert record is not None
    assert " " in record.path.name
    assert scanned.resolve("24_10_pygmalion or mistral_cydonia-22b-v1-q6_k") is record


def test_multipart_id_drops_shard_suffix(scanned: Registry) -> None:
    assert BEHEMOTH in scanned.known_ids()
    assert not any("00001-of" in model_id for model_id in scanned.known_ids())


def test_ids_stable_across_mtime_change(scanned: Registry, library: Path) -> None:
    before = scanned.known_ids()
    target = library / TREE[0]
    os.utime(target, (time.time() + 60, time.time() + 60))
    scanned.scan()
    assert scanned.known_ids() == before


def test_ids_stable_across_fresh_registry(
    library: Path, tmp_path: Path, db: Database, reader: FakeMetaReader
) -> None:
    first = Registry(make_config(library, tmp_path), db, meta_reader=reader)
    first.scan()
    second = Registry(make_config(library, tmp_path), db, meta_reader=FakeMetaReader())
    second.scan()
    assert first.known_ids() == second.known_ids()


# ---------------------------------------------------------------------------
# mmproj pairing
# ---------------------------------------------------------------------------


def test_mmproj_core_strips_token_in_every_position() -> None:
    assert _mmproj_core("mmproj-TinyVision") == "TinyVision"
    assert _mmproj_core("mmproj-gemma-4-31B-it-QAT-BF16") == "gemma-4-31B-it-QAT-BF16"
    assert _mmproj_core("Qwen2.5-VL-7B.mmproj-f16") == "Qwen2.5-VL-7B.f16"
    assert _mmproj_core("G4-MeroMero-31B-mmproj-BF16") == "G4-MeroMero-31B-BF16"
    assert _mmproj_core("mmproj-F32") == "F32"


def test_pairing_rule_a_exact_name(scanned: Registry, recorder: LogRecorder) -> None:
    record = scanned.get(TINY_VISION)
    assert record is not None
    assert record.mmproj_path is not None
    assert record.mmproj_path.name == "mmproj-TinyVision.gguf"
    assert record.capabilities.vision is True


@pytest.mark.parametrize(
    ("model_id", "mmproj_name"),
    [
        (GEMMA_31B, "mmproj-gemma-4-31B-it-QAT-BF16.gguf"),
        (QWEN_VL, "Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-f16.gguf"),
        (MEROMERO, "G4-MeroMero-31B-uncensored-heretic-mmproj-BF16.gguf"),
    ],
)
def test_pairing_rule_b_prefix_styles(scanned: Registry, model_id: str, mmproj_name: str) -> None:
    record = scanned.get(model_id)
    assert record is not None, model_id
    assert record.mmproj_path is not None, model_id
    assert record.mmproj_path.name == mmproj_name
    assert record.capabilities.vision is True


def test_pairing_rule_c_sole_generic_mmproj(scanned: Registry) -> None:
    """``mmproj-F32.gguf`` shares nothing with the base name, so only the
    directory's arity can pair them."""
    record = scanned.get(UNSLOTH)
    assert record is not None
    assert record.mmproj_path is not None
    assert record.mmproj_path.name == "mmproj-F32.gguf"
    assert record.capabilities.vision is True


def test_pairing_rule_d_ambiguous_directory_left_unpaired(scanned: Registry) -> None:
    alpha = scanned.get("ambig/Two-Bases-GGUF/alpha-thing-Q4_K_M")
    beta = scanned.get("ambig/Two-Bases-GGUF/beta-thing-Q4_K_M")
    assert alpha is not None and beta is not None
    assert alpha.mmproj_path is None
    assert beta.mmproj_path is None
    assert alpha.capabilities.vision is False


def test_unpaired_mmproj_logged_at_info(reg: Registry, recorder: LogRecorder) -> None:
    reg.scan()
    unpaired = {kw["model_id"] for kw in recorder.of("registry.mmproj_unpaired", "info")}
    assert "ambig/Two-Bases-GGUF/alpha-thing-Q4_K_M" in unpaired
    assert "ambig/Two-Bases-GGUF/beta-thing-Q4_K_M" in unpaired


def test_pairing_rules_are_reported(reg: Registry, recorder: LogRecorder) -> None:
    reg.scan()
    by_model = {kw["model_id"]: kw["rule"] for kw in recorder.of("registry.mmproj_paired", "info")}
    assert by_model[TINY_VISION] == "exact"
    assert by_model[GEMMA_31B] == "prefix"
    assert by_model[UNSLOTH] == "sole-candidate"


def test_mmproj_is_never_a_model(scanned: Registry) -> None:
    for record in scanned.all():
        assert "mmproj" not in record.path.name.lower()
    assert not any("mmproj" in model_id.lower() for model_id in scanned.known_ids())


def test_vision_without_mmproj_from_metadata(scanned: Registry) -> None:
    record = scanned.get("single/VL-Single-GGUF/VL-Single-Q4_K_M")
    assert record is not None
    assert record.mmproj_path is None
    assert record.capabilities.vision is True


# ---------------------------------------------------------------------------
# Multi-part
# ---------------------------------------------------------------------------


def test_multipart_is_one_record_with_summed_size(scanned: Registry) -> None:
    record = scanned.get(BEHEMOTH)
    assert record is not None
    assert len(record.shards) == 2
    assert record.path.name.endswith("-00001-of-00002.gguf")
    assert record.shards == sorted(record.shards)
    assert record.size_bytes == SHARD_ONE_SIZE + SHARD_TWO_SIZE
    assert record.capabilities.multi_part is True


def test_multipart_meta_reader_gets_all_shards(reg: Registry, reader: FakeMetaReader) -> None:
    reg.scan()
    first = "TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf"
    assert reader.shard_args[first] == 2
    assert reader.shard_args["Qwen2.5-0.5B-Instruct-Q8_0.gguf"] == 0


def test_missing_shard_is_an_error_not_a_crash(reg: Registry) -> None:
    result = reg.scan()
    errors = dict(result.errors)
    assert BROKEN_SPLIT in errors
    assert "00002-of-00003" in errors[BROKEN_SPLIT]
    assert reg.get(BROKEN_SPLIT) is None


def test_second_shard_is_not_its_own_record(scanned: Registry) -> None:
    assert not any(model_id.endswith("-00002-of-00002") for model_id in scanned.known_ids())
    assert not any(model_id.endswith("-00003-of-00003") for model_id in scanned.known_ids())


# ---------------------------------------------------------------------------
# Kind + capabilities
# ---------------------------------------------------------------------------


def test_embedding_kind_and_capability(scanned: Registry) -> None:
    record = scanned.get(EMBEDDING)
    assert record is not None
    assert record.kind == "embedding"
    assert record.capabilities.embedding is True


def test_rerank_kind(scanned: Registry) -> None:
    record = scanned.get(RERANK)
    assert record is not None
    assert record.kind == "rerank"
    assert record.capabilities.embedding is False


def test_chat_is_the_default_kind(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert record is not None
    assert record.kind == "chat"


def test_tools_capability_from_chat_template(scanned: Registry) -> None:
    with_tools = scanned.get(TOOLS)
    without = scanned.get(QWEN_TINY)
    assert with_tools is not None and without is not None
    assert with_tools.capabilities.tools is True
    assert without.capabilities.tools is False


def test_quant_and_architecture_come_from_metadata(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert record is not None
    assert record.quant == "Q8_0"
    assert record.architecture == "qwen3"
    assert record.meta is not None
    assert record.meta.n_ctx_train == 131072


# ---------------------------------------------------------------------------
# Clutter + robustness
# ---------------------------------------------------------------------------


def test_non_gguf_clutter_is_ignored(scanned: Registry) -> None:
    haystack = " ".join(scanned.known_ids()).lower()
    for token in ("readme", "config.yaml", ".txt", ".xlsx", ".docx", ".url", ".kcpps", "template"):
        assert token not in haystack


def test_dot_directories_are_skipped(scanned: Registry) -> None:
    assert not any(".cache" in model_id for model_id in scanned.known_ids())


def test_parse_error_is_collected_not_raised(reg: Registry) -> None:
    result = reg.scan()
    errors = dict(result.errors)
    assert CORRUPT in errors
    assert "GgufError" in errors[CORRUPT]
    # ...and the other ~20 models still scanned fine.
    assert len(reg.all()) > 15


def test_missing_model_dir_does_not_raise(tmp_path: Path, db: Database) -> None:
    config = Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=tmp_path / "nope"))
    reg = Registry(config, db, meta_reader=FakeMetaReader())
    result = reg.scan()
    assert reg.all() == []
    assert result.added == []


def test_scan_is_reentrant_and_removes_deleted_models(reg: Registry, library: Path) -> None:
    reg.scan()
    assert reg.get(QWEN_TINY) is not None
    (library / TREE[0]).unlink()
    result = reg.scan()
    assert QWEN_TINY in result.removed
    assert reg.get(QWEN_TINY) is None


def test_reconcile_is_scan(reg: Registry) -> None:
    result = reg.reconcile()
    assert isinstance(result, ScanResult)
    assert reg.get(QWEN_TINY) is not None


# ---------------------------------------------------------------------------
# Aliases + resolution
# ---------------------------------------------------------------------------


def test_resolve_by_full_id_and_bare_stem(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert scanned.resolve(QWEN_TINY) is record
    assert scanned.resolve("Qwen2.5-0.5B-Instruct-Q8_0") is record
    # LM Studio hands us the lowercased short form; it must work.
    assert scanned.resolve("qwen2.5-0.5b-instruct-q8_0") is record


def test_resolve_by_publisher_and_repo_scoped_names(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert scanned.resolve("lmstudio-community/Qwen2.5-0.5B-Instruct-Q8_0") is record
    assert scanned.resolve("Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0") is record


def test_resolve_by_repo_name_without_gguf_suffix(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert scanned.resolve("qwen2.5-0.5b-instruct") is record
    assert scanned.resolve("GEMMA-4-31B-IT-QAT") is scanned.get(GEMMA_31B)


def test_resolve_by_the_repo_name_as_pasted_from_huggingface(scanned: Registry) -> None:
    """The string an agent copies off the model page -- suffix and all, with or
    without the publisher -- used to 404 while its abbreviation resolved."""
    record = scanned.get(QWEN_TINY)
    assert scanned.resolve("Qwen2.5-0.5B-Instruct-GGUF") is record
    assert scanned.resolve("lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF") is record
    assert scanned.resolve("lmstudio-community/Qwen2.5-0.5B-Instruct") is record


def test_resolve_is_whitespace_tolerant_and_misses_cleanly(scanned: Registry) -> None:
    assert scanned.resolve(f"  {QWEN_TINY}  ") is scanned.get(QWEN_TINY)
    assert scanned.resolve("no-such-model") is None


def test_alias_collision_keeps_first_and_warns(reg: Registry, recorder: LogRecorder) -> None:
    reg.scan()
    winner = "collide-a/repo-one-GGUF/samename-Q4_K_M"
    loser = "collide-b/repo-two-GGUF/samename-Q4_K_M"
    resolved = reg.resolve("samename-q4_k_m")
    assert resolved is not None
    assert resolved.id == winner
    collisions = recorder.of("registry.alias_collision", "warning")
    assert any(kw["alias"] == "samename-q4_k_m" and kw["dropped"] == loser for kw in collisions)
    # The shadowed model is still reachable by its full id -- never lost.
    assert reg.get(loser) is not None
    assert reg.resolve(loser) is not None


def test_full_ids_never_shadowed_by_short_aliases(scanned: Registry) -> None:
    for model_id in scanned.known_ids():
        resolved = scanned.resolve(model_id)
        assert resolved is not None
        assert resolved.id == model_id


# ---------------------------------------------------------------------------
# openai_list
# ---------------------------------------------------------------------------


def test_openai_list_shape(scanned: Registry) -> None:
    payload = scanned.openai_list()
    assert len(payload) == len(scanned.all())
    entry = next(e for e in payload if e["id"] == QWEN_TINY)
    assert entry["object"] == "model"
    assert entry["owned_by"] == "lmstudio-community"
    assert entry["type"] == "llm"
    assert entry["studioforge"]["kind"] == "chat"
    embedding = next(e for e in payload if e["id"] == EMBEDDING)
    assert embedding["type"] == "embeddings"
    assert "embedding" in embedding["capabilities"]


def test_all_is_sorted_by_id(scanned: Registry) -> None:
    ids = [record.id for record in scanned.all()]
    assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_default_to_empty(scanned: Registry) -> None:
    settings = scanned.get_settings(QWEN_TINY)
    assert settings.ctx_size is None
    assert settings.pinned is False
    assert settings.adapters == []


def test_save_settings_round_trip_keeps_none_as_none(scanned: Registry, db: Database) -> None:
    record = scanned.save_settings(
        QWEN_TINY, ModelSettings(ctx_size=32768, pinned=True, temperature=0.7)
    )
    assert record.settings.ctx_size == 32768

    payload = db.get_model_settings(QWEN_TINY)
    assert payload is not None
    assert payload["ctx_size"] == 32768
    assert payload["pinned"] is True
    # "Auto" must survive as null so the planner still decides at load time.
    assert payload["kv_cache_type"] is None
    assert payload["flash_attn"] is None
    assert payload["split_mode"] is None
    assert payload["top_k"] is None

    assert scanned.get_settings(QWEN_TINY).ctx_size == 32768


def test_saved_settings_survive_rescan(scanned: Registry) -> None:
    scanned.save_settings(QWEN_TINY, ModelSettings(ctx_size=16384, mlock=True))
    scanned.scan()
    settings = scanned.get_settings(QWEN_TINY)
    assert settings.ctx_size == 16384
    assert settings.mlock is True
    assert settings.ttl_s is None


def test_save_settings_rejects_invalid_values(scanned: Registry) -> None:
    bad = ModelSettings.model_construct(ctx_size=-1)
    with pytest.raises(BadRequestError):
        scanned.save_settings(QWEN_TINY, bad)


def test_save_settings_unknown_model(scanned: Registry) -> None:
    with pytest.raises(ModelNotFoundError):
        scanned.save_settings("nope/nope", ModelSettings())


def test_touch_sets_last_used_at(scanned: Registry) -> None:
    record = scanned.get(QWEN_TINY)
    assert record is not None and record.last_used_at is None
    scanned.touch(QWEN_TINY)
    assert record.last_used_at is not None
    scanned.touch("nope/nope")  # must not raise


def test_added_at_survives_rescan(scanned: Registry) -> None:
    before = scanned.get(QWEN_TINY)
    assert before is not None
    original = before.added_at
    scanned.touch(QWEN_TINY)
    scanned.scan()
    after = scanned.get(QWEN_TINY)
    assert after is not None
    assert after.added_at == original
    assert after.last_used_at is not None


# ---------------------------------------------------------------------------
# Metadata cache
# ---------------------------------------------------------------------------


def test_warm_scan_makes_no_reader_calls(reg: Registry, reader: FakeMetaReader) -> None:
    reg.scan()
    cold_calls = reader.count
    assert cold_calls > 20  # every gguf, including mmprojs and the adapter

    reader.reset()
    reg.scan()
    # The only file re-read is the one that cannot be cached because it never
    # parsed; every good file is served from SQLite.
    assert [p.name for p in reader.calls] == ["corrupt-me-Q4_K_M.gguf"]


def test_unparsable_file_is_retried_on_every_scan(
    reg: Registry, reader: FakeMetaReader, db: Database, library: Path
) -> None:
    """A failed parse writes no cache row on purpose: the file may be a
    download still in flight, and a later scan must pick it up."""
    reg.scan()
    corrupt = library / "broken/Corrupt-GGUF/corrupt-me-Q4_K_M.gguf"
    stat = corrupt.stat()
    assert db.get_cached_meta(str(corrupt), stat.st_mtime, stat.st_size) is None
    reader.reset()
    reg.scan()
    assert corrupt.name in [p.name for p in reader.calls]


def test_touching_mtime_reparses_only_that_file(
    reg: Registry, reader: FakeMetaReader, library: Path
) -> None:
    reg.scan()
    reader.reset()
    target = library / TREE[0]
    future = time.time() + 120
    os.utime(target, (future, future))
    reg.scan()
    # corrupt-me never caches, so it always shows up; nothing else should.
    assert {p.name for p in reader.calls} == {target.name, "corrupt-me-Q4_K_M.gguf"}


def test_changing_size_reparses(reg: Registry, reader: FakeMetaReader, library: Path) -> None:
    reg.scan()
    reader.reset()
    target = library / TREE[0]
    with target.open("ab") as handle:
        handle.write(b"\0" * 128)
    reg.scan()
    assert target.name in [p.name for p in reader.calls]


def test_force_bypasses_the_cache(reg: Registry, reader: FakeMetaReader) -> None:
    reg.scan()
    cold = reader.count
    reader.reset()
    reg.scan(force=True)
    assert reader.count == cold


def test_scan_prunes_stale_cache_rows(reg: Registry, db: Database) -> None:
    stale = "Z:/gone/deleted-model-Q4_K_M.gguf"
    db.put_cached_meta(stale, 1.0, 10, {"architecture": "llama"})
    assert db.get_cached_meta(stale, 1.0, 10) is not None
    reg.scan()
    assert db.get_cached_meta(stale, 1.0, 10) is None


def test_cache_rows_written_for_every_file(reg: Registry, db: Database, library: Path) -> None:
    reg.scan()
    path = library / TREE[0]
    stat = path.stat()
    cached = db.get_cached_meta(_cache_key(path), stat.st_mtime, stat.st_size)
    assert cached is not None
    assert cached["architecture"] == "qwen3"


def test_corrupt_cache_row_is_reparsed(
    reg: Registry, db: Database, reader: FakeMetaReader, library: Path
) -> None:
    path = library / TREE[0]
    stat = path.stat()
    db.put_cached_meta(_cache_key(path), stat.st_mtime, stat.st_size, {"n_layer": "not-an-int"})
    reader.reset()
    reg.scan()
    assert path.name in [p.name for p in reader.calls]
    record = reg.get(QWEN_TINY)
    assert record is not None
    assert record.architecture == "qwen3"


def test_the_cache_key_carries_the_parser_version(library: Path) -> None:
    """(path, mtime, size) cannot detect a change to the *parser*.

    A model registered before ``read_meta`` learned to read
    ``full_attention_interval`` would otherwise serve metadata without it
    forever -- nothing about the file changed -- and the planner would keep
    charging that model four times its real KV cache.
    """
    path = library / TREE[0]
    assert _cache_key(path).startswith(str(path))
    assert _cache_key(path).endswith(f"#meta{gguf.META_FORMAT_VERSION}")


def test_a_parser_version_bump_reparses_every_registered_model(
    reg: Registry, db: Database, reader: FakeMetaReader, library: Path, monkeypatch
) -> None:
    """The whole point: an ordinary scan after a bump re-reads the headers."""
    reg.scan()
    reader.reset()
    reg.scan()
    # Unchanged files are served from cache. (The deliberately corrupt fixture
    # never caches, so it is re-read on every scan and is not evidence here.)
    assert library / TREE[0] not in reader.calls

    monkeypatch.setattr(gguf, "META_FORMAT_VERSION", gguf.META_FORMAT_VERSION + 1)
    reader.reset()
    reg.scan()
    assert library / TREE[0] in reader.calls

    # And the superseded rows do not accumulate: they are absent from the new
    # scan's keep-list, so the same scan prunes them.
    path = library / TREE[0]
    stat = path.stat()
    stale = f"{path}#meta{gguf.META_FORMAT_VERSION - 1}"
    assert db.get_cached_meta(stale, stat.st_mtime, stat.st_size) is None


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_adapters_are_detected_and_persisted(scanned: Registry, db: Database) -> None:
    adapters = scanned.adapters()
    assert [a.id for a in adapters] == [LORA]
    adapter = adapters[0]
    assert adapter.base_architecture == "llama"
    assert adapter.rank == 32
    assert adapter.n_layer == 32
    assert adapter.base_model_hint == "meta-llama/Llama-3-17B"
    assert adapter.publisher == "LORAs"
    assert adapter.repo == "Zeta-17b-GGUF"

    rows = db.list_adapters()
    assert [row["id"] for row in rows] == [LORA]


def test_scan_adapters_returns_them(reg: Registry) -> None:
    adapters = reg.scan_adapters()
    assert [a.id for a in adapters] == [LORA]


def test_get_adapter(scanned: Registry) -> None:
    assert scanned.get_adapter(LORA) is not None
    assert scanned.get_adapter("nope") is None


def test_adapters_never_appear_as_models(scanned: Registry) -> None:
    assert LORA not in scanned.known_ids()
    assert scanned.get(LORA) is None
    assert all(entry["id"] != LORA for entry in scanned.openai_list())


def test_adapters_hydrate_from_db_without_a_scan(
    scanned: Registry, library: Path, tmp_path: Path, db: Database
) -> None:
    fresh = Registry(make_config(library, tmp_path), db, meta_reader=FakeMetaReader())
    assert [a.id for a in fresh.adapters()] == [LORA]


def test_delete_adapter_row_only(scanned: Registry, db: Database, library: Path) -> None:
    path = library / f"{LORA}.gguf"
    scanned.delete_adapter(LORA)
    assert scanned.get_adapter(LORA) is None
    assert db.get_adapter(LORA) is None
    assert path.exists()


def test_delete_adapter_with_file(scanned: Registry, library: Path) -> None:
    path = library / f"{LORA}.gguf"
    scanned.delete_adapter(LORA, delete_file=True)
    assert not path.exists()


def test_delete_unknown_adapter(scanned: Registry) -> None:
    with pytest.raises(BadRequestError):
        scanned.delete_adapter("nope")


# ---------------------------------------------------------------------------
# Virtual models
# ---------------------------------------------------------------------------


def _make_virtual(reg: Registry, model_id: str = "virtual/tiny-rp") -> Any:
    return reg.create_virtual_model(
        id=model_id,
        base_model_id=QWEN_TINY,
        name="Tiny RP",
        adapters=[AdapterAttachment(adapter_id=LORA, scale=0.8)],
    )


def test_create_virtual_model_copies_base_files(scanned: Registry) -> None:
    base = scanned.get(QWEN_TINY)
    record = _make_virtual(scanned)
    assert base is not None
    assert record.is_virtual is True
    assert record.base_model_id == QWEN_TINY
    assert record.path == base.path
    assert record.shards == base.shards
    assert record.mmproj_path == base.mmproj_path
    assert record.meta is base.meta
    assert record.capabilities.model_dump() == base.capabilities.model_dump()
    assert [a.adapter_id for a in record.settings.adapters] == [LORA]
    assert record.settings.adapters[0].scale == 0.8


def test_virtual_model_copies_vision_capability(scanned: Registry) -> None:
    record = scanned.create_virtual_model(
        id="virtual/see-things", base_model_id=GEMMA_31B, name=None, adapters=[]
    )
    assert record.capabilities.vision is True
    assert record.mmproj_path is not None
    assert record.name == "virtual/see-things"


def test_virtual_model_is_listed_and_resolvable(scanned: Registry) -> None:
    _make_virtual(scanned)
    ids = scanned.known_ids()
    assert "virtual/tiny-rp" in ids
    assert scanned.resolve("virtual/tiny-rp") is not None
    assert scanned.resolve("VIRTUAL/TINY-RP") is not None
    assert scanned.resolve("tiny rp") is not None
    entry = next(e for e in scanned.openai_list() if e["id"] == "virtual/tiny-rp")
    assert entry["studioforge"]["is_virtual"] is True
    assert entry["studioforge"]["base_model_id"] == QWEN_TINY
    assert entry["studioforge"]["adapters"] == [{"adapter_id": LORA, "scale": 0.8}]


def test_virtual_model_rejects_unknown_base(scanned: Registry) -> None:
    with pytest.raises(BadRequestError, match="base model"):
        scanned.create_virtual_model(
            id="virtual/x", base_model_id="nope/nope", name=None, adapters=[]
        )


def test_virtual_model_rejects_virtual_base(scanned: Registry) -> None:
    _make_virtual(scanned)
    with pytest.raises(BadRequestError):
        scanned.create_virtual_model(
            id="virtual/y", base_model_id="virtual/tiny-rp", name=None, adapters=[]
        )


def test_virtual_model_rejects_unknown_adapter(scanned: Registry) -> None:
    with pytest.raises(BadRequestError, match="adapter"):
        scanned.create_virtual_model(
            id="virtual/z",
            base_model_id=QWEN_TINY,
            name=None,
            adapters=[AdapterAttachment(adapter_id="LORAs/ghost")],
        )


def test_virtual_model_rejects_real_id_collision(scanned: Registry) -> None:
    with pytest.raises(BadRequestError, match="collides"):
        scanned.create_virtual_model(id=QWEN_TINY, base_model_id=QWEN_TINY, name=None, adapters=[])


def test_virtual_model_rejects_empty_id(scanned: Registry) -> None:
    with pytest.raises(BadRequestError):
        scanned.create_virtual_model(id="   ", base_model_id=QWEN_TINY, name=None, adapters=[])


def test_virtual_models_survive_rescan_and_restart(
    scanned: Registry, library: Path, tmp_path: Path, db: Database
) -> None:
    _make_virtual(scanned)
    scanned.scan()
    still = scanned.get("virtual/tiny-rp")
    assert still is not None
    assert [a.adapter_id for a in still.settings.adapters] == [LORA]

    fresh = Registry(make_config(library, tmp_path), db, meta_reader=FakeMetaReader())
    fresh.scan()
    reloaded = fresh.get("virtual/tiny-rp")
    assert reloaded is not None
    assert reloaded.is_virtual is True
    assert reloaded.base_model_id == QWEN_TINY
    assert [a.adapter_id for a in reloaded.settings.adapters] == [LORA]


def test_virtual_model_settings_round_trip(scanned: Registry) -> None:
    _make_virtual(scanned)
    updated = scanned.save_settings(
        "virtual/tiny-rp",
        ModelSettings(ctx_size=8192, adapters=[AdapterAttachment(adapter_id=LORA, scale=0.5)]),
    )
    assert updated.settings.ctx_size == 8192
    scanned.scan()
    after = scanned.get("virtual/tiny-rp")
    assert after is not None
    assert after.settings.ctx_size == 8192
    assert after.settings.adapters[0].scale == 0.5


def test_virtual_model_dropped_when_base_disappears(
    scanned: Registry, library: Path, recorder: LogRecorder
) -> None:
    _make_virtual(scanned)
    (library / TREE[0]).unlink()
    scanned.scan()
    assert scanned.get("virtual/tiny-rp") is None
    assert recorder.of("registry.virtual_base_missing", "warning")


def test_delete_virtual_model(scanned: Registry, db: Database) -> None:
    _make_virtual(scanned)
    scanned.delete_virtual_model("virtual/tiny-rp")
    assert scanned.get("virtual/tiny-rp") is None
    assert db.list_virtual_models() == []
    assert scanned.resolve("tiny rp") is None


def test_delete_virtual_model_rejects_real_model(scanned: Registry) -> None:
    with pytest.raises(BadRequestError):
        scanned.delete_virtual_model(QWEN_TINY)


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_delete_model_dry_run_lists_files(scanned: Registry, library: Path) -> None:
    files = scanned.delete_model(QWEN_TINY)
    assert [p.name for p in files] == ["Qwen2.5-0.5B-Instruct-Q8_0.gguf"]
    assert (library / TREE[0]).exists()
    assert scanned.get(QWEN_TINY) is None


def test_delete_model_with_files_removes_shards_and_mmproj(
    scanned: Registry, library: Path
) -> None:
    behemoth = scanned.get(BEHEMOTH)
    assert behemoth is not None
    files = scanned.delete_model(BEHEMOTH, delete_files=True)
    assert len(files) == 2
    assert not any(path.exists() for path in files)

    gemma = scanned.get(GEMMA_31B)
    assert gemma is not None and gemma.mmproj_path is not None
    removed = scanned.delete_model(GEMMA_31B, delete_files=True)
    assert len(removed) == 2  # base + its projector
    assert not any(path.exists() for path in removed)
    del library


def test_delete_model_keeps_a_shared_mmproj(scanned: Registry, library: Path) -> None:
    """Both bases in a two-quant directory point at one projector, so deleting
    one model must not take the other's projector with it."""
    shared_dir = library / "shared/Two-Quants-GGUF"
    shared_dir.mkdir(parents=True)
    for name in ("shared-model-Q4_K_M.gguf", "shared-model-Q8_0.gguf", "shared-model-mmproj.gguf"):
        (shared_dir / name).write_bytes(b"\0" * 128)
    scanned.scan()
    first = scanned.get("shared/Two-Quants-GGUF/shared-model-Q4_K_M")
    second = scanned.get("shared/Two-Quants-GGUF/shared-model-Q8_0")
    assert first is not None and second is not None
    assert first.mmproj_path == second.mmproj_path is not None

    removed = scanned.delete_model(first.id, delete_files=True)
    assert [p.name for p in removed] == ["shared-model-Q4_K_M.gguf"]
    assert (shared_dir / "shared-model-mmproj.gguf").exists()


def test_delete_model_refuses_paths_outside_model_dirs(scanned: Registry, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "precious.gguf"
    outside.parent.mkdir(parents=True)
    outside.write_bytes(b"\0" * 16)
    record = scanned.get(QWEN_TINY)
    assert record is not None
    record.path = outside
    record.shards = [outside]

    with pytest.raises(BadRequestError, match="outside the configured model directories"):
        scanned.delete_model(QWEN_TINY, delete_files=True)
    assert outside.exists()


def test_delete_model_unknown_id(scanned: Registry) -> None:
    with pytest.raises(ModelNotFoundError):
        scanned.delete_model("nope/nope")


def test_delete_model_removes_dependent_virtual_models(scanned: Registry, db: Database) -> None:
    _make_virtual(scanned)
    scanned.delete_model(QWEN_TINY)
    assert scanned.get("virtual/tiny-rp") is None
    assert db.list_virtual_models() == []


def test_delete_virtual_via_delete_model_touches_no_files(scanned: Registry, library: Path) -> None:
    _make_virtual(scanned)
    assert scanned.delete_model("virtual/tiny-rp", delete_files=True) == []
    assert (library / TREE[0]).exists()


# ---------------------------------------------------------------------------
# Multiple model directories
# ---------------------------------------------------------------------------


def test_extra_dirs_are_scanned(library: Path, tmp_path: Path, db: Database) -> None:
    second = tmp_path / "models2"
    (second / "other/Extra-GGUF").mkdir(parents=True)
    (second / "other/Extra-GGUF/extra-model-Q4_K_M.gguf").write_bytes(b"\0" * 64)
    config = make_config(library, tmp_path, extra_dirs=[second])
    reg = Registry(config, db, meta_reader=FakeMetaReader())
    reg.scan()
    assert "other/Extra-GGUF/extra-model-Q4_K_M" in reg.known_ids()
    assert QWEN_TINY in reg.known_ids()


def test_duplicate_id_across_dirs_is_an_error(library: Path, tmp_path: Path, db: Database) -> None:
    second = tmp_path / "models2"
    duplicate = second / TREE[0]
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"\0" * 64)
    config = make_config(library, tmp_path, extra_dirs=[second])
    reg = Registry(config, db, meta_reader=FakeMetaReader())
    result = reg.scan()
    assert any(model_id == QWEN_TINY and "duplicate" in msg for model_id, msg in result.errors)
    record = reg.get(QWEN_TINY)
    assert record is not None
    # Primary dir wins: config.model_dirs() lists it first.
    assert record.path.is_relative_to(library)


# ---------------------------------------------------------------------------
# Concurrent scans (startup + GUI run_blocking + download-completion callback)
# ---------------------------------------------------------------------------
#
# scan() is now genuinely called from multiple threads at once: the startup
# to_thread scan, the /api/models/scan route, the GUI, and the post-download
# completion callback. These tests hammer that combination against the real
# Registry + real SQLite Database to prove the locking holds.


def test_concurrent_scans_from_many_threads_stay_consistent(
    library: Path, tmp_path: Path, db: Database, reader: FakeMetaReader
) -> None:
    import concurrent.futures

    reg = Registry(make_config(library, tmp_path), db, meta_reader=reader)
    reg.scan()
    baseline = set(reg.known_ids())
    assert baseline, "the synthetic library must produce models"

    errors: list[BaseException] = []

    def hammer_scan(force: bool) -> None:
        try:
            for _ in range(5):
                reg.scan(force=force)
        except BaseException as exc:  # noqa: BLE001 - collected for the assert
            errors.append(exc)

    def hammer_read() -> None:
        try:
            for _ in range(50):
                for record in reg.all():
                    assert reg.resolve(record.id) is not None
                reg.openai_list()
                reg.adapters()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            *(pool.submit(hammer_scan, force=(i % 2 == 0)) for i in range(4)),
            *(pool.submit(hammer_read) for _ in range(4)),
        ]
        for future in futures:
            future.result(timeout=120)

    assert errors == [], f"concurrent scan/read raised: {errors!r}"
    assert set(reg.known_ids()) == baseline, "concurrent scans changed the model set"
    # The metadata cache must still serve every model (no rows lost to a
    # concurrent prune): one more scan re-parses nothing except the corrupt
    # fixture, whose failures are never cached even in a serial scan.
    reader.reset()
    reg.scan()
    unexpected = [p.name for p in reader.calls if "corrupt" not in p.name.lower()]
    assert unexpected == [], (
        f"concurrent scans lost cache rows; these files were re-parsed: {unexpected}"
    )


def test_concurrent_scans_with_virtual_model_churn(
    library: Path, tmp_path: Path, db: Database, reader: FakeMetaReader
) -> None:
    """Virtual models created/deleted while scans rebuild the map must survive
    (a scan swaps the whole record dict, then re-hydrates virtuals from SQLite;
    a create racing that swap must not vanish)."""
    import concurrent.futures

    reg = Registry(make_config(library, tmp_path), db, meta_reader=reader)
    reg.scan()

    errors: list[BaseException] = []

    def hammer_scan() -> None:
        try:
            for _ in range(10):
                reg.scan()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def churn_virtuals(slot: int) -> None:
        try:
            for i in range(10):
                vid = f"virtual/churn-{slot}"
                reg.create_virtual_model(
                    id=vid, base_model_id=QWEN_TINY, name=f"churn {slot}.{i}", adapters=[]
                )
                assert reg.resolve(vid) is not None
                reg.delete_virtual_model(vid)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            *(pool.submit(hammer_scan) for _ in range(2)),
            *(pool.submit(churn_virtuals, slot) for slot in range(2)),
        ]
        for future in futures:
            future.result(timeout=120)

    assert errors == [], f"concurrent scan/virtual churn raised: {errors!r}"

    # A virtual model that exists when the dust settles must still resolve
    # after one more scan (it is re-hydrated from SQLite inside the lock).
    reg.create_virtual_model(id="virtual/final", base_model_id=QWEN_TINY, name="f", adapters=[])
    reg.scan()
    record = reg.resolve("virtual/final")
    assert record is not None and record.is_virtual


# ---------------------------------------------------------------------------
# Live test over the real library
# ---------------------------------------------------------------------------

def _library_root() -> Path | None:
    """``SF_TEST_MODELS_DIR``, else the library the app itself would detect."""
    env = os.environ.get("SF_TEST_MODELS_DIR", "").strip()
    if env:
        return Path(env)
    from studioforge.config import detect_model_dir

    return detect_model_dir()


REAL_LIBRARY = _library_root() or Path("<no-model-library-detected>")

live = pytest.mark.skipif(
    not REAL_LIBRARY.is_dir() or not hasattr(gguf, "read_meta"),
    reason=(
        "no real model library found; set SF_TEST_MODELS_DIR to run the live "
        "registry scan"
    ),
)


@live
def test_live_real_library(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    database = Database(tmp_path / "live.sqlite3")
    database.migrate()
    config = Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=REAL_LIBRARY))
    reg = Registry(config, database)  # real gguf.read_meta

    cold = reg.scan()
    warm = reg.scan()

    records = reg.all()
    gguf_files = [p for p in REAL_LIBRARY.rglob("*.gguf") if ".cache" not in p.parts]

    lines: list[str] = []
    lines.append("")
    lines.append(f"model dir            : {REAL_LIBRARY}")
    lines.append(f"gguf files on disk   : {len(gguf_files)}")
    lines.append(f"logical models       : {len(records)}")
    lines.append(f"adapters             : {len(reg.adapters())}")
    lines.append(f"scan errors          : {len(cold.errors)}")
    lines.append(f"cold scan            : {cold.duration_s:.3f}s")
    lines.append(f"warm scan            : {warm.duration_s:.3f}s")
    if warm.duration_s > 0:
        lines.append(f"speedup              : {cold.duration_s / warm.duration_s:.1f}x")
    for model_id, message in cold.errors:
        lines.append(f"  ERROR {model_id}: {message}")

    header = f"{'id':<74} {'kind':<9} {'quant':<10} {'GiB':>8} {'vis':<4} {'ctx':>7} {'files':>5}"
    lines.append("")
    lines.append(header)
    lines.append("-" * len(header))
    for record in records:
        lines.append(
            f"{record.id[:74]:<74} "
            f"{record.kind:<9} "
            f"{record.quant:<10} "
            f"{record.size_bytes / GIB:>8.2f} "
            f"{('yes' if record.capabilities.vision else '-'):<4} "
            f"{(record.meta.n_ctx_train if record.meta else 0):>7} "
            f"{len(record.shards):>5}"
        )
    vision = [r for r in records if r.capabilities.vision]
    lines.append("")
    lines.append("mmproj pairing:")
    for record in vision:
        proj = record.mmproj_path.name if record.mmproj_path else "(metadata only)"
        lines.append(f"  {record.id[:70]:<70} <- {proj}")
    lines.append("")
    lines.append(
        "totals: "
        f"{sum(r.size_bytes for r in records) / GIB:.1f} GiB, "
        f"vision={len(vision)}, "
        f"embedding={sum(1 for r in records if r.kind == 'embedding')}, "
        f"multipart={sum(1 for r in records if r.capabilities.multi_part)}, "
        f"tools={sum(1 for r in records if r.capabilities.tools)}"
    )

    with capsys.disabled():
        print("\n".join(lines))

    # --- assertions ---------------------------------------------------
    # This runs against the operator's REAL library, which grows whenever they
    # download something -- so there is no upper bound to assert. The real
    # invariant is that folding works: projectors and extra shards must be
    # absorbed into their parent rather than counted as models of their own,
    # which the structural assertions below pin down.
    assert len(records) >= 24, f"unexpected model count: {len(records)}"
    assert len(records) == len({r.id for r in records}), "duplicate model ids"

    behemoth = next(r for r in records if "Behemoth" in r.id)
    assert len(behemoth.shards) == 2
    assert behemoth.capabilities.multi_part is True
    assert behemoth.size_bytes == sum(p.stat().st_size for p in behemoth.shards)

    paired = [r for r in records if r.mmproj_path is not None]
    assert len(paired) >= 6, f"only {len(paired)} models got a projector"
    assert all(r.capabilities.vision for r in paired)

    embeddings = [r for r in records if r.kind == "embedding"]
    assert embeddings, "expected at least one embedding model"
    assert any("Qwen3-Embedding-8B" in r.id for r in embeddings)

    for record in records:
        assert "mmproj" not in record.path.name.lower(), record.id
        assert record.meta is not None
        assert record.meta.is_mmproj is False
        assert record.meta.is_adapter is False
        assert record.path.exists()

    # Warm re-scan must be dramatically cheaper than the cold one.
    assert warm.duration_s < cold.duration_s
    assert len(reg.all()) == len(records)

    # Every id resolves, and every short alias that is unique resolves too.
    for record in records:
        assert reg.resolve(record.id) is record
        assert reg.resolve(record.name.lower()) is not None

    database.close()
