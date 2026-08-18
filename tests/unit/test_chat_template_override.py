"""Per-model chat-template override.

Incident (production, the system this replaces): *"a model's baked-in Jinja
template used ``raise_exc``, which the engine could not compile, producing a
400 on certain request types."* With no override there is no fix short of
re-quantising the model.

Also a spec requirement: *"use the template embedded in the GGUF by default;
allow per-model override (Jinja file). Never hardcode a template."* The default
half was already done; this is the override half.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.db import Database
from studioforge.errors import BadRequestError
from studioforge.types import (
    GgufMeta,
    LoadPlan,
    ModelRecord,
    ModelSettings,
    validate_chat_template_file,
)

MODEL_REL = "publisher/repo-GGUF/model-Q4_K_M.gguf"
MODEL_ID = "publisher/repo-GGUF/model-Q4_K_M"

TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


def meta_reader(path: Path, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
    return GgufMeta(architecture="qwen3", n_layer=32, tensor_bytes=4096, quant_label="Q4_K_M")


def resolver(path: Path) -> Callable[[str | None], Path]:
    def resolve(tag: str | None) -> Path:
        return path

    return resolve


@pytest.fixture()
def template_file(tmp_path: Path) -> Path:
    path = tmp_path / "templates" / "fixed.jinja"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The flag
# ---------------------------------------------------------------------------


def build_argv(tmp_path: Path, settings: ModelSettings) -> list[str]:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    binary = tmp_path / "engines" / "b10425" / "llama-server.exe"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("stub", encoding="utf-8")
    model_path = tmp_path / "models" / "model.gguf"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"GGUF")
    record = ModelRecord(id="m", name="m", path=model_path, settings=settings)
    supervisor = Supervisor(config, resolve_binary=resolver(binary))
    return supervisor.build_command(record, LoadPlan(model_id="m", devices=[0]), port=18100)


def test_the_flag_is_emitted_when_a_template_is_set(tmp_path: Path, template_file: Path) -> None:
    argv = build_argv(tmp_path, ModelSettings(chat_template_file=template_file))

    assert "--chat-template-file" in argv
    assert argv[argv.index("--chat-template-file") + 1] == str(template_file)


def test_no_flag_without_an_override(tmp_path: Path) -> None:
    """The GGUF's embedded template is the default and must stay the default."""
    argv = build_argv(tmp_path, ModelSettings())

    assert "--chat-template-file" not in argv
    assert not any(arg == "--chat-template" for arg in argv), "a template must never be hardcoded"


# ---------------------------------------------------------------------------
# Save-time validation
# ---------------------------------------------------------------------------


def test_validator_accepts_a_readable_file(template_file: Path) -> None:
    assert validate_chat_template_file(template_file) == template_file


def test_validator_accepts_none() -> None:
    assert validate_chat_template_file(None) is None


def test_validator_names_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jinja"

    with pytest.raises(ValueError) as excinfo:
        validate_chat_template_file(missing)

    assert str(missing) in str(excinfo.value)
    assert "does not exist" in str(excinfo.value)


def test_validator_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="is not a file"):
        validate_chat_template_file(tmp_path)


@pytest.fixture()
def registry(tmp_path: Path) -> Registry:
    library = tmp_path / "models"
    path = library / MODEL_REL
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * 4096)
    db = Database(tmp_path / "data" / "registry.sqlite3")
    db.migrate()
    config = Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=library))
    reg = Registry(config, db, meta_reader=meta_reader)
    reg.scan()
    return reg


def test_saving_a_missing_template_is_a_clear_400(registry: Registry, tmp_path: Path) -> None:
    """Every save path goes through the registry, so validation lives there."""
    missing = tmp_path / "gone.jinja"

    with pytest.raises(BadRequestError) as excinfo:
        registry.save_settings(MODEL_ID, ModelSettings(chat_template_file=missing))

    assert excinfo.value.param == "chat_template_file"
    assert "gone.jinja" in excinfo.value.message
    assert registry.get_settings(MODEL_ID).chat_template_file is None


def test_saving_a_real_template_round_trips(registry: Registry, template_file: Path) -> None:
    registry.save_settings(MODEL_ID, ModelSettings(chat_template_file=template_file))

    assert registry.get_settings(MODEL_ID).chat_template_file == template_file


def test_a_template_deleted_after_saving_does_not_reset_the_model(
    registry: Registry, template_file: Path
) -> None:
    """Why this is not a pydantic field validator.

    If existence were checked on every hydration, deleting the file would make
    the stored settings row invalid and the model would silently fall back to
    defaults -- an invisible behaviour change instead of a fixable error.
    """
    registry.save_settings(MODEL_ID, ModelSettings(chat_template_file=template_file, ctx_size=4096))
    template_file.unlink()

    settings = registry.get_settings(MODEL_ID)

    assert settings.ctx_size == 4096
    assert settings.chat_template_file == template_file
