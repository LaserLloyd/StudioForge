"""Config loading and validation: the edge cases a stranger's config.yaml hits.

* out-of-range values fail at load with a message naming the key, instead of
  being accepted and crashing later (``server.port: 70000`` used to reach
  ``socket.bind`` as an ``OverflowError`` traceback);
* a typo in ``logging.level`` is refused instead of silently downgraded to INFO;
* ``mcp.path`` without a leading slash is normalised instead of silently
  disabling MCP (and with it the PIN);
* unknown keys are ignored *loudly*;
* ``data_dir`` is never written into config.yaml and never read from it (D31):
  the data directory is ``SF_DATA_DIR``, else where a named config file lives,
  else the checkout/platform default;
* an empty config.yaml is treated as missing (defaults, a warning, regenerated),
  not as "every setting silently back to default forever";
* ``save()`` fsyncs and keeps a ``.bak``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from studioforge.config import Config, apply_overrides, load_config, resolve_data_dir
from studioforge.errors import ConfigError


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _logged(capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture) -> str:
    """Whatever the log line landed in: structlog's console (stdout) or, once
    another test has configured file/JSON logging, the stdlib record."""
    out = capsys.readouterr()
    return out.out + out.err + caplog.text


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("server", "port", 70000),
        ("server", "port", 0),
        ("gui", "port", -3),
        ("watchdog", "port", 65536),
        ("models", "default_ctx", -1),
        ("models", "default_ctx", 0),
        ("models", "default_ttl_s", -5),
        ("models", "target_ctx", 0),
        ("engine", "keep_versions", 0),
        ("hf", "max_concurrent_downloads", 0),
        ("hf", "chunk_bytes", 0),
        ("planner", "cuda_context_mb", -100),
        ("gateway", "max_image_bytes", -1),
        ("gateway", "load_timeout_s", 0),
        ("server", "drain_timeout_s", -1),
        ("logging", "level", "LOUD"),
        ("mcp", "path", "/"),
    ],
)
def test_out_of_range_values_fail_at_load_and_name_the_key(
    tmp_path: Path, section: str, key: str, value: Any
) -> None:
    path = tmp_path / "data" / "config.yaml"
    _write(path, {section: {key: value}})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert key in excinfo.value.message
    assert section in excinfo.value.message


@pytest.mark.parametrize("value", ["info", "Warning", "warn", "DEBUG"])
def test_logging_level_is_case_insensitive_and_warn_is_accepted(tmp_path: Path, value: str) -> None:
    path = tmp_path / "data" / "config.yaml"
    _write(path, {"logging": {"level": value}})
    config = load_config(path)
    assert config.logging.level == (
        "WARNING" if value.lower().startswith("warn") else value.upper()
    )


@pytest.mark.parametrize(
    ("given", "expected"), [("mcp", "/mcp"), ("/mcp/", "/mcp"), (" /x ", "/x")]
)
def test_mcp_path_is_normalised_to_a_rooted_route(
    tmp_path: Path, given: str, expected: str
) -> None:
    config = Config(data_dir=tmp_path, mcp={"path": given})
    assert config.mcp.path == expected


def test_models_dir_pointing_at_a_file_is_refused(tmp_path: Path) -> None:
    file = tmp_path / "not-a-dir.gguf"
    file.write_bytes(b"GGUF")
    path = tmp_path / "data" / "config.yaml"
    _write(path, {"models": {"dir": str(file)}})
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "models.dir" in excinfo.value.message
    assert "not a directory" in excinfo.value.message


def test_unknown_keys_are_ignored_but_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "data" / "config.yaml"
    _write(path, {"foo": 1, "server": {"api_kye": "x", "port": 1234}, "models": {"max_loaded": 3}})
    config = load_config(path)
    assert config.server.port == 1234
    text = _logged(capsys, caplog)
    assert "foo" in text and "server.api_kye" in text and "models.max_loaded" in text


# ---------------------------------------------------------------------------
# data_dir (D31)
# ---------------------------------------------------------------------------


def test_data_dir_is_never_written_to_config_yaml(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    saved = config.save()
    on_disk = yaml.safe_load(saved.read_text(encoding="utf-8"))
    assert "data_dir" not in on_disk
    assert "data_dir" not in config.to_yaml_dict()


def test_a_named_config_file_lives_in_its_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    monkeypatch.delenv("SF_CONFIG", raising=False)
    path = tmp_path / "somewhere" / "config.yaml"
    _write(path, {"server": {"port": 1240}})
    config = load_config(path)
    assert config.data_dir == path.parent.resolve()
    assert config.logs_dir == path.parent.resolve() / "logs"


def test_a_data_dir_key_inside_the_file_is_ignored_with_a_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The copy-an-old-install trap: a config.yaml from another install names
    that install's data dir; it must not relocate this one."""
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    path = tmp_path / "new" / "config.yaml"
    _write(path, {"data_dir": str(tmp_path / "old-install"), "server": {"port": 1240}})
    config = load_config(path)
    assert config.data_dir == path.parent.resolve()
    assert "old-install" in _logged(capsys, caplog)


def test_sf_data_dir_beats_the_config_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "env-data"
    monkeypatch.setenv("SF_DATA_DIR", str(env_dir))
    path = tmp_path / "elsewhere" / "config.yaml"
    _write(path, {"data_dir": str(tmp_path / "in-file")})
    config = load_config(path)
    assert config.data_dir == env_dir.resolve()
    assert resolve_data_dir(path, explicit=True) == env_dir.resolve()


def test_apply_overrides_keeps_the_data_dir_and_refuses_to_edit_it(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    updated = apply_overrides(config, {"models.default_ctx": 16384})
    assert updated.data_dir == config.data_dir
    assert updated.models.default_ctx == 16384
    with pytest.raises(ConfigError) as excinfo:
        apply_overrides(config, {"data_dir": str(tmp_path / "other")})
    assert "SF_DATA_DIR" in excinfo.value.message


def test_a_saved_and_reloaded_config_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    config = Config(data_dir=tmp_path / "data")
    config.models.default_ctx = 12288
    config.ensure_dirs()
    config.save()
    reloaded = load_config(config.config_path)
    assert reloaded.data_dir == config.data_dir.resolve()
    assert reloaded.models.default_ctx == 12288


# ---------------------------------------------------------------------------
# empty file, fsync, .bak
# ---------------------------------------------------------------------------


def test_an_empty_config_yaml_is_treated_as_missing_and_regenerated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("SF_DATA_DIR", raising=False)
    path = tmp_path / "data" / "config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("", encoding="utf-8")
    config = load_config(path, create=True)
    assert "empty" in _logged(capsys, caplog)
    assert config.source_path == path
    on_disk = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(on_disk, dict) and "server" in on_disk, "regenerated with defaults"


def test_save_keeps_a_backup_of_the_previous_file(tmp_path: Path) -> None:
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    first = config.save()
    config.models.default_ctx = 4096
    config.save()
    backup = first.with_suffix(first.suffix + ".bak")
    assert backup.is_file()
    assert yaml.safe_load(backup.read_text(encoding="utf-8"))["models"]["default_ctx"] == 8192
    assert yaml.safe_load(first.read_text(encoding="utf-8"))["models"]["default_ctx"] == 4096
    assert not first.with_suffix(first.suffix + ".tmp").exists()


def test_save_fsyncs_before_the_rename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    synced: list[int] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    config = Config(data_dir=tmp_path / "data")
    config.ensure_dirs()
    config.save()
    assert synced, "the temp file must be fsynced before it replaces config.yaml"
