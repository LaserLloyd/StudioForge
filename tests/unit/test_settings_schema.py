"""The settings schema registry covers every config key, and its outputs do not drift.

The registry (``studioforge.settings_schema``) is derived from the pydantic
``Config`` model, so the claim worth testing is the one a hand-written list
cannot make: *every* key the model persists is in it, with a description, a
level, an env override that actually works, and the committed
``docs/reference/settings.md`` is exactly what it renders (the P5 docs-drift
gate).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from studioforge import settings_schema
from studioforge.api.app import build_state, create_app
from studioforge.config import (
    CONFIG_SCHEMA_VERSION,
    RESTART_REQUIRED_KEYS,
    SECRET_CONFIG_PATHS,
    Config,
    load_config,
)
from studioforge.settings_schema import BASIC_KEYS, build_registry, render_markdown

REPO_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_MD = REPO_ROOT / "docs" / "reference" / "settings.md"


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith("SF_"):
            monkeypatch.delenv(name)


def _flatten(tree: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(tree, dict) or not tree:
        return {prefix: tree}
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        flat.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return flat


def _persisted_keys(tmp_path: Path) -> set[str]:
    """Top-level or section keys config.yaml can hold, to the depth the registry stops at."""
    dumped = Config(data_dir=tmp_path).to_yaml_dict()
    keys: set[str] = set()
    for top, value in dumped.items():
        if isinstance(value, dict):
            keys.update(f"{top}.{sub}" for sub in value)
        else:
            keys.add(top)
    return keys


def test_registry_covers_exactly_the_keys_config_yaml_holds(tmp_path: Path) -> None:
    registry = {s.key for s in build_registry()}
    persisted = _persisted_keys(tmp_path)
    assert registry == persisted, (
        f"missing from registry: {sorted(persisted - registry)}; "
        f"not a persisted key: {sorted(registry - persisted)}"
    )
    assert len(registry) >= 100


def test_every_key_has_the_documented_metadata() -> None:
    for s in build_registry():
        assert s.description.strip(), f"{s.key}: no description (add a comment or a fallback)"
        assert s.level in {"basic", "advanced"}
        assert s.type
        assert s.env.startswith("SF_") and s.env == s.env.upper()
        assert isinstance(s.restart_required, bool)


def test_basic_keys_are_exactly_the_agreed_set() -> None:
    registry = {s.key: s for s in build_registry()}
    assert set(BASIC_KEYS) <= set(registry), sorted(set(BASIC_KEYS) - set(registry))
    basic = {k for k, s in registry.items() if s.level == "basic"}
    assert basic == set(BASIC_KEYS)
    for required in (
        "server.host",
        "server.port",
        "server.api_key",
        "gui.port",
        "models.dir",
        "engine.pinned_tag",
        "update.repo",
    ):
        assert registry[required].level == "basic", required


def test_defaults_match_the_shipped_model(tmp_path: Path) -> None:
    shipped = _flatten(Config(data_dir=tmp_path).to_yaml_dict())
    for s in build_registry():
        if s.key in shipped:
            assert s.default == shipped[s.key], s.key
        else:  # a map/list key flattened deeper in the dump
            sub = {k: v for k, v in shipped.items() if k.startswith(s.key + ".")}
            assert sub or s.default in ({}, []), s.key


def test_constraints_and_choices_come_from_the_model() -> None:
    registry = {s.key: s for s in build_registry()}
    assert registry["server.port"].constraints == {"ge": 1, "le": 65535}
    assert registry["planner.on_insufficient"].choices == ["evict", "reject"]
    assert registry["models.default_parallel"].choices == ["auto"]
    assert registry["server.api_key"].constraints == {"min_length": 6}
    assert registry["logging.file_backups"].constraints == {"ge": 1, "le": 100}


def test_secret_flags_match_the_redaction_list() -> None:
    secrets = {s.key for s in build_registry() if s.secret}
    assert secrets == {f"{a}.{b}" for a, b in SECRET_CONFIG_PATHS}


def test_restart_flag_covers_the_existing_restart_list() -> None:
    registry = {s.key: s for s in build_registry()}
    for key in RESTART_REQUIRED_KEYS - {"data_dir"}:
        assert registry[key].restart_required, key
    assert registry["engine.pinned_tag"].restart_required
    assert not registry["models.default_ctx"].restart_required


@pytest.mark.parametrize(
    ("key", "attr_check"),
    [
        ("server.port", lambda c: c.server.port == 1299),
        ("logging.json", lambda c: c.logging.json_logs is True),
        ("models.default_ctx", lambda c: c.models.default_ctx == 1299),
    ],
)
def test_documented_env_override_name_really_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: str, attr_check: Any
) -> None:
    env = next(s.env for s in build_registry() if s.key == key)
    assert env == "SF_" + key.replace(".", "__").upper()
    monkeypatch.setenv(env, "true" if key == "logging.json" else "1299")
    assert attr_check(Config(data_dir=tmp_path))


def test_schema_version_defaults_old_files_and_round_trips(tmp_path: Path) -> None:
    assert Config(data_dir=tmp_path).schema_version == CONFIG_SCHEMA_VERSION == 1
    assert Config(data_dir=tmp_path).to_yaml_dict()["schema"] == 1
    # A config.yaml written before the key existed loads unchanged.
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"server": {"port": 4321}}), encoding="utf-8")
    old = load_config(path)
    assert old.server.port == 4321 and old.schema_version == 1
    # And a file that carries it keeps it through save().
    path.write_text(yaml.safe_dump({"schema": 1, "gui": {"port": 8099}}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.gui.port == 8099
    saved = yaml.safe_load(loaded.save(tmp_path / "out.yaml").read_text(encoding="utf-8"))
    assert saved["schema"] == 1


def test_a_stray_schema_environment_variable_is_not_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The attribute is `schema_version` precisely so a bare SCHEMA env var cannot reach it."""
    monkeypatch.setenv("SCHEMA", "public")
    assert Config(data_dir=tmp_path).schema_version == 1


def test_endpoint_serves_the_registry(tmp_path: Path) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    app = create_app(config, state=build_state(config), start_background=False)
    with TestClient(app) as http:
        response = http.get("/api/settings/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == CONFIG_SCHEMA_VERSION and body["product"] == "StudioForge"
    assert body["count"] == len(body["settings"]) == len(build_registry())
    port = next(s for s in body["settings"] if s["key"] == "server.port")
    assert port["default"] == 1234 and port["env"] == "SF_SERVER__PORT"
    assert port["level"] == "basic" and port["restart_required"] is True


def test_settings_reference_is_generated_and_committed() -> None:
    """The docs-drift gate: docs/reference/settings.md is exactly what the registry renders."""
    committed = SETTINGS_MD.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert committed == render_markdown(), (
        "docs/reference/settings.md is stale -- run `uv run python scripts/gen_docs.py`"
    )


def test_markdown_lists_every_key_once() -> None:
    text = render_markdown()
    for s in build_registry():
        assert text.count(f"| `{s.key}`") == 1, s.key


def test_missing_descriptions_is_empty() -> None:
    assert settings_schema.missing_descriptions() == []


def test_endpoint_has_the_same_gate_as_other_api_reads(tmp_path: Path) -> None:
    """Open without a key (like every GET /api read); the key is required once one is set."""
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234, "api_key": "sf-test-key-123456"},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    app = create_app(config, state=build_state(config), start_background=False)
    with TestClient(app) as http:
        assert http.get("/api/settings/schema").status_code == 401
        assert http.get("/api/config").status_code == 401  # the read it must match
        ok = http.get(
            "/api/settings/schema", headers={"Authorization": "Bearer sf-test-key-123456"}
        )
    assert ok.status_code == 200 and ok.json()["count"] > 100
