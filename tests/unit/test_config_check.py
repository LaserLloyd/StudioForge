"""``studioforge config check`` and a ``Config.save()`` that cannot write an unloadable file (D81).

The 2026-10-19 defect: the panel's Serving tab let ``server.cors_allow_credentials: true``
be saved next to ``server.cors_origins: ['*']``. ``Config`` rejects that pair when it loads,
so the next ``serve`` would have refused to start. Now nothing that calls ``save()`` can
write it, the rejection carries the validator's message (not pydantic's boilerplate), a
hand-edited file gets the same check from the CLI, and ``serve`` names the key and the
command that explains it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import studioforge.__main__ as entry
from studioforge.api.app import build_state, create_app
from studioforge.config import (
    Config,
    apply_overrides,
    check_config_file,
    explain_validation_error,
    load_config,
)
from studioforge.errors import ConfigError

runner = CliRunner()

BAD_PAIR = {"server": {"cors_origins": ["*"], "cors_allow_credentials": True}}


@pytest.fixture
def data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "data"
    monkeypatch.setenv("SF_DATA_DIR", str(path))
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    return path


@pytest.fixture
def good(data: Path, tmp_path: Path) -> Path:
    config = Config(data_dir=data, models={"dir": tmp_path / "models"}, logging={"level": "ERROR"})
    config.ensure_dirs()
    return config.save(data / "config.yaml")


def write_bad(path: Path) -> None:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["server"].update(BAD_PAIR["server"])
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


# --- save() refuses what load would refuse -----------------------------------


def test_save_refuses_a_config_mutated_into_an_unloadable_state(good: Path) -> None:
    config = load_config(good)  # may persist a freshly minted MCP pin: read the file after it
    before = good.read_text(encoding="utf-8")
    config.server.cors_origins = ["*"]  # in-place: pydantic does not re-validate assignment
    config.server.cors_allow_credentials = True
    with pytest.raises(ConfigError) as caught:
        config.save()
    message = caught.value.message
    assert caught.value.code == "invalid_config"
    assert "server.cors_allow_credentials=true requires explicit server.cors_origins" in message
    assert "nothing was written" in message and "input_value" not in message
    assert good.read_text(encoding="utf-8") == before, "the file on disk is untouched"
    assert not good.with_name("config.yaml.tmp").exists()


def test_save_still_writes_a_valid_config(good: Path) -> None:
    config = load_config(good)
    config.server.cors_origins = ["https://app.example"]
    config.server.cors_allow_credentials = True
    config.save()
    reread = load_config(good)
    assert reread.server.cors_allow_credentials and reread.server.cors_origins == [
        "https://app.example"
    ]


def test_apply_overrides_rejects_with_the_validators_words(good: Path) -> None:
    config = load_config(good)
    with pytest.raises(ConfigError) as caught:
        apply_overrides(
            config, {"server.cors_origins": ["*"], "server.cors_allow_credentials": True}
        )
    assert "nothing was saved" in caught.value.message
    assert "requires explicit server.cors_origins" in caught.value.message
    assert "pydantic.dev" not in caught.value.message and "input_value" not in caught.value.message


def test_explain_validation_error_is_one_key_per_problem() -> None:
    with pytest.raises(Exception) as caught:
        Config(server={"port": 99999}, planner={"headroom_fraction": 7})
    text = explain_validation_error(caught.value)
    assert "server.port:" in text and "pydantic.dev" not in text
    assert explain_validation_error(ValueError("plain")) == "plain"


def test_the_panel_save_path_rejects_the_pair_and_writes_nothing(good: Path) -> None:
    """``PATCH /api/config`` is the one handler under the panel, the MCP tool's twin."""
    config = load_config(good)
    app = create_app(config, state=build_state(config), start_background=False)
    before = good.read_text(encoding="utf-8")
    pin = {"X-MCP-Pin": str(config.mcp.pin)}
    with TestClient(app) as http:
        ok = http.patch("/api/config", json={"server.cors_origins": ["*"]}, headers=pin)
        assert ok.status_code == 200, ok.text
        after_wildcard = good.read_text(encoding="utf-8")
        bad = http.patch("/api/config", json={"server.cors_allow_credentials": True}, headers=pin)
    assert bad.status_code == 400, bad.text
    assert "requires explicit server.cors_origins" in bad.json()["error"]["message"]
    assert good.read_text(encoding="utf-8") == after_wildcard != before
    assert load_config(good).server.cors_allow_credentials is False


def test_a_drifted_in_memory_config_cannot_write_the_pair_either(good: Path) -> None:
    """The in-memory ``server`` section is not swapped live for restart-required keys, so an
    earlier save can leave memory and disk disagreeing; save() is the chokepoint that holds."""
    config = load_config(good)
    app = create_app(config, state=build_state(config), start_background=False)
    state_config = app.state.config
    pin = {"X-MCP-Pin": str(config.mcp.pin)}
    with TestClient(app) as http:
        first = http.patch("/api/config", json={"server.cors_allow_credentials": True}, headers=pin)
        assert first.status_code == 200, first.text
        # The wildcard now arrives while disk says credentials=true and memory still says false.
        assert state_config.server.cors_allow_credentials is False
        second = http.patch("/api/config", json={"server.cors_origins": ["*"]}, headers=pin)
        assert second.status_code == 200, second.text
    on_disk = load_config(good)  # whatever was written must load
    assert not (on_disk.server.cors_allow_credentials and "*" in on_disk.server.cors_origins)


# --- load error message ------------------------------------------------------


def test_load_config_error_names_the_key_and_the_command(good: Path) -> None:
    write_bad(good)
    good.with_name("config.yaml.bak").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_config(good)
    message = caught.value.message
    assert message.startswith(f"invalid configuration in {good}:")
    assert "server.cors_allow_credentials=true requires" in message
    assert "studioforge config check" in message and "config.yaml.bak" in message
    assert "pydantic.dev" not in message


# --- config check -------------------------------------------------------------


def test_config_check_ok(good: Path) -> None:
    result = runner.invoke(entry.app, ["config", "check", "--config", str(good)])
    assert result.exit_code == 0, result.output
    assert "OK: config.yaml loads cleanly" in result.stdout
    assert f"checking {good}" in result.stderr and "from --config" in result.stderr


def test_config_check_missing_file_is_fine(data: Path) -> None:
    result = runner.invoke(entry.app, ["config", "check", "--config", str(data / "config.yaml")])
    assert result.exit_code == 0 and "No config.yaml yet" in result.stdout
    assert not (data / "config.yaml").exists(), "check never creates the file"


def test_config_check_names_every_bad_key_and_exits_1(good: Path) -> None:
    write_bad(good)
    raw = yaml.safe_load(good.read_text(encoding="utf-8"))
    raw["planner"]["headroom_fraction"] = 7
    good.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    good.with_name("config.yaml.bak").write_text("{}", encoding="utf-8")
    before = good.read_bytes()
    result = runner.invoke(entry.app, ["config", "check", "-c", str(good)])
    assert result.exit_code == 1
    out = result.stdout
    assert "INVALID: `studioforge serve` will refuse to start (2 problem(s))" in out
    assert "planner.headroom_fraction:" in out
    assert "server.cors_allow_credentials=true requires explicit server.cors_origins" in out
    assert str(good) in out and "config.yaml.bak" in out
    assert good.read_bytes() == before, "check writes nothing"


def test_config_check_reports_yaml_syntax_and_non_mappings(data: Path) -> None:
    data.mkdir(parents=True)
    path = data / "config.yaml"
    path.write_text("server: [unclosed\n", encoding="utf-8")
    syntax = check_config_file(path)
    assert not syntax.ok and syntax.problems[0][0] == "(file)" and "YAML" in syntax.problems[0][1]
    path.write_text("- just\n- a list\n", encoding="utf-8")
    assert "mapping" in check_config_file(path).problems[0][1]
    path.write_text("", encoding="utf-8")
    assert check_config_file(path).ok, "an empty file loads as defaults (D31)"


def test_config_check_agrees_with_load_config(good: Path) -> None:
    assert check_config_file(good).ok
    load_config(good)
    write_bad(good)
    assert not check_config_file(good).ok
    with pytest.raises(ConfigError):
        load_config(good)


def test_config_without_a_subcommand_still_prints_the_effective_config(good: Path) -> None:
    shown = runner.invoke(entry.app, ["config", "--config", str(good)])
    assert shown.exit_code == 0 and "server:" in shown.stdout and "models:" in shown.stdout
    path_only = runner.invoke(entry.app, ["config", "--path", "--config", str(good)])
    assert path_only.stdout.strip() == str(good)


# --- serve ---------------------------------------------------------------------


def test_serve_with_an_invalid_config_says_which_key_and_what_to_run(good: Path) -> None:
    write_bad(good)
    result = runner.invoke(entry.app, ["serve", "--config", str(good), "--no-gui", "--no-watchdog"])
    assert result.exit_code == 2
    text = result.output
    assert "server.cors_allow_credentials=true requires explicit server.cors_origins" in text
    assert "studioforge config check" in text
    assert "Traceback" not in text


def test_config_group_option_reaches_the_check_subcommand(good: Path, tmp_path: Path) -> None:
    """`config --config X check` and `config check --config X` read the same file."""
    other = tmp_path / "other" / "config.yaml"
    other.parent.mkdir()
    other.write_text(yaml.safe_dump({"server": BAD_PAIR["server"]}), encoding="utf-8")
    before = runner.invoke(entry.app, ["config", "--config", str(other), "check"])
    after = runner.invoke(entry.app, ["config", "check", "--config", str(other)])
    assert before.exit_code == after.exit_code == 1
    assert str(other) in before.stderr and str(other) in after.stderr
