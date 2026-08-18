"""CLI failure modes a new user hits first: bad config, tray on Linux (WP17 F7/F8)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from studioforge import __main__ as main_cli


def _invoke(*args: str, env: dict[str, str] | None = None) -> Any:
    return CliRunner().invoke(main_cli.app, list(args), env=env, catch_exceptions=False)


def test_corrupt_config_yaml_is_one_readable_line(tmp_path: Path) -> None:
    """A stray tab in config.yaml must not greet the user with a traceback."""
    bad = tmp_path / "config.yaml"
    bad.write_text("server:\n\tport: 1234\n", encoding="utf-8")  # a tab is invalid YAML here
    result = _invoke("config", "--config", str(bad))
    assert result.exit_code == 2, result.output
    assert "not valid YAML" in result.output
    assert "Traceback" not in result.output
    assert "SF_DEBUG=1" in result.output


def test_corrupt_config_yaml_tracebacks_only_on_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.errors import ConfigError

    bad = tmp_path / "config.yaml"
    bad.write_text("server:\n\tport: 1234\n", encoding="utf-8")
    monkeypatch.setenv("SF_DEBUG", "1")
    with pytest.raises(ConfigError):
        CliRunner().invoke(
            main_cli.app, ["config", "--config", str(bad)], catch_exceptions=False
        )


def test_tray_refuses_off_windows_with_a_pointer_to_systemd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_cli.os, "name", "posix")
    result = _invoke("tray")
    assert result.exit_code == 2
    assert "Windows-only" in result.output
    assert "deploy/" in result.output
