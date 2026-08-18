"""The systemd units in deploy/ must be loadable user units (WP17 F6).

The shipped files carried ``User=%i`` (only valid in system units; makes a user
unit fail to load) and pointed ExecStart at ``~/.local/bin/python``, which
does not exist. These tests parse the units the way systemd would and pin the
layout the deploy README documents.
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
UNITS = ["studioforge.service", "studioforge-watchdog.service"]


def _parse(name: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    # systemd allows repeated keys (Environment=); keep the last for these checks
    # but the file must still be well-formed INI.
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string((DEPLOY / name).read_text(encoding="utf-8"))
    return parser


@pytest.mark.parametrize("name", UNITS)
def test_units_are_user_units_without_user_directive(name: str) -> None:
    unit = _parse(name)
    assert unit["Install"]["WantedBy"] == "default.target", "a user unit targets default.target"
    assert "User" not in unit["Service"], "User= is only valid in system units"
    assert "Group" not in unit["Service"]


@pytest.mark.parametrize("name", UNITS)
def test_units_point_at_the_documented_layout(name: str) -> None:
    text = (DEPLOY / name).read_text(encoding="utf-8")
    unit = _parse(name)
    exec_start = unit["Service"]["ExecStart"]
    assert exec_start.startswith("%h/studioforge/.venv/bin/"), exec_start
    assert unit["Service"]["WorkingDirectory"] == "%h/studioforge"
    assert "Environment=SF_DATA_DIR=%h/studioforge/data" in text
    assert "github.com/studioforge" not in text, "no placeholder project URLs"


def test_watchdog_is_not_bound_to_the_gateway() -> None:
    unit = _parse("studioforge-watchdog.service")
    for key in ("BindsTo", "PartOf", "Requires"):
        assert key not in unit["Unit"], f"{key} would take the watchdog down with the gateway"
    assert unit["Service"]["Restart"] == "always"
    assert "--config %h/studioforge/data/config.yaml" in unit["Service"]["ExecStart"]


def test_deploy_readme_documents_the_recipe() -> None:
    text = (DEPLOY / "README.md").read_text(encoding="utf-8")
    for needle in ("systemctl --user enable --now", "loginctl enable-linger", "SF_DATA_DIR"):
        assert needle in text
