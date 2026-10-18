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


# --- container image ---------------------------------------------------------
# Since 2.0.0 the app binds 127.0.0.1 by default and refuses a network bind
# without an owner account or admin key. Inside a container a loopback bind is
# unreachable from the published ports (the 2026-10-18 review found every
# `-p` dead while the in-container HEALTHCHECK said "healthy"), so the image
# must bind the container's interfaces AND bootstrap a credential.

REPO = DEPLOY.parent


def test_the_image_binds_the_container_interfaces() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    for var in ("SF_SERVER__HOST=0.0.0.0", "SF_GUI__HOST=0.0.0.0", "SF_WATCHDOG__HOST=0.0.0.0"):
        assert var in dockerfile, f"{var} missing: published ports would be unreachable"


def test_the_entrypoint_bootstraps_the_owner_account() -> None:
    script = (DEPLOY / "compose" / "entrypoint.sh").read_text(encoding="utf-8")
    assert "studioforge users create" in script
    assert "--password-env SF_OWNER_PASSWORD" in script
    assert "owner-password.initial" in script, "the generated password must land on the volume"
    assert "umask 077" in script, "the generated password file must be owner-only"
    # The password is never echoed into the container log.
    for line in script.splitlines():
        if line.strip().startswith("log ") and "SF_OWNER_PASSWORD" in line:
            assert "$SF_OWNER_PASSWORD" not in line and "${SF_OWNER_PASSWORD" not in line, line
    # Only a fresh volume is touched: the app's own posture check decides, so an
    # existing account, an admin key or the legacy server.api_key all win.
    assert "has_admin_credential" in script


def test_the_compose_env_example_documents_the_owner_password() -> None:
    env = (DEPLOY / "compose" / "env.example").read_text(encoding="utf-8")
    assert "SF_OWNER_PASSWORD" in env
    docs = (REPO / "docs" / "deploy" / "docker.md").read_text(encoding="utf-8")
    assert "SF_OWNER_PASSWORD" in docs and "owner-password.initial" in docs
