"""The Linux installer's systemd units stop cleanly and the installer can be re-run.

The server exits 130 (SIGINT) / 143 (SIGTERM) on a clean stop, which systemd counts as a
failure unless the unit says otherwise: every ``systemctl --user stop`` left
``studioforge.service`` in the ``failed`` state and the next ``--uninstall`` or reinstall
tripped over it. The units are embedded in ``install.sh`` (it is published as a
``curl | bash`` one-liner, so it cannot depend on sibling template files); these tests read
the templates out of the script without running a shell, plus the hand-install units in
``deploy/``.
"""

from __future__ import annotations

import configparser
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
INSTALL_SH = DEPLOY / "linux" / "install.sh"

#: What the installer's shell variables expand to; only the structure matters here.
_PLACEHOLDERS = {
    "${DATA_DIR}": "/data",
    "${VENV}": "/venv",
    "${REPO}": "LaserLloyd/StudioForge",
    "${wanted}": "default.target",
    "${user_lines}": "",
}


def _script() -> str:
    return INSTALL_SH.read_text(encoding="utf-8")


def _function_body(name: str) -> str:
    text = _script()
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}\n", text, re.S | re.M)
    assert match, f"{name}() not found in install.sh"
    return match.group(1)


def _heredocs(body: str) -> list[str]:
    return re.findall(r"cat <<EOF\n(.*?)\nEOF\n", body, re.S)


def _parse(unit_text: str) -> configparser.ConfigParser:
    for key, value in _PLACEHOLDERS.items():
        unit_text = unit_text.replace(key, value)
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(unit_text)
    return parser


def _rendered_units() -> list[tuple[str, configparser.ConfigParser]]:
    units = []
    for func in ("render_gateway_unit", "render_watchdog_unit"):
        for index, doc in enumerate(_heredocs(_function_body(func))):
            units.append((f"{func}[{index}]", _parse(doc)))
    return units


def test_the_templates_are_found() -> None:
    names = [name for name, _ in _rendered_units()]
    # gateway: system + user; watchdog: one template shared by both scopes
    assert names == ["render_gateway_unit[0]", "render_gateway_unit[1]", "render_watchdog_unit[0]"]


@pytest.mark.parametrize(
    ("name", "unit"), _rendered_units(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_every_rendered_unit_treats_a_signal_exit_as_success(
    name: str, unit: configparser.ConfigParser
) -> None:
    assert unit["Service"]["SuccessExitStatus"] == "130 143", name


@pytest.mark.parametrize("name", ["studioforge.service", "studioforge-watchdog.service"])
def test_the_hand_install_units_carry_the_same_directive(name: str) -> None:
    unit = configparser.ConfigParser(strict=False, interpolation=None)
    unit.optionxform = str  # type: ignore[assignment,method-assign]
    unit.read_string((DEPLOY / name).read_text(encoding="utf-8"))
    assert unit["Service"]["SuccessExitStatus"] == "130 143"


def test_uninstall_resets_the_failed_state_before_disabling() -> None:
    text = _script()
    start = text.index('if [ "$MODE" = uninstall ]')
    block = text[start : text.index("exit 0", start)]
    assert "reset-failed" in block
    assert block.index("reset-failed") < block.index("disable --now"), (
        "reset-failed must run before the units are disabled"
    )
    # ...and once the files are gone, so a unit from an older installer does not linger as failed.
    assert block.rindex("reset-failed") > block.index("daemon-reload")


def test_a_rerun_stops_the_live_units_and_clears_failed_state_before_replacing_the_venv() -> None:
    text = _script()
    assert text.index("stop $WAS_ACTIVE") < text.index('"$UV" venv'), (
        "the running server executes from the venv that `uv venv --clear` deletes"
    )
    # the units are written atomically (temp file + rename), never truncated in place
    assert 'mv -f "$tmp" "$path"' in text
    # restart, not start: correct whether the unit is stopped, running or failed
    assert "reset-failed studioforge.service studioforge-watchdog.service" in text
    assert "restart studioforge.service studioforge-watchdog.service" in text


# --- the script itself, where a POSIX shell and Linux are really available -----

needs_linux_bash = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None,
    reason="install.sh refuses non-Linux hosts; run on the Linux CI leg",
)


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
    }
    (tmp_path / "home").mkdir(exist_ok=True)
    return subprocess.run(
        ["bash", str(INSTALL_SH), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


@needs_linux_bash
def test_help_is_green() -> None:
    result = subprocess.run(
        ["bash", str(INSTALL_SH), "--help"], capture_output=True, text=True, timeout=30, check=False
    )
    assert result.returncode == 0
    assert "--uninstall" in result.stdout


@needs_linux_bash
def test_dry_run_changes_nothing_and_repeats_identically(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    first = _run(tmp_path, "--dry-run", "--from", str(source))
    second = _run(tmp_path, "--dry-run", "--from", str(source))
    assert first.returncode == 0, first.stderr
    assert "SuccessExitStatus" not in first.stdout  # units are written, not echoed
    assert first.stdout == second.stdout, "a dry run is deterministic"
    assert not (tmp_path / "data").exists() and not (tmp_path / "config").exists()


@needs_linux_bash
def test_uninstall_dry_run_is_green(tmp_path: Path) -> None:
    result = _run(tmp_path, "--uninstall", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "KEPT your data" in result.stdout
