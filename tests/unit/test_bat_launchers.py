"""Static checks on the double-click ``.bat`` launchers.

These scripts are the only part of the product with no interpreter-level test
coverage, and cmd.exe fails silently in exactly the ways users cannot
diagnose. The checks here are the ones that have actually bitten:

* ``!VAR!`` delayed expansion inside a parenthesised block silently compares
  the literal text ``!VAR!`` unless ``EnableDelayedExpansion`` is on -- in
  ``Update StudioForge.bat`` that made the "git pull" step unreachable while
  printing "Not a git checkout" inside a genuine checkout.
* A launcher that does not ``cd /d "%~dp0"`` breaks when double-clicked from a
  shortcut or "Run as", because the working directory is then System32.
* An unquoted ``%PY%`` breaks as soon as the checkout lives under a path with
  a space in it, which on Windows is the common case (Desktop, OneDrive,
  "Program Files", any project folder with a space in its name).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``local-env.bat`` is the gitignored per-machine hook that the launchers
#: *call*; it is a settings file, not a launcher, and has none of their
#: obligations. It may or may not exist on any given checkout.
_NOT_A_LAUNCHER = {"local-env.bat"}
BAT_FILES = sorted(p for p in REPO_ROOT.glob("*.bat") if p.name.lower() not in _NOT_A_LAUNCHER)

#: ``!NAME!`` delayed-expansion reads. ``^!`` escapes and lone ``!`` in echoed
#: prose do not match; two ``!`` around a variable-ish name do.
_DELAYED_RE = re.compile(r"!([A-Za-z_][A-Za-z0-9_]*)!")


def _bat_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_the_launchers_exist() -> None:
    names = {p.name for p in BAT_FILES}
    assert {
        "Start StudioForge.bat",
        "Open StudioForge GUI.bat",
        "Update StudioForge.bat",
        "StudioForge Autostart.bat",
        "StudioForge Tray.bat",
    } <= names


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_delayed_expansion_is_enabled_wherever_bang_vars_are_used(bat: Path) -> None:
    """``!ERRORLEVEL!`` without EnableDelayedExpansion is a literal-string
    comparison that can never be true -- the branch it guards is dead code."""
    text = _bat_text(bat)
    used = _DELAYED_RE.findall(text)
    if not used:
        return
    assert re.search(r"setlocal\b[^\n]*EnableDelayedExpansion", text, re.IGNORECASE), (
        f"{bat.name} reads {sorted(set(used))} with !...! delayed expansion but never "
        "enables it: cmd compares the literal text '!VAR!' instead of the value, so the "
        "guarded branch silently never runs. Add EnableDelayedExpansion to setlocal."
    )


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_anchor_to_their_own_directory(bat: Path) -> None:
    """Double-clicked from a shortcut, cwd is System32; every launcher must
    ``cd /d "%~dp0"`` so relative paths (.venv, data) resolve."""
    assert 'cd /d "%~dp0"' in _bat_text(bat)


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_source_the_local_override_hook(bat: Path) -> None:
    """``local-env.bat`` is how a machine-specific ``SF_DATA_DIR`` reaches a
    launcher without editing a tracked file. A launcher that forgets to call
    it silently runs against a different data directory than its siblings --
    different config, different registry, an apparently empty library."""
    assert 'if exist "%~dp0local-env.bat" call "%~dp0local-env.bat"' in _bat_text(bat)


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_default_the_data_dir_inside_the_repo(bat: Path) -> None:
    """One data-dir story (DECISIONS.md D25): SF_DATA_DIR, else <repo>/data --
    which is what ``config.default_data_dir()`` resolves to as well. The old
    ``%~dp0..\\data`` put it beside the checkout, where the CLI never looked."""
    text = _bat_text(bat)
    assert 'set "SF_DATA_DIR=%~dp0data"' in text
    assert "%~dp0..\\data" not in text


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_python_invocations_are_quoted_for_paths_with_spaces(bat: Path) -> None:
    """A checkout under a path with a space in it -- an unquoted %PY% cannot work."""
    text = _bat_text(bat)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%PY%"):
            pytest.fail(f"{bat.name}: unquoted %PY% invocation: {stripped!r}")
