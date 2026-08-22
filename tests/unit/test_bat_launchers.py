"""Static checks on the double-click ``.bat`` launchers in ``launchers/``.

These scripts are the only part of the product with no interpreter-level test
coverage, and cmd.exe fails silently in exactly the ways users cannot
diagnose. The checks here are the ones that have actually bitten:

* ``!VAR!`` delayed expansion inside a parenthesised block silently compares
  the literal text ``!VAR!`` unless ``EnableDelayedExpansion`` is on -- in
  ``Update StudioForge.bat`` that made the "git pull" step unreachable while
  printing "Not a git checkout" inside a genuine checkout.
* A launcher that does not anchor itself to the repo breaks when
  double-clicked from a shortcut or "Run as", because the working directory
  is then System32. Since the launchers moved into ``launchers/`` the anchor
  is the repo root *one level up*, resolved once into ``%REPO%``.
* An unquoted ``%PY%`` breaks as soon as the checkout lives under a path with
  a space in it, which on Windows is the common case (Desktop, OneDrive,
  "Program Files", any project folder with a space in its name).
* A ``.bat`` with bare LF line endings mis-parses ``goto :label``
  (``.gitattributes`` pins them to CRLF; a generator that forgets is caught here).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = REPO_ROOT / "launchers"

#: ``local-env.bat`` is the gitignored per-machine hook that the launchers
#: *call*, and ``local-env.example.bat`` is its tracked template; neither is a
#: launcher and neither has a launcher's obligations.
_NOT_A_LAUNCHER = {"local-env.bat", "local-env.example.bat"}
BAT_FILES = sorted(p for p in LAUNCHERS.glob("*.bat") if p.name.lower() not in _NOT_A_LAUNCHER)

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


def test_no_launcher_is_left_at_the_repo_root() -> None:
    """They all moved into launchers/; a straggler at the root would resolve
    ``%REPO%`` one level too high and run against the wrong checkout."""
    stragglers = [p.name for p in REPO_ROOT.glob("*.bat") if p.name.lower() not in _NOT_A_LAUNCHER]
    assert stragglers == []


def test_the_local_env_template_is_tracked_and_documents_the_data_dir() -> None:
    text = _bat_text(LAUNCHERS / "local-env.example.bat")
    assert 'set "SF_DATA_DIR=' in text
    assert "local-env.bat" in text, "the template must say where the real file goes"


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_use_crlf_line_endings(bat: Path) -> None:
    raw = bat.read_bytes()
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b""), f"{bat.name} has a bare LF line ending"


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
def test_launchers_anchor_to_the_repo_root(bat: Path) -> None:
    """Double-clicked from a shortcut, cwd is System32. Every launcher resolves
    the repo (one level above launchers/) into an absolute ``%REPO%`` and
    changes into it, so .venv, data and local-env.bat all resolve."""
    text = _bat_text(bat)
    assert 'for %%I in ("%~dp0..") do set "REPO=%%~fI"' in text
    assert 'cd /d "%REPO%"' in text


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_source_the_local_override_hook(bat: Path) -> None:
    """``local-env.bat`` is how a machine-specific ``SF_DATA_DIR`` reaches a
    launcher without editing a tracked file. A launcher that forgets to call
    it silently runs against a different data directory than its siblings --
    different config, different registry, an apparently empty library."""
    text = _bat_text(bat)
    assert 'if exist "%REPO%\\local-env.bat" call "%REPO%\\local-env.bat"' in text


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_launchers_default_the_data_dir_inside_the_repo(bat: Path) -> None:
    """One data-dir story (DECISIONS.md D25): SF_DATA_DIR, else <repo>/data --
    which is what ``config.default_data_dir()`` resolves to as well. With the
    launchers one level down, ``%~dp0data`` would now be launchers/data, which
    the CLI never looks at."""
    text = _bat_text(bat)
    assert 'set "SF_DATA_DIR=%REPO%\\data"' in text
    assert "%~dp0data" not in text


@pytest.mark.parametrize("bat", BAT_FILES, ids=lambda p: p.name)
def test_python_invocations_are_quoted_for_paths_with_spaces(bat: Path) -> None:
    """A checkout under a path with a space in it -- an unquoted %PY% cannot work."""
    text = _bat_text(bat)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%PY%"):
            pytest.fail(f"{bat.name}: unquoted %PY% invocation: {stripped!r}")
    assert '"%PY%"' in text or 'start "" /b "%PY%"' in text
