"""Unit-suite fixtures that keep every test off the developer's real machine.

The secret store (``studioforge.secret_store``) prefers the OS keyring when the
optional ``keyring`` package is installed. A test that creates an account or
saves ``hf.token`` must never write into the real Credential Manager / Keychain /
Secret Service, so the whole suite pins the file backend (under each test's own
data directory) unless a test deliberately injects a fake keyring.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _secrets_use_files(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SF_SECRETS_BACKEND", "file")


@pytest.fixture(autouse=True)
def _plain_wide_cli_help(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render typer/rich help and errors as plain, unwrapped text.

    typer forces ANSI output when ``GITHUB_ACTIONS``/``FORCE_COLOR``/``PY_COLORS`` is set
    and boxes errors at the terminal width, which splits ``--gpu`` across escape codes
    and wrapped lines. Tests that look for option names must see the same text on a
    developer's machine and on a CI runner.
    """
    try:
        from typer import rich_utils
    except ImportError:  # typer's rich extra is optional
        return
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", False, raising=False)
    monkeypatch.setattr(rich_utils, "MAX_WIDTH", 200, raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
