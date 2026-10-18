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
