"""Unit-suite fixtures that keep every test off the developer's real machine.

The secret store (``studioforge.secret_store``) prefers the OS keyring when the
optional ``keyring`` package is installed. A test that creates an account or
saves ``hf.token`` must never write into the real Credential Manager / Keychain /
Secret Service, so the whole suite pins the file backend (under each test's own
data directory) unless a test deliberately injects a fake keyring.
"""

from __future__ import annotations

from collections.abc import Iterator

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


@pytest.fixture(autouse=True)
def _no_install_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI must never discover the developer's real install from a test (D81).

    ``studioforge.install_locator`` reads the installed Startup launcher / systemd unit
    to find the live data directory when no ``--config``/``SF_DATA_DIR`` is given. A test
    that forgets to set either would otherwise act on that real install, so discovery is
    off for the whole suite; the tests of the locator turn it on against fakes.
    """
    monkeypatch.setenv("SF_NO_INSTALL_DISCOVERY", "1")


@pytest.fixture
def restore_logging() -> Iterator[None]:
    """Undo the global logging setup a CLI command leaves behind.

    Every CLI command loads its config through ``__main__._load``, which calls
    ``configure_logging`` with the scratch config's level (``ERROR`` in these tests): the
    root logger's level, handlers, structlog's filtering level and the open log-file handler
    are process-wide, so a later test that expects to see a WARNING (``test_config_hardening``'s
    ``_logged``) would capture nothing. Tests that drive the CLI opt in with
    ``pytestmark = pytest.mark.usefixtures("restore_logging")``.
    """
    yield
    from studioforge.logging import configure_logging

    configure_logging("INFO")


@pytest.fixture(autouse=True)
def _testclient_host_is_this_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clients' default hosts are this server.

    Starlette's ``TestClient`` sends ``Host: testserver``; NiceGUI's ``User`` sends ``test``.

    ``studioforge.api.auth.host_header_trusted`` refuses any ``Host`` that is not a
    loopback literal or a configured one (the DNS-rebinding guard), and a unit test that
    simulates "a caller on this machine" with ``TestClient(app, client=("127.0.0.1", n))``
    would otherwise be refused for its default ``testserver`` host. Tests of the guard
    itself send an explicit foreign ``Host`` (``evil.test``), which still fails.
    """
    from studioforge.api import auth

    original = auth.host_header_trusted

    def patched(request: object, config: object) -> bool:
        headers = getattr(request, "headers", None)
        if headers is not None and (headers.get("host") or "").strip().lower() in {
            "testserver",
            "test",
        }:
            return True
        return original(request, config)  # type: ignore[arg-type]

    monkeypatch.setattr(auth, "host_header_trusted", patched)
