"""Build identity (D70, item 9): ``build`` beside ``version`` wherever the version shows.

On 2026-09-22 the live server reported ``1.26-09-04-3`` from a checkout ~50
commits past that tag. ``version`` names the last *release*, and nothing named
the code that was actually running. ``build`` names the checkout: the short
commit SHA, ``-dirty`` when tracked files carry uncommitted changes, and
``unknown`` when there is nothing truthful to say. It is resolved once, at
startup, and never on a request path -- the watchdog polls ``/health`` every
few seconds.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import studioforge
from studioforge import build as build_module
from studioforge.api import app as app_module
from studioforge.api import mgmt_routes
from studioforge.build import GIT_TIMEOUT_S, UNKNOWN, build_id, checkout_root
from studioforge.mcp import management
from studioforge.mcp.management import build_management_mcp
from tests.unit.test_catalog_routes import app  # noqa: F401 - fixture
from tests.unit.test_companion import (  # noqa: F401 - fixture and helpers
    ServerHandle,
    _all_output,
    _invoke,
    cli_module,
    live_server,
)
from tests.unit.test_mcp import State, call, state  # noqa: F401 - fixture

SHA_SHAPE = re.compile(r"^[0-9a-f]{7,40}(-dirty)?$")


@pytest.fixture(autouse=True)
def _fresh_build_cache() -> Iterator[None]:
    build_id.cache_clear()
    yield
    build_id.cache_clear()


def _stub_git(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    rev_parse: str = "abc1234\n",
    status: str = "",
    fail: set[str] = frozenset(),  # type: ignore[assignment]
    raise_for: dict[str, BaseException] | None = None,
) -> list[tuple[list[str], dict[str, Any]]]:
    """A checkout at ``tmp_path`` whose ``git`` answers are scripted per subcommand."""
    # A linked worktree's ``.git`` is a FILE, not a directory; both must count.
    (tmp_path / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    monkeypatch.setattr(build_module, "checkout_root", lambda: tmp_path)
    monkeypatch.setattr(build_module.shutil, "which", lambda name: "git")
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), kwargs))
        sub = argv[1]
        if raise_for and sub in raise_for:
            raise raise_for[sub]
        if sub in fail:
            return subprocess.CompletedProcess(argv, 128, "", "fatal: not a git repository")
        return subprocess.CompletedProcess(argv, 0, rev_parse if sub == "rev-parse" else status, "")

    monkeypatch.setattr(build_module.subprocess, "run", run)
    return calls


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_unknown_outside_a_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wheel install has no ``.git`` beside the package: say so, never guess."""
    monkeypatch.setattr(build_module, "checkout_root", lambda: None)
    assert build_id() == UNKNOWN


def test_unknown_without_git_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(build_module, "checkout_root", lambda: tmp_path)
    monkeypatch.setattr(build_module.shutil, "which", lambda name: None)
    assert build_id() == UNKNOWN


def test_a_clean_tree_is_the_bare_short_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_git(monkeypatch, tmp_path, rev_parse="abc1234\n", status="")
    assert build_id() == "abc1234"
    assert [argv[1] for argv, _ in calls] == ["rev-parse", "status"]
    for _argv, kwargs in calls:
        assert kwargs["timeout"] == GIT_TIMEOUT_S, "every git call is bounded"
        assert kwargs["cwd"] == str(tmp_path), "git runs in the checkout, not the CWD"
        assert kwargs["check"] is False
    assert "--untracked-files=no" in calls[1][0], "a scratch file is not a modified build"


def test_uncommitted_tracked_changes_mark_the_build_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_git(monkeypatch, tmp_path, status=" M src/studioforge/core/manager.py\n")
    assert build_id() == "abc1234-dirty"


def test_a_git_timeout_is_unknown_and_never_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_git(
        monkeypatch,
        tmp_path,
        raise_for={"rev-parse": subprocess.TimeoutExpired(["git"], GIT_TIMEOUT_S)},
    )
    assert build_id() == UNKNOWN


def test_a_failing_git_is_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _stub_git(monkeypatch, tmp_path, fail={"rev-parse"})
    assert build_id() == UNKNOWN


def test_an_unreadable_status_keeps_the_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An index lock or a slow status cannot tell dirty from clean; the SHA is still true."""
    _stub_git(monkeypatch, tmp_path, fail={"status"})
    assert build_id() == "abc1234"


def test_the_build_is_resolved_once_per_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _stub_git(monkeypatch, tmp_path)
    assert build_id() == build_id() == build_id()
    assert len(calls) == 2, "rev-parse and status, once; every later read is the cache"


def test_checkout_root_wants_the_git_entry_at_the_package_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``git`` walks upward on its own; a wheel in a venv inside some other repo
    would otherwise report that repository's commit."""
    fake_module = tmp_path / "src" / "studioforge" / "build.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("", encoding="utf-8")
    monkeypatch.setattr(build_module, "__file__", str(fake_module))
    assert checkout_root() is None
    (tmp_path / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    assert checkout_root() == tmp_path


def test_the_real_checkout_identifies_itself() -> None:
    """This suite runs from a git checkout with git on PATH; the answer must be a SHA."""
    if checkout_root() is None or shutil.which("git") is None:
        pytest.skip("not running from a git checkout with git available")
    assert SHA_SHAPE.match(build_id()), build_id()


# ---------------------------------------------------------------------------
# Where it shows
# ---------------------------------------------------------------------------


def test_health_status_and_version_routes_carry_the_build(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app_module, "build_id", lambda: "abc1234-dirty")
    monkeypatch.setattr(mgmt_routes, "build_id", lambda: "abc1234-dirty")
    with TestClient(app) as http:
        health = http.get("/health").json()
        api_health = http.get("/api/health").json()
        version = http.get("/api/version").json()
        status = http.get("/api/status").json()
    assert health["version"] == studioforge.__version__
    assert health["build"] == "abc1234-dirty"
    assert api_health["build"] == "abc1234-dirty"
    assert version == {"version": studioforge.__version__, "build": "abc1234-dirty"}
    assert status["build"] == "abc1234-dirty", "sfctl status renders this payload"


def test_health_never_runs_git_on_the_request_path(
    app: Any,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The watchdog polls /health constantly; the startup call is the only git call."""
    _stub_git(monkeypatch, tmp_path)
    resolved = build_id()  # what create_app does once, on the startup path

    def no_git(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("git ran on a request path")

    monkeypatch.setattr(build_module.subprocess, "run", no_git)
    with TestClient(app) as http:
        assert http.get("/health").json()["build"] == resolved
        assert http.get("/api/version").json()["build"] == resolved


async def test_mcp_server_status_carries_the_build(
    state: State,  # noqa: F811 - the imported fixture
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(management, "build_id", lambda: "abc1234")
    server = build_management_mcp(state)
    payload = await call(server, "server_status")
    assert payload["build"] == "abc1234"
    assert "version" in payload


def test_sfctl_status_renders_the_build_row(
    monkeypatch: Any,
    live_server: ServerHandle,  # noqa: F811 - the imported fixture
) -> None:
    real_status = cli_module.StudioForgeClient.status

    async def status_with_build(self: Any) -> Any:
        payload = await real_status(self)
        payload["build"] = "abc1234-dirty"
        return payload

    monkeypatch.setattr(cli_module.StudioForgeClient, "status", status_with_build)
    result = _invoke(live_server, "status", env={"COLUMNS": "200"})
    assert result.exit_code == 0, _all_output(result)
    output = _all_output(result)
    assert "build" in output
    assert "abc1234-dirty" in output
    assert output.index("version") < output.index("abc1234-dirty"), "beside the version row"


def test_sfctl_status_json_passes_the_build_through(
    live_server: ServerHandle,  # noqa: F811 - the imported fixture
) -> None:
    """The machine-readable form is the server payload, so the field just arrives."""
    result = _invoke(live_server, "status", "--json")
    assert result.exit_code == 0, _all_output(result)
    assert '"build"' in result.output
