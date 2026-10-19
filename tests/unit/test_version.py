"""The version is spelled in several places; this pins them to one string.

StudioForge is semantically versioned since 2.0.0 (the prosumer refactor). The
display string -- what ``/health``, the GUI footer, the MCP ``server_status``
tool and the GitHub User-Agent all report -- is ``2.0.0``, the two
``pyproject.toml`` files carry the same string, and release tags are ``v2.0.0``.
Releases before 2.0.0 were calendar-versioned (``1.26-09-23`` on display,
``1.26.9.23`` in PEP 440 metadata), and the updater's ``_version_key`` still
reads both spellings so an install on either scheme orders the other correctly.

Several spellings is several chances to drift, and the drift is silent: the
updater compares ``/health``'s version against a GitHub tag, so a mismatch
shows up as "already running the latest" against a release that is genuinely
newer, or as a rollback of a perfectly healthy update. Hence this file.

The two user-facing docs that quote the version verbatim (the README status
line and the OPENCLAW-SETUP ``/health`` sample) are pinned here too: on
2026-09-22 a live server reported a version ~50 commits stale, and the docs
had drifted the same way.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import studioforge
from studioforge.core.updater import _version_key

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPANION_SRC = REPO_ROOT / "packages" / "studioforge-companion" / "src"
if str(COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(COMPANION_SRC))

import studioforge_companion  # noqa: E402

#: The human/display version, and the tag with its ``v``. SemVer since 2.0.0.
DISPLAY_VERSION = "2.2.0"
#: The package metadata; identical to the display version under SemVer.
PEP440_VERSION = "2.2.0"
#: The release before this one (the last calendar-versioned tag); the new
#: version must sort above it across the two schemes.
PREVIOUS_RELEASE = "1.26-09-23"
#: Every earlier calendar release must sort below 2.0.0 too, or the updater
#: would offer an old tag as an "update" to a 2.x install.
OLDER_CALENDAR_RELEASES = ("1.26-09-04-3", "1.26-08-23", "1.26.8.23", "v1.26-09-23")


def _pyproject_version(path: Path) -> str:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def test_display_version() -> None:
    assert studioforge.__version__ == DISPLAY_VERSION


def test_companion_matches_the_server() -> None:
    """The companion ships in the same release and carries the same version."""
    assert studioforge_companion.__version__ == studioforge.__version__


def test_package_metadata_carries_the_same_date() -> None:
    assert _pyproject_version(REPO_ROOT / "pyproject.toml") == PEP440_VERSION
    assert (
        _pyproject_version(REPO_ROOT / "packages" / "studioforge-companion" / "pyproject.toml")
        == PEP440_VERSION
    )


def test_the_two_spellings_are_one_version() -> None:
    """What keeps the updater from mistaking the running build for an update."""
    assert _version_key(studioforge.__version__) == _version_key(PEP440_VERSION)
    assert _version_key(f"v{DISPLAY_VERSION}") == _version_key(DISPLAY_VERSION)


def test_the_new_version_is_newer_than_the_old_one() -> None:
    assert _version_key(studioforge.__version__) > _version_key("0.2.0")
    assert _version_key(studioforge.__version__) > _version_key(PREVIOUS_RELEASE), (
        "the updater would offer the previous release as an update"
    )


def test_semver_sorts_above_every_calendar_release() -> None:
    """2.0.0 follows the calendar-versioned 1.26-* line in the updater's ordering.

    ``_version_key`` reads ``1.26-09-23`` as ``(1, 26, 9, 23)`` and ``2.0.0`` as
    ``(2, 0, 0)``; the major number is what makes the scheme change safe, so a
    2.x server never sees an old calendar tag as newer, and a 1.26 server sees
    the 2.0.0 tag as the update it is.
    """
    new = _version_key(studioforge.__version__)
    assert new == (2, 2, 0)
    for old in OLDER_CALENDAR_RELEASES + (PREVIOUS_RELEASE,):
        assert new > _version_key(old), old
    assert _version_key("v2.2.0") == new
    assert _version_key("2.2.1") > new > _version_key("1.99-12-31")


def test_the_docs_quote_the_current_version() -> None:
    """The README status line and the setup guide's ``/health`` sample."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    setup = (REPO_ROOT / "docs" / "OPENCLAW-SETUP.md").read_text(encoding="utf-8")
    assert f"**Status:** v{DISPLAY_VERSION}." in readme
    assert f"`{DISPLAY_VERSION}`" in readme
    assert f"`v{DISPLAY_VERSION}`" in readme
    assert f'"version":"{DISPLAY_VERSION}"' in setup
    for text, name in ((readme, "README.md"), (setup, "OPENCLAW-SETUP.md")):
        assert PREVIOUS_RELEASE not in text, f"{name} still quotes the previous release"
