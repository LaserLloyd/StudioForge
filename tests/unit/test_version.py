"""The version is spelled three ways; this pins all three to the same date.

StudioForge is calendar-versioned. The display string -- what ``/health``, the
GUI footer, the MCP ``server_status`` tool and the GitHub User-Agent all report
-- is ``1.26-08-28``: a major, then the release date. PEP 440 has no way to
spell a hyphenated date, so the two ``pyproject.toml`` files carry the same
date as ``1.26.8.28``, which is what ``pip``/``uv`` see in the wheel metadata.
Release tags are ``v1.26-08-28``.

Three spellings is two chances to drift, and the drift is silent: the updater
compares ``/health``'s version against a GitHub tag, so a mismatch shows up as
"already running the latest" against a release that is genuinely newer, or as
a rollback of a perfectly healthy update. Hence this file. Nothing in the app
parses ``__version__`` with ``packaging`` -- it could not, the string is not
PEP 440 -- and ``_version_key`` is the junk-tolerant parser that has to read
both spellings as the same number.
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

#: The human/display version, and the tag with its ``v``.
DISPLAY_VERSION = "1.26-08-28"
#: The same date, as the only thing PEP 440 will accept.
PEP440_VERSION = "1.26.8.28"


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
