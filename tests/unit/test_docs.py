"""Docs-vs-reality checks for the user-facing documentation.

A wrong doc is a defect: every command spelled out in README/docs must exist
with the syntax shown. These checks caught two real ones -- README told users
to run ``sfctl config set server.url <url>`` (that command edits the *server's*
config over HTTP and takes ``key=value`` pairs, a double mismatch), and
OPENCLAW.md passed ``--url`` to ``sfctl servers add`` whose URL is positional.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPANION_SRC = REPO_ROOT / "packages" / "studioforge-companion" / "src"
if str(COMPANION_SRC) not in sys.path:
    sys.path.insert(0, str(COMPANION_SRC))

DOC_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "OPENCLAW.md",
    REPO_ROOT / "docs" / "LIMITATIONS.md",
    REPO_ROOT / "docs" / "COMPARISON.md",
]


def _doc_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOC_FILES)


def _typer_command_names(app) -> set[str]:  # type: ignore[no-untyped-def]
    names = set()
    for command in app.registered_commands:
        names.add(command.name or command.callback.__name__.replace("_", "-"))
    for group in app.registered_groups:
        names.add(group.name)
    return names


def test_readme_does_not_teach_the_wrong_setup_command() -> None:
    """``sfctl config set`` edits the SERVER's config over HTTP and takes
    key=value pairs; pointing the companion at a server is ``sfctl servers
    add`` (or ``config-local``). The old snippet failed twice over."""
    text = _doc_text()
    assert "sfctl config set server.url" not in text


def test_docs_use_the_real_servers_add_signature() -> None:
    """``sfctl servers add`` takes NAME then URL positionally; ``--url`` is a
    global option on the app, not an option of this command."""
    text = _doc_text()
    for line in text.splitlines():
        if "sfctl servers add" in line:
            assert "--url" not in line, f"servers add has no --url option: {line.strip()!r}"
            match = re.search(r"sfctl servers add\s+(\S+)\s+(\S+)", line)
            assert match, f"servers add needs NAME and URL: {line.strip()!r}"
            assert match.group(2).startswith("http"), (
                f"the second positional argument is the URL: {line.strip()!r}"
            )


def test_every_bat_launcher_is_documented_in_the_readme() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for bat in sorted(REPO_ROOT.glob("*.bat")):
        assert bat.name in readme, f"{bat.name} exists but the README never mentions it"


def test_every_sfctl_command_named_in_the_docs_exists() -> None:
    from studioforge_companion import cli as companion_cli

    known = _typer_command_names(companion_cli.app)
    used = set(re.findall(r"sfctl\s+([a-z][a-z-]*)", _doc_text()))
    unknown = used - known
    assert not unknown, f"docs name sfctl commands that do not exist: {sorted(unknown)}"


def test_every_studioforge_command_named_in_the_docs_exists() -> None:
    from studioforge import __main__ as main_cli

    known = _typer_command_names(main_cli.app)
    # "studioforge serve --open" etc.; capture the first word after the binary
    # on the SAME line (a line break separates a directory name from a command).
    used = set(re.findall(r"studioforge[ \t]+([a-z][a-z-]*)", _doc_text()))
    # Prose like "studioforge is", "studioforge instance" should not trip this:
    # only check words that look like commands (present in a code context is
    # hard to detect cheaply, so intersect against an allowlist of nouns).
    prose = {"is", "and", "the", "a", "an", "can", "does", "exists", "listens", "never"}
    unknown = used - known - prose
    assert not unknown, f"docs name studioforge commands that do not exist: {sorted(unknown)}"
