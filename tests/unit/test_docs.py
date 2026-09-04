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
import tomllib
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


#: The per-machine hook and its tracked template are not launchers (see
#: test_bat_launchers.py); the README documents the template by its own row.
_NOT_A_LAUNCHER = {"local-env.bat", "local-env.example.bat"}


def test_every_bat_launcher_is_documented_in_the_readme() -> None:
    """The launchers live in ``launchers/`` (not the repo root) since the
    public-repo reshuffle; globbing the root here would pass vacuously, which
    is exactly what happened once. Guard against an empty glob so a future
    move cannot hollow the check out again."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    launchers = sorted(
        p for p in (REPO_ROOT / "launchers").glob("*.bat") if p.name.lower() not in _NOT_A_LAUNCHER
    )
    assert launchers, "no launchers found under launchers/ -- did they move again?"
    for bat in launchers:
        assert bat.name in readme, f"{bat.name} exists but the README never mentions it"
    assert "launchers\\local-env.example.bat" in readme, (
        "the local-env template must stay documented with its launchers/ path"
    )


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


def test_the_repository_carries_a_licence_and_both_packages_declare_it() -> None:
    """A public repo with no LICENSE is 'all rights reserved'.

    That contradicted CONTRIBUTING inviting contributions, and left every fork
    and every published wheel legally unusable. The three artefacts have to
    agree: the file, the server's metadata, and the companion's -- the
    companion is built as its own wheel, so a root-only LICENSE would ship a
    licence-less package.
    """
    licence = REPO_ROOT / "LICENSE"
    assert licence.is_file(), "no LICENSE file at the repository root"
    text = licence.read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Copyright (c)" in text

    for pyproject in (
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "packages" / "studioforge-companion" / "pyproject.toml",
    ):
        project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
        assert project.get("license") == "MIT", f"{pyproject} declares no licence"
        assert project.get("license-files"), f"{pyproject} ships no licence file"
        # PEP 639: an SPDX expression and a trove classifier together is a
        # build error, so the classifier must stay out.
        assert not [c for c in project.get("classifiers", []) if c.startswith("License ::")], (
            f"{pyproject} mixes a licence classifier with the SPDX expression"
        )

    companion_licence = REPO_ROOT / "packages" / "studioforge-companion" / "LICENSE"
    assert companion_licence.is_file(), "the companion wheel would ship without a LICENSE"
    assert companion_licence.read_text(encoding="utf-8") == text

    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "all rights reserved" not in readme.lower()


def test_the_documented_mcp_tool_counts_are_the_real_ones() -> None:
    """A tool count in prose is a fact that rots silently.

    The docs said "the app's 14 management tools" while nineteen were
    registered, and an audit had to reconcile the total three separate ways to
    find out. Counted from the servers themselves, so the number cannot drift
    again without this failing.
    """
    import asyncio

    from studioforge.mcp.management import build_management_mcp

    class _State:
        config = None
        registry = None
        supervisor = None
        manager = None
        engine_manager = None
        downloader = None

    management = asyncio.run(build_management_mcp(_State()).list_tools())
    assert len(management) == 20, [tool.name for tool in management]

    setup = (REPO_ROOT / "src" / "studioforge" / "api" / "mgmt_routes.py").read_text(
        encoding="utf-8"
    )
    assert f"app's {len(management)} management tools" in setup
    # 20 + 10 watchdog tools. The total is what an operator reads in the README.
    assert "30 MCP tools" in (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_recovery_prefix_rule_is_described_as_the_allowlist_it_is() -> None:
    """It is an allowlist, not collision detection: the watchdog's `health`
    collides with nothing and is still exposed as `recovery_health`. The docs
    said only colliding names were prefixed, which sends an agent author
    looking for a collision that is not there."""
    from studioforge_companion import mcp_proxy

    assert "health" not in mcp_proxy.WATCHDOG_UNPREFIXED
    for text in (mcp_proxy.__doc__ or "", _doc_text()):
        assert "only colliding" not in text.lower()
    proxy_source = (
        REPO_ROOT
        / "packages"
        / "studioforge-companion"
        / "src"
        / "studioforge_companion"
        / "mcp_proxy.py"
    ).read_text(encoding="utf-8")
    assert "allowlist" in proxy_source.lower()


def test_no_doc_still_says_the_licence_is_unchosen_or_calls_the_bench_gauntlet() -> None:
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(REPO_ROOT.glob("*.md")) + sorted((REPO_ROOT / "docs").glob("*.md"))
    ).lower()
    assert "all rights reserved" not in text
    assert "gauntlet" not in text, "the benchmark suite is called CrucibleForge now"


# ---------------------------------------------------------------------------
# The prompt-prefix cache is documented with the signals that can show it (D54)
# ---------------------------------------------------------------------------


def test_engine_features_doc_names_the_real_cache_surfaces() -> None:
    """The grant lives on the instance, not on a bare GET /api/models row, and
    a cache hit shows in ``timings.cache_n`` -- ``prompt_n`` is the work done,
    not a proxy for the hit."""
    text = (REPO_ROOT / "docs" / "ENGINE-FEATURES.md").read_text(encoding="utf-8")
    assert "/api/status" in text
    assert "timings.cache_n" in text
    assert "is on its row in `GET /api/models`" not in text
    assert "much smaller `timings.prompt_n`" not in text
    assert "## Prompt-prefix reuse" in text
    assert "`effective`" in text and "setting_inert" in text


def test_the_prefix_cache_doc_names_the_real_signals() -> None:
    """OPENCLAW-LONG-CONTEXT.md must teach the truthful fields and must not
    promise that ``usage.prompt_tokens`` drops -- the misreading behind A1."""
    text = (REPO_ROOT / "docs" / "OPENCLAW-LONG-CONTEXT.md").read_text(encoding="utf-8")
    assert "## 4. Concurrent requests that share a prefix" in text
    assert "timings.cache_n" in text
    assert "prompt_tokens_cached_total" in text
    assert "`usage.prompt_tokens` never moves" in text
    assert "never `usage.prompt_tokens`" in text
    assert "3.6·P" in text, "the no-cross-slot-sharing arithmetic at parallel 3"
    assert "begin a user message" in text, "the hybrid checkpoint rule"
    assert 'cannot say "expected"' in text, "the honest A2 answer"
    assert "`spec_type` `auto` vs `none`" in text and "benchmark_parallel" in text
    assert "scripts/measure_prefix_cache.py" in text
    assert (REPO_ROOT / "scripts" / "measure_prefix_cache.py").is_file()


def test_openclaw_links_the_prefix_cache_section() -> None:
    text = (REPO_ROOT / "docs" / "OPENCLAW.md").read_text(encoding="utf-8")
    assert "OPENCLAW-LONG-CONTEXT.md" in text
    assert "§4" in text and "share a long prefix" in text
