"""The last gate must scan every commit that travels, not only the tip.

A secret added in commit N and tidied away in N+1 still publishes: `git show N`
and every clone carry it forever. Scanning only the pushed tip reports that
exact case as clean — so the hook is driven here against a throwaway
repository built to contain it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts" / "hooks" / "pre-push"
SCANNER = ROOT / "scripts" / "scrub_check.py"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or sys.platform == "win32",
    reason="needs git and a POSIX shell",
)

#: Shaped like a real credential, entirely invented: it matches the generic
#: secret-token rule and contains no private identifier. Assembled at runtime
#: so this source file carries no line the scanner would (correctly) flag —
#: a fixture must not need an allow-pragma to exist.
PLANTED = 'DEPLOY_KEY = "ghp_' + "A" * 30 + '"\n'

ZERO = "0" * 40


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.fixture
def planted(tmp_path: Path) -> tuple[Path, str, str]:
    """A repo whose MIDDLE commit carries a secret its tip does not."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(SCANNER, repo / "scripts" / "scrub_check.py")
    shutil.copy(HOOK, repo / "scripts" / "hooks-pre-push")

    _git(repo.parent, "init", "-q", "-b", "main", str(repo))
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "user.email", "test@example.com")

    (repo / "README.md").write_text("clean\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    (repo / "planted.py").write_text(PLANTED, encoding="utf-8")
    _git(repo, "add", "planted.py")
    _git(repo, "commit", "-qm", "middle commit adds a token")
    middle = _git(repo, "rev-parse", "HEAD")

    _git(repo, "rm", "-q", "planted.py")
    _git(repo, "commit", "-qm", "tip removes it again")
    tip = _git(repo, "rev-parse", "HEAD")
    return repo, tip, middle


def _run_hook(repo: Path, tip: str) -> subprocess.CompletedProcess[str]:
    """Drive the hook exactly as git does: argv + refs on stdin."""
    env = {**os.environ, "PATH": os.environ.get("PATH", "")}
    return subprocess.run(
        ["sh", str(repo / "scripts" / "hooks-pre-push"), "origin", "https://example.com/repo.git"],
        input=f"refs/heads/main {tip} refs/heads/main {ZERO}\n",
        cwd=str(repo),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_the_tip_alone_looks_clean(planted):
    """The premise. If this ever fails, the fixture stopped reproducing."""
    repo, tip, _middle = planted
    tip_scan = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "scrub_check.py"),
            "--rev",
            tip,
            "--path",
            str(repo),
        ],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert tip_scan.returncode == 0, tip_scan.stdout


def test_pre_push_blocks_a_secret_only_an_intermediate_commit_carries(planted):
    repo, tip, middle = planted
    result = _run_hook(repo, tip)
    assert result.returncode != 0, (
        "pre-push allowed a push whose middle commit publishes a token\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert middle in combined, combined  # names the offending commit
    assert "REFUSING to publish" in combined


def test_pre_push_reports_how_many_trees_it_scanned(planted):
    """'scanned N' is the evidence that (a) really covered every commit."""
    repo, tip, _middle = planted
    result = _run_hook(repo, tip)
    assert "scanned 3 commit tree(s)" in result.stdout, result.stdout


def test_a_clean_history_still_passes(planted, tmp_path):
    """The gate must block the bad push without blocking every push."""
    repo, _tip, middle = planted
    _git(repo, "reset", "-q", "--hard", f"{middle}^")
    clean_tip = _git(repo, "rev-parse", "HEAD")
    result = _run_hook(repo, clean_tip)
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_hook_header_describes_what_it_does():
    """The header claimed 'the TREE of every commit' while the body scanned
    one. A gate that overstates its coverage is trusted for work it is not
    doing, so the claim is pinned to the implementation."""
    body = HOOK.read_text(encoding="utf-8")
    assert "rev-list" in body, "hook must iterate the outgoing commits"
    assert "EVERY commit" in body
