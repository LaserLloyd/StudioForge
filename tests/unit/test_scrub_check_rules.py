"""The privacy gate's content rules, driven on every platform.

test_pre_push_hook.py needs a POSIX shell and is skipped on Windows -- the
reference platform -- so nothing there exercises the RULES themselves where the
maintainer actually commits. These tests import scripts/scrub_check.py
directly and run the rule set over text and a throwaway directory.

The personal-identifier file (scripts/scrub-rules.local.txt) is git-ignored and
present only on the maintainer's machine, so every test here pins the local
rules to "none": the verdicts must not depend on whose clone runs them.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "scripts" / "scrub_check.py"


def _load(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """A fresh copy of the scanner module (it lives in scripts/, not a package)."""
    spec = importlib.util.spec_from_file_location("scrub_check_under_test", SCANNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "scrub_check_under_test", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def scrub(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "load_local_rules", lambda path=None: [])
    return module


# Assembled at runtime so this file carries no line the scanner would flag: a
# fixture must not need an allow-pragma to exist (see test_pre_push_hook.py).
_CLAUDE = "https://claude" + ".ai/"
_TAILNET_HOST = "100." + "101.102.103"


@pytest.mark.parametrize(
    "text",
    [
        _CLAUDE + "chat/0b7e2c4a-1f2d-4e5b-9a8c-0123456789ab",
        _CLAUDE + "share/5f2e8c1d-0000-4bbb-8ccc-0123456789ab",
        _CLAUDE + "project/0199aabb-ccdd-7eef-8000-0123456789ab",
        _CLAUDE + "code/artifact/5d0c9e8f-aaaa-bbbb",
        _CLAUDE + "code/session_01ABCdefGHI",
    ],
)
def test_every_claude_link_with_an_id_is_flagged(scrub: ModuleType, text: str) -> None:
    problems = scrub.scan_text(f"see {text} for the discussion", "msg")
    assert problems, text
    assert "claude" in problems[0].lower() or "session" in problems[0].lower()


@pytest.mark.parametrize(
    "text",
    [
        "no `claude" + ".ai/code/session` URLs and no trailer lines in the log",
        "sign in at " + _CLAUDE + " first",
    ],
)
def test_claude_prose_without_an_id_is_not_flagged(scrub: ModuleType, text: str) -> None:
    assert scrub.scan_text(text, "msg") == []


def test_a_tailnet_host_address_is_flagged_outside_tests(scrub: ModuleType) -> None:
    problems = scrub.scan_text(f"point sfctl at http://{_TAILNET_HOST}:1234", "msg")
    assert problems and scrub.CGNAT_HOST_WHY in problems[0]


@pytest.mark.parametrize(
    "text",
    [
        "Tailscale hands out 100." + "64.0.0/10 addresses",
        "the 100." + "64.0.0 / 10 block",
        "the tailnet lives in 100." + "100.0.0 and up",
        "allow 100." + "101.12.0/24 through the firewall",
        "a public address like 100." + "128.0.1 is not CGNAT",
        "version 1.26-08-31, 100 GiB, 100.5 tok/s",
    ],
)
def test_the_cgnat_range_in_prose_is_not_a_host(scrub: ModuleType, text: str) -> None:
    assert scrub.scan_text(text, "msg") == []


def test_tailnet_addresses_are_fixtures_under_tests_but_not_in_docs(
    scrub: ModuleType, tmp_path: Path
) -> None:
    line = f'HOST = "{_TAILNET_HOST}"\n'
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(line, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text(line, encoding="utf-8")
    problems = scrub.scan(tmp_path)
    assert [p for p in problems if p.startswith("docs/guide.md:1:")]
    assert not [p for p in problems if p.startswith("tests/")]


def test_a_local_rule_still_fires_inside_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture exemption covers GENERIC heuristics only; a maintainer's real
    node address, listed as a local rule, is a finding everywhere. Driven with
    an INVENTED identifier through a throwaway rules file."""
    scrub = _load(monkeypatch)
    rules = tmp_path / "rules.txt"
    rules.write_text(r"\bwintermute-node\b" + "\n", encoding="utf-8")
    real_loader = scrub.load_local_rules
    monkeypatch.setattr(scrub, "load_local_rules", lambda path=None: real_loader(rules))
    tree = tmp_path / "tree"
    (tree / "tests").mkdir(parents=True)
    (tree / "tests" / "test_y.py").write_text('PEER = "wintermute-node"\n', encoding="utf-8")
    problems = scrub.scan(tree)
    assert any(p.startswith("tests/test_y.py:1:") and scrub.LOCAL_RULE_WHY in p for p in problems)


def test_the_selftest_passes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--selftest drives its own throwaway rules file for the LICENCE checks, so
    it gets the module unpatched -- the `scrub` fixture's no-op loader would
    make those checks fail for the wrong reason."""
    scrub = _load(monkeypatch)
    assert scrub.selftest() == 0
    assert "OK" in capsys.readouterr().out
