"""The S7 docs (D70, item 22): one combined table, and D48's stale TTL text points at D60.

Tiers lived in OPENCLAW-RIG.md §5, TTLs in SETUP.md, and the terminal-versus-
retry split in three different tables; no single table carried all three, and
the superseded D48 entry still said the TTL map shipped empty. These pin the
one table to the shipped config, and the amendments to D60.
"""

from __future__ import annotations

import re
from pathlib import Path

from studioforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
RIG = REPO_ROOT / "docs" / "OPENCLAW-RIG.md"
DECISIONS = REPO_ROOT / "DECISIONS.md"
LEAD_IN = "**Tiers, idle timeouts and the retry split, in one table**"


def _section(text: str, start: str, end: str) -> str:
    begin = text.index(start)
    return text[begin : text.index(end, begin)]


def _codes(row: str) -> set[str]:
    return set(re.findall(r"`([a-z_]+)`", row))


def _rows() -> dict[str, str]:
    section = _section(RIG.read_text(encoding="utf-8"), "## 9.", "## 10.")
    assert section.count(LEAD_IN) == 1, "exactly ONE combined table"
    table = section[section.index(LEAD_IN) :]
    rows = [line for line in table.splitlines() if line.startswith("| ")]
    return {line.split("|")[1].strip().strip("*"): line for line in rows}


def test_the_combined_table_carries_the_shipped_tier_ttls() -> None:
    """The numbers come from the code, so the docs cannot drift from the config again."""
    models = Config(data_dir=REPO_ROOT / "tests" / "unit" / "_unused").models
    rows = _rows()
    assert f"{models.ttl_by_priority[1]} s" in rows["tier 1"]
    assert f"{models.ttl_by_priority[2]} s" in rows["tier 2"]
    assert f"{models.ttl_by_priority[3]} s" in rows["tier 3 (and **omitted**)"]
    assert "priority_hold" in rows["tier 1"], "tier 1 is what holds the others off"
    assert f"`models.default_ttl_s: {models.default_ttl_s}`" in RIG.read_text(encoding="utf-8")


def test_the_combined_table_splits_terminal_from_retry_after() -> None:
    rows = _rows()
    retry, terminal = rows["retry-after"], rows["terminal"]
    assert retry.rstrip().endswith("| **yes** |")
    assert terminal.rstrip().endswith("| **no** |")
    assert "`unsupported_architecture` 400" in terminal, "terminal: never retry unchanged"
    assert "not in the active llama.cpp build" in terminal
    for code in ("priority_hold", "model_busy", "benchmark_busy", "lease_vacating"):
        assert code in _codes(retry), code
    for code in ("insufficient_vram", "context_exceeded", "lease_conflict", "model_load_failed"):
        assert code in _codes(terminal), code
    # gpu_leased is the one code on both sides, split by the lease kind.
    assert _codes(retry) & _codes(terminal) == {"gpu_leased"}
    assert "`retry_after_s`" in retry


def test_d48s_stale_ttl_text_now_points_at_d60_without_a_new_heading() -> None:
    text = DECISIONS.read_text(encoding="utf-8")
    d48 = _section(text, "## D48", "## D49")
    amendments = [line for line in d48.splitlines() if "Amended 2026-09-23" in line]
    assert len(amendments) == 2, "both stale passages carry the pointer"
    for line in amendments:
        block = d48[d48.index(line) : d48.index(line) + 600]
        assert "D60" in block
        assert "900" in block, "the amendment states the numbers D60 shipped"
    assert "## D70" not in text, "the D70 record is docs/pending until the owner promotes it"
