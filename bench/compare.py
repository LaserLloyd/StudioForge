#!/usr/bin/env python
"""Compare two results files: ``python compare.py <new.json> [<old.json>]``.

Prints a markdown table of median and p95 per metric per prompt length with %
deltas and a verdict line applying the amended rule (README.md). It reports; it
never gates -- exit code is always 0. ``old`` defaults to the newest other
non-smoke file in the same results directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_stats as bs  # noqa: E402


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def newest_other(new: Path) -> Path | None:
    candidates = [
        p
        for p in new.parent.glob("*.json")
        if p.resolve() != new.resolve() and "-smoke" not in p.name
    ]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _delta(new: float | None, old: float | None, higher_is_better: bool) -> str:
    d = bs.delta_pct(new, old)
    if d is None:
        return "-"
    good = d >= 0 if higher_is_better else d <= 0
    mark = "" if abs(d) <= bs.REGRESSION_TOLERANCE_PCT else (" +" if good else " !")
    return f"{d:+.1f}%{mark}"


def describe(tag: str, data: dict[str, Any]) -> str:
    git = data.get("git") or {}
    cfg = data.get("config") or {}
    eff = cfg.get("effective") or {}
    plan = cfg.get("plan") or {}
    dirty = " (dirty)" if git.get("dirty") else ""
    validity = (data.get("validity") or {}).get("valid")
    engine = cfg.get("resolved_engine_tag") or (data.get("server") or {}).get("engine")
    return (
        f"{tag}: sha {str(git.get('sha') or '')[:12]}{dirty}"
        f" label={data.get('label')} at {data.get('timestamp')} valid={validity}"
        f" smoke={data.get('smoke')} engine={engine}"
        f"\n    plan: devices {plan.get('devices')} split {plan.get('split_mode')}"
        f" tensor_split {plan.get('tensor_split')}"
        f" ctx {plan.get('ctx_size')} x {plan.get('parallel')}"
        f" kv {plan.get('kv_cache_type')}/{plan.get('kv_cache_type_v')}"
        f" flash_attn {plan.get('flash_attn')}"
        f"\n    effective: batch {eff.get('batch_size')} ubatch {eff.get('ubatch_size')}"
        f" flash_attn {eff.get('flash_attn')} -- {eff.get('summary')}"
        f"\n    load wall: {(data.get('load') or {}).get('wall_s')} s"
    )


def report(new_path: Path, old_path: Path | None) -> None:
    new = load(new_path)
    print(f"# Benchmark comparison\n\n{describe('new', new)}")
    if old_path is None:
        print("\nno previous results file to compare against; new summary only\n")
        old: dict[str, Any] = {"summary": {}}
    else:
        old = load(old_path)
        print(f"{describe('old', old)}\n")
    ns, os_ = new.get("summary") or {}, old.get("summary") or {}
    lengths = sorted(set(ns) | set(os_), key=int)
    print(
        "| length | metric | old median | new median | delta |"
        " old p95 | new p95 | delta | n old/new |"
    )
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |")
    for key in lengths:
        n, o = ns.get(key) or {}, os_.get(key) or {}
        for metric, label, hib in bs.METRICS:
            nm, om = n.get(metric) or {}, o.get(metric) or {}
            om_med, nm_med = om.get("median"), nm.get("median")
            om_p95, nm_p95 = om.get("p95"), nm.get("p95")
            print(
                f"| {key} | {label} | {_fmt(om_med)} | {_fmt(nm_med)}"
                f" | {_delta(nm_med, om_med, hib)}"
                f" | {_fmt(om_p95)} | {_fmt(nm_p95)} | {_delta(nm_p95, om_p95, hib)}"
                f" | {om.get('n', 0)}/{nm.get('n', 0)} |"
            )
        for dev in sorted(
            set(n.get("peak_vram_bytes") or {}) | set(o.get("peak_vram_bytes") or {})
        ):
            nv, ov = (
                (n.get("peak_vram_bytes") or {}).get(dev),
                (o.get("peak_vram_bytes") or {}).get(dev),
            )
            gib = lambda b: None if b is None else b / 2**30  # noqa: E731
            print(
                f"| {key} | peak VRAM GiB cuda{dev} | {_fmt(gib(ov))} | {_fmt(gib(nv))}"
                f" | {_delta(gib(nv), gib(ov), False)} | | | | |"
            )
        flags = [
            f"cache-hit-suspected runs new={n.get('cache_hit_suspected_runs')}"
            if n.get("cache_hit_suspected_runs")
            else "",
            f"timings-missing runs new={n.get('timings_missing_runs')}"
            if n.get("timings_missing_runs")
            else "",
        ]
        if any(flags):
            print(f"| {key} | flags | | | {'; '.join(f for f in flags if f)} | | | | |")
    print("\n(delta marks: `+` better than the 2% tolerance, `!` a regression beyond it)")
    if old_path is None:
        return
    graded = bs.verdict(
        ns,
        os_,
        new_valid=bool((new.get("validity") or {}).get("valid", True)),
        old_valid=bool((old.get("validity") or {}).get("valid", True)),
    )
    if new.get("smoke") or old.get("smoke"):
        print(
            "\nnote: a smoke file is involved; smoke runs use 1024 tokens / 2 runs "
            "and cannot grade anything"
        )
    print(f"\n{graded['line']}")


def main(argv: list[str]) -> int:
    if not argv or len(argv) > 2:
        print(__doc__, file=sys.stderr)
        return 0
    new_path = Path(argv[0])
    if not new_path.exists():
        print(f"no such file: {new_path}", file=sys.stderr)
        return 0
    old_path = Path(argv[1]) if len(argv) == 2 else newest_other(new_path)
    if old_path is not None and not old_path.exists():
        print(f"no such file: {old_path}", file=sys.stderr)
        old_path = None
    report(new_path, old_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
