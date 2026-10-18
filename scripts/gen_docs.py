#!/usr/bin/env python3
"""Regenerate (or check) every generated contract and reference doc.

    uv run python scripts/gen_docs.py            # rewrite what changed
    uv run python scripts/gen_docs.py --check    # exit 1 if anything is stale (CI)

Writes docs/contracts/{openapi,mcp-tools,errors,settings-schema,capabilities}.json
docs/reference/{settings,errors}.md and docs/CLI.md from code. See studioforge.contracts.
Nothing here binds a port, starts an engine or touches a GPU.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from studioforge import contracts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 and show a diff summary if any generated file is stale",
    )
    args = parser.parse_args(argv)

    generated = contracts.render_contracts()
    stale = contracts.stale_files(generated, ROOT)
    if args.check:
        if not stale:
            print(f"generated docs are up to date ({len(generated)} files)")
            return 0
        for path in stale:
            old = contracts.read_committed(path, ROOT) or ""
            diff = difflib.unified_diff(
                old.splitlines(),
                generated[path].splitlines(),
                "committed",
                "generated",
                lineterm="",
            )
            print(f"STALE: {path}")
            for line in list(diff)[:20]:
                print("   ", line)
        print("\nrun `uv run python scripts/gen_docs.py` and commit the result", file=sys.stderr)
        return 1
    written = contracts.write_files(generated, ROOT)
    for path in written:
        print(f"wrote {path}")
    if not written:
        print(f"nothing to do ({len(generated)} files up to date)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
