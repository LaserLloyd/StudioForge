#!/usr/bin/env python3
"""Regenerate ``src/studioforge/data/engine_capabilities.json`` from llama.cpp.

The architecture / ftype / ggml-type lists in that file are the ones
``studioforge capabilities`` and the Setup tab show, and they are extracted from
llama.cpp's own source rather than hand-written -- 142 architectures move too
fast for a list a human maintains. The shipped copy is a **snapshot taken at one
tag**, so it goes stale the moment the engine pin moves: a model whose
architecture landed after the snapshot then looks unsupported by an engine that
supports it perfectly well. Since D49-8 the report refuses to state that as a
verdict when the snapshot's ``source_tag`` is not the running tag -- and this
script is how the snapshot catches up.

Run it when bumping ``engine.pinned_tag`` in the shipped defaults (see
docs/RELEASING.md), not on a schedule: it is a deliberate, reviewable change to
a tracked data file.

    python scripts/refresh_engine_capabilities.py b10549
    python scripts/refresh_engine_capabilities.py b10549 --dry-run
    python scripts/refresh_engine_capabilities.py b10549 --checkout /path/to/llama.cpp

What it does: shallow-clones ``ggml-org/llama.cpp`` at the tag into a temporary
directory (``git clone --branch <tag> --depth 1``), reuses
:func:`studioforge.core.capabilities.extract_from_checkout` -- the same parser
the live report uses, so the snapshot cannot drift from it in format -- and
rewrites the JSON with ``source_tag`` set to the tag. ``--checkout`` skips the
clone for a tree you already have (verify it is at the right tag yourself).

Needs ``git`` on PATH and network access. It writes exactly one file in the
repository and prints a diff summary; nothing else is touched.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = REPO_ROOT / "src" / "studioforge" / "data" / "engine_capabilities.json"
UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"


def _load_extractor() -> Callable[[Path], dict[str, list[str]] | None]:
    """Import the live extractor, adding ``src/`` to the path if need be.

    Deliberately the *same* function the running report uses: a second parser
    here would drift, and the whole point of the snapshot is that it is what
    the extractor would have produced against that tag.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from studioforge.core.capabilities import extract_from_checkout

    return extract_from_checkout


def clone(tag: str, into: Path) -> Path:
    """Shallow-clone llama.cpp at ``tag``. Returns the checkout root."""
    dest = into / "llama.cpp"
    cmd = ["git", "clone", "--branch", tag, "--depth", "1", UPSTREAM, str(dest)]
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)  # fixed argv, never a shell
    if result.returncode != 0:
        raise SystemExit(
            f"git clone failed (exit {result.returncode}). Is '{tag}' a real llama.cpp "
            "tag, and is git on PATH?"
        )
    return dest


def summarise(label: str, data: dict[str, object]) -> None:
    counts = {k: len(v) for k, v in data.items() if isinstance(v, list)}
    print(f"  {label}: " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="llama.cpp build tag, e.g. b10549")
    parser.add_argument(
        "--checkout",
        type=Path,
        default=None,
        help="Use this existing llama.cpp checkout instead of cloning",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and report, but do not rewrite the snapshot",
    )
    args = parser.parse_args(argv)

    extract_from_checkout = _load_extractor()

    old: dict[str, object] = {}
    if SNAPSHOT.is_file():
        old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        summarise(f"current snapshot ({old.get('source_tag', 'unknown tag')})", old)

    tmp: str | None = None
    try:
        if args.checkout is not None:
            root = args.checkout.expanduser().resolve()
        else:
            tmp = tempfile.mkdtemp(prefix="sf-llamacpp-")
            root = clone(args.tag, Path(tmp))
        extracted = extract_from_checkout(root)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

    if not extracted:
        raise SystemExit(
            f"could not extract the architecture tables from {args.checkout or 'the clone'}: "
            "src/llama-arch.cpp or include/llama.h is missing or unparsable"
        )

    payload = {
        "architectures": extracted["architectures"],
        "ggml_types": extracted["ggml_types"],
        "file_types": extracted["file_types"],
        "source_tag": args.tag,
    }
    summarise(f"extracted ({args.tag})", payload)

    if args.dry_run:
        print("--dry-run: the snapshot was not written")
        return 0

    SNAPSHOT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT}")
    print(
        "Review the diff, then update the pinned tag and docs/RELEASING.md's "
        "capability-snapshot step in the same commit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
