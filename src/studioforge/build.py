"""Build identity: the git commit this process runs from, resolved once.

``__version__`` names the last *release*. A server running a commit past its
tag reports the tag -- on 2026-09-22 a live server said ``1.26-09-04-3`` while
running ~50 commits past it, and the only honest signal was the
``implemented`` block of ``/api/capabilities`` (D64, CR-2). The ``build``
field beside the version names the checkout itself: the short commit SHA,
``-dirty`` when tracked files carry uncommitted changes, and ``unknown`` when
there is nothing truthful to say -- a wheel install, a checkout without git on
PATH, a git that does not answer in time.

Resolved once per process (:func:`functools.cache`) and eagerly at startup, so
no request path ever runs a subprocess; ``/health`` is polled every few
seconds by the watchdog and must stay free.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
from pathlib import Path

#: What ``build`` says when the checkout cannot be identified. Never a guess:
#: a wrong SHA is worse than none, because a reader would trust it.
UNKNOWN = "unknown"

#: Ceiling per git call. A cold disk or an index lock held by another git
#: process must delay startup by at most this much, never hang it.
GIT_TIMEOUT_S = 3.0


def checkout_root() -> Path | None:
    """The repository this module runs from, or ``None`` from a wheel.

    ``src/studioforge/build.py`` -> ``src/studioforge`` -> ``src`` -> the root.
    The ``.git`` entry must exist *there*: ``git`` walks upward on its own, so
    without this check a wheel installed into a venv that happens to live
    inside some other repository would report that repository's commit.
    ``.git`` is a directory in a main checkout and a file in a linked
    worktree; both count.
    """
    root = Path(__file__).resolve().parents[2]
    return root if (root / ".git").exists() else None


def _git(args: list[str], cwd: Path) -> str | None:
    """Stdout of one git command, or ``None`` for any failure at all."""
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


@functools.cache
def build_id() -> str:
    """``<short sha>``, ``<short sha>-dirty`` or ``unknown``; computed once.

    Never raises. Tests reset it with ``build_id.cache_clear()``.
    """
    root = checkout_root()
    if root is None or shutil.which("git") is None:
        return UNKNOWN
    sha = _git(["rev-parse", "--short", "HEAD"], root)
    if not sha:
        return UNKNOWN
    # Tracked files only: an untracked scratch file in the checkout is not a
    # modification of the build. A status that cannot be read (an index lock,
    # a timeout) leaves the SHA standing -- it is still true on its own.
    status = _git(["status", "--porcelain", "--untracked-files=no"], root)
    if status:
        return f"{sha}-dirty"
    return sha
