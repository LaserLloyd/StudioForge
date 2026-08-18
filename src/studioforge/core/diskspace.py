"""How much room is left on the volume the model library lives on.

A download queue that cannot say "412 GiB free, 380 GiB queued" is asking the
user to discover the answer by filling the disk, and on this system that is an
expensive way to find out: a half-written ``.part``, an ENOSPC the downloader
correctly reports as fatal rather than retryable, and a model library sharing a
volume with everything else on the drive. The number costs one syscall and it is
shown at the two moments it can still change a decision -- above the queue, and
next to any quant that would not fit in what is left.

The report is deliberately about the **volume**, not the directory. Free space
is a property of the mount, so a ``models.dir`` that does not exist yet (nothing
has been downloaded on a fresh install) still has a drive underneath it: the
lookup walks up to the nearest existing ancestor rather than refusing to answer.

Nothing here raises. A volume that cannot be measured reports zeroes with the
reason attached, because the Download tab must degrade to "no line" rather than
to a broken panel -- and because "I could not tell" must never render as "you
are about to run out", which is how a warning stops being believed.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any, Final

from studioforge.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DISK_USAGE_TTL_S",
    "LOW_FREE_BYTES",
    "LOW_FREE_FRACTION",
    "clear_cache",
    "disk_report",
]

#: Cache lifetime for one ``shutil.disk_usage`` call. The GUI polls the queue on
#: a timer -- twice a second by default, for as long as a multi-hour download
#: runs -- and free space does not move fast enough to justify a syscall per
#: tick. Two seconds is still short enough that the line visibly tracks a
#: transfer in flight.
DISK_USAGE_TTL_S: Final = 2.0

#: "Low" is whichever of these two bounds is *larger*. The floor exists because
#: a single GGUF in this library is routinely 20-40 GiB, so a volume with under
#: 10 GiB spare cannot take another one whatever its size. The fraction exists
#: because on a 4 TB library drive 10 GiB free is already an emergency that the
#: floor alone would happily call fine.
LOW_FREE_BYTES: Final = 10 * 1024**3
LOW_FREE_FRACTION: Final = 0.05

#: ``{resolved path: (monotonic stamp, (total, used, free))}``. Deliberately not
#: locked: the worst a race can do is two syscalls where one would have done.
_CACHE: dict[str, tuple[float, tuple[int, int, int]]] = {}


def clear_cache() -> None:
    """Drop the cached ``disk_usage`` results.

    For tests, and for anything that has just changed ``models.dir`` and wants
    the next report to describe the new volume rather than the old one.
    """
    _CACHE.clear()


def _existing_ancestor(path: Path) -> Path:
    """The nearest ancestor of *path* that exists.

    ``models.dir`` is created on the first download, so asking about free space
    before anything has been fetched must not fail. The volume is the answer,
    and the volume is whatever the first existing parent sits on. Stops at the
    root, which exists on every platform this runs on -- and if even that is
    gone (an unmapped drive letter), ``disk_usage`` reports the reason.
    """
    probe = path
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return probe


def _drive_label(path: Path) -> str:
    """Shortest thing a human recognises as "which disk": ``E:`` or ``/home``.

    On Windows that is the drive letter (or the UNC share), which is what the
    user sees in Explorer. POSIX has no such label, so the mount point the path
    actually lives on is walked to instead -- ``/`` is a useless answer when the
    library is on a separate ``/mnt/models`` volume.
    """
    if path.drive:
        return path.drive
    probe = path
    while not os.path.ismount(probe):
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return str(probe)


def _usage(path: Path) -> tuple[int, int, int]:
    """``(total, used, free)`` for the volume holding *path*, cached briefly.

    Unpacked as a tuple rather than read off ``.total``/``.free`` so a test can
    stand in a plain 3-tuple for the platform call.
    """
    key = str(path)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached is not None and now - cached[0] < DISK_USAGE_TTL_S:
        return cached[1]
    total, used, free = shutil.disk_usage(path)
    value = (int(total), int(used), int(free))
    _CACHE[key] = (now, value)
    return value


def disk_report(models_dir: Path | str, queued_remaining_bytes: int) -> dict[str, Any]:
    """Free space where downloads land, before and after the queue drains.

    ``queued_remaining_bytes`` is what the caller still has to fetch (see
    :meth:`studioforge.core.downloader.Downloader.queued_remaining_bytes`), and
    subtracting it is the entire point: "412 GiB free" is reassuring right up to
    the moment you remember the 380 GiB already queued behind it.

    ``free_after_queue_bytes`` is signed. A negative value is the useful case --
    it says by how much the queue overruns the disk -- and clamping it to zero
    would turn "you are 40 GiB short" into "you have nothing left", which reads
    like a different, smaller problem.
    """
    queued = max(0, int(queued_remaining_bytes or 0))
    try:
        path = Path(models_dir).expanduser().resolve()
    except OSError:  # pragma: no cover - resolve(strict=False) is near-total
        path = Path(models_dir)
    base = _existing_ancestor(path)
    report: dict[str, Any] = {
        "path": str(path),
        "drive": _drive_label(base),
        "total_bytes": 0,
        "free_bytes": 0,
        "queued_bytes": queued,
        "free_after_queue_bytes": 0,
        # Unknown is not low. See the module docstring.
        "low": False,
        "error": None,
    }
    try:
        total, _used, free = _usage(base)
    except (OSError, ValueError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        log.debug("diskspace.unavailable", path=str(base), error=str(exc))
        return report
    after = free - queued
    report["total_bytes"] = total
    report["free_bytes"] = free
    report["free_after_queue_bytes"] = after
    report["low"] = after < max(LOW_FREE_BYTES, int(total * LOW_FREE_FRACTION))
    return report
