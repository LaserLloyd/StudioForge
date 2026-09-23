"""Size-bounded log files that stay safe when two processes share one (D69 §15).

Standard library only: the watchdog imports this module, and its dependency
surface is a deliberate invariant (see :mod:`studioforge.watchdog.__main__`).

**Who writes what.** Rotation is only safe once that is settled:

* ``studioforge.log`` -- the server (``studioforge serve``) holds it open and
  rotates it (:class:`SafeRotatingFileHandler`). Every other process that logs
  into it (the tray, one-shot CLI commands such as ``studioforge engine``, the
  stdio MCP server) appends one record at a time and never holds it
  (:class:`AppendFileHandler`), so it cannot pin the file against the server's
  rename. Two servers overlap for a moment during a restart handover.
* ``watchdog.log`` -- the watchdog alone; it rotates it.
* ``tray-server.log`` -- the server child's console. The tray opens it, the
  child inherits the handle, and no process can rotate a handle another one
  writes through. So the tray rotates it only when it (re)starts the server,
  before the new child exists (:func:`rotate_if_large`).

**Why not ``RotatingFileHandler`` as it is.** On Windows a file another process
has open cannot be renamed (Python opens files without ``FILE_SHARE_DELETE``).
The stdlib handler shifts ``.1 -> .2 -> ...`` *before* it renames the live file,
so each failed attempt pushed every backup one step further out and deleted
the oldest; the record that triggered the attempt was dropped with a traceback
on stderr, and the next record did it all again. Here:

1. the live file is renamed first, to a new timestamped name, and that is the
   only step that can fail on a shared file. Nothing else is touched until it
   has succeeded;
2. a failed rename is not an error: the handler keeps appending to the same
   file, says so once in the file itself, and tries again after ``retry_s``;
3. backups are never renamed again, only pruned oldest-first after a new one
   exists. A backup that cannot be deleted (open in a viewer) stays until the
   next rotation.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import time
from pathlib import Path

MB = 1 << 20

#: 20 MiB x 5 backups: a month or more of ``studioforge.log`` on the reference
#: rig (0.5-2 MB a day) in 120 MiB at most per log.
DEFAULT_MAX_BYTES = 20 * MB
DEFAULT_BACKUP_COUNT = 5

#: How long a handler waits before retrying a rotation that could not rename
#: the live file (another process held it).
DEFAULT_RETRY_S = 300.0

_STAMP = "%Y%m%d-%H%M%S"


def backup_pattern(path: Path) -> re.Pattern[str]:
    """Names of ``path``'s rotated copies: ``<stem>.<YYYYmmdd-HHMMSS>[_N]<suffix>``.

    The ``.log`` suffix is kept so a backup still opens in whatever opens a
    log. ``_N`` only appears when two rotations land in the same second.
    """
    stem, suffix = re.escape(path.stem), re.escape(path.suffix)
    return re.compile(rf"^{stem}\.(\d{{8}}-\d{{6}})(?:_(\d+))?{suffix}$")


def _backup_keys(path: Path) -> list[tuple[tuple[str, int], Path]]:
    """``((stamp, n), backup)`` for every rotated copy, oldest first."""
    pattern = backup_pattern(path)
    found: list[tuple[tuple[str, int], Path]] = []
    try:
        entries = [entry for entry in os.scandir(path.parent) if entry.is_file()]
    except OSError:
        return []
    for entry in entries:
        match = pattern.match(entry.name)
        if match:
            found.append(((match.group(1), int(match.group(2) or 0)), path.parent / entry.name))
    return sorted(found)


def backups_of(path: Path) -> list[Path]:
    """The rotated copies of ``path``, oldest first. ``[]`` when none can be listed."""
    return [backup for _key, backup in _backup_keys(path)]


def _backup_name(path: Path) -> Path:
    """A new backup name that sorts after every existing one.

    Within one second the counter continues from the highest one present, so
    a name the pruning freed is never reused out of order.
    """
    stamp = time.strftime(_STAMP, time.localtime())
    same_second = [n for (seen, n), _backup in _backup_keys(path) if seen == stamp]
    if not same_second:
        return path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    return path.with_name(f"{path.stem}.{stamp}_{max(same_second) + 1}{path.suffix}")


def prune_backups(path: Path, keep: int) -> list[Path]:
    """Delete all but the newest ``keep`` backups of ``path``; returns what went.

    Never raises. A backup another process has open is left in place and goes
    at the next rotation.
    """
    removed: list[Path] = []
    backups = backups_of(path)
    for old in backups[: max(0, len(backups) - max(1, keep))]:
        try:
            old.unlink()
        except OSError:
            continue
        removed.append(old)
    return removed


def rotate(path: Path, *, backup_count: int) -> Path:
    """Rename ``path`` to a new timestamped backup, then prune; returns the backup.

    Raises :class:`OSError` when the rename fails (on Windows, typically another
    process holding the file). Nothing at all has changed in that case.
    """
    dest = _backup_name(path)
    path.rename(dest)  # the only step that can fail on a shared file
    prune_backups(path, backup_count)
    return dest


def rotate_if_large(path: Path, *, max_bytes: int, backup_count: int) -> Path | None:
    """Rotate ``path`` when it has reached ``max_bytes``; for a file nobody holds.

    The tray calls this before it starts a server, when no child holds the
    console log any more. ``None`` when nothing was done: the file is smaller,
    missing, ``max_bytes`` is 0 (never rotate), or it could not be renamed
    (someone still holds it). The caller then simply appends as before.
    """
    if max_bytes <= 0:
        return None
    try:
        if path.stat().st_size < max_bytes:
            return None
        return rotate(path, backup_count=backup_count)
    except OSError:
        return None


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """A size-rotating handler for the ONE process that owns a log file.

    See the module docstring for the ordering that makes a shared file safe.
    ``max_bytes <= 0`` never rotates; the handler is then a plain append.
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        encoding: str = "utf-8",
        retry_s: float = DEFAULT_RETRY_S,
    ) -> None:
        super().__init__(
            filename,
            mode="a",
            maxBytes=max(0, int(max_bytes)),
            backupCount=max(1, int(backup_count)),
            encoding=encoding,
        )
        self.retry_s = float(retry_s)
        self._retry_at = 0.0
        #: Why the last rotation could not happen, or ``None``.
        self.last_rotation_error: str | None = None
        #: Rotations this handler has performed.
        self.rotations = 0

    def shouldRollover(self, record: logging.LogRecord) -> bool:  # noqa: N802 - stdlib name
        if self.maxBytes <= 0:
            return False
        if self._retry_at and time.monotonic() < self._retry_at:
            return False
        if self.stream is None:
            self.stream = self._open()
        try:
            # The size on disk, not our own write position: other processes
            # append to this file between our records.
            return os.fstat(self.stream.fileno()).st_size >= self.maxBytes
        except (OSError, ValueError):
            return False

    def doRollover(self) -> None:  # noqa: N802 - stdlib name
        path = Path(self.baseFilename)
        moved = self._moved_under_us()
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment,unused-ignore]
        if moved:
            # Someone else rotated it already (POSIX lets a rename happen under
            # an open handle): follow them to the new file instead of rotating
            # the fresh one they just started.
            self.stream = self._open()
            return
        try:
            backup = rotate(path, backup_count=self.backupCount)
        except OSError as exc:
            first = self.last_rotation_error is None
            self._retry_at = time.monotonic() + self.retry_s
            self.last_rotation_error = f"{type(exc).__name__}: {exc}"
            self.stream = self._open()
            if first:
                # Once per episode, not once per retry: a holder that keeps
                # the file for days would otherwise add a line every retry_s.
                self._note(
                    logging.WARNING,
                    f"log rotation deferred ({self.last_rotation_error}): another "
                    f"process holds this file; still appending to it, and retrying "
                    f"every {self.retry_s:.0f} s",
                )
            return
        self._retry_at = 0.0
        self.last_rotation_error = None
        self.rotations += 1
        self.stream = self._open()
        self._note(logging.INFO, f"log rotated; the previous file is {backup.name}")

    def _moved_under_us(self) -> bool:
        """Whether the path no longer names the file our stream writes to."""
        if self.stream is None:
            return False
        try:
            mine = os.fstat(self.stream.fileno())
        except (OSError, ValueError):
            return False
        try:
            current = Path(self.baseFilename).stat()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        return (mine.st_dev, mine.st_ino) != (current.st_dev, current.st_ino)

    def _note(self, level: int, text: str) -> None:
        """One line written straight into the file, formatted like its records.

        Not through the logging machinery: this runs inside ``emit`` with the
        handler's lock held, and a nested log call is exactly the kind of
        re-entrancy a handler must not depend on.
        """
        if self.stream is None:
            return
        record = logging.LogRecord("studioforge.logfiles", level, __file__, 0, text, None, None)
        try:
            self.stream.write(self.format(record) + self.terminator)
            self.stream.flush()
        except Exception:  # noqa: BLE001 - a note is never worth a failed record
            pass


class AppendFileHandler(logging.Handler):
    """Open, append one record, close: for a process that shares a log it does not own.

    Holding the file open for the life of the process is what would stop the
    owner from ever renaming it on Windows, and the tray, the process most
    likely to share ``studioforge.log``, runs for weeks. It writes a few lines
    a day, so an open per record costs nothing. It never rotates.
    """

    terminator = "\n"

    def __init__(self, filename: str | os.PathLike[str], *, encoding: str = "utf-8") -> None:
        super().__init__()
        self.baseFilename = str(Path(os.fspath(filename)).absolute())
        self.encoding = encoding

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            with Path(self.baseFilename).open("a", encoding=self.encoding) as handle:
                handle.write(message + self.terminator)
        except Exception:  # noqa: BLE001 - logging must never take the caller down
            self.handleError(record)
