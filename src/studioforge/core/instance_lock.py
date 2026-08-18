"""Exactly one StudioForge instance owns a data directory at a time.

**The failure this prevents.** On 2026-08-18 a test process built a second
``create_app`` against the *live* data directory. Its :class:`Downloader`
loaded the live queue out of ``registry.sqlite3`` and started writing the same
``.part`` file the live server was already streaming into. The live transfer
died with ``WinError 32``; the other writer then "completed" an interleaved
22.58 GB file over a destination declared as 19.27 GB and removed the partial,
so the user's Resume restarted from zero. Neither process did anything wrong on
its own -- *two processes pointed at one data directory* was the entire bug.
See DECISIONS.md D24.

**Why an OS lock and not a pid file.** A pid file records an intention; it stops
nothing, and it outlives a crash as a lie that the next start has to guess
about. An exclusive byte-range lock (``msvcrt.locking`` on Windows,
``fcntl.flock`` on POSIX) is released by the kernel however the holder dies --
clean exit, ``SIGKILL``, Task Manager's End Task, a bluescreen -- so a stale
lock file is simply taken over and no reaping heuristic is needed. The pid,
process creation time and start time written *inside* the file exist for the
error message and for :meth:`InstanceLock.holder`, never for the decision: the
decision is the kernel's.

**Why the locked byte lives past the end of the file.** On Windows a locked byte
range cannot be read by any other process. Locking byte zero would therefore
make the holder's own identity unreadable to exactly the process that needs to
name it in an error message. The lock byte sits at an offset no real file will
ever reach, so the JSON payload stays plainly readable while the lock is held.

**What this module deliberately does not import.** Nothing from the rest of the
stack beyond the logger. It is the guard that decides whether the stack may
start its background workers at all; a guard that drags in the machinery it
guards is one import cycle away from being the reason a server will not boot.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Final

from studioforge.logging import get_logger

log = get_logger(__name__)

__all__ = ["INSTANCE_LOCK_NAME", "InstanceLock"]

#: Lock file name inside the data directory. Dotted so it sorts out of the way
#: and reads as machinery rather than as something a user should open.
INSTANCE_LOCK_NAME: Final = ".instance.lock"

#: Byte offset of the locked region. Past the end of any file that will ever
#: exist, so the readable JSON at offset 0 is never inside a locked range (see
#: the module docstring). Verified on Windows: locking beyond EOF succeeds, is
#: refused to a second handle, and allocates nothing.
_LOCK_OFFSET: Final = 1 << 62

#: ``O_BINARY`` only exists on Windows; on POSIX the flag is meaningless.
_OPEN_FLAGS: Final = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)


def _lock_fd(fd: int) -> None:
    """Take an exclusive, non-blocking OS lock on ``fd``.

    Raises ``OSError`` when another handle -- in this process or any other --
    already holds it. Both back ends are non-blocking on purpose: a startup path
    that blocks on a lock held by a healthy server would hang instead of
    reporting the one thing the operator needs to know.
    """
    # `sys.platform` rather than the module constant: a type checker narrows
    # this form, so the branch for the other platform is not analysed with the
    # wrong stdlib (``fcntl`` does not exist on Windows, ``msvcrt`` not on POSIX).
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        finally:
            # The lock is on a byte range, not on the file position, so putting
            # the cursor back is free and keeps every later read/write honest.
            os.lseek(fd, 0, os.SEEK_SET)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_fd(fd: int) -> None:
    """Drop the lock taken by :func:`_lock_fd`.

    Closing the descriptor releases it too on both platforms; this exists so
    ``release()`` is explicit rather than relying on a side effect of close.
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _process_create_time(pid: int) -> float | None:
    """Creation timestamp of ``pid``, or ``None`` when it cannot be read."""
    try:
        import psutil

        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 - no psutil, no such process, access denied
        return None


def _pid_is_live(pid: int, create_time: float | None) -> bool:
    """Whether ``pid`` is a live process that really is the one we recorded.

    ``create_time`` is the pid-reuse guard: on a busy box the OS hands a dead
    process's number to something else, and treating that stranger as the lock
    holder would keep a perfectly startable server in secondary mode forever.
    A process whose creation time is more than a second away from the recorded
    one is a different process wearing the same number.

    The two "cannot tell" cases are answered in opposite directions on purpose.
    *No psutil, or access denied*: the process may well be there, and calling a
    live holder stale would invite a second instance -- the exact failure this
    module exists to stop -- so it reports live. *No such process*: that is not
    an inability to tell, it is an answer.
    """
    if pid <= 0:
        return False
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a hard dependency here
        return True
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            actual = proc.create_time()
            if abs(actual - float(create_time)) > 1.0:
                return False
        return bool(proc.is_running())
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, ValueError):  # access denied, or a nonsense create_time
        return True


class InstanceLock:
    """Exclusive OS-level ownership of one data directory.

    Usage is deliberately blunt::

        lock = InstanceLock(config.data_dir)
        if lock.acquire():
            ...start background workers...
        else:
            log.error("another instance owns this directory", holder=lock.holder())

    ``acquire()`` is non-blocking by default because the answer "somebody else
    has it" is actionable immediately, while a blocking wait at startup looks
    exactly like a hang.
    """

    def __init__(self, data_dir: Path | str, *, name: str = INSTANCE_LOCK_NAME) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / name
        self._fd: int | None = None

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    @property
    def held(self) -> bool:
        """Whether *this* object currently owns the lock."""
        return self._fd is not None

    def acquire(
        self, blocking: bool = False, *, timeout_s: float = 30.0, poll_s: float = 0.25
    ) -> bool:
        """Take the lock; return ``False`` when another live process holds it.

        Re-acquiring an already-held lock is a no-op returning ``True`` so a
        caller does not have to track whether it has already run.

        A lock file left behind by a crash is taken over silently: the kernel
        released the lock when the process died, so the file is just bytes at
        that point and its recorded pid is only used to say *who* it used to be.
        """
        if self._fd is not None:
            return True
        self.data_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            stale = self.holder() is None and self.path.exists()
            fd = os.open(self.path, _OPEN_FLAGS, 0o644)
            try:
                _lock_fd(fd)
            except OSError:
                os.close(fd)
                if not blocking or time.monotonic() >= deadline:
                    return False
                time.sleep(poll_s)
                continue
            self._fd = fd
            self._write_payload(fd)
            if stale:
                log.info(
                    "instance_lock.stale_takeover",
                    path=str(self.path),
                    reason="the previous holder is gone; its lock was released by the OS",
                )
            return True

    def release(self) -> None:
        """Drop the lock. Safe to call when it was never held.

        The file itself is left in place on purpose. Deleting it races every
        other process that is a microsecond away from opening it, and the file
        is worthless without the lock anyway -- the next acquirer overwrites the
        payload.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            _unlock_fd(fd)
        with contextlib.suppress(OSError):
            os.close(fd)

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def holder(self) -> dict[str, Any] | None:
        """Who owns this data directory, or ``None`` when nobody live does.

        Read straight out of the lock file, which stays readable while the lock
        is held (the locked byte is past the end of the file). A payload naming
        a dead pid -- or a pid that has been recycled since -- reports ``None``:
        it describes a crash, not an owner.
        """
        try:
            raw = self.path.read_bytes()
        except OSError:
            return None
        if not raw.strip():
            return None
        try:
            data = json.loads(raw.decode("utf-8", "replace"))
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        pid = data.get("pid")
        if not isinstance(pid, int):
            return None
        if not _pid_is_live(pid, data.get("create_time")):
            return None
        return dict(data)

    def _write_payload(self, fd: int) -> None:
        """Stamp our identity into the file we have just locked.

        Never fatal: the lock is the guarantee, the payload is the explanation.
        A directory that cannot be written is a much bigger problem than an
        anonymous lock, and it will be reported by something that matters more.
        """
        pid = os.getpid()
        payload = {
            "pid": pid,
            "create_time": _process_create_time(pid),
            "started_at": time.time(),
            "data_dir": str(self.data_dir),
        }
        blob = json.dumps(payload, indent=2, sort_keys=True).encode()
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, blob)
            os.ftruncate(fd, len(blob))
            os.fsync(fd)
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("instance_lock.payload_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Context manager / cleanup
    # ------------------------------------------------------------------

    def __enter__(self) -> InstanceLock:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        # A leaked descriptor keeps a Windows directory undeletable, which turns
        # a forgotten release() into a failing temp-dir cleanup somewhere else
        # entirely. Closing here costs nothing and localises the mistake.
        with contextlib.suppress(Exception):
            self.release()
