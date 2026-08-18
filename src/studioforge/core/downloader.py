"""Resumable, restart-surviving GGUF downloads into the shared model library.

Three design decisions carry most of the weight here.

**LM Studio layout.** Files land in ``<models.dir>/<publisher>/<repo>/<filename>``
-- byte for byte the convention LM Studio already uses on disk. StudioForge
scans the same directory LM Studio does, so writing anywhere else would fork the
user's library into two half-populated copies and force a manual copy after
every download. Nothing existing is ever moved, copied or rewritten: the
downloader only creates directories and new files, and refuses to clobber a file
that is already complete.

**The ``.part`` + ``Range`` protocol.** Bytes stream into ``<dest>.part`` and the
file is renamed onto ``<dest>`` only after its length -- and its sha256, when HF
publishes one -- check out. A partially written file that already has the final
name is indistinguishable from a good one to the registry scanner, which is how
you get a mysterious "invalid GGUF magic" three days later; the ``.part`` suffix
makes an interrupted transfer self-evidently incomplete. Resume sends
``Range: bytes=<have>-`` and then **verifies the server actually honoured it**:
a ``206`` with a matching start offset means append, while a ``200`` means the
server ignored the header and is sending the whole object from byte zero.
Appending that to existing bytes would produce a file of exactly the right
length whose contents are garbage -- the single worst failure mode available
here -- so a ``200`` restarts the transfer from zero instead.

**One writer, and completion proven on disk.** The ``.part`` is opened once per
transfer and held under an exclusive OS lock for the whole of it -- streaming,
verification and the final rename included. A second process that tries to write
the same file is refused immediately with a sentence naming the cause, because
the alternative is what happened on 2026-08-18: two Downloaders interleaved
their chunks into one ``.part``, one died with ``WinError 32`` and the other
renamed a 22.58 GB file over a destination declared as 19.27 GB. It passed
verification because verification trusted the *stream* -- the running hash and
the byte counter -- and never asked the filesystem what it actually held. It
does now: fsync, then ``stat`` the ``.part`` and compare against both the
streamed count and the declared size before publishing. See DECISIONS.md D24.

**Transient failures are retried, everything else fails fast.** A dropped
connection, a 5xx, a 429 or a momentary ``PermissionError`` on the ``.part`` is
the network or the antivirus scanner, not a broken download: those back off with
jitter and resume from the bytes already on disk. A 404, a 401, a size mismatch,
a checksum mismatch or a full disk are answers, not hiccups, and are reported
immediately instead of being retried four more times.

**Honest pre-download estimates.** :func:`fit_verdict` sizes a model before its
GGUF exists locally, which means the KV cache cannot be computed: layer counts
and head dimensions live inside the file. It therefore reports a *bounded
allowance* and says so, rather than inventing a precise-looking number that
turns into a load rejection later.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import os
import random
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast

import httpx

from studioforge.config import Config
from studioforge.core.gguf import looks_like_mmproj as _looks_like_mmproj
from studioforge.core.hf_search import (
    DEFAULT_HF_ENDPOINT,
    UNKNOWN_QUANT,
    GgufFileInfo,
    LogicalDownload,
    file_url,
    safe_filename,
)
from studioforge.core.planner import Planner, kv_alloc_bytes
from studioforge.db import Database
from studioforge.errors import BadRequestError, ConfigError, StudioForgeError, UpstreamError
from studioforge.logging import get_logger
from studioforge.types import GB, MB, GgufMeta

log = get_logger(__name__)

__all__ = [
    "ADOPT_HASH_MAX_BYTES",
    "DOWNLOAD_MAX_ATTEMPTS",
    "PREDOWNLOAD_KV_CEILING_BYTES",
    "PREDOWNLOAD_KV_FLOOR_BYTES",
    "PREDOWNLOAD_KV_FRACTION",
    "DownloadProgress",
    "DownloadStatus",
    "Downloader",
    "PartFileLockedError",
    "fit_verdict",
    "resolve_download_choice",
]

DownloadStatus = Literal["queued", "running", "paused", "completed", "failed", "canceled"]

#: Statuses whose bytes are still coming: everything a resume-on-restart or an
#: in-flight transfer will eventually write. Used for the disk-space estimate.
_PENDING: Final[tuple[DownloadStatus, ...]] = ("queued", "running", "paused")

#: Statuses a ``start()`` picks back up. ``running`` rows are the crash case:
#: nothing is running after a restart, so a ``running`` row means "we died
#: mid-transfer, continue from whatever the .part file holds".
_RESUMABLE: Final[tuple[DownloadStatus, ...]] = ("queued", "running")

#: GUI/WebSocket emit rate. 4 Hz is fast enough that a progress bar looks live
#: and slow enough that a 40 GB download over an 8 MiB chunk size does not push
#: thousands of frames at every connected client.
_EMIT_INTERVAL_S: Final = 0.25

#: DB persistence rate. Deliberately slower than the emit rate -- each write is
#: an fsync-backed SQLite UPDATE. 1 Hz bounds crash-time accounting loss to one
#: second of transfer, which resume re-derives from the .part size anyway.
_DB_INTERVAL_S: Final = 1.0

#: Rolling window for the speed estimate. Long enough to ride out a stalled
#: chunk, short enough that the ETA reacts when bandwidth actually changes.
_SPEED_WINDOW_S: Final = 5.0

#: Floor on the streaming read size. Below this the per-chunk Python overhead
#: starts to show on a fast link, but it is low enough that a small
#: ``hf.chunk_bytes`` is still honoured rather than silently overridden.
_MIN_CHUNK_BYTES: Final = 16 * 1024

_REHASH_CHUNK: Final = 4 * MB
_CONTENT_RANGE_RE: Final = re.compile(
    r"\Abytes\s+(?P<start>\d+)-(?P<end>\d+)/(?P<total>\d+|\*)\Z", re.I
)

# --- retry policy -----------------------------------------------------------
#
# A 40 GiB transfer runs for an hour over a home link; expecting zero dropped
# connections in that window is not a design, it is a hope. Five attempts with
# an exponentially growing, jittered pause covers the whole realistic range --
# a two-second blip, a router reboot, a rate limit, an antivirus scanner holding
# the .part for a moment -- while still reaching a terminal `failed` inside a
# couple of minutes when the cause is not going away. Every attempt resumes from
# the bytes already on disk, so a retry costs a Range request, not a re-download.
DOWNLOAD_MAX_ATTEMPTS: Final = 5
_RETRY_BASE_S: Final = 2.0
_RETRY_CAP_S: Final = 60.0
#: Jitter band. Without it, a multi-file group whose transfers all died on the
#: same network hiccup would retry in lockstep forever.
_RETRY_JITTER: Final = (0.8, 1.2)

#: A file already at the destination is re-hashed on adoption only below this
#: size. Above it the read alone costs minutes of disk bandwidth on every
#: enqueue of a model the user already has, which is a worse deal than trusting
#: the size -- and the size check is the one that would have caught the
#: 2026-08-18 corruption (22,576,551,872 bytes where 19,270,036,448 were
#: declared). Adoption by size alone is logged as such, never as "verified".
ADOPT_HASH_MAX_BYTES: Final = 2 * GB

#: Byte offset of the exclusive lock taken on a ``.part``. Past the end of any
#: file that will ever exist, so the lock never overlaps real data: on Windows a
#: locked byte range is unreadable to every other handle, and a diagnostic tool
#: reading a 40 GiB partial must not trip over our bookkeeping.
_PART_LOCK_OFFSET: Final = 1 << 62

#: ``O_BINARY`` only exists on Windows; on POSIX the flag is meaningless.
_PART_OPEN_FLAGS: Final = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)

#: Errno values that mean "this will not get better by waiting".
_FATAL_ERRNOS: Final = frozenset(
    {
        getattr(errno, name)
        for name in ("ENOSPC", "EDQUOT", "EFBIG", "EROFS", "ENAMETOOLONG")
        if hasattr(errno, name)
    }
)

# --- pre-download KV allowance ---------------------------------------------
#
# The KV cache needs n_layer, n_head_kv and the head dimensions, all of which
# are GGUF metadata -- unavailable until the file is on disk. These constants
# define a *bounded allowance* instead of a computation. Calibrated against the
# GQA models this project actually serves: at 8k total context, KV lands around
# 2-8% of weight bytes for 7B-123B models at 4-8 bit quants, and the fraction
# grows roughly linearly with context. 20% at the 8k baseline therefore leaves
# real slack, the floor keeps small models from getting an allowance smaller
# than a single layer's cache, and the ceiling stops a 200 GB file from being
# charged 40 GB of imaginary cache on top of an already-hopeless weight total.
PREDOWNLOAD_KV_FRACTION: Final = 0.20
PREDOWNLOAD_KV_FLOOR_BYTES: Final = 1 * GB
PREDOWNLOAD_KV_CEILING_BYTES: Final = 8 * GB
PREDOWNLOAD_CTX_BASELINE: Final = 8192


def _part_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


@dataclass
class DownloadProgress:
    """Snapshot of one file's transfer, as pushed to the GUI."""

    id: str
    group_id: str
    repo_id: str
    filename: str
    status: DownloadStatus
    downloaded_bytes: int
    total_bytes: int
    speed_bps: float
    eta_s: float | None
    error: str | None
    #: Attempt number in flight, 1-based; 0 when no retry has been needed.
    attempt: int = 0
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS
    #: Wall-clock instant of the next attempt while backing off, else ``None``.
    #: Wall clock rather than monotonic because this crosses a process boundary
    #: (HTTP payload, WebSocket frame) where a monotonic number is meaningless.
    next_retry_at: float | None = None
    #: What the last attempt died of, kept even after a later attempt succeeds
    #: so "it stalled twice on the way" is visible rather than folklore.
    last_error: str | None = None
    #: Bytes in the ``.part`` at the moment a transfer failed. This is what a
    #: Resume would continue from, and saying so is the difference between
    #: "click Resume" and "click Resume and lose 19 GB of progress".
    part_bytes: int = 0

    @property
    def percent(self) -> float:
        """Completion 0-100. Zero when the total size is unknown."""
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, 100.0 * self.downloaded_bytes / self.total_bytes)

    @property
    def retry_in_s(self) -> float | None:
        """Seconds until the next attempt, or ``None`` when not backing off."""
        if self.next_retry_at is None:
            return None
        return max(0.0, self.next_retry_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "group_id": self.group_id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "status": self.status,
            "downloaded_bytes": self.downloaded_bytes,
            "total_bytes": self.total_bytes,
            "percent": self.percent,
            "speed_bps": self.speed_bps,
            "eta_s": self.eta_s,
            "error": self.error,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "next_retry_at": self.next_retry_at,
            "retry_in_s": self.retry_in_s,
            "last_error": self.last_error,
            "part_bytes": self.part_bytes,
        }


@dataclass
class _FileState:
    """Durable row plus the runtime bookkeeping that never hits the DB."""

    id: str
    group_id: str
    repo_id: str
    filename: str
    dest: Path
    status: DownloadStatus
    total_bytes: int = 0
    downloaded_bytes: int = 0
    sha256: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    samples: deque[tuple[float, int]] = field(default_factory=deque)
    #: Retry bookkeeping. Runtime only -- a retry that a restart interrupts
    #: should start its budget fresh rather than inherit an exhausted one from
    #: a network outage that ended hours ago.
    attempt: int = 0
    next_retry_at: float | None = None
    last_error: str | None = None
    #: Size of the ``.part`` when this file last failed; 0 when there is none.
    part_bytes: int = 0

    def observe(self, now: float) -> None:
        """Record a speed sample and drop everything outside the window."""
        self.samples.append((now, self.downloaded_bytes))
        cutoff = now - _SPEED_WINDOW_S
        while len(self.samples) > 2 and self.samples[0][0] < cutoff:
            self.samples.popleft()

    @property
    def speed_bps(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        (t0, b0), (t1, b1) = self.samples[0], self.samples[-1]
        span = t1 - t0
        if span <= 0:
            return 0.0
        return max(0.0, (b1 - b0) / span)

    @property
    def eta_s(self) -> float | None:
        speed = self.speed_bps
        if speed <= 0 or self.total_bytes <= 0:
            return None
        remaining = max(0, self.total_bytes - self.downloaded_bytes)
        return remaining / speed

    def snapshot(self) -> DownloadProgress:
        return DownloadProgress(
            id=self.id,
            group_id=self.group_id,
            repo_id=self.repo_id,
            filename=self.filename,
            status=self.status,
            downloaded_bytes=self.downloaded_bytes,
            total_bytes=self.total_bytes,
            speed_bps=self.speed_bps,
            eta_s=self.eta_s,
            error=self.error,
            attempt=self.attempt,
            max_attempts=DOWNLOAD_MAX_ATTEMPTS,
            next_retry_at=self.next_retry_at,
            last_error=self.last_error,
            part_bytes=self.part_bytes,
        )

    def row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repo_id": self.repo_id,
            "filename": self.filename,
            "dest_path": str(self.dest),
            "status": self.status,
            "total_bytes": self.total_bytes,
            "downloaded_bytes": self.downloaded_bytes,
            "sha256": self.sha256,
            "group_id": self.group_id,
            "error": self.error,
            "created_at": self.created_at,
        }


#: Terminal download rows (completed/failed/canceled) older than this are
#: deleted at startup. Long enough that "what did I download last week?" still
#: has an answer in the GUI, short enough that the table cannot grow without
#: bound -- these rows are pure history; the files themselves are untouched.
TERMINAL_ROW_RETENTION_S: float = 30 * 86400.0


class _RangeUnsatisfiable(Exception):
    """HTTP 416: the ``.part`` is at or past the object size. Restart clean."""


class ChecksumMismatchError(StudioForgeError):
    """The downloaded bytes do not match the checksum HuggingFace published."""

    status_code = 502
    error_type = "server_error"
    code = "checksum_mismatch"


class PartFileLockedError(StudioForgeError):
    """Somebody else is already writing this ``.part``.

    Deliberately *not* retryable. A held lock is not a hiccup: it is another
    writer doing exactly what we were about to do, and the only outcome worth
    having is that this one stops. Interleaving is what produced the 22.58 GB
    file on 2026-08-18.
    """

    status_code = 409
    error_type = "invalid_request_error"
    code = "part_file_locked"


class PublishFailedError(StudioForgeError):
    """A complete, verified partial could not be renamed onto its destination.

    Not retryable through the transfer loop (the handle is already closed) and
    not a reason to discard anything: the ``.part`` is kept and the next
    attempt publishes it without a request.
    """

    status_code = 409
    error_type = "invalid_request_error"
    code = "publish_failed"


class TransfersDisabledError(StudioForgeError):
    """This process may not write into the model directory (D24 secondary)."""

    status_code = 409
    error_type = "invalid_request_error"
    code = "instance_secondary"


class _PartFile:
    """Exclusive owner of one ``.part`` for the whole life of a transfer.

    Opened once and locked once. Every read (the resume re-hash), every write
    and the size check that proves completion go through this one handle, so
    there is no window in which a second process can slip in -- not between
    attempts, not during verification, not between the last byte and the rename.

    The lock is an exclusive OS byte-range lock (``msvcrt.locking`` on Windows,
    ``fcntl.flock`` on POSIX) rather than a lock file next to it: the kernel
    releases it however the holder dies, so a crashed download leaves a
    resumable ``.part`` and not a permanently "locked" one.

    Two operations have to happen *after* the handle closes, because Windows
    refuses both while a file is open: deleting a rejected partial, and renaming
    a verified one onto its destination. Both are therefore driven from here --
    :meth:`discard` marks the file for deletion at close, :meth:`publish` closes
    and then renames -- rather than left to a caller that would have to
    understand the ordering.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None
        self._discard = False

    # --- lifecycle ----------------------------------------------------

    def __enter__(self) -> _PartFile:
        fd = os.open(self.path, _PART_OPEN_FLAGS, 0o644)
        try:
            _lock_part_fd(fd)
        except OSError as exc:
            os.close(fd)
            raise PartFileLockedError(
                f"{self.path.name}: another process is writing this file "
                f"({type(exc).__name__}: {exc}). Two StudioForge instances pointed at one "
                "model directory will interleave their downloads into one corrupt file, so "
                "this transfer is refused rather than joined. Stop the other instance and "
                "resume.",
                details={"path": str(self.path)},
            ) from exc
        self._fd = fd
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the lock and the handle, deleting the file if discarded."""
        fd, self._fd = self._fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if self._discard:
            self._discard = False
            self.path.unlink(missing_ok=True)

    # --- state --------------------------------------------------------

    @property
    def fd(self) -> int:
        if self._fd is None:  # pragma: no cover - programming error
            raise RuntimeError("the .part file is not open")
        return self._fd

    @property
    def size(self) -> int:
        """Bytes currently in the file, straight from the descriptor."""
        return os.fstat(self.fd).st_size

    def discard(self) -> None:
        """Mark the partial as garbage; it is deleted when the handle closes.

        Deferred rather than immediate because Windows refuses to unlink a file
        that is still open -- and dropping the lock early to delete it would
        hand the corrupt bytes to whatever is waiting.
        """
        self._discard = True

    # --- I/O ----------------------------------------------------------

    def truncate_to_zero(self) -> None:
        """Throw the partial away without giving up ownership of the name."""
        os.ftruncate(self.fd, 0)
        os.lseek(self.fd, 0, os.SEEK_SET)

    def seek_to(self, offset: int) -> None:
        os.lseek(self.fd, offset, os.SEEK_SET)

    def write(self, chunk: bytes) -> None:
        """Write ALL of ``chunk``, looping on short writes.

        ``os.write`` may write fewer bytes than asked (a signal, a full pipe on
        some filesystems, a network share flushing) and returns the count
        instead of raising. A single unchecked call silently dropped the tail
        of a chunk while the streamed hash and byte counter both moved on -- a
        resumable partial that would then fail its on-disk verification (WP17
        F9). Looping until every byte is on the file is the only correct
        behaviour for a raw descriptor.
        """
        view = memoryview(chunk)
        while view:
            written = os.write(self.fd, view)
            if written <= 0:
                raise OSError(f"os.write wrote {written} bytes to {self.path}")
            view = view[written:]

    def rehash(self, hasher: hashlib._Hash) -> int:
        """Feed the existing bytes into ``hasher``; return how many there were.

        Blocking and deliberately called from a worker thread: a 40 GiB ``.part``
        takes tens of seconds to read, and doing that on the event loop froze
        the gateway hard enough for the watchdog to declare it dead.
        """
        os.lseek(self.fd, 0, os.SEEK_SET)
        size = 0
        while True:
            block = os.read(self.fd, _REHASH_CHUNK)
            if not block:
                break
            hasher.update(block)
            size += len(block)
        return size

    def sync(self) -> None:
        """Push our writes to the disk before anything claims they are there."""
        os.fsync(self.fd)

    def publish(self, dest: Path) -> None:
        """Close the handle, then rename the verified partial onto ``dest``.

        The close has to come first (Windows will not rename an open file), which
        leaves a sub-millisecond window where the name is unlocked. It is the one
        gap the lock cannot cover, and it is survivable: the bytes are already
        verified, and after the rename the ``.part`` name no longer exists, so a
        racing writer that grabbed it finds an empty file and a destination that
        is already complete. The real defence against a second writer is
        :class:`~studioforge.core.instance_lock.InstanceLock`; this is depth.
        """
        self._discard = False
        self.close()
        self.path.replace(dest)


def _lock_part_fd(fd: int) -> None:
    """Exclusive, non-blocking OS lock on ``fd``. Raises ``OSError`` if taken.

    Branching on ``sys.platform`` rather than on a module constant so a type
    checker narrows it: ``fcntl`` does not exist on Windows and ``msvcrt`` does
    not exist on POSIX, and only this form tells the checker which half to read.
    """
    if sys.platform == "win32":
        import msvcrt

        os.lseek(fd, _PART_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        finally:
            os.lseek(fd, 0, os.SEEK_SET)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _retry_after_s(exc: BaseException) -> float | None:
    """``Retry-After`` the upstream asked for, when it named one."""
    if isinstance(exc, StudioForgeError):
        value = exc.details.get("retry_after_s")
        if isinstance(value, int | float) and value >= 0:
            return float(value)
    return None


def _is_transient(exc: BaseException) -> bool:
    """Whether waiting could plausibly change the answer.

    The split matters more than the list. Retrying a 404 five times over two
    minutes turns "that file does not exist" into "the download hung"; *not*
    retrying a read timeout turns a 40 GiB transfer into a coin flip on the
    stability of a home connection. So: transport faults, 5xx, 429 and a
    momentary OS-level block on the ``.part`` (antivirus, indexer, a backup
    agent) are retried; a definite answer from the server, a size or checksum
    mismatch, and a full or read-only disk are not.
    """
    if isinstance(exc, PartFileLockedError | ChecksumMismatchError | PublishFailedError):
        return False
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, StudioForgeError):
        status = exc.details.get("status")
        return isinstance(status, int) and (status == 429 or status >= 500)
    if isinstance(exc, OSError):
        return exc.errno not in _FATAL_ERRNOS
    return False


def _backoff_delay(attempt: int, *, retry_after: float | None = None) -> float:
    """Jittered exponential backoff for ``attempt`` (1-based), in seconds.

    An explicit ``Retry-After`` wins when it is longer than our own delay --
    ignoring it is how a rate limit becomes a ban -- but it is still clamped to
    the cap, because a server asking us to wait an hour is not something a
    progress bar can honour.
    """
    delay = min(_RETRY_CAP_S, _RETRY_BASE_S * (2 ** max(0, attempt - 1)))
    if retry_after is not None:
        delay = max(delay, min(_RETRY_CAP_S, retry_after))
    return delay * random.uniform(*_RETRY_JITTER)  # noqa: S311 - jitter, not crypto


def _describe(exc: BaseException) -> str:
    """One-line rendering of a failure, for the state field and the GUI."""
    if isinstance(exc, StudioForgeError):
        return exc.message
    return f"{type(exc).__name__}: {exc}"


def _file_sha256(path: Path) -> str:
    """sha256 of a file on disk. Blocking; call it from a worker thread."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(_REHASH_CHUNK)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


class Downloader:
    """Manages resumable HTTP downloads of GGUF files, grouped by logical model."""

    def __init__(
        self,
        config: Config,
        db: Database,
        *,
        client: httpx.AsyncClient | None = None,
        on_progress: Callable[[DownloadProgress], None] | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self._endpoint = (endpoint or DEFAULT_HF_ENDPOINT).rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            # No overall timeout: a single GGUF shard can legitimately stream for
            # an hour. The read timeout is what detects a dead connection.
            timeout=httpx.Timeout(None, connect=30.0, read=120.0),
            follow_redirects=True,
        )
        self._files: dict[str, _FileState] = {}
        self._groups: dict[str, list[str]] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._intent: dict[str, str] = {}
        #: Injected by the app: "is this file open by a running model?".
        #: A plain callable rather than a supervisor reference so the
        #: downloader stays testable and free of a circular import.
        self._is_in_use: Callable[[Path], bool] | None = None
        self._subscribers: list[Callable[[DownloadProgress], None]] = []
        if on_progress is not None:
            self._subscribers.append(on_progress)
        self._semaphore = asyncio.Semaphore(max(1, config.hf.max_concurrent_downloads))
        self._started = False
        #: Set by the app when this process is a *secondary* on the data dir
        #: (D24): it may read the queue but never write into the model
        #: directory. Checked by enqueue/resume, which is every way a transfer
        #: starts, so the API, the MCP tool and the GUI are all covered.
        self._transfers_disabled_reason: str | None = None
        #: dest -> file id currently writing that destination, so two groups
        #: that share a file (one mmproj attached to every quant of a repo)
        #: take turns instead of one failing on the other's lock.
        self._writing: dict[Path, tuple[str, asyncio.Event]] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_in_use_check(self, check: Callable[[Path], bool] | None) -> None:
        """Register the "file is open by a loaded model" predicate."""
        self._is_in_use = check

    def disable_transfers(self, reason: str) -> None:
        """Refuse to start any transfer from this process, saying why.

        A secondary instance (another process owns the data directory, D24)
        still composes a Downloader so it can *show* the queue, but it must not
        write into the shared model directory: on 2026-08-18 a second writer
        interleaved into the primary's ``.part`` and published a corrupt 22 GB
        file. Only ``start()`` was skipped for secondaries; ``enqueue`` from
        the API/MCP/GUI still launched a real transfer.
        """
        self._transfers_disabled_reason = reason

    def _check_transfers_allowed(self) -> None:
        if self._transfers_disabled_reason:
            raise TransfersDisabledError(
                f"downloads cannot start from this process: {self._transfers_disabled_reason}"
            )

    async def start(self) -> None:
        """Load persisted rows and resume anything interrupted.

        Rows left in ``running`` are, by definition, orphans: no transfer
        survives process exit. They are folded back to ``queued`` and restarted
        from the ``.part`` size, which is the single source of truth for "how
        much do we actually have" -- the DB's ``downloaded_bytes`` can be up to
        one second stale by design.
        """
        if self._started:
            # Reloading would clobber the runtime state of live transfers.
            log.debug("downloader.already_started")
            return
        # Before loading state, so pruned history is never hydrated at all.
        # Failure is non-fatal: pruning is hygiene, not a startup dependency.
        try:
            self.db.prune_downloads(older_than_s=TERMINAL_ROW_RETENTION_S)
        except Exception as exc:
            log.warning("downloader.prune_failed", error=str(exc))
        try:
            self._load_state()
        except Exception as exc:
            # Latching _started before this meant a failed load was swallowed
            # by the caller and never retried: rows stayed "running" with no
            # task and the queue looked busy forever. Say what happened.
            log.error("downloader.load_state_failed", error=_describe(exc))
            raise
        self._started = True
        groups = [
            gid
            for gid, ids in self._groups.items()
            if any(self._files[i].status in _RESUMABLE for i in ids)
        ]
        for gid in groups:
            for file_id in self._groups[gid]:
                state = self._files[file_id]
                if state.status == "running":
                    self._set_status(state, "queued")
            self._launch(gid)
        if groups:
            log.info("downloader.resumed", groups=len(groups))

    async def stop(self) -> None:
        """Cancel in-flight transfers, leaving every row resumable.

        Partial ``.part`` files are kept on purpose: the next ``start()`` picks
        them up with a ``Range`` request instead of re-downloading tens of GiB.
        """
        for gid in list(self._tasks):
            self._intent[gid] = "stop"
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, callback: Callable[[DownloadProgress], None]) -> Callable[[], None]:
        """Register a progress callback; returns its unsubscribe function."""
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(callback)

        return unsubscribe

    def _emit(self, state: _FileState) -> None:
        """Fan a snapshot out to every subscriber.

        A raising subscriber is logged and dropped rather than allowed to abort
        the transfer: a closed WebSocket must not cost the user a 40 GiB
        download.
        """
        if not self._subscribers:
            return
        progress = state.snapshot()
        for callback in list(self._subscribers):
            try:
                callback(progress)
            except Exception as exc:
                log.warning(
                    "downloader.subscriber_failed",
                    error=f"{type(exc).__name__}: {exc}",
                    download_id=state.id,
                )
                with contextlib.suppress(ValueError):
                    self._subscribers.remove(callback)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def all(self) -> list[DownloadProgress]:
        return [s.snapshot() for s in sorted(self._files.values(), key=lambda s: s.created_at)]

    def active(self) -> list[DownloadProgress]:
        return [p for p in self.all() if p.status in ("queued", "running")]

    def group(self, group_id: str) -> list[DownloadProgress]:
        return [self._files[i].snapshot() for i in self._groups.get(group_id, [])]

    def queued_remaining_bytes(self) -> int:
        """Bytes still to fetch across everything that has not finished.

        The disk-space question the Download tab asks -- "will what I have
        queued still fit?" -- is about the *remainder*, not the totals: 380 GiB
        of queued models with 340 GiB already on disk needs 40 GiB, and quoting
        the full figure would cry wolf on every large queue. A backing-off retry
        still counts (it is ``running``, and it intends to finish); a ``failed``
        or ``canceled`` file does not, because nothing will write those bytes
        until a human presses Resume.
        """
        return sum(
            max(0, state.total_bytes - state.downloaded_bytes)
            for state in self._files.values()
            if state.status in _PENDING
        )

    def group_status(self, group_id: str) -> DownloadStatus:
        """Status of a whole logical download, derived from its files.

        The group is only ``completed`` when every file is: a base model without
        its projector, or missing shard 3 of 5, is not a usable model, and
        reporting it as done would let the registry try to load it.
        """
        statuses = [self._files[i].status for i in self._groups.get(group_id, [])]
        if not statuses:
            return "queued"
        if "failed" in statuses:
            return "failed"
        if "running" in statuses:
            return "running"
        if all(s == "completed" for s in statuses):
            return "completed"
        if "queued" in statuses:
            return "queued"
        if "paused" in statuses:
            return "paused"
        # Nothing active is left. If anything was canceled the group is
        # canceled -- `all(canceled)` missed the ordinary case where one file
        # (e.g. the projector) had already completed before the cancel, and
        # those groups reported "paused", implying they could be resumed.
        if "canceled" in statuses:
            return "canceled"
        return "paused"

    # ------------------------------------------------------------------
    # Destination paths
    # ------------------------------------------------------------------

    def models_dir(self) -> Path:
        if self.config.models.dir is None:
            raise ConfigError(
                "no model directory configured; set 'models.dir' before downloading",
                param="models.dir",
            )
        return Path(self.config.models.dir)

    def dest_for(self, repo_id: str, filename: str) -> Path:
        """``<models.dir>/<publisher>/<repo>/<filename>``.

        Every component is validated by :func:`safe_filename` first. ``repo_id``
        is as untrusted as the filename -- it also comes from the API -- so a
        repo id of ``bartowski/../../../etc`` is refused here rather than
        silently escaping the model directory.
        """
        publisher, _, name = repo_id.partition("/")
        if not name:
            # Unscoped canonical repo (rare for GGUF): keep the three-level
            # layout by repeating the name, so the registry's publisher/repo
            # derivation still works.
            publisher = name = repo_id
        return (
            self.models_dir()
            / safe_filename(publisher)
            / safe_filename(name)
            / safe_filename(filename)
        )

    # ------------------------------------------------------------------
    # Enqueue / control
    # ------------------------------------------------------------------

    async def enqueue(
        self,
        item: LogicalDownload,
        *,
        include_mmproj: bool = True,
        force: bool = False,
    ) -> str:
        """Queue a logical download and return its ``group_id``.

        Files that already exist complete on disk are marked ``completed``
        without a byte of traffic unless ``force=True`` -- re-downloading a
        model the user already has is a multi-GiB mistake, and overwriting it
        while llama-server has it mmapped is worse.
        """
        self._check_transfers_allowed()
        files: list[GgufFileInfo] = list(item.files)
        if include_mmproj and item.mmproj is not None:
            files.append(item.mmproj)
        if not files:
            raise ConfigError(f"logical download {item.group_id} has no files")

        group_id = item.group_id
        ids: list[str] = []
        base_time = time.time()
        for index, info in enumerate(files):
            dest = self.dest_for(item.repo_id, info.filename)
            file_id = f"{group_id}:{info.filename}"
            existing = self._files.get(file_id)
            if existing is not None and existing.status in ("running", "queued"):
                ids.append(file_id)
                continue

            if force and dest.is_file():
                # The docstring above is explicit that overwriting a file while
                # llama-server has it mmapped is worse than refusing -- but the
                # force path did it unconditionally. On POSIX the running model
                # keeps serving from an unlinked inode and its next load fails;
                # on Windows the unlink raises mid-loop, leaving files already
                # queued in the DB while `self._groups[group_id]` is never
                # assigned, so the next start() resurrects downloads nobody
                # asked for. Fail before touching anything.
                if self._is_in_use is not None and self._is_in_use(dest):
                    raise BadRequestError(
                        f"{dest.name} belongs to a model that is loaded right now. "
                        "Unload it first, then download with force again.",
                        code="model_loaded",
                    )
                log.warning("downloader.force_overwrite", download_id=file_id)
                dest.unlink()

            state = _FileState(
                id=file_id,
                group_id=group_id,
                repo_id=item.repo_id,
                filename=info.filename,
                dest=dest,
                status="queued",
                total_bytes=info.size_bytes,
                sha256=info.sha256,
                # Sub-millisecond offsets so the DB's (created_at, id) ordering
                # reproduces enqueue order -- shards before the projector.
                created_at=base_time + index * 1e-3,
            )
            # Adopts an already-present file as completed; see _adopt_complete.
            # `quarantine=False`: enqueueing must not move anything on disk, so
            # a wrong-size file at the destination is simply not adopted here
            # and is dealt with when the transfer actually starts.
            await self._adopt_complete(state, quarantine=False)
            self._files[file_id] = state
            self.db.upsert_download(state.row())
            ids.append(file_id)
            self._emit(state)

        self._groups[group_id] = ids
        self._refuse_if_disk_cannot_hold(ids)
        if any(self._files[i].status in _RESUMABLE for i in ids):
            self._launch(group_id)
        log.info(
            "downloader.enqueued",
            group_id=group_id,
            repo_id=item.repo_id,
            files=len(ids),
            total_bytes=item.total_bytes,
        )
        return group_id

    def _refuse_if_disk_cannot_hold(self, ids: Sequence[str]) -> None:
        """Refuse a queue that provably overruns the disk it lands on.

        Display-only until now (the Download tab's badge said "the download
        will still run"), while the API, the MCP tool and the CLI had no
        check at all: a 40 GiB model queued onto a drive with 12 GiB free ran
        for an hour and failed with ``ENOSPC`` at 30%. The remainder of *this*
        group is added to everything already queued; a report that cannot be
        produced (an unmounted path, an exotic filesystem) never refuses.
        """
        from studioforge.core.diskspace import disk_report

        remaining = sum(
            max(0, self._files[i].total_bytes - self._files[i].downloaded_bytes)
            for i in ids
            if self._files[i].status in _PENDING
        )
        if remaining <= 0:
            return
        try:
            report = disk_report(self.models_dir(), self.queued_remaining_bytes())
        except Exception as exc:  # noqa: BLE001 - a failed report must never refuse
            log.debug("downloader.disk_preflight_unavailable", error=str(exc))
            return
        if report.get("error") is not None:
            return
        after = int(report.get("free_after_queue_bytes") or 0)
        if after >= 0:
            return
        short = -after
        for file_id in ids:
            state = self._files[file_id]
            if state.status in _PENDING:
                state.status = "failed"
                state.error = (
                    f"not enough disk space on {report.get('drive')}: "
                    f"{report.get('free_bytes', 0) / GB:.1f} GiB free, "
                    f"{int(report.get('queued_bytes') or 0) / GB:.1f} GiB queued "
                    f"(this download included) -- {short / GB:.1f} GiB short"
                )
                self._persist_status(state)
                self._emit(state)
        raise BadRequestError(
            f"not enough disk space on {report.get('drive')} for this download: "
            f"{report.get('free_bytes', 0) / GB:.1f} GiB free, "
            f"{int(report.get('queued_bytes') or 0) / GB:.1f} GiB queued including it "
            f"({short / GB:.1f} GiB short). Free space or point models.dir at a bigger "
            "drive, then Resume.",
            code="insufficient_disk",
        )

    async def _adopt_complete(self, state: _FileState, *, quarantine: bool = True) -> bool:
        """Mark an already-present file as completed -- if it really is complete.

        Returns True when adopted. A size of zero from the API means "unknown",
        so an existing non-empty file is adopted on presence alone; that is the
        conservative choice because the alternative is overwriting a file the
        user may already be serving.

        **A wrong-size file is not left where it is.** On 2026-08-18 a corrupt
        22,576,551,872-byte file sat at a destination declared as
        19,270,036,448 bytes. The old code declined to adopt it and moved on,
        which is correct as far as it goes -- but the file kept its ``.gguf``
        name, so the registry scanner found it, registered it as a model, and
        every load and every GGUF test against it failed with something that
        read like a parser bug. It is renamed to ``<dest>.corrupt-<ts>``
        instead: out of the scanner's way, still on disk for a human to look at
        or delete, and never silently destroyed. The download then proceeds
        normally.

        The rename is skipped -- and the whole download refused -- when a loaded
        model has the file open. Renaming a mmapped weight file out from under
        llama-server is exactly the class of damage this method exists to
        prevent.

        ``quarantine=False`` makes the method read-only, for the enqueue path:
        queueing a download must not rearrange the model library before the
        transfer has even been scheduled.
        """
        if not state.dest.is_file():
            return False
        try:
            size = state.dest.stat().st_size
        except OSError:  # pragma: no cover - defensive
            return False
        if size <= 0:
            return False
        declared = state.total_bytes
        if declared > 0 and size != declared:
            if quarantine:
                self._quarantine(
                    state,
                    reason=(
                        f"it holds {size} bytes but {declared} were declared by the repository"
                    ),
                )
            return False
        # Size agrees (or is unknown). A published checksum settles it properly,
        # but only where reading the file back is not itself a multi-minute
        # operation -- see ADOPT_HASH_MAX_BYTES.
        if state.sha256 and declared > 0:
            if size <= ADOPT_HASH_MAX_BYTES:
                actual = await asyncio.to_thread(_file_sha256, state.dest)
                if actual != state.sha256.lower():
                    if quarantine:
                        self._quarantine(
                            state,
                            reason=(
                                f"its sha256 is {actual}, not the {state.sha256.lower()} "
                                "the repository published"
                            ),
                        )
                    return False
            else:
                log.info(
                    "downloader.adopted_by_size_only",
                    download_id=state.id,
                    size_bytes=size,
                    reason=(
                        "the file is larger than the re-hash ceiling, so its size was "
                        "checked against the repository but its checksum was not"
                    ),
                )
        state.downloaded_bytes = size
        if state.total_bytes <= 0:
            state.total_bytes = size
        state.status = "completed"
        state.error = None
        log.info("downloader.already_present", download_id=state.id, size_bytes=size)
        return True

    def _quarantine(self, state: _FileState, *, reason: str) -> None:
        """Rename a file that is at the destination but is not the destination.

        Never deletes: the bytes may be the only copy of something, and a
        product that silently removes 22 GB from a user's library has a worse
        bug than the one it was fixing.
        """
        dest = state.dest
        if self._is_in_use is not None and self._is_in_use(dest):
            raise BadRequestError(
                f"{dest.name} is wrong ({reason}) but a model that is loaded right now has "
                "it open, so it cannot be moved aside. Unload that model, then retry.",
                code="model_loaded",
            )
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        target = dest.with_name(f"{dest.name}.corrupt-{stamp}")
        suffix = 1
        while target.exists():
            target = dest.with_name(f"{dest.name}.corrupt-{stamp}-{suffix}")
            suffix += 1
        try:
            dest.rename(target)
        except OSError as exc:
            # Not fatal: the transfer will overwrite the destination anyway. The
            # loss is that the bad file is gone rather than kept, which is worth
            # a loud line and not worth refusing the download over.
            log.error(
                "downloader.quarantine_failed",
                download_id=state.id,
                path=str(dest),
                error=str(exc),
            )
            return
        log.error(
            "downloader.quarantined",
            download_id=state.id,
            path=str(dest),
            moved_to=str(target),
            reason=reason,
        )

    def _launch(self, group_id: str) -> None:
        if group_id in self._tasks:
            return
        self._intent.pop(group_id, None)
        task = asyncio.create_task(self._run_group(group_id), name=f"download:{group_id}")
        self._tasks[group_id] = task

    async def resume(self, group_id: str) -> None:
        """Re-queue a paused/failed group and restart its transfer."""
        self._check_transfers_allowed()
        ids = self._groups.get(group_id)
        if not ids:
            raise ConfigError(f"unknown download group: {group_id}")
        for file_id in ids:
            state = self._files[file_id]
            if state.status in ("paused", "failed", "canceled"):
                state.error = None
                # A fresh retry budget: the reason this failed may have been
                # fixed between then and the user pressing Resume, and inheriting
                # an exhausted counter would fail the first attempt on principle.
                state.attempt = 0
                state.next_retry_at = None
                state.part_bytes = 0
                self._set_status(state, "queued")
        self._launch(group_id)

    async def pause(self, group_id: str) -> None:
        """Stop a group, keeping the ``.part`` files so resume is cheap."""
        await self._interrupt(group_id, "pause")

    async def cancel(self, group_id: str, *, delete_partial: bool = True) -> None:
        """Stop a group for good, optionally deleting its partial files."""
        await self._interrupt(group_id, "cancel" if delete_partial else "cancel-keep")

    async def _interrupt(self, group_id: str, intent: str) -> None:
        ids = self._groups.get(group_id)
        if ids is None:
            raise ConfigError(f"unknown download group: {group_id}")
        task = self._tasks.get(group_id)
        self._intent[group_id] = intent
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        else:
            # Nothing running (queued or already paused): apply directly.
            self._intent.pop(group_id, None)
            self._finalize_interrupted(group_id, intent)

    def _finalize_interrupted(self, group_id: str, intent: str) -> None:
        targets: dict[str, DownloadStatus] = {
            "pause": "paused",
            "cancel": "canceled",
            "cancel-keep": "canceled",
            "stop": "queued",
        }
        target: DownloadStatus = targets.get(intent, "paused")
        for file_id in self._groups.get(group_id, []):
            state = self._files[file_id]
            if state.status == "completed":
                continue
            if intent == "cancel":
                # Best effort: on Windows the .part may still be open by a
                # sibling group sharing this destination, and a failed unlink
                # must not abort the finalize loop for the rest of the group.
                with contextlib.suppress(OSError):
                    _part_path(state.dest).unlink(missing_ok=True)
                state.downloaded_bytes = 0
            self._set_status(state, target)

    # ------------------------------------------------------------------
    # Transfer
    # ------------------------------------------------------------------

    async def _run_group(self, group_id: str) -> None:
        """Download one group's files sequentially, under the concurrency cap.

        Sequential within a group because the files land in the same directory
        and are almost always one large base file plus a small projector: racing
        them buys nothing and doubles the seek pressure. Groups run in parallel
        up to ``hf.max_concurrent_downloads``.
        """
        try:
            async with self._semaphore:
                for file_id in self._groups.get(group_id, []):
                    state = self._files[file_id]
                    if state.status in ("completed", "canceled", "paused"):
                        continue
                    await self._download_file(state)
                    if state.status == "failed":
                        # Don't burn bandwidth on the rest of a group that can
                        # no longer produce a loadable model.
                        self._note_group_failure(group_id, state)
                        break
        except asyncio.CancelledError:
            intent = self._intent.pop(group_id, "cancel")
            self._finalize_interrupted(group_id, intent)
            raise
        except Exception as exc:  # noqa: BLE001 - the task must not die silently
            # _download_file catches everything a transfer can raise; what
            # reaches here is a bookkeeping failure (a persistence write, a
            # subscriber). Left alone, every remaining file in the group sat
            # in queued/running forever with no task behind it, and
            # /api/update/install refused with downloads_active until a
            # restart. Fail what is left, loudly.
            log.error("downloader.group_task_failed", group_id=group_id, error=_describe(exc))
            for file_id in self._groups.get(group_id, []):
                stranded = self._files.get(file_id)
                if stranded is not None and stranded.status in ("queued", "running"):
                    with contextlib.suppress(Exception):
                        self._fail(stranded, f"download task failed: {_describe(exc)}")
        finally:
            self._tasks.pop(group_id, None)

    def _note_group_failure(self, group_id: str, failed: _FileState) -> None:
        """Say what a group's failure means for the model on disk.

        The projector is fetched last, so "the mmproj failed" usually means the
        base weights are already published: the next scan registers a vision
        model as text-only with nothing in the queue explaining why. Name it.
        """
        others = [
            self._files[i]
            for i in self._groups.get(group_id, [])
            if i != failed.id and self._files[i].status == "completed"
        ]
        if others and _looks_like_mmproj(Path(failed.filename)):
            failed.error = (
                f"{failed.error or 'failed'} -- the model's weights are downloaded, but "
                "without its projector it will register as text-only; Resume this download "
                "to fetch the projector"
            )
            self._persist_status(failed)
            self._emit(failed)

    async def _download_file(self, state: _FileState) -> None:
        """Own the ``.part``, then transfer it -- with retries -- under that lock.

        The lock is taken once, around the whole retry budget, and not per
        attempt: a backoff pause is precisely the window in which a second
        writer would otherwise take the file, and "we let go of it for eight
        seconds" is not a meaningfully better story than not locking at all.
        """
        dest = state.dest
        part = _part_path(dest)
        owns_dest = False
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            await self._wait_for_sibling_writer(state)
            if await self._adopt_complete(state):
                self._persist_status(state)
                self._emit(state)
                return
            self._writing[dest] = (state.id, asyncio.Event())
            owns_dest = True
            with _PartFile(part) as part_file:
                await self._transfer_with_retries(state, part_file)
        except asyncio.CancelledError:
            raise
        except StudioForgeError as exc:
            self._fail(state, exc.message)
        except OSError as exc:
            self._fail(state, self._describe_os_error(state, exc))
        except Exception as exc:
            self._fail(state, _describe(exc))
        finally:
            if owns_dest:
                entry = self._writing.get(dest)
                if entry is not None and entry[0] == state.id:
                    self._writing.pop(dest, None)
                    entry[1].set()

    async def _wait_for_sibling_writer(self, state: _FileState) -> None:
        """Wait while another group in this process writes ``state.dest``.

        One projector is attached to every quant of a repo, so queueing two
        quants queues the same mmproj twice under two group ids. Both ran at
        once; the loser hit its sibling's exclusive ``.part`` lock, and that
        error is deliberately non-retryable and blamed "another StudioForge
        instance" -- false, and it failed the whole second group. Waiting for
        the sibling and then adopting what it published costs nothing.
        """
        while True:
            entry = self._writing.get(state.dest)
            if entry is None or entry[0] == state.id:
                return
            log.info(
                "downloader.waiting_for_sibling",
                download_id=state.id,
                writer=entry[0],
                filename=state.filename,
            )
            await entry[1].wait()

    async def _transfer_with_retries(self, state: _FileState, part_file: _PartFile) -> None:
        """Retry transient failures with jittered backoff, resuming each time.

        The backoff is a plain ``asyncio.sleep`` inside the group's task, which
        is what makes pause and cancel instant: ``_interrupt`` cancels that task
        and the sleep raises ``CancelledError`` on the spot. A sleep implemented
        as a polled deadline would leave the GUI's Pause button doing nothing
        for up to a minute.
        """
        attempt = 0
        while True:
            attempt += 1
            state.attempt = attempt
            state.next_retry_at = None
            try:
                await self._transfer_once(state, part_file)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt >= DOWNLOAD_MAX_ATTEMPTS or not _is_transient(exc):
                    raise
                delay = _backoff_delay(attempt, retry_after=_retry_after_s(exc))
                state.last_error = _describe(exc)
                state.next_retry_at = time.time() + delay
                log.warning(
                    "downloader.retrying",
                    download_id=state.id,
                    attempt=attempt,
                    max_attempts=DOWNLOAD_MAX_ATTEMPTS,
                    delay_s=round(delay, 1),
                    resume_from_bytes=part_file.size,
                    error=state.last_error,
                )
                self._emit(state)
                await asyncio.sleep(delay)
                state.next_retry_at = None
                self._emit(state)
                continue
            state.attempt = 0
            state.next_retry_at = None
            return

    async def _transfer_once(self, state: _FileState, part_file: _PartFile) -> None:
        """One attempt, including the "the server keeps saying 416" fallback."""
        for allow_resume in (True, False):
            try:
                await self._transfer(state, part_file, allow_resume=allow_resume)
                return
            except _RangeUnsatisfiable:
                # The .part is >= the object: it is stale (repo re-upload) or
                # complete-but-unverified. Either way, start clean. Truncating
                # rather than unlinking keeps our exclusive hold on the name.
                log.warning("downloader.range_unsatisfiable", download_id=state.id)
                part_file.truncate_to_zero()
                state.downloaded_bytes = 0
        # Both attempts answered 416, including the one that sent no Range
        # header at all. Falling through left status="running" with no task
        # behind it: `active()` reported it forever and /api/update/install
        # refused every call with downloads_active until a restart.
        raise UpstreamError("the server kept answering HTTP 416 even without a Range request")

    async def _transfer(
        self, state: _FileState, part_file: _PartFile, *, allow_resume: bool
    ) -> None:
        have = 0
        hasher = hashlib.sha256()
        if allow_resume and part_file.size > 0:
            # The stream hash must cover the bytes already on disk, so they are
            # re-read. Cheap next to re-downloading them -- but a 40 GiB .part
            # takes tens of seconds, and doing that inline froze the gateway:
            # every stream, every /health poll, long enough for the watchdog to
            # declare a healthy server dead and "recover" it.
            have = await asyncio.to_thread(part_file.rehash, hasher)
            if have > 0 and state.total_bytes > 0 and have == state.total_bytes:
                # Every declared byte is already here: a crash landed between
                # the last write and the rename. Before this the code asked
                # the server for ``bytes=<total>-``, was answered 416, and
                # truncated a complete, verifiable 19 GB partial to zero one
                # line before it would have proved it. Prove it now, from
                # the hash just computed, and publish without a request.
                if not state.sha256 or hasher.hexdigest() == state.sha256.lower():
                    log.info(
                        "downloader.publishing_complete_partial",
                        download_id=state.id,
                        bytes=have,
                        verified="sha256" if state.sha256 else "length-only",
                    )
                    self._set_status(state, "running")
                    self._finish(state, part_file, have, hasher)
                    return
                # Same length, different bytes: a stale partial from a
                # re-uploaded revision. Start clean, keeping the name.
                log.warning(
                    "downloader.complete_partial_hash_mismatch",
                    download_id=state.id,
                    bytes=have,
                )
                part_file.truncate_to_zero()
                have = 0
                hasher = hashlib.sha256()

        url = file_url(state.repo_id, state.filename, endpoint=self._endpoint)
        headers = self._headers()
        if have > 0:
            headers["Range"] = f"bytes={have}-"

        state.downloaded_bytes = have
        state.samples.clear()
        self._set_status(state, "running")

        async with self._client.stream("GET", url, headers=headers) as response:
            if response.status_code == 416:
                raise _RangeUnsatisfiable
            if response.status_code >= 400:
                await response.aread()
                raise self._http_error(state, response)

            resumed = have > 0 and _range_honoured(response, have)
            if have > 0 and not resumed:
                # 200 (or a 206 starting somewhere else) to a Range request: the
                # server is sending the whole object. Appending would splice a
                # duplicate prefix into a correctly sized, corrupt file.
                log.warning(
                    "downloader.range_ignored",
                    download_id=state.id,
                    status=response.status_code,
                    discarded_bytes=have,
                )
                have = 0
                hasher = hashlib.sha256()
                state.downloaded_bytes = 0

            declared = state.total_bytes
            total = _total_size(response, have, declared)
            if declared > 0 and total > 0 and total != declared:
                # The repo's blob listing and the object being served disagree.
                # Something changed under us (re-upload, wrong revision, a
                # captive-portal HTML body) and either number could be the lie,
                # so refuse rather than write a file we cannot vouch for.
                part_file.discard()
                raise UpstreamError(
                    f"{state.filename}: size mismatch, the repository lists "
                    f"{declared} bytes but the server is sending {total}; refusing to "
                    "write a file that matches neither",
                    details={"declared_bytes": declared, "server_bytes": total},
                )
            if total > 0:
                state.total_bytes = total

            written = have
            now = time.monotonic()
            state.observe(now)
            last_emit = last_db = now
            chunk_size = max(_MIN_CHUNK_BYTES, int(self.config.hf.chunk_bytes))

            # Append after a honoured Range, otherwise start the file again --
            # the same choice the old "ab"/"wb" open made, expressed on a handle
            # we never let go of.
            if resumed:
                part_file.seek_to(have)
            else:
                part_file.truncate_to_zero()

            async for chunk in response.aiter_bytes(chunk_size):
                if not chunk:
                    continue
                part_file.write(chunk)
                hasher.update(chunk)
                written += len(chunk)
                state.downloaded_bytes = written

                now = time.monotonic()
                if now - last_emit >= _EMIT_INTERVAL_S:
                    state.observe(now)
                    self._emit(state)
                    last_emit = now
                if now - last_db >= _DB_INTERVAL_S:
                    # A momentarily locked registry must not fail a transfer
                    # that is 90% done: the row is a progress hint, and the
                    # next tick (or the completion write) refreshes it.
                    try:
                        self.db.update_download_progress(
                            state.id, written, total_bytes=state.total_bytes or None
                        )
                    except Exception as exc:  # noqa: BLE001 - see above
                        log.debug("downloader.progress_write_failed", error=str(exc))
                    last_db = now

        await asyncio.to_thread(part_file.sync)
        self._finish(state, part_file, written, hasher)

    def _finish(
        self, state: _FileState, part_file: _PartFile, written: int, hasher: hashlib._Hash
    ) -> None:
        """Verify on disk, publish, and mark complete -- the tail every path shares.

        A ``publish`` that fails is **not** a transfer failure to retry: the
        bytes are complete and verified, the ``.part`` stays on disk (closed,
        not discarded), and the next attempt -- a Resume, or the next boot --
        publishes it without a request through the complete-partial path in
        :meth:`_transfer`. Retrying it through the transfer loop was worse than
        useless: the handle was already closed, so the retry died on
        ``the .part file is not open`` and the user read a programming error.
        On Windows the usual cause is a model that still has the destination
        mmapped, and the message says so.
        """
        dest = state.dest
        self._verify(state, part_file, written, hasher)
        try:
            part_file.publish(dest)
        except OSError as exc:
            raise PublishFailedError(
                f"{state.filename}: downloaded and verified, but it could not be moved "
                f"into place ({type(exc).__name__}: {exc}). "
                + (
                    "A loaded model has the destination open; unload it, then Resume "
                    "-- the verified partial is kept and will be published without "
                    "re-downloading."
                    if self._is_in_use is not None and self._is_in_use(dest)
                    else "Check the destination is writable, then Resume -- the verified "
                    "partial is kept and will be published without re-downloading."
                ),
                details={"path": str(dest), "errno": exc.errno},
            ) from exc
        state.downloaded_bytes = written
        state.part_bytes = 0
        if state.total_bytes <= 0:
            state.total_bytes = written
        try:
            self.db.update_download_progress(state.id, written, total_bytes=state.total_bytes)
        except Exception as exc:  # noqa: BLE001 - the status write below is the one that counts
            log.debug("downloader.progress_write_failed", error=str(exc))
        state.observe(time.monotonic())
        self._set_status(state, "completed")
        log.info(
            "downloader.completed",
            download_id=state.id,
            repo_id=state.repo_id,
            filename=state.filename,
            bytes=written,
            on_disk_bytes=_stat_size(dest),
            verified="sha256" if state.sha256 else "length-only",
        )

    def _http_error(self, state: _FileState, response: httpx.Response) -> UpstreamError:
        """The error for a >= 400 answer, actionable for the ones a user can fix.

        A gated or private repo answers 401/403 at ``resolve/`` even when its
        listing was public, and the generic "HTTP 403 downloading X" sent
        people to the network tab. The message names the repo page, and says
        whether the problem is a missing token or a token that has not
        accepted the licence -- the two states that need different actions.
        """
        status = response.status_code
        details = {
            "status": status,
            "repo_id": state.repo_id,
            # Carried so the backoff can honour a rate limit instead of
            # guessing a delay the server already named.
            "retry_after_s": _parse_retry_after(response),
        }
        if status in (401, 403):
            page = f"https://huggingface.co/{state.repo_id}"
            if self.config.hf.token:
                action = (
                    "your hf.token is set but was refused: open the model page, accept "
                    "its licence with the account that owns the token, and Resume"
                )
            else:
                action = (
                    "no hf.token is configured: open the model page, accept its licence, "
                    "create a read token at https://huggingface.co/settings/tokens, set "
                    "config key 'hf.token' (Setup tab), and Resume"
                )
            return UpstreamError(
                f"HTTP {status} downloading {state.filename} from {state.repo_id}: the "
                f"repository is gated or private ({page}); {action}",
                details={**details, "code": "gated_repo"},
            )
        return UpstreamError(
            f"HTTP {status} downloading {state.filename} from {state.repo_id}",
            details=details,
        )

    def _verify(
        self, state: _FileState, part_file: _PartFile, written: int, hasher: hashlib._Hash
    ) -> None:
        """Prove the file is complete *on disk*, then let it be published.

        The 2026-08-18 corruption walked straight through the old version of
        this method. It compared the streamed byte counter against the declared
        size and the streamed hash against the published one -- both of which
        described the bytes *this process sent to write()*, not the bytes the
        filesystem ended up holding. With a second writer in the same file,
        those are different things, and the file that got renamed into the
        library was 3.3 GB longer than the number that had just been "verified".

        So the size check now runs against ``fstat`` after an ``fsync``, and it
        is checked twice: against what we streamed (catching a foreign writer,
        a short write, a truncation) and against what the repository declared
        (catching a transfer that ended early). A partial that fails either is
        unknowable garbage -- there is no way to tell which bytes are ours --
        and is deleted rather than kept for a resume that would append to
        rubbish.
        """
        if state.total_bytes > 0 and written != state.total_bytes:
            part_file.discard()
            raise UpstreamError(
                f"{state.filename}: transfer ended at {written} bytes but "
                f"{state.total_bytes} were expected; the partial file was discarded",
                details={"expected_bytes": state.total_bytes, "actual_bytes": written},
            )
        on_disk = part_file.size
        if on_disk != written:
            part_file.discard()
            raise UpstreamError(
                f"{state.filename}: {written} bytes were streamed but the partial file holds "
                f"{on_disk}; something else wrote to it, so it was discarded rather than "
                "published. Check that only one StudioForge instance is running against this "
                "model directory.",
                details={"streamed_bytes": written, "on_disk_bytes": on_disk},
            )
        if state.total_bytes > 0 and on_disk != state.total_bytes:  # pragma: no cover - implied
            part_file.discard()
            raise UpstreamError(
                f"{state.filename}: the partial file holds {on_disk} bytes but "
                f"{state.total_bytes} were declared; discarded.",
                details={"expected_bytes": state.total_bytes, "on_disk_bytes": on_disk},
            )
        if not state.sha256:
            # No published hash: the length is all we can check, and saying so
            # beats implying a verification that did not happen.
            log.info(
                "downloader.length_verified_only",
                download_id=state.id,
                bytes=written,
                on_disk_bytes=on_disk,
                reason="HuggingFace published no sha256 for this blob",
            )
            return
        actual = hasher.hexdigest()
        if actual != state.sha256.lower():
            part_file.discard()
            raise ChecksumMismatchError(
                f"{state.filename}: sha256 mismatch, expected {state.sha256.lower()} "
                f"but got {actual}; the downloaded file was deleted",
                details={"expected_sha256": state.sha256.lower(), "actual_sha256": actual},
            )

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Auth header only. The token never appears in a URL or a log line."""
        headers: dict[str, str] = {}
        token = self.config.hf.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _set_status(self, state: _FileState, status: DownloadStatus) -> None:
        state.status = status
        self._persist_status(state)
        self._emit(state)

    def _persist_status(self, state: _FileState) -> None:
        """Write the row; never let a persistence hiccup lose a state transition.

        The in-memory state is the truth for the running process; the row is
        what the next boot reads. A locked or briefly unwritable registry must
        not turn a completed transfer into a raised exception inside
        ``_fail`` -- which is exactly where it would escape the group task and
        strand every other file in the group.
        """
        try:
            if self.db.get_download(state.id) is None:
                self.db.upsert_download(state.row())
            else:
                self.db.set_download_status(state.id, state.status, error=state.error)
        except Exception as exc:  # noqa: BLE001 - see docstring
            log.warning(
                "downloader.persist_failed",
                download_id=state.id,
                status=state.status,
                error=_describe(exc),
            )

    def _describe_os_error(self, state: _FileState, exc: OSError) -> str:
        """``ENOSPC`` names the drive and how far short it is; the rest is generic."""
        if exc.errno not in (errno.ENOSPC, getattr(errno, "EDQUOT", -1)):
            return _describe(exc)
        try:
            from studioforge.core.diskspace import clear_cache, disk_report

            clear_cache()
            report = disk_report(state.dest.parent, 0)
            free = int(report.get("free_bytes") or 0)
            need = max(0, state.total_bytes - state.downloaded_bytes)
            return (
                f"the disk is full: {report.get('drive')} has {free / GB:.1f} GiB free and "
                f"{state.filename} still needs {need / GB:.1f} GiB. The partial is kept; free "
                f"space (or move models.dir), then Resume."
            )
        except Exception:  # noqa: BLE001 - the report is a courtesy
            return (
                f"the disk is full ({_describe(exc)}); the partial is kept -- free space, "
                "then Resume"
            )

    def _fail(self, state: _FileState, message: str) -> None:
        """Give up on a file, recording what a Resume would actually resume from.

        The ``.part`` size is captured here, once, rather than stat-ed on every
        progress snapshot. It is the answer to the only question a user has in
        front of a failed 19 GB download -- "does Resume continue or start
        over?" -- and on 2026-08-18 the answer was "start over" with nothing in
        the product willing to say so.
        """
        state.error = message
        state.last_error = message
        state.next_retry_at = None
        state.attempt = 0
        state.part_bytes = _stat_size(_part_path(state.dest))
        log.error(
            "downloader.failed",
            download_id=state.id,
            error=message,
            resume_from_bytes=state.part_bytes,
        )
        self._set_status(state, "failed")

    def _load_state(self) -> None:
        """Rebuild in-memory state from the DB.

        Everything needed is on the row: the URL is re-derived from
        ``repo_id`` + ``filename``, so no download-specific column can go stale
        against a future HF URL layout change.
        """
        self._files.clear()
        self._groups.clear()
        for row in self.db.list_downloads():
            group_id = row.get("group_id") or str(row["id"])
            status = cast(DownloadStatus, str(row["status"]))
            state = _FileState(
                id=str(row["id"]),
                group_id=str(group_id),
                repo_id=str(row["repo_id"]),
                filename=str(row["filename"]),
                dest=Path(str(row["dest_path"])),
                status=status,
                total_bytes=int(row.get("total_bytes") or 0),
                downloaded_bytes=int(row.get("downloaded_bytes") or 0),
                sha256=row.get("sha256"),
                error=row.get("error"),
                created_at=float(row.get("created_at") or time.time()),
            )
            self._files[state.id] = state
            self._groups.setdefault(state.group_id, []).append(state.id)


def _stat_size(path: Path) -> int:
    """Size of ``path`` in bytes, or 0 when it is not there."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _parse_retry_after(response: httpx.Response) -> float | None:
    """``Retry-After`` in seconds, when the server sent a parseable one.

    Only the delta-seconds form is honoured. The HTTP-date form is legal but
    rare from object stores, and mis-parsing a date into a 40-year sleep is a
    worse outcome than falling back to our own backoff.
    """
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    try:
        value = float(raw.strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _range_honoured(response: httpx.Response, have: int) -> bool:
    """Whether the server really resumed at ``have``.

    A ``206`` alone is not enough: the ``Content-Range`` start offset must match
    what we already hold, or the appended bytes land at the wrong position.
    """
    if response.status_code != 206:
        return False
    header = response.headers.get("Content-Range")
    if not header:
        # 206 without Content-Range is malformed; refuse to trust it.
        return False
    match = _CONTENT_RANGE_RE.match(header.strip())
    if match is None:
        return False
    return int(match.group("start")) == have


def _total_size(response: httpx.Response, have: int, fallback: int) -> int:
    """Object size in bytes, preferring ``Content-Range`` over ``Content-Length``."""
    header = response.headers.get("Content-Range")
    if header:
        match = _CONTENT_RANGE_RE.match(header.strip())
        if match is not None and match.group("total") != "*":
            return int(match.group("total"))
    length = response.headers.get("Content-Length")
    if length is not None:
        with contextlib.suppress(ValueError):
            return have + int(length)
    return fallback


async def resolve_download_choice(
    config: Config,
    planner: Planner,
    repo_id: str,
    quant: str | None = None,
) -> LogicalDownload:
    """Resolve a repo (+ optional quant) to the LogicalDownload to enqueue.

    Shared by the HTTP ``POST /api/downloads`` route and the MCP
    ``download_model`` tool so both pick quants identically: an explicit quant
    must exist, and with none named the largest quant the planner says fits is
    chosen. Raises :class:`BadRequestError` with the available options when the
    request cannot be satisfied.
    """
    from studioforge.core.hf_search import HfSearch

    search = HfSearch(config)
    try:
        repo = await search.repo_info(repo_id)
    finally:
        await search.aclose()

    options = repo.logical_models()
    if not options:
        raise BadRequestError(f"repo '{repo_id}' has no downloadable GGUF models")
    if quant:
        wanted = quant.upper()
        chosen = next((o for o in options if o.quant.upper() == wanted), None)
        if chosen is None:
            available = ", ".join(sorted({o.quant for o in options}))
            raise BadRequestError(
                f"quant '{quant}' not found in {repo_id}; available: {available}",
                param="quant",
            )
        return chosen
    # No quant named: take the largest that the planner says fits, so the
    # default choice is the best quality this box can actually run.
    fitting = [
        o
        for o in options
        if (fit_verdict(o, planner=planner, siblings=options) or {}).get("verdict")
        in {"fits-one-gpu", "needs-multiple-gpus"}
    ]
    return max(fitting or options, key=lambda o: o.total_bytes)


# ---------------------------------------------------------------------------
# Task C: pre-download fit estimate
# ---------------------------------------------------------------------------


def _kv_allowance(weights_bytes: int, ctx_size: int) -> int:
    """Bounded KV allowance for a model whose GGUF we have not seen.

    Scaled linearly with context against an 8k baseline (KV size *is* linear in
    context), then clamped to [floor, ceiling]. See the module constants for the
    calibration this comes from.
    """
    scale = max(1, ctx_size) / PREDOWNLOAD_CTX_BASELINE
    raw = int(weights_bytes * PREDOWNLOAD_KV_FRACTION * scale)
    return max(PREDOWNLOAD_KV_FLOOR_BYTES, min(PREDOWNLOAD_KV_CEILING_BYTES, raw))


def fit_verdict(
    item: LogicalDownload,
    *,
    planner: Planner,
    ctx_size: int | None = None,
    arch_hint: GgufMeta | None = None,
    siblings: Sequence[LogicalDownload] = (),
) -> dict[str, Any]:
    """Will this download fit in VRAM once it lands? Answered honestly.

    The file is not on disk, so the KV cache cannot be computed: layer counts
    and head dimensions live inside the GGUF. What *is* known is the weight
    total from the API, and that alone settles most of the interesting cases --
    a 200 GB file on a 32 GiB card is hopeless whatever the KV term is.

    So: weights are exact, the compute/CUDA-context terms reuse the planner's
    own calibrated overheads, and the KV term is an explicitly bounded
    allowance (see :func:`_kv_allowance`) unless ``arch_hint`` supplies real
    metadata -- e.g. a different quant of the same model already in the
    registry -- in which case KV is computed properly and ``approximate``
    becomes ``False``. When even the file size is missing the verdict is
    ``"unknown"``: a fit estimate built on a fabricated size is worse than one
    that admits it does not know, because the user acts on it and then eats a
    load rejection an hour of downloading later.

    ``siblings`` are the other :class:`LogicalDownload` entries of the same repo;
    they are what makes ``suggested_quant`` possible, steering the user to a
    smaller quant at pick time rather than after the download.
    """
    planner_cfg = planner.config.planner
    ctx = ctx_size or planner.config.models.default_ctx
    weights = int(item.total_bytes)

    gpus = planner.probe.list_gpus()
    usable = {gpu.index: planner.usable_bytes(gpu) for gpu in gpus}
    largest = max(usable.values(), default=0)
    total_free = sum(usable.values())

    result: dict[str, Any] = {
        "verdict": "unknown",
        "message": "",
        "required_bytes": 0,
        "largest_gpu_free_bytes": largest,
        "total_gpu_free_bytes": total_free,
        "suggested_quant": None,
        "group_id": item.group_id,
        "quant": item.quant,
        "weights_bytes": weights,
        "kv_allowance_bytes": 0,
        "overhead_bytes": 0,
        "ctx_size": ctx,
        "gpu_count": len(gpus),
        "approximate": True,
        "size_known": item.size_known,
    }

    if not item.size_known or weights <= 0:
        result["message"] = (
            f"HuggingFace did not report a size for {item.quant or UNKNOWN_QUANT} in "
            f"{item.repo_id}, so no fit estimate is possible. Sizes come from the "
            "repository's blob listing; try re-fetching the repo details."
        )
        return result

    if not gpus:
        result["message"] = (
            "No GPU was detected, so there is nothing to fit this model into. "
            "StudioForge is GPU-only: there is no CPU fallback."
        )
        return result

    compute = max(
        planner_cfg.compute_overhead_floor_mb * MB,
        int(weights * planner_cfg.compute_overhead_fraction),
    )
    cuda_context = planner_cfg.cuda_context_mb * MB

    if arch_hint is not None and arch_hint.n_layer > 0:
        # The per-layer geometry (iSWA windows, hybrid recurrent layers) --
        # NOT the uniform formula, which sized a hybrid 27B's KV at 8.7 GiB for
        # 32k where the loader allocates ~2 GiB, and so called a quant that fits
        # one card "needs multiple GPUs" right beside a context matrix (built on
        # the planner) saying the opposite.
        kv = kv_alloc_bytes(
            arch_hint,
            ctx_total=ctx,
            kv_k=planner.config.models.default_kv_cache_type,
            kv_v=planner.config.models.default_kv_cache_type,
            parallel=1,
        )
        result["approximate"] = kv <= 0
        if kv <= 0:
            kv = _kv_allowance(weights, ctx)
    else:
        kv = _kv_allowance(weights, ctx)

    required = weights + compute + cuda_context + kv
    result["required_bytes"] = required
    result["kv_allowance_bytes"] = kv
    result["overhead_bytes"] = compute + cuda_context

    caveat = (
        " The KV-cache portion is an approximation until the file is present -- "
        "layer and head counts live inside the GGUF."
        if result["approximate"]
        else ""
    )
    headroom_note = f" ({planner_cfg.headroom_fraction:.0%} VRAM headroom reserved.)"

    if required <= largest:
        result["verdict"] = "fits-one-gpu"
        result["message"] = (
            f"{item.quant} should fit on a single GPU: about {required / GB:.1f} GiB needed "
            f"at ctx {ctx}, {largest / GB:.1f} GiB free on the largest card."
            f"{caveat}{headroom_note}"
        )
        return result

    if len(gpus) > 1 and required <= total_free:
        result["verdict"] = "needs-multiple-gpus"
        result["message"] = (
            f"{item.quant} needs about {required / GB:.1f} GiB at ctx {ctx}, more than the "
            f"{largest / GB:.1f} GiB free on any one card but within the "
            f"{total_free / GB:.1f} GiB free across {len(gpus)} GPUs, so it will be split."
            f"{caveat}{headroom_note}"
        )
        return result

    result["verdict"] = "wont-fit"
    suggestion = _suggest_smaller(item, siblings, budget=total_free, planner=planner, ctx=ctx)
    result["suggested_quant"] = suggestion
    message = (
        f"{item.quant} will not fit: about {required / GB:.1f} GiB needed at ctx {ctx} but only "
        f"{total_free / GB:.1f} GiB free across {len(gpus)} GPU(s)."
    )
    if suggestion is not None:
        message += f" The largest quant in this repo that would fit is {suggestion}."
    else:
        message += (
            " No quant in this repo is small enough; try a smaller model, a smaller "
            "context, or free VRAM by unloading something."
        )
    result["message"] = message + caveat + headroom_note
    return result


def _suggest_smaller(
    item: LogicalDownload,
    siblings: Iterable[LogicalDownload],
    *,
    budget: int,
    planner: Planner,
    ctx: int,
) -> str | None:
    """Largest sibling quant whose projected footprint fits in ``budget``.

    "Largest that fits" rather than "smallest available" on purpose: the point
    is to keep as much quality as the hardware allows, not to push the user to
    IQ1.
    """
    planner_cfg = planner.config.planner
    best: tuple[int, str] | None = None
    for sibling in siblings:
        if sibling.group_id == item.group_id or not sibling.size_known:
            continue
        weights = sibling.total_bytes
        if weights <= 0 or weights >= item.total_bytes:
            continue
        required = (
            weights
            + max(
                planner_cfg.compute_overhead_floor_mb * MB,
                int(weights * planner_cfg.compute_overhead_fraction),
            )
            + planner_cfg.cuda_context_mb * MB
            + _kv_allowance(weights, ctx)
        )
        if required <= budget and (best is None or weights > best[0]):
            best = (weights, sibling.quant)
    return best[1] if best is not None else None
