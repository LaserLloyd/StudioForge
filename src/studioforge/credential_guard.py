"""Lockout for repeated wrong credentials, shared by the API and the watchdog.

The MCP pairing PIN is eight digits -- a keyspace of 10^8. That is a
deliberate trade (you read it off a startup banner and type it into a
connector), and it is only defensible if guessing is *slow*. It was not: the
gate answered a wrong PIN in about 13 ms flat, with no counter and no lockout,
so the whole keyspace was reachable by one machine in hours. The PIN is what
stands in front of ``restart_server``, ``nuke_all_models``, ``rollback_update``,
``reclaim_orphan_engines`` and ``set_config`` on an install with no
``server.api_key`` -- which is the shipped default.

The policy, deliberately shaped so that an operator fat-fingering the PIN is
barely inconvenienced while a script is stopped dead:

* Failures are counted per client address, and forgotten after
  :data:`WINDOW_S` of quiet.
* The first :data:`FREE_ATTEMPTS` wrong answers are free.
* Every failure after that locks the client out for a doubling backoff --
  1s, 2s, 4s ... capped at :data:`MAX_LOCKOUT_S`. A locked-out client is
  refused *before* the comparison runs, so a correct guess arriving during a
  lockout wins nothing.
* Any success clears the client's record.

At the cap that is 300 s per attempt, i.e. under 300 guesses a day from one
address -- roughly a million years for a 10^8 keyspace, against hours before.

This is a leaf module: stdlib only, no imports from the rest of the package.
The watchdog runs as a separate process whose whole point is to work when the
main server does not, and it keeps its import surface minimal; it gets its own
independent instance of this guard, which is correct -- the two surfaces are
separate doors and neither should be able to lock the other.

Not a substitute for a real credential. It is a rate limiter keyed on a
client-supplied address, so a caller with many source addresses degrades it.
Setting ``server.api_key`` is still the answer for anything reachable off the
box; this makes the default install defensible rather than perfect.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

#: Wrong answers allowed before the backoff starts. Three is enough for a
#: mistyped PIN and a retry; the fourth is already a pattern.
FREE_ATTEMPTS = 3

#: A client's failure record is dropped after this long with no failures, so
#: an operator who got it wrong last Tuesday starts clean.
WINDOW_S = 900.0

#: First lockout, doubled on each subsequent failure.
BASE_LOCKOUT_S = 1.0

#: Cap. Long enough to make brute force pointless, short enough that an
#: operator locked out by a stale script is not stuck for the afternoon.
MAX_LOCKOUT_S = 300.0

#: Ceiling on how many client records are tracked at once. Reached only under
#: a spray from many addresses -- exactly when unbounded growth would be a
#: memory-exhaustion bug -- and the oldest records are evicted first.
MAX_TRACKED = 4096


@dataclass
class _Record:
    failures: int = 0
    locked_until: float = 0.0
    last_seen: float = field(default_factory=time.monotonic)


class CredentialGuard:
    """Per-client failure counter with a doubling lockout.

    Thread-safe: the API runs the auth check on the event loop but Starlette's
    ``BaseHTTPMiddleware`` and the GUI's own callers can reach it from worker
    threads, and the watchdog's ASGI wrapper is bare.
    """

    def __init__(
        self,
        *,
        free_attempts: int = FREE_ATTEMPTS,
        window_s: float = WINDOW_S,
        base_lockout_s: float = BASE_LOCKOUT_S,
        max_lockout_s: float = MAX_LOCKOUT_S,
        max_tracked: int = MAX_TRACKED,
    ) -> None:
        self.free_attempts = free_attempts
        self.window_s = window_s
        self.base_lockout_s = base_lockout_s
        self.max_lockout_s = max_lockout_s
        self.max_tracked = max_tracked
        self._records: dict[str, _Record] = {}
        self._lock = threading.Lock()

    # -- queries ---------------------------------------------------------

    def retry_after(self, client: str | None) -> float:
        """Seconds this client must wait, or 0.0 if it may try now.

        ``None`` means "no identifiable peer" -- an in-process call, which
        never crossed a network -- and is never throttled.
        """
        if not client:
            return 0.0
        now = time.monotonic()
        with self._lock:
            record = self._records.get(client)
            if record is None:
                return 0.0
            if now - record.last_seen > self.window_s:
                del self._records[client]
                return 0.0
            remaining = record.locked_until - now
        # Round up: reporting "0 seconds" while still refusing would send a
        # well-behaved client straight back into the wall.
        return max(0.0, remaining)

    # -- updates ---------------------------------------------------------

    def record_failure(self, client: str | None) -> float:
        """Count a wrong credential. Returns the new lockout in seconds."""
        if not client:
            return 0.0
        now = time.monotonic()
        with self._lock:
            record = self._records.get(client)
            if record is None or now - record.last_seen > self.window_s:
                record = _Record()
                self._records[client] = record
            record.failures += 1
            record.last_seen = now
            over = record.failures - self.free_attempts
            if over <= 0:
                lockout = 0.0
            else:
                # 1, 2, 4, 8 ... capped. `min` on the exponent as well as the
                # result: 2 ** (a few thousand) is a real number in Python and
                # computing it would be the denial of service.
                lockout = min(
                    self.max_lockout_s,
                    self.base_lockout_s * (2 ** min(over - 1, 32)),
                )
            record.locked_until = now + lockout
            self._evict_locked()
        return lockout

    def record_success(self, client: str | None) -> None:
        """Forget a client's failures. Called on any accepted credential."""
        if not client:
            return
        with self._lock:
            self._records.pop(client, None)

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    # -- internals -------------------------------------------------------

    def _evict_locked(self) -> None:
        """Drop the least recently seen records once over the cap.

        Called with ``self._lock`` held.
        """
        if len(self._records) <= self.max_tracked:
            return
        ordered = sorted(self._records.items(), key=lambda item: item[1].last_seen)
        for key, _ in ordered[: len(self._records) - self.max_tracked]:
            del self._records[key]


def client_key(host: str | None) -> str | None:
    """Normalise a peer address into a bucket key."""
    text = (host or "").strip().lower()
    return text or None
