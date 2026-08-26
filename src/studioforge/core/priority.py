"""Load-priority tiers, and the priority-aware load gate (D46).

Three tiers, given per load request and remembered per model:

* **1 -- active chat.** The model a person is talking to right now. Its load
  outranks everything: it may take the fastest placement even when lower
  tiers must be unloaded to give it, it jumps the machine-wide load queue,
  and while it loads the server refuses new inference for lower-tier models
  so they drain and stop competing for the cards.
* **2 -- dispatched agent.** A model a person sent off to work. Same rights,
  but only over tier 3.
* **3 -- background (the default).** Takes a backseat to everything: it may
  never displace a tier-1 or tier-2 resident, and its traffic is the first
  refused while a higher tier is loading.

An unspecified priority means background, so the minimal load request --
model plus context size -- keeps working unchanged, and every pre-existing
instance behaves exactly as before this field existed.

This is a leaf module below the planner and the manager: stdlib only.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
from collections.abc import AsyncIterator

from studioforge.errors import BadRequestError

#: The model a person is chatting with -- outranks everything.
PRIORITY_CHAT = 1
#: A model a person dispatched to work on something -- outranks background.
PRIORITY_AGENT = 2
#: Everything else, and every load that never said. The pre-D46 behaviour.
PRIORITY_BACKGROUND = 3

PRIORITY_LEVELS: tuple[int, ...] = (PRIORITY_CHAT, PRIORITY_AGENT, PRIORITY_BACKGROUND)


def normalise_priority(value: object) -> int | None:
    """``None`` stays ``None`` (the caller applies its default); else 1, 2 or 3.

    ``bool`` is refused explicitly: ``True`` is an ``int`` in Python and would
    silently become tier 1, which is exactly the tier that must never be
    reached by accident.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise BadRequestError(
            "priority must be 1 (active chat), 2 (dispatched agent) or 3 (background)",
            param="priority",
        )
    if value not in PRIORITY_LEVELS:
        raise BadRequestError(
            f"priority {value} is not a tier: 1 is the active chat model, 2 a "
            f"dispatched agent, 3 (or omitted) a background task",
            param="priority",
        )
    return int(value)


class PriorityLock:
    """An ``asyncio.Lock`` whose waiters are served best tier first.

    D29 serialises loads behind one machine-wide gate, and an
    ``asyncio.Lock`` is strictly FIFO -- so the chat model a person is
    waiting on queued behind whatever background loads got there first.
    This lock keeps D29's one-at-a-time guarantee and changes only the
    order of the queue: lowest tier number first, arrival order within a
    tier. A load already holding the gate is never interrupted -- a
    priority is a place in line, not a licence to cancel someone's spawn.

    Cancellation follows ``asyncio.Lock``'s own discipline: a waiter
    cancelled while queued leaves a cancelled future the release walk
    skips; one cancelled *after* the lock was handed to it releases on the
    way out so the hand-off is never lost.

    The traded-away guarantee is FIFO progress: a sustained stream of
    tier-1/2 loads parks a background waiter indefinitely. That is the
    deliberate D46 trade -- on this rig loads are minutes apart and the
    background waiter's client retries -- accepted rather than papered over
    with an aging scheme nothing yet needs.
    """

    def __init__(self) -> None:
        self._locked = False
        #: ``(priority, arrival, future)`` min-heap; arrival keeps FIFO in-tier.
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []
        self._arrivals = itertools.count()

    def locked(self) -> bool:
        return self._locked

    async def acquire(self, priority: int = PRIORITY_BACKGROUND) -> None:
        if not self._locked:
            self._locked = True
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority, next(self._arrivals), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # release() handed us the lock in the same instant the
                # cancellation landed; pass it on or it is held forever.
                self.release()
            raise

    def release(self) -> None:
        if not self._locked:
            raise RuntimeError("release of an unheld PriorityLock")
        while self._waiters:
            _, _, fut = heapq.heappop(self._waiters)
            if not fut.done():
                # Ownership transfers directly: _locked stays True and the
                # woken waiter's acquire() returns.
                fut.set_result(None)
                return
        self._locked = False

    @contextlib.asynccontextmanager
    async def held(self, priority: int = PRIORITY_BACKGROUND) -> AsyncIterator[None]:
        await self.acquire(priority)
        try:
            yield
        finally:
            self.release()
