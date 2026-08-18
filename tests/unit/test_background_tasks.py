"""Fire-and-forget tasks must be strongly referenced until they finish.

The asyncio event loop keeps only weak references to tasks
(``loop.create_task`` docs warn about exactly this), so a task spawned without
a saved reference can be garbage-collected mid-flight. For the restart
endpoints that failure mode is "the API said 'restarting: true' and nothing
ever restarted" -- invisible except as a mystery.
"""

from __future__ import annotations

import asyncio

from studioforge.api import admin_routes


async def test_restart_tasks_are_strongly_referenced_until_done() -> None:
    release = asyncio.Event()
    ran: list[bool] = []

    async def job() -> None:
        await release.wait()
        ran.append(True)

    task = admin_routes._spawn_restart_task(job())
    assert task in admin_routes._RESTART_TASKS, (
        "no strong reference to the restart task: it can be garbage-collected "
        "before the restart happens"
    )
    release.set()
    await asyncio.wait_for(task, 5.0)
    await asyncio.sleep(0)  # let the done-callback run
    assert ran == [True]
    assert task not in admin_routes._RESTART_TASKS, "finished tasks must not accumulate"


async def test_restart_task_reference_survives_a_forced_gc() -> None:
    """A GC sweep while the task waits must not collect it."""
    import gc

    release = asyncio.Event()
    done: list[bool] = []

    async def job() -> None:
        await release.wait()
        done.append(True)

    admin_routes._spawn_restart_task(job())
    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0.01)
    release.set()
    await asyncio.sleep(0.05)
    assert done == [True], "the restart task was lost before it could run"
