"""`load_recommended` plans around a lease instead of refusing itself (D64, CR-9).

The incident, 2026-09-12 22:29-22:36: CUDA 1 -- one of the two RTX 5090s -- was
leased to ClawForge2 for ComfyUI. ``POST /api/models/{id}/load-recommended``
walked its hardware modes, chose ``dual_5090`` (``[0, 1]``), and its own real
load then refused that placement with ``gpu_leased``; the client retried every
60 s and got the identical placement and the identical refusal eight times,
while the 3090 pair sat free and a JIT load in the same minute planned around
the same lease without trouble. The walk's throwaway planner had been built
without the lease book, so every mode looked lease-free. The refusal also told
a caller that had sent ``{ctx_size, priority, max_slots}`` to "load without
the device override".

What these pin:

* a mode that needs a card leased to someone else never wins the walk, and no
  path loads onto the leased card -- the next fitting mode is taken;
* the placement taken around a lease says so on the plan, naming the lease, its
  holder and how it ends, the way the JIT path always did;
* nobody is told to drop a device override they never sent; the advice stays
  for a caller who did send one;
* when no mode fits without the leased card, the refusal names the lease and is
  ``gpu_leased`` only when waiting for the release is what changes the answer;
* a lease granted between the walk and the load walks again rather than failing.
"""

from __future__ import annotations

from typing import Any

import pytest

from studioforge.core.leases import LeaseBook
from studioforge.core.planner import Planner
from studioforge.errors import InsufficientVramError
from studioforge.types import GB, LoadRejected
from tests.unit.test_catalog import dense_meta, record
from tests.unit.test_load_recommended import MODEL, make_manager
from tests.unit.test_planner import StubProbe, gpu, make_config, rig_5090x2_3090x2


def leased_manager(
    *, probe: Any = None, n_ctx_train: int = 131072, devices: list[int] | None = None
) -> tuple[Any, Any, LeaseBook, Any]:
    """The incident's shape: CUDA 1 of two same-generation 5090s leased to ClawForge2."""
    manager, supervisor = make_manager(probe=probe, n_ctx_train=n_ctx_train)
    book = LeaseBook()
    manager.leases = book
    manager.planner.leases = book
    lease = book.acquire(
        devices or [1],
        holder="clawforge2",
        reason="ComfyUI (re-leased)",
        idle_ttl_s=600,
    )
    return manager, supervisor, book, lease


def tight_3090s() -> StubProbe:
    """Both 5090s roomy, both 3090s nearly full: only the 5090 pair reaches a big f16 window."""
    return StubProbe(
        [
            gpu(0, 31.84, 31.0, (12, 0)),
            gpu(1, 31.84, 31.0, (12, 0)),
            gpu(2, 24.0, 2.0, (8, 6)),
            gpu(3, 24.0, 2.0, (8, 6)),
        ]
    )


# ---------------------------------------------------------------------------
# The walk treats the lease as a constraint
# ---------------------------------------------------------------------------


async def test_a_mode_needing_a_card_leased_to_someone_else_does_not_win_the_walk() -> None:
    """The 2026-09-12 outage: dual_5090 won while CUDA 1 was leased, then refused itself.

    With the lease book in the walk, dual_5090 is refused inside the walk and
    the next fitting mode -- the free 3090 pair -- is loaded, first time.
    """
    manager, supervisor, _book, _lease = leased_manager()
    instance = await manager.load_recommended(MODEL, 32768, priority=1, max_slots=1)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]
    assert 1 not in instance.plan.devices, "a lease is a promise, never overridden (D53)"
    assert supervisor.starts == 1


async def test_without_the_lease_the_same_call_still_takes_the_headline_pair() -> None:
    """The lease is the only reason the placement moved -- the walk is otherwise unchanged."""
    manager, _supervisor = make_manager()
    manager.leases = LeaseBook()
    manager.planner.leases = manager.leases
    instance = await manager.load_recommended(MODEL, 32768, priority=1, max_slots=1)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [0, 1]


async def test_the_placement_taken_around_a_lease_says_so_in_its_notes() -> None:
    """A slower placement with no reason attached is how the JIT path's good note got lost.

    The JIT load in the incident wrote "CUDA [1] is leased to clawforge2 ...
    DELETE /api/leases/<id>" on its plan; the walk now writes the same, plus
    which mode it passed over.
    """
    manager, _supervisor, _book, lease = leased_manager()
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None
    notes = " ".join(instance.plan.notes)
    assert "passed over dual_5090" in notes
    assert "took dual_3090" in notes
    assert "clawforge2" in notes
    assert lease.id in notes
    assert f"DELETE /api/leases/{lease.id}" in notes
    assert "600 s" in notes, "how the lease ends"


async def test_a_lease_on_a_card_no_earlier_mode_needed_adds_no_downgrade_note() -> None:
    """Only a mode actually passed over for a lease earns the "passed over" sentence."""
    manager, _supervisor, _book, _lease = leased_manager(devices=[3])
    instance = await manager.load_recommended(MODEL, 32768)
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [0, 1]
    assert not any("passed over" in note for note in instance.plan.notes)


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


async def test_a_walk_refused_by_a_lease_never_advises_dropping_a_device_override() -> None:
    """ClawChat sent only {ctx_size, priority, max_slots}; the advice cost the diagnosis time.

    Here only the leased 5090 pair reaches the f16 window, so the call is
    refused -- naming the lease, saying the pair would fit once it ends, and
    saying nothing about an override nobody sent.
    """
    manager, supervisor, _book, lease = leased_manager(probe=tight_3090s(), n_ctx_train=262144)
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144, kv_min="f16", priority=1, max_slots=1)
    error = excinfo.value
    assert "device override" not in error.message
    assert not any("device override" in s for s in error.details["suggestions"])
    assert not any("device override" in (m["reason"] or "") for m in error.details["modes"])
    assert error.code == "gpu_leased", "waiting for the release is what changes the answer"
    assert error.details["lease"]["id"] == lease.id
    assert error.details["retry_after_s"]
    assert "clawforge2" in error.message
    assert "dual_5090, all_gpus would fit at this context once that lease ends" in error.message
    modes = {m["mode"]: m for m in error.details["modes"]}
    assert modes["dual_5090"]["leased_devices"] == [1]
    assert modes["dual_5090"]["fits_once_lease_released"] is True
    assert modes["dual_3090"]["leased_devices"] is None
    assert supervisor.starts == 0


async def test_when_the_leased_mode_would_not_fit_either_the_lease_is_context_not_the_code() -> (
    None
):
    """D53: a lease beside a genuine shortfall is not a reason to tell a client to wait."""
    manager, _supervisor, _book, lease = leased_manager(
        probe=rig_5090x2_3090x2(1.0), n_ctx_train=262144
    )
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 262144)
    error = excinfo.value
    assert error.code == "insufficient_vram"
    assert error.details["retry_after_s"] is None
    assert "lease" not in error.details
    assert lease.id in error.message, "still named, as context"
    assert error.details["shortfall_bytes"] and error.details["shortfall_bytes"] > 0
    assert error.details["largest_term"] is not None


async def test_every_mode_needing_the_leased_card_is_gpu_leased_with_no_invented_zeros() -> None:
    """All four cards leased: every mode is refused by the lease, none by arithmetic."""
    manager, _supervisor, _book, lease = leased_manager(devices=[0, 1, 2, 3])
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 32768)
    error = excinfo.value
    assert error.code == "gpu_leased"
    assert error.details["lease"]["id"] == lease.id
    assert error.details["shortfall_bytes"] is None
    assert "every placement of this box" in error.message
    assert "unload something" not in error.message


# ---------------------------------------------------------------------------
# The planner's words
# ---------------------------------------------------------------------------


def _forced(devices: list[int]) -> Any:
    rec = record(MODEL, dense_meta(32768), mtime=1.0, size_bytes=8 * GB)
    return rec.model_copy(
        update={"settings": rec.settings.model_copy(update={"device_override": devices})}
    )


def test_a_caller_forced_placement_onto_a_lease_still_gets_the_override_advice() -> None:
    """The advice is right for a caller that sent ``devices`` -- only that caller."""
    book = LeaseBook()
    book.acquire([1], holder="clawforge2")
    planner = Planner(make_config(), rig_5090x2_3090x2(), log_plans=False, leases=book)

    by_caller = planner.plan_load(_forced([0, 1]), ctx_size=8192)
    assert isinstance(by_caller, LoadRejected)
    assert by_caller.reason_code == "gpu_leased"
    assert any("device override" in s for s in by_caller.suggestions)
    assert "the requested placement names CUDA [1]" in by_caller.reason

    by_server = planner.plan_load(_forced([0, 1]), ctx_size=8192, override_chosen_by_server=True)
    assert isinstance(by_server, LoadRejected)
    assert by_server.reason_code == "gpu_leased"
    assert not any("device override" in s for s in by_server.suggestions)
    assert "requested" not in by_server.reason
    assert "leased" in by_server.reason, "substring matchers keep working (D53)"


# ---------------------------------------------------------------------------
# A lease granted between the walk and the load
# ---------------------------------------------------------------------------


async def test_a_lease_granted_between_the_walk_and_the_load_walks_again_instead_of_failing() -> (
    None
):
    """The commit-time fallback: the second walk sees the new lease and takes the next mode."""
    manager, supervisor = make_manager()
    book = LeaseBook()
    manager.leases = book
    manager.planner.leases = book
    real = manager._decide_recommended
    calls: list[Any] = []

    def racing(prep: Any, **kwargs: Any) -> Any:
        decision = real(prep, **kwargs)
        calls.append(decision)
        if len(calls) == 1:
            book.acquire([1], holder="clawforge2", reason="ComfyUI")
        return decision

    manager._decide_recommended = racing  # type: ignore[method-assign]
    instance = await manager.load_recommended(MODEL, 32768)
    assert [sorted(c.winner["devices"]) for c in calls] == [[0, 1], [2, 3]]
    assert instance.plan is not None
    assert sorted(instance.plan.devices) == [2, 3]
    assert supervisor.starts == 1


async def test_a_placement_leased_away_twice_is_refused_as_the_walks_not_the_callers() -> None:
    """Two races in a row: refuse, but name the placement as this call's own choice."""
    manager, supervisor = make_manager()
    book = LeaseBook()
    manager.leases = book
    manager.planner.leases = book
    real = manager._decide_recommended
    grabs = iter([[1], [3]])

    def racing(prep: Any, **kwargs: Any) -> Any:
        decision = real(prep, **kwargs)
        book.acquire(next(grabs), holder="clawforge2")
        return decision

    manager._decide_recommended = racing  # type: ignore[method-assign]
    with pytest.raises(InsufficientVramError) as excinfo:
        await manager.load_recommended(MODEL, 32768)
    error = excinfo.value
    assert error.code == "gpu_leased"
    assert "mode walk chose" in error.message
    assert "device override" not in error.message
    assert not any("device override" in s for s in error.details["suggestions"])
    assert supervisor.starts == 0
