"""GPU leases survive a restart (D61).

The book was in-memory only until 2026-09-10, when ``POST /api/restart/server``
emptied it and two tenants' loads landed on the card ClawForge2 had leased for
ComfyUI. Pinned here, layer by layer: the registry's rows carry every field a
restore needs (the vacate token and the peer included); the book writes
through on every mutation and coalesces touches; a fresh book from the same
store restores byte-for-byte with the vacate window cleared; the rows a
restart must NOT keep (idle past TTL, this server's own benchmark, a card a
newer lease holds, a device this box lacks) are dropped and deleted; a store
that fails never refuses a lease; and the manager restores at start, leaves
the rows in place at stop, and lets the sweep release an idle restored lease.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.core import leases as leases_mod
from studioforge.core.leases import (
    TOUCH_PERSIST_INTERVAL_S,
    DatabaseLeaseStore,
    LeaseBook,
    lease_store_for,
    lease_view,
)
from studioforge.core.manager import ModelManager
from studioforge.db import Database
from studioforge.types import GpuLease
from tests.unit.test_gateway_lifecycle import CountingSupervisor, StubRegistry, make_manager

HOLDER_URL = "http://10.0.0.7:8700/vacate"
TOKEN = "cf2-secret-token"
PEER = "10.0.0.7"

LEASE_COLUMNS = {
    "id",
    "devices_json",
    "holder",
    "model_ids_json",
    "reason",
    "created_at",
    "last_activity_at",
    "idle_ttl_s",
    "priority",
    "vacate_url",
    "vacate_token",
    "holder_peer",
    "updated_at",
}


@pytest.fixture()
def db(tmp_path: Path) -> Iterator[Database]:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


class CountingStore:
    """A store that counts what reaches the disk, over a real one."""

    def __init__(self, inner: DatabaseLeaseStore) -> None:
        self.inner = inner
        self.puts = 0
        self.deletes = 0

    def put(self, lease: GpuLease) -> None:
        self.puts += 1
        self.inner.put(lease)

    def delete(self, lease_id: str) -> None:
        self.deletes += 1
        self.inner.delete(lease_id)

    def load(self) -> list[GpuLease]:
        return self.inner.load()


class BrokenStore:
    """A store whose disk is on fire."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc if exc is not None else OSError("disk on fire")
        self.calls = 0

    def put(self, lease: GpuLease) -> None:
        self.calls += 1
        raise self.exc

    def delete(self, lease_id: str) -> None:
        self.calls += 1
        raise self.exc

    def load(self) -> list[GpuLease]:
        self.calls += 1
        raise self.exc


class _Recorder:
    """A stand-in for the module logger that keeps warnings and swallows the rest."""

    def __init__(self) -> None:
        self.warnings: list[tuple[str, dict[str, Any]]] = []
        self.infos: list[tuple[str, dict[str, Any]]] = []

    def warning(self, event: str, **fields: Any) -> None:
        self.warnings.append((event, fields))

    def info(self, event: str, **fields: Any) -> None:
        self.infos.append((event, fields))

    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *_a, **_k: None


def acquire_full(book: LeaseBook, *, now: float, devices: list[int] | None = None) -> GpuLease:
    """A lease with every optional field set, so a restore has something to lose."""
    return book.acquire(
        devices or [2],
        holder="clawforge2",
        model_ids=["pub/render-helper"],
        reason="ComfyUI render",
        idle_ttl_s=3600.0,
        priority=3,
        vacate_url=HOLDER_URL,
        vacate_token=TOKEN,
        holder_peer=PEER,
        now=now,
    )


def build_manager(
    db: Database | None, *, known_devices: list[int] | None = None, **models_cfg: Any
) -> tuple[ModelManager, CountingSupervisor]:
    """A manager over a REAL registry, the way the app builds one.

    ``known_devices`` stands in for the probe; ``None`` is a manager with no
    planner at all, whose ``_known_devices`` answers ``None`` ("cannot ask").
    """
    config = Config(data_dir="/tmp/sf-lease-persistence")
    config.gateway.ttl_sweep_interval_s = 0.02
    config.models.auto_load_pinned = False
    config.models.preload_default_model = False
    if hasattr(config.planner, "rebalance"):
        config.planner.rebalance = "off"
    for key, value in models_cfg.items():
        setattr(config.models, key, value)
    supervisor = CountingSupervisor()
    planner: Any = None
    if known_devices is not None:
        gpus = [SimpleNamespace(index=index) for index in known_devices]
        planner = SimpleNamespace(probe=SimpleNamespace(list_gpus=lambda: list(gpus)))
    manager = ModelManager(
        config,
        registry=StubRegistry([]),  # type: ignore[arg-type]
        planner=planner,
        supervisor=supervisor,  # type: ignore[arg-type]
        db=db,  # type: ignore[arg-type]
    )
    return manager, supervisor


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------


def test_migration_008_creates_the_lease_table(db: Database) -> None:
    columns = {
        row["name"]: row for row in db.connect().execute("PRAGMA table_info(gpu_leases)").fetchall()
    }
    assert set(columns) == LEASE_COLUMNS
    assert columns["id"]["pk"] == 1
    for required in ("devices_json", "holder", "model_ids_json", "created_at", "updated_at"):
        assert columns[required]["notnull"] == 1, required
    assert columns["idle_ttl_s"]["notnull"] == 0, "NULL is 'until released'"
    assert columns["vacate_token"]["notnull"] == 0 and columns["holder_peer"]["notnull"] == 0


def test_save_lease_upserts_and_list_leases_answers_oldest_first(db: Database) -> None:
    db.save_lease(
        {
            "id": "b",
            "devices": [3],
            "holder": "later",
            "model_ids": [],
            "created_at": 2000.0,
            "last_activity_at": 2000.0,
            "idle_ttl_s": None,
        }
    )
    db.save_lease(
        {
            "id": "a",
            "devices": [1, 0],
            "holder": "earlier",
            "model_ids": ["m/x"],
            "reason": "warm",
            "created_at": 1000.0,
            "last_activity_at": 1000.0,
            "idle_ttl_s": 600,
            "priority": 2,
            "vacate_url": HOLDER_URL,
            "vacate_token": TOKEN,
            "holder_peer": PEER,
        }
    )
    rows = db.list_leases()
    assert [row["id"] for row in rows] == ["a", "b"]
    first = rows[0]
    assert first["devices"] == [1, 0] and first["model_ids"] == ["m/x"]
    assert first["idle_ttl_s"] == 600.0 and first["priority"] == 2
    assert first["vacate_url"] == HOLDER_URL
    assert first["vacate_token"] == TOKEN and first["holder_peer"] == PEER
    assert rows[1]["idle_ttl_s"] is None and rows[1]["reason"] == ""
    assert "devices_json" not in first, "callers never see the _json suffix"

    # The upsert: a later write moves the clocks and keeps the birth.
    db.save_lease(
        {
            "id": "a",
            "devices": [1, 0],
            "holder": "earlier",
            "model_ids": ["m/x"],
            "created_at": 5555.0,
            "last_activity_at": 1500.0,
            "idle_ttl_s": 600,
        }
    )
    again = db.list_leases()[0]
    assert again["last_activity_at"] == 1500.0
    assert again["created_at"] == 1000.0, "created_at survives a re-write (cf. adapter added_at)"

    db.delete_lease("a")
    assert [row["id"] for row in db.list_leases()] == ["b"]
    assert db.clear_leases() == 1
    assert db.list_leases() == []


def test_list_leases_skips_an_unreadable_row(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    """One torn row must not cost the restore every other lease on the box."""
    import studioforge.db as db_mod

    recorder = _Recorder()
    monkeypatch.setattr(db_mod, "log", recorder)
    for lease_id in ("good", "torn"):
        db.save_lease(
            {
                "id": lease_id,
                "devices": [0],
                "holder": "h",
                "model_ids": [],
                "created_at": 1.0,
                "last_activity_at": 1.0,
            }
        )
    db.connect().execute("UPDATE gpu_leases SET devices_json = 'not json' WHERE id = 'torn'")

    rows = db.list_leases()

    assert [row["id"] for row in rows] == ["good"]
    assert [f["lease_id"] for e, f in recorder.warnings if e == "db.lease_row_unreadable"] == [
        "torn"
    ]


# ---------------------------------------------------------------------------
# Write-through
# ---------------------------------------------------------------------------


def test_acquire_writes_a_row_with_every_field_including_token_and_peer(db: Database) -> None:
    book = LeaseBook(DatabaseLeaseStore(db))
    lease = acquire_full(book, now=1000.0)

    rows = db.list_leases()
    assert len(rows) == 1
    row = rows[0]
    updated_at = row.pop("updated_at")
    assert isinstance(updated_at, float) and updated_at > 0
    assert row == {
        "id": lease.id,
        "devices": [2],
        "holder": "clawforge2",
        "model_ids": ["pub/render-helper"],
        "reason": "ComfyUI render",
        "created_at": 1000.0,
        "last_activity_at": 1000.0,
        "idle_ttl_s": 3600.0,
        "priority": 3,
        "vacate_url": HOLDER_URL,
        "vacate_token": TOKEN,
        "holder_peer": PEER,
    }
    assert "vacate_token" not in lease.model_dump(), "still excluded from every API dump"


def test_release_deletes_the_row(db: Database) -> None:
    book = LeaseBook(DatabaseLeaseStore(db))
    lease = acquire_full(book, now=1000.0)
    other = book.acquire([0], holder="other", now=1000.0)

    book.release(lease.id)

    assert [row["id"] for row in db.list_leases()] == [other.id]


def test_the_sweeps_expiry_deletes_the_row(db: Database) -> None:
    manager, _ = build_manager(db)
    lease = manager.leases.acquire([2], holder="crashed", idle_ttl_s=60.0, now=time.time() - 120)
    assert [row["id"] for row in db.list_leases()] == [lease.id]

    manager._expire_leases()

    assert manager.leases.get(lease.id) is None
    assert db.list_leases() == [], "an expired lease leaves no row to be restored"


def test_vacate_marks_are_written_through(db: Database) -> None:
    store = CountingStore(DatabaseLeaseStore(db))
    book = LeaseBook(store)
    lease = acquire_full(book, now=1000.0)
    assert store.puts == 1

    book.mark_vacating(lease.id, requested_by="chat", deadline_s=180.0, now=1001.0)
    assert store.puts == 2
    book.mark_vacate_delivery(
        lease.id, delivered=False, status="ConnectError", quiet_s=180.0, now=1002.0
    )
    assert store.puts == 3
    assert (
        book.mark_vacate_delivery("nope", delivered=True, status="200", quiet_s=180.0, now=1003.0)
        is None
    )
    assert store.puts == 3, "a mark on a lease that is gone writes nothing"


def test_touches_are_coalesced_on_the_five_second_clock(db: Database) -> None:
    """A keep-alive every second is one write per five, never one per second."""
    store = CountingStore(DatabaseLeaseStore(db))
    book = LeaseBook(store)
    lease = acquire_full(book, now=1000.0)
    assert store.puts == 1

    book.touch(lease.id, at=1000.0 + TOUCH_PERSIST_INTERVAL_S)  # exactly the interval: written
    assert store.puts == 2
    book.touch(lease.id, at=1006.0)  # 1 s after the last write: coalesced
    assert store.puts == 2
    assert lease.last_activity_at == 1006.0, "the book itself always moves"
    assert db.list_leases()[0]["last_activity_at"] == 1005.0, "the row lags by less than 5 s"
    book.touch(lease.id, at=1012.0)  # 7 s after the last write: written
    assert store.puts == 3
    assert db.list_leases()[0]["last_activity_at"] == 1012.0

    book.touch(lease.id, at=999.0)  # backwards: no move, no write
    assert store.puts == 3 and lease.last_activity_at == 1012.0
    book.touch(lease.id, at=1012.0)  # unchanged clock: no write
    assert store.puts == 3


def test_a_failed_write_is_retried_by_the_next_touch(db: Database) -> None:
    """The coalescing clock is only set by a write that landed."""
    real = DatabaseLeaseStore(db)
    flaky = BrokenStore()
    book = LeaseBook(flaky)
    lease = book.acquire([2], holder="clawforge2", now=1000.0)
    assert db.list_leases() == [], "the acquire's write failed"

    book.bind_store(real)
    book.touch(lease.id, at=1001.0)  # 1 s later, but nothing was ever written

    assert [row["id"] for row in db.list_leases()] == [lease.id]


# ---------------------------------------------------------------------------
# Restore
# ---------------------------------------------------------------------------


def test_a_fresh_book_from_the_same_store_restores_byte_for_byte(db: Database) -> None:
    first = LeaseBook(DatabaseLeaseStore(db))
    lease = acquire_full(first, now=1000.0)
    first.touch(lease.id, at=1010.0)
    first.mark_vacating(lease.id, requested_by="chat", deadline_s=180.0, now=1010.0)
    assert lease.vacating(1020.0), "the window is open when the 'restart' happens"
    expected = lease.model_dump()

    second = LeaseBook(DatabaseLeaseStore(db))
    restored = second.restore(now=2000.0)

    assert len(restored) == 1
    back = restored[0]
    assert second.get(lease.id) is back and len(second) == 1
    dump = back.model_dump()
    assert dump.pop("restored_at") == 2000.0
    expected.pop("restored_at")
    for field in (
        "vacate_requested_at",
        "vacate_requested_by",
        "vacate_deadline",
        "vacate_reask_at",
        "vacate_delivery",
        "vacate_delivery_status",
    ):
        assert dump[field] is None, f"{field}: a restart ends any ask in flight"
        expected[field] = None
    assert dump == expected, "every standing field and both clocks came back as written"
    assert back.vacate_token == TOKEN, "the token is what lets a vacate be sent after a restart"
    assert back.holder_peer == PEER, "the peer is the proof of holdership the unload routes read"
    assert not back.vacating(2000.0)

    view = lease_view(back, now=2000.0)
    assert view["restored_at"] == 2000.0
    assert view["vacate_registered"] is True and "vacate_url" not in view
    assert "vacate_token" not in view and "holder_peer" not in view
    assert lease_view(lease, now=2000.0)["restored_at"] is None, "a lease this server granted"


def test_restore_is_idempotent_and_never_duplicates_a_standing_lease(db: Database) -> None:
    book = LeaseBook(DatabaseLeaseStore(db))
    lease = acquire_full(book, now=1000.0)

    assert book.restore(now=1001.0) == [], "already standing: nothing to re-enter"
    assert len(book) == 1 and book.get(lease.id) is lease
    assert lease.restored_at is None


def test_expired_rows_are_dropped_and_deleted_at_restore(db: Database) -> None:
    first = LeaseBook(DatabaseLeaseStore(db))
    stale = first.acquire([2], holder="clawforge2", idle_ttl_s=600.0, now=1000.0)
    open_ended = first.acquire([3], holder="training", idle_ttl_s=None, now=1000.0)
    fresh = first.acquire([0], holder="agent", idle_ttl_s=3600.0, now=1000.0)

    second = LeaseBook(DatabaseLeaseStore(db))
    restored = second.restore(now=2000.0)

    assert sorted(lease.id for lease in restored) == sorted([open_ended.id, fresh.id])
    assert second.get(stale.id) is None, "idle 1000 s past a 600 s ttl: the sweep would have had it"
    assert sorted(row["id"] for row in db.list_leases()) == sorted([open_ended.id, fresh.id])


def test_this_servers_own_benchmark_leases_are_not_restored(db: Database) -> None:
    """The benchmarker cannot outlive the process; CrucibleForge can."""
    first = LeaseBook(DatabaseLeaseStore(db))
    run = first.acquire([0, 1], holder="benchmark", idle_ttl_s=3600.0, now=1000.0)
    sweep = first.acquire([2], holder="benchmark:parallel", idle_ttl_s=3600.0, now=1000.0)
    external = first.acquire([3], holder="crucibleforge-judge", idle_ttl_s=3600.0, now=1000.0)

    second = LeaseBook(DatabaseLeaseStore(db))
    restored = second.restore(now=1500.0)

    assert [lease.id for lease in restored] == [external.id]
    assert [row["id"] for row in db.list_leases()] == [external.id]
    assert second.get(run.id) is None and second.get(sweep.id) is None


def test_a_row_whose_cards_a_newer_lease_holds_is_dropped(db: Database) -> None:
    first = LeaseBook(DatabaseLeaseStore(db))
    old = first.acquire([2, 3], holder="before-restart", idle_ttl_s=3600.0, now=1000.0)

    second = LeaseBook(DatabaseLeaseStore(db))
    newer = second.acquire([3], holder="after-restart", idle_ttl_s=3600.0, now=1500.0)
    restored = second.restore(now=1600.0)

    assert restored == []
    assert second.get(old.id) is None and second.get(newer.id) is newer
    assert [row["id"] for row in db.list_leases()] == [newer.id]


def test_restore_without_a_store_is_the_pre_d61_book() -> None:
    book = LeaseBook()
    assert book.store is None
    assert book.restore() == []
    lease = book.acquire([0], holder="x")
    book.touch(lease.id)
    book.release(lease.id)
    assert len(book) == 0


# ---------------------------------------------------------------------------
# A store that fails
# ---------------------------------------------------------------------------


def test_a_store_that_raises_never_breaks_the_book(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(leases_mod, "log", recorder)
    store = BrokenStore()
    book = LeaseBook(store)

    lease = book.acquire([2], holder="clawforge2", now=1000.0)
    assert book.get(lease.id) is lease, "the in-memory book is authoritative"
    book.touch(lease.id, at=1010.0)
    book.mark_vacating(lease.id, requested_by="chat", deadline_s=180.0, now=1010.0)
    other = book.acquire([3], holder="other", now=1010.0)
    book.release(lease.id)
    book.release(other.id)
    assert book.restore(now=1100.0) == []
    assert len(book) == 0
    assert store.calls >= 6, "every mutation reached the store and failed there"

    kinds = [fields["op"] for event, fields in recorder.warnings if "gpu lease store" in event]
    assert kinds == ["put", "delete", "load"], "one warning per failure kind, not per failure"


def test_a_store_that_recovers_and_fails_again_is_heard_again(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(leases_mod, "log", recorder)
    broken = BrokenStore()
    book = LeaseBook(broken)

    book.acquire([0], holder="a", now=1000.0)
    book.acquire([1], holder="b", now=1000.0)
    assert len([w for w in recorder.warnings if w[1].get("op") == "put"]) == 1

    book.bind_store(DatabaseLeaseStore(db))
    book.acquire([2], holder="c", now=1000.0)  # lands: clears the memory of the outage
    book.bind_store(broken)
    book.acquire([3], holder="d", now=1000.0)

    assert len([w for w in recorder.warnings if w[1].get("op") == "put"]) == 2


def test_lease_store_for_is_duck_typed(db: Database) -> None:
    assert lease_store_for(None) is None
    assert lease_store_for(SimpleNamespace()) is None
    assert lease_store_for(SimpleNamespace(latest_benchmark=lambda _m: None)) is None
    assert isinstance(lease_store_for(db), DatabaseLeaseStore)


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


def test_the_manager_binds_a_store_over_a_real_database_and_leaves_stubs_alone(
    db: Database,
) -> None:
    shared = LeaseBook()
    config = Config(data_dir="/tmp/sf-lease-persistence")
    manager = ModelManager(
        config,
        registry=StubRegistry([]),  # type: ignore[arg-type]
        planner=None,  # type: ignore[arg-type]
        supervisor=CountingSupervisor(),  # type: ignore[arg-type]
        db=db,
        leases=shared,
    )
    assert manager.leases is shared, "the book the planner shares is the one mirrored"
    assert isinstance(shared.store, DatabaseLeaseStore)

    stubbed, _ = make_manager([])
    assert stubbed.leases.store is None, "a stub database has no lease table: no mirror, no noise"


def test_unknown_devices_are_dropped_at_manager_start(
    db: Database, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = LeaseBook(DatabaseLeaseStore(db))
    ours = seed.acquire([2], holder="clawforge2", idle_ttl_s=3600.0, now=time.time())
    gone = seed.acquire([5], holder="moved-rig", idle_ttl_s=3600.0, now=time.time())

    import studioforge.core.manager as manager_mod

    recorder = _Recorder()
    monkeypatch.setattr(manager_mod, "log", recorder)
    manager, _ = build_manager(db, known_devices=[0, 1, 2, 3])
    manager._restore_leases()

    assert manager.leases.get(ours.id) is not None
    assert manager.leases.get(gone.id) is None
    assert [row["id"] for row in db.list_leases()] == [ours.id]
    dropped = [f for e, f in recorder.warnings if e.startswith("gpu lease dropped at restore")]
    assert [f["unknown_devices"] for f in dropped] == [[5]]
    restored = [f for e, f in recorder.infos if e == "gpu lease restored"]
    assert [f["lease_id"] for f in restored] == [ours.id]
    assert restored[0]["devices"] == [2] and restored[0]["holder"] == "clawforge2"
    assert restored[0]["expires_in"] is not None and restored[0]["idle_s"] >= 0
    summary = [f for e, f in recorder.infos if e == "gpu leases restored from the registry"]
    assert summary == [{"restored": 1, "dropped": 1}]


def test_a_probe_that_cannot_answer_keeps_every_restored_lease(db: Database) -> None:
    """``None`` known devices is "cannot ask", not "has none" -- like acquire_lease."""
    seed = LeaseBook(DatabaseLeaseStore(db))
    lease = seed.acquire([5], holder="somebody", idle_ttl_s=3600.0, now=time.time())

    manager, _ = build_manager(db)  # no planner: _known_devices() is None
    manager._restore_leases()

    assert manager.leases.get(lease.id) is not None


def test_the_restore_never_raises(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    import studioforge.core.manager as manager_mod

    recorder = _Recorder()
    monkeypatch.setattr(manager_mod, "log", recorder)
    manager, _ = build_manager(db, known_devices=[0, 1, 2, 3])
    manager.leases.bind_store(BrokenStore())
    manager._restore_leases()  # the store's load raises inside the book: swallowed there
    assert len(manager.leases) == 0

    def explode() -> list[GpuLease]:
        raise RuntimeError("restore exploded")

    manager.leases.restore = explode  # type: ignore[method-assign]
    manager._restore_leases()  # ...and a throw from the book itself is caught here
    assert len(manager.leases) == 0
    assert [e for e, _f in recorder.infos if e == "gpu leases restored from the registry"] == [
        "gpu leases restored from the registry"
    ], "the summary line is logged for the clean (empty) restore, not for the failed one"


async def test_manager_start_restores_and_the_sweep_releases_an_idle_restored_lease(
    db: Database,
) -> None:
    """The whole path: rows before the 'restart', a book after it, and the TTL still bounds it."""
    seed = LeaseBook(DatabaseLeaseStore(db))
    kept = acquire_full(seed, now=time.time() - 100)
    stale = seed.acquire([3], holder="dead-holder", idle_ttl_s=60.0, now=time.time() - 120)

    manager, _ = build_manager(db, known_devices=[0, 1, 2, 3])
    await manager.start()
    try:
        assert manager.leases.get(stale.id) is None, "idle past its ttl during the downtime"
        restored = manager.leases.get(kept.id)
        assert restored is not None and restored.restored_at is not None
        assert restored.vacate_token == TOKEN and restored.holder_peer == PEER
        assert manager.leases.blocked_for("some/other-model") == frozenset({2}), (
            "the planner sees the restored card as taken"
        )
        assert [row["id"] for row in db.list_leases()] == [kept.id]

        # The holder never comes back: the sweep releases it and the row goes.
        restored.last_activity_at = time.time() - 4000
        for _ in range(200):
            if manager.leases.get(kept.id) is None:
                break
            await asyncio.sleep(0.01)
        assert manager.leases.get(kept.id) is None, "the sweep never released the restored lease"
        assert db.list_leases() == []
    finally:
        await manager.stop(drain_timeout_s=0.1)


async def test_stop_leaves_the_rows_in_place_and_the_next_start_restores_them(
    db: Database,
) -> None:
    manager, _ = build_manager(db, known_devices=[0, 1, 2, 3])
    await manager.start()
    lease = await manager.acquire_lease([2], holder="clawforge2", reason="ComfyUI render")
    await manager.stop(drain_timeout_s=0.1)
    assert [row["id"] for row in db.list_leases()] == [lease.id], "stop() is not a release"

    replacement, _ = build_manager(db, known_devices=[0, 1, 2, 3])
    await replacement.start()
    try:
        back = replacement.leases.get(lease.id)
        assert back is not None and back.holder == "clawforge2" and back.devices == [2]
        assert back.restored_at is not None
        assert [row["restored_at"] for row in (lease_view(x) for x in replacement.leases.all())]
        replacement.release_lease(lease.id)
        assert db.list_leases() == []
    finally:
        await replacement.stop(drain_timeout_s=0.1)
