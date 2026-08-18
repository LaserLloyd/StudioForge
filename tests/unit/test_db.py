"""Unit tests for studioforge.db: migrations, CRUD, cache keying, threads."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from studioforge.db import SCHEMA_VERSION, Database


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "nested" / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


def _download(id: str = "dl-1", **overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": id,
        "repo_id": "org/repo",
        "filename": "model-Q4_K_M.gguf",
        "dest_path": "E:/models/model-Q4_K_M.gguf",
        "status": "queued",
    }
    record.update(overrides)
    return record


# ---------------------------------------------------------------------------
# Migrations
# ---------------------------------------------------------------------------


def test_migrate_fresh_then_idempotent(tmp_path: Path) -> None:
    database = Database(tmp_path / "deep" / "dir" / "db.sqlite3")
    assert database.schema_version() == 0
    database.migrate()
    assert database.schema_version() == SCHEMA_VERSION

    database.migrate()  # second call must be a no-op
    assert database.schema_version() == SCHEMA_VERSION

    rows = (
        database.connect()
        .execute("SELECT version, name FROM schema_migrations ORDER BY version")
        .fetchall()
    )
    assert len(rows) == SCHEMA_VERSION
    assert rows[0]["version"] == 1
    assert rows[0]["name"] == "001_initial.sql"
    database.close()


def test_migrate_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "db.sqlite3"
    database = Database(path)
    database.migrate()
    assert path.is_file()
    database.close()


# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------


def test_model_settings_roundtrip_and_upsert(db: Database) -> None:
    assert db.get_model_settings("m1") is None

    first = {"ctx_size": 8192, "pinned": True, "adapters": [{"adapter_id": "a1", "scale": 0.5}]}
    db.save_model_settings("m1", first)
    got = db.get_model_settings("m1")
    assert got == first  # JSON round-trips, nested structures intact

    second = {"ctx_size": 4096, "pinned": False}
    db.save_model_settings("m1", second)  # upsert: second wins, still one row
    assert db.get_model_settings("m1") == second
    count = db.connect().execute("SELECT COUNT(*) AS n FROM model_settings").fetchone()["n"]
    assert count == 1

    db.save_model_settings("m2", {"ctx_size": 2048})
    everything = db.all_model_settings()
    assert set(everything) == {"m1", "m2"}
    assert everything["m1"] == second
    assert "settings_json" not in everything["m1"]

    db.delete_model_settings("m1")
    assert db.get_model_settings("m1") is None
    assert set(db.all_model_settings()) == {"m2"}


# ---------------------------------------------------------------------------
# Virtual models
# ---------------------------------------------------------------------------


def test_virtual_models_crud(db: Database) -> None:
    adapters = [{"adapter_id": "lora-1", "scale": 0.8}]
    db.save_virtual_model("vm1", "base-1", "My Tuned", adapters)
    listed = db.list_virtual_models()
    assert len(listed) == 1
    row = listed[0]
    assert row["id"] == "vm1"
    assert row["base_model_id"] == "base-1"
    assert row["name"] == "My Tuned"
    assert row["adapters"] == adapters  # parsed, un-suffixed
    assert "adapters_json" not in row
    assert isinstance(row["created_at"], float)

    # Upsert: same id saved again -> one row, second wins.
    db.save_virtual_model("vm1", "base-2", "Renamed", [])
    listed = db.list_virtual_models()
    assert len(listed) == 1
    assert listed[0]["base_model_id"] == "base-2"
    assert listed[0]["adapters"] == []

    db.delete_virtual_model("vm1")
    assert db.list_virtual_models() == []


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


def test_adapters_crud_and_upsert(db: Database) -> None:
    record = {
        "id": "ad1",
        "name": "style-lora",
        "path": "E:/loras/style.gguf",
        "size_bytes": 123456,
        "base_architecture": "llama",
        "rank": 16,
    }
    db.save_adapter(record)
    got = db.get_adapter("ad1")
    assert got is not None
    assert got["name"] == "style-lora"
    assert got["path"] == "E:/loras/style.gguf"
    assert got["size_bytes"] == 123456
    assert got["base_architecture"] == "llama"
    assert got["rank"] == 16
    assert got["publisher"] is None  # omitted optional column
    assert isinstance(got["added_at"], float)

    # Upsert keyed on id: second save wins, still one row.
    db.save_adapter({**record, "name": "renamed", "rank": 32})
    listed = db.list_adapters()
    assert len(listed) == 1
    assert listed[0]["name"] == "renamed"
    assert listed[0]["rank"] == 32

    db.delete_adapter("ad1")
    assert db.get_adapter("ad1") is None
    assert db.list_adapters() == []


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_download_upsert_and_get(db: Database) -> None:
    db.upsert_download(_download(total_bytes=1000, group_id="g1"))
    got = db.get_download("dl-1")
    assert got is not None
    assert got["repo_id"] == "org/repo"
    assert got["status"] == "queued"
    assert got["total_bytes"] == 1000
    assert got["group_id"] == "g1"
    created = got["created_at"]

    # Re-upsert (startup reconciliation): one row, fields updated, created_at kept.
    db.upsert_download(_download(status="running", total_bytes=2000, created_at=created))
    rows = db.list_downloads()
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["total_bytes"] == 2000
    assert rows[0]["created_at"] == created


def test_download_status_validation(db: Database) -> None:
    with pytest.raises(ValueError):
        db.upsert_download(_download(status="exploded"))
    db.upsert_download(_download())
    with pytest.raises(ValueError):
        db.set_download_status("dl-1", "exploded")


def test_list_downloads_filters(db: Database) -> None:
    db.upsert_download(_download("d1", status="queued", group_id="g1"))
    db.upsert_download(_download("d2", status="running", group_id="g1"))
    db.upsert_download(_download("d3", status="completed", group_id="g2"))
    db.upsert_download(_download("d4", status="failed"))

    assert {r["id"] for r in db.list_downloads()} == {"d1", "d2", "d3", "d4"}
    assert {r["id"] for r in db.list_downloads(statuses=["queued", "running"])} == {"d1", "d2"}
    assert {r["id"] for r in db.list_downloads(group_id="g1")} == {"d1", "d2"}
    assert {r["id"] for r in db.list_downloads(statuses=["running"], group_id="g1")} == {"d2"}
    assert db.list_downloads(statuses=["paused"]) == []


def test_download_progress_and_status(db: Database) -> None:
    db.upsert_download(_download())
    before = db.get_download("dl-1")
    assert before is not None

    db.update_download_progress("dl-1", 512)
    got = db.get_download("dl-1")
    assert got is not None
    assert got["downloaded_bytes"] == 512
    assert got["total_bytes"] is None

    db.update_download_progress("dl-1", 1024, total_bytes=4096)
    got = db.get_download("dl-1")
    assert got is not None
    assert got["downloaded_bytes"] == 1024
    assert got["total_bytes"] == 4096
    assert got["updated_at"] >= before["updated_at"]

    db.set_download_status("dl-1", "failed", error="connection reset")
    got = db.get_download("dl-1")
    assert got is not None
    assert got["status"] == "failed"
    assert got["error"] == "connection reset"

    db.set_download_status("dl-1", "running")  # retry clears the stale error
    got = db.get_download("dl-1")
    assert got is not None
    assert got["error"] is None

    db.delete_download("dl-1")
    assert db.get_download("dl-1") is None


# ---------------------------------------------------------------------------
# GGUF scan cache
# ---------------------------------------------------------------------------


def test_cache_hit_and_invalidation(db: Database) -> None:
    meta = {"architecture": "llama", "n_layer": 32, "quant_label": "Q4_K_M"}
    db.put_cached_meta("E:/models/a.gguf", 1000.5, 4096, meta)

    assert db.get_cached_meta("E:/models/a.gguf", 1000.5, 4096) == meta
    assert db.get_cached_meta("E:/models/a.gguf", 1000.6, 4096) is None  # mtime changed
    assert db.get_cached_meta("E:/models/a.gguf", 1000.5, 4097) is None  # size changed
    assert db.get_cached_meta("E:/models/b.gguf", 1000.5, 4096) is None  # unknown path

    # Overwrite (file re-parsed after change) replaces, not duplicates.
    new_meta = {"architecture": "qwen3", "n_layer": 48}
    db.put_cached_meta("E:/models/a.gguf", 2000.0, 8192, new_meta)
    assert db.get_cached_meta("E:/models/a.gguf", 1000.5, 4096) is None
    assert db.get_cached_meta("E:/models/a.gguf", 2000.0, 8192) == new_meta


def test_prune_cache(db: Database) -> None:
    for name in ("a", "b", "c", "d"):
        db.put_cached_meta(f"E:/models/{name}.gguf", 1.0, 100, {"name": name})

    removed = db.prune_cache(["E:/models/a.gguf", "E:/models/c.gguf"])
    assert removed == 2
    assert db.get_cached_meta("E:/models/a.gguf", 1.0, 100) is not None
    assert db.get_cached_meta("E:/models/b.gguf", 1.0, 100) is None
    assert db.get_cached_meta("E:/models/c.gguf", 1.0, 100) is not None
    assert db.get_cached_meta("E:/models/d.gguf", 1.0, 100) is None

    assert db.prune_cache(["E:/models/a.gguf", "E:/models/c.gguf"]) == 0  # idempotent
    assert db.prune_cache([]) == 2  # empty keep set wipes everything


# ---------------------------------------------------------------------------
# Planner calibration
# ---------------------------------------------------------------------------


def test_load_observations_ordering_and_limit(db: Database) -> None:
    base = time.time()
    for i in range(5):
        db.record_load_observation(
            model_id="m1",
            ts=base + i,
            ctx_size=4096 * (i + 1),
            predicted_bytes=1000 + i,
            actual_bytes=1100 + i,
            ok=i % 2 == 0,
        )
    db.record_load_observation(model_id="m2", ts=base + 10, ok=False, note="oom")

    rows = db.load_observations("m1")
    assert len(rows) == 5
    assert [r["ctx_size"] for r in rows] == [20480, 16384, 12288, 8192, 4096]  # newest first
    assert rows[0]["ok"] is True and rows[1]["ok"] is False  # bool round-trip

    assert len(db.load_observations("m1", limit=2)) == 2
    assert [r["ctx_size"] for r in db.load_observations("m1", limit=2)] == [20480, 16384]

    everything = db.load_observations()
    assert len(everything) == 6
    assert everything[0]["model_id"] == "m2"
    assert everything[0]["note"] == "oom"

    with pytest.raises(ValueError):
        db.record_load_observation(model_id="m1", bogus_field=1)
    with pytest.raises(ValueError):
        db.record_load_observation(ts=base)  # model_id required


def test_load_observation_default_ts(db: Database) -> None:
    before = time.time()
    db.record_load_observation(model_id="m1", ok=True)
    row = db.load_observations("m1")[0]
    assert before <= row["ts"] <= time.time()


# ---------------------------------------------------------------------------
# Throughput calibration
# ---------------------------------------------------------------------------


def _throughput_columns(database: Database) -> set[str]:
    return {
        row["name"]
        for row in database.connect().execute("PRAGMA table_info(throughput_observations)")
    }


def test_a_fresh_database_has_the_estimator_version_column(db: Database) -> None:
    """Migration 004. Without it, record_throughput_observation would raise on
    every sample the moment the manager starts stamping the version."""
    assert "estimator_version" in _throughput_columns(db)


def test_throughput_observation_roundtrips_the_estimator_version(db: Database) -> None:
    db.record_throughput_observation(
        model_id="m1",
        ts=1000.0,
        devices="0,1",
        gpu_class="RTX 5090x2",
        ctx_size=32768,
        parallel=1,
        kv_cache_type="f16",
        prompt_tps=1200.0,
        gen_tps=40.0,
        est_prompt_tps=1300.0,
        est_gen_tps=44.0,
        estimator_version=2,
    )
    row = db.throughput_observations("m1")[0]
    assert row["estimator_version"] == 2
    assert row["gen_tps"] == 40.0
    assert row["devices"] == "0,1"


def test_a_pre_004_row_comes_back_with_a_null_version(db: Database) -> None:
    """Rows written before the column existed are kept and readable.

    They are excluded from throughput.calibrate() -- their ratio corrects a
    formula that no longer exists -- but measured_for() still reports them,
    because a measured tokens/second does not expire when our arithmetic does.
    """
    db.record_throughput_observation(model_id="m1", ts=1000.0, gen_tps=40.0)
    rows = db.throughput_observations("m1")
    assert len(rows) == 1
    assert rows[0]["estimator_version"] is None
    assert rows[0]["gen_tps"] == 40.0


def test_a_typo_in_a_throughput_field_raises(db: Database) -> None:
    """A dropped keyword would look exactly like "no data yet"."""
    with pytest.raises(ValueError):
        db.record_throughput_observation(model_id="m1", estimator_verison=2)
    with pytest.raises(ValueError):
        db.record_throughput_observation(estimator_version=2)  # model_id required


def test_throughput_observations_newest_first_and_pruned_per_model(db: Database) -> None:
    for i in range(5):
        db.record_throughput_observation(
            model_id="m1", ts=1000.0 + i, gen_tps=10.0 + i, estimator_version=2
        )
    db.record_throughput_observation(model_id="m2", ts=2000.0, gen_tps=99.0, estimator_version=2)

    rows = db.throughput_observations("m1")
    assert [r["gen_tps"] for r in rows] == [14.0, 13.0, 12.0, 11.0, 10.0]
    assert len(db.throughput_observations("m1", limit=2)) == 2
    assert db.throughput_observations()[0]["model_id"] == "m2"

    assert db.prune_throughput_observations(keep_per_model=2) == 3
    assert len(db.throughput_observations("m1")) == 2
    assert len(db.throughput_observations("m2")) == 1


# ---------------------------------------------------------------------------
# Key/value
# ---------------------------------------------------------------------------


def test_kv_roundtrip(db: Database) -> None:
    assert db.get_meta("missing") is None
    assert db.get_meta("missing", "fallback") == "fallback"

    db.set_meta("engine_tag", "b10425")
    assert db.get_meta("engine_tag") == "b10425"

    db.set_meta("engine_tag", "b10500")  # upsert
    assert db.get_meta("engine_tag") == "b10500"

    db.delete_meta("engine_tag")
    assert db.get_meta("engine_tag") is None
    db.delete_meta("engine_tag")  # deleting a missing key is a no-op


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_writes_and_reads(db: Database) -> None:
    """8 threads interleaving writes and reads: no sqlite3 errors, exact counts."""
    threads_n = 8
    iterations = 25
    errors: list[BaseException] = []
    barrier = threading.Barrier(threads_n)

    def worker(worker_id: int) -> None:
        try:
            barrier.wait(timeout=30)
            dl_id = f"dl-{worker_id}"
            db.upsert_download(_download(dl_id, group_id="load-test"))
            for i in range(iterations):
                db.update_download_progress(dl_id, i * 1024, total_bytes=iterations * 1024)
                db.set_meta(f"worker-{worker_id}", str(i))
                db.record_load_observation(model_id=f"m-{worker_id}", ok=True, ctx_size=i)
                db.put_cached_meta(f"E:/models/w{worker_id}-{i}.gguf", float(i), i, {"i": i})
                # Interleaved reads on the same thread-local connection.
                assert db.get_download(dl_id) is not None
                assert db.get_meta(f"worker-{worker_id}") == str(i)
                db.list_downloads(group_id="load-test")
        except (sqlite3.ProgrammingError, sqlite3.OperationalError) as exc:
            errors.append(exc)
        except BaseException as exc:  # noqa: BLE001 - surface anything to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, f"worker errors: {errors!r}"

    downloads = db.list_downloads(group_id="load-test")
    assert len(downloads) == threads_n
    for row in downloads:
        assert row["downloaded_bytes"] == (iterations - 1) * 1024
    assert len(db.load_observations(limit=10_000)) == threads_n * iterations
    for n in range(threads_n):
        assert db.get_meta(f"worker-{n}") == str(iterations - 1)


# ---------------------------------------------------------------------------
# Download-row pruning (unbounded-growth guard)
# ---------------------------------------------------------------------------


def test_prune_downloads_removes_only_stale_terminal_rows(db: Database) -> None:
    """History rows age out; live intent (paused/queued) never does."""
    old = time.time() - 90 * 86400
    for id, status in (
        ("old-done", "completed"),
        ("old-failed", "failed"),
        ("old-canceled", "canceled"),
        ("old-paused", "paused"),
        ("old-queued", "queued"),
    ):
        db.upsert_download(_download(id, status=status))
        # upsert stamps updated_at = now; age the row directly.
        db.connect().execute("UPDATE downloads SET updated_at = ? WHERE id = ?", (old, id))
    db.upsert_download(_download("new-done", status="completed"))

    removed = db.prune_downloads(older_than_s=30 * 86400)

    assert removed == 3
    remaining = {row["id"] for row in db.list_downloads()}
    assert remaining == {"old-paused", "old-queued", "new-done"}


def test_prune_downloads_noop_on_fresh_rows(db: Database) -> None:
    db.upsert_download(_download("fresh", status="completed"))
    assert db.prune_downloads(older_than_s=30 * 86400) == 0
    assert len(db.list_downloads()) == 1


# ---------------------------------------------------------------------------
# Corruption recovery (startup resilience)
# ---------------------------------------------------------------------------


def test_migrate_with_recovery_is_a_noop_on_a_healthy_db(tmp_path: Path) -> None:
    database = Database(tmp_path / "db.sqlite3")
    assert database.migrate_with_recovery() is False
    assert database.schema_version() == SCHEMA_VERSION
    database.close()


def test_migrate_with_recovery_moves_a_corrupt_file_aside(tmp_path: Path) -> None:
    """A corrupt registry.sqlite3 must degrade the server, not stop the boot.

    The corrupt file is renamed (never deleted -- the user may want to recover
    virtual models from it) and a fresh, migrated database takes its place.
    """
    path = tmp_path / "registry.sqlite3"
    path.write_bytes(b"this is definitely not a sqlite database" * 100)

    database = Database(path)
    assert database.migrate_with_recovery() is True

    assert database.schema_version() == SCHEMA_VERSION
    database.set_meta("alive", "yes")
    assert database.get_meta("alive") == "yes"
    backups = list(tmp_path.glob("registry.sqlite3.corrupt-*"))
    assert len(backups) == 1, "the corrupt file must be kept aside, not deleted"
    assert backups[0].read_bytes().startswith(b"this is definitely not")
    database.close()


def test_migrate_with_recovery_recovers_a_corrupted_existing_db(tmp_path: Path) -> None:
    """Corruption after a healthy run (torn write) also recovers on next boot."""
    path = tmp_path / "registry.sqlite3"
    first = Database(path)
    first.migrate()
    first.set_meta("k", "v")
    first.close()

    # Overwrite the header in place: sqlite now sees a malformed file.
    with path.open("r+b") as handle:
        handle.write(b"\xff" * 64)

    second = Database(path)
    assert second.migrate_with_recovery() is True
    assert second.schema_version() == SCHEMA_VERSION
    assert second.get_meta("k") is None, "recovery starts fresh; old data lives in the backup"
    second.close()


# ---------------------------------------------------------------------------
# WP17 F5: a LOCKED database is not a CORRUPT database
# ---------------------------------------------------------------------------


def test_locked_healthy_db_is_never_replaced(tmp_path: Path, monkeypatch: Any) -> None:
    """Another process holding the registry mid-write raises OperationalError
    ("database is locked") -- a subclass of DatabaseError. Before the fix that
    renamed the LIVE registry to .corrupt-* and started a fresh one. Now: retry
    briefly, then raise, and the file is untouched."""
    from studioforge import db as dbmod

    path = tmp_path / "registry.sqlite3"
    first = Database(path)
    first.migrate()
    first.set_meta("k", "v")
    first.close()

    monkeypatch.setattr(dbmod, "LOCK_RETRY_DELAY_S", 0.0)
    calls = {"n": 0}

    def locked_migrate(self: Any) -> None:
        calls["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(Database, "migrate", locked_migrate)
    second = Database(path)
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        second.migrate_with_recovery()
    assert "NOT being replaced" in str(excinfo.value)
    assert calls["n"] == dbmod.LOCK_RETRY_ATTEMPTS + 1, "the lock is retried, then given up on"
    assert not list(tmp_path.glob("registry.sqlite3.corrupt-*")), "never moved aside"
    second.close()

    monkeypatch.undo()
    third = Database(path)
    third.migrate()
    assert third.get_meta("k") == "v", "the data survived"
    third.close()


def test_lock_that_clears_is_transparent(tmp_path: Path, monkeypatch: Any) -> None:
    from studioforge import db as dbmod

    path = tmp_path / "registry.sqlite3"
    monkeypatch.setattr(dbmod, "LOCK_RETRY_DELAY_S", 0.0)
    real_migrate = Database.migrate
    calls = {"n": 0}

    def flaky_migrate(self: Any) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        real_migrate(self)

    monkeypatch.setattr(Database, "migrate", flaky_migrate)
    database = Database(path)
    assert database.migrate_with_recovery() is False
    assert database.schema_version() == SCHEMA_VERSION
    database.close()


def test_non_corruption_operational_errors_do_not_trigger_recovery(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """'unable to open database file' / 'disk I/O error' are environment faults;
    replacing the file cures nothing and could destroy a good registry."""
    path = tmp_path / "registry.sqlite3"
    Database(path).migrate()

    def io_error(self: Any) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(Database, "migrate", io_error)
    with pytest.raises(sqlite3.OperationalError):
        Database(path).migrate_with_recovery()
    assert not list(tmp_path.glob("registry.sqlite3.corrupt-*"))


def test_error_classifiers() -> None:
    from studioforge.db import is_corruption_error, is_locked_error

    assert is_locked_error(sqlite3.OperationalError("database is locked"))
    assert not is_corruption_error(sqlite3.OperationalError("database is locked"))
    assert is_corruption_error(sqlite3.DatabaseError("file is not a database"))
    assert is_corruption_error(sqlite3.DatabaseError("database disk image is malformed"))
    assert not is_corruption_error(sqlite3.OperationalError("unable to open database file"))
    assert not is_locked_error(ValueError("locked"))
