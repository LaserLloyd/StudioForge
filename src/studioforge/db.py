"""SQLite persistence with versioned migrations.

This is the durable state that must survive both app restarts and app
self-updates: per-model saved settings, virtual models, LoRA adapters,
download state, a GGUF-metadata scan cache, planner calibration
observations, and a small key/value table.

Design notes:

* **No ORM, deliberately.** The schema is small and the queries are trivial;
  stdlib ``sqlite3`` keeps the dependency surface flat and the behavior
  transparent.
* **WAL journal mode** so the GUI and API can read while the downloader
  writes progress on every chunk -- readers never block on the writer.
* **JSON blob columns** (``settings_json``, ``adapters_json``, ``meta_json``)
  instead of wide typed schemas: the llama-server flag surface changes with
  every engine release, and a column-per-flag layout would need a migration
  each time. Callers never see the ``_json`` suffix; values come back parsed
  under their natural key.
* **Never log row contents at INFO** -- download URLs and repo ids can carry
  tokens. Only ids and counts appear in log lines.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

from studioforge.logging import get_logger

log = get_logger(__name__)

#: Highest migration version this build of the code ships. ``migrate()``
#: brings any older database up to this.
SCHEMA_VERSION: int = 7

#: SQLite's own words for "this file is not a usable database". Only a message
#: carrying one of these is treated as corruption by ``migrate_with_recovery``;
#: everything else (locked, busy, unable to open, disk I/O, read-only) is a
#: condition of the *environment* and replacing the file would destroy a good
#: registry to cure a symptom that was never the file's fault.
CORRUPTION_MARKERS: tuple[str, ...] = (
    "file is not a database",
    "malformed",
    "database disk image",
    "unsupported file format",
    "file is encrypted",
    "not a database",
)

#: SQLite's words for "someone else has it right now". Retried briefly.
LOCK_MARKERS: tuple[str, ...] = ("database is locked", "database is busy", "locked")

#: How long ``migrate_with_recovery`` waits for a lock to clear before it gives
#: up -- and gives up by *raising*, never by replacing the file. Migrations
#: happen once at boot, so a couple of seconds is cheap; a lock that lasts
#: longer means another instance is genuinely running against this data dir.
LOCK_RETRY_ATTEMPTS: int = 6
LOCK_RETRY_DELAY_S: float = 0.5


def is_locked_error(exc: BaseException) -> bool:
    """Whether ``exc`` is SQLite saying the database is held by someone else."""
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in LOCK_MARKERS)


def is_corruption_error(exc: BaseException) -> bool:
    """Whether ``exc`` is SQLite saying the file itself is not a valid database.

    Deliberately narrow. ``sqlite3.DatabaseError`` is also the base class of
    ``OperationalError`` (locks, missing files, I/O errors), so the *type* alone
    cannot tell a broken file from a busy one -- the message can.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in CORRUPTION_MARKERS)


DOWNLOAD_STATUSES: tuple[str, ...] = (
    "queued",
    "running",
    "paused",
    "completed",
    "failed",
    "canceled",
)

_OBSERVATION_COLUMNS: tuple[str, ...] = (
    "model_id",
    "ts",
    "ctx_size",
    "parallel",
    "kv_cache_type",
    "devices",
    "predicted_bytes",
    "actual_bytes",
    "weights_bytes",
    "ok",
    "note",
    # Where the load landed, per CUDA index, as JSON text (migration 006, D40):
    # the plan's share and the child's measured footprint on each card.
    "per_gpu_planned",
    "per_gpu_actual",
    # The V half of the KV cache type (migration 007, D51). NULL on older rows;
    # see :meth:`Database.matching_observation` for what NULL is allowed to
    # match, and why it is not read as "the same as K".
    "kv_cache_type_v",
)

#: The only ``load_observations.note`` whose ``actual_bytes`` is trustworthy:
#: our own child's VRAM, attributed per pid AND per device. Everything before
#: it either summed whole-device usage (contaminated by every other process on
#: the card) or, on Windows, counted one per-process total once per card, so a
#: four-GPU load "measured" four times its footprint. Duplicated here as a
#: literal rather than imported: ``db`` is a leaf module and importing
#: ``core.planner`` for one string would invert the dependency and make the
#: database layer depend on the planner. The authority is
#: :data:`studioforge.core.planner.OBSERVATION_NOTE_PER_PID_DEVICE`, and
#: ``tests/unit/test_planner_observed.py`` asserts the two agree.
OBSERVATION_NOTE_TRUSTED: str = "per_pid_v2"

#: How many of a model's newest observations :meth:`Database.matching_observation`
#: scans for one matching configuration. A model is loaded at a handful of
#: distinct (context, slots, cache, device-count) shapes at most, and the
#: ladder walks contexts downward, so a match that is not in the newest 50 rows
#: is old enough that the engine build or the rig around it has probably moved.
_OBSERVATION_MATCH_WINDOW: int = 50

_THROUGHPUT_COLUMNS: tuple[str, ...] = (
    "model_id",
    "ts",
    "devices",
    "gpu_class",
    "ctx_size",
    "parallel",
    "kv_cache_type",
    "prompt_tps",
    "gen_tps",
    "est_prompt_tps",
    "est_gen_tps",
    "n_busy_slots",
    "requests_deferred",
    "sample_s",
    # Which build of throughput.estimate() produced est_prompt_tps/est_gen_tps.
    # NULL on rows written before migration 004; calibration reads only rows
    # stamped with the current version, so a formula change retires its own
    # history instead of learning from it (see 004_throughput_version.sql).
    "estimator_version",
)

#: One row per (parallel-benchmark run, concurrency level). See
#: ``migrations/005_parallel.sql`` for why each column is stored rather than
#: derived; the short version is that CUDA ordinals, engine builds and KV cache
#: types all move, and an observation that cannot say which of them produced it
#: cannot be retired when one of them changes.
_PARALLEL_COLUMNS: tuple[str, ...] = (
    "model_id",
    "ts",
    "run_id",
    "devices",
    "gpu_class",
    "ctx_per_slot",
    "kv_cache_type",
    "kv_cache_type_v",
    "n_streams",
    "per_stream_tps",
    "aggregate_tps",
    "p95_latency_s",
    "prompt_tokens",
    "completion_tokens",
    "n_busy_slots",
    "engine_tag",
)

_ADAPTER_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "size_bytes",
    "base_architecture",
    "base_model_hint",
    "publisher",
    "repo",
    "n_layer",
    "rank",
)

_DOWNLOAD_OPTIONAL_COLUMNS: tuple[str, ...] = (
    "total_bytes",
    "downloaded_bytes",
    "sha256",
    "etag",
    "error",
    "group_id",
)


def _migration_sources() -> list[tuple[int, str, str]]:
    """All shipped migrations as ``(version, filename, sql)``, sorted.

    Loaded via :mod:`importlib.resources` so a zip/wheel install works, with a
    plain-filesystem fallback for source checkouts where the package metadata
    may not expose the data files.
    """
    entries: dict[int, tuple[str, str]] = {}
    try:
        root = resources.files("studioforge") / "migrations"
        for item in root.iterdir():
            if item.name.endswith(".sql") and item.is_file():
                version = int(item.name.split("_", 1)[0])
                entries[version] = (item.name, item.read_text(encoding="utf-8"))
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError, OSError):
        pass
    if not entries:
        fs_dir = Path(__file__).resolve().parent / "migrations"
        for sql_path in sorted(fs_dir.glob("*.sql")):
            version = int(sql_path.name.split("_", 1)[0])
            entries[version] = (sql_path.name, sql_path.read_text(encoding="utf-8"))
    return [(version, name, sql) for version, (name, sql) in sorted(entries.items())]


def _split_statements(sql: str) -> list[str]:
    """Split a migration script into individual statements.

    ``executescript`` cannot be used because it force-commits any open
    transaction, which would break the migration-per-transaction guarantee.
    This splitter is intentionally simple (strip ``--`` line comments, split
    on ``;``) -- migration files must not embed semicolons inside string
    literals or use BEGIN...END trigger bodies. Ours don't.
    """
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    return [chunk.strip() for chunk in "\n".join(lines).split(";") if chunk.strip()]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Copy a Row into a plain dict so sqlite3 types never leak to callers."""
    return dict(row)


class Database:
    """Thread-safe SQLite wrapper for all StudioForge durable state.

    The app is asyncio plus a threadpool and sqlite3 connections cannot be
    shared across threads, so each thread gets its own connection via
    ``threading.local()``. A single process-wide lock serializes writes;
    reads go lock-free, which WAL makes safe.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._conns_lock = threading.Lock()
        self._conns: list[sqlite3.Connection] = []

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use.

        Opened in autocommit (``isolation_level=None``): every write in this
        module is a single statement (atomic on its own), and migrations
        manage explicit BEGIN/COMMIT -- implicit transaction magic would only
        obscure who holds the write lock when.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._path), check_same_thread=False, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=5000")
        except sqlite3.Error:
            # A corrupt file makes the very first PRAGMA raise. The connection
            # is not yet registered anywhere, so without this close it would
            # leak an open OS handle -- which on Windows blocks the corruption
            # recovery path from renaming the file aside.
            with contextlib.suppress(sqlite3.Error):
                conn.close()
            raise
        self._local.conn = conn
        with self._conns_lock:
            self._conns.append(conn)
        return conn

    def close(self) -> None:
        """Close every connection ever opened, across all threads.

        Safe because connections are created with ``check_same_thread=False``.
        Thread-local references are invalidated wholesale by swapping the
        ``threading.local`` object, so a later ``connect()`` reopens cleanly.
        """
        with self._conns_lock:
            conns, self._conns = self._conns, []
        for conn in conns:
            with contextlib.suppress(sqlite3.Error):  # pragma: no cover - best-effort teardown
                conn.close()
        self._local = threading.local()

    def _write(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Execute one write statement under the process-wide write lock."""
        conn = self.connect()
        with self._write_lock:
            return conn.execute(sql, params)

    # ------------------------------------------------------------------
    # Migrations
    # ------------------------------------------------------------------

    def migrate(self) -> None:
        """Apply pending migrations in filename order. Idempotent.

        Called at every startup. Each migration runs inside its own
        transaction together with its ``schema_migrations`` bookkeeping row,
        so a crash mid-migration leaves the database at the previous version
        rather than half-applied.
        """
        conn = self.connect()
        with self._write_lock:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at REAL, name TEXT)"
            )
            applied = {
                row["version"] for row in conn.execute("SELECT version FROM schema_migrations")
            }
            for version, name, sql in _migration_sources():
                if version in applied:
                    continue
                conn.execute("BEGIN IMMEDIATE")
                try:
                    for statement in _split_statements(sql):
                        conn.execute(statement)
                    conn.execute(
                        "INSERT INTO schema_migrations (version, applied_at, name) "
                        "VALUES (?, ?, ?)",
                        (version, time.time(), name),
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
                log.info("db.migration_applied", version=version, name=name)

    def migrate_with_recovery(self) -> bool:
        """Apply migrations, recovering from a corrupt database file.

        A corrupt ``registry.sqlite3`` (power loss mid-write, disk fault, a
        stray file with the same name) must degrade the server, not stop it
        from booting: everything in this database is either a cache (GGUF
        metadata), re-creatable (adapters re-appear on the next scan) or a
        convenience (saved settings, download history). The corrupt file is
        moved aside -- never deleted, so the user can attempt manual recovery
        of their virtual models and settings -- and a fresh database is
        created in its place. Returns True when recovery happened.
        """
        # ONLY real corruption reaches the recovery path. ``sqlite3.DatabaseError``
        # is also the base class of ``OperationalError``, which is what a
        # *locked* database raises when another process -- the running server,
        # a second instance, a backup tool -- holds it mid-write. Catching the
        # base class here therefore renamed a perfectly healthy live registry
        # to ``.corrupt-<stamp>`` and started a fresh one beside it (WP17 F5).
        # A lock is transient: wait for it briefly, then fail loudly and leave
        # the file exactly where it is.
        attempts = 0
        while True:
            try:
                self.migrate()
                return False
            except sqlite3.DatabaseError as exc:
                if is_locked_error(exc) and attempts < LOCK_RETRY_ATTEMPTS:
                    attempts += 1
                    log.warning(
                        "db.locked",
                        path=str(self._path),
                        attempt=attempts,
                        of=LOCK_RETRY_ATTEMPTS,
                        error=str(exc),
                    )
                    time.sleep(LOCK_RETRY_DELAY_S)
                    continue
                if not is_corruption_error(exc):
                    # Locked past the retry budget, unable to open, disk I/O
                    # error, read-only filesystem, ...: none of these are cured
                    # by replacing the file, and every one of them destroys data
                    # if we do. Surface the real reason instead.
                    raise sqlite3.OperationalError(
                        f"database at {self._path} could not be opened for migration "
                        f"({exc}); it is NOT being replaced because this is not a "
                        "corruption error -- if another StudioForge instance is running "
                        "against this data dir, stop it first"
                    ) from exc
                log.error(
                    "db.corrupt",
                    path=str(self._path),
                    error=str(exc),
                    action="moving the corrupt file aside and starting with a fresh database",
                )
                break
        self.close()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            source = Path(f"{self._path}{suffix}")
            if not source.exists():
                continue
            backup = source.with_name(f"{source.name}.corrupt-{stamp}")
            try:
                source.replace(backup)
            except OSError as replace_exc:
                # Renaming can fail (another handle open, permissions). Without
                # the rename the fresh database cannot be created either, so
                # this is the one case where boot genuinely cannot proceed.
                raise sqlite3.DatabaseError(
                    f"database at {self._path} is corrupt and could not be moved "
                    f"aside: {replace_exc}"
                ) from replace_exc
        self.migrate()
        log.warning("db.recovered", path=str(self._path), backup_suffix=f".corrupt-{stamp}")
        return True

    def schema_version(self) -> int:
        """Highest applied migration version, 0 for a virgin database."""
        try:
            row = (
                self.connect()
                .execute("SELECT MAX(version) AS version FROM schema_migrations")
                .fetchone()
            )
        except sqlite3.OperationalError:
            return 0
        if row is None or row["version"] is None:
            return 0
        return int(row["version"])

    # ------------------------------------------------------------------
    # Model settings
    # ------------------------------------------------------------------

    def get_model_settings(self, model_id: str) -> dict[str, Any] | None:
        row = (
            self.connect()
            .execute("SELECT settings_json FROM model_settings WHERE model_id = ?", (model_id,))
            .fetchone()
        )
        if row is None:
            return None
        settings: dict[str, Any] = json.loads(row["settings_json"])
        return settings

    def save_model_settings(self, model_id: str, settings: dict[str, Any]) -> None:
        self._write(
            "INSERT INTO model_settings (model_id, settings_json, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT (model_id) DO UPDATE SET "
            "settings_json = excluded.settings_json, updated_at = excluded.updated_at",
            (model_id, json.dumps(settings), time.time()),
        )

    def delete_model_settings(self, model_id: str) -> None:
        self._write("DELETE FROM model_settings WHERE model_id = ?", (model_id,))

    def all_model_settings(self) -> dict[str, dict[str, Any]]:
        rows = (
            self.connect().execute("SELECT model_id, settings_json FROM model_settings").fetchall()
        )
        return {row["model_id"]: json.loads(row["settings_json"]) for row in rows}

    # ------------------------------------------------------------------
    # Virtual models
    # ------------------------------------------------------------------

    def save_virtual_model(
        self,
        id: str,
        base_model_id: str,
        name: str,
        adapters: list[dict[str, Any]],
        preset: dict[str, Any] | None = None,
    ) -> None:
        """Upsert a virtual model.

        ``adapters_json`` is the extension point for everything a virtual
        model carries beyond its base (no new columns -- see the module
        docstring on JSON blob columns): with a preset the column holds
        ``{"adapters": [...], "preset": {...}}``, without one it stays the
        plain adapter list older rows already use, so a rollback to a build
        that predates presets keeps reading its own format.
        """
        payload: Any = {"adapters": adapters, "preset": preset} if preset else adapters
        self._write(
            "INSERT INTO virtual_models (id, base_model_id, name, adapters_json, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET "
            "base_model_id = excluded.base_model_id, name = excluded.name, "
            "adapters_json = excluded.adapters_json",
            (id, base_model_id, name, json.dumps(payload), time.time()),
        )

    def list_virtual_models(self) -> list[dict[str, Any]]:
        rows = (
            self.connect()
            .execute(
                "SELECT id, base_model_id, name, adapters_json, created_at "
                "FROM virtual_models ORDER BY created_at, id"
            )
            .fetchall()
        )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = _row_to_dict(row)
            # Both storage shapes: a bare adapter list (pre-preset rows) or an
            # {"adapters", "preset"} object. See save_virtual_model.
            payload = json.loads(record.pop("adapters_json"))
            if isinstance(payload, dict):
                record["adapters"] = payload.get("adapters") or []
                record["preset"] = payload.get("preset")
            else:
                record["adapters"] = payload
                record["preset"] = None
            out.append(record)
        return out

    def delete_virtual_model(self, id: str) -> None:
        self._write("DELETE FROM virtual_models WHERE id = ?", (id,))

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------

    def save_adapter(self, record: dict[str, Any]) -> None:
        """Upsert an adapter keyed on id; ``added_at`` survives re-scans."""
        values: dict[str, Any] = {
            "id": record["id"],
            "name": record["name"],
            "path": str(record["path"]),
            "added_at": record.get("added_at") or time.time(),
        }
        for column in _ADAPTER_OPTIONAL_COLUMNS:
            values[column] = record.get(column)
        columns = list(values)
        updates = ", ".join(
            f"{col} = excluded.{col}" for col in columns if col not in ("id", "added_at")
        )
        self._write(
            f"INSERT INTO adapters ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}",
            [values[col] for col in columns],
        )

    def list_adapters(self) -> list[dict[str, Any]]:
        rows = self.connect().execute("SELECT * FROM adapters ORDER BY added_at, id").fetchall()
        return [_row_to_dict(row) for row in rows]

    def get_adapter(self, id: str) -> dict[str, Any] | None:
        row = self.connect().execute("SELECT * FROM adapters WHERE id = ?", (id,)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def delete_adapter(self, id: str) -> None:
        self._write("DELETE FROM adapters WHERE id = ?", (id,))

    # ------------------------------------------------------------------
    # Downloads
    # ------------------------------------------------------------------

    def upsert_download(self, record: dict[str, Any]) -> None:
        """Upsert a download row; startup reconciliation and resume both
        re-write the same rows constantly, so conflict-update is the norm."""
        status = record["status"]
        if status not in DOWNLOAD_STATUSES:
            raise ValueError(f"invalid download status: {status!r}")
        now = time.time()
        values: dict[str, Any] = {
            "id": record["id"],
            "repo_id": record["repo_id"],
            "filename": record["filename"],
            "dest_path": str(record["dest_path"]),
            "status": status,
            "created_at": record.get("created_at") or now,
            "updated_at": now,
        }
        for column in _DOWNLOAD_OPTIONAL_COLUMNS:
            values[column] = record.get(column)
        columns = list(values)
        updates = ", ".join(
            f"{col} = excluded.{col}" for col in columns if col not in ("id", "created_at")
        )
        self._write(
            f"INSERT INTO downloads ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}",
            [values[col] for col in columns],
        )

    def get_download(self, id: str) -> dict[str, Any] | None:
        row = self.connect().execute("SELECT * FROM downloads WHERE id = ?", (id,)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list_downloads(
        self, *, statuses: Sequence[str] | None = None, group_id: str | None = None
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if statuses is not None:
            clauses.append(f"status IN ({', '.join('?' * len(statuses))})")
            params.extend(statuses)
        if group_id is not None:
            clauses.append("group_id = ?")
            params.append(group_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = (
            self.connect()
            .execute(f"SELECT * FROM downloads{where} ORDER BY created_at, id", params)
            .fetchall()
        )
        return [_row_to_dict(row) for row in rows]

    def update_download_progress(
        self, id: str, downloaded_bytes: int, *, total_bytes: int | None = None
    ) -> None:
        """Hot path: called on every downloaded chunk, so it touches only the
        progress integers plus ``updated_at`` -- no JSON, no row rewrite."""
        if total_bytes is None:
            self._write(
                "UPDATE downloads SET downloaded_bytes = ?, updated_at = ? WHERE id = ?",
                (downloaded_bytes, time.time(), id),
            )
        else:
            self._write(
                "UPDATE downloads SET downloaded_bytes = ?, total_bytes = ?, updated_at = ? "
                "WHERE id = ?",
                (downloaded_bytes, total_bytes, time.time(), id),
            )

    def set_download_status(self, id: str, status: str, *, error: str | None = None) -> None:
        if status not in DOWNLOAD_STATUSES:
            raise ValueError(f"invalid download status: {status!r}")
        self._write(
            "UPDATE downloads SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, time.time(), id),
        )
        log.info("db.download_status", download_id=id, status=status)

    def delete_download(self, id: str) -> None:
        self._write("DELETE FROM downloads WHERE id = ?", (id,))

    def prune_downloads(self, *, older_than_s: float) -> int:
        """Delete terminal download rows not touched for ``older_than_s``.

        Only ``completed``/``failed``/``canceled`` rows are eligible: those are
        pure history, and without pruning the table grows one row per file
        forever. ``queued``/``paused``/``running`` rows are live intent -- a
        paused 40 GiB download from last month must still resume -- so they are
        never touched regardless of age. Returns the number of rows removed.
        """
        cutoff = time.time() - max(0.0, older_than_s)
        cursor = self._write(
            "DELETE FROM downloads WHERE status IN ('completed', 'failed', 'canceled') "
            "AND updated_at < ?",
            (cutoff,),
        )
        removed = cursor.rowcount if cursor.rowcount is not None and cursor.rowcount > 0 else 0
        if removed:
            log.info("db.downloads_pruned", removed=removed)
        return removed

    # ------------------------------------------------------------------
    # GGUF scan cache
    # ------------------------------------------------------------------

    def get_cached_meta(self, path: str, mtime: float, size_bytes: int) -> dict[str, Any] | None:
        """Cached GGUF metadata, or None on any mismatch.

        The (mtime, size) pair is part of the lookup key on purpose: a file
        replaced out-of-band (re-download, manual swap) gets re-parsed instead
        of serving stale metadata for a different quant.
        """
        row = (
            self.connect()
            .execute("SELECT mtime, size_bytes, meta_json FROM gguf_cache WHERE path = ?", (path,))
            .fetchone()
        )
        if row is None or row["mtime"] != mtime or row["size_bytes"] != size_bytes:
            return None
        meta: dict[str, Any] = json.loads(row["meta_json"])
        return meta

    def put_cached_meta(
        self, path: str, mtime: float, size_bytes: int, meta: dict[str, Any]
    ) -> None:
        self._write(
            "INSERT INTO gguf_cache (path, mtime, size_bytes, meta_json, cached_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (path) DO UPDATE SET "
            "mtime = excluded.mtime, size_bytes = excluded.size_bytes, "
            "meta_json = excluded.meta_json, cached_at = excluded.cached_at",
            (path, mtime, size_bytes, json.dumps(meta), time.time()),
        )

    def prune_cache(self, keep_paths: Iterable[str]) -> int:
        """Drop cache entries whose path is not in ``keep_paths``.

        The survivor set is diffed in Python rather than via ``NOT IN (...)``
        placeholders so an arbitrarily large model library never trips
        SQLite's bound-parameter limit. Returns the number of rows removed.
        """
        keep = set(keep_paths)
        conn = self.connect()
        with self._write_lock:
            rows = conn.execute("SELECT path FROM gguf_cache").fetchall()
            doomed = [(row["path"],) for row in rows if row["path"] not in keep]
            if doomed:
                conn.executemany("DELETE FROM gguf_cache WHERE path = ?", doomed)
        if doomed:
            log.info("db.cache_pruned", removed=len(doomed))
        return len(doomed)

    # ------------------------------------------------------------------
    # Planner calibration
    # ------------------------------------------------------------------

    def record_load_observation(self, **fields: Any) -> None:
        """Insert one predicted-vs-actual VRAM observation.

        Accepts only known columns so a typo'd keyword fails loudly instead of
        silently dropping calibration data.
        """
        unknown = set(fields) - set(_OBSERVATION_COLUMNS)
        if unknown:
            raise ValueError(f"unknown load_observation fields: {sorted(unknown)}")
        if "model_id" not in fields:
            raise ValueError("model_id is required")
        fields.setdefault("ts", time.time())
        if isinstance(fields.get("ok"), bool):
            fields["ok"] = int(fields["ok"])
        columns = list(fields)
        self._write(
            f"INSERT INTO load_observations ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            [fields[col] for col in columns],
        )

    def load_observations(
        self, model_id: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Observations newest-first; ``ok`` comes back as a bool."""
        if model_id is None:
            rows = (
                self.connect()
                .execute(
                    "SELECT * FROM load_observations ORDER BY ts DESC, id DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        else:
            rows = (
                self.connect()
                .execute(
                    "SELECT * FROM load_observations WHERE model_id = ? "
                    "ORDER BY ts DESC, id DESC LIMIT ?",
                    (model_id, limit),
                )
                .fetchall()
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = _row_to_dict(row)
            if record.get("ok") is not None:
                record["ok"] = bool(record["ok"])
            out.append(record)
        return out

    def matching_observation(
        self,
        model_id: str,
        *,
        ctx_size: int,
        parallel: int,
        kv_cache_type: str | None,
        kv_cache_type_v: str | None,
        device_count: int,
    ) -> dict[str, Any] | None:
        """The newest trustworthy load of *this exact configuration*, or ``None``.

        The planner spends this row's ``actual_bytes`` as its estimate (D51),
        which makes the match rule a safety property rather than a convenience:
        a row that describes a different placement is worse than no row at all,
        because it is confidently wrong instead of merely approximate.

        What must be equal, and why each one:

        * ``model_id``, ``ctx_size``, ``parallel`` -- the three inputs the
          weights and KV terms are computed from. ``ctx_size`` is compared as
          the planner stores it (``LoadPlan.ctx_size``), so caller and row are
          always speaking about the same number whatever the per-slot/total
          reading of "context" is elsewhere.
        * ``kv_cache_type`` / ``kv_cache_type_v`` -- the cache quantization,
          which moves the KV term by a factor of two per half. A pre-007 row
          has NULL for V (see ``migrations/007_observation_kv_v.sql``) and is
          treated as *unable to say* rather than as symmetric: it matches only
          a lookup asking for the same V as its K, and is skipped for any
          asymmetric cache it might not describe.

        What must NOT be equal: the device *list*. Only the device COUNT is
        compared. Each card costs a CUDA context, so the count is a real term
        in the estimate -- but D42's rebalancer moves a model between
        same-count device sets, and the tensor-split proportions are recomputed
        from live free VRAM on every load, so the exact list changes run to run
        for placements that are otherwise identical. Matching on the list would
        make this feature almost never fire, which is the same as not shipping
        it.

        Only ``ok`` rows carrying :data:`OBSERVATION_NOTE_TRUSTED` are
        eligible. Everything else in this table is contaminated history whose
        median actual/predicted ratio is 2.97 -- a single such row adopted as
        an estimate would triple the model's apparent footprint and refuse
        loads that fit, so the note check is what makes a dirty row unable to
        poison the planner rather than merely unlikely to.

        Newest first; the first clean match wins. An older row is not averaged
        in: the newest measurement is the one taken against the engine build
        and driver in use now.
        """
        for row in self.load_observations(model_id, limit=_OBSERVATION_MATCH_WINDOW):
            if not row.get("ok"):
                continue
            if str(row.get("note") or "") != OBSERVATION_NOTE_TRUSTED:
                continue
            actual = row.get("actual_bytes")
            if not isinstance(actual, int) or actual <= 0:
                continue
            if row.get("ctx_size") != ctx_size or row.get("parallel") != parallel:
                continue
            row_k = row.get("kv_cache_type")
            if row_k != kv_cache_type:
                continue
            row_v = row.get("kv_cache_type_v")
            if row_v is None or row_v == "":
                # Pre-007: V unknown. Safe only where V was never asked to
                # differ from K -- the symmetric default.
                if kv_cache_type_v is not None and kv_cache_type_v != row_k:
                    continue
            elif row_v != kv_cache_type_v:
                continue
            devices = str(row.get("devices") or "")
            row_count = len([part for part in devices.split(",") if part.strip()])
            if row_count != device_count:
                continue
            return row
        return None

    # ------------------------------------------------------------------
    # Throughput calibration
    # ------------------------------------------------------------------

    def record_throughput_observation(self, **fields: Any) -> None:
        """Insert one measured tokens/second sample.

        Mirrors :meth:`record_load_observation`: unknown keywords raise rather
        than being silently dropped, because a typo here would look exactly
        like "no data yet" and would leave the catalog permanently
        uncalibrated with nothing in the logs to say why.
        """
        unknown = set(fields) - set(_THROUGHPUT_COLUMNS)
        if unknown:
            raise ValueError(f"unknown throughput_observation fields: {sorted(unknown)}")
        if "model_id" not in fields:
            raise ValueError("model_id is required")
        fields.setdefault("ts", time.time())
        columns = list(fields)
        self._write(
            f"INSERT INTO throughput_observations ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            [fields[col] for col in columns],
        )

    def throughput_observations(
        self, model_id: str | None = None, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Throughput samples newest-first, optionally for one model."""
        if model_id is None:
            rows = (
                self.connect()
                .execute(
                    "SELECT * FROM throughput_observations ORDER BY ts DESC, id DESC LIMIT ?",
                    (limit,),
                )
                .fetchall()
            )
        else:
            rows = (
                self.connect()
                .execute(
                    "SELECT * FROM throughput_observations WHERE model_id = ? "
                    "ORDER BY ts DESC, id DESC LIMIT ?",
                    (model_id, limit),
                )
                .fetchall()
            )
        return [_row_to_dict(row) for row in rows]

    def prune_throughput_observations(self, *, keep_per_model: int = 200) -> int:
        """Keep only the newest ``keep_per_model`` rows per model.

        The collector samples on a timer for as long as a model is resident, so
        without a cap a pinned model accumulates rows forever. Calibration only
        ever reads a recent window, so older rows are pure storage.
        """
        conn = self.connect()
        with self._write_lock:
            cursor = conn.execute(
                "DELETE FROM throughput_observations WHERE id NOT IN ("
                "  SELECT id FROM ("
                "    SELECT id, ROW_NUMBER() OVER ("
                "      PARTITION BY model_id ORDER BY ts DESC, id DESC"
                "    ) AS rn FROM throughput_observations"
                "  ) WHERE rn <= ?"
                ")",
                (int(keep_per_model),),
            )
            removed = cursor.rowcount if cursor.rowcount and cursor.rowcount > 0 else 0
        if removed:
            log.info("db.throughput_pruned", removed=removed)
        return removed

    # ------------------------------------------------------------------
    # Parallel-slot calibration
    # ------------------------------------------------------------------

    def record_parallel_observation(self, **fields: Any) -> None:
        """Insert one concurrency level of one parallel-benchmark run.

        Mirrors :meth:`record_throughput_observation`: an unknown keyword raises
        rather than being dropped, because a typo would look exactly like "this
        model has never been measured" and would leave ``recommended_parallel``
        permanently on its estimate with nothing in the log to say why.
        """
        unknown = set(fields) - set(_PARALLEL_COLUMNS)
        if unknown:
            raise ValueError(f"unknown parallel_observation fields: {sorted(unknown)}")
        if "model_id" not in fields:
            raise ValueError("model_id is required")
        fields.setdefault("ts", time.time())
        columns = list(fields)
        self._write(
            f"INSERT INTO parallel_observations ({', '.join(columns)}) "
            f"VALUES ({', '.join('?' * len(columns))})",
            [fields[col] for col in columns],
        )

    def parallel_observations(
        self,
        model_id: str | None = None,
        *,
        devices: Sequence[int] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Parallel-slot samples newest-first, optionally narrowed.

        ``devices`` is matched on the same canonical string the writer stores
        (``",".join(sorted(indices))``, the encoding
        :func:`studioforge.core.throughput.measured_for` already uses), so the
        caller passes CUDA indices and never has to know the encoding.

        Filtering in SQL rather than in Python because the catalog asks this
        once per hardware mode per model per build, and a table that a pinned
        model's nightly re-measurement grows without bound would otherwise be
        read whole four times for every row of a forty-model library.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if devices is not None:
            clauses.append("devices = ?")
            params.append(",".join(str(int(d)) for d in sorted(devices)))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        rows = (
            self.connect()
            .execute(
                f"SELECT * FROM parallel_observations{where} ORDER BY ts DESC, id DESC LIMIT ?",
                params,
            )
            .fetchall()
        )
        return [_row_to_dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def save_benchmark(self, model_id: str, report: dict[str, Any]) -> int:
        """Store one finished benchmark report; returns its row id.

        ``ts`` is taken from the report's own ``started_at`` when present so a
        report queued for a while still sorts by when it was *measured*, which
        is what makes two reports comparable.
        """
        ts = report.get("started_at")
        row = self._write(
            "INSERT INTO benchmarks (model_id, ts, ctx_size, max_tokens, report_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                model_id,
                float(ts) if isinstance(ts, int | float) else time.time(),
                report.get("ctx_size"),
                report.get("max_tokens"),
                json.dumps(report),
            ),
        )
        row_id = int(row.lastrowid or 0)
        log.info("db.benchmark_saved", model_id=model_id, benchmark_id=row_id)
        return row_id

    def list_benchmarks(self, model_id: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Benchmarks newest-first; ``report_json`` comes back parsed as ``report``."""
        if model_id is None:
            rows = (
                self.connect()
                .execute("SELECT * FROM benchmarks ORDER BY ts DESC, id DESC LIMIT ?", (limit,))
                .fetchall()
            )
        else:
            rows = (
                self.connect()
                .execute(
                    "SELECT * FROM benchmarks WHERE model_id = ? ORDER BY ts DESC, id DESC LIMIT ?",
                    (model_id, limit),
                )
                .fetchall()
            )
        out: list[dict[str, Any]] = []
        for row in rows:
            record = _row_to_dict(row)
            record["report"] = json.loads(record.pop("report_json"))
            out.append(record)
        return out

    def latest_benchmark(self, model_id: str) -> dict[str, Any] | None:
        rows = self.list_benchmarks(model_id, limit=1)
        return rows[0] if rows else None

    # ------------------------------------------------------------------
    # Key/value
    # ------------------------------------------------------------------

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.connect().execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return str(row["value"])

    def set_meta(self, key: str, value: str) -> None:
        self._write(
            "INSERT INTO kv (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (key) DO UPDATE SET "
            "value = excluded.value, updated_at = excluded.updated_at",
            (key, value, time.time()),
        )

    def delete_meta(self, key: str) -> None:
        self._write("DELETE FROM kv WHERE key = ?", (key,))
