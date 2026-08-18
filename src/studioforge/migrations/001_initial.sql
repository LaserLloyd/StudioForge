-- 001_initial.sql: the durable state that must survive restarts and self-updates.
--
-- Settings are stored as one JSON blob per model rather than a column per flag:
-- the llama-server flag surface changes with every engine release, and a
-- column-per-flag schema would need a migration each time. JSON keeps the
-- schema stable while ModelSettings (pydantic) owns validation at the edges.

CREATE TABLE IF NOT EXISTS model_settings (
    model_id      TEXT PRIMARY KEY,
    settings_json TEXT NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS virtual_models (
    id            TEXT PRIMARY KEY,
    base_model_id TEXT NOT NULL,
    name          TEXT NOT NULL,
    adapters_json TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_virtual_models_base
    ON virtual_models (base_model_id);

CREATE TABLE IF NOT EXISTS adapters (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    path              TEXT NOT NULL UNIQUE,
    size_bytes        INTEGER,
    base_architecture TEXT,
    base_model_hint   TEXT,
    publisher         TEXT,
    repo              TEXT,
    n_layer           INTEGER,
    rank              INTEGER,
    added_at          REAL NOT NULL
);

-- group_id ties a multi-part GGUF and its mmproj into one logical download so
-- the UI can show (and cancel) them as a unit.
CREATE TABLE IF NOT EXISTS downloads (
    id               TEXT PRIMARY KEY,
    repo_id          TEXT NOT NULL,
    filename         TEXT NOT NULL,
    dest_path        TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (
        status IN ('queued', 'running', 'paused', 'completed', 'failed', 'canceled')
    ),
    total_bytes      INTEGER,
    downloaded_bytes INTEGER,
    sha256           TEXT,
    etag             TEXT,
    error            TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL,
    group_id         TEXT
);

CREATE INDEX IF NOT EXISTS idx_downloads_status
    ON downloads (status);

CREATE INDEX IF NOT EXISTS idx_downloads_group
    ON downloads (group_id);

-- Parsing dozens of GGUF headers at every startup is wasteful; entries are
-- keyed on (path, mtime, size) so any out-of-band file change invalidates.
CREATE TABLE IF NOT EXISTS gguf_cache (
    path       TEXT PRIMARY KEY,
    mtime      REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    meta_json  TEXT NOT NULL,
    cached_at  REAL NOT NULL
);

-- Predicted-vs-actual VRAM per load: lets the planner's overhead fudge factor
-- be tuned from real data instead of guesswork.
CREATE TABLE IF NOT EXISTS load_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT NOT NULL,
    ts              REAL NOT NULL,
    ctx_size        INTEGER,
    parallel        INTEGER,
    kv_cache_type   TEXT,
    devices         TEXT,
    predicted_bytes INTEGER,
    actual_bytes    INTEGER,
    weights_bytes   INTEGER,
    ok              INTEGER,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS idx_load_observations_model
    ON load_observations (model_id);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
