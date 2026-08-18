-- 002_benchmarks.sql: persisted model benchmark reports.
--
-- The whole report is one JSON blob for the same reason model settings are
-- (see 001): the per-mode result shape follows what llama-server's `timings`
-- object exposes, which changes with the engine, and a column-per-metric
-- schema would need a migration every time a new one appears. The columns that
-- ARE promoted out of the blob (model_id, ts, ctx_size, max_tokens) are the
-- ones queries filter and sort on.
CREATE TABLE IF NOT EXISTS benchmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    ts          REAL NOT NULL,
    ctx_size    INTEGER,
    max_tokens  INTEGER,
    report_json TEXT NOT NULL
);

-- "the last N benchmarks for this model" is the only read pattern.
CREATE INDEX IF NOT EXISTS idx_benchmarks_model_ts
    ON benchmarks (model_id, ts);
