-- 003_throughput.sql: measured tokens/second, so the catalog's speed column
-- stops being arithmetic and starts being evidence.
--
-- One row per (model, placement) sample taken from the child's Prometheus
-- /metrics endpoint. Unlike load_observations -- which is written once per
-- load -- these accumulate while a model serves, so the table is written by a
-- slow timer and read only when the catalog is built.
--
-- est_prompt_tps / est_gen_tps are the estimator's prediction AT THE MOMENT OF
-- MEASUREMENT, stored beside the measurement on purpose. Calibration needs the
-- ratio measured/estimated, and re-deriving the estimate later would need the
-- exact per-device byte split the plan had at the time -- which is gone once
-- the model is unloaded. Storing both makes a row self-contained.
--
-- gpu_class ("RTX 5090x2+RTX 3090x2") is recorded rather than derived from the
-- device indices: it is the key calibration falls back to, and CUDA ordinals
-- are not stable across driver updates or a card being moved.
CREATE TABLE IF NOT EXISTS throughput_observations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id          TEXT NOT NULL,
    ts                REAL NOT NULL,
    devices           TEXT,
    gpu_class         TEXT,
    ctx_size          INTEGER,
    parallel          INTEGER,
    kv_cache_type     TEXT,
    prompt_tps        REAL,
    gen_tps           REAL,
    est_prompt_tps    REAL,
    est_gen_tps       REAL,
    n_busy_slots      REAL,
    requests_deferred REAL,
    sample_s          REAL
);

-- "the last N samples for this model" (calibration) and "everything on this
-- hardware class" (the fallback tier) are the only two read patterns.
CREATE INDEX IF NOT EXISTS idx_throughput_model_ts
    ON throughput_observations (model_id, ts);

CREATE INDEX IF NOT EXISTS idx_throughput_class_ts
    ON throughput_observations (gpu_class, ts);
