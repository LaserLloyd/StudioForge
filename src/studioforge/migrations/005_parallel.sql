-- 005_parallel.sql: measured slot scaling, so "how many parallel slots are
-- worth running" stops being D17's arithmetic and starts being evidence.
--
-- One row per (run, concurrency level). A parallel benchmark loads the model
-- once and fires N concurrent completions for each N in 1, 2, 4, 8; every level
-- writes a row here. That is a different shape from throughput_observations,
-- which is a timer sampling a model while it happens to serve: these rows are
-- a deliberate sweep in which the ONLY thing that varies is the number of busy
-- slots, which is exactly what makes them comparable to each other.
--
-- Why the columns are what they are:
--
-- * per_stream_tps and aggregate_tps are the two halves of the question. Slot
--   batching raises the aggregate while lowering the per-stream rate (D17: each
--   decode step reads the active weights once whatever the batch size, so busy
--   slots amortise that read, and each slot's KV read is added on top). A
--   recommendation needs both -- "more total tokens" is worthless if one
--   conversation has become unusably slow.
-- * p95_latency_s is the latency the rule is allowed to notice. A knee shows up
--   as a plateau in aggregate_tps with p95 still climbing.
-- * n_busy_slots is the PROOF that batching happened at all. A client that
--   sends 8 requests to a 1-slot server still gets 8 answers, just serialized,
--   and the throughput table alone cannot tell those two runs apart. It is
--   derived from the delta in llama.cpp's own `llamacpp:n_decode_total` across
--   the level (completion tokens / decode steps), not read off the
--   `n_busy_slots_per_decode` gauge, because that gauge is a cumulative average
--   over the child's whole life and is therefore dragged down by every earlier
--   level of the same run.
-- * ctx_per_slot, kv_cache_type and kv_cache_type_v pin the placement the
--   measurement describes: the knee moves with the KV cache size per slot, so a
--   run at 8192/f16 says nothing about the same model at 131072/q8_0.
-- * gpu_class ("RTX 5090x2+RTX 3090x2") and engine_tag are recorded rather than
--   derived. CUDA ordinals move when a card is added or the driver reorders
--   them (D22), and a llama.cpp build can change the batching behaviour being
--   measured -- an observation that cannot say which engine produced it cannot
--   be retired when that engine is replaced.
-- * run_id groups the levels of one sweep. "The newest run on these devices" is
--   the unit the recommendation rule reads, and without a group key the newest
--   rows could straddle two runs at different contexts.
CREATE TABLE IF NOT EXISTS parallel_observations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id          TEXT NOT NULL,
    ts                REAL NOT NULL,
    run_id            TEXT,
    devices           TEXT,
    gpu_class         TEXT,
    ctx_per_slot      INTEGER,
    kv_cache_type     TEXT,
    kv_cache_type_v   TEXT,
    n_streams         INTEGER,
    per_stream_tps    REAL,
    aggregate_tps     REAL,
    p95_latency_s     REAL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    n_busy_slots      REAL,
    engine_tag        TEXT
);

-- The only two read patterns: "the newest run for this model" (the
-- recommendation) and "everything for this model on these devices" (the
-- placement column, which is asked once per hardware mode per catalog build).
CREATE INDEX IF NOT EXISTS idx_parallel_model_ts
    ON parallel_observations (model_id, ts);
