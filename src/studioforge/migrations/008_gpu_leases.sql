-- 008_gpu_leases.sql: the lease book, mirrored to disk, so a restart does not
-- forget who holds which card.
--
-- The book (core/leases.py, D43) lived in memory only, on purpose: a lease
-- describes a live situation and a restart was to be a clean slate, with the
-- idle TTL as the safety net. That design failed live on 2026-09-10. A
-- POST /api/restart/server at 03:32 emptied the book, and with it the lease
-- ClawForge2 held on CUDA 2 -- the card ComfyUI renders on. At 03:45 a tier-3
-- load_recommended from a third tenant (transforge) placed a 30B model on
-- CUDA [2, 3], exactly where the lease would have refused it; at 04:55 a JIT
-- load from ClawChat V13 landed a 27B on the same pair the same way. Every
-- re-ask by the image gateway since answered 503 busy, because a lease never
-- interrupts a stream (D36): the holder that had done everything right was
-- locked out of its own card by a restart it did not ask for.
--
-- One row per standing lease, written through on acquire, release and the
-- vacate marks, and on a coalesced subset of touches (core/leases.py, D61).
-- At start the manager reads the rows back, drops any idle past its TTL --
-- the sweep would have released it had the server stayed up -- and re-enters
-- the rest with their clocks. The idle TTL still bounds a holder that died
-- while the server was down.
--
-- vacate_token and holder_peer are excluded from every API dump (D55, D56)
-- but MUST be stored: the token is what lets this server POST a vacate
-- request to the holder after a restart, and the peer is the proof of
-- holdership the open unload routes check. The registry file is local to
-- this box -- it sits in the data dir beside the model settings and the
-- download queue -- so a token here is no more exposed than the process
-- memory it lived in before. The vacate WINDOW (requested_at, deadline,
-- delivery) is deliberately not stored: a restart ends any ask in flight,
-- and a better class asking again re-runs the D56 protocol from scratch.
CREATE TABLE IF NOT EXISTS gpu_leases (
    id TEXT PRIMARY KEY,
    devices_json TEXT NOT NULL,
    holder TEXT NOT NULL,
    model_ids_json TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    last_activity_at REAL NOT NULL,
    idle_ttl_s REAL,
    priority INTEGER NOT NULL DEFAULT 3,
    vacate_url TEXT,
    vacate_token TEXT,
    holder_peer TEXT,
    updated_at REAL NOT NULL
);
