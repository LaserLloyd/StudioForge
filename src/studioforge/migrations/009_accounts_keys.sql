-- 009_accounts_keys.sql: the owner account and scoped, hashed API keys (W3).
--
-- Until now StudioForge had one shared credential (server.api_key) and an
-- eight-digit MCP PIN, both plaintext in config.yaml. This adds the real thing:
--
--   accounts  the owner account (and any later user). The password HASH is not
--             here: it lives in the secret store (OS keyring, else an owner-only
--             file) so a copied registry.sqlite3 carries no login material.
--   api_keys  named keys with a scope (read, inference, admin), an optional
--             expiry and per-minute rate limit, last-used time and a revoked
--             flag. Only the SHA-256 of the token is stored. The token is shown
--             once, when the key is created. prefix is the first nine characters
--             (sf-i- plus four) so a key can be recognised in a list.
--
-- The legacy server.api_key is migrated into this table on first start as an
-- admin key named legacy (see core/accounts.py), and the config key stays,
-- marked deprecated, so existing clients keep working.

CREATE TABLE IF NOT EXISTS accounts (
    id                  TEXT PRIMARY KEY,
    username            TEXT NOT NULL UNIQUE,
    role                TEXT NOT NULL DEFAULT 'owner',
    created_at          REAL NOT NULL,
    password_changed_at REAL,
    last_login_at       REAL,
    disabled            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS api_keys (
    id             TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    scope          TEXT NOT NULL CHECK (scope IN ('read', 'inference', 'admin')),
    key_hash       TEXT NOT NULL UNIQUE,
    prefix         TEXT NOT NULL,
    created_at     REAL NOT NULL,
    expires_at     REAL,
    last_used_at   REAL,
    rate_limit_rpm INTEGER,
    revoked_at     REAL,
    created_by     TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys (revoked_at, expires_at);
