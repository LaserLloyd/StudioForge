# Security

How StudioForge decides who may do what, what it protects, and what it does not. For reporting a
vulnerability see [`SECURITY.md`](../SECURITY.md) at the repository root.

## The short version

- **Out of the box nothing is reachable from the network.** All three services (the gateway on
  `:1234`, the control panel on `:8080`, the recovery watchdog on `:1235`) bind `127.0.0.1`.
- **A server that would listen on the network refuses to start** until an owner account or an
  admin API key exists. The message names the services and the two commands that fix it
  (`studioforge setup`, `studioforge keys create --name admin --scope admin`).
- **Credentials are named, scoped and hashed.** `read` < `inference` < `admin`; an optional expiry
  and requests-per-minute limit; last use is recorded; only a SHA-256 is stored.
- **Everything an admin does is written to an append-only audit log.**
- **Secrets at rest are owner-only files or the OS keyring.**

## What is protected, and from whom

| Asset | Who we defend it from |
| --- | --- |
| GPU time and VRAM (a request can load a model and evict another) | other machines on the network; a web page in your browser; an agent holding a key that is too strong |
| The ability to change the box (config, files, engine, updates, restarts, kill) | the same, and every caller that is not an admin |
| Credentials (`config.yaml`, keys, the owner password hash, the HF token) | other accounts on the same computer; anyone who gets a copy of the database, a backup, a support bundle or a log |
| Model files and the model library | the same as the box; deletion needs `admin` and a confirmation |
| Prompts and completions | not stored by StudioForge; logs record sizes and timings, never content by default |

Not defended: a user with administrator rights on the machine (they can read process memory and the
data directory), and a network you do not control carrying plain HTTP (turn TLS on, see below).
There is no telemetry; the only outbound calls are listed in the architecture notes (GitHub for
llama.cpp releases and optional app updates, Hugging Face for search and downloads).

## Binding and the bind rule

| Setting | Default | Meaning |
| --- | --- | --- |
| `server.host` | `127.0.0.1` | gateway (`/v1`, `/api`, `/mcp`, `/metrics`) |
| `gui.host` | `127.0.0.1` | control panel |
| `watchdog.host` | `127.0.0.1` | recovery watchdog |

Loopback-only installs need no credential at all (first run is friction-free). The moment any of the
three is set to a non-loopback address (`0.0.0.0`, a LAN address, a hostname), `studioforge serve`
checks for a credential that could be used to log in: an **owner account**, an **admin API key**, or
the deprecated `server.api_key`. If there is none it prints `auth_required_for_remote_bind` with the
exact fix and exits with code 4 (the tray reads that as "needs setup", not a crash to respawn).

Once a credential exists and an address is on the network, **every request needs one**: authentication
is enforced when `server.api_key` is set, when any active key exists, or when a listener is on the
network and an owner account exists. `/health` stays public for probes.

Existing installs: a `config.yaml` written by an earlier version names `host: 0.0.0.0` explicitly, so
the new default does not change it, and the server will refuse to start until you run
`studioforge setup` (see the CHANGELOG for the exact steps).

## Credentials

Tried in this order:

1. **`server.api_key`** (deprecated). Still accepted everywhere as an `admin` credential. It is
   mirrored into the key table on start as the admin key named `legacy`, so it shows up in
   `keys list` and `keys revoke legacy` really disables it.
2. **Scoped API keys**: `sf-<r|i|a>-<43 base62 characters>` (256 bits). Shown once when created.
   The letter is a hint for humans and secret scanners; the scope that counts is stored beside the
   hash.
3. **`mcp.pin`** (deprecated). Eight digits; accepted on the MCP endpoints only, as an `admin`
   credential, with a warning in the log. On an install with no keys it also opens the box-changing
   HTTP routes for a remote caller (the old D32 rule) so existing pairings keep working.

Sessions in the panel use the owner password (argon2id) or an admin key; see below.

### Scopes

| Scope | May | Examples |
| --- | --- | --- |
| `read` | look | `GET` anything except accounts, keys, audit, doctor; `/v1/models`; `/metrics`; MCP read tools |
| `inference` | read + use the models | `POST /v1/chat/completions`, `/v1/messages`, `/v1/responses`, `/api/v0/*`, `/api/v1/*`; load, unload and test a model; MCP `load_model` |
| `admin` | change the box | config, restart, engine install/activate, updates, downloads, delete, pin, leases, tenants, keys, accounts, audit, doctor, support bundle; MCP destructive tools |

The mapping is code, not convention: `api/auth.py:required_scope`. It matches by prefix
(`/v1/`, `/api/v0/`, `/api/v1/`), so routes added later are covered without an edit, and any
mutating route nobody classified needs `admin`.

A scope is enforced whenever a credential is presented, even on an install that needs none: a read
key cannot `POST`.

### Keys

```
studioforge keys create --name openclaw --scope inference --rate-limit-rpm 120
studioforge keys create --name my-agent --scope read --expires-in-days 30
studioforge keys list            # name, scope, prefix, last used, expiry, state - never the token
studioforge keys revoke openclaw
```

Creating the first key turns authentication on (clients on this machine too). The CLI, the tray and
the panel on the same machine keep working: they use a local admin key kept in `<data-dir>/cli.key`
(owner-only), created on demand. Errors are specific: `api_key_revoked`, `api_key_expired` (401) and
`key_rate_limited` (429 with `Retry-After`).

### The owner account

`studioforge setup` (or `studioforge users create`, or the panel's Users & Keys page) creates it. The
password is hashed with argon2id and the hash is stored through the secret store (OS keyring, else a
`0600` file), not in `registry.sqlite3`, so a copied database carries no login material. Passwords
must be at least 10 characters.

## The control panel

- Sign in with the owner password or an `admin` key. A `read` or `inference` key cannot open the panel.
- The session cookie is signed with a per-install secret, `HttpOnly`, `SameSite=Lax`, `Secure` when TLS
  is on, and bound to the password (or the key's hash): changing the password or revoking the key ends
  the session.
- Login attempts share the API's lockout (three free failures, then a doubling wait up to five
  minutes, per client address). The login `POST` refuses a foreign `Origin`.
- The control channel (socket.io) already refused cross-site `Origin`s and every response carries
  `X-Frame-Options: DENY` and `frame-ancestors 'none'`.

## TLS

`server.tls.mode`, one switch for all three listeners:

| Mode | What it does |
| --- | --- |
| `off` | plain HTTP (fine on loopback) |
| `self_signed` | an EC P-256 certificate generated into `<data-dir>/tls/` on first use, renewed 30 days before it expires or when the machine's addresses change; its **SHA-256 fingerprint** is shown in the panel and in `GET /api/status` (`tls.fingerprint_sha256`) so clients can pin it |
| `cert` | your PEM files: `server.tls.cert_file` and `key_file` |

A self-signed certificate protects against passive eavesdropping, and against an active attacker only
for clients that pin the fingerprint. For a public name use `cert` behind a real CA, a reverse proxy,
or `tailscale serve`. If TLS is misconfigured `serve` stops with `invalid_tls_config` instead of
quietly serving plain HTTP. Internal callers (the watchdog hand-off, the tray, the updater, the port
probe) follow the same switch and skip verification only for loopback.

## CORS and Origin validation

`server.cors_origins` is **empty by default** and `allow_credentials` is off. Non-browser clients
(OpenClaw, SDKs, curl) are unaffected. A browser page can read the API only from an origin you list.

Both MCP endpoints (the gateway's `/mcp` and the watchdog's `/mcp` and `POST /restart`) add an
**Origin allowlist** on top: a request that carries an `Origin` header is refused with `403
origin_not_allowed` unless it comes from a loopback page (`http(s)://127.0.0.1|localhost|[::1]`, any
port) or is listed in `server.cors_origins`. A wildcard `*` is deliberately **not** honoured here: it
means "any page may read the API", not "any page may drive the management tools". This is what stops
DNS rebinding and drive-by cross-site pages, which always send an `Origin`.

## MCP

- **Annotations.** Every tool is published with `readOnlyHint`, `destructiveHint` and
  `idempotentHint`, derived from one table (`studioforge/mcp_policy.py`). A tool nobody classified is
  treated as destructive.
- **Scopes per tool.** Read tools need `read`; state-changing tools need `inference`; destructive tools
  (`delete_model`, `set_config`, `download_model`, `reserve_gpus`, `unload_model`, and the watchdog's
  `restart_server`, `kill_model`, `nuke_all_models`, `reclaim_orphan_engines`, `rollback_update`,
  `set_config`) need `admin`. `pin_model` and `load_recommended` with `persist: true` write saved
  settings and need `admin` too. A refusal is `permission_denied`, never a silent no-op.
- **Audit.** Every non-read call is recorded, including the refused ones.
- **Protocol.** The handshake revision `2025-11-25` and the stateless `2026-07-28` revision both work
  and are tested through the SDK client; the policy is identical on both.
- **Agents.** Give an agent a scoped key, never the owner password. The panel's Integrations page
  shows a `--mcp-config` file and an Agent SDK `mcp_servers` block for it.

Tool names are a stable public API; see `docs/contracts/mcp-tools.json`.

## Secrets at rest

| Secret | Where | Protection |
| --- | --- | --- |
| `server.api_key`, `mcp.pin` | `config.yaml` (and its `.bak`) | written `0600` on POSIX, owner-only ACL on Windows (`icacls`) |
| `hf.token` | OS keyring when the optional `keyring` package has a real backend, else `config.yaml` | as above |
| Owner password hash | secret store: keyring, else `<data-dir>/secrets/` | argon2id; `0600` file fallback |
| API keys | `registry.sqlite3`, SHA-256 only | the token is never stored |
| Key snapshot for the watchdog | `<data-dir>/keys.snapshot.json` | hashes only; `0600` |
| TLS private key | `<data-dir>/tls/` | `0600` |
| Local admin key | `<data-dir>/cli.key` | `0600` |

Install the keyring extra with `pip install "studioforge[keyring]"`. `SF_SECRETS_BACKEND=file` forces
the file backend (containers, tests). `studioforge doctor` flags a credential file other accounts can
read. Credentials are also registered with the log redactor, never put in a URL, and removed from the
support bundle (config values by key name, literal secrets, credential-shaped strings, the home path).

## Audit log

`<data-dir>/logs/audit.jsonl`, one JSON object per line, append-only (rotation renames the file; it is
never edited):

```json
{"ts":"2026-10-18T09:30:00.123Z","actor":"key:3f9a1c2b4d5e","action":"api.PATCH /api/config","target":"models.default_ctx","result":"ok","surface":"api","peer":"192.168.1.20"}
```

`actor` is `key:<id>`, `legacy`, `pin`, `owner:<name>`, `panel`, `local` (this machine, no credential)
or `anonymous`. `target` never holds a secret: a configuration change lists the **keys** it touched,
not their values. It records every admin mutation over HTTP, every non-read MCP call (and the denials)
and the watchdog's restart. `logging.audit_file_max_mb` and `audit_file_backups` control rotation;
`GET /api/audit` and the panel read it back.

## Rate limiting

Wrong credentials are throttled per client address (three free failures, then 1, 2, 4 ... 300 s), on the
API, the panel login and the watchdog separately. Each key can also carry a requests-per-minute
limit. Inference traffic itself has no global limit; the model queue is the limit.

## The watchdog

The recovery sidecar is a separate process that never opens the database. It verifies scoped keys
against `keys.snapshot.json`, which the gateway rewrites on every key change. Its tools follow the same
scopes; `POST /restart` needs `admin`; with no credential configured it accepts this machine only.

## Hardening checklist for a LAN or tailnet

1. `studioforge setup` (owner account), choose network exposure with TLS.
2. One key per client, the weakest scope that works, an expiry for anything temporary.
3. Remove `server.api_key` and `mcp.pin` from `config.yaml` once clients use scoped keys
   (`studioforge doctor` keeps flagging both while they are set).
4. Keep `server.cors_origins` empty unless a browser app needs the API.
5. Pin the certificate fingerprint in clients that support it.
6. `studioforge doctor --strict` in your monitoring.
