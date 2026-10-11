# The VRAM Tenant Protocol (`vt.1`)

How several GPU programs share one box without stepping on each other's memory. StudioForge is
the **arbiter**: it owns the lease book (D43), asks a holder to leave on a better class's behalf
(D56), remembers the book across its own restarts (D61), and — since D77 — keeps a **registry** of
who the other programs are and whether they are up. Everything else on the box is a **tenant**: an
image gateway in front of ComfyUI, a speech server, a benchmark harness, an agent that wants a card
for a while.

The protocol is small on purpose. A tenant that follows it needs an HTTP client and nothing from
StudioForge's code; `packages/studioforge-tenant/` is that client, once, for anyone who wants it.

**What this page is not.** It is not a supervisor. StudioForge never starts, stops, restarts,
installs or updates a tenant (D79). Each program keeps its own repository, port, launcher and
update path; the protocol is only how they share the cards.

---

## 1. Concepts

| Term | Meaning |
| --- | --- |
| **tenant** | Any program other than StudioForge that allocates GPU memory on the box and takes a lease before it does. |
| **module** | A tenant that is also an optional add-on StudioForge shows on its own Tenants card and may *ask* to warm (one POST to the module's `warm_url`). A module is still its own process with its own launcher; `kind: module` changes what the GUI offers, nothing about who starts it. |
| **holder** | The `holder` string a lease carries, e.g. `speechforge`, `clawforge2`, `crucibleforge-judge`, `openclaw-main`. |
| **holder family** | Everything before the first `-` or `:` in a holder, lower-cased: `crucibleforge-judge` → `crucibleforge`. Every lease view carries it as `holder_family`; a tenant's registry row carries it as `family`, and the row's `leases[]` are the book filtered by it. Match families, never exact holder strings (D53). |
| **class** (`priority`) | The D46 tier of a claim: **1** the model a person is actively chatting with, **2** a dispatched agent's work (a voice session a person is waiting on belongs here too — see §2), **3** background (renders, benchmarks, anything that never said). The same three numbers a load carries. |
| **lease** | An exclusive claim on one or more whole CUDA devices. While it stands StudioForge plans nothing else onto those cards (D43). |
| **vacate** | StudioForge *asking* a holder to release, on behalf of a strictly better class (D56). Never a revocation. |
| **registry** | The list of known tenants: declared by the operator in `tenants.registry`, or announced by the program through `POST /api/tenants/register`, merged by `name`. |

## 2. Priority classes, as the code implements them

The classes are StudioForge's D46 tiers, unchanged, and the lease plane uses them exactly as D56
item 6 wrote them down:

| Class | Name in the code | Who claims it | What it may displace when it takes a card |
| --- | --- | --- | --- |
| 1 | `PRIORITY_CHAT` — "active chat" | the companion a person is typing to | an idle resident of class 2 or 3 |
| 2 | `PRIORITY_AGENT` — "dispatched agent" | an agent's model; **a voice session** (SpeechForge S3: a person is waiting, but voice must never evict the chat model); StudioForge's own benchmarks (`benchmark.lease_priority`, default 2) | an idle resident of class 2 or 3 |
| 3 | `PRIORITY_BACKGROUND` — the default | renders (ClawForge2), harness runs, anything that never said | an idle resident of class 3 |

Three rules hold at every class:

* a **busy** resident (a request in flight, or still loading) is never interrupted — the grant
  answers `503 model_busy` with `retry_after_s` (D36);
* a **pinned** idle resident refuses the grant unless `force`, and a **better-class** idle
  resident refuses it the same way (equal class is displaceable, strictly better is not);
* a card already **leased** is a conflict, never a takeover — but a strictly better class may
  *ask* the holder to leave (§4).

## 3. Lease lifecycle

```
acquire ──► (held) ──touch──► (held) ──release──► gone
                │                           ▲
                └── idle for idle_ttl_s ────┘  (the sweep releases it)
```

* **Acquire** — `POST /api/leases`. Name the devices, or let StudioForge choose (§5). Idle
  residents on the granted cards are unloaded as part of the grant; that is what "the card is
  yours" means. The reply is the lease record with its `id`.
* **Touch** — `POST /api/leases/{id}/touch` restarts the idle clock. StudioForge cannot see a
  tenant's own activity, so a long job must say it is still running; a lease nobody touched for
  `idle_ttl_s` (default 3600 s; `null` = until released) is released by the sweep. Touches are
  cheap: StudioForge coalesces the disk writes.
* **Release** — `DELETE /api/leases/{id}`. A `404` means it is already gone, which is the outcome
  you wanted.
* **Expiry** — the sweep's release. The safety net for a holder that died; not the normal exit.
* **Across a StudioForge restart** — leases are persisted and restored with their clocks (D61).
  A lease your previous process took is still standing when your new one boots: release it
  (§7, "release stale leases").

Lease states, on every view: `active`, `idle` (quiet for half its TTL, at most 5 min),
`expiring` (inside the last quarter, at most 5 min), `vacating` (asked to leave).

## 4. Vacate (D56)

A lease may register a `vacate_url` and a `vacate_token` the holder minted. When a **strictly
better class** asks for cards the lease holds, and every lease in the way registered a URL:

1. StudioForge POSTs **one** request to the URL — body `{lease_id, devices, requester,
   requester_priority, deadline_s, deadline_at}`, headers `X-SF-Vacate-Token: <your token>` and
   `X-SF-Lease-Id: <id>`, no redirects followed, `trust_env` off, a callback timeout of
   `leases.vacate_callback_timeout_s` (30 s) for your `2xx`.
2. The lease is marked `vacating`; the asker is answered `409 lease_vacating` with
   `retry_after_s` (15 s) and re-asks on that interval.
3. The holder has **`deadline_s`** (`leases.vacate_timeout_s`, 180 s) — the **drain budget** — to
   stop taking new work, finish the piece in flight, free its VRAM and `DELETE` its lease. The
   moment it does, the asker's next request is granted.
4. If the window lapses, the asker is answered a plain `409 lease_conflict` with
   `error.studioforge.vacate.state: "timed_out"`; if the POST itself failed (connection refused,
   a timeout, a redirect, a non-2xx) the window collapses at once and the word is
   `"undeliverable"`. Either way a **cooldown** equal to one window follows before the holder may
   be asked again (`vacate.reask_at`).

What a tenant should do with the request is its own business and should be bounded: SpeechForge
drains in-flight work for at most 15 s, unloads, releases and stays off the card for its
`vacate_cooldown_s`; ClawForge2 stops accepting renders, lets the one inside ComfyUI finish, frees
what it can and releases. The token comparison must be constant-time and the route should accept
only loopback (StudioForge and the tenant share the box); `vacate_asgi_route` in the SDK does both.

Automatic placement (§5) **never** triggers a vacate: an ask for a card the asker did not need is
noise. To invoke D56, name the card.

## 5. What StudioForge promises

* **No JIT placement onto a leased card.** The planner sees leased cards as absent and places a
  model elsewhere or refuses it `507 gpu_leased` with the lease named (D43, D53). This is the
  check-then-act race the protocol exists to close: a tenant that reads free VRAM and allocates
  can have a model land on the same bytes in between (the 2026-08-27 render that came back as
  noise); a tenant that leases cannot.
* **A busy resident is never interrupted** by a grant, at any class (D36).
* **Idle residents are weighed against the claim's class** — a class-2 claim may displace an
  idle class-2 or class-3 resident, never an idle class-1 one; `force` is the only override and a
  tenant never sends it.
* **A lease is never taken away.** Not by a better class, not by `force`, not by a restart (D61).
  It ends when the holder releases it, or when it has been idle for its TTL.
* **Automatic placement, when asked** (`devices: "auto"`, D78): one card, chosen the way the
  grant judges a card — never a busy, leased, pinned or better-class-held card; then fewest
  evictions first, the caller's `prefer` order, most free VRAM, lowest index. A refusal lists
  every card `considered` with its verdict.
* **A registry row with live health** for every tenant that declares or registers itself, and a
  health poll every 30 s that never follows a redirect and never leaves the allowed hosts (§8).
* **It never starts or stops a tenant** (D79). A `module` may be *asked* to warm, through the
  module's own `warm_url`, when an operator clicks or calls; that is the whole of it.

## 6. What a tenant must do

1. **Lease before you allocate**, on the card you are about to use, under a stable `holder`
   whose family is your name. Send `X-SF-Client: <holder>` on every call too (D70).
2. **Say your class** honestly (§2) and **never send `force`**. A pinned or deliberately placed
   model is the operator's decision (ClawForge2's rule; SpeechForge S3).
3. **Release your own stale leases at boot.** D61 keeps a lease across a StudioForge restart and
   a crash of yours leaves one standing for up to its TTL: `GET /api/leases`, filter by your
   `holder_family`, `DELETE` each. Nothing of yours is loaded at boot, so all of them are stale.
4. **Keep your footprint bounded** to the cards you lease. A lease is a whole card; do not spill
   onto another one unleased.
5. **Touch while you work, and let it expire when you stop.** Touch only on activity; an idle
   lease that expires is the protocol working.
6. **Honour a vacate within the drain budget** (§4): finish or stop the work in flight, free, and
   `DELETE` the lease. Answer the callback `2xx` promptly even if the freeing takes longer.
7. **Fail open only onto a clearly free card** when StudioForge is unreachable. The reference
   behaviour is **SpeechForge S16**: with StudioForge down, load unleased only if NVML shows the
   room you need on the card *and* no `llama-server` process holds memory there; otherwise refuse
   and cool down. Silence is not a grant.
8. **Register at boot** (`POST /api/tenants/register`, §9) so the operator sees you, and say your
   `version`. Optional, but a tenant with a row is a tenant the operator can reason about.
9. **Branch on `kind`, not on holder strings,** when another lease is in your way: `benchmark`
   means stand down; `render` and `agent` mean wait `retry_after_s` and re-ask; `other` means make
   no assumption (OPENCLAW-RIG.md §6).

## 7. The HTTP contract

Auth is StudioForge's (D32): reads are open on the LAN; every mutation below needs a caller on the
box, the MCP PIN (`X-MCP-Pin`), or `server.api_key` as a bearer when one is set. Errors are the
OpenAI envelope: `{"error": {"message", "type", "code", "param", "studioforge": {...details}}}`;
every retryable refusal also carries a `Retry-After` header.

### `POST /api/leases` — acquire

```json
{
  "devices": "auto",
  "holder": "speechforge",
  "priority": 2,
  "reason": "voice session",
  "idle_ttl_s": 1200,
  "model_ids": [],
  "force": false,
  "exclusive": true,
  "min_free_gib": 8,
  "prefer": [3, 2],
  "avoid": [0],
  "vacate_url": "http://127.0.0.1:1236/v1/speech/vacate",
  "vacate_token": "<minted by you, printable ASCII, no whitespace>"
}
```

`devices` is a list of CUDA indices, `"auto"`, or omitted (= auto). `min_free_gib`, `prefer` and
`avoid` steer an automatic placement only (a 400 beside explicit devices). `exclusive` must be
`true` (§10). `model_ids` names StudioForge's own models that may be planned onto the cards; a
tenant that allocates its own memory sends none.

Reply `200`: the lease record —

```json
{
  "id": "ab12cd34ef56", "devices": [3], "holder": "speechforge", "model_ids": [], "reason": "...",
  "priority": 2, "created_at": 1.7e9, "last_activity_at": 1.7e9, "idle_ttl_s": 1200.0,
  "idle_s": 0, "expires_at": 1.7e9, "state": "active", "holder_family": "speechforge",
  "kind": "other", "retry_after_s": 300, "vacate_registered": true, "vacate_url": "...",
  "vacate_deadline": null, "vacate_requested_by": null, "restored_at": null,
  "placement": {"mode": "auto", "devices": [3], "considered": [
    {"device": 3, "name": "...", "free_gib": 23.5, "reclaimable_gib": 0.0, "projected_free_gib": 23.5,
     "residents": [], "evictions": 0, "verdict": "chosen", "detail": null},
    {"device": 0, "verdict": "avoided", "detail": "named in avoid", "...": "..."}
  ]}
}
```

`placement` is present only for an automatic placement. `vacate_url` is echoed to the registrant
and to admin callers; the token is never shown to anyone.

| Status | `code` | Meaning | Do |
| --- | --- | --- | --- |
| 200 | — | granted | keep `id`; touch; release |
| 400 | `invalid_request_error` (param named) | a bad field: unknown device, `devices` word, steering beside explicit devices, `vacate_token` without a URL, a `vacate_url` onto StudioForge's own listener | fix the request |
| 400 | `unsupported` | `exclusive: false` asked for a share lease (§10) | lease a whole card |
| 400 | `placement_unavailable` | auto asked and the GPU probe did not answer | name the devices |
| 409 | `lease_conflict` | a card is leased by an equal or better class, by a holder with no vacate URL, or the ask lapsed (`vacate.state`: `timed_out` / `undeliverable`); or a pinned / better-class idle resident is on it (`param: force`) | do not force; wait `retry_after_s` only if you expect a release, else lease elsewhere |
| 409 | `lease_vacating` | the holder has been asked to leave | re-ask every `retry_after_s` until granted or `lease_conflict` |
| 503 | `model_busy` | a resident on the card is mid-request, or a load is in flight | wait `retry_after_s`, re-ask |
| 507 | `insufficient_vram` | auto with `min_free_gib`: no card would have that much | ask for less, or wait |

Every automatic refusal carries `error.studioforge.considered` (the per-card verdicts:
`chosen`, `candidate`, `avoided`, `leased`, `busy`, `pinned`, `better_tier_resident`,
`too_little_free`) and `error.studioforge.placement: "auto"`.

### `POST /api/leases/{id}/touch` — keep alive

Reply `200` with the lease record; `404 lease_not_found` when it is gone.

### `DELETE /api/leases/{id}` — release

Reply `200 {"released": {...}}`; `404 lease_not_found` when already gone (treat as success).

### `GET /api/leases` — the book

`{"leases": [...], "count": n}`, every row as above (URLs only to admin callers). Open to read.

### `GET /api/capabilities` — feature gating (D64)

`implemented.features` contains `lease_vacate` (D56), `lease_persistence` (D61),
`tenant_registry` and `tenant_protocol_vt1` (D77), `lease_auto_placement` (D78). Gate on these,
not on the version string.

## 8. The host rule for every URL you register

A `health_url`, `warm_url`, `status_url` or `vacate_url` is a URL StudioForge will later call on
your behalf, so the set of hosts it may name is closed: **loopback** (every spelling), the
**tailnet range `100.64.0.0/10`**, and hosts the operator lists in `tenants.allowed_hosts`
(IP literals, or names compared verbatim). DNS names other than `localhost` are **never
resolved** — rebinding is the classic hole — and anything else is `400 tenant_url_refused`
naming the field. A `vacate_url` additionally may not point at StudioForge's own listeners (D56).
StudioForge calls these with no proxy environment, never follows a redirect, and gives a health
probe 2 s.

## 9. The registry

### Config: `tenants.registry`

```yaml
tenants:
  registry:
    - name: speechforge          # the registry key; should equal your holder_family
      kind: module               # module | tenant
      health_url: http://127.0.0.1:1236/health
      warm_url: http://127.0.0.1:1236/v1/speech/warm   # modules only
      status_url: null           # a richer document, linked, never fetched
      default_priority: 2        # the class your leases normally carry
      notes: speech-to-text and text-to-speech on one card
  health_interval_s: 30.0
  health_timeout_s: 2.0
  allowed_hosts: []
```

A declared tenant appears on the Tenants card with live health before it has ever registered.
`config.example.yaml` carries SpeechForge (a module) and ClawForge2 (a tenant), commented out.

### `POST /api/tenants/register` — announce yourself (idempotent; send at every boot)

```json
{
  "name": "speechforge", "family": "speechforge", "kind": "module", "protocol": "vt.1",
  "version": "0.9.1", "health_url": "http://127.0.0.1:1236/health",
  "warm_url": "http://127.0.0.1:1236/v1/speech/warm",
  "vacate_url": "http://127.0.0.1:1236/v1/speech/vacate",
  "default_priority": 2, "capabilities": ["stt", "tts"], "notes": "..."
}
```

`name` is lower-case (`^[a-z0-9][a-z0-9._-]{0,63}$`); a name the operator declared keeps its
declared `kind`, `family`, `notes` and `default_priority` (a program may not promote itself to a
module) and takes the URLs you send. `protocol`, if sent, must be `vt.1`
(`400 tenant_protocol_unsupported` otherwise). `capabilities` is at most 32 short tags. The reply
is your row as stored, URLs included. **Never send your `vacate_token` here** — it rides on the
lease only.

### `GET /api/tenants`, `GET /api/tenants/{name}` — the rows

```json
{"tenants": [{
  "name": "speechforge", "family": "speechforge", "kind": "module", "source": "config",
  "protocol": "vt.1", "registered": true, "version": "0.9.1", "capabilities": ["stt", "tts"],
  "default_priority": 2, "notes": "...",
  "health": {"state": "healthy", "checked_at": 1.7e9, "status_code": 200, "latency_ms": 4.2,
             "error": null, "summary": {"status": "ready", "version": "0.9.1"}},
  "health_target": "127.0.0.1:1236", "health_registered": true, "status_registered": false,
  "warm_registered": true, "vacate_registered": true,
  "registered_at": 1.7e9, "deregistered_at": null, "last_seen": 1.7e9,
  "leases": [{"...": "a lease row"}], "devices": [3], "lease_count": 1
}], "count": 1, "protocol": "vt.1"}
```

`health.state` is `healthy` (any 2xx), `unhealthy` (answered, not 2xx — a redirect counts),
`unreachable` (no answer; `error` is the exception's class name) or `unpolled`. `summary` is a
bounded lift of `status`/`state`/`ok`/`ready`/`version`/`loaded`/`can_serve`/`device` from a JSON
health body. The URLs themselves are shown to admin callers only; everyone else sees the
`*_registered` booleans and `health_target`. `last_seen` is the later of your last registration
and the last healthy probe. `/api/status` carries the same rows under `tenants`.

### `DELETE /api/tenants/{name}` — goodbye

A self-registered row is removed; a declared row stays, marked `registered: false`.
`404 tenant_not_found` if unknown.

### `POST /api/tenants/{name}/warm` — ask a module to warm

Relays one `POST {"requested_by": "<X-SF-Client or studioforge>", "protocol": "vt.1"}` to the
module's `warm_url` (30 s timeout) and answers `{"tenant", "ok": true, "status_code", "summary"}`.
`400 tenant_not_warmable` for a plain tenant or a module without a `warm_url`;
`502 tenant_unreachable` (details: `error` class name, or `status_code`) when the module does not
answer `2xx`. Nothing is launched (D79).

### Where to see it

The **Tenants** card on the Dashboard (name, kind, health, version, class, cards and leases,
last seen; a **Warm** button on modules behind the usual confirmation), `sfctl tenants list`,
`sfctl tenants warm <name>`, `sfctl status`, and the read-only MCP tools `gpu_tenants` and
`gpu_leases`.

## 10. Extension point: share leases

A lease is a whole card. The partial-card lease — "reserve N GiB on card 3 and let an LLM share
the rest" — was considered when SpeechForge was designed (its DECISIONS S3) and deliberately not
built: a speech model and an LLM sharing one card compete for compute as well as memory, and the
jitter has not been measured. The seam is marked: `POST /api/leases` takes `exclusive` and refuses
`false` with `400 unsupported`; the SDK sends `exclusive: true`. When the day comes, the
`exclusive: false` request gains `reserve_gib`, the planner learns a per-card reservation, and
nothing in a tenant that follows this page changes but one field.

## 11. The SDK

`packages/studioforge-tenant/` — pure Python 3.12, `httpx` only, never imports `studioforge`:

```python
from studioforge_tenant import TenantClient, Granted, Refused, Unreachable, vacate_asgi_route

client = TenantClient("http://127.0.0.1:1234", "speechforge", priority=2,
                      vacate_url="http://127.0.0.1:1236/v1/speech/vacate",
                      vacate_token=minted_token, version="0.9.1",
                      health_url="http://127.0.0.1:1236/health", kind="module",
                      warm_url="http://127.0.0.1:1236/v1/speech/warm")
await client.release_stale()          # at boot
await client.register()               # optional, idempotent
outcome = await client.acquire("auto", prefer=[3, 2], min_free_gib=8)
match outcome:
    case Granted(lease_id=lease_id, devices=[device]): ...   # allocate on `device`; run touch_loop
    case Refused(retry_after_s=wait, retryable=True): ...    # cool down, re-ask
    case Refused(): ...                                      # stand down or go elsewhere
    case Unreachable(): ...                                  # S16: only a clearly free card
app.mount("/v1/speech/vacate", vacate_asgi_route(client.token_matches, drain_and_release))
```

Its README says how SpeechForge and ClawForge2 would adopt it without changing their behaviour.

## 12. Decisions

D43 (leases), D53 (`gpu_leased`, `holder_family`, `kind`), D56 (vacate), D61 (persistence), D65
(a saved override onto a leased card), D77 (this protocol and the registry), D78 (automatic
placement), D79 (StudioForge never launches a tenant) in `DECISIONS.md`.
