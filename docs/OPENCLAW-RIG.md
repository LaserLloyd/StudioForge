# The whole rig, for one agent

One box, four GPUs, one language-model server in front of them. **StudioForge** runs the language
models; other programs on the same box (an image-generation front end over ComfyUI, a speech
server, a benchmark harness) share the same cards by taking **leases** from it. Everything that
goes wrong between an agent and that arrangement — a refusal caused by another tenant, an identity
that does not match, a card that is spoken for — is on this page.

This is the cross-program page. The single-service references are
[OPENCLAW.md](OPENCLAW.md) (StudioForge's tools and the loop an agent runs) and
[OPENCLAW-SETUP.md](OPENCLAW-SETUP.md) (the two-machine install). Nothing here restates them; it
says how StudioForge fits with the other programs that share its GPUs. Other programs on the rig
that lease cards document their own tools and errors.

Every hostname below is a placeholder. Substitute your own: `<rig-host>` is the machine
StudioForge runs on (a tailnet name survives network changes; an IP does not), `<agent>` is the
name you choose for the agent in the next section.

---

## 1. The rig in one table

| Port | Service | What is there |
| --- | --- | --- |
| `1234` | StudioForge gateway | `/v1` OpenAI-compatible inference, `/api` REST, `/mcp` management MCP |
| `1235` | StudioForge watchdog | `/mcp` recovery tools; a separate process, answers when 1234 does not |
| `8080` | StudioForge control panel | the operator's GUI |
| `8288` | ComfyUI | the image models' backend, owned by whichever program fronts it. **Never talk to it directly** — it has no queue discipline, no lease awareness and no job store |

Which card is whose is a question with a live answer, never a written-down one:
`server_status().loaded[].devices` is where each language model sits and
`gpu_leases()` (or `GET /api/leases`) is which cards are spoken for. Both change during a day.

## 2. Registration

**StudioForge is a stdio bridge.** The companion CLI (`sfctl`) merges the app's 27 management tools
and the watchdog's 10 recovery tools into one local stdio MCP server, so OpenClaw launches a
process rather than opening a socket:

```bash
openclaw mcp add studioforge --command sfctl --arg mcp
```

Editing OpenClaw's config directly, its key is `mcp.servers` (a nested map), **not** the top-level
`mcpServers` most other clients use. `GET http://<rig-host>:1234/api/openclaw-setup` prints both
spellings, the inference env and the `next_steps` an agent should follow; `sfctl openclaw-setup`
is the same thing from the CLI.

Text generation is registered in a different place again — it is not MCP at all, it is
`OPENAI_BASE_URL=http://<rig-host>:1234/v1`. Any other program on the rig that has MCP tools of its
own is registered by that program's own instructions.

## 3. Auth: what is open, what is not

**StudioForge.** `/v1` inference is open unless `server.api_key` is set; so are the read endpoints
(`GET /api/status`, `GET /api/leases`, `/health`). Every *box change* — settings, config writes,
`POST`/`DELETE /api/leases`, and every mutating MCP call — is gated: the caller must be on the rig
itself, or send the MCP pairing PIN. `/mcp` always wants that PIN from a remote caller, even with
no API key set. Both credentials go in a header (`Authorization: Bearer …` or `X-MCP-Pin: …`);
`?pin=` in a URL is refused, because URLs end up in logs and history.

## 4. Identity: one string per agent

Pick one name for the agent — `openclaw-<agent>` — and use the same string in both places:

| Where | Field | Who sees it |
| --- | --- | --- |
| StudioForge `/v1` | `X-SF-Client: openclaw-<agent>` header | `GET /api/status` → `clients`, the trailing-hour rollup of who sent inference |
| The lease book | `holder="openclaw-<agent>"` on `reserve_gpus` | every lease view, as `holder_family` `openclaw` → `kind` `agent` |

OpenClaw sends static extra headers per provider (`models.providers.<id>.headers`; check the key
names against your version):

```json
{
  "models": {
    "providers": {
      "studioforge": {
        "baseUrl": "http://<rig-host>:1234/v1",
        "api": "openai-completions",
        "headers": { "X-SF-Client": "openclaw-<agent>" }
      }
    }
  }
}
```

Without the header, StudioForge attributes the traffic to a peer IP, and an agent's own text work
is indistinguishable from anything else on the tailnet. One caveat worth knowing: LLM calls that
another program makes *on your behalf* reach StudioForge under that program's own client label,
not yours.

## 5. Priority: how StudioForge ranks requests

| Tier | StudioForge (load admission + request admission) |
| --- | --- |
| 1 | the model a person is actively chatting with; jumps the load queue and holds worse tiers off with `503 priority_hold` |
| 2 | a dispatched agent's model |
| 3 | background |
| **omitted** | **3 — background**, and any tier-1/2 load can hold it off |

That is the trap: an agent that "leaves priority alone" gets background service for text.
**Send `"priority": 2` explicitly in every `/v1` body** (or have
the operator set `settings.priority = 2` on the agent's model, which is the floor a per-request
tier can lift but not lower). Only the companion a human is waiting on sends `1`. On StudioForge
the value is validated: anything that is not 1, 2 or 3 is a `400`.

## 6. Leases: who stands down for whom

A **lease** is how one tenant tells the others to keep off a card. StudioForge owns the book; every
holder reads the same records (`server_status().leases`, `GET /api/leases`),
and every record carries `state`, `holder`, `holder_family`, `kind`,
`expires_at` and `retry_after_s`. `state` is `active`, `idle` (quiet for half its TTL, at most
5 min), `expiring` (inside the last quarter of its TTL, at most 5 min) or `vacating` (asked to
leave, D56); `holder_family` is everything before the first `-` or `:` in `holder`, lowercased.

Who takes them: CrucibleForge for a benchmark run; StudioForge's own sweeps
(`test_model`, `benchmark_parallel`) for their duration; an image-generation tenant for the card
ComfyUI is pinned to, released when it idles out; and any agent, via `reserve_gpus`.

**Branch on `kind`, and only on `kind`:**

- **`benchmark`** — a sweep owns those cards for the whole run. **Stand down**: no loads, no
  renders on them, and *no polling loop*. Do the work elsewhere or tell the user and stop.
  `retry_after_s` is capped at 300 s and is a re-ask interval, not an estimate of the wait.
- **`render`** — somebody's picture, and pictures take seconds. Wait `retry_after_s`, ask again.
- **`agent`** — another agent's model, pinned to cards. Treat it like `render`. If it is *yours*,
  release it (`release_gpus`) before you ask for a render on those cards.
- **`other`** — make no assumption. Wait once, then report.

A lease with no `expires_at` is held until released, not forever-stuck; its `retry_after_s` is the
open-ended default.

Leases are **first-come-first-served and never revoked**: nothing on this rig takes a standing
lease away from its holder. `kind` is descriptive — the server does not enforce your etiquette, it
publishes the facts you need to have it.

**Who the holders are.** `gpu_tenants()` (or `GET /api/tenants`) lists every program that shares the
cards — declared by the operator or self-registered — with its `family` (what its leases carry as
`holder_family`), `kind`, live `health` and the leases it holds right now; `gpu_leases()` is the
lease book on its own. A tenant that is `unreachable` is simply not running, and StudioForge never
starts one. The contract those programs follow is [TENANTS.md](TENANTS.md).

### The vacate protocol

Standing down is asked for, never taken. A lease may register a `vacate_url` when it is created;
when a strictly better-class claim arrives for the same cards, StudioForge POSTs one vacate request
to that URL, marks the lease as vacating, and answers the *asker* `409 lease_vacating` with a
re-ask interval rather than holding its request open. Nothing is unloaded, nothing is killed, and a
holder that registered no URL is not asked at all — it simply keeps its cards and the asker gets
the ordinary `409 lease_conflict`.

What the holder does with the request is the holder's business, and it is bounded: an
image-generation tenant, for example, stops accepting new renders, lets the render already inside
ComfyUI finish (it never interrupts one), frees what it can and releases the lease — an adopted
ComfyUI keeps its CUDA context, so a few hundred megabytes stay behind and are reported rather
than pretended away. If the deadline passes with no release, the answer degrades to today's plain `409 lease_conflict` and it becomes an
operator problem. **For an agent, both 409s mean the same thing: do not force, do not loop.**
`lease_vacating` is a wait — re-send the same request every `retry_after_s` until it is granted
or the answer becomes `lease_conflict` with an `error.studioforge.vacate` block. Its `state` is one
of three words: `"vacating"` only ever rides on `lease_vacating` (the ask is out, the window is
open); `"timed_out"` says the holder heard the ask and ignored it for the whole window;
`"undeliverable"` says the holder never heard it — the POST failed, and the window was closed on
the spot so you are not left re-asking for a release that cannot come. In both closing cases
`vacate.reask_at` says when the holder may be asked again (one quiet window later), and each lease
row carries `vacate_delivery` (`delivered` / `failed`) and `vacate_delivery_status` (the HTTP code
or the error class) so you can see which case you are in. A plain `lease_conflict` on the *first*
ask means nobody could be asked: an equal or better class holds the cards, or the holder registered
no `vacate_url`. `lease_conflict` is a no — read `server_status().leases` and decide whether the
work belongs on this rig at all.

## 7. What to read first, and what it costs

One call opens a session. It loads nothing, and it answers the question that decides what you do
next:

```
check_loaded_model(min_params="20b", vision=true)      ~0.3k tokens
    answer "yes" -> use the `model` id it returns, load nothing
    answer "no"  -> `reason` names the gap
```

Then, on a loaded row from `server_status()`: `effective.summary` is what the engine was really
launched with and `prompt_cache.hit_ratio` is whether prefix reuse is happening. Never read
`settings` to answer either question — a `null` there means **inherit the default**, not "off".

The expensive habit, and what to do instead: `list_models` only after the gate has said "no".
`server_status().leases` is the cheap way to see whether a card you want is spoken for.

## 8. Sessions, restarts and what survives them

**StudioForge over `sfctl mcp`** holds no session: the bridge reconnects for every call, so there
is nothing to lose across a restart. If a management tool reports the server unreachable, the
watchdog is still up — `restart_server`, then retry. After a restart the pinned model comes back on
its own, every other model loads again on its first request, per-model saved settings persist, and
so do standing leases (D61: restored with their clocks; one already idle past its TTL is dropped,
and StudioForge's own benchmark leases are never restored).

A request to StudioForge's `/mcp` carrying a session id the server does not know is answered:

```
HTTP 404  {"error": {"code": -32600, "message": "Session not found"}}
```

which means one of two things: the server restarted, or your session went quiet for long enough
to be reaped. Both have the same fix — run `initialize` again and use the new id. **Never replay
the old one.** Send `DELETE /mcp` when you are finished.

## 9. Every refusal, and what to do about it

Branch on the **code**, never on the prose. StudioForge puts it in the OpenAI error envelope
(`error.code`, extras under `error.studioforge`).

| Service | HTTP / channel | Code | What it means | What to do |
| --- | --- | --- | --- | --- |
| SF | 507 | `gpu_leased` | cards leased to someone else; `error.studioforge.lease` has `kind`, `holder_family`, `retry_after_s`, `expires_at`, plus a `Retry-After` header | `kind: benchmark` → **stand down**; otherwise wait `retry_after_s` and re-ask. If it keeps coming back for a model you only *use* (you never asked for cards), suspect a saved `device_override` on that model touching a leased card: `plan_load(model_id)` shows it, and the operator clears it — see [OPENCLAW.md](OPENCLAW.md#on-demand-models-the-default) |
| SF | 507 | `insufficient_vram` | it genuinely does not fit; `suggestions`, `max_ctx_that_fits`, `max_parallel_that_fits` | load smaller / shorter context / cheaper KV. Never retry unchanged |
| SF | 507 + `busy_models` | `insufficient_vram` | busy, not full — those models would free the VRAM but are mid-request | wait `retry_after_s` |
| SF | 507 | `allowed_devices_unavailable` | an `allowed_devices` — the model's saved setting, or the one this request sent — names no usable card, and no lease is why | if you sent one, widen it; otherwise an operator setting, so report |
| SF | 503 | `priority_hold` | a tier-1/2 load is in flight; `details.priority_hold` names it | honour `Retry-After`, resend at your true tier |
| SF | 503 | `model_busy` / `benchmark_busy` / `model_benchmarking` | serving, benchmarking or smoke-testing | wait `retry_after_s` |
| SF | 400 | `context_exceeded` | prompt larger than the loaded slot (`ctx_per_slot`; `prompt_tokens` when measured). Nothing is ever truncated | shorten, or `load_recommended` at a larger `ctx_size` |
| SF | 400 | `unsupported_architecture` | the installed llama.cpp build cannot load that model's architecture (`error.studioforge.architecture`, `engine_tag`, `source`, `remedy`); refused before any lease, eviction or spawn, and marked `arch_supported: false` in `list_models` / `/v1/models` | never retry; use another model and report it |
| SF | 400 | `invalid_config` / a rejected `priority` | the request is malformed | fix the body |
| SF | 404 | `model_not_found` | unknown id or alias | fix the id (`list_models`) |
| SF | 404 | `no_loaded_model` | you named `loaded`, and no model of that route's kind is resident or even loading | load a model, or name one explicitly |
| SF | 409 | `lease_conflict` | `reserve_gpus` overlaps a standing lease | read `server_status().leases`; do not force |
| SF | 409 | `lease_vacating` | that holder has been asked to stand down and has not finished | re-send every `retry_after_s`; stop when it becomes `lease_conflict` (`vacate.state: "timed_out"` — it ignored the ask; `"undeliverable"` — it never heard it) |
| SF | 403 | `remote_admin_requires_credential` | a box change from off-rig without the PIN or key | the operator's call; do not retry |
| SF | 502 | `model_load_failed` / `upstream_error` | the engine failed to start, or faulted | report (the stderr tail is in the message); `restart_server` only if `/health` is wedged |
| SF | 401 | `invalid_api_key` / `invalid_mcp_pin` | credential | operator |

Other programs on the rig that lease cards document their own errors. If one of them reports a
card as leased, the lease `kind` rules above apply to it unchanged.

The codes that mean **wait and retry unchanged** are exactly: `priority_hold`, `model_busy`,
`benchmark_busy`, `model_benchmarking`, `lease_vacating`, and a `gpu_leased` whose
lease `kind` is not `benchmark`. Every one of them carries `retry_after_s` or `Retry-After`.
Everything else means *change the request*, *stand down*, or *report*.

**Tiers, idle timeouts and the retry split, in one table** (StudioForge; the shipped policy is
`models.ttl_by_priority: {1: 900, 2: 900, 3: 600}` with `models.default_ttl_s: 600`, D60):

| | What | Idle TTL / what to do | Retry unchanged? |
| --- | --- | --- | --- |
| tier 1 | the model a person is chatting with; holds tiers 2–3 off with `503 priority_hold` and may displace their idle residents | 900 s | — |
| tier 2 | a dispatched agent's model; holds tier 3 off | 900 s | — |
| tier 3 (and **omitted**) | background; holds nobody off, first to be evicted | 600 s | — |
| any tier | a request-level `ttl` shortens the timer and never lengthens it (D61); `settings.ttl_s: 0` is pinned and never idles out | — | — |
| **retry-after** | `priority_hold` 503, `model_busy` 503, `benchmark_busy` 503, `model_benchmarking` 503, `lease_vacating` 409, `gpu_leased` 507 with lease `kind` ≠ `benchmark` | wait `retry_after_s` / `Retry-After`, then resend the same request | **yes** |
| **terminal** | `unsupported_architecture` 400 (the model's architecture is not in the active llama.cpp build), `context_exceeded` 400, `invalid_config` 400, `model_not_multimodal` 400, `model_not_found` 404, `no_loaded_model` 404, `insufficient_vram` 507, `allowed_devices_unavailable` 507, `gpu_leased` 507 with `kind: benchmark`, `lease_conflict` 409, `model_load_failed` 502, `upstream_error` 502, `upstream_timeout` 504, `remote_admin_requires_credential` 403, `invalid_api_key` / `invalid_mcp_pin` 401 | change the request, stand down, or report | **no** |

---

## 10. The short version (this is the part to hand an agent)

> **You are talking to one GPU rig.** `studioforge` (37 tools) manages the LLM host; text
> generation is NOT a tool — it is `POST http://<rig-host>:1234/v1/chat/completions`. Other
> programs share the four GPUs, and a **lease** is how one tenant tells the others to keep off a
> card.
>
> **Say who you are:** `X-SF-Client: <your name>` on every `/v1` request, and as `holder` if you
> ever `reserve_gpus`.
>
> **Say what your work is:** `priority` 1 = a person is waiting on this turn, 2 = you are a
> dispatched agent, 3 = background. Send `"priority": 2` in every `/v1` body — omitted means
> background, and a chat-tier load can hold you off (`503 priority_hold`). Only a companion a
> human is talking to sends 1.
>
> **Models load on demand; a pin is not needed for that.** Name any model in a `/v1` request and
> it loads itself (the request waits), then unloads after its idle TTL; the next request loads it
> again with its saved settings. `pin_model` is only for a model that must never pay that cold
> load. An unloaded, unpinned model is idle, not broken. What *does* break on-demand loading is a
> saved `device_override`: it forces exactly those cards, so a lease on any one of them refuses
> every load with `507 gpu_leased`. `plan_load(model_id)` tells you whether the next request will
> load it; a persistent refusal there is an operator fix, not something to retry.
>
> **Start a session with one cheap call.** `check_loaded_model(min_params="20b", vision=true)` —
> `answer=="yes"` gives you the `model` id; load nothing. `"no"` is about what is loaded *now*: if
> you already have a configured model, name it in the request instead — it loads on demand, so do
> not fall back just because nothing is loaded. Never `list_models` before the gate has said "no".
>
> **Read these fields before you act:** `server_status().leases[].kind`; `busy.priority_hold`;
> `effective.summary` / `prompt_cache.hit_ratio` on a loaded row (a `null` setting means
> *inherit*, not off — never read `settings` to learn whether the prompt cache is on).
>
> **A lease tells you what to do, by `kind`.** `benchmark` — the cards are owned for the run:
> **stand down**. Do not load, do not render on those cards, do not poll in a loop; do the work
> elsewhere or tell the user and stop. `render` — somebody's picture: wait `retry_after_s`
> (seconds) and ask again. `agent` — another agent's model: wait like `render`; if it is yours,
> release it before asking for those cards. `other` — make no assumption: wait once, then report.
>
> **Branch on the code, never on the prose:** `error.code` in the OpenAI envelope (extras under
> `error.studioforge`). The only codes that mean *wait and retry unchanged* are
> `priority_hold`, `model_busy`, `benchmark_busy`, `model_benchmarking`,
> `lease_vacating`, and a `gpu_leased` whose lease `kind` is `render`/`agent`/`other` — all carry
> `retry_after_s` / `Retry-After`. `gpu_leased` with `kind: benchmark` means stand down.
> `insufficient_vram`, `context_exceeded`, `model_not_found`, `invalid_config` mean *change the
> request*; `unsupported_architecture` means *change the model* — this llama.cpp build cannot
> load it. `model_load_failed` means *report*.
>
> **Sessions.** The StudioForge tools are a local stdio bridge and reconnect on every call; if a
> management tool says the server is unreachable, call `restart_server` (watchdog), then retry. A
> `404 Session not found` from `/mcp` means the server restarted or your session idled out — run
> `initialize` again, never replay the old id.

---

## 11. Verifying

```bash
curl -s -m 8 http://<rig-host>:1234/health      # StudioForge
sfctl status                                     # the rig through the companion
```

Then eyeball three fields, which between them cover most of what goes wrong:
`server_status().leases`, `server_status().busy.priority_hold`, and
`server_status().vram_orphan_count` — above zero means
leaked engine processes are holding VRAM, and the watchdog's `reclaim_orphan_engines` kills exactly
those and nothing else.
