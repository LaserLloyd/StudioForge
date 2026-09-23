# The whole rig, for one agent

One box, four GPUs, two servers in front of them. **StudioForge** runs the language models;
**ClawForge2** runs the image models through ComfyUI. An agent that writes text and makes pictures
talks to both, and everything that goes wrong between them — a refusal on one side caused by the
other, an identity that does not match, a card that is spoken for — is on this page.

This is the cross-service page. The single-service references are
[OPENCLAW.md](OPENCLAW.md) (StudioForge's tools and the loop an agent runs),
[OPENCLAW-SETUP.md](OPENCLAW-SETUP.md) (the two-machine install) and ClawForge2's own
`docs/MCP_TOOLS.md`. Nothing here restates them; it says how the two fit together.

Every hostname below is a placeholder. Substitute your own: `<rig-host>` is the machine both
servers run on (a tailnet name survives network changes; an IP does not), `<agent>` is the name you
choose for the agent in the next section.

---

## 1. The rig in one table

| Port | Service | What is there |
| --- | --- | --- |
| `1234` | StudioForge gateway | `/v1` OpenAI-compatible inference, `/api` REST, `/mcp` management MCP |
| `1235` | StudioForge watchdog | `/mcp` recovery tools; a separate process, answers when 1234 does not |
| `8080` | StudioForge control panel | the operator's GUI |
| `8700` | ClawForge2 | `/mcp` (streamable HTTP), the web GUI, `/ui/api/*` REST |
| `8288` | ComfyUI | ClawForge2's backend. **Never talk to it directly** — it has no queue discipline, no lease awareness and no job store |

Which card is whose is a question with a live answer, never a written-down one:
`comfy_status().pin.index` is the card ComfyUI renders on — the LEAD of `pin.devices`, which can
now be a set of more than one card — and `server_status().loaded[].devices` is
where each language model sits. Both change during a day.

## 2. Two registrations, two different files

**StudioForge is a stdio bridge.** The companion CLI (`sfctl`) merges the app's 21 management tools
and the watchdog's 10 recovery tools into one local stdio MCP server, so OpenClaw launches a
process rather than opening a socket:

```bash
openclaw mcp add studioforge --command sfctl --arg mcp
```

Editing OpenClaw's config directly, its key is `mcp.servers` (a nested map), **not** the top-level
`mcpServers` most other clients use. `GET http://<rig-host>:1234/api/openclaw-setup` prints both
spellings, the inference env and the `next_steps` an agent should follow; `sfctl openclaw-setup`
is the same thing from the CLI.

**ClawForge2 is a URL.** It speaks streamable HTTP directly, so it is registered wherever your
gateway keeps HTTP MCP servers — for mcporter, `config/mcporter.json`:

```json
{
  "mcpServers": {
    "clawforge2": {
      "url": "http://<rig-host>:8700/mcp",
      "headers": { "X-MCP-Pin": "<clawforge auth token>" }
    }
  }
}
```

Those are two different files on the agent box, and registering one does not register the other.
Text generation is registered in a third place again — it is not MCP at all, it is
`OPENAI_BASE_URL=http://<rig-host>:1234/v1`.

## 3. Auth: what is open, what is not

**StudioForge.** `/v1` inference is open unless `server.api_key` is set; so are the read endpoints
(`GET /api/status`, `GET /api/leases`, `/health`). Every *box change* — settings, config writes,
`POST`/`DELETE /api/leases`, and every mutating MCP call — is gated: the caller must be on the rig
itself, or send the MCP pairing PIN. `/mcp` always wants that PIN from a remote caller, even with
no API key set. Both credentials go in a header (`Authorization: Bearer …` or `X-MCP-Pin: …`);
`?pin=` in a URL is refused, because URLs end up in logs and history.

**ClawForge2.** A token is optional. Without one, a loopback caller is privileged and a remote
caller is not; with one, a remote caller that sends it (`X-MCP-Pin` or `Authorization: Bearer`) is
privileged too. Privileged unlocks the mutating surface — `import_workflow`, `comfy_control`,
`list_jobs(scope="all")` and the acquisition writes — and un-redacts rows that would otherwise come
back trimmed. Ordinary generation needs no credential at all.

## 4. Identity: one string per agent

Pick one name for the agent — `openclaw-<agent>` — and use the same string in all three places:

| Where | Field | Who sees it |
| --- | --- | --- |
| StudioForge `/v1` | `X-SF-Client: openclaw-<agent>` header | `GET /api/status` → `clients`, the trailing-hour rollup of who sent inference |
| ClawForge2 tools | `client="openclaw-<agent>"` argument | `list_jobs`, the `"last"` shortcut, `client_max_active`, `jobs.per_client` |
| Either lease book | `holder="openclaw-<agent>"` on `reserve_gpus` | every lease view on both servers, as `holder_family` `openclaw` → `kind` `agent` |

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
is indistinguishable from anything else on the tailnet. One caveat worth knowing: the LLM calls
ClawForge2 makes *on your behalf* (prompt enhancement, image captioning) reach StudioForge as
ClawForge2's own client label, not yours — image-driven text work shows up under ClawForge2 in the
rollup however carefully you tag your calls.

## 5. Priority: the same numbers, two different defaults

| Tier | StudioForge (load admission + request admission) | ClawForge2 (queue order) |
| --- | --- | --- |
| 1 | the model a person is actively chatting with; jumps the load queue and holds worse tiers off with `503 priority_hold` | interactive: a person is waiting. Jumps queued work, never interrupts the running render |
| 2 | a dispatched agent's model | normal |
| 3 | background | background |
| **omitted** | **3 — background**, and any tier-1/2 load can hold it off | **2 — normal** |

That difference is the trap: an agent that "leaves priority alone" gets normal service for pictures
and background service for text. **Send `"priority": 2` explicitly in every `/v1` body** (or have
the operator set `settings.priority = 2` on the agent's model, which is the floor a per-request
tier can lift but not lower). Only the companion a human is waiting on sends `1`. On StudioForge
the value is validated: anything that is not 1, 2 or 3 is a `400`.

## 6. Leases: who stands down for whom

A **lease** is how one tenant tells the others to keep off a card. StudioForge owns the book; every
holder reads the same records (`server_status().leases`, `GET /api/leases`,
`comfy_status().leases`), and every record carries `state`, `holder`, `holder_family`, `kind`,
`expires_at` and `retry_after_s`. `state` is `active`, `idle` (quiet for half its TTL, at most
5 min), `expiring` (inside the last quarter of its TTL, at most 5 min) or `vacating` (asked to
leave, D56); `holder_family` is everything before the first `-` or `:` in `holder`, lowercased.

Who takes them: CrucibleForge for a benchmark run; StudioForge's own sweeps
(`test_model`, `benchmark_parallel`) for their duration; ClawForge2 for the card ComfyUI is pinned
to, released when it idles out; and any agent, via `reserve_gpus`.

**Branch on `kind`, and only on `kind`:**

- **`benchmark`** — a sweep owns those cards for the whole run. **Stand down**: no loads, no
  renders on them, and *no polling loop*. Do the work elsewhere or tell the user and stop.
  `retry_after_s` is capped at 300 s and is a re-ask interval, not an estimate of the wait.
- **`render`** — somebody's picture, and pictures take seconds. Wait `retry_after_s`, ask again.
- **`agent`** — another agent's model, pinned to cards. Treat it like `render`. If it is *yours*,
  release it (`release_gpus`) before you ask for a render on those cards.
- **`other`** — make no assumption. Wait once, then report.

Two null-versus-empty distinctions decide whether you stop or continue.
`comfy_status().leases.foreign: null` means ClawForge2 **could not ask** StudioForge — that is not
a reason to stand down; `[]` means it asked and there are none. A lease with no `expires_at` is
held until released, not forever-stuck; its `retry_after_s` is the open-ended default.

Leases are **first-come-first-served and never revoked**: nothing on this rig takes a standing
lease away from its holder. `kind` is descriptive — the server does not enforce your etiquette, it
publishes the facts you need to have it.

### The vacate protocol

Standing down is asked for, never taken. A lease may register a `vacate_url` when it is created;
when a strictly better-class claim arrives for the same cards, StudioForge POSTs one vacate request
to that URL, marks the lease as vacating, and answers the *asker* `409 lease_vacating` with a
re-ask interval rather than holding its request open. Nothing is unloaded, nothing is killed, and a
holder that registered no URL is not asked at all — it simply keeps its cards and the asker gets
the ordinary `409 lease_conflict`.

What the holder does with the request is the holder's business, and it is bounded: ClawForge2
stops accepting new renders, lets the render already inside ComfyUI finish (it never interrupts
one), frees what it can and releases the lease — an adopted ComfyUI keeps its CUDA context, so a
few hundred megabytes stay behind and are reported rather than pretended away. If the deadline
passes with no release, the answer degrades to today's plain `409 lease_conflict` and it becomes an
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

Two calls open a session. Neither loads anything, and together they answer every question that
decides what you do next:

```
comfy_status()                              ~1.5k tokens
    can_render / can_render_reason  - the only honest "may I render" (NOT `reachable`,
                                      NOT `comfy.running`)
    leases.foreign[].kind           - who owns the cards (null != [])
    workflow_names                  - every installed workflow name, ~480 chars
    default_workflow, pin.index     - what you get by default, and on which card
    pin.cannot_run                  - workflows this card cannot run at all

check_loaded_model(min_params="20b", vision=true)      ~0.3k tokens
    answer "yes" -> use the `model` id it returns, load nothing
    answer "no"  -> `reason` names the gap
```

Then, on a loaded row from `server_status()`: `effective.summary` is what the engine was really
launched with and `prompt_cache.hit_ratio` is whether prefix reuse is happening. Never read
`settings` to answer either question — a `null` there means **inherit the default**, not "off".

The expensive habits, and what to do instead: `list_workflows()` (brief, ~3.4k tokens) only when
you are *choosing* a workflow and need media type, VRAM class or aspect-ratio names —
`workflow_names` already answered "does X exist"; `list_workflows(detail="full")` (~10.4k) only for
pixel sizes and per-workflow defaults; `list_models` only after the gate has said "no". Registering
both servers costs roughly 40k tokens of tool schemas before the first call, so an agent that only
makes pictures should be given an allowlist rather than all 54 tools.

## 8. Sessions, restarts and what survives them

**StudioForge over `sfctl mcp`** holds no session: the bridge reconnects for every call, so there
is nothing to lose across a restart. If a management tool reports the server unreachable, the
watchdog is still up — `restart_server`, then retry. After a restart the pinned model comes back on
its own, every other model loads again on its first request, per-model saved settings persist, and
so do standing leases (D61: restored with their clocks; one already idle past its TTL is dropped,
and StudioForge's own benchmark leases are never restored).

**ClawForge2 over HTTP** does hold a session. Reuse your `Mcp-Session-Id` across calls — a client
that re-initialises per call pays for `tools/list` and the instructions every time, and leaks a
session each round. A request carrying an id the server does not know is answered:

```
HTTP 404  {"error": {"code": -32600, "message": "Session not found"}}
```

which means one of two things: the server restarted, or your session went quiet for longer than
`mcp_session_idle_seconds` (600 s) and was reaped. Both have the same fix — run `initialize` again
and use the new id. **Never replay the old one.** Send `DELETE /mcp` when you are finished.
(StudioForge's `/mcp` answers the same 404 to a stale id, for the same reason.)

Across a ClawForge2 restart, **job ids are gone and files are not** — the job store is bounded and
the ids fall off the end anyway. Keep `files_rel` from every result; it is the durable handle.

## 9. Every refusal, and what to do about it

Branch on the **code**, never on the prose. StudioForge puts it in the OpenAI error envelope
(`error.code`, extras under `error.studioforge`); ClawForge2 puts it at the front of the message as
`[code]` and lifts it into `structuredContent.error.code`.

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
| CF2 | tool error | `gpu_leased` | ComfyUI's card is inside a foreign lease; **not submitted**. Carries `holder`, `holder_family`, `lease_kind`, `expires_at`, `retry_after_s` | exactly as SF `gpu_leased` — branch on `lease_kind` |
| CF2 | status field | `can_render_reason` = `cards_leased` | the same fact, before you submit anything | read `leases.foreign[].kind` |
| CF2 | tool error | `client_quota` | your `client` tag already has `client_max_active` jobs in flight | wait `retry_after_s` |
| CF2 | tool error | `insufficient_vram` / `insufficient_compute_cap` | a pre-submit refusal on the pinned card. Not transient | lighter workflow; `pin.cannot_run` lists what cannot run here |
| CF2 | tool error | `backend_unavailable` | ComfyUI unreachable or unstartable | read `can_render_reason`: `circuit_open`/`backoff` → wait `comfy.backoff.retry_in_s`; `unmanaged_backend_down`/`comfyui_path_invalid`/`port_taken`/`no_suitable_gpu` → a human |
| CF2 | job state | `stalled` | ComfyUI accepted the work, moved it along, then went quiet past the ceiling | report; at most one resubmission |
| CF2 | job state | `cancelled` | by you or by an operator | do not retry unless you meant it |
| CF2 | tool error | `workflow_not_found` / `unknown_shot` / `unknown_detector` / `invalid_priority` / `invalid_detail` / `invalid_scope` / `missing_client` / `empty_prompt` | a bad argument. `detail="ful"` is refused, not downgraded | fix the argument |
| CF2 | tool error | `job_not_found` | the bounded job store forgot that id (or a restart did) | use `files_rel` |
| CF2 | warning | `ambiguous_last` | `"last"` was resolved while more than one producer was active — it may be someone else's image | pass `job_id=` instead |
| CF2 | tool error | `captioner_unavailable` | StudioForge's vision model is unreachable or refusing | retry later, or proceed without a caption |
| CF2 | tool error | `not_privileged` | a mutating tool from off-rig with no token | operator |
| CF2 | tool error | `egress_blocked` / `fetch_failed` | the `image_url` egress policy refused the fetch, or it failed | use a public URL, or send bytes |
| CF2 | tool error | `adopted_backend` / `restart_no_op` | `comfy_control` cannot restart or re-pin a ComfyUI it did not start | operator |
| CF2 | HTTP | 404 `Session not found` | session reaped (600 s idle) or the server restarted | `initialize` again; never replay the id |
| CF2 | HTTP | 401 / 403 / 415 | token, cross-site `Origin`, or a missing JSON content type | operator / client config |

The codes that mean **wait and retry unchanged** are exactly: `priority_hold`, `model_busy`,
`benchmark_busy`, `model_benchmarking`, `client_quota`, `lease_vacating`, and a `gpu_leased` whose
lease `kind` is not `benchmark`. Every one of them carries `retry_after_s` or `Retry-After`.
Everything else means *change the request*, *stand down*, or *report*.

---

## 10. The short version (this is the part to hand an agent)

> **You are talking to one GPU rig through two servers.** `studioforge` (31 tools) manages the
> LLM host; text generation is NOT a tool — it is `POST http://<rig-host>:1234/v1/chat/completions`.
> `clawforge2` (24 tools) makes pictures. They share four GPUs, and a **lease** is how one tenant
> tells the others to keep off a card.
>
> **Say who you are, once, the same way everywhere:** `X-SF-Client: <your name>` on every `/v1`
> request, `client="<your name>"` on every ClawForge2 call, and as `holder` if you ever
> `reserve_gpus`.
>
> **Say what your work is:** `priority` 1 = a person is waiting on this turn, 2 = you are a
> dispatched agent, 3 = background. Send `"priority": 2` in every `/v1` body — omitted means
> background there and a chat-tier load can hold you off (`503 priority_hold`). ClawForge2's
> default is already 2. Only a companion a human is talking to sends 1.
>
> **Models load on demand; a pin is not needed for that.** Name any model in a `/v1` request and
> it loads itself (the request waits), then unloads after its idle TTL; the next request loads it
> again with its saved settings. `pin_model` is only for a model that must never pay that cold
> load. An unloaded, unpinned model is idle, not broken. What *does* break on-demand loading is a
> saved `device_override`: it forces exactly those cards, so a lease on any one of them refuses
> every load with `507 gpu_leased`. `plan_load(model_id)` tells you whether the next request will
> load it; a persistent refusal there is an operator fix, not something to retry.
>
> **Start a session with two cheap calls.** `comfy_status()` — read `can_render`,
> `leases.foreign`, `workflow_names` (every installed workflow name; do not call `list_workflows`
> just to check a name), `pin.index` (ComfyUI's card). `check_loaded_model(min_params="20b",
> vision=true)` — `answer=="yes"` gives you the `model` id; load nothing. `"no"` is about what is
> loaded *now*: if you already have a configured model, name it in the request instead — it loads
> on demand, so do not fall back just because nothing is loaded. `list_workflows()`
> (brief) only to choose a workflow; `detail="full"` only for pixel sizes or defaults. Never
> `list_models` before the gate has said "no".
>
> **Read these fields before you act:** `can_render` (not `reachable`, not `comfy.running`);
> `leases.foreign[].kind` and `server_status().leases[].kind`; `busy.priority_hold`;
> `effective.summary` / `prompt_cache.hit_ratio` on a loaded row (a `null` setting means
> *inherit*, not off — never read `settings` to learn whether the prompt cache is on).
>
> **A lease tells you what to do, by `kind`.** `benchmark` — the cards are owned for the run:
> **stand down**. Do not load, do not render on those cards, do not poll in a loop; do the work
> elsewhere or tell the user and stop. `render` — somebody's picture: wait `retry_after_s`
> (seconds) and ask again. `agent` — another agent's model: wait like `render`; if it is yours,
> release it before rendering. `other` — make no assumption: wait once, then report.
> `leases.foreign: null` means ClawForge2 could not ask StudioForge — that is not a reason to stop.
>
> **Branch on the code, never on the prose.** StudioForge: `error.code` in the OpenAI envelope
> (extras under `error.studioforge`); ClawForge2: `[code]` at the front of the message and
> `structuredContent.error.code`. The only codes that mean *wait and retry unchanged* are
> `priority_hold`, `model_busy`, `benchmark_busy`, `model_benchmarking`, `client_quota`,
> `lease_vacating`, and a `gpu_leased` whose lease `kind` is `render`/`agent`/`other` — all carry
> `retry_after_s` / `Retry-After`. `gpu_leased` with `kind: benchmark` means stand down.
> `insufficient_vram`, `insufficient_compute_cap`, `context_exceeded`, `workflow_not_found`,
> `model_not_found`, `invalid_priority`, `invalid_detail` mean *change the request*;
> `unsupported_architecture` means *change the model* — this llama.cpp build cannot load it.
> `model_load_failed`, `backend_unavailable`, `stalled` mean *report*, then at most one
> resubmission.
>
> **Jobs.** Every picture call is a job; `wait=false` for batches; a `job_id` coming back is the
> slow path, not an error; `get_job(job_id)`; keep `files_rel` (job ids die on restart, files do
> not); pass `job_id=` to `identify_image`/`crop_to_face` — `"last"` is a guess. Cancel what you
> abandon. Never retry a call that has not returned.
>
> **Sessions.** Reuse your `Mcp-Session-Id`; a `404 Session not found` means ClawForge2 restarted
> or your session idled out (10 min) — run `initialize` again, never replay the old id. The
> StudioForge tools are a local stdio bridge and reconnect on every call; if a management tool
> says the server is unreachable, call `restart_server` (watchdog), then retry.

---

## 11. Verifying the pair

```bash
curl -s -m 8 http://<rig-host>:1234/health      # StudioForge
curl -s -m 8 http://<rig-host>:8700/health      # ClawForge2
sfctl status                                     # the rig through the companion
```

Then eyeball four fields, which between them cover most of what goes wrong:
`comfy_status().can_render` (**not** `reachable`), `comfy_status().leases.foreign`,
`server_status().busy.priority_hold`, and `server_status().vram_orphan_count` — above zero means
leaked engine processes are holding VRAM, and the watchdog's `reclaim_orphan_engines` kills exactly
those and nothing else.
