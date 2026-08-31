# Pointing OpenClaw at StudioForge

Two snippets get you running. After them, *[The loop an agent actually runs](#the-loop-an-agent-actually-runs)*
is the section to read if you are the agent; everything else is tuning and troubleshooting.

> Looking for a step-by-step install for a two-machine (rig → agent box) deployment, with
> verification at each step? See [OPENCLAW-SETUP.md](OPENCLAW-SETUP.md) — every hostname and
> address on it is a placeholder to substitute. This page is the shorter reference.
> For the formulas behind the catalog's columns see [CATALOG.md](CATALOG.md), and for
> what long context costs on real models see [OPENCLAW-LONG-CONTEXT.md](OPENCLAW-LONG-CONTEXT.md).

---

## 1. Inference

StudioForge listens on **port 1234** — the same port LM Studio uses — so migrating is a *host*
change, not a host-and-port change.

```bash
OPENAI_BASE_URL=http://<studioforge-host>:1234/v1
OPENAI_API_KEY=<server.api_key, or any non-empty string when auth is disabled>
```

That is the whole inference setup. `GET /v1/models` lists every **downloaded** model, and naming
an unloaded one in a request just-in-time loads it.

> **Port already in use?** If LM Studio is running it owns 1234. Either quit LM Studio, or set
> `server.port` to something else and change the base URL to match. Both can share the same model
> library on disk — StudioForge never moves or copies your files.

## 2. Model management as agent tools

Install the companion on the OpenClaw machine. It is not published to PyPI — build the wheel from
`packages/studioforge-companion` (`uv build --wheel -o dist`) and copy it across:

```bash
uv tool install ./studioforge_companion-<version>-py3-none-any.whl
sfctl servers add rig http://<studioforge-host>:1234 --api-key <key>
```

`--api-key` is not optional in practice: when `server.api_key` is unset the `/mcp` endpoint still
enforces the MCP pairing PIN for any caller that is not on the rig itself — `mcp.pin_required:
false` only relaxes same-machine callers (D32) — so pass the PIN here or every management tool
returns 401. The PIN is in the startup banner and at `GET /api/mcp/info` (served to a remote
caller only when a credential was needed to get there). `sfctl` sends it as the bearer token.
`?pin=` in the URL is **refused**: a URL ends up in proxy logs, browser history and shell history,
and an eight-digit PIN is not a credential to leave lying in them. A connector that can only be
given a URL needs a header field, or `server.api_key` instead.

Then register it as a local stdio MCP server:

```bash
openclaw mcp add studioforge --command sfctl --arg mcp
```

or, editing OpenClaw's config file directly — its key is `mcp.servers`, a nested map, **not** the
top-level `mcpServers` one:

```json
{
  "mcp": {
    "servers": {
      "studioforge": {
        "command": "sfctl",
        "args": ["mcp"]
      }
    }
  }
}
```

Clients that take the generic top-level map instead — Claude Code, Cline, LibreChat and most
others — want this shape:

```json
{
  "mcpServers": {
    "studioforge": {
      "command": "sfctl",
      "args": ["mcp"]
    }
  }
}
```

`sfctl mcp` merges **two** upstream toolsets into one list of 30 tools:

| Tools | Source | Available when the main server is wedged? |
| --- | --- | --- |
| `list_models`, `model_options`, `model_info`, `check_loaded_model`, `load_model`, `load_recommended`, `unload_model`, `pin_model`, `reserve_gpus`, `release_gpus`, `test_model`, `benchmark_parallel`, `search_models`, `repo_details`, `download_model`, `delete_model`, `server_status`, `connection_info`, `get_config`, `set_config` | main app | no |
| `restart_server`, `kill_model`, `nuke_all_models`, `reclaim_orphan_engines`, `tail_logs`, `gpu_status`, `rollback_update`, `recovery_health`, `recovery_get_config`, `recovery_set_config` | watchdog sidecar | **yes** |

That split is the point: when the main server locks up, OpenClaw still holds working tools to
diagnose and restart it. Calling a management tool while the server is down returns an error
*result* naming `restart_server` — not a protocol error that kills the session.

There is deliberately **no inference tool**. Generation belongs on `POST /v1/chat/completions`,
which streams; an MCP tool would buffer whole responses through a JSON-RPC envelope.

Benchmarking has its own playbook, written for the agent: [BENCHMARKING.md](BENCHMARKING.md) —
the two benchmarks, the exact calls, the three rules (quiet box, never mid-session, benchmark at
the context you will run), and how a lease locks the result in. Hand it to OpenClaw as a skill.

The server can print both snippets pre-filled for you:

```bash
sfctl openclaw-setup
```

---

## The loop an agent actually runs

Every number an agent needs to choose a model is in one call. This is the whole sequence.

### 1. `check_loaded_model(...)` — is a load even needed?

Before choosing or loading anything, ask whether what is **already** loaded meets the need. The
operator usually has a model resident and most loaded models can do most jobs, so this is the cheap
question that avoids a 40-second load, or a call out to a cloud provider, you did not have to make:

```
check_loaded_model(min_params="20b")                 # at least 20 billion parameters
check_loaded_model(min_params="7b", vision=true)     # ...and it must read images
check_loaded_model(tags=["coding"])                  # any size, must be a coding model
```

`min_params` is in **billions** — `20`, `"20"` and `"20b"` are the same bar, `"500m"` is 0.5B. The
five booleans (`vision`, `audio`, `tools`, `thinking`, `uncensored`) are sugar for `tags`, which is
free-form and takes a list or a comma-separated string.

Then branch on `answer`:

| `answer` | What to do |
| --- | --- |
| `"yes"` | Send the work to `POST /v1/chat/completions` with `model` set to the **`model` field of the response**. It is loaded and warm; do not call `load_model`. `connection_info` has the base URL. |
| `"no"` | Read `reason` — "nothing is loaded", "largest loaded model is 4B, below the 20B bar", "no loaded model reports audio", "no loaded model can be verified as 'uncensored'" — then either `load_recommended` something that clears the bar (step 2 below picks it) or fall back to your other provider. |

The answer carries the id on purpose. A bare yes would still cost you a `GET /v1/models` and a guess
between three resident models; `model` goes straight into the next request. `params_b` and
`params_source` say how big and how that was known (`metadata` exact, `name` read off the model id,
`estimated` from file bytes and a few percent out, `unknown`). A mixture-of-experts model also
reports `active_params_b`, its routed per-token count — **the bar is checked against the total**, not
that. `instances[]` is the working for every resident model, each with `meets` and a `why_not` list,
so a "no" is diagnosable without a second call.

**Unknown is a "no".** The gate only ever passes what it can prove. A model whose size cannot be
determined fails every `min_params` bar, and a tag no local source can speak to fails its bar with
`cannot verify 'x'` rather than a confident `no x`. The cost of a false negative is one unnecessary
load; the cost of a false positive is vision work sent to a text-only model.

Where each tag's answer comes from, which is what decides whether an absence is a real "no" or an
"unknown":

| Tag | Source | An absence means |
| --- | --- | --- |
| `vision`, `tools`, `thinking` | The registry's per-model capabilities, derived from the GGUF at scan time (`vision` also passes on a running child's reported modalities, for a model served with a projector the record predates) | **`no`** — the file was inspected and said so |
| `audio`, `video` | Only a *running* child's reported modalities. No GGUF field describes them | `unknown` — a silent child cannot be quoted |
| `uncensored`, `coding`, `roleplay` | Whole name tokens (`abliterated`, `coder`, `rp`, …) and the GGUF `general.tags` array | `unknown` — a name without "uncensored" in it does not prove the model is censored |
| anything else | Generic matcher: the literal tag as a whole name token, or in `general.tags` | `unknown` — same reason |
| `embedding` | Not a gate tag | Answers `no` and points you at `/v1/embeddings`. Embedding models cannot chat, so they are never candidates and are excluded from the pool |

Whole tokens only, both sides: `tags=["qwen"]` matches the `Qwen/` publisher token but **not**
`Qwen3`. New tags need no server change — `tags=["vietnamese"]` or `tags=["medical"]` gets a truthful
`yes`/`unknown` today.

The same answer is on `GET /api/model-gate?min_params=20b&vision=true` for anything that speaks HTTP
rather than MCP — a `GET`, so the D32 admin gate does not apply to it (D52).

### 2. `list_models(limit=N)` — the catalog

Sorted **newest download first**, so `limit=5` means "the models the user most recently got",
which is usually what "my new model" means. Each model carries `options`: one row per context
tier, and exactly one row is `recommended: true`.

Read `catalog_hint` once — it defines every column — then read the recommended row:

| Field | What to do with it |
| --- | --- |
| `ctx_per_slot` | Context **each** conversation gets. `--ctx-size` is the total across slots, so the engine is launched with `ctx_per_slot × parallel`. |
| `fits` | Will it load *right now*. If false, check `if_gpus_idle` on the same row. |
| `devices` | Which CUDA indices it would use. Not an argument — the planner decides. |
| `max_parallel`, `parallel_limited_by` | How many conversations the placement sustains, and what caps it: `vram`, `knee` (extra slots stop buying throughput), or `cap`. Match your own concurrency to it; beyond it llama.cpp queues rather than refusing. |
| `est_gen_tps` | One stream with ~8k tokens in the window — an ordinary turn. |
| `est_gen_tps_full_ctx` | The same stream with the window nearly full. Generation slows as context fills, so **the truth is between the two**, and how far apart they sit is the real price of a wide row. |
| `confidence` | `measured` (this exact placement was observed), `calibrated` (corrected by a learned factor), `estimated` (nominal vendor bandwidth/FLOPS — an order of magnitude, not a promise). |
| `load_args` | Pass **verbatim** to `load_model`. Do not add fields or recompute anything. |

The model entry (not the row) carries `attention_kind`, which explains why one model's 262k row is
cheap and another's does not exist:

- `full` — every layer holds the whole window. Context is expensive and gets more so.
- `iswa` — Gemma 3/4: five sliding-window layers per full one, so most of the window is never
  cached. Wide rows stay cheap *and* keep multiple slots.
- `hybrid` — Qwen3.5/3.6/3.8: only every 4th layer holds a KV cache at all, the rest are recurrent
  with a fixed per-sequence state. Same effect, different mechanism.
- `unknown` — the geometry could not be derived. Read that as "distrust these numbers", **not** as
  "this model is cheap".

**The recommendation has a floor.** It is the highest context at or above `models.default_ctx`
(raised to `models.thinking_default_ctx` for a thinking model; both are **32768** on the reference
rig — check yours with `get_config`), preferring a row that also sustains two slots. The floor is
never traded away to buy a second slot: a queued second conversation is a latency problem, but a
window that cannot hold an agent's tool transcript is a failed task. `recommended_basis` names
which rule fired, and says `(below floor)` out loud when a model's own trained window cannot reach
it.

### 3. `load_model(**row["load_args"])`

Returns once the child is serving. Loading may evict other **idle** models to make room; that is
normal and is reported in the plan's `evict_model_ids`. A model with in-flight requests is never
evicted.

You often do not need this call at all — see the next step — so reach for it to pre-warm a model,
or to load one at a specific context size and slot count.

**Say what the load is for.** `priority` tiers it: `1` the model a person is actively chatting
with, `2` a model a dispatched agent will use, `3` — or omitted — background work. A tier-1/2 load
is planned as if the idle worse-tier residents were already gone, so it gets the placement an idle
box would have given it; it jumps the one-at-a-time load queue; and while it runs, loads and
inference for worse-tier models are refused with `503` + `Retry-After` (the `priority_hold` row in
[the error table](#what-makes-this-reliable-for-an-agent-workload)). Background can never displace a
chat or agent model. The tier is remembered per model, and a model whose saved settings carry
`priority` keeps its tier across a **restart** of the server as well — so a bot that tiers its
model once keeps it through an engine update or a crash (D46, D48). Nothing else needs to change:
omitting `priority` is background, exactly as before it existed.

A single *request* can state a tier too, in the chat-completions body:

```json
{"model": "...", "messages": [...], "priority": 1}
```

That gates this request's own admission — a tier-1 turn passes a hold that would 503 a background
turn for the same model — and tiers the just-in-time load it triggers, **only upwards**: the load
runs at the better of the tier you asked for and the model's standing one, so a background-tagged
turn against the model a person is chatting with cannot cold-load it at tier 3 and leave it there.
It deliberately does **not** re-tier a model that is already running either: one client cannot
promote the instance everybody else is sharing. `1`, `2`, `3` or absent; anything else is a `400`
naming the parameter, so a typo cannot quietly demote a chat turn.

**Break for old clients (D48).** Before this field existed, an unrecognised `priority` in a
chat/completions body was forwarded to `llama-server` and ignored. It is now validated and consumed,
so anything outside `1`, `2` and `3` — `true`, `"high"`, and vLLM's `priority: 0` among them — is a
`400` naming the parameter rather than a silent no-op. That break is deliberate: a mistyped tier
that was quietly accepted would demote a chat turn to background and then have it refused during
the next load, which is the failure this field exists to prevent.

**If your client will not send unknown body fields.** Some strict OpenAI SDKs strip or reject
anything outside the documented schema. There is deliberately **no header form** of this field — a
tier is a property of the workload, not of the transport, and a second spelling of it is a second
thing to keep in step. Use the standing setting instead; it survives restarts, and every load of
that model (just-in-time ones included) inherits it:

```bash
sfctl models settings pub/chat-model --set priority=1
```

The same write is `PATCH /api/models/{id}/settings` with `{"priority": 1}`, or the *Load priority*
select in the GUI's model settings dialog. The full resolution ladder is: the explicit `priority` on
this call, else the tier of this model's last **explicit load** (remembered until the server
restarts), else `settings.priority`, else background. So the setting is the floor — a per-request
tier decides how that one turn is admitted and can lift the load it triggers above the floor, never
below it — and nothing about a request is remembered afterwards.

### 4. Inference over HTTP, not MCP

```bash
POST http://<host>:1234/v1/chat/completions
```

Naming an unloaded model **just-in-time loads it** with planner defaults. That is the caveat worth
knowing: a JIT load takes the planner's own choice of context, not the catalog row you were
reading. If you need a specific window or slot count, call `load_model` first — a model that is
already resident is used as-is.

### 5. `model_options(model_id)` when the recommended row is not enough

The full table for one model: every context tier, each with its own `fits`, `devices`,
`kv_cache_type`, `max_parallel` and both speed columns. Use it when the task needs a bigger window
than the recommended row offers, or more concurrency than its `max_parallel`.

The trade-offs are visible rather than guessed: doubling `ctx_per_slot` costs `max_parallel` (by
how much depends on `attention_kind`), a `kv_cache_type` of `q8_0`/`q4_0` buys context back at some
quality cost, and a row spread over more devices is usually slower per token than a single-GPU row.

### 6. `search_models` → `repo_details` → `download_model`

```
search_models(query="qwen3", sort="downloads", newer_than_days=90)
   -> compact rows: repo_id, publisher, downloads, likes, quants, mmproj, file_count
repo_details(repo_id)
   -> per-quant total_gb, fit verdict, and the context matrix
download_model(repo_id, quant)
   -> queued; it appears in list_models when it lands
```

**Search results carry no file sizes** — HuggingFace's model-list endpoint does not publish them —
so a search row deliberately says nothing about size or fit, and a quant label alone is not enough
to guess from. `repo_details` is where that is answered: it reads the model's GGUF header remotely
over range requests (a few seconds the first time, then cached) and returns, per quant, the real
size, a fit verdict with a `suggested_quant` when it will not fit, and `context_fit` — for each GPU
placement, which of the 64k/128k/256k/512k tiers actually fit, plus `max_ctx` (the largest window
at a full-quality f16 cache) and `max_ctx_q8` when a q8_0 cache reaches further. Tiers above the
model's trained window are absent, never offered.

The matrix is computed with the same planner a real load uses, so it cannot promise a context the
loader would then refuse. A real example, `unsloth/Qwen3.8-27B-GGUF` on 2× 5090 + 2× 3090:

| quant | 1× 5090 | 2× 5090 | all 4 |
| --- | --- | --- | --- |
| BF16 (51.8 GiB) | — (weights alone do not fit) | 32k at q8_0 | 256k |
| Q8_0 (27.9 GiB) | — | 256k | 256k |
| Q5_K_M (19.3 GiB) | 128k at q8_0 | 256k | 256k |
| IQ2_M (10.5 GiB) | 256k | 256k | 256k |

Downloads run in the background, survive a restart and resume from the partial file — five attempts
with exponential backoff, and 404/401/403 or a checksum mismatch fails immediately rather than
being retried into a hang. Per-file progress is on `GET /api/downloads`; on a stumble the same rows
carry `attempt`/`max_attempts`, `retry_in_s`, `last_error`, and `part_bytes` — how much a Resume
would keep, which is the question a failed download used to leave unanswerable.
`server_status.active_downloads` is just the count.

`search_models` caps `limit` at 25. `truncated: true` means the date window holds more matches than
were walked (HuggingFace has no server-side date filter, so a windowed search pages until a cap);
narrow the query rather than trusting the tail. `trending_score` appears only with
`sort="trending"` — HF omits the field entirely under any other ordering, so its absence is not a
zero.

### 7. `pin_model` — the model that must always answer

```
pin_model(model_id="pub/agent-model")                 # keep loaded at all times
pin_model(model_id="pub/agent-model", pinned=false)   # undo
```

A pinned model has no idle TTL and is never evicted to make room for another load. With
`models.auto_load_pinned` on (the default) it is also loaded at server startup and brought back
automatically (within ~15 s) if it goes down — so pinning an unloaded model also loads it; no
separate `load_model` call is needed. With it off, a pin is only "no TTL, never evicted". This is
the answer to the eviction ping-pong between an agent's workhorse and everything else: pin the
workhorse once and stop re-warming it.

Two things to know. `unload_model` on a pinned model is honoured and it **stays down** until
loaded or pinned again — an explicit unload always wins, so don't unload-then-pin to "restart" a
model (use `load_model(force=true)` for that). Housekeeping unloads are not explicit: `test_model`
and both benchmarks unload and reload the models they measure, and a pinned one is put back by the
reconciler within a sweep of the run ending (D41). And every pin permanently occupies its VRAM, so
each one shrinks what the planner can offer other loads; `server_status` shows the cost, and a
refused load's suggestions will name the pinned models standing in the way. Pinning is a box
change: `pin_model` (or `ttl_s: 0` in saved settings) does it, a request-level `ttl` cannot.

**A card of its own.** When a model must run at full speed with nothing beside it -- or a
card must stay empty for ComfyUI -- take a *lease*:

```
reserve_gpus(devices=[0, 1], model_id="pub/agent-model")   # these two cards are its alone
reserve_gpus(devices=[3], reason="ComfyUI render")           # nothing loads here until released
release_gpus(lease_id="...")                                  # early exit; idle_ttl_s (60 min) is the default one
```

While it stands nobody else is planned onto those cards; the named model is loaded onto exactly
them, sized for as many slots as its context allows, in the split mode its own benchmark measured
fastest there (tensor split is never assumed -- it measured *slower* than layer split on the
reference rig, so benchmark first if you want it considered). Idle residents on the cards are
unloaded; a model mid-request refuses the call, including one that picked up a request between
the scan and its unload; a pinned one needs `force=true`. Two overlapping grants conflict before
either unloads anything. Benchmarks take their own leases, which is what keeps a neighbour's load
out of their numbers. A leased model stays exactly where the lease put it: the D42 rebalancer
never moves it. Any *other* idle model may find itself quietly reloaded onto fewer or less
contended cards once a minute on a quiet box — that is housekeeping (`planner.rebalance`), not
something you did, and it never evicts or interrupts anything.

### 8. `server_status` and `connection_info`

`server_status` is free VRAM per GPU, every loaded model with its port and remaining TTL, queue
depth, the active engine tag — and who is holding the VRAM (next section).

Each loaded row also carries `priority`, the tier that load was made at (`1` chat, `2` agent, `3`
background) — read it beside `loaded_by` to see who would win the cards — and `speculative`, what
drafting actually resolved to for that launch. `null` and `{"type": "none", "reason": ...}` are
different answers: the first means the launch has not decided yet, the second that drafting was
considered and declined, with `reason` saying why (above four slots it always declines itself —
speculation trades spare compute for latency and a saturated batch has none to spare).

`busy` carries one more key beside `active_requests`, `busy_models`, `loading` and `testing`:
`priority_hold` is `{model_id, priority}` while a chat- or agent-tier load is holding worse-tier
traffic off, and `null` otherwise. It is the reason behind any `priority_hold` 503 you have just
been given.

`connection_info` is what to call when the network moves: it returns every address this server
answers on, tailnet first (those survive network changes), with the OpenAI base URL alongside.

---

## "I need a 262144-token window" — `load_recommended`

When what you know is the *window* rather than the hardware, skip the table entirely:

```
load_recommended(model_id="pub/big-model", ctx_size=262144)
```

The server walks its hardware modes in headline order, picks the KV cache type and the slot count
under the same quality-first policy the catalog uses, and loads at **exactly** that context per
conversation. `prefer_mode` ("dual_3090") pins the cards if you care.

It is the one load path that never quietly shrinks your window. Everywhere else a context that does
not fit steps down to one that does; here a window that does not fit is a structured refusal listing,
per set of cards, the largest context that *would* work and what is in the way — a model that is
serving (with `retry_after_s`, so waiting is the right move), memory held by something that is not
us, or the model's own trained window. Ask again with the number it names and it will load. Above
`n_ctx_train` it is a `400` carrying that number, because serving past the trained window needs RoPE
scaling and degrades quality.

Everything else about the call is optional:

| Argument | What it does |
| --- | --- |
| `prefer_mode` | Try this hardware mode first (`"dual_3090"`), instead of the headline order. |
| `kv_min` | The lowest KV cache quality this load may accept — `"f16"`, `"q8_0"` or `"q4_0"`. A floor on the quality-first ladder, not a choice: `"f16"` means "do not quantize the cache to reach this window at all", and a window that would need a worse cache is refused rather than reached quietly at lower quality. |
| `max_slots` | Ceiling on the slot count this load may choose (>= 1). The server's own number is what the placement *could* sustain; if you know only three bots will ever talk to it, cap it here and the KV cache for the slots you will not use is never priced into the fit — which often buys a larger window instead. Below 1 is a `400` naming the parameter. |
| `persist` | Write the profile this call resolved into the model's saved settings (below). |
| `priority` | The load's tier, exactly as on `load_model`: `1` chat, `2` agent, `3` or omitted background. |

**`persist` — keeping the answer.** The walk costs minutes, and its result used to evaporate with
the instance. `persist: true` writes what *actually launched* — context per slot, both KV cache
types and the slot count — into the model's saved settings, so the next plain load, including the
automatic one an inference request triggers, reproduces it without asking again. The tier is
written too, but **only when this call named a `priority`**: a `persist` that says nothing about
priority leaves the saved tier exactly as it was, rather than freezing whatever tier the model
happened to be running at into a standing setting nobody chose.

The trade is real and worth stating: it **freezes** the KV ladder and the slot estimator for that
model, which then stop adapting to what else is resident or to a changed machine. Hand the decision
back by nulling the fields — `PATCH /api/models/{id}/settings` with
`{"ctx_size": null, "kv_cache_type": null, "kv_cache_type_v": null, "parallel": null}`; a PATCH
sets the keys present, clears the ones sent as `null` and leaves absent keys untouched (D47), and
the settings dialog's Auto/blank fields do the same thing from the control panel. From a shell it
is `sfctl models settings <model> --set ctx_size=null` (repeat `--set` per field); a literal `null`
clears the field whatever it currently holds. Devices are
deliberately **not** persisted: a placement is a one-shot load argument, so the cards are still
chosen fresh on every load.

Two calls are refused outright, before anything loads: `persist` on a preset-only virtual model
(the write would land on the base model and every persona sharing it — call it on the base id), and
`persist` while that model is being benchmarked (the run rewrites those very fields). Afterwards
the write is best-effort: if a benchmark starts *during* the load, or the settings validation
rejects the resolved row, the load stands and only the write is skipped, with a warning in the
server log. So a success means the model is loaded; it does not by itself prove the profile was
saved — read the settings back if it matters.

**`persist` needs a credential when you are remote.** The route itself is ungated, because
residency is open (LM Studio parity) — but a flag that writes settings is a change to the box, so
it needs `server.api_key`, a caller on the rig itself, or the MCP pairing PIN, exactly like
`PUT /settings` does (D32). Otherwise it is a `403 remote_admin_requires_credential`, and the same
request without `persist` still loads. The MCP tool needs nothing extra: `sfctl mcp` reaches `/mcp`
with the key or the PIN it was registered with, which is the same D32 gate this flag asks for.

Use `list_models` -> a `placements[]` row -> `load_model` when you want to choose the hardware
yourself. Use `load_recommended` the rest of the time.

## How many conversations at once? — `recommended_parallel`

Every row carries two slot numbers. `max_parallel` is how many slots **fit**;
`recommended_parallel` is how many are **worth running**, and it is the one `load_args.parallel`
asks for. Match your client concurrency to the second.

`recommended_parallel_basis` says where it came from. `estimated` is a bandwidth calculation;
`measured` means somebody ran `benchmark_parallel(model_id)`, which loads the model once, sweeps 1 /
2 / 4 / 8 concurrent requests and records the curve. On this rig a 1.5B model whose estimate said 8
slots measured out at 2: aggregate throughput kept climbing to 8, but a single conversation had
collapsed to 27% of its solo speed by then. That is the trade the estimate could not see.

`benchmark_parallel` takes minutes and refuses outright on a busy server — the contention *is* the
measurement — so read `server_status.busy` first, and expect to run it once per model per set of
cards rather than routinely.

## "Which hardware should I use for this model?"

Every catalog entry answers that before you ask. `recommended` is the default load — the model's
optimal settings on this rig's best pair of GPUs, computed **as if those cards were free** — and
its `load_args` name the `devices`, so passing it to `load_model` places the model as well as
sizing it. `placements` repeats the answer for every other set of cards the box has (here
`dual_5090`, `dual_3090`, `all_gpus`, `single_5090`), each with a `ranking` of `fastest`,
`largest_context` and `cheapest`, so "run it on the 3090s and leave the 5090s free for something
else" is a row you pick rather than arithmetic you do. Because an optimal is computed on idle
cards, `fits_now` tells you whether it would load right now and `would_evict` says what is in the
way; in the compact `list_models` view the non-recommended modes carry settings and devices but no
`load_args`, so call `model_options` for the one you choose. The settings are chosen quality first:
the best KV cache that reaches the server's context floor, then the largest context at that
quality. A 4-bit K cache is never chosen for you.

## Loading on a server other agents are using

Two rules, both aimed at the same failure: an agent rearranging a box mid-conversation.

**A load never evicts a model that is serving a request.** If the only way to make room is to stop
somebody's stream, the load is refused with the busy model named, its in-flight request count, and
a `retry_after_s` — waiting is usually cheaper than retrying. `force=true` overrides it, and is the
only thing that does; a just-in-time load from an inference request can never set it.

**`test_model` is a smoke test, not a way to pre-warm a model** — use `load_model` for that. It is
one-at-a-time, refuses outright while anything is serving, loading or benchmarking, loads at the
server's default context with one slot when the model is cold, and unloads it again afterwards.
Read `server_status.busy` (`active_requests`, `loading`, `testing`) before either call and you will
avoid the busy refusals; a priority hold can still appear whenever a better-tier load starts —
honour `Retry-After`. `busy.priority_hold` names the model whose load is holding you off, and it
can become true between your read and your call, which is the case the header exists for. Every
loaded model also reports `loaded_by` and `priority`, so on a shared box you can see which client
asked for what, and at which tier.

## Sharing one model between bots

One `llama-server` serves several conversations at once, one per **slot**, so two bots on the same
model do not need two loads — they need enough slots. Either set `parallel` in the model's saved
settings, so every load of it (just-in-time ones included) comes up with them:

```bash
sfctl models settings pub/agent-model --set parallel=3
```

or resolve and remember the whole shape in one call:

```
load_recommended(model_id="pub/agent-model", ctx_size=32768, max_slots=3, persist=true)
```

The second is usually the better trade: the plan is sized for the concurrency you actually have,
the window is not paying for slots nobody will use, and `persist` means the next load repeats it.
Past the slot count llama.cpp queues rather than refusing, so guessing low costs latency, not
errors. `max_parallel` and `recommended_parallel` on the catalog row say what the placement will
really sustain — and a model whose settings carry `max_parallel_cap` now **refuses** a `parallel`
above it with a `400` naming both numbers rather than quietly clamping it (D48).

**Mixed tiers on one instance.** The saved `settings.priority` is the model's standing tier; each
request may then say what *it* is. Three bots sharing one model whose settings say `priority: 2`:

* the companion a person is talking to sends `priority: 1` on its turns — it passes every hold
  below tier 1, and a reload it triggers takes the fast placement and jumps the load queue;
* a beta bot sends nothing, so it inherits the model's tier 2 — while some *other* model is loading
  at tier 2 it still passes (equal tiers pass), but a tier-1 load holds it off;
* a nightly email-summarising bot sends `priority: 3` and waits (`503 priority_hold` +
  `Retry-After`) whenever a better-tier load is in flight — which is what you want from a job
  nobody is watching.

A `priority: 3` on a request makes that bot's own turns yield; it cannot pull the shared model down
to tier 3 if its turn is the one that happens to trigger a reload. The load takes the better of the
requested tier and the standing one, so the model comes back at tier 2 either way.

Traffic for the model that is *itself* loading always queues rather than being refused, so bots
sharing an instance never hold each other off during its own reload.

## When the VRAM is gone

A row that says `fits: false` but whose `if_gpus_idle.fits` is true is not a hardware limit — the
memory exists and something is holding it. `unload_model` on something else makes the row available.

When *nothing* of yours is loaded and the VRAM is still missing, `server_status` names the holder.
Every `llama-server` on the box is classified:

| Classification | Meaning | Reclaimed? |
| --- | --- | --- |
| `ours` | a child of this server | never |
| `child_of_live_process` | somebody else's live llama-server — another install, a test run, a hand-started one | **never**, however much it holds |
| `orphan` | our binary, out of our engines directory, parent gone | yes |

`vram_orphan_count` above zero is recoverable memory, and the watchdog's `reclaim_orphan_engines`
tool kills exactly those processes and nothing else. `null` for those counters means the process
table could not be read — which is not the same as zero.

This is D23, and it is worth one paragraph of why. On 2026-08-18 about 25 GiB across two cards was
unavailable with "everything stopped". The holders were three `llama-server.exe` children of a
`pytest` run started by a coding agent; StudioForge listed them as foreign processes holding zero
bytes, because NVML reports per-process VRAM as 0 for every process on Windows under WDDM. Nothing
in the product could say who launched them or whether killing them was safe. Four fixes came out of
it: children are now assigned to a Windows job object so the kernel kills them however their parent
dies, a startup sweep reclaims what earlier runs leaked, per-process bytes are read from the same
performance counter Task Manager's "Dedicated GPU memory" column uses, and every holder is reported
with its parent named. A live foreign holder is still never killed automatically — it belongs to
something that is still running, and taking its model away is not recovery.

---

## Engine updates

`studioforge engine --check` (or `GET /api/capabilities?check_update=true`) compares the **active**
build against GitHub and answers `current`, `latest`, `update_available`, `latest_variant` and a
`skipped` list saying why each rejected tag was rejected.

Two things it deliberately does not do. It never offers a tag that is not a `bNNNN` build: on
2026-08-18 llama.cpp published a prerelease tagged `v0.1.2` with **no assets at all**, and the
update check duly offered it — the install then failed with "no GPU-capable llama-server build for
win/x64 at v0.1.2". A `vX.Y.Z` tag ships no engine under any flag, so it is filtered out
unconditionally rather than hidden behind an option. And it compares **build numbers, not
strings**: `latest != current` calls a downgrade an update, and string ordering additionally breaks
across a digit boundary (`"b10000" < "b9999"`).

`--update` installs a tag already verified to have an asset this box can run, smoke-tests it, and
only then repins — so a broken release cannot become the default. Running instances keep the engine
they started with; reload a model to move it onto the new one.

---

## Zero-config inference: a default model

By default a request must name a model. If you would rather OpenClaw just talk to the server
without picking one, set a default:

```yaml
models:
  default_model: lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF/Qwen2.5-1.5B-Instruct-Q4_K_M
  preload_default_model: true
```

With that set:

* A request that **omits** `model` entirely is served by the default.
* A request naming `local-model`, `default`, `auto`, or `current` resolves to the default. (LM
  Studio clients send the literal string `local-model` as a fallback; 404-ing it would break them
  for no reason.)
* `preload_default_model: true` loads it at **startup**, so the first real request is a warm one
  rather than a multi-minute cold load.

Set it from the CLI without editing YAML:

```bash
sfctl config set models.default_model=<model-id> models.preload_default_model=true
```

## Keeping a model resident

Three ways, in increasing order of stickiness:

```bash
sfctl models pin <model>              # never idle-unloads (ttl 0)
sfctl config set models.default_ttl_s=3600   # idle timeout for everything
```

Or per request, LM Studio style — attach a `ttl` (seconds) to a chat completion and the idle timer
for that model resets to it:

```json
{"model": "...", "messages": [...], "ttl": 1800}
```

`ttl` is consumed by StudioForge and never forwarded to the engine. It can shorten or lengthen
the idle timer only: `0` is the wire form of *pinned*, and pinning is a box change, so a request
carrying `ttl: 0` (or a negative value) is served with no override rather than pinning the model
— use `pin_model` for that.

---

## What makes this reliable for an agent workload

These are behaviours OpenClaw can depend on, each chosen because the absence of it is a known way
local LLM servers appear broken.

**Cold loads do not look like hangs.** A streaming request against an unloaded model starts
returning immediately and emits an SSE keep-alive comment (`: loading <model> (15s)`) every few
seconds until the model is ready. Your HTTP read timeout never fires on a load that is progressing.
Comment lines are valid SSE that every compliant parser ignores.

**Bad requests fail *before* the stream starts.** Model resolution, image/vision checks, tool and
`response_format` validation all run against the registry — which needs no load — so a malformed
request gets a real `4xx` with a JSON body instead of an error frame buried inside a `200` stream.

**Concurrent requests queue, they do not error.** A burst against a cold model produces exactly one
load; every request is served once it is up. A model with in-flight requests is never evicted.

**Errors are OpenAI-shaped, always, with a stable `code`.** Never an HTML error page, never a `200`
hiding a failure, and never a `200` for a route that does not exist.

| Situation | HTTP | `error.code` | Retry? |
| --- | --- | --- | --- |
| Unknown model | 404 | `model_not_found` | no |
| Image sent to a text-only model | 400 | `model_not_multimodal` | no |
| Model too big for VRAM | 507 | `insufficient_vram` | no — read `suggestions` |
| Engine failed to start | 502 | `model_load_failed` | no — message carries the stderr tail |
| Busy / transient | 503 | varies | **yes**, honour `Retry-After` |
| Held by a higher-tier load | 503 | `priority_hold` | yes — honour `Retry-After` |

A `507` carries the numbers *and* what to do about it under `error.studioforge.suggestions` —
fewer slots at the same window (`max_parallel_that_fits`, when an explicit `parallel` was the
problem), a smaller context that would fit, a cheaper KV cache type, or a smaller quant.

**Loads are one at a time.** Two agents asking for two cold models at once do not race for the
same VRAM: the second load plans after the first has actually allocated, and either fits beside it
or is refused with the numbers (DECISIONS.md D29). A streaming request that is queued behind another
load sees the same keep-alive comments as its own load would.

**Thinking models never return an empty reply.** llama.cpp's default routes a reasoning model's
output into `reasoning_content` and leaves `content` empty; measured on DeepSeek-R1, content length
**0**. StudioForge defaults to `--reasoning-format none` so the text stays in `content`, and as a
safety net will merge reasoning into an otherwise-empty `content`.

**Both sampler spellings work.** `repeat_penalty` and `repetition_penalty` are accepted; LM Studio
silently ignores the latter, which makes a sampler look broken with no error.

**Model ids round-trip.** Any `id` from `/v1/models` is accepted verbatim as `model`. Short aliases
work too (bare filename, `publisher/name`), case-insensitively.

**`/v1/models` tells you what is loaded.** Each entry carries `state` (`loaded` / `not-loaded`) and,
when resident, `loaded_context_length`. LM Studio's `/api/v0/models` is also implemented for clients
that expect it.

---

## Troubleshooting

**Empty replies from a thinking model.** Check `reasoning_format` for that model is `none`
(the default). If you set it to `deepseek`, thoughts move to `reasoning_content`, which is not in
the OpenAI schema and which most clients ignore.

**Avoid `]]` (and similar) as a stop sequence with reasoning models.** With thoughts inline, a
reasoning model that *narrates* a bracketed format will trip the stop sequence and end generation
early. This is a real, previously-observed failure — verified on gemma4-31b returning `""`.

**A model will not load.** `sfctl models plan <model>` shows the fit verdict, the per-GPU
projection, and concrete suggestions, without loading anything. Over MCP, the same answer is the
row's `fits` / `if_gpus_idle` pair.

**VRAM is used but nothing of yours is loaded.** See
[When the VRAM is gone](#when-the-vram-is-gone) — `server_status` names every holder, and
`reclaim_orphan_engines` recovers the leaked ones.

**The server is answering but nothing takes effect.** Check `GET /health`: `instance` reads
`secondary` when another process already owns the data directory. A secondary serves reads but
starts no background work at all — no download resume, no TTL sweeper, no auto-load — and reports
the holder's pid in `instance_holder_pid`. That is one process per data directory being enforced
(D24), not a bug.

**The server stopped responding.** The watchdog runs as a separate process for exactly this case:

```bash
sfctl recover                 # diagnose: up / degraded / wedged / down
sfctl recover --gpus          # GPU state, read THROUGH the watchdog
sfctl recover --logs 200      # tail the server log through the watchdog
sfctl recover --config        # the server config, secrets redacted
sfctl recover --kill <model>  # free VRAM without a full restart
sfctl recover --restart       # restart the wedged server
```

The first four read; they change nothing. They go through the watchdog on purpose: the
same diagnostics on the main server are served by the process that is not answering, so
looking before choosing which hammer to reach for needs the sidecar too.

**Something loaded but behaves oddly.** `sfctl models info <model>` shows what the engine *actually*
reports (real context, slot count, whether speculative decoding is armed) next to what was
requested — the two differing is usually the answer.
