# HTTP API compatibility

StudioForge listens on LM Studio's port (`1234` by default) and answers the same HTTP surfaces LM Studio
does, so tools written for LM Studio, the OpenAI API or the Anthropic API work by changing a base URL.
This page is the contract: which endpoints exist, which are complete, which are partial, and which say
"not supported" out loud instead of pretending.

One server, one port, one credential per client (a scoped key, if you set up access control):

| Base URL | Family |
| --- | --- |
| `http://HOST:1234/v1` | OpenAI-compatible (`openai` SDK `base_url`) |
| `http://HOST:1234` | Anthropic-compatible (`anthropic` SDK `base_url`; the SDK appends `/v1/messages`) |
| `http://HOST:1234/api/v1` | LM Studio native v1 REST |
| `http://HOST:1234/api/v0` | LM Studio legacy v0 mirror (deprecated upstream, kept) |
| `http://HOST:1234/api` | StudioForge's own management API (not an LM Studio surface) |

## Authentication

On a loopback-only install with no keys and no owner account the server is open, like LM Studio out of
the box (changing the *box* still needs a caller on the machine). Once you create a scoped key
(`studioforge keys create`, or Advanced > Accounts & keys in the panel) or an owner account, and always
when you bind the network, every route except the liveness probes takes a key in **either** header, on
every surface:

```
Authorization: Bearer <key>
x-api-key: <key>
```

Keys have a scope (`read`, `inference`, `admin`) and may carry an expiry and a rate limit; a route above a
key's scope answers `403 permission_denied`. The old single `server.api_key` still works as an `admin`
credential and the MCP pairing PIN as an MCP-only one, but both are deprecated. The scope table and the
rules are in [`SECURITY.md`](SECURITY.md).

The `anthropic` SDK sends `x-api-key`, the `openai` SDK sends `Authorization: Bearer`; both work against
every endpoint below.

## Compatibility matrix

Legend: **yes** = implemented as documented upstream; **partial** = implemented with the limits in the
notes column; **501** = answered with an explicit `not_implemented` error; **no** = a JSON 404.

### LM Studio native v1 (`/api/v1`)

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /api/v1/models` | yes | LM Studio's `models[]` entries: `key`, `type`, `publisher`, `display_name`, `architecture`, `quantization{name,bits_per_weight}`, `size_bytes`, `params_string`, `loaded_instances[]`, `max_context_length`, `format`, `capabilities{vision,trained_for_tool_use}`. Additive: `arch`, `state` (`loaded`/`loading`/`not-loaded`), `loaded_context_length`, `loaded_instances[].context_length`, and `object`/`data` mirroring the OpenAI list shape. `bits_per_weight` is `null` (not measured). |
| `POST /api/v1/models/load` | yes | Same manager call as `POST /api/models/{id}/load`: the planner decides placement, and a model that cannot fit is its honest `507 insufficient_vram` with the numbers. `context_length` is the per-conversation window (`ctx_size`). `eval_batch_size`, `flash_attention` and `num_experts` are accepted and reported under `ignored` (the planner owns them). `offload_kv_cache_to_gpu: false` is a 400: the server is GPU-only. Extras: `ttl`, `priority`, `parallel`. |
| `POST /api/v1/models/unload` | yes | `{instance_id}` (or `{model}`); 404 `instance_not_found` when nothing is loaded under that name. |
| `POST /api/v1/models/download` | partial | `model` is a Hugging Face `owner/repo` id or link (`quantization` or `hf.co/owner/repo:QUANT` picks the quant); LM Studio's own catalog ids are not available. Needs a caller on the server's machine or an `admin` key (it writes to disk), exactly like `POST /api/downloads`. |
| `GET /api/v1/models/download/status/{job_id}` | yes | `downloading`/`paused`/`completed`/`failed`. `job_id` is a StudioForge download group. |
| `GET /api/v1/models/download/status` | extension | The same shape for every job (`{"jobs": [...]}`). |
| `POST /api/v1/chat` | **501** | LM Studio's chat is stateful and MCP-aware; this server is stateless. Use `/v1/chat/completions`, `/v1/messages` or `/v1/responses`. |
| any other `/api/v1/*` | no | JSON 404 in the standard error envelope. |

A StudioForge model has at most one running instance, so an `instance_id` is the serving model's id. A
persona (a preset-only virtual model) is served by its base model and reports the base's id.

### OpenAI-compatible (`/v1`)

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /v1/models`, `GET /v1/models/{id}` | yes | Lists every downloaded model. Additive `state`, `loaded_context_length` and a `studioforge` block. |
| `POST /v1/chat/completions` | yes | Streaming, tools and `tool_choice`, `response_format` (`text`, `json_object`, `json_schema`), vision (`image_url`) for models with an mmproj. Additive `ttl`, `priority`. |
| `POST /v1/completions` | yes | Same JIT loading, `ttl` and `priority`. |
| `POST /v1/embeddings` | yes | Embedding models only. |
| `POST /v1/rerank` | yes | Rerank models only. |
| `POST /v1/responses` | partial | Stateless subset, see below. |
| `POST /v1/tokenize` | extension | llama-server's tokenizer. |
| `/v1/audio/*`, `/v1/images/*`, `/v1/files`, `/v1/batches`, Assistants | no | Out of scope. |

### llama-server router and native routes (D90)

A client written for stock `llama-server` (build b12110, `tools/server/README.md`) works unchanged; the
extras are StudioForge's. The router vocabulary is canonical for model state and load/unload; the older
`/api/models/{id}/load|load-recommended|unload|plan|plan-recommended` routes keep working and answer with
`Deprecation: @1794182400` and `Link: <docs/MIGRATION.md>; rel="deprecation"` (no `Sunset` yet).

| Endpoint | Status | Notes |
| --- | --- | --- |
| `GET /models` | yes | `data[]` of `{id, path, status{value, args?, failed?, exit_code?}, architecture{input_modalities, output_modalities}, sidecars?, studioforge{...}}`. `status.value` is `unloaded`, `loading` or `loaded` (`failed: true` on an unloaded model whose last launch died). `?reload=1` rescans the library first (admin, like `POST /api/models/scan`). |
| `POST /models/load` | yes | `{"model"}` as in llama-server, plus `context_length` (tokens per slot), `devices`, `allowed_devices`, `parallel`, `min_slots`, `max_slots`, `kv_cache_type`, `kv_cache_type_v`, `kv_unified`, `priority`, `persist`, `lease_id`. `context_length` alone is the strict walk of `load-recommended`; `devices` / `parallel` / `kv_cache_type*` / `force` is the explicit load; a bare model is a plain load. Reply: `{"success": true, ...instance}`. Unknown keys are ignored. |
| `POST /models/load?dryRun=true` | extension | The same decision, nothing started (the same routing rule as the load, `lease_id` included): the plan body with `dry_run: true` and `dry_run_of` (`load` or `load_recommended`). |
| `POST /models/unload` | yes | `{"model"}` -> `{"success": true, "unloaded": bool}`. |
| `POST /tokenize`, `/detokenize`, `/apply-template`, `/completion`, `/infill`, `/embedding`, `/reranking` (`/rerank`) | yes | The child's answer, status and body byte for byte (`stream: true` relays the child's SSE untouched, no `[DONE]` added). Needs an `inference` key. |
| `GET /props`, `GET /slots` | yes | The child's JSON; `/props` gains one key, `studioforge` (instance id, devices, split, parallel, `ctx_per_slot`, KV types, batch sizes, engine tag, `lease_id` when leased). `POST /props` is a 405: per-model settings are the source of truth. |
| `GET /v2/health/live`, `GET /v2/health/ready`, `GET /v2/models/{id}/ready` | yes | KServe v2 probes. The two server probes need no key; ready is 503 while booting or draining. A model is ready (200) only when resident and serving, else 503 with its state. |

**Which model.** A POST names it in the JSON body (`model`), a GET in `?model=`; `X-SF-Model` works on both
(body, then query, then header). With no name: the one resident model that can serve the route, else
`models.default_model`, else `400 model_required` listing the loaded ids. A named model that is not
resident is loaded just in time on a POST; a GET never loads unless it says `?autoload=true`.
`?autoload=false` on any inference POST (native, `/v1/chat/completions`, `/v1/completions`,
`/v1/embeddings`, `/v1/rerank`, `/v1/tokenize`) refuses the load with `404 model_not_loaded`.

**Identity headers** (`server.identity_headers`, default on) on every response: `Server: StudioForge/<ver>`,
`X-Request-Id` (the client's own, when it sent a token of 1-128 safe characters; also bound into the log
line), and on responses a model produced `X-SF-Model` (the instance that served it) and `X-SF-Engine`
(the llama.cpp build). `GET /health` adds `product: "StudioForge"`; `GET /api/capabilities?brief=true`
(`product`, `version`, `features`, `decisions`, `feature_decisions`) needs no key
(`server.public_capabilities_brief`, default on).

`GET /v1/models` entries gain llama-server's `status{value, args}`, `architecture{...}` and `meta{n_vocab,
n_ctx_train, n_embd, n_params, size}`, plus `studioforge.loaded_instances[]` and
`studioforge.max_context_length`; `?loaded=true` keeps only resident models. `state` (`not-loaded`
spelling) stays as before.

### Anthropic-compatible

| Endpoint | Status | Notes |
| --- | --- | --- |
| `POST /v1/messages` | partial | See below. Streaming, custom tools, images, `tool_use`/`tool_result` round trip. |
| `POST /v1/messages/count_tokens` | no | Use `POST /v1/tokenize`. |
| Message batches, Files, server-side tools (web search, bash, ...) | no | Refused with a 400 naming the parameter. |

### LM Studio v0 (`/api/v0`)

`GET /api/v0/models[/{id}]`, `POST /api/v0/chat/completions`, `/completions`, `/embeddings`: **yes**, kept
as a mirror. Upstream calls v0 deprecated; new clients should use `/api/v1` and `/v1`.

## Per-request extensions

Honoured **identically** on `/v1/chat/completions`, `/v1/completions`, `/v1/messages`, `/v1/responses`
(top-level body fields) and, for `ttl` and `priority`, on `POST /api/v1/models/load`. They are consumed
by the gateway and never forwarded to the engine.

| Field / header | Meaning |
| --- | --- |
| `ttl` (seconds) | Idle time after which a model this request loaded may unload. A request may **shorten** the idle timer or lengthen it up to its tier's ceiling (`models.ttl_by_priority`); it can never pin a model (`0` and negatives are ignored) and never unpin one the owner pinned. LM Studio defaults to 60 minutes; StudioForge uses its own configured defaults. |
| `priority` (`1`, `2`, `3`) | The load tier: 1 the model a person is chatting with, 2 a dispatched agent, 3 (default) background. Anything else is a 400 naming `priority`. |
| `X-SF-Client: <label>` | Names the caller in logs and `GET /api/status` (otherwise its IP address). |

Just-in-time loading is the default everywhere: naming an unloaded model in any inference request loads
it, and concurrent requests queue behind the one load. Streams carry keep-alive frames while it loads.

## `POST /v1/messages` (Anthropic Messages)

The request is translated to a chat-completions body and run through the same path as
`/v1/chat/completions` (model resolution, vision handling, tool validation, tiering, `ttl`), so the two
surfaces behave the same.

* Supported: `model`, `max_tokens` (required), `system` (string or text blocks), `messages` with `text`,
  `image` (base64 or URL), `tool_use` and `tool_result` blocks, `tools` + `tool_choice`
  (`auto`, `any`, `none`, `tool`; `disable_parallel_tool_use`), `temperature`, `top_p`, `top_k`,
  `stop_sequences`, `stream`.
* Response: `content[]` blocks (`text`, `tool_use`, and `thinking` only when the request enabled
  `thinking` and the engine separates reasoning), `stop_reason` (`end_turn`, `max_tokens`, `tool_use`,
  `stop_sequence`), `usage.input_tokens` / `output_tokens` from the engine's own counts.
* `stop_sequence` is reported only when the engine says which word stopped generation; otherwise a
  stop-sequence stop reads `end_turn`.
* Streaming events: `message_start`, `content_block_start`, `content_block_delta` (`text_delta`,
  `input_json_delta`, `thinking_delta`), `content_block_stop`, `message_delta`, `message_stop`, plus
  `ping` while a model loads and `error` if the engine fails mid-stream.
* Not supported (400, named parameter): document and search-result blocks, server tools, `thinking`
  blocks replayed from history are dropped.
* Errors use Anthropic's envelope, `{"type":"error","error":{"type":"invalid_request_error", "message":
  "...", "code":"..."}}`, with StudioForge's stable `code` added. Authentication failures come from the
  shared middleware and keep the OpenAI envelope; both official SDKs read `error.message` from either.
* Anthropic recommends a large context for agentic use; load the model with `context_length` 32768 or
  more (`POST /api/v1/models/load`).

## `POST /v1/responses` (OpenAI Responses, stateless subset)

* Supported: `model`, `input` (a string, or items: `message`, `function_call`, `function_call_output`;
  parts `input_text`, `output_text`, `input_image`), `instructions`, `tools` (type `function`),
  `tool_choice`, `parallel_tool_calls`, `temperature`, `top_p`, `max_output_tokens`, `text.format`
  (`text`, `json_object`, `json_schema`), `stream`.
* Response: a `response` object with `output[]` (`message` with `output_text` content, `function_call`
  items) and `usage`; `status` is `incomplete` with `incomplete_details.reason = max_output_tokens`
  when the budget ran out.
* Streaming events: `response.created`, `response.in_progress`, `response.output_item.added`,
  `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`,
  `response.function_call_arguments.delta` / `.done`, `response.content_part.done`,
  `response.output_item.done`, `response.completed` (`response.incomplete` when truncated), `error`.
* Not supported (400 `unsupported_parameter`): `previous_response_id`, `item_reference` and
  `background` (they need a response store; send the whole conversation in `input`), built-in tools
  (web search, file search, MCP, computer use, code interpreter), `input_file`. `store`, `metadata`,
  `user`, `include`, `reasoning` and `truncation` are accepted and ignored; `store` is echoed `false`.
  Reasoning items are not emitted.

## The OpenClaw-style flow

OpenClaw preloads through the native load endpoint, then chats. The same three calls work for any
LM Studio client:

```bash
# 1. What is downloaded, and what is loaded? (`key` is the id to use everywhere else.)
curl -s http://HOST:1234/api/v1/models -H "Authorization: Bearer $KEY" | jq '.models[] | {key, state, loaded_instances}'

# 2. Preload a model with a 32k window, and let it idle out after ten minutes.
curl -s http://HOST:1234/api/v1/models/load -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "vendor/model-Q4_K_M", "context_length": 32768, "ttl": 600}'
# -> {"type":"llm","instance_id":"vendor/model-Q4_K_M","context_length":32768,"status":"loaded",...}

# 3. Chat with that key, on whichever surface the client speaks.
curl -s http://HOST:1234/v1/chat/completions -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "vendor/model-Q4_K_M", "messages": [{"role": "user", "content": "hello"}]}'
```

`POST /api/v1/models/unload` with `{"instance_id": "vendor/model-Q4_K_M"}` releases it. If the load
does not fit, the answer is `507` with the planner's numbers (`error.studioforge`), never a silent CPU
spill; see [`OPENCLAW.md`](OPENCLAW.md) for how an agent should react.

## Errors

Every route answers errors with `{"error": {"message", "type", "code", "param"}}` (an optional
`studioforge` block carries diagnostics; `/v1/messages` uses Anthropic's envelope with the same `code`).
Unknown routes, including anything unrecognised under `/api/v1`, are JSON 404s in that envelope.

Management routes (`/api/*` except the LM Studio trees, and `/models/*`) answer RFC 9457
`application/problem+json` to a caller that sends `Accept: application/problem+json`: `type`
(`urn:studioforge:error:<code>`), `title`, `status`, `detail`, `instance` (`urn:studioforge:request:<id>`,
the `X-Request-Id`), `code`, `fix`, `retry_after_s` and `studioforge`, with the legacy `error{}` object
embedded whole. Without that header the body is the envelope above, unchanged, plus a `Deprecation` marker;
the default flips only once every caller has migrated. Inference routes never negotiate.

Codes introduced with the compatibility surface:

| `code` | HTTP | Where | Meaning |
| --- | --- | --- | --- |
| `not_implemented` | 501 | `POST /api/v1/chat` | This server does not implement the endpoint; `error.studioforge.alternatives` lists what to use. |
| `unsupported_parameter` | 400 | `/v1/messages`, `/v1/responses`, `/api/v1/models/*` | A field or content type that would change the answer is not supported; `param` names it. |
| `instance_not_found` | 404 | `POST /api/v1/models/unload` | The model exists but has no loaded instance. |
| `download_not_found` | 404 | `GET /api/v1/models/download/status/{job_id}` | No such download job. |
| `model_required` | 400 | native routes, `POST /models/load` and `/models/unload` | No model was named and the choice is not unique; `error.studioforge.resident` lists the loaded ids. |
| `model_not_loaded` | 404 | any inference POST with `?autoload=false`; native `GET`s | The model is not resident and this request said not to load it. |
| `tool_call_parse_failed` | 502 | `/v1/*` | The engine could not parse the model's tool call; its own message is kept verbatim. |
| `context_exceeds_training` | 400 | `load-recommended`, `POST /models/load` | The context asked for is past the trained window; `error.studioforge.n_ctx_train` has the number. |

The existing codes (`model_not_found`, `insufficient_vram`, `model_busy`, `context_exceeded`, ...) apply
unchanged.

## Verifying compatibility

`tests/unit/compat/` pins the wire formats offline, including the official `openai` and `anthropic`
SDKs against an in-process app. `tests/compat/` runs the same flows against a real test instance on an
alternate port (never the live one):

```bash
COMPAT_BASE_URL=http://127.0.0.1:11234 COMPAT_MODEL=<chat model id> \
  uv run pytest tests/compat -m compat -q -p no:cacheprovider
```

The example payloads in `tests/fixtures/lmstudio/` are derived from LM Studio's documentation, not
captured from a running LM Studio; the README beside them says which. Set `COMPAT_LMSTUDIO_URL` to diff
the model-list shape against a real LM Studio 0.4 or later.
