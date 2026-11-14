# Running several copies of one model

One model id used to mean one `llama-server` child. Since D91 a model can run as several
**instances** at once -- each on its own cards, with its own context, slots and KV cache -- and two
clients that both want the same model each get their own instead of fighting over one. This page is
the contract: how to ask for a copy, how to address it, how it ends, and what every refusal means.

Instances live under a GPU lease ([TENANTS.md](TENANTS.md)): the lease is the resource boundary, and a
copy lives and dies with the lease that created it. Plain mode (no leases) is unchanged: one id, one
instance, loaded on demand.

## The ideas in one screen

* **Ids follow LM Studio's instance identifiers.** The base model is its plain id `X`. A copy is
  `X:2`, `X:3`, ... Numbers are **never reused while the server runs**: with `X:2` released, the
  next copy is `X:4` (or higher), so a late request for `X:2` is always `404 instance_not_found` and
  never lands on someone else's copy. The counter stays above every id restored at start. Ask for a
  name instead with a `tag`: `X@<tag>` (the virtual-copy convention) -- a tag *is* reusable once its
  lease ended, when a new lease asks for it explicitly (reproducible names across runs). An id that
  is already a model, an alias, a persistent copy (such as a judge's `X@b`) or another lease's
  instance is refused with `409 instance_id_conflict`; two grants racing never share an id.
* **The id goes in `model`.** Every OpenAI-shape route (`/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/rerank`, `/v1/tokenize`, `/v1/messages`, `/v1/responses`) and every native
  route (`/completion`, `/tokenize`, `/props?model=`, ...) accepts it. `X-SF-Model` on the response
  says which instance answered.
* **Parity is the server's guarantee.** A copy is a deep copy of the base's settings -- file, chat
  template, sampling, adapters, engine build, reasoning format, draft, flags. Its `launch` may change
  only `ctx_size` (tokens per slot; `context_length` is accepted), `parallel` *or*
  `min_slots`/`max_slots`, `kv_cache_type` and `kv_cache_type_v`. Its placement is its cards.
* **A second holder never replaces the first.** If you lease (or bind a load to) a model another
  holder already holds, you get your own copy on your cards; theirs is never moved, reloaded or
  unloaded.
* **Holder identity, not key equality.** Several clients often share one API key. Who you are, for
  "which instance do I mean", is: the lease you name (`X-SF-Lease: <lease id>`, or a body
  `lease_id`), else your `X-SF-Client` label matched exactly to a lease's holder, else the key that
  took the lease -- but the key counts only when you send no label, the holder's own name, the label
  you took the lease with (it is recorded: lease as `openclaw` while labelling `openclaw-main` and
  that label stays yours), or exactly the family root (`crucibleforge` for `crucibleforge-judge`).
  An exact match wins: with `crucibleforge` and `crucibleforge-judge` both standing, the label
  `crucibleforge` means the first only. Send any other label and you are someone
  else, whatever key you share: you cannot bind a load to that lease, release or renew it, or name
  it in `X-SF-Lease` (`403 permission_denied`). **Send `X-SF-Client` from every run** -- a request is
  yours only when it carried your label; the peer address (the same `127.0.0.1` for every tenant on
  one box) never makes another client's stream yours.

## 1. Ask for copies: `POST /api/leases` with `instances`

```http
POST /api/leases
Authorization: Bearer <inference key>
X-SF-Client: bench-run-7
Content-Type: application/json

{
  "devices": [0, 1, 2, 3],
  "holder": "bench-run-7",
  "idle_ttl_s": 1800,
  "instances": [
    {"model": "vendor/Model-27B-GGUF/Model-27B-Q8_0", "gpus": [0, 1],
     "launch": {"ctx_size": 32768, "parallel": 6}},
    {"model": "vendor/Model-27B-GGUF/Model-27B-Q8_0", "gpus": [2, 3], "tag": "judge",
     "launch": {"ctx_size": 16384, "min_slots": 4, "max_slots": 6, "kv_cache_type": "q8_0"}}
  ]
}
```

One entry is `{"model", "count"?, "gpus"?, "tag"?, "launch"?, "load"?}` (Triton's `instance_group`
shape; `model_id` is accepted for `model`):

| Field | Default | Meaning |
| --- | --- | --- |
| `model` | required | The model to copy (an id or alias; never an instance id). |
| `count` | `1` | How many copies of this entry, all on `gpus` (1-16). With a `tag` they are `X@<tag>1`, `X@<tag>2`, ... |
| `gpus` | the lease's `devices` | The cards each copy of this entry is split across; must be lease cards. |
| `tag` | none | Name the copy `X@<tag>` instead of `X:<n>` (1-48 of `A-Z a-z 0-9 . _ -`). |
| `launch` | `{}` | `ctx_size`, `parallel`, `min_slots`, `max_slots`, `kv_cache_type`, `kv_cache_type_v`. Anything else: `400 launch_key_not_allowed`. `min_slots`/`max_slots` without `parallel` need a window (`ctx_size`, or the base's saved one): the strict load-recommended walk then sizes the slots on the copy's cards. |
| `load` | `true` | Load before the reply. `false` reserves the copy; its first request (or a load naming it) loads it. |

**All or nothing.** The grant loads every `load: true` copy before it answers. If one cannot be loaded,
the copies already up are unloaded, every record is dropped, the lease is released, and the call
answers **one** error -- the failing load's status and `code` (for example `507 insufficient_vram`) --
with:

```json
{"error": {"code": "insufficient_vram", "message": "lease not granted -- instance X:3 could not be loaded: ...",
  "studioforge": {"lease_rolled_back": true, "lease_id": "ab12cd34ef56",
    "instances": [
      {"id": "X:2", "model": "X:2", "base_model_id": "X", "devices": [0, 1], "state": "rolled_back"},
      {"id": "X:3", "model": "X:3", "base_model_id": "X", "devices": [2, 3], "state": "failed",
       "code": "insufficient_vram", "status": 507, "reason": "..."}],
    "evicted": []}}}
```

`state` per instance: `loaded`, `rolled_back`, `failed`, `not_attempted`, `not_loaded` (`load: false`).
Idle residents the grant evicted are not brought back.

**The reply** is the lease record (as before) plus `instances[]`, one row per copy, as actually loaded:

```json
{
  "id": "ab12cd34ef56", "devices": [0, 1, 2, 3], "holder": "bench-run-7",
  "model_ids": ["vendor/Model-27B-GGUF/Model-27B-Q8_0:2", "vendor/Model-27B-GGUF/Model-27B-Q8_0@judge"],
  "instances": [
    {"id": "vendor/Model-27B-GGUF/Model-27B-Q8_0:2",
     "model": "vendor/Model-27B-GGUF/Model-27B-Q8_0:2",
     "base_model_id": "vendor/Model-27B-GGUF/Model-27B-Q8_0",
     "lease_id": "ab12cd34ef56", "holder": "bench-run-7",
     "devices": [0, 1], "ctx_per_slot": 32768, "context_length": 32768, "parallel": 6,
     "kv_cache_type": "f16", "kv_cache_type_v": "f16", "state": "loaded", "engine_tag": "b12110",
     "tag": null, "origin": "requested", "launch": {"ctx_size": 32768, "parallel": 6},
     "created_at": 1794612741.4, "ephemeral": true},
    {"id": "vendor/Model-27B-GGUF/Model-27B-Q8_0@judge", "...": "..."}
  ],
  "lease": {"...": "the same record, canonical place"},
  "evicted": [], "adopted": []
}
```

`model` is the exact string to send in chat requests. `state` uses llama-server's router words:
`loaded`, `loading`, `unloaded`.

The same field on MCP: `reserve_gpus(devices=[2, 3], instances=[{"model": "<id>"}])`; on the CLI:
`sfctl leases add --devices 2,3 --instance <id>` (or `--instance '{"model": "<id>", "tag": "b",
"launch": {"ctx_size": 32768}}'`, repeatable).

### The old shape still works -- and never takes another holder's model

`{"devices": [0, 1], "model_ids": ["X"]}` behaves as before (the lease names `X`; its next load lands
on these cards) **unless another holder holds `X`**: another standing lease names it, or it is serving
another client's requests on other cards. Then the lease gets its own copy instead: `model_ids` is
`["X:2"]` and `instances[]` lists it with `origin: "second_holder"` and `state: "unloaded"`; it loads
on first use. Give it a shape with `instances[].launch` if the base's saved settings are not what you
want.

A lease-bound load (`POST /api/models/X/load` or `/load-recommended` or `/models/load` with
`lease_id`) of a model another holder holds does the same: the lease gets `X:<n>` on its cards, loaded
with the call's `ctx_size` / `parallel` / KV types, and the reply is that instance. A bound load of a
model the lease already has a copy of loads that copy.

## 2. Address a copy

```http
POST /v1/chat/completions
Authorization: Bearer <inference key>
X-SF-Client: bench-run-7

{"model": "vendor/Model-27B-GGUF/Model-27B-Q8_0:2", "messages": [{"role": "user", "content": "hi"}]}
```

* **An exact instance id routes exactly**, whoever sends it.
* **A base id routes to the caller's own instance**: the lease named by `X-SF-Lease` (or the body's
  `lease_id`), else the lease whose holder equals the `X-SF-Client` label, else the lease the caller's
  key took. With several copies of its own, the least busy one answers. A caller that holds no copy --
  and one whose identity matches two different holders' leases (a shared key and no label) -- gets the
  base resident exactly as before, or a just-in-time load of it.
* `X-SF-Model` on the response names the instance that answered.

## 3. List copies

* `GET /api/leases`, `GET /api/status` (`leases[].instances[]`), MCP `gpu_leases` / `server_status`:
  each lease row carries `instances[]` (the reply rows above, with live `state`).
* `GET /api/status` `loaded[]`: every running child, with `lease_id`, `holder`, `ephemeral` and
  `base_model_id`.
* `GET /v1/models`: the base entry's `studioforge.loaded_instances[]` lists **every loaded instance**
  of the model -- its own and each copy -- as `{id, config: {context_length, parallel},
  context_length, devices, parallel, lease_id, holder, ephemeral}` (LM Studio's shape plus where and
  whose). Each copy is also its own entry with `studioforge.ephemeral: true`, `lease_id`, `holder`,
  `devices`. The base entry's `state` / `status.value` still describe the base resident only.
* `GET /models` (router) lists copies too; `GET /api/models` only with `?include_ephemeral=true`.
* `sfctl status` (a "Lease instances" table), `sfctl leases list` (an Instances column), and the
  panel's lease line.

## 4. Load and unload

* `POST /api/models/{id}/load` (or `/models/load`) with an instance id and no shape loads it with its
  stored shape (a loaded one is returned as it is); with `ctx_size` / `parallel` / KV types it loads in
  that shape -- `force: true` to reshape a loaded one -- on its own cards only (`devices` may name a
  subset). Reshaping another holder's copy is admin-only (`409 instance_held`).
* An unload of a **base id** by a holder takes down **its own** copy; the reply says which:

  ```http
  POST /api/models/vendor/Model-27B-GGUF/Model-27B-Q8_0/unload
  X-SF-Client: bench-run-7
  ```
  ```json
  {"model_id": "vendor/Model-27B-GGUF/Model-27B-Q8_0", "unloaded": true,
   "unloaded_id": "vendor/Model-27B-GGUF/Model-27B-Q8_0:2"}
  ```

  With two copies of its own, a base-id unload is `409 instance_ambiguous`: name the exact id.
* **Another holder's instance** is touched only on its exact id, by an admin, with `{"force": true}`
  in the body (or `?force=true`): `409 instance_held` otherwise. Loading it with a shape (or
  `force`, or through load-recommended) is admin-only the same way, and an admin's reshape never
  rewrites the shape the holder's restart restores -- only the holder's own loads do.
* **Never another client's streams without saying so.** An unload of a model a lease holds that is
  serving requests from other clients is `409 instance_in_use`, naming them in
  `error.studioforge.in_flight_clients`, unless the call says `force: true`. A lease waiver (you proved
  you hold the lease) is not a `force`.
* `POST /models/unload {"model", "force"?}` and `POST /api/v1/models/unload {"instance_id", "force"?}`
  follow the same rules; LM Studio's unload of a base id finds your own loaded copy even when the
  base itself is not loaded.
* MCP `load_model` / `load_recommended` carry your key, `lease_id` and admin-ness but no label, so
  there the key never says whose copy you mean: a non-admin forced or shaped load of a copy must
  pass `lease_id` naming the copy's own lease (`instance_held` otherwise).
* A forced reload never cuts requests someone else labelled: with another client's labelled
  requests in flight it is `409 instance_in_use`, even from the copy's own lease (admins excepted).

## 5. Release, expiry, restart

```http
DELETE /api/leases/ab12cd34ef56
```
```json
{"released": {"...": "the lease"}, "unloaded": ["X:2", "X@judge"], "kept": [],
 "instances_ended": ["X:2", "X@judge"]}
```

* A lease's copies end with it -- release, `?unload=false` too, idle expiry, any end. The ids become
  unreachable at once, the children are stopped (busy or not), the records dropped, then the lease.
* A request **in flight** on a copy when it ends gets `410 instance_released` (non-streamed) or, mid
  stream, an SSE error frame `data: {"error": {"code": "instance_released", ...}}` followed by
  `data: [DONE]`.
* A **later** request naming the id gets `404 instance_not_found` (with `lease_id`, `reason` and
  `released_at` in `error.studioforge`) -- never a reload on the default context ladder, and never a
  new holder's copy (numbers are not reused while the server runs). Any `X:<n>` nobody created is
  the same 404. A load that was waiting when the lease ended stops what it started and answers
  `410 instance_released`.
* **Restart.** Copies are stored with their lease (`gpu_leases.instances_json`, D61's mirror). After
  a restart the lease comes back, each copy is rebuilt **unloaded**, and its first request loads it
  with exactly the shape it last ran with -- the same cards, context per slot, slots and KV types.

## 6. Errors

| HTTP | `code` | Where | Meaning | Do |
| --- | --- | --- | --- | --- |
| 400 | `launch_key_not_allowed` | `POST /api/leases`, `reserve_gpus` | a `launch` key outside the six | drop it; change the base's settings instead |
| 400 | `invalid_request_error` (param named) | `POST /api/leases` | a bad entry: unknown field, `gpus` off the lease, `count` out of range, a bad `tag`, `min_slots` without a window | fix the field named in `param` |
| 404 | `instance_not_found` | every route taking `model` | the instance ended with its lease, or never existed | take a new lease |
| 409 | `instance_id_conflict` | `POST /api/leases` | the `tag`'s id is taken | another tag, or none |
| 403 | `permission_denied` | bind a load, release, renew, `X-SF-Lease` | your `X-SF-Client` says you are not that lease's holder (a shared key does not change that) | act on your own lease |
| 409 | `instance_held` | load, unload, grant | the instance belongs to another holder (a shaped or forced load of it included) | lease your own copy; admin + `force` on the exact id |
| 409 | `instance_in_use` | unload | it serves other clients' requests | wait (`Retry-After`), unload your own copy, or `force: true` |
| 409 | `instance_ambiguous` | unload | the base id names several of your copies | name the exact id |
| 410 | `instance_released` | an in-flight request | its lease ended under it | take a new lease |
| any | the failing load's code | `POST /api/leases` with `instances` | all-or-nothing rollback; `error.studioforge.instances[]` says what became of each | read `instances[]`, fix, retry |

`GET /api/capabilities` lists `lease_instances` and `multi_instance_default` (D91) in
`implemented.features`: gate on those.

## 7. What it does not do (yet)

* A dry run (`POST /models/load?dryRun=true`, `GET .../plan-recommended`) answers for the model as
  named: it does not preview the copy a real bound load would make.
* A base id is not load-balanced across different holders' copies (owner decision F-4): only a
  holder's own copies share its base-id traffic.
* A copy made for a second holder from the old `model_ids` shape has no shape of its own until its
  first load, which follows the base's saved settings (and so the default context ladder when the base
  saves none). Send `instances[].launch` to fix the shape up front.
