## D66 -- A model the engine cannot load is refused before anything happens, from the build's own library

**Status.** Built on branch `lane/arch-preflight` (on `efd06bb`), `tests/unit` green, ruff and mypy
clean. Not merged or deployed; the orchestrator folds this record into `DECISIONS.md` and bumps
`LATEST_DECISION`. Nothing migrates: the verdict is read from the installed engine directories, so
builds installed before D66 need no reinstall.

**Incident, 2026-09-16 .. 09-22.** `InfinimindCreations/K2-Horizon-MoVA-36B-A4B-uncensored-GGUF`
(Q6_K, 30.8 GB, downloaded 09-18) has `general.architecture = 'k2-horizon'` and
`tokenizer.ggml.pre = 'k2-horizon'`. No llama.cpp release includes that architecture -- upstream
master (b11102) does not; only a fork branch does -- and the owner decided against downloading or
building a fork and against touching the engine. CrucibleForge asked for it three times (09-20
11:22, 09-20 14:00, 09-22 12:05), each time the same sequence: a GPU lease on CUDA [0, 1] for a
benchmark (a lease grant unloads the idle residents on its cards), `load-recommended` walking the
hardware modes, a plan (`estimate_mb=40473`), a composed launch (`--spec-type ngram-mod`, a
reasoning-format warning), a spawn, and 0.3 s later `llama_model_load: error loading model: unknown
model architecture: 'k2-horizon'`. The client got `502 model_load_failed` -- a code that reads as a
fault to report, not a fact to act on -- and nothing remembered the answer. The catalog went on
offering the model (`recommended.fits_now: true`, four placements), `/api/models` said only
`stopped` after each death (the supervisor drops a child that failed to start), and
`/api/capabilities` listed it as merely *unknown to the architecture list* because that list was a
b10425 snapshot. An earlier `longcat-flash-sparse` model (09-16, three spawns) died the same way.
No resident happened to be evicted for those six spawns -- the 5090 pair had just been vacated -- but
nothing prevented it: with the chat model resident on [0, 1], the benchmark lease would have unloaded
it for a load that could not succeed. (The `re-planned after eviction ... K2-Horizon` lines of 09-20
evening and 09-22 morning are the catalog's preview planners, not loads.)

**Decision.**

1. **The build's own library is the source of truth.** `llama.dll` (Linux `libllama.so`, possibly
   versioned; macOS `libllama.dylib`) beside the engine's `llama-server` holds llama.cpp's
   architecture table as one NUL-terminated ASCII literal per name. `core/engine.py` reads it once
   per (resolved path, mtime, size) -- warmed off the event loop at boot, re-read after a
   reinstall -- into an `ArchitectureTable`: every maximal run of identifier characters that ends at
   a NUL, kept as sorted reversed tails plus the exact runs instead of the bytes (b11037: ~3,600 runs,
   under 0.5 MiB, ~0.1 s to build, once). The server binary itself (a 9 KB stub on Windows),
   `llama-server-impl.dll` and `llama-common.dll` are never read as the table.
2. **The check is one-sided.** A name is known when `name + NUL` occurs anywhere in the library.
   *Absent* is certain -- llama.cpp resolves `general.architecture` against exactly these literals --
   and is the only answer that refuses. *Present* means "assume supported": a stray string that ends
   in the name, or a pre-tokenizer of the same name, is a false *yes* that costs the spawn it would
   have cost anyway (and the runtime memo below remembers it). A name found only as the **tail of a
   longer literal** still counts (`evidence: "suffix"`), because linkers tail-merge string literals:
   on the live, MSVC-built b11037 `llama.dll` the literal `qwen2` exists *only* as the end of
   `rwkv6qwen2` (`\0qwen2\0` occurs zero times). Requiring a clean preceding byte -- as an
   independent review suggested -- would have refused both qwen2 models in the library, and failed
   the `qwen2` canary so that the whole table read as unrecognised.
3. **Fail open, always.** No library, an unreadable one (an I/O error is not cached), an empty one,
   an architecture that is not architecture-shaped (`unknown`, uppercase, over 64 characters), a pin
   naming a build that is not installed, a supervisor stand-in without the accessor: each is
   `None` -- "cannot tell" -- and never refuses. A library that does not contain all three canaries
   (`llama`, `gemma`, `qwen2`) is not a table this reader understands and is `None` too.
4. **One typed refusal, before any side effect.** `UnsupportedArchitectureError`: `400
   unsupported_architecture`, `type: invalid_request_error`, `param: "model"`, `error.studioforge`
   `{model_id, architecture, engine_tag, source: "binary"|"runtime", first_failed_at, remedy,
   engine_tag_pinned}` (plus `rejected {kind, name}` for a runtime verdict and the active build's
   answer when the model pins a build). 400 like `context_exceeded` -- change the request, never
   retry it unchanged; not 502 (a fault), 503/507 (a wait) or 409 (retried by the OpenAI SDKs). It
   joins `WARNING_REJECTION_CODES`. The message is written for both an operator and an agent: *"'<id>'
   uses the model architecture 'k2-horizon', which llama.cpp build b11037 does not include, so it
   cannot be loaded. No StudioForge setting changes that; it needs a llama.cpp build that supports
   'k2-horizon'."* When a saved `engine_tag` pin is the cause and the active build includes the
   architecture, it says to clear the pin instead.
5. **Every side-effecting entry calls one helper first** (`ModelManager._refuse_unsupported`), for
   the build that would serve the model -- its `settings.engine_tag` pin, else the active engine,
   resolved through the same `resolve_binary` a spawn uses: `load` (before the hold, the tier memo
   and the lock; a ready resident handed back untouched is exempt, a forced reload is not and keeps
   the running child serving), `ensure_loaded` (before `_refuse_if_held`), `_recommended_prep`
   (`load_recommended` and the `plan_recommended` dry run refuse identically), `lease_check` (the
   pre-SSE refusal), `acquire_lease` (per `model_ids`, before the conflict scan and any eviction),
   `_load_locked` (before `_loading` and the priority hold) and `_load_gated` (a backstop behind the
   gate, before the lease profile and the planner). `plan_preview` (and MCP `plan_load`) renders the
   same refusal as `fits: false`, `reason_code: "unsupported_architecture"`. Streaming chat checks
   before the `200`; the benchmark routes and both benchmark runners check before their first lease.
6. **The background passes skip, and say so once.** The D41 pin reconciler, the D42 rebalancer and
   the boot autoload skip such a model with one WARNING per (model, build, name) -- the reconciler's
   60-900 s backoff would otherwise have spawned a doomed child for a pinned model all day.
7. **A runtime memo for what the library cannot answer.** When a child dies at startup with
   `unknown model architecture: '<x>'` or `unknown pre-tokenizer type: '<x>'` in its log tail -- only
   those two markers, never `CONFIG_ERROR_MARKERS`' "does not exist" / "no such file" -- the failure
   becomes the typed 400 (`source: "runtime"`, the stderr tail attached) and is remembered against
   (model path, mtime, resolved build tag) -- when the rejection is the model's own: a different
   architecture name, or a pre-tokenizer while a draft model rode along, may be the draft's, so that
   request is refused but nothing is remembered (detaching a draft changes no file). Later loads of a
   remembered file on that build are refused before they hold, plan, lease, evict or spawn. It
   lapses when the file changes, when the build's library
   is reinstalled (its signature moves) or when the library is rescanned, and is cleared on an engine
   install or activation (`EngineManager.on_engine_change`, wired in `build_state`) or a restart. A
   pre-tokenizer probe of the library was considered and rejected on live data: six library models
   declare `tokenizer.ggml.pre = 'default'`, which llama.cpp accepts, yet `default\0` occurs nowhere
   in any installed build's `llama.dll` (a short constant compare is compiled inline), so the probe
   would have refused six working models.
8. **The explicit tier is remembered only after a load succeeds** (review item 19). `load()` used to
   write `_model_priority` before the lock, so a load the planner, the hold or the engine then
   refused still re-tiered the model. Decay on unload remains the owner's policy call.
9. **Every surface says the same thing.** `arch_supported: true | false | null` (+ `arch_note` when
   false) on `GET /api/models`, the `/v1/models` `studioforge` block, catalog rows (`list_models`,
   `model_options`) and MCP `model_info`; for a certain no also `engine_supported: false` and
   `unsupported_reason`. A catalog row that cannot load is not planned: `fits_now: false` with
   `fits_now_basis`, no options, no placements, `recommended: null` -- the compact view drops a
   `true` so loadable rows cost nothing. `last_load_failure {at, code, message, engine_tag}` shows the
   last launch that died until one succeeds. `ModelManager.unsupported_reason(record)` is the Chat
   tab's short answer. The GUI Models tab shows an "Unsupported arch" badge (theme `negative`) with
   the reason, and Load explains instead of loading; the settings dialog and its fit verdict say so.
   `/api/capabilities` judges each model from its build's library (`library.unsupported_by_engine`
   rows with `source: "binary"`, `architecture_verdict_source`), and when the report's build's
   library is readable its engine block says `capability_source: "binary"`,
   `capability_describes_engine: true`, `architecture_library: "llama.dll"`, with the architecture
   list reduced to the names that build contains; the bundled snapshot now supplies only that list's
   candidates and the ftype/ggml-type names (`docs/RELEASING.md` amended).

**Not taken.** A fork build or any engine change (the owner's decision). Requiring a non-identifier
byte before the name for a refusal (item 2: tail merging refuses working models, `qwen2` on the live
build). Probing pre-tokenizers from the library (item 7). Reading the server binary when no library
exists. Persisting the memo across restarts: the library probe answers again at the next load, and a
memo that outlived its cause would be a new way to refuse a working model. Refusing a lease that
names no model. A new MCP tool (the count is pinned). A capability feature key: adding one to
`SERVER_FEATURES` needs `LATEST_DECISION = 66`, which is the fold's job -- suggested key
`unsupported_architecture_code` (and `arch_supported_field` if a second is wanted).

**Consequences.** Wire changes, all additive except one: a load that used to die as `502
model_load_failed` for these two markers is now `400 unsupported_architecture` -- earlier, and
usually without any spawn at all. New fields as listed; catalog rows for unloadable models lose
their options, placements and `recommended`; `/api/capabilities` reports `capability_source:
"binary"` on a rig whose library is readable, and its architecture list shrinks to names the build
contains. The `list_models` payload budget in `tests/unit/test_mcp.py` is re-anchored by the one
`catalog_hint` sentence (~150 characters, fixed per call). One-sidedness means a model the library
cannot judge still costs one spawn before the memo refuses it; a false *yes* costs exactly what every
load cost before D66.

**Live check (read-only).** The committed table against the live `b11037` `llama.dll`, for all 34
records in the library: 33 supported -- 31 with exact evidence, 2 with suffix evidence (the two
`qwen2` models, per item 2) -- and exactly one refused: K2-Horizon, `'k2-horizon'`. The same result on
all eight installed builds, b10425 through b11037. The independently reviewed exotic names are all found
exactly (`qwen35`, `qwen35moe`, `gemma4`, `kimi-linear`, `laguna`, `muse-glimmer`, `hy_v3`,
`deepseek4`, `nemotron_h_moe`); `k2-horizon` and `longcat-flash-sparse` are absent.

**Tests.** `tests/unit/test_arch_probe.py` (41): the table (absent/exact/suffix, the live build's
`qwen2`-inside-`rwkv6qwen2` bytes, a longer literal does not make its prefix known,
non-architecture names, canaries), the library search per platform
name and `lib/`, the per-signature cache and its re-read, an I/O error not cached, the engine manager
and supervisor answers, `on_engine_change`, the two startup markers and the failures that are not
them, the memo's keying and lapses, the verdict's words, remedy, details and 400 shape.
`tests/unit/test_arch_preflight.py` (34): every entry above refuses before any hold, plan, lease,
eviction or spawn (a lease for K2 evicts nobody; the same lease for a loadable model does); a ready
resident is handed back but not force-reloaded; "cannot tell" loads; a pinned build names the fix;
the tier memo only after success; the reconciler, rebalancer and autoload skip with one WARNING; the
runtime memo and its lapses; a rejection that may be a draft's and a missing file are not memoised;
the Models tab rendered for real shows the badge and the Load refusal; through the real app with a
fake engine
on disk, streaming and plain chat, completions, load, load-recommended, plan-recommended, leases and
both benchmark routes are 400s (streaming before any SSE byte), logged at WARNING, and `/api/models`,
`/v1/models`, the catalog, `/api/capabilities`, MCP `model_info` / `plan_load` / `load_model` carry
the verdict. `tests/unit/test_docs.py` and `tests/unit/test_mcp.py` pin the new code in the rig
page's failure table and in INSTRUCTIONS; `tests/unit/test_rejection_log_levels.py` expects it in
`WARNING_REJECTION_CODES`.
