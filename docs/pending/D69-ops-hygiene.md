## D69 — Ops hygiene from the 2026-09-22 review: log lines that say what happened, log files that stop growing, a restart that does not wait for nothing

**Status.** Pending, lane `lane/ops-hygiene` (based on `b73de7e`). One commit per item, numbered as
in the 2026-09-22 review's prioritized list. `tests/unit` green at each commit; ruff, `ruff format
--check` and mypy (the gated packages) clean.

**Context.** The 2026-09-22 review read the live logs from 2026-09-13 on. Most of the warning volume
was not a fault. It was the same per-state fact repeated on every load, preview planners logging at
INFO as if they were loads, a miss the D51 safety band already absorbs, and exception text that
came out empty. The real faults it found were NiceGUI tracebacks, a 30 s restart drain with nothing
in flight, and three log files with no rotation. The sections below give the evidence, the change
and the tests for each item.

### §7 — Upstream failures name their exception

**Evidence.** `str()` of several httpx exceptions is empty. `openai_routes._stream_upstream` logged
`stream failed error=` six times on 2026-09-20 20:48:17, the only record of six cut streams.
`_forward` raised `llama-server for '…' failed: . Recent llama-server output: …`, four ERROR lines
on 09-16/17.

**Change.** `openai_routes._exc_text(exc)` returns `ClassName: message`, or the class name alone when
the message is empty. It is the shape `supervisor._watch` has used since D60. Both sites use it: the
`stream failed` warning, the mid-stream SSE error frame and the non-streamed `UpstreamError`.
`_consume_load_exception` was left alone: its exceptions are StudioForge errors with messages.

**Tests.** `tests/unit/test_upstream_error_text.py`: the helper, a stream cut by `ReadError('')`
(warning field, frame text, `[DONE]` still sent, request slot released), and a non-streamed
`RemoteProtocolError('')`.

### §8 — Preview planners log at DEBUG

**Evidence.** `_log_plan` and the terminal refusal honour `Planner._log_plans`, but two lines did not.
`re-planned after eviction` was logged at INFO from every catalog, placements and fit preview.
On 2026-09-20 that was 560 of 1585 log lines (154 of 670 on 09-22), each one naming an eviction of
the embedding or judge model "for" K2 that never happened. The review's first reading took them for
real K2 evictions. `load rejected: device leased to another holder` had the same gap: a preview
planning a model whose saved `device_override` names a leased card logged it at INFO. Almost all of
its 340 live lines came from real refused loads (332 sit beside a request refusal); those still log
at INFO, and only the preview copies go.

**Change.** Both now use `log.info if self._log_plans else log.debug`, the same rule as `load
planned`. A real load (`log_plans=True`) logs exactly as before. The server-chosen candidate case of
the lease line stays at DEBUG, as it was.

**Tests.** `tests/unit/test_preview_planner_logging.py`: both lines at DEBUG with `log_plans=False` and
at INFO with `log_plans=True` (both fail against the old code).

### §14 — `auto` is sized as f16, silently; an unknown KV type warns once

**Evidence.** 54 `unknown kv cache type, assuming f16` warnings since 09-13, all with
`kv_cache_type=auto`, in bursts of 3-5 within 20 ms. `models.default_kv_cache_type: auto` is a
request for the quality ladder, which `Planner._kv_options` fans out. The download-fit preview
(`downloader.py`, the "does this remote file fit" sizing) passes the config default straight into
`kv_alloc_bytes`, and so into `kv_bytes_per_element`.

**Change.** `kv_bytes_per_element("auto")` returns the f16 width without logging. f16 is the
ladder's first and most expensive rung, so the answer can only err toward refusing. A genuinely
unknown type still gets f16 and a WARNING, once per type per process; repeats go to DEBUG. The
once-per-process rule is a small shared helper, `studioforge.logging.first_time(*key)`: a bounded
process-wide set (4,096 keys, then it starts over) with `reset_first_time()` for tests. §16 uses it
too.

**Tests.** `tests/unit/test_kv_type_auto.py`: `auto` equals f16 with no log line, both directly and
through `kv_alloc_bytes`. An unknown type warns once and then logs at DEBUG, and a second unknown type
warns on its own. `first_time` returns True once per key and resets at its cap.

### §16 — Per-state facts are said once

**Evidence.** Between 09-13 and 09-22, `registry.alias_collision` logged 92 times: three lines per
rescan for the same Dark-Scarlett IQ4_XS/Q5_K_M pair, one per shared short alias.
`thinking model loads with no reasoning_format` logged 136 times, once per load of every thinking
model, although it describes a per-model setting.

**Change.** `Registry._rebuild_aliases` collects the collisions per `(kept, dropped)` pair and logs one
WARNING per pair, with `aliases=[...]` (sorted) in place of the old single `alias=` field. It warns
only when that `(kept, dropped, aliases)` state is new to the process. A rescan that finds the same
collision logs it at DEBUG, and a collision that goes away and comes back is warned again. The
thinking-format warning in the manager's load path is a WARNING the first time per model per process
(`first_time("reasoning_format", model_id)`, §14) and DEBUG after. The manager edit is the call site
and the import only.

**Tests.** `tests/unit/test_registry.py::test_an_alias_collision_is_warned_once_per_state` (one line
per pair, DEBUG on rescans, re-warned after the collision comes back) and the existing collision test
reading `aliases`. `tests/unit/test_thinking_format_nag.py` covers three loads of one model, then a
second model: WARNING for each model once, DEBUG for the repeats, nothing for a model with a
`reasoning_format`. Both fail against the old code.

### §6 — The D63 dead zone: an over-estimate D51 cannot correct is not warned about

**Evidence.** `observed_correction` returns `None` when the factor is within 0.005 of 1, and the
factor is `ratio × OBS_SAFETY` (1.10). So for every over-estimate between 8.6 % and 9.5 % (ratio
0.9045 to 0.9136) D51 makes no correction. The plan stays on the formula, `observe` treats it as
uncorrected and applies the 5 % bar, and the load is warned. Every repeat does the same. The registry
has two Hy-MT2 rows 13 s apart, same configuration, both ratio 0.911 and both `corrected=False`.
There were 52 `vram prediction error exceeds the bar` warnings since 09-13: 28 were Hy-MT2 at −8.9 %,
and the rest were Precog-123B −9.2 %, Dark-Scarlett Q5_K_M −9.1 %, Orion-26B −9.4 %, Qwen3-VL-Emb-8B
−9.9 % and Hy-MT2 −9.1/−11.4 %.

**Choice.** The bar is asymmetric, not "skip the warning when D51 would make no correction". An
over-estimate costs headroom, never an OOM. Up to `1 − 1/OBS_SAFETY` (9.1 %) of headroom is what D51
reserves on purpose on every corrected plan, so a formula that far over already sits where a measured
plan would put the load. It is not a defect worth a WARNING, on the first load or on any later one.
Skipping only the uncorrectable band would still warn at −7 % on first loads and then correct the
plan *upward*. An under-estimate is the OOM direction and keeps the 5 % bar, so it stays loud.

**Change.** `PREDICTION_OVER_ESTIMATE_WARN_PCT = 10.0` sits beside `PREDICTION_ERROR_WARN_PCT = 5.0`
in `core/planner.py`, and the literal `0.005` became `OBS_NOOP_TOLERANCE`. `observe()` holds a
negative error (the child holds less than the formula said) to the 10 % bar and a positive one to
5 %. The warning's `bar_pct` and `last_observation()`'s `bar_pct` name the bar that applied, and
`last_observation()` also carries `over_bar_pct` (an additive field; `model_info.vram_prediction` passes
it through). The corrected-plan warning ("measured footprint exceeds the corrected estimate") is
unchanged: it is the under-estimate direction. Nothing about the correction itself changed.

**Tests.** `tests/unit/test_planner_prediction_error.py`:
- the over-estimate bar is pinned at or past the far edge of the uncorrectable band, computed from
  `OBS_SAFETY` and `OBS_NOOP_TOLERANCE` and walked ratio by ratio;
- the four live ratios (0.911, 0.908, 0.906, 0.901) are not warnings;
- a −20 % miss still is, with `bar_pct` 10;
- the old "7 % off either way warns" case is now split: +7 % warns at bar 5, −7 % does not.

### §11 — Unknown KV geometry is never planned as a free cache

**Evidence.** `kv_layers()` returns `[]` when a GGUF lacks a head dimension or every head count.
`kv_alloc_bytes` then returned 0, `Planner.estimate()` added the 0, and `size_slots` answered one slot
`"unknown"`. The model was planned from its weights alone and "fit" wherever they did. No library
model hits this today. A new architecture, or a conversion that dropped keys, would.

**Change** (`core/planner.py`).
- `fallback_kv_layers(meta)` covers the partial case, where a layer count and a width are known but
  `kv_layers` cannot answer. It charges every layer as full multi-head attention with no GQA: K and V
  each `max(n_embd, heads × head_dim)` wide. `kv_alloc_bytes` prices that at `FALLBACK_KV_CACHE_TYPE`
  (`f16`) whatever type was asked for, because a guessed geometry does not earn a quantized discount.
  `effective_kv_bytes_per_token`, the slot sizer and `max_ctx_for_budget_geometry` all fold over the
  same number.
- `kv_geometry_unknown(meta)` covers the case where nothing is known: no layer count, or no width at
  all. `Planner._plan_load` then returns `kv_geometry_rejection(record)` before any rung is tried.
  That refusal is a `LoadRejected` whose reason starts with `KV geometry unknown:` and names the
  missing GGUF keys. It carries the weights as its estimate and gives one suggestion: re-download or
  re-convert, then rescan. `fits_on` returns `None` for the same models, so the pre-download context
  matrix and the catalog agree with the load. A record with no metadata at all (`meta is None`) takes
  the same refusal. `load_recommended` already refused those.
- The refusal keeps the class code: `reason_code` is unset, so the wire code is `insufficient_vram`
  (a 507, WARNING in the log, "do not retry unchanged"). A dedicated code would be a new public
  contract row in OPENCLAW.md, which the D66 lane is editing, for a case no library model reaches.
  The reason text is what names it. The KV sizing rule for every model `kv_layers` *can* describe is
  unchanged.

**Tests.** `tests/unit/test_kv_geometry_fallback.py`:
- the fallback's arithmetic, and f16 even when q4_0 is asked;
- a known geometry never takes the fallback;
- a fallback model plans with a non-zero KV term, and `fits_on` agrees;
- a nothing-known model is refused naming `block_count`, `embedding_length` and
  `attention.key_length`, with no `0.00 GiB` in the message and `fits_on` returning `None`;
- a metadata-less record is refused the same way. The existing
  `test_rejection_without_metadata_still_reports_a_size` still passes.

### §13 — The parser keeps laguna's window and the MLA keys (capture only)

**Evidence.** laguna (`Laguna-S-2.1`) declares `attention.sliding_window = 512` with no
`sliding_window_pattern`. It also has a per-layer `attention.head_count` array and
`leading_dense_block_count = 1`. The parser required the pattern before it kept the window, collapsed
the head array to its maximum, and dropped the rest. MLA models (DeepSeek2/3, Kimi, GLM-DSA) declare
`attention.kv_lora_rank` and related keys, and none of them was kept. None of this data would be
there when someone checks the sizing rule against upstream.

**Change** (`core/gguf.py`, inside `meta_from_gguf` and the version constant only). `GgufMeta.extra`
gains these keys:
- `sliding_window`, whenever the key exists, pattern or not;
- `head_count_values`, `head_count_len` and, when the array arrived truncated, `head_count_truncated`,
  for a per-layer `attention.head_count`;
- `leading_dense_block_count`;
- `kv_lora_rank`, `q_lora_rank`, `key_length_mla`, `value_length_mla`, and `rope_dimension_count`
  when `kv_lora_rank` is present.

Nothing reads them. The planner's iSWA keys (`swa_window`, `swa_pattern`) still require the pattern,
so every KV number is unchanged, byte for byte: the tests compare each shape's allocation with and
without the new keys. Upstream's semantics for laguna are unverified, so this change does not alter
the sizing rule.

`META_FORMAT_VERSION` 2 → 3, so every registered model re-parses its header on the first scan after
the deploy (the boot scan runs in the background, D33). The remote-header cache in `hf_meta`
re-fetches on the next browse for the same reason.

A read-only parse of the real `Laguna-S-2.1…Q3_K_S.gguf` with this parser gives `sliding_window 512`,
`leading_dense_block_count 1` and `head_count_values` of length 48 in a 48, 72, 72, 72 repeat (one
48-head layer in four). That is the data the later check needs. The planner still charges 48 uniform
full layers: 24,576 MiB at 131,072 f16.

**Tests.** `tests/unit/test_gguf_capture.py`:
- the version bump;
- a laguna-shaped header keeps the window, the head values and the dense prefix, gains no
  `swa_*` keys, and sizes exactly as without the new keys;
- a DeepSeek2-shaped MLA header keeps all five MLA keys, with unchanged sizing;
- an iSWA header keeps both `swa_window` and the raw `sliding_window`, with unchanged sizing;
- a plain llama with `rope.dimension_count` gains none of the new keys.

### §17 — Dashboard actions draw on the page, not on the card the refresh deleted

**Evidence.** NiceGUI runs a click handler inside the slot of the button that fired it, and
`ui.notify` finds its client through that slot. The Loaded models panel rebuilds every card on its
refresh timer. By the time `_unload_one` had awaited `manager.unload`, the card was gone: `ui.notify`
raised `The parent element this slot belongs to has been deleted`, and NiceGUI's own exception handler
raised it again (35 lines each on 09-13 17:27, 09-15 22:16, 09-16 16:39, 09-22 23:10 and 23:19).

**Change** (`gui/tabs/dashboard.py` only). A small `_Page` helper captures `ui.context.client` at
handler entry, while the button's slot still exists. `with page:` enters the page's own content slot
for the whole action, so the busy spinner and the toast are drawn at page level and outlive a card
rebuild. After the await, `page.notify` / `page.error` draw only if the client is still alive; a closed
browser gets a log line (`gui action failed … page=closed`), not NiceGUI's "Client has been deleted but
is still being used". With no page context (a direct call), nothing is bound and behaviour is as before.

Every awaiting handler with that shape uses it: `_unload_one`, `_restart_model`, `_toggle_pin` and
`_reclaim_orphans` (whose Reclaim button sits in a row the holders timer rebuilds), the Unload-all and
Restart-server dialogs' `confirm`, and `_restart_engines`. The restart dialog also paints its banner
only while `element_alive(banner)`. `gui/tabs/chat.py` is untouched, as the brief asked, and the D50
single-flight and D32 guards are unchanged.

**Tests.** `tests/unit/test_dashboard_page_context.py` stands in for NiceGUI's slot lookup (the client
is reachable through the card's slot only while the card exists).
- Unload, restart and pin each toast at page level after their card was deleted mid-await.
- A failure after the card is gone is still a red toast.
- A page whose browser went away gets no toast, and its refresh still runs.
- A static guard checks that every awaiting handler captures the page and none toasts through the
  slot.

All six fail against the old code. The existing static guards in `test_gui.py` (single-flight keys,
`require_local_admin`) pass unchanged.

### §15 — Log files rotate, safely on a shared file; the watchdog stops logging every probe

**Evidence.** `logging.py` installed a plain `FileHandler`. `watchdog.log` was 61 MB, about 10.7k lines
a day, nearly all httpx INFO `HTTP Request: GET …/health 200 OK` lines: the watchdog's own logging
setup never quieted httpx, although the server's did. `studioforge.log` was 16.9 MB and
`tray-server.log` 13.5 MB.

**Who writes which file** (checked in code and in the live log):
- `studioforge.log` is written by the server (a `FileHandler`, configured twice: `_load`, then
  `create_app`) and by **the tray**. Every CLI command, the tray included, called `_load` →
  `configure_logging(log_dir=…)`, so the tray held the file open for weeks: its `server exited for a
  requested restart; respawning` lines are in `studioforge.log`. The file is also written by any
  one-shot CLI command while it runs, by the stdio MCP server (`run_stdio`), and by two servers at
  once for a moment during a restart handover.
- `watchdog.log` is written by the watchdog only.
- `tray-server.log` is opened by the tray at each spawn and inherited by the server child as its
  stdout/stderr. The child writes the file, and nothing can rotate a handle another process writes
  through.

**The hazard.** On Windows a file another process has open cannot be renamed, because Python opens
files without `FILE_SHARE_DELETE`. The stdlib `RotatingFileHandler` shifts `.1 → .2 …` *before* it
renames the live file. On a shared file, every failed attempt therefore pushes each backup one step
further out, deletes the oldest, and drops the triggering record with a traceback on stderr, and every
later record repeats it. Swapping the handler in on its own would have destroyed data on this rig.

**Change.**
- `studioforge/logfiles.py` (new, standard library only, so the watchdog may import it):
  - `SafeRotatingFileHandler` is for the one owner of a file. It renames the live file first, to a new
    timestamped name (`studioforge.20260923-210507.log`, `_N` for a second rotation in the same
    second, `.log` kept so it still opens as a log). That rename is the only step that can fail. If
    it does, nothing else is touched: the handler keeps appending, writes one `log rotation deferred`
    line into the file per episode, and retries every 5 min. After a successful rename it prunes the
    oldest copies beyond `backup_count` (a copy it cannot delete stays for next time) and writes `log
    rotated; the previous file is …` at the top of the new file. On POSIX, if another process rotated
    the file under the open handle, the handler follows it rather than rotating the fresh file.
  - `AppendFileHandler` is for a guest process: open, append one record, close. It never holds the
    file and never rotates.
  - `rotate_if_large` is for a file nobody holds at that moment.
- **Ownership.** `configure_logging(..., owner=True|False, max_bytes, backup_count)`. `studioforge
  serve` (`_load(…, owns_log=True)`) and `create_app` own `studioforge.log`. Every other CLI command,
  above all the tray, is a guest, and so is `run_stdio`. The flag is not sticky, so a test suite's
  earlier CLI call cannot change what a later `create_app` does. A replaced handler is closed, not just
  detached, so the server's second `configure_logging` does not pin the file either. Two servers during
  a handover both hold the file for a moment; the rename is then deferred, and nothing is lost.
- **The watchdog** (`watchdog/__main__.py`) owns and rotates `watchdog.log`, and sets `httpx` /
  `httpcore` to WARNING. Its poll already logs every health transition, and a failed probe raises into
  `health poll failed`. `tail_logs` (`watchdog/server.py:tail_file`) puts the tail of the newest
  rotated copy in front when the live file is short. `studioforge.logfiles` joins the watchdog's
  import allowlist beside `config` and `credential_guard`, and a test keeps it a stdlib-only leaf.
- **The tray** rotates `tray-server.log` in `_open_server_log`, when it is about to start a server:
  the previous child has exited and nobody holds the file. If a rename fails (an old child is still
  alive), the file just grows until the next start. The file's growth is bounded by one server
  lifetime. It duplicates the server's own records (stderr), and removing that duplication was left
  alone as a behaviour change nobody asked for.
- **Config.** `logging.file_max_mb: 20` (0 = never rotate) and `logging.file_backups: 5` (1–100) are
  in `RESTART_REQUIRED_KEYS`, `config.example.yaml`, and the Setup tab's generated Advanced section.
  `docs/RUNBOOK.md` "Where the logs are" says all of the above.

**Tests.** `tests/unit/test_log_rotation.py`:
- rotation at the limit with nothing lost between the newest backup and the live file, pruning, and the
  pointer line;
- a refused rename loses no record, touches no backup and notes it once;
- a retry after `retry_s`, and one note per episode across many refused retries;
- **a real Windows sharing violation**: a second handle held open without share-delete;
- `max_bytes 0`, the POSIX "rotated under us" case, and prune touching only this log's own backups;
- a guest never pinning the file, `rotate_if_large`, and the tray rotating its console before a spawn;
- `configure_logging` owner/guest (not sticky, replaced handler closed), and only `serve` owning (source
  check plus a real `_load`);
- the watchdog's handler and quieted httpx, `tail_file` reaching into the newest copy, the config keys,
  and backup naming order past `_9`.

`test_watchdog.py` gains the stdlib-only leaf check and the allowlist entry. `test_lifecycle_hardening`
checks for a `FileHandler` by `isinstance` rather than by class name.

**Deploy note.** The old tray process holds `studioforge.log` open until it is restarted on this code.
Until then the server's rotation is deferred: one note, retried every 5 min, nothing lost. The
watchdog outlives server restarts by design (D21), so its rotation and the silenced probe lines start
only when the watchdog process itself restarts. The first rotation of the 61 MB `watchdog.log` then
happens at its first line.
