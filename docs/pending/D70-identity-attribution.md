## D70 — Identity and attribution: who loaded it, whose requests are running, who took it down, and which build is answering

**Context.** The 2026-09-22 read-only review of the live rig found four questions the system
could not answer about itself. `/health` said `1.26-09-04-3` from a checkout ~50 commits past that
tag. On 2026-09-20 20:48:17 an explicit unload cut six live streams and no log line said who had
called: the unload paths were the only stop paths that logged nothing. `loaded_by` was the route
literal every caller passed, `active_requests` a bare count, so "who loaded that model" and "whose
streams are those" were unanswerable — the S6 and X2 items the change-request triage had already
approved. And on 2026-09-15 12:24 a D42 rebalance stopped a resident, the relaunch died at startup,
and the log said "the model keeps its placement" over a model that was down.

Pending record, not a DECISIONS.md heading: `LATEST_DECISION` is untouched. When promoted, drop the
`"## D70" not in text` assertion in `tests/unit/test_docs_d70.py` and consider adding
`implemented.features` entries (`build_identity`, `in_flight_requests`, `loaded_by_client`,
`unload_attribution_log`) mapped to D70.

**Decision.**

1. **Version `1.26-09-23` and a `build` field (item 9).** The six pinned places move to
   `1.26-09-23` (PEP 440 `1.26.9.23`); `tests/unit/test_version.py` now also pins the README
   status line and the OPENCLAW-SETUP `/health` sample. `build` sits beside `version` on `/health`,
   `/api/health`, `/api/version`, `/api/status`, the MCP `server_status` tool, the
   "studioforge starting" log line and the `sfctl status` Server table: the short commit SHA,
   `-dirty` when tracked files carry uncommitted changes, `unknown` from a wheel, without git on
   PATH, or when git does not answer within 3 s. `src/studioforge/build.py` resolves it once per
   process (`functools.cache`), eagerly in `create_app`, so no request path runs a subprocess; a
   `.git` entry must sit at the package root before git is asked, so a wheel in a venv inside some
   other repository never reports that repository's commit. No tag is cut here.

2. **In-flight request records (item 20, S6).** `InstanceInfo.in_flight` lists the requests
   behind `active_requests`, oldest first, as `{id, started_at, client}`.
   `Supervisor.mark_request_start(model_id, *, client=None)` returns the request id;
   `mark_request_end(model_id, *, tokens_per_second=None, request_id=None)` removes exactly that
   record. The window is bounded (`IN_FLIGHT_RECORDS_MAX = 64`; past it a request is counted but
   not described) and never longer than the count, which every caller decrements in a `finally` —
   so a record leaves on success, error and cancel alike. A caller without an id (the pre-D70
   signature: the benchmarks, the smoke test) trims from the oldest end. Exposed on `/api/status`
   rows, the MCP compact instance row (`in_flight`) and `sfctl status`.

3. **`X-SF-Client` on loads (item 21, X2-SF).** The routes compose the label — else the peer
   address, else nothing — into the load's `source`: `jit:/v1/chat/completions (clawchat)`,
   `api:/api/models/{id}/load (crucibleforge)`. `loaded_by` keeps the route literal in front, so a
   reader matching on the prefix is unaffected, and the supervisor splits the label back into the
   new `loaded_by_client` field for readers that want it without parsing. Chosen over a new manager
   argument because it changes no load-path signature — the D66 lane is in those. The convention
   lives in `src/studioforge/core/attribution.py` (`client_label`, `client_of`,
   `attributed_source`, `split_source`). MCP and GUI loads carry no label (`loaded_by_client:
   null`); the GUI chat tab marks its requests `gui:chat`.

4. **`sfctl status` columns (item 21).** `Client` and `Started` sit after `Active`: the distinct
   labels in flight and the age of the oldest request; an idle row says `loaded by <label>`; a
   pre-D70 server gets dashes. `--json` is the server payload, so the two new fields simply arrive.
   The Server table gains a `build` row when the server sends one.

5. **Every explicit unload is logged (item 5a).** `manager.unload(..., source=, client=, peer=)`
   and `unload_all` log one line per deliberate unload: `explicit unload` at INFO, `explicit unload
   cuts live requests` at WARNING when `active_requests > 0` (the path does not drain), with
   `source` (`rest`, `mcp`, `gui`, `gui:chat`; `in-process` when the caller did not say), `client`,
   `peer`, `force`, `active_requests`, `in_flight_clients` and `loaded_by`. The REST routes pass
   `rest` plus the request's label and peer; the MCP tool passes `mcp` (the PIN gate is its
   identity). Behaviour is unchanged on purpose: refusing a busy unload (item 5b) waits for the
   owner. Left to the orchestrator: the Dashboard's `_unload_one` is in the function the
   ops-hygiene lane is fixing; one line there (`source="gui"`) completes the set.

6. **Rebalance self-heal (item 10).** `_rebalance` captures the resident before the move. When the
   load fails and the model is gone, `_restore_after_failed_rebalance` relaunches the PREVIOUS plan
   on the PREVIOUS devices with `allow_evict=False`, at the tier it had, as
   `source="rebalance-restore"`. When that fails too the model is down, and the log (ERROR
   "... the model is DOWN until its next load") and the eviction book (`reason: rebalance-failed`,
   `evicted_by: rebalance`, on `/api/evictions`) say so. When the model is still resident — a
   refusal before the stop — the old message stands, because there it is true. The cooldown stays
   stamped, so a failing move cannot loop. Kept inside the rebalance path; `_start_with_retry`'s
   docstring still describes the pre-D70 consequence and can be trimmed when that function is next
   touched.

7. **S7 docs (item 22).** D48's two stale TTL passages carry an in-place "Amended 2026-09-23"
   pointer to D60 (no new heading). OPENCLAW-RIG.md §9 gains one combined table: the three tiers
   with their shipped idle TTLs, the request-level `ttl` rule, and the codes that are retry-after
   versus terminal — `unsupported_architecture` (400, the model's architecture is not in the
   active llama.cpp build) listed as terminal. `tests/unit/test_docs_d70.py` pins the TTL numbers
   to `Config` and the split to the table.

**Wire changes, all additive.** `/health`, `/api/health`, `/api/version`, `/api/status` and
`server_status` gain `build: str`. `/api/status` rows, `/api/models/{id}/load*` responses and the
MCP compact instance gain `loaded_by_client: str | null` and `in_flight: [{id, started_at,
client}]`. `loaded_by` may now carry a ` (label)` suffix. `/api/evictions` may carry
`reason: "rebalance-failed"`. Nothing was renamed or removed.

**Merge notes.** `manager.py`: only `unload`/`unload_all`/`_log_explicit_unload` and
`_rebalance`/`_restore_after_failed_rebalance` changed. `supervisor.py`: the `start()` stamp, the
`model_spawn` line, `mark_request_*` and one constant. `openai_routes.py`: the `client=` thread
through `_forward`/`_stream_upstream`/`_stream_with_jit_load` and the five `ensure_loaded` calls.
`mgmt_routes.py`: imports, `/health`, `/status`, `/version`, both load routes' `source`, the two
unload routes, `_peer_host`, the evictions docstring. `management.py`: imports, `_compact_instance`,
`server_status`, `unload_model`. `errors.py`, `planner.py`, `capabilities.py`, `catalog.py`,
`dashboard.py` and `models.py` (beyond one `source="gui"` line) were not touched.
