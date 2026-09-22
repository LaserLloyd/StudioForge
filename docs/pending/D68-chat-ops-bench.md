## D68 — The Chat tab is an ops bench: "(Loaded model)" by default, and every reply carries its numbers

**Context.** On this rig the Chat tab is used operationally, not conversationally:
open it, check that the loaded model answers, and see how fast it is. Or pick a model
that was just downloaded, load it and test it. The old tab was built for chatting. It
had a "Use the loaded model" switch that was off by default, a picker that defaulted
to the first model in the library, a single live tok/s label (one streamed chunk
counted as one token, measured from the click, so the load and prefill were folded
into the "rate"), and no way to see how the target was launched. The switch followed
the most recently *used* model, so another client's traffic could move it.

**Decision.**

1. **The picker's first entry is "(Loaded model)", and it is the default.** It is not a
   model id. It is re-resolved on every poll and every send (`state.chat_pick`):
   - the most recently **loaded** ready chat model (`InstanceInfo.started_at` is stamped
     when a child becomes ready). Loaded, not used, so traffic from other clients
     cannot move it;
   - with nothing ready, a chat model that is loading right now (Send waits for it);
   - with nothing loaded at all, the **newest download** (the catalog's
     `downloaded_at` rule, `mtime or added_at`). Virtual models are skipped, and so are
     models the server says it cannot load (the D66 hook `manager.unsupported_reason`,
     used only when present), so Send or Load is a one-click "does the model I just
     downloaded work?".

   The entry's label always says what it resolves to ("(Loaded model) — <id>",
   "… (loading…)", "… — nothing loaded · newest: <id>"). The rest of the list is loaded
   models first, then loading ones, then the library newest download first. Each entry
   is marked "· loaded", "· loading", "· failed" or "· cannot load". An explicit pick is
   honoured while the model exists. A pick that has left the library falls back to the
   "(Loaded model)" rules.

2. **A target card** above the conversation shows the target's state badge
   (Loaded / Loading… / Not loaded / Failed) and why it is the target.
   - Loaded: GPUs with card names, context × slots, KV types, speculative mode (MTP
     heads / draft model / n-gram), engine build, when and by whom it was loaded, tier,
     idle-unload TTL, request count, last decode rate.
   - Not loaded: size (+ vision projector), quant, arch, parameters, trained context,
     when it was downloaded.
   - Always: features (vision, thinking, tools, MTP heads).

   **Load** (not loaded) and **Unload** (loaded) buttons sit beside the picker. Load is
   `ensure_loaded(model_id, priority=1)`, the same call Send makes. Unload mirrors the
   Dashboard (D55 lease rules, single-flight per model).

3. **Every reply carries its numbers** (`state.chat_run_metrics`, `chat_metric_tiles`,
   `chat_metric_footer`):
   - **Load**: wall time from Send until ready, only when the send had to load.
   - **TTFT**: from the request reaching the child until the first streamed token
     (thinking counts). Excludes the load; includes prefill.
   - **Prefill** and **Decode**: from llama-server's own `timings` block (`prompt_n`,
     `prompt_ms`, `prompt_per_second`, `cache_n`, `predicted_*`, `draft_n`,
     `draft_n_accepted`). The block arrives on the final stream chunk when the request
     sets `stream_options.include_usage`, which the tab now always does. This was
     measured on b11037: the final chunk carries `usage` and `timings` with empty
     `choices`. A fully cached prompt shows "cached", not a meaningless rate.
   - **Overall**: completion tokens over the whole request (prefill included, load
     excluded), plus "incl. load" after a cold start.
   - **Total**: click to last token.
   - Footer: tokens in and out, speculative acceptance, and how the reply ended
     ("hit max_tokens", "stopped by you").

   If the engine's timings never arrive (Stop, older build), decode is estimated from
   the stream and labelled "estimated".

4. **Quick tests**: Hello (smoke/TTFT), Count to 100 (decode), Long answer (sustained
   decode), Prefill ~4k (a deterministic ~4,000-token prompt; run it twice to see the
   prompt cache). Plus a **Stop** button, thinking folded into a "Thinking" expansion
   (inline `<think>` or `reasoning_content`), a "Send the conversation so far" switch
   (off = each prompt measured on its own), max_tokens default 2048 (thinking models
   ran out of room at 512), request settings folded away.

5. Fixed in passing: the stream went to `supervisor.base_url(record.id)`, which is wrong
   for a virtual model (its child belongs to the base). It now uses `instance.model_id`
   for the port and for request accounting. A failed turn is dropped from the history,
   so the next message does not carry an unanswered one.

**Consequences.** The tab still talks to the model's own llama-server after
`manager.ensure_loaded`, so a good result is still evidence that a client will work. It
bypasses the gateway's `/v1` layer (virtual presets, `status.clients` attribution), as it
always did. `ChatTarget`/`chat_target` and the switch are gone (their only user was this
tab). The metrics are per reply and not persisted. The Benchmark tab stays the place for
repeatable numbers.

**Tests.** `tests/unit/test_gui_chat.py` (46): resolution rules (most recently loaded
beats most recently used; newest download; skips virtual and unloadable; loading;
embedding never a target; explicit pick; vanished pick; virtual follows its base;
failed), picker order and labels, card facts for loaded and not-loaded models, GPU
summary, metrics from the measured b11037 timings, cold start, client fallback, fully
cached prompt, speculative acceptance and Stop in the footer, junk timings, formatting,
the thinking split, quick-test prompts, the SSE parser, `include_usage` in the payload,
the optional unloadable hook, and a rendered page naming the loaded model. The static
guards in `test_gui.py` (explicit-zero samplers, chat tier 1) still pass unchanged.
