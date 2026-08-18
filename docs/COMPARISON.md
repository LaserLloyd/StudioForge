# Other model loaders: what was worth taking

StudioForge exists to replace LM Studio for one specific workload (an agent that needs JIT loading,
vision, tools, and a machine-readable control plane). Other loaders solve overlapping problems and
some of their ideas are worth having. This records what was considered, what was adopted, and what
was deliberately declined.

## LM Studio — the incumbent

**Adopted, because compatibility is the whole point:**

* Port `1234`, so migration is a host change.
* `GET /v1/models` lists **downloaded** models, not loaded ones.
* JIT load on first use; TTL idle-unload; per-request `ttl`.
* The `publisher/repo/` on-disk library layout, used **in place** — no import step, no copying, and
  LM Studio and StudioForge can share one library.
* `/api/v0/models` with per-model `state`, because that is the endpoint clients reach for when they
  want to know what is resident.
* The `lmstudio://open_from_hf?model=…` deep link from HuggingFace's "Use this model" button.

**Deliberately not copied** — these are the behaviours that made LM Studio painful to build against,
documented from a real client's workaround code:

| LM Studio behaviour | What StudioForge does |
| --- | --- |
| HTTP `200` for unrouted paths ("Returning 200 anyway") | `404` with a JSON error envelope |
| HTML error pages on 5xx | JSON always, every status |
| Errors as unstructured prose that clients regex-match | Stable `error.code` on every failure |
| `repetition_penalty` silently ignored | Accepted as an alias for `repeat_penalty` |
| `context_length` silently ignored on one of two load paths | One load path, all fields honoured, effective values echoed |
| Strict rejection of unknown load-config keys, varying by build | Unknown keys warned, not fatal |
| Three model-list endpoints with three different shapes (`data`/`models`, `id`/`key`) | One shape, `id` everywhere |

## Ollama

**The good idea: `Modelfile`.** A named model that bundles a base plus a system prompt and sampler
settings, selectable by name — the client picks a *persona*, not a file.

StudioForge already has the mechanism: **virtual models** (a registry entry that is a base plus an
adapter set, listed in `/v1/models` and JIT-loadable like any model). That was built for LoRA, and
it generalizes to the Modelfile idea cleanly. Extending virtual models to also carry sampler
defaults and a system prompt is the single most worthwhile borrow from another loader.

Also adopted in spirit: `keep_alive` per request → our per-request `ttl`.

**Declined:** Ollama's own registry and `/api/generate`/`/api/tags` endpoints. Models here come from
HuggingFace as GGUF (a fixed constraint), and a second API surface is maintenance cost for clients
that could simply use the OpenAI one.

## oobabooga text-generation-webui

**The good ideas:**

* **Per-model loader selection.** StudioForge has the equivalent in per-model **engine pinning** —
  a model can pin an older `llama.cpp` build when a new one breaks it. Same escape hatch, narrower
  scope (there is only one inference engine here, by design).
* **Named sampler presets.** Currently sampler defaults are per-model only. A named, reusable
  preset would be a real usability win and folds naturally into the virtual-model work above.
* **A three-tier settings surface** (basic / advanced / raw flags) — adopted directly, including
  the raw "extra flags" escape hatch, with the addition that flags are validated against the pinned
  engine's own `--help` at save time rather than failing mysteriously at load.

**Declined:** multiple inference backends (ExLlamaV2, Transformers). The core architectural rule
here is "do not implement inference, and have exactly one engine to supervise". Multiple backends
would multiply the VRAM planner, the flag surface, and the failure modes by the number of backends.

## KoboldCpp

**Adopted:** context shifting is exposed (`--no-context-shift` is a per-model setting), and the
single-artifact deployment philosophy shows up as the versioned `engines/<tag>/` directories.

**Declined:** the KoboldAI API surface, and "smart context" heuristics — llama.cpp's own prompt
cache reuse (`--cache-reuse`, on by default here) covers the same need for agent workloads, where
prompts are near-identical between turns.

## vLLM

**Adopted in spirit:** continuous batching (`--cont-batching` + `--parallel`) and prefix caching
(`--cache-reuse`) are both exposed and on by default where appropriate.

**Declined:** vLLM itself. It is the better choice for high-concurrency serving of a single
unquantized model on datacentre GPUs. This box runs many different quantized GGUFs, one or two at a
time, swapped constantly — which is llama.cpp's strength and vLLM's weakness.

## Summary of what is worth doing next

1. ~~**Extend virtual models into full presets**~~ — **done.** A virtual model can now carry a
   system prompt and request-time sampler defaults (`POST /api/virtual-models`); presets that
   differ only in request-time fields share one `llama-server` instance over their base. See
   DECISIONS.md D13.
2. **Named sampler presets** reusable across models — still open; today a preset lives on one
   virtual model rather than being a named, shareable object.

Everything else worth having from these projects is already in.
