# LM Studio / Anthropic / Responses wire fixtures

Example requests and responses for the compatibility surface added in W2. The unit tests in
`tests/unit/compat/` load them and assert that StudioForge's replies keep the **shape** (key set and
value types) of the response fixtures and that the request fixtures translate as documented. The
`tests/compat/` suite (needs a running instance) reuses the request fixtures.

## Provenance: nothing here was captured from a running LM Studio

No LM Studio install was available when these were written (it is not installed on the Windows rig).
Every file is therefore **derived from documentation, not captured**:

| File(s) | Derived from | Status |
| --- | --- | --- |
| `models_list_response.json` | LM Studio REST docs, "List your models" (`GET /api/v1/models`), fetched 2026-10-18 | derived; `key`, `loaded_instances[].id/config.context_length`, `capabilities.vision/trained_for_tool_use`, `quantization.{name,bits_per_weight}`, `size_bytes`, `params_string`, `max_context_length`, `format`, `display_name`, `publisher`, `architecture` are the documented names. Additive StudioForge fields: `arch`, `state`, `loaded_context_length`, `loaded_instances[].context_length`, top-level `object`/`data` |
| `load_request.json`, `load_response.json` | docs, "Load a model" (`POST /api/v1/models/load`) | derived; documented: `type`, `instance_id`, `load_time_seconds`, `status: "loaded"`, `load_config`. Additive: `model`, `context_length`, `ignored`, `warnings`, request `ttl`/`priority`/`parallel` |
| `unload_request.json`, `unload_response.json` | docs, "Unload a model" | derived; `{instance_id}` in and out (`model` accepted in as an additive alias) |
| `download_*.json` | docs, "Download a model" and "Get download status" | derived; `job_id`, `status` (`downloading`/`paused`/`completed`/`failed`/`already_downloaded`), `total_size_bytes`, `downloaded_bytes`, `started_at`, `completed_at`, `bytes_per_second`, `estimated_completion`. The docs' status route is `/api/v1/models/download/status/:job_id`; the brief's job-less form lists every job (additive). `job_id` is a StudioForge download group id. LM Studio's catalog identifiers are not supported, only Hugging Face `owner/repo` ids and links |
| `chat_501_response.json` | the W2 brief | not an LM Studio shape; it is StudioForge's own error envelope (`error.code = not_implemented`) |
| `anthropic_*.json` | Anthropic Messages API reference | derived from the public API reference (and LM Studio's `/v1/messages` docs for the supported subset) |
| `responses_*.json` | OpenAI Responses API reference (LM Studio serves the same shape since 0.3.29) | derived |

Open verification items (see the report): the docs page for `GET /api/v1/models` lists a top-level
`reasoning` field in its field table while its example nests it under `capabilities`; StudioForge
emits neither. Capture real LM Studio responses on a machine that has it and diff them against
these files before the first release.

All ids, token counts and timestamps are illustrative placeholders.
