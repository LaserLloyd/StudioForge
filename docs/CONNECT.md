# Connect a client

Three ways to use a running StudioForge: have a coding agent do the install, call it as an API from any OpenAI client, or manage it from an MCP agent such as OpenClaw. The README's [quick start](../README.md#quick-start) covers the shortest path; this page has the rest.

Install once (above), then pick whichever of these matches how you will use it — or let an agent
do the install for you.

## A. Let an AI coding agent do it

Open [Claude Code](https://docs.anthropic.com/en/docs/claude-code), a DeepSeek-based harness, or
any coding agent that can run shell commands **in the cloned folder**, and paste:

> Install StudioForge on this machine by following `docs/SETUP.md`: create the virtualenv with uv,
> install the llama.cpp engine with `studioforge engine --update`, point `models.dir` at my GGUF
> library, start it with `studioforge serve --open`, and confirm `GET http://127.0.0.1:1234/health`
> reports `"can_serve": true`. Only run `tests/unit`, never `tests/contract`.

The repo is written for that: [`DECISIONS.md`](../DECISIONS.md) explains every non-obvious rule,
[`docs/RUNBOOK.md`](RUNBOOK.md) says what each `/health` field means, and
[`docs/BENCHMARKING.md`](BENCHMARKING.md) is a playbook an agent can follow verbatim.

## B. Manually, as an API (any OpenAI-compatible client)

Once the server is up, it *is* a cloud-style API on your own hardware. Point any OpenAI client at it:

```bash
export OPENAI_BASE_URL=http://<studioforge-host>:1234/v1
export OPENAI_API_KEY=not-required        # any non-empty string until you create a key; then use that key
```

```bash
curl http://<studioforge-host>:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "<id from /v1/models>", "messages": [{"role": "user", "content": "hello"}]}'
```

```python
from openai import OpenAI  # https://github.com/openai/openai-python
client = OpenAI(base_url="http://<studioforge-host>:1234/v1", api_key="not-required")  # scrub-ok: the literal string clients send when auth is off
print(client.models.list())   # every downloaded model; naming an unloaded one loads it on demand
```

`GET /v1/models` lists every **downloaded** model, like LM Studio, and a request that names an
unloaded model just-in-time loads it. All three services listen on this machine only until you say
otherwise: to use it from another machine run `studioforge setup` (account, HTTPS, an admin key),
give each client its own key (`studioforge keys create`) and pass it as the `api_key` — see
[`docs/SECURITY.md`](SECURITY.md) and [`docs/SETUP.md` §6](SETUP.md#6-network--access).

## C. With OpenClaw (or any MCP agent)

Inference is path B. Management — load, unload, pin, benchmark, download, recover — is one stdio
MCP server on the agent's machine, provided by the companion package:

```bash
uv build --wheel -o dist                                              # in packages/studioforge-companion, on the host
uv tool install ./studioforge_companion-<version>-py3-none-any.whl    # on the agent's machine
sfctl servers add rig http://<studioforge-host>:1234 --api-key <inference-or-admin key> --use
```

OpenClaw's config key is `mcp.servers`, not the top-level `mcpServers` map:

```bash
openclaw mcp add studioforge --command sfctl --arg mcp
```

```json
{ "mcp": { "servers": { "studioforge": { "command": "sfctl", "args": ["mcp"] } } } }
```

Claude Code, Cline, LibreChat and other clients that take the generic map want this instead:

```json
{ "mcpServers": { "studioforge": { "command": "sfctl", "args": ["mcp"] } } }
```

That gives the agent 39 tools: the server's 29 management tools plus the watchdog's 10 recovery
tools, so it still holds `restart_server` when the main server is wedged. The step-by-step
two-machine install with a check after each step is
[`docs/OPENCLAW-SETUP.md`](OPENCLAW-SETUP.md); the loop an agent actually runs is
[`docs/OPENCLAW.md`](OPENCLAW.md). If an image service shares the same GPUs,
[`docs/OPENCLAW-RIG.md`](OPENCLAW-RIG.md) is the whole-rig page: one identity string, one set
of priority tiers, the lease etiquette and one failure-code table for both services.

---

