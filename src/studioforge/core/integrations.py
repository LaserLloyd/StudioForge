"""Copy-paste integration snippets: OpenClaw, the OpenAI and Anthropic SDKs, DisPatch, curl, MCP.

One builder shared by the first-run wizard, the panel's Integrations page and
``studioforge setup``, so every surface hands out the same, correct text. The
builder takes the *base URL* the client will use (not the bind address), an
optional key and the MCP URL, and returns ready-to-paste strings. Nothing here
reads a secret: the caller passes the key it wants shown (a freshly created
scoped key, or a placeholder).

MCP for Claude agents uses the two shapes the Claude clients accept (verified in
``RESEARCH.md`` R-MCP): a ``--mcp-config`` JSON file with ``type: "http"`` and a
``headers`` map, and the Agent SDK's ``mcp_servers`` dict of the same shape. The
SDK exposes the tools as ``mcp__studioforge__<tool>``; the allow-list wildcard is
``mcp__studioforge__*``. A **scoped key** is the right credential: ``read`` for an
agent that should only look, ``inference`` to let it load models, ``admin`` for the
destructive tools (delete, unload, config).
"""

from __future__ import annotations

import json
from typing import Any

__all__ = ["KEY_PLACEHOLDER", "build_snippets", "mcp_config"]

KEY_PLACEHOLDER = "<your-api-key>"
SERVER_NAME = "studioforge"


def mcp_config(mcp_url: str, key: str | None) -> dict[str, Any]:
    """The server entry both Claude shapes share: Streamable HTTP plus an Authorization header."""
    entry: dict[str, Any] = {"type": "http", "url": mcp_url}
    if key:
        entry["headers"] = {"Authorization": f"Bearer {key}"}
    return entry


def build_snippets(
    base_url: str,
    *,
    key: str | None = None,
    mcp_url: str | None = None,
    model: str = "<model-id>",
) -> dict[str, str]:
    """``{name: text}`` for every integration. ``base_url`` has no trailing ``/v1``."""
    base = base_url.rstrip("/")
    shown_key = key or KEY_PLACEHOLDER
    auth_header = f' -H "Authorization: Bearer {shown_key}"' if key else ""
    mcp = mcp_url or f"{base}/mcp"
    entry = mcp_config(mcp, key or KEY_PLACEHOLDER)

    snippets: dict[str, str] = {
        "base_url": f"{base}/v1",
        "openclaw": (
            "# OpenClaw: point its OpenAI-compatible provider at StudioForge\n"
            f"export OPENAI_BASE_URL={base}/v1\n"
            f"export OPENAI_API_KEY={shown_key}\n"
            "# Models preload through LM Studio's native endpoint, which StudioForge serves:\n"
            f'#   POST {base}/api/v1/models/load   {{"model": "{model}"}}'
        ),
        "openai_sdk": (
            "from openai import OpenAI\n\n"
            f'client = OpenAI(base_url="{base}/v1", api_key="{shown_key}")\n'
            f'reply = client.chat.completions.create(model="{model}",\n'
            '    messages=[{"role": "user", "content": "Say hello."}])\n'
            "print(reply.choices[0].message.content)"
        ),
        "anthropic_sdk": (
            "import anthropic\n\n"
            f'client = anthropic.Anthropic(base_url="{base}", api_key="{shown_key}")\n'
            f'reply = client.messages.create(model="{model}", max_tokens=256,\n'
            '    messages=[{"role": "user", "content": "Say hello."}])\n'
            "print(reply.content[0].text)"
        ),
        "dispatch": (
            "# DisPatch: Settings -> Connections -> add an OpenAI-compatible connection\n"
            f"Base URL:  {base}/v1\n"
            f"API key:   {shown_key}\n"
            "Model:     pick one from the list (StudioForge loads it on first use)"
        ),
        "curl": (
            f"curl {base}/v1/chat/completions \\\n"
            f'  -H "Content-Type: application/json"{auth_header} \\\n'
            f'  -d \'{{"model": "{model}", "messages": '
            '[{"role": "user", "content": "Say hello."}]}\''
        ),
        "mcp_url": mcp,
        "mcp_claude_cli": (
            "# Save as studioforge-mcp.json, then:  claude --mcp-config studioforge-mcp.json\n"
            + json.dumps({"mcpServers": {SERVER_NAME: entry}}, indent=2)
        ),
        "mcp_agent_sdk": (
            "# Claude Agent SDK (Python)\n"
            "from claude_agent_sdk import ClaudeAgentOptions, query\n\n"
            "options = ClaudeAgentOptions(\n"
            f"    mcp_servers={{{SERVER_NAME!r}: {json.dumps(entry)}}},\n"
            f'    allowed_tools=["mcp__{SERVER_NAME}__*"],  # all StudioForge tools\n'
            ")"
        ),
        "mcp_note": (
            "Give the agent a SCOPED key, not the owner password: "
            "`studioforge keys create --name my-agent --scope read` can look but not change "
            "anything; `--scope inference` may load models; only `--scope admin` may delete "
            "models, change config or unload. Tools are named mcp__studioforge__<tool>."
        ),
    }
    return snippets
