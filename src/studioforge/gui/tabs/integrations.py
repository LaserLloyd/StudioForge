"""Integrations: ready-to-paste settings for the programs that will use this server.

One builder, :func:`studioforge.core.integrations.build_snippets`, serves the
first-run wizard, the CLI and this page, so every surface hands out the same
text. The page adds two things: a key picker (with an inline "create a key"
that shows the token once, in a dialog, and never again) and the existing
OpenClaw and protocol-handler panels.

The snippets never contain a real token: a token exists in one place, the
one-time dialog. The picker only decides which *name* the placeholder carries,
so the text says which key to paste where.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from studioforge.core.integrations import KEY_PLACEHOLDER, build_snippets
from studioforge.gui.tabs import GuiContext, panel_guard
from studioforge.gui.tabs.overview import server_root
from studioforge.gui.tabs.users_keys import create_key_form
from studioforge.gui.tabs.widgets import code_block
from studioforge.logging import get_logger

log = get_logger(__name__)

#: Snippet key -> title shown above the block, in display order.
SNIPPET_TITLES: tuple[tuple[str, str], ...] = (
    ("openclaw", "OpenClaw"),
    ("openai_sdk", "OpenAI Python SDK"),
    ("anthropic_sdk", "Anthropic Python SDK"),
    ("dispatch", "DisPatch"),
    ("curl", "curl"),
    ("mcp_url", "MCP address (for agents)"),
    ("mcp_claude_cli", "Claude Code: --mcp-config file"),
    ("mcp_agent_sdk", "Claude Agent SDK: mcp_servers"),
)

ANY_KEY = "(any key — show a placeholder)"


def key_choices(ctx: GuiContext) -> list[str]:
    """Names of the active keys, for the picker."""
    accounts = getattr(ctx.api_state, "accounts", None)
    if accounts is None:
        return []
    try:
        return [k.name for k in accounts.list_keys(include_revoked=False) if k.active()]
    except Exception as exc:  # noqa: BLE001
        log.debug("could not list keys for the picker", error=str(exc))
        return []


def library_model_ids(ctx: GuiContext) -> list[str]:
    try:
        if ctx.registry is None:
            return []
        return [r.id for r in ctx.registry.all() if not r.is_virtual]
    except Exception:  # noqa: BLE001
        return []


def key_label(choice: str | None) -> str | None:
    """The placeholder text for a picked key (``None`` = the generic placeholder)."""
    if not choice or choice == ANY_KEY:
        return None
    return f"<paste-the-{choice}-key-here>"


def snippet_set(ctx: GuiContext, choice: str | None, model: str | None) -> dict[str, str]:
    """The snippets for the current choice. Never contains a real token."""
    base = server_root(ctx)
    mcp_path = ctx.config.mcp.path if ctx.config.mcp.enabled else "/mcp"
    return build_snippets(
        base,
        key=key_label(choice),
        mcp_url=f"{base}{mcp_path}",
        model=model or "<model-id>",
    )


def render(ctx: GuiContext) -> None:
    from studioforge.gui.tabs import server

    with ui.column().classes("w-full gap-4 p-2"):
        with panel_guard("Connect your programs"):
            _snippets_card(ctx)
        server.integration_panels(ctx)


def _snippets_card(ctx: GuiContext) -> None:
    holder = ui.column().classes("w-full gap-3")
    state: dict[str, Any] = {"key": ANY_KEY, "model": None}

    def draw() -> None:
        holder.clear()
        models = library_model_ids(ctx)

        def set_state(**changes: Any) -> None:
            state.update(changes)
            draw()

        keys = key_choices(ctx)
        if state["key"] not in {ANY_KEY, *keys}:
            state["key"] = ANY_KEY
        if models and state["model"] not in models:
            state["model"] = models[0]
        with holder:
            ui.label("Connect your programs").classes("text-lg font-medium")
            ui.label(
                "Copy the block for the program you are setting up. Replace the key placeholder "
                "with a key from the list below, or make a new one."
            ).classes("text-xs opacity-70")
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                picker = ui.select(
                    [ANY_KEY, *keys], value=state["key"], label="Which key will it use?"
                ).props("dense outlined")
                picker.classes("w-72")
                picker.on_value_change(lambda e: set_state(key=e.value))
                if models:
                    chooser = ui.select(models, value=state["model"], label="Example model").props(
                        "dense outlined"
                    )
                    chooser.classes("w-72")
                    chooser.on_value_change(lambda e: set_state(model=e.value))
            snippets = snippet_set(ctx, state["key"], state["model"])
            code_block(
                "Base URL",
                snippets["base_url"],
                note="Use this wherever a program asks for an OpenAI-compatible server address.",
            )
            for name, title in SNIPPET_TITLES:
                code_block(title, snippets[name])
            ui.label(snippets["mcp_note"]).classes("text-xs opacity-70")
            if not keys:
                ui.label(
                    f"No keys exist yet, so the placeholder {KEY_PLACEHOLDER} is shown. While no "
                    "sign-in is required, programs on this computer can use any text as the key."
                ).classes("text-xs opacity-70")
            create_key_form(ctx, draw)

    draw()
