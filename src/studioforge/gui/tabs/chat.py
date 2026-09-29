"""Chat tab: an ops bench for checking a model end to end (D68).

On this rig the tab is used operationally: open it, see which model is loaded,
poke it, read the numbers -- or pick a model, load it and test it. So the picker
defaults to "(Loaded model)", the card above the conversation says what that is
and how it was launched, and every reply carries its load time, time to first
token, prefill and decode rates and overall throughput.

It still goes through the *same* code the OpenAI endpoints use --
``manager.ensure_loaded`` then a stream from the child's own port -- so a
successful chat here is real evidence that a client will work, not a separate
mock path that can drift. That includes image attachment: being able to paste a
screenshot and get an answer is the only practical way to verify a vision model
end to end without wiring up OpenClaw first.

The conversation itself is a :class:`~studioforge.gui.chat_conversation.Conversation`
of message blocks with stable ids, each painted once and then touched on its
own (edit, delete, regenerate, copy); nothing is saved -- it lives for the page
view. Model output is rendered with ``render_markdown`` (escaped, allowlisted
HTML), never ``ui.markdown``, which would pass a reply's raw HTML through.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import httpx
from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.chat_assets import CHAT_CSS, CHAT_JS
from studioforge.gui.chat_conversation import (
    ChatMessage,
    Conversation,
    plain_text,
    render_markdown,
)
from studioforge.gui.tabs import (
    GuiContext,
    busy,
    element_alive,
    notify_error,
    single_flight,
    viewer_may_change_box,
)

#: Guard against a paste of a 40 MP screenshot filling the socket buffer.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

#: Repaint a streaming reply at most this often. Each repaint re-renders the
#: whole Markdown body and ships it over the websocket, and a 0.5B model streams
#: ~700 tokens/s -- one repaint per token would only make the browser lag.
_REPAINT_S = 0.1

_STATE_BADGES: dict[str, tuple[str, str]] = {
    "ready": ("Loaded", "positive"),
    "loading": ("Loading…", "warning"),
    "failed": ("Failed", "negative"),
    "not_loaded": ("Not loaded", "grey"),
    "none": ("No model", "grey"),
}

#: Client-side focus handler for the model picker (see where it is attached).
_SELECT_ALL_ON_FOCUS = (
    "(e) => { const i = e && e.target; if (i && i.select) setTimeout(() => i.select(), 0); }"
)

_PASTE_SCRIPT = """
<script>
document.addEventListener('paste', (event) => {
  const items = (event.clipboardData || {}).items || [];
  for (const item of items) {
    if (item.kind === 'file' && (item.type || '').startsWith('image/')) {
      const file = item.getAsFile();
      if (!file) continue;
      const reader = new FileReader();
      reader.onload = () => emitEvent('sf_paste_image',
        {data: reader.result, name: file.name || 'pasted-image'});
      reader.readAsDataURL(file);
    }
  }
});
</script>
"""

_EMPTY_HINT = (
    "Send a message or pick a quick test. Every reply shows load time, time to first "
    "token, prefill and decode speed, and overall tokens per second."
)

_STOP_FIRST = "Stop the reply first."

_CTX_LEVEL_CLASSES = ("sfc-ctx-unknown", "sfc-ctx-ok", "sfc-ctx-warn", "sfc-ctx-full")


class ChatRequestError(RuntimeError):
    """The engine refused or failed a chat request; ``str()`` is the readable line.

    ``context_full`` marks the "prompt does not fit the slot" refusal, whose
    message already says what to do (Clear, delete older messages, reload).
    """

    def __init__(self, message: str, *, context_full: bool = False) -> None:
        super().__init__(message)
        self.context_full = context_full


def _request_error(
    status_code: int | None, body: Any, *, n_ctx: int | None, model_id: str
) -> ChatRequestError:
    full = st.context_overflow(status_code, body, n_ctx=n_ctx, model_id=model_id) is not None
    text = st.chat_error_text(status_code, body, n_ctx=n_ctx, model_id=model_id)
    return ChatRequestError(text, context_full=full)


def render(ctx: GuiContext) -> None:  # noqa: C901, PLR0915 - one screen, one flow
    images: list[dict[str, str]] = []
    conversation = Conversation()
    #: One entry per message id: that block's elements (see ``add_block``).
    blocks: dict[str, SimpleNamespace] = {}
    #: What a quick test's user block shows instead of its (long) prompt.
    shown: dict[str, str] = {}
    #: ``active`` while a send is in flight; ``stop`` is the Stop button's request.
    run: dict[str, Any] = {"active": False, "stop": False}
    #: Last painted picker options and card signature, so a poll that changes
    #: nothing sends nothing to the browser (and never closes an open dropdown).
    view: dict[str, Any] = {"options": None, "card": None, "actions": None}
    #: The per-conversation context window each reply was served with.
    limits: dict[str, int | None] = {}
    unloadable = _unloadable_check(ctx)

    ui.add_head_html(_PASTE_SCRIPT)
    ui.add_css(CHAT_CSS)
    ui.add_body_html(f"<script>{CHAT_JS}</script>")

    def records_now() -> list[Any]:
        return list(ctx.registry.all()) if ctx.registry is not None else []

    def instances_now() -> list[Any]:
        return list(ctx.supervisor.list()) if ctx.supervisor is not None else []

    # The tab is one flex column sized to the viewport (see ``fit`` in
    # chat_assets): model line, conversation window, composer all on screen,
    # the window taking whatever height is left.
    with ui.column().classes("w-full gap-2 sfc-root"):
        # --- what are we talking to? (one line; the rest behind Details) ------
        with ui.card().classes("w-full gap-1 py-2 px-3 sfc-none"):
            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                model = ui.select(
                    {st.LOADED_MODEL_CHOICE: st.LOADED_MODEL_LABEL},
                    value=st.LOADED_MODEL_CHOICE,
                    label="Model",
                    with_input=True,
                )
                model.props("dense outlined options-dense").classes("grow min-w-[14rem]")
                # NiceGUI fills the filter input with the selected label, so typing
                # edited "(Loaded model) — <id>" in place instead of filtering.
                # Selecting the text on focus makes the first keystroke replace it.
                model.on("focus", js_handler=_SELECT_ALL_ON_FOCUS)
                status_badge = ui.badge("", color="grey").classes("text-xs")
                load_button = ui.button("Load", icon="play_arrow").props("outline dense no-caps")
                with load_button:
                    load_tip = ui.tooltip("")
                unload_button = ui.button("Unload", icon="stop_circle").props("flat dense no-caps")
                details_button = ui.button(
                    "Details", icon="expand_more", on_click=lambda: toggle_details()
                ).props("flat dense no-caps")
                details_button.tooltip("Where and how the model runs")
            # Warnings stay out of the fold: they are why a Load is disabled.
            warn_label = ui.label("").classes("text-xs text-warning whitespace-pre-wrap")
            details = ui.column().classes("w-full gap-1")
            details.set_visibility(False)
            with details:
                with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                    target_name = ui.label("").classes("text-sm font-mono break-all")
                    reason_label = ui.label("").classes("text-xs opacity-70")
                facts_row = ui.row().classes("w-full gap-x-6 gap-y-2 flex-wrap")
                others_label = ui.label("").classes("text-xs opacity-70")
                hidden_label = ui.label("").classes("text-xs opacity-60")

        # --- the conversation ----------------------------------------------
        with ui.row().classes("w-full items-center gap-2 no-wrap sfc-none"):
            ui.label("Conversation").classes("text-sm font-medium")
            count_label = ui.label("").classes("text-xs opacity-60 whitespace-nowrap")
            ctx_label = ui.label("").classes("sfc-ctx sfc-ctx-unknown")
            with ctx_label:
                ctx_tip = ui.tooltip("")
            ctx_label.set_visibility(False)
            ui.space()
            copy_all_button = ui.button(
                "Copy all", icon="content_copy", on_click=lambda: copy_all()
            ).props("flat dense no-caps")
            copy_all_button.tooltip("Copy the whole conversation as Markdown")
            clear_button = ui.button("Clear all", icon="clear_all", on_click=lambda: clear())
            clear_button.props("flat dense no-caps")
            clear_button.tooltip("Remove every message (nothing is saved)")
        with ui.element("div").classes("sfc-wrap"):
            window = ui.element("div").classes("sfc-window")
            ui.button("Latest", icon="arrow_downward").props(
                "unelevated dense rounded no-caps size=sm color=primary"
            ).classes("sfc-latest").on("click", js_handler="() => window.sfChat.bottom()")

        # --- composer ------------------------------------------------------
        thumbs = ui.row().classes("gap-2 flex-wrap sfc-none")
        with ui.row().classes("w-full items-end gap-2 flex-wrap sfc-none"):
            # A hidden file input read in the browser and handed over through
            # the same event as a paste, so attaching costs no row of its own.
            file_input = ui.element("input").props('type=file accept="image/*" multiple')
            file_input.classes("sfc-file hidden")
            attach_button = ui.button(icon="attach_file").props("flat dense round")
            attach_button.on(
                "click",
                js_handler=f"() => document.getElementById('{file_input.html_id}').click()",
            )
            attach_button.tooltip("Attach an image (or paste one anywhere on the page)")
            prompt = ui.textarea(
                placeholder="Message… (Enter to send, Shift+Enter for a new line, "
                "↑ to edit your last message)"
            )
            prompt.props("dense outlined autogrow").classes("grow min-w-[14rem] sfc-composer")
            with ui.row().classes("items-center gap-1 no-wrap"):
                quick_button = ui.button(icon="bolt").props("flat dense round")
                quick_button.tooltip("Quick tests")
                with quick_button, ui.menu():
                    for test in st.CHAT_QUICK_TESTS:
                        item = ui.menu_item(
                            test.label,
                            on_click=lambda _event=None, key=test.key: send_quick(key),
                        )
                        item.tooltip(test.tooltip)
                send_button = ui.button("Send", icon="send").props("color=primary no-caps")
                stop_button = ui.button("Stop", icon="stop").props("flat no-caps")
                stop_button.tooltip("Stop the reply (Esc)")

        with (
            ui.expansion("Request settings", icon="tune")
            .props("dense")
            .classes("w-full sfc-none sfc-settings"),
            ui.column().classes("w-full gap-2"),
        ):
            system = (
                ui.textarea("System prompt", value="You are a helpful assistant.")
                .props("dense outlined autogrow")
                .classes("w-full")
            )
            sampler_inputs: dict[str, Any] = {}
            with ui.element("div").classes("w-full sfc-sampler-grid"):
                for spec in st.CHAT_SAMPLER_FIELDS:
                    field: Any
                    if spec.kind == "text":
                        field = ui.textarea(spec.label).props("dense outlined autogrow")
                    else:
                        field = ui.number(
                            spec.label,
                            value=spec.default if isinstance(spec.default, int | float) else None,
                            step=spec.step,
                            min=spec.minimum,
                            max=spec.maximum,
                        )
                        field.props("dense outlined clearable")
                    field.props("stack-label").tooltip(spec.tooltip)
                    sampler_inputs[spec.key] = field
                thinking = ui.select(
                    dict(st.CHAT_THINKING_CHOICES), value="auto", label="Thinking"
                ).props("dense outlined options-dense")
                thinking.tooltip(st.CHAT_THINKING_TOOLTIP)
                thinking.set_visibility(False)
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                keep_history = ui.switch("Send the conversation so far", value=True)
                keep_history.props("dense")
                ui.label(
                    "Blank = the model's own recommendation (greyed out), then the "
                    "engine's default. Requests go straight to the model's own "
                    "llama-server at the chat tier (1), exactly as a client's would "
                    "after the gateway."
                ).classes("text-xs opacity-60")

    def show_empty_hint() -> None:
        with window:
            view["hint"] = ui.label(_EMPTY_HINT).classes("text-sm opacity-60")

    def drop_empty_hint() -> None:
        hint = view.pop("hint", None)
        if hint is not None and element_alive(hint):
            hint.delete()

    def follow_bottom() -> None:
        ui.run_javascript("window.sfChat && window.sfChat.bottom()")

    show_empty_hint()

    # --- message blocks -------------------------------------------------------

    def body_text(message: ChatMessage) -> str:
        return shown.get(message.id, message.content)

    def add_block(message: ChatMessage) -> SimpleNamespace:
        """Paint one message block at the end of the window and remember it."""
        user = message.role == "user"
        block = SimpleNamespace(id=message.id, editing=None, fold=ThinkingFold(), expected=False)
        with window:
            block.root = ui.column().classes(
                "sfc-msg gap-1 " + ("sfc-user" if user else "sfc-assistant")
            )
        with block.root:
            with ui.element("div").classes("sfc-head w-full"):
                ui.label("You" if user else (message.model or "assistant")).classes("sfc-who")
                ui.label(_clock(message.created_at))
                block.edited = ui.label("(edited)")
                block.edited.set_visibility(message.edited)
                block.status = ui.label(message.status if message.status != "streaming" else "")
                with ui.element("div").classes("sfc-actions"):
                    _action(
                        "content_copy", "Copy raw (Markdown source)", lambda: copy_raw(block.id)
                    )
                    _action("content_paste_go", "Copy formatted", lambda: copy_formatted(block.id))
                    block.edit_btn = _action("edit", "Edit", lambda: open_editor(block.id))
                    block.regen_btn = _action(
                        "replay" if user else "refresh",
                        "Retry: send this message again" if user else "Regenerate this reply",
                        lambda: regenerate(block.id),
                    )
                    block.delete_btn = _action("delete", "Delete", lambda: delete(block.id))
            if user:
                block.think = block.think_body = None
            else:
                block.think = (
                    ui.expansion("Thinking", icon="psychology")
                    .props("dense")
                    .classes("w-full sfc-think")
                )
                with block.think:
                    block.think_body = ui.label("").classes("sfc-think-body")
                block.think.set_visibility(False)
                block.think.on_value_change(lambda event: on_fold_toggle(block, event.value))
            block.body = ui.html(render_markdown(body_text(message)), sanitize=False).classes(
                "ui-markdown sfc-body w-full"
            )
            block.editor_slot = ui.column().classes("w-full gap-1")
            block.error = ui.label("").classes("sfc-error")
            block.error.set_visibility(False)
            if user and message.images:
                with ui.row().classes("gap-2 flex-wrap"):
                    for url in message.images:
                        ui.image(url).classes("w-16 h-16 object-cover rounded")
            block.metrics = None if user else ui.column().classes("w-full gap-1")
        blocks[message.id] = block
        return block

    def remove_block(message_id: str) -> None:
        block = blocks.pop(message_id, None)
        shown.pop(message_id, None)
        if block is not None and element_alive(block.root):
            block.root.delete()

    def repaint_body(message: ChatMessage) -> None:
        block = blocks.get(message.id)
        if block is None or not element_alive(block.body):
            return
        block.body.set_content(render_markdown(body_text(message)))
        block.edited.set_visibility(message.edited)

    def refresh_actions() -> None:
        """Enable/disable per-block actions; only blocks whose state changed are touched."""
        active = bool(run["active"])
        for message_id, block in blocks.items():
            wanted = (active, conversation.can_regenerate(message_id))
            if getattr(block, "action_state", None) == wanted:
                continue
            block.action_state = wanted
            for control in (block.edit_btn, block.delete_btn, block.regen_btn):
                control.set_enabled(not active)
            block.regen_btn.set_visibility(wanted[1])
        count = len(conversation)
        state = (active, count)
        if view["actions"] != state:
            view["actions"] = state
            count_label.set_text(f"· {count} message{'' if count == 1 else 's'}" if count else "")
            clear_button.set_enabled(not active and count > 0)
            copy_all_button.set_enabled(count > 0)
        refresh_context()

    def sampler_values() -> dict[str, Any]:
        return {key: field.value for key, field in sampler_inputs.items()}

    def refresh_context() -> None:
        """The header's context readout, from the latest answered reply."""
        latest = next(
            (
                m
                for m in reversed(conversation.messages)
                if m.role == "assistant" and not m.failed and m.metrics is not None
            ),
            None,
        )
        usage = None
        if latest is not None:
            max_tokens = st.build_sampler_payload(sampler_values()).get("max_tokens")
            usage = st.chat_context_usage(
                latest.metrics, limits.get(latest.id), max_tokens=max_tokens
            )
        painted = (usage.text, usage.level, usage.tooltip) if usage is not None else None
        if painted == view.get("ctx"):
            return
        view["ctx"] = painted
        ctx_label.set_visibility(usage is not None)
        if usage is None:
            return
        ctx_label.set_text(usage.text)
        ctx_label.classes(remove=" ".join(_CTX_LEVEL_CLASSES), add=f"sfc-ctx-{usage.level}")
        ctx_tip.set_text(usage.tooltip)

    def toggle_details() -> None:
        opened = not details.visible
        details.set_visibility(opened)
        details_button.props(f"icon={'expand_less' if opened else 'expand_more'}")

    # --- thinking fold ----------------------------------------------------------

    def on_fold_toggle(block: SimpleNamespace, value: bool) -> None:
        # A change we did not ask for is the reader's: stop auto-toggling it.
        if bool(value) != block.expected:
            block.fold.manual = True

    def paint_reply(block: SimpleNamespace, message: ChatMessage) -> Callable[..., None]:
        """The stream's painter for one reply block: thinking fold + answer."""

        def paint(content: str, reasoning_stream: str, final: bool) -> None:
            inline_reasoning, answer = st.split_reasoning(content)
            reasoning = "\n\n".join(part for part in (reasoning_stream, inline_reasoning) if part)
            message.content = answer
            message.reasoning = reasoning
            if not element_alive(block.body):
                return
            if reasoning and block.think is not None:
                header, want_open = block.fold.step(
                    len(reasoning), bool(answer.strip()), final, time.perf_counter()
                )
                block.think.set_visibility(True)
                block.think.set_text(header)
                block.think_body.set_text(reasoning)
                if want_open is not None and bool(block.think.value) != want_open:
                    block.expected = want_open
                    block.think.set_value(want_open)
            block.body.set_content(render_markdown(answer))

        return paint

    # --- target resolution and painting -------------------------------------

    def pick_now(records: list[Any] | None = None, instances: list[Any] | None = None) -> Any:
        return st.chat_pick(
            records if records is not None else records_now(),
            instances if instances is not None else instances_now(),
            choice=model.value,
            unloadable=unloadable,
        )

    def serving_instance(record: Any, instances: list[Any]) -> Any:
        if record is None:
            return None
        wanted = [record.id]
        if record.is_virtual and record.base_model_id:
            wanted.append(record.base_model_id)
        for model_id in wanted:
            found = next((i for i in instances if i.model_id == model_id), None)
            if found is not None:
                return found
        return None

    def gpus_now() -> list[Any]:
        try:
            return list(ctx.probe.list_gpus()) if ctx.probe is not None else []
        except Exception:  # noqa: BLE001 - facts degrade to device numbers only
            return []

    def paint_facts(record: Any, instance: Any) -> None:
        facts_row.clear()
        with facts_row:
            for label, value in st.chat_target_facts(record, instance, gpus_now()):
                with ui.column().classes("gap-0"):
                    ui.label(label).classes("text-[11px] uppercase tracking-wide opacity-60")
                    ui.label(value).classes("text-sm font-mono")

    def sync() -> None:
        """Repaint the target card from live state; runs on the GUI's poll cadence."""
        if not element_alive(model):
            return
        records = records_now()
        instances = instances_now()
        options = st.chat_picker_options(records, instances, unloadable=unloadable)
        value = model.value if model.value in options else st.LOADED_MODEL_CHOICE
        if options != view["options"]:
            view["options"] = options
            model.set_options(options, value=value)
        elif value != model.value:
            model.set_value(value)

        pick = st.chat_pick(records, instances, choice=value, unloadable=unloadable)
        record = next((r for r in records if r.id == pick.model_id), None)
        instance = serving_instance(record, instances)

        text, colour = _STATE_BADGES.get(pick.state, ("", "grey"))
        status_badge.set_text(text)
        status_badge.props(f"color={colour}")
        target_name.set_text(pick.model_id or "—")
        reason_label.set_text(pick.reason)

        # Facts carry relative times ("3 minutes ago"), so they are repainted
        # when anything they show changes, and at least once a minute.
        signature = (
            pick.model_id,
            pick.state,
            getattr(instance, "started_at", None),
            getattr(instance, "total_requests", None),
            getattr(instance, "last_tokens_per_second", None),
            int(time.time() // 60),
        )
        if signature != view["card"]:
            view["card"] = signature
            paint_facts(record, instance)

        why_not = unloadable(record) if (unloadable and record is not None) else None
        warning = why_not or ""
        if not warning and pick.state == "failed" and instance is not None:
            warning = f"Last start failed: {(instance.last_error or 'no detail')[:400]}"
        warn_label.set_text(warning)
        warn_label.set_visibility(bool(warning))

        if pick.follows_loaded and pick.other_loaded:
            others_label.set_text(
                f"Also loaded: {', '.join(pick.other_loaded)}. (Loaded model) follows the "
                "one loaded most recently; pick one below to pin the choice."
            )
        elif not pick.follows_loaded and pick.loaded_id and pick.loaded_id != pick.model_id:
            others_label.set_text(f"Currently loaded: {pick.loaded_id}")
        else:
            others_label.set_text("")
        others_label.set_visibility(bool(others_label.text))

        hidden = st.hidden_chat_models_note(records) or ""
        hidden_label.set_text(hidden)
        hidden_label.set_visibility(bool(hidden))

        sync_settings(record, records)
        sync_controls(pick, record, why_not)

    def sync_settings(record: Any, records: list[Any]) -> None:
        """Placeholders say what a blank setting will be for *this* model."""
        base = None
        if record is not None and record.is_virtual and record.base_model_id:
            base = next((r for r in records if r.id == record.base_model_id), None)
        recommended = st.recommended_sampling(record, base=base)
        signature = (
            tuple(sorted(recommended.items())),
            st.thinking_toggle_supported(record),
        )
        if signature == view.get("settings"):
            return
        view["settings"] = signature
        for spec in st.CHAT_SAMPLER_FIELDS:
            text = st.sampler_placeholder(spec, recommended).replace('"', "'")
            sampler_inputs[spec.key].props(f'placeholder="{text}"')
        thinking.set_visibility(signature[1])
        if not signature[1]:
            thinking.set_value("auto")

    def sync_controls(pick: Any, record: Any, why_not: str | None) -> None:
        active = bool(run["active"])
        can_target = pick.model_id is not None and not why_not
        load_button.set_visibility(pick.state != "ready")
        unload_button.set_visibility(pick.state == "ready")
        if can_target and not active and pick.state in ("not_loaded", "failed"):
            load_button.enable()
            load_tip.set_text("Load it now at the chat tier, without sending anything.")
        else:
            load_button.disable()
            load_tip.set_text(
                why_not
                or ("A reply is in progress." if active else "")
                or ("It is loading now." if pick.state == "loading" else "Nothing to load.")
            )
        if active:
            unload_button.disable()
        else:
            unload_button.enable()
        for control in (send_button, quick_button):
            if can_target and not active:
                control.enable()
            else:
                control.disable()
        if active:
            stop_button.enable()
        else:
            stop_button.disable()
        sync_attach_state(record)
        refresh_actions()

    # --- image attachment ------------------------------------------------------

    def sync_attach_state(record: Any) -> None:
        vision = st.vision_attach_reason(record) is None
        if vision != view.get("vision"):
            view["vision"] = vision
            attach_button.set_visibility(vision)
        if not vision and images:
            images.clear()
            _render_thumbs(thumbs, images)

    def target_record() -> Any:
        pick = pick_now()
        return next((r for r in records_now() if r.id == pick.model_id), None)

    def add_image(data_url: str, name: str) -> None:
        if st.vision_attach_reason(target_record()) is not None:
            ui.notify("this model cannot accept images", type="warning")
            return
        if len(data_url) > MAX_IMAGE_BYTES * 2:  # base64 is ~4/3 of the bytes
            ui.notify("image too large", type="negative")
            return
        images.append({"name": name, "url": data_url})
        _render_thumbs(thumbs, images)

    def on_paste(event: Any) -> None:
        payload = event.args
        if isinstance(payload, list):
            payload = payload[0] if payload else {}
        if not isinstance(payload, dict):
            return
        data = str(payload.get("data") or "")
        if data.startswith("data:image/"):
            add_image(data, str(payload.get("name") or "pasted-image"))

    ui.on("sf_paste_image", on_paste)
    model.on_value_change(lambda _: sync())
    sync()
    # Same cadence as the rest of the panel: "(Loaded model)" must follow loads
    # and unloads made anywhere else, including by other clients.
    ui.timer(ctx.refresh_interval, sync)

    # --- load / unload ---------------------------------------------------------

    async def load_target() -> None:
        pick = pick_now()
        if pick.model_id is None:
            return
        with single_flight(f"chat.load:{pick.model_id}", f"load of {pick.model_id}") as claimed:
            if not claimed:
                return
            started = time.perf_counter()
            with busy(load_button, message=f"Loading {pick.model_id}…"):
                try:
                    await _ensure(ctx, pick.model_id)
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="load")
                    sync()
                    return
            elapsed = st.format_latency(time.perf_counter() - started)
            ui.notify(f"{pick.model_id} loaded in {elapsed}", type="positive")
            sync()

    async def unload_target() -> None:
        pick = pick_now()
        records = records_now()
        record = next((r for r in records if r.id == pick.model_id), None)
        instance = serving_instance(record, instances_now())
        if instance is None:
            return
        serving_id = instance.model_id
        with single_flight(f"models.unload:{serving_id}", f"unload of {serving_id}") as claimed:
            if not claimed:
                return
            with busy(unload_button, message=f"Unloading {serving_id}…"):
                try:
                    # D55: a viewer who passes D32 may unload a lease-held
                    # model; a remote viewer on an open install gets the
                    # manager's 409 as a red toast, as on the Dashboard.
                    await ctx.manager.unload(
                        serving_id, force=viewer_may_change_box(ctx), source="gui:chat"
                    )
                except Exception as exc:  # noqa: BLE001
                    notify_error(exc, what="unload")
                    sync()
                    return
            ui.notify(f"{serving_id} unloaded", type="positive")
            sync()

    load_button.on_click(load_target)
    unload_button.on_click(unload_target)

    # --- sending ---------------------------------------------------------------

    def resolve_target() -> tuple[str, bool, Any] | None:
        """The model the next reply goes to, whether it is loaded, and its record."""
        records = records_now()
        instances = instances_now()
        pick = pick_now(records, instances)
        if not pick.model_id:
            ui.notify("no model to send to", type="warning")
            return None
        record = next((r for r in records if r.id == pick.model_id), None)
        was_ready = record is not None and st.chat_model_state(record, instances) == "ready"
        return pick.model_id, was_ready, record

    async def send(text: str | None = None, display: str | None = None) -> None:
        if run["active"]:
            return
        typed = text is None
        message = str(prompt.value or "").strip() if typed else str(text)
        if not message and not images:
            return
        target = resolve_target()
        if target is None:
            return
        drop_empty_hint()
        user = conversation.add_user(message, [image["url"] for image in images])
        if display is not None and display != message:
            shown[user.id] = display
        add_block(user)
        if typed:
            prompt.set_value("")
        images.clear()
        _render_thumbs(thumbs, images)
        await reply_to(user.id, target)

    async def reply_to(user_id: str, target: tuple[str, bool, Any] | None = None) -> None:
        """Stream a fresh reply to the user turn ``user_id`` into a new block."""
        # Regenerate / "Save & resend" arrive from a button inside a block they
        # just deleted; NiceGUI resolves notify/run_javascript through the
        # caller's slot, so re-anchor on the window, which outlives every block.
        with window:
            await _reply_to(user_id, target)

    async def _reply_to(user_id: str, target: tuple[str, bool, Any] | None) -> None:
        target = target or resolve_target()
        if target is None:
            refresh_actions()
            return
        model_id, was_ready, record = target
        system_text = str(system.value or "").strip()
        messages = conversation.request_messages(
            system=system_text or None, keep_history=bool(keep_history.value), upto=user_id
        )
        reply = conversation.add_assistant(model_id)
        block = add_block(reply)
        follow_bottom()

        payload: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "stream": True,
            # usage + llama-server's own timings ride on the final chunk.
            "stream_options": {"include_usage": True},
            # prompt_progress frames: a prefill readout, and bytes on the wire
            # during a long prefill.
            "return_progress": True,
        }
        # Blank settings are left out so the model's recommendation applies; an
        # explicit 0 (greedy temperature) is sent as 0.
        payload.update(st.build_sampler_payload({**sampler_values(), "thinking": thinking.value}))
        # The gateway folds a persona preset in; this tab talks to the child
        # directly, so it must do the same or a persona would chat as its base.
        preset = getattr(record, "preset", None)
        if preset is not None:
            preset.apply_to_payload(payload, chat=True)

        def status(text: str) -> None:
            if element_alive(block.status):
                block.status.set_text(text)

        run["active"] = True
        run["stop"] = False
        sync()
        clicked_at = time.perf_counter()
        load_s: float | None = None
        try:
            ticker = None
            if not was_ready:
                ticker = asyncio.create_task(_tick_loading(block.status, clicked_at))
            try:
                _record, instance = await _ensure(ctx, model_id)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - clicked_at
                fail(reply, block, f"load failed after {st.format_latency(elapsed)}", exc)
                notify_error(exc, what="chat")
                return
            finally:
                if ticker is not None:
                    ticker.cancel()
            if not was_ready:
                load_s = time.perf_counter() - clicked_at
            limit = st.chat_context_limit(instance)
            limits[reply.id] = limit
            serving_id = instance.model_id
            base = ctx.supervisor.base_url(serving_id)
            if base is None:
                fail(reply, block, "not serving", RuntimeError(f"'{serving_id}' has no port"))
                return
            payload["model"] = serving_id
            status("waiting for the first token…")
            # A task, so Stop can cancel it even while the engine is still
            # reading the prompt and nothing arrives to check the flag against.
            task = asyncio.create_task(
                _stream(
                    ctx,
                    serving_id,
                    base,
                    payload,
                    paint_reply(block, reply),
                    run,
                    on_first=lambda: status("streaming…"),
                    on_progress=status,
                    n_ctx=limit,
                )
            )
            run["task"] = task
            try:
                result = await task
            finally:
                run.pop("task", None)
            metrics = st.chat_run_metrics(
                clicked_at=clicked_at,
                sent_at=result.sent_at,
                first_token_at=result.first_token_at,
                last_token_at=result.last_token_at or time.perf_counter(),
                load_s=load_s,
                chunks=result.chunks,
                usage=result.usage,
                timings=result.timings,
                finish_reason=result.finish_reason,
                stopped=result.stopped,
            )
            reply.metrics = metrics
            # A stopped reply keeps its partial text as ordinary context.
            reply.status = "stopped" if result.stopped else ""
            status(reply.status)
            if block.metrics is not None and element_alive(block.metrics):
                _render_metrics(block.metrics, metrics)
        except ChatRequestError as exc:
            fail(reply, block, "context full" if exc.context_full else "failed", exc)
        except Exception as exc:  # noqa: BLE001
            fail(reply, block, "failed", exc)
            notify_error(exc, what="chat")
        finally:
            run["active"] = False
            run["stop"] = False
            sync()

    def fail(message: ChatMessage, block: SimpleNamespace, what: str, exc: BaseException) -> None:
        # A failed reply is never context: the next request (or a Retry on the
        # user turn above) behaves as if it had not been answered.
        message.failed = True
        message.status = what
        if not element_alive(block.root):
            return
        block.status.set_text(what)
        block.root.classes(add="sfc-failed")
        # The engine's refusal is already a plain sentence (with what to do
        # about a full context); anything else keeps the "[what: detail]" form.
        text = str(exc) if isinstance(exc, ChatRequestError) else f"[{what}: {exc}]"
        block.error.set_text(text)
        block.error.set_visibility(True)

    async def send_quick(key: str) -> None:
        test = next(t for t in st.CHAT_QUICK_TESTS if t.key == key)
        await send(st.quick_test_prompt(key), test.display)

    def request_stop() -> None:
        if run["active"]:
            run["stop"] = True
            task = run.get("task")
            if task is not None and not task.done():
                task.cancel()

    # --- per-message actions -----------------------------------------------------

    def copy_raw(message_id: str) -> None:
        message = conversation.get(message_id)
        if message is not None:
            _copy_text(message.content)

    def copy_formatted(message_id: str) -> None:
        message = conversation.get(message_id)
        block = blocks.get(message_id)
        if message is None or block is None:
            return
        ui.run_javascript(
            f"window.sfChat.copyHtml({json.dumps(block.body.html_id)}, "
            f"{json.dumps(plain_text(message.content))})"
        )

    def copy_all() -> None:
        if len(conversation):
            _copy_text(conversation.as_markdown())

    def blocked() -> bool:
        if run["active"]:
            ui.notify(_STOP_FIRST, type="warning")
            return True
        return False

    def delete(message_id: str) -> None:
        if blocked():
            return
        conversation.delete(message_id)
        remove_block(message_id)
        if not len(conversation):
            show_empty_hint()
        refresh_actions()

    async def regenerate(message_id: str) -> None:
        if blocked():
            return
        point = conversation.regenerate_point(message_id)
        if point is None:
            return
        for dropped in conversation.truncate_after(point.id):
            remove_block(dropped.id)
        await reply_to(point.id)

    def clear() -> None:
        if blocked():
            return
        conversation.clear()
        blocks.clear()
        shown.clear()
        window.clear()
        view.pop("hint", None)
        show_empty_hint()
        refresh_actions()

    # --- inline editor -----------------------------------------------------------

    def close_editor(block: SimpleNamespace) -> None:
        block.editing = None
        if element_alive(block.editor_slot):
            block.editor_slot.clear()
            block.body.set_visibility(True)

    def open_editor(message_id: str) -> None:
        if blocked():
            return
        message = conversation.get(message_id)
        block = blocks.get(message_id)
        if message is None or block is None:
            return
        if block.editing is not None:
            block.editing.run_method("focus")
            return
        user = message.role == "user"
        block.body.set_visibility(False)
        with block.editor_slot:
            editor = ui.textarea(value=message.content).props("dense outlined autogrow autofocus")
            editor.classes("w-full sfc-editor")
            with ui.row().classes("gap-2 items-center"):
                if user:
                    ui.button(
                        "Save & resend", icon="send", on_click=lambda: save(block, resend=True)
                    ).props("color=primary dense no-caps")
                    ui.button("Save", on_click=lambda: save(block, resend=False)).props(
                        "flat dense no-caps"
                    )
                else:
                    ui.button("Save", on_click=lambda: save(block, resend=False)).props(
                        "color=primary dense no-caps"
                    )
                ui.button("Cancel", on_click=lambda: close_editor(block)).props(
                    "flat dense no-caps"
                )
                ui.label(
                    "Ctrl+Enter to " + ("save & resend" if user else "save") + " · Esc to cancel"
                ).classes("text-xs opacity-60")
        block.editing = editor
        editor.on("keydown.ctrl.enter", lambda: save(block, resend=user))
        editor.on("keydown.esc", lambda: close_editor(block))

    async def save(block: SimpleNamespace, *, resend: bool) -> None:
        editor = block.editing
        message = conversation.get(block.id)
        if editor is None or message is None:
            return
        if resend and blocked():
            return
        text = str(editor.value or "")
        if conversation.edit(block.id, text):
            shown.pop(block.id, None)
        close_editor(block)
        repaint_body(message)
        if resend:
            for dropped in conversation.truncate_after(block.id):
                remove_block(dropped.id)
            await reply_to(block.id)
        else:
            refresh_actions()

    def edit_last_user() -> None:
        if str(prompt.value or "") or run["active"]:
            return
        last = conversation.last_user()
        if last is not None:
            open_editor(last.id)

    send_button.on_click(lambda: send())
    stop_button.on_click(request_stop)
    # Enter sends; Shift+Enter falls through to the browser's default and
    # inserts the newline (``exact`` keeps modifier combinations out, and
    # ``prevent`` stops the sent message from also gaining a newline).
    # Ctrl+Enter stays as an alias for muscle memory from the old binding.
    prompt.on("keydown.enter.exact.prevent", lambda: send())
    prompt.on("keydown.ctrl.enter", lambda: send())
    prompt.on("keydown.esc", request_stop)
    prompt.on("keydown.up.exact", edit_last_user)
    refresh_actions()


@dataclass
class ThinkingFold:
    """When a reply's Thinking fold opens and closes by itself, and what it says.

    It opens on the first reasoning token and collapses once thinking ends --
    the first answer token, or the end of the stream -- so the latest thought
    is visible while the model thinks without a long monologue pushing the
    answer down afterwards. Once the reader toggles it by hand (``manual``) it
    is theirs: :meth:`step` stops asking for a change.
    """

    started: float | None = None
    ended: float | None = None
    manual: bool = False

    def step(self, chars: int, answering: bool, final: bool, now: float) -> tuple[str, bool | None]:
        """``(header, want_open)``; ``want_open`` is ``None`` to leave it as it is."""
        if self.started is None:
            self.started = now
        if self.ended is None and (answering or final):
            self.ended = now
        if self.ended is None:
            header = f"Thinking… ({chars:,} chars, {_seconds(now - self.started)})"
            want: bool | None = True
        else:
            header = f"Thought for {_seconds(self.ended - self.started)} ({chars:,} chars)"
            want = False
        return header, None if self.manual else want


def _seconds(value: float) -> str:
    value = max(0.0, value)
    if value < 10:
        return f"{value:.1f}s"
    if value < 60:
        return f"{value:.0f}s"
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}m {seconds:02d}s"


def _clock(epoch: float) -> str:
    return time.strftime("%H:%M", time.localtime(epoch))


def _action(icon: str, tip: str, handler: Callable[..., Any]) -> Any:
    button = ui.button(icon=icon, on_click=handler).props("flat dense round size=sm")
    button.tooltip(tip)
    button.props(f'aria-label="{tip}"')
    return button


def _copy_text(text: str) -> None:
    """Copy through the page's helper, which falls back off secure contexts."""
    ui.run_javascript(f"window.sfChat.copyText({json.dumps(text)})")


def _unloadable_check(ctx: GuiContext) -> Callable[[Any], str | None] | None:
    """Why a model cannot be loaded at all, when the server can say (D66).

    Used to keep ``(Loaded model)`` from defaulting to a download the engine
    cannot run, and to explain a disabled Load button instead of letting it
    fail. ``None`` when this build has no such check.
    """
    reason_for = getattr(ctx.manager, "unsupported_reason", None)
    if not callable(reason_for):
        return None

    def check(record: Any) -> str | None:
        try:
            reason = reason_for(record)
        except Exception:  # noqa: BLE001 - a failed check must never block a load
            return None
        return str(reason) if reason else None

    return check


async def _ensure(ctx: GuiContext, model_id: str) -> tuple[Any, Any]:
    """Load (or find) the model the way a client's first request would.

    Tier 1, literally: a person typing in this tab *is* the active chat, which
    is D46's own definition of the tier. Claiming nothing left the server's own
    UI as background work -- an agent's tier-2 load could 503 the human sitting
    in front of it. The number is spelled out rather than imported because this
    package imports nothing from ``core``; the tiers live in
    ``studioforge/core/priority.py`` (PRIORITY_CHAT).
    """
    record, instance = await ctx.manager.ensure_loaded(model_id, priority=1, source="gui:chat")
    return record, instance


async def _tick_loading(label: Any, started: float) -> None:
    """Count the seconds of a cold load in the reply's status line."""
    while True:
        if element_alive(label):
            label.set_text(f"loading the model… {time.perf_counter() - started:.0f} s")
        await asyncio.sleep(0.5)


def _render_thumbs(container: Any, images: list[dict[str, str]]) -> None:
    container.clear()
    with container:
        for image in images:
            with ui.column().classes("gap-0 items-center"):
                ui.image(image["url"]).classes("w-16 h-16 object-cover rounded")
                ui.label(image["name"][:16]).classes("text-[11px] opacity-60")


def _render_metrics(container: Any, metrics: Any) -> None:
    container.clear()
    with container:
        with ui.row().classes("w-full gap-x-5 gap-y-1 flex-wrap"):
            for tile in st.chat_metric_tiles(metrics):
                with ui.column().classes("gap-0 min-w-[5.5rem]") as column:
                    ui.label(tile.label).classes("text-[10px] uppercase tracking-wide opacity-60")
                    ui.label(tile.value).classes("text-sm font-mono")
                    ui.label(tile.detail).classes("text-[10px] opacity-70")
                column.tooltip(tile.tooltip)
        footer = st.chat_metric_footer(metrics)
        if footer:
            ui.label(footer).classes("text-xs opacity-70")


async def _stream(
    ctx: GuiContext,
    serving_id: str,
    base: str,
    payload: dict[str, Any],
    paint: Callable[[str, str, bool], None],
    run: dict[str, Any],
    *,
    on_first: Callable[[], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    n_ctx: int | None = None,
) -> SimpleNamespace:
    """Stream a completion from the model's own llama-server child.

    The base URL comes from the supervisor, so it is always the loopback port of
    the child we started -- there is no configured or guessed URL anywhere, which
    is what keeps this working behind any proxy.

    ``paint(content, reasoning, final)`` is called at most every
    ``_REPAINT_S`` while tokens arrive and exactly once more, with
    ``final=True``, when the stream ends for any reason (done, stopped, failed).

    Stop is honoured two ways: the ``run["stop"]`` flag, checked per frame, and
    a cancel of the task running this coroutine -- the only thing that reaches a
    request still in prefill. A cancel that follows a Stop ends the reply as
    stopped, partial text kept; any other cancel propagates. Engine refusals
    (an HTTP error, or a ``data: {"error": ...}`` frame after the 200) raise
    :class:`ChatRequestError` with a readable line; ``n_ctx`` lets a context
    overflow name the window when the engine does not.
    """
    result = SimpleNamespace(
        content="",
        reasoning="",
        chunks=0,
        sent_at=None,
        first_token_at=None,
        last_token_at=None,
        usage=None,
        timings=None,
        finish_reason=None,
        stopped=False,
    )
    content: list[str] = []
    reasoning: list[str] = []
    painted_at = 0.0
    request_id = ctx.supervisor.mark_request_start(serving_id, client="gui:chat")
    try:
        # Loopback plain HTTP: skipping the TLS context (certifi load) and proxy
        # discovery keeps ~0.15 s of client setup out of the measured total.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=10.0), verify=False, trust_env=False
        ) as client:
            result.sent_at = time.perf_counter()
            async with client.stream(
                "POST", f"{base}/v1/chat/completions", json=payload
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")
                    raise _request_error(
                        response.status_code, body, n_ctx=n_ctx, model_id=serving_id
                    )
                async for line in response.aiter_lines():
                    if run.get("stop"):
                        result.stopped = True
                        break
                    data = _parse_sse(line)
                    if data is None:
                        continue
                    if st.stream_error(data):
                        raise _request_error(None, data, n_ctx=n_ctx, model_id=serving_id)
                    progress = st.prefill_progress(data)
                    if progress is not None and on_progress is not None:
                        on_progress(progress.text)
                    if isinstance(data.get("usage"), dict):
                        result.usage = data["usage"]
                    if isinstance(data.get("timings"), dict):
                        result.timings = data["timings"]
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        result.finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or ""
                    if not piece and not thought:
                        continue
                    now = time.perf_counter()
                    if result.first_token_at is None:
                        result.first_token_at = now
                        if on_first is not None:
                            on_first()
                    result.last_token_at = now
                    result.chunks += 1
                    if piece:
                        content.append(piece)
                    if thought:
                        reasoning.append(thought)
                    if now - painted_at >= _REPAINT_S:
                        painted_at = now
                        paint("".join(content), "".join(reasoning), False)
    except asyncio.CancelledError:
        if not run.get("stop"):
            raise
        # Our own Stop: finish as a normal stopped reply.
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
        result.stopped = True
    finally:
        result.content = "".join(content)
        result.reasoning = "".join(reasoning)
        paint(result.content, result.reasoning, True)
        elapsed = (result.last_token_at or time.perf_counter()) - (
            result.first_token_at or result.sent_at or time.perf_counter()
        )
        rate = None
        if isinstance(result.timings, dict):
            rate = result.timings.get("predicted_per_second")
        ctx.supervisor.mark_request_end(
            serving_id,
            request_id=request_id,
            tokens_per_second=round(float(rate), 2)
            if isinstance(rate, int | float) and rate > 0
            else st.tokens_per_second(max(0, result.chunks - 1), elapsed),
        )
    return result


def _parse_sse(line: str) -> dict[str, Any] | None:
    """One ``data:`` line of an OpenAI stream as a dict, or ``None``."""
    if not line.startswith("data:"):
        return None
    chunk = line[5:].strip()
    if not chunk or chunk == "[DONE]":
        return None
    try:
        data = json.loads(chunk)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
