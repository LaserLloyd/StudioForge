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
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import httpx
from nicegui import ui

from studioforge.gui import state as st
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

#: Repaint a streaming reply at most this often. A 0.5B model streams ~700
#: tokens/s, and one websocket message per token only makes the browser lag.
_REPAINT_S = 0.05

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


def render(ctx: GuiContext) -> None:  # noqa: C901, PLR0915 - one screen, one flow
    images: list[dict[str, str]] = []
    history: list[dict[str, Any]] = []
    #: ``active`` while a send is in flight; ``stop`` is the Stop button's request.
    run: dict[str, Any] = {"active": False, "stop": False}
    #: Last painted picker options and card signature, so a poll that changes
    #: nothing sends nothing to the browser (and never closes an open dropdown).
    view: dict[str, Any] = {"options": None, "card": None}
    unloadable = _unloadable_check(ctx)

    ui.add_head_html(_PASTE_SCRIPT)

    def records_now() -> list[Any]:
        return list(ctx.registry.all()) if ctx.registry is not None else []

    def instances_now() -> list[Any]:
        return list(ctx.supervisor.list()) if ctx.supervisor is not None else []

    with ui.column().classes("w-full gap-3 p-2"):
        # --- what are we talking to? ---------------------------------------
        with ui.card().classes("w-full gap-2"):
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
                load_button = ui.button("Load", icon="play_arrow").props("outline no-caps")
                with load_button:
                    load_tip = ui.tooltip("")
                unload_button = ui.button("Unload", icon="stop_circle").props("flat no-caps")
            with ui.row().classes("w-full items-center gap-2 flex-wrap"):
                status_badge = ui.badge("", color="grey").classes("text-xs")
                target_name = ui.label("").classes("text-sm font-mono break-all")
            reason_label = ui.label("").classes("text-xs opacity-70")
            facts_row = ui.row().classes("w-full gap-x-6 gap-y-2 flex-wrap")
            warn_label = ui.label("").classes("text-xs text-warning whitespace-pre-wrap")
            others_label = ui.label("").classes("text-xs opacity-70")
            hidden_label = ui.label("").classes("text-xs opacity-60")

        # --- the conversation ----------------------------------------------
        transcript = ui.column().classes("w-full gap-4 p-3 rounded sf-well min-h-[10rem]")

        # --- composer ------------------------------------------------------
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.label("Quick tests").classes("text-xs opacity-70")
            quick_buttons: list[Any] = []
            for test in st.CHAT_QUICK_TESTS:
                button = ui.button(
                    test.label,
                    on_click=lambda _event=None, key=test.key: send_quick(key),
                ).props("outline dense no-caps")
                button.tooltip(test.tooltip)
                quick_buttons.append(button)
        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            prompt = ui.textarea(placeholder="Message… (Enter to send, Shift+Enter for a new line)")
            prompt.props("dense outlined autogrow").classes("grow")
            send_button = ui.button("Send", icon="send").props("color=primary no-caps")
            stop_button = ui.button("Stop", icon="stop").props("flat no-caps")
            clear_button = ui.button("Clear", icon="clear_all").props("flat no-caps")
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            upload = (
                ui.upload(
                    label="Attach image",
                    auto_upload=True,
                    multiple=True,
                    max_file_size=MAX_IMAGE_BYTES,
                )
                .props('flat dense accept="image/*"')
                .classes("max-w-[14rem]")
            )
            attach_note = ui.label("").classes("text-xs opacity-70")
        thumbs = ui.row().classes("gap-2 flex-wrap")

        with (
            ui.expansion("Request settings", icon="tune").classes("w-full"),
            ui.column().classes("w-full gap-2"),
        ):
            system = (
                ui.textarea("System prompt", value="You are a helpful assistant.")
                .props("dense outlined autogrow")
                .classes("w-full")
            )
            with ui.row().classes("w-full items-center gap-3 flex-wrap"):
                temperature = ui.number("temperature", value=0.7, precision=2, step=0.05)
                temperature.props("dense outlined").classes("w-32")
                top_p = ui.number("top_p", value=0.95, precision=2, step=0.05)
                top_p.props("dense outlined").classes("w-32")
                max_tokens = ui.number("max_tokens", value=2048, precision=0)
                max_tokens.props("dense outlined").classes("w-32")
                keep_history = ui.switch("Send the conversation so far", value=True)
                keep_history.props("dense")
            ui.label(
                "Requests go straight to the model's own llama-server at the chat tier "
                "(1), exactly as a client's would after the gateway. Turn the "
                "conversation off to measure each prompt on its own."
            ).classes("text-xs opacity-60")

    def show_empty_hint() -> None:
        with transcript:
            view["hint"] = ui.label(
                "Send a message or pick a quick test. Every reply shows load time, time to "
                "first token, prefill and decode speed, and overall tokens per second."
            ).classes("text-sm opacity-60")

    def drop_empty_hint() -> None:
        hint = view.pop("hint", None)
        if hint is not None and element_alive(hint):
            hint.delete()

    show_empty_hint()

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

        sync_controls(pick, record, why_not)

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
        for control in (send_button, *quick_buttons):
            if can_target and not active:
                control.enable()
            else:
                control.disable()
        if active:
            stop_button.enable()
        else:
            stop_button.disable()
        sync_attach_state(record)

    # --- image attachment ------------------------------------------------------

    def sync_attach_state(record: Any) -> None:
        reason = st.vision_attach_reason(record)
        if reason:
            upload.set_visibility(False)
            attach_note.set_text(
                "Images: this model has no vision projector." if record is not None else ""
            )
            if images:
                images.clear()
                _render_thumbs(thumbs, images)
        else:
            upload.set_visibility(True)
            attach_note.set_text("Vision model: attach a file or paste an image into the page.")

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

    def on_upload(event: Any) -> None:
        content = event.content.read()
        mime = getattr(event, "type", None) or "image/png"
        encoded = base64.b64encode(content).decode("ascii")
        add_image(f"data:{mime};base64,{encoded}", getattr(event, "name", "upload"))
        upload.reset()

    upload.on_upload(on_upload)

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

    async def send(text: str | None = None, display: str | None = None) -> None:
        if run["active"]:
            return
        typed = text is None
        message = str(prompt.value or "").strip() if typed else str(text)
        if not message and not images:
            return
        records = records_now()
        instances = instances_now()
        pick = pick_now(records, instances)
        target = pick.model_id
        if not target:
            ui.notify("no model to send to", type="warning")
            return
        record = next((r for r in records if r.id == target), None)
        was_ready = record is not None and st.chat_model_state(record, instances) == "ready"

        attached = [image["url"] for image in images]
        user_turn = {"role": "user", "content": st.build_chat_content(message, attached)}
        drop_empty_hint()
        shown = display if display is not None else message
        with transcript:
            _bubble("you", shown + (f"\n[{len(attached)} image(s)]" if attached else ""))
            reply = _reply_block(target)
        if typed:
            prompt.set_value("")
        images.clear()
        _render_thumbs(thumbs, images)

        messages: list[dict[str, Any]] = []
        system_text = str(system.value or "").strip()
        if system_text:
            messages.append({"role": "system", "content": system_text})
        if keep_history.value:
            messages.extend(history)
        messages.append(user_turn)
        history.append(user_turn)

        payload: dict[str, Any] = {
            "model": target,
            "messages": messages,
            "stream": True,
            # usage + llama-server's own timings ride on the final chunk.
            "stream_options": {"include_usage": True},
            # number_value, not ``or``: an explicit 0 (greedy temperature) must
            # be sent as 0, never silently replaced with the default.
            "temperature": st.number_value(temperature.value, 0.7),
            "top_p": st.number_value(top_p.value, 0.95),
            "max_tokens": int(st.number_value(max_tokens.value, 2048)),
        }

        run["active"] = True
        run["stop"] = False
        sync()
        clicked_at = time.perf_counter()
        load_s: float | None = None
        try:
            ticker = None
            if not was_ready:
                ticker = asyncio.create_task(_tick_loading(reply.status, clicked_at))
            try:
                _record, instance = await _ensure(ctx, target)
            except Exception as exc:  # noqa: BLE001
                elapsed = time.perf_counter() - clicked_at
                _show_failure(reply, f"load failed after {st.format_latency(elapsed)}", exc)
                notify_error(exc, what="chat")
                history.pop()
                return
            finally:
                if ticker is not None:
                    ticker.cancel()
            if not was_ready:
                load_s = time.perf_counter() - clicked_at
            serving_id = instance.model_id
            base = ctx.supervisor.base_url(serving_id)
            if base is None:
                _show_failure(reply, "not serving", RuntimeError(f"'{serving_id}' has no port"))
                history.pop()
                return
            payload["model"] = serving_id
            if element_alive(reply.status):
                reply.status.set_text("waiting for the first token…")
            result = await _stream(ctx, serving_id, base, payload, reply, run)
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
            if element_alive(reply.status):
                reply.status.set_text("stopped" if result.stopped else "")
                _render_metrics(reply.metrics, metrics)
            _reasoning, answer = st.split_reasoning(result.content)
            history.append({"role": "assistant", "content": answer or result.content})
        except Exception as exc:  # noqa: BLE001
            _show_failure(reply, "failed", exc)
            notify_error(exc, what="chat")
            # A turn the model never answered must not ride along with the next
            # message, or the next answer replies to both.
            if history and history[-1] is user_turn:
                history.pop()
        finally:
            run["active"] = False
            run["stop"] = False
            sync()

    async def send_quick(key: str) -> None:
        test = next(t for t in st.CHAT_QUICK_TESTS if t.key == key)
        await send(st.quick_test_prompt(key), test.display)

    def request_stop() -> None:
        if run["active"]:
            run["stop"] = True

    def clear() -> None:
        _clear(transcript, history)
        show_empty_hint()

    send_button.on_click(lambda: send())
    stop_button.on_click(request_stop)
    clear_button.on_click(clear)
    # Enter sends; Shift+Enter falls through to the browser's default and
    # inserts the newline (``exact`` keeps modifier combinations out, and
    # ``prevent`` stops the sent message from also gaining a newline).
    # Ctrl+Enter stays as an alias for muscle memory from the old binding.
    prompt.on("keydown.enter.exact.prevent", lambda: send())
    prompt.on("keydown.ctrl.enter", lambda: send())


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


def _clear(transcript: Any, history: list[dict[str, Any]] | None = None) -> None:
    transcript.clear()
    if history is not None:
        history.clear()


def _render_thumbs(container: Any, images: list[dict[str, str]]) -> None:
    container.clear()
    with container:
        for image in images:
            with ui.column().classes("gap-0 items-center"):
                ui.image(image["url"]).classes("w-16 h-16 object-cover rounded")
                ui.label(image["name"][:16]).classes("text-[11px] opacity-60")


def _bubble(who: str, text: str) -> Any:
    with ui.column().classes("w-full gap-0"):
        ui.label(who).classes("text-[11px] uppercase tracking-wide opacity-60")
        return ui.label(text).classes("text-sm whitespace-pre-wrap")


def _reply_block(model_id: str) -> SimpleNamespace:
    """A reply: who answered, its thinking (folded), the answer, then its numbers."""
    with ui.column().classes("w-full gap-1"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.label(model_id).classes("text-[11px] tracking-wide opacity-60 font-mono break-all")
            status = ui.label("").classes("text-xs opacity-70")
        thinking = ui.expansion("Thinking", icon="psychology").classes("w-full").props("dense")
        with thinking:
            reasoning = ui.label("").classes("text-xs whitespace-pre-wrap opacity-80")
        thinking.set_visibility(False)
        answer = ui.label("").classes("text-sm whitespace-pre-wrap")
        metrics = ui.column().classes("w-full gap-1 mt-1")
    return SimpleNamespace(
        status=status, thinking=thinking, reasoning=reasoning, answer=answer, metrics=metrics
    )


def _show_failure(reply: SimpleNamespace, what: str, exc: BaseException) -> None:
    if not element_alive(reply.answer):
        return
    reply.status.set_text(what)
    reply.answer.classes(add="text-negative")
    reply.answer.set_text(f"{reply.answer.text}\n\n[{what}: {exc}]".strip())


def _paint_reply(reply: SimpleNamespace, content: str, reasoning_stream: str) -> None:
    """Show the answer, with any thinking folded away above it."""
    if not element_alive(reply.answer):
        return
    inline_reasoning, answer = st.split_reasoning(content)
    reasoning = "\n\n".join(part for part in (reasoning_stream, inline_reasoning) if part)
    if reasoning:
        reply.thinking.set_visibility(True)
        reply.thinking.set_text(f"Thinking ({len(reasoning):,} chars)")
        reply.reasoning.set_text(reasoning)
    reply.answer.set_text(answer)


def _render_metrics(container: Any, metrics: Any) -> None:
    container.clear()
    with container:
        with ui.row().classes("w-full gap-x-6 gap-y-2 flex-wrap"):
            for tile in st.chat_metric_tiles(metrics):
                with ui.column().classes("gap-0 min-w-[6.5rem]") as column:
                    ui.label(tile.label).classes("text-[11px] uppercase tracking-wide opacity-60")
                    ui.label(tile.value).classes("text-base font-mono")
                    ui.label(tile.detail).classes("text-[11px] opacity-70")
                column.tooltip(tile.tooltip)
        footer = st.chat_metric_footer(metrics)
        if footer:
            ui.label(footer).classes("text-xs opacity-70")


async def _stream(
    ctx: GuiContext,
    serving_id: str,
    base: str,
    payload: dict[str, Any],
    reply: SimpleNamespace,
    run: dict[str, Any],
) -> SimpleNamespace:
    """Stream a completion from the model's own llama-server child.

    The base URL comes from the supervisor, so it is always the loopback port of
    the child we started -- there is no configured or guessed URL anywhere, which
    is what keeps this working behind any proxy.
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
                    raise RuntimeError(f"HTTP {response.status_code}: {body[:600]}")
                async for line in response.aiter_lines():
                    if run.get("stop"):
                        result.stopped = True
                        break
                    data = _parse_sse(line)
                    if data is None:
                        continue
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
                        if element_alive(reply.status):
                            reply.status.set_text("streaming…")
                    result.last_token_at = now
                    result.chunks += 1
                    if piece:
                        content.append(piece)
                    if thought:
                        reasoning.append(thought)
                    if now - painted_at >= _REPAINT_S:
                        painted_at = now
                        _paint_reply(reply, "".join(content), "".join(reasoning))
    finally:
        result.content = "".join(content)
        result.reasoning = "".join(reasoning)
        _paint_reply(reply, result.content, result.reasoning)
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
