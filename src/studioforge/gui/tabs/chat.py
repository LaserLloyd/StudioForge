"""Chat tab: exercise the real JIT path, including vision.

This deliberately goes through the *same* code the OpenAI endpoints use --
``manager.ensure_loaded`` then a stream from ``supervisor.base_url(...)`` -- so a
successful chat here is real evidence that a client will work, not a separate
mock path that can drift. That includes image attachment: being able to paste a
screenshot and get an answer is the only practical way to verify a vision model
end to end without wiring up OpenClaw first.
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx
from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, notify_error

#: Guard against a paste of a 40 MP screenshot filling the socket buffer.
MAX_IMAGE_BYTES = 20 * 1024 * 1024

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


def render(ctx: GuiContext) -> None:  # noqa: C901 - one screen, one flow
    records = list(ctx.registry.all()) if ctx.registry is not None else []
    chat_records = st.chat_model_records(records)
    options = [r.id for r in chat_records]
    images: list[dict[str, str]] = []

    ui.add_head_html(_PASTE_SCRIPT)

    with ui.column().classes("w-full gap-2 p-2"):
        if not options:
            ui.label("No chat models in the library yet.").classes("text-sm opacity-70")
        hidden_note = st.hidden_chat_models_note(records)
        if hidden_note:
            # An embedding model has no chat endpoint; it is deliberately
            # absent from the picker rather than failing at send time.
            ui.label(hidden_note).classes("text-xs opacity-70")
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            use_loaded = ui.switch("Use the loaded model", value=False)
            use_loaded.props("dense")
            # Created once and retargeted: ``.tooltip()`` appends a new child
            # element each call, and this label is repainted on a timer.
            with use_loaded:
                use_loaded_tip = ui.tooltip("")
            loaded_label = ui.label("").classes("text-sm font-mono opacity-90")
        loaded_note = ui.label("").classes("text-xs opacity-70")
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            model = ui.select(
                options, value=options[0] if options else None, label="Model", with_input=True
            )
            model.props("dense outlined").classes("w-96")
            temperature = ui.number("temperature", value=0.7, precision=2, step=0.05)
            temperature.props("dense outlined").classes("w-32")
            top_p = ui.number("top_p", value=0.95, precision=2, step=0.05)
            top_p.props("dense outlined").classes("w-32")
            max_tokens = ui.number("max_tokens", value=512, precision=0)
            max_tokens.props("dense outlined").classes("w-32")
            rate = ui.label("").classes("text-xs font-mono opacity-80")

        system = (
            ui.textarea("System prompt", value="You are a helpful assistant.")
            .props("dense outlined autogrow")
            .classes("w-full")
        )

        transcript = ui.column().classes(
            "w-full gap-2 p-2 rounded bg-black/5 dark:bg-white/5 min-h-[12rem]"
        )
        attach_note = ui.label("").classes("text-xs opacity-70")
        thumbs = ui.row().classes("gap-2 flex-wrap")

        with ui.row().classes("w-full items-end gap-2 no-wrap"):
            prompt = ui.textarea(placeholder="Message… (Enter to send, Shift+Enter for a new line)")
            prompt.props("dense outlined autogrow").classes("grow")
            upload = (
                ui.upload(
                    label="image", auto_upload=True, multiple=True, max_file_size=MAX_IMAGE_BYTES
                )
                .props('flat dense accept="image/*"')
                .classes("max-w-[12rem]")
            )
            send_button = ui.button("Send", icon="send").props("color=primary")
            clear_button = ui.button("Clear", icon="clear_all").props("flat")

    history: list[dict[str, Any]] = []
    # Clearing only the transcript left `history` intact, so the "cleared"
    # conversation was still sent to the model on the next turn -- it shaped
    # the answer, ate context, and grew without bound for the page's lifetime.
    clear_button.on_click(lambda: _clear(transcript, history))

    # --- which model are we actually talking to? -------------------------
    #
    # The switch is a convenience over a real ambiguity (nothing loaded / one
    # thing loaded / several), so the resolution lives in ``state.chat_target``
    # and this only paints the answer. ``manual`` is kept separately so turning
    # the switch off restores what the user had picked rather than snapping back
    # to the first model in the library.

    manual: dict[str, Any] = {"choice": model.value}

    def resolve_target() -> Any:
        records_now = list(ctx.registry.all()) if ctx.registry is not None else []
        instances = list(ctx.supervisor.list()) if ctx.supervisor is not None else []
        return st.chat_target(
            records_now,
            instances,
            use_loaded=bool(use_loaded.value),
            manual_choice=manual["choice"],
        )

    def sync_target() -> None:
        """Repaint the switch from live state; runs on the GUI's poll cadence."""
        target = resolve_target()
        if target.switch_disabled and use_loaded.value:
            # Nothing loadable to point at: the switch must never sit silently
            # on while chat quietly falls back to the picker.
            use_loaded.set_value(False)
            target = resolve_target()
        use_loaded.set_enabled(not target.switch_disabled)
        use_loaded_tip.set_text(
            target.disabled_reason
            or "Send to whatever is already resident, instead of the model picked below."
        )
        loaded_label.set_text(target.label)
        # ``label`` already carries the disabled reason when there is nothing to
        # point at, so the second line only appears when there is genuinely more
        # to say (several models loaded, and which one won).
        loaded_note.set_text(target.note)
        if target.picker_disabled:
            model.disable()
        else:
            model.enable()
        sync_attach_state()

    # --- image attachment ------------------------------------------------

    def selected_record() -> Any:
        """The record chat will actually use, switch state included."""
        target_id = resolve_target().model_id
        return next((r for r in chat_records if r.id == target_id), None)

    def sync_attach_state() -> None:
        reason = st.vision_attach_reason(selected_record())
        if reason:
            upload.disable()
            attach_note.set_text(reason)
            images.clear()
            _render_thumbs(thumbs, images)
        else:
            upload.enable()
            attach_note.set_text(
                "Vision model: attach a file or just paste an image into the page."
            )

    def add_image(data_url: str, name: str) -> None:
        if st.vision_attach_reason(selected_record()) is not None:
            ui.notify("this model cannot accept images", type="warning")
            return
        if len(data_url) > MAX_IMAGE_BYTES * 2:  # base64 is ~4/3 of the bytes
            ui.notify("image too large", type="negative")
            return
        images.append({"name": name, "url": data_url})
        _render_thumbs(thumbs, images)

    def on_upload(event: Any) -> None:
        import base64

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

    def on_model_change() -> None:
        manual["choice"] = model.value
        sync_target()

    ui.on("sf_paste_image", on_paste)
    model.on_value_change(lambda _: on_model_change())
    use_loaded.on_value_change(lambda _: sync_target())
    sync_target()
    # Same cadence as the rest of the panel: the named model must follow loads
    # and unloads made anywhere else, including by other clients.
    ui.timer(ctx.refresh_interval, sync_target)

    # --- sending ---------------------------------------------------------

    async def send() -> None:
        text = str(prompt.value or "").strip()
        if not text and not images:
            return
        target = str(resolve_target().model_id or "")
        if not target:
            ui.notify("no model selected", type="warning")
            return
        attached = [image["url"] for image in images]
        history.append({"role": "user", "content": st.build_chat_content(text, attached)})
        with transcript:
            _bubble("you", text + (f"\n[{len(attached)} image(s)]" if attached else ""))
            answer = _bubble(target, "")
        prompt.set_value("")
        images.clear()
        _render_thumbs(thumbs, images)
        send_button.disable()
        rate.set_text("loading…")
        try:
            await _stream(
                ctx, target, system.value, history, answer, rate, temperature, top_p, max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what="chat")
            answer.set_text(f"{answer.text}\n\n[failed: {exc}]")
            rate.set_text("")
        finally:
            send_button.enable()

    send_button.on_click(send)
    # Enter sends; Shift+Enter falls through to the browser's default and
    # inserts the newline (``exact`` keeps modifier combinations out, and
    # ``prevent`` stops the sent message from also gaining a newline).
    # Ctrl+Enter stays as an alias for muscle memory from the old binding.
    prompt.on("keydown.enter.exact.prevent", send)
    prompt.on("keydown.ctrl.enter", send)


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
                ui.label(image["name"][:16]).classes("text-[10px] opacity-60")


def _bubble(who: str, text: str) -> Any:
    with ui.column().classes("w-full gap-0"):
        ui.label(who).classes("text-[10px] uppercase opacity-50")
        return ui.label(text).classes("text-sm whitespace-pre-wrap")


async def _stream(
    ctx: GuiContext,
    model_id: str,
    system_prompt: str | None,
    history: list[dict[str, Any]],
    target_label: Any,
    rate_label: Any,
    temperature: Any,
    top_p: Any,
    max_tokens: Any,
) -> None:
    """Stream a completion from the model's own llama-server child.

    The base URL comes from the supervisor, so it is always the loopback port of
    the child we started -- there is no configured or guessed URL anywhere, which
    is what keeps this working behind any proxy.
    """
    record, _instance = await ctx.manager.ensure_loaded(model_id)
    base = ctx.supervisor.base_url(record.id)
    if base is None:
        raise RuntimeError(f"model '{record.id}' is not serving")

    messages: list[dict[str, Any]] = []
    if system_prompt and str(system_prompt).strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.extend(history)

    payload: dict[str, Any] = {
        "model": record.id,
        "messages": messages,
        "stream": True,
        # number_value, not ``or``: an explicit 0 (greedy temperature) must be
        # sent as 0, never silently replaced with the default.
        "temperature": st.number_value(temperature.value, 0.7),
        "top_p": st.number_value(top_p.value, 0.95),
        "max_tokens": int(st.number_value(max_tokens.value, 512)),
    }

    started = time.perf_counter()
    tokens = 0
    collected: list[str] = []
    ctx.supervisor.mark_request_start(record.id)
    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0)) as client,
            client.stream("POST", f"{base}/v1/chat/completions", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if not chunk or chunk == "[DONE]":
                    continue
                try:
                    data = json.loads(chunk)
                except ValueError:
                    continue
                choices = data.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if not piece:
                    continue
                tokens += 1
                collected.append(piece)
                target_label.set_text("".join(collected))
                elapsed = time.perf_counter() - started
                tps = st.tokens_per_second(tokens, elapsed)
                rate_label.set_text(f"{tps} tok/s" if tps else "")
    finally:
        elapsed = time.perf_counter() - started
        ctx.supervisor.mark_request_end(
            record.id, tokens_per_second=st.tokens_per_second(tokens, elapsed)
        )
    history.append({"role": "assistant", "content": "".join(collected)})
