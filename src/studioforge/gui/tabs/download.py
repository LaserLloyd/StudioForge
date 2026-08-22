"""Download tab: find a GGUF on HuggingFace and pick a quant that will fit.

The fit verdict at *pick* time is the point of this tab. A user choosing a quant
from a list of eleven has no way to know which of them their hardware can
actually serve, and finding out after a 40 GiB download is the worst possible
time. So every quant row carries a weights-only estimate against current free
VRAM -- deliberately crude, clearly labelled, and enough to steer.

This tab is also the landing page for HuggingFace's download button. The
protocol handler turns ``lmstudio://open_from_hf?model=owner/repo`` into
``/?tab=download&repo=owner/repo``, and a link like that skips the search box
entirely: the repo's quant list is fetched immediately and put at the top of the
screen, which is the "LM Studio pops up and asks which version I want" moment
the whole deep-link path exists to reproduce.

``studioforge.core.hf_search`` is imported lazily inside the callbacks, and the
queue degrades to an explanatory panel when ``api_state.downloader is None``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nicegui import ui

from studioforge.gui import state as st
from studioforge.gui.tabs import GuiContext, busy, notify_error, require_local_admin

#: The only external link in the GUI: a repo page on HuggingFace, so a user can
#: read the model card before committing to a download.
HF_WEB_BASE = "https://huggingface.co"


def render(ctx: GuiContext, params: Mapping[str, Any] | None = None) -> None:
    params = params or {}
    with ui.column().classes("w-full gap-3 p-2"):
        _deep_link_panel(ctx, params)
        _search_panel(ctx)
        _queue_panel(ctx)


def _gpus(ctx: GuiContext) -> list[Any]:
    try:
        return list(ctx.probe.list_gpus()) if ctx.probe is not None else []
    except Exception:  # noqa: BLE001 - a fit estimate is a nicety, not a blocker
        return []


def _disk(ctx: GuiContext) -> dict[str, Any] | None:
    """Free space where downloads land, or ``None`` when there is nothing to say.

    Swallows everything on purpose: no ``models.dir`` configured yet, a
    downloader that is not wired in, an unmapped drive. All three mean "no disk
    line", and none of them is a reason for the tab to stop working.

    Cheap enough to call per render -- ``core.diskspace`` caches the syscall for
    a couple of seconds, which is what makes it safe on the queue's poll timer
    and in the quant picker at the same time.
    """
    from studioforge.core.diskspace import disk_report

    models_dir = ctx.config.models.dir
    if models_dir is None:
        return None
    queued = 0
    downloader = ctx.downloader
    if downloader is not None:
        try:
            queued = int(downloader.queued_remaining_bytes())
        except Exception:  # noqa: BLE001 - the free figure alone is still worth showing
            queued = 0
    try:
        return disk_report(Path(models_dir), queued)
    except Exception:  # noqa: BLE001 - disk_report is total, but it is not ours to trust blindly
        return None


async def _repo_info(ctx: GuiContext, repo_id: str) -> Any:
    """Full file listing for one repo, with sizes."""
    from studioforge.core.hf_search import HfSearch

    client = HfSearch(ctx.config)
    try:
        return await client.repo_info(repo_id)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Deep-link landing
# ---------------------------------------------------------------------------


def _deep_link_panel(ctx: GuiContext, params: Mapping[str, Any]) -> None:
    """The quant picker for a repo named in the URL, above everything else.

    Only rendered when the URL actually carried one, so a plain visit to the
    tab is unchanged.
    """
    repo_id = params.get("repo")
    error = params.get("error")
    if not repo_id and not error:
        return

    quant = params.get("quant")
    with ui.card().classes("w-full border-l-4 border-primary"):
        ui.label("From HuggingFace").classes("text-xs uppercase opacity-60")
        if not repo_id:
            _link_error_card(str(error))
            return
        ui.label(st.deep_link_headline(str(repo_id), quant)).classes("text-lg font-medium")
        ui.label("Choose which version to download.").classes("text-xs opacity-70")
        body = ui.column().classes("w-full gap-1")
        with body:
            ui.spinner(size="sm")
            ui.label(f"fetching {repo_id} from HuggingFace…").classes("text-xs opacity-60")

        async def load() -> None:
            body.clear()
            try:
                full = await _repo_info(ctx, str(repo_id))
            except Exception as exc:  # noqa: BLE001 - every failure is explained below
                with body:
                    _repo_error_card(ctx, str(repo_id), exc, retry=load)
                return
            with body:
                _quant_rows(ctx, full, highlight=quant, on_picked=None)

        ui.timer(0.05, load, once=True)


def _link_error_card(message: str) -> None:
    ui.label("That link could not be used").classes("text-negative font-medium text-sm")
    ui.label(message).classes("text-xs opacity-80")
    ui.label("Search for the model below instead.").classes("text-xs opacity-60")


def _repo_error_card(ctx: GuiContext, repo_id: str, exc: BaseException, *, retry: Any) -> None:
    """Explain a failed repo lookup in terms of what to do about it.

    The three real cases -- no such repo, gated/private, and network/rate-limit
    -- need different actions, and a bare traceback (or worse, a blank panel)
    tells the user none of them.
    """
    from studioforge.errors import StudioForgeError

    message = exc.message if isinstance(exc, StudioForgeError) else f"{type(exc).__name__}: {exc}"
    ui.label(f"Could not load {repo_id}").classes("text-negative font-medium text-sm")
    ui.label(message).classes("text-xs opacity-80 whitespace-pre-wrap")

    param = getattr(exc, "param", None)
    status = (getattr(exc, "details", None) or {}).get("status")
    if param == "hf.token" or status in (401, 403):
        if ctx.config.hf.token:
            ui.label(
                "A HuggingFace token is configured, so this is most likely a licence you have "
                "not accepted yet — open the model card, accept it, then retry."
            ).classes("text-xs text-warning")
        else:
            ui.label(
                "This repository is gated. Accept its licence on huggingface.co, then set "
                "hf.token on the Server tab and retry."
            ).classes("text-xs text-warning")
    elif status == 404:
        ui.label(
            "Check the owner/repo spelling — the link may point at a repository that has been "
            "renamed or removed."
        ).classes("text-xs text-warning")
    else:
        ui.label("This looks like a network or rate-limit problem; retrying often works.").classes(
            "text-xs text-warning"
        )

    with ui.row().classes("gap-2 items-center"):
        ui.button("Retry", icon="refresh", on_click=retry).props("outline dense")
        ui.link("open the model card", f"{HF_WEB_BASE}/{repo_id}", new_tab=True).classes("text-xs")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


#: Sort menu: label shown -> the name `HfSearch.search(sort=...)` expects.
#:
#: "Downloads (30d)" spells out the window because HF's download figure is the
#: trailing-30-day count, not an all-time total. Users read an unqualified
#: "downloads" as lifetime popularity and then wonder why a two-year-old
#: classic ranks below last week's merge.
SORT_CHOICES: dict[str, str] = {
    "Downloads (30d)": "downloads",
    "Likes": "likes",
    "Recently updated": "updated",
    "Newly created": "created",
    "Trending": "trending",
}

#: Period menu: label -> ``newer_than_days``. ``None`` means no window at all,
#: which is also the only setting that lets HF do the sorting server-side over
#: the whole hub rather than over a locally-walked slice of it.
PERIOD_CHOICES: dict[str, int | None] = {
    "Any time": None,
    "Past day": 1,
    "Past week": 7,
    "Past month": 30,
    "Past 3 months": 90,
}

#: The sort whose period filter means "created in the window" rather than
#: "updated in the window". Asking for newly *created* repos and then filtering
#: on last-modified would return long-lived repos that were merely touched.
_CREATED_SORT = "created"


def _period_tooltip(sort_value: str) -> str:
    return (
        "Period filters on the repo's creation date, because the sort is 'Newly created'."
        if sort_value == _CREATED_SORT
        else "Period filters on when the repo was last updated."
    )


def _search_panel(ctx: GuiContext) -> None:
    ui.label("Find a model").classes("text-lg font-medium")
    with ui.row().classes("w-full items-end gap-2 flex-wrap"):
        query = ui.input(placeholder="e.g. qwen3 coder gguf").props("dense outlined clearable")
        query.classes("grow min-w-[16rem]")
        sort = ui.select(list(SORT_CHOICES), value="Downloads (30d)", label="Sort")
        sort.props("dense outlined").classes("w-48")
        period = ui.select(list(PERIOD_CHOICES), value="Any time", label="Period")
        period.props("dense outlined").classes("w-40")
        # Nested inside the select rather than via `period.tooltip(...)`, which
        # returns the select and would leave no handle to retarget the text on.
        with period:
            period_hint = ui.tooltip(_period_tooltip("downloads"))
        button = ui.button("Search", icon="search").props("color=primary")
    results = ui.column().classes("w-full gap-2")

    def _sort_value() -> str:
        return SORT_CHOICES.get(str(sort.value or ""), "downloads")

    def _retarget_tooltip() -> None:
        # The tooltip has to be rebuilt on change: which date the period applies
        # to depends on the sort, and a stale "last updated" hint next to a
        # "Newly created" sort is worse than no hint.
        period_hint.text = _period_tooltip(_sort_value())

    sort.on_value_change(lambda _e: _retarget_tooltip())

    async def search() -> None:
        results.clear()
        sort_value = _sort_value()
        window = PERIOD_CHOICES.get(str(period.value or ""))
        # "Newly created" is the one sort that asks about birth rather than
        # activity, so its period has to be measured the same way.
        date_field = "created" if sort_value == _CREATED_SORT else "updated"
        with busy(button, message="Searching HuggingFace…"):
            try:
                from studioforge.core.hf_search import HfSearch

                client = HfSearch(ctx.config)
                try:
                    repos = await client.search(
                        str(query.value or ""),
                        limit=20,
                        sort=sort_value,
                        newer_than_days=window,
                        date_field=date_field,
                    )
                    truncated = client.last_search_truncated
                finally:
                    await client.aclose()
            except Exception as exc:  # noqa: BLE001
                notify_error(exc, what="HuggingFace search")
                with results:
                    ui.label(f"search failed: {exc}").classes("text-negative text-sm")
                return
        with results:
            if not repos:
                ui.label("No GGUF repos matched.").classes("text-sm opacity-70")
                return
            if truncated:
                # Saying so beats quietly presenting a partial window as the
                # whole period: HF has no date filter, so a broad query over a
                # long period is cut off by our page cap, not by the hub. The
                # wording avoids "most recent matches" -- the rows shown are the
                # top ones by the chosen sort, but only the newest slice of the
                # period was searched to find them.
                ui.label(
                    "This period has more matches than one search can fetch, so only its "
                    "most recent part was searched. Narrow the query or shorten the period."
                ).classes("text-xs text-warning")
            for repo in repos:
                _repo_row(ctx, repo, date_field=date_field)

    button.on_click(search)
    query.on("keydown.enter", search)


def _age_label(days: float | None) -> str:
    """``today`` / ``3d ago`` / ``2w ago`` / ``5mo ago`` from a fractional age.

    Coarsens with distance on purpose: "updated 412d ago" is arithmetic, not
    information, and the only question a model shopper is asking is "is this
    stale?". Returns ``""`` for an unknown date so callers can drop the
    separator rather than print "unknown ago".
    """
    if days is None:
        return ""
    if days < 1:
        return "today"
    if days < 14:
        return f"{int(days)}d ago"
    if days < 60:
        return f"{int(days // 7)}w ago"
    return f"{int(days // 30)}mo ago"


def _repo_row(ctx: GuiContext, repo: Any, *, date_field: str = "updated") -> None:
    verb = "created" if date_field == "created" else "updated"
    age = _age_label(repo.created_days_ago if date_field == "created" else repo.updated_days_ago)
    summary = (
        f"{repo.publisher} · {repo.downloads:,} downloads · {repo.likes:,} likes"
        f" · {len(repo.quant_variants)} quant(s)"
    )
    if age:
        # Appended only when HF sent a date, so an unknown one drops the whole
        # clause instead of rendering a dangling "· updated".
        summary += f" · {verb} {age}"
    with ui.card().classes("w-full"), ui.row().classes("w-full items-center gap-3 no-wrap"):
        with ui.column().classes("gap-0 grow min-w-0"):
            ui.label(repo.repo_id).classes("font-medium text-sm truncate")
            ui.label(summary).classes("text-xs opacity-70 font-mono")
        if repo.needs_token:
            ui.badge("gated", color="warning").classes("text-xs").tooltip(
                "Accept the terms on HuggingFace and set hf.token on the Server tab"
            )
        ui.link("model card", f"{HF_WEB_BASE}/{repo.repo_id}", new_tab=True).classes("text-xs")
        ui.button(
            "Quants", icon="expand_more", on_click=lambda r=repo: _quant_dialog(ctx, r)
        ).props("outline dense")


def _quant_dialog(ctx: GuiContext, repo: Any) -> None:
    dialog = ui.dialog()
    with dialog, ui.card().classes("min-w-[40rem] max-w-[95vw]"):
        ui.label(f"{repo.repo_id} — pick a quant").classes("font-medium")
        body = ui.column().classes("w-full gap-1")
        with body:
            ui.label("loading file list…").classes("text-xs opacity-60")
        ui.button("Close", on_click=dialog.close).props("flat")

    async def populate() -> None:
        body.clear()
        try:
            full = await _repo_info(ctx, repo.repo_id)
        except Exception as exc:  # noqa: BLE001
            with body:
                _repo_error_card(ctx, repo.repo_id, exc, retry=populate)
            return
        with body:
            _quant_rows(ctx, full, highlight=None, on_picked=dialog.close)

    dialog.open()
    ui.timer(0.05, populate, once=True)


# ---------------------------------------------------------------------------
# Quant picker (shared by the dialog and the deep-link landing)
# ---------------------------------------------------------------------------


def _quant_rows(ctx: GuiContext, full: Any, *, highlight: Any, on_picked: Any) -> None:
    """One row per logical download, with a fit verdict and a Download button.

    Rendered in two passes on purpose. The first is synchronous and uses only
    what HuggingFace already told us (file sizes vs free VRAM), so the picker is
    on screen immediately. The second reads the model's GGUF header over the
    network and fills in the context line -- and upgrades the fit badge to the
    planner's answer -- so a slow CDN costs the user a late line, not a frozen
    dialog.

    The disk check is the other half of "can I have this?": VRAM decides whether
    it will run, the drive decides whether it can even arrive. It is a warning
    and never a block -- freeing 40 GiB is a thing a user can go and do, and a
    greyed-out button would not tell them that is all it takes.
    """
    ui.label(
        "Fit is a weights-only estimate against free VRAM right now; it is replaced by the "
        "planner's own answer as soon as the model header has been read."
    ).classes("text-xs opacity-70")
    options = full.logical_models()
    if not options:
        ui.label("no loadable GGUF files in this repo").classes("text-xs opacity-60")
        return

    geometry = ui.label("").classes("text-xs font-mono opacity-60")
    gpus = _gpus(ctx)
    disk = _disk(ctx)
    matched = False
    cells: list[tuple[Any, Any, Any, Any]] = []
    for option in options:
        is_match = st.quant_matches(highlight, option.quant)
        matched = matched or is_match
        verdict = st.download_fit_verdict(
            option.total_bytes, gpus, headroom_fraction=ctx.config.planner.headroom_fraction
        )
        classes = "w-full items-center gap-3 no-wrap rounded p-1"
        if is_match:
            classes += " bg-primary/20 border border-primary"
        with ui.row().classes(classes):
            with ui.column().classes("gap-0 grow min-w-0"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(option.label).classes("text-sm truncate")
                    if is_match:
                        ui.badge("from your link", color="primary").classes("text-xs")
                ui.label(f"{st.format_gib(option.total_bytes)} · {verdict}").classes(
                    "text-xs font-mono opacity-70"
                )
                context = ui.label(_HEADER_PENDING).classes("text-xs font-mono opacity-60")
                # Nested rather than `.tooltip(...)`, which returns the label and
                # leaves no handle to retarget once the header arrives.
                with context:
                    tip = ui.tooltip("").classes("whitespace-pre-line text-xs")
            colour = (
                "positive"
                if verdict.startswith("fits")
                else ("warning" if verdict.startswith("needs") else "negative")
            )
            fit_badge = ui.badge(verdict.split(" (")[0], color=colour).classes("text-xs")
            cells.append((option, context, tip, fit_badge))
            if st.disk_would_overflow(disk, option.total_bytes):
                ui.badge("not enough disk", color="warning").classes("text-xs").tooltip(
                    f"{st.format_bytes(option.total_bytes)} needed, "
                    f"{st.format_bytes(disk['free_after_queue_bytes']) if disk else '?'} "
                    "free after what is already queued. The download will still run."
                )
            ui.button(
                "Download",
                icon="download",
                on_click=lambda o=option: _enqueue(ctx, o, on_picked),
            ).props("outline dense" if not is_match else "color=primary dense")

    if highlight and not matched:
        ui.label(
            f"The link asked for '{highlight}', which this repo does not publish. "
            "Pick one of the versions above instead."
        ).classes("text-xs text-warning")

    async def fill() -> None:
        await _fill_context_lines(ctx, full, cells, geometry)

    ui.timer(0.05, fill, once=True)


#: Placeholder while the remote header is in flight. Present from the first
#: paint so the row does not jump when the real line lands.
_HEADER_PENDING = "reading model header…"


async def _fill_context_lines(ctx: GuiContext, full: Any, cells: list[Any], geometry: Any) -> None:
    """Second pass: the real per-quant context line, and the real fit badge.

    One header read for the whole repo (every quant shares the geometry), then
    pure arithmetic per row. Failures are shown as the reason on the first row
    rather than swallowed: "why does this repo have no context line" is a
    question the user can only answer if we say.

    The badge is rewritten here too. Until the header lands it is
    ``download_fit_verdict``'s weights-vs-free-VRAM guess; once the planner has
    placed the model it becomes the planner's answer, which counts the compute
    buffers and the CUDA context the guess cannot see. Leaving the guess in
    place produced the contradiction WP7 documented: ``fits one GPU`` on the
    same row as ``1x5090: --``.
    """
    from studioforge.core.hf_meta import (
        context_line,
        context_matrix,
        context_tooltip,
        geometry_line,
        idle_planner,
        repo_arch_meta,
    )

    planner = ctx.planner
    if planner is None or not cells:
        for _option, label, _tip, _badge in cells:
            label.set_text("")
        return
    try:
        arch = await repo_arch_meta(ctx.config, full, registry=ctx.registry)
        idle = idle_planner(planner)
    except Exception as exc:  # noqa: BLE001 - a missing line must not kill the picker
        for _option, label, _tip, _badge in cells:
            label.set_text(f"context unknown: {exc}")
        return

    first: dict[str, Any] | None = None
    for option, label, tip, fit_badge in cells:
        mmproj_bytes = option.mmproj.size_bytes if option.mmproj else 0
        try:
            matrix = context_matrix(
                arch.meta,
                max(0, int(option.total_bytes) - int(mmproj_bytes)),
                planner=idle,
                mmproj_bytes=mmproj_bytes,
                model_id=f"hf:{option.repo_id}#{option.quant}",
                source=arch.source,
                unavailable=arch.unavailable,
            )
        except Exception as exc:  # noqa: BLE001
            label.set_text(f"context unknown: {exc}")
            continue
        first = first or matrix
        label.set_text(context_line(matrix))
        tip.text = context_tooltip(matrix)
        upgraded = st.fit_badge_from_context(matrix)
        if upgraded is not None:
            text, colour = upgraded
            fit_badge.set_text(text)
            fit_badge.props(f"color={colour}")
    if first is not None:
        geometry.set_text(geometry_line(first))


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


async def _enqueue(ctx: GuiContext, option: Any, on_picked: Any = None) -> None:
    downloader = ctx.downloader
    if downloader is None:
        ui.notify("downloads are not available in this build", type="warning")
        return
    try:
        # Same rule as POST /api/downloads (D32): writing to the library is a
        # box change.
        require_local_admin(ctx, "enqueue download")
        group_id = await downloader.enqueue(option)
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what="enqueue download")
        return
    if on_picked is not None:
        on_picked()
    ui.notify(f"queued {option.label} ({group_id})", type="positive")


#: Base classes for the disk line. The colour is appended rather than added
#: separately so the whole string can be replaced on every poll.
_DISK_CLASSES = "text-xs font-mono"


def _queue_panel(ctx: GuiContext) -> None:
    ui.label("Downloads").classes("text-lg font-medium")
    if ctx.downloader is None:
        with ui.card().classes("w-full"):
            ui.label("Downloads not available").classes("font-medium text-sm")
            ui.label(
                "The download subsystem is not wired into this server, so searching works but "
                "nothing can be fetched. Everything else in the panel is unaffected."
            ).classes("text-xs opacity-70")
        return

    # Above the queue rather than beside a row, because it is a property of the
    # whole queue: what matters is whether everything in it still fits.
    disk = ui.label("").classes(_DISK_CLASSES)
    show_finished = ui.checkbox("show finished", value=False)
    stale = ui.label("").classes("text-xs text-warning")
    rows = ui.column().classes("w-full gap-2")

    def paint_disk() -> None:
        report = _disk(ctx)
        disk.set_text(st.disk_line(report))
        # `replace` rather than `add`: a queue that drains back below the
        # threshold has to lose the warning colour, not keep it forever.
        warn = " text-warning" if st.disk_is_low(report) else " opacity-70"
        disk.classes(replace=_DISK_CLASSES + warn)

    def refresh() -> None:
        """Poll the downloader.

        Polling rather than subscribing: ``active()`` is a cheap in-memory
        snapshot, and a push callback would fire from the transfer task at
        arbitrary rates, forcing us to throttle it back down to exactly this.
        """
        # A timer tick renders into the slot the timer was created in, so a
        # panel_guard error card here would be APPENDED once per tick, forever,
        # on every connected client. Keep the last good rows and show a single
        # stale marker instead -- the same discipline the Dashboard uses.
        try:
            downloader = ctx.downloader
            source = downloader.all() if show_finished.value else downloader.active()
            payloads = [progress.to_dict() for progress in source]
            groups = st.group_download_rows(payloads, status_for=downloader.group_status)
            notes = _queue_notes(payloads)
            paint_disk()
        except Exception as exc:  # noqa: BLE001 - a poll must never kill the tab
            stale.set_text(f"download list is stale: {exc}")
            return
        stale.set_text("")
        rows.clear()
        with rows:
            if not groups:
                ui.label("nothing queued").classes("text-xs opacity-60")
                return
            for group in groups:
                _download_row(ctx, group, note=notes.get(group.group_id))

    refresh()
    show_finished.on_value_change(lambda _: refresh())
    ui.timer(max(1.0, ctx.refresh_interval), refresh)


def _queue_notes(payloads: list[dict[str, Any]]) -> dict[str, str]:
    """One extra line per group: what is happening, or what Resume will do.

    Two questions the queue could not answer before, and both of them cost real
    money in bytes. "It says running but nothing is moving" -- because it is
    thirty seconds into a backoff, which is now stated with the attempt number
    and the error. And "it failed; if I press Resume, do I keep my 19 GB?" --
    which on 2026-08-18 was answered *no* by a second writer that had already
    consumed the partial file, with nothing in the GUI willing to say so.

    Built from the raw progress payloads rather than from
    :class:`~studioforge.gui.state.DownloadGroupRow`, because a group is an
    aggregate and this is per file: the retrying one is the interesting one.
    """
    notes: dict[str, str] = {}
    for row in payloads:
        group_id = str(row.get("group_id") or row.get("id") or "?")
        if group_id in notes:
            continue
        retry_in = row.get("retry_in_s")
        attempt = int(row.get("attempt") or 0)
        if isinstance(retry_in, int | float) and row.get("next_retry_at") and attempt:
            reason = str(row.get("last_error") or "transfer failed")
            notes[group_id] = (
                f"retrying in {max(0, round(float(retry_in)))}s "
                f"(attempt {attempt}/{int(row.get('max_attempts') or 5)}): {reason[:160]}"
            )
        elif row.get("status") == "failed":
            part = int(row.get("part_bytes") or 0)
            notes[group_id] = (
                f"Resume continues from {st.format_bytes(part)}"
                if part > 0
                else "Resume will restart from the beginning (no partial file)"
            )
    return notes


def _download_row(ctx: GuiContext, group: st.DownloadGroupRow, *, note: str | None = None) -> None:
    with ui.card().classes("w-full"), ui.column().classes("w-full gap-1"):
        with ui.row().classes("w-full items-center gap-2 no-wrap"):
            ui.label(group.label).classes("text-sm grow truncate")
            ui.badge(group.status, color=st.download_status_colour(group.status)).classes("text-xs")
            # Only the controls that can do something for this status: a
            # completed row gets none (see ``state.download_actions``).
            specs = {
                "pause": ("pause", "Pause"),
                "resume": ("play_arrow", "Resume"),
                "cancel": ("close", "Cancel (deletes the partial file)"),
            }
            for action in st.download_actions(group.status):
                icon, tooltip = specs[action]
                ui.button(
                    icon=icon,
                    on_click=lambda a=action, g=group.group_id: _control(ctx, a, g),
                ).props("flat dense").tooltip(tooltip)
        ui.linear_progress(value=group.fraction, show_value=False, size="12px").props("rounded")
        ui.label(group.detail).classes("text-xs font-mono opacity-70")
        if note:
            ui.label(note).classes("text-xs text-warning whitespace-pre-wrap")
        if group.error:
            ui.label(group.error[:300]).classes("text-xs text-negative whitespace-pre-wrap")


async def _control(ctx: GuiContext, action: str, group_id: str) -> None:
    downloader = ctx.downloader
    if downloader is None:
        return
    try:
        require_local_admin(ctx, action)
        await getattr(downloader, action)(group_id)
    except Exception as exc:  # noqa: BLE001
        notify_error(exc, what=action)
        return
    ui.notify(f"{action}: {group_id}", type="positive")
