"""The first-run wizard as a page: ``/setup``.

Eight steps in a stepper that mirror :mod:`studioforge.setup_wizard`, the same
orchestration ``studioforge setup`` runs in a terminal. The panel runs *inside*
the server, so the wizard talks to the live objects directly through
:class:`PanelServices` (an in-process implementation of the wizard's
``WizardServices`` protocol): no HTTP request back to ourselves, and nothing to
start.

* Every step is usable on its own. Next, Back and Skip work whatever the other
  steps did, and the wizard never requires the owner account for local-only use.
* Slow steps (engine install, a model download or load, a test chat) run on a
  worker thread. The thread calls back onto the event loop for anything async
  (:meth:`PanelServices._await`), so the page stays responsive and every engine,
  downloader and manager call happens where the rest of the server expects it.
* Every change a step makes is gated by :func:`require_local_admin` on the event
  loop *before* the worker starts, and writes an audit line with the signed-in
  actor captured there (a worker thread has no page to ask).
* A secret created here (the network-exposure admin key) goes to the one-time
  dialog and is removed from the step's result before anything else sees it.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import ui

from studioforge.core.integrations import build_snippets
from studioforge.gui.tabs import (
    GuiContext,
    api_request,
    busy,
    error_text,
    notify_error,
    panel_guard,
    require_local_admin,
    run_blocking,
    viewer_actor,
)
from studioforge.gui.tabs.dashboard import _Page
from studioforge.gui.tabs.overview import server_root
from studioforge.gui.tabs.widgets import code_block, one_time_secret_dialog, status_row
from studioforge.logging import get_logger
from studioforge.setup_wizard import (
    SetupWizard,
    StepResult,
    WizardOptions,
    pick_starter,
)

log = get_logger(__name__)

#: ``(key, title shown in the stepper)`` in order; the keys are the wizard's own.
STEPS: tuple[tuple[str, str], ...] = (
    ("account", "1. Owner account"),
    ("hardware", "2. Detect hardware"),
    ("engine", "3. Engine"),
    ("library", "4. Model libraries"),
    ("model", "5. Starter model"),
    ("chat", "6. Test chat"),
    ("connect", "7. Connect your tools"),
    ("exposure", "8. Who can reach it"),
)

PEER_LABEL = "Other StudioForge machines (peers). Coming in a later release"

DOWNLOAD_TIMEOUT_S = 3 * 3600.0

_STATUS_STYLE = {
    "done": ("check_circle", "positive"),
    "skipped": ("skip_next", "info"),
    "failed": ("error", "negative"),
    "manual": ("build_circle", "warning"),
}


# ---------------------------------------------------------------------------
# The in-process services
# ---------------------------------------------------------------------------


class PanelServices:
    """:class:`~studioforge.setup_wizard.WizardServices` over the running server.

    Every method is synchronous (the protocol is) and meant for a worker
    thread. Async work is submitted to ``loop`` and waited for; with no loop
    (a test, a script) it runs on a private one.
    """

    def __init__(
        self,
        ctx: GuiContext,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        actor: str = "panel",
        poll_s: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        download_timeout_s: float = DOWNLOAD_TIMEOUT_S,
    ) -> None:
        self.ctx = ctx
        self._loop = loop
        self.actor = actor
        self._poll_s = poll_s
        self._sleep = sleep
        self._download_timeout_s = download_timeout_s
        #: What a progress bar can show while a slow step runs.
        self.progress: dict[str, Any] = {"phase": "", "fraction": 0.0}
        #: Set to abandon a download that is being waited for.
        self.cancelled = threading.Event()

    # -- plumbing ----------------------------------------------------------

    def _await(self, coro: Any) -> Any:
        """Run a coroutine where the server lives and return its result."""
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                here: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
            except RuntimeError:
                here = None
            if here is loop:  # pragma: no cover - a programming error, not a runtime state
                coro.close()
                raise RuntimeError("PanelServices must be called from a worker thread")
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        return asyncio.run(coro)

    def _audit(self, action: str, target: str = "", *, result: str = "ok") -> None:
        audit = getattr(self.ctx.api_state, "audit", None)
        if audit is None:
            return
        try:
            audit.record(
                actor=self.actor, action=action, target=target, result=result, surface="panel"
            )
        except Exception as exc:  # noqa: BLE001 - bookkeeping never fails a step
            log.debug("could not write the audit line", action=action, error=str(exc))

    def _accounts(self) -> Any:
        accounts = getattr(self.ctx.api_state, "accounts", None)
        if accounts is None:
            raise RuntimeError("Accounts are not available on this server build.")
        return accounts

    def _apply(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Save settings through the management route (validation, atomic write, live apply)."""
        from studioforge.api.mgmt_routes import set_config

        payload = dict(self._await(set_config(api_request(self.ctx), updates)))
        self._audit("panel.config.set", ", ".join(sorted(str(k) for k in updates)))
        return payload

    # -- 1. account --------------------------------------------------------

    def has_owner(self) -> bool:
        accounts = getattr(self.ctx.api_state, "accounts", None)
        return bool(accounts is not None and accounts.has_owner())

    def create_owner(self, username: str, password: str) -> None:
        account = self._accounts().create_account(username, password)
        self._audit("panel.account.create", f"{account.username} ({account.role})")

    # -- 2. hardware -------------------------------------------------------

    def detect_hardware(self) -> dict[str, Any]:
        probe = self.ctx.probe
        gpus = probe.list_gpus() if probe is not None else []
        return {
            "backend": str(getattr(probe, "backend", "unknown")),
            "gpus": [g.model_dump(mode="json") if hasattr(g, "model_dump") else g for g in gpus],
        }

    # -- 3. engine ---------------------------------------------------------

    def engine_state(self) -> dict[str, Any]:
        manager = self.ctx.engine_manager
        active = manager.active() if manager is not None else None
        return {
            "active": active.tag if active else None,
            "smoke_tested": bool(active and getattr(active, "smoke_tested", False)),
            "pinned_tag": self.ctx.config.engine.pinned_tag,
        }

    def ensure_engine(self, tag: str | None) -> dict[str, Any]:
        manager = self.ctx.engine_manager
        if manager is None:
            raise RuntimeError("The engine manager is not available on this server build.")
        wanted = tag or self.ctx.config.engine.pinned_tag

        async def work() -> dict[str, Any]:
            def on_progress(phase: str, fraction: float) -> None:
                self.progress.update(phase=phase, fraction=fraction)

            info = await manager.install(wanted, force=False, activate=False, progress=on_progress)
            self.progress.update(phase="smoke test", fraction=1.0)
            ok, smoke_result, message = await manager.smoke_check(info.tag)
            if ok:
                from studioforge.api.mgmt_routes import engine_activate

                await engine_activate(api_request(self.ctx), tag=info.tag)
            return {"tag": info.tag, "smoke_tested": smoke_result == "passed", "message": message}

        result: dict[str, Any] = self._await(work())
        self._audit("panel.engine.install", str(result.get("tag")))
        return result

    # -- 4. library --------------------------------------------------------

    def find_libraries(self) -> list[dict[str, Any]]:
        from studioforge.setup_wizard import _count_ggufs, detect_lmstudio_library

        found: list[dict[str, Any]] = []
        configured = self.ctx.config.models.dir
        if configured:
            found.append(
                {
                    "path": str(configured),
                    "source": "configured",
                    "models": _count_ggufs(Path(configured)),
                }
            )
        known = {str(Path(p).resolve()) for p in [configured] if p}
        for path in detect_lmstudio_library():
            if str(path.resolve()) not in known:
                found.append(
                    {"path": str(path), "source": "lmstudio", "models": _count_ggufs(path)}
                )
        return found

    def set_library(self, path: str) -> None:
        self._apply({"models.dir": path})

    # -- 5. model ----------------------------------------------------------

    def ensure_server(self) -> str:
        """The panel runs inside the server, so there is nothing to start."""
        return server_root(self.ctx)

    def library_models(self) -> list[dict[str, Any]]:
        registry = self.ctx.registry
        if registry is None:
            return []
        return [
            {
                "id": record.id,
                "name": getattr(record, "name", record.id),
                "size_bytes": int(getattr(record, "size_bytes", 0) or 0),
                "quant": getattr(record, "quant", None),
            }
            for record in registry.all()
            if not getattr(record, "is_virtual", False)
        ]

    def download_model(self, repo: str) -> dict[str, Any]:
        from studioforge.api.mgmt_routes import start_download

        downloader = self.ctx.downloader
        if downloader is None:
            raise RuntimeError("Downloads are not available on this server build.")
        self.cancelled.clear()
        started = self._await(
            start_download(
                api_request(self.ctx), repo_id=repo, quant=None, include_mmproj=True, force=False
            )
        )
        group = str(started["group_id"])
        files = [Path(str(name)).name for name in started.get("files") or []]
        self._audit("panel.download.start", f"{repo} {started.get('quant') or ''}".strip())
        deadline = time.monotonic() + self._download_timeout_s
        while time.monotonic() < deadline:
            if self.cancelled.is_set():
                self._await(downloader.cancel(group, delete_partial=False))
                raise RuntimeError(f"The download of {repo} was canceled.")
            rows = downloader.group(group)
            done = sum(int(getattr(r, "downloaded_bytes", 0) or 0) for r in rows)
            total = sum(int(getattr(r, "total_bytes", 0) or 0) for r in rows)
            self.progress.update(
                phase=f"downloading {repo}", fraction=(done / total) if total else 0
            )
            status = downloader.group_status(group)
            if status == "completed":
                break
            if status in ("failed", "canceled"):
                raise RuntimeError(
                    f"The download of {repo} {status}. Open the Models page, Download tab, to see "
                    "why and to retry."
                )
            self._sleep(self._poll_s)
        else:
            raise RuntimeError(f"The download of {repo} did not finish in time.")
        self.progress.update(phase="indexing", fraction=1.0)
        registry = self.ctx.registry
        if registry is None:
            raise RuntimeError("The model registry is not available on this server build.")
        registry.scan(force=False)
        stem = repo.split("/")[-1].lower().replace("-gguf", "")
        for record in registry.all():
            if getattr(record, "is_virtual", False):
                continue
            path_name = Path(str(getattr(record, "path", ""))).name
            if path_name in files or stem in record.id.lower():
                return {"model_id": record.id}
        raise RuntimeError(f"{repo} downloaded but no matching model appeared in the library.")

    def load_model(self, model_id: str) -> dict[str, Any]:
        manager = self.ctx.manager
        if manager is None:
            raise RuntimeError("The model manager is not available on this server build.")
        ctx_size = int(self.ctx.config.models.default_ctx)
        self.progress.update(phase=f"loading {model_id}", fraction=0.0)
        # Same call the load-recommended route and the Models page make: the
        # server picks the placement, the KV cache and the slot count.
        self._await(manager.load_recommended(model_id, ctx_size, source="gui"))
        self._audit("panel.model.load", model_id)
        return {"ctx_size": ctx_size}

    # -- 6. chat -----------------------------------------------------------

    def test_chat(self, model_id: str) -> dict[str, Any]:
        manager = self.ctx.manager
        if manager is None:
            raise RuntimeError("The model manager is not available on this server build.")
        result = self._await(manager.test_model(model_id, None, keep_loaded=True))
        ok = bool(result.get("ok", True))
        return {
            "ok": ok,
            "reply": result.get("text"),
            "tokens_per_second": result.get("tokens_per_second"),
            "error": None if ok else "the model answered with no text",
        }

    # -- 7. connect --------------------------------------------------------

    def base_url(self) -> str:
        return server_root(self.ctx)

    # -- 8. exposure -------------------------------------------------------

    def create_admin_key(self, name: str) -> str | None:
        record, token = self._accounts().create_key(name, "admin", created_by=self.actor)
        self._audit("panel.key.create", f"{record.name} ({record.scope}) id={record.id}")
        return str(token)

    def apply_exposure(self, mode: str) -> dict[str, Any]:
        from studioforge.authkeys import is_loopback_host

        config = self.ctx.config
        if mode == "local":
            hosts = (config.server.host, config.gui.host, config.watchdog.host)
            if all(is_loopback_host(h) for h in hosts):
                return {"hosts": config.server.host, "tls": config.server.tls.mode}
            self._apply(
                {"server.host": "127.0.0.1", "gui.host": "127.0.0.1", "watchdog.host": "127.0.0.1"}
            )
            return {"hosts": "127.0.0.1", "tls": config.server.tls.mode, "restart_required": True}
        self._apply(
            {
                "server.host": "0.0.0.0",
                "gui.host": "0.0.0.0",
                "watchdog.host": "0.0.0.0",
                "server.tls.mode": "self_signed",
            }
        )
        return {"hosts": "0.0.0.0", "tls": "self_signed", "restart_required": True}


# ---------------------------------------------------------------------------
# Driving one step
# ---------------------------------------------------------------------------


def step_method(wizard: SetupWizard, key: str) -> Callable[[], StepResult]:
    return {
        "account": wizard.step_account,
        "hardware": wizard.step_hardware,
        "engine": wizard.step_engine,
        "library": wizard.step_library,
        "model": wizard.step_model,
        "chat": wizard.step_chat,
        "connect": wizard.step_connect,
        "exposure": wizard.step_exposure,
    }[key]


def drive_step(wizard: SetupWizard, key: str) -> StepResult:
    """Run one wizard step. A step that raises becomes a ``failed`` result, never an exception.

    Mirrors what :meth:`SetupWizard.run` does for each step, so a step run on its
    own behaves exactly as it does inside the full run.
    """
    index = [k for k, _ in STEPS].index(key)
    title = STEPS[index][1].split(". ", 1)[1]
    try:
        return step_method(wizard, key)()
    except Exception as exc:  # noqa: BLE001 - a failing step must not end the wizard
        result = StepResult(index + 1, key, title, "failed", error_text(exc))
        wizard.results.append(result)
        return result


def adopt_model(wizard: SetupWizard, model_id: str) -> None:
    """Tell the wizard which model the test chat should use (one loaded outside the wizard)."""
    wizard._model_id = model_id


def scrub_secret(result: StepResult, name: str = "admin_key") -> str | None:
    """Remove and return a secret from a step result's data."""
    value = result.data.pop(name, None)
    return str(value) if value else None


def starter_for(services: PanelServices) -> dict[str, Any]:
    """The suggested starter model for the smallest GPU."""
    try:
        gpus = services.detect_hardware().get("gpus") or []
        smallest = min(float(g.get("total_bytes", 0)) for g in gpus) / 1024**3 if gpus else 0.0
    except Exception:  # noqa: BLE001
        smallest = 0.0
    return pick_starter(smallest)


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def show_result(box: Any, result: StepResult) -> None:
    box.clear()
    icon, colour = _STATUS_STYLE.get(result.status, ("info", "info"))
    with box:
        status_row(icon, colour, f"{result.title}: {result.status}", result.detail)


def _nav(stepper: Any, *, back: bool = True, last: bool = False) -> None:
    with ui.stepper_navigation():
        if back:
            ui.button("Back", icon="arrow_back", on_click=stepper.previous).props("flat")
        if last:
            ui.button("Finish", icon="check", on_click=lambda: ui.navigate.to("/")).props(
                "color=primary"
            )
        else:
            ui.button("Skip", on_click=stepper.next).props("flat").tooltip(
                "Leave this step for later"
            )
            ui.button("Next", icon="arrow_forward", on_click=stepper.next).props("color=primary")


class WizardPage:
    """The stepper and the state the steps share (the wizard, its services, the loop)."""

    def __init__(self, ctx: GuiContext) -> None:
        self.ctx = ctx
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        self.services = PanelServices(ctx, loop=loop, actor=viewer_actor())
        self.options = WizardOptions(interactive=False, start_server=True)
        self.wizard = SetupWizard(self.services, self.options)
        self.stepper: Any = None

    async def run(self, key: str, what: str, *, message: str, box: Any) -> StepResult | None:
        """Gate, then run one step on a worker thread with a spinner; show the outcome."""
        page = _Page.capture()
        try:
            require_local_admin(self.ctx, what)
        except Exception as exc:  # noqa: BLE001
            notify_error(exc, what=what)
            return None
        self.services.actor = viewer_actor()
        with page, busy(message=message):
            result: StepResult = await run_blocking(drive_step, self.wizard, key)
        if page.alive:
            show_result(box, result)
        return result

    # -- steps -------------------------------------------------------------

    def step_account(self) -> None:
        has_owner = self.services.has_owner()
        ui.label(
            "An owner account lets you sign in to this panel and share the server on your "
            "network. You do not need one to use StudioForge on this computer only."
        ).classes("text-sm")
        box = ui.column().classes("w-full")
        if has_owner:
            with box:
                status_row("check_circle", "positive", "An owner account already exists")
        else:
            username = ui.input("Username", value="owner").props(
                'dense outlined autocomplete="off"'
            )
            password = ui.input(
                "Password (at least 10 characters)", password=True, password_toggle_button=True
            ).props('dense outlined autocomplete="new-password"')
            again = ui.input("Type the password again", password=True).props(
                'dense outlined autocomplete="new-password"'
            )

            async def create() -> None:
                if password.value != again.value:
                    ui.notify(
                        "The two passwords do not match. Type the same password in both boxes.",
                        type="warning",
                    )
                    return
                self.options.username = (username.value or "").strip()
                self.options.password = password.value
                self.options.skip_account = False
                try:
                    await self.run(
                        "account",
                        "create the owner account",
                        message="Creating the account…",
                        box=box,
                    )
                finally:
                    self.options.password = None
                    password.set_value("")
                    again.set_value("")

            ui.button("Create owner account", icon="person_add", on_click=create).props(
                "color=primary"
            )
        _nav(self.stepper, back=False)

    def step_hardware(self) -> None:
        ui.label(
            "StudioForge serves models from GPU memory. This looks for your graphics cards."
        ).classes("text-sm")
        box = ui.column().classes("w-full")

        async def detect() -> None:
            result = await self.run(
                "hardware", "detect hardware", message="Looking for GPUs…", box=box
            )
            if result is None:
                return
            with box:
                for gpu in result.data.get("gpus") or []:
                    gib = float(gpu.get("total_bytes", 0)) / 1024**3
                    ui.label(f"GPU {gpu.get('index')}: {gpu.get('name')} · {gib:.0f} GiB").classes(
                        "text-xs font-mono"
                    )

        ui.button("Detect hardware", icon="memory", on_click=detect).props("color=primary")
        _nav(self.stepper)

    def step_engine(self) -> None:
        state = self.services.engine_state()
        ui.label(
            "The engine is the program that actually runs models (llama.cpp). StudioForge "
            "downloads a build for your GPUs and checks that it starts."
        ).classes("text-sm")
        box = ui.column().classes("w-full")
        if state.get("active"):
            with box:
                status_row(
                    "check_circle" if state.get("smoke_tested") else "info",
                    "positive" if state.get("smoke_tested") else "info",
                    f"Engine {state['active']} is installed",
                    "It passed its start-up check."
                    if state.get("smoke_tested")
                    else "It has not been checked yet.",
                )
        else:
            ui.label(f"No engine yet. The planned build is {state.get('pinned_tag')}.").classes(
                "text-xs opacity-70"
            )
        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)
        phase = ui.label("").classes("text-xs opacity-70")

        async def install() -> None:
            progress.set_visibility(True)
            ticker = ui.timer(
                0.5,
                lambda: (
                    progress.set_value(float(self.services.progress.get("fraction", 0) or 0)),
                    phase.set_text(str(self.services.progress.get("phase", ""))),
                ),
            )
            try:
                await self.run(
                    "engine",
                    "install the engine",
                    message="Installing and testing the engine (this can take several minutes)…",
                    box=box,
                )
            finally:
                ticker.cancel()
                progress.set_visibility(False)
                phase.set_text("")

        ui.button("Install and test the engine", icon="download", on_click=install).props(
            "color=primary"
        )
        ui.label("This downloads several hundred megabytes.").classes("text-xs opacity-70")
        _nav(self.stepper)

    def step_library(self) -> None:
        ui.label(
            "A model library is a folder of GGUF model files. You can use the one LM Studio "
            "already made. Nothing is copied or moved."
        ).classes("text-sm")
        configured = self.ctx.config.models.dir
        ui.label(f"Current library: {configured or 'none set'}").classes("text-xs font-mono")
        box = ui.column().classes("w-full")
        choices = ui.column().classes("w-full gap-1")
        custom = ui.input("Or type a folder path").props('dense outlined autocomplete="off"')
        chosen: dict[str, str] = {"path": ""}

        async def look() -> None:
            found: list[dict[str, Any]] = await run_blocking(self.services.find_libraries)
            choices.clear()
            with choices:
                if not found:
                    ui.label("No model library was found on this computer.").classes("text-xs")
                    return
                options = {
                    str(f["path"]): f"{f['path']} · {f.get('models', '?')} model file(s)"
                    f" ({'current' if f.get('source') == 'configured' else f.get('source')})"
                    for f in found
                }
                chosen["path"] = next(iter(options))
                ui.radio(
                    options, value=chosen["path"], on_change=lambda e: chosen.update(path=e.value)
                ).props("dense")

        async def use() -> None:
            path = (custom.value or "").strip() or chosen["path"]
            if not path:
                ui.notify("Look for libraries and pick one, or type a folder path.", type="warning")
                return
            self.options.library = path
            await self.run(
                "library", "choose the model library", message="Saving the library…", box=box
            )

        with ui.row().classes("gap-2"):
            ui.button("Look for model libraries", icon="search", on_click=look).props("outline")
            ui.button("Use this library", icon="folder", on_click=use).props("color=primary")
        _nav(self.stepper)

    def step_model(self) -> None:
        starter = starter_for(self.services)
        ui.label(
            "Pick a model you already have, or download a small one to start with. It is loaded "
            "so the next step can talk to it."
        ).classes("text-sm")
        box = ui.column().classes("w-full")
        models = [m["id"] for m in self.services.library_models()]
        mode = {"value": "have" if models else "download"}
        have = ui.select(
            models, value=models[0] if models else None, label="A model I already have"
        )
        have.props("dense outlined")
        repo = ui.input(f"Or download ({starter['note']})", value=str(starter["repo"])).props(
            'dense outlined autocomplete="off"'
        )
        choice = ui.radio(
            {"have": "Use the model I already have", "download": "Download the model above"},
            value=mode["value"],
            on_change=lambda e: mode.update(value=e.value),
        ).props("dense")
        choice.set_enabled(bool(models))

        def refresh_list() -> None:
            found = [m["id"] for m in self.services.library_models()]
            have.set_options(found, value=found[0] if found else None)
            choice.set_enabled(bool(found))
            if found:
                choice.set_value("have")

        ui.button("Refresh the list", icon="refresh", on_click=refresh_list).props("flat dense")

        progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
        progress.set_visibility(False)
        phase = ui.label("").classes("text-xs opacity-70")

        async def go() -> None:
            wanted = str(have.value) if (mode["value"] == "have" and have.value) else repo.value
            self.options.model = (wanted or "").strip()
            self.options.skip_model = False
            progress.set_visibility(True)
            ticker = ui.timer(
                1.0,
                lambda: (
                    progress.set_value(float(self.services.progress.get("fraction", 0) or 0)),
                    phase.set_text(str(self.services.progress.get("phase", ""))),
                ),
            )
            try:
                await self.run(
                    "model",
                    "download or load a model",
                    message="Getting the model ready (a download can take a while)…",
                    box=box,
                )
            finally:
                ticker.cancel()
                progress.set_visibility(False)
                phase.set_text("")

        def stop() -> None:
            self.services.cancelled.set()
            ui.notify("Stopping the download…")

        with ui.row().classes("gap-2"):
            ui.button("Get this model ready", icon="play_arrow", on_click=go).props("color=primary")
            ui.button("Stop the download", icon="stop", on_click=stop).props("flat")
        _nav(self.stepper)

    def step_chat(self) -> None:
        ui.label(
            "Sends one short message to the loaded model and shows how quickly it answers."
        ).classes("text-sm")
        box = ui.column().classes("w-full")
        reply = ui.label("").classes("text-sm font-mono whitespace-pre-wrap")

        async def send() -> None:
            if self.wizard._model_id is None:
                loaded = self._loaded_model()
                if loaded is None:
                    ui.notify(
                        "No model is loaded yet. Go back one step and get a model ready.",
                        type="warning",
                    )
                    return
                adopt_model(self.wizard, loaded)
            self.options.skip_chat = False
            result = await self.run("chat", "run a test chat", message="Asking the model…", box=box)
            if result is not None and result.data.get("reply"):
                reply.set_text(f"Reply: {result.data['reply']}")

        ui.button("Send a test message", icon="chat", on_click=send).props("color=primary")
        _nav(self.stepper)

    def _loaded_model(self) -> str | None:
        try:
            loaded = self.ctx.supervisor.list() if self.ctx.supervisor is not None else []
            return str(loaded[0].model_id) if loaded else None
        except Exception:  # noqa: BLE001
            return None

    def step_connect(self) -> None:
        ui.label(
            "The address other programs use, and ready-to-paste settings for the common ones."
        ).classes("text-sm")
        box = ui.column().classes("w-full gap-2")

        def show() -> None:
            box.clear()
            result = drive_step(self.wizard, "connect")
            with box:
                snippets: dict[str, str] = result.data.get("snippets") or build_snippets(
                    self.services.base_url()
                )
                code_block("Base URL", snippets["base_url"])
                for name, title in (
                    ("openclaw", "OpenClaw"),
                    ("openai_sdk", "OpenAI Python SDK"),
                    ("curl", "curl"),
                    ("mcp_url", "MCP address (for agents)"),
                ):
                    code_block(title, snippets[name])
                ui.label(
                    "Replace the key placeholder with a key from the Users & Keys page once you "
                    "require sign-in."
                ).classes("text-xs opacity-70")

        ui.button("Show the connection details", icon="link", on_click=show).props("color=primary")
        _nav(self.stepper)

    def step_exposure(self) -> None:
        ui.label("Who should be able to reach this server?").classes("text-sm")
        box = ui.column().classes("w-full")
        mode = {"value": "local"}
        ui.radio(
            {
                "local": "Only this computer (recommended)",
                "lan": "My local network, over HTTPS (needs the owner account from step 1)",
            },
            value="local",
            on_change=lambda e: mode.update(value=e.value),
        ).props("dense")
        ui.checkbox(PEER_LABEL).props("disable dense")

        async def apply() -> None:
            self.options.exposure = mode["value"]
            result = await self.run(
                "exposure", "change who can reach the server", message="Saving…", box=box
            )
            if result is None:
                return
            key = scrub_secret(result)
            if key:
                one_time_secret_dialog(
                    "Your admin key",
                    key,
                    hint="Use this key to sign in from another computer, or give it to a program "
                    "that needs full access.",
                )
            if result.status == "done" and mode["value"] == "lan":
                with box:
                    ui.label(
                        "Restart required: the server listens on the network and uses HTTPS "
                        "after it restarts."
                    ).classes("text-sm text-warning")

        ui.button("Apply", icon="public", on_click=apply).props("color=primary")
        _nav(self.stepper, last=True)

    # -- layout ------------------------------------------------------------

    def render(self) -> None:
        builders = [
            ("account", "person", self.step_account),
            ("hardware", "memory", self.step_hardware),
            ("engine", "settings_suggest", self.step_engine),
            ("library", "folder", self.step_library),
            ("model", "psychology", self.step_model),
            ("chat", "chat", self.step_chat),
            ("connect", "link", self.step_connect),
            ("exposure", "public", self.step_exposure),
        ]
        titles = dict(STEPS)
        with ui.stepper().props("vertical header-nav").classes("w-full") as stepper:
            self.stepper = stepper
            for key, icon, build in builders:
                with ui.step(titles[key], icon=icon), panel_guard(titles[key]):
                    build()


def render(ctx: GuiContext) -> None:
    """Build the ``/setup`` page."""
    ui.page_title("StudioForge setup")
    with ui.header().classes("items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("memory", size="1.6rem")
            ui.label("StudioForge setup").classes("text-lg font-semibold")
        ui.link("Back to the panel", "/").classes("text-sm")
    with ui.column().classes("w-full max-w-4xl mx-auto gap-3 p-4"):
        ui.label("Get StudioForge ready in eight steps").classes("text-xl font-medium")
        ui.label(
            "Every step is optional and you can come back to this page any time. Nothing runs "
            "until you press a button."
        ).classes("text-sm opacity-70")
        if not _wizard_allowed(ctx):
            ui.label(
                "Setup changes the server, so it only works from a browser on this computer or "
                "after you sign in. Open this page on the machine that runs StudioForge."
            ).classes("text-sm text-warning")
        WizardPage(ctx).render()


def _wizard_allowed(ctx: GuiContext) -> bool:
    from studioforge.gui.tabs import viewer_may_change_box

    return viewer_may_change_box(ctx)


__all__ = [
    "STEPS",
    "PanelServices",
    "WizardPage",
    "adopt_model",
    "drive_step",
    "render",
    "scrub_secret",
    "starter_for",
]
