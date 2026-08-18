"""StudioForge system-tray application.

Supervises ``python -m studioforge serve`` as a *child process* rather than
merely attaching to one that happens to be up. That matters more here than in a
normal tray: the server spawns ``llama-server`` grandchildren that each hold
gigabytes of VRAM, so quitting the tray has to be able to take the whole tree
down. Owning the child is what makes that possible (see :meth:`TrayApp.stop_server`).

Everything that can be exercised without a desktop session lives in module-level
pure functions at the top -- icon drawing, URL building, status strings, the
single-instance guard. :class:`TrayApp` holds the stateful bits and can be built
with ``create_icon=False`` so tests never touch the Win32 message pump.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

import pystray  # type: ignore[import-untyped]
from PIL import Image, ImageDraw, ImageFont

from studioforge.config import Config
from studioforge.core import autostart
from studioforge.core.engine import kill_process_tree
from studioforge.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "StudioForge"

SINGLE_INSTANCE_HOST = "127.0.0.1"
#: Abstract mutex: a bound socket nobody else can take while we are alive.
#: Deliberately *not* ClawForge's 47821 -- both trays are expected to run side
#: by side on this machine, and a shared port would make each one look to the
#: other like "already running".
SINGLE_INSTANCE_PORT = 47823

POLL_INTERVAL = 2.0  # seconds between child-process liveness polls
STATUS_INTERVAL = 5.0  # seconds between /api/status refreshes
MAX_RESTARTS = 3  # unexpected-exit restarts before giving up
RESTART_BACKOFF = 5.0  # seconds to wait before each restart
HEALTHY_AFTER = 60.0  # child alive this long => reset the restart counter
STOP_TIMEOUT = 20.0  # grace given to the tree before SIGKILL/TerminateProcess

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

STATE_STOPPED = "stopped"
STATE_STARTING = "starting"
STATE_RUNNING = "running"
STATE_CRASHED = "crashed"

#: States in which the server can be asked to start.
DOWN_STATES = (STATE_STOPPED, STATE_CRASHED)
#: States in which a server process exists (or is on its way up).
LIVE_STATES = (STATE_RUNNING, STATE_STARTING)

# A cool blue/cyan rather than ClawForge's forge-orange: the two trays sit next
# to each other in the same notification area and have to stay tellable apart
# at 16 pixels.
SF_CYAN = (56, 189, 248, 255)
SF_BLUE = (59, 130, 246, 255)
ICON_BG = (15, 23, 42, 255)
DOT_GREEN = (64, 200, 96, 255)
DOT_GRAY = (130, 134, 142, 255)


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable, no display required)
# ---------------------------------------------------------------------------


def _console_python() -> str:
    """``python.exe`` even when the tray itself is running under ``pythonw.exe``.

    The launcher starts the tray windowless, so ``sys.executable`` is the
    console-less build; a child inheriting it gets ``sys.stdout is None``, which
    breaks anything the server prints before logging is configured. The console
    interpreter is the same environment, just with working std handles.
    """
    exe = Path(sys.executable)
    if exe.name.lower().startswith("pythonw"):
        candidate = exe.with_name(exe.name.lower().replace("pythonw", "python", 1))
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def server_command(config: Config) -> list[str]:
    """Argv that starts the server, mirroring :func:`core.autostart.launch_command`.

    The config path is passed explicitly for the same reason it is there: the
    tray may have been started from a shortcut or the Startup folder, neither of
    which carries ``SF_DATA_DIR``.
    """
    return [
        _console_python(),
        "-m",
        "studioforge",
        "serve",
        "--config",
        str(config.config_path),
    ]


def _browsable_host(host: str) -> str:
    """Turn a *bind* address into one a browser on this machine can open.

    ``0.0.0.0`` means "every interface", which is not an address a client can
    connect to. The tray and the browser it launches both live on the server
    box, so loopback is the honest answer for a wildcard bind.
    """
    return "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host


def api_base_url(config: Config) -> str:
    """Root of the local API, derived from ``server.host`` / ``server.port``."""
    return f"http://{_browsable_host(config.server.host)}:{config.server.port}"


def api_docs_url(config: Config) -> str:
    """The FastAPI docs page of the running gateway."""
    return f"{api_base_url(config)}/docs"


def control_panel_url(config: Config) -> str:
    """Browsable URL of the GUI.

    Delegates to the CLI's own helper so the tray, ``studioforge gui`` and
    ``serve --open`` can never disagree about where the control panel is.
    """
    from studioforge.__main__ import gui_url

    return gui_url(config)


def fallback_mcp_url(config: Config) -> str:
    """Where MCP lives when the server is not up to tell us itself.

    Tailscale-first, matching what ``GET /api/mcp/info`` would have answered: a
    tailnet address survives a network change where a LAN address quietly stops
    resolving.
    """
    from studioforge.core.netinfo import primary_url

    return primary_url(config.server.port, config.mcp.path, host=config.server.host)


def models_folder(config: Config) -> Path:
    """The directory to reveal for "Open models folder"."""
    dirs = config.model_dirs()
    return dirs[0] if dirs else config.data_dir / "models"


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / 2**30:.1f} GiB"


@dataclass(frozen=True)
class ServerSnapshot:
    """The bit of ``GET /api/status`` the status line needs."""

    reachable: bool = False
    loaded_models: int = 0
    free_vram_bytes: int = 0


def snapshot_from_status(data: Mapping[str, Any]) -> ServerSnapshot:
    """Reduce a ``/api/status`` body to :class:`ServerSnapshot`.

    Defensive about shapes: a status payload that grew a field must never be
    able to blank out the tray's only visible readout.
    """
    loaded = data.get("loaded")
    count = len(loaded) if isinstance(loaded, list) else 0
    free = 0
    gpus = data.get("gpus")
    if isinstance(gpus, list):
        for gpu in gpus:
            if isinstance(gpu, Mapping):
                value = gpu.get("free_bytes")
                if isinstance(value, int | float):
                    free += int(value)
    return ServerSnapshot(reachable=True, loaded_models=count, free_vram_bytes=free)


def status_text(state: str, snapshot: ServerSnapshot) -> str:
    """Text of the disabled header item, and of the icon tooltip.

    Free VRAM and the loaded-model count are the two numbers that decide
    whether the next load will fit, so they are what the one always-visible
    line spends its space on.
    """
    if state == STATE_RUNNING:
        if not snapshot.reachable:
            return "Running — API not answering yet"
        plural = "" if snapshot.loaded_models == 1 else "s"
        return (
            f"Running — {snapshot.loaded_models} model{plural} loaded, "
            f"{format_gib(snapshot.free_vram_bytes)} free"
        )
    if state == STATE_STARTING:
        return "Starting..."
    if state == STATE_CRASHED:
        return "Crashed — see the logs folder"
    return "Stopped"


def acquire_single_instance(
    port: int = SINGLE_INSTANCE_PORT, host: str = SINGLE_INSTANCE_HOST
) -> socket.socket | None:
    """Bind the mutex socket; ``None`` when another instance already holds it.

    ``SO_REUSEADDR`` is deliberately *not* set: on Windows it would let a second
    process bind the same address and defeat the whole guard.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.listen(1)
    except OSError:
        with contextlib.suppress(OSError):
            sock.close()
        return None
    return sock


def _load_font(size: int) -> Any:
    for name in ("arialbd.ttf", "seguisb.ttf", "segoeuib.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def make_icon_image(running: bool, size: int = 64) -> Image.Image:
    """Draw the tray icon: dark rounded square, an ``SF`` glyph, status dot.

    Generated with PIL rather than shipped as a binary ``.ico`` so the package
    stays pure Python and the icon can change with the state. The dot is the
    only part that carries information -- green when the server is running,
    grey when it is not -- so it is drawn large and in the corner, where it
    survives being scaled down to 16px.
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = max(1, size // 16)
    radius = max(3, size // 5)
    draw.rounded_rectangle(
        [pad, pad, size - pad - 1, size - pad - 1],
        radius=radius,
        fill=ICON_BG,
        outline=SF_CYAN,
        width=max(2, size // 24),
    )

    # Accent bar under the glyph, kept clear of the status dot's corner.
    bar_h = max(2, size // 12)
    draw.rounded_rectangle(
        [size * 0.22, size * 0.70, size * 0.60, size * 0.70 + bar_h],
        radius=bar_h // 2,
        fill=SF_BLUE,
    )

    font = _load_font(max(8, int(size * 0.42)))
    try:
        draw.text((size / 2, size * 0.44), "SF", font=font, fill=SF_CYAN, anchor="mm")
    except (ValueError, TypeError):  # bitmap fallback font: no anchor support
        draw.text((size * 0.22, size * 0.26), "SF", font=font, fill=SF_CYAN)

    r = size * 0.185
    cx = size - pad - r * 0.95
    cy = size - pad - r * 0.95
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=DOT_GREEN if running else DOT_GRAY,
        outline=ICON_BG,
        width=max(1, size // 32),
    )
    return img


def copy_to_clipboard(text: str) -> bool:
    """Put *text* on the Windows clipboard using stdlib tkinter, then ``clip``."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()  # let Tk hand the selection to the window manager
        root.update()
        root.destroy()
        return True
    except Exception:  # pragma: no cover - depends on the desktop session
        log.warning("tkinter clipboard failed, trying clip.exe")
    if os.name == "nt":  # pragma: no cover - Windows shell fallback
        try:
            subprocess.run(
                ["cmd", "/c", "clip"],
                input=text.encode("utf-8"),
                check=True,
                creationflags=CREATE_NO_WINDOW,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            log.warning("clip.exe clipboard fallback failed")
    return False


def show_already_running() -> None:  # pragma: no cover - needs a desktop session
    """Tell the user why the second launch did nothing, then go away."""
    try:
        import tkinter
        from tkinter import messagebox

        root = tkinter.Tk()
        root.withdraw()
        messagebox.showinfo(APP_NAME, f"{APP_NAME} is already in your system tray.")
        root.destroy()
    except Exception:
        log.warning("another tray instance is already running")


# ---------------------------------------------------------------------------
# API access
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiResult:
    """Outcome of one API call. ``error`` is safe to show the user."""

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class ApiClient(Protocol):
    """What :class:`TrayApp` needs from an HTTP client, so tests can fake it."""

    def get(self, path: str, *, timeout: float = 10.0) -> ApiResult: ...

    def post(
        self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0
    ) -> ApiResult: ...


def _http_error_detail(exc: urlerror.HTTPError) -> str:
    """A sentence from a StudioForge error body, or the bare status code."""
    try:
        payload = json.loads(exc.read().decode("utf-8", "replace"))
    except Exception:
        return f"server returned HTTP {exc.code}"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(error, str) and error:
        return error
    return f"server returned HTTP {exc.code}"


class HttpApiClient:
    """Stdlib HTTP client for the local StudioForge API.

    ``urllib`` rather than ``httpx``: every call runs on a short-lived worker
    thread off the tray message pump, where there is no event loop to hang an
    async client on, and spinning one up to POST forty bytes would be silly.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    @property
    def base_url(self) -> str:
        return api_base_url(self._config)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._config.server.api_key
        if key:
            # Header only. The API key must never reach a URL, a log line, a
            # notification balloon or the clipboard.
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _call(
        self, method: str, path: str, payload: dict[str, Any] | None, timeout: float
    ) -> ApiResult:
        url = self.base_url.rstrip("/") + path
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urlrequest.Request(url, data=body, method=method, headers=self._headers())
        try:
            with urlrequest.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", "replace")
        except urlerror.HTTPError as exc:
            return ApiResult(False, error=_http_error_detail(exc))
        except (OSError, ValueError) as exc:
            return ApiResult(False, error=f"the server is not answering ({exc})")
        try:
            data = json.loads(raw or "{}")
        except ValueError:
            return ApiResult(False, error="the server sent a reply that was not JSON")
        if not isinstance(data, dict):
            return ApiResult(False, error="unexpected reply from the server")
        return ApiResult(True, data)

    def get(self, path: str, *, timeout: float = 10.0) -> ApiResult:
        return self._call("GET", path, None, timeout)

    def post(
        self, path: str, payload: dict[str, Any] | None = None, *, timeout: float = 60.0
    ) -> ApiResult:
        # An empty body still has to be ``{}`` rather than nothing: the restart
        # endpoint reads ``confirm`` out of an embedded JSON body.
        return self._call("POST", path, payload or {}, timeout)


# ---------------------------------------------------------------------------
# The tray application
# ---------------------------------------------------------------------------


class TrayApp:
    """pystray icon plus a supervised ``studioforge serve`` child process."""

    def __init__(
        self,
        config: Config,
        *,
        client: ApiClient | None = None,
        guard: socket.socket | None = None,
        create_icon: bool = True,
    ) -> None:
        self.config = config
        self.client: ApiClient = client if client is not None else HttpApiClient(config)
        self.state = STATE_STOPPED
        self.snapshot = ServerSnapshot()
        self.proc: subprocess.Popen[bytes] | None = None
        #: False when we merely attached to a server someone else started; the
        #: tray must not kill a process it did not launch.
        self.owns_child = False
        self.user_stopped = True
        self.restarts = 0

        self._guard = guard
        self._spawn_time = 0.0
        self._last_status_poll = 0.0
        self._log_handle: Any = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._supervisor: threading.Thread | None = None
        self.icon: Any = None
        if create_icon:
            self.icon = pystray.Icon(
                APP_NAME.lower(),
                icon=make_icon_image(False),
                title=APP_NAME,
                menu=self._build_menu(),
            )

    # ---- child process ---------------------------------------------------

    def _open_server_log(self) -> Any:
        self.config.logs_dir.mkdir(parents=True, exist_ok=True)
        return (self.config.logs_dir / "tray-server.log").open("ab", buffering=0)

    def _close_log(self) -> None:
        if self._log_handle is not None:
            with contextlib.suppress(OSError):
                self._log_handle.close()
            self._log_handle = None

    def _spawn(self) -> bool:
        """Launch the server child. The caller holds the lock."""
        self._close_log()
        try:
            handle = self._open_server_log()
        except OSError as exc:
            log.error("cannot open the tray server log", error=str(exc))
            self.state = STATE_CRASHED
            return False
        try:
            self.proc = subprocess.Popen(
                server_command(self.config),
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
        except OSError as exc:
            log.error("failed to start the server child process", error=str(exc))
            with contextlib.suppress(OSError):
                handle.close()
            self.state = STATE_CRASHED
            return False
        self._log_handle = handle
        self._spawn_time = time.monotonic()
        self.owns_child = True
        self.state = STATE_STARTING
        log.info("tray started the server", pid=self.proc.pid)
        return True

    def start_server(self) -> None:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.user_stopped = False
            self.restarts = 0
            self._spawn()
        self._refresh()

    def stop_server(self, timeout: float = STOP_TIMEOUT) -> None:
        """Stop the server *and every descendant*.

        A plain ``terminate()`` on the gateway leaves its ``llama-server``
        grandchildren running, and each of those keeps its model resident in
        VRAM -- the exact orphan this project exists to prevent. So the tree
        goes first and the parent's exit is only what we wait for.
        """
        with self._lock:
            self.user_stopped = True
            proc = self.proc
            owned = self.owns_child
            self.proc = None

        if proc is None or proc.poll() is not None:
            with self._lock:
                self._close_log()
                if owned or self.state != STATE_RUNNING:
                    self.state = STATE_STOPPED
            self._refresh()
            return

        log.info("tray stopping the server tree", pid=proc.pid)
        kill_process_tree(proc.pid, timeout=timeout)
        try:
            proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, OSError):
            log.warning("server process did not report an exit", pid=proc.pid)

        with self._lock:
            self._close_log()
            self.owns_child = False
            self.state = STATE_STOPPED
            self.snapshot = ServerSnapshot()
        self._refresh()

    def restart_server(self) -> None:
        """Restart the whole server process.

        When the tray owns the child this is a local stop-then-start.
        ``POST /api/restart/server`` deliberately respawns the server
        *detached* (or hands off to the watchdog) and then exits, which would
        leave the tray supervising a dead pid next to a second server it can no
        longer stop. The API is the right lever only when the server is not
        ours to restart -- so that is exactly when we reach for it.
        """
        if self.owns_child:
            self.stop_server()
            self.start_server()
            return
        result = self.api_restart_server()
        if result.ok:
            self._notify("Restarting the server; it will be back in a few seconds.")
        else:
            self._notify(f"Restart failed: {result.error}")

    # ---- API actions (each returns the raw result so tests can assert) ----

    def api_restart_server(self) -> ApiResult:
        """``POST /api/restart/server`` with ``confirm`` ALWAYS true.

        The endpoint refuses without it, on purpose -- but from a menu item
        that refusal is indistinguishable from a broken menu item, so the
        confirmation is expressed by the user picking the item at all.
        """
        return self.client.post("/api/restart/server", {"confirm": True}, timeout=60.0)

    def unload_all_models(self) -> ApiResult:
        """``POST /api/models/unload-all`` -- the headline VRAM action."""
        return self.client.post("/api/models/unload-all", timeout=180.0)

    def restart_engines(self) -> ApiResult:
        """``POST /api/restart/backend`` -- reload the llama-server children.

        Distinct from :meth:`restart_server`: the API stays up throughout, only
        the inference processes are recycled.
        """
        return self.client.post("/api/restart/backend", timeout=600.0)

    def mcp_info(self) -> tuple[str, str | None]:
        """``(url, pin)`` from the running server, falling back to the config."""
        result = self.client.get("/api/mcp/info")
        if result.ok:
            url = result.data.get("recommended")
            pin = result.data.get("pin")
            if isinstance(url, str) and url:
                return url, pin if isinstance(pin, str) and pin else None
        return fallback_mcp_url(self.config), self.config.mcp.pin

    # ---- supervision -----------------------------------------------------

    def _poll_status(self) -> None:
        """Refresh the status line from ``/api/status`` (never raises)."""
        result = self.client.get("/api/status", timeout=5.0)
        with self._lock:
            self.snapshot = (
                snapshot_from_status(result.data) if result.ok else ServerSnapshot(reachable=False)
            )

    def _server_is_answering(self) -> bool:
        return self.client.get("/health", timeout=3.0).ok

    def adopt_running_server(self) -> bool:
        """Attach to a server that is already up instead of starting a rival.

        Starting a second one would just fail the port preflight and look like
        a crash, so if the port already answers we mark ourselves running but
        NOT owning -- which keeps Quit from killing a process we did not start.
        """
        if not self._server_is_answering():
            return False
        with self._lock:
            self.state = STATE_RUNNING
            self.owns_child = False
            self.user_stopped = False
        self._poll_status()
        self._refresh()
        return True

    def _supervise(self) -> None:
        """Watch the child, reflect crashes, and keep the status line fresh."""
        while not self._stop_event.is_set():
            self._stop_event.wait(POLL_INTERVAL)
            if self._stop_event.is_set():
                break

            restart_needed = False
            changed = False
            with self._lock:
                proc = self.proc
                if proc is None:
                    changed = self._poll_unowned_locked()
                else:
                    rc = proc.poll()
                    if rc is None:
                        if self.state == STATE_STARTING and self._server_is_answering():
                            self.state = STATE_RUNNING
                            changed = True
                        if self.restarts and time.monotonic() - self._spawn_time > HEALTHY_AFTER:
                            self.restarts = 0
                    else:
                        self.proc = None
                        self.owns_child = False
                        self._close_log()
                        self.snapshot = ServerSnapshot()
                        changed = True
                        if self.user_stopped:
                            self.state = STATE_STOPPED
                        elif self.restarts < MAX_RESTARTS:
                            self.restarts += 1
                            log.error(
                                "server exited unexpectedly; restarting",
                                returncode=rc,
                                attempt=self.restarts,
                                of=MAX_RESTARTS,
                            )
                            self.state = STATE_STARTING
                            restart_needed = True
                        else:
                            log.error("server keeps dying; giving up", returncode=rc)
                            self.state = STATE_CRASHED
                            self.user_stopped = True

            if self.state == STATE_RUNNING and (
                time.monotonic() - self._last_status_poll > STATUS_INTERVAL
            ):
                self._last_status_poll = time.monotonic()
                self._poll_status()
                changed = True
            if changed:
                self._refresh()
            if restart_needed:
                self._notify("The server exited unexpectedly; restarting it.")
                self._stop_event.wait(RESTART_BACKOFF)
                if self._stop_event.is_set():
                    break
                with self._lock:
                    if not self.user_stopped and self.proc is None:
                        self._spawn()
                self._refresh()

    def _poll_unowned_locked(self) -> bool:
        """Track a server we attached to rather than launched. Lock held."""
        if self.state not in LIVE_STATES:
            return False
        if self.owns_child:
            return False
        if self._server_is_answering():
            return False
        self.state = STATE_STOPPED
        self.snapshot = ServerSnapshot()
        return True

    # ---- icon / menu -----------------------------------------------------

    def _refresh(self) -> None:
        icon = self.icon
        if icon is None:
            return
        try:
            icon.icon = make_icon_image(self.state == STATE_RUNNING)
            icon.title = f"{APP_NAME} - {status_text(self.state, self.snapshot)}"
            icon.update_menu()
        except Exception as exc:  # pragma: no cover - backend specific
            log.warning("could not refresh the tray icon", error=str(exc))

    def _notify(self, message: str, title: str = APP_NAME) -> None:
        """Balloon notification, degrading to a log line when unsupported."""
        try:
            if self.icon is not None and getattr(self.icon, "HAS_NOTIFICATION", False):
                self.icon.notify(message, title)
                return
        except Exception as exc:  # pragma: no cover - backend specific
            log.warning("notification failed", error=str(exc))
        log.info("tray notification", title=title, message=message)

    def autostart_enabled(self) -> bool:
        """Whether login-autostart is on, per :mod:`studioforge.core.autostart`."""
        try:
            return autostart.status(self.config).enabled
        except Exception as exc:  # pragma: no cover - platform specific
            log.warning("could not read the autostart status", error=str(exc))
            return False

    def _build_menu(self) -> Any:
        item = pystray.MenuItem
        sep = pystray.Menu.SEPARATOR
        return pystray.Menu(
            # The one always-visible line: state, resident models, free VRAM.
            item(lambda _i: status_text(self.state, self.snapshot), None, enabled=False),
            sep,
            # default=True is what a LEFT click on the icon invokes.
            item("Open control panel", self._on_open_control_panel, default=True),
            item("Open API docs", self._on_open_api_docs),
            item("Open logs folder", self._on_open_logs),
            item("Open models folder", self._on_open_models),
            sep,
            item(
                "Unload all models (free VRAM)",
                self._on_unload_all,
                enabled=lambda _i: self.state == STATE_RUNNING,
            ),
            item(
                "Restart engines (reload models, API stays up)",
                self._on_restart_engines,
                enabled=lambda _i: self.state == STATE_RUNNING,
            ),
            sep,
            item("Start server", self._on_start, enabled=lambda _i: self.state in DOWN_STATES),
            item("Stop server", self._on_stop, enabled=lambda _i: self.state in LIVE_STATES),
            item(
                "Restart server (whole process)",
                self._on_restart_server,
                enabled=lambda _i: self.state in LIVE_STATES,
            ),
            sep,
            item("Copy MCP URL", self._on_copy_mcp_url),
            item("Copy MCP PIN", self._on_copy_mcp_pin),
            sep,
            item(
                "Start at login",
                self._on_toggle_autostart,
                checked=lambda _i: self.autostart_enabled(),
            ),
            sep,
            item("Quit", self._on_quit),
        )

    # ---- menu handlers ---------------------------------------------------

    def _spawn_thread(self, fn: Callable[[], None]) -> None:
        """Run a handler off the tray message-pump thread.

        pystray dispatches menu callbacks on the thread that owns the Win32
        message loop; anything slow done there (a 600s engine restart, killing
        a process tree) freezes the menu and the icon until it finishes.
        """
        threading.Thread(
            target=fn, daemon=True, name=f"studioforge-tray-{getattr(fn, '__name__', 'action')}"
        ).start()

    def _open_path(self, path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(str(path))
            else:  # pragma: no cover - the tray targets Windows
                subprocess.Popen(["xdg-open", str(path)])
        except (OSError, AttributeError) as exc:
            self._notify(f"Cannot open {path}: {exc}")

    def _open_browser(self, url: str) -> None:
        import webbrowser

        if not webbrowser.open(url):
            self._notify(f"Could not open a browser - {url}")

    def _on_open_control_panel(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(lambda: self._open_browser(control_panel_url(self.config)))

    def _on_open_api_docs(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(lambda: self._open_browser(api_docs_url(self.config)))

    def _on_open_logs(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(lambda: self._open_path(self.config.logs_dir))

    def _on_open_models(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(lambda: self._open_path(models_folder(self.config)))

    def _on_start(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(self.start_server)

    def _on_stop(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            # `owns_child` is cleared the moment a supervised child exits, so
            # during a crash-restart window it is False even though the tray is
            # about to respawn. Refusing there told the user we did not start
            # this server AND left `user_stopped` False, so the supervisor
            # respawned anyway -- a crash-looping server could not be stopped
            # from the menu. Anything the tray is actively supervising is
            # stoppable; `stop_server` sets `user_stopped`, which is what
            # actually suppresses the respawn, and it handles a dead process.
            supervising = self.owns_child or self.state in LIVE_STATES
            if not supervising:
                self._notify(
                    "This server was not started by the tray, so the tray will not "
                    "stop it. Stop it where you started it."
                )
                return
            self.stop_server()
            self._notify("Server stopped; all VRAM released.")

        self._spawn_thread(work)

    def _on_restart_server(self, _icon: Any = None, _item: Any = None) -> None:
        self._spawn_thread(self.restart_server)

    def _on_unload_all(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            result = self.unload_all_models()
            if result.ok:
                count = result.data.get("count", 0)
                self._notify(f"Unloaded {count} model(s); VRAM freed.")
            else:
                self._notify(f"Unload failed: {result.error}")
            self._poll_status()
            self._refresh()

        self._spawn_thread(work)

    def _on_restart_engines(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            result = self.restart_engines()
            if result.ok:
                count = result.data.get("count", 0)
                failed = result.data.get("failed") or []
                suffix = f", {len(failed)} failed" if failed else ""
                self._notify(f"Reloaded {count} engine process(es){suffix}.")
            else:
                self._notify(f"Engine restart failed: {result.error}")
            self._poll_status()
            self._refresh()

        self._spawn_thread(work)

    def _on_copy_mcp_url(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            url, _pin = self.mcp_info()
            ok = copy_to_clipboard(url)
            self._notify(f"Copied {url}" if ok else f"Clipboard failed - {url}")

        self._spawn_thread(work)

    def _on_copy_mcp_pin(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            _url, pin = self.mcp_info()
            if not pin:
                self._notify("This server does not require an MCP PIN.")
                return
            # The PIN is the only credential the tray ever puts on the
            # clipboard, and only because the user asked for it by name. It is
            # still kept out of the balloon text.
            ok = copy_to_clipboard(pin)
            self._notify(
                "MCP PIN copied to the clipboard."
                if ok
                else "Clipboard failed - read the PIN from the startup banner."
            )

        self._spawn_thread(work)

    def _on_toggle_autostart(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            try:
                if self.autostart_enabled():
                    result = autostart.disable(self.config)
                else:
                    # tray=True, or the icon the user just toggled on would be
                    # absent at the next login: a bare `serve` shim starts the
                    # server headless and there is nothing in the tray.
                    result = autostart.enable(self.config, tray=True)
            except Exception as exc:
                self._notify(f"Could not change the login entry: {exc}")
                return
            self._notify(result.describe())
            self._refresh()

        self._spawn_thread(work)

    def _on_quit(self, _icon: Any = None, _item: Any = None) -> None:
        def work() -> None:
            log.info("tray quitting")
            self._stop_event.set()
            self.stop_server()
            if self._supervisor is not None and self._supervisor.is_alive():
                self._supervisor.join(timeout=5)
            self._release_guard()
            try:
                if self.icon is not None:
                    self.icon.stop()
            except Exception as exc:  # pragma: no cover - backend specific
                log.warning("icon.stop() failed", error=str(exc))

        self._spawn_thread(work)

    # ---- lifecycle -------------------------------------------------------

    def _release_guard(self) -> None:
        if self._guard is not None:
            with contextlib.suppress(OSError):
                self._guard.close()
            self._guard = None

    def _setup(self, icon: Any) -> None:  # pragma: no cover - needs a session
        """Runs on a pystray-managed thread once the message loop is up."""
        icon.visible = True
        if self.adopt_running_server():
            self._notify("Attached to the StudioForge server already running here.")
            return
        self.start_server()

    def run(self) -> int:  # pragma: no cover - needs a desktop session
        self._supervisor = threading.Thread(
            target=self._supervise, daemon=True, name="studioforge-tray-supervisor"
        )
        self._supervisor.start()
        try:
            self.icon.run(setup=self._setup)
        finally:
            self._stop_event.set()
            self.stop_server()
            self._release_guard()
        return 0


def main(config: Config | None = None) -> int:
    """Tray entrypoint (``studioforge tray``)."""
    if config is None:
        from studioforge.config import load_config

        config = load_config(create=True)
    config.ensure_dirs()

    guard = acquire_single_instance()
    if guard is None:
        log.warning(
            "another tray instance holds the mutex port",
            host=SINGLE_INSTANCE_HOST,
            port=SINGLE_INSTANCE_PORT,
        )
        show_already_running()
        return 0

    return TrayApp(config, guard=guard).run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
