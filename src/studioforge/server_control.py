"""Start, stop and ask about the StudioForge server running on this machine.

Backs ``studioforge server start|stop|status`` and the first-run wizard (which
starts the server when it needs one for the model-loading steps). Deliberately
small and side-effect-honest:

* ``status``  -- never changes anything: asks ``/health`` and reads the data
  directory's instance lock for the pid.
* ``start``   -- spawns ``python -m studioforge serve --config <file>`` detached
  (no console window, survives this CLI exiting) and waits for ``/health``. It
  refuses when something already answers, and it does **not** supervise: the
  tray or a service unit is the supervisor; this is the "just run it" path.
* ``stop``    -- asks the running server to restart is NOT what this does; it
  ends the process (graceful: SIGTERM / CTRL_BREAK first, kill after a wait)
  found through the instance lock. A server run by the tray is respawned by
  the tray, as it always was: stop the tray to stop it for good.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from studioforge.cli_client import CliError, base_url_for
from studioforge.config import Config
from studioforge.tlsutil import scheme_for

__all__ = ["ServerStatus", "server_status", "start_server", "stop_server"]


@dataclass(frozen=True)
class ServerStatus:
    running: bool
    url: str
    pid: int | None = None
    version: str | None = None
    uptime_s: float | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "url": self.url,
            "pid": self.pid,
            "version": self.version,
            "uptime_s": self.uptime_s,
            "detail": self.detail,
        }


def _holder_pid(config: Config) -> int | None:
    from studioforge.core.instance_lock import InstanceLock

    try:
        holder = InstanceLock(config.data_dir).holder() or {}
    except OSError:
        return None
    pid = holder.get("pid")
    return int(pid) if isinstance(pid, int) else None


def server_status(config: Config, *, timeout_s: float = 3.0) -> ServerStatus:
    url = base_url_for(config)
    try:
        response = httpx.get(
            f"{url}/health", timeout=timeout_s, verify=scheme_for(config) == "http"
        )
    except httpx.HTTPError as exc:
        return ServerStatus(False, url, _holder_pid(config), detail=type(exc).__name__)
    body: dict[str, Any] = {}
    with contextlib.suppress(ValueError):
        body = response.json()
    return ServerStatus(
        response.status_code < 500,
        url,
        _holder_pid(config),
        version=body.get("version"),
        uptime_s=body.get("uptime_s"),
        detail=str(body.get("status") or response.status_code),
    )


def start_server(config: Config, *, wait_s: float = 90.0) -> ServerStatus:
    """Spawn the server detached and wait for it to answer. Raises :class:`CliError`."""
    current = server_status(config)
    if current.running:
        raise CliError(
            f"StudioForge is already running at {current.url}.",
            "Use `studioforge server status`, or stop it first with `studioforge server stop`.",
        )
    argv = [sys.executable, "-m", "studioforge", "serve", "--config", str(config.config_path)]
    kwargs: dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        kwargs["start_new_session"] = True
    log_path = config.logs_dir / "server-start.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as sink:
        try:
            process = subprocess.Popen(  # noqa: S603 - argv is ours, no shell
                argv, stdout=sink, stderr=sink, stdin=subprocess.DEVNULL, **kwargs
            )
        except OSError as exc:
            raise CliError(
                f"Could not start the server: {exc}", "Run `studioforge serve`."
            ) from exc
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(0.5)
        if process.poll() is not None:
            tail = ""
            with contextlib.suppress(OSError):
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-600:]
            raise CliError(
                f"The server exited right away (code {process.returncode}).\n{tail}".rstrip(),
                "Run `studioforge doctor`; the full output is in " + str(log_path),
            )
        status = server_status(config, timeout_s=1.5)
        if status.running:
            return status
    raise CliError(
        f"The server did not answer within {wait_s:.0f}s.",
        f"It may still be starting; check `studioforge server status` and {log_path}.",
    )


def stop_server(config: Config, *, wait_s: float = 30.0, client: Any = None) -> bool:
    """End the server. ``True`` if one was running and is now gone.

    Asks politely first (``POST /api/server/stop``: the server drains and shuts
    both listeners down), then ends the process found through the instance lock
    if it is still there after ``wait_s``.
    """
    import psutil

    current = server_status(config)
    if not current.running and current.pid is None:
        return False
    if current.running and client is not None:
        with contextlib.suppress(CliError):
            client.post("/api/server/stop", {"confirm": True})
    pid = current.pid
    if pid is not None and psutil.pid_exists(pid):
        process = psutil.Process(pid)
        try:
            process.wait(timeout=wait_s)
        except psutil.TimeoutExpired:
            with contextlib.suppress(psutil.Error):
                process.terminate()
                process.wait(timeout=10)
        except psutil.NoSuchProcess:
            pass
        return True
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if not server_status(config, timeout_s=1.0).running:
            return True
        time.sleep(0.5)
    raise CliError(
        "The server was asked to stop but still answers, and its process could not be found "
        "(it runs in a container or another account).",
        "Stop it where it runs.",
    )


def pidfile_hint(config: Config) -> Path:
    return config.data_dir / ".instance.lock"
