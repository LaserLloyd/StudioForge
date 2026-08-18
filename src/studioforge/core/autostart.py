"""Start StudioForge automatically when the user logs in.

Two platform mechanisms, both **user-level** so neither needs administrator
rights:

* **Windows** -- a ``.vbs`` shim in the per-user Startup folder. A ``.lnk``
  shortcut would be the conventional choice but creating one needs COM
  (``pywin32``), which is not a dependency; and a plain ``.bat`` would flash a
  console window at every login. The VBScript one-liner launches the same
  launcher with the window hidden and no extra packages.
* **Linux** -- a systemd *user* unit, enabled with ``systemctl --user``. The
  packaged units in ``deploy/`` are the system-wide equivalent for a server
  install; this is the desktop case.

Everything here is reversible by :func:`disable`, and :func:`status` reports
what is actually on disk rather than what we believe we wrote.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from studioforge.config import Config
from studioforge.errors import StudioForgeError
from studioforge.logging import get_logger

log = get_logger(__name__)

ENTRY_NAME = "StudioForge"
WINDOWS_SHIM = f"{ENTRY_NAME}.vbs"
LINUX_UNIT = "studioforge.service"


class AutostartError(StudioForgeError):
    status_code = 500
    code = "autostart_failed"


@dataclass(frozen=True)
class AutostartStatus:
    enabled: bool
    mechanism: str
    path: Path | None
    detail: str = ""

    def describe(self) -> str:
        state = "enabled" if self.enabled else "not enabled"
        where = f" ({self.path})" if self.path else ""
        suffix = f" - {self.detail}" if self.detail else ""
        return f"autostart {state} via {self.mechanism}{where}{suffix}"


def startup_dir() -> Path:
    """Per-user Startup folder (Windows only)."""
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _user_unit_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "systemd" / "user"


def _tray_interpreter() -> str:
    """``pythonw.exe`` where it exists, so the tray owns no console window.

    The VBS shim already launches hidden, so this is belt-and-braces on
    Windows -- but a console-less interpreter also means nothing flashes if the
    shim is ever run directly, and it matches what ``StudioForge Tray.bat``
    does.
    """
    if os.name != "nt":
        return sys.executable
    candidate = Path(sys.executable).with_name("pythonw.exe")
    return str(candidate) if candidate.is_file() else sys.executable


def launch_command(
    config: Config, *, open_gui: bool = False, tray: bool = False
) -> list[str]:
    """Argv that starts StudioForge the same way a manual launch would.

    Prefers the installed console script; falls back to ``<python> -m
    studioforge`` so a source checkout autostarts too. The config path is passed
    explicitly because a login shell may not carry ``SF_DATA_DIR``.

    With ``tray=True`` the entry point is the system-tray app instead of the
    bare server. The tray is a superset: it adopts a server that is already
    running and starts one otherwise, so this yields the icon *and* the server
    from a single login entry. ``open_gui`` does not apply -- the tray's own
    menu is how you reach the control panel.
    """
    if tray:
        script = shutil.which("studioforge")
        base = [script] if script else [_tray_interpreter(), "-m", "studioforge"]
        return [*base, "tray", "--config", str(config.config_path)]
    script = shutil.which("studioforge")
    base = [script] if script else [sys.executable, "-m", "studioforge"]
    argv = [*base, "serve", "--config", str(config.config_path)]
    if open_gui:
        argv.append("--open")
    return argv


def _quote_for_vbs(argv: list[str]) -> str:
    parts = [f'""{part}""' if " " in part else part for part in argv]
    return " ".join(parts)


def enable(
    config: Config, *, open_gui: bool = False, tray: bool = False
) -> AutostartStatus:
    if os.name == "nt":
        return _enable_windows(config, open_gui=open_gui, tray=tray)
    if tray:
        # The Linux path is a systemd *user* unit intended for a headless
        # server. Starting a tray from it would fail at login with no display,
        # and failing loudly here is kinder than a unit that flaps.
        raise AutostartError(
            "tray autostart is Windows-only; the Linux mechanism is a systemd "
            "user unit that runs the server headless. Enable it without --tray."
        )
    return _enable_linux(config, open_gui=open_gui)


def disable(config: Config) -> AutostartStatus:
    if os.name == "nt":
        return _disable_windows()
    return _disable_linux()


def status(config: Config) -> AutostartStatus:
    if os.name == "nt":
        path = startup_dir() / WINDOWS_SHIM
        if not path.is_file():
            return AutostartStatus(
                enabled=False, mechanism="Windows Startup folder", path=None
            )
        # Report which entry point the shim actually launches. "Enabled" alone
        # is not enough to answer "why is there no tray icon?", which is the
        # question this status is usually being asked to settle.
        try:
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            body = ""
        launches_tray = " tray " in f" {body} " or '"tray"' in body
        mode = (
            "tray (which starts the server too)"
            if launches_tray
            else "server only - no tray icon at login"
        )
        return AutostartStatus(
            enabled=True,
            mechanism="Windows Startup folder",
            path=path,
            detail=mode,
        )
    unit = _user_unit_dir() / LINUX_UNIT
    enabled = False
    detail = ""
    if shutil.which("systemctl"):
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-enabled", LINUX_UNIT],
                capture_output=True,
                text=True,
                timeout=20,
            )
            enabled = result.stdout.strip() == "enabled"
            detail = result.stdout.strip() or result.stderr.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            detail = f"systemctl unavailable: {exc}"
    return AutostartStatus(
        enabled=enabled,
        mechanism="systemd --user",
        path=unit if unit.is_file() else None,
        detail=detail,
    )


# --- Windows ---------------------------------------------------------------


def _enable_windows(config: Config, *, open_gui: bool, tray: bool = False) -> AutostartStatus:
    target = startup_dir()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AutostartError(f"could not create the Startup folder {target}: {exc}") from exc

    argv = launch_command(config, open_gui=open_gui, tray=tray)
    # WScript.Shell Run with intWindowStyle 0 = hidden, bWaitOnReturn False.
    # Quoting is doubled because the whole command is a VBScript string literal.
    script = (
        "' Created by 'studioforge autostart enable'. Delete this file (or run\n"
        "' 'studioforge autostart disable') to stop StudioForge starting at login.\n"
        'Set shell = CreateObject("WScript.Shell")\n'
        f'shell.CurrentDirectory = "{config.data_dir}"\n'
        f'shell.Run "{_quote_for_vbs(argv)}", 0, False\n'
    )
    path = target / WINDOWS_SHIM
    try:
        path.write_text(script, encoding="utf-8")
    except OSError as exc:
        raise AutostartError(f"could not write {path}: {exc}") from exc
    log.info("autostart enabled", path=str(path), tray=tray)
    return AutostartStatus(
        enabled=True,
        mechanism="Windows Startup folder",
        path=path,
        detail="starts the tray, which brings up the server" if tray else "",
    )


def _disable_windows() -> AutostartStatus:
    path = startup_dir() / WINDOWS_SHIM
    existed = path.is_file()
    if existed:
        try:
            path.unlink()
        except OSError as exc:
            raise AutostartError(f"could not remove {path}: {exc}") from exc
    log.info("autostart disabled", removed=existed)
    return AutostartStatus(
        enabled=False,
        mechanism="Windows Startup folder",
        path=None,
        detail="removed" if existed else "was not enabled",
    )


# --- Linux -----------------------------------------------------------------


def _enable_linux(config: Config, *, open_gui: bool) -> AutostartStatus:
    unit_dir = _user_unit_dir()
    unit_dir.mkdir(parents=True, exist_ok=True)
    argv = launch_command(config, open_gui=open_gui)
    exec_start = " ".join(argv)
    unit = unit_dir / LINUX_UNIT
    unit.write_text(
        "[Unit]\n"
        "Description=StudioForge - GPU-only OpenAI-compatible LLM gateway\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"Environment=SF_DATA_DIR={config.data_dir}\n"
        f"ExecStart={exec_start}\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        # Long enough to drain and stop every llama-server child.
        "TimeoutStopSec=120\n"
        "KillMode=mixed\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n",
        encoding="utf-8",
    )

    systemctl = shutil.which("systemctl")
    if systemctl is None:
        return AutostartStatus(
            enabled=False,
            mechanism="systemd --user",
            path=unit,
            detail="unit written, but systemctl was not found to enable it",
        )
    for args in (["daemon-reload"], ["enable", LINUX_UNIT]):
        result = subprocess.run(
            [systemctl, "--user", *args], capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            raise AutostartError(
                f"systemctl --user {' '.join(args)} failed: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
    log.info("autostart enabled", unit=str(unit))
    return AutostartStatus(enabled=True, mechanism="systemd --user", path=unit)


def _disable_linux() -> AutostartStatus:
    unit = _user_unit_dir() / LINUX_UNIT
    systemctl = shutil.which("systemctl")
    detail = ""
    if systemctl is not None:
        result = subprocess.run(
            [systemctl, "--user", "disable", LINUX_UNIT],
            capture_output=True,
            text=True,
            timeout=60,
        )
        detail = result.stderr.strip() or result.stdout.strip()
    if unit.is_file():
        unit.unlink()
    return AutostartStatus(
        enabled=False, mechanism="systemd --user", path=None, detail=detail or "removed"
    )
