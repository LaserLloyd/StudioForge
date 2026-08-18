"""``studioforge`` entrypoint.

Runs the gateway and (optionally) the GUI as two uvicorn servers inside ONE
process on separate ports, plus the watchdog as a genuinely separate child
process.

The GUI shares the process because it reads the live registry/supervisor/planner
objects by reference -- an out-of-process GUI would need an IPC layer to show
state the gateway already holds. The watchdog is deliberately the opposite: its
entire job is to answer when this process cannot, so it must not share a fate
with it. See DECISIONS.md D1 and the watchdog module docstring.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from studioforge import __version__
from studioforge.config import Config, find_config_path, load_config
from studioforge.errors import ConfigError
from studioforge.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from studioforge.core.ports import PortConflict

log = get_logger(__name__)

app = typer.Typer(
    name="studioforge",
    help="GPU-only OpenAI-compatible LLM server built on llama.cpp llama-server.",
    no_args_is_help=True,
    add_completion=False,
)


def _load(config_path: Path | None) -> Config:
    """Load config for a CLI command, turning a bad file into one readable line.

    ``load_config`` already raises :class:`ConfigError` with the YAML error and
    its location, but nothing caught it here, so a stray tab in ``config.yaml``
    greeted the user with a forty-line traceback ending in ``ConfigError``
    (WP17 F8). Exit code 2 = "usage/config problem"; the traceback is still
    available with ``SF_DEBUG=1`` for the case where the message is not enough.
    """
    try:
        config = load_config(config_path, create=True)
    except ConfigError as exc:
        if os.environ.get("SF_DEBUG"):
            raise
        typer.echo(f"error: {exc.message}", err=True)
        typer.echo("  (set SF_DEBUG=1 for the full traceback)", err=True)
        raise typer.Exit(2) from exc
    configure_logging(
        config.logging.level, json_logs=config.logging.json_logs, log_dir=config.logs_dir
    )
    return config


@app.command()
def serve(
    config_path: Path | None = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    host: str | None = typer.Option(None, help="Override server.host"),
    port: int | None = typer.Option(None, help="Override server.port"),
    no_gui: bool = typer.Option(False, "--no-gui", help="Do not start the web GUI"),
    no_watchdog: bool = typer.Option(
        False, "--no-watchdog", help="Do not spawn the recovery watchdog sidecar"
    ),
    open_gui: bool = typer.Option(
        False, "--open", help="Open the control panel in your browser once it is up"
    ),
) -> None:
    """Run the gateway (and GUI + watchdog unless disabled)."""
    config = _load(config_path)
    if host:
        config.server.host = host
    if port:
        config.server.port = port
    if no_gui:
        config.gui.enabled = False
    if no_watchdog:
        config.watchdog.enabled = False

    _preflight_ports(config)
    asyncio.run(_serve(config, open_gui=open_gui))


def _console(message: str, *, err: bool = False) -> None:
    """Write to the console, if this process still has a usable one.

    A banner is a courtesy to a human at a terminal. When there is no terminal
    it must be a no-op, never a fatal -- and it *was* fatal, in a way that left
    no traceback anywhere: the last log line was "management MCP mounted" and the
    process was simply gone.

    Two different windowless cases, and only one of them is safe by default:

    * ``sys.stdout is None`` (``pythonw.exe``). Measured on CPython 3.12 with
      click 8.4: ``print`` and ``typer.echo`` both no-op here. Safe *today*, and
      relied upon by nothing -- click has changed this behaviour across major
      versions and the app must not be one dependency bump from dying at
      startup.
    * The stream exists but is dead -- detached console, closed handle, a
      redirect whose far end went away. ``print`` raises
      ``ValueError: I/O operation on closed file`` and takes the process with
      it. This is the one that actually bites, and nothing in the stdlib guards
      it for us.
    """
    stream = sys.stderr if err else sys.stdout
    if stream is None:
        return
    try:
        stream.write(message + "\n")
        stream.flush()
    except (OSError, ValueError, AttributeError):  # closed, detached or dead stream
        return


def _drop_adoptable_watchdog(config: Config, conflicts: list[PortConflict]) -> list[PortConflict]:
    """Remove the watchdog port from ``conflicts`` when we can adopt its owner.

    See DECISIONS.md D21: exactly one watchdog exists per config and it outlives
    main-process restarts, so the watchdog restarting us finds its own port
    "busy" by design. Treating that as a startup conflict is what made a
    watchdog-driven restart impossible.
    """
    from studioforge.core.ports import inspect_running_watchdog

    if not any(c.role == "watchdog" for c in conflicts):
        return conflicts
    presence = inspect_running_watchdog(config)
    if not presence.adoptable:
        log.info("the watchdog port is taken and not adoptable", reason=presence.reason)
        return conflicts
    log.info(
        "adopting the watchdog already running for this config",
        port=config.watchdog.port,
        pid=presence.pid,
        uptime_s=presence.uptime_s,
    )
    return [c for c in conflicts if c.role != "watchdog"]


def _preflight_ports(config: Config) -> None:
    """Refuse to start on a busy port, with a sentence instead of a traceback.

    Runs before uvicorn so the failure names the port, the process holding it
    (LM Studio and another StudioForge are called out by name) and the fix --
    rather than surfacing as a bare ``OSError: [WinError 10048]`` from inside
    the server. See :mod:`studioforge.core.ports`.

    Two conflicts are *expected* rather than fatal, and both were previously
    fatal (DECISIONS.md D21): the watchdog we are about to adopt, and the ports
    still held by the process we were spawned to replace, which is exiting right
    now. Everything else still fails, with the same message as before.
    """
    from studioforge.core.ports import (
        ENV_RESPAWN_PARENT_PID,
        check_startup_ports,
        describe_conflicts,
        respawn_parent_pid,
        respawn_wait_s,
        wait_for_ports,
    )

    conflicts = _drop_adoptable_watchdog(config, check_startup_ports(config))
    parent = respawn_parent_pid()
    # Consume it. The variable is inherited by everything we spawn from here --
    # the watchdog, and through it every future replacement -- and a stale one
    # naming a pid that died days ago would make a genuine port conflict sit in
    # a 45-second wait before reporting itself. It describes one startup only.
    os.environ.pop(ENV_RESPAWN_PARENT_PID, None)
    if conflicts and parent is not None:
        timeout_s = respawn_wait_s()
        log.info(
            "waiting for the process being replaced to release its ports",
            parent_pid=parent,
            ports=[c.port for c in conflicts],
            timeout_s=timeout_s,
        )
        conflicts = wait_for_ports(conflicts, timeout_s)
        conflicts = _drop_adoptable_watchdog(config, conflicts)
        if not conflicts:
            log.info("the process being replaced released its ports", parent_pid=parent)
        else:
            log.error(
                "the process being replaced still holds its ports",
                parent_pid=parent,
                ports=[c.port for c in conflicts],
                waited_s=timeout_s,
            )
    if not conflicts:
        return
    message = describe_conflicts(conflicts)
    log.error("startup port conflict", ports=[c.port for c in conflicts])
    _console(message, err=True)
    raise typer.Exit(3)


async def _serve(config: Config, *, open_gui: bool = False) -> None:
    import uvicorn

    from studioforge.api.app import create_app

    api = create_app(config)
    servers: list[uvicorn.Server] = []

    api_server = uvicorn.Server(
        uvicorn.Config(
            api,
            host=config.server.host,
            port=config.server.port,
            log_level=config.logging.level.lower(),
            access_log=False,
            timeout_graceful_shutdown=int(config.server.drain_timeout_s),
        )
    )
    servers.append(api_server)

    gui_server: uvicorn.Server | None = None
    if config.gui.enabled:
        try:
            from studioforge.gui.app import create_gui_app

            gui_app = create_gui_app(config, api_state=api.state)
            gui_server = uvicorn.Server(
                uvicorn.Config(
                    gui_app,
                    host=config.gui.host,
                    port=config.gui.port,
                    log_level="warning",
                    access_log=False,
                )
            )
            servers.append(gui_server)
        except Exception as exc:
            # The GUI is a convenience; the API is the product. A GUI import
            # failure must never stop the server from serving inference.
            log.error("gui failed to start; continuing without it", error=str(exc))

    watchdog_proc = _spawn_watchdog(config) if config.watchdog.enabled else None
    # The restart path needs to reach these: a process handing over to its own
    # replacement must leave the watchdog running (D21).
    api.state.watchdog_proc = watchdog_proc
    api.state.handing_over = False

    log.info(
        "studioforge starting",
        version=__version__,
        api=f"http://{config.server.host}:{config.server.port}",
        gui=(f"http://{config.gui.host}:{config.gui.port}" if gui_server else "disabled"),
        watchdog=(
            f"http://{config.watchdog.host}:{config.watchdog.port}" if watchdog_proc else "disabled"
        ),
        models_dir=str(config.models.dir),
        data_dir=str(config.data_dir),
    )

    _log_mcp_banner(config)

    if open_gui and gui_server is not None:
        asyncio.create_task(_open_gui_when_ready(config), name="studioforge-open-gui")
    elif open_gui:
        log.warning("--open was requested but the GUI is disabled")

    try:
        await asyncio.gather(*(server.serve() for server in servers))
    finally:
        if watchdog_proc is not None and watchdog_proc.poll() is None:
            if getattr(api.state, "handing_over", False):
                # We are exiting so a replacement can take our place. Killing the
                # watchdog on the way out would leave the new process with no
                # supervisor and a port that frees a moment too late; the
                # replacement adopts this one instead (D21).
                log.info(
                    "leaving the watchdog running for the replacement process",
                    pid=watchdog_proc.pid,
                )
            else:
                with contextlib.suppress(Exception):
                    watchdog_proc.terminate()
                    watchdog_proc.wait(timeout=10)


def _log_mcp_banner(config: Config) -> None:
    """Print where the MCP endpoint is and the PIN needed to pair with it.

    The PIN is printed **deliberately**, which is why it is not registered as a
    redacted secret: it is a LAN/tailnet pairing code whose entire purpose is to
    be read off this banner and typed into a client, in the way a Chromecast
    shows a code on the TV. ``server.api_key`` remains the strong credential and
    is never printed.

    Tailscale is listed first because it is the address that keeps working: a
    tailnet IP survives a network change, where a LAN address silently stops
    resolving.
    """
    mcp = getattr(config, "mcp", None)
    if mcp is None or not mcp.enabled or not mcp.advertise:
        return

    from studioforge.core.netinfo import reachable_urls

    endpoints = reachable_urls(config.server.port, mcp.path, host=config.server.host)
    if not endpoints:
        return

    log.info(
        "mcp endpoint",
        recommended=endpoints[0]["url"],
        pin=(mcp.pin if mcp.pin_required and mcp.pin else "not required"),
        alternatives=[e["url"] for e in endpoints[1:]],
    )
    lines = ["", "  MCP control plane", "  " + "-" * 44]
    for entry in endpoints:
        marker = "->" if entry is endpoints[0] else "  "
        lines.append(f"  {marker} {entry['url']}  ({entry['label']})")
    if mcp.pin_required and mcp.pin:
        lines.append(f"     PIN: {mcp.pin}    (header 'X-MCP-Pin', or ?pin=)")
    else:
        lines.append("     PIN: not required")
    lines.append("     ask the 'connection_info' tool for a direct LAN address")
    lines.append("")
    for line in lines:
        _console(line)


def _spawn_watchdog(config: Config) -> subprocess.Popen[bytes] | None:
    """Start the watchdog as a separate process, unless one is already running.

    Deliberately not a thread or task: its whole purpose is to stay responsive
    when this process is wedged, which an in-process server cannot do.

    Exactly one watchdog exists per config file, and it outlives us (D21). When
    the watchdog is the thing that restarted us it is still on its port, so
    spawning here would either fail to bind or -- worse -- succeed on a second
    port and give the deployment two supervisors with different ideas about
    which process is the server.
    """
    from studioforge.core.ports import inspect_running_watchdog

    presence = inspect_running_watchdog(config)
    if presence.adoptable:
        log.info(
            "adopted the running watchdog instead of starting a second one",
            port=config.watchdog.port,
            pid=presence.pid,
            uptime_s=presence.uptime_s,
        )
        return None
    try:
        creationflags = 0
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            kwargs["creationflags"] = creationflags
        else:
            kwargs["start_new_session"] = True
        return subprocess.Popen(
            [sys.executable, "-m", "studioforge.watchdog", "--config", str(config.config_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except Exception as exc:
        log.error("could not spawn watchdog sidecar", error=str(exc))
        return None


def gui_url(config: Config) -> str:
    """Browsable URL for the control panel.

    ``gui.host`` is a *bind* address and is usually ``0.0.0.0``, which no
    browser can open -- so a wildcard bind is reported as loopback.
    """
    host = config.gui.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    return f"http://{host}:{config.gui.port}/"


async def _gui_is_up(config: Config, *, timeout_s: float = 120.0) -> bool:
    """Poll the GUI until it answers, so we never open a dead tab."""
    import httpx

    url = gui_url(config)
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(url, follow_redirects=False)
            # A redirect to /login still means the GUI is serving.
            if response.status_code < 500:
                return True
        except Exception:
            pass
        await asyncio.sleep(0.5)
    return False


async def _open_gui_when_ready(config: Config) -> None:
    import webbrowser

    if not await _gui_is_up(config):
        log.warning("gui did not come up in time; not opening a browser", url=gui_url(config))
        return
    url = gui_url(config)
    log.info("opening control panel", url=url)
    try:
        webbrowser.open(url)
    except Exception as exc:  # pragma: no cover - platform dependent
        log.warning("could not open a browser", error=str(exc), url=url)


@app.command("gui")
def gui_cmd(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    print_only: bool = typer.Option(False, "--print", help="Print the URL, do not open"),
    wait: bool = typer.Option(False, "--wait", help="Wait for the GUI to come up first"),
) -> None:
    """Open the control panel of a running server in your browser."""
    import webbrowser

    config = _load(config_path)
    url = gui_url(config)
    if print_only:
        typer.echo(url)
        return
    if wait and not asyncio.run(_gui_is_up(config)):
        typer.echo(f"the GUI at {url} is not responding; is the server running?", err=True)
        raise typer.Exit(4)
    typer.echo(url)
    if not webbrowser.open(url):
        typer.echo("could not open a browser; paste the URL above", err=True)
        raise typer.Exit(1)


@app.command()
def scan(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    force: bool = typer.Option(False, "--force", help="Ignore the metadata cache"),
) -> None:
    """Scan the model directories and print what was found."""
    from studioforge.core.registry import Registry
    from studioforge.db import Database

    config = _load(config_path)
    config.ensure_dirs()
    db = Database(config.db_path)
    db.migrate()
    registry = Registry(config, db)
    result = registry.scan(force=force)
    typer.echo(
        f"scanned in {result.duration_s:.2f}s: "
        f"{len(result.added)} added, {len(result.removed)} removed, "
        f"{result.unchanged} unchanged, {len(result.errors)} errors"
    )
    for path, error in result.errors:
        typer.echo(f"  ERROR {path}: {error}")
    for record in registry.all():
        flags = "".join(
            [
                "V" if record.capabilities.vision else "-",
                "T" if record.capabilities.tools else "-",
                "M" if record.capabilities.multi_part else "-",
            ]
        )
        typer.echo(
            f"  {record.id:<70} {record.kind:<6} {record.quant:<9} "
            f"{record.size_bytes / 2**30:6.1f} GiB {flags}"
        )
    db.close()


@app.command("engine")
def engine_cmd(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    install: str | None = typer.Option(None, help="Install a specific llama.cpp tag"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Smoke-test the active engine"),
    check: bool = typer.Option(False, "--check", help="Report the newest release, install nothing"),
    update: bool = typer.Option(
        False, "--update", help="Install the newest llama.cpp release and make it the default"
    ),
    list_releases: bool = typer.Option(
        False, "--list", help="List the installable llama.cpp release tags and exit"
    ),
) -> None:
    """Inspect, install or update llama.cpp engine builds."""
    from studioforge.core.engine import EngineManager

    config = _load(config_path)
    config.ensure_dirs()
    manager = EngineManager(config)

    async def run() -> None:
        if list_releases:
            tags = await manager.list_releases(limit=20)
            typer.echo(" ".join(tags) or "no installable releases were returned")
            typer.echo(
                "(prereleases and non-bNNNN tags are hidden: they ship no llama-server build)"
            )
            return
        if check or update:
            status = await manager.check_update(limit=10)
            latest = status["latest"]
            # The ACTIVE tag, not the pin: active.json wins at load time, so
            # comparing against the pin can report an update for a build that is
            # not the one running.
            current = status["current"]
            variant = status.get("latest_variant")
            suffix = f" ({variant})" if variant else ""
            typer.echo(f"active: {current}    newest installable: {latest or 'none'}{suffix}")
            for entry in status.get("skipped") or []:
                typer.echo(f"  skipped {entry['tag']}: {entry['reason']}")
            if latest is None:
                typer.echo("no llama.cpp release offers a build this box can install", err=True)
                raise typer.Exit(1)
            if not status["update_available"]:
                typer.echo("already on the newest release")
                if not smoke_test:
                    return
            elif check:
                typer.echo(f"run 'studioforge engine --update' to move to {latest}")
                return
            else:
                # Install and smoke-test BEFORE repinning: an engine that fails
                # its micro-load must not become the default, or every
                # subsequent load breaks. Verified engines only.
                typer.echo(f"installing {latest} ...")
                info = await manager.install(latest)
                ok, detail = await manager.smoke_test(info.tag)
                if not ok:
                    typer.echo(f"smoke test FAILED for {latest}; keeping {current}\n{detail}")
                    raise typer.Exit(1)
                config.engine.pinned_tag = latest
                config.save()
                typer.echo(f"engine updated to {latest} and pinned in {config.config_path}")
                typer.echo(
                    "running instances keep their current engine; reload a model to pick "
                    "up the new one"
                )
        if install:
            info = await manager.install(install)
            typer.echo(f"installed {info.tag} ({info.variant}) at {info.path}")
        else:
            info = await manager.ensure_engine()
            typer.echo(f"active engine: {info.tag} ({info.variant})")
        for entry in manager.installed():
            marker = "*" if entry.active else " "
            typer.echo(
                f" {marker} {entry.tag:<12} {entry.variant:<16} smoke_tested={entry.smoke_tested}"
            )
        if smoke_test:
            ok, detail = await manager.smoke_test(info.tag)
            typer.echo(f"smoke test: {'PASS' if ok else 'FAIL'}\n{detail}")

    asyncio.run(run())


@app.command("config")
def config_cmd(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    show_path: bool = typer.Option(False, "--path", help="Print the config file path only"),
) -> None:
    """Print the effective configuration (secrets redacted)."""
    import yaml

    from studioforge.api.auth import redact_config_dict

    if show_path:
        typer.echo(str(find_config_path(config_path)))
        return
    config = _load(config_path)
    data = redact_config_dict(config.to_yaml_dict())
    typer.echo(f"# {config.config_path}")
    typer.echo(yaml.safe_dump(data, sort_keys=False))


@app.command("open")
def open_link(
    url: str = typer.Argument(..., help="A studioforge:// or lmstudio:// deep link"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    print_only: bool = typer.Option(False, "--print", help="Print the URL, do not open"),
) -> None:
    """Handle a deep link from HuggingFace's "Use this model" button.

    HuggingFace emits ``lmstudio://open_from_hf?model=<owner>/<repo>``; this
    resolves it to the StudioForge GUI's Download tab with that repo's quant
    picker open.
    """
    from studioforge.core.protocol import gui_url_for, open_in_browser, parse_deep_link

    config = _load(config_path)
    link = parse_deep_link(url)
    target = gui_url_for(link, config)
    typer.echo(target)
    if not print_only and not open_in_browser(target):
        typer.echo("could not open a browser; paste the URL above", err=True)
        raise typer.Exit(1)


@app.command("protocol")
def protocol_cmd(
    action: str = typer.Argument("status", help="status | register | unregister"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    takeover_lmstudio: bool = typer.Option(
        False,
        "--takeover-lmstudio",
        help=(
            "Also claim lmstudio:// so HuggingFace's LM Studio button opens "
            "StudioForge. Reversible: the previous handler is backed up and "
            "restored by 'protocol unregister'."
        ),
    ),
) -> None:
    """Register StudioForge as a URL-scheme handler for model deep links."""
    import json as _json

    from studioforge.core import protocol

    config = _load(config_path)
    config.ensure_dirs()
    if action == "status":
        typer.echo(_json.dumps(protocol.status(config), indent=2))
        return
    if action == "register":
        result = protocol.register(config, takeover_lmstudio=takeover_lmstudio)
        typer.echo(_json.dumps(result, indent=2))
        if takeover_lmstudio:
            typer.echo(
                "\nlmstudio:// now opens StudioForge. Run "
                "'studioforge protocol unregister' to give it back to LM Studio."
            )
        return
    if action == "unregister":
        typer.echo(_json.dumps(protocol.unregister(config), indent=2))
        return
    typer.echo(f"unknown action {action!r}; expected status|register|unregister", err=True)
    raise typer.Exit(2)


@app.command("autostart")
def autostart_cmd(
    action: str = typer.Argument("status", help="status | enable | disable"),
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    open_gui: bool = typer.Option(
        False, "--open", help="Also open the control panel when it starts at login"
    ),
    tray: bool = typer.Option(
        False,
        "--tray",
        help="Start the system tray at login, which brings the server up with it (Windows)",
    ),
) -> None:
    """Start StudioForge automatically when you log in."""
    from studioforge.core import autostart

    config = _load(config_path)
    config.ensure_dirs()
    if action == "status":
        typer.echo(autostart.status(config).describe())
        return
    if action == "enable":
        result = autostart.enable(config, open_gui=open_gui, tray=tray)
        typer.echo(result.describe())
        typer.echo("run 'studioforge autostart disable' to undo")
        return
    if action == "disable":
        typer.echo(autostart.disable(config).describe())
        return
    typer.echo(f"unknown action {action!r}; expected status|enable|disable", err=True)
    raise typer.Exit(2)


@app.command("tray")
def tray_cmd(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
) -> None:
    """Run the system-tray app, which supervises the server for you."""
    if os.name != "nt":
        # The tray is a Windows feature: pystray's Linux backends need a
        # display and a GTK/X stack that a server box does not have, and the
        # import itself crashed on a headless Linux host (WP17 F7). The Linux
        # mechanism is the systemd user unit; say so instead of tracebacking.
        typer.echo(
            "error: the system tray is Windows-only. On Linux use the systemd units in "
            "deploy/ (see deploy/README.md) or run `studioforge serve` directly.",
            err=True,
        )
        raise typer.Exit(2)
    config = _load(config_path)
    config.ensure_dirs()
    try:
        from studioforge.tray.tray_app import main as tray_main
    except ImportError as exc:
        from studioforge.tray import MISSING_PYSTRAY_HINT

        typer.echo(f"{MISSING_PYSTRAY_HINT}\n  (import failed: {exc})", err=True)
        raise typer.Exit(5) from exc
    raise typer.Exit(tray_main(config))


@app.command("capabilities")
def capabilities_cmd(
    config_path: Path | None = typer.Option(None, "--config", "-c"),
    check_update: bool = typer.Option(
        False, "--check-update", help="Ask GitHub for a newer engine"
    ),
    show_architectures: bool = typer.Option(
        False, "--architectures", help="List every supported architecture"
    ),
) -> None:
    """Show the backend, what it can run, and what your hardware allows."""
    from studioforge.core.capabilities import build_report, format_bytes
    from studioforge.core.engine import EngineManager
    from studioforge.core.gpu import get_probe
    from studioforge.core.registry import Registry
    from studioforge.db import Database

    config = _load(config_path)
    config.ensure_dirs()
    db = Database(config.db_path)
    db.migrate()
    registry = Registry(config, db)
    registry.scan()
    probe = get_probe()
    report = build_report(
        config,
        gpus=probe.list_gpus(),
        records=registry.all(),
        engine_manager=EngineManager(config, probe=probe),
        probe=probe,
    )
    engine, hardware, library = report.engine, report.hardware, report.library

    typer.echo("")
    typer.echo("BACKEND")
    typer.echo(f"  engine          {engine.tag}  ({engine.variant})")
    if engine.version_string:
        typer.echo(f"  build           {engine.version_string}")
    typer.echo(f"  smoke tested    {'yes' if engine.smoke_tested else 'NO'}")
    typer.echo(
        f"  supports        {len(engine.architectures)} architectures, "
        f"{len(engine.quant_types)} quantizations"
    )
    typer.echo(f"  list source     {engine.source} ({engine.source_detail})")

    typer.echo("")
    typer.echo("HARDWARE")
    for gpu in hardware.gpus:
        typer.echo(
            f"  CUDA{gpu['index']}  {gpu['name']:<26} "
            f"{gpu['total_bytes'] / 2**30:5.0f} GiB  cc {gpu['compute_capability']}"
        )
    typer.echo(f"  driver          {hardware.driver_version} (CUDA {hardware.cuda_driver_version})")
    typer.echo(
        f"  usable now      {format_bytes(hardware.usable_largest_bytes)} on the largest GPU, "
        f"{format_bytes(hardware.usable_total_bytes)} across all"
    )

    typer.echo("")
    typer.echo("WHAT YOU CAN RUN")
    sizing = library["sizing"]
    typer.echo(f"  {sizing['fits_one_gpu']} of your {library['model_count']} models fit on one GPU")
    typer.echo(f"  {sizing['needs_multiple_gpus']} need a multi-GPU split")
    typer.echo(f"  {sizing['too_big']} are too big for this box")
    typer.echo(f"  ({sizing['note']})")
    caps = library["capabilities"]
    typer.echo(
        f"  features in your library: vision {caps['vision']}, tools {caps['tools']}, "
        f"thinking {caps['thinking']}, embedding {caps['embedding']}, "
        f"multi-part {caps['multi_part']}"
    )
    unsupported = library["unsupported_by_engine"]
    if unsupported:
        typer.echo(
            f"  !! {len(unsupported)} model(s) use an architecture this engine does not know:"
        )
        for entry in unsupported[:5]:
            typer.echo(f"       {entry['model_id']}  ({entry['architecture']})")
    else:
        typer.echo("  every model in your library uses an architecture this engine supports")

    typer.echo("")
    typer.echo("SUPPORTED FEATURES")
    for name, note in report.features.items():
        typer.echo(f"  {name:<14} {note}")

    if show_architectures:
        typer.echo("")
        typer.echo("ARCHITECTURES")
        line = "  "
        for arch in engine.architectures:
            if len(line) + len(arch) > 96:
                typer.echo(line)
                line = "  "
            line += arch + " "
        if line.strip():
            typer.echo(line)

    if check_update:
        from studioforge.core.engine import EngineManager as _EM

        async def _check() -> None:
            manager = _EM(config, probe=probe)
            status = await manager.check_update(limit=5)
            latest = status["latest"]
            variant = status.get("latest_variant")
            typer.echo("")
            typer.echo("UPDATES")
            typer.echo(f"  active          {status['current']}")
            typer.echo(f"  pinned          {config.engine.pinned_tag}")
            typer.echo(f"  newest          {latest or 'none installable'}")
            if variant:
                typer.echo(f"  variant         {variant}")
            for entry in status.get("skipped") or []:
                typer.echo(f"  skipped         {entry['tag']}: {entry['reason']}")
            if status["update_available"]:
                typer.echo("  run 'studioforge engine --update' to move to it")
            else:
                typer.echo("  you are on the newest release")

        asyncio.run(_check())
    typer.echo("")
    db.close()


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


def main() -> None:
    # Windows lacks SIGTERM semantics for console apps; uvicorn handles SIGINT.
    if os.name != "nt":
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGTERM, signal.default_int_handler)
    app()


if __name__ == "__main__":
    main()
