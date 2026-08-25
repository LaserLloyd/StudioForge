"""Startup port preflight: turn a bind clash into a sentence.

The gateway defaults to port 1234 because that is LM Studio's default and the
whole migration story is "change the host, not the port" (DECISIONS.md D8).
The direct consequence is that the *most likely* startup failure on this box is
LM Studio -- or a StudioForge that is already running -- still holding 1234.

Left alone, that surfaces as ``OSError: [WinError 10048] Only one usage of each
socket address ... is normally permitted``, from inside uvicorn, after the app
has already been built. It names no port, no process and no fix. The failure
that motivated this was worse still: two LM Studio instances, one stale, where
the "reset" launched a second server rather than killing the first, so the
symptom was not a crash but a machine serving from the wrong process.

So: check every port we are about to bind *before* uvicorn starts, and if one
is taken, say which port, what is probably holding it, and what to do.
Identifying the holder needs ``psutil.net_connections``, which on Windows and
macOS may refuse without elevation -- that degrades to "could not identify the
process", never to a traceback.
"""

from __future__ import annotations

import contextlib
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from studioforge.config import Config
from studioforge.logging import get_logger

log = get_logger(__name__)

#: Executable names that mean "LM Studio is holding this port".
_LMSTUDIO_HINTS = ("lm studio", "lmstudio", "lms.exe", "llama-server")

#: Marker for another copy of *this* program, found in the holder's cmdline.
_SELF_MARKER = "studioforge"

#: Set by a process that is respawning itself, naming the pid that is about to
#: exit. Its ports are *expected* to be busy for a moment; see
#: :func:`wait_for_ports` and DECISIONS.md D21.
ENV_RESPAWN_PARENT_PID = "SF_RESPAWN_PARENT_PID"
#: How long the replacement waits for those ports. Generous: the parent has to
#: finish draining in-flight requests first.
ENV_RESPAWN_WAIT_S = "SF_RESPAWN_WAIT_S"
DEFAULT_RESPAWN_WAIT_S = 45.0

#: Set to 0/no/false to refuse to adopt a running watchdog and demand a free
#: port instead -- the escape hatch for "the watchdog's own code changed".
ENV_ADOPT_WATCHDOG = "SF_ADOPT_WATCHDOG"

#: Exit code ``studioforge serve`` uses when a port it needs is held by someone
#: else (:func:`studioforge.__main__._preflight_ports`). One name for it,
#: because the tray reads it back: a child that exits with this code did not
#: crash, and respawning it cannot help -- the port is still taken (D28).
EXIT_PORT_CONFLICT = 3

#: Exit code ``studioforge serve`` uses when it was asked to restart and is
#: leaving the respawn to the process that launched it (the tray). EX_TEMPFAIL
#: in sysexits terms: "try again". The tray respawns immediately and does not
#: count it as a crash (D28).
EXIT_RESTART_REQUESTED = 75

#: Set by a supervisor on the ``serve`` child it launches, naming itself
#: (``"tray"``). A server that sees it restarts by *exiting* with
#: :data:`EXIT_RESTART_REQUESTED` rather than by respawning itself or asking
#: the watchdog to, so exactly one process -- the one that owns it -- brings
#: it back.
ENV_SUPERVISOR = "SF_SUPERVISOR"

_FALSEY = {"0", "false", "no", "off"}


def env_without_supervisor(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment with :data:`ENV_SUPERVISOR` removed.

    The tray sets it on the child it spawns, and it means "the process that
    launched you respawns you" (D28). It is true of that child only -- not of
    the watchdog the child spawns, nor of a replacement the watchdog spawns.
    Both inherited it unchanged, so a watchdog-launched server believed a tray
    would bring it back, exited 75 on its next restart, and nobody did.
    """
    source = dict(os.environ if base is None else base)
    source.pop(ENV_SUPERVISOR, None)
    return source


def supervising_tray_is_alive() -> bool:
    """Whether this process's *direct parent* is a live StudioForge tray.

    :func:`supervised_by` reports the environment variable; this checks the
    fact behind it. A tray that was quit leaves the variable set on the server
    it launched, and an exit-75 restart then goes nowhere -- the D28 rule is
    "the process that launched you respawns you", which needs that process to
    exist.
    """
    try:
        me = psutil.Process(os.getpid())
        argv = list(me.cmdline())[1:]
        created = float(me.create_time())
        current = me
        parent: psutil.Process | None = None
        # Through venv launcher stubs: on Windows `.venv/Scripts/python.exe` is
        # a redirector that runs the real interpreter as its child with the
        # same argv, so our direct parent is usually that stub and the tray is
        # one hop above it. Same rule as the watchdog's launch_parent().
        for _ in range(8):
            candidate = current.parent()
            if candidate is None:
                return False
            with candidate.oneshot():
                if candidate.status() == psutil.STATUS_ZOMBIE:
                    return False
                if float(candidate.create_time()) > created + 1.0:
                    return False  # pid reused: this is not the process that launched us
                candidate_argv = list(candidate.cmdline())
            if candidate_argv[1:] != argv or not argv:
                parent = candidate
                break
            current = candidate
        if parent is None:
            return False
        cmdline = [str(part).lower() for part in parent.cmdline()]
    except Exception:  # noqa: BLE001 - psutil raises a family; "unknown" is "no"
        return False
    return "studioforge" in " ".join(cmdline) and any(part == "tray" for part in cmdline)


def supervised_by() -> str | None:
    """Who launched this process as a supervised child, per :data:`ENV_SUPERVISOR`."""
    value = os.environ.get(ENV_SUPERVISOR, "").strip().lower()
    return value or None


@dataclass(slots=True)
class PortHolder:
    """The process listening on a port, as far as we can tell."""

    pid: int | None = None
    name: str | None = None
    cmdline: str = ""
    #: True when the holder is (another) StudioForge process.
    is_studioforge: bool = False
    #: True when the holder looks like LM Studio or a stray llama-server.
    is_lmstudio: bool = False

    def describe(self) -> str:
        if self.pid is None:
            return "could not identify the process holding it (try running as administrator)"
        who = self.name or "an unknown process"
        return f"held by {who} (pid {self.pid})"


@dataclass(slots=True)
class PortConflict:
    """One configured port that cannot be bound, and why."""

    role: str  # "server" | "gui" | "watchdog"
    port: int
    host: str
    holder: PortHolder = field(default_factory=PortHolder)
    setting: str = ""

    def message(self) -> str:
        lines = [f"Port {self.port} ({self.role}) is already in use: {self.holder.describe()}."]
        if self.holder.is_studioforge:
            lines.append(
                "  That looks like another StudioForge instance. Stop it first "
                "(only one can own this port), or change the port below."
            )
        elif self.holder.is_lmstudio:
            lines.append(
                "  That looks like LM Studio (or a llama-server it left running). "
                "Quit LM Studio -- including its tray icon and any 'Local Server' "
                "tab -- or change the port below."
            )
        lines.append(f"  Fix: quit whatever owns the port, or set {self.setting} in config.yaml.")
        return "\n".join(lines)


def port_has_listener(port: int, host: str = "127.0.0.1", timeout_s: float = 1.0) -> bool:
    """Whether something ACCEPTS connections on ``host:port`` right now.

    The complement of :func:`port_is_bindable`, and not the same question. On
    Windows a bind to a *specific* address (``127.0.0.1:1235``) succeeds even
    while another socket holds the *wildcard* (``0.0.0.0:1235``) unless that
    socket set ``SO_EXCLUSIVEADDRUSE`` -- so "can I bind 127.0.0.1:port" says
    **free** for a port our own watchdog is serving on ``0.0.0.0``. That is
    exactly how a restarted server concluded "nothing is listening on port
    1235", declined to adopt the watchdog it had just been told to keep, and
    then died on the port conflict the wildcard bind correctly reported
    (2026-08-19, first restart after the V2 switch). To learn whether a
    *listener* exists, connect to it.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def port_is_bindable(port: int, host: str = "0.0.0.0") -> bool:
    """Whether ``host:port`` can be bound right now.

    This is the startup preflight (see :func:`check_startup_ports`), and what
    it must answer is not "can a bare socket bind here" but "can the server we
    are about to start bind here". Those differ, in opposite directions on the
    two platforms, so the probe models the real thing on each:

    * **Windows** -- set ``SO_EXCLUSIVEADDRUSE``. Without it a second bind
      against a listener that set ``SO_REUSEADDR`` (most servers do) succeeds,
      so the probe calls a busy port free: the exact way two servers end up
      sharing a port and one of them silently never gets a request. The danger
      here is a false *free*.
    * **POSIX** -- set ``SO_REUSEADDR``, because uvicorn does. asyncio's
      ``create_server`` sets it by default on POSIX, so a port whose previous
      listener left connections in ``TIME_WAIT`` is one uvicorn will take
      happily -- while a *plain* bind fails for as long as those last, up to
      about a minute. Without this the preflight refused to start over a port
      the server would have bound, and exited ``EXIT_PORT_CONFLICT`` blaming a
      conflict that did not exist. The danger here is a false *busy*.

    ``supervisor._port_is_bindable`` makes the same split for llama-server
    children; the two are deliberately identical in behaviour and differ only
    in which process they are predicting.
    """
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            with contextlib.suppress(OSError, AttributeError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            with contextlib.suppress(OSError, AttributeError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


def find_port_holder(port: int) -> PortHolder:
    """Identify the process listening on ``port``. Never raises."""
    holder = PortHolder()
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.Error, OSError, RuntimeError) as exc:
        log.debug("port_holder_lookup_denied", port=port, error=str(exc))
        return holder
    for conn in connections:
        laddr = conn.laddr
        if not laddr or getattr(laddr, "port", None) != port:
            continue
        if conn.status not in (psutil.CONN_LISTEN, psutil.CONN_NONE):
            continue
        holder.pid = conn.pid
        break
    if holder.pid is None:
        return holder
    try:
        proc = psutil.Process(holder.pid)
        holder.name = proc.name()
        with contextlib.suppress(psutil.Error):
            holder.cmdline = " ".join(proc.cmdline())
    except (psutil.Error, ValueError):
        return holder

    haystack = f"{holder.name or ''} {holder.cmdline}".lower()
    # Our own process tree is identified from the command line rather than the
    # executable name, because on Windows it is "python.exe" like everything
    # else -- the module path is what makes it recognisable.
    holder.is_studioforge = _SELF_MARKER in haystack and holder.pid != os.getpid()
    holder.is_lmstudio = not holder.is_studioforge and any(
        hint in haystack for hint in _LMSTUDIO_HINTS
    )
    return holder


def check_startup_ports(config: Config) -> list[PortConflict]:
    """Every configured port we are about to bind that is already taken.

    Only ports this run will actually bind are checked: a disabled GUI or
    watchdog cannot clash with anything, and reporting it would be noise.
    """
    targets: list[tuple[str, str, int, str]] = [
        ("server", config.server.host, config.server.port, "server.port"),
    ]
    if config.gui.enabled:
        targets.append(("gui", config.gui.host, config.gui.port, "gui.port"))
    if config.watchdog.enabled:
        targets.append(("watchdog", config.watchdog.host, config.watchdog.port, "watchdog.port"))

    conflicts: list[PortConflict] = []
    for role, host, port, setting in targets:
        probe_host = "0.0.0.0" if host in ("", "::") else host
        if port_is_bindable(port, probe_host):
            continue
        conflicts.append(
            PortConflict(
                role=role,
                port=port,
                host=probe_host,
                holder=find_port_holder(port),
                setting=setting,
            )
        )
    return conflicts


def describe_conflicts(conflicts: list[PortConflict]) -> str:
    """A complete, actionable message for every conflicting port."""
    body = "\n".join(conflict.message() for conflict in conflicts)
    return f"StudioForge cannot start: {len(conflicts)} port(s) are already in use.\n{body}"


# ---------------------------------------------------------------------------
# Restart handover (DECISIONS.md D21)
# ---------------------------------------------------------------------------


def respawn_parent_pid() -> int | None:
    """The pid this process is replacing, when it was spawned to replace one."""
    raw = os.environ.get(ENV_RESPAWN_PARENT_PID, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        log.warning("ignoring a non-numeric respawn parent pid", value=raw)
        return None


def respawn_wait_s() -> float:
    """How long to wait for the process being replaced to release its ports."""
    raw = os.environ.get(ENV_RESPAWN_WAIT_S, "").strip()
    if not raw:
        return DEFAULT_RESPAWN_WAIT_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_RESPAWN_WAIT_S


def adoption_enabled() -> bool:
    """Whether a healthy watchdog on our port may be adopted rather than clashed with."""
    return os.environ.get(ENV_ADOPT_WATCHDOG, "1").strip().lower() not in _FALSEY


def wait_for_ports(conflicts: list[PortConflict], timeout_s: float) -> list[PortConflict]:
    """Re-probe ``conflicts`` until they are free, returning the ones that are not.

    Deliberately does *not* verify who holds each port. Identifying a holder
    needs ``psutil.net_connections``, which is the call that fails without
    elevation on exactly the machines this runs on -- gating the wait on it
    would turn "wait a moment for my parent to exit" into "fail instantly
    because I am not an administrator". Waiting on a port a stranger holds
    costs one bounded delay and then the same clear error as before.
    """
    if not conflicts:
        return []
    deadline = time.monotonic() + max(0.0, timeout_s)
    remaining = list(conflicts)
    while True:
        remaining = [c for c in remaining if not port_is_bindable(c.port, c.host)]
        if not remaining or time.monotonic() >= deadline:
            return remaining
        time.sleep(0.25)


def _same_file(left: str | Path, right: str | Path) -> bool:
    """Whether two paths name the same config file, tolerating case and separators."""
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return str(left).casefold() == str(right).casefold()


def _config_flag(cmdline: list[str]) -> str | None:
    """The value of ``--config``/``-c`` in an argv, or ``None``."""
    found: str | None = None
    for index, token in enumerate(cmdline):
        if token in ("--config", "-c") and index + 1 < len(cmdline):
            found = cmdline[index + 1]
        elif token.startswith("--config="):
            found = token.split("=", 1)[1]
    return found


def find_watchdog_pids(config: Config) -> list[int]:
    """Pids of every StudioForge watchdog process guarding *this* config file.

    Found by command line rather than by port, because the caller is the tray
    taking the whole deployment down: a watchdog that has already lost its
    port (or never bound it) still needs stopping. Identity is the module in
    the argv plus a ``--config`` naming our file; a watchdog with a different
    ``--config`` belongs to another install and is left alone, and one with no
    ``--config`` at all cannot be told apart and is left alone too -- killing
    a stranger's recovery sidecar is worse than leaving one of ours behind.
    Never raises; a process that vanishes mid-scan is skipped.
    """
    ours = str(config.config_path)
    found: list[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = list(proc.info.get("cmdline") or [])
        except (psutil.Error, ValueError):  # pragma: no cover - race with exit
            continue
        if not cmdline or proc.info["pid"] == os.getpid():
            continue
        joined = " ".join(cmdline).lower()
        if "studioforge.watchdog" not in joined and "studioforge-watchdog" not in joined:
            continue
        theirs = _config_flag(cmdline)
        if theirs is None or not _same_file(theirs, ours):
            continue
        found.append(int(proc.info["pid"]))
    return found


@dataclass(slots=True)
class WatchdogPresence:
    """What, if anything, is already listening on the watchdog port."""

    #: True only for a StudioForge watchdog guarding *our* config file.
    adoptable: bool = False
    reason: str = "the watchdog port is free"
    pid: int | None = None
    config_path: str | None = None
    uptime_s: float | None = None


def inspect_running_watchdog(config: Config, *, timeout_s: float = 3.0) -> WatchdogPresence:
    """Ask whatever holds ``watchdog.port`` whether it is our watchdog.

    Identity comes from the watchdog's own ``/health`` body -- which carries the
    ``config_path`` it guards -- rather than from process inspection, because
    ``/health`` needs no credential and no elevation, and answers even while the
    main server it supervises is down. That last part matters: during a restart
    the watchdog reports ``503``/``down``, which is a *healthy watchdog* and must
    still be adoptable.
    """
    port = config.watchdog.port
    url = f"http://127.0.0.1:{port}/health"
    # Connect, do not bind: see port_has_listener for the Windows wildcard trap.
    if not port_has_listener(port, "127.0.0.1"):
        return WatchdogPresence(reason=f"nothing is listening on port {port}")

    import httpx

    try:
        response = httpx.get(url, timeout=timeout_s)
    except Exception as exc:  # noqa: BLE001 - any failure means "not adoptable"
        return WatchdogPresence(
            reason=f"whatever holds port {port} did not answer {url} ({exc.__class__.__name__})"
        )
    try:
        body = response.json()
    except Exception:  # noqa: BLE001 - a non-JSON answer is not our watchdog
        return WatchdogPresence(reason=f"the process on port {port} did not answer /health as JSON")
    if not isinstance(body, dict) or "watchdog_uptime_s" not in body:
        return WatchdogPresence(
            reason=f"the process on port {port} answers /health but is not a StudioForge watchdog"
        )

    theirs = str(body.get("config_path") or "")
    ours = str(config.config_path)
    holder = find_port_holder(port)
    if not theirs or not _same_file(theirs, ours):
        return WatchdogPresence(
            reason=(
                f"the watchdog on port {port} guards {theirs or 'an unknown config'}, not {ours}"
            ),
            pid=holder.pid,
            config_path=theirs or None,
        )
    if not adoption_enabled():
        return WatchdogPresence(
            reason=f"{ENV_ADOPT_WATCHDOG} is off, so the running watchdog is treated as a conflict",
            pid=holder.pid,
            config_path=theirs,
        )
    uptime = body.get("watchdog_uptime_s")
    return WatchdogPresence(
        adoptable=True,
        reason=f"a StudioForge watchdog for {ours} is already running on port {port}",
        pid=holder.pid,
        config_path=theirs,
        uptime_s=float(uptime) if isinstance(uptime, int | float) else None,
    )
