"""Recovery watchdog: the control surface that works when nothing else does.

Why this is a separate process
------------------------------

Every other management surface in StudioForge -- the HTTP API, the GUI, the
management-plane MCP -- lives inside the main application process and reaches
the model manager through the same asyncio event loop and the same locks. That
is the right design for them and exactly the wrong design for recovery: the
failures that most need a remote fix are the ones that take that process out.
A blocked event loop, a deadlocked per-model lock, a CUDA call that never
returns, a config value that makes startup fail, an OOM that leaves a zombie
llama-server holding 24 GiB of VRAM -- in every one of those cases an in-process
tool is exactly as stuck as the thing it is supposed to repair.

So the watchdog is a genuinely separate OS process, on its own port, with its
own event loop, and it deliberately shares **nothing** with the main app:

* It imports only stdlib, ``psutil``, ``httpx``, ``pyyaml``, ``pydantic``,
  ``mcp``, the NVML bindings, and :mod:`studioforge.config` (the schema, so
  config edits are validated by the same rules the app enforces). It never
  imports ``studioforge.core``, ``studioforge.api`` or ``studioforge.db``.
  Importing the app's machinery would drag in the supervisor's process
  bookkeeping, the SQLite registry and the engine manager -- objects that
  acquire the very locks and file handles a wedged app is stuck on, and whose
  import alone can block. A test parses this module's AST and fails the build if
  such an import appears.
* It never opens ``registry.sqlite3``. SQLite locking is process-wide; a
  watchdog waiting on the app's write lock is a watchdog that cannot help.
* It holds no long-lived state derived from the app. Configuration is re-read
  from ``config.yaml`` **on every call**, because the file is the only thing the
  two processes share and a cached copy would be wrong the moment the app (or a
  hand edit, or a previous watchdog call) changed it. Writes go through a temp
  file plus an atomic replace, so a reader can never observe a half-written
  config -- which is precisely the failure that would turn a recovery attempt
  into a worse outage.

Wedged is not the same as down
------------------------------

The distinction the whole design turns on: a **wedged** main app is a process
that exists but does not answer, and it needs to be *killed and restarted*; a
**down** main app is a process that does not exist, and it needs to be
*started*. Reporting one as the other means either restarting a healthy-ish
process for no reason or waiting forever for a process that will never answer.
So the watchdog always corroborates the HTTP probe with a psutil process lookup
and reports the four states separately -- see :meth:`Watchdog.health`.

There is a third, rarer case worth naming: the main app answers HTTP but no
matching process can be found (it runs in a container, under another user, or
behind a proxy). Then the watchdog reports ``degraded``, because inference works
but ``restart_server`` and ``kill_model`` cannot reach anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import hmac
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import psutil
import yaml
from mcp.server.mcpserver import MCPServer

from studioforge.config import (
    RESTART_REQUIRED_KEYS,
    Config,
    apply_overrides,
    find_config_path,
    resolve_data_dir,
)
from studioforge.credential_guard import CredentialGuard, client_key

log = logging.getLogger("studioforge.watchdog")

#: Backwards-compatible alias: the MCP SDK's ``FastMCP`` is now ``MCPServer``.
FastMCP = MCPServer

SERVER_NAME = "studioforge-watchdog"

#: Executable basenames that identify a llama.cpp inference child.
LLAMA_SERVER_STEMS = ("llama-server",)

#: Env override for how the main app is respawned. Set it when StudioForge runs
#: under a wrapper (a venv activator, a container shim, a test harness) instead
#: of plain ``python -m studioforge serve``.
ENV_RESTART_CMD = "SF_WATCHDOG_RESTART_CMD"

#: Env override for the systemd unit name used on Linux.
ENV_SERVICE_UNIT = "SF_WATCHDOG_SERVICE_UNIT"

DEFAULT_SERVICE_UNIT = "studioforge"

#: Seconds to let the driver actually release VRAM before measuring what a kill
#: freed. NVML reports the allocation as live until the CUDA context is torn
#: down, which happens slightly after the process disappears.
VRAM_SETTLE_S = 1.2

HealthStatus = Literal["up", "degraded", "wedged", "down"]

INSTRUCTIONS = """\
StudioForge recovery watchdog: a separate always-on process for fixing a broken
StudioForge server. Use these tools when the main server is slow, unresponsive,
failing to start, or holding VRAM it should have released.

This is NOT the management plane. It cannot list models, load a model or run
inference -- for that, talk to the main server's own MCP endpoint or its
OpenAI-compatible HTTP API. What this server can do is see and repair, and it
keeps working when the main server does not.

Start with `health`. It distinguishes four states, and the difference matters:

  up       - main server and every llama-server child answer.
  degraded - one of them answers and another does not.
  wedged   - the main server's PROCESS EXISTS but will not answer. It needs
             restart_server; waiting will not help.
  down     - no main server process at all. Something must start it.

Then: `get_config`/`set_config` edit config.yaml directly on disk, so they work
with the main server completely dead -- this is the escape hatch when a bad
setting prevents startup. `kill_model` frees one model's VRAM without a full
restart. `reclaim_orphan_engines` frees VRAM held by LEAKED llama-server
processes (our binary, parent gone) and needs no confirmation because it cannot
touch anything a live process owns. `restart_server`, `nuke_all_models` and
`rollback_update` are destructive and require confirm=true.

Tools return {"ok": false, "error": {...}} instead of failing, so read the
message: it says what state was found and what to do about it.
"""


# ---------------------------------------------------------------------------
# Small self-contained helpers (deliberately duplicated -- see module docstring)
# ---------------------------------------------------------------------------


def safe_log_name(model_id: str) -> str:
    """Filesystem-safe log file name for a model id.

    **Deliberate duplication** of :func:`studioforge.core.supervisor.safe_log_name`.
    Importing the supervisor to share ten lines of string munging would pull the
    entire model-supervision stack -- asyncio subprocess bookkeeping, the engine
    manager, httpx clients -- into the recovery process, which is the one thing
    this module must never do. The two implementations must stay byte-identical
    in behaviour; a divergence means ``tail_logs`` reads the wrong file, which
    is a mildly confusing bug, whereas a shared import means the watchdog can
    deadlock with the app it is repairing, which is a total loss of recovery.
    """
    cleaned = []
    for char in model_id:
        cleaned.append(char if (char.isalnum() or char in "-_.") else "_")
    name = "".join(cleaned).strip("._") or "model"
    return name[:120]


def kill_process_tree(
    pid: int, *, timeout: float = 10.0, force: bool = True, exclude: Iterable[int] | None = None
) -> list[int]:
    """Kill ``pid`` and every descendant; returns the pids that were signalled.

    **Deliberate duplication** of the supervisor's function, for the same reason
    as :func:`safe_log_name`. Killing only the named process is not enough: a
    surviving llama-server descendant keeps its CUDA context, and on a GPU-only
    server that is permanently leaked VRAM.

    ``exclude`` names pids the sweep must skip -- **exactly those pids, not
    their descendants**. The watchdog is normally a CHILD of the main server, so
    restarting the server means killing a tree this process is standing in:
    without the exclusion the watchdog killed itself half way through its own
    restart, and ``wait_procs`` would then have been waiting for its own death.
    Callers pass ``os.getpid()``. Protecting descendants too would be actively
    wrong -- after one watchdog-driven restart the *new main server* is a child
    of the watchdog, and the next restart must still be able to kill it. See
    DECISIONS.md D21.
    """
    protected = set(exclude or ())
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return []
    try:
        procs: list[psutil.Process] = parent.children(recursive=True)
    except psutil.Error:
        procs = []
    procs.append(parent)
    procs = [p for p in procs if p.pid not in protected]
    if not procs:
        return []
    pids = [p.pid for p in procs]
    for proc in procs:
        with contextlib.suppress(psutil.Error):
            if force:
                proc.kill()
            else:
                proc.terminate()
    _, alive = psutil.wait_procs(procs, timeout=max(0.0, timeout))
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    if alive:
        psutil.wait_procs(alive, timeout=5.0)
    return pids


def tail_file(path: Path, n: int) -> list[str]:
    """Last ``n`` lines of a text file; ``[]`` when it does not exist.

    A missing log is a normal state (a model that has never been loaded), not an
    error worth failing a diagnostic call over.
    """
    if n <= 0 or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in deque(handle, maxlen=n)]
    except OSError:
        return []


def _needs_confirmation(what: str, consequence: str) -> dict[str, Any]:
    return {
        "ok": False,
        "confirmed": False,
        "error": {
            "message": (
                f"Refusing to {what} without confirmation. {consequence} "
                f"Re-call this tool with confirm=true if that is what you want."
            ),
            "code": "confirmation_required",
            "type": "invalid_request_error",
            "param": "confirm",
        },
    }


def _error(message: str, code: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message, "code": code, "type": "watchdog_error"}
    payload.update(extra)
    return {"ok": False, "error": payload}


def _guard[**P](
    func: Callable[P, Awaitable[dict[str, Any]]],
) -> Callable[P, Awaitable[dict[str, Any]]]:
    """Return every failure as a readable result instead of a protocol error.

    A watchdog tool that raises is a watchdog tool that tells the operator
    nothing. The whole point of this server is to answer while things are
    broken, so "broken" must be a value, not an exception.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all
            log.exception("watchdog tool failed: %s", func.__name__)
            return _error(
                f"{func.__name__} failed: {exc}",
                "internal_error",
                exception=exc.__class__.__name__,
            )

    return wrapper


# ---------------------------------------------------------------------------
# Process discovery
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChildProcess:
    """A discovered ``llama-server`` inference child."""

    pid: int
    name: str
    alias: str | None
    port: int | None
    create_time: float | None = None
    cmdline: list[str] = field(default_factory=list)

    @property
    def attributable(self) -> bool:
        """Whether this process can be attributed to THIS instance.

        Our own children always carry ``--port`` inside the configured range
        (the supervisor puts it on every argv). A llama-server with no
        ``--port`` at all -- LM Studio's, or a hand-run one taking
        ``LLAMA_ARG_PORT`` from the environment -- cannot be attributed, so it
        is *reported* as a VRAM holder but must never be killed: destroying a
        neighbour's loaded model to recover our own is not recovery.
        """
        return self.port is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "alias": self.alias,
            "port": self.port,
            "attributable": self.attributable,
            "uptime_s": (round(time.time() - self.create_time, 1) if self.create_time else None),
        }


def _flag_value(cmdline: Sequence[str], flag: str) -> str | None:
    """Value following ``flag`` in an argv list, or ``None``.

    Handles both ``--port 8080`` and ``--port=8080``; the last occurrence wins,
    matching llama.cpp's own "last flag wins" argument handling.
    """
    found: str | None = None
    for index, token in enumerate(cmdline):
        if token == flag:
            if index + 1 < len(cmdline):
                found = cmdline[index + 1]
        elif token.startswith(flag + "="):
            found = token.split("=", 1)[1]
    return found


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _iter_processes() -> Iterator[dict[str, Any]]:
    """Yield ``{pid, name, cmdline, create_time}`` for every visible process.

    Individual processes that vanish or refuse access mid-iteration are skipped:
    on a busy box that happens constantly and must never abort a health check.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = dict(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            continue
        info["cmdline"] = list(info.get("cmdline") or [])
        yield info


def _looks_like_llama_server(name: str, cmdline: Sequence[str]) -> bool:
    stem = Path(name or "").stem.lower()
    if any(stem.startswith(candidate) for candidate in LLAMA_SERVER_STEMS):
        return True
    if cmdline:
        exe_stem = Path(cmdline[0]).stem.lower()
        if any(exe_stem.startswith(candidate) for candidate in LLAMA_SERVER_STEMS):
            return True
    return False


def find_llama_children(config: Config) -> list[ChildProcess]:
    """Find *this instance's* inference child processes, without asking the app.

    Two independent matchers, because either one alone has a blind spot:

    * the executable is named ``llama-server``, or
    * the argv carries both an ``--alias`` and a ``--port``.

    The second matcher is what finds a child launched through a wrapper (a CUDA
    shim, a profiler, ``taskset``) whose process name is not ``llama-server`` at
    all -- exactly the situation where a leaked child is hardest to spot by hand.

    Both matchers are then **scoped by the configured child port range**. A
    llama-server on a port outside ``gateway.child_port_start..child_port_end``
    is not ours: it belongs to a second StudioForge instance, or to someone
    running llama.cpp by hand. ``nuke_all_models`` kills everything this function
    returns, so the scoping is a safety property, not a nicety -- an unscoped
    match would let one instance's watchdog destroy another instance's loaded
    models. A process with no ``--port`` at all is still reported, since it
    cannot be attributed either way and a nameless VRAM holder is worth seeing.
    """
    low = config.gateway.child_port_start
    high = config.gateway.child_port_end
    found: list[ChildProcess] = []
    for info in _iter_processes():
        cmdline = cast(list[str], info["cmdline"])
        if not cmdline:
            continue
        name = str(info.get("name") or "")
        alias = _flag_value(cmdline, "--alias")
        port = _int_or_none(_flag_value(cmdline, "--port"))
        if port is not None and not (low <= port <= high):
            continue
        matched = _looks_like_llama_server(name, cmdline) or (
            alias is not None and port is not None
        )
        if not matched:
            continue
        found.append(
            ChildProcess(
                pid=int(info["pid"]),
                name=name,
                alias=alias,
                port=port,
                create_time=info.get("create_time"),
                cmdline=cmdline,
            )
        )
    found.sort(key=lambda c: c.pid)
    return found


def engine_orphan_state(child: ChildProcess, engines_root: str) -> dict[str, Any]:
    """Is this discovered child our engine binary, and is its parent gone?

    **Deliberate duplication** of :mod:`studioforge.core.vram_holders`, for the
    same reason as :func:`kill_process_tree` above: importing that module pulls
    in the supervisor, and with it the whole model-supervision stack, into the
    process whose entire value is being independent of it. The two must agree
    on the *rule*, which is stated once here and once there:

    * the executable lives under ``<data_dir>/engines/`` -- nothing else on the
      box launches binaries from there, so such a process is ours by
      construction, and
    * its parent pid no longer resolves to a live process that predates it.

    The second clause carries a pid-reuse guard: a parent that started *after*
    its supposed child cannot have spawned it, it merely inherited the number.
    Without that check a genuine orphan gets filed as somebody's live child and
    survives every sweep.
    """
    exe = child.cmdline[0] if child.cmdline else ""
    try:
        exe_norm = os.path.normcase(os.path.normpath(str(Path(exe).resolve())))
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        exe_norm = os.path.normcase(os.path.normpath(exe))
    under_engines = bool(exe) and (
        exe_norm == engines_root or exe_norm.startswith(engines_root.rstrip(os.sep) + os.sep)
    )

    parent_pid: int | None = None
    parent_name: str | None = None
    parent_alive = False
    parent_recycled = False
    try:
        parent_pid = int(psutil.Process(child.pid).ppid())
    except (psutil.Error, ValueError):
        parent_pid = None
    if parent_pid:
        try:
            parent = psutil.Process(parent_pid)
            with parent.oneshot():
                created = float(parent.create_time())
                parent_name = str(parent.name())
                parent_alive = parent.status() != psutil.STATUS_ZOMBIE
        except (psutil.Error, ValueError):
            parent_alive = False
        else:
            if child.create_time is not None and created > child.create_time + 1.0:
                parent_alive = False
                parent_recycled = True
    return {
        "pid": child.pid,
        "alias": child.alias,
        "port": child.port,
        "exe": exe,
        "under_engines_dir": under_engines,
        "parent_pid": parent_pid,
        "parent_name": parent_name,
        "parent_alive": parent_alive,
        "parent_recycled": parent_recycled,
        "orphan": under_engines and not parent_alive,
    }


def find_orphan_engines(config: Config) -> list[dict[str, Any]]:
    """Every discovered llama-server, classified for orphan reclamation."""
    try:
        engines_root = os.path.normcase(os.path.normpath(str(config.engines_dir.resolve())))
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        engines_root = os.path.normcase(os.path.normpath(str(config.engines_dir)))
    return [engine_orphan_state(child, engines_root) for child in find_llama_children(config)]


def _pid_listening_on(port: int) -> int | None:
    """The pid holding a LISTEN socket on ``port``, if it can be determined.

    Port ownership is the most reliable identification available: it does not
    care how the process was launched, what it is called, or which interpreter
    started it. The cmdline heuristic below is only a fallback for platforms or
    permission setups where connection enumeration is unavailable.
    """
    return _listening_pid(port)[0]


def _listening_pid(port: int) -> tuple[int | None, bool]:
    """``(pid_or_None, enumeration_succeeded)`` for a LISTEN socket on ``port``.

    The second element is what makes "nothing is listening" distinguishable from
    "I could not look", and callers need that distinction: the first is proof the
    server is not running, the second is merely absence of evidence.
    """
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.Error, RuntimeError, OSError):
        return None, False
    for conn in connections:
        if conn.status != psutil.CONN_LISTEN or conn.laddr is None:
            continue
        if getattr(conn.laddr, "port", None) == port and conn.pid:
            return int(conn.pid), True
    return None, True


def _is_watchdog_cmdline(cmdline: Sequence[str]) -> bool:
    joined = " ".join(cmdline).lower()
    return "studioforge.watchdog" in joined or "studioforge-watchdog" in joined


def _is_tray_cmdline(cmdline: Sequence[str]) -> bool:
    """Whether an argv is the StudioForge system tray (``studioforge tray``)."""
    joined = " ".join(cmdline).lower()
    return "studioforge" in joined and ("tray" in joined.split() or "tray_app" in joined)


def launch_parent(proc: psutil.Process) -> psutil.Process | None:
    """The process that *launched* ``proc``: its parent, seen through launcher stubs.

    On Windows a venv's ``Scripts/python.exe`` (uv's and CPython's own) is a
    small redirector: it starts the real interpreter as its child with the same
    arguments and waits, and the two are tied by a job object -- kill the stub
    and the interpreter dies. So the *direct* parent of a StudioForge process
    is very often a stub whose argv is its own, and the process that actually
    decided to start it -- the tray, the watchdog, a shell -- is one hop up.
    Walks up while the ancestor's argv (past the executable) equals ``proc``'s,
    applying the pid-reuse guard at every hop (an ancestor must predate the
    child). Returns ``None`` when there is no such process.
    """
    try:
        argv = list(proc.cmdline())[1:]
        created = float(proc.create_time())
        current = proc
        for _ in range(8):  # stubs do not nest deeper than this; a bound, not a limit
            parent = current.parent()
            if parent is None:
                return None
            with parent.oneshot():
                if parent.status() == psutil.STATUS_ZOMBIE:
                    return None
                if float(parent.create_time()) > created + 1.0:
                    return None
                parent_argv = list(parent.cmdline())
            if parent_argv[1:] != argv or not argv:
                return parent
            current = parent
    except (psutil.Error, ValueError, OSError):
        return None
    return None


def supervising_tray_pid(pid: int) -> int | None:
    """The pid of a live StudioForge tray that launched ``pid``, or ``None``.

    The tray supervises the server as its own child and respawns it when it
    exits (``tray_app.TrayApp._supervise``). A restart performed here by
    killing that child must therefore *not* spawn a replacement: the tray
    will, and two spawns race for the ports (D28). Only the launching parent
    counts (:func:`launch_parent` -- the direct parent, or the process above a
    venv launcher stub) -- a tray somewhere higher up the tree did not launch
    this process and will not respawn it -- and the parent must predate the
    child, the same pid-reuse guard :func:`engine_orphan_state` applies.
    """
    try:
        parent = launch_parent(psutil.Process(pid))
        if parent is None:
            return None
        cmdline = list(parent.cmdline())
    except (psutil.Error, ValueError, OSError):
        return None
    return int(parent.pid) if _is_tray_cmdline(cmdline) else None


def own_launcher_chain(root_pid: int) -> set[int]:
    """Our own pid plus every ancestor of ours that sits *below* ``root_pid``.

    The watchdog is normally a child of the server it restarts; under a venv
    launcher it is a grandchild, with the stub in between -- and the stub's job
    object takes the real interpreter down with it. Killing the server's tree
    with only ``os.getpid()`` excluded killed that stub, and the watchdog died
    in the middle of its own restart with nothing spawned (seen live on this
    box, WP13). The set to protect is the chain from us up to, but never
    including, the target root.
    """
    protected = {os.getpid()}
    try:
        current = psutil.Process(os.getpid())
        for _ in range(16):
            parent = current.parent()
            if parent is None or parent.pid == root_pid:
                break
            protected.add(parent.pid)
            current = parent
    except (psutil.Error, ValueError, OSError):
        pass
    return protected


def _config_matches(cmdline: Sequence[str], config_path: Path | None) -> bool:
    """Whether an argv's ``--config`` names the same file this watchdog guards.

    Several StudioForge instances can share a machine (a stable one and a dev
    one, or one per GPU pair). Matching purely on "the argv mentions
    studioforge and serve" would let this watchdog restart somebody else's
    server, so an explicit ``--config`` that names a different file is a
    definitive mismatch. An argv with no ``--config`` cannot be ruled out and is
    accepted.
    """
    if config_path is None:
        return True
    value = _flag_value(cmdline, "--config") or _flag_value(cmdline, "-c")
    if value is None:
        return True
    try:
        return Path(value).resolve() == config_path.resolve()
    except OSError:  # pragma: no cover - unresolvable path
        return value == str(config_path)


def find_main_process(config: Config, *, exclude_pid: int | None = None) -> ChildProcess | None:
    """Locate the main StudioForge server process for *this* configuration.

    Port ownership is authoritative and is tried first: whoever holds a LISTEN
    socket on ``server.port`` is the server, regardless of how it was launched.
    Crucially, a *successful* enumeration that finds no such listener is also
    authoritative -- it means this instance is not running, and the search stops
    there. Falling through to an argv heuristic at that point is how a watchdog
    ends up "finding" a completely unrelated StudioForge instance and reporting a
    dead server as wedged (or, far worse, restarting the wrong one).

    The argv heuristic therefore runs only when connection enumeration is
    unavailable (a permission-restricted or containerised host), and even then it
    demands that any explicit ``--config`` matches. The watchdog never matches
    itself: this process also has "studioforge" in its command line, and one that
    mistook itself for the app would kill itself during ``restart_server``.
    """
    own = os.getpid() if exclude_pid is None else exclude_pid
    port = config.server.port
    config_path = config.source_path

    pid, enumerated = _listening_pid(port)
    if pid is not None and pid != own:
        try:
            proc = psutil.Process(pid)
            cmdline = list(proc.cmdline())
            if not _is_watchdog_cmdline(cmdline):
                return ChildProcess(
                    pid=pid,
                    name=proc.name(),
                    alias=None,
                    port=port,
                    create_time=proc.create_time(),
                    cmdline=cmdline,
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            pass
    if enumerated:
        return None

    for info in _iter_processes():
        pid = int(info["pid"])
        if pid == own:
            continue
        cmdline = cast(list[str], info["cmdline"])
        if not cmdline or _is_watchdog_cmdline(cmdline):
            continue
        joined = " ".join(cmdline).lower()
        if "studioforge" not in joined or "serve" not in joined:
            continue
        if not _config_matches(cmdline, config_path):
            continue
        return ChildProcess(
            pid=pid,
            name=str(info.get("name") or ""),
            alias=None,
            port=port,
            create_time=info.get("create_time"),
            cmdline=cmdline,
        )
    return None


# ---------------------------------------------------------------------------
# GPU reads, done by the watchdog itself
# ---------------------------------------------------------------------------


def _load_nvml() -> Any:
    """Import the NVML bindings. Split out so tests can make it fail."""
    import pynvml

    return pynvml


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def read_gpus_nvml() -> list[dict[str, Any]]:
    """Per-GPU VRAM straight from NVML, initialised and torn down per call.

    Per-call init/shutdown is a deliberate choice: a long-lived NVML handle in a
    process whose job is to survive driver-level trouble is a liability, and the
    call costs a couple of milliseconds. Every optional field degrades on its
    own so one unsupported query cannot hide a whole device.
    """
    nvml = _load_nvml()
    nvml.nvmlInit()
    try:
        count = int(nvml.nvmlDeviceGetCount())
        gpus: list[dict[str, Any]] = []
        for index in range(count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            entry: dict[str, Any] = {"index": index, "name": "unknown"}
            with contextlib.suppress(Exception):
                entry["name"] = _decode(nvml.nvmlDeviceGetName(handle))
            total = free = used = 0
            with contextlib.suppress(Exception):
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                total, free, used = int(mem.total), int(mem.free), int(mem.used)
            entry.update(
                total_bytes=total,
                free_bytes=free,
                used_bytes=used,
                total_mib=round(total / 1048576),
                free_mib=round(free / 1048576),
                used_mib=round(used / 1048576),
            )
            entry["utilization_pct"] = None
            with contextlib.suppress(Exception):
                entry["utilization_pct"] = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            entry["temperature_c"] = None
            with contextlib.suppress(Exception):
                sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
                entry["temperature_c"] = float(nvml.nvmlDeviceGetTemperature(handle, sensor))
            entry["compute_capability"] = None
            with contextlib.suppress(Exception):
                major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
                entry["compute_capability"] = f"{int(major)}.{int(minor)}"
            gpus.append(entry)
        return gpus
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


_SMI_FIELDS = (
    "index",
    "name",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "temperature.gpu",
)


def read_gpus_smi(timeout: float = 10.0) -> list[dict[str, Any]]:
    """Fallback GPU read by parsing ``nvidia-smi`` CSV output.

    NVML can be unavailable while the driver is perfectly fine -- a missing or
    mismatched ``pynvml``, a partially upgraded driver package. The CLI is a
    separate binary shipped with the driver, so it frequently still works, and a
    slightly less detailed answer beats "no VRAM information" when the operator
    is trying to find out what is holding 24 GiB.
    """
    args = [
        "nvidia-smi",
        f"--query-gpu={','.join(_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < len(_SMI_FIELDS):
            continue
        total_mib = _int_or_none(parts[2]) or 0
        free_mib = _int_or_none(parts[3]) or 0
        used_mib = _int_or_none(parts[4]) or 0
        gpus.append(
            {
                "index": _int_or_none(parts[0]),
                "name": parts[1],
                "total_bytes": total_mib * 1048576,
                "free_bytes": free_mib * 1048576,
                "used_bytes": used_mib * 1048576,
                "total_mib": total_mib,
                "free_mib": free_mib,
                "used_mib": used_mib,
                "utilization_pct": float(parts[5]) if _int_or_none(parts[5]) is not None else None,
                "temperature_c": float(parts[6]) if _int_or_none(parts[6]) is not None else None,
                "compute_capability": None,
            }
        )
    return gpus


# ---------------------------------------------------------------------------
# The watchdog
# ---------------------------------------------------------------------------


class Watchdog:
    """All recovery logic, independent of the MCP transport.

    Constructed with the path to ``config.yaml`` and nothing else -- no app
    objects, no database handle, no shared state. Everything else is read from
    disk, from ``psutil`` or from NVML at the moment it is needed.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        restart_command: Sequence[str] | None = None,
        service_unit: str | None = None,
    ) -> None:
        self.config_path = find_config_path(config_path)
        self._restart_command = list(restart_command) if restart_command else None
        self._service_unit = service_unit or os.environ.get(ENV_SERVICE_UNIT, DEFAULT_SERVICE_UNIT)
        #: Consecutive failed main-app probes, across calls and the poll loop.
        #: The wedged verdict is a *streak*, not a single timeout: one slow
        #: response during a big model load must not read as a wedge.
        self.consecutive_failures = 0
        self.last_status: HealthStatus | None = None
        self.started_at = time.time()
        #: Set for the duration of :meth:`restart_server` and published on
        #: ``/health`` as ``restart_in_progress``. The tray reads it to tell a
        #: restart it did not initiate from a crash of the child it owns (D28):
        #: without it, a GUI "Restart server" killed the tray's child, the tray
        #: counted a crash attempt and respawned, the watchdog respawned too,
        #: and the loser of that race exited on a port conflict -- which the
        #: tray counted as a second crash, then a third, then "Crashed".
        self._restart_in_progress: dict[str, Any] | None = None

    # -- config ----------------------------------------------------------

    def read_raw_config(self) -> tuple[dict[str, Any], str | None]:
        """Raw YAML mapping from disk, plus a parse error if there was one.

        Read fresh every time. The main app owns the same file, and after a
        ``set_config`` the watchdog's own previous view is stale -- but more
        importantly, an operator hand-editing the file to escape a bad setting
        must see their edit reflected immediately.
        """
        if not self.config_path.is_file():
            return {}, f"config file {self.config_path} does not exist"
        try:
            text = self.config_path.read_text(encoding="utf-8")
        except OSError as exc:
            return {}, f"cannot read {self.config_path}: {exc}"
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return {}, f"{self.config_path} is not valid YAML: {exc}"
        if loaded is None:
            return {}, None
        if not isinstance(loaded, dict):
            return {}, f"{self.config_path} must contain a YAML mapping"
        return cast(dict[str, Any], loaded), None

    def load_config(self) -> tuple[Config, str | None]:
        """Validated config, or schema defaults plus the reason it failed.

        Degrading to defaults rather than raising is what makes the watchdog
        usable in its most important scenario: the config file itself is what
        broke the server. ``health``, ``tail_logs`` and ``gpu_status`` keep
        working on best-effort values while ``get_config``/``set_config`` report
        and repair the real problem.
        """
        raw, error = self.read_raw_config()
        # The watchdog is always started with --config, so the data directory
        # is where that file lives (or SF_DATA_DIR), never a key inside it --
        # the same rule load_config applies (D31).
        data_dir = resolve_data_dir(self.config_path, explicit=True)
        if error is not None:
            config = Config(data_dir=data_dir)
            config.source_path = self.config_path
            return config, error
        raw = {k: v for k, v in raw.items() if k != "data_dir"}
        try:
            config = Config(data_dir=data_dir, **raw)
        except Exception as exc:  # noqa: BLE001 - any pydantic failure degrades
            fallback = Config(data_dir=data_dir)
            fallback.source_path = self.config_path
            return fallback, f"invalid configuration in {self.config_path}: {exc}"
        config.source_path = self.config_path
        return config, None

    def write_config(self, config: Config) -> None:
        """Persist atomically: temp file in the same directory, then replace.

        The main app reads this file at startup and the GUI reads it live. A
        partially written file would be a *worse* outage than the one being
        repaired, and ``os.replace`` on the same filesystem is atomic on both
        Windows and POSIX.
        """
        target = self.config_path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(config.to_yaml_dict(), sort_keys=False, default_flow_style=False)
        tmp = target.with_name(target.name + f".watchdog-{os.getpid()}.tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(target)
        finally:
            with contextlib.suppress(OSError):
                if tmp.exists():
                    tmp.unlink()

    def redacted_config(self, config: Config) -> dict[str, Any]:
        """Config as YAML-shaped data with every credential fingerprinted.

        Shares :func:`studioforge.config.redact_config_dict` with the three
        other config-dumping surfaces so a newly added secret cannot be
        redacted in some of them and returned in full here.
        """
        from studioforge.config import redact_config_dict

        return redact_config_dict(config.to_yaml_dict())

    # -- HTTP probes -----------------------------------------------------

    async def _probe(
        self,
        url: str,
        timeout: float,  # noqa: ASYNC109 - httpx owns the deadline, not asyncio
    ) -> tuple[bool, str | None, Any]:
        """One HTTP GET. Returns ``(ok, error, parsed_body)``; never raises."""
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
        except httpx.TimeoutException:
            return False, f"timed out after {timeout:g}s", None
        except (httpx.HTTPError, OSError) as exc:
            return False, f"{exc.__class__.__name__}: {exc}", None
        if response.status_code != 200:
            return False, f"HTTP {response.status_code}", None
        body: Any = None
        with contextlib.suppress(ValueError):
            body = response.json()
        return True, None, body

    def _main_health_url(self, config: Config) -> str:
        host = config.server.host
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{config.server.port}/health"

    # -- health ----------------------------------------------------------

    async def probe_main(self, config: Config) -> dict[str, Any]:
        """Probe the main app and classify it as up / wedged / down.

        The retry loop is the wedged/down discriminator in action. A single
        failed probe proves nothing -- the app may be mid-load, mid-GC or just
        slow. So when the probe fails *and* a process exists, the probe is
        repeated until ``watchdog.wedged_after_failures`` consecutive failures
        have accumulated; only then is the verdict ``wedged``. When no process
        exists there is nothing to wait for and the answer is ``down``
        immediately, which also keeps the common "server not started" case fast
        instead of burning failures x timeout seconds.
        """
        timeout = config.watchdog.health_timeout_s
        threshold = max(1, config.watchdog.wedged_after_failures)
        url = self._main_health_url(config)

        attempts: list[dict[str, Any]] = []
        process = find_main_process(config)
        while True:
            ok, error, body = await self._probe(url, timeout)
            attempts.append({"ok": ok, "error": error})
            if ok:
                self.consecutive_failures = 0
                detail: dict[str, Any] = {
                    "status": "up",
                    "url": url,
                    "attempts": len(attempts),
                    "body": body,
                    "pid": process.pid if process else None,
                    "process_found": process is not None,
                }
                if process is None:
                    # Answers HTTP, but we cannot find it with psutil -- so
                    # restart_server/kill_model have nothing to act on. Say so
                    # rather than reporting a clean bill of health.
                    detail["status"] = "up_unmanaged"
                    detail["reason"] = (
                        "The main server answers /health but no matching process could be "
                        "found on this host, so restart_server and kill_model cannot reach "
                        "it. It is probably running in a container, under another user "
                        "account, or behind a proxy."
                    )
                return detail

            self.consecutive_failures += 1
            process = find_main_process(config)
            if process is None:
                return {
                    "status": "down",
                    "url": url,
                    "attempts": len(attempts),
                    "process_found": False,
                    "consecutive_failures": self.consecutive_failures,
                    "error": error,
                    "reason": (
                        "No StudioForge server process is listening on port "
                        f"{config.server.port} and none could be found by command line. "
                        "The server is not running -- it needs to be started, not "
                        "restarted."
                    ),
                }
            if len(attempts) >= threshold:
                age = round(time.time() - process.create_time, 1) if process.create_time else None
                return {
                    "status": "wedged",
                    "url": url,
                    "attempts": len(attempts),
                    "process_found": True,
                    "pid": process.pid,
                    "process_name": process.name,
                    "process_uptime_s": age,
                    "consecutive_failures": self.consecutive_failures,
                    "error": error,
                    "reason": (
                        f"Process {process.pid} ({process.name}) EXISTS but did not answer "
                        f"{url} within {timeout:g}s on {len(attempts)} consecutive attempts. "
                        "The server is wedged, not dead: waiting will not recover it. Use "
                        "restart_server(confirm=true), which kills the process tree and "
                        "starts a fresh one."
                    ),
                }
            # Brief pause so consecutive attempts are genuinely separate
            # observations rather than one burst against a busy accept queue.
            await asyncio.sleep(min(0.25, timeout / 4 if timeout else 0.25))

    async def probe_children(self, config: Config) -> list[dict[str, Any]]:
        """Probe every discovered llama-server child's own ``/health``.

        Probing the children directly, rather than trusting the main app's
        report, is what catches the case that matters: the app says a model is
        loaded and ready while the child has actually stopped answering. Only
        the child's own port can tell you that.
        """
        timeout = config.watchdog.health_timeout_s
        results: list[dict[str, Any]] = []
        for child in find_llama_children(config):
            entry = child.as_dict()
            if child.port is None:
                entry.update(healthy=None, error="no --port in command line")
                results.append(entry)
                continue
            ok, error, _ = await self._probe(f"http://127.0.0.1:{child.port}/health", timeout)
            entry.update(healthy=ok, error=error)
            results.append(entry)
        return results

    async def health(self) -> dict[str, Any]:
        """Full health verdict: overall status plus per-component detail."""
        config, config_error = self.load_config()
        main = await self.probe_main(config)
        children = await self.probe_children(config)

        main_status = str(main["status"])
        unhealthy = [c for c in children if c.get("healthy") is False]
        if main_status == "down":
            status: HealthStatus = "down"
        elif main_status == "wedged":
            status = "wedged"
        elif main_status == "up_unmanaged" or unhealthy:
            status = "degraded"
        else:
            status = "up"

        summary = {
            "up": "Main server and all inference children are healthy.",
            "degraded": (
                "Partial failure: "
                + (
                    f"{len(unhealthy)} of {len(children)} llama-server children are not answering."
                    if unhealthy
                    else "the main server answers but cannot be managed from here."
                )
            ),
            "wedged": (
                "The main server process exists but is unresponsive. It needs "
                "restart_server(confirm=true)."
            ),
            "down": "No main server process found. It needs to be started.",
        }[status]

        self.last_status = status
        payload: dict[str, Any] = {
            "ok": True,
            "status": status,
            "summary": summary,
            "main": main,
            "children": children,
            "children_total": len(children),
            "children_unhealthy": len(unhealthy),
            "config_path": str(self.config_path),
            "watchdog_uptime_s": round(time.time() - self.started_at, 1),
            "wedged_after_failures": config.watchdog.wedged_after_failures,
            "health_timeout_s": config.watchdog.health_timeout_s,
            # Non-null while restart_server is running. See __init__.
            "restart_in_progress": self._restart_in_progress,
        }
        if config_error is not None:
            payload["config_error"] = config_error
        return payload

    async def poll_loop(self) -> None:
        """Background liveness poll, for logs and the failure streak counter.

        The watchdog does not auto-restart anything: an unattended restart loop
        turns a recoverable wedge into a crash-loop that destroys the evidence.
        Recovery stays an explicit, confirmed decision -- this loop only makes
        sure that when someone asks, the answer is already known and the
        transition is in the log.
        """
        while True:
            config, _ = self.load_config()
            interval = max(1.0, config.watchdog.poll_interval_s)
            try:
                previous = self.last_status
                result = await self.health()
                current = result["status"]
                if current != previous:
                    log.warning(
                        "health transition: %s -> %s (%s)", previous, current, result["summary"]
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the poll loop must never die
                log.warning("health poll failed: %s", exc)
            await asyncio.sleep(interval)

    # -- restart ---------------------------------------------------------

    def restart_command(self, config: Config) -> list[str]:
        """The argv used to start a fresh main app.

        Overridable via ``SF_WATCHDOG_RESTART_CMD`` because the correct command
        is deployment-specific (a wrapper script, a different interpreter, a
        container exec). The default reproduces what a plain install does.
        """
        if self._restart_command:
            return list(self._restart_command)
        override = os.environ.get(ENV_RESTART_CMD)
        if override:
            return shlex.split(override, posix=os.name != "nt")
        return [
            sys.executable,
            "-m",
            "studioforge",
            "serve",
            "--config",
            str(config.source_path or self.config_path),
        ]

    def _spawn_main(self, config: Config) -> tuple[int | None, list[str], str | None]:
        """Start the main app fully detached from the watchdog.

        Detachment is essential: the child must outlive this process and must
        not inherit the watchdog's console, or a Ctrl+C in the watchdog's
        terminal (or the watchdog itself being restarted) would take the server
        down with it -- the exact opposite of the job.
        """
        argv = self.restart_command(config)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            # Windows has no fork/setsid. DETACHED_PROCESS gives the child no
            # console at all and CREATE_NEW_PROCESS_GROUP takes it out of our
            # Ctrl+C group; CREATE_BREAKAWAY_FROM_JOB is deliberately omitted
            # because it fails when the job object forbids breakaway (common
            # under CI and some service wrappers).
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            kwargs["start_new_session"] = True
        # A server we spawn is OUR child: it must not believe a tray will
        # respawn it (D28). Inlined rather than imported from core.ports: the
        # watchdog imports nothing from the app machinery, by contract.
        env = {key: value for key, value in os.environ.items() if key != "SF_SUPERVISOR"}
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv built from config/env, no shell
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path.cwd()),
                env=env,
                **kwargs,
            )
        except OSError as exc:
            return None, argv, f"could not spawn {argv[0]}: {exc}"
        return proc.pid, argv, None

    async def _wait_for_health(
        self,
        config: Config,
        timeout: float,  # noqa: ASYNC109 - polling budget, not a cancellation scope
    ) -> tuple[bool, float]:
        """Poll the main app's ``/health`` until it answers or ``timeout``."""
        url = self._main_health_url(config)
        started = time.monotonic()
        step = min(1.0, max(0.1, config.watchdog.health_timeout_s / 2))
        while time.monotonic() - started < timeout:
            ok, _, _ = await self._probe(url, config.watchdog.health_timeout_s)
            if ok:
                return True, round(time.monotonic() - started, 2)
            await asyncio.sleep(step)
        return False, round(time.monotonic() - started, 2)

    async def _wait_port_free(
        self,
        port: int,
        timeout: float = 10.0,  # noqa: ASYNC109 - polling budget, not a cancellation scope
    ) -> bool:
        """Wait for ``port`` to be released before respawning.

        Without this the new process can lose a race with the dying one and fail
        to bind, which would look exactly like "the restart did not work".
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _pid_listening_on(port) is None:
                return True
            await asyncio.sleep(0.2)
        return _pid_listening_on(port) is None

    async def _systemctl_restart(self, unit: str) -> tuple[bool, str]:
        try:
            completed = await asyncio.to_thread(
                subprocess.run,  # noqa: S603 - fixed argv, no shell
                ["systemctl", "restart", unit],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"systemctl unavailable: {exc}"
        if completed.returncode == 0:
            return True, f"systemctl restart {unit}"
        detail = (completed.stderr or completed.stdout or "").strip()
        return False, f"systemctl restart {unit} failed (rc={completed.returncode}): {detail}"

    async def restart_server(
        self, *, confirm: bool = False, timeout_s: float = 120.0
    ) -> dict[str, Any]:
        """Kill and restart the main app; see the MCP tool docstring."""
        if not confirm:
            return _needs_confirmation(
                "restart the StudioForge server",
                "In-flight inference requests will be aborted and every loaded model "
                "will be unloaded (all VRAM released, then reloaded on demand).",
            )
        config, config_error = self.load_config()
        process = find_main_process(config)
        # A tray that launched the server respawns it itself; see _spawn below.
        tray_pid = supervising_tray_pid(process.pid) if process is not None else None
        self._restart_in_progress = {
            "since": time.time(),
            "previous_pid": process.pid if process else None,
            "respawned_by": "tray" if tray_pid else "watchdog",
            "tray_pid": tray_pid,
        }
        try:
            return await self._restart_server(
                config, config_error, process, tray_pid=tray_pid, timeout_s=timeout_s
            )
        finally:
            self._restart_in_progress = None

    async def _restart_server(
        self,
        config: Config,
        config_error: str | None,
        process: ChildProcess | None,
        *,
        tray_pid: int | None,
        timeout_s: float,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "method": None,
            "killed_pids": [],
            "previous_pid": process.pid if process else None,
            "new_pid": None,
            "healthy": False,
            "waited_s": 0.0,
            "respawned_by": "tray" if tray_pid else "watchdog",
        }
        if config_error is not None:
            result["config_error"] = config_error

        # Linux: let the service manager own the lifecycle when it does. It
        # handles ordering, restart policy and logging far better than we can,
        # and bypassing it would leave systemd's idea of the unit's state wrong.
        if os.name != "nt":
            ok, detail = await self._systemctl_restart(self._service_unit)
            result["systemctl"] = detail
            if ok:
                result["method"] = "systemctl"
                healthy, waited = await self._wait_for_health(config, timeout_s)
                result.update(ok=healthy, healthy=healthy, waited_s=waited)
                if not healthy:
                    result["error"] = {
                        "message": (
                            f"systemctl restarted {self._service_unit} but /health did not "
                            f"answer within {timeout_s:g}s. Check `journalctl -u "
                            f"{self._service_unit}` and tail_logs()."
                        ),
                        "code": "restart_unhealthy",
                        "type": "watchdog_error",
                    }
                return result

        # Windows (and any box without a working service manager): do it by
        # hand. Kill the whole tree -- a surviving llama-server child would keep
        # its CUDA context and the fresh server would then fail to fit models
        # into VRAM that nothing appears to be using.
        #
        # `exclude`: we are usually a CHILD of the process we are killing, so
        # our own pid is inside that tree. Without the exclusion this call
        # killed the watchdog in the middle of the restart it was performing,
        # and no fresh server was ever spawned (D21).
        keep_alive = own_launcher_chain(process.pid) if process is not None else {os.getpid()}
        if process is not None:
            log.warning("restarting the main server: killing pid %s", process.pid)
            killed = await asyncio.to_thread(
                kill_process_tree, process.pid, timeout=15.0, exclude=keep_alive
            )
            result["killed_pids"] = killed
        else:
            log.warning("no main server process found; starting one")
        for child in find_llama_children(config):
            if not child.attributable:
                # Reported-but-unattributable (no --port): not provably ours,
                # so it survives the sweep. See ChildProcess.attributable.
                continue
            killed_child = await asyncio.to_thread(
                kill_process_tree, child.pid, timeout=10.0, exclude=keep_alive
            )
            result["killed_pids"] = list(result["killed_pids"]) + killed_child
        # Every port the replacement's own preflight will check, not just the
        # API one: the GUI shares the main process, so it frees at the same
        # moment, and a replacement that raced it died at startup. Our own
        # watchdog port is deliberately NOT waited on -- we are holding it, and
        # the replacement adopts us (D21).
        freed = {config.server.port: await self._wait_port_free(config.server.port)}
        if config.gui.enabled:
            freed[config.gui.port] = await self._wait_port_free(config.gui.port)
        result["ports_free"] = freed
        if not all(freed.values()):
            log.warning("some ports were still held after the kill: %s", freed)

        result["watchdog_pid"] = os.getpid()
        if tray_pid is not None:
            # The server was the tray's child, and the tray respawns a child
            # that exits -- it reads our /health (`restart_in_progress`) to
            # know this exit was asked for. Spawning here as well would give
            # the deployment two servers racing for the ports, and the loser's
            # exit code 3 would be counted by the tray as a crash (D28).
            result["method"] = "process-tree kill; the tray that owns the server respawns it"
            result["tray_pid"] = tray_pid
            log.warning(
                "the server was launched by the tray (pid %s); leaving the respawn to it "
                "and waiting for /health",
                tray_pid,
            )
            healthy, waited = await self._wait_for_health(config, timeout_s)
            result.update(ok=healthy, healthy=healthy, waited_s=waited)
            if healthy:
                replacement = find_main_process(config)
                result["new_pid"] = replacement.pid if replacement else None
            else:
                result["error"] = {
                    "message": (
                        f"The server was killed and the tray (pid {tray_pid}) that launched "
                        f"it was left to respawn it, but /health did not answer within "
                        f"{timeout_s:g}s. Check the tray's status line and "
                        f"logs/tray-server.log; if the tray is gone, restart_server again "
                        f"and the watchdog will spawn the server itself."
                    ),
                    "code": "restart_unhealthy",
                    "type": "watchdog_error",
                }
            return result

        pid, argv, error = self._spawn_main(config)
        result["method"] = "process-tree kill + respawn"
        result["command"] = argv
        result["new_pid"] = pid
        if error is not None:
            log.error("could not spawn the replacement server: %s", error)
            result["error"] = {"message": error, "code": "spawn_failed", "type": "watchdog_error"}
            return result
        log.warning("spawned the replacement server (pid %s); waiting for /health", pid)

        healthy, waited = await self._wait_for_health(config, timeout_s)
        result.update(ok=healthy, healthy=healthy, waited_s=waited)
        if not healthy:
            result["error"] = {
                "message": (
                    f"Started {' '.join(argv)} (pid {pid}) but /health did not answer within "
                    f"{timeout_s:g}s. The process may still be scanning the model library, or "
                    f"it may have failed to start -- call tail_logs() to see why, and "
                    f"get_config() to check for a bad setting."
                ),
                "code": "restart_unhealthy",
                "type": "watchdog_error",
            }
        return result

    # -- model kills -----------------------------------------------------

    def _vram_snapshot(self) -> dict[int, int] | None:
        try:
            return {int(g["index"]): int(g["used_bytes"]) for g in read_gpus_nvml()}
        except Exception:  # noqa: BLE001 - VRAM accounting is a nice-to-have
            return None

    @staticmethod
    def _vram_delta(before: dict[int, int] | None, after: dict[int, int] | None) -> dict[str, Any]:
        if before is None or after is None:
            return {
                "vram_freed_bytes": None,
                "vram_freed_mib": None,
                "note": "NVML unavailable; VRAM accounting skipped.",
            }
        per_gpu = {
            index: before[index] - after.get(index, before[index]) for index in sorted(before)
        }
        total = sum(per_gpu.values())
        return {
            "vram_freed_bytes": total,
            "vram_freed_mib": round(total / 1048576),
            "vram_freed_per_gpu_mib": {k: round(v / 1048576) for k, v in per_gpu.items()},
            "note": (
                "Measured as total NVML used-VRAM before minus after; other activity on "
                "the GPUs can skew it slightly."
            ),
        }

    def _match_children(self, config: Config, model_name: str) -> list[ChildProcess]:
        """Find children whose ``--alias`` names ``model_name``.

        Exact match first, then case-insensitive, then substring -- an operator
        (or an agent reading a truncated log line) rarely has the full id, and
        refusing a near-miss is unhelpful when there is exactly one candidate.
        """
        children = find_llama_children(config)
        exact = [c for c in children if c.alias == model_name]
        if exact:
            return exact
        lowered = model_name.lower()
        ci = [c for c in children if (c.alias or "").lower() == lowered]
        if ci:
            return ci
        return [c for c in children if lowered and lowered in (c.alias or "").lower()]

    async def kill_model(self, model_name: str) -> dict[str, Any]:
        """SIGKILL one inference child; see the MCP tool docstring."""
        config, _ = self.load_config()
        matches = self._match_children(config, model_name)
        if not matches:
            known = [c.alias for c in find_llama_children(config) if c.alias]
            return _error(
                f"No running llama-server child has an --alias matching '{model_name}'."
                + (
                    f" Running aliases: {', '.join(sorted(known))}."
                    if known
                    else " No llama-server children are running at all."
                ),
                "model_not_running",
                running_aliases=sorted(known),
            )
        before = self._vram_snapshot()
        killed: list[dict[str, Any]] = []
        for child in matches:
            pids = await asyncio.to_thread(kill_process_tree, child.pid, timeout=10.0)
            gone = not psutil.pid_exists(child.pid)
            killed.append(
                {
                    "pid": child.pid,
                    "alias": child.alias,
                    "port": child.port,
                    "tree_pids": pids,
                    "gone": gone,
                }
            )
        await asyncio.sleep(VRAM_SETTLE_S)
        after = self._vram_snapshot()
        return {
            "ok": all(entry["gone"] for entry in killed),
            "model": model_name,
            "killed": killed,
            "pids": [entry["pid"] for entry in killed],
            **self._vram_delta(before, after),
        }

    async def nuke_all_models(self, *, confirm: bool = False) -> dict[str, Any]:
        """Kill every inference child; see the MCP tool docstring."""
        if not confirm:
            return _needs_confirmation(
                "kill every llama-server child process",
                "Every loaded model dies immediately and all in-flight inference "
                "requests fail. The main server stays up and will reload models on "
                "demand.",
            )
        config, _ = self.load_config()
        discovered = find_llama_children(config)
        # A llama-server with no --port cannot be attributed to this instance
        # (LM Studio and hand-run servers configured via env land here), so it
        # is listed as spared, never killed.
        spared = [child.as_dict() for child in discovered if not child.attributable]
        children = [child for child in discovered if child.attributable]
        if not children:
            return {
                "ok": True,
                "killed": [],
                "count": 0,
                "spared": spared,
                "message": "No llama-server children were running; nothing to do.",
            }
        before = self._vram_snapshot()
        killed: list[dict[str, Any]] = []
        for child in children:
            pids = await asyncio.to_thread(kill_process_tree, child.pid, timeout=10.0)
            killed.append(
                {
                    "pid": child.pid,
                    "alias": child.alias,
                    "port": child.port,
                    "tree_pids": pids,
                    "gone": not psutil.pid_exists(child.pid),
                }
            )
        await asyncio.sleep(VRAM_SETTLE_S)
        after = self._vram_snapshot()
        return {
            "ok": all(entry["gone"] for entry in killed),
            "killed": killed,
            "count": len(killed),
            "spared": spared,
            "survivors": [e["pid"] for e in killed if not e["gone"]],
            **self._vram_delta(before, after),
        }

    async def reclaim_orphan_engines(self, *, dry_run: bool = False) -> dict[str, Any]:
        """Kill leaked engine children; see the MCP tool docstring."""
        config, _ = self.load_config()
        classified = await asyncio.to_thread(find_orphan_engines, config)
        orphans = [entry for entry in classified if entry["orphan"]]
        spared = [entry for entry in classified if not entry["orphan"]]
        if not orphans:
            return {
                "ok": True,
                "dry_run": dry_run,
                "orphans": [],
                "count": 0,
                "spared": spared,
                "message": (
                    "No orphaned llama-server processes found. "
                    f"{len(spared)} llama-server process(es) are running with a live parent; "
                    "they are listed in `spared` and were not touched."
                ),
            }
        before = self._vram_snapshot()
        killed: list[dict[str, Any]] = []
        for entry in orphans:
            record = dict(entry)
            if dry_run:
                record["gone"] = False
            else:
                record["tree_pids"] = await asyncio.to_thread(
                    kill_process_tree, entry["pid"], timeout=10.0
                )
                record["gone"] = not psutil.pid_exists(entry["pid"])
            killed.append(record)
        if dry_run:
            return {
                "ok": True,
                "dry_run": True,
                "orphans": killed,
                "count": len(killed),
                "spared": spared,
                "message": f"{len(killed)} orphan(s) would be killed. Re-call with dry_run=false.",
            }
        await asyncio.sleep(VRAM_SETTLE_S)
        after = self._vram_snapshot()
        return {
            "ok": all(entry["gone"] for entry in killed),
            "dry_run": False,
            "orphans": killed,
            "count": len(killed),
            "spared": spared,
            "survivors": [entry["pid"] for entry in killed if not entry["gone"]],
            **self._vram_delta(before, after),
        }

    # -- logs ------------------------------------------------------------

    def tail_logs(self, n: int = 200, model_id: str | None = None) -> dict[str, Any]:
        """Read log tails straight off disk; see the MCP tool docstring."""
        config, config_error = self.load_config()
        server_log = config.logs_dir / "studioforge.log"
        watchdog_log = config.logs_dir / "watchdog.log"
        payload: dict[str, Any] = {
            "ok": True,
            "server_log_path": str(server_log),
            "server_log_exists": server_log.is_file(),
            "server": tail_file(server_log, n),
            "watchdog_log_path": str(watchdog_log),
            "watchdog": tail_file(watchdog_log, n),
        }
        if model_id is not None:
            model_log = config.model_logs_dir / f"{safe_log_name(model_id)}.log"
            payload.update(
                model_id=model_id,
                model_log_path=str(model_log),
                model_log_exists=model_log.is_file(),
                model=tail_file(model_log, n),
            )
        if config_error is not None:
            payload["config_error"] = config_error
        return payload

    # -- releases / rollback ---------------------------------------------

    def _releases(self, config: Config) -> list[Path]:
        """Release directories, oldest first.

        Ordered by directory mtime with the name as a tiebreak, rather than by
        parsing version strings: an install's mtime is what actually records the
        order things were deployed in, and it does not need to understand
        whatever versioning scheme is in use.
        """
        root = config.releases_dir
        if not root.is_dir():
            return []
        entries = [p for p in root.iterdir() if p.is_dir()]

        def key(path: Path) -> tuple[float, str]:
            try:
                return (path.stat().st_mtime, path.name)
            except OSError:
                return (0.0, path.name)

        return sorted(entries, key=key)

    def _current_pointer(self, config: Config) -> tuple[Path, Path | None, str]:
        """Locate the ``current`` release pointer and read it.

        Returns ``(pointer_path, target_or_None, kind)`` where ``kind`` is
        ``"file"``, ``"link"``, ``"dir"`` or ``"missing"``.

        **Windows choice: a ``current.txt`` pointer file.** A real symlink needs
        SeCreateSymbolicLinkPrivilege (i.e. admin or Developer Mode), which a
        recovery tool cannot assume it has; a directory junction avoids that but
        can only be created through ``mklink /J`` or an undocumented reparse-point
        ioctl, cannot be replaced atomically, and confuses ordinary tools that
        try to delete it recursively. A one-line text file naming the active
        release directory needs no privileges, is written atomically with the
        same temp-file-plus-replace trick as ``config.yaml``, is trivially
        readable by a launcher script, and works on every filesystem including
        network shares. On POSIX a symlink at ``<data_dir>/current`` is still
        preferred and read first, since that is the idiomatic form there.
        """
        data_dir = config.data_dir
        pointer_file = data_dir / "current.txt"
        link = data_dir / "current"
        if os.name != "nt" and link.is_symlink():
            with contextlib.suppress(OSError):
                return link, link.readlink(), "link"
        if pointer_file.is_file():
            try:
                name = pointer_file.read_text(encoding="utf-8").strip()
            except OSError:
                return pointer_file, None, "file"
            if not name:
                return pointer_file, None, "file"
            target = Path(name)
            if not target.is_absolute():
                target = config.releases_dir / name
            return pointer_file, target, "file"
        if link.is_symlink():
            with contextlib.suppress(OSError):
                return link, link.readlink(), "link"
        if link.is_dir():
            return link, link, "dir"
        return (pointer_file if os.name == "nt" else link), None, "missing"

    def _write_current(self, config: Config, target: Path) -> dict[str, Any]:
        """Point ``current`` at ``target``, using the platform-appropriate form."""
        data_dir = config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)
        if os.name != "nt":
            link = data_dir / "current"
            tmp = data_dir / f"current.{os.getpid()}.tmp"
            with contextlib.suppress(OSError):
                if tmp.is_symlink() or tmp.exists():
                    tmp.unlink()
            try:
                tmp.symlink_to(target, target_is_directory=True)
                tmp.replace(link)
                return {"pointer": str(link), "kind": "link"}
            except OSError as exc:
                with contextlib.suppress(OSError):
                    if tmp.is_symlink() or tmp.exists():
                        tmp.unlink()
                log.warning("symlink flip failed, falling back to current.txt: %s", exc)
        pointer = data_dir / "current.txt"
        tmp_file = data_dir / f"current.txt.{os.getpid()}.tmp"
        tmp_file.write_text(target.name + "\n", encoding="utf-8")
        tmp_file.replace(pointer)
        return {"pointer": str(pointer), "kind": "file"}

    async def rollback_update(
        self, *, confirm: bool = False, timeout_s: float = 120.0
    ) -> dict[str, Any]:
        """Flip ``current`` back one release and restart; see the tool docstring."""
        config, _ = self.load_config()
        releases = self._releases(config)
        pointer, target, kind = self._current_pointer(config)
        names = [p.name for p in releases]

        if len(releases) < 2:
            return _error(
                "Cannot roll back: "
                + (
                    f"only one release is installed ({names[0]})."
                    if len(releases) == 1
                    else f"no releases found under {config.releases_dir}."
                )
                + " A rollback needs a previous release directory to switch to.",
                "no_previous_release",
                releases_dir=str(config.releases_dir),
                releases=names,
            )

        index: int | None = None
        if target is not None:
            for position, release in enumerate(releases):
                if release.name == target.name:
                    index = position
                    break
        if index is None:
            # Pointer missing or naming something that is no longer installed:
            # the newest release is the best available guess at "current", so
            # rolling back means the one before it.
            index = len(releases) - 1
        if index == 0:
            return _error(
                f"Cannot roll back: {releases[0].name} is the oldest installed release. "
                f"Installed (oldest first): {', '.join(names)}.",
                "no_previous_release",
                releases=names,
                current=releases[index].name,
            )

        previous = releases[index - 1]
        if not confirm:
            return _needs_confirmation(
                f"roll back from {releases[index].name} to {previous.name}",
                "The server will be restarted onto the previous release; in-flight "
                "requests are aborted and all models unload.",
            )

        pointer_info = self._write_current(config, previous)
        log.warning("rolled back current -> %s", previous.name)
        restart = await self.restart_server(confirm=True, timeout_s=timeout_s)
        return {
            "ok": bool(restart.get("ok")),
            "rolled_back_from": releases[index].name,
            "rolled_back_to": previous.name,
            "previous_pointer": str(pointer),
            "previous_pointer_kind": kind,
            "pointer_path": pointer_info["pointer"],
            "pointer_kind": pointer_info["kind"],
            "releases": names,
            "restart": restart,
            "note": (
                f"'current' is a {pointer_info['kind']} at {pointer_info['pointer']}. "
                "On Windows a pointer FILE is used rather than a symlink, because "
                "symlinks require elevation."
            ),
        }

    # -- gpus ------------------------------------------------------------

    def gpu_status(self) -> dict[str, Any]:
        """Read GPU VRAM; see the MCP tool docstring."""
        errors: list[str] = []
        try:
            gpus = read_gpus_nvml()
            if gpus:
                return {"ok": True, "source": "nvml", "count": len(gpus), "gpus": gpus}
            errors.append("NVML initialised but reported zero devices")
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            errors.append(f"NVML failed: {exc}")
        try:
            gpus = read_gpus_smi()
            if gpus:
                return {
                    "ok": True,
                    "source": "nvidia-smi",
                    "count": len(gpus),
                    "gpus": gpus,
                    "degraded": errors,
                }
            errors.append("nvidia-smi returned no parsable rows")
        except Exception as exc:  # noqa: BLE001 - degrade, never raise
            errors.append(f"nvidia-smi failed: {exc}")
        return _error(
            "Could not read GPU state by any method. Tried NVML and nvidia-smi: "
            + "; ".join(errors)
            + ". The NVIDIA driver may be absent, crashed, or mid-upgrade.",
            "gpu_unavailable",
            attempts=errors,
        )


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------


def build_watchdog_mcp(watchdog: Watchdog) -> MCPServer:
    """Wrap a :class:`Watchdog` in an MCP server exposing the recovery tools."""
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="StudioForge recovery watchdog",
        instructions=INSTRUCTIONS,
    )

    @_guard
    async def health() -> dict[str, Any]:
        """Check whether StudioForge is up, degraded, wedged or down.

        Always call this first. It probes the main server's /health AND finds
        every llama-server child process itself with psutil, probing each
        child's own /health too -- so it does not depend on the main server
        being able to tell the truth about its children.

        The four statuses mean different actions:

          up       - everything answers. Nothing to do.
          degraded - main server and children disagree; check `children` for
                     which one is unresponsive, and consider kill_model on it.
          wedged   - the main server PROCESS EXISTS but does not answer within
                     the health timeout, on several consecutive attempts. It
                     will not recover on its own: use restart_server(confirm=true).
          down     - no main server process exists. A restart cannot help
                     something that is not running; it must be started (by the
                     service manager, or restart_server which will also spawn a
                     fresh one).

        Returns:
            Overall status and summary, plus a `main` block (URL probed, attempt
            count, pid, uptime, why it failed) and a `children` list (pid,
            alias, port, healthy).
        """
        return await watchdog.health()

    @_guard
    async def get_config() -> dict[str, Any]:
        """Read config.yaml directly from disk, with secrets redacted.

        This reads the FILE, not the running server's memory, so it works when
        the main server is wedged or dead -- which is the point: when a bad
        setting stops the server from starting, this is how you see it. If the
        file is currently invalid, the error is reported in `config_error`
        alongside the raw contents, so you can tell exactly what to fix.

        `server.api_key`, `mcp.pin` and `hf.token` come back as short
        fingerprints, never in full.

        Returns:
            The parsed config, the raw file text, the file path, any validation
            error, and which keys need a restart to take effect.
        """
        config, error = watchdog.load_config()
        raw, raw_error = watchdog.read_raw_config()
        payload: dict[str, Any] = {
            "ok": error is None,
            "config_path": str(watchdog.config_path),
            "config": watchdog.redacted_config(config),
            "restart_required_keys": sorted(RESTART_REQUIRED_KEYS),
            "raw_keys": sorted(raw),
        }
        if error is not None or raw_error is not None:
            payload["config_error"] = error or raw_error
            payload["hint"] = (
                "The config file on disk is not usable as-is; the values shown are schema "
                "defaults. Use set_config to correct the offending key."
            )
        return payload

    @_guard
    async def set_config(updates: dict[str, Any]) -> dict[str, Any]:
        """Edit config.yaml on disk, validating against the real schema.

        THE ESCAPE HATCH. Because this reads and writes the file directly, it
        works with the main server completely dead -- so a setting that prevents
        startup (an impossible port, a bad model directory, a port collision)
        can be fixed from here and then `restart_server` brings the server back.

        Takes dotted paths, e.g. {"server.port": 1234, "models.default_ctx":
        8192}. Validation is all-or-nothing and uses the same schema the server
        itself enforces: if any value is rejected, NOTHING is written, so a bad
        call cannot make the situation worse. The write is atomic (temp file plus
        rename), so a reader can never see a half-written config.

        Args:
            updates: Mapping of dotted config path to new value. Unknown keys are
                rejected rather than silently ignored.

        Returns:
            Which keys were written, which of them need a restart, and the file
            path. On rejection, the validation error explaining why.
        """
        if not isinstance(updates, dict) or not updates:
            return _error(
                "updates must be a non-empty object of dotted config paths, e.g. "
                '{"models.default_ctx": 8192}',
                "invalid_request",
                param="updates",
            )
        base, base_error = watchdog.load_config()
        try:
            if base_error is None:
                # Normal path: one validation implementation, shared with the
                # main app's own set_config.
                updated = apply_overrides(base, updates)
            else:
                # Recovery path: the file on disk does not validate, so there is
                # no valid base Config to layer onto. Merge into the RAW mapping
                # instead -- checking each key against the schema's shape -- so
                # the operator's other settings survive being fixed.
                raw, _ = watchdog.read_raw_config()
                raw = {k: v for k, v in raw.items() if k != "data_dir"}
                merged = _set_dotted(raw, updates, reference=Config().to_yaml_dict())
                updated = Config(
                    data_dir=resolve_data_dir(watchdog.config_path, explicit=True), **merged
                )
        except Exception as exc:  # noqa: BLE001 - rejection is a result
            return _error(
                f"Rejected: {exc} Nothing was written to {watchdog.config_path}.",
                "invalid_config",
                updates=sorted(updates),
            )
        updated.source_path = watchdog.config_path
        watchdog.write_config(updated)
        changed = sorted(updates)
        needs_restart = [key for key in changed if key in RESTART_REQUIRED_KEYS]
        log.warning("config updated on disk via watchdog: %s", changed)
        return {
            "ok": True,
            "updated": changed,
            "restart_required": needs_restart,
            "config_path": str(watchdog.config_path),
            "repaired_invalid_config": base_error is not None,
            "note": (
                "Written to disk. The running server does not re-read config.yaml, so "
                "call restart_server(confirm=true) for these to take effect."
                if needs_restart
                else "Written to disk. A restart is not required for these keys, but a "
                "running server keeps its in-memory values until it restarts."
            ),
        }

    @_guard
    async def restart_server(confirm: bool = False, timeout_s: float = 120.0) -> dict[str, Any]:
        """Restart the main StudioForge server. DESTRUCTIVE; needs confirm=true.

        Use when `health` reports `wedged`: the process exists but will not
        answer, and nothing short of killing it will help. Also use after
        set_config changed a restart-required key.

        What it does, in order: on Linux, `systemctl restart` the unit (name
        from SF_WATCHDOG_SERVICE_UNIT, default "studioforge") and stop there if
        that worked. Otherwise -- always on Windows, which has no systemd -- it
        kills the server's whole process TREE with psutil (not just the parent:
        a surviving llama-server child would keep its CUDA context and leak
        VRAM), kills any remaining inference children, waits for the port to be
        released, then spawns a fresh detached server and polls /health until it
        answers.

        Consequences: every in-flight inference request fails, every loaded
        model is unloaded and its VRAM released. Models reload on demand
        afterwards, so the first request after a restart is slow.

        Args:
            confirm: Must be true. Without it nothing happens.
            timeout_s: How long to wait for the new server to answer /health.
                A cold start that rescans a large model library can take a
                while; the default is generous.

        Returns:
            The method used, the pids killed, the new pid, whether /health came
            back and how long it took. If it did not come back, an error telling
            you to check tail_logs() -- a config error will show up there.
        """
        return await watchdog.restart_server(confirm=confirm, timeout_s=timeout_s)

    @_guard
    async def kill_model(model_name: str) -> dict[str, Any]:
        """SIGKILL one model's llama-server child, freeing its VRAM.

        The surgical alternative to restart_server: it frees one model's VRAM
        without touching the main server or any other loaded model. Use it when
        a single model is wedged (`health` shows that child unhealthy), or when
        you need VRAM back right now for something bigger and the main server is
        not responding to a normal unload.

        Matching is on the child's `--alias`, which is the model id -- exact
        first, then case-insensitive, then substring, so a partial name works
        when it is unambiguous. `health` lists the running aliases.

        The whole process tree is killed, because a surviving descendant keeps
        its CUDA context and the VRAM stays allocated.

        Args:
            model_name: Model id / alias of the child to kill.

        Returns:
            The pids killed and how much VRAM that freed, measured with NVML
            before and after. VRAM accounting degrades to null if NVML is
            unavailable; the kill still happens.
        """
        return await watchdog.kill_model(model_name)

    @_guard
    async def nuke_all_models(confirm: bool = False) -> dict[str, Any]:
        """Kill EVERY llama-server child process. DESTRUCTIVE; needs confirm=true.

        The "give me all my VRAM back now" button. Every loaded model dies
        immediately and every in-flight inference request fails. The main server
        process is left alone and will reload models on demand, so this is much
        less disruptive than restart_server -- use it when VRAM is exhausted or
        when a stale child is holding memory the main server has lost track of.

        Args:
            confirm: Must be true. Without it nothing happens.

        Returns:
            Every pid killed, with alias and port, plus the VRAM freed and any
            survivors (a pid that refused to die needs OS-level attention).
        """
        return await watchdog.nuke_all_models(confirm=confirm)

    @_guard
    async def reclaim_orphan_engines(dry_run: bool = False) -> dict[str, Any]:
        """Kill LEAKED llama-server processes -- the ones nothing owns any more.

        The narrow, always-safe version of nuke_all_models, and it needs no
        confirmation because of how narrow it is: a process only qualifies when
        its executable lives under this install's `engines/` directory AND its
        parent process no longer exists. Nothing else on the box launches
        binaries from there, and nothing alive is waiting on it, so its VRAM is
        pure leak.

        What it will NOT touch: a llama-server whose parent is still running.
        That is somebody's working process -- another StudioForge instance, a
        test run, a hand-started server -- and killing it to recover memory is
        not recovery. Those are returned in `spared`, with the parent named, so
        you can decide. Use nuke_all_models if you really want them gone.

        Args:
            dry_run: List what would be killed without killing anything.

        Returns:
            `orphans` (pid, alias, port, exe, parent pid/name, whether the
            parent's pid had been recycled), `spared`, and the VRAM freed
            measured with NVML before and after.
        """
        return await watchdog.reclaim_orphan_engines(dry_run=dry_run)

    @_guard
    async def tail_logs(n: int = 200, model_id: str | None = None) -> dict[str, Any]:
        """Read the tail of the server, watchdog and per-model log files.

        Reads the files directly from <data_dir>/logs/, so it works when the
        main server is wedged or dead and its own /api/logs endpoint is
        unreachable. This is where a failed startup explains itself.

        Pass `model_id` to also get that model's llama-server stderr from
        <data_dir>/logs/models/, which is where load failures (bad flags, out
        of VRAM, corrupt GGUF) are actually reported. A model that has never
        been loaded has no log file yet; that returns an empty list, not an
        error.

        Args:
            n: Number of trailing lines per file.
            model_id: Optional model id to also fetch the per-model log for.

        Returns:
            Line lists plus the resolved paths and whether each file exists.
        """
        return watchdog.tail_logs(n=n, model_id=model_id)

    @_guard
    async def gpu_status() -> dict[str, Any]:
        """Read per-GPU VRAM, utilization and temperature.

        Read by the watchdog itself via NVML, NOT by asking the main server --
        so you can still see which GPU is full when the server is dead, which is
        exactly when you need to know whether a leaked llama-server child is
        holding VRAM. If NVML is unavailable it falls back to parsing
        nvidia-smi, and if that fails too it returns a clear error rather than
        pretending there are no GPUs.

        Returns:
            `source` (nvml or nvidia-smi) and one entry per GPU with total /
            free / used VRAM in bytes and MiB, utilization, temperature and
            compute capability.
        """
        return watchdog.gpu_status()

    @_guard
    async def rollback_update(confirm: bool = False, timeout_s: float = 120.0) -> dict[str, Any]:
        """Switch back to the previous installed release and restart.

        DESTRUCTIVE; needs confirm=true. Use when a self-update produced a
        server that will not start or misbehaves: it repoints <data_dir>/current
        at the release directory installed before the current one and then
        restarts.

        On Windows `current` is a POINTER FILE (`current.txt` naming the release
        directory) rather than a symlink, because creating symlinks on Windows
        requires elevation that a recovery tool cannot assume it has. On Linux a
        symlink is used and preferred.

        If there is no previous release this refuses and tells you which
        releases are installed. Config and the model registry live outside the
        release directories, so a rollback never touches them.

        Args:
            confirm: Must be true. Without it you get a preview of which release
                it would roll back from and to.
            timeout_s: How long to wait for the rolled-back server's /health.

        Returns:
            Which release it moved from and to, the pointer it wrote, and the
            nested restart result.
        """
        return await watchdog.rollback_update(confirm=confirm, timeout_s=timeout_s)

    for tool in (
        health,
        get_config,
        set_config,
        restart_server,
        kill_model,
        nuke_all_models,
        reclaim_orphan_engines,
        tail_logs,
        gpu_status,
        rollback_update,
    ):
        server.add_tool(tool)

    return server


def _set_dotted(
    raw: dict[str, Any], updates: dict[str, Any], *, reference: dict[str, Any]
) -> dict[str, Any]:
    """Apply dotted-path updates to a raw mapping, checking keys against a schema.

    Only used on the recovery path, when the file on disk does not validate and
    :func:`studioforge.config.apply_overrides` therefore has no valid base to
    work from. ``reference`` is a default ``Config`` rendered as a dict, so an
    unknown key is still rejected; the merged result is validated by
    ``Config(**merged)`` at the call site, so this function never decides
    whether a *value* is acceptable.
    """
    merged = copy.deepcopy(raw)
    for dotted, value in updates.items():
        parts = dotted.split(".")
        ref: Any = reference
        for part in parts[:-1]:
            if not isinstance(ref, dict) or part not in ref:
                raise ValueError(f"unknown config key: {dotted}")
            ref = ref[part]
        if not isinstance(ref, dict) or parts[-1] not in ref:
            raise ValueError(f"unknown config key: {dotted}")
        cursor = merged
        for part in parts[:-1]:
            nested = cursor.get(part)
            if not isinstance(nested, dict):
                nested = {}
                cursor[part] = nested
            cursor = nested
        cursor[parts[-1]] = value
    return merged


# ---------------------------------------------------------------------------
# ASGI plumbing
# ---------------------------------------------------------------------------


def _json_response_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, default=str).encode("utf-8")


async def _send_json(
    send: Any,
    status: int,
    payload: dict[str, Any],
    *,
    extra_headers: Sequence[tuple[bytes, bytes]] = (),
) -> None:
    body = _json_response_bytes(payload)
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                *extra_headers,
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def wrap_asgi(
    inner: Any,
    watchdog: Watchdog,
    api_key: str | None = None,
    *,
    credentials: Callable[[], tuple[str | None, str | None]] | None = None,
) -> Any:
    """Add a plain ``GET /health`` and optional bearer auth around the MCP app.

    Written as a bare ASGI callable rather than Starlette middleware to keep the
    watchdog's own import surface as small as possible, and because the health
    endpoint is then served by the *outermost* layer: it answers even if the MCP
    session machinery inside is unhappy, which is what a load balancer or the
    ``sfctl`` poller needs. ``/health`` is intentionally unauthenticated, exactly
    like the main app's, so a probe never needs a credential.

    ``credentials`` returns the *current* ``(api_key, mcp_pin)`` pair, re-read
    per request: the watchdog's contract is that configuration is never
    cached, and the credential was the one thing that was -- rotating
    ``server.api_key`` through the app locked the operator out of the recovery
    surface until the watchdog restarted. The static ``api_key`` parameter is
    kept for callers (and tests) that want a fixed key. The MCP pairing PIN is
    accepted as an alternative bearer, mirroring the main app's ``/mcp``.
    """

    async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await inner(scope, receive, send)
            return
        path = scope.get("path", "")
        method = scope.get("method", "GET")
        if path.rstrip("/") in ("/health", "/healthz") and method == "GET":
            result = await watchdog.health()
            status = 200 if result.get("status") in ("up", "degraded") else 503
            await _send_json(send, status, result)
            return
        expected_key, expected_pin = credentials() if credentials else (api_key, None)
        restart_route = path.rstrip("/") == "/restart" and method == "POST"
        # Enforce whenever EITHER credential exists. Gating on the key alone
        # left this surface wide open in the default install, where
        # `server.api_key` is unset and only the MCP pairing PIN is set: the
        # main app still demanded the PIN on /mcp, while the watchdog -- which
        # carries the destructive tools (nuke_all_models, kill_model,
        # restart_server, set_config) and POST /restart -- accepted anyone.
        # That inverted the rule that the control plane is never the
        # least-protected surface. Open only when neither credential is set.
        if expected_key or expected_pin:
            provided = _bearer_from_headers(scope.get("headers") or [])
            pin_carrier = _pin_from_request(scope)
            peer = _peer_key(scope) if (provided or pin_carrier) else None
            # Same lockout as the main app, and for a sharper reason: this is
            # the surface with nuke_all_models, kill_model, restart_server and
            # rollback_update on it, and on the default install the only thing
            # in front of them is an eight-digit PIN. Separate process, so
            # separate counter -- neither door can lock the other.
            wait = _CREDENTIAL_GUARD.retry_after(peer)
            if wait > 0:
                seconds = max(1, math.ceil(wait))
                await _send_json(
                    send,
                    429,
                    {
                        "error": {
                            "message": (
                                "Too many incorrect credentials from this address. Wait "
                                f"{seconds}s and try again."
                            ),
                            "type": "invalid_request_error",
                            "code": "too_many_credential_attempts",
                            "studioforge": {"retry_after_s": seconds},
                        }
                    },
                    extra_headers=((b"retry-after", str(seconds).encode("ascii")),),
                )
                return
            key_ok = (
                expected_key is not None
                and provided is not None
                and _constant_time_eq(provided, expected_key)
            )
            pin_ok = expected_pin is not None and any(
                candidate is not None and _constant_time_eq(candidate, expected_pin)
                for candidate in (provided, pin_carrier)
            )
            if key_ok or pin_ok:
                _CREDENTIAL_GUARD.record_success(peer)
            if not key_ok and not pin_ok:
                if provided or pin_carrier:
                    _CREDENTIAL_GUARD.record_failure(peer)
                query_note = (
                    " The PIN is no longer accepted as '?pin=': a URL is written to access "
                    "logs and shell history, which is no place for a credential."
                    if _pin_in_query(scope)
                    else ""
                )
                await _send_json(
                    send,
                    401,
                    {
                        "error": {
                            "message": (
                                "This endpoint needs a credential. The watchdog "
                                "accepts the same server.api_key as the main server, "
                                "or the MCP pairing PIN when no key is set -- as "
                                "'Authorization: Bearer <credential>', as the "
                                "'X-MCP-Pin' header. The PIN is "
                                "on the control panel (Setup -> Network & access -> "
                                "the eye button next to 'MCP pairing PIN'), in "
                                "`studioforge config` on the host, and at GET "
                                "/api/mcp/info from the host itself. For sfctl: "
                                "`sfctl servers add rig <url> --api-key <PIN> --use`." + query_note
                            ),
                            "type": "invalid_request_error",
                            "code": "invalid_api_key",
                        }
                    },
                )
                return

        # Plain HTTP restart, deliberately *not* routed through MCP. The main
        # app hands off to us when it cannot restart itself, and a JSON-RPC
        # tools/call over streamable-HTTP first needs an initialize handshake
        # and a session id -- machinery that is exactly what tends to be broken
        # when someone reaches for the recovery sidecar. Same auth as the rest
        # of the surface; ``/health`` remains the only open route.
        if restart_route:
            log.warning("POST /restart accepted; restarting the main server")
            result = await watchdog.restart_server(confirm=True)
            log.warning(
                "POST /restart finished: ok=%s new_pid=%s", result.get("ok"), result.get("new_pid")
            )
            # The caller is normally the very process we just killed, so writing
            # the reply can fail on a closed connection. That is the successful
            # path, not an error worth a traceback in the recovery log.
            with contextlib.suppress(Exception):
                await _send_json(send, 200 if result.get("ok") else 500, result)
            return

        await inner(scope, receive, send)

    return app


def _bearer_from_headers(headers: Iterable[tuple[bytes, bytes]]) -> str | None:
    for name, value in headers:
        lowered = name.lower()
        if lowered == b"authorization":
            text = value.decode("latin-1").strip()
            prefix, _, token = text.partition(" ")
            if prefix.lower() == "bearer" and token:
                return token.strip()
            if text and not _:
                return text
        elif lowered == b"x-api-key":
            return value.decode("latin-1").strip()
    return None


#: One lockout counter for the whole watchdog process; see
#: :mod:`studioforge.credential_guard`.
_CREDENTIAL_GUARD = CredentialGuard()

#: Header names the MAIN server accepts the MCP pairing PIN under (see
#: ``studioforge.api.auth.extract_pin``). The watchdog must accept the same
#: ones: a client that paired with the main ``/mcp`` using ``X-MCP-Pin`` got a
#: 401 from the recovery surface with a message that only mentioned Bearer --
#: the one place a confusing credential error is least affordable is the
#: endpoint you reach for when everything else is down (2026-08-19).
_PIN_HEADERS: tuple[bytes, ...] = (b"x-mcp-pin", b"x-studioforge-pin")


def _pin_from_request(scope: dict[str, Any]) -> str | None:
    """The PIN sent as ``X-MCP-Pin``/``X-StudioForge-Pin``. Headers only.

    ``?pin=`` is refused here for the same reason the main app refuses it
    (``studioforge.api.auth.PIN_IN_QUERY_NOTE``): a URL ends up in access logs
    and shell history, and this surface carries the destructive tools.
    """
    for name, value in scope.get("headers") or []:
        if name.lower() in _PIN_HEADERS:
            text = value.decode("latin-1").strip()
            if text:
                return text
    return None


def _pin_in_query(scope: dict[str, Any]) -> bool:
    """Whether the caller tried the retired ``?pin=`` form."""
    query = (scope.get("query_string") or b"").decode("latin-1")
    for part in query.split("&"):
        key, _, val = part.partition("=")
        if key == "pin" and val.strip():
            return True
    return False


def _peer_key(scope: dict[str, Any]) -> str | None:
    client = scope.get("client")
    if not client:
        return None
    host = client[0] if isinstance(client, list | tuple) else client
    return client_key(str(host) if host is not None else None)


def _constant_time_eq(left: str, right: str) -> bool:
    # Bytes, not str: Starlette/ASGI header values are latin-1 decoded, and
    # hmac.compare_digest raises TypeError for non-ASCII str operands -- which
    # would escape as a 500 instead of a 401.
    return hmac.compare_digest(
        left.encode("utf-8", "surrogateescape"),
        right.encode("utf-8", "surrogateescape"),
    )


def create_asgi_app(watchdog: Watchdog, *, path: str = "/mcp") -> tuple[Any, MCPServer]:
    """Build the watchdog's ASGI application (streamable-HTTP MCP + /health)."""
    config, _ = watchdog.load_config()
    server = build_watchdog_mcp(watchdog)
    inner = server.streamable_http_app(
        streamable_http_path=path,
        host=config.watchdog.host,
    )

    def _current_credentials() -> tuple[str | None, str | None]:
        # Re-read per request, like every other piece of watchdog config: a
        # key rotated through the app must bite here immediately, in both
        # directions. Failure degrades to the boot-time snapshot rather than
        # silently disabling auth.
        try:
            current, _source = watchdog.load_config()
        except Exception:  # pragma: no cover - unreadable config mid-request
            return config.server.api_key, None
        mcp_config = getattr(current, "mcp", None)
        pin = getattr(mcp_config, "pin", None) if mcp_config else None
        # `or config.server.api_key`: a malformed config.yaml degrades
        # load_config to defaults (api_key=None), and that must fail towards
        # "auth still enforced with the boot-time key", never towards "auth
        # silently off".
        return current.server.api_key or config.server.api_key, (str(pin) if pin else None)

    return wrap_asgi(inner, watchdog, credentials=_current_credentials), server


async def serve(
    config_path: Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    path: str = "/mcp",
    poll: bool = True,
) -> None:
    """Run the watchdog until cancelled.

    ``uvicorn`` is imported here rather than at module scope: it arrives as a
    dependency of the MCP SDK's own HTTP transport, is used purely as the
    transport, and a lazy import keeps ``import studioforge.watchdog.server``
    fast and dependency-light for the many callers that only want the
    :class:`Watchdog` logic.
    """
    import uvicorn

    watchdog = Watchdog(config_path)
    config, config_error = watchdog.load_config()
    if config_error is not None:
        log.error("config problem (continuing with defaults): %s", config_error)

    asgi, _server = create_asgi_app(watchdog, path=path)
    bind_host = host or config.watchdog.host
    bind_port = port or config.watchdog.port

    log.warning(
        "watchdog listening on http://%s:%s%s (health: http://%s:%s/health)",
        bind_host,
        bind_port,
        path,
        bind_host,
        bind_port,
    )

    poll_task: asyncio.Task[None] | None = None
    if poll:
        poll_task = asyncio.create_task(watchdog.poll_loop(), name="studioforge-watchdog-poll")
    uv = uvicorn.Server(
        uvicorn.Config(
            asgi,
            host=bind_host,
            port=bind_port,
            log_level="warning",
            access_log=False,
        )
    )
    try:
        await uv.serve()
    finally:
        if poll_task is not None:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task
