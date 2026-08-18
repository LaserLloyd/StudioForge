"""Who is holding the VRAM, who launched them, and which of them may be killed.

Three separate blind spots met in the incident of 2026-08-18 (DECISIONS.md
D23): ~25 GiB was held across two GPUs with "everything stopped", the holders
were three ``llama-server.exe`` processes launched by a ``pytest`` run, and
``/api/status`` could say nothing about them beyond ``llama-server.exe`` with
``used_bytes: 0`` and ``is_ours: false``. Every part of this module exists to
turn that answer into a useful one:

* :func:`find_engine_processes` -- every ``llama-server`` running out of *our*
  engines directory, with its parent named and its parentage classified.
* :func:`reclaim_orphans` -- kills the ones whose parent is dead, and only
  those. A binary under our own engines directory whose parent no longer
  exists can only be a child we lost, so killing it is safe by construction.
  A child of a *live* process is reported and never touched: during the
  incident those three children belonged to a pytest run that was legitimately
  using them, and killing them would have been the second bug.
* :func:`pdh_process_dedicated_bytes` -- per-process VRAM on Windows, which
  NVML cannot provide under WDDM (it returns 0 for every process). This is the
  same counter Task Manager's "Dedicated GPU memory" column reads.
* :func:`holders_view` / :func:`annotate_status_payload` -- the merged answer,
  with desktop noise collapsed into a count so what is left is the handful of
  processes that actually matter.

Nothing here may raise into a caller: this is diagnostics, and diagnostics that
can break a load or a status poll are worse than no diagnostics at all.
"""

from __future__ import annotations

import contextlib
import os
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from studioforge.core.supervisor import kill_process_tree, process_is_alive
from studioforge.logging import get_logger

if TYPE_CHECKING:
    from studioforge.core.gpu import GpuProbe

log = get_logger(__name__)

#: Executable basenames that are an inference engine of ours.
ENGINE_EXE_NAMES = frozenset({"llama-server", "llama-server.exe"})

#: A holder below this is desktop noise unless it is ours or interesting by
#: name. 256 MiB is comfortably above a compositor/browser tab and far below
#: anything doing real work on a GPU.
HOLDER_MIN_BYTES = 256 * 1024 * 1024

#: Process stems that are always worth listing however little they hold: these
#: are the things that take VRAM *from* us on this class of box.
INTERESTING_STEMS = frozenset(
    {
        "llama-server",
        "llama-cli",
        "llama-bench",
        "python",
        "pythonw",
        "comfyui",
        "ollama",
        "studioforge",
    }
)

#: Parent command lines are truncated: a llama-server argv is ~40 tokens of
#: absolute paths, and this field exists to identify the parent, not to
#: reproduce it.
PARENT_CMDLINE_CHARS = 200

CLASS_OURS = "ours"
CLASS_CHILD_OF_LIVE = "child-of-live-process"
CLASS_ORPHAN = "orphan"
#: A VRAM holder that is not one of our engine binaries at all (a browser, a
#: compositor, ComfyUI). Reported, never killed.
CLASS_FOREIGN = "foreign"


# ---------------------------------------------------------------------------
# Engine processes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EngineProcess:
    """A ``llama-server`` running out of our engines directory."""

    pid: int
    exe: str
    create_time: float | None = None
    cmdline: list[str] = field(default_factory=list)
    alias: str | None = None
    port: int | None = None
    parent_pid: int | None = None
    parent_alive: bool = False
    parent_name: str | None = None
    parent_cmdline: str | None = None
    #: True when the parent pid now belongs to a *different* process than the
    #: one that spawned this child (see :func:`_parent_state`).
    parent_recycled: bool = False
    is_ours: bool = False
    classification: str = CLASS_ORPHAN

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "exe": self.exe,
            "alias": self.alias,
            "port": self.port,
            "uptime_s": (
                round(time.time() - self.create_time, 1) if self.create_time else None
            ),
            "parent_pid": self.parent_pid,
            "parent_alive": self.parent_alive,
            "parent_name": self.parent_name,
            "parent_cmdline": self.parent_cmdline,
            "parent_recycled": self.parent_recycled,
            "is_ours": self.is_ours,
            "classification": self.classification,
        }


def _normcase(path: str | Path) -> str:
    """Comparable form of a path.

    ``normcase`` and not just ``resolve``: Windows paths compare
    case-insensitively, and an engines dir spelled ``C:\\...`` would otherwise
    fail to contain a child whose exe psutil reports as ``c:\\...`` -- which
    would silently turn every one of our own processes into "not ours".
    """
    try:
        return os.path.normcase(os.path.normpath(str(Path(path).resolve())))
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        return os.path.normcase(os.path.normpath(str(path)))


def _is_under(candidate: str | None, root: str) -> bool:
    if not candidate:
        return False
    normalised = _normcase(candidate)
    return normalised == root or normalised.startswith(root.rstrip(os.sep) + os.sep)


def flag_value(cmdline: Sequence[str], flag: str) -> str | None:
    """Value following ``flag`` in an argv list, or ``None``.

    Handles ``--port 8080`` and ``--port=8080``, last occurrence winning, which
    is how llama.cpp itself resolves a repeated flag.
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


@dataclass(slots=True)
class _Parent:
    alive: bool
    name: str | None = None
    cmdline: str | None = None
    recycled: bool = False


def _parent_state(pid: int | None, child_create_time: float | None) -> _Parent:
    """Whether ``pid`` is still the process that spawned a child born then.

    The pid-reuse guard is the whole point and mirrors
    :func:`studioforge.core.supervisor.process_is_alive`: on a busy box the OS
    hands a dead parent's pid to something new, and a naive "the parent pid
    still exists" check would then classify a genuine orphan as somebody's live
    child -- i.e. exactly the leak this module exists to find would be filed as
    "not ours to kill". A process that started *after* its supposed child did
    not spawn it.
    """
    if pid is None or pid <= 0:
        return _Parent(alive=False)
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            created = float(proc.create_time())
            name = str(proc.name())
            status = proc.status()
            try:
                cmdline = " ".join(proc.cmdline())
            except (psutil.Error, ValueError):
                cmdline = ""
    except (psutil.Error, ValueError):
        return _Parent(alive=False)

    truncated = (cmdline[:PARENT_CMDLINE_CHARS] or None) if cmdline else None
    if child_create_time is not None and created > child_create_time + 1.0:
        # Somebody else is wearing the parent's number. Report who, so the
        # log line reads "parent pid 4242 is now explorer.exe" rather than
        # leaving an operator to wonder.
        return _Parent(alive=False, name=name, cmdline=truncated, recycled=True)
    if status == psutil.STATUS_ZOMBIE:
        return _Parent(alive=False, name=name, cmdline=truncated)
    return _Parent(alive=True, name=name, cmdline=truncated)


def find_engine_processes(
    engines_dir: str | Path, *, own_pids: Iterable[int] = ()
) -> list[EngineProcess]:
    """Every ``llama-server`` process running from under ``engines_dir``.

    Two conditions, both required: the executable is named ``llama-server``
    *and* it lives under our own engines directory. The second is what makes
    :func:`reclaim_orphans` safe -- a llama-server started by LM Studio, by a
    second StudioForge install, or by hand is not under our engines dir and is
    therefore never a kill candidate, only a reported holder.

    Access-denied processes are skipped silently: on Windows a good fraction of
    the process table belongs to other sessions and to SYSTEM, and none of it
    is ours.
    """
    root = _normcase(engines_dir)
    ours = {int(pid) for pid in own_pids}
    found: list[EngineProcess] = []
    try:
        iterator = psutil.process_iter(["name"])
    except Exception:  # noqa: BLE001 - pragma: no cover - psutil unavailable
        return []
    for proc in iterator:
        try:
            name = str(proc.info.get("name") or "")
        except (psutil.Error, AttributeError):  # pragma: no cover - vanished
            continue
        # Cheap prefilter: fetching exe/cmdline for every process on the box is
        # slow and mostly access-denied, and this endpoint is polled.
        if "llama-server" not in name.lower():
            continue
        try:
            with proc.oneshot():
                exe = proc.exe()
                cmdline = list(proc.cmdline() or [])
                create_time = float(proc.create_time())
                parent_pid = int(proc.ppid())
        except (psutil.Error, ValueError, OSError):
            continue
        if Path(exe).name.lower() not in ENGINE_EXE_NAMES or not _is_under(exe, root):
            continue

        parent = _parent_state(parent_pid, create_time)
        pid = int(proc.pid)
        if pid in ours:
            classification = CLASS_OURS
        elif parent.alive:
            classification = CLASS_CHILD_OF_LIVE
        else:
            classification = CLASS_ORPHAN
        found.append(
            EngineProcess(
                pid=pid,
                exe=exe,
                create_time=create_time,
                cmdline=cmdline,
                alias=flag_value(cmdline, "--alias"),
                port=_int_or_none(flag_value(cmdline, "--port")),
                parent_pid=parent_pid,
                parent_alive=parent.alive,
                parent_name=parent.name,
                parent_cmdline=parent.cmdline,
                parent_recycled=parent.recycled,
                is_ours=pid in ours,
                classification=classification,
            )
        )
    found.sort(key=lambda entry: entry.pid)
    return found


def reclaim_orphans(
    engines_dir: str | Path, *, own_pids: Iterable[int] = (), dry_run: bool = False
) -> list[dict[str, Any]]:
    """Kill every orphaned engine process, and nothing else.

    "Orphan" means: our binary, our engines directory, and a parent that is
    gone (or whose pid has been recycled). Such a process cannot belong to
    anyone else -- nobody else launches binaries out of our engines tree -- and
    nothing alive is waiting on it, so its VRAM is pure leak.

    What is deliberately **not** killed: a ``child-of-live-process``. During the
    incident that was a pytest run mid-suite, and a sweep that killed it would
    have destroyed work in progress to recover memory nobody was short of yet.
    Those are reported by :func:`holders_view` instead, with the parent named so
    the operator can decide.

    Returns one record per orphan (empty list when there are none), so the
    caller can log, display or assert on exactly what happened.
    """
    orphans = [
        entry
        for entry in find_engine_processes(engines_dir, own_pids=own_pids)
        if entry.classification == CLASS_ORPHAN
    ]
    actions: list[dict[str, Any]] = []
    for entry in orphans:
        action: dict[str, Any] = {
            "pid": entry.pid,
            "alias": entry.alias,
            "port": entry.port,
            "exe": entry.exe,
            "parent_pid": entry.parent_pid,
            "parent_recycled": entry.parent_recycled,
            "dry_run": dry_run,
            "killed": False,
        }
        if not dry_run:
            try:
                kill_process_tree(entry.pid, timeout=5.0, force=True)
            except Exception as exc:  # noqa: BLE001 - one failure must not stop the sweep
                action["error"] = str(exc)
            action["killed"] = not process_is_alive(entry.pid, create_time=entry.create_time)
        actions.append(action)
    if actions:
        log.warning(
            "vram_orphans_found",
            count=len(actions),
            dry_run=dry_run,
            pids=[a["pid"] for a in actions],
            aliases=[a["alias"] for a in actions],
            killed=sum(1 for a in actions if a["killed"]),
            detail=(
                "llama-server processes under our engines dir whose parent is gone; "
                "their VRAM was leaked"
            ),
        )
    return actions


# ---------------------------------------------------------------------------
# Per-process VRAM on Windows (PDH)
# ---------------------------------------------------------------------------

#: PDH counter set and counter. ``Dedicated Usage`` is the number Task Manager
#: shows in Details -> "Dedicated GPU memory".
PDH_OBJECT = "GPU Process Memory"
PDH_COUNTER = "Dedicated Usage"

#: Cache lifetime. ``/api/status`` and the Dashboard poll continuously and a
#: PDH sample costs a few milliseconds plus, on a cold sample, a 100 ms settle.
PDH_CACHE_TTL_S = 2.0

_pdh_lock = threading.Lock()
_pdh_cache: tuple[float, dict[int, int]] = (0.0, {})
_pdh_disabled = False


def _load_win32pdh() -> Any:
    """Import the PDH bindings (patchable by tests)."""
    import win32pdh

    return win32pdh


def _pid_from_instance(name: str) -> int | None:
    """Pid out of a PDH instance name.

    Instances look like ``pid_19544_luid_0x00000000_0x0000E5F5_phys_0``: the
    pid, the adapter LUID, and the physical adapter index within it. Only the
    pid is usable here -- NVML exposes no LUID, so there is no way to map an
    instance back to a CUDA ordinal (see :func:`pdh_process_dedicated_bytes`).
    """
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "pid":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _formatted_array(win32pdh: Any, counter: Any) -> list[tuple[str, Any]]:
    """Instance/value pairs from a wildcard counter, across binding versions.

    Current pywin32 returns a ``dict``; older builds returned ``(count, list)``
    or a bare list. An invalid sample (the counter set exists but has not been
    populated yet) raises, and that is a normal transient, not an error.
    """
    try:
        data = win32pdh.GetFormattedCounterArray(counter, win32pdh.PDH_FMT_LARGE)
    except Exception:  # noqa: BLE001 - PDH_INVALID_DATA on a cold query
        return []
    if isinstance(data, Mapping):
        return list(data.items())
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], list):
        return list(data[1])
    if isinstance(data, list):
        return list(data)
    return []


def _pdh_sample() -> dict[int, int]:
    win32pdh = _load_win32pdh()
    query = win32pdh.OpenQuery()
    try:
        path = win32pdh.MakeCounterPath((None, PDH_OBJECT, "*", None, -1, PDH_COUNTER))
        counter = win32pdh.AddCounter(query, path)
        win32pdh.CollectQueryData(query)
        pairs = _formatted_array(win32pdh, counter)
        if not pairs:
            # A wildcard counter needs a populated sample; one collect is
            # usually enough, so the settle only happens when it was not.
            time.sleep(0.1)
            win32pdh.CollectQueryData(query)
            pairs = _formatted_array(win32pdh, counter)
    finally:
        with contextlib.suppress(Exception):
            win32pdh.CloseQuery(query)

    per_pid: dict[int, int] = {}
    for instance, value in pairs:
        pid = _pid_from_instance(str(instance))
        if pid is None:
            continue
        try:
            used = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if used <= 0:
            continue
        # Summed across adapters: one process can hold memory on several GPUs.
        per_pid[pid] = per_pid.get(pid, 0) + used
    return per_pid


def pdh_process_dedicated_bytes(*, ttl_s: float = PDH_CACHE_TTL_S) -> dict[int, int]:
    """Dedicated GPU memory per pid, in bytes. ``{}`` when unavailable.

    **Why this exists.** NVML's per-process memory is 0 for every process on
    Windows under the WDDM driver model -- video memory is owned by the OS
    there, not by the driver, so ``nvmlDeviceGetComputeRunningProcesses``
    enumerates the holders but cannot size them. That is what made
    ``/api/status`` useless during the incident: it listed
    ``llama-server.exe ... used_bytes: 0`` three times while 25 GiB was gone.
    The Windows performance-counter subsystem *does* know, and this is the
    counter Task Manager reads.

    **What the number means.** A per-process total across every adapter. The
    PDH instance name carries an adapter LUID, but NVML exposes no LUID, so
    there is no sound mapping from an instance to a CUDA ordinal; splitting the
    total per GPU would be invention. Callers that show per-GPU rows must
    therefore not sum this across rows -- see :func:`holders_view`, which
    aggregates per pid for exactly this reason.

    Never raises, returns ``{}`` on any failure, and disables itself after the
    first failure (a missing counter set does not come back) having logged one
    warning. Non-Windows always returns ``{}``.
    """
    global _pdh_cache, _pdh_disabled
    if os.name != "nt":
        return {}
    now = time.monotonic()
    with _pdh_lock:
        sampled_at, cached = _pdh_cache
        if cached and now - sampled_at < ttl_s:
            return dict(cached)
        if _pdh_disabled:
            return {}
        try:
            fresh = _pdh_sample()
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            _pdh_disabled = True
            log.warning(
                "pdh_unavailable",
                error=str(exc),
                detail=(
                    f"cannot read '{PDH_OBJECT}' / '{PDH_COUNTER}'; per-process VRAM "
                    "will be reported as unknown"
                ),
            )
            return {}
        _pdh_cache = (now, fresh)
        return dict(fresh)


def reset_pdh_cache() -> None:
    """Drop the PDH cache and re-enable sampling (tests)."""
    global _pdh_cache, _pdh_disabled
    with _pdh_lock:
        _pdh_cache = (0.0, {})
        _pdh_disabled = False


# ---------------------------------------------------------------------------
# The merged view
# ---------------------------------------------------------------------------


def _stem(name: str) -> str:
    return Path(name or "").stem.lower()


def holder_matters(row: Mapping[str, Any]) -> bool:
    """Whether a holder belongs in the list a human or an LLM reads.

    The unfiltered list on this box is ~20 rows of ``dwm.exe``,
    ``explorer.exe``, ``SearchHost.exe`` and browser tabs -- which is how three
    leaked ``llama-server.exe`` entries went unnoticed. Kept: anything of ours,
    any engine binary, anything named like a GPU workload, and anything holding
    real memory. Everything else is counted, not listed, so nothing is hidden
    -- only summarised.
    """
    if row.get("is_ours") or row.get("engine_process"):
        return True
    if _stem(str(row.get("name") or "")) in INTERESTING_STEMS:
        return True
    try:
        return int(row.get("used_bytes") or 0) > HOLDER_MIN_BYTES
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _describe_process(pid: int) -> dict[str, Any]:
    """Name and parentage of an arbitrary holder, best effort."""
    out: dict[str, Any] = {
        "parent_pid": None,
        "parent_name": None,
        "parent_cmdline": None,
        "parent_alive": False,
        "parent_recycled": False,
    }
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            created = float(proc.create_time())
            parent_pid = int(proc.ppid())
    except (psutil.Error, ValueError):
        return out
    parent = _parent_state(parent_pid, created)
    out.update(
        parent_pid=parent_pid,
        parent_name=parent.name,
        parent_cmdline=parent.cmdline,
        parent_alive=parent.alive,
        parent_recycled=parent.recycled,
    )
    return out


def holders_view(
    probe: GpuProbe | None,
    engines_dir: str | Path,
    *,
    own_pids: Iterable[int] = (),
) -> dict[str, Any]:
    """Everything holding VRAM, aggregated per process and classified.

    Merges two sources because neither alone is enough: NVML/the probe knows
    which *GPU* a process is on but (on Windows) not how much it holds, and the
    process table knows which processes are our engine binaries and who spawned
    them but nothing about GPUs. An engine process NVML does not list at all
    still appears here -- a holder we cannot see is the failure mode this whole
    module is about.

    One row per pid, never per (gpu, pid): the byte figure on Windows is a
    per-process total across adapters, so per-GPU rows would double count.

    The probe's *raw* holders are read here rather than
    :func:`studioforge.core.gpu.vram_processes`, precisely because that function
    back-fills zeros from PDH. Once back-filled, a pid on two GPUs carries the
    same per-process total on both of its rows and there is no longer any way
    to tell a real NVML pair from one PDH figure counted twice. Reading raw
    keeps summing safe and keeps ``used_bytes_source`` honest.
    """
    ours = {int(pid) for pid in own_pids}
    engines = {entry.pid: entry for entry in find_engine_processes(engines_dir, own_pids=ours)}

    entries: list[Any] = []
    getter = getattr(probe, "compute_processes", None) if probe is not None else None
    if getter is not None:
        try:
            entries = list(getter())
        except Exception as exc:  # noqa: BLE001 - a probe gap is not an error
            log.debug("vram_holders_probe_failed", error=str(exc))
    per_pid_bytes = pdh_process_dedicated_bytes()

    rows: dict[int, dict[str, Any]] = {}
    for entry in entries:
        row = rows.setdefault(
            entry.pid,
            {
                "pid": entry.pid,
                "name": entry.name,
                "gpu_indices": [],
                "nvml_bytes": 0,
                "is_ours": entry.pid in ours,
            },
        )
        if entry.gpu_index not in row["gpu_indices"]:
            row["gpu_indices"].append(entry.gpu_index)
        row["nvml_bytes"] += int(entry.used_bytes or 0)

    for pid, found in engines.items():
        row = rows.setdefault(
            pid,
            {"pid": pid, "name": Path(found.exe).name, "gpu_indices": [], "nvml_bytes": 0},
        )
        row["is_ours"] = bool(row.get("is_ours") or found.is_ours)

    holders: list[dict[str, Any]] = []
    desktop_count = 0
    desktop_bytes = 0
    sources: set[str] = set()
    for pid, row in rows.items():
        engine = engines.get(pid)
        pdh_bytes = per_pid_bytes.get(pid, 0)
        nvml_bytes = int(row.get("nvml_bytes") or 0)
        if nvml_bytes > 0:
            used, source = nvml_bytes, "nvml"
        elif pdh_bytes > 0:
            used, source = pdh_bytes, "pdh"
        else:
            used, source = 0, "unknown"
        sources.add(source)
        merged: dict[str, Any] = {
            "pid": pid,
            "name": row["name"],
            "used_bytes": used,
            "used_bytes_source": source,
            "gpu_indices": sorted(row["gpu_indices"]),
            "is_ours": bool(row.get("is_ours")),
            "engine_process": engine is not None,
            "alias": engine.alias if engine else None,
            "port": engine.port if engine else None,
            "classification": engine.classification if engine else CLASS_FOREIGN,
        }
        if not holder_matters(merged):
            desktop_count += 1
            desktop_bytes += used
            continue
        if engine is not None:
            merged.update(
                parent_pid=engine.parent_pid,
                parent_name=engine.parent_name,
                parent_cmdline=engine.parent_cmdline,
                parent_alive=engine.parent_alive,
                parent_recycled=engine.parent_recycled,
            )
        else:
            merged.update(_describe_process(pid))
        holders.append(merged)

    holders.sort(key=lambda row: (-int(row["used_bytes"]), row["pid"]))
    orphans = [row for row in holders if row["classification"] == CLASS_ORPHAN]
    return {
        "holders": holders,
        "orphan_count": len(orphans),
        "orphan_pids": [row["pid"] for row in orphans],
        "desktop_processes_count": desktop_count,
        "desktop_processes_bytes": desktop_bytes,
        "per_process_bytes": _bytes_source(sources),
        "engines_dir": str(engines_dir),
        "engine_process_count": len(engines),
    }


def _bytes_source(sources: Iterable[str]) -> str:
    """Where the byte figures came from, so a zero can be read correctly.

    Computed over *every* holder, not only the listed ones: on an idle box the
    listed holders can be empty while the collapsed desktop rows are all
    perfectly well measured, and reporting "unavailable" there would tell the
    user their box cannot do something it just did.
    """
    seen = set(sources)
    for candidate in ("nvml", "pdh"):
        if candidate in seen:
            return candidate
    return "unavailable"


def annotate_status_payload(
    payload: dict[str, Any],
    *,
    engines_dir: str | Path,
    own_pids: Iterable[int] = (),
) -> dict[str, Any]:
    """Make ``/api/status.vram_processes`` answer "who has my VRAM".

    Mutates ``payload`` in place and returns it. Existing keys on each entry
    are preserved -- clients (and the planner's own rejection messages) already
    read ``gpu_index``/``pid``/``name``/``used_bytes``/``is_ours`` -- and four
    are added for engine processes: ``parent_pid``, ``parent_name``,
    ``parent_cmdline`` and ``classification``.

    Desktop noise is collapsed into ``desktop_processes_count`` rather than
    listed. The old payload was ~20 entries of compositor and browser
    processes, all reporting zero bytes, which is precisely how three leaked
    engine children hid in plain sight.
    """
    entries = payload.get("vram_processes")
    if not isinstance(entries, list):
        return payload
    ours = {int(pid) for pid in own_pids}
    engines = {entry.pid: entry for entry in find_engine_processes(engines_dir, own_pids=ours)}
    per_pid_bytes = pdh_process_dedicated_bytes()

    kept: list[dict[str, Any]] = []
    # Collapsed rows are accounted per *pid*, not per row: NVML lists one entry
    # per (gpu, pid), and a PDH figure is already a per-process total, so
    # counting rows would report a process on two GPUs twice -- in both the
    # count and the bytes.
    desktop_by_pid: dict[int, int] = {}
    seen: set[int] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pid = int(entry.get("pid") or 0)
        seen.add(pid)
        engine = engines.get(pid)
        used = int(entry.get("used_bytes") or 0)
        if used <= 0 and per_pid_bytes.get(pid):
            entry["used_bytes"] = used = per_pid_bytes[pid]
        # Provenance by equality: these entries have usually been through
        # gpu._fill_missing_used_bytes already, so "did NVML measure this?"
        # cannot be answered by "is it non-zero?". It matters because a PDH
        # number is a per-process total across adapters -- a client that sums
        # the per-GPU rows of one pid would double count it.
        if used > 0:
            entry["used_bytes_source"] = "pdh" if per_pid_bytes.get(pid) == used else "nvml"
        else:
            entry["used_bytes_source"] = "unknown"
        entry["classification"] = engine.classification if engine else CLASS_FOREIGN
        entry["engine_process"] = engine is not None
        if not holder_matters(entry):
            if entry["used_bytes_source"] == "pdh":
                desktop_by_pid[pid] = used
            else:
                desktop_by_pid[pid] = desktop_by_pid.get(pid, 0) + used
            continue
        if engine is not None:
            entry["parent_pid"] = engine.parent_pid
            entry["parent_name"] = engine.parent_name
            entry["parent_cmdline"] = engine.parent_cmdline
            entry["parent_alive"] = engine.parent_alive
            entry["alias"] = engine.alias
        kept.append(entry)

    # An engine process the probe never reported still belongs in the list: a
    # holder nothing can see is the exact failure this endpoint now exists to
    # prevent.
    for pid, engine in engines.items():
        if pid in seen:
            continue
        kept.append(
            {
                "gpu_index": -1,
                "pid": pid,
                "name": Path(engine.exe).name,
                "used_bytes": per_pid_bytes.get(pid, 0),
                "used_bytes_source": "pdh" if per_pid_bytes.get(pid) else "unknown",
                "is_ours": engine.is_ours,
                "engine_process": True,
                "classification": engine.classification,
                "parent_pid": engine.parent_pid,
                "parent_name": engine.parent_name,
                "parent_cmdline": engine.parent_cmdline,
                "parent_alive": engine.parent_alive,
                "alias": engine.alias,
            }
        )

    payload["vram_processes"] = kept
    payload["desktop_processes_count"] = len(desktop_by_pid)
    payload["desktop_processes_bytes"] = sum(desktop_by_pid.values())
    payload["vram_orphan_count"] = sum(
        1 for entry in engines.values() if entry.classification == CLASS_ORPHAN
    )
    return payload
