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
* :func:`luid_to_cuda_index` / :func:`pdh_process_gpu_bytes` -- *where* that
  memory is. The PDH instance name carries an adapter LUID; Windows maps a LUID
  to a PCI address and NVML maps a PCI bus to a CUDA ordinal, so the two ends
  join (DECISIONS.md D39). Without it the device column could only show the
  GPUs a process has a CUDA *context* on, which for llama.cpp is every visible
  device whether or not a byte of the model landed there.
* :func:`holders_view` / :func:`annotate_status_payload` -- the merged answer,
  with desktop noise collapsed into a count so what is left is the handful of
  processes that actually matter.

Nothing here may raise into a caller: this is diagnostics, and diagnostics that
can break a load or a status poll are worse than no diagnostics at all.
"""

from __future__ import annotations

import contextlib
import ctypes
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
#: A ``llama-server`` that is *not* running out of our engines directory: a
#: second StudioForge install, a scratch copy of the binary, a hand-launched
#: llama.cpp. Never killed (it is not ours to kill), but it is not anonymous
#: either -- its ``--alias``/``--port`` say exactly which install it belongs to,
#: which is the difference between "foreign" and an actionable answer.
CLASS_OTHER_INSTANCE = "other-instance"


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
            "uptime_s": (round(time.time() - self.create_time, 1) if self.create_time else None),
            "parent_pid": self.parent_pid,
            "parent_alive": self.parent_alive,
            "parent_name": self.parent_name,
            "parent_cmdline": self.parent_cmdline,
            "parent_recycled": self.parent_recycled,
            "is_ours": self.is_ours,
            "classification": self.classification,
        }


@dataclass(slots=True)
class OtherLlamaServer:
    """A ``llama-server`` belonging to somebody else's install.

    Identity, not ownership: this process is never a kill candidate (see
    :func:`reclaim_orphans`), but "foreign" is the wrong word for it and hides
    the one fact that resolves the situation -- *which* install it is. A
    llama.cpp argv always carries ``--alias`` and ``--port``, and those two plus
    the directory the binary was launched from name it unambiguously.
    """

    pid: int
    exe: str
    alias: str | None = None
    port: int | None = None

    @property
    def detail(self) -> str:
        bits = []
        if self.alias:
            bits.append(f"alias {self.alias}")
        if self.port:
            bits.append(f"port {self.port}")
        with contextlib.suppress(ValueError, OSError):
            bits.append(f"exe {Path(self.exe).parent}")
        joined = ", ".join(bits)
        return "llama-server from another install" + (f" ({joined})" if joined else "")


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


def scan_llama_servers(
    engines_dir: str | Path, *, own_pids: Iterable[int] = ()
) -> tuple[list[EngineProcess], dict[int, OtherLlamaServer]]:
    """Every ``llama-server`` on the box, split into ours and everyone else's.

    One walk of the process table for both answers, because both callers want
    both and the walk is the expensive part of a polled endpoint.

    *Ours* needs two conditions, both required: the executable is named
    ``llama-server`` **and** it lives under our own engines directory. The
    second is what makes :func:`reclaim_orphans` safe -- a llama-server started
    by LM Studio, by a second StudioForge install, or by hand is not under our
    engines dir and is therefore never a kill candidate.

    *Everyone else's* is the same binary from anywhere else. It used to be
    reported as an anonymous ``foreign`` holder, which is true and useless: on
    2026-08-19 an 18.95 GiB holder on this box was a llama-server child of a
    scratch StudioForge (``--alias scratch --port 1258``, exe copied to a temp
    directory), and its own argv said so all along.

    Access-denied processes are skipped silently: on Windows a good fraction of
    the process table belongs to other sessions and to SYSTEM, and none of it
    is ours.
    """
    root = _normcase(engines_dir)
    ours = {int(pid) for pid in own_pids}
    found: list[EngineProcess] = []
    others: dict[int, OtherLlamaServer] = {}
    try:
        iterator = psutil.process_iter(["name"])
    except Exception:  # noqa: BLE001 - pragma: no cover - psutil unavailable
        return [], {}
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
        if Path(exe).name.lower() not in ENGINE_EXE_NAMES:
            continue
        pid = int(proc.pid)
        if not _is_under(exe, root):
            others[pid] = OtherLlamaServer(
                pid=pid,
                exe=exe,
                alias=flag_value(cmdline, "--alias"),
                port=_int_or_none(flag_value(cmdline, "--port")),
            )
            continue

        parent = _parent_state(parent_pid, create_time)
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
    return found, others


def find_engine_processes(
    engines_dir: str | Path, *, own_pids: Iterable[int] = ()
) -> list[EngineProcess]:
    """Every ``llama-server`` process running from under ``engines_dir``.

    The kill-candidate half of :func:`scan_llama_servers`; kept as its own name
    because that is the contract :func:`reclaim_orphans` and the startup sweep
    are written against -- they must never see a binary from another install.
    """
    return scan_llama_servers(engines_dir, own_pids=own_pids)[0]


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
            # Between classification and this line the pid could have been
            # recycled to an unrelated process; kill_process_tree resolves by
            # pid alone, so re-check identity first (the same create_time
            # guard the aliveness check below uses).
            if not process_is_alive(entry.pid, create_time=entry.create_time):
                action["killed"] = True
                action["note"] = "already gone before the sweep reached it"
                actions.append(action)
                continue
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

#: An adapter LUID as the PDH instance name spells it: ``(HighPart, LowPart)``,
#: in that order, because that is the order they appear in
#: ``pid_19544_luid_0x00000000_0x0000E5F5_phys_0``. The D3DKMT ``LUID`` struct
#: declares them the other way round, which is exactly the sort of detail worth
#: pinning to one name.
LuidKey = tuple[int, int]

#: Stand-in for an instance whose LUID could not be parsed. It can never equal
#: a real LUID (they are unsigned), so it maps to "some other adapter" (-1) and
#: still contributes its bytes to the per-process total.
UNKNOWN_LUID: LuidKey = (-1, -1)

#: Device key for bytes held on an adapter that is not a CUDA device we can see
#: (the Microsoft Basic Render adapter, an iGPU, a card excluded by
#: ``CUDA_VISIBLE_DEVICES``). Reported rather than dropped: the bytes are real
#: and dropping them would make the per-device split fail to add up.
OTHER_ADAPTER = -1

_pdh_lock = threading.Lock()
_pdh_cache: tuple[float, dict[int, dict[LuidKey, int]]] = (0.0, {})
_pdh_disabled = False


def _load_win32pdh() -> Any:
    """Import the PDH bindings (patchable by tests)."""
    import win32pdh

    return win32pdh


def _pid_from_instance(name: str) -> int | None:
    """Pid out of a PDH instance name.

    Instances look like ``pid_19544_luid_0x00000000_0x0000E5F5_phys_0``: the
    pid, the adapter LUID, and the physical adapter index within it. See
    :func:`_luid_from_instance` for the other half.
    """
    parts = name.split("_")
    if len(parts) < 2 or parts[0] != "pid":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _luid_from_instance(name: str) -> LuidKey:
    """Adapter LUID out of a PDH instance name, or :data:`UNKNOWN_LUID`.

    ``pid_19544_luid_0x00000000_0x0000E5F5_phys_0`` -> ``(0x0, 0xE5F5)``. The
    two words are hex with an ``0x`` prefix; the trailing ``phys_<n>`` is the
    physical adapter *within* that LUID (linked-display-adapter/SLI), which is
    always 0 here and is deliberately not part of the key -- a linked pair is
    one CUDA device.
    """
    parts = name.split("_")
    try:
        marker = parts.index("luid")
    except ValueError:
        return UNKNOWN_LUID
    if marker + 2 >= len(parts):
        return UNKNOWN_LUID
    try:
        return (int(parts[marker + 1], 16), int(parts[marker + 2], 16))
    except ValueError:
        return UNKNOWN_LUID


# ---------------------------------------------------------------------------
# LUID -> PCI bus -> CUDA ordinal (DECISIONS.md D39)
# ---------------------------------------------------------------------------

#: ``KMTQAITYPE_ADAPTERADDRESS``: ask a D3DKMT adapter handle for its PCI
#: bus/device/function.
_KMTQAITYPE_ADAPTERADDRESS = 6

#: How long a resolved LUID map is trusted. Rebuilt rather than persisted:
#: ``CUDA_VISIBLE_DEVICES``, a driver reset or a hot-plugged eGPU all renumber
#: CUDA ordinals, and a stale ordinal is worse than no ordinal -- it attributes
#: one card's memory to another. 30 s because the map costs a handful of
#: syscalls and the topology of a workstation does not change faster than that.
LUID_MAP_TTL_S = 30.0

_luid_lock = threading.Lock()
_luid_cache: tuple[float, dict[LuidKey, int]] = (0.0, {})
_luid_seen: set[LuidKey] = set()


class _LUID(ctypes.Structure):
    #: LowPart first, HighPart second -- the Win32 declaration order, and the
    #: reverse of how the PDH instance name prints them.
    _fields_ = (("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32))


class _D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = (("AdapterLuid", _LUID), ("hAdapter", ctypes.c_uint32))


class _D3DKMT_ADAPTERADDRESS(ctypes.Structure):
    _fields_ = (
        ("BusNumber", ctypes.c_uint32),
        ("DeviceNumber", ctypes.c_uint32),
        ("FunctionNumber", ctypes.c_uint32),
    )


class _D3DKMT_QUERYADAPTERINFO(ctypes.Structure):
    _fields_ = (
        ("hAdapter", ctypes.c_uint32),
        ("Type", ctypes.c_uint32),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", ctypes.c_uint32),
    )


class _D3DKMT_CLOSEADAPTER(ctypes.Structure):
    _fields_ = (("hAdapter", ctypes.c_uint32),)


def _load_gdi32() -> Any:
    """The kernel-mode-thunk entry points, out of gdi32 (patchable by tests)."""
    return ctypes.WinDLL("gdi32")


def _luid_pci_buses(luids: Iterable[LuidKey]) -> dict[LuidKey, int]:
    """PCI bus number for each adapter LUID, best effort.

    ``D3DKMTOpenAdapterFromLuid`` -> ``D3DKMTQueryAdapterInfo(ADAPTERADDRESS)``
    -> ``D3DKMTCloseAdapter``. A LUID that fails any of the three, or that opens
    and has no real PCI address (the Microsoft Basic Render adapter answers
    ``0xFFFFFFFF`` on this box), is simply absent from the result; the caller
    treats absence as "some other adapter" rather than guessing.
    """
    wanted = [key for key in luids if key != UNKNOWN_LUID]
    if not wanted:
        return {}
    gdi32 = _load_gdi32()
    out: dict[LuidKey, int] = {}
    for high, low in wanted:
        opened = _D3DKMT_OPENADAPTERFROMLUID()
        opened.AdapterLuid.LowPart = low & 0xFFFFFFFF
        opened.AdapterLuid.HighPart = ctypes.c_int32(high & 0xFFFFFFFF).value
        if int(gdi32.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened))) != 0:
            continue
        try:
            address = _D3DKMT_ADAPTERADDRESS()
            query = _D3DKMT_QUERYADAPTERINFO()
            query.hAdapter = opened.hAdapter
            query.Type = _KMTQAITYPE_ADAPTERADDRESS
            query.pPrivateDriverData = ctypes.cast(ctypes.byref(address), ctypes.c_void_p)
            query.PrivateDriverDataSize = ctypes.sizeof(address)
            if int(gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query))) != 0:
                continue
            bus = int(address.BusNumber)
            # The Basic Render adapter reports 0xFFFFFFFF; a real PCI bus is a
            # byte. Anything outside that is not an address we can join on.
            if 0 <= bus <= 0xFF:
                out[(high, low)] = bus
        finally:
            closed = _D3DKMT_CLOSEADAPTER()
            closed.hAdapter = opened.hAdapter
            with contextlib.suppress(Exception):
                gdi32.D3DKMTCloseAdapter(ctypes.byref(closed))
    return out


def _load_nvml() -> Any:
    """Import the NVML bindings (patchable by tests)."""
    import pynvml

    return pynvml


def _pci_bus_of(info: Any) -> int | None:
    """Bus number out of an ``nvmlPciInfo_t``, whichever fields it has.

    ``.bus`` is an int in every binding that exposes it; ``busId`` is the
    ``"00000000:42:00.0"`` domain:bus:device.function string, whose bus half is
    **hex**, which is the trap -- reading ``42`` as decimal silently maps CUDA1
    onto whatever sits at bus 42.
    """
    bus = getattr(info, "bus", None)
    if isinstance(bus, int):
        return int(bus)
    raw = getattr(info, "busId", None) or getattr(info, "busIdLegacy", None)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    parts = str(raw or "").split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-2], 16)
    except ValueError:
        return None


def _nvml_bus_to_index() -> dict[int, int]:
    """``{pci bus: CUDA ordinal}`` from NVML.

    NVML's own device order *is* the CUDA ordinal order that llama.cpp's
    ``--device CUDA<n>`` names, which is the whole reason the join lands on a
    number the planner already speaks. ``nvmlInit``/``nvmlShutdown`` are
    reference counted, so this balanced pair does not disturb the long-lived
    probe in :mod:`studioforge.core.gpu`.
    """
    nvml = _load_nvml()
    nvml.nvmlInit()
    try:
        out: dict[int, int] = {}
        for index in range(int(nvml.nvmlDeviceGetCount())):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            bus = _pci_bus_of(nvml.nvmlDeviceGetPciInfo(handle))
            if bus is not None:
                out.setdefault(bus, index)
        return out
    finally:
        with contextlib.suppress(Exception):
            nvml.nvmlShutdown()


def luid_to_cuda_index(luids: Iterable[LuidKey] = ()) -> dict[LuidKey, int]:
    """``{adapter LUID: CUDA ordinal}`` for the adapters in ``luids``.

    The LUIDs have to be supplied: there is no enumeration here, they come from
    the PDH instance names that need mapping (:func:`pdh_process_gpu_bytes`).
    A bare call answers for the adapters already asked about, which is ``{}`` in
    a process that has not sampled PDH yet.

    **Why this is needed at all.** PDH knows how much VRAM a process holds and
    on which *adapter LUID*; NVML knows the CUDA ordinals but cannot size a
    process on Windows (D23) and exposes no LUID. Neither side can be joined to
    the other directly -- which is why D23 declared the per-GPU split
    unknowable and reported a per-process total instead. The missing edge is
    the PCI address: ``D3DKMTOpenAdapterFromLuid`` +
    ``D3DKMTQueryAdapterInfo(KMTQAITYPE_ADAPTERADDRESS)`` give a LUID's bus, and
    ``nvmlDeviceGetPciInfo`` gives each ordinal's bus. Measured on the reference
    rig: LUID ``0x13C35`` -> bus 0x01 -> CUDA0, ``0x155BF`` -> 0x42 -> CUDA1,
    ``0x1671A`` -> 0xC1 -> CUDA2, ``0x175FB`` -> 0xC2 -> CUDA3.

    Rebuilt from scratch whenever it goes stale or an unseen LUID turns up, and
    never persisted: CUDA ordinals are a per-process, per-environment
    numbering, and a cached one that has silently shifted attributes a card's
    memory to its neighbour.

    Returns ``{}`` off Windows and on any failure; never raises. An empty map
    is a complete answer -- it means "report the per-process total, as before".
    """
    global _luid_cache
    if os.name != "nt":
        return {}
    wanted = {key for key in luids if key != UNKNOWN_LUID}
    now = time.monotonic()
    with _luid_lock:
        stamped, cached = _luid_cache
        if now - stamped < LUID_MAP_TTL_S and not (wanted - _luid_seen):
            return dict(cached)
        # Every LUID ever asked for, so an adapter that cannot be resolved (the
        # Basic Render adapter) is remembered as tried and does not force a
        # rebuild on every single poll.
        tried = wanted | _luid_seen
        _luid_seen.clear()
        _luid_seen.update(tried)
        try:
            buses = _luid_pci_buses(tried)
            by_bus = _nvml_bus_to_index() if buses else {}
        except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
            log.debug("luid_map_failed", error=str(exc))
            _luid_cache = (now, {})
            return {}
        fresh = {luid: by_bus[bus] for luid, bus in buses.items() if bus in by_bus}
        _luid_cache = (now, fresh)
        return dict(fresh)


def reset_luid_cache() -> None:
    """Drop the LUID map (tests)."""
    global _luid_cache
    with _luid_lock:
        _luid_cache = (0.0, {})
        _luid_seen.clear()


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


def _pdh_sample() -> dict[int, dict[LuidKey, int]]:
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

    per_pid: dict[int, dict[LuidKey, int]] = {}
    for instance, value in pairs:
        name = str(instance)
        pid = _pid_from_instance(name)
        if pid is None:
            continue
        try:
            used = int(value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
        if used <= 0:
            continue
        # Kept per adapter, not summed: one process holds memory on several
        # GPUs and *which* one is the question the totals could not answer.
        # A linked pair (``phys_1``) folds into its LUID's bucket.
        luid = _luid_from_instance(name)
        buckets = per_pid.setdefault(pid, {})
        buckets[luid] = buckets.get(luid, 0) + used
    return per_pid


def _pdh_per_luid(*, ttl_s: float = PDH_CACHE_TTL_S) -> dict[int, dict[LuidKey, int]]:
    """The cached raw sample: ``{pid: {adapter LUID: bytes}}``.

    One sample feeds both public views, so a caller that wants the total and
    the split (every caller) still costs exactly one PDH collect per TTL.
    """
    global _pdh_cache, _pdh_disabled
    if os.name != "nt":
        return {}
    now = time.monotonic()
    with _pdh_lock:
        sampled_at, cached = _pdh_cache
        if cached and now - sampled_at < ttl_s:
            return {pid: dict(buckets) for pid, buckets in cached.items()}
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
        return {pid: dict(buckets) for pid, buckets in fresh.items()}


def pdh_process_dedicated_bytes(*, ttl_s: float = PDH_CACHE_TTL_S) -> dict[int, int]:
    """Dedicated GPU memory per pid, in bytes, across every adapter. ``{}`` when
    unavailable.

    **Why this exists.** NVML's per-process memory is 0 for every process on
    Windows under the WDDM driver model -- video memory is owned by the OS
    there, not by the driver, so ``nvmlDeviceGetComputeRunningProcesses``
    enumerates the holders but cannot size them. That is what made
    ``/api/status`` useless during the incident: it listed
    ``llama-server.exe ... used_bytes: 0`` three times while 25 GiB was gone.
    The Windows performance-counter subsystem *does* know, and this is the
    counter Task Manager reads.

    **What the number means.** A per-process *total*, still: it is the right
    figure for "how much has this process taken" and the wrong one for "how
    much is on CUDA1". Use :func:`pdh_process_gpu_bytes` for the second
    question and do not sum this across per-GPU rows.

    Never raises, returns ``{}`` on any failure, and disables itself after the
    first failure (a missing counter set does not come back) having logged one
    warning. Non-Windows always returns ``{}``.
    """
    return {pid: sum(buckets.values()) for pid, buckets in _pdh_per_luid(ttl_s=ttl_s).items()}


def pdh_process_gpu_bytes(*, ttl_s: float = PDH_CACHE_TTL_S) -> dict[int, dict[int, int]]:
    """``{pid: {CUDA ordinal: bytes}}`` -- where each process's VRAM actually is.

    The same PDH sample as :func:`pdh_process_dedicated_bytes`, kept per adapter
    and joined to CUDA ordinals through :func:`luid_to_cuda_index`. Bytes on an
    adapter that does not resolve to a visible CUDA device land under
    :data:`OTHER_ADAPTER` (``-1``) rather than being dropped or spread, so the
    inner values always add back up to the per-process total.

    When the LUID map is unavailable every byte lands under ``-1``: "this
    process holds 30 GiB, on adapters we could not name" is the honest
    degradation, and the caller falls back to NVML's context list for the
    device column.
    """
    per_luid = _pdh_per_luid(ttl_s=ttl_s)
    if not per_luid:
        return {}
    mapping = luid_to_cuda_index({key for buckets in per_luid.values() for key in buckets})
    out: dict[int, dict[int, int]] = {}
    for pid, buckets in per_luid.items():
        merged: dict[int, int] = {}
        for luid, used in buckets.items():
            index = mapping.get(luid, OTHER_ADAPTER)
            merged[index] = merged.get(index, 0) + used
        out[pid] = merged
    return out


def process_gpu_bytes(pid: int) -> dict[int, int]:
    """Per-device bytes held by one pid, ``{}`` when unknown.

    The single-process form of :func:`pdh_process_gpu_bytes`, for callers that
    have a pid and want to know what its load actually placed where -- notably
    the calibration loop, which records a plan's *intended* per-device split and
    can now record the achieved one beside it (D18/D39).
    """
    return dict(pdh_process_gpu_bytes().get(int(pid), {}))


def reset_pdh_cache() -> None:
    """Drop the PDH and LUID caches and re-enable sampling (tests)."""
    global _pdh_cache, _pdh_disabled
    with _pdh_lock:
        _pdh_cache = (0.0, {})
        _pdh_disabled = False
    reset_luid_cache()


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


#: Source label for a device column that is a measurement.
DEVICES_FROM_PDH = "pdh"
#: ...and for one that is only NVML's list of devices the process has a CUDA
#: *context* on. llama.cpp opens a context on every visible device at startup,
#: so this answers "where could it allocate", never "where did it".
DEVICES_FROM_NVML_CONTEXT = "nvml-context"


def _device_column(split: Mapping[int, int], contexts: Sequence[int]) -> tuple[list[int], str]:
    """The GPUs to name for a holder, and where that list came from.

    PDH wins when it can name a CUDA device for the pid, because it is a
    measurement: the devices listed are the ones holding at least
    :data:`HOLDER_MIN_BYTES`, which drops llama.cpp's ~0.2 GiB per-device CUDA
    context and keeps the cards the weights are actually on. An empty list with
    source ``pdh`` is a real answer -- "measured, and nothing on any GPU crosses
    the floor".

    When every measured byte sits on an adapter that could not be resolved to a
    CUDA ordinal (no LUID map, an iGPU, a card hidden by
    ``CUDA_VISIBLE_DEVICES``) there is nothing to name, so the column falls back
    to NVML's context list and says so -- letting a caller read a context as a
    placement is the mistake this function exists to stop.
    """
    if any(index >= 0 for index in split):
        named = sorted(
            index for index, used in split.items() if index >= 0 and used >= HOLDER_MIN_BYTES
        )
        return named, DEVICES_FROM_PDH
    return sorted({int(index) for index in contexts}), DEVICES_FROM_NVML_CONTEXT


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

    One row per pid, never per (gpu, pid). ``per_gpu_bytes`` carries the split
    *inside* the row, from PDH's per-adapter counters (D39), so the device
    column can say "CUDA0 15.5 GiB, CUDA1 14.5 GiB" instead of naming every
    device the process merely has a context on -- which for llama.cpp is all of
    them, and which read as "this 30 GiB model is on all four cards".

    The probe's *raw* holders are read here rather than
    :func:`studioforge.core.gpu.vram_processes`, precisely because that function
    back-fills zeros from PDH. Once back-filled, a pid on two GPUs carries the
    same per-process total on both of its rows and there is no longer any way
    to tell a real NVML pair from one PDH figure counted twice. Reading raw
    keeps summing safe and keeps ``used_bytes_source`` honest.
    """
    ours = {int(pid) for pid in own_pids}
    found_engines, others = scan_llama_servers(engines_dir, own_pids=ours)
    engines = {entry.pid: entry for entry in found_engines}

    entries: list[Any] = []
    getter = getattr(probe, "compute_processes", None) if probe is not None else None
    if getter is not None:
        try:
            entries = list(getter())
        except Exception as exc:  # noqa: BLE001 - a probe gap is not an error
            log.debug("vram_holders_probe_failed", error=str(exc))
    per_pid_bytes = pdh_process_dedicated_bytes()
    per_gpu_bytes = pdh_process_gpu_bytes()

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
    for pid, stray in others.items():
        rows.setdefault(
            pid,
            {"pid": pid, "name": Path(stray.exe).name, "gpu_indices": [], "nvml_bytes": 0},
        )

    holders: list[dict[str, Any]] = []
    desktop_count = 0
    desktop_bytes = 0
    sources: set[str] = set()
    for pid, row in rows.items():
        engine = engines.get(pid)
        other = others.get(pid)
        pdh_bytes = per_pid_bytes.get(pid, 0)
        nvml_bytes = int(row.get("nvml_bytes") or 0)
        if nvml_bytes > 0:
            used, source = nvml_bytes, "nvml"
        elif pdh_bytes > 0:
            used, source = pdh_bytes, "pdh"
        else:
            used, source = 0, "unknown"
        sources.add(source)
        split = per_gpu_bytes.get(pid) or {}
        indices, indices_source = _device_column(split, row["gpu_indices"])
        if engine is not None:
            classification = engine.classification
        elif other is not None:
            classification = CLASS_OTHER_INSTANCE
        else:
            classification = CLASS_FOREIGN
        merged: dict[str, Any] = {
            "pid": pid,
            "name": row["name"],
            "used_bytes": used,
            "used_bytes_source": source,
            "gpu_indices": indices,
            "gpu_indices_source": indices_source,
            "per_gpu_bytes": {str(index): split[index] for index in sorted(split)},
            "is_ours": bool(row.get("is_ours")),
            "engine_process": engine is not None,
            "alias": engine.alias if engine else (other.alias if other else None),
            "port": engine.port if engine else (other.port if other else None),
            "classification": classification,
            "detail": other.detail if other else None,
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
        # "pdh" when at least one row's device column is a measurement rather
        # than an NVML context list, so a client can tell a real placement
        # from "every device this process can see".
        "per_gpu_bytes_source": (
            DEVICES_FROM_PDH
            if any(row.get("gpu_indices_source") == DEVICES_FROM_PDH for row in holders)
            else DEVICES_FROM_NVML_CONTEXT
        ),
        "engines_dir": str(engines_dir),
        "engine_process_count": len(engines),
        "other_instance_count": len(others),
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
    read ``gpu_index``/``pid``/``name``/``used_bytes``/``is_ours``.

    Each entry is one ``(gpu, pid)`` pair as NVML enumerated it, i.e. one
    *context*, not one placement. ``device_bytes`` is what the pid actually
    holds on **this** row's ``gpu_index`` (D39) and is the only figure here that
    may be summed across rows; ``used_bytes`` stays the per-process total, and
    ``per_gpu_bytes`` repeats the whole split on every row of the pid.
    ``gpu_indices_source`` says which of the two the device attribution is:
    ``pdh`` (measured) or ``nvml-context`` (the process merely has a context
    there).

    Desktop noise is collapsed into ``desktop_processes_count`` rather than
    listed. The old payload was ~20 entries of compositor and browser
    processes, all reporting zero bytes, which is precisely how three leaked
    engine children hid in plain sight.
    """
    entries = payload.get("vram_processes")
    if not isinstance(entries, list):
        return payload
    ours = {int(pid) for pid in own_pids}
    found_engines, others = scan_llama_servers(engines_dir, own_pids=ours)
    engines = {entry.pid: entry for entry in found_engines}
    per_pid_bytes = pdh_process_dedicated_bytes()
    per_gpu_bytes = pdh_process_gpu_bytes()

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
        other = others.get(pid)
        split = per_gpu_bytes.get(pid) or {}
        _, indices_source = _device_column(split, ())
        entry["per_gpu_bytes"] = {str(index): split[index] for index in sorted(split)}
        entry["gpu_indices_source"] = indices_source
        entry["device_bytes"] = split.get(int(entry.get("gpu_index") or 0)) if split else None
        if engine is not None:
            entry["classification"] = engine.classification
        elif other is not None:
            entry["classification"] = CLASS_OTHER_INSTANCE
        else:
            entry["classification"] = CLASS_FOREIGN
        entry["detail"] = other.detail if other else None
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
        elif other is not None:
            entry["alias"] = other.alias
            entry["port"] = other.port
        kept.append(entry)

    # A llama-server the probe never reported still belongs in the list: a
    # holder nothing can see is the exact failure this endpoint now exists to
    # prevent. Ours and somebody else's alike -- an unseen 19 GiB holder is no
    # less of a problem for having been launched from another directory.
    for pid, engine in engines.items():
        if pid in seen:
            continue
        split = per_gpu_bytes.get(pid) or {}
        kept.append(
            {
                "gpu_index": -1,
                "pid": pid,
                "name": Path(engine.exe).name,
                "used_bytes": per_pid_bytes.get(pid, 0),
                "used_bytes_source": "pdh" if per_pid_bytes.get(pid) else "unknown",
                "per_gpu_bytes": {str(index): split[index] for index in sorted(split)},
                "gpu_indices_source": _device_column(split, ())[1],
                "device_bytes": None,
                "is_ours": engine.is_ours,
                "engine_process": True,
                "classification": engine.classification,
                "detail": None,
                "parent_pid": engine.parent_pid,
                "parent_name": engine.parent_name,
                "parent_cmdline": engine.parent_cmdline,
                "parent_alive": engine.parent_alive,
                "alias": engine.alias,
            }
        )
    for pid, stray in others.items():
        if pid in seen:
            continue
        split = per_gpu_bytes.get(pid) or {}
        kept.append(
            {
                "gpu_index": -1,
                "pid": pid,
                "name": Path(stray.exe).name,
                "used_bytes": per_pid_bytes.get(pid, 0),
                "used_bytes_source": "pdh" if per_pid_bytes.get(pid) else "unknown",
                "per_gpu_bytes": {str(index): split[index] for index in sorted(split)},
                "gpu_indices_source": _device_column(split, ())[1],
                "device_bytes": None,
                "is_ours": False,
                "engine_process": False,
                "classification": CLASS_OTHER_INSTANCE,
                "detail": stray.detail,
                "alias": stray.alias,
                "port": stray.port,
            }
        )

    payload["vram_processes"] = kept
    payload["desktop_processes_count"] = len(desktop_by_pid)
    payload["desktop_processes_bytes"] = sum(desktop_by_pid.values())
    payload["vram_orphan_count"] = sum(
        1 for entry in engines.values() if entry.classification == CLASS_ORPHAN
    )
    return payload
