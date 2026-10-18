"""GPU hardware probe behind a swappable interface.

The planner and the engine selector both need GPU facts (free VRAM, compute
capability, driver's CUDA level). Everything flows through :class:`GpuProbe`
so that tests never touch real hardware. Real implementations:
CUDA/NVML (:class:`NvmlGpuProbe`), ROCm (:class:`RocmGpuProbe`, via amd-smi /
rocm-smi / rocminfo), Vulkan (:class:`VulkanGpuProbe`, via vulkaninfo) and
Metal (:class:`MetalGpuProbe`, via system_profiler). :class:`AutoGpuProbe`
tries them in that order and is the default (``gpu.backend: auto``);
:class:`NullGpuProbe` stands in on GPU-less boxes and :class:`FakeGpuProbe`
serves tests and synthetic-hardware integration runs. The vendor-tool output
parsers live in :mod:`studioforge.core.hw_parsers`.

Design rules:

* NVML is initialised lazily on first use, never at import time -- importing
  this module must be instant and safe on a machine with no NVIDIA driver.
* A broken/missing NVML degrades to "no GPUs" (one warning logged), never a
  traceback: the planner turns an empty GPU list into a clean rejection.
* Every optional per-GPU field degrades individually to ``None``/0 so one
  flaky NVML call cannot hide the whole device.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import psutil

from studioforge.core.hw_parsers import (
    RocmAgent,
    gpus_from_rocminfo,
    gpus_from_vulkan,
    merge_amd_smi,
    parse_amd_smi_metric,
    parse_amd_smi_static,
    parse_rocm_smi_json,
    parse_rocminfo,
    parse_system_profiler_displays,
    parse_vulkaninfo_json,
    parse_vulkaninfo_text,
)
from studioforge.errors import ConfigError
from studioforge.logging import get_logger
from studioforge.types import GpuInfo, VramProcess

log = get_logger(__name__)

_ENV_PROBE = "SF_GPU_PROBE"
_ENV_FAKE_GPUS = "SF_FAKE_GPUS"


def _load_nvml() -> Any:
    """Import the NVML bindings.

    Split out into a module-level function so tests can monkeypatch it to
    raise (simulating a GPU-less box) or to return a fake module.
    """
    import pynvml

    return pynvml


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


@runtime_checkable
class GpuProbe(Protocol):
    """Read-only view of the machine's GPUs.

    Implementations must never raise from the query methods: a broken backend
    reports itself as "no GPUs available" instead.
    """

    @property
    def backend(self) -> str:
        """Backend identifier: ``"cuda"``, ``"rocm"``, ``"vulkan"``, ``"metal"``,
        ``"null"``, or ``"fake"`` (an :class:`AutoGpuProbe` reports the one it
        selected)."""
        ...

    def available(self) -> bool:
        """Whether the backend is usable and at least one GPU is present."""
        ...

    def list_gpus(self) -> list[GpuInfo]:
        """All GPUs sorted by index, with live free-VRAM numbers."""
        ...

    def get_gpu(self, index: int) -> GpuInfo | None:
        """A single GPU by index, or ``None`` when it does not exist."""
        ...

    def driver_version(self) -> str | None:
        """Display driver version string, e.g. ``"610.88"``."""
        ...

    def compute_processes(self) -> list[VramProcess]:
        """Processes currently holding VRAM, across every GPU.

        Best-effort by contract: an implementation that cannot enumerate
        returns an empty list rather than raising. NVML frequently cannot see
        process-level usage inside containers, WSL, or under MIG, so callers
        must treat "no holders" as "unknown", never as "nothing is running".
        """
        ...

    def cuda_driver_version(self) -> tuple[int, int] | None:
        """Maximum CUDA version the installed driver supports, e.g. ``(13, 0)``."""
        ...

    def shutdown(self) -> None:
        """Release backend resources. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# NVML (CUDA) implementation
# ---------------------------------------------------------------------------


class NvmlGpuProbe:
    """Real probe over NVML via ``nvidia-ml-py`` (the ``pynvml`` module).

    NVML init happens lazily on first query. If the import or ``nvmlInit``
    fails, a warning is logged once and the probe behaves like
    :class:`NullGpuProbe` from then on.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._nvml: Any = None
        self._handles: list[Any] = []
        self._initialized = False
        self._failed = False
        #: Why NVML is unusable (import/init failure or zero devices); ``None``
        #: while it works. Read by :class:`AutoGpuProbe` to explain a skip.
        self.failure_reason: str | None = None

    @property
    def backend(self) -> str:
        return "cuda"

    # -- lifecycle ------------------------------------------------------

    def _ensure_init(self) -> bool:
        with self._lock:
            if self._initialized:
                return True
            if self._failed:
                return False
            try:
                nvml = _load_nvml()
                nvml.nvmlInit()
            except Exception as exc:
                self._failed = True
                self.failure_reason = f"NVML unavailable: {exc}"
                log.warning(
                    "nvml_unavailable",
                    error=str(exc),
                    hint="No NVIDIA driver/NVML found; reporting zero GPUs.",
                )
                return False
            self._nvml = nvml
            try:
                count = int(nvml.nvmlDeviceGetCount())
                self._handles = [nvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            except Exception as exc:
                log.warning("nvml_enumeration_failed", error=str(exc))
                self._handles = []
                self.failure_reason = f"NVML could not enumerate devices: {exc}"
            if not self._handles and self.failure_reason is None:
                self.failure_reason = "NVML initialised but reported 0 devices"
            self._initialized = True
            return True

    def shutdown(self) -> None:
        with self._lock:
            if self._initialized and self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception as exc:  # pragma: no cover - NVML teardown quirk
                    log.warning("nvml_shutdown_failed", error=str(exc))
            self._nvml = None
            self._handles = []
            self._initialized = False
            self._failed = False
            self.failure_reason = None

    # -- queries --------------------------------------------------------

    def available(self) -> bool:
        return self._ensure_init() and bool(self._handles)

    def list_gpus(self) -> list[GpuInfo]:
        if not self._ensure_init():
            return []
        return [self._read_gpu(i, h) for i, h in enumerate(self._handles)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        if not self._ensure_init():
            return None
        if 0 <= index < len(self._handles):
            return self._read_gpu(index, self._handles[index])
        return None

    def driver_version(self) -> str | None:
        if not self._ensure_init():
            return None
        try:
            return _decode(self._nvml.nvmlSystemGetDriverVersion())
        except Exception as exc:
            log.warning("nvml_driver_version_failed", error=str(exc))
            return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        """Max CUDA version the driver supports, from NVML's packed int.

        NVML returns e.g. ``13000`` for CUDA 13.0 and ``12040`` for 12.4:
        ``major = v // 1000``, ``minor = (v % 1000) // 10``. The engine
        selector uses this to pick between CUDA 12.x and 13.x prebuilts.
        """
        if not self._ensure_init():
            return None
        try:
            getter = getattr(
                self._nvml,
                "nvmlSystemGetCudaDriverVersion_v2",
                None,
            ) or getattr(self._nvml, "nvmlSystemGetCudaDriverVersion", None)
            if getter is None:
                return None
            packed = int(getter())
        except Exception as exc:
            log.warning("nvml_cuda_version_failed", error=str(exc))
            return None
        return (packed // 1000, (packed % 1000) // 10)

    def compute_processes(self) -> list[VramProcess]:
        """Every process NVML reports as holding VRAM, per GPU.

        Both compute *and* graphics contexts are queried: on a workstation the
        holder is as likely to be a desktop compositor or a game as it is a
        CUDA job, and either way the VRAM is gone. Every step degrades on its
        own -- an NVML that cannot enumerate processes (common in containers,
        WSL and under MIG) yields an empty list, never an exception, because
        this feeds a diagnostic and must not be able to break a load.
        """
        if not self._ensure_init():
            return []
        nvml = self._nvml
        out: list[VramProcess] = []
        for index, handle in enumerate(self._handles):
            seen: set[int] = set()
            entries: list[Any] = []
            # Newest binding spelling first, falling back to the older alias:
            # the _v3 variants exist only in recent nvidia-ml-py, and the plain
            # names only in older ones.
            for names in (
                ("nvmlDeviceGetComputeRunningProcesses_v3", "nvmlDeviceGetComputeRunningProcesses"),
                (
                    "nvmlDeviceGetGraphicsRunningProcesses_v3",
                    "nvmlDeviceGetGraphicsRunningProcesses",
                ),
            ):
                entries.extend(self._running_processes(nvml, handle, index, names))
            for entry in entries:
                try:
                    pid = int(getattr(entry, "pid", 0) or 0)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    continue
                if pid <= 0 or pid in seen:
                    continue
                seen.add(pid)
                used = getattr(entry, "usedGpuMemory", None)
                out.append(
                    VramProcess(
                        gpu_index=index,
                        pid=pid,
                        name=_process_name(nvml, pid),
                        used_bytes=int(used) if isinstance(used, int) else 0,
                    )
                )
        return sorted(out, key=lambda p: (p.gpu_index, -p.used_bytes, p.pid))

    @staticmethod
    def _running_processes(nvml: Any, handle: Any, index: int, names: Sequence[str]) -> list[Any]:
        """First NVML getter in ``names`` that answers, or an empty list."""
        for name in names:
            getter = getattr(nvml, name, None)
            if getter is None:
                continue
            try:
                return list(getter(handle) or [])
            except Exception as exc:
                log.debug("nvml_processes_failed", index=index, getter=name, error=str(exc))
        return []

    # -- per-device read ------------------------------------------------

    def _read_gpu(self, index: int, handle: Any) -> GpuInfo:
        """Read one device; every optional field degrades individually."""
        nvml = self._nvml
        name = "unknown"
        try:
            name = _decode(nvml.nvmlDeviceGetName(handle))
        except Exception:
            log.debug("nvml_field_failed", field="name", index=index)

        total = free = used = 0
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            total, free, used = int(mem.total), int(mem.free), int(mem.used)
        except Exception as exc:
            log.warning("nvml_field_failed", field="memory", index=index, error=str(exc))

        utilization: float | None = None
        try:
            utilization = float(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            log.debug("nvml_field_failed", field="utilization", index=index)

        temperature: float | None = None
        try:
            sensor = getattr(nvml, "NVML_TEMPERATURE_GPU", 0)
            temperature = float(nvml.nvmlDeviceGetTemperature(handle, sensor))
        except Exception:
            log.debug("nvml_field_failed", field="temperature", index=index)

        capability: tuple[int, int] | None = None
        try:
            major, minor = nvml.nvmlDeviceGetCudaComputeCapability(handle)
            capability = (int(major), int(minor))
        except Exception:
            log.debug("nvml_field_failed", field="compute_capability", index=index)

        return GpuInfo(
            index=index,
            name=name,
            total_bytes=total,
            free_bytes=free,
            used_bytes=used,
            utilization_pct=utilization,
            temperature_c=temperature,
            compute_capability=capability,
            backend="cuda",
            vendor="nvidia",
        )


def _process_name(nvml: Any, pid: int) -> str:
    """Executable name for a pid: psutil first, NVML second, else "unknown".

    psutil is asked first because NVML's own ``nvmlSystemGetProcessName``
    returns a full path (and fails outright for a process owned by another
    user), while the whole point of the name here is to be recognisable in a
    one-line error message.
    """
    try:
        return psutil.Process(pid).name()
    except (psutil.Error, ValueError):
        pass
    try:
        return Path(_decode(nvml.nvmlSystemGetProcessName(pid))).name or "unknown"
    except Exception:
        return "unknown"


def _decode(value: str | bytes) -> str:
    """NVML strings arrive as ``bytes`` in old bindings and ``str`` in new ones."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


# ---------------------------------------------------------------------------
# Null implementation
# ---------------------------------------------------------------------------


class NullGpuProbe:
    """Probe for a machine with no usable GPUs. Always empty, never raises."""

    @property
    def backend(self) -> str:
        return "null"

    def available(self) -> bool:
        return False

    def list_gpus(self) -> list[GpuInfo]:
        return []

    def get_gpu(self, index: int) -> GpuInfo | None:
        return None

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def compute_processes(self) -> list[VramProcess]:
        return []

    def shutdown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Fake implementation (tests / synthetic-hardware integration runs)
# ---------------------------------------------------------------------------


class FakeGpuProbe:
    """In-memory probe built from a list of :class:`GpuInfo`.

    Meant to be genuinely useful to other test suites: ``set_free`` and
    ``consume`` let a planner test simulate a partly-occupied GPU without any
    NVML in sight. The constructor deep-copies its input so mutations stay
    local to the probe.
    """

    def __init__(
        self,
        gpus: Sequence[GpuInfo],
        *,
        driver: str | None = "0.0",
        cuda_version: tuple[int, int] | None = (13, 0),
    ) -> None:
        self._gpus: dict[int, GpuInfo] = {g.index: g.model_copy(deep=True) for g in gpus}
        self._driver = driver
        self._cuda_version = cuda_version
        self._processes: list[VramProcess] = []

    @property
    def backend(self) -> str:
        return "fake"

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [self._gpus[i].model_copy(deep=True) for i in sorted(self._gpus)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        gpu = self._gpus.get(index)
        return gpu.model_copy(deep=True) if gpu is not None else None

    def driver_version(self) -> str | None:
        return self._driver

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return self._cuda_version

    def compute_processes(self) -> list[VramProcess]:
        return [p.model_copy(deep=True) for p in self._processes]

    def shutdown(self) -> None:
        return None

    # -- test helpers ---------------------------------------------------

    def set_processes(self, processes: Sequence[VramProcess]) -> None:
        """Set the VRAM holders this probe reports (tests / synthetic runs)."""
        self._processes = [p.model_copy(deep=True) for p in processes]

    def set_free(self, index: int, free_bytes: int) -> None:
        """Set a GPU's free VRAM directly (``used`` is recomputed)."""
        gpu = self._require(index)
        if not 0 <= free_bytes <= gpu.total_bytes:
            raise ValueError(
                f"free_bytes={free_bytes} out of range for GPU {index} (total={gpu.total_bytes})"
            )
        gpu.free_bytes = free_bytes
        gpu.used_bytes = gpu.total_bytes - free_bytes

    def consume(self, index: int, nbytes: int) -> None:
        """Simulate an allocation of ``nbytes`` on a GPU."""
        gpu = self._require(index)
        if nbytes < 0:
            raise ValueError("nbytes must be >= 0")
        if nbytes > gpu.free_bytes:
            raise ValueError(
                f"cannot consume {nbytes} bytes on GPU {index}: only {gpu.free_bytes} free"
            )
        self.set_free(index, gpu.free_bytes - nbytes)

    def _require(self, index: int) -> GpuInfo:
        gpu = self._gpus.get(index)
        if gpu is None:
            raise KeyError(f"FakeGpuProbe has no GPU with index {index}")
        return gpu


def _fake_probe_from_env() -> FakeGpuProbe:
    """Build a :class:`FakeGpuProbe` from the ``SF_FAKE_GPUS`` JSON env var.

    The spec is a list of objects with ``index``, ``name``, ``total_bytes``,
    ``free_bytes``, and ``compute_capability`` (a two-int list). Missing
    ``free_bytes`` defaults to ``total_bytes``.
    """
    raw = os.environ.get(_ENV_FAKE_GPUS, "").strip() or "[]"
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{_ENV_FAKE_GPUS} is not valid JSON: {exc}") from exc
    if not isinstance(spec, list):
        raise ConfigError(f"{_ENV_FAKE_GPUS} must be a JSON list of GPU objects")
    gpus: list[GpuInfo] = []
    for item in spec:
        if not isinstance(item, dict):
            raise ConfigError(f"{_ENV_FAKE_GPUS} entries must be JSON objects")
        try:
            total = int(item["total_bytes"])
            free = int(item.get("free_bytes", total))
            cc_raw = item.get("compute_capability")
            capability = (int(cc_raw[0]), int(cc_raw[1])) if cc_raw is not None else None
            gpus.append(
                GpuInfo(
                    index=int(item["index"]),
                    name=str(item.get("name", "Fake GPU")),
                    total_bytes=total,
                    free_bytes=free,
                    used_bytes=int(item.get("used_bytes", total - free)),
                    compute_capability=capability,
                )
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ConfigError(f"Bad {_ENV_FAKE_GPUS} entry {item!r}: {exc}") from exc
    return FakeGpuProbe(gpus)


# ---------------------------------------------------------------------------
# Command-line probes: ROCm, Vulkan, Metal
# ---------------------------------------------------------------------------
#
# Budget formula (the same for every backend; there is deliberately NO
# heuristic bonus anywhere):
#
#   total_bytes = what the driver / tool reports as this device's memory
#   free_bytes  = what the driver / tool reports as still available
#   budget      = free_bytes   (the planner then applies its own headroom)
#
# Unified-memory systems (``GpuInfo.unified_memory``: Strix Halo and other
# APUs, Apple silicon) are NOT given extra memory for the system RAM they
# could in principle borrow. They use the same formula on the figure the
# driver reports for the device:
#
# * ROCm APU: amd-smi / rocm-smi ``VRAM`` total (the BIOS/driver carve-out),
#   free = total - used. GTT is not added.
# * Vulkan integrated GPU: the largest DEVICE_LOCAL heap, free = its
#   ``VK_EXT_memory_budget`` budget (capped at the heap size).
# * Metal (Apple silicon): system_profiler reports no VRAM; the pool is the
#   machine's physical memory (psutil) and free is the OS's "available"
#   memory. Metal itself caps a process below physical RAM (its recommended
#   working set), so on Metal ``free_bytes`` is an UPPER bound and the
#   planner's headroom fraction is what absorbs the difference.
#
# Raising the number for an APU (e.g. counting GTT) is a user decision made
# in the BIOS / driver, not something this module guesses.

CommandRunner = Callable[[Sequence[str], float], "tuple[int, str]"]


def run_command(argv: Sequence[str], timeout: float = 20.0) -> tuple[int, str]:
    """Run a read-only query tool; ``(returncode, stdout)``, never raises.

    A missing binary is ``(127, "")``, a timeout ``(124, "")``. Injected into
    every probe so tests never spawn anything.
    """
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell, read-only query tools
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    except OSError:
        return 126, ""
    return done.returncode, done.stdout or ""


class _ProbeFailure(Exception):
    """A backend that cannot list GPUs, with the reason to show the operator."""


class _CliProbe:
    """Shared plumbing for probes that shell out to a vendor tool.

    Subclasses implement :meth:`_query` returning ``(gpus, driver)`` or
    raising :class:`_ProbeFailure` with a reason. The result is cached for
    ``ttl_s`` seconds (the tools take 50-500 ms and the planner polls), and a
    failure is remembered in :attr:`failure_reason` so the auto-selector can
    say why a backend was skipped.
    """

    _BACKEND = "?"

    def __init__(
        self,
        *,
        runner: CommandRunner | None = None,
        which: Callable[[str], str | None] | None = None,
        ttl_s: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._run: CommandRunner = runner or run_command
        self._which = which or shutil.which
        self._ttl = ttl_s
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._gpus: list[GpuInfo] = []
        self._driver: str | None = None
        self._stamp: float | None = None
        self.failure_reason: str | None = None
        #: Human-readable caveats from the last query (e.g. "free memory unknown").
        self.notes: list[str] = []
        self.source: str | None = None

    @property
    def backend(self) -> str:
        return self._BACKEND

    # subclasses -------------------------------------------------------
    def _query(self) -> tuple[list[GpuInfo], str | None]:  # pragma: no cover - abstract
        raise NotImplementedError

    # shared -----------------------------------------------------------
    def _refresh(self) -> list[GpuInfo]:
        with self._lock:
            now = self._clock()
            if self._stamp is not None and now - self._stamp < self._ttl:
                return self._gpus
            try:
                gpus, driver = self._query()
                self.failure_reason = None
            except _ProbeFailure as exc:
                gpus, driver = [], None
                self.failure_reason = str(exc)
                log.debug("gpu_probe_failed", backend=self._BACKEND, reason=str(exc))
            except Exception as exc:  # noqa: BLE001 - a probe must never raise
                gpus, driver = [], None
                self.failure_reason = f"unexpected error: {exc}"
                log.warning("gpu_probe_error", backend=self._BACKEND, error=str(exc))
            self._gpus, self._driver, self._stamp = gpus, driver, now
            return self._gpus

    def available(self) -> bool:
        return bool(self._refresh())

    def list_gpus(self) -> list[GpuInfo]:
        return [g.model_copy(deep=True) for g in self._refresh()]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return next((g for g in self.list_gpus() if g.index == index), None)

    def driver_version(self) -> str | None:
        self._refresh()
        return self._driver

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def compute_processes(self) -> list[VramProcess]:
        # Per-process VRAM attribution needs vendor-specific tools; callers
        # treat an empty list as "unknown" (see the Protocol docstring).
        return []

    def shutdown(self) -> None:
        with self._lock:
            self._gpus, self._driver, self._stamp = [], None, None
            self.failure_reason = None

    def _text(self, argv: Sequence[str], *, timeout: float = 20.0) -> str | None:
        """stdout of ``argv`` when the tool exists and exits 0, else ``None``."""
        if self._which(argv[0]) is None:
            return None
        code, out = self._run(argv, timeout)
        return out if code == 0 and out.strip() else None


class RocmGpuProbe(_CliProbe):
    """AMD GPUs through ``amd-smi``, ``rocm-smi`` or (identity only) ``rocminfo``.

    Tried in this order, first usable wins: ``amd-smi static|metric --json``
    (current ROCm), ``rocm-smi --showmeminfo vram --showproductname --json``
    (older ROCm), ``rocminfo`` (no live counters: names, gfx target and total
    memory only, free is reported equal to total and a note says so). The gfx
    target always comes from the driver (``target_graphics_version`` /
    ``GFX Version`` / the rocminfo agent name), never from the marketing name.
    """

    _BACKEND = "rocm"

    def _query(self) -> tuple[list[GpuInfo], str | None]:
        gpus: list[GpuInfo] = []
        driver: str | None = None
        self.notes = []
        self.source = None

        static_text = self._text(["amd-smi", "static", "--json"])
        if static_text is not None:
            static = parse_amd_smi_static(static_text)
            if static:
                metric_text = self._text(["amd-smi", "metric", "--json"], timeout=30.0)
                metric = parse_amd_smi_metric(metric_text) if metric_text else {}
                if not metric:
                    self.notes.append(
                        "amd-smi metric unavailable: free VRAM assumed equal to total"
                    )
                gpus, driver = merge_amd_smi(static, metric)
                self.source = "amd-smi"
        if not gpus:
            smi_text = self._text(
                ["rocm-smi", "--showmeminfo", "vram", "--showproductname", "--json"]
            )
            if smi_text is not None:
                parsed = parse_rocm_smi_json(smi_text)
                gpus, driver = parsed.gpus, parsed.driver
                if gpus:
                    self.source = "rocm-smi"
        agents: list[RocmAgent] = []
        if not gpus or any(g.gfx_target is None for g in gpus):
            info_text = self._text(["rocminfo"])
            agents = parse_rocminfo(info_text) if info_text else []
        if not gpus and agents:
            gpus = gpus_from_rocminfo(agents)
            self.source = "rocminfo"
            self.notes.append("rocminfo has no live counters: free VRAM is reported equal to total")
        elif agents:
            for gpu, agent in zip(gpus, agents, strict=False):
                if gpu.gfx_target is None:
                    gpu.gfx_target = agent.gfx_target
        if not gpus:
            tools = [t for t in ("amd-smi", "rocm-smi", "rocminfo") if self._which(t)]
            raise _ProbeFailure(
                "no AMD GPU reported: "
                + (
                    f"{', '.join(tools)} found but listed no devices"
                    if tools
                    else "none of amd-smi, rocm-smi, rocminfo is installed"
                )
            )
        return gpus, driver


class VulkanGpuProbe(_CliProbe):
    """Any Vulkan GPU, through ``vulkaninfo``.

    ``vulkaninfo --summary`` names the devices; the memory heaps only appear
    in the full ``vulkaninfo`` output, so both are read. If the full output is
    unusable ``vulkaninfo --json`` is tried for a single device. Software
    rasterisers (llvmpipe) are dropped: they are CPU inference by another
    name, which STRICT_GPU forbids. Device order follows ``vulkaninfo``;
    llama.cpp's ``Vulkan<N>`` numbering is checked against
    ``llama-server --list-devices`` by the engine manager, not assumed here.
    """

    _BACKEND = "vulkan"

    def _query(self) -> tuple[list[GpuInfo], str | None]:
        self.notes = []
        summary = self._text(["vulkaninfo", "--summary"])
        if summary is None:
            raise _ProbeFailure(
                "vulkaninfo is not installed or failed (install the Vulkan runtime tools)"
            )
        devices = parse_vulkaninfo_text(summary)
        full = self._text(["vulkaninfo"], timeout=60.0)
        if full:
            by_index = {d.index: d for d in parse_vulkaninfo_text(full)}
            for device in devices:
                other = by_index.get(device.index)
                if other is not None and other.heaps:
                    device.heaps = other.heaps
        if devices and not any(d.heaps for d in devices):
            blob = self._text(["vulkaninfo", "--json"], timeout=60.0)
            parsed = parse_vulkaninfo_json(blob) if blob else []
            if parsed and len(devices) == 1:
                devices[0].heaps = parsed[0].heaps
        gpus, notes = gpus_from_vulkan(devices)
        self.notes.extend(notes)
        if not gpus:
            raise _ProbeFailure(
                "vulkaninfo listed no usable GPU" + (f" ({'; '.join(notes)})" if notes else "")
            )
        self.source = "vulkaninfo"
        driver = next((d.driver_info for d in devices if d.driver_info), None)
        return gpus, driver


def _physical_memory() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


class MetalGpuProbe(_CliProbe):
    """Apple GPUs through ``system_profiler SPDisplaysDataType -json`` (macOS only)."""

    _BACKEND = "metal"

    def __init__(
        self,
        *,
        system: Callable[[], str] | None = None,
        memory: Callable[[], tuple[int, int]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._system = system or platform.system
        #: ``() -> (total_bytes, available_bytes)`` of physical memory.
        self._memory = memory or _physical_memory

    def _query(self) -> tuple[list[GpuInfo], str | None]:
        self.notes = []
        if self._system() != "Darwin":
            raise _ProbeFailure("not macOS: the Metal backend only exists on Apple platforms")
        text = self._text(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30.0)
        if text is None:
            raise _ProbeFailure("system_profiler SPDisplaysDataType failed")
        total, available = self._memory()
        gpus = parse_system_profiler_displays(text, memory_bytes=total, available_bytes=available)
        if not gpus:
            raise _ProbeFailure("system_profiler listed no GPU")
        if any(g.unified_memory for g in gpus):
            self.notes.append(
                "unified memory: pool is physical RAM; Metal's per-process working-set cap "
                "is lower, so free VRAM is an upper bound"
            )
        self.source = "system_profiler"
        return gpus, None


# ---------------------------------------------------------------------------
# Auto selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeAttempt:
    """One backend the auto-selector tried, and what came of it."""

    backend: str
    ok: bool
    reason: str = ""
    gpu_count: int = 0


class AutoGpuProbe:
    """Tries NVML -> ROCm -> Vulkan -> Metal; the first that lists a GPU wins.

    Order matters and is deliberate: an NVIDIA box with a working driver always
    resolves to NVML (identical to the pre-``auto`` behaviour, with real
    per-process VRAM and a CUDA driver version); Vulkan is only reached when
    no vendor-native API lists anything, so an NVIDIA or AMD card is never
    driven through Vulkan merely because the Vulkan runtime is installed.
    One backend per process: a box with an NVIDIA and an AMD card uses the
    first one that answers.

    Resolution is lazy and sticky until :meth:`shutdown`. :attr:`attempts`
    records every backend tried and why the failures failed, which is what
    ``/api/gpus`` and the setup wizard should show.
    """

    def __init__(self, candidates: Sequence[GpuProbe] | None = None) -> None:
        self._candidates: list[GpuProbe] = (
            list(candidates)
            if candidates is not None
            else [NvmlGpuProbe(), RocmGpuProbe(), VulkanGpuProbe(), MetalGpuProbe()]
        )
        self._lock = threading.Lock()
        self._selected: GpuProbe | None = None
        self._resolved = False
        self.attempts: list[ProbeAttempt] = []

    def _resolve(self) -> GpuProbe | None:
        with self._lock:
            if self._resolved:
                return self._selected
            attempts: list[ProbeAttempt] = []
            selected: GpuProbe | None = None
            for probe in self._candidates:
                reason = ""
                try:
                    count = len(probe.list_gpus())
                except Exception as exc:  # noqa: BLE001 - a probe must never break detection
                    count = 0
                    reason = f"error: {exc}"
                if not count and not reason:
                    reason = getattr(probe, "failure_reason", None) or "no GPUs listed"
                attempts.append(ProbeAttempt(probe.backend, count > 0, reason, count))
                if count > 0:
                    selected = probe
                    break
            self.attempts, self._selected, self._resolved = attempts, selected, True
            log.info(
                "gpu_probe_auto",
                selected=selected.backend if selected else None,
                attempts=[{"backend": a.backend, "ok": a.ok, "reason": a.reason} for a in attempts],
            )
            return selected

    @property
    def backend(self) -> str:
        selected = self._resolve()
        return selected.backend if selected is not None else "null"

    @property
    def selected(self) -> GpuProbe | None:
        return self._resolve()

    def diagnostics(self) -> dict[str, Any]:
        """Which backend was used and why each earlier one was not."""
        selected = self._resolve()
        notes = list(getattr(selected, "notes", []) or []) if selected is not None else []
        return {
            "selected": selected.backend if selected is not None else None,
            "source": getattr(selected, "source", None) if selected is not None else None,
            "attempts": [
                {"backend": a.backend, "ok": a.ok, "reason": a.reason, "gpu_count": a.gpu_count}
                for a in self.attempts
            ],
            "notes": notes,
        }

    def available(self) -> bool:
        selected = self._resolve()
        return selected is not None and selected.available()

    def list_gpus(self) -> list[GpuInfo]:
        selected = self._resolve()
        return selected.list_gpus() if selected is not None else []

    def get_gpu(self, index: int) -> GpuInfo | None:
        selected = self._resolve()
        return selected.get_gpu(index) if selected is not None else None

    def driver_version(self) -> str | None:
        selected = self._resolve()
        return selected.driver_version() if selected is not None else None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        selected = self._resolve()
        return selected.cuda_driver_version() if selected is not None else None

    def compute_processes(self) -> list[VramProcess]:
        selected = self._resolve()
        return selected.compute_processes() if selected is not None else []

    def shutdown(self) -> None:
        with self._lock:
            for probe in self._candidates:
                with contextlib.suppress(Exception):
                    probe.shutdown()
            self._selected, self._resolved, self.attempts = None, False, []


# ---------------------------------------------------------------------------
# Module-level probe factory + helpers
# ---------------------------------------------------------------------------

_probe: GpuProbe | None = None
_probe_lock = threading.Lock()

#: Accepted ``gpu.backend`` / ``SF_GPU_PROBE`` values -> the probe family they
#: select. ``auto`` is its own family: the cache is keyed on the *request*, not
#: on the backend the auto-selector happened to resolve to.
_BACKEND_FOR_CHOICE = {
    "auto": "auto",
    "nvml": "cuda",
    "cuda": "cuda",
    "rocm": "rocm",
    "vulkan": "vulkan",
    "metal": "metal",
    "null": "null",
    "fake": "fake",
}
_probe_family: str | None = None


def get_probe(*, force: str | None = None) -> GpuProbe:
    """The process-wide probe, created on first call and cached.

    ``force`` (or env ``SF_GPU_PROBE``) selects ``auto`` (the default: NVML ->
    ROCm -> Vulkan -> Metal), ``nvml``/``cuda``, ``rocm``, ``vulkan``,
    ``metal``, ``null`` or ``fake``; ``fake`` reads its synthetic GPUs from env
    ``SF_FAKE_GPUS``. A force that names a different family than the cached
    probe rebuilds the cache.
    """
    global _probe, _probe_family
    choice = (force or os.environ.get(_ENV_PROBE, "")).strip().lower() or None
    if choice is not None and choice not in _BACKEND_FOR_CHOICE:
        raise ConfigError(
            f"Unknown GPU probe backend {choice!r}; expected one of "
            f"{sorted(set(_BACKEND_FOR_CHOICE))}"
        )
    family = _BACKEND_FOR_CHOICE[choice] if choice is not None else "auto"
    with _probe_lock:
        if _probe is not None and (choice is None or _probe_family == family):
            return _probe
        if family == "auto":
            _probe = AutoGpuProbe()
        elif family == "cuda":
            _probe = NvmlGpuProbe()
        elif family == "rocm":
            _probe = RocmGpuProbe()
        elif family == "vulkan":
            _probe = VulkanGpuProbe()
        elif family == "metal":
            _probe = MetalGpuProbe()
        elif family == "null":
            _probe = NullGpuProbe()
        else:  # "fake"
            _probe = _fake_probe_from_env()
        _probe_family = family
        log.debug("gpu_probe_created", family=family)
        return _probe


def reset_probe() -> None:
    """Drop the cached probe (tests). Shuts the old one down first."""
    global _probe, _probe_family
    with _probe_lock:
        if _probe is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                _probe.shutdown()
        _probe = None
        _probe_family = None


#: llama.cpp device-name prefix per backend (``--device CUDA0``, ``ROCm0``,
#: ``Vulkan0``, ``MTL0``). The numeric suffix is the index within that
#: backend's own enumeration.
_DEVICE_PREFIX = {"cuda": "CUDA", "rocm": "ROCm", "vulkan": "Vulkan", "metal": "MTL"}


def llama_device_name(backend: str | None, index: int) -> str:
    """The ``--device`` token llama.cpp uses for GPU ``index`` on ``backend``.

    Unknown / fake / absent backends keep the historical ``CUDA`` prefix, so a
    supervisor built without a probe (tests, the command preview) is unchanged.
    """
    return f"{_DEVICE_PREFIX.get(backend or '', 'CUDA')}{index}"


def probe_for_config(config: Any) -> GpuProbe:
    """The process-wide probe for ``config.gpu.backend``.

    ``auto`` defers to ``SF_GPU_PROBE`` (and then to auto-detection); any other
    value is an explicit choice and wins over the environment.
    """
    choice = str(getattr(getattr(config, "gpu", None), "backend", "auto") or "auto")
    return get_probe(force=None if choice == "auto" else choice)


def probe_diagnostics(probe: GpuProbe) -> dict[str, Any]:
    """Which backend a probe is using and why others were skipped (best effort)."""
    getter = getattr(probe, "diagnostics", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            return dict(getter())
    return {"selected": probe.backend, "source": None, "attempts": [], "notes": []}


def system_ram() -> tuple[int, int]:
    """System RAM as ``(total_bytes, used_bytes)`` via psutil."""
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.used)


def vram_processes(probe: GpuProbe, *, own_pids: Sequence[int] = ()) -> list[VramProcess]:
    """VRAM holders from ``probe``, annotated with whether they are ours.

    Goes through ``getattr`` rather than calling the method directly so an
    older or third-party probe object that predates ``compute_processes``
    degrades to "unknown" instead of raising an ``AttributeError`` in the
    middle of a load rejection. Same reason every failure here is swallowed:
    this is attribution for an error message, and it must never become the
    error.
    """
    getter = getattr(probe, "compute_processes", None)
    if getter is None:
        return []
    try:
        found = list(getter())
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("vram_processes_failed", backend=probe.backend, error=str(exc))
        return []
    ours = {int(pid) for pid in own_pids}
    for entry in found:
        if entry.pid in ours:
            entry.is_ours = True
    _fill_missing_used_bytes(found)
    return found


def _fill_missing_used_bytes(entries: list[VramProcess]) -> None:
    """Size the holders NVML could enumerate but not measure.

    Under the Windows WDDM driver model video memory is owned by the OS, so
    NVML reports ``usedGpuMemory`` as 0 for *every* process -- the holder list
    is correct and completely unquantified. That is how, on 2026-08-18,
    ``/api/status`` listed three ``llama-server.exe`` processes at
    ``used_bytes: 0`` while ~25 GiB was unavailable. The Windows performance
    counters do know (it is what Task Manager's "Dedicated GPU memory" column
    reads), so they fill the gap.

    Two deliberate limits. The figure filled in here is the PDH *per-process
    total across adapters*, so a process split over two GPUs shows its full
    total on each of its rows and those rows must not be summed. The per-device
    split does exist (``vram_holders.pdh_process_gpu_bytes``, D39) but it is not
    applied here: this function's contract is one number per NVML row, and
    callers that want placement read ``per_gpu_bytes``/``device_bytes`` from
    ``/api/status`` or ``/api/vram/holders`` instead. And a real NVML number is
    never overwritten: on Linux NVML is authoritative and per-GPU, which is
    strictly better.
    """
    if not entries or all(entry.used_bytes > 0 for entry in entries):
        return
    # Imported lazily: vram_holders imports the supervisor, and this module is
    # imported by everything.
    from studioforge.core.vram_holders import pdh_process_dedicated_bytes

    try:
        per_pid = pdh_process_dedicated_bytes()
    except Exception as exc:  # noqa: BLE001 - pragma: no cover - defensive
        log.debug("pdh_process_bytes_failed", error=str(exc))
        return
    if not per_pid:
        return
    for entry in entries:
        if entry.used_bytes <= 0 and per_pid.get(entry.pid):
            entry.used_bytes = per_pid[entry.pid]


def total_free_vram(probe: GpuProbe) -> int:
    """Sum of free VRAM across every GPU the probe reports, in bytes."""
    return sum(gpu.free_bytes for gpu in probe.list_gpus())


def fastest_gpu_order(gpus: Sequence[GpuInfo]) -> list[int]:
    """GPU indices ranked best-first for the "single fastest GPU" policy.

    Heuristic, not a benchmark: newer architecture (higher compute
    capability) first, then larger total VRAM, then lower index for a stable
    order. Compute capability tracks generational throughput well enough for
    placement (a cc 12.0 RTX 5090 beats a cc 8.6 RTX 3090), and VRAM breaks
    ties toward the card that can also hold more. GPUs with unknown
    capability rank as ``(0, 0)``, i.e. last. On the reference rig
    (2x RTX 5090 + 2x RTX 3090) this yields ``[0, 1, 2, 3]``.
    """

    def key(gpu: GpuInfo) -> tuple[int, int, int, int]:
        major, minor = gpu.compute_capability or (0, 0)
        return (-major, -minor, -gpu.total_bytes, gpu.index)

    return [gpu.index for gpu in sorted(gpus, key=key)]
