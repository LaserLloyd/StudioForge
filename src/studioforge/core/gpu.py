"""GPU hardware probe behind a swappable interface.

The planner and the engine selector both need GPU facts (free VRAM, compute
capability, driver's CUDA level). Everything flows through :class:`GpuProbe`
so that a ROCm/Vulkan backend can be added later and so tests never touch
real hardware. CUDA/NVML (:class:`NvmlGpuProbe`) is the only real
implementation today; :class:`NullGpuProbe` stands in on GPU-less boxes and
:class:`FakeGpuProbe` serves tests and synthetic-hardware integration runs.

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
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import psutil

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
        """Backend identifier: ``"cuda"``, ``"null"``, or ``"fake"``."""
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
    def _running_processes(
        nvml: Any, handle: Any, index: int, names: Sequence[str]
    ) -> list[Any]:
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
# Module-level probe factory + helpers
# ---------------------------------------------------------------------------

_probe: GpuProbe | None = None
_probe_lock = threading.Lock()

_BACKEND_FOR_CHOICE = {"nvml": "cuda", "cuda": "cuda", "null": "null", "fake": "fake"}


def get_probe(*, force: str | None = None) -> GpuProbe:
    """The process-wide probe, created on first call and cached.

    ``force`` (or env ``SF_GPU_PROBE``) selects ``nvml``/``cuda``, ``null``,
    or ``fake``; ``fake`` reads its synthetic GPUs from env ``SF_FAKE_GPUS``.
    A force that differs from the cached probe's backend rebuilds the cache.
    """
    global _probe
    choice = (force or os.environ.get(_ENV_PROBE, "")).strip().lower() or None
    if choice is not None and choice not in _BACKEND_FOR_CHOICE:
        raise ConfigError(
            f"Unknown GPU probe backend {choice!r}; expected one of "
            f"{sorted(set(_BACKEND_FOR_CHOICE))}"
        )
    with _probe_lock:
        if _probe is not None and (choice is None or _probe.backend == _BACKEND_FOR_CHOICE[choice]):
            return _probe
        if choice in ("nvml", "cuda", None):
            _probe = NvmlGpuProbe()
        elif choice == "null":
            _probe = NullGpuProbe()
        else:  # "fake"
            _probe = _fake_probe_from_env()
        log.debug("gpu_probe_created", backend=_probe.backend)
        return _probe


def reset_probe() -> None:
    """Drop the cached probe (tests). Shuts the old one down first."""
    global _probe
    with _probe_lock:
        if _probe is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                _probe.shutdown()
        _probe = None


def system_ram() -> tuple[int, int]:
    """System RAM as ``(total_bytes, used_bytes)`` via psutil."""
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.used)


def vram_processes(
    probe: GpuProbe, *, own_pids: Sequence[int] = ()
) -> list[VramProcess]:
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

    Two deliberate limits. The PDH figure is a per-process total across
    adapters -- there is no sound instance-to-CUDA-ordinal mapping, since NVML
    exposes no adapter LUID -- so a process split over two GPUs shows its full
    total on each of its rows and those rows must not be summed. And a real
    NVML number is never overwritten: on Linux NVML is authoritative and
    per-GPU, which is strictly better.
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
