"""Supervision of ``llama-server`` child processes.

One loaded model == one supervised child process on a private loopback port.
This module owns everything about that child's life: the exact argv it is
launched with, its port, its log file, its readiness, its crash/restart policy
and -- most importantly -- making sure it is *really* dead when we say it is.

Four rules are load-bearing enough to be stated up front, because getting any
of them wrong fails silently rather than loudly:

* **A child can never outlive this process.** On Windows every child is placed
  in a job object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, so the kernel
  kills it when our last handle closes -- which happens however we die,
  including a ``SIGKILL``/Task Manager kill that never runs ``atexit``. See
  :class:`WindowsChildJob` and DECISIONS.md D23.
* **``--n-gpu-layers`` is always 999.** StudioForge is GPU-only by design; a
  model that does not fit is rejected by the planner, never quietly split onto
  the CPU. There is deliberately no code path here that computes a layer count.
* **``--ctx-size`` is the TOTAL context shared across ``--parallel`` slots**, so
  we pass ``ctx_size * parallel``. Verified against b10425: ``--ctx-size 4096``
  with 4 slots reports ``n_ctx: 4096`` and ``total_slots: 4``, i.e. 1024 tokens
  per conversation. Without the multiplication a user who asks for 8192 gets a
  quarter of it.
* **Speculative decoding needs ``--spec-type draft-simple``.** b10425 renamed
  every drafting flag and defaults ``--spec-type`` to ``none``; the old names
  are *accepted and ignored* ("the argument has been removed"), so a wrong
  spelling here looks like speculative decoding simply not helping.
"""

from __future__ import annotations

import asyncio
import atexit
import builtins
import contextlib
import os
import shlex
import socket
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import httpx
import psutil

from studioforge.config import Config
from studioforge.errors import ModelLoadError, ModelUnloadError
from studioforge.logging import get_logger
from studioforge.types import AdapterRecord, InstanceInfo, LoadPlan, ModelRecord

if TYPE_CHECKING:
    from studioforge.core.gpu import GpuProbe

log = get_logger(__name__)

#: Children always bind loopback only; the gateway is the sole public surface.
CHILD_HOST = "127.0.0.1"

#: GPU-only means *all* layers, unconditionally. Not a tunable.
ALL_GPU_LAYERS = "999"

#: b10425 default for ``--spec-draft-n-max``; emitted explicitly so the GUI's
#: displayed command line matches what the child actually runs.
DEFAULT_SPEC_DRAFT_N_MAX = 16

#: ``--spec-type`` value that enables draft-model speculative decoding.
SPEC_TYPE_DRAFT = "draft-simple"

#: ``--batch-size`` used once a model serves more than four slots. The engine
#: default (2048) is a *shared* logical batch, so many slots ingesting prompts
#: at once queue behind each other. ``--ubatch-size`` stays at its 512 default
#: -- that one is a VRAM term the planner models (planner.DEFAULT_UBATCH).
BATCH_SIZE_MANY_SLOTS = 4096

#: ``--slot-prompt-similarity`` for a multi-slot launch. The 0.10 default makes
#: slot reuse almost accidental; 0.3 keeps an agent's near-identical prompts
#: landing on the slot that already has the prefix cached.
SLOT_PROMPT_SIMILARITY_MULTI = 0.3

#: Lines of child output kept in memory per instance for error reporting.
STDERR_RING_SIZE = 200

#: How many of those lines are quoted in a load failure message.
ERROR_TAIL_LINES = 30

_HTTP_TIMEOUT = 5.0

# Flags that would move work off the GPU. Asserted against in tests; listed
# here so the prohibition is documented in one obvious place.
CPU_OFFLOAD_FLAGS = frozenset({"--cpu-moe", "--n-cpu-moe", "--override-tensor", "-ot", "--cpu"})

# ---------------------------------------------------------------------------
# Interpreter-exit safety net
# ---------------------------------------------------------------------------

_TRACKED_PIDS: set[int] = set()
_ATEXIT_REGISTERED = False


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_kill_tracked_pids)
        _ATEXIT_REGISTERED = True


def _kill_tracked_pids() -> None:
    """Last-resort cleanup: an abrupt exit must not orphan a VRAM holder."""
    for pid in list(_TRACKED_PIDS):
        with contextlib.suppress(Exception):
            kill_process_tree(pid, timeout=2.0, force=True)
    _TRACKED_PIDS.clear()


# ---------------------------------------------------------------------------
# Kernel-enforced child lifetime (Windows job object)
# ---------------------------------------------------------------------------

#: ``CreateProcess`` flag: the child exists but its initial thread is suspended
#: until someone resumes it. Not exported by :mod:`subprocess`, so spelled out.
CREATE_SUSPENDED = 0x00000004

#: ``OpenProcess`` rights needed to move a process into a job object.
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


def _load_win32() -> tuple[Any, Any]:
    """Import the pywin32 job-object bindings.

    A module-level function purely so tests can monkeypatch it to raise
    (simulating a box without pywin32) or to return fakes.
    """
    import win32api
    import win32job

    return win32job, win32api


class WindowsChildJob:
    """A job object whose members die when this process does.

    **The failure this exists to prevent** (DECISIONS.md D23): on 2026-08-18
    three ``llama-server`` children holding ~25 GiB of VRAM were found running
    with "everything stopped". Their parent was a ``pytest`` process, and they
    only exited because that parent exited *cleanly*. The ``atexit`` net above
    is exactly that -- an ``atexit`` net: it does not run for a ``SIGKILL``, a
    Task Manager "End task", a hard power-cycle of the interpreter, or a
    segfault. On a GPU-only server every one of those cases leaks VRAM that
    nothing on the box knows how to attribute.

    A job object moves the guarantee into the kernel.
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` terminates every process still in
    the job the moment its **last handle** closes, and the handle closes when
    this process ends however it ends. The job is anonymous (no name) so two
    supervisors in two processes never share one.

    Nested jobs are legal on Windows 8+, which matters here: the serve process
    routinely already lives inside somebody else's job (the tray launcher, a
    terminal, a CI runner). Where nesting is refused anyway --
    ``ERROR_ACCESS_DENIED`` from ``AssignProcessToJobObject`` -- the failure is
    logged once at WARNING and the load continues. A safety net that refuses to
    be hung must never be the reason a model will not load.
    """

    def __init__(self) -> None:
        win32job, _ = _load_win32()
        self._win32job = win32job
        # Anonymous: a named job would be shared with any other process that
        # guessed the name, and closing it there would kill our children.
        self._handle: Any = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            self._handle, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] |= win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(
            self._handle, win32job.JobObjectExtendedLimitInformation, info
        )
        self._closed = False
        self._warned = False

    @property
    def available(self) -> bool:
        return not self._closed and self._handle is not None

    def assign(self, pid: int) -> bool:
        """Put ``pid`` in the job. Never raises; ``False`` means "unprotected".

        Only the first failure is logged: on a box where nesting is refused,
        every single load would otherwise emit the same warning forever.
        """
        if not self.available:
            return False
        try:
            win32job, win32api = self._win32job, _load_win32()[1]
            handle = win32api.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
            try:
                win32job.AssignProcessToJobObject(self._handle, handle)
            finally:
                with contextlib.suppress(Exception):
                    handle.Close()
        except Exception as exc:  # noqa: BLE001 - the net must not break the load
            if not self._warned:
                self._warned = True
                log.warning(
                    "child_job_assign_failed",
                    pid=pid,
                    error=str(exc),
                    detail=(
                        "llama-server children are not protected by a job object on this "
                        "box; a hard kill of this process would leave them holding VRAM. "
                        "Usually means the process is already in a job that refuses "
                        "nesting (pre-Windows 8, or a job without BREAKAWAY_OK)."
                    ),
                )
            return False
        return True

    def close(self) -> None:
        """Close the job handle, killing anything still in it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            self._handle.Close()
        self._handle = None


def create_child_job() -> WindowsChildJob | None:
    """A job object for this process's children, or ``None`` where impossible.

    ``None`` on every non-Windows platform (POSIX gets the same guarantee from
    process groups plus the ``atexit`` net, and a missing job object is not a
    reason to fail) and on a Windows box without pywin32.
    """
    if os.name != "nt":
        return None
    try:
        return WindowsChildJob()
    except Exception as exc:  # noqa: BLE001 - degrade, never refuse to serve
        log.warning(
            "child_job_unavailable",
            error=str(exc),
            detail=(
                "could not create a Windows job object; llama-server children will "
                "not be killed automatically if this process is hard-killed"
            ),
        )
        return None


def process_create_time(pid: int) -> float | None:
    """Creation timestamp of ``pid``, or ``None`` if it cannot be read.

    Captured at spawn so a later liveness check cannot be fooled by pid reuse
    -- on a busy box the OS can hand our dead child's pid to something else,
    and mistaking that for "the model is still running" would be a false alarm
    in exactly the code whose job is to be trusted.
    """
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, ValueError):  # pragma: no cover - race with exit
        return None


def process_is_alive(pid: int, *, create_time: float | None = None) -> bool:
    """Whether ``pid`` is a live (non-zombie) process, honouring ``create_time``."""
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        if create_time is not None:
            actual = proc.create_time()
            # More than a second apart means the pid was recycled: our process
            # is gone and this is a stranger wearing its number.
            if abs(actual - create_time) > 1.0:
                return False
        return bool(proc.is_running())
    except (psutil.Error, ValueError):
        return False


@dataclass(slots=True)
class UnloadReport:
    """Evidence that an unload actually happened.

    Kept per model and exposed through :meth:`Supervisor.unload_report` so the
    claim "unloaded" is backed by something checkable rather than by the fact
    that a kill call returned.
    """

    model_id: str
    pid: int | None
    pid_gone: bool
    escalated: bool = False
    vram_before_bytes: int = 0
    vram_after_bytes: int = 0
    at: float = 0.0

    @property
    def vram_reclaimed_bytes(self) -> int:
        return max(0, self.vram_before_bytes - self.vram_after_bytes)


def kill_process_tree(pid: int, *, timeout: float = 15.0, force: bool = False) -> None:
    """Terminate ``pid`` and every descendant, escalating to SIGKILL.

    Killing only the direct child is not enough: llama-server can spawn helpers,
    and any survivor keeps its CUDA context -- which on a GPU-only server means
    permanently leaked VRAM and a model that can never be loaded again without a
    reboot. So we enumerate the whole tree with psutil, signal it, wait, then
    hard-kill whatever is left.
    """
    try:
        parent = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return
    try:
        procs: list[psutil.Process] = parent.children(recursive=True)
    except psutil.Error:
        procs = []
    procs.append(parent)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_float(value: float, places: int = 4) -> str:
    """Format a float for the command line: fixed precision, no noise digits."""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def safe_log_name(model_id: str) -> str:
    """Turn a model id into a single filesystem-safe log file name."""
    cleaned = []
    for char in model_id:
        cleaned.append(char if (char.isalnum() or char in "-_.") else "_")
    name = "".join(cleaned).strip("._") or "model"
    return name[:120]


def _port_is_bindable(port: int, host: str = CHILD_HOST) -> bool:
    """True when nothing else currently holds ``port``.

    Bookkeeping alone is not enough -- another process (a stale llama-server, a
    dev server) can own a port inside our range -- so we actually try to bind.
    ``SO_EXCLUSIVEADDRUSE`` is essential on Windows: a listener that set
    ``SO_REUSEADDR`` (llama-server and most servers do) otherwise lets a second
    plain bind succeed, so the probe would report a busy port as free and we
    would end up talking to the wrong process.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            with contextlib.suppress(OSError, AttributeError):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        sock.bind((host, port))
    except OSError:
        return False
    finally:
        sock.close()
    return True


class _Instance:
    """Mutable supervisor-side state for one child process."""

    def __init__(
        self,
        *,
        info: InstanceInfo,
        record: ModelRecord,
        plan: LoadPlan,
        port: int,
        engine_tag: str | None,
        draft: ModelRecord | None,
        adapters: Sequence[tuple[AdapterRecord, float]],
        log_path: Path,
    ) -> None:
        self.info = info
        self.record = record
        self.plan = plan
        self.port = port
        self.engine_tag = engine_tag
        self.draft = draft
        self.adapters = list(adapters)
        self.log_path = log_path
        self.proc: asyncio.subprocess.Process | None = None
        self.wait_task: asyncio.Task[int] | None = None
        self.pumps: list[asyncio.Task[None]] = []
        self.watcher: asyncio.Task[None] | None = None
        self.stderr_ring: deque[str] = deque(maxlen=STDERR_RING_SIZE)
        self.argv: list[str] = []
        # Explicit intent flag: a deliberate stop() must never look like a
        # crash. Guessing from the exit code cannot work -- a terminated
        # process and a crashed one report the same thing on Windows.
        self.stopping = False
        # Alias reported by a *foreign* server found on our port, if any.
        self.port_conflict: str | None = None
        # Captured at spawn so an unload check cannot be fooled by pid reuse.
        self.create_time: float | None = None
        self._log_file: IO[str] | None = None

    # --- logging -------------------------------------------------------

    def open_log(self) -> None:
        if self._log_file is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self.log_path.open("a", encoding="utf-8", errors="replace")

    def write_log(self, line: str) -> None:
        if self._log_file is None:
            return
        with contextlib.suppress(ValueError, OSError):
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def close_log(self) -> None:
        if self._log_file is not None:
            with contextlib.suppress(OSError):
                self._log_file.close()
            self._log_file = None

    def stderr_tail(self, n: int = ERROR_TAIL_LINES) -> list[str]:
        return list(self.stderr_ring)[-n:]


class Supervisor:
    """Owns every ``llama-server`` child process.

    ``resolve_binary`` is injected rather than imported so this module never
    depends on the engine manager's internals: anything that maps an optional
    engine tag to a server binary works, including a test stub.
    """

    def __init__(
        self,
        config: Config,
        *,
        resolve_binary: Callable[[str | None], Path],
        client: httpx.AsyncClient | None = None,
        launch_prefix: Sequence[str] = (),
        probe: GpuProbe | None = None,
    ) -> None:
        self._config = config
        self._resolve_binary = resolve_binary
        # Optional purely so a supervisor can be built without hardware; when
        # present it is used to log the VRAM an unload actually reclaimed.
        self._probe = probe
        self._client = client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._owns_client = client is None
        # Interpreter/wrapper args placed before the engine binary. Real
        # deployments leave this empty; tests use it to run a fake child.
        self._launch_prefix = list(launch_prefix)
        self._instances: dict[str, _Instance] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._ports_in_use: set[int] = set()
        self._unload_reports: dict[str, UnloadReport] = {}
        # Kernel-level "children die with me" net (Windows only). Held for the
        # supervisor's whole life: the guarantee IS the open handle.
        self._job = create_child_job()
        _register_atexit()

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def build_command(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        *,
        port: int,
        engine_tag: str | None = None,
        draft: ModelRecord | None = None,
        adapters: Sequence[tuple[AdapterRecord, float]] = (),
    ) -> list[str]:
        """Build the full argv for one ``llama-server`` child.

        Ordering is deliberate: our own flags first, the user's expert
        ``extra_flags`` last, so a deliberate override actually wins (llama.cpp
        takes the last occurrence of a repeated option).

        See the module docstring for why ``--n-gpu-layers`` is hardcoded, why
        ``--ctx-size`` is multiplied by ``parallel``, and why ``--spec-type`` is
        mandatory for drafting.
        """
        settings = record.settings
        binary = self._resolve_binary(engine_tag or settings.engine_tag)

        argv: list[str] = [
            str(binary),
            "--model",
            str(record.path),
            "--alias",
            record.id,
            "--host",
            CHILD_HOST,
            "--port",
            str(port),
            # GPU-only: never conditional, never computed.
            "--n-gpu-layers",
            ALL_GPU_LAYERS,
            # TOTAL context across all slots -- see module docstring.
            "--ctx-size",
            str(plan.ctx_size * plan.parallel),
            "--parallel",
            str(plan.parallel),
        ]

        argv += self._placement_args(plan)
        argv += ["--cache-type-k", plan.kv_cache_type, "--cache-type-v", plan.kv_cache_type_v]
        # b10425: --flash-attn takes a value (on|off|auto), it is not a switch.
        argv += ["--flash-attn", plan.flash_attn]
        # We proxy everything; the bundled web UI is dead weight, while the
        # introspection endpoints are what the Dashboard and planner feed on.
        argv += ["--no-webui", "--props", "--slots", "--metrics"]
        # b10425 added `--fit` (default ON), which "adjusts unset arguments to fit
        # in device memory". StudioForge's planner already decided placement and
        # context against live free VRAM, so engine-side auto-adjustment is at
        # best redundant and at worst a silent partial-offload path -- exactly the
        # degradation the GPU-only policy exists to prevent. Turn it off and let a
        # genuine over-commit fail loudly instead. Verified accepted by b10425.
        argv += ["--fit", "off"]

        argv += self._optional_args(record)
        argv += self._concurrency_args(record, plan)

        if record.kind == "embedding":
            argv.append("--embedding")
        elif record.mmproj_path is not None:
            # Vision projector. Never for embedding models -- llama-server
            # rejects the combination.
            argv += ["--mmproj", str(record.mmproj_path)]

        for adapter, scale in adapters:
            if abs(scale - 1.0) < 1e-9:
                argv += ["--lora", str(adapter.path)]
            else:
                argv += ["--lora-scaled", str(adapter.path), _fmt_float(scale)]

        if draft is not None:
            argv += self._draft_args(record, plan, draft)

        if settings.extra_flags.strip():
            # posix=False on Windows so backslash paths survive verbatim.
            argv += shlex.split(settings.extra_flags, posix=(os.name != "nt"))

        return argv

    def _placement_args(self, plan: LoadPlan) -> list[str]:
        """Device selection, split mode and main GPU.

        ``--main-gpu`` indexes the *filtered* device list produced by
        ``--device``, not the physical CUDA ordinal: passing ``--device CUDA1
        --main-gpu 1`` makes llama.cpp reject the load with an out-of-range main
        GPU. So the physical ordinal in the plan is translated to its position.
        """
        devices = list(plan.devices) or [0]
        device_arg = ",".join(f"CUDA{i}" for i in devices)
        if len(devices) == 1:
            return ["--device", device_arg, "--main-gpu", "0", "--split-mode", "none"]

        args = ["--device", device_arg]
        if plan.tensor_split:
            args += ["--tensor-split", ",".join(_fmt_float(v) for v in plan.tensor_split)]
        args += ["--split-mode", plan.split_mode]
        main = devices.index(plan.main_gpu) if plan.main_gpu in devices else 0
        args += ["--main-gpu", str(main)]
        return args

    def _optional_args(self, record: ModelRecord) -> list[str]:
        settings = record.settings
        args: list[str] = []

        if settings.batch_size is not None:
            args += ["--batch-size", str(settings.batch_size)]
        if settings.ubatch_size is not None:
            args += ["--ubatch-size", str(settings.ubatch_size)]
        if settings.threads is not None:
            args += ["--threads", str(settings.threads)]
        if settings.threads_batch is not None:
            args += ["--threads-batch", str(settings.threads_batch)]
        if settings.cont_batching:
            args.append("--cont-batching")

        # Prompt-cache reuse is ON by default: OpenClaw re-sends near-identical
        # long agent prompts constantly, and reusing the cached prefix is the
        # single biggest real-world latency win available here.
        cache_reuse = (
            settings.cache_reuse
            if settings.cache_reuse is not None
            else self._config.models.default_cache_reuse
        )
        if cache_reuse and cache_reuse > 0:
            args += ["--cache-reuse", str(cache_reuse)]

        # Reasoning/thinking models: llama.cpp defaults --reasoning-format to
        # 'auto', which routes thoughts into message.reasoning_content and can
        # leave message.content EMPTY. Verified against DeepSeek-R1-0528-Qwen3-8B
        # on b10425: content len 0, reasoning_content len 316. Any OpenAI client
        # (OpenClaw included) then sees an empty reply. 'none' keeps the thoughts
        # inline in content, which also keeps SSE pass-through correct. See
        # DECISIONS.md D12.
        reasoning_format = (
            settings.reasoning_format
            if settings.reasoning_format is not None
            else self._config.models.default_reasoning_format
        )
        if reasoning_format:
            args += ["--reasoning-format", reasoning_format]
        if settings.reasoning is not None:
            args += ["--reasoning", settings.reasoning]
        if settings.reasoning_budget is not None:
            args += ["--reasoning-budget", str(settings.reasoning_budget)]

        # Per-model chat-template override. The GGUF's own template is used by
        # default and is right almost always -- but "almost" is the problem: a
        # baked-in template the engine cannot compile (Jinja `raise_exception`
        # is the known case) turns certain request shapes into a 400 with no
        # way out. This flag is that way out, and it is never set implicitly.
        if settings.chat_template_file is not None:
            args += ["--chat-template-file", str(settings.chat_template_file)]

        if settings.no_context_shift:
            args.append("--no-context-shift")
        # --defrag-thold is deliberately NOT emitted: b10425 marks it
        # deprecated, and passing a deprecated flag to keep a stored setting
        # "working" is how a value quietly stops meaning anything. The field
        # survives on ModelSettings only so old rows and the GUI form keep
        # loading; see types.ModelSettings.defrag_thold.
        if settings.mlock:
            args.append("--mlock")
        if settings.no_mmap:
            args.append("--no-mmap")

        if settings.rope_freq_base is not None:
            args += ["--rope-freq-base", _fmt_float(settings.rope_freq_base)]
        if settings.rope_freq_scale is not None:
            args += ["--rope-freq-scale", _fmt_float(settings.rope_freq_scale)]
        if settings.rope_scaling is not None:
            args += ["--rope-scaling", settings.rope_scaling]

        # Sampler defaults are only emitted when explicitly set; otherwise the
        # engine's own defaults apply and per-request values stay in charge.
        if settings.temperature is not None:
            args += ["--temp", _fmt_float(settings.temperature)]
        if settings.top_p is not None:
            args += ["--top-p", _fmt_float(settings.top_p)]
        if settings.top_k is not None:
            args += ["--top-k", str(settings.top_k)]
        if settings.min_p is not None:
            args += ["--min-p", _fmt_float(settings.min_p)]
        if settings.repeat_penalty is not None:
            args += ["--repeat-penalty", _fmt_float(settings.repeat_penalty)]

        return args

    def _concurrency_args(self, record: ModelRecord, plan: LoadPlan) -> list[str]:
        """Flags that only make sense once a model serves more than one slot.

        Emitted from the *plan*, not from settings, because the slot count can
        be chosen by the planner (DECISIONS.md D17) and these have to follow
        whatever it decided rather than whatever was saved.

        None of them is on at one slot, which is what keeps the default launch
        byte-identical to before the estimator existed.
        """
        settings = record.settings
        slots = max(1, plan.parallel)
        args: list[str] = []

        # The default 2048-token logical batch is shared across slots, so past
        # roughly four of them prompt ingestion starts serialising for no
        # reason. Only when the user has not pinned a batch size: llama.cpp
        # takes the last occurrence of a repeated flag, so appending here would
        # otherwise silently overrule an explicit setting.
        if slots > 4 and settings.batch_size is None:
            args += ["--batch-size", str(BATCH_SIZE_MANY_SLOTS)]

        # Slot affinity. With --cache-reuse on, routing a request to the slot
        # that already holds a similar prefix is what makes the prompt cache
        # pay off; the 0.10 default is loose enough to scatter an agent's
        # near-identical prompts across slots and lose the cache each time.
        if slots > 1:
            args += ["--slot-prompt-similarity", _fmt_float(SLOT_PROMPT_SIMILARITY_MULTI)]

        # One shared KV pool instead of an equal slice per slot. Same VRAM
        # either way. Opt-in per model: llama.cpp enables it only when the slot
        # count is auto, and we always pass an explicit one, so switching it on
        # is a real change to how a long request behaves -- verify with a real
        # load before recommending it (D17).
        if settings.kv_unified:
            args.append("--kv-unified")

        return args

    def _draft_args(self, record: ModelRecord, plan: LoadPlan, draft: ModelRecord) -> list[str]:
        """Speculative decoding flags, b10425 spelling.

        ``--spec-type`` defaults to ``none``; without it the draft model is
        loaded (costing VRAM) and then never used, which is invisible except as
        "speculation did not help". The old ``--draft*`` names are accepted and
        ignored by b10425, so they must never appear here.
        """
        settings = record.settings
        args: list[str] = [
            "--spec-type",
            settings.spec_type or SPEC_TYPE_DRAFT,
            "--spec-draft-model",
            str(draft.path),
            "--spec-draft-n-max",
            str(
                settings.spec_draft_n_max
                if settings.spec_draft_n_max is not None
                else DEFAULT_SPEC_DRAFT_N_MAX
            ),
        ]
        if settings.spec_draft_n_min is not None:
            args += ["--spec-draft-n-min", str(settings.spec_draft_n_min)]
        if settings.spec_draft_p_min is not None:
            args += ["--spec-draft-p-min", _fmt_float(settings.spec_draft_p_min)]
        # The draft model is GPU-only too; a CPU-resident draft is slower than
        # not drafting at all.
        args += ["--spec-draft-ngl", ALL_GPU_LAYERS]
        draft_devices = settings.draft_device_override or plan.devices or [0]
        args += ["--spec-draft-device", ",".join(f"CUDA{i}" for i in draft_devices)]
        # "auto" is OUR sentinel, resolved by the planner; llama-server has no
        # such value and would refuse to start. The draft cache type is not
        # planned, so pass it through only when it is a real type.
        if draft.settings.kv_cache_type not in (None, "auto"):
            args += ["--spec-draft-type-k", str(draft.settings.kv_cache_type)]
        if draft.settings.kv_cache_type_v not in (None, "auto"):
            args += ["--spec-draft-type-v", str(draft.settings.kv_cache_type_v)]
        return args

    # ------------------------------------------------------------------
    # Ports
    # ------------------------------------------------------------------

    def _allocate_port(self, preferred: int | None = None) -> int:
        gateway = self._config.gateway
        candidates: Iterable[int] = range(gateway.child_port_start, gateway.child_port_end + 1)
        if preferred is not None:
            candidates = [preferred, *candidates]
        for port in candidates:
            if port in self._ports_in_use:
                continue
            if not _port_is_bindable(port):
                continue
            self._ports_in_use.add(port)
            return port
        raise ModelLoadError(
            "No free port in the llama-server child range "
            f"{gateway.child_port_start}-{gateway.child_port_end}; "
            "widen gateway.child_port_start/child_port_end or unload models."
        )

    def _release_port(self, port: int | None) -> None:
        if port is not None:
            self._ports_in_use.discard(port)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _lock(self, model_id: str) -> asyncio.Lock:
        lock = self._locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[model_id] = lock
        return lock

    def _prune_lock(self, model_id: str) -> None:
        """Drop the per-model lock once its instance is gone and nobody holds it.

        Keeps the lock table from growing one entry per model id ever started.
        A waiter woken between our ``locked()`` check and its own resume would
        find its lock discarded and a later caller on a fresh one -- benign
        here, because every operation re-checks ``self._instances`` after
        acquiring, so two locks can never produce two children for one id.
        """
        lock = self._locks.get(model_id)
        if lock is not None and not lock.locked() and model_id not in self._instances:
            del self._locks[model_id]

    async def start(
        self,
        record: ModelRecord,
        plan: LoadPlan,
        *,
        engine_tag: str | None = None,
        draft: ModelRecord | None = None,
        adapters: Sequence[tuple[AdapterRecord, float]] = (),
    ) -> InstanceInfo:
        """Launch (or return) the child for ``record``; returns once /health is ok."""
        async with self._lock(record.id):
            existing = self._instances.get(record.id)
            if existing is not None and existing.info.state in ("ready", "loading"):
                # A concurrent caller already started this model.
                return existing.info

            tag = engine_tag or record.settings.engine_tag
            port = self._allocate_port()
            info = InstanceInfo(
                model_id=record.id,
                state="loading",
                port=port,
                engine_tag=tag,
                plan=plan,
                ttl_s=record.settings.ttl_s,
                log_path=self._log_path_for(record.id),
            )
            inst = _Instance(
                info=info,
                record=record,
                plan=plan,
                port=port,
                engine_tag=tag,
                draft=draft,
                adapters=adapters,
                log_path=self._log_path_for(record.id),
            )
            self._instances[record.id] = inst
            try:
                await self._spawn(inst)
                await self._await_ready(inst)
            except BaseException as exc:
                inst.stopping = True
                inst.info.state = "failed"
                if isinstance(exc, ModelLoadError):
                    inst.info.last_error = exc.message
                await self._teardown(inst, timeout=5.0, force=True)
                # Only live children stay in the table; the diagnostics travel
                # with the raised ModelLoadError (stderr tail + argv).
                self._instances.pop(record.id, None)
                self._release_port(port)
                raise

            inst.info.state = "ready"
            inst.info.started_at = time.time()
            inst.info.last_activity_at = time.time()
            inst.watcher = asyncio.create_task(self._watch(inst), name=f"sf-watch-{record.id}")
            log.info(
                "model_ready", model_id=record.id, port=port, pid=inst.info.pid, engine_tag=tag
            )
            return inst.info

    async def _spawn(self, inst: _Instance) -> None:
        """Create the child process and start pumping its output."""
        argv = self.build_command(
            inst.record,
            inst.plan,
            port=inst.port,
            engine_tag=inst.engine_tag,
            draft=inst.draft,
            adapters=inst.adapters,
        )
        inst.argv = argv
        full_argv = [*self._launch_prefix, *argv]
        inst.open_log()
        inst.write_log(f"=== studioforge launch: {' '.join(full_argv)}")

        kwargs: dict[str, Any] = {}
        # Start suspended when there is a job to put the child in, so the window
        # between "process exists" and "process is in the job" contains no
        # executed instruction at all. Assigning right after Popen would leave a
        # sub-millisecond gap in which a hard kill of this process could strand a
        # child -- vanishingly unlikely, but the gap is avoidable and this is the
        # code whose entire job is to make the guarantee unconditional. A child
        # that is never resumed has allocated nothing, so the worst case of this
        # path is strictly better than the worst case of the other one.
        suspended = os.name == "nt" and self._job is not None and self._job.available
        if os.name == "nt":
            # New process group so a console Ctrl+C does not race our own
            # orderly shutdown, and so the whole group can be signalled.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            if suspended:
                kwargs["creationflags"] |= CREATE_SUSPENDED
        else:
            kwargs["start_new_session"] = True

        # Run from the engine directory: the CUDA/ggml DLLs (or .so files) ship
        # alongside llama-server and are found relative to it.
        engine_dir = Path(argv[0]).parent
        log.info("model_spawn", model_id=inst.record.id, port=inst.port, argv=" ".join(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(engine_dir) if engine_dir.is_dir() else None,
                **kwargs,
            )
        except OSError as exc:
            raise ModelLoadError(
                f"Could not launch llama-server for '{inst.record.id}': {exc}",
                details={"argv": argv},
            ) from exc

        if self._job is not None:
            # Best-effort by design: assign() logs and returns False rather than
            # raising, because an unprotected child still serves inference.
            self._job.assign(proc.pid)
        if suspended:
            self._resume(inst, proc)

        inst.proc = proc
        inst.info.pid = proc.pid
        inst.info.state = "loading"
        inst.create_time = process_create_time(proc.pid)
        _TRACKED_PIDS.add(proc.pid)
        inst.pumps = []
        if proc.stdout is not None:
            inst.pumps.append(asyncio.create_task(self._pump(inst, proc.stdout, stderr=False)))
        if proc.stderr is not None:
            inst.pumps.append(asyncio.create_task(self._pump(inst, proc.stderr, stderr=True)))
        inst.wait_task = asyncio.create_task(proc.wait(), name=f"sf-wait-{inst.record.id}")

    @staticmethod
    def _resume(inst: _Instance, proc: asyncio.subprocess.Process) -> None:
        """Start the suspended child, or kill it and fail the load loudly.

        The one new failure mode ``CREATE_SUSPENDED`` introduces is a child that
        never runs. It must not be survivable-but-invisible: a suspended child
        holds a port and answers nothing, so the load would fail later as an
        unexplained health timeout. Kill it here and say why.
        """
        try:
            psutil.Process(proc.pid).resume()
        except Exception as exc:  # noqa: BLE001 - reported as a load failure below
            with contextlib.suppress(Exception):
                kill_process_tree(proc.pid, timeout=2.0, force=True)
            raise ModelLoadError(
                f"llama-server for '{inst.record.id}' was created suspended (so it could "
                f"be placed in this process's job object) and could not be resumed: {exc}",
                details={"pid": proc.pid, "argv": inst.argv},
            ) from exc

    async def _pump(self, inst: _Instance, stream: asyncio.StreamReader, *, stderr: bool) -> None:
        while True:
            try:
                raw = await stream.readline()
            except (asyncio.LimitOverrunError, ValueError):
                continue
            except Exception:  # pragma: no cover - transport teardown races
                return
            if not raw:
                return
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if stderr:
                inst.stderr_ring.append(line)
            inst.write_log(line)

    async def _await_ready(self, inst: _Instance) -> None:
        """Poll ``GET /health`` until ok, the process dies, or we time out."""
        gateway = self._config.gateway
        deadline = time.monotonic() + gateway.load_timeout_s
        interval = max(0.05, gateway.health_poll_interval_s)
        url = f"http://{CHILD_HOST}:{inst.port}/health"

        while True:
            code = inst.proc.returncode if inst.proc is not None else None
            if code is not None:
                # Fail fast: waiting out a 10 minute timeout for a process that
                # already exited buries the actual error.
                await self._drain_pumps(inst)
                raise ModelLoadError(
                    self._failure_message(inst, f"exited with code {code} during startup"),
                    details={
                        "exit_code": code,
                        "stderr": inst.stderr_tail(),
                        "argv": inst.argv,
                    },
                )
            try:
                resp = await self._client.get(url, timeout=_HTTP_TIMEOUT)
                if resp.status_code == 200 and await self._confirm_identity(inst):
                    return
            except (httpx.HTTPError, OSError):
                pass

            if time.monotonic() >= deadline:
                raise ModelLoadError(
                    self._failure_message(
                        inst, f"did not become healthy within {gateway.load_timeout_s:g}s"
                    ),
                    details={
                        "stderr": inst.stderr_tail(),
                        "argv": inst.argv,
                        "port_conflict": inst.port_conflict,
                    },
                )
            await asyncio.sleep(interval)

    @staticmethod
    def _expected_alias(inst: _Instance) -> str:
        """The alias we actually launched with (extra_flags may override ours)."""
        alias = inst.record.id
        for index, token in enumerate(inst.argv[:-1]):
            if token == "--alias":
                alias = inst.argv[index + 1]
        return alias

    async def _confirm_identity(self, inst: _Instance) -> bool:
        """Check that the healthy server on our port is really *our* child.

        A stale llama-server (or anything else) squatting on the port answers
        ``/health`` perfectly happily, and adopting it would mean proxying
        requests to a process we do not control, with the wrong model and the
        wrong context size -- a failure that looks like a mystery bug rather
        than a port clash. ``--alias`` is echoed by ``/props``, so it doubles as
        an identity check. When ``/props`` cannot be read we do not block the
        load; the goal is catching an impostor, not adding a hard dependency.
        """
        base = f"http://{CHILD_HOST}:{inst.port}"
        try:
            resp = await self._client.get(f"{base}/props", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return True
        if resp.status_code != 200:
            return True
        try:
            data = resp.json()
        except ValueError:
            return True
        if not isinstance(data, dict):
            return True
        alias = data.get("model_alias")
        expected = self._expected_alias(inst)
        if isinstance(alias, str) and alias != expected:
            inst.port_conflict = alias
            log.error(
                "child_port_conflict",
                model_id=inst.record.id,
                port=inst.port,
                foreign_alias=alias,
            )
            return False
        inst.port_conflict = None
        return True

    def _failure_message(self, inst: _Instance, what: str) -> str:
        tail = inst.stderr_tail()
        text = f"llama-server for '{inst.record.id}' {what}."
        if inst.port_conflict is not None:
            text += (
                f" Port {inst.port} is held by another server "
                f"(alias '{inst.port_conflict}'), not by this child."
            )
        if tail:
            text += " Last output:\n" + "\n".join(tail)
        else:
            text += f" No output captured; see {inst.log_path}."
        return text

    async def _drain_pumps(
        self,
        inst: _Instance,
        timeout: float = 2.0,  # noqa: ASYNC109 - drain deadline
    ) -> None:
        if not inst.pumps:
            return
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*inst.pumps, return_exceptions=True), timeout=timeout
            )

    async def _watch(self, inst: _Instance) -> None:
        """Restart the child on unexpected exit, with exponential backoff."""
        gateway = self._config.gateway
        while True:
            if inst.stopping or inst.wait_task is None:
                return
            # Shielded so cancelling the watcher (a deliberate stop) does not
            # cancel the underlying wait() that stop() itself needs to observe.
            code = await asyncio.shield(inst.wait_task)
            if inst.stopping:
                return

            await self._drain_pumps(inst)
            inst.info.last_error = self._failure_message(inst, f"exited with code {code}")
            _TRACKED_PIDS.discard(inst.info.pid or -1)
            log.warning(
                "model_exited",
                model_id=inst.record.id,
                exit_code=code,
                restarts=inst.info.restarts,
            )

            if inst.info.restarts >= gateway.max_restarts:
                inst.info.state = "failed"
                log.error(
                    "model_restart_limit",
                    model_id=inst.record.id,
                    max_restarts=gateway.max_restarts,
                )
                return

            attempt = inst.info.restarts
            inst.info.restarts += 1
            inst.info.state = "loading"
            delay = gateway.restart_backoff_s * (2**attempt)
            await asyncio.sleep(delay)
            if inst.stopping:
                return

            # Reuse the same port when it is still ours and still free, so
            # anything caching the base URL keeps working across a restart.
            self._release_port(inst.port)
            try:
                inst.port = self._allocate_port(preferred=inst.port)
            except ModelLoadError as exc:
                inst.info.state = "failed"
                inst.info.last_error = exc.message
                return
            inst.info.port = inst.port

            try:
                await self._spawn(inst)
                await self._await_ready(inst)
            except ModelLoadError as exc:
                inst.info.last_error = exc.message
                # A failed relaunch can leave a live child behind: _await_ready
                # times out on a hung load with the process still running.
                # start() tears that case down; without the same here the hung
                # llama-server would outlive its watcher, silently keeping its
                # port and -- far worse -- its VRAM. Teardown is a no-op for a
                # child that already exited.
                await self._teardown(inst, timeout=5.0, force=True)
                if inst.info.restarts >= gateway.max_restarts:
                    inst.info.state = "failed"
                    return
                continue

            inst.info.state = "ready"
            inst.info.started_at = time.time()
            log.info(
                "model_restarted",
                model_id=inst.record.id,
                port=inst.port,
                restarts=inst.info.restarts,
            )

    async def _teardown(
        self,
        inst: _Instance,
        *,
        timeout: float,  # noqa: ASYNC109 - psutil.wait_procs grace period
        force: bool,
    ) -> None:
        """Cancel supervision tasks and make sure the process tree is gone."""
        if inst.watcher is not None and inst.watcher is not asyncio.current_task():
            inst.watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await inst.watcher
            inst.watcher = None

        proc = inst.proc
        if proc is not None and proc.returncode is None:
            await asyncio.to_thread(kill_process_tree, proc.pid, timeout=timeout, force=force)
        if inst.wait_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(asyncio.shield(inst.wait_task), timeout=5.0)
        await self._drain_pumps(inst)
        for pump in inst.pumps:
            pump.cancel()
        inst.pumps = []
        if inst.info.pid is not None:
            _TRACKED_PIDS.discard(inst.info.pid)
        inst.close_log()

    def _vram_used_bytes(self, devices: Sequence[int]) -> int:
        """Used VRAM across ``devices`` right now, or 0 without a probe."""
        if self._probe is None:
            return 0
        try:
            gpus = {gpu.index: gpu for gpu in self._probe.list_gpus()}
        except Exception:  # pragma: no cover - a probe must not break an unload
            return 0
        wanted = list(devices) or sorted(gpus)
        return sum(gpus[d].used_bytes for d in wanted if d in gpus)

    async def _verify_unloaded(self, inst: _Instance, pid: int | None, before: int) -> UnloadReport:
        """Confirm the child is really gone, and log the VRAM it gave back.

        A kill call that returned is not proof: the whole reason this project
        supervises processes itself is that an unload which *reports* success
        while the model stays resident leaves VRAM permanently spoken for. So
        the pid is re-checked, a survivor is escalated to an unconditional
        tree-kill, and the before/after VRAM numbers are recorded so "unloaded
        13.4 GiB" can be verified instead of assumed.
        """
        escalated = False
        alive = pid is not None and await asyncio.to_thread(
            process_is_alive, pid, create_time=inst.create_time
        )
        if alive and pid is not None:
            escalated = True
            log.warning(
                "unload_survivor",
                model_id=inst.record.id,
                pid=pid,
                detail="process still alive after teardown; escalating to a forced tree kill",
            )
            await asyncio.to_thread(kill_process_tree, pid, timeout=5.0, force=True)
            alive = await asyncio.to_thread(
                process_is_alive, pid, create_time=inst.create_time
            )

        report = UnloadReport(
            model_id=inst.record.id,
            pid=pid,
            pid_gone=not alive,
            escalated=escalated,
            vram_before_bytes=before,
            vram_after_bytes=self._vram_used_bytes(inst.plan.devices),
            at=time.time(),
        )
        self._unload_reports[inst.record.id] = report
        if report.pid_gone:
            log.info(
                "model_unload_verified",
                model_id=inst.record.id,
                pid=pid,
                escalated=escalated,
                vram_reclaimed_mb=round(report.vram_reclaimed_bytes / (1024 * 1024)),
            )
        else:
            log.error(
                "model_unload_unverified",
                model_id=inst.record.id,
                pid=pid,
                detail="process survived SIGTERM and SIGKILL; its VRAM is still held",
            )
        return report

    def unload_report(self, model_id: str) -> UnloadReport | None:
        """Evidence from the last unload of ``model_id``, if there was one."""
        return self._unload_reports.get(model_id)

    async def stop(
        self,
        model_id: str,
        *,
        timeout: float = 15.0,  # noqa: ASYNC109 - SIGTERM grace
    ) -> None:
        """Terminate the child gracefully, and verify that it actually died.

        Raises :class:`~studioforge.errors.ModelUnloadError` when the process
        outlives both signals: returning normally there would report freed VRAM
        that is still held, and every subsequent plan would be computed against
        a lie.
        """
        async with self._lock(model_id):
            inst = self._instances.get(model_id)
            if inst is not None:
                inst.stopping = True
                inst.info.state = "unloading"
                pid = inst.info.pid
                before = self._vram_used_bytes(inst.plan.devices)
                await self._teardown(inst, timeout=timeout, force=False)
                report = await self._verify_unloaded(inst, pid, before)
                if not report.pid_gone:
                    # Keep the instance in the table: it is still real, still
                    # holding VRAM, and hiding it would make the leak invisible.
                    inst.info.state = "failed"
                    inst.info.last_error = (
                        f"unload could not be verified: pid {pid} is still running"
                    )
                    raise ModelUnloadError(
                        f"Unloaded '{model_id}' but its llama-server process (pid {pid}) is "
                        "still alive, so its VRAM has not been reclaimed. Kill it manually "
                        "before loading anything else.",
                        details={"pid": pid, "model_id": model_id},
                    )
                inst.info.state = "stopped"
                inst.info.pid = None
                self._instances.pop(model_id, None)
                self._release_port(inst.port)
                log.info(
                    "model_stopped",
                    model_id=model_id,
                    vram_reclaimed_mb=round(report.vram_reclaimed_bytes / (1024 * 1024)),
                )
        self._prune_lock(model_id)

    async def stop_all(
        self,
        *,
        timeout: float = 15.0,  # noqa: ASYNC109 - per-child SIGTERM grace
    ) -> None:
        await asyncio.gather(
            *(self.stop(model_id, timeout=timeout) for model_id in list(self._instances)),
            return_exceptions=True,
        )

    async def kill(self, model_id: str) -> bool:
        """Hard-kill the child immediately, without draining requests.

        Returns True only when the process is *verified* gone -- a survivor
        reports False and stays in the instance table rather than being
        forgotten while it still holds VRAM.
        """
        killed = False
        async with self._lock(model_id):
            inst = self._instances.get(model_id)
            if inst is not None:
                inst.stopping = True
                inst.info.state = "unloading"
                pid = inst.info.pid
                before = self._vram_used_bytes(inst.plan.devices)
                await self._teardown(inst, timeout=0.0, force=True)
                report = await self._verify_unloaded(inst, pid, before)
                if not report.pid_gone:
                    inst.info.state = "failed"
                    inst.info.last_error = (
                        f"kill could not be verified: pid {pid} is still running"
                    )
                else:
                    inst.info.state = "stopped"
                    inst.info.pid = None
                    self._instances.pop(model_id, None)
                    self._release_port(inst.port)
                    log.warning("model_killed", model_id=model_id)
                    killed = True
        self._prune_lock(model_id)
        return killed

    async def aclose(self) -> None:
        await self.stop_all(timeout=5.0)
        if self._owns_client:
            await self._client.aclose()
        # Closing the job kills anything still in it. Deliberately last, and
        # deliberately after stop_all: by here every child should already be
        # gone, so this only catches a survivor -- which is the point.
        if self._job is not None:
            self._job.close()

    def child_pids(self) -> set[int]:
        """Pids of the children this supervisor currently owns.

        Used by the orphan sweep to tell "ours" from "someone else's": a
        llama-server under our engines dir that this set does not contain is
        either another live process's child or a leak. See
        :mod:`studioforge.core.vram_holders`.
        """
        return {
            inst.info.pid for inst in self._instances.values() if inst.info.pid is not None
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, model_id: str) -> InstanceInfo | None:
        inst = self._instances.get(model_id)
        return inst.info if inst is not None else None

    def list(self) -> builtins.list[InstanceInfo]:
        return [inst.info for inst in self._instances.values()]

    def is_ready(self, model_id: str) -> bool:
        inst = self._instances.get(model_id)
        return inst is not None and inst.info.state == "ready"

    def base_url(self, model_id: str) -> str | None:
        inst = self._instances.get(model_id)
        if inst is None or inst.port is None:
            return None
        return f"http://{CHILD_HOST}:{inst.port}"

    def _log_path_for(self, model_id: str) -> Path:
        return self._config.model_logs_dir / f"{safe_log_name(model_id)}.log"

    def log_path(self, model_id: str) -> Path | None:
        inst = self._instances.get(model_id)
        if inst is not None:
            return inst.log_path
        path = self._log_path_for(model_id)
        return path if path.is_file() else None

    def tail_log(self, model_id: str, n: int = 200) -> builtins.list[str]:
        path = self.log_path(model_id)
        if path is None or not path.is_file():
            return []
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return [line.rstrip("\n") for line in deque(handle, maxlen=n)]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Child HTTP proxies (never raise: the Dashboard polls these constantly)
    # ------------------------------------------------------------------

    async def _get_json(self, model_id: str, path: str) -> Any:
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return None
        try:
            resp = await self._client.get(f"{base}{path}", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    async def props(self, model_id: str) -> dict[str, Any] | None:
        data = await self._get_json(model_id, "/props")
        return data if isinstance(data, dict) else None

    async def slots(self, model_id: str) -> builtins.list[dict[str, Any]] | None:
        data = await self._get_json(model_id, "/slots")
        return data if isinstance(data, list) else None

    async def metrics(self, model_id: str) -> str | None:
        """Raw Prometheus text from the child's ``/metrics``, or ``None``.

        Enabled by the ``--metrics`` flag every child is launched with. Returns
        text rather than a parsed structure because parsing belongs to
        :func:`studioforge.core.throughput.parse_metrics`, which is unit-tested
        against real exposition output; the supervisor's job here is only to
        fetch it without ever raising -- this is called from a background timer
        that must not die because a child is mid-restart.
        """
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return None
        try:
            resp = await self._client.get(f"{base}/metrics", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return None
        if resp.status_code != 200:
            return None
        return resp.text

    async def health(self, model_id: str) -> bool:
        base = self.base_url(model_id)
        if base is None:
            return False
        try:
            resp = await self._client.get(f"{base}/health", timeout=_HTTP_TIMEOUT)
        except (httpx.HTTPError, OSError):
            return False
        return resp.status_code == 200

    async def set_lora_scales(self, model_id: str, scales: builtins.list[dict[str, Any]]) -> bool:
        """Hot-adjust LoRA scales via ``POST /lora-adapters``."""
        base = self.base_url(model_id)
        if base is None or not self.is_ready(model_id):
            return False
        try:
            resp = await self._client.post(
                f"{base}/lora-adapters", json=scales, timeout=_HTTP_TIMEOUT
            )
        except (httpx.HTTPError, OSError):
            return False
        return resp.status_code == 200

    # ------------------------------------------------------------------
    # Activity accounting (feeds TTL unloading and the Dashboard)
    # ------------------------------------------------------------------

    def mark_request_start(self, model_id: str) -> None:
        inst = self._instances.get(model_id)
        if inst is None:
            return
        inst.info.active_requests += 1
        inst.info.total_requests += 1
        inst.info.last_activity_at = time.time()

    def mark_request_end(self, model_id: str, *, tokens_per_second: float | None = None) -> None:
        inst = self._instances.get(model_id)
        if inst is None:
            return
        inst.info.active_requests = max(0, inst.info.active_requests - 1)
        inst.info.last_activity_at = time.time()
        if tokens_per_second is not None:
            inst.info.last_tokens_per_second = tokens_per_second
