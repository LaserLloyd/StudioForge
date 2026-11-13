"""A deterministic stand-in for ``llama-server`` (D85).

Two users share it: the unit suite (``tests/unit/test_supervisor.py`` and its
neighbours launch it as the child) and a **no-GPU staging instance**
(``engine.stub: true``, see ``docs/DEVELOPMENT.md``), where every load starts
one of these instead of the real engine, so every API and MCP flow can be
exercised without a GPU being touched.

It is standard-library only on purpose (it runs as a child process, from a
path, with no ``studioforge`` import needed), parses the real ``llama-server``
argv (unknown flags are ignored, as the real one would reject them only at
startup and the supervisor never sends one the help does not list), and
answers deterministically: the text of a completion is a pure function of the
prompt and ``seed``.

Run by hand::

    python stub_llama_server.py --model m.gguf --alias m --port 19420 --ctx-size 8192 --parallel 2

Tuning, as flags or environment (the flag wins):

* ``--stub-load-ms`` / ``SF_STUB_LOAD_MS``: ``/health`` answers 503 for this
  long after start (default 0).
* ``--stub-tps`` / ``SF_STUB_TPS``: generation speed in tokens per second,
  which paces the response so busy slots and tok/s are visible (default 0 =
  unpaced; the reported ``timings`` then use a fixed synthetic speed).
* ``--stub-vram-mib``: how much VRAM the child "uses" (the supervisor's fake
  ledger reads it from the argv; ``/props`` echoes it).
* ``--stub-build N`` (default 0), ``--stub-embed-dim N`` (default 64).

Fault flags kept from the old in-test fake: ``--fake-exit-code N``,
``--fake-unhealthy``, ``--fake-unhealthy-if PATH``, ``--fake-crash-after S``,
``--fake-crash-once PATH``, ``--fake-touch-on-crash PATH``,
``--fake-no-listen``, ``--fake-ignore-sigterm``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

STUB_VERSION = 1

# ---------------------------------------------------------------------------
# The CLI surface: --help text the engine feature detection parses
# ---------------------------------------------------------------------------

#: ``(flags, placeholder, description)``: laid out like llama-server's own help
#: so :func:`studioforge.core.engine.parse_engine_features` reads it the way it
#: reads the real thing (enum values in braces / brackets, ``(default: ...)``).
_FLAG_TABLE: tuple[tuple[str, str, str], ...] = (
    ("-h, --help, --usage", "", "print usage and exit"),
    ("--version", "", "show version and build info"),
    ("--list-devices", "", "print list of available devices and exit"),
    ("-m, --model", "FNAME", "model path"),
    ("-a, --alias", "STRING", "set alias for model name (to be used by REST API)"),
    ("-c, --ctx-size", "N", "size of the prompt context (default: 0, 0 = loaded from model)"),
    ("-np, --parallel", "N", "number of server slots (default: -1, -1 = auto)"),
    ("-b, --batch-size", "N", "logical maximum batch size (default: 2048)"),
    ("-ub, --ubatch-size", "N", "physical maximum batch size (default: 512)"),
    ("-ngl, --gpu-layers, --n-gpu-layers", "N", "max. number of layers to store in VRAM"),
    ("-sm, --split-mode", "{none,layer,row,tensor}", "how to split the model across multiple GPUs"),
    ("-ts, --tensor-split", "N0,N1,N2,...", "fraction of the model to offload to each GPU"),
    ("-mg, --main-gpu", "INDEX", "the GPU to use for the model"),
    ("-dev, --device", "<dev1,dev2,..>", "comma-separated list of devices to use for offloading"),
    ("-ctk, --cache-type-k", "TYPE", "KV cache data type for K (default: f16)"),
    ("-ctv, --cache-type-v", "TYPE", "KV cache data type for V (default: f16)"),
    ("-fa, --flash-attn", "[on|off|auto]", "set Flash Attention use (default: auto)"),
    ("-fit, --fit", "[on|off]", "adjust unset arguments to fit in device memory (default: on)"),
    ("-kvu, --kv-unified", "", "use single unified KV buffer for the KV cache of all sequences"),
    ("-no-kvu, --no-kv-unified", "", "do not use a unified KV buffer (default: disabled)"),
    ("-kvo, --kv-offload", "", "enable KV cache offloading (default: enabled)"),
    ("-nkvo, --no-kv-offload", "", "disable KV cache offloading"),
    ("--mmap", "", "memory-map the model (default: enabled)"),
    ("--no-mmap", "", "do not memory-map the model"),
    ("--mlock", "", "force the system to keep the model in RAM"),
    ("-t, --threads", "N", "number of CPU threads to use during generation"),
    ("-tb, --threads-batch", "N", "number of threads to use during batch and prompt processing"),
    ("--rope-scaling", "{none,linear,yarn}", "RoPE frequency scaling method"),
    ("--rope-freq-base", "N", "RoPE base frequency"),
    ("--rope-freq-scale", "N", "RoPE frequency scaling factor"),
    ("--lora", "FNAME", "path to LoRA adapter"),
    ("--lora-scaled", "FNAME SCALE", "path to LoRA adapter with user defined scaling"),
    ("--mmproj", "FILE", "path to a multimodal projector file"),
    ("-mmdev, --mmproj-device", "<dev>", "device to use for the multimodal projector"),
    ("-devd, --device-draft", "<dev1,dev2,..>", "devices to use for offloading the draft model"),
    (
        "-ngld, --gpu-layers-draft, --n-gpu-layers-draft",
        "N",
        "max. number of draft model layers to store in VRAM",
    ),
    ("--mmproj-offload", "", "enable GPU offloading for multimodal projector (default: enabled)"),
    ("--no-mmproj-offload", "", "do not offload multimodal projector to GPU"),
    ("--temp", "N", "sampling temperature (default: 0.8)"),
    ("--top-k", "N", "top-k sampling (default: 40)"),
    ("--top-p", "N", "top-p sampling (default: 0.95)"),
    ("--min-p", "N", "min-p sampling (default: 0.05)"),
    ("--repeat-penalty", "N", "penalize repeat sequence of tokens (default: 1.00)"),
    ("--host", "HOST", "ip address to listen on (default: 127.0.0.1)"),
    ("--port", "PORT", "port to listen on (default: 8080)"),
    ("--api-key", "KEY", "API key to use for authentication"),
    ("--api-key-file", "FNAME", "path to file containing API keys"),
    ("--no-webui", "", "disable the Web UI"),
    ("--embedding, --embeddings", "", "restrict to only support embedding use case"),
    ("--metrics", "", "enable prometheus compatible metrics endpoint (default: disabled)"),
    ("--props", "", "enable changing global properties via POST /props (default: disabled)"),
    ("--slots", "", "enable slots monitoring endpoint (default: enabled)"),
    ("--jinja", "", "use jinja template for chat (default: enabled)"),
    ("--chat-template-file", "JINJA_TEMPLATE_FILE", "set custom jinja chat template file"),
    ("--reasoning-format", "FORMAT", "controls whether thought tags are allowed (default: auto)"),
    ("--reasoning", "[on|off|auto]", "enable thinking (default: auto)"),
    ("--reasoning-budget", "N", "token budget for thinking (default: -1)"),
    ("-cb, --cont-batching", "", "enable continuous batching (default: enabled)"),
    ("-nocb, --no-cont-batching", "", "disable continuous batching"),
    ("--cache-prompt", "", "enable prompt caching (default: enabled)"),
    ("--no-cache-prompt", "", "disable prompt caching"),
    ("--cache-reuse", "N", "min chunk size to attempt reusing from the cache via KV shifting"),
    (
        "-cram, --cache-ram",
        "N",
        "set the maximum cache size in MiB (default: 8192, -1 - no limit, 0 - disable)",
    ),
    ("-ctxcp, --ctx-checkpoints", "N", "max number of context checkpoints per slot (default: 8)"),
    ("--swa-checkpoints", "N", "max number of SWA checkpoints per slot (default: 3)"),
    ("--swa-full", "", "use full-size SWA cache"),
    ("--checkpoint-min-step", "N", "minimum tokens between checkpoints (default: 256)"),
    ("--cache-idle-slots", "", "save and clear idle slots on new tasks (default: enabled)"),
    ("--no-cache-idle-slots", "", "do not save and clear idle slots"),
    ("-cms, --no-context-shift", "", "disable context shift on infinite text generation"),
    (
        "-sps, --slot-prompt-similarity",
        "SIMILARITY",
        "how much the prompt must match (default: 0.10)",
    ),
    ("--backend-sampling", "", "enable backend sampling (experimental) (default: disabled)"),
    (
        "--spec-type",
        "none,draft-simple,draft-mtp,ngram-simple",
        "type of speculative decoding to use when no draft model is provided",
    ),
    ("--spec-draft-model", "FNAME", "draft model for speculative decoding"),
    ("--spec-draft-n-max", "N", "number of tokens to draft (default: 16)"),
    ("--spec-draft-n-min", "N", "minimum number of draft tokens (default: 0)"),
    ("--spec-draft-p-min", "P", "minimum speculative decoding probability (default: 0.75)"),
    ("--spec-draft-ngl", "N", "max. number of draft model layers to store in VRAM"),
    ("--spec-draft-device", "<dev1,dev2,..>", "devices to use for the draft model"),
    ("--spec-draft-type-k", "TYPE", "KV cache data type for K of the draft model"),
    ("--spec-draft-type-v", "TYPE", "KV cache data type for V of the draft model"),
    ("-hft, --hf-token", "TOKEN", "Hugging Face access token"),
    ("--ssl-key-file", "FNAME", "path to file a PEM-encoded SSL private key"),
    ("--stub-load-ms", "N", "stub only: /health answers 503 for N ms after start (default: 0)"),
    ("--stub-tps", "N", "stub only: generation speed in tokens per second (default: 0)"),
    ("--stub-vram-mib", "N", "stub only: MiB of VRAM this child pretends to hold"),
    ("--stub-build", "N", "stub only: build number reported in /props (default: 0)"),
    ("--stub-embed-dim", "N", "stub only: embedding dimension (default: 64)"),
)


def help_text() -> str:
    """The ``--help`` output, in llama-server's column layout."""
    lines = ["----- stub llama-server (StudioForge staging) -----", ""]
    for flags, placeholder, desc in _FLAG_TABLE:
        head = f"{flags} {placeholder}".rstrip()
        lines.append(f"{head:<44}{desc}" if len(head) < 44 else f"{head}  {desc}")
    lines.append("")
    return "\n".join(lines)


def version_text(build: int = 0) -> str:
    """The ``--version`` output: a ``version:`` line and a ``build`` line."""
    return f"version: {build} (stub)\nbuild {build} (stub)\nbuilt with stub for stub\n"


def _fake_gpus() -> list[dict[str, Any]]:
    raw = os.environ.get("SF_FAKE_GPUS", "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    return [g for g in data if isinstance(g, dict)] if isinstance(data, list) else []


def list_devices_text() -> str:
    """``--list-devices``: the cards ``SF_FAKE_GPUS`` describes, CUDA-style."""
    lines = ["Available devices:"]
    for gpu in _fake_gpus():
        try:
            index = int(gpu["index"])
            total = int(gpu["total_bytes"]) // (1024 * 1024)
            free = int(gpu.get("free_bytes", gpu["total_bytes"])) // (1024 * 1024)
        except (KeyError, TypeError, ValueError):
            continue
        lines.append(f"  CUDA{index}: {gpu.get('name', 'Fake GPU')} ({total} MiB, {free} MiB free)")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Config parsed from the real argv
# ---------------------------------------------------------------------------

_ALIASES = {
    "-m": "--model",
    "-a": "--alias",
    "-c": "--ctx-size",
    "-np": "--parallel",
    "-b": "--batch-size",
    "-ub": "--ubatch-size",
    "-ngl": "--n-gpu-layers",
    "--gpu-layers": "--n-gpu-layers",
    "-sm": "--split-mode",
    "-ts": "--tensor-split",
    "-dev": "--device",
    "-ctk": "--cache-type-k",
    "-ctv": "--cache-type-v",
    "-fa": "--flash-attn",
    "--embeddings": "--embedding",
}


def _normalize(argv: Sequence[str]) -> list[str]:
    out: list[str] = []
    for token in argv:
        if token.startswith("--") and "=" in token:
            name, _, value = token.partition("=")
            out.extend([_ALIASES.get(name, name), value])
        else:
            out.append(_ALIASES.get(token, token))
    return out


def _int(value: str | None, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _float(value: str | None, default: float) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


@dataclass
class StubConfig:
    """Everything the stub reads off its command line."""

    model: str = "stub.gguf"
    alias: str = "stub"
    host: str = "127.0.0.1"
    port: int = 0
    ctx_size: int = 4096
    parallel: int = 1
    n_gpu_layers: str | None = None
    device: str | None = None
    tensor_split: str | None = None
    split_mode: str | None = None
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    flash_attn: str | None = None
    reasoning_format: str | None = None
    chat_template_file: str | None = None
    metrics: bool = False
    load_ms: float = 0.0
    tps: float = 0.0
    vram_mib: int | None = None
    build: int = 0
    embed_dim: int = 64
    # fault injection (kept from the old in-test fake)
    exit_code: int | None = None
    unhealthy: bool = False
    unhealthy_if: str | None = None
    crash_after: float | None = None
    crash_once: str | None = None
    touch_on_crash: str | None = None
    no_listen: bool = False
    ignore_sigterm: bool = False
    args: list[str] = field(default_factory=list)

    @classmethod
    def from_argv(cls, argv: Sequence[str]) -> StubConfig:
        args = _normalize(argv)

        def opt(name: str) -> str | None:
            if name in args:
                index = args.index(name)
                if index + 1 < len(args):
                    return args[index + 1]
            return None

        parallel = _int(opt("--parallel"), 1)
        # llama-server's -1 is "auto": the real engine picks a handful of slots.
        parallel = 4 if parallel < 0 else max(1, parallel)
        ctx = _int(opt("--ctx-size"), 4096)
        ctx = 4096 if ctx <= 0 else ctx
        tps = _float(opt("--stub-tps"), _float(os.environ.get("SF_STUB_TPS"), 0.0))
        load_ms = _float(opt("--stub-load-ms"), _float(os.environ.get("SF_STUB_LOAD_MS"), 0.0))
        crash = opt("--fake-crash-after")
        exit_code = opt("--fake-exit-code")
        vram = opt("--stub-vram-mib")
        return cls(
            model=opt("--model") or "stub.gguf",
            alias=opt("--alias") or "stub",
            host=opt("--host") or "127.0.0.1",
            port=_int(opt("--port"), 0),
            ctx_size=ctx,
            parallel=parallel,
            n_gpu_layers=opt("--n-gpu-layers"),
            device=opt("--device"),
            tensor_split=opt("--tensor-split"),
            split_mode=opt("--split-mode"),
            cache_type_k=opt("--cache-type-k"),
            cache_type_v=opt("--cache-type-v"),
            batch_size=_int(opt("--batch-size"), 0) or None,
            ubatch_size=_int(opt("--ubatch-size"), 0) or None,
            flash_attn=opt("--flash-attn"),
            reasoning_format=opt("--reasoning-format"),
            chat_template_file=opt("--chat-template-file"),
            metrics="--metrics" in args,
            load_ms=max(0.0, load_ms),
            tps=max(0.0, tps),
            vram_mib=_int(vram, 0) if vram is not None else None,
            build=_int(opt("--stub-build"), 0),
            embed_dim=max(1, _int(opt("--stub-embed-dim"), 64)),
            exit_code=_int(exit_code, 1) if exit_code is not None else None,
            unhealthy="--fake-unhealthy" in args,
            unhealthy_if=opt("--fake-unhealthy-if"),
            crash_after=_float(crash, 0.0) if crash is not None else None,
            crash_once=opt("--fake-crash-once"),
            touch_on_crash=opt("--fake-touch-on-crash"),
            no_listen="--fake-no-listen" in args,
            ignore_sigterm="--fake-ignore-sigterm" in args,
            args=args,
        )

    @property
    def slot_ctx(self) -> int:
        return max(1, self.ctx_size // self.parallel)

    @property
    def build_info(self) -> str:
        return f"stub-b{self.build}"


# ---------------------------------------------------------------------------
# Deterministic "model": tokens, text, vectors
# ---------------------------------------------------------------------------

_VOCAB = (
    "the", "a", "stub", "model", "answers", "with", "fixed", "words", "for", "each", "prompt",
    "and", "seed", "so", "tests", "can", "compare", "text", "exactly", "every", "time", "it",
    "runs", "on", "no", "gpu", "at", "all", "slot", "cache", "token", "context", "plan", "lease",
    "load", "serve", "route", "stream", "batch", "layer", "card", "memory", "free", "used",
)  # fmt: skip

_BYTES_PER_TOKEN = 4
_DEFAULT_NEW_TOKENS = 24
_MAX_NEW_TOKENS = 4096


def tokenize(text: str) -> list[int]:
    """Pseudo tokens: four UTF-8 bytes each, id = the chunk read big-endian."""
    raw = text.encode("utf-8")
    return [
        int.from_bytes(raw[i : i + _BYTES_PER_TOKEN], "big")
        for i in range(0, len(raw), _BYTES_PER_TOKEN)
    ]


def detokenize(tokens: Sequence[int]) -> str:
    """Inverse of :func:`tokenize` (leading NULs of a chunk are padding)."""
    out = bytearray()
    for token in tokens:
        value = int(token) & 0xFFFFFFFF
        out += value.to_bytes(_BYTES_PER_TOKEN, "big").lstrip(b"\x00")
    return out.decode("utf-8", "replace")


def _piece(token: int) -> str:
    return detokenize([token])


def generate(prompt: str, seed: int, count: int) -> list[str]:
    """``count`` text pieces: a pure function of ``(prompt, seed)``."""
    digest = hashlib.sha256(f"{seed}\x00{prompt}".encode()).digest()
    pieces: list[str] = []
    while len(pieces) < count:
        for byte in digest:
            word = _VOCAB[byte % len(_VOCAB)]
            pieces.append(word if not pieces else " " + word)
            if len(pieces) >= count:
                break
        digest = hashlib.sha256(digest).digest()
    return pieces


def embed(text: str, dim: int) -> list[float]:
    """A unit-length hash vector of ``dim`` floats (same text -> same vector)."""
    values: list[float] = []
    block = hashlib.sha256(text.encode("utf-8")).digest()
    while len(values) < dim:
        for i in range(0, len(block) - 1, 2):
            values.append(int.from_bytes(block[i : i + 2], "big") / 32767.5 - 1.0)
            if len(values) >= dim:
                break
        block = hashlib.sha256(block).digest()
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [round(v / norm, 6) for v in values]


def apply_template(messages: Sequence[Any]) -> str:
    """The fixed chat template: ``<|role|>`` blocks and an open assistant turn."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        parts.append(f"<|{message.get('role', 'user')}|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Slots and counters
# ---------------------------------------------------------------------------


class _Engine:
    """Slot accounting and the ``llamacpp:*`` counters, thread-safe."""

    def __init__(self, cfg: StubConfig) -> None:
        self.cfg = cfg
        self.started = time.monotonic()
        self._lock = threading.Lock()
        self._sem = threading.Semaphore(cfg.parallel)
        self._free = list(range(cfg.parallel))
        self._busy: set[int] = set()
        self.deferred = 0
        self.prompt_tokens = 0
        self.prompt_seconds = 0.0
        self.predicted_tokens = 0
        self.predicted_seconds = 0.0
        self.n_decode = 0
        self._busy_at_decode = 0
        self.tokens_max = 0
        self.requests = 0

    @contextlib.contextmanager
    def slot(self) -> Iterator[int]:
        with self._lock:
            self.deferred += 1
        self._sem.acquire()
        with self._lock:
            self.deferred -= 1
            slot_id = self._free.pop(0)
            self._busy.add(slot_id)
        try:
            yield slot_id
        finally:
            with self._lock:
                self._busy.discard(slot_id)
                self._free.append(slot_id)
                self._free.sort()
            self._sem.release()

    def busy_count(self) -> int:
        with self._lock:
            return len(self._busy)

    def busy_ids(self) -> set[int]:
        with self._lock:
            return set(self._busy)

    def record(self, n_prompt: int, n_new: int, prompt_ms: float, predicted_ms: float) -> None:
        with self._lock:
            busy = max(1, len(self._busy))
            self.requests += 1
            self.prompt_tokens += n_prompt
            self.prompt_seconds += prompt_ms / 1000.0
            self.predicted_tokens += n_new
            self.predicted_seconds += predicted_ms / 1000.0
            self.n_decode += max(1, n_new)
            self._busy_at_decode += busy * max(1, n_new)
            self.tokens_max = max(self.tokens_max, n_prompt + n_new)

    def metrics_text(self) -> str:
        with self._lock:
            busy = len(self._busy)
            deferred = self.deferred
            per_decode = self._busy_at_decode / self.n_decode if self.n_decode else 0.0
            prompt_tps = self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0
            gen_tps = (
                self.predicted_tokens / self.predicted_seconds if self.predicted_seconds else 0.0
            )
            rows: list[tuple[str, str, str, float]] = [
                ("prompt_tokens_total", "counter", "Number of prompt tokens processed.",
                 self.prompt_tokens),
                ("prompt_seconds_total", "counter", "Prompt process time", self.prompt_seconds),
                ("tokens_predicted_total", "counter", "Number of generation tokens processed.",
                 self.predicted_tokens),
                ("tokens_predicted_seconds_total", "counter", "Predict process time",
                 self.predicted_seconds),
                ("n_decode_total", "counter", "Total number of llama_decode() calls",
                 self.n_decode),
                ("n_busy_slots_per_decode", "gauge", "Average number of busy slots per decode",
                 per_decode),
                ("prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s.",
                 prompt_tps),
                ("predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s.",
                 gen_tps),
                ("requests_processing", "gauge", "Number of requests processing.", busy),
                ("requests_deferred", "gauge", "Number of requests deferred.", deferred),
                ("n_tokens_max", "counter", "Largest observed n_tokens.", self.tokens_max),
            ]  # fmt: skip
        out: list[str] = []
        for name, kind, doc, value in rows:
            out.append(f"# HELP llamacpp:{name} {doc}")
            out.append(f"# TYPE llamacpp:{name} {kind}")
            shown = f"{value:.6f}".rstrip("0").rstrip(".") if isinstance(value, float) else value
            out.append(f"llamacpp:{name} {shown or 0}")
        return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class _StopStream(Exception):
    """The client went away mid-stream."""


class StubServer(ThreadingHTTPServer):
    """The HTTP server; carries the config and the engine state."""

    daemon_threads = True
    # Mirror the real llama-server, which sets SO_REUSEADDR on POSIX only: the
    # flag decides whether TIME_WAIT connections leave the port bindable after
    # the child exits. On Windows it would let a second bind silently share a
    # LIVE listener, hiding a genuine port clash.
    allow_reuse_address = os.name != "nt"

    def __init__(self, cfg: StubConfig) -> None:
        super().__init__((cfg.host, cfg.port), _Handler)
        self.cfg = cfg
        self.engine = _Engine(cfg)


def make_server(cfg: StubConfig) -> StubServer:
    """Bind the server (``port 0`` picks a free one; read ``server_address``)."""
    return StubServer(cfg)


def _error(code: int, message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "type": kind}}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: StubServer

    # -- plumbing -----------------------------------------------------------

    @property
    def cfg(self) -> StubConfig:
        return self.server.cfg

    @property
    def engine(self) -> _Engine:
        return self.server.engine

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - base signature
        sys.stderr.write("fake: " + (format % args) + "\n")
        sys.stderr.flush()

    def _send(self, code: int, payload: Any, *, ctype: str = "application/json") -> None:
        body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self, *, allow_list: bool = False) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except ValueError:
            self._send(400, _error(400, "invalid JSON body"))
            return None
        if allow_list and isinstance(data, list):
            return {"_list": data}
        if not isinstance(data, dict):
            self._send(400, _error(400, "request body must be a JSON object"))
            return None
        return data

    def _sse_open(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload: Any) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        try:
            self.wfile.write(f"data: {text}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            raise _StopStream from exc

    # -- routing ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        try:
            self._get(path)
        except _StopStream:
            return

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        body = self._body(allow_list=path == "/lora-adapters")
        if body is None:
            return
        try:
            self._post(path, body)
        except _StopStream:
            return

    def _healthy(self) -> bool:
        cfg = self.cfg
        if cfg.unhealthy or (cfg.unhealthy_if and Path(cfg.unhealthy_if).exists()):
            return False
        return (time.monotonic() - self.engine.started) * 1000.0 >= cfg.load_ms

    def _get(self, path: str) -> None:
        cfg = self.cfg
        if path == "/health":
            if self._healthy():
                self._send(200, {"status": "ok"})
            else:
                self._send(503, _error(503, "Loading model", "unavailable_error"))
        elif path == "/props":
            self._send(200, self._props())
        elif path == "/slots":
            busy = self.engine.busy_ids()
            self._send(
                200,
                [
                    {
                        "id": i,
                        "n_ctx": cfg.slot_ctx,
                        "speculative": False,
                        "is_processing": i in busy,
                    }
                    for i in range(cfg.parallel)
                ],
            )
        elif path == "/metrics":
            if cfg.metrics:
                self._send(200, self.engine.metrics_text(), ctype="text/plain; version=0.0.4")
            else:
                self._send(
                    501,
                    _error(
                        501,
                        "This server does not support metrics endpoint. Start it with `--metrics`",
                        "not_supported_error",
                    ),
                )
        elif path in ("/v1/models", "/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": cfg.alias,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "llamacpp",
                            "meta": {"n_ctx_train": cfg.ctx_size, "n_vocab": len(_VOCAB)},
                        }
                    ],
                },
            )
        elif path == "/lora-adapters":
            self._send(200, [])
        else:
            self._send(404, _error(404, "File Not Found", "not_found_error"))

    def _props(self) -> dict[str, Any]:
        cfg = self.cfg
        template = "{{ messages }}"
        if cfg.chat_template_file:
            with contextlib.suppress(OSError):
                template = Path(cfg.chat_template_file).read_text(encoding="utf-8")
        return {
            "default_generation_settings": {
                "n_ctx": cfg.slot_ctx,
                "params": {"n_predict": -1, "temperature": 0.8, "seed": -1},
            },
            "total_slots": cfg.parallel,
            "model_alias": cfg.alias,
            "model_path": cfg.model,
            "modalities": {"vision": False, "audio": False},
            "chat_template": template,
            "bos_token": "<s>",
            "eos_token": "</s>",
            "build_info": cfg.build_info,
            "endpoint_slots": True,
            "endpoint_props": False,
            "endpoint_metrics": cfg.metrics,
            "stub": {
                "version": STUB_VERSION,
                "vram_mib": cfg.vram_mib,
                "launch": {
                    "model": cfg.model,
                    "alias": cfg.alias,
                    "ctx_size": cfg.ctx_size,
                    "parallel": cfg.parallel,
                    "n_gpu_layers": cfg.n_gpu_layers,
                    "device": cfg.device,
                    "tensor_split": cfg.tensor_split,
                    "split_mode": cfg.split_mode,
                    "cache_type_k": cfg.cache_type_k,
                    "cache_type_v": cfg.cache_type_v,
                    "batch_size": cfg.batch_size,
                    "ubatch_size": cfg.ubatch_size,
                    "flash_attn": cfg.flash_attn,
                    "reasoning_format": cfg.reasoning_format,
                    "chat_template_file": cfg.chat_template_file,
                    "metrics": cfg.metrics,
                },
            },
        }

    def _post(self, path: str, body: dict[str, Any]) -> None:
        cfg = self.cfg
        if path == "/lora-adapters":
            self._send(200, {"success": True})
        elif path == "/tokenize":
            text = str(body.get("content", ""))
            ids = tokenize(text)
            if body.get("with_pieces"):
                self._send(200, {"tokens": [{"id": i, "piece": _piece(i)} for i in ids]})
            else:
                self._send(200, {"tokens": ids})
        elif path == "/detokenize":
            tokens = body.get("tokens")
            if not isinstance(tokens, list):
                self._send(400, _error(400, "'tokens' must be a list of ints"))
                return
            self._send(200, {"content": detokenize([int(t) for t in tokens])})
        elif path == "/apply-template":
            messages = body.get("messages")
            if not isinstance(messages, list):
                self._send(400, _error(400, "'messages' must be a list"))
                return
            self._send(200, {"prompt": apply_template(messages)})
        elif path == "/completion":
            self._completion(body, shape="native")
        elif path in ("/v1/completions", "/completions"):
            self._completion(body, shape="openai")
        elif path in ("/v1/chat/completions", "/chat/completions"):
            self._chat(body)
        elif path == "/infill":
            self._infill(body)
        elif path == "/v1/embeddings":
            self._embeddings(body, native=False)
        elif path in ("/embedding", "/embeddings"):
            self._embeddings(body, native=True)
        elif path in ("/reranking", "/rerank", "/v1/rerank", "/v1/reranking"):
            self._rerank(body)
        else:
            self._send(404, _error(404, f"File Not Found: {path} ({cfg.alias})", "not_found_error"))

    # -- generation ---------------------------------------------------------

    @staticmethod
    def _prompt_text(prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list):
            if prompt and all(isinstance(p, int) for p in prompt):
                return detokenize(prompt)
            parts = []
            for part in prompt:
                if isinstance(part, dict):
                    parts.append(str(part.get("prompt_string", "")))
                else:
                    parts.append(_Handler._prompt_text(part))
            return "".join(parts)
        return ""

    @staticmethod
    def _seed(body: dict[str, Any]) -> int:
        seed = body.get("seed")
        return seed if isinstance(seed, int) and seed >= 0 else 0

    @staticmethod
    def _limit(body: dict[str, Any], *names: str) -> int | None:
        for name in names:
            value = body.get(name)
            if isinstance(value, int) and value > 0:
                return min(value, _MAX_NEW_TOKENS)
        return None

    def _timings(self, n_prompt: int, n_new: int) -> dict[str, Any]:
        tps = self.cfg.tps if self.cfg.tps > 0 else 10000.0
        prompt_ms = n_prompt * 0.05
        predicted_ms = n_new * 1000.0 / tps
        return {
            "cache_n": 0,
            "prompt_n": n_prompt,
            "prompt_ms": round(prompt_ms, 3),
            "prompt_per_token_ms": 0.05,
            "prompt_per_second": 20000.0,
            "predicted_n": n_new,
            "predicted_ms": round(predicted_ms, 3),
            "predicted_per_token_ms": round(1000.0 / tps, 3),
            "predicted_per_second": round(tps, 3),
        }

    def _overflow(self, n_prompt: int) -> bool:
        """Answer the engine's own refusal when the prompt cannot fit a slot."""
        if n_prompt < self.cfg.slot_ctx:
            return False
        self._send(
            400,
            {
                "error": {
                    "code": 400,
                    "message": (
                        f"request ({n_prompt} tokens) exceeds the available context size "
                        f"({self.cfg.slot_ctx} tokens), try increasing it"
                    ),
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": n_prompt,
                    "n_ctx": self.cfg.slot_ctx,
                }
            },
        )
        return True

    def _pace(self, n_new: int) -> None:
        if self.cfg.tps > 0 and n_new > 0:
            time.sleep(n_new / self.cfg.tps)

    def _run(
        self, prompt: str, body: dict[str, Any], *names: str
    ) -> tuple[list[str], int, int, bool]:
        """Generate non-streaming: ``(pieces, n_prompt, n_new, hit_limit)``."""
        n_prompt = len(tokenize(prompt))
        limit = self._limit(body, *names)
        n_new = limit if limit is not None else _DEFAULT_NEW_TOKENS
        n_new = min(n_new, max(1, self.cfg.slot_ctx - n_prompt))
        pieces = generate(prompt, self._seed(body), n_new)
        return pieces, n_prompt, n_new, limit is not None

    def _completion(self, body: dict[str, Any], *, shape: str) -> None:
        prompt = self._prompt_text(body.get("prompt", ""))
        n_prompt = len(tokenize(prompt))
        if self._overflow(n_prompt):
            return
        stream = bool(body.get("stream"))
        names = ("n_predict", "max_tokens", "max_completion_tokens")
        pieces, n_prompt, n_new, limited = self._run(prompt, body, *names)
        timings = self._timings(n_prompt, n_new)
        cfg = self.cfg
        rid = "cmpl-" + hashlib.sha256(f"{prompt}{self._seed(body)}".encode()).hexdigest()[:16]
        created = int(time.time())
        with self.engine.slot() as slot_id:
            if stream:
                self._sse_open()
                for piece in pieces:
                    self._pace(1)
                    if shape == "native":
                        self._sse({"index": 0, "content": piece, "stop": False, "id_slot": slot_id})
                    else:
                        self._sse(
                            {
                                "id": rid,
                                "object": "text_completion",
                                "created": created,
                                "model": cfg.alias,
                                "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                            }
                        )
                reason = "length" if limited else "stop"
                if shape == "native":
                    self._sse(
                        {
                            "index": 0,
                            "content": "",
                            "stop": True,
                            "id_slot": slot_id,
                            "model": cfg.alias,
                            "stop_type": "limit" if limited else "eos",
                            "tokens_predicted": n_new,
                            "tokens_evaluated": n_prompt,
                            "timings": timings,
                        }
                    )
                else:
                    self._sse(
                        {
                            "id": rid,
                            "object": "text_completion",
                            "created": created,
                            "model": cfg.alias,
                            "choices": [{"index": 0, "text": "", "finish_reason": reason}],
                            "timings": timings,
                        }
                    )
                    if (body.get("stream_options") or {}).get("include_usage"):
                        self._sse(
                            self._usage_chunk(rid, "text_completion", created, n_prompt, n_new)
                        )
                    self._sse("[DONE]")
            else:
                self._pace(n_new)
                text = "".join(pieces)
                if shape == "native":
                    payload: dict[str, Any] = {
                        "index": 0,
                        "content": text,
                        "id_slot": slot_id,
                        "stop": True,
                        "model": cfg.alias,
                        "tokens_predicted": n_new,
                        "tokens_evaluated": n_prompt,
                        "stop_type": "limit" if limited else "eos",
                        "stopping_word": "",
                        "truncated": False,
                        "prompt": prompt,
                        "generation_settings": {"n_ctx": cfg.slot_ctx, "seed": self._seed(body)},
                        "timings": timings,
                    }
                else:
                    payload = {
                        "id": rid,
                        "object": "text_completion",
                        "created": created,
                        "model": cfg.alias,
                        "system_fingerprint": cfg.build_info,
                        "choices": [
                            {
                                "index": 0,
                                "text": text,
                                "finish_reason": "length" if limited else "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": n_prompt,
                            "completion_tokens": n_new,
                            "total_tokens": n_prompt + n_new,
                        },
                        "timings": timings,
                    }
                self._send(200, payload)
        self.engine.record(n_prompt, n_new, timings["prompt_ms"], timings["predicted_ms"])

    def _usage_chunk(
        self, rid: str, obj: str, created: int, n_prompt: int, n_new: int
    ) -> dict[str, Any]:
        return {
            "id": rid,
            "object": obj,
            "created": created,
            "model": self.cfg.alias,
            "choices": [],
            "usage": {
                "prompt_tokens": n_prompt,
                "completion_tokens": n_new,
                "total_tokens": n_prompt + n_new,
            },
        }

    def _chat(self, body: dict[str, Any]) -> None:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send(400, _error(400, "'messages' must be a non-empty array"))
            return
        prompt = apply_template(messages)
        n_prompt = len(tokenize(prompt))
        if self._overflow(n_prompt):
            return
        cfg = self.cfg
        pieces, n_prompt, n_new, limited = self._run(
            prompt, body, "max_tokens", "max_completion_tokens", "n_predict"
        )
        timings = self._timings(n_prompt, n_new)
        rid = "chatcmpl-" + hashlib.sha256(f"{prompt}{self._seed(body)}".encode()).hexdigest()[:16]
        created = int(time.time())
        reason = "length" if limited else "stop"
        with self.engine.slot():
            if body.get("stream"):
                self._sse_open()

                def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                    return {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": cfg.alias,
                        "system_fingerprint": cfg.build_info,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }

                self._sse(chunk({"role": "assistant", "content": None}))
                for piece in pieces:
                    self._pace(1)
                    self._sse(chunk({"content": piece}))
                final = chunk({}, reason)
                final["timings"] = timings
                self._sse(final)
                if (body.get("stream_options") or {}).get("include_usage"):
                    self._sse(
                        self._usage_chunk(rid, "chat.completion.chunk", created, n_prompt, n_new)
                    )
                self._sse("[DONE]")
            else:
                self._pace(n_new)
                self._send(
                    200,
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "created": created,
                        "model": cfg.alias,
                        "system_fingerprint": cfg.build_info,
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": reason,
                                "message": {"role": "assistant", "content": "".join(pieces)},
                            }
                        ],
                        "usage": {
                            "prompt_tokens": n_prompt,
                            "completion_tokens": n_new,
                            "total_tokens": n_prompt + n_new,
                        },
                        "timings": timings,
                    },
                )
        self.engine.record(n_prompt, n_new, timings["prompt_ms"], timings["predicted_ms"])

    def _infill(self, body: dict[str, Any]) -> None:
        prefix = str(body.get("input_prefix", ""))
        suffix = str(body.get("input_suffix", ""))
        prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"
        n_prompt = len(tokenize(prompt))
        if self._overflow(n_prompt):
            return
        pieces, n_prompt, n_new, limited = self._run(prompt, body, "n_predict", "max_tokens")
        timings = self._timings(n_prompt, n_new)
        with self.engine.slot() as slot_id:
            self._pace(n_new)
            self._send(
                200,
                {
                    "index": 0,
                    "content": "".join(pieces),
                    "id_slot": slot_id,
                    "stop": True,
                    "model": self.cfg.alias,
                    "tokens_predicted": n_new,
                    "tokens_evaluated": n_prompt,
                    "stop_type": "limit" if limited else "eos",
                    "timings": timings,
                },
            )
        self.engine.record(n_prompt, n_new, timings["prompt_ms"], timings["predicted_ms"])

    def _inputs(self, body: dict[str, Any]) -> list[str] | None:
        raw = body.get("input", body.get("content"))
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list) and raw:
            if all(isinstance(r, str) for r in raw):
                return list(raw)
            if all(isinstance(r, int) for r in raw):
                return [detokenize(raw)]
            if all(isinstance(r, list) for r in raw):
                return [detokenize(r) for r in raw]
        return None

    def _embeddings(self, body: dict[str, Any], *, native: bool) -> None:
        texts = self._inputs(body)
        if texts is None:
            self._send(400, _error(400, "'input' must be a string or a non-empty array"))
            return
        n_prompt = sum(len(tokenize(t)) for t in texts)
        vectors = [embed(t, self.cfg.embed_dim) for t in texts]
        with self.engine.slot():
            if native:
                self._send(200, [{"index": i, "embedding": [v]} for i, v in enumerate(vectors)])
            else:
                self._send(
                    200,
                    {
                        "object": "list",
                        "model": self.cfg.alias,
                        "data": [
                            {"object": "embedding", "index": i, "embedding": v}
                            for i, v in enumerate(vectors)
                        ],
                        "usage": {"prompt_tokens": n_prompt, "total_tokens": n_prompt},
                    },
                )
        self.engine.record(n_prompt, 0, n_prompt * 0.05, 0.0)

    def _rerank(self, body: dict[str, Any]) -> None:
        query = body.get("query")
        documents = body.get("documents")
        if not isinstance(query, str) or not isinstance(documents, list) or not documents:
            self._send(400, _error(400, "'query' and a non-empty 'documents' array are required"))
            return
        q_words = {w for w in query.lower().split() if w}
        results: list[dict[str, Any]] = []
        for index, doc in enumerate(documents):
            text = doc if isinstance(doc, str) else str((doc or {}).get("text", ""))
            d_words = {w for w in text.lower().split() if w}
            overlap = len(q_words & d_words) / max(1, len(q_words))
            jitter = hashlib.sha256(f"{query}\x00{text}".encode()).digest()[0] / 2550.0
            results.append(
                {"index": index, "relevance_score": round(min(1.0, overlap + jitter), 6)}
            )
        results.sort(key=lambda r: (-r["relevance_score"], r["index"]))
        top_n = body.get("top_n")
        if isinstance(top_n, int) and top_n > 0:
            results = results[:top_n]
        n_prompt = len(tokenize(query)) + sum(len(tokenize(str(d))) for d in documents)
        with self.engine.slot():
            self._send(
                200,
                {
                    "model": self.cfg.alias,
                    "object": "list",
                    "usage": {"prompt_tokens": n_prompt, "total_tokens": n_prompt},
                    "results": results,
                },
            )
        self.engine.record(n_prompt, 0, n_prompt * 0.05, 0.0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _apply_faults(cfg: StubConfig) -> None:
    """The start-up faults: exit immediately, schedule a crash, ignore SIGTERM."""
    sys.stderr.write(f"fake-llama-server starting on port {cfg.port}\n")
    sys.stderr.flush()
    if cfg.exit_code is not None:
        sys.stderr.write("CUDA error: out of memory\n")
        sys.stderr.write("fatal: failed to load model\n")
        sys.stderr.flush()
        sys.exit(cfg.exit_code)
    if cfg.ignore_sigterm:
        for name in ("SIGTERM", "SIGBREAK"):
            sig = getattr(signal, name, None)
            if sig is not None:
                with contextlib.suppress(ValueError, OSError):
                    signal.signal(sig, signal.SIG_IGN)
    crash = cfg.crash_after is not None
    if crash and cfg.crash_once:
        if Path(cfg.crash_once).exists():
            crash = False
        else:
            Path(cfg.crash_once).write_text("1", encoding="utf-8")
    if crash:
        delay = float(cfg.crash_after or 0.0)

        def _crash() -> None:
            time.sleep(delay)
            sys.stderr.write("fake: simulated crash\n")
            sys.stderr.flush()
            if cfg.touch_on_crash:
                Path(cfg.touch_on_crash).write_text("1", encoding="utf-8")
            os._exit(9)

        threading.Thread(target=_crash, daemon=True).start()


def main(argv: Sequence[str] | None = None) -> int:
    """Run as ``llama-server`` would: CLI shims first, else serve until killed."""
    args = list(sys.argv[1:] if argv is None else argv)
    normalized = _normalize(args)
    build = 0
    if "--stub-build" in normalized:
        idx = normalized.index("--stub-build")
        build = _int(normalized[idx + 1] if idx + 1 < len(normalized) else None, 0)
    if "--version" in normalized:
        sys.stdout.write(version_text(build))
        return 0
    if "--list-devices" in normalized:
        sys.stdout.write(list_devices_text())
        return 0
    if any(flag in normalized for flag in ("--help", "-h", "--usage")):
        sys.stdout.write(help_text())
        return 0

    cfg = StubConfig.from_argv(args)
    _apply_faults(cfg)
    if cfg.no_listen:
        # Alive but deaf: tests that a foreign listener on our port is not
        # mistaken for this child.
        while True:
            time.sleep(0.1)
    server = make_server(cfg)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
