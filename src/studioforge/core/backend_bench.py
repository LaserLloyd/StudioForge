"""The on-box micro-benchmark that feeds backend selection (``llama-bench``).

**Running :func:`run_backend_benchmark` loads a model onto the GPU and
saturates it for up to several minutes.** It is therefore never reached by the
server, a timer or a test: the only caller is the explicit
``studioforge engine benchmark --yes`` command, and the function itself refuses
to run unless ``confirm=True`` is passed. Everything else in this module (the
argv builder, the JSON parser) is pure.

STRICT_GPU applies to the benchmark exactly as to a load: the argv always
carries ``-ngl 999`` and accepts no extra flags, so a measured number is a
fully-resident number, comparable with what the server will actually do.

The two numbers are llama-bench's ``pp512`` (512-token prompt processing) and
``tg128`` (128-token generation), in tokens per second, as the mean over
``repetitions`` runs.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from studioforge.core.backend_select import BackendBenchResult
from studioforge.core.engine import EngineError, child_environment

BENCH_NAME = "llama-bench.exe" if os.name == "nt" else "llama-bench"

#: ``(argv, timeout_s) -> (returncode, stdout)``; injected by tests.
BenchRunner = Callable[[Sequence[str], float], Awaitable[tuple[int, str]]]


class BenchParseError(ValueError):
    """llama-bench output that does not contain both a pp512 and a tg128 row."""


def find_bench_binary(server_binary: Path) -> Path | None:
    """``llama-bench`` next to ``llama-server``, if the engine shipped one.

    Prebuilt archives include it; this manager's source build compiles only the
    ``llama-server`` target, so a source-built engine has none.
    """
    candidate = server_binary.with_name(BENCH_NAME)
    return candidate if candidate.is_file() else None


def llama_bench_command(
    binary: Path, model: Path, *, device: str | None = None, repetitions: int = 3
) -> list[str]:
    """The exact argv: pp512 + tg128, all layers on the GPU, JSON on stdout."""
    argv = [
        str(binary),
        "-m",
        str(model),
        "-p",
        "512",
        "-n",
        "128",
        "-ngl",
        "999",
        "-r",
        str(max(1, int(repetitions))),
        "-o",
        "json",
    ]
    if device:
        argv += ["--device", device]
    return argv


def parse_llama_bench_json(text: str) -> tuple[float, float]:
    """``(pp512, tg128)`` tokens/s out of ``llama-bench -o json`` output.

    The output is a JSON array with one object per test; the prompt test has
    ``n_prompt == 512`` and ``n_gen == 0``, the generation test ``n_prompt == 0``
    and ``n_gen == 128``, each with the mean in ``avg_ts``. Log noise before
    the array is tolerated.
    """
    start = text.find("[")
    if start < 0:
        raise BenchParseError("llama-bench printed no JSON array")
    try:
        rows = json.loads(text[start:])
    except ValueError as exc:
        raise BenchParseError(f"llama-bench JSON did not parse: {exc}") from exc
    pp = tg = None
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        speed = row.get("avg_ts")
        if not isinstance(speed, int | float) or speed <= 0:
            continue
        if row.get("n_prompt") == 512 and not row.get("n_gen"):
            pp = float(speed)
        elif row.get("n_gen") == 128 and not row.get("n_prompt"):
            tg = float(speed)
    if pp is None or tg is None:
        raise BenchParseError(
            "llama-bench output has no usable pp512 and tg128 rows "
            f"(pp512={'ok' if pp else 'missing'}, tg128={'ok' if tg else 'missing'})"
        )
    return pp, tg


async def _spawn(argv: Sequence[str], timeout_s: float) -> tuple[int, str]:
    env, _stripped = child_environment()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=env,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return 124, ""
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


async def run_backend_benchmark(
    binary: Path,
    model: Path,
    *,
    backend: str,
    gpu: str,
    engine_tag: str,
    model_class: str = "",
    device: str | None = None,
    repetitions: int = 3,
    timeout_s: float = 900.0,
    confirm: bool = False,
    runner: BenchRunner | None = None,
) -> BackendBenchResult:
    """Measure one (backend, GPU, model) point with ``llama-bench``.

    SPAWNS A GPU WORKLOAD. ``confirm`` must be ``True`` -- the guard that keeps
    any accidental call (a refactor, a test) from loading a model. ``runner``
    replaces the subprocess for tests. Raises :class:`EngineError` if the tool
    fails, times out, or prints something unparseable.
    """
    if not confirm:
        raise EngineError(
            "refusing to run llama-bench without confirm=True: it loads a model onto the GPU. "
            "Use `studioforge engine benchmark --yes`",
            code="engine_benchmark_unconfirmed",
        )
    argv = llama_bench_command(binary, model, device=device, repetitions=repetitions)
    code, out = await (runner or _spawn)(argv, timeout_s)
    if code == 124:
        raise EngineError(f"llama-bench did not finish within {timeout_s:.0f}s")
    if code != 0:
        raise EngineError(f"llama-bench exited {code}")
    try:
        pp, tg = parse_llama_bench_json(out)
    except BenchParseError as exc:
        raise EngineError(str(exc)) from exc
    return BackendBenchResult(
        backend=backend,
        gpu=gpu,
        engine_tag=engine_tag,
        pp512=pp,
        tg128=tg,
        model_class=model_class,
        measured_at=time.time(),
    )


def describe(result: BackendBenchResult) -> dict[str, Any]:
    return {
        "backend": result.backend,
        "gpu": result.gpu,
        "engine_tag": result.engine_tag,
        "model_class": result.model_class,
        "pp512_tps": round(result.pp512, 2),
        "tg128_tps": round(result.tg128, 2),
    }
