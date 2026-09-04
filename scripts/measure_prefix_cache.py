#!/usr/bin/env python3
"""Measure what the prompt-prefix cache really does for concurrent requests.

Stand-alone like ``bench_parallel.py``: httpx only, no repo imports, so it
runs against any StudioForge (or a bare llama-server behind one) this build
does not control. Read-only except for the completions it sends.

WHY THIS EXISTS (DECISIONS.md D54). A client measured ``usage.prompt_tokens``
across seven concurrent chapter requests, saw 68k, and concluded prefix
caching was off. ``usage.prompt_tokens`` is the SIZE of each prompt and never
moves; the work done is ``timings.prompt_n`` (tokens actually processed) and
the work saved is ``timings.cache_n`` (tokens reused). Both are on the final
response of every completion, streamed or not. This script reads those.

WHAT IT SHOWS. Two hard limits of llama-server's cache, with numbers:

* no cross-slot sharing -- at ``--parallel N`` the first N concurrent requests
  each prefill the whole prompt; only the requests that land on a warm slot
  reuse. Serial vs concurrent phases make the difference visible;
* hybrid/recurrent models (Qwen3.5/3.6/3.8) reuse back to a *context
  checkpoint*, and checkpoints sit at user-message starts. ``--diverge
  mid-message`` puts the per-request difference INSIDE the shared message and
  shows ``cache_n`` collapse to the previous checkpoint.

IT REFUSES TO RUN on a rig that is not idle: a lease on ``/api/leases``
(something outside this script holds cards), ``/health.busy.active_requests >
0`` (someone is being served), or a benchmark in ``/api/status``. ``--yes``
overrides *only* those idle checks. It never loads or unloads a model: the
target must already be ``ready`` (``load_recommended`` it first), and
``--concurrency`` must not exceed the instance's ``parallel`` -- both are
refused even with ``--yes``, because a load would change what is measured.

Exit status: 0 on a completed run, 2 on refusal, 3 if any request errored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

EXIT_OK = 0
EXIT_REFUSED = 2
EXIT_REQUEST_ERROR = 3

METRIC_PROMPT = "llamacpp:prompt_tokens_total"
METRIC_PROMPT_CACHED = "llamacpp:prompt_tokens_cached_total"
METRIC_DECODE = "llamacpp:n_decode_total"
METRIC_PREDICTED = "llamacpp:tokens_predicted_total"
METRICS_OF_INTEREST = (METRIC_PROMPT, METRIC_PROMPT_CACHED, METRIC_DECODE, METRIC_PREDICTED)

#: Deterministic pseudo-prose vocabulary. Nonsense on purpose: the model must
#: not be able to predict the text (an n-gram drafter would make the
#: generation side look faster than it is -- D38's ngram trap), and no
#: sentence may repeat (a repeated sentence IS a cache hit in disguise).
_WORDS = (
    "harbour", "lantern", "quartz", "meadow", "signal", "copper", "thistle", "ledger",
    "orbit", "velvet", "granite", "whisper", "marble", "cinder", "saddle", "pylon",
    "ember", "glacier", "tundra", "parlour", "beacon", "mortar", "sextant", "ballast",
    "tallow", "furrow", "kestrel", "gantry", "lattice", "plinth", "rivet", "sepia",
    "trellis", "umber", "vellum", "wharf", "yarrow", "zephyr", "anvil", "bramble",
    "cobalt", "dune", "estuary", "fathom", "gable", "hollow", "isthmus", "juniper", "knoll",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested; nothing below here touches the network)
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="measure_prefix_cache",
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", default="http://127.0.0.1:1234", help="gateway base URL")
    parser.add_argument("--model", required=True, help="model id; must already be loaded")
    parser.add_argument("--prefix-tokens", type=int, default=8000, help="shared prefix size")
    parser.add_argument("--tail-tokens", type=int, default=1200, help="per-request differing part")
    parser.add_argument("--requests", type=int, default=7, help="requests per phase")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="in-flight requests in the concurrent phase"
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--pairs", type=int, default=2, help="serial+concurrent pairs to alternate (rig noise)"
    )
    parser.add_argument(
        "--diverge",
        choices=("user-boundary", "mid-message"),
        default="user-boundary",
        help="where the per-request difference begins (mid-message shows the hybrid cliff)",
    )
    parser.add_argument(
        "--warm-first",
        action="store_true",
        help="extra phase: one tiny request carrying the prefix, then the concurrent fan-out",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--client-label", default="measure_prefix_cache", help="X-SF-Client")
    parser.add_argument("--json", dest="json_out", default=None, help="write every row here")
    parser.add_argument(
        "--yes",
        "--force",
        dest="yes",
        action="store_true",
        help="run even if the rig is not idle (lease, active requests, benchmark)",
    )
    args = parser.parse_args(argv)
    if args.requests < 1 or args.concurrency < 1 or args.pairs < 1:
        parser.error("--requests, --concurrency and --pairs must be >= 1")
    return args


def find_instance(status: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    """The ``/api/status loaded[]`` row for ``model_id``, or ``None``."""
    for row in status.get("loaded") or []:
        if row.get("model_id") == model_id:
            return row
    return None


def instance_parallel(row: dict[str, Any]) -> int:
    """Slots the instance was launched with: ``effective`` first (D54), then the plan."""
    effective = row.get("effective") or {}
    if isinstance(effective.get("parallel"), int):
        return int(effective["parallel"])
    plan = row.get("plan") or {}
    if isinstance(plan.get("parallel"), int):
        return int(plan["parallel"])
    return int(row.get("max_parallel") or 1)


def refusal_reason(
    status: dict[str, Any],
    leases: list[dict[str, Any]] | dict[str, Any] | None,
    health: dict[str, Any],
    model_id: str,
    concurrency: int,
    *,
    yes: bool = False,
) -> str | None:
    """Why this run must not start, or ``None``.

    The idle checks (lease, active requests, running benchmark) bow to
    ``yes``. The two that would change what is being measured -- the model is
    not resident and ready, or the concurrency exceeds its slots -- never do.
    """
    row = find_instance(status, model_id)
    if row is None or row.get("state") != "ready":
        state = "not loaded" if row is None else f"state={row.get('state')}"
        return (
            f"model {model_id!r} is {state}; this script never loads anything -- "
            "load it first (e.g. load_recommended) and rerun"
        )
    slots = instance_parallel(row)
    if concurrency > slots:
        return (
            f"--concurrency {concurrency} exceeds the instance's parallel={slots}; "
            "the extra requests would only queue"
        )
    if yes:
        return None

    lease_rows = leases.get("leases") if isinstance(leases, dict) else leases
    lease_rows = lease_rows or []
    if lease_rows:
        first = lease_rows[0]
        holder = first.get("holder", "?")
        devices = first.get("devices", [])
        return (
            f"a GPU lease is held by {holder!r} on devices {devices}; the cards are not idle "
            "(--yes to override)"
        )
    busy = health.get("busy") or {}
    active = int(busy.get("active_requests") or 0)
    if active > 0:
        return f"the server is serving {active} request(s) right now (--yes to override)"
    if status.get("benchmark"):
        bench = status["benchmark"]
        return (
            f"a benchmark is running ({bench.get('mode')} on {bench.get('model_id')}); "
            "measuring beside it would corrupt both (--yes to override)"
        )
    return None


def build_prose(word_count: int, seed: int) -> str:
    """Deterministic nonsense prose with no repeated sentence."""
    rng = random.Random(seed)
    sentences: list[str] = []
    words_used = 0
    line = 0
    while words_used < word_count:
        length = rng.randint(7, 15)
        body = " ".join(rng.choice(_WORDS) for _ in range(length))
        line += 1
        # The running index guarantees uniqueness even if the RNG repeats.
        sentences.append(f"{body.capitalize()} {line}.")
        words_used += length + 1
    return " ".join(sentences)


def build_messages(
    prefix: str, tail: str, index: int, diverge: str = "user-boundary"
) -> list[dict[str, str]]:
    """The request messages for chapter ``index``.

    ``user-boundary`` puts the per-request text in its own final user message,
    which is where a hybrid model has a context checkpoint to roll back to.
    ``mid-message`` appends it INSIDE the shared bible message: the divergence
    then sits before the last checkpoint and reuse falls back to the previous
    one, usually the very first batch.
    """
    system = "You are a novelist. Write in plain prose. Follow the story bible exactly."
    chapter = f"Chapter {index}: {tail}"
    if diverge == "mid-message":
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Story bible:\n{prefix}\n\n{chapter}"},
        ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Story bible:\n{prefix}"},
        {"role": "user", "content": chapter},
    ]


def parse_metrics(text: str) -> dict[str, float]:
    """Prometheus exposition -> ``{name: value}`` for the counters we read."""
    out: dict[str, float] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        name = name.split("{", 1)[0]
        if name in METRICS_OF_INTEREST:
            try:
                out[name] = float(value.strip())
            except ValueError:
                continue
    return out


def row_from_response(
    index: int, phase: str, wall_s: float, data: dict[str, Any]
) -> dict[str, Any]:
    """One per-request row from a non-streaming chat completion body."""
    usage = data.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    timings = data.get("timings") or {}
    return {
        "index": index,
        "phase": phase,
        "wall_s": round(wall_s, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "cache_n": timings.get("cache_n"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "error": data.get("error"),
    }


def summarize_phase(
    rows: list[dict[str, Any]],
    before: dict[str, float],
    after: dict[str, float],
    wall_s: float,
) -> dict[str, Any]:
    """Per-phase totals, plus D37's ``achieved_batch`` control."""

    def total(key: str) -> int:
        return int(sum(float(r[key] or 0) for r in rows if r.get("error") is None))

    prompt_tokens = total("prompt_tokens")
    prompt_n = total("prompt_n")
    cache_n = total("cache_n")
    predicted_n = total("predicted_n")
    seen = prompt_n + cache_n
    decode_delta = after.get(METRIC_DECODE, 0.0) - before.get(METRIC_DECODE, 0.0)
    cached_delta = after.get(METRIC_PROMPT_CACHED, 0.0) - before.get(METRIC_PROMPT_CACHED, 0.0)
    return {
        "requests": len(rows),
        "errors": sum(1 for r in rows if r.get("error") is not None),
        "sum_prompt_tokens": prompt_tokens,
        "sum_prompt_n": prompt_n,
        "sum_cache_n": cache_n,
        "hit_ratio": round(cache_n / seen, 4) if seen else None,
        "wall_s": round(wall_s, 3),
        "sum_predicted_n": predicted_n,
        "aggregate_gen_tps": round(predicted_n / wall_s, 2) if wall_s > 0 else None,
        "decode_delta": int(decode_delta),
        "achieved_batch": round(predicted_n / decode_delta, 3) if decode_delta > 0 else None,
        "child_cached_delta": int(cached_delta) if METRIC_PROMPT_CACHED in after else None,
    }


def expected_shape(phase: str, concurrency: int, diverge: str) -> str:
    """What the numbers should look like if the cache behaves as documented."""
    if diverge == "mid-message":
        return (
            "cache_n collapsing to the previous context checkpoint (often the first batch) "
            "on a hybrid model; near-full reuse on a full-attention one"
        )
    if phase.startswith("serial"):
        return "first request cache_n 0, the rest ~ shared prefix (prompt_n ~ tail only)"
    if phase.startswith("concurrent"):
        return (
            f"the first {concurrency} requests cache_n 0 (no cross-slot sharing), "
            "the rest ~ shared prefix"
        )
    return "one warm slot serves one of the first-wave requests; the others still prefill cold"


def format_report(
    model_id: str,
    launch: dict[str, Any] | None,
    phases: list[tuple[str, dict[str, Any], list[dict[str, Any]]]],
    *,
    concurrency: int,
    diverge: str,
) -> str:
    lines = [f"prefix-cache measurement -- {model_id}"]
    if launch:
        summary = launch.get("summary")
        lines.append(f"  effective: {summary}" if summary else f"  launch: {json.dumps(launch)}")
    else:
        lines.append("  effective: (not reported by this server; pre-D54)")
    lines.append(f"  diverge: {diverge}   concurrency: {concurrency}")
    for name, summary, rows in phases:
        lines.append("")
        lines.append(
            f"[{name}] {summary['requests']} req, wall {summary['wall_s']} s, "
            f"sum prompt_tokens {summary['sum_prompt_tokens']}, "
            f"sum prompt_n {summary['sum_prompt_n']}, sum cache_n {summary['sum_cache_n']}, "
            f"hit {summary['hit_ratio']}, gen {summary['aggregate_gen_tps']} tok/s, "
            f"achieved_batch {summary['achieved_batch']}"
            + (f", errors {summary['errors']}" if summary["errors"] else "")
        )
        lines.append(f"  expected: {expected_shape(name, concurrency, diverge)}")
        for r in rows:
            lines.append(
                f"  #{r['index']:<2} {r['wall_s']:>7.2f}s  "
                f"prompt_tokens {r['prompt_tokens']!s:>6}  "
                f"cache_n {r['cache_n']!s:>6}  prompt_n {r['prompt_n']!s:>6}  "
                f"predicted_n {r['predicted_n']!s:>4}"
                + (f"  ERROR {r['error']}" if r.get("error") else "")
            )
    lines.append("")
    lines.append(
        "read cache_n / prompt_n, never usage.prompt_tokens: the latter is the prompt's size "
        "and cannot move (D54)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Network side
# ---------------------------------------------------------------------------


async def _get_json(client: Any, url: str) -> Any:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def _count_tokens(client: Any, child_base: str, text: str) -> int:
    """Token count via the child's native ``/tokenize``; chars/4 if refused."""
    try:
        response = await client.post(f"{child_base}/tokenize", json={"content": text})
        response.raise_for_status()
        tokens = response.json().get("tokens")
        if isinstance(tokens, list):
            return len(tokens)
    except Exception:  # noqa: BLE001 - a size estimate is fine
        pass
    return max(1, len(text) // 4)


async def _fit_prose(client: Any, child_base: str, target_tokens: int, seed: int) -> str:
    """Grow the prose until it tokenises to about ``target_tokens``."""
    words = max(50, int(target_tokens * 0.7))
    text = build_prose(words, seed)
    for _ in range(6):
        count = await _count_tokens(client, child_base, text)
        if abs(count - target_tokens) <= max(50, target_tokens // 50):
            break
        words = max(50, int(words * target_tokens / max(1, count)))
        text = build_prose(words, seed)
    return text


async def _one_request(
    client: Any,
    base: str,
    model_id: str,
    messages: list[dict[str, str]],
    *,
    index: int,
    phase: str,
    max_tokens: int,
    seed: int,
) -> dict[str, Any]:
    body = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "temperature": 0,
        "seed": seed,
        "max_tokens": max_tokens,
        "cache_prompt": True,
    }
    started = time.perf_counter()
    try:
        response = await client.post(f"{base}/v1/chat/completions", json=body)
        data = response.json()
        if response.status_code != 200 and "error" not in data:
            data = {"error": f"HTTP {response.status_code}"}
    except Exception as exc:  # noqa: BLE001 - the row records it
        data = {"error": f"{type(exc).__name__}: {exc}"}
    return row_from_response(index, phase, time.perf_counter() - started, data)


async def _run_phase(
    client: Any,
    base: str,
    child_base: str,
    model_id: str,
    name: str,
    messages_per_request: list[list[dict[str, str]]],
    *,
    concurrency: int,
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    before = parse_metrics((await client.get(f"{child_base}/metrics")).text)
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(index: int, messages: list[dict[str, str]]) -> dict[str, Any]:
        async with semaphore:
            return await _one_request(
                client,
                base,
                model_id,
                messages,
                index=index,
                phase=name,
                max_tokens=max_tokens,
                seed=seed,
            )

    started = time.perf_counter()
    rows = await asyncio.gather(
        *(guarded(i, m) for i, m in enumerate(messages_per_request, start=1))
    )
    wall = time.perf_counter() - started
    after = parse_metrics((await client.get(f"{child_base}/metrics")).text)
    return summarize_phase(list(rows), before, after, wall), list(rows)


async def main_async(args: argparse.Namespace) -> tuple[int, dict[str, Any] | None]:
    """Run everything; returns ``(exit code, json payload or None)``."""
    try:
        import httpx
    except ImportError:
        print("httpx is required: pip install httpx", file=sys.stderr)
        return EXIT_REFUSED, None

    headers = {"X-SF-Client": args.client_label}
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), headers=headers) as c:
        status = await _get_json(c, f"{args.base}/api/status")
        health = await _get_json(c, f"{args.base}/health")
        try:
            leases = await _get_json(c, f"{args.base}/api/leases")
        except Exception:  # noqa: BLE001 - older server without the route
            leases = status.get("leases") or []
        reason = refusal_reason(status, leases, health, args.model, args.concurrency, yes=args.yes)
        if reason:
            print(f"refused: {reason}", file=sys.stderr)
            return EXIT_REFUSED, None
        row = find_instance(status, args.model) or {}
        port = row.get("port")
        if not port:
            print("refused: the instance reports no child port", file=sys.stderr)
            return EXIT_REFUSED, None
        child_base = f"http://127.0.0.1:{port}"
        launch = row.get("effective")
        if launch is None:
            try:
                models = await _get_json(c, f"{args.base}/api/models")
                launch = next(
                    (
                        m.get("settings")
                        for m in models.get("models", [])
                        if m.get("id") == args.model
                    ),
                    None,
                )
            except Exception:  # noqa: BLE001
                launch = None

        prefix = await _fit_prose(c, child_base, args.prefix_tokens, args.seed)
        tails = [
            await _fit_prose(c, child_base, args.tail_tokens, args.seed * 1000 + i)
            for i in range(1, args.requests + 1)
        ]
        requests = [
            build_messages(prefix, tail, i, args.diverge) for i, tail in enumerate(tails, start=1)
        ]

        phases: list[tuple[str, dict[str, Any], list[dict[str, Any]]]] = []
        errors = 0
        for pair in range(1, args.pairs + 1):
            for name, conc in (("serial", 1), ("concurrent", args.concurrency)):
                label = f"{name}#{pair}"
                summary, rows = await _run_phase(
                    c,
                    args.base,
                    child_base,
                    args.model,
                    label,
                    requests,
                    concurrency=conc,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
                errors += summary["errors"]
                phases.append((label, summary, rows))
            if args.warm_first:
                warm = build_messages(prefix, "warm-up", 0, args.diverge)
                await _one_request(
                    c, args.base, args.model, warm, index=0, phase="warm", max_tokens=1, seed=1
                )
                summary, rows = await _run_phase(
                    c,
                    args.base,
                    child_base,
                    args.model,
                    f"warm-first#{pair}",
                    requests,
                    concurrency=args.concurrency,
                    max_tokens=args.max_tokens,
                    seed=args.seed,
                )
                errors += summary["errors"]
                phases.append((f"warm-first#{pair}", summary, rows))

    print(
        format_report(
            args.model, launch, phases, concurrency=args.concurrency, diverge=args.diverge
        )
    )
    payload = {
        "model": args.model,
        "effective": launch,
        "args": vars(args),
        "phases": [{"name": n, "summary": s, "rows": r} for n, s, r in phases],
    }
    return (EXIT_REQUEST_ERROR if errors else EXIT_OK), payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, payload = asyncio.run(main_async(args))
    if args.json_out and payload is not None:
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
