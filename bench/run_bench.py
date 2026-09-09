#!/usr/bin/env python
"""StudioForge Phase 0 benchmark harness. Talks ONLY to the gateway's REST API.

See README.md for the method. Exit codes: 0 done and valid, 1 harness/usage
error, 2 the server refused (its message is printed verbatim), 3 completed but
``valid: false`` (the results file is still written).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_stats as bs  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = "ReadyArt/Dark-Scarlett-27B-v2.0-GGUF/Dark-Scarlett-27B-v2.0.i1-Q5_K_M_hb16"
DEFAULT_LENGTHS = "1024,32768,131072,262144"
CLIENT_LABEL = "bench"
EXIT_OK, EXIT_ERROR, EXIT_REFUSED, EXIT_INVALID = 0, 1, 2, 3
LOAD_TIMEOUT_S = 1800.0
RUN_TIMEOUT_S = 3600.0
TOKEN_TOLERANCE = 0.01  # aim 1% under the clamp so the fitted prompt never exceeds it
GPU_SAMPLE_INTERVAL_S = 0.5
#: How often the sampler reads /api/vram/holders during a run (a co-tenant check).
HOLDERS_SAMPLE_INTERVAL_S = 10.0


class Refused(Exception):
    """The server said no (or is not in a state to run); message is verbatim."""


def error_text(response: httpx.Response) -> str:
    """Render the OpenAI-shaped envelope (errors.py) verbatim, details included."""
    try:
        err = response.json().get("error") or {}
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:800]}"
    text = f"HTTP {response.status_code} [{err.get('code')}]: {err.get('message')}"
    if err.get("studioforge"):
        text += "\n  studioforge: " + json.dumps(err["studioforge"], default=str)
    if response.headers.get("retry-after"):
        text += f"\n  Retry-After: {response.headers['retry-after']}"
    return text


class Gateway:
    def __init__(self, base: str, api_key: str | None, label: str | None) -> None:
        headers = {"X-SF-Client": f"{CLIENT_LABEL}:{label}" if label else CLIENT_LABEL}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.base = base.rstrip("/")
        self.headers = headers
        self.client = httpx.Client(
            base_url=self.base, headers=headers, timeout=httpx.Timeout(120.0, connect=10.0)
        )

    def get(self, path: str) -> Any:
        response = self.client.get(path)
        if response.status_code >= 400:
            raise Refused(f"GET {path} -> {error_text(response)}")
        return response.json()

    def post(self, path: str, body: dict[str, Any], timeout: float = 120.0) -> Any:
        response = self.client.post(path, json=body, timeout=httpx.Timeout(timeout, connect=10.0))
        if response.status_code >= 400:
            raise Refused(f"POST {path} -> {error_text(response)}")
        return response.json()

    def new_client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base, headers=self.headers, timeout=httpx.Timeout(30.0, connect=10.0)
        )


def model_path(model: str, suffix: str) -> str:
    return f"/api/models/{quote(model, safe='/')}{suffix}"


def git_info(repo_root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"sha": "", "dirty": None, "error": f"git unavailable: {exc}"}
    return {"sha": sha, "dirty": bool(dirty)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--server", default="http://127.0.0.1:1234")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--ctx", type=int, default=262144, help="per-slot context to load at")
    p.add_argument("--devices", default="0,1", help="CUDA indices, comma-separated")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument(
        "--prompt-lengths",
        default=DEFAULT_LENGTHS,
        help="tokens; each clamped to ctx - max-tokens - 64",
    )
    p.add_argument("--runs", type=int, default=6, help="per length; run 0 is warm-up and discarded")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--results-dir", default=str(HERE / "results"))
    p.add_argument(
        "--smoke",
        action="store_true",
        help="resident instance only: 1024 tokens, 2 runs, 64 tokens, no load, no restore",
    )
    p.add_argument(
        "--no-restore",
        action="store_true",
        help="leave the bench instance up; do not reload what was resident before",
    )
    p.add_argument(
        "--api-key",
        default=os.environ.get("SF_API_KEY") or None,
        help="bearer key (default: env SF_API_KEY)",
    )
    p.add_argument("--label", default=None, help="suffix for the results file name")
    p.add_argument(
        "--dry-run", action="store_true", help="print the plan and the requests; contact nothing"
    )
    args = p.parse_args(argv)
    args.device_list = bs.parse_int_list(args.devices)
    if args.smoke:
        args.prompt_lengths, args.runs, args.max_tokens = "1024", 2, 64
    args.length_list = bs.parse_int_list(args.prompt_lengths)
    if args.runs < 2 and not args.smoke:
        p.error("--runs must be >= 2 (run 0 is the discarded warm-up)")
    return args


# --------------------------------------------------------------------------
# protocol steps
# --------------------------------------------------------------------------


def preflight(gw: Gateway, args: argparse.Namespace) -> dict[str, Any]:
    health = gw.get("/api/health")
    busy = health.get("busy") or {}
    problems = []
    if health.get("status") != "ok" or health.get("draining"):
        problems.append(f"health status={health.get('status')} draining={health.get('draining')}")
    if busy.get("active_requests"):
        problems.append(
            f"busy.active_requests={busy['active_requests']} ({busy.get('busy_models')})"
        )
    if busy.get("loading"):
        problems.append(f"a load is in progress: {busy['loading']}")
    if busy.get("testing"):
        problems.append(f"a smoke test is running: {busy['testing']}")
    if busy.get("priority_hold"):
        problems.append(f"a priority hold is active: {busy['priority_hold']}")
    status = gw.get("/api/status")
    if status.get("benchmark"):
        problems.append(f"a server benchmark is running: {status['benchmark']}")
    if status.get("queue_depth"):
        problems.append(f"load queue depth {status['queue_depth']}")
    own_lease: dict[str, Any] | None = None
    for lease in status.get("leases") or []:
        if not (set(lease.get("devices") or []) & set(args.device_list)):
            continue
        if args.model in (lease.get("model_ids") or []):
            # A lease taken FOR the model under test: the cards are its alone, which is the
            # quietest rig a baseline can have (the first 2026-09-10 baseline shared the 5090
            # pair with a JIT load for three of its four lengths). The load is allowed by the
            # lease and the unload is the loopback holder's own.
            own_lease = lease
            print(
                f"[preflight] target devices leased to the model under test: {lease.get('id')} ({lease.get('holder')}, priority {lease.get('priority')})"
            )
            continue
        # A baseline unloads/reloads on these cards; a loopback caller would be
        # waived past the D55 lease guard, so refuse instead. Smoke touches nothing.
        (print if args.smoke else problems.append)(
            f"GPU lease on the target devices: {json.dumps(lease, default=str)}"
        )
    models = gw.get("/api/models")
    row = next((m for m in models.get("models") or [] if m.get("id") == args.model), None)
    if row is None:
        problems.append(f"model {args.model!r} is not in the registry (GET /api/models)")
    residents = [i for i in status.get("loaded") or [] if i.get("state") == "ready"]
    mine = next((i for i in residents if i.get("model_id") == args.model), None)
    if args.smoke and mine is None:
        problems.append(
            "--smoke needs the model already resident and ready; nothing is loaded for it"
        )
    if problems:
        raise Refused("preflight refused:\n  - " + "\n  - ".join(problems))
    return {
        "health": health,
        "status": status,
        "model_row": row,
        "residents": residents,
        "resident_instance": mine,
        "own_lease": own_lease,
    }


def instance_of(gw: Gateway, model: str) -> dict[str, Any] | None:
    for row in gw.get("/api/status").get("loaded") or []:
        if row.get("model_id") == model and row.get("state") == "ready":
            return row
    return None


def validity_check(
    gw: Gateway, devices: list[int], own_pids: list[int], phase: str
) -> dict[str, Any]:
    holders = gw.get("/api/vram/holders")
    result = bs.classify_validity(holders, devices, own_pids)
    result["phase"] = phase
    result["at"] = time.time()
    flag = "valid" if result["valid"] else "INVALID"
    print(
        f"[validity/{phase}] {flag}: {len(result['own'])} own, {len(result['offending'])} foreign compute holder(s) on {devices}; desktop processes collapsed: {result.get('desktop_processes_count')}"
    )
    for h in result["offending"]:
        print(
            f"    offending pid={h['pid']} {h['name']} class={h['classification']} alias={h['alias']} bytes={h['used_bytes']} per_gpu={h['per_gpu_bytes']} via={h['attribution']}"
        )
    return result


def load_model(
    gw: Gateway, args: argparse.Namespace, resident: dict[str, Any] | None
) -> dict[str, Any]:
    """Load at the fixed config (agent tier, never force). Returns the load record."""
    wanted = (args.ctx, args.parallel, tuple(sorted(args.device_list)))
    if resident is not None:
        have = bs.plan_tuple(resident.get("plan")) or ()
        if have[:3] == wanted:
            print(
                f"[load] already resident at the requested shape (priority {resident.get('priority')}); reusing, no reload"
            )
            return {
                "performed": False,
                "reused_resident": True,
                "wall_s": None,
                "previous_instance": resident,
            }
        print(
            f"[load] resident at a different shape {have} (priority {resident.get('priority')}); unloading it first (no force is ever passed)"
        )
        gw.post(model_path(args.model, "/unload"), {})
    body = {
        "ctx_size": args.ctx,
        "parallel": args.parallel,
        "devices": args.device_list,
        "priority": bs.TIER_AGENT,
    }
    print(f"[load] POST {model_path(args.model, '/load')} {json.dumps(body)}")
    started = time.perf_counter()
    instance = gw.post(model_path(args.model, "/load"), body, timeout=LOAD_TIMEOUT_S)
    wall = time.perf_counter() - started
    print(f"[load] ready in {wall:.1f}s: {(instance.get('effective') or {}).get('summary')}")
    return {
        "performed": True,
        "reused_resident": False,
        "wall_s": round(wall, 2),
        "previous_instance": resident,
        "instance": instance,
    }


def count_tokens(gw: Gateway, model: str, text: str) -> tuple[int, str]:
    """Tokens in ``text`` via POST /v1/tokenize (forwarded to llama-server's /tokenize),
    else a one-token completion's ``timings.prompt_n``. Both JIT-load an absent
    model, which is why this only ever runs after the load step."""
    response = gw.client.post(
        "/v1/tokenize",
        json={"model": model, "content": text},
        timeout=httpx.Timeout(600.0, connect=10.0),
    )
    if response.status_code == 200:
        return len(response.json().get("tokens") or []), "tokenize"
    if response.status_code not in (404, 405):
        raise Refused(f"POST /v1/tokenize -> {error_text(response)}")
    probe = gw.post(
        "/v1/completions",
        {
            "model": model,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "cache_prompt": False,
            "priority": bs.TIER_AGENT,
        },
        timeout=600.0,
    )
    timings = probe.get("timings") or {}
    n = int(timings.get("prompt_n") or (probe.get("usage") or {}).get("prompt_tokens") or 0)
    if n <= 0:
        raise Refused(
            "cannot count tokens: no /v1/tokenize route and the probe completion reported no prompt_n"
        )
    return n, "probe-completion"


def fit_prompts(gw: Gateway, args: argparse.Namespace, ctx: int) -> dict[int, dict[str, Any]]:
    """One calibrated prompt body per requested length, never exceeding the clamp."""
    sample_words = 2000
    sample_tokens, method = count_tokens(gw, args.model, bs.build_prompt(sample_words, args.seed))
    ratio = sample_tokens / sample_words
    print(
        f"[prompt] calibration via {method}: {sample_tokens} tokens for {sample_words} words -> {ratio:.3f} tok/word"
    )
    prompts: dict[int, dict[str, Any]] = {}
    for requested in args.length_list:
        clamp = bs.clamp_prompt_length(requested, ctx, args.max_tokens)
        target = int(clamp * (1 - TOKEN_TOLERANCE))  # aim 1% under; accept [clamp-3%, clamp]
        words = bs.estimate_words(target, ratio)
        text, tokens = "", 0
        for _ in range(6):
            text = bs.build_prompt(words, args.seed)
            tokens, _ = count_tokens(gw, args.model, text)
            if int(clamp * (1 - 3 * TOKEN_TOLERANCE)) <= tokens <= clamp:
                break
            words = bs.estimate_words(target, tokens / max(1, words))
        if tokens > clamp:
            raise Refused(
                f"prompt for {requested} fitted to {tokens} tokens, above the clamp {clamp}; refusing to exceed ctx"
            )
        prompts[requested] = {
            "requested": requested,
            "clamp": clamp,
            "target": target,
            "words": words,
            "tokens": tokens,
            "text": text,
            "method": method,
        }
        print(f"[prompt] {requested}: clamp {clamp}, fitted {tokens} tokens ({words} words)")
    return prompts


class VramSampler(threading.Thread):
    """Polls GET /api/gpus every 0.5 s (per-device peak used_bytes) and GET /api/vram/holders
    every 10 s (any compute holder on the target devices that is not the model under test).

    The bracketing snapshots cannot see a co-tenant that arrives after the post-load check
    and leaves before the post-run one: on 2026-09-10 a JIT load shared the 5090 pair with
    the bench instance for three of the four prompt lengths and the file was still "valid".
    """

    def __init__(
        self, gw: Gateway, devices: list[int] | None = None, own_pids: list[int] | None = None
    ) -> None:
        super().__init__(daemon=True)
        self.client = gw.new_client()
        self.stop_event = threading.Event()
        self.peak: dict[str, int] = {}
        self.samples = 0
        self.error: str | None = None
        self.devices = [int(d) for d in (devices or [])]
        self.own_pids = [int(p) for p in (own_pids or [])]
        self.foreign: dict[str, dict[str, Any]] = {}
        self.holders_samples = 0
        self._next_holders = 0.0

    def _sample_holders(self) -> None:
        if not self.devices or time.monotonic() < self._next_holders:
            return
        self._next_holders = time.monotonic() + HOLDERS_SAMPLE_INTERVAL_S
        result = bs.classify_validity(
            self.client.get("/api/vram/holders").json(), self.devices, self.own_pids
        )
        self.holders_samples += 1
        for row in result["offending"]:
            key = str(row.get("pid") or f"{row.get('name')}:{row.get('alias')}")
            if key not in self.foreign:
                self.foreign[key] = {
                    **row,
                    "first_seen_at": time.time(),
                    "holders_sample": self.holders_samples,
                }

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                for gpu in self.client.get("/api/gpus").json().get("gpus") or []:
                    key = str(gpu["index"])
                    self.peak[key] = max(self.peak.get(key, 0), int(gpu.get("used_bytes") or 0))
                self.samples += 1
                self._sample_holders()
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                self.error = str(exc)
            self.stop_event.wait(GPU_SAMPLE_INTERVAL_S)
        self.client.close()


def run_one(
    gw: Gateway,
    args: argparse.Namespace,
    prompt: dict[str, Any],
    run: int,
    own_pids: list[int] | None = None,
) -> dict[str, Any]:
    header = f"[bench run {run} seed {args.seed} len {prompt['requested']}]\n"
    payload = {
        "model": args.model,
        "prompt": header + prompt["text"],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "top_k": 1,
        "seed": args.seed,
        "cache_prompt": False,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "priority": bs.TIER_AGENT,
    }
    row: dict[str, Any] = {
        "prompt_length": prompt["requested"],
        "run": run,
        "warmup": run == 0,
        "prompt_tokens_fitted": prompt["tokens"],
    }
    sampler = VramSampler(gw, args.device_list, own_pids or [])
    sampler.start()
    started = time.perf_counter()
    ttft: float | None = None
    timings: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    try:
        with gw.client.stream(
            "POST",
            "/v1/completions",
            json=payload,
            timeout=httpx.Timeout(RUN_TIMEOUT_S, connect=10.0),
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise Refused(
                    f"POST /v1/completions (run {run}, length {prompt['requested']}) -> {error_text(response)}"
                )
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue  # SSE comments (the gateway's ': prefilling' keep-alives) and blanks
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise Refused(
                        f"in-stream error (run {run}, length {prompt['requested']}): {json.dumps(chunk['error'])}"
                    )
                if ttft is None and any(
                    (c.get("text") or (c.get("delta") or {}).get("content"))
                    for c in chunk.get("choices") or []
                ):
                    ttft = time.perf_counter() - started
                if isinstance(chunk.get("timings"), dict):
                    timings = chunk["timings"]
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
    finally:
        wall = time.perf_counter() - started
        sampler.stop_event.set()
        sampler.join(timeout=5.0)
    row.update(
        {
            "wall_s": round(wall, 4),
            "ttft_s": round(ttft, 4) if ttft is not None else None,
            "peak_vram_bytes": sampler.peak,
            "vram_samples": sampler.samples,
            "usage": usage,
            "foreign_holders_seen": list(sampler.foreign.values()),
            "holders_samples": sampler.holders_samples,
        }
    )
    if timings is None:
        completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        row.update(
            {
                "timings_missing": True,
                "prompt_n": (usage or {}).get("prompt_tokens"),
                "predicted_n": completion_tokens or None,
                "prefill_tps": round(prompt["tokens"] / ttft, 2) if ttft else None,
                "decode_tps": round(completion_tokens / (wall - (ttft or 0)), 2)
                if completion_tokens and wall > (ttft or 0)
                else None,
            }
        )
    else:
        row.update(
            {
                "timings_missing": False,
                "prompt_n": timings.get("prompt_n"),
                "cache_n": timings.get("cache_n"),
                "predicted_n": timings.get("predicted_n"),
                "prefill_tps": timings.get("prompt_per_second"),
                "decode_tps": timings.get("predicted_per_second"),
                "timings": timings,
            }
        )
    expected = prompt["tokens"]
    row["cache_hit_suspected"] = bool(
        (row.get("prompt_n") or 0) < expected * 0.98 or (row.get("cache_n") or 0) > 0
    )
    peak_gib = {k: round(v / 2**30, 2) for k, v in sorted(sampler.peak.items())}
    print(
        f"  run {run}{' (warm-up)' if run == 0 else ''}: prompt_n={row.get('prompt_n')} prefill={row.get('prefill_tps')} tok/s  decode={row.get('decode_tps')} tok/s  ttft={row.get('ttft_s')}s  wall={row['wall_s']}s  peak GiB={peak_gib}{'  CACHE HIT SUSPECTED' if row['cache_hit_suspected'] else ''}{'  FOREIGN HOLDER ON TARGET DEVICES' if sampler.foreign else ''}"
    )
    return row


def restore(
    gw: Gateway,
    args: argparse.Namespace,
    residents: list[dict[str, Any]],
    load_record: dict[str, Any],
) -> list[dict[str, Any]]:
    """Put back what was resident: unload the bench instance, reload each prior shape."""
    actions: list[dict[str, Any]] = []
    previous = {r["model_id"]: r for r in residents}
    if load_record.get("performed"):
        print(f"[restore] unloading the bench instance of {args.model}")
        actions.append(
            {
                "action": "unload",
                "model_id": args.model,
                "result": gw.post(model_path(args.model, "/unload"), {}),
            }
        )
    for (
        model_id,
        row,
    ) in previous.items():  # includes the model under test if it was resident before
        plan = row.get("plan") or {}
        current = instance_of(gw, model_id)
        if current is not None and bs.plan_tuple(current.get("plan")) == bs.plan_tuple(plan):
            actions.append(
                {
                    "action": "skip",
                    "model_id": model_id,
                    "reason": "already resident at the previous shape",
                }
            )
            continue
        body = {
            "ctx_size": plan.get("ctx_size"),
            "parallel": plan.get("parallel"),
            "devices": plan.get("devices"),
            "kv_cache_type": plan.get("kv_cache_type"),
            "kv_cache_type_v": plan.get("kv_cache_type_v"),
            "priority": row.get("priority"),
        }
        body = {k: v for k, v in body.items() if v is not None}
        print(f"[restore] reloading {model_id} with {json.dumps(body)}")
        try:
            gw.post(model_path(model_id, "/load"), body, timeout=LOAD_TIMEOUT_S)
            actions.append({"action": "load", "model_id": model_id, "body": body, "ok": True})
        except Refused as exc:
            print(f"[restore] FAILED for {model_id}: {exc}")
            actions.append(
                {
                    "action": "load",
                    "model_id": model_id,
                    "body": body,
                    "ok": False,
                    "error": str(exc),
                }
            )
    return actions


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _stat(entry: dict[str, Any], metric: str, stat: str) -> str:
    value = (entry.get(metric) or {}).get(stat)
    return "-" if value is None else f"{value:.2f}"


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "\nlength    prefill tok/s (med / p95)   decode tok/s (med / p95)   TTFT s (med / p95)   wall s (med)   peak GiB   n"
    )
    for key in sorted(summary, key=int):
        e = summary[key]
        peak = " ".join(
            f"{d}:{b / 2**30:.1f}" for d, b in sorted(e.get("peak_vram_bytes", {}).items())
        )
        flags = (
            (" CACHE?" if e.get("cache_hit_suspected_runs") else "")
            + (" NO-TIMINGS" if e.get("timings_missing_runs") else "")
            + (" FOREIGN" if e.get("foreign_holder_runs") else "")
        )
        cells = [
            _stat(e, m, s)
            for m, s in (
                ("prefill_tps", "median"),
                ("prefill_tps", "p95"),
                ("decode_tps", "median"),
                ("decode_tps", "p95"),
                ("ttft_s", "median"),
                ("ttft_s", "p95"),
                ("wall_s", "median"),
            )
        ]
        print(
            f"{key:>7}   {cells[0]:>9} / {cells[1]:<9}   {cells[2]:>8} / {cells[3]:<8}   {cells[4]:>7} / {cells[5]:<7}   {cells[6]:>8}   {peak:<12} {e['prefill_tps']['n']}{flags}"
        )


def scrub_personal_paths(value: Any) -> Any:
    """Drop or neutralise anything that names a user profile directory.

    A results file is committed to the repo, and the repo's scrub check refuses a
    Windows user-profile path (a snapshot of `/api/status` carries the child's
    ``log_path`` under the data dir, which lives there on this rig). Keys named
    ``log_path`` are dropped; any other string that starts with the current
    user's home directory is rewritten with a ``<home>`` prefix.
    """
    home = str(Path.home())
    home_alt = home.replace("\\", "/")
    if isinstance(value, dict):
        return {
            key: scrub_personal_paths(item)
            for key, item in value.items()
            if key not in ("log_path",)
        }
    if isinstance(value, list):
        return [scrub_personal_paths(item) for item in value]
    if isinstance(value, str):
        for prefix in (home, home_alt):
            if prefix and value.startswith(prefix):
                return "<home>" + value[len(prefix) :]
    return value


def write_results(
    results_dir: Path, sha: str, args: argparse.Namespace, payload: dict[str, Any]
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = scrub_personal_paths(payload)
    base = bs.result_filename(sha, args.label, args.smoke)
    path, n = results_dir / base, 1
    while path.exists():
        n += 1
        path = results_dir / base.replace(".json", f"-{n}.json")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def print_plan(args: argparse.Namespace) -> None:
    lp = model_path(args.model, "/load")
    print(f"DRY RUN -- nothing is contacted. Server {args.server}, model {args.model}")
    print(
        f"config: ctx {args.ctx}, devices {args.device_list}, parallel {args.parallel}, priority {bs.TIER_AGENT} (agent tier), max_tokens {args.max_tokens}, runs {args.runs} (run 0 discarded), seed {args.seed}, smoke {args.smoke}"
    )
    for requested in args.length_list:
        clamp = bs.clamp_prompt_length(requested, args.ctx, args.max_tokens)
        print(
            f"  prompt {requested}: clamp {clamp}, target {int(clamp * (1 - TOKEN_TOLERANCE))} tokens, ~{bs.estimate_words(clamp, 1.3)} words at the 1.3 tok/word estimate (calibrated live via POST /v1/tokenize)"
        )
    steps = [
        "GET /api/health (ready, busy.active_requests == 0, no load in progress)",
        "GET /api/status (leases, benchmark, queue, resident instances snapshot)",
        "GET /api/models (model in registry)",
        "GET /api/vram/holders + GET /api/gpus (validity, pre-load)",
    ]
    if not args.smoke:
        steps += [
            f"POST {model_path(args.model, '/unload')} (only if resident at a different shape)",
            f"POST {lp} {json.dumps({'ctx_size': args.ctx, 'parallel': args.parallel, 'devices': args.device_list, 'priority': bs.TIER_AGENT})} (timed: load wall)",
        ]
    steps += [
        f"GET {model_path(args.model, '/introspect')} + GET /api/status (effective launch, plan, engine tag)",
        "GET /api/vram/holders (validity, post-load)",
        "POST /v1/tokenize x ~1-4 per length (calibration)",
        f"POST /v1/completions x {args.runs} per length {{stream, cache_prompt false, temperature 0, top_k 1, seed, ignore_eos, max_tokens {args.max_tokens}, priority {bs.TIER_AGENT}}} with GET /api/gpus every {GPU_SAMPLE_INTERVAL_S}s in a thread",
        "GET /api/vram/holders (validity, post-run)",
    ]
    if not args.smoke and not args.no_restore:
        steps += [
            f"POST {model_path(args.model, '/unload')}",
            "POST /api/models/{id}/load for each previously resident model (its plan tuple + priority)",
        ]
    steps.append(
        f"write {Path(args.results_dir) / bs.result_filename('<sha>', args.label, args.smoke)}"
    )
    print("requests, in order:\n  " + "\n  ".join(steps))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print_plan(args)
        return EXIT_OK
    gw = Gateway(args.server, args.api_key, args.label)
    git = git_info(HERE.parent)
    started_at = dt.datetime.now(dt.UTC)
    payload: dict[str, Any] = {
        "schema": 1,
        "smoke": args.smoke,
        "label": args.label,
        "git": git,
        "timestamp": started_at.isoformat(),
        "argv": sys.argv[1:],
    }
    residents: list[dict[str, Any]] = []
    load_record: dict[str, Any] = {"performed": False}
    checks: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    exit_code = EXIT_OK
    try:
        pre = preflight(gw, args)
        residents = pre["residents"]
        payload["server"] = {
            "url": args.server,
            "version": pre["health"].get("version"),
            "engine": (pre["status"].get("engine") or {}).get("tag"),
        }
        payload["gpus"] = gw.get("/api/gpus").get("gpus")
        payload["residents_before"] = [
            {
                k: r.get(k)
                for k in ("model_id", "plan", "priority", "pid", "port", "resolved_engine_tag")
            }
            for r in residents
        ]
        pids_before = [
            r["pid"] for r in residents if r.get("model_id") == args.model and r.get("pid")
        ]
        payload["own_lease"] = pre.get("own_lease")
        checks.append(validity_check(gw, args.device_list, pids_before, "pre-load"))
        if args.smoke:
            load_record = {
                "performed": False,
                "reused_resident": True,
                "wall_s": None,
                "previous_instance": pre["resident_instance"],
            }
        else:
            load_record = load_model(gw, args, pre["resident_instance"])
        instance = instance_of(gw, args.model)
        if instance is None:
            raise Refused(f"{args.model} is not ready after the load step (GET /api/status)")
        introspect = gw.get(model_path(args.model, "/introspect"))
        ctx_per_slot = int(
            ((instance.get("effective") or {}).get("ctx_per_slot"))
            or (instance.get("plan") or {}).get("ctx_size")
            or args.ctx
        )
        payload["config"] = {
            "requested": {
                "model": args.model,
                "ctx_size": args.ctx,
                "parallel": args.parallel,
                "devices": args.device_list,
                "priority": bs.TIER_AGENT,
                "max_tokens": args.max_tokens,
                "runs": args.runs,
                "prompt_lengths": args.length_list,
                "seed": args.seed,
            },
            "effective": instance.get("effective"),
            "plan": instance.get("plan"),
            "launch_args": instance.get("launch_args"),
            "instance_priority": instance.get("priority"),
            "resolved_engine_tag": instance.get("resolved_engine_tag"),
            "introspect_actual": introspect.get("actual"),
            "ctx_per_slot": ctx_per_slot,
        }
        payload["load"] = {k: v for k, v in load_record.items() if k != "instance"}
        checks.append(
            validity_check(
                gw, args.device_list, [instance["pid"]] if instance.get("pid") else [], "post-load"
            )
        )
        prompts = fit_prompts(gw, args, ctx_per_slot)
        for requested in args.length_list:
            print(
                f"\n== prompt length {requested} ({prompts[requested]['tokens']} tokens), {args.runs} runs =="
            )
            for run in range(args.runs):
                runs.append(
                    run_one(
                        gw,
                        args,
                        prompts[requested],
                        run,
                        [instance["pid"]] if instance.get("pid") else [],
                    )
                )
        checks.append(
            validity_check(
                gw, args.device_list, [instance["pid"]] if instance.get("pid") else [], "post-run"
            )
        )
        payload["prompts"] = {
            str(k): {kk: vv for kk, vv in v.items() if kk != "text"} for k, v in prompts.items()
        }
    except Refused as exc:
        print(f"\nREFUSED: {exc}", file=sys.stderr)
        payload["aborted"], exit_code = str(exc), EXIT_REFUSED
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        payload["aborted"], exit_code = "KeyboardInterrupt", EXIT_ERROR
    except httpx.HTTPError as exc:
        print(f"\nHTTP failure talking to {args.server}: {exc!r}", file=sys.stderr)
        payload["aborted"], exit_code = repr(exc), EXIT_ERROR
    finally:
        if not args.smoke and not args.no_restore and (load_record.get("performed") or residents):
            try:
                payload["restore"] = restore(gw, args, residents, load_record)
            except (Refused, httpx.HTTPError) as exc:
                print(f"[restore] FAILED: {exc}", file=sys.stderr)
                payload["restore"] = {"error": str(exc)}
    payload["runs"] = runs
    payload["summary"] = bs.summarize_runs(runs)
    # Decided on the two snapshots that bracket the measurement; the pre-load
    # snapshot is context (an idle background model the load displaced never
    # shared the cards with a run).
    decisive = [c for c in checks if c["phase"] in ("post-load", "post-run")]
    runs_with_foreign = [
        f"{r['prompt_length']}/{r['run']}" for r in runs if r.get("foreign_holders_seen")
    ]
    payload["validity"] = {
        "valid": len(decisive) == 2 and all(c["valid"] for c in decisive) and not runs_with_foreign,
        "checks": checks,
        "runs_with_foreign_holders": runs_with_foreign,
    }
    path = write_results(Path(args.results_dir), git.get("sha", ""), args, payload)
    print(f"\nresults: {path}")
    if runs:
        print_summary(payload["summary"])
    if not payload["validity"]["valid"]:
        print(
            f"\n*** INVALID: a compute holder that is not the model under test sat on the target devices during the runs (see validity.checks; runs {payload['validity']['runs_with_foreign_holders']}) ***"
        )
    if exit_code != EXIT_OK:
        return exit_code
    previous = sorted(
        (p for p in Path(args.results_dir).glob("*.json") if p != path and "-smoke" not in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    if previous and not args.smoke:
        import compare  # noqa: PLC0415 - sibling module

        compare.report(path, previous[-1])
    return EXIT_OK if payload["validity"]["valid"] else EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
