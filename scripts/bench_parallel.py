#!/usr/bin/env python3
"""
Concurrency-calibration harness for StudioForge (FastAPI-over-llama.cpp LLM server).

Measures aggregate throughput and latency as a function of client concurrency for
ONE loaded model, cross-checking against llama.cpp's slot accounting. A client that
sends N requests to a 1-slot server still gets N answers, just serialized; only
`llamacpp:n_busy_slots_per_decode` metrics prove batching happened.

PROCEDURE:
1. Quiesce other models (optional: POST /api/models/unload-all beforehand).
2. Load the target model with explicit ctx_size, kv_cache_type, and parallel=N
   via GUI or API (e.g., POST /api/models/load with parameters).
3. Verify total_slots == N using the /props endpoint on the child port.
4. Run concurrency levels (1, 2, 4, 8 by default).
5. (Optional) Repeat with kv_unified and note whether slots[i].n_ctx differs.

INTERPRETATION:
- Aggregate tok/s that plateaus while p95 latency climbs linearly ≈ the knee
  → set the model's `parallel` parameter one level below it.
- achieved_batch ≈ N and busy_slots > 1: batching worked.
- achieved_batch ≈ 1.0 with N > 1: launch used --parallel 1 or requests serialized elsewhere.
"""

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from statistics import median
from typing import Optional

try:
    import httpx
except ImportError:
    print("Error: httpx is required. Install with: pip install httpx", file=sys.stderr)
    sys.exit(1)


@dataclass
class RequestResult:
    """Result of a single request."""
    latency: float
    prompt_tokens: int
    completion_tokens: int
    error: Optional[str] = None


@dataclass
class MetricsSnapshot:
    """Snapshot of prometheus metrics from /metrics endpoint."""
    n_decode_total: float
    n_busy_slots_per_decode: float
    requests_deferred: float
    prompt_tokens_per_second: float
    predicted_tokens_per_second: float

    @classmethod
    def from_text(cls, text: str) -> "MetricsSnapshot":
        """Parse Prometheus text format."""
        metrics = {}
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract metric name and value: name{...} value
            match = re.match(r'llamacpp:(\w+)\s+([\d.e+-]+)', line)
            if match:
                name, value = match.groups()
                try:
                    metrics[name] = float(value)
                except ValueError:
                    pass

        return cls(
            n_decode_total=metrics.get('n_decode_total', 0.0),
            n_busy_slots_per_decode=metrics.get('n_busy_slots_per_decode', 0.0),
            requests_deferred=metrics.get('requests_deferred', 0.0),
            prompt_tokens_per_second=metrics.get('prompt_tokens_seconds', 0.0),
            predicted_tokens_per_second=metrics.get('predicted_tokens_seconds', 0.0),
        )


class StudioForgeBench:
    """Concurrency benchmarking harness."""

    def __init__(self, model_id: str, gateway: str, max_tokens: int,
                 prompt_tokens: int, timeout: float = 600.0):
        self.model_id = model_id
        self.gateway = gateway.rstrip('/')
        self.max_tokens = max_tokens
        self.prompt_tokens = prompt_tokens
        self.timeout = timeout
        self.child_port = None
        self.child_host = None
        self.total_slots = None
        self.n_ctx = None
        self.prompt = self._build_prompt(prompt_tokens)

    def _build_prompt(self, target_tokens: int) -> str:
        """Build a filler prompt of roughly target_tokens."""
        base = "Summarize the following technical text. "
        filler = (
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
            "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
            "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. "
        )
        # Rough estimate: base ~8 tokens, filler ~54 tokens per repeat
        needed = max(target_tokens - len(base.split()), 0)
        repeats = (needed // 54) + 1
        return base + (filler * repeats)

    def discover_child_port(self) -> bool:
        """Discover the child port for the loaded model."""
        try:
            response = httpx.get(
                f"{self.gateway}/api/status",
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

            loaded = data.get('loaded', [])
            for model_entry in loaded:
                if model_entry.get('model_id') == self.model_id:
                    self.child_port = model_entry.get('port')
                    # Extract host from gateway URL or default to 127.0.0.1
                    if '://' in self.gateway:
                        self.child_host = self.gateway.split('://')[1].split(':')[0]
                    else:
                        self.child_host = '127.0.0.1'
                    return True

            print(f"Error: Model '{self.model_id}' not loaded on {self.gateway}")
            print("Load it first via the GUI/API with explicit ctx_size, parallel parameters.")
            return False
        except Exception as e:
            print(f"Error discovering model: {e}", file=sys.stderr)
            return False

    def get_props(self) -> bool:
        """Fetch and print /props from child server."""
        try:
            url = f"http://{self.child_host}:{self.child_port}/props"
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            self.total_slots = data.get('total_slots', 'n/a')
            self.n_ctx = data.get('default_generation_settings', {}).get('n_ctx', 'n/a')

            print(f"[Props] total_slots={self.total_slots}, n_ctx={self.n_ctx}")
            return True
        except Exception as e:
            print(f"Error fetching props: {e}", file=sys.stderr)
            return False

    async def _single_request(self, client: httpx.AsyncClient) -> RequestResult:
        """Execute a single chat completion request."""
        try:
            start = time.time()
            response = await asyncio.wait_for(
                client.post(
                    f"{self.gateway}/v1/chat/completions",
                    json={
                        "model": self.model_id,
                        "messages": [{"role": "user", "content": self.prompt}],
                        "max_tokens": self.max_tokens,
                        "temperature": 0,
                        "seed": 0,
                        "stream": False,
                    },
                    timeout=self.timeout,
                ),
                timeout=self.timeout + 5.0,
            )
            latency = time.time() - start

            response.raise_for_status()
            data = response.json()
            usage = data.get('usage', {})

            return RequestResult(
                latency=latency,
                prompt_tokens=usage.get('prompt_tokens', 0),
                completion_tokens=usage.get('completion_tokens', 0),
            )
        except asyncio.TimeoutError:
            return RequestResult(latency=0, prompt_tokens=0, completion_tokens=0, error="timeout")
        except Exception as e:
            return RequestResult(latency=0, prompt_tokens=0, completion_tokens=0, error=str(e))

    async def run_level(self, concurrency: int, warm_up: bool = True) -> tuple:
        """Run N concurrent requests and collect metrics."""
        try:
            # Warm-up (single request to prime cache)
            if warm_up:
                async with httpx.AsyncClient() as client:
                    await self._single_request(client)
                await asyncio.sleep(0.1)

            # Scrape metrics before
            try:
                metrics_before = await asyncio.to_thread(
                    lambda: httpx.get(
                        f"http://{self.child_host}:{self.child_port}/metrics",
                        timeout=10.0,
                    ).text
                )
                snapshot_before = MetricsSnapshot.from_text(metrics_before)
            except Exception as e:
                print(f"Warning: Failed to scrape metrics before: {e}", file=sys.stderr)
                snapshot_before = MetricsSnapshot(0, 0, 0, 0, 0)

            # Run concurrent requests
            wall_start = time.time()
            async with httpx.AsyncClient() as client:
                results = await asyncio.gather(
                    *[self._single_request(client) for _ in range(concurrency)],
                    return_exceptions=False,
                )
            wall_time = time.time() - wall_start

            # Scrape metrics after
            try:
                metrics_after = await asyncio.to_thread(
                    lambda: httpx.get(
                        f"http://{self.child_host}:{self.child_port}/metrics",
                        timeout=10.0,
                    ).text
                )
                snapshot_after = MetricsSnapshot.from_text(metrics_after)
            except Exception as e:
                print(f"Warning: Failed to scrape metrics after: {e}", file=sys.stderr)
                snapshot_after = MetricsSnapshot(0, 0, 0, 0, 0)

            # Analyze results
            successful = [r for r in results if r.error is None]
            failed = [r for r in results if r.error is not None]

            if not successful:
                print(f"Level {concurrency}: All {len(failed)} requests failed")
                return None

            latencies = [r.latency for r in successful]
            total_completion = sum(r.completion_tokens for r in successful)
            total_prompt = sum(r.prompt_tokens for r in successful)

            p50_lat = median(latencies)
            sorted_lat = sorted(latencies)
            p95_lat = sorted_lat[int(0.95 * len(sorted_lat))] if len(sorted_lat) > 1 else sorted_lat[0]

            agg_tok_per_sec = total_completion / wall_time if wall_time > 0 else 0

            delta_decodes = snapshot_after.n_decode_total - snapshot_before.n_decode_total
            achieved_batch = total_completion / delta_decodes if delta_decodes > 0 else 0

            busy_slots = snapshot_after.n_busy_slots_per_decode
            deferred = snapshot_after.requests_deferred - snapshot_before.requests_deferred

            return {
                "concurrency": concurrency,
                "wall_time": wall_time,
                "successful": len(successful),
                "failed": len(failed),
                "latencies": latencies,
                "p50_latency": p50_lat,
                "p95_latency": p95_lat,
                "total_completion_tokens": total_completion,
                "total_prompt_tokens": total_prompt,
                "agg_tok_per_sec": agg_tok_per_sec,
                "achieved_batch": achieved_batch,
                "busy_slots": busy_slots,
                "requests_deferred": deferred,
            }
        except Exception as e:
            print(f"Error during level {concurrency}: {e}", file=sys.stderr)
            return None


async def main():
    parser = argparse.ArgumentParser(
        description="Concurrency calibration for StudioForge models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "EXAMPLE:\n"
            "  python bench_parallel.py llama-7b --levels 1,2,4,8 --prompt-tokens 512\n"
            "\n"
            "INTERPRETATION:\n"
            "  Aggregate tok/s that plateaus while p95 latency climbs is the knee.\n"
            "  Set model parallel one level below the knee for best throughput.\n"
        )
    )
    parser.add_argument("model_id", help="Model ID to benchmark")
    parser.add_argument(
        "--gateway",
        default="http://127.0.0.1:1234",
        help="StudioForge gateway URL (default: http://127.0.0.1:1234)",
    )
    parser.add_argument(
        "--levels",
        default="1,2,4,8",
        help="Concurrency levels to test (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Target prompt length in tokens (default: 512)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Write results table to JSON file (optional)",
    )
    parser.add_argument(
        "--no-warm",
        action="store_true",
        help="Disable warm-up requests (default: enabled)",
    )

    args = parser.parse_args()

    # Parse levels
    try:
        levels = [int(x.strip()) for x in args.levels.split(',')]
    except ValueError:
        print(f"Error: --levels must be comma-separated integers, got '{args.levels}'")
        sys.exit(1)

    # Create benchmark harness
    bench = StudioForgeBench(
        args.model_id,
        args.gateway,
        args.max_tokens,
        args.prompt_tokens,
    )

    # Discover and validate
    print(f"Discovering model on {args.gateway}...")
    if not bench.discover_child_port():
        sys.exit(1)

    print(f"Model child server: http://{bench.child_host}:{bench.child_port}")

    if not bench.get_props():
        sys.exit(1)

    print(f"Benchmarking with prompt ~{args.prompt_tokens} tokens, max_tokens={args.max_tokens}")
    print()

    # Header
    print(
        f"{'Level':>6} {'Wall(s)':>8} {'P50(s)':>8} {'P95(s)':>8} "
        f"{'Agg Tok/s':>12} {'Achieved':>10} {'Busy':>6} {'Deferred':>8}"
    )
    print("-" * 78)

    results = []
    for level in levels:
        result = await bench.run_level(level, warm_up=not args.no_warm)
        if result:
            print(
                f"{result['concurrency']:>6} "
                f"{result['wall_time']:>8.2f} "
                f"{result['p50_latency']:>8.3f} "
                f"{result['p95_latency']:>8.3f} "
                f"{result['agg_tok_per_sec']:>12.2f} "
                f"{result['achieved_batch']:>10.2f} "
                f"{result['busy_slots']:>6.1f} "
                f"{result['requests_deferred']:>8.0f}"
            )
            results.append(result)
        else:
            print(f"{level:>6} [FAILED]")

    print()
    print("INTERPRETATION:")
    print("  - Aggregate tok/s plateau with climbing p95: the knee point.")
    print("  - achieved_batch ≈ Level and busy_slots > 1: batching worked.")
    print("  - achieved_batch ≈ 1.0 with Level > 1: --parallel 1 or requests serialized.")

    # Write JSON if requested
    if args.json:
        try:
            with open(args.json, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.json}")
        except Exception as e:
            print(f"Error writing JSON: {e}", file=sys.stderr)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(130)
