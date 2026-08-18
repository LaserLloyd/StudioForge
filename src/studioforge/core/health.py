"""Deep health probing: prove that generation works, not that a port answers.

A status endpoint returning 200 proves the process is alive and its HTTP
listener is up. It does not prove the model can generate, and the two come
apart in exactly the case that hurts most: the control channel stayed
"connected" and ``GET /models`` returned 200 for hours while every long stream
died mid-generation. Short streams completed, so nothing looked broken; long
ones were silently cut. The only check that would have caught it is a real
streamed completion.

So the deep probe issues a genuine streamed completion against each ready
instance and asserts that tokens actually arrive **and** that the stream
terminates properly. Two design rules follow from the incident:

* **A probe that cannot fail is worse than no probe.** With nothing loaded the
  result is ``no_models_loaded``, never a pass -- otherwise a green dashboard
  means "we checked nothing".
* **A probe must never wedge the thing it is checking.** Every probe has a hard
  timeout, every failure is recorded rather than raised, and the probe
  deliberately does *not* count as activity: if it did, running it on a
  schedule would keep every model past its TTL forever, and a health check that
  pins VRAM is a bug, not a feature.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, Field

from studioforge.logging import get_logger

log = get_logger(__name__)

#: Tokens to ask for. Enough that streaming has to actually stream (several
#: SSE frames), small enough to be cheap on a schedule.
PROBE_MAX_TOKENS = 8

#: Hard ceiling for one probe, including connect and first token. A loaded
#: model that cannot emit 8 greedy tokens in this long is not healthy.
#: (A ``gateway.deep_probe_timeout_s`` config field would be the natural home
#: for this number -- config.py was off-limits for this change.)
PROBE_TIMEOUT_S = 20.0

PROBE_PROMPT = "Reply with the single word: ok"

PROBE_EMBEDDING_INPUT = "studioforge health probe"


class ProbeResult(BaseModel):
    """Outcome of one model's deep probe."""

    model_id: str
    ok: bool = False
    first_token_ms: float | None = None
    tokens: int = 0
    duration_ms: float | None = None
    error: str | None = None
    kind: str = "chat"
    embedding_dims: int | None = None


class DeepHealth(BaseModel):
    """Aggregate deep-probe verdict."""

    status: str = "no_models_loaded"
    checked: int = 0
    models: list[ProbeResult] = Field(default_factory=list)
    duration_ms: float = 0.0


class _SupervisorLike(Protocol):
    """The slice of the supervisor a probe needs."""

    def base_url(self, model_id: str) -> str | None: ...

    def is_ready(self, model_id: str) -> bool: ...

    def list(self) -> list[Any]: ...


async def probe_model(
    model_id: str,
    base_url: str,
    client: httpx.AsyncClient,
    *,
    kind: str = "chat",
    max_tokens: int = PROBE_MAX_TOKENS,
    timeout_s: float = PROBE_TIMEOUT_S,
) -> ProbeResult:
    """Run one real generation against a loaded model. Never raises.

    Returns a failed :class:`ProbeResult` for anything that goes wrong --
    connection refused, non-200, an empty stream, a stream that stops without
    its terminator, or the hard timeout. A health check that can itself throw
    just moves the outage somewhere less visible.
    """
    started = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            _probe_once(model_id, base_url, client, kind=kind, max_tokens=max_tokens),
            timeout=timeout_s,
        )
    except TimeoutError:
        result = ProbeResult(
            model_id=model_id,
            kind=kind,
            ok=False,
            error=f"probe timed out after {timeout_s:g}s",
        )
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, never raises them
        result = ProbeResult(
            model_id=model_id,
            kind=kind,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    result.duration_ms = round((time.perf_counter() - started) * 1000, 1)
    if not result.ok:
        log.warning("deep_probe_failed", model_id=model_id, error=result.error)
    return result


async def _probe_once(
    model_id: str,
    base_url: str,
    client: httpx.AsyncClient,
    *,
    kind: str,
    max_tokens: int,
) -> ProbeResult:
    if kind == "embedding":
        return await _probe_embedding(model_id, base_url, client)
    return await _probe_stream(model_id, base_url, client, max_tokens=max_tokens)


async def _probe_stream(
    model_id: str,
    base_url: str,
    client: httpx.AsyncClient,
    *,
    max_tokens: int,
) -> ProbeResult:
    """A genuinely streamed completion: the only check the incident would fail.

    ``temperature=0`` so the probe is deterministic and cannot be blamed for
    sampling weirdness. Success requires tokens *and* the ``[DONE]``
    terminator: a stream cut mid-generation looks exactly like a healthy one
    until you notice the sentinel never arrived.
    """
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_ms: float | None = None
    tokens = 0
    saw_done = False

    async with client.stream(
        "POST", f"{base_url}/v1/chat/completions", json=payload
    ) as response:
        if response.status_code != 200:
            body = await response.aread()
            return ProbeResult(
                model_id=model_id,
                ok=False,
                error=(
                    f"HTTP {response.status_code} from the model server: "
                    f"{body.decode('utf-8', 'replace')[:300]}"
                ),
            )
        async for line in response.aiter_lines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                saw_done = True
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content:
                    tokens += 1
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000, 1)

    if tokens == 0:
        return ProbeResult(
            model_id=model_id,
            ok=False,
            first_token_ms=first_token_ms,
            tokens=0,
            error="the stream produced no content: the model accepted the request but "
            "generated nothing",
        )
    if not saw_done:
        return ProbeResult(
            model_id=model_id,
            ok=False,
            first_token_ms=first_token_ms,
            tokens=tokens,
            error=f"the stream stopped after {tokens} token(s) without its [DONE] "
            "terminator: generation was cut short",
        )
    return ProbeResult(
        model_id=model_id,
        ok=True,
        first_token_ms=first_token_ms,
        tokens=tokens,
    )


async def _probe_embedding(
    model_id: str, base_url: str, client: httpx.AsyncClient
) -> ProbeResult:
    """Embedding models do not stream; a vector with dimensions is the proof."""
    started = time.perf_counter()
    response = await client.post(
        f"{base_url}/v1/embeddings", json={"input": PROBE_EMBEDDING_INPUT}
    )
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    if response.status_code != 200:
        return ProbeResult(
            model_id=model_id,
            kind="embedding",
            ok=False,
            error=f"HTTP {response.status_code} from the model server",
        )
    vectors = (response.json() or {}).get("data") or []
    dims = len(vectors[0].get("embedding") or []) if vectors else 0
    return ProbeResult(
        model_id=model_id,
        kind="embedding",
        ok=dims > 0,
        first_token_ms=elapsed if dims > 0 else None,
        embedding_dims=dims,
        error=None if dims > 0 else "the embeddings response contained no vector",
    )


def _model_kind(registry: Any, model_id: str) -> str:
    if registry is None:
        return "chat"
    try:
        record = registry.get(model_id) or registry.resolve(model_id)
    except Exception:  # pragma: no cover - registry must not break a probe
        return "chat"
    return str(getattr(record, "kind", "chat")) if record is not None else "chat"


async def deep_health(
    supervisor: _SupervisorLike,
    client: httpx.AsyncClient,
    *,
    registry: Any = None,
    model_ids: list[str] | None = None,
    timeout_s: float = PROBE_TIMEOUT_S,
    max_tokens: int = PROBE_MAX_TOKENS,
) -> DeepHealth:
    """Probe every ready instance (or the named ones) and summarise.

    ``no_models_loaded`` is a distinct status on purpose. Reporting "ok"
    because there was nothing to check is how a monitoring dashboard ends up
    green while the server serves nothing at all.

    Probes run concurrently: they are short, independent, and a serial sweep
    over several resident models would make the endpoint too slow to schedule.
    """
    started = time.perf_counter()
    candidates = (
        model_ids
        if model_ids is not None
        else [i.model_id for i in supervisor.list() if getattr(i, "state", None) == "ready"]
    )

    results: list[ProbeResult] = []
    tasks: list[Any] = []
    for model_id in candidates:
        base = supervisor.base_url(model_id)
        if base is None or not supervisor.is_ready(model_id):
            results.append(
                ProbeResult(
                    model_id=model_id,
                    ok=False,
                    error="model is not loaded, so there is nothing to probe",
                )
            )
            continue
        tasks.append(
            probe_model(
                model_id,
                base,
                client,
                kind=_model_kind(registry, model_id),
                max_tokens=max_tokens,
                timeout_s=timeout_s,
            )
        )
    if tasks:
        results.extend(await asyncio.gather(*tasks))

    results.sort(key=lambda r: r.model_id)
    if not results:
        status = "no_models_loaded"
    elif all(r.ok for r in results):
        status = "ok"
    else:
        status = "degraded"
    return DeepHealth(
        status=status,
        checked=len(results),
        models=results,
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )
