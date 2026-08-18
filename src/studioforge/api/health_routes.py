"""Deep-probe route, mounted under ``/api``.

Lives in its own module rather than in the management router so the probe
surface stays next to nothing else: it is the one endpoint that deliberately
does real inference work, and it must stay cheap to reason about.

See :mod:`studioforge.core.health` for why a streamed completion is the only
health check that would have caught the incident behind this.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from studioforge.core.health import PROBE_MAX_TOKENS, PROBE_TIMEOUT_S, deep_health
from studioforge.errors import ModelNotFoundError

router = APIRouter()


@router.post("/models/{model_id:path}/probe")
async def probe_model_route(
    model_id: str,
    request: Request,
    timeout_s: float = Query(PROBE_TIMEOUT_S, gt=0, le=300),
    max_tokens: int = Query(PROBE_MAX_TOKENS, gt=0, le=256),
) -> dict[str, Any]:
    """Run one real streamed completion against a loaded model.

    Deliberately does **not** load the model: a probe that loads is a load, and
    it would turn a monitoring poll into minutes of VRAM allocation. An
    unloaded model is reported as such, honestly, instead of being made to pass.
    """
    state = request.app.state
    record = state.registry.resolve(model_id)
    if record is None:
        raise ModelNotFoundError(model_id, known=state.registry.known_ids())
    # A preset-only virtual model is served by its base's instance.
    serving_id = state.manager.serving_record(record).id
    result = await deep_health(
        state.supervisor,
        state.client,
        registry=state.registry,
        model_ids=[serving_id],
        timeout_s=timeout_s,
        max_tokens=max_tokens,
    )
    payload = result.model_dump(mode="json")
    payload["model_id"] = record.id
    payload["serving_model_id"] = serving_id
    return payload
