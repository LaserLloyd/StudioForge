"""#36: the metrics route label is the matched template, whatever the parameter value."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from studioforge.api.app import _route_label


def test_label_uses_route_template_even_when_value_repeats_substring() -> None:
    seen: list[str] = []
    app = FastAPI()
    router = APIRouter()

    @router.get("/models/{model_id}/load")
    async def load(model_id: str) -> dict[str, str]:
        return {"id": model_id}

    app.include_router(router, prefix="/api")

    class Spy(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):  # type: ignore[no-untyped-def]
            response = await call_next(request)
            seen.append(_route_label(request.scope))
            return response

    app.add_middleware(Spy)
    client = TestClient(app)
    for value in ("a", "api", "models", "load"):
        client.get(f"/api/models/{value}/load")
    client.get("/nope")
    assert seen == ["/api/models/{model_id}/load"] * 4 + ["(unmatched)"]
