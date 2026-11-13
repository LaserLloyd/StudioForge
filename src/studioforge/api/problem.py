"""RFC 9457 ``application/problem+json`` on the management routes, by negotiation (D90).

Management routes (``/api/*`` except the LM Studio compatibility trees, and the
router-vocabulary ``/models/*``) answer an error as a problem document when the
caller sends ``Accept: application/problem+json``. Without that header the body
is exactly what it always was (the OpenAI-shaped ``{"error": {...}}``), plus a
``Deprecation`` marker: the default flips only after the bench team confirms
every caller reads the new shape (lead decision F-9).

During the window the problem body ALSO embeds the legacy ``error{...}`` object,
so a caller that opts in early and still reads ``error.code`` keeps working::

    {"type": "urn:studioforge:error:<code>", "title": ..., "status": 409,
     "detail": <message>, "instance": "urn:studioforge:request:<id>",
     "code": ..., "fix": ..., "docs": ..., "retry_after_s": ..., "param": ...,
     "studioforge": {...}, "error": {...legacy...}}

Inference routes (``/v1/*``, the llama-server natives) never negotiate: they keep
the OpenAI / Anthropic / llama-server error shapes their clients parse.
"""

from __future__ import annotations

import http
from typing import Any

from fastapi.responses import JSONResponse

from studioforge.api.identity import deprecation_headers, request_id_of

PROBLEM_CONTENT_TYPE = "application/problem+json"

#: LM Studio's native trees keep LM Studio's error shape.
_NOT_MANAGEMENT_PREFIXES = ("/api/v0/", "/api/v1/")


def is_management_path(path: str) -> bool:
    """Whether ``path`` is a management route (negotiates problem+json)."""
    if path.startswith(_NOT_MANAGEMENT_PREFIXES):
        return False
    return path.startswith("/api/") or path == "/models" or path.startswith("/models/")


def accepts_problem(accept: str | None) -> bool:
    """Whether an ``Accept`` header lists ``application/problem+json`` with a non-zero weight."""
    if not accept:
        return False
    for part in accept.split(","):
        pieces = [p.strip() for p in part.split(";")]
        if pieces[0].lower() != PROBLEM_CONTENT_TYPE:
            continue
        weight = 1.0
        for param in pieces[1:]:
            if param.lower().startswith("q="):
                try:
                    weight = float(param[2:])
                except ValueError:
                    weight = 1.0
        if weight > 0:
            return True
    return False


def _title(code: str | None, status: int) -> str:
    if code:
        return code.replace("_", " ").capitalize()
    try:
        return http.HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def problem_from_envelope(envelope: dict[str, Any], status: int, instance: str) -> dict[str, Any]:
    """The RFC 9457 document for an OpenAI-shaped ``{"error": {...}}`` envelope."""
    error = envelope.get("error") if isinstance(envelope, dict) else None
    error = error if isinstance(error, dict) else {}
    code = error.get("code")
    details = error.get("studioforge")
    details = details if isinstance(details, dict) else {}
    body: dict[str, Any] = {
        "type": f"urn:studioforge:error:{code}" if code else "about:blank",
        "title": _title(code, status),
        "status": status,
        "detail": error.get("message"),
        "instance": f"urn:studioforge:request:{instance}",
        "code": code,
    }
    for key in ("fix", "docs", "param"):
        if error.get(key) is not None:
            body[key] = error[key]
    retry_after = details.get("retry_after_s", error.get("retry_after_s"))
    if retry_after is not None:
        body["retry_after_s"] = retry_after
    if "suggestions" in details:
        body["suggestions"] = details["suggestions"]
    if details:
        body["studioforge"] = details
    # The legacy object, whole, so every `error.*` reader keeps working.
    body["error"] = error
    return body


def error_response(
    request: Any,
    envelope: dict[str, Any],
    status: int,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Render ``envelope`` for ``request``: problem+json when negotiated, else unchanged."""
    out = dict(headers or {})
    path = getattr(getattr(request, "url", None), "path", "")
    if is_management_path(path):
        accept = getattr(request, "headers", {}).get("accept")
        if accepts_problem(accept):
            body = problem_from_envelope(envelope, status, request_id_of(request))
            return JSONResponse(
                body, status_code=status, headers=out, media_type=PROBLEM_CONTENT_TYPE
            )
        # The legacy envelope stays for the window, marked so a client can see it.
        for name, value in deprecation_headers():
            out.setdefault(name, value)
    return JSONResponse(envelope, status_code=status, headers=out)
