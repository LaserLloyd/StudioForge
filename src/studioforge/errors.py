"""OpenAI-shaped error types.

Every error surfaced by the HTTP API is rendered as::

    {"error": {"message": ..., "type": ..., "code": ..., "param": ...}}

which is what the ``openai`` client expects; it parses this envelope to build
its exception hierarchy. Deviating from it breaks client-side error handling,
so all API errors flow through :class:`StudioForgeError`.
"""

from __future__ import annotations

from typing import Any


class StudioForgeError(Exception):
    """Base class for errors that map onto an OpenAI-shaped HTTP response."""

    status_code: int = 500
    error_type: str = "server_error"
    code: str | None = None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        param: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        """Render the OpenAI error envelope."""
        error: dict[str, Any] = {
            "message": self.message,
            "type": self.error_type,
            "code": self.code,
            "param": self.param,
        }
        if self.details:
            # Extra diagnostic context (planner numbers, suggestions, stderr tail).
            # Additive only -- the four keys above stay exactly where clients expect them.
            error["studioforge"] = self.details
        return {"error": error}


class BadRequestError(StudioForgeError):
    status_code = 400
    error_type = "invalid_request_error"


class AuthError(StudioForgeError):
    status_code = 401
    error_type = "invalid_request_error"
    code = "invalid_api_key"


class ModelNotFoundError(StudioForgeError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"

    def __init__(self, model: str, *, known: list[str] | None = None) -> None:
        msg = f"The model '{model}' does not exist."
        if known:
            preview = ", ".join(sorted(known)[:8])
            msg += f" Known models include: {preview}."
        super().__init__(msg, param="model")


class ModelLoadError(StudioForgeError):
    """llama-server failed to start or never became healthy."""

    status_code = 502
    error_type = "server_error"
    code = "model_load_failed"


class ModelUnloadError(StudioForgeError):
    """An unload could not be verified: the child process is still alive.

    Deliberately an error rather than a silent success. An unload API that
    reports success while the process stays resident (and keeps its CUDA
    context, i.e. all of its VRAM) is the single most expensive lie this
    system can tell: every later load is planned against VRAM that is not
    actually free, so it either fails to fit or OOMs at launch, and nothing in
    the logs points at the unload that never happened.
    """

    status_code = 500
    error_type = "server_error"
    code = "unload_failed"


class InsufficientVramError(StudioForgeError):
    """The model cannot fit in VRAM. GPU-only: there is no CPU fallback."""

    status_code = 507  # Insufficient Storage -- closest standard code for "no room"
    error_type = "server_error"
    code = "insufficient_vram"


class ModelBusyError(StudioForgeError):
    status_code = 503
    error_type = "server_error"
    code = "model_busy"


class LeaseConflictError(StudioForgeError):
    """The devices are held by an existing GPU lease, or by a pinned resident (D43)."""

    status_code = 409
    error_type = "invalid_request_error"
    code = "lease_conflict"


class LeaseNotFoundError(StudioForgeError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "lease_not_found"


class UpstreamError(StudioForgeError):
    """llama-server returned an error or died mid-request."""

    status_code = 502
    error_type = "server_error"
    code = "upstream_error"


class ConfigError(StudioForgeError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_config"


class NotSupportedError(StudioForgeError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "unsupported"
