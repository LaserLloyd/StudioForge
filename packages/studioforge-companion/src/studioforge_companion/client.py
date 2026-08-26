"""Async HTTP client for the StudioForge management + inference API.

Two things this module exists to get right.

**The exit-code contract.** ``sfctl`` is meant to be scriptable, so failures are
typed rather than stringly. Every exception here carries ``exit_code``, and the
CLI's single handler turns that into the process status:

===== =====================================================================
  0   success
  1   API error -- the server answered and refused (unknown model, bad flag,
      won't fit in VRAM)
  2   usage error -- bad arguments or bad local config (Typer also uses 2)
  3   confirmation required -- a destructive command needed ``--yes`` and
      stdin was not a terminal
  4   server unreachable -- connection refused, DNS failure, timeout
  5   auth failed -- missing or wrong API key
===== =====================================================================

**Degrading cleanly.** A wedged or absent server is the *normal* failure, not an
exceptional one, so it must produce one short line naming the URL and pointing
at ``sfctl recover`` (which talks to the watchdog on a different port and
therefore still works). A traceback here is a bug.

Server errors arrive in the OpenAI envelope
``{"error": {"message", "type", "code", "param", "studioforge": {...}}}``. The
``studioforge`` block is where a VRAM rejection puts its ``suggestions`` list --
actionable advice such as "try ctx 8192" -- so it is parsed out and preserved
rather than flattened into the message.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from types import TracebackType
from typing import Any
from urllib.parse import quote

import httpx

from studioforge_companion.config import ServerProfile

EXIT_OK = 0
EXIT_API_ERROR = 1
EXIT_USAGE = 2
EXIT_CONFIRM = 3
EXIT_UNREACHABLE = 4
EXIT_AUTH = 5
#: A human was asked and said no. Distinct from EXIT_CONFIRM on purpose: one is
#: "you did not pass --yes and there is no terminal to ask", which a wrapper
#: can fix by passing the flag, and the other is a refusal, which it must not.
EXIT_DECLINED = 6

#: Rendered into ``sfctl --help`` so the contract is discoverable from the shell.
EXIT_CODE_TABLE: tuple[tuple[int, str], ...] = (
    (EXIT_OK, "success"),
    (EXIT_API_ERROR, "API error (the server refused the request)"),
    (EXIT_USAGE, "usage error (bad arguments or bad local config)"),
    (EXIT_CONFIRM, "confirmation required (destructive command, no --yes, no tty)"),
    (EXIT_UNREACHABLE, "server unreachable (refused / timed out / DNS)"),
    (EXIT_AUTH, "auth failed (missing or wrong API key)"),
    (EXIT_DECLINED, "declined at the confirmation prompt"),
)


def _path_segment(model: str) -> str:
    """URL-encode a model id for a path.

    StudioForge ids are ``publisher/repo/file-stem``, so the slashes are real
    path structure and must survive -- but a ``?`` or ``#`` in an id silently
    truncated the request into a different endpoint, and ``..`` resolved
    somewhere else entirely once httpx normalised dot-segments.
    """
    return quote(model, safe="/")


class CompanionError(Exception):
    """Base class for anything the CLI should report as a clean failure."""

    exit_code: int = EXIT_API_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class ServerUnreachable(CompanionError):
    """No HTTP conversation happened at all."""

    exit_code = EXIT_UNREACHABLE


class AuthFailed(CompanionError):
    """The server answered 401."""

    exit_code = EXIT_AUTH


class ApiError(CompanionError):
    """The server answered with an OpenAI-shaped error envelope."""

    exit_code = EXIT_API_ERROR

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        error_type: str | None = None,
        param: str | None = None,
        status_code: int | None = None,
        suggestions: list[str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type
        self.param = param
        self.status_code = status_code
        self.suggestions: list[str] = suggestions or []
        self.details: dict[str, Any] = details or {}


class ConfirmationRequired(CompanionError):
    """A destructive command needs an explicit ``--yes`` in a non-interactive run."""

    exit_code = EXIT_CONFIRM


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values; httpx would otherwise serialise them as 'None'."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        out[key] = str(value).lower() if isinstance(value, bool) else value
    return out


class StudioForgeClient:
    """Thin async wrapper over the server's HTTP surface.

    Path convention: a path starting with ``/`` is resolved against the server
    root (``/health``, ``/v1/chat/completions``); anything else is resolved
    against ``/api`` (``status``, ``models/x/load``). Management calls are the
    overwhelming majority, so they get the terse form.
    """

    def __init__(self, profile: ServerProfile) -> None:
        self.profile = profile
        self._client: httpx.AsyncClient | None = None
        #: Usage block from the last streamed completion, when the server sent one.
        self.last_usage: dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------

    def _build(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self.profile.auth_headers(),
            # Short connect timeout so an unreachable rig fails in seconds, long
            # read timeout because loading a large model legitimately blocks.
            timeout=httpx.Timeout(self.profile.timeout_s, connect=6.0),
            follow_redirects=True,
        )

    async def __aenter__(self) -> StudioForgeClient:
        self._client = self._build()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._build()
        return self._client

    # -- plumbing ----------------------------------------------------------

    def url_for(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if path.startswith("/"):
            return f"{self.profile.url}{path}"
        return f"{self.profile.api_base}/{path.lstrip('/')}"

    def _unreachable(self, exc: Exception) -> ServerUnreachable:
        """One short, actionable line. Never a traceback."""
        detail = str(exc) or exc.__class__.__name__
        return ServerUnreachable(
            f"Cannot reach the StudioForge server at {self.profile.url} ({detail}).\n"
            f"  - is the server running?  try: sfctl recover\n"
            f"    (that talks to the watchdog on {self.profile.effective_watchdog_url}, "
            f"which answers even when the main server is wedged)\n"
            f"  - if the rig is remote, check the tailnet/VPN is up and the host resolves"
        )

    @staticmethod
    def _error_from(response: httpx.Response) -> CompanionError:
        """Turn a non-2xx response into a typed error, keeping suggestions."""
        body: Any = None
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError):
            body = None

        envelope = body.get("error") if isinstance(body, dict) else None
        if isinstance(envelope, dict):
            message = str(envelope.get("message") or f"HTTP {response.status_code}")
            code = envelope.get("code")
            error_type = envelope.get("type")
            param = envelope.get("param")
            diagnostics = envelope.get("studioforge")
        else:
            text = (response.text or "").strip()
            message = text[:800] if text else f"HTTP {response.status_code}"
            code = error_type = param = None
            diagnostics = None

        details = diagnostics if isinstance(diagnostics, dict) else {}
        raw_suggestions = details.get("suggestions") if details else None
        suggestions = [str(s) for s in raw_suggestions] if isinstance(raw_suggestions, list) else []

        # A 503 carries a wait, in the header and again in the diagnostics
        # block. Both were dropped, so "busy, try again in 10 minutes" reached
        # the caller as an unexplained failure -- the shape most likely to be
        # hit while a GPU lease is held.
        retry_after = details.get("retry_after_s") if details else None
        if retry_after is None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    retry_after = float(header)
                except ValueError:  # HTTP-date form; a wait we cannot use
                    retry_after = None
        if retry_after is not None:
            details = {**details, "retry_after_s": retry_after}
            wait = int(float(retry_after))
            message = f"{message} (the server asked for {wait}s before retrying)"
            suggestions = [*suggestions, f"wait {wait}s and run the same command again"]

        if response.status_code in (401, 403):
            return AuthFailed(
                f"{message}\n  (server: {response.request.url.host}; "
                f"set the key locally with 'sfctl servers add <name> <url> --api-key <key>' "
                f"or 'sfctl config-local set', or set SF_API_KEY in the environment)"
            )
        return ApiError(
            message,
            code=str(code) if code else None,
            error_type=str(error_type) if error_type else None,
            param=str(param) if param else None,
            status_code=response.status_code,
            suggestions=suggestions,
            details=details,
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        try:
            response = await self.http.request(
                method,
                self.url_for(path),
                json=json_body,
                params=_clean_params(params or {}),
            )
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unreachable(exc) from None
        except httpx.TimeoutException as exc:
            raise ServerUnreachable(
                f"Timed out after {self.profile.timeout_s:.0f}s talking to "
                f"{self.profile.url} ({exc.__class__.__name__}). The server may be "
                f"loading a model, or wedged -- try: sfctl recover"
            ) from None
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from None

        if response.status_code >= 400:
            raise self._error_from(response)
        if not response.content:
            return None
        try:
            return response.json()
        except (json.JSONDecodeError, ValueError):
            # Returning the raw text let callers do .get() on a str, which
            # escaped as an AttributeError traceback -- and "a traceback here
            # is a bug" is this module's own contract. A 200 of HTML is a
            # captive portal or a proxy, not this server.
            snippet = (response.text or "").strip()[:200]
            raise CompanionError(
                f"expected JSON from {response.request.url} but got "
                f"{response.headers.get('content-type') or 'an unparseable body'}"
                + (f": {snippet}" if snippet else "")
            ) from None

    async def get(self, path: str, **params: Any) -> Any:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json: Any = None, **params: Any) -> Any:
        return await self.request("POST", path, json_body=json, params=params)

    async def put(self, path: str, json: Any = None, **params: Any) -> Any:
        return await self.request("PUT", path, json_body=json, params=params)

    async def patch(self, path: str, json: Any = None, **params: Any) -> Any:
        return await self.request("PATCH", path, json_body=json, params=params)

    async def delete(self, path: str, **params: Any) -> Any:
        return await self.request("DELETE", path, params=params)

    async def stream_lines(
        self, method: str, path: str, json: Any = None, **params: Any
    ) -> AsyncIterator[str]:
        """Yield response lines as they arrive (SSE and NDJSON both fit)."""
        try:
            async with self.http.stream(
                method,
                self.url_for(path),
                json=json,
                params=_clean_params(params),
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise self._error_from(response)
                async for line in response.aiter_lines():
                    yield line
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise self._unreachable(exc) from None
        except httpx.TimeoutException as exc:
            raise ServerUnreachable(
                f"Stream from {self.profile.url} timed out ({exc.__class__.__name__})"
            ) from None
        except httpx.HTTPError as exc:
            raise self._unreachable(exc) from None

    # -- management conveniences ------------------------------------------

    async def health(self) -> Any:
        """Unauthenticated liveness probe (``/health`` is a public path)."""
        return await self.get("/health")

    async def status(self) -> Any:
        return await self.get("status")

    async def models(self) -> Any:
        return await self.get("models")

    async def gpus(self) -> Any:
        return await self.get("gpus")

    async def engine(self) -> Any:
        return await self.get("engine")

    async def engine_releases(self, limit: int = 20) -> Any:
        return await self.get("engine/releases", limit=limit)

    async def engine_install(self, tag: str, *, force: bool = False) -> Any:
        return await self.post("engine/install", {"tag": tag, "force": force})

    async def version(self) -> Any:
        return await self.get("version")

    async def load(
        self,
        model: str,
        *,
        ctx_size: int | None = None,
        kv_cache_type: str | None = None,
        parallel: int | None = None,
        force: bool = False,
        priority: int | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "ctx_size": ctx_size,
            "kv_cache_type": kv_cache_type,
            "parallel": parallel,
            "force": force,
        }
        if priority is not None:
            body["priority"] = priority
        return await self.post(f"models/{_path_segment(model)}/load", body)

    async def unload(self, model: str) -> Any:
        return await self.post(f"models/{_path_segment(model)}/unload")

    async def test(self, model: str, prompt: str | None = None) -> Any:
        return await self.post(f"models/{_path_segment(model)}/test", {"prompt": prompt})

    async def pin(self, model: str, pinned: bool) -> Any:
        return await self.post(f"models/{_path_segment(model)}/pin", {"pinned": pinned})

    async def delete_model(self, model: str, *, delete_files: bool, confirm: bool) -> Any:
        return await self.delete(
            f"models/{_path_segment(model)}", delete_files=delete_files, confirm=confirm
        )

    async def settings(self, model: str) -> Any:
        return await self.get(f"models/{_path_segment(model)}/settings")

    async def put_settings(self, model: str, settings: dict[str, Any]) -> Any:
        return await self.put(f"models/{_path_segment(model)}/settings", settings)

    async def plan(
        self,
        model: str,
        *,
        ctx_size: int | None = None,
        kv_cache_type: str | None = None,
        parallel: int | None = None,
    ) -> Any:
        return await self.get(
            f"models/{_path_segment(model)}/plan",
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            parallel=parallel,
        )

    async def introspect(self, model: str) -> Any:
        return await self.get(f"models/{_path_segment(model)}/introspect")

    async def placement_profiles(self, model: str) -> Any:
        """Best achievable load per hardware mode -- the ``model_options`` table."""
        return await self.get(f"models/{_path_segment(model)}/profiles")

    async def load_recommended(
        self,
        model: str,
        *,
        ctx_size: int,
        prefer_mode: str | None = None,
        kv_min: str | None = None,
        priority: int | None = None,
    ) -> Any:
        """Load at exactly ``ctx_size`` per slot, or refuse with a 507.

        The distinction from :meth:`load` is the whole point: this one either
        gives the window that was asked for or says it cannot, instead of
        quietly shrinking it.
        """
        body: dict[str, Any] = {"ctx_size": ctx_size}
        if prefer_mode is not None:
            body["prefer_mode"] = prefer_mode
        if kv_min is not None:
            body["kv_min"] = kv_min
        if priority is not None:
            body["priority"] = priority
        return await self.post(f"models/{_path_segment(model)}/load-recommended", body)

    # -- GPU leases (D43) --------------------------------------------------
    #
    # A co-tenant that cannot SEE a standing lease will keep trying to plan
    # onto held cards and read the refusals as a broken rig, so the read side
    # matters at least as much as the write side.

    async def leases(self) -> Any:
        return await self.get("leases")

    async def create_lease(
        self,
        *,
        devices: list[int],
        model_ids: list[str] | None = None,
        holder: str = "sfctl",
        reason: str = "",
        idle_ttl_s: float | None = 3600.0,
        force: bool = False,
    ) -> Any:
        return await self.post(
            "leases",
            {
                "devices": devices,
                "model_ids": model_ids,
                "holder": holder,
                "reason": reason,
                "idle_ttl_s": idle_ttl_s,
                "force": force,
            },
        )

    async def release_lease(self, lease_id: str) -> Any:
        return await self.delete(f"leases/{_path_segment(lease_id)}")

    async def touch_lease(self, lease_id: str) -> Any:
        return await self.post(f"leases/{_path_segment(lease_id)}/touch")

    # -- Hugging Face -------------------------------------------------------

    async def hf_search(
        self,
        query: str,
        *,
        limit: int = 20,
        author: str | None = None,
        sort: str = "downloads",
        newer_than_days: int | None = None,
        with_context: bool = False,
    ) -> Any:
        return await self.get(
            "hf/search",
            q=query,
            limit=limit,
            author=author,
            sort=sort,
            newer_than_days=newer_than_days,
            with_context=with_context,
        )

    async def hf_repo(self, repo_id: str) -> Any:
        """One repo's quants with a fit verdict and a per-GPU-set context matrix."""
        return await self.get(f"hf/repo/{_path_segment(repo_id)}")

    async def logs(self, n: int = 200, level: str | None = None) -> Any:
        return await self.get("logs", n=n, level=level)

    async def model_logs(self, model: str, n: int = 200) -> Any:
        return await self.get(f"logs/models/{model}", n=n)

    async def get_config(self) -> Any:
        return await self.get("config")

    async def set_config(self, updates: dict[str, Any]) -> Any:
        return await self.patch("config", updates)

    async def scan(self, force: bool = False) -> Any:
        return await self.post("models/scan", None, force=force)

    async def openclaw_setup(self) -> Any:
        return await self.get("openclaw-setup")

    # -- downloads ---------------------------------------------------------
    #
    # The download plane is optional server-side (``state.downloader`` may be
    # None, in which case every route answers 400 ``downloads_unavailable``), so
    # callers must be ready for "this server cannot download" as a normal answer.

    async def start_download(
        self,
        repo_id: str,
        *,
        quant: str | None = None,
        include_mmproj: bool = True,
        force: bool = False,
    ) -> Any:
        return await self.post(
            "downloads",
            {
                "repo_id": repo_id,
                "quant": quant,
                "include_mmproj": include_mmproj,
                "force": force,
            },
        )

    async def downloads(self) -> Any:
        return await self.get("downloads")

    # -- self-update -------------------------------------------------------

    async def update_status(self, *, check: bool = False) -> Any:
        """Current vs latest. ``check`` makes the server query GitHub."""
        return await self.get("update", check=check)

    async def update_releases(self, limit: int = 20) -> Any:
        return await self.get("update/releases", limit=limit)

    async def update_install(
        self, tag: str | None = None, *, confirm: bool, restart: bool = True
    ) -> Any:
        return await self.post(
            "update/install", {"tag": tag, "confirm": confirm, "restart": restart}
        )

    async def update_rollback(self, *, confirm: bool) -> Any:
        return await self.post("update/rollback", {"confirm": confirm})

    # -- inference ---------------------------------------------------------

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> AsyncIterator[str]:
        """Stream assistant text from ``POST /v1/chat/completions``.

        Deliberately goes through the OpenAI endpoint rather than a management
        route: that exercises the same just-in-time load path a real client
        would hit, so ``sfctl chat`` doubles as an end-to-end check.

        Token counts reported by the server land in :attr:`last_usage`.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **{k: v for k, v in extra.items() if v is not None},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.last_usage = {}
        async for line in self.stream_lines("POST", "/v1/chat/completions", json=payload):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                self.last_usage = usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield str(piece)
