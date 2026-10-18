"""The ``studioforge`` CLI's client for the server running on this machine.

``studioforge ls``, ``ps``, ``load``, ``unload``, ``get``, ``update`` and
``log stream`` are thin clients of the local HTTP API -- the same routes the
panel and ``sfctl`` use -- so they act on the *running* service (its loaded
models, its queue) instead of building a second stack in the CLI process.

Authentication is automatic and local-only:

1. ``SF_CLI_API_KEY`` if set (an explicit key);
2. else the admin key in ``<data-dir>/cli.key`` if there is one;
3. and if the server answers 401/403, one is minted straight in the database
   (the CLI runs on the machine that owns the data directory, so anyone who can
   run it can already read the database) and the call is retried once.

A loopback install with no keys needs none of this: the first attempt succeeds
without a credential.

This deliberately mirrors the small subset of ``sfctl``'s client that the
server CLI needs rather than importing it: ``sfctl`` ships as a separate package
(``packages/studioforge-companion``) for machines that have no server, and the
server must not depend on it.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import httpx

from studioforge.authkeys import is_loopback_host
from studioforge.config import Config
from studioforge.tlsutil import scheme_for

__all__ = ["CliError", "LocalClient", "base_url_for"]

ENV_KEY = "SF_CLI_API_KEY"


class CliError(Exception):
    """A failure to show the user as one readable line (plus its fix)."""

    def __init__(self, message: str, fix: str = "", status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.fix = fix
        self.status = status

    def __str__(self) -> str:
        return f"{self.message}\n  fix: {self.fix}" if self.fix else self.message


def base_url_for(config: Config) -> str:
    """Where the gateway answers *from this machine*."""
    host = config.server.host.strip().strip("[]")
    if not is_loopback_host(host) or host in {"localhost"}:
        host = "127.0.0.1"
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme_for(config)}://{host}:{config.server.port}"


def _error_from(response: httpx.Response) -> CliError:
    message = f"HTTP {response.status_code}"
    fix = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = str(error.get("message") or message)
            fix = str(error.get("fix") or "")
            code = error.get("code")
            if code:
                message = f"[{code}] {message}"
    return CliError(message, fix, response.status_code)


class LocalClient:
    """Blocking HTTP client for ``127.0.0.1:<server.port>``."""

    def __init__(
        self,
        config: Config,
        *,
        timeout_s: float = 30.0,
        key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.base_url = base_url_for(config)
        self._explicit_key = key
        self._key: str | None = None
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            verify=scheme_for(config) == "http",
            transport=transport,
        )

    # -- credentials -------------------------------------------------------

    def _known_key(self) -> str | None:
        if self._key:
            return self._key
        if self._explicit_key:
            return self._explicit_key
        env = os.environ.get(ENV_KEY, "").strip()
        if env:
            return env
        try:
            token = (self.config.data_dir / "cli.key").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return token or None

    def _mint_key(self) -> str:
        from studioforge.core.accounts import AccountStore
        from studioforge.db import Database

        db = Database(self.config.db_path)
        try:
            db.migrate_with_recovery()
            return AccountStore(db, self.config.data_dir).local_cli_key()
        finally:
            db.close()

    # -- requests ------------------------------------------------------------

    def _send(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}) or {})
        key = self._known_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        try:
            return self._http.request(method, path, headers=headers, **kwargs)
        except httpx.ConnectError as exc:
            raise CliError(
                f"StudioForge is not running at {self.base_url}.",
                "Start it with `studioforge server start` (or `studioforge serve`).",
            ) from exc
        except httpx.TimeoutException as exc:
            raise CliError(
                f"StudioForge at {self.base_url} did not answer in time.",
                "Check `studioforge doctor`, or raise the wait with --timeout.",
            ) from exc

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
        raw: bool = False,
    ) -> Any:
        """Call the API; returns parsed JSON (or the response with ``raw=True``)."""
        kwargs: dict[str, Any] = {}
        if json is not None:
            kwargs["json"] = json
        if params:
            kwargs["params"] = {k: v for k, v in params.items() if v is not None}
        response = self._send(method, path, **kwargs)
        if response.status_code in {401, 403} and self._key is None:
            body: Any = {}
            with contextlib.suppress(ValueError):
                body = response.json()
            code = (body.get("error") or {}).get("code") if isinstance(body, dict) else None
            if code in {
                "invalid_api_key",
                "invalid_mcp_pin",
                "remote_admin_requires_credential",
                "api_key_revoked",
                "api_key_expired",
            }:
                try:
                    self._key = self._mint_key()
                except Exception as exc:  # noqa: BLE001
                    raise CliError(
                        f"The server requires a key and one could not be created: {exc}",
                        f"Run `studioforge keys create --scope admin` and export it as {ENV_KEY}.",
                    ) from exc
                response = self._send(method, path, **kwargs)
        if response.status_code >= 400:
            raise _error_from(response)
        if raw:
            return response
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def get(self, path: str, **params: Any) -> Any:
        return self.request("GET", path, params=params)

    def post(self, path: str, json: Any = None, **params: Any) -> Any:
        return self.request("POST", path, json=json if json is not None else {}, params=params)

    def delete(self, path: str, **params: Any) -> Any:
        return self.request("DELETE", path, params=params)

    def is_up(self) -> bool:
        try:
            return self._http.get("/health", timeout=3.0).status_code < 500
        except httpx.HTTPError:
            return False

    def log_lines(self, n: int = 200) -> Iterator[str]:
        """The tail of the server log as plain lines (already path-redacted for non-admins)."""
        payload = self.get("/api/logs", n=n)
        for line in payload.get("lines", []):
            if isinstance(line, dict):
                yield str(line.get("message") or line.get("msg") or line)
            else:
                yield str(line)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> LocalClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
