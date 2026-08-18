"""The watchdog's plain-HTTP recovery route, and the main app's handoff to it.

Regression cover for a restart that reported success and did nothing: the app
posted a bare JSON-RPC ``tools/call`` to the watchdog's streamable-HTTP ``/mcp``
endpoint. That transport requires an ``initialize`` handshake and a session id
first, so the call was rejected -- and the reply was discarded, so
``POST /api/restart/server`` answered ``{"restarting": true}`` while the process
carried on running.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from studioforge.watchdog.server import wrap_asgi


class _FakeWatchdog:
    def __init__(self, *, ok: bool = True) -> None:
        self.ok = ok
        self.restart_calls: list[dict[str, Any]] = []

    async def health(self) -> dict[str, Any]:
        return {"status": "up"}

    async def restart_server(self, **kwargs: Any) -> dict[str, Any]:
        self.restart_calls.append(kwargs)
        return {"ok": self.ok, "method": "kill+respawn", "new_pid": 4242}


async def _inner(scope: dict[str, Any], receive: Any, send: Any) -> None:
    body = b'{"reached":"mcp"}'
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _call(
    app: Any, method: str, path: str, headers: list[tuple[bytes, bytes]] | None = None
) -> tuple[int, dict[str, Any]]:
    scope = {"type": "http", "method": method, "path": path, "headers": headers or []}
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    raw = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(raw)


class TestWatchdogRestartRoute:
    @pytest.mark.asyncio
    async def test_post_restart_restarts_without_any_mcp_handshake(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, api_key=None)

        status, body = await _call(app, "POST", "/restart")

        assert status == 200
        assert body["ok"] is True
        # Confirmed on our behalf: reaching this route is the confirmation.
        assert watchdog.restart_calls == [{"confirm": True}]

    @pytest.mark.asyncio
    async def test_failed_restart_reports_500_not_a_cheerful_200(self) -> None:
        watchdog = _FakeWatchdog(ok=False)
        app = wrap_asgi(_inner, watchdog, api_key=None)

        status, body = await _call(app, "POST", "/restart")

        assert status == 500
        assert body["ok"] is False

    @pytest.mark.asyncio
    async def test_restart_needs_the_key_when_one_is_set(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, api_key="s3cret")

        status, body = await _call(app, "POST", "/restart")
        assert status == 401
        assert watchdog.restart_calls == []

        status, body = await _call(app, "POST", "/restart", [(b"authorization", b"Bearer s3cret")])
        assert status == 200
        assert watchdog.restart_calls == [{"confirm": True}]

    @pytest.mark.asyncio
    async def test_health_stays_open_and_get_restart_falls_through(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, api_key=None)

        status, body = await _call(app, "GET", "/health")
        assert status == 200 and body["status"] == "up"

        # Only POST is the recovery verb; a GET is not a restart trigger.
        status, body = await _call(app, "GET", "/restart")
        assert body == {"reached": "mcp"}
        assert watchdog.restart_calls == []


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeClient:
    """Records the handoff request the app makes to the watchdog."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]], response: _Response) -> None:
        self._calls = calls
        self._response = response

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> _Response:
        self._calls.append((url, kwargs))
        return self._response


class TestAppHandoff:
    @pytest.mark.asyncio
    async def test_handoff_posts_to_the_plain_route_not_mcp(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from studioforge.api import admin_routes

        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx,
            "AsyncClient",
            lambda **_: _FakeClient(calls, _Response(200)),
        )
        fell_back: list[bool] = []

        async def _fake_self_restart(_state: Any) -> None:
            fell_back.append(True)

        monkeypatch.setattr(admin_routes, "_self_restart", _fake_self_restart)
        monkeypatch.setattr(admin_routes.asyncio, "sleep", _noop_sleep)

        state = _state_with_key(None)
        await admin_routes._ask_watchdog_to_restart(state, "http://127.0.0.1:1235")

        assert [url for url, _ in calls] == ["http://127.0.0.1:1235/restart"]
        # A cold tools/call would have been rejected by the MCP transport.
        assert "/mcp" not in calls[0][0]
        assert fell_back == []

    @pytest.mark.asyncio
    async def test_handoff_sends_the_key_when_one_is_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from studioforge.api import admin_routes

        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx, "AsyncClient", lambda **_: _FakeClient(calls, _Response(200))
        )
        monkeypatch.setattr(admin_routes.asyncio, "sleep", _noop_sleep)

        await admin_routes._ask_watchdog_to_restart(_state_with_key("k3y"), "http://wd:1235")

        assert calls[0][1]["headers"]["Authorization"] == "Bearer k3y"

    @pytest.mark.asyncio
    async def test_a_refused_handoff_falls_back_to_respawn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from studioforge.api import admin_routes

        calls: list[tuple[str, dict[str, Any]]] = []
        monkeypatch.setattr(
            admin_routes.httpx,
            "AsyncClient",
            lambda **_: _FakeClient(calls, _Response(401, "nope")),
        )
        fell_back: list[bool] = []

        async def _fake_self_restart(_state: Any) -> None:
            fell_back.append(True)

        monkeypatch.setattr(admin_routes, "_self_restart", _fake_self_restart)
        monkeypatch.setattr(admin_routes.asyncio, "sleep", _noop_sleep)

        await admin_routes._ask_watchdog_to_restart(_state_with_key(None), "http://wd:1235")

        # The caller was already told the process is going down; keep that promise.
        assert fell_back == [True]


async def _noop_sleep(_seconds: float) -> None:
    return None


def _state_with_key(key: str | None) -> Any:
    class _Server:
        api_key = key

    class _Config:
        server = _Server()

    class _State:
        config = _Config()

    return _State()


class TestWatchdogPinIsEnforcedWithoutAnApiKey:
    """The recovery plane must never be the least-protected surface.

    In the default install `server.api_key` is unset and only the MCP pairing
    PIN exists. The main app still demands the PIN on /mcp; the watchdog used
    to gate its whole auth block on the key alone, so with no key set every
    destructive recovery tool -- and POST /restart -- was reachable by anyone
    who could route to port 1235.
    """

    @pytest.mark.asyncio
    async def test_pin_alone_protects_restart(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, credentials=lambda: (None, "87654321"))

        status, _ = await _call(app, "POST", "/restart")
        assert status == 401
        assert watchdog.restart_calls == []

        status, _ = await _call(app, "POST", "/restart", [(b"authorization", b"Bearer 87654321")])
        assert status == 200
        assert watchdog.restart_calls == [{"confirm": True}]

    @pytest.mark.asyncio
    async def test_pin_alone_protects_the_mcp_plane(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, credentials=lambda: (None, "87654321"))

        status, body = await _call(app, "POST", "/mcp")
        assert status == 401, "recovery tools were reachable with no credential"
        assert body != {"reached": "mcp"}

        status, body = await _call(app, "POST", "/mcp", [(b"authorization", b"Bearer 87654321")])
        assert status == 200 and body == {"reached": "mcp"}

    @pytest.mark.asyncio
    async def test_health_stays_open_even_with_a_pin(self) -> None:
        app = wrap_asgi(_inner, _FakeWatchdog(), credentials=lambda: (None, "87654321"))
        status, body = await _call(app, "GET", "/health")
        assert status == 200 and body["status"] == "up"

    @pytest.mark.asyncio
    async def test_wrong_pin_is_refused(self) -> None:
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, credentials=lambda: (None, "87654321"))
        status, _ = await _call(app, "POST", "/restart", [(b"authorization", b"Bearer 00000000")])
        assert status == 401
        assert watchdog.restart_calls == []

    @pytest.mark.asyncio
    async def test_open_only_when_neither_credential_exists(self) -> None:
        # Explicitly no key and no PIN: the documented fully-open mode.
        watchdog = _FakeWatchdog()
        app = wrap_asgi(_inner, watchdog, credentials=lambda: (None, None))
        status, _ = await _call(app, "POST", "/restart")
        assert status == 200
