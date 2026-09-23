"""A shutdown waits the full drain only while inference is in flight (D69 §18).

Every restart route drains the models first (``manager.stop()``), so by the
time uvicorn shuts down nothing is in flight -- and yet on 2026-09-22 09:26:49
the process waited the whole ``drain_timeout_s`` (30 s) on idle connections
("Cancel 0 running task(s), timeout graceful shutdown exceeded"): an MCP
client's standing SSE stream on the API side (09-04), a browser tab on the GUI
side (09-10, 09-22). The drain is now decided at shutdown: the full window
while a child is serving somebody, :data:`IDLE_DRAIN_S` otherwise.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn

from studioforge import __main__ as cli


class _Supervisor:
    def __init__(self, active: list[int] | None = None, *, broken: bool = False) -> None:
        self.active = active or []
        self.broken = broken

    def list(self) -> list[Any]:
        if self.broken:
            raise RuntimeError("supervisor table mid-rebuild")
        return [SimpleNamespace(active_requests=n) for n in self.active]


def test_inference_in_flight_counts_every_childs_requests() -> None:
    assert cli._inference_in_flight(SimpleNamespace(supervisor=_Supervisor([0, 2, 1]))) == 3
    assert cli._inference_in_flight(SimpleNamespace(supervisor=_Supervisor([]))) == 0
    assert cli._inference_in_flight(SimpleNamespace()) == 0  # no supervisor, no children
    # Unknown is not idle: a guess must never cut a stream.
    assert cli._inference_in_flight(SimpleNamespace(supervisor=_Supervisor(broken=True))) is None


@pytest.mark.parametrize(
    ("in_flight", "expected"),
    [(0, cli.IDLE_DRAIN_S), (3, 30), (None, 30)],
    ids=["idle", "streaming", "unknown"],
)
async def test_the_drain_is_decided_at_shutdown(
    monkeypatch: pytest.MonkeyPatch, in_flight: int | None, expected: int
) -> None:
    seen: list[Any] = []

    async def base_shutdown(self: uvicorn.Server, sockets: Any = None) -> None:
        seen.append(self.config.timeout_graceful_shutdown)

    monkeypatch.setattr(uvicorn.Server, "shutdown", base_shutdown)
    server_cls = cli._draining_server_class()
    server = server_cls(
        uvicorn.Config(app=lambda *_a: None, timeout_graceful_shutdown=30, log_config=None),
        name="api",
        full_drain_s=lambda: 30.0,
        in_flight=lambda: in_flight,
    )

    await server.shutdown()

    assert seen == [expected]
    assert server.drain_s == expected


async def test_the_full_drain_is_read_live_and_never_exceeded_by_the_idle_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Any] = []

    async def base_shutdown(self: uvicorn.Server, sockets: Any = None) -> None:
        seen.append(self.config.timeout_graceful_shutdown)

    monkeypatch.setattr(uvicorn.Server, "shutdown", base_shutdown)
    drain = {"s": 30.0}
    server = cli._draining_server_class()(
        uvicorn.Config(app=lambda *_a: None, timeout_graceful_shutdown=30, log_config=None),
        name="api",
        full_drain_s=lambda: drain["s"],
        in_flight=lambda: 1,
    )
    drain["s"] = 45.0  # changed through PATCH /api/config while the server ran
    await server.shutdown()
    drain["s"] = 0.0  # an operator who wants no drain at all gets none, idle or not
    server._in_flight = lambda: 0
    await server.shutdown()
    assert seen == [45, 0]


def test_serve_builds_both_servers_with_the_right_in_flight_rule() -> None:
    source = inspect.getsource(cli._serve)
    assert source.count("draining_server(") == 2
    assert "in_flight=lambda: _inference_in_flight(api.state)" in source
    assert "in_flight=lambda: 0" in source  # the GUI's connections are browser tabs


# ---------------------------------------------------------------------------
# The real thing: a uvicorn server holding an endless streaming response
# ---------------------------------------------------------------------------


async def _endless_stream(scope: dict[str, Any], receive: Any, send: Any) -> None:
    """What an MCP SSE stream or a long-poll looks like to uvicorn: a response
    that is still being written when the shutdown starts."""
    if scope["type"] != "http":
        return
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/event-stream")],
        }
    )
    await send({"type": "http.response.body", "body": b"data: hello\n\n", "more_body": True})
    while True:
        await asyncio.sleep(0.05)
        await send({"type": "http.response.body", "body": b": ping\n\n", "more_body": True})


async def _shutdown_time(monkeypatch: pytest.MonkeyPatch, *, in_flight: int) -> float:
    monkeypatch.setattr(cli, "IDLE_DRAIN_S", 0)
    server = cli._draining_server_class()(
        uvicorn.Config(
            _endless_stream,
            host="127.0.0.1",
            port=0,
            lifespan="off",
            log_config=None,
            access_log=False,
            timeout_graceful_shutdown=3,
        ),
        name="api",
        full_drain_s=lambda: 3.0,
        in_flight=lambda: in_flight,
    )
    serving = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.02)
    assert server.started, "the test server never started"
    port = server.servers[0].sockets[0].getsockname()[1]

    got_first = asyncio.Event()

    async def client() -> None:
        async with httpx.AsyncClient(timeout=10.0) as http:
            try:
                async with http.stream("GET", f"http://127.0.0.1:{port}/sse") as response:
                    async for _chunk in response.aiter_raw():
                        got_first.set()
            except httpx.HTTPError:
                pass  # the server cut the stream: expected

    reading = asyncio.create_task(client())
    await asyncio.wait_for(got_first.wait(), timeout=10.0)
    started = time.monotonic()
    server.should_exit = True
    await asyncio.wait_for(serving, timeout=20.0)
    elapsed = time.monotonic() - started
    reading.cancel()
    await asyncio.gather(reading, return_exceptions=True)
    return elapsed


async def test_an_idle_shutdown_does_not_wait_for_a_standing_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await _shutdown_time(monkeypatch, in_flight=0) < 1.5


async def test_a_shutdown_mid_inference_still_gets_the_full_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert await _shutdown_time(monkeypatch, in_flight=1) >= 2.5
