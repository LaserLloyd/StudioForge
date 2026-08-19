"""Failure-injection tests for the recovery watchdog.

This is the suite that decides whether the watchdog is worth having. Its central
claim is not "the tools return sensible JSON" but "a StudioForge server that has
stopped cooperating can be diagnosed and recovered **through watchdog calls
alone**", so almost nothing here is mocked:

* The "main app" is a real subprocess (:data:`FAKE_APP_SOURCE`) that serves
  ``GET /health`` from a single-threaded HTTP server and accepts ``POST /wedge``,
  which answers and then blocks its only serving thread forever. The process
  stays alive, keeps its listening socket, and never answers again -- which is
  exactly what a blocked event loop or a hung CUDA call looks like from outside,
  and is the state that the wedged-vs-down distinction exists to name.
* The llama-server children are real processes with a real ``--alias``/``--port``
  argv, so discovery goes through ``psutil`` for real and the kills are real
  kills.
* The recovery path (:func:`test_restart_recovers_a_wedged_server_over_http`)
  goes through a genuine MCP client session over the streamable-HTTP transport,
  because that is the deployed shape: an operator or an agent reaching a dying
  box over the network, not an in-process function call.

Safety: every config here uses ephemeral ports and a child port range
(:data:`CHILD_PORT_START`) far away from the production default, so the
destructive tools cannot reach a StudioForge instance the developer happens to
be running. ``nuke_all_models`` is additionally asserted to have found exactly
the fake children before it is allowed to fire.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest
import yaml

from studioforge.config import Config
from studioforge.watchdog import server as wd_module
from studioforge.watchdog.server import (
    Watchdog,
    create_asgi_app,
    find_llama_children,
    find_main_process,
    safe_log_name,
)

# Deliberately far from the production 18100-18200 so a live StudioForge on this
# box is invisible to -- and therefore safe from -- these tests.
#
# The range is *claimed per pytest session* rather than hard-coded, because
# ``find_llama_children`` scopes discovery by exactly this range and by nothing
# else: two sessions sharing one range would each discover the other's fake
# children -- and ``nuke_all_models`` would kill them. That is what a stray
# ``children_total == 2`` looks like when the whole unit suite runs beside
# another run of this file.
_RANGE_BASE = 19600
_RANGE_SIZE = 64
_RANGE_SLOTS = 32


def _range_is_clear(start: int, end: int) -> bool:
    """True when no process on this box carries a ``--port`` inside the range.

    Deliberately broader than :func:`find_llama_children`: *anything* holding a
    port in the range -- an orphan from a session that was killed before its
    teardown ran, a hand-started llama.cpp -- makes the range unusable, and
    moving is cheaper than explaining the resulting off-by-one later.
    """
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = list(proc.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            continue
        raw = wd_module._flag_value(cmdline, "--port")  # noqa: SLF001 - test introspection
        try:
            port = int(str(raw))
        except (TypeError, ValueError):
            continue
        if start <= port <= end:
            return False
    return True


def _claim_child_ports() -> tuple[int, int, socket.socket]:
    """Reserve a child port range that belongs to this pytest session alone.

    The claim is a real listening socket on the range's first port, so the OS --
    not luck -- arbitrates between concurrent sessions. The socket is held for
    the life of the process, and that first port is never given to a fake child.
    """
    for slot in range(_RANGE_SLOTS):
        start = _RANGE_BASE + slot * _RANGE_SIZE
        end = start + _RANGE_SIZE - 1
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", start))
            sock.listen(1)
        except OSError:
            sock.close()
            continue
        if _range_is_clear(start + 1, end):
            return start, end, sock
        sock.close()
    top = _RANGE_BASE + _RANGE_SLOTS * _RANGE_SIZE - 1
    raise RuntimeError(f"no free child port range in {_RANGE_BASE}-{top}")


CHILD_PORT_START, CHILD_PORT_END, _CHILD_PORT_CLAIM = _claim_child_ports()

#: A child port outside *every* slot, for the "not ours" case. It has to miss
#: every other session's range too, or the scoping test would plant a child in
#: someone else's count.
OUTSIDE_PORT = _RANGE_BASE + _RANGE_SLOTS * _RANGE_SIZE + 100

HEALTH_TIMEOUT_S = 0.6
WEDGED_AFTER = 2

#: Interpreter used to launch the fake processes.
#:
#: NOT ``sys.executable``: on Windows a virtualenv's ``python.exe`` is often a
#: *trampoline* that re-execs the real interpreter as a child process with the
#: same argv. Every fake process would then exist twice, with identical command
#: lines, and "which pid is the server" would have two right answers -- which
#: makes pid assertions meaningless rather than merely awkward.
#: ``sys._base_executable`` is the real interpreter binary, so one ``Popen`` is
#: one process. The fake scripts use nothing outside the stdlib, so losing the
#: venv's site-packages costs nothing.
PYTHON = getattr(sys, "_base_executable", None) or sys.executable

#: A stand-in for the main app: answers /health, and can be told to wedge.
#:
#: ``HTTPServer`` (not ``ThreadingHTTPServer``) is essential -- there is exactly
#: one serving thread, so blocking inside a handler blocks the whole server while
#: the process and its listening socket survive. ``/wedge`` sends its response
#: first and only then blocks, so the caller is not left hanging on the request
#: that causes the wedge.
FAKE_APP_SOURCE = """
import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/api/health"):
            self._json(200, {"status": "ok", "fake": True, "pid": os.getpid()})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") == "/wedge":
            self._json(200, {"wedged": True, "pid": os.getpid()})
            # The whole point: answer, then never return. The process stays
            # alive and keeps accepting TCP connections it will never serve.
            time.sleep(1e9)
        else:
            self._json(404, {"error": "not found"})

    def log_message(self, *args):
        pass


parser = argparse.ArgumentParser()
parser.add_argument("--config", default=None)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()
HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
"""

#: A stand-in llama-server child: it only has to exist and carry the right argv.
#:
#: It blocks on stdin rather than sleeping, and is launched with ``stdin=PIPE``.
#: A timed sleep would outlive a pytest session that dies without running its
#: teardown -- a timeout kill, a Ctrl+C, a crashed worker -- and the orphan would
#: then be discovered by the *next* session as an extra llama-server child.
#: Reading the pipe instead makes the parent's death an EOF, so the child exits
#: with the session that owns it.
FAKE_CHILD_SOURCE = """
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--alias", required=True)
parser.add_argument("--port", type=int, required=True)
parser.parse_args()
sys.stdin.read()
"""


def free_ports(count: int) -> list[int]:
    """``count`` distinct free localhost ports.

    All sockets are held open until every port has been chosen: binding to port 0
    one at a time can hand out the *same* port twice (and on Windows tends to
    hand out consecutive ones), which would silently turn a test into a port
    collision that the config schema then rejects for the wrong reason.
    """
    socks: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            socks.append(sock)
        return [int(sock.getsockname()[1]) for sock in socks]
    finally:
        for sock in socks:
            sock.close()


class _suppress:
    """Tiny context manager: test cleanup must never fail a test."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return True


def kill_tree(pid: int) -> None:
    """Kill a pid and its descendants, tolerating anything going wrong."""
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, ValueError):
        return
    procs: list[psutil.Process] = []
    with _suppress():
        procs = proc.children(recursive=True)
    procs.append(proc)
    for entry in procs:
        with _suppress():
            entry.kill()
    psutil.wait_procs(procs, timeout=5)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class FakeMainApp:
    """A real subprocess standing in for ``studioforge serve``."""

    def __init__(self, script: Path, config_path: Path, port: int) -> None:
        self.script = script
        self.config_path = config_path
        self.port = port
        self.proc = subprocess.Popen(
            self.argv(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def argv(self) -> list[str]:
        return [
            PYTHON,
            str(self.script),
            "--config",
            str(self.config_path),
            "--port",
            str(self.port),
        ]

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def serving_pid(self) -> int:
        """The pid that actually holds the listening socket.

        "The main app process" means the process bound to the port, which is
        what the watchdog discovers and what it must kill. Resolving it from the
        socket rather than trusting ``Popen.pid`` keeps these assertions correct
        even if the interpreter used to launch it inserts a wrapper process.
        """
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            pid = wd_module._pid_listening_on(self.port)  # noqa: SLF001 - test introspection
            if pid:
                return int(pid)
            time.sleep(0.05)
        return self.pid

    def wait_healthy(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"fake app exited early with {self.proc.returncode}")
            try:
                response = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001 - retry until the deadline
                last = exc
            time.sleep(0.1)
        raise RuntimeError(f"fake app never became healthy on {self.port}: {last}")

    def wedge(self) -> None:
        """Make the process stop answering while staying alive."""
        with _suppress():
            httpx.post(f"http://127.0.0.1:{self.port}/wedge", timeout=5.0)
        # Confirm from the outside that it really is unresponsive now.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=0.5)
            except Exception:  # noqa: BLE001 - this is the success condition
                return
            time.sleep(0.1)
        raise RuntimeError("fake app kept answering after /wedge")

    def stop(self) -> None:
        kill_tree(self.pid)


class Harness:
    """Everything one watchdog test needs, with guaranteed cleanup."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.app_script = tmp_path / "fake_app.py"
        self.app_script.write_text(FAKE_APP_SOURCE, encoding="utf-8")
        self.child_script = tmp_path / "fake_child.py"
        self.child_script.write_text(FAKE_CHILD_SOURCE, encoding="utf-8")

        (
            self.server_port,
            self.watchdog_port,
            self.gui_port,
            self.spare_port,
        ) = free_ports(4)

        config = Config(data_dir=tmp_path / "data")
        config.server.host = "127.0.0.1"
        config.server.port = self.server_port
        config.server.api_key = None
        config.gui.port = self.gui_port
        config.watchdog.host = "127.0.0.1"
        config.watchdog.port = self.watchdog_port
        config.watchdog.health_timeout_s = HEALTH_TIMEOUT_S
        config.watchdog.wedged_after_failures = WEDGED_AFTER
        config.watchdog.poll_interval_s = 3600.0  # the tests drive health directly
        config.gateway.child_port_start = CHILD_PORT_START
        config.gateway.child_port_end = CHILD_PORT_END
        config.models.dir = tmp_path / "models"
        config.models.dir.mkdir(parents=True, exist_ok=True)
        config.ensure_dirs()
        # The production layout: config.yaml lives IN the data directory, and a
        # process handed `--config <path>` takes the data dir from where the
        # file is (D31) -- the file itself never carries a data_dir key.
        self.config_path = config.data_dir / "config.yaml"
        config.save(self.config_path)
        self.config = config

        self.app: FakeMainApp | None = None
        self.children: list[subprocess.Popen[bytes]] = []
        self.extra_pids: list[int] = []

    # -- lifecycle -------------------------------------------------------

    def watchdog(self) -> Watchdog:
        """A watchdog whose restart command respawns the *fake* app.

        Overriding the command is what makes a real restart testable: the
        production default would launch an actual StudioForge server (slow, and
        it would scan the developer's model library).
        """
        return Watchdog(
            self.config_path,
            restart_command=[
                PYTHON,
                str(self.app_script),
                "--config",
                str(self.config_path),
                "--port",
                str(self.server_port),
            ],
        )

    def start_app(self) -> FakeMainApp:
        self.app = FakeMainApp(self.app_script, self.config_path, self.server_port)
        self.app.wait_healthy()
        return self.app

    def start_child(self, alias: str, port: int) -> int:
        """Spawn a fake llama-server child; returns the pid the watchdog sees.

        Returning the *discovered* pid rather than ``Popen.pid`` keeps the
        assertions about the thing under test: what the watchdog finds is what it
        will kill. It also proves discovery works before the test relies on it.
        """
        proc = subprocess.Popen(
            [
                PYTHON,
                str(self.child_script),
                "--alias",
                alias,
                "--port",
                str(port),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.children.append(proc)
        # Wait until psutil can see the argv, otherwise discovery races startup.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            matches = [c for c in find_llama_children(self.config) if c.alias == alias]
            if matches:
                assert len(matches) == 1, f"ambiguous discovery for {alias}: {matches}"
                return matches[0].pid
            time.sleep(0.05)
        raise RuntimeError(f"fake child {alias} never became discoverable")

    def read_config_from_disk(self) -> dict[str, Any]:
        return dict(yaml.safe_load(self.config_path.read_text(encoding="utf-8")))

    def cleanup(self) -> None:
        if self.app is not None:
            self.app.stop()
        for proc in self.children:
            kill_tree(proc.pid)
        for pid in self.extra_pids:
            kill_tree(pid)
        # Anything the watchdog respawned during a restart test is detached from
        # this process, so it has to be hunted down by the port it holds.
        with _suppress():
            for conn in psutil.net_connections(kind="inet"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and getattr(conn.laddr, "port", None) == self.server_port
                    and conn.pid
                ):
                    kill_tree(int(conn.pid))


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's ``SF_*`` environment out of these tests."""
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def harness(tmp_path: Path) -> Iterator[Harness]:
    instance = Harness(tmp_path)
    try:
        yield instance
    finally:
        instance.cleanup()


async def call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    result = await server.call_tool(name, arguments)
    assert result.content, f"{name} returned no content"
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


def build_mcp(watchdog: Watchdog) -> Any:
    return wd_module.build_watchdog_mcp(watchdog)


# ===========================================================================
# 1 + 2. The fake app runs, and the watchdog says "up"
# ===========================================================================


async def test_health_reports_up_before_wedging(harness: Harness) -> None:
    app = harness.start_app()
    server = build_mcp(harness.watchdog())

    result = await call(server, "health")
    assert result["status"] == "up", result
    assert result["main"]["status"] == "up"
    assert result["main"]["process_found"] is True
    assert result["main"]["pid"] == app.serving_pid
    assert result["main"]["body"]["fake"] is True
    assert result["children"] == []


async def test_health_finds_children_and_reports_degraded(harness: Harness) -> None:
    """A child that does not answer its own /health is a partial failure.

    The fake child never binds its port, which is precisely the state the
    watchdog must catch: a process holding VRAM that is not serving.
    """
    harness.start_app()
    child_pid = harness.start_child("vendor/Model-Q4_K_M", CHILD_PORT_START + 1)
    server = build_mcp(harness.watchdog())

    result = await call(server, "health")
    assert result["status"] == "degraded", result
    assert result["children_total"] == 1
    entry = result["children"][0]
    assert entry["pid"] == child_pid
    assert entry["alias"] == "vendor/Model-Q4_K_M"
    assert entry["port"] == CHILD_PORT_START + 1
    assert entry["healthy"] is False


async def test_health_reports_down_when_nothing_is_running(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    result = await call(server, "health")
    assert result["status"] == "down"
    assert result["main"]["process_found"] is False
    assert "needs to be started" in result["main"]["reason"]
    # Down must be cheap: no point burning failures x timeout on a process that
    # demonstrably does not exist.
    assert result["main"]["attempts"] == 1


# ===========================================================================
# 3. Wedged, not down
# ===========================================================================


async def test_health_reports_wedged_not_down_after_wedging(harness: Harness) -> None:
    app = harness.start_app()
    watchdog = harness.watchdog()
    server = build_mcp(watchdog)
    assert (await call(server, "health"))["status"] == "up"

    serving_pid = app.serving_pid
    app.wedge()
    assert psutil.pid_exists(serving_pid), "the wedged process must still be alive"

    result = await call(server, "health")
    assert result["status"] == "wedged", result
    assert result["status"] != "down"
    main = result["main"]
    assert main["process_found"] is True
    assert main["pid"] == serving_pid
    assert main["attempts"] == WEDGED_AFTER
    assert watchdog.consecutive_failures >= WEDGED_AFTER
    # The detail must state the distinction explicitly, because that is what
    # tells the operator to restart rather than to start.
    assert "EXISTS" in main["reason"]
    assert "did not answer" in main["reason"]
    assert "restart_server" in main["reason"]
    assert "wedged, not dead" in main["reason"]
    assert "unresponsive" in result["summary"]


async def test_plain_health_endpoint_reports_wedged_without_an_mcp_handshake(
    harness: Harness,
) -> None:
    """Load balancers and ``sfctl`` poll plain HTTP, not JSON-RPC."""
    app = harness.start_app()
    serving_pid = app.serving_pid
    app.wedge()
    watchdog = harness.watchdog()
    asgi, _server = create_asgi_app(watchdog)

    transport = httpx.ASGITransport(app=asgi)
    async with httpx.AsyncClient(transport=transport, base_url="http://watchdog") as client:
        response = await client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "wedged"
    assert body["main"]["pid"] == serving_pid


# ===========================================================================
# 4. Config repair while the main app is wedged
# ===========================================================================


async def test_set_config_rejects_an_invalid_value_while_wedged(harness: Harness) -> None:
    """A port collision is genuinely invalid, and nothing may be written."""
    app = harness.start_app()
    app.wedge()
    server = build_mcp(harness.watchdog())
    before = harness.read_config_from_disk()

    result = await call(server, "set_config", updates={"server.port": harness.watchdog_port})
    assert result["ok"] is False
    assert result["error"]["code"] == "invalid_config"
    assert "collides" in result["error"]["message"]
    assert "Nothing was written" in result["error"]["message"]
    assert harness.read_config_from_disk() == before


async def test_set_config_rejects_a_type_error_and_an_unknown_key(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    before = harness.read_config_from_disk()

    bad_type = await call(server, "set_config", updates={"models.default_ctx": "not-a-number"})
    assert bad_type["ok"] is False

    unknown = await call(server, "set_config", updates={"models.nonexistent": 1})
    assert unknown["ok"] is False
    assert "unknown config key" in unknown["error"]["message"]

    assert harness.read_config_from_disk() == before


async def test_set_config_writes_a_valid_change_that_survives_a_reload(
    harness: Harness,
) -> None:
    app = harness.start_app()
    app.wedge()
    server = build_mcp(harness.watchdog())

    result = await call(
        server,
        "set_config",
        updates={"models.default_ctx": 12288, "watchdog.port": harness.spare_port},
    )
    assert result["ok"] is True, result
    assert result["updated"] == ["models.default_ctx", "watchdog.port"]
    assert result["restart_required"] == ["watchdog.port"]

    # On disk...
    on_disk = harness.read_config_from_disk()
    assert on_disk["models"]["default_ctx"] == 12288
    assert on_disk["watchdog"]["port"] == harness.spare_port

    # ...and it survives a fresh validated load through the real schema.
    reloaded, error = harness.watchdog().load_config()
    assert error is None
    assert reloaded.models.default_ctx == 12288
    assert reloaded.watchdog.port == harness.spare_port


async def test_set_config_writes_atomically(harness: Harness) -> None:
    """No temp files may be left behind next to config.yaml."""
    server = build_mcp(harness.watchdog())
    assert (await call(server, "set_config", updates={"models.default_ctx": 2048}))["ok"] is True
    leftovers = [p.name for p in harness.config_path.parent.glob("config.yaml*")]
    assert leftovers == ["config.yaml"]


async def test_get_config_redacts_secrets_read_from_disk(harness: Harness) -> None:
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["server"]["api_key"] = "sf-super-secret-value"
    raw["hf"]["token"] = "hf_super_secret_token_x"
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    server = build_mcp(harness.watchdog())
    text = (await server.call_tool("get_config", {})).content[0].text
    assert "sf-super-secret-value" not in text
    assert "hf_super_secret_token_x" not in text
    payload = json.loads(text)
    assert payload["config"]["server"]["api_key"] == "sf-s...ue"
    assert payload["config"]["hf"]["token"] == "hf_s..._x"


async def test_config_escape_hatch_repairs_an_unloadable_file(harness: Harness) -> None:
    """The headline scenario: a bad setting stops the server, and this fixes it.

    The file is left in a state the real schema rejects, so there is no valid
    base ``Config`` to layer an update onto -- and the watchdog still has to
    apply the fix while preserving the operator's other settings.
    """
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["planner"]["headroom_fraction"] = 0.99  # validator rejects >= 0.9
    raw["models"]["default_ttl_s"] = 4242  # an unrelated setting that must survive
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    watchdog = harness.watchdog()
    _config, error = watchdog.load_config()
    assert error is not None and "headroom_fraction" in error

    server = build_mcp(watchdog)
    reported = await call(server, "get_config")
    assert reported["ok"] is False
    assert "headroom_fraction" in reported["config_error"]

    fixed = await call(server, "set_config", updates={"planner.headroom_fraction": 0.1})
    assert fixed["ok"] is True
    assert fixed["repaired_invalid_config"] is True

    reloaded, error = watchdog.load_config()
    assert error is None
    assert reloaded.planner.headroom_fraction == pytest.approx(0.1)
    assert reloaded.models.default_ttl_s == 4242, "unrelated settings must survive the repair"


# ===========================================================================
# 5 + 6. Recovery through the watchdog alone, over a real MCP session
# ===========================================================================


class WatchdogHttp:
    """The watchdog served over real HTTP, in a background uvicorn thread."""

    def __init__(self, watchdog: Watchdog, port: int) -> None:
        import uvicorn

        asgi, self.mcp = create_asgi_app(watchdog)
        self.port = port
        self.server = uvicorn.Server(
            uvicorn.Config(asgi, host="127.0.0.1", port=port, log_level="error", access_log=False)
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    def __enter__(self) -> WatchdogHttp:
        self.thread.start()
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("watchdog HTTP server did not start")

    def __exit__(self, *exc: Any) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=15.0)


@pytest.mark.timeout(180)
async def test_restart_recovers_a_wedged_server_over_http(harness: Harness) -> None:
    """THE acceptance criterion.

    The main app never cooperates: it is wedged before the restart and is killed
    without ever handling another request. Recovery happens entirely through MCP
    tool calls issued by a real client over the streamable-HTTP transport --
    which is the shape this actually ships in.
    """
    app = harness.start_app()
    old_pid = app.serving_pid
    app.wedge()

    watchdog = harness.watchdog()
    with WatchdogHttp(watchdog, harness.watchdog_port) as http:
        from mcp import Client

        async with Client(http.url) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            assert names == {
                "health",
                "get_config",
                "set_config",
                "restart_server",
                "kill_model",
                "nuke_all_models",
                "reclaim_orphan_engines",
                "tail_logs",
                "gpu_status",
                "rollback_update",
            }

            wedged = json.loads((await client.call_tool("health", {})).content[0].text)
            assert wedged["status"] == "wedged"

            # (6) No confirmation: refuse, and change nothing at all.
            refused = json.loads((await client.call_tool("restart_server", {})).content[0].text)
            assert refused["ok"] is False
            assert refused["confirmed"] is False
            assert refused["error"]["code"] == "confirmation_required"
            assert psutil.pid_exists(old_pid), "a refused restart must not kill anything"

            # (5) Confirmed: kill the wedged process and bring a fresh one up.
            result = json.loads(
                (await client.call_tool("restart_server", {"confirm": True, "timeout_s": 60.0}))
                .content[0]
                .text
            )
            assert result["ok"] is True, result
            assert result["healthy"] is True
            assert result["previous_pid"] == old_pid
            assert old_pid in result["killed_pids"]
            assert result["new_pid"] not in (None, old_pid)
            harness.extra_pids.append(int(result["new_pid"]))

            recovered = json.loads((await client.call_tool("health", {})).content[0].text)
            assert recovered["status"] == "up"
            assert recovered["main"]["pid"] == result["new_pid"]

    # The wedged process is genuinely gone, and a different one is serving.
    assert not _alive(old_pid)
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(f"http://127.0.0.1:{harness.server_port}/health")
    assert response.status_code == 200
    assert response.json()["pid"] != old_pid


def _alive(pid: int) -> bool:
    """True only for a genuinely running (non-zombie) process."""
    try:
        proc = psutil.Process(pid)
        return proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, ValueError):
        return False


@pytest.mark.timeout(120)
async def test_restart_starts_a_server_that_was_completely_down(harness: Harness) -> None:
    """``restart_server`` also covers the cold-start case, and says which it did."""
    watchdog = harness.watchdog()
    server = build_mcp(watchdog)
    assert (await call(server, "health"))["status"] == "down"

    result = await call(server, "restart_server", confirm=True, timeout_s=60.0)
    assert result["ok"] is True
    assert result["previous_pid"] is None
    assert result["new_pid"] is not None
    harness.extra_pids.append(int(result["new_pid"]))
    assert (await call(server, "health"))["status"] == "up"


async def test_restart_without_confirm_explains_the_consequences(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    result = await call(server, "restart_server")
    assert result["ok"] is False
    assert "VRAM" in result["error"]["message"]
    assert result["error"]["param"] == "confirm"


# ---------------------------------------------------------------------------
# D28: exactly one respawner. When the server is the tray's child, the
# watchdog kills it and leaves the respawn to the tray, publishing
# restart_in_progress on /health so the tray can tell the exit from a crash.
# ---------------------------------------------------------------------------


def _fake_main(pid: int = 4242) -> wd_module.ChildProcess:
    return wd_module.ChildProcess(
        pid=pid, name="python.exe", alias=None, port=1234, create_time=1.0, cmdline=["serve"]
    )


@pytest.mark.skipif(os.name != "nt", reason="the tray, and this branch, are Windows-only")
async def test_restart_leaves_the_respawn_to_a_tray_that_owns_the_server(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = harness.watchdog()
    killed: list[int] = []
    seen_in_progress: list[dict[str, Any] | None] = []

    monkeypatch.setattr(wd_module, "find_main_process", lambda config, **kw: _fake_main())
    monkeypatch.setattr(wd_module, "supervising_tray_pid", lambda pid: 777)
    monkeypatch.setattr(wd_module, "find_llama_children", lambda config: [])
    monkeypatch.setattr(
        wd_module, "kill_process_tree", lambda pid, **kw: (killed.append(pid), [pid])[1]
    )

    async def port_free(port: int, timeout: float = 10.0) -> bool:  # noqa: ASYNC109
        return True

    async def wait_health(config: Any, timeout: float) -> tuple[bool, float]:  # noqa: ASYNC109
        # Captured mid-restart: this is what the tray reads on /health.
        seen_in_progress.append(dict(watchdog._restart_in_progress or {}))
        return True, 1.5

    def must_not_spawn(config: Any) -> Any:  # pragma: no cover - the assertion
        raise AssertionError("the watchdog spawned a server the tray was going to respawn")

    monkeypatch.setattr(watchdog, "_wait_port_free", port_free)
    monkeypatch.setattr(watchdog, "_wait_for_health", wait_health)
    monkeypatch.setattr(watchdog, "_spawn_main", must_not_spawn)

    result = await watchdog.restart_server(confirm=True, timeout_s=30.0)

    assert killed == [4242], "the tray's child is still killed -- that IS the restart"
    assert result["ok"] is True and result["healthy"] is True
    assert result["respawned_by"] == "tray"
    assert result["tray_pid"] == 777
    assert "tray" in result["method"]
    assert seen_in_progress and seen_in_progress[0]["respawned_by"] == "tray"
    assert seen_in_progress[0]["previous_pid"] == 4242
    # ...and cleared once the restart is over, so a later crash is a crash.
    assert watchdog._restart_in_progress is None
    health = await watchdog.health()
    assert health["restart_in_progress"] is None


async def test_restart_spawns_itself_when_no_tray_owns_the_server(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = harness.watchdog()
    spawned: list[bool] = []

    monkeypatch.setattr(wd_module, "find_main_process", lambda config, **kw: _fake_main())
    monkeypatch.setattr(wd_module, "supervising_tray_pid", lambda pid: None)
    monkeypatch.setattr(wd_module, "find_llama_children", lambda config: [])
    monkeypatch.setattr(wd_module, "kill_process_tree", lambda pid, **kw: [pid])

    async def port_free(port: int, timeout: float = 10.0) -> bool:  # noqa: ASYNC109
        return True

    async def wait_health(config: Any, timeout: float) -> tuple[bool, float]:  # noqa: ASYNC109
        return True, 1.0

    def spawn(config: Any) -> tuple[int | None, list[str], str | None]:
        spawned.append(True)
        return 5150, ["fake"], None

    monkeypatch.setattr(watchdog, "_wait_port_free", port_free)
    monkeypatch.setattr(watchdog, "_wait_for_health", wait_health)
    monkeypatch.setattr(watchdog, "_spawn_main", spawn)
    if os.name != "nt":
        # On POSIX the systemctl branch runs first; make it fall through.
        async def no_systemctl(unit: str) -> tuple[bool, str]:
            return False, "no systemd here"

        monkeypatch.setattr(watchdog, "_systemctl_restart", no_systemctl)

    result = await watchdog.restart_server(confirm=True, timeout_s=30.0)
    assert spawned == [True]
    assert result["respawned_by"] == "watchdog"
    assert result["new_pid"] == 5150


def test_tray_cmdline_recognition() -> None:
    assert wd_module._is_tray_cmdline(["pythonw.exe", "-m", "studioforge", "tray", "--config", "x"])
    assert wd_module._is_tray_cmdline([r"C:\v\Scripts\studioforge.exe", "tray"])
    assert wd_module._is_tray_cmdline(["python", "-m", "studioforge.tray.tray_app"])
    assert not wd_module._is_tray_cmdline(["python", "-m", "studioforge", "serve", "--config", "x"])
    assert not wd_module._is_tray_cmdline(["python", "-m", "studioforge.watchdog"])
    assert not wd_module._is_tray_cmdline(["explorer.exe"])


def test_supervising_tray_pid_is_none_for_a_process_pytest_launched() -> None:
    """Our own parent is pytest (or a shell), never a tray."""
    assert wd_module.supervising_tray_pid(os.getpid()) is None
    assert wd_module.supervising_tray_pid(2**22 + 12345) is None  # no such pid


@pytest.mark.skipif(os.name != "nt", reason="Windows respawn mechanism")
def test_windows_restart_uses_a_detached_process_group(harness: Harness) -> None:
    """The respawn must not inherit our console or Ctrl+C group.

    Asserted on the flags rather than by observing a console, because the
    consequence of getting it wrong -- the recovered server dying with the
    watchdog's terminal -- would only show up in production.
    """
    source = Path(wd_module.__file__ or "").read_text(encoding="utf-8")
    spawn = source.split("def _spawn_main")[1].split("async def _wait_for_health")[0]
    assert "subprocess.CREATE_NEW_PROCESS_GROUP" in spawn
    assert "subprocess.DETACHED_PROCESS" in spawn
    assert "start_new_session" in spawn  # POSIX branch still present


def test_default_restart_command_targets_the_studioforge_cli(harness: Harness) -> None:
    watchdog = Watchdog(harness.config_path)
    config, _ = watchdog.load_config()
    argv = watchdog.restart_command(config)
    assert argv[0] == sys.executable
    assert argv[1:4] == ["-m", "studioforge", "serve"]
    assert argv[4] == "--config"
    assert Path(argv[5]) == harness.config_path


def test_restart_command_is_overridable_by_env(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SF_WATCHDOG_RESTART_CMD", "my-wrapper serve --flag")
    watchdog = Watchdog(harness.config_path)
    config, _ = watchdog.load_config()
    assert watchdog.restart_command(config) == ["my-wrapper", "serve", "--flag"]


# ===========================================================================
# 7 + 8. Killing children
# ===========================================================================


async def test_kill_model_kills_the_matching_child_and_reports_it(harness: Harness) -> None:
    keep_pid = harness.start_child("keeper/Keep-Me-Q4_K_M", CHILD_PORT_START + 2)
    victim_pid = harness.start_child("vendor/Doomed-Q8_0", CHILD_PORT_START + 3)
    server = build_mcp(harness.watchdog())

    result = await call(server, "kill_model", model_name="vendor/Doomed-Q8_0")
    assert result["ok"] is True
    assert result["pids"] == [victim_pid]
    assert result["killed"][0]["alias"] == "vendor/Doomed-Q8_0"
    assert result["killed"][0]["port"] == CHILD_PORT_START + 3
    assert result["killed"][0]["gone"] is True
    # VRAM accounting is present (a number on this box, null without NVML).
    assert "vram_freed_mib" in result

    assert not _alive(victim_pid)
    assert _alive(keep_pid), "only the named model may be killed"


async def test_kill_model_matches_case_insensitively_and_by_substring(
    harness: Harness,
) -> None:
    child_pid = harness.start_child("vendor/Qwen2.5-0.5B-Instruct-Q8_0", CHILD_PORT_START + 4)
    server = build_mcp(harness.watchdog())
    result = await call(server, "kill_model", model_name="qwen2.5-0.5b")
    assert result["ok"] is True
    assert result["pids"] == [child_pid]


async def test_kill_model_on_an_unknown_model_lists_what_is_running(
    harness: Harness,
) -> None:
    harness.start_child("vendor/Running-Q4_K_M", CHILD_PORT_START + 5)
    server = build_mcp(harness.watchdog())
    result = await call(server, "kill_model", model_name="not-loaded")
    assert result["ok"] is False
    assert result["error"]["code"] == "model_not_running"
    assert result["error"]["running_aliases"] == ["vendor/Running-Q4_K_M"]


async def test_nuke_all_models_requires_confirmation(harness: Harness) -> None:
    child_pid = harness.start_child("vendor/Survivor-Q4_K_M", CHILD_PORT_START + 6)
    server = build_mcp(harness.watchdog())

    result = await call(server, "nuke_all_models")
    assert result["ok"] is False
    assert result["error"]["code"] == "confirmation_required"
    assert _alive(child_pid), "an unconfirmed nuke must not kill anything"


async def test_nuke_all_models_with_confirm_kills_every_child(harness: Harness) -> None:
    pids = {
        harness.start_child("a/Alpha-Q4_K_M", CHILD_PORT_START + 7),
        harness.start_child("b/Beta-Q8_0", CHILD_PORT_START + 8),
        harness.start_child("c/Gamma-Q6_K", CHILD_PORT_START + 9),
    }
    watchdog = harness.watchdog()

    # Safety gate: this test kills whatever discovery returns, so prove first
    # that discovery sees exactly our fakes and nothing of the developer's.
    discovered = {child.pid for child in find_llama_children(watchdog.load_config()[0])}
    assert discovered == pids, f"unexpected llama-server-like processes: {discovered - pids}"

    result = await call(build_mcp(watchdog), "nuke_all_models", confirm=True)
    assert result["ok"] is True
    assert result["count"] == 3
    assert set(result["survivors"]) == set()
    assert {entry["pid"] for entry in result["killed"]} == pids
    for pid in pids:
        assert not _alive(pid)


async def test_nuke_all_models_with_nothing_running_is_a_clean_noop(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    result = await call(server, "nuke_all_models", confirm=True)
    assert result["ok"] is True
    assert result["count"] == 0
    assert "nothing to do" in result["message"]


async def test_nuke_spares_a_portless_llama_server(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A llama-server with no ``--port`` cannot be attributed to this instance.

    LM Studio's children and hand-run servers configured through
    ``LLAMA_ARG_PORT`` look exactly like this. They must be *reported* as VRAM
    holders but never killed: our own children always carry ``--port``, so a
    portless one is provably not ours.
    """
    ours_pid = harness.start_child("a/Ours-Q4_K_M", CHILD_PORT_START + 11)
    foreign = subprocess.Popen(  # noqa: ASYNC220 - a real process is the point
        [PYTHON, "-c", "import time; time.sleep(600)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    harness.children.append(foreign)
    await asyncio.sleep(0.3)

    real_iter = wd_module._iter_processes

    def fake_iter() -> Any:
        yield from real_iter()  # our real child stays discoverable
        yield {
            "pid": foreign.pid,
            "name": "llama-server.exe",
            "cmdline": ["llama-server.exe", "--model", "someone-elses.gguf"],
            "create_time": time.time(),
        }

    monkeypatch.setattr(wd_module, "_iter_processes", fake_iter)

    result = await call(build_mcp(harness.watchdog()), "nuke_all_models", confirm=True)
    assert result["ok"] is True
    assert {entry["pid"] for entry in result["killed"]} == {ours_pid}
    assert [entry["pid"] for entry in result["spared"]] == [foreign.pid]
    assert _alive(foreign.pid), (
        "nuke_all_models killed a llama-server it cannot attribute to this instance "
        "(on this box that is LM Studio's loaded model)"
    )
    assert not _alive(ours_pid)


async def test_children_outside_the_configured_port_range_are_never_touched(
    harness: Harness,
) -> None:
    """Scoping by child port range is what stops one instance nuking another."""
    stranger = subprocess.Popen(  # noqa: ASYNC220 - a real process is the point here
        [
            PYTHON,
            str(harness.child_script),
            "--alias",
            "other-instance/Model-Q4_K_M",
            "--port",
            str(OUTSIDE_PORT),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    harness.children.append(stranger)
    await asyncio.sleep(0.5)

    config, _ = harness.watchdog().load_config()
    assert stranger.pid not in {c.pid for c in find_llama_children(config)}

    result = await call(build_mcp(harness.watchdog()), "nuke_all_models", confirm=True)
    assert result["count"] == 0
    assert _alive(stranger.pid)


# ===========================================================================
# 9. Logs
# ===========================================================================


async def test_tail_logs_returns_written_lines(harness: Harness) -> None:
    config, _ = harness.watchdog().load_config()
    server_log = config.logs_dir / "studioforge.log"
    server_log.write_text("first line\nsecond line\nthird line\n", encoding="utf-8")

    model_id = "vendor/Some-Model-Q4_K_M"
    model_log = config.model_logs_dir / f"{safe_log_name(model_id)}.log"
    model_log.parent.mkdir(parents=True, exist_ok=True)
    model_log.write_text("llama_model_loader: loaded\nggml_cuda: ok\n", encoding="utf-8")

    server = build_mcp(harness.watchdog())
    result = await call(server, "tail_logs", n=2, model_id=model_id)
    assert result["ok"] is True
    assert result["server"] == ["second line", "third line"]
    assert result["server_log_exists"] is True
    assert result["model"] == ["llama_model_loader: loaded", "ggml_cuda: ok"]
    assert result["model_log_exists"] is True
    assert Path(result["model_log_path"]) == model_log


async def test_tail_logs_for_a_missing_model_log_is_empty_not_an_error(
    harness: Harness,
) -> None:
    """A model that has never been loaded has no log; that is normal."""
    server = build_mcp(harness.watchdog())
    result = await call(server, "tail_logs", model_id="never/Loaded-Q4_K_M")
    assert result["ok"] is True
    assert result["model"] == []
    assert result["model_log_exists"] is False
    assert result["server"] == []


def test_log_name_sanitization_matches_the_supervisor(harness: Harness) -> None:
    """The duplication is deliberate, but it must not drift.

    Importing the supervisor here (in the *test*, not in the watchdog) is the
    right place to check the two implementations still agree.
    """
    from studioforge.core.supervisor import safe_log_name as supervisor_version

    for model_id in (
        "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0",
        "24_10_Pygmalion or Mistral_cydonia-22b-v1-q6_k",
        "weird/../../name",
        "...",
        "x" * 200,
        "a.b-c_d",
    ):
        assert safe_log_name(model_id) == supervisor_version(model_id), model_id


# ===========================================================================
# 10. GPUs
# ===========================================================================


def _nvml_available() -> bool:
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlDeviceGetCount()) > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - absence is the thing being detected
        return False


requires_nvml = pytest.mark.skipif(not _nvml_available(), reason="no NVML/NVIDIA GPU on this host")


@requires_nvml
async def test_gpu_status_reads_the_real_gpu_table(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    result = await call(server, "gpu_status")
    assert result["ok"] is True
    assert result["source"] == "nvml"
    assert result["count"] >= 1
    for gpu in result["gpus"]:
        assert gpu["total_bytes"] > 0
        assert gpu["free_bytes"] <= gpu["total_bytes"]
        assert gpu["total_mib"] == round(gpu["total_bytes"] / 1048576)
        assert isinstance(gpu["name"], str) and gpu["name"]


@requires_nvml
async def test_gpu_status_sees_all_four_gpus_on_the_reference_rig(harness: Harness) -> None:
    """The dev box has 2x5090 + 2x3090; assert the whole table is visible."""
    server = build_mcp(harness.watchdog())
    result = await call(server, "gpu_status")
    if result["count"] != 4:
        pytest.skip(f"not the 4-GPU reference rig (found {result['count']})")
    assert [gpu["index"] for gpu in result["gpus"]] == [0, 1, 2, 3]
    assert all(gpu["compute_capability"] for gpu in result["gpus"])


async def test_gpu_status_degrades_to_nvidia_smi(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NVML can be broken while the driver is fine; the CLI is the second try."""

    def boom() -> Any:
        raise RuntimeError("simulated NVML import failure")

    monkeypatch.setattr(wd_module, "_load_nvml", boom)
    monkeypatch.setattr(
        wd_module,
        "read_gpus_smi",
        lambda timeout=10.0: [
            {
                "index": 0,
                "name": "Fake 5090",
                "total_bytes": 34359738368,
                "free_bytes": 34359738368,
                "used_bytes": 0,
                "total_mib": 32768,
                "free_mib": 32768,
                "used_mib": 0,
                "utilization_pct": 0.0,
                "temperature_c": 40.0,
                "compute_capability": None,
            }
        ],
    )
    server = build_mcp(harness.watchdog())
    result = await call(server, "gpu_status")
    assert result["ok"] is True
    assert result["source"] == "nvidia-smi"
    assert any("simulated NVML import failure" in note for note in result["degraded"])


async def test_gpu_status_returns_an_explicit_error_when_everything_fails(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never raise, and never silently pretend there are no GPUs."""

    def boom() -> Any:
        raise RuntimeError("simulated NVML failure")

    def smi_boom(timeout: float = 10.0) -> Any:
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(wd_module, "_load_nvml", boom)
    monkeypatch.setattr(wd_module, "read_gpus_smi", smi_boom)

    server = build_mcp(harness.watchdog())
    result = await call(server, "gpu_status")
    assert result["ok"] is False
    assert result["error"]["code"] == "gpu_unavailable"
    assert "simulated NVML failure" in result["error"]["message"]
    assert "nvidia-smi not found" in result["error"]["message"]


# ===========================================================================
# rollback_update
# ===========================================================================


async def test_rollback_without_a_previous_release_says_so(harness: Harness) -> None:
    server = build_mcp(harness.watchdog())
    result = await call(server, "rollback_update", confirm=True)
    assert result["ok"] is False
    assert result["error"]["code"] == "no_previous_release"
    assert "no releases found" in result["error"]["message"]


async def test_rollback_with_a_single_release_says_so(harness: Harness) -> None:
    config, _ = harness.watchdog().load_config()
    (config.releases_dir / "v0.1.0").mkdir(parents=True)
    server = build_mcp(harness.watchdog())
    result = await call(server, "rollback_update", confirm=True)
    assert result["ok"] is False
    assert result["error"]["code"] == "no_previous_release"
    assert "only one release" in result["error"]["message"]


async def test_rollback_without_confirm_previews_the_change(harness: Harness) -> None:
    config, _ = harness.watchdog().load_config()
    _make_releases(config, ["v0.1.0", "v0.2.0"])
    (config.data_dir / "current.txt").write_text("v0.2.0\n", encoding="utf-8")

    server = build_mcp(harness.watchdog())
    result = await call(server, "rollback_update")
    assert result["ok"] is False
    assert result["error"]["code"] == "confirmation_required"
    assert "v0.2.0 to v0.1.0" in result["error"]["message"]
    assert (config.data_dir / "current.txt").read_text(encoding="utf-8").strip() == "v0.2.0"


@pytest.mark.timeout(120)
async def test_rollback_flips_the_pointer_and_restarts(harness: Harness) -> None:
    """On Windows the pointer is a ``current.txt`` file, not a symlink."""
    config, _ = harness.watchdog().load_config()
    _make_releases(config, ["v0.1.0", "v0.2.0"])
    pointer = config.data_dir / "current.txt"
    pointer.write_text("v0.2.0\n", encoding="utf-8")

    watchdog = harness.watchdog()
    result = await call(build_mcp(watchdog), "rollback_update", confirm=True, timeout_s=60.0)
    assert result["rolled_back_from"] == "v0.2.0"
    assert result["rolled_back_to"] == "v0.1.0"
    if os.name == "nt":
        assert result["pointer_kind"] == "file"
        assert Path(result["pointer_path"]).name == "current.txt"
        assert pointer.read_text(encoding="utf-8").strip() == "v0.1.0"
    assert result["restart"]["new_pid"] is not None
    if result["restart"].get("new_pid"):
        harness.extra_pids.append(int(result["restart"]["new_pid"]))


def _make_releases(config: Config, names: list[str]) -> None:
    """Create release dirs with increasing mtimes so ordering is unambiguous."""
    config.releases_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for offset, name in enumerate(names):
        path = config.releases_dir / name
        path.mkdir(exist_ok=True)
        os.utime(path, (now + offset, now + offset))


# ===========================================================================
# API key wrapper
# ===========================================================================


async def test_health_is_public_but_mcp_requires_the_api_key(harness: Harness) -> None:
    """The watchdog shares the main server's key, and /health stays open.

    An unauthenticated liveness probe is the point: a load balancer must not
    need a credential to discover that the server is wedged.
    """
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["server"]["api_key"] = "sf-watchdog-key-12345"
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    asgi, _server = create_asgi_app(harness.watchdog())
    transport = httpx.ASGITransport(app=asgi)
    async with httpx.AsyncClient(transport=transport, base_url="http://watchdog") as client:
        open_probe = await client.get("/health")
        assert open_probe.status_code == 503  # down, but answered without a key

        unauthorized = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1})
        assert unauthorized.status_code == 401
        assert unauthorized.json()["error"]["code"] == "invalid_api_key"

        wrong = await client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1},
            headers={"Authorization": "Bearer nope"},
        )
        assert wrong.status_code == 401


@pytest.mark.timeout(120)
async def test_watchdog_auth_tracks_a_rotated_key_and_accepts_the_pin(
    harness: Harness,
) -> None:
    """Per-request credentials: rotation bites immediately, both directions.

    The watchdog's contract is that config is re-read on every call; the
    credential was the one thing snapshotted at boot -- rotating the key
    through the app locked the operator out of the recovery surface (the
    watchdog kept accepting the old key and rejecting the new one) until the
    watchdog itself restarted. The MCP pairing PIN is also accepted as a
    bearer: it is the control-plane credential, and the watchdog is part of
    the control plane. And a credential with a >=0x80 byte must be a 401,
    never a TypeError-turned-500.
    """
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["server"]["api_key"] = "sf-first-key-12345"
    raw.setdefault("mcp", {})["pin"] = "87654321"
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    def status(client: httpx.Client, url: str, key: str | bytes) -> int:
        raw_key = key if isinstance(key, bytes) else key.encode("ascii")
        return client.post(
            url,
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            headers=httpx.Headers(
                [
                    (b"authorization", b"Bearer " + raw_key),
                    (b"accept", b"application/json, text/event-stream"),
                ]
            ),
        ).status_code

    with (
        WatchdogHttp(harness.watchdog(), harness.watchdog_port) as http,
        httpx.Client(timeout=10.0) as client,
    ):
        assert status(client, http.url, "sf-first-key-12345") != 401
        assert status(client, http.url, "87654321") != 401, (
            "a PIN-holder must not be locked out of the recovery surface"
        )
        assert status(client, http.url, "nope") == 401
        assert status(client, http.url, b"k\xe9y") == 401, (
            "a non-ASCII credential byte must be a 401, not a crash"
        )

        # Rotate the key on disk, exactly as the app's set_config does.
        raw["server"]["api_key"] = "sf-second-key-67890"
        harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        assert status(client, http.url, "sf-first-key-12345") == 401, (
            "the old key must stop working without a watchdog restart"
        )
        assert status(client, http.url, "sf-second-key-67890") != 401, (
            "the rotated key must work without a watchdog restart"
        )


# ===========================================================================
# 11. Import hygiene -- the invariant that makes all of the above possible
# ===========================================================================

WATCHDOG_MODULES = ("studioforge.watchdog.server", "studioforge.watchdog.__main__")

FORBIDDEN_PREFIXES = ("studioforge.core", "studioforge.api", "studioforge.db")

#: The only StudioForge modules the watchdog may touch.
ALLOWED_STUDIOFORGE = ("studioforge.config", "studioforge.watchdog")


def _module_path(dotted: str) -> Path:
    module = __import__(dotted, fromlist=["__file__"])
    path = getattr(module, "__file__", None)
    assert path, f"{dotted} has no __file__"
    return Path(path)


def _imported_names(path: Path) -> list[str]:
    """Every module name imported by a file, from its AST.

    AST rather than runtime inspection on purpose: a lazily imported
    ``studioforge.core`` inside a function would not show up in
    ``sys.modules`` during a test that never calls that function, and it is
    exactly the kind of "harmless" edit this invariant exists to catch.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                names.append("." * node.level + (node.module or ""))
            elif node.module:
                names.append(node.module)
    return names


@pytest.mark.parametrize("dotted", WATCHDOG_MODULES)
def test_watchdog_never_imports_the_app_machinery(dotted: str) -> None:
    """The invariant: no ``studioforge.core``/``api``/``db`` anywhere.

    Importing the app's machinery would drag the supervisor's process
    bookkeeping, the SQLite registry and the engine manager into the recovery
    process -- objects that take the very locks a wedged app is stuck on. That
    would make the watchdog exactly as stuck as the thing it repairs.
    """
    offenders = [
        name
        for name in _imported_names(_module_path(dotted))
        if any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES)
    ]
    assert offenders == [], f"{dotted} must not import {offenders}"


@pytest.mark.parametrize("dotted", WATCHDOG_MODULES)
def test_watchdog_only_imports_config_from_studioforge(dotted: str) -> None:
    strays = [
        name
        for name in _imported_names(_module_path(dotted))
        if name.startswith("studioforge")
        and not any(name == a or name.startswith(a + ".") for a in ALLOWED_STUDIOFORGE)
    ]
    assert strays == [], f"{dotted} may only import {ALLOWED_STUDIOFORGE}, found {strays}"


@pytest.mark.parametrize("dotted", WATCHDOG_MODULES)
def test_watchdog_third_party_imports_stay_minimal(dotted: str) -> None:
    """Only the small allowlist of third-party packages."""
    allowed_third_party = {
        "httpx",
        "psutil",
        "yaml",
        "pydantic",
        "mcp",
        "pynvml",
        "uvicorn",  # transport only, and a dependency of the MCP SDK itself
    }
    stdlib = set(sys.stdlib_module_names)
    strays = []
    for name in _imported_names(_module_path(dotted)):
        root = name.split(".")[0]
        if root in stdlib or root.startswith("studioforge") or root == "":
            continue
        if root not in allowed_third_party:
            strays.append(name)
    assert strays == [], f"{dotted} imports unexpected third-party modules: {strays}"


def test_watchdog_never_opens_the_registry_database() -> None:
    """SQLite locking is process-wide; waiting on the app's lock defeats the point.

    Checked against code rather than raw text, because the module *docstring*
    legitimately explains why it never touches ``registry.sqlite3``.
    """
    path = _module_path("studioforge.watchdog.server")
    assert "sqlite3" not in _imported_names(path)
    code = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(code):
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("db_path", "connect_db"), ast.dump(node)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "Database"


def test_watchdog_module_imports_without_the_app_present() -> None:
    """A fresh interpreter must import the watchdog without loading the app.

    Proves the AST assertion at runtime: if any of the app's packages appear in
    ``sys.modules`` after importing the watchdog, something imports them
    transitively even though the source text looks clean.
    """
    code = (
        "import sys\n"
        "import studioforge.watchdog.server as s\n"
        "bad = sorted(m for m in sys.modules if m.startswith(('studioforge.core',"
        " 'studioforge.api', 'studioforge.db')))\n"
        "print(','.join(bad))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert completed.stdout.strip() == "", f"leaked imports: {completed.stdout.strip()}"


# ===========================================================================
# Discovery unit checks
# ===========================================================================


def test_find_main_process_never_matches_the_watchdog_itself(harness: Harness) -> None:
    """A watchdog that mistakes itself for the app kills itself on restart."""
    config, _ = harness.watchdog().load_config()
    found = find_main_process(config)
    assert found is None or found.pid != os.getpid()


def test_flag_value_handles_both_spellings() -> None:
    from studioforge.watchdog.server import _flag_value

    assert _flag_value(["x", "--port", "8080"], "--port") == "8080"
    assert _flag_value(["x", "--port=8080"], "--port") == "8080"
    # Last occurrence wins, matching llama.cpp's own argument handling.
    assert _flag_value(["--port", "1", "--port", "2"], "--port") == "2"
    assert _flag_value(["x"], "--port") is None


def test_tail_file_on_a_missing_path_is_empty() -> None:
    from studioforge.watchdog.server import tail_file

    assert tail_file(Path("does-not-exist-anywhere.log"), 10) == []


async def test_poll_loop_survives_a_failing_health_check(harness: Harness) -> None:
    """The always-on loop must never die; a dead poll loop is a silent watchdog."""
    watchdog = harness.watchdog()
    calls = {"n": 0}

    async def exploding_health() -> dict[str, Any]:
        calls["n"] += 1
        raise RuntimeError("boom")

    watchdog.health = exploding_health  # type: ignore[method-assign]
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["watchdog"]["poll_interval_s"] = 1.0
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    task = asyncio.create_task(watchdog.poll_loop())
    await asyncio.sleep(1.4)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls["n"] >= 2, "the loop stopped after the first failure"


async def test_watchdog_accepts_the_pin_through_the_main_servers_carriers(
    harness: Harness,
) -> None:
    """A client that paired with the main /mcp via ``X-MCP-Pin`` (or ``?pin=``)
    must not get a 401 from the recovery surface -- and a wrong PIN in those
    carriers is still a 401 (2026-08-19)."""
    raw = yaml.safe_load(harness.config_path.read_text(encoding="utf-8"))
    raw["server"]["api_key"] = None
    raw.setdefault("mcp", {})["pin"] = "87654321"
    harness.config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    body = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    accept = {"accept": "application/json, text/event-stream"}

    def post(client: httpx.Client, url: str, headers: dict[str, str]) -> httpx.Response:
        return client.post(url, json=body, headers={**accept, **headers})

    with (
        WatchdogHttp(harness.watchdog(), harness.watchdog_port) as http,
        httpx.Client(timeout=10.0) as client,
    ):
        assert post(client, http.url, {"X-MCP-Pin": "87654321"}).status_code != 401
        assert post(client, http.url, {"X-StudioForge-Pin": "87654321"}).status_code != 401
        assert post(client, http.url + "?pin=87654321", {}).status_code != 401
        wrong = post(client, http.url, {"X-MCP-Pin": "00000000"})
        assert wrong.status_code == 401
        message = wrong.json()["error"]["message"]
        assert "X-MCP-Pin" in message and "Setup" in message and "sfctl servers add" in message
