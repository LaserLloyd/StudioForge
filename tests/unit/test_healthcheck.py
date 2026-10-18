"""The container HEALTHCHECK follows ``server.tls.mode``.

The image used to probe a hard-coded ``http://127.0.0.1:1234/health``, so the
moment TLS was on (``setup --exposure lan`` turns it on) a healthy gateway
reported "unhealthy" forever. ``python -m studioforge.healthcheck`` reads the
effective config instead. These tests run a real loopback server (plain and
TLS, on ephemeral ports) and the real command in a subprocess.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from studioforge import healthcheck
from studioforge.config import Config
from studioforge.tlsutil import resolve_tls

REPO = Path(__file__).resolve().parents[2]


class _Health(BaseHTTPRequestHandler):
    body = {"status": "ok"}
    code = 200

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        payload = json.dumps(type(self).body).encode()
        self.send_response(type(self).code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.path == "/health":
            self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # silence the test output
        pass


def _serve(tls: tuple[Path, Path] | None = None) -> Iterator[int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    if tls is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(str(tls[0]), str(tls[1]))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5.0)


@pytest.fixture
def plain_port() -> Iterator[int]:
    yield from _serve()


@pytest.fixture
def tls_setup(tmp_path: Path) -> Iterator[tuple[int, Path]]:
    """A TLS listener using the same self-signed pair the server would generate."""
    data = tmp_path / "data"
    config = Config(data_dir=data, server={"tls": {"mode": "self_signed"}})
    material = resolve_tls(config)
    assert material is not None
    yield from ((port, data) for port in _serve((material.cert_file, material.key_file)))


@pytest.fixture(autouse=True)
def _reset_handler() -> Iterator[None]:
    _Health.body, _Health.code = {"status": "ok"}, 200
    yield
    _Health.body, _Health.code = {"status": "ok"}, 200


def _config(tmp_path: Path, port: int, mode: str) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        server={"host": "0.0.0.0", "port": port, "tls": {"mode": mode}},
    )


def test_off_probes_http_and_tls_probes_https(tmp_path: Path) -> None:
    assert healthcheck.candidate_urls(_config(tmp_path, 1234, "off")) == [
        "http://127.0.0.1:1234/health"
    ]
    assert healthcheck.candidate_urls(_config(tmp_path, 4321, "self_signed")) == [
        "https://127.0.0.1:4321/health"
    ]


def test_an_unreadable_config_tries_both_schemes() -> None:
    assert healthcheck.candidate_urls(None) == [
        "http://127.0.0.1:1234/health",
        "https://127.0.0.1:1234/health",
    ]


def test_a_healthy_plain_gateway_passes(plain_port: int) -> None:
    assert healthcheck.probe(f"http://127.0.0.1:{plain_port}/health") is True
    assert healthcheck.main([f"http://127.0.0.1:{plain_port}/health"]) == 0


def test_a_healthy_tls_gateway_passes_with_a_self_signed_certificate(
    tls_setup: tuple[int, Path],
) -> None:
    port, _ = tls_setup
    url = f"https://127.0.0.1:{port}/health"
    assert healthcheck.probe(url) is True  # unverified: the loopback cert has no matching name
    assert healthcheck.main([url]) == 0
    # The old hard-coded probe: http against a TLS listener is "unhealthy".
    assert healthcheck.probe(f"http://127.0.0.1:{port}/health") is False


@pytest.mark.parametrize(
    ("code", "body"),
    [(503, {"status": "ok"}), (200, {"status": "degraded"}), (200, ["not", "an", "object"])],
)
def test_an_unhealthy_answer_fails(plain_port: int, code: int, body: object) -> None:
    _Health.code, _Health.body = code, body  # type: ignore[assignment]
    assert healthcheck.main([f"http://127.0.0.1:{plain_port}/health"]) == 1


def test_a_closed_port_fails_without_raising() -> None:
    with socket.socket() as sock:  # a port that was free a moment ago, and is closed now
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    assert healthcheck.probe(f"http://127.0.0.1:{port}/health", timeout=1.0) is False
    assert healthcheck.main([f"http://127.0.0.1:{port}/health"]) == 1


def _run_command(data: Path, **env: str) -> int:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("SF_")}
    clean.update({"SF_DATA_DIR": str(data), "SF_GPU_PROBE": "null", **env})
    done = subprocess.run(  # noqa: S603 - the interpreter running this test, a module of this repo
        [sys.executable, "-m", "studioforge.healthcheck"],
        env=clean,
        cwd=str(REPO),
        capture_output=True,
        timeout=60,
        check=False,
    )
    return done.returncode


def test_the_command_follows_the_env_and_config_the_container_runs_with(
    tls_setup: tuple[int, Path], tmp_path: Path
) -> None:
    port, data = tls_setup
    # TLS on (as `setup --exposure lan` leaves it): https, so a TLS listener is healthy ...
    assert _run_command(data, SF_SERVER__PORT=str(port), SF_SERVER__TLS__MODE="self_signed") == 0
    # ... and the same listener probed as plain http (the old behaviour) is not.
    assert _run_command(data, SF_SERVER__PORT=str(port), SF_SERVER__TLS__MODE="off") == 1


def test_the_command_probes_plain_http_when_tls_is_off(plain_port: int, tmp_path: Path) -> None:
    assert _run_command(tmp_path / "d", SF_SERVER__PORT=str(plain_port)) == 0


def test_the_dockerfile_uses_the_tls_aware_probe() -> None:
    dockerfile = (REPO / "Dockerfile").read_text(encoding="utf-8")
    (line,) = [ln for ln in dockerfile.splitlines() if ln.lstrip().startswith('CMD ["python"')]
    assert "studioforge.healthcheck" in line
    assert "http://" not in line, "a hard-coded scheme is what broke the check under TLS"
