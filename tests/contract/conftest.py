"""Fixtures for the OpenAI contract suite.

These tests run against a REAL server process with a REAL `llama-server`
engine and REAL GGUF models, driven by the official ``openai`` client. That is
the point: parity with LM Studio is a claim about what an actual OpenAI client
observes, and only an end-to-end stack can verify it.

**This suite takes the GPUs, and it is opt-in twice over.** On 2026-08-18 a
coding agent ran ``pytest tests``; ``testpaths`` collected this directory, the
fixture below started the real gateway against the real rig, and three
``llama-server`` children were still holding ~25 GiB of VRAM afterwards. Two
independent gates now stand in front of that (DECISIONS.md D23):

* every item here is marked ``contract``, and ``addopts = -m 'not contract'``
  in ``pyproject.toml`` deselects the mark by default; and
* ``SF_RUN_CONTRACT=1`` must be set. Belt and braces on purpose -- ``-m
  contract`` alone must not be enough for a stray CI job to reach live
  hardware, and a marker is one ``-m`` away from being overridden.

**It also no longer opens the live data directory.** The fixture used to point
``Config(data_dir=...)`` at ``<repo>/../data``: the running install's
``registry.sqlite3``, logs and downloads. Tests wrote the production registry.
They now get a temporary data dir whose ``engines/<tag>`` is a
junction/symlink to the installed engine -- 670 MB per engine makes copying
absurd, and the binary is the only thing from that tree this suite actually
needs. Where the link cannot be created the suite skips rather than falling
back to the live directory.
"""

from __future__ import annotations

import base64
import io
import os
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Where the *installed* engine lives. Read-only as far as this suite is
#: concerned: nothing but ``engines/<tag>/`` is borrowed from it. Follows the
#: same rule as the app itself -- ``SF_DATA_DIR`` if set, else ``<repo>/data``.
_DATA_ENV = os.environ.get("SF_DATA_DIR", "").strip()
DEV_DATA_DIR = Path(_DATA_ENV) if _DATA_ENV else REPO_ROOT / "data"
#: The GGUF library to run against. Env-only and deliberately without a
#: default: a hard-coded path is right on exactly one machine, and a *wrong*
#: default here would make the suite skip with a confusing reason instead of
#: telling the operator to point it somewhere.
_MODELS_ENV = os.environ.get("SF_TEST_MODELS_DIR", "").strip()
MODELS_DIR = Path(_MODELS_ENV) if _MODELS_ENV else None
ENGINE_TAG = "b10425"

#: The explicit "yes, use the real GPUs" switch.
RUN_ENV = "SF_RUN_CONTRACT"

# Small, fast models chosen so the suite stays runnable: a 0.5B chat model loads
# in a couple of seconds, and the 4GB VL model is the cheapest real vision model
# in the library.
TINY_CHAT = "Qwen2.5-0.5B-Instruct-Q8_0"
VISION_MODEL = "Qwen2.5-VL-7B-Abliterated-Caption-it.Q4_K_S"
EMBEDDING_MODEL = "qwen3-embedding-8b-q4_k_m"

API_KEY = "sf-contract-test-key"


def engine_binary() -> Path:
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    return DEV_DATA_DIR / "engines" / ENGINE_TAG / name


requires_engine = pytest.mark.skipif(
    not engine_binary().is_file(),
    reason=f"llama-server engine not installed at {engine_binary()}",
)
requires_models = pytest.mark.skipif(
    MODELS_DIR is None or not MODELS_DIR.is_dir(),
    reason=(
        "set SF_TEST_MODELS_DIR to your GGUF library to run the contract suite"
        if MODELS_DIR is None
        else f"model library not found at {MODELS_DIR}"
    ),
)


def contract_enabled() -> bool:
    return os.environ.get(RUN_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """Mark this directory ``contract``, and skip it without the opt-in env.

    The marker is applied here rather than by hand in each module so a new
    contract test cannot be added *without* it -- the failure mode being
    guarded against is an unmarked test quietly rejoining the default run.

    Only items under this directory are touched: ``pytest_collection_modifyitems``
    is a global hook and receives the whole collection.
    """
    here = Path(__file__).parent
    enabled = contract_enabled()
    skip = pytest.mark.skip(
        reason=(
            f"contract suite loads real models onto the real GPUs; set {RUN_ENV}=1 "
            "to run it (see DECISIONS.md D23)"
        )
    )
    for item in items:
        try:
            path = Path(str(item.fspath))
        except Exception:  # noqa: BLE001 - pragma: no cover - exotic collectors
            continue
        if here not in path.parents:
            continue
        item.add_marker(pytest.mark.contract)
        if not enabled:
            item.add_marker(skip)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tiny_png_data_url(size: int = 64) -> str:
    """A small image with an unmistakable feature, for vision assertions.

    A solid red square on white: any working vision model should mention red
    and/or a square, which makes the assertion robust without being brittle
    about exact wording.
    """
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    inset = size // 6
    draw.rectangle([inset, inset, size - inset, size - inset], fill=(220, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class ServerHandle:
    def __init__(self, base_url: str, api_key: str, app: Any) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.app = app

    @property
    def openai_base(self) -> str:
        return f"{self.base_url}/v1"

    def resolve_model(self, needle: str) -> str | None:
        """Find a served model id containing ``needle`` (case-insensitive)."""
        response = httpx.get(
            f"{self.openai_base}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30.0,
        )
        response.raise_for_status()
        wanted = needle.lower()
        for entry in response.json()["data"]:
            if wanted in entry["id"].lower():
                return str(entry["id"])
        return None


def _link_directory(link: Path, target: Path) -> bool:
    """Point ``link`` at ``target`` without copying it. False if impossible.

    A directory junction on Windows (``mklink /J``), which needs no
    administrator rights and no Developer Mode -- unlike a symlink, which needs
    one of the two and would make this suite un-runnable on a default install.
    ``symlink_to`` is tried first anyway because it is the whole answer on
    POSIX.

    Verified before use: ``shutil.rmtree`` (and therefore pytest's temp-dir
    retention sweep) removes a junction without recursing through it, so
    cleaning up this fixture's data dir cannot delete the installed engine.
    """
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError, AttributeError):
        pass
    else:
        return link.is_dir()
    if os.name != "nt":
        return False
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return link.is_dir()


@pytest.fixture(scope="session")
def contract_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A throwaway data dir that borrows only the installed engine binary.

    Everything the gateway writes -- ``registry.sqlite3``, ``config.yaml``,
    logs, downloads -- lands here and dies with the session. The one thing that
    cannot be synthesised is the 670 MB engine directory, so ``engines/<tag>``
    is linked to the installed one.

    ``engine.json``/``active.json`` bookkeeping therefore still lands inside the
    linked engine directory (the manager writes meta next to the binary it
    verified). That is a knowingly-accepted write to shared state: it is
    idempotent, it is the same tag the live server already uses, and it is the
    only alternative to either copying 670 MB per session or reopening the live
    registry, which is the thing this fixture exists to stop.
    """
    root = tmp_path_factory.mktemp("sf-contract-data")
    engine_src = DEV_DATA_DIR / "engines" / ENGINE_TAG
    if not engine_src.is_dir():
        pytest.skip(f"engine {ENGINE_TAG} is not installed at {engine_src}")
    if not _link_directory(root / "engines" / ENGINE_TAG, engine_src):
        pytest.skip(
            f"could not link {engine_src} into a temporary data dir; the contract suite "
            "refuses to fall back to the live data directory"
        )
    return root


@pytest.fixture(scope="session")
def live_server(contract_data_dir: Path) -> Iterator[ServerHandle]:
    """Start the real gateway in a background thread for the whole session."""
    if MODELS_DIR is None:
        pytest.skip("set SF_TEST_MODELS_DIR to your GGUF library to run the contract suite")
    import uvicorn

    from studioforge.api.app import create_app
    from studioforge.config import Config

    port = free_port()
    config = Config(
        data_dir=contract_data_dir,
        server={
            "host": "127.0.0.1",
            "port": port,
            "api_key": API_KEY,
            "request_timeout_s": 600.0,
            "drain_timeout_s": 5.0,
        },
        gui={"enabled": False, "port": free_port()},
        watchdog={"enabled": False, "port": free_port()},
        models={
            "dir": MODELS_DIR,
            "default_ctx": 4096,
            # Aim == floor: contract loads stay small and deterministic.
            "target_ctx": 4096,
            "default_ttl_s": 600,
            "default_parallel": 1,
            "auto_load_pinned": False,
        },
        gateway={
            # The suite serves its fixture image from a loopback test server,
            # which the SSRF guard refuses by default.
            "allow_private_image_hosts": True,
        },
        engine={"pinned_tag": ENGINE_TAG},
        logging={"level": "INFO"},
    )
    app = create_app(config, start_background=True)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, name="contract-server", daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        if not thread.is_alive():
            raise RuntimeError("contract server thread died during startup")
        time.sleep(0.25)
    else:
        raise RuntimeError(f"contract server did not become healthy at {base_url}")

    try:
        yield ServerHandle(base_url, API_KEY, app)
    finally:
        server.should_exit = True
        thread.join(timeout=90)


@pytest.fixture(scope="session")
def client(live_server: ServerHandle) -> Any:
    """The official OpenAI client, pointed at StudioForge."""
    from openai import OpenAI

    return OpenAI(
        base_url=live_server.openai_base,
        api_key=live_server.api_key,
        max_retries=0,
        timeout=600.0,
    )


@pytest.fixture(scope="session")
def chat_model(live_server: ServerHandle) -> str:
    model = live_server.resolve_model(TINY_CHAT)
    if model is None:
        pytest.skip(f"chat model matching {TINY_CHAT!r} not present in {MODELS_DIR}")
    return model


@pytest.fixture(scope="session")
def vision_model(live_server: ServerHandle) -> str:
    model = live_server.resolve_model(VISION_MODEL)
    if model is None:
        pytest.skip(f"vision model matching {VISION_MODEL!r} not present in {MODELS_DIR}")
    return model


@pytest.fixture(scope="session")
def embedding_model(live_server: ServerHandle) -> str:
    model = live_server.resolve_model(EMBEDDING_MODEL)
    if model is None:
        pytest.skip(f"embedding model matching {EMBEDDING_MODEL!r} not present")
    return model


@pytest.fixture(scope="session")
def httpbin_image() -> Iterator[str]:
    """Serve a PNG over real HTTP so the server-side fetch path is exercised.

    Local rather than a public URL: the test must verify our fetch/decode/resize
    plumbing, not the internet.
    """
    import http.server
    import socketserver

    from PIL import Image, ImageDraw

    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).ellipse([16, 16, 112, 112], fill=(20, 90, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib naming
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: Any) -> None:
            return

    port = free_port()
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}/circle.png"
        finally:
            httpd.shutdown()


@pytest.fixture
def raw(live_server: ServerHandle) -> Iterator[httpx.Client]:
    """Raw HTTP client for assertions the openai SDK abstracts away."""
    with httpx.Client(
        base_url=live_server.base_url,
        headers={"Authorization": f"Bearer {live_server.api_key}"},
        timeout=600.0,
    ) as http_client:
        yield http_client
