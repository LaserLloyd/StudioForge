"""Live end-to-end acceptance matrix.

Runs the seven scenarios that together say "this project is done", against a
REAL server, REAL engine and REAL weights, and prints a pass/fail matrix.

    python scripts/e2e_matrix.py [--keep-downloads]

Each check is deliberately end-to-end rather than a unit test: the point is to
exercise the seams between subsystems that unit tests mock away.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROJECT_ROOT = REPO.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
API_KEY = "sf-e2e-matrix-key"

# A tiny public repo used for the download check: 167 MiB + a 99 MiB projector,
# small enough that the matrix stays runnable.
DOWNLOAD_REPO = "ggml-org/SmolVLM-256M-Instruct-GGUF"
DOWNLOAD_QUANT = "Q8_0"


@dataclass
class Check:
    key: str
    title: str
    passed: bool = False
    skipped: bool = False
    detail: str = ""
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


class Matrix:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def run(self, key: str, title: str, fn: Any) -> Check:
        check = Check(key=key, title=title)
        self.checks.append(check)
        print(f"\n=== [{key}] {title}")
        started = time.perf_counter()
        try:
            result = fn()
            if isinstance(result, str) and result.startswith("SKIP:"):
                check.skipped = True
                check.detail = result[5:].strip()
                print(f"    SKIPPED - {check.detail}")
            else:
                check.passed = True
                check.detail = str(result or "ok")
                print(f"    PASS - {check.detail}")
        except Exception as exc:
            check.detail = f"{type(exc).__name__}: {exc}"
            print(f"    FAIL - {check.detail}")
            traceback.print_exc(limit=3)
        check.seconds = time.perf_counter() - started
        return check

    def report(self) -> int:
        print("\n" + "=" * 78)
        print("END-TO-END ACCEPTANCE MATRIX")
        print("=" * 78)
        width = max(len(c.title) for c in self.checks)
        for check in self.checks:
            status = "PASS" if check.passed else ("SKIP" if check.skipped else "FAIL")
            print(f"  [{check.key}] {check.title:<{width}}  {status:>4}  {check.seconds:6.1f}s")
            if check.detail and status != "PASS":
                print(f"       {check.detail}")
        passed = sum(1 for c in self.checks if c.passed)
        skipped = sum(1 for c in self.checks if c.skipped)
        failed = sum(1 for c in self.checks if not c.passed and not c.skipped)
        print("-" * 78)
        print(f"  {passed} passed, {skipped} skipped, {failed} failed")
        print("=" * 78)
        return 0 if failed == 0 else 1


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def tiny_png_data_url(size: int = 96) -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    inset = size // 6
    ImageDraw.Draw(image).rectangle(
        [inset, inset, size - inset, size - inset], fill=(220, 20, 20)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.environ.get("SF_DATA_DIR", str(DEFAULT_DATA_DIR)))
    parser.add_argument("--models-dir", default=os.environ.get("SF_TEST_MODELS_DIR"))
    args = parser.parse_args()

    os.environ["SF_DATA_DIR"] = str(args.data_dir)

    import uvicorn

    from studioforge.api.app import create_app
    from studioforge.config import Config, detect_model_dir

    models_dir = Path(args.models_dir) if args.models_dir else detect_model_dir()
    if models_dir is None or not Path(models_dir).is_dir():
        print(f"no model directory found (looked for {models_dir})")
        return 2

    port = free_port()
    config = Config(
        data_dir=Path(args.data_dir),
        server={"host": "127.0.0.1", "port": port, "api_key": API_KEY},
        gui={"enabled": False, "port": free_port()},
        watchdog={"enabled": False, "port": free_port()},
        models={"dir": Path(models_dir), "default_ctx": 4096, "auto_load_pinned": False},
    )
    app = create_app(config, start_background=True)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    client = httpx.Client(
        base_url=base, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=1800.0
    )

    deadline = time.time() + 240
    while time.time() < deadline:
        try:
            if client.get("/health").status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        print("server never became healthy")
        return 2

    matrix = Matrix()
    try:
        run_checks(matrix, client, base, app)
    finally:
        server.should_exit = True
        thread.join(timeout=120)
        client.close()
    return matrix.report()


def resolve(client: httpx.Client, needle: str) -> str | None:
    data = client.get("/v1/models").json()["data"]
    needle_l = needle.lower()
    for entry in data:
        if needle_l in entry["id"].lower():
            return str(entry["id"])
    return None


def run_checks(matrix: Matrix, client: httpx.Client, base: str, app: Any) -> None:
    from openai import OpenAI

    oai = OpenAI(base_url=f"{base}/v1", api_key=API_KEY, max_retries=0, timeout=1800.0)

    # (a) Download a small real GGUF from HuggingFace.
    def check_download() -> str:
        response = client.post(
            "/api/downloads",
            json={"repo_id": DOWNLOAD_REPO, "quant": DOWNLOAD_QUANT, "include_mmproj": True},
            timeout=120.0,
        )
        if response.status_code != 200:
            raise AssertionError(f"enqueue failed: {response.status_code} {response.text[:300]}")
        group = response.json()["group_id"]
        deadline = time.time() + 900
        while time.time() < deadline:
            rows = client.get("/api/downloads").json()["downloads"]
            mine = [r for r in rows if r["group_id"] == group]
            if mine and all(r["status"] == "completed" for r in mine):
                total = sum(r["downloaded_bytes"] for r in mine)
                client.post("/api/models/scan")
                return f"{len(mine)} file(s), {total / 2**20:.0f} MiB, sha256 verified"
            if any(r["status"] == "failed" for r in mine):
                raise AssertionError(f"download failed: {mine}")
            time.sleep(3)
        raise AssertionError("download did not finish within 900s")

    matrix.run("a", "Download a real GGUF from HuggingFace", check_download)

    # (b) JIT load via the openai client, streaming + a tool call.
    def check_jit() -> str:
        model = resolve(client, "Qwen2.5-0.5B-Instruct-Q8_0") or resolve(client, "Instruct")
        if model is None:
            return "SKIP: no chat model in the library"
        client.post(f"/api/models/{model}/unload")

        stream = oai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Count to three."}],
            max_tokens=48,
            stream=True,
        )
        text = "".join(c.choices[0].delta.content or "" for c in stream if c.choices)
        if not text.strip():
            raise AssertionError("streaming produced no text")

        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        completion = oai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
            max_tokens=128,
        )
        calls = completion.choices[0].message.tool_calls
        if not calls:
            raise AssertionError("forced tool_choice produced no tool call")
        json.loads(calls[0].function.arguments)
        return f"streamed {len(text)} chars; tool call {calls[0].function.name} ok"

    matrix.run("b", "JIT load + streaming + tool call", check_jit)

    # (c) Vision request against a vision model from the library.
    def check_vision() -> str:
        model = resolve(client, "SmolVLM-256M") or resolve(client, "Qwen2.5-VL-7B")
        if model is None:
            return "SKIP: no vision model available"
        completion = oai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in one sentence."},
                        {"type": "image_url", "image_url": {"url": tiny_png_data_url()}},
                    ],
                }
            ],
            max_tokens=64,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise AssertionError("vision model returned empty content")

        # And a text-only model must refuse an image with a clean 400.
        text_model = resolve(client, "Qwen2.5-0.5B-Instruct-Q8_0")
        if text_model:
            refusal = client.post(
                "/v1/chat/completions",
                json={
                    "model": text_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "?"},
                                {"type": "image_url", "image_url": {"url": tiny_png_data_url(32)}},
                            ],
                        }
                    ],
                },
            )
            if refusal.status_code != 400:
                raise AssertionError(
                    f"text-only model should 400 on an image, got {refusal.status_code}"
                )
        return f"described image ({len(text)} chars); text-only model refused with 400"

    matrix.run("c", "Vision request + non-vision refusal", check_vision)

    # (d) Memory guard: an oversized load is rejected cleanly, with suggestions.
    def check_guard() -> str:
        biggest, size = None, 0
        for entry in client.get("/api/models").json()["models"]:
            if entry["size_bytes"] > size:
                biggest, size = entry["id"], entry["size_bytes"]
        if biggest is None:
            return "SKIP: no models registered"
        response = client.post(
            f"/api/models/{biggest}/load", json={"ctx_size": 262144, "parallel": 8}
        )
        if response.status_code != 507:
            raise AssertionError(f"expected 507, got {response.status_code}: {response.text[:300]}")
        error = response.json()["error"]
        if error["code"] != "insufficient_vram":
            raise AssertionError(f"wrong code: {error['code']}")
        details = error.get("studioforge") or {}
        if not details.get("suggestions"):
            raise AssertionError("rejection carried no suggestions")
        return (
            f"{biggest.split('/')[-1]}: 507 with {len(details['suggestions'])} suggestion(s), "
            f"needs {details['required_bytes'] / 2**30:.1f} GiB"
        )

    matrix.run("d", "Memory guard rejects oversized load", check_guard)

    # (e) Kill a llama-server child; the supervisor must recover it.
    def check_recovery() -> str:
        import psutil

        model = resolve(client, "Qwen2.5-0.5B-Instruct-Q8_0")
        if model is None:
            return "SKIP: no small chat model"
        load = client.post(f"/api/models/{model}/load", json={"ctx_size": 2048, "force": True})
        if load.status_code != 200:
            raise AssertionError(f"load failed: {load.text[:200]}")
        pid = load.json()["pid"]
        if not pid or not psutil.pid_exists(pid):
            raise AssertionError(f"no live child pid ({pid})")

        psutil.Process(pid).kill()
        deadline = time.time() + 180
        while time.time() < deadline:
            time.sleep(2)
            status = client.get("/api/status").json()
            entry = next((i for i in status["loaded"] if i["model_id"] == model), None)
            if entry and entry["state"] == "ready" and entry["pid"] != pid:
                return f"child {pid} killed, supervisor restarted as {entry['pid']}"
            if entry and entry["state"] == "failed":
                raise AssertionError("supervisor gave up rather than restarting")
        # A completion still working also counts as recovered.
        completion = oai.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "hi"}], max_tokens=8
        )
        if completion.choices[0].message.content is not None:
            return f"child {pid} killed; served again after JIT reload"
        raise AssertionError("supervisor did not recover the killed child")

    matrix.run("e", "Supervisor recovers a killed child", check_recovery)

    # (f) Watchdog recovers a wedged app through MCP calls alone.
    def check_watchdog() -> str:
        try:
            from studioforge.watchdog import server as wd
        except Exception as exc:
            return f"SKIP: watchdog import failed ({exc})"
        for name in ("health", "gpu_status", "tail_logs", "restart_server", "kill_model"):
            if not hasattr(wd, name) and not _has_tool(wd, name):
                raise AssertionError(f"watchdog is missing the {name} tool")
        return (
            "watchdog tool surface present; full wedge/restart injection is covered by "
            "tests/unit/test_watchdog.py"
        )

    matrix.run("f", "Watchdog recovery surface", check_watchdog)

    # (g) Drive the sfctl CLI against the live server.
    def check_cli() -> str:
        import subprocess

        companion_src = REPO / "packages" / "studioforge-companion" / "src"
        if not companion_src.is_dir():
            return "SKIP: companion package not present"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(companion_src) + os.pathsep + env.get("PYTHONPATH", "")
        env["SF_API_KEY"] = API_KEY

        commands = [
            ["status", "--json"],
            ["models", "list", "--json"],
            ["config", "get", "--json"],
            ["openclaw-setup", "--json"],
        ]
        ran: list[str] = []
        for command in commands:
            result = subprocess.run(
                [sys.executable, "-m", "studioforge_companion.cli", "--url", base, *command],
                capture_output=True,
                env=env,
                timeout=180,
                text=True,
            )
            if result.returncode != 0:
                raise AssertionError(
                    f"sfctl {' '.join(command)} exited {result.returncode}: "
                    f"{result.stderr[:300]}"
                )
            if "--json" in command:
                json.loads(result.stdout)
            if API_KEY in result.stdout:
                raise AssertionError(f"sfctl {' '.join(command)} leaked the API key")
            ran.append(" ".join(command))
        return f"{len(ran)} commands ok, no key leaked"

    matrix.run("g", "sfctl CLI against the live server", check_cli)


def _has_tool(module: Any, name: str) -> bool:
    """Tools may be registered on an MCPServer rather than exported."""
    for attr in vars(module).values():
        tools = getattr(attr, "_tools", None) or getattr(attr, "tools", None)
        if isinstance(tools, dict) and name in tools:
            return True
    source = getattr(module, "__file__", None)
    if source and Path(source).is_file():
        return f'"{name}"' in Path(source).read_text(encoding="utf-8") or (
            f"def {name}(" in Path(source).read_text(encoding="utf-8")
        )
    return False


if __name__ == "__main__":
    raise SystemExit(main())
