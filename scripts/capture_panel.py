#!/usr/bin/env python3
"""Capture the panel's README screenshots from the demo server, in headless Chrome.

    uv run python scripts/panel_demo.py &              # the fake rig on :18080
    uv run python scripts/capture_panel.py             # writes docs/screenshots/readme/*.png
    uv run python scripts/capture_panel.py --only status models

Drives Chrome over the DevTools protocol (no extra dependency beyond ``websockets``, which
the server already needs). Only the demo ports may be captured: the allowlist below is
127.0.0.1 ports 18080-18199, so this can never screenshot the live panel (a real library,
real paths, real keys). Each PNG is quantised until it is at most 400 KB.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots" / "readme"
DEBUG_PORT = 18222
ALLOWED_PORTS = range(18080, 18200)
MAX_BYTES = 400 * 1024
SIZE = (1440, 900)

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chromium",
)

# Click a button or tab whose visible text starts with ``text`` (case-insensitive).
CLICK_JS = """
(text) => {
  const want = text.toLowerCase();
  const nodes = [...document.querySelectorAll('button, .q-tab, .q-item, [role=button]')];
  const hit = nodes.find(n => (n.innerText || '').trim().toLowerCase().startsWith(want));
  if (!hit) return 'missing: ' + text;
  hit.click();
  return 'clicked';
}
"""

TYPE_JS = """
(selector, value) => {
  const el = document.querySelector(selector);
  if (!el) return 'missing: ' + selector;
  const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, 'value').set.call(el, value);
  el.dispatchEvent(new Event('input', {bubbles: true}));
  return 'typed';
}
"""


class Shot:
    """One screenshot: where to go, what to do there, what to call it."""

    def __init__(self, name: str, path: str, *, wait: float = 3.0, steps: list[Any] | None = None):
        self.name, self.path, self.wait, self.steps = name, path, wait, steps or []


SHOTS = [
    Shot("panel-status", "/", wait=4.0),
    Shot("panel-models", "/?section=Models", wait=5.0),
    Shot("panel-model-drawer", "/?section=Models&model=Qwen3-32B-Q4_K_M", wait=6.0),
    Shot(
        "panel-test-chat",
        "/?section=Test%20Chat",
        wait=3.0,
        steps=[
            ("type", "textarea.sfc-composer, .sfc-composer textarea", "Explain in two lines why a model server plans VRAM before loading."),
            ("click", "Send"),
            ("sleep", 6.0),
        ],
    ),
    Shot("panel-downloads", "/?section=Models&filter=downloads", wait=4.0),
    Shot("panel-advanced", "/?section=Advanced&page=serving", wait=4.0),
]


def chrome_path() -> str:
    for candidate in CHROME_CANDIDATES:
        found = shutil.which(candidate) or (candidate if Path(candidate).exists() else None)
        if found:
            return found
    raise SystemExit("no Chrome, Chromium or Edge found")


def check_url(base: str) -> str:
    parsed = urlparse(base)
    if parsed.hostname not in ("127.0.0.1", "localhost") or parsed.port not in ALLOWED_PORTS:
        raise SystemExit(
            f"refusing {base}: only the demo panel on 127.0.0.1:18080-18199 may be captured"
        )
    return base.rstrip("/")


class Cdp:
    def __init__(self, ws_url: str) -> None:
        self.ws = connect(ws_url, max_size=64 * 1024 * 1024)
        self.next_id = 0

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self.next_id += 1
        self.ws.send(json.dumps({"id": self.next_id, "method": method, "params": params}))
        while True:
            message = json.loads(self.ws.recv(timeout=60))
            if message.get("id") == self.next_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result.get("result", {}).get("value")


def shrink(png: bytes) -> bytes:
    """Quantise until the file fits the size budget (the glacier palette is small)."""
    from PIL import Image

    if len(png) <= MAX_BYTES:
        return png
    image = Image.open(io.BytesIO(png)).convert("RGB")
    for colours in (256, 128, 64, 32):
        out = io.BytesIO()
        image.quantize(colors=colours, method=Image.Quantize.MEDIANCUT).save(
            out, format="PNG", optimize=True
        )
        if out.tell() <= MAX_BYTES:
            return out.getvalue()
    return out.getvalue()


def run(base: str, only: set[str], out: Path = OUT) -> int:
    base = check_url(base)
    out.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="sf-capture-")
    chrome = subprocess.Popen(  # noqa: S603 - fixed argv, a local browser
        [
            chrome_path(), "--headless=new", f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={profile}", f"--window-size={SIZE[0]},{SIZE[1]}",
            "--hide-scrollbars", "--no-first-run", "--disable-gpu", "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        targets: list[dict[str, Any]] = []
        for _ in range(50):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json/list") as resp:  # noqa: S310
                    targets = json.load(resp)
                if targets:
                    break
            except OSError:
                time.sleep(0.2)
        page = next(t for t in targets if t.get("type") == "page")
        cdp = Cdp(page["webSocketDebuggerUrl"])
        cdp.call("Page.enable")
        cdp.call(
            "Emulation.setDeviceMetricsOverride",
            width=SIZE[0], height=SIZE[1], deviceScaleFactor=1, mobile=False,
        )
        for shot in SHOTS:
            if only and shot.name not in only:
                continue
            cdp.call("Page.navigate", url=base + shot.path)
            time.sleep(shot.wait)
            for step in shot.steps:
                kind = step[0]
                if kind == "sleep":
                    time.sleep(float(step[1]))
                elif kind == "click":
                    print(f"  click {step[1]!r}: {cdp.evaluate(f'({CLICK_JS})({json.dumps(step[1])})')}")
                elif kind == "type":
                    print(f"  type: {cdp.evaluate(f'({TYPE_JS})({json.dumps(step[1])}, {json.dumps(step[2])})')}")
                    time.sleep(0.3)
            data = cdp.call("Page.captureScreenshot", format="png")["data"]
            png = shrink(base64.b64decode(data))
            target = out / f"{shot.name}.png"
            target.write_bytes(png)
            print(f"{shot.name}: {len(png) // 1024} KB -> {target}")
        return 0
    finally:
        chrome.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="http://127.0.0.1:18080")
    parser.add_argument("--only", nargs="*", default=[], help="shot names, e.g. panel-status")
    parser.add_argument("--out", type=Path, default=OUT, help="output folder")
    args = parser.parse_args(argv)
    return run(args.base, set(args.only), args.out)


if __name__ == "__main__":
    sys.exit(main())
