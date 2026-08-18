"""Unit tests for HuggingFace discovery, the downloader and the pre-download fit check.

Everything runs against a local ``http.server`` that speaks just enough of the
HF API (``/api/models``, ``/api/models/{repo}``, ``/{repo}/resolve/main/{file}``)
including byte ranges, so the suite passes with no network at all. Nothing here
ever touches huggingface.co.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import http.server
import json
import re
import socketserver
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core.downloader import (
    Downloader,
    DownloadProgress,
    fit_verdict,
)
from studioforge.core.hf_search import (
    GgufFileInfo,
    GgufRepoInfo,
    HfSearch,
    LogicalDownload,
    parse_quant,
    safe_filename,
)
from studioforge.core.planner import Planner
from studioforge.db import Database
from studioforge.errors import BadRequestError, UpstreamError
from studioforge.types import GB, GpuInfo

# ---------------------------------------------------------------------------
# Fake GPU probe (self-contained, like tests/unit/test_planner.py)
# ---------------------------------------------------------------------------


class StubProbe:
    """Minimal GpuProbe: a fixed GPU list."""

    backend = "fake"

    def __init__(self, gpus: list[GpuInfo]) -> None:
        self._gpus = {g.index: g.model_copy(deep=True) for g in gpus}

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [self._gpus[i].model_copy(deep=True) for i in sorted(self._gpus)]

    def get_gpu(self, index: int) -> GpuInfo | None:
        gpu = self._gpus.get(index)
        return gpu.model_copy(deep=True) if gpu else None

    def driver_version(self) -> str | None:
        return "610.88"

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return (13, 3)

    def shutdown(self) -> None:
        return None


def fake_gpu(index: int, total_gib: float, free_gib: float) -> GpuInfo:
    total = int(total_gib * GB)
    free = int(free_gib * GB)
    return GpuInfo(
        index=index,
        name=f"FakeGPU{index}",
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free,
        compute_capability=(12, 0),
    )


# ---------------------------------------------------------------------------
# Local stand-in for the HuggingFace endpoints
# ---------------------------------------------------------------------------


@dataclass
class ServerState:
    """Mutable knobs and observations for the fake HF server."""

    files: dict[str, bytes] = field(default_factory=dict)
    json_routes: dict[str, Any] = field(default_factory=dict)
    #: path -> successive page payloads, served in order. Every page but the
    #: last advertises ``Link: <...?cursor=N>; rel="next"``, which is what the
    #: real ``/api/models`` does and what the date-window walk follows.
    page_routes: dict[str, list[Any]] = field(default_factory=dict)
    #: path -> queued status codes, popped one per request (for 429 retry tests).
    status_script: dict[str, list[int]] = field(default_factory=dict)
    #: When True, a Range request is answered with a full 200 body.
    ignore_range: bool = False
    slice_bytes: int = 16 * 1024
    slice_delay_s: float = 0.0
    requests: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    #: (path, parsed query) per request, so a test can assert which HF ``sort``
    #: value a user-facing sort name actually turned into.
    queries: list[tuple[str, dict[str, list[str]]]] = field(default_factory=list)
    responses: list[tuple[str, int]] = field(default_factory=list)
    endpoint: str = ""
    inflight: int = 0
    max_inflight: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def range_headers(self, path: str) -> list[str]:
        with self.lock:
            return [h["Range"] for p, h in self.requests if p == path and "Range" in h]

    def statuses_for(self, path: str) -> list[int]:
        with self.lock:
            return [code for p, code in self.responses if p == path]

    def request_count(self, path: str) -> int:
        with self.lock:
            return sum(1 for p, _ in self.requests if p == path)

    def queries_for(self, path: str) -> list[dict[str, list[str]]]:
        with self.lock:
            return [q for p, q in self.queries if p == path]


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        return None

    def _record(self, path: str, code: int) -> None:
        with self.state.lock:
            self.state.responses.append((path, code))

    def _send(self, code: int, body: bytes, content_type: str, extra: dict[str, str]) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in extra.items():
            self.send_header(key, value)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        state = self.state
        split = urllib.parse.urlsplit(self.path)
        path = urllib.parse.unquote(split.path)
        query = urllib.parse.parse_qs(split.query)
        with state.lock:
            state.requests.append((path, dict(self.headers)))
            state.queries.append((path, query))

        script = state.status_script.get(path)
        if script:
            code = script.pop(0)
            if code >= 400:
                self._send(code, b'{"error":"scripted"}', "application/json", {"Retry-After": "0"})
                self._record(path, code)
                return

        if path in state.page_routes:
            self._serve_page(path, state, query)
            return

        if path in state.json_routes:
            body = json.dumps(state.json_routes[path]).encode()
            self._send(200, body, "application/json", {})
            self._record(path, 200)
            return

        if path not in state.files:
            self._send(404, b'{"error":"not found"}', "application/json", {})
            self._record(path, 404)
            return

        self._serve_file(path, state)

    def _serve_page(self, path: str, state: ServerState, query: dict[str, list[str]]) -> None:
        """Serve page ``?cursor=N`` of ``page_routes[path]``, HF-style.

        The ``rel="next"`` URL is absolute and rooted at this server's own
        endpoint, matching HF and satisfying the same-origin check that stops
        the client following a cursor to somebody else's host with the token
        attached.
        """
        pages = state.page_routes[path]
        index = int(query.get("cursor", ["0"])[0])
        body = json.dumps(pages[index] if index < len(pages) else []).encode()
        extra: dict[str, str] = {}
        if index + 1 < len(pages):
            extra["Link"] = f'<{state.endpoint}{path}?cursor={index + 1}>; rel="next"'
        self._send(200, body, "application/json", extra)
        self._record(path, 200)

    def _serve_file(self, path: str, state: ServerState) -> None:
        with state.lock:
            state.inflight += 1
            state.max_inflight = max(state.max_inflight, state.inflight)
        try:
            data = state.files[path]
            start = 0
            code = 200
            rng = self.headers.get("Range")
            if rng and not state.ignore_range:
                match = re.match(r"bytes=(\d+)-", rng)
                if match:
                    start = int(match.group(1))
                    if start >= len(data):
                        self._send(
                            416,
                            b"",
                            "application/octet-stream",
                            {"Content-Range": f"bytes */{len(data)}"},
                        )
                        self._record(path, 416)
                        return
                    code = 206
            body = data[start:]
            extra = {"Accept-Ranges": "bytes"}
            if code == 206:
                extra["Content-Range"] = f"bytes {start}-{len(data) - 1}/{len(data)}"
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            for key, value in extra.items():
                self.send_header(key, value)
            self.end_headers()
            self._record(path, code)
            for offset in range(0, len(body), state.slice_bytes):
                self.wfile.write(body[offset : offset + state.slice_bytes])
                self.wfile.flush()
                if state.slice_delay_s:
                    time.sleep(state.slice_delay_s)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with state.lock:
                state.inflight -= 1


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    state: ServerState


@pytest.fixture
def hf_server() -> Any:
    state = ServerState()
    server = _Server(("127.0.0.1", 0), _Handler)
    server.state = state
    state.endpoint = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"


def resolve_path(repo_id: str, filename: str) -> str:
    return f"/{repo_id}/resolve/main/{filename}"


def blob(size: int, seed: int = 0) -> bytes:
    """Deterministic pseudo-random payload (not compressible, not all zeros)."""
    out = bytearray()
    value = seed or 1
    while len(out) < size:
        value = (value * 1103515245 + 12345) & 0xFFFFFFFF
        out.extend(value.to_bytes(4, "little"))
    return bytes(out[:size])


def finfo(
    filename: str,
    size: int,
    *,
    sha256: str | None = None,
    is_mmproj: bool | None = None,
) -> GgufFileInfo:
    from studioforge.core.hf_search import shard_parts

    index, total = shard_parts(filename)
    return GgufFileInfo(
        filename=filename,
        size_bytes=size,
        quant=parse_quant(filename),
        is_mmproj="mmproj" in filename.lower() if is_mmproj is None else is_mmproj,
        shard_index=index,
        shard_total=total,
        sha256=sha256,
        lfs_oid=sha256,
    )


def one(
    filename: str,
    data: bytes,
    *,
    repo_id: str = REPO,
    with_sha: bool = True,
    sha_override: str | None = None,
) -> LogicalDownload:
    digest = sha_override or (hashlib.sha256(data).hexdigest() if with_sha else None)
    info = finfo(filename, len(data), sha256=digest)
    return LogicalDownload(
        repo_id=repo_id,
        quant=info.quant,
        files=[info],
        mmproj=None,
        total_bytes=info.size_bytes,
    )


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        hf={"max_concurrent_downloads": 4, "chunk_bytes": 16 * 1024},
    )


@pytest.fixture
def db(tmp_path: Path) -> Any:
    database = Database(tmp_path / "registry.sqlite3")
    database.migrate()
    try:
        yield database
    finally:
        database.close()


def make_downloader(
    config: Config,
    db: Database,
    endpoint: str,
    *,
    on_progress: Any = None,
) -> Downloader:
    return Downloader(config, db, endpoint=endpoint, on_progress=on_progress)


async def wait_until(predicate: Any, timeout_s: float = 10.0) -> bool:
    """Poll a filesystem/side-effect predicate.

    Polling (rather than an ``asyncio.Event``) on purpose: the condition being
    waited on is a real file appearing on disk inside production code that has
    no test hook, and adding one would be test scaffolding leaking into the
    downloader.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:  # noqa: ASYNC110
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


async def wait_group(
    downloader: Downloader, group_id: str, status: str = "completed", timeout_s: float = 30.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if downloader.group_status(group_id) == status:
            return
        await asyncio.sleep(0.02)
    rows = [p.to_dict() for p in downloader.group(group_id)]
    raise AssertionError(
        f"group {group_id} is {downloader.group_status(group_id)!r}, expected {status!r}: {rows}"
    )


# ===========================================================================
# Quant parsing
# ===========================================================================


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # The real spread from the reference library.
        ("DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen2.5-0.5B-Instruct-Q8_0.gguf", "Q8_0"),
        ("Precog-123B-v1.i1-IQ3_XXS.gguf", "IQ3_XXS"),
        ("TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf", "IQ3_M"),
        ("TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00002-of-00002.gguf", "IQ3_M"),
        ("gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "Q4_K_XL"),
        ("gemma-4-Ortenzya-31B-it-uncensored-heretic-NVFP4.gguf", "NVFP4"),
        ("G4-MeroMero-31B-uncensored-heretic-mmproj-BF16.gguf", "BF16"),
        ("Qwen2.5-VL-7B-Abliterated-Caption-it.mmproj-f16.gguf", "F16"),
        ("mmproj-F32.gguf", "F32"),
        ("mmproj-Qwen.Qwen3-VL-Embedding-2B.f16.gguf", "F16"),
        ("mmproj-Gemma4-31B-QAT-Uncensored-Balanced-BF16.gguf", "BF16"),
        # Lowercase repos and odd punctuation still parse.
        ("qwen3-embedding-8b-q4_k_m.gguf", "Q4_K_M"),
        ("24_10_Pygmalion or Mistral_cydonia-22b-v1-q6_k.gguf", "Q6_K"),
        ("gemma-4-26B-A4B-it-heretic.q8_0.gguf", "Q8_0"),
        ("Qwen3.5-Queen-27B.i1-Q5_K_M.gguf", "Q5_K_M"),
        ("gemma-2-2b-it-abliterated-Q5_K_S.gguf", "Q5_K_S"),
        # Trailing junk must not extend the label...
        ("24_10_Mistrial_Celeste-12B-V1.6.Q8_0NSFW.gguf", "Q8_0"),
        # ...and a model name that merely looks quant-ish must not invent one.
        ("Qwen2.5-7B-Instruct.gguf", "unknown"),
        ("Qwen3-Next-80B-A3B.gguf", "unknown"),
        ("model.gguf", "unknown"),
    ],
)
def test_parse_quant_covers_real_filenames(filename: str, expected: str) -> None:
    assert parse_quant(filename) == expected


# ===========================================================================
# logical_models() grouping
# ===========================================================================


def repo_with(files: list[GgufFileInfo], repo_id: str = REPO) -> GgufRepoInfo:
    owner, _, name = repo_id.partition("/")
    return GgufRepoInfo(
        repo_id=repo_id,
        publisher=owner,
        name=name,
        downloads=1,
        likes=0,
        gated=False,
        private=False,
        last_modified=None,
        files=files,
    )


def test_logical_models_collapses_shards_and_sums_bytes() -> None:
    repo = repo_with(
        [
            finfo("Behemoth-IQ3_M-00002-of-00002.gguf", 30),
            finfo("Behemoth-IQ3_M-00001-of-00002.gguf", 70),
            finfo("Behemoth-Q4_K_M.gguf", 200),
        ]
    )
    models = {m.quant: m for m in repo.logical_models()}
    assert set(models) == {"IQ3_M", "Q4_K_M"}

    sharded = models["IQ3_M"]
    assert [f.filename for f in sharded.files] == [
        "Behemoth-IQ3_M-00001-of-00002.gguf",
        "Behemoth-IQ3_M-00002-of-00002.gguf",
    ]
    assert sharded.total_bytes == 100
    assert sharded.is_sharded
    assert models["Q4_K_M"].total_bytes == 200
    assert not models["Q4_K_M"].is_sharded


def test_single_mmproj_is_attached_to_every_quant_and_never_standalone() -> None:
    projector = finfo("mmproj-F32.gguf", 500)
    repo = repo_with(
        [
            finfo("gemma-4-31B-Q4_0.gguf", 1000),
            finfo("gemma-4-31B-Q8_0.gguf", 2000),
            projector,
        ]
    )
    models = repo.logical_models()
    assert {m.quant for m in models} == {"Q4_0", "Q8_0"}
    assert all(m.mmproj == projector for m in models)
    # Bytes include the projector -- both files must land for the model to load.
    assert {m.total_bytes for m in models} == {1500, 2500}
    # The projector is never selectable on its own.
    assert "F32" not in {m.quant for m in models}
    assert repo.quant_variants == ["Q4_0", "Q8_0"]
    assert repo.mmproj_files == [projector]


def test_multiple_mmproj_prefers_exact_quant_then_f16() -> None:
    f16 = finfo("model-mmproj-f16.gguf", 10)
    f32 = finfo("model-mmproj-F32.gguf", 20)
    q8 = finfo("model-mmproj-Q8_0.gguf", 5)
    repo = repo_with([finfo("model-Q4_K_M.gguf", 100), finfo("model-Q8_0.gguf", 200), f16, f32, q8])
    models = {m.quant: m for m in repo.logical_models()}
    # Q8_0 has a matching projector -> exact match wins.
    assert models["Q8_0"].mmproj == q8
    # Q4_K_M has none -> f16 beats F32.
    assert models["Q4_K_M"].mmproj == f16


def test_multiple_mmproj_falls_back_to_bf16_when_no_f16() -> None:
    bf16 = finfo("model-mmproj-BF16.gguf", 10)
    f32 = finfo("model-mmproj-F32.gguf", 20)
    repo = repo_with([finfo("model-Q4_K_M.gguf", 100), bf16, f32])
    assert repo.logical_models()[0].mmproj == bf16


def test_incomplete_shard_listing_is_never_offered() -> None:
    """A listing naming 3 shards but carrying 2 is not a downloadable model.

    Offering it would let the download "complete" and report success while
    producing an unloadable partial set -- the files must arrive together or
    the model does not exist.
    """
    repo = repo_with(
        [
            finfo("Big-Q4_K_M-00001-of-00003.gguf", 100),
            finfo("Big-Q4_K_M-00003-of-00003.gguf", 100),
            finfo("Small-Q8_0.gguf", 50),
        ]
    )
    models = {m.quant: m for m in repo.logical_models()}
    assert "Q4_K_M" not in models, "an incomplete shard set was offered for download"
    assert "Q8_0" in models, "the complete entry must survive the filtering"


def test_complete_shard_listing_is_still_offered() -> None:
    repo = repo_with(
        [
            finfo("Big-Q4_K_M-00002-of-00002.gguf", 100),
            finfo("Big-Q4_K_M-00001-of-00002.gguf", 100),
        ]
    )
    models = repo.logical_models()
    assert len(models) == 1
    assert models[0].total_bytes == 200


def test_retry_after_junk_value_falls_back_to_backoff() -> None:
    """'Retry-After: soon' must degrade to the backoff, not raise out of the
    search route as a 500 (parsedate_to_datetime raises on Python >= 3.10)."""
    from studioforge.core.hf_search import _retry_after_seconds

    assert _retry_after_seconds("soon", 2.5) == 2.5
    assert _retry_after_seconds("13", 2.5) == 13.0
    assert _retry_after_seconds(None, 2.5) == 2.5


def test_mmproj_only_repo_yields_no_logical_models() -> None:
    repo = repo_with([finfo("mmproj-F32.gguf", 500)])
    assert repo.logical_models() == []


def test_duplicate_quant_gets_a_distinct_group_id() -> None:
    repo = repo_with([finfo("alpha-Q4_K_M.gguf", 10), finfo("beta-Q4_K_M.gguf", 20)])
    models = repo.logical_models()
    assert len(models) == 2
    assert len({m.group_id for m in models}) == 2


def test_group_id_is_stable_and_slugified() -> None:
    item = one("model-Q4_K_M.gguf", b"x")
    assert item.group_id == "bartowski-qwen2-5-7b-instruct-gguf-q4-k-m"
    assert item.group_id == one("model-Q4_K_M.gguf", b"y").group_id


def test_size_known_flags_the_zero_placeholder() -> None:
    unsized = finfo("model-Q4_K_M.gguf", 0)
    assert not unsized.size_known
    item = LogicalDownload(
        repo_id=REPO, quant="Q4_K_M", files=[unsized], mmproj=None, total_bytes=0
    )
    assert not item.size_known
    assert one("model-Q4_K_M.gguf", b"abc").size_known


def test_dest_relpath_is_lm_studio_layout() -> None:
    item = one("Qwen2.5-7B-Instruct-Q4_K_M.gguf", b"x")
    assert item.dest_relpath == (
        "bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )


# ===========================================================================
# Path safety (security)
# ===========================================================================


@pytest.mark.parametrize(
    "hostile",
    [
        "../../evil.gguf",
        "..",
        "../evil.gguf",
        "/etc/passwd",
        "C:\\Windows\\x.gguf",
        "C:x.gguf",
        "sub/dir/model.gguf",
        "sub\\dir\\model.gguf",
        "",
        "   ",
        "nul.gguf",
        "model\x00.gguf",
    ],
)
def test_safe_filename_rejects_hostile_names(hostile: str) -> None:
    with pytest.raises(BadRequestError):
        safe_filename(hostile)


def test_safe_filename_accepts_real_names() -> None:
    for name in (
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "TheDrummer_Behemoth-X-123B-v2.1-IQ3_M-00001-of-00002.gguf",
        "mmproj-F32.gguf",
    ):
        assert safe_filename(name) == name


def test_dest_for_rejects_traversal_in_filename_and_repo_id(config: Config, db: Database) -> None:
    downloader = make_downloader(config, db, "http://127.0.0.1:1")
    with pytest.raises(BadRequestError):
        downloader.dest_for(REPO, "../../evil.gguf")
    with pytest.raises(BadRequestError):
        downloader.dest_for(REPO, "/etc/passwd")
    with pytest.raises(BadRequestError):
        downloader.dest_for(REPO, "C:\\Windows\\x.gguf")
    with pytest.raises(BadRequestError):
        downloader.dest_for(REPO, "sub/dir/x.gguf")
    # repo_id is API-supplied too, so it is validated on the same footing.
    with pytest.raises(BadRequestError):
        downloader.dest_for("../../../etc/passwd", "model.gguf")


async def test_enqueue_refuses_hostile_filename_and_writes_nothing(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    models_dir = Path(config.models.dir or "")
    hostile = LogicalDownload(
        repo_id=REPO,
        quant="Q4_K_M",
        files=[finfo("../../evil.gguf", 10)],
        mmproj=None,
        total_bytes=10,
    )
    downloader = make_downloader(config, db, hf_server.endpoint)
    with pytest.raises(BadRequestError):
        await downloader.enqueue(hostile)
    assert db.list_downloads() == []
    escaped = models_dir.parent.parent / "evil.gguf"
    assert not escaped.exists()
    await downloader.stop()


def test_dest_layout_is_publisher_repo_filename(config: Config, db: Database) -> None:
    downloader = make_downloader(config, db, "http://127.0.0.1:1")
    dest = downloader.dest_for(REPO, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    expected = (
        Path(config.models.dir or "")
        / "bartowski"
        / "Qwen2.5-7B-Instruct-GGUF"
        / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )
    assert dest == expected


# ===========================================================================
# Downloads
# ===========================================================================


async def test_full_download_renames_part_and_records_transitions(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(300 * 1024, seed=7)
    name = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    item = one(name, data)

    seen: list[str] = []
    downloader = make_downloader(
        config, db, hf_server.endpoint, on_progress=lambda p: seen.append(p.status)
    )
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    dest = downloader.dest_for(REPO, name)
    assert dest.read_bytes() == data
    assert not dest.with_name(dest.name + ".part").exists()

    # queued -> running -> completed, in that order, with nothing after completed.
    trimmed = [s for i, s in enumerate(seen) if i == 0 or seen[i - 1] != s]
    assert trimmed[0] == "queued"
    assert "running" in trimmed
    assert trimmed[-1] == "completed"

    rows = db.list_downloads(group_id=group_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["downloaded_bytes"] == len(data)
    assert rows[0]["total_bytes"] == len(data)
    assert Path(rows[0]["dest_path"]) == dest
    await downloader.stop()


async def test_group_completes_only_when_base_and_mmproj_are_both_done(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    base = blob(80 * 1024, seed=3)
    proj = blob(20 * 1024, seed=4)
    hf_server.files[resolve_path(REPO, "model-Q4_K_M.gguf")] = base
    hf_server.files[resolve_path(REPO, "mmproj-F32.gguf")] = proj
    item = LogicalDownload(
        repo_id=REPO,
        quant="Q4_K_M",
        files=[finfo("model-Q4_K_M.gguf", len(base), sha256=hashlib.sha256(base).hexdigest())],
        mmproj=finfo("mmproj-F32.gguf", len(proj), sha256=hashlib.sha256(proj).hexdigest()),
        total_bytes=len(base) + len(proj),
    )
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    assert len(downloader.group(group_id)) == 2
    assert downloader.dest_for(REPO, "model-Q4_K_M.gguf").read_bytes() == base
    assert downloader.dest_for(REPO, "mmproj-F32.gguf").read_bytes() == proj
    await downloader.stop()


async def test_resume_sends_range_and_server_206_is_honoured(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(200 * 1024, seed=11)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prefix = 64 * 1024
    dest.with_name(dest.name + ".part").write_bytes(data[:prefix])

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    assert hf_server.range_headers(path) == [f"bytes={prefix}-"]
    assert hf_server.statuses_for(path) == [206]
    assert dest.read_bytes() == data
    await downloader.stop()


async def test_range_ignored_restarts_from_zero_without_corruption(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(200 * 1024, seed=13)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.ignore_range = True
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name(dest.name + ".part").write_bytes(data[: 64 * 1024])

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    assert hf_server.range_headers(path) == [f"bytes={64 * 1024}-"]
    assert hf_server.statuses_for(path) == [200]
    # The prefix must have been discarded, not appended to.
    assert dest.stat().st_size == len(data)
    assert dest.read_bytes() == data
    await downloader.stop()


async def test_stale_part_larger_than_object_restarts_clean(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """A 416 means the .part is stale (repo re-upload); start over rather than fail."""
    data = blob(50 * 1024, seed=17)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name(dest.name + ".part").write_bytes(blob(120 * 1024, seed=99))

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)
    assert 416 in hf_server.statuses_for(path)
    assert dest.read_bytes() == data
    await downloader.stop()


async def test_sha256_verification_success(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(100 * 1024, seed=19)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    item = one(name, data)
    assert item.files[0].sha256 == hashlib.sha256(data).hexdigest()

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)
    assert downloader.dest_for(REPO, name).read_bytes() == data
    await downloader.stop()


async def test_sha256_mismatch_fails_deletes_file_and_names_both_hashes(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(100 * 1024, seed=23)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    wrong = "0" * 64
    item = one(name, data, sha_override=wrong)

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id, status="failed")

    dest = downloader.dest_for(REPO, name)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()

    error = downloader.group(group_id)[0].error or ""
    assert wrong in error
    assert hashlib.sha256(data).hexdigest() in error
    assert db.list_downloads(group_id=group_id)[0]["status"] == "failed"
    await downloader.stop()


async def test_progress_is_throttled_monotonic_and_survives_a_raising_subscriber(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    size = 1024 * 1024
    chunk = int(config.hf.chunk_bytes)
    expected_chunks = size // chunk
    data = blob(size, seed=29)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    # Slow the transfer to ~1.3s so the 4 Hz throttle is actually exercised.
    hf_server.slice_bytes = chunk
    hf_server.slice_delay_s = 0.02
    item = one(name, data)

    events: list[DownloadProgress] = []
    calls = {"bad": 0}

    def bad(_progress: DownloadProgress) -> None:
        calls["bad"] += 1
        raise RuntimeError("subscriber exploded")

    downloader = make_downloader(config, db, hf_server.endpoint)
    downloader.subscribe(bad)
    downloader.subscribe(events.append)

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    # A raising subscriber is dropped after its first failure, never retried,
    # and never breaks the download.
    assert calls["bad"] == 1
    assert downloader.dest_for(REPO, name).read_bytes() == data

    # Far fewer emissions than chunks: 4 Hz, not per-chunk.
    assert expected_chunks >= 60
    assert len(events) < expected_chunks // 2

    progressed = [e.downloaded_bytes for e in events]
    assert progressed == sorted(progressed)
    final = events[-1]
    assert final.status == "completed"
    assert final.downloaded_bytes == size
    assert final.percent == pytest.approx(100.0)

    running = [e for e in events if e.status == "running" and e.downloaded_bytes > 0]
    assert running, "expected at least one mid-transfer progress event"
    assert any(e.speed_bps > 0 for e in running)
    assert any(e.eta_s is not None and e.eta_s >= 0 for e in running)
    await downloader.stop()


async def test_pause_keeps_part_then_resume_completes(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(600 * 1024, seed=31)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.03
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)

    dest = downloader.dest_for(REPO, name)
    part = dest.with_name(dest.name + ".part")
    assert await wait_until(lambda: part.exists() and part.stat().st_size > 0)

    await downloader.pause(group_id)
    assert downloader.group_status(group_id) == "paused"
    assert part.exists()
    assert db.list_downloads(group_id=group_id)[0]["status"] == "paused"

    hf_server.slice_delay_s = 0.0
    await downloader.resume(group_id)
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == data
    assert not part.exists()
    # Resume really resumed rather than restarting.
    assert any(h.startswith("bytes=") for h in hf_server.range_headers(path))
    await downloader.stop()


async def test_cancel_deletes_the_part_file(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(600 * 1024, seed=37)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.03
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    dest = downloader.dest_for(REPO, name)
    part = dest.with_name(dest.name + ".part")
    assert await wait_until(part.exists)

    await downloader.cancel(group_id)
    assert downloader.group_status(group_id) == "canceled"
    assert not part.exists()
    assert not dest.exists()
    assert db.list_downloads(group_id=group_id)[0]["status"] == "canceled"
    await downloader.stop()


async def test_cancel_can_keep_the_partial_file(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(600 * 1024, seed=41)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.03

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    part = downloader.dest_for(REPO, name)
    part = part.with_name(part.name + ".part")
    assert await wait_until(part.exists)

    await downloader.cancel(group_id, delete_partial=False)
    assert downloader.group_status(group_id) == "canceled"
    assert part.exists()
    await downloader.stop()


async def test_restart_survival_new_downloader_finishes_the_file(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(800 * 1024, seed=43)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.03
    item = one(name, data)

    first = make_downloader(config, db, hf_server.endpoint)
    group_id = await first.enqueue(item)
    dest = first.dest_for(REPO, name)
    part = dest.with_name(dest.name + ".part")
    assert await wait_until(lambda: part.exists() and part.stat().st_size > 0)
    # Simulate process death: tasks die, the .part and the DB row survive.
    await first.stop()
    assert part.exists()
    partial = part.stat().st_size
    assert 0 < partial < len(data)

    hf_server.slice_delay_s = 0.0
    second = make_downloader(config, db, hf_server.endpoint)
    await second.start()
    await wait_group(second, group_id)

    assert dest.read_bytes() == data
    assert not part.exists()
    assert db.list_downloads(group_id=group_id)[0]["status"] == "completed"
    await second.stop()


async def test_refuses_to_overwrite_an_existing_complete_file(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(100 * 1024, seed=47)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    # No published checksum, which is the common case for GGUF repos and the
    # only one where "right size, different bytes" can be asserted at all:
    # where HF *does* publish a sha256 the adopt path now hashes the file and
    # quarantines a mismatch (see the corrupt-file tests below).
    item = one(name, data, with_sha=False)

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = blob(100 * 1024, seed=48)
    dest.write_bytes(existing)

    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id)

    # Existing bytes untouched, and no request was ever made.
    assert dest.read_bytes() == existing
    assert hf_server.request_count(path) == 0
    assert db.list_downloads(group_id=group_id)[0]["status"] == "completed"
    await downloader.stop()


async def test_force_overwrites_an_existing_file(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(100 * 1024, seed=53)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"stale")

    group_id = await downloader.enqueue(one(name, data), force=True)
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == data
    await downloader.stop()


async def test_concurrency_cap_is_respected(
    tmp_path: Path, db: Database, hf_server: ServerState
) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        hf={"max_concurrent_downloads": 1, "chunk_bytes": 16 * 1024},
    )
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.02

    items: list[LogicalDownload] = []
    for index in range(3):
        data = blob(160 * 1024, seed=100 + index)
        repo = f"pub{index}/Repo{index}-GGUF"
        name = f"model{index}-Q4_K_M.gguf"
        hf_server.files[resolve_path(repo, name)] = data
        items.append(one(name, data, repo_id=repo))

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_ids = [await downloader.enqueue(item) for item in items]
    for group_id in group_ids:
        await wait_group(downloader, group_id)

    assert hf_server.max_inflight == 1
    assert len(downloader.all()) == 3
    await downloader.stop()


async def test_http_error_marks_the_download_failed(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    # The file is not registered on the server, so the GET 404s.
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one("missing-Q4_K_M.gguf", b"x" * 32))
    await wait_group(downloader, group_id, status="failed")
    error = downloader.group(group_id)[0].error or ""
    assert "404" in error
    await downloader.stop()


async def test_truncated_transfer_is_rejected(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """A total_bytes the server disagrees with must fail, not rename a short file."""
    data = blob(50 * 1024, seed=59)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    info = finfo(name, len(data) * 2)  # lie about the size, no checksum
    item = LogicalDownload(
        repo_id=REPO, quant="Q4_K_M", files=[info], mmproj=None, total_bytes=info.size_bytes
    )
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(item)
    await wait_group(downloader, group_id, status="failed")
    dest = downloader.dest_for(REPO, name)
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()
    error = downloader.group(group_id)[0].error or ""
    assert str(len(data)) in error
    assert str(len(data) * 2) in error
    await downloader.stop()


async def test_active_and_all_views(config: Config, db: Database, hf_server: ServerState) -> None:
    data = blob(64 * 1024, seed=61)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    assert downloader.active() == []
    assert [p.status for p in downloader.all()] == ["completed"]
    assert downloader.all()[0].percent == pytest.approx(100.0)
    await downloader.stop()


async def test_unsubscribe_stops_delivery(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    data = blob(32 * 1024, seed=67)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    seen: list[DownloadProgress] = []
    downloader = make_downloader(config, db, hf_server.endpoint)
    unsubscribe = downloader.subscribe(seen.append)
    unsubscribe()
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    assert seen == []
    await downloader.stop()


# ===========================================================================
# HfSearch against the local server
# ===========================================================================


def _sibling(name: str, size: int | None = None, sha: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"rfilename": name}
    if size is not None:
        entry["lfs"] = {"size": size, "oid": sha or ("a" * 64)}
        if sha:
            entry["lfs"]["sha256"] = sha
    return entry


async def test_search_parses_repos_and_skips_non_gguf(
    config: Config, hf_server: ServerState
) -> None:
    hf_server.json_routes["/api/models"] = [
        {
            "id": REPO,
            "downloads": 4242,
            "likes": 7,
            "gated": False,
            "private": False,
            "lastModified": "2026-01-01T00:00:00.000Z",
            "siblings": [
                _sibling("README.md"),
                _sibling("Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
                _sibling("Qwen2.5-7B-Instruct-Q8_0.gguf"),
                _sibling("imatrix.dat"),
            ],
        },
        {"id": "nobody/no-gguf", "siblings": [_sibling("README.md")]},
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("qwen", limit=5)
    finally:
        await search.aclose()

    assert [r.repo_id for r in repos] == [REPO]
    repo = repos[0]
    assert repo.publisher == "bartowski"
    assert repo.downloads == 4242
    assert repo.likes == 7
    assert repo.quant_variants == ["Q4_K_M", "Q8_0"]
    # The list endpoint carries no sizes, and that is reported rather than faked.
    assert not repo.sizes_known
    assert all(f.size_bytes == 0 for f in repo.files)


async def test_repo_info_uses_blob_sizes_and_lfs_sha256(
    config: Config, hf_server: ServerState
) -> None:
    sha = "b" * 64
    hf_server.json_routes[f"/api/models/{REPO}"] = {
        "id": REPO,
        "gated": "manual",
        "private": False,
        "siblings": [
            _sibling("model-Q4_K_M.gguf", 4096, sha),
            # No explicit sha256: the LFS oid is the sha256 per the LFS spec.
            {"rfilename": "mmproj-F32.gguf", "lfs": {"size": 512, "oid": "c" * 64}},
            # Neither lfs nor a usable size -> flagged unknown, never invented.
            {"rfilename": "model-Q8_0.gguf"},
        ],
    }
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repo = await search.repo_info(REPO)
    finally:
        await search.aclose()

    by_name = {f.filename: f for f in repo.files}
    assert by_name["model-Q4_K_M.gguf"].size_bytes == 4096
    assert by_name["model-Q4_K_M.gguf"].sha256 == sha
    assert by_name["mmproj-F32.gguf"].sha256 == "c" * 64
    assert by_name["mmproj-F32.gguf"].is_mmproj
    assert by_name["model-Q8_0.gguf"].size_bytes == 0
    assert not by_name["model-Q8_0.gguf"].size_known
    assert repo.gated == "manual"
    assert repo.needs_token
    assert not repo.sizes_known

    models = {m.quant: m for m in repo.logical_models()}
    assert models["Q4_K_M"].total_bytes == 4096 + 512
    assert models["Q4_K_M"].size_known
    assert not models["Q8_0"].size_known


async def test_gated_repo_401_names_the_token_config_key(
    config: Config, hf_server: ServerState
) -> None:
    hf_server.status_script[f"/api/models/{REPO}"] = [401]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(BadRequestError) as excinfo:
            await search.repo_info(REPO)
    finally:
        await search.aclose()
    message = excinfo.value.message
    assert "hf.token" in message
    assert "gated" in message.lower()
    assert excinfo.value.param == "hf.token"


async def test_403_also_points_at_the_token(config: Config, hf_server: ServerState) -> None:
    hf_server.status_script[f"/api/models/{REPO}"] = [403]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(BadRequestError):
            await search.repo_info(REPO)
    finally:
        await search.aclose()


async def test_rate_limit_is_retried_then_succeeds(config: Config, hf_server: ServerState) -> None:
    path = f"/api/models/{REPO}"
    hf_server.status_script[path] = [429, 429]
    hf_server.json_routes[path] = {
        "id": REPO,
        "siblings": [_sibling("model-Q4_K_M.gguf", 128, "d" * 64)],
    }
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repo = await search.repo_info(REPO)
    finally:
        await search.aclose()
    assert repo.repo_id == REPO
    assert hf_server.request_count(path) == 3


async def test_rate_limit_exhaustion_raises_a_clear_error(
    config: Config, hf_server: ServerState
) -> None:
    path = f"/api/models/{REPO}"
    hf_server.status_script[path] = [429] * 10
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(UpstreamError) as excinfo:
            await search.repo_info(REPO)
    finally:
        await search.aclose()
    assert "rate-limited" in excinfo.value.message
    assert "hf.token" in excinfo.value.message


async def test_missing_repo_404_is_a_bad_request(config: Config, hf_server: ServerState) -> None:
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(BadRequestError):
            await search.repo_info("nobody/nope")
    finally:
        await search.aclose()


async def test_token_goes_in_the_header_never_the_url(
    tmp_path: Path, hf_server: ServerState
) -> None:
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        hf={"token": "hf_secret_token_value"},
    )
    path = f"/api/models/{REPO}"
    hf_server.json_routes[path] = {"id": REPO, "siblings": [_sibling("m-Q4_K_M.gguf", 8, "e" * 64)]}
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        await search.repo_info(REPO)
    finally:
        await search.aclose()
    recorded = [(p, h) for p, h in hf_server.requests if p == path]
    assert recorded
    assert recorded[0][1]["Authorization"] == "Bearer hf_secret_token_value"
    assert "hf_secret_token_value" not in recorded[0][0]


async def test_injected_client_is_not_closed_by_aclose(
    config: Config, hf_server: ServerState
) -> None:
    hf_server.json_routes["/api/models"] = []
    async with httpx.AsyncClient() as client:
        search = HfSearch(config, client=client, endpoint=hf_server.endpoint)
        assert await search.search("anything") == []
        await search.aclose()
        assert not client.is_closed


# ===========================================================================
# Sorting, and the client-side date window
# ===========================================================================

MODELS = "/api/models"


def _iso(days_ago: float, *, now: float | None = None) -> str:
    """An HF-shaped UTC timestamp ``days_ago`` days before ``now``."""
    reference = time.time() if now is None else now
    stamp = dt.datetime.fromtimestamp(reference - days_ago * 86400.0, tz=dt.UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _entry(
    repo_id: str,
    *,
    downloads: int = 0,
    likes: int = 0,
    updated_days: float = 0.0,
    created_days: float | None = None,
    trending: int | None = None,
    files: tuple[str, ...] = ("model-Q4_K_M.gguf",),
) -> dict[str, Any]:
    """One ``/api/models`` list entry, shaped like the live payload."""
    entry: dict[str, Any] = {
        "id": repo_id,
        "downloads": downloads,
        "likes": likes,
        "lastModified": _iso(updated_days),
        "createdAt": _iso(updated_days if created_days is None else created_days),
        "siblings": [_sibling(name) for name in files],
    }
    if trending is not None:
        entry["trendingScore"] = trending
    return entry


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("downloads", "downloads"),
        ("likes", "likes"),
        ("updated", "lastModified"),
        ("created", "createdAt"),
        ("trending", "trendingScore"),
    ],
)
async def test_sort_names_map_to_the_hf_api_spelling(
    config: Config, hf_server: ServerState, name: str, expected: str
) -> None:
    """The user-facing name is never sent as-is: HF 400s on `sort=trending`."""
    hf_server.json_routes[MODELS] = []
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        await search.search("qwen", sort=name)
    finally:
        await search.aclose()
    query = hf_server.queries_for(MODELS)[0]
    assert query["sort"] == [expected]
    assert query["direction"] == ["-1"]


async def test_sort_keys_covers_exactly_the_documented_menu() -> None:
    from studioforge.core.hf_search import DATE_FIELDS, SORT_KEYS

    assert list(SORT_KEYS) == ["downloads", "likes", "updated", "created", "trending"]
    assert DATE_FIELDS == ("updated", "created")


async def test_unknown_sort_names_the_allowed_values(
    config: Config, hf_server: ServerState
) -> None:
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(BadRequestError) as excinfo:
            await search.search("qwen", sort="popular")
    finally:
        await search.aclose()
    message = excinfo.value.message
    assert excinfo.value.param == "sort"
    assert "popular" in message
    for allowed in ("downloads", "likes", "updated", "created", "trending"):
        assert allowed in message
    # Rejected before any request went out.
    assert hf_server.request_count(MODELS) == 0


async def test_unknown_date_field_names_the_allowed_values(
    config: Config, hf_server: ServerState
) -> None:
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        with pytest.raises(BadRequestError) as excinfo:
            await search.search("qwen", newer_than_days=7, date_field="modified")
    finally:
        await search.aclose()
    assert excinfo.value.param == "date_field"
    assert "updated" in excinfo.value.message
    assert "created" in excinfo.value.message


async def test_sort_names_tolerate_case_and_whitespace(
    config: Config, hf_server: ServerState
) -> None:
    """A hand-typed URL saying `sort=Downloads` is unambiguous, so it is accepted."""
    hf_server.json_routes[MODELS] = []
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        await search.search("qwen", sort=" Updated ")
    finally:
        await search.aclose()
    assert hf_server.queries_for(MODELS)[0]["sort"] == ["lastModified"]


async def test_search_parses_created_at_and_trending_score(
    config: Config, hf_server: ServerState
) -> None:
    hf_server.json_routes[MODELS] = [
        _entry(REPO, downloads=10, updated_days=3.0, created_days=400.0, trending=42),
        # No trendingScore at all: HF omits it unless it is the sort key, and
        # "absent" must not be flattened into a score of 0.
        _entry("other/thing-GGUF", downloads=5, updated_days=1.0),
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = {r.repo_id: r for r in await search.search("q")}
    finally:
        await search.aclose()

    assert repos[REPO].trending_score == 42
    assert repos["other/thing-GGUF"].trending_score is None
    assert repos[REPO].created_at is not None
    assert repos[REPO].updated_days_ago == pytest.approx(3.0, abs=0.01)
    assert repos[REPO].created_days_ago == pytest.approx(400.0, abs=0.01)


# -- the date window --------------------------------------------------------


async def test_window_walk_orders_by_date_not_by_the_requested_sort(
    config: Config, hf_server: ServerState
) -> None:
    """The walk must be date-ordered, or the early cutoff stop is unsound."""
    hf_server.page_routes[MODELS] = [[_entry("a/b-GGUF", downloads=5, updated_days=1.0)]]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        await search.search("q", sort="downloads", newer_than_days=7)
    finally:
        await search.aclose()
    query = hf_server.queries_for(MODELS)[0]
    assert query["sort"] == ["lastModified"]
    assert query["limit"] == ["100"]


async def test_window_stops_at_the_first_entry_past_the_cutoff(
    config: Config, hf_server: ServerState
) -> None:
    """One old entry ends the walk: the rest of the page and every later page."""
    hf_server.page_routes[MODELS] = [
        [
            _entry("in/one-GGUF", downloads=1, updated_days=0.5),
            _entry("in/two-GGUF", downloads=2, updated_days=6.0),
            _entry("out/old-GGUF", downloads=999, updated_days=30.0),
            # Newer than the cutoff but *after* an old entry, so unreachable.
            # Real HF could not return this; it proves we stop rather than filter.
            _entry("in/unreachable-GGUF", downloads=500, updated_days=0.1),
        ],
        [_entry("never/fetched-GGUF", downloads=1000, updated_days=0.1)],
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("q", sort="downloads", newer_than_days=7)
        truncated = search.last_search_truncated
    finally:
        await search.aclose()

    assert [r.repo_id for r in repos] == ["in/two-GGUF", "in/one-GGUF"]
    assert truncated is False
    # Page 2 was never requested.
    assert hf_server.request_count(MODELS) == 1


async def test_window_follows_the_cursor_then_re_sorts_and_cuts_to_limit(
    config: Config, hf_server: ServerState
) -> None:
    """Across pages, HF's date order is replaced by the sort the user asked for."""
    hf_server.page_routes[MODELS] = [
        [
            _entry("p1/newest-GGUF", downloads=10, updated_days=0.1),
            _entry("p1/second-GGUF", downloads=900, updated_days=0.2),
        ],
        [
            _entry("p2/third-GGUF", downloads=50, updated_days=1.0),
            _entry("p2/fourth-GGUF", downloads=5000, updated_days=2.0),
            _entry("p2/stale-GGUF", downloads=100_000, updated_days=99.0),
        ],
        [_entry("p3/unreached-GGUF", downloads=1, updated_days=0.1)],
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("q", limit=3, sort="downloads", newer_than_days=7)
        truncated = search.last_search_truncated
    finally:
        await search.aclose()

    # Four survivors, re-sorted by downloads, cut to limit=3.
    assert [r.repo_id for r in repos] == ["p2/fourth-GGUF", "p1/second-GGUF", "p2/third-GGUF"]
    assert truncated is False
    assert hf_server.request_count(MODELS) == 2
    # The cursor really was used, rather than the first page being refetched.
    assert hf_server.queries_for(MODELS)[1]["cursor"] == ["1"]


async def test_window_sorts_by_likes_when_asked(config: Config, hf_server: ServerState) -> None:
    hf_server.page_routes[MODELS] = [
        [
            _entry("a/one-GGUF", downloads=9_000, likes=1, updated_days=0.1),
            _entry("a/two-GGUF", downloads=1, likes=500, updated_days=0.2),
        ]
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("q", sort="likes", newer_than_days=7)
    finally:
        await search.aclose()
    assert [r.repo_id for r in repos] == ["a/two-GGUF", "a/one-GGUF"]


async def test_window_on_created_uses_created_at_for_both_filter_and_sort(
    config: Config, hf_server: ServerState
) -> None:
    """`date_field="created"` must not silently fall back to lastModified.

    The "old repo, freshly re-quantised" entry is the whole point: it is inside
    an *updated* window and outside a *created* one.
    """
    hf_server.page_routes[MODELS] = [
        [
            _entry("new/repo-GGUF", downloads=1, updated_days=0.1, created_days=2.0),
            _entry("old/touched-GGUF", downloads=99, updated_days=0.2, created_days=800.0),
        ]
    ]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("q", sort="created", newer_than_days=7, date_field="created")
    finally:
        await search.aclose()

    assert hf_server.queries_for(MODELS)[0]["sort"] == ["createdAt"]
    assert [r.repo_id for r in repos] == ["new/repo-GGUF"]


async def test_window_treats_a_missing_date_as_past_the_cutoff(
    config: Config, hf_server: ServerState
) -> None:
    """HF sorts null dates last, so a null is the end of the window, not a skip."""
    entry = _entry("no/date-GGUF", downloads=1)
    entry["lastModified"] = None
    hf_server.page_routes[MODELS] = [[_entry("in/one-GGUF", downloads=1, updated_days=0.1), entry]]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        repos = await search.search("q", newer_than_days=7)
    finally:
        await search.aclose()
    assert [r.repo_id for r in repos] == ["in/one-GGUF"]


async def test_window_reports_truncation_when_the_page_cap_is_hit(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A window wider than 3 pages returns a partial answer, and says so."""
    from studioforge.core import hf_search as hs

    pages_served = 0

    async def fake_get_page(
        self: Any, path: str, params: Any = None, *, url: str | None = None
    ) -> tuple[Any, str | None]:
        nonlocal pages_served
        pages_served += 1
        page = [_entry(f"page{pages_served}/repo-GGUF", downloads=pages_served, updated_days=0.1)]
        # Always another page, and never an out-of-window entry: the cap is the
        # only thing that can stop this walk.
        return page, f"https://huggingface.co/api/models?cursor={pages_served}"

    monkeypatch.setattr(hs.HfSearch, "_get_page", fake_get_page)
    search = hs.HfSearch(config)
    try:
        repos = await search.search("q", newer_than_days=3650)
        truncated = search.last_search_truncated
    finally:
        await search.aclose()

    assert pages_served == 3, "the page cap must bound the walk"
    assert truncated is True
    assert len(repos) == 3


async def test_unwindowed_search_never_reports_truncation(
    config: Config, hf_server: ServerState
) -> None:
    hf_server.json_routes[MODELS] = [_entry("a/b-GGUF", downloads=1)]
    search = HfSearch(config, endpoint=hf_server.endpoint)
    try:
        await search.search("q")
        assert search.last_search_truncated is False
    finally:
        await search.aclose()


async def test_pagination_refuses_a_cross_origin_cursor(
    config: Config, hf_server: ServerState
) -> None:
    """A `rel="next"` pointing off-endpoint must not be followed with our token.

    Following it would send the `Authorization: Bearer <hf token>` header to a
    host named by the response, which is a token exfiltration primitive handed
    to whoever controls (or MITMs) the endpoint.
    """
    from studioforge.core.hf_search import _next_page_url

    same = _next_page_url(
        '<https://hf.example/api/models?cursor=x>; rel="next"', endpoint="https://hf.example"
    )
    assert same == "https://hf.example/api/models?cursor=x"
    assert (
        _next_page_url('<https://evil.example/steal>; rel="next"', endpoint="https://hf.example")
        is None
    )
    assert (
        _next_page_url('<https://hf.example/api/models>; rel="prev"', endpoint="https://hf.example")
        is None
    )
    assert _next_page_url(None, endpoint="https://hf.example") is None


# -- age_days ---------------------------------------------------------------


def test_age_days_is_fractional_and_parses_the_z_suffix() -> None:
    from studioforge.core.hf_search import age_days

    now = dt.datetime(2026, 8, 18, 12, 0, tzinfo=dt.UTC).timestamp()
    assert age_days("2026-08-11T12:00:00.000Z", now=now) == pytest.approx(7.0)
    assert age_days("2026-08-18T00:00:00.000Z", now=now) == pytest.approx(0.5)
    # Explicit offsets, not just Z.
    assert age_days("2026-08-18T13:00:00+01:00", now=now) == pytest.approx(0.0)


def test_age_days_returns_none_rather_than_guessing() -> None:
    from studioforge.core.hf_search import age_days

    assert age_days(None) is None
    assert age_days("") is None
    assert age_days("last tuesday") is None
    assert age_days(12345) is None  # type: ignore[arg-type]


def test_age_days_clamps_a_future_timestamp_to_zero() -> None:
    """Clock skew against HF must not render as a negative age."""
    from studioforge.core.hf_search import age_days

    now = dt.datetime(2026, 8, 18, 12, 0, tzinfo=dt.UTC).timestamp()
    assert age_days("2026-08-19T12:00:00.000Z", now=now) == 0.0


# ===========================================================================
# GET /api/hf/search -- the query parameters, through the real app
# ===========================================================================


@pytest.fixture
def search_app(tmp_path: Path, hf_server: ServerState, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real app, with HfSearch pointed at the local stand-in via HF_ENDPOINT.

    Going through ``HF_ENDPOINT`` rather than patching ``HfSearch`` keeps the
    route's own construction of the client under test -- that is where a
    forgotten parameter would actually go missing.
    """
    from studioforge.api.app import build_state, create_app

    monkeypatch.setenv("HF_ENDPOINT", hf_server.endpoint)
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    state = build_state(config)
    try:
        yield create_app(config, state=state, start_background=False)
    finally:
        state.db.close()


def test_hf_search_route_defaults_to_downloads_and_no_window(
    search_app: Any, hf_server: ServerState
) -> None:
    from fastapi.testclient import TestClient

    hf_server.json_routes[MODELS] = [_entry(REPO, downloads=42, likes=7, updated_days=3.0)]
    with TestClient(search_app) as http:
        body = http.get("/api/hf/search", params={"q": "qwen"}).json()

    assert body["sort"] == "downloads"
    assert body["newer_than_days"] is None
    assert body["date_field"] == "updated"
    assert body["truncated"] is False
    assert hf_server.queries_for(MODELS)[0]["sort"] == ["downloads"]


def test_hf_search_route_advertises_its_own_options(
    search_app: Any, hf_server: ServerState
) -> None:
    """The envelope carries the menu, so a client (or an LLM) can discover it."""
    from fastapi.testclient import TestClient

    hf_server.json_routes[MODELS] = []
    with TestClient(search_app) as http:
        body = http.get("/api/hf/search", params={"q": "qwen"}).json()

    assert body["sort_options"] == ["downloads", "likes", "updated", "created", "trending"]
    assert body["date_field_options"] == ["updated", "created"]


def test_hf_search_route_passes_sort_and_window_through(
    search_app: Any, hf_server: ServerState
) -> None:
    from fastapi.testclient import TestClient

    hf_server.page_routes[MODELS] = [
        [
            _entry("new/one-GGUF", downloads=5, updated_days=0.5, created_days=1.0),
            _entry("old/two-GGUF", downloads=900, updated_days=0.6, created_days=900.0),
        ]
    ]
    with TestClient(search_app) as http:
        body = http.get(
            "/api/hf/search",
            params={"q": "qwen", "sort": "created", "newer_than_days": 7, "date_field": "created"},
        ).json()

    assert body["sort"] == "created"
    assert body["newer_than_days"] == 7
    assert body["date_field"] == "created"
    # The window really was applied against createdAt, not lastModified.
    assert hf_server.queries_for(MODELS)[0]["sort"] == ["createdAt"]
    assert [r["repo_id"] for r in body["repos"]] == ["new/one-GGUF"]


def test_hf_search_route_payload_carries_dates_ages_and_trending(
    search_app: Any, hf_server: ServerState
) -> None:
    from fastapi.testclient import TestClient

    hf_server.json_routes[MODELS] = [
        _entry(REPO, downloads=42, likes=7, updated_days=3.0, created_days=400.0, trending=11)
    ]
    with TestClient(search_app) as http:
        repo = http.get("/api/hf/search", params={"q": "qwen"}).json()["repos"][0]

    assert repo["downloads"] == 42
    assert repo["likes"] == 7
    assert repo["trending_score"] == 11
    assert repo["last_modified"] is not None
    assert repo["created_at"] is not None
    assert repo["updated_days_ago"] == pytest.approx(3.0, abs=0.01)
    assert repo["created_days_ago"] == pytest.approx(400.0, abs=0.01)


def test_hf_search_route_rejects_an_unknown_sort(search_app: Any) -> None:
    from fastapi.testclient import TestClient

    with TestClient(search_app) as http:
        response = http.get("/api/hf/search", params={"q": "qwen", "sort": "popular"})

    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "popular" in message
    assert "trending" in message, "the error must name the allowed values"


@pytest.mark.parametrize("days", [0, 3651])
def test_hf_search_route_bounds_the_window(search_app: Any, days: int) -> None:
    """`ge=1`/`le=3650` is enforced before the search runs.

    The status is 400, not FastAPI's default 422: per D5 every error on this
    app renders as the OpenAI-shaped envelope, and the app-wide
    ``RequestValidationError`` handler rewrites validation failures to match.
    """
    from fastapi.testclient import TestClient

    with TestClient(search_app) as http:
        response = http.get("/api/hf/search", params={"q": "qwen", "newer_than_days": days})

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert "newer_than_days" in error["message"]


# ===========================================================================
# Task C: fit_verdict
# ===========================================================================


def make_planner(config: Config, gpus: list[GpuInfo]) -> Planner:
    return Planner(config, StubProbe(gpus))  # type: ignore[arg-type]


def sized_item(quant: str, size: int, repo_id: str = REPO) -> LogicalDownload:
    info = finfo(f"model-{quant}.gguf", size)
    return LogicalDownload(
        repo_id=repo_id,
        quant=quant,
        files=[info],
        mmproj=None,
        total_bytes=size,
    )


def test_fit_verdict_small_model_fits_one_gpu(config: Config) -> None:
    planner = make_planner(config, [fake_gpu(0, 32, 32)])
    verdict = fit_verdict(sized_item("Q4_K_M", 4 * GB), planner=planner)
    assert verdict["verdict"] == "fits-one-gpu"
    assert verdict["required_bytes"] > 4 * GB
    assert verdict["largest_gpu_free_bytes"] > 0
    assert verdict["suggested_quant"] is None
    assert verdict["approximate"] is True
    assert "approximation" in verdict["message"]


def test_fit_verdict_huge_model_wont_fit_and_suggests_a_smaller_sibling(
    config: Config,
) -> None:
    planner = make_planner(config, [fake_gpu(0, 32, 32)])
    siblings = [
        sized_item("Q8_0", 200 * GB),
        sized_item("Q4_K_M", 20 * GB),
        sized_item("IQ3_XXS", 12 * GB),
    ]
    verdict = fit_verdict(siblings[0], planner=planner, siblings=siblings)
    assert verdict["verdict"] == "wont-fit"
    # Largest that fits, not smallest available.
    assert verdict["suggested_quant"] == "Q4_K_M"
    assert "Q4_K_M" in verdict["message"]


def test_fit_verdict_no_sibling_fits(config: Config) -> None:
    planner = make_planner(config, [fake_gpu(0, 32, 32)])
    siblings = [sized_item("Q8_0", 200 * GB), sized_item("Q6_K", 150 * GB)]
    verdict = fit_verdict(siblings[0], planner=planner, siblings=siblings)
    assert verdict["verdict"] == "wont-fit"
    assert verdict["suggested_quant"] is None
    assert "smaller model" in verdict["message"]


def test_fit_verdict_needs_multiple_gpus(config: Config) -> None:
    planner = make_planner(config, [fake_gpu(0, 32, 32), fake_gpu(1, 32, 32)])
    verdict = fit_verdict(sized_item("Q4_K_M", 40 * GB), planner=planner)
    assert verdict["verdict"] == "needs-multiple-gpus"
    assert verdict["total_gpu_free_bytes"] > verdict["largest_gpu_free_bytes"]


def test_fit_verdict_unknown_when_size_is_missing(config: Config) -> None:
    planner = make_planner(config, [fake_gpu(0, 32, 32)])
    verdict = fit_verdict(sized_item("Q4_K_M", 0), planner=planner)
    assert verdict["verdict"] == "unknown"
    assert verdict["required_bytes"] == 0
    assert "did not report a size" in verdict["message"]
    assert verdict["size_known"] is False


def test_fit_verdict_unknown_without_gpus(config: Config) -> None:
    planner = make_planner(config, [])
    verdict = fit_verdict(sized_item("Q4_K_M", 4 * GB), planner=planner)
    assert verdict["verdict"] == "unknown"
    assert "GPU-only" in verdict["message"]


def test_fit_verdict_kv_allowance_grows_with_context(config: Config) -> None:
    planner = make_planner(config, [fake_gpu(0, 80, 80)])
    item = sized_item("Q4_K_M", 30 * GB)
    small = fit_verdict(item, planner=planner, ctx_size=8192)
    large = fit_verdict(item, planner=planner, ctx_size=65536)
    assert large["kv_allowance_bytes"] > small["kv_allowance_bytes"]
    assert large["required_bytes"] > small["required_bytes"]


def test_fit_verdict_uses_arch_hint_for_an_exact_kv_term(config: Config) -> None:
    from studioforge.types import GgufMeta

    planner = make_planner(config, [fake_gpu(0, 32, 32)])
    item = sized_item("Q4_K_M", 4 * GB)
    hint = GgufMeta(
        architecture="qwen2",
        n_layer=28,
        n_embd=3584,
        n_head=28,
        n_head_kv=4,
        n_ctx_train=32768,
    )
    verdict = fit_verdict(item, planner=planner, ctx_size=8192, arch_hint=hint)
    assert verdict["approximate"] is False
    assert "approximation" not in verdict["message"]
    # A real GQA KV cache at 8k is far smaller than the blanket 20% allowance.
    blanket = fit_verdict(item, planner=planner, ctx_size=8192)
    assert verdict["kv_allowance_bytes"] < blanket["kv_allowance_bytes"]
    assert verdict["verdict"] == "fits-one-gpu"


# ---------------------------------------------------------------------------
# Startup pruning of stale history rows
# ---------------------------------------------------------------------------


async def test_start_prunes_stale_terminal_rows(config: Config, db: Any, tmp_path: Path) -> None:
    """Terminal rows age out at startup; a paused transfer of any age survives."""
    import time as time_module

    old = time_module.time() - 90 * 86400
    for id, status in (("ancient-done", "completed"), ("ancient-paused", "paused")):
        db.upsert_download(
            {
                "id": id,
                "repo_id": "org/repo",
                "filename": f"{id}.gguf",
                "dest_path": str(tmp_path / f"{id}.gguf"),
                "status": status,
                "group_id": f"grp-{id}",
            }
        )
        db.connect().execute("UPDATE downloads SET updated_at = ? WHERE id = ?", (old, id))

    downloader = make_downloader(config, db, "http://127.0.0.1:1")
    try:
        await downloader.start()
        remaining = {row["id"] for row in db.list_downloads()}
        assert "ancient-done" not in remaining, "completed history must be pruned"
        assert "ancient-paused" in remaining, "paused rows are live intent, never pruned"
    finally:
        await downloader.stop()


# ---------------------------------------------------------------------------
# Registry rescan on group completion (api.app wiring)
# ---------------------------------------------------------------------------
#
# A finished download must become visible to /v1/models, the Models tab and the
# MCP list_models tool WITHOUT a manual scan -- the download_model tool's own
# docstring promises "list_models to see the model appear once complete", and
# before this wiring existed that promise was false.


def _progress(status: str, group_id: str = "grp-1") -> DownloadProgress:
    return DownloadProgress(
        id=f"{group_id}:file.gguf",
        group_id=group_id,
        repo_id="org/repo",
        filename="file.gguf",
        status=status,  # type: ignore[arg-type]
        downloaded_bytes=10,
        total_bytes=10,
        speed_bps=0.0,
        eta_s=None,
        error=None,
    )


class _RecordingRegistry:
    def __init__(self, *, fail: bool = False) -> None:
        self.scans = 0
        self.fail = fail

    def scan(self) -> Any:
        self.scans += 1
        if self.fail:
            raise RuntimeError("disk fell off")

        @dataclass
        class _Result:
            added: list[str] = field(default_factory=list)

        return _Result()


class _GroupStatusDownloader:
    def __init__(self, status: str) -> None:
        self._status = status

    def group_status(self, group_id: str) -> str:
        return self._status


async def test_rescan_fires_only_when_the_whole_group_is_complete() -> None:
    """One file done != model usable: shard 3 of 5 still missing means no scan."""
    from studioforge.api.app import rescan_when_group_completes

    registry = _RecordingRegistry()
    callback = rescan_when_group_completes(_GroupStatusDownloader("running"), registry)
    callback(_progress("completed"))
    await asyncio.sleep(0.05)
    assert registry.scans == 0

    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    callback(_progress("completed"))
    await asyncio.sleep(0.2)
    assert registry.scans == 1


async def test_rescan_ignores_non_completed_progress_events() -> None:
    from studioforge.api.app import rescan_when_group_completes

    registry = _RecordingRegistry()
    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    for status in ("queued", "running", "paused", "failed", "canceled"):
        callback(_progress(status))
    await asyncio.sleep(0.1)
    assert registry.scans == 0


async def test_rescan_never_raises_into_the_transfer_task() -> None:
    """A raising subscriber gets dropped by the downloader; this one must not be."""
    from studioforge.api.app import rescan_when_group_completes

    registry = _RecordingRegistry(fail=True)
    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    callback(_progress("completed"))  # must not raise
    await asyncio.sleep(0.2)
    assert registry.scans == 1  # it tried; the failure was contained and logged


def test_rescan_works_without_a_running_event_loop() -> None:
    """The callback degrades to a synchronous scan outside an event loop."""
    from studioforge.api.app import rescan_when_group_completes

    registry = _RecordingRegistry()
    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    callback(_progress("completed"))
    assert registry.scans == 1


async def test_rescan_task_is_strongly_referenced_until_done() -> None:
    """The scheduled scan task must be held by a strong reference.

    The event loop keeps only weak references to tasks, so an unreferenced
    ``loop.create_task(...)`` can be garbage-collected mid-flight -- the
    download says "completed" and the scan silently never runs.
    """
    from studioforge.api import app as app_module
    from studioforge.api.app import rescan_when_group_completes

    gate = threading.Event()

    class _BlockingRegistry:
        def __init__(self) -> None:
            self.scans = 0

        def scan(self) -> Any:
            self.scans += 1
            gate.wait(5.0)

            @dataclass
            class _Result:
                added: list = field(default_factory=list)

            return _Result()

    registry = _BlockingRegistry()
    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    try:
        callback(_progress("completed", group_id="grp-ref"))
        await asyncio.sleep(0.1)  # let the scan task start and block
        task = app_module._RESCAN_TASKS.get("grp-ref")
        assert task is not None and not task.done(), (
            "no strong reference to the in-flight scan task: it can be "
            "garbage-collected before it runs"
        )
    finally:
        gate.set()
    await asyncio.wait_for(task, 5.0)
    await asyncio.sleep(0.05)  # let the done-callback run
    assert registry.scans == 1
    assert "grp-ref" not in app_module._RESCAN_TASKS, "finished tasks must not accumulate"


async def test_rescan_is_deduplicated_while_one_is_pending_for_the_group() -> None:
    """Re-enqueueing an already-complete multi-shard group emits one completed
    event per file; that must queue ONE library scan, not one per shard."""
    from studioforge.api.app import rescan_when_group_completes

    gate = threading.Event()

    class _BlockingRegistry:
        def __init__(self) -> None:
            self.scans = 0

        def scan(self) -> Any:
            self.scans += 1
            gate.wait(5.0)

            @dataclass
            class _Result:
                added: list = field(default_factory=list)

            return _Result()

    registry = _BlockingRegistry()
    callback = rescan_when_group_completes(_GroupStatusDownloader("completed"), registry)
    try:
        callback(_progress("completed", group_id="grp-dedup"))
        await asyncio.sleep(0.1)  # first scan is now running and blocked
        for _ in range(4):
            callback(_progress("completed", group_id="grp-dedup"))
        await asyncio.sleep(0.1)
    finally:
        gate.set()
    await asyncio.sleep(0.3)
    assert registry.scans == 1, f"{registry.scans} scans queued for one group"


async def test_resumed_group_scans_exactly_once_on_completion(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """The start() resume path must end in one scan when the group completes.

    A crash leaves a ``running`` row and a ``.part`` file; the next start()
    resumes it. The subscriber must see no scan while the transfer runs and
    exactly one when the resumed group finishes -- resuming is the path the
    app actually takes after a restart, so the "model appears without a manual
    scan" promise has to hold here too.
    """
    from studioforge.api.app import rescan_when_group_completes

    data = blob(200 * 1024, seed=71)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    group_id = f"{REPO}:Q4_K_M"

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_name(dest.name + ".part").write_bytes(data[: 64 * 1024])
    db.upsert_download(
        {
            "id": f"{group_id}:{name}",
            "repo_id": REPO,
            "filename": name,
            "dest_path": str(dest),
            "status": "running",  # the crash case: nothing survives a restart
            "total_bytes": len(data),
            "downloaded_bytes": 64 * 1024,
            "sha256": hashlib.sha256(data).hexdigest(),
            "group_id": group_id,
        }
    )

    registry = _RecordingRegistry()
    downloader.subscribe(rescan_when_group_completes(downloader, registry))
    await downloader.start()
    await wait_group(downloader, group_id)
    await asyncio.sleep(0.3)  # let the scheduled scan task run

    assert dest.read_bytes() == data
    assert registry.scans == 1, (
        f"a resumed group that completed produced {registry.scans} scans, expected 1"
    )
    await downloader.stop()


async def test_canceled_group_never_triggers_a_scan(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """Cancellation must not rescan: the partial file is not a model."""
    from studioforge.api.app import rescan_when_group_completes

    data = blob(800 * 1024, seed=73)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    hf_server.slice_bytes = 16 * 1024
    hf_server.slice_delay_s = 0.03
    item = one(name, data)

    downloader = make_downloader(config, db, hf_server.endpoint)
    registry = _RecordingRegistry()
    downloader.subscribe(rescan_when_group_completes(downloader, registry))

    group_id = await downloader.enqueue(item)
    dest = downloader.dest_for(REPO, name)
    part = dest.with_name(dest.name + ".part")
    assert await wait_until(lambda: part.exists() and part.stat().st_size > 0)
    await downloader.cancel(group_id)
    await asyncio.sleep(0.3)

    assert downloader.group_status(group_id) == "canceled"
    assert registry.scans == 0, "a canceled download must not trigger a registry scan"
    await downloader.stop()


def test_build_state_subscribes_the_rescan_callback(config: Config) -> None:
    """The wiring itself: a Downloader built by the app carries the subscriber."""
    from studioforge.api.app import build_state

    state = build_state(config)
    try:
        subscribers = state.downloader._subscribers
        assert subscribers, "the app-built downloader must have the rescan subscriber"
        names = {getattr(cb, "__qualname__", "") for cb in subscribers}
        assert any("rescan_when_group_completes" in name for name in names)
    finally:
        state.db.close()


# ===========================================================================
# WP11 -- one writer per .part, completion proven on disk, retries
#
# Every test below traces to the 2026-08-18 incident (DECISIONS.md D24): a
# second process wrote into a live ``.part``, the live transfer died with
# WinError 32, and the other writer published a 22,576,551,872-byte file over a
# destination declared as 19,270,036,448 bytes -- passing verification, because
# verification only ever looked at the stream.
# ===========================================================================


def part_of(downloader: Downloader, name: str, repo_id: str = REPO) -> Path:
    dest = downloader.dest_for(repo_id, name)
    return dest.with_name(dest.name + ".part")


# --- exclusive ownership ---------------------------------------------------


async def test_a_second_writer_is_refused_instead_of_interleaving(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """The incident's first cause: nothing stopped two writers sharing one file."""
    from studioforge.core.downloader import _PartFile

    data = blob(200 * 1024, seed=101)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = part_of(downloader, name)
    marker = b"bytes belonging to the other process"

    with _PartFile(part) as owned:
        owned.write(marker)
        group_id = await downloader.enqueue(one(name, data))
        await wait_group(downloader, group_id, status="failed")
        error = downloader.group(group_id)[0].error or ""
        assert "another process is writing this file" in error

    # Not one byte was appended to somebody else's partial, and nothing was
    # published: refusing beats joining.
    assert part.read_bytes() == marker
    assert not dest.exists()
    assert hf_server.request_count(path) == 0
    await downloader.stop()


async def test_a_refused_part_file_is_not_retried(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """A held lock is another writer, not a hiccup. Backing off would not help."""
    from studioforge.core.downloader import _PartFile

    data = blob(64 * 1024, seed=103)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    with _PartFile(part_of(downloader, name)):
        group_id = await downloader.enqueue(one(name, data))
        await wait_group(downloader, group_id, status="failed")
    # Five jittered backoffs would take well over a minute; this must be instant.
    assert time.monotonic() - started < 5.0
    await downloader.stop()


async def test_the_lock_is_released_when_the_transfer_finishes(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """A completed download must not leave the name locked for the next one."""
    from studioforge.core.downloader import _PartFile

    data = blob(64 * 1024, seed=105)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    part = part_of(downloader, name)
    assert not part.exists()
    with _PartFile(part):  # would raise PartFileLockedError if the lock leaked
        pass
    part.unlink(missing_ok=True)
    await downloader.stop()


# --- completion proven on disk ---------------------------------------------


async def test_a_part_file_longer_than_the_stream_is_never_published(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident, reproduced exactly.

    The streamed byte count and the streamed sha256 both check out -- they
    describe the bytes *this* process sent to ``write()``. Foreign bytes landed
    in the file as well, so what is on disk is longer than what was streamed and
    is not the model. The old verifier published it; this one refuses.
    """
    from studioforge.core import downloader as dl

    data = blob(64 * 1024, seed=107)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data

    original = dl._PartFile.write
    intruded = {"once": False}

    def write(self: Any, chunk: bytes) -> None:
        original(self, chunk)
        if not intruded["once"]:
            intruded["once"] = True
            original(self, b"\x00" * 4096)  # somebody else's chunk

    monkeypatch.setattr(dl._PartFile, "write", write)

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id, status="failed")

    error = downloader.group(group_id)[0].error or ""
    assert "were streamed but the partial file holds" in error
    assert "65536" in error and "69632" in error
    dest = downloader.dest_for(REPO, name)
    assert not dest.exists(), "an interleaved file was published into the library"
    assert not part_of(downloader, name).exists(), "unknowable garbage was kept for a resume"
    await downloader.stop()


# --- adoption of a file already at the destination -------------------------


async def test_a_wrong_size_file_at_the_destination_is_quarantined(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """22.58 GB where 19.27 GB were declared, in miniature.

    The old code declined to adopt it and left it there, keeping its ``.gguf``
    name -- so the registry scanned it, listed it as a model, and every load
    against it failed somewhere far away from the cause.
    """
    data = blob(100 * 1024, seed=109)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    corrupt = blob(len(data) + 3000, seed=110)
    dest.write_bytes(corrupt)

    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)

    assert dest.read_bytes() == data
    quarantined = sorted(dest.parent.glob(dest.name + ".corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == corrupt, "the bad file was destroyed, not set aside"
    assert not quarantined[0].name.endswith(".gguf"), "still visible to the registry scanner"
    await downloader.stop()


async def test_a_checksum_mismatch_at_the_destination_is_quarantined(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """Right size, wrong bytes: only the published sha256 can tell."""
    data = blob(100 * 1024, seed=111)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob(100 * 1024, seed=112))

    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)

    assert dest.read_bytes() == data
    assert len(sorted(dest.parent.glob(dest.name + ".corrupt-*"))) == 1
    await downloader.stop()


async def test_a_large_file_is_adopted_on_size_alone_and_says_so(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Above the re-hash ceiling the checksum is not consulted. Documented, not hidden.

    Re-reading 20 GB on every enqueue of a model the user already has costs
    minutes of disk bandwidth; the size check is the one that catches the
    failure this whole work package exists for.
    """
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "ADOPT_HASH_MAX_BYTES", 1)
    data = blob(100 * 1024, seed=113)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    other = blob(100 * 1024, seed=114)  # correct size, different bytes
    dest.write_bytes(other)

    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)
    assert dest.read_bytes() == other
    assert hf_server.request_count(path) == 0
    assert not sorted(dest.parent.glob(dest.name + ".corrupt-*"))
    await downloader.stop()


async def test_enqueue_never_moves_anything_on_disk(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """Queueing is not the moment to rearrange somebody's model library.

    The quarantine happens when the transfer starts instead, so a repo whose
    blob listing is simply out of date cannot cost a user their file before
    they have even seen the download begin.
    """
    data = blob(100 * 1024, seed=115)
    name = "model-Q4_K_M.gguf"
    hf_server.files[resolve_path(REPO, name)] = data

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    corrupt = blob(len(data) + 3000, seed=116)
    dest.write_bytes(corrupt)

    item = one(name, data)
    state = _state_for(downloader, item, name, len(data))
    assert await downloader._adopt_complete(state, quarantine=False) is False
    assert dest.read_bytes() == corrupt
    assert not sorted(dest.parent.glob(dest.name + ".corrupt-*"))
    await downloader.stop()


def _state_for(downloader: Downloader, item: Any, name: str, size: int) -> Any:
    from studioforge.core.downloader import _FileState

    return _FileState(
        id=f"{item.group_id}:{name}",
        group_id=item.group_id,
        repo_id=item.repo_id,
        filename=name,
        dest=downloader.dest_for(item.repo_id, name),
        status="queued",
        total_bytes=size,
    )


# --- retry with resume -----------------------------------------------------


async def test_a_transient_5xx_is_retried_and_then_succeeds(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "_RETRY_BASE_S", 0.01)
    data = blob(64 * 1024, seed=117)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [503, 500]

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)

    assert downloader.dest_for(REPO, name).read_bytes() == data
    assert hf_server.statuses_for(path) == [503, 500, 200]
    await downloader.stop()


async def test_retries_are_exhausted_before_a_download_is_failed(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "_RETRY_BASE_S", 0.01)
    data = blob(32 * 1024, seed=119)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [503] * 10

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id, status="failed")

    assert hf_server.request_count(path) == dl.DOWNLOAD_MAX_ATTEMPTS
    assert "503" in (downloader.group(group_id)[0].error or "")
    await downloader.stop()


async def test_a_404_is_not_retried(config: Config, db: Database, hf_server: ServerState) -> None:
    """A definite answer from the server is an answer, not a hiccup."""
    name = "missing-Q4_K_M.gguf"
    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, b"x" * 32))
    await wait_group(downloader, group_id, status="failed")
    assert hf_server.request_count(resolve_path(REPO, name)) == 1
    await downloader.stop()


async def test_a_retry_resumes_from_the_bytes_already_written(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of retrying at all: a 40 GiB transfer must not start over."""
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "_RETRY_BASE_S", 0.01)
    data = blob(200 * 1024, seed=121)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [503]

    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    prefix = 64 * 1024
    part_of(downloader, name).write_bytes(data[:prefix])

    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)

    assert dest.read_bytes() == data
    # Both attempts asked to continue from the same offset.
    assert hf_server.range_headers(path) == [f"bytes={prefix}-"] * 2
    await downloader.stop()


async def test_the_retry_state_is_visible_to_the_gui(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"It says running and nothing is moving" must be answerable from the payload."""
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "_RETRY_BASE_S", 0.05)
    data = blob(32 * 1024, seed=123)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [503]

    seen: list[dict[str, Any]] = []
    downloader = make_downloader(
        config, db, hf_server.endpoint, on_progress=lambda p: seen.append(p.to_dict())
    )
    group_id = await downloader.enqueue(one(name, data))
    await wait_group(downloader, group_id)

    backing_off = [row for row in seen if row["next_retry_at"] is not None]
    assert backing_off, "nothing in the payload said the download was waiting to retry"
    row = backing_off[0]
    assert row["attempt"] == 1
    assert row["max_attempts"] == dl.DOWNLOAD_MAX_ATTEMPTS
    assert row["retry_in_s"] is not None and row["retry_in_s"] >= 0
    assert "503" in str(row["last_error"])
    # And it clears once the transfer is running again.
    assert seen[-1]["next_retry_at"] is None
    await downloader.stop()


async def test_pause_interrupts_a_backoff_immediately(
    config: Config,
    db: Database,
    hf_server: ServerState,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Pause button that does nothing for 30 seconds is a broken Pause button."""
    from studioforge.core import downloader as dl

    monkeypatch.setattr(dl, "_RETRY_BASE_S", 30.0)
    data = blob(32 * 1024, seed=125)
    name = "model-Q4_K_M.gguf"
    path = resolve_path(REPO, name)
    hf_server.files[path] = data
    hf_server.status_script[path] = [503]

    downloader = make_downloader(config, db, hf_server.endpoint)
    group_id = await downloader.enqueue(one(name, data))
    assert await wait_until(
        lambda: any(p.next_retry_at is not None for p in downloader.group(group_id))
    ), "the download never reached its backoff"

    started = time.monotonic()
    await downloader.pause(group_id)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"pause waited {elapsed:.1f}s for the backoff sleep to end"
    assert downloader.group_status(group_id) == "paused"
    await downloader.stop()


async def test_a_failed_download_records_what_resume_would_continue_from(
    config: Config, db: Database, hf_server: ServerState
) -> None:
    """The question the GUI could not answer on 2026-08-18."""
    name = "missing-Q4_K_M.gguf"
    downloader = make_downloader(config, db, hf_server.endpoint)
    dest = downloader.dest_for(REPO, name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_of(downloader, name).write_bytes(b"z" * 4096)

    group_id = await downloader.enqueue(one(name, b"z" * 8192))
    await wait_group(downloader, group_id, status="failed")
    progress = downloader.group(group_id)[0]
    assert progress.part_bytes == 4096
    assert progress.to_dict()["part_bytes"] == 4096
    await downloader.stop()


# --- retry policy in isolation ---------------------------------------------


def test_transient_and_fatal_errors_are_told_apart() -> None:
    from studioforge.core.downloader import (
        ChecksumMismatchError,
        PartFileLockedError,
        _is_transient,
    )

    assert _is_transient(httpx.ReadTimeout("read timed out")) is True
    assert _is_transient(httpx.ConnectError("connection refused")) is True
    assert _is_transient(UpstreamError("boom", details={"status": 503})) is True
    assert _is_transient(UpstreamError("slow down", details={"status": 429})) is True
    assert _is_transient(PermissionError(13, "in use")) is True

    assert _is_transient(UpstreamError("gone", details={"status": 404})) is False
    assert _is_transient(UpstreamError("no", details={"status": 401})) is False
    assert _is_transient(UpstreamError("size mismatch", details={"declared_bytes": 1})) is False
    assert _is_transient(ChecksumMismatchError("bad hash")) is False
    assert _is_transient(PartFileLockedError("taken")) is False
    assert _is_transient(OSError(28, "No space left on device")) is False


def test_backoff_grows_capped_and_jittered() -> None:
    from studioforge.core.downloader import _backoff_delay

    first = [_backoff_delay(1) for _ in range(50)]
    fourth = [_backoff_delay(4) for _ in range(50)]
    assert min(first) >= 2.0 * 0.8 - 1e-9
    assert max(first) <= 2.0 * 1.2 + 1e-9
    assert min(fourth) > max(first), "the backoff does not grow"
    assert max(_backoff_delay(20) for _ in range(50)) <= 60.0 * 1.2 + 1e-9
    assert len({round(value, 6) for value in first}) > 1, "no jitter: retries stay in lockstep"


def test_an_explicit_retry_after_is_honoured_but_still_capped() -> None:
    from studioforge.core.downloader import _backoff_delay

    assert _backoff_delay(1, retry_after=30.0) >= 30.0 * 0.8
    assert _backoff_delay(1, retry_after=86400.0) <= 60.0 * 1.2 + 1e-9


# --- the queue panel's extra line ------------------------------------------


def test_queue_notes_say_what_is_happening_and_what_resume_will_do() -> None:
    from studioforge.gui.tabs.download import _queue_notes

    retrying = _queue_notes(
        [
            {
                "group_id": "g1",
                "status": "running",
                "attempt": 2,
                "max_attempts": 5,
                "next_retry_at": time.time() + 8,
                "retry_in_s": 8.4,
                "last_error": "timeout",
            }
        ]
    )
    assert retrying["g1"] == "retrying in 8s (attempt 2/5): timeout"

    with_part = _queue_notes([{"group_id": "g2", "status": "failed", "part_bytes": 3 * GB}])
    assert with_part["g2"] == "Resume continues from 3.0 GiB"

    without_part = _queue_notes([{"group_id": "g3", "status": "failed", "part_bytes": 0}])
    assert without_part["g3"] == "Resume will restart from the beginning (no partial file)"

    assert _queue_notes([{"group_id": "g4", "status": "running", "attempt": 0}]) == {}


# --- the in-use guard ------------------------------------------------------


def test_file_in_use_check_calls_the_api_the_supervisor_actually_has(tmp_path: Path) -> None:
    """``supervisor.all()`` does not exist; this guard called it anyway.

    Every invocation raised ``AttributeError``, so the one thing standing
    between a forced download and the weights of a *running* model was a method
    name that had never been executed.
    """
    from studioforge.api.app import _file_in_use_check

    loaded = tmp_path / "loaded.gguf"
    loaded.write_bytes(b"weights")
    other = tmp_path / "other.gguf"
    other.write_bytes(b"weights")

    class _Instance:
        model_id = "loaded-model"

    class _Supervisor:
        def list(self) -> list[Any]:
            return [_Instance()]

        def all(self) -> list[Any]:  # pragma: no cover - must never be called
            raise AssertionError("supervisor.all() does not exist")

    class _Record:
        shards = [loaded]
        mmproj_path = None

    class _Registry:
        def resolve(self, name: str) -> Any:
            return _Record() if name == "loaded-model" else None

    check = _file_in_use_check(_Supervisor(), _Registry())
    assert check(loaded) is True
    assert check(other) is False


def test_file_in_use_check_answers_conservatively_when_it_cannot_tell(tmp_path: Path) -> None:
    """"I do not know" must not read as "go ahead and delete it"."""
    from studioforge.api.app import _file_in_use_check

    class _BrokenSupervisor:
        def list(self) -> list[Any]:
            raise RuntimeError("supervisor is wedged")

    class _Registry:
        def resolve(self, name: str) -> Any:  # pragma: no cover - never reached
            return None

    check = _file_in_use_check(_BrokenSupervisor(), _Registry())
    assert check(tmp_path / "anything.gguf") is True
