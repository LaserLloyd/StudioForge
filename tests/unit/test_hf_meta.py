"""Tests for reading a GGUF header off HuggingFace and the context matrix.

Three halves (the arithmetic is worth its own):

* **RemoteRangeFile** against an ``httpx.MockTransport`` that implements real
  ``Range`` semantics. These own the contract that makes the whole feature
  affordable: a ``seek()`` past a numeric array must fetch nothing, and the read
  must refuse to become a download.
* **meta_from_gguf parity** -- the remote header and the local file must produce
  the same geometry, because the entire point is to answer before downloading
  exactly what the registry will answer after.
* **context_matrix** on a fake 2x32 GiB + 2x24 GiB rig, with a hybrid (Qwen3.5)
  and an iSWA (Gemma-4) model, plus the degrade path when no header is readable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core import hf_meta
from studioforge.core.gguf import read_meta
from studioforge.core.hf_meta import (
    CONTEXT_TIERS,
    RemoteHeaderError,
    RemoteRangeFile,
    context_line,
    context_matrix,
    context_tooltip,
    geometry_line,
    registry_sibling_meta,
    remote_meta,
)
from studioforge.core.planner import Planner
from studioforge.types import GB, GgufMeta, GpuInfo, ModelRecord
from tests.unit.test_gguf import ARRAY, INT32, STRING, Arr, llm_kv, write_gguf

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProbe:
    """A fixed GPU inventory; nothing here talks to NVML."""

    backend = "fake"

    def __init__(self, gpus: list[GpuInfo]) -> None:
        self._gpus = gpus

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [g.model_copy(deep=True) for g in self._gpus]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return next((g.model_copy(deep=True) for g in self._gpus if g.index == index), None)

    def compute_processes(self) -> list[Any]:
        return []

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


def gpu(index: int, name: str, gib: float, cc: tuple[int, int]) -> GpuInfo:
    total = int(gib * GB)
    return GpuInfo(
        index=index,
        name=name,
        total_bytes=total,
        free_bytes=total,
        used_bytes=0,
        compute_capability=cc,
    )


def rig_4() -> list[GpuInfo]:
    """The real box this was built for: 2x RTX 5090 + 2x RTX 3090."""
    return [
        gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(1, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(2, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
        gpu(3, "NVIDIA GeForce RTX 3090", 24, (8, 6)),
    ]


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )


@pytest.fixture(autouse=True)
def _clear_header_cache() -> Any:
    """A module-level cache would otherwise leak headers between tests."""
    hf_meta.clear_memory_cache()
    yield
    hf_meta.clear_memory_cache()


def planner_for(config: Config, gpus: list[GpuInfo] | None = None) -> Planner:
    return Planner(config, FakeProbe(gpus if gpus is not None else rig_4()), log_plans=False)


# ---------------------------------------------------------------------------
# Range-serving transport
# ---------------------------------------------------------------------------


class RangeServer:
    """A MockTransport handler with the Range semantics HF's CDN has."""

    def __init__(self, data: bytes, *, honour_range: bool = True) -> None:
        self.data = data
        self.honour_range = honour_range
        self.ranges: list[str | None] = []
        self.served_bytes = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        self.ranges.append(rng)
        if not self.honour_range or not rng:
            self.served_bytes += len(self.data)
            return httpx.Response(200, content=self.data)
        match = re.match(r"bytes=(\d+)-(\d+)", rng)
        assert match is not None
        start, end = int(match.group(1)), int(match.group(2))
        if start >= len(self.data):
            return httpx.Response(416, headers={"Content-Range": f"bytes */{len(self.data)}"})
        chunk = self.data[start : end + 1]
        self.served_bytes += len(chunk)
        return httpx.Response(
            206,
            content=chunk,
            headers={
                "Content-Range": f"bytes {start}-{start + len(chunk) - 1}/{len(self.data)}",
                "Accept-Ranges": "bytes",
            },
        )


def range_file(server: RangeServer, **kwargs: Any) -> RemoteRangeFile:
    client = httpx.Client(transport=httpx.MockTransport(server))
    return RemoteRangeFile(client, "https://example.invalid/model.gguf", **kwargs)


def use_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Point ``hf_meta``'s header reads at a mock transport.

    Patches the module's own ``open_client`` rather than ``httpx.Client``:
    replacing the class globally breaks any library that evaluates
    ``httpx.Client | None`` at import time, and ``huggingface_hub`` -- imported
    lazily by ``hf_search`` -- does exactly that.
    """
    monkeypatch.setattr(
        hf_meta, "open_client", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )


def refusing(status: int) -> Any:
    """A transport that answers every request with ``status``."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    return handler


# ---------------------------------------------------------------------------
# Synthetic models
# ---------------------------------------------------------------------------


def hybrid_gguf(path: Path, *, vocab: int = 0) -> Path:
    """A Qwen3.5-shaped hybrid: 64 blocks, every 4th one attention."""
    kv = llm_kv(
        "qwen35",
        block_count=64,
        embedding_length=6144,
        context_length=262144,
        **{
            "attention.head_count": 24,
            "attention.head_count_kv": 4,
            "attention.key_length": 256,
            "attention.value_length": 256,
            "full_attention_interval": 4,
            "ssm.conv_kernel": 4,
            "ssm.inner_size": 6144,
            "ssm.state_size": 128,
            "ssm.group_count": 16,
            "general.file_type": 17,
        },
    )
    if vocab:
        # A long numeric array in the MIDDLE of the block, with a key after it,
        # so a test can prove the parser seeks past it instead of reading it.
        kv.append(("tokenizer.ggml.token_type", ARRAY, Arr(INT32, [1] * vocab)))
        kv.append(("general.name", STRING, "after-the-big-array"))
    return write_gguf(path, kv, [("blk.0.attn_q.weight", (256, 256), 8)])


def hybrid_meta() -> GgufMeta:
    return GgufMeta(
        architecture="qwen35",
        n_layer=64,
        n_embd=6144,
        n_head=24,
        n_head_kv=4,
        n_ctx_train=262144,
        n_embd_head_k=256,
        n_embd_head_v=256,
        quant_label="Q5_K_M",
        extra={
            "full_attention_interval": 4,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 6144,
            "ssm_state_size": 128,
            "ssm_group_count": 16,
        },
    )


def iswa_meta() -> GgufMeta:
    """Gemma-4-shaped: 5 sliding-window layers per full-attention one."""
    pattern = [(i % 6) != 5 for i in range(60)]
    return GgufMeta(
        architecture="gemma4",
        n_layer=60,
        n_embd=4096,
        n_head=32,
        n_head_kv=16,
        n_ctx_train=131072,
        n_embd_head_k=512,
        n_embd_head_v=512,
        quant_label="Q8_0",
        extra={
            "swa_window": 1024,
            "swa_pattern": pattern,
            "swa_key_length": 256,
            "swa_value_length": 256,
            "head_count_kv_values": [16 if swa else 4 for swa in pattern],
        },
    )


# ===========================================================================
# RemoteRangeFile
# ===========================================================================


def test_range_file_reads_forward_in_chunks() -> None:
    server = RangeServer(bytes(range(256)) * 64)  # 16 KiB
    fh = range_file(server, chunk_bytes=4096)

    assert fh.read(10) == server.data[:10]
    assert fh.tell() == 10
    assert fh.read(20) == server.data[10:30]
    # Both reads came out of the same cached chunk.
    assert server.ranges == ["bytes=0-4095"]
    assert fh.size == len(server.data)


def test_range_file_seek_fetches_nothing() -> None:
    """The whole feature rests on this: a skip must not become a download."""
    server = RangeServer(bytes(range(256)) * 4096)  # 1 MiB
    fh = range_file(server, chunk_bytes=4096)

    fh.read(8)
    before = fh.bytes_fetched
    fh.seek(900_000)
    assert fh.bytes_fetched == before
    assert fh.tell() == 900_000
    fh.read(4)
    # Exactly one more chunk, at the seek target -- not everything in between.
    assert fh.bytes_fetched == before + 4096
    assert server.ranges[-1] == "bytes=897024-901119"


def test_range_file_refuses_to_exceed_the_cap() -> None:
    server = RangeServer(b"x" * (64 * 1024))
    fh = range_file(server, chunk_bytes=4096, max_bytes=8192)

    fh.read(4096)
    fh.read(4096)
    with pytest.raises(RemoteHeaderError, match="remote-read cap"):
        fh.read(4096)


def test_range_file_stops_at_end_of_file() -> None:
    server = RangeServer(b"abcdef")
    fh = range_file(server, chunk_bytes=4096)

    assert fh.read(100) == b"abcdef"
    assert fh.read(1) == b""


def test_range_file_refuses_a_server_that_ignores_range() -> None:
    """A 200 at a non-zero offset means the body is the whole 40 GB file."""
    server = RangeServer(b"y" * 20_000, honour_range=False)
    fh = range_file(server, chunk_bytes=4096)

    fh.read(16)  # offset 0 is fine: take the head of the stream
    fh.seek(10_000)
    with pytest.raises(RemoteHeaderError, match="ignored the Range header"):
        fh.read(16)


def test_range_file_refuses_an_unbounded_read() -> None:
    fh = range_file(RangeServer(b"z" * 10))
    with pytest.raises(RemoteHeaderError, match="unbounded read"):
        fh.read(-1)


def test_range_file_reports_an_http_error_in_terms_of_the_token() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "gated"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fh = RemoteRangeFile(client, "https://example.invalid/m.gguf")
    with pytest.raises(RemoteHeaderError, match="hf.token"):
        fh.read(8)


# ===========================================================================
# remote_meta / meta_from_gguf parity
# ===========================================================================

_GEOMETRY_FIELDS = (
    "architecture",
    "n_layer",
    "n_embd",
    "n_head",
    "n_head_kv",
    "n_ctx_train",
    "n_embd_head_k",
    "n_embd_head_v",
    "quant_label",
)


async def test_remote_meta_matches_read_meta_on_the_same_bytes(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One implementation of kv -> GgufMeta, proven on the same file twice."""
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    local = read_meta(path)
    remote = await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")

    for field in _GEOMETRY_FIELDS:
        assert getattr(remote, field) == getattr(local, field), field
    # The hybrid keys WP1 added must survive the remote path too -- they are the
    # difference between charging this model 4x its real KV and not.
    for key in ("full_attention_interval", "ssm_inner_size", "ssm_state_size"):
        assert remote.extra[key] == local.extra[key]


async def test_remote_meta_leaves_tensor_bytes_to_the_caller(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tensor table is read remotely; HF's file sizes are the weight total."""
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    use_transport(monkeypatch, RangeServer(path.read_bytes()))

    remote = await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")

    assert remote.tensor_bytes == 0
    assert read_meta(path).tensor_bytes > 0
    # And no invented "missing shards" from probing a disk the file is not on.
    assert "missing_shards" not in remote.extra


async def test_remote_meta_skips_the_bulk_arrays(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200k-entry token_type array is seeked over, not downloaded.

    Chunked small so the assertion is about the seek and not about the whole
    fixture happening to fit in one 1 MiB chunk; the real tokenizer arrays on a
    250k-vocabulary model are 10-15 MB, i.e. many chunks.
    """
    path = hybrid_gguf(tmp_path / "big-Q5_K_M.gguf", vocab=200_000)
    data = path.read_bytes()
    server = RangeServer(data)
    use_transport(monkeypatch, server)

    meta = await remote_meta(config, "acme/big-GGUF", "big-Q5_K_M.gguf", chunk_bytes=64 * 1024)

    assert meta.n_layer == 64  # the keys after the array were still read
    assert meta.n_vocab == 0  # token_type is not the vocabulary
    assert len(data) > 700_000
    # Well under the file: the array's payload never crossed the wire.
    assert server.served_bytes < len(data) // 2


async def test_remote_meta_is_cached_in_memory_and_on_disk(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    first = await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")
    requests_after_first = len(server.ranges)
    await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")
    assert len(server.ranges) == requests_after_first  # memory hit

    hf_meta.clear_memory_cache()
    third = await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")
    assert len(server.ranges) == requests_after_first  # disk hit
    assert third.n_layer == first.n_layer

    cached = list(hf_meta.cache_dir(config).glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text())["key"].startswith("acme/hybrid-GGUF")


async def test_a_stale_parser_version_invalidates_the_disk_cache(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same reasoning as the registry's cache key: a changed parser must re-read."""
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")
    entry = next(iter(hf_meta.cache_dir(config).glob("*.json")))
    payload = json.loads(entry.read_text())
    payload["meta_format_version"] = 1
    entry.write_text(json.dumps(payload))
    hf_meta.clear_memory_cache()
    before = len(server.ranges)

    await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")

    assert len(server.ranges) > before


async def test_remote_meta_reports_a_gated_repo_rather_than_guessing(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_transport(monkeypatch, refusing(401))

    with pytest.raises(RemoteHeaderError, match="hf.token"):
        await remote_meta(config, "acme/gated-GGUF", "m.gguf")


async def test_remote_meta_sends_the_token_as_a_header(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never in the URL: a token in a query string lands in every proxy log."""
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    seen: list[httpx.Request] = []

    server = RangeServer(path.read_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return server(request)

    use_transport(monkeypatch, handler)
    config.hf.token = "hf_secret"

    await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")

    assert seen[0].headers["Authorization"] == "Bearer hf_secret"
    assert "hf_secret" not in str(seen[0].url)


async def test_a_truncated_remote_header_is_an_error_not_a_blank_meta(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = hybrid_gguf(tmp_path / "hybrid-Q5_K_M.gguf")
    use_transport(monkeypatch, RangeServer(path.read_bytes()[:40]))

    with pytest.raises(RemoteHeaderError, match="not readable as GGUF"):
        await remote_meta(config, "acme/hybrid-GGUF", "hybrid-Q5_K_M.gguf")


# ===========================================================================
# Where the geometry comes from
# ===========================================================================


class FakeRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = records

    def all(self) -> list[ModelRecord]:
        return list(self._records)


def sibling_record(publisher: str, repo: str) -> ModelRecord:
    return ModelRecord(
        id=f"{publisher}/{repo}",
        name=repo,
        path=Path(f"/models/{publisher}/{repo}/model.gguf"),
        publisher=publisher,
        repo=repo,
        meta=hybrid_meta(),
    )


def test_registry_sibling_supplies_the_geometry_for_free() -> None:
    registry = FakeRegistry([sibling_record("acme", "Hybrid-GGUF")])
    meta = registry_sibling_meta(registry, "acme/Hybrid-GGUF")
    assert meta is not None
    assert meta.n_layer == 64


def test_registry_sibling_matches_case_insensitively_and_only_that_repo() -> None:
    registry = FakeRegistry([sibling_record("acme", "Hybrid-GGUF")])
    assert registry_sibling_meta(registry, "ACME/hybrid-gguf") is not None
    assert registry_sibling_meta(registry, "acme/Other-GGUF") is None
    assert registry_sibling_meta(registry, "not-a-repo-id") is None
    assert registry_sibling_meta(None, "acme/Hybrid-GGUF") is None


async def test_repo_arch_meta_prefers_a_sibling_over_the_network(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core.hf_search import GgufFileInfo, GgufRepoInfo

    repo = GgufRepoInfo(
        repo_id="acme/Hybrid-GGUF",
        publisher="acme",
        name="Hybrid-GGUF",
        downloads=1,
        likes=0,
        gated=False,
        private=False,
        last_modified=None,
        files=[
            GgufFileInfo(
                filename="hybrid-Q5_K_M.gguf",
                size_bytes=21 * GB,
                quant="Q5_K_M",
                is_mmproj=False,
                shard_index=None,
                shard_total=None,
                sha256=None,
                lfs_oid=None,
            )
        ],
    )
    server = RangeServer(b"")
    use_transport(monkeypatch, server)

    arch = await hf_meta.repo_arch_meta(
        config, repo, registry=FakeRegistry([sibling_record("acme", "Hybrid-GGUF")])
    )

    assert arch.source == "registry-sibling"
    assert arch.meta is not None
    assert server.ranges == []  # not one byte of network


async def test_repo_arch_meta_degrades_with_a_reason(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.core.hf_search import GgufFileInfo, GgufRepoInfo

    repo = GgufRepoInfo(
        repo_id="acme/Gated-GGUF",
        publisher="acme",
        name="Gated-GGUF",
        downloads=1,
        likes=0,
        gated=True,
        private=False,
        last_modified=None,
        files=[
            GgufFileInfo(
                filename="m-Q4_K_M.gguf",
                size_bytes=5 * GB,
                quant="Q4_K_M",
                is_mmproj=False,
                shard_index=None,
                shard_total=None,
                sha256=None,
                lfs_oid=None,
            )
        ],
    )
    use_transport(monkeypatch, refusing(403))

    arch = await hf_meta.repo_arch_meta(config, repo, registry=None)

    assert arch.meta is None
    assert arch.source is None
    assert "hf.token" in (arch.unavailable or "")


# ===========================================================================
# Placement profiles
# ===========================================================================


def test_placements_name_the_real_cards(config: Config) -> None:
    profiles = hf_meta.placements_for(hf_meta.idle_planner(planner_for(config)))

    assert [p.key for p in profiles] == ["single_best", "dual_best", "all"]
    assert [p.label for p in profiles] == [
        "1x RTX 5090",
        "2x RTX 5090",
        "all 4 GPUs (2x RTX 5090 + 2x RTX 3090)",
    ]
    assert [p.short_label for p in profiles] == ["1x5090", "2x5090", "all"]
    assert [p.devices for p in profiles] == [(0,), (0, 1), (0, 1, 2, 3)]


def test_placements_pick_the_best_class_not_the_biggest_card(config: Config) -> None:
    """Compute capability outranks VRAM: a 5090 beats a bigger, older card."""
    gpus = [
        gpu(0, "NVIDIA RTX A6000", 48, (8, 6)),
        gpu(1, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(2, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
    ]
    profiles = hf_meta.placements_for(hf_meta.idle_planner(planner_for(config, gpus)))

    assert profiles[0].devices == (1,)
    assert profiles[1].devices == (1, 2)
    assert profiles[-1].key == "all"


def test_a_single_gpu_rig_gets_exactly_one_profile(config: Config) -> None:
    profiles = hf_meta.placements_for(
        hf_meta.idle_planner(planner_for(config, [gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0))]))
    )
    assert len(profiles) == 1
    assert profiles[0].label == "1x RTX 5090"


def test_a_two_gpu_rig_collapses_dual_and_all(config: Config) -> None:
    gpus = [
        gpu(0, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
        gpu(1, "NVIDIA GeForce RTX 5090", 32, (12, 0)),
    ]
    profiles = hf_meta.placements_for(hf_meta.idle_planner(planner_for(config, gpus)))

    assert [p.devices for p in profiles] == [(0,), (0, 1)]
    assert profiles[-1].key == "all"


def test_excluded_devices_are_not_placements(config: Config) -> None:
    config.planner.excluded_devices = [2, 3]
    profiles = hf_meta.placements_for(hf_meta.idle_planner(planner_for(config)))

    assert [p.devices for p in profiles] == [(0,), (0, 1)]


def test_capacity_ignores_what_is_loaded_right_now(config: Config) -> None:
    """The pre-download question is about the hardware, not this instant."""
    busy = rig_4()
    busy[0].free_bytes = 0
    busy[0].used_bytes = busy[0].total_bytes
    profiles = hf_meta.placements_for(hf_meta.idle_planner(planner_for(config, busy)))

    # 32 GiB minus the 10% headroom, as if nothing were loaded.
    assert profiles[0].capacity_bytes == pytest.approx(int(32 * GB * 0.9), rel=0.01)


# ===========================================================================
# The context matrix
# ===========================================================================


def by_label(matrix: dict[str, Any], label: str) -> dict[str, Any]:
    return next(p for p in matrix["placements"] if p["short_label"] == label)


def test_hybrid_27b_context_matrix(config: Config) -> None:
    """A 21 GB hybrid 27B: one 5090 serves 64k, the pair serves the full 256k.

    The single-card row is the interesting one. 21 GiB of weights plus ~6% of
    compute buffers leaves ~6 GiB of a 28.8 GiB idle budget for KV, and this
    architecture costs ~64 KiB/token (only every 4th layer caches), so 64k fits
    at f16, 128k needs the q8_0 cache and 256k does not fit at all.
    """
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config))

    assert matrix["attention_kind"] == "hybrid"
    assert matrix["source"] is None
    assert matrix["approximate"] is False

    single = by_label(matrix, "1x5090")
    assert single["weights_fit"] is True
    assert single["fits"]["65536"] is True
    assert single["kv_cache_type"]["65536"] == "f16"
    assert single["fits"]["131072"] is True
    assert single["kv_cache_type"]["131072"] == "q8_0"
    assert single["fits"]["262144"] is False
    assert single["max_ctx"] >= 65536
    assert single["max_ctx_q8"] > single["max_ctx"]

    dual = by_label(matrix, "2x5090")
    assert dual["fits"] == {"65536": True, "131072": True, "262144": True}
    assert set(dual["kv_cache_type"].values()) == {"f16"}
    assert dual["max_ctx"] == 262144
    assert dual["max_ctx_q8"] is None  # nothing to gain, so nothing reported


def test_tiers_above_the_trained_window_are_absent(config: Config) -> None:
    """512k on a 256k model is not a memory question (D14)."""
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config))

    assert matrix["tiers"] == [65536, 131072, 262144]
    assert 524288 not in matrix["tiers"]
    for placement in matrix["placements"]:
        assert "524288" not in placement["fits"]
    assert all(p["max_ctx"] <= 262144 for p in matrix["placements"])


def test_iswa_geometry_is_charged_per_layer(config: Config) -> None:
    """Gemma-4's sliding-window layers must not be sized at full context."""
    meta = iswa_meta()
    matrix = context_matrix(meta, 20 * GB, planner=planner_for(config))

    assert matrix["attention_kind"] == "iswa"
    assert matrix["tiers"] == [65536, 131072]
    single = by_label(matrix, "1x5090")
    assert single["fits"]["131072"] is True
    # The uniform figure for this model is ~1.9 MB/token; the real one is a
    # fraction of that, and the matrix must be built on the real one.
    assert matrix["kv_bytes_per_token_f16"] < 200 * 1024


def test_weights_that_do_not_fit_are_distinguishable_from_a_small_context(
    config: Config,
) -> None:
    matrix = context_matrix(iswa_meta(), 40 * GB, planner=planner_for(config))

    single = by_label(matrix, "1x5090")
    assert single["weights_fit"] is False
    assert single["max_ctx"] == 0
    assert not any(single["fits"].values())

    everything = by_label(matrix, "all")
    assert everything["weights_fit"] is True


def test_a_model_larger_than_the_rig_fits_nowhere(config: Config) -> None:
    matrix = context_matrix(hybrid_meta(), 300 * GB, planner=planner_for(config))

    assert all(p["weights_fit"] is False for p in matrix["placements"])
    assert all(p["max_ctx"] == 0 for p in matrix["placements"])


def test_the_degrade_path_keeps_the_shape_and_says_why(config: Config) -> None:
    matrix = context_matrix(
        None,
        21 * GB,
        planner=planner_for(config),
        unavailable="gated repository; set hf.token",
        source="remote-gguf-header",
    )

    assert matrix["approximate"] is True
    assert matrix["source"] is None  # no meta means no source, whatever we tried
    assert matrix["unavailable"] == "gated repository; set hf.token"
    assert matrix["attention_kind"] == "unknown"
    assert matrix["tiers"] == list(CONTEXT_TIERS)  # no trained window to cap by
    assert [p["short_label"] for p in matrix["placements"]] == ["1x5090", "2x5090", "all"]
    assert by_label(matrix, "1x5090")["weights_fit"] is True


def test_a_vision_repo_is_charged_for_its_projector(config: Config) -> None:
    planner = planner_for(config)
    plain = context_matrix(hybrid_meta(), 21 * GB, planner=planner)
    vision = context_matrix(hybrid_meta(), 21 * GB, planner=planner, mmproj_bytes=2 * GB)

    assert by_label(vision, "1x5090")["max_ctx"] < by_label(plain, "1x5090")["max_ctx"]


def test_the_matrix_agrees_with_the_planner_it_will_be_checked_by(config: Config) -> None:
    """The cell and a real plan must not disagree; that is the whole contract."""
    planner = hf_meta.idle_planner(planner_for(config))
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner)
    record = hf_meta._throwaway_record("hf:acme/x#Q5_K_M", hybrid_meta(), 21 * GB, 0)

    for placement in matrix["placements"]:
        for tier, fits in placement["fits"].items():
            kv = placement["kv_cache_type"].get(tier, "f16")
            estimate = planner.fits_on(
                record, devices=placement["devices"], ctx_size=int(tier), kv_cache_type=kv
            )
            assert (estimate is not None) is fits, (placement["label"], tier)


def test_no_gpus_means_no_placements(config: Config) -> None:
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config, []))
    assert matrix["placements"] == []
    assert context_line(matrix) == "  (trained 256k)"


# ===========================================================================
# Rendering
# ===========================================================================


def test_context_line_reads_as_the_user_asked_for_it(config: Config) -> None:
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config))
    line = context_line(matrix)

    assert line.startswith("1x5090: ")
    assert "2x5090: 256k" in line
    assert "all: 256k" in line
    assert line.endswith("(trained 256k)")
    assert "(q8)" in line  # the single card only reaches 128k with a q8 cache


def test_context_line_marks_a_placement_the_weights_miss(config: Config) -> None:
    matrix = context_matrix(iswa_meta(), 40 * GB, planner=planner_for(config))
    assert "1x5090: --" in context_line(matrix)
    # And the tooltip says why, rather than showing four bare crosses next to a
    # weights-only badge that may still read "fits one GPU".
    assert "1x RTX 5090: weights + overhead do not fit" in context_tooltip(matrix)


def test_context_line_flags_an_approximate_matrix(config: Config) -> None:
    matrix = context_matrix(None, 21 * GB, planner=planner_for(config), unavailable="offline")
    assert context_line(matrix).endswith("approx")


def test_tooltip_lists_every_tier_per_placement(config: Config) -> None:
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config))
    tooltip = context_tooltip(matrix)

    lines = tooltip.splitlines()
    assert lines[0].startswith("1x RTX 5090: 64k OK  128k OK(q8_0)  256k x")
    assert "all 4 GPUs (2x RTX 5090 + 2x RTX 3090):" in tooltip
    assert "GiB]" in tooltip


def test_tooltip_says_why_the_numbers_are_missing(config: Config) -> None:
    matrix = context_matrix(None, 21 * GB, planner=planner_for(config), unavailable="offline")
    assert "header unavailable: offline" in context_tooltip(matrix)


def test_geometry_line_describes_the_architecture(config: Config) -> None:
    matrix = context_matrix(hybrid_meta(), 21 * GB, planner=planner_for(config))
    line = geometry_line(matrix)

    assert line.startswith("attention: hybrid · 64 layers · KV ")
    assert "KB/token (f16 @ 256k)" in line


def test_geometry_line_is_silent_when_it_would_be_a_guess(config: Config) -> None:
    matrix = context_matrix(None, 21 * GB, planner=planner_for(config), unavailable="offline")
    assert geometry_line(matrix) == ""


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "--"), (65536, "64k"), (131072, "128k"), (262144, "256k"), (1000, "1000")],
)
def test_format_ctx(value: int, expected: str) -> None:
    assert hf_meta.format_ctx(value) == expected


@pytest.mark.parametrize(
    ("name", "short", "tiny"),
    [
        ("NVIDIA GeForce RTX 5090", "RTX 5090", "5090"),
        ("NVIDIA GeForce RTX 3090", "RTX 3090", "3090"),
        ("NVIDIA RTX A6000", "RTX A6000", "A6000"),
        ("Tesla T4", "Tesla T4", "T4"),
    ],
)
def test_gpu_name_shortening(name: str, short: str, tiny: str) -> None:
    assert hf_meta.short_gpu_name(name) == short
    assert hf_meta.tiny_gpu_name(name) == tiny


# ===========================================================================
# Planner.fits_on
# ===========================================================================


def test_fits_on_returns_the_estimate_or_none(config: Config) -> None:
    planner = hf_meta.idle_planner(planner_for(config))
    record = hf_meta._throwaway_record("hf:x", hybrid_meta(), 21 * GB, 0)

    ok = planner.fits_on(record, devices=[0, 1], ctx_size=65536)
    assert ok is not None
    assert ok.weights_bytes == 21 * GB
    assert planner.fits_on(record, devices=[0], ctx_size=524288) is None


def test_fits_on_charges_a_cuda_context_per_device(config: Config) -> None:
    planner = hf_meta.idle_planner(planner_for(config))
    record = hf_meta._throwaway_record("hf:x", hybrid_meta(), 21 * GB, 0)

    one = planner.fits_on(record, devices=[0], ctx_size=32768)
    four = planner.fits_on(record, devices=[0, 1, 2, 3], ctx_size=32768)
    assert one is not None and four is not None
    assert four.cuda_context_bytes == 4 * one.cuda_context_bytes


def test_fits_on_refuses_a_device_that_is_not_there(config: Config) -> None:
    planner = hf_meta.idle_planner(planner_for(config))
    record = hf_meta._throwaway_record("hf:x", hybrid_meta(), 1 * GB, 0)
    assert planner.fits_on(record, devices=[9], ctx_size=4096) is None


def test_fits_on_honours_the_quantised_cache(config: Config) -> None:
    """The q8_0 fallback the matrix relies on has to actually change the answer."""
    planner = hf_meta.idle_planner(planner_for(config))
    record = hf_meta._throwaway_record("hf:x", hybrid_meta(), 21 * GB, 0)

    assert planner.fits_on(record, devices=[0], ctx_size=131072, kv_cache_type="f16") is None
    assert planner.fits_on(record, devices=[0], ctx_size=131072, kv_cache_type="q8_0") is not None


# ===========================================================================
# The GUI's second pass
# ===========================================================================


class FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def set_text(self, value: str) -> None:
        self.text = value


class FakeTip:
    def __init__(self) -> None:
        self.text = ""


class FakeBadge:
    """The fit badge, which the second pass rewrites with the planner's answer."""

    def __init__(self, text: str = "fits one GPU") -> None:
        self.text = text
        self.props_applied: list[str] = []

    def set_text(self, value: str) -> None:
        self.text = value

    def props(self, value: str) -> None:
        self.props_applied.append(value)


class FakeGuiContext:
    def __init__(self, config: Config, planner: Any, registry: Any = None) -> None:
        self.config = config
        self.planner = planner
        self.registry = registry


def repo_with(quants: list[tuple[str, int]]) -> Any:
    from studioforge.core.hf_search import GgufFileInfo, GgufRepoInfo

    return GgufRepoInfo(
        repo_id="acme/Hybrid-GGUF",
        publisher="acme",
        name="Hybrid-GGUF",
        downloads=1,
        likes=0,
        gated=False,
        private=False,
        last_modified=None,
        files=[
            GgufFileInfo(
                filename=f"hybrid-{quant}.gguf",
                size_bytes=size,
                quant=quant,
                is_mmproj=False,
                shard_index=None,
                shard_total=None,
                sha256=None,
                lfs_oid=None,
            )
            for quant, size in quants
        ],
    )


async def test_gui_second_pass_fills_every_row(config: Config) -> None:
    from studioforge.gui.tabs.download import _fill_context_lines

    repo = repo_with([("Q4_K_M", 17 * GB), ("Q5_K_M", 21 * GB)])
    options = repo.logical_models()
    cells = [(option, FakeLabel(), FakeTip(), FakeBadge()) for option in options]
    geometry = FakeLabel()
    ctx = FakeGuiContext(
        config, planner_for(config), FakeRegistry([sibling_record("acme", "Hybrid-GGUF")])
    )

    await _fill_context_lines(ctx, repo, cells, geometry)

    for _option, label, tip, badge in cells:
        assert "1x5090:" in label.text
        assert "trained 256k" in label.text
        assert "64k OK" in tip.text
        # The badge is replaced by the planner's own placement answer, so it
        # can no longer contradict the context line beside it.
        assert badge.text in {"fits one GPU", "needs multiple GPUs", "will not fit"}
        assert badge.props_applied and badge.props_applied[0].startswith("color=")
    assert geometry.text.startswith("attention: hybrid")


async def test_gui_second_pass_clears_the_placeholder_without_a_planner(config: Config) -> None:
    """No planner (a GUI with no gateway state) must not leave "reading..." forever."""
    from studioforge.gui.tabs.download import _fill_context_lines

    repo = repo_with([("Q4_K_M", 17 * GB)])
    cells = [(option, FakeLabel(), FakeTip(), FakeBadge()) for option in repo.logical_models()]
    for _option, label, _tip, _badge in cells:
        label.set_text("reading model header…")

    await _fill_context_lines(FakeGuiContext(config, None), repo, cells, FakeLabel())

    assert cells[0][1].text == ""
    # Nothing better to say about fit either, so the first-paint badge stands.
    assert cells[0][3].text == "fits one GPU"


async def test_gui_second_pass_says_why_when_the_header_is_gated(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs.download import _fill_context_lines

    use_transport(monkeypatch, refusing(403))
    repo = repo_with([("Q4_K_M", 17 * GB)])
    cells = [(option, FakeLabel(), FakeTip(), FakeBadge()) for option in repo.logical_models()]

    await _fill_context_lines(FakeGuiContext(config, planner_for(config)), repo, cells, FakeLabel())

    assert cells[0][1].text.endswith("approx")
    assert "hf.token" in cells[0][2].text
    # An approximate matrix is not the planner reading a header, so the badge
    # is left as the weights-only estimate rather than dressed up as better.
    assert cells[0][3].props_applied == []


# ===========================================================================
# GET /api/hf/repo/{repo_id}
# ===========================================================================


@pytest.fixture
def repo_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real app with a fake GPU inventory and a stubbed HF listing."""
    from studioforge.api.app import build_state, create_app
    from studioforge.core.hf_search import GgufFileInfo, GgufRepoInfo, HfSearch

    def make_file(name: str, size: int, quant: str) -> GgufFileInfo:
        return GgufFileInfo(
            filename=name,
            size_bytes=size,
            quant=quant,
            is_mmproj=False,
            shard_index=None,
            shard_total=None,
            sha256=None,
            lfs_oid=None,
        )

    repo = GgufRepoInfo(
        repo_id="acme/Hybrid-GGUF",
        publisher="acme",
        name="Hybrid-GGUF",
        downloads=10,
        likes=1,
        gated=False,
        private=False,
        last_modified="2026-08-01T00:00:00.000Z",
        files=[
            make_file("hybrid-Q4_K_M.gguf", 17 * GB, "Q4_K_M"),
            make_file("hybrid-Q5_K_M.gguf", 21 * GB, "Q5_K_M"),
        ],
    )

    async def fake_repo_info(self: Any, repo_id: str) -> Any:
        return repo

    monkeypatch.setattr(HfSearch, "repo_info", fake_repo_info)

    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    state = build_state(config)
    state.planner = planner_for(config)
    try:
        yield create_app(config, state=state, start_background=False)
    finally:
        state.db.close()


def test_repo_route_attaches_a_context_matrix_per_quant(
    repo_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    path = hybrid_gguf(tmp_path / "hybrid-Q4_K_M.gguf")
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    with TestClient(repo_app) as http:
        body = http.get("/api/hf/repo/acme/Hybrid-GGUF").json()

    quants = {q["quant"]: q for q in body["quants"]}
    assert set(quants) == {"Q4_K_M", "Q5_K_M"}
    # ONE header read answered for every quant in the repo.
    assert len(set(server.ranges)) >= 1
    for entry in quants.values():
        fit = entry["context_fit"]
        assert fit["source"] == "remote-gguf-header"
        assert fit["attention_kind"] == "hybrid"
        assert fit["tiers"] == [65536, 131072, 262144]
        assert [p["short_label"] for p in fit["placements"]] == ["1x5090", "2x5090", "all"]
    # The smaller quant reaches at least as far as the bigger one.
    assert (
        quants["Q4_K_M"]["context_fit"]["placements"][0]["max_ctx"]
        >= quants["Q5_K_M"]["context_fit"]["placements"][0]["max_ctx"]
    )


def test_repo_route_turns_the_fit_verdict_exact(
    repo_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header is passed to fit_verdict as arch_hint; that is what drops the caveat."""
    from fastapi.testclient import TestClient

    path = hybrid_gguf(tmp_path / "hybrid-Q4_K_M.gguf")
    use_transport(monkeypatch, RangeServer(path.read_bytes()))

    with TestClient(repo_app) as http:
        body = http.get("/api/hf/repo/acme/Hybrid-GGUF").json()

    for entry in body["quants"]:
        assert entry["fit"]["approximate"] is False
        assert "approximation until the file is present" not in entry["fit"]["message"]


def test_repo_route_still_answers_when_the_header_cannot_be_read(
    repo_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    use_transport(monkeypatch, refusing(403))

    with TestClient(repo_app) as http:
        response = http.get("/api/hf/repo/acme/Hybrid-GGUF")

    assert response.status_code == 200
    entry = response.json()["quants"][0]
    assert entry["fit"]["approximate"] is True
    assert entry["context_fit"]["approximate"] is True
    assert "hf.token" in entry["context_fit"]["unavailable"]
    assert entry["context_fit"]["placements"]  # weights_fit is still answerable


def test_search_route_refuses_a_wide_context_search(repo_app: Any) -> None:
    """Twenty header reads per search would be hundreds of MB."""
    from fastapi.testclient import TestClient

    with TestClient(repo_app) as http:
        response = http.get("/api/hf/search", params={"q": "x", "with_context": 1, "limit": 20})

    assert response.status_code == 400
    assert "with_context" in response.json()["error"]["message"]


# ---------------------------------------------------------------------------
# Regression: a sibling's tensor_bytes must not size every quant of the repo
# ---------------------------------------------------------------------------


def test_throwaway_record_pins_meta_tensor_bytes_to_this_option() -> None:
    """Live bug (2026-08-18): with a registry-sibling meta whose ``tensor_bytes``
    was the local Q5_K_S size, every quant of ``unsloth/Qwen3.8-27B-GGUF`` --
    including a 51.8 GB BF16 -- was reported fitting one 32 GiB card at 98k,
    because ``Planner.estimate`` prefers ``meta.tensor_bytes`` over
    ``size_bytes``. The throwaway record must carry THIS option's bytes in both.
    """
    meta = hybrid_meta().model_copy(update={"tensor_bytes": 17_900_000_000})
    record = hf_meta._throwaway_record("hf:repo#BF16", meta, 51_770_000_000, 0)
    assert record.size_bytes == 51_770_000_000
    assert record.meta is not None
    assert record.meta.tensor_bytes == 51_770_000_000
    # And the original meta object is untouched (it is shared across options).
    assert meta.tensor_bytes == 17_900_000_000
