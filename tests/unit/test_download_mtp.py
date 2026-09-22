"""Per-quant MTP detection for the Download tab, the REST routes and MCP.

What is being pinned down:

* **The name is a hint, the header is the verdict.** ``looks_like_mtp_name``
  is token-based and feeds ``mtp_hint``; ``remote_mtp`` reads the GGUF and
  answers ``mtp_source: "header"``. A repo *named* MTP whose header carries no
  ``nextn_predict_layers`` must come out ``mtp: False`` (LIMITATIONS.md has a
  real one), and a file whose name never says MTP but whose header does must
  come out ``True`` (unsloth's Qwen3.8 quants).
* **A probe is one small range request.** The walk stops at the tokenizer, so
  the served byte count is a chunk, not the header -- the whole reason a
  22-quant repo can be probed per file at all.
* **Concurrency is bounded and failures are per quant.**
* **The wire shape** on ``GET /api/hf/repo``, ``repo_details`` and the search
  rows, and the GUI's badge derivation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from studioforge.config import Config
from studioforge.core import gguf as gguf_mod
from studioforge.core import hf_meta
from studioforge.core.gguf import read_meta
from studioforge.core.hf_meta import (
    MtpStatus,
    RemoteHeaderError,
    mtp_from_kv,
    quant_mtp_status,
    registry_file_meta,
    remote_meta,
    remote_mtp,
    repo_mtp_status,
)
from studioforge.core.hf_search import (
    GgufFileInfo,
    GgufRepoInfo,
    LogicalDownload,
    looks_like_mtp_name,
)
from studioforge.types import GB, GgufMeta, ModelRecord
from tests.unit.test_gguf import ARRAY, STRING, UINT32, Arr, KvEntry, write_gguf
from tests.unit.test_hf_meta import RangeServer, refusing, use_transport

# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


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
    hf_meta.clear_memory_cache()
    yield
    hf_meta.clear_memory_cache()


def gguf_file(name: str, size: int = 10 * GB, *, quant: str = "Q4_K_M") -> GgufFileInfo:
    from studioforge.core.gguf import looks_like_auxiliary_gguf, looks_like_mmproj
    from studioforge.core.hf_search import parse_quant, shard_parts

    index, total = shard_parts(name)
    return GgufFileInfo(
        filename=name,
        size_bytes=size,
        quant=parse_quant(name) if parse_quant(name) != "unknown" else quant,
        is_mmproj=looks_like_mmproj(Path(name)),
        shard_index=index,
        shard_total=total,
        sha256=None,
        lfs_oid=None,
        is_auxiliary=looks_like_auxiliary_gguf(name, size_bytes=size),
    )


def repo(repo_id: str, files: list[GgufFileInfo]) -> GgufRepoInfo:
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


def mtp_gguf(
    path: Path,
    *,
    nextn: int | None = 1,
    vocab: int = 50_000,
    tokenizer_first: bool = False,
) -> Path:
    """A qwen35-shaped header with a vocabulary big enough to matter.

    50k entries of ``tok`` are ~550 KB of length-prefixed strings after the
    architecture block, which is what a probe must NOT fetch. ``nextn=None``
    leaves the key out entirely, the way a stripped variant is written.
    ``tokenizer_first`` is the pathological writer order the probe must
    refuse to guess about.
    """
    arch: list[KvEntry] = [
        ("qwen35.block_count", UINT32, 65),
        ("qwen35.context_length", UINT32, 262144),
        ("qwen35.embedding_length", UINT32, 5120),
        ("qwen35.attention.head_count", UINT32, 24),
        ("qwen35.attention.head_count_kv", UINT32, 4),
        ("qwen35.attention.key_length", UINT32, 256),
        ("qwen35.attention.value_length", UINT32, 256),
        ("qwen35.full_attention_interval", UINT32, 4),
    ]
    if nextn is not None:
        arch.append(("qwen35.nextn_predict_layers", UINT32, nextn))
    tokenizer: list[KvEntry] = [
        ("tokenizer.ggml.model", STRING, "gpt2"),
        ("tokenizer.ggml.tokens", ARRAY, Arr(STRING, ["tok"] * vocab)),
        ("general.file_type", UINT32, 15),  # llama-quantize appends this at the END
    ]
    head: list[KvEntry] = [("general.architecture", STRING, "qwen35")]
    kv = head + (tokenizer + arch if tokenizer_first else arch + tokenizer)
    return write_gguf(path, kv, [("blk.0.attn_q.weight", (256, 256), 8)])


# ===========================================================================
# The name hint
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "want"),
    [
        ("Qwen3.8-27B-Ultra-Uncensored-Heretic-Native-MTP-Preserved-NVFP4-GGUF", True),
        ("Qwen3.8-27B-NVFP4-MTP-Q6_K.gguf", True),
        ("Gemma4-31B-QAT-Uncensored-Balanced-MTP-Q4_K_M.gguf", True),
        ("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", True),
        ("some_model_mtp.gguf", True),
        ("Qwen3.8-27B-GGUF", False),
        ("Qwen3.8-27B-UD-Q4_K_M.gguf", False),
        ("smtp-relay-model-Q4_K_M.gguf", False),  # substring, not a token
        ("", False),
    ],
)
def test_looks_like_mtp_name_matches_tokens_not_substrings(name: str, want: bool) -> None:
    assert looks_like_mtp_name(name) is want


def test_a_repo_named_mtp_hints_every_quant_when_no_file_says_it() -> None:
    """The llmfan46 case: the repo name carries it, the file names never do."""
    info = repo(
        "llmfan46/Qwen3.8-27B-Native-MTP-Preserved-NVFP4-GGUF",
        [
            gguf_file("Qwen3.8-27B-Heretic-NVFP4-BF16.gguf", 18 * GB),
            gguf_file("Qwen3.8-27B-Heretic-NVFP4-Q8_0.gguf", 15 * GB),
            gguf_file("Qwen3.8-27B-Heretic-mmproj-BF16.gguf", 1 * GB),
        ],
    )
    assert info.mtp_hint is True
    options = info.logical_models()
    assert len(options) == 2
    assert all(o.mtp_hint for o in options)
    # Both parse to the same label and are told apart by their base name.
    assert {o.quant for o in options} == {"NVFP4"}
    assert all(o.discriminator for o in options)


def test_a_file_that_says_mtp_hints_only_itself() -> None:
    """Same quant twice, one with the heads: the unmarked file is the stripped one."""
    info = repo(
        "acme/Foo-27B-GGUF",
        [
            gguf_file("Foo-27B-MTP-Q4_K_M.gguf", 19 * GB),
            gguf_file("Foo-27B-Q4_K_M.gguf", 18 * GB),
        ],
    )
    assert info.mtp_hint is True
    by_file = {o.files[0].filename: o for o in info.logical_models()}
    assert by_file["Foo-27B-MTP-Q4_K_M.gguf"].mtp_hint is True
    assert by_file["Foo-27B-Q4_K_M.gguf"].mtp_hint is False


def test_a_draft_module_directory_is_not_a_hint() -> None:
    """unsloth's ``MTP/`` folder says nothing about the quants beside it."""
    info = repo(
        "unsloth/Qwen3.8-27B-GGUF",
        [
            gguf_file("MTP/mtp-Qwen3.8-27B-Q4_0.gguf", 1_300_000_000),
            gguf_file("Qwen3.8-27B-UD-Q4_K_M.gguf", 16 * GB),
        ],
    )
    assert info.mtp_hint is False
    options = info.logical_models()
    assert [o.files[0].filename for o in options] == ["Qwen3.8-27B-UD-Q4_K_M.gguf"]
    assert options[0].mtp_hint is False


def test_the_owner_handle_is_not_consulted() -> None:
    info = repo("mtp-labs/Plain-7B-GGUF", [gguf_file("Plain-7B-Q4_K_M.gguf")])
    assert info.mtp_hint is False


def test_a_subfolder_file_is_flagged_rather_than_refused_on_click() -> None:
    option = repo(
        "unsloth/Qwen3.8-27B-GGUF",
        [
            gguf_file("BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf", 46 * GB),
            gguf_file("BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf", 4 * GB),
        ],
    ).logical_models()[0]
    assert option.in_subfolder is True
    assert repo("a/b", [gguf_file("b-Q4_K_M.gguf")]).logical_models()[0].in_subfolder is False


def test_logical_download_defaults_keep_existing_constructions_working() -> None:
    option = LogicalDownload(
        repo_id="a/b",
        quant="Q4_K_M",
        files=[gguf_file("b-Q4_K_M.gguf")],
        mmproj=None,
        total_bytes=1,
    )
    assert option.mtp_hint is False


# ===========================================================================
# The early-stop walk
# ===========================================================================


def test_read_stream_stops_before_the_named_key(tmp_path: Path) -> None:
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", vocab=10)
    with path.open("rb") as fh:
        partial = gguf_mod._read_stream(
            fh,
            path,
            load_tensors=True,  # ignored: the table is unreachable past a stop
            max_array_len=64,
            stop_before=lambda key, _kv: key.startswith("tokenizer."),
        )
    assert partial.kv_complete is False
    assert partial.kv["qwen35.nextn_predict_layers"] == 1
    assert "tokenizer.ggml.tokens" not in partial.kv
    assert "general.file_type" not in partial.kv  # after the stop, never read
    assert partial.tensors == []

    with path.open("rb") as fh:
        full = gguf_mod._read_stream(fh, path, load_tensors=True, max_array_len=64)
    assert full.kv_complete is True
    assert full.kv["general.file_type"] == 15
    assert len(full.tensors) == 1


def test_the_full_parser_still_reports_the_heads(tmp_path: Path) -> None:
    """``read_meta`` keeps the key in ``extra``; ``mtp_from_meta`` reads it there."""
    meta = read_meta(mtp_gguf(tmp_path / "m-Q4_K_M.gguf", nextn=2, vocab=10))
    status = hf_meta.mtp_from_meta(meta)
    assert status == MtpStatus(mtp=True, source="header", layers=2)
    stripped = read_meta(mtp_gguf(tmp_path / "s-Q4_K_M.gguf", nextn=None, vocab=10))
    assert hf_meta.mtp_from_meta(stripped) == MtpStatus(mtp=False, source="header")


@pytest.mark.parametrize(
    ("kv", "complete", "want"),
    [
        ({"general.architecture": "qwen35", "qwen35.nextn_predict_layers": 1}, False, (True, 1)),
        (
            {"general.architecture": "qwen35", "qwen35.nextn_predict_layers": 0},
            False,
            (False, None),
        ),
        ({"general.architecture": "llama", "llama.block_count": 32}, False, (False, None)),
        ({"general.architecture": "llama", "llama.block_count": 32}, True, (False, None)),
        ({"general.architecture": "llama"}, True, (False, None)),
    ],
)
def test_mtp_from_kv_settles_when_it_can(
    kv: dict[str, Any], complete: bool, want: tuple[bool, int | None]
) -> None:
    status = mtp_from_kv(kv, complete=complete)
    assert (status.mtp, status.layers) == want
    assert status.source == "header"


def test_mtp_from_kv_refuses_to_guess_from_a_walk_that_saw_no_architecture() -> None:
    """Stopped before the block: nothing proves the key is absent."""
    stopped_early = mtp_from_kv({"general.architecture": "qwen35"}, complete=False)
    assert stopped_early.mtp is None
    assert stopped_early.source is None
    assert "before the architecture block" in (stopped_early.detail or "")
    assert mtp_from_kv({}, complete=True).mtp is None


# ===========================================================================
# remote_mtp: one small request, cached
# ===========================================================================


async def test_remote_mtp_confirms_the_heads_without_walking_the_tokenizer(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", nextn=1)
    data = path.read_bytes()
    server = RangeServer(data)
    use_transport(monkeypatch, server)

    status = await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf", chunk_bytes=64 * 1024)

    assert status == MtpStatus(mtp=True, source="header", layers=1)
    assert len(data) > 500_000
    # ONE chunk, and nothing of the tokenizer: the whole point of the probe.
    assert server.ranges == ["bytes=0-65535"]
    assert server.served_bytes == 64 * 1024


async def test_remote_mtp_denies_the_heads_when_the_header_lacks_the_key(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repo NAMED MTP with no such key in the file comes out False, by header."""
    path = mtp_gguf(tmp_path / "m-MTP-Q4_K_M.gguf", nextn=None)
    use_transport(monkeypatch, RangeServer(path.read_bytes()))

    status = await remote_mtp(config, "acme/M-MTP-GGUF", "m-MTP-Q4_K_M.gguf")

    assert status.mtp is False
    assert status.source == "header"
    assert status.layers is None


async def test_remote_mtp_is_inconclusive_when_the_tokenizer_comes_first(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "odd-Q4_K_M.gguf", nextn=1, tokenizer_first=True)
    use_transport(monkeypatch, RangeServer(path.read_bytes()))

    status = await remote_mtp(config, "acme/Odd-GGUF", "odd-Q4_K_M.gguf")

    assert status.mtp is None
    assert status.source is None
    # And an inconclusive answer is not cached: nothing to trust for a day.
    assert not list(hf_meta.cache_dir(config).glob("*.json"))


async def test_remote_mtp_is_cached_in_memory_and_on_disk(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", nextn=1)
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    first = await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf")
    after_first = len(server.ranges)
    assert await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf") == first
    assert len(server.ranges) == after_first  # memory hit

    hf_meta.clear_memory_cache()
    assert await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf") == first
    assert len(server.ranges) == after_first  # disk hit

    cached = list(hf_meta.cache_dir(config).glob("*.json"))
    assert len(cached) == 1
    payload = json.loads(cached[0].read_text())
    assert payload["kind"] == "mtp"  # distinguishable from a full-header entry
    assert payload["mtp"] is True
    assert payload["layers"] == 1
    assert payload["key"].startswith("acme/M-GGUF")


async def test_a_full_header_already_read_answers_mtp_for_free(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The context-fit read of the smallest quant leaves its header cached."""
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", nextn=1)
    server = RangeServer(path.read_bytes())
    use_transport(monkeypatch, server)

    await remote_meta(config, "acme/M-GGUF", "m-Q4_K_M.gguf")
    before = len(server.ranges)

    status = await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf")

    assert status == MtpStatus(mtp=True, source="header", layers=1)
    assert len(server.ranges) == before  # not one more request


async def test_remote_mtp_reports_a_gated_repo(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_transport(monkeypatch, refusing(403))
    with pytest.raises(RemoteHeaderError, match="hf.token"):
        await remote_mtp(config, "acme/Gated-GGUF", "m-Q4_K_M.gguf")


async def test_a_truncated_header_is_an_error_not_a_verdict(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", nextn=1, vocab=10)
    use_transport(monkeypatch, RangeServer(path.read_bytes()[:40]))
    with pytest.raises(RemoteHeaderError, match="not readable as GGUF"):
        await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf")


async def test_remote_mtp_sends_the_token_as_a_header(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "m-Q4_K_M.gguf", vocab=10)
    server = RangeServer(path.read_bytes())
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return server(request)

    use_transport(monkeypatch, handler)
    config.hf.token = "hf_secret"

    await remote_mtp(config, "acme/M-GGUF", "m-Q4_K_M.gguf")

    assert seen[0].headers["Authorization"] == "Bearer hf_secret"
    assert "hf_secret" not in str(seen[0].url)


# ===========================================================================
# quant_mtp_status / repo_mtp_status
# ===========================================================================


class FakeRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = records

    def all(self) -> list[ModelRecord]:
        return list(self._records)


def local_record(publisher: str, repo_name: str, filename: str, *, nextn: int) -> ModelRecord:
    return ModelRecord(
        id=f"{publisher}/{repo_name}/{filename}",
        name=filename,
        path=Path(f"/models/{publisher}/{repo_name}/{filename}"),
        publisher=publisher,
        repo=repo_name,
        meta=GgufMeta(architecture="qwen35", n_layer=65, extra={"nextn_predict_layers": nextn}),
    )


def test_registry_file_meta_matches_the_exact_file_only() -> None:
    registry = FakeRegistry([local_record("acme", "M-GGUF", "m-Q4_K_M.gguf", nextn=1)])
    assert registry_file_meta(registry, "ACME/m-gguf", "M-Q4_K_M.GGUF") is not None
    assert registry_file_meta(registry, "acme/M-GGUF", "m-Q8_0.gguf") is None  # a sibling
    assert registry_file_meta(registry, "acme/Other-GGUF", "m-Q4_K_M.gguf") is None
    assert registry_file_meta(None, "acme/M-GGUF", "m-Q4_K_M.gguf") is None


async def test_a_downloaded_copy_answers_without_the_network(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = RangeServer(b"")
    use_transport(monkeypatch, server)
    option = repo("acme/M-GGUF", [gguf_file("m-Q4_K_M.gguf")]).logical_models()[0]
    registry = FakeRegistry([local_record("acme", "M-GGUF", "m-Q4_K_M.gguf", nextn=1)])

    status = await quant_mtp_status(config, option, registry=registry)

    assert status == MtpStatus(mtp=True, source="header", layers=1)
    assert server.ranges == []


async def test_shard_one_is_the_file_that_is_probed(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = mtp_gguf(tmp_path / "m-Q8_0-00001-of-00002.gguf", nextn=1, vocab=10)
    seen: list[str] = []
    server = RangeServer(path.read_bytes())

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return server(request)

    use_transport(monkeypatch, handler)
    option = repo(
        "acme/M-GGUF",
        [
            gguf_file("m-Q8_0-00002-of-00002.gguf", 4 * GB),
            gguf_file("m-Q8_0-00001-of-00002.gguf", 20 * GB),
        ],
    ).logical_models()[0]

    status = await quant_mtp_status(config, option)

    assert status.mtp is True
    assert len(seen) == 1
    assert seen[0].endswith("m-Q8_0-00001-of-00002.gguf")


async def test_an_unreadable_header_falls_back_to_the_name_with_the_reason(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_transport(monkeypatch, refusing(403))
    named = repo("acme/M-MTP-GGUF", [gguf_file("m-Q4_K_M.gguf")]).logical_models()[0]
    plain = repo("acme/M-GGUF", [gguf_file("m-Q4_K_M.gguf")]).logical_models()[0]

    likely = await quant_mtp_status(config, named)
    unknown = await quant_mtp_status(config, plain)

    assert (likely.mtp, likely.source) == (True, "name")
    assert "hf.token" in (likely.detail or "")
    assert (unknown.mtp, unknown.source) == (None, None)
    assert "hf.token" in (unknown.detail or "")


async def test_repo_mtp_status_bounds_concurrency_and_reports_per_row(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
    files = [gguf_file(f"m-{label}.gguf", (n + 2) * GB) for n, label in enumerate(labels)]
    options = repo("acme/M-GGUF", files).logical_models()
    assert [o.quant for o in options] == sorted(labels)
    in_flight = 0
    peak = 0

    async def fake_quant(_config: Config, option: Any, *, registry: Any = None) -> MtpStatus:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        if option.quant == "Q3_K_M":
            raise RuntimeError("boom")  # must not take the others down
        return MtpStatus(mtp=option.quant == "Q4_K_M", source="header")

    monkeypatch.setattr(hf_meta, "quant_mtp_status", fake_quant)
    landed: list[str] = []

    def on_result(option: Any, status: MtpStatus) -> None:
        landed.append(option.quant)
        if option.quant == "Q2_K_M":
            raise ValueError("a dead row")  # swallowed, logged

    results = await repo_mtp_status(config, options, concurrency=2, on_result=on_result)

    assert peak <= 2
    assert set(results) == {o.group_id for o in options}
    assert sum(1 for s in results.values() if s.mtp) == 1
    # The row whose probe blew up is unknown, with the reason; the rest landed.
    broken = next(s for gid, s in results.items() if gid.endswith("q3-k-m"))
    assert broken.mtp is None
    assert "boom" in (broken.detail or "")
    assert sorted(landed) == sorted(o.quant for o in options)


# ===========================================================================
# The wire: GET /api/hf/repo, GET /api/hf/search, MCP rows
# ===========================================================================


class MultiFileServer:
    """One RangeServer per file, dispatched on the URL's basename."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.servers = {name: RangeServer(data) for name, data in files.items()}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        server = self.servers.get(name)
        if server is None:
            return httpx.Response(404, json={"error": f"no such file {name}"})
        return server(request)

    def ranges(self, name: str) -> list[str | None]:
        return self.servers[name].ranges


@pytest.fixture
def mtp_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real app, a fake 4-GPU rig, and a stubbed HF listing of a mixed repo.

    ``Hybrid-MTP-GGUF`` (the name says MTP) ships a Q4_K_M whose header
    carries the heads and a Q5_K_M whose header does not.
    """
    from studioforge.api.app import build_state, create_app
    from studioforge.core.hf_search import HfSearch
    from tests.unit.test_hf_meta import planner_for

    info = repo(
        "acme/Hybrid-MTP-GGUF",
        [gguf_file("hybrid-Q4_K_M.gguf", 17 * GB), gguf_file("hybrid-Q5_K_M.gguf", 21 * GB)],
    )

    async def fake_repo_info(self: Any, repo_id: str) -> Any:
        return info

    async def fake_search(self: Any, query: str, **_kwargs: Any) -> Any:
        return [info]

    monkeypatch.setattr(HfSearch, "repo_info", fake_repo_info)
    monkeypatch.setattr(HfSearch, "search", fake_search)

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


def two_headers(tmp_path: Path) -> MultiFileServer:
    return MultiFileServer(
        {
            "hybrid-Q4_K_M.gguf": mtp_gguf(tmp_path / "q4.gguf", nextn=1).read_bytes(),
            "hybrid-Q5_K_M.gguf": mtp_gguf(tmp_path / "q5.gguf", nextn=None).read_bytes(),
        }
    )


def test_repo_route_settles_mtp_per_quant_from_each_header(
    mtp_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    server = two_headers(tmp_path)
    use_transport(monkeypatch, server)

    with TestClient(mtp_app) as http:
        body = http.get("/api/hf/repo/acme/Hybrid-MTP-GGUF").json()

    assert body["mtp_likely"] is True  # the name
    quants = {q["quant"]: q for q in body["quants"]}
    assert (quants["Q4_K_M"]["mtp"], quants["Q4_K_M"]["mtp_source"]) == (True, "header")
    assert quants["Q4_K_M"]["mtp_layers"] == 1
    # Named MTP, header says otherwise: the header wins, and says so.
    assert (quants["Q5_K_M"]["mtp"], quants["Q5_K_M"]["mtp_source"]) == (False, "header")
    assert quants["Q5_K_M"]["mtp_layers"] is None
    # The smallest quant's full header (the context read) answered its probe:
    # the file was fetched exactly once. The other file cost one small probe.
    assert len(server.ranges("hybrid-Q4_K_M.gguf")) == 1
    assert server.ranges("hybrid-Q5_K_M.gguf") == ["bytes=0-262143"]
    assert quants["Q4_K_M"]["context_fit"]["source"] == "remote-gguf-header"


def test_search_route_carries_the_name_hint_and_reads_no_header(
    mtp_app: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    server = two_headers(tmp_path)
    use_transport(monkeypatch, server)

    with TestClient(mtp_app) as http:
        body = http.get("/api/hf/search", params={"q": "hybrid"}).json()

    row = body["repos"][0]
    assert row["mtp_likely"] is True
    for entry in row["quants"]:
        assert entry["mtp"] is True
        assert entry["mtp_source"] == "name"
        assert entry["mtp_layers"] is None
    assert all(not s.ranges for s in server.servers.values())


def test_repo_route_keeps_the_name_hint_when_the_header_is_refused(
    mtp_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    use_transport(monkeypatch, refusing(403))

    with TestClient(mtp_app) as http:
        body = http.get("/api/hf/repo/acme/Hybrid-MTP-GGUF").json()

    for entry in body["quants"]:
        assert (entry["mtp"], entry["mtp_source"]) == (True, "name")
        assert "hf.token" in entry["context_fit"]["unavailable"]


def test_mcp_repo_details_keeps_the_mtp_fields_compact() -> None:
    from studioforge.mcp.management import _compact_repo

    payload = {
        "repo_id": "acme/Hybrid-MTP-GGUF",
        "quants": [
            {
                "quant": "Q4_K_M",
                "total_bytes": 17 * GB,
                "files": ["hybrid-Q4_K_M.gguf"],
                "mmproj": None,
                "group_id": "g",
                "fit": {"verdict": "fits-one-gpu", "message": "", "approximate": False},
                "mtp": True,
                "mtp_source": "header",
                "mtp_layers": 1,
            },
            {
                "quant": "Q5_K_M",
                "total_bytes": 21 * GB,
                "files": ["hybrid-Q5_K_M.gguf"],
                "mmproj": None,
                "group_id": "h",
                "fit": {},
                "mtp": False,
                "mtp_source": "header",
                "mtp_layers": None,
            },
            {"quant": "Q8_0", "total_bytes": 0, "files": [], "mmproj": None, "fit": {}},
        ],
    }
    compact = {q["quant"]: q for q in _compact_repo(payload)["quants"]}
    assert (compact["Q4_K_M"]["mtp"], compact["Q4_K_M"]["mtp_source"]) == (True, "header")
    assert compact["Q4_K_M"]["mtp_layers"] == 1
    assert (compact["Q5_K_M"]["mtp"], compact["Q5_K_M"]["mtp_source"]) == (False, "header")
    assert "mtp_layers" not in compact["Q5_K_M"]  # only when there is a count
    # An older payload without the fields still compacts, as unknown.
    assert (compact["Q8_0"]["mtp"], compact["Q8_0"]["mtp_source"]) == (None, None)


def test_mcp_search_row_carries_the_name_hint_only() -> None:
    from studioforge.mcp.management import _search_row

    named = repo("acme/Foo-MTP-GGUF", [gguf_file("foo-Q4_K_M.gguf")])
    plain = repo("acme/Foo-GGUF", [gguf_file("foo-Q4_K_M.gguf")])
    assert _search_row(named, trending=False)["mtp_likely"] is True
    assert _search_row(plain, trending=False)["mtp_likely"] is False
    assert "mtp_source" not in _search_row(named, trending=False)


# ===========================================================================
# GUI helpers (pure)
# ===========================================================================


@pytest.mark.parametrize(
    ("label", "bits"),
    [
        ("Q4_K_M", 4),
        ("IQ3_XXS", 3),
        ("Q2_K_L", 2),
        ("TQ1_0", 1),
        ("NVFP4", 4),
        ("MXFP4", 4),
        ("Q8_0", 8),
        ("F16", 16),
        ("BF16", 16),
        ("F32", 32),
        ("unknown", None),
        ("", None),
        (None, None),
    ],
)
def test_quant_bits(label: str | None, bits: int | None) -> None:
    from studioforge.gui import state as st

    assert st.quant_bits(label) == bits
    assert st.quant_bit_group(label) == ("other" if bits is None else f"{bits}-bit")


def test_quant_bit_groups_are_shelves_smallest_first() -> None:
    from studioforge.gui import state as st

    info = repo(
        "acme/M-GGUF",
        [
            gguf_file("m-BF16.gguf", 52 * GB),
            gguf_file("m-Q4_K_M.gguf", 16 * GB),
            gguf_file("m-IQ4_XS.gguf", 14 * GB),
            gguf_file("m-Q8_0.gguf", 28 * GB),
            gguf_file("m-IQ2_M.gguf", 9 * GB),
            gguf_file("m-weird.gguf", 1 * GB, quant="unknown"),
        ],
    )
    shelves = st.quant_bit_groups(info.logical_models())

    assert [shelf for shelf, _ in shelves] == ["2-bit", "4-bit", "8-bit", "16-bit", "other"]
    four_bit = [o.quant for o in dict(shelves)["4-bit"]]
    assert four_bit == ["IQ4_XS", "Q4_K_M"]  # 14 GB before 16 GB, not alphabetical


def test_quant_note_and_files_tooltip_describe_the_whole_download() -> None:
    from studioforge.gui import state as st

    info = repo(
        "acme/Vision-GGUF",
        [
            gguf_file("v-Q8_0-00001-of-00002.gguf", 20 * GB),
            gguf_file("v-Q8_0-00002-of-00002.gguf", 8 * GB),
            gguf_file("mmproj-F16.gguf", 1 * GB),
        ],
    )
    option = info.logical_models()[0]
    assert st.quant_note(option) == "2 parts · +mmproj"
    tooltip = st.quant_files_tooltip(option)
    assert tooltip.splitlines() == [
        option.label,
        "v-Q8_0-00001-of-00002.gguf",
        "v-Q8_0-00002-of-00002.gguf",
        "mmproj-F16.gguf",
    ]
    plain = repo("acme/P-GGUF", [gguf_file("p-Q4_K_M.gguf")]).logical_models()[0]
    assert st.quant_note(plain) == ""


def test_rig_summary_names_the_shelf_the_fit_column_is_measured_against() -> None:
    from studioforge.gui import state as st
    from tests.unit.test_hf_meta import rig_4

    assert st.rig_summary(rig_4()) == "Rig: 2× RTX 5090 (32 GiB) + 2× RTX 3090 (24 GiB)"
    assert st.rig_summary([]) == ""


def test_mtp_badge_states() -> None:
    from studioforge.gui import state as st

    # First paint: only the name is known.
    likely = st.mtp_badge(None, name_hint=True)
    assert likely is not None
    assert (likely.text, likely.colour, likely.outline) == ("likely MTP", "info", True)
    assert "not been read yet" in likely.tooltip
    assert st.mtp_badge(None, name_hint=False) is None

    confirmed = st.mtp_badge(MtpStatus(mtp=True, source="header", layers=1), name_hint=False)
    assert confirmed is not None
    assert (confirmed.text, confirmed.outline) == ("MTP", False)
    assert "1 head" in confirmed.tooltip and "draft-mtp" in confirmed.tooltip

    two = st.mtp_badge(MtpStatus(mtp=True, source="header", layers=2), name_hint=True)
    assert two is not None and "2 heads" in two.tooltip

    # The name promised, the header did not deliver: said, in grey.
    denied = st.mtp_badge(MtpStatus(mtp=False, source="header"), name_hint=True)
    assert denied is not None
    assert (denied.text, denied.colour) == ("no MTP heads", "grey")
    # Nothing promised, nothing there: no badge at all.
    assert st.mtp_badge(MtpStatus(mtp=False, source="header"), name_hint=False) is None

    # Header unreadable, name says MTP: still "likely", with the reason.
    fallback = st.mtp_badge(
        MtpStatus(mtp=True, source="name", detail="HTTP 403: set hf.token"), name_hint=True
    )
    assert fallback is not None
    assert fallback.text == "likely MTP" and "hf.token" in fallback.tooltip
    assert st.mtp_badge(MtpStatus(detail="offline"), name_hint=False) is None


# ===========================================================================
# GUI: the badge pass and the rendered picker
# ===========================================================================


class FakeBadge:
    def __init__(self) -> None:
        self.text = ""
        self.visible = True
        self.props_added: list[str] = []
        self.props_removed: list[str] = []

    def set_text(self, value: str) -> None:
        self.text = value

    def props(self, add: str | None = None, *, remove: str | None = None) -> None:
        if add:
            self.props_added.append(add)
        if remove:
            self.props_removed.append(remove)

    def set_visibility(self, visible: bool) -> None:
        self.visible = visible


class FakeTip:
    def __init__(self) -> None:
        self.text = ""


class FakeGuiContext:
    def __init__(self, config: Config, registry: Any = None) -> None:
        self.config = config
        self.registry = registry


async def test_gui_mtp_pass_paints_each_row_from_its_own_header(
    tmp_path: Path, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs.download import _fill_mtp_badges

    use_transport(
        monkeypatch,
        MultiFileServer(
            {
                "h-Q4_K_M.gguf": mtp_gguf(tmp_path / "q4.gguf", nextn=1).read_bytes(),
                "h-MTP-Q5_K_M.gguf": mtp_gguf(tmp_path / "q5.gguf", nextn=None).read_bytes(),
                "h-Q6_K.gguf": mtp_gguf(tmp_path / "q6.gguf", nextn=None).read_bytes(),
            }
        ),
    )
    info = repo(
        "acme/H-GGUF",
        [
            gguf_file("h-Q4_K_M.gguf", 16 * GB),
            gguf_file("h-MTP-Q5_K_M.gguf", 19 * GB),
            gguf_file("h-Q6_K.gguf", 22 * GB),
        ],
    )
    cells = [(option, FakeBadge(), FakeTip()) for option in info.logical_models()]

    await _fill_mtp_badges(FakeGuiContext(config), cells)

    by_quant = {option.quant: (badge, tip) for option, badge, tip in cells}
    # No hint in the name, heads in the header: confirmed, filled.
    badge, tip = by_quant["Q4_K_M"]
    assert (badge.text, badge.visible) == ("MTP", True)
    assert "color=info" in badge.props_added and "outline" in badge.props_removed
    assert "Confirmed from the GGUF header" in tip.text
    # Named MTP, no heads: the grey correction.
    badge, _tip = by_quant["Q5_K_M"]
    assert (badge.text, badge.visible) == ("no MTP heads", True)
    assert "color=grey" in badge.props_added
    # Neither: hidden.
    assert by_quant["Q6_K"][0].visible is False


async def test_gui_mtp_pass_keeps_likely_when_headers_cannot_be_read(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    from studioforge.gui.tabs.download import _fill_mtp_badges

    use_transport(monkeypatch, refusing(403))
    info = repo("acme/H-MTP-GGUF", [gguf_file("h-Q4_K_M.gguf", 16 * GB)])
    cells = [(option, FakeBadge(), FakeTip()) for option in info.logical_models()]

    await _fill_mtp_badges(FakeGuiContext(config), cells)

    badge, tip = cells[0][1], cells[0][2]
    assert badge.text == "likely MTP"
    assert "outline" in badge.props_added
    assert "hf.token" in tip.text


def test_quant_picker_renders_shelves_badges_and_the_rig_line(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One real NiceGUI render of the grouped picker, first paint only."""
    import shutil

    from fastapi.testclient import TestClient
    from nicegui import ui

    from studioforge.core import diskspace
    from studioforge.gui.app import create_gui_app
    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab
    from tests.unit.test_gui import _FakeOption, _FakeRepoFiles, _FakeState

    diskspace.clear_cache()
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: (4 * 1024**4, 0, 400 * GB))
    state = _FakeState(config)
    ctx = GuiContext(config=config, api_state=state)
    full = _FakeRepoFiles(
        [
            _FakeOption("Q4_K_M", 12 * GB, mtp_hint=True),
            _FakeOption("Q8_0", 30 * GB, in_subfolder=True),
            _FakeOption("F16", 60 * GB, discriminator="other-base"),
        ]
    )

    @ui.page("/_mtp_picker_smoke")
    def _page() -> None:
        tab._quant_rows(ctx, full, highlight=None, on_picked=None)

    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/_mtp_picker_smoke")
    diskspace.clear_cache()

    assert response.status_code == 200
    text = response.text
    for shelf in ("4-bit", "8-bit", "16-bit"):
        assert shelf in text
    assert "Rig: 1× RTX 5090 (32 GiB)" in text
    assert "likely MTP" in text  # the name hint, at first paint
    assert text.count("model-F16.gguf") >= 2  # inline (duplicate label) and in the hover
    assert "subfolder of the repository" in text  # the disabled button's reason
    assert "fits one GPU" in text


def test_download_tab_offers_the_mtp_filter(config: Config) -> None:
    from fastapi.testclient import TestClient

    from studioforge.gui.app import create_gui_app
    from tests.unit.test_gui import _FakeState

    app = create_gui_app(config, api_state=_FakeState(config))
    with TestClient(app) as client:
        response = client.get("/?tab=download")
    assert response.status_code == 200
    assert "MTP only" in response.text


def test_search_row_chip_is_filled_only_for_a_downloaded_quant_with_heads(
    config: Config,
) -> None:
    from fastapi.testclient import TestClient
    from nicegui import ui

    from studioforge.gui.app import create_gui_app
    from studioforge.gui.tabs import GuiContext
    from studioforge.gui.tabs import download as tab
    from tests.unit.test_gui import _FakeRegistry, _FakeState

    known = repo("acme/Known-GGUF", [gguf_file("known-Q4_K_M.gguf")])
    named = repo("acme/Named-MTP-GGUF", [gguf_file("named-Q4_K_M.gguf")])
    plain = repo("acme/Plain-GGUF", [gguf_file("plain-Q4_K_M.gguf")])
    state = _FakeState(config)
    state.registry = _FakeRegistry([local_record("acme", "Known-GGUF", "known-Q8_0.gguf", nextn=1)])
    ctx = GuiContext(config=config, api_state=state)

    statuses = {r.repo_id: tab._repo_mtp(ctx, r) for r in (known, named, plain)}
    assert statuses["acme/Known-GGUF"].source == "header"
    assert statuses["acme/Named-MTP-GGUF"].source == "name"
    assert statuses["acme/Plain-GGUF"].mtp is None

    @ui.page("/_mtp_rows_smoke")
    def _page() -> None:
        for info in (known, named, plain):
            tab._repo_row(ctx, info, mtp=statuses[info.repo_id])

    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/_mtp_rows_smoke")
    assert response.status_code == 200
    assert response.text.count("likely MTP") == 1
    assert "Confirmed from the GGUF header" in response.text
