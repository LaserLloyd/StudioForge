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
