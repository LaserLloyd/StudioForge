"""Unit tests for the management-plane MCP server.

The server is exercised through the real :class:`MCPServer` object -- tools are
invoked with ``call_tool`` exactly as a client would, so the JSON schema
generation, argument coercion and result serialisation are all in the loop. What
is faked is only the *hardware*: a :class:`FakeGpuProbe` instead of NVML and a
stub engine manager, because none of the assertions here are about GPUs.

The registry, planner, supervisor and model manager are the real ones, over a
small synthetic model directory in ``tmp_path`` with an injected GGUF metadata
reader. That matters for the compactness assertions: a real ``ModelRecord``
carries a real ``chat_template``, so "the list output does not contain the
template" is only a meaningful claim when the template is genuinely there to
leak.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core.gpu import FakeGpuProbe
from studioforge.core.manager import ModelManager
from studioforge.core.planner import Planner
from studioforge.core.registry import Registry
from studioforge.core.supervisor import Supervisor
from studioforge.db import Database
from studioforge.mcp.management import (
    build_management_mcp,
    mount_management_mcp,
)
from studioforge.types import EngineInfo, GgufMeta, GpuInfo, ModelSettings

GIB = 1024**3

#: The exact tool surface. Pinned as a set *and* asserted for absence of
#: inference verbs: adding a chat tool here would be an architectural
#: regression, not a feature, so it has to fail a test.
EXPECTED_TOOLS = {
    # Where to reach this server from elsewhere -- lets a connected agent move
    # from a tailnet address to a faster direct LAN one, or discover the
    # endpoint again after the host's address changes.
    "connection_info",
    "list_models",
    # The per-model loading table. Separate from list_models so the common
    # case (one recommended row per model) stays cheap and the detailed case
    # is opt-in for the one model the agent actually cares about.
    "model_options",
    "model_info",
    "load_model",
    "unload_model",
    # The HuggingFace pair, split for the same reason list_models and
    # model_options are: browsing is cheap and knows nothing about sizes,
    # choosing costs a remote GGUF header read and answers exactly.
    "search_models",
    "repo_details",
    "download_model",
    "delete_model",
    "server_status",
    "test_model",
    "get_config",
    "set_config",
}

#: Substrings that would betray an inference tool sneaking onto the control plane.
INFERENCE_VERBS = ("chat", "complet", "generat", "embed", "predict", "sample", "infer")

TREE = (
    "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    "vendor/TinyVision-GGUF/TinyVision-Q4_K_M.gguf",
    "vendor/TinyVision-GGUF/mmproj-TinyVision-F16.gguf",
    "endyjasmi/Qwen3-Embedding-8B-GGUF/qwen3-embedding-8b-q4_k_m.gguf",
)

TINY = "lmstudio-community/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q8_0"
VISION = "vendor/TinyVision-GGUF/TinyVision-Q4_K_M"
EMBEDDING = "endyjasmi/Qwen3-Embedding-8B-GGUF/qwen3-embedding-8b-q4_k_m"

#: A stand-in that is *long*, because the whole point of the compactness test is
#: that a real template would blow an agent's context budget.
BIG_TEMPLATE = (
    "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
    "{% if tools %}{{ tools }}{% endif %}" + ("{# padding #}" * 400)
)

API_KEY = "sf-secret-key-abcdef123456"
#: Fabricated. A redaction test has to put a token-SHAPED string in the tree,
#: so this one spells out what it is; the assertions below pin the first four
#: and last two characters, which is the redaction format itself.
HF_TOKEN = "hf_secrettokenvalue0987654321"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_sf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's ``SF_*`` environment must not leak into these tests.

    ``Config`` is a ``BaseSettings`` with an ``SF_`` prefix, so a real
    ``SF_DATA_DIR`` would silently redirect the test's data directory at the
    developer's live install.
    """
    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)


def fake_meta(path: Path, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
    """Filename-driven GGUF metadata, so no real GGUF bytes are needed."""
    name = path.name.lower()
    if "mmproj" in name:
        return GgufMeta(
            architecture="clip",
            is_mmproj=True,
            has_vision_tensors=True,
            quant_label="F16",
            vision_image_size=896,
            vision_patch_size=14,
        )
    meta = GgufMeta(
        architecture="qwen3",
        n_layer=24,
        n_embd=896,
        n_head=14,
        n_head_kv=2,
        n_ctx_train=32768,
        n_vocab=151936,
        quant_label="Q8_0" if "q8_0" in name else "Q4_K_M",
        tensor_bytes=512 * 1024 * 1024,
        chat_template=BIG_TEMPLATE,
    )
    if "embedding" in name:
        meta = meta.model_copy(update={"extra": {"embedding": True}})
    return meta


class StubEngineManager:
    """Just enough engine manager for ``server_status``."""

    def __init__(self, root: Path) -> None:
        self._info = EngineInfo(
            tag="b10425",
            path=root,
            server_binary=root / "llama-server.exe",
            variant="cuda-13.3",
        )

    def active(self) -> EngineInfo:
        return self._info


class State:
    """Mirror of ``build_state``'s output, with fake hardware."""

    config: Config
    db: Database
    probe: FakeGpuProbe
    registry: Registry
    planner: Planner
    supervisor: Supervisor
    manager: ModelManager
    engine_manager: StubEngineManager
    downloader: Any
    started_at: float
    version: str


@pytest.fixture()
def state(tmp_path: Path) -> Iterator[State]:
    models_root = tmp_path / "models"
    for rel in TREE:
        path = models_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 8192)

    engines = tmp_path / "data" / "engines" / "b10425"
    engines.mkdir(parents=True, exist_ok=True)
    binary = engines / "llama-server.exe"
    binary.write_text("stub", encoding="utf-8")

    config = Config(
        data_dir=tmp_path / "data",
        models=ModelsConfig(dir=models_root, default_ctx=4096, default_ttl_s=900),
    )
    config.server.api_key = API_KEY
    config.hf.token = HF_TOKEN
    config.ensure_dirs()
    config.save(config.data_dir / "config.yaml")
    config.source_path = config.data_dir / "config.yaml"

    db = Database(config.db_path)
    db.migrate()

    probe = FakeGpuProbe(
        [
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 5090",
                total_bytes=32 * GIB,
                free_bytes=30 * GIB,
                used_bytes=2 * GIB,
                compute_capability=(12, 0),
            ),
            GpuInfo(
                index=1,
                name="NVIDIA GeForce RTX 3090",
                total_bytes=24 * GIB,
                free_bytes=23 * GIB,
                used_bytes=1 * GIB,
                compute_capability=(8, 6),
            ),
        ]
    )
    registry = Registry(config, db, meta_reader=fake_meta)
    planner = Planner(config, probe)
    supervisor = Supervisor(config, resolve_binary=lambda _tag: binary)
    manager = ModelManager(config, registry=registry, planner=planner, supervisor=supervisor, db=db)

    composed = State()
    composed.config = config
    composed.db = db
    composed.probe = probe
    composed.registry = registry
    composed.planner = planner
    composed.supervisor = supervisor
    composed.manager = manager
    composed.engine_manager = StubEngineManager(engines)
    composed.downloader = None
    composed.started_at = 0.0
    composed.version = "test"

    registry.scan()
    yield composed
    db.close()


async def call(server: Any, name: str, **arguments: Any) -> dict[str, Any]:
    """Invoke a tool the way a client does, returning the decoded JSON result."""
    result = await server.call_tool(name, arguments)
    assert result.content, f"{name} returned no content"
    payload = json.loads(result.content[0].text)
    assert isinstance(payload, dict)
    return payload


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


async def test_tool_list_is_exactly_the_management_tools(state: State) -> None:
    server = build_management_mcp(state)
    names = {tool.name for tool in await server.list_tools()}
    assert names == EXPECTED_TOOLS


async def test_no_inference_tool_exists(state: State) -> None:
    """Inference stays on the OpenAI API; the MCP control plane must not offer it.

    ``test_model`` is allowed: it is a fixed smoke test with a canned prompt and
    a truncated result, not a generation endpoint.
    """
    server = build_management_mcp(state)
    tools = await server.list_tools()
    for tool in tools:
        if tool.name == "test_model":
            continue
        lowered = tool.name.lower()
        assert not any(verb in lowered for verb in INFERENCE_VERBS), (
            f"tool '{tool.name}' looks like an inference tool"
        )


async def test_instructions_say_inference_is_elsewhere(state: State) -> None:
    server = build_management_mcp(state)
    assert server.instructions is not None
    assert "/v1/chat/completions" in server.instructions
    assert "INFERENCE IS NOT HERE" in server.instructions


async def test_every_tool_has_an_agent_facing_docstring(state: State) -> None:
    server = build_management_mcp(state)
    for tool in await server.list_tools():
        assert tool.description, f"{tool.name} has no description"
        # Long enough to actually explain arguments and consequences.
        assert len(tool.description) > 200, f"{tool.name} description is too thin"


# ---------------------------------------------------------------------------
# list_models / model_info
# ---------------------------------------------------------------------------


#: The lifecycle keys ``list_models`` owns. Every one of them must survive with
#: its original meaning, because nothing in the catalog replaces them.
LIFECYCLE_LIST_MODELS_KEYS = {
    "id",
    "kind",
    "loaded",
    "state",
    "port",
    "pinned",
    "ttl_remaining_s",
    "effective_ttl_s",
}

#: Keys that used to sit beside a catalog field saying the same thing under a
#: different name. Carrying both cost tokens on every row and left an agent
#: guessing which spelling was authoritative; the catalog's spelling won.
RETIRED_DUPLICATE_KEYS = {
    "quant": "quantization",
    "size_gib": "size_gb",
    "vision": "capabilities",
    "tools": "capabilities",
}


async def test_list_models_keeps_every_lifecycle_field(state: State) -> None:
    """What the catalog does not answer -- is it running, for how long -- stays here."""
    server = build_management_mcp(state)
    result = await call(server, "list_models")
    assert result["ok"] is True
    assert result["count"] == 3
    ids = {row["id"] for row in result["models"]}
    assert ids == {TINY, VISION, EMBEDDING}

    row = next(r for r in result["models"] if r["id"] == TINY)
    assert set(row) >= LIFECYCLE_LIST_MODELS_KEYS
    assert row["loaded"] is False
    assert row["effective_ttl_s"] == 900
    # `state` kept its old vocabulary rather than the catalog's
    # loaded/not-loaded, because changing what an existing key *means* is the
    # one kind of addition a client cannot detect. "stopped" means not loaded.
    assert row["state"] == "stopped"


async def test_list_models_says_each_thing_once(state: State) -> None:
    """No field repeats a catalog field under a second name."""
    server = build_management_mcp(state)
    result = await call(server, "list_models")
    row = next(r for r in result["models"] if r["id"] == TINY)
    for retired, survivor in RETIRED_DUPLICATE_KEYS.items():
        assert retired not in row, f"{retired} duplicates {survivor}"
        assert survivor in row
    assert isinstance(row["capabilities"], list)


async def test_list_models_limit_returns_the_newest_downloads(state: State) -> None:
    """The user works from the last thing they got; limit=1 is that question."""
    server = build_management_mcp(state)
    everything = await call(server, "list_models")
    trimmed = await call(server, "list_models", limit=1)
    assert trimmed["count"] == 1
    assert len(trimmed["models"]) == 1
    assert trimmed["models"][0]["id"] == everything["models"][0]["id"]


async def test_list_models_limit_applies_after_the_filters(state: State) -> None:
    """"The newest embedding model", not "the newest model, if it is an embedding"."""
    server = build_management_mcp(state)
    result = await call(server, "list_models", kind="embedding", limit=1)
    assert [row["id"] for row in result["models"]] == [EMBEDDING]


async def test_list_models_limit_above_the_library_size_is_not_an_error(
    state: State,
) -> None:
    server = build_management_mcp(state)
    result = await call(server, "list_models", limit=99)
    assert result["count"] == 3


async def test_list_models_carries_the_catalog_columns(state: State) -> None:
    server = build_management_mcp(state)
    result = await call(server, "list_models")
    assert "recommended" in result["catalog_hint"]

    row = next(r for r in result["models"] if r["id"] == TINY)
    assert row["downloaded_at"].endswith("Z")
    assert row["summary"]
    assert row["n_ctx_train"] == 32768
    assert row["attention_kind"] in {"full", "iswa", "hybrid", "unknown"}
    # Compact by default: the recommended row only.
    assert len(row["options"]) == 1
    option = row["options"][0]
    assert option["recommended"] is True
    assert set(option["load_args"]) == {"model_id", "ctx_size", "parallel", "kv_cache_type"}
    # Both ends of the window, because decode slows as the KV cache fills.
    assert option["est_gen_tps"] > 0
    assert 0 < option["est_gen_tps_full_ctx"] <= option["est_gen_tps"]


async def test_list_models_full_returns_every_context_tier(state: State) -> None:
    server = build_management_mcp(state)
    compact = await call(server, "list_models")
    full = await call(server, "list_models", full=True)
    compact_row = next(r for r in compact["models"] if r["id"] == TINY)
    full_row = next(r for r in full["models"] if r["id"] == TINY)
    assert len(full_row["options"]) > len(compact_row["options"])


async def test_list_models_never_leaks_a_chat_template_or_meta_dump(state: State) -> None:
    """An MCP client pays tokens per byte; a chat template is thousands of them.

    ``n_ctx_train`` is deliberately *not* on this list any more: it is one
    small integer and it is exactly what a model needs to know before choosing
    a context size. What must never appear is the bulk -- the template, the
    tokenizer, the raw tensor accounting.
    """
    server = build_management_mcp(state)
    raw = (await server.call_tool("list_models", {})).content[0].text
    assert "chat_template" not in raw
    assert "tensor_bytes" not in raw
    assert "tokenizer_model" not in raw
    assert "{% for m in messages %}" not in raw
    # Compact by default, so three models with load recipes stay affordable.
    assert len(raw) < 7000, f"list_models output is {len(raw)} chars"


async def test_model_options_returns_the_whole_table_for_one_model(state: State) -> None:
    server = build_management_mcp(state)
    result = await call(server, "model_options", model_id=TINY)
    assert result["ok"] is True
    model = result["model"]
    assert model["id"] == TINY
    assert len(model["options"]) > 1
    assert sum(1 for r in model["options"] if r["recommended"]) == 1
    for option in model["options"]:
        assert "ctx_per_slot" in option
        assert "fits" in option
        assert "if_gpus_idle" in option


async def test_model_options_rejects_an_unknown_model(state: State) -> None:
    server = build_management_mcp(state)
    result = await call(server, "model_options", model_id="nope/not-a-model")
    assert result["ok"] is False
    assert result["error"]["code"] == "model_not_found"


async def test_list_models_filters(state: State) -> None:
    server = build_management_mcp(state)
    embeddings = await call(server, "list_models", kind="embedding")
    assert [row["id"] for row in embeddings["models"]] == [EMBEDDING]

    loaded = await call(server, "list_models", loaded_only=True)
    assert loaded["models"] == []


async def test_model_info_is_detailed_but_summarises_the_template(state: State) -> None:
    server = build_management_mcp(state)
    result = await call(server, "model_info", model_id=TINY)
    assert result["ok"] is True
    assert result["id"] == TINY
    assert result["settings"]["pinned"] is False
    meta = result["meta"]
    assert meta["n_ctx_train"] == 32768
    assert meta["chat_template_present"] is True
    assert meta["chat_template_chars"] > 1000
    assert meta["chat_template_supports_tools"] is True
    assert "chat_template" not in meta
    # Not loaded, so there is no live introspection block.
    assert result["loaded"] is False
    assert "actual" not in result


async def test_model_info_reports_vision_capability(state: State) -> None:
    server = build_management_mcp(state)
    result = await call(server, "model_info", model_id=VISION)
    assert result["capabilities"]["vision"] is True
    assert result["mmproj_path"] is not None


async def test_unknown_model_is_a_result_not_an_exception(state: State) -> None:
    """A protocol error is opaque to an agent; a structured error is actionable."""
    server = build_management_mcp(state)
    result = await server.call_tool("model_info", {"model_id": "no-such-model"})
    assert result.is_error is False
    payload = json.loads(result.content[0].text)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "model_not_found"
    # The message names real models so the agent can retry usefully.
    assert "Known models include" in payload["error"]["message"]


async def test_load_of_unknown_model_returns_error_result(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "load_model", model_id="nope")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "model_not_found"


async def test_unload_of_unloaded_model_is_not_an_error(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "unload_model", model_id=TINY)
    assert payload["ok"] is True
    assert payload["unloaded"] is False


# ---------------------------------------------------------------------------
# delete_model confirmation gate
# ---------------------------------------------------------------------------


async def test_delete_model_without_confirm_refuses_and_changes_nothing(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "delete_model", model_id=TINY)
    assert payload["ok"] is False
    assert payload["confirmed"] is False
    assert payload["error"]["code"] == "confirmation_required"
    assert payload["error"]["param"] == "confirm"
    assert state.registry.resolve(TINY) is not None


async def test_delete_model_without_confirm_says_what_it_would_do(state: State) -> None:
    server = build_management_mcp(state)
    keep = await call(server, "delete_model", model_id=TINY, delete_files=False)
    assert "registry entry" in keep["error"]["message"]
    wipe = await call(server, "delete_model", model_id=TINY, delete_files=True)
    assert "permanently deletes" in wipe["error"]["message"]


async def test_delete_model_with_confirm_delegates_to_the_registry(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "delete_model", model_id=TINY, confirm=True)
    assert payload["ok"] is True
    assert payload["model_id"] == TINY
    assert payload["files_deleted"] is False
    assert payload["removed"], "the registry should report which files it would remove"
    assert state.registry.resolve(TINY) is None
    # confirm without delete_files must not touch the filesystem.
    assert Path(payload["removed"][0]).is_file()  # noqa: ASYNC240 - tiny tmp_path stat


async def test_delete_model_with_files_actually_deletes(state: State) -> None:
    server = build_management_mcp(state)
    target = state.registry.resolve(EMBEDDING)
    assert target is not None
    path = target.path
    payload = await call(
        server, "delete_model", model_id=EMBEDDING, delete_files=True, confirm=True
    )
    assert payload["ok"] is True
    assert payload["files_deleted"] is True
    assert not path.exists()


# ---------------------------------------------------------------------------
# server_status
# ---------------------------------------------------------------------------


async def test_server_status_reports_gpus_loaded_and_engine(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "server_status")
    assert payload["ok"] is True
    assert len(payload["gpus"]) == 2
    first = payload["gpus"][0]
    assert first["index"] == 0
    assert first["free_gib"] == pytest.approx(30.0)
    assert first["compute_capability"] == "12.0"
    assert payload["loaded"] == []
    assert payload["model_count"] == 3
    assert payload["queue_depth"] == 0
    assert payload["engine_tag"] == "b10425"
    assert payload["active_downloads"] == 0
    assert payload["draining"] is False


async def test_server_status_survives_a_missing_engine_manager(state: State) -> None:
    state.engine_manager = None  # type: ignore[assignment]
    server = build_management_mcp(state)
    payload = await call(server, "server_status")
    assert payload["ok"] is True
    assert payload["engine_tag"] is None


async def test_server_status_names_who_is_holding_vram(state: State) -> None:
    """D23: an agent short of VRAM must be able to tell a leak from a live owner."""
    server = build_management_mcp(state)
    payload = await call(server, "server_status")
    # Nothing runs out of this test's engines dir, so the counts are real zeros.
    assert payload["vram_orphan_count"] == 0
    assert payload["engine_processes"] == {
        "ours": 0,
        "child_of_live_process": 0,
        "orphan": 0,
    }


async def test_server_status_answers_null_rather_than_zero_when_it_cannot_look(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"I could not read the process table" must not render as "nothing is leaking"."""
    from studioforge.core import vram_holders

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("access denied")

    monkeypatch.setattr(vram_holders, "find_engine_processes", boom)
    server = build_management_mcp(state)
    payload = await call(server, "server_status")
    assert payload["ok"] is True
    assert payload["vram_orphan_count"] is None
    assert payload["engine_processes"] is None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


async def test_get_config_redacts_both_secrets(state: State) -> None:
    server = build_management_mcp(state)
    raw = (await server.call_tool("get_config", {})).content[0].text
    assert API_KEY not in raw
    assert HF_TOKEN not in raw
    payload = json.loads(raw)
    assert payload["config"]["server"]["api_key"] == "sf-s...56"
    assert payload["config"]["hf"]["token"] == "hf_s...21"
    assert "server.port" in payload["restart_required_keys"]


async def test_set_config_rejects_an_unknown_key(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "set_config", updates={"models.no_such_setting": 1})
    assert payload["ok"] is False
    assert "unknown config key" in payload["error"]["message"]
    assert payload["error"]["code"] == "invalid_config"


async def test_set_config_rejects_an_invalid_value_atomically(state: State) -> None:
    """A rejected batch must write nothing at all."""
    server = build_management_mcp(state)
    before = state.config.models.default_ctx
    payload = await call(
        server,
        "set_config",
        updates={"models.default_ctx": 16384, "planner.headroom_fraction": 0.95},
    )
    assert payload["ok"] is False
    assert state.config.models.default_ctx == before


async def test_set_config_applies_live_and_reports_restart_needs(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(
        server,
        "set_config",
        updates={"models.default_ctx": 16384, "server.port": 4321},
    )
    assert payload["ok"] is True
    assert payload["updated"] == ["models.default_ctx", "server.port"]
    assert payload["restart_required"] == ["server.port"]
    # A live-applicable key takes effect on the shared Config immediately...
    assert state.config.models.default_ctx == 16384
    # ...and the whole change is on disk.
    import yaml

    written = yaml.safe_load(
        Path(payload["config_path"]).read_text(encoding="utf-8")  # noqa: ASYNC240 - tmp file
    )
    assert written["models"]["default_ctx"] == 16384
    assert written["server"]["port"] == 4321


async def test_set_config_requires_a_non_empty_mapping(state: State) -> None:
    server = build_management_mcp(state)
    payload = await call(server, "set_config", updates={})
    assert payload["ok"] is False
    assert payload["error"]["param"] == "updates"


# ---------------------------------------------------------------------------
# download_model degradation
# ---------------------------------------------------------------------------


async def test_download_model_reports_unavailable_instead_of_raising(state: State) -> None:
    """The downloader is a separate subsystem; its absence must not break the plane."""
    state.downloader = None
    server = build_management_mcp(state)
    payload = await call(server, "download_model", repo_id="bartowski/Whatever-GGUF")
    assert payload["ok"] is False
    assert payload["error"]["code"] == "downloads_unavailable"
    assert "manually" in payload["error"]["message"]


def _logical_download(repo_id: str = "bartowski/Whatever-GGUF") -> Any:
    from studioforge.core.hf_search import GgufFileInfo, LogicalDownload

    info = GgufFileInfo(
        filename="whatever-Q4_K_M.gguf",
        size_bytes=123,
        quant="Q4_K_M",
        is_mmproj=False,
        shard_index=None,
        shard_total=None,
        sha256=None,
        lfs_oid=None,
    )
    return LogicalDownload(
        repo_id=repo_id, quant="Q4_K_M", files=[info], mmproj=None, total_bytes=123
    )


async def test_download_model_calls_enqueue_with_its_real_signature(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tool must call ``Downloader.enqueue(item, *, include_mmproj=...)``.

    It previously called ``enqueue(repo_id=..., quant=..., include_mmproj=...)``
    -- kwargs the real Downloader does not accept -- so every real call raised
    TypeError and the tool was completely broken, while a kwargs-shaped stub
    kept this suite green. The stub below has the real signature, and the quant
    resolution is patched at the shared helper the HTTP route also uses.
    """
    from studioforge.core import downloader as downloader_module

    item = _logical_download()
    seen: list[tuple[str, str | None]] = []

    async def fake_resolve(
        config: Any, planner: Any, repo_id: str, quant: str | None = None
    ) -> Any:
        seen.append((repo_id, quant))
        return item

    monkeypatch.setattr(downloader_module, "resolve_download_choice", fake_resolve)

    calls: list[tuple[Any, bool]] = []

    class StubDownloader:
        async def enqueue(
            self, item: Any, *, include_mmproj: bool = True, force: bool = False
        ) -> str:
            calls.append((item, include_mmproj))
            return "grp-1"

        def active(self) -> list[str]:
            return []

    state.downloader = StubDownloader()
    server = build_management_mcp(state)
    payload = await call(
        server, "download_model", repo_id="bartowski/Whatever-GGUF", quant="Q4_K_M"
    )
    assert payload["ok"] is True, payload
    assert payload["queued"]["group_id"] == "grp-1"
    assert payload["queued"]["quant"] == "Q4_K_M"
    assert seen == [("bartowski/Whatever-GGUF", "Q4_K_M")]
    assert calls == [(item, True)]


async def test_download_model_default_includes_mmproj(state: State) -> None:
    server = build_management_mcp(state)
    schema = next(t for t in await server.list_tools() if t.name == "download_model").input_schema
    assert schema["properties"]["include_mmproj"]["default"] is True
    assert schema["required"] == ["repo_id"]


# search_models / repo_details
# ---------------------------------------------------------------------------


def _gguf_file(
    filename: str, *, quant: str = "Q4_K_M", size: int | None = 456, mmproj: bool = False
) -> Any:
    from studioforge.core.hf_search import GgufFileInfo

    return GgufFileInfo(
        filename=filename,
        size_bytes=size,
        quant=quant,
        is_mmproj=mmproj,
        shard_index=None,
        shard_total=None,
        sha256=None,
        lfs_oid=None,
    )


def _gguf_repo_info(
    repo_id: str = "test/TestModel-GGUF",
    *,
    files: Any = None,
    trending_score: int | None = None,
    gated: bool | str = False,
) -> Any:
    from studioforge.core.hf_search import GgufRepoInfo

    return GgufRepoInfo(
        repo_id=repo_id,
        publisher="test",
        name="Test Model",
        downloads=100,
        likes=50,
        gated=gated,
        private=False,
        last_modified="2024-01-15T10:30:00Z",
        created_at="2023-01-15T10:30:00Z",
        trending_score=trending_score,
        files=list(files) if files is not None else [_gguf_file("test-Q4_K_M.gguf")],
    )


async def test_search_models_calls_hfsearch_with_mapped_arguments(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """search_models calls HfSearch.search with the given parameters."""
    from studioforge.core import hf_search as hf_search_module

    repo = _gguf_repo_info()
    seen: list[tuple[str, dict[str, Any]]] = []

    class FakeHfSearch:
        last_search_truncated = False

        def __init__(self, config: Any) -> None:
            pass

        async def search(
            self,
            query: str,
            *,
            limit: int = 20,
            author: str | None = None,
            sort: str = "downloads",
            newer_than_days: int | None = None,
            date_field: str = "updated",
        ) -> list[Any]:
            seen.append(
                (
                    query,
                    {
                        "limit": limit,
                        "author": author,
                        "sort": sort,
                        "newer_than_days": newer_than_days,
                        "date_field": date_field,
                    },
                )
            )
            return [repo]

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(hf_search_module, "HfSearch", FakeHfSearch)

    server = build_management_mcp(state)
    payload = await call(
        server,
        "search_models",
        query="llama",
        limit=15,
        sort="likes",
        newer_than_days=30,
        date_field="created",
        author="bartowski",
    )

    assert payload["ok"] is True
    assert payload["truncated"] is False
    assert "repos" in payload
    assert len(payload["repos"]) == 1
    repo_row = payload["repos"][0]
    assert repo_row["repo_id"] == "test/TestModel-GGUF"
    assert repo_row["publisher"] == "test"
    assert repo_row["downloads"] == 100
    assert repo_row["quants"] == ["Q4_K_M"]
    assert payload["sort_options"][0] == "downloads"
    assert payload["date_field_options"] == ["updated", "created"]

    # Verify HfSearch.search was called with correct arguments
    assert len(seen) == 1
    called_query, called_kwargs = seen[0]
    assert called_query == "llama"
    assert called_kwargs["limit"] == 15
    assert called_kwargs["sort"] == "likes"
    assert called_kwargs["newer_than_days"] == 30
    assert called_kwargs["date_field"] == "created"
    assert called_kwargs["author"] == "bartowski"


async def test_search_models_caps_limit_at_25(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """search_models caps limit at 25 to keep context compact."""
    from studioforge.core import hf_search as hf_search_module

    class FakeHfSearch:
        last_search_truncated = False
        seen_limit: int | None = None

        def __init__(self, config: Any) -> None:
            pass

        async def search(
            self,
            query: str,
            *,
            limit: int = 20,
            author: str | None = None,
            sort: str = "downloads",
            newer_than_days: int | None = None,
            date_field: str = "updated",
        ) -> list[Any]:
            self.seen_limit = limit
            return []

        async def aclose(self) -> None:
            pass

    fake = FakeHfSearch(None)
    monkeypatch.setattr(hf_search_module, "HfSearch", lambda config: fake)

    server = build_management_mcp(state)
    payload = await call(server, "search_models", query="test", limit=100)

    assert payload["ok"] is True
    assert fake.seen_limit == 25


async def test_search_models_invalid_sort_returns_error(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """search_models returns ok: false when sort is invalid."""
    from studioforge.core import hf_search as hf_search_module
    from studioforge.errors import BadRequestError

    class FakeHfSearch:
        def __init__(self, config: Any) -> None:
            pass

        async def search(self, query: str, **kwargs: Any) -> list[Any]:
            raise BadRequestError(
                "Unknown sort key",
                param="sort",
                code="bad_request",
            )

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr(hf_search_module, "HfSearch", FakeHfSearch)

    server = build_management_mcp(state)
    payload = await call(server, "search_models", query="test", sort="invalid")

    assert payload["ok"] is False
    assert payload["error"]["code"] == "bad_request"
    assert payload["error"]["param"] == "sort"


def _patch_search(monkeypatch: pytest.MonkeyPatch, repos: list[Any]) -> dict[str, Any]:
    """Point ``HfSearch`` at a canned result set; return what the tool asked for."""
    from studioforge.core import hf_search as hf_search_module

    seen: dict[str, Any] = {}

    class FakeHfSearch:
        last_search_truncated = False

        def __init__(self, config: Any) -> None:
            pass

        async def search(self, query: str, **kwargs: Any) -> list[Any]:
            seen["query"] = query
            seen.update(kwargs)
            return repos

        async def repo_info(self, repo_id: str) -> Any:
            seen["repo_id"] = repo_id
            return repos[0]

        async def aclose(self) -> None:
            seen["closed"] = True

    monkeypatch.setattr(hf_search_module, "HfSearch", FakeHfSearch)
    return seen


async def test_search_rows_carry_no_sizes_and_no_fit_blocks(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the rewrite: a search row is thin and says nothing it cannot know.

    HuggingFace's list endpoint reports no file sizes, so the previous version --
    which reused the repo-detail payload -- emitted a full per-quant fit block
    whose every field was a variation on "unknown", ~60 KB for three repos.
    """
    repo = _gguf_repo_info(
        files=[
            _gguf_file("test-Q4_K_M.gguf", quant="Q4_K_M", size=None),
            _gguf_file("test-Q8_0.gguf", quant="Q8_0", size=None),
            _gguf_file("mmproj-F16.gguf", quant="F16", mmproj=True, size=None),
        ]
    )
    _patch_search(monkeypatch, [repo])

    server = build_management_mcp(state)
    payload = await call(server, "search_models", query="test")

    row = payload["repos"][0]
    assert set(row) == {
        "repo_id",
        "publisher",
        "downloads",
        "likes",
        "updated_days_ago",
        "created_days_ago",
        "gated",
        "quants",
        "mmproj",
        "file_count",
    }
    assert row["quants"] == ["Q4_K_M", "Q8_0"]  # the projector is not a choice
    assert row["mmproj"] is True
    assert row["file_count"] == 3
    assert "fit" not in json.dumps(row)
    assert "total_bytes" not in json.dumps(row)


async def test_search_row_reports_trending_score_only_when_sorted_by_it(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF omits ``trendingScore`` under every other ordering, so ``None`` is not a zero."""
    repo = _gguf_repo_info(trending_score=17)
    _patch_search(monkeypatch, [repo])
    server = build_management_mcp(state)

    ranked = await call(server, "search_models", query="test", sort="trending")
    assert ranked["repos"][0]["trending_score"] == 17

    by_downloads = await call(server, "search_models", query="test", sort="downloads")
    assert "trending_score" not in by_downloads["repos"][0]


async def test_search_row_flattens_the_two_gated_spellings(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HF answers ``"auto"``/``"manual"``; both mean "accept the terms first"."""
    _patch_search(monkeypatch, [_gguf_repo_info(gated="manual")])
    server = build_management_mcp(state)
    payload = await call(server, "search_models", query="test")
    assert payload["repos"][0]["gated"] is True


# repo_details
# ---------------------------------------------------------------------------

#: A ``_repo_payload(..., with_context=True)`` result, in WP7's documented
#: shape. Written out here rather than produced by the real function because the
#: real one reads a GGUF header over the network.
FULL_REPO_PAYLOAD: dict[str, Any] = {
    "repo_id": "test/TestModel-GGUF",
    "publisher": "test",
    "name": "Test Model",
    "downloads": 100,
    "likes": 50,
    "trending_score": None,
    "gated": "manual",
    "private": False,
    "last_modified": "2024-01-15T10:30:00Z",
    "created_at": "2023-01-15T10:30:00Z",
    "updated_days_ago": 3.14159,
    "created_days_ago": 400.5,
    "quants": [
        {
            "quant": "Q4_K_M",
            "total_bytes": 21 * GIB,
            "files": ["test-Q4_K_M.gguf"],
            "mmproj": "mmproj-F16.gguf",
            "group_id": "test-testmodel-gguf-q4-k-m",
            "fit": {
                "verdict": "fits-one-gpu",
                "message": "Q4_K_M needs about 24.1 GiB at ctx 32768.",
                "required_bytes": 25878000000,
                "largest_gpu_free_bytes": 29879262413,
                "total_gpu_free_bytes": 99473753704,
                "suggested_quant": None,
                "group_id": "test-testmodel-gguf-q4-k-m",
                "quant": "Q4_K_M",
                "weights_bytes": 21 * GIB,
                "kv_allowance_bytes": 0,
                "overhead_bytes": 0,
                "ctx_size": 32768,
                "gpu_count": 4,
                "approximate": False,
                "size_known": True,
            },
            "context_fit": {
                "tiers": [65536, 131072, 262144],
                "n_ctx_train": 262144,
                "attention_kind": "hybrid",
                "n_layer": 65,
                "kv_bytes_per_token_f16": 66134,
                "kv_bytes_per_token_ctx": 262144,
                "source": "remote-gguf-header",
                "approximate": False,
                "unavailable": None,
                "placements": [
                    {
                        "key": "single_best",
                        "label": "1x RTX 5090",
                        "short_label": "1x5090",
                        "devices": [0],
                        "capacity_gib": 28.7,
                        "weights_bytes": 21 * GIB,
                        "weights_fit": True,
                        "fits": {"65536": True, "131072": True, "262144": False},
                        "kv_cache_type": {"65536": "f16", "131072": "q8_0"},
                        "max_ctx": 98304,
                        "max_ctx_q8": 131072,
                    }
                ],
            },
        }
    ],
}


def _patch_repo_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from studioforge.api import mgmt_routes

    seen: dict[str, Any] = {}

    async def fake_payload(_state: Any, repo: Any, *, with_context: bool = False) -> dict[str, Any]:
        seen["repo"] = repo
        seen["with_context"] = with_context
        payload = json.loads(json.dumps(FULL_REPO_PAYLOAD))
        if not with_context:
            for entry in payload["quants"]:
                entry.pop("context_fit", None)
        return payload

    monkeypatch.setattr(mgmt_routes, "_repo_payload", fake_payload)
    return seen


async def test_repo_details_returns_sizes_fit_and_the_context_matrix(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    search_seen = _patch_search(monkeypatch, [_gguf_repo_info()])
    payload_seen = _patch_repo_payload(monkeypatch)

    server = build_management_mcp(state)
    payload = await call(server, "repo_details", repo_id="test/TestModel-GGUF")

    assert payload["ok"] is True
    assert search_seen["repo_id"] == "test/TestModel-GGUF"
    assert search_seen["closed"] is True
    assert payload_seen["with_context"] is True
    assert payload["gated"] is True
    assert payload["updated_days_ago"] == 3.1

    quant = payload["quants"][0]
    assert quant["quant"] == "Q4_K_M"
    assert quant["total_gb"] == 21.0  # same GiB-valued unit as the catalog's size_gb
    assert quant["mmproj"] == "mmproj-F16.gguf"
    # The fit block keeps only what changes a decision.
    assert set(quant["fit"]) == {"verdict", "message", "suggested_quant", "approximate"}

    context = quant["context_fit"]
    assert set(context) == {"tiers", "n_ctx_train", "attention_kind", "source", "placements"}
    placement = context["placements"][0]
    assert placement["label"] == "1x RTX 5090"
    assert placement["devices"] == [0]
    assert placement["fits"] == {"65536": True, "131072": True, "262144": False}
    # 131072 fits, but only above max_ctx -- i.e. on a q8_0 cache, which is what
    # max_ctx_q8 says. That pair replaces the per-tier kv_cache_type map.
    assert placement["max_ctx"] == 98304
    assert placement["max_ctx_q8"] == 131072
    # Rendering-only keys are dropped; nothing an agent reads is.
    assert "key" not in placement
    assert "short_label" not in placement
    assert "weights_bytes" not in placement
    assert "kv_cache_type" not in placement
    assert placement["weights_fit"] is True


async def test_repo_details_surfaces_an_unreadable_header_rather_than_hiding_it(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``approximate``/``unavailable`` only appear when they are true, and then they must."""
    _patch_search(monkeypatch, [_gguf_repo_info()])
    from studioforge.api import mgmt_routes

    async def fake_payload(_state: Any, repo: Any, *, with_context: bool = False) -> dict[str, Any]:
        payload = json.loads(json.dumps(FULL_REPO_PAYLOAD))
        matrix = payload["quants"][0]["context_fit"]
        matrix["approximate"] = True
        matrix["unavailable"] = "gated repo: no HF token configured"
        matrix["source"] = None
        return payload

    monkeypatch.setattr(mgmt_routes, "_repo_payload", fake_payload)

    server = build_management_mcp(state)
    payload = await call(server, "repo_details", repo_id="test/TestModel-GGUF")
    context = payload["quants"][0]["context_fit"]
    assert context["approximate"] is True
    assert context["unavailable"] == "gated repo: no HF token configured"
    assert context["source"] is None


async def test_repo_details_can_skip_the_header_read(
    state: State, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_search(monkeypatch, [_gguf_repo_info()])
    seen = _patch_repo_payload(monkeypatch)

    server = build_management_mcp(state)
    payload = await call(
        server, "repo_details", repo_id="test/TestModel-GGUF", with_context=False
    )

    assert seen["with_context"] is False
    assert "context_fit" not in payload["quants"][0]
    assert payload["quants"][0]["total_gb"] > 0


async def test_repo_details_defaults_to_reading_the_context_matrix(state: State) -> None:
    server = build_management_mcp(state)
    schema = next(t for t in await server.list_tools() if t.name == "repo_details").input_schema
    assert schema["properties"]["with_context"]["default"] is True
    assert schema["required"] == ["repo_id"]


# ---------------------------------------------------------------------------
# HTTP mounting
# ---------------------------------------------------------------------------


async def test_mount_adds_the_mcp_route_at_the_exact_path(state: State) -> None:
    """The route must land on ``/mcp`` exactly -- not ``/mcp/mcp``, not a redirect.

    Mounting a sub-application would do one of those, which is why the
    implementation appends the routes instead.
    """
    from fastapi import FastAPI

    app = FastAPI()
    before = {getattr(route, "path", None) for route in app.router.routes}
    mount_management_mcp(app, state, path="/mcp")
    after = {getattr(route, "path", None) for route in app.router.routes}
    assert "/mcp" in after - before


async def test_mount_wires_the_session_manager_into_the_app_lifespan(state: State) -> None:
    """Streamable HTTP needs a running task group for the app's whole lifetime."""
    from fastapi import FastAPI

    app = FastAPI()
    original = app.router.lifespan_context
    mount_management_mcp(app, state)
    assert app.router.lifespan_context is not original
    assert app.state.management_mcp is not None

    # Entering the wrapped lifespan must start the session manager without
    # error and tear it down cleanly.
    async with app.router.lifespan_context(app):
        pass


def test_settings_model_is_not_dumped_into_list_output() -> None:
    """Guard against a future refactor re-adding the full pydantic dump.

    Asserted on the source text rather than at runtime because the failure mode
    is a well-meaning "just return record.model_dump()" edit.
    """
    source = Path(__import__("studioforge.mcp.management", fromlist=["__file__"]).__file__ or "")
    text = source.read_text(encoding="utf-8")
    compact = text.split("def _compact_model")[1].split("def _compact_gpu")[0]
    assert "model_dump" not in compact


def test_model_settings_default_is_unpinned() -> None:
    """Sanity anchor for the ttl/pinned assertions above."""
    assert ModelSettings().pinned is False


# ---------------------------------------------------------------------------
# connection_info
# ---------------------------------------------------------------------------


async def test_connection_info_returns_a_reachable_url(state: State) -> None:
    """An agent must be able to re-find this server without out-of-band help."""
    server = build_management_mcp(state)
    result = await call(server, "connection_info")
    assert result["ok"] is True
    assert result["recommended"].startswith("http://")
    assert result["recommended"].endswith("/mcp")
    assert result["endpoints"]
    assert result["openai_base_urls"]


async def test_connection_info_prefers_tailscale_by_default(state: State) -> None:
    """A tailnet address survives a network change; a LAN address does not."""
    server = build_management_mcp(state)
    result = await call(server, "connection_info")
    if result["tailscale"]:
        assert result["recommended_kind"] == "tailscale"
        assert result["recommended"] == result["tailscale"][0]["url"]


async def test_connection_info_can_hand_back_a_lan_address(state: State) -> None:
    """The user's case: connected over Tailscale, wants a direct local hop."""
    server = build_management_mcp(state)
    result = await call(server, "connection_info", prefer="lan")
    assert result["ok"] is True
    if result["lan"]:
        assert result["recommended"] == result["lan"][0]["url"]
    else:
        assert "no lan address" in result.get("note", "").lower()


async def test_connection_info_says_so_when_the_preference_is_unavailable(
    state: State,
) -> None:
    """Silently returning a different network would be worse than saying no."""
    server = build_management_mcp(state)
    result = await call(server, "connection_info", prefer="loopback")
    assert result["ok"] is True
    assert result["recommended"]


async def test_connection_info_reports_whether_a_pin_is_needed(state: State) -> None:
    server = build_management_mcp(state)
    state.config.mcp.pin = "12345678"
    state.config.mcp.pin_required = True
    result = await call(server, "connection_info")
    assert result["pin_required"] is True
    # The PIN itself is not echoed here: this tool answers "where", and the
    # caller already had to authenticate to ask.
    assert "12345678" not in str(result)
