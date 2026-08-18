"""Virtual-model presets: system prompt + sampler defaults + shared instances.

This is the "Modelfile/persona" feature (docs/COMPARISON.md): a virtual model
optionally carries a system prompt and request-time sampler defaults, applied
by the gateway per request. Three properties are load-bearing and each has a
test that fails without the implementation:

* a client's own system message SURVIVES -- ours is prepended, theirs kept;
* sampler defaults never override an explicit request value (alias spellings
  included);
* two presets over one base share one llama-server instance, while any
  launch-time delta (adapters, ctx/kv override) still gets a dedicated one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from studioforge.config import Config, ModelsConfig
from studioforge.core.registry import Registry
from studioforge.db import Database
from studioforge.types import (
    GB,
    AdapterAttachment,
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    ModelRecord,
    ModelSettings,
    VirtualPreset,
)

# ---------------------------------------------------------------------------
# Preset -> payload application (pure logic)
# ---------------------------------------------------------------------------


def chat_payload(**extra: Any) -> dict[str, Any]:
    return {"model": "m", "messages": [{"role": "user", "content": "hi"}], **extra}


def full_preset() -> VirtualPreset:
    return VirtualPreset(
        system_prompt="You are Tiny RP.",
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.1,
        max_tokens=256,
    )


def test_system_prompt_injected_when_client_sends_none() -> None:
    payload = chat_payload()
    full_preset().apply_to_payload(payload, chat=True)
    assert payload["messages"][0] == {"role": "system", "content": "You are Tiny RP."}
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_client_system_message_survives_and_ours_is_prepended() -> None:
    """A preset must never silently discard the client's own instructions."""
    payload = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "Client rules."},
            {"role": "user", "content": "hi"},
        ],
    }
    full_preset().apply_to_payload(payload, chat=True)
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "system", "user"]
    assert payload["messages"][0]["content"] == "You are Tiny RP."
    assert payload["messages"][1]["content"] == "Client rules."


def test_sampler_defaults_fill_absent_fields() -> None:
    payload = chat_payload()
    full_preset().apply_to_payload(payload, chat=True)
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 40
    assert payload["min_p"] == 0.05
    assert payload["repeat_penalty"] == 1.1
    assert payload["max_tokens"] == 256


def test_sampler_defaults_never_override_explicit_request_values() -> None:
    payload = chat_payload(temperature=1.5, top_p=1.0, top_k=1, min_p=0.5, max_tokens=8)
    full_preset().apply_to_payload(payload, chat=True)
    assert payload["temperature"] == 1.5
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 1
    assert payload["min_p"] == 0.5
    assert payload["max_tokens"] == 8


def test_repetition_penalty_alias_blocks_the_preset_default() -> None:
    """A client sending the HF spelling has chosen a value; the preset yields."""
    payload = chat_payload(repetition_penalty=1.3)
    full_preset().apply_to_payload(payload, chat=True)
    assert "repeat_penalty" not in payload
    assert payload["repetition_penalty"] == 1.3


def test_max_completion_tokens_alias_blocks_the_preset_default() -> None:
    payload = chat_payload(max_completion_tokens=32)
    full_preset().apply_to_payload(payload, chat=True)
    assert "max_tokens" not in payload


def test_explicit_null_counts_as_absent() -> None:
    """OpenAI semantics: an explicit JSON null means unset."""
    payload = chat_payload(temperature=None)
    full_preset().apply_to_payload(payload, chat=True)
    assert payload["temperature"] == 0.7


def test_non_chat_application_skips_the_system_prompt() -> None:
    payload = {"model": "m", "prompt": "Once upon"}
    full_preset().apply_to_payload(payload, chat=False)
    assert "messages" not in payload
    assert payload["temperature"] == 0.7


def test_temperature_zero_is_a_real_default_not_falsy() -> None:
    payload = chat_payload()
    VirtualPreset(temperature=0.0).apply_to_payload(payload, chat=True)
    assert payload["temperature"] == 0.0


def test_preset_validation_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        VirtualPreset(temperature=-1.0)
    with pytest.raises(ValueError):
        VirtualPreset(top_p=1.5)
    with pytest.raises(ValueError):
        VirtualPreset(max_tokens=0)
    with pytest.raises(ValueError):
        VirtualPreset(repeat_penalty=0)


def test_is_empty() -> None:
    assert VirtualPreset().is_empty() is True
    assert VirtualPreset(top_k=1).is_empty() is False


# ---------------------------------------------------------------------------
# Registry + DB persistence
# ---------------------------------------------------------------------------

BASE = "pub/Tiny-GGUF/Tiny-Q8_0"
LORA = "LORAs/Tiny-lora-F16"


def _meta_reader(path: Path, shard_paths: Any = None) -> GgufMeta:
    if "lora" in path.name.lower():
        return GgufMeta(architecture="llama", n_layer=8, is_adapter=True, quant_label="F16")
    return GgufMeta(
        architecture="llama",
        n_layer=8,
        n_embd=512,
        n_head=8,
        n_head_kv=8,
        quant_label="Q8_0",
        tensor_bytes=4096,
    )


@pytest.fixture()
def db(tmp_path: Path) -> Any:
    database = Database(tmp_path / "data" / "registry.sqlite3")
    database.migrate()
    yield database
    database.close()


@pytest.fixture()
def library(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    for rel in (f"{BASE}.gguf", f"{LORA}.gguf"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * 4096)
    return root


def _config(library: Path, tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data", models=ModelsConfig(dir=library))


@pytest.fixture()
def scanned(library: Path, tmp_path: Path, db: Database) -> Registry:
    reg = Registry(_config(library, tmp_path), db, meta_reader=_meta_reader)
    reg.scan()
    return reg


def test_db_round_trips_the_preset(db: Database) -> None:
    db.save_virtual_model("vm", "base", "VM", [], preset={"system_prompt": "s", "top_k": 5})
    row = db.list_virtual_models()[0]
    assert row["adapters"] == []
    assert row["preset"] == {"system_prompt": "s", "top_k": 5}


def test_db_without_preset_keeps_the_legacy_list_shape(db: Database) -> None:
    """No preset -> plain adapter list, so a rollback build still reads its rows."""
    import json

    db.save_virtual_model("vm", "base", "VM", [{"adapter_id": "a", "scale": 1.0}])
    raw = (
        db.connect().execute("SELECT adapters_json FROM virtual_models WHERE id = 'vm'").fetchone()
    )
    assert isinstance(json.loads(raw["adapters_json"]), list)
    row = db.list_virtual_models()[0]
    assert row["adapters"] == [{"adapter_id": "a", "scale": 1.0}]
    assert row["preset"] is None


def test_registry_creates_and_reloads_a_preset(scanned: Registry, db: Database) -> None:
    preset = VirtualPreset(system_prompt="Persona.", temperature=0.3, max_tokens=64)
    record = scanned.create_virtual_model(
        id="virtual/persona", base_model_id=BASE, name=None, adapters=[], preset=preset
    )
    assert record.preset == preset

    # Survives a re-scan (rebuild from SQLite)...
    scanned.scan()
    reloaded = scanned.get("virtual/persona")
    assert reloaded is not None and reloaded.preset == preset

    # ...and a full restart with a fresh Registry over the same DB.
    fresh = Registry(scanned._config, db, meta_reader=_meta_reader)
    fresh.scan()
    again = fresh.get("virtual/persona")
    assert again is not None and again.preset == preset


def test_preset_appears_in_openai_listing(scanned: Registry) -> None:
    scanned.create_virtual_model(
        id="virtual/persona",
        base_model_id=BASE,
        name=None,
        adapters=[],
        preset=VirtualPreset(system_prompt="P.", top_k=5),
    )
    entry = next(e for e in scanned.openai_list() if e["id"] == "virtual/persona")
    assert entry["studioforge"]["preset"] == {"system_prompt": "P.", "top_k": 5}
    base_entry = next(e for e in scanned.openai_list() if e["id"] == BASE)
    assert base_entry["studioforge"]["preset"] is None


def test_empty_preset_is_stored_as_absent(scanned: Registry) -> None:
    record = scanned.create_virtual_model(
        id="virtual/plain", base_model_id=BASE, name=None, adapters=[], preset=VirtualPreset()
    )
    assert record.preset is None


def test_invalid_stored_preset_degrades_to_none(scanned: Registry, db: Database) -> None:
    """A hand-edited/older row must not make the virtual model disappear."""
    db.save_virtual_model("virtual/broken", BASE, "B", [], preset={"temperature": "hot"})
    scanned.scan()
    record = scanned.get("virtual/broken")
    assert record is not None
    assert record.preset is None


def test_saving_settings_preserves_the_preset(scanned: Registry) -> None:
    preset = VirtualPreset(system_prompt="Keep me.")
    scanned.create_virtual_model(
        id="virtual/persona", base_model_id=BASE, name=None, adapters=[], preset=preset
    )
    scanned.save_settings("virtual/persona", ModelSettings(ctx_size=4096))
    scanned.scan()
    record = scanned.get("virtual/persona")
    assert record is not None
    assert record.settings.ctx_size == 4096
    assert record.preset == preset, "a settings save must not erase the persona"


# ---------------------------------------------------------------------------
# Instance sharing: serving_record
# ---------------------------------------------------------------------------


class _StubRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = {r.id: r for r in records}

    def get(self, model_id: str) -> ModelRecord | None:
        return self._records.get(model_id)

    def resolve(self, name: str) -> ModelRecord | None:
        return self._records.get(name)

    def all(self) -> list[ModelRecord]:
        return list(self._records.values())

    def known_ids(self) -> list[str]:
        return sorted(self._records)

    def touch(self, model_id: str) -> None:
        return None


def _record(model_id: str, **kwargs: Any) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id,
        path="/models/base.gguf",
        size_bytes=GB,
        meta=GgufMeta(architecture="llama", n_layer=8, n_head=8, n_head_kv=8, n_embd=512),
        **kwargs,
    )


def _manager(records: list[ModelRecord]) -> Any:
    from studioforge.core.manager import ModelManager

    return ModelManager(
        Config(data_dir="/tmp/sf-presets"),
        registry=_StubRegistry(records),  # type: ignore[arg-type]
        planner=None,  # type: ignore[arg-type]
        supervisor=None,  # type: ignore[arg-type]
        db=None,  # type: ignore[arg-type]
    )


def test_preset_only_virtual_model_serves_from_its_base() -> None:
    """The whole efficiency win: request-time-only deltas share the instance."""
    base = _record("pub/base")
    virtual = _record(
        "virtual/persona",
        is_virtual=True,
        base_model_id="pub/base",
        preset=VirtualPreset(system_prompt="P."),
    )
    manager = _manager([base, virtual])
    assert manager.serving_record(virtual) is base
    assert manager.serving_record(base) is base


def test_virtual_model_with_adapters_keeps_its_own_instance() -> None:
    base = _record("pub/base")
    virtual = _record(
        "virtual/lora",
        is_virtual=True,
        base_model_id="pub/base",
        settings=ModelSettings(adapters=[AdapterAttachment(adapter_id="a")]),
    )
    manager = _manager([base, virtual])
    assert manager.serving_record(virtual) is virtual


def test_virtual_model_with_launch_override_keeps_its_own_instance() -> None:
    """ctx/kv overrides change the child argv, so no sharing."""
    base = _record("pub/base")
    virtual = _record(
        "virtual/bigctx",
        is_virtual=True,
        base_model_id="pub/base",
        settings=ModelSettings(ctx_size=32768),
    )
    manager = _manager([base, virtual])
    assert manager.serving_record(virtual) is virtual


def test_virtual_model_with_missing_base_falls_back_to_itself() -> None:
    virtual = _record("virtual/orphan", is_virtual=True, base_model_id="gone/base")
    manager = _manager([virtual])
    assert manager.serving_record(virtual) is virtual


async def test_ensure_loaded_routes_a_preset_virtual_to_the_loaded_base() -> None:
    """Naming the persona while the base is resident must not start a load."""

    class ReadySupervisor:
        def __init__(self, instance: InstanceInfo) -> None:
            self._instance = instance

        def get(self, model_id: str) -> InstanceInfo | None:
            return self._instance if model_id == self._instance.model_id else None

        def list(self) -> list[InstanceInfo]:
            return [self._instance]

    base = _record("pub/base")
    virtual = _record(
        "virtual/persona",
        is_virtual=True,
        base_model_id="pub/base",
        preset=VirtualPreset(temperature=0.2),
    )
    manager = _manager([base, virtual])
    instance = InstanceInfo(
        model_id="pub/base", state="ready", plan=LoadPlan(model_id="pub/base", devices=[0])
    )
    manager.supervisor = ReadySupervisor(instance)  # type: ignore[assignment]

    record, served = await manager.ensure_loaded("virtual/persona")
    assert record.id == "virtual/persona", "the caller-facing identity is the virtual model"
    assert served is instance, "the serving instance is the base's -- no second load"
    assert manager._locks == {}, "the fast path must not allocate a lock"


# ---------------------------------------------------------------------------
# The management route: POST /api/virtual-models with preset fields
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The real app over a real (tiny) GGUF library, no engine started."""
    import os

    from fastapi.testclient import TestClient

    from studioforge.api.app import create_app
    from studioforge.core.gpu import reset_probe
    from tests.unit.test_gguf import llm_kv, write_gguf

    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    reset_probe()

    models_dir = tmp_path / "models" / "pub" / "Tiny-GGUF"
    models_dir.mkdir(parents=True)
    write_gguf(
        models_dir / "Tiny-Q8_0.gguf",
        llm_kv(),
        [("blk.0.attn_q.weight", (64, 64), 8)],
    )
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    app = create_app(config, start_background=False)
    with TestClient(app) as client:
        yield client
    reset_probe()


def test_api_creates_a_full_preset_virtual_model(api_client: Any) -> None:
    base_id = "pub/Tiny-GGUF/Tiny-Q8_0"
    response = api_client.post(
        "/api/virtual-models",
        json={
            "id": "virtual/writer",
            "base_model_id": base_id,
            "name": "Writer",
            "system_prompt": "You write prose.",
            "temperature": 0.8,
            "top_k": 50,
            "max_tokens": 512,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["is_virtual"] is True
    assert body["preset"]["system_prompt"] == "You write prose."
    assert body["preset"]["temperature"] == 0.8
    assert body["preset"]["top_k"] == 50
    assert body["preset"]["max_tokens"] == 512
    # No launch-time overrides: settings stay default, so it shares the base.
    assert body["settings"]["ctx_size"] is None

    # Listed by both surfaces, exactly like any other model.
    v1 = api_client.get("/v1/models").json()["data"]
    entry = next(e for e in v1 if e["id"] == "virtual/writer")
    assert entry["state"] == "not-loaded"
    assert entry["studioforge"]["preset"]["system_prompt"] == "You write prose."
    v0 = api_client.get("/api/v0/models").json()["data"]
    assert any(e["id"] == "virtual/writer" for e in v0)


def test_api_ctx_override_becomes_a_launch_time_setting(api_client: Any) -> None:
    """ctx_size/kv_cache_type go through ModelSettings so the planner honours
    them -- and they cost the virtual model its own instance."""
    response = api_client.post(
        "/api/virtual-models",
        json={
            "id": "virtual/bigctx",
            "base_model_id": "pub/Tiny-GGUF/Tiny-Q8_0",
            "ctx_size": 16384,
            "kv_cache_type": "q8_0",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["settings"]["ctx_size"] == 16384
    assert body["settings"]["kv_cache_type"] == "q8_0"

    settings = api_client.get("/api/models/virtual/bigctx/settings").json()
    assert settings["ctx_size"] == 16384


def test_api_rejects_an_invalid_preset(api_client: Any) -> None:
    response = api_client.post(
        "/api/virtual-models",
        json={
            "id": "virtual/bad",
            "base_model_id": "pub/Tiny-GGUF/Tiny-Q8_0",
            "temperature": -3,
        },
    )
    assert response.status_code == 400
    assert "preset" in response.json()["error"]["message"]


def test_api_rejects_an_invalid_ctx_override(api_client: Any) -> None:
    response = api_client.post(
        "/api/virtual-models",
        json={
            "id": "virtual/bad-ctx",
            "base_model_id": "pub/Tiny-GGUF/Tiny-Q8_0",
            "ctx_size": -5,
        },
    )
    assert response.status_code == 400
