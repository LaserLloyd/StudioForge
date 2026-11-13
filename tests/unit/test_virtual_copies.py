"""Virtual-record correctness (WP1, D86): alias policy, resident guards,
launch fields on create, settings cascade, TTL via the serving id.

Covers the review defects A-4, A-5, A-6, A-7, A-16, A-17, A-18. A "copy" is a
virtual record with launch settings of its own (``base@b``), which gets its own
llama-server instance over the base's GGUF.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.types import InstanceInfo, LoadPlan

BASE = "pub/Tiny-GGUF/Tiny-Q8_0"
SHORT_ALIASES = [
    "Tiny-Q8_0",
    "pub/Tiny-Q8_0",
    "Tiny-GGUF/Tiny-Q8_0",
    "Tiny-GGUF",
    "pub/Tiny-GGUF",
    "Tiny",
    "pub/Tiny",
]


@pytest.fixture()
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from fastapi.testclient import TestClient

    from studioforge.api.app import create_app
    from studioforge.core.gpu import reset_probe
    from tests.unit.test_gguf import llm_kv, write_gguf

    for key in list(os.environ):
        if key.startswith("SF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SF_GPU_PROBE", "null")
    reset_probe()
    d = tmp_path / "models" / "pub" / "Tiny-GGUF"
    d.mkdir(parents=True)
    write_gguf(d / "Tiny-Q8_0.gguf", llm_kv(), [("blk.0.attn_q.weight", (64, 64), 8)])
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    app = create_app(config, start_background=False)
    with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000)) as client:
        yield client
    reset_probe()


def _mk(client: Any, vid: str, **extra: Any) -> dict[str, Any]:
    r = client.post("/api/virtual-models", json={"id": vid, "base_model_id": BASE, **extra})
    assert r.status_code == 200, r.text
    return r.json()


def _resident(client: Any, *model_ids: str) -> None:
    """Make the supervisor report these ids as resident instances."""
    sup = client.app.state.supervisor
    insts = {
        m: InstanceInfo(model_id=m, state="ready", plan=LoadPlan(model_id=m, devices=[0]))
        for m in model_ids
    }
    sup.get = lambda m: insts.get(m)  # type: ignore[method-assign]
    sup.list = lambda: list(insts.values())  # type: ignore[method-assign]


# --- B.9 / A-4: alias policy -----------------------------------------------------


@pytest.mark.parametrize("vid", ["a-copy", "A/early", BASE + "@b"])
def test_virtual_sorting_either_side_never_steals_base_aliases(api_client: Any, vid: str) -> None:
    reg = api_client.app.state.registry
    api_client.get("/api/models")  # let the boot scan finish
    known = [a for a in SHORT_ALIASES if reg.resolve(a) is not None] + [BASE]
    assert len(known) >= 5  # the repo-level forms are among them
    assert all(reg.resolve(a).id == BASE for a in known)
    _mk(api_client, vid)
    for alias in known:
        assert reg.resolve(alias).id == BASE, alias
    assert reg.resolve(vid).id == vid


def test_virtual_registers_only_its_id_and_name(api_client: Any) -> None:
    reg = api_client.app.state.registry
    _mk(api_client, "persona/x", name="Pirate")
    assert reg.resolve("persona/x").id == "persona/x"
    assert reg.resolve("pirate").id == "persona/x"
    # nothing repo-level is registered for it
    assert reg.resolve("Tiny-GGUF").id == BASE


def test_virtual_name_colliding_with_a_base_alias_loses(api_client: Any) -> None:
    reg = api_client.app.state.registry
    _mk(api_client, "persona/y", name="Tiny-GGUF")
    assert reg.resolve("Tiny-GGUF").id == BASE


@pytest.mark.parametrize("alias", ["Tiny-GGUF", "pub/Tiny-GGUF", "tiny-q8_0", BASE])
def test_create_with_an_existing_alias_is_409(api_client: Any, alias: str) -> None:
    r = api_client.post("/api/virtual-models", json={"id": alias, "base_model_id": BASE})
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "model_id_conflict"
    assert err["studioforge"]["colliding_model_id"] == BASE
    assert BASE in err["message"]
    assert err["fix"]
    reg = api_client.app.state.registry
    assert reg.resolve(alias).id == BASE and not reg.resolve(alias).is_virtual


def test_create_colliding_with_another_virtual_name_is_409(api_client: Any) -> None:
    _mk(api_client, "persona/z", name="Zed")
    r = api_client.post("/api/virtual-models", json={"id": "Zed", "base_model_id": BASE})
    assert r.status_code == 409 and r.json()["error"]["code"] == "model_id_conflict"


# --- A-5: base delete with a resident copy -----------------------------------------


def test_delete_base_with_resident_copy_is_409(api_client: Any) -> None:
    reg = api_client.app.state.registry
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    r = api_client.delete(f"/api/models/{BASE}", params={"delete_files": "true", "confirm": "true"})
    assert r.status_code == 409, r.text
    err = r.json()["error"]
    assert err["code"] == "model_loaded" and err["fix"]
    assert BASE + "@b" in err["message"]
    assert reg.get(BASE) is not None and reg.get(BASE + "@b") is not None
    assert Path(reg.get(BASE).path).exists()


def test_instance_holding_names_a_copy(api_client: Any) -> None:
    from studioforge.api import mgmt_routes

    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    assert mgmt_routes._instance_holding(api_client.app.state, BASE) == BASE + "@b"


def test_delete_base_without_resident_copy_removes_copy_and_settings_row(api_client: Any) -> None:
    state = api_client.app.state
    _mk(api_client, BASE + "@b", device_override=[2, 3])
    assert BASE + "@b" in state.db.all_model_settings()
    r = api_client.delete(f"/api/models/{BASE}")
    assert r.status_code == 200, r.text
    assert state.registry.get(BASE + "@b") is None
    # A-16: the copy's settings row went with it
    assert BASE + "@b" not in state.db.all_model_settings()


def test_recreated_copy_does_not_inherit_stale_settings(api_client: Any) -> None:
    state = api_client.app.state
    state.db.save_model_settings(BASE + "@b", {"device_override": [2, 3], "ctx_size": 4096})
    body = _mk(api_client, BASE + "@b")
    assert body["settings"]["device_override"] is None and body["settings"]["ctx_size"] is None
    assert state.db.all_model_settings().get(BASE + "@b") in (None, {})


# --- A-6: virtual delete while resident ---------------------------------------------


def test_delete_virtual_while_resident_is_409(api_client: Any) -> None:
    reg = api_client.app.state.registry
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    r = api_client.delete(f"/api/virtual-models/{BASE}@b")
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "model_loaded"
    assert reg.get(BASE + "@b") is not None


def test_delete_virtual_when_not_resident_still_works(api_client: Any) -> None:
    _mk(api_client, "persona/gone")
    r = api_client.delete("/api/virtual-models/persona/gone")
    assert r.status_code == 200
    assert api_client.app.state.registry.get("persona/gone") is None


def test_delete_riding_persona_while_base_is_resident_works(api_client: Any) -> None:
    _mk(api_client, "persona/rider", system_prompt="be brief")
    _resident(api_client, BASE)
    assert api_client.delete("/api/virtual-models/persona/rider").status_code == 200


def test_delete_virtual_via_models_route_while_resident_is_409(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    assert api_client.delete(f"/api/models/{BASE}@b").status_code == 409


# --- A-7: create accepts and validates the launch fields -----------------------------


def test_create_persists_every_launch_field(api_client: Any) -> None:
    from studioforge.core.manager import ModelManager

    body = _mk(
        api_client,
        BASE + "@b",
        ctx_size=8192,
        kv_cache_type="q8_0",
        kv_cache_type_v="q8_0",
        device_override=[2, 3],
        priority=1,
        parallel=2,
        pinned=False,
    )
    s = body["settings"]
    assert s["ctx_size"] == 8192 and s["kv_cache_type"] == "q8_0"
    assert s["kv_cache_type_v"] == "q8_0"
    assert s["device_override"] == [2, 3]
    assert s["priority"] == 1 and s["parallel"] == 2
    again = api_client.get(f"/api/models/{BASE}@b/settings").json()
    assert again["device_override"] == [2, 3] and again["kv_cache_type_v"] == "q8_0"
    assert again["priority"] == 1
    reg = api_client.app.state.registry
    copy = reg.get(BASE + "@b")
    serving = ModelManager.serving_record(SimpleNamespace(registry=reg), copy)
    assert serving is copy  # its own serving record
    # and it survives a rescan of the registry from SQLite
    row = api_client.app.state.db.all_model_settings()[BASE + "@b"]
    assert row["device_override"] == [2, 3]


def test_create_accepts_a_nested_settings_object(api_client: Any) -> None:
    body = _mk(api_client, "persona/n", settings={"ctx_size": 4096, "device_override": [1]})
    assert body["settings"]["ctx_size"] == 4096 and body["settings"]["device_override"] == [1]


def test_create_rejects_an_invalid_launch_value_and_leaves_nothing(api_client: Any) -> None:
    state = api_client.app.state
    r = api_client.post(
        "/api/virtual-models",
        json={"id": "persona/bad", "base_model_id": BASE, "ctx_size": "huge"},
    )
    assert r.status_code == 400, r.text
    assert state.registry.get("persona/bad") is None
    assert "persona/bad" not in state.db.all_model_settings()


def test_create_rejects_parallel_above_cap_and_leaves_nothing(api_client: Any) -> None:
    state = api_client.app.state
    r = api_client.post(
        "/api/virtual-models",
        json={"id": "persona/cap", "base_model_id": BASE, "parallel": 6, "max_parallel_cap": 2},
    )
    assert r.status_code == 400, r.text
    assert state.registry.get("persona/cap") is None


def test_create_names_fields_it_does_not_know(api_client: Any) -> None:
    body = _mk(api_client, "persona/q", ctx_size=2048, bogus_field=1)
    assert body["ignored_fields"] == ["bogus_field"]
    assert body["settings"]["ctx_size"] == 2048


def test_repost_of_a_resident_copy_is_409(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    r = api_client.post(
        "/api/virtual-models", json={"id": BASE + "@b", "base_model_id": BASE, "ctx_size": 1024}
    )
    assert r.status_code == 409, r.text
    assert api_client.app.state.registry.get(BASE + "@b").settings.ctx_size is None


def test_repost_of_a_non_resident_copy_replaces_it(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    body = _mk(api_client, BASE + "@b", ctx_size=1024, device_override=[1])
    assert body["settings"]["ctx_size"] == 1024 and body["settings"]["device_override"] == [1]


def test_preset_only_create_still_rides_the_base(api_client: Any) -> None:
    from studioforge.core.manager import ModelManager

    _mk(api_client, "persona/rides", system_prompt="hi", temperature=0.2)
    reg = api_client.app.state.registry
    rec = reg.get("persona/rides")
    assert ModelManager.serving_record(SimpleNamespace(registry=reg), rec).id == BASE


# --- A-18: revert-while-resident ------------------------------------------------------


def test_revert_resident_copy_to_defaults_is_409(api_client: Any) -> None:
    reg = api_client.app.state.registry
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    r = api_client.patch(f"/api/models/{BASE}@b/settings", json={"device_override": None})
    assert r.status_code == 409, r.text
    assert r.json()["error"]["code"] == "model_loaded"
    assert reg.get(BASE + "@b").settings.device_override == [0]


def test_put_default_settings_on_resident_copy_is_409(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    assert api_client.put(f"/api/models/{BASE}@b/settings", json={}).status_code == 409


def test_edit_that_keeps_own_server_on_resident_copy_is_fine(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    _resident(api_client, BASE + "@b")
    r = api_client.patch(f"/api/models/{BASE}@b/settings", json={"ctx_size": 2048})
    assert r.status_code == 200, r.text


def test_revert_when_not_resident_is_allowed(api_client: Any) -> None:
    _mk(api_client, BASE + "@b", device_override=[0])
    r = api_client.patch(f"/api/models/{BASE}@b/settings", json={"device_override": None})
    assert r.status_code == 200, r.text


# --- A-17: refresh_ttl / request_ttl_cap via the serving id ---------------------------


def test_refresh_ttl_reaches_the_base_instance_for_a_riding_persona(api_client: Any) -> None:
    mgr = api_client.app.state.manager
    _mk(api_client, "persona/rides", system_prompt="hi")
    inst = InstanceInfo(model_id=BASE, state="ready", plan=LoadPlan(model_id=BASE, devices=[0]))
    inst.ttl_s = 999
    api_client.app.state.supervisor.get = lambda m: inst if m == BASE else None  # type: ignore[method-assign]
    got = mgr.refresh_ttl("persona/rides")
    assert got == mgr.ttl_for(api_client.app.state.registry.get(BASE))
    assert inst.ttl_s == got != 999


def test_refresh_ttl_for_a_copy_with_its_own_instance_is_unchanged(api_client: Any) -> None:
    mgr = api_client.app.state.manager
    _mk(api_client, BASE + "@b", device_override=[0], ttl_s=77)
    inst = InstanceInfo(
        model_id=BASE + "@b", state="ready", plan=LoadPlan(model_id=BASE + "@b", devices=[0])
    )
    api_client.app.state.supervisor.get = (  # type: ignore[method-assign]
        lambda m: inst if m == BASE + "@b" else None
    )
    assert mgr.refresh_ttl(BASE + "@b") == 77 == inst.ttl_s


def test_request_ttl_cap_resolves_the_serving_instance(api_client: Any) -> None:
    mgr = api_client.app.state.manager
    mgr.config.models.ttl_by_priority = {1: 900, 2: 600, 3: 300}
    _mk(api_client, "persona/rides", system_prompt="hi")
    inst = InstanceInfo(model_id=BASE, state="ready", plan=LoadPlan(model_id=BASE, devices=[0]))
    inst.priority = 2
    api_client.app.state.supervisor.get = lambda m: inst if m == BASE else None  # type: ignore[method-assign]
    assert mgr.request_ttl_cap("persona/rides") == 600
