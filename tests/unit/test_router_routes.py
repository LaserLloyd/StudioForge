"""llama-server's router vocabulary on StudioForge (D90).

``GET /models`` (``status.value``), ``POST /models/load`` (with ``?dryRun=true``),
``POST /models/unload``, the router fields on ``/v1/models``, ``/health.product``,
the public brief capabilities and the KServe v2 probes. The manager's load /
plan / unload methods are replaced by recorders: what is under test is the
mapping from the router body onto the decision the existing paths already make.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.errors import InsufficientVramError, ModelBusyError
from tests.unit.native_stub import StubChild, build_app, instance, record

A = "vendor/alpha-Q4_K_M"
B = "vendor/beta-Q4_K_M"
V = "vendor/vision-Q4_K_M"


@pytest.fixture()
def child(monkeypatch: pytest.MonkeyPatch) -> StubChild:
    return StubChild().install(monkeypatch)


@pytest.fixture()
def app(tmp_path: Path, child: StubChild) -> Any:
    return build_app(
        tmp_path,
        [record(A), record(B), record(V, vision=True)],
        [instance(A), instance(B, state="loading")],
        child,
    )


class Calls:
    """Records the manager calls a router route makes, and answers like the real ones."""

    def __init__(self, app: Any, *, lease_support: bool = False) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        manager = app.state.manager

        async def load(name: str, **kwargs: Any) -> Any:
            self.calls.append(("load", (name,), kwargs))
            return instance(name)

        async def load_recommended(name: str, ctx_size: int, **kwargs: Any) -> Any:
            self.calls.append(("load_recommended", (name, ctx_size), kwargs))
            return instance(name, ctx=ctx_size)

        async def load_with_lease(name: str, *, lease_id: str | None = None, **kwargs: Any) -> Any:
            self.calls.append(("load", (name,), {"lease_id": lease_id, **kwargs}))
            return instance(name)

        def plan_preview(name: str, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("plan_preview", (name,), kwargs))
            return {"fits": True, "model_id": name, "kind": "preview"}

        async def plan_recommended(name: str, ctx_size: int, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(("plan_recommended", (name, ctx_size), kwargs))
            return {"fits": True, "model_id": name, "kind": "recommended", "ctx_size": ctx_size}

        async def unload(name: str, **kwargs: Any) -> bool:
            self.calls.append(("unload", (name,), kwargs))
            return name == A

        manager.load = load_with_lease if lease_support else load
        manager.load_recommended = load_recommended
        manager.plan_preview = plan_preview
        manager.plan_recommended = plan_recommended
        manager.unload = unload

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


def test_models_lists_every_model_with_the_router_status_vocabulary(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    rows = {row["id"]: row for row in body["data"]}
    assert set(rows) == {A, B, V}
    assert rows[A]["status"] == {"value": "loaded", "args": ["llama-server", "-c", "8192"]}
    assert rows[B]["status"] == {"value": "loading", "args": ["llama-server", "-c", "8192"]}
    assert rows[V]["status"] == {"value": "unloaded"}
    assert rows[A]["path"].endswith(".gguf")
    assert rows[A]["studioforge"]["state"] == "loaded"
    assert rows[B]["studioforge"]["state"] == "loading"
    assert rows[V]["studioforge"]["state"] == "not-loaded"
    assert rows[A]["studioforge"]["loaded_context_length"] == 8192
    assert rows[V]["studioforge"]["loaded_context_length"] is None


def test_models_architecture_is_the_modalities_object(app: Any) -> None:
    with TestClient(app) as http:
        rows = {row["id"]: row for row in http.get("/models").json()["data"]}
    assert rows[A]["architecture"] == {"input_modalities": ["text"], "output_modalities": ["text"]}
    assert rows[V]["architecture"] == {
        "input_modalities": ["text", "image"],
        "output_modalities": ["text"],
    }
    # StudioForge's own architecture string moved into its block.
    assert rows[A]["studioforge"]["arch"] == "unknown" or isinstance(
        rows[A]["studioforge"]["arch"], str
    )


def test_models_marks_an_unloaded_model_whose_last_launch_died(app: Any) -> None:
    app.state.manager.load_support_fields = lambda rec, memo=None: (
        {"last_load_failure": {"code": "model_load_failed", "message": "boom", "exit_code": 1}}
        if rec.id == V
        else {}
    )
    with TestClient(app) as http:
        rows = {row["id"]: row for row in http.get("/models").json()["data"]}
    assert rows[V]["status"] == {"value": "unloaded", "failed": True, "exit_code": 1}
    assert rows[A]["status"]["value"] == "loaded"
    assert "failed" not in rows[A]["status"]


class ScanResult:
    added: list[str] = []
    removed: list[str] = []
    unchanged = 0
    stale: list[str] = []
    errors: list[Any] = []
    duration_s = 0.0


def _count_scans(app: Any) -> list[bool]:
    scans: list[bool] = []

    def scan(*, force: bool = False) -> Any:
        scans.append(force)
        return ScanResult()

    app.state.registry.scan = scan
    return scans


def test_models_reload_rescans_the_library(app: Any) -> None:
    scans = _count_scans(app)
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        scans.clear()  # the boot scan is not under test
        plain = http.get("/models")
        assert scans == []
        reloaded = http.get("/models", params={"reload": "1"})
    assert plain.status_code == reloaded.status_code == 200
    assert scans == [False]


def test_models_reload_from_a_remote_caller_without_a_key_is_refused(app: Any) -> None:
    scans = _count_scans(app)
    with TestClient(app, client=("203.0.113.9", 50000)) as http:
        scans.clear()
        listed = http.get("/models")
        reloaded = http.get("/models", params={"reload": "1"})
    assert listed.status_code == 200
    assert reloaded.status_code == 403
    assert reloaded.json()["error"]["code"] == "remote_admin_requires_credential"
    assert scans == []


def test_models_reload_with_an_admin_key_is_allowed(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(A)], [], child, server={"api_key": "admin-key-0123456789"})
    scans = _count_scans(app)
    with TestClient(app, client=("203.0.113.9", 50000)) as http:
        scans.clear()
        reloaded = http.get(
            "/models",
            params={"reload": "1"},
            headers={"Authorization": "Bearer admin-key-0123456789"},
        )
    assert reloaded.status_code == 200
    assert scans == [False]


# ---------------------------------------------------------------------------
# /v1/models gains the router fields
# ---------------------------------------------------------------------------


def test_v1_models_carries_status_meta_architecture_and_loaded_instances(app: Any) -> None:
    with TestClient(app) as http:
        rows = {row["id"]: row for row in http.get("/v1/models").json()["data"]}
    loaded, loading, unloaded = rows[A], rows[B], rows[V]
    assert loaded["status"]["value"] == "loaded"
    assert loaded["status"]["args"] == ["llama-server", "-c", "8192"]
    assert loading["status"]["value"] == "loading"
    assert unloaded["status"] == {"value": "unloaded"}
    # The legacy spellings stay, with equal meaning.
    assert (loaded["state"], loading["state"], unloaded["state"]) == (
        "loaded",
        "loading",
        "not-loaded",
    )
    assert loaded["studioforge"]["state"] == "loaded"
    assert loaded["meta"] == {
        "n_vocab": 32000,
        "n_ctx_train": 32768,
        "n_embd": 4096,
        "n_params": 8_000_000_000,
        "size": 4_000_000_000,
    }
    assert loaded["studioforge"]["max_context_length"] == 32768
    assert loaded["studioforge"]["loaded_instances"] == [
        {"id": A, "config": {"context_length": 8192, "parallel": 2}, "context_length": 8192}
    ]
    assert unloaded["studioforge"]["loaded_instances"] == []
    assert unloaded["architecture"]["input_modalities"] == ["text", "image"]


def test_v1_models_loaded_filter_keeps_only_resident_models(app: Any) -> None:
    with TestClient(app) as http:
        everything = http.get("/v1/models").json()["data"]
        only_loaded = http.get("/v1/models", params={"loaded": "true"}).json()["data"]
        single = http.get(f"/v1/models/{A}").json()
    assert len(everything) == 3
    assert [row["id"] for row in only_loaded] == [A]
    assert single["status"]["value"] == "loaded"


# ---------------------------------------------------------------------------
# POST /models/load
# ---------------------------------------------------------------------------


def test_load_with_only_context_length_is_the_strict_walk(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        response = http.post(
            "/models/load",
            json={
                "model": V,
                "context_length": 16384,
                "min_slots": 2,
                "max_slots": 4,
                "kv_unified": True,
                "kv_min": "q8_0",
                "prefer_mode": "single",
                "allowed_devices": [0, 1],
                "priority": 2,
            },
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["success"] is True
    assert body["model_id"] == V
    assert calls.names() == ["load_recommended"]
    _, args, kwargs = calls.calls[0]
    assert args == (V, 16384)
    assert kwargs["min_slots"] == 2
    assert kwargs["max_slots"] == 4
    assert kwargs["kv_unified"] is True
    assert kwargs["kv_min"] == "q8_0"
    assert kwargs["prefer_modes"] == ["single"]
    assert kwargs["allowed_devices"] == [0, 1]
    assert kwargs["priority"] == 2
    assert kwargs["persist"] is False
    assert kwargs["source"].startswith("api:/models/load")


def test_load_with_explicit_placement_is_the_explicit_load(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        response = http.post(
            "/models/load",
            json={
                "model": V,
                "context_length": 4096,
                "devices": [1],
                "parallel": 3,
                "kv_cache_type": "q8_0",
                "kv_cache_type_v": "q4_0",
                "force": True,
            },
        )
    assert response.status_code == 200, response.text
    assert calls.names() == ["load"]
    _, args, kwargs = calls.calls[0]
    assert args == (V,)
    assert kwargs["ctx_size"] == 4096
    assert kwargs["devices"] == [1]
    assert kwargs["parallel"] == 3
    assert kwargs["kv_cache_type"] == "q8_0"
    assert kwargs["kv_cache_type_v"] == "q4_0"
    assert kwargs["force"] is True


def test_load_with_just_a_model_is_a_plain_load_and_ctx_size_is_accepted(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        plain = http.post("/models/load", json={"model": V})
        legacy = http.post("/models/load", json={"model": V, "ctx_size": 2048})
    assert plain.json() == {"success": True, **plain.json()}
    assert calls.names() == ["load", "load_recommended"]
    assert calls.calls[0][2]["ctx_size"] is None
    assert calls.calls[1][1] == (V, 2048)
    assert legacy.status_code == 200


def test_load_ignores_unknown_keys_like_llama_server(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        response = http.post("/models/load", json={"model": V, "sidecars": ["mmproj"], "x": 1})
    assert response.status_code == 200
    assert calls.names() == ["load"]


def test_load_rejects_mixed_shapes_and_missing_context(app: Any) -> None:
    Calls(app)
    with TestClient(app) as http:
        mixed = http.post(
            "/models/load",
            json={"model": V, "context_length": 4096, "devices": [0], "min_slots": 2},
        )
        orphan = http.post("/models/load", json={"model": V, "min_slots": 2})
        both = http.post(
            "/models/load", json={"model": V, "devices": [0], "allowed_devices": [0, 1]}
        )
    for response in (mixed, orphan, both):
        assert response.status_code == 400
    assert "min_slots" in mixed.json()["error"]["message"]
    assert orphan.json()["error"]["param"] == "context_length"


def test_load_needs_a_known_model(app: Any) -> None:
    Calls(app)
    with TestClient(app) as http:
        missing = http.post("/models/load", json={})
        unknown = http.post("/models/load", json={"model": "nope/none"})
    assert missing.status_code == 400
    assert missing.json()["error"]["code"] == "model_required"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"


def test_load_dry_run_is_the_walks_plan_and_starts_nothing(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        walk = http.post(
            "/models/load", params={"dryRun": "true"}, json={"model": V, "context_length": 16384}
        )
        explicit = http.post(
            "/models/load",
            params={"dryRun": "true"},
            json={"model": V, "devices": [0], "parallel": 2},
        )
    assert walk.status_code == explicit.status_code == 200
    assert walk.json() == {
        "fits": True,
        "model_id": V,
        "kind": "recommended",
        "ctx_size": 16384,
        "dry_run": True,
        "dry_run_of": "load_recommended",
    }
    assert explicit.json()["kind"] == "preview"
    assert explicit.json()["dry_run"] is True
    assert explicit.json()["dry_run_of"] == "load"
    assert calls.names() == ["plan_recommended", "plan_preview"]
    assert "success" not in walk.json()


def test_load_dry_run_false_really_loads(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        http.post("/models/load", params={"dryRun": "false"}, json={"model": V})
    assert calls.names() == ["load"]


def test_load_refusals_keep_their_codes(app: Any) -> None:
    async def refuse(name: str, ctx_size: int, **_kwargs: Any) -> Any:
        raise InsufficientVramError("does not fit", details={"max_ctx_that_fits": 8192})

    app.state.manager.load_recommended = refuse
    with TestClient(app) as http:
        response = http.post("/models/load", json={"model": V, "context_length": 999999})
    assert response.status_code == 507
    assert response.json()["error"]["studioforge"]["max_ctx_that_fits"] == 8192


def test_load_lease_id_is_passed_when_the_load_path_supports_it(app: Any) -> None:
    calls = Calls(app, lease_support=True)
    with TestClient(app) as http:
        response = http.post("/models/load", json={"model": V, "lease_id": "L1"})
    assert response.status_code == 200
    assert calls.calls[0][2]["lease_id"] == "L1"


def test_load_lease_id_reaches_the_walk_and_its_dry_run(app: Any) -> None:
    """A window-only body walks with the lease (D87): the load and ``?dryRun`` alike."""
    calls = Calls(app)
    with TestClient(app) as http:
        walked = http.post(
            "/models/load", json={"model": V, "context_length": 16384, "lease_id": "L1"}
        )
        dry = http.post(
            "/models/load",
            params={"dryRun": "true", "lease_id": "L1"},
            json={"model": V, "context_length": 16384},
        )
    assert walked.status_code == dry.status_code == 200
    assert calls.calls[0][0] == "load_recommended" and calls.calls[0][2]["lease_id"] == "L1"
    assert calls.calls[1][0] == "plan_recommended" and calls.calls[1][2]["lease_id"] == "L1"


def test_load_lease_id_sent_twice_must_agree(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        response = http.post(
            "/models/load", params={"lease_id": "L2"}, json={"model": V, "lease_id": "L1"}
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "lease_id"
    assert calls.calls == []


def test_load_force_is_an_explicit_load_in_the_dry_run_too(app: Any) -> None:
    """``force`` routes to the explicit load, and the dry run follows the same rule (D88)."""
    calls = Calls(app)
    with TestClient(app) as http:
        dry = http.post(
            "/models/load",
            params={"dryRun": "true"},
            json={"model": V, "context_length": 8192, "force": True},
        )
        real = http.post("/models/load", json={"model": V, "context_length": 8192, "force": True})
    assert dry.status_code == real.status_code == 200
    assert dry.json()["dry_run_of"] == "load"
    assert calls.names() == ["plan_preview", "load"]


def test_load_persist_is_an_admin_action(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(V)], [], child, server={"api_key": "admin-key-0123456789"})
    calls = Calls(app)
    with TestClient(app) as http:
        response = http.post(
            "/models/load",
            json={"model": V, "context_length": 4096, "persist": True},
            headers={"Authorization": "Bearer admin-key-0123456789"},
        )
    assert response.status_code == 200
    assert calls.calls[0][2]["persist"] is True


def test_load_busy_503_carries_retry_after(app: Any) -> None:
    async def busy(name: str, **_kwargs: Any) -> Any:
        raise ModelBusyError("busy", details={"retry_after_s": 7})

    app.state.manager.load = busy
    with TestClient(app) as http:
        response = http.post("/models/load", json={"model": V})
    assert response.status_code == 503
    assert response.headers["retry-after"] == "7"


# ---------------------------------------------------------------------------
# POST /models/unload
# ---------------------------------------------------------------------------


def test_unload_answers_success_true(app: Any) -> None:
    calls = Calls(app)
    with TestClient(app) as http:
        done = http.post("/models/unload", json={"model": A})
        idle = http.post("/models/unload", json={"model": V})
    assert done.json() == {"success": True, "model": A, "unloaded": True}
    assert idle.json() == {"success": True, "model": V, "unloaded": False}
    assert calls.names() == ["unload", "unload"]
    assert calls.calls[0][2]["source"] == "rest"


def test_unload_needs_a_known_model(app: Any) -> None:
    Calls(app)
    with TestClient(app) as http:
        assert http.post("/models/unload", json={}).status_code == 400
        assert http.post("/models/unload", json={"model": "nope/none"}).status_code == 404


# ---------------------------------------------------------------------------
# /health.product, brief capabilities, KServe v2
# ---------------------------------------------------------------------------


def test_health_says_what_answered(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/health").json()
    assert body["product"] == "StudioForge"
    assert body["api"] == {"openai": "v1", "studioforge": "v1", "capabilities": "/api/capabilities"}
    assert body["status"] == "ok"
    assert "version" in body


KEY = {"Authorization": "Bearer admin-key-0123456789"}


def test_brief_capabilities_is_public_and_carries_only_names_and_numbers(
    tmp_path: Path, child: StubChild
) -> None:
    app = build_app(tmp_path, [record(A)], [], child, server={"api_key": "admin-key-0123456789"})
    with TestClient(app) as http:
        full_anon = http.get("/api/capabilities")
        brief_anon = http.get("/api/capabilities", params={"brief": "true"})
        full_key = http.get("/api/capabilities", headers=KEY)
    assert full_anon.status_code == 401
    assert brief_anon.status_code == 200
    brief = brief_anon.json()
    assert set(brief) == {
        "product",
        "version",
        "latest_decision",
        "features",
        "decisions",
        "feature_decisions",
    }
    assert brief["product"] == "StudioForge"
    for feature in ("native_routes", "router_models_api", "identity_headers", "capabilities_brief"):
        assert feature in brief["features"]
        assert brief["feature_decisions"][feature] == "D90"
    assert "D1" in brief["decisions"]
    # The full report keeps its shape and its key requirement.
    assert full_key.status_code == 200
    assert "hardware" in full_key.json()
    assert "hardware" not in brief and "engine" not in brief


def test_brief_capabilities_can_be_switched_off(tmp_path: Path, child: StubChild) -> None:
    app = build_app(
        tmp_path,
        [record(A)],
        [],
        child,
        server={"api_key": "admin-key-0123456789", "public_capabilities_brief": False},
    )
    with TestClient(app) as http:
        assert http.get("/api/capabilities", params={"brief": "true"}).status_code == 401
        assert (
            http.get("/api/capabilities", params={"brief": "true"}, headers=KEY).status_code == 200
        )


def test_brief_false_is_not_the_public_form(tmp_path: Path, child: StubChild) -> None:
    app = build_app(tmp_path, [record(A)], [], child, server={"api_key": "admin-key-0123456789"})
    with TestClient(app) as http:
        assert http.get("/api/capabilities", params={"brief": "false"}).status_code == 401


def test_v2_probes_need_no_key(tmp_path: Path, child: StubChild) -> None:
    app = build_app(
        tmp_path,
        [record(A), record(B)],
        [instance(A)],
        child,
        server={"api_key": "admin-key-0123456789"},
    )
    with TestClient(app) as http:
        live = http.get("/v2/health/live")
        ready = http.get("/v2/health/ready")
        model_ready = http.get(f"/v2/models/{A}/ready")
        model_ready_keyed = http.get(f"/v2/models/{A}/ready", headers=KEY)
    assert live.status_code == 200
    assert live.json() == {"live": True}
    assert ready.status_code == 200
    assert ready.json() == {"ready": True}
    # A model route is not a probe of the server: it keeps its key.
    assert model_ready.status_code == 401
    assert model_ready_keyed.status_code == 200


def test_v2_server_ready_is_503_while_booting_or_draining(app: Any) -> None:
    class Boot:
        ready = False
        phase = "scanning"

    with TestClient(app) as http:
        app.state.boot = Boot()
        booting = http.get("/v2/health/ready")
        app.state.boot = None
        app.state.manager._draining = True
        draining = http.get("/v2/health/ready")
        live_while_draining = http.get("/v2/health/live")
        app.state.manager._draining = False
    assert booting.status_code == 503
    assert booting.json()["reason"] == "still starting (scanning)"
    assert draining.status_code == 503
    assert draining.json() == {"ready": False, "reason": "draining"}
    assert live_while_draining.status_code == 200


def test_v2_model_ready_follows_the_instance_state(app: Any) -> None:
    with TestClient(app) as http:
        ready = http.get(f"/v2/models/{A}/ready")
        loading = http.get(f"/v2/models/{B}/ready")
        unloaded = http.get(f"/v2/models/{V}/ready")
        unknown = http.get("/v2/models/nope/none/ready")
    assert ready.status_code == 200
    assert ready.json() == {"name": A, "ready": True}
    assert loading.status_code == 503
    assert loading.json() == {"name": B, "ready": False, "state": "loading"}
    assert unloaded.status_code == 503
    assert unloaded.json()["state"] == "unloaded"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
