"""LM Studio's native v1 REST API (``/api/v1``) -- W2.

Routes under test: ``GET /api/v1/models``, ``POST /api/v1/models/load`` and
``/unload``, ``POST /api/v1/models/download`` with ``GET .../download/status``,
and the 501 on ``POST /api/v1/chat``. The manager, supervisor and downloader
are fakes; what is pinned is the wire shape (against ``tests/fixtures/lmstudio``)
and that each route lands on the SAME manager method the management API uses.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.core.downloader import DownloadProgress
from studioforge.types import InstanceInfo
from tests.unit.compat.conftest import (
    EMBED_ID,
    Registry,
    assert_shape,
    instance_of,
    load_fixture,
)
from tests.unit.test_catalog_routes import MODEL_ID, make_record

LOOPBACK = ("127.0.0.1", 50000)


# ---------------------------------------------------------------------------
# GET /api/v1/models
# ---------------------------------------------------------------------------


def test_models_list_matches_the_documented_shape(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/v1/models")
    assert response.status_code == 200, response.text
    body = response.json()
    fixture = load_fixture("models_list_response.json")
    assert_shape(body, fixture)
    by_key = {m["key"]: m for m in body["models"]}
    assert set(by_key) >= {MODEL_ID, EMBED_ID}

    chat = by_key[MODEL_ID]
    assert chat["type"] == "llm"
    assert chat["state"] == "loaded"
    assert chat["loaded_instances"] == [
        {
            "id": MODEL_ID,
            "config": {"context_length": 16384, "parallel": 3, "offload_kv_cache_to_gpu": True},
            "context_length": 16384,
        }
    ]
    assert chat["max_context_length"] == 32768
    assert chat["params_string"] == "8B"
    assert chat["capabilities"] == {"vision": False, "trained_for_tool_use": True}
    assert chat["quantization"] == {"name": "Q4_K_M", "bits_per_weight": None}
    assert chat["format"] == "gguf"

    embed = by_key[EMBED_ID]
    assert embed["type"] == "embedding"
    assert embed["loaded_instances"] == []
    assert embed["state"] == "not-loaded"
    assert embed["params_string"] == "137M"


def test_models_list_mirrors_the_openai_list_keys_for_clients_that_read_data(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/v1/models").json()
    assert body["object"] == "list"
    assert body["data"] == body["models"]


def test_a_loading_model_reports_loading_and_no_instance_yet(app: Any) -> None:
    app.state.supervisor._instances = [InstanceInfo(model_id=MODEL_ID, state="loading")]
    with TestClient(app) as http:
        entry = next(m for m in http.get("/api/v1/models").json()["models"] if m["key"] == MODEL_ID)
    assert entry["state"] == "loading"
    assert entry["loaded_instances"] == []


def test_a_persona_reports_its_bases_instance(app: Any) -> None:
    """A preset-only virtual model is served by its base: same instance id."""
    persona = make_record("persona/helper").model_copy(
        update={"is_virtual": True, "base_model_id": MODEL_ID}
    )
    app.state.registry = Registry([make_record(), persona])
    app.state.manager.registry = app.state.registry
    with TestClient(app) as http:
        models = {m["key"]: m for m in http.get("/api/v1/models").json()["models"]}
    assert models["persona/helper"]["loaded_instances"][0]["id"] == MODEL_ID


def test_the_api_key_gate_applies_and_both_header_spellings_pass(app: Any) -> None:
    app.state.config.server.api_key = "sk-test"
    with TestClient(app) as http:
        assert http.get("/api/v1/models").status_code == 401
        bearer = http.get("/api/v1/models", headers={"Authorization": "Bearer sk-test"})
        assert bearer.status_code == 200
        header = http.get("/api/v1/models", headers={"x-api-key": "sk-test"})
        assert header.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/models/load
# ---------------------------------------------------------------------------


def _capture_load(app: Any) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    async def load(model_id: str, **kwargs: Any) -> InstanceInfo:
        seen.clear()
        seen.update(model_id=model_id, **kwargs)
        return instance_of(app, model_id)

    app.state.manager.load = load
    return seen


def test_load_lands_on_the_manager_load_method_the_management_api_uses(app: Any) -> None:
    seen = _capture_load(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json=load_fixture("load_request.json"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert_shape(body, load_fixture("load_response.json"))
    assert body["instance_id"] == MODEL_ID
    assert body["model"] == MODEL_ID
    assert body["context_length"] == 16384
    assert body["status"] == "loaded"
    assert body["ignored"] == ["flash_attention"]
    assert body["load_config"]["context_length"] == 16384

    assert seen["model_id"] == MODEL_ID
    assert seen["ctx_size"] == 16384, "context_length is the per-conversation ctx_size"
    assert seen["source"].startswith("api:/api/v1/models/load")
    assert "ttl" not in seen, "ttl is consumed here, never handed to the planner"


def test_load_without_options_asks_the_planner_for_its_own_default(app: Any) -> None:
    seen = _capture_load(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json={"model": MODEL_ID})
    assert response.status_code == 200, response.text
    assert seen["ctx_size"] is None
    assert seen["parallel"] is None
    assert seen["priority"] is None
    assert "load_config" not in response.json(), "only when echo_load_config is true"


def test_load_forwards_priority_and_parallel(app: Any) -> None:
    seen = _capture_load(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post(
            "/api/v1/models/load", json={"model": MODEL_ID, "priority": 2, "parallel": 2}
        )
    assert response.status_code == 200, response.text
    assert seen["priority"] == 2
    assert seen["parallel"] == 2


def test_load_reports_the_real_window_when_it_differs_from_the_request(app: Any) -> None:
    _capture_load(app)
    with TestClient(app, client=LOOPBACK) as http:
        body = http.post(
            "/api/v1/models/load", json={"model": MODEL_ID, "context_length": 4096}
        ).json()
    assert body["context_length"] == 16384
    assert "already loaded with context_length 16384" in body["warnings"][0]


@pytest.mark.parametrize(
    ("body", "param"),
    [
        ({}, "model"),
        ({"model": MODEL_ID, "context_length": 0}, "context_length"),
        ({"model": MODEL_ID, "context_length": "big"}, "context_length"),
        ({"model": MODEL_ID, "context_length": True}, "context_length"),
        ({"model": MODEL_ID, "parallel": -1}, "parallel"),
        ({"model": MODEL_ID, "offload_kv_cache_to_gpu": False}, "offload_kv_cache_to_gpu"),
        ({"model": MODEL_ID, "priority": 9}, "priority"),
    ],
)
def test_load_refuses_bad_bodies_naming_the_parameter(
    app: Any, body: dict[str, Any], param: str
) -> None:
    _capture_load(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json=body)
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == param


def test_load_of_an_unknown_model_is_the_standard_404(app: Any) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json={"model": "nope/none"})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_load_refuses_a_body_that_is_not_json_object(app: Any) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json=["x"])
    assert response.status_code == 400


def test_a_planner_refusal_surfaces_with_its_own_status_and_code(app: Any) -> None:
    from studioforge.errors import InsufficientVramError

    async def refuse(model_id: str, **_kwargs: Any) -> InstanceInfo:
        raise InsufficientVramError("needs 40 GB, 30 GB free", details={"required_bytes": 40})

    app.state.manager.load = refuse
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/load", json={"model": MODEL_ID})
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "insufficient_vram"
    assert response.json()["error"]["studioforge"]["required_bytes"] == 40


def test_load_is_open_on_an_open_install_like_the_management_load(app: Any) -> None:
    """Residency is the product (D32): a LAN peer may preload, as with /api/models/{id}/load."""
    _capture_load(app)
    with TestClient(app, client=("192.168.1.50", 50000)) as lan:
        assert lan.post("/api/v1/models/load", json={"model": MODEL_ID}).status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/models/unload
# ---------------------------------------------------------------------------


def _capture_unload(app: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def unload(name: str, **kwargs: Any) -> bool:
        calls.append({"name": name, **kwargs})
        app.state.supervisor.remove(name)
        return True

    app.state.manager.unload = unload
    return calls


def test_unload_by_instance_id(app: Any) -> None:
    calls = _capture_unload(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/unload", json=load_fixture("unload_request.json"))
    assert response.status_code == 200, response.text
    assert_shape(response.json(), load_fixture("unload_response.json"))
    assert response.json() == {"instance_id": MODEL_ID}
    assert calls[0]["name"] == MODEL_ID
    assert calls[0]["source"] == "rest"


def test_unload_accepts_model_as_an_alias_for_instance_id(app: Any) -> None:
    calls = _capture_unload(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/unload", json={"model": MODEL_ID})
    assert response.status_code == 200
    assert calls[0]["name"] == MODEL_ID


def test_unloading_a_model_that_is_not_loaded_is_a_404_not_a_silent_success(app: Any) -> None:
    calls = _capture_unload(app)
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/unload", json={"instance_id": EMBED_ID})
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "instance_not_found"
    assert calls == []


def test_unloading_an_unknown_model_and_an_empty_body(app: Any) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        unknown = http.post("/api/v1/models/unload", json={"instance_id": "nope/none"})
        empty = http.post("/api/v1/models/unload", json={})
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"
    assert empty.status_code == 400
    assert empty.json()["error"]["param"] == "instance_id"


def test_unloading_a_persona_unloads_its_bases_instance(app: Any) -> None:
    calls = _capture_unload(app)
    persona = make_record("persona/helper").model_copy(
        update={"is_virtual": True, "base_model_id": MODEL_ID}
    )
    app.state.registry = Registry([make_record(), persona])
    app.state.manager.registry = app.state.registry
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/unload", json={"model": "persona/helper"})
    assert response.status_code == 200
    assert response.json() == {"instance_id": MODEL_ID}
    assert calls[0]["name"] == MODEL_ID


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

GIB = 1024**3


class FakeDownloader:
    """The five methods the v1 routes (and ``start_download``) touch."""

    def __init__(self) -> None:
        self.progress: dict[str, list[DownloadProgress]] = {}
        self.enqueued: list[Any] = []
        self.complete_on_enqueue = False

    def _file(self, status: str, done: int, total: int = 20 * GIB, speed: float = 0.0) -> Any:
        return DownloadProgress(
            id="g-qwen:q.gguf",
            group_id="g-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            filename="q.gguf",
            status=status,  # type: ignore[arg-type]
            downloaded_bytes=done,
            total_bytes=total,
            speed_bps=speed,
            eta_s=None,
            error=None,
        )

    async def enqueue(self, item: Any, *, include_mmproj: bool = True, force: bool = False) -> str:
        self.enqueued.append(item)
        if self.complete_on_enqueue:
            self.progress["g-qwen"] = [self._file("completed", 20 * GIB)]
        else:
            self.progress["g-qwen"] = [self._file("queued", 0)]
        return "g-qwen"

    def group(self, group_id: str) -> list[DownloadProgress]:
        return list(self.progress.get(group_id, []))

    def group_status(self, group_id: str) -> str:
        items = self.progress.get(group_id, [])
        return items[0].status if items else "queued"

    def all(self) -> list[DownloadProgress]:
        return [p for items in self.progress.values() for p in items]


class Choice:
    group_id = "g-qwen"
    repo_id = "unsloth/Qwen3.8-27B-GGUF"
    quant = "Q4_K_M"
    total_bytes = 20 * GIB
    mmproj = None
    files = [type("F", (), {"filename": "q.gguf"})()]


@pytest.fixture()
def downloads(app: Any, monkeypatch: pytest.MonkeyPatch) -> FakeDownloader:
    downloader = FakeDownloader()
    app.state.downloader = downloader
    resolved: list[tuple[str, str | None]] = []

    async def fake_resolve(
        config: Any, planner: Any, repo_id: str, quant: str | None = None
    ) -> Any:
        resolved.append((repo_id, quant))
        return Choice()

    monkeypatch.setattr("studioforge.core.downloader.resolve_download_choice", fake_resolve)
    downloader.resolved = resolved  # type: ignore[attr-defined]
    return downloader


def test_download_queues_the_hf_repo_and_answers_in_lm_studios_shape(
    app: Any, downloads: FakeDownloader
) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/download", json=load_fixture("download_request.json"))
    assert response.status_code == 200, response.text
    body = response.json()
    assert_shape(body, load_fixture("download_response.json"))
    assert body["job_id"] == "g-qwen"
    assert body["status"] == "downloading"
    assert body["total_size_bytes"] == 20 * GIB
    assert downloads.resolved == [("unsloth/Qwen3.8-27B-GGUF", "Q4_K_M")]  # type: ignore[attr-defined]
    assert len(downloads.enqueued) == 1


@pytest.mark.parametrize(
    ("model", "quant", "want"),
    [
        ("unsloth/Qwen3.8-27B-GGUF", None, ("unsloth/Qwen3.8-27B-GGUF", None)),
        ("hf.co/unsloth/Qwen3.8-27B-GGUF:Q5_K_M", None, ("unsloth/Qwen3.8-27B-GGUF", "Q5_K_M")),
        (
            "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/q.gguf?download=true",
            "Q8_0",
            ("unsloth/Qwen3.8-27B-GGUF", "Q8_0"),
        ),
    ],
)
def test_download_target_parsing(
    app: Any, downloads: FakeDownloader, model: str, quant: str | None, want: tuple[str, str | None]
) -> None:
    body: dict[str, Any] = {"model": model}
    if quant:
        body["quantization"] = quant
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/download", json=body)
    assert response.status_code == 200, response.text
    assert downloads.resolved[-1] == want  # type: ignore[attr-defined]


@pytest.mark.parametrize("model", ["openai", "https://example.com/model.gguf", "", None, 7])
def test_download_refuses_what_it_cannot_resolve(
    app: Any, downloads: FakeDownloader, model: Any
) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/download", json={"model": model})
    assert response.status_code == 400, response.text
    assert response.json()["error"]["param"] == "model"
    assert downloads.enqueued == []


def test_download_of_a_model_already_on_disk(app: Any, downloads: FakeDownloader) -> None:
    downloads.complete_on_enqueue = True
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/download", json={"model": "unsloth/Qwen3.8-27B-GGUF"})
    assert response.status_code == 200
    assert response.json() == load_fixture("download_already_response.json")


def test_download_status_reports_progress_then_completion(
    app: Any, downloads: FakeDownloader
) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        http.post("/api/v1/models/download", json={"model": "unsloth/Qwen3.8-27B-GGUF"})
        downloads.progress["g-qwen"] = [downloads._file("running", 4 * GIB, speed=50_000_000.0)]
        running = http.get("/api/v1/models/download/status/g-qwen")
        assert running.status_code == 200, running.text
        assert_shape(running.json(), load_fixture("download_status_response.json"))
        assert running.json()["status"] == "downloading"
        assert running.json()["downloaded_bytes"] == 4 * GIB
        assert running.json()["bytes_per_second"] == 50_000_000

        downloads.progress["g-qwen"] = [downloads._file("completed", 20 * GIB)]
        done = http.get("/api/v1/models/download/status/g-qwen")
        assert_shape(done.json(), load_fixture("download_status_completed_response.json"))
        assert done.json()["status"] == "completed"
        assert "completed_at" in done.json()
        assert "bytes_per_second" not in done.json()


@pytest.mark.parametrize(
    ("file_status", "job_status"),
    [("paused", "paused"), ("failed", "failed"), ("canceled", "failed"), ("queued", "downloading")],
)
def test_download_status_folds_studioforge_states_onto_lm_studios(
    app: Any, downloads: FakeDownloader, file_status: str, job_status: str
) -> None:
    downloads.progress["g-qwen"] = [downloads._file(file_status, GIB)]
    with TestClient(app, client=LOOPBACK) as http:
        body = http.get("/api/v1/models/download/status/g-qwen").json()
    assert body["status"] == job_status


def test_download_status_without_a_job_id_lists_every_job(
    app: Any, downloads: FakeDownloader
) -> None:
    downloads.progress["g-qwen"] = [downloads._file("running", GIB, speed=1.0)]
    with TestClient(app, client=LOOPBACK) as http:
        body = http.get("/api/v1/models/download/status").json()
    assert [job["job_id"] for job in body["jobs"]] == ["g-qwen"]


def test_download_status_of_an_unknown_job(app: Any, downloads: FakeDownloader) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        response = http.get("/api/v1/models/download/status/job_nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "download_not_found"


def test_downloads_are_unavailable_without_a_downloader(app: Any) -> None:
    app.state.downloader = None
    with TestClient(app, client=LOOPBACK) as http:
        response = http.post("/api/v1/models/download", json={"model": "a/b"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "downloads_unavailable"


def test_the_download_post_is_an_admin_mutation_like_post_api_downloads(
    app: Any, downloads: FakeDownloader
) -> None:
    """A write to the box's disk: a LAN peer without a credential is refused, a local caller
    passes -- and the status read stays open, like ``GET /api/downloads``."""
    from studioforge.api.auth import is_admin_mutation

    assert is_admin_mutation("POST", "/api/downloads")
    assert is_admin_mutation("POST", "/api/v1/models/download")
    assert not is_admin_mutation("GET", "/api/v1/models/download/status/x")
    assert not is_admin_mutation("POST", "/api/v1/models/load")
    assert not is_admin_mutation("POST", "/api/v1/models/unload")

    with TestClient(app, client=("192.168.1.50", 50000)) as lan:
        refused = lan.post("/api/v1/models/download", json={"model": "unsloth/Qwen3.8-27B-GGUF"})
        status = lan.get("/api/v1/models/download/status")
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "remote_admin_requires_credential"
    assert downloads.enqueued == []
    assert status.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/v1/chat, unknown routes
# ---------------------------------------------------------------------------


def test_stateful_chat_is_a_501_with_a_clear_body(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/api/v1/chat", json={"model": MODEL_ID, "input": "hi"})
    assert response.status_code == 501
    assert response.json() == load_fixture("chat_501_response.json")
    assert response.json()["error"]["code"] == "not_implemented"
    assert "/v1/chat/completions" in response.json()["error"]["message"]


def test_an_unknown_api_v1_route_is_a_json_404(app: Any) -> None:
    with TestClient(app) as http:
        got = http.get("/api/v1/unknown")
        posted = http.post("/api/v1/unknown", json={})
        wrong_method = http.get("/api/v1/models/load")
    for response in (got, posted):
        assert response.status_code == 404
        assert response.headers["content-type"].startswith("application/json")
        assert response.json()["error"]["type"] == "invalid_request_error"
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["message"]
