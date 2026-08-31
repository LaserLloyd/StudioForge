"""The HTTP surfaces the catalog added, exercised through the real app.

Three code paths that the MCP tests do not reach, because they are wiring
rather than logic and wiring is exactly where a rename goes unnoticed:

* ``GET /api/catalog`` and its query parameters,
* ``/api/status`` gaining each loaded model's ``requests_deferred``,
* ``/v1/models`` gaining ``ctx_per_slot`` / ``max_parallel`` for loaded models,
* ``GET /api/models`` and ``POST .../load-recommended`` gaining the D48 load
  tier and the ``max_slots`` / ``persist`` knobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import throughput
from studioforge.errors import BadRequestError
from studioforge.types import (
    GB,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
    VirtualPreset,
)

MODEL_ID = "vendor/thing-Q4_K_M"


def make_record(model_id: str = MODEL_ID) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name="thing",
        path=Path("/models/thing.gguf"),
        size_bytes=8 * GB,
        quant="Q4_K_M",
        architecture="llama",
        mtime=1_755_000_000.0,
        added_at=1_755_000_000.0,
        capabilities=ModelCapabilities(tools=True),
        meta=GgufMeta(
            architecture="llama",
            n_layer=32,
            n_head=32,
            n_head_kv=8,
            n_embd=4096,
            n_embd_head_k=128,
            n_embd_head_v=128,
            n_ctx_train=32768,
            param_count=8_000_000_000,
            tensor_bytes=8 * GB,
            quant_label="Q4_K_M",
        ),
    )


def make_plan(model_id: str = MODEL_ID) -> LoadPlan:
    return LoadPlan(
        model_id=model_id,
        devices=[0],
        ctx_size=16384,
        parallel=3,
        ctx_per_slot=16384,
        max_parallel=3,
        parallel_limited_by="knee",
        kv_cache_type="f16",
    )


class FakeRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = records

    def all(self) -> list[ModelRecord]:
        return list(self._records)

    def get(self, model_id: str) -> ModelRecord | None:
        return next((r for r in self._records if r.id == model_id), None)

    def resolve(self, name: str) -> ModelRecord | None:
        return self.get(name)

    def known_ids(self) -> list[str]:
        return [r.id for r in self._records]

    def openai_list(self) -> list[dict[str, Any]]:
        return [r.openai_dict() for r in self._records]

    def scan(self, *, force: bool = False) -> Any:  # pragma: no cover - lifespan only
        raise RuntimeError("not used")

    def save_settings(self, model_id: str, settings: Any) -> ModelRecord:
        record = self.get(model_id)
        assert record is not None
        # Mirrors the real registry's D48 guard: a preset-only persona shares
        # its base's llama-server, and a saved tier is a launch-time setting --
        # writing one would silently split the persona onto a child of its own,
        # which is the opposite of what "give my persona priority 1" means. The
        # guard itself belongs to the registry's own suite; what the route test
        # below pins is that PATCH/PUT surface this refusal as a clean 400.
        if (
            record.is_virtual
            and record.base_model_id is not None
            and getattr(settings, "priority", None) is not None
        ):
            raise BadRequestError(
                f"'{model_id}' is served by its base model '{record.base_model_id}', "
                f"so a priority cannot be saved on it -- save it on the base id "
                f"instead, or give the persona launch settings of its own.",
                param="priority",
            )
        updated = record.model_copy(update={"settings": settings})
        self._records = [updated if r.id == model_id else r for r in self._records]
        return updated


class FakeSupervisor:
    def __init__(self, instances: list[InstanceInfo]) -> None:
        self._instances = instances

    def list(self) -> list[InstanceInfo]:
        return list(self._instances)

    def get(self, model_id: str) -> InstanceInfo | None:
        return next((i for i in self._instances if i.model_id == model_id), None)


class FakeProbe:
    backend = "fake"

    def available(self) -> bool:
        return True

    def list_gpus(self) -> list[GpuInfo]:
        return [
            GpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 5090",
                total_bytes=32 * GB,
                free_bytes=30 * GB,
                used_bytes=2 * GB,
                compute_capability=(12, 0),
            )
        ]

    def get_gpu(self, index: int) -> GpuInfo | None:
        return self.list_gpus()[0] if index == 0 else None

    def compute_processes(self) -> list[Any]:
        return []

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


@pytest.fixture()
def app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        server={"host": "127.0.0.1", "port": 1234},
        models={"dir": tmp_path / "models"},
        gui={"enabled": False},
        watchdog={"enabled": False},
        logging={"level": "ERROR"},
    )
    built = create_app(config, state=build_state(config), start_background=False)
    built.state.registry = FakeRegistry([make_record()])
    built.state.probe = FakeProbe()
    built.state.planner.probe = FakeProbe()
    built.state.manager.registry = built.state.registry
    return built


def loaded(app: Any, instance: InstanceInfo) -> None:
    supervisor = FakeSupervisor([instance])
    app.state.supervisor = supervisor
    app.state.manager.supervisor = supervisor


# ---------------------------------------------------------------------------
# GET /api/catalog
# ---------------------------------------------------------------------------


def test_catalog_endpoint_returns_models_with_options(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/catalog").json()

    assert body["count"] == 1
    assert "recommended" in body["catalog_hint"]
    entry = body["models"][0]
    assert entry["id"] == MODEL_ID
    assert entry["downloaded_at"].endswith("Z")
    assert entry["options"]
    assert sum(1 for r in entry["options"] if r["best_now"]) == 1


def test_catalog_compact_keeps_only_the_best_now_row(app: Any) -> None:
    with TestClient(app) as http:
        full = http.get("/api/catalog").json()
        compact = http.get("/api/catalog", params={"compact": 1}).json()

    assert compact["compact"] is True
    assert len(compact["models"][0]["options"]) == 1
    assert len(full["models"][0]["options"]) > 1


def test_catalog_filters_to_one_model(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/catalog", params={"model": MODEL_ID}).json()
    assert [m["id"] for m in body["models"]] == [MODEL_ID]


def test_catalog_404s_on_an_unknown_model(app: Any) -> None:
    with TestClient(app) as http:
        response = http.get("/api/catalog", params={"model": "nope/missing"})
    assert response.status_code == 404


def test_catalog_refresh_rebuilds_rather_than_serving_the_cache(app: Any) -> None:
    """`fits` is a claim about this instant, so a client must be able to force it.

    Proved by changing the library underneath: the cached response must NOT see
    the new model and the refreshed one must.
    """
    with TestClient(app) as http:
        first = http.get("/api/catalog").json()
        assert first["count"] == 1

        second = make_record("vendor/newer-Q8_0")
        second.mtime = 1_755_999_999.0
        app.state.registry = FakeRegistry([make_record(), second])
        app.state.manager.registry = app.state.registry

        cached = http.get("/api/catalog").json()
        refreshed = http.get("/api/catalog", params={"refresh": 1}).json()

    assert cached["count"] == 1, "a cache that rebuilds on every call is not a cache"
    assert refreshed["count"] == 2
    # And the newer model sorts first, through the endpoint as well.
    assert refreshed["models"][0]["id"] == "vendor/newer-Q8_0"


def test_catalog_rows_carry_load_args_the_load_endpoint_accepts(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/catalog", params={"compact": 1}).json()
    args = body["models"][0]["options"][0]["load_args"]
    assert set(args) == {"model_id", "ctx_size", "parallel", "kv_cache_type"}
    assert args["model_id"] == MODEL_ID


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------


def test_status_reports_requests_deferred_per_loaded_model(app: Any) -> None:
    """The only place an overloaded-but-healthy server shows up (D17)."""
    instance = InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())
    loaded(app, instance)
    app.state.manager._throughput_gauges = {
        MODEL_ID: {
            "sampled_at": 1_755_000_100.0,
            throughput.METRIC_REQUESTS_DEFERRED: 4.0,
            throughput.METRIC_REQUESTS_PROCESSING: 3.0,
            throughput.METRIC_BUSY_SLOTS: 2.5,
        }
    }

    with TestClient(app) as http:
        body = http.get("/api/status").json()

    entry = body["loaded"][0]
    assert entry["requests_deferred"] == 4.0
    assert entry["requests_processing"] == 3.0
    assert entry["busy_slots_per_decode"] == 2.5
    assert entry["max_parallel"] == 3
    assert entry["ctx_per_slot"] == 16384


def test_status_reports_nulls_before_the_first_scrape(app: Any) -> None:
    """A model loaded seconds ago has no sample yet; that is not an error."""
    instance = InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())
    loaded(app, instance)

    with TestClient(app) as http:
        body = http.get("/api/status").json()

    entry = body["loaded"][0]
    assert entry["requests_deferred"] is None
    assert entry["metrics_sampled_at"] is None
    # The plan-derived fields do not depend on a scrape.
    assert entry["max_parallel"] == 3


# ---------------------------------------------------------------------------
# /v1/models
# ---------------------------------------------------------------------------


def test_v1_models_exposes_concurrency_for_a_loaded_model(app: Any) -> None:
    """--ctx-size is the total across slots, so ctx_per_slot is the honest field."""
    instance = InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())
    loaded(app, instance)

    with TestClient(app) as http:
        body = http.get("/v1/models").json()

    entry = body["data"][0]
    assert entry["state"] == "loaded"
    assert entry["loaded_context_length"] == 16384
    assert entry["studioforge"]["ctx_per_slot"] == 16384
    assert entry["studioforge"]["max_parallel"] == 3
    assert entry["studioforge"]["parallel"] == 3
    assert entry["studioforge"]["parallel_limited_by"] == "knee"


def test_v1_models_stays_quiet_about_concurrency_when_nothing_is_loaded(app: Any) -> None:
    loaded(app, InstanceInfo(model_id="other/model", state="ready"))

    with TestClient(app) as http:
        body = http.get("/v1/models").json()

    entry = body["data"][0]
    assert entry["state"] == "not-loaded"
    assert "max_parallel" not in entry["studioforge"]


def test_v1_single_model_carries_the_same_runtime_fields_as_the_list(app: Any) -> None:
    """The same model must not read ``loaded`` from the list and nothing here."""
    instance = InstanceInfo(model_id=MODEL_ID, state="ready", port=18100, plan=make_plan())
    loaded(app, instance)

    with TestClient(app) as http:
        list_entry = http.get("/v1/models").json()["data"][0]
        single = http.get(f"/v1/models/{MODEL_ID}").json()

    assert single == list_entry


def test_v1_single_model_reports_not_loaded_when_cold(app: Any) -> None:
    with TestClient(app) as http:
        single = http.get(f"/v1/models/{MODEL_ID}").json()

    assert single["state"] == "not-loaded"
    assert single["studioforge"]["state"] == "not-loaded"


# ---------------------------------------------------------------------------
# GPU leases (D43)
# ---------------------------------------------------------------------------


def test_lease_routes_round_trip(app: Any) -> None:
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        created = http.post("/api/leases", json={"devices": [0], "reason": "test"})
        assert created.status_code == 200, created.text
        lease = created.json()
        assert lease["devices"] == [0] and lease["holder"] == "api" and lease["model_ids"] == []

        listed = http.get("/api/leases").json()
        assert listed["count"] == 1 and listed["leases"][0]["id"] == lease["id"]

        clash = http.post("/api/leases", json={"devices": [0]})
        assert clash.status_code == 409
        assert clash.json()["error"]["code"] == "lease_conflict"

        touched = http.post(f"/api/leases/{lease['id']}/touch")
        assert touched.status_code == 200

        released = http.delete(f"/api/leases/{lease['id']}")
        assert released.status_code == 200
        assert released.json()["released"]["id"] == lease["id"]
        assert http.get("/api/leases").json()["count"] == 0

        missing = http.delete(f"/api/leases/{lease['id']}")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "lease_not_found"


def test_lease_routes_validate_the_body(app: Any) -> None:
    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        empty = http.post("/api/leases", json={"devices": []})
        assert empty.status_code == 400
        assert empty.json()["error"]["param"] == "devices"
        bad_ttl = http.post("/api/leases", json={"devices": [0], "idle_ttl_s": 0})
        assert bad_ttl.status_code == 400


# ---------------------------------------------------------------------------
# The WP19 routes
# ---------------------------------------------------------------------------


def test_load_recommended_refuses_a_context_past_the_trained_window(app: Any) -> None:
    """A 400 naming the number that would work, not an attempt that fails later."""
    with TestClient(app) as http:
        response = http.post(
            f"/api/models/{MODEL_ID}/load-recommended", json={"ctx_size": 4_000_000}
        )
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["param"] == "ctx_size"
    assert "trained to" in body["message"]


def test_load_recommended_rejects_a_nonsense_context_without_planning(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(f"/api/models/{MODEL_ID}/load-recommended", json={"ctx_size": 0})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "ctx_size"


def test_load_recommended_404s_on_an_unknown_model(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post("/api/models/nope/load-recommended", json={"ctx_size": 8192})
    assert response.status_code == 404


def test_load_recommended_forwards_every_body_knob_to_the_manager(app: Any) -> None:
    """Wiring, which is exactly where a rename goes unnoticed: ``kv_min``,
    ``priority``, ``max_slots`` and ``persist`` are all optional body fields
    and all have to arrive as the manager's keyword arguments (D48)."""
    seen: dict[str, Any] = {}

    async def capture(model_id: str, ctx_size: int, **kwargs: Any) -> InstanceInfo:
        seen.clear()
        seen.update(model_id=model_id, ctx_size=ctx_size, **kwargs)
        return InstanceInfo(model_id=model_id, state="ready", port=18100, plan=make_plan())

    app.state.manager.load_recommended = capture

    with TestClient(app, client=("127.0.0.1", 50000)) as http:
        full = http.post(
            f"/api/models/{MODEL_ID}/load-recommended",
            json={
                "ctx_size": 16384,
                "kv_min": "q8_0",
                "priority": 2,
                "max_slots": 3,
                "persist": True,
            },
        )
        assert full.status_code == 200, full.text
        assert seen["ctx_size"] == 16384
        assert seen["kv_min"] == "q8_0"
        assert seen["priority"] == 2
        assert seen["max_slots"] == 3
        assert seen["persist"] is True

        # Omitted, they are the pre-D48 defaults -- an existing client's body
        # means exactly what it meant before.
        plain = http.post(f"/api/models/{MODEL_ID}/load-recommended", json={"ctx_size": 16384})
        assert plain.status_code == 200, plain.text
        assert seen["kv_min"] is None
        assert seen["priority"] is None
        assert seen["max_slots"] is None
        assert seen["persist"] is False


# ---------------------------------------------------------------------------
# GET /api/models: the tier beside the TTL
# ---------------------------------------------------------------------------


def test_api_models_reports_the_tier_each_model_would_load_at(app: Any) -> None:
    """Same soft-answer caveat as ``effective_ttl_s``: it is what the *next*
    load would use, and a request may still name its own tier (D46/D48)."""
    record = app.state.registry.get(MODEL_ID)
    assert record is not None

    with TestClient(app) as http:
        assert http.get("/api/models").json()["models"][0]["priority"] == 3

        # The saved setting is the floor a restart falls back to...
        record.settings = ModelSettings(priority=2)
        assert http.get("/api/models").json()["models"][0]["priority"] == 2

        # ...and a running instance is the truth about what is serving.
        loaded(
            app,
            InstanceInfo(
                model_id=MODEL_ID, state="ready", port=18100, priority=1, plan=make_plan()
            ),
        )
        row = http.get("/api/models").json()["models"][0]

    assert row["loaded"] is True
    assert row["priority"] == 1


def test_v1_models_carries_the_tier_only_for_a_loaded_model(app: Any) -> None:
    """An unloaded row's tier is a prediction, so it is left to GET /api/models
    rather than put in the OpenAI vendor block beside real runtime facts."""
    with TestClient(app) as http:
        cold = http.get("/v1/models").json()["data"][0]
        assert "priority" not in cold["studioforge"]

        loaded(
            app,
            InstanceInfo(
                model_id=MODEL_ID, state="ready", port=18100, priority=2, plan=make_plan()
            ),
        )
        warm = http.get("/v1/models").json()["data"][0]

    assert warm["studioforge"]["priority"] == 2
    # ...and no new top-level OpenAI key was invented for it.
    assert "priority" not in warm


def test_parallel_observations_start_empty_and_are_readable(app: Any) -> None:
    """The rows behind recommended_parallel, so a caller can see the evidence."""
    with TestClient(app) as http:
        response = http.get(f"/api/models/{MODEL_ID}/parallel-observations")
    assert response.status_code == 200
    assert response.json() == {"model_id": MODEL_ID, "observations": []}


def test_parallel_observations_answer_empty_when_the_table_cannot_be_read(app: Any) -> None:
    """A data directory that predates migration 005 is a [] here, not a 500."""

    class NoTable:
        def parallel_observations(self, *a: Any, **k: Any) -> list[dict[str, Any]]:
            raise RuntimeError("no such table: parallel_observations")

    with TestClient(app) as http:
        app.state.manager.db = NoTable()
        response = http.get(f"/api/models/{MODEL_ID}/parallel-observations")
    assert response.status_code == 200
    assert response.json()["observations"] == []


def test_the_parallel_benchmark_validates_its_body_before_starting_a_job(app: Any) -> None:
    with TestClient(app) as http:
        response = http.post(
            f"/api/models/{MODEL_ID}/benchmark-parallel", json={"streams": ["four"]}
        )
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "streams"


def test_the_parallel_benchmark_refuses_a_bad_cache_type_before_the_202(app: Any) -> None:
    """The same validation an ordinary load gets, up front -- not a job that
    fails minutes later where the caller has to go and find it."""
    with TestClient(app) as http:
        response = http.post(
            f"/api/models/{MODEL_ID}/benchmark-parallel", json={"kv_cache_type": "pwned"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "kv_cache_type"
        response = http.post(f"/api/models/{MODEL_ID}/benchmark-parallel", json={"devices": [7]})
        assert response.status_code == 400
        assert response.json()["error"]["param"] == "devices"


def test_the_parallel_benchmark_refuses_a_busy_server_before_the_202(app: Any) -> None:
    """Found on the scratch server: a busy rig got a 202 and a job that failed
    on its first line. The route answers 503 with retry_after_s up front."""
    with TestClient(app) as http:
        app.state.manager.busy_snapshot = lambda: {
            "active_requests": 1,
            "busy_models": [{"model_id": "pub/other", "active_requests": 1}],
            "loading": [],
            "testing": None,
        }
        response = http.post(f"/api/models/{MODEL_ID}/benchmark-parallel", json={"streams": [1]})
    assert response.status_code == 503
    assert response.json()["error"]["studioforge"]["retry_after_s"] == 15.0


def test_the_parallel_benchmark_starts_a_job_on_the_shared_table(app: Any) -> None:
    """Same job table as the placement benchmark: one thing for a client to poll."""
    with TestClient(app) as http:
        started = http.post(f"/api/models/{MODEL_ID}/benchmark-parallel", json={"streams": [1, 2]})
        assert started.status_code == 202
        job_id = started.json()["job_id"]
        polled = http.get(f"/api/benchmark/jobs/{job_id}")
    assert polled.status_code == 200
    assert polled.json()["model_id"] == MODEL_ID


def test_catalog_rows_say_how_many_slots_are_worth_running(app: Any) -> None:
    with TestClient(app) as http:
        body = http.get("/api/catalog").json()
    row = next(r for r in body["models"][0]["options"] if r["fits"])
    assert 1 <= row["recommended_parallel"] <= row["max_parallel"]
    assert row["recommended_parallel_basis"] == "estimated"
    assert row["load_args"]["parallel"] == row["recommended_parallel"]


# ---------------------------------------------------------------------------
# The 2026-08-26 ops surfaces: PATCH settings, evictions, jobs list, options,
# client attribution
# ---------------------------------------------------------------------------

#: PATCH settings is D32-gated; TestClient's default peer is "testclient",
#: which is not loopback, so these speak as the operator's own machine.
LOOPBACK = ("127.0.0.1", 50000)


def test_patch_settings_changes_only_the_named_field(app: Any) -> None:
    """RFC 7386: a present key is set, everything else stays byte-identical."""
    with TestClient(app, client=LOOPBACK) as http:
        before = http.get(f"/api/models/{MODEL_ID}/settings").json()
        patched = http.patch(f"/api/models/{MODEL_ID}/settings", json={"ctx_size": 131072})
        assert patched.status_code == 200
        after = http.get(f"/api/models/{MODEL_ID}/settings").json()

    assert after["ctx_size"] == 131072
    assert {**after, "ctx_size": before["ctx_size"]} == before


def test_patch_settings_null_clears_exactly_that_field(app: Any) -> None:
    with TestClient(app, client=LOOPBACK) as http:
        http.patch(
            f"/api/models/{MODEL_ID}/settings",
            json={"ctx_size": 131072, "reasoning_format": "deepseek"},
        )
        cleared = http.patch(
            f"/api/models/{MODEL_ID}/settings", json={"reasoning_format": None}
        ).json()

    assert cleared["reasoning_format"] is None
    assert cleared["ctx_size"] == 131072  # untouched by the second patch


def test_patch_settings_rejects_an_unknown_field(app: Any) -> None:
    """A typoed key that 'worked' is the failure class PATCH exists to close."""
    with TestClient(app, client=LOOPBACK) as http:
        response = http.patch(f"/api/models/{MODEL_ID}/settings", json={"ctxsize": 1})
    assert response.status_code == 400
    assert "ctxsize" in response.json()["error"]["message"]


PERSONA_ID = "vendor/thing-Q4_K_M:coach"  # scrub-ok: generic-English fixture word


def make_persona(record_id: str = PERSONA_ID, base: str = MODEL_ID) -> ModelRecord:
    """A preset-only persona: request-time behaviour only, so it is served by
    its base's instance rather than a child of its own."""
    return make_record(record_id).model_copy(
        update={
            "is_virtual": True,
            "base_model_id": base,
            "preset": VirtualPreset(system_prompt="you are a coach"),  # scrub-ok: generic-English fixture
        }
    )


@pytest.mark.parametrize("verb", ["patch", "put"])
def test_saving_a_tier_on_a_persona_is_a_400_that_names_the_base(app: Any, verb: str) -> None:
    """A persona that shares its base's llama-server cannot carry a tier of its
    own (D48), and the refusal has to reach the client as a readable 400 naming
    the base id -- not a 500, and not a success that silently split the
    persona onto its own child. The registry raises it; this pins that the
    settings routes let it through unmangled, both verbs.
    """
    persona = make_persona()
    registry = FakeRegistry([make_record(), persona])
    app.state.registry = registry
    app.state.manager.registry = registry
    body: dict[str, Any] = (
        {"priority": 1}
        if verb == "patch"
        else {**persona.settings.model_dump(mode="json"), "priority": 1}
    )

    with TestClient(app, client=LOOPBACK) as http:
        response = getattr(http, verb)(f"/api/models/{PERSONA_ID}/settings", json=body)
        # The base itself takes the same write: the refusal is about sharing an
        # instance, not about the tier.
        allowed = http.patch(f"/api/models/{MODEL_ID}/settings", json={"priority": 1})

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert error["param"] == "priority"
    assert MODEL_ID in error["message"], "the refusal has to say where to save it instead"

    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["priority"] == 1


def test_evictions_endpoint_names_who_evicted_what_and_why(app: Any) -> None:
    manager = app.state.manager
    manager._record_eviction("vendor/old-model", reason="plan", evicted_by=MODEL_ID)
    manager._record_eviction("vendor/idle-model", reason="ttl")
    # Windows' clock is coarse enough that both events can share a timestamp;
    # the since-filter assertion needs them a real interval apart.
    manager._evictions[0]["ts"] -= 60.0
    with TestClient(app) as http:
        body = http.get("/api/evictions").json()
        since = http.get("/api/evictions", params={"since": body["evictions"][0]["ts"]}).json()

    assert body["count"] == 2
    newest = body["evictions"][0]
    assert newest["evicted"] == "vendor/idle-model"
    assert newest["reason"] == "ttl"
    oldest = body["evictions"][-1]
    assert oldest["evicted_by"] == MODEL_ID
    assert oldest["reason"] == "plan"
    assert since["count"] == 1


def test_benchmark_jobs_are_listable_and_status_carries_the_running_one(app: Any) -> None:
    """The job table used to be job_id-or-nothing: a client that lost the id
    could only rediscover a run by colliding with it."""
    from studioforge.api.mgmt_routes import _benchmark_jobs

    job = _benchmark_jobs(app.state).create(MODEL_ID, ["single_gpu"])
    with TestClient(app) as http:
        listed = http.get("/api/benchmark/jobs").json()
        status = http.get("/api/status").json()
        job.state = "completed"
        done = http.get("/api/status").json()

    assert listed["count"] == 1
    assert listed["jobs"][0]["job_id"] == job.job_id
    assert status["benchmark"] == {
        "job_id": job.job_id,
        "model_id": MODEL_ID,
        "mode": None,
        "phase": "planning",
        "fraction": 0.0,
    }
    assert done["benchmark"] is None


def test_model_options_is_a_plain_get(app: Any) -> None:
    """Read-only capacity math must not need the MCP plane (wish W5)."""
    with TestClient(app) as http:
        body = http.get(f"/api/models/{MODEL_ID}/options").json()
    assert body["model"]["id"] == MODEL_ID
    assert body["model"]["options"]


def test_status_attributes_inference_to_clients(app: Any) -> None:
    manager = app.state.manager
    manager.note_client(host="100.66.1.2", label=None, model_id=MODEL_ID)
    manager.note_client(host="100.66.1.2", label=None, model_id=MODEL_ID)
    manager.note_client(host="100.66.1.3", label="sift-forge", model_id=MODEL_ID)
    with TestClient(app) as http:
        clients = http.get("/api/status").json()["clients"]

    assert clients[0]["client"] == "100.66.1.2"
    assert clients[0]["requests"] == 2
    assert clients[0]["models"] == {MODEL_ID: 2}
    assert {c["client"] for c in clients} == {"100.66.1.2", "sift-forge"}
