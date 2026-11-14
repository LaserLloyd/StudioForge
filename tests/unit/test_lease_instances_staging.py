"""Two copies of one model on two fake card pairs, end to end on stub staging (D91, D85).

The real gateway (``create_app``), real stub ``llama-server`` children on the
reserved 19420-19460 ports, four synthetic cards -- nothing touches a GPU or a
live port. Two holders share ONE inference key, as every bench run on one
client does; that is exactly the 2026-11-14 02:10 situation:

* holder A serves the model on CUDA [2, 3] with three streams in flight;
* holder B leases [0, 1] naming the same id (the old ``model_ids`` shape);
* B gets its own instance ``<model>:2``, its requests for the base id land
  there, its unload of the base id takes only that down, and nothing B does --
  naming A's lease, the key alone, an unload once its own lease is gone -- cuts
  A's streams;
* B's released instance answers 404 ``instance_not_found`` without a load.

The second test is the restart: a lease with ``instances`` survives a restart
unloaded and the first request loads exactly the stored shape.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge import build
from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core import gpu as gpu_module
from tests.unit.test_mcp import fake_meta

GIB = 1024**3
FAKE_GPUS = json.dumps(
    [
        {"index": 0, "name": "RTX 5090", "total_bytes": 32 * GIB, "compute_capability": [12, 0]},
        {"index": 1, "name": "RTX 5090", "total_bytes": 32 * GIB, "compute_capability": [12, 0]},
        {"index": 2, "name": "RTX 3090", "total_bytes": 24 * GIB, "compute_capability": [8, 6]},
        {"index": 3, "name": "RTX 3090", "total_bytes": 24 * GIB, "compute_capability": [8, 6]},
    ]
)
MODEL = "pub/Tiny-Q4_K_M"
COPY = f"{MODEL}:2"
A = "benchrun-tput-d"
B = "benchrun-tput-p"
MESSAGES = [{"role": "user", "content": "say something"}]
#: An operator on this machine: the lease and unload routes are admin from here (D32).
LOOPBACK = ("127.0.0.1", 50000)


def _config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.engine.stub = True
    cfg.gpu.backend = "fake"
    cfg.models.dir = tmp_path / "models"
    cfg.gateway.child_port_start = 19440
    cfg.gateway.child_port_end = 19460
    cfg.gateway.load_timeout_s = 30.0
    cfg.gateway.health_poll_interval_s = 0.05
    cfg.ensure_dirs()
    return cfg


def _app(cfg: Config) -> tuple[Any, Any]:
    state = build_state(cfg)
    state.registry._meta_reader = fake_meta  # noqa: SLF001 - filename-driven GGUF metadata
    return create_app(cfg, state=state), state


@pytest.fixture
def staging_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("SF_FAKE_GPUS", FAKE_GPUS)
    monkeypatch.delenv("SF_GPU_PROBE", raising=False)
    gpu_module.reset_probe()
    models = tmp_path / "models" / "pub"
    models.mkdir(parents=True)
    (models / "Tiny-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * (4 * 1024 * 1024))
    try:
        yield tmp_path
    finally:
        build.mark_stub_engine(False)
        build.build_id.cache_clear()  # type: ignore[attr-defined]
        gpu_module.reset_probe()


def _booted(state: Any, timeout_s: float = 20.0) -> None:
    """Wait for the boot to finish: the lease restore runs in it, after the port answers."""
    deadline = time.monotonic() + timeout_s
    while state.boot.phase != "ready":
        assert time.monotonic() < deadline, f"boot stuck at {state.boot.phase!r}"
        time.sleep(0.05)


def _loaded(client: TestClient, headers: dict[str, str]) -> dict[str, dict[str, Any]]:
    rows = client.get("/api/status", headers=headers).json()["loaded"]
    return {row["model_id"]: row for row in rows}


def test_two_holders_one_key_the_second_never_replaces_the_first(staging_dir: Path) -> None:
    cfg = _config(staging_dir)
    app, state = _app(cfg)
    _record, secret = state.accounts.create_key("bench", "inference")
    key = {"Authorization": f"Bearer {secret}"}
    as_a = {**key, "X-SF-Client": A}
    as_b = {**key, "X-SF-Client": B}
    with TestClient(app) as client:
        # Holder A: a lease on [2, 3] naming the model, then the model loaded on it.
        lease_a = client.post(
            "/api/leases", json={"devices": [2, 3], "model_ids": [MODEL], "holder": A}, headers=as_a
        )
        assert lease_a.status_code == 200, lease_a.text
        lease_a_id = lease_a.json()["id"]
        loaded_a = client.post(
            f"/api/models/{MODEL}/load",
            json={"lease_id": lease_a_id, "ctx_size": 4096},
            headers=as_a,
        )
        assert loaded_a.status_code == 200, loaded_a.text
        assert sorted(loaded_a.json()["plan"]["devices"]) == [2, 3]
        pid_a = loaded_a.json()["pid"]
        # Three of A's streams in flight (the records a long stream holds open).
        in_flight = [state.supervisor.mark_request_start(MODEL, client=A) for _ in range(3)]

        def a_is_intact() -> None:
            row = _loaded(client, key)[MODEL]
            assert row["pid"] == pid_a and row["state"] == "ready"
            assert row["active_requests"] == 3
            assert {entry["client"] for entry in row["in_flight"]} == {A}

        # Holder B, SAME key: a lease on [0, 1] naming the same model, the old shape.
        lease_b = client.post(
            "/api/leases", json={"devices": [0, 1], "model_ids": [MODEL], "holder": B}, headers=as_b
        )
        assert lease_b.status_code == 200, lease_b.text
        body = lease_b.json()
        lease_b_id = body["id"]
        assert body["model_ids"] == [COPY], "B never names A's instance"
        (row,) = body["instances"]
        assert row["model"] == COPY and row["origin"] == "second_holder"
        assert row["devices"] == [0, 1] and row["state"] == "unloaded"
        assert body["lease"]["instances"][0]["id"] == COPY
        a_is_intact()

        # B's chat naming the BASE id lands on B's own copy, on B's cards.
        chat = {"model": MODEL, "messages": MESSAGES, "max_tokens": 4, "seed": 1}
        answered = client.post("/v1/chat/completions", json=chat, headers=as_b)
        assert answered.status_code == 200, answered.text
        assert answered.headers["X-SF-Model"] == COPY
        rows = _loaded(client, key)
        assert sorted(rows[COPY]["plan"]["devices"]) == [0, 1]
        assert rows[COPY]["ephemeral"] is True and rows[COPY]["lease_id"] == lease_b_id
        assert rows[COPY]["holder"] == B and rows[COPY]["base_model_id"] == MODEL
        assert rows[MODEL]["lease_id"] == lease_a_id and rows[MODEL]["holder"] == A
        # ... and A's chat still lands on A's.
        a_chat = client.post("/v1/chat/completions", json=chat, headers=as_a)
        assert a_chat.status_code == 200 and a_chat.headers["X-SF-Model"] == MODEL
        # Each by its exact id, whoever asks.
        exact = client.post("/v1/chat/completions", json={**chat, "model": COPY}, headers=as_a)
        assert exact.status_code == 200 and exact.headers["X-SF-Model"] == COPY

        # /v1/models: every instance under the base, and each copy its own entry.
        listing = {m["id"]: m for m in client.get("/v1/models", headers=key).json()["data"]}
        instances = {i["id"]: i for i in listing[MODEL]["studioforge"]["loaded_instances"]}
        assert instances[MODEL]["lease_id"] == lease_a_id and instances[MODEL]["holder"] == A
        assert instances[MODEL]["devices"] == [2, 3]
        assert instances[COPY]["lease_id"] == lease_b_id and instances[COPY]["holder"] == B
        assert instances[COPY]["devices"] == [0, 1] and instances[COPY]["config"]["context_length"]
        assert listing[COPY]["studioforge"]["ephemeral"] is True
        assert listing[COPY]["studioforge"]["lease_id"] == lease_b_id
        assert listing[COPY]["status"]["value"] == "loaded"
        # GET /api/leases shows the copy with its holder, cards and live state.
        leases = {
            lease["id"]: lease for lease in client.get("/api/leases", headers=key).json()["leases"]
        }
        assert leases[lease_b_id]["instances"][0]["state"] == "loaded"
        assert leases[lease_b_id]["instances"][0]["holder"] == B
        # The library listing keeps lease instances out unless asked.
        library = client.get("/api/models", headers=key).json()["models"]
        assert COPY not in {m["id"] for m in library}
        with_copies = client.get("/api/models?include_ephemeral=true", headers=key).json()
        assert COPY in {m["id"] for m in with_copies["models"]}

        # B naming A's lease while saying it is B: a shared key does not make B
        # A's holder (D91 review fix 3) -- 403 before anything is looked at.
        stolen = client.post(
            f"/api/models/{MODEL}/unload", headers={**as_b, "X-SF-Lease": lease_a_id}
        )
        assert stolen.status_code == 403, stolen.text
        assert stolen.json()["error"]["code"] == "permission_denied"
        a_is_intact()
        # The shared key alone matches both holders: ambiguous, refused, A intact.
        ambiguous = client.post(f"/api/models/{MODEL}/unload", headers=key)
        assert ambiguous.status_code == 409, ambiguous.text
        assert ambiguous.json()["error"]["code"] == "instance_ambiguous"
        a_is_intact()
        # B's unload of the base id takes down B's own instance only.
        unloaded = client.post(f"/api/models/{MODEL}/unload", headers=as_b)
        assert unloaded.status_code == 200, unloaded.text
        assert unloaded.json()["unloaded_id"] == COPY and unloaded.json()["unloaded"] is True
        assert COPY not in _loaded(client, key)
        a_is_intact()

        # B releases its lease; its copy ends with it.
        released = client.delete(f"/api/leases/{lease_b_id}", headers=as_b)
        assert released.status_code == 200, released.text
        assert released.json()["instances_ended"] == [COPY]
        # With no lease of its own, B is still not A: its label stops the shared key.
        cut = client.post(f"/api/models/{MODEL}/unload", headers=as_b)
        assert cut.status_code == 409, cut.text
        assert cut.json()["error"]["code"] == "lease_conflict"
        a_is_intact()
        # The key alone (no label) does prove the lease -- and then meets A's streams.
        keyed = client.post(f"/api/models/{MODEL}/unload", headers=key)
        assert keyed.status_code == 409, keyed.text
        error = keyed.json()["error"]
        assert error["code"] == "instance_in_use"
        assert error["studioforge"]["in_flight_clients"] == [A]
        a_is_intact()

        # The released copy's id: a 404 of its own, and no load attempted.
        gone = client.post("/v1/chat/completions", json={**chat, "model": COPY}, headers=as_b)
        assert gone.status_code == 404, gone.text
        assert gone.json()["error"]["code"] == "instance_not_found"
        assert state.supervisor.get(COPY) is None
        never = client.post(
            "/v1/chat/completions", json={**chat, "model": f"{MODEL}:7"}, headers=as_b
        )
        assert never.status_code == 404 and never.json()["error"]["code"] == "instance_not_found"

        for request_id in in_flight:
            state.supervisor.mark_request_end(MODEL, request_id=request_id)


def test_a_restart_restores_the_copies_unloaded_and_loads_the_stored_shape(
    staging_dir: Path,
) -> None:
    cfg = _config(staging_dir)
    app, state = _app(cfg)
    with TestClient(app, client=LOOPBACK) as client:
        granted = client.post(
            "/api/leases",
            json={
                "devices": [0, 1, 2, 3],
                "holder": "judge",
                "idle_ttl_s": None,
                "instances": [
                    {"model": MODEL, "gpus": [0, 1], "launch": {"ctx_size": 4096, "parallel": 2}},
                    {"model": MODEL, "gpus": [2, 3], "tag": "b", "launch": {"ctx_size": 2048}},
                ],
            },
        )
        assert granted.status_code == 200, granted.text
        body = granted.json()
        lease_id = body["id"]
        rows = {row["id"]: row for row in body["instances"]}
        assert set(rows) == {COPY, f"{MODEL}@b"}
        assert rows[COPY]["state"] == "loaded" and rows[COPY]["devices"] == [0, 1]
        assert rows[COPY]["ctx_per_slot"] == 4096 and rows[COPY]["parallel"] == 2
        assert rows[f"{MODEL}@b"]["devices"] == [2, 3]
        assert rows[f"{MODEL}@b"]["ctx_per_slot"] == 2048
        shape = {row_id: (row["ctx_per_slot"], row["parallel"]) for row_id, row in rows.items()}
        # Two copies, two card pairs, both answering.
        for model in rows:
            reply = client.post(
                "/v1/chat/completions",
                json={"model": model, "messages": MESSAGES, "max_tokens": 4},
            )
            assert reply.status_code == 200 and reply.headers["X-SF-Model"] == model
        # Not a virtual model: nothing in the virtual-model table, only the lease row.
        assert state.db.list_virtual_models() == []
        # The base cannot be deleted from under a resident copy (D86's guard holds).
        refused = client.delete(f"/api/models/{MODEL}")
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "model_loaded"
        # A second lease on the same cards is a conflict, never a takeover.
        clash = client.post("/api/leases", json={"devices": [0], "holder": "other"})
        assert clash.status_code == 409 and clash.json()["error"]["code"] == "lease_conflict"

    # The process ends and a new one starts on the same data directory.
    reborn, reborn_state = _app(cfg)
    with TestClient(reborn, client=LOOPBACK) as client:
        _booted(reborn.state)
        leases = client.get("/api/leases").json()["leases"]
        (lease,) = leases
        assert lease["id"] == lease_id
        restored = {row["id"]: row for row in lease["instances"]}
        assert {row["state"] for row in restored.values()} == {"unloaded"}
        assert reborn_state.supervisor.get(COPY) is None
        reply = client.post(
            "/v1/chat/completions", json={"model": COPY, "messages": MESSAGES, "max_tokens": 4}
        )
        assert reply.status_code == 200, reply.text
        loaded = _loaded(client, {})
        plan = loaded[COPY]["plan"]
        assert sorted(plan["devices"]) == [0, 1], "exactly the stored cards"
        assert (plan["ctx_per_slot"], plan["parallel"]) == shape[COPY], "never the ladder"
        released = client.delete(f"/api/leases/{lease_id}")
        assert sorted(released.json()["instances_ended"]) == sorted([COPY, f"{MODEL}@b"])
        assert COPY not in _loaded(client, {})


# ---------------------------------------------------------------------------
# Review fixes, end to end (2026-11-14)
# ---------------------------------------------------------------------------


def _keyed(staging_dir: Path) -> tuple[Any, Any, dict[str, str]]:
    cfg = _config(staging_dir)
    app, state = _app(cfg)
    _record, secret = state.accounts.create_key("bench", "inference")
    return app, state, {"Authorization": f"Bearer {secret}"}


def test_review1_b_with_no_lease_and_the_shared_key_never_cuts_as_streams(
    staging_dir: Path,
) -> None:
    """B holds no lease; A's streams are unlabeled (the peer is all they carry)."""
    app, state, key = _keyed(staging_dir)
    as_a = {**key, "X-SF-Client": A}
    as_b = {**key, "X-SF-Client": B}
    with TestClient(app) as client:
        lease_a = client.post(
            "/api/leases", json={"devices": [2, 3], "model_ids": [MODEL], "holder": A}, headers=as_a
        ).json()["id"]
        loaded = client.post(
            f"/api/models/{MODEL}/load", json={"lease_id": lease_a, "ctx_size": 4096}, headers=as_a
        )
        assert loaded.status_code == 200, loaded.text
        pid_a = loaded.json()["pid"]
        # The same peer every caller here has: an unlabeled stream names only it.
        in_flight = [
            state.supervisor.mark_request_start(MODEL, client="testclient") for _ in range(3)
        ]

        def a_is_intact() -> None:
            row = _loaded(client, key)[MODEL]
            assert row["pid"] == pid_a and row["active_requests"] == 3

        # B's label is not A's: the shared key does not make it A's holder.
        refused = client.post(f"/api/models/{MODEL}/unload", headers=as_b)
        assert refused.status_code == 409, refused.text
        assert refused.json()["error"]["code"] == "lease_conflict"
        a_is_intact()
        # No label at all: the key proves the lease, but the peer it shares with
        # A's streams does not make them its own.
        cut = client.post(f"/api/models/{MODEL}/unload", headers=key)
        assert cut.status_code == 409, cut.text
        assert cut.json()["error"]["code"] == "instance_in_use"
        assert cut.json()["error"]["studioforge"]["in_flight_clients"] == ["testclient"]
        a_is_intact()
        for request_id in in_flight:
            state.supervisor.mark_request_end(MODEL, request_id=request_id)


def test_review2_two_concurrent_grants_get_two_ids(staging_dir: Path) -> None:
    """Both grants evict a resident (an await between planning and making the copy)."""
    from concurrent.futures import ThreadPoolExecutor

    other = staging_dir / "models" / "pub" / "Other-Q4_K_M.gguf"
    other.write_bytes(b"GGUF" + b"\0" * (4 * 1024 * 1024))
    cfg = _config(staging_dir)
    app, state = _app(cfg)
    with TestClient(app, client=LOOPBACK) as client:
        _booted(app.state)
        for model, cards in ((MODEL, [0, 1]), ("pub/Other-Q4_K_M", [2, 3])):
            placed = client.post(
                f"/api/models/{model}/load", json={"devices": cards, "ctx_size": 2048}
            )
            assert placed.status_code == 200, placed.text

        def grant(cards: list[int], holder: str) -> Any:
            return client.post(
                "/api/leases",
                json={
                    "devices": cards,
                    "holder": holder,
                    "instances": [{"model": MODEL, "load": False}],
                },
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(grant, ([0, 1], [2, 3]), ("run-a", "run-b")))
        for reply in replies:
            assert reply.status_code == 200, reply.text
        ids = [reply.json()["instances"][0]["id"] for reply in replies]
        assert sorted(ids) == [f"{MODEL}:2", f"{MODEL}:3"], ids
        by_lease = {
            lease["id"]: [row["id"] for row in lease["instances"]]
            for lease in client.get("/api/leases").json()["leases"]
        }
        assert sorted(by_lease.values()) == [[f"{MODEL}:2"], [f"{MODEL}:3"]]
        for reply in replies:
            body = reply.json()
            record = state.registry.get(body["instances"][0]["id"])
            assert record is not None and record.lease_id == body["id"]


def test_review3_b_naming_as_lease_is_refused_everywhere(staging_dir: Path) -> None:
    """B, with its own lease and the shared key, cannot act on A's lease by naming it."""
    app, state, key = _keyed(staging_dir)
    as_a = {**key, "X-SF-Client": A}
    as_b = {**key, "X-SF-Client": B}
    with TestClient(app) as client:
        granted = client.post(
            "/api/leases",
            json={
                "devices": [2, 3],
                "holder": A,
                "instances": [{"model": MODEL, "launch": {"ctx_size": 4096}}],
            },
            headers=as_a,
        )
        assert granted.status_code == 200, granted.text
        lease_a = granted.json()["id"]
        copy_a = granted.json()["instances"][0]["id"]
        pid_a = _loaded(client, key)[copy_a]["pid"]
        lease_b = client.post(
            "/api/leases", json={"devices": [0, 1], "holder": B}, headers=as_b
        ).json()["id"]
        in_flight = [state.supervisor.mark_request_start(copy_a, client=A) for _ in range(2)]

        bound = client.post(
            f"/api/models/{MODEL}/load", json={"lease_id": lease_a, "force": True}, headers=as_b
        )
        assert bound.status_code == 403, bound.text
        assert bound.json()["error"]["code"] == "permission_denied"
        renew = client.patch(f"/api/leases/{lease_a}", json={"renew_time": "now"}, headers=as_b)
        assert renew.status_code == 403, renew.text
        release = client.delete(f"/api/leases/{lease_a}", headers=as_b)
        assert release.status_code == 403, release.text
        named = client.post(
            "/v1/chat/completions",
            json={"model": copy_a, "messages": MESSAGES, "max_tokens": 2},
            headers={**as_b, "X-SF-Lease": lease_a},
        )
        assert named.status_code == 200, "naming a lease never gates inference"
        reshape = client.post(f"/api/models/{copy_a}/load", json={"ctx_size": 2048}, headers=as_b)
        assert reshape.status_code == 409, reshape.text
        assert reshape.json()["error"]["code"] == "instance_held"
        row = _loaded(client, key)[copy_a]
        assert row["pid"] == pid_a and row["plan"]["ctx_per_slot"] == 4096
        assert {
            lease["id"] for lease in client.get("/api/leases", headers=key).json()["leases"]
        } == {
            lease_a,
            lease_b,
        }
        for request_id in in_flight:
            state.supervisor.mark_request_end(copy_a, request_id=request_id)


def test_review_lm_studio_unload_by_base_id_finds_the_callers_own_copy(staging_dir: Path) -> None:
    app, _state, key = _keyed(staging_dir)
    as_b = {**key, "X-SF-Client": B}
    with TestClient(app) as client:
        granted = client.post(
            "/api/leases",
            json={
                "devices": [0, 1],
                "holder": B,
                "instances": [{"model": MODEL, "launch": {"ctx_size": 2048}}],
            },
            headers=as_b,
        )
        assert granted.status_code == 200, granted.text
        assert MODEL not in _loaded(client, key), "only the copy is loaded"
        unloaded = client.post("/api/v1/models/unload", json={"instance_id": MODEL}, headers=as_b)
        assert unloaded.status_code == 200, unloaded.text
        assert unloaded.json() == {"instance_id": COPY}
        assert COPY not in _loaded(client, key)


def test_r1_a_holder_whose_label_differs_from_its_lease_name_keeps_its_lease(
    staging_dir: Path,
) -> None:
    """Leased as ``openclaw``, labelled ``openclaw-main``, its own key: all of it is its own."""
    app, _state, key = _keyed(staging_dir)
    me = {**key, "X-SF-Client": "openclaw-main"}
    with TestClient(app) as client:
        granted = client.post(
            "/api/leases",
            json={
                "devices": [0, 1],
                "holder": "openclaw",
                "instances": [{"model": MODEL, "launch": {"ctx_size": 2048}}],
            },
            headers=me,
        )
        assert granted.status_code == 200, granted.text
        lease_id = granted.json()["id"]
        chat = client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": MESSAGES, "max_tokens": 2},
            headers=me,
        )
        assert chat.status_code == 200 and chat.headers["X-SF-Model"] == COPY
        unloaded = client.post(f"/api/models/{MODEL}/unload", headers=me)
        assert unloaded.status_code == 200, unloaded.text
        assert unloaded.json()["unloaded_id"] == COPY
        renewed = client.patch(f"/api/leases/{lease_id}", json={"renew_time": "now"}, headers=me)
        assert renewed.status_code == 200, renewed.text
        released = client.delete(f"/api/leases/{lease_id}", headers=me)
        assert released.status_code == 200, released.text
        assert released.json()["instances_ended"] == [COPY]
