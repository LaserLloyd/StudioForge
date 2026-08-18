"""D30: a forced reload is planned as if the resident child were already gone.

Before this the manager stopped the running instance and *then* planned, so a
reload whose arguments no longer fit -- VRAM moved to ComfyUI, a bigger context
asked for -- left the model unloaded. Now the planner is told which instance is
being replaced, credits its footprint back to the cards it sits on, and returns
it as the first eviction; a refusal leaves the child running.

Also here: ``max_ctx_that_fits`` in a refusal is computed on the real per-layer
KV geometry (D22), so an iSWA model is offered the window that actually fits
rather than the uniform figure, which under-offered by an order of magnitude.
"""

from __future__ import annotations

from studioforge.core.planner import (
    Planner,
    kv_alloc_bytes,
    max_ctx_for_budget,
    max_ctx_for_budget_geometry,
)
from studioforge.types import GB, InstanceInfo, LoadPlan, LoadRejected, ModelSettings
from tests.unit.test_kv_geometry import meta_gemma4_31b
from tests.unit.test_planner import StubProbe, gpu, make_config, make_meta, make_record


def resident_instance(model_id: str, *, device: int, bytes_held: int, pinned: bool) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state="ready",
        port=18100,
        pid=4242,
        ttl_s=0 if pinned else 1800,
        started_at=1.0,
        last_activity_at=1.0,
        plan=LoadPlan(model_id=model_id, devices=[device], per_gpu_bytes={device: bytes_held}),
    )


# ---------------------------------------------------------------------------
# reload_of
# ---------------------------------------------------------------------------


def test_a_forced_reload_credits_the_resident_footprint_back() -> None:
    """A 20 GiB model on a 24 GiB card with 3 GiB free: the reload only fits
    because the child being replaced is going away."""
    probe = StubProbe([gpu(0, 24.0, 3.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=16 * GB), size_bytes=16 * GB)
    resident = resident_instance(record.id, device=0, bytes_held=20 * GB, pinned=True)

    without_hint = planner.plan_load(record, ctx_size=8192, loaded=[resident])
    assert isinstance(without_hint, LoadRejected), "sanity: 3 GiB free is not enough"

    plan = planner.plan_load(record, ctx_size=8192, loaded=[resident], reload_of=record.id)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids[0] == record.id
    assert plan.devices == [0]
    assert any("forced reload" in note for note in plan.notes)


def test_a_pinned_resident_is_not_an_obstacle_to_its_own_reload() -> None:
    """The resident being pinned must not appear as 'unpin X' in its own reload,
    and must not block eviction-free planning."""
    probe = StubProbe([gpu(0, 24.0, 3.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=16 * GB), size_bytes=16 * GB)
    resident = resident_instance(record.id, device=0, bytes_held=20 * GB, pinned=True)

    plan = planner.plan_load(record, ctx_size=8192, loaded=[resident], reload_of=record.id)
    assert isinstance(plan, LoadPlan)
    # Only the resident is stopped; nothing else was there to evict.
    assert plan.evict_model_ids == [record.id]


def test_a_refused_reload_reports_the_resident_pid_as_ours() -> None:
    """The child being replaced must not show up as a foreign VRAM holder."""
    from studioforge.types import VramProcess

    class HolderProbe(StubProbe):
        def compute_processes(self) -> list[VramProcess]:
            return [VramProcess(gpu_index=0, pid=4242, name="llama-server.exe", used_bytes=20 * GB)]

    probe = HolderProbe([gpu(0, 24.0, 1.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=40 * GB), size_bytes=40 * GB)
    resident = resident_instance(record.id, device=0, bytes_held=20 * GB, pinned=False)

    result = planner.plan_load(record, ctx_size=8192, loaded=[resident], reload_of=record.id)
    assert isinstance(result, LoadRejected)
    ours = [h for h in result.vram_holders if h.pid == 4242]
    assert ours and all(h.is_ours for h in ours)
    assert not any("held by other processes" in s for s in result.suggestions)


def test_reload_of_a_model_that_is_not_loaded_is_an_ordinary_plan() -> None:
    probe = StubProbe([gpu(0, 32.0, 31.0, (12, 0))])
    planner = Planner(make_config(), probe)
    record = make_record()
    plan = planner.plan_load(record, ctx_size=8192, loaded=[], reload_of=record.id)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids == []
    assert not any("forced reload" in note for note in plan.notes)


def test_a_reload_still_evicts_other_idle_models_when_it_must() -> None:
    probe = StubProbe([gpu(0, 32.0, 2.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(meta=make_meta(tensor_bytes=24 * GB), size_bytes=24 * GB)
    resident = resident_instance(record.id, device=0, bytes_held=10 * GB, pinned=False)
    other = resident_instance("other/model", device=0, bytes_held=18 * GB, pinned=False)

    plan = planner.plan_load(record, ctx_size=8192, loaded=[resident, other], reload_of=record.id)
    assert isinstance(plan, LoadPlan)
    assert plan.evict_model_ids[0] == record.id
    assert "other/model" in plan.evict_model_ids


def test_a_reload_that_names_a_device_override_credits_that_device() -> None:
    probe = StubProbe([gpu(0, 24.0, 3.0, (12, 0)), gpu(1, 24.0, 23.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    record = make_record(
        meta=make_meta(tensor_bytes=16 * GB),
        size_bytes=16 * GB,
        settings=ModelSettings(device_override=[0]),
    )
    resident = resident_instance(record.id, device=0, bytes_held=20 * GB, pinned=True)
    plan = planner.plan_load(record, ctx_size=8192, loaded=[resident], reload_of=record.id)
    assert isinstance(plan, LoadPlan)
    assert plan.devices == [0]


# ---------------------------------------------------------------------------
# max_ctx_that_fits on the real geometry
# ---------------------------------------------------------------------------


def test_geometry_max_ctx_matches_the_uniform_answer_for_a_uniform_model() -> None:
    meta = make_meta()
    budget = 2 * GB
    uniform = max_ctx_for_budget(
        budget_bytes=budget,
        n_layer=meta.n_layer,
        n_head_kv=meta.n_head_kv,
        head_dim_k=meta.head_dim_k,
        head_dim_v=meta.head_dim_v,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    geometry = max_ctx_for_budget_geometry(meta, budget_bytes=budget, kv_k="f16", kv_v="f16")
    # The geometry walk answers on the ladder; the uniform one is exact. The
    # ladder rung must be <= the exact figure and the next rung up must not fit.
    assert 0 < geometry <= uniform
    assert kv_alloc_bytes(meta, ctx_total=geometry, kv_k="f16", kv_v="f16") <= budget


def test_geometry_max_ctx_offers_an_iswa_model_the_window_that_really_fits() -> None:
    """Gemma-4 31B: the uniform figure under-offers by more than an order of
    magnitude, because five of six layers keep a 1024-token window."""
    meta = meta_gemma4_31b()
    budget = 4 * GB
    uniform = max_ctx_for_budget(
        budget_bytes=budget,
        n_layer=meta.n_layer,
        n_head_kv=meta.n_head_kv,
        head_dim_k=meta.head_dim_k,
        head_dim_v=meta.head_dim_v,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    geometry = max_ctx_for_budget_geometry(meta, budget_bytes=budget, kv_k="f16", kv_v="f16")
    assert geometry >= 8 * uniform
    assert kv_alloc_bytes(meta, ctx_total=geometry, kv_k="f16", kv_v="f16") <= budget


def test_geometry_max_ctx_honours_the_slot_count() -> None:
    meta = meta_gemma4_31b()
    one = max_ctx_for_budget_geometry(meta, budget_bytes=4 * GB, kv_k="f16", kv_v="f16")
    four = max_ctx_for_budget_geometry(
        meta, budget_bytes=4 * GB, kv_k="f16", kv_v="f16", parallel=4
    )
    assert 0 < four < one


def test_geometry_max_ctx_is_zero_when_nothing_fits_or_geometry_is_unknown() -> None:
    assert max_ctx_for_budget_geometry(make_meta(), budget_bytes=0, kv_k="f16", kv_v="f16") == 0
    assert max_ctx_for_budget_geometry(make_meta(), budget_bytes=1, kv_k="f16", kv_v="f16") == 0
    no_geometry = make_meta(n_layer=0)
    assert max_ctx_for_budget_geometry(no_geometry, budget_bytes=GB, kv_k="f16", kv_v="f16") == 0


def test_a_refusal_offers_an_iswa_model_a_context_the_next_load_accepts() -> None:
    """The offered context must be one that then plans -- and it must be the
    real one, not the uniform figure that sent users to a 4k window."""
    probe = StubProbe([gpu(0, 32.0, 32.0, (12, 0))])
    planner = Planner(make_config(headroom_fraction=0.0), probe)
    meta = meta_gemma4_31b(tensor_bytes=28 * GB)
    record = make_record(meta=meta, size_bytes=28 * GB)

    result = planner.plan_load(record, ctx_size=262144, kv_cache_type="f16")
    assert isinstance(result, LoadRejected)
    assert result.max_ctx_that_fits is not None
    uniform = max_ctx_for_budget(
        budget_bytes=result.available_bytes
        - (result.estimate.total_bytes - result.estimate.kv_bytes),
        n_layer=meta.n_layer,
        n_head_kv=meta.n_head_kv,
        head_dim_k=meta.head_dim_k,
        head_dim_v=meta.head_dim_v,
        kv_type_k="f16",
        kv_type_v="f16",
    )
    assert result.max_ctx_that_fits > uniform
    replan = planner.plan_load(record, ctx_size=result.max_ctx_that_fits, kv_cache_type="f16")
    assert isinstance(replan, LoadPlan)
