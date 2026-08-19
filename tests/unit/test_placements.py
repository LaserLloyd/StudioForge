"""Hardware modes and the per-mode optimal load (D36).

The user's sentence this file exists to satisfy: *"these should be the optimal
settings to run on either the 5090s, the 3090s, a single 5090, or all"*, with
*"assume you can fill them both"* -- so every mode is computed against its own
cards **idle**, and what is in the way right now is reported beside it rather
than folded into it.

Three claims are load-bearing beyond "the code works":

* the modes are **derived from the inventory**, so a different box gets honest
  labels rather than the hard-coded ``(0, 1)`` / ``(2, 3)`` / "all" that
  ``/profiles`` used to carry;
* ``recommended`` is ``placements[0]``'s optimal, so "what should I load" has
  one answer and it names the GPUs;
* ``fits_now`` describes exactly the load ``load_args`` would submit, slots
  included -- a table that says yes to a call that then fails is worse than no
  table.
"""

from __future__ import annotations

from studioforge.core.catalog import build_catalog
from studioforge.core.placements import hardware_modes, placement_report
from studioforge.core.planner import Planner
from studioforge.types import GB, GgufMeta, GpuInfo, InstanceInfo, LoadPlan, ModelSettings
from tests.unit.test_catalog import (
    NOW,
    FakeRegistry,
    FakeSupervisor,
    dense_meta,
    record,
)
from tests.unit.test_planner import StubProbe, make_config, rig_5090x2_3090x2


def gpu(index: int, name: str, total_gib: float, free_gib: float | None = None) -> GpuInfo:
    total = int(total_gib * GB)
    free = int((total_gib if free_gib is None else free_gib) * GB)
    return GpuInfo(
        index=index,
        name=name,
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free,
        compute_capability=(12, 0) if "5090" in name else (8, 6),
    )


# ---------------------------------------------------------------------------
# Deriving the modes
# ---------------------------------------------------------------------------


def test_the_reference_rig_gets_the_four_modes_the_user_named() -> None:
    gpus = [
        gpu(0, "NVIDIA GeForce RTX 5090", 31.84),
        gpu(1, "NVIDIA GeForce RTX 5090", 31.84),
        gpu(2, "NVIDIA GeForce RTX 3090", 24.0),
        gpu(3, "NVIDIA GeForce RTX 3090", 24.0),
    ]
    modes = hardware_modes(gpus)
    assert [m.key for m in modes] == ["dual_5090", "dual_3090", "all_gpus", "single_5090"]
    assert [m.devices for m in modes] == [(0, 1), (2, 3), (0, 1, 2, 3), (0,)]
    assert [m.label for m in modes] == [
        "2x RTX 5090",
        "2x RTX 3090",
        "all 4 GPUs (2x RTX 5090 + 2x RTX 3090)",
        "1x RTX 5090",
    ]


def test_the_best_pair_leads_because_that_is_what_the_user_asks_for_first() -> None:
    """`placements[0]` is the default load, so its order is a decision."""
    gpus = [gpu(i, "NVIDIA GeForce RTX 5090", 31.84) for i in range(2)]
    gpus += [gpu(i, "NVIDIA GeForce RTX 3090", 24.0) for i in (2, 3)]
    assert hardware_modes(gpus)[0].key == "dual_5090"


def test_a_single_gpu_box_gets_exactly_one_mode() -> None:
    modes = hardware_modes([gpu(0, "NVIDIA GeForce RTX 5090", 31.84)])
    assert [m.key for m in modes] == ["single_5090"]


def test_two_identical_cards_do_not_produce_a_duplicate_all_gpus_mode() -> None:
    """ "2x RTX 5090" and "all 2 GPUs" are the same placement; the first says more."""
    gpus = [gpu(i, "NVIDIA GeForce RTX 5090", 31.84) for i in range(2)]
    modes = hardware_modes(gpus)
    assert [m.key for m in modes] == ["dual_5090", "single_5090"]


def test_a_mixed_pair_has_no_dual_mode_of_its_own() -> None:
    """One card of each class: the only two-card option is the whole box."""
    gpus = [gpu(0, "NVIDIA GeForce RTX 5090", 31.84), gpu(1, "NVIDIA GeForce RTX 3090", 24.0)]
    assert [m.key for m in hardware_modes(gpus)] == ["all_gpus", "single_5090"]


def test_an_unnamed_card_is_classified_by_its_vram() -> None:
    """CUDA names are not guaranteed; 32 GiB on this class of box is a 5090."""
    gpus = [gpu(0, "Graphics Device", 31.84), gpu(1, "Graphics Device", 31.84)]
    assert [m.key for m in hardware_modes(gpus)] == ["dual_5090", "single_5090"]


def test_no_gpus_means_no_modes_rather_than_a_crash() -> None:
    assert hardware_modes([]) == []


# ---------------------------------------------------------------------------
# The per-mode optimal
# ---------------------------------------------------------------------------


def iswa_31b() -> GgufMeta:
    """An Ortenzya-shaped Gemma-4 31B: 60 layers, 50 sliding + 10 full."""
    return GgufMeta(
        architecture="gemma4",
        n_layer=60,
        n_head=32,
        n_head_kv=8,
        n_embd=5376,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=262144,
        sliding_window=1024,
        sliding_window_pattern=6,
        param_count=31_000_000_000,
        tensor_bytes=17 * GB,
        quant_label="Q4_K_M",
    )


def report_for(rec, *, free_gib: float = 31.0, loaded=()):  # noqa: ANN001, ANN201
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(free_gib), log_plans=False)
    return placement_report(
        rec,
        planner=planner,
        loaded=loaded,
        floor=config.models.default_ctx,
    )


def test_every_mode_reports_an_optimal_against_its_own_idle_cards() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    # Busy machine: the modes must not care.
    report = report_for(rec, free_gib=6.0)
    assert [e["mode"] for e in report] == ["dual_5090", "dual_3090", "all_gpus", "single_5090"]
    for entry in report:
        assert entry["optimal"] is not None, entry["mode"]
        assert entry["optimal"]["load_args"]["devices"] == entry["devices"]
        assert entry["fits_now"] is False  # 6 GiB free: nothing loads right now


def test_an_iswa_31b_on_the_two_5090s_is_wide_and_unquantized() -> None:
    """The acceptance case, and the whole point of the rule.

    17.4 GB of weights on two 5090s reaches 131072 tokens with an **f16** KV
    cache. 262144 is reachable there only with a quantized cache (80 GB of f16
    KV against 58 GB usable), and Gemma-4 measures KL 0.108 at q8_0 -- so
    quality-first stops at 131072 rather than doubling the window. All four
    cards do reach 262144 at f16, which is why that mode exists.
    """
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    report = report_for(rec)
    dual = next(e for e in report if e["mode"] == "dual_5090")
    assert dual["optimal"]["ctx_per_slot"] == 131072
    assert dual["optimal"]["kv_cache_type"] == "f16"
    assert dual["basis"] == "f16 KV, highest ctx 131072, 1 slot"
    every = next(e for e in report if e["mode"] == "all_gpus")
    assert every["optimal"]["ctx_per_slot"] == 262144
    assert every["optimal"]["kv_cache_type"] == "f16"


def test_the_quantized_262k_row_is_never_the_answer_for_a_gemma() -> None:
    """The live 2026-08-19 defect: 262144 on a q4_0 cache read as RECOMMENDED."""
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    for entry in report_for(rec):
        assert entry["optimal"]["kv_cache_type"] == "f16", entry["mode"]


def test_the_optimal_load_args_are_a_complete_load_model_call() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    args = report_for(rec)[0]["optimal"]["load_args"]
    assert set(args) >= {"model_id", "ctx_size", "parallel", "kv_cache_type", "devices"}
    assert args["model_id"] == "pub/iswa-31b"


def test_fits_now_is_false_and_would_evict_names_the_obstacle() -> None:
    resident = InstanceInfo(
        model_id="pub/other",
        state="ready",
        port=18100,
        ttl_s=1800,
        plan=LoadPlan(
            model_id="pub/other",
            devices=[0, 1],
            ctx_size=32768,
            per_gpu_bytes={0: int(25 * GB), 1: int(25 * GB)},
        ),
    )
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    dual = next(
        e for e in report_for(rec, free_gib=6.0, loaded=[resident]) if e["mode"] == "dual_5090"
    )
    assert dual["fits_now"] is False
    assert dual["would_evict"] == ["pub/other"]


def test_a_mode_too_small_for_the_model_says_so_rather_than_guessing() -> None:
    huge = record("pub/huge", dense_meta(262144), mtime=NOW, size_bytes=60 * GB)
    huge = huge.model_copy(update={"meta": huge.meta.model_copy(update={"tensor_bytes": 60 * GB})})
    single = next(e for e in report_for(huge) if e["mode"] == "single_5090")
    assert single["optimal"] is None
    assert "does not fit" in single["reason"]
    assert single["ranking"] == []


def test_the_rankings_are_assigned_across_modes() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    report = report_for(rec)
    labels = {label for e in report for label in e["ranking"]}
    assert "fastest" in labels
    assert "largest_context" in labels
    assert "cheapest" in labels


def test_a_pinned_kv_type_is_shown_beside_what_it_costs() -> None:
    """The two Gemma-4 QAT records on this rig pin q8_0; Gemma is KV-sensitive."""
    rec = record(
        "pub/iswa-31b",
        iswa_31b(),
        mtime=NOW,
        size_bytes=17 * GB,
        settings=ModelSettings(kv_cache_type="q8_0"),
    )
    dual = next(e for e in report_for(rec) if e["mode"] == "dual_5090")
    assert dual["optimal"]["kv_cache_type"] == "q8_0"
    assert dual["if_unpinned"]["kv_cache_type"] == "f16"


def test_a_pinned_placement_never_writes_back_to_the_record() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    report_for(rec)
    assert rec.settings.device_override is None
    assert rec.settings.kv_cache_type is None


# ---------------------------------------------------------------------------
# ...as the catalog surfaces it
# ---------------------------------------------------------------------------


def catalog_of(records, *, free_gib: float = 31.0, supervisor=None, **kwargs):  # noqa: ANN001, ANN201
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(free_gib))
    return build_catalog(
        registry=FakeRegistry(records),
        planner=planner,
        supervisor=supervisor or FakeSupervisor(),
        db=None,
        now=NOW,
        **kwargs,
    )


def test_the_catalog_recommends_a_placement_not_merely_a_context_size() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    entry = catalog_of([rec])["models"][0]
    assert entry["recommended"]["mode"] == "dual_5090"
    assert entry["recommended"]["devices"] == [0, 1]
    assert entry["recommended"]["load_args"]["devices"] == [0, 1]
    assert entry["recommended_basis"].startswith("2x RTX 5090:")
    assert entry["placements"][0]["mode"] == "dual_5090"


def test_the_compact_view_keeps_the_placements_the_user_asked_for() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    entry = catalog_of([rec], compact=True)["models"][0]
    assert [p["mode"] for p in entry["placements"]] == [
        "dual_5090",
        "dual_3090",
        "all_gpus",
        "single_5090",
    ]
    # would_evict collapses to a count, and only `recommended` carries load_args.
    assert entry["placements"][0]["would_evict"] == 0
    assert "load_args" not in entry["placements"][0]["optimal"]
    assert entry["recommended"]["load_args"]["devices"] == [0, 1]


def test_a_gemma_with_a_pinned_q8_cache_gets_a_quality_note() -> None:
    rec = record(
        "pub/iswa-31b",
        iswa_31b(),
        mtime=NOW,
        size_bytes=17 * GB,
        settings=ModelSettings(kv_cache_type="q8_0"),
    )
    entry = catalog_of([rec])["models"][0]
    assert entry["quality_notes"]
    assert "0.108" in entry["quality_notes"][0]
    assert "clear it" in entry["quality_notes"][0]


def test_a_kv_tolerant_family_gets_no_such_note() -> None:
    meta = iswa_31b().model_copy(update={"architecture": "qwen35"})
    rec = record(
        "pub/qwen",
        meta,
        mtime=NOW,
        size_bytes=17 * GB,
        settings=ModelSettings(kv_cache_type="q8_0"),
    )
    assert "quality_notes" not in catalog_of([rec])["models"][0]


def test_the_mode_probe_never_spills_onto_a_card_the_mode_excludes() -> None:
    """A mode that could quietly use a third card is not the mode it claims."""
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    for entry in report_for(rec):
        assert set(entry["optimal"]["devices"]) <= set(entry["devices"])


def test_a_one_card_box_reports_one_placement() -> None:
    config = make_config()
    probe = StubProbe([gpu(0, "NVIDIA GeForce RTX 5090", 31.84)])
    planner = Planner(config, probe, log_plans=False)
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    report = placement_report(rec, planner=planner, floor=config.models.default_ctx)
    assert [e["mode"] for e in report] == ["single_5090"]


# ---------------------------------------------------------------------------
# recommended_parallel: how many of those slots are worth running (WP19 / D37)
# ---------------------------------------------------------------------------


def parallel_sweep(devices: str, ctx: int, *rows) -> list[dict[str, object]]:  # noqa: ANN002
    """``(n_streams, per_stream_tps, aggregate_tps)`` triples as stored rows."""
    return [
        {
            "model_id": "pub/dense-8b",
            "ts": float(index),
            "run_id": "run-a",
            "devices": devices,
            "ctx_per_slot": ctx,
            "kv_cache_type": "f16",
            "kv_cache_type_v": "f16",
            "n_streams": n,
            "per_stream_tps": per_stream,
            "aggregate_tps": aggregate,
        }
        for index, (n, per_stream, aggregate) in enumerate(rows)
    ]


def test_every_placement_says_how_many_slots_are_worth_running() -> None:
    rec = record("pub/iswa-31b", iswa_31b(), mtime=NOW, size_bytes=17 * GB)
    for entry in report_for(rec):
        optimal = entry["optimal"]
        assert 1 <= optimal["recommended_parallel"] <= optimal["max_parallel"]
        assert optimal["recommended_parallel_basis"] == "estimated"
        # The recipe asks for the recommendation, not the ceiling.
        assert optimal["load_args"]["parallel"] == optimal["recommended_parallel"]


def test_a_measured_sweep_lowers_the_recommendation_and_the_recipe_follows() -> None:
    """The whole point: a measurement can say fewer slots than the arithmetic did."""
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(31.0), log_plans=False)
    # A 32k-window 8B, so the quality-first rule lands on a row with slots to
    # spare -- a 131072 row is one slot on any placement and has nothing to say.
    rec = record("pub/dense-8b", dense_meta(32768), mtime=NOW, size_bytes=8 * GB)
    plain = placement_report(rec, planner=planner, floor=config.models.default_ctx)
    head = plain[0]["optimal"]
    assert head["max_parallel"] > 1, "fixture must have room to be lowered"

    measured = placement_report(
        rec,
        planner=planner,
        floor=config.models.default_ctx,
        parallel_observations=parallel_sweep(
            ",".join(str(d) for d in sorted(head["devices"])),
            head["ctx_per_slot"],
            (1, 100.0, 100.0),
            (2, 90.0, 102.0),
        ),
    )
    optimal = measured[0]["optimal"]
    assert optimal["recommended_parallel"] == 1
    assert optimal["recommended_parallel_basis"] == "measured"
    assert optimal["max_parallel"] == head["max_parallel"]  # capacity is unchanged
    assert optimal["load_args"]["parallel"] == 1


def test_a_sweep_on_other_cards_does_not_steer_this_mode() -> None:
    """`basis: measured` has to be literally true of the placement it sits on."""
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(31.0), log_plans=False)
    rec = record("pub/dense-8b", dense_meta(32768), mtime=NOW, size_bytes=8 * GB)
    report = placement_report(
        rec,
        planner=planner,
        floor=config.models.default_ctx,
        parallel_observations=parallel_sweep("2,3", 8192, (1, 100.0, 100.0), (2, 90.0, 102.0)),
    )
    assert report[0]["optimal"]["recommended_parallel_basis"] == "estimated"
