"""The model catalog: ordering, tiers, the recommendation rule, load_args.

These tests describe the contract an LLM caller depends on. Three of them are
load-bearing beyond "the code works":

* **Newest first** is the one ordering the catalog promises, because the user
  works from what they last downloaded.
* **``load_args`` is passed verbatim to ``load_model``**, so its keys are
  checked against that tool's real signature -- a rename there must fail here
  rather than in an agent's tool call.
* **Exactly one recommended row per model**, so "pick the recommended one" is
  an instruction that always has an answer.

The library is a fake of three models plus edge cases, shaped like the real
one: a big MoE, an ordinary dense model, a vision model with an mmproj.
"""

from __future__ import annotations

import inspect

import pytest

from studioforge.core import throughput
from studioforge.core.catalog import (
    CATALOG_HINT,
    CTX_TIERS,
    build_catalog,
    capability_list,
    compact_row,
    ctx_tiers_for,
    mark_recommended,
    model_type,
    pinned_settings,
    summarize,
)
from studioforge.core.planner import Planner
from studioforge.types import (
    GB,
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
    ModelSettings,
)
from tests.unit.test_planner import make_config, rig_5090x2_3090x2

DAY = 86400.0


# ---------------------------------------------------------------------------
# A fake library
# ---------------------------------------------------------------------------


def moe_meta() -> GgufMeta:
    """The reference rig's resident 122B, to its real GGUF dimensions.

    48 layers, 2 KV heads, head_dim 256 (96 KiB/token at f16), and -- the part
    that matters for the active-parameter count -- 8 routed experts of 256,
    which is 3.1% of the file while the model is named A10B.
    """
    return GgufMeta(
        architecture="qwen3moe",
        n_layer=48,
        n_head=32,
        n_head_kv=2,
        n_embd=3072,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_vocab=248320,
        n_expert=256,
        n_expert_used=8,
        n_ctx_train=262144,
        param_count=122_000_000_000,
        tensor_bytes=60 * GB,
        quant_label="Q5_K_M",
        chat_template="{% if tools %}...{% endif %}",
    )


def dense_meta(n_ctx_train: int = 131072) -> GgufMeta:
    """An ordinary 8B: 36 layers, 8 KV heads, head_dim 128 -> 144 KiB/token."""
    return GgufMeta(
        architecture="llama",
        n_layer=36,
        n_head=32,
        n_head_kv=8,
        n_embd=4096,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=n_ctx_train,
        param_count=8_000_000_000,
        tensor_bytes=8 * GB,
        quant_label="Q8_0",
    )


def vlm_meta() -> GgufMeta:
    """A 7B vision model with a short trained window."""
    return GgufMeta(
        architecture="qwen2vl",
        n_layer=28,
        n_head=28,
        n_head_kv=4,
        n_embd=3584,
        n_embd_head_k=128,
        n_embd_head_v=128,
        n_ctx_train=32768,
        param_count=7_000_000_000,
        tensor_bytes=5 * GB,
        quant_label="Q4_K_S",
        has_vision_tensors=True,
    )


def record(
    model_id: str,
    meta: GgufMeta | None,
    *,
    mtime: float,
    size_bytes: int,
    kind: str = "chat",
    vision: bool = False,
    tools: bool = False,
    thinking: bool = False,
    mmproj: str | None = None,
    settings: ModelSettings | None = None,
) -> ModelRecord:
    return ModelRecord(
        id=model_id,
        name=model_id.rsplit("/", 1)[-1],
        kind=kind,
        path=f"/models/{model_id}.gguf",
        size_bytes=size_bytes,
        quant=meta.quant_label if meta else "unknown",
        architecture=meta.architecture if meta else "unknown",
        meta=meta,
        mtime=mtime,
        added_at=mtime,
        capabilities=ModelCapabilities(vision=vision, tools=tools, thinking=thinking),
        mmproj_path=mmproj,
        mmproj_bytes=800 * 1024 * 1024 if mmproj else 0,
        settings=settings or ModelSettings(),
    )


class FakeRegistry:
    def __init__(self, records: list[ModelRecord]) -> None:
        self._records = list(records)

    def all(self) -> list[ModelRecord]:
        return list(self._records)

    def get(self, model_id: str) -> ModelRecord | None:
        return next((r for r in self._records if r.id == model_id), None)

    def resolve(self, name: str) -> ModelRecord | None:
        found = self.get(name)
        if found is not None:
            return found
        return next((r for r in self._records if r.name == name), None)

    def known_ids(self) -> list[str]:
        return [r.id for r in self._records]


class FakeSupervisor:
    def __init__(self, instances: list[InstanceInfo] | None = None) -> None:
        self._instances = instances or []

    def list(self) -> list[InstanceInfo]:
        return list(self._instances)


class FakeDb:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = rows or []

    def throughput_observations(self, model_id=None, limit=200):  # noqa: ANN001, ANN201
        if model_id is None:
            return list(self.rows)
        return [r for r in self.rows if r.get("model_id") == model_id]


NOW = 1_755_000_000.0


def library() -> list[ModelRecord]:
    """Newest to oldest: MoE (today), dense (yesterday), VLM (a week ago)."""
    return [
        # Deliberately NOT in date order, so the sort is doing the work.
        record("pub/dense-8b", dense_meta(), mtime=NOW - DAY, size_bytes=8 * GB, tools=True),
        record(
            "pub/moe-122b", moe_meta(), mtime=NOW, size_bytes=60 * GB, tools=True, thinking=True
        ),
        record(
            "pub/vlm-7b",
            vlm_meta(),
            mtime=NOW - 7 * DAY,
            size_bytes=5 * GB,
            vision=True,
            mmproj="/models/pub/vlm-7b-mmproj.gguf",
        ),
    ]


def catalog_for(
    records: list[ModelRecord] | None = None,
    *,
    free_gib: float = 31.0,
    supervisor: FakeSupervisor | None = None,
    db: FakeDb | None = None,
    **kwargs,
) -> dict:
    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2(free_gib))
    return build_catalog(
        registry=FakeRegistry(records if records is not None else library()),
        planner=planner,
        supervisor=supervisor or FakeSupervisor(),
        db=db,
        now=NOW,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Ordering -- the one guarantee
# ---------------------------------------------------------------------------


def test_models_are_sorted_newest_download_first() -> None:
    """The user works from the last thing they downloaded."""
    result = catalog_for()
    assert [m["id"] for m in result["models"]] == [
        "pub/moe-122b",
        "pub/dense-8b",
        "pub/vlm-7b",
    ]


def test_download_date_is_iso_and_comes_from_the_file_mtime() -> None:
    entry = catalog_for()["models"][0]
    assert entry["downloaded_at"].endswith("Z")
    assert entry["downloaded_at"].startswith("20")


def test_a_model_with_no_timestamp_sorts_last_rather_than_crashing() -> None:
    records = library()
    records.append(record("pub/undated", dense_meta(), mtime=0.0, size_bytes=8 * GB))
    result = catalog_for(records)
    assert result["models"][-1]["id"] == "pub/undated"


def test_the_internal_sort_key_never_reaches_the_client() -> None:
    for entry in catalog_for()["models"]:
        assert "downloaded_at_ts" not in entry


# ---------------------------------------------------------------------------
# Context tiers
# ---------------------------------------------------------------------------


def test_tiers_are_capped_at_the_trained_window() -> None:
    """Serving past n_ctx_train needs RoPE scaling and degrades quality (D14)."""
    vlm = next(m for m in catalog_for()["models"] if m["id"] == "pub/vlm-7b")
    assert [r["ctx_per_slot"] for r in vlm["options"]] == [16384, 32768]


def test_a_long_context_model_gets_the_whole_ladder_up_to_its_window() -> None:
    moe = next(m for m in catalog_for()["models"] if m["id"] == "pub/moe-122b")
    assert [r["ctx_per_slot"] for r in moe["options"]] == [16384, 32768, 65536, 131072, 262144]


def test_a_model_shorter_than_the_smallest_tier_still_gets_a_row() -> None:
    """Its own window is the only meaningful option; an empty table helps nobody."""
    short = record("pub/tiny", dense_meta(n_ctx_train=8192), mtime=NOW, size_bytes=2 * GB)
    assert ctx_tiers_for(short) == [8192]


def test_a_pinned_ctx_size_is_added_to_the_table() -> None:
    """A pinned value is what a load actually uses, so it must be describable."""
    pinned = record(
        "pub/pinned",
        dense_meta(),
        mtime=NOW,
        size_bytes=8 * GB,
        settings=ModelSettings(ctx_size=24576),
    )
    tiers = ctx_tiers_for(pinned)
    assert 24576 in tiers
    assert tiers == sorted(tiers)


def test_tiers_are_the_documented_ladder() -> None:
    assert CTX_TIERS == (16384, 32768, 65536, 131072, 262144, 524288, 1048576)


# ---------------------------------------------------------------------------
# The recommendation rule
# ---------------------------------------------------------------------------


def test_exactly_one_row_per_model_is_recommended() -> None:
    """ "Pick the recommended row" must always have exactly one answer."""
    for entry in catalog_for()["models"]:
        marked = [r for r in entry["options"] if r["recommended"]]
        assert len(marked) == 1, entry["id"]


def test_a_chat_model_trades_context_for_a_second_conversation() -> None:
    """Above the floor, one slot means every request queues -- fatal on an agent host."""
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    chosen = next(r for r in dense["options"] if r["recommended"])
    assert chosen["max_parallel"] >= 2
    assert dense["recommended_basis"] == "highest ctx >= floor with max_parallel >= 2"
    # ...and it really is the *highest* such context, not merely one of them.
    better = [
        r
        for r in dense["options"]
        if r["fits"] and r["max_parallel"] >= 2 and r["ctx_per_slot"] > chosen["ctx_per_slot"]
    ]
    assert better == []


def test_a_non_chat_model_just_takes_the_biggest_window() -> None:
    """Embeddings are called in bursts; a single slot is not the bottleneck."""
    rows = [
        {"ctx_per_slot": 16384, "fits": True, "max_parallel": 8, "recommended": False},
        {"ctx_per_slot": 32768, "fits": True, "max_parallel": 1, "recommended": False},
    ]
    basis = mark_recommended(rows, chat_class=False, floor=16384)
    assert basis == "highest ctx that fits >= floor"
    assert rows[1]["recommended"] is True


def test_a_chat_model_with_no_concurrent_row_falls_back_to_the_biggest() -> None:
    rows = [
        {"ctx_per_slot": 16384, "fits": True, "max_parallel": 1, "recommended": False},
        {"ctx_per_slot": 32768, "fits": True, "max_parallel": 1, "recommended": False},
    ]
    assert mark_recommended(rows, chat_class=True, floor=16384) == "highest ctx that fits >= floor"
    assert rows[1]["recommended"] is True


# -- the context floor (D22) ------------------------------------------------


def test_the_floor_outranks_a_second_slot() -> None:
    """The failure this rule exists to stop: 16k with two slots over 32k with one.

    An OpenClaw agent's tool transcript stops fitting below the configured
    default context. A queued second conversation is a latency problem; a
    window that cannot hold the task is a failed task.
    """
    rows = [
        {"ctx_per_slot": 16384, "fits": True, "max_parallel": 2, "recommended": False},
        {"ctx_per_slot": 32768, "fits": True, "max_parallel": 1, "recommended": False},
    ]
    basis = mark_recommended(rows, chat_class=True, floor=32768)
    assert basis == "highest ctx that fits >= floor"
    assert rows[1]["recommended"] is True
    assert rows[0]["recommended"] is False


def test_the_second_slot_still_wins_among_rows_above_the_floor() -> None:
    """The floor only removes candidates; above it, concurrency is still worth a doubling."""
    rows = [
        {"ctx_per_slot": 32768, "fits": True, "max_parallel": 4, "recommended": False},
        {"ctx_per_slot": 65536, "fits": True, "max_parallel": 1, "recommended": False},
    ]
    basis = mark_recommended(rows, chat_class=True, floor=32768)
    assert basis == "highest ctx >= floor with max_parallel >= 2"
    assert rows[0]["recommended"] is True


def test_nothing_reaching_the_floor_still_gets_a_recommendation() -> None:
    """A small window beats no answer -- but the basis says so out loud."""
    rows = [
        {"ctx_per_slot": 16384, "fits": True, "max_parallel": 4, "recommended": False},
        {"ctx_per_slot": 32768, "fits": True, "max_parallel": 1, "recommended": False},
    ]
    basis = mark_recommended(rows, chat_class=True, floor=131072)
    assert basis == "highest ctx that fits (below floor)"
    assert rows[1]["recommended"] is True


def test_a_thinking_model_gets_the_higher_floor() -> None:
    """It spends its budget reasoning before it answers -- that is why the setting exists."""
    from studioforge.core.catalog import recommendation_floor

    config = make_config()
    plain = record("x", dense_meta(), mtime=NOW, size_bytes=8 * GB)
    reasoner = record("y", dense_meta(), mtime=NOW, size_bytes=8 * GB, thinking=True)
    assert recommendation_floor(config, plain) == config.models.default_ctx
    assert recommendation_floor(config, reasoner) == config.models.thinking_default_ctx
    assert recommendation_floor(config, reasoner) > recommendation_floor(config, plain)


def test_the_idle_fallback_applies_the_same_preference_order() -> None:
    """ "Unload something" still deserves the best row, not merely the biggest."""
    rows = [
        {
            "ctx_per_slot": 16384,
            "fits": False,
            "if_gpus_idle": {"fits": True, "max_parallel": 4},
            "recommended": False,
        },
        {
            "ctx_per_slot": 32768,
            "fits": False,
            "if_gpus_idle": {"fits": True, "max_parallel": 4},
            "recommended": False,
        },
        {
            "ctx_per_slot": 65536,
            "fits": False,
            "if_gpus_idle": {"fits": True, "max_parallel": 1},
            "recommended": False,
        },
    ]
    assert mark_recommended(rows, chat_class=True, floor=16384) == "if_gpus_idle"
    assert rows[1]["recommended"] is True


def test_when_nothing_fits_the_recommendation_points_at_freeing_vram() -> None:
    """ "Unload something" is actionable; "impossible" is not."""
    rows = [
        {
            "ctx_per_slot": 16384,
            "fits": False,
            "if_gpus_idle": {"fits": True},
            "recommended": False,
        },
        {
            "ctx_per_slot": 32768,
            "fits": False,
            "if_gpus_idle": {"fits": False},
            "recommended": False,
        },
    ]
    assert mark_recommended(rows, chat_class=True) == "if_gpus_idle"
    assert rows[0]["recommended"] is True


def test_a_model_that_fits_nowhere_recommends_nothing() -> None:
    rows = [
        {
            "ctx_per_slot": 16384,
            "fits": False,
            "if_gpus_idle": {"fits": False},
            "recommended": False,
        }
    ]
    assert mark_recommended(rows, chat_class=True) is None


# ---------------------------------------------------------------------------
# load_args -- the whole point
# ---------------------------------------------------------------------------


def test_load_args_keys_match_the_load_model_tool_signature() -> None:
    """A rename in the MCP tool must break here, not in an agent's tool call."""
    from studioforge.mcp.management import build_management_mcp

    class State:
        config = make_config()

    server = build_management_mcp(State())
    tool = next(t for t in server._tool_manager.list_tools() if t.name == "load_model")
    accepted = set(tool.parameters["properties"])

    row = next(r for m in catalog_for()["models"] for r in m["options"] if r["recommended"])
    assert set(row["load_args"]) <= accepted
    assert set(row["load_args"]) == {"model_id", "ctx_size", "parallel", "kv_cache_type"}


def test_load_args_repeat_the_row_they_belong_to() -> None:
    """An agent that passes them through must get the load the row described."""
    for entry in catalog_for()["models"]:
        for row in entry["options"]:
            if not row["fits"]:
                assert row["load_args"] is None
                continue
            args = row["load_args"]
            assert args["model_id"] == entry["id"]
            assert args["ctx_size"] == row["ctx_per_slot"]
            assert args["parallel"] == row["max_parallel"]
            assert args["kv_cache_type"] == row["kv_cache_type"]


def test_load_args_carry_a_concrete_kv_cache_type_never_auto() -> None:
    """ "auto" is a planner input, not an answer; the row states what it picked."""
    for entry in catalog_for()["models"]:
        for row in entry["options"]:
            if row["fits"]:
                assert row["kv_cache_type"] in {"f16", "q8_0", "q4_0"}


# ---------------------------------------------------------------------------
# Placement, concurrency and the idle variant
# ---------------------------------------------------------------------------


def test_a_fitting_row_names_the_devices_the_planner_chose() -> None:
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    row = next(r for r in dense["options"] if r["ctx_per_slot"] == 16384)
    assert row["fits"] is True
    assert row["devices"]
    assert all(0 <= d <= 3 for d in row["devices"])
    assert row["vram_mb"] > 8000


def test_a_big_model_is_split_across_several_gpus() -> None:
    moe = next(m for m in catalog_for()["models"] if m["id"] == "pub/moe-122b")
    row = next(r for r in moe["options"] if r["fits"])
    assert len(row["devices"]) > 1


def test_more_context_costs_concurrency() -> None:
    """The trade-off the table exists to make visible."""
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    fitting = [r for r in dense["options"] if r["fits"]]
    slots = [r["max_parallel"] for r in fitting]
    assert slots == sorted(slots, reverse=True)


def test_every_row_says_what_bounds_its_concurrency() -> None:
    for entry in catalog_for()["models"]:
        for row in entry["options"]:
            assert row["parallel_limited_by"] in {"vram", "knee", "cap", "explicit", "unknown"}


def test_if_gpus_idle_shows_what_a_cleared_rig_could_do() -> None:
    """With every card busy, live rows fail and the idle column still answers."""
    busy = catalog_for(free_gib=2.0)
    dense = next(m for m in busy["models"] if m["id"] == "pub/dense-8b")
    row = dense["options"][0]
    assert row["fits"] is False
    assert row["if_gpus_idle"]["fits"] is True
    assert row["if_gpus_idle"]["devices"]


def test_if_gpus_idle_is_present_on_fitting_rows_too() -> None:
    """It is a column, not an error path: an agent can always compare."""
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    for row in dense["options"]:
        assert "if_gpus_idle" in row


def test_a_row_that_does_not_fit_carries_a_reason() -> None:
    busy = catalog_for(free_gib=2.0)
    row = busy["models"][0]["options"][0]
    assert row["fits"] is False
    assert row["reason"]
    assert row["load_args"] is None


# ---------------------------------------------------------------------------
# Speed columns
# ---------------------------------------------------------------------------


def test_fitting_rows_carry_speed_estimates() -> None:
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    row = next(r for r in dense["options"] if r["fits"])
    assert row["est_gen_tps"] > 0
    assert row["est_prompt_tps"] > 0
    assert row["est_gen_tps_batched"] >= row["est_gen_tps"]


def test_speed_is_estimated_until_something_is_measured() -> None:
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    row = next(r for r in dense["options"] if r["fits"])
    assert row["confidence"] == "estimated"
    assert row["measured_gen_tps"] is None


def test_a_matching_observation_makes_a_row_measured() -> None:
    """The word "measured" in the output must be literally true."""
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    target = next(r for r in dense["options"] if r["fits"])
    rows = [
        {
            "model_id": "pub/dense-8b",
            "devices": ",".join(str(d) for d in sorted(target["devices"])),
            "gpu_class": "RTX 5090x1",
            "ctx_size": target["ctx_per_slot"],
            "parallel": target["max_parallel"],
            "gen_tps": 111.0,
            "est_gen_tps": 150.0,
            "prompt_tps": 900.0,
            "est_prompt_tps": 1000.0,
        }
    ] * 2
    result = catalog_for(db=FakeDb(rows))
    dense = next(m for m in result["models"] if m["id"] == "pub/dense-8b")
    row = next(r for r in dense["options"] if r["ctx_per_slot"] == target["ctx_per_slot"])
    assert row["measured_gen_tps"] == 111.0
    assert row["confidence"] == "measured"


def test_observations_calibrate_the_rows_that_were_not_measured() -> None:
    """A model observed at one context gets a corrected estimate at every other."""
    baseline = catalog_for()
    dense_baseline = next(m for m in baseline["models"] if m["id"] == "pub/dense-8b")
    target = next(r for r in dense_baseline["options"] if r["fits"])
    rows = [
        {
            "model_id": "pub/dense-8b",
            "devices": ",".join(str(d) for d in sorted(target["devices"])),
            "gpu_class": "RTX 5090x1",
            "ctx_size": target["ctx_per_slot"],
            "parallel": target["max_parallel"],
            "gen_tps": 50.0,
            "est_gen_tps": 100.0,
            "prompt_tps": 500.0,
            "est_prompt_tps": 1000.0,
            # Only rows recorded by the current estimator calibrate anything:
            # a ratio is a correction to one specific formula (D22).
            "estimator_version": throughput.ESTIMATOR_VERSION,
        }
    ] * 3
    result = catalog_for(db=FakeDb(rows))
    dense = next(m for m in result["models"] if m["id"] == "pub/dense-8b")
    assert dense["calibration"]["gen_factor"] == pytest.approx(0.5)
    assert dense["calibration"]["basis"] == "model+devices"
    other = next(
        r for r in dense["options"] if r["fits"] and r["ctx_per_slot"] != target["ctx_per_slot"]
    )
    baseline_other = next(
        r for r in dense_baseline["options"] if r["ctx_per_slot"] == other["ctx_per_slot"]
    )
    assert other["est_gen_tps"] == pytest.approx(baseline_other["est_gen_tps"] * 0.5, rel=0.02)
    assert other["confidence"] == "calibrated"


def test_a_missing_database_is_not_an_error() -> None:
    result = catalog_for(db=None)
    assert result["models"]


def test_every_fitting_row_quotes_both_ends_of_the_window() -> None:
    """One number cannot describe a row: decode slows as the KV cache fills."""
    for entry in catalog_for()["models"]:
        for row in entry["options"]:
            if not row["fits"]:
                assert row["est_gen_tps_full_ctx"] is None
                continue
            assert row["est_gen_tps_full_ctx"] > 0
            assert row["est_gen_tps_full_ctx"] <= row["est_gen_tps"]


def test_a_window_at_or_below_the_reference_fill_quotes_one_number_twice() -> None:
    """8k of context in an 8k window *is* the full-context case."""
    short = record("pub/short", dense_meta(n_ctx_train=8192), mtime=NOW, size_bytes=2 * GB)
    entry = catalog_for([short])["models"][0]
    row = next(r for r in entry["options"] if r["fits"])
    assert row["ctx_per_slot"] <= throughput.REFERENCE_FILL_TOKENS
    assert row["est_gen_tps_full_ctx"] == row["est_gen_tps"]


def test_the_full_context_column_falls_away_from_the_reference_one() -> None:
    """The gap is the point: a 131k row is materially slower than an 8k turn."""
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    widest = max((r for r in dense["options"] if r["fits"]), key=lambda r: r["ctx_per_slot"])
    assert widest["ctx_per_slot"] >= 65536
    assert widest["est_gen_tps_full_ctx"] < widest["est_gen_tps"]


def test_the_idle_column_carries_the_full_context_speed_too() -> None:
    busy = catalog_for(free_gib=2.0)
    dense = next(m for m in busy["models"] if m["id"] == "pub/dense-8b")
    idle = dense["options"][0]["if_gpus_idle"]
    assert idle["fits"] is True
    assert idle["est_gen_tps_full_ctx"] is not None


# ---------------------------------------------------------------------------
# Attention kind -- why one model's 262k is cheap and another's is not
# ---------------------------------------------------------------------------


def iswa_meta() -> GgufMeta:
    """A Gemma-4-shaped 31B: five sliding-window layers per full-attention one."""
    return GgufMeta(
        architecture="gemma4",
        n_layer=60,
        n_head=32,
        n_head_kv=4,
        n_embd=4096,
        n_embd_head_k=512,
        n_embd_head_v=512,
        n_ctx_train=262144,
        param_count=31_000_000_000,
        tensor_bytes=30 * GB,
        quant_label="Q8_0",
        extra={
            "swa_pattern": [True, True, True, True, True, False],
            "swa_window": 1024,
            "swa_key_length": 256,
            "swa_value_length": 256,
            "head_count_kv_values": [16, 16, 16, 16, 16, 4],
        },
    )


def hybrid_meta() -> GgufMeta:
    """A Qwen3.5-shaped 27B: only every fourth block keeps a KV cache."""
    return GgufMeta(
        architecture="qwen35",
        n_layer=64,
        n_head=24,
        n_head_kv=4,
        n_embd=6144,
        n_embd_head_k=256,
        n_embd_head_v=256,
        n_ctx_train=262144,
        param_count=27_000_000_000,
        tensor_bytes=20 * GB,
        quant_label="Q5_K_M",
        extra={
            "full_attention_interval": 4,
            "ssm_conv_kernel": 4,
            "ssm_inner_size": 6144,
            "ssm_state_size": 128,
            "ssm_group_count": 16,
        },
    )


def test_every_model_reports_how_its_attention_is_shaped() -> None:
    records = [
        record("pub/dense-8b", dense_meta(), mtime=NOW, size_bytes=8 * GB),
        record("pub/gemma4-31b", iswa_meta(), mtime=NOW - DAY, size_bytes=30 * GB),
        record("pub/qwen35-27b", hybrid_meta(), mtime=NOW - 2 * DAY, size_bytes=20 * GB),
    ]
    kinds = {m["id"]: m["attention_kind"] for m in catalog_for(records)["models"]}
    assert kinds == {
        "pub/dense-8b": "full",
        "pub/gemma4-31b": "iswa",
        "pub/qwen35-27b": "hybrid",
    }


def test_the_summary_tags_the_two_kinds_that_change_what_context_costs() -> None:
    """A forty-line list has to show it: "full" is the default nobody needs told."""
    plain = record("pub/dense-8b", dense_meta(), mtime=NOW, size_bytes=8 * GB)
    assert "iSWA" not in summarize(plain, 8.0, 8.0, "full")
    assert "hybrid" not in summarize(plain, 8.0, 8.0, "unknown")
    assert "iSWA" in summarize(plain, 8.0, 8.0, "iswa")
    assert "hybrid" in summarize(plain, 8.0, 8.0, "hybrid")


def test_the_summary_carries_the_tag_through_the_real_catalog() -> None:
    entry = catalog_for([record("pub/gemma4-31b", iswa_meta(), mtime=NOW, size_bytes=30 * GB)])[
        "models"
    ][0]
    assert "iSWA" in entry["summary"]


def test_a_model_with_no_usable_geometry_says_unknown_not_full() -> None:
    """ "Cannot estimate" must never be reported as the cheap case."""
    blind = record(
        "pub/blind",
        GgufMeta(architecture="mystery", n_ctx_train=32768, tensor_bytes=4 * GB),
        mtime=NOW,
        size_bytes=4 * GB,
    )
    entry = catalog_for([blind])["models"][0]
    assert entry["attention_kind"] == "unknown"


# ---------------------------------------------------------------------------
# Model-level fields
# ---------------------------------------------------------------------------


def test_moe_reports_both_total_and_active_parameters() -> None:
    moe = next(m for m in catalog_for()["models"] if m["id"] == "pub/moe-122b")
    assert moe["is_moe"] is True
    assert moe["params_total_b"] == pytest.approx(122.0)
    # Dense trunk (attention + lm_head, ~3.3B) plus 8/256 of the experts.
    # NOT the flat routed share, which would say 3.8B -- see
    # throughput.active_params.
    assert 6.5 <= moe["params_active_b"] <= 7.6


def test_a_dense_model_has_no_separate_active_count() -> None:
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    assert dense["is_moe"] is False
    assert dense["params_total_b"] == dense["params_active_b"]


def test_a_vision_model_is_typed_vlm_and_reports_its_projector() -> None:
    vlm = next(m for m in catalog_for()["models"] if m["id"] == "pub/vlm-7b")
    assert vlm["type"] == "vlm"
    assert vlm["has_mmproj"] is True
    assert "vision" in vlm["capabilities"]


def test_model_type_matches_the_lmstudio_surface() -> None:
    """One implementation behind /api/v0/models and the catalog."""
    from studioforge.api.openai_routes import _v0_type

    for rec in library():
        assert model_type(rec) == _v0_type(rec)


def test_embedding_models_are_typed_as_embeddings() -> None:
    rec = record("pub/embed", dense_meta(), mtime=NOW, size_bytes=1 * GB, kind="embedding")
    assert model_type(rec) == "embeddings"


def test_summary_is_one_readable_line() -> None:
    moe = next(m for m in catalog_for()["models"] if m["id"] == "pub/moe-122b")
    summary = moe["summary"]
    assert "\n" not in summary
    assert "Q5_K_M" in summary
    assert "MoE" in summary
    assert "qwen3moe" in summary


def test_summary_names_capabilities() -> None:
    rec = record("x", dense_meta(), mtime=NOW, size_bytes=8 * GB, tools=True, thinking=True)
    assert "tools" in summarize(rec, 8.0, 8.0)
    assert "thinking" in summarize(rec, 8.0, 8.0)


def test_capability_list_only_names_what_is_true() -> None:
    rec = record("x", dense_meta(), mtime=NOW, size_bytes=8 * GB, tools=True)
    assert capability_list(rec) == ["tools"]


def test_pinned_settings_reports_only_real_overrides() -> None:
    """So an agent can see why a model ignored the ctx_size it asked for."""
    plain = record("x", dense_meta(), mtime=NOW, size_bytes=8 * GB)
    assert pinned_settings(plain) == {}

    overridden = record(
        "y",
        dense_meta(),
        mtime=NOW,
        size_bytes=8 * GB,
        settings=ModelSettings(ctx_size=65536, device_override=[2], pinned=True, ttl_s=0),
    )
    pinned = pinned_settings(overridden)
    assert pinned["ctx_size"] == 65536
    assert pinned["device_override"] == [2]
    assert pinned["pinned"] is True


def test_the_loaded_model_reports_its_running_plan() -> None:
    instance = InstanceInfo(
        model_id="pub/dense-8b",
        state="ready",
        port=18100,
        plan=LoadPlan(
            model_id="pub/dense-8b",
            devices=[0],
            ctx_size=32768,
            parallel=3,
            max_parallel=3,
            parallel_limited_by="knee",
            ctx_per_slot=32768,
        ),
    )
    result = catalog_for(supervisor=FakeSupervisor([instance]))
    dense = next(m for m in result["models"] if m["id"] == "pub/dense-8b")
    assert dense["state"] == "loaded"
    assert dense["port"] == 18100
    assert dense["loaded_plan"]["ctx_per_slot"] == 32768
    assert dense["loaded_plan"]["parallel"] == 3
    assert dense["loaded_plan"]["parallel_limited_by"] == "knee"


def test_an_unloaded_model_says_so_plainly() -> None:
    dense = next(m for m in catalog_for()["models"] if m["id"] == "pub/dense-8b")
    assert dense["state"] == "not-loaded"
    assert dense["loaded_plan"] is None


def test_state_uses_the_same_word_as_the_other_model_surfaces() -> None:
    """One vocabulary across /v1/models, /api/v0/models and the catalog."""
    for entry in catalog_for()["models"]:
        assert entry["state"] in {"loaded", "not-loaded"}


# ---------------------------------------------------------------------------
# Filtering, compaction, robustness
# ---------------------------------------------------------------------------


def test_compact_keeps_only_the_recommended_row() -> None:
    full = catalog_for()
    compact = catalog_for(compact=True)
    assert compact["compact"] is True
    for entry in compact["models"]:
        assert len(entry["options"]) <= 1
    assert sum(len(m["options"]) for m in compact["models"]) < sum(
        len(m["options"]) for m in full["models"]
    )


def test_compact_drops_an_idle_verdict_that_only_differs_in_device_order() -> None:
    """[1, 0] and [0, 1] are the same placement; shipping both is pure noise.

    The planner orders devices by its own candidate preference, which depends on
    free VRAM -- so a busy rig and an idle one routinely name the same two cards
    in different orders, and comparing the lists kept the duplicate on nearly
    every row.
    """
    row = {
        "ctx_per_slot": 65536,
        "fits": True,
        "devices": [1, 0],
        "max_parallel": 2,
        "if_gpus_idle": {"fits": True, "devices": [0, 1], "max_parallel": 2},
    }
    assert "if_gpus_idle" not in compact_row(row)


def test_compact_keeps_an_idle_verdict_that_names_a_different_placement() -> None:
    row = {
        "ctx_per_slot": 65536,
        "fits": True,
        "devices": [2, 3],
        "max_parallel": 1,
        "if_gpus_idle": {"fits": True, "devices": [0, 1], "max_parallel": 4},
    }
    assert compact_row(row)["if_gpus_idle"]["devices"] == [0, 1]


def test_compact_keeps_the_idle_verdict_on_a_row_that_does_not_fit() -> None:
    """That is the whole reason the column exists."""
    row = {
        "ctx_per_slot": 65536,
        "fits": False,
        "devices": [],
        "if_gpus_idle": {"fits": True, "devices": [0, 1], "max_parallel": 2},
    }
    assert "if_gpus_idle" in compact_row(row)


def test_filtering_to_one_model_returns_only_it() -> None:
    result = catalog_for(model="pub/vlm-7b")
    assert [m["id"] for m in result["models"]] == ["pub/vlm-7b"]
    assert result["count"] == 1


def test_filtering_to_an_unknown_model_returns_nothing_rather_than_raising() -> None:
    """The manager's catalog() raises 404; the builder itself is total."""
    assert catalog_for(model="nope")["models"] == []


def test_a_model_with_unparsed_metadata_is_listed_not_hidden() -> None:
    """The catalog is the only place a user would learn the file is broken."""
    records = library()
    records.append(record("pub/broken", None, mtime=NOW - 2 * DAY, size_bytes=4 * GB))
    result = catalog_for(records)
    broken = next(m for m in result["models"] if m["id"] == "pub/broken")
    assert broken["options"] == []
    assert "unavailable" in broken


def test_the_hint_is_returned_with_every_catalog() -> None:
    result = catalog_for()
    assert result["catalog_hint"] == CATALOG_HINT
    assert "recommended" in result["catalog_hint"]
    assert "load_args" in result["catalog_hint"]


def test_the_catalog_reports_the_vram_snapshot_it_was_built_from() -> None:
    """Every row describes the same instant, and the client can see which."""
    result = catalog_for()
    assert len(result["gpus"]) == 4
    assert result["generated_at"].endswith("Z")
    assert result["gpu_class"] == "RTX 3090x2+RTX 5090x2"


def test_an_empty_library_is_an_empty_catalog_not_a_crash() -> None:
    result = catalog_for([])
    assert result["models"] == []
    assert result["count"] == 0


def test_a_sick_supervisor_does_not_empty_the_catalog() -> None:
    class Broken:
        def list(self):  # noqa: ANN201
            raise RuntimeError("supervisor is wedged")

    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2())
    result = build_catalog(
        registry=FakeRegistry(library()), planner=planner, supervisor=Broken(), now=NOW
    )
    assert len(result["models"]) == 3


def test_build_catalog_takes_only_keyword_arguments() -> None:
    """Positional args here would be four interchangeable objects."""
    signature = inspect.signature(build_catalog)
    assert all(p.kind is p.KEYWORD_ONLY for p in signature.parameters.values())


# ---------------------------------------------------------------------------
# The calibration loop that feeds the speed columns
# ---------------------------------------------------------------------------


class RecordingDb:
    """Captures what the sweeper writes, and nothing else."""

    def __init__(self) -> None:
        self.observations: list[dict] = []

    def record_throughput_observation(self, **fields) -> None:  # noqa: ANN003
        self.observations.append(fields)

    def prune_throughput_observations(self, **_kwargs) -> None:  # noqa: ANN003
        return None


class MetricsSupervisor(FakeSupervisor):
    def __init__(self, instances, text: str) -> None:  # noqa: ANN001
        super().__init__(instances)
        self._text = text

    async def metrics(self, model_id: str) -> str:  # noqa: ARG002
        return self._text


def _metrics_text(*, gen_tokens: float, gen_seconds: float, decodes: float) -> str:
    return "\n".join(
        [
            f"llamacpp:prompt_tokens_total {gen_tokens * 10}",
            "llamacpp:prompt_seconds_total 1.0",
            f"llamacpp:tokens_predicted_total {gen_tokens}",
            f"llamacpp:tokens_predicted_seconds_total {gen_seconds}",
            f"llamacpp:n_decode_total {decodes}",
            "llamacpp:n_busy_slots_per_decode 1",
            "llamacpp:requests_deferred 0",
        ]
    )


async def test_a_recorded_observation_is_stamped_with_the_estimator_version() -> None:
    """Without the stamp, calibration never sees another sample as long as it runs.

    A calibration factor is a correction to one specific formula, so
    ``throughput.calibrate`` only reads rows whose ``estimator_version`` matches
    the estimator that wrote them (D22). A NULL-versioned row is invisible
    forever -- which is exactly what every new row would be if this line were
    dropped in a refactor.
    """
    from studioforge.core.manager import ModelManager

    plan = LoadPlan(
        model_id="pub/dense-8b",
        devices=[0],
        ctx_size=32768,
        parallel=2,
        max_parallel=2,
        ctx_per_slot=32768,
        kv_cache_type="f16",
        per_gpu_bytes={0: 12 * GB},
    )
    instance = InstanceInfo(model_id="pub/dense-8b", state="ready", port=18100, plan=plan)
    db = RecordingDb()
    config = make_config()
    manager = ModelManager(
        config,
        registry=FakeRegistry(library()),
        planner=Planner(config, rig_5090x2_3090x2()),
        supervisor=MetricsSupervisor(
            [instance], _metrics_text(gen_tokens=6000, gen_seconds=100.0, decodes=6000)
        ),
        db=db,
    )
    # Prime the baseline, then sample a window long enough to be recorded.
    manager._throughput_baseline["pub/dense-8b"] = (
        0.0,
        throughput.parse_metrics(_metrics_text(gen_tokens=0, gen_seconds=0.0, decodes=0)),
    )
    await manager._sample_one(instance, manager.THROUGHPUT_RECORD_MIN_S + 1.0)

    assert len(db.observations) == 1
    row = db.observations[0]
    assert row["estimator_version"] == throughput.ESTIMATOR_VERSION
    assert row["gen_tps"] == pytest.approx(60.0)
    assert row["est_gen_tps"] > 0
    # The row is self-contained: calibration must never have to reconstruct a
    # placement that has since been unloaded.
    assert row["devices"] == "0"
    assert row["ctx_size"] == 32768


async def test_the_prediction_is_taken_at_the_same_fill_the_catalog_quotes() -> None:
    """The learned factor is "real traffic / what the catalog promised", or it is nothing.

    ``est_gen_tps`` in the catalog is quoted at ``REFERENCE_FILL_TOKENS``. If
    the sweeper predicted at some other fill, every learned factor would carry a
    systematic offset between two different questions, and no amount of data
    would remove it.
    """
    from studioforge.core import catalog as catalog_mod
    from studioforge.core.manager import ModelManager

    config = make_config()
    planner = Planner(config, rig_5090x2_3090x2())
    dense = next(r for r in library() if r.id == "pub/dense-8b")
    plan = planner.plan_load(dense, ctx_size=131072, loaded=())
    assert isinstance(plan, LoadPlan)

    instance = InstanceInfo(model_id=dense.id, state="ready", port=18100, plan=plan)
    db = RecordingDb()
    manager = ModelManager(
        config,
        registry=FakeRegistry(library()),
        planner=planner,
        supervisor=MetricsSupervisor(
            [instance], _metrics_text(gen_tokens=6000, gen_seconds=100.0, decodes=6000)
        ),
        db=db,
    )
    manager._throughput_baseline[dense.id] = (
        0.0,
        throughput.parse_metrics(_metrics_text(gen_tokens=0, gen_seconds=0.0, decodes=0)),
    )
    await manager._sample_one(instance, manager.THROUGHPUT_RECORD_MIN_S + 1.0)

    quoted = catalog_mod.estimate_speed(planner, dense, plan, plan.parallel, {})
    assert db.observations[0]["est_gen_tps"] == pytest.approx(quoted["gen_tps"])
    # ...and deliberately NOT the pessimistic full-window number, which is a
    # different question and is reported as its own column.
    assert quoted["gen_tps_full_ctx"] < quoted["gen_tps"]
