"""The Chat tab as an ops bench (D68): target resolution, card facts, metrics.

Everything here is pure (``gui.state`` plus the chat tab's stream parser), so it
runs without NiceGUI, a GPU or a model.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.gui import state as st
from studioforge.gui.tabs import chat
from studioforge.types import (
    EffectiveLaunch,
    GgufMeta,
    GpuInfo,
    InstanceInfo,
    LoadPlan,
    ModelCapabilities,
    ModelRecord,
)

GIB = 1024**3


def rec(
    model_id: str,
    *,
    mtime: float = 0.0,
    kind: str = "chat",
    virtual: bool = False,
    base: str | None = None,
    vision: bool = False,
    thinking: bool = False,
    nextn: int = 0,
) -> ModelRecord:
    extra: dict[str, Any] = {"nextn_predict_layers": nextn} if nextn else {}
    return ModelRecord(
        id=model_id,
        name=model_id,
        kind=kind,  # type: ignore[arg-type]
        path=Path(f"/models/{model_id}.gguf"),
        size_bytes=8 * GIB,
        quant="Q4_K_M",
        architecture="qwen35",
        capabilities=ModelCapabilities(
            vision=vision, thinking=thinking, embedding=kind == "embedding"
        ),
        meta=GgufMeta(
            architecture="qwen35",
            n_ctx_train=262144,
            param_count=27_000_000_000,
            extra=extra,
        ),
        is_virtual=virtual,
        base_model_id=base,
        mtime=mtime,
        added_at=1.0,
    )


def inst(
    model_id: str,
    state: str = "ready",
    *,
    started_at: float | None = None,
    last_activity_at: float | None = None,
    **extra: Any,
) -> InstanceInfo:
    return InstanceInfo(
        model_id=model_id,
        state=state,  # type: ignore[arg-type]
        port=18100,
        started_at=started_at,
        last_activity_at=last_activity_at,
        **extra,
    )


# ---------------------------------------------------------------------------
# (Loaded model): what the next send goes to
# ---------------------------------------------------------------------------


def test_loaded_model_is_the_default_and_names_the_loaded_model() -> None:
    records = [rec("a", mtime=10), rec("b", mtime=20)]
    pick = st.chat_pick(records, [inst("a", started_at=5)], choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id == "a"
    assert pick.state == "ready"
    assert pick.follows_loaded is True
    assert pick.will_load is False
    assert "most recently loaded" in pick.reason


def test_several_loaded_follows_the_most_recently_loaded_not_the_most_used() -> None:
    """The owner asked for "most recently loaded": busy traffic must not move it."""
    records = [rec("a"), rec("b"), rec("c")]
    instances = [
        inst("a", started_at=100, last_activity_at=900),  # busiest, loaded first
        inst("b", started_at=300, last_activity_at=310),  # loaded last
        inst("c", started_at=200, last_activity_at=250),
    ]
    pick = st.chat_pick(records, instances, choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id == "b"
    assert pick.other_loaded == ("c", "a")
    assert "of 3 loaded" in pick.reason


def test_nothing_loaded_falls_back_to_the_newest_download() -> None:
    records = [rec("old", mtime=100), rec("new", mtime=300), rec("mid", mtime=200)]
    pick = st.chat_pick(records, [], choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id == "new"
    assert pick.state == "not_loaded"
    assert pick.will_load is True
    assert "newest download" in pick.reason


def test_newest_download_skips_virtual_and_unloadable_models() -> None:
    records = [
        rec("real", mtime=100),
        rec("persona", mtime=900, virtual=True, base="real"),
        rec("k2-horizon", mtime=800),
    ]

    def unloadable(record: ModelRecord) -> str | None:
        return "unsupported architecture" if record.id == "k2-horizon" else None

    pick = st.chat_pick(records, [], choice=st.LOADED_MODEL_CHOICE, unloadable=unloadable)
    assert pick.model_id == "real"


def test_a_load_in_flight_is_followed_when_nothing_is_ready() -> None:
    records = [rec("a", mtime=10), rec("b", mtime=20)]
    pick = st.chat_pick(records, [inst("a", "loading")], choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id == "a"
    assert pick.state == "loading"
    assert pick.will_load is True


def test_embedding_models_never_count_as_loaded_chat_targets() -> None:
    records = [rec("chatty", mtime=5), rec("embedder", kind="embedding", mtime=50)]
    pick = st.chat_pick(records, [inst("embedder")], choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id == "chatty"
    assert pick.state == "not_loaded"


def test_empty_library_has_no_target() -> None:
    pick = st.chat_pick([], [], choice=st.LOADED_MODEL_CHOICE)
    assert pick.model_id is None
    assert pick.state == "none"
    assert pick.will_load is False


def test_an_explicit_pick_is_honoured_and_the_loaded_model_still_reported() -> None:
    records = [rec("a"), rec("b")]
    pick = st.chat_pick(records, [inst("a", started_at=1)], choice="b")
    assert pick.model_id == "b"
    assert pick.state == "not_loaded"
    assert pick.follows_loaded is False
    assert pick.loaded_id == "a"
    assert pick.will_load is True


def test_an_explicit_pick_that_left_the_library_falls_back() -> None:
    records = [rec("a")]
    pick = st.chat_pick(records, [inst("a", started_at=1)], choice="deleted-model")
    assert pick.model_id == "a"
    assert pick.follows_loaded is True


def test_a_virtual_model_is_as_loaded_as_its_base() -> None:
    records = [rec("base"), rec("persona", virtual=True, base="base")]
    assert st.chat_model_state(records[1], [inst("base")]) == "ready"
    pick = st.chat_pick(records, [inst("base")], choice="persona")
    assert pick.state == "ready"
    assert pick.will_load is False


def test_a_failed_child_is_reported_as_failed() -> None:
    pick = st.chat_pick([rec("a")], [inst("a", "failed")], choice="a")
    assert pick.state == "failed"
    assert pick.will_load is True


# ---------------------------------------------------------------------------
# Picker options
# ---------------------------------------------------------------------------


def test_picker_puts_loaded_model_first_and_says_what_it_is() -> None:
    records = [
        rec("old", mtime=100),
        rec("new", mtime=300),
        rec("loaded", mtime=50),
        rec("warming", mtime=60),
        rec("embedder", kind="embedding", mtime=999),
    ]
    instances = [inst("loaded", started_at=10), inst("warming", "loading")]
    options = st.chat_picker_options(records, instances)
    keys = list(options)
    assert keys[0] == st.LOADED_MODEL_CHOICE
    assert options[st.LOADED_MODEL_CHOICE] == "(Loaded model) — loaded"
    assert keys[1:] == ["loaded", "warming", "new", "old"]
    assert options["loaded"].endswith(" · loaded")
    assert options["warming"].endswith(" · loading")
    assert "embedder" not in options


def test_loaded_choice_label_when_nothing_is_loaded_names_the_newest() -> None:
    options = st.chat_picker_options([rec("x", mtime=1), rec("y", mtime=2)], [])
    assert options[st.LOADED_MODEL_CHOICE] == "(Loaded model) — nothing loaded · newest: y"


def test_loaded_choice_label_variants() -> None:
    assert st.loaded_choice_label(st.ChatPick(st.LOADED_MODEL_CHOICE, None, "none", True)).endswith(
        "nothing loaded"
    )
    assert st.loaded_choice_label(
        st.ChatPick(st.LOADED_MODEL_CHOICE, "m", "loading", True)
    ).endswith("m (loading…)")


def test_picker_marks_models_the_engine_cannot_load() -> None:
    options = st.chat_picker_options(
        [rec("ok", mtime=1), rec("bad", mtime=2)],
        [],
        unloadable=lambda r: "no" if r.id == "bad" else None,
    )
    assert options["bad"].endswith(" · cannot load")
    assert options[st.LOADED_MODEL_CHOICE].endswith("newest: ok")


# ---------------------------------------------------------------------------
# Target card facts
# ---------------------------------------------------------------------------


def _gpus() -> list[GpuInfo]:
    return [
        GpuInfo(index=0, name="NVIDIA GeForce RTX 5090", total_bytes=32 * GIB, free_bytes=0),
        GpuInfo(index=1, name="NVIDIA GeForce RTX 5090", total_bytes=32 * GIB, free_bytes=0),
        GpuInfo(index=2, name="NVIDIA GeForce RTX 3090", total_bytes=24 * GIB, free_bytes=0),
    ]


def test_facts_for_a_loaded_model_say_where_and_how_it_runs() -> None:
    record = rec("m", thinking=True, nextn=1)
    instance = inst(
        "m",
        started_at=1_000.0,
        plan=LoadPlan(
            model_id="m",
            devices=[0, 1],
            ctx_size=262144,
            parallel=1,
            kv_cache_type="q8_0",
            kv_cache_type_v="f16",
        ),
        effective=EffectiveLaunch(parallel=1, ctx_per_slot=262144, ctx_total=262144),
        speculative={"type": "draft-mtp", "draft_n_max": 3},
        resolved_engine_tag="b11037",
        loaded_by="gui",
        priority=1,
        ttl_s=900,
        total_requests=4,
        last_tokens_per_second=61.2,
    )
    facts = dict(st.chat_target_facts(record, instance, _gpus(), now=1_060.0))
    assert facts["GPUs"] == "GPU 0,1 · RTX 5090 ×2"
    assert facts["Context"] == "262,144 × 1 slot"
    assert facts["KV cache"] == "q8_0 / f16"
    assert facts["Speculative"] == "MTP heads (up to 3 per step)"
    assert facts["Engine"] == "b11037"
    assert facts["Loaded"].endswith("by gui")
    assert facts["Tier"] == "1 chat"
    assert facts["Idle unload"] == "after 15m 00s"
    assert facts["Requests"] == "4"
    assert facts["Last decode"] == "61.2 tok/s"
    assert facts["Features"] == "thinking · MTP ×1"


def test_facts_for_a_model_that_is_not_loaded_describe_the_download() -> None:
    record = rec("m", vision=True, mtime=500.0)
    facts = dict(st.chat_target_facts(record, None, _gpus(), now=530.0))
    assert facts["Size"] == "8.0 GiB"
    assert facts["Quant"] == "Q4_K_M"
    assert facts["Arch"] == "qwen35"
    assert facts["Params"] == "27.0B"
    assert facts["Trained ctx"] == "262,144"
    assert facts["Downloaded"] == "just now"
    assert facts["Features"] == "vision"
    assert "GPUs" not in facts


def test_gpu_summary_mixes_generations_and_survives_unknown_cards() -> None:
    assert st.gpu_summary([1, 2], _gpus()) == "GPU 1,2 · RTX 5090 + RTX 3090"
    assert st.gpu_summary([7], _gpus()) == "GPU 7"
    assert st.gpu_summary([], _gpus()) == st.UNKNOWN


def test_speculative_off_reads_off() -> None:
    record = rec("m")
    instance = inst("m", started_at=1.0)
    facts = dict(st.chat_target_facts(record, instance, [], now=2.0))
    assert facts["Speculative"] == "off"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: The final chunk llama-server b11037 sent for a 40-token reply (measured).
_TIMINGS = {
    "cache_n": 0,
    "prompt_n": 38,
    "prompt_ms": 29.041,
    "prompt_per_second": 1308.4948865397198,
    "predicted_n": 40,
    "predicted_ms": 54.009,
    "predicted_per_second": 722.1018719102372,
}
_USAGE = {
    "completion_tokens": 40,
    "prompt_tokens": 38,
    "total_tokens": 78,
    "prompt_tokens_details": {"cached_tokens": 0},
}


def _metrics(**overrides: Any) -> st.ChatRunMetrics:
    values: dict[str, Any] = {
        "clicked_at": 100.0,
        "sent_at": 100.0,
        "first_token_at": 100.25,
        "last_token_at": 101.0,
        "load_s": None,
        "chunks": 40,
        "usage": _USAGE,
        "timings": _TIMINGS,
        "finish_reason": "length",
    }
    values.update(overrides)
    return st.chat_run_metrics(**values)


def test_engine_timings_drive_prefill_and_decode() -> None:
    m = _metrics()
    assert m.source == "engine"
    assert m.ttft_s == pytest.approx(0.25)
    assert m.prefill_tokens == 38
    assert m.prefill_tps == pytest.approx(1308.49, rel=1e-3)
    assert m.decode_tokens == 40
    assert m.decode_tps == pytest.approx(722.1, rel=1e-3)
    assert m.overall_tps == pytest.approx(40.0)  # 40 tokens over the 1.0 s request
    assert m.overall_with_load_tps is None

    tiles = {t.label: t for t in st.chat_metric_tiles(m)}
    assert list(tiles) == ["Load", "TTFT", "Prefill", "Decode", "Overall", "Total"]
    assert tiles["Load"].detail == "already loaded"
    assert tiles["TTFT"].value == "250 ms"
    assert tiles["Prefill"].value == "1,308 tok/s"
    assert tiles["Prefill"].detail == "38 tok · 29 ms"
    assert tiles["Decode"].value == "722 tok/s"
    assert tiles["Decode"].detail == "40 tok · 54 ms"
    assert tiles["Overall"].value == "40.0 tok/s"
    assert tiles["Overall"].detail == "40 tok in 1.00 s"
    footer = st.chat_metric_footer(m)
    assert "tokens: 38 in, 40 out" in footer
    assert "finish: hit max_tokens" in footer


def test_a_cold_start_reports_the_load_and_overall_with_it() -> None:
    m = _metrics(clicked_at=90.0, load_s=10.0)
    assert m.total_s == pytest.approx(11.0)
    assert m.overall_with_load_tps == pytest.approx(round(40 / 11.0, 2))
    tiles = {t.label: t for t in st.chat_metric_tiles(m)}
    assert tiles["Load"].value == "10.0 s"
    assert tiles["Load"].detail == "cold start"
    assert "incl. load" in tiles["Overall"].detail
    assert tiles["Total"].detail == "click to last token, load incl."


def test_without_engine_timings_decode_is_estimated_from_the_stream() -> None:
    m = _metrics(timings=None, usage=None, chunks=11, first_token_at=100.0, last_token_at=102.0)
    assert m.source == "client"
    assert m.decode_tps == pytest.approx(5.0)  # 10 tokens after the first, over 2 s
    assert m.completion_tokens == 11
    tiles = {t.label: t for t in st.chat_metric_tiles(m)}
    assert "estimated" in tiles["Decode"].detail
    assert "decode estimated" in st.chat_metric_footer(m)


def test_a_fully_cached_prompt_is_shown_as_cached_not_as_a_speed() -> None:
    timings = dict(_TIMINGS, prompt_n=0, cache_n=512, prompt_per_second=1e9)
    m = _metrics(timings=timings)
    assert m.prefill_tps is None
    tile = next(t for t in st.chat_metric_tiles(m) if t.label == "Prefill")
    assert tile.value == "cached"
    assert "512" in tile.detail


def test_speculative_acceptance_and_stop_reach_the_footer() -> None:
    timings = dict(_TIMINGS, draft_n=40, draft_n_accepted=31)
    footer = st.chat_metric_footer(_metrics(timings=timings, stopped=True))
    assert "speculative: 31/40 drafted accepted (78%)" in footer
    assert "stopped by you" in footer
    assert "finish:" not in footer


def test_junk_timings_never_raise() -> None:
    m = _metrics(timings={"predicted_per_second": "nan?", "prompt_n": True, "draft_n": None})
    assert m.prefill_tokens is None
    assert st.chat_metric_tiles(m)


@pytest.mark.parametrize(
    ("value", "text"),
    [(None, st.UNKNOWN), (0, st.UNKNOWN), (722.1, "722"), (24.84, "24.8"), (3.125, "3.12")],
)
def test_format_tps(value: float | None, text: str) -> None:
    assert st.format_tps(value) == text


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(None, st.UNKNOWN), (0.42, "420 ms"), (1.614, "1.61 s"), (11.52, "11.5 s"), (125, "2m 05s")],
)
def test_format_latency(seconds: float | None, text: str) -> None:
    assert st.format_latency(seconds) == text


# ---------------------------------------------------------------------------
# Thinking split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "reasoning", "answer"),
    [
        ("plain answer", "", "plain answer"),
        ("<think>why</think>because", "why", "because"),
        ("<think>still thinking", "still thinking", ""),
        ("pre-filled opener</think>answer", "pre-filled opener", "answer"),
        ("lead <think>x</think> tail", "x", "lead  tail"),
    ],
)
def test_split_reasoning(content: str, reasoning: str, answer: str) -> None:
    assert st.split_reasoning(content) == (reasoning, answer)


# ---------------------------------------------------------------------------
# Quick tests
# ---------------------------------------------------------------------------


def test_every_quick_test_has_a_prompt() -> None:
    keys = [t.key for t in st.CHAT_QUICK_TESTS]
    assert keys == ["hello", "count", "long", "prefill"]
    for key in keys:
        assert st.quick_test_prompt(key)
    with pytest.raises(KeyError):
        st.quick_test_prompt("nope")


def test_prefill_test_is_long_deterministic_and_ends_with_the_question() -> None:
    first = st.quick_test_prompt("prefill")
    assert first == st.quick_test_prompt("prefill")
    assert len(first) > 12_000  # ~4k tokens at ~3-4 characters per token
    assert first.endswith("Answer in one sentence.")
    assert "Section 2." in first


# ---------------------------------------------------------------------------
# The tab module itself
# ---------------------------------------------------------------------------


def test_parse_sse() -> None:
    assert chat._parse_sse('data: {"choices": []}') == {"choices": []}
    assert chat._parse_sse("data: [DONE]") is None
    assert chat._parse_sse(": prefilling") is None
    assert chat._parse_sse("data: {broken") is None
    assert chat._parse_sse("data: [1, 2]") is None


def test_the_tab_asks_for_usage_and_timings() -> None:
    source = inspect.getsource(chat)
    assert '"stream_options": {"include_usage": True}' in source


def test_unloadable_check_is_optional_and_never_raises() -> None:
    assert chat._unloadable_check(SimpleNamespace(manager=object())) is None  # type: ignore[arg-type]

    class Manager:
        def unsupported_reason(self, record: Any) -> str | None:
            if record == "boom":
                raise RuntimeError("probe failed")
            return "unsupported" if record == "bad" else None

    check = chat._unloadable_check(SimpleNamespace(manager=Manager()))  # type: ignore[arg-type]
    assert check is not None
    assert check("bad") == "unsupported"
    assert check("good") is None
    assert check("boom") is None


# ---------------------------------------------------------------------------
# The page itself renders (every tab paints into the landing document)
# ---------------------------------------------------------------------------


def test_the_chat_tab_renders_with_the_loaded_model_named(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from studioforge.config import Config
    from studioforge.gui.app import create_gui_app
    from tests.unit.test_gui import _FakeRegistry, _FakeState

    config = Config(data_dir=tmp_path / "data")
    config.models.dir = tmp_path / "models"
    config.models.dir.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()

    class Supervisor:
        def list(self) -> list[InstanceInfo]:
            return [inst("pub/repo/loaded-model", started_at=1.0)]

        def get(self, model_id: str) -> InstanceInfo | None:
            return next((i for i in self.list() if i.model_id == model_id), None)

    state = _FakeState(config)
    state.registry = _FakeRegistry([rec("pub/repo/loaded-model"), rec("pub/repo/other", mtime=9)])
    state.supervisor = Supervisor()
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        response = client.get("/?tab=chat")
    assert response.status_code == 200
    assert "(Loaded model) — pub/repo/loaded-model" in response.text
    assert "Quick tests" in response.text
