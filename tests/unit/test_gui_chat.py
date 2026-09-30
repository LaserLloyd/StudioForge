"""The Chat tab as an ops bench (D68): target resolution, card facts, metrics.

Everything here is pure (``gui.state`` plus the chat tab's stream parser), so it
runs without NiceGUI, a GPU or a model.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
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


def test_slots_over_one_pool_read_as_one_shared_window() -> None:
    """D72: each slot of a pool may use the whole window, so "x 2 slots" would
    read as two of them. Partitioned slots keep the old wording."""
    pool = LoadPlan(
        model_id="m",
        devices=[0, 1],
        ctx_size=131072,
        ctx_per_slot=131072,
        parallel=2,
        kv_unified=True,
    )
    launched = EffectiveLaunch(parallel=2, ctx_per_slot=131072, ctx_total=131072, kv_unified=True)
    facts = dict(st.chat_target_facts(rec("m"), inst("m", plan=pool, effective=launched), []))
    assert facts["Context"] == "131,072 shared by 2 slots"
    # Before the argv is read, the plan says the same thing.
    facts = dict(st.chat_target_facts(rec("m"), inst("m", plan=pool), []))
    assert facts["Context"] == "131,072 shared by 2 slots"
    split = pool.model_copy(update={"kv_unified": False, "ctx_size": 65536, "ctx_per_slot": 65536})
    facts = dict(st.chat_target_facts(rec("m"), inst("m", plan=split), []))
    assert facts["Context"] == "65,536 × 2 slots"


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


def test_an_explicit_pick_the_engine_cannot_load_does_not_promise_a_load() -> None:
    records = [rec("ok"), rec("k2")]
    pick = st.chat_pick(
        records,
        [],
        choice="k2",
        unloadable=lambda r: "unsupported architecture" if r.id == "k2" else None,
    )
    assert pick.model_id == "k2"
    assert pick.reason == "picked · this engine cannot load it"


# ---------------------------------------------------------------------------
# The conversation window (message blocks, thinking fold, streaming, assets)
# ---------------------------------------------------------------------------


def test_the_chat_tab_has_a_conversation_window_with_its_actions(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from studioforge.config import Config
    from studioforge.gui.app import create_gui_app
    from tests.unit.test_gui import _FakeRegistry, _FakeState

    config = Config(data_dir=tmp_path / "data")
    config.models.dir = tmp_path / "models"
    config.models.dir.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()
    state = _FakeState(config)
    state.registry = _FakeRegistry([rec("m")])
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        text = client.get("/?tab=chat").text
    for needle in ("Conversation", "Clear all", "Copy all", "sfc-window", "window.sfChat"):
        assert needle in text


def test_thinking_fold_opens_while_thinking_and_collapses_on_the_answer() -> None:
    fold = chat.ThinkingFold()
    header, want = fold.step(10, answering=False, final=False, now=100.0)
    assert want is True
    assert header == "Thinking… (10 chars, 0.0s)"
    header, want = fold.step(2_500, answering=False, final=False, now=101.5)
    assert (header, want) == ("Thinking… (2,500 chars, 1.5s)", True)
    header, want = fold.step(2_600, answering=True, final=False, now=102.0)
    assert (header, want) == ("Thought for 2.0s (2,600 chars)", False)
    # Later paints keep the end time of thinking, not the end of the answer.
    header, _ = fold.step(2_600, answering=True, final=True, now=130.0)
    assert header == "Thought for 2.0s (2,600 chars)"


def test_thinking_fold_closes_when_the_stream_ends_mid_thought() -> None:
    fold = chat.ThinkingFold()
    fold.step(5, answering=False, final=False, now=0.0)
    header, want = fold.step(9, answering=False, final=True, now=12.4)
    assert (header, want) == ("Thought for 12s (9 chars)", False)


def test_thinking_fold_leaves_a_hand_toggled_fold_alone() -> None:
    fold = chat.ThinkingFold()
    fold.step(5, answering=False, final=False, now=0.0)
    fold.manual = True
    assert fold.step(50, answering=False, final=False, now=1.0)[1] is None
    header, want = fold.step(60, answering=True, final=False, now=2.0)
    assert want is None
    assert header.startswith("Thought for")


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(-1, "0.0s"), (0.44, "0.4s"), (9.94, "9.9s"), (42.4, "42s"), (125, "2m 05s")],
)
def test_seconds(seconds: float, text: str) -> None:
    assert chat._seconds(seconds) == text


class _Supervisor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def mark_request_start(self, model_id: str, *, client: str) -> str:
        self.calls.append(("start", client))
        return "req-1"

    def mark_request_end(self, model_id: str, **kwargs: Any) -> None:
        self.calls.append(("end", kwargs))


def _sse(*chunks: dict[str, Any]) -> bytes:
    lines = [f"data: {json.dumps(chunk)}\n\n" for chunk in chunks]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


def _delta(**delta: str) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta}]}


def _patch_transport(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    import httpx

    real = httpx.AsyncClient

    def client(*args: Any, **kwargs: Any) -> Any:
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(chat.httpx, "AsyncClient", client)


async def test_stream_paints_reasoning_and_answer_then_a_final_paint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        body = _sse(
            _delta(reasoning_content="let me "),
            _delta(reasoning_content="think"),
            _delta(content="**Hi**"),
            _delta(content=" there"),
            {
                "choices": [],
                "usage": {"completion_tokens": 4},
                "timings": {"predicted_per_second": 50.0},
            },
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    _patch_transport(monkeypatch, handler)
    supervisor = _Supervisor()
    paints: list[tuple[str, str, bool]] = []
    firsts: list[bool] = []
    result = await chat._stream(
        SimpleNamespace(supervisor=supervisor),  # type: ignore[arg-type]
        "m",
        "http://127.0.0.1:1",
        {"model": "m", "messages": []},
        lambda c, r, f: paints.append((c, r, f)),
        {"stop": False},
        on_first=lambda: firsts.append(True),
    )
    assert seen["url"] == "http://127.0.0.1:1/v1/chat/completions"
    assert result.content == "**Hi** there"
    assert result.reasoning == "let me think"
    assert result.chunks == 4
    assert result.usage == {"completion_tokens": 4}
    assert firsts == [True]
    assert paints[-1] == ("**Hi** there", "let me think", True)
    assert [p for p in paints if p[2]] == [paints[-1]]  # exactly one final paint
    assert supervisor.calls[0] == ("start", "gui:chat")
    assert supervisor.calls[-1][1]["tokens_per_second"] == 50.0


async def test_stream_stop_keeps_the_partial_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse(_delta(content="partial"), _delta(content=" more")))

    _patch_transport(monkeypatch, handler)
    run = {"stop": False}
    paints: list[tuple[str, str, bool]] = []

    def paint(content: str, reasoning: str, final: bool) -> None:
        paints.append((content, reasoning, final))
        run["stop"] = True  # the Stop button, pressed after the first token

    result = await chat._stream(
        SimpleNamespace(supervisor=_Supervisor()),  # type: ignore[arg-type]
        "m",
        "http://127.0.0.1:1",
        {},
        paint,
        run,
    )
    assert result.stopped is True
    assert result.content == "partial"
    assert paints[-1] == ("partial", "", True)


async def test_stream_http_error_still_paints_and_ends_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    _patch_transport(monkeypatch, lambda request: httpx.Response(500, text="boom"))
    supervisor = _Supervisor()
    paints: list[tuple[str, str, bool]] = []
    with pytest.raises(RuntimeError, match="HTTP 500: boom"):
        await chat._stream(
            SimpleNamespace(supervisor=supervisor),  # type: ignore[arg-type]
            "m",
            "http://127.0.0.1:1",
            {},
            lambda c, r, f: paints.append((c, r, f)),
            {"stop": False},
        )
    assert paints == [("", "", True)]
    assert supervisor.calls[-1][0] == "end"


def test_model_output_is_never_rendered_with_ui_markdown() -> None:
    """``ui.markdown`` passes raw HTML through; replies go via render_markdown."""
    source = inspect.getsource(chat)
    assert "ui.markdown(" not in source
    assert "render_markdown(" in source


def test_copy_helpers_fall_back_off_secure_contexts() -> None:
    from studioforge.gui import chat_assets

    js = chat_assets.CHAT_JS
    assert "isSecureContext" in js
    assert "ClipboardItem" in js
    assert "execCommand('copy')" in js
    assert "'text/html'" in js


def test_chat_css_uses_theme_tokens_not_colour_literals() -> None:
    from studioforge.gui import chat_assets

    css = chat_assets.CHAT_CSS
    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not re.search(r"\b(rgb|rgba|hsl|hsla)\(", css)
    assert "var(--accent)" in css


# ---------------------------------------------------------------------------
# Round 2: request settings, engine errors, prefill progress, Stop in prefill
# ---------------------------------------------------------------------------


def test_the_tab_builds_samplers_from_state_and_asks_for_progress() -> None:
    source = inspect.getsource(chat)
    assert "st.CHAT_SAMPLER_FIELDS" in source
    assert '"return_progress": True' in source
    assert "apply_to_payload(payload, chat=True)" in source
    # The hard-coded defaults are gone: blank means the model's recommendation.
    assert "0.7)" not in source
    assert "0.95)" not in source


def test_the_page_shows_every_sampler_field_and_the_compact_layout(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from studioforge.config import Config
    from studioforge.gui.app import create_gui_app
    from tests.unit.test_gui import _FakeRegistry, _FakeState

    config = Config(data_dir=tmp_path / "data")
    config.models.dir = tmp_path / "models"
    config.models.dir.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()
    state = _FakeState(config)
    state.registry = _FakeRegistry([rec("m")])
    app = create_gui_app(config, api_state=state)
    with TestClient(app) as client:
        text = client.get("/?tab=chat").text
    for spec in st.CHAT_SAMPLER_FIELDS:
        assert spec.label in text
    for needle in ("sfc-root", "Details", "sfc-file", "Quick tests"):
        assert needle in text


def _stream_ctx(supervisor: _Supervisor) -> Any:
    return SimpleNamespace(supervisor=supervisor)


async def test_stream_error_frame_after_the_200_fails_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    frame = {"error": {"code": 500, "message": "slot crashed", "type": "server_error"}}
    _patch_transport(
        monkeypatch, lambda request: httpx.Response(200, content=_sse(_delta(content="Hi"), frame))
    )
    supervisor = _Supervisor()
    paints: list[tuple[str, str, bool]] = []
    with pytest.raises(chat.ChatRequestError, match="slot crashed") as caught:
        await chat._stream(
            _stream_ctx(supervisor),
            "m",
            "http://127.0.0.1:1",
            {},
            lambda c, r, f: paints.append((c, r, f)),
            {"stop": False},
        )
    assert caught.value.context_full is False
    assert paints[-1] == ("Hi", "", True)  # the partial text is still painted
    assert supervisor.calls[-1][0] == "end"


async def test_stream_context_overflow_is_a_plain_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    body = {
        "error": {
            "code": 400,
            "type": "exceed_context_size_error",
            "message": "request (9120 tokens) exceeds the available context size (8192 tokens)",
            "n_prompt_tokens": 9120,
            "n_ctx": 8192,
        }
    }
    _patch_transport(monkeypatch, lambda request: httpx.Response(400, json=body))
    with pytest.raises(chat.ChatRequestError) as caught:
        await chat._stream(
            _stream_ctx(_Supervisor()),
            "pub/model",
            "http://127.0.0.1:1",
            {},
            lambda c, r, f: None,
            {"stop": False},
            n_ctx=8192,
        )
    assert caught.value.context_full is True
    message = str(caught.value)
    assert "9,120 tokens" in message
    assert "8,192" in message
    assert "Clear the chat" in message
    assert "{" not in message  # no raw JSON in the reply block


async def test_stream_reports_prefill_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    progress = {"prompt_progress": {"total": 4000, "cache": 0, "processed": 1000, "time_ms": 9}}
    _patch_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=_sse(progress, _delta(content="ok"))),
    )
    seen: list[str] = []
    result = await chat._stream(
        _stream_ctx(_Supervisor()),
        "m",
        "http://127.0.0.1:1",
        {},
        lambda c, r, f: None,
        {"stop": False},
        on_progress=seen.append,
    )
    assert seen == ["reading the prompt… 25% of 4,000 tok"]
    assert result.content == "ok"


def _hanging_transport(monkeypatch: pytest.MonkeyPatch, started: asyncio.Event) -> None:
    """A child that sends one token, then goes quiet (a long prefill or stall)."""
    import httpx

    async def body() -> Any:
        yield f"data: {json.dumps(_delta(content='so far'))}\n\n".encode()
        started.set()
        await asyncio.sleep(3600)

    _patch_transport(monkeypatch, lambda request: httpx.Response(200, content=body()))


async def test_stop_cancels_a_stream_that_has_gone_quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    _hanging_transport(monkeypatch, started)
    supervisor = _Supervisor()
    run: dict[str, Any] = {"stop": False}
    paints: list[tuple[str, str, bool]] = []
    task = asyncio.create_task(
        chat._stream(
            _stream_ctx(supervisor),
            "m",
            "http://127.0.0.1:1",
            {},
            lambda c, r, f: paints.append((c, r, f)),
            run,
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    run["stop"] = True  # what the Stop button does ...
    task.cancel()  # ... and the cancel that reaches a silent stream
    result = await asyncio.wait_for(task, 5)
    assert result.stopped is True
    assert result.content == "so far"
    assert paints[-1] == ("so far", "", True)
    assert supervisor.calls[-1][0] == "end"


async def test_a_cancel_that_is_not_a_stop_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    _hanging_transport(monkeypatch, started)
    supervisor = _Supervisor()
    task = asyncio.create_task(
        chat._stream(
            _stream_ctx(supervisor),
            "m",
            "http://127.0.0.1:1",
            {},
            lambda c, r, f: None,
            {"stop": False},
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()  # e.g. the browser tab went away
    with pytest.raises(asyncio.CancelledError):
        await task
    assert supervisor.calls[-1][0] == "end"  # the request is still accounted for


def test_chat_css_colours_the_context_readout_by_level() -> None:
    from studioforge.gui import chat_assets

    css = chat_assets.CHAT_CSS
    for level in ("ok", "warn", "full", "unknown"):
        assert f".sfc-ctx-{level}" in css
    assert ".sfc-root" in css
    assert "fit" in chat_assets.CHAT_JS


# ---------------------------------------------------------------------------
# Review fixes: request accounting, Stop in every phase, off-loop rendering
# ---------------------------------------------------------------------------


async def test_a_painter_that_raises_still_ends_the_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    _patch_transport(
        monkeypatch, lambda request: httpx.Response(200, content=_sse(_delta(content="x")))
    )
    supervisor = _Supervisor()

    def paint(content: str, reasoning: str, final: bool) -> None:
        if final:
            raise RuntimeError("element gone")

    with pytest.raises(RuntimeError, match="element gone"):
        await chat._stream(
            _stream_ctx(supervisor), "m", "http://127.0.0.1:1", {}, paint, {"stop": False}
        )
    assert [call[0] for call in supervisor.calls] == ["start", "end"]


async def test_stoppable_reports_a_stop_that_lands_before_the_task_runs() -> None:
    ran: list[bool] = []

    async def work() -> str:
        ran.append(True)
        return "done"

    run: dict[str, Any] = {"stop": False}
    waiting = asyncio.create_task(chat._stoppable(work(), run))
    await asyncio.sleep(0)  # _stoppable has created the task, which has not stepped yet
    run["stop"] = True
    run["task"].cancel()
    assert await waiting is chat._STOPPED
    assert ran == []
    assert "task" not in run


async def test_stop_during_a_load_abandons_the_wait_not_the_load() -> None:
    gate = asyncio.Event()
    finished: list[str] = []

    async def load() -> str:
        await gate.wait()
        finished.append("loaded")
        return "instance"

    shared = asyncio.ensure_future(load())  # other clients may be queued on it
    run: dict[str, Any] = {"stop": False}
    waiting = asyncio.create_task(chat._stoppable(asyncio.shield(shared), run))
    await asyncio.sleep(0.01)
    run["stop"] = True
    run["task"].cancel()
    assert await waiting is chat._STOPPED
    assert not shared.cancelled()
    gate.set()
    assert await shared == "instance"
    assert finished == ["loaded"]


async def test_stoppable_propagates_a_cancel_that_is_not_a_stop() -> None:
    run: dict[str, Any] = {"stop": False}
    waiting = asyncio.create_task(chat._stoppable(asyncio.sleep(3600), run))
    await asyncio.sleep(0.01)
    run["task"].cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    # The handler's own cancel (shutdown) propagates even after a Stop.
    run = {"stop": False}
    waiting = asyncio.create_task(chat._stoppable(asyncio.sleep(3600), run))
    await asyncio.sleep(0.01)
    run["stop"] = True
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting


async def test_the_renderer_works_off_the_loop_and_the_newest_text_wins() -> None:
    import threading
    import time as _time

    loop_thread = threading.get_ident()
    rendered: list[tuple[str, int]] = []
    applied: list[tuple[str, str]] = []

    def render(text: str) -> str:
        rendered.append((text, threading.get_ident()))
        _time.sleep(0.05)
        return f"<p>{text}</p>"

    renderer = chat._ReplyRenderer(
        lambda html, reasoning: applied.append((html, reasoning)), render
    )
    for n in range(20):
        renderer.submit(f"t{n}", "r")
        await asyncio.sleep(0.005)
    renderer.submit("final", "thought", final=True)
    await renderer.drain()
    assert applied[-1] == ("<p>final</p>", "thought")
    assert len(rendered) < 21  # paints that arrived mid-render collapsed
    assert all(thread != loop_thread for _, thread in rendered)


async def test_a_slow_render_spaces_out_the_next_one_but_not_the_final() -> None:
    import time as _time

    def slow(text: str) -> str:
        _time.sleep(0.3)
        return text

    applied: list[str] = []
    renderer = chat._ReplyRenderer(lambda html, reasoning: applied.append(html), slow)
    started = _time.perf_counter()
    renderer.submit("a", "")
    await asyncio.sleep(0.05)
    renderer.submit("b", "")  # mid-stream: waits ~2x the last render first
    await renderer.drain()
    assert applied == ["a", "b"]
    assert _time.perf_counter() - started >= 1.1

    applied.clear()
    renderer = chat._ReplyRenderer(lambda html, reasoning: applied.append(html), slow)
    started = _time.perf_counter()
    renderer.submit("a", "")
    await asyncio.sleep(0.05)
    renderer.submit("end", "", final=True)  # the end of the stream is never held back
    await renderer.drain()
    assert applied == ["a", "end"]
    assert _time.perf_counter() - started < 1.0


# ---------------------------------------------------------------------------
# The whole tab, driven through NiceGUI's user simulation
# ---------------------------------------------------------------------------
#
# ``user_simulation`` resets NiceGUI's process-wide globals (routes included),
# which would take the panel's "/" page away from every later test in this
# process -- so the simulation runs in a child interpreter and reports back.
# Nothing in it can reach a real server: the supervisor's base URL is port 1
# and every HTTP request goes to an in-memory transport.


class _SimRig:
    def __init__(self) -> None:
        self.ready = True
        self.gate = asyncio.Event()
        self.gate.set()
        self.calls: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.hang = False
        self.stream_started = asyncio.Event()
        self.stream_closed = False
        self.load_finished = False
        record = rec("m")
        instance = inst("m", started_at=1.0)
        rig = self

        class Supervisor:
            def list(self) -> list[InstanceInfo]:
                return [instance] if rig.ready else []

            def get(self, model_id: str) -> InstanceInfo | None:
                return instance if rig.ready and model_id == "m" else None

            def base_url(self, model_id: str) -> str:
                return "http://127.0.0.1:1"

            def mark_request_start(self, model_id: str, *, client: str) -> str:
                rig.calls.append("start")
                return "r"

            def mark_request_end(self, model_id: str, **kwargs: Any) -> None:
                rig.calls.append("end")

        class Manager:
            async def ensure_loaded(self, model_id: str, **kwargs: Any) -> Any:
                rig.calls.append("ensure")
                await rig.gate.wait()
                rig.ready = True
                rig.load_finished = True
                return record, instance

        self.ctx = SimpleNamespace(
            registry=SimpleNamespace(all=lambda: [record]),
            supervisor=Supervisor(),
            manager=Manager(),
            probe=None,
            refresh_interval=60.0,
        )

    async def handler(self, request: Any) -> Any:
        import httpx

        self.requests.append(json.loads(request.content))
        if not self.hang:
            return httpx.Response(200, content=_sse(_delta(content="hello")))
        rig = self

        async def body() -> Any:
            try:
                yield f"data: {json.dumps(_delta(content='so far'))}\n\n".encode()
                rig.stream_started.set()
                await asyncio.sleep(3600)
            finally:
                rig.stream_closed = True

        return httpx.Response(200, content=body())


async def _until(condition: Any, within: float = 5.0) -> bool:
    for _ in range(int(within / 0.01)):
        if condition():
            return True
        await asyncio.sleep(0.01)
    return bool(condition())


async def _chat_simulation() -> dict[str, Any]:  # noqa: C901, PLR0915 - one script, many checks
    import httpx
    from nicegui import ui
    from nicegui.testing.user_interaction import UserInteraction
    from nicegui.testing.user_simulation import user_simulation

    out: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    async def scenario(rig: _SimRig, steps: Any) -> None:
        async with user_simulation(lambda: chat.render(rig.ctx)) as user:
            await user.open("/")

            def client(*args: Any, **kwargs: Any) -> Any:
                kwargs["transport"] = httpx.MockTransport(rig.handler)
                return real_client(*args, **kwargs)

            chat.httpx.AsyncClient = client  # type: ignore[misc]
            try:
                await steps(user)
            finally:
                chat.httpx.AsyncClient = real_client  # type: ignore[misc]

    def elements(user: Any) -> list[Any]:
        return list(user.client.elements.values())

    def texts(user: Any) -> list[str]:
        return [str(getattr(e, "text", "")) for e in elements(user)]

    def button(user: Any, label: str) -> Any:
        return next(e for e in elements(user) if isinstance(e, ui.button) and e.text == label)

    def action(user: Any, tip: str) -> list[Any]:
        return [e for e in elements(user) if e.props.get("aria-label") == tip]

    def assistants(user: Any) -> int:
        return sum(1 for e in elements(user) if "sfc-assistant" in e.classes)

    def idle(user: Any) -> bool:
        return bool(button(user, "Send").enabled)

    async def send(user: Any, text: str) -> None:
        composer = next(
            e
            for e in elements(user)
            if isinstance(e, ui.textarea) and "Message" in str(e.props.get("placeholder"))
        )
        composer.set_value(text)
        UserInteraction(user, {button(user, "Send")}, None).click()
        await asyncio.sleep(0.02)

    # 1. Stop during a cold load: the wait ends, the load does not.
    rig = _SimRig()
    rig.ready = False
    rig.gate.clear()

    async def stop_during_load(user: Any) -> None:
        await send(user, "hi")
        await _until(lambda: "ensure" in rig.calls)
        UserInteraction(user, {button(user, "Stop")}, None).click()
        out["load_stop_idle"] = await _until(lambda: idle(user))
        out["load_stop_note"] = any("still loading" in t for t in texts(user))
        rig.gate.set()
        out["load_finished"] = await _until(lambda: rig.load_finished)
        out["load_stop_calls"] = list(rig.calls)

    await scenario(rig, stop_during_load)

    # 2. Regenerate, double-clicked: one new request, the same question once.
    rig = _SimRig()

    async def regenerate(user: Any) -> None:
        await send(user, "hi")
        await _until(lambda: len(rig.requests) == 1 and idle(user))
        clicks = UserInteraction(user, set(action(user, "Regenerate this reply")), None)
        clicks.trigger("click")
        clicks.trigger("click")
        await asyncio.sleep(0.05)
        await _until(lambda: idle(user))
        out["regen_requests"] = len(rig.requests)
        out["regen_messages"] = rig.requests[-1]["messages"]
        out["regen_assistants"] = assistants(user)
        out["regen_calls"] = list(rig.calls)

    await scenario(rig, regenerate)

    # 3. Delete / Clear are refused mid-stream; Stop keeps the partial reply,
    #    which is then sent back as context.
    rig = _SimRig()
    rig.hang = True

    async def stop_keeps_partial(user: Any) -> None:
        await send(user, "first")
        await asyncio.wait_for(rig.stream_started.wait(), 5)
        UserInteraction(user, set(action(user, "Delete")), None).trigger("click")
        UserInteraction(user, {button(user, "Clear all")}, None).trigger("click")
        await asyncio.sleep(0.05)
        out["busy_assistants"] = assistants(user)
        UserInteraction(user, {button(user, "Stop")}, None).click()
        out["partial_idle"] = await _until(lambda: idle(user))
        out["partial_closed"] = await _until(lambda: rig.stream_closed)
        rig.hang = False
        await send(user, "second")
        await _until(lambda: len(rig.requests) == 2 and idle(user))
        out["partial_messages"] = rig.requests[-1]["messages"]
        out["partial_calls"] = list(rig.calls)

    await scenario(rig, stop_keeps_partial)

    # 4. The viewer goes away mid-reply: the generation is stopped.
    rig = _SimRig()
    rig.hang = True

    async def viewer_leaves(user: Any) -> None:
        await send(user, "hi")
        await asyncio.wait_for(rig.stream_started.wait(), 5)
        user.client.delete()
        out["gone_closed"] = await _until(lambda: rig.stream_closed)
        out["gone_calls"] = list(rig.calls)

    await scenario(rig, viewer_leaves)
    return out


@pytest.fixture(scope="module")
def chat_simulation() -> dict[str, Any]:
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[2]
    code = (
        "import asyncio, json\n"
        "from tests.unit import test_gui_chat as t\n"
        "print('SIM:' + json.dumps(asyncio.run(t._chat_simulation())))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    line = next((x for x in done.stdout.splitlines() if x.startswith("SIM:")), None)
    assert line is not None, done.stdout[-3000:] + done.stderr[-3000:]
    result: dict[str, Any] = json.loads(line[4:])
    return result


def test_stop_during_a_cold_load_ends_the_wait_and_the_load_carries_on(
    chat_simulation: dict[str, Any],
) -> None:
    assert chat_simulation["load_stop_idle"] is True
    assert chat_simulation["load_stop_note"] is True
    assert chat_simulation["load_finished"] is True
    assert "start" not in chat_simulation["load_stop_calls"]  # nothing was sent


def test_a_double_clicked_regenerate_sends_one_request_with_one_question(
    chat_simulation: dict[str, Any],
) -> None:
    assert chat_simulation["regen_requests"] == 2
    assert chat_simulation["regen_messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    assert chat_simulation["regen_assistants"] == 1
    assert chat_simulation["regen_calls"].count("start") == 2
    assert chat_simulation["regen_calls"].count("end") == 2


def test_a_stream_refuses_delete_and_clear_and_stop_keeps_the_partial_reply(
    chat_simulation: dict[str, Any],
) -> None:
    assert chat_simulation["busy_assistants"] == 1
    assert chat_simulation["partial_idle"] is True
    assert chat_simulation["partial_closed"] is True
    roles = [(m["role"], m["content"]) for m in chat_simulation["partial_messages"][1:]]
    assert roles == [("user", "first"), ("assistant", "so far"), ("user", "second")]
    calls = chat_simulation["partial_calls"]
    assert calls.count("start") == calls.count("end") == 2


def test_a_viewer_leaving_mid_reply_stops_the_generation(
    chat_simulation: dict[str, Any],
) -> None:
    assert chat_simulation["gone_closed"] is True
    assert chat_simulation["gone_calls"][-1] == "end"
