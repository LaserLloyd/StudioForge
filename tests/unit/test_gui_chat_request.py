"""Chat tab request settings and context-window helpers in ``gui/state.py``.

Pure functions: no NiceGUI, no server, no GPU. The engine bodies below are the
shapes llama-server b11037 produces (the overflow prose is copied from a live
child log), so a change in the engine's wording shows up here first.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from studioforge.gui import state as st
from studioforge.types import (
    EffectiveLaunch,
    GgufMeta,
    InstanceInfo,
    LoadPlan,
    ModelRecord,
    ModelSettings,
    VirtualPreset,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _record(
    model_id: str = "m",
    *,
    sampling: dict[str, float | int] | None = None,
    template: str | None = None,
    settings: ModelSettings | None = None,
    virtual_of: str | None = None,
    preset: VirtualPreset | None = None,
    meta: bool = True,
) -> ModelRecord:
    extra = {"sampling": sampling} if sampling is not None else {}
    return ModelRecord(
        id=model_id,
        name=model_id,
        path=Path(f"{model_id}.gguf"),
        meta=GgufMeta(chat_template=template, extra=extra) if meta else None,
        settings=settings or ModelSettings(),
        is_virtual=virtual_of is not None,
        base_model_id=virtual_of,
        preset=preset,
    )


def _instance(
    *,
    effective: EffectiveLaunch | None = None,
    plan: LoadPlan | None = None,
) -> InstanceInfo:
    return InstanceInfo(model_id="m", state="ready", port=1, effective=effective, plan=plan)


# ---------------------------------------------------------------------------
# sampler fields and the payload
# ---------------------------------------------------------------------------


def test_every_sampler_field_is_a_llama_server_request_key() -> None:
    keys = [f.key for f in st.CHAT_SAMPLER_FIELDS]
    assert keys == [
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repeat_penalty",
        "max_tokens",
        "seed",
        "stop",
    ]
    assert set(st.CHAT_SAMPLER_FIELD_BY_KEY) == set(keys)
    for spec in st.CHAT_SAMPLER_FIELDS:
        assert spec.kind in ("float", "int", "text")
        assert spec.tooltip and spec.engine_default
        if spec.minimum is not None and spec.maximum is not None:
            assert spec.minimum < spec.maximum


def test_only_max_tokens_starts_filled_so_the_model_recommendation_applies() -> None:
    filled = {f.key: f.default for f in st.CHAT_SAMPLER_FIELDS if f.default is not None}
    assert filled == {"max_tokens": 4096}
    defaults = {f.key: f.default for f in st.CHAT_SAMPLER_FIELDS}
    assert st.build_sampler_payload(defaults) == {"max_tokens": 4096}


def test_blank_values_are_left_out_of_the_payload() -> None:
    values = {f.key: "" for f in st.CHAT_SAMPLER_FIELDS} | {"thinking": "auto"}
    assert st.build_sampler_payload(values) == {}
    assert st.build_sampler_payload({}) == {}
    assert st.build_sampler_payload({"temperature": None, "top_k": "abc"}) == {}


def test_an_explicit_zero_is_sent_not_replaced() -> None:
    payload = st.build_sampler_payload({"temperature": 0, "top_k": 0.0, "min_p": "0"})
    assert payload == {"temperature": 0.0, "top_k": 0, "min_p": 0.0}


def test_numbers_are_clamped_and_ints_rounded() -> None:
    payload = st.build_sampler_payload(
        {
            "temperature": 9,
            "top_p": -1,
            "top_k": "20.6",
            "repeat_penalty": 0.1,
            "max_tokens": 1500.4,
            "seed": 42.0,
        }
    )
    assert payload == {
        "temperature": 2.0,
        "top_p": 0.0,
        "top_k": 21,
        "repeat_penalty": 0.5,
        "max_tokens": 1500,
        "seed": 42,
    }
    assert isinstance(payload["top_k"], int) and isinstance(payload["seed"], int)


@pytest.mark.parametrize("raw", [0, -5, "0"])
def test_max_tokens_zero_or_less_means_no_cap(raw: object) -> None:
    assert "max_tokens" not in st.build_sampler_payload({"max_tokens": raw})


@pytest.mark.parametrize("raw", [-1, "-1", None, ""])
def test_a_random_seed_is_not_sent(raw: object) -> None:
    assert "seed" not in st.build_sampler_payload({"seed": raw})


def test_nan_infinity_and_bools_are_never_sent() -> None:
    values = {"temperature": float("nan"), "top_p": float("inf"), "top_k": True}
    assert st.build_sampler_payload(values) == {}


def test_stop_sequences_one_per_line_with_escapes() -> None:
    raw = "</s>\r\n User:\n\n   \n\\n\\n\n</s>\n"
    assert st.parse_stop_sequences(raw) == ["</s>", " User:", "\n\n"]
    assert st.build_sampler_payload({"stop": raw})["stop"] == ["</s>", " User:", "\n\n"]
    assert st.parse_stop_sequences(["a", "", "a", "b"]) == ["a", "b"]
    assert st.parse_stop_sequences(None) == []
    assert len(st.parse_stop_sequences("\n".join(f"s{i}" for i in range(40)))) == 16
    assert "stop" not in st.build_sampler_payload({"stop": "  \n"})


@pytest.mark.parametrize(("choice", "want"), [("on", True), ("off", False)])
def test_thinking_switch_rides_on_chat_template_kwargs(choice: str, want: bool) -> None:
    payload = st.build_sampler_payload({"thinking": choice})
    assert payload == {"chat_template_kwargs": {"enable_thinking": want}}
    json.dumps(payload)  # the body must serialise as is


@pytest.mark.parametrize("choice", ["auto", None, "", "maybe"])
def test_thinking_default_sends_nothing(choice: object) -> None:
    assert st.build_sampler_payload({"thinking": choice}) == {}


def test_thinking_choices_cover_the_values_the_payload_understands() -> None:
    assert list(st.CHAT_THINKING_CHOICES) == ["auto", "on", "off"]
    assert st.CHAT_THINKING_TOOLTIP


def test_thinking_toggle_only_when_the_template_has_the_switch() -> None:
    qwen = "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>"
    assert st.thinking_toggle_supported(_record(template=qwen))
    assert not st.thinking_toggle_supported(_record(template="{{ messages }}"))
    assert not st.thinking_toggle_supported(_record(template=None))
    assert not st.thinking_toggle_supported(_record(meta=False))
    assert not st.thinking_toggle_supported(None)


# ---------------------------------------------------------------------------
# recommended sampling
# ---------------------------------------------------------------------------

#: What a real Qwen3.8 GGUF on the rig carries (general.sampling.*).
_QWEN_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def test_recommended_sampling_reads_the_gguf() -> None:
    assert st.recommended_sampling(_record(sampling=_QWEN_SAMPLING)) == _QWEN_SAMPLING
    assert st.recommended_sampling(_record()) == {}
    assert st.recommended_sampling(_record(meta=False)) == {}
    assert st.recommended_sampling(None) == {}


def test_a_launch_flag_beats_the_gguf() -> None:
    record = _record(
        sampling=_QWEN_SAMPLING, settings=ModelSettings(temperature=0.6, repeat_penalty=1.1)
    )
    assert st.recommended_sampling(record) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "repeat_penalty": 1.1,
    }


def test_a_preset_persona_layers_over_its_base() -> None:
    base = _record("base", sampling=_QWEN_SAMPLING, settings=ModelSettings(min_p=0.02))
    persona = _record(
        "persona",
        virtual_of="base",
        meta=False,
        preset=VirtualPreset(system_prompt="Be terse.", top_k=40, max_tokens=512),
    )
    assert st.recommended_sampling(persona, base) == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.02,
        "max_tokens": 512,
    }


def test_a_persona_with_its_own_instance_does_not_inherit_base_launch_flags() -> None:
    base = _record("base", settings=ModelSettings(temperature=0.3))
    persona = _record("persona", virtual_of="base", settings=ModelSettings(ctx_size=4096))
    assert st.recommended_sampling(persona, base) == {}


def test_placeholder_names_where_a_blank_value_comes_from() -> None:
    fields = st.CHAT_SAMPLER_FIELD_BY_KEY
    recommended = {"temperature": 1.0, "top_k": 20}
    assert st.sampler_placeholder(fields["temperature"], recommended) == "model: 1"
    assert st.sampler_placeholder(fields["top_k"], recommended) == "model: 20"
    assert st.sampler_placeholder(fields["min_p"], recommended) == "engine: 0.05"
    assert st.sampler_placeholder(fields["seed"], {}) == "engine: random"


# ---------------------------------------------------------------------------
# the context window
# ---------------------------------------------------------------------------


def test_context_limit_is_the_slot_not_the_whole_pool() -> None:
    effective = EffectiveLaunch(parallel=4, ctx_per_slot=32768, ctx_total=131072)
    assert st.chat_context_limit(_instance(effective=effective)) == 32768


def test_context_limit_with_a_unified_pool_is_the_whole_pool() -> None:
    effective = EffectiveLaunch(parallel=4, ctx_per_slot=32768, ctx_total=131072, kv_unified=True)
    assert st.chat_context_limit(_instance(effective=effective)) == 131072


def test_context_limit_falls_back_to_the_plan_before_the_argv_is_parsed() -> None:
    plan = LoadPlan(model_id="m", devices=[0], ctx_size=65536, ctx_per_slot=65536)
    assert st.chat_context_limit(_instance(plan=plan)) == 65536
    assert st.chat_context_limit(_instance(plan=LoadPlan(model_id="m", devices=[0]))) == 8192
    assert st.chat_context_limit(_instance()) is None
    assert st.chat_context_limit(None) is None


def _metrics(**over: object) -> st.ChatRunMetrics:
    return st.ChatRunMetrics(**over)  # type: ignore[arg-type]


def test_usage_counts_the_whole_prompt_and_the_reply() -> None:
    usage = st.chat_context_usage(_metrics(prompt_tokens=10_000, completion_tokens=500), 200_000)
    assert (usage.used, usage.limit, usage.level) == (10_500, 200_000, "ok")
    assert usage.text == "context 10,500 / 200,000 (5%)"
    assert usage.fraction == pytest.approx(0.0525)
    assert usage.tooltip


def test_usage_falls_back_to_the_engine_timings() -> None:
    metrics = _metrics(prefill_tokens=100, cached_tokens=4_000, decode_tokens=250)
    assert st.chat_context_usage(metrics, 8192).used == 4_350


def test_usage_warns_near_the_limit_and_when_the_next_reply_will_not_fit() -> None:
    assert st.chat_context_usage(_metrics(prompt_tokens=7_000), 8192).level == "warn"
    full = st.chat_context_usage(_metrics(prompt_tokens=7_900), 8192)
    assert full.level == "full" and "nearly full" in full.text
    tight = st.chat_context_usage(_metrics(prompt_tokens=5_000), 8192, max_tokens=4096)
    assert tight.level == "warn"
    assert "under 4,096 tok left for a reply" in tight.text
    roomy = st.chat_context_usage(_metrics(prompt_tokens=5_000), 200_000, max_tokens=4096)
    assert roomy.level == "ok"


def test_usage_is_unknown_after_a_stop_or_without_a_window() -> None:
    stopped = st.chat_context_usage(_metrics(stopped=True), 8192)
    assert (stopped.used, stopped.level) == (None, "unknown")
    assert stopped.text == "context — / 8,192"
    assert st.chat_context_usage(None, None).text == "context —"
    no_window = st.chat_context_usage(_metrics(prompt_tokens=10), None)
    assert (no_window.level, no_window.text, no_window.fraction) == (
        "unknown",
        "context 10 tok",
        None,
    )


# ---------------------------------------------------------------------------
# the overflow refusal and other errors
# ---------------------------------------------------------------------------

#: b11037's typed refusal, as the gateway's own tests and D53 describe it.
_TYPED = json.dumps(
    {
        "error": {
            "code": 400,
            "message": "request (8240 tokens) exceeds the available context size "
            "(8192 tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 8240,
            "n_ctx": 8192,
        }
    }
)


def test_the_typed_refusal_becomes_plain_advice() -> None:
    overflow = st.context_overflow(400, _TYPED, model_id="qwen")
    assert overflow is not None
    assert (overflow.prompt_tokens, overflow.n_ctx) == (8240, 8192)
    assert overflow.message.startswith(
        "This conversation is 8,240 tokens, and qwen is loaded with room for 8,192 "
        "per conversation."
    )
    assert "Clear the chat, delete older messages" in overflow.message
    assert "larger context" in overflow.message


def test_an_untyped_refusal_is_read_from_the_engine_prose() -> None:
    body = {
        "error": {
            "message": "request (9000 tokens) exceeds the available context size "
            "(8192 tokens), try increasing it"
        }
    }
    overflow = st.context_overflow(400, body)
    assert overflow is not None
    assert (overflow.prompt_tokens, overflow.n_ctx) == (9000, 8192)
    assert overflow.message.startswith("This conversation is 9,000 tokens, and the model")


def test_the_callers_window_fills_in_when_the_engine_does_not_say() -> None:
    body = b'{"error": {"message": "the prompt is larger than the max context size"}}'
    overflow = st.context_overflow(400, body, n_ctx=4096)
    assert overflow is not None
    assert (overflow.prompt_tokens, overflow.n_ctx) == (None, 4096)
    assert overflow.message.startswith("This conversation no longer fits the 4,096 tokens")
    bare = st.context_overflow(400, "context size exceeded")
    assert bare is not None and bare.n_ctx is None
    assert bare.message.startswith("This conversation no longer fits the context")


def test_an_in_stream_error_frame_is_recognised() -> None:
    frame = json.loads(_TYPED)
    assert st.stream_error(frame)
    overflow = st.context_overflow(None, frame)
    assert overflow is not None and overflow.prompt_tokens == 8240


def test_other_errors_are_not_mistaken_for_an_overflow() -> None:
    sampler = {"error": {"message": "top_k must be a positive integer", "type": "invalid_request"}}
    assert st.context_overflow(400, sampler) is None
    # A 500 that mentions the context is the engine failing, not a long prompt.
    crashed = {"error": {"message": "context size has been exceeded", "type": "server_error"}}
    assert st.context_overflow(500, crashed) is None
    assert st.context_overflow(400, "") is None
    assert st.context_overflow(400, None) is None


def test_error_text_uses_the_engines_message_not_raw_json() -> None:
    body = json.dumps({"error": {"code": 400, "message": "Unknown   role\n'tool'", "type": "x"}})
    assert st.chat_error_text(400, body) == "HTTP 400: Unknown role 'tool'"
    assert st.chat_error_text(503, "Loading model") == "HTTP 503: Loading model"
    assert st.chat_error_text(None, {"error": "boom"}) == "boom"
    assert st.chat_error_text(502, "") == "HTTP 502: no details"
    assert st.chat_error_text(500, "x" * 2000) == "HTTP 500: " + "x" * 600
    assert st.chat_error_text(400, _TYPED, model_id="m").startswith("This conversation is 8,240")


def test_stream_error_ignores_ordinary_frames() -> None:
    assert not st.stream_error({"choices": [{"delta": {"content": "hi"}}]})
    assert not st.stream_error({"error": None})
    assert not st.stream_error(None)


# ---------------------------------------------------------------------------
# prefill progress
# ---------------------------------------------------------------------------


def test_prefill_progress_frame() -> None:
    frame = {
        "choices": [],
        "prompt_progress": {"total": 20_000, "cache": 4_000, "processed": 5_000, "time_ms": 900},
    }
    progress = st.prefill_progress(frame)
    assert progress == st.PrefillProgress(total=20_000, processed=5_000, cached=4_000)
    assert progress is not None
    assert progress.fraction == pytest.approx(0.25)
    assert progress.text == "reading the prompt… 25% of 20,000 tok"


@pytest.mark.parametrize(
    "frame",
    [
        None,
        {"choices": [{"delta": {"content": "x"}}]},
        {"prompt_progress": "half"},
        {"prompt_progress": {"total": 0, "processed": 0}},
        {"prompt_progress": {"total": 10, "processed": True}},
        {"prompt_progress": {"total": 10}},
    ],
)
def test_prefill_progress_ignores_anything_else(frame: object) -> None:
    assert st.prefill_progress(frame) is None  # type: ignore[arg-type]


def test_prefill_progress_never_passes_one_hundred_percent() -> None:
    progress = st.prefill_progress({"prompt_progress": {"total": 10, "processed": 12}})
    assert progress is not None and progress.fraction == 1.0
