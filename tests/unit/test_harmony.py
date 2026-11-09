"""D82: harmony (gpt-oss) markup never comes back as history; array tool params get ``items``.

The shapes below are the ones llama.cpp b11718 actually produced on 2026-11-07
(a 7-turn role-play session against unsloth's gpt-oss-20b Q8_0):

* ``--reasoning-format none`` content: the analysis block WITH its tags, then the
  answer with its final-channel header dropped;
* what a model that has seen that history writes on turn 4: plain text straight
  after ``<|start|>assistant``, which the engine's gpt-oss PEG parser rejects.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from studioforge.api.openai_routes import _repair_for_template
from studioforge.config import Config
from studioforge.core.harmony import (
    effective_reasoning_format,
    guard_tool_schemas,
    record_uses_harmony,
    repair_history,
    split_harmony,
    uses_harmony,
)
from studioforge.types import GgufMeta, ModelRecord, ModelSettings
from tests.unit.test_supervisor_features import (
    B10425,
    make_binary,
    make_plan,
    make_record,
    sup,
    value_after,
)

#: The opening of unsloth's gpt-oss template, enough for detection.
HARMONY_TEMPLATE = (
    "{%- for message in messages -%}<|start|>{{ message.role }}<|channel|>final<|message|>"
    "{{ message.content }}<|end|>{%- endfor -%}<|start|>assistant"
)
CHATML_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
)

#: Exactly what b11718 returned as message.content under --reasoning-format none.
NONE_FORMAT_CONTENT = (
    "<|channel|>analysis<|message|>We need to respond as Mina, third person.<|end|>"
    "Mina looked up from the stove and smiled."
)


def harmony_meta(**kwargs: Any) -> GgufMeta:
    params: dict[str, Any] = {"architecture": "gpt-oss", "chat_template": HARMONY_TEMPLATE}
    params.update(kwargs)
    return GgufMeta(**params)


def record(meta: GgufMeta | None, settings: ModelSettings | None = None) -> ModelRecord:
    return ModelRecord(
        id="unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0",
        name="gpt-oss-20b",
        path=Path("gpt-oss-20b-Q8_0.gguf"),
        meta=meta,
        settings=settings or ModelSettings(),
    )


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------


def test_harmony_is_detected_from_the_template_or_the_architecture() -> None:
    assert uses_harmony(HARMONY_TEMPLATE)
    assert uses_harmony(None, "gpt-oss")
    assert uses_harmony(CHATML_TEMPLATE, "GPT-OSS")
    assert not uses_harmony(CHATML_TEMPLATE, "qwen3")
    assert not uses_harmony(None, None)
    # <|channel|> alone is not enough: a template must also open messages with <|start|>.
    assert not uses_harmony("<|channel|>x<|message|>{{ messages }}")


def test_record_detection_reads_the_embedded_template_not_an_override() -> None:
    assert record_uses_harmony(record(harmony_meta()))
    assert record_uses_harmony(record(GgufMeta(architecture="gpt-oss")))
    assert not record_uses_harmony(
        record(GgufMeta(architecture="qwen3", chat_template=CHATML_TEMPLATE))
    )
    assert not record_uses_harmony(record(None))
    overridden = record(harmony_meta(), ModelSettings(chat_template_file=None))
    assert record_uses_harmony(overridden)


@pytest.mark.parametrize(
    ("setting", "default", "harmony", "expected"),
    [
        (None, "none", True, ("deepseek", "harmony")),
        (None, "none", False, ("none", "default")),
        ("none", "none", True, ("none", "setting")),
        ("deepseek-legacy", "none", True, ("deepseek-legacy", "setting")),
        (None, "deepseek", True, ("deepseek", "default")),
        (None, "deepseek", False, ("deepseek", "default")),
    ],
)
def test_effective_reasoning_format(
    setting: str | None, default: str, harmony: bool, expected: tuple[str, str]
) -> None:
    assert effective_reasoning_format(setting, default, harmony=harmony) == expected


# ---------------------------------------------------------------------------
# split_harmony
# ---------------------------------------------------------------------------


def test_text_without_markup_is_left_alone() -> None:
    assert split_harmony("Mina smiled. <b>not markup</b> <|im_end|>") is None


def test_the_none_format_shape_splits_into_answer_and_analysis() -> None:
    assert split_harmony(NONE_FORMAT_CONTENT) == (
        "Mina looked up from the stove and smiled.",
        "We need to respond as Mina, third person.",
    )


def test_a_well_formed_harmony_reply() -> None:
    text = (
        "<|channel|>analysis<|message|>Think.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>The answer.<|return|>"
    )
    assert split_harmony(text) == ("The answer.", "Think.")


def test_the_turn_four_shape_a_headerless_reply_then_a_proper_one() -> None:
    """What the model wrote once the history was broken: text right after the role."""
    text = (
        "Mina turned to the fire.<|end|><|start|>assistant<|channel|>commentary<|message|>"
        "We have complied.<|end|><|start|>assistant<|channel|>final<|message|>Mina nodded."
    )
    answer, analysis = split_harmony(text) or ("", "")
    assert answer == "Mina turned to the fire.\n\nWe have complied.\n\nMina nodded."
    assert analysis == ""


def test_a_tool_call_body_is_dropped_and_a_preamble_kept() -> None:
    text = (
        "<|channel|>commentary<|message|>Let me check.<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json"
        '<|message|>{"city":"Oslo"}<|call|>'
    )
    assert split_harmony(text) == ("Let me check.", "")


def test_a_recipient_in_the_role_header_is_a_tool_call_too() -> None:
    text = (
        '<|start|>assistant to=functions.lookup<|channel|>commentary json<|message|>{"q":1}<|call|>'
    )
    assert split_harmony(text) == ("", "")


def test_an_analysis_cut_off_by_max_tokens() -> None:
    assert split_harmony("<|channel|>analysis<|message|>We need to") == ("", "We need to")


def test_a_constrained_final_channel() -> None:
    text = "<|channel|>final <|constrain|>finish<|message|>Implemented the fixes."
    assert split_harmony(text) == ("Implemented the fixes.", "")


def test_a_reply_glued_to_the_role_with_no_message_token_is_kept() -> None:
    """The turn-4 shape at its barest: ``<|start|>assistant`` then the reply, nothing else."""
    assert split_harmony("Mina turned.<|end|><|start|>assistantMina nodded.") == (
        "Mina turned.\n\nMina nodded.",
        "",
    )
    assert split_harmony("<|start|>assistant Mina nodded.<|end|>") == ("Mina nodded.", "")
    # ...but a role that names a recipient is still a tool call, not a reply.
    assert split_harmony('<|start|>assistant to=functions.f {"a":1}') == ("", "")


@pytest.mark.parametrize(
    "text",
    [
        "Harmony uses <|start|>, <|channel|>, <|message|> and <|end|> tokens.",
        "Example:\n```\n<|start|>assistant<|channel|>final<|message|>Hi<|end|>\n```\nThat's it.",
        "Each turn ends with <|end|> and the last one with <|return|>.",
        "Use the <|end|> token to stop.",
        "The header is <|channel|>, then a name.",
        "A body starts after <|message|>.",
        "<|start|> opens a message.",
    ],
)
def test_prose_that_names_the_tokens_is_left_alone(text: str) -> None:
    assert split_harmony(text) is None
    messages = [{"role": "assistant", "content": text}]
    fixed, repaired = repair_history(messages)
    assert repaired == 0 and fixed is messages


def test_terminated_replies_still_count_as_structure() -> None:
    assert split_harmony("Done.<|end|>") == ("Done.", "")
    assert split_harmony("Done.<|end|>\n") == ("Done.", "")
    # text after a header-opened body needs no header of its own: the none-format shape
    assert split_harmony("<|channel|>final<|message|>A.<|end|> B.") == ("A.\n\nB.", "")


# ---------------------------------------------------------------------------
# repair_history
# ---------------------------------------------------------------------------


def test_repair_cuts_assistant_turns_back_to_the_answer() -> None:
    messages = [
        {"role": "system", "content": "You are Mina."},
        {"role": "user", "content": "Hello <|channel|> literally"},
        {"role": "assistant", "content": NONE_FORMAT_CONTENT, "name": "mina"},
        {"role": "user", "content": "Next."},
    ]
    original = copy.deepcopy(messages)
    fixed, repaired = repair_history(messages)
    assert repaired == 1
    assert fixed[2] == {
        "role": "assistant",
        "content": "Mina looked up from the stove and smiled.",
        "name": "mina",
    }
    # Only assistant turns are touched, and the caller's list is not mutated.
    assert fixed[1] == original[1]
    assert messages == original


def test_repair_returns_the_same_list_when_nothing_needs_it() -> None:
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    fixed, repaired = repair_history(messages)
    assert repaired == 0
    assert fixed is messages


def test_repair_keeps_the_thought_that_led_to_a_tool_call() -> None:
    calls = [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    messages = [{"role": "assistant", "content": NONE_FORMAT_CONTENT, "tool_calls": calls}]
    fixed, _ = repair_history(messages)
    assert fixed[0]["reasoning_content"] == "We need to respond as Mina, third person."
    assert fixed[0]["tool_calls"] == calls
    # ...but never over reasoning the client already sent.
    own = [{**messages[0], "reasoning_content": "mine"}]
    assert repair_history(own)[0][0]["reasoning_content"] == "mine"


def test_repair_handles_content_parts() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": NONE_FORMAT_CONTENT},
                {"type": "text", "text": "plain"},
            ],
        }
    ]
    fixed, repaired = repair_history(messages)
    assert repaired == 1
    assert fixed[0]["content"] == [
        {"type": "text", "text": "Mina looked up from the stove and smiled."},
        {"type": "text", "text": "plain"},
    ]


# ---------------------------------------------------------------------------
# guard_tool_schemas
# ---------------------------------------------------------------------------


def _tool(parameters: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": "f", "parameters": parameters}}


def test_array_parameters_without_items_get_items() -> None:
    tools = [
        _tool(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array"},
                    "ok": {"type": "array", "items": {"type": "string"}},
                    "maybe": {"type": ["array", "null"]},
                    "nested": {
                        "type": "object",
                        "properties": {"deep": {"anyOf": [{"type": "array"}, {"type": "string"}]}},
                    },
                    "grid": {"type": "array", "items": {"type": "array"}},
                },
            }
        )
    ]
    original = copy.deepcopy(tools)
    fixed, added = guard_tool_schemas(tools)
    assert added == 4
    props = fixed[0]["function"]["parameters"]["properties"]
    assert props["tags"] == {"type": "array", "items": {}}
    assert props["ok"] == {"type": "array", "items": {"type": "string"}}
    assert props["maybe"]["items"] == {}
    assert props["nested"]["properties"]["deep"]["anyOf"][0] == {"type": "array", "items": {}}
    assert props["grid"]["items"] == {"type": "array", "items": {}}
    assert tools == original  # never the caller's object


def test_tools_that_need_nothing_come_back_as_the_same_object() -> None:
    tools = [_tool({"type": "object", "properties": {"x": {"type": "array", "items": {}}}})]
    fixed, added = guard_tool_schemas(tools)
    assert added == 0
    assert fixed is tools
    assert guard_tool_schemas(None) == (None, 0)
    assert guard_tool_schemas([]) == ([], 0)


def test_prefix_items_counts_as_items() -> None:
    tools = [_tool({"type": "array", "prefixItems": [{"type": "string"}]})]
    assert guard_tool_schemas(tools)[1] == 0


# ---------------------------------------------------------------------------
# the request path: _repair_for_template
# ---------------------------------------------------------------------------


def test_the_request_path_repairs_history_only_for_harmony_models() -> None:
    payload: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": NONE_FORMAT_CONTENT},
            {"role": "user", "content": "again"},
        ],
        "tools": [_tool({"type": "object", "properties": {"t": {"type": "array"}}})],
    }
    harmony_payload = copy.deepcopy(payload)
    _repair_for_template(harmony_payload, record(harmony_meta()), "bench")
    assert harmony_payload["messages"][1]["content"] == "Mina looked up from the stove and smiled."
    assert harmony_payload["tools"][0]["function"]["parameters"]["properties"]["t"]["items"] == {}

    other_payload = copy.deepcopy(payload)
    other = record(GgufMeta(architecture="qwen3", chat_template=CHATML_TEMPLATE))
    _repair_for_template(other_payload, other, "bench")
    assert other_payload["messages"] == payload["messages"]  # not harmony: history untouched
    assert other_payload["tools"][0]["function"]["parameters"]["properties"]["t"]["items"] == {}


# ---------------------------------------------------------------------------
# the launch flag
# ---------------------------------------------------------------------------


def test_a_harmony_model_launches_with_deepseek_under_the_none_default(
    tmp_path: Path, config: Config
) -> None:
    binary = make_binary(tmp_path)
    harmony = make_record(tmp_path, meta=harmony_meta())
    argv = sup(config, binary).build_command(harmony, make_plan(), port=18100, features=B10425)
    assert value_after(argv, "--reasoning-format") == "deepseek"


def test_a_saved_choice_still_wins_for_a_harmony_model(tmp_path: Path, config: Config) -> None:
    binary = make_binary(tmp_path)
    chosen = make_record(
        tmp_path, meta=harmony_meta(), settings=ModelSettings(reasoning_format="none")
    )
    argv = sup(config, binary).build_command(chosen, make_plan(), port=18100, features=B10425)
    assert value_after(argv, "--reasoning-format") == "none"


def test_other_models_keep_the_global_default(tmp_path: Path, config: Config) -> None:
    binary = make_binary(tmp_path)
    plain = make_record(tmp_path)
    argv = sup(config, binary).build_command(plain, make_plan(), port=18100, features=B10425)
    assert value_after(argv, "--reasoning-format") == "none"


@pytest.fixture
def config(tmp_path: Path) -> Config:
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    return cfg
