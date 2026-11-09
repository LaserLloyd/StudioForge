"""Harmony chat format (gpt-oss): detection, the reasoning-format default, history repair (D82).

gpt-oss speaks OpenAI's *harmony* format: every assistant turn is a run of
``<|start|>assistant<|channel|>NAME<|message|>BODY<|end|>`` blocks, ``analysis``
for the chain of thought and ``final`` for the answer. llama.cpp's gpt-oss
handler parses that back out of the generated text with a strict PEG grammar,
and a reply that does not fit the grammar is not degraded to plain text -- the
server throws, and the client gets HTTP 500 ``The model produced output that
does not match the expected peg-native format`` (an error frame when
streaming).

Measured on b11718 (2026-11-07, a 7-turn role-play session, temperature 0): the
500 is self-inflicted. Under StudioForge's ``--reasoning-format none`` default
(D12) the handler hands the analysis block back INSIDE ``message.content``, tags
and all, followed by the answer with its ``<|start|>assistant<|channel|>final
<|message|>`` header dropped. A multi-turn client sends that content back as the
assistant turn; the template wraps it in a final-channel message and the engine
re-tokenizes the markup as real control tokens, so the history now shows answers
that follow ``<|end|>`` with no header at all. By turn 3-4 the model copies the
pattern, writes its reply straight after ``<|start|>assistant``, and the parser
throws. With the analysis extracted (``deepseek``/``auto``) the same session ran
7/7.

So, for a model whose template is harmony:

* :func:`effective_reasoning_format` -- with no per-model choice, the launch uses
  ``deepseek`` instead of the global ``none``: harmony markup in ``content`` is
  never something an OpenAI client can use, and D12's reason for ``none`` (an
  empty ``content`` when the budget is spent thinking) does not hold here -- a
  truncated analysis still comes back as text.
* :func:`repair_history` -- an assistant message that still carries harmony
  markup (a client replaying replies from before this fix, or one that asked
  for ``reasoning_format: none`` per request) is cut back to its answer before it
  reaches the engine, so the history can never teach the model the broken shape.

And for every model, :func:`guard_tool_schemas`: an ``array`` parameter with no
``items`` gets ``items: {}`` (the same schema -- "any item"). Templates index
``param_spec['items']``; for a missing key Jinja falls back to the dict's
built-in ``items()`` method, and gpt-oss's template then dies with "Function is
not a bool value" before a token is generated (seen 2026-11-06 on every tool
case of a benchmark). OpenAI's own API rejects such a schema outright.
"""

from __future__ import annotations

import copy
import re
from typing import Any

__all__ = [
    "HARMONY_CONTROL_RE",
    "effective_reasoning_format",
    "guard_tool_schemas",
    "record_uses_harmony",
    "repair_history",
    "split_harmony",
    "uses_harmony",
]

#: The control tokens harmony uses. ``<|start|>`` opens a message header,
#: ``<|channel|>`` names the channel, ``<|constrain|>`` qualifies a recipient's
#: payload type, ``<|message|>`` ends the header, and ``<|end|>`` / ``<|return|>``
#: / ``<|call|>`` end the body.
HARMONY_CONTROL_RE = re.compile(r"<\|(start|end|message|channel|constrain|return|call)\|>")

#: The GGUF ``general.architecture`` of OpenAI's open-weight models.
_HARMONY_ARCHITECTURES = frozenset({"gpt-oss"})


def uses_harmony(template: str | None, architecture: str | None = None) -> bool:
    """Whether a chat template (or a model architecture) speaks harmony."""
    if architecture and architecture.lower() in _HARMONY_ARCHITECTURES:
        return True
    if not template:
        return False
    return "<|channel|>" in template and "<|message|>" in template and "<|start|>" in template


def record_uses_harmony(record: Any) -> bool:
    """:func:`uses_harmony` for a registry record (its embedded template and architecture).

    The embedded template is the model's identity: a ``chat_template_file``
    override changes how prompts are rendered, not what the weights were
    trained to emit.
    """
    meta = getattr(record, "meta", None)
    template = getattr(meta, "chat_template", None) if meta is not None else None
    architecture = getattr(meta, "architecture", None) or getattr(record, "architecture", None)
    return uses_harmony(template, architecture)


def effective_reasoning_format(
    setting: str | None, default: str | None, *, harmony: bool
) -> tuple[str | None, str]:
    """The ``--reasoning-format`` a launch uses, and why: ``(format, source)``.

    ``source`` is ``"setting"`` (the model's own saved choice, always wins),
    ``"harmony"`` (a harmony model under the ``none`` default, moved to
    ``deepseek``) or ``"default"`` (``models.default_reasoning_format``).
    """
    if setting is not None:
        return setting, "setting"
    if harmony and (default or "none") == "none":
        return "deepseek", "harmony"
    return default, "default"


#: What may follow ``<|start|>``: a role (``functions.NAME`` is a tool's reply).
#: No word boundary after it: the broken turn-4 shape is ``<|start|>assistantMina
#: nodded.``, the reply glued to the role.
_ROLE_RE = re.compile(r"\s*(?:assistant|user|system|developer|tool|functions\.[\w.-]+)")
#: What may follow ``<|channel|>``: one of harmony's three channels.
_CHANNEL_RE = re.compile(r"\s*(?:analysis|commentary|final)\b")


def split_harmony(text: str) -> tuple[str, str] | None:
    """Split text that carries harmony markup into ``(answer, analysis)``.

    ``None`` when there is nothing to repair: no harmony control token, or
    tokens that do not form harmony structure -- prose or a code sample that
    merely *names* ``<|start|>`` or ``<|end|>`` is left exactly as written.
    Structure means: a header (``<|start|>`` + a role, or ``<|channel|>`` + a
    channel name) opens only where a message can begin (the start of the text
    or right after ``<|end|>`` / ``<|return|>`` / ``<|call|>``); ``<|message|>``
    and ``<|constrain|>`` only inside a header; and a terminator that ends text
    no header opened is followed by the next header or by nothing.

    Text that is not under any header -- including text after an ``<|end|>``
    with no new header, the exact shape ``--reasoning-format none`` produces,
    and a reply written straight after ``<|start|>assistant`` with no
    ``<|message|>`` -- counts as the answer. ``analysis`` bodies are the chain
    of thought; ``final`` and a recipient-less ``commentary`` (a preamble,
    which llama.cpp also returns as content) are the answer; a body addressed
    to a recipient (``to=...``, a tool call) is dropped, since the message's
    ``tool_calls`` carry it.
    """
    matches = list(HARMONY_CONTROL_RE.finditer(text))
    if not matches:
        return None
    answer: list[str] = []
    analysis: list[str] = []
    state = "body"  # body | role | channel | constrain
    channel = "final"
    recipient = False
    headed = False  # the current body was opened by a header's <|message|>
    pos = 0

    def flush(chunk: str) -> None:
        if not chunk or recipient:
            return
        (analysis if channel == "analysis" else answer).append(chunk)

    def flush_role(chunk: str) -> None:
        # A role with no <|message|> after it: whatever follows the role word is
        # the reply (the turn-4 shape), unless the header named a recipient.
        if " to=" not in f" {chunk}":
            role = _ROLE_RE.match(chunk)
            flush(chunk[role.end() :] if role else chunk)

    for i, match in enumerate(matches):
        chunk = text[pos : match.start()]
        token = match.group(1)
        pos = match.end()
        following = text[pos : matches[i + 1].start() if i + 1 < len(matches) else len(text)]
        if token in ("start", "channel"):
            opens = state == "body" and not headed and not chunk.strip()
            if token == "start":
                if not (opens and _ROLE_RE.match(following)):
                    return None
                state, channel, recipient = "role", "final", False
                continue
            if not (opens or state == "role") or not _CHANNEL_RE.match(following):
                return None
            if state == "role" and " to=" in f" {chunk}":
                recipient = True
            state = "channel"
        elif token in ("constrain", "message"):
            if state == "body":
                return None
            if state in ("role", "channel") and " to=" in f" {chunk}":
                recipient = True
            if state == "channel":
                channel = (chunk.split() or ["final"])[0]
            state = "constrain" if token == "constrain" else "body"
            headed = token == "message"
        else:  # end / return / call: the body is over; bare text after it is answer
            if state in ("channel", "constrain"):
                return None
            if state == "role":
                flush_role(chunk)
            else:
                flush(chunk)
                if not headed:
                    # Text no header opened ends only where the next header (or
                    # the text) does: "the <|end|> token" in prose is not a turn.
                    after_next = matches[i + 1].group(1) if i + 1 < len(matches) else None
                    if following.strip() or after_next not in (None, "start", "channel"):
                        return None
            state, channel, recipient, headed = "body", "final", False, False
    if state == "body":
        flush(text[pos:])
    elif state == "role":
        flush_role(text[pos:])
    # One body per block; blocks are separate paragraphs, not one run-on line.
    return (
        "\n\n".join(part.strip() for part in answer if part.strip()),
        "\n\n".join(part.strip() for part in analysis if part.strip()),
    )


def repair_history(messages: list[Any]) -> tuple[list[Any], int]:
    """Cut harmony markup out of assistant turns: ``(messages, how_many_repaired)``.

    Returns the input list itself when nothing needed repair. A repaired
    message keeps every other field. Its analysis is dropped (harmony drops a
    finished turn's chain of thought from the prompt anyway) unless the message
    carries ``tool_calls`` and no reasoning of its own, where harmony keeps the
    thought that led to the call: then it moves to ``reasoning_content``, which
    llama.cpp maps to the template's ``thinking`` field.
    """
    out: list[Any] = []
    repaired = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            out.append(message)
            continue
        content = message.get("content")
        if isinstance(content, str):
            split = split_harmony(content)
            if split is None:
                out.append(message)
                continue
            answer, analysis = split
            fixed = {**message, "content": answer}
            if (
                analysis
                and message.get("tool_calls")
                and not (message.get("reasoning_content") or message.get("thinking"))
            ):
                fixed["reasoning_content"] = analysis
            out.append(fixed)
            repaired += 1
        elif isinstance(content, list):
            parts: list[Any] = []
            changed = False
            for part in content:
                text = part.get("text") if isinstance(part, dict) else None
                split = split_harmony(text) if isinstance(text, str) else None
                if split is None:
                    parts.append(part)
                else:
                    parts.append({**part, "text": split[0]})
                    changed = True
            out.append({**message, "content": parts} if changed else message)
            repaired += int(changed)
        else:
            out.append(message)
    return (out if repaired else messages), repaired


#: Schema keywords whose value is a map of name -> subschema.
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
#: Schema keywords whose value is a list of subschemas.
_SCHEMA_LISTS = ("anyOf", "oneOf", "allOf", "prefixItems")
#: Schema keywords whose value is one subschema.
_SCHEMA_ONES = (
    "items",
    "additionalProperties",
    "additionalItems",
    "unevaluatedProperties",
    "unevaluatedItems",
    "not",
    "if",
    "then",
    "else",
    "contains",
    "propertyNames",
)
_MAX_SCHEMA_DEPTH = 64


def _guard_schema(schema: Any, depth: int = 0) -> int:
    if not isinstance(schema, dict) or depth > _MAX_SCHEMA_DEPTH:
        return 0
    added = 0
    kind = schema.get("type")
    is_array = kind == "array" or (isinstance(kind, list) and "array" in kind)
    if is_array and "items" not in schema and "prefixItems" not in schema:
        schema["items"] = {}
        added += 1
    for key in _SCHEMA_MAPS:
        sub = schema.get(key)
        if isinstance(sub, dict):
            for value in sub.values():
                added += _guard_schema(value, depth + 1)
    for key in _SCHEMA_LISTS:
        sub = schema.get(key)
        if isinstance(sub, list):
            for value in sub:
                added += _guard_schema(value, depth + 1)
    for key in _SCHEMA_ONES:
        added += _guard_schema(schema.get(key), depth + 1)
    return added


def guard_tool_schemas(tools: Any) -> tuple[Any, int]:
    """Give every ``array`` schema in the tools' parameters an ``items``: ``(tools, added)``.

    Works on a deep copy, never the caller's object; returns the input itself
    (same object) when no schema needed the guard.
    """
    if not isinstance(tools, list) or not tools:
        return tools, 0
    fixed = copy.deepcopy(tools)
    added = 0
    for tool in fixed:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict):
            added += _guard_schema(function.get("parameters"))
    return (fixed, added) if added else (tools, 0)
