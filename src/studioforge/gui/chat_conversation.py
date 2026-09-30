"""The Chat tab's conversation model and its Markdown rendering.

The Chat tab used to keep a flat ``history`` list of OpenAI dicts, which made
every per-message action (edit, delete, regenerate, copy) an index juggling act
inside element callbacks. This module owns the conversation as a list of
:class:`ChatMessage` blocks with stable ids instead, and derives everything the
tab needs from it -- the request ``messages``, the "Copy all" transcript, which
blocks may be regenerated -- as plain functions that are testable without
NiceGUI, a GPU or a model.

Nothing here is persisted: a conversation lives in memory for one page view.

Two invariants are load-bearing and covered by tests:

* **Model output is untrusted.** :func:`render_markdown` escapes every piece of
  raw HTML in the source and re-emits markdown2's own output through a tag and
  attribute allowlist, so neither a prompt-injected ``<script>`` nor a
  ``javascript:`` link can reach the browser.
* **Only real turns are context.** Failed assistant turns, empty placeholders
  and the model's reasoning never travel back to the model.
"""

from __future__ import annotations

import html
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Final

import markdown2  # type: ignore[import-untyped]  # a NiceGUI dependency, unstubbed

from studioforge.gui import state as st

USER: Final = "user"
ASSISTANT: Final = "assistant"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ChatMessage:
    """One block of the conversation.

    ``content`` is always the raw Markdown source, never rendered HTML, so that
    "Copy raw" and edits round-trip exactly what the model or user wrote. For an
    assistant turn it holds the *answer* only; the thinking lives in
    ``reasoning`` because it is shown folded and must never be sent back.
    """

    id: str
    role: str
    content: str
    reasoning: str = ""
    images: list[str] = field(default_factory=list)
    model: str | None = None
    metrics: Any = None
    status: str = ""
    failed: bool = False
    created_at: float = field(default_factory=time.time)
    edited: bool = False

    @property
    def in_context(self) -> bool:
        """Whether this turn is sent to the model as history.

        A failed or errored reply is noise the model would imitate, and a blank
        assistant turn (a placeholder, or a stop before the first token) is an
        empty message some chat templates reject outright.
        """
        if self.role != ASSISTANT:
            return True
        return not self.failed and bool(self.content.strip())


class Conversation:
    """An ordered, in-memory list of :class:`ChatMessage` blocks."""

    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []

    def __len__(self) -> int:
        return len(self.messages)

    # -- lookup ---------------------------------------------------------------

    def get(self, message_id: str) -> ChatMessage | None:
        for message in self.messages:
            if message.id == message_id:
                return message
        return None

    def index(self, message_id: str) -> int:
        """Position of ``message_id``, or ``-1`` when it is not (or no longer) here."""
        for position, message in enumerate(self.messages):
            if message.id == message_id:
                return position
        return -1

    def _last(self, role: str) -> ChatMessage | None:
        for message in reversed(self.messages):
            if message.role == role:
                return message
        return None

    def last_assistant(self) -> ChatMessage | None:
        return self._last(ASSISTANT)

    def last_user(self) -> ChatMessage | None:
        return self._last(USER)

    # -- mutation -------------------------------------------------------------

    def add_user(self, text: str, images: Sequence[str] = ()) -> ChatMessage:
        message = ChatMessage(id=_new_id(), role=USER, content=text, images=list(images))
        self.messages.append(message)
        return message

    def add_assistant(self, model: str | None) -> ChatMessage:
        """An empty reply block the stream fills in; ``status`` starts as ``"streaming"``."""
        message = ChatMessage(
            id=_new_id(), role=ASSISTANT, content="", model=model, status="streaming"
        )
        self.messages.append(message)
        return message

    def edit(self, message_id: str, content: str) -> bool:
        """Replace a block's raw text; ``False`` when absent or nothing changed.

        An unchanged save is not an edit, so it must not earn the "edited" mark.
        """
        message = self.get(message_id)
        if message is None or message.content == content:
            return False
        message.content = content
        message.edited = True
        return True

    def delete(self, message_id: str) -> bool:
        position = self.index(message_id)
        if position == -1:
            return False
        del self.messages[position]
        return True

    def truncate_after(self, message_id: str) -> list[ChatMessage]:
        """Drop every block after ``message_id`` and return them (``[]`` when absent)."""
        position = self.index(message_id)
        if position == -1:
            return []
        dropped = self.messages[position + 1 :]
        del self.messages[position + 1 :]
        return dropped

    def clear(self) -> None:
        self.messages.clear()

    # -- regenerate -----------------------------------------------------------

    def can_regenerate(self, message_id: str) -> bool:
        """Whether a regenerate / retry button belongs on this block.

        An assistant block qualifies only when it is the *last block of all* and
        a user turn precedes it: regenerating truncates after the question it
        answers, so an earlier reply would silently take later turns with it.
        A user block qualifies as a "retry" when it is the last user turn and
        nothing after it is context -- no reply yet, or only failed or blank
        ones (which the retry replaces).
        """
        position = self.index(message_id)
        if position == -1:
            return False
        message = self.messages[position]
        if message.role == ASSISTANT:
            if position != len(self.messages) - 1:
                return False
            # Everything between the question and this reply goes too, so it
            # must be nothing worth keeping (a failed or blank reply). A real
            # answer sitting there (its question deleted) would vanish.
            for earlier in reversed(self.messages[:position]):
                if earlier.role == USER:
                    return True
                if earlier.in_context:
                    return False
            return False
        if message.role == USER:
            return all(
                m.role == ASSISTANT and not m.in_context for m in self.messages[position + 1 :]
            )
        return False

    def regenerate_point(self, message_id: str) -> ChatMessage | None:
        """The user turn a regenerate answers, or ``None`` when it is not allowed.

        The caller truncates after this block and streams a fresh reply to it.
        """
        if not self.can_regenerate(message_id):
            return None
        position = self.index(message_id)
        for message in reversed(self.messages[: position + 1]):
            if message.role == USER:
                return message
        return None

    # -- derivations ----------------------------------------------------------

    def request_messages(
        self, *, system: str | None, keep_history: bool, upto: str | None = None
    ) -> list[dict[str, Any]]:
        """OpenAI ``messages`` for a request answering the turn ``upto``.

        ``upto`` defaults to the last user turn. An assistant id is accepted and
        resolves to the user turn before it, which is what a regenerate means.
        Consecutive user turns (left by a delete or a failed reply) are sent as
        they are: merging them would put words in the user's mouth.

        Raises ``ValueError`` when there is no user turn to answer -- sending a
        request without one is a bug in the caller, not something to paper over.
        """
        if upto is None:
            target = self.last_user()
        else:
            position = self.index(upto)
            target = None
            if position != -1:
                target = next(
                    (m for m in reversed(self.messages[: position + 1]) if m.role == USER), None
                )
        if target is None:
            raise ValueError("no user message to answer")

        out: list[dict[str, Any]] = []
        if system is not None and system.strip():
            out.append({"role": "system", "content": system})
        end = self.index(target.id)
        turns = self.messages[: end + 1] if keep_history else [target]
        for message in turns:
            if not message.in_context:
                continue
            if message.role == USER:
                content = st.build_chat_content(message.content, message.images)
                out.append({"role": USER, "content": content})
            else:
                out.append({"role": ASSISTANT, "content": message.content})
        return out

    def as_markdown(self) -> str:
        """The whole transcript as Markdown, for "Copy all".

        Reasoning is left out (it is scaffolding, not the conversation), blank
        placeholders are skipped, and failed turns stay in but are marked so a
        pasted transcript does not pass an error off as an answer.
        """
        blocks: list[str] = []
        for message in self.messages:
            if message.role == USER:
                header = "**You:**"
            else:
                if not message.failed and not message.content.strip():
                    continue
                header = f"**{message.model or 'Assistant'}:**"
            parts = [header]
            if message.failed:
                note = message.status.strip()
                plain = not note or note.lower() == "failed"
                parts.append("_(failed)_" if plain else f"_(failed: {note})_")
            if message.content.strip():
                parts.append(message.content.strip())
            if message.images:
                count = len(message.images)
                parts.append(f"[{count} image{'s' if count != 1 else ''}]")
            blocks.append("\n\n".join(parts))
        return "\n\n---\n\n".join(blocks)


# -- Markdown rendering -------------------------------------------------------

MARKDOWN_EXTRAS: Final[tuple[str, ...]] = (
    "fenced-code-blocks",
    "tables",
    "strike",
    "cuddled-lists",
    "break-on-newline",
    "code-friendly",
)
"""markdown2 extras for chat text.

``break-on-newline`` because chat models (and users) rely on single newlines;
``code-friendly`` so ``snake_case_names`` are not italicised (it also disables
``_underscore_`` emphasis -- ``*stars*`` still work). Pygments is installed, so
fenced blocks with a known language come out as ``div.codehilite`` spans.
"""

_FENCE = re.compile(r"^ {0,3}(`{3,})")
_SAFE_LINK = re.compile(r"^(https?:|mailto:)", re.IGNORECASE)
_SAFE_IMAGE = re.compile(r"^https?:", re.IGNORECASE)
_TEXT_ALIGN = re.compile(r"^text-align:\s*(left|right|center);?$")

_ALLOWED_TAGS: Final = frozenset(
    {
        "p", "br", "hr", "strong", "b", "em", "i", "s", "del", "code", "pre",
        "div", "span", "ul", "ol", "li", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "thead", "tbody", "tr", "th", "td",
        "a", "img",
    }
)  # fmt: skip
_VOID_TAGS: Final = frozenset({"br", "hr", "img"})
_CLASS_TAGS: Final = frozenset({"div", "span", "code", "pre"})


def _closes(line: str, ticks: str, opener: str) -> bool:
    """A closing fence is at least as long as the opener and carries no info string."""
    return len(ticks) >= len(opener) and not line.strip()[len(ticks) :].strip()


def close_open_fence(text: str) -> str:
    """``text`` with a closing fence appended if a ``` block is still open.

    Mid-stream the model has opened a code block but not yet closed it; without
    this, everything after the opening fence renders as one paragraph with
    literal backticks until the closing fence arrives, then jumps.
    """
    opener: str | None = None
    for line in text.splitlines():
        match = _FENCE.match(line)
        if match is None:
            continue
        ticks = match.group(1)
        if opener is None:
            opener = ticks
        elif _closes(line, ticks, opener):
            opener = None
    if opener is None:
        return text
    return text + ("" if text.endswith("\n") else "\n") + opener


class _Sanitizer(HTMLParser):
    """Re-emits markdown2's HTML through a tag/attribute allowlist.

    markdown2's ``safe_mode="escape"`` escapes raw HTML in the source, but it
    still emits ``<img src="javascript:...">`` for image syntax, so its output is
    walked once more and only known-safe tags and attributes survive.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.out: list[str] = []
        self._skipped: list[str] = []  # end tags to drop (an <a> whose href was unsafe)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in _ALLOWED_TAGS:
            return
        values = {name: value or "" for name, value in attrs}
        kept: list[tuple[str, str]] = []
        if tag == "a":
            href = values.get("href", "").strip()
            if not _SAFE_LINK.match(href):
                self._skipped.append("a")
                return
            kept.append(("href", href))
            if values.get("title"):
                kept.append(("title", values["title"]))
            kept += [("target", "_blank"), ("rel", "noopener noreferrer nofollow")]
            self._skipped.append("")
        elif tag == "img":
            src = values.get("src", "").strip()
            if not _SAFE_IMAGE.match(src):
                self.out.append(html.escape(values.get("alt", ""), quote=False))
                return
            kept.append(("src", src))
            kept.append(("alt", values.get("alt", "")))
            kept += [("loading", "lazy"), ("referrerpolicy", "no-referrer")]
        elif tag in _CLASS_TAGS and values.get("class"):
            kept.append(("class", values["class"]))
        elif tag in ("th", "td") and _TEXT_ALIGN.match(values.get("style", "")):
            kept.append(("style", values["style"]))
        rendered = "".join(f' {name}="{html.escape(value, quote=True)}"' for name, value in kept)
        self.out.append(f"<{tag}{rendered}{' /' if tag in _VOID_TAGS else ''}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag not in _ALLOWED_TAGS or tag in _VOID_TAGS:
            return
        if tag == "a" and self._skipped and self._skipped.pop() == "a":
            return
        self.out.append(f"</{tag}>")

    def handle_data(self, data: str) -> None:
        self.out.append(html.escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        self.out.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.out.append(f"&#{name};")


def _escaped_pre(text: str) -> str:
    return f"<pre>{html.escape(text, quote=False)}</pre>"


def render_markdown(text: str) -> str:
    """Safe HTML for a chat block's Markdown; never raises.

    Raw HTML in ``text`` is shown as text, links keep only ``http(s):`` and
    ``mailto:`` targets (others become plain text) and open in a new tab, and
    images load only over ``http(s):``. Any failure falls back to the source in
    an escaped ``<pre>`` -- a chat must never lose a reply to a renderer bug.
    """
    try:
        if not text:
            return ""
        source = close_open_fence(text)
        raw = markdown2.markdown(source, extras=list(MARKDOWN_EXTRAS), safe_mode="escape")
        sanitizer = _Sanitizer()
        sanitizer.feed(str(raw))
        sanitizer.close()
        return "".join(sanitizer.out).strip()
    except Exception:
        try:
            return _escaped_pre(str(text))
        except Exception:
            return "<pre></pre>"


# -- plain text ---------------------------------------------------------------

# Every span below has a length bound. Unbounded, each match attempt can scan
# to the end of the line and the URL part used to backtrack against itself, so
# a long line of ``[a](`` or ``*a _b`` took minutes -- on "Copy formatted",
# inside the event loop the API gateway shares. Spans longer than the bound
# keep their markers, which a clipboard approximation can afford.
_IMAGE_MD = re.compile(r"!\[([^\]]{0,500})\]\(([^)\s]{0,2000}+)(?:\s[^)]{0,500})?\)")
_LINK_MD = re.compile(r"\[([^\]]{1,500}+)\]\(([^)\s]{0,2000}+)(?:\s[^)]{0,500})?\)")
_AUTOLINK = re.compile(r"<((?:https?:|mailto:)[^>\s]{1,2000})>", re.IGNORECASE)
_HEADING = re.compile(r"^ {0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^(\s*)[*+-]\s+")
_QUOTE = re.compile(r"^\s{0,3}>\s?")
_TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_HRULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.{1,300}?)(?<=\S)\1")
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)(.{1,300}?)(?<=\S)\*(?![\w*])")
_STRIKE = re.compile(r"~~(?=\S)(.{1,300}?)(?<=\S)~~")
_INLINE_CODE = re.compile(r"(`{1,16})(.{1,2000}?)\1")


def _plain_inline(line: str) -> str:
    # Inline code is lifted out first so emphasis stripping never touches it.
    codes: list[str] = []

    def stash(match: re.Match[str]) -> str:
        codes.append(match.group(2).strip())
        return f"\x00{len(codes) - 1}\x00"

    line = _INLINE_CODE.sub(stash, line)
    line = _IMAGE_MD.sub(lambda m: m.group(1) or "image", line)
    line = _LINK_MD.sub(
        lambda m: m.group(1) if m.group(2) in ("", m.group(1)) else f"{m.group(1)} ({m.group(2)})",
        line,
    )
    line = _AUTOLINK.sub(r"\1", line)
    line = _BOLD.sub(r"\2", line)
    line = _ITALIC.sub(r"\1", line)
    line = _STRIKE.sub(r"\1", line)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], line)


def plain_text(text: str) -> str:
    """Markdown as readable plain text: the ``text/plain`` half of "Copy formatted".

    Code block contents are kept verbatim (fences dropped), emphasis markers are
    removed, bullets become ``- ``, links become ``text (url)``. Deliberately a
    line-based approximation: a clipboard fallback must be robust, not exact.
    """
    try:
        out: list[str] = []
        opener: str | None = None
        for line in (text or "").splitlines():
            fence = _FENCE.match(line)
            if fence is not None:
                ticks = fence.group(1)
                if opener is None:
                    opener = ticks
                    continue
                if _closes(line, ticks, opener):
                    opener = None
                    continue
            if opener is not None:
                out.append(line)
                continue
            if _TABLE_RULE.match(line) and "|" in line:
                continue
            if _HRULE.match(line):
                out.append("")
                continue
            heading = _HEADING.match(line)
            if heading is not None:
                line = heading.group(1)
            line = _QUOTE.sub("", line)
            line = _BULLET.sub(r"\1- ", line)
            out.append(_plain_inline(line))
        return "\n".join(out).strip("\n")
    except Exception:
        return text or ""
