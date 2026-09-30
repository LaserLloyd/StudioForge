"""The Chat tab's conversation model and its Markdown rendering.

Pure Python (no NiceGUI, no GPU, no model): the block operations, what a
request sends, the "Copy all" transcript, and -- load-bearing -- that untrusted
model output can never inject HTML or script into the page.
"""

from __future__ import annotations

import random
import re
import time
from html.parser import HTMLParser

import pytest

from studioforge.gui.chat_conversation import (
    ChatMessage,
    Conversation,
    close_open_fence,
    plain_text,
    render_markdown,
)

IMG = "data:image/png;base64,AAAA"


def _chat(*turns: str) -> tuple[Conversation, list[ChatMessage]]:
    """A conversation from alternating texts: user, assistant, user, ..."""
    convo = Conversation()
    made: list[ChatMessage] = []
    for position, text in enumerate(turns):
        if position % 2 == 0:
            made.append(convo.add_user(text))
        else:
            reply = convo.add_assistant("m1")
            reply.content, reply.status = text, ""
            made.append(reply)
    return convo, made


# -- blocks -------------------------------------------------------------------


def test_add_user_and_assistant_defaults() -> None:
    convo = Conversation()
    user = convo.add_user("hi", [IMG])
    reply = convo.add_assistant("qwen")
    assert len(convo) == 2
    assert (user.role, user.content, user.images) == ("user", "hi", [IMG])
    assert (reply.role, reply.content, reply.model, reply.status) == (
        "assistant",
        "",
        "qwen",
        "streaming",
    )
    assert not reply.failed and not reply.edited and reply.reasoning == ""
    assert user.id != reply.id and len(user.id) == 12
    assert convo.get(user.id) is user and convo.index(reply.id) == 1
    assert convo.get("nope") is None and convo.index("nope") == -1


def test_add_user_copies_images() -> None:
    images = [IMG]
    user = Conversation().add_user("x", images)
    images.append("other")
    assert user.images == [IMG]


def test_edit_sets_flag_only_on_change() -> None:
    convo, (user,) = _chat("hello")
    assert convo.edit(user.id, "hello") is False
    assert user.edited is False
    assert convo.edit(user.id, "hello there") is True
    assert (user.content, user.edited) == ("hello there", True)
    assert convo.edit("missing", "x") is False


def test_delete_truncate_clear() -> None:
    convo, (u1, a1, u2, a2) = _chat("q1", "a1", "q2", "a2")
    assert convo.delete(a1.id) is True
    assert convo.delete(a1.id) is False
    assert [m.id for m in convo.messages] == [u1.id, u2.id, a2.id]
    assert convo.truncate_after("missing") == []
    dropped = convo.truncate_after(u1.id)
    assert [m.id for m in dropped] == [u2.id, a2.id]
    assert [m.id for m in convo.messages] == [u1.id]
    assert convo.truncate_after(u1.id) == []
    convo.clear()
    assert len(convo) == 0 and convo.last_user() is None and convo.last_assistant() is None


def test_last_user_and_last_assistant() -> None:
    convo, (_u1, a1, u2) = _chat("q1", "a1", "q2")
    assert convo.last_user() is u2
    assert convo.last_assistant() is a1


# -- regenerate ---------------------------------------------------------------


def test_regenerate_last_assistant() -> None:
    convo, (u1, a1, u2, a2) = _chat("q1", "a1", "q2", "a2")
    assert convo.can_regenerate(a2.id) is True
    assert convo.regenerate_point(a2.id) is u2
    # An earlier reply would drag later turns away with it.
    assert convo.can_regenerate(a1.id) is False
    assert convo.regenerate_point(a1.id) is None
    # A user turn that already has a real answer is not a retry.
    assert convo.can_regenerate(u2.id) is False
    assert convo.can_regenerate(u1.id) is False


def test_regenerate_assistant_needs_a_preceding_user() -> None:
    convo, (u1, a1) = _chat("q1", "a1")
    convo.delete(u1.id)
    assert convo.can_regenerate(a1.id) is False


def test_regenerate_assistant_not_last_after_trailing_user() -> None:
    convo, (_u1, a1, u2) = _chat("q1", "a1", "q2")
    assert convo.can_regenerate(a1.id) is False
    assert convo.can_regenerate(u2.id) is True
    assert convo.regenerate_point(u2.id) is u2


def test_retry_user_after_failed_or_blank_reply() -> None:
    convo, (_u1, _a1, u2) = _chat("q1", "a1", "q2")
    failed = convo.add_assistant("m1")
    failed.failed, failed.status = True, "error: boom"
    assert convo.can_regenerate(u2.id) is True
    assert convo.regenerate_point(u2.id) is u2
    assert convo.can_regenerate(failed.id) is True
    assert convo.regenerate_point(failed.id) is u2
    failed.failed, failed.content = False, "real answer"
    assert convo.can_regenerate(u2.id) is False


def test_retry_user_after_stop_before_first_token() -> None:
    convo, (u1,) = _chat("q1")
    blank = convo.add_assistant("m1")
    blank.status = "stopped"
    assert convo.can_regenerate(u1.id) is True


def test_regenerate_empty_and_unknown() -> None:
    convo = Conversation()
    assert convo.can_regenerate("x") is False
    assert convo.regenerate_point("x") is None


# -- request messages ---------------------------------------------------------


def test_request_messages_system_blank_vs_set() -> None:
    convo, _ = _chat("q1")
    assert convo.request_messages(system="   ", keep_history=True) == [
        {"role": "user", "content": "q1"}
    ]
    assert convo.request_messages(system=None, keep_history=True) == [
        {"role": "user", "content": "q1"}
    ]
    assert convo.request_messages(system="be brief", keep_history=True)[0] == {
        "role": "system",
        "content": "be brief",
    }


def test_request_messages_history_and_single_turn() -> None:
    convo, _ = _chat("q1", "a1", "q2")
    assert convo.request_messages(system="S", keep_history=True) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    assert convo.request_messages(system="S", keep_history=False) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "q2"},
    ]


def test_request_messages_skips_failed_blank_and_reasoning() -> None:
    convo, (_u1, a1, _u2) = _chat("q1", "a1", "q2")
    a1.reasoning = "secret thoughts"
    failed = convo.add_assistant("m1")
    failed.failed, failed.content = True, "partial garbage"
    convo.add_user("q3")
    convo.add_assistant("m1")  # the streaming placeholder for q3
    sent = convo.request_messages(system=None, keep_history=True)
    assert sent == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "user", "content": "q3"},
    ]
    assert "secret thoughts" not in repr(sent)
    assert all("reasoning" not in m for m in sent)


def test_request_messages_images_become_content_parts() -> None:
    convo = Conversation()
    convo.add_user("what is this", [IMG])
    (turn,) = convo.request_messages(system=None, keep_history=True)
    assert turn["content"] == [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": IMG}},
    ]


def test_request_messages_upto_in_the_middle() -> None:
    convo, (u1, a1, u2, a2) = _chat("q1", "a1", "q2", "a2")
    convo.add_user("q3")
    assert convo.request_messages(system=None, keep_history=True, upto=u2.id) == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    # An assistant id means "the question it answered".
    assert convo.request_messages(system=None, keep_history=True, upto=a2.id)[-1] == {
        "role": "user",
        "content": "q2",
    }
    assert convo.request_messages(system=None, keep_history=False, upto=u1.id) == [
        {"role": "user", "content": "q1"}
    ]
    assert a1.content == "a1"


def test_request_messages_without_a_user_turn_raises() -> None:
    with pytest.raises(ValueError):
        Conversation().request_messages(system="S", keep_history=True)
    convo, _ = _chat("q1")
    with pytest.raises(ValueError):
        convo.request_messages(system=None, keep_history=True, upto="missing")


# -- transcript ---------------------------------------------------------------


def test_as_markdown() -> None:
    convo = Conversation()
    convo.add_user("look", [IMG, IMG])
    reply = convo.add_assistant("qwen3")
    reply.content, reply.reasoning, reply.status = "It is a cat.", "hmm thinking", ""
    convo.add_user("and this?", [IMG])
    failed = convo.add_assistant(None)
    failed.failed, failed.status = True, "error: HTTP 500"
    convo.add_assistant(None)  # blank placeholder is skipped
    text = convo.as_markdown()
    assert text == (
        "**You:**\n\nlook\n\n[2 images]"
        "\n\n---\n\n**qwen3:**\n\nIt is a cat."
        "\n\n---\n\n**You:**\n\nand this?\n\n[1 image]"
        "\n\n---\n\n**Assistant:**\n\n_(failed: error: HTTP 500)_"
    )
    assert "hmm thinking" not in text
    assert Conversation().as_markdown() == ""


# -- render_markdown: safety --------------------------------------------------


@pytest.mark.parametrize(
    ("source", "forbidden"),
    [
        ("<script>alert(1)</script>", "<script"),
        ("<img src=x onerror=alert(1)>", "<img"),
        ("[x](javascript:alert(1))", "javascript:"),
        ("[x](JaVaScRiPt:alert(1))", "avascript"),
        ("[d](data:text/html,<b>x</b>)", "data:"),
        ("![pic](javascript:alert(1))", "javascript:"),
        ('<a href="javascript:alert(1)">x</a>', "<a "),
        ("<iframe src=https://evil></iframe>", "<iframe"),
        ('[t](https://a.com "x\\" onmouseover=\\"alert(1)")', 'onmouseover="'),
    ],
)
def test_render_markdown_never_emits_active_html(source: str, forbidden: str) -> None:
    out = render_markdown(source)
    assert forbidden not in out.lower()


def test_render_markdown_escapes_raw_html() -> None:
    out = render_markdown("raw <b>bold</b> & <script>x</script>")
    assert "&lt;b&gt;bold&lt;/b&gt;" in out
    assert "&lt;script&gt;" in out
    assert "<b>" not in out


def test_render_markdown_unsafe_link_keeps_text() -> None:
    out = render_markdown("see [here](javascript:alert(1)) now")
    assert "here" in out and "<a" not in out


def test_render_markdown_safe_links_open_in_new_tab() -> None:
    out = render_markdown("[site](https://example.com/?a=1&b=2) and [mail](mailto:a@b.c)")
    assert 'href="https://example.com/?a=1&amp;b=2"' in out
    assert 'href="mailto:a@b.c"' in out
    assert out.count('target="_blank"') == 2
    assert out.count('rel="noopener noreferrer nofollow"') == 2


def test_render_markdown_http_image_allowed() -> None:
    out = render_markdown("![cat](https://example.com/cat.png)")
    assert '<img src="https://example.com/cat.png" alt="cat"' in out
    assert 'referrerpolicy="no-referrer"' in out


# -- render_markdown: formatting ----------------------------------------------


def test_render_markdown_fenced_code_escapes_content() -> None:
    out = render_markdown("```python\nprint('<hi>')\n```")
    assert "<pre>" in out and "<code>" in out
    assert "&lt;hi&gt;" in out and "<hi>" not in out


def test_render_markdown_plain_fence() -> None:
    out = render_markdown("```\nline1\nline2\n```")
    assert out == "<pre><code>line1\nline2\n</code></pre>"


def test_render_markdown_table() -> None:
    out = render_markdown("| a | b |\n|:--|--:|\n| 1 | 2 |")
    assert "<table>" in out and "<th" in out and "<td" in out
    assert 'style="text-align:left;"' in out


def test_render_markdown_unclosed_fence_mid_stream() -> None:
    out = render_markdown("Here:\n```python\ndef f():\n    return 1")
    assert "<pre>" in out and "```" not in out
    assert "def" in out


def test_close_open_fence() -> None:
    assert close_open_fence("no code") == "no code"
    assert close_open_fence("```\nx\n```") == "```\nx\n```"
    assert close_open_fence("```py\nx") == "```py\nx\n```"
    assert close_open_fence("````\nx\n```\n") == "````\nx\n```\n````"
    # An info string never closes a block.
    assert close_open_fence("```\nx\n```py\n").endswith("```py\n```")


def test_render_markdown_single_newline_breaks() -> None:
    assert "<br />" in render_markdown("line1\nline2")


def test_render_markdown_snake_case_not_italic() -> None:
    out = render_markdown("call my_long_function_name now")
    assert "<em>" not in out and "my_long_function_name" in out
    assert "<em>x</em>" in render_markdown("*x*")


def test_render_markdown_lists_strike_and_blank() -> None:
    out = render_markdown("Steps:\n- one\n- two\n\n~~old~~")
    assert "<ul>" in out and "<li>one</li>" in out and "<s>old</s>" in out
    assert render_markdown("") == ""


def test_render_markdown_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import studioforge.gui.chat_conversation as cc

    def boom(*_args: object, **_kwargs: object) -> str:
        raise RuntimeError("renderer bug")

    monkeypatch.setattr(cc.markdown2, "markdown", boom)
    assert cc.render_markdown("<b>x</b>") == "<pre>&lt;b&gt;x&lt;/b&gt;</pre>"


# -- plain_text ---------------------------------------------------------------


def test_plain_text_basics() -> None:
    source = (
        "# Title\n"
        "Some **bold**, *italic*, ~~gone~~ and `snake_case` text.\n"
        "* first\n"
        "+ second\n"
        "> quoted\n"
        "See [docs](https://example.com) or <https://a.b>.\n"
        "```python\n"
        "x = **not_bold**\n"
        "```"
    )
    assert plain_text(source) == (
        "Title\n"
        "Some bold, italic, gone and snake_case text.\n"
        "- first\n"
        "- second\n"
        "quoted\n"
        "See docs (https://example.com) or https://a.b.\n"
        "x = **not_bold**"
    )


def test_plain_text_keeps_snake_case_and_tables() -> None:
    assert plain_text("my_var_name stays") == "my_var_name stays"
    assert plain_text("| a | b |\n|---|---|\n| 1 | 2 |") == "| a | b |\n| 1 | 2 |"
    assert plain_text("") == ""


# -- review regressions -------------------------------------------------------


def test_regenerate_never_drops_a_real_answer_between_question_and_reply() -> None:
    # u1's second answer is left after its question was deleted; regenerating
    # the last reply truncates after u1 and would silently take a1 with it.
    convo, (u1, a1, u2, a2) = _chat("q1", "a1", "q2", "a2")
    convo.delete(u2.id)
    assert convo.can_regenerate(a2.id) is False
    assert convo.regenerate_point(a2.id) is None
    # A failed or blank reply in between is what a regenerate replaces.
    a1.failed = True
    assert convo.can_regenerate(a2.id) is True
    assert convo.regenerate_point(a2.id) is u1
    a1.failed, a1.content = False, "   "
    assert convo.can_regenerate(a2.id) is True


class _Audit(HTMLParser):
    """Parses render_markdown output the way a browser would see its tags."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.problems: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in ALLOWED:
            self.problems.append(f"tag {tag}")
        for name, value in attrs:
            value = value or ""
            if name.startswith("on") or name not in ATTRS:
                self.problems.append(f"attr {tag}.{name}")
            if name == "href" and not re.match(r"(https?|mailto):", value, re.I):
                self.problems.append(f"href {value!r}")
            if name == "src" and not re.match(r"https?:", value, re.I):
                self.problems.append(f"src {value!r}")
            if name == "style" and not re.fullmatch(r"text-align:(left|right|center);?", value):
                self.problems.append(f"style {value!r}")


ALLOWED = {
    "p", "br", "hr", "strong", "b", "em", "i", "s", "del", "code", "pre", "div", "span",
    "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    "table", "thead", "tbody", "tr", "th", "td", "a", "img",
}  # fmt: skip
ATTRS = {"href", "title", "target", "rel", "src", "alt", "loading", "referrerpolicy", "class",
         "style"}  # fmt: skip

ATTACKS = [
    "[x](jav&#x61;script:alert(1))",
    "[x](jav&#97;script:alert(1))",
    "[x](&#106;avascript:alert(1))",
    '<a href="jav&#x61;script:alert(1)">x</a>',
    "[x](data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==)",
    "![x](data:image/svg+xml,<svg onload=alert(1)>)",
    "[x](vbscript:msgbox(1))",
    "[x]( javascript:alert(1))",
    "[x](java\tscript:alert(1))",
    "[x](java\nscript:alert(1))",
    "[x](\x01javascript:alert(1))",
    "[x](<javascript:alert(1)>)",
    "[x](//evil.example/x)",
    "<svg onload=alert(1)>",
    "<svg><script>alert(1)</script></svg>",
    "<style>body{display:none}</style>",
    "<math><mi xlink:href=javascript:alert(1)>x</mi></math>",
    '[t](https://a.com "a\\" onmouseover=\\"x")',
    "[t](https://a.com 'a\" onmouseover=\"x')",
    '![a" onerror="alert(1)](https://a.com/x.png)',
    "![a' onerror=alert(1) x='](https://a.com/x.png)",
    "[r][1]\n\n[1]: javascript:alert(1)",
    '[r][1]\n\n[1]: https://ok.com "t\\" onclick=\\"x"',
    "![r][1]\n\n[1]: javascript:alert(1)",
    "<javascript:alert(1)>",
    '<http://a.com/"onmouseover="alert(1)>',
    "&lt;script&gt;alert(1)&lt;/script&gt;",
    "&#60;img src=x onerror=alert(1)&#62;",
    "&#x3C;script&#x3E;alert(1)&#x3C;/script&#x3E;",
    "```html\n<script>alert(1)</script>\n```",
    "```python\n'</code></pre><img src=x onerror=alert(1)>'\n```",
    '```x" onclick="alert(1)\nhi\n```',
    "```\n<script>alert(1)</script>",  # unclosed fence plus HTML
    "text\n```\n<img src=x onerror=alert(1)>\n",
    "`</code><script>alert(1)</script>`",
    "<b><i>nested</b></i><div><span>unclosed",
    '<a href="https://x"><a href="javascript:1">y</a></a>',
    "[a [b](javascript:1)](https://ok)",
    "[![img](javascript:1)](https://ok)",
    "[![img](https://ok/i.png)](javascript:1)",
    "**<img src=x onerror=1>**\n__<x>__",
    "<!-- --><script>x</script> <![CDATA[<script>]]> <?php x ?> <!doctype html>",
    "| a |\n|---|\n| <script>alert(1)</script> |",
    "| a |\n|:--|\n| [x](javascript:1) |",
    '<div markdown="1">*x*</div>',
    "x\n<details open ontoggle=alert(1)>",
    "    <script>indented</script>",
    "[x](https://ok.com/</a><script>alert(1)</script>)",
    "line1\n<script>\nline2\n</script>",
    "a_<script>_b *<img src=x onerror=1>*",
]


@pytest.mark.parametrize("source", ATTACKS, ids=[f"attack{i}" for i in range(len(ATTACKS))])
def test_render_markdown_output_passes_a_browser_eye_audit(source: str) -> None:
    out = render_markdown(source)
    audit = _Audit()
    audit.feed(out)
    audit.close()
    assert audit.problems == [], (source, out)


def test_render_markdown_fuzz_never_emits_active_html() -> None:
    rng = random.Random(20260929)
    alphabet = list("<>\"'`&#;:=/()[]!*_~|-\n ") + ["javascript:", "onerror", "script", "a", "x"]
    for _ in range(300):
        source = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 120)))
        out = render_markdown(source)
        audit = _Audit()
        audit.feed(out)
        audit.close()
        assert audit.problems == [], (source, out)


@pytest.mark.parametrize(
    "source",
    ["\ud800 lone surrogate", "nul \x00 byte <b>", "\x1b[31mred", ">" * 3000 + " deep",
     "".join("  " * i + "- x\n" for i in range(200)), "\r\n\r\n", "```", "````\n```"],
    ids=["surrogate", "nul", "ansi", "deep-quote", "deep-list", "crlf", "fence", "fence4"],
)  # fmt: skip
def test_render_markdown_odd_input_returns_safe_html(source: str) -> None:
    out = render_markdown(source)
    assert isinstance(out, str)
    assert "<b>" not in out and "<script" not in out


def test_plain_text_is_not_quadratic_on_pathological_lines() -> None:
    # Unbounded, these lines took minutes (the event loop is the gateway's).
    for line in ("[a](" * 5000, "![" * 10000, "*a _b " * 4000, "~~a " * 5000):
        started = time.perf_counter()
        assert plain_text(line)
        assert time.perf_counter() - started < 2.0, line[:12]


def test_plain_text_links_with_titles_and_images() -> None:
    source = '[docs](https://e.com/x "Title") and ![cat](https://e.com/c.png "t") ok'
    assert plain_text(source) == "docs (https://e.com/x) and cat ok"
