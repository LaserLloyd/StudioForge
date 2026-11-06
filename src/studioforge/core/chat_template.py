"""A model's chat template: the one a launch uses, its known problems, an override (D82).

The GGUF's embedded template is right almost always, and "almost" is the
problem: a template the engine renders differently from Python Jinja, a
template whose output the engine's parser cannot read back, a reasoning format
that turns the model's own markup into the next turn's history. Until D82 the
only way out was a ``chat_template_file`` path typed into the settings of a
box you were sitting at. The MCP tools ``get_model_template`` /
``set_model_template`` (management.py) make it something an agent can diagnose
and fix: they read the template a launch would use, run the known checks
below, and write an override into ``<data_dir>/templates/``.

Nothing here touches a running child: a template is read at launch, so an
override applies from the model's next load.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from studioforge.core.harmony import effective_reasoning_format, record_uses_harmony

__all__ = [
    "MAX_TEMPLATE_BYTES",
    "TemplateSource",
    "diagnose",
    "effective_template",
    "save_override",
    "validate_template_text",
]

#: An override larger than this is refused. The biggest embedded template in a
#: 23-model library is ~16 KiB (gpt-oss); 256 KiB is room for anything real and
#: a ceiling on what one MCP call can write to disk.
MAX_TEMPLATE_BYTES = 256 * 1024

_ITEMS_SUBSCRIPT = re.compile(r"""\[\s*['"]items['"]\s*\]""")


@dataclass(frozen=True)
class TemplateSource:
    """The template a launch of this model would use, and where it came from."""

    #: ``override`` (``settings.chat_template_file``), ``gguf`` (embedded), or
    #: ``engine_default`` (neither: llama.cpp falls back to its own guess).
    source: str
    text: str | None
    path: Path | None = None
    #: Why an override could not be read, when it could not.
    error: str | None = None

    @property
    def sha256(self) -> str | None:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest() if self.text else None


def effective_template(record: Any) -> TemplateSource:
    """Read the template a launch of ``record`` would use."""
    path = getattr(record.settings, "chat_template_file", None)
    if path is not None:
        try:
            return TemplateSource("override", Path(path).read_text(encoding="utf-8"), Path(path))
        except (OSError, UnicodeDecodeError) as exc:
            return TemplateSource("override", None, Path(path), error=str(exc))
    meta = getattr(record, "meta", None)
    text = getattr(meta, "chat_template", None) if meta is not None else None
    if text:
        return TemplateSource("gguf", text)
    return TemplateSource("engine_default", None)


def diagnose(record: Any, default_reasoning_format: str | None) -> list[dict[str, str]]:
    """The known template problems for ``record``, each ``{code, severity, message, fix}``.

    Static checks only -- nothing is rendered. Each one is a failure seen on
    this rig, not a style rule.
    """
    found: list[dict[str, str]] = []
    template = effective_template(record)
    harmony = record_uses_harmony(record)
    reasoning, _source = effective_reasoning_format(
        record.settings.reasoning_format, default_reasoning_format, harmony=harmony
    )

    def add(code: str, severity: str, message: str, fix: str) -> None:
        found.append({"code": code, "severity": severity, "message": message, "fix": fix})

    if template.error is not None:
        add(
            "override_unreadable",
            "error",
            f"chat_template_file {template.path} cannot be read ({template.error}); the next "
            "load will fail.",
            "set_model_template(model_id, template=...) with the text, or reset=true.",
        )
    if template.source == "engine_default":
        add(
            "no_template",
            "warning",
            "The GGUF carries no chat template, so the engine guesses one; tool calls and "
            "reasoning parsing are unlikely to work.",
            "set_model_template(model_id, template=...) with the model's published template.",
        )
    if harmony and reasoning == "none":
        add(
            "harmony_reasoning_none",
            "error",
            "Harmony model (gpt-oss) with reasoning_format none: replies carry raw channel "
            "markup in content, clients send it back as history, and a few turns in the "
            "engine's gpt-oss parser answers HTTP 500 'does not match the expected "
            "peg-native format'. StudioForge strips the markup from history on the way in "
            "(D82), but the setting is the real fix.",
            "set_model_template(model_id, reasoning_format='deepseek'), or 'default' "
            "(harmony models default to deepseek since D82).",
        )
    if template.text and _ITEMS_SUBSCRIPT.search(template.text):
        add(
            "items_subscript",
            "info",
            "The template indexes param_spec['items']. For an array tool parameter with no "
            "'items', Jinja returns the dict's built-in items() method instead and the "
            "render fails ('Function is not a bool value'). Requests through StudioForge "
            "are safe: it gives such parameters items: {} (D82).",
            "None needed through StudioForge; a client calling the engine directly should "
            "always send 'items' for array parameters.",
        )
    return found


def validate_template_text(text: Any) -> str:
    """Check an override before it is written; returns the text, raises ``ValueError``.

    Size and shape only, plus a Jinja *parse* when Python Jinja is installed.
    The engine has its own Jinja implementation, so a parse that passes here
    can still fail there -- the load reports that -- but a syntax error is
    caught before anything is saved.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("template must be a non-empty string")
    size = len(text.encode("utf-8"))
    if size > MAX_TEMPLATE_BYTES:
        raise ValueError(f"template is {size} bytes; the limit is {MAX_TEMPLATE_BYTES}")
    if "\x00" in text:
        raise ValueError("template contains a NUL byte")
    if "messages" not in text:
        raise ValueError("this does not look like a chat template: it never reads 'messages'")
    try:
        import jinja2  # noqa: PLC0415 - optional: only a syntax check
    except ImportError:  # pragma: no cover - jinja2 ships with the GUI stack
        return text
    try:
        jinja2.Environment().parse(text)
    except jinja2.TemplateSyntaxError as exc:
        raise ValueError(f"template does not parse: line {exc.lineno}: {exc.message}") from exc
    return text


def save_override(data_dir: Path, model_id: str, text: str) -> Path:
    """Write ``text`` as ``<data_dir>/templates/<model>-<sha8>.jinja`` and return the path.

    Content-addressed, so saving the same text twice is one file, and an older
    override is never overwritten in place (a load that is reading it keeps a
    consistent file; the previous text stays on disk to go back to).
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")[:120] or "model"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    folder = Path(data_dir) / "templates"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{slug}-{digest}.jinja"
    if not path.exists():
        tmp = path.with_suffix(".jinja.tmp")
        tmp.write_text(text, encoding="utf-8", newline="")
        tmp.replace(path)
    return path
