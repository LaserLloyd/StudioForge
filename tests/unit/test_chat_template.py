"""D82: reading, diagnosing and overriding a model's chat template."""

from __future__ import annotations

from pathlib import Path

import pytest

from studioforge.core.chat_template import (
    MAX_TEMPLATE_BYTES,
    diagnose,
    effective_template,
    save_override,
    validate_template_text,
)
from studioforge.types import GgufMeta, ModelRecord, ModelSettings

HARMONY = "<|start|>{% for m in messages %}<|channel|>final<|message|>{{ m.content }}{% endfor %}"
ITEMS = "{% for m in messages %}{% if p['items'] %}x{% endif %}{% endfor %}"


def rec(template: str | None, arch: str = "qwen3", **settings: object) -> ModelRecord:
    return ModelRecord(
        id="vendor/Model-GGUF/Model-Q4_K_M",
        name="Model",
        path=Path("Model-Q4_K_M.gguf"),
        meta=GgufMeta(architecture=arch, chat_template=template),
        settings=ModelSettings(**settings),  # type: ignore[arg-type]
    )


def codes(found: list[dict[str, str]]) -> set[str]:
    return {item["code"] for item in found}


def test_effective_template_prefers_the_override(tmp_path: Path) -> None:
    override = tmp_path / "t.jinja"
    override.write_text("{{ messages }}", encoding="utf-8")
    assert effective_template(rec("{{ messages }}x")).source == "gguf"
    over = effective_template(rec("embedded", chat_template_file=override))
    assert (over.source, over.text, over.path) == ("override", "{{ messages }}", override)
    assert effective_template(rec(None)).source == "engine_default"


def test_an_unreadable_override_is_reported_not_raised(tmp_path: Path) -> None:
    gone = tmp_path / "gone.jinja"
    gone.write_text("{{ messages }}", encoding="utf-8")
    record = rec("embedded", chat_template_file=gone)
    gone.unlink()
    template = effective_template(record)
    assert template.source == "override" and template.text is None and template.error
    assert "override_unreadable" in codes(diagnose(record, "none"))


def test_diagnose_names_the_gpt_oss_failure_and_clears_once_fixed() -> None:
    broken = rec(HARMONY, arch="gpt-oss", reasoning_format="none")
    found = diagnose(broken, "none")
    assert "harmony_reasoning_none" in codes(found)
    assert all({"code", "severity", "message", "fix"} <= set(item) for item in found)
    # The D82 default (no setting -> deepseek) is not a problem.
    assert "harmony_reasoning_none" not in codes(diagnose(rec(HARMONY, arch="gpt-oss"), "none"))
    assert "harmony_reasoning_none" not in codes(
        diagnose(rec(HARMONY, arch="gpt-oss", reasoning_format="deepseek"), "none")
    )


def test_diagnose_flags_items_subscripts_and_a_missing_template() -> None:
    assert "items_subscript" in codes(diagnose(rec(ITEMS), "none"))
    assert "items_subscript" in codes(diagnose(rec('{{ messages }}{{ x["items"] }}'), "none"))
    assert "no_template" in codes(diagnose(rec(None), "none"))
    assert diagnose(rec("{% for m in messages %}{{ m.content }}{% endfor %}"), "none") == []


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        pytest.param("", "non-empty", id="empty"),
        pytest.param("   ", "non-empty", id="blank"),
        pytest.param(None, "non-empty", id="none"),
        pytest.param("{{ prompt }}", "messages", id="no-messages"),
        pytest.param("{% for m in messages %}", "does not parse", id="syntax"),
        pytest.param("{{ messages }}\x00", "NUL", id="nul"),
        pytest.param("{{ messages }}" + "x" * MAX_TEMPLATE_BYTES, "limit", id="too-big"),
    ],
)
def test_validate_refuses_what_cannot_be_a_template(text: object, fragment: str) -> None:
    with pytest.raises(ValueError, match=fragment):
        validate_template_text(text)


def test_validate_accepts_the_engine_dialect_it_cannot_run() -> None:
    """Parse only: raise_exception and other engine globals are fine at save time."""
    text = "{% if not messages %}{{ raise_exception('no messages') }}{% endif %}{{ messages }}"
    assert validate_template_text(text) == text


def test_save_override_is_content_addressed(tmp_path: Path) -> None:
    first = save_override(tmp_path, "vendor/Model-GGUF/Model-Q4_K_M", "{{ messages }}")
    again = save_override(tmp_path, "vendor/Model-GGUF/Model-Q4_K_M", "{{ messages }}")
    other = save_override(tmp_path, "vendor/Model-GGUF/Model-Q4_K_M", "{{ messages }} v2")
    assert first == again != other
    assert first.parent == tmp_path / "templates"
    assert first.name.startswith("vendor_Model-GGUF_Model-Q4_K_M-") and first.suffix == ".jinja"
    assert first.read_text(encoding="utf-8") == "{{ messages }}"
    assert not list((tmp_path / "templates").glob("*.tmp"))
