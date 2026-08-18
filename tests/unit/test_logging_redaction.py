"""Secret redaction is a log processor, not a call-site discipline (D6).

The processor must cover every shape a secret can travel in: bare strings,
nested dicts, *lists* (an argv is a list of strings, and launch command lines
are exactly what carries tokens), and the exception text that
``format_exc_info`` inserts into the event dict.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import structlog

from studioforge import logging as sf_logging
from studioforge.logging import _redact, configure_logging, register_secret

SECRET = "hf_secret_token_abc123"


@pytest.fixture(autouse=True)
def _isolated_secrets() -> Iterator[None]:
    saved = set(sf_logging._secret_values)
    sf_logging._secret_values.clear()
    register_secret(SECRET)
    try:
        yield
    finally:
        sf_logging._secret_values.clear()
        sf_logging._secret_values.update(saved)


def run_redact(event: dict[str, Any]) -> dict[str, Any]:
    return dict(_redact(None, "info", event))


def test_secret_in_a_list_value_is_scrubbed() -> None:
    """argv-shaped values: log.info("launch", argv=[...]) must not leak."""
    out = run_redact({"event": "launch", "argv": ["--hf-token", SECRET, "--x"]})
    assert SECRET not in repr(out["argv"]), "a secret inside a list leaked verbatim"


def test_secret_in_a_tuple_value_is_scrubbed() -> None:
    out = run_redact({"event": "launch", "cmd": ("run", f"token={SECRET}")})
    assert SECRET not in repr(out["cmd"])


def test_secret_in_a_list_nested_inside_a_dict_is_scrubbed() -> None:
    """details={"argv": [...]} -- the ModelLoadError shape."""
    out = run_redact({"event": "failed", "details": {"argv": ["--key", SECRET]}})
    assert SECRET not in repr(out["details"])


def test_plain_string_and_dict_scrubbing_still_work() -> None:
    out = run_redact(
        {
            "event": f"calling with {SECRET}",
            "nested": {"msg": f"token {SECRET} used"},
            "api_key": "whatever",
        }
    )
    assert SECRET not in out["event"]
    assert SECRET not in out["nested"]["msg"]
    assert out["api_key"] == "***REDACTED***"


def test_redaction_runs_after_exception_formatting() -> None:
    """Tracebacks must be scrubbed too.

    ``format_exc_info`` inserts the exception text into the event dict; if
    redaction runs before it, any secret inside an exception message (an httpx
    Request repr, a subprocess error carrying an argv) reaches stderr, the log
    file and the ring buffer verbatim.
    """
    configure_logging("INFO")
    processors = structlog.get_config()["processors"]
    redact_index = processors.index(_redact)
    exc_index = processors.index(structlog.processors.format_exc_info)
    assert redact_index > exc_index, (
        "redaction runs before format_exc_info: exception text bypasses scrubbing"
    )

    # End to end: run the pre-renderer chain over an event carrying exc_info.
    try:
        raise RuntimeError(f"upstream rejected token {SECRET}")
    except RuntimeError:
        import sys
        import types

        event: Any = {"event": "boom", "exc_info": sys.exc_info()}
    fake_logger = types.SimpleNamespace(name="test")
    for processor in processors[:-1]:  # everything except the final renderer
        event = processor(fake_logger, "error", event)
    assert SECRET not in str(event), "the exception text leaked the secret"


def test_mcp_pin_is_registered_as_a_secret(tmp_path: Any) -> None:
    """The PIN grants the management tools; it must scrub from log strings."""
    from studioforge.config import load_config

    config = load_config(tmp_path / "config.yaml", create=True)
    assert config.mcp.pin, "a pin should have been minted"
    assert config.mcp.pin in sf_logging._secret_values, (
        "the MCP PIN was never registered for scrubbing"
    )
