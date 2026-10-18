"""The startup banner prints the (deprecated) MCP PIN only to a real terminal.

Review 2026-10-18: the banner wrote ``PIN: 13831848`` to stdout unconditionally,
so in a container it landed in ``docker logs`` (and under systemd in the
journal) for the life of the container -- a credential persisted in a log, which
the structured log line beside it carefully redacts. A human at a terminal still
sees it; a log sink gets a pointer to where it lives instead.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any

import pytest

from studioforge.__main__ import _log_mcp_banner
from studioforge.config import Config

PIN = "12345678"


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _config(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        server={"port": 11234},
        gui={"enabled": False, "port": 18080},
        watchdog={"enabled": False, "port": 11235},
        mcp={"pin": PIN, "pin_required": True, "enabled": True, "advertise": True},
    )


@pytest.mark.parametrize(("tty", "shown"), [(True, True), (False, False)])
def test_the_pin_reaches_a_terminal_but_not_a_log_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tty: bool, shown: bool
) -> None:
    stream: Any = _Stream(tty)
    monkeypatch.setattr(sys, "stdout", stream)

    _log_mcp_banner(_config(tmp_path))

    text = stream.getvalue()
    assert "MCP control plane" in text
    assert (PIN in text) is shown
    if not shown:
        assert "not shown here" in text and "/api/mcp/info" in text
    assert "keys create --scope admin" in text, "the scoped-key hint is always there"
