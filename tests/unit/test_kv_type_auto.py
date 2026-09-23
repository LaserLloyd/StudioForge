"""``auto`` is sized as f16 without a warning; a genuinely unknown type warns once (D69 §14).

``models.default_kv_cache_type: auto`` asks the planner to walk the KV quality
ladder, and ``Planner._kv_options`` fans it out. A caller that sizes it
directly (the download-fit preview) reached ``kv_bytes_per_element("auto")``
and logged ``unknown kv cache type, assuming f16`` in bursts of 3-5 lines, 54
times between 2026-09-13 and 09-22. The number was right (f16 is the ladder's
first and most expensive rung); the warning was not.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from studioforge import logging as sf_logging
from studioforge.core import planner as planner_module
from studioforge.core.planner import KV_BYTES_PER_ELEMENT, kv_alloc_bytes, kv_bytes_per_element
from tests.unit.test_planner import make_meta


class _LevelRecorder:
    def __init__(self) -> None:
        self.lines: list[tuple[str, str, dict[str, Any]]] = []

    def __getattr__(self, level: str) -> Any:
        return lambda event, *_a, **kw: self.lines.append((level, event, kw))

    def at(self, level: str) -> list[dict[str, Any]]:
        return [kw for lvl, _event, kw in self.lines if lvl == level]


@pytest.fixture()
def recorder(monkeypatch: pytest.MonkeyPatch) -> Iterator[_LevelRecorder]:
    rec = _LevelRecorder()
    monkeypatch.setattr(planner_module, "log", rec)
    sf_logging.reset_first_time()
    yield rec
    sf_logging.reset_first_time()


def test_auto_is_sized_as_f16_and_says_nothing(recorder: _LevelRecorder) -> None:
    assert kv_bytes_per_element("auto") == KV_BYTES_PER_ELEMENT["f16"]
    meta = make_meta()
    as_auto = kv_alloc_bytes(meta, ctx_total=8192, kv_k="auto", kv_v="auto")
    as_f16 = kv_alloc_bytes(meta, ctx_total=8192, kv_k="f16", kv_v="f16")
    assert as_auto == as_f16 > 0
    assert recorder.lines == []


def test_an_unknown_type_warns_once_then_drops_to_debug(recorder: _LevelRecorder) -> None:
    for _ in range(5):
        assert kv_bytes_per_element("q3_k_nonsense") == 2.0

    assert recorder.at("warning") == [{"kv_cache_type": "q3_k_nonsense"}]
    assert len(recorder.at("debug")) == 4
    # A second unknown type is its own first time.
    kv_bytes_per_element("iq9_future")
    assert [kw["kv_cache_type"] for kw in recorder.at("warning")] == ["q3_k_nonsense", "iq9_future"]


def test_first_time_is_true_once_per_key_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    sf_logging.reset_first_time()
    try:
        assert sf_logging.first_time("reasoning_format", "a/model") is True
        assert sf_logging.first_time("reasoning_format", "a/model") is False
        assert sf_logging.first_time("reasoning_format", "b/model") is True
        assert sf_logging.first_time("kv_cache_type", "a/model") is True

        monkeypatch.setattr(sf_logging, "_FIRST_TIME_CAP", 3)
        # The cap is reached: the set starts over rather than growing.
        assert sf_logging.first_time("x") is True
        assert len(sf_logging._first_time_seen) == 1
        assert sf_logging.first_time("reasoning_format", "a/model") is True
    finally:
        sf_logging.reset_first_time()
