"""Can the llama.cpp build that would serve a model load its architecture? (D66)

The question used to be answered by spawning ``llama-server`` and reading its
last words. On 2026-09-20..22 a K2-Horizon GGUF (``general.architecture =
'k2-horizon'``, which no llama.cpp release includes) was asked for three times
by CrucibleForge: each time it leased two cards for a benchmark (a lease grant
unloads the idle residents on them), planned a placement, spawned a child that
died 0.3 s into startup with ``unknown model architecture: 'k2-horizon'``, and
answered ``502 model_load_failed`` -- a code that reads like a fault to report,
not a fact to remember. Nothing remembered it.

Two sources of truth, both one-sided:

* **the build's own library** (``source: "binary"``,
  :func:`studioforge.core.engine.architecture_table`) -- a name absent from it
  is certain to fail, so every load path refuses before anything is planned,
  held, leased, evicted or spawned;
* **a startup that already failed that way** (``source: "runtime"``,
  :class:`StartupRejectionMemo`) -- the belt and braces for what the library
  cannot answer (a pre-tokenizer the build does not know, a library that could
  not be read). Keyed on the model file's path and mtime and the build's tag, and
  forgotten on an engine install/activate, a library rescan, or a restart.

Everything here answers ``None`` ("cannot tell") rather than ``False`` when in
doubt, and ``None`` never refuses anything.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from studioforge.errors import UnsupportedArchitectureError

#: How a verdict of ``False`` was reached: the build's library does not name the
#: architecture (``binary``), or a launch on that build already died of it
#: (``runtime``).
ArchSource = Literal["binary", "runtime"]

#: What the build rejected: the model architecture, or its pre-tokenizer.
RejectedKind = Literal["architecture", "pre_tokenizer"]

#: llama.cpp's own wording for the two "this build does not know this model"
#: failures, as they appear in a child's log tail. Deliberately NOT the rest of
#: ``CONFIG_ERROR_MARKERS``: "does not exist" / "no such file" is a missing
#: file, and a missing file is not an unsupported model.
_STARTUP_REJECTIONS: tuple[tuple[RejectedKind, re.Pattern[str]], ...] = (
    ("architecture", re.compile(r"unknown model architecture:\s*'([^']+)'")),
    ("pre_tokenizer", re.compile(r"unknown pre-tokenizer type:\s*'([^']+)'")),
)

#: The exact phrase each kind is reported with, quoted back to the caller.
_ENGINE_PHRASE: dict[str, str] = {
    "architecture": "unknown model architecture",
    "pre_tokenizer": "unknown pre-tokenizer type",
}


def startup_rejection(stderr_tail: Sequence[Any]) -> tuple[RejectedKind, str] | None:
    """``(kind, name)`` when a failed launch says the build does not know the model."""
    for line in stderr_tail:
        text = str(line)
        for kind, pattern in _STARTUP_REJECTIONS:
            match = pattern.search(text)
            if match:
                return kind, match.group(1)
    return None


def file_signature(path: Path | str | None) -> tuple[int, int] | None:
    """``(st_mtime_ns, st_size)`` of ``path``, or ``None`` when it cannot be read."""
    if path is None:
        return None
    try:
        info = Path(path).stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


@dataclass(frozen=True)
class ArchVerdict:
    """Whether the build that would serve ``model_id`` can load it.

    ``supported`` is ``True`` (the build's library names the architecture --
    assume it loads), ``False`` (certain not to) or ``None`` (cannot tell: no
    engine, no readable library, an unknown architecture, a stand-in
    supervisor). Only ``False`` ever refuses anything.
    """

    model_id: str
    architecture: str
    supported: bool | None
    #: The build that would serve the next load: the model's own
    #: ``settings.engine_tag`` pin, else the active engine.
    engine_tag: str | None = None
    pinned: bool = False
    source: ArchSource | None = None
    rejected_kind: RejectedKind = "architecture"
    #: The name the build rejected, when it differs from ``architecture`` or is
    #: a pre-tokenizer (runtime verdicts only).
    rejected_name: str | None = None
    #: When a runtime verdict's launch failed (epoch seconds).
    first_failed_at: float | None = None
    #: Whether a runtime rejection is remembered for later loads -- false when
    #: the rejected name may be a draft model's, or the file could not be keyed.
    remembered: bool = True
    #: For a pinned build that lacks the architecture: the active build, and
    #: whether it includes it -- the one case a StudioForge setting does fix.
    active_tag: str | None = None
    active_supports: bool | None = None

    @property
    def _build(self) -> str:
        return f"llama.cpp build {self.engine_tag}" if self.engine_tag else "this llama.cpp build"

    @property
    def _name(self) -> str:
        return self.rejected_name or self.architecture

    @property
    def _clearing_the_pin_fixes_it(self) -> bool:
        return bool(self.pinned and self.active_supports and self.active_tag)

    def note(self) -> str | None:
        """One short line for a badge, a catalog row or a log; ``None`` unless refused."""
        if self.supported is not False:
            return None
        if self.rejected_kind == "pre_tokenizer":
            return f"{self._build} does not include the pre-tokenizer '{self._name}'"
        return f"{self._build} does not include the '{self._name}' architecture"

    def remedy(self) -> str:
        """What would make the model loadable, in one sentence."""
        if self._clearing_the_pin_fixes_it:
            return (
                f"clear the model's engine_tag so it loads on the active build "
                f"{self.active_tag}, which includes it"
            )
        needs = (
            f"a llama.cpp build that supports '{self._name}'"
            if self.rejected_kind == "architecture"
            else "a llama.cpp build that supports that pre-tokenizer"
        )
        return f"use another model; this one needs {needs}, and no StudioForge setting changes that"

    def message(self) -> str:
        """The refusal, in words an operator and an agent can both act on."""
        what = (
            f"the pre-tokenizer '{self._name}'"
            if self.rejected_kind == "pre_tokenizer"
            else f"the model architecture '{self._name}'"
        )
        build = self._build + (" (this model's engine_tag pin)" if self.pinned else "")
        if self.source == "runtime":
            phrase = _ENGINE_PHRASE[self.rejected_kind]
            text = (
                f"'{self.model_id}' uses {what}, which {build} rejected at startup "
                f"('{phrase}'), so it cannot be loaded."
            )
            if self.remembered:
                text += (
                    " StudioForge will not launch it on this build again until the engine "
                    "or the model file changes."
                )
        else:
            text = (
                f"'{self.model_id}' uses {what}, which {build} does not include, so it "
                f"cannot be loaded."
            )
        if self.rejected_kind == "architecture" and self._name != self.architecture:
            text += (
                f" (The model's own architecture is '{self.architecture}'; the rejected "
                f"name may belong to a draft model launched with it.)"
            )
        if self._clearing_the_pin_fixes_it:
            text += (
                f" The active build {self.active_tag} does include it: clear the model's "
                f"engine_tag to load it there."
            )
        elif self.rejected_kind == "architecture":
            text += (
                f" No StudioForge setting changes that; it needs a llama.cpp build that "
                f"supports '{self._name}'."
            )
        else:
            text += (
                " No StudioForge setting changes that; it needs a llama.cpp build that "
                "supports that pre-tokenizer."
            )
        return text

    def details(self) -> dict[str, Any]:
        """``error.studioforge`` for the refusal."""
        out: dict[str, Any] = {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "engine_tag": self.engine_tag,
            "source": self.source,
            "first_failed_at": self.first_failed_at,
            "remedy": self.remedy(),
            "engine_tag_pinned": self.pinned,
        }
        if self.source == "runtime" or self.rejected_name:
            out["rejected"] = {"kind": self.rejected_kind, "name": self._name}
        if self.pinned:
            out["active_engine_tag"] = self.active_tag
            out["active_engine_supports"] = self.active_supports
        return out

    def error(self) -> UnsupportedArchitectureError:
        return UnsupportedArchitectureError(self.message(), param="model", details=self.details())

    def fields(self, *, with_message: bool = False) -> dict[str, Any]:
        """``arch_supported`` (+ ``arch_note`` when false) for a listing row.

        ``with_message`` adds ``arch_message`` -- the full refusal text -- for a
        surface that shows it instead of attempting the load (the GUI).
        """
        out: dict[str, Any] = {"arch_supported": self.supported}
        note = self.note()
        if note:
            out["arch_note"] = note
            if with_message:
                out["arch_message"] = self.message()
        return out


@dataclass
class StartupRejection:
    """One launch that died because the build did not know the model."""

    path: str
    mtime_ns: int
    engine_tag: str | None
    model_id: str
    kind: RejectedKind
    name: str
    architecture: str
    #: ``(st_mtime_ns, st_size)`` of the build's library when it was recorded;
    #: a reinstall over the same tag changes it.
    engine_signature: tuple[int, int] | None = None
    #: The registry's ``last_scan_at`` when it was recorded: a rescan forgets it.
    scan_marker: float | None = None
    first_failed_at: float = field(default_factory=time.time)
    failures: int = 1


class StartupRejectionMemo:
    """Launches that already died as "unknown architecture / pre-tokenizer" (D66).

    The runtime half of the preflight, keyed on ``(model path, mtime, build
    tag)``: a re-downloaded file or a different build is a different key. An
    entry also lapses when the build's library is reinstalled (its signature
    moves) or the library is rescanned, and the whole memo is cleared on an
    engine install or activation. In memory only: a restart forgets it, and
    the library probe answers again at the next load.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, int, str | None], StartupRejection] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, entry: StartupRejection) -> StartupRejection:
        """Remember ``entry``; a repeat keeps the first failure's timestamp."""
        key = (entry.path, entry.mtime_ns, entry.engine_tag)
        previous = self._entries.get(key)
        if previous is not None:
            entry.first_failed_at = previous.first_failed_at
            entry.failures = previous.failures + 1
        self._entries[key] = entry
        return entry

    def lookup(
        self,
        path: Path | str,
        engine_tag: str | None,
        *,
        engine_signature: tuple[int, int] | None,
        scan_marker: float | None,
    ) -> StartupRejection | None:
        """The standing rejection for this file and build, or ``None``.

        Stats ``path`` only when an entry for it exists, so a listing pays
        nothing for the models that never failed.
        """
        text = str(path)
        if not any(key[0] == text and key[2] == engine_tag for key in self._entries):
            return None
        signature = file_signature(path)
        if signature is None:
            return None
        key = (text, signature[0], engine_tag)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.engine_signature != engine_signature or entry.scan_marker != scan_marker:
            self._entries.pop(key, None)
            return None
        return entry

    def clear(self) -> int:
        """Forget everything; returns how many entries there were."""
        count = len(self._entries)
        self._entries.clear()
        return count

    def snapshot(self) -> list[dict[str, Any]]:
        return [
            {
                "model_id": e.model_id,
                "engine_tag": e.engine_tag,
                "kind": e.kind,
                "name": e.name,
                "first_failed_at": e.first_failed_at,
                "failures": e.failures,
            }
            for e in self._entries.values()
        ]
