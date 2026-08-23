"""HuggingFace GGUF discovery: search repos, enumerate files, group them.

Why the plain HTTP API rather than ``huggingface_hub``'s helpers: every
``HfApi`` method is synchronous and does blocking network I/O, so calling one
from the event loop would stall the whole gateway (model loads, in-flight
completions, the GUI's WebSocket) for the duration of the request. The two JSON
endpoints we need are trivial, so they are called with :mod:`httpx` directly.
``huggingface_hub`` is still used where it is genuinely better: :func:`hf_hub_url`
is a pure, no-I/O function that knows the ``resolve/<revision>`` URL layout and
the correct percent-encoding, and it honours ``HF_ENDPOINT`` the same way the
rest of the HF ecosystem does.

The unit that matters downstream is not "a file" but a **logical download**: a
base GGUF -- possibly split into ``-00001-of-0000N`` shards -- plus the vision
projector it needs to be useful. Those files must arrive together or the model
is unloadable, so :meth:`GgufRepoInfo.logical_models` collapses them into one
selectable entry sharing a single ``group_id``.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import email.utils
import os
import re
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

import httpx
from huggingface_hub import hf_hub_url

from studioforge.config import Config
from studioforge.core.gguf import (
    looks_like_auxiliary_gguf,
    looks_like_mmproj,
    quant_label_from_filename,
)
from studioforge.errors import BadRequestError, UpstreamError
from studioforge.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "DATE_FIELDS",
    "DEFAULT_HF_ENDPOINT",
    "SORT_KEYS",
    "UNKNOWN_QUANT",
    "GgufFileInfo",
    "GgufRepoInfo",
    "HfSearch",
    "LogicalDownload",
    "age_days",
    "file_url",
    "parse_quant",
    "safe_filename",
    "shard_parts",
]

DEFAULT_HF_ENDPOINT: Final = "https://huggingface.co"

#: User-facing sort name -> the value HF's ``/api/models`` wants in ``?sort=``.
#:
#: Kept as an explicit whitelist rather than passing the caller's string
#: through, because HF answers an unknown ``sort`` with an unsorted 200 rather
#: than a 4xx: a typo would silently return arbitrary repos that *look* like a
#: valid result. Validating here turns that into a message naming the five
#: options.
#:
#: ``downloads`` is HF's **trailing-30-day** download count, not an all-time
#: total -- the same number the model page shows as "Downloads last month". A
#: repo published last week can therefore out-rank a two-year-old classic, which
#: is usually what a model shopper wants but is worth saying out loud in the UI.
SORT_KEYS: Final[dict[str, str]] = {
    "downloads": "downloads",
    "likes": "likes",
    "updated": "lastModified",
    "created": "createdAt",
    "trending": "trendingScore",
}

#: Which timestamp a ``newer_than_days`` window is measured against. "updated"
#: catches repos that were re-quantised or had files added; "created" is the
#: only way to ask for genuinely new repos, since popular repos are touched
#: constantly and would otherwise all look brand new.
DATE_FIELDS: Final[tuple[str, ...]] = ("updated", "created")

#: Page size for the date-window walk. HF caps ``limit`` at 100 on this
#: endpoint, and a bigger page means fewer round trips before the cutoff.
_WINDOW_PAGE_SIZE: Final = 100

#: Hard bound on pages fetched for one windowed search. A wide window over a
#: busy filter (``gguf`` alone sees hundreds of repos touched per hour) would
#: otherwise walk for thousands of entries while the user watches a spinner.
#: Hitting the cap is logged as ``truncated`` rather than passed off as a
#: complete answer.
_MAX_WINDOW_PAGES: Final = 3

#: Returned instead of a guess when the filename carries no recognisable quant
#: token. A wrong label would mis-steer the planner's quant/hardware affinity,
#: so "unknown" is the honest answer.
UNKNOWN_QUANT: Final = "unknown"

_DEFAULT_TIMEOUT_S: Final = 30.0
_MAX_RATE_LIMIT_RETRIES: Final = 3
_MAX_RETRY_SLEEP_S: Final = 30.0

_SHA256_RE: Final = re.compile(r"\A[0-9a-f]{64}\Z")
# llama.cpp's own split naming: "<base>-00001-of-00005.gguf".
_SHARD_RE: Final = re.compile(r"\A(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf\Z", re.I)
_SLUG_RE: Final = re.compile(r"[^a-z0-9]+")
# RFC 8288 Link header entry: `<https://host/path?cursor=..>; rel="next"`.
# HF quotes the rel value, but the RFC allows it bare, so both are accepted.
_LINK_RE: Final = re.compile(r"<(?P<url>[^>]*)>\s*;\s*[^,]*\brel\s*=\s*\"?next\"?", re.I)

#: Projector precision preference when a repo ships several mmproj files and
#: none matches the base file's quant. f16/bf16 projectors are the ones vision
#: models are actually validated against upstream; f32 only doubles VRAM for no
#: measurable quality gain, and a quantised projector can visibly degrade OCR.
_MMPROJ_PRECISION_ORDER: Final = ("F16", "BF16", "F32")

# Reserved DOS device names. Windows resolves these to devices regardless of
# extension, so a repo file called "aux.gguf" would open a device handle
# instead of creating a file.
_WINDOWS_RESERVED: Final = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(1, 10)}
    | {f"lpt{n}" for n in range(1, 10)}
)


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_quant(filename: str) -> str:
    """Quantisation label parsed out of a GGUF filename.

    Delegates to :func:`studioforge.core.gguf.quant_label_from_filename`, which
    matches a candidate token against a whitelist rather than trusting the
    regex alone. That is what makes the real spread in the wild parse correctly
    -- ``-Q4_K_M``, ``.Q8_0``, ``.i1-IQ3_XXS``, ``-IQ3_M-00001-of-00002``,
    ``-UD-Q4_K_XL``, ``-NVFP4``, ``-mmproj-BF16``, ``.mmproj-f16``,
    ``mmproj-F32`` -- while ``Qwen3`` and ``Q8_0NSFW`` cannot invent a label.

    Returns :data:`UNKNOWN_QUANT` when nothing whitelisted matches.
    """
    label = quant_label_from_filename(Path(filename))
    return label or UNKNOWN_QUANT


def shard_parts(filename: str) -> tuple[int | None, int | None]:
    """``(index, total)`` from a ``-0000N-of-0000M.gguf`` suffix, else ``(None, None)``."""
    match = _SHARD_RE.match(filename)
    if match is None:
        return (None, None)
    return (int(match.group("index")), int(match.group("total")))


def shard_base(filename: str) -> str:
    """Filename with any shard suffix and the ``.gguf`` extension removed.

    Two files belong to the same split model iff they share this base, which is
    a stronger key than the quant label alone: a repo can ship two different
    models at the same quant.
    """
    match = _SHARD_RE.match(filename)
    if match is not None:
        return str(match.group("base"))
    name = filename
    if name.lower().endswith(".gguf"):
        name = name[: -len(".gguf")]
    return name


def safe_filename(name: str) -> str:
    """Validate a repo-supplied filename as a single, safe path component.

    **Security boundary.** Everything here comes from a third-party repo whose
    owner we do not trust, and the result is joined onto the user's model
    directory. A filename containing ``..``, a leading separator, or a Windows
    drive letter would let a hostile repo write anywhere the process can reach,
    so those are refused outright rather than sanitised: silently rewriting a
    hostile name produces a file the user did not ask for under a name they
    cannot correlate with the repo.

    Path separators are rejected too. GGUF repos are flat by convention, and
    flattening ``Q4_K_M/model-00001-of-00002.gguf`` to its basename risks
    silently colliding two different variants onto one destination.
    """
    if not name or not name.strip():
        raise BadRequestError("repository file name is empty", param="filename")
    if "\x00" in name:
        raise BadRequestError("repository file name contains a NUL byte", param="filename")
    if "/" in name or "\\" in name:
        raise BadRequestError(
            f"refusing repository file name {name!r}: it contains a path separator, "
            "which cannot be a single file inside the model directory",
            param="filename",
        )
    if name in {".", ".."}:
        raise BadRequestError(
            f"refusing repository file name {name!r}: relative path component",
            param="filename",
        )
    # Catches "C:x.gguf" (drive-relative) as well as "C:\\Windows\\x.gguf".
    if PureWindowsPath(name).drive or PureWindowsPath(name).is_absolute():
        raise BadRequestError(
            f"refusing repository file name {name!r}: absolute or drive-qualified path",
            param="filename",
        )
    if PurePosixPath(name).is_absolute():
        raise BadRequestError(
            f"refusing repository file name {name!r}: absolute path",
            param="filename",
        )
    if name != name.rstrip(". ") or ":" in name:
        # Win32 silently strips trailing dots and spaces, so "model.gguf " would
        # land on the user's real model.gguf (and be quarantined as wrong-size);
        # a colon names an NTFS alternate data stream on the file before it.
        raise BadRequestError(
            f"refusing repository file name {name!r}: trailing dot/space or ':' would "
            "resolve to a different file on Windows",
            param="filename",
        )
    if name.partition(".")[0].strip().lower() in _WINDOWS_RESERVED:
        raise BadRequestError(
            f"refusing repository file name {name!r}: reserved device name",
            param="filename",
        )
    return name


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", text.lower()).strip("-")


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------


def parse_hf_timestamp(iso: str | None) -> dt.datetime | None:
    """HF's ``2026-08-18T12:59:16.000Z`` as an aware UTC datetime, else ``None``.

    ``datetime.fromisoformat`` only learned to accept a trailing ``Z`` in 3.11,
    and every HF timestamp ends in one, so the suffix is rewritten to ``+00:00``
    before parsing. Anything unparseable (a mirror inventing its own format, a
    ``null``) returns ``None`` rather than raising: a missing date must degrade
    the sort, never fail the whole search.

    The result is always timezone-aware. A naive datetime here would silently
    compare wrong against the aware cutoff used by the date-window walk, which
    is the sort of bug that only shows up for users a few hours either side of
    UTC.
    """
    if not isinstance(iso, str) or not iso.strip():
        return None
    text = iso.strip()
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # HF always sends UTC; a mirror that drops the zone means UTC too.
        return parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def age_days(iso: str | None, now: float | None = None) -> float | None:
    """How many days ago ``iso`` was, or ``None`` if it cannot be parsed.

    ``now`` is a Unix timestamp, injectable so tests and the API can pin "now"
    instead of racing the clock. Fractional by design -- the GUI wants to say
    "today" for anything under a day, and rounding to whole days here would
    make a repo updated four hours ago read as "0d ago".

    Never negative: a repo whose ``lastModified`` is a few seconds in the future
    (clock skew between us and HF) clamps to 0 rather than rendering "-0d ago".
    """
    parsed = parse_hf_timestamp(iso)
    if parsed is None:
        return None
    reference = time.time() if now is None else now
    return max(0.0, (reference - parsed.timestamp()) / 86400.0)


def file_url(repo_id: str, filename: str, *, endpoint: str, revision: str = "main") -> str:
    """Resolve URL for one repo file.

    Built with :func:`huggingface_hub.hf_hub_url` -- a pure function, safe to
    call from the event loop -- so the ``resolve/<revision>`` layout and the
    percent-encoding stay in step with the rest of the HF ecosystem.
    """
    return str(hf_hub_url(repo_id=repo_id, filename=filename, revision=revision, endpoint=endpoint))


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GgufFileInfo:
    """One GGUF file in a repo, as the HF API describes it."""

    filename: str
    size_bytes: int
    quant: str
    is_mmproj: bool
    shard_index: int | None
    shard_total: int | None
    sha256: str | None
    lfs_oid: str | None
    #: A GGUF in the repo that is not a loadable model -- an MTP draft module
    #: or an imatrix calibration file. Defaults False so every existing
    #: construction site keeps its meaning.
    is_auxiliary: bool = False

    @property
    def size_known(self) -> bool:
        """Whether ``size_bytes`` is a real size rather than the 0 placeholder.

        The models *list* endpoint returns siblings without sizes; only
        ``?blobs=true`` has them. A fit estimate built on a fabricated size is
        worse than one that says "unknown", so the placeholder is flagged
        rather than filled in with a guess.
        """
        return self.size_bytes > 0


@dataclass(frozen=True)
class LogicalDownload:
    """One selectable download: a base model (possibly sharded) + optional mmproj."""

    repo_id: str
    quant: str
    files: list[GgufFileInfo]
    mmproj: GgufFileInfo | None
    total_bytes: int
    #: Empty in the common case. Set only when one repo ships several distinct
    #: models at the same quant, so ``group_id`` stays unique without changing
    #: shape for the 99% case.
    discriminator: str = ""

    @property
    def group_id(self) -> str:
        """Stable identifier tying every file of this download together.

        Stable across restarts because it is derived from the repo id and quant
        rather than generated: a resumed download must land on the same row.
        """
        base = _slug(f"{self.repo_id}:{self.quant}")
        if self.discriminator:
            return f"{base}--{_slug(self.discriminator)}"
        return base

    @property
    def publisher(self) -> str:
        """Repo owner, falling back to the model name for unscoped repos.

        The fallback keeps the three-level ``publisher/repo/file`` layout intact
        for the rare canonical repo that has no organisation prefix.
        """
        owner, _, _name = self.repo_id.partition("/")
        return owner

    @property
    def repo_name(self) -> str:
        owner, _, name = self.repo_id.partition("/")
        return name or owner

    @property
    def all_files(self) -> list[GgufFileInfo]:
        """Base shards in order, with the projector last."""
        files = list(self.files)
        if self.mmproj is not None:
            files.append(self.mmproj)
        return files

    @property
    def dest_relpath(self) -> str:
        """``publisher/repo/filename`` for the primary file (LM Studio layout)."""
        primary = self.files[0].filename if self.files else ""
        return f"{self.publisher}/{self.repo_name}/{primary}"

    @property
    def size_known(self) -> bool:
        """True only when every file in the download has a real size."""
        return bool(self.all_files) and all(f.size_known for f in self.all_files)

    @property
    def is_sharded(self) -> bool:
        return len(self.files) > 1

    @property
    def label(self) -> str:
        """Human-facing name, e.g. ``bartowski/Foo-GGUF Q4_K_M (2 parts +mmproj)``."""
        bits = [f"{self.repo_id} {self.quant}"]
        extra: list[str] = []
        if self.is_sharded:
            extra.append(f"{len(self.files)} parts")
        if self.mmproj is not None:
            extra.append("+mmproj")
        if extra:
            bits.append(f"({', '.join(extra)})")
        return " ".join(bits)


@dataclass(frozen=True)
class GgufRepoInfo:
    """A HuggingFace repo, filtered down to its GGUF files."""

    repo_id: str
    publisher: str
    name: str
    downloads: int
    likes: int
    #: HF reports ``False`` for open repos and the strings ``"auto"``/``"manual"``
    #: for the two gating flavours; both string forms mean "accept terms first".
    gated: bool | str
    private: bool
    last_modified: str | None
    files: list[GgufFileInfo]
    # The three fields below are appended *with defaults* rather than slotted in
    # next to their relatives, so every existing keyword construction (and the
    # tests' `GgufRepoInfo(...)` helpers) keeps working unchanged.
    #: HF's ``createdAt``, i.e. when the repo first appeared. Kept as the raw
    #: ISO string, like ``last_modified``, so the exact upstream value survives
    #: into the API payload and only the display layer decides on a format.
    created_at: str | None = None
    #: HF's ``trendingScore``, or ``None`` when the response did not carry one.
    #:
    #: Verified against the live API: ``/api/models?full=true`` includes this
    #: field **only when it is also the sort key**. Sorting by downloads, likes
    #: or a date omits it entirely. ``None`` therefore means "not reported",
    #: never "zero", and the difference matters -- treating an absent score as 0
    #: would rank every repo in a date-windowed search identically.
    trending_score: int | None = None

    @property
    def updated_days_ago(self) -> float | None:
        """Days since ``last_modified``, or ``None`` if HF sent no date."""
        return age_days(self.last_modified)

    @property
    def created_days_ago(self) -> float | None:
        """Days since ``created_at``, or ``None`` if HF sent no date."""
        return age_days(self.created_at)

    @property
    def quant_variants(self) -> list[str]:
        """Distinct quant labels of the loadable (non-projector) files, sorted."""
        return sorted({f.quant for f in self.files if not f.is_mmproj})

    @property
    def mmproj_files(self) -> list[GgufFileInfo]:
        return [f for f in self.files if f.is_mmproj]

    @property
    def needs_token(self) -> bool:
        return bool(self.gated) or self.private

    @property
    def sizes_known(self) -> bool:
        """False for results from :meth:`HfSearch.search` (list endpoint has no sizes)."""
        return bool(self.files) and all(f.size_known for f in self.files)

    def logical_models(self) -> list[LogicalDownload]:
        """Group the repo's files into individually selectable downloads.

        Rules, all of which exist because a partial set is unloadable:

        * every shard of one split model collapses into a single entry whose
          ``total_bytes`` is the sum of the parts;
        * a projector is never a logical model of its own -- it cannot be
          loaded without a base model;
        * if the repo ships a projector at all it is a vision repo, so the best
          matching projector is attached to every base entry;
        * a GGUF that is not a model at all -- an MTP speculative-decoding
          draft module, an imatrix calibration file -- is not a quant and is
          never offered. It parses as one by filename, which is how a 26B repo
          came to list "Q4_0 2.14 GiB" and "unknown 0.88 GiB" beside its one
          real 13 GiB weight file, each badged as fitting on a single GPU.
        """
        buckets: dict[tuple[str, str], list[GgufFileInfo]] = {}
        for info in self.files:
            if info.is_mmproj or info.is_auxiliary:
                continue
            buckets.setdefault((info.quant, shard_base(info.filename)), []).append(info)

        # Only disambiguate when a quant really is ambiguous, so the common
        # repo keeps the plain "<repo>:<quant>" group id.
        quant_counts = Counter(quant for quant, _ in buckets)
        projectors = self.mmproj_files

        out: list[LogicalDownload] = []
        for (quant, base), parts in sorted(buckets.items(), key=lambda kv: kv[0]):
            parts.sort(key=lambda f: (f.shard_index or 1, f.filename))
            missing = _missing_shard_count(parts)
            if missing:
                # A listing that names shard N-of-M but not every sibling (a
                # repo mid-upload, an LFS entry the API omitted) would download
                # "successfully" and produce an unloadable model. Refusing to
                # offer it is the only honest option.
                log.warning(
                    "hf.incomplete_shard_set",
                    repo_id=self.repo_id,
                    quant=quant,
                    base=base,
                    present=len(parts),
                    missing=missing,
                )
                continue
            mmproj = _pick_mmproj(quant, projectors)
            total = sum(f.size_bytes for f in parts)
            if mmproj is not None:
                total += mmproj.size_bytes
            out.append(
                LogicalDownload(
                    repo_id=self.repo_id,
                    quant=quant,
                    files=parts,
                    mmproj=mmproj,
                    total_bytes=total,
                    discriminator="" if quant_counts[quant] == 1 else base,
                )
            )
        return out


def _missing_shard_count(parts: list[GgufFileInfo]) -> int:
    """How many shards a group's own filenames say exist but the listing lacks.

    A single-file entry (no shard suffix) is always complete. For a sharded
    set, the filenames declare the total (``-of-000NN``); every index from 1
    to that total must be present exactly once.
    """
    totals = {f.shard_total for f in parts if f.shard_total}
    if not totals:
        return 0
    total = max(totals)
    present = {f.shard_index for f in parts if f.shard_index}
    return max(0, total - len(present & set(range(1, total + 1))))


def _pick_mmproj(base_quant: str, projectors: list[GgufFileInfo]) -> GgufFileInfo | None:
    """Best projector for a base file at ``base_quant``.

    A single projector serves every quant in the repo, which is the common case.
    With several, an exact quant match wins (some publishers ship a matched
    pair), then the precision preference in :data:`_MMPROJ_PRECISION_ORDER`.
    """
    if not projectors:
        return None
    if len(projectors) == 1:
        return projectors[0]
    for candidate in projectors:
        if candidate.quant == base_quant and base_quant != UNKNOWN_QUANT:
            return candidate
    for precision in _MMPROJ_PRECISION_ORDER:
        for candidate in projectors:
            if candidate.quant == precision:
                return candidate
    # Nothing recognised: the smallest projector is the least harmful default.
    return sorted(projectors, key=lambda f: (f.size_bytes or 1 << 62, f.filename))[0]


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def _retry_after_seconds(header: str | None, fallback: float) -> float:
    """Parse ``Retry-After`` (delta-seconds or HTTP-date), falling back on backoff."""
    if not header:
        return fallback
    header = header.strip()
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(header)
    except ValueError:
        # Python >= 3.10 raises for an unparseable date instead of returning
        # None. A proxy sending "Retry-After: soon" must degrade to the
        # ordinary backoff, not 500 the whole search request.
        return fallback
    if parsed is None:  # pragma: no cover - older interpreters
        return fallback
    import datetime as _dt

    now = _dt.datetime.now(tz=parsed.tzinfo or _dt.UTC)
    return max(0.0, (parsed - now).total_seconds())


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_optional_int(value: Any) -> int | None:
    """``int(value)`` for a real number, ``None`` for anything else.

    Distinct from :func:`_as_int` because "the field was absent" and "the field
    was 0" are different facts for ``trendingScore`` -- see the note on
    :attr:`GgufRepoInfo.trending_score`. ``bool`` is excluded explicitly since
    it is an ``int`` subclass and ``True`` would otherwise become a score of 1.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)


def _next_page_url(link_header: str | None, *, endpoint: str) -> str | None:
    """The ``rel="next"`` URL from an RFC 8288 ``Link`` header, if it is ours.

    **Security boundary.** Every request carries the user's HF token in an
    ``Authorization`` header, and this URL comes from the response body's
    metadata -- i.e. from upstream, not from us. Following it blindly would let
    a compromised or misconfigured endpoint (or a redirect-happy corporate
    proxy) point ``rel="next"`` at a host of its choosing and collect the token.
    So the origin is required to match the endpoint we were configured with;
    anything else ends pagination rather than being followed.

    Verified against the live API: ``/api/models`` does emit
    ``Link: <...&cursor=...>; rel="next"`` with ``full=true``, and following it
    returns the next, strictly older page.
    """
    if not link_header:
        return None
    for match in _LINK_RE.finditer(link_header):
        url = match.group("url").strip()
        if not url:
            continue
        if url.startswith(endpoint.rstrip("/")):
            return url
        log.warning("hf.pagination_cross_origin", endpoint=endpoint)
        return None
    return None


def _file_from_sibling(entry: dict[str, Any]) -> GgufFileInfo | None:
    """Build a :class:`GgufFileInfo` from one ``siblings``/blob entry.

    Returns ``None`` for non-GGUF entries (READMEs, configs, ``.imatrix``
    files) and for anything whose name is not a usable path component. A GGUF
    that is not a *model* -- an MTP draft module, an ``imatrix_*.gguf`` -- is
    still returned, flagged ``is_auxiliary``; the repo keeps the full picture
    and :meth:`GgufRepoInfo.logical_models` is what refuses to offer it.
    """
    name = entry.get("rfilename") or entry.get("filename")
    if not isinstance(name, str) or not name.lower().endswith(".gguf"):
        return None

    lfs = entry.get("lfs")
    lfs_dict: dict[str, Any] = lfs if isinstance(lfs, dict) else {}
    # ``lfs.size`` is the real object size; the plain ``size`` field can be the
    # size of the LFS *pointer* on some responses, which would be ~130 bytes.
    size = _as_int(lfs_dict.get("size")) or _as_int(entry.get("size"))

    oid = lfs_dict.get("oid")
    oid_str = oid if isinstance(oid, str) else None
    sha = lfs_dict.get("sha256")
    sha_str = sha if isinstance(sha, str) else None
    if sha_str is None and oid_str is not None and _SHA256_RE.match(oid_str.lower()):
        # Per the Git-LFS spec the pointer oid *is* the sha256 of the object, and
        # HF returns it bare (no "sha256:" prefix). Using it gives us end-to-end
        # verification on repos where the explicit field is absent.
        sha_str = oid_str.lower()

    index, total = shard_parts(name)
    return GgufFileInfo(
        filename=name,
        size_bytes=size,
        quant=parse_quant(name),
        is_mmproj=looks_like_mmproj(Path(name)),
        shard_index=index,
        shard_total=total,
        sha256=sha_str.lower() if sha_str else None,
        lfs_oid=oid_str,
        is_auxiliary=looks_like_auxiliary_gguf(name, size_bytes=size),
    )


def _repo_from_payload(payload: dict[str, Any]) -> GgufRepoInfo:
    repo_id = str(payload.get("id") or payload.get("modelId") or "")
    owner, _, name = repo_id.partition("/")
    siblings = payload.get("siblings")
    files: list[GgufFileInfo] = []
    if isinstance(siblings, list):
        for entry in siblings:
            if not isinstance(entry, dict):
                continue
            info = _file_from_sibling(entry)
            if info is not None:
                files.append(info)
    files.sort(key=lambda f: f.filename)

    gated_raw = payload.get("gated", False)
    gated: bool | str = gated_raw if isinstance(gated_raw, (bool, str)) else False
    last_modified = payload.get("lastModified")
    created_at = payload.get("createdAt")

    return GgufRepoInfo(
        repo_id=repo_id,
        publisher=owner,
        name=name or owner,
        downloads=_as_int(payload.get("downloads")),
        likes=_as_int(payload.get("likes")),
        gated=gated,
        private=bool(payload.get("private", False)),
        last_modified=last_modified if isinstance(last_modified, str) else None,
        files=files,
        created_at=created_at if isinstance(created_at, str) else None,
        trending_score=_as_optional_int(payload.get("trendingScore")),
    )


def _validate_choice(value: str, allowed: Iterable[str], *, param: str) -> str:
    """Normalise ``value`` against ``allowed`` or raise a :class:`BadRequestError`.

    The message lists the allowed values because this is reached from an HTTP
    query parameter, a GUI select and (soon) an MCP tool call: the caller that
    got it wrong is often an LLM, and "unknown sort 'popular'" without the menu
    just produces a second wrong guess.

    Case and surrounding whitespace are forgiven -- ``"Downloads"`` from a
    hand-written URL is unambiguous -- but nothing else is.
    """
    options = tuple(allowed)
    text = str(value or "").strip().lower()
    if text not in options:
        raise BadRequestError(
            f"unknown {param} {value!r}; allowed values are: {', '.join(options)}",
            param=param,
        )
    return text


def _repos_from_list(payload: Any) -> list[GgufRepoInfo]:
    """Parse a ``/api/models`` list response, dropping repos with no GGUF in them.

    The ``gguf`` filter matches on tags, so a repo tagged for GGUF that has not
    uploaded one yet (or keeps them in a subdirectory we refuse to flatten)
    comes back with an empty file list. Offering it would produce a picker row
    with zero quants to choose from.
    """
    if not isinstance(payload, list):
        raise UpstreamError("HuggingFace model search did not return a list")
    repos: list[GgufRepoInfo] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        repo = _repo_from_payload(entry)
        if repo.repo_id and repo.files:
            repos.append(repo)
    return repos


def _stamp(iso: str | None) -> float:
    """Sortable epoch seconds, with an unparseable/absent date sorting oldest."""
    parsed = parse_hf_timestamp(iso)
    return parsed.timestamp() if parsed is not None else 0.0


def _sort_key_fn(sort: str, *, date_field: str) -> Any:
    """Local descending sort for the windowed walk, matching HF's own ordering.

    Only needed because the window walk has to be fetched in date order (see
    :meth:`HfSearch._search_window`); the unwindowed path lets HF sort and never
    calls this.

    Every key negates rather than using ``reverse=True`` so the ``repo_id``
    tie-break stays *ascending*: a stable, alphabetical order for equal scores
    means two identical searches render the same list, instead of shuffling rows
    under the user's cursor because HF returned the page in a different order.

    ``trending`` is the one lossy case. HF only reports ``trendingScore`` when
    it is also the sort key, so a windowed trending search sees ``None`` for
    every repo; the ``-1`` sentinel makes them all tie and the download count
    decides, which is a reasonable proxy and, more importantly, not a fabricated
    score.
    """
    if sort == "likes":
        return lambda r: (-r.likes, -r.downloads, r.repo_id)
    if sort == "trending":
        return lambda r: (
            -(r.trending_score if r.trending_score is not None else -1),
            -r.downloads,
            r.repo_id,
        )
    if sort == "updated":
        return lambda r: (-_stamp(r.last_modified), -r.downloads, r.repo_id)
    if sort == "created":
        return lambda r: (-_stamp(r.created_at), -r.downloads, r.repo_id)
    # downloads: newest-inside-the-window breaks ties, using whichever date the
    # window was measured against so the secondary order matches the filter.
    date_of = (lambda r: r.created_at) if date_field == "created" else (lambda r: r.last_modified)
    return lambda r: (-r.downloads, -_stamp(date_of(r)), r.repo_id)


class HfSearch:
    """Async client for the two HuggingFace endpoints the model picker needs."""

    def __init__(
        self,
        config: Config,
        *,
        client: httpx.AsyncClient | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.config = config
        self._endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip(
            "/"
        )
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(_DEFAULT_TIMEOUT_S),
            follow_redirects=True,
        )
        #: Whether the last :meth:`search` stopped on the page cap rather than
        #: on the date cutoff -- i.e. matches exist that it did not return.
        #:
        #: Reported here rather than in the return value so the three existing
        #: ``search()`` call sites keep working unchanged. It is safe to read
        #: straight after ``await search(...)`` because every caller builds an
        #: ``HfSearch``, uses it, and closes it within one request; nothing
        #: shares an instance across concurrent searches.
        #:
        #: This matters more than it looks. Measured live: a bare browse (empty
        #: query) hits the cap at a 7-day window, and "gemma" hits it at 90
        #: days, while "llama" over 30 days completes. So the cap is reached by
        #: ordinary use, not just pathological queries, and silently presenting
        #: a capped walk as "the past month" would understate the hub.
        self.last_search_truncated: bool = False

    @property
    def endpoint(self) -> str:
        return self._endpoint

    # -- lifecycle ------------------------------------------------------

    async def aclose(self) -> None:
        """Close the HTTP client, but only if we created it."""
        if self._owns_client:
            await self._client.aclose()

    # -- requests -------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Request headers.

        The token goes in the ``Authorization`` header, never in the URL, so it
        cannot leak through a logged URL, a redirect ``Location``, or an HTTP
        access log. It is also registered as a secret value at config load, so
        the structlog redaction processor scrubs it from any line that does
        embed it.
        """
        headers = {"Accept": "application/json"}
        token = self.config.hf.token
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET a JSON endpoint with bounded 429 retries."""
        payload, _next_url = await self._get_page(path, params)
        return payload

    async def _get_page(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        url: str | None = None,
    ) -> tuple[Any, str | None]:
        """One page of a JSON endpoint, plus the ``rel="next"`` URL if there is one.

        ``url`` supersedes ``path`` for cursor follow-ups, because HF's
        ``Link`` header hands back a fully-formed URL (opaque ``cursor=`` blob
        and all) that must be replayed verbatim -- reassembling it from parsed
        query parameters risks re-encoding the cursor into something HF rejects.
        ``path`` is still passed so error messages and log lines name the
        endpoint rather than a URL carrying the user's search terms.
        """
        target = url or f"{self._endpoint}{path}"
        backoff = 1.0
        response: httpx.Response | None = None
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = await self._client.get(target, params=params, headers=self._headers())
            except httpx.HTTPError as exc:
                raise UpstreamError(
                    f"HuggingFace request to {path} failed: {type(exc).__name__}: {exc}"
                ) from exc
            if response.status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                wait = min(
                    _retry_after_seconds(response.headers.get("Retry-After"), backoff),
                    _MAX_RETRY_SLEEP_S,
                )
                # `path` only, never the full URL: the query carries the user's
                # search terms and we keep those out of the log.
                log.warning("hf.rate_limited", path=path, attempt=attempt + 1, wait_s=wait)
                await asyncio.sleep(wait)
                backoff = min(backoff * 2, _MAX_RETRY_SLEEP_S)
                continue
            break

        assert response is not None  # noqa: S101 - loop always runs at least once
        self._raise_for_status(response, path)
        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError(f"HuggingFace returned non-JSON for {path}") from exc
        return payload, _next_page_url(response.headers.get("Link"), endpoint=self._endpoint)

    def _raise_for_status(self, response: httpx.Response, path: str) -> None:
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise BadRequestError(
                f"HuggingFace refused access to {path} (HTTP {status}). This repository is "
                "gated or private; set an access token in config key 'hf.token' "
                "(or the SF_HF__TOKEN environment variable) and accept the model's "
                "licence on huggingface.co first.",
                param="hf.token",
                details={"status": status, "path": path},
            )
        if status == 404:
            raise BadRequestError(
                f"HuggingFace has no such repository or file: {path}",
                details={"status": status, "path": path},
            )
        if status == 429:
            raise UpstreamError(
                "HuggingFace rate-limited this client and kept doing so after "
                f"{_MAX_RATE_LIMIT_RETRIES} retries. Wait a few minutes, or set "
                "'hf.token' -- authenticated requests get a much higher quota.",
                details={"status": status, "path": path},
            )
        raise UpstreamError(
            f"HuggingFace returned HTTP {status} for {path}",
            details={"status": status, "path": path},
        )

    # -- public API -----------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        author: str | None = None,
        sort: str = "downloads",
        newer_than_days: int | None = None,
        date_field: str = "updated",
    ) -> list[GgufRepoInfo]:
        """Search GGUF repos, ordered by ``sort`` and optionally limited to a recency window.

        ``full=true`` is requested so each hit already carries its ``siblings``
        list and the picker can show quant variants without an extra round trip
        per repo. Sizes are *not* in this response -- call :meth:`repo_info` for
        the repo the user actually picks.

        ``sort`` is one of :data:`SORT_KEYS`; direction is always descending,
        because there is no user story for "least downloaded GGUF on the hub".

        ``newer_than_days`` restricts results to repos touched (``date_field``
        ``"updated"``) or first published (``"created"``) within that many days.
        HF has **no server-side date filter**, so this is not a parameter we can
        hand off -- see :meth:`_search_window` for how the window is actually
        satisfied.
        """
        sort_name = _validate_choice(sort, SORT_KEYS, param="sort")
        date_key = _validate_choice(date_field, DATE_FIELDS, param="date_field")
        limit = max(1, int(limit))

        if newer_than_days is None:
            # SORT_KEYS[...] rather than the user-facing name: HF answers
            # `sort=trending` (as opposed to `sort=trendingScore`) with a 400,
            # and `sort=updated` with an unsorted 200.
            repos = await self._search_page(
                query, limit=limit, author=author, sort=SORT_KEYS[sort_name]
            )
            truncated = False
        else:
            repos, truncated = await self._search_window(
                query,
                limit=limit,
                author=author,
                sort=sort_name,
                newer_than_days=int(newer_than_days),
                date_field=date_key,
            )

        log.info(
            "hf.search",
            query_len=len(query),
            results=len(repos),
            limit=limit,
            sort=sort_name,
            newer_than_days=newer_than_days,
            date_field=date_key if newer_than_days is not None else None,
            truncated=truncated,
        )
        self.last_search_truncated = truncated
        return repos

    async def _search_page(
        self, query: str, *, limit: int, author: str | None, sort: str
    ) -> list[GgufRepoInfo]:
        """The plain, unwindowed case: one request, HF does the ordering."""
        payload, _next_url = await self._get_page(
            "/api/models", self._search_params(query, limit=limit, author=author, sort=sort)
        )
        return _repos_from_list(payload)

    async def _search_window(
        self,
        query: str,
        *,
        limit: int,
        author: str | None,
        sort: str,
        newer_than_days: int,
        date_field: str,
    ) -> tuple[list[GgufRepoInfo], bool]:
        """Repos inside a date window, re-ordered locally by the requested sort.

        HF offers no ``since=`` parameter, so the window has to be produced from
        an ordering we *can* ask for. The naive alternative -- fetch one page
        sorted by downloads and drop the old entries -- is wrong in the way that
        looks right: asking for "most-downloaded GGUF from the past week" would
        return whatever slice of the all-time top 100 happened to be touched
        recently, i.e. usually nothing, and the empty result would read as "no
        new models this week" rather than "wrong query".

        So the walk is ordered by the *date* instead (``lastModified`` or
        ``createdAt``, descending). That ordering is monotonic, which buys the
        one property that makes this cheap: the first entry older than the
        cutoff proves every remaining entry is too, so the walk stops there
        instead of paging to the end of the hub. Only then are the survivors
        sorted by what the user actually asked for and cut to ``limit``.

        Pagination is HF's cursor, taken from the ``Link: rel="next"`` header --
        verified live against ``/api/models?filter=gguf&full=true``, which does
        emit it and does return strictly older entries on the next page.

        Returns ``(repos, truncated)``. ``truncated`` is ``True`` when the page
        cap was reached with the window still open, i.e. matches exist that this
        result does not include.
        """
        cutoff = time.time() - newer_than_days * 86400.0
        api_date_sort = SORT_KEYS[date_field]
        params = self._search_params(
            query, limit=_WINDOW_PAGE_SIZE, author=author, sort=api_date_sort
        )

        collected: list[GgufRepoInfo] = []
        next_url: str | None = None
        reached_cutoff = False
        pages = 0

        while pages < _MAX_WINDOW_PAGES:
            payload, next_url = await self._get_page(
                "/api/models", None if next_url else params, url=next_url
            )
            pages += 1
            if not isinstance(payload, list):
                raise UpstreamError("HuggingFace model search did not return a list")
            for entry in payload:
                if not isinstance(entry, dict):
                    continue
                stamp = parse_hf_timestamp(entry.get(api_date_sort))
                if stamp is None or stamp.timestamp() < cutoff:
                    # Sorted descending, so this is the end of the window. HF
                    # sorts null dates last, which is why an unparseable stamp
                    # is treated as "past the end" rather than skipped -- a repo
                    # with no date cannot be shown to be inside the window.
                    reached_cutoff = True
                    break
                repo = _repo_from_payload(entry)
                if repo.repo_id and repo.files:
                    collected.append(repo)
            if reached_cutoff or not next_url:
                break

        truncated = not reached_cutoff and next_url is not None
        if truncated:
            log.info(
                "hf.search_window_truncated",
                pages=pages,
                newer_than_days=newer_than_days,
                date_field=date_field,
                collected=len(collected),
            )
        collected.sort(key=_sort_key_fn(sort, date_field=date_field))
        return collected[:limit], truncated

    def _search_params(
        self, query: str, *, limit: int, author: str | None, sort: str
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "filter": "gguf",
            "limit": limit,
            "full": "true",
            "sort": sort,
            "direction": -1,
        }
        if query:
            params["search"] = query
        if author:
            params["author"] = author
        return params

    async def repo_info(self, repo_id: str) -> GgufRepoInfo:
        """Full file listing for one repo, with sizes and LFS checksums.

        ``?blobs=true`` is what turns ``siblings`` from bare filenames into
        entries with ``size`` and ``lfs.{oid,size,sha256}`` -- i.e. the only way
        to get both a real size for the fit estimate and a hash to verify the
        download against.
        """
        repo_id = repo_id.strip().strip("/")
        if not repo_id:
            raise BadRequestError("repo_id is required", param="repo_id")
        payload = await self._get_json(f"/api/models/{repo_id}", {"blobs": "true"})
        if not isinstance(payload, dict):
            raise UpstreamError(f"HuggingFace returned an unexpected payload for {repo_id}")
        repo = _repo_from_payload(payload)
        if not repo.repo_id:
            # Some mirrors omit "id"; trust the caller's spelling in that case.
            repo = GgufRepoInfo(
                repo_id=repo_id,
                publisher=repo_id.partition("/")[0],
                name=repo_id.partition("/")[2] or repo_id,
                downloads=repo.downloads,
                likes=repo.likes,
                gated=repo.gated,
                private=repo.private,
                last_modified=repo.last_modified,
                files=repo.files,
                created_at=repo.created_at,
                trending_score=repo.trending_score,
            )
        unsized = [f.filename for f in repo.files if not f.size_known]
        if unsized:
            log.warning("hf.repo_missing_sizes", repo_id=repo.repo_id, count=len(unsized))
        return repo

    def url_for(self, repo_id: str, filename: str, *, revision: str = "main") -> str:
        """Download URL for one file in a repo."""
        return file_url(repo_id, filename, endpoint=self._endpoint, revision=revision)
