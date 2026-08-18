"""The model registry: turn a directory of GGUF files into addressable models.

Responsibilities
----------------

* Walk the configured model directories and build one :class:`ModelRecord` per
  *logical* model -- collapsing multi-part shards into a single record and
  attaching vision projectors (``mmproj``) to the base model they belong to.
* Give every model a **stable identifier** that survives restarts, re-scans and
  file-timestamp churn, plus a set of short aliases so clients that hardcode
  LM Studio-style names still resolve.
* Track GGUF LoRA adapters and *virtual* models (a base plus an adapter set,
  selectable by name through the OpenAI API).
* Cache parsed GGUF metadata in SQLite so a library of 60 GB+ files does not
  get re-parsed on every startup.

Design notes worth knowing before editing
-----------------------------------------

**Why the ID is a relative path, not a hash.** OpenClaw (and every other
client) persists the selected model id in its own config. A content hash would
change when a file is re-downloaded; an mtime-derived id would change when a
backup tool touches the file; an auto-increment id would change when the
library is re-indexed on another machine. The path relative to the model
directory is the only thing that is both human-readable and genuinely stable,
so that is the id: POSIX separators, ``.gguf`` stripped, and the
``-00001-of-0000N`` shard suffix stripped so a model does not change its name
when it is re-quantised into a different number of parts.

**Why mmproj pairing has four rules.** There is no standard for naming a
vision projector. The wild contains at least ``mmproj-<base>.gguf`` (prefix),
``<base>-mmproj-<quant>.gguf`` (infix), ``<base>.mmproj-<quant>.gguf``
(suffix), and a bare ``mmproj-F32.gguf`` that shares *nothing* with the base
name. A single heuristic cannot cover all four without also mis-pairing, so
the rules are tried in decreasing order of confidence and the weakest one
(directory arity) only fires when the directory is unambiguous.

**Why the cache key is (path + parser version, mtime, size).** Hashing a 60 GB
file to detect change costs more than re-parsing it. mtime alone is fooled by a
restore that preserves timestamps; size alone is fooled by an equal-size
re-quant. Together they are cheap and wrong only in contrived cases -- and
``scan(force=True)`` exists for those. What they cannot detect is a change to
the *parser*: see :func:`_cache_key`.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from studioforge.config import Config
from studioforge.core import gguf
from studioforge.db import Database
from studioforge.errors import BadRequestError, ModelNotFoundError
from studioforge.logging import get_logger
from studioforge.types import (
    AdapterAttachment,
    AdapterRecord,
    GgufMeta,
    ModelCapabilities,
    ModelKind,
    ModelRecord,
    ModelSettings,
    VirtualPreset,
    validate_chat_template_file,
)

log = get_logger(__name__)

#: Signature of the injectable GGUF metadata reader. Positional so tests can
#: substitute a plain function without mirroring ``read_meta``'s keyword-only
#: ``shard_paths`` parameter.
MetaReader = Callable[[Path, "Sequence[Path] | None"], GgufMeta]

#: ``<base>-00003-of-00007`` as written by ``llama-gguf-split``.
_SHARD_RE = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")

#: The ``mmproj`` token plus the separators glued to it, so the token can be
#: excised and the remaining name compared against a base model's name.
_MMPROJ_TOKEN_RE = re.compile(r"(?P<pre>[-._ ])?mmproj(?P<post>[-._ ])?", re.IGNORECASE)

#: Trailing ``-GGUF`` on a HuggingFace repo directory is packaging noise, not
#: part of the model's name; strip it when deriving the short alias.
_REPO_SUFFIX_RE = re.compile(r"[-_.]g+uf+$", re.IGNORECASE)

_EMBEDDING_NAME_RE = re.compile(r"embedding|embeddings|-embed\b|_embed\b|\bembed-", re.IGNORECASE)
_RERANK_NAME_RE = re.compile(r"rerank", re.IGNORECASE)

#: Architectures that are embedding-only regardless of what the filename says.
_EMBEDDING_ARCHS: frozenset[str] = frozenset(
    {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "gte", "roberta", "xlm-roberta"}
)

#: Minimum shared-prefix length (absolute, and as a fraction of the base name)
#: before an mmproj is considered to belong to a base model. Low enough to
#: catch ``<base>.mmproj-f16`` and high enough that ``mmproj-F32`` never
#: accidentally latches onto a model beginning with "F".
_MIN_PREFIX_CHARS = 4
_MIN_PREFIX_FRACTION = 0.25

_SKIP_DIR_PREFIXES = (".", "__")


@dataclass(slots=True)
class ScanResult:
    """Outcome of one :meth:`Registry.scan`."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unchanged: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    #: Models whose file is still on disk but failed to parse this time, so the
    #: previous record was kept instead of being dropped. See
    #: :meth:`Registry.scan`.
    stale: list[str] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def total(self) -> int:
        return len(self.added) + self.unchanged


@dataclass(slots=True)
class _Entry:
    """One GGUF file on disk, classified and with its metadata resolved."""

    model_id: str
    stem: str
    path: Path
    model_dir: Path
    shards: list[Path]
    size_bytes: int
    mtime: float
    meta: GgufMeta
    publisher: str | None
    repo: str | None


def _default_meta_reader(path: Path, shard_paths: Sequence[Path] | None = None) -> GgufMeta:
    """Adapter around :func:`gguf.read_meta`'s keyword-only shard parameter."""
    return gguf.read_meta(path, shard_paths=shard_paths)


def _common_prefix_len(left: str, right: str) -> int:
    left_cf, right_cf = left.casefold(), right.casefold()
    limit = min(len(left_cf), len(right_cf))
    index = 0
    while index < limit and left_cf[index] == right_cf[index]:
        index += 1
    return index


def _mmproj_core(stem: str) -> str:
    """``stem`` with the ``mmproj`` token (and one glued separator) removed."""
    match = _MMPROJ_TOKEN_RE.search(stem)
    if match is None:
        return stem
    before = stem[: match.start()]
    after = stem[match.end() :]
    if before and after:
        separator = match.group("pre") or match.group("post") or "-"
        return f"{before}{separator}{after}"
    return before or after


def _file_size(path: Path | None) -> int:
    """Size of a file, or 0 when absent.

    The projector size is captured at scan time so the planner never touches
    the filesystem while deciding placement.
    """
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0



def _cache_key(path: Path) -> str:
    """SQLite key for one file's parsed metadata: the path plus a parser version.

    ``(mtime, size)`` already guards the *file* side of the cache. The parser
    side had no guard at all: after this module learns to read a new GGUF key,
    every already-registered model would keep serving metadata parsed without
    it -- forever, since nothing about the file changed. That is a silent
    failure of exactly the kind the planner cannot survive (a Qwen3.5 charged 4x
    its real KV, with no sign anything was stale). Folding
    :data:`gguf.META_FORMAT_VERSION` into the key means a bump invalidates every
    row at once, and the next ordinary ``scan()`` re-reads the headers.

    Rows under an older key are not in ``cache_keep``, so ``prune_cache`` drops
    them at the end of that same scan; the cache does not grow a version's worth
    of dead rows.
    """
    return f"{path}#meta{gguf.META_FORMAT_VERSION}"


def _is_under(path: Path, root: Path) -> bool:
    """Whether ``path`` lies under ``root`` (lexically; the root may not exist)."""
    try:
        return path.resolve(strict=False).is_relative_to(root.resolve(strict=False))
    except (OSError, ValueError):
        return False


def _newest_mtime(paths: Sequence[Path], *, fallback: float) -> float:
    """Newest mtime across a logical model's files.

    A multi-part download finishes on its LAST shard, so keying recency off
    shard 1 would rank a just-downloaded 2-part model by when its first part
    landed. Taking the max is what makes "recently downloaded" honest.
    """
    newest = fallback
    for path in paths:
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest

class Registry:
    """In-memory model index backed by the filesystem and SQLite.

    Thread-safety: the record/alias maps are guarded by a re-entrant lock
    because :meth:`scan` runs on a worker thread while the HTTP API reads
    :meth:`all`/:meth:`resolve` concurrently. The lock is held only around map
    mutation -- never across filesystem or GGUF parsing work.
    """

    def __init__(
        self,
        config: Config,
        db: Database,
        *,
        meta_reader: MetaReader | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._meta_reader: MetaReader = meta_reader or _default_meta_reader
        self._lock = threading.RLock()
        self._models: dict[str, ModelRecord] = {}
        self._aliases: dict[str, str] = {}
        self._adapters: dict[str, AdapterRecord] = {}
        self._adapters_loaded = False
        self._last_scan_at: float | None = None

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, *, force: bool = False) -> ScanResult:
        """Walk the model directories and rebuild the index.

        One bad file never aborts the scan: parse failures land in
        :attr:`ScanResult.errors`, because a library of forty models must not
        become unusable because one download truncated.

        **A parse failure does not remove a known model.** If the file is still
        on disk, the record from the previous scan is carried over and marked
        ``stale``. A model that disappears from the catalogue and comes back
        minutes later is one of the worst failure modes there is: the load
        fails with "model not found", the client's saved model id looks wrong,
        and by the time anyone investigates the file parses fine again. Files
        genuinely get locked mid-write, by antivirus, or by a sync client, so
        the transient case is the common one. Only a model whose file is gone
        is dropped.

        ``force=True`` re-parses every file (ignoring the metadata cache) but
        keeps the same stickiness: forcing a re-read is how a user fixes bad
        cached metadata, and it should not double as "delete anything that
        happens to be locked right now". Removal is
        :meth:`delete_model`'s job, which is explicit.
        """
        started = time.perf_counter()
        result = ScanResult()

        files, missing_roots = self._walk()
        entries, mmprojs, adapters, cache_keep, failures = self._classify(
            files, result, force=force
        )
        pairs = self._pair_mmproj(entries, mmprojs)

        saved_settings = self._db.all_model_settings()
        records: dict[str, ModelRecord] = {}
        for entry in entries.values():
            records[entry.model_id] = self._build_record(entry, pairs.get(entry.model_id))

        with self._lock:
            previous = self._models
            self._carry_over_stale(previous, records, failures, result)
            self._carry_over_unreachable(previous, records, missing_roots, result)
            for model_id, record in records.items():
                old = previous.get(model_id)
                if old is not None and not old.is_virtual:
                    # added_at/last_used_at are session facts, not file facts:
                    # a re-scan must not make every model look brand new.
                    record.added_at = old.added_at
                    record.last_used_at = old.last_used_at
                record.settings = self._settings_from(model_id, saved_settings)
            result.added = sorted(k for k in records if k not in previous)
            result.removed = sorted(
                k for k, v in previous.items() if k not in records and not v.is_virtual
            )
            result.unchanged = sum(1 for k in records if k in previous)

            self._models = records
            self._adapters = {a.id: a for a in adapters}
            self._adapters_loaded = True
            self._load_virtual_models(saved_settings)
            self._rebuild_aliases()
            self._last_scan_at = time.time()

        for adapter in adapters:
            self._db.save_adapter(adapter.model_dump())
        self._db.prune_cache(cache_keep)

        result.duration_s = time.perf_counter() - started
        log.info(
            "registry.scan",
            models=len(records),
            adapters=len(adapters),
            added=len(result.added),
            removed=len(result.removed),
            errors=len(result.errors),
            stale=len(result.stale),
            duration_s=round(result.duration_s, 3),
            force=force,
        )
        return result

    def _carry_over_stale(
        self,
        previous: dict[str, ModelRecord],
        records: dict[str, ModelRecord],
        failures: dict[str, tuple[Path, str]],
        result: ScanResult,
    ) -> None:
        """Keep known models whose file still exists but would not parse.

        Caller holds the lock. A failure for a model we have never seen is left
        alone -- there is nothing to carry over, and it stays in
        ``result.errors`` exactly as before.
        """
        for model_id, (path, error) in failures.items():
            if model_id in records:
                continue
            old = previous.get(model_id)
            if old is None or old.is_virtual:
                continue
            try:
                still_there = path.exists()
            except OSError:  # pragma: no cover - unreadable mount
                still_there = True
            if not still_there:
                continue
            old.stale = True
            old.stale_reason = error
            records[model_id] = old
            result.stale.append(model_id)
            log.warning(
                "registry.kept_stale_record",
                model_id=model_id,
                path=str(path),
                error=error,
                hint=(
                    "the file is still on disk but could not be read this scan; "
                    "keeping the previously indexed model rather than removing it"
                ),
            )

    def _carry_over_unreachable(
        self,
        previous: dict[str, ModelRecord],
        records: dict[str, ModelRecord],
        missing_roots: Sequence[Path],
        result: ScanResult,
    ) -> None:
        """Keep every known model whose root directory is not there right now.

        Caller holds the lock. This is the "unreachable, not removed"
        distinction: a model is *removed* only when its root was walked and
        the file was not in it. A model under a root that could not be walked
        at all is carried over as ``stale`` with a reason naming the root, so
        the index, ``/v1/models`` and the TTL sweeper all keep treating it as
        the model it still is until the drive comes back.
        """
        if not missing_roots:
            return
        roots = [Path(root) for root in missing_roots]
        for model_id, old in previous.items():
            if model_id in records or old.is_virtual:
                continue
            path = Path(old.path)
            under = next((root for root in roots if _is_under(path, root)), None)
            if under is None:
                continue
            old.stale = True
            old.stale_reason = f"model directory {under} is not available right now"
            records[model_id] = old
            result.stale.append(model_id)
        if any(r.stale and "not available" in (r.stale_reason or "") for r in records.values()):
            log.warning(
                "registry.model_dir_unreachable",
                roots=[str(r) for r in roots],
                hint=(
                    "the models under it are kept in the index as stale rather than "
                    "removed; they return when the directory is reachable again"
                ),
            )

    def reconcile(self) -> ScanResult:
        """Startup alias for :meth:`scan`."""
        return self.scan()

    # --- walk ---------------------------------------------------------

    def _walk(self) -> tuple[list[tuple[Path, Path]], list[Path]]:
        """``(found, missing_roots)``: every GGUF under the roots, and the roots
        that were not there to walk.

        Symlinks are followed (people keep models on a second drive and link
        them in) but every directory's real path is remembered so a loop
        terminates instead of recursing forever.

        A root that is not a directory right now -- a second drive that has
        not mounted yet, a network share that dropped -- is reported rather
        than silently skipped, because "removed" and "unreachable" are
        different facts: the models under it are still there, and dropping
        them from the index (which is what an empty walk used to mean) would
        make a serving model disappear from ``/v1/models`` and, on Linux, could
        talk a sweeper into unloading everything on that drive.
        """
        found: list[tuple[Path, Path]] = []
        missing: list[Path] = []
        seen_dirs: set[str] = set()
        seen_files: set[str] = set()
        for model_dir in self._config.model_dirs():
            root = Path(model_dir)
            if not root.is_dir():
                log.warning("registry.model_dir_missing", path=str(root))
                missing.append(root)
                continue
            for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
                try:
                    real = str(Path(dirpath).resolve())
                except OSError:  # pragma: no cover - unreadable junction
                    dirnames[:] = []
                    continue
                if real in seen_dirs:
                    dirnames[:] = []
                    continue
                seen_dirs.add(real)
                # Dot/dunder dirs are tool caches (.cache/huggingface, .git,
                # __pycache__): never a model library, often huge.
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(_SKIP_DIR_PREFIXES))
                for filename in sorted(filenames):
                    if not filename.lower().endswith(".gguf"):
                        continue
                    path = Path(dirpath) / filename
                    key = str(path).casefold()
                    if key in seen_files:
                        continue
                    seen_files.add(key)
                    found.append((root, path))
        return found, missing

    # --- classification -----------------------------------------------

    def _classify(
        self,
        files: Sequence[tuple[Path, Path]],
        result: ScanResult,
        *,
        force: bool,
    ) -> tuple[
        dict[str, _Entry],
        dict[str, _Entry],
        list[AdapterRecord],
        list[str],
        dict[str, tuple[Path, str]],
    ]:
        """Split discovered files into base models, mmprojs and adapters.

        The fifth element maps ``model_id -> (path, error)`` for files that
        could not be parsed, which is what lets :meth:`scan` keep a known model
        instead of dropping it (see that method's docstring).
        """
        bases: dict[str, _Entry] = {}
        mmprojs: dict[str, _Entry] = {}
        adapters: list[AdapterRecord] = []
        cache_keep: list[str] = []
        failures: dict[str, tuple[Path, str]] = {}

        for model_dir, path in files:
            cache_keep.append(_cache_key(path))
            stem = path.stem
            shard = _SHARD_RE.match(stem)
            base_stem = stem
            shards = [path]
            if shard is not None:
                if int(shard.group("index")) != 1:
                    # Only shard 1 becomes a record; the rest are its files.
                    continue
                base_stem = shard.group("base")
                missing = self._missing_shards(path, base_stem, int(shard.group("total")))
                if missing:
                    model_id = self._model_id(model_dir, path, base_stem)
                    result.errors.append(
                        (model_id, f"incomplete multi-part model, missing: {', '.join(missing)}")
                    )
                    continue
                shards = gguf.shard_paths_for(path)

            model_id = self._model_id(model_dir, path, base_stem)
            try:
                stat = path.stat()
                meta = self._meta_for(path, shards, force=force)
                size_bytes = sum(s.stat().st_size for s in shards)
            except (gguf.GgufError, OSError, ValueError, ValidationError) as exc:
                message = f"{type(exc).__name__}: {exc}"
                result.errors.append((model_id, message))
                # Recorded (not just logged) so scan() can decide whether this
                # is a transient read failure over a file that is still there.
                failures[model_id] = (path, message)
                log.warning("registry.parse_failed", model_id=model_id, error=str(exc))
                continue

            publisher, repo = self._publisher_repo(model_dir, path)
            entry = _Entry(
                model_id=model_id,
                stem=base_stem,
                path=path,
                model_dir=model_dir,
                shards=shards,
                size_bytes=size_bytes,
                mtime=_newest_mtime(shards, fallback=stat.st_mtime),
                meta=meta,
                publisher=publisher,
                repo=repo,
            )

            if meta.is_adapter:
                adapters.append(self._build_adapter(entry))
                continue
            if self._is_mmproj(entry):
                mmprojs[model_id] = entry
                continue
            if model_id in bases:
                result.errors.append((model_id, "duplicate model id across model directories"))
                log.warning("registry.duplicate_id", model_id=model_id, path=str(path))
                continue
            bases[model_id] = entry

        return bases, mmprojs, adapters, cache_keep, failures

    def _is_mmproj(self, entry: _Entry) -> bool:
        """A projector, not a loadable model.

        Metadata is authoritative (``clip.*`` keys / vision-only tensors); the
        filename heuristic only rescues files whose metadata is thin. Filename
        alone would misfire on single-file multimodal models that happen to
        mention mmproj in their repo name.
        """
        if entry.meta.is_mmproj:
            return True
        return gguf.looks_like_mmproj(entry.path) and entry.meta.has_vision_tensors

    def _missing_shards(self, path: Path, base: str, total: int) -> list[str]:
        """Shard filenames implied by ``path``'s name that are absent on disk."""
        missing: list[str] = []
        for index in range(1, total + 1):
            name = f"{base}-{index:05d}-of-{total:05d}{path.suffix}"
            if not path.with_name(name).is_file():
                missing.append(name)
        return missing

    # --- ids ----------------------------------------------------------

    def _model_id(self, model_dir: Path, path: Path, base_stem: str) -> str:
        """Path relative to the model dir, POSIX-separated, extension stripped.

        See the module docstring for why this, and not a hash, is the id.
        """
        try:
            rel = path.relative_to(model_dir)
        except ValueError:  # pragma: no cover - _walk guarantees containment
            rel = Path(path.name)
        return "/".join([*rel.parts[:-1], base_stem])

    def _publisher_repo(self, model_dir: Path, path: Path) -> tuple[str | None, str | None]:
        """``publisher/repo/file.gguf`` is the LM Studio layout; adapt to depth."""
        try:
            rel = path.relative_to(model_dir)
        except ValueError:  # pragma: no cover
            return None, None
        dirs = rel.parts[:-1]
        repo = dirs[-1] if dirs else None
        publisher = dirs[-2] if len(dirs) >= 2 else None
        return publisher, repo

    # --- metadata + cache ---------------------------------------------

    def _meta_for(self, path: Path, shards: Sequence[Path], *, force: bool) -> GgufMeta:
        """Parsed metadata, served from the SQLite cache when the file is unchanged."""
        stat = path.stat()
        key = _cache_key(path)
        if not force:
            cached = self._db.get_cached_meta(key, stat.st_mtime, stat.st_size)
            if cached is not None:
                try:
                    return GgufMeta.model_validate(cached)
                except ValidationError:
                    # A cache row written by an older schema: re-parse instead
                    # of failing the whole scan.
                    log.warning("registry.cache_row_invalid", path=key)
        meta = self._meta_reader(path, list(shards) if len(shards) > 1 else None)
        self._db.put_cached_meta(key, stat.st_mtime, stat.st_size, meta.model_dump(mode="json"))
        return meta

    # --- mmproj pairing -----------------------------------------------

    def _pair_mmproj(self, bases: dict[str, _Entry], mmprojs: dict[str, _Entry]) -> dict[str, Path]:
        """Map base model id -> projector path.

        Rules, strongest first (see the module docstring for the why):

        a. the projector's name *is* the base name with an ``mmproj`` token
           inserted -- an exact match once the token is excised;
        b. the longest shared prefix between the token-excised projector name
           and the base name, above a length threshold;
        c. the directory contains exactly one projector and exactly one base
           model, so pairing is unambiguous even with unrelated names
           (``mmproj-F32.gguf``);
        d. otherwise unpaired -- logged at INFO, never guessed.
        """
        by_dir_base: dict[Path, list[_Entry]] = {}
        by_dir_proj: dict[Path, list[_Entry]] = {}
        for entry in bases.values():
            by_dir_base.setdefault(entry.path.parent, []).append(entry)
        for entry in mmprojs.values():
            by_dir_proj.setdefault(entry.path.parent, []).append(entry)

        pairs: dict[str, Path] = {}
        for directory, candidates in by_dir_proj.items():
            local_bases = by_dir_base.get(directory, [])
            if not local_bases:
                log.info("registry.mmproj_orphan", dir=str(directory), count=len(candidates))
                continue
            cores = {c.model_id: _mmproj_core(c.stem) for c in candidates}
            for base in local_bases:
                chosen = self._choose_mmproj(base, candidates, cores)
                if chosen is None and len(candidates) == 1 and len(local_bases) == 1:
                    chosen = candidates[0]
                    log.info(
                        "registry.mmproj_paired",
                        model_id=base.model_id,
                        mmproj=chosen.path.name,
                        rule="sole-candidate",
                    )
                if chosen is None:
                    log.info(
                        "registry.mmproj_unpaired",
                        model_id=base.model_id,
                        candidates=[c.path.name for c in candidates],
                    )
                    continue
                pairs[base.model_id] = chosen.path
        return pairs

    def _choose_mmproj(
        self, base: _Entry, candidates: Sequence[_Entry], cores: dict[str, str]
    ) -> _Entry | None:
        for candidate in candidates:
            if cores[candidate.model_id].casefold() == base.stem.casefold():
                log.info(
                    "registry.mmproj_paired",
                    model_id=base.model_id,
                    mmproj=candidate.path.name,
                    rule="exact",
                )
                return candidate
        best: _Entry | None = None
        best_score = 0
        threshold = max(_MIN_PREFIX_CHARS, int(len(base.stem) * _MIN_PREFIX_FRACTION))
        for candidate in candidates:
            score = _common_prefix_len(cores[candidate.model_id], base.stem)
            if score >= threshold and score > best_score:
                best, best_score = candidate, score
        if best is not None:
            log.info(
                "registry.mmproj_paired",
                model_id=base.model_id,
                mmproj=best.path.name,
                rule="prefix",
                score=best_score,
            )
        return best

    # --- record construction ------------------------------------------

    def _build_record(self, entry: _Entry, mmproj: Path | None) -> ModelRecord:
        meta = entry.meta
        kind = self._detect_kind(entry)
        capabilities = ModelCapabilities(
            vision=mmproj is not None or meta.has_vision_tensors,
            embedding=kind == "embedding",
            tools=meta.supports_tools,
            thinking=meta.supports_thinking,
            multi_part=len(entry.shards) > 1,
        )
        return ModelRecord(
            id=entry.model_id,
            name=entry.stem,
            kind=kind,
            path=entry.path,
            shards=list(entry.shards),
            mmproj_path=mmproj,
            mmproj_bytes=_file_size(mmproj),
            size_bytes=entry.size_bytes,
            quant=meta.quant_label,
            publisher=entry.publisher,
            repo=entry.repo,
            architecture=meta.architecture,
            capabilities=capabilities,
            meta=meta,
            mtime=entry.mtime,
            # Stable across rescans: `added_at` surfaces as OpenAI's `created`,
            # and a value that changed every scan would make clients think the
            # model was replaced. Defaulting it to now() did exactly that.
            added_at=entry.mtime,
        )

    def _detect_kind(self, entry: _Entry) -> ModelKind:
        """Embedding/rerank models take a different llama-server flag set, so
        mis-classifying one produces a server that answers nothing useful."""
        haystack = f"{entry.stem} {entry.repo or ''} {entry.publisher or ''}"
        if _RERANK_NAME_RE.search(haystack) or "rerank" in entry.meta.architecture.lower():
            return "rerank"
        meta_says = bool(entry.meta.extra.get("embedding")) or bool(
            entry.meta.extra.get("is_embedding")
        )
        if meta_says or entry.meta.architecture.lower() in _EMBEDDING_ARCHS:
            return "embedding"
        if _EMBEDDING_NAME_RE.search(haystack):
            return "embedding"
        return "chat"

    def _build_adapter(self, entry: _Entry) -> AdapterRecord:
        rank = entry.meta.extra.get("adapter_rank")
        return AdapterRecord(
            id=entry.model_id,
            name=entry.stem,
            path=entry.path,
            size_bytes=entry.size_bytes,
            base_architecture=entry.meta.architecture or None,
            base_model_hint=(
                str(entry.meta.extra["base_model"])
                if entry.meta.extra.get("base_model") is not None
                else None
            ),
            publisher=entry.publisher,
            repo=entry.repo,
            n_layer=entry.meta.n_layer or None,
            rank=int(rank) if isinstance(rank, int) else None,
        )

    # --- settings hydration -------------------------------------------

    def _settings_from(self, model_id: str, saved: dict[str, dict[str, Any]]) -> ModelSettings:
        payload = saved.get(model_id)
        if payload is None:
            return ModelSettings()
        try:
            return ModelSettings.model_validate(payload)
        except ValidationError as exc:
            log.warning("registry.settings_invalid", model_id=model_id, error=str(exc))
            return ModelSettings()

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------

    def _alias_candidates(self, record: ModelRecord) -> list[str]:
        stem = record.name
        aliases = [stem]
        if record.publisher:
            aliases.append(f"{record.publisher}/{stem}")
        if record.repo:
            aliases.append(f"{record.repo}/{stem}")
            trimmed = _REPO_SUFFIX_RE.sub("", record.repo)
            if trimmed and trimmed != record.repo:
                aliases.append(trimmed)
        return [a.lower() for a in aliases if a]

    def _rebuild_aliases(self) -> None:
        """Full ids first, then short forms -- so a short alias can never
        shadow another model's canonical id. First writer wins; the loser is
        logged, never silently dropped."""
        aliases: dict[str, str] = {}
        for model_id in sorted(self._models):
            aliases[model_id.lower()] = model_id
        for model_id in sorted(self._models):
            record = self._models[model_id]
            for alias in self._alias_candidates(record):
                owner = aliases.get(alias)
                if owner is None:
                    aliases[alias] = model_id
                elif owner != model_id:
                    log.warning(
                        "registry.alias_collision",
                        alias=alias,
                        kept=owner,
                        dropped=model_id,
                    )
        self._aliases = aliases

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def all(self) -> list[ModelRecord]:
        with self._lock:
            return sorted(self._models.values(), key=lambda r: r.id)

    def get(self, model_id: str) -> ModelRecord | None:
        with self._lock:
            return self._models.get(model_id)

    def resolve(self, name: str) -> ModelRecord | None:
        """Look up by exact id first, then by any registered alias."""
        key = name.strip()
        with self._lock:
            record = self._models.get(key)
            if record is not None:
                return record
            model_id = self._aliases.get(key.lower())
            return self._models.get(model_id) if model_id else None

    def known_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._models)

    def openai_list(self) -> list[dict[str, Any]]:
        return [record.openai_dict() for record in self.all()]

    @property
    def last_scan_at(self) -> float | None:
        return self._last_scan_at

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def get_settings(self, model_id: str) -> ModelSettings:
        with self._lock:
            record = self._models.get(model_id)
        if record is not None:
            return record.settings
        payload = self._db.get_model_settings(model_id)
        if payload is None:
            return ModelSettings()
        try:
            return ModelSettings.model_validate(payload)
        except ValidationError:
            return ModelSettings()

    def save_settings(self, model_id: str, settings: ModelSettings) -> ModelRecord:
        """Persist per-model settings and update the live record.

        Re-validated through pydantic rather than trusted: this is reachable
        from the HTTP API. ``None`` fields are persisted *as null*, never
        back-filled with the global defaults -- "Auto" must stay a decision
        deferred to load time, when the planner can see actual free VRAM.
        """
        record = self.get(model_id)
        if record is None:
            raise ModelNotFoundError(model_id, known=self.known_ids())
        try:
            validated = ModelSettings.model_validate(settings.model_dump())
        except ValidationError as exc:
            raise BadRequestError(f"invalid settings: {exc}", param="settings") from exc

        # Filesystem checks live here rather than in a pydantic validator so a
        # template file deleted *after* it was saved cannot make the stored row
        # invalid and silently reset the model to defaults. Save time is the
        # moment the user can act on it, so that is where it is enforced -- and
        # doing it in the registry covers every save path (HTTP, MCP, GUI, CLI)
        # rather than one route.
        try:
            validated.chat_template_file = validate_chat_template_file(
                validated.chat_template_file
            )
        except ValueError as exc:
            raise BadRequestError(str(exc), param="chat_template_file") from exc

        self._db.save_model_settings(model_id, validated.model_dump(mode="json"))
        if record.is_virtual and record.base_model_id is not None:
            self._db.save_virtual_model(
                model_id,
                record.base_model_id,
                record.name,
                [a.model_dump() for a in validated.adapters],
                # The preset rides in the same row; dropping it here would make
                # any settings save silently erase the persona.
                preset=(
                    record.preset.model_dump(exclude_none=True)
                    if record.preset is not None
                    else None
                ),
            )
        with self._lock:
            live = self._models.get(model_id)
            if live is not None:
                live.settings = validated
                record = live
        return record

    def touch(self, model_id: str) -> None:
        with self._lock:
            record = self._models.get(model_id)
            if record is not None:
                record.last_used_at = time.time()

    # ------------------------------------------------------------------
    # Adapters
    # ------------------------------------------------------------------

    def _ensure_adapters(self) -> None:
        """Hydrate adapters from SQLite on first read (never in ``__init__``)."""
        with self._lock:
            if self._adapters_loaded:
                return
            loaded: dict[str, AdapterRecord] = {}
            for row in self._db.list_adapters():
                payload = {k: v for k, v in row.items() if k != "added_at"}
                try:
                    record = AdapterRecord.model_validate(payload)
                except ValidationError:  # pragma: no cover - defensive
                    continue
                loaded[record.id] = record
            self._adapters = loaded
            self._adapters_loaded = True

    def scan_adapters(self) -> list[AdapterRecord]:
        """Re-walk the model directories for GGUF LoRA adapters."""
        self.scan()
        return self.adapters()

    def adapters(self) -> list[AdapterRecord]:
        self._ensure_adapters()
        with self._lock:
            return sorted(self._adapters.values(), key=lambda a: a.id)

    def get_adapter(self, adapter_id: str) -> AdapterRecord | None:
        self._ensure_adapters()
        with self._lock:
            return self._adapters.get(adapter_id)

    def delete_adapter(self, adapter_id: str, *, delete_file: bool = False) -> None:
        record = self.get_adapter(adapter_id)
        if record is None:
            raise BadRequestError(f"unknown adapter: {adapter_id!r}", param="adapter_id")
        if delete_file:
            self._assert_inside_model_dirs([record.path])
            record.path.unlink(missing_ok=True)
        self._db.delete_adapter(adapter_id)
        with self._lock:
            self._adapters.pop(adapter_id, None)
        log.info("registry.adapter_deleted", adapter_id=adapter_id, deleted_file=delete_file)

    # ------------------------------------------------------------------
    # Virtual models
    # ------------------------------------------------------------------

    def _load_virtual_models(self, saved_settings: dict[str, dict[str, Any]]) -> None:
        """Rebuild virtual records from SQLite. Caller holds the lock."""
        for row in self._db.list_virtual_models():
            model_id = str(row["id"])
            base = self._models.get(str(row["base_model_id"]))
            if base is None:
                log.warning(
                    "registry.virtual_base_missing",
                    model_id=model_id,
                    base_model_id=row["base_model_id"],
                )
                continue
            attachments = [AdapterAttachment.model_validate(a) for a in row.get("adapters", [])]
            settings = self._settings_from(model_id, saved_settings)
            settings = settings.model_copy(update={"adapters": attachments})
            self._models[model_id] = self._virtual_record(
                model_id,
                str(row["name"] or model_id),
                base,
                settings,
                row.get("created_at"),
                preset=self._preset_from(model_id, row.get("preset")),
            )

    def _preset_from(self, model_id: str, payload: Any) -> VirtualPreset | None:
        """Validate a stored preset payload; an invalid one degrades to None.

        Degrading (with a warning) rather than raising keeps a hand-edited or
        older-schema row from making the whole virtual model disappear -- the
        model still resolves and serves, just without its preset.
        """
        if not isinstance(payload, dict) or not payload:
            return None
        try:
            preset = VirtualPreset.model_validate(payload)
        except ValidationError as exc:
            log.warning("registry.preset_invalid", model_id=model_id, error=str(exc))
            return None
        return None if preset.is_empty() else preset

    def _virtual_record(
        self,
        model_id: str,
        name: str,
        base: ModelRecord,
        settings: ModelSettings,
        created_at: Any = None,
        *,
        preset: VirtualPreset | None = None,
    ) -> ModelRecord:
        return ModelRecord(
            id=model_id,
            name=name,
            kind=base.kind,
            path=base.path,
            shards=list(base.shards),
            mmproj_path=base.mmproj_path,
            size_bytes=base.size_bytes,
            quant=base.quant,
            publisher=base.publisher,
            repo=base.repo,
            architecture=base.architecture,
            capabilities=base.capabilities.model_copy(),
            meta=base.meta,
            settings=settings,
            is_virtual=True,
            base_model_id=base.id,
            preset=preset,
            added_at=float(created_at) if isinstance(created_at, int | float) else time.time(),
            mtime=base.mtime,
        )

    def create_virtual_model(
        self,
        *,
        id: str,
        base_model_id: str,
        name: str | None,
        adapters: list[AdapterAttachment],
        preset: VirtualPreset | None = None,
    ) -> ModelRecord:
        """Register a base + adapter-set (+ optional preset) as its own model name."""
        model_id = id.strip()
        if not model_id:
            raise BadRequestError("virtual model id must not be empty", param="id")
        existing = self.get(model_id)
        if existing is not None and not existing.is_virtual:
            raise BadRequestError(f"id {model_id!r} collides with an existing model", param="id")
        base = self.get(base_model_id)
        if base is None or base.is_virtual:
            raise BadRequestError(f"unknown base model: {base_model_id!r}", param="base_model_id")
        for attachment in adapters:
            if self.get_adapter(attachment.adapter_id) is None:
                raise BadRequestError(
                    f"unknown adapter: {attachment.adapter_id!r}", param="adapters"
                )

        if preset is not None and preset.is_empty():
            preset = None
        display = name or model_id
        self._db.save_virtual_model(
            model_id,
            base.id,
            display,
            [a.model_dump() for a in adapters],
            preset=preset.model_dump(exclude_none=True) if preset is not None else None,
        )
        settings = self._settings_from(model_id, self._db.all_model_settings())
        settings = settings.model_copy(update={"adapters": list(adapters)})
        record = self._virtual_record(model_id, display, base, settings, preset=preset)
        with self._lock:
            self._models[model_id] = record
            self._rebuild_aliases()
        log.info(
            "registry.virtual_created",
            model_id=model_id,
            base_model_id=base.id,
            adapters=len(adapters),
            has_preset=preset is not None,
        )
        return record

    def delete_virtual_model(self, model_id: str) -> None:
        record = self.get(model_id)
        if record is None or not record.is_virtual:
            raise BadRequestError(f"{model_id!r} is not a virtual model", param="model_id")
        self._db.delete_virtual_model(model_id)
        self._db.delete_model_settings(model_id)
        with self._lock:
            self._models.pop(model_id, None)
            self._rebuild_aliases()
        log.info("registry.virtual_deleted", model_id=model_id)

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def _assert_inside_model_dirs(self, paths: Iterable[Path]) -> list[Path]:
        """Resolve and verify containment. Raises on any escape attempt.

        Model ids come from the filesystem, but virtual-model ids and adapter
        ids come from the API; a symlink or a crafted record must never let a
        delete reach outside the configured library roots.
        """
        roots: list[Path] = []
        for model_dir in self._config.model_dirs():
            try:
                roots.append(Path(model_dir).resolve())
            except OSError:  # pragma: no cover
                continue
        checked: list[Path] = []
        for path in paths:
            try:
                resolved = Path(path).resolve()
            except OSError as exc:  # pragma: no cover
                raise BadRequestError(f"cannot resolve {path}: {exc}") from exc
            if not any(resolved.is_relative_to(root) for root in roots):
                raise BadRequestError(
                    f"refusing to touch {resolved}: outside the configured model directories",
                    param="model_id",
                )
            checked.append(resolved)
        return checked

    def delete_model(self, model_id: str, *, delete_files: bool = False) -> list[Path]:
        """Remove a model from the registry, optionally deleting its files.

        Returns the files that were (or would be) removed. The caller is
        responsible for checking that the model is not currently loaded.
        """
        record = self.get(model_id)
        if record is None:
            raise ModelNotFoundError(model_id, known=self.known_ids())

        if record.is_virtual:
            self.delete_virtual_model(model_id)
            return []

        targets = list(record.shards) if record.shards else [record.path]
        if record.mmproj_path is not None and not self._mmproj_shared(model_id, record.mmproj_path):
            targets.append(record.mmproj_path)
        resolved = self._assert_inside_model_dirs(targets)

        if delete_files:
            for path in resolved:
                try:
                    path.unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("registry.delete_failed", path=str(path), error=str(exc))

        with self._lock:
            self._models.pop(model_id, None)
            for virtual_id, virtual in list(self._models.items()):
                if virtual.is_virtual and virtual.base_model_id == model_id:
                    self._models.pop(virtual_id, None)
                    self._db.delete_virtual_model(virtual_id)
            self._rebuild_aliases()
        self._db.delete_model_settings(model_id)
        log.info(
            "registry.model_deleted",
            model_id=model_id,
            files=len(resolved),
            deleted_files=delete_files,
        )
        return resolved

    def _mmproj_shared(self, model_id: str, mmproj: Path) -> bool:
        with self._lock:
            return any(
                other.mmproj_path == mmproj and other.id != model_id and not other.is_virtual
                for other in self._models.values()
            )
