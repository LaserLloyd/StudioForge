"""Read a GGUF header straight off HuggingFace, and turn it into a context matrix.

Two things live here, and the second is the reason for the first.

**Reading the header before the file exists.** The quant picker can only say
"fits / needs multiple GPUs / won't fit" with a *bounded guess* at the KV cache
(``downloader._kv_allowance``), because everything that decides the real cost --
layer count, KV head count, head dimensions, the iSWA pattern, the hybrid
``full_attention_interval``, the trained window -- lives inside the GGUF, and the
GGUF is the 40 GB file the user is trying to decide whether to download. The
metadata block sits in the first few megabytes of it, HuggingFace's CDN honours
``Range``, and :class:`RemoteRangeFile` is the file-like that turns those two
facts into a real answer: ~2-15 MB of range requests (mostly the tokenizer's
string arrays, which have to be walked because they are length-prefixed) instead
of a download, cached on disk so a repo opened twice costs nothing.

**The context matrix.** "Will it fit?" is the wrong question for a rig with four
GPUs; the real one is "how much context do I get, and on how many cards?". So
:func:`context_matrix` answers it as a small table: placement profiles derived
from the actual GPU inventory (one card of the best class, two of them, all of
them) crossed with the context tiers people care about (64k/128k/256k/512k),
each cell decided by the *planner's own arithmetic* -- the same code path a real
load takes, so the picker cannot promise a context the loader then refuses.

Everything in this module degrades rather than fails: a gated repo, an offline
box or a CDN that ignores ``Range`` produces a matrix with ``source: None``, an
``unavailable`` reason and the old bounded allowance, flagged ``approximate``.
An estimate that admits what it does not know is worth more than a confident
wrong one an hour of downloading later.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Final, cast

import httpx

from studioforge.config import Config, KvCacheType
from studioforge.core import gguf as gguf_mod
from studioforge.core.planner import (
    _CTX_LADDER,
    Planner,
    attention_kind,
    effective_kv_bytes_per_token,
)
from studioforge.errors import StudioForgeError
from studioforge.logging import get_logger
from studioforge.types import GB, GpuInfo, ModelCapabilities, ModelRecord

if TYPE_CHECKING:
    from studioforge.types import GgufMeta

log = get_logger(__name__)

__all__ = [
    "CONTEXT_TIERS",
    "MAX_HEADER_BYTES",
    "RemoteHeaderError",
    "RemoteRangeFile",
    "context_line",
    "context_matrix",
    "context_tooltip",
    "geometry_line",
    "open_client",
    "registry_sibling_meta",
    "remote_meta",
    "repo_arch_meta",
]


# ---------------------------------------------------------------------------
# Remote range reader
# ---------------------------------------------------------------------------

#: One range request's worth of bytes. Sized against what the reader actually
#: does: it asks for 4 MiB at a time while parsing and then *seeks* over the
#: numeric arrays, so chunks want to be big enough that a header costs a handful
#: of requests and small enough that a seek past a 20 MB tensor-name table
#: cannot accidentally pull it.
CHUNK_BYTES: Final = 1 << 20

#: Hard ceiling on bytes fetched for one header. A well-behaved GGUF puts its
#: whole metadata block in the first few MB; the worst case in this library is a
#: 250k-token vocabulary whose ``tokenizer.ggml.tokens``/``.merges`` arrays are
#: ~15 MB of length-prefixed strings that must be walked rather than skipped.
#: 64 MiB leaves room for that and still refuses to quietly download a model.
MAX_HEADER_BYTES: Final = 64 << 20

_HTTP_TIMEOUT_S: Final = 30.0

#: Content-Range: bytes 0-1048575/23841456128
_CONTENT_RANGE_RE: Final = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.I)


class RemoteHeaderError(StudioForgeError):
    """The GGUF header could not be read over HTTP.

    Deliberately its own type rather than a bare :class:`OSError`: every caller
    treats it as "degrade to the bounded estimate and say why", and the message
    is written to be shown to a user (or read by an LLM) verbatim.
    """

    status_code = 502
    error_type = "server_error"
    code = "remote_header_unreadable"


class RemoteRangeFile:
    """A seekable, read-only file-like over HTTP ``Range`` requests.

    Enough of the file protocol for :func:`studioforge.core.gguf.read_gguf`'s
    parser: ``read``, ``seek``, ``tell``. Chunks are fetched at
    :data:`CHUNK_BYTES` granularity and cached, which is what makes the parser's
    two access patterns cheap:

    * ``skip()`` over a numeric array becomes a ``seek()`` -- and a seek fetches
      nothing at all, so a 500k-entry ``token_type`` array costs zero bytes;
    * walking the length-prefixed string arrays reads forward through cached
      chunks, one request per megabyte.

    Two failure modes are refused loudly rather than absorbed:

    * a server that answers a ranged request with ``200`` and the whole body --
      streaming a 40 GB file into a "header read" is exactly the accident this
      class exists to avoid, so a non-``206`` answer at a non-zero offset is an
      error (at offset 0 the leading chunk is taken from the stream and the rest
      is dropped);
    * exceeding :data:`MAX_HEADER_BYTES` in total.

    Not thread-safe, and deliberately synchronous: the GGUF parser is
    synchronous, so this runs inside :func:`asyncio.to_thread`.
    """

    def __init__(
        self,
        client: httpx.Client,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        chunk_bytes: int = CHUNK_BYTES,
        max_bytes: int = MAX_HEADER_BYTES,
    ) -> None:
        self._client = client
        self._url = url
        self._headers = dict(headers or {})
        self._chunk = max(4096, int(chunk_bytes))
        self._max_bytes = max(self._chunk, int(max_bytes))
        self._chunks: dict[int, bytes] = {}
        self._pos = 0
        #: Total content length, learned from the first ``Content-Range``.
        self.size: int | None = None
        #: Bytes actually pulled over the wire; the cheap assertion a test (or a
        #: log line) needs to prove a seek did not fetch.
        self.bytes_fetched = 0
        self.requests = 0

    # -- file protocol ---------------------------------------------------

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            target = offset
        elif whence == 1:
            target = self._pos + offset
        elif whence == 2:
            if self.size is None:
                raise RemoteHeaderError("cannot seek from the end: file size is unknown")
            target = self.size + offset
        else:  # pragma: no cover - the stdlib only defines 0/1/2
            raise ValueError(f"invalid whence {whence}")
        if target < 0:
            raise ValueError(f"negative seek position {target}")
        self._pos = target
        return self._pos

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes, never crossing more than one chunk.

        **Deliberately a short read.** The GGUF parser's buffered reader asks
        for 4 MiB at a time and loops until it has what the current field needs,
        so returning one chunk means the wire cost tracks what the parser
        actually *consumes* rather than what it optimistically asked for -- a
        small model's header costs one request, not four.

        ``b""`` at (and past) end of file, which the parser reads as "unexpected
        end of file" -- the right answer for a truncated upload.
        """
        if size is None or size < 0:
            # The GGUF parser never does this (it always asks for a bounded
            # amount); serving it would mean streaming the whole model.
            raise RemoteHeaderError("unbounded read() is not supported over a range reader")
        if size == 0:
            return b""
        index = self._pos // self._chunk
        offset = self._pos - index * self._chunk
        chunk = self._chunk_at(index)
        if not chunk or offset >= len(chunk):
            return b""
        take = chunk[offset : offset + size]
        self._pos += len(take)
        return take

    def close(self) -> None:
        self._chunks.clear()

    # -- fetching --------------------------------------------------------

    def _chunk_at(self, index: int) -> bytes:
        cached = self._chunks.get(index)
        if cached is not None:
            return cached
        start = index * self._chunk
        if self.size is not None and start >= self.size:
            self._chunks[index] = b""
            return b""
        data = self._fetch(start, self._chunk)
        self._chunks[index] = data
        return data

    def _fetch(self, start: int, length: int) -> bytes:
        if self.bytes_fetched + length > self._max_bytes:
            raise RemoteHeaderError(
                f"GGUF header exceeded the {self._max_bytes // (1 << 20)} MiB remote-read cap "
                f"at offset {start}. This file's metadata block is unusually large; download it "
                f"and let the registry read it locally instead."
            )
        headers = dict(self._headers)
        headers["Range"] = f"bytes={start}-{start + length - 1}"
        self.requests += 1
        try:
            with self._client.stream("GET", self._url, headers=headers) as response:
                if response.status_code == 416:
                    self.size = self.size if self.size is not None else start
                    return b""
                if response.status_code >= 400:
                    response.read()
                    raise RemoteHeaderError(
                        f"HuggingFace returned HTTP {response.status_code} for the model header "
                        f"({self._url}). A gated repository needs an access token in config key "
                        f"'hf.token'."
                    )
                if response.status_code != 206:
                    # No Content-Range means the body is the WHOLE file. Taking
                    # the leading chunk from the stream is fine; anything else
                    # would mean re-downloading the model per chunk.
                    if start != 0:
                        raise RemoteHeaderError(
                            "the server ignored the Range header, so the model header cannot be "
                            "read without downloading the whole file"
                        )
                    self._note_length(response)
                buffer = bytearray()
                for part in response.iter_bytes():
                    buffer += part
                    if len(buffer) >= length:
                        break
                if response.status_code == 206:
                    self._note_content_range(response)
        except httpx.HTTPError as exc:
            raise RemoteHeaderError(f"could not read the model header: {exc}") from exc
        data = bytes(buffer[:length])
        self.bytes_fetched += len(data)
        return data

    def _note_content_range(self, response: httpx.Response) -> None:
        match = _CONTENT_RANGE_RE.search(response.headers.get("Content-Range", ""))
        if match and match.group(3) != "*":
            self.size = int(match.group(3))

    def _note_length(self, response: httpx.Response) -> None:
        raw = response.headers.get("Content-Length")
        if raw and raw.isdigit():
            self.size = int(raw)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

#: How long a cached header stays trusted. The metadata of a given *revision*
#: never changes, but ``main`` moves, and a re-quantised re-upload under the
#: same filename is a real thing publishers do.
CACHE_TTL_S: Final = 24 * 3600


@dataclass
class _CacheEntry:
    meta: GgufMeta
    stored_at: float


_MEMORY_CACHE: dict[str, _CacheEntry] = {}


def _cache_key(repo_id: str, revision: str, filename: str) -> str:
    return f"{repo_id}\x00{revision}\x00{filename}"


def cache_dir(config: Config) -> Path:
    """Where parsed remote headers are kept between runs."""
    return Path(config.data_dir) / "cache" / "hf_meta"


def _cache_path(config: Config, key: str) -> Path:
    # Hashed rather than slugified: repo ids and filenames contain characters
    # Windows refuses in a path, and a 200-character quant filename plus a deep
    # data dir blows past MAX_PATH. The key itself is stored inside the file.
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
    return cache_dir(config) / f"{digest}.json"


def _cache_get(config: Config, key: str) -> GgufMeta | None:
    from studioforge.types import GgufMeta

    now = time.time()
    hit = _MEMORY_CACHE.get(key)
    if hit is not None:
        if now - hit.stored_at <= CACHE_TTL_S:
            return hit.meta
        _MEMORY_CACHE.pop(key, None)

    path = _cache_path(config, key)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    # The parser version is part of the freshness check for the same reason the
    # registry's cache key carries it: a header parsed before this module
    # learned to read `full_attention_interval` would keep answering without it
    # forever, and the planner would keep charging Qwen3.5 four times its KV.
    if raw.get("meta_format_version") != gguf_mod.META_FORMAT_VERSION:
        return None
    if now - float(raw.get("stored_at") or 0) > CACHE_TTL_S:
        return None
    try:
        meta = GgufMeta.model_validate(raw["meta"])
    except Exception:  # noqa: BLE001 - a corrupt cache file must not break browsing
        return None
    _MEMORY_CACHE[key] = _CacheEntry(meta=meta, stored_at=float(raw["stored_at"]))
    return meta


def _cache_put(config: Config, key: str, meta: GgufMeta) -> None:
    now = time.time()
    _MEMORY_CACHE[key] = _CacheEntry(meta=meta, stored_at=now)
    path = _cache_path(config, key)
    payload = {
        "key": key,
        "stored_at": now,
        "meta_format_version": gguf_mod.META_FORMAT_VERSION,
        "meta": meta.model_dump(mode="json"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - a read-only data dir must not break browsing
        log.debug("hf_meta cache write failed", path=str(path), error=str(exc))


def clear_memory_cache() -> None:
    """Drop the in-process header cache (tests; a config reload)."""
    _MEMORY_CACHE.clear()


# ---------------------------------------------------------------------------
# Fetching one header
# ---------------------------------------------------------------------------


def _auth_headers(config: Config) -> dict[str, str]:
    headers = {"User-Agent": "studioforge"}
    token = config.hf.token
    if token:
        # Header, never the URL: a token in a query string ends up in every
        # proxy log between here and the CDN.
        headers["Authorization"] = f"Bearer {token}"
    return headers


def open_client() -> httpx.Client:
    """The HTTP client one header read uses.

    A named function rather than an inline constructor so the transport policy
    lives in one place -- and so a test can point the read at a stand-in without
    monkeypatching ``httpx.Client`` globally, which breaks any library that
    evaluates ``httpx.Client | None`` at import time (``huggingface_hub`` does).

    ``follow_redirects`` is not optional: ``/resolve/<rev>/<file>`` is a 302 to
    the CDN, and the CDN is the thing that honours ``Range``.
    """
    return httpx.Client(timeout=httpx.Timeout(_HTTP_TIMEOUT_S), follow_redirects=True)


def _read_remote(
    config: Config, url: str, *, max_bytes: int, chunk_bytes: int = CHUNK_BYTES
) -> tuple[GgufMeta, dict[str, int]]:
    """Blocking half of :func:`remote_meta`: fetch + parse. Runs in a thread."""
    from studioforge.core.gguf import meta_from_gguf

    filename = url.rsplit("/", 1)[-1]
    with open_client() as client:
        remote = RemoteRangeFile(
            client,
            url,
            headers=_auth_headers(config),
            chunk_bytes=chunk_bytes,
            max_bytes=max_bytes,
        )
        try:
            parsed = gguf_mod._read_stream(
                cast(BinaryIO, remote),
                Path(filename),
                load_tensors=False,
                max_array_len=64,
            )
            meta = meta_from_gguf(parsed, path=Path(filename), local=False)
        finally:
            remote.close()
        return meta, {"bytes_fetched": remote.bytes_fetched, "requests": remote.requests}


async def remote_meta(
    config: Config,
    repo_id: str,
    filename: str,
    *,
    revision: str = "main",
    max_bytes: int = MAX_HEADER_BYTES,
    chunk_bytes: int = CHUNK_BYTES,
) -> GgufMeta:
    """Parsed GGUF metadata for a file that is still on HuggingFace.

    Cached in memory and on disk under ``<data_dir>/cache/hf_meta`` for
    :data:`CACHE_TTL_S`, because the quant dialog is opened repeatedly and a
    10 MB re-read per open would make the picker feel broken.

    Raises :class:`RemoteHeaderError` for anything that stops the read: a gated
    repo without a token, an offline box, a mirror that ignores ``Range``, a
    metadata block past the cap. Callers degrade; they never propagate it into a
    browsing failure.
    """
    from studioforge.core.gguf import GgufError
    from studioforge.core.hf_search import file_url

    key = _cache_key(repo_id, revision, filename)
    cached = _cache_get(config, key)
    if cached is not None:
        return cached

    endpoint = _endpoint()
    url = file_url(repo_id, filename, endpoint=endpoint, revision=revision)
    started = time.perf_counter()
    try:
        meta, stats = await asyncio.to_thread(
            _read_remote, config, url, max_bytes=max_bytes, chunk_bytes=chunk_bytes
        )
    except GgufError as exc:
        raise RemoteHeaderError(
            f"the remote header of {repo_id}/{filename} is not readable as GGUF: {exc}"
        ) from exc
    log.debug(
        "hf.remote_header",
        repo_id=repo_id,
        filename=filename,
        kib=stats["bytes_fetched"] // 1024,
        requests=stats["requests"],
        ms=int((time.perf_counter() - started) * 1000),
    )
    _cache_put(config, key, meta)
    return meta


def _endpoint() -> str:
    """The HF origin, honouring ``HF_ENDPOINT`` exactly as ``HfSearch`` does.

    Read per call rather than captured at import: the test suite points a whole
    app at a local stand-in by setting the variable, and a cached value would
    silently send those reads to the real hub.
    """
    from studioforge.core.hf_search import DEFAULT_HF_ENDPOINT

    return (os.environ.get("HF_ENDPOINT") or DEFAULT_HF_ENDPOINT).rstrip("/")


# ---------------------------------------------------------------------------
# Which file to read, and where else the geometry might already be known
# ---------------------------------------------------------------------------


def header_file_for(repo: Any) -> str | None:
    """The one file in a repo worth reading a header from.

    Every quant of a repo shares the geometry the planner needs -- layer count,
    head counts, head dims, the attention pattern, the trained window -- so one
    header answers for all of them. The smallest logical download wins (a small
    file's metadata block is no smaller, but a mis-sized read costs less), and
    for a split model it must be shard 1: llama.cpp writes the full metadata
    only there.
    """
    options = [o for o in repo.logical_models() if o.files]
    if not options:
        return None
    options.sort(key=lambda o: (o.total_bytes or 1 << 62, o.quant))
    first = options[0].files[0]
    return str(first.filename)


def registry_sibling_meta(registry: Any, repo_id: str) -> GgufMeta | None:
    """Metadata from a quant of the same repo that is already downloaded.

    Free and offline: downloads land as ``<models_dir>/publisher/repo/file``,
    so a registered model whose ``publisher``/``repo`` match the HF repo id is
    literally another quant of the model being browsed, and its parsed header
    carries the same geometry. Costs one dictionary walk against a network read
    of several megabytes, so it is tried first.
    """
    if registry is None or "/" not in repo_id:
        return None
    publisher, _, name = repo_id.partition("/")
    publisher = publisher.strip().lower()
    name = name.strip().lower()
    try:
        records = registry.all()
    except Exception:  # noqa: BLE001 - a sick registry must not break browsing
        return None
    for record in records:
        meta = getattr(record, "meta", None)
        if meta is None or getattr(meta, "n_layer", 0) <= 0 or meta.is_mmproj:
            continue
        if (record.publisher or "").strip().lower() != publisher:
            continue
        if (record.repo or "").strip().lower() != name:
            continue
        return cast("GgufMeta", meta)
    return None


@dataclass(frozen=True)
class ArchMeta:
    """The geometry for a repo, plus where it came from and why not."""

    meta: GgufMeta | None = None
    #: ``"registry-sibling"`` / ``"remote-gguf-header"`` / ``None``.
    source: str | None = None
    #: Human-readable reason the header is missing; ``None`` when it is not.
    unavailable: str | None = None


async def repo_arch_meta(
    config: Config,
    repo: Any,
    *,
    registry: Any = None,
    max_bytes: int = MAX_HEADER_BYTES,
) -> ArchMeta:
    """Best available geometry for a repo being browsed, cheapest source first."""
    sibling = registry_sibling_meta(registry, repo.repo_id)
    if sibling is not None:
        return ArchMeta(meta=sibling, source="registry-sibling")

    filename = header_file_for(repo)
    if filename is None:
        return ArchMeta(unavailable="this repo has no loadable GGUF file to read a header from")
    try:
        meta = await remote_meta(config, repo.repo_id, filename, max_bytes=max_bytes)
    except StudioForgeError as exc:
        log.debug("hf.remote_header_failed", repo_id=repo.repo_id, error=exc.message)
        return ArchMeta(unavailable=exc.message)
    except Exception as exc:  # noqa: BLE001 - browsing must survive any transport surprise
        log.debug("hf.remote_header_failed", repo_id=repo.repo_id, error=str(exc))
        return ArchMeta(unavailable=f"{type(exc).__name__}: {exc}")
    return ArchMeta(meta=meta, source="remote-gguf-header")


# ---------------------------------------------------------------------------
# Placement profiles
# ---------------------------------------------------------------------------

#: The context tiers the matrix reports on. Powers of two from 64k because that
#: is where the interesting decisions are: below it everything fits and above it
#: nothing does, and the user asked for exactly these four.
CONTEXT_TIERS: Final[tuple[int, ...]] = (65536, 131072, 262144, 524288)

_NAME_NOISE: Final = re.compile(r"\b(nvidia|geforce|rtx|gtx|tesla|quadro)\b", re.I)


def short_gpu_name(name: str) -> str:
    """``NVIDIA GeForce RTX 5090`` -> ``RTX 5090``; used in profile labels."""
    cleaned = re.sub(r"\b(nvidia|geforce)\b", "", name, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -")
    return cleaned or name


def tiny_gpu_name(name: str) -> str:
    """``NVIDIA GeForce RTX 5090`` -> ``5090``, for the one-line GUI summary.

    The compact line has to fit next to a quant label, so it keeps only the part
    that distinguishes one card from another: the last token carrying a digit.
    """
    cleaned = _NAME_NOISE.sub(" ", name)
    tokens = [t for t in re.split(r"[\s]+", cleaned) if t]
    for token in reversed(tokens):
        if any(ch.isdigit() for ch in token):
            return token[:12]
    return (tokens[-1] if tokens else name)[:12]


@dataclass
class Placement:
    """One candidate way to place a model on this rig."""

    key: str  # single_best | dual_best | all
    devices: tuple[int, ...]
    label: str
    short_label: str
    capacity_bytes: int
    gpus: list[GpuInfo] = field(default_factory=list)


def _gpu_class(gpu: GpuInfo) -> tuple[tuple[int, int], str]:
    return (gpu.compute_capability or (0, 0), gpu.name)


def placements_for(planner: Planner) -> list[Placement]:
    """Placement profiles derived from the rig itself, best card first.

    Three questions worth answering before a download -- "on one good card?",
    "on the pair?", "on everything?" -- resolved against the *real* inventory
    rather than a hard-coded 1/2/4, so a single-GPU box gets one profile and a
    mixed box gets honest labels ("all 4 GPUs (2x RTX 5090 + 2x RTX 3090)").

    Capacity is the idle capacity: total VRAM minus the configured headroom and
    ``reserved_mb``, with excluded devices dropped. What is loaded *right now*
    is deliberately ignored -- the user is asking what this model could do on
    this hardware, not what it could do in the next ten seconds.
    """
    gpus = [g for g in planner.probe.list_gpus() if planner.usable_bytes(g) > 0]
    if not gpus:
        return []
    ranked = sorted(
        gpus, key=lambda g: (-_gpu_class(g)[0][0], -_gpu_class(g)[0][1], -g.total_bytes)
    )
    best_class = _gpu_class(ranked[0])
    same_class = [g for g in ranked if _gpu_class(g) == best_class]

    candidates: list[Placement] = []
    candidates.append(_placement("single_best", [same_class[0]], planner))
    if len(same_class) >= 2:
        candidates.append(_placement("dual_best", same_class[:2], planner))
    if len(gpus) >= 2:
        candidates.append(_placement("all", sorted(gpus, key=lambda g: g.index), planner))

    # Dedupe on the device set, keeping the *last* description of it: on a
    # two-card rig "all 2 GPUs" says more than "2x RTX 5090" (it tells you there
    # is nothing else), while a single card keeps its own label because
    # "all 1 GPUs" is noise.
    by_devices: dict[tuple[int, ...], Placement] = {}
    for candidate in candidates:
        if len(candidate.devices) == 1 and candidate.devices in by_devices:
            continue
        by_devices[candidate.devices] = candidate
    return sorted(by_devices.values(), key=lambda p: (len(p.devices), p.devices))


def _placement(key: str, gpus: Sequence[GpuInfo], planner: Planner) -> Placement:
    devices = tuple(g.index for g in gpus)
    capacity = sum(planner.usable_bytes(g) for g in gpus)
    return Placement(
        key=key,
        devices=devices,
        label=_placement_label(key, gpus),
        short_label=_placement_short_label(key, gpus),
        capacity_bytes=capacity,
        gpus=list(gpus),
    )


def _class_counts(gpus: Sequence[GpuInfo]) -> list[tuple[int, str]]:
    counts: dict[str, int] = {}
    for gpu in gpus:
        name = short_gpu_name(gpu.name)
        counts[name] = counts.get(name, 0) + 1
    return [(count, name) for name, count in counts.items()]


def _placement_label(key: str, gpus: Sequence[GpuInfo]) -> str:
    parts = _class_counts(gpus)
    if key != "all":
        count, name = parts[0]
        return f"{count}x {name}"
    mix = " + ".join(f"{count}x {name}" for count, name in parts)
    plural = "" if len(gpus) == 1 else "s"
    return f"all {len(gpus)} GPU{plural} ({mix})"


def _placement_short_label(key: str, gpus: Sequence[GpuInfo]) -> str:
    if key == "all":
        return "all"
    return f"{len(gpus)}x{tiny_gpu_name(gpus[0].name)}"


class _IdleGpuProbe:
    """The real probe's GPUs, reported as if nothing were loaded on them.

    A pre-download question ("what context would this model get?") must not
    depend on what happens to be loaded at the instant the dialog opened,
    otherwise the same repo answers differently every time. Headroom,
    ``reserved_mb`` and ``excluded_devices`` still apply -- those describe memory
    that is never ours regardless of what is running.

    Local to this module rather than reused from ``catalog._IdleProbe`` on
    purpose: this is a two-line projection, and importing a private helper out of
    the catalog would couple the download picker to the catalog's build cycle.
    """

    def __init__(self, inner: Any) -> None:
        self.backend = getattr(inner, "backend", "unknown")
        self._gpus = [
            g.model_copy(update={"free_bytes": g.total_bytes, "used_bytes": 0})
            for g in inner.list_gpus()
        ]

    def available(self) -> bool:
        return bool(self._gpus)

    def list_gpus(self) -> list[GpuInfo]:
        return [g.model_copy(deep=True) for g in self._gpus]

    def get_gpu(self, index: int) -> GpuInfo | None:
        for gpu in self._gpus:
            if gpu.index == index:
                return gpu.model_copy(deep=True)
        return None

    def compute_processes(self) -> list[Any]:
        return []

    def driver_version(self) -> str | None:
        return None

    def cuda_driver_version(self) -> tuple[int, int] | None:
        return None

    def shutdown(self) -> None:
        return None


def idle_planner(planner: Planner) -> Planner:
    """The same planner, asking about idle GPUs and not logging its plans.

    Idempotent: a planner already built on an idle probe is returned unchanged.
    That matters because building one enumerates the GPUs through NVML, and the
    quant dialog builds a matrix per quant -- eleven NVML enumerations to answer
    one question about one repo would be visible.
    """
    if isinstance(planner.probe, _IdleGpuProbe):
        return planner
    return Planner(planner.config, _IdleGpuProbe(planner.probe), log_plans=False)


# ---------------------------------------------------------------------------
# The context matrix
# ---------------------------------------------------------------------------

#: KV types tried per tier, best first. q4_0 is deliberately absent: halving the
#: cache again is a quality decision the user should make explicitly, not
#: something a picker quietly assumes to make a number look better.
_KV_LADDER: Final[tuple[KvCacheType, ...]] = ("f16", "q8_0")


def context_matrix(
    meta: GgufMeta | None,
    weights_bytes: int,
    *,
    planner: Planner,
    mmproj_bytes: int = 0,
    tiers: Sequence[int] = CONTEXT_TIERS,
    model_id: str = "hf:pending",
    source: str | None = None,
    unavailable: str | None = None,
) -> dict[str, Any]:
    """How much context this download would get, per placement, on this rig.

    ``weights_bytes`` is the base model only; ``mmproj_bytes`` is the projector,
    kept separate because :meth:`Planner.estimate` charges it its own compute
    buffer. Both come from HuggingFace's blob listing and are exact.

    Every cell is decided by :meth:`Planner.fits_on`, i.e. the same
    ``_try_devices`` a real load runs: per-device proportional split viability,
    per-device CUDA context, the real per-layer KV geometry. A picker that used
    its own arithmetic would eventually promise a context the loader refuses,
    and the user would have spent an hour downloading to find out.

    Tiers above the model's trained window are dropped rather than reported as
    failures (D14): 512k on a 262k model is not a memory question.

    Without ``meta`` (gated repo, offline, header too big) the shape is
    identical but ``fits`` comes from the bounded pre-download allowance and
    ``approximate`` is True -- so a client renders the same row either way and
    the caveat travels with the numbers.
    """
    n_ctx_train = int(getattr(meta, "n_ctx_train", 0) or 0) if meta is not None else 0
    usable_tiers = [int(t) for t in tiers if t > 0]
    if n_ctx_train > 0:
        usable_tiers = [t for t in usable_tiers if t <= n_ctx_train]

    idle = idle_planner(planner)
    profiles = placements_for(idle)
    record = _throwaway_record(model_id, meta, weights_bytes, mmproj_bytes)
    approximate = meta is None

    reference_ctx = usable_tiers[-1] if usable_tiers else (n_ctx_train or 8192)
    kv_per_token = 0
    if meta is not None:
        kv_per_token = effective_kv_bytes_per_token(
            meta, kv_k="f16", kv_v="f16", ctx_per_slot=reference_ctx
        )

    placements: list[dict[str, Any]] = []
    for profile in profiles:
        placements.append(
            _placement_row(
                profile,
                record=record,
                meta=meta,
                planner=idle,
                tiers=usable_tiers,
                n_ctx_train=n_ctx_train,
                weights_bytes=weights_bytes + mmproj_bytes,
            )
        )

    return {
        "tiers": usable_tiers,
        "n_ctx_train": n_ctx_train or None,
        "attention_kind": attention_kind(meta) if meta is not None else "unknown",
        "n_layer": int(getattr(meta, "n_layer", 0) or 0) if meta is not None else 0,
        "kv_bytes_per_token_f16": kv_per_token,
        #: The context the per-token figure is quoted at. A sliding-window model
        #: costs less per token the longer the window, so the number is
        #: meaningless without it (see WP1's note on LoadPlan.kv_bytes_per_token).
        "kv_bytes_per_token_ctx": reference_ctx,
        "source": source if meta is not None else None,
        "approximate": approximate,
        "unavailable": unavailable if meta is None else None,
        "placements": placements,
    }


def _throwaway_record(
    model_id: str, meta: GgufMeta | None, weights_bytes: int, mmproj_bytes: int
) -> ModelRecord:
    """A ModelRecord that exists only to be asked a question.

    The planner's arithmetic is entirely a function of the record, so building
    one is how the picker gets *exactly* the loader's answer rather than a
    re-implementation of it. The path is a placeholder and is never touched:
    ``Planner.estimate`` only stats a file when ``mmproj_bytes`` is missing, and
    it is supplied here.
    """
    # The metadata may have come from a *sibling* quant already on disk (the
    # registry path), and its ``tensor_bytes`` is that sibling's size. The
    # planner prefers ``meta.tensor_bytes`` over ``size_bytes`` -- correct for a
    # real record, wrong here: every quant of the repo would be sized as the one
    # we happen to own (a 52 GB BF16 was reported fitting one 32 GB card that
    # way). Pin the metadata's byte count to THIS option's weights.
    if meta is not None and int(getattr(meta, "tensor_bytes", 0) or 0) != int(weights_bytes):
        meta = meta.model_copy(update={"tensor_bytes": max(0, int(weights_bytes))})
    return ModelRecord(
        id=model_id,
        name=model_id,
        path=Path(f"{model_id}.gguf"),
        size_bytes=max(0, int(weights_bytes)),
        meta=meta,
        mmproj_path=Path(f"{model_id}.mmproj.gguf") if mmproj_bytes > 0 else None,
        mmproj_bytes=max(0, int(mmproj_bytes)),
        capabilities=ModelCapabilities(vision=mmproj_bytes > 0),
    )


def _placement_row(
    profile: Placement,
    *,
    record: ModelRecord,
    meta: GgufMeta | None,
    planner: Planner,
    tiers: Sequence[int],
    n_ctx_train: int,
    weights_bytes: int,
) -> dict[str, Any]:
    devices = list(profile.devices)
    fixed = _weights_and_overhead(record, planner, meta, n_devices=len(devices))
    weights_fit = fixed <= profile.capacity_bytes

    fits: dict[str, bool] = {}
    kv_types: dict[str, str] = {}
    for tier in tiers:
        kv_type = _first_kv_type_that_fits(
            record, planner, meta, devices=devices, ctx=tier, budget=profile.capacity_bytes - fixed
        )
        fits[str(tier)] = kv_type is not None
        if kv_type is not None:
            kv_types[str(tier)] = kv_type

    budget = profile.capacity_bytes - fixed
    max_ctx = _max_ctx(
        record,
        planner,
        meta,
        devices=devices,
        kv_type="f16",
        n_ctx_train=n_ctx_train,
        budget=budget,
        tiers=tiers,
    )
    max_ctx_q8 = _max_ctx(
        record,
        planner,
        meta,
        devices=devices,
        kv_type="q8_0",
        n_ctx_train=n_ctx_train,
        budget=budget,
        tiers=tiers,
    )
    return {
        "key": profile.key,
        "label": profile.label,
        "short_label": profile.short_label,
        "devices": devices,
        "capacity_gib": round(profile.capacity_bytes / GB, 1),
        "weights_bytes": int(weights_bytes),
        "weights_fit": weights_fit,
        "fits": fits,
        "kv_cache_type": kv_types,
        "max_ctx": max_ctx,
        # Only reported when the cheaper cache actually buys something.
        "max_ctx_q8": max_ctx_q8 if max_ctx_q8 > max_ctx else None,
    }


def _weights_and_overhead(
    record: ModelRecord, planner: Planner, meta: GgufMeta | None, *, n_devices: int
) -> int:
    """Everything a load costs except the KV cache, for this device count.

    Split out so "won't fit at all" is distinguishable from "fits, but only at
    32k" -- two very different pieces of advice that a single boolean collapses.
    """
    planner_cfg = planner.config.planner
    if meta is not None:
        estimate = planner.estimate(
            record,
            ctx_size=1024,
            parallel=1,
            kv_cache_type="f16",
            kv_cache_type_v="f16",
            n_devices=n_devices,
        )
        return int(estimate.total_bytes - estimate.kv_bytes)
    weights = int(record.size_bytes) + int(record.mmproj_bytes or 0)
    compute = max(
        planner_cfg.compute_overhead_floor_mb * (1 << 20),
        int(weights * planner_cfg.compute_overhead_fraction),
    )
    return weights + compute + planner_cfg.cuda_context_mb * (1 << 20) * max(1, n_devices)


def _first_kv_type_that_fits(
    record: ModelRecord,
    planner: Planner,
    meta: GgufMeta | None,
    *,
    devices: Sequence[int],
    ctx: int,
    budget: int,
) -> str | None:
    """f16 if it fits, else q8_0, else None -- the cell of the matrix."""
    if meta is None:
        from studioforge.core.downloader import _kv_allowance

        # No geometry: the bounded allowance, the same one `fit_verdict` uses.
        # Reported so the row still says something, marked approximate upstream.
        return "f16" if _kv_allowance(int(record.size_bytes), ctx) <= budget else None
    for kv_type in _KV_LADDER:
        fit = planner.fits_on(record, devices=devices, ctx_size=ctx, kv_cache_type=kv_type)
        if fit is not None:
            return kv_type
    return None


def _max_ctx(
    record: ModelRecord,
    planner: Planner,
    meta: GgufMeta | None,
    *,
    devices: Sequence[int],
    kv_type: KvCacheType,
    n_ctx_train: int,
    budget: int,
    tiers: Sequence[int] = (),
) -> int:
    """Largest rung of the planner's own ladder that fits at one slot.

    The planner's ladder rather than the four tiers, because "the expected
    context" is what a load would actually settle on -- and that is exactly the
    ladder ``Planner._context_ladder`` walks. The reported tiers are merged in
    so a 512k-trained model cannot show ``512k OK`` in the table and ``256k`` as
    its maximum (the ladder tops out at 262144). Capped at the trained window: a
    bigger number would be a rope-scaling promise nothing here makes.
    """
    for rung in sorted({*_CTX_LADDER, *tiers}, reverse=True):
        if n_ctx_train and rung > n_ctx_train:
            continue
        if meta is None:
            from studioforge.core.downloader import _kv_allowance

            if _kv_allowance(int(record.size_bytes), rung) <= budget:
                return rung
            continue
        if planner.fits_on(record, devices=devices, ctx_size=rung, kv_cache_type=kv_type):
            return rung
    return 0


# ---------------------------------------------------------------------------
# Rendering (shared by the GUI and anything else that wants one line)
# ---------------------------------------------------------------------------


def format_ctx(value: int) -> str:
    """``131072`` -> ``128k``; ``0`` -> ``--``."""
    if value <= 0:
        return "--"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)


def context_line(matrix: dict[str, Any] | None) -> str:
    """``1x5090: 128k · 2x5090: 256k · all: 256k  (trained 262k)``.

    One line, monospace, per quant row: the largest context each placement can
    actually serve. ``(q8)`` marks a number only the quantised cache reaches,
    ``--`` marks a placement the weights do not even fit on -- which is the
    honest rendering of "this card cannot run this quant at any context".
    """
    if not matrix:
        return ""
    parts: list[str] = []
    for placement in matrix.get("placements") or []:
        label = placement.get("short_label") or placement.get("label") or "?"
        if not placement.get("weights_fit"):
            parts.append(f"{label}: --")
            continue
        max_ctx = int(placement.get("max_ctx") or 0)
        max_q8 = int(placement.get("max_ctx_q8") or 0)
        if max_q8 > max_ctx:
            parts.append(f"{label}: {format_ctx(max_q8)}(q8)")
        else:
            parts.append(f"{label}: {format_ctx(max_ctx)}")
    line = " · ".join(parts)
    trained = matrix.get("n_ctx_train")
    if trained:
        line += f"  (trained {format_ctx(int(trained))})"
    if matrix.get("approximate"):
        line += "  approx"
    return line


def context_tooltip(matrix: dict[str, Any] | None) -> str:
    """The tier ticks, one placement per line, for the hover on that line."""
    if not matrix:
        return ""
    tiers: list[int] = list(matrix.get("tiers") or [])
    lines: list[str] = []
    for placement in matrix.get("placements") or []:
        fits = placement.get("fits") or {}
        kv_types = placement.get("kv_cache_type") or {}
        ticks: list[str] = []
        for tier in tiers:
            key = str(tier)
            mark = "OK" if fits.get(key) else "x"
            if fits.get(key) and kv_types.get(key) not in (None, "f16"):
                mark = f"OK({kv_types[key]})"
            ticks.append(f"{format_ctx(tier)} {mark}")
        capacity = placement.get("capacity_gib")
        suffix = f"  [{capacity} GiB]" if capacity is not None else ""
        if not placement.get("weights_fit"):
            # Says the *reason* rather than a row of crosses. The weights-only
            # badge next to this line can still read "fits one GPU" -- it
            # compares the file against free VRAM and stops there, while this
            # includes the compute buffers and the CUDA context a load also
            # needs. A row of four crosses would leave that difference
            # unexplained.
            body = "weights + overhead do not fit"
        else:
            body = "  ".join(ticks) or "no tier at or below the trained window"
        lines.append(f"{placement.get('label')}: {body}{suffix}")
    if matrix.get("unavailable"):
        lines.append(f"header unavailable: {matrix['unavailable']}")
    elif matrix.get("source"):
        lines.append(f"source: {matrix['source']}")
    return "\n".join(lines)


def geometry_line(matrix: dict[str, Any] | None) -> str:
    """``attention: hybrid · 64 layers · KV 64 KB/token (f16 @ 256k)``."""
    if not matrix or matrix.get("approximate"):
        return ""
    bits: list[str] = []
    kind = matrix.get("attention_kind")
    if kind and kind != "unknown":
        bits.append(f"attention: {kind}")
    layers = int(matrix.get("n_layer") or 0)
    if layers:
        bits.append(f"{layers} layers")
    per_token = int(matrix.get("kv_bytes_per_token_f16") or 0)
    if per_token:
        at = format_ctx(int(matrix.get("kv_bytes_per_token_ctx") or 0))
        bits.append(f"KV {per_token / 1024:.0f} KB/token (f16 @ {at})")
    return " · ".join(bits)


def merge_context_fits(entries: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    """First non-empty ``context_fit`` in a list of quant entries (GUI header)."""
    for entry in entries:
        fit = entry.get("context_fit")
        if fit:
            return cast("dict[str, Any]", fit)
    return None
