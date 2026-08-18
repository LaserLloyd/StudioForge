"""Management-plane MCP server.

This is the *control* surface for an agent (OpenClaw, Claude Code, anything that
speaks MCP): list/inspect/load/unload models, read GPU state, read and edit
configuration, download and delete models. It runs **inside** the main
StudioForge process and talks to the very same objects the HTTP management API
(:mod:`studioforge.api.mgmt_routes`) talks to -- ``state.manager``,
``state.registry``, ``state.supervisor``, ``state.engine_manager`` -- so there
is exactly one implementation of "load a model" and the two control surfaces can
never drift apart. Delegating over localhost HTTP instead would add a
round-trip, a second auth hop and a self-deadlock risk during shutdown for no
behavioural gain.

Three deliberate choices, all of them about the fact that the consumer is a
language model rather than a browser:

**No inference tools.** There is no ``chat``, ``complete`` or ``generate`` tool
and there must never be one. Inference belongs on the OpenAI-compatible
endpoints (``/v1/chat/completions``), which stream, support vision and tools,
and are what every client already implements. Exposing generation through MCP
would mean a second, worse inference API that buffers whole responses through a
JSON-RPC envelope.

**Compact by default, detailed on request.** An MCP client pays context tokens
for every byte a tool returns, and a single GGUF ``chat_template`` is routinely
several thousand tokens. So :func:`list_models` returns a small fixed record set
and even ``model_info`` reports the template's presence and size rather than its
text. The same split runs through the HuggingFace pair: ``search_models``
returns one thin row per repo (no sizes -- HF does not publish them at search
time, so a fit block there could only say "unknown" in twenty different ways),
and ``repo_details`` pays for one remote GGUF header read to answer sizes, fit
and the per-placement context matrix for the one repo that was chosen.

**Errors are results, not exceptions.** Every tool catches
:class:`~studioforge.errors.StudioForgeError` and returns
``{"ok": False, "error": {...}}``. An MCP protocol-level error reaches the model
as an opaque failure; a structured result reaches it as something it can read,
reason about and act on -- "insufficient VRAM, here are the numbers and three
suggestions" is actionable, "tool call failed" is not. Unexpected exceptions are
caught too, and reported with their type, because a traceback that only lands in
the server log is invisible to the agent that caused it.

**Destructive tools require ``confirm=True``.** ``delete_model`` (and any future
destructive tool) refuses without it. The refusal is a normal result explaining
what would happen, so the model can decide to re-call with confirmation instead
of guessing why it failed.

.. note::
   The MCP Python SDK renamed ``FastMCP`` to
   :class:`mcp.server.mcpserver.MCPServer` in 2.0. ``FastMCP`` is re-exported
   here as an alias so call sites and type annotations can keep using the
   familiar name.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from fastapi.concurrency import run_in_threadpool
from mcp.server.mcpserver import MCPServer

from studioforge import __version__
from studioforge.api.auth import redact_config_dict
from studioforge.config import RESTART_REQUIRED_KEYS, Config, apply_overrides, load_config
from studioforge.errors import BadRequestError, ModelNotFoundError, StudioForgeError
from studioforge.logging import get_logger
from studioforge.types import GB

#: Backwards-compatible alias: the SDK's ``FastMCP`` is now ``MCPServer``.
FastMCP = MCPServer

log = get_logger(__name__)

SERVER_NAME = "studioforge"

INSTRUCTIONS = """\
StudioForge management plane: control a GPU-only llama.cpp serving host.

Use these tools to see what models exist, load/unload them, watch VRAM, find
and download new ones, and change configuration.

START WITH list_models. It returns a catalog sorted newest-download-first, and
each model carries a table of loading options. Pass limit=5 when the user means
"the model I just got". Pick the option row marked recommended:true -- the
highest context at or above this server's default context floor, preferring one
that also serves two conversations; the floor is never traded away for a second
slot. Call model_options for the full table when you need a different context
size or more concurrency. Then pass that row's `load_args` object verbatim to
load_model. Every number you need to choose is already in the row: whether it
fits in the VRAM free right now, which GPUs it would use, how many conversations
the placement sustains (max_parallel, and parallel_limited_by saying what caps
it), tokens/second for one stream at an ordinary ~8k of context (est_gen_tps)
and with the window nearly full (est_gen_tps_full_ctx). The model's
attention_kind explains why its context tiers are priced the way they are:
'iswa' and 'hybrid' models keep only a fraction of the window in KV, so their
huge contexts stay cheap. Speeds are estimates unless confidence says otherwise
('measured' = this exact placement was observed, 'calibrated' = corrected by a
learned factor, 'estimated' = nominal hardware numbers, an order of magnitude).

INFERENCE IS NOT HERE. This server exposes no chat/completion/generation tool
by design. To actually run a prompt, use the OpenAI-compatible HTTP API on the
gateway port (POST /v1/chat/completions, /v1/embeddings; GET /v1/models).
Naming an unloaded model in a request just-in-time loads it, so you usually do
not need load_model at all -- reach for it only to pre-warm a model or to load
one with non-default context/quantization settings.

TO GET A NEW MODEL: search_models (compact rows -- repo, popularity, quant
labels; HuggingFace publishes no file sizes at search time, so nothing there
knows what fits) -> repo_details(repo_id) for that repo's real per-quant sizes,
fit verdicts and the exact context each GPU placement reaches ->
download_model(repo_id, quant). The download runs in the background; the model
appears in list_models when it lands.

WHEN VRAM IS MISSING: server_status reports free VRAM per GPU plus who is
holding it. A row whose fits is false but whose if_gpus_idle.fits is true only
needs an unload_model. vram_orphan_count above zero means leaked llama-server
processes are holding VRAM with nothing waiting on them -- the watchdog's
reclaim_orphan_engines tool kills exactly those.

Every tool returns a JSON object. Failures come back as
{"ok": false, "error": {...}} rather than as a protocol error, so read the
error message: it normally tells you exactly what to do next (for example,
which context size would fit in the free VRAM).

Destructive tools (delete_model) require confirm=true and refuse otherwise.
"""


def _error_result(exc: BaseException) -> dict[str, Any]:
    """Render an exception as a readable, machine-parseable tool result."""
    if isinstance(exc, StudioForgeError):
        payload: dict[str, Any] = {
            "message": exc.message,
            "code": exc.code,
            "type": exc.error_type,
        }
        if exc.param:
            payload["param"] = exc.param
        if exc.details:
            payload["details"] = exc.details
        return {"ok": False, "error": payload}
    return {
        "ok": False,
        "error": {
            "message": str(exc) or exc.__class__.__name__,
            "code": "internal_error",
            "type": exc.__class__.__name__,
        },
    }


def _guard[**P](
    func: Callable[P, Awaitable[dict[str, Any]]],
) -> Callable[P, Awaitable[dict[str, Any]]]:
    """Turn any raised exception into an ``{"ok": False, "error": ...}`` result.

    Applied to every tool. ``functools.wraps`` keeps ``__doc__`` and the
    signature intact, which matters because the SDK derives both the tool
    description and its JSON schema from them.
    """

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        try:
            return await func(*args, **kwargs)
        except StudioForgeError as exc:
            log.info("mcp tool rejected", tool=func.__name__, error=exc.message)
            return _error_result(exc)
        except Exception as exc:  # noqa: BLE001 - never surface a protocol error
            log.exception("mcp tool failed", tool=func.__name__, error=str(exc))
            return _error_result(exc)

    return wrapper


def _needs_confirmation(what: str, consequence: str) -> dict[str, Any]:
    return {
        "ok": False,
        "confirmed": False,
        "error": {
            "message": (
                f"Refusing to {what} without confirmation. {consequence} "
                f"Re-call this tool with confirm=true if that is what you want."
            ),
            "code": "confirmation_required",
            "type": "invalid_request_error",
            "param": "confirm",
        },
    }


# ---------------------------------------------------------------------------
# Projections (kept small on purpose -- see the module docstring)
# ---------------------------------------------------------------------------


def _compact_model(record: Any, instance: Any, ttl_default: int | None) -> dict[str, Any]:
    """The **lifecycle** half of a :func:`list_models` row: is it running, for how long.

    Everything a row says about *what the model is* -- quantization, size,
    capabilities -- comes from the catalog entry this is merged into. It used to
    be duplicated here under different spellings (``quant`` beside
    ``quantization``, ``size_gib`` beside ``size_gb``, ``vision``/``tools``
    beside ``capabilities``), which cost every caller tokens to read the same
    fact twice and left an agent guessing which spelling was authoritative. The
    catalog's spelling won; these keys are gone.

    Deliberately excludes ``meta`` (and therefore ``chat_template``) and the
    per-model settings block. Use ``model_info`` when the detail is needed.
    """
    return {
        "id": record.id,
        "kind": record.kind,
        "loaded": instance is not None,
        "state": instance.state if instance is not None else "stopped",
        "port": instance.port if instance is not None else None,
        "pinned": record.settings.pinned,
        "ttl_remaining_s": (
            round(instance.ttl_remaining_s)
            if instance is not None and instance.ttl_remaining_s is not None
            else None
        ),
        "effective_ttl_s": ttl_default,
    }


def _compact_gpu(gpu: Any) -> dict[str, Any]:
    return {
        "index": gpu.index,
        "name": gpu.name,
        "total_gib": round(gpu.total_bytes / GB, 2),
        "free_gib": round(gpu.free_bytes / GB, 2),
        "used_gib": round(gpu.used_bytes / GB, 2),
        "utilization_pct": gpu.utilization_pct,
        "temperature_c": gpu.temperature_c,
        "compute_capability": gpu.cc_str,
    }


def _compact_instance(instance: Any) -> dict[str, Any]:
    return {
        "model_id": instance.model_id,
        "state": instance.state,
        "port": instance.port,
        "pid": instance.pid,
        "devices": list(instance.plan.devices) if instance.plan is not None else [],
        "ctx_size": instance.plan.ctx_size if instance.plan is not None else None,
        "parallel": instance.plan.parallel if instance.plan is not None else None,
        # Context PER conversation, and how many conversations this load was
        # planned for. `ctx_size` alone is ambiguous at parallel > 1, and a
        # client that runs more streams than max_parallel just queues.
        "ctx_per_slot": (
            (instance.plan.ctx_per_slot or instance.plan.ctx_size)
            if instance.plan is not None
            else None
        ),
        "max_parallel": instance.plan.max_parallel if instance.plan is not None else None,
        "parallel_limited_by": (
            instance.plan.parallel_limited_by if instance.plan is not None else None
        ),
        "ttl_s": instance.ttl_s,
        "ttl_remaining_s": (
            round(instance.ttl_remaining_s) if instance.ttl_remaining_s is not None else None
        ),
        "active_requests": instance.active_requests,
        "total_requests": instance.total_requests,
        "last_tokens_per_second": instance.last_tokens_per_second,
        "last_error": instance.last_error,
    }


def _detailed_meta(meta: Any) -> dict[str, Any] | None:
    """GGUF metadata with the chat template replaced by a size summary.

    The template is the single largest field in a model record -- commonly
    2000-6000 tokens of Jinja -- and an agent listing models never needs its
    text, only whether one exists and whether it looks tool-capable.
    """
    if meta is None:
        return None
    data = cast(dict[str, Any], meta.model_dump(mode="json"))
    template = data.pop("chat_template", None)
    data.pop("extra", None)
    data["chat_template_present"] = bool(template)
    data["chat_template_chars"] = len(template) if template else 0
    data["chat_template_supports_tools"] = meta.supports_tools
    return data


def _engine_holders(state: Any) -> dict[str, Any]:
    """Count the llama-server processes on this box by who owns them (D23).

    Reuses ``mgmt_routes._own_child_pids`` and ``vram_holders`` rather than
    re-deriving ownership, so ``/api/status`` and this tool cannot disagree
    about whose process a given pid is -- the disagreement that made the
    2026-08-18 leak unattributable in the first place.

    Never raises: a status call that fails because psutil was denied a process
    is worse than one that answers ``null`` for two counters.
    """
    try:
        from studioforge.api.mgmt_routes import _own_child_pids
        from studioforge.core.vram_holders import find_engine_processes

        processes = find_engine_processes(
            state.config.engines_dir, own_pids=_own_child_pids(state)
        )
    except Exception as exc:  # noqa: BLE001 - attribution is never load-bearing
        log.debug("engine holder scan failed", error=str(exc))
        return {"vram_orphan_count": None, "engine_processes": None}
    # Only the three classes ``find_engine_processes`` can produce are seeded --
    # it looks under our engines directory only, so it never reports a foreign
    # holder (``/api/vram/holders`` is the surface for those). An unseeded class
    # would still be counted, so a new one appears rather than being swallowed.
    counts = {"ours": 0, "child_of_live_process": 0, "orphan": 0}
    for process in processes:
        key = process.classification.replace("-", "_")
        counts[key] = counts.get(key, 0) + 1
    return {"vram_orphan_count": counts["orphan"], "engine_processes": counts}


def _search_row(repo: Any, *, trending: bool) -> dict[str, Any]:
    """One HuggingFace search hit, as small as it can honestly be.

    Search results carry **no file sizes** -- HF's model-list endpoint does not
    report them -- so this row deliberately says nothing about size or fit. The
    previous version reused the repo-detail payload, which meant every quant of
    every hit shipped a full fit block whose every field was a variation on "I
    do not know": ~60 KB of an agent's context for three repos, none of it
    actionable. Sizes and fit come from ``repo_details``, one repo at a time.

    ``trending_score`` is included only when the search was *sorted* by it: HF
    omits the field entirely under any other ordering, so emitting it always
    would report ``None`` as though the repo scored nothing.
    """
    row: dict[str, Any] = {
        "repo_id": repo.repo_id,
        "publisher": repo.publisher,
        # HF's trailing-30-day count, not an all-time total. Same spelling the
        # HTTP payload uses; the meaning is documented on hf_search.SORT_KEYS.
        "downloads": repo.downloads,
        "likes": repo.likes,
    }
    if trending:
        row["trending_score"] = repo.trending_score
    row.update(
        {
            "updated_days_ago": _round_or_none(repo.updated_days_ago),
            "created_days_ago": _round_or_none(repo.created_days_ago),
            # HF answers False, "auto" or "manual"; both strings mean "accept
            # the terms first", which is all a caller can act on.
            "gated": bool(repo.gated),
            "quants": repo.quant_variants,
            "mmproj": bool(repo.mmproj_files),
            "file_count": len(repo.files),
        }
    )
    return row


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(float(value), 1)


#: Placement keys an agent does not read. ``key``/``short_label`` exist for the
#: GUI's layout, ``weights_bytes`` restates the quant's own ``total_gb``, and
#: ``kv_cache_type`` restates the pair ``max_ctx`` (f16) / ``max_ctx_q8``.
_PLACEMENT_DROP = ("key", "short_label", "weights_bytes", "kv_cache_type")


def _compact_repo(payload: dict[str, Any]) -> dict[str, Any]:
    """Trim :func:`_repo_payload` down to what a model choosing a quant needs.

    The HTTP payload is written for the GUI, which renders byte counts, group
    ids and short labels. An agent reads gigabytes and verdicts, so bytes become
    ``total_gb`` (in the same GiB-valued unit as the catalog's ``size_gb``, so
    the two are comparable), the fit block keeps only its four decision fields,
    and the context matrix keeps the per-placement tier table without the
    rendering keys. Measured on the 22-quant ``unsloth/Qwen3.8-27B-GGUF``: 44.5
    KB down to 24.3 KB, i.e. ~1.4 KB per quant, with nothing dropped that
    changes a decision.
    """
    quants: list[dict[str, Any]] = []
    for entry in payload.get("quants", []):
        fit = entry.get("fit") or {}
        compact: dict[str, Any] = {
            "quant": entry.get("quant"),
            "total_gb": round(int(entry.get("total_bytes") or 0) / GB, 2),
            "files": entry.get("files", []),
            "mmproj": entry.get("mmproj"),
            "fit": {
                "verdict": fit.get("verdict"),
                "message": fit.get("message"),
                "suggested_quant": fit.get("suggested_quant"),
                # True == the KV term is a bounded allowance, not this model's
                # real geometry. The word an agent needs before trusting a
                # "just fits".
                "approximate": fit.get("approximate"),
            },
        }
        context = _compact_context_fit(entry.get("context_fit"))
        if context:
            compact["context_fit"] = context
        quants.append(compact)
    return {
        "repo_id": payload.get("repo_id"),
        "publisher": payload.get("publisher"),
        "name": payload.get("name"),
        "downloads": payload.get("downloads"),
        "likes": payload.get("likes"),
        "gated": bool(payload.get("gated")),
        "updated_days_ago": _round_or_none(payload.get("updated_days_ago")),
        "created_days_ago": _round_or_none(payload.get("created_days_ago")),
        "quants": quants,
    }


def _compact_context_fit(matrix: Any) -> dict[str, Any]:
    """The per-placement context table, minus the keys only a renderer wants."""
    if not isinstance(matrix, dict) or not matrix:
        return {}
    placements = []
    for placement in matrix.get("placements", []):
        if not isinstance(placement, dict):
            continue
        placements.append({k: v for k, v in placement.items() if k not in _PLACEMENT_DROP})
    compact: dict[str, Any] = {
        "tiers": matrix.get("tiers", []),
        "n_ctx_train": matrix.get("n_ctx_train"),
        "attention_kind": matrix.get("attention_kind"),
        # "remote-gguf-header" / "registry-sibling" / null. Null means the two
        # honesty flags below are the ones to read.
        "source": matrix.get("source"),
        "placements": placements,
    }
    if matrix.get("approximate"):
        compact["approximate"] = True
    if matrix.get("unavailable"):
        compact["unavailable"] = matrix["unavailable"]
    return compact


def _redacted_config(config: Config) -> dict[str, Any]:
    return redact_config_dict(config.to_yaml_dict())


# ---------------------------------------------------------------------------
# Server construction
# ---------------------------------------------------------------------------


def build_management_mcp(state: Any) -> MCPServer:
    """Build the management MCP server over a composed app state.

    ``state`` is either ``app.state`` of a running FastAPI app or the object
    returned by :func:`studioforge.api.app.build_state`; only the attributes
    ``config``, ``registry``, ``supervisor``, ``manager``, ``engine_manager``
    and ``downloader`` are used, which is what makes this testable with
    stand-ins.
    """
    server: MCPServer = MCPServer(
        name=SERVER_NAME,
        title="StudioForge management",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    # -- models ----------------------------------------------------------

    @_guard
    async def list_models(
        loaded_only: bool = False,
        kind: str | None = None,
        full: bool = False,
        refresh: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """**Start here.** Every model, newest download first, with how to load it.

        This is the catalog: one call tells you what exists, what each model
        is, and -- per model -- a table of loading options with the exact
        arguments to load each one. You should not need any other tool to
        choose a model.

        How to use the result:

        1. Read ``catalog_hint`` once; it explains every column.
        2. Models are ordered by ``downloaded_at``, **newest first**, which is
           usually the order the user thinks in ("the model I just got").
        3. In each model's ``options``, take the row with
           ``recommended: true`` -- the highest context that fits at or above
           this server's default context floor, preferring one that also serves
           two conversations. Only look at another row if your task needs more
           context than it offers, or more concurrency than its
           ``max_parallel``.
        4. Pass that row's ``load_args`` object **verbatim** to ``load_model``.
           Do not modify it, recompute it, or fill in extra fields.

        A row with ``fits: false`` will not load right now. Check its
        ``if_gpus_idle`` block: if that says ``fits: true``, the VRAM exists
        but something else is holding it, and ``unload_model`` on another model
        makes the row available.

        Two speed columns per row, because generation slows as the context
        window fills: ``est_gen_tps`` is one stream at about 8k tokens of
        context (an ordinary turn) and ``est_gen_tps_full_ctx`` is the same
        stream with the window nearly full. The truth is between them. Each
        model's ``attention_kind`` ("full", "iswa", "hybrid") says why its
        context tiers cost what they do -- iSWA and hybrid models keep only a
        fraction of the window in KV, so their huge contexts stay cheap.

        How much to trust a speed: ``confidence`` is ``measured`` (this exact
        placement and context were observed -- ``measured_gen_tps`` is real),
        ``calibrated`` (corrected by a factor learned from observations; the
        model's ``calibration.basis`` says from what -- ``model+devices``,
        ``model``, or ``peers``, meaning other models of the same density on
        the same hardware) or ``estimated`` (nominal vendor bandwidth and
        FLOPS: an order of magnitude, not a promise).

        Args:
            loaded_only: Only models with a running llama-server child.
            kind: Filter by kind -- "chat", "embedding" or "rerank".
            full: Return every context-size row. The default returns only the
                recommended row per model, which is roughly seven times
                smaller and is what you want unless you are comparing context
                sizes.
            refresh: Rebuild instead of serving the few-second cache. Only
                needed right after loading or unloading something, because
                ``fits`` depends on free VRAM.
            limit: Return only the first N models after the other filters.
                Because the ordering is newest-download-first, ``limit=5`` is
                "the five models the user most recently got", which is usually
                what "my new model" means. Omit it to see the whole library.

        Returns:
            ``{"ok": true, "catalog_hint": "...", "models": [...],
            "count": N}``. Every ``id`` is usable verbatim as the ``model``
            field in OpenAI API requests. ``count`` is the number of rows
            returned, so it reflects ``limit``. ``state`` is the child
            process's lifecycle word, kept from before the catalog existed:
            ``"stopped"`` there means simply *not loaded*, not that anything
            failed.
        """
        catalog = await run_in_threadpool(
            state.manager.catalog, compact=not full, refresh=refresh
        )
        loaded = {i.model_id: i for i in state.supervisor.list()}
        rows: list[dict[str, Any]] = []
        for entry in catalog["models"]:
            record = state.registry.get(entry["id"])
            if record is None:
                continue
            if kind is not None and record.kind != kind:
                continue
            instance = loaded.get(record.id)
            if loaded_only and instance is None:
                continue
            # Catalog fields first, the pre-catalog projection second, so on
            # any key they share the OLD meaning wins. `state` is the one that
            # matters: it was the instance's lifecycle word ("stopped",
            # "ready") long before the catalog wanted it for
            # "loaded"/"not-loaded", and silently changing the vocabulary of an
            # existing key is worse than adding a new one.
            rows.append(
                {
                    **entry,
                    **_compact_model(record, instance, state.manager.ttl_for(record)),
                }
            )
            # Applied after the filters, not before: "the five newest chat
            # models" is the question, and truncating the catalog first would
            # answer "whichever of the five newest models happen to be chat".
            if limit is not None and limit >= 0 and len(rows) >= limit:
                break
        return {
            "ok": True,
            "catalog_hint": catalog["catalog_hint"],
            "generated_at": catalog["generated_at"],
            "gpus": catalog["gpus"],
            "models": rows,
            "count": len(rows),
        }

    @_guard
    async def model_options(model_id: str, refresh: bool = False) -> dict[str, Any]:
        """Every loading option for ONE model: all context sizes, with speeds.

        Use this after ``list_models`` when the recommended row is not what you
        want -- for example you need a 262144-token window and the recommended
        row offers 65536, or you need six concurrent conversations and it
        offers two.

        Each row is independent and complete: ``ctx_per_slot`` (context per
        conversation), ``fits`` (right now, on current free VRAM), ``devices``,
        ``kv_cache_type``, ``vram_mb``, ``max_parallel`` with
        ``parallel_limited_by``, estimated and measured tokens/second, and
        ``load_args``.

        Trade-offs the table makes visible, so you can choose deliberately:
        doubling ``ctx_per_slot`` costs ``max_parallel`` (by how much depends
        on the model's ``attention_kind`` -- an "iswa" or "hybrid" model keeps
        only a fraction of the window in KV, so its wide rows stay cheap while
        a "full" model's do not); a ``kv_cache_type`` of q8_0 or q4_0 buys
        context back at some quality cost; and a row spread over more
        ``devices`` is usually slower per token than a single-GPU row.
        ``est_gen_tps`` shows that at ~8k of context, ``est_gen_tps_full_ctx``
        with the window nearly full -- the wider the row, the further those two
        numbers sit apart, and that gap is the real price of the context.

        The row marked ``recommended`` is the highest context at or above this
        server's default context floor, preferring one that also sustains two
        conversations; a second slot is never bought by dropping below the
        floor. Reach past it when your task genuinely needs more.

        Args:
            model_id: Model id or alias, as returned by ``list_models``.
            refresh: Rebuild rather than serve the few-second cache.

        Returns:
            ``{"ok": true, "model": {...with "options": [...]}}``. Pass the
            chosen row's ``load_args`` verbatim to ``load_model``.
        """
        catalog = await run_in_threadpool(
            state.manager.catalog, model=model_id, compact=False, refresh=refresh
        )
        entries = catalog["models"]
        if not entries:
            raise ModelNotFoundError(model_id, known=state.registry.known_ids())
        return {
            "ok": True,
            "catalog_hint": catalog["catalog_hint"],
            "model": entries[0],
        }

    @_guard
    async def model_info(model_id: str) -> dict[str, Any]:
        """Full detail for one model, including what it is *actually* running as.

        Use this before loading a model (to see its trained context length and
        capabilities) or when a loaded model is behaving unexpectedly: the
        ``actual`` block is read live from llama-server's own ``/props``, so it
        shows the real context size and slot count rather than what was
        requested. A mismatch between ``requested`` and ``actual`` is the first
        thing to check when context seems smaller than asked for.

        Args:
            model_id: A model id or alias as returned by ``list_models``.

        Returns:
            Identity, files, capabilities, saved settings, GGUF metadata (the
            chat template is summarised, not included -- it is thousands of
            tokens), plus ``loaded``/``actual``/``slots`` when running.
        """
        record = state.registry.resolve(model_id)
        if record is None:
            raise ModelNotFoundError(model_id, known=state.registry.known_ids())
        instance = state.supervisor.get(record.id)
        payload: dict[str, Any] = {
            "ok": True,
            "id": record.id,
            "name": record.name,
            "kind": record.kind,
            "quant": record.quant,
            "architecture": record.architecture,
            "path": str(record.path),
            "shards": [str(p) for p in record.shards],
            "mmproj_path": str(record.mmproj_path) if record.mmproj_path else None,
            "size_gib": round(record.size_bytes / GB, 2),
            "publisher": record.publisher,
            "repo": record.repo,
            "capabilities": record.capabilities.model_dump(mode="json"),
            "is_virtual": record.is_virtual,
            "base_model_id": record.base_model_id,
            "preset": (
                record.preset.model_dump(mode="json", exclude_none=True)
                if record.preset is not None
                else None
            ),
            "settings": record.settings.model_dump(mode="json"),
            "effective_ttl_s": state.manager.ttl_for(record),
            "meta": _detailed_meta(record.meta),
            "loaded": instance is not None,
        }
        if instance is not None:
            detail = await state.manager.introspect(record.id)
            payload["instance"] = _compact_instance(instance)
            payload["requested"] = detail.get("requested")
            payload["actual"] = detail.get("actual")
            payload["slots"] = detail.get("slots")
        return payload

    @_guard
    async def load_model(
        model_id: str,
        ctx_size: int | None = None,
        kv_cache_type: str | None = None,
        parallel: int | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Load a model into VRAM now, returning once it is serving.

        **Pass a catalog row's ``load_args`` object straight in.** Its four
        keys are exactly this tool's four arguments; that is the intended way
        to call it and it needs nothing added::

            row = list_models()["models"][0]["options"][0]
            load_model(**row["load_args"])

        You often do not need this tool at all: naming an unloaded model in an
        OpenAI API request loads it automatically with planner defaults. Reach
        for it to pre-warm a model, or -- the usual reason -- to load one at a
        specific context size and slot count from the catalog.

        This is GPU-only. If the model cannot fit entirely in VRAM the call
        fails with the required-vs-available byte counts, the largest context
        that *would* fit, and concrete suggestions -- there is no CPU fallback,
        so read those numbers and retry with a smaller ctx_size or a smaller
        kv_cache_type instead of retrying the same call. Better: call
        ``model_options`` and pick a row whose ``fits`` is true.

        Loading may evict other idle models to make room (this is normal and is
        reported in the plan's ``evict_model_ids``).

        Placement (which GPUs, and how the model is split across them) is
        decided by the planner against live free VRAM and is not an argument
        here -- the catalog's ``devices`` column tells you what it will choose.

        Args:
            model_id: Model id or alias.
            ctx_size: Context tokens per parallel slot. None uses the saved
                per-model setting, else the global default.
            kv_cache_type: KV cache quantization -- "f16", "q8_0" or "q4_0"
                (those three only). Each step down roughly halves KV cache
                VRAM at some quality cost. None lets the planner pick the
                best-quality cache that reaches the requested context.
            parallel: Number of concurrent slots, i.e. how many conversations
                this load can serve at once. The engine is launched with
                ctx_size * parallel total context, so each slot really gets
                ctx_size. Match your own client concurrency to this number. A
                catalog row's ``parallel`` is the **most** that placement
                sustains, not a requirement -- asking for fewer is always fine
                and simply leaves VRAM free for something else.
            force: Reload even if the model is already running (use after
                changing its settings).

        Returns:
            The running instance: port, pid, devices, planned VRAM breakdown,
            and the plan's ``max_parallel``.
        """
        instance = await state.manager.load(
            model_id,
            ctx_size=ctx_size,
            kv_cache_type=kv_cache_type,
            parallel=parallel,
            force=force,
        )
        return {
            "ok": True,
            **_compact_instance(instance),
            "plan": instance.plan.model_dump(mode="json") if instance.plan else None,
        }

    @_guard
    async def unload_model(model_id: str) -> dict[str, Any]:
        """Unload a model, freeing its VRAM immediately.

        Graceful: the llama-server child is asked to stop and its process tree
        is confirmed gone, so the VRAM really is released. In-flight requests to
        that model will fail, so prefer letting the idle TTL unload it unless you
        specifically need the VRAM now (for example to load something bigger).

        Args:
            model_id: Model id or alias.

        Returns:
            ``{"ok": true, "unloaded": true}``, or ``unloaded: false`` when the
            model was not running (which is not an error).
        """
        unloaded = await state.manager.unload(model_id)
        return {"ok": True, "model_id": model_id, "unloaded": unloaded}

    @_guard
    async def test_model(model_id: str, prompt: str | None = None) -> dict[str, Any]:
        """Smoke-test a model end to end and report its speed.

        Loads the model if needed, sends one short canned request through it and
        reports latency and tokens/second. This is a *health and performance*
        check, not an inference API -- the response text is truncated and there
        is no streaming, no system prompt and no sampling control. For real
        generation use the OpenAI endpoints.

        Use it to confirm a freshly loaded model actually produces coherent
        output, or to compare tokens/second after changing settings.

        Args:
            model_id: Model id or alias.
            prompt: Optional replacement for the default one-sentence prompt.

        Returns:
            ``latency_s``, ``tokens_per_second``, ``completion_tokens`` and a
            truncated ``text`` (or ``embedding_dims`` for embedding models).
        """
        result = await state.manager.test_model(model_id, prompt)
        return {"ok": True, **result}

    @_guard
    async def download_model(
        repo_id: str,
        quant: str | None = None,
        include_mmproj: bool = True,
    ) -> dict[str, Any]:
        """Queue a GGUF model download from HuggingFace.

        Returns as soon as the download is *queued* -- it proceeds in the
        background, survives a restart, and resumes from its partial file.
        Poll ``server_status`` (``active_downloads``) for a count, and
        ``list_models`` to see the model appear once it lands. Per-file detail
        (percent, MB/s, and on a stumble ``attempt``/``max_attempts``,
        ``retry_in_s`` and ``last_error``) is on ``GET /api/downloads``, which
        is HTTP rather than a tool because it is a polling surface.

        Choose the quant with ``repo_details`` first: it is the only place the
        real size and fit of each quantization are known.

        Args:
            repo_id: HuggingFace repo, e.g. "bartowski/Qwen2.5-7B-Instruct-GGUF".
            quant: Quantization to pick, e.g. "Q4_K_M". When omitted the repo is
                inspected and a sensible default chosen; if several match, the
                available options are listed back to you so you can pick one.
            include_mmproj: Also fetch the vision projector when the repo has
                one. Leave this true for vision models -- without the mmproj file
                the model loads but cannot see images.

        Returns:
            The queued download's id and resolved file list, or an error
            explaining what to fix (unknown repo, ambiguous quant, no HF token
            for a gated repo).
        """
        # Imported lazily and defensively: downloads are an optional subsystem,
        # and the management plane must stay fully usable on a build where it is
        # absent or failed to initialise.
        downloader = getattr(state, "downloader", None)
        if downloader is None:
            return {
                "ok": False,
                "error": {
                    "message": (
                        "Downloads are unavailable on this server: no downloader is "
                        "configured. Copy the GGUF file into the models directory "
                        "manually, then re-scan the library."
                    ),
                    "code": "downloads_unavailable",
                    "type": "invalid_request_error",
                },
            }
        try:
            from studioforge.core.downloader import Downloader  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - optional subsystem
            return {
                "ok": False,
                "error": {
                    "message": (
                        f"Downloads are unavailable on this server: the downloader "
                        f"module could not be imported ({exc}). Copy the GGUF file "
                        f"into the models directory manually, then re-scan."
                    ),
                    "code": "downloads_unavailable",
                    "type": "invalid_request_error",
                },
            }

        # Same resolution the HTTP route uses: repo -> logical models -> quant
        # pick (largest that fits when none is named). Calling enqueue with the
        # raw repo_id/quant kwargs does not match Downloader.enqueue's real
        # signature and broke this tool entirely.
        from studioforge.core import downloader as downloader_module

        chosen = await downloader_module.resolve_download_choice(
            state.config, state.planner, repo_id, quant
        )
        queued = downloader.enqueue(chosen, include_mmproj=include_mmproj)
        if inspect.isawaitable(queued):
            queued = await queued
        group_id = queued if isinstance(queued, str) else _jsonable(queued)
        return {
            "ok": True,
            "queued": {
                "group_id": group_id,
                "repo_id": repo_id,
                "quant": chosen.quant,
                "total_bytes": chosen.total_bytes,
                "files": [f.filename for f in chosen.files],
                "mmproj": chosen.mmproj.filename if chosen.mmproj else None,
            },
        }

    @_guard
    async def search_models(
        query: str,
        limit: int = 10,
        sort: str = "downloads",
        newer_than_days: int | None = None,
        date_field: str = "updated",
        author: str | None = None,
    ) -> dict[str, Any]:
        """Find GGUF repos on HuggingFace. One compact row per repo.

        This is the *browse* step, and it is deliberately thin: a row names the
        repo, how popular and how fresh it is, and which quantization labels it
        publishes. **Sizes and fit are not known at search time** -- HuggingFace's
        model-list endpoint carries no file sizes, so nothing here can say how
        big a quant is or whether it would fit. Do not guess from the label.

        Pick a promising ``repo_id`` and call ``repo_details(repo_id)``: that is
        where real per-quant sizes, fit verdicts and the per-GPU context matrix
        come from. Then ``download_model(repo_id, quant)``.

        Args:
            query: Search text, e.g. "llama" or "qwen". Minimum 1 character.
            limit: Maximum rows to return. Capped at 25 to keep results compact.
            sort: Ordering -- one of ``downloads`` (HuggingFace's TRAILING
                30-DAY count, not an all-time total), ``likes``, ``updated``,
                ``created``, ``trending``.
            newer_than_days: Restrict results to repos changed/created in the
                last N days. When absent, no time restriction is applied.
            date_field: Which date ``newer_than_days`` filters on -- ``updated``
                (default) or ``created``.
            author: Restrict results to one publisher/author name.

        Returns:
            ``{"ok": true, "repos": [...], "truncated": bool, "sort_options":
            [...], "date_field_options": [...]}``. Each row: ``repo_id``,
            ``publisher``, ``downloads``, ``likes``, ``updated_days_ago``,
            ``created_days_ago``, ``gated`` (true means you must accept terms
            and have an HF token), ``quants`` (the labels on offer), ``mmproj``
            (true if the repo ships a vision projector) and ``file_count``.
            ``trending_score`` appears only with ``sort="trending"`` -- HF omits
            the field under every other ordering, so its absence is not a zero.
            ``truncated: true`` means the date window holds more matches than
            were walked; narrow the query or the window rather than trusting
            the tail.
        """
        from studioforge.core.hf_search import DATE_FIELDS, SORT_KEYS, HfSearch

        # Cap limit to avoid bloating context.
        capped_limit = min(limit, 25)
        want_trending = sort.strip().lower() == "trending"

        search = HfSearch(state.config)
        try:
            repos = await search.search(
                query,
                limit=capped_limit,
                author=author,
                sort=sort,
                newer_than_days=newer_than_days,
                date_field=date_field,
            )
            return {
                "ok": True,
                "repos": [_search_row(repo, trending=want_trending) for repo in repos],
                "truncated": search.last_search_truncated,
                "sort_options": list(SORT_KEYS),
                "date_field_options": list(DATE_FIELDS),
            }
        finally:
            await search.aclose()

    @_guard
    async def repo_details(repo_id: str, with_context: bool = True) -> dict[str, Any]:
        """One HuggingFace repo in full: quant sizes, fit, and the context matrix.

        Call this on a ``repo_id`` from ``search_models`` (or one the user named)
        **before** downloading. It reads the model's GGUF header remotely over
        range requests -- a few seconds the first time, then cached -- which is
        what makes the answers exact rather than rules of thumb:

        * ``total_gb`` per quant, from the repo's real blob sizes, in the same
          unit as ``list_models``'s ``size_gb`` so the two compare directly;
        * ``fit`` -- would this quant load on this box at the default context,
          with ``suggested_quant`` naming a smaller one when it would not;
        * ``context_fit`` -- for each GPU placement (one card, a matched pair,
          every card), which of the 64k/128k/256k/512k tiers actually fit.
          ``max_ctx`` is the largest context reachable with a full-quality f16
          KV cache; ``max_ctx_q8`` is present only when a q8_0 cache reaches
          further, so a tier that is ``true`` in ``fits`` but above ``max_ctx``
          is one you would be buying with a quantized cache. Tiers above the
          model's trained window are absent, never offered.

        The matrix is computed with the same planner a real load uses, so it
        cannot promise a context the loader would refuse. ``weights_fit: false``
        on a placement means the weights alone do not fit there, whatever the
        context. ``approximate: true`` (or a non-null ``unavailable``) means the
        header could not be read and the numbers are a bounded estimate.

        Then: ``download_model(repo_id, quant)``. The model appears in
        ``list_models`` once the download finishes.

        Args:
            repo_id: Full HuggingFace repo id, e.g. "unsloth/Qwen3.8-27B-GGUF".
            with_context: Read the header and include ``context_fit``. Set false
                only when you want sizes quickly and do not care about context.

        Returns:
            Repo identity and popularity, plus ``quants``: one entry per
            downloadable quantization with ``quant``, ``total_gb``, ``files``,
            ``mmproj``, ``fit`` and (by default) ``context_fit``.
        """
        from studioforge.api.mgmt_routes import _repo_payload
        from studioforge.core.hf_search import HfSearch

        search = HfSearch(state.config)
        try:
            repo = await search.repo_info(repo_id)
        finally:
            await search.aclose()
        payload = await _repo_payload(state, repo, with_context=with_context)
        return {"ok": True, **_compact_repo(payload)}

    @_guard
    async def delete_model(
        model_id: str,
        delete_files: bool = False,
        confirm: bool = False,
    ) -> dict[str, Any]:
        """Remove a model from the registry, optionally deleting its GGUF files.

        DESTRUCTIVE and requires ``confirm=true``. With ``delete_files=true``
        this permanently deletes multi-gigabyte files from disk; there is no
        undo and no recycle bin. The model must not be loaded -- unload it
        first.

        With ``delete_files=false`` only the registry entry and saved settings
        go away; the next library scan will find the files again. That is the
        safer choice and almost never what you actually want, so say which you
        mean.

        Args:
            model_id: Model id or alias.
            delete_files: Also unlink the GGUF shards and the paired mmproj (a
                shared mmproj used by another model is kept).
            confirm: Must be true. Without it this call does nothing and
                explains what it would have done.

        Returns:
            ``{"ok": true, "removed": [paths]}`` -- the files that were deleted,
            or that *would* be deleted when ``delete_files`` was false.
        """
        if not confirm:
            scope = (
                "This permanently deletes the model's GGUF files from disk."
                if delete_files
                else "This removes the registry entry and saved settings (files are kept)."
            )
            return _needs_confirmation(f"delete model '{model_id}'", scope)
        record = state.registry.resolve(model_id)
        if record is None:
            raise ModelNotFoundError(model_id, known=state.registry.known_ids())
        if state.supervisor.get(record.id) is not None:
            raise BadRequestError(
                f"model '{record.id}' is loaded; call unload_model first",
                code="model_loaded",
            )
        removed = state.registry.delete_model(record.id, delete_files=delete_files)
        return {
            "ok": True,
            "model_id": record.id,
            "files_deleted": delete_files,
            "removed": [str(p) for p in removed],
        }

    # -- status ----------------------------------------------------------

    @_guard
    async def server_status() -> dict[str, Any]:
        """Live snapshot: VRAM, who is holding it, loaded models, queue, engine.

        The first thing to call when deciding whether a model will fit, or when
        something is slow: ``gpus[].free_gib`` is the real constraint on this
        GPU-only host, and ``queue_depth`` shows how many requests are waiting
        on a model that is still loading.

        **When VRAM is missing, read ``engine_processes``.** Every llama-server
        on this box is classified: ``ours`` (started by this server),
        ``child_of_live_process`` (someone else's -- another install, a test run
        -- and never killed automatically), and ``orphan`` (our binary, parent
        gone, pure leak). ``vram_orphan_count`` above zero is recoverable
        memory: the watchdog's ``reclaim_orphan_engines`` tool kills exactly
        those and nothing else. Both counts are ``null`` if the process table
        could not be read, which is not the same as zero.

        Returns:
            Per-GPU VRAM and utilization, every loaded model with its port and
            remaining TTL, the active llama.cpp engine tag, the total number of
            models in the library, queue depth, active downloads, engine-process
            attribution and whether the server is draining for shutdown.
        """
        engine = state.engine_manager.active() if state.engine_manager is not None else None
        downloader = getattr(state, "downloader", None)
        active_downloads = 0
        if downloader is not None:
            try:
                active_downloads = len(downloader.active())
            except Exception:  # noqa: BLE001 - status must never fail on a sub-part
                active_downloads = 0
        status = state.manager.status(engine=engine, active_downloads=active_downloads)
        # Off the event loop: it walks the process table (~7 ms) and, on
        # Windows, may sample a performance counter.
        holders = await run_in_threadpool(_engine_holders, state)
        return {
            "ok": True,
            "version": status.version,
            "uptime_s": round(status.uptime_s, 1),
            "gpus": [_compact_gpu(g) for g in status.gpus],
            "loaded": [_compact_instance(i) for i in status.loaded],
            "model_count": status.model_count,
            "queue_depth": status.queue_depth,
            "active_downloads": status.active_downloads,
            "draining": status.draining,
            "engine_tag": status.engine.tag if status.engine is not None else None,
            "engine_variant": status.engine.variant if status.engine is not None else None,
            "system_ram_total_gib": round(status.system_ram_total_bytes / GB, 2),
            "system_ram_used_gib": round(status.system_ram_used_bytes / GB, 2),
            **holders,
        }

    # -- connectivity ------------------------------------------------------

    @_guard
    async def connection_info(prefer: str = "auto") -> dict[str, Any]:
        """Every address this server can be reached on, best first.

        Use this when you want to move an existing connection somewhere better.
        A tailnet (Tailscale) address is returned first because it keeps working
        across network changes; a LAN address is faster but only valid while you
        are on the same network, so ask for ``prefer="lan"`` when you know you
        are local and want the shortest hop.

        Args:
            prefer: ``"auto"`` (tailnet, then LAN, then loopback), ``"lan"`` for
                a direct local address, ``"tailscale"``, or ``"loopback"``.

        Returns the recommended URL plus every alternative, the OpenAI base URL
        for inference, and whether a pairing PIN is required.
        """
        from studioforge.core.netinfo import reachable_urls

        config: Config = state.config
        mcp_path = getattr(getattr(config, "mcp", None), "path", "/mcp")
        endpoints = reachable_urls(config.server.port, mcp_path, host=config.server.host)
        api = reachable_urls(config.server.port, "/v1", host=config.server.host)

        buckets: dict[str, list[dict[str, str]]] = {"tailscale": [], "lan": [], "loopback": []}
        for entry in endpoints:
            kind = entry["kind"]
            buckets.setdefault("lan" if kind == "bound" else kind, []).append(entry)

        wanted = prefer.strip().lower()
        if wanted in buckets and buckets[wanted]:
            chosen = buckets[wanted][0]
        else:
            chosen = next(
                (e for group in ("tailscale", "lan", "loopback") for e in buckets[group]),
                endpoints[0] if endpoints else {},
            )
            if wanted not in {"auto", ""} and wanted in buckets:
                # Asked for something we do not have; say so rather than
                # silently handing back a different network.
                return {
                    "ok": True,
                    "recommended": chosen.get("url"),
                    "requested": wanted,
                    "note": f"no {wanted} address is available on this host",
                    "endpoints": endpoints,
                    "tailscale": buckets["tailscale"],
                    "lan": buckets["lan"],
                    "loopback": buckets["loopback"],
                }

        mcp_cfg = getattr(config, "mcp", None)
        return {
            "ok": True,
            "recommended": chosen.get("url"),
            "recommended_kind": chosen.get("kind"),
            "endpoints": endpoints,
            "tailscale": buckets["tailscale"],
            "lan": buckets["lan"],
            "loopback": buckets["loopback"],
            "openai_base_urls": [e["url"] for e in api],
            "pin_required": bool(
                getattr(mcp_cfg, "pin_required", False) and getattr(mcp_cfg, "pin", None)
            ),
            "watchdog_port": config.watchdog.port,
            "note": (
                "Tailscale addresses survive network changes; LAN addresses are "
                "faster but only valid on the same network."
            ),
        }

    # -- config ----------------------------------------------------------

    @_guard
    async def get_config() -> dict[str, Any]:
        """Read the effective configuration, with secrets redacted.

        ``server.api_key``, ``mcp.pin`` and ``hf.token`` are never returned in
        full -- only a short fingerprint -- so this is safe to read into a
        transcript. Use it to discover the exact dotted key names that
        ``set_config`` accepts.

        Returns:
            The whole config tree, the path of the file it came from, and the
            list of keys that only take effect after a restart.
        """
        config = state.config
        return {
            "ok": True,
            "config": _redacted_config(config),
            "config_path": str(config.config_path),
            "restart_required_keys": sorted(RESTART_REQUIRED_KEYS),
        }

    @_guard
    async def set_config(updates: dict[str, Any]) -> dict[str, Any]:
        """Change configuration values, validate them, and persist to disk.

        Takes dotted paths exactly as ``get_config`` reports them, for example
        ``{"models.default_ctx": 16384, "planner.headroom_fraction": 0.15}``.
        The whole update is validated as one unit against the real schema: if
        any value is invalid *nothing* is written, so a bad call cannot leave a
        half-applied config behind.

        Most changes apply immediately to the running server. Ports, hosts and
        the data directory cannot -- those are listed in ``restart_required``,
        and you need the watchdog's ``restart_server`` tool (or a manual
        restart) for them to take effect.

        Args:
            updates: Mapping of dotted config path to new value. An unknown key
                is an error, not a silent no-op.

        Returns:
            ``{"ok": true, "updated": [keys], "restart_required": [keys]}``.
        """
        if not isinstance(updates, dict) or not updates:
            raise BadRequestError(
                "updates must be a non-empty object of dotted config paths, e.g. "
                '{"models.default_ctx": 8192}',
                param="updates",
            )
        config: Config = state.config
        updated = apply_overrides(config, updates)
        updated.save()
        changed = sorted(updates)
        needs_restart = [key for key in changed if key in RESTART_REQUIRED_KEYS]

        # Mutate the shared config in place for the sections that are safe to
        # swap live; every component holds the same Config by reference, which
        # is what makes a change take effect without a restart. Mirrors
        # mgmt_routes.set_config deliberately -- both surfaces must behave the
        # same way.
        for section in ("models", "planner", "gateway", "hf", "logging", "update", "engine"):
            setattr(config, section, getattr(updated, section))
        config.server.api_key = updated.server.api_key
        config.server.cors_origins = updated.server.cors_origins
        config.server.drain_timeout_s = updated.server.drain_timeout_s
        config.server.request_timeout_s = updated.server.request_timeout_s

        log.info("config updated via mcp", keys=changed, restart_required=needs_restart)
        return {
            "ok": True,
            "updated": changed,
            "restart_required": needs_restart,
            "config_path": str(updated.config_path),
        }

    for tool in (
        connection_info,
        list_models,
        model_options,
        model_info,
        load_model,
        unload_model,
        search_models,
        repo_details,
        download_model,
        delete_model,
        server_status,
        test_model,
        get_config,
        set_config,
    ):
        server.add_tool(tool)

    return server


def _jsonable(value: Any) -> Any:
    """Best-effort JSON projection of a foreign object (e.g. a download record).

    The downloader is written by another component and may hand back a pydantic
    model, a dataclass or a plain dict; MCP results must be JSON-serializable
    regardless.
    """
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    return str(value)


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


def mount_management_mcp(app: Any, state: Any, *, path: str = "/mcp") -> None:
    """Mount the management MCP on an existing FastAPI app as streamable HTTP.

    The routes are appended to the parent app's router rather than mounted as a
    sub-application: a Starlette ``Mount`` would put the endpoint at
    ``<path><path>`` (or force a trailing-slash redirect that MCP POSTs do not
    follow cleanly), and it would not run the sub-app's lifespan. Instead the
    streamable-HTTP session manager -- which needs a live task group for the
    whole server lifetime -- is started by wrapping the app's existing lifespan.

    Mounting also means the MCP endpoint sits behind StudioForge's own
    ``AuthMiddleware``, so it uses the same ``server.api_key`` as everything
    else instead of a second credential scheme.
    """
    import contextlib
    from collections.abc import AsyncIterator

    server = build_management_mcp(state)
    host = getattr(getattr(state, "config", None), "server", None)
    bind_host = getattr(host, "host", "0.0.0.0")

    sub_app = server.streamable_http_app(streamable_http_path=path, host=bind_host)
    session_manager = server.session_manager

    for route in sub_app.routes:
        app.router.routes.append(route)

    previous = app.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scoped_app: Any) -> AsyncIterator[Any]:
        async with session_manager.run(), previous(scoped_app) as maybe_state:
            yield maybe_state

    app.router.lifespan_context = lifespan
    app.state.management_mcp = server
    log.info("management MCP mounted", path=path)


def run_stdio(config_path: Path | None = None) -> None:
    """Run the management MCP over stdio (one client, local, no HTTP).

    This composes a *complete* StudioForge stack in-process -- registry,
    planner, supervisor -- so it can load and unload models on its own. It is
    the "run it from an MCP client config with a command line" shape; the
    always-on server exposes the same tools over HTTP via
    :func:`mount_management_mcp`.

    Logging goes to stderr and the log file only: stdout is the JSON-RPC
    channel, and a single stray print there corrupts the protocol.
    """
    import asyncio

    from studioforge.api.app import build_state
    from studioforge.logging import configure_logging

    config = load_config(config_path, create=True)
    configure_logging(
        config.logging.level, json_logs=config.logging.json_logs, log_dir=config.logs_dir
    )
    state = build_state(config)
    server = build_management_mcp(state)
    asyncio.run(server.run_stdio_async())
