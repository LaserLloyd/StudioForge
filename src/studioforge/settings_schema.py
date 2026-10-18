"""The settings schema registry: one row per ``config.yaml`` key, derived from the model.

This is the single source of truth for P4 ("discoverable settings"). It walks
:class:`studioforge.config.Config` -- the pydantic model that already owns every
key, default and range -- and yields, for each key: dotted name, type, default,
description, valid range / choices, Basic/Advanced level, environment override
name, whether a change needs a restart, and whether the value is a secret.

It feeds, and is checked against, three outputs:

* ``GET /api/settings/schema`` (the registry as JSON, for the panel and the CLI);
* ``docs/contracts/settings-schema.json`` (the frozen contract);
* ``docs/reference/settings.md`` (:func:`render_markdown`).

Nothing here is hand-listed key by key. A new field on a config model appears in
all three on the next ``scripts/gen_docs.py`` run; the drift tests fail until it
does, and :func:`missing_descriptions` fails until it is explained.

Where descriptions come from, in priority order:

1. the comment block directly above the key in ``config.example.yaml``;
2. the annotated key list in that file's header;
3. the comment directly above the field in ``config.py`` (first paragraph);
4. :data:`_FALLBACK_DESCRIPTIONS`, for the keys nobody had written a comment for.

Levels: ``basic`` only for :data:`BASIC_KEYS`; everything else is ``advanced``.

Environment overrides follow the model's own rule (``env_prefix="SF_"``,
``env_nested_delimiter="__"``): ``server.port`` is ``SF_SERVER__PORT``. They are
first-run values only -- once ``config.yaml`` names a key, the file wins
(``config.py`` module docstring).
"""

from __future__ import annotations

import ast
import json
import re
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import to_jsonable_python

from studioforge import config as config_module
from studioforge.config import (
    CONFIG_SCHEMA_VERSION,
    RESTART_REQUIRED_KEYS,
    SECRET_CONFIG_PATHS,
    Config,
)

#: Keys shown in the Basic view. Everything else is Advanced.
#: ``gateway TTLs`` is read as the idle-unload clocks the gateway enforces:
#: the per-tier model TTLs and the sweep that applies them.
BASIC_KEYS: frozenset[str] = frozenset(
    {
        "server.host",
        "server.port",
        "server.api_key",
        "gui.port",
        "models.dir",
        "engine.pinned_tag",
        "models.default_ttl_s",
        "models.ttl_by_priority",
        "gateway.ttl_sweep_interval_s",
        "update.repo",
    }
)

#: Sections ``PATCH /api/config`` does NOT re-apply to the running config
#: (``api/mgmt_routes.py``, the live-apply loop): a change there only takes
#: effect after a restart.
_NOT_LIVE_SECTIONS = frozenset({"gui", "watchdog", "leases", "tenants", "benchmark"})

#: Engine keys that choose WHICH build serves: changing them needs the engine
#: re-activated (and the models reloaded), so they are flagged like a restart.
_ENGINE_IDENTITY_KEYS = frozenset({"engine.pinned_tag", "engine.cuda_variant"})

_CONSTRAINT_ATTRS = (
    "gt",
    "ge",
    "lt",
    "le",
    "min_length",
    "max_length",
    "multiple_of",
    "pattern",
)

_CONFIG_FILE = Path(config_module.__file__)
_REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = _REPO_ROOT / "config.example.yaml"


@dataclass(frozen=True)
class Setting:
    key: str
    section: str
    type: str
    default: Any
    description: str
    level: str
    env: str
    restart_required: bool
    secret: bool = False
    #: ``{"ge": 1, "le": 65535}``-style bounds the model already enforces.
    constraints: dict[str, Any] = field(default_factory=dict)
    #: Allowed literal values, when the type is (or includes) a ``Literal``.
    choices: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Walking the model
# ---------------------------------------------------------------------------


def _strip_annotated(annotation: Any) -> tuple[Any, list[Any]]:
    """``Annotated[T, m1, m2]`` -> ``(T, [m1, m2])`` (metadata flattened, FieldInfo opened)."""
    metadata: list[Any] = []
    while get_origin(annotation) is typing.Annotated:
        base, *extra = get_args(annotation)
        for item in extra:
            if isinstance(item, FieldInfo):
                metadata.extend(item.metadata)
            else:
                metadata.append(item)
        annotation = base
    return annotation, metadata


def _union_members(annotation: Any) -> list[Any] | None:
    if get_origin(annotation) in (Union, types.UnionType):
        return list(get_args(annotation))
    return None


def _type_name(annotation: Any) -> str:
    """A short, stable, human-readable type: ``int``, ``string | null``, ``list[string]``."""
    annotation, _ = _strip_annotated(annotation)
    if annotation is type(None) or annotation is None:
        return "null"
    members = _union_members(annotation)
    if members is not None:
        names = [_type_name(member) for member in members]
        # `null` last, deduplicated, original order otherwise.
        seen: list[str] = []
        for name in names:
            if name not in seen:
                seen.append(name)
        seen.sort(key=lambda n: n == "null")
        return " | ".join(seen)
    origin = get_origin(annotation)
    if origin is Literal:
        return " | ".join(json.dumps(value) for value in get_args(annotation))
    if origin in (list, set, frozenset, tuple):
        args = get_args(annotation)
        return f"list[{_type_name(args[0])}]" if args else "list"
    if origin is dict:
        args = get_args(annotation)
        return f"map[{_type_name(args[0])}, {_type_name(args[1])}]" if len(args) == 2 else "map"
    if isinstance(annotation, type):
        if issubclass(annotation, bool):
            return "bool"
        if issubclass(annotation, int):
            return "int"
        if issubclass(annotation, float):
            return "float"
        if issubclass(annotation, str):
            return "string"
        if issubclass(annotation, Path):
            return "path"
        return annotation.__name__
    return str(annotation)


def _collect_metadata(annotation: Any, field_info: FieldInfo) -> tuple[dict[str, Any], list[Any]]:
    """Range constraints and Literal choices, wherever in the annotation they sit."""
    constraints: dict[str, Any] = {}
    choices: list[Any] = []

    def absorb(metadata: list[Any]) -> None:
        for item in metadata:
            for attr in _CONSTRAINT_ATTRS:
                value = getattr(item, attr, None)
                if value is not None:
                    constraints[attr] = value

    def visit(node: Any) -> None:
        base, nested = _strip_annotated(node)
        absorb(nested)
        members = _union_members(base)
        if members is not None:
            for member in members:
                visit(member)
        elif get_origin(base) is Literal:
            for value in get_args(base):
                if value not in choices:
                    choices.append(value)

    absorb(list(field_info.metadata))
    visit(annotation)
    return constraints, choices


def _jsonable(value: Any) -> Any:
    """JSON-safe default with string keys everywhere (deterministic across Python versions)."""
    data = to_jsonable_python(value)

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(k): fix(v) for k, v in node.items()}
        if isinstance(node, list):
            return [fix(v) for v in node]
        return node

    return fix(data)


def _is_section(annotation: Any) -> type[BaseModel] | None:
    base, _ = _strip_annotated(annotation)
    if isinstance(base, type) and issubclass(base, BaseModel):
        return base
    return None


def _key_name(name: str, info: FieldInfo) -> str:
    """The key as written in config.yaml (the alias, e.g. ``logging.json``)."""
    return info.serialization_alias or info.alias or name


def _env_name(key: str, attr_path: str, info: FieldInfo) -> str:
    """``SF_`` + the dotted key, ``.`` -> ``__``, upper case.

    The *name the loader reads*: for an aliased key (``logging.json``) that is the
    alias; for ``schema`` it is the attribute (``schema_version``) -- the key has
    only a serialization alias precisely so a bare ``SCHEMA`` variable is not read.
    """
    return "SF_" + (key if info.alias else attr_path).replace(".", "__").upper()


def _walk(
    model: type[BaseModel], prefix: str = "", attr_prefix: str = ""
) -> list[tuple[str, str, FieldInfo, Any]]:
    """``(dotted_key, dotted_attr_path, field_info, annotation)`` for every leaf key."""
    rows: list[tuple[str, str, FieldInfo, Any]] = []
    for name, info in model.model_fields.items():
        if info.exclude or name in Config.UNPERSISTED_KEYS:
            continue
        key = f"{prefix}{_key_name(name, info)}"
        attr = f"{attr_prefix}{name}"
        section = _is_section(info.annotation)
        if section is not None:
            rows.extend(_walk(section, f"{key}.", f"{attr}."))
        else:
            rows.append((key, attr, info, info.annotation))
    return rows


# ---------------------------------------------------------------------------
# Descriptions
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """One line: RST ``double backticks`` become `single`, runs of space collapse,
    and the first letter is capitalised ("the context FLOOR" -> "The context FLOOR")."""
    text = re.sub(r"\s+", " ", text.replace("``", "`")).strip()
    return text[:1].upper() + text[1:] if text[:1].isalpha() else text


def _first_paragraph(lines: list[str], limit: int = 280) -> str:
    """The first paragraph of a comment block, cut at a sentence end under ``limit``."""
    paragraph: list[str] = []
    for line in lines:
        if not line.strip():
            if paragraph:
                break
            continue
        paragraph.append(line)
    text = _clean(" ".join(paragraph))
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("; "))
    if end >= 80:
        return cut[: end + 1].rstrip(";")
    return cut.rsplit(" ", 1)[0].rstrip(",;:") + "..."


def _example_comments(text: str) -> tuple[dict[str, str], dict[str, str]]:
    """``(header, block)`` descriptions parsed from ``config.example.yaml``.

    ``header`` is the annotated key list at the top of the file (``#   models.dir
    your GGUF library...`` with indented continuation lines); ``block`` is the
    comment directly above a key in the YAML body.
    """
    lines = text.splitlines()
    body_start = next(
        (i for i, line in enumerate(lines) if line.strip() and not line.startswith("#")),
        len(lines),
    )

    header: dict[str, list[str]] = {}
    current: str | None = None
    for line in lines[:body_start]:
        entry = re.match(r"#   ([a-z][a-z0-9_]*\.[a-z0-9_.]+)\s{2,}(\S.*)", line)
        if entry:
            current = entry.group(1)
            header[current] = [entry.group(2)]
        elif current and re.match(r"#\s{10,}\S", line):
            header[current].append(line.lstrip("#").strip())
        else:
            current = None

    block: dict[str, list[str]] = {}
    section: str | None = None
    pending: list[str] = []
    for line in lines[body_start:]:
        stripped = line.strip()
        if stripped.startswith("#"):
            pending.append(stripped.lstrip("#").strip())
            continue
        match = re.match(r"([A-Za-z_]\w*):(?:\s|$)", stripped)
        indent = len(line) - len(line.lstrip())
        if match and indent == 0:
            section = match.group(1)
            if pending:
                block[section] = pending
        elif match and indent == 2 and section:
            if pending:
                block[f"{section}.{match.group(1)}"] = pending
        pending = []
    return (
        {key: _clean(" ".join(value)) for key, value in header.items()},
        {key: _clean(" ".join(value)) for key, value in block.items()},
    )


def _source_comments() -> dict[str, str]:
    """``{"section.key": first paragraph}`` from the comments above fields in ``config.py``."""
    source = _CONFIG_FILE.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    out: dict[str, str] = {}

    def comment_above(lineno: int) -> list[str]:
        collected: list[str] = []
        i = lineno - 2
        while i >= 0 and lines[i].strip().startswith("#"):
            collected.append(re.sub(r"^\s*#:?\s?", "", lines[i]))
            i -= 1
        collected.reverse()
        trailing = re.search(r"\s#\s+(.*)$", lines[lineno - 1])
        if not collected and trailing:
            collected = [trailing.group(1)]
        return collected

    for section_name, info in Config.model_fields.items():
        model = _is_section(info.annotation)
        if model is None or model.__name__ not in classes:
            continue
        for stmt in classes[model.__name__].body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                field_info = model.model_fields.get(stmt.target.id)
                if field_info is None:
                    continue
                key = f"{section_name}.{_key_name(stmt.target.id, field_info)}"
                comment = comment_above(stmt.lineno)
                if comment:
                    out[key] = _first_paragraph(comment)
    return out


#: Keys that carry no comment anywhere. One line each: what it is and how it is used.
_FALLBACK_DESCRIPTIONS: dict[str, str] = {
    "schema": "Layout version of config.yaml. Older files without it load as version 1.",
    "server.host": "Address the OpenAI-compatible gateway (/v1, /api, /mcp) binds to. "
    "127.0.0.1 (the default) keeps it on this machine; 0.0.0.0 listens on every interface "
    "and requires an owner account or admin key.",
    "server.port": "TCP port of the gateway. 1234 is LM Studio's port, so pointing an existing "
    "client here is a host change.",
    "server.api_key": "Deprecated shared API key (at least 6 characters), mirrored into the "
    "key table as the admin key 'legacy'. Prefer scoped keys (`studioforge keys create`).",
    "server.cors_origins": "Origins allowed to call the API from a browser, and to reach the "
    "MCP endpoints from one. Empty (the default) allows none.",
    "server.cors_allow_credentials": "Let browsers send credentials cross-origin. Requires "
    'explicit cors_origins (no "*").',
    "server.tls.mode": "HTTPS for the gateway, panel and watchdog: off, self_signed (a "
    "certificate generated into the data directory, fingerprint shown in the panel) or cert "
    "(your own PEM files).",
    "server.tls.cert_file": "PEM certificate (chain) served when server.tls.mode is cert.",
    "server.tls.key_file": "PEM private key served when server.tls.mode is cert.",
    "server.drain_timeout_s": "Seconds to let in-flight requests finish when the server "
    "shuts down.",
    "server.request_timeout_s": "Seconds a single upstream request may take before it is "
    "abandoned with a 504.",
    "gui.enabled": "Serve the control panel.",
    "gui.host": "Address the control panel binds to.",
    "gui.port": "TCP port of the control panel.",
    "gui.refresh_interval_s": "How often the panel refreshes its live views, in seconds.",
    "watchdog.enabled": "Run the recovery watchdog sidecar (health, restart, kill tools).",
    "watchdog.host": "Address the watchdog binds to.",
    "watchdog.port": "TCP port of the watchdog (health, restart and its MCP endpoint).",
    "watchdog.poll_interval_s": "Seconds between the watchdog's own health probes.",
    "watchdog.health_timeout_s": "Seconds the watchdog waits for the server's /health before "
    "counting a failure.",
    "watchdog.wedged_after_failures": "Consecutive failed probes before the server is called "
    "wedged.",
    "mcp.enabled": "Serve the management MCP endpoint on the gateway.",
    "mcp.path": "URL path the MCP endpoint is mounted at.",
    "models.dir": "Primary directory of GGUF models. Left empty, first run looks for an "
    "existing LM Studio library.",
    "models.extra_dirs": "Further directories scanned for models, after models.dir.",
    "models.default_ctx": "The context floor: the planner never recommends less than this.",
    "models.default_ttl_s": "Idle seconds before a loaded model is unloaded, when no tier "
    "TTL applies.",
    "models.default_kv_cache_type": "KV cache precision for loads that do not set one "
    "(auto picks per model).",
    "models.default_parallel": 'Conversation slots per model: "auto" sizes them per model '
    "and placement; an integer is used as given.",
    "models.ctx_per_slot_default": "Context per slot the estimator assumes when sizing slots.",
    "models.default_flash_attn": "Flash attention for loads that do not set it (on, off or auto).",
    "models.default_cache_reuse": "Minimum chunk size for KV cache reuse between requests "
    "(llama.cpp --cache-reuse).",
    "models.auto_ctx_checkpoints": "Context checkpoints kept for models whose cache cannot "
    "be rolled back.",
    "models.auto_checkpoint_min_step": "Minimum tokens between those checkpoints; empty "
    "uses the engine default.",
    "models.default_reasoning_format": "How reasoning text is returned for thinking models "
    "(none, deepseek, auto).",
    "models.thinking_default_ctx": "Context floor for thinking models, which need room to think.",
    "models.target_ctx": "The context every load aims for, halving down to what fits in VRAM.",
    "models.auto_load_pinned": "Load pinned models at startup and keep them loaded.",
    "models.default_model": "Model id that 'default' resolves to in requests.",
    "models.preload_default_model": "Load models.default_model at startup.",
    "engine.pinned_tag": "The llama.cpp build in use (for example b10425). Changing it "
    "does not switch the running engine: activate the build to do that.",
    "engine.cuda_variant": "CUDA build to install: auto picks the one matching the driver.",
    "engine.keep_versions": "Installed engine builds to keep; older ones are pruned.",
    "engine.allow_source_build": "Build llama.cpp from source when no prebuilt asset fits.",
    "engine.repo": "GitHub repository engine builds are downloaded from.",
    "engine.smoke_test_timeout_s": "Seconds a new engine build gets to pass its smoke test.",
    "engine.cache_ram_mb": "Host-RAM prompt cache per model (llama.cpp --cache-ram); auto "
    "shares a pool, 0 disables.",
    "planner.headroom_fraction": "Fraction of every GPU's VRAM left free (0.10 is 10%).",
    "planner.on_insufficient": "What to do when a load does not fit: evict other models or "
    "reject the load.",
    "planner.compute_overhead_fraction": "Fraction of model size reserved for compute buffers.",
    "planner.compute_overhead_floor_mb": "Minimum compute-buffer reserve per card, in MiB.",
    "planner.cuda_context_mb": "VRAM reserved per GPU for the CUDA context, in MiB.",
    "planner.image_tokens_default": "Tokens assumed per image when planning a vision load.",
    "planner.mmproj_compute_mb": "VRAM reserved for the multimodal projector's compute, in MiB.",
    "planner.prefer_single_gpu": "Prefer a placement on one GPU over splitting across several.",
    "planner.rebalance": "Whether the planner may move loaded models to make room "
    "(auto, always, never).",
    "planner.preference": "What the catalog's recommended load optimises: quality or throughput.",
    "planner.quant_affinity": "Quantization formats the planner prefers on GPUs of a given "
    "compute capability.",
    "planner.excluded_devices": "CUDA device indices the planner may never use.",
    "planner.reserved_mb": "VRAM in MiB held back per device index, even for forced placements.",
    "mcp.pin": "Pairing code for the MCP endpoint only, generated on first run; separate from "
    "the API key.",
    "mcp.pin_required": "Require the pairing PIN (or API key) on the MCP endpoint.",
    "mcp.advertise": "Print the reachable MCP URLs and the PIN in the startup banner.",
    "gateway.child_port_start": "First loopback port used for llama-server children.",
    "gateway.child_port_end": "Last loopback port used for llama-server children.",
    "gateway.load_timeout_s": "Seconds a model gets to become healthy after it is started.",
    "gateway.health_poll_interval_s": "Seconds between health polls while a model loads.",
    "gateway.max_restarts": "Times a crashed llama-server is restarted before giving up.",
    "gateway.restart_backoff_s": "Seconds to wait between restart attempts (grows each time).",
    "gateway.max_images_per_request": "Most images accepted in one request.",
    "gateway.max_image_bytes": "Largest accepted image, in bytes.",
    "gateway.allow_private_image_hosts": "Allow image_url fetches from private network "
    "addresses. Off protects against server-side request forgery.",
    "gateway.image_fetch_timeout_s": "Seconds allowed to download a remote image.",
    "gateway.max_image_dim": "Largest image side in pixels; larger images are downscaled.",
    "gateway.ttl_sweep_interval_s": "Seconds between the sweeps that unload idle models.",
    "gateway.stream_keepalive_interval_s": "Seconds between keep-alive comments on a "
    "streaming response.",
    "gateway.merge_reasoning_into_content": "Return reasoning text inside content for "
    "clients that cannot read a separate reasoning field.",
    "gateway.deep_probe_timeout_s": "Seconds allowed for a deep health probe's test completion.",
    "gateway.deep_probe_max_tokens": "Tokens a deep health probe generates.",
    "leases.vacate_timeout_s": "Seconds a lease holder gets to release a card after being asked.",
    "leases.vacate_retry_after_s": "Retry-After given to a caller waiting on a vacating lease.",
    "leases.vacate_callback_timeout_s": "Seconds allowed for the holder's vacate callback "
    "to answer.",
    "tenants.registry": "Programs declared as sharing the GPUs (name, URLs, kind).",
    "tenants.health_interval_s": "Seconds between tenant health checks.",
    "tenants.health_timeout_s": "Seconds a tenant has to answer a health check.",
    "tenants.allowed_hosts": "Extra hosts tenant URLs may point at, beyond loopback and the "
    "tailnet range.",
    "benchmark.lease_priority": "Priority class a benchmark's GPU lease is taken at.",
    "hf.token": "HuggingFace access token for gated repositories. A secret: never commit it.",
    "hf.cache_dir": "Reserved: no code reads this key yet (the metadata cache lives under the "
    "data directory).",
    "hf.max_concurrent_downloads": "Downloads that run at the same time.",
    "hf.chunk_bytes": "Bytes requested per HTTP range read while downloading.",
    "logging.level": "Minimum log level: DEBUG, INFO, WARNING or ERROR.",
    "logging.json": "Write logs as JSON lines instead of text.",
    "logging.file_max_mb": "Size in MiB at which a log file rotates; 0 never rotates.",
    "logging.file_backups": "Rotated log files kept per log.",
    "update.repo": "GitHub owner/repo the app checks for updates. Empty disables the check "
    "and no network call is made.",
    "update.channel": "Release channel the updater follows (stable or prerelease).",
    "update.auto_check": "Reserved: not read by any code path yet.",
    "update.check_interval_h": "Reserved: not read by any code path yet.",
    "update.health_check_timeout_s": "Seconds a freshly installed release gets to report "
    "healthy before it is rolled back.",
}


def descriptions() -> dict[str, str]:
    """Resolved description per dotted key (see the module docstring for the order)."""
    example_text = EXAMPLE_PATH.read_text(encoding="utf-8") if EXAMPLE_PATH.is_file() else ""
    header, block = _example_comments(example_text)
    source = _source_comments()
    resolved: dict[str, str] = {}
    for key, _attr, _info, _ann in _walk(Config):
        text = (
            block.get(key)
            or header.get(key)
            or source.get(key)
            or _FALLBACK_DESCRIPTIONS.get(key, "")
        )
        resolved[key] = text
    return resolved


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def build_registry() -> list[Setting]:
    """Every config key as a :class:`Setting`, in the model's own order."""
    texts = descriptions()
    secret_keys = {f"{section}.{name}" for section, name in SECRET_CONFIG_PATHS}
    settings: list[Setting] = []
    for key, attr, info, annotation in _walk(Config):
        constraints, choices = _collect_metadata(annotation, info)
        section = key.split(".", 1)[0] if "." in key else "general"
        restart = (
            key in RESTART_REQUIRED_KEYS
            or section in _NOT_LIVE_SECTIONS
            or key in _ENGINE_IDENTITY_KEYS
        )
        settings.append(
            Setting(
                key=key,
                section=section,
                type=_type_name(annotation),
                default=_jsonable(info.get_default(call_default_factory=True)),
                description=texts.get(key, ""),
                level="basic" if key in BASIC_KEYS else "advanced",
                env=_env_name(key, attr, info),
                restart_required=restart,
                secret=key in secret_keys,
                constraints=constraints,
                choices=_jsonable(choices) if choices else None,
            )
        )
    return settings


def missing_descriptions() -> list[str]:
    """Keys with no description anywhere (a test fails on a non-empty list)."""
    return [s.key for s in build_registry() if not s.description.strip()]


def export() -> dict[str, Any]:
    """The whole registry as plain data: the body of ``GET /api/settings/schema``."""
    settings = build_registry()
    return {
        "product": "StudioForge",
        "schema": CONFIG_SCHEMA_VERSION,
        "env_prefix": "SF_",
        "env_nested_delimiter": "__",
        "levels": ["basic", "advanced"],
        "count": len(settings),
        "settings": [s.to_dict() for s in settings],
    }


# ---------------------------------------------------------------------------
# Markdown reference
# ---------------------------------------------------------------------------


def _cell(value: Any) -> str:
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text if text else "-"


def _default_cell(value: Any) -> str:
    if value is None:
        return "`null`"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, separators=(",", ":"), sort_keys=True) + "`"
    return f"`{value}`"


def _valid_cell(setting: Setting) -> str:
    parts: list[str] = []
    labels = {
        "gt": ">",
        "ge": ">=",
        "lt": "<",
        "le": "<=",
        "min_length": "min length",
        "max_length": "max length",
        "multiple_of": "multiple of",
        "pattern": "pattern",
    }
    for attr in _CONSTRAINT_ATTRS:
        if attr in setting.constraints:
            parts.append(f"{labels[attr]} {setting.constraints[attr]}")
    if setting.choices:
        parts.append("one of " + ", ".join(f"`{c}`" for c in setting.choices))
    return "; ".join(parts) if parts else "-"


def render_markdown() -> str:
    """``docs/reference/settings.md``: generated, never hand-edited."""
    settings = build_registry()
    out: list[str] = [
        "# Settings reference",
        "",
        "<!-- GENERATED by scripts/gen_docs.py from studioforge.settings_schema. "
        "Do not edit by hand. -->",
        "",
        f"Settings schema version **{CONFIG_SCHEMA_VERSION}** (`schema:` in `config.yaml`). "
        f"{len(settings)} keys.",
        "",
        "Settings live in `<data dir>/config.yaml`. The machine-readable form of this page "
        "is `GET /api/settings/schema` and `docs/contracts/settings-schema.json`.",
        "",
        "**Environment overrides.** Every key can also be given as an environment variable: "
        "`SF_` plus the dotted key in upper case with `.` written as `__` "
        "(`server.port` is `SF_SERVER__PORT`). Environment values are first-run defaults: "
        "once `config.yaml` names a key, the file wins. Only `SF_DATA_DIR` and `SF_CONFIG` "
        "are authoritative at all times.",
        "",
        "**Levels.** *Basic* keys are the ones most installs touch; everything else is "
        "*Advanced*. **Restart** means a change does not take effect on a running server.",
        "",
    ]
    by_section: dict[str, list[Setting]] = {}
    for setting in settings:
        by_section.setdefault(setting.section, []).append(setting)
    for section, rows in by_section.items():
        out.append(f"## {section}" if section != "general" else "## general")
        out.append("")
        out.append("| Key | Type | Default | Valid values | Level | Restart | Env | Description |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for s in rows:
            key = f"`{s.key}`" + (" (secret)" if s.secret else "")
            out.append(
                "| "
                + " | ".join(
                    [
                        key,
                        f"`{_cell(s.type)}`",
                        _default_cell(s.default),
                        _cell(_valid_cell(s)),
                        s.level,
                        "yes" if s.restart_required else "no",
                        f"`{s.env}`",
                        _cell(s.description),
                    ]
                )
                + " |"
            )
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"
