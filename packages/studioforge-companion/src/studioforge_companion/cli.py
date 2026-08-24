"""``sfctl`` -- the StudioForge companion CLI.

Every command has a ``--json`` twin because this thing is as likely to be run
from a script or an agent as from a keyboard, and every failure maps onto the
documented exit codes in :mod:`studioforge_companion.client`.

Two commands deliberately bypass the main server:

* ``sfctl recover`` talks to the watchdog (separate process, port 1235), because
  the situation it exists for is the main server being unreachable;
* ``sfctl mcp`` merges both control planes for the same reason.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table

from studioforge_companion.client import (
    EXIT_CODE_TABLE,
    EXIT_CONFIRM,
    EXIT_USAGE,
    ApiError,
    CompanionError,
    StudioForgeClient,
)
from studioforge_companion.config import (
    CompanionConfig,
    CompanionConfigError,
    ServerProfile,
    config_path,
    load_companion_config,
    redact,
    save_companion_config,
)

_EXIT_HELP = "\n".join(f"  {code}  {text}" for code, text in EXIT_CODE_TABLE)

# The lone "\b" line is click's marker for "do not rewrap the block below", which
# is what keeps the exit-code table one-per-line in --help.
HELP = f"""Remote control for a StudioForge LLM server.

\b
Exit codes (scriptable):
{_EXIT_HELP}
"""


def _json_flag(value: bool) -> bool:
    """Record `--json` at PARSE time, not when the command body reads it.

    A per-command `--json` used to reach STATE only via want_json(), which runs
    inside the command body -- after the network call. So `sfctl status --json`
    against an unreachable server failed before the flag was ever recorded, and
    the error came out as Rich prose. A parse-time callback fires first.
    """
    if value:
        STATE.json_out = True
    return value


JSON_OPTION = typer.Option(
    False, "--json", callback=_json_flag, help="Machine-readable JSON output."
)

app = typer.Typer(
    name="sfctl",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)
models_app = typer.Typer(help="Inspect and control models.", no_args_is_help=True)
config_app = typer.Typer(help="Read and change server-side configuration.", no_args_is_help=True)
servers_app = typer.Typer(help="Manage local server profiles.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")
app.add_typer(servers_app, name="servers")


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------


# Registry kinds the server filters on. Module scope because a default
# argument is evaluated at import, not at call time.
_MODEL_KINDS = ("chat", "embedding", "rerank", "vision", "draft")


@dataclass
class CliState:
    server: str | None = None
    url: str | None = None
    api_key: str | None = None
    json_out: bool = False
    no_color: bool = False
    console: Console = field(default_factory=Console)
    err: Console = field(default_factory=lambda: Console(stderr=True))

    def profile(self) -> ServerProfile:
        """Ad-hoc profile from ``--url``, else the named/default local profile."""
        if self.url:
            return ServerProfile(name="cli", url=self.url, api_key=self.api_key)
        cfg = load_companion_config()
        profile = cfg.profile(self.server)
        if self.api_key:
            return profile.model_copy(update={"api_key": self.api_key})
        return profile


STATE = CliState()


@app.callback()
def main_options(
    server: str | None = typer.Option(
        None, "--server", "-s", help="Named server profile from companion.toml."
    ),
    url: str | None = typer.Option(None, "--url", help="Server URL, bypassing local profiles."),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="SF_API_KEY", help="API key (never echoed)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON output."),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colour and styling."),
) -> None:
    width = int(os.environ.get("COLUMNS") or 0) or 120
    STATE.server = server
    STATE.url = url
    STATE.api_key = api_key
    STATE.json_out = json_out
    STATE.no_color = no_color
    STATE.console = Console(no_color=no_color, soft_wrap=False, width=width)
    STATE.err = Console(stderr=True, no_color=no_color, width=width)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def _report(exc: CompanionError | CompanionConfigError) -> None:
    """Print a failure the way a human wants it: message, then advice.

    Suggestions from a VRAM rejection live in the error's ``studioforge``
    diagnostics block; losing them here would turn actionable advice ("try
    --ctx 8192") into a dead end.

    Under ``--json`` the same failure is emitted as one JSON object instead.
    The flag is documented as "machine-readable output"; printing Rich prose
    on the error path -- hard-wrapped at terminal width, so not even reliably
    greppable -- broke that contract exactly when a script needed structure.
    """
    if STATE.json_out:
        payload: dict[str, Any] = {
            "ok": False,
            "error": str(exc),
            "exit_code": getattr(exc, "exit_code", 1),
        }
        for attr in ("code", "status", "details", "suggestions"):
            value = getattr(exc, attr, None)
            if value:
                payload[attr] = list(value) if attr == "suggestions" else value
        # stdout, like every other --json payload: a caller redirecting stdout
        # into a parser must not have to merge two streams to see a failure.
        print(json.dumps(payload, indent=2, default=str))
        return
    STATE.err.print(f"[red]error:[/red] {exc}" if not STATE.no_color else f"error: {exc}")
    suggestions: Sequence[str] = getattr(exc, "suggestions", ()) or ()
    for suggestion in suggestions:
        STATE.err.print(f"  - {suggestion}")


def run(coro: Awaitable[Any]) -> Any:
    """Drive one coroutine, translating failures into documented exit codes."""
    try:
        return asyncio.run(_await(coro))
    except (CompanionError, CompanionConfigError) as exc:
        _report(exc)
        raise typer.Exit(getattr(exc, "exit_code", 1)) from None
    except KeyboardInterrupt:
        # Ctrl-C is how you leave `logs --follow` and `chat`; that is success.
        STATE.console.print()
        raise typer.Exit(0) from None


async def _await(coro: Awaitable[Any]) -> Any:
    return await coro


def with_client(work: Callable[[StudioForgeClient], Awaitable[Any]]) -> Any:
    """Open a client for the resolved profile, run ``work``, close it."""
    profile = _resolve_profile()

    async def _go() -> Any:
        async with StudioForgeClient(profile) as client:
            return await work(client)

    return run(_go())


def _resolve_profile() -> ServerProfile:
    try:
        return STATE.profile()
    except CompanionConfigError as exc:
        _report(exc)
        raise typer.Exit(exc.exit_code) from None


def emit(data: Any) -> None:
    """JSON to stdout, unwrapped by rich so it always parses."""
    typer.echo(json.dumps(data, indent=2, default=str, sort_keys=False))


def want_json(local: bool) -> bool:
    """Did the caller ask for JSON, globally or on this command?

    Records the answer on STATE so the ERROR path can honour it too. `--json`
    is accepted both before the subcommand (global) and after it (per-command),
    but only the global form reached STATE -- so `sfctl status --json` failing
    printed Rich prose, which is the one moment a caller piping to a parser
    cannot cope with.
    """
    if local:
        STATE.json_out = True
    return local or STATE.json_out


def fmt_bytes(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(number) < 1024 or unit == "TiB":
            return f"{number:,.1f} {unit}" if unit != "B" else f"{number:,.0f} B"
        number /= 1024
    return f"{number:,.1f} TiB"


def fmt_duration(seconds: Any) -> str:
    try:
        total = int(float(seconds))
    except (TypeError, ValueError):
        return "-"
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def fmt_bool(value: Any) -> str:
    return "yes" if value else "-"


def _table(*columns: str, title: str | None = None) -> Table:
    table = Table(title=title, header_style="bold", expand=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    return table


def _confirm(question: str, *, yes: bool) -> None:
    """Gate a destructive action.

    In a non-tty (CI, an agent, a cron job) an interactive prompt would hang
    forever, so this exits ``3`` with an explanation instead -- a script can
    detect "needed confirmation" distinctly from "was refused".
    """
    if yes:
        return
    if not sys.stdin.isatty():
        STATE.err.print(
            f"error: {question} requires confirmation. "
            f"Re-run with --yes (stdin is not a terminal, so it cannot be asked)."
        )
        raise typer.Exit(EXIT_CONFIRM)
    try:
        answered = typer.confirm(question)
    except (typer.Abort, EOFError):
        # An unreadable stdin that still claims to be a tty (Windows NUL does)
        # must land on the same exit code as the honest non-tty case.
        STATE.err.print(f"error: {question} requires confirmation. Re-run with --yes.")
        raise typer.Exit(EXIT_CONFIRM) from None
    if not answered:
        STATE.console.print("aborted")
        raise typer.Exit(EXIT_CONFIRM)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(json_out: bool = JSON_OPTION) -> None:
    """VRAM per GPU, loaded models, queue depth, engine and uptime."""

    async def work(client: StudioForgeClient) -> tuple[Any, Any]:
        payload = await client.status()
        models: Any = None
        try:
            # /api/models is the authoritative source of the TTL countdown
            # (it is a computed property the status payload does not carry).
            models = await client.models()
        except CompanionError:
            models = None
        return payload, models

    payload, models = with_client(work)
    if want_json(json_out):
        emit(payload)
        return

    console = STATE.console
    gpu_table = _table("GPU", "Name", "Total", "Used", "Free", "Util", "Temp", title="GPUs")
    for gpu in payload.get("gpus") or []:
        cc = gpu.get("compute_capability")
        name = gpu.get("name", "?")
        if cc:
            name = f"{name} (sm_{cc[0]}{cc[1]})"
        util = gpu.get("utilization_pct")
        temp = gpu.get("temperature_c")
        gpu_table.add_row(
            str(gpu.get("index")),
            name,
            fmt_bytes(gpu.get("total_bytes")),
            fmt_bytes(gpu.get("used_bytes")),
            fmt_bytes(gpu.get("free_bytes")),
            f"{util:.0f}%" if isinstance(util, (int, float)) else "-",
            f"{temp:.0f}C" if isinstance(temp, (int, float)) else "-",
        )
    console.print(gpu_table)

    ttl_by_model: dict[str, Any] = {}
    for record in (models or {}).get("models") or []:
        ttl_by_model[str(record.get("id"))] = record.get("ttl_remaining_s")

    loaded = payload.get("loaded") or []
    if loaded:
        table = _table("Model", "State", "Ctx", "Port", "PID", "TTL left", "Active", "tok/s")
        for instance in loaded:
            plan = instance.get("plan") or {}
            model_id = str(instance.get("model_id"))
            ttl = ttl_by_model.get(model_id, instance.get("ttl_s"))
            tps = instance.get("last_tokens_per_second")
            table.add_row(
                model_id,
                str(instance.get("state")),
                str(plan.get("ctx_size") or "-"),
                str(instance.get("port") or "-"),
                str(instance.get("pid") or "-"),
                "pinned" if not instance.get("ttl_s") else fmt_duration(ttl),
                str(instance.get("active_requests") or 0),
                f"{tps:.1f}" if isinstance(tps, (int, float)) else "-",
            )
        console.print(table)
    else:
        console.print("no models loaded")

    engine = payload.get("engine") or {}
    summary = _table("Field", "Value", title="Server")
    summary.add_row("version", str(payload.get("version", "?")))
    summary.add_row("uptime", fmt_duration(payload.get("uptime_s")))
    summary.add_row("engine", str(engine.get("tag") or "not installed"))
    summary.add_row("models in registry", str(payload.get("model_count", 0)))
    summary.add_row("queue depth", str(payload.get("queue_depth", 0)))
    summary.add_row("active downloads", str(payload.get("active_downloads", 0)))
    summary.add_row("draining", fmt_bool(payload.get("draining")))
    summary.add_row(
        "system RAM",
        f"{fmt_bytes(payload.get('system_ram_used_bytes'))} / "
        f"{fmt_bytes(payload.get('system_ram_total_bytes'))}",
    )
    console.print(summary)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _badges(record: dict[str, Any]) -> str:
    caps = record.get("capabilities") or {}
    badges = []
    if caps.get("vision"):
        badges.append("vision")
    if caps.get("embedding"):
        badges.append("embed")
    if caps.get("tools"):
        badges.append("tools")
    if record.get("mmproj_path"):
        badges.append("mmproj")
    if (record.get("settings") or {}).get("pinned"):
        badges.append("pinned")
    if record.get("is_virtual"):
        badges.append("virtual")
    return ",".join(badges) or "-"


@models_app.command("list")
def models_list(
    loaded: bool = typer.Option(False, "--loaded", help="Only currently loaded models."),
    vision: bool = typer.Option(False, "--vision", help="Only vision-capable models."),
    kind: str | None = typer.Option(
        None, "--kind", help=f"Filter by kind: {', '.join(_MODEL_KINDS)}."
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """List registry models with capability badges."""
    if kind is not None and kind.lower() not in _MODEL_KINDS:
        # An unknown kind silently returned zero rows and exit 0, so a typo was
        # indistinguishable from "the registry has none of those".
        raise typer.BadParameter(
            f"unknown kind {kind!r}; expected one of {', '.join(_MODEL_KINDS)}", param_hint="--kind"
        )

    async def work(client: StudioForgeClient) -> Any:
        return await client.models()

    payload = with_client(work)
    records: list[dict[str, Any]] = list(payload.get("models") or [])
    if loaded:
        records = [r for r in records if r.get("loaded")]
    if vision:
        records = [r for r in records if (r.get("capabilities") or {}).get("vision")]
    if kind:
        records = [r for r in records if str(r.get("kind")) == kind]

    if want_json(json_out):
        emit({"models": records, "count": len(records)})
        return

    table = _table("Model", "Kind", "Quant", "Size", "Caps", "Loaded", "Port", "TTL left")
    for record in sorted(records, key=lambda r: str(r.get("id"))):
        table.add_row(
            str(record.get("id")),
            str(record.get("kind")),
            str(record.get("quant")),
            fmt_bytes(record.get("size_bytes")),
            _badges(record),
            str(record.get("state") if record.get("loaded") else "-"),
            str(record.get("port") or "-"),
            fmt_duration(record.get("ttl_remaining_s"))
            if record.get("ttl_remaining_s") is not None
            else "-",
        )
    STATE.console.print(table)
    STATE.console.print(f"{len(records)} model(s)")


@models_app.command("info")
def models_info(model: str, json_out: bool = JSON_OPTION) -> None:
    """Full details, with the REQUESTED settings and the ACTUAL running values."""

    async def work(client: StudioForgeClient) -> dict[str, Any]:
        listing = await client.models()
        record: dict[str, Any] | None = None
        for candidate in listing.get("models") or []:
            if str(candidate.get("id")) == model or str(candidate.get("name")) == model:
                record = candidate
                break
        settings = await client.settings(model)
        actual = await client.introspect(model)
        if record is None:
            record = {"id": model}
        return {"model": record, "settings": settings, "actual": actual}

    data = with_client(work)
    if want_json(json_out):
        emit(data)
        return

    record, settings, actual = data["model"], data["settings"], data["actual"]
    console = STATE.console
    facts = _table("Field", "Value", title=str(record.get("id")))
    for key in (
        "name",
        "kind",
        "quant",
        "architecture",
        "publisher",
        "repo",
        "path",
        "mmproj_path",
        "size_bytes",
        "loaded",
        "state",
        "port",
    ):
        if key not in record:
            continue
        value = record[key]
        if key == "size_bytes":
            value = fmt_bytes(value)
        facts.add_row(key, str(value) if value is not None else "-")
    facts.add_row("capabilities", _badges(record))
    console.print(facts)

    # Requested vs actual, side by side: llama-server silently clamps some
    # values (ctx to the trained maximum, slots to what fits), so "what I asked
    # for" and "what is running" are genuinely different questions.
    running = actual.get("actual") if isinstance(actual.get("actual"), dict) else {}
    compare = _table("Setting", "Requested", "Actual (running)", title="settings")
    keys = sorted({*(settings or {}).keys(), *(running or {}).keys()})
    for key in keys:
        requested = (settings or {}).get(key)
        got = (running or {}).get(key)
        if requested in (None, "", [], {}) and got is None:
            continue
        compare.add_row(key, _short(requested), _short(got))
    console.print(compare)
    if not actual.get("loaded"):
        console.print("(not loaded -- 'Actual' columns are empty until it is)")


def _short(value: Any) -> str:
    if value is None:
        return "-"
    text = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _print_plan(plan: dict[str, Any]) -> None:
    console = STATE.console
    if plan.get("fits"):
        table = _table("Field", "Value", title=f"fit plan: {plan.get('model_id')}")
        table.add_row("verdict", "fits")
        for key in ("ctx_size", "parallel", "kv_cache_type", "flash_attn", "split_mode"):
            if plan.get(key) is not None:
                table.add_row(key, str(plan[key]))
        devices = plan.get("devices") or []
        table.add_row("devices", ", ".join(str(d) for d in devices) or "-")
        split = plan.get("tensor_split") or []
        if split:
            table.add_row("tensor_split", ", ".join(f"{s:.3f}" for s in split))
        per_gpu = plan.get("per_gpu_bytes") or {}
        for gpu, amount in per_gpu.items() if isinstance(per_gpu, dict) else []:
            table.add_row(f"gpu {gpu} projected", fmt_bytes(amount))
        evict = plan.get("evict_model_ids") or []
        if evict:
            table.add_row("would evict", ", ".join(str(e) for e in evict))
        console.print(table)
        return

    table = _table("Field", "Value", title=f"fit plan: {plan.get('model_id')}")
    table.add_row("verdict", "DOES NOT FIT")
    table.add_row("reason", str(plan.get("reason") or "-"))
    if plan.get("message"):
        table.add_row("message", str(plan["message"]))
    table.add_row("required", fmt_bytes(plan.get("required_bytes")))
    table.add_row("available", fmt_bytes(plan.get("available_bytes")))
    if plan.get("max_ctx_that_fits") is not None:
        table.add_row("max ctx that fits", str(plan["max_ctx_that_fits"]))
    per_gpu_free = plan.get("per_gpu_free") or {}
    for gpu, amount in per_gpu_free.items() if isinstance(per_gpu_free, dict) else []:
        table.add_row(f"gpu {gpu} free", fmt_bytes(amount))
    estimate = plan.get("estimate_mb") or {}
    for key, amount in estimate.items() if isinstance(estimate, dict) else []:
        table.add_row(f"estimate {key}", f"{amount} MiB")
    console.print(table)
    for suggestion in plan.get("suggestions") or []:
        console.print(f"  - {suggestion}")


@models_app.command("plan")
def models_plan(
    model: str,
    ctx: int | None = typer.Option(None, "--ctx", help="Context size to plan for."),
    kv_type: str | None = typer.Option(None, "--kv-type", help="KV cache type, e.g. q8_0."),
    parallel: int | None = typer.Option(None, "--parallel", help="Parallel slots."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Fit verdict, per-GPU projection and suggestions -- loads nothing."""

    async def work(client: StudioForgeClient) -> Any:
        return await client.plan(model, ctx_size=ctx, kv_cache_type=kv_type, parallel=parallel)

    plan = with_client(work)
    if want_json(json_out):
        emit(plan)
        return
    _print_plan(plan)


@models_app.command("load")
def models_load(
    model: str,
    ctx: int | None = typer.Option(None, "--ctx", help="Context size."),
    kv_type: str | None = typer.Option(None, "--kv-type", help="KV cache type, e.g. q8_0."),
    parallel: int | None = typer.Option(None, "--parallel", help="Parallel slots."),
    force: bool = typer.Option(
        False, "--force", help="Load even if the plan says it will not fit."
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """Show the fit plan, then load."""
    use_json = want_json(json_out)

    async def work(client: StudioForgeClient) -> Any:
        plan: Any = None
        try:
            plan = await client.plan(model, ctx_size=ctx, kv_cache_type=kv_type, parallel=parallel)
        except CompanionError as exc:
            if isinstance(exc, ApiError) and exc.status_code == 404:
                raise
            # A 500 from /plan used to leave plan=None, which skipped the
            # "will not fit, pass --force" guard entirely -- turning this into
            # an implicit --force without saying so. Say so, and make the
            # operator opt in.
            if not force:
                raise ApiError(
                    f"could not check whether {model} fits ({exc}); "
                    f"pass --force to load without the check",
                    code="plan_unavailable",
                    status_code=getattr(exc, "status_code", None),
                ) from None
            STATE.err.print(
                f"[yellow]warning:[/yellow] fit check unavailable ({exc}); "
                f"loading anyway because --force was given"
                if not STATE.no_color
                else f"warning: fit check unavailable ({exc}); loading anyway (--force)"
            )
            plan = None
        if plan is not None and not use_json:
            _print_plan(plan)
        if plan is not None and not plan.get("fits") and not force:
            raise ApiError(
                f"{model} will not fit as requested; pass --force to try anyway, "
                f"or apply one of the suggestions",
                code="insufficient_vram",
                status_code=507,
                suggestions=[str(s) for s in plan.get("suggestions") or []],
            )
        if use_json:
            return await client.load(
                model, ctx_size=ctx, kv_cache_type=kv_type, parallel=parallel, force=force
            )
        with STATE.console.status(f"loading {model}...", spinner="dots"):
            return await client.load(
                model, ctx_size=ctx, kv_cache_type=kv_type, parallel=parallel, force=force
            )

    instance = with_client(work)
    if use_json:
        emit(instance)
        return
    plan = instance.get("plan") or {}
    model_id = str(instance.get("model_id"))
    name = model_id if STATE.no_color else f"[bold]{model_id}[/bold]"
    STATE.console.print(
        f"loaded {name} state={instance.get('state')} port={instance.get('port')} "
        f"ctx={plan.get('ctx_size')} pid={instance.get('pid')}"
    )


@models_app.command("unload")
def models_unload(model: str, json_out: bool = JSON_OPTION) -> None:
    """Unload a model and free its VRAM."""

    async def work(client: StudioForgeClient) -> Any:
        return await client.unload(model)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    if result.get("unloaded"):
        STATE.console.print(f"unloaded {result.get('model_id')}")
    else:
        STATE.console.print(f"{result.get('model_id')} was not loaded")


@models_app.command("pin")
def models_pin(
    model: str,
    off: bool = typer.Option(False, "--off", help="Unpin instead of pinning."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Pin a model: kept loaded at all times, reloaded if it goes down."""

    async def work(client: StudioForgeClient) -> Any:
        return await client.pin(model, not off)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    STATE.console.print(f"{result.get('model_id')}: pinned={fmt_bool(result.get('pinned'))}")


@models_app.command("test")
def models_test(
    model: str,
    prompt: str | None = typer.Option(None, "--prompt", help="Prompt to send."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Run a short completion; report latency, tok/s and the reply."""

    async def work(client: StudioForgeClient) -> Any:
        if want_json(json_out):
            return await client.test(model, prompt)
        with STATE.console.status(f"testing {model} (loads it if needed)...", spinner="dots"):
            return await client.test(model, prompt)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    console = STATE.console
    table = _table("Field", "Value", title=f"test: {result.get('model_id')}")
    table.add_row("ok", fmt_bool(result.get("ok")))
    table.add_row("latency", f"{result.get('latency_s')} s")
    if result.get("tokens_per_second") is not None:
        table.add_row("tokens/s", str(result.get("tokens_per_second")))
    if result.get("completion_tokens") is not None:
        table.add_row("completion tokens", str(result.get("completion_tokens")))
    if result.get("embedding_dims") is not None:
        table.add_row("embedding dims", str(result.get("embedding_dims")))
    console.print(table)
    if result.get("text"):
        console.print(Panel(str(result["text"]), title="reply", expand=False))


@models_app.command("delete")
def models_delete(
    model: str,
    files: bool = typer.Option(False, "--files", help="Also delete the GGUF files from disk."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Remove a model from the registry (optionally deleting its files)."""
    what = f"delete '{model}' and its files from disk" if files else f"delete '{model}'"
    _confirm(f"{what}?", yes=yes)

    async def work(client: StudioForgeClient) -> Any:
        return await client.delete_model(model, delete_files=files, confirm=True)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    removed = result.get("removed") or []
    STATE.console.print(f"deleted {result.get('model_id')}")
    for path in removed:
        STATE.console.print(f"  removed {path}")


@models_app.command("scan")
def models_scan(
    force: bool = typer.Option(False, "--force", help="Re-read metadata for every file."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Rescan the model directories."""

    async def work(client: StudioForgeClient) -> Any:
        if want_json(json_out):
            return await client.scan(force)
        with STATE.console.status("scanning model directories...", spinner="dots"):
            return await client.scan(force)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    console = STATE.console
    console.print(
        f"added={len(result.get('added') or [])} "
        f"removed={len(result.get('removed') or [])} "
        f"unchanged={result.get('unchanged')} in {result.get('duration_s')}s"
    )
    for item in result.get("added") or []:
        console.print(f"  + {item}")
    for item in result.get("removed") or []:
        console.print(f"  - {item}")
    for error in result.get("errors") or []:
        console.print(f"  ! {error.get('path')}: {error.get('error')}")


def _coerce_like(current: Any, raw: str) -> Any:
    """Coerce ``raw`` to the type of the value it replaces.

    Typing ``--set ctx_size=8192`` should not save the string ``"8192"``; the
    current value is the most reliable type hint available client-side, and the
    server validates whatever survives.
    """
    if isinstance(current, bool):
        lowered = raw.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
        raise CompanionConfigError(f"expected a boolean for this key, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(raw)
        except ValueError as exc:
            raise CompanionConfigError(f"expected an integer, got {raw!r}") from exc
    if isinstance(current, float):
        try:
            return float(raw)
        except ValueError as exc:
            raise CompanionConfigError(f"expected a number, got {raw!r}") from exc
    if isinstance(current, (list, dict)):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CompanionConfigError(f"expected JSON for this key, got {raw!r}") from exc
    if current is None:
        if raw.lower() in ("null", "none", ""):
            return None
        for caster in (int, float):
            try:
                return caster(raw)
            except ValueError:
                continue
        if raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        return raw
    return raw


def _assign(target: dict[str, Any], dotted: str, raw: str) -> None:
    parts = dotted.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    leaf = parts[-1]
    cursor[leaf] = _coerce_like(cursor.get(leaf), raw)


def _split_pairs(pairs: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key.strip():
            raise CompanionConfigError(f"expected key=value, got {pair!r}")
        out.append((key.strip(), value))
    return out


@models_app.command("settings")
def models_settings(
    model: str,
    set_: list[str] = typer.Option(
        [], "--set", help="key=value (dotted keys allowed); repeatable."
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """Show or change per-model settings.

    Server-side validation errors (notably ``extra_flags``, which is checked
    against the engine's own ``--help``) are reported verbatim.
    """

    async def work(client: StudioForgeClient) -> Any:
        current = await client.settings(model)
        if not set_:
            return current
        updates = dict(current)
        for key, value in _split_pairs(set_):
            _assign(updates, key, value)
        return await client.put_settings(model, updates)

    settings = with_client(work)
    if want_json(json_out):
        emit(settings)
        return
    table = _table("Setting", "Value", title=f"settings: {model}")
    for key in sorted(settings or {}):
        table.add_row(key, _short(settings[key]))
    STATE.console.print(table)
    if set_:
        STATE.console.print("saved")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


#: Download statuses the server will not move on from.
_TERMINAL_DOWNLOAD = frozenset({"completed", "failed", "canceled", "cancelled", "paused"})


@app.command()
def download(
    repo_id: str = typer.Argument(..., help="Hugging Face repo, e.g. unsloth/gemma-3-27b-it-GGUF"),
    quant: str | None = typer.Option(None, "--quant", help="Quant to pick, e.g. Q4_K_M."),
    no_mmproj: bool = typer.Option(False, "--no-mmproj", help="Skip the vision projector."),
    force: bool = typer.Option(False, "--force", help="Re-download files already on disk."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Download a model from Hugging Face, with a live progress bar.

    Progress comes from polling the server's own download state, so the bar
    survives Ctrl-C: the download keeps running server-side, and re-running this
    command re-attaches to it.
    """
    use_json = want_json(json_out)

    async def work(client: StudioForgeClient) -> Any:
        try:
            started = await client.start_download(
                repo_id, quant=quant, include_mmproj=not no_mmproj, force=force
            )
        except ApiError as exc:
            if exc.code == "downloads_unavailable" or exc.status_code in (404, 405, 501):
                raise ApiError(
                    f"{client.profile.url} cannot download models "
                    f"({exc.message}). Copy the GGUF into the server's model directory and "
                    f"run 'sfctl models scan' instead.",
                    code="downloads_unavailable",
                    status_code=exc.status_code,
                ) from None
            raise
        if use_json:
            return started

        group_id = str(started.get("group_id") or "")
        STATE.console.print(
            f"{started.get('repo_id')} quant={started.get('quant')} "
            f"total={fmt_bytes(started.get('total_bytes'))} group={group_id}"
        )
        return await _follow_download(client, group_id, started)

    result = with_client(work)
    if use_json:
        emit(result)
        return

    files = result.get("downloads") or []
    failed = [f for f in files if str(f.get("status")) == "failed"]
    if failed:
        for entry in failed:
            STATE.err.print(
                f"error: {entry.get('filename')}: {entry.get('error') or 'download failed'}"
            )
        raise typer.Exit(1)
    STATE.console.print(
        f"download {result.get('status', 'finished')}: {result.get('repo_id') or repo_id}"
    )
    STATE.console.print("run 'sfctl models scan' if the new model is not listed yet")


async def _follow_download(
    client: StudioForgeClient, group_id: str, started: dict[str, Any]
) -> dict[str, Any]:
    """Poll the server's download state, one progress row per file."""
    latest: dict[str, Any] = dict(started)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=STATE.console,
    ) as progress:
        tasks: dict[str, Any] = {}
        while True:
            snapshot = await client.downloads()
            entries = [
                entry
                for entry in (snapshot.get("downloads") or [])
                if not group_id or str(entry.get("group_id")) == group_id
            ]
            if not entries:
                break
            for entry in entries:
                key = str(entry.get("id"))
                label = f"{entry.get('filename')} [{entry.get('status')}]"
                total = entry.get("total_bytes")
                done = entry.get("downloaded_bytes") or 0
                if key not in tasks:
                    tasks[key] = progress.add_task(label, total=float(total) if total else None)
                progress.update(
                    tasks[key],
                    description=label,
                    total=float(total) if total else None,
                    completed=float(done),
                )
            statuses = {str(entry.get("status")) for entry in entries}
            latest = {
                "group_id": group_id,
                "repo_id": started.get("repo_id"),
                "quant": started.get("quant"),
                "status": _group_status(statuses),
                "downloads": entries,
            }
            if statuses <= _TERMINAL_DOWNLOAD:
                break
            await asyncio.sleep(1.0)
    return latest


def _group_status(statuses: set[str]) -> str:
    for candidate in ("failed", "running", "queued", "paused", "canceled", "completed"):
        if candidate in statuses:
            return candidate
    return "unknown"


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------


def _new_lines(previous: Sequence[str], current: Sequence[str]) -> list[str]:
    """Lines in ``current`` not already printed, by longest-suffix overlap.

    The log endpoint returns a tail, not a cursor, so overlap detection is how
    ``--follow`` avoids reprinting the window on every poll.
    """
    if not previous:
        return list(current)
    limit = min(len(previous), len(current))
    for size in range(limit, 0, -1):
        if list(previous[-size:]) == list(current[:size]):
            return list(current[size:])
    return list(current)


def _format_log_line(line: Any) -> str:
    """Render one log entry for a human.

    The server sends structured entries for its own logs and plain strings for
    a model's llama-server passthrough. ``str()`` on the former printed a
    Python dict repr -- single quotes, a raw float timestamp, and a `message`
    field that already carries its own ISO timestamp and level, so every line
    double-printed its metadata. The pre-formatted message IS the line; the
    composed fallback only runs for an entry that lacks one.
    """
    if isinstance(line, str):
        return line
    if isinstance(line, dict):
        message = line.get("message")
        if message:
            return str(message)
        ts = line.get("ts")
        stamp = ""
        if isinstance(ts, (int, float)):
            stamp = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") + " "
        level = str(line.get("level") or "").upper()
        logger = str(line.get("logger") or "")
        return f"{stamp}{level:<8}{logger} {line.get('msg') or ''}".rstrip()
    return str(line)


@app.command()
def logs(
    model: str | None = typer.Argument(None, help="Model id for per-model llama-server logs."),
    follow: bool = typer.Option(False, "--follow", "-f", help="Poll and print new lines."),
    n: int = typer.Option(200, "-n", help="How many lines to show."),
    level: str | None = typer.Option(None, "--level", help="Minimum level (server logs only)."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Show server logs, or one model's llama-server output. Ctrl-C exits 0."""

    async def fetch(client: StudioForgeClient) -> list[str]:
        if model:
            payload = await client.model_logs(model, n)
        else:
            payload = await client.logs(n, level)
        lines = payload.get("lines") or []
        if model and not lines and not payload.get("path"):
            # The server answers 200 with an empty list for a model id it does
            # not know, so rendering it verbatim printed nothing and exited 0 --
            # a typo'd id looked like a quiet success. `models info` on the same
            # id exits 1; match that.
            raise CompanionError(f"no logs for {model!r} -- the server does not know that model id")
        return [_format_log_line(line) for line in lines]

    async def work(client: StudioForgeClient) -> Any:
        lines = await fetch(client)
        if want_json(json_out):
            return {"model_id": model, "lines": lines}
        for line in lines:
            typer.echo(line)
        if not follow:
            return None
        previous = lines
        while True:
            await asyncio.sleep(1.5)
            current = await fetch(client)
            for line in _new_lines(previous, current):
                typer.echo(line)
            previous = current

    result = with_client(work)
    if want_json(json_out) and result is not None:
        emit(result)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

# "pin" is here because GET /api/openclaw-setup returns the MCP pairing PIN
# as `mcp_pin`, and that name contains none of the other hints. The server
# only fills it in when `server.api_key` is set, so the leak stayed latent
# on a rig with auth off -- and would have armed itself the moment someone
# hardened the server by setting a key. This command's whole contract is
# that its default output is safe to paste into a chat or an issue.
_SECRET_HINTS = ("api_key", "token", "secret", "password", "pin")

#: The server's sentinel for "auth is disabled" (see ``GET /api/openclaw-setup``).
#: It is not a secret, and redacting it once produced the nonsense value
#: ``not-...ed`` in the printed snippets.
_NO_KEY_SENTINEL = "not-required"


def _redact_tree(value: Any, key: str = "") -> Any:
    """Belt-and-braces redaction of anything secret-looking.

    The server already redacts, but this command also prints values the user
    just typed, and a key must never reach a terminal, a scrollback buffer or a
    pasted bug report. The one exception is the server's ``not-required``
    sentinel, which means "there is no key" and must stay readable.
    """
    if isinstance(value, dict):
        return {k: _redact_tree(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_tree(v, key) for v in value]
    if (
        isinstance(value, str)
        and value != _NO_KEY_SENTINEL
        and any(hint in key.lower() for hint in _SECRET_HINTS)
    ):
        return redact(value)
    return value


@config_app.command("get")
def config_get(
    key: str | None = typer.Argument(None, help="Dotted key, e.g. models.default_ctx."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Read the server configuration (secrets redacted)."""

    async def work(client: StudioForgeClient) -> Any:
        return await client.get_config()

    payload = with_client(work)
    config = _redact_tree(payload.get("config") or {})
    if key:
        cursor: Any = config
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                STATE.err.print(f"error: no such config key: {key}")
                raise typer.Exit(EXIT_USAGE)
            cursor = cursor[part]
        if want_json(json_out):
            emit({key: cursor})
        else:
            typer.echo(
                json.dumps(cursor, indent=2, default=str)
                if isinstance(cursor, (dict, list))
                else str(cursor)
            )
        return

    if want_json(json_out):
        emit(
            {
                "config": config,
                "config_path": payload.get("config_path"),
                "restart_required_keys": payload.get("restart_required_keys"),
            }
        )
        return

    table = _table("Key", "Value", title=f"server config ({payload.get('config_path')})")
    for dotted, value in sorted(_flatten(config)):
        table.add_row(dotted, _short(value))
    STATE.console.print(table)


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            out.extend(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
    else:
        out.append((prefix, value))
    return out


@config_app.command("set")
def config_set(
    pairs: list[str] = typer.Argument(..., help="key=value pairs, e.g. models.default_ctx=8192"),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Change server configuration keys."""
    # This writes config.yaml on the rig and can disable auth or repoint the
    # model directory. Every other server-mutating command here is gated; this
    # one was not, so an agent or a typo could reconfigure the server with no
    # prompt and no tty check.
    _confirm(f"change server config on the rig ({len(pairs)} key(s))?", yes=yes)

    async def work(client: StudioForgeClient) -> Any:
        current = await client.get_config()
        config = current.get("config") or {}
        updates: dict[str, Any] = {}
        for key, raw in _split_pairs(pairs):
            cursor: Any = config
            for part in key.split("."):
                cursor = cursor.get(part) if isinstance(cursor, dict) else None
            updates[key] = _coerce_like(cursor, raw)
        return await client.set_config(updates)

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return
    STATE.console.print(f"updated: {', '.join(result.get('updated') or [])}")
    restart = result.get("restart_required") or []
    if restart:
        STATE.console.print(
            f"restart required for: {', '.join(restart)}  (sfctl recover --restart)"
        )


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@app.command()
def update(
    engine: str | None = typer.Option(
        None, "--engine", help="Install this llama.cpp engine tag instead of updating the app."
    ),
    check: bool = typer.Option(False, "--check", help="Only report current vs latest."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Update the llama.cpp engine (--engine TAG) or the server itself.

    Installing either restarts things, so it needs confirmation; ``--check``
    never changes anything.
    """
    if not check and not yes:
        target = f"install llama.cpp engine {engine}" if engine else "install the server update"
        _confirm(f"{target} (restarts the server)?", yes=yes)

    async def work(client: StudioForgeClient) -> Any:
        if engine:
            info = await client.engine()
            if check:
                releases: Any
                try:
                    releases = await client.engine_releases(10)
                except CompanionError as exc:
                    releases = {"error": str(exc)}
                return {"mode": "engine-check", "engine": info, "releases": releases}
            if want_json(json_out):
                return {
                    "mode": "engine-install",
                    "result": await client.engine_install(engine),
                }
            with STATE.console.status(
                f"installing engine {engine} (downloads, extracts, smoke-tests)...", spinner="dots"
            ):
                return {"mode": "engine-install", "result": await client.engine_install(engine)}

        if check:
            return {"mode": "app-check", "status": await client.update_status(check=True)}
        if want_json(json_out):
            return {"mode": "app-install", "result": await client.update_install(confirm=True)}
        with STATE.console.status("installing update (drains, verifies, can roll back)..."):
            return {"mode": "app-install", "result": await client.update_install(confirm=True)}

    result = with_client(work)
    if want_json(json_out):
        emit(result)
        return

    console = STATE.console
    mode = result["mode"]
    if mode == "engine-check":
        info = result["engine"] or {}
        active = info.get("active") or {}
        table = _table("Field", "Value", title="llama.cpp engine")
        table.add_row("pinned tag", str(info.get("pinned_tag")))
        table.add_row("active tag", str(active.get("tag") or "not installed"))
        table.add_row("variant", str(active.get("variant") or "-"))
        table.add_row(
            "installed", ", ".join(str(e.get("tag")) for e in info.get("installed") or []) or "-"
        )
        releases = result.get("releases")
        tags = _release_tags(releases)
        if tags:
            table.add_row("latest available", tags[0])
            table.add_row("recent", ", ".join(tags[:5]))
        elif isinstance(releases, dict) and releases.get("error"):
            table.add_row("latest available", f"unavailable: {releases['error']}")
        console.print(table)
        return

    if mode == "app-check":
        payload = result["status"] or {}
        table = _table("Field", "Value", title="server update")
        for key in (
            "current_version",
            "current_release",
            "latest_version",
            "latest_tag",
            "update_available",
            "previous_release",
            "checked_at",
            "error",
        ):
            if payload.get(key) is not None:
                table.add_row(key, str(payload[key]))
        console.print(table)
        if payload.get("update_available"):
            console.print("run 'sfctl update --yes' to install it")
        return

    console.print(json.dumps(result.get("result"), indent=2, default=str))


def _release_tags(releases: Any) -> list[str]:
    """Tag list from either shape a releases endpoint may return.

    ``engine/releases`` yields plain tag strings and ``update/releases`` yields
    objects; both spellings show up depending on which endpoint answered.
    """
    if isinstance(releases, dict):
        items = releases.get("releases") or []
    elif isinstance(releases, list):
        items = releases
    else:
        return []
    tags: list[str] = []
    for item in items:
        if isinstance(item, dict):
            tag = item.get("tag") or item.get("version") or item.get("name")
            if tag:
                tags.append(str(tag))
        elif item:
            tags.append(str(item))
    return tags


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(
    model: str,
    system: str | None = typer.Option(None, "--system", help="System prompt."),
    temp: float | None = typer.Option(None, "--temp", help="Temperature."),
) -> None:
    """Interactive streaming chat. '/exit' quits; Ctrl-C exits 0.

    Goes through the OpenAI endpoint on purpose, so this exercises the same
    just-in-time load path any real client would.
    """
    console = STATE.console
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    async def work(client: StudioForgeClient) -> None:
        console.print(f"chat with {model} -- /exit to quit, Ctrl-C to abort")
        while True:
            try:
                # Off-thread so the event loop is free while the user types.
                line = await asyncio.to_thread(input, "you> ")
            except EOFError:
                return
            text = line.strip()
            if not text:
                continue
            if text in ("/exit", "/quit"):
                return
            messages.append({"role": "user", "content": text})
            prompt_label = "assistant> " if STATE.no_color else "[bold]assistant>[/bold] "
            console.print(prompt_label, end="")
            started = time.perf_counter()
            chunks = 0
            reply: list[str] = []
            async for piece in client.chat_stream(model, messages, temperature=temp):
                chunks += 1
                reply.append(piece)
                console.print(piece, end="", highlight=False, markup=False)
            elapsed = max(time.perf_counter() - started, 1e-6)
            tokens = int(client.last_usage.get("completion_tokens") or 0) or chunks
            console.print()
            console.print(
                f"  [{tokens} tok in {elapsed:.1f}s = {tokens / elapsed:.1f} tok/s]",
                highlight=False,
            )
            messages.append({"role": "assistant", "content": "".join(reply)})

    with_client(work)


# ---------------------------------------------------------------------------
# recover (watchdog)
# ---------------------------------------------------------------------------


@app.command()
def recover(
    restart: bool = typer.Option(False, "--restart", help="Restart the main server process."),
    kill: str | None = typer.Option(None, "--kill", help="Kill one model's llama-server child."),
    nuke: bool = typer.Option(False, "--nuke", help="Kill every model child."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Talk to the WATCHDOG, which answers when the main server is wedged.

    With no flags this prints the watchdog's own health diagnosis. The watchdog
    is a separate process on a separate port (default 1235) precisely so this
    command still works when nothing else does.
    """
    from studioforge_companion.mcp_proxy import (
        call_watchdog_tool,
        describe_exception,
        probe_watchdog_auth,
        result_text,
    )

    if restart:
        _confirm("restart the StudioForge server (drops in-flight requests)?", yes=yes)
    if nuke:
        _confirm("kill every loaded model?", yes=yes)
    if kill:
        # The watchdog resolves this alias exact -> case-insensitive -> substring,
        # so "gemma" can match more than one loaded model. Name what was asked
        # for; the operator is the only one who knows which they meant.
        _confirm(f"SIGKILL the model child matching {kill!r}?", yes=yes)

    profile = _resolve_profile()
    # The user already confirmed above (or passed --yes), so the watchdog's own
    # confirm gate is satisfied here. Sending {} made the tool return its
    # "needs confirmation" refusal -- printed as success with exit 0, i.e. a
    # recover command that never recovered anything. And the kill tool's
    # parameter is model_name, not model_id; the wrong name failed schema
    # validation.
    tool, arguments = "health", {}
    if restart:
        tool, arguments = "restart_server", {"confirm": True}
    elif kill:
        tool, arguments = "kill_model", {"model_name": kill}
    elif nuke:
        tool, arguments = "nuke_all_models", {"confirm": True}

    async def work() -> Any:
        try:
            result = await call_watchdog_tool(profile, tool, arguments)
        except Exception as exc:
            detail = describe_exception(exc)
            lowered = detail.lower()
            refused = "401" in detail or "credential" in lowered or "unauthor" in lowered
            if not refused:
                # The MCP client hides the status code; ask the watchdog directly.
                refused = await probe_watchdog_auth(profile) == "unauthorized"
            if refused:
                # The watchdog guards the recovery tools with the same
                # credential as the main server: server.api_key, or the MCP
                # pairing PIN when no key is set. "Cannot reach" would send the
                # user off to check a process that is up and answering.
                have = "this profile has a key" if profile.api_key else "this profile has no key"
                raise CompanionError(
                    f"the StudioForge watchdog at {profile.watchdog_mcp_url} needs a credential "
                    f"({have}).\n"
                    "  - it accepts the server's API key, or the MCP pairing PIN when no key "
                    "is set\n"
                    "  - the PIN is on the control panel: Setup -> Network & access -> the eye "
                    "button next to 'MCP pairing PIN' (or `studioforge config` on the host)\n"
                    f"  - pair this profile with it: sfctl servers add {profile.name} "
                    f"{profile.url} --api-key <PIN> --use"
                ) from None
            raise CompanionError(
                f"cannot reach the StudioForge watchdog at {profile.watchdog_mcp_url} "
                f"({detail}).\n"
                f"  - the watchdog is a separate process: check it is running on the host\n"
                f"  - it defaults to port 1235; set 'servers.<name>.watchdog_url' if it moved"
            ) from None
        return {
            "tool": tool,
            "is_error": bool(result.is_error),
            "text": result_text(result),
            "structured": result.structured_content,
        }

    payload = run(work())
    if want_json(json_out):
        emit(payload)
        if payload["is_error"]:
            raise typer.Exit(1)
        return
    STATE.console.print(Panel(payload["text"] or "(no output)", title=f"watchdog: {tool}"))
    if payload["is_error"]:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# openclaw-setup
# ---------------------------------------------------------------------------


@app.command("openclaw-setup")
def openclaw_setup(
    reveal_key: bool = typer.Option(
        False, "--reveal-key", help="Print the API key in full instead of redacted."
    ),
    json_out: bool = JSON_OPTION,
) -> None:
    """Print the snippets that point OpenClaw at this server.

    The key is redacted unless ``--reveal-key`` is passed: the default has to be
    safe to paste into a chat or an issue.
    """

    async def work(client: StudioForgeClient) -> Any:
        return await client.openclaw_setup()

    payload = with_client(work)
    if not reveal_key:
        payload = _redact_tree(payload)
        inference = payload.get("inference") or {}
        if isinstance(inference, dict) and inference.get("OPENAI_API_KEY") is None:
            inference["OPENAI_API_KEY"] = "not-required"

    if want_json(json_out):
        emit(payload)
        return

    console = STATE.console
    inference = payload.get("inference") or {}
    env_lines = "\n".join(f"{key}={value}" for key, value in inference.items())
    console.print(Panel(env_lines or "(none)", title="1. inference endpoint", expand=False))
    console.print(
        Panel(
            json.dumps(payload.get("mcp") or {}, indent=2),
            title="2. MCP registration (stdio, merges management + recovery tools)",
            expand=False,
        )
    )
    companion = payload.get("companion_config") or {}
    # With auth disabled the server sends the "not-required" sentinel; printing
    # ``sfctl config-local set server.api_key=not-required`` would have the user
    # store that literal string as a key. Skip the line instead.
    lines = "\n".join(
        f"sfctl config-local set {key}={value}"
        for key, value in companion.items()
        if not (key.endswith("api_key") and value == _NO_KEY_SENTINEL)
    )
    if lines:
        console.print(Panel(lines, title="3. companion profile", expand=False))
    key_redacted = (
        not reveal_key
        and isinstance(inference, dict)
        and inference.get("OPENAI_API_KEY") not in (None, _NO_KEY_SENTINEL)
    )
    if key_redacted:
        console.print("(API key redacted -- re-run with --reveal-key to paste the real one)")


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


@app.command()
def mcp() -> None:
    """Run the stdio MCP server OpenClaw registers.

    Merges the main app's management tools with the watchdog's recovery tools
    into one tool list. Watchdog tools stay available when the main server is
    down, which is the entire reason the two planes are separate; colliding
    names are exposed with a 'recovery_' prefix.
    """
    from studioforge_companion.mcp_proxy import McpProxy

    profile = _resolve_profile()
    proxy = McpProxy(profile)
    try:
        asyncio.run(proxy.serve_stdio())
    except KeyboardInterrupt:
        raise typer.Exit(0) from None


# ---------------------------------------------------------------------------
# servers (local profiles)
# ---------------------------------------------------------------------------


@servers_app.command("list")
def servers_list(json_out: bool = JSON_OPTION) -> None:
    """List local server profiles (keys redacted)."""
    cfg = _load_local()
    rows = [profile.describe() for _, profile in sorted(cfg.servers.items())]
    if want_json(json_out):
        emit({"default": cfg.default, "config_path": str(config_path()), "servers": rows})
        return
    table = _table("Name", "URL", "API key", "Watchdog", "Default", title=str(config_path()))
    for row in rows:
        table.add_row(
            str(row["name"]),
            str(row["url"]),
            str(row["api_key"] or "-"),
            str(row["watchdog_url"]) + ("" if row["watchdog_url_explicit"] else " (derived)"),
            fmt_bool(row["name"] == cfg.default),
        )
    STATE.console.print(table)
    if not rows:
        STATE.console.print("no servers yet: sfctl servers add rig http://<host>:1234")


def _load_local() -> CompanionConfig:
    try:
        return load_companion_config()
    except CompanionConfigError as exc:
        _report(exc)
        raise typer.Exit(exc.exit_code) from None


@servers_app.command("add")
def servers_add(
    name: str,
    url: str,
    api_key: str | None = typer.Option(None, "--api-key", help="API key for this server."),
    watchdog_url: str | None = typer.Option(
        None, "--watchdog-url", help="Override the derived watchdog URL (default: port 1235)."
    ),
    timeout: float | None = typer.Option(None, "--timeout", help="Request timeout in seconds."),
    use: bool = typer.Option(False, "--use", help="Also make this the default server."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Add or update a local server profile."""
    cfg = _load_local()
    try:
        profile = ServerProfile(
            name=name,
            url=url,
            api_key=api_key,
            watchdog_url=watchdog_url,
            **({"timeout_s": timeout} if timeout is not None else {}),
        )
    except CompanionConfigError as exc:
        _report(exc)
        raise typer.Exit(exc.exit_code) from None
    except Exception as exc:
        STATE.err.print(f"error: {exc}")
        raise typer.Exit(EXIT_USAGE) from None

    cfg.servers[name] = profile
    if use or cfg.default is None:
        cfg.default = name
    path = save_companion_config(cfg)
    if want_json(json_out):
        emit({"saved": profile.describe(), "default": cfg.default, "config_path": str(path)})
        return
    STATE.console.print(f"saved server '{name}' -> {profile.url} (default={cfg.default}) in {path}")


@servers_app.command("remove")
def servers_remove(name: str, json_out: bool = JSON_OPTION) -> None:
    """Remove a local server profile."""
    cfg = _load_local()
    if name not in cfg.servers:
        STATE.err.print(
            f"error: unknown server {name!r}. Known: {', '.join(sorted(cfg.servers)) or '(none)'}"
        )
        raise typer.Exit(EXIT_USAGE)
    del cfg.servers[name]
    if cfg.default == name:
        cfg.default = next(iter(sorted(cfg.servers)), None)
    path = save_companion_config(cfg)
    if want_json(json_out):
        emit({"removed": name, "default": cfg.default, "config_path": str(path)})
        return
    STATE.console.print(f"removed '{name}' (default={cfg.default})")


@servers_app.command("use")
def servers_use(name: str, json_out: bool = JSON_OPTION) -> None:
    """Set the default server profile."""
    cfg = _load_local()
    if name not in cfg.servers:
        STATE.err.print(
            f"error: unknown server {name!r}. Known: {', '.join(sorted(cfg.servers)) or '(none)'}"
        )
        raise typer.Exit(EXIT_USAGE)
    cfg.default = name
    save_companion_config(cfg)
    if want_json(json_out):
        emit({"default": name})
        return
    STATE.console.print(f"default server is now '{name}'")


@app.command("config-local")
def config_local(
    action: str = typer.Argument(..., help="'set' or 'path'."),
    pairs: list[str] = typer.Argument(None, help="key=value, e.g. servers.rig.api_key=sf-..."),
    json_out: bool = JSON_OPTION,
) -> None:
    """Edit the LOCAL companion config (as opposed to 'sfctl config', the server's)."""
    from studioforge_companion.config import set_value

    if action == "path":
        path = str(config_path())
        if want_json(json_out):
            emit({"config_path": path})
        else:
            typer.echo(path)
        return
    if action != "set":
        STATE.err.print("error: action must be 'set' or 'path'")
        raise typer.Exit(EXIT_USAGE)
    try:
        items = _split_pairs(pairs or [])
        for key, value in items:
            set_value(key, value)
    except CompanionConfigError as exc:
        _report(exc)
        raise typer.Exit(exc.exit_code) from None
    if want_json(json_out):
        emit({"updated": [key for key, _ in items], "config_path": str(config_path())})
        return
    STATE.console.print(f"updated {', '.join(key for key, _ in items)} in {config_path()}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
