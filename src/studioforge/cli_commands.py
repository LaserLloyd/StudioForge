"""The ``studioforge`` command line, ``lms``-style: the commands people reach for.

``setup``, ``ls``, ``ps``, ``load``, ``unload``, ``get``, ``server``, ``log``,
``keys``, ``users``, ``update``, ``doctor``, ``support-bundle``, ``migrate``.

Commands that act on the running service (``ls``, ``ps``, ``load``, ``unload``,
``get``, ``update``, ``log stream``, ``server stop``) are thin clients of the
local HTTP API (:class:`studioforge.cli_client.LocalClient`), so they work against
the live process. Commands that are about *who may log in* (``keys``, ``users``)
and the offline ones (``doctor``, ``support-bundle``, ``migrate``) work straight on
the data directory, so they work with the server stopped -- which is exactly when
you need ``keys create`` to fix a ``refusing to start`` message.

Registered onto the main app by :func:`register`; kept in its own module so
``__main__`` stays the process entry point and these stay testable.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from studioforge.cli_client import CliError, LocalClient

CONFIG_OPTION = typer.Option(None, "--config", "-c", help="Path to config.yaml")
JSON_OPTION = typer.Option(False, "--json", help="Print machine-readable JSON.")

keys_app = typer.Typer(help="API keys: named, scoped, revocable.", no_args_is_help=True)
users_app = typer.Typer(help="Owner accounts for the control panel.", no_args_is_help=True)
server_app = typer.Typer(help="Start, stop and inspect the local server.", no_args_is_help=True)
log_app = typer.Typer(help="Server logs.", no_args_is_help=True)
update_app = typer.Typer(help="Check for, install and roll back updates.", no_args_is_help=True)


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------


def _config(config_path: Path | None) -> Any:
    from studioforge.__main__ import _load

    return _load(config_path)


def _emit(data: Any) -> None:
    typer.echo(json.dumps(data, indent=2, default=str))


def _fail(exc: CliError) -> typer.Exit:
    typer.echo(f"error: {exc.message}", err=True)
    if exc.fix:
        typer.echo(f"  fix: {exc.fix}", err=True)
    return typer.Exit(1)


def _run[R](config_path: Path | None, work: Callable[[LocalClient, Any], R]) -> R:
    """Run ``work(client, config)`` against the local server, turning errors into one line."""
    config = _config(config_path)
    try:
        with LocalClient(config) as client:
            return work(client, config)
    except CliError as exc:
        raise _fail(exc) from exc


def _gib(value: Any) -> str:
    try:
        return f"{float(value) / 1024**3:.1f} GiB"
    except (TypeError, ValueError):
        return "-"


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    out = [line, "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)) for row in rows]
    return "\n".join(out)


def _read_password(
    *, password_env: str | None, from_stdin: bool, prompt: str, confirm: bool = True
) -> str:
    if password_env:
        value = os.environ.get(password_env, "")
        if not value:
            raise CliError(f"The environment variable {password_env} is empty or not set.")
        return value
    if from_stdin:
        return sys.stdin.readline().rstrip("\r\n")
    return str(typer.prompt(prompt, hide_input=True, confirmation_prompt=confirm))


# ---------------------------------------------------------------------------
# models: ls / ps / load / unload / get
# ---------------------------------------------------------------------------


def ls_cmd(
    loaded: bool = typer.Option(False, "--loaded", help="Only models that are loaded now."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """List the models in the library."""

    def work(client: LocalClient, _config: Any) -> Any:
        payload = client.get("/api/models")
        return payload.get("models", payload) if isinstance(payload, dict) else payload

    models = [m for m in _run(config_path, work) if not (loaded and not m.get("loaded"))]
    if json_out:
        _emit(models)
        return
    if not models:
        typer.echo("No models yet. Download one with `studioforge get <owner/repo>`.")
        return
    rows = [
        [
            str(m.get("id")),
            _gib(m.get("size_bytes")),
            str(m.get("quant") or "-"),
            str(m.get("architecture") or "-"),
            str(m.get("state") or ("loaded" if m.get("loaded") else "-")),
        ]
        for m in models
    ]
    typer.echo(_table(rows, ["MODEL", "SIZE", "QUANT", "ARCH", "STATE"]))


def ps_cmd(
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Show the models loaded right now, where, and what they are doing."""
    status = _run(config_path, lambda client, _c: client.get("/api/status"))
    loaded = status.get("loaded", [])
    if json_out:
        _emit(loaded)
        return
    if not loaded:
        typer.echo("Nothing is loaded.")
        return
    rows = []
    for row in loaded:
        plan = row.get("plan") or {}
        rows.append(
            [
                str(row.get("model_id")),
                str(row.get("state")),
                ",".join(str(d) for d in plan.get("devices", [])) or "-",
                str(plan.get("ctx_size") or "-"),
                str(row.get("ttl_s") if row.get("ttl_s") is not None else "-"),
                str(row.get("active_requests", 0)),
            ]
        )
    typer.echo(_table(rows, ["MODEL", "STATE", "GPUS", "CONTEXT", "TTL(s)", "BUSY"]))


def load_cmd(
    key: str = typer.Argument(..., help="Model id (see `studioforge ls`)."),
    ttl: int | None = typer.Option(None, "--ttl", min=1, help="Idle seconds before it unloads."),
    context_length: int | None = typer.Option(
        None, "--context-length", "--ctx", min=1, help="Context window per conversation."
    ),
    gpu: str | None = typer.Option(None, "--gpu", help="GPU indices to use, e.g. 0 or 0,1."),
    preset: str | None = typer.Option(
        None, "--preset", help="Placement mode for the recommended load (see `GET /api/catalog`)."
    ),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Load a model. Without options the server picks the context and the cards."""
    devices = None
    if gpu is not None:
        try:
            devices = [int(part) for part in gpu.split(",") if part.strip() != ""]
        except ValueError as exc:
            raise typer.BadParameter("--gpu takes GPU indices like 0 or 0,1") from exc

    def work(client: LocalClient, config: Any) -> Any:
        if preset is not None:
            body: dict[str, Any] = {
                "ctx_size": context_length or int(config.models.default_ctx),
                "prefer_mode": preset,
            }
            if devices:
                body["allowed_devices"] = devices
            result = client.post(f"/api/models/{key}/load-recommended", body)
        elif devices:
            body = {"devices": devices}
            if context_length:
                body["ctx_size"] = context_length
            result = client.post(f"/api/models/{key}/load", body)
        else:
            body = {"model": key}
            if context_length:
                body["context_length"] = context_length
            if ttl:
                body["ttl"] = ttl
            return client.post("/api/v1/models/load", body)
        if ttl:
            # The per-request idle timeout only exists on the LM Studio native route;
            # the model is already resident, so this just records it.
            client.post("/api/v1/models/load", {"model": key, "ttl": ttl})
        return result

    result = _run(config_path, work)
    if json_out:
        _emit(result)
        return
    ctx = (
        (result.get("context_length") or result.get("ctx_size"))
        if isinstance(result, dict)
        else None
    )
    typer.echo(f"Loaded {key}" + (f" (context {ctx})" if ctx else "") + ".")
    for warning in (result.get("warnings") or []) if isinstance(result, dict) else []:
        typer.echo(f"warning: {warning}", err=True)


def unload_cmd(
    key: str | None = typer.Argument(None, help="Model id; omit with --all."),
    all_models: bool = typer.Option(False, "--all", help="Unload every loaded model."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Unload a model (or every model with --all) and free its VRAM."""
    if bool(key) == all_models:
        raise typer.BadParameter("give a model id, or --all (not both, not neither)")

    def work(client: LocalClient, _config: Any) -> Any:
        if all_models:
            return client.post("/api/models/unload-all", {})
        return client.post("/api/v1/models/unload", {"model": key})

    result = _run(config_path, work)
    if json_out:
        _emit(result)
        return
    typer.echo("Unloaded every model." if all_models else f"Unloaded {key}.")


def get_cmd(
    repo: str = typer.Argument(..., help="Hugging Face repo, e.g. unsloth/Qwen3-8B-GGUF."),
    quant: str | None = typer.Option(
        None, "--quant", "-q", help="Quantization (default: best fit)."
    ),
    no_wait: bool = typer.Option(False, "--no-wait", help="Queue it and return."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Download a model from Hugging Face (resumable; the quant is picked to fit)."""

    def work(client: LocalClient, _config: Any) -> Any:
        started = client.post("/api/downloads", {"repo_id": repo, "quant": quant})
        if no_wait:
            return started
        group = started["group_id"]
        typer.echo(
            f"Downloading {repo} ({started.get('quant')}, {_gib(started.get('total_bytes'))})"
        )
        last = -1
        while True:
            rows = [
                r
                for r in client.get("/api/downloads").get("downloads", [])
                if r["group_id"] == group
            ]
            statuses = {str(r.get("status")) for r in rows}
            if statuses & {"failed", "canceled"}:
                raise CliError(
                    f"The download {', '.join(sorted(statuses))}.",
                    "Re-run `studioforge get` to resume it; partial files are kept.",
                )
            done = sum(int(r.get("downloaded_bytes") or 0) for r in rows)
            total = sum(int(r.get("total_bytes") or 0) for r in rows) or 1
            percent = 100 * done // total
            if percent != last and not json_out:
                typer.echo(f"  {percent}%")
                last = percent
            if rows and statuses <= {"completed"}:
                return started
            time.sleep(2)

    result = _run(config_path, work)
    if json_out:
        _emit(result)
    elif not no_wait:
        typer.echo("Done. Load it with `studioforge load <id>`; `studioforge ls` lists the ids.")


# ---------------------------------------------------------------------------
# server / log / update
# ---------------------------------------------------------------------------


@server_app.command("status")
def server_status_cmd(
    json_out: bool = JSON_OPTION, config_path: Path | None = CONFIG_OPTION
) -> None:
    """Is the server running, and where?"""
    from studioforge.server_control import server_status

    status = server_status(_config(config_path))
    if json_out:
        _emit(status.to_dict())
    elif status.running:
        extra = f", up {status.uptime_s:.0f}s" if status.uptime_s else ""
        typer.echo(f"running at {status.url} (version {status.version}{extra}, pid {status.pid})")
    else:
        typer.echo(f"not running (would listen on {status.url})")
        raise typer.Exit(1)


@server_app.command("start")
def server_start_cmd(
    wait: float = typer.Option(90.0, "--wait", help="Seconds to wait for it to answer."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Start the server in the background and wait until it answers."""
    from studioforge.server_control import start_server

    config = _config(config_path)
    try:
        status = start_server(config, wait_s=wait)
    except CliError as exc:
        raise _fail(exc) from exc
    typer.echo(f"started: {status.url} (version {status.version}, pid {status.pid})")


@server_app.command("stop")
def server_stop_cmd(
    wait: float = typer.Option(30.0, "--wait", help="Seconds to wait for a graceful stop."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Stop the server (drains, unloads every model, exits)."""
    from studioforge.server_control import stop_server

    config = _config(config_path)
    try:
        with LocalClient(config) as client:
            stopped = stop_server(config, wait_s=wait, client=client)
    except CliError as exc:
        raise _fail(exc) from exc
    typer.echo("stopped." if stopped else "not running.")


@log_app.command("stream")
def log_stream_cmd(
    n: int = typer.Option(50, "-n", help="Lines to show first."),
    follow: bool = typer.Option(True, "--follow/--no-follow", help="Keep printing new lines."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Print the server log and keep following it. Ctrl-C exits."""
    config = _config(config_path)
    seen: list[str] = []
    try:
        with LocalClient(config) as client:
            while True:
                lines = list(client.log_lines(max(n, 200)))
                fresh = _new_lines(seen, lines)
                if not seen:
                    fresh = fresh[-n:]
                for line in fresh:
                    typer.echo(line)
                seen = lines
                if not follow:
                    return
                time.sleep(1.5)
    except KeyboardInterrupt:
        return
    except CliError as exc:
        raise _fail(exc) from exc


def _new_lines(previous: list[str], current: list[str]) -> list[str]:
    """Lines in ``current`` not already printed (the endpoint returns a tail, not a cursor)."""
    if not previous:
        return list(current)
    for size in range(min(len(previous), len(current)), 0, -1):
        if previous[-size:] == current[:size]:
            return current[size:]
    return list(current)


@update_app.command("check")
def update_check_cmd(
    json_out: bool = JSON_OPTION, config_path: Path | None = CONFIG_OPTION
) -> None:
    """Compare the running version with the latest release."""
    result = _run(config_path, lambda client, _c: client.get("/api/update", check="true"))
    if json_out:
        _emit(result)
        return
    if not result.get("configured", True):
        typer.echo(
            f"current {result.get('current_version')}: update checks are off "
            "(update.repo is not set). " + str(result.get("note") or "")
        )
        return
    if result.get("error"):
        typer.echo(f"current {result.get('current_version')}: the check failed: {result['error']}")
        raise typer.Exit(1)
    typer.echo(
        f"current {result.get('current_version')}, latest "
        f"{result.get('latest_version') or 'unknown'}: "
        + ("an update is available." if result.get("update_available") else "up to date.")
    )


@update_app.command("apply")
def update_apply_cmd(
    tag: str | None = typer.Option(
        None, "--tag", help="Install this release instead of the latest."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask first."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Install the latest release (restarts the server)."""
    if not yes:
        typer.confirm("Installing an update restarts the server. Continue?", abort=True)
    result = _run(
        config_path,
        lambda client, _c: client.post("/api/update/install", {"tag": tag, "confirm": True}),
    )
    _emit(result) if json_out else typer.echo("Update installed; the server is restarting.")


@update_app.command("rollback")
def update_rollback_cmd(
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask first."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Go back to the previous release (restarts the server)."""
    if not yes:
        typer.confirm("Rolling back restarts the server. Continue?", abort=True)
    result = _run(
        config_path, lambda client, _c: client.post("/api/update/rollback", {"confirm": True})
    )
    _emit(result) if json_out else typer.echo("Rolled back; the server is restarting.")


# ---------------------------------------------------------------------------
# keys / users (straight on the data directory)
# ---------------------------------------------------------------------------


def _store(config: Any) -> tuple[Any, Any]:
    from studioforge.core.accounts import AccountStore
    from studioforge.db import Database

    db = Database(config.db_path)
    db.migrate_with_recovery()
    return db, AccountStore(db, config.data_dir)


@keys_app.command("create")
def keys_create_cmd(
    name: str = typer.Option(..., "--name", "-n", help="A label, e.g. openclaw or my-agent."),
    scope: str = typer.Option(
        "inference", "--scope", "-s", help="read (look only) | inference (chat, load) | admin."
    ),
    expires_in_days: float | None = typer.Option(None, "--expires-in-days", min=0.01),
    rate_limit_rpm: int | None = typer.Option(None, "--rate-limit-rpm", min=1),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Create an API key. The token is printed once and cannot be shown again.

    Creating the first key turns authentication on: every client (on this machine
    too) then needs a key. The CLI, the tray and the panel on this machine get
    their own automatically.
    """
    config = _config(config_path)
    db, store = _store(config)
    try:
        first = not store.has_active_keys()
        record, token = store.create_key(
            name,
            scope,
            expires_in_days=expires_in_days,
            rate_limit_rpm=rate_limit_rpm,
            created_by="cli",
        )
        store.local_cli_key()
    except Exception as exc:  # noqa: BLE001 - one readable line, not a traceback
        message = getattr(exc, "message", str(exc))
        typer.echo(f"error: {message}", err=True)
        raise typer.Exit(1) from exc
    finally:
        db.close()
    if json_out:
        _emit({"key": record.public(), "token": token})
        return
    typer.echo(f"Created key '{record.name}' (id {record.id}, scope {record.scope}).")
    typer.echo("")
    typer.echo(f"  {token}")
    typer.echo("")
    typer.echo("Copy it now: it is not stored and cannot be shown again.")
    if first:
        typer.echo(
            "Authentication is now ON: clients must send this (or another) key as "
            "'Authorization: Bearer <key>'."
        )


@keys_app.command("list")
def keys_list_cmd(
    all_keys: bool = typer.Option(False, "--all", help="Include revoked keys."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """List keys (never the tokens)."""
    config = _config(config_path)
    db, store = _store(config)
    try:
        keys = store.list_keys(include_revoked=all_keys)
    finally:
        db.close()
    if json_out:
        _emit([k.public() for k in keys])
        return
    if not keys:
        typer.echo("No keys. Create one with `studioforge keys create --name <label>`.")
        return

    def when(value: float | None) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(value)) if value else "-"

    rows = [
        [
            k.id,
            k.name,
            k.scope,
            k.prefix,
            when(k.last_used_at),
            when(k.expires_at),
            str(k.rate_limit_rpm or "-"),
            "revoked" if k.revoked else ("expired" if k.expired() else "active"),
        ]
        for k in keys
    ]
    typer.echo(
        _table(rows, ["ID", "NAME", "SCOPE", "PREFIX", "LAST USED", "EXPIRES", "RPM", "STATE"])
    )


@keys_app.command("revoke")
def keys_revoke_cmd(
    key: str = typer.Argument(..., help="Key id (or its name)."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Revoke a key. It stops working immediately."""
    config = _config(config_path)
    db, store = _store(config)
    try:
        record = store.revoke_key(key)
    finally:
        db.close()
    if record is None:
        typer.echo(
            f"error: no key '{key}'. List them with `studioforge keys list --all`.", err=True
        )
        raise typer.Exit(1)
    typer.echo(f"Revoked '{record.name}' ({record.id}).")


@users_app.command("create")
def users_create_cmd(
    username: str = typer.Argument(..., help="Username (the first account is the owner)."),
    password_env: str | None = typer.Option(
        None, "--password-env", help="Read it from this env var."
    ),
    password_stdin: bool = typer.Option(False, "--password-stdin", help="Read it from stdin."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Create an account for the control panel (and satisfy the network-bind rule)."""
    config = _config(config_path)
    db, store = _store(config)
    try:
        password = _read_password(
            password_env=password_env, from_stdin=password_stdin, prompt="Password (10+ characters)"
        )
        account = store.create_account(username, password)
    except CliError as exc:
        raise _fail(exc) from exc
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: {getattr(exc, 'message', exc)}", err=True)
        raise typer.Exit(1) from exc
    finally:
        db.close()
    typer.echo(f"Created account '{account.username}' ({account.role}).")


@users_app.command("passwd")
def users_passwd_cmd(
    username: str = typer.Argument(...),
    password_env: str | None = typer.Option(None, "--password-env"),
    password_stdin: bool = typer.Option(False, "--password-stdin"),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Change an account's password."""
    config = _config(config_path)
    db, store = _store(config)
    try:
        password = _read_password(
            password_env=password_env, from_stdin=password_stdin, prompt="New password"
        )
        account = store.set_password(username, password)
    except CliError as exc:
        raise _fail(exc) from exc
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: {getattr(exc, 'message', exc)}", err=True)
        raise typer.Exit(1) from exc
    finally:
        db.close()
    typer.echo(f"Changed the password of '{account.username}'.")


@users_app.command("list")
def users_list_cmd(json_out: bool = JSON_OPTION, config_path: Path | None = CONFIG_OPTION) -> None:
    """List accounts."""
    config = _config(config_path)
    db, store = _store(config)
    try:
        accounts = store.list_accounts()
    finally:
        db.close()
    if json_out:
        _emit([a.public() for a in accounts])
        return
    if not accounts:
        typer.echo("No accounts. Create the owner with `studioforge users create <name>`.")
        return
    typer.echo(_table([[a.username, a.role] for a in accounts], ["USERNAME", "ROLE"]))


# ---------------------------------------------------------------------------
# doctor / support-bundle / migrate
# ---------------------------------------------------------------------------

_MARK = {"ok": "ok  ", "info": "info", "warn": "WARN", "fail": "FAIL"}


def doctor_cmd(
    json_out: bool = JSON_OPTION,
    strict: bool = typer.Option(False, "--strict", help="Exit non-zero on warnings too."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Check ports, GPU, engine, disk, permissions, certificate, migrations and auth."""
    from studioforge.core.doctor import run_doctor
    from studioforge.core.engine import EngineManager
    from studioforge.core.gpu import get_probe

    config = _config(config_path)
    probe = get_probe()
    report = run_doctor(
        config,
        probe=probe,
        engine_manager=EngineManager(config, probe=probe),
        accounts=_doctor_accounts(config),
    )
    if json_out:
        _emit(report.to_dict())
    else:
        for finding in report.findings:
            typer.echo(f"[{_MARK[finding.severity]}] {finding.title}: {finding.message}")
            if finding.fix and finding.severity in {"warn", "fail"}:
                typer.echo(f"        fix: {finding.fix}")
        counts = report.counts()
        typer.echo(
            f"\n{counts['ok']} ok, {counts['info']} info, {counts['warn']} warnings, "
            f"{counts['fail']} failures."
        )
    if not report.ok or (strict and report.worst == "warn"):
        raise typer.Exit(1)


def _doctor_accounts(config: Any) -> Any:
    """An :class:`AccountStore` for the doctor, or ``None`` when there is no database yet."""
    if not Path(config.db_path).is_file():
        return None
    try:
        _db, store = _store(config)
        return store
    except Exception:  # noqa: BLE001 - the doctor reports around a database it cannot open
        return None


def support_bundle_cmd(
    output: Path | None = typer.Option(None, "--output", "-o", help="Where to write the zip."),
    lines: int = typer.Option(2000, "--log-lines", help="Log lines to include per file."),
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Write a zip for bug reports: redacted config, doctor, versions, GPUs, log tails."""
    from studioforge.core.gpu import get_probe
    from studioforge.core.support_bundle import build_bundle

    config = _config(config_path)
    payload = build_bundle(config, probe=get_probe(), log_lines=lines)
    target = output or Path(f"studioforge-support-{time.strftime('%Y%m%d-%H%M%S')}.zip")
    target.write_bytes(payload)
    typer.echo(
        f"Wrote {target} ({len(payload) // 1024} KiB). Credentials are removed; "
        "look before you share."
    )


def migrate_cmd(
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change; touch nothing."),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Apply pending database migrations (after a backup), or preview them with --dry-run."""
    import sqlite3

    from studioforge.db import SCHEMA_VERSION, Database, _migration_sources

    config = _config(config_path)
    path = Path(config.db_path)
    applied: set[int] = set()
    if path.is_file():
        try:
            connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=3)
            try:
                applied = {
                    r[0] for r in connection.execute("SELECT version FROM schema_migrations")
                }
            finally:
                connection.close()
        except sqlite3.Error as exc:
            typer.echo(f"error: the database could not be read: {exc}", err=True)
            raise typer.Exit(1) from exc
    pending = [(v, name) for v, name, _sql in _migration_sources() if v not in applied]
    summary: dict[str, Any] = {
        "database": str(path),
        "exists": path.is_file(),
        "current": max(applied, default=0),
        "target": SCHEMA_VERSION,
        "pending": [{"version": v, "name": n} for v, n in pending],
        "dry_run": dry_run,
    }
    if not dry_run and pending:
        backup = None
        if path.is_file():
            backup = _backup_database(path, summary["current"])
            summary["backup"] = str(backup)
        db = Database(path)
        try:
            db.migrate()
        finally:
            db.close()
        summary["applied"] = [v for v, _n in pending]
        summary["current"] = SCHEMA_VERSION
    if json_out:
        _emit(summary)
        return
    typer.echo(f"database: {path}" + ("" if path.is_file() else " (does not exist yet)"))
    typer.echo(f"schema:   {summary['current']} -> {SCHEMA_VERSION}")
    if not pending:
        typer.echo("Nothing to migrate: the database is current.")
    elif dry_run:
        typer.echo("Would apply:")
        for version, name in pending:
            typer.echo(f"  {version:03d}  {name}")
        typer.echo("(dry run: nothing was changed)")
    else:
        if summary.get("backup"):
            typer.echo(f"backup:   {summary['backup']}")
        typer.echo(f"Applied {len(pending)} migration(s).")


#: Pre-migration backups kept (newest first); older ones are removed.
BACKUPS_KEPT = 5


def _backup_database(path: Path, version: int) -> Path:
    """Copy the database to ``backups/<app version>-schema<N>-<time>/registry.sqlite3``.

    Uses SQLite's online backup API, so a database a running server holds open
    (WAL) is copied consistently. Keeps the newest :data:`BACKUPS_KEPT` folders.
    """
    import shutil
    import sqlite3

    from studioforge import __version__

    root = path.parent / "backups"
    folder = root / f"{__version__}-schema{version}-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / path.name
    source = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5)
    try:
        destination = sqlite3.connect(str(target))
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
    from studioforge.secure_files import restrict_to_owner

    restrict_to_owner(folder, directory=True)
    restrict_to_owner(target)
    for old in sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.stat().st_mtime)[
        :-BACKUPS_KEPT
    ]:
        with contextlib.suppress(OSError):
            shutil.rmtree(old)
    return target


# ---------------------------------------------------------------------------
# setup (the first-run wizard)
# ---------------------------------------------------------------------------


class _TyperPrompter:
    def ask(self, question: str, default: str | None = None, *, secret: bool = False) -> str:
        return str(
            typer.prompt(question, default=default, hide_input=secret, confirmation_prompt=secret)
        )

    def confirm(self, question: str, default: bool = True) -> bool:
        return bool(typer.confirm(question, default=default))

    def choose(self, question: str, options: list[str], default: int = 0) -> int:
        typer.echo(question)
        for i, option in enumerate(options, 1):
            typer.echo(f"  {i}) {option}")
        picked = typer.prompt("Choose", default=str(default + 1))
        try:
            index = int(picked) - 1
        except ValueError:
            return default
        return index if 0 <= index < len(options) else default


def setup_cmd(
    non_interactive: bool = typer.Option(
        False, "--non-interactive", "-y", help="Never ask; use the flags (for scripts and CI)."
    ),
    username: str | None = typer.Option(None, "--username", help="Owner account username."),
    password_env: str | None = typer.Option(
        None, "--password-env", help="Env var holding the owner password."
    ),
    password_stdin: bool = typer.Option(
        False, "--password-stdin", help="Read the password on stdin."
    ),
    skip_account: bool = typer.Option(False, "--skip-account", help="Local-only: no account."),
    library: str | None = typer.Option(
        None, "--library", help="'lmstudio', 'auto', or a folder to use as the model library."
    ),
    engine_tag: str | None = typer.Option(None, "--engine-tag", help="llama.cpp build to install."),
    skip_engine: bool = typer.Option(False, "--skip-engine"),
    model: str | None = typer.Option(
        None, "--model", help="Hugging Face repo to download, or a model id already in the library."
    ),
    skip_model: bool = typer.Option(False, "--skip-model"),
    skip_chat: bool = typer.Option(False, "--skip-chat"),
    exposure: str = typer.Option(
        "local", "--exposure", help="local (default) | lan (HTTPS on the network) | peer (later)."
    ),
    no_start: bool = typer.Option(
        False, "--no-start", help="Do not start the server for steps 5-6."
    ),
    json_out: bool = JSON_OPTION,
    config_path: Path | None = CONFIG_OPTION,
) -> None:
    """Guided first run: account, hardware, engine, library, starter model, test chat, exposure."""
    from studioforge.setup_wizard import LiveServices, SetupWizard, WizardOptions

    config = _config(config_path)
    password: str | None = None
    if password_env or password_stdin:
        try:
            password = _read_password(
                password_env=password_env, from_stdin=password_stdin, prompt="", confirm=False
            )
        except CliError as exc:
            raise _fail(exc) from exc
    options = WizardOptions(
        username=username,
        password=password,
        skip_account=skip_account,
        library=library,
        engine_tag=engine_tag,
        skip_engine=skip_engine,
        model=model,
        skip_model=skip_model,
        skip_chat=skip_chat,
        exposure=exposure,
        interactive=not non_interactive and sys.stdin.isatty(),
        start_server=not no_start,
    )
    say = (lambda _m: None) if json_out else (lambda message: typer.echo(message))
    wizard = SetupWizard(
        LiveServices(config, say=say),
        options,
        prompter=_TyperPrompter() if options.interactive else None,
        say=say,
    )
    wizard.run()
    summary = wizard.summary()
    if json_out:
        _emit(summary)
    else:
        _print_setup_summary(wizard)
    if not wizard.ok:
        raise typer.Exit(1)


def _print_setup_summary(wizard: Any) -> None:
    typer.echo("")
    for result in wizard.results:
        if result.key == "connect":
            snippets = result.data.get("snippets", {})
            typer.echo("Connect your tools")
            typer.echo(f"  base URL: {snippets.get('base_url')}")
            typer.echo(
                "  snippets for OpenClaw, the OpenAI and Anthropic SDKs, curl and MCP are in "
                "the panel under Integrations."
            )
        if result.key == "exposure" and result.data.get("admin_key"):
            typer.echo("")
            typer.echo("Admin key for your network clients (shown once, not stored):")
            typer.echo(f"  {result.data['admin_key']}")
    failed = [r for r in wizard.results if r.status == "failed"]
    typer.echo("")
    typer.echo(
        "Setup finished with problems: " + "; ".join(f"{r.title}: {r.detail}" for r in failed)
        if failed
        else "Setup complete."
    )


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def register(app: typer.Typer) -> None:
    """Attach every command and sub-command group to the main ``studioforge`` app."""
    app.command("setup")(setup_cmd)
    app.command("ls")(ls_cmd)
    app.command("ps")(ps_cmd)
    app.command("load")(load_cmd)
    app.command("unload")(unload_cmd)
    app.command("get")(get_cmd)
    app.command("doctor")(doctor_cmd)
    app.command("support-bundle")(support_bundle_cmd)
    app.command("migrate")(migrate_cmd)
    app.add_typer(server_app, name="server")
    app.add_typer(log_app, name="log")
    app.add_typer(keys_app, name="keys")
    app.add_typer(users_app, name="users")
    app.add_typer(update_app, name="update")
