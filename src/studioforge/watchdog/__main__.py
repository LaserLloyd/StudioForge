"""``python -m studioforge.watchdog`` -- start the recovery watchdog sidecar.

Kept deliberately tiny and dependency-light. This process is spawned by
``studioforge serve`` (and can be run standalone, or as a systemd unit /
Windows service), and its entire value proposition is that it starts and keeps
answering when the main application cannot. So there is no typer/click CLI here,
no rich console, no app imports: ``argparse`` from the stdlib, stdlib
``logging``, and :mod:`studioforge.watchdog.server`.

Logging goes to ``<data_dir>/logs/watchdog.log`` as well as stderr. The file
matters more than usual here: when the watchdog is the only thing that was
running during an incident, its log is the only account of what happened.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

from studioforge.config import Config, find_config_path
from studioforge.watchdog.server import Watchdog, serve

LOG_FORMAT = "%(asctime)s %(levelname)s [watchdog] %(message)s"


def configure_logging(config: Config, level: str = "INFO") -> Path | None:
    """Send watchdog logs to stderr and to ``<data_dir>/logs/watchdog.log``.

    Uses stdlib logging rather than the app's structlog setup: importing
    :mod:`studioforge.logging` would be harmless today, but the watchdog's
    dependency surface is a deliberate invariant and log formatting is not worth
    spending it on. A log directory that cannot be created is not fatal -- stderr
    still works, and refusing to start would remove the only recovery surface.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(stream)

    log_path: Path | None = None
    try:
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = config.logs_dir / "watchdog.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root.addHandler(file_handler)
    except OSError as exc:  # pragma: no cover - unwritable data dir
        log_path = None
        root.warning("could not open watchdog log file: %s", exc)
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="studioforge-watchdog",
        description=(
            "StudioForge recovery watchdog: a separate always-on MCP server that can "
            "inspect, reconfigure, restart and roll back a StudioForge instance even "
            "when that instance is wedged."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.yaml (defaults to $SF_CONFIG, else <data_dir>/config.yaml)",
    )
    parser.add_argument("--host", default=None, help="Override watchdog.host")
    parser.add_argument("--port", type=int, default=None, help="Override watchdog.port")
    parser.add_argument(
        "--path", default="/mcp", help="URL path for the streamable-HTTP MCP endpoint"
    )
    parser.add_argument(
        "--no-poll",
        action="store_true",
        help="Do not run the background health poll (tools still work on demand)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = find_config_path(args.config)

    # Load through the Watchdog itself so a broken config degrades to defaults
    # here exactly as it does for every tool call: the watchdog must start even
    # when config.yaml is the thing that is broken -- that is when its
    # set_config escape hatch is needed most.
    watchdog = Watchdog(config_path)
    config, config_error = watchdog.load_config()
    log_path = configure_logging(config, args.log_level)
    log = logging.getLogger("studioforge.watchdog")

    log.warning("watchdog starting (config=%s, log=%s)", config_path, log_path)
    if config_error is not None:
        log.error("config is not usable, running on schema defaults: %s", config_error)

    try:
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(
                serve(
                    config_path,
                    host=args.host,
                    port=args.port,
                    path=args.path,
                    poll=not args.no_poll,
                )
            )
    except Exception as exc:  # noqa: BLE001 - report, do not traceback into the void
        log.exception("watchdog exited with an error: %s", exc)
        return 1
    log.warning("watchdog stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
