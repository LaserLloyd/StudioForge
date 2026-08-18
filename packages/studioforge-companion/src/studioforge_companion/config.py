"""Local client-side configuration for ``sfctl``.

One small TOML file holds *named server profiles* rather than a single
URL/key pair, because the realistic deployment is several rigs: a workstation
with the GPUs, a laptop, maybe a spare box. ``sfctl -s laptop status`` should
be a flag, not an edit.

The file holds an API key, so writes are atomic and mode ``0600`` on POSIX, and
:func:`redact` is the only way a key is ever rendered.

Layout::

    default = "rig"

    [servers.rig]
    url = "http://100.x.y.z:1234"
    api_key = "sf-..."
    watchdog_url = "http://100.x.y.z:1235"   # optional

    [servers.laptop]
    url = "http://192.168.1.50:1234"
"""

from __future__ import annotations

import os
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import tomli_w
from pydantic import BaseModel, Field, field_validator

APP_NAME = "studioforge"
CONFIG_FILENAME = "companion.toml"

#: The watchdog listens here by default (server config ``watchdog.port``). It is
#: a *separate* process on a *separate* port precisely so it answers when the
#: main server does not, which is why the companion can derive it.
DEFAULT_WATCHDOG_PORT = 1235
DEFAULT_SERVER_PORT = 1234

#: Escape hatch used by tests and by anyone juggling several config files.
ENV_CONFIG_PATH = "SF_COMPANION_CONFIG"

#: Name used when a bare ``server.*`` key is set and no profile exists yet.
IMPLICIT_PROFILE = "default"


class CompanionConfigError(ValueError):
    """A local configuration problem: bad URL, unknown profile, unparsable file.

    Carries ``exit_code = 2`` (usage error) so the CLI's single error handler can
    map any failure -- local or remote -- to the documented exit-code contract
    without knowing which module raised it.
    """

    exit_code: int = 2


def redact(value: str | None) -> str | None:
    """Render a secret safe for a terminal, a log line or a bug report.

    Matches the server's ``studioforge.api.auth.redact`` so the same key looks
    the same on both sides, which is what makes "do these match?" answerable
    without ever showing the key.
    """
    if not value:
        return None
    if len(value) > 8:
        return f"{value[:4]}...{value[-2:]}"
    return "***"


def normalize_url(raw: str, *, field: str = "url") -> str:
    """Return ``scheme://host[:port]`` or raise :class:`CompanionConfigError`.

    Accepts what people actually type (``100.64.0.3:1234``), rejects what
    silently breaks path joining later (a trailing path component), and strips
    the trailing slash so ``f"{url}/api"`` never produces a double slash.
    """
    text = (raw or "").strip()
    if not text:
        raise CompanionConfigError(f"{field} must not be empty")
    if "://" not in text:
        text = f"http://{text}"
    parts = urlsplit(text)
    if parts.scheme not in ("http", "https"):
        raise CompanionConfigError(
            f"{field} must use http:// or https:// (got {parts.scheme!r} in {raw!r})"
        )
    if not parts.netloc:
        raise CompanionConfigError(f"{field} is missing a host: {raw!r}")
    if parts.path.strip("/"):
        raise CompanionConfigError(
            f"{field} must be a bare origin without a path -- "
            f"use {parts.scheme}://{parts.netloc} instead of {raw!r}"
        )
    if parts.query or parts.fragment:
        raise CompanionConfigError(f"{field} must not contain a query or fragment: {raw!r}")
    return f"{parts.scheme}://{parts.netloc}"


def _swap_port(url: str, port: int) -> str:
    """Same scheme and host, different port. Used to derive the watchdog URL."""
    parts = urlsplit(url)
    host = parts.hostname or "127.0.0.1"
    if ":" in host and not host.startswith("["):  # bare IPv6
        host = f"[{host}]"
    return f"{parts.scheme}://{host}:{port}"


class ServerProfile(BaseModel):
    """One reachable StudioForge server."""

    name: str
    url: str
    api_key: str | None = None
    watchdog_url: str | None = None
    # Model loads legitimately take minutes (a 70B off a cold disk), so the
    # default read timeout is generous; connect timeouts stay short in client.py.
    timeout_s: float = 600.0

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return normalize_url(value, field="url")

    @field_validator("watchdog_url")
    @classmethod
    def _check_watchdog_url(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return normalize_url(str(value), field="watchdog_url")

    @property
    def api_base(self) -> str:
        """Management API root."""
        return f"{self.url}/api"

    @property
    def openai_base(self) -> str:
        """OpenAI-compatible inference root (what a client would be pointed at)."""
        return f"{self.url}/v1"

    @property
    def mcp_url(self) -> str:
        """The main app's management MCP endpoint."""
        return f"{self.url}/mcp"

    @property
    def effective_watchdog_url(self) -> str:
        """Explicit ``watchdog_url``, else the same host on port 1235.

        Deriving it keeps the common case zero-config: the watchdog runs beside
        the server, so only the port differs.
        """
        if self.watchdog_url:
            return self.watchdog_url
        return _swap_port(self.url, DEFAULT_WATCHDOG_PORT)

    @property
    def watchdog_mcp_url(self) -> str:
        """The watchdog's recovery MCP endpoint."""
        return f"{self.effective_watchdog_url}/mcp"

    def auth_headers(self) -> dict[str, str]:
        """Bearer header, or nothing when the server runs open."""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def describe(self) -> dict[str, Any]:
        """Printable view. The key is redacted here, never anywhere else."""
        return {
            "name": self.name,
            "url": self.url,
            "api_key": redact(self.api_key),
            "watchdog_url": self.effective_watchdog_url,
            "watchdog_url_explicit": self.watchdog_url is not None,
            "timeout_s": self.timeout_s,
        }


class CompanionConfig(BaseModel):
    """The whole local config: a default profile name plus the profiles."""

    default: str | None = None
    servers: dict[str, ServerProfile] = Field(default_factory=dict)

    def profile(self, name: str | None = None) -> ServerProfile:
        """Resolve a profile by name, by ``default``, or by being the only one.

        Errors list the known names, because "unknown server 'rig2'" alone
        sends the user hunting for a file they may never have opened.
        """
        known = sorted(self.servers)
        if not self.servers:
            raise CompanionConfigError(
                "no servers configured. Add one with:\n"
                "  sfctl servers add rig http://<host>:1234 --api-key <key>\n"
                f"(config file: {config_path()})"
            )
        if name:
            if name in self.servers:
                return self.servers[name]
            raise CompanionConfigError(
                f"unknown server {name!r}. Known servers: {', '.join(known)}"
            )
        if self.default:
            if self.default in self.servers:
                return self.servers[self.default]
            raise CompanionConfigError(
                f"default server {self.default!r} is not defined. Known servers: {', '.join(known)}"
            )
        if len(self.servers) == 1:
            return next(iter(self.servers.values()))
        raise CompanionConfigError(
            "no default server set and several are defined "
            f"({', '.join(known)}). Pick one with 'sfctl servers use <name>' or pass -s <name>."
        )


def config_dir() -> Path:
    """``%APPDATA%\\studioforge`` on Windows, ``~/.config/studioforge`` elsewhere."""
    override = os.environ.get(ENV_CONFIG_PATH)
    if override:
        return Path(override).expanduser().parent
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / APP_NAME


def config_path() -> Path:
    """Full path of ``companion.toml``."""
    override = os.environ.get(ENV_CONFIG_PATH)
    if override:
        return Path(override).expanduser()
    return config_dir() / CONFIG_FILENAME


def load_companion_config(path: Path | None = None) -> CompanionConfig:
    """Read the config, returning an empty one when the file does not exist.

    A missing file is not an error: a fresh install should be able to run
    ``sfctl servers add`` (and ``sfctl --url ... status``) with no setup.
    """
    target = path or config_path()
    if not target.is_file():
        return CompanionConfig()
    try:
        raw = tomllib.loads(target.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CompanionConfigError(f"{target} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise CompanionConfigError(f"cannot read {target}: {exc}") from exc

    servers: dict[str, ServerProfile] = {}
    table = raw.get("servers")
    if table is not None and not isinstance(table, dict):
        raise CompanionConfigError(f"{target}: [servers] must be a table")
    for name, body in (table or {}).items():
        if not isinstance(body, dict):
            raise CompanionConfigError(f"{target}: [servers.{name}] must be a table")
        servers[str(name)] = _profile_from_table(str(name), body, target)

    # A bare top-level [server] table is accepted as the profile named
    # "default": it is what a hand-written config or an older release looks
    # like, and silently ignoring it would look like the key was wrong.
    legacy = raw.get("server")
    if isinstance(legacy, dict) and legacy.get("url") and IMPLICIT_PROFILE not in servers:
        servers[IMPLICIT_PROFILE] = _profile_from_table(IMPLICIT_PROFILE, legacy, target)

    default = raw.get("default")
    if default is not None and not isinstance(default, str):
        raise CompanionConfigError(f"{target}: 'default' must be a string")
    return CompanionConfig(default=default, servers=servers)


def _profile_from_table(name: str, body: dict[str, Any], source: Path) -> ServerProfile:
    data = {k: v for k, v in body.items() if k != "name"}
    if "url" not in data:
        raise CompanionConfigError(f"{source}: [servers.{name}] is missing 'url'")
    try:
        return ServerProfile(name=name, **data)
    except CompanionConfigError:
        raise
    except Exception as exc:  # pydantic validation
        raise CompanionConfigError(f"{source}: [servers.{name}] is invalid: {exc}") from exc


def _to_table(cfg: CompanionConfig) -> dict[str, Any]:
    servers: dict[str, Any] = {}
    for name, profile in sorted(cfg.servers.items()):
        body: dict[str, Any] = {"url": profile.url}
        if profile.api_key:
            body["api_key"] = profile.api_key
        if profile.watchdog_url:
            body["watchdog_url"] = profile.watchdog_url
        if profile.timeout_s != ServerProfile.model_fields["timeout_s"].default:
            body["timeout_s"] = profile.timeout_s
        servers[name] = body
    out: dict[str, Any] = {}
    if cfg.default:
        out["default"] = cfg.default
    out["servers"] = servers
    return out


def save_companion_config(cfg: CompanionConfig, path: Path | None = None) -> Path:
    """Persist atomically with restrictive permissions.

    Atomic because a half-written config would lock the user out of their own
    rig; ``0600`` because the file contains an API key and the default umask on
    a shared box does not.
    """
    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = tomli_w.dumps(_to_table(cfg))

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            tmp.chmod(0o600)
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if os.name != "nt":
        target.chmod(0o600)
    return target


#: Fields settable through :func:`set_value`, with their coercions.
_SCALAR_FIELDS = ("url", "api_key", "watchdog_url", "timeout_s")


def set_value(dotted: str, value: str, *, path: Path | None = None) -> None:
    """Set one dotted key in the local config and save.

    Accepted forms::

        default                     -> which profile is used when none is named
        server.url                  -> shorthand for the default/only profile
        server.api_key
        servers.<name>.url          -> explicit profile (created if new)
        servers.<name>.api_key
        servers.<name>.watchdog_url
        servers.<name>.timeout_s
    """
    target = path or config_path()
    cfg = load_companion_config(target)
    parts = dotted.split(".")

    if parts == ["default"]:
        if value and value not in cfg.servers:
            raise CompanionConfigError(
                f"cannot default to unknown server {value!r}. "
                f"Known servers: {', '.join(sorted(cfg.servers)) or '(none)'}"
            )
        cfg.default = value or None
        save_companion_config(cfg, target)
        return

    if len(parts) == 2 and parts[0] == "server":
        name = cfg.default or (next(iter(cfg.servers)) if len(cfg.servers) == 1 else None)
        name = name or IMPLICIT_PROFILE
        field = parts[1]
    elif len(parts) == 3 and parts[0] == "servers":
        name, field = parts[1], parts[2]
    else:
        raise CompanionConfigError(
            f"unrecognised key {dotted!r}. Use 'default', 'server.<field>' or "
            f"'servers.<name>.<field>' where field is one of {', '.join(_SCALAR_FIELDS)}"
        )

    if field not in _SCALAR_FIELDS:
        raise CompanionConfigError(
            f"unknown field {field!r}; expected one of {', '.join(_SCALAR_FIELDS)}"
        )

    existing = cfg.servers.get(name)
    data: dict[str, Any] = (
        existing.model_dump() if existing is not None else {"name": name, "url": ""}
    )
    data["name"] = name

    if field == "timeout_s":
        try:
            data[field] = float(value)
        except ValueError as exc:
            raise CompanionConfigError(f"timeout_s must be a number, got {value!r}") from exc
    elif field in ("api_key", "watchdog_url"):
        data[field] = value or None
    else:
        data[field] = value

    if not data.get("url"):
        raise CompanionConfigError(
            f"server {name!r} has no url yet; set 'servers.{name}.url' first "
            f"(or use 'sfctl servers add {name} http://<host>:1234')"
        )
    try:
        cfg.servers[name] = ServerProfile(**data)
    except CompanionConfigError:
        raise
    except Exception as exc:
        raise CompanionConfigError(f"invalid value for {dotted}: {exc}") from exc

    if cfg.default is None:
        cfg.default = name
    save_companion_config(cfg, target)
