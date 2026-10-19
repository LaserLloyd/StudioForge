"""Which data directory does the *installed* StudioForge use? (D81)

The server never guesses: the tray, the launchers, the systemd units and the
container all hand it an explicit ``--config`` or ``SF_DATA_DIR``. The CLI used to
guess anyway: with neither given, ``studioforge users passwd`` fell through to
``<checkout>/data`` (a source checkout) or the platform directory, which on a rig
whose data lives elsewhere (``local-env.bat``, ``autostart enable``, a unit file)
is an empty, unrelated install. The command "worked" -- on the wrong database -- and
the owner stayed locked out.

This module answers the question the way the launcher does, for the commands that
manage who may sign in (``users``, ``keys``) and the offline ones (``doctor``,
``migrate``, ``support-bundle``, ``config``). In order:

1. ``--config`` or ``SF_CONFIG``, then ``SF_DATA_DIR``: explicit, used as is.
2. The directory of a **running** instance (``.instance.lock`` names a live pid), among
   the candidates below and the default.
3. The first existing candidate, in this order: the **installed Startup launcher**
   (Windows ``StudioForge.vbs``: its ``--config``), the **systemd unit**
   (``SF_DATA_DIR=`` / ``--config``, user unit then system unit), the checkout's
   ``local-env.bat`` (``SF_DATA_DIR``).
4. **A launcher is adopted only if it is ours**: its interpreter (the shim's ``pythonw``,
   the unit's ``ExecStart``) must live in the Python environment this command runs from
   (``sys.prefix``), and the data directory must be owned by the current user (a
   ``--system`` install owned by ``studioforge`` is not ours to open). Anything else is
   *foreign*: the CLI says where it is and the caller must pass ``--config`` (or run it
   as that install's user) instead of silently acting on, or around, another install.
5. Otherwise :func:`studioforge.config.default_data_dir` (checkout ``data/``, else the
   platform directory), as before.

Read-only: it parses a launcher and looks at a lock file; it never writes, starts or
stops anything. Set ``SF_NO_INSTALL_DISCOVERY=1`` to turn steps 2-3 off (the unit
tests do, so a developer's real install can never be reached by a test).
"""

from __future__ import annotations

import codecs
import contextlib
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from studioforge.config import _checkout_data_dir, default_data_dir
from studioforge.core.autostart import LINUX_UNIT, WINDOWS_SHIM, startup_dir
from studioforge.core.instance_lock import InstanceLock
from studioforge.logging import get_logger

log = get_logger(__name__)

__all__ = ["Foreign", "Located", "describe", "locate"]

#: Set to ``1`` to skip discovery and use the pre-D81 resolution.
DISABLE_ENV = "SF_NO_INSTALL_DISCOVERY"


@dataclass(frozen=True)
class Foreign:
    """An installed StudioForge the CLI found and deliberately did not adopt."""

    config_path: Path
    source: str
    #: Why not, and what to do instead, in one sentence.
    reason: str

    def describe(self) -> str:
        return f"{self.config_path.parent} (from {self.source}): {self.reason}"


@dataclass(frozen=True)
class Located:
    """Where the CLI is acting, and how that was decided."""

    #: Pass to ``load_config`` as ``--config``; ``None`` = let the loader resolve it
    #: (environment, or the checkout / platform default).
    config_path: Path | None
    #: Human words for the source: ``"SF_DATA_DIR"``, ``"the running server (pid 123)"``...
    source: str
    #: Installs that were found but are not this environment's to act on (step 4).
    foreign: tuple[Foreign, ...] = field(default=())


@dataclass(frozen=True)
class Candidate:
    config_path: Path
    source: str
    #: ``None`` = ours; otherwise why it is foreign.
    foreign_reason: str | None = None


# ---------------------------------------------------------------------------
# parsers (pure; unit-testable with strings)
# ---------------------------------------------------------------------------


def _decode(raw: bytes) -> str:
    """BOM-aware: the Windows shim is UTF-16 LE, a unit file is UTF-8."""
    if raw.startswith(codecs.BOM_UTF16_LE):
        return raw[len(codecs.BOM_UTF16_LE) :].decode("utf-16-le", errors="replace")
    if raw.startswith(codecs.BOM_UTF16_BE):
        return raw[len(codecs.BOM_UTF16_BE) :].decode("utf-16-be", errors="replace")
    return raw.decode("utf-8-sig", errors="replace")


_VBS_CONFIG = re.compile(r'--config\s+(?:""(?P<q>.+?)""|(?P<p>[^\s"]+))')
_VBS_CWD = re.compile(r'CurrentDirectory\s*=\s*"(?P<d>[^"\r\n]+)"')


def config_from_shim(text: str) -> Path | None:
    """The ``--config`` path in a Startup shim (``autostart enable`` wrote it), else ``None``."""
    match = _VBS_CONFIG.search(text)
    if match:
        return Path(match.group("q") or match.group("p"))
    # An older shim without --config: it starts in the data directory.
    cwd = _VBS_CWD.search(text)
    if cwd:
        return Path(cwd.group("d")) / "config.yaml"
    return None


_UNIT_ENV = re.compile(r'^\s*Environment\s*=\s*"?SF_DATA_DIR=(?P<v>[^"\r\n]*)"?\s*$', re.M)
_UNIT_EXEC = re.compile(r"^\s*ExecStart\s*=\s*(?P<v>.+?)\s*$", re.M)


def config_from_unit(text: str, *, home: Path | None = None) -> Path | None:
    """The data directory a systemd unit sets (``SF_DATA_DIR=`` or ``--config``), else ``None``."""
    base = home if home is not None else Path.home()

    def expand(value: str) -> Path:
        return Path(value.replace("%h", str(base)))

    for match in _UNIT_EXEC.finditer(text):
        try:
            argv = shlex.split(match.group("v"))
        except ValueError:
            continue
        for index, token in enumerate(argv[:-1]):
            if token == "--config":
                return expand(argv[index + 1])
            if token.startswith("--config="):
                return expand(token.split("=", 1)[1])
    env = _UNIT_ENV.search(text)
    if env:
        value = env.group("v").replace('\\"', '"').replace("\\\\", "\\").strip()
        if value:
            return expand(value) / "config.yaml"
    return None


_VBS_RUN = re.compile(r'\.Run\s+"(?:""(?P<q>.+?)""|(?P<p>[^\s"]+))')


def interpreter_from_shim(text: str) -> Path | None:
    """The program the Startup shim runs (``pythonw.exe``), else ``None``."""
    match = _VBS_RUN.search(text)
    return Path(match.group("q") or match.group("p")) if match else None


def interpreter_from_unit(text: str, *, home: Path | None = None) -> Path | None:
    """The program a systemd unit's ``ExecStart`` runs, else ``None``."""
    base = home if home is not None else Path.home()
    match = _UNIT_EXEC.search(text)
    if not match:
        return None
    try:
        argv = shlex.split(match.group("v"))
    except ValueError:
        return None
    if not argv:
        return None
    return Path(argv[0].lstrip("-@!:+").replace("%h", str(base)))


def _norm(path: Path) -> str:
    return os.path.normcase(str(path.absolute()))


def belongs_to_this_environment(program: Path | None) -> bool:
    """Whether ``program`` is the Python (or console script) of the environment running now."""
    if program is None:
        return False
    here = {_norm(Path(sys.executable).parent), _norm(Path(sys.prefix))}
    parents = {_norm(program.parent)}
    with contextlib.suppress(OSError):
        parents.add(_norm(program.parent.resolve()))
    if parents & here:
        return True
    prefix = _norm(Path(sys.prefix)) + os.sep
    return any(p.startswith(prefix) for p in parents)


def _foreign_owner_reason(data_dir: Path) -> str | None:
    """On POSIX, why a data directory owned by someone else is not ours to open."""
    if sys.platform == "win32":
        return None
    try:
        owner = data_dir.stat().st_uid
    except OSError:
        return None
    me = os.getuid()
    if owner == me or me == 0:
        return None
    try:
        import pwd

        name = pwd.getpwuid(owner).pw_name
    except (ImportError, KeyError):
        name = str(owner)
    return (
        f"owned by the user '{name}', not you; run it as that user, for example "
        f"`sudo -u {name} env SF_DATA_DIR={data_dir} studioforge <command>`"
    )


#: ``set "SF_DATA_DIR=..."`` (whatever follows the closing quote -- ``rem ...``, ``& rem ...`` --
#: is not part of the value) or ``set SF_DATA_DIR=...`` (up to ``&``), each optionally behind
#: ``@`` and/or ``if not defined SF_DATA_DIR``. A ``rem`` or ``::`` line never matches.
_BAT_SET = re.compile(
    r"^[ \t@]*(?:if\s+not\s+defined\s+SF_DATA_DIR\s+)?set\s+"
    r'(?:"SF_DATA_DIR=(?P<q>[^"\r\n]*)"|SF_DATA_DIR=(?P<u>[^&\r\n]*))',
    re.I | re.M,
)
_BAT_VAR = re.compile(r"%([A-Za-z0-9_]+)%")


def data_dir_from_local_env(text: str, env: dict[str, str] | None = None) -> Path | None:
    """``SF_DATA_DIR`` as ``local-env.bat`` sets it (``%VAR%`` expanded), else ``None``.

    The last ``set`` wins, like cmd. A value that still contains an unresolved
    ``%...%`` or a batch parameter (``%~dp0``) is ignored rather than guessed at.
    """
    values = [
        (m.group("q") if m.group("q") is not None else (m.group("u") or "")).strip()
        for m in _BAT_SET.finditer(text)
    ]
    if not values:
        return None
    source = env if env is not None else dict(os.environ)
    lowered = {k.lower(): v for k, v in source.items()}
    value = values[-1].strip()
    value = _BAT_VAR.sub(lambda m: lowered.get(m.group(1).lower(), m.group(0)), value)
    if "%" in value or not value:
        return None
    return Path(value)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def _read(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return _decode(path.read_bytes()[:65536])
    except OSError:
        return None


def _candidates() -> list[Candidate]:
    """Installs named by a launcher, in priority order. Existing places only; no duplicates."""
    found: list[Candidate] = []

    def add(config_path: Path | None, source: str, mine: bool, why_not: str) -> None:
        if config_path is None:
            return
        try:
            config_path = config_path.expanduser()
        except RuntimeError:
            return
        if not config_path.is_file() and not config_path.parent.is_dir():
            return
        resolved = config_path.resolve()
        if any(resolved == existing.config_path.resolve() for existing in found):
            return
        reason = None if mine else why_not
        if mine:
            reason = _foreign_owner_reason(config_path.parent)
        found.append(Candidate(config_path, source, reason))

    wrong_env = (
        "its launcher runs a different Python environment than this command "
        f"({sys.prefix}); run that install's own `studioforge`, or pass --config"
    )
    if os.name == "nt":
        shim = startup_dir() / WINDOWS_SHIM
        text = _read(shim)
        if text:
            add(
                config_from_shim(text),
                f"the installed Startup launcher ({shim})",
                belongs_to_this_environment(interpreter_from_shim(text)),
                wrong_env,
            )
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        user_unit = (
            (Path(xdg) if xdg else Path.home() / ".config") / "systemd" / "user" / LINUX_UNIT
        )
        for unit, label in (
            (user_unit, "the systemd user unit"),
            (Path("/etc/systemd/system") / LINUX_UNIT, "the systemd unit"),
        ):
            text = _read(unit)
            if text:
                add(
                    config_from_unit(text),
                    f"{label} ({unit})",
                    belongs_to_this_environment(interpreter_from_unit(text)),
                    wrong_env,
                )
    checkout = _checkout_data_dir()
    if checkout is not None:
        local_env = checkout.parent / "local-env.bat"
        text = _read(local_env)
        if text:
            data = data_dir_from_local_env(text)
            # local-env.bat sits in THIS checkout's root: it is ours by construction.
            add(data / "config.yaml" if data else None, f"local-env.bat ({local_env})", True, "")
    return found


def locate(explicit_config: Path | str | None) -> Located:
    """Decide which install the CLI acts on. See the module docstring for the order."""
    if explicit_config is not None:
        return Located(Path(explicit_config), "--config")
    if os.environ.get("SF_CONFIG"):
        return Located(None, "SF_CONFIG")
    if os.environ.get("SF_DATA_DIR"):
        return Located(None, "SF_DATA_DIR")
    if os.environ.get(DISABLE_ENV, "").strip() not in {"", "0"}:
        return Located(None, "the default location")
    candidates = _candidates()
    mine = [c for c in candidates if c.foreign_reason is None]
    foreign = tuple(
        Foreign(c.config_path, c.source, c.foreign_reason)
        for c in candidates
        if c.foreign_reason is not None
    )
    default_config = default_data_dir() / "config.yaml"
    pool = [(c.config_path, c.source) for c in mine] + [(default_config, "the default location")]
    for config_path, source in pool:
        holder = InstanceLock(config_path.parent).holder()
        if holder is not None:
            return Located(
                config_path, f"the running server (pid {holder.get('pid')}; {source})", foreign
            )
    if mine:
        return Located(mine[0].config_path, mine[0].source, foreign)
    return Located(None, "the default location", foreign)


def describe(data_dir: Path, db_path: Path, source: str) -> str:
    """The one line every account-managing command prints before it acts."""
    return f"studioforge: acting on {data_dir} (from {source}); database {db_path}"
