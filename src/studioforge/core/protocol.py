"""Deep-link handling for HuggingFace's "Use this model" button.

HuggingFace model pages carry a button that emits a URL for the local app to
handle. The exact shape, read off a live model page rather than guessed::

    lmstudio://open_from_hf?model=lmstudio-community/Qwen2.5-1.5B-Instruct-GGUF

LM Studio claims that scheme by registering ``HKCU\\Software\\Classes\\lmstudio``
with ``URL Protocol`` and an open command. StudioForge registers its own
``studioforge://`` scheme unconditionally, and can *optionally* take over
``lmstudio://`` so the HuggingFace button opens StudioForge's quant picker
instead.

The takeover is opt-in and reversible on purpose: the user still has LM Studio
installed and sharing the same model library, so silently hijacking its scheme
would break a working app they did not ask us to touch. The previous command is
backed up before replacement and restored on ``unregister``.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from studioforge.config import Config
from studioforge.errors import BadRequestError
from studioforge.logging import get_logger

log = get_logger(__name__)

OWN_SCHEME = "studioforge"
LMSTUDIO_SCHEME = "lmstudio"
BACKUP_FILENAME = "protocol-backup.json"


@dataclass(frozen=True)
class DeepLink:
    """A parsed deep link."""

    action: str
    repo_id: str | None = None
    quant: str | None = None
    model_id: str | None = None

    @property
    def is_download(self) -> bool:
        return self.action in {"open_from_hf", "download", "models"} and bool(self.repo_id)


def parse_deep_link(url: str) -> DeepLink:
    """Parse ``lmstudio://`` / ``studioforge://`` URLs into an action.

    Accepts the HuggingFace form (``open_from_hf?model=owner/repo``) plus a few
    natural variants, because the button's format is not a contract we control
    and a near-miss should still do the obvious thing rather than error.
    """
    if not url or "://" not in url:
        raise BadRequestError(f"not a deep link: {url!r}")

    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in {OWN_SCHEME, LMSTUDIO_SCHEME}:
        raise BadRequestError(
            f"unsupported scheme '{scheme}'; expected {OWN_SCHEME}:// or {LMSTUDIO_SCHEME}://"
        )

    # urlparse puts the first path segment in .netloc for custom schemes, so
    # "lmstudio://open_from_hf?model=x" gives netloc="open_from_hf".
    action = (parsed.netloc or parsed.path.lstrip("/")).strip("/").lower()
    query = parse_qs(parsed.query)

    def first(*names: str) -> str | None:
        for name in names:
            values = query.get(name)
            if values and values[0].strip():
                return unquote(values[0]).strip()
        return None

    repo_id = first("model", "repo", "repo_id")
    quant = first("quant", "quantization", "file")
    model_id = first("id", "model_id")

    # "studioforge://download/owner/repo" style, with the repo in the path.
    if repo_id is None and action in {"download", "open_from_hf", "models"}:
        path = parsed.path.strip("/")
        if path.count("/") >= 1:
            repo_id = unquote(path)

    if repo_id is not None and repo_id.count("/") != 1:
        raise BadRequestError(
            f"deep link model must be 'owner/repo', got {repo_id!r}", param="model"
        )

    return DeepLink(action=action or "open", repo_id=repo_id, quant=quant, model_id=model_id)


def gui_url_for(link: DeepLink, config: Config) -> str:
    """Where the GUI should open for this link.

    Uses a loopback host deliberately: the handler runs on the same machine that
    just clicked the button, and the configured bind address is often
    ``0.0.0.0``, which is not a browsable address.
    """
    host = config.gui.host
    if host in {"0.0.0.0", "::", ""}:
        host = "127.0.0.1"
    base = f"http://{host}:{config.gui.port}/"
    if link.is_download and link.repo_id:
        from urllib.parse import quote

        url = f"{base}?tab=download&repo={quote(link.repo_id, safe='/')}"
        if link.quant:
            url += f"&quant={quote(link.quant)}"
        return url
    if link.model_id:
        from urllib.parse import quote

        return f"{base}?tab=models&model={quote(link.model_id, safe='/')}"
    return base


def open_in_browser(url: str) -> bool:
    """Open a URL in the user's default browser."""
    import webbrowser

    try:
        return bool(webbrowser.open(url))
    except Exception as exc:  # pragma: no cover - platform dependent
        log.error("could not open a browser", error=str(exc), url=url)
        return False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def handler_command() -> list[str]:
    """The command line the OS should run for a deep link.

    Prefers the installed ``studioforge`` console script; falls back to
    ``<python> -m studioforge`` so a source checkout works too.
    """
    script = shutil.which("studioforge")
    if script:
        return [script, "open"]
    return [sys.executable, "-m", "studioforge", "open"]


def _backup_path(config: Config) -> Path:
    return config.data_dir / BACKUP_FILENAME


def register(config: Config, *, takeover_lmstudio: bool = False) -> dict[str, object]:
    """Register the URL scheme(s). Returns a summary of what changed."""
    if os.name == "nt":
        return _register_windows(config, takeover_lmstudio=takeover_lmstudio)
    return _register_linux(config, takeover_lmstudio=takeover_lmstudio)


def unregister(config: Config) -> dict[str, object]:
    """Remove our scheme and restore any hijacked ``lmstudio://`` command."""
    if os.name == "nt":
        return _unregister_windows(config)
    return _unregister_linux(config)


def status(config: Config) -> dict[str, object]:
    """What is registered right now, for the GUI's Server tab."""
    if os.name == "nt":
        return {
            "platform": "windows",
            "studioforge": _read_windows_command(OWN_SCHEME),
            "lmstudio": _read_windows_command(LMSTUDIO_SCHEME),
            "backup": _read_backup(config),
            "handler_command": handler_command(),
        }
    desktop = _linux_desktop_path()
    return {
        "platform": "linux",
        "studioforge": desktop.read_text(encoding="utf-8") if desktop.is_file() else None,
        "handler_command": handler_command(),
        "backup": _read_backup(config),
    }


def _read_backup(config: Config) -> dict[str, object] | None:
    path = _backup_path(config)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# --- Windows ---------------------------------------------------------------


def _read_windows_command(scheme: str) -> str | None:
    try:
        import winreg
    except ImportError:  # pragma: no cover - non-Windows
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, rf"Software\Classes\{scheme}\shell\open\command"
        ) as key:
            value, _ = winreg.QueryValueEx(key, "")
            return str(value)
    except OSError:
        return None


def _write_windows_scheme(scheme: str, command: str) -> None:
    import winreg

    base = rf"Software\Classes\{scheme}"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, base) as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, f"URL:{scheme} Protocol")
        # The presence of this empty-valued name is what marks a key as a URL
        # protocol handler; without it Windows ignores the registration.
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, rf"{base}\shell\open\command") as key:
        winreg.SetValueEx(key, "", 0, winreg.REG_SZ, command)


def _delete_windows_scheme(scheme: str) -> None:
    import winreg

    for sub in (
        rf"Software\Classes\{scheme}\shell\open\command",
        rf"Software\Classes\{scheme}\shell\open",
        rf"Software\Classes\{scheme}\shell",
        rf"Software\Classes\{scheme}",
    ):
        with contextlib.suppress(OSError):
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)


def _quoted_command() -> str:
    parts = handler_command()
    quoted = " ".join(f'"{p}"' for p in parts)
    return f'{quoted} "%1"'


def _register_windows(config: Config, *, takeover_lmstudio: bool) -> dict[str, object]:
    command = _quoted_command()
    changed: dict[str, object] = {"scheme": OWN_SCHEME, "command": command}
    _write_windows_scheme(OWN_SCHEME, command)

    if takeover_lmstudio:
        previous = _read_windows_command(LMSTUDIO_SCHEME)
        backup = _read_backup(config) or {}
        # Only record the first takeover, so repeated registration cannot
        # overwrite the genuine LM Studio command with our own. `previous !=
        # command` guards the case where the backup file was lost (data dir
        # moved or wiped) on a machine where the takeover is already applied:
        # without it we would record OUR command as the thing to restore, and
        # "give it back to LM Studio" would hand it back to StudioForge.
        if previous and previous != command and "lmstudio_command" not in backup:
            backup["lmstudio_command"] = previous
            config.data_dir.mkdir(parents=True, exist_ok=True)
            _backup_path(config).write_text(json.dumps(backup, indent=2), encoding="utf-8")
        _write_windows_scheme(LMSTUDIO_SCHEME, command)
        changed["lmstudio_taken_over"] = True
        changed["lmstudio_previous"] = previous
    return changed


def _unregister_windows(config: Config) -> dict[str, object]:
    _delete_windows_scheme(OWN_SCHEME)
    result: dict[str, object] = {"removed": OWN_SCHEME}
    backup = _read_backup(config) or {}
    previous = backup.get("lmstudio_command")
    if isinstance(previous, str) and previous:
        _write_windows_scheme(LMSTUDIO_SCHEME, previous)
        result["lmstudio_restored"] = previous
        backup.pop("lmstudio_command", None)
        _backup_path(config).write_text(json.dumps(backup, indent=2), encoding="utf-8")
    elif _read_windows_command(LMSTUDIO_SCHEME) == _quoted_command():
        # We took the scheme over on a machine that had no previous handler
        # (LM Studio not installed), so there was nothing to back up. Leaving
        # our command behind would point HuggingFace's button at a handler that
        # no longer exists once StudioForge is gone. Only remove it when it is
        # provably still ours.
        _delete_windows_scheme(LMSTUDIO_SCHEME)
        result["lmstudio_removed"] = True
    return result


# --- Linux -----------------------------------------------------------------


def _linux_desktop_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "applications" / "studioforge-url-handler.desktop"


def _register_linux(config: Config, *, takeover_lmstudio: bool) -> dict[str, object]:
    parts = handler_command()
    exec_line = " ".join(parts) + " %u"
    mimes = [f"x-scheme-handler/{OWN_SCHEME}"]
    if takeover_lmstudio:
        mimes.append(f"x-scheme-handler/{LMSTUDIO_SCHEME}")

    desktop = _linux_desktop_path()
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name=StudioForge\n"
        "Comment=Open HuggingFace models in StudioForge\n"
        f"Exec={exec_line}\n"
        "Terminal=false\n"
        "NoDisplay=true\n"
        f"MimeType={';'.join(mimes)};\n",
        encoding="utf-8",
    )
    for tool, args in (
        ("update-desktop-database", [str(desktop.parent)]),
        *(("xdg-mime", ["default", desktop.name, mime]) for mime in mimes),
    ):
        binary = shutil.which(tool)
        if binary:
            with_args = [binary, *args]
            try:
                subprocess.run(with_args, capture_output=True, timeout=30, check=False)
            except (OSError, subprocess.TimeoutExpired):
                log.warning("could not run desktop integration tool", tool=tool)
                continue
    return {"desktop_file": str(desktop), "mimetypes": mimes}


def _unregister_linux(config: Config) -> dict[str, object]:
    desktop = _linux_desktop_path()
    existed = desktop.is_file()
    if existed:
        desktop.unlink()
        tool = shutil.which("update-desktop-database")
        if tool:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    [tool, str(desktop.parent)], capture_output=True, timeout=30, check=False
                )
    return {"removed": str(desktop) if existed else None}
