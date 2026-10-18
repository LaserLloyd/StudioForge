"""The first-run wizard: from a fresh install to a first answer in eight steps.

Shared by ``studioforge setup`` (interactive, or fully non-interactive with
flags for CI) and the panel's ``/setup`` page. The wizard itself is plain
orchestration over a :class:`WizardServices` object, so it is tested with a fake
and runs against the real server through :class:`LiveServices`.

1. **Owner account** -- username and password (argon2id, secret store).
2. **Detect hardware** -- the GPU probe; the wizard codes against what the probe
   reports, so backends added later (ROCm, Vulkan, Metal) need no change here.
3. **Engine** -- find or install a llama.cpp build and smoke-test it.
4. **Model libraries** -- offer the LM Studio library (``~/.lmstudio/models``, or
   the relocated ``downloadsFolder``) and the configured ``models.dir``.
5. **Starter model** -- load one that is already in the library (the server's
   own ``load-recommended`` decides the context and placement), or download a
   suggested one that fits the smallest GPU. No planner logic lives here.
6. **Test chat** -- one short completion through the loopback API.
7. **Connect** -- the base URL and copy-paste snippets (OpenClaw, SDKs, curl, MCP).
8. **Exposure** -- ``local`` (default), ``lan`` (needs the account; binds the
   network and turns on a self-signed certificate) or ``peer`` (multi-host
   pairing: not built yet -- planned for milestone M3, so it stays local).

Every step reports ``done`` / ``skipped`` / ``failed`` / ``manual`` and the
wizard continues after a failure whenever later steps do not depend on it,
because a first run that stops at the first problem teaches nothing.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from studioforge.core.integrations import build_snippets

__all__ = [
    "STARTER_MODELS",
    "LiveServices",
    "Prompter",
    "SetupWizard",
    "StepResult",
    "WizardOptions",
    "WizardServices",
    "detect_lmstudio_library",
]

#: Suggested starters, by the VRAM of the *smallest* GPU. Repos only: the quant
#: is picked server-side by the planner when the download is queued. These are
#: suggestions the owner can override (``--model``), not an endorsement.
STARTER_MODELS: tuple[dict[str, Any], ...] = (
    {"min_vram_gib": 0.0, "repo": "unsloth/Qwen3-4B-GGUF", "note": "small and quick"},
    {"min_vram_gib": 12.0, "repo": "unsloth/Qwen3-8B-GGUF", "note": "a good all-rounder"},
    {"min_vram_gib": 20.0, "repo": "unsloth/Qwen3-14B-GGUF", "note": "stronger, needs a big card"},
    {"min_vram_gib": 40.0, "repo": "unsloth/Qwen3-32B-GGUF", "note": "needs 48 GB or more"},
)

EXPOSURES = ("local", "lan", "peer")


@dataclass
class WizardOptions:
    """Everything the wizard needs to run without asking a question."""

    username: str | None = None
    password: str | None = None
    skip_account: bool = False
    #: ``lmstudio`` (use the detected LM Studio library), ``auto`` (keep the
    #: configured one, else LM Studio), or a directory.
    library: str | None = None
    engine_tag: str | None = None
    skip_engine: bool = False
    #: A Hugging Face repo (``owner/name``) to download, or an id already in the library.
    model: str | None = None
    skip_model: bool = False
    skip_chat: bool = False
    exposure: str = "local"
    interactive: bool = False
    start_server: bool = True


@dataclass
class StepResult:
    n: int
    key: str
    title: str
    status: str  # done | skipped | failed | manual
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.n,
            "key": self.key,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "data": self.data,
        }


class Prompter(Protocol):
    def ask(self, question: str, default: str | None = None, *, secret: bool = False) -> str: ...

    def confirm(self, question: str, default: bool = True) -> bool: ...

    def choose(self, question: str, options: list[str], default: int = 0) -> int: ...


class WizardServices(Protocol):
    """What the wizard needs from the machine. Real: :class:`LiveServices`. Tests: a fake."""

    def has_owner(self) -> bool: ...

    def create_owner(self, username: str, password: str) -> None: ...

    def detect_hardware(self) -> dict[str, Any]: ...

    def engine_state(self) -> dict[str, Any]: ...

    def ensure_engine(self, tag: str | None) -> dict[str, Any]: ...

    def find_libraries(self) -> list[dict[str, Any]]: ...

    def set_library(self, path: str) -> None: ...

    def ensure_server(self) -> str: ...

    def library_models(self) -> list[dict[str, Any]]: ...

    def download_model(self, repo: str) -> dict[str, Any]: ...

    def load_model(self, model_id: str) -> dict[str, Any]: ...

    def test_chat(self, model_id: str) -> dict[str, Any]: ...

    def base_url(self) -> str: ...

    def create_admin_key(self, name: str) -> str | None: ...

    def apply_exposure(self, mode: str) -> dict[str, Any]: ...


def detect_lmstudio_library() -> list[Path]:
    """Existing LM Studio model folders (the relocated ``downloadsFolder`` first)."""
    from studioforge.config import lmstudio_model_dir_candidates

    return [p for p in lmstudio_model_dir_candidates() if p.is_dir()]


def pick_starter(smallest_gpu_gib: float) -> dict[str, Any]:
    """The largest suggested starter whose ``min_vram_gib`` the smallest GPU meets."""
    chosen = STARTER_MODELS[0]
    for entry in STARTER_MODELS:
        if smallest_gpu_gib >= float(entry["min_vram_gib"]):
            chosen = entry
    return dict(chosen)


class SetupWizard:
    """Runs the eight steps. ``run()`` never raises for a step failure."""

    def __init__(
        self,
        services: WizardServices,
        options: WizardOptions,
        *,
        prompter: Prompter | None = None,
        say: Callable[[str], None] | None = None,
    ) -> None:
        self.services = services
        self.options = options
        self.prompter = prompter if options.interactive else None
        self._say = say or (lambda _msg: None)
        self.results: list[StepResult] = []
        self._model_id: str | None = None
        self._smallest_gpu_gib = 0.0
        self._owner_ready = False

    # -- plumbing ----------------------------------------------------------

    def _record(
        self, n: int, key: str, title: str, status: str, detail: str = "", **data: Any
    ) -> StepResult:
        result = StepResult(n, key, title, status, detail, data)
        self.results.append(result)
        self._say(f"[{n}/8] {title}: {status}" + (f" - {detail}" if detail else ""))
        return result

    def _guard(self, n: int, key: str, title: str, work: Callable[[], StepResult]) -> StepResult:
        try:
            return work()
        except Exception as exc:  # noqa: BLE001 - a step failing must not end the wizard
            return self._record(n, key, title, "failed", str(exc))

    def run(self) -> list[StepResult]:
        self.results = []
        self._guard(1, "account", "Owner account", self.step_account)
        self._guard(2, "hardware", "Detect hardware", self.step_hardware)
        self._guard(3, "engine", "Engine", self.step_engine)
        self._guard(4, "library", "Model library", self.step_library)
        self._guard(5, "model", "Starter model", self.step_model)
        self._guard(6, "chat", "Test chat", self.step_chat)
        self._guard(7, "connect", "Connect your tools", self.step_connect)
        self._guard(8, "exposure", "Network exposure", self.step_exposure)
        return self.results

    # -- 1 -----------------------------------------------------------------

    def step_account(self) -> StepResult:
        title = "Owner account"
        opts = self.options
        if self.services.has_owner():
            self._owner_ready = True
            return self._record(1, "account", title, "skipped", "an owner account already exists")
        if opts.skip_account:
            return self._record(
                1,
                "account",
                title,
                "skipped",
                "skipped by request (no account is needed for local-only use)",
            )
        username, password = opts.username, opts.password
        if self.prompter is not None:
            username = username or self.prompter.ask("Owner username", "owner")
            if not password:
                password = self.prompter.ask("Owner password (10+ characters)", secret=True)
        if not username or not password:
            if opts.exposure == "lan":
                raise ValueError(
                    "--exposure lan needs an owner account: pass --username and a password"
                )
            return self._record(
                1,
                "account",
                title,
                "skipped",
                "no username/password given; local-only use needs none",
            )
        self.services.create_owner(username, password)
        self._owner_ready = True
        return self._record(1, "account", title, "done", f"created '{username}'", username=username)

    # -- 2 -----------------------------------------------------------------

    def step_hardware(self) -> StepResult:
        info = self.services.detect_hardware()
        gpus = info.get("gpus") or []
        if gpus:
            self._smallest_gpu_gib = min(float(g.get("total_bytes", 0)) for g in gpus) / 1024**3
        names = ", ".join(str(g.get("name")) for g in gpus)
        if not gpus:
            return self._record(
                2,
                "hardware",
                "Detect hardware",
                "failed",
                "no GPU was detected; StudioForge serves models from GPU memory only. Install the "
                "GPU driver and run `studioforge doctor`.",
                **info,
            )
        return self._record(
            2,
            "hardware",
            "Detect hardware",
            "done",
            f"{len(gpus)} GPU(s) via {info.get('backend')}: {names}",
            **info,
        )

    # -- 3 -----------------------------------------------------------------

    def step_engine(self) -> StepResult:
        title = "Engine"
        if self.options.skip_engine:
            return self._record(3, "engine", title, "skipped", "skipped by request")
        state = self.services.engine_state()
        if state.get("active") and state.get("smoke_tested") and not self.options.engine_tag:
            return self._record(
                3,
                "engine",
                title,
                "done",
                f"engine {state['active']} is installed and passed its smoke test",
                **state,
            )
        build = state.get("pinned_tag") or "default build"
        if (
            self.prompter is not None
            and not self.options.engine_tag
            and not self.prompter.confirm(
                f"Download and smoke-test the llama.cpp engine ({build})?"
            )
        ):
            return self._record(3, "engine", title, "skipped", "declined")
        result = self.services.ensure_engine(self.options.engine_tag)
        if result.get("smoke_tested"):
            return self._record(
                3,
                "engine",
                title,
                "done",
                f"engine {result.get('tag')} passed its smoke test",
                **result,
            )
        return self._record(
            3,
            "engine",
            title,
            "failed",
            str(result.get("message") or "the engine did not pass its smoke test"),
            **result,
        )

    # -- 4 -----------------------------------------------------------------

    def step_library(self) -> StepResult:
        title = "Model library"
        libraries = self.services.find_libraries()
        wanted = (self.options.library or "").strip()
        chosen: dict[str, Any] | None = None
        if wanted and wanted not in {"auto", "lmstudio"}:
            path = Path(wanted).expanduser()
            if not path.is_dir():
                raise ValueError(f"the model library {path} does not exist or is not a folder")
            self.services.set_library(str(path))
            return self._record(4, "library", title, "done", f"using {path}", path=str(path))
        configured = next((lib for lib in libraries if lib.get("source") == "configured"), None)
        lmstudio = next((lib for lib in libraries if lib.get("source") == "lmstudio"), None)
        if wanted == "lmstudio":
            lmstudio = next(
                (
                    lib
                    for lib in libraries
                    if lib.get("is_lmstudio") or lib.get("source") == "lmstudio"
                ),
                None,
            )
            if lmstudio is None:
                raise ValueError(
                    "no LM Studio model library was found (looked in ~/.lmstudio/models)"
                )
            chosen = lmstudio
        elif configured is not None:
            chosen = configured
        elif lmstudio is not None:
            if self.prompter is None or self.prompter.confirm(
                f"Use your LM Studio library at {lmstudio['path']} "
                f"({lmstudio.get('models', '?')} models)? "
                "Nothing is copied or moved."
            ):
                chosen = lmstudio
        if chosen is None:
            return self._record(
                4,
                "library",
                title,
                "manual",
                "no library found; downloads go to the default folder. Set models.dir to "
                "use a folder.",
            )
        if chosen.get("source") != "configured":
            self.services.set_library(str(chosen["path"]))
        return self._record(
            4,
            "library",
            title,
            "done",
            f"using {chosen['path']}",
            path=str(chosen["path"]),
            source=chosen.get("source"),
        )

    # -- 5 -----------------------------------------------------------------

    def step_model(self) -> StepResult:
        title = "Starter model"
        opts = self.options
        if opts.skip_model:
            return self._record(5, "model", title, "skipped", "skipped by request")
        if opts.start_server:
            self.services.ensure_server()
        models = self.services.library_models()
        wanted = (opts.model or "").strip()
        if wanted and wanted in {str(m.get("id")) for m in models}:
            model_id = wanted
        elif wanted:
            fetched = self.services.download_model(wanted)
            model_id = str(fetched["model_id"])
        elif models:
            model_id = str(sorted(models, key=lambda m: int(m.get("size_bytes") or 0))[0]["id"])
            if self.prompter is not None:
                options = [str(m["id"]) for m in models]
                model_id = options[
                    self.prompter.choose("Load which model?", options, options.index(model_id))
                ]
        else:
            suggestion = pick_starter(self._smallest_gpu_gib)
            if self.prompter is not None and not self.prompter.confirm(
                f"Download {suggestion['repo']} ({suggestion['note']})?"
            ):
                return self._record(
                    5, "model", title, "skipped", "declined; use `studioforge get <repo>` later"
                )
            fetched = self.services.download_model(str(suggestion["repo"]))
            model_id = str(fetched["model_id"])
        loaded = self.services.load_model(model_id)
        self._model_id = model_id
        return self._record(
            5, "model", title, "done", f"loaded {model_id}", model_id=model_id, **loaded
        )

    # -- 6 -----------------------------------------------------------------

    def step_chat(self) -> StepResult:
        title = "Test chat"
        if self.options.skip_chat or self._model_id is None:
            reason = "skipped by request" if self.options.skip_chat else "no model is loaded"
            return self._record(6, "chat", title, "skipped", reason)
        started = time.monotonic()
        reply = self.services.test_chat(self._model_id)
        if not reply.get("ok", True):
            return self._record(
                6,
                "chat",
                title,
                "failed",
                str(reply.get("error") or "the test chat failed"),
                **reply,
            )
        tps = reply.get("tokens_per_second")
        detail = f"answered in {time.monotonic() - started:.1f}s" + (
            f" at {tps:.0f} tok/s" if tps else ""
        )
        return self._record(6, "chat", title, "done", detail, **reply)

    # -- 7 -----------------------------------------------------------------

    def step_connect(self) -> StepResult:
        base = self.services.base_url()
        snippets = build_snippets(base, key=None, model=self._model_id or "<model-id>")
        return self._record(
            7, "connect", "Connect your tools", "done", f"base URL {base}/v1", snippets=snippets
        )

    # -- 8 -----------------------------------------------------------------

    def step_exposure(self) -> StepResult:
        title = "Network exposure"
        mode = self.options.exposure
        if mode not in EXPOSURES:
            raise ValueError(f"exposure must be one of {', '.join(EXPOSURES)}")
        if self.prompter is not None and mode == "local" and self.options.exposure == "local":
            choice = self.prompter.choose(
                "Who can reach this server?",
                [
                    "Only this computer (recommended)",
                    "My local network, over HTTPS (needs the owner account)",
                    "Other StudioForge machines (peers - not available yet)",
                ],
                0,
            )
            mode = EXPOSURES[choice]
        if mode == "peer":
            return self._record(
                8,
                "exposure",
                title,
                "manual",
                "peer mode (several StudioForge hosts sharing models) is planned for milestone M3 "
                "and "
                "is not available yet; this install stays local-only.",
            )
        if mode == "lan" and not (self._owner_ready or self.services.has_owner()):
            raise ValueError(
                "network exposure needs an owner account, and none exists: run `studioforge setup` "
                "with --username, or `studioforge users create`"
            )
        applied = self.services.apply_exposure(mode)
        key = None
        if mode == "lan":
            key = self.services.create_admin_key("setup-admin")
        detail = (
            "local-only (127.0.0.1)"
            if mode == "local"
            else "the network can reach this server over HTTPS with the new key; restart to apply"
        )
        return self._record(
            8, "exposure", title, "done", detail, mode=mode, admin_key=key, **applied
        )

    # -- summary -----------------------------------------------------------

    @property
    def ok(self) -> bool:
        return all(r.status != "failed" for r in self.results)

    def summary(self) -> dict[str, Any]:
        return {"ok": self.ok, "steps": [r.to_dict() for r in self.results]}


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


class LiveServices:
    """:class:`WizardServices` over the real install.

    Direct for the account and the config file, HTTP for everything the running server owns.
    """

    def __init__(self, config: Any, *, say: Callable[[str], None] | None = None) -> None:
        self.config = config
        self._say = say or (lambda _msg: None)
        self._client: Any = None

    # -- helpers -----------------------------------------------------------

    def _store(self) -> tuple[Any, Any]:
        from studioforge.core.accounts import AccountStore
        from studioforge.db import Database

        db = Database(self.config.db_path)
        db.migrate_with_recovery()
        return db, AccountStore(db, self.config.data_dir)

    @property
    def client(self) -> Any:
        from studioforge.cli_client import LocalClient

        if self._client is None:
            self._client = LocalClient(self.config, timeout_s=120.0)
        return self._client

    def _save_config(self, updates: dict[str, Any]) -> None:
        from studioforge.config import apply_overrides

        updated = apply_overrides(self.config, updates)
        updated.save()
        self.config = updated
        if self._client is not None:
            self._client.config = updated

    # -- account -----------------------------------------------------------

    def has_owner(self) -> bool:
        db, store = self._store()
        try:
            return bool(store.has_owner())
        finally:
            db.close()

    def create_owner(self, username: str, password: str) -> None:
        db, store = self._store()
        try:
            store.create_account(username, password)
        finally:
            db.close()

    # -- hardware / engine -------------------------------------------------

    def detect_hardware(self) -> dict[str, Any]:
        from studioforge.core.gpu import get_probe

        probe = get_probe()
        return {
            "backend": probe.backend,
            "gpus": [g.model_dump(mode="json") for g in probe.list_gpus()],
        }

    def _engine_manager(self) -> Any:
        from studioforge.core.engine import EngineManager
        from studioforge.core.gpu import get_probe

        return EngineManager(self.config, probe=get_probe())

    def engine_state(self) -> dict[str, Any]:
        manager = self._engine_manager()
        active = manager.active()
        return {
            "active": active.tag if active else None,
            "smoke_tested": bool(active and active.smoke_tested),
            "pinned_tag": self.config.engine.pinned_tag,
        }

    def ensure_engine(self, tag: str | None) -> dict[str, Any]:
        import asyncio

        manager = self._engine_manager()
        wanted = tag or self.config.engine.pinned_tag

        async def work() -> dict[str, Any]:
            try:
                info = await manager.install(wanted, force=False, activate=False)
                ok, message = await manager.smoke_test(info.tag)
                if ok:
                    await manager.activate(info.tag)
                return {"tag": info.tag, "smoke_tested": bool(ok), "message": message}
            finally:
                with contextlib.suppress(Exception):
                    await manager.aclose()

        return asyncio.run(work())

    # -- library -----------------------------------------------------------

    def find_libraries(self) -> list[dict[str, Any]]:
        lmstudio = {str(p.resolve()): p for p in detect_lmstudio_library()}
        found: list[dict[str, Any]] = []
        configured_key = None
        if self.config.models.dir:
            configured = Path(self.config.models.dir)
            configured_key = str(configured.resolve())
            found.append(
                {
                    "path": str(configured),
                    "source": "configured",
                    "models": _count_ggufs(configured),
                    # models.dir is often the LM Studio folder itself (first run adopts it)
                    "is_lmstudio": configured_key in lmstudio,
                }
            )
        for key, path in lmstudio.items():
            if key != configured_key:
                found.append(
                    {
                        "path": str(path),
                        "source": "lmstudio",
                        "models": _count_ggufs(path),
                        "is_lmstudio": True,
                    }
                )
        return found

    def set_library(self, path: str) -> None:
        self._save_config({"models.dir": path})

    # -- server-backed steps ----------------------------------------------

    def ensure_server(self) -> str:
        from studioforge.server_control import server_status, start_server

        status = server_status(self.config)
        if status.running:
            return status.url
        self._say("starting the server...")
        return start_server(self.config).url

    def library_models(self) -> list[dict[str, Any]]:
        payload = self.client.get("/api/models")
        rows = payload.get("models", []) if isinstance(payload, dict) else payload
        return [m for m in rows if not m.get("is_virtual")]

    def download_model(self, repo: str) -> dict[str, Any]:
        started = self.client.post("/api/downloads", {"repo_id": repo})
        group = started["group_id"]
        deadline = time.monotonic() + 60 * 60 * 3
        while time.monotonic() < deadline:
            rows = self.client.get("/api/downloads").get("downloads", [])
            mine = [r for r in rows if r.get("group_id") == group]
            statuses = {str(r.get("status")) for r in mine}
            if mine and statuses <= {"completed"}:
                break
            if statuses & {"failed", "canceled"}:
                raise RuntimeError(f"the download of {repo} {', '.join(sorted(statuses))}")
            done = sum(int(r.get("downloaded_bytes") or 0) for r in mine)
            total = sum(int(r.get("total_bytes") or 0) for r in mine) or 1
            self._say(f"downloading {repo}: {100 * done // total}%")
            time.sleep(3)
        else:
            raise RuntimeError(f"the download of {repo} did not finish in time")
        self.client.post("/api/models/scan", {})
        models = self.library_models()
        stem = repo.split("/")[-1].lower().removesuffix("-gguf")
        for model in models:
            if str(model.get("repo") or "").lower() == repo.lower():
                return {"model_id": model["id"]}
        for model in models:
            if stem and stem in str(model.get("id", "")).lower():
                return {"model_id": model["id"]}
        raise RuntimeError(f"{repo} downloaded but no matching model appeared in the library")

    def load_model(self, model_id: str) -> dict[str, Any]:
        ctx = int(self.config.models.default_ctx)
        result = self.client.post(f"/api/models/{model_id}/load-recommended", {"ctx_size": ctx})
        return {"ctx_size": result.get("ctx_size") or ctx}

    def test_chat(self, model_id: str) -> dict[str, Any]:
        result = self.client.post(f"/api/models/{model_id}/test", {"keep_loaded": True})
        return {
            "ok": bool(result.get("ok", True)),
            "reply": result.get("text"),
            "tokens_per_second": result.get("tokens_per_second"),
            "error": None if result.get("ok", True) else "the model answered with no text",
        }

    def base_url(self) -> str:
        from studioforge.cli_client import base_url_for

        return base_url_for(self.config)

    # -- exposure ----------------------------------------------------------

    def create_admin_key(self, name: str) -> str | None:
        db, store = self._store()
        try:
            _record, token = store.create_key(name, "admin", created_by="setup")
            store.local_cli_key()
            return str(token)
        finally:
            db.close()

    def apply_exposure(self, mode: str) -> dict[str, Any]:
        if mode == "local":
            self._save_config(
                {"server.host": "127.0.0.1", "gui.host": "127.0.0.1", "watchdog.host": "127.0.0.1"}
            )
            return {"hosts": "127.0.0.1", "tls": self.config.server.tls.mode}
        self._save_config(
            {
                "server.host": "0.0.0.0",
                "gui.host": "0.0.0.0",
                "watchdog.host": "0.0.0.0",
                "server.tls.mode": "self_signed",
            }
        )
        return {"hosts": "0.0.0.0", "tls": "self_signed", "restart_required": True}


def _count_ggufs(path: Path) -> int:
    try:
        return sum(1 for _ in path.rglob("*.gguf"))
    except OSError:
        return 0
