#!/usr/bin/env python3
"""A scratch StudioForge panel on alternate ports, for screenshots and manual checks.

    uv run python scripts/panel_demo.py            # panel on http://127.0.0.1:18080
    uv run python scripts/panel_demo.py --port 18081

Nothing here touches a GPU, a real model, the live service or its data directory:

* the GPUs are a fake 2x RTX 5090 + 2x RTX 3090 rig whose "used" memory follows the fake
  loads below;
* the library is seven invented models; the planner, the catalog and ``plan_preview`` are
  the real ones, so the "fits on" column and the fit bars are the product's own arithmetic;
* "loading" a model starts no process: a fake supervisor answers after a short pause, and
  Test Chat talks to a tiny fake OpenAI-compatible child on 127.0.0.1:18199;
* the data directory is a fresh temp folder.

The script refuses to bind any port outside 18080-18199, so it cannot collide with the live
panel (8080), gateway (1234), watchdog (1235) or any sibling service.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHILD_PORT = 18199
ALLOWED = range(18080, 18200)
GIB = 1024**3


def _guard(port: int) -> int:
    if port not in ALLOWED:
        raise SystemExit(f"refusing to bind port {port}: demo ports are 18080-18199 only")
    return port


# ---------------------------------------------------------------------------
# The fake rig
# ---------------------------------------------------------------------------


def build_state(data_dir: Path) -> Any:
    from studioforge.config import Config
    from studioforge.core.manager import ModelManager
    from studioforge.core.planner import Planner
    from studioforge.types import (
        GgufMeta,
        GpuInfo,
        InstanceInfo,
        LoadPlan,
        ModelCapabilities,
        ModelRecord,
        ModelSettings,
        VirtualPreset,
    )

    config = Config(data_dir=data_dir)
    config.models.dir = data_dir / "models"
    config.models.dir.mkdir(parents=True, exist_ok=True)
    config.server.port = 18234
    config.gui.port = 18080
    config.watchdog.port = 18235
    config.ensure_dirs()

    now = time.time()

    def meta(arch: str, layers: int, kv: int, embd: int, train: int, tensor_gib: float, quant: str,
             params: float, experts: int = 0, used: int = 0, vision: bool = False) -> GgufMeta:
        return GgufMeta(
            architecture=arch, n_layer=layers, n_head=32, n_head_kv=kv, n_embd=embd,
            n_embd_head_k=128, n_embd_head_v=128, n_vocab=152064, n_ctx_train=train,
            param_count=int(params * 1e9), tensor_bytes=int(tensor_gib * GIB), quant_label=quant,
            n_expert=experts, n_expert_used=used, has_vision_tensors=vision,
            chat_template="{{ messages }}",
        )

    def rec(model_id: str, name: str, gib: float, quant: str, arch: str, m: GgufMeta,
            days_ago: float, **caps: bool) -> ModelRecord:
        stamp = now - days_ago * 86400
        return ModelRecord(
            id=model_id, name=name, path=data_dir / "models" / f"{model_id}.gguf",
            size_bytes=int(gib * GIB), quant=quant, architecture=arch, meta=m,
            mtime=stamp, added_at=stamp,
            capabilities=ModelCapabilities(**caps), settings=ModelSettings(),
        )

    records = [
        rec("Qwen3-32B-Q4_K_M", "Qwen3 32B", 19.8, "Q4_K_M", "qwen3",
            meta("qwen3", 64, 8, 5120, 40960, 19.8, "Q4_K_M", 32), 1.2, tools=True, thinking=True),
        rec("Llama-3.3-70B-Instruct-Q4_K_M", "Llama 3.3 70B Instruct", 42.5, "Q4_K_M", "llama",
            meta("llama", 80, 8, 8192, 131072, 42.5, "Q4_K_M", 70), 6, tools=True),
        rec("Qwen3-Coder-122B-A10B-Q5_K_M", "Qwen3 Coder 122B A10B", 84.0, "Q5_K_M", "qwen3moe",
            meta("qwen3moe", 48, 2, 3072, 262144, 84.0, "Q5_K_M", 122, 256, 8), 12, tools=True),
        rec("Gemma-3-12B-it-Q8_0", "Gemma 3 12B it", 12.5, "Q8_0", "gemma3",
            meta("gemma3", 48, 8, 3840, 131072, 12.5, "Q8_0", 12, vision=True), 20,
            vision=True),
        rec("Mistral-Small-3.2-24B-Q6_K", "Mistral Small 3.2 24B", 19.3, "Q6_K", "mistral3",
            meta("mistral3", 40, 8, 5120, 131072, 19.3, "Q6_K", 24), 31, tools=True),
        rec("Llama-3.1-8B-Instruct-Q4_K_M", "Llama 3.1 8B Instruct", 4.9, "Q4_K_M", "llama",
            meta("llama", 32, 8, 4096, 131072, 4.9, "Q4_K_M", 8), 45, tools=True),
        rec("Qwen3-Embedding-8B-Q8_0", "Qwen3 Embedding 8B", 8.1, "Q8_0", "qwen3",
            meta("qwen3", 36, 8, 4096, 32768, 8.1, "Q8_0", 8), 60, embedding=True),
    ]
    persona = ModelRecord(
        id="terse-assistant", name="terse-assistant", path=records[0].path,
        size_bytes=records[0].size_bytes, quant=records[0].quant,
        architecture=records[0].architecture, meta=records[0].meta, mtime=now, added_at=now,
        kind="chat", is_virtual=True, base_model_id=records[0].id,
        preset=VirtualPreset(system_prompt="Answer in two sentences or fewer.", temperature=0.3),
        settings=ModelSettings(),
    )
    records.append(persona)

    class Registry:
        def __init__(self) -> None:
            self._records = list(records)

        def all(self) -> list[Any]:
            return list(self._records)

        def get(self, model_id: str) -> Any:
            return next((r for r in self._records if r.id == model_id), None)

        def resolve(self, name: str) -> Any:
            hit = self.get(name)
            if hit is not None:
                return hit
            return next((r for r in self._records if name.lower() == r.name.lower()), None)

        def known_ids(self) -> list[str]:
            return [r.id for r in self._records]

        def adapters(self) -> list[Any]:
            return []

        def get_adapter(self, adapter_id: str) -> None:
            return None

        def touch(self, model_id: str) -> None:
            return None

        def scan(self, *, force: bool = False) -> Any:
            from types import SimpleNamespace

            return SimpleNamespace(added=[], removed=[], unchanged=len(self._records), errors=[])

        def save_settings(self, model_id: str, settings: Any) -> Any:
            updated = self.get(model_id).model_copy(update={"settings": settings})
            self._records = [updated if r.id == model_id else r for r in self._records]
            return updated

        def _alias_candidates(self, record: Any) -> list[str]:
            stem = record.id.lower()
            return [stem, record.name.lower().replace(" ", "-")]

    registry = Registry()

    class Probe:
        backend = "demo"

        def __init__(self) -> None:
            self.supervisor: Any = None

        def _used(self, index: int) -> int:
            base = int(0.5 * GIB)
            if self.supervisor is None:
                return base
            for inst in self.supervisor.list():
                if inst.plan is not None:
                    base += int(inst.plan.per_gpu_bytes.get(index, 0))
            return base

        def available(self) -> bool:
            return True

        def list_gpus(self) -> list[Any]:
            spec = [
                (0, "NVIDIA GeForce RTX 5090", 31.84, (12, 0)),
                (1, "NVIDIA GeForce RTX 5090", 31.84, (12, 0)),
                (2, "NVIDIA GeForce RTX 3090", 24.0, (8, 6)),
                (3, "NVIDIA GeForce RTX 3090", 24.0, (8, 6)),
            ]
            out = []
            for index, name, total_gib, cc in spec:
                total = int(total_gib * GIB)
                used = min(total, self._used(index))
                out.append(
                    GpuInfo(
                        index=index, name=name, total_bytes=total, used_bytes=used,
                        free_bytes=total - used, utilization_pct=4.0 + 3 * index,
                        temperature_c=38.0 + 2 * index, compute_capability=cc,
                    )
                )
            return out

        def get_gpu(self, index: int) -> Any:
            return next((g for g in self.list_gpus() if g.index == index), None)

        def compute_processes(self) -> list[Any]:
            return []

        def driver_version(self) -> str | None:
            return "demo"

        def cuda_driver_version(self) -> tuple[int, int] | None:
            return (13, 0)

        def shutdown(self) -> None:
            return None

    probe = Probe()

    class Supervisor:
        def __init__(self) -> None:
            self.instances: dict[str, Any] = {}
            self.busy: dict[str, int] = {}

        def architecture_table(self, tag: str | None) -> None:
            return None

        def resolved_engine_tag(self, tag: str | None) -> str | None:
            return tag or "b11370"

        async def start(self, record: Any, plan: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(2.5)  # long enough to see the row's loading state
            info = InstanceInfo(
                model_id=record.id, state="ready", port=CHILD_PORT, pid=4242, plan=plan,
                priority=kwargs.get("priority", 3), loaded_by=kwargs.get("source"),
                started_at=time.time(), last_activity_at=time.time(),
                resolved_engine_tag="b11370", last_tokens_per_second=61.4,
            )
            self.instances[record.id] = info
            return info

        async def stop(self, model_id: str, **kwargs: Any) -> None:
            self.instances.pop(model_id, None)

        def get(self, model_id: str) -> Any:
            return self.instances.get(model_id)

        def list(self) -> list[Any]:
            return list(self.instances.values())

        def base_url(self, model_id: str) -> str | None:
            return f"http://127.0.0.1:{CHILD_PORT}" if model_id in self.instances else None

        def mark_request_start(self, model_id: str, **kwargs: Any) -> str:
            inst = self.instances.get(model_id)
            if inst is not None:
                inst.active_requests += 1
            return "req"

        def mark_request_end(self, model_id: str, **kwargs: Any) -> None:
            inst = self.instances.get(model_id)
            if inst is not None:
                inst.active_requests = max(0, inst.active_requests - 1)
                inst.total_requests += 1
                rate = kwargs.get("tokens_per_second")
                if rate:
                    inst.last_tokens_per_second = rate

    supervisor = Supervisor()
    probe.supervisor = supervisor

    planner = Planner(config, probe)
    manager = ModelManager(
        config, registry=registry, planner=planner, supervisor=supervisor, db=None  # type: ignore[arg-type]
    )

    # A resident model, so Status and the Loaded chip have something to show.
    seed = records[5]
    seed_plan = LoadPlan(
        model_id=seed.id, devices=[2], ctx_size=32768, parallel=2, ctx_per_slot=16384,
        kv_cache_type="q8_0", per_gpu_bytes={2: int(seed.size_bytes + 3 * GIB)},
    )
    supervisor.instances[seed.id] = InstanceInfo(
        model_id=seed.id, state="ready", port=CHILD_PORT, pid=4141, plan=seed_plan, priority=1,
        started_at=now - 5400, last_activity_at=now - 20, resolved_engine_tag="b11370",
        last_tokens_per_second=84.2, total_requests=37,
    )

    # A second resident, busy right now, so Status shows both states.
    busy = records[4]
    busy_plan = LoadPlan(
        model_id=busy.id, devices=[3], ctx_size=32768, parallel=1, ctx_per_slot=32768,
        kv_cache_type="q8_0", per_gpu_bytes={3: int(busy.size_bytes + 2.5 * GIB)},
    )
    supervisor.instances[busy.id] = InstanceInfo(
        model_id=busy.id, state="ready", port=CHILD_PORT, pid=4143, plan=busy_plan, priority=2,
        started_at=now - 1900, last_activity_at=now - 1, resolved_engine_tag="b11370",
        last_tokens_per_second=47.2, total_requests=12, active_requests=1,
    )

    class Downloads:
        """Two invented transfers: one running, one waiting."""

        def __init__(self) -> None:
            from studioforge.core.downloader import DownloadProgress

            def item(i: str, repo: str, fname: str, done: float, total: float, status: str) -> Any:
                return DownloadProgress(
                    id=i, group_id=i, repo_id=repo, filename=fname, status=status,  # type: ignore[arg-type]
                    downloaded_bytes=int(done * GIB), total_bytes=int(total * GIB),
                    speed_bps=58 * 1024**2 if status == "running" else 0.0,
                    eta_s=310.0 if status == "running" else None, error=None,
                )

            self.rows = [
                item("g1", "bartowski/GLM-4.5-Air-GGUF", "GLM-4.5-Air-Q4_K_M.gguf", 31.2, 68.4, "running"),
                item("g2", "unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf", 0, 16.5, "queued"),
            ]

        def active(self) -> list[Any]:
            return list(self.rows)

        def all(self) -> list[Any]:
            return list(self.rows)

        def group_status(self, group_id: str) -> str:
            return next((r.status for r in self.rows if r.group_id == group_id), "queued")

        def queued_remaining_bytes(self) -> int:
            return int(53.7 * GIB)

        async def enqueue(self, option: Any) -> str:
            return "demo"

        async def pause(self, gid: str) -> None: ...
        async def resume(self, gid: str) -> None: ...
        async def cancel(self, gid: str) -> None: ...

    class Boot:
        def snapshot(self) -> dict[str, Any]:
            return {"phase": "ready", "ready": True, "elapsed_s": 6.1, "error": None}

    class State:
        pass

    state = State()
    state.config = config
    state.registry = registry
    state.supervisor = supervisor
    state.probe = probe
    state.planner = planner
    state.manager = manager
    from types import SimpleNamespace

    engine = SimpleNamespace(tag="b11370", variant="cuda-13", active=True, smoke_tested=True)
    state.engine_manager = SimpleNamespace(
        active=lambda: engine, list=lambda: [engine], installed=lambda: [engine],
        get=lambda tag: engine,
    )
    state.downloader = Downloads()
    state.db = None
    state.tenants = None
    state.started_at = now - 3 * 3600 - 420
    state.boot = Boot()
    return state


# ---------------------------------------------------------------------------
# A fake llama-server child for Test Chat
# ---------------------------------------------------------------------------


def start_child(port: int) -> None:
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()
    words = (
        "Sure. A llama.cpp server keeps each model in its own process, so a crash in one "
        "never takes the others down. StudioForge plans where each one fits before it starts."
    ).split()

    @app.post("/v1/chat/completions")
    async def chat(body: dict[str, Any]) -> StreamingResponse:
        async def stream() -> Any:
            await asyncio.sleep(0.35)
            for word in words:
                chunk = {"choices": [{"delta": {"content": word + " "}, "index": 0}]}
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.03)
            final = {
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 28, "completion_tokens": len(words)},
                "timings": {"prompt_per_second": 912.0, "predicted_per_second": 61.4,
                            "prompt_n": 28, "predicted_n": len(words)},
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/slots")
    async def slots() -> list[dict[str, Any]]:
        return []

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args(argv)
    port = _guard(args.port)
    _guard(CHILD_PORT)

    import uvicorn

    from studioforge.gui.app import create_gui_app

    with tempfile.TemporaryDirectory(prefix="sf-panel-demo-") as scratch:
        state = build_state(Path(scratch))
        start_child(CHILD_PORT)
        app = create_gui_app(state.config, api_state=state)
        print(f"panel demo on http://127.0.0.1:{port}  (fake rig, scratch data in {scratch})")
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
