"""Prometheus ``/metrics``: gateway counters plus the loaded models' own metrics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from studioforge.api.app import build_state, create_app
from studioforge.config import Config
from studioforge.core.metrics import GatewayMetrics, label_child_metrics, render_metrics

CHILD_TEXT = """\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 1234
# HELP llamacpp:requests_processing Number of requests processing.
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
llamacpp:kv_cache_usage_ratio{slot="0"} 0.5
"""


def test_child_samples_gain_a_model_label_and_metadata_is_untouched() -> None:
    lines = label_child_metrics(CHILD_TEXT, 'org/Model "7B"')
    assert "# TYPE llamacpp:prompt_tokens_total counter" in lines
    assert 'llamacpp:prompt_tokens_total{model="org/Model \\"7B\\""} 1234' in lines
    assert 'llamacpp:kv_cache_usage_ratio{slot="0",model="org/Model \\"7B\\""} 0.5' in lines


def test_request_counters_use_route_templates_and_status_classes() -> None:
    metrics = GatewayMetrics()
    metrics.begin()
    metrics.observe("GET", "/api/models/{model_id}", 200, 0.5)
    metrics.begin()
    metrics.observe("GET", "/api/models/{model_id}", 404, 0.25)
    snap = metrics.snapshot()
    assert snap["requests"][("GET", "/api/models/{model_id}", "2xx")] == 1
    assert snap["requests"][("GET", "/api/models/{model_id}", "4xx")] == 1
    assert snap["duration_sum"][("GET", "/api/models/{model_id}")] == pytest.approx(0.75)
    assert snap["in_flight"] == 0


class FakeSupervisor:
    def __init__(self, texts: dict[str, str | None]) -> None:
        self._texts = texts

    def list(self) -> list[Any]:
        return [SimpleNamespace(model_id=m) for m in self._texts]

    async def metrics(self, model_id: str) -> str | None:
        text = self._texts[model_id]
        if text == "boom":
            raise RuntimeError("child died")
        return text


async def test_render_merges_children_and_survives_a_dead_one() -> None:
    gpu = SimpleNamespace(index=0, name="GPU 0", total_bytes=24 << 30, free_bytes=10 << 30)
    state = SimpleNamespace(
        metrics=GatewayMetrics(),
        started_at=0.0,
        supervisor=FakeSupervisor({"a/one": CHILD_TEXT, "b/two": CHILD_TEXT, "c/dead": "boom"}),
        probe=SimpleNamespace(list_gpus=lambda: [gpu]),
        accounts=None,
    )
    state.metrics.observe("GET", "/api/status", 200, 0.1)
    text = await render_metrics(state)
    assert (
        'studioforge_http_requests_total{method="GET",route="/api/status",status="2xx"} 1' in text
    )
    assert "studioforge_models_loaded 3" in text
    assert 'studioforge_gpu_memory_total_bytes{gpu="0",name="GPU 0"}' in text
    assert 'llamacpp:prompt_tokens_total{model="a/one"} 1234' in text
    assert 'llamacpp:prompt_tokens_total{model="b/two"} 1234' in text
    assert text.count("# TYPE llamacpp:prompt_tokens_total counter") == 1  # metadata once
    assert 'model="c/dead"' not in text  # a dead child contributes nothing
    assert text.endswith("\n")


def make_app(tmp_path: Path) -> Any:
    config = Config(
        data_dir=tmp_path / "data",
        models={"dir": tmp_path / "models"},
        gui={"enabled": False, "port": 18080},
        watchdog={"enabled": False, "port": 11235},
        server={"port": 11234},
        logging={"level": "ERROR"},
    )
    return create_app(config, state=build_state(config), start_background=False)


def test_metrics_route_counts_requests_by_route_template(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        client.get("/api/status")
        client.get("/api/models/some/model/settings")
        reply = client.get("/metrics")
    assert reply.status_code == 200
    assert reply.headers["content-type"].startswith("text/plain")
    body = reply.text
    assert 'route="/api/status"' in body
    assert 'route="/api/models/{model_id}/settings"' in body  # a template, not the URL
    assert "some/model" not in body
    assert "studioforge_build_info" in body and "studioforge_uptime_seconds" in body


def test_metrics_need_a_read_key_once_authentication_is_on(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    _r, read = app.state.accounts.create_key("prom", "read")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": f"Bearer {read}"})
    assert ok.status_code == 200
    assert 'studioforge_api_keys_active{scope="read"} 1' in ok.text
    assert "sf-r-" not in ok.text  # counts, never keys


def test_rejected_requests_are_counted_too(tmp_path: Path) -> None:
    app = make_app(tmp_path)
    app.state.accounts.create_key("x", "admin")
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        client.get("/api/status")  # 401
        _r, read = app.state.accounts.create_key("prom", "read")
        body = client.get("/metrics", headers={"Authorization": f"Bearer {read}"}).text
    assert 'route="(rejected)"' in body and 'status="4xx"' in body
