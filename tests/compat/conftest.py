"""Compatibility suite against a RUNNING StudioForge test instance (brief G3).

Opt-in twice, like ``tests/contract``: every item is marked ``compat`` and is
skipped unless ``COMPAT_BASE_URL`` is set. It is not part of the default unit
run (``pytest tests/unit``) and never starts a server or loads a model itself --
point it at an instance you started on an ALTERNATE port (the brief's test
instance is ``http://127.0.0.1:11234``), never at a live install::

    COMPAT_BASE_URL=http://127.0.0.1:11234 \
    COMPAT_MODEL=<a chat model id the instance has> \
    uv run pytest tests/compat -m compat -q -p no:cacheprovider

Environment:

``COMPAT_BASE_URL``        required; the instance root (no ``/v1``).
``COMPAT_API_KEY``         bearer / x-api-key value (default ``sk-compat``; an open
                           instance accepts any value).
``COMPAT_MODEL``           chat + tools model id (needed by most tests).
``COMPAT_EMBED_MODEL``     embedding model id (the embeddings test skips without it).
``COMPAT_VISION_MODEL``    a model with an mmproj attached (the vision test skips without it).
``COMPAT_OVERSIZE_MODEL``  a model that cannot fit this rig (the 507 test skips without it).
``COMPAT_LMSTUDIO_URL``    a real LM Studio >= 0.4 to diff JSON shapes against (optional).
``COMPAT_REFUSE_PORTS``    comma list of ports the suite refuses to touch; defaults to the
                           live service ports.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

LIVE_PORTS = "1234,1235,1236,8080,8700,8288,7865"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "compat: needs a running StudioForge test instance (COMPAT_BASE_URL); never run in CI",
    )


def _base_url() -> str | None:
    value = os.environ.get("COMPAT_BASE_URL", "").strip().rstrip("/")
    return value or None


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).parent
    ours = [item for item in items if here in Path(str(item.fspath)).parents]
    for item in ours:
        item.add_marker(pytest.mark.compat)
    if _base_url() is None:
        skip = pytest.mark.skip(reason="COMPAT_BASE_URL is not set (no test instance)")
        for item in ours:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def base_url() -> str:
    url = _base_url()
    assert url is not None
    port = urlsplit(url).port
    refused = {
        int(p) for p in os.environ.get("COMPAT_REFUSE_PORTS", LIVE_PORTS).split(",") if p.strip()
    }
    if port in refused:
        pytest.exit(
            f"COMPAT_BASE_URL points at port {port}, which is a live service port. "
            "Start a test instance on an alternate port (e.g. 11234) instead.",
            returncode=2,
        )
    return url


@pytest.fixture(scope="session")
def api_key() -> str:
    return os.environ.get("COMPAT_API_KEY", "sk-compat")


def _need(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is not set")
    return value


@pytest.fixture(scope="session")
def chat_model() -> str:
    return _need("COMPAT_MODEL")


@pytest.fixture(scope="session")
def embed_model() -> str:
    return _need("COMPAT_EMBED_MODEL")


@pytest.fixture(scope="session")
def vision_model() -> str:
    return _need("COMPAT_VISION_MODEL")


@pytest.fixture(scope="session")
def oversize_model() -> str:
    return _need("COMPAT_OVERSIZE_MODEL")


@pytest.fixture()
def http(base_url: str, api_key: str) -> Iterator[httpx.Client]:
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=600.0) as client:
        yield client


@pytest.fixture()
def openai_client(base_url: str, api_key: str) -> Any:
    openai = pytest.importorskip("openai")
    return openai.OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=600.0, max_retries=0)


@pytest.fixture()
def anthropic_client(base_url: str, api_key: str) -> Any:
    anthropic = pytest.importorskip("anthropic")
    return anthropic.Anthropic(base_url=base_url, api_key=api_key, timeout=600.0, max_retries=0)
