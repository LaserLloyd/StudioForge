"""Remote image fetch edge cases (WP17 review, open item 3) and the CORS validator (item 1)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from studioforge.api import vision
from studioforge.config import Config, ServerConfig
from studioforge.errors import BadRequestError


def _config() -> Config:
    config = Config(data_dir="/tmp/sf-vision")
    # Loopback would be refused by the SSRF guard; the test transport never
    # touches the network anyway, so allow it here.
    config.gateway.allow_private_image_hosts = True
    return config


async def test_redirect_budget_exhaustion_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past _MAX_REDIRECTS the old loop handed the 3xx BODY to the image decoder,
    so the user read 'could not decode image' for 'too many redirects'."""
    hops = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hops["n"] += 1
        return httpx.Response(302, headers={"location": f"http://127.0.0.1/hop{hops['n']}"},
                              content=b"<html>moved</html>")

    monkeypatch.setattr(vision, "_is_public_address", lambda host: True)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BadRequestError) as excinfo:
            await vision._fetch("http://127.0.0.1/start", config=_config(), client=client)
    assert "too many redirects" in excinfo.value.message
    assert hops["n"] == vision._MAX_REDIRECTS + 1, "the budget is followed, then refused"


def test_credentialed_wildcard_cors_is_refused() -> None:
    with pytest.raises(ValueError, match="cors_allow_credentials"):
        ServerConfig(cors_origins=["*"], cors_allow_credentials=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cors_origins": ["*"], "cors_allow_credentials": False},
        {"cors_origins": ["https://app.example"], "cors_allow_credentials": True},
    ],
)
def test_safe_cors_pairs_are_accepted(kwargs: dict[str, Any]) -> None:
    assert ServerConfig(**kwargs).cors_allow_credentials is kwargs["cors_allow_credentials"]
