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
        return httpx.Response(
            302,
            headers={"location": f"http://127.0.0.1/hop{hops['n']}"},
            content=b"<html>moved</html>",
        )

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


# ---------------------------------------------------------------------------
# DNS-rebinding TOCTOU: connect to the address that was vetted, with the
# original Host header and SNI (WP17 open item 2).
# ---------------------------------------------------------------------------


async def test_the_fetch_connects_to_the_vetted_ip_with_the_original_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["host_header"] = request.headers.get("host")
        seen["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, content=b"\x89PNG", headers={"content-type": "image/png"})

    # The resolver is the only network the guard touches; pin it.
    monkeypatch.setattr(vision, "_resolve_public", lambda host: "93.184.216.34")
    config = Config(data_dir="/tmp/sf-vision")
    config.gateway.allow_private_image_hosts = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        raw, mime = await vision._fetch(
            "https://images.example.com:8443/cat.png", config=config, client=client
        )
    assert raw == b"\x89PNG" and mime == "image/png"
    assert seen["url"] == "https://93.184.216.34:8443/cat.png"
    assert seen["host_header"] == "images.example.com:8443"
    assert seen["sni"] == "images.example.com"


async def test_a_redirect_is_re_vetted_and_re_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []
    answers = {"first.example.com": "93.184.216.34", "second.example.com": "198.51.100.7"}

    def handler(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if len(urls) == 1:
            return httpx.Response(302, headers={"location": "http://second.example.com/img"})
        return httpx.Response(200, content=b"img", headers={"content-type": "image/jpeg"})

    monkeypatch.setattr(vision, "_resolve_public", lambda host: answers.get(host))
    config = Config(data_dir="/tmp/sf-vision")
    config.gateway.allow_private_image_hosts = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await vision._fetch("http://first.example.com/start", config=config, client=client)
    assert urls == ["http://93.184.216.34/start", "http://198.51.100.7/img"]


async def test_a_redirect_into_private_space_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://internal.example.com/secret"})

    def resolver(host: str) -> str | None:
        return "93.184.216.34" if host == "first.example.com" else None

    monkeypatch.setattr(vision, "_resolve_public", resolver)
    config = Config(data_dir="/tmp/sf-vision")
    config.gateway.allow_private_image_hosts = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BadRequestError) as excinfo:
            await vision._fetch("http://first.example.com/start", config=config, client=client)
    assert excinfo.value.code == "image_fetch_refused"


def test_one_private_answer_among_public_ones_fails_the_vet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket

    def fake_getaddrinfo(host: str, port: Any) -> list[Any]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert vision._resolve_public("rebinder.example.com") is None
    assert vision._is_public_address("rebinder.example.com") is False


@pytest.mark.parametrize(
    "literal",
    [
        # CGNAT is the tailnet (netinfo._TAILSCALE_NET): `is_private` never
        # covered it, so the guard's own stated purpose was not met.
        "100.100.7.42",
        "100.64.0.1",
        "100.127.255.254",
        "::ffff:100.100.7.42",
        # The NAT64 well-known prefix is refused today via `is_reserved`; pin
        # that the additive `is_global` check did not loosen it.
        "64:ff9b::808:808",
    ],
)
def test_cgnat_and_other_non_global_space_is_refused(
    monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    import socket

    family = socket.AF_INET6 if ":" in literal else socket.AF_INET

    def fake_getaddrinfo(host: str, port: Any) -> list[Any]:
        return [(family, socket.SOCK_STREAM, 6, "", (literal, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert vision._resolve_public("peer.example") is None
    assert vision._is_public_address("peer.example") is False


@pytest.mark.parametrize("literal", ["8.8.8.8", "2001:4860:4860::8888", "::ffff:8.8.8.8"])
def test_genuinely_public_space_still_resolves(
    monkeypatch: pytest.MonkeyPatch, literal: str
) -> None:
    import ipaddress
    import socket

    family = socket.AF_INET6 if ":" in literal else socket.AF_INET

    def fake_getaddrinfo(host: str, port: Any) -> list[Any]:
        return [(family, socket.SOCK_STREAM, 6, "", (literal, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert vision._resolve_public("cdn.example") == str(ipaddress.ip_address(literal))


async def test_a_tailnet_image_url_is_refused_before_any_request_is_made() -> None:
    """End to end on the shipped default: no credential, no private hosts."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(403, content=b"secret")

    config = Config(data_dir="/tmp/sf-vision")
    config.gateway.allow_private_image_hosts = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BadRequestError) as excinfo:
            await vision._fetch("http://100.100.7.42:8080/x.png", config=config, client=client)
    assert excinfo.value.code == "image_fetch_refused"
    assert calls == [], "the target's status must not be observable"


def test_a_malformed_content_length_is_not_a_500() -> None:
    import asyncio

    async def run() -> None:
        response = httpx.Response(
            200, content=b"img", headers={"content-length": "lots", "content-type": "image/png"}
        )
        gateway = Config(data_dir="/tmp/sf-vision").gateway
        raw, _ = await vision._read_capped(response, gateway)
        assert raw == b"img"

    asyncio.run(run())
