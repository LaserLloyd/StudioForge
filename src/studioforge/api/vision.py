"""Multimodal content handling for OpenAI-style content arrays.

OpenClaw sends images today, so this path is load-bearing, not an add-on. The
job here is to turn every shape a client might send into the one shape
``llama-server`` reliably accepts -- a base64 ``data:`` URL -- while enforcing
count/size limits and failing clearly when the target model cannot see.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import re
from typing import Any

import httpx

from studioforge.config import Config
from studioforge.errors import BadRequestError
from studioforge.logging import get_logger
from studioforge.types import ModelRecord

log = get_logger(__name__)

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.S
)

# Formats llama.cpp's mtmd stack reads. Anything else gets transcoded to PNG.
_SAFE_MIME = {"image/png", "image/jpeg", "image/webp", "image/bmp", "image/gif"}


class ImageStats:
    """Counters for one request, surfaced in logs."""

    def __init__(self) -> None:
        self.count = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.resized = 0
        self.fetched = 0


def message_has_image(messages: list[dict[str, Any]]) -> bool:
    """True when any message carries an image part."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "input_image", "image"}:
                return True
    return False


async def prepare_messages(
    messages: list[dict[str, Any]],
    *,
    record: ModelRecord,
    config: Config,
    client: httpx.AsyncClient,
) -> tuple[list[dict[str, Any]], ImageStats]:
    """Normalize content arrays and enforce the image policy.

    Returns a new message list; the input is not mutated (callers reuse the
    original body for logging).
    """
    stats = ImageStats()
    if not message_has_image(messages):
        return messages, stats

    if not record.capabilities.vision:
        # A clear 400 rather than forwarding an image llama-server would either
        # ignore or crash on. This is an explicit acceptance criterion.
        raise BadRequestError(
            f"Model '{record.id}' does not support image input. It has no multimodal "
            f"projector (mmproj) attached. Choose a vision-capable model -- "
            f"GET /v1/models reports 'vision' in each model's capabilities.",
            code="model_not_multimodal",
            param="messages",
        )

    limit = config.gateway.max_images_per_request
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        new_parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                new_parts.append(part)
                continue
            if part.get("type") not in {"image_url", "input_image", "image"}:
                new_parts.append(part)
                continue
            stats.count += 1
            if stats.count > limit:
                raise BadRequestError(
                    f"Too many images in one request: limit is {limit} "
                    f"(gateway.max_images_per_request).",
                    code="too_many_images",
                    param="messages",
                )
            url = _extract_url(part)
            data_url = await _to_data_url(url, config=config, client=client, stats=stats)
            new_parts.append({"type": "image_url", "image_url": {"url": data_url}})
        out.append({**message, "content": new_parts})

    log.info(
        "prepared images",
        model_id=record.id,
        images=stats.count,
        fetched=stats.fetched,
        resized=stats.resized,
        bytes_in=stats.bytes_in,
        bytes_out=stats.bytes_out,
    )
    return out, stats


def _extract_url(part: dict[str, Any]) -> str:
    """Pull the URL out of any of the shapes clients use."""
    image_url = part.get("image_url")
    if isinstance(image_url, dict):
        url = image_url.get("url")
    elif isinstance(image_url, str):
        url = image_url
    else:
        url = part.get("url") or part.get("image")
    if not isinstance(url, str) or not url:
        raise BadRequestError(
            "image part is missing a usable 'url'", code="invalid_image", param="messages"
        )
    return url


async def _to_data_url(
    url: str, *, config: Config, client: httpx.AsyncClient, stats: ImageStats
) -> str:
    gateway = config.gateway
    if url.startswith("data:"):
        raw, mime = _decode_data_url(url)
        stats.bytes_in += len(raw)
    elif url.startswith(("http://", "https://")):
        raw, mime = await _fetch(url, config=config, client=client)
        stats.fetched += 1
        stats.bytes_in += len(raw)
    else:
        raise BadRequestError(
            "image_url must be a data: URL or an http(s) URL; local file paths are "
            "not accepted by the server.",
            code="invalid_image_url",
            param="messages",
        )

    if len(raw) > gateway.max_image_bytes:
        raise BadRequestError(
            f"image is {len(raw)} bytes, over the {gateway.max_image_bytes} byte limit "
            f"(gateway.max_image_bytes).",
            code="image_too_large",
            param="messages",
        )

    # Decode + resize + re-encode is CPU-bound; on the loop it would stall every
    # concurrent SSE stream for the duration of a large image.
    raw, mime, resized = await asyncio.to_thread(_maybe_resize, raw, mime, gateway.max_image_dim)
    if resized:
        stats.resized += 1
    stats.bytes_out += len(raw)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _decode_data_url(url: str) -> tuple[bytes, str]:
    match = _DATA_URL_RE.match(url)
    if match is None:
        raise BadRequestError("malformed data: URL", code="invalid_image", param="messages")
    mime = match.group("mime") or "image/png"
    payload = match.group("data")
    if match.group("b64"):
        try:
            raw = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise BadRequestError(
                f"image data is not valid base64: {exc}", code="invalid_image", param="messages"
            ) from exc
    else:
        from urllib.parse import unquote_to_bytes

        raw = unquote_to_bytes(payload)
    if not raw:
        raise BadRequestError("image data is empty", code="invalid_image", param="messages")
    return raw, mime


#: Fetching an image is the one place the server makes an outbound request to
#: an address a CALLER chose, so it is the one place SSRF is possible. On an
#: open install /v1/chat/completions needs no credential, which made the server
#: a probe for anything else on the tailnet or LAN: the error text echoed the
#: target's HTTP status straight back. Refuse to talk to non-public addresses,
#: and re-check on every redirect hop rather than trusting the first one.
def _is_public_address(host: str) -> bool:
    """False for loopback, link-local, private, ULA and other reserved space."""
    import ipaddress
    import socket

    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    if not infos:
        return False
    for info in infos:
        raw = info[4][0]
        try:
            address = ipaddress.ip_address(raw)
        except ValueError:
            return False
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            return False
    return True


async def _guard_target(url: str, *, allow_private: bool) -> None:
    from urllib.parse import urlparse

    if allow_private:
        return
    host = urlparse(url).hostname or ""
    if not host:
        raise BadRequestError(
            "image URL has no host", code="image_fetch_refused", param="messages"
        )
    if not await asyncio.to_thread(_is_public_address, host):
        # Deliberately does NOT say whether the host resolved, was refused or
        # timed out: that difference is the oracle worth denying.
        raise BadRequestError(
            "refusing to fetch an image from a non-public address. Pass the image "
            "inline as a data: URL instead.",
            code="image_fetch_refused",
            param="messages",
        )


async def _fetch(url: str, *, config: Config, client: httpx.AsyncClient) -> tuple[bytes, str]:
    """Fetch a remote image with a hard size cap and timeout.

    The cap is enforced while streaming rather than after the fact, so a
    hostile or mistaken URL pointing at a huge file cannot exhaust memory
    before we notice.
    """
    gateway = config.gateway
    await _guard_target(url, allow_private=gateway.allow_private_image_hosts)
    try:
        # follow_redirects=False: a public URL that 302s into 127.0.0.1 would
        # otherwise walk straight past the guard above. Hops are followed by
        # hand so each new target is re-checked.
        hops = 0
        current = url
        while True:
            async with client.stream(
                "GET", current, timeout=gateway.image_fetch_timeout_s, follow_redirects=False
            ) as response:
                if response.is_redirect and hops >= _MAX_REDIRECTS:
                    # Past the budget the old code fell through and handed the
                    # 3xx *body* to the image decoder, so the user read "could
                    # not decode image" for what was really "too many redirects".
                    break
                if response.is_redirect:
                    location = response.headers.get("location") or ""
                    if not location:
                        break
                    current = str(httpx.URL(current).join(location))
                    await _guard_target(
                        current, allow_private=gateway.allow_private_image_hosts
                    )
                    hops += 1
                    continue
                return await _read_capped(response, gateway)
    except httpx.TimeoutException as exc:
        raise BadRequestError(
            f"timed out after {gateway.image_fetch_timeout_s}s fetching the image",
            code="image_fetch_timeout",
            param="messages",
        ) from exc
    except httpx.HTTPError as exc:
        raise BadRequestError(
            f"could not fetch the image: {type(exc).__name__}",
            code="image_fetch_failed",
            param="messages",
        ) from exc
    raise BadRequestError(
        "too many redirects fetching the image",
        code="image_fetch_failed",
        param="messages",
    )


_MAX_REDIRECTS = 3


async def _read_capped(
    response: httpx.Response, gateway: Any
) -> tuple[bytes, str]:
    """Stream the body with the size cap enforced as it arrives.

    The cap is checked while streaming rather than after the fact, so a hostile
    or mistaken URL pointing at a huge file cannot exhaust memory before we
    notice. Error text never echoes the URL or the upstream status -- that
    difference is exactly the SSRF oracle worth denying.
    """
    if response.status_code >= 400:
        raise BadRequestError(
            f"could not fetch image: HTTP {response.status_code}",
            code="image_fetch_failed",
            param="messages",
        )
    declared = response.headers.get("content-length")
    if declared and int(declared) > gateway.max_image_bytes:
        raise BadRequestError(
            f"remote image is {declared} bytes, over the "
            f"{gateway.max_image_bytes} byte limit.",
            code="image_too_large",
            param="messages",
        )
    mime = (response.headers.get("content-type") or "image/png").split(";")[0].strip()
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > gateway.max_image_bytes:
            raise BadRequestError(
                f"remote image exceeded the {gateway.max_image_bytes} byte limit "
                f"while downloading.",
                code="image_too_large",
                param="messages",
            )
        chunks.append(chunk)
    return b"".join(chunks), mime


def _maybe_resize(raw: bytes, mime: str, max_dim: int) -> tuple[bytes, str, bool]:
    """Downscale oversized images and transcode exotic formats to PNG.

    Oversized images cost context tokens and encode time far more than they add
    accuracy, and a format llama.cpp cannot decode would fail deep inside the
    child process where the error is unrecoverable.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow is a hard dependency
        return raw, mime, False

    needs_transcode = mime not in _SAFE_MIME
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            width, height = opened.size
            if max(width, height) <= max_dim and not needs_transcode:
                return raw, mime, False
            # A separate name from the context-managed ImageFile: convert() and
            # resize() return plain Image objects, and rebinding the original
            # confuses both readers and type checkers.
            image: Any = opened
            if opened.mode not in {"RGB", "RGBA", "L"}:
                image = opened.convert("RGB")
            did_resize = False
            if max(width, height) > max_dim:
                scale = max_dim / float(max(width, height))
                image = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
                did_resize = True
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            return buffer.getvalue(), "image/png", did_resize
    except Exception as exc:
        raise BadRequestError(
            f"could not decode image ({mime}): {exc}", code="invalid_image", param="messages"
        ) from exc
