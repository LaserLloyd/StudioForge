"""Container liveness probe: ``python -m studioforge.healthcheck``.

The image's ``HEALTHCHECK`` runs this instead of a hard-coded ``http://`` URL,
because ``server.tls.mode`` (``self_signed`` / ``cert``, which ``studioforge
setup --exposure lan`` turns on) moves the gateway to HTTPS and an ``http://``
probe would then report a perfectly healthy container as unhealthy forever.

The scheme and port come from the same place the server reads them: the
effective config (``config.yaml`` in ``$SF_DATA_DIR`` with the ``SF_*``
environment on top). The probe always talks to loopback inside its own
container, so with TLS on it does not verify the certificate (a self-signed
certificate has no name for ``127.0.0.1``; there is no network between the
probe and the process). If the config cannot be read, both schemes are tried:
an unreadable config is the server's problem to report, not a reason to mark
a running gateway dead.

Exit status: ``0`` when ``/health`` answers 200 with ``"status": "ok"``, else ``1``.
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence

DEFAULT_PORT = 1234
TIMEOUT_S = 4.0


def candidate_urls(config: object | None = None) -> list[str]:
    """The ``/health`` URLs to try, most likely first.

    With a config: exactly the one the server listens on (``server.tls.mode``
    decides the scheme, ``server.port`` the port). Without one (it could not be
    loaded): ``http`` then ``https`` on the default port.
    """
    if config is not None:
        from studioforge.cli_client import base_url_for

        return [f"{base_url_for(config)}/health"]  # type: ignore[arg-type]
    return [f"http://127.0.0.1:{DEFAULT_PORT}/health", f"https://127.0.0.1:{DEFAULT_PORT}/health"]


def _load_config() -> object | None:
    try:
        from studioforge.config import load_config

        return load_config()
    except Exception:  # noqa: BLE001 - see the module docstring
        return None


def probe(url: str, *, timeout: float = TIMEOUT_S) -> bool:
    """Whether ``url`` answers 200 with ``{"status": "ok"}``. Never raises."""
    context: ssl.SSLContext | None = None
    if url.startswith("https://"):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    try:
        # Fixed scheme (http/https, checked above) and a loopback URL we built.
        with urllib.request.urlopen(url, timeout=timeout, context=context) as response:  # noqa: S310
            if response.status != 200:
                return False
            body = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return isinstance(body, dict) and body.get("status") == "ok"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. ``argv`` may carry one explicit URL (used by the tests and by hand)."""
    args = list(sys.argv[1:] if argv is None else argv)
    targets = args[:1] or candidate_urls(_load_config())
    return 0 if any(probe(url) for url in targets) else 1


if __name__ == "__main__":  # pragma: no cover - exercised through `python -m`
    raise SystemExit(main())
