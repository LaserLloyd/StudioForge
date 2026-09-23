"""Client attribution: the ``X-SF-Client`` label, carried into loads and requests.

Clients name themselves with ``X-SF-Client`` on every request (OPENCLAW-RIG §4).
Until D70 the label fed an hourly rollup on ``/api/status`` and the lease
holder default, and nothing else: an instance's ``loaded_by`` was the route
literal every caller passed (``jit:/v1/chat/completions``), the request
counter was a bare number, and an explicit unload logged nothing. "Who loaded
that model?" and "whose streams did that unload cut?" were unanswerable, and
on 2026-09-20 the second question had six streams behind it.

Two conventions, both here so the composer and the reader cannot drift:

* :func:`attributed_source` appends the label to a load's ``source`` in
  parentheses -- ``jit:/v1/chat/completions (clawchat)`` -- so ``loaded_by``
  keeps its route literal in front (a reader matching on the prefix is
  unaffected) and :func:`split_source` gives the supervisor the label alone
  for ``loaded_by_client``. Composing it into the string every load path
  already carries means no load signature changes for a label only the
  supervisor stamps.
* :func:`client_of` answers "who is this request from" the way the rollup
  does: the label when the caller sent one, else its peer address, else
  ``None``. The address is the fallback on purpose -- the clients the
  attribution exists to catch are the ones that never identify themselves.
"""

from __future__ import annotations

import re
from typing import Any

#: Longest label carried anywhere. A header is client-controlled text; this is
#: the bound that keeps a log line and a status row readable.
MAX_CLIENT_LABEL = 64

_SUFFIX = re.compile(r"^(?P<base>.*?) \((?P<client>[^()]+)\)$")
_COLLAPSE = re.compile(r"\s+")


def client_label(raw: Any) -> str | None:
    """A clean ``X-SF-Client`` value, or ``None`` when there is nothing usable.

    Whitespace is collapsed, parentheses are dropped (they are the
    :func:`attributed_source` delimiter) and the result is capped, so any
    header value composes into a source string that :func:`split_source`
    reads back exactly.
    """
    if raw is None:
        return None
    text = _COLLAPSE.sub(" ", str(raw)).replace("(", "").replace(")", "").strip()
    if not text:
        return None
    return text[:MAX_CLIENT_LABEL]


def client_of(request: Any) -> str | None:
    """The label the request carried, else its peer address, else ``None``.

    Duck-typed on ``request.headers`` and ``request.client.host`` so the routes
    and the tests' stand-ins can both hand in what they have.
    """
    headers = getattr(request, "headers", None)
    label = client_label(headers.get("x-sf-client") if headers is not None else None)
    if label:
        return label
    peer = getattr(getattr(request, "client", None), "host", None)
    return client_label(peer)


def attributed_source(source: str, client: str | None) -> str:
    """``source`` with the client label appended: ``"jit:/v1/x (clawchat)"``.

    Unchanged when there is no label, so a load nobody attributed keeps the
    exact ``loaded_by`` it always had.
    """
    label = client_label(client)
    return f"{source} ({label})" if label else source


def split_source(source: str | None) -> tuple[str | None, str | None]:
    """The route literal and the label :func:`attributed_source` composed, if any."""
    if not source:
        return source, None
    match = _SUFFIX.match(source)
    if match is None:
        return source, None
    return match.group("base"), match.group("client")
