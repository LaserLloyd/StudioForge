"""The panel's navigation: four tabs, one Advanced sub-navigation, and every old name.

This module is pure data and pure functions (no NiceGUI), so the link grammar can be
tested without rendering a page. ``app.py`` builds the shell from it; ``docs/panel.md``
documents it.

The 2.0.x panel had eleven top-level sections (and, before that, seven tabs). Both name
sets are still accepted in links: :func:`resolve_location` maps any of them to its new
place, and :func:`needs_redirect` says whether the address bar should be rewritten to the
canonical form.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import quote, urlencode

#: The top line, exactly four entries, in this order.
STATUS: Final = "Status"
MODELS: Final = "Models"
TEST_CHAT: Final = "Test Chat"
ADVANCED: Final = "Advanced"
TOP_TABS: Final[tuple[str, ...]] = (STATUS, MODELS, TEST_CHAT, ADVANCED)

#: Models page filter chips.
FILTER_ALL: Final = "all"
FILTER_LOADED: Final = "loaded"
FILTER_DOWNLOADS: Final = "downloads"
MODEL_FILTERS: Final[tuple[str, ...]] = (FILTER_ALL, FILTER_LOADED, FILTER_DOWNLOADS)


@dataclass(frozen=True)
class AdvancedPage:
    """One entry of the Advanced sub-navigation."""

    slug: str
    label: str
    icon: str
    summary: str


#: The Advanced sub-navigation, top to bottom. ``slug`` is the ``?page=`` value.
ADVANCED_PAGES: Final[tuple[AdvancedPage, ...]] = (
    AdvancedPage(
        "serving",
        "Serving & network",
        "lan",
        "Port, bind addresses, HTTPS, idle unloading, browser access and credentials.",
    ),
    AdvancedPage(
        "placement",
        "GPUs & placement",
        "developer_board",
        "What the planner may use, who holds VRAM, standing leases and registered tenants.",
    ),
    AdvancedPage(
        "accounts",
        "Accounts & keys",
        "vpn_key",
        "Owner accounts, API keys and the audit log.",
    ),
    AdvancedPage(
        "engines",
        "Engines",
        "memory",
        "Which llama.cpp build serves, installs, smoke tests and what each build can run.",
    ),
    AdvancedPage(
        "watchdog",
        "Watchdog & recovery",
        "health_and_safety",
        "The recovery sidecar's settings, plus restart and unload-everything controls.",
    ),
    AdvancedPage(
        "catalog",
        "Catalog & downloads",
        "folder_open",
        "Where models live, defaults for every load, and Hugging Face download settings.",
    ),
    AdvancedPage(
        "integrations",
        "Integrations",
        "link",
        "Copy-paste connection snippets, OpenClaw, MCP and the lmstudio:// handler.",
    ),
    AdvancedPage(
        "peers",
        "Peers",
        "hub",
        "Other StudioForge machines. Arrives with multi-host support.",
    ),
    AdvancedPage(
        "logs",
        "Logs & metrics",
        "article",
        "The live log and the Prometheus text.",
    ),
    AdvancedPage(
        "config",
        "Config",
        "tune",
        "Every setting, rendered from the settings schema, and the raw configuration.",
    ),
    AdvancedPage(
        "diagnostics",
        "Diagnostics",
        "troubleshoot",
        "Readiness checklist, health, doctor checks and the support bundle.",
    ),
    AdvancedPage(
        "about",
        "About & updates",
        "info",
        "Version, build and uptime; check for updates, install, roll back.",
    ),
)

ADVANCED_SLUGS: Final[tuple[str, ...]] = tuple(page.slug for page in ADVANCED_PAGES)
DEFAULT_ADVANCED_PAGE: Final = ADVANCED_PAGES[0].slug
#: Where an old ``?section=Advanced`` (the 2.0.x page of every setting) lands.
LEGACY_ADVANCED_PAGE: Final = "config"

_PAGE_BY_SLUG: Final[Mapping[str, AdvancedPage]] = {page.slug: page for page in ADVANCED_PAGES}


def advanced_page(slug: str) -> AdvancedPage | None:
    return _PAGE_BY_SLUG.get(slug)


@dataclass(frozen=True)
class Location:
    """Where the panel opens: a tab, plus the page or filter inside it."""

    tab: str = STATUS
    page: str | None = None  # Advanced only
    filter: str | None = None  # Models only
    q: str = ""  # Models search text
    model: str | None = None  # Models: open this model's drawer
    repo: str | None = None  # Models: open the add-model dialog on this repo
    quant: str | None = None

    def __post_init__(self) -> None:
        if self.tab not in TOP_TABS:
            raise ValueError(f"unknown tab {self.tab!r}")
        if self.page is not None and self.page not in ADVANCED_SLUGS:
            raise ValueError(f"unknown Advanced page {self.page!r}")


#: Every name the 2.0.x panel accepted, lower-cased, and where it goes now.
#: ``(tab, advanced page, models filter)``; the first two keep their meaning, the
#: third is the Models chip the old place maps to.
#:
#: Two generations are covered. The seven 2.0 tabs (``?tab=``) are the keys of
#: ``gui.state.DEEP_LINK_TABS``; the eleven 2.0.x sections (``?section=``) are
#: ``TAB_NAMES`` of the previous ``app.py``.
LEGACY_ROUTES: Final[Mapping[str, tuple[str, str | None, str | None]]] = {
    # --- the eleven sections of 2.0.x ----------------------------------------
    "overview": (STATUS, None, None),
    "serving": (ADVANCED, "serving", None),
    "engines": (ADVANCED, "engines", None),
    "placement": (ADVANCED, "placement", None),
    "integrations": (ADVANCED, "integrations", None),
    "users & keys": (ADVANCED, "accounts", None),
    "users": (ADVANCED, "accounts", None),
    "users-keys": (ADVANCED, "accounts", None),
    "keys": (ADVANCED, "accounts", None),
    "updates": (ADVANCED, "about", None),
    "logs & metrics": (ADVANCED, "logs", None),
    "logs-metrics": (ADVANCED, "logs", None),
    "metrics": (ADVANCED, "logs", None),
    "diagnostics": (ADVANCED, "diagnostics", None),
    # --- the seven tabs of 2.0 (state.DEEP_LINK_TABS) ------------------------
    "dashboard": (STATUS, None, None),
    "setup": (STATUS, None, None),
    "download": (MODELS, None, FILTER_DOWNLOADS),
    "downloads": (MODELS, None, FILTER_DOWNLOADS),
    "chat": (TEST_CHAT, None, None),
    "server": (ADVANCED, "integrations", None),
    "settings": (ADVANCED, "integrations", None),
    "logs": (ADVANCED, "logs", None),
    "log": (ADVANCED, "logs", None),
    "model": (MODELS, None, None),
}

#: Spellings of the new canonical names (and the short forms a hand-typed link uses).
_TAB_ALIASES: Final[Mapping[str, str]] = {
    "status": STATUS,
    "models": MODELS,
    "test chat": TEST_CHAT,
    "test-chat": TEST_CHAT,
    "testchat": TEST_CHAT,
    "advanced": ADVANCED,
}

#: Query keys the canonical URL keeps.
_PASSTHROUGH: Final[tuple[str, ...]] = ("q", "model", "repo", "quant")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def old_names() -> tuple[str, ...]:
    """Every legacy name that must still redirect, lower-cased (for tests and docs)."""
    return tuple(sorted(LEGACY_ROUTES))


def resolve_location(query: Mapping[str, Any] | None) -> Location:
    """The place a ``/?...`` query string opens.

    ``section`` (new and 2.0.x) and ``tab`` (2.0) are both read; ``section`` wins when
    both are present. An unknown or missing name opens Status, never an error page. A
    ``repo`` with no tab opens the add-model dialog on Models, which is what the protocol
    handler's ``?tab=download&repo=owner/repo`` has always meant.
    """
    query = query or {}
    name = (_clean(query.get("section")) or _clean(query.get("tab"))).lower()
    page = _clean(query.get("page")).lower() or None
    flt = _clean(query.get("filter")).lower() or None
    text = _clean(query.get("q"))
    model = _clean(query.get("model")) or None
    repo = _clean(query.get("repo") or query.get("model_repo")) or None
    quant = _clean(query.get("quant")) or None

    had_page = page is not None
    tab: str | None = _TAB_ALIASES.get(name)
    legacy = LEGACY_ROUTES.get(name)
    if tab is None and legacy is not None:
        tab, legacy_page, legacy_filter = legacy
        page = page or legacy_page
        flt = flt or legacy_filter
    if tab is None:
        # No (known) name: a bare repo/model link still means "the Models page".
        tab = MODELS if (repo or model) else STATUS
    if tab == ADVANCED:
        if page not in ADVANCED_SLUGS:
            # In 2.0.x "Advanced" was the page of every setting, and the only way to reach
            # it was ``?section=Advanced``. The app itself always writes a ``page`` now, so
            # a link without one is an old link: it keeps its meaning (Config) and is
            # redirected to the canonical form. The top tab itself opens its first page.
            page = (
                LEGACY_ADVANCED_PAGE
                if name == "advanced" and not had_page
                else DEFAULT_ADVANCED_PAGE
            )
        return Location(tab=tab, page=page)
    if tab == MODELS:
        flt = flt if flt in MODEL_FILTERS else None
        if repo and not flt:
            flt = FILTER_DOWNLOADS
        return Location(
            tab=tab, filter=flt or FILTER_ALL, q=text, model=model, repo=repo, quant=quant
        )
    return Location(tab=tab)


def canonical_query(location: Location) -> dict[str, str]:
    """The canonical query string for ``location`` (empty for the default, Status)."""
    out: dict[str, str] = {}
    if location.tab == STATUS:
        return out
    out["section"] = location.tab
    if location.tab == ADVANCED and location.page:
        out["page"] = location.page
    if location.tab == MODELS:
        if location.filter and location.filter != FILTER_ALL:
            out["filter"] = location.filter
        for key, value in (
            ("q", location.q),
            ("model", location.model),
            ("repo", location.repo),
            ("quant", location.quant),
        ):
            if value:
                out[key] = value
    return out


def canonical_url(location: Location) -> str:
    """A relative link to ``location`` (never an absolute URL)."""
    query = canonical_query(location)
    return "/?" + urlencode(query, safe="/", quote_via=quote) if query else "/"


def needs_redirect(query: Mapping[str, Any] | None) -> bool:
    """Whether ``query`` uses an old name, so ``/`` should answer with a redirect.

    A canonical link (``?section=Models``, ``?section=Advanced&page=logs``) and a bare
    ``/`` are served as they are. Anything the 2.0.x panel accepted that is not
    canonical today is redirected so the address bar, bookmarks and copied links settle
    on the new form.
    """
    if not query:
        return False
    if "tab" in query or "model_repo" in query:
        return True  # the 2.0 grammar: the canonical form carries neither
    if not any(key in query for key in ("section", "page", "filter", "repo", "model")):
        return False
    wanted = canonical_query(resolve_location(query))
    got = {
        key: _clean(value)
        for key, value in query.items()
        if key in {"section", "page", "filter", *_PASSTHROUGH} and _clean(value)
    }
    return got != wanted


def section_url(tab: str, page: str | None = None) -> str:
    """A relative link to a tab (and Advanced page) for in-panel ``ui.link`` calls."""
    return canonical_url(Location(tab=tab, page=page if tab == ADVANCED else None))


__all__ = [
    "ADVANCED",
    "ADVANCED_PAGES",
    "ADVANCED_SLUGS",
    "DEFAULT_ADVANCED_PAGE",
    "LEGACY_ADVANCED_PAGE",
    "FILTER_ALL",
    "FILTER_DOWNLOADS",
    "FILTER_LOADED",
    "LEGACY_ROUTES",
    "MODELS",
    "MODEL_FILTERS",
    "STATUS",
    "TEST_CHAT",
    "TOP_TABS",
    "AdvancedPage",
    "Location",
    "advanced_page",
    "canonical_query",
    "canonical_url",
    "needs_redirect",
    "old_names",
    "resolve_location",
    "section_url",
]
