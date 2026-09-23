"""Dashboard actions draw on the page, not on the card the refresh deleted (D69 §17).

NiceGUI runs a click handler inside the slot of the button that fired it and
finds the client for ``ui.notify`` through that slot. The Loaded models panel
rebuilds its cards on a timer, so after an awaited unload or reload the slot's
parent was gone and the toast raised "The parent element this slot belongs to
has been deleted" (09-13, 09-15, 09-16, 09-22 twice). These tests stand in for
NiceGUI with the one behaviour that matters: the client is reachable through
the button's slot only while the card exists, and through the page's content
slot while the page is entered.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from studioforge.config import Config
from studioforge.gui import tabs
from studioforge.gui.tabs import GuiContext, dashboard


class _Client:
    def __init__(self) -> None:
        self.is_deleted = False
        self.depth = 0
        self.ip = "127.0.0.1"

    def __enter__(self) -> _Client:
        self.depth += 1
        return self

    def __exit__(self, *_exc: object) -> None:
        self.depth -= 1


class _Context:
    def __init__(self, client: _Client) -> None:
        self._client = client
        self.card_deleted = False

    @property
    def client(self) -> _Client:
        if self.card_deleted and self._client.depth == 0:
            raise RuntimeError("The parent element this slot belongs to has been deleted.")
        return self._client


class _Notification:
    def __init__(self) -> None:
        self.message = ""

    def dismiss(self) -> None:
        return None

    def update(self) -> None:
        return None


class _Ui:
    def __init__(self) -> None:
        self.client = _Client()
        self.context = _Context(self.client)
        self.notes: list[tuple[str, dict[str, Any], int]] = []

    def notify(self, message: Any, **kwargs: Any) -> None:
        client = self.context.client  # exactly how NiceGUI's notify finds its client
        self.notes.append((str(message), kwargs, client.depth))

    def notification(self, *_a: Any, **_kw: Any) -> _Notification:
        _ = self.context.client
        return _Notification()


@pytest.fixture()
def fake_ui(monkeypatch: pytest.MonkeyPatch) -> _Ui:
    fake = _Ui()
    monkeypatch.setattr(dashboard, "ui", fake)
    monkeypatch.setattr(tabs, "ui", fake)
    return fake


def _ctx(manager: Any) -> GuiContext:
    return GuiContext(
        config=Config(data_dir="/tmp/sf-dashboard"), api_state=SimpleNamespace(manager=manager)
    )


class _Refresh:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


async def test_unload_toasts_on_the_page_after_the_refresh_deleted_its_card(fake_ui: _Ui) -> None:
    class Manager:
        async def unload(self, model_id: str, *, force: bool) -> None:
            fake_ui.context.card_deleted = True  # the 2 s repaint ran during the await

    refresh = _Refresh()
    await dashboard._unload_one(_ctx(Manager()), "a/model", refresh)

    assert fake_ui.notes == [("a/model unloaded", {"type": "positive"}, 1)]
    assert refresh.calls == 1
    assert fake_ui.client.depth == 0, "the page context was left entered"


async def test_restart_toasts_on_the_page_too(fake_ui: _Ui) -> None:
    class Manager:
        async def load(self, model_id: str, **_kw: Any) -> Any:
            fake_ui.context.card_deleted = True
            return SimpleNamespace(port=18123)

    await dashboard._restart_model(_ctx(Manager()), "a/model", _Refresh())

    assert fake_ui.notes == [("a/model restarted on port 18123", {"type": "positive"}, 1)]


async def test_a_failure_after_the_card_is_gone_is_still_a_red_toast(fake_ui: _Ui) -> None:
    class Manager:
        async def unload(self, model_id: str, *, force: bool) -> None:
            fake_ui.context.card_deleted = True
            raise RuntimeError("child did not exit")

    refresh = _Refresh()
    await dashboard._unload_one(_ctx(Manager()), "b/model", refresh)

    (message, kwargs, depth) = fake_ui.notes[0]
    assert message == "unload: RuntimeError: child did not exit"
    assert kwargs["type"] == "negative" and depth == 1
    assert refresh.calls == 0


async def test_a_page_whose_browser_went_away_is_not_drawn_into(fake_ui: _Ui) -> None:
    class Manager:
        async def unload(self, model_id: str, *, force: bool) -> None:
            fake_ui.context.card_deleted = True
            fake_ui.client.is_deleted = True

    refresh = _Refresh()
    await dashboard._unload_one(_ctx(Manager()), "c/model", refresh)

    assert fake_ui.notes == []
    assert refresh.calls == 1  # the panel's own liveness check decides the repaint


async def test_a_pin_toggle_toasts_on_the_page(fake_ui: _Ui) -> None:
    class Manager:
        def set_pinned(self, model_id: str, pinned: bool) -> Any:
            fake_ui.context.card_deleted = True
            return SimpleNamespace(id=model_id, settings=SimpleNamespace(pinned=pinned)), None

    await dashboard._toggle_pin(_ctx(Manager()), "d/model", False, _Refresh())

    assert fake_ui.notes == [("d/model pinned", {"type": "positive"}, 1)]


async def test_a_page_that_cannot_be_entered_never_blocks_the_action(
    fake_ui: _Ui, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_enter(self: _Client) -> _Client:
        raise RuntimeError("content slot unavailable")

    monkeypatch.setattr(_Client, "__enter__", broken_enter)
    unloaded: list[str] = []

    class Manager:
        async def unload(self, model_id: str, *, force: bool) -> None:
            unloaded.append(model_id)

    await dashboard._unload_one(_ctx(Manager()), "e/model", _Refresh())

    assert unloaded == ["e/model"]
    assert fake_ui.notes == [("e/model unloaded", {"type": "positive"}, 0)]


def test_every_awaiting_dashboard_action_captures_the_page() -> None:
    """Static guard: each handler that awaits and then draws binds the page first."""
    import inspect

    for func in (
        dashboard._reclaim_orphans,
        dashboard._toggle_pin,
        dashboard._unload_one,
        dashboard._restart_model,
        dashboard._unload_all_dialog,
        dashboard._restart_engines,
        dashboard._restart_server_dialog,
    ):
        source = inspect.getsource(func)
        assert "_Page.capture()" in source, f"{func.__name__} draws after an await unbound"
        assert "ui.notify(" not in source, f"{func.__name__} still toasts through the slot"
