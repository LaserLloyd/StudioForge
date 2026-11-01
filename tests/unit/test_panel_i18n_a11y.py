"""Panel i18n (#13) and accessibility polish (#12): static assets, wiring and source checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from studioforge.config import Config
from studioforge.gui import nav
from studioforge.gui.app import _PANEL_HEAD_HTML, create_gui_app
from tests.unit.test_gui import _FakeState

GUI = Path(__file__).resolve().parents[2] / "src" / "studioforge" / "gui"
STATIC = GUI / "static"


def _catalogue(lang: str) -> dict[str, str]:
    return json.loads((STATIC / "locales" / f"{lang}.json").read_text(encoding="utf-8"))


def test_catalogues_have_identical_keys_and_placeholders() -> None:
    en, ja = _catalogue("en"), _catalogue("ja")
    assert en.keys() == ja.keys()
    assert en["$locale"] == "en" and ja["$locale"] == "ja"
    for key in en:
        assert re.findall(r"\{\w+\}", en[key]) == re.findall(r"\{\w+\}", ja[key]), key
        assert ja[key].strip(), key


def test_every_navigation_label_has_a_key() -> None:
    en = _catalogue("en")
    for name in nav.TOP_TABS:
        key = "tab." + name.lower().replace(" ", "_")
        assert en[key] == name
    for page in nav.ADVANCED_PAGES:
        assert en[f"adv.{page.slug}"] == page.label


def test_python_uses_only_catalogued_keys() -> None:
    en = _catalogue("en")
    used: set[str] = set()
    for path in GUI.rglob("*.py"):
        used |= set(re.findall(r'data-i18n(?:-aria)?=\\?"([\w.]+)"', path.read_text("utf-8")))
    used |= {"lang.label", "theme.label", "link.setup", "link.signout"}
    assert used - set(en) == set()


def test_loader_is_wired_into_the_head_and_assets_are_served(tmp_path: Path) -> None:
    assert "/sf-static/sf-i18n.js?v=" in _PANEL_HEAD_HTML
    assert "/sf-static/sf-a11y.css?v=" in _PANEL_HEAD_HTML
    cfg = Config(data_dir=tmp_path / "data")
    cfg.ensure_dirs()
    with TestClient(create_gui_app(cfg, api_state=_FakeState(cfg))) as client:
        js = client.get("/sf-static/sf-i18n.js")
        ja = client.get("/sf-static/locales/ja.json")
        helper = client.get("/sf-theme/locales/ja.json")
    assert js.status_code == 200 and "SFI18n" in js.text
    assert ja.status_code == 200 and ja.json()["$locale"] == "ja"
    assert helper.status_code == 200


def test_loader_sets_lang_persists_choice_and_translates() -> None:
    js = (STATIC / "sf-i18n.js").read_text("utf-8")
    assert "documentElement.setAttribute('lang'" in js
    assert "localStorage.setItem" in js and "studioforge.lang" in js
    assert "data-i18n" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_loader_runs_against_a_stub_dom(tmp_path: Path) -> None:
    harness = tmp_path / "run.js"
    harness.write_text(
        """
const fs = require('fs');
const root = process.argv[2];
const store = {};
const attrs = {};
const label = { textContent: 'Status' };
const tab = { classList: { contains: c => c === 'q-tab' }, querySelector: () => label,
  getAttribute: k => (k === 'data-i18n' ? 'tab.status' : null) };
const select = { id: 'sf-lang-select', value: 'en' };
const listeners = {};
global.window = global;
global.document = { readyState: 'complete', currentScript: { src: '/sf-static/sf-i18n.js?v=1' },
  documentElement: { setAttribute: (k, v) => { attrs[k] = v; } }, body: {},
  addEventListener: (e, f) => { listeners[e] = f; },
  getElementById: () => select,
  querySelectorAll: () => [tab] };
global.localStorage = { getItem: k => store[k] || null, setItem: (k, v) => { store[k] = v; } };
global.navigator = { language: 'en-US' };
global.requestAnimationFrame = f => f();
global.MutationObserver = class { observe() {} };
global.fetch = url => {
  const m = /locales\\/(\\w+)\\.json/.exec(url);
  const p = url.startsWith('/sf-theme/') ? root + '/theme/locales/' + m[1] + '.json'
                                         : root + '/static/locales/' + m[1] + '.json';
  return Promise.resolve({ ok: true, json: () => Promise.resolve(JSON.parse(fs.readFileSync(p))) });
};
require(root + '/static/sf-i18n.js');
(async () => {
  await SFI18n.set('ja');
  console.log(JSON.stringify({ lang: attrs.lang, stored: store['studioforge.lang'],
    label: label.textContent, select: select.value,
    helper: SFI18n.t('resetNamed', { name: 'X' }), missing: SFI18n.t('nope') }));
})();
""",
        encoding="utf-8",
    )
    out = subprocess.run(
        ["node", str(harness), str(GUI)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    )
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["lang"] == "ja" and got["stored"] == "ja"
    assert got["label"] == _catalogue("ja")["tab.status"]
    assert got["select"] == "ja"
    assert "X" in got["helper"] and got["missing"] is None


def test_helper_text_is_not_dimmed_below_contrast_floor() -> None:
    css = (STATIC / "sf-a11y.css").read_text("utf-8")
    for level in (50, 60, 70):
        assert f".opacity-{level}" in css
    for floor in re.findall(r"\{ opacity: ([\d.]+); \}", css):
        assert float(floor) >= 0.85
    assert 'data-contrast-profile="high"' in css


def test_status_text_is_announced_and_forms_submit_on_enter() -> None:
    tabs = GUI / "tabs"
    for name in ("status.py", "dashboard.py", "setup_wizard_page.py", "diagnostics.py"):
        assert 'aria-live="polite"' in (tabs / name).read_text("utf-8"), name
    assert 'aria-live="polite"' in (GUI / "app.py").read_text("utf-8")
    wizard = (tabs / "setup_wizard_page.py").read_text("utf-8")
    keys = (tabs / "users_keys.py").read_text("utf-8")
    assert wizard.count('"keydown.enter"') >= 3
    assert keys.count('"keydown.enter"') >= 3
