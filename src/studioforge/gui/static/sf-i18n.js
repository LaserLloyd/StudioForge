/* StudioForge panel i18n (minimal). Loads static/locales/<lang>.json (and the ForgeUI helper
   strings from /sf-theme/locales/<lang>.json), translates elements carrying data-i18n, sets
   <html lang>, and remembers the choice in localStorage. English is the source text in the
   page; a key missing from a catalogue leaves that element untouched. Coverage is incremental:
   the shell chrome and navigation are wired, tab bodies are still English.

   data-i18n="key"        replace the element's text (Quasar tabs and fields: their label node)
   data-i18n-aria="key"   set aria-label
   SFI18n.t(key, {name})  look a string up; SFI18n.set('ja'); SFI18n.get() */
(function (global) {
  'use strict';
  var KEY = 'studioforge.lang';
  var SUPPORTED = ['en', 'ja'];
  var doc = global.document;
  var scriptSrc = (doc.currentScript && doc.currentScript.src) || '';
  var base = scriptSrc.replace(/[^/]*(\?.*)?$/, '');
  var themeBase = '/sf-theme/locales/';
  var catalogue = {};
  var lang = 'en';
  var observer = null;
  var scheduled = false;

  function store(value) {
    try { global.localStorage.setItem(KEY, value); } catch (e) { /* storage may be blocked */ }
  }
  function saved() {
    try { return global.localStorage.getItem(KEY); } catch (e) { return null; }
  }
  function normalise(value) {
    var l = String(value || '').toLowerCase().replace(/_/g, '-').split('-')[0];
    return SUPPORTED.indexOf(l) >= 0 ? l : 'en';
  }
  function initial() {
    var s = saved();
    if (s) return normalise(s);
    var nav = global.navigator;
    return normalise((nav && (nav.language || (nav.languages || [])[0])) || 'en');
  }

  function t(key, params) {
    var text = Object.prototype.hasOwnProperty.call(catalogue, key) ? catalogue[key] : null;
    if (text === null) return null;
    return String(text).replace(/\{(\w+)\}/g, function (m, name) {
      return params && name in params ? String(params[name]) : m;
    });
  }

  function textTarget(el) {
    if (el.classList.contains('q-tab')) return el.querySelector('.q-tab__label') || el;
    if (el.classList.contains('q-field')) return el.querySelector('.q-field__label') || el;
    return el;
  }

  function apply(root) {
    root = root || doc;
    var nodes = root.querySelectorAll ? root.querySelectorAll('[data-i18n],[data-i18n-aria]') : [];
    Array.prototype.forEach.call(nodes, function (el) {
      var k = el.getAttribute('data-i18n');
      if (k) {
        var txt = t(k);
        var target = textTarget(el);
        if (txt !== null && target.textContent !== txt) target.textContent = txt;
      }
      var a = el.getAttribute('data-i18n-aria');
      if (a) {
        var label = t(a);
        if (label !== null && el.getAttribute('aria-label') !== label) el.setAttribute('aria-label', label);
      }
    });
    var select = doc.getElementById('sf-lang-select');
    if (select && select.value !== lang) select.value = lang;
  }

  function schedule() {
    if (scheduled) return;
    scheduled = true;
    global.requestAnimationFrame(function () { scheduled = false; apply(doc); });
  }

  function fetchJson(url) {
    return global.fetch(url, { credentials: 'same-origin' }).then(function (r) {
      return r.ok ? r.json() : {};
    }).catch(function () { return {}; });
  }

  function load(next) {
    next = normalise(next);
    var urls = [base + 'locales/' + next + '.json', themeBase + next + '.json'];
    return Promise.all(urls.map(fetchJson)).then(function (parts) {
      var merged = {};
      parts.forEach(function (p) { Object.keys(p).forEach(function (k) { merged[k] = p[k]; }); });
      catalogue = merged;
      lang = next;
      doc.documentElement.setAttribute('lang', next);
      apply(doc);
      return next;
    });
  }

  function set(next) {
    next = normalise(next);
    store(next);
    return load(next);
  }

  doc.addEventListener('change', function (event) {
    var el = event.target;
    if (el && el.id === 'sf-lang-select') set(el.value);
  });

  function start() {
    load(initial());
    // NiceGUI builds the page after load and rebuilds parts of it; translate whatever arrives.
    observer = new global.MutationObserver(schedule);
    observer.observe(doc.body, { childList: true, subtree: true });
  }
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', start);
  else start();

  global.SFI18n = {
    t: t, set: set, get: function () { return lang; }, apply: apply,
    supported: SUPPORTED.slice(), storageKey: KEY
  };
})(window);
