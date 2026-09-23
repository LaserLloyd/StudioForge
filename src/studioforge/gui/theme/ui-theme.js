/* GENERATED theme runtime: unifyingTheme V26-09-16 drop-in bundle (ui-theme/),
   the same in every app. Do not edit this copy: change the package's src/
   or adapters/, run `python tools/sync_theme.py build`, then copy ui-theme/
   into each app (or run `python tools/sync_theme.py install`).
   body sha256: 4cd32791614780f3 */
/* ============================================================================
   ui-theme.js — portable theme runtime · unifyingTheme V26-09-16
   ----------------------------------------------------------------------------
   Load it as a BLOCKING script in <head>, before the theme stylesheets, so the
   theme is on the page before first paint:

     <script src="ui-theme.js"></script>

   Configuration, first match wins:
     1. window.UI_THEME_MANIFEST, if a page sets one before this script (the
        old per-app builds did; the drop-in bundle does not).
     2. data-* attributes on this <script> tag. This is how every app
        configures the drop-in bundle (ui-theme/), which carries no app
        settings of its own:
          data-themes="purple,midnight-gold,glacier,forest,paper,daylight"
          data-default="purple"
          data-storage-key="theme"
          data-families="true"
          data-legacy-key="cc-skin"
          data-legacy-map="obsidian:midnight-gold,neon:purple"
          data-mirror-attr="data-skin"
          data-fonts-href="fonts/"

   Before first paint it:
     - reads the stored slug, migrating a configured legacy key once;
     - validates the slug against the enabled list, falling back to the default;
     - sets <html data-palette>, or removes it for Purple (the :root base);
     - sets the derived <html data-theme>: amoled | dark | light;
     - updates <meta name="theme-color"> (created if missing) and any
       <meta name="color-scheme"> the page already has.

   Afterwards it exposes window.UITheme (API at the end of this file) and
   mounts every <select data-ui-theme-picker> and [data-ui-theme-toggle] it
   finds once the DOM is ready. Component code never names a slug; it reads
   tokens.
   ============================================================================ */
(function (global) {
  'use strict';

  var doc = global.document;
  var root = doc.documentElement;
  var script = doc.currentScript;

  /* Registry: generated from src/themes.json by `sync_theme.py build`. */
  var REGISTRY = /*@registry*/[
    {"slug": "purple", "name": "Purple", "family": "purple", "ground": "dark", "colorScheme": "dark", "themeColor": "#0e0e1b", "set": "core", "swatch": "#7c3aed", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "midnight-gold", "name": "Midnight Gold", "family": "midnight-gold", "ground": "oled", "colorScheme": "dark", "themeColor": "#000000", "set": "core", "swatch": "#e3bf5a", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "glacier", "name": "Glacier", "family": "glacier", "ground": "oled", "colorScheme": "dark", "themeColor": "#000000", "set": "core", "swatch": "#7fe0f2", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "forest", "name": "Forest", "family": "forest", "ground": "dark", "colorScheme": "dark", "themeColor": "#0a0806", "set": "core", "swatch": "#d68a42", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "paper", "name": "Paper", "family": "paper", "ground": "light", "colorScheme": "light", "themeColor": "#e6d2a8", "set": "core", "swatch": "#a3241c", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "daylight", "name": "Daylight", "family": "daylight", "ground": "light", "colorScheme": "light", "themeColor": "#ffffff", "set": "core", "swatch": "#2c56c9", "fonts": ["Inter", "JetBrains Mono"]},
    {"slug": "electric-yellow", "name": "Electric Yellow", "family": "electric-yellow", "ground": "dark", "colorScheme": "dark", "themeColor": "#0a0a0b", "set": "opt-in", "swatch": "#e8ff00", "fonts": ["Archivo", "JetBrains Mono"]},
    {"slug": "laserlloyd", "name": "LaserLloyd", "family": "laserlloyd", "ground": "dark", "colorScheme": "dark", "themeColor": "#080a0d", "set": "opt-in", "swatch": "#2ea8ff", "fonts": ["Inter", "Space Grotesk"]},
    {"slug": "laserlloyd-light", "name": "LaserLloyd Light", "family": "laserlloyd", "ground": "light", "colorScheme": "light", "themeColor": "#f2f5f9", "set": "opt-in", "swatch": "#0b78d0", "fonts": ["Inter", "Space Grotesk"]},
    {"slug": "night-red", "name": "Night Red", "family": "night-red", "ground": "oled", "colorScheme": "dark", "themeColor": "#000000", "set": "opt-in", "swatch": "#ff0000", "fonts": ["Inter", "JetBrains Mono"]}
  ]/*@end-registry*/;

  var BASE = 'purple';
  var DATA_THEME = { oled: 'amoled', dark: 'dark', light: 'light' };
  var EVENT = 'ui-theme-change';

  function find(slug) {
    for (var i = 0; i < REGISTRY.length; i++) {
      if (REGISTRY[i].slug === slug) return REGISTRY[i];
    }
    return null;
  }

  function copy(theme) {
    var out = {};
    for (var k in theme) {
      if (Object.prototype.hasOwnProperty.call(theme, k)) {
        out[k] = Array.isArray(theme[k]) ? theme[k].slice() : theme[k];
      }
    }
    return out;
  }

  // Storage can throw (private mode, blocked site data); a theme must still paint.
  function storeGet(key) {
    try { return global.localStorage.getItem(key); } catch (e) { return null; }
  }
  function storeSet(key, value) {
    try { global.localStorage.setItem(key, value); } catch (e) { /* not persisted */ }
  }

  function splitList(text) {
    return String(text || '').split(',')
      .map(function (s) { return s.trim(); })
      .filter(Boolean);
  }
  function splitPairs(text) {
    var out = {};
    splitList(text).forEach(function (pair) {
      var i = pair.indexOf(':');
      if (i > 0) out[pair.slice(0, i).trim()] = pair.slice(i + 1).trim();
    });
    return out;
  }

  function readConfig() {
    var m = global.UI_THEME_MANIFEST;
    if (m && typeof m === 'object') return m;
    var d = (script && script.dataset) || {};
    return {
      themes: splitList(d.themes),
      'default': d['default'],
      storageKey: d.storageKey,
      families: d.families === 'true',
      legacy: d.legacyKey ? { key: d.legacyKey, map: splitPairs(d.legacyMap) } : null,
      mirrorAttr: d.mirrorAttr || null,
      fontsHref: d.fontsHref || null
    };
  }

  /* ---- configuration ---- */

  var raw = readConfig();
  var requested = raw.themes && raw.themes.length
    ? raw.themes
    : REGISTRY.filter(function (t) { return t.set === 'core'; }).map(function (t) { return t.slug; });
  var enabled = [];
  requested.forEach(function (slug) {
    if (find(slug) && enabled.indexOf(slug) < 0) enabled.push(slug);
  });
  // Purple is always in the CSS (it is :root), so it is a valid default even
  // when an app leaves it out of its picker.
  var fallback = raw['default'] === BASE || enabled.indexOf(raw['default']) >= 0
    ? raw['default']
    : (enabled[0] || BASE);

  var config = {
    themes: enabled.slice(),
    'default': fallback,
    storageKey: raw.storageKey || 'ui-theme',
    families: !!raw.families,
    legacy: raw.legacy && raw.legacy.key ? { key: raw.legacy.key, map: raw.legacy.map || {} } : null,
    mirrorAttr: raw.mirrorAttr || null,
    fontsHref: raw.fontsHref || null
  };

  function allowed(slug) {
    return slug === config['default'] || enabled.indexOf(slug) >= 0;
  }
  function resolve(slug) {
    return allowed(slug) ? slug : config['default'];
  }

  function stored() {
    var value = storeGet(config.storageKey);
    if (value === null && config.legacy) {
      var old = storeGet(config.legacy.key);
      var mapped = old !== null ? config.legacy.map[old] : undefined;
      if (mapped && allowed(mapped)) {
        // One-time migration, and only onto a theme this app offers: a
        // mapping to a disabled theme would be stored and then fall back to
        // the default on every load. The legacy key is left in place so an
        // older build of the app still finds it.
        value = mapped;
        storeSet(config.storageKey, mapped);
      }
    }
    return value;
  }

  /* ---- painting ---- */

  function meta(name, create) {
    var el = doc.querySelector('meta[name="' + name + '"]');
    if (!el && create) {
      el = doc.createElement('meta');
      el.setAttribute('name', name);
      (doc.head || root).appendChild(el);
    }
    return el;
  }

  var loadedFonts = {};
  function loadFonts(theme) {
    // Optional: only when the app self-hosts font CSS (one file per family,
    // e.g. fonts/space-grotesk.css). Loads fonts for the active theme only.
    if (!config.fontsHref || !theme.fonts) return;
    var base = config.fontsHref.replace(/\/?$/, '/');
    theme.fonts.forEach(function (family) {
      var file = family.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.css';
      if (loadedFonts[file]) return;
      loadedFonts[file] = true;
      var link = doc.createElement('link');
      link.rel = 'stylesheet';
      link.href = base + file;
      (doc.head || root).appendChild(link);
    });
  }

  var current = null;

  function paint(slug) {
    var theme = find(slug) || find(BASE);
    if (theme.slug === BASE) root.removeAttribute('data-palette');
    else root.setAttribute('data-palette', theme.slug);
    root.setAttribute('data-theme', DATA_THEME[theme.ground] || 'dark');
    if (config.mirrorAttr) root.setAttribute(config.mirrorAttr, theme.slug);
    var themeColor = meta('theme-color', true);
    if (themeColor) themeColor.setAttribute('content', theme.themeColor);
    var scheme = meta('color-scheme', false);
    if (scheme) scheme.setAttribute('content', theme.colorScheme);
    loadFonts(theme);
    current = theme.slug;
    return theme;
  }

  var listeners = [];

  function announce(theme, previous) {
    var detail = { slug: theme.slug, theme: copy(theme), previous: previous };
    listeners.slice().forEach(function (fn) {
      try { fn(detail); } catch (e) { if (global.console) global.console.error(e); }
    });
    try {
      doc.dispatchEvent(new global.CustomEvent(EVENT, { detail: detail }));
    } catch (e) { /* very old engines: listeners above still ran */ }
  }

  /* ---- public operations ---- */

  function set(slug) {
    var previous = current;
    var theme = paint(resolve(slug));
    storeSet(config.storageKey, theme.slug);
    if (previous !== theme.slug) announce(theme, previous);
    return theme.slug;
  }

  function list() {
    var slugs = enabled.slice();
    if (slugs.indexOf(config['default']) < 0) slugs.unshift(config['default']);
    return slugs.map(function (slug) { return copy(find(slug)); });
  }

  function partner(slug) {
    var theme = find(slug || current);
    if (!theme) return null;
    var light = theme.ground === 'light';
    for (var i = 0; i < enabled.length; i++) {
      var other = find(enabled[i]);
      if (other.slug !== theme.slug && other.family === theme.family &&
          (other.ground === 'light') !== light) {
        return other.slug;
      }
    }
    return null;
  }

  function toggleFamily() {
    if (!config.families) return null;
    var other = partner(current);
    return other ? set(other) : null;
  }

  function onChange(fn) {
    listeners.push(fn);
    return function () {
      var i = listeners.indexOf(fn);
      if (i >= 0) listeners.splice(i, 1);
    };
  }

  function tokenName(name) {
    return name.indexOf('--') === 0 ? name : '--' + name;
  }
  // For canvas and chart code, which cannot use var(): read computed tokens.
  // Gradient-valued tokens (e.g. --scrim-media) come back as gradient text.
  function token(name, el) {
    return global.getComputedStyle(el || root).getPropertyValue(tokenName(name)).trim();
  }
  function tokens(names, el) {
    var style = global.getComputedStyle(el || root);
    var out = {};
    names.forEach(function (name) {
      out[name] = style.getPropertyValue(tokenName(name)).trim();
    });
    return out;
  }

  // Fills a <select> (or appends one to a container) with the enabled themes
  // and keeps it in step with every theme change, including other tabs.
  function mountPicker(target, options) {
    var opts = options || {};
    var host = typeof target === 'string' ? doc.querySelector(target) : target;
    if (!host) return null;
    var select = host.tagName === 'SELECT' ? host : host.appendChild(doc.createElement('select'));
    var labelled = select.getAttribute('aria-label') || select.getAttribute('aria-labelledby') ||
      (select.labels && select.labels.length);
    if (!labelled) select.setAttribute('aria-label', opts.label || 'Theme');

    while (select.firstChild) select.removeChild(select.firstChild);
    var themes = list();
    var grouped = themes.some(function (t) { return t.set !== 'core'; });
    var groups = {};
    themes.forEach(function (t) {
      var parent = select;
      if (grouped) {
        var label = t.set === 'core' ? (opts.coreLabel || 'Core') : (opts.optInLabel || 'Opt-in');
        if (!groups[label]) {
          groups[label] = doc.createElement('optgroup');
          groups[label].setAttribute('label', label);
          select.appendChild(groups[label]);
        }
        parent = groups[label];
      }
      var option = doc.createElement('option');
      option.value = t.slug;
      option.textContent = t.name;
      parent.appendChild(option);
    });
    select.value = current;

    if (!select.__uiThemeMounted) {
      select.__uiThemeMounted = true;
      select.addEventListener('change', function () { set(select.value); });
      onChange(function (detail) {
        if (select.value !== detail.slug) select.value = detail.slug;
      });
    }
    return select;
  }

  function mountToggle(el) {
    if (el.__uiThemeMounted) return;
    el.__uiThemeMounted = true;
    el.addEventListener('click', function () { toggleFamily(); });
  }

  function autoMount() {
    var pickers = doc.querySelectorAll('[data-ui-theme-picker]');
    for (var i = 0; i < pickers.length; i++) mountPicker(pickers[i]);
    var toggles = doc.querySelectorAll('[data-ui-theme-toggle]');
    for (var j = 0; j < toggles.length; j++) mountToggle(toggles[j]);
  }

  /* ---- boot ---- */

  paint(resolve(stored()));

  global.addEventListener('storage', function (e) {
    if (e.key !== config.storageKey) return;
    var previous = current;
    var theme = paint(resolve(e.newValue));
    if (previous !== theme.slug) announce(theme, previous);
  });

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', autoMount);
  else autoMount();

  /* ---- API ----
     UITheme.current()              -> active slug
     UITheme.theme([slug])          -> metadata copy (active theme by default)
     UITheme.list()                 -> enabled themes, picker order
     UITheme.set(slug)              -> apply + persist; unknown/disabled -> default
     UITheme.partner([slug])        -> the other ground in the same family, or null
     UITheme.toggleFamily()         -> dark <-> light within the family (manifest families: true)
     UITheme.onChange(fn)           -> fn({slug, theme, previous}); returns an unsubscribe
     UITheme.token(name[, el])      -> computed token value, e.g. token('--canvas-mask')
     UITheme.tokens(names[, el])    -> {name: value}
     UITheme.mountPicker(target[, {label, coreLabel, optInLabel}]) -> the <select>
     document 'ui-theme-change' event carries the same detail as onChange. */
  global.UITheme = {
    version: 'V26-09-16',
    config: config,
    current: function () { return current; },
    theme: function (slug) { var t = find(slug || current); return t ? copy(t) : null; },
    list: list,
    set: set,
    partner: partner,
    toggleFamily: toggleFamily,
    onChange: onChange,
    token: token,
    tokens: tokens,
    mountPicker: mountPicker
  };
})(window);
