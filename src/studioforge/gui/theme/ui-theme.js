/* GENERATED theme runtime: ThemeForge 1.0.0, drop-in bundle (ui-theme/).
   https://github.com/LaserLloyd/ThemeForge . Do not edit this copy: update the whole
   folder (python ui-theme/update.py) and keep app styles in the app.
   body sha256: b5b5faa78e1ca015 */
/* ============================================================================
   ui-theme.js — the theme runtime · ThemeForge
   ----------------------------------------------------------------------------
   Load it as a BLOCKING script in <head>, before the theme stylesheets, so the
   theme is on the page before first paint:

     <script src="ui-theme.js"></script>

   Configuration, first match wins:
     1. window.UI_THEME_MANIFEST, if a page sets an object there before this
        script: themes (array or comma string), default, defaultDark,
        defaultLight, storageKey, families, legacy {key, map}, mirrorAttr,
        fontsHref.
     2. data-* attributes on this <script> tag. This is the usual way: the
        bundle (ui-theme/) carries no app settings of its own:
          data-themes="purple,midnight-gold,glacier,forest,paper,daylight"
          data-default="midnight-gold"      or "auto": follow the OS light/dark
          data-default-dark="midnight-gold"   preference until the user picks a
          data-default-light="daylight"       theme (both optional with auto)
          data-storage-key="myapp.theme"
          data-print-theme="auto"           auto (default): print a dark theme in
                                            a light one; or a slug; or none
          data-families="true"
          data-legacy-key="old-theme-key"
          data-legacy-map="dark:midnight-gold,light:daylight"
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
    {"slug": "forest", "name": "Forest", "family": "forest", "ground": "oled", "colorScheme": "dark", "themeColor": "#000000", "set": "core", "swatch": "#b5d59b", "fonts": ["Inter", "JetBrains Mono"]},
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
  function storeRemove(key) {
    try { global.localStorage.removeItem(key); } catch (e) { /* nothing stored */ }
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
    if (m && typeof m === 'object') {
      return {
        themes: Array.isArray(m.themes) ? m.themes : splitList(m.themes),
        'default': m['default'],
        defaultDark: m.defaultDark,
        defaultLight: m.defaultLight,
        storageKey: m.storageKey,
        families: m.families === true || m.families === 'true',
        legacy: m.legacy && m.legacy.key ? m.legacy : null,
        mirrorAttr: m.mirrorAttr || null,
        fontsHref: m.fontsHref || null,
        printTheme: m.printTheme || null
      };
    }
    var d = (script && script.dataset) || {};
    return {
      themes: splitList(d.themes),
      'default': d['default'],
      defaultDark: d.defaultDark,
      defaultLight: d.defaultLight,
      storageKey: d.storageKey,
      families: d.families === 'true',
      legacy: d.legacyKey ? { key: d.legacyKey, map: splitPairs(d.legacyMap) } : null,
      mirrorAttr: d.mirrorAttr || null,
      fontsHref: d.fontsHref || null,
      printTheme: d.printTheme || null
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
  function usable(slug) {
    return slug === BASE || enabled.indexOf(slug) >= 0;
  }

  // data-default="auto": the default follows the OS colour-scheme preference
  // (prefers-color-scheme) until the user picks a theme. The light and dark
  // defaults are data-default-light / data-default-dark, or else the first
  // enabled theme of that kind.
  var auto = raw['default'] === 'auto';
  var schemeQuery = auto && global.matchMedia ? global.matchMedia('(prefers-color-scheme: light)') : null;

  function autoDefault() {
    var light = !!(schemeQuery && schemeQuery.matches);
    var wanted = light ? raw.defaultLight : raw.defaultDark;
    if (wanted && usable(wanted)) return wanted;
    for (var i = 0; i < enabled.length; i++) {
      if ((find(enabled[i]).ground === 'light') === light) return enabled[i];
    }
    return enabled[0] || BASE;
  }

  var fallback = auto ? autoDefault()
    : usable(raw['default']) ? raw['default']
    : (enabled[0] || BASE);

  var config = {
    themes: enabled.slice(),
    'default': fallback,
    auto: auto,
    storageKey: raw.storageKey || 'ui-theme',
    families: !!raw.families,
    legacy: raw.legacy && raw.legacy.key ? { key: raw.legacy.key, map: raw.legacy.map || {} } : null,
    mirrorAttr: raw.mirrorAttr || null,
    fontsHref: raw.fontsHref || null,
    printTheme: raw.printTheme || 'auto'
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
  // Every mounted picker's sync function. A picker shows "Match system" or a
  // slug, which can change while the painted slug does not (set() of the
  // current theme, reset(), another tab), so pickers resync on every change
  // of choice, not only on announce().
  var pickerSyncs = [];
  function syncPickers() {
    for (var i = 0; i < pickerSyncs.length; i++) {
      try { pickerSyncs[i](); } catch (e) { /* a removed picker */ }
    }
  }

  function announce(theme, previous, printing) {
    var detail = { slug: theme.slug, theme: copy(theme), previous: previous, print: !!printing };
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
    syncPickers();
    if (previous !== theme.slug) announce(theme, previous);
    return theme.slug;
  }

  // Forget the user's choice and go back to the default (with
  // data-default="auto", the theme that matches the OS preference).
  function reset() {
    var previous = current;
    storeRemove(config.storageKey);
    if (auto) config['default'] = autoDefault();
    var theme = paint(config['default']);
    syncPickers();
    if (previous !== theme.slug) announce(theme, previous);
    return theme.slug;
  }

  function choosing() {
    return storeGet(config.storageKey) !== null;
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

  // Colour tokens as rgb()/rgba() strings: a token may be a var() alias or a
  // color-mix(), which canvas and chart libraries cannot read. A hidden probe
  // element lets the browser compute the final colour.
  var probe = null;
  function color(name) {
    if (!probe) {
      probe = doc.createElement('span');
      probe.setAttribute('aria-hidden', 'true');
      probe.style.display = 'none';
    }
    if (!probe.parentNode || !root.contains(probe)) (doc.body || root).appendChild(probe);
    probe.style.color = '';
    probe.style.color = 'var(' + tokenName(name) + ')';
    var value = global.getComputedStyle(probe).color;
    // A mixed colour computes to color(srgb r g b [/ a]); make it rgb()/rgba().
    var m = /^color\(srgb\s+([-\d.e]+)\s+([-\d.e]+)\s+([-\d.e]+)(?:\s*\/\s*([-\d.e]+))?\)$/.exec(value);
    if (!m) return value;
    var c = [m[1], m[2], m[3]].map(function (v) {
      return Math.round(Math.min(1, Math.max(0, parseFloat(v))) * 255);
    });
    var a = m[4] === undefined ? 1 : parseFloat(m[4]);
    return a < 1 ? 'rgba(' + c.join(', ') + ', ' + a + ')' : 'rgb(' + c.join(', ') + ')';
  }
  function colors(names) {
    var out = {};
    names.forEach(function (name) { out[name] = color(name); });
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
    if (auto) {
      var system = doc.createElement('option');
      system.value = '';
      system.textContent = opts.systemLabel || 'Match system';
      select.appendChild(system);
    }
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
    function sync() {
      select.value = auto && !choosing() ? '' : current;
    }
    sync();

    if (!select.__uiThemeMounted) {
      select.__uiThemeMounted = true;
      select.addEventListener('change', function () {
        if (select.value === '') reset(); else set(select.value);
      });
      pickerSyncs.push(sync);
    }
    return select;
  }

  function mountToggle(el) {
    if (el.__uiThemeMounted) return;
    el.__uiThemeMounted = true;
    el.addEventListener('click', function () { toggleFamily(); });
  }

  function mountIn(node) {
    if (node.nodeType !== 1) return;
    if (node.hasAttribute('data-ui-theme-picker') && !node.__uiThemeMounted) mountPicker(node);
    if (node.hasAttribute('data-ui-theme-toggle')) mountToggle(node);
    var pickers = node.querySelectorAll('[data-ui-theme-picker]');
    for (var i = 0; i < pickers.length; i++) {
      if (!pickers[i].__uiThemeMounted) mountPicker(pickers[i]);
    }
    var toggles = node.querySelectorAll('[data-ui-theme-toggle]');
    for (var j = 0; j < toggles.length; j++) mountToggle(toggles[j]);
  }

  // Mount what is in the page now, then anything a framework renders later.
  function autoMount() {
    mountIn(doc.documentElement);
    if (!global.MutationObserver) return;
    new global.MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var added = records[i].addedNodes;
        for (var j = 0; j < added.length; j++) mountIn(added[j]);
      }
    }).observe(doc.documentElement, { childList: true, subtree: true });
  }

  /* ---- printing ----
     Paper is light, so a dark theme prints in a light one: the family's
     light partner, else the first light theme the app offers, else Daylight
     (every theme is in ui-theme.css). The saved choice is never touched. */
  function printSlug() {
    var want = config.printTheme;
    if (want === 'none') return null;
    if (want !== 'auto') return find(want) ? want : null;
    var theme = find(current);
    if (!theme || theme.ground === 'light') return null;
    var other = partner(current);
    if (other) return other;
    for (var i = 0; i < enabled.length; i++) {
      if (find(enabled[i]).ground === 'light') return enabled[i];
    }
    return find('daylight') ? 'daylight' : null;
  }

  var screenSlug = null;
  global.addEventListener('beforeprint', function () {
    var slug = printSlug();
    if (!slug || slug === current) return;
    screenSlug = current;
    // Announced (detail.print is true) so charts redraw for paper; nothing is
    // stored and pickers keep showing the screen choice.
    announce(paint(slug), screenSlug, true);
  });
  global.addEventListener('afterprint', function () {
    if (screenSlug === null) return;
    var back = screenSlug;
    var printed = current;
    screenSlug = null;
    announce(paint(back), printed, true);
  });

  /* ---- boot ---- */

  paint(resolve(stored()));

  global.addEventListener('storage', function (e) {
    // key null: another document called localStorage.clear().
    if (e.key !== config.storageKey && e.key !== null) return;
    var value = e.key === null ? null : e.newValue;
    var previous = current;
    if (auto && value === null) config['default'] = autoDefault();
    var theme = paint(resolve(value));
    syncPickers();
    if (previous !== theme.slug) announce(theme, previous);
  });

  // With data-default="auto", follow the OS while the user has not chosen.
  if (schemeQuery) {
    var onScheme = function () {
      config['default'] = autoDefault();
      if (choosing()) return;
      var previous = current;
      var theme = paint(config['default']);
      syncPickers();
      if (previous !== theme.slug) announce(theme, previous);
    };
    if (schemeQuery.addEventListener) schemeQuery.addEventListener('change', onScheme);
    else if (schemeQuery.addListener) schemeQuery.addListener(onScheme);
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', autoMount);
  else autoMount();

  /* ---- API ----
     UITheme.current()              -> active slug
     UITheme.theme([slug])          -> metadata copy (active theme by default)
     UITheme.list()                 -> enabled themes, picker order
     UITheme.set(slug)              -> apply + persist; unknown/disabled -> default
     UITheme.reset()                -> forget the choice; back to the default
                                       (with data-default="auto": the OS preference)
     UITheme.partner([slug])        -> the other ground in the same family, or null
     UITheme.toggleFamily()         -> dark <-> light within the family (manifest families: true)
     UITheme.onChange(fn)           -> fn({slug, theme, previous, print}); returns an unsubscribe
                                       (print is true for the swap around printing)
     UITheme.token(name[, el])      -> computed token value, e.g. token('--canvas-mask')
     UITheme.tokens(names[, el])    -> {name: value}
     UITheme.color(name)            -> the token as an rgb()/rgba() string (for canvas)
     UITheme.colors(names)          -> {name: rgb()} for several tokens
     UITheme.mountPicker(target[, {label, coreLabel, optInLabel, systemLabel}]) -> the <select>
     document 'ui-theme-change' event carries the same detail as onChange. */
  global.UITheme = {
    version: /*@version*/'1.0.0',
    config: config,
    current: function () { return current; },
    theme: function (slug) { var t = find(slug || current); return t ? copy(t) : null; },
    list: list,
    set: set,
    reset: reset,
    partner: partner,
    toggleFamily: toggleFamily,
    onChange: onChange,
    token: token,
    tokens: tokens,
    color: color,
    colors: colors,
    mountPicker: mountPicker
  };
})(window);
