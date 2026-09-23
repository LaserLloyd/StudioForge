/* GENERATED adapter quasar.js: unifyingTheme V26-09-16 drop-in bundle (ui-theme/),
   the same in every app. Do not edit this copy: change the package's src/
   or adapters/, run `python tools/sync_theme.py build`, then copy ui-theme/
   into each app (or run `python tools/sync_theme.py install`).
   body sha256: a558aa540d7d0947 */

/* ---- adapter: quasar.js — Quasar / NiceGUI ----
   Keeps Quasar's dark mode in step with the theme's ground, because Quasar
   switches its --dark component variants from Dark.isActive, not from CSS.
   NiceGUI calls Quasar.Dark.set() while it creates the Vue app (and whenever a
   ui.dark_mode element updates), so body class changes are watched and the
   theme's ground is re-asserted. Light themes turn dark mode off. */
(function (global) {
  'use strict';
  var T = global.UITheme;
  if (!T) return;
  var doc = global.document;

  function wantDark() {
    var theme = T.theme();
    return !theme || theme.ground !== 'light';
  }

  function sync() {
    var Q = global.Quasar;
    if (!Q || !Q.Dark || typeof Q.Dark.set !== 'function') return;
    var want = wantDark();
    if (Q.Dark.isActive !== want) Q.Dark.set(want);
  }

  function watch() {
    if (!doc.body || !global.MutationObserver) return;
    new global.MutationObserver(sync).observe(doc.body, { attributes: true, attributeFilter: ['class'] });
    sync();
  }

  T.onChange(sync);
  if (doc.body) watch();
  else doc.addEventListener('DOMContentLoaded', watch);
  global.addEventListener('load', sync);
})(window);
