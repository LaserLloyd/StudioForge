/* GENERATED adapter tailwind-v3.preset.js: ThemeForge 1.0.0, drop-in bundle (ui-theme/).
   https://github.com/LaserLloyd/ThemeForge . Do not edit this copy: update the whole
   folder (python ui-theme/update.py) and keep app styles in the app.
   body sha256: dc1a09a021606711 */
/* ---- adapter: tailwind-v3.preset.js — the contract tokens for Tailwind CSS v3 ----
   In tailwind.config.js, add it to presets (path relative to that file):

     presets: [require('./static/ui-theme/adapters/tailwind-v3.preset.js')]

   Same names as the v4 adapter (adapters/tailwind.css): bg-surface-2, text-fg,
   text-fg-muted, bg-accent, text-on-accent, border-line, ... Each colour is a
   var() reference, so the active theme decides the value. Load the ui-theme
   files in the page as usual; this file only names the tokens for Tailwind.
   Opacity modifiers (bg-accent/50) do not apply to var() colours in v3: use the
   -subtle tokens instead. Tailwind v4 projects use adapters/tailwind.css. */
(function () {
  'use strict';
  var v = function (name) { return 'var(--' + name + ')'; };
  var colors = {
    'surface-void': v('surface-void'),
    'surface-0': v('surface-0'), 'surface-1': v('surface-1'), 'surface-2': v('surface-2'),
    'surface-3': v('surface-3'), 'surface-4': v('surface-4'),
    'surface-sunken': v('surface-sunken'), 'surface-overlay': v('surface-overlay'),
    fg: v('text-primary'), 'fg-muted': v('text-secondary'), 'fg-subtle': v('text-tertiary'),
    'fg-disabled': v('text-disabled'), 'fg-inverse': v('text-inverse'),
    link: v('text-link'), 'link-hover': v('text-link-hover'),
    accent: v('accent'), 'accent-hover': v('accent-hover'), 'accent-pressed': v('accent-pressed'),
    'accent-subtle': v('accent-subtle'), 'on-accent': v('on-accent'),
    cta: v('cta'), 'on-cta': v('on-cta'), highlight: v('highlight'), 'on-highlight': v('on-highlight'),
    selected: v('selected'), 'on-selected': v('on-selected'),
    line: v('border'), 'line-subtle': v('border-subtle'), 'line-strong': v('border-strong'),
    divider: v('divider'), focus: v('focus-ring')
  };
  ['success', 'warning', 'danger', 'info'].forEach(function (s) {
    colors[s] = v(s);
    colors[s + '-text'] = v(s + '-text');
    colors[s + '-subtle'] = v(s + '-subtle');
    colors[s + '-border'] = v(s + '-border');
    colors['on-' + s] = v('on-' + s);
  });
  for (var i = 1; i <= 12; i++) colors['cat-' + i] = v('cat-' + i);

  var preset = {
    theme: {
      extend: {
        colors: colors,
        borderRadius: { xs: v('radius-xs'), sm: v('radius-sm'), md: v('radius-md'), lg: v('radius-lg'), xl: v('radius-xl') },
        fontFamily: { sans: v('font-sans'), mono: v('font-mono'), display: v('font-display') },
        boxShadow: { 1: v('shadow-1'), 2: v('shadow-2'), 3: v('shadow-3'), 4: v('shadow-4') }
      }
    }
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = preset;
})();
