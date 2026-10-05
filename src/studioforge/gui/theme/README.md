# ui-theme 1.0.0

A drop-in theme for web apps: ten colour themes, a theme picker, design tokens, a base
element layer, a text standard and ready-made component classes. Plain CSS and two small
scripts; no build step, no dependencies. Source and full docs:
https://github.com/LaserLloyd/ThemeForge

**Do not edit anything in this folder.** It is replaced whole on every update. Keep your
app's own styles in your app's own CSS files.

## Files

| File | What it is | How to load it |
|---|---|---|
| `ui-theme.js` | the runtime: applies the saved theme before first paint, fills pickers, `window.UITheme` | classic blocking `<script>`, first thing in `<head>` |
| `ui-theme-base.css` | element defaults (body type, links, focus ring, scrollbars, selection), text tiers `.ui-text-*`, markdown `.ui-markdown` | `<link>` **before** your CSS |
| `ui-components.css` | component classes: `.ui-btn`, `.ui-input`, `.ui-card`, `.ui-switch`, `.ui-dialog`, ... | `<link>` **before** your CSS |
| `ui-theme.css` | the colour tokens of every theme | `<link>` **after** your CSS |
| `ui-components.js` | optional: themed `confirm()`/`alert()`/`prompt()` dialogs and toasts | `<script defer>` anywhere |
| `adapters/quasar.js`, `adapters/quasar.css` | NiceGUI / Quasar apps only | JS right after `ui-theme.js`, CSS right after `ui-theme.css` |
| `adapters/tailwind.css` | Tailwind CSS v4 apps: `bg-surface-2`, `text-fg`, ... | `@import` it after `tailwindcss` |
| `adapters/tailwind-v3.preset.js` | Tailwind CSS v3: the same names as a preset | `presets: [require(...)]` in `tailwind.config.js` |
| `ui-theme.d.ts` | TypeScript declarations for `window.UITheme` and `window.UIComponents` | `include` it in `tsconfig.json` |
| `themes.json` | the theme list, for server-side code | not loaded by pages |
| `update.py` | installs and updates this folder | `python ui-theme/update.py` |
| `VERSION`, `files.json` | version and file checksums | not loaded |

## Wire it into a page

```html
<head>
  <script src="/static/ui-theme/ui-theme.js"
          data-themes="purple,midnight-gold,glacier,forest,paper,daylight"
          data-default="auto"
          data-default-dark="midnight-gold"
          data-default-light="daylight"
          data-storage-key="myapp.theme"></script>
  <link rel="stylesheet" href="/static/ui-theme/ui-theme-base.css">
  <link rel="stylesheet" href="/static/ui-theme/ui-components.css">
  <!-- your stylesheets here -->
  <link rel="stylesheet" href="/static/ui-theme/ui-theme.css">
</head>
```

1. The script is a plain, blocking `<script>` first in `<head>`: never `type="module"`,
   `defer` or `async`. It reads its settings from its own tag and paints the theme before
   the page shows, so there is no flash.
2. `ui-theme-base.css` and `ui-components.css` load before your CSS, so your rules win.
3. `ui-theme.css` loads after your CSS, so no old variable of yours can shadow a token.

Settings on the script tag:

| Attribute | Meaning |
|---|---|
| `data-themes` | the picker's themes, in order. Core: `purple, midnight-gold, glacier, forest, paper, daylight`. Opt-in (only when wanted): `electric-yellow, laserlloyd, laserlloyd-light, night-red` |
| `data-default` | a theme slug, or `auto` to follow the visitor's OS light/dark setting until they pick |
| `data-default-dark`, `data-default-light` | the themes `auto` uses (optional) |
| `data-storage-key` | localStorage key for the choice; use your app's name, e.g. `myapp.theme` |
| `data-families` | `true` enables a dark/light toggle within a family (`[data-ui-theme-toggle]`) |

Theme picker: `<select data-ui-theme-picker aria-label="Theme"></select>` anywhere on the
page. It fills and syncs itself.

Serve this folder so updates show up: `Cache-Control: no-cache` (or `max-age=0`), or add
`?v=<hash of the file>` to each URL.

## Build with it

Use tokens, never colour literals: `color: var(--text-primary); background: var(--surface-2)`.

| Need | Token |
|---|---|
| page / nav / card / hover / active ground | `--surface-0` / `-1` / `-2` / `-3` / `-4`; inset wells `--surface-sunken`; menus `--surface-overlay` |
| text | `--text-primary`, quieter `--text-secondary`, hints `--text-tertiary`, disabled `--text-disabled`, links `--text-link` |
| primary action | fill `--accent` (hover `--accent-hover`) with label `--on-accent` |
| on / checked / selected | fill or indicator `--selected`, with `--on-selected` on it |
| lines | `--border`, control edges `--border-strong`, `--divider`, focus `--focus-ring` |
| status | `--success`, `--warning`, `--danger`, `--info`; text `--danger-text` etc.; tinted fill `--danger-subtle`; label on a solid fill `--on-danger` |
| shape and space | `--radius-sm/md/lg`, `--space-1` ... `--space-8`, `--fs-xs` ... `--fs-xl`, `--shadow-1` ... `--shadow-4` |

Components (all in `ui-components.css`): `.ui-btn` (`--primary`, `--ghost`, `--outline`,
`--danger`, `--sm`, `--lg`, `--icon`), `.ui-field` with `.ui-label`, `.ui-input`,
`.ui-select`, `.ui-textarea`, `.ui-help`, `.ui-error`; `.ui-check` with `.ui-checkbox`,
`.ui-radio` or `.ui-switch` (+ `.ui-switch-state`); `.ui-card`, `.ui-well`, `.ui-stat`;
`.ui-badge` (+ status variants), `.ui-dot`, `.ui-count`; `.ui-callout` (+ `--info`,
`--success`, `--warning`, `--danger`); `.ui-tabs`/`.ui-tab`, `.ui-segmented`; `.ui-table`
in `.ui-table-wrap`; `.ui-dialog`, `.ui-menu`, `[data-ui-tooltip]`, `.ui-toast`;
`.ui-progress`, `.ui-spinner`, `.ui-skeleton`; `.ui-codeblock`; `.ui-app` shell with
`__header`, `__nav`, `__main`; `.ui-nav__item`; `.ui-stack`, `.ui-row`, `.ui-grid`. State
comes from attributes: `disabled`, `checked`, `aria-pressed`, `aria-selected`,
`aria-current`, `aria-invalid`, `aria-busy`.

Rules:

- No colour literals in your CSS; never redeclare a token (`--accent: ...`) in your CSS.
- Quieter text is a lower text tier, never `opacity`.
- Labels on an accent fill use `--on-accent`; accent-coloured text uses `--text-link`.
- Rendered markdown goes in an element with class `ui-markdown`.
- Keep the focus ring: never `outline: none` without a replacement.
- Charts and canvas: `UITheme.token('--cat-1')`, and redraw in `UITheme.onChange(fn)`.

## Update

```
python ui-theme/update.py            # latest release
python ui-theme/update.py --check    # is there one?
```

It replaces only the files it owns and refuses if one was edited by hand.
