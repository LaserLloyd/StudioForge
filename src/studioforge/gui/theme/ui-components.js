/* GENERATED component helpers: ThemeForge 1.2.0, drop-in bundle (ui-theme/).
   https://github.com/LaserLloyd/ThemeForge . Do not edit this copy: update the whole
   folder (python ui-theme/update.py) and keep app styles in the app.
   body sha256: 5c931e3de9b5806d */
/* ============================================================================
   ui-components.js — behaviour for the component layer (optional) · ThemeForge
   ----------------------------------------------------------------------------
   Themed dialogs and toasts, a copy-to-clipboard helper, inline field
   validation, settings-row wiring and a string table for all of them. Load it
   anywhere (defer is fine):

     <script src="ui-theme/ui-components.js" defer></script>

     if (await UIComponents.confirm({ title: 'Delete job?', message: 'This cannot be undone.',
                                      confirmLabel: 'Delete', danger: true })) { ... }
     await UIComponents.alert('Saved.');
     const name = await UIComponents.prompt({ title: 'Rename', value: 'draft-1' });  // null on cancel
     UIComponents.toast('Settings saved', { kind: 'success' });  // kind: info|success|warning|danger

     await UIComponents.copy(button);                       // copies .ui-copy / .ui-codeblock text
     UIComponents.validate(input, 'Enter a port from 1 to 65535.');   // aria-invalid + message
     UIComponents.validate(input, null);                    // clear it
     UIComponents.setting(row, { default: '8080' });        // "reset to default" for a .ui-setting

   Every string is set as text, never as HTML, so it is safe to show user
   input. A string argument is shorthand for { message: string }.

   Words this file puts on the page (Cancel, OK, Copy, Copied, Reset to default, ...)
   come from a string table, so an app can translate them:

     UIComponents.setStrings({ cancel: 'キャンセル', ok: 'OK' });   // merge overrides
     await UIComponents.loadLocale('ja');          // fetches locales/ja.json next to this file
     <script src=".../ui-components.js" data-locale="auto">   // loads the page's <html lang>
     <html data-ui-str-cancel="Annuler">           // any element: its subtree uses this word

   Order of precedence for a word: a data-ui-str-<key> attribute on the element
   or an ancestor, then setStrings()/loadLocale(), then the English default.
   Placeholders like {name} are filled from the call. ui-theme/locales/ holds
   en.json (the defaults) and ja.json.
   ============================================================================ */
(function (global) {
  'use strict';

  var doc = global.document;
  var script = doc.currentScript;
  var base = script && script.src ? script.src.replace(/[^\/]*$/, '') : '';

  /* ---- Strings ---------------------------------------------------------- */

  // Keep in step with locales/en.json (the browser tests compare them).
  var DEFAULTS = {
    cancel: 'Cancel',
    ok: 'OK',
    confirm: 'Confirm',
    input: 'Input',
    message: 'Message',
    value: 'Value',
    copy: 'Copy',
    copied: 'Copied',
    copyFailed: 'Copy failed',
    reset: 'Reset to default',
    resetNamed: 'Reset {name} to default',
    basic: 'Basic',
    advanced: 'Advanced',
    setByEnv: 'Set by environment'
  };
  var table = {};

  function resetStrings() {
    table = {};
    for (var k in DEFAULTS) if (Object.prototype.hasOwnProperty.call(DEFAULTS, k)) table[k] = DEFAULTS[k];
    return strings();
  }

  function strings() {
    var copy = {};
    for (var k in table) if (Object.prototype.hasOwnProperty.call(table, k)) copy[k] = table[k];
    return copy;
  }

  function setStrings(map) {
    if (map && typeof map === 'object') {
      for (var k in map) {
        if (Object.prototype.hasOwnProperty.call(map, k) && k.charAt(0) !== '$' && typeof map[k] === 'string') {
          table[k] = map[k];
        }
      }
    }
    return strings();
  }

  function kebab(key) {
    return key.replace(/([A-Z])/g, function (c) { return '-' + c.toLowerCase(); });
  }

  // A word by key. `el` (optional) lets a data-ui-str-<key> attribute on it or an
  // ancestor win; `vars` fills {placeholders}.
  function t(key, el, vars) {
    var text = null;
    if (el && el.closest) {
      var holder = el.closest('[data-ui-str-' + kebab(key) + ']');
      if (holder) text = holder.getAttribute('data-ui-str-' + kebab(key));
    }
    if (text === null) text = doc.documentElement.getAttribute('data-ui-str-' + kebab(key));
    if (text === null) text = Object.prototype.hasOwnProperty.call(table, key) ? table[key] : (DEFAULTS[key] || key);
    if (vars) {
      text = text.replace(/\{(\w+)\}/g, function (m, name) {
        return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : m;
      });
    }
    return text;
  }

  function locale() {
    return doc.documentElement.lang || (global.navigator && global.navigator.language) || 'en';
  }

  // Fetch <this file's folder>/locales/<lang>.json and merge it over the current words
  // (so earlier setStrings() calls for keys the file lacks survive). Tries the full tag
  // (pt-BR) then the language (pt). English is built in: loadLocale('en') restores it.
  // Resolves true when a file was applied.
  function loadLocale(lang, url) {
    lang = String(lang || locale()).toLowerCase().replace(/_/g, '-');
    if (!url && /^en(-|$)/.test(lang)) {  // English is built in
      resetStrings();
      init(doc);
      return Promise.resolve(true);
    }
    if (typeof global.fetch !== 'function') return Promise.resolve(false);
    var tries = url ? [url] : [base + 'locales/' + lang + '.json'];
    if (!url && lang.indexOf('-') > 0) tries.push(base + 'locales/' + lang.split('-')[0] + '.json');
    function attempt(i) {
      if (i >= tries.length) return Promise.resolve(false);
      return global.fetch(tries[i], { credentials: 'same-origin' }).then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      }).then(function (map) {
        setStrings(map);
        init(doc);
        return true;
      }).catch(function () { return attempt(i + 1); });
    }
    return attempt(0);
  }

  // Locale-aware numbers and dates (Intl), in the page's language.
  function formatNumber(value, options) {
    try { return new Intl.NumberFormat(locale(), options).format(value); } catch (e) { return String(value); }
  }

  function formatDate(value, options) {
    var d = value instanceof Date ? value : new Date(value);
    try {
      return new Intl.DateTimeFormat(locale(), options || { dateStyle: 'medium', timeStyle: 'short' }).format(d);
    } catch (e) { return String(value); }
  }

  /* ---- DOM helpers ------------------------------------------------------ */

  function el(tag, cls, text) {
    var node = doc.createElement(tag);
    if (cls) node.className = cls;
    if (text != null && text !== '') node.textContent = String(text);
    return node;
  }

  function options(arg) {
    return typeof arg === 'object' && arg !== null ? arg : { message: arg == null ? '' : String(arg) };
  }

  var uid = 0;

  // A polite live region for messages that have no element of their own
  // (copied, a validation error). Reusing one node lets screen readers announce
  // each change.
  var live = null;
  var liveTimer = null;

  function announce(message) {
    if (!message) return;
    if (!live || !live.isConnected) {
      live = el('div', 'ui-visually-hidden');
      live.setAttribute('role', 'status');
      live.setAttribute('aria-live', 'polite');
      live.setAttribute('data-ui-live', '');
      doc.body.appendChild(live);
    }
    live.textContent = '';
    global.clearTimeout(liveTimer);
    liveTimer = global.setTimeout(function () { live.textContent = String(message); }, 30);
  }

  /* ---- Dialogs ---------------------------------------------------------- */

  // One modal at a time; each call resolves with the result of its own dialog.
  function modal(kind, arg) {
    var o = options(arg);
    var id = 'ui-dialog-' + (++uid);
    return new Promise(function (resolve) {
      var dialog = el('dialog', 'ui-dialog');
      var form = el('form');
      form.method = 'dialog';
      if (o.title) {
        var h = el('h2', 'ui-dialog__title', o.title);
        h.id = id + '-title';
        dialog.setAttribute('aria-labelledby', h.id);
        form.appendChild(h);
      } else {
        dialog.setAttribute('aria-label', o.label || t(kind === 'confirm' ? 'confirm' : kind === 'prompt' ? 'input' : 'message'));
      }
      var body = el('div', 'ui-dialog__body');
      body.id = id + '-body';
      if (o.message) body.appendChild(el('p', null, o.message));
      dialog.setAttribute('aria-describedby', body.id);
      var input = null;
      if (kind === 'prompt') {
        input = el('input', 'ui-input');
        input.type = o.type || 'text';
        input.value = o.value == null ? '' : String(o.value);
        if (o.placeholder) input.placeholder = o.placeholder;
        input.setAttribute('aria-label', o.label || o.title || o.message || t('value'));
        body.appendChild(input);
      }
      form.appendChild(body);

      var actions = el('div', 'ui-dialog__actions');
      if (kind !== 'alert') {
        var cancel = el('button', 'ui-btn', o.cancelLabel || t('cancel', doc.activeElement));
        cancel.value = 'cancel';
        actions.appendChild(cancel);
      }
      var ok = el('button', 'ui-btn ' + (o.danger ? 'ui-btn--danger' : 'ui-btn--primary'),
                  o.confirmLabel || t('ok', doc.activeElement));
      ok.value = 'ok';
      actions.appendChild(ok);
      form.appendChild(actions);
      dialog.appendChild(form);

      var previous = doc.activeElement;
      dialog.addEventListener('close', function () {
        var accepted = dialog.returnValue === 'ok';
        dialog.remove();
        if (previous && typeof previous.focus === 'function') {
          try { previous.focus(); } catch (e) { /* element gone */ }
        }
        if (kind === 'confirm') resolve(accepted);
        else if (kind === 'prompt') resolve(accepted ? input.value : null);
        else resolve();
      });
      // Enter in the prompt's field submits as OK.
      if (input) {
        input.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') { e.preventDefault(); dialog.close('ok'); }
        });
      }
      doc.body.appendChild(dialog);
      dialog.returnValue = '';
      dialog.showModal();
      (input || (kind === 'alert' ? ok : (o.danger ? actions.firstChild : ok))).focus();
      if (input) input.select();
    });
  }

  /* ---- Toasts ----------------------------------------------------------- */

  var region = null;

  function toast(message, opts) {
    var o = opts || {};
    if (!region || !region.isConnected) {
      region = el('div', 'ui-toasts');
      region.setAttribute('role', 'status');
      region.setAttribute('aria-live', 'polite');
      doc.body.appendChild(region);
    }
    var kinds = { info: 1, success: 1, warning: 1, danger: 1 };
    var node = el('div', 'ui-toast' + (kinds[o.kind] ? ' ui-toast--' + o.kind : ''), message);
    region.appendChild(node);
    var timeout = o.timeout == null ? 4000 : o.timeout;
    if (timeout > 0) {
      global.setTimeout(function () { node.remove(); }, timeout);
    }
    return node;
  }

  /* ---- Copy to clipboard ------------------------------------------------ */

  var COPY_BUTTON = '[data-ui-copy], .ui-copy__button, .ui-codeblock__button';

  // The element whose text a copy button copies.
  function copySource(node) {
    var spec = node.getAttribute && node.getAttribute('data-ui-copy');
    if (spec && spec !== 'true' && spec !== '') {
      var named = doc.querySelector(spec);
      if (named) return named;
    }
    var field = node.closest && node.closest('.ui-copy');
    if (field && node.matches && node.matches(COPY_BUTTON)) return field.querySelector('.ui-copy__value') || field;
    var block = node.closest && node.closest('.ui-codeblock');
    if (block && node.matches && node.matches(COPY_BUTTON)) return block.querySelector('pre') || block;
    return node;
  }

  function textOf(source) {
    if (typeof source === 'string') return source;
    if (source && source.matches && source.matches('input, textarea, select')) return source.value;
    return source ? source.textContent : '';
  }

  // execCommand fallback for pages that are not a secure context (plain http on a
  // LAN address), where navigator.clipboard does not exist.
  function legacyCopy(text) {
    var area = el('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.setAttribute('aria-hidden', 'true');
    area.style.cssText = 'position:fixed;top:0;inset-inline-start:0;width:1px;height:1px;padding:0;border:0;opacity:0';
    var previous = doc.activeElement;
    doc.body.appendChild(area);
    var ok = false;
    try {
      area.select();
      area.setSelectionRange(0, text.length);
      ok = !!doc.execCommand('copy');
    } catch (e) { ok = false; }
    area.remove();
    if (previous && typeof previous.focus === 'function') {
      try { previous.focus(); } catch (e) { /* element gone */ }
    }
    return ok;
  }

  function writeClipboard(text) {
    var nav = global.navigator;
    if (nav && nav.clipboard && typeof nav.clipboard.writeText === 'function' && global.isSecureContext !== false) {
      return nav.clipboard.writeText(text).then(function () { return true; }, function () { return legacyCopy(text); });
    }
    return Promise.resolve(legacyCopy(text));
  }

  function copyFeedback(button, ok) {
    if (!button || !button.matches || !button.matches('button, [role="button"]')) return;
    var word = ok ? t('copied', button) : t('copyFailed', button);
    var hasText = button.textContent.replace(/\s+/g, '') !== '';
    if (hasText && button.__uiCopyLabel === undefined) button.__uiCopyLabel = button.textContent;
    button.setAttribute('data-copied', ok ? 'true' : 'failed');
    if (hasText) button.textContent = word;
    announce(word);
    global.clearTimeout(button.__uiCopyTimer);
    button.__uiCopyTimer = global.setTimeout(function () {
      button.removeAttribute('data-copied');
      if (button.__uiCopyLabel !== undefined) button.textContent = button.__uiCopyLabel;
      button.__uiCopyLabel = undefined;
    }, 1800);
  }

  // copy(element | string): copies text, resolves true on success. For a copy
  // button, a .ui-copy or a .ui-codeblock it finds the text itself; an input or
  // textarea copies its value, any other element its text.
  function copy(target) {
    var button = target && target.matches && target.matches(COPY_BUTTON) ? target : null;
    if (!button && target && target.querySelector) button = target.querySelector(COPY_BUTTON);
    var source = typeof target === 'string' ? target : copySource(target);
    var text = textOf(source);
    return writeClipboard(text).then(function (ok) {
      copyFeedback(button, ok);
      return ok;
    });
  }

  /* ---- Inline field validation ------------------------------------------ */

  function controlOf(field) {
    if (!field) return null;
    if (field.matches && field.matches('input, select, textarea')) return field;
    return field.querySelector ? field.querySelector('input, select, textarea') : null;
  }

  function tokens(value) {
    return (value || '').split(/\s+/).filter(Boolean);
  }

  // validate(field, message[, { focus, announce }])
  //   message set:   aria-invalid="true", a .ui-error element after the control that
  //                  aria-describedby points at, and the message announced politely.
  //   message empty: all of that removed.
  // Returns the .ui-error element, or null when cleared.
  function validate(field, message, opts) {
    var o = opts || {};
    var control = controlOf(field);
    if (!control) return null;
    if (!control.id) control.id = 'ui-field-' + (++uid);
    var errId = control.id + '-error';
    var existing = doc.getElementById(errId);
    var describedby = tokens(control.getAttribute('aria-describedby'));
    if (!message) {
      control.removeAttribute('aria-invalid');
      if (existing) existing.remove();
      describedby = describedby.filter(function (id) { return id !== errId; });
      if (describedby.length) control.setAttribute('aria-describedby', describedby.join(' '));
      else control.removeAttribute('aria-describedby');
      return null;
    }
    var host = control.closest('.ui-field, .ui-setting__control') || control.parentNode;
    var err = existing;
    if (!err) {
      err = el('span', 'ui-error');
      err.id = errId;
      host.appendChild(err);
    }
    var changed = err.textContent !== String(message);
    err.textContent = String(message);
    control.setAttribute('aria-invalid', 'true');
    if (describedby.indexOf(errId) < 0) describedby.push(errId);
    control.setAttribute('aria-describedby', describedby.join(' '));
    if (changed && o.announce !== false) announce(message);
    if (o.focus && typeof control.focus === 'function') control.focus();
    return err;
  }

  /* ---- Settings rows ---------------------------------------------------- */

  function readValue(control) {
    if (control.type === 'checkbox' || control.type === 'radio') return control.checked ? 'true' : 'false';
    return String(control.value);
  }

  function writeValue(control, value) {
    if (control.type === 'checkbox' || control.type === 'radio') control.checked = String(value) === 'true';
    else control.value = value == null ? '' : String(value);
  }

  function fire(control, type) {
    var e;
    try { e = new global.Event(type, { bubbles: true }); } catch (err) { e = doc.createEvent('Event'); e.initEvent(type, true, false); }
    control.dispatchEvent(e);
  }

  // Put the string table's words into the empty parts of a row: the level marker,
  // the "set by environment" badge and the reset button (with a name of its own, so
  // twenty of them are not twenty identical buttons). Parts the app filled stay.
  function fill(node, key, vars) {
    if (!node) return;
    if (!node.textContent.trim() || node.__uiWord) {
      node.textContent = t(key, node, vars);
      node.__uiWord = true;
    }
  }

  function words(row) {
    var label = row.querySelector('.ui-setting__label');
    var name = label ? (label.getAttribute('data-ui-name') ||
      (label.querySelector('label') || label).textContent.trim()) : '';
    fill(row.querySelector('.ui-setting__level'), row.getAttribute('data-level') === 'advanced' ? 'advanced' : 'basic');
    fill(row.querySelector('.ui-setting__env'), 'setByEnv');
    var reset = row.querySelector('.ui-setting__reset');
    if (reset) {
      fill(reset, 'reset');
      if (!reset.hasAttribute('aria-label') || reset.__uiAutoLabel) {
        reset.setAttribute('aria-label', t('resetNamed', reset, { name: name }));
        reset.__uiAutoLabel = true;
      }
    }
  }

  // setting(row[, { default, onReset }])
  //   Wires a .ui-setting row: the reset button shows only while the control differs
  //   from its default (opts.default, else the row's data-default), clicking it
  //   restores the default and fires input and change; empty badges and the reset
  //   button get their words from the string table; a row with data-source="env"
  //   disables its control, because the environment owns that value.
  // Returns { update(), destroy() }.
  function setting(row, opts) {
    var o = opts || {};
    var control = row.querySelector('.ui-setting__control input, .ui-setting__control select, .ui-setting__control textarea');
    var reset = row.querySelector('.ui-setting__reset');

    words(row);
    var env = row.getAttribute('data-source') === 'env';
    if (env && control) control.disabled = true;

    function defaultValue() {
      if (Object.prototype.hasOwnProperty.call(o, 'default')) return o.default;
      return row.getAttribute('data-default');
    }

    function update() {
      var d = defaultValue();
      var modified = !!control && d !== null && d !== undefined && readValue(control) !== String(d);
      if (modified) row.setAttribute('data-modified', ''); else row.removeAttribute('data-modified');
      if (reset) reset.hidden = !modified || env || (control && control.disabled);
    }

    function onReset(e) {
      e.preventDefault();
      var d = defaultValue();
      if (!control || d === null || d === undefined) return;
      writeValue(control, d);
      fire(control, 'input');
      fire(control, 'change');
      if (typeof o.onReset === 'function') o.onReset(control, row);
      update();
      if (typeof control.focus === 'function') control.focus();
    }

    if (control) {
      control.addEventListener('input', update);
      control.addEventListener('change', update);
    }
    if (reset) reset.addEventListener('click', onReset);
    update();
    row.__uiSetting = true;
    return {
      update: update,
      destroy: function () {
        if (control) { control.removeEventListener('input', update); control.removeEventListener('change', update); }
        if (reset) reset.removeEventListener('click', onReset);
        row.__uiSetting = false;
      }
    };
  }

  /* ---- Page wiring ------------------------------------------------------ */

  // init([root]): fills empty copy buttons with the word for "Copy" and wires every
  // .ui-setting row (words that came from the table are re-translated when it
  // changes). Runs on DOMContentLoaded and after a locale loads; call it after adding
  // such markup later.
  function init(root) {
    root = root || doc;
    var buttons = root.querySelectorAll ? root.querySelectorAll(COPY_BUTTON) : [];
    for (var i = 0; i < buttons.length; i++) {
      if (!buttons[i].hasAttribute('aria-label')) fill(buttons[i], 'copy');
    }
    var rows = root.querySelectorAll ? root.querySelectorAll('.ui-setting') : [];
    for (var j = 0; j < rows.length; j++) {
      if (rows[j].__uiSetting) words(rows[j]); else setting(rows[j]);
    }
  }

  // One delegated handler, so copy buttons added later work too.
  doc.addEventListener('click', function (e) {
    var button = e.target && e.target.closest ? e.target.closest(COPY_BUTTON) : null;
    if (!button || button.disabled || button.getAttribute('data-ui-copy') === 'off') return;
    copy(button);
  });

  resetStrings();
  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', function () { init(doc); });
  else init(doc);

  var wanted = script && script.getAttribute('data-locale');
  if (wanted) loadLocale(wanted === 'auto' ? locale() : wanted);

  global.UIComponents = {
    confirm: function (arg) { return modal('confirm', arg); },
    alert: function (arg) { return modal('alert', arg); },
    prompt: function (arg) { return modal('prompt', arg); },
    toast: toast,
    copy: copy,
    validate: validate,
    setting: setting,
    announce: announce,
    init: init,
    t: t,
    strings: strings,
    setStrings: setStrings,
    resetStrings: resetStrings,
    loadLocale: loadLocale,
    locale: locale,
    formatNumber: formatNumber,
    formatDate: formatDate
  };
})(window);
