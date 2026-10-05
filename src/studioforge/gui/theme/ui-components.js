/* GENERATED component helpers: ThemeForge 1.0.0, drop-in bundle (ui-theme/).
   https://github.com/LaserLloyd/ThemeForge . Do not edit this copy: update the whole
   folder (python ui-theme/update.py) and keep app styles in the app.
   body sha256: 48cec7393720860e */
/* ============================================================================
   ui-components.js — themed dialogs and toasts (optional) · ThemeForge
   ----------------------------------------------------------------------------
   Drop-in replacements for the browser's confirm()/alert()/prompt(), which
   ignore the theme, plus toast notifications. Uses the .ui-dialog and
   .ui-toast classes from ui-components.css. Load it anywhere (defer is fine):

     <script src="ui-theme/ui-components.js" defer></script>

     if (await UIComponents.confirm({ title: 'Delete job?', message: 'This cannot be undone.',
                                      confirmLabel: 'Delete', danger: true })) { ... }
     await UIComponents.alert('Saved.');
     const name = await UIComponents.prompt({ title: 'Rename', value: 'draft-1' });  // null on cancel
     UIComponents.toast('Settings saved', { kind: 'success' });  // kind: info|success|warning|danger

   Every string is set as text, never as HTML, so it is safe to show user
   input. A string argument is shorthand for { message: string }.
   ============================================================================ */
(function (global) {
  'use strict';

  var doc = global.document;

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
        dialog.setAttribute('aria-label', o.label || (kind === 'confirm' ? 'Confirm' : kind === 'prompt' ? 'Input' : 'Message'));
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
        input.setAttribute('aria-label', o.label || o.title || o.message || 'Value');
        body.appendChild(input);
      }
      form.appendChild(body);

      var actions = el('div', 'ui-dialog__actions');
      if (kind !== 'alert') {
        var cancel = el('button', 'ui-btn', o.cancelLabel || 'Cancel');
        cancel.value = 'cancel';
        actions.appendChild(cancel);
      }
      var ok = el('button', 'ui-btn ' + (o.danger ? 'ui-btn--danger' : 'ui-btn--primary'), o.confirmLabel || 'OK');
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

  global.UIComponents = {
    confirm: function (arg) { return modal('confirm', arg); },
    alert: function (arg) { return modal('alert', arg); },
    prompt: function (arg) { return modal('prompt', arg); },
    toast: toast
  };
})(window);
