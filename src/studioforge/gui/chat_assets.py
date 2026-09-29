"""CSS and browser-side helpers for the Chat tab's conversation window.

Kept out of ``tabs/chat.py`` so the tab reads as the flow it is. Colours come
only from the theme bundle's tokens (``--surface-*``, ``--text-*``,
``--accent``, ``--syn-*``), so every theme -- light, dark, OLED black -- paints
the conversation without a rule here knowing which one is active.

The script is page-global and idempotent (``window.sfChat``). It does three
things the server cannot do well:

* **Copying.** ``navigator.clipboard`` exists only in a secure context, and
  this panel is routinely opened over plain HTTP on a tailnet. Every copy
  therefore falls back to ``document.execCommand('copy')`` with a one-shot
  ``copy`` listener that sets ``text/html`` + ``text/plain`` itself (so
  "copy formatted" keeps its formatting there too), then to a hidden textarea.
* **Following the stream.** A MutationObserver keeps a conversation window
  pinned to the bottom while text arrives -- but only while the reader is
  already near the bottom -- and shows the "Latest" button otherwise. The same
  rule keeps a thinking fold's small scroll box on its newest line.
* **Code blocks.** Each ``<pre>`` in a rendered reply gets a small "copy"
  button that copies the raw code text.
"""

from __future__ import annotations

CHAT_CSS = """
.sfc-wrap { position: relative; width: 100%; }
.sfc-window {
  height: calc(100vh - 22rem);
  min-height: 18rem;
  overflow-y: auto;
  overscroll-behavior: contain;
  display: flex;
  flex-direction: column;
  gap: .75rem;
  padding: .75rem;
  border-radius: var(--radius-md);
  background: var(--surface-sunken);
  border: 1px solid var(--border-subtle);
}
.sfc-latest {
  position: absolute;
  right: 1.25rem;
  bottom: .75rem;
  display: none !important;
  box-shadow: var(--shadow-2);
}
.sfc-wrap[data-away="1"] .sfc-latest { display: inline-flex !important; }

.sfc-msg {
  width: 100%;
  padding: .45rem .75rem .6rem;
  border-radius: var(--radius-md);
  flex: 0 0 auto;
}
.sfc-user {
  background: var(--surface-2);
  border: 1px solid var(--border-subtle);
}
.sfc-assistant {
  border-left: 3px solid var(--accent);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
}
.sfc-assistant.sfc-failed { border-left-color: var(--danger); }

.sfc-head {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: .15rem .5rem;
  min-height: 1.9rem;
  font-size: var(--fs-xs);
  color: var(--text-tertiary);
}
.sfc-who {
  font-weight: var(--fw-semibold);
  color: var(--text-secondary);
  overflow-wrap: anywhere;
}
.sfc-user .sfc-who { text-transform: uppercase; letter-spacing: var(--ls-label); }
.sfc-assistant .sfc-who { font-family: var(--font-mono); }
.sfc-actions {
  margin-left: auto;
  display: flex;
  gap: 1px;
  opacity: 0;
  transition: opacity var(--dur-fast) var(--ease-standard);
}
.sfc-msg:hover .sfc-actions,
.sfc-msg:focus-within .sfc-actions { opacity: 1; }
@media (hover: none) { .sfc-actions { opacity: 1; } }

.sfc-body { overflow-wrap: anywhere; color: var(--text-primary); }
.sfc-body:empty { display: none; }
/* Tailwind's preflight strips list markers; a reply's lists need them back. */
.sfc-body ol { list-style: decimal; }
.sfc-body ul { list-style: disc; }
.sfc-body ul ul, .sfc-body ol ul { list-style: circle; }
.sfc-body pre { position: relative; }
.sfc-code-copy {
  position: absolute;
  top: 4px;
  right: 4px;
  font: 11px/1.4 var(--font-sans);
  padding: 1px 7px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--glass-stroke);
  background: var(--syn-bg);
  color: var(--syn-fg);
  opacity: .65;
  cursor: pointer;
}
.sfc-code-copy:hover, .sfc-code-copy:focus-visible { opacity: 1; }

/* Pygments (codehilite) token classes -> the theme's syntax palette. */
.sfc-body .codehilite { margin: 0 0 .75em; background: none; }
.sfc-body .codehilite:last-child { margin-bottom: 0; }
.sfc-body .codehilite :is(.k, .kc, .kd, .kn, .kp, .kr) { color: var(--syn-keyword); }
.sfc-body .codehilite :is(.kt, .nc, .nn) { color: var(--syn-type); }
.sfc-body .codehilite :is(.s, .s1, .s2, .sa, .sb, .sc, .sd, .sh, .sx, .ss) {
  color: var(--syn-string);
}
.sfc-body .codehilite :is(.se, .si, .sr) { color: var(--syn-literal); }
.sfc-body .codehilite :is(.c, .c1, .ch, .cm, .cs, .cpf) {
  color: var(--syn-comment); font-style: italic;
}
.sfc-body .codehilite :is(.cp, .nd) { color: var(--syn-meta); }
.sfc-body .codehilite :is(.m, .mb, .mf, .mh, .mi, .mo, .il) { color: var(--syn-number); }
.sfc-body .codehilite :is(.nf, .fm) { color: var(--syn-function); }
.sfc-body .codehilite :is(.nb, .bp) { color: var(--syn-builtin); }
.sfc-body .codehilite .na { color: var(--syn-attr); }
.sfc-body .codehilite .nt { color: var(--syn-tag); }
.sfc-body .codehilite :is(.nv, .vc, .vg, .vi, .vm) { color: var(--syn-variable); }
.sfc-body .codehilite :is(.o, .ow) { color: var(--syn-operator); }
.sfc-body .codehilite .p { color: var(--syn-punctuation); }
.sfc-body .codehilite .gi { color: var(--syn-addition); }
.sfc-body .codehilite :is(.gd, .err) { color: var(--syn-deletion); }
.sfc-body .codehilite .gh, .sfc-body .codehilite .gu { color: var(--syn-title); }

.sfc-think .q-item { min-height: 1.75rem; padding: 0 .25rem; }
.sfc-think .q-item__label { font-size: var(--fs-xs); color: var(--text-tertiary); }
.sfc-think-body {
  max-height: calc(9 * 1.5em);
  overflow-y: auto;
  line-height: 1.5;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  font-size: var(--fs-xs);
  color: var(--text-secondary);
  padding: .25rem .5rem;
  border-left: 2px solid var(--border);
}
.sfc-error { color: var(--danger-text); white-space: pre-wrap; font-size: var(--fs-sm); }
.sfc-composer textarea { max-height: 16rem; max-height: 12lh; }  /* Quasar autogrow honours it */
.sfc-editor textarea { max-height: 24rem; }
"""

CHAT_JS = r"""
(function () {
  if (window.sfChat) return;
  const NEAR = 48;

  function toast(ok) {
    try {
      Quasar.Notify.create({
        message: ok ? 'Copied' : 'Copy failed (the browser blocked the clipboard)',
        type: ok ? 'positive' : 'negative', timeout: ok ? 1200 : 3000, position: 'bottom',
      });
    } catch (_) { /* no Quasar: nothing to say it with */ }
  }

  // execCommand('copy') works without a secure context; a one-shot copy
  // listener decides what lands on the clipboard, HTML included.
  function copyViaEvent(html, plain) {
    let set = false;
    const onCopy = (event) => {
      try {
        if (html !== null) event.clipboardData.setData('text/html', html);
        event.clipboardData.setData('text/plain', plain);
        event.preventDefault();
        set = true;
      } catch (_) { /* leave set false */ }
    };
    document.addEventListener('copy', onCopy, true);
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
    document.removeEventListener('copy', onCopy, true);
    return ok && set;
  }

  function copyViaTextarea(text) {
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.left = '-9999px';
    area.style.top = '0';
    document.body.appendChild(area);
    area.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
    document.body.removeChild(area);
    return ok;
  }

  function copyViaSelection(element) {
    const selection = window.getSelection();
    if (!selection) return false;
    const range = document.createRange();
    range.selectNodeContents(element);
    selection.removeAllRanges();
    selection.addRange(range);
    let ok = false;
    try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
    selection.removeAllRanges();
    return ok;
  }

  async function copyText(text, quiet) {
    let ok = false;
    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.writeText) {
      try { await navigator.clipboard.writeText(text); ok = true; } catch (_) { ok = false; }
    }
    if (!ok) ok = copyViaEvent(null, text) || copyViaTextarea(text);
    if (!quiet) toast(ok);
    return ok;
  }

  function cleanHtml(element) {
    const clone = element.cloneNode(true);
    clone.querySelectorAll('.sfc-code-copy').forEach((b) => b.remove());
    return clone.innerHTML;
  }

  async function copyHtml(id, plain) {
    const element = document.getElementById(id);
    if (!element) return copyText(plain);
    const html = cleanHtml(element);
    let ok = false;
    if (window.isSecureContext && navigator.clipboard && navigator.clipboard.write
        && window.ClipboardItem) {
      try {
        await navigator.clipboard.write([new ClipboardItem({
          'text/html': new Blob([html], { type: 'text/html' }),
          'text/plain': new Blob([plain], { type: 'text/plain' }),
        })]);
        ok = true;
      } catch (_) { ok = false; }
    }
    if (!ok) ok = copyViaEvent(html, plain) || copyViaSelection(element);
    toast(ok);
    return ok;
  }

  function flag(win, near) {
    const wrap = win.parentElement;
    if (wrap && wrap.classList.contains('sfc-wrap')) wrap.dataset.away = near ? '0' : '1';
  }

  // Was the reader at the bottom *before* this growth? Measured against the
  // height last seen, not the current one, so a scroll-up whose scroll event
  // has not been dispatched yet still counts as "reading above".
  function pin(box, isWindow) {
    const seen = box._sfcH === undefined ? box.scrollHeight : box._sfcH;
    const near = seen - box.scrollTop - box.clientHeight < NEAR;
    if (near) box.scrollTop = box.scrollHeight;
    box._sfcH = box.scrollHeight;
    if (isWindow) flag(box, near);
  }

  function bottom() {
    document.querySelectorAll('.sfc-window').forEach((win) => {
      win.scrollTop = win.scrollHeight;
      win._sfcH = win.scrollHeight;
      flag(win, true);
    });
  }

  function decorate(win) {
    win.querySelectorAll('.sfc-body pre').forEach((pre) => {
      if (pre.querySelector(':scope > .sfc-code-copy')) return;
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'sfc-code-copy';
      button.textContent = 'copy';
      button.title = 'Copy this code';
      pre.appendChild(button);
    });
  }

  function follow(win) {
    decorate(win);
    win.querySelectorAll('.sfc-think-body').forEach((box) => pin(box, false));
    pin(win, true);
  }

  document.addEventListener('scroll', (event) => {
    const el = event.target;
    if (!el || !el.classList) return;
    const isWindow = el.classList.contains('sfc-window');
    if (!isWindow && !el.classList.contains('sfc-think-body')) return;
    el._sfcH = el.scrollHeight;
    if (isWindow) flag(el, el.scrollHeight - el.scrollTop - el.clientHeight < NEAR);
  }, true);

  document.addEventListener('click', (event) => {
    const button = event.target && event.target.closest && event.target.closest('.sfc-code-copy');
    if (!button) return;
    event.preventDefault();
    const pre = button.parentElement;
    const clone = pre.cloneNode(true);
    clone.querySelectorAll('.sfc-code-copy').forEach((b) => b.remove());
    copyText(clone.textContent.replace(/\n$/, ''));
  });

  const observer = new MutationObserver((mutations) => {
    const windows = new Set();
    for (const m of mutations) {
      const node = m.target.nodeType === 1 ? m.target : m.target.parentElement;
      const win = node && node.closest ? node.closest('.sfc-window') : null;
      if (win) { windows.add(win); continue; }
      for (const added of m.addedNodes) {
        if (added.nodeType !== 1) continue;
        if (added.matches('.sfc-window')) windows.add(added);
        else added.querySelectorAll('.sfc-window').forEach((w) => windows.add(w));
      }
    }
    windows.forEach(follow);
  });

  function start() {
    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
    document.querySelectorAll('.sfc-window').forEach(follow);
  }
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', start);

  window.sfChat = { copyText, copyHtml, bottom };
})();
"""
