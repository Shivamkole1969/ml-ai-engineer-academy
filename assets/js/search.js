/* ============================================================================
   search.js — lunr full-text index + the ⌘K command palette (signature nav).
   Indexes lessons, sections, system-design archs, FAQ, glossary (whatever is
   registered). Falls back to substring matching if lunr fails to load.
   ========================================================================== */
import { ensureLunr } from './libs.js';
import { navigate } from './router.js';

let docs = [];        // { id, title, sub, route, body, kind }
let idx = null;

/** Register searchable docs (called by main.js once curriculum is loaded). */
export function registerDocs(newDocs) {
  docs = docs.concat(newDocs);
  idx = null; // rebuild lazily
}

async function buildIndex() {
  if (idx) return idx;
  try {
    const lunr = await ensureLunr();
    idx = lunr(function () {
      this.ref('id');
      this.field('title', { boost: 10 });
      this.field('sub', { boost: 3 });
      this.field('body');
      docs.forEach((d) => this.add(d));
    });
  } catch (e) {
    idx = 'fallback';
  }
  return idx;
}

function query(q) {
  q = q.trim();
  if (!q) return docs.slice(0, 12);
  if (idx && idx !== 'fallback') {
    try {
      // build a forgiving query: each term fuzzy + trailing-wildcard
      const qs = q.split(/\s+/).filter(Boolean)
        .map((t) => `${t}~1 ${t}*`).join(' ');
      const hits = idx.search(qs);
      const map = new Map(docs.map((d) => [d.id, d]));
      const ordered = hits.map((h) => map.get(h.ref)).filter(Boolean).slice(0, 20);
      if (ordered.length) return ordered;
    } catch (e) { /* fall through to substring */ }
  }
  const ql = q.toLowerCase();
  return docs.filter((d) => (d.title + ' ' + (d.sub || '') + ' ' + (d.body || '')).toLowerCase().includes(ql)).slice(0, 20);
}

/* ---- palette UI ---------------------------------------------------------- */

let palette = null, listEl = null, inputEl = null, results = [], active = 0;

function buildPalette() {
  palette = document.createElement('div');
  palette.id = 'palette';
  palette.className = 'palette-overlay';
  palette.hidden = true;
  palette.innerHTML = `
    <div class="palette glass glass-hero" role="dialog" aria-modal="true" aria-label="Command palette">
      <input class="palette-input mono" type="text" placeholder="Search lessons, FAQs, designs…  (Esc to close)" aria-label="Search" autocomplete="off" spellcheck="false" />
      <ul class="palette-list" role="listbox"></ul>
      <div class="palette-foot mono"><kbd>↑</kbd><kbd>↓</kbd> navigate · <kbd>↵</kbd> open · <kbd>esc</kbd> close</div>
    </div>`;
  document.body.appendChild(palette);
  listEl = palette.querySelector('.palette-list');
  inputEl = palette.querySelector('.palette-input');

  inputEl.addEventListener('input', () => refresh(inputEl.value));
  inputEl.addEventListener('keydown', onKey);
  palette.addEventListener('click', (e) => { if (e.target === palette) close(); });
}

const KIND_ICON = { lesson: '📘', section: '✦', design: '🏗️', faq: '❓', glossary: '📖', widget: '🎛️' };

function refresh(q) {
  results = query(q);
  active = 0;
  render();
}

function render() {
  if (!results.length) { listEl.innerHTML = `<li class="palette-empty mono">No matches</li>`; return; }
  listEl.innerHTML = results.map((d, i) => `
    <li class="palette-item ${i === active ? 'is-active' : ''}" role="option" data-i="${i}" aria-selected="${i === active}">
      <span class="palette-icon">${KIND_ICON[d.kind] || '•'}</span>
      <span class="palette-text"><span class="palette-title">${escapeHtml(d.title)}</span>
      ${d.sub ? `<span class="palette-sub mono">${escapeHtml(d.sub)}</span>` : ''}</span>
    </li>`).join('');
  listEl.querySelectorAll('.palette-item').forEach((li) => {
    li.addEventListener('mousemove', () => { active = +li.dataset.i; render(); });
    li.addEventListener('click', () => choose(+li.dataset.i));
  });
  const act = listEl.querySelector('.is-active');
  if (act) act.scrollIntoView({ block: 'nearest' });
}

function onKey(e) {
  if (e.key === 'ArrowDown') { e.preventDefault(); active = Math.min(results.length - 1, active + 1); render(); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); active = Math.max(0, active - 1); render(); }
  else if (e.key === 'Enter') { e.preventDefault(); choose(active); }
  else if (e.key === 'Escape') { e.preventDefault(); close(); }
}

function choose(i) {
  const d = results[i];
  if (!d) return;
  close();
  navigate(d.route);
}

export async function openPalette() {
  if (!palette) buildPalette();
  await buildIndex();
  palette.hidden = false;
  document.body.style.overflow = 'hidden';
  inputEl.value = '';
  refresh('');
  inputEl.focus();
}

export function close() {
  if (!palette) return;
  palette.hidden = true;
  document.body.style.overflow = '';
}

export function initPaletteHotkey() {
  addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (palette && !palette.hidden) close(); else openPalette();
    }
  });
}

function escapeHtml(s = '') {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
