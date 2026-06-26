/* ============================================================================
   components.js — render helpers for the §6.7 typed lesson blocks.
   Each returns an HTML string (already-sanitized inner content is passed in by
   render.js, which controls the markdown→HTML + DOMPurify pipeline).
   ========================================================================== */

export const BADGE_META = {
  HOT:        { cls: 'badge-hot',        icon: '🔥', label: 'HOT-2026' },
  CORE:       { cls: 'badge-core',       icon: '⭐', label: 'CORE' },
  FOUNDATION: { cls: 'badge-foundation', icon: '📎', label: 'FOUNDATION' },
  SKIM:       { cls: 'badge-skim',       icon: '🧊', label: 'SKIM' },
};

export function badgePill(badge) {
  const m = BADGE_META[badge] || BADGE_META.CORE;
  return `<span class="badge ${m.cls}" title="${m.label}">${m.icon} ${m.label}</span>`;
}

function esc(s = '') {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

/* ---- callout blocks ------------------------------------------------------ */

export function whyProd(html) {
  return `<aside class="block block-whyprod" role="note">
    <div class="block-label">Why it matters in production</div>
    <div class="block-body">${html}</div></aside>`;
}

export function gotcha(html) {
  return `<aside class="block block-gotcha" role="note">
    <div class="block-label">⚠️ Gotcha</div>
    <div class="block-body">${html}</div></aside>`;
}

export function warStory(html, title = 'War story') {
  return `<aside class="block block-warstory" role="note">
    <div class="block-label">⚔️ ${esc(title)}</div>
    <div class="block-body">${html}</div></aside>`;
}

export function interviewLine(html) {
  return `<aside class="block block-interview" role="note">
    <div class="block-label">★ The line that lands</div>
    <div class="block-body">${html}</div></aside>`;
}

export function keyTakeaway(html) {
  return `<aside class="block block-takeaway" role="note">
    <div class="block-label">✓ Key takeaway</div>
    <div class="block-body">${html}</div></aside>`;
}

/* ---- interactive blocks -------------------------------------------------- */

let uid = 0;
const nextId = (p) => `${p}-${++uid}`;

export function qa(question, answerHtml) {
  const id = nextId('qa');
  return `<details class="block block-qa">
    <summary><span class="qa-q">${esc(question)}</span></summary>
    <div class="qa-a" id="${id}">${answerHtml}</div></details>`;
}

/**
 * drill({type:'mcq', q, options:[{text, correct}]}) — instant client-side check.
 */
export function drill({ q, options = [], type = 'mcq' }) {
  const id = nextId('drill');
  const opts = options.map((o, i) => `
    <li>
      <button class="drill-opt" data-correct="${o.correct ? '1' : '0'}" data-drill="${id}" type="button">
        <span class="drill-mark" aria-hidden="true"></span>${esc(o.text)}
      </button>
    </li>`).join('');
  return `<div class="block block-drill" data-type="${esc(type)}">
    <div class="block-label">Self-check</div>
    <p class="drill-q">${esc(q)}</p>
    <ul class="drill-opts" role="list">${opts}</ul>
    <p class="drill-feedback" data-drill-fb="${id}" aria-live="polite"></p>
  </div>`;
}

export function widgetMount(name) {
  return `<div class="block block-widget" data-widget="${esc(name)}">
    <noscript>Interactive widget "${esc(name)}".</noscript>
  </div>`;
}

/* ---- post-render wiring (delegated) -------------------------------------- */

/** Attach drill interactivity within a root element. Idempotent. */
export function wireDrills(root) {
  root.querySelectorAll('.drill-opt').forEach((btn) => {
    if (btn.dataset.wired) return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', () => {
      const id = btn.dataset.drill;
      const correct = btn.dataset.correct === '1';
      const group = root.querySelectorAll(`.drill-opt[data-drill="${id}"]`);
      group.forEach((b) => { b.classList.remove('is-correct', 'is-wrong'); b.disabled = false; });
      btn.classList.add(correct ? 'is-correct' : 'is-wrong');
      if (!correct) {
        group.forEach((b) => { if (b.dataset.correct === '1') b.classList.add('is-correct'); });
      }
      const fb = root.querySelector(`[data-drill-fb="${id}"]`);
      if (fb) {
        fb.textContent = correct ? '✓ Correct.' : '✗ Not quite — the right answer is highlighted.';
        fb.style.color = correct ? 'var(--ok)' : 'var(--bad)';
      }
    });
  });
}
