/* ============================================================================
   pages-extra.js — System Design, Interview Mode (FAQ/flashcards/drill),
   Projects, and Glossary route handlers.
   ========================================================================== */
import { store, todayISO } from './store.js';
import { renderMarkdown, hydrate } from './render.js';
import { badgePill } from './components.js';
import { loadCurriculum } from './pages.js';
import { mountWidgets } from './widgets/index.js';

const el = (html) => { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; };
const esc = (s = '') => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function getJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(path); return r.json(); }

/* ========================================================================== */
/* SYSTEM DESIGN                                                              */
/* ========================================================================== */
let SD = null;
async function sd() { if (!SD) SD = await getJSON('./content/system-design/index.json'); return SD; }

export async function systemDesignIndex() {
  const data = await sd();
  return el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:26px">
      <p class="eyebrow mono">// track 16</p>
      <h1>ML/AI System Design 🔥</h1>
      <p class="lede">Premium deep-dives. Each architecture has a diagram, components, data-flow,
      scaling levers, failure modes, cost levers, tradeoffs, and an interview framing.</p>
    </header>
    <section class="glass" style="padding:24px">
      <h2>${esc(data.framework.title)}</h2>
      <p class="lede">A reusable checklist — work through it out loud in any design interview.</p>
      <ol class="path-list">${data.framework.steps.map((s) => `<li>${esc(s)}</li>`).join('')}</ol>
    </section>
    <section class="lesson-list">
      ${data.architectures.map((a) => `<div class="lesson-row glass"><a href="#/system-design/${a.id}">
        <span class="lesson-meta"><span class="lesson-title">${esc(a.title)}</span>
        <span class="lesson-sub mono">${esc(a.blurb)}</span></span>${badgePill(a.badge)}</a></div>`).join('')}
    </section>
  </main>`);
}

export async function systemDesignArch({ archId }) {
  const data = await sd();
  const meta = data.architectures.find((a) => a.id === archId);
  let html, missing = false;
  try { html = await renderMarkdown(await (await fetch(`./content/system-design/${archId}.md`)).text(), { frontmatter: true }); }
  catch (e) { missing = true; }
  const node = el(`<main class="page page-lesson" role="main">
    <article class="lesson glass">
      <header class="lesson-header">
        <p class="eyebrow mono">System Design</p>
        <h1>${esc(meta ? meta.title : 'Architecture')} ${meta ? badgePill(meta.badge) : ''}</h1>
      </header>
      <div class="lesson-body" data-body></div>
      <footer class="lesson-foot"><a class="btn" href="#/system-design">← All architectures</a></footer>
    </article></main>`);
  node.querySelector('[data-body]').innerHTML = missing
    ? `<div class="lesson-soon"><p>🏗️ <strong>This architecture deep-dive is being authored.</strong></p><p class="mono" style="color:var(--ink-500)">${esc(meta ? meta.blurb : '')}</p></div>`
    : html;
  return node;
}

/* ========================================================================== */
/* PROJECTS                                                                   */
/* ========================================================================== */
let PROJ = null;
async function proj() { if (!PROJ) PROJ = await getJSON('./content/projects/index.json'); return PROJ; }

export async function projectsIndex() {
  const data = await proj();
  return el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:26px">
      <p class="eyebrow mono">// track 18</p>
      <h1>Projects &amp; Portfolio ⭐</h1>
      <p class="lede">${esc(data.intro)}</p>
    </header>
    <section class="spot-cards">
      ${data.projects.map((p) => `<a class="spot-card glass" href="#/project/${p.id}">${badgePill(p.badge)}
        <span class="spot-title">${esc(p.title)}</span>
        <span class="spot-track mono">${esc(p.blurb)}</span>
        <span class="spot-track mono" style="color:var(--accent)">maps to: ${esc(p.maps)}</span></a>`).join('')}
    </section>
  </main>`);
}

export async function projectPage({ id }) {
  const data = await proj();
  const meta = data.projects.find((p) => p.id === id);
  let html, missing = false;
  try { html = await renderMarkdown(await (await fetch(`./content/projects/${id}.md`)).text(), { frontmatter: true }); }
  catch (e) { missing = true; }
  const node = el(`<main class="page page-lesson" role="main">
    <article class="lesson glass">
      <header class="lesson-header"><p class="eyebrow mono">Project guide</p>
        <h1>${esc(meta ? meta.title : 'Project')} ${meta ? badgePill(meta.badge) : ''}</h1></header>
      <div class="lesson-body" data-body></div>
      <footer class="lesson-foot"><a class="btn" href="#/projects">← All projects</a></footer>
    </article></main>`);
  node.querySelector('[data-body]').innerHTML = missing
    ? `<div class="lesson-soon"><p>🛠️ <strong>This build guide is being authored.</strong></p><p class="mono" style="color:var(--ink-500)">${esc(meta ? meta.blurb : '')}</p></div>`
    : html;
  return node;
}

/* ========================================================================== */
/* GLOSSARY                                                                   */
/* ========================================================================== */
export async function glossaryPage() {
  const data = await getJSON('./content/glossary.json');
  const terms = data.terms.slice().sort((a, b) => a.term.localeCompare(b.term));
  const node = el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:24px">
      <p class="eyebrow mono">// quick lookup</p><h1>Glossary</h1>
      <input class="gloss-search mono" type="text" placeholder="Filter terms…" aria-label="Filter glossary" />
    </header>
    <section class="gloss-list" data-list>
      ${terms.map((t) => `<div class="glass gloss-item"><span class="gloss-term">${esc(t.term)}</span>
        <span class="gloss-def">${esc(t.def)}</span></div>`).join('')}
    </section></main>`);
  const input = node.querySelector('.gloss-search');
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    node.querySelectorAll('.gloss-item').forEach((it) => {
      it.style.display = it.textContent.toLowerCase().includes(q) ? '' : 'none';
    });
  });
  return node;
}

/* ========================================================================== */
/* STUDY PLAN — 20 days × 1 hour, auto-ticks as lessons complete              */
/* ========================================================================== */
export async function studyPlanPage() {
  await loadCurriculum();
  const data = await getJSON('./content/study-plan.json');
  const s = store.get();
  const titleOf = {};
  (await loadCurriculum()).tracks.forEach((t) => t.lessons.forEach((l) => { titleOf[l.id] = l.title; }));

  // overall progress
  const allLessonIds = data.days.flatMap((d) => d.lessons);
  const doneCount = allLessonIds.filter((id) => s.completed[id]).length;
  const pct = allLessonIds.length ? Math.round((doneCount / allLessonIds.length) * 100) : 0;

  const dayHtml = data.days.map((d) => {
    const done = d.lessons.length ? d.lessons.filter((id) => s.completed[id]).length : 0;
    const complete = d.lessons.length ? done === d.lessons.length : false;
    const lessonsHtml = d.lessons.map((id) => `<li><a href="#/lesson/${id}" class="plan-lesson ${s.completed[id] ? 'is-done' : ''}">
      <span class="plan-dot">${s.completed[id] ? '✓' : '○'}</span>${esc(titleOf[id] || id)}</a></li>`).join('');
    const tasksHtml = (d.tasks || []).map((t) => `<li class="plan-task">▸ ${esc(t)}</li>`).join('');
    return `<details class="glass plan-day ${complete ? 'is-complete' : ''}" ${complete ? '' : 'open'}>
      <summary>
        <span class="plan-daynum mono">Day ${d.day}</span>
        <span class="plan-daytitle">${esc(d.title)}</span>
        <span class="plan-daystat mono">${d.lessons.length ? `${done}/${d.lessons.length}` : (d.route ? '→' : '')}${complete ? ' ✓' : ''}</span>
      </summary>
      <div class="plan-body">
        ${lessonsHtml ? `<ul class="plan-lessons">${lessonsHtml}</ul>` : ''}
        ${tasksHtml ? `<ul class="plan-tasks">${tasksHtml}</ul>` : ''}
        ${d.route ? `<a class="btn" href="${d.route}">Open →</a>` : ''}
        ${d.note ? `<p class="plan-note mono">${esc(d.note)}</p>` : ''}
      </div></details>`;
  }).join('');

  return el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:24px">
      <p class="eyebrow mono">// 20 days · 1 hour/day</p>
      <h1>📅 Your Study Plan</h1>
      <p class="lede">${esc(data.intro)}</p>
      <div class="plan-progress"><div class="plan-progress-bar"><div class="plan-progress-fill" style="width:${pct}%"></div></div>
        <span class="mono">${doneCount}/${allLessonIds.length} lessons · ${pct}%</span></div>
    </header>
    <section class="plan-days">${dayHtml}</section>
  </main>`);
}

/* ========================================================================== */
/* CHEATSHEET — 1-hour interview crash prep                                   */
/* ========================================================================== */
export async function cheatsheetPage() {
  let html, missing = false;
  try { html = await renderMarkdown(await (await fetch('./content/cheatsheet.md')).text()); }
  catch (e) { missing = true; }
  const node = el(`<main class="page page-lesson" role="main">
    <div class="reading-bar"><div class="reading-bar-fill" data-readbar></div></div>
    <article class="lesson glass">
      <header class="lesson-header">
        <p class="eyebrow mono">// 1-hour crash prep</p>
        <h1>⚡ Interview Cheatsheet</h1>
        <div class="lesson-tags">${badgePill('HOT')}
          <span class="badge">~60 min</span>
          <span class="badge">covers 80–90% of hot topics</span></div>
        <p class="lede">One hour to interview. Read top to bottom — this is the 80/20 of 2026 GenAI
        interviews: the numbers, the frameworks, the snippets, the designs, the lines that land.</p>
      </header>
      <div class="lesson-body" data-body></div>
      <footer class="lesson-foot">
        <a class="btn btn-accent" href="#/interview">Drill the FAQ-100 →</a>
        <a class="btn" href="#/playground">Open the instruments →</a>
      </footer>
    </article></main>`);
  const body = node.querySelector('[data-body]');
  if (missing) { body.innerHTML = '<div class="lesson-soon"><p>Cheatsheet is being authored.</p></div>'; return node; }
  body.innerHTML = html;
  await hydrate(body);
  // reading bar
  const bar = node.querySelector('[data-readbar]');
  const onScroll = () => { const sc = document.getElementById('view'); const max = sc.scrollHeight - sc.clientHeight; bar.style.width = (max > 0 ? Math.min(100, (sc.scrollTop / max) * 100) : 0) + '%'; };
  requestAnimationFrame(() => { const v = document.getElementById('view'); if (v) v.addEventListener('scroll', onScroll, { passive: true }); onScroll(); });
  return node;
}

/* ========================================================================== */
/* PLAYGROUND — index of the interactive instruments                          */
/* ========================================================================== */
const WIDGETS = [
  { name: 'vram', title: 'VRAM calculator', desc: 'Will your model fit? The 16·N training rule + inference precision, with a "fits on 80GB?" verdict.', lesson: 'gpu-memory-math' },
  { name: 'kvcache', title: 'KV-cache calculator', desc: 'Watch the cache overtake the weights as sequence length and concurrency grow. Toggle GQA + KV-quant.', lesson: 'kv-cache' },
  { name: 'tokencost', title: 'Token & cost meter', desc: 'Paste text → approximate tokens and per-call cost. Prefill (parallel) vs decode (sequential).', lesson: 'tokenization' },
  { name: 'biasvar', title: 'Bias–variance', desc: 'Sweep model complexity and watch train vs validation error trace the under/overfit curve.', lesson: 'bias-variance' },
  { name: 'calibration', title: 'Calibration demo', desc: 'A reliability diagram + temperature scaling. Bend the curve to the diagonal and drop ECE.', lesson: 'calibration' },
  { name: 'throughput', title: 'Throughput ↔ latency', desc: 'Trade tokens/sec against per-request latency. See why static batching wastes slots.', lesson: 'batching' },
  { name: 'agenterror', title: 'Agent compounding-error', desc: 'Why 0.95¹⁰ ≈ 60%. Per-step reliability × steps → end-to-end success.', lesson: 'compounding-error' },
];

export async function playgroundPage() {
  const node = el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:24px">
      <p class="eyebrow mono">// instruments</p><h1>Playground</h1>
      <p class="lede">Every interactive instrument in one place. Poke them, break them, build intuition —
      then read the lesson each one lives in.</p>
    </header>
    ${WIDGETS.map((w) => `<section class="glass" style="padding:20px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap">
        <h2 style="margin:0">${esc(w.title)}</h2>
        <a class="btn" href="#/lesson/${w.lesson}">Open lesson →</a>
      </div>
      <p class="lede">${esc(w.desc)}</p>
      <div class="block-widget" data-widget="${w.name}"></div>
    </section>`).join('')}
  </main>`);
  await mountWidgets(node);
  return node;
}

/* ========================================================================== */
/* INTERVIEW MODE — FAQ-100 + flashcards + mock drill                         */
/* ========================================================================== */
let FAQ = null;
async function faq() { if (!FAQ) FAQ = await getJSON('./content/faq/faq-100.json'); return FAQ; }

export async function interviewPage() {
  await loadCurriculum();
  const data = await faq();
  const node = el(`<main class="page" role="main">
    <header class="glass glass-hero" style="padding:24px">
      <p class="eyebrow mono">// track 17</p><h1>Interview Mode 🔥</h1>
      <p class="lede">${data.questions.length} of 100 questions live · spaced-repetition flashcards · mock drill.</p>
      <div class="iv-tabs">
        <button class="btn iv-tab is-active" data-tab="faq">FAQ-${data.questions.length}</button>
        <button class="btn iv-tab" data-tab="cards">Flashcards</button>
        <button class="btn iv-tab" data-tab="drill">Mock drill</button>
        <button class="btn iv-tab" data-tab="tips">India tips</button>
      </div>
    </header>
    <section data-panel></section></main>`);
  const panel = node.querySelector('[data-panel]');
  const tabs = node.querySelectorAll('.iv-tab');
  const show = (t) => {
    tabs.forEach((b) => b.classList.toggle('is-active', b.dataset.tab === t));
    if (t === 'faq') renderFaq(panel, data);
    else if (t === 'cards') renderCards(panel, data);
    else if (t === 'drill') renderDrill(panel, data);
    else renderTips(panel);
  };
  tabs.forEach((b) => b.addEventListener('click', () => show(b.dataset.tab)));
  show('faq');
  return node;
}

function renderFaq(panel, data) {
  const s = store.get();
  panel.innerHTML = `
    <div class="glass" style="padding:16px;margin-bottom:14px">
      <input class="gloss-search mono" data-q type="text" placeholder="Search questions…" aria-label="Search FAQ"/>
      <div class="iv-filters mono">
        <select data-diff><option value="">all difficulty</option><option>easy</option><option>med</option><option>hard</option></select>
        <select data-type><option value="">all types</option><option>concept</option><option>practical</option><option>systemdesign</option><option>behavioral</option></select>
      </div>
    </div>
    <div class="faq-list" data-faq></div>`;
  const list = panel.querySelector('[data-faq]');
  const draw = () => {
    const q = panel.querySelector('[data-q]').value.toLowerCase();
    const diff = panel.querySelector('[data-diff]').value;
    const type = panel.querySelector('[data-type]').value;
    const items = data.questions.filter((it) =>
      (!q || (it.q + it.a).toLowerCase().includes(q)) && (!diff || it.difficulty === diff) && (!type || it.type === type));
    list.innerHTML = items.map((it) => {
      const known = store.get().faq.known[it.id];
      return `<details class="glass faq-item ${known ? 'is-known' : ''}">
        <summary><span class="faq-q">${esc(it.q)}</span>
          <span class="faq-tags mono">${it.difficulty || ''} · ${it.type || ''}</span></summary>
        <div class="faq-a">${esc(it.a)}</div>
        <div class="faq-actions">
          <button class="btn" data-known="${it.id}">${known ? '✓ Known' : 'Mark known'}</button>
          <button class="btn" data-unknown="${it.id}">Needs review</button>
        </div></details>`;
    }).join('') || '<p class="lede">No matches.</p>';
    list.querySelectorAll('[data-known]').forEach((b) => b.addEventListener('click', (e) => {
      e.preventDefault(); store.update((st) => { st.faq.known[b.dataset.known] = true; delete st.faq.unknown[b.dataset.known]; }); draw();
    }));
    list.querySelectorAll('[data-unknown]').forEach((b) => b.addEventListener('click', (e) => {
      e.preventDefault(); store.update((st) => { st.faq.unknown[b.dataset.unknown] = true; delete st.faq.known[b.dataset.unknown]; }); draw();
    }));
  };
  panel.querySelectorAll('input,select').forEach((c) => c.addEventListener('input', draw));
  draw();
}

/* ---- flashcards: Leitner / SM-2-lite ------------------------------------- */
const BOX_DAYS = [0, 1, 2, 4, 8, 16];
function cardDue(c) { return !c || !c.due || c.due <= todayISO(); }
function addDays(iso, n) { const d = new Date(iso + 'T00:00:00'); d.setDate(d.getDate() + n); return d.toISOString().slice(0, 10); }

function renderCards(panel, data) {
  const cards = data.questions.map((q) => ({ id: q.id, front: q.q, back: q.a }));
  const s = store.get();
  let due = cards.filter((c) => cardDue(s.cards[c.id]));
  if (!due.length) due = cards.slice(0, Math.min(10, cards.length)); // nothing scheduled yet → start a round
  let i = 0, flipped = false;
  function draw() {
    if (i >= due.length) {
      panel.innerHTML = `<div class="glass glass-hero" style="padding:30px;text-align:center">
        <h2>✓ Round complete</h2><p class="lede">${due.length} cards reviewed. Come back tomorrow for the next due set.</p></div>`;
      store.touchStreak();
      return;
    }
    const c = due[i];
    panel.innerHTML = `<div class="glass glass-hero flashcard" style="padding:30px;text-align:center;min-height:200px">
      <p class="mono" style="color:var(--ink-500)">card ${i + 1} / ${due.length}</p>
      <div class="fc-face">${flipped ? esc(c.back) : esc(c.front)}</div>
      ${flipped ? `<div class="fc-rate">
        <button class="btn" data-r="again">Again</button>
        <button class="btn" data-r="hard">Hard</button>
        <button class="btn" data-r="good">Good</button>
        <button class="btn btn-accent" data-r="easy">Easy</button></div>`
        : `<button class="btn btn-accent" data-flip>Show answer</button>`}</div>`;
    if (!flipped) panel.querySelector('[data-flip]').addEventListener('click', () => { flipped = true; draw(); });
    else panel.querySelectorAll('[data-r]').forEach((b) => b.addEventListener('click', () => {
      const r = b.dataset.r;
      store.update((st) => {
        const cur = st.cards[c.id] || { box: 1 };
        let box = cur.box || 1;
        if (r === 'again') box = 1; else if (r === 'hard') box = Math.max(1, box - 1);
        else if (r === 'good') box = Math.min(5, box + 1); else box = Math.min(5, box + 2);
        st.cards[c.id] = { box, due: addDays(todayISO(), BOX_DAYS[box]) };
      });
      i++; flipped = false; draw();
    }));
  }
  draw();
}

/* ---- mock drill ---------------------------------------------------------- */
function renderDrill(panel, data) {
  const start = () => {
    // weight toward HOT tracks + flagged-unknown
    const s = store.get();
    const pool = data.questions.slice();
    const weighted = [];
    pool.forEach((q) => { let w = 1; if (s.faq.unknown[q.id]) w += 2; weighted.push(...Array(w).fill(q)); });
    const picked = []; const seen = new Set();
    while (picked.length < Math.min(10, pool.length) && weighted.length) {
      const q = weighted[Math.floor((picked.length * 2654435761) % weighted.length)];
      if (!seen.has(q.id)) { seen.add(q.id); picked.push(q); } else weighted.splice(weighted.indexOf(q), 1);
      if (seen.size >= pool.length) break;
    }
    let i = 0, flipped = false, score = 0; const revisit = {};
    const draw = () => {
      if (i >= picked.length) {
        const tracks = Object.keys(revisit);
        panel.innerHTML = `<div class="glass glass-hero" style="padding:30px;text-align:center">
          <h2>Score: ${score} / ${picked.length}</h2>
          <p class="lede">${tracks.length ? 'Revisit: ' + tracks.join(', ') : 'Strong round — nice.'}</p>
          <button class="btn btn-accent" data-restart>Run another round</button></div>`;
        store.touchStreak();
        panel.querySelector('[data-restart]').addEventListener('click', start);
        return;
      }
      const q = picked[i];
      panel.innerHTML = `<div class="glass glass-hero" style="padding:26px;min-height:200px">
        <p class="mono" style="color:var(--ink-500)">question ${i + 1} / ${picked.length}</p>
        <h3>${esc(q.q)}</h3>
        ${flipped ? `<div class="faq-a">${esc(q.a)}</div>
          <div class="fc-rate"><button class="btn" data-miss>Missed it</button><button class="btn btn-accent" data-got>Got it</button></div>`
          : `<button class="btn btn-accent" data-show>Reveal answer</button>`}</div>`;
      if (!flipped) panel.querySelector('[data-show]').addEventListener('click', () => { flipped = true; draw(); });
      else {
        panel.querySelector('[data-got]').addEventListener('click', () => { score++; i++; flipped = false; draw(); });
        panel.querySelector('[data-miss]').addEventListener('click', () => { revisit[q.track] = true; store.update((st) => { st.faq.unknown[q.id] = true; }); i++; flipped = false; draw(); });
      }
    };
    draw();
  };
  panel.innerHTML = `<div class="glass glass-hero" style="padding:30px;text-align:center">
    <h2>Mock drill</h2><p class="lede">A 10-question round, weighted toward 🔥 topics and what you flagged for review. Self-rated.</p>
    <button class="btn btn-accent" data-start>Start a round</button></div>`;
  panel.querySelector('[data-start]').addEventListener('click', start);
}

function renderTips(panel) {
  panel.innerHTML = `<div class="glass" style="padding:24px">
    <h2>Pune / remote-India interview tips</h2>
    <ul class="path-list" style="padding-left:1.2em">
      <li><strong>PyTorch-first.</strong> Product companies expect PyTorch ≫ TensorFlow. Speak it fluently.</li>
      <li><strong>SQL is table-stakes.</strong> Expect a live SQL question even for ML roles.</li>
      <li><strong>Talk about shipping &amp; owning</strong> a system in production — not just notebooks. The bar is "can you keep it alive."</li>
      <li><strong>Bring 3 deployed projects</strong> — an end-to-end pipeline, a RAG app, an MLOps deployment. Portfolio beats certificates.</li>
      <li><strong>Expect depth on RAG, agents, and LLMOps.</strong> GenAI now appears in 40%+ of ML JDs; it carries a 20–40% premium.</li>
      <li><strong>Remote-friendly:</strong> ~30% of postings are remote. Pune clusters: GenAI/RAG product teams, fintech, automotive/ADAS CV.</li>
    </ul></div>`;
}
