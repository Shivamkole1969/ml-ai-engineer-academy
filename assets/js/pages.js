/* ============================================================================
   pages.js — route handlers. Each returns a DOM node for #view.
   Dashboard, track overview, lesson reader (fully wired), and graceful
   "coming soon" placeholders for sections authored in later phases.
   ========================================================================== */
import { store } from './store.js';
import { renderLesson, hydrate } from './render.js';
import { mountWidgets } from './widgets/index.js';
import { badgePill, BADGE_META } from './components.js';
import { readiness, trackProgress, xpSummary, celebrateCompletion } from './gamify.js';
import { mountHero } from './hero3d.js';

let CURRICULUM = null;
const lessonCache = new Map();

export async function loadCurriculum() {
  if (CURRICULUM) return CURRICULUM;
  const res = await fetch('./content/curriculum.json');
  CURRICULUM = await res.json();
  return CURRICULUM;
}

export function getCurriculum() { return CURRICULUM; }

/** Flatten lessons with track context + ordering for prev/next + lookup. */
export function allLessons() {
  const out = [];
  CURRICULUM.tracks.forEach((t) => (t.lessons || []).forEach((l) => out.push({ ...l, trackId: t.id, trackTitle: t.title })));
  return out;
}

function findLesson(id) { return allLessons().find((l) => l.id === id); }
function findTrack(id) { return CURRICULUM.tracks.find((t) => t.id === id); }

const el = (html) => { const d = document.createElement('div'); d.innerHTML = html.trim(); return d.firstElementChild; };

/* ---- progress ring SVG --------------------------------------------------- */
function ring(pct, size = 84, label = '') {
  const r = (size - 12) / 2, c = 2 * Math.PI * r, off = c * (1 - pct / 100);
  return `<svg class="ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" role="img" aria-label="${pct}% ${label}">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--glass-stroke)" stroke-width="8"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="url(#rg)" stroke-width="8"
      stroke-linecap="round" stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 ${size / 2} ${size / 2})"/>
    <text x="50%" y="50%" text-anchor="middle" dy=".35em" class="ring-num mono">${pct}%</text>
    <defs><linearGradient id="rg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3ad6ff"/><stop offset="1" stop-color="#8b7bff"/></linearGradient></defs>
  </svg>`;
}

/* ========================================================================== */
/* DASHBOARD                                                                  */
/* ========================================================================== */
export async function dashboardPage() {
  await loadCurriculum();
  const r = readiness(CURRICULUM);
  const xp = xpSummary();
  const lessons = allLessons();
  const s = store.get();
  const completedCount = lessons.filter((l) => s.completed[l.id]).length;

  // continue / recommended
  const lastLesson = lessons.find((l) => (s.last.route || '').includes(l.id));
  const nextRec = lessons.find((l) => !s.completed[l.id] && (l.badge === 'HOT')) || lessons.find((l) => !s.completed[l.id]);
  const hot = lessons.filter((l) => l.badge === 'HOT' && !s.completed[l.id]).slice(0, 3);

  const trackCards = CURRICULUM.tracks.map((t, i) => {
    const p = trackProgress(t);
    const m = BADGE_META[t.badge] || BADGE_META.CORE;
    return `<a class="track-card glass" href="#/track/${t.id}" style="--i:${i}">
      <div class="track-card-top">
        <span class="track-card-num mono">${String(i + 1).padStart(2, '0')}</span>
        ${miniRing(p.pct)}
      </div>
      <span class="track-card-title">${escapeHtml(t.title)}</span>
      <span class="track-card-meta mono"><span class="tc-badge ${m.cls}">${m.icon}</span> ${p.done}/${p.total} lessons</span>
      <span class="track-card-bar"><span style="width:${p.pct}%"></span></span>
    </a>`;
  }).join('');

  const node = el(`<main class="page page-dash" role="main">
    <header class="dash-hero glass-hero reveal">
      <div class="dash-hero-aurora" aria-hidden="true"></div>
      <div class="dash-hero-left">
        <p class="eyebrow mono">// your AI/ML engineering journey</p>
        <h1 class="dash-title">Become a <span class="grad-text">production AI/ML engineer</span></h1>
        <p class="lede">A complete, hands-on path — from Python, SQL &amp; classic ML to deep learning, LLMs,
        RAG, agents, MLOps and system design. Learn by reading, poking live instruments, and drilling interviews.</p>
        <div class="dash-cta">
          ${nextRec ? `<a class="btn btn-accent btn-lg" href="#/lesson/${nextRec.id}">▶ ${lastLesson ? 'Continue' : 'Start learning'}</a>` : ''}
          <a class="btn btn-lg" href="#/study-plan">📅 Study plan</a>
          <a class="btn btn-lg" href="#/cheatsheet">⚡ Cheatsheet</a>
        </div>
      </div>
      <div class="dash-hero-right">
        <div class="hero-object">
          <div id="hero3d" class="hero3d-mount" aria-hidden="true"></div>
          <div class="hero-object-readout">
            ${ring(r, 150, 'readiness')}
            <span class="dash-ring-label mono">readiness</span>
          </div>
        </div>
      </div>
    </header>

    <section class="dash-stats reveal">
      <div class="glass stat stat-glow"><span class="stat-num mono" data-count="${xp.streak}">0</span><span class="stat-lbl">🔥 day streak</span></div>
      <div class="glass stat stat-glow"><span class="stat-num mono" data-count="${xp.xp}">0</span><span class="stat-lbl">✦ XP · Lvl ${xp.level}</span></div>
      <div class="glass stat stat-glow"><span class="stat-num mono" data-count="${completedCount}">0</span><span class="stat-lbl">/ ${lessons.length} lessons</span></div>
      <div class="glass stat stat-glow"><span class="stat-num mono" data-count="${CURRICULUM.tracks.length}">0</span><span class="stat-lbl">chapters to master</span></div>
    </section>

    ${hot.length ? `<section class="dash-spotlight glass reveal">
      <h2>🔥 Start here — highest-impact topics</h2>
      <div class="spot-cards">
        ${hot.map((l) => `<a class="spot-card glass" href="#/lesson/${l.id}">
          ${badgePill(l.badge)}<span class="spot-title">${escapeHtml(l.title)}</span>
          <span class="spot-track mono">${escapeHtml(l.trackTitle)} · ${l.minutes}m</span></a>`).join('')}
      </div>
    </section>` : ''}

    <section class="dash-map reveal">
      <div class="dash-map-head">
        <h2>The full curriculum · ${CURRICULUM.tracks.length} chapters</h2>
        <button class="btn" data-act="focus">⚡ Jump to 🔥 topics</button>
      </div>
      <div class="track-grid">${trackCards}</div>
    </section>
  </main>`);

  node.querySelector('[data-act="focus"]').addEventListener('click', () => {
    location.hash = '#/track/' + (CURRICULUM.tracks.find((t) => t.badge === 'HOT') || CURRICULUM.tracks[0]).id;
  });
  staggerReveal(node);
  countUp(node);
  // mount the living-object hero once the node is in the DOM (next frame)
  requestAnimationFrame(() => {
    const mount = document.getElementById('hero3d');
    if (mount) mountHero(mount, { readiness: r });
  });
  return node;
}

/* small ring for track cards */
function miniRing(pct, size = 38) {
  const r = (size - 6) / 2, c = 2 * Math.PI * r, off = c * (1 - pct / 100);
  return `<svg class="mini-ring" width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" aria-hidden="true">
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="var(--glass-stroke)" stroke-width="4"/>
    <circle cx="${size / 2}" cy="${size / 2}" r="${r}" fill="none" stroke="url(#rg)" stroke-width="4" stroke-linecap="round"
      stroke-dasharray="${c}" stroke-dashoffset="${off}" transform="rotate(-90 ${size / 2} ${size / 2})"/>
    <text x="50%" y="50%" text-anchor="middle" dy=".35em" class="mini-ring-num mono">${pct}</text></svg>`;
}

/* count-up animation for stat tiles (reduced-motion safe) */
function countUp(node) {
  const motionOff = document.documentElement.getAttribute('data-motion') === 'off'
    || matchMedia('(prefers-reduced-motion: reduce)').matches;
  node.querySelectorAll('[data-count]').forEach((elm) => {
    const target = +elm.dataset.count;
    if (motionOff || target <= 0) { elm.textContent = target; return; }
    const dur = 700, t0 = performance.now();
    const tick = (t) => {
      const k = Math.min(1, (t - t0) / dur);
      elm.textContent = Math.round(target * (1 - Math.pow(1 - k, 3)));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

/* ========================================================================== */
/* TRACK OVERVIEW                                                              */
/* ========================================================================== */
export async function trackPage({ trackId }) {
  await loadCurriculum();
  const track = findTrack(trackId);
  if (!track) return el(`<main class="page"><div class="glass" style="padding:2rem"><h2>Track not found</h2><a href="#/">← Dashboard</a></div></main>`);
  const prog = trackProgress(track);
  const s = store.get();

  const node = el(`<main class="page page-track" role="main">
    <header class="track-head glass glass-hero reveal">
      <div>
        <p class="eyebrow mono">// ${escapeHtml(track.chapter || '')}</p>
        <h1>${escapeHtml(track.title)} ${badgePill(track.badge)}</h1>
        <p class="lede">${prog.done} of ${prog.total} lessons complete.</p>
      </div>
      ${ring(prog.pct, 96, 'track progress')}
    </header>
    <ol class="lesson-list reveal">
      ${track.lessons.map((l, i) => {
        const done = !!s.completed[l.id];
        return `<li class="lesson-row glass ${done ? 'is-done' : ''}">
          <a href="#/lesson/${l.id}">
            <span class="lesson-idx mono">${done ? '✓' : String(i + 1).padStart(2, '0')}</span>
            <span class="lesson-meta"><span class="lesson-title">${escapeHtml(l.title)}</span>
              <span class="lesson-sub mono">${l.minutes} min · ${l.xp} XP</span></span>
            ${badgePill(l.badge)}
          </a></li>`;
      }).join('')}
    </ol>
  </main>`);
  staggerReveal(node);
  return node;
}

/* ========================================================================== */
/* LESSON READER                                                              */
/* ========================================================================== */
export async function lessonPage({ id }) {
  await loadCurriculum();
  const lesson = findLesson(id);
  if (!lesson) return el(`<main class="page"><div class="glass" style="padding:2rem"><h2>Lesson not found</h2><a href="#/">← Dashboard</a></div></main>`);

  const lessons = allLessons();
  const i = lessons.findIndex((l) => l.id === id);
  const prev = lessons[i - 1], next = lessons[i + 1];
  store.setLast('#/lesson/' + id);

  // fetch + render content (graceful placeholder if not yet authored)
  let rendered = null, missing = false;
  try {
    let raw = lessonCache.get(id);
    if (raw == null) {
      const path = lesson.src ? `./${lesson.src}` : `./content/tracks/${lesson.trackId}/${id}.md`;
      const res = await fetch(path);
      if (!res.ok) throw new Error('not found');
      raw = await res.text();
      lessonCache.set(id, raw);
    }
    rendered = await renderLesson(raw);
  } catch (e) { missing = true; }

  const s = store.get();
  const done = !!s.completed[id];
  const booked = store.isBookmarked(id);

  const node = el(`<main class="page page-lesson" role="main">
    <div class="reading-bar"><div class="reading-bar-fill" data-readbar></div></div>
    <article class="lesson glass">
      <header class="lesson-header">
        <p class="eyebrow mono">${escapeHtml(lesson.trackTitle)}</p>
        <h1>${escapeHtml(lesson.title)}</h1>
        <div class="lesson-tags">${badgePill(lesson.badge)}
          <span class="badge">${lesson.minutes} min</span>
          <span class="badge">${lesson.xp} XP</span>
          <button class="btn lesson-bookmark ${booked ? 'is-on' : ''}" data-act="bookmark" type="button">${booked ? '★ Bookmarked' : '☆ Bookmark'}</button>
        </div>
      </header>
      <div class="lesson-body" data-body></div>
      <footer class="lesson-foot">
        <button class="btn btn-accent lesson-complete" data-act="complete" type="button">${done ? '✓ Completed' : 'Mark complete'}</button>
        <div class="lesson-nav">
          ${prev ? `<a class="btn" href="#/lesson/${prev.id}">← ${escapeHtml(truncate(prev.title, 28))}</a>` : '<span></span>'}
          ${next ? `<a class="btn" href="#/lesson/${next.id}">${escapeHtml(truncate(next.title, 28))} →</a>` : '<span></span>'}
        </div>
      </footer>
      <section class="lesson-notes glass">
        <label class="notes-label mono" for="note-${id}">✎ Your notes (saved locally)</label>
        <textarea id="note-${id}" class="notes-area" placeholder="Jot a thought, a question, a connection…">${escapeHtml(store.getNote(id))}</textarea>
      </section>
    </article>
  </main>`);

  const body = node.querySelector('[data-body]');
  if (missing) {
    body.innerHTML = `<div class="lesson-soon">
      <p>📝 <strong>This lesson is being authored.</strong> The curriculum slot and progress tracking are live —
      full content lands in the content phases.</p>
      <p class="mono" style="color:var(--ink-500)">${escapeHtml(lesson.trackTitle)} · ${lesson.minutes} min planned</p>
    </div>`;
  } else {
    body.innerHTML = rendered.html;
    await hydrate(body);
    await mountWidgets(body);
  }

  // bookmark
  node.querySelector('[data-act="bookmark"]').addEventListener('click', (e) => {
    store.toggleBookmark(id);
    const on = store.isBookmarked(id);
    e.target.classList.toggle('is-on', on);
    e.target.textContent = on ? '★ Bookmarked' : '☆ Bookmark';
  });

  // complete / uncomplete
  const completeBtn = node.querySelector('[data-act="complete"]');
  completeBtn.addEventListener('click', () => {
    if (store.isComplete(id)) {
      store.uncompleteLesson(id, lesson.xp);
      completeBtn.textContent = 'Mark complete';
      completeBtn.classList.remove('is-done');
    } else {
      const result = store.completeLesson(id, lesson.xp);
      completeBtn.textContent = '✓ Completed';
      celebrateCompletion(result, lesson);
      window.dispatchEvent(new CustomEvent('mla:progress'));
    }
  });

  // notes (debounced save)
  const ta = node.querySelector('.notes-area');
  let t = 0;
  ta.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => store.setNote(id, ta.value), 400); });

  // reading progress bar
  const bar = node.querySelector('[data-readbar]');
  const onScroll = () => {
    const sc = document.getElementById('view');
    const max = sc.scrollHeight - sc.clientHeight;
    bar.style.width = (max > 0 ? Math.min(100, (sc.scrollTop / max) * 100) : 0) + '%';
  };
  requestAnimationFrame(() => { const v = document.getElementById('view'); if (v) v.addEventListener('scroll', onScroll, { passive: true }); onScroll(); });

  return node;
}

/* ========================================================================== */
/* PLACEHOLDER SECTIONS (authored in later phases)                            */
/* ========================================================================== */
export function comingSoon(title, blurb) {
  return () => el(`<main class="page" role="main">
    <div class="glass glass-hero" style="padding:2.5rem;max-width:720px;margin:0 auto">
      <p class="eyebrow mono">// authoring in progress</p>
      <h1>${escapeHtml(title)}</h1>
      <p class="lede">${escapeHtml(blurb)}</p>
      <a class="btn btn-accent" href="#/">← Back to dashboard</a>
    </div></main>`);
}

/* ---- helpers ------------------------------------------------------------- */
function staggerReveal(node) {
  const motionOff = document.documentElement.getAttribute('data-motion') === 'off';
  if (motionOff) return;
  node.querySelectorAll('.reveal').forEach((n, i) => { n.style.animationDelay = (i * 50) + 'ms'; });
}
function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + '…' : s; }
function escapeHtml(s = '') {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
