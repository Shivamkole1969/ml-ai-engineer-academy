/* ============================================================================
   main.js — bootstrap: shell layout, router, sidebar, top bar, settings,
   command palette. (Phase 1: app shell & engine.)
   ========================================================================== */
import { initBg3d, refresh as refreshBg } from './bg3d.js';
import { store } from './store.js';
import { route, setNotFound, startRouter, navigate, setOnNavigate } from './router.js';
import { registerDocs, initPaletteHotkey, openPalette } from './search.js';
import { xpSummary } from './gamify.js';
import * as pages from './pages.js';
import * as extra from './pages-extra.js';
import { badgePill } from './components.js';
import { initCursor } from './cursor.js';
import { disposeHero } from './hero3d.js';

/* ---- prefs --------------------------------------------------------------- */
function applyPrefs() {
  const s = store.get();
  document.documentElement.setAttribute('data-theme', s.theme);
  document.documentElement.setAttribute('data-motion', s.motion);
  document.documentElement.setAttribute('data-bg3d', s.bg3d);
}
function toggleTheme() {
  store.set('theme', store.get().theme === 'light' ? 'dark' : 'light');
  applyPrefs(); refreshBg(); syncTopbar();
}
function toggleMotion() {
  store.set('motion', store.get().motion === 'off' ? 'on' : 'off');
  applyPrefs(); refreshBg(); syncSettingsLabels();
}
function toggleBg3d() {
  store.set('bg3d', store.get().bg3d === 'off' ? 'on' : 'off');
  applyPrefs(); refreshBg(); syncSettingsLabels();
}

/* ---- shell --------------------------------------------------------------- */
const esc = (s = '') => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

function buildShell() {
  const app = document.getElementById('app');
  app.innerHTML = `
    <a class="sr-only skip" href="#view">Skip to content</a>
    <header id="topbar" class="glass">
      <button class="icon-btn sidebar-burger" data-act="burger" aria-label="Toggle menu">☰</button>
      <a class="brand" href="#/"><span class="brand-mark mono">‹/›</span> <span class="brand-name">ML/AI Academy</span></a>
      <button class="topbar-search mono" data-act="search" aria-label="Search (Command-K)">
        <span>Search…</span><kbd>⌘K</kbd></button>
      <div class="topbar-stats mono">
        <span class="t-streak" title="Day streak">🔥 <b data-stat="streak">0</b></span>
        <span class="t-xp" title="XP and level">✦ <b data-stat="xp">0</b> · L<b data-stat="lvl">1</b></span>
        <button class="icon-btn" data-act="theme" aria-label="Toggle theme">☼/☾</button>
        <button class="icon-btn" data-act="settings" aria-label="Settings">⚙</button>
      </div>
    </header>
    <div id="layout">
      <aside id="sidebar" class="glass" aria-label="Curriculum"><nav data-sidebar></nav></aside>
      <div id="view" tabindex="-1"></div>
    </div>
    <div id="settings-panel" class="glass" hidden role="dialog" aria-label="Settings"></div>
    <div id="scrim" hidden></div>`;

  app.querySelector('[data-act="search"]').addEventListener('click', openPalette);
  app.querySelector('[data-act="theme"]').addEventListener('click', toggleTheme);
  app.querySelector('[data-act="settings"]').addEventListener('click', toggleSettings);
  app.querySelector('[data-act="burger"]').addEventListener('click', toggleSidebar);
  app.querySelector('#scrim').addEventListener('click', () => { closeSidebar(); closeSettings(); });
}

function buildSidebar(curriculum) {
  const s = store.get();
  const nav = document.querySelector('[data-sidebar]');
  const trackHtml = curriculum.tracks.map((t) => {
    const expanded = !!s.sidebar[t.id];
    const lessons = t.lessons || [];
    const done = lessons.filter((l) => s.completed[l.id]).length;
    const pct = lessons.length ? Math.round((done / lessons.length) * 100) : 0;
    return `<div class="side-track" data-track="${t.id}">
      <button class="side-track-head" data-toggle="${t.id}" aria-expanded="${expanded}">
        <span class="side-caret">${expanded ? '▾' : '▸'}</span>
        <span class="side-track-title">${esc(t.title)}</span>
        <span class="side-mini-ring mono" style="--p:${pct}">${pct}%</span>
      </button>
      <ul class="side-lessons" ${expanded ? '' : 'hidden'}>
        ${lessons.map((l) => `<li><a href="#/lesson/${l.id}" class="side-lesson ${s.completed[l.id] ? 'is-done' : ''}">
          <span class="side-dot">${s.completed[l.id] ? '✓' : '•'}</span>${esc(l.title)}</a></li>`).join('')}
      </ul></div>`;
  }).join('');

  const sectionHtml = curriculum.sections.map((sec) =>
    `<a class="side-section" href="${sec.route}">${esc(sec.title)} ${badgePill(sec.badge)}</a>`).join('');

  nav.innerHTML = `
    <a class="side-home" href="#/">⌂ Dashboard</a>
    <div class="side-label mono">TRACKS</div>
    ${trackHtml}
    <div class="side-label mono">SECTIONS</div>
    ${sectionHtml}`;

  nav.querySelectorAll('[data-toggle]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const id = btn.dataset.toggle;
      const open = !store.get().sidebar[id];
      store.update((st) => { st.sidebar[id] = open; });
      const ul = btn.parentElement.querySelector('.side-lessons');
      ul.hidden = !open;
      btn.setAttribute('aria-expanded', String(open));
      btn.querySelector('.side-caret').textContent = open ? '▾' : '▸';
    });
  });
}

/* ---- top bar / sidebar sync --------------------------------------------- */
function syncTopbar() {
  const x = xpSummary();
  const set = (k, v) => { const e = document.querySelector(`[data-stat="${k}"]`); if (e) e.textContent = v; };
  set('streak', x.streak); set('xp', x.xp); set('lvl', x.level);
}
function highlightActive(hash) {
  document.querySelectorAll('.side-lesson, .side-section, .side-home').forEach((a) => {
    a.classList.toggle('is-active', a.getAttribute('href') === hash);
  });
}

/* ---- settings panel ------------------------------------------------------ */
function toggleSettings() {
  const p = document.getElementById('settings-panel');
  if (p.hidden) openSettings(); else closeSettings();
}
function openSettings() {
  const s = store.get();
  const p = document.getElementById('settings-panel');
  p.innerHTML = `
    <h3>Settings</h3>
    <label class="set-row"><span>Theme</span>
      <button class="btn" data-act="theme2">${s.theme === 'light' ? 'Light ☼' : 'Dark ☾'}</button></label>
    <label class="set-row"><span>Reduce motion</span>
      <button class="btn" data-act="motion">${s.motion === 'off' ? 'On (motion off)' : 'Off'}</button></label>
    <label class="set-row"><span>3D background</span>
      <button class="btn" data-act="bg3d">${s.bg3d === 'off' ? 'Disabled' : 'Enabled'}</button></label>
    <p class="mono set-note">Progress is saved locally in this browser. No accounts, no tracking.</p>
    <button class="btn" data-act="close-set">Close</button>`;
  p.hidden = false;
  document.getElementById('scrim').hidden = false;
  p.querySelector('[data-act="theme2"]').addEventListener('click', () => { toggleTheme(); openSettings(); });
  p.querySelector('[data-act="motion"]').addEventListener('click', () => { toggleMotion(); openSettings(); });
  p.querySelector('[data-act="bg3d"]').addEventListener('click', () => { toggleBg3d(); openSettings(); });
  p.querySelector('[data-act="close-set"]').addEventListener('click', closeSettings);
}
function closeSettings() {
  document.getElementById('settings-panel').hidden = true;
  if (document.getElementById('sidebar').classList.contains('open') === false) document.getElementById('scrim').hidden = true;
}
function syncSettingsLabels() { if (!document.getElementById('settings-panel').hidden) openSettings(); }

/* ---- mobile sidebar ------------------------------------------------------ */
function toggleSidebar() { document.getElementById('sidebar').classList.contains('open') ? closeSidebar() : openSidebar(); }
function openSidebar() { document.getElementById('sidebar').classList.add('open'); document.getElementById('scrim').hidden = false; }
function closeSidebar() { document.getElementById('sidebar').classList.remove('open'); if (document.getElementById('settings-panel').hidden) document.getElementById('scrim').hidden = true; }

/* ---- routes -------------------------------------------------------------- */
function registerRoutes() {
  route('#/', pages.dashboardPage);
  route('#/track/:trackId', pages.trackPage);
  route('#/lesson/:id', pages.lessonPage);
  route('#/study-plan', extra.studyPlanPage);
  route('#/cheatsheet', extra.cheatsheetPage);
  route('#/system-design', extra.systemDesignIndex);
  route('#/system-design/:archId', extra.systemDesignArch);
  route('#/interview', extra.interviewPage);
  route('#/projects', extra.projectsIndex);
  route('#/project/:id', extra.projectPage);
  route('#/glossary', extra.glossaryPage);
  route('#/playground', extra.playgroundPage);
  setNotFound(pages.comingSoon('Lost in space', 'That route does not exist yet. Head back to the dashboard.'));

  setOnNavigate((hash) => {
    highlightActive(hash);
    closeSidebar();
    syncTopbar();
    if (hash !== '#/') disposeHero();   // free the WebGL hero when leaving Home
  });
}

/* ---- search docs --------------------------------------------------------- */
function registerSearch(curriculum) {
  const docs = [];
  curriculum.tracks.forEach((t) => (t.lessons || []).forEach((l) =>
    docs.push({ id: 'lesson:' + l.id, title: l.title, sub: t.title, route: '#/lesson/' + l.id, body: (l.tags || []).join(' '), kind: 'lesson' })));
  curriculum.tracks.forEach((t) =>
    docs.push({ id: 'track:' + t.id, title: t.title, sub: t.chapter || 'Track', route: '#/track/' + t.id, body: '', kind: 'section' }));
  curriculum.sections.forEach((sec) =>
    docs.push({ id: 'sec:' + sec.id, title: sec.title, sub: sec.note || '', route: sec.route, body: '', kind: 'section' }));
  registerDocs(docs);
}

/** Fetch + index system-design archs, projects, glossary terms, FAQ (non-blocking). */
async function registerExtraSearch() {
  const tryJSON = async (p) => { try { return await (await fetch(p)).json(); } catch (e) { return null; } };
  const [sd, pj, gl, fq] = await Promise.all([
    tryJSON('./content/system-design/index.json'),
    tryJSON('./content/projects/index.json'),
    tryJSON('./content/glossary.json'),
    tryJSON('./content/faq/faq-100.json'),
  ]);
  const docs = [];
  sd?.architectures?.forEach((a) => docs.push({ id: 'sd:' + a.id, title: a.title, sub: a.blurb, route: '#/system-design/' + a.id, body: '', kind: 'design' }));
  pj?.projects?.forEach((p) => docs.push({ id: 'pj:' + p.id, title: p.title, sub: p.blurb, route: '#/project/' + p.id, body: p.maps || '', kind: 'section' }));
  gl?.terms?.forEach((t) => docs.push({ id: 'gl:' + t.term, title: t.term, sub: 'Glossary', route: '#/glossary', body: t.def, kind: 'glossary' }));
  fq?.questions?.forEach((q) => docs.push({ id: 'fq:' + q.id, title: q.q, sub: 'FAQ', route: '#/interview', body: q.a, kind: 'faq' }));
  if (docs.length) registerDocs(docs);
}

/* ---- boot ---------------------------------------------------------------- */
async function boot() {
  applyPrefs();
  buildShell();
  const curriculum = await pages.loadCurriculum();
  buildSidebar(curriculum);
  registerSearch(curriculum);
  registerExtraSearch();
  registerRoutes();
  initPaletteHotkey();
  syncTopbar();

  window.addEventListener('mla:progress', () => { buildSidebar(curriculum); syncTopbar(); highlightActive(location.hash || '#/'); });

  startRouter();
  initBg3d();
  initCursor();
  registerServiceWorker();
}

/** Offline capability (DoD). Registered after first paint; non-fatal on failure. */
function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  if (location.protocol === 'file:') return; // SW needs http(s)
  addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js').catch(() => { /* offline is a nice-to-have */ });
  });
}

boot();
