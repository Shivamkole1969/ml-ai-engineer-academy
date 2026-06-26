/* ============================================================================
   main.js — bootstrap. (Phase 0: mission-control hello + 3D field + theme.)
   Phase 1 will mount the router, store, layout, and command palette here.
   ========================================================================== */
import { initBg3d, refresh as refreshBg } from './bg3d.js';

const THEME_KEY = 'mlacademy.theme';
const MOTION_KEY = 'mlacademy.motion';

/* ---- theme / motion (minimal; store.js takes this over in Phase 1) ------- */
function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem(THEME_KEY, theme); } catch (e) {}
}
function applyMotion(off) {
  document.documentElement.setAttribute('data-motion', off ? 'off' : 'on');
  try { localStorage.setItem(MOTION_KEY, off ? 'off' : 'on'); } catch (e) {}
  refreshBg();
}
function initPrefs() {
  let theme = 'dark';
  try { theme = localStorage.getItem(THEME_KEY) || 'dark'; } catch (e) {}
  applyTheme(theme);
  let motionOff = false;
  try { motionOff = localStorage.getItem(MOTION_KEY) === 'off'; } catch (e) {}
  if (motionOff) document.documentElement.setAttribute('data-motion', 'off');
}

/* ---- Phase-0 boot screen ------------------------------------------------- */
function renderBoot() {
  const app = document.getElementById('app');
  if (!app) return;
  app.innerHTML = `
    <main class="boot" role="main">
      <section class="glass glass-hero boot-card reveal" aria-labelledby="boot-title">
        <p class="eyebrow">// systems online</p>
        <h1 id="boot-title">ML/AI Engineer Academy</h1>
        <p class="lede">Production-grade ML/AI engineering — from first principles to
        interview &amp; on-call mastery. A calm mission-control HUD for shipping models.</p>
        <p>
          <button class="btn btn-accent" id="theme-toggle" type="button">☼ / ☾ Toggle theme</button>
          <button class="btn" id="motion-toggle" type="button">Reduce motion</button>
        </p>
        <p class="boot-stat">Phase 0 · scaffold &amp; deploy skeleton · 3D field active</p>
      </section>
    </main>`;

  document.getElementById('theme-toggle').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    applyTheme(cur === 'light' ? 'dark' : 'light');
    refreshBg();
  });
  document.getElementById('motion-toggle').addEventListener('click', () => {
    const off = document.documentElement.getAttribute('data-motion') !== 'off';
    applyMotion(off);
  });
}

/* ---- boot ---------------------------------------------------------------- */
initPrefs();
renderBoot();
initBg3d();
