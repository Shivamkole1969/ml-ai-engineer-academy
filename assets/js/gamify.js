/* ============================================================================
   gamify.js — XP, streak, readiness %, and the level-up toast.
   Readiness weights HOT/CORE lessons more heavily than FOUNDATION/SKIM, so the
   global gauge reflects job-readiness, not raw lesson count.
   ========================================================================== */
import { store, levelProgress } from './store.js';

const WEIGHT = { HOT: 3, CORE: 2, FOUNDATION: 1, SKIM: 0.25 };

/** Readiness % across the whole curriculum, weighted by badge. */
export function readiness(curriculum) {
  let total = 0, done = 0;
  const s = store.get();
  for (const track of curriculum.tracks) {
    for (const lesson of (track.lessons || [])) {
      const w = WEIGHT[lesson.badge] ?? 1;
      total += w;
      if (s.completed[lesson.id]) done += w;
    }
  }
  return total ? Math.round((done / total) * 100) : 0;
}

/** Per-track completion {done, total, pct}. */
export function trackProgress(track) {
  const s = store.get();
  const lessons = track.lessons || [];
  const done = lessons.filter((l) => s.completed[l.id]).length;
  return { done, total: lessons.length, pct: lessons.length ? Math.round((done / lessons.length) * 100) : 0 };
}

export function xpSummary() {
  const s = store.get();
  return { xp: s.xp, ...levelProgress(s.xp), streak: s.streak.count };
}

/* ---- toast --------------------------------------------------------------- */

let toastHost = null;
function host() {
  if (toastHost && document.body.contains(toastHost)) return toastHost;
  toastHost = document.createElement('div');
  toastHost.id = 'toast-host';
  toastHost.setAttribute('aria-live', 'polite');
  toastHost.style.cssText = 'position:fixed;right:18px;bottom:18px;z-index:60;display:flex;flex-direction:column;gap:10px;pointer-events:none';
  document.body.appendChild(toastHost);
  return toastHost;
}

export function toast(msg, kind = 'xp') {
  const t = document.createElement('div');
  t.className = 'glass glass-hero reveal';
  t.style.cssText = 'padding:.7rem 1rem;min-width:200px;max-width:320px;pointer-events:auto;font-size:.9rem';
  const icon = kind === 'level' ? '⚡' : kind === 'streak' ? '🔥' : '✦';
  t.innerHTML = `<span class="mono" style="color:var(--accent)">${icon}</span> ${msg}`;
  host().appendChild(t);
  const motionOff = document.documentElement.getAttribute('data-motion') === 'off';
  const ttl = 2600;
  setTimeout(() => {
    if (!motionOff) { t.style.transition = 'opacity .3s, transform .3s'; t.style.opacity = '0'; t.style.transform = 'translateY(6px)'; }
    setTimeout(() => t.remove(), motionOff ? 0 : 300);
  }, ttl);
}

/** Celebrate completing a lesson: XP toast (+ level-up + streak milestones). */
export function celebrateCompletion(result, lesson) {
  if (result.awarded > 0) toast(`+${result.awarded} XP — ${lesson.title}`, 'xp');
  if (result.leveledUp) {
    const { level } = levelProgress(store.get().xp);
    setTimeout(() => toast(`Level ${level} reached`, 'level'), 350);
  }
  const streak = store.get().streak.count;
  if (streak > 1 && streak % 5 === 0) setTimeout(() => toast(`${streak}-day streak`, 'streak'), 700);
}
