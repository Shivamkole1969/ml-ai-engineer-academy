/* ============================================================================
   store.js — versioned localStorage state (progress, XP, streaks, bookmarks,
   notes, theme, motion, flashcards, last-position). Survives reloads, fails
   soft when storage is unavailable (private mode / quota).
   ========================================================================== */

const KEY = 'mlacademy.v1';

const DEFAULTS = {
  v: 1,
  theme: 'dark',          // 'dark' | 'light'
  motion: 'on',           // 'on' | 'off'  (manual reduce-motion / disable-3D)
  bg3d: 'on',             // 'on' | 'off'
  completed: {},          // { lessonId: ISODate }
  xp: 0,
  streak: { count: 0, last: null },   // last = YYYY-MM-DD
  bookmarks: {},          // { lessonId: true }
  notes: {},              // { lessonId: "text" }
  faq: { known: {}, unknown: {} },    // { id: true }
  cards: {},              // { cardId: { box, due } }  Leitner/SM-2-lite
  last: { route: '#/', scroll: 0 },   // continue where you left off
  sidebar: {},            // { trackId: true (expanded) }
};

let cache = null;
const listeners = new Set();

function read() {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return structuredClone(DEFAULTS);
    const parsed = JSON.parse(raw);
    // shallow-merge defaults so new fields appear after upgrades
    return { ...structuredClone(DEFAULTS), ...parsed };
  } catch (e) {
    return structuredClone(DEFAULTS);
  }
}

function persist() {
  try {
    localStorage.setItem(KEY, JSON.stringify(cache));
  } catch (e) { /* quota / private mode — keep in-memory */ }
  listeners.forEach((fn) => { try { fn(cache); } catch (e) {} });
}

export const store = {
  get() {
    if (!cache) cache = read();
    return cache;
  },

  /** Mutate via callback, then persist + notify. */
  update(fn) {
    if (!cache) cache = read();
    fn(cache);
    persist();
    return cache;
  },

  set(key, value) {
    return this.update((s) => { s[key] = value; });
  },

  subscribe(fn) {
    listeners.add(fn);
    return () => listeners.delete(fn);
  },

  /* ---- domain helpers --------------------------------------------------- */

  isComplete(lessonId) {
    return !!this.get().completed[lessonId];
  },

  /** Mark a lesson complete: award xp once, advance streak. Returns {awarded, leveledUp}. */
  completeLesson(lessonId, xp = 0) {
    const s = this.get();
    if (s.completed[lessonId]) return { awarded: 0, leveledUp: false };
    const prevLevel = levelFor(s.xp);
    this.update((st) => {
      st.completed[lessonId] = todayISO();
      st.xp += xp;
      bumpStreak(st);
    });
    return { awarded: xp, leveledUp: levelFor(s.xp) > prevLevel };
  },

  uncompleteLesson(lessonId, xp = 0) {
    return this.update((st) => {
      if (st.completed[lessonId]) {
        delete st.completed[lessonId];
        st.xp = Math.max(0, st.xp - xp);
      }
    });
  },

  toggleBookmark(lessonId) {
    return this.update((st) => {
      if (st.bookmarks[lessonId]) delete st.bookmarks[lessonId];
      else st.bookmarks[lessonId] = true;
    });
  },
  isBookmarked(lessonId) { return !!this.get().bookmarks[lessonId]; },

  setNote(lessonId, text) {
    return this.update((st) => {
      if (text && text.trim()) st.notes[lessonId] = text;
      else delete st.notes[lessonId];
    });
  },
  getNote(lessonId) { return this.get().notes[lessonId] || ''; },

  setLast(route, scroll = 0) {
    return this.update((st) => { st.last = { route, scroll }; });
  },

  /** Record an activity day (used by flashcards / drills to keep the streak warm). */
  touchStreak() {
    return this.update((st) => bumpStreak(st));
  },
};

/* ---- streak / level math -------------------------------------------------- */

export function todayISO() {
  // local date YYYY-MM-DD (no Date.now dependency on UTC drift for streak UX)
  const d = new Date();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

function dayDiff(aISO, bISO) {
  const a = new Date(aISO + 'T00:00:00');
  const b = new Date(bISO + 'T00:00:00');
  return Math.round((a - b) / 86400000);
}

function bumpStreak(st) {
  const today = todayISO();
  const last = st.streak.last;
  if (last === today) return;            // already counted today
  if (last && dayDiff(today, last) === 1) st.streak.count += 1;
  else st.streak.count = 1;              // first day or a gap → reset to 1
  st.streak.last = today;
}

/** XP → level: simple escalating curve (level n needs 100·n·(n+1)/2 cumulative). */
export function levelFor(xp) {
  let lvl = 1, need = 0, step = 100;
  while (xp >= need + step) { need += step; lvl++; step += 50; }
  return lvl;
}

export function levelProgress(xp) {
  let lvl = 1, need = 0, step = 100;
  while (xp >= need + step) { need += step; lvl++; step += 50; }
  return { level: lvl, into: xp - need, span: step, pct: Math.round(((xp - need) / step) * 100) };
}
