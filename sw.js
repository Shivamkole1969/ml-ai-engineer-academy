/* ============================================================================
   sw.js — service worker for offline capability (DoD: works offline after
   first load). Strategy:
   - Navigations & same-origin GET: network-first, fall back to cache.
   - Successful responses are cached so a later offline visit still works.
   - CDN libraries (cross-origin) are cached opportunistically (cache-first).
   This keeps dev/QA fresh (online always wins) while enabling offline use.
   ========================================================================== */

const VERSION = 'mlacademy-v1';
const CORE = [
  './',
  './index.html',
  './assets/css/theme.css',
  './assets/css/app.css',
  './assets/js/main.js',
  './content/curriculum.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(VERSION).then((c) => c.addAll(CORE)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const sameOrigin = url.origin === self.location.origin;

  // CDN assets (fonts, libs): cache-first (they're versioned + immutable).
  if (!sameOrigin) {
    e.respondWith(
      caches.match(req).then((hit) => hit || fetch(req).then((res) => {
        const copy = res.clone();
        caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
        return res;
      }).catch(() => hit))
    );
    return;
  }

  // Same-origin: network-first, fall back to cache (offline), then index for navigations.
  e.respondWith(
    fetch(req).then((res) => {
      const copy = res.clone();
      caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
      return res;
    }).catch(() =>
      caches.match(req).then((hit) => hit || (req.mode === 'navigate' ? caches.match('./index.html') : undefined))
    )
  );
});
