/* ============================================================================
   router.js — tiny hash router. Pattern segments like ":id" capture params.
   Routes are registered as { pattern, handler }. The handler receives
   (params, ctx) and returns an HTMLElement (or a Promise of one).
   ========================================================================== */

const routes = [];
let notFound = null;
let onNavigate = null;
let current = null;

function compile(pattern) {
  // '#/lesson/:id' -> regex + param names
  const names = [];
  const rx = pattern
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')          // escape regex specials ('/' and ':' are safe)
    .replace(/:([A-Za-z0-9_]+)/g, (_, n) => { names.push(n); return '([^/]+)'; });
  return { rx: new RegExp('^' + rx + '$'), names };
}

export function route(pattern, handler) {
  routes.push({ ...compile(pattern), handler, pattern });
}

export function setNotFound(handler) { notFound = handler; }
export function setOnNavigate(fn) { onNavigate = fn; }

export function currentHash() {
  return location.hash || '#/';
}

export function navigate(hash) {
  if (location.hash === hash) resolve();
  else location.hash = hash;
}

function match(hash) {
  const path = hash.split('?')[0];
  for (const r of routes) {
    const m = r.rx.exec(path);
    if (m) {
      const params = {};
      r.names.forEach((n, i) => { params[n] = decodeURIComponent(m[i + 1]); });
      return { r, params };
    }
  }
  return null;
}

async function resolve() {
  const hash = currentHash();
  const app = document.getElementById('view');
  if (!app) return;
  const found = match(hash);
  const handler = found ? found.r.handler : notFound;
  const params = found ? found.params : {};
  if (!handler) return;

  current = hash;
  let node;
  try {
    node = await handler(params, { hash });
  } catch (e) {
    console.error('[router] handler failed for', hash, e);
    node = errorNode(e);
  }
  if (current !== hash) return; // a newer navigation won the race

  app.replaceChildren(node);
  app.scrollTop = 0;
  if (onNavigate) onNavigate(hash, params);
}

function errorNode(e) {
  const d = document.createElement('div');
  d.className = 'glass';
  d.style.padding = '2rem';
  d.innerHTML = `<h2>Something broke</h2><p class="mono">${String(e && e.message || e)}</p>`;
  return d;
}

export function startRouter() {
  addEventListener('hashchange', resolve);
  resolve();
}
