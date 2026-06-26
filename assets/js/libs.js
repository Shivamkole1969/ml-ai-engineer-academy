/* ============================================================================
   libs.js — lazy CDN loaders for runtime libraries.
   Each loader caches its promise so a library is fetched at most once and only
   when first needed (keeps first paint fast; nothing blocks lesson content).
   ========================================================================== */

const cache = new Map();

function loadScript(url) {
  if (cache.has(url)) return cache.get(url);
  const p = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url; s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error('Failed to load ' + url));
    document.head.appendChild(s);
  });
  cache.set(url, p);
  return p;
}

export function loadCss(url) {
  if (cache.has(url)) return cache.get(url);
  const p = new Promise((resolve) => {
    const l = document.createElement('link');
    l.rel = 'stylesheet'; l.href = url;
    l.onload = () => resolve();
    l.onerror = () => resolve(); // non-fatal
    document.head.appendChild(l);
  });
  cache.set(url, p);
  return p;
}

const V = {
  marked: 'https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js',
  dompurify: 'https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js',
  hljs: 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/highlight.min.js',
  hljsCss: 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.9.0/build/styles/github-dark.min.css',
  lunr: 'https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js',
  katex: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js',
  katexAuto: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js',
  katexCss: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
};

export async function ensureMarked() {
  if (window.marked) return window.marked;
  await loadScript(V.marked);
  return window.marked;
}

export async function ensureDOMPurify() {
  if (window.DOMPurify) return window.DOMPurify;
  await loadScript(V.dompurify);
  return window.DOMPurify;
}

export async function ensureHighlight() {
  if (window.hljs) return window.hljs;
  loadCss(V.hljsCss);
  await loadScript(V.hljs);
  return window.hljs;
}

export async function ensureLunr() {
  if (window.lunr) return window.lunr;
  await loadScript(V.lunr);
  return window.lunr;
}

export async function ensureKatex() {
  if (window.katex && window.renderMathInElement) return window.katex;
  loadCss(V.katexCss);
  await loadScript(V.katex);
  await loadScript(V.katexAuto);
  return window.katex;
}
