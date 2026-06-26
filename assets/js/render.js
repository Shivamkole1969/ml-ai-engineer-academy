/* ============================================================================
   render.js — lesson content pipeline.
   Markdown (marked) + custom ::: block fences (§6.7) -> HTML -> DOMPurify.
   Also wires post-render behaviour: copy buttons, drills, KaTeX, widgets.
   ========================================================================== */
import { ensureMarked, ensureDOMPurify, ensureHighlight, ensureKatex } from './libs.js';
import * as C from './components.js';

let marked, DOMPurify, hljs;

async function ensureCore() {
  if (!marked) {
    [marked, DOMPurify, hljs] = await Promise.all([ensureMarked(), ensureDOMPurify(), ensureHighlight()]);
    configureMarked();
  }
}

function configureMarked() {
  const renderer = new marked.Renderer();
  renderer.code = function (codeOrToken, infostring) {
    let code, info;
    if (typeof codeOrToken === 'object') { code = codeOrToken.text; info = codeOrToken.lang || ''; }
    else { code = codeOrToken; info = infostring || ''; }
    const lang = (info.split(/\s+/)[0] || '').toLowerCase();
    const attrs = parseAttrs((info.match(/\{[^}]*\}/) || [''])[0]);
    let highlighted;
    try {
      highlighted = (lang && hljs.getLanguage(lang))
        ? hljs.highlight(code, { language: lang }).value
        : hljs.highlightAuto(code).value;
    } catch (e) { highlighted = escapeHtml(code); }
    const title = attrs.title ? `<span class="code-title">${escapeHtml(attrs.title)}</span>` : '';
    const langTag = lang ? `<span class="code-lang">${escapeHtml(lang)}</span>` : '';
    return `<figure class="code-block">
      <figcaption class="code-head">${langTag}${title}
        <button class="code-copy btn" type="button" aria-label="Copy code">Copy</button>
      </figcaption>
      <pre><code class="hljs language-${escapeHtml(lang)}">${highlighted}</code></pre>
    </figure>`;
  };
  marked.setOptions({ renderer, gfm: true, breaks: false });
}

/* ---- frontmatter + attribute parsing ------------------------------------- */

export function splitFrontmatter(raw) {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/);
  if (!m) return { fm: {}, body: raw };
  return { fm: parseYaml(m[1]), body: raw.slice(m[0].length) };
}

function parseYaml(text) {
  const fm = {};
  for (const line of text.split('\n')) {
    const m = line.match(/^([A-Za-z0-9_]+):\s*(.*)$/);
    if (!m) continue;
    fm[m[1]] = coerce(m[2].trim());
  }
  return fm;
}

function coerce(v) {
  if (v === '') return '';
  if (v.startsWith('[') && v.endsWith(']')) {
    return v.slice(1, -1).split(',').map((s) => stripQuotes(s.trim())).filter((s) => s !== '');
  }
  if (v === 'true') return true;
  if (v === 'false') return false;
  if (/^-?\d+(\.\d+)?$/.test(v)) return Number(v);
  return stripQuotes(v);
}

function stripQuotes(s) {
  return s.replace(/^["']|["']$/g, '');
}

function parseAttrs(s) {
  const attrs = {};
  if (!s) return attrs;
  const re = /([A-Za-z0-9_]+)=("([^"]*)"|'([^']*)'|[^\s}]+)/g;
  let m;
  while ((m = re.exec(s))) {
    let val = m[3] ?? m[4] ?? m[2];
    if (val === 'true') val = true;
    else if (val === 'false') val = false;
    attrs[m[1]] = val;
  }
  return attrs;
}

/* ---- block tokenizer ----------------------------------------------------- */

function parseBlocks(body) {
  const lines = body.split('\n');
  const out = [];
  let buf = [];
  const flush = () => { if (buf.length) { out.push({ type: 'md', text: buf.join('\n') }); buf = []; } };
  let inCode = false;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (/^\s*```/.test(line)) inCode = !inCode;     // don't treat ::: inside code fences as blocks
    const open = !inCode && line.match(/^:::([a-z-]+)\s*(\{.*\})?\s*$/);
    if (open) {
      flush();
      const inner = [];
      i++;
      while (i < lines.length && !/^:::\s*$/.test(lines[i])) { inner.push(lines[i]); i++; }
      out.push({ type: 'block', name: open[1], attrs: parseAttrs(open[2] || ''), inner: inner.join('\n') });
    } else {
      buf.push(line);
    }
  }
  flush();
  return out;
}

function md(text) { return marked.parse(text || ''); }

function renderBlock(b) {
  switch (b.name) {
    case 'why-prod':      return C.whyProd(md(b.inner));
    case 'gotcha':        return C.gotcha(md(b.inner));
    case 'war-story':     return C.warStory(md(b.inner), b.attrs.title || 'War story');
    case 'interview-line':return C.interviewLine(md(b.inner));
    case 'key-takeaway':  return C.keyTakeaway(md(b.inner));
    case 'table':
      return `<div class="block block-table">${b.attrs.title ? `<div class="block-label">${escapeHtml(b.attrs.title)}</div>` : ''}<div class="table-scroll">${md(b.inner)}</div></div>`;
    case 'qa':            return C.qa(b.attrs.q || 'Question', md(b.inner));
    case 'drill':         return C.drill({ type: b.attrs.type || 'mcq', q: b.attrs.q || '', options: parseDrillOptions(b.inner) });
    case 'widget':        return C.widgetMount(b.attrs.name || '');
    default:              return md(b.inner);
  }
}

function parseDrillOptions(inner) {
  return inner.split('\n')
    .map((l) => l.match(/^\s*-\s*\[([ xX])\]\s*(.*)$/))
    .filter(Boolean)
    .map((m) => ({ correct: m[1].toLowerCase() === 'x', text: m[2].trim() }));
}

/* ---- public API ---------------------------------------------------------- */

/** Render a full lesson file: returns { fm, html } (html is sanitized). */
export async function renderLesson(raw) {
  await ensureCore();
  const { fm, body } = splitFrontmatter(raw);
  const html = parseBlocks(body).map((b) => (b.type === 'md' ? md(b.text) : renderBlock(b))).join('\n');
  return { fm, html: sanitize(html) };
}

/** Render plain markdown (system-design, projects, FAQ answers). Sanitized HTML string. */
export async function renderMarkdown(raw, { frontmatter = false } = {}) {
  await ensureCore();
  const src = frontmatter ? splitFrontmatter(raw).body : raw;
  const html = parseBlocks(src).map((b) => (b.type === 'md' ? md(b.text) : renderBlock(b))).join('\n');
  return sanitize(html);
}

function sanitize(html) {
  return DOMPurify.sanitize(html, {
    ADD_TAGS: ['details', 'summary', 'figure', 'figcaption', 'aside', 'svg', 'path', 'g', 'circle', 'rect', 'line', 'text', 'polyline', 'polygon'],
    ADD_ATTR: ['data-correct', 'data-drill', 'data-drill-fb', 'data-widget', 'data-type', 'aria-live', 'open', 'viewBox', 'd', 'points', 'x', 'y', 'cx', 'cy', 'r', 'fill', 'stroke', 'stroke-width', 'x1', 'x2', 'y1', 'y2', 'transform'],
  });
}

/* ---- post-render wiring -------------------------------------------------- */

/** Wire copy buttons, drills, KaTeX inside a rendered root element. */
export async function hydrate(root) {
  // copy buttons
  root.querySelectorAll('.code-copy').forEach((btn) => {
    if (btn.dataset.wired) return;
    btn.dataset.wired = '1';
    btn.addEventListener('click', async () => {
      const code = btn.closest('.code-block')?.querySelector('code')?.textContent || '';
      try { await navigator.clipboard.writeText(code); btn.textContent = 'Copied ✓'; }
      catch (e) { btn.textContent = 'Copy failed'; }
      setTimeout(() => { btn.textContent = 'Copy'; }, 1400);
    });
  });

  C.wireDrills(root);

  // KaTeX only if math is present
  if (/\$\$?[^$]/.test(root.textContent || '')) {
    try {
      await ensureKatex();
      window.renderMathInElement(root, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
        ],
        throwOnError: false,
      });
    } catch (e) { /* math optional */ }
  }
}

function escapeHtml(s = '') {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}
