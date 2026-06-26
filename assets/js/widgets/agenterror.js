/* agenterror.js — compounding-error: success = p^n. Shows why long agent chains
   collapse, and why "reduce the steps that must all go right" is the real fix. */
import { gaugeShell, field } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('AGENT COMPOUNDING-ERROR', 'p ^ steps');
  body.innerHTML = `
    <div class="widget-grid">
      ${field('Per-step reliability', `<input type="range" min="0.80" max="0.999" step="0.001" value="0.95" data-k="p">`, '<output data-o="p">0.95</output>')}
      ${field('Steps that must all succeed', `<input type="range" min="1" max="30" step="1" value="10" data-k="n">`, '<output data-o="n">10</output>')}
    </div>
    <canvas data-canvas width="640" height="240" aria-label="Success probability vs steps"></canvas>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);
  const cv = wrap.querySelector('[data-canvas]');
  const ctx = cv.getContext('2d');

  function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

  function draw() {
    const p = +get('p').value, n = +get('n').value;
    out('p').textContent = p.toFixed(3);
    out('n').textContent = n;
    const W = cv.width, H = cv.height, pad = 30;
    ctx.clearRect(0, 0, W, H);
    const ink = css('--ink-500') || '#7e8cb8';
    const accent = css('--accent') || '#3ad6ff';
    const bad = css('--bad') || '#ff6b8a';

    // axes
    ctx.strokeStyle = ink; ctx.globalAlpha = 0.5; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad / 2); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad / 2, H - pad); ctx.stroke();
    ctx.globalAlpha = 1;

    const maxN = 30;
    const x = (i) => pad + (i / maxN) * (W - pad - pad / 2);
    const y = (v) => (H - pad) - v * (H - pad - pad / 2);

    // curve p^i
    ctx.strokeStyle = accent; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= maxN; i++) { const v = Math.pow(p, i); i === 0 ? ctx.moveTo(x(i), y(v)) : ctx.lineTo(x(i), y(v)); }
    ctx.stroke();

    // marker at n
    const vn = Math.pow(p, n);
    ctx.fillStyle = vn < 0.5 ? bad : accent;
    ctx.beginPath(); ctx.arc(x(n), y(vn), 5, 0, 7); ctx.fill();

    // 50% guide
    ctx.strokeStyle = bad; ctx.globalAlpha = 0.4; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pad, y(0.5)); ctx.lineTo(W - pad / 2, y(0.5)); ctx.stroke();
    ctx.setLineDash([]); ctx.globalAlpha = 1;

    ctx.fillStyle = ink; ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('100%', 2, y(1) + 4); ctx.fillText('50%', 6, y(0.5) + 4); ctx.fillText('0', pad - 8, H - pad + 14);
    ctx.fillText(`${maxN} steps`, W - 70, H - pad + 14);

    const pct = (vn * 100);
    readout.innerHTML = `At <strong>${(p * 100).toFixed(1)}%</strong>/step over <strong>${n}</strong> steps → <strong>${pct.toFixed(1)}%</strong> end-to-end success.<br>
      <span style="color:var(--ink-500)">${pct < 50 ? '⚠️ A coin-flip or worse. ' : ''}Each extra step multiplies the risk. The fix isn't a smarter model per step — it's <em>fewer steps that must all go right</em> (and validation/retries between them).</span>`;
  }

  wrap.querySelectorAll('input').forEach((el) => el.addEventListener('input', draw));
  draw();
}
