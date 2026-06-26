/* throughput.js — batch-size dial trading tokens/sec vs per-request latency.
   Contrasts static batching (empty-slot waste) with continuous batching
   (saturated). Shows p50/p99 feel. */
import { gaugeShell, field } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('THROUGHPUT ↔ LATENCY', 'batching dial');
  body.innerHTML = `
    <div class="widget-grid">
      ${field('Batch size', `<input type="range" min="1" max="64" step="1" value="16" data-k="b">`, '<output data-o="b">16</output>')}
      ${field('Batching mode', `<select data-k="mode"><option value="cont">Continuous (in-flight)</option><option value="static">Static</option></select>`)}
    </div>
    <canvas data-canvas width="640" height="200" aria-label="Throughput and latency vs batch size"></canvas>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);
  const cv = wrap.querySelector('[data-canvas]');
  const ctx = cv.getContext('2d');
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  function draw() {
    const b = +get('b').value, mode = get('mode').value;
    out('b').textContent = b;
    const W = cv.width, H = cv.height, pad = 34;
    ctx.clearRect(0, 0, W, H);
    const ink = css('--ink-500'), accent = css('--accent'), accent2 = css('--accent-2') || '#8b7bff', bad = css('--bad');

    // efficiency: continuous saturates the batch; static wastes empty slots
    const util = mode === 'cont' ? 1 : Math.max(0.35, 1 - (b - 1) / 90);   // static loses util as batch grows
    const tps = (b * util);                 // throughput ~ batch * utilization
    const tpsMax = 64;                       // for scaling
    const latency = 1 + b * 0.09 * (mode === 'static' ? 1.4 : 1);          // per-request latency grows with batch
    const latMax = 8;

    const x = (v) => pad + v * (W - 2 * pad);
    const barY = H - pad;

    // throughput bar
    ctx.fillStyle = accent;
    const tBar = (tps / tpsMax) * (W - 2 * pad);
    ctx.fillRect(pad, pad, tBar, 26);
    // latency bar
    ctx.fillStyle = latency / latMax > 0.7 ? bad : accent2;
    const lBar = Math.min(1, latency / latMax) * (W - 2 * pad);
    ctx.fillRect(pad, pad + 44, lBar, 26);

    // batch slot visualization (filled vs empty)
    const slots = Math.min(b, 32);
    const sw = (W - 2 * pad) / 32;
    for (let i = 0; i < 32; i++) {
      const filled = i < slots;
      ctx.fillStyle = filled ? css('--ok') : 'rgba(255,255,255,0.06)';
      ctx.fillRect(pad + i * sw + 1, pad + 96, sw - 2, 18);
    }
    // static waste overlay
    if (mode === 'static') {
      const wasteFrac = 1 - util;
      ctx.fillStyle = 'rgba(255,107,138,0.35)';
      ctx.fillRect(pad + (1 - wasteFrac) * (W - 2 * pad), pad + 96, wasteFrac * (W - 2 * pad), 18);
    }

    ctx.fillStyle = ink; ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('throughput (tokens/s)', pad, pad - 6);
    ctx.fillText('per-request latency', pad, pad + 40);
    ctx.fillText(mode === 'static' ? 'batch slots (red = wasted while waiting)' : 'batch slots (continuously refilled)', pad, pad + 92);

    const p50 = latency.toFixed(1), p99 = (latency * 2.4).toFixed(1);
    readout.innerHTML = `<strong>${tps.toFixed(0)}</strong> rel. tokens/s &nbsp;·&nbsp; p50 ≈ ${p50}× · p99 ≈ ${p99}× base<br>
      <span style="color:var(--ink-500)">${mode === 'cont'
        ? 'Continuous batching refills finished slots immediately — high utilization, throughput climbs with batch while latency rises gently.'
        : 'Static batching waits for the whole batch — bigger batches waste slots on early-finishers and inflate tail latency. This is why vLLM uses continuous batching.'}</span>`;
  }

  wrap.querySelectorAll('input,select').forEach((el) => el.addEventListener('input', draw));
  draw();
}
