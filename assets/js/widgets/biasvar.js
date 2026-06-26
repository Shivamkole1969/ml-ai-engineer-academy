/* biasvar.js — model-complexity slider animates train vs val error.
   Train error falls monotonically; val error is U-shaped (bias→variance). */
import { gaugeShell, field } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('BIAS–VARIANCE', 'complexity sweep');
  body.innerHTML = `
    ${field('Model complexity', `<input type="range" min="0" max="100" step="1" value="35" data-k="c">`, '<output data-o="c">35</output>')}
    <canvas data-canvas width="640" height="240" aria-label="Train and validation error vs complexity"></canvas>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);
  const cv = wrap.querySelector('[data-canvas]');
  const ctx = cv.getContext('2d');
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // error models over complexity t in [0,1]
  const trainErr = (t) => 0.05 + 0.6 * Math.exp(-3.2 * t);                 // falls toward ~0
  const valErr = (t) => 0.05 + 0.55 * Math.exp(-3.4 * t) + 0.5 * t * t;    // U-shape

  function draw() {
    const c = +get('c').value; const t = c / 100;
    out('c').textContent = c;
    const W = cv.width, H = cv.height, pad = 34;
    ctx.clearRect(0, 0, W, H);
    const ink = css('--ink-500'), accent = css('--accent'), warn = css('--warn'), ok = css('--ok');
    const x = (tt) => pad + tt * (W - 2 * pad);
    const y = (v) => (H - pad) - Math.min(1, v) * (H - 1.6 * pad);

    // axes
    ctx.strokeStyle = ink; ctx.globalAlpha = .5; ctx.beginPath();
    ctx.moveTo(pad, pad / 2); ctx.lineTo(pad, H - pad); ctx.lineTo(W - pad / 2, H - pad); ctx.stroke(); ctx.globalAlpha = 1;

    const curve = (fn, color) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
      for (let i = 0; i <= 100; i++) { const tt = i / 100; i === 0 ? ctx.moveTo(x(tt), y(fn(tt))) : ctx.lineTo(x(tt), y(fn(tt))); }
      ctx.stroke();
    };
    curve(trainErr, ok);       // train
    curve(valErr, accent);     // val

    // optimal (min of val)
    let best = 0, bv = 1; for (let i = 0; i <= 100; i++) { const v = valErr(i / 100); if (v < bv) { bv = v; best = i / 100; } }
    ctx.strokeStyle = warn; ctx.globalAlpha = .5; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(best), pad / 2); ctx.lineTo(x(best), H - pad); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;

    // current position
    ctx.fillStyle = accent; ctx.beginPath(); ctx.arc(x(t), y(valErr(t)), 5, 0, 7); ctx.fill();
    ctx.fillStyle = ok; ctx.beginPath(); ctx.arc(x(t), y(trainErr(t)), 5, 0, 7); ctx.fill();

    ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillStyle = ok; ctx.fillText('train error', pad + 6, pad / 2 + 8);
    ctx.fillStyle = accent; ctx.fillText('val error', pad + 90, pad / 2 + 8);
    ctx.fillStyle = ink; ctx.fillText('underfit', pad + 4, H - pad + 16); ctx.fillText('overfit', W - pad - 44, H - pad + 16);

    const gap = valErr(t) - trainErr(t);
    let zone, fix;
    if (t < best - 0.12) { zone = 'Underfitting (high bias)'; fix = 'Add capacity / features / train longer.'; }
    else if (t > best + 0.12) { zone = 'Overfitting (high variance)'; fix = 'Regularize, add data, or reduce capacity.'; }
    else { zone = '✓ Sweet spot'; fix = 'Balanced — train and val are both low and close.'; }
    readout.innerHTML = `<strong>${zone}</strong> &nbsp;·&nbsp; train ${(trainErr(t) * 100).toFixed(0)}% · val ${(valErr(t) * 100).toFixed(0)}% · gap ${(gap * 100).toFixed(0)}%<br><span style="color:var(--ink-500)">${fix}</span>`;
  }

  wrap.querySelectorAll('input').forEach((el) => el.addEventListener('input', draw));
  draw();
}
