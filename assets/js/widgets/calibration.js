/* calibration.js — reliability diagram + temperature scaling.
   Toggle over/under/well-calibrated; temperature slider bends the curve toward
   the diagonal; live ECE readout. */
import { gaugeShell, field } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('CALIBRATION', 'reliability diagram');
  body.innerHTML = `
    <div class="widget-grid">
      ${field('Model', `<select data-k="mode"><option value="over">Overconfident</option><option value="under">Underconfident</option><option value="well">Well-calibrated</option></select>`)}
      ${field('Temperature T', `<input type="range" min="0.5" max="3" step="0.05" value="1" data-k="T">`, '<output data-o="T">1.00</output>')}
    </div>
    <canvas data-canvas width="320" height="300" aria-label="Reliability diagram"></canvas>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);
  const cv = wrap.querySelector('[data-canvas]');
  const ctx = cv.getContext('2d');
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

  // raw model confidence -> actual accuracy mapping (before temperature)
  function actualAcc(conf, mode) {
    if (mode === 'over') return Math.pow(conf, 2.1);          // says 0.9, really ~0.8
    if (mode === 'under') return Math.pow(conf, 0.55);        // says 0.6, really ~0.75
    return conf;                                              // diagonal
  }
  // temperature reshapes confidence: T>1 softens (less confident), T<1 sharpens
  function tempConf(conf, T) {
    // approximate logit temperature scaling on a single probability
    const eps = 1e-4; const c = Math.min(1 - eps, Math.max(eps, conf));
    const logit = Math.log(c / (1 - c)) / T;
    return 1 / (1 + Math.exp(-logit));
  }

  function draw() {
    const mode = get('mode').value, T = +get('T').value;
    out('T').textContent = T.toFixed(2);
    const S = 300, pad = 34;
    ctx.clearRect(0, 0, cv.width, cv.height);
    const ink = css('--ink-500'), accent = css('--accent'), ok = css('--ok');
    const X = (v) => pad + v * (S - 1.5 * pad);
    const Y = (v) => (S - pad) - v * (S - 1.5 * pad);

    // perfect diagonal
    ctx.strokeStyle = ink; ctx.globalAlpha = .6; ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;

    // reliability curve + ECE
    let ece = 0, bins = 0;
    ctx.strokeStyle = accent; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 50; i++) {
      const conf = i / 50;
      const shown = tempConf(conf, T);                 // confidence after temperature
      const acc = actualAcc(conf, mode);               // true accuracy at this raw conf
      i === 0 ? ctx.moveTo(X(shown), Y(acc)) : ctx.lineTo(X(shown), Y(acc));
      if (i % 5 === 0) { ece += Math.abs(shown - acc); bins++; }
    }
    ctx.stroke();
    ece = ece / bins;

    // bars hint (sparse points)
    ctx.fillStyle = accent;
    for (let i = 1; i <= 9; i++) { const conf = i / 10; const shown = tempConf(conf, T), acc = actualAcc(conf, mode); ctx.beginPath(); ctx.arc(X(shown), Y(acc), 3, 0, 7); ctx.fill(); }

    ctx.fillStyle = ink; ctx.font = '11px JetBrains Mono, monospace';
    ctx.fillText('confidence →', X(0.3), S - pad + 18); ctx.save(); ctx.translate(12, Y(0.4)); ctx.rotate(-Math.PI / 2); ctx.fillText('accuracy →', 0, 0); ctx.restore();

    const good = ece < 0.04;
    readout.innerHTML = `<strong style="color:${good ? ok : accent}">ECE ≈ ${ece.toFixed(3)}</strong> &nbsp;·&nbsp; ${good ? '✓ well-calibrated' : 'miscalibrated — curve is off the diagonal'}<br>
      <span style="color:var(--ink-500)">Curve below the diagonal = overconfident (says 90%, right 80%). Nudge temperature T until the line hugs the dashed diagonal and ECE drops.</span>`;
  }

  wrap.querySelectorAll('input,select').forEach((el) => el.addEventListener('input', draw));
  draw();
}
