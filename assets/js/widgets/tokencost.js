/* tokencost.js — paste text → approximate tokens (~4 chars/token English) →
   estimated input/output cost. Flags the code/non-English "multilingual tax".
   Client-side estimate only; recommend the model's own tokenizer for billing. */
import { gaugeShell, field, num } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('TOKEN & COST METER', '≈ estimate');
  body.innerHTML = `
    ${field('Paste text', `<textarea data-k="text" rows="4" placeholder="Paste a prompt or document…">Explain how a KV cache grows with sequence length and concurrency, in two short paragraphs.</textarea>`)}
    <div class="widget-grid">
      ${field('Expected output tokens', `<input type="range" min="0" max="4000" step="50" value="500" data-k="out">`, '<output data-o="out">500</output>')}
      ${field('Input $ / 1M tokens', `<input type="number" min="0" step="0.05" value="0.50" data-k="pin">`)}
      ${field('Output $ / 1M tokens', `<input type="number" min="0" step="0.05" value="1.50" data-k="pout">`)}
    </div>
    <div class="widget-breakdown mono" data-breakdown></div>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);

  function estTokens(text) {
    const chars = text.length;
    // ~4 chars/token English; code & non-Latin run denser → apply a small tax
    const nonAscii = (text.match(/[^\x00-\x7F]/g) || []).length;
    const codey = (text.match(/[{}();=<>\[\]]/g) || []).length;
    const tax = 1 + Math.min(0.6, (nonAscii / Math.max(1, chars)) * 1.5 + (codey / Math.max(1, chars)) * 1.2);
    return Math.ceil((chars / 4) * tax);
  }

  function compute() {
    const text = get('text').value;
    const inTok = estTokens(text);
    const outTok = +get('out').value;
    out('out').textContent = outTok;
    const pin = +get('pin').value, pout = +get('pout').value;
    const costIn = (inTok / 1e6) * pin;
    const costOut = (outTok / 1e6) * pout;
    const total = costIn + costOut;

    wrap.querySelector('[data-breakdown]').innerHTML = `
      <div class="kv-row"><span>Input tokens (prefill — parallel)</span><span>${num(inTok)}</span></div>
      <div class="kv-row"><span>Output tokens (decode — sequential)</span><span>${num(outTok)}</span></div>
      <div class="kv-row"><span>Input cost</span><span>$${costIn.toFixed(5)}</span></div>
      <div class="kv-row"><span>Output cost</span><span>$${costOut.toFixed(5)}</span></div>`;
    readout.innerHTML = `<strong>$${total.toFixed(5)}</strong> per call &nbsp;·&nbsp; ≈ <strong>$${(total * 1000).toFixed(2)}</strong> per 1,000 calls<br>
      <span style="color:var(--ink-500)">Estimate only — prefill is parallel (cheap/fast), decode is sequential (the slow, per-token part). Use the model's real tokenizer for billing.</span>`;
  }

  wrap.querySelectorAll('input,textarea').forEach((el) => el.addEventListener('input', compute));
  compute();
}
