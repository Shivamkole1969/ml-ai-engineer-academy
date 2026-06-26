/* kvcache.js — KV-cache vs weights calculator.
   cache_bytes = 2 · layers · kv_heads · head_dim · seq · batch · dtype_bytes
   Shows the crossover where the cache overtakes the weights. GQA + KV-quant. */
import { gaugeShell, field, fmtBytes } from './index.js';

const PRESETS = {
  '7B (Llama-style)':  { layers: 32, heads: 32, headDim: 128, weightsGB: 14 },
  '13B':               { layers: 40, heads: 40, headDim: 128, weightsGB: 26 },
  '70B':               { layers: 80, heads: 64, headDim: 128, weightsGB: 140 },
};

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('KV-CACHE CALCULATOR', 'cache vs weights');
  body.innerHTML = `
    <div class="widget-grid">
      ${field('Preset', `<select data-k="preset">${Object.keys(PRESETS).map((k) => `<option>${k}</option>`).join('')}</select>`)}
      ${field('Seq length (tokens)', `<input type="range" min="512" max="131072" step="512" value="8192" data-k="seq">`, '<output data-o="seq">8192</output>')}
      ${field('Concurrency (batch)', `<input type="range" min="1" max="256" step="1" value="16" data-k="batch">`, '<output data-o="batch">16</output>')}
      ${field('GQA groups (heads ÷ groups)', `<input type="range" min="1" max="8" step="1" value="1" data-k="gqa">`, '<output data-o="gqa">1 (MHA)</output>')}
      ${field('KV dtype', `<select data-k="dtype"><option value="2">fp16 (2B)</option><option value="1">int8 (1B)</option><option value="0.5">int4 (0.5B)</option></select>`)}
    </div>
    <div class="widget-bar"><div class="widget-bar-fill" data-bar></div><div class="widget-bar-label mono" data-bar-label></div></div>`;
  slot.appendChild(wrap);

  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);
  const out = (k) => wrap.querySelector(`[data-o="${k}"]`);

  function compute() {
    const p = PRESETS[get('preset').value];
    const seq = +get('seq').value, batch = +get('batch').value, gqa = +get('gqa').value, dtype = +get('dtype').value;
    out('seq').textContent = seq.toLocaleString();
    out('batch').textContent = batch;
    out('gqa').textContent = gqa === 1 ? '1 (MHA)' : `${gqa} groups`;
    const kvHeads = Math.max(1, Math.round(p.heads / gqa));
    const cacheBytes = 2 * p.layers * kvHeads * p.headDim * seq * batch * dtype;
    const weightsBytes = p.weightsGB * 1e9;
    const ratio = cacheBytes / weightsBytes;
    const total = cacheBytes + weightsBytes;
    const cachePct = Math.min(100, (cacheBytes / total) * 100);

    wrap.querySelector('[data-bar]').style.width = cachePct + '%';
    wrap.querySelector('[data-bar-label]').textContent = `cache ${cachePct.toFixed(0)}% of (cache+weights)`;

    const verdict = ratio >= 1
      ? `⚠️ KV cache (${fmtBytes(cacheBytes)}) now EXCEEDS the weights (${fmtBytes(weightsBytes)}). The cache, not the model, gates your concurrency.`
      : `KV cache ${fmtBytes(cacheBytes)} vs weights ${fmtBytes(weightsBytes)} (${(ratio * 100).toFixed(0)}% of weights).`;
    readout.innerHTML = `<strong>${fmtBytes(cacheBytes)}</strong> KV cache &nbsp;·&nbsp; ${verdict}`;
  }

  wrap.querySelectorAll('input,select').forEach((el) => el.addEventListener('input', compute));
  compute();
}
