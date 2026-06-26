/* vram.js — VRAM calculator. Mixed-precision Adam training ≈ 16·N bytes:
   2N fp16 weights + 2N fp16 grads + 4N fp32 master + 4N+4N Adam m/v.
   Inference: fp16 ≈ 2N, int8 ≈ N, int4 ≈ 0.5N. */
import { gaugeShell, field, fmtBytes } from './index.js';

export default function mount(slot) {
  const { wrap, body, readout } = gaugeShell('VRAM CALCULATOR', '16·N rule');
  body.innerHTML = `
    <div class="widget-grid">
      ${field('Params N', `<select data-k="N">
        <option value="7e8">0.7B</option>
        <option value="7e9" selected>7B</option>
        <option value="13e9">13B</option>
        <option value="70e9">70B</option></select>`)}
      ${field('Mode', `<select data-k="mode">
        <option value="train">Training (Adam, mixed precision)</option>
        <option value="infer16">Inference fp16</option>
        <option value="infer8">Inference int8</option>
        <option value="infer4">Inference int4</option></select>`)}
    </div>
    <div class="widget-breakdown mono" data-breakdown></div>
    <div class="widget-bar"><div class="widget-bar-fill" data-bar></div><div class="widget-bar-label mono" data-bar-label></div></div>`;
  slot.appendChild(wrap);

  const GPU = 80e9; // 1×80GB
  const get = (k) => wrap.querySelector(`[data-k="${k}"]`);

  function compute() {
    const N = +get('N').value;
    const mode = get('mode').value;
    let parts = [], total = 0, fix = '';
    if (mode === 'train') {
      parts = [
        ['fp16 weights', 2 * N], ['fp16 grads', 2 * N],
        ['fp32 master', 4 * N], ['Adam m', 4 * N], ['Adam v', 4 * N],
      ];
      total = 16 * N;
      fix = 'Too big? → FSDP/ZeRO shard states, or offload optimizer (DeepSpeed ZeRO-3).';
    } else {
      const perParam = mode === 'infer16' ? 2 : mode === 'infer8' ? 1 : 0.5;
      parts = [[`weights (${perParam}B/param)`, perParam * N]];
      total = perParam * N;
      fix = 'Add KV cache on top — see the KV-cache calculator.';
    }
    const fits = total <= GPU;
    wrap.querySelector('[data-breakdown]').innerHTML = parts
      .map(([n, b]) => `<div class="kv-row"><span>${n}</span><span>${fmtBytes(b)}</span></div>`).join('');
    const pct = Math.min(100, (total / GPU) * 100);
    wrap.querySelector('[data-bar]').style.width = pct + '%';
    wrap.querySelector('[data-bar]').style.background = fits ? 'var(--accent-grad)' : 'linear-gradient(120deg,#ff7a59,#ff6b8a)';
    wrap.querySelector('[data-bar-label]').textContent = `${(total / GPU).toFixed(2)}× of one 80GB GPU`;
    readout.innerHTML = `<strong>${fmtBytes(total)}</strong> total &nbsp;·&nbsp; ${fits ? '✓ fits on 1×80GB' : '✗ does NOT fit on 1×80GB'} <br><span style="color:var(--ink-500)">${fix}</span>`;
  }

  wrap.querySelectorAll('input,select').forEach((el) => el.addEventListener('input', compute));
  compute();
}
