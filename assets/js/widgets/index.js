/* ============================================================================
   widgets/index.js — registry + mounting for the interactive "instruments".
   A lesson's `:::widget {name="kvcache"}` block leaves a placeholder div;
   mountWidgets() lazily imports the matching module and mounts it.
   ========================================================================== */

const REGISTRY = {
  vram:        () => import('./vram.js'),
  kvcache:     () => import('./kvcache.js'),
  tokencost:   () => import('./tokencost.js'),
  biasvar:     () => import('./biasvar.js'),
  calibration: () => import('./calibration.js'),
  throughput:  () => import('./throughput.js'),
  agenterror:  () => import('./agenterror.js'),
};

/** Mount any unmounted widget placeholders found under `root`. */
export async function mountWidgets(root) {
  const slots = root.querySelectorAll('.block-widget[data-widget]');
  for (const slot of slots) {
    if (slot.dataset.mounted) continue;
    const name = slot.dataset.widget;
    const loader = REGISTRY[name];
    if (!loader) { slot.innerHTML = `<p class="mono" style="color:var(--ink-500)">Widget "${name}" coming soon.</p>`; continue; }
    slot.dataset.mounted = '1';
    try {
      const mod = await loader();
      (mod.default || mod.mount)(slot);
    } catch (e) {
      console.error('[widgets] failed to mount', name, e);
      slot.innerHTML = `<p class="mono" style="color:var(--bad)">Widget "${name}" failed to load.</p>`;
    }
  }
}

/* ---- shared gauge helpers ------------------------------------------------ */

export function gaugeShell(title, subtitle = '') {
  const wrap = document.createElement('div');
  wrap.className = 'widget glass';
  wrap.innerHTML = `
    <div class="widget-head">
      <span class="widget-title mono">${title}</span>
      ${subtitle ? `<span class="widget-sub mono">${subtitle}</span>` : ''}
    </div>
    <div class="widget-body"></div>
    <div class="widget-readout mono" aria-live="polite"></div>`;
  return { wrap, body: wrap.querySelector('.widget-body'), readout: wrap.querySelector('.widget-readout') };
}

export function field(label, inputHtml, hint = '') {
  return `<label class="widget-field">
    <span class="widget-field-label mono">${label}</span>
    ${inputHtml}
    ${hint ? `<span class="widget-field-hint mono">${hint}</span>` : ''}
  </label>`;
}

export function fmtBytes(b) {
  if (b >= 1e12) return (b / 1e12).toFixed(2) + ' TB';
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB';
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB';
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' KB';
  return b.toFixed(0) + ' B';
}

export function num(v) { return Number(v).toLocaleString('en-US'); }
