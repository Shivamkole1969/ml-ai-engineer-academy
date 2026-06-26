/* ============================================================================
   cursor.js — custom cursor: a ring that trails the pointer with damping and
   scales/inverts over interactive elements. Disabled on touch + reduced-motion.
   ========================================================================== */

export function initCursor() {
  const fine = matchMedia('(pointer: fine)').matches;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
    || document.documentElement.getAttribute('data-motion') === 'off';
  if (!fine || reduce) return;            // no custom cursor on touch / reduced-motion

  const dot = document.createElement('div');
  const ring = document.createElement('div');
  dot.className = 'cursor-dot';
  ring.className = 'cursor-ring';
  document.body.appendChild(ring);
  document.body.appendChild(dot);
  document.documentElement.classList.add('has-custom-cursor');

  let mx = innerWidth / 2, my = innerHeight / 2;   // target
  let rx = mx, ry = my;                            // ring (lagged)

  addEventListener('pointermove', (e) => {
    mx = e.clientX; my = e.clientY;
    dot.style.transform = `translate(${mx}px,${my}px)`;
  }, { passive: true });

  addEventListener('pointerdown', () => ring.classList.add('is-down'));
  addEventListener('pointerup', () => ring.classList.remove('is-down'));

  // hover state over interactive targets (event delegation)
  const interactive = 'a,button,input,textarea,select,summary,[role="button"],.drill-opt,.track-card,.spot-card,.lesson-row';
  addEventListener('pointerover', (e) => {
    if (e.target.closest && e.target.closest(interactive)) ring.classList.add('is-active');
  }, { passive: true });
  addEventListener('pointerout', (e) => {
    if (e.target.closest && e.target.closest(interactive)) ring.classList.remove('is-active');
  }, { passive: true });

  addEventListener('mouseleave', () => { ring.style.opacity = '0'; dot.style.opacity = '0'; });
  addEventListener('mouseenter', () => { ring.style.opacity = ''; dot.style.opacity = ''; });

  (function loop() {
    rx += (mx - rx) * 0.18; ry += (my - ry) * 0.18;   // trailing lag
    ring.style.transform = `translate(${rx}px,${ry}px)`;
    requestAnimationFrame(loop);
  })();
}
