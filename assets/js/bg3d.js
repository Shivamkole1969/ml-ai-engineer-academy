/* ============================================================================
   bg3d.js — the subtle 3D neural/particle field ("the atmosphere").
   - Three.js via importmap (loaded in index.html).
   - Respects prefers-reduced-motion + a manual disable toggle.
   - Pauses on document.hidden; caps DPR; adaptively degrades on slow frames.
   - Falls back silently to the CSS gradient (#bg3d) if WebGL is unavailable.
   ========================================================================== */

let state = {
  started: false,
  running: false,
  renderer: null,
  raf: 0,
  scene: null,
  cam: null,
  points: null,
  lines: null,
  mat: null,
  N: 0,
  mx: 0, my: 0,
  // adaptive degrade
  slowFrames: 0,
  degraded: false,
  lastT: 0,
};

function prefersReduced() {
  return matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/** Whether the field should animate at all (motion allowed + not user-disabled). */
function motionAllowed() {
  const off = document.documentElement.getAttribute('data-motion') === 'off';
  const disabled3d = document.documentElement.getAttribute('data-bg3d') === 'off';
  return !off && !disabled3d && !prefersReduced();
}

function particleCount() {
  const small = innerWidth < 700;
  const lowCore = (navigator.hardwareConcurrency || 4) <= 4;
  if (small || lowCore) return 900;
  return 1800;
}

async function init() {
  if (state.started) return;
  state.started = true;

  const el = document.getElementById('bg3d');
  if (!el) return;

  let THREE;
  try {
    THREE = await import('three');
  } catch (e) {
    // No WebGL / module load failed — CSS gradient fallback stays. Done.
    return;
  }

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power' });
  } catch (e) {
    return; // gradient fallback
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
  renderer.setSize(innerWidth, innerHeight);
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 100);
  cam.position.z = 6;

  const N = particleCount();
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) pos[i] = (Math.random() - 0.5) * 16;

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));

  const accent = readAccentColor() || 0x6fb6ff;
  const mat = new THREE.PointsMaterial({ size: 0.022, color: accent, transparent: true, opacity: 0.5, depthWrite: false });
  const points = new THREE.Points(geo, mat);
  scene.add(points);

  // thin connecting lines between near neighbours (capped for perf)
  const lines = buildLines(THREE, pos, N, accent);
  if (lines) scene.add(lines);

  Object.assign(state, { renderer, scene, cam, points, lines, mat, N });

  addEventListener('pointermove', onPointer, { passive: true });
  addEventListener('resize', onResize, { passive: true });
  document.addEventListener('visibilitychange', onVisibility);

  if (motionAllowed()) {
    start();
  } else {
    renderOnce(); // single static frame
  }
}

function readAccentColor() {
  try {
    const c = getComputedStyle(document.documentElement).getPropertyValue('--accent').trim();
    if (c.startsWith('#')) return parseInt(c.slice(1), 16);
  } catch (e) {}
  return null;
}

/** Build a capped LineSegments mesh connecting close particles. */
function buildLines(THREE, pos, N, color) {
  const maxLines = 700;
  const maxDist2 = 0.7 * 0.7;
  const segs = [];
  // Sample pairs sparsely to keep this O(N) rather than O(N^2).
  for (let i = 0; i < N && segs.length < maxLines * 6; i++) {
    const ax = pos[i * 3], ay = pos[i * 3 + 1], az = pos[i * 3 + 2];
    for (let k = 1; k <= 3; k++) {
      const j = (i + k * 53) % N;
      const bx = pos[j * 3], by = pos[j * 3 + 1], bz = pos[j * 3 + 2];
      const dx = ax - bx, dy = ay - by, dz = az - bz;
      if (dx * dx + dy * dy + dz * dz < maxDist2) {
        segs.push(ax, ay, az, bx, by, bz);
        if (segs.length >= maxLines * 6) break;
      }
    }
  }
  if (!segs.length) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(segs), 3));
  const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.08 });
  return new THREE.LineSegments(g, m);
}

function onPointer(e) {
  state.mx = (e.clientX / innerWidth - 0.5);
  state.my = (e.clientY / innerHeight - 0.5);
}

function onResize() {
  if (!state.renderer) return;
  state.cam.aspect = innerWidth / innerHeight;
  state.cam.updateProjectionMatrix();
  state.renderer.setSize(innerWidth, innerHeight);
  if (!state.running) renderOnce();
}

function onVisibility() {
  if (document.hidden) {
    stop();
  } else if (motionAllowed()) {
    start();
  }
}

function renderOnce() {
  if (state.renderer) state.renderer.render(state.scene, state.cam);
}

function loop(t) {
  if (document.hidden || !state.running) return;
  state.raf = requestAnimationFrame(loop);

  // adaptive degrade: if frames are consistently slow, thin the field once.
  if (state.lastT) {
    const dt = t - state.lastT;
    if (dt > 24) state.slowFrames++; else state.slowFrames = Math.max(0, state.slowFrames - 1);
    if (state.slowFrames > 45 && !state.degraded) degrade();
  }
  state.lastT = t;

  const p = state.points, cam = state.cam;
  p.rotation.y += 0.0007;
  p.rotation.x += 0.00028;
  if (state.lines) { state.lines.rotation.copy(p.rotation); }
  cam.position.x += (state.mx * 0.6 - cam.position.x) * 0.03;
  cam.position.y += (-state.my * 0.6 - cam.position.y) * 0.03;
  cam.lookAt(0, 0, 0);
  state.renderer.render(state.scene, cam);
}

/** Drop visual cost: hide lines, shrink point opacity. One-shot. */
function degrade() {
  state.degraded = true;
  if (state.lines) state.lines.visible = false;
  if (state.mat) state.mat.opacity = 0.4;
}

function start() {
  if (state.running || !state.renderer) return;
  state.running = true;
  state.lastT = 0;
  state.raf = requestAnimationFrame(loop);
}

function stop() {
  state.running = false;
  if (state.raf) cancelAnimationFrame(state.raf);
  state.raf = 0;
}

/** Public: react to settings changes (motion / disable-3D toggles). */
export function refresh() {
  if (!state.renderer) return;
  if (motionAllowed()) start();
  else { stop(); renderOnce(); }
}

export function initBg3d() {
  // Defer to idle so it never blocks first paint / lesson content.
  const go = () => init();
  if ('requestIdleCallback' in window) requestIdleCallback(go, { timeout: 1200 });
  else setTimeout(go, 300);
}
