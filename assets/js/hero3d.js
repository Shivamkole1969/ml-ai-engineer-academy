/* ============================================================================
   hero3d.js — the Home "Living Object": a translucent, glowing icosahedron
   distorted by a custom simplex-noise GLSL shader, with a fresnel rim and an
   UnrealBloom pass. Mouse-reactive, readiness-reactive, fully disposed on leave.
   Static-deploy friendly (three + three/addons via importmap, no build step).
   ========================================================================== */

let current = null; // the active instance's dispose fn

const REDUCE = () => matchMedia('(prefers-reduced-motion: reduce)').matches
  || document.documentElement.getAttribute('data-motion') === 'off';

/* GLSL: Ashima/Stefan Gustavson 3D simplex noise (public domain) */
const SNOISE = `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;

const VERT = `
uniform float uTime; uniform float uDistort; uniform vec2 uMouse;
varying float vN; varying vec3 vNormalW; varying vec3 vViewW;
${SNOISE}
void main(){
  float n = snoise(normal*1.6 + uTime*0.35);
  float n2 = snoise(position*2.6 - uTime*0.22);
  float amp = uDistort * (0.55 + 0.45*sin(uTime*0.5));
  vec3 disp = position + normal * (n*amp + n2*amp*0.4);
  // gentle pull toward mouse for a "reaching" feel
  disp.xy += uMouse * 0.10 * (0.5 + 0.5*n);
  vN = n;
  vec4 wp = modelMatrix * vec4(disp,1.0);
  vNormalW = normalize(mat3(modelMatrix) * normal);
  vViewW = normalize(cameraPosition - wp.xyz);
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const FRAG = `
precision highp float;
uniform vec3 uColorA; uniform vec3 uColorB; uniform float uGlow;
varying float vN; varying vec3 vNormalW; varying vec3 vViewW;
void main(){
  float fres = pow(1.0 - max(dot(normalize(vNormalW), normalize(vViewW)), 0.0), 2.4);
  vec3 base = mix(uColorA, uColorB, smoothstep(-0.6, 0.6, vN));
  vec3 col = base * (0.25 + 0.75*fres);          // glassy core, glowing rim
  col += base * fres * uGlow;                      // bloom-able rim energy
  float alpha = clamp(0.30 + fres*0.9, 0.0, 1.0);
  gl_FragColor = vec4(col, alpha);
}`;

function hexToRGB(hex, fallback) {
  const h = (hex || '').trim().replace('#', '');
  if (h.length !== 6) return fallback;
  return [parseInt(h.slice(0, 2), 16) / 255, parseInt(h.slice(2, 4), 16) / 255, parseInt(h.slice(4, 6), 16) / 255];
}
function cssVar(name, fb) { try { return getComputedStyle(document.documentElement).getPropertyValue(name) || fb; } catch (e) { return fb; } }

/**
 * Mount the living object into `el`. opts.readiness (0..100) scales the glow.
 * Returns a dispose() function. Falls back silently (no-op) if WebGL fails.
 */
export async function mountHero(el, opts = {}) {
  if (current) { current(); current = null; }      // only one at a time
  if (!el) return () => {};

  let THREE, EffectComposer, RenderPass, UnrealBloomPass, OutputPass;
  try {
    THREE = await import('three');
    ({ EffectComposer } = await import('three/addons/postprocessing/EffectComposer.js'));
    ({ RenderPass } = await import('three/addons/postprocessing/RenderPass.js'));
    ({ UnrealBloomPass } = await import('three/addons/postprocessing/UnrealBloomPass.js'));
    ({ OutputPass } = await import('three/addons/postprocessing/OutputPass.js'));
  } catch (e) {
    return () => {};                                // CSS aurora fallback stays
  }

  const isMobile = innerWidth < 760;
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ alpha: true, antialias: !isMobile, powerPreference: 'high-performance' });
  } catch (e) { return () => {}; }

  const size = () => Math.max(180, Math.min(el.clientWidth || 280, 360));
  let W = size(), H = W;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, isMobile ? 1.3 : 2));
  renderer.setSize(W, H);
  renderer.setClearColor(0x000000, 0);
  el.appendChild(renderer.domElement);
  renderer.domElement.style.cssText = 'width:100%;height:auto;display:block';

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
  cam.position.z = 4.2;

  const accentA = hexToRGB(cssVar('--accent', '#3ad6ff'), [0.23, 0.84, 1.0]);
  const accentB = hexToRGB(cssVar('--accent-2', '#8b7bff'), [0.55, 0.48, 1.0]);
  const readiness = Math.max(0, Math.min(100, opts.readiness || 0));

  const uniforms = {
    uTime: { value: 0 },
    uDistort: { value: 0.28 },
    uMouse: { value: new THREE.Vector2(0, 0) },
    uColorA: { value: new THREE.Vector3(...accentA) },
    uColorB: { value: new THREE.Vector3(...accentB) },
    uGlow: { value: 0.8 + (readiness / 100) * 1.6 },   // more complete → brighter
  };
  const geo = new THREE.IcosahedronGeometry(1.25, isMobile ? 12 : 24);
  const mat = new THREE.ShaderMaterial({
    uniforms, vertexShader: VERT, fragmentShader: FRAG,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  });
  const mesh = new THREE.Mesh(geo, mat);
  scene.add(mesh);

  // a faint inner solid for body
  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(1.05, 6),
    new THREE.MeshBasicMaterial({ color: 0x0a1024, transparent: true, opacity: 0.5 })
  );
  scene.add(core);

  // bloom
  let composer = null;
  try {
    composer = new EffectComposer(renderer);
    composer.setSize(W, H);
    composer.addPass(new RenderPass(scene, cam));
    const bloom = new UnrealBloomPass(new THREE.Vector2(W, H), isMobile ? 0.6 : 1.0, 0.6, 0.18);
    composer.addPass(bloom);
    composer.addPass(new OutputPass());
  } catch (e) { composer = null; }

  // interaction
  let mx = 0, my = 0, tmx = 0, tmy = 0;
  const onMove = (e) => {
    const r = el.getBoundingClientRect();
    tmx = ((e.clientX - r.left) / r.width - 0.5) * 2;
    tmy = -((e.clientY - r.top) / r.height - 0.5) * 2;
  };
  addEventListener('pointermove', onMove, { passive: true });

  const onResize = () => {
    W = size(); H = W;
    renderer.setSize(W, H);
    if (composer) composer.setSize(W, H);
  };
  const ro = ('ResizeObserver' in window) ? new ResizeObserver(onResize) : null;
  if (ro) ro.observe(el);
  addEventListener('resize', onResize);

  let raf = 0, alive = true, t0 = performance.now();
  const render = () => (composer ? composer.render() : renderer.render(scene, cam));

  function loop(t) {
    if (!alive) return;
    raf = requestAnimationFrame(loop);
    if (document.hidden) return;
    uniforms.uTime.value = (t - t0) / 1000;
    mx += (tmx - mx) * 0.045; my += (tmy - my) * 0.045;   // heavy damping = weighty feel
    uniforms.uMouse.value.set(mx, my);
    mesh.rotation.y += 0.0016; mesh.rotation.x = my * 0.3; mesh.rotation.z = mx * 0.15;
    core.rotation.copy(mesh.rotation);
    cam.position.x += (mx * 0.5 - cam.position.x) * 0.04;
    cam.position.y += (my * 0.5 - cam.position.y) * 0.04;
    cam.lookAt(0, 0, 0);
    render();
  }

  if (REDUCE()) { uniforms.uTime.value = 2.0; render(); }
  else raf = requestAnimationFrame(loop);

  const dispose = () => {
    alive = false; if (raf) cancelAnimationFrame(raf);
    removeEventListener('pointermove', onMove);
    removeEventListener('resize', onResize);
    if (ro) ro.disconnect();
    geo.dispose(); mat.dispose(); core.geometry.dispose(); core.material.dispose();
    if (composer) composer.dispose?.();
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    if (current === dispose) current = null;
  };
  current = dispose;
  return dispose;
}

/** Dispose any active hero (call on route change away from Home). */
export function disposeHero() { if (current) { current(); current = null; } }
