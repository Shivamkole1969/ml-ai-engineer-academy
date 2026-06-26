# 🛰️ ML/AI ENGINEER ACADEMY — Build Spec for Claude Code

> **This file is a build brief, not the final app README.**
> Drop this file at the root of an empty repo, open Claude Code in VS Code, and say:
> *"Read README.md fully, then scaffold and build this project phase by phase. Start with Phase 0."*
>
> Claude Code: **read this entire document before writing any code.** Then:
> 1. Move this spec to `docs/BUILD_SPEC.md` (keep it as the source of truth).
> 2. Generate a **fresh `README.md`** at the repo root containing the Hugging Face Spaces front-matter from §17.
> 3. Build in the phases defined in §16, committing after each phase.

---

## 0. TL;DR — what we are building

A **single-page, offline-capable web app** that teaches **production-grade ML/AI engineering** from first principles to interview/on-call mastery, deployed **free** on a **Hugging Face Static Space**. It is the owner's *only* study resource, so it must be **complete, practical, code-rich, and a joy to use** — never heavy, never boring.

- **Audience:** one learner (the repo owner), targeting **AI/ML + GenAI Engineer roles in Pune and remote-India**, ~2 years experience level, who needs to *truly* learn topics currently only on the résumé (RAG, Agentic RAG, PEFT/LoRA/QLoRA, vector DBs, GraphRAG, agents, MCP, AWS/GCP/Snowflake GenAI).
- **The single job of the app:** turn a strong-but-uneven engineer into someone *interview-ready and work-ready* by reading + interacting, with zero paid dependencies.
- **Signature feel:** an **engineering mission-control HUD** — a calm dark "deep space" canvas with a subtle 3D neural/particle field, glass instrument panels, one electric accent, monospaced telemetry, gamified progress. Energetic and motivating, **but easy on the eyes and never nauseating** (motion is subtle and fully respects `prefers-reduced-motion`).

The full source teaching material is the attached **`The ML/AI Engineer Handbook 2026`** (11 chapters). **Place that PDF at `reference/handbook-2026.pdf`** so content can be authored at full depth. This spec extends it with the GenAI/agentic/cloud tracks the market demands.

---

## 1. Non-negotiables (read twice)

1. **Free forever.** Static Space, CDN libraries, no backend, no API keys required to *use* the app. Every cloud/platform concept is taught with **copy-paste code snippets** so the learner never has to pay a platform to understand it.
2. **Content lives in the app.** Lessons, snippets, diagrams, FAQs, system designs — all bundled. Works offline after first load.
3. **Comfortable pace, low cognitive load.** Short scrollable lessons, one idea per screen-block, generous whitespace, progressive disclosure. No walls of text.
4. **Practical + production-grade.** Every concept answers *"why does this matter in production?"* and ends with an interview-ready line.
5. **Hot-topic priority is visible.** Every track/lesson carries a priority badge (§3) so the learner spends energy where Pune/remote-India jobs actually pay.
6. **Accessibility floor:** responsive to mobile, visible keyboard focus, reduced-motion honored, WCAG-AA contrast, semantic HTML. Build this silently, don't announce it.
7. **Progress is saved** (localStorage) — completion, streaks, XP, bookmarks, notes — and survives reloads.

---

## 2. Market context that drives priorities (Pune + remote India, 2026)

Authored from current market research — encode this into the app's roadmap copy and badges.

- AI-linked hiring growing ~32% in 2026 toward ~1M+ roles; **~53% talent gap** — the largest of any tech discipline. The bar is *"can you ship AI safely at scale and keep it alive in production,"* not *"can you train a model in a notebook."*
- **Pune ≈ 15% of postings** (3rd nationally); **~30% of postings are remote-friendly.** Pune clusters: GenAI/LLM/RAG product teams (e.g. services + product firms), fintech, and automotive/ADAS computer-vision (KPIT, Tata Elxsi, Bosch).
- **Biggest premium & demand-supply gap:** GenAI / LLM / **RAG** / **Agentic AI** / **MLOps** (+20–40% on offers). GenAI now appears in 40%+ of ML JDs.
- **PyTorch ≫ TensorFlow** at product companies. **SQL is table-stakes.**
- Pune JDs explicitly request: **RAG pipelines (chunking, retrieval, context optimization), vector DBs (Pinecone/FAISS/Weaviate/Milvus/pgvector), LangChain/LlamaIndex/LangGraph/CrewAI/AutoGen, MCP, embeddings & semantic search, evaluation/benchmarking, Docker/K8s, AWS/GCP/Azure (pick one deep).**
- **Portfolio > certificates.** Three deployed projects move interviews most: an end-to-end ML pipeline, a RAG app, an MLOps deployment. (See Track 18.)

**Implication for this app:** classic ML stays as *foundation* (table-stakes, taught well but tightly). The **energy and depth go into GenAI, RAG/Agentic RAG, fine-tuning, agents/MCP, serving, MLOps/monitoring, cloud GenAI, and ML system design.**

---

## 3. Priority legend (use as badges on every track + lesson)

| Badge | Meaning | Treatment |
|---|---|---|
| 🔥 **HOT-2026** | Highest hiring demand / salary premium | Most depth, most snippets, most interview drills, flagged on the dashboard |
| ⭐ **CORE** | Expected of any ML engineer; commonly tested | Full depth, solid drills |
| 📎 **FOUNDATION** | Must understand, rarely the headline | Tight, clear, fewer drills |
| 🧊 **SKIM** | Low 2026 relevance; awareness only | One short "what & why you can mostly ignore" card |

Render the badge as a small glass pill with an icon. The dashboard shows a **"Focus first"** filter that surfaces only 🔥 + ⭐.

---

## 4. Tech stack (chosen for free static hosting + rich UI)

**Deploy target:** Hugging Face **Static Space** (serves `index.html` + assets; no server, no build server needed).

**Core (all via CDN — no npm/build required, keeps Claude Code's job simple and HF happy):**
- **Vanilla JS (ES modules)** + a tiny hash-router (no framework lock-in, fast, debuggable). *If Claude Code strongly prefers a build step, Vite + preact is acceptable, but the committed output must be plain static files HF can serve from root.*
- **Three.js** (CDN, `importmap`) — the subtle 3D background only. Lazy-init, pauses when tab hidden, disabled under reduced-motion.
- **Tailwind CSS via CDN** (Play CDN) for utility styling, **plus a hand-written `theme.css`** for the design tokens, glassmorphism, and signature elements (do not rely on Tailwind alone for the identity — see §6).
- **marked.js** (Markdown → HTML for lesson content) + **DOMPurify** (sanitize).
- **highlight.js** (syntax highlighting) with a **custom copy-to-clipboard button** on every code block.
- **KaTeX** (CDN) for the few math expressions (bias²+variance, O(n²), 16·N rule, ECE).
- **lunr.js** (CDN) for client-side full-text search / command palette.
- Optional, behind a lazy "Run it" button on ≤3 demos only: **Pyodide** (Python in browser). Keep optional — do not block lessons on it.

**State:** `localStorage` (progress, XP, streaks, bookmarks, notes, theme, reduced-motion override, last-position). Wrap in a small `store.js` with try/catch and a versioned schema key (`mlacademy.v1`).

**No analytics, no trackers, no external calls at runtime** except CDN asset fetches.

> ⚠️ HF note: A Static Space requires **`index.html` at repo root** and a `README.md` with `sdk: static` front-matter (§17). CDN `<script>`/`importmap` works on HF static. Pyodide also works but is large — lazy-load it.

---

## 5. Information architecture & navigation

**Layout = "mission control":**

```
┌───────────────────────────────────────────────────────────────┐
│  TOP BAR: logo ‹ML/AI Academy›  | search ⌘K | streak🔥 | XP | ☼/☾│
├───────────┬───────────────────────────────────────────────────┤
│  SIDEBAR  │   MAIN PANEL (glass)                               │
│  (tracks, │   • Lesson reader / Dashboard / System Design /     │
│  collapsi-│     Interview Mode / Projects                       │
│  ble,     │   • reading-progress bar pinned to top of panel     │
│  shows %  │   • prev / next lesson footer with "mark complete"  │
│  per      │                                                     │
│  track)   │   RIGHT RAIL (optional, desktop): on-this-page TOC, │
│           │   bookmark, personal notes drawer                   │
└───────────┴───────────────────────────────────────────────────┘
        3D neural/particle field sits BEHIND everything, dimmed.
```

**Routes (hash router):**
- `#/` Dashboard (progress rings, streak, "Focus first", "Continue where you left off", next recommended lesson, hot-topic spotlight)
- `#/track/:trackId` Track overview (lesson list + % + badges)
- `#/lesson/:lessonId` Lesson reader
- `#/system-design` and `#/system-design/:archId`
- `#/interview` Interview Mode (FAQ-100 + flashcards + mock drill)
- `#/projects` Portfolio projects
- `#/playground` (optional) the interactive widgets index
- `#/glossary` quick term lookup

**Command palette (⌘K / Ctrl-K):** fuzzy search across all lessons, FAQs, system designs, widgets, glossary; arrow-key nav; Enter to jump. This is a signature interaction — make it feel instant.

**Navigation principles:** never more than 2 clicks to any concept; "Continue" always returns to last scroll position; sidebar remembers expanded/collapsed state.

---

## 6. Design system & visual identity

**Ground the look in the subject: a calm, premium engineering HUD for someone who ships models.** Spend boldness in ONE place — the 3D field + the glass panels + the single electric accent — and keep everything else quiet and disciplined. Avoid the three templated "AI-generated" looks (cream+serif+terracotta; near-black + single acid accent used carelessly; broadsheet hairline columns).

### 6.1 Palette (define as CSS variables in `theme.css`)
Dark is default (eye-comfort); light theme is a toggle.

```css
:root {
  /* deep-space canvas */
  --bg-900:#070b18;  --bg-800:#0b1226;  --bg-700:#111a33;
  /* glass surfaces */
  --glass-bg:rgba(20,30,58,0.45);
  --glass-stroke:rgba(140,170,255,0.18);
  --glass-hi:rgba(255,255,255,0.06);
  /* text */
  --ink-100:#eaf0ff; --ink-300:#aebbdf; --ink-500:#7e8cb8;
  /* ONE energetic accent: electric cyan → violet */
  --accent:#3ad6ff; --accent-2:#8b7bff; --accent-grad:linear-gradient(120deg,#3ad6ff,#8b7bff);
  /* semantic */
  --hot:#ff7a59; --core:#3ad6ff; --foundation:#7e8cb8; --skim:#516089;
  --ok:#48e0a0; --warn:#ffd166; --bad:#ff6b8a;
  --radius:18px; --radius-sm:12px;
  --shadow-glass:0 8px 40px rgba(2,8,30,0.45);
}
[data-theme="light"]{ /* a soft "daylight HUD": pale slate, same accent, reduced glow */
  --bg-900:#eef2fb; --bg-800:#e7edfa; --bg-700:#dde6f7;
  --glass-bg:rgba(255,255,255,0.6); --glass-stroke:rgba(60,90,180,0.18);
  --ink-100:#0c1530; --ink-300:#3a4a78; --ink-500:#6376a3;
}
```

### 6.2 Typography (deliberate pairing — not the same families used everywhere)
Load via Google Fonts CDN.
- **Display / headings:** **"Space Grotesk"** (technical, slightly geometric, characteristic). Use with restraint at large sizes.
- **Body / UI:** **"Inter"** (clean, legible at small sizes).
- **Code / telemetry / numbers:** **"JetBrains Mono"** — also used for stat readouts, badges, the VRAM/KV calculators (reinforces the "instrument" feel).
- Type scale (rem): 0.78 / 0.875 / 1 / 1.125 / 1.375 / 1.75 / 2.5 / 3.5. Tight letter-spacing on display, comfortable line-height (1.65) on body.

### 6.3 Glassmorphism panels (the recurring surface)
```css
.glass{
  background:var(--glass-bg);
  border:1px solid var(--glass-stroke);
  border-radius:var(--radius);
  backdrop-filter:blur(16px) saturate(135%);
  -webkit-backdrop-filter:blur(16px) saturate(135%);
  box-shadow:var(--shadow-glass), inset 0 1px 0 var(--glass-hi);
}
.glass:hover{ border-color:rgba(140,170,255,0.32); }
```
Use glass for: main panel, sidebar, cards, callouts, the command palette, calculators. **Add a hairline gradient top-border** on "hero" cards using `--accent-grad`. Provide a **fallback** for browsers without `backdrop-filter` (solid `--bg-700`).

### 6.4 The 3D background (subtle, the "atmosphere", not a distraction)
Three.js scene behind the UI (`z-index:-1`, `position:fixed`, `pointer-events:none`):
- A **slowly drifting particle/neural field**: ~1,200–2,500 points (scale down on mobile/low-DPR), faint accent-colored, gentle parallax tied to pointer + scroll, ultra-slow rotation. Optionally thin connecting lines between near neighbors (cap count for perf).
- **Performance + comfort rules (mandatory):**
  - Respect `prefers-reduced-motion: reduce` → render a single static frame (or a CSS gradient fallback), no animation.
  - Add a manual **"Reduce motion / disable 3D"** toggle in settings; persist it.
  - Pause `requestAnimationFrame` when `document.hidden`.
  - Cap DPR at ~1.5; drop particle count if `navigator.hardwareConcurrency` is low or frame time > 24ms (simple adaptive degrade).
  - Keep opacity low (it's atmosphere) so text contrast always wins.
- A clean CSS-only animated gradient fallback if WebGL is unavailable.

Minimal starter (Claude Code to expand, modularize, and add adaptive degrade):
```html
<div id="bg3d"></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const el = document.getElementById('bg3d');
const renderer = new THREE.WebGLRenderer({alpha:true, antialias:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
renderer.setSize(innerWidth, innerHeight); el.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 100); cam.position.z = 6;
const N = innerWidth < 700 ? 900 : 1800;
const pos = new Float32Array(N*3);
for(let i=0;i<N*3;i++) pos[i] = (Math.random()-0.5)*16;
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos,3));
const mat = new THREE.PointsMaterial({size:0.02, color:0x6fb6ff, transparent:true, opacity:0.55});
const pts = new THREE.Points(geo, mat); scene.add(pts);
let mx=0,my=0; addEventListener('pointermove', e=>{mx=(e.clientX/innerWidth-0.5); my=(e.clientY/innerHeight-0.5);});
addEventListener('resize', ()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();renderer.setSize(innerWidth,innerHeight);});
function loop(){
  if(document.hidden) return requestAnimationFrame(loop);
  pts.rotation.y += 0.0008; pts.rotation.x += 0.0003;
  cam.position.x += (mx*0.6 - cam.position.x)*0.03;
  cam.position.y += (-my*0.6 - cam.position.y)*0.03; cam.lookAt(0,0,0);
  renderer.render(scene,cam); requestAnimationFrame(loop);
}
if(reduce){ renderer.render(scene,cam); } else { loop(); }
</script>
```

### 6.5 Motion language (deliberate, never nauseating)
- Page/lesson transitions: 180–240ms ease-out fade + 6px rise. No parallax scrolling of text. No auto-playing carousels.
- Hover micro-interactions on cards/buttons only (subtle lift + border glow).
- One orchestrated "page-load reveal" on the dashboard (staggered card fade-in ~40ms apart). Everything else stays calm.
- XP/level-up: a small, satisfying glass "toast" + a brief accent pulse — celebratory but quick.
- **All motion gated by reduced-motion.**

### 6.6 Signature elements (what makes it memorable)
1. **The ⌘K command palette** as the primary nav verb.
2. **Live "instrument" widgets** (VRAM calc, KV-cache calc, token/cost meter, throughput–latency dial) styled like cockpit gauges in JetBrains Mono.
3. **Progress as fuel gauges / rings** per track, with a global "Readiness %" on the dashboard.

### 6.7 Reusable content components (build as JS render helpers)
A lesson is assembled from these typed blocks (mirrors the handbook's rhythm):
- `concept` — short prose intro.
- `why-prod` — "Why it matters in production" callout (accent left-border).
- `table` — responsive; collapses to cards on mobile.
- `diagram` — inline SVG or a simple flow component (e.g. the 5-step training loop, the agent loop, the eval funnel, the rollout ladder). Prefer crisp SVG over images.
- `gotcha` — ⚠️ warning card (warn color).
- `war-story` — ⚔️ incident card (distinct style; these are gold for retention).
- `interview-line` — ★ "the line that lands" (accent card, quotable).
- `code` — titled, language-tagged, copy button, syntax-highlighted; optional "Run it" (Pyodide) only where specified.
- `qa` — collapsible interview Q→A.
- `drill` — a 1–3 question self-check (instant feedback, no grading server).
- `key-takeaway` — ✓ pinned summary at lesson end.

---

## 7. File / folder structure

```
/
├─ index.html                  # entry (importmaps, CDN libs, mount points)
├─ README.md                   # HF Space front-matter (generated from §17)
├─ .gitattributes              # (if needed for HF; PDF via LFS optional)
├─ assets/
│  ├─ css/theme.css            # tokens, glass, typography, components
│  ├─ js/
│  │  ├─ main.js               # bootstrap, router, layout
│  │  ├─ router.js             # hash router
│  │  ├─ store.js              # localStorage (versioned)
│  │  ├─ bg3d.js               # Three.js field + adaptive degrade
│  │  ├─ search.js             # lunr index + command palette
│  │  ├─ render.js             # markdown + block components → DOM
│  │  ├─ gamify.js             # XP, streak, badges, readiness
│  │  ├─ widgets/              # interactive calculators (one file each)
│  │  │  ├─ vram.js  kvcache.js  tokencost.js  biasvar.js
│  │  │  ├─ calibration.js  throughput.js  agenterror.js
│  │  └─ components.js         # block renderers (qa, drill, war-story…)
│  └─ svg/                     # inline diagrams (training loop, agent loop…)
├─ content/
│  ├─ curriculum.json          # ordered tracks → lessons (ids, titles, badges)
│  ├─ tracks/<trackId>/<lessonId>.md   # lesson source (frontmatter + blocks)
│  ├─ system-design/<archId>.md
│  ├─ faq/faq-100.json
│  ├─ flashcards.json
│  └─ projects/<projectId>.md
├─ reference/
│  └─ handbook-2026.pdf        # owner places the attached PDF here
└─ docs/
   └─ BUILD_SPEC.md            # this file, moved here
```

---

## 8. Content model (how a lesson is stored & rendered)

Lessons are **Markdown with YAML frontmatter + lightweight block fences** so authoring stays fast and Claude Code can generate many. `render.js` parses frontmatter, then renders Markdown via marked, upgrading custom fenced blocks into the §6.7 components.

**Lesson file frontmatter:**
```yaml
---
id: llm-kv-cache
track: 07-llm-foundations
title: "KV Cache — the real inference memory bill"
badge: HOT          # HOT | CORE | FOUNDATION | SKIM
minutes: 9
prereqs: [llm-attention-context]
tags: [inference, memory, gqa, paged-attention]
xp: 60
hot2026: true
---
```

**Custom block fences inside the body** (Claude Code: implement these parsers):
````
:::why-prod
Weights are a fixed cost; the KV cache grows with sequence length × concurrency...
:::

:::table {title="KV optimizations"}
| Idea | What it does | Tradeoff |
| MQA/GQA | share K/V across heads/groups | big cache cut, tiny quality cost |
:::

:::gotcha
Don't trust perplexity alone after KV quantization — re-run your real eval.
:::

:::war-story {title="The p99 that wasn't the p50"}
A team shipped great median latency...
:::

:::interview-line
"The KV cache, not the weights, gates concurrency — that's what GQA, PagedAttention, and KV-quant attack."
:::

```python {title="Estimate KV cache bytes" run=false}
def kv_bytes(layers, heads, head_dim, seq, batch, dtype_bytes=2):
    return 2 * layers * heads * head_dim * seq * batch * dtype_bytes
```

:::qa {q="Why does the KV cache, not the weights, often limit throughput?"}
Weights load once; the cache grows with seq length and concurrent requests...
:::

:::drill {type="mcq" q="Doubling context roughly does what to the attention score matrix?"}
- [ ] Doubles it
- [x] Quadruples it (O(n²))
- [ ] No change
:::

:::widget {name="kvcache"}   # mounts assets/js/widgets/kvcache.js
:::

:::key-takeaway
KV cache ≈ 2·layers·heads·head_dim·seq·batch·dtype. It, not weights, is the long-context bottleneck.
:::
````

Mark lesson **complete** → award `xp`, advance streak, update track ring and global Readiness %.

---

## 9. FULL CURRICULUM (the index Claude Code must populate)

Author every lesson from the handbook PDF (`reference/handbook-2026.pdf`) + this spec, in the handbook's punchy, production-grade voice (concept → why-prod → table/diagram → gotcha → war-story → interview-line → Q&A → drill → key-takeaway). **Tracks 1–8, 12–14 map to handbook chapters 1–8, 10–12. Tracks 9, 10, 11, 15, 16 are new (the résumé-gap + market-hot material) — author from this spec + the snippet inventory in §13.** Handbook Chapter 9 is intentionally absent; mirror that (skip).

Legend badges in brackets.

### Track 01 — Core ML Stack & Systems Thinking ⭐ (Handbook Ch.1)
1. Systems thinking: a model is a tensor program [⭐]
2. PyTorch vs JAX — pick on purpose [⭐]
3. What `.fit()` hides — the 5-step training loop [⭐]
4. **GPU memory math & the 16-bytes/param rule** [🔥] → `widget: vram`
5. Mixed precision: fp32 / fp16 / bf16 / fp8 (why bf16 won) [🔥]
6. Kernels, fusion, `torch.compile`, FlashAttention [🔥]
7. Profiling & the on-call debugging table (OOM, NaN, low util) [⭐]

### Track 02 — Data Reality & Failure Modes ⭐ (Ch.2)
1. The upstream principle (most failures are data failures) [⭐]
2. Where data comes from & its built-in distortion [⭐]
3. How data lies: selection, survivorship, feedback loops, Simpson's, position bias, MNAR [🔥]
4. Drift kept distinct: covariate vs label vs concept [🔥]
5. Label degradation: agreement, guideline drift, soft labels [⭐]
6. **Leakage — the silent score inflator** + the one-sentence test [🔥]
7. Shortcuts & spurious correlations + defenses [⭐]
8. The debugging order (data→labels→splits→features→model) [⭐]

### Track 03 — Statistics That Actually Matter ⭐ (Ch.3)
1. Bias–variance + the learning-curve diagnostic [⭐] → `widget: biasvar`
2. Confidence intervals & the bootstrap [⭐]
3. **Calibration: reliability diagrams, ECE, temperature/Platt/isotonic** [🔥] → `widget: calibration`
4. Why "95% accuracy" lies: base rates, P/R/F1, ROC-AUC vs PR-AUC [🔥]
5. Detecting distribution shift: PSI, KS, Wasserstein, MMD [⭐]

### Track 04 — Losses & Objective Design ⭐ (Ch.4)
1. The loss zoo & the behavior each induces [⭐]
2. Gradients tell the story (CE vs focal) [⭐]
3. Imbalance & regularizing the objective (class weight, focal, resampling, label smoothing) [⭐]
4. Misaligned losses: the proxy gap (how bad products are born) [🔥]

### Track 05 — Evaluation Beyond Benchmarks ⭐ (Ch.5)
1. Kaggle metrics vs real-world metrics [⭐]
2. Offline vs online & the evaluation funnel [⭐]
3. A/B testing traps (peeking, novelty, interference, SRM) [⭐]
4. Model regression tests: golden sets & per-slice gates [🔥]
5. Goodhart's law & defenses [⭐]
6. Decision-driven eval (work backwards from the decision) [🔥]

### Track 06 — Training at Scale ⭐ (Ch.6)
1. Parallelism taxonomy: DDP / FSDP / ZeRO / TP / PP [🔥]
2. The DDP→FSDP→TP→PP decision (how to choose, out loud) [🔥]
3. Gradient accumulation (+ the two traps) [⭐]
4. Checkpointing that survives a crash (resumable ≠ weights-only) [⭐]
5. Reproducibility and its limits [📎]
6. Preventing multi-day run failures (elastic, fault tolerance) [⭐]

### Track 07 — LLM Foundations & Tradeoffs 🔥 (Ch.7)
1. Tokenization — the layer everyone skips then debugs [🔥] → `widget: tokencost`
2. Attention & context limits — the O(n²) wall [🔥]
3. **KV cache — the real inference memory bill** (MQA/GQA, PagedAttention, KV-quant, prefix caching) [🔥] → `widget: kvcache`
4. LoRA vs fine-tune vs RAG — the decision you make constantly [🔥]
5. Where hallucinations originate & layered mitigations [🔥]

### Track 08 — Inference & Serving Systems 🔥 (Ch.8)
1. Batching: static → dynamic → continuous (in-flight) [🔥] → `widget: throughput`
2. Streaming & the two latencies: TTFT vs TPOT [🔥]
3. Quantization: INT8/INT4/FP8, PTQ vs QAT, GPTQ/AWQ [🔥]
4. vLLM — the reference serving engine [🔥]
5. Cold starts & the scale-to-zero trap [⭐]
6. GPU vs CPU — choose by workload [📎]
7. Why serving is harder than training (the tail is the product) [⭐]

### Track 09 — Generative AI Engineering & RAG 🔥 (NEW — top market priority + résumé gap)
1. LLM app fundamentals: model APIs, prompting, structured outputs, tool/function calling [🔥]
2. Embeddings & semantic search (what an embedding is, cosine sim, ANN) [🔥]
3. **Vector databases**: FAISS, ChromaDB, pgvector, Pinecone, Weaviate/Milvus — when to use which [🔥]
4. The RAG pipeline: load → chunk → embed → index → retrieve → rerank → generate → cite [🔥]
5. Chunking strategies & context optimization (size/overlap, semantic, parent-doc, late chunking) [🔥]
6. Reranking & hybrid search (BM25 + dense + cross-encoder rerank) [🔥]
7. **Advanced RAG**: query rewriting, **Corrective RAG (CRAG)**, **Self-RAG**, **Agentic RAG** [🔥]
8. **GraphRAG & knowledge graphs** (Neo4j / AWS Neptune; entities, relations, multi-hop) [🔥]
9. **RAG evaluation** (RAGAS: faithfulness, answer relevancy, context precision/recall; groundedness & citation checks) [🔥]
10. Orchestration frameworks: LangChain vs LlamaIndex vs LangGraph — what each is for [🔥]
11. Production RAG concerns: freshness, chunk drift, cost (token bloat), guardrails, caching [🔥]

### Track 10 — Fine-Tuning & Model Adaptation 🔥 (NEW — résumé gap: PEFT/LoRA/QLoRA)
1. When to prompt vs RAG vs fine-tune (decision framework) [🔥]
2. SFT & instruction tuning: data format & data quality [🔥]
3. **PEFT family**: LoRA, QLoRA, adapters, prefix/prompt tuning — what actually changes [🔥]
4. Hands-on: HF `transformers` + `peft` + `trl` (`SFTTrainer`), 4-bit (bitsandbytes) [🔥]
5. Evaluating a fine-tune; catastrophic forgetting; merging adapters; serving swappable LoRAs [🔥]
6. Alignment overview (conceptual): RLHF, DPO, why "helpfulness-only" rewards hallucinate [⭐]
7. Distillation as adaptation (student/teacher) [⭐]

### Track 11 — Agents, Tools & MCP 🔥 (NEW — résumé gap + Ch.12 agent material)
1. The agent loop (reason → tool → observe → loop/stop) & where each part breaks [🔥]
2. **The compounding-error math** (0.95¹⁰ ≈ 60%) → `widget: agenterror` [🔥]
3. Tool calling, memory, planning, multi-agent — capabilities & failure modes [🔥]
4. Frameworks: LangGraph vs CrewAI vs AutoGen — when to reach for which [🔥]
5. **Model Context Protocol (MCP)**: clients, servers, tools/resources; why it's a 2026 standard [🔥]
6. Designing agents that survive production (bound the loop, idempotency, validate tool output, human-in-loop, full tracing) [🔥]
7. Agent evaluation & observability (traces, tool-success rate, task success) [🔥]

### Track 12 — Pipelines & Data Infrastructure ⭐ (Ch.10)
1. Feature stores: offline vs online, materialization, point-in-time join [🔥]
2. When you actually need a feature store (and when a SQL job is plenty) [⭐]
3. Offline vs online data — the same value, two clocks [⭐]
4. Backfills & point-in-time correctness (where the future leaks into the past) [🔥]
5. Late-arriving events, watermarks, windowing [⭐]
6. Schema evolution (backward/forward/breaking; the silent unit-change bug) [⭐]
7. Broken joins & silent corruption; data contracts [🔥]

### Track 13 — Monitoring, Optimization & Cost Control 🔥 (Ch.11)
1. The four monitoring layers (operational/data/model/business) & ground-truth lag [⭐]
2. Drift detection in production (reference→compare→alert; killing alert noise) [⭐]
3. Latency, outliers & the tail (never alert on averages; percentiles don't average) [🔥]
4. **LLM-specific monitoring**: token spend, hallucination rate, silent decay, provider/prompt drift [🔥]
5. Cost toolkit: distillation, pruning, quantization, **caching (exact/semantic/prefix)**, prompt compression, model routing [🔥]
6. A sane cost-reduction order of operations (measure → cache → compress → route → quantize → distill) [🔥]

### Track 14 — Production, Safety & Org Reality 🔥 (Ch.12)
1. Agents recap as a production system (links to Track 11) [⭐]
2. **LLM security/threat model**: prompt injection, indirect injection, exfiltration, tool misuse, jailbreak + defenses [🔥]
3. The core security principle: architecture beats wording (least privilege, isolate untrusted, gate irreversible actions) [🔥]
4. Deployment: shadow → canary → progressive → rollback (and why model rollback is harder than code) [🔥]
5. Distributed-systems discipline: queues, retries+backoff, idempotency, backpressure, the unbounded-queue trap [🔥]
6. Documentation that makes a system operable: model cards, data contracts, eval reports, ADRs, runbooks [⭐]

### Track 15 — Cloud GenAI Platforms (free-tier oriented) 🔥 (NEW — résumé gap: AWS/GCP/Snowflake)
> Every lesson includes **a free way to learn it** (free tier, local/open-source equivalent, or emulator) plus runnable snippets. Pick ONE cloud to go deep; teach all so the learner can map across.
1. The lay of the land: managed model APIs vs training vs RAG vs agents on cloud [⭐]
2. **AWS Bedrock**: model access, Knowledge Bases (managed RAG), Agents, Guardrails [🔥]
3. **AWS SageMaker**: training jobs, endpoints (real-time/serverless), Pipelines; + S3/EC2 basics [⭐]
4. **GCP Vertex AI**: training, endpoints, Vertex RAG Engine, Agent Builder [🔥]
5. **Gemini API & Gemini Enterprise**: models, grounding, function calling, enterprise search [🔥]
6. **BigQuery ML**: train/predict in SQL (great free-tier learning) [⭐]
7. **Snowflake Cortex**: LLM functions, Cortex Search, Cortex Analyst; Snowpark basics [⭐]
8. MLOps glue: **MLflow** (tracking/registry), **Docker**, **FastAPI** model serving, a note on **Terraform (IaC)** [🔥]
9. Cost & free-tier survival guide (how to practice each platform for ₹0) [⭐]

### Track 16 — ML/AI System Design 🔥 (NEW — explicitly requested; see §11)

### Track 17 — Interview Mode: FAQ-100 + Flashcards + Mock Drill 🔥 (see §12)

### Track 18 — Projects & Portfolio ⭐ (see §14)

> **Coverage check (so nothing the owner faked is missed):** Agentic & Corrective RAG ✔(T9), GraphRAG/Neo4j/Neptune ✔(T9), LlamaIndex/LangChain/LangGraph ✔(T9/T11), MCP ✔(T11), PEFT/LoRA/QLoRA ✔(T10), BERT/embeddings ✔(T9/T10), AWS Bedrock/SageMaker ✔(T15), GCP Vertex/BigQuery ML/Gemini Enterprise ✔(T15), Snowflake Cortex/Snowpark ✔(T15), vector DBs (Pinecone/FAISS/Chroma/pgvector) ✔(T9), MLflow/Docker/FastAPI/Terraform ✔(T15), prompt engineering/context & memory ✔(T9/T11), XGBoost/CNN/classic ML ✔(T1–T5), Power BI/Tableau ➜ note as adjacent in T18 portfolio (not core to author deeply). 🧊-skim anything genuinely stale (e.g. standalone prompt-engineer role, pure-TensorFlow-first paths).

---

## 10. Interactive widgets (the "instruments") — specs & formulas

Each is a self-contained module mounted by a `:::widget` block, styled as a glass cockpit gauge, fully keyboard-accessible, with live readouts in JetBrains Mono.

1. **VRAM calculator (`vram`)** — inputs: params N (e.g. 7e9), precision, optimizer (Adam/SGD), train vs infer. Outputs each consumer + total. Use the 16·N rule for mixed-precision Adam training (2N fp16 w + 2N fp16 g + 4N fp32 master + 4N+4N Adam m/v); inference fp16 ≈ 2N, int4 ≈ 0.5N. Show "fits on 1×80GB?" verdict + suggested fix (FSDP/ZeRO/offload). Include the example table (7B train ≈112GB, 7B fp16 ≈14GB, 7B int4 ≈3.5GB, 70B train ≈1.1TB).
2. **KV-cache calculator (`kvcache`)** — `2 · layers · heads · head_dim · seq · batch · dtype_bytes`. Sliders for seq length & concurrency; show cache vs weights crossover; toggle GQA (divide heads by group) and KV-quant (dtype bytes) to see the win.
3. **Token & cost meter (`tokencost`)** — paste text → approximate tokens (~4 chars/token English; flag code/non-English "multilingual tax"); set input/output price per 1M → estimated cost; visualize prefill (parallel) vs decode (sequential). (Approximate client-side; note it's an estimate, recommend the model's own tokenizer.)
4. **Bias–variance slider (`biasvar`)** — model-complexity slider animates train vs val error curves + the learning-curve diagnostic; labels under/overfit zones and the fix.
5. **Calibration demo (`calibration`)** — reliability diagram with an over/under/well-calibrated toggle; a temperature-scaling slider that bends the curve toward the diagonal; live ECE readout.
6. **Throughput–latency dial (`throughput`)** — batch-size dial trading tokens/sec vs per-request latency; static vs continuous batching visualized (empty-slot waste vs saturated batch); p50/p99 readout.
7. **Agent compounding-error (`agenterror`)** — per-step reliability slider × step count → success probability (0.95ⁿ), with a curve; shows why "reduce the number of steps that must all go right."

All widgets: pure JS/Canvas/SVG, no external data, respect reduced-motion (no looping animation; update on input only).

---

## 11. System Design section (Track 16) — spec

A standalone, premium section. **Each architecture is a deep-dive page** with: a clean **inline SVG diagram**, a **components table**, **data-flow walkthrough**, **scaling levers**, **failure modes**, **cost levers**, **tradeoffs/alternatives**, and a **"how to present this in an interview"** framing. Tie each back to the relevant tracks.

**Open with a reusable framework page — "How to answer an ML system design question":**
`Clarify requirements → constraints & SLA (latency/throughput/cost/freshness) → data & features → model choice → serving & scaling → evaluation → monitoring & drift → failure modes & rollback → cost.` Make this a printable checklist.

**The 10 architectures (author all):**
1. **Production RAG system** 🔥 — ingestion + chunking + embedding + vector store + retriever + reranker + LLM + citation + cache + eval; freshness & guardrails.
2. **Agentic RAG / multi-agent system** 🔥 — planner, tool router, retrievers, critic/corrective loop (CRAG/Self-RAG), memory, MCP tools; bounded loops, idempotency, tracing.
3. **LLM serving platform** 🔥 — gateway → router → vLLM (PagedAttention + continuous batching) → KV-cache mgmt → autoscaling with warm floor; TTFT/TPOT SLOs.
4. **Multi-tenant LLM gateway** 🔥 — routing (small/big model), semantic + exact + prefix caching, rate limiting, cost metering, guardrails, observability.
5. **Recommendation system** ⭐ — two-tower retrieval + ranking; candidate gen → ranking → re-rank; feedback loops & position bias.
6. **Real-time fraud detection** ⭐ — streaming features + feature store (online/offline) + low-latency scoring + threshold/calibration + drift; PR-AUC focus, base-rate trap.
7. **Feature store architecture** ⭐ — offline (warehouse) + online (KV) + materialization + point-in-time joins; train/serve skew prevention.
8. **Semantic / hybrid search** ⭐ — BM25 + dense + cross-encoder rerank; ANN index; evaluation (NDCG, recall@k).
9. **ML monitoring & drift platform** ⭐ — four layers, reference windows, PSI/KS, alert design, ground-truth reconciliation.
10. **Distributed training platform** 📎 — orchestration, FSDP/ZeRO, checkpointing, experiment tracking (MLflow), fault tolerance.

---

## 12. Interview Mode (Track 17): FAQ-100 + Flashcards + Mock Drill — spec

### 12.1 FAQ-100 (`content/faq/faq-100.json`)
Exactly **100 questions**, each: `{id, q, a, track, difficulty: easy|med|hard, tags, type: concept|practical|systemdesign|behavioral}`. Keep answers crisp (the handbook's Q&A style — a strong 3–6 sentence answer that lands). Include the handbook's existing Q&A and expand to 100. **Suggested distribution** (Claude Code may adjust ±2 to balance):

- Core ML & systems (T1): 8 · Data/leakage/drift (T2): 9 · Stats/calibration/metrics (T3): 9 · Losses/objectives (T4): 5 · Evaluation/A-B/Goodhart (T5): 7 · Training at scale (T6): 6 · **LLM foundations (T7): 9** · **Serving/quantization/vLLM (T8): 8** · **GenAI & RAG (T9): 12** · **Fine-tuning/PEFT (T10): 6** · **Agents/MCP (T11): 7** · Pipelines/feature store (T12): 5 · Monitoring/cost (T13): 5 · Security/deployment/distributed (T14): 6 · Cloud GenAI (T15): 4 · Behavioral/India-market (T18): 4 → **100**.

UI: searchable, filter by track/difficulty/type, "reveal answer", mark known/unknown (feeds flashcards), copy-question.

### 12.2 Flashcards (`content/flashcards.json`) — lightweight spaced repetition
Generate from FAQ + lesson key-takeaways. Implement a simple **Leitner/SM-2-lite** box system in localStorage: rate "Again / Hard / Good / Easy" → reschedule. Daily "due" count on the dashboard fuels the streak.

### 12.3 Mock drill
"Start a 10-question round" → random pull weighted toward 🔥 tracks and the learner's weak/unknown items → self-rated → score + which tracks to revisit. Add a **"Pune/remote-India interview tips"** card: PyTorch-first, SQL expected, talk about *shipping & owning* a system not just notebooks, bring 3 deployed projects, expect RAG/agents/LLMOps depth.

---

## 13. Code snippet inventory (so the learner never pays a platform)

Every cloud/framework lesson must ship **runnable, minimal, copy-paste snippets** (with a one-line "free way to run this" note). Author at least the following. **Pin to current stable libs; Claude Code: verify import names against current docs at build time and add a small comment with the version used.** A few are inlined here as seeds; author the rest into the lessons.

**RAG / vector DB / orchestration (T9):**
- FAISS local index (build + similarity search) — *free, fully local.*
- ChromaDB persistent collection + query — *free, local.*
- pgvector: create extension, embed, `<=>` cosine query — *free, local Postgres.*
- Pinecone upsert/query (free starter tier).
- LangChain RAG chain (loader → splitter → embeddings → FAISS → retriever → LLM).
- LlamaIndex `VectorStoreIndex` query engine.
- Cross-encoder reranking (sentence-transformers) — *free, local.*
- Hybrid search (BM25 via `rank_bm25` + dense) fusion.
- RAGAS evaluation (faithfulness/answer-relevancy/context-precision).
- GraphRAG-lite with Neo4j (`neo4j` driver: create nodes/edges, Cypher retrieval) — *free Neo4j Aura / local Docker.*

**Fine-tuning (T10):**
- HF + PEFT LoRA SFT with `trl.SFTTrainer`.
- QLoRA: 4-bit `BitsAndBytesConfig` + LoRA.
- Merge adapter & push to Hub; load swappable LoRA at inference.

**Agents / MCP (T11):**
- LangGraph minimal agent (state graph: reason→tool→observe→loop, step cap).
- Tool/function-calling with a provider SDK (structured args).
- Minimal **MCP server** (expose a tool) + how a client lists/calls it.
- CrewAI / AutoGen "hello multi-agent" for contrast.

**Serving / MLOps (T8/T15):**
- FastAPI + Pydantic inference endpoint (with streaming SSE).
- vLLM `LLM`/`OpenAI`-compatible server quickstart.
- MLflow tracking (`log_params/metrics`, model registry).
- Dockerfile for a model API; `docker run` notes.
- Terraform skeleton for a cloud ML resource (read-only learning).

**Cloud GenAI (T15):**
- AWS Bedrock `boto3` invoke_model (text + embeddings); Knowledge Base retrieve_and_generate.
- GCP Vertex AI / Gemini: `google-genai` generate_content + function calling; grounding.
- BigQuery ML: `CREATE MODEL` + `ML.PREDICT` in SQL.
- Snowflake Cortex: `SNOWFLAKE.CORTEX.COMPLETE` / `EMBED_TEXT` / Cortex Search SQL.

**Seed example — LangChain RAG (author full versions like this):**
```python
# pip install langchain langchain-community faiss-cpu sentence-transformers
# FREE & LOCAL: embeddings + FAISS run on your machine, no paid API needed.
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text(big_text)
store = FAISS.from_texts(chunks, emb)
docs = store.similarity_search("How does the KV cache grow?", k=4)
# feed `docs` as grounded context to any LLM (local or API) + ask for citations
```

**Seed example — QLoRA fine-tune skeleton:**
```python
# pip install transformers peft trl bitsandbytes accelerate datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
tok = AutoTokenizer.from_pretrained(BASE)
lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
SFTTrainer(model=model, tokenizer=tok, train_dataset=ds, peft_config=lora,
           dataset_text_field="text", max_seq_length=1024).train()
# QLoRA = 4-bit frozen base + tiny trainable LoRA adapters → fine-tune on modest GPUs (e.g. free Colab/Kaggle T4).
```

**Seed example — AWS Bedrock invoke (boto3):**
```python
# pip install boto3   # FREE TIER: small monthly allowances; or learn the API shape offline.
import boto3, json
br = boto3.client("bedrock-runtime", region_name="us-east-1")
resp = br.invoke_model(modelId="amazon.titan-text-express-v1",
    body=json.dumps({"inputText":"Explain RAG in 2 lines."}))
print(json.loads(resp["body"].read()))
```

**Seed example — Snowflake Cortex (SQL):**
```sql
-- Runs inside Snowflake; free trial credits let you learn it at ₹0.
SELECT SNOWFLAKE.CORTEX.COMPLETE('mistral-large',
  'Summarize this broker note: ' || note_text) AS summary
FROM research_notes LIMIT 5;
```

---

## 14. Projects & Portfolio (Track 18) — spec

Market says **portfolio beats certificates.** Provide **3 flagship, step-by-step build guides** mapped to the owner's real résumé so they can speak truthfully and deeply:

1. **End-to-end ML pipeline** (e.g. customer-retention / churn with XGBoost) — data → features → train → eval (PR-AUC, calibration) → FastAPI serve → MLflow → Docker. *Maps to the owner's "Chat-with-SQL + churn" project.*
2. **Production RAG app** (equity/broker-research Q&A with citations + CRAG/Agentic loop) — chunking, FAISS/Chroma, rerank, RAGAS eval, Streamlit/Gradio UI, deploy on HF Space. *Maps to "Financial Analysis Assistant (Agentic + Corrective RAG)" and "Estimates Extractor."*
3. **MLOps deployment** — containerized model + monitoring (drift, latency p99, token spend), canary/rollback notes. *Maps to "AWS SageMaker/Bedrock" claims.*

Each guide: objective, architecture diagram, repo structure, milestone checklist, "what to show in an interview", and honest talking points. Add a short **"how to de-fake your résumé"** note: for each claimed skill, the exact lesson(s) + project step that makes the claim true.

---

## 15. Feature checklist (acceptance-level)

- [ ] Dashboard: Readiness %, streak, XP, "Focus first" (🔥+⭐), continue, hot-topic spotlight, daily flashcards-due.
- [ ] Sidebar with per-track progress rings; remembers expand/collapse.
- [ ] Lesson reader with all §6.7 block components; reading-progress bar; prev/next; mark-complete (XP + streak).
- [ ] ⌘K command palette (lunr) across lessons/FAQ/system-design/glossary/widgets.
- [ ] Bookmarks + per-lesson personal notes (localStorage).
- [ ] Code blocks: syntax highlight + copy button + language label; optional Pyodide "Run it" on ≤3 demos.
- [ ] All 7 interactive widgets.
- [ ] System Design section (framework + 10 architectures, SVG diagrams).
- [ ] Interview Mode (FAQ-100, flashcards w/ spaced repetition, mock drill, India tips).
- [ ] Projects section (3 guides).
- [ ] Glossary.
- [ ] Theme toggle (dark default / light); reduced-motion + disable-3D toggle.
- [ ] 3D background with adaptive degrade + reduced-motion respect + tab-hidden pause.
- [ ] Fully responsive (mobile sidebar drawer; tables → cards).
- [ ] Offline-capable after first load (consider a tiny service worker caching the app shell + content; optional but nice).
- [ ] Keyboard focus visible; WCAG-AA contrast; semantic landmarks.

---

## 16. Build phases (commit after each)

**Phase 0 — Scaffold & deploy skeleton.** Repo structure (§7), `index.html`, `theme.css` tokens, generate HF `README.md` (§17), move spec to `docs/`. Deploy a "Hello, mission control" glass page with the 3D field working + reduced-motion fallback. *Goal: it's live on HF and looks intentional.*

**Phase 1 — App shell & engine.** Router, store, layout (top bar/sidebar/main), theme toggle, command-palette shell, `render.js` with all block-component parsers, gamify.js (XP/streak/Readiness), search index build step. Seed with 2–3 real lessons end-to-end (use the KV-cache example in §8 as the reference lesson).

**Phase 2 — Foundation tracks (1–6, 12) content.** Author all lessons from the handbook PDF in the prescribed rhythm. Add widgets vram/biasvar/calibration as their lessons land.

**Phase 3 — 🔥 Hot tracks (7, 8, 9, 10, 11, 13, 14) content.** The market core. Author with maximal depth + every snippet from §13. Add widgets kvcache/tokencost/throughput/agenterror.

**Phase 4 — Cloud GenAI (15) + System Design (16).** Snippets + free-tier notes; 10 architecture pages with SVG diagrams + the answering framework.

**Phase 5 — Interview Mode (17) + Projects (18) + Glossary.** FAQ-100, flashcards (spaced repetition), mock drill, India tips; 3 project guides; glossary.

**Phase 6 — Polish & QA.** Mobile pass, a11y pass (keyboard/contrast/reduced-motion), perf pass (3D adaptive degrade, lazy-load Pyodide/KaTeX, image/SVG sizes), offline service worker, final content proofread, Definition-of-Done (§19).

After each phase: commit, push, confirm the HF Space rebuilds and renders.

---

## 17. Hugging Face Spaces deployment

The repo-root **`README.md`** Claude Code generates must begin with this front-matter (Static SDK serves `index.html` automatically):

```yaml
---
title: ML/AI Engineer Academy
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: true
license: mit
short_description: Production-grade ML/AI engineering — basics to interview-ready, free.
---
```

Steps (document these in the generated README too):
1. Create a **Space** → SDK **Static** (or push this repo and set SDK to static in Space settings).
2. Push the repo (git) to the Space remote. HF serves `index.html` from root; no build server needed.
3. If `reference/handbook-2026.pdf` is large, track it with **Git LFS** (or keep it out of the Space and only in local dev — it's a build-time content source, not needed at runtime).
4. All libraries load from CDN at runtime — no install step on HF.
5. **Verify against current HF Spaces docs at build time** (front-matter fields can evolve); adjust if needed.

---

## 18. Content authoring guidelines (voice & depth)

- **Voice:** the handbook's — direct, senior, production-first, lightly witty. Short sentences. No fluff. Write from the engineer's side of the screen.
- **Every lesson** ends with at least one **interview-line** and one **drill**; 🔥 lessons also get a **war-story** and ≥1 **code** block.
- **Always answer "why does this matter in production?"** before mechanics.
- **Code:** minimal, runnable, commented with the *free* way to run it. Prefer local/open-source first, then cloud.
- **Diagrams:** crisp inline SVG (training loop, agent loop, eval funnel, rollout ladder, RAG pipeline, feature-store halves). No heavy raster images.
- **Length discipline:** target ~6–12 min/lesson. If longer, split. One idea per block. Lots of air.
- **Honesty:** when a topic is hype or low-2026-value, say so (🧊) and move on.
- **No copyrighted text dumps.** Teach in your own words; the handbook is a *source to learn from*, not to reproduce verbatim at length. Paraphrase, restructure, and add the new tracks.
- **India/Pune framing** where natural (salary bands, PyTorch-first, SQL expected, portfolio-over-cert, remote-friendly roles).

---

## 19. Definition of Done

The app is "done enough to depend on" when:
1. It's **live on a free HF Static Space** and loads in <2s on a mid laptop, works offline after first load.
2. **Every track in §9 has its lessons authored** in the prescribed rhythm, with 🔥 tracks fully code-rich.
3. **All 7 widgets, the 10 system-design pages, FAQ-100, flashcards, and 3 project guides** exist and work.
4. **Progress/XP/streak/notes/bookmarks persist** across reloads; readiness % reflects real completion.
5. **A11y floor met** (keyboard, contrast, reduced-motion, mobile).
6. The owner can open any "faked" résumé skill and find a lesson + snippet + project step that genuinely teaches it.
7. It **feels** like a premium instrument — calm, energetic, motivating — and is **not** heavy or nauseating (motion subtle, defaults dark, generous spacing).

---

## 20. Suggested learning order for the owner (put this on the Dashboard)

Given Pune + remote-India targeting and the résumé gaps, recommend this path (the app should surface it as "Your recommended path"):

1. **Quick foundation refresh:** T1 (mem math, mixed precision), T2 (leakage/drift), T3 (calibration, PR-AUC), T5 (decision-driven eval). *(You know much of this — move fast.)*
2. **Go deep where the jobs are:** T7 → T8 → **T9 (RAG/Agentic RAG)** → **T10 (PEFT/LoRA)** → **T11 (Agents/MCP)** → T13 → T14.
3. **Round out the platform story:** T12 (feature store / skew), **T15 (AWS/GCP/Snowflake, free-tier)**.
4. **Make it hireable:** T16 (System Design), then **build the 3 projects (T18)**, drilling **T17 (FAQ-100/flashcards)** alongside.

> Owner note: this app turns the résumé claims into real, demonstrable skill — Agentic & Corrective RAG, GraphRAG, PEFT/LoRA/QLoRA, vector DBs, agents/MCP, and AWS/GCP/Snowflake GenAI — all learnable here for ₹0. Ship the 3 projects; portfolio beats certificates in this market.

— End of build spec —
