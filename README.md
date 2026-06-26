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
# take all permissions at once dont flood with the permissions 
# 🛰️ ML/AI Engineer Academy

A single-page, offline-capable web app that teaches **production-grade ML/AI engineering** —
from first principles to interview/on-call mastery. Built as a calm "mission-control HUD":
a dark deep-space canvas, a subtle 3D neural field, glass instrument panels, one electric
accent, gamified progress. **Free forever** — no backend, no API keys, all libraries via CDN.

> The full build specification lives in [`docs/BUILD_SPEC.md`](docs/BUILD_SPEC.md).

## What it covers

Classic ML foundations taught tightly, with **depth where 2026 Pune / remote-India jobs pay**:
GenAI & RAG, Agentic/Corrective RAG, GraphRAG, fine-tuning (PEFT/LoRA/QLoRA), agents & MCP,
inference & serving (vLLM, KV-cache, quantization), MLOps & monitoring, cloud GenAI
(AWS Bedrock/SageMaker, GCP Vertex/Gemini, Snowflake Cortex), and ML system design.

## Features

- **18 tracks** of lessons in a production-first voice (concept → why-prod → gotcha → war-story → interview-line → drill).
- **7 live "instrument" widgets** — VRAM, KV-cache, token/cost, bias–variance, calibration, throughput, agent compounding-error.
- **10 system-design deep-dives** with inline SVG diagrams.
- **Interview Mode** — FAQ-100, spaced-repetition flashcards, mock drill.
- **⌘K command palette**, full-text search, bookmarks, personal notes.
- **Progress saved** to `localStorage` — XP, streaks, readiness %, last position.
- Dark default + light theme; reduced-motion honored; WCAG-AA; mobile responsive; offline after first load.

## Tech

Vanilla JS (ES modules) + a tiny hash-router · Three.js (subtle 3D background) ·
Tailwind Play CDN + hand-written `theme.css` design tokens · marked.js + DOMPurify ·
highlight.js · KaTeX · lunr.js · optional Pyodide. No npm/build step required.

## Run locally

```bash
# Any static file server works — no build step.
python3 -m http.server 8000
# then open http://localhost:8000
```

## Deploy on Hugging Face Spaces

1. Create a **Space** → SDK **Static**.
2. Push this repo to the Space git remote. HF serves `index.html` from the repo root — no build server needed.
3. All libraries load from CDN at runtime; nothing to install on HF.
4. The handbook PDF under `reference/` is a build-time content source only — not needed at runtime
   (track it with Git LFS if you keep it in the Space, or leave it in local dev).

## License

MIT
