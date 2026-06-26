---
id: project-ladder
track: t7-systemdesign-capstones
title: "The capstone project ladder"
badge: HOT
minutes: 9
prereqs: []
tags: [portfolio, capstone, projects, career, system-design, production]
xp: 60
hot2026: true
---

You've sent out forty applications. Three callbacks. Two of those callbacks ask for a take-home or a "show us something you've built." The third skips straight to a system design round and, when you share your screen, they ask: "walk us through a project where you had to make a real trade-off."

That moment separates the people who studied ML from the people who *practice* ML. A capstone project is your proof of practice — a story you own, a system you can diagram, and trade-offs you can defend. The ladder is how you build that story, rung by rung, without spending six months on a single overengineered monolith.

## Why a "Ladder" and Not a "Portfolio"

Most advice says "build a portfolio." That word implies a collection of equal items. The ladder is different: each rung teaches a new dimension of production thinking. You climb in order. Skipping rungs shows.

- Rung 1 proves you can ship something.
- Rung 2 proves you understand the full data-to-prediction pipeline.
- Rung 3 proves you think about failure, cost, and scale.
- Rung 4 proves you can design a system you *haven't* built yet — which is what the interview actually tests.

:::why-prod
Hiring managers at product companies don't care about your Kaggle leaderboard rank. They care whether you've shipped a model to real traffic, handled a misprediction in production, and made a deliberate cost/quality trade-off. Projects are the cheapest way to get that story before you have a job.
:::

## Rung 1 — Notebook to API (1–2 days)

Take any model — a fine-tuned classifier, a recommender, a regression — and wrap it in a real HTTP endpoint. No Streamlit demo. An API that another service could call.

What you prove: you know that ML code that runs in a notebook is not the same as ML code that serves requests.

The minimum bar for Rung 1:
- `/predict` endpoint with request validation (Pydantic or similar)
- Latency logged on every call
- A README that shows how to run it locally in one command

```python {title="Rung 1 — Production-flavored inference skeleton" run=false}
# Run locally: uvicorn app:app --reload
# Free to host: Railway, Render, or Hugging Face Spaces (free tier)

import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

# Load your model once at startup, not per request
model = None  # replace with: model = load_your_model()

@app.on_event("startup")
async def startup():
    global model
    # model = load_your_model()  # e.g. from joblib, transformers, etc.
    logger.info("Model loaded and ready")

@app.post("/predict")
async def predict(req: PredictRequest):
    start = time.perf_counter()
    try:
        # result = model.predict(req.text)
        result = {"label": "positive", "score": 0.92}  # stub
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(f"predict latency={latency_ms:.1f}ms input_len={len(req.text)}")
        return result
    except Exception as e:
        logger.error(f"predict failed: {e}")
        raise HTTPException(status_code=500, detail="model error")

@app.get("/health")
async def health():
    return {"status": "ok"}
```

Notice two things in that skeleton: the model loads once at startup (not per request), and every call logs its latency. Those two lines alone put you ahead of most notebook-to-demo projects.

## Rung 2 — The Full Pipeline (1 week)

Add the plumbing that lives before and after the model:

- A data ingestion script (scrape, download, or generate synthetic data)
- A training script that produces a versioned artifact (even just a timestamped `.pkl`)
- Your Rung 1 serving layer, loading that artifact
- A basic evaluation report written to a file (accuracy, F1, or whatever fits)

What you prove: you understand that a model is the *output* of a pipeline, not the pipeline itself.

:::table {title="Rung 2 Checklist"}
| Component | Minimum bar | Why it matters |
|---|---|---|
| Data ingest | Script, not manual download | Reproducibility |
| Training | CLI with `--output-path` flag | Artifact versioning |
| Evaluation | Saved metrics JSON | Regression detection |
| Serving | Load from artifact path | Decoupled from training |
| README | One-command re-run | Reviewer's first filter |
:::

## Rung 3 — Production Thinking (1–2 weeks)

This is where the ladder gets steep — and interesting. Pick one of these dimensions to add depth:

**Cost awareness** — Track how many tokens or GPU seconds each request consumes. Add a `/metrics` endpoint. Write down what it would cost per million requests.

**Failure handling** — What happens when your model returns `None`? When the upstream data feed goes stale? Add a fallback path and log when it fires.

**Observability** — Add structured JSON logs. Show that you can answer "what was the p95 latency yesterday?" from your own logs.

You don't have to add all three. Picking *one* and going deep is far more compelling than adding all three at surface level.

:::gotcha
The most common Rung 3 mistake is adding monitoring *dashboards* before you have anything worth monitoring. Set up logging first. Log the right things (latency, input shape, model version, output confidence). Dashboards are just a query on top of good logs — get the logs right and the dashboard is trivial.
:::

## Rung 4 — The Design Doc (2–3 days)

Here's the secret: Rung 4 is not another project. It's a written design for a system that is *10x bigger* than what you built in Rung 3.

Write a 1–2 page document (not slides) that answers:
- What would need to change if this system had to serve 10,000 RPS instead of 10?
- Where would you add a cache? Why?
- What would you shard, and on which key?
- What is the SLA, and what happens if you miss it?

This is exactly what a system design interview asks. The difference: you're writing about a system you already built and understand intimately. You're not guessing. You're scaling up something real.

:::war-story {title="The project that saved a Friday interview"}
A candidate was interviewing for an ML platform role at a mid-size company. The system design round opened with: "Design a real-time fraud detection service." The candidate had built a Rung 3 project — a transaction classifier with latency logging and a fallback rule engine. They pulled up their own design doc, described the exact same problem at small scale, then walked through every scaling decision: "At my scale I used in-process caching. At your scale I'd move to Redis with a 60-second TTL because..." The interviewer stopped taking notes and started drawing architecture diagrams *with* the candidate. Offer came the same week.
:::

## The Presentation Layer

No project exists until someone else can understand it. For each rung, your README should answer three questions in under 60 seconds of reading:

1. What problem does this solve?
2. What trade-off did you make, and why?
3. How do I run it?

That's it. Not your accuracy score. Not your loss curve. The trade-off. That's what a senior engineer reads for.

:::interview-line
"I built it end-to-end, shipped it to real traffic, and the design doc shows how I'd scale it to production load — want me to walk through the trade-offs?"
:::

:::qa {q="Why is a capstone project more valuable than a Kaggle top-10 finish?"}
Kaggle proves optimization skill on a fixed, clean dataset. A capstone proves you can ingest messy data, version artifacts, serve predictions reliably, and handle failure — the full production loop. Interviewers hire for the full loop, not the leaderboard position.
:::

:::qa {q="What if I don't have a GPU or expensive cloud credits for a capstone?"}
Rung 1 and 2 projects run comfortably on a free-tier CPU instance. Use quantized models (GGUF via llama.cpp, or smaller HuggingFace models) and free hosting on Railway or Render. The production thinking — logging, versioning, fallback logic — is what impresses, not the model size.
:::

:::qa {q="How many capstone projects do I need before I start applying?"}
Rung 1 is enough to apply and learn from rejections. Rung 2 before your first real screen. Rung 3 before senior or platform roles. Rung 4 (the design doc) before any system design round. Climb while you apply — don't wait until the ladder is "complete."
:::

:::drill {type="mcq" q="A recruiter has 90 seconds to scan your project README. Which section matters most?"}
- [ ] Your model's validation accuracy and loss curves
- [x] The trade-off you made and why, explained in plain language
- [ ] The full list of libraries and their versions
- [ ] A screenshot of your Jupyter notebook output
:::

:::drill {type="mcq" q="What distinguishes a Rung 3 project from a Rung 2 project?"}
- [ ] Rung 3 uses a larger or more complex model architecture
- [ ] Rung 3 includes a Streamlit or Gradio UI for the demo
- [x] Rung 3 adds at least one production dimension: cost tracking, failure handling, or observability
- [ ] Rung 3 must be deployed to a paid cloud provider
:::

:::drill {type="mcq" q="The Rung 4 design doc asks you to scale your Rung 3 project by 1000x. What is the PRIMARY goal of writing it?"}
- [ ] To prove you can build a distributed system from scratch
- [ ] To generate content for your blog and personal brand
- [x] To practice the exact reasoning pattern that system design interviews test, using a system you already know deeply
- [ ] To identify bugs in your Rung 3 implementation before interviews
:::

:::key-takeaway
The ladder works because each rung teaches a different production skill: shipping, pipeline thinking, failure handling, and design reasoning. You don't need all four before you start applying — but each rung makes every interview easier to walk into.
:::
