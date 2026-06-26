---
id: failure-handling
track: t7-systemdesign-capstones
title: "Failure handling & graceful degradation"
badge: HOT
minutes: 9
prereqs: []
tags: [system-design, reliability, fallback, circuit-breaker, production]
xp: 60
hot2026: true
---

Imagine your recommendation model has been running perfectly for six weeks. Then, at 2 a.m. on a Friday, the GPU inference server runs out of memory and starts timing out. Every product page now hangs for 10 seconds before returning a 500 error. Users bounce. Revenue tanks. The on-call engineer wakes up to a disaster — not because the model was wrong, but because nobody designed what should happen *when the model breaks*.

That gap between "model works in notebooks" and "system survives failures gracefully" is exactly what separates a junior ML engineer from a senior one. It's also one of the highest-signal questions in system design interviews.

## What graceful degradation actually means

Graceful degradation means your system keeps delivering *some* value even when a component fails, instead of blowing up completely. Think of it as a ladder of fallbacks.

At the top rung: the full ML prediction, fresh and personalized. One rung down: a cached prediction from 5 minutes ago. Another rung down: a simpler rule-based recommendation ("show best-sellers"). Bottom rung: a static default list that always loads. The user sees *something useful* at every rung — not an error page.

This is different from fault tolerance (which means hiding failures entirely). Degradation acknowledges the failure internally but maintains user experience externally.

:::why-prod
In production, ML components fail more often than engineers expect — GPU OOMs, model timeouts, upstream feature service outages, stale embeddings. Without an explicit fallback plan, a single component failure cascades into a full user-facing outage. Graceful degradation keeps revenue flowing and buys the team time to fix the root cause.
:::

## The four patterns you must know

**1. Cached responses.** Store the last successful prediction per user/item. If the live model fails, serve the cache. Stale-by-minutes beats "error" every time.

**2. Simpler fallback model.** Keep a cheap, fast model (logistic regression, heuristic ranker) in the serving path. Route traffic there when the heavy model is unhealthy. This is the "sidecar" pattern.

**3. Circuit breaker.** After N consecutive failures, stop calling the failing service altogether and immediately return the fallback. After a timeout (say, 30 seconds), send a probe request to check recovery. This prevents pile-on — thousands of requests queueing up and making the failure worse.

**4. Timeout + retry with exponential backoff.** Never wait forever. Set aggressive timeouts (e.g., 200 ms for inference) and retry once with a short backoff. Cap total retries so you don't amplify traffic on a struggling server.

:::table {title="Fallback pattern comparison"}
| Pattern | Latency cost | Staleness | Best for |
|---|---|---|---|
| Cached response | Near-zero | Minutes to hours | Personalization, rankings |
| Simpler fallback model | Low | Real-time | Scoring, classification |
| Circuit breaker | Zero (fail-fast) | N/A | Any downstream service |
| Static default | Zero | Days | Catalog, onboarding |
:::

## Designing the health check

Every ML service should expose a `/health` or `/ready` endpoint that your load balancer and circuit breaker poll. A good health check verifies the model is loaded *and* can produce a prediction — not just that the HTTP server is up.

```python {title="Health check endpoint (FastAPI)" run=false}
from fastapi import FastAPI, HTTPException
import time

app = FastAPI()

# Simulate a model object loaded at startup
model = None  # replace with: model = load_model(...)

@app.on_event("startup")
def load():
    global model
    # model = load_your_model_here()
    model = {"ready": True}  # placeholder

@app.get("/health")
def health():
    """
    Return 200 only when the model can actually serve predictions.
    Load balancers use this to drain traffic before a failure spreads.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Optionally run a tiny smoke-inference to confirm GPU is responsive
    try:
        start = time.time()
        # _ = model.predict([[0.0] * 128])  # smoke test
        latency_ms = (time.time() - start) * 1000
        if latency_ms > 500:
            raise HTTPException(status_code=503, detail="Model too slow")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"status": "ok"}


@app.get("/fallback/recommend")
def fallback_recommend(user_id: str):
    """
    Called by the API gateway when /recommend times out or returns 5xx.
    Returns the pre-computed best-sellers list — zero ML, always fast.
    """
    BEST_SELLERS = ["item_42", "item_7", "item_99"]  # swap with DB read
    return {"items": BEST_SELLERS, "source": "fallback"}
```

## Circuit breaker in plain English

Picture a physical circuit breaker in your home. Overload? It trips and cuts power — preventing a fire. Once you fix the problem and reset it, power flows again.

In software: after 5 consecutive 503s from the inference server, the circuit "opens" — all requests skip the server and go straight to the fallback, with zero latency cost. Every 30 seconds, one probe request checks if the server is back. If it succeeds, the circuit closes and traffic resumes. Libraries like `pybreaker` (Python), `resilience4j` (Java/Kotlin), and `opossum` (Node) implement this out of the box.

:::gotcha
Forgetting to set timeouts is the most common mistake. Without a timeout, a slow model holds a thread open indefinitely. Under load, you exhaust your thread pool, and the *entire* API goes down — even endpoints that don't touch ML. Always set a hard timeout shorter than your SLA (e.g., if your p99 SLA is 500 ms, set the inference timeout at 300 ms and leave room for serialization).
:::

:::war-story {title="The retry storm that took down prod"}
A team added retry logic to their recommendation API: if inference failed, retry up to 3 times. Reasonable. But they forgot exponential backoff — all retries fired immediately. One evening a GPU host degraded and started returning slow 504s instead of fast failures. Each incoming request spawned 3 more before timing out. Within 90 seconds, request volume tripled, the healthy GPU host fell over too, and the entire recommendation surface went dark. The fix took 20 minutes; the retry-storm investigation took two days. The lesson: retries must use jitter + exponential backoff, and you need a total-timeout budget per user request, not per retry.
:::

## Observability ties it together

Fallbacks are silent killers — if you degrade to a cache and nobody notices, you could serve stale data for days. Instrument every fallback path:

- Emit a metric (`fallback_served_total`, labeled by reason) every time you hit a fallback.
- Alert when fallback rate exceeds 1% of traffic for more than 2 minutes.
- Log which users/requests got degraded responses so you can replay them once the system recovers.

:::interview-line
"Every ML service I design has a fallback ladder: cached predictions, a simpler model, then a static default — and a circuit breaker that fails fast into whichever rung is healthy."
:::

:::qa {q="How would you handle an inference service that starts returning timeouts under load?"}
First, I'd fail fast with a circuit breaker — after a few consecutive timeouts, stop sending traffic and route to a fallback (cache or simpler model). Then I'd investigate the root cause: GPU memory, batch size, or upstream bottleneck. I'd also make sure my retry logic uses exponential backoff with jitter so retries don't amplify the load. The circuit breaker recovers automatically with a probe once the service is healthy again.
:::

:::qa {q="What's the difference between a retry and a fallback?"}
A retry sends the same request to the same service again, hoping for a transient fix — useful for network blips. A fallback routes to a *different* code path entirely when the primary is persistently unhealthy. You need both: retries for transient errors, fallbacks for sustained failures. Over-relying on retries without fallbacks is how you cause retry storms.
:::

:::drill {type="mcq" q="A circuit breaker is in the OPEN state. What does that mean for incoming requests?"}
- [ ] All requests are retried with exponential backoff
- [ ] Requests are queued until the service recovers
- [x] Requests skip the failing service and go directly to the fallback
- [ ] The circuit breaker sends a health check on every request
:::

:::drill {type="mcq" q="Your p99 SLA for a recommendation API is 400 ms. Where should you set the ML inference timeout?"}
- [ ] 400 ms — match the SLA exactly
- [ ] 600 ms — give the model extra headroom
- [x] ~200–250 ms — leave room for network, serialization, and fallback logic within the SLA
- [ ] No timeout — let the model finish to avoid serving stale data
:::

:::key-takeaway
Design the failure path before the happy path: define your fallback ladder, set hard timeouts shorter than your SLA, use a circuit breaker to fail fast, and alert loudly whenever you're serving degraded responses.
:::
