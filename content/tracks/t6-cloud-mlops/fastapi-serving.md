---
id: fastapi-serving
track: t6-cloud-mlops
title: "Serving with FastAPI (+ streaming)"
badge: HOT
minutes: 9
prereqs: []
tags: [fastapi, serving, inference, streaming, pydantic, async, production]
xp: 60
hot2026: true
---

Your sentiment model scores 99 on every benchmark. Stakeholders are thrilled. Then someone asks the one question you hadn't thought about: "How do I actually *call* it?" You hand them a Jupyter notebook. The room goes quiet. You need an API — and you need it today. This is where FastAPI enters the story, and it changes everything.

## Why FastAPI won the ML serving wars

Flask was the original go-to for ML APIs. Simple, familiar, done in ten lines. But Flask is synchronous: while it waits for your slow model to finish, the whole thread sits idle. Under load, requests queue up and latency spikes.

FastAPI is built on ASGI (Asynchronous Server Gateway Interface) and Starlette. It can handle many requests concurrently without spawning extra threads. Add automatic request/response validation via Pydantic, auto-generated `/docs` Swagger UI, and native support for Server-Sent Events (SSE) — and it is the clear default for ML services in 2025.

:::why-prod
In production, a synchronous Flask endpoint serving a 200 ms model hits a wall around 20–30 requests per second on a single process. A FastAPI async endpoint with the same model can handle 5–10× that throughput on the same hardware, because I/O waits (database, logging, caching) no longer block the event loop.
:::

## The anatomy of a FastAPI ML endpoint

Here is the minimal pattern that every ML engineer should internalize — one prediction endpoint with proper schema validation.

```python {title="Basic FastAPI inference endpoint" run=false}
# pip install fastapi uvicorn pydantic
# Run locally: uvicorn app:app --reload
from fastapi import FastAPI
from pydantic import BaseModel
import time

app = FastAPI(title="Sentiment API", version="1.0")

# --- Schemas ---
class PredictRequest(BaseModel):
    text: str
    threshold: float = 0.5  # default, documented automatically

class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float

# --- Fake model (swap in your real one) ---
def run_model(text: str, threshold: float) -> tuple[str, float]:
    score = min(1.0, len(text) / 100)          # toy logic
    label = "POSITIVE" if score >= threshold else "NEGATIVE"
    return label, round(score, 4)

# --- Endpoint ---
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    t0 = time.perf_counter()
    label, score = run_model(req.text, req.threshold)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return PredictResponse(label=label, score=score, latency_ms=latency)

# Health check — required by every load balancer and k8s liveness probe
@app.get("/health")
async def health():
    return {"status": "ok"}
```

Notice three things: the schema lives in a `BaseModel` (Pydantic validates and coerces types automatically), the endpoint is `async`, and there is a `/health` route. That last one is not optional in real deployments.

## Adding streaming — the LLM use case

The moment you swap your model for an LLM, users sit staring at a blank screen for five seconds before a wall of text appears. Streaming fixes this. FastAPI supports Server-Sent Events natively via `StreamingResponse`.

```python {title="SSE streaming for LLM tokens" run=false}
# pip install fastapi uvicorn
# Run: uvicorn stream_app:app --reload
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

async def token_generator(prompt: str):
    """Yields newline-delimited SSE events. Swap the loop for your LLM call."""
    words = f"You asked: {prompt}. Here is a streamed answer word by word.".split()
    for word in words:
        yield f"data: {word}\n\n"   # SSE format: "data: <payload>\n\n"
        await asyncio.sleep(0.05)   # simulate per-token latency
    yield "data: [DONE]\n\n"        # sentinel so the client knows to stop

@app.post("/stream")
async def stream_response(req: PromptRequest):
    return StreamingResponse(
        token_generator(req.prompt),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no"},  # disables nginx response buffering
    )
```

The `X-Accel-Buffering: no` header is easy to forget and hard to debug: nginx will buffer the whole response and destroy the streaming effect unless you send it.

:::table {title="Sync vs async vs streaming — when to use what"}
| Pattern | When to use | Typical latency profile |
|---|---|---|
| `def` (sync) | CPU-bound model, no I/O | Fast model, low concurrency |
| `async def` | I/O-bound calls (DB, cache, external API) | Mixed workloads, high concurrency |
| `StreamingResponse` | LLMs, long generation, large file output | First-token fast, total slow |
| Background task | Fire-and-forget logging, webhooks | Non-blocking side effects |
:::

## Loading your model once — not on every request

New engineers make a classic mistake: they load the model *inside* the endpoint function. That means every request pays the 2–10 second model loading cost.

Use FastAPI's `lifespan` context manager (or the older `@app.on_event("startup")`) to load the model once at startup and store it on `app.state`.

:::gotcha
Loading a heavy model inside `@app.post("/predict")` means your p50 latency is 200 ms but every cold-request (or every request if you reload on each call) is 8 seconds. Always load models at startup using `app.state.model = load_model()` inside the lifespan handler, then access it via `request.app.state.model` inside your endpoint.
:::

## Request validation is free — use it

Pydantic catches bad inputs before they touch your model. No `if not isinstance(text, str)` guards needed. If a client sends `{"text": 12345}`, Pydantic coerces it to `"12345"`. If they send `{"prompt": "hello"}` to your `/predict` endpoint that expects `text`, FastAPI returns a structured `422 Unprocessable Entity` automatically — with the exact field that failed. Your model never sees garbage.

:::war-story {title="The 3 AM streaming buffer incident"}
A team shipped a streaming LLM endpoint behind an nginx reverse proxy. In local testing, tokens appeared one by one. In production, users saw a five-second blank screen, then the entire response appeared at once. The on-call engineer spent two hours checking the model, the async loop, and the SSE format. Everything looked correct. The fix was a single header: `X-Accel-Buffering: no`. Nginx was faithfully buffering the SSE stream before forwarding it. The header had been in the FastAPI docs example all along, but it had been skipped as "probably not needed."
:::

:::interview-line
"FastAPI gives me async concurrency, Pydantic validation, and auto-docs out of the box — I load the model once at startup on app.state, add a /health route for the load balancer, and add X-Accel-Buffering: no the moment I go streaming."
:::

:::qa {q="Why use async endpoints in FastAPI for ML serving?"}
Async endpoints free the event loop to handle other requests while waiting on I/O (database lookups, caching, downstream API calls). For CPU-bound models you still gain — you can offload heavy computation to a thread pool with `asyncio.run_in_executor` and keep the endpoint non-blocking. The result is higher throughput on the same hardware.
:::

:::qa {q="What is Server-Sent Events (SSE) and why does it matter for LLMs?"}
SSE is a one-way HTTP streaming protocol where the server pushes newline-delimited `data:` events to the client over a single long-lived connection. For LLMs it lets you send tokens as they are generated rather than waiting for the full response. This dramatically reduces perceived latency — users see output start within milliseconds of the first token, even if the full response takes ten seconds.
:::

:::drill {type="mcq" q="A FastAPI endpoint loads a 500 MB scikit-learn model on every POST request. What is the correct fix?"}
- [ ] Move to Flask which handles large models better
- [ ] Load the model in a global variable at module level
- [x] Load the model once in the lifespan handler and attach it to app.state
- [ ] Use StreamingResponse to reduce memory pressure
:::

:::drill {type="mcq" q="You deploy a streaming FastAPI endpoint behind nginx. In production, clients receive the entire response at once instead of token-by-token. What is the most likely cause?"}
- [ ] asyncio.sleep() is blocking the event loop
- [ ] The SSE event format is missing the double newline
- [ ] Pydantic is buffering the response schema
- [x] nginx is buffering the response; add X-Accel-Buffering: no header
:::

:::drill {type="mcq" q="A client sends {\"text\": 42} to a FastAPI endpoint that declares text: str in its Pydantic schema. What happens?"}
- [ ] FastAPI raises a 500 Internal Server Error
- [ ] The endpoint receives the integer 42 and crashes the model
- [x] Pydantic coerces 42 to "42" and the endpoint runs normally
- [ ] FastAPI returns 422 because integers are never valid
:::

:::key-takeaway
FastAPI's three superpowers for ML serving are: async concurrency for high throughput, Pydantic schemas for free request validation, and StreamingResponse + SSE for real-time LLM output. Load your model once at startup, always add /health, and never forget X-Accel-Buffering: no behind nginx.
:::
