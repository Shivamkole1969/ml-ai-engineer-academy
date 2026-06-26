# LLM Serving Platform

Your team just shipped a RAG-powered assistant for a B2B SaaS product. Day one: 50 users. Day 30: 50,000. The model is the same. The infrastructure is very, very different.

This is the design problem — how do you build a platform that can serve LLM inference at scale, cost you don't hate, and latency your users don't notice?

---

## Architecture at a Glance

<svg viewBox="0 0 820 130" width="100%" role="img" aria-label="LLM serving platform pipeline: Gateway → Router → Inference Fleet → Cache → Observability">
  <!-- Gateway -->
  <rect x="10" y="42" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="65" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">API Gateway</text>
  <text x="65" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">auth · rate limit</text>

  <line x1="120" y1="64" x2="155" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="150,59 155,64 150,69" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Request Router -->
  <rect x="155" y="42" width="115" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="212" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Router</text>
  <text x="212" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">model · priority</text>

  <line x1="270" y1="64" x2="305" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="300,59 305,64 300,69" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Cache -->
  <rect x="305" y="42" width="115" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="362" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Semantic Cache</text>
  <text x="362" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">KV · vector hit</text>

  <line x1="420" y1="64" x2="455" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="450,59 455,64 450,69" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Inference Fleet -->
  <rect x="455" y="42" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="520" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Inference Fleet</text>
  <text x="520" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">GPU workers · batching</text>

  <line x1="585" y1="64" x2="620" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="615,59 620,64 615,69" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Observability -->
  <rect x="620" y="42" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="685" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Observability</text>
  <text x="685" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">latency · cost · evals</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **API Gateway** | Single entry point. Handles auth (API keys, JWT), rate limiting per tenant, request validation, and TLS termination. First line of abuse prevention. |
| **Request Router** | Decides which model and which replica handles this request. Routes based on model version, tenant tier, context length, and current queue depth. |
| **Prompt Manager** | Stores and versions system prompts. Injects the right template before the request hits the model. Decouples prompt iteration from code deploys. |
| **Semantic Cache** | Checks if a semantically equivalent request was answered recently. Two layers: exact KV match (Redis), then vector-similarity match against cached embeddings. Returns cached tokens for free. |
| **Inference Fleet** | The GPU workers actually running the model. vLLM or TGI under the hood. Handles continuous batching, KV-cache management, and streaming via SSE. |
| **Context / RAG Retriever** | Fetches relevant chunks from a vector store (Qdrant, Weaviate, pgvector) and injects them into the prompt. Lives between Router and Inference — adds tokens, adds latency. |
| **Feature Store** | Serves user-level context fast (recent actions, preferences, plan tier). Pre-materialized. Avoids hitting the DB on every inference call. |
| **Autoscaler** | Watches GPU utilization and queue depth. Provisions new workers on spot/reserved instances. Scales down after a cooldown. |
| **Observability Stack** | Captures every request: input tokens, output tokens, latency (TTFT + total), model version, cost. Feeds dashboards, alerts, and offline evals. |
| **Eval Pipeline** | Async job that samples production traces, runs them through LLM-as-judge or golden-set comparisons, and catches regressions before they compound. |

---

## Data Flow

A numbered walk through what happens when a user sends a message.

1. **Request arrives at the API Gateway.** Auth header is validated, the tenant's rate limit bucket is checked, and the payload is parsed. Malformed or oversized requests are rejected here — not at the GPU.

2. **Router inspects the request.** It reads the model hint (if any), checks the tenant's plan (does their SLA require the large model or is the small one fine?), and looks at current queue depths across replicas. It picks a target and stamps a `request_id`.

3. **Semantic Cache lookup.** The router hashes the prompt (or embeds it) and checks the cache. Cache hit — response is returned directly, zero GPU cost. Near-hit — the cached response surfaces as a candidate but the model still runs. Miss — continue.

4. **Context retrieval (RAG path).** If the product is RAG-enabled, the retriever runs a vector search against the knowledge base. Top-K chunks are fetched and appended to the system prompt. The feature store is also queried here for any user-specific personalization tokens.

5. **Prompt assembly.** The Prompt Manager stitches together: `system prompt` + `retrieved context` + `feature store signals` + `conversation history` + `user message`. The final prompt is token-counted. If it exceeds the context window, older turns are truncated or summarized.

6. **Inference.** The assembled prompt hits the chosen GPU worker. The worker uses continuous batching — it slots this request into an in-flight batch alongside other requests sharing the same iteration step. Tokens stream back via SSE as they're generated.

7. **Response returned.** Streamed tokens flow back through the router and gateway to the client. The final response is also written to the semantic cache asynchronously (fire and forget).

8. **Observability write.** A trace is emitted: request metadata, input/output token counts, TTFT, total latency, model version, cost estimate. This lands in a time-series store and is also enqueued for the async eval pipeline.

---

## Scaling Levers

**Horizontal GPU scaling** is the most direct. When queue depth rises, spin up more inference workers. This is the expensive lever — use it last.

**Continuous batching** (vLLM's core trick) lets one GPU handle many concurrent requests by batching at the iteration level, not the request level. You get much higher throughput per GPU before you need to scale out.

**Model tiering** routes cheap/short requests to a smaller, faster model (7B or 13B) and reserves the large model for complex queries or premium tenants. A well-tuned routing policy can cut GPU spend 40–60%.

**Semantic caching** is the highest-leverage lever early on. FAQ-style products (support bots, onboarding assistants) often see 20–40% cache hit rates. That's free throughput.

**Prompt caching** (e.g., Anthropic's prompt caching feature) prefills the KV cache for the static parts of your system prompt. Repeated long system prompts stop costing you on every call.

**Speculative decoding** uses a tiny draft model to propose tokens, and the large model verifies them in parallel. Cuts latency 2–3x on output-heavy workloads with no quality loss.

**Horizontal read replicas** for the vector store handle retrieval load independently of inference load — important to scale these separately.

---

## Failure Modes

| What breaks | Why | Guard |
|---|---|---|
| **GPU OOM** | Batch too large, or context length spike | Set max context limit in the gateway. Monitor KV cache usage. Autoscaler headroom. |
| **Cache stampede** | Sudden load + cold cache = all requests fall through to GPUs | Cache warming on deploy. Circuit breaker with a queue so the fleet doesn't get overwhelmed on cold start. |
| **Retrieval latency spike** | Vector DB under load or index rebuild | Timeout + fallback to no-context answer. Never let retrieval block the entire request indefinitely. |
| **Model version drift** | Shadow deployment of new model gets routed live traffic too early | Canary routing: 5% traffic to new version, compare eval scores before increasing. |
| **Thundering herd on startup** | Autoscaler spins up workers, they all register at once, router floods them before they're warm | Health check gate: workers only register as ready after first successful inference. |
| **Runaway tenant** | One tenant's batch job saturates the fleet | Per-tenant rate limits + priority queues. Burst limits with backpressure signaled via 429. |
| **Prompt injection** | User input escapes the prompt boundary | Input sanitization layer before prompt assembly. System prompt in a separate, unescapable slot (e.g., Anthropic's system turn). |

---

## Cost Levers

GPU inference cost is almost entirely a function of tokens in × tokens out × model size. Attack each axis.

**Reduce tokens in.** Compress conversation history (summarize older turns rather than carrying raw text). Tune your RAG retriever's K — do you really need 10 chunks, or does 3 do the job? Audit your system prompt for bloat.

**Reduce tokens out.** Add clear length instructions to your system prompt. Stream responses so users see output immediately and stop the call once they have what they need (partial generation).

**Use the right model.** Route by complexity. A "what are your business hours?" question does not need a 70B model. A coding task with 10k tokens of context might.

**Spot instances.** Inference workers are stateless and restartable. Spot/preemptible instances at 60–80% discount are viable if you have enough fleet redundancy to absorb evictions.

**Prompt caching.** Static system prompts and long documents that repeat across calls should be cached at the model layer. The savings compound fast at scale.

**Batch offline workloads separately.** Async jobs (bulk summarization, nightly evals, indexing) should run on separate, cheaper infrastructure and not compete with latency-sensitive live traffic.

---

## Tradeoffs & Alternatives

**Self-hosted vs. managed API** — Managed APIs (Anthropic, OpenAI, Vertex) have zero infra overhead but you lose cost control at scale and have limited deployment flexibility. Self-hosted (vLLM, TGI on your own GPUs) gives full control but adds significant ops burden. The crossover point is typically around a few million tokens per day — below that, managed wins. Above it, do the math.

**vLLM vs. TGI** — Both support continuous batching and tensor parallelism. vLLM has more community momentum and better speculative decoding support right now. TGI integrates cleanly into the Hugging Face ecosystem. Either is fine; the choice usually comes down on what your team already knows.

**Single large model vs. mixture of models** — A single powerful model is simple to operate. A tiered or mixture approach (small fast model for simple queries, large model for hard ones, specialized fine-tuned models for specific domains) is more complex to route but dramatically cheaper at scale. Build the routing before you need it.

**Serverless inference** — Platforms like Modal, Replicate, or AWS Bedrock serverless handle scale-to-zero automatically. Great for low-traffic or bursty workloads, expensive per-token at sustained high throughput. Know your traffic shape before committing.

**RAG vs. fine-tuning** — RAG is operationally simpler, keeps knowledge fresh, and lets you audit what the model retrieves. Fine-tuning bakes knowledge into weights — faster inference, no retrieval latency, but stale the moment your knowledge base changes. Most production systems use RAG for dynamic knowledge and fine-tuning for style/format/task alignment, not for knowledge.

:::why-prod
Latency is bimodal in LLM serving: time-to-first-token (TTFT) controls perceived responsiveness, and total generation time controls when the user can act on the response. Streaming fixes the UX of the second without changing the cost. Always stream in production.
:::

:::gotcha
The semantic cache is powerful but dangerous if you cache stale or hallucinated answers. Add a TTL and evict aggressively after model version changes. A cached wrong answer is worse than a fresh wrong answer because it repeats forever.
:::

:::interview-line
"We separate the serving concern from the retrieval concern. The gateway handles abuse and auth, the router handles model selection and load, the cache handles repeated work, and the inference fleet handles the actual compute. Each layer scales independently — and fails independently, which matters more."
:::

---

## How to Present This in an Interview

Start with the problem, not the architecture.

> "We're serving LLM inference at scale. The core challenge is that GPUs are expensive, latency is visible to users, and the load patterns are spiky. So I want a design that handles these three things: it never wastes GPU time on work we've already done, it routes requests to the right model so we're not burning a 70B model on a simple query, and it scales inference workers horizontally without losing any in-flight requests."

Then walk the happy path from request to response — Gateway → Router → Cache → Inference → Observability. One sentence per box.

When you hit scaling, frame it as a cost/latency tradeoff: "Semantic caching is free throughput. Model tiering is the second lever. Horizontal GPU scaling is the last resort because it's the most expensive."

For failure modes, pick two and go deep. Cache stampede and runaway tenants are interview favorites — they show you've operated this stuff, not just read about it.

Tie back to adjacent systems: "The retrieval side connects to the RAG track — the vector store here is the same system. The feature store is the same low-latency serving layer from the feature store track. The eval pipeline is the monitoring track. The design isn't standalone — it's the serving layer that sits in front of everything else."

:::key-takeaway
An LLM serving platform is a latency-sensitive, cost-sensitive pipeline. The expensive part is the GPU. Everything else — caching, routing, prompt compression, model tiering — exists to protect the GPU from doing unnecessary work. Design from that principle and the rest follows.
:::
