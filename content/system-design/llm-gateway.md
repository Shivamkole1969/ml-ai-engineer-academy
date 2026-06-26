# Multi-tenant LLM Gateway

Your team just shipped an internal AI platform. Three product squads, two data-science groups, and one compliance team all want to call GPT-4o. Each team has a different budget, a different SLA, and a very different tolerance for hallucination risk. You could hand everyone an API key and pray — or you could build a **multi-tenant LLM gateway**: a single controlled chokepoint that every team calls, and that routes, rates-limits, audits, and bills them correctly.

---

## Architecture at a glance

<svg viewBox="0 0 860 130" width="100%" role="img" aria-label="Multi-tenant LLM gateway pipeline: Client → Auth/Tenant Resolver → Rate Limiter → Prompt Guard → Router → Model Backend → Response Auditor → Client">
  <!-- Client -->
  <rect x="10" y="42" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="60" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Client</text>
  <text x="60" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(any team)</text>
  <line x1="110" y1="64" x2="145" y2="64" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Auth / Tenant Resolver -->
  <rect x="145" y="42" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="200" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Auth &amp;</text>
  <text x="200" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Tenant ID</text>
  <line x1="255" y1="64" x2="285" y2="64" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Rate Limiter -->
  <rect x="285" y="42" width="105" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="337" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Rate Limiter</text>
  <text x="337" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(per tenant)</text>
  <line x1="390" y1="64" x2="420" y2="64" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Prompt Guard -->
  <rect x="420" y="42" width="105" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="472" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Prompt Guard</text>
  <text x="472" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">&amp; PII filter</text>
  <line x1="525" y1="64" x2="555" y2="64" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Router -->
  <rect x="555" y="42" width="95" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="602" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Model</text>
  <text x="602" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Router</text>
  <line x1="650" y1="64" x2="680" y2="64" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Backend -->
  <rect x="680" y="42" width="95" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="727" y="62" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">LLM Backend</text>
  <text x="727" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(OAI / self)</text>

  <!-- arrow marker -->
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="#8b7bff"/>
    </marker>
  </defs>
</svg>

---

## Components

| Component | Role |
|---|---|
| **API Gateway / Ingress** | TLS termination, connection pooling, first-line throttling |
| **Auth & Tenant Resolver** | Validates API key or JWT, attaches `tenant_id` + policy config to every request |
| **Rate Limiter** | Per-tenant token-bucket or sliding-window limits; backed by Redis for distributed state |
| **Prompt Guard** | PII detection, prompt-injection heuristics, regex/classifier-based content policy |
| **Model Router** | Maps (tenant + task type + budget tier) → model endpoint; supports fallback chain |
| **Semantic Cache** | Embedding-based exact-and-near-match cache; cuts duplicate spend significantly |
| **LLM Backend pool** | OpenAI, Anthropic, Azure OpenAI, or self-hosted vLLM/TGI behind a unified adapter |
| **Response Auditor** | Logs completion + metadata, checks output safety, strips residual PII |
| **Usage Metering** | Writes token counts + latency to a time-series store; drives billing & dashboards |
| **Admin API** | CRUD for tenants, quotas, model policies, cost budgets |
| **Observability** | Traces (OpenTelemetry), metrics (Prometheus/Grafana), log aggregation (Loki/CloudWatch) |

---

## Data flow

A request travels through the gateway like this.

1. **Client sends request.** Any internal service or product team POSTs to `https://llm-gateway.internal/v1/chat/completions` — same OpenAI-compatible schema so existing SDKs work without changes.

2. **Auth & Tenant Resolver.** The gateway extracts the bearer token, looks it up in a fast in-memory store (or Redis), and attaches `tenant_id`, tier, and model policy. Requests without a valid token get a 401 immediately.

3. **Rate limiter.** A sliding-window counter keyed on `(tenant_id, minute)` is checked against the tenant's configured TPM (tokens per minute) quota. If the counter is already at the limit, the gateway returns 429 with a `Retry-After` header. This protects every other tenant from a single runaway caller.

4. **Prompt Guard.** The prompt is scanned for PII (names, phone numbers, Aadhar-like patterns), prompt-injection strings (`ignore previous instructions…`), and policy violations (hate, CSAM, competitive intel depending on tenant config). PII is masked or the request is rejected based on policy.

5. **Semantic Cache lookup.** The prompt embedding is compared against cached entries in a vector store (Qdrant/Redis VSS). A hit above a similarity threshold returns the cached completion without ever calling the LLM — instant and free.

6. **Model Router.** On a cache miss, the router reads the tenant's policy: `{"preferred": "gpt-4o", "fallback": "gpt-4o-mini", "allow_self_hosted": true}`. It picks the cheapest model that satisfies the task's latency/quality requirements. It also does cost-budget checks — if the tenant's monthly spend is near their cap, it may force-downgrade to a smaller model.

7. **LLM Backend call.** The request goes to the chosen backend via an async HTTP client. The gateway streams the response back to the caller using SSE/chunked transfer, so the user sees tokens as they arrive.

8. **Response Auditor.** Each completion chunk (or the final assembled response) is scanned for output-side policy violations and residual PII. Metadata — model used, input/output token counts, latency, tenant, request ID — is written to the metering store.

9. **Billing & observability.** A background consumer aggregates usage events into per-tenant dashboards and triggers alerts when a tenant is within 10% of their monthly budget cap.

---

## Scaling levers

**Horizontal scale the gateway itself.** The gateway is stateless at the request level; all shared state (rate-limit counters, cache, tenant config) lives outside it. Deploy as many pods as you need behind a load balancer. Kubernetes HPA on CPU or request-rate works well.

**Redis Cluster for rate limiting.** A single Redis node can handle ~100K ops/sec easily. When you have thousands of tenants hammering it, shard by tenant-id range across a Redis Cluster. Use Lua scripts for atomic read-increment-expire — avoids race conditions.

**Semantic cache eviction policy.** Cache hit rates vary wildly by workload. Customer-support bots asking the same FAQ-style prompts hit 40–60%. Creative writing bots hit near 0%. Let tenants opt out of caching if freshness matters. Use TTLs based on content type — factual answers can live for hours; breaking-news prompts should have very short TTLs.

**Async backends with connection pooling.** LLM APIs are slow (seconds per response). Use async HTTP (httpx, aiohttp, or a Rust proxy like `litellm-proxy`) and pre-warm a pool of keep-alive connections per backend. Avoids TLS handshake cost on every request.

**Priority queuing under load.** When the gateway is at capacity, route enterprise-tier tenant requests to a high-priority queue and free-tier traffic to a best-effort queue. This keeps SLA promises without over-provisioning for average load.

---

## Failure modes

| What breaks | Impact | Guard |
|---|---|---|
| **OpenAI API outage** | All tenants on that model go dark | Fallback chain to Azure OpenAI or self-hosted; circuit breaker trips after N failures |
| **Redis unavailable** | Rate limiting blind; cache unusable | Gateway fails open on rate-limiting (log + alert); falls back to in-process counters per pod |
| **Prompt Guard slow** | P99 latency spikes; timeout cascade | Async guard with 200 ms hard timeout; on timeout → pass through + flag for async audit |
| **Tenant over-budget** | Surprise bills for platform team | Soft cap (alert at 80%) + hard cap (reject at 100%) stored in tenant config |
| **Streaming mid-flight failure** | Client gets partial response, no error | Gateway buffers last chunk sequence; on backend disconnect → send error event on SSE stream |
| **Prompt injection** | Tenant A's prompt manipulates system prompt to leak Tenant B's data | Tenant namespace isolation in system prompt; output auditor checks for cross-tenant tokens |

:::gotcha
The scariest multi-tenant failure is **data bleed** — Tenant A's context leaking into Tenant B's response, usually via a misconfigured shared conversation history or a prompt cache that doesn't key on tenant_id. Always namespace your cache keys as `(tenant_id, hash(prompt))`, never just `hash(prompt)`.
:::

---

## Cost levers

**Model routing by complexity.** Not every request needs GPT-4o. A simple intent-classification call can go to a cheap small model (Haiku, GPT-4o-mini, or a self-hosted Llama 3.1 8B). Route by task type, prompt length, or even a fast pre-classifier. Most platforms see 60–70% of requests that can be handled by the small model tier.

**Semantic caching.** Duplicate or near-duplicate prompts are shockingly common in production — especially in RAG pipelines where the retrieved context doesn't change much. A good semantic cache can cut LLM spend by 20–40% on FAQ/support workloads.

**Prompt compression.** Long system prompts with boilerplate instructions repeated every request are expensive. Store them server-side and reference them by ID. On Anthropic models, use prompt caching — up to 90% cost reduction on the static portion of the prompt.

**Token budgets per tenant.** Hard monthly token budgets prevent one team from burning your entire platform bill over a long weekend. The router checks remaining budget before each call and can force-downgrade to a smaller model rather than reject the request outright.

**Batching for async workloads.** If a tenant's use case is offline (batch summarization, nightly report generation), queue requests and batch them via the Batch API where available. OpenAI's Batch API costs 50% less than the synchronous API.

:::why-prod
Cost is the number one reason platform teams build a gateway. When you have 20 teams with direct API keys, your bill is unpredictable and you can't attribute spend. A gateway makes cost a first-class concern — you know exactly which team, which model, and which use case is responsible for every rupee spent.
:::

---

## Tradeoffs & alternatives

**Gateway vs. direct API keys.** Direct keys are simpler to start. Gateways add latency (5–20 ms overhead) and an operational burden. The crossover point is roughly: >3 teams, or any compliance/audit requirement, or a real budget.

**Build vs. buy.** OSS options like LiteLLM, Portkey, and Helicone give you 80% of a gateway out of the box. Build your own only if you have unusual routing logic, deep on-prem constraints, or want to avoid vendor lock-in on the gateway itself.

**Sidecar vs. centralised gateway.** In a service-mesh world, you could run an LLM-aware sidecar proxy next to each service. Centralised is easier to operate and reason about for cross-cutting concerns like billing; sidecars give better blast-radius isolation.

**Streaming vs. buffered responses.** Streaming is UX-essential for chat. But auditing and metering are easier on a complete response. Solution: audit asynchronously in a separate queue for non-critical compliance; only block for hard safety checks.

**Semantic cache consistency.** A cache tuned for high hit-rate can return stale knowledge (model trained in a certain month, cache TTL hasn't expired). For tenants where freshness matters more than cost, let them pass a `Cache-Control: no-store` header.

---

## How to present this in an interview

:::interview-line
"A multi-tenant LLM gateway is basically an API proxy with four jobs: authentication and tenant isolation, traffic control and rate-limiting, intelligent model routing to balance cost and quality, and full observability so the platform team can audit and bill correctly. The hardest part isn't the happy path — it's ensuring strict tenant isolation so one team's prompts and responses never touch another's, and building a fallback chain so a single model outage doesn't take down all your product squads at once. The semantic cache is the surprise cost lever: on repetitive workloads it can cut LLM spend by 30–40% without any change to the callers."
:::

**Tie to adjacent tracks:**

- **RAG** — the gateway is where retrieval results land before going to the LLM. Prompt compression and context-window budgeting belong here.
- **Serving & inference** — if you run self-hosted models (vLLM, TGI), the gateway manages the load balancer in front of them.
- **Feature store** — per-tenant policy config (model preferences, cost caps, content policy) is essentially a feature store for routing logic; it needs fast reads and consistent writes.
- **Monitoring** — token usage, latency histograms, cache hit rates, safety-check rejection rates, and cost-per-tenant are your SLIs. Alert when a tenant's rejection rate spikes — it usually means a prompt regression.

:::key-takeaway
The gateway turns LLM access from a shared, uncontrolled resource into a managed platform primitive. Tenant isolation + smart routing + semantic caching are the three levers that separate a toy deployment from production infrastructure.
:::
