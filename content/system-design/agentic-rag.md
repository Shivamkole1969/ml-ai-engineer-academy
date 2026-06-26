# Agentic RAG / Multi-Agent System

Your team needs to build a support assistant for a fintech product. Users ask questions like "Why was my UPI transfer declined?" — and the answer might live across three places: a policy document, a live transaction database, and a third-party risk API. A plain RAG pipeline (embed → search → generate) can't do this alone. It needs to *plan*, *decide which tool to call*, *verify the result*, then *synthesise*. That's Agentic RAG.

---

<svg viewBox="0 0 860 130" width="100%" role="img" aria-label="Agentic RAG pipeline: User Query → Planner Agent → Tool Router → [Vector Store | SQL DB | External API] → Synthesiser → Response">
  <!-- User Query -->
  <rect x="10" y="42" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="65" y="61" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">User Query</text>
  <text x="65" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">+ history</text>

  <line x1="120" y1="64" x2="158" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="152,60 158,64 152,68" fill="#8b7bff"/>

  <!-- Planner Agent -->
  <rect x="158" y="42" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="218" y="61" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Planner Agent</text>
  <text x="218" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">LLM + memory</text>

  <line x1="278" y1="64" x2="316" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="310,60 316,64 310,68" fill="#8b7bff"/>

  <!-- Tool Router -->
  <rect x="316" y="42" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="371" y="61" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Tool Router</text>
  <text x="371" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">dispatch</text>

  <!-- Fork lines to tools -->
  <line x1="426" y1="64" x2="480" y2="30" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="426" y1="64" x2="480" y2="64" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="426" y1="64" x2="480" y2="98" stroke="#8b7bff" stroke-width="1.5"/>

  <!-- Vector Store -->
  <rect x="480" y="10" width="110" height="36" rx="8" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="535" y="32" fill="#eaf0ff" font-size="10" text-anchor="middle" font-family="monospace">Vector Store</text>

  <!-- SQL / DB -->
  <rect x="480" y="46" width="110" height="36" rx="8" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="535" y="68" fill="#eaf0ff" font-size="10" text-anchor="middle" font-family="monospace">SQL / Live DB</text>

  <!-- External API -->
  <rect x="480" y="82" width="110" height="36" rx="8" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="535" y="104" fill="#eaf0ff" font-size="10" text-anchor="middle" font-family="monospace">External API</text>

  <!-- Merge to Synthesiser -->
  <line x1="590" y1="28" x2="634" y2="64" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="590" y1="64" x2="634" y2="64" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="590" y1="100" x2="634" y2="64" stroke="#8b7bff" stroke-width="1.5"/>

  <!-- Synthesiser -->
  <rect x="634" y="42" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="689" y="61" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Synthesiser</text>
  <text x="689" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">grounded LLM</text>

  <line x1="744" y1="64" x2="782" y2="64" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="776,60 782,64 776,68" fill="#8b7bff"/>

  <!-- Response -->
  <rect x="782" y="42" width="68" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="816" y="61" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Response</text>
  <text x="816" y="76" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">+ citations</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Query analyser** | Parses intent, rewrites ambiguous language, injects session history |
| **Planner Agent** | Decomposes the question into sub-tasks; decides which tools are needed and in what order |
| **Tool Router** | Maps each sub-task to a concrete tool call (vector search, SQL, HTTP, code interpreter, etc.) |
| **Vector Store** | Semantic retrieval over docs, FAQs, policy PDFs — your classic RAG track |
| **Structured DB / Feature Store** | Live or near-real-time facts: account status, transaction history, feature flags |
| **External APIs** | Third-party signals: risk scores, KYC status, logistics tracking |
| **Sub-agents** | Specialised workers (e.g. a "refund eligibility" agent); each has its own prompt + tools |
| **Context / Memory store** | Short-term scratchpad (in-context) + long-term per-user key-value memory |
| **Synthesiser LLM** | Reads all tool outputs, reasons over them, produces a grounded, cited answer |
| **Guardrails layer** | Input/output checks: PII redaction, toxicity, hallucination scoring, citation verification |
| **Observability bus** | Trace IDs, span timings, tool-call logs → your monitoring track |

---

## Data flow

1. **User sends a query.** The query analyser strips PII where needed, injects the last N turns of chat history, and passes the enriched context to the Planner.

2. **Planner decomposes.** The Planner LLM (often a stronger, cheaper-to-call model like Claude Haiku or Sonnet) produces a plan: `["search_policy_docs(topic=upi_decline)", "query_db(txn_id=X)", "call_risk_api(user_id=Y)"]`. This is your ReAct / function-calling loop.

3. **Tool Router dispatches in parallel (where possible).** Steps with no dependency on each other run concurrently — this is the biggest latency win in agentic systems. The router enforces a per-request tool budget (max calls, max tokens, max wall-clock time).

4. **Tools execute.** Vector store returns top-K chunks with scores; the DB returns structured rows; the API returns JSON. Each result is tagged with a source ID for citation.

5. **Observation → re-plan (optional loop).** The Planner inspects the results. If a tool returned "no data" or low-confidence chunks, it can issue follow-up calls — up to a configurable max iteration cap.

6. **Synthesiser generates the answer.** All retrieved context is packed into a single grounded prompt. The Synthesiser produces the final response with inline citations (`[source_id]`). It does NOT call tools — it only reads.

7. **Guardrails run.** Output is checked for: hallucinated citations, PII leakage, policy violations. If a check fails, the response is either regenerated or replaced with a safe fallback.

8. **Response is streamed to the user.** Trace ID, latency breakdown, and tool call log are emitted to the observability bus.

---

## Scaling levers

- **Parallelise tool calls.** If sub-tasks are independent, fan them out concurrently. Cuts P50 latency by the number of parallel branches.
- **Cache aggressively.** Embed query → result cache (semantic hash key) for repeated questions. Cache external API responses with a short TTL. Saves cost and latency.
- **Agent pool with autoscaling.** Each sub-agent is stateless; deploy them as short-lived containers or serverless functions. Scale to zero when idle.
- **Model tiering.** Use a small/fast model for planning and a larger model only for final synthesis. 60–70% cost reduction in practice.
- **Context window management.** Truncate, summarise, or compress long histories before feeding the Synthesiser. Prevents token blowouts.
- **Vector index sharding.** Partition your vector store by domain (policy docs, product catalogue, legal). Router picks the right shard — smaller search space, faster recall.
- **Async / queue-backed agents.** For non-interactive queries (reports, batch reconciliation), push to a job queue and poll. Removes timeout pressure entirely.

---

## Failure modes

| What breaks | Why it hurts | The guard |
|---|---|---|
| Planner loops infinitely | LLM keeps re-planning without converging | Hard cap on iterations + wall-clock timeout; return partial answer |
| Tool returns empty / noisy results | Synthesiser hallucinates to fill the gap | Require minimum retrieval confidence score; fallback to "I don't know" |
| External API goes down | Dependent sub-tasks block forever | Circuit breaker + cached stale result with freshness label |
| Context window overflow | Truncation cuts critical context silently | Token budget enforced at Router; long-context compression summariser |
| Cascading agent failures | One failed sub-agent invalidates the whole answer | Partial-result mode: synthesise from available results, flag gaps |
| PII leakage through tool results | DB row contains Aadhaar / phone number in output | Guardrail scrubs structured output before it enters the Synthesiser prompt |
| Latency spikes | Slow external API stalls the whole response | Per-tool timeout + optimistic response without that tool's result |
| Prompt injection via docs | Retrieved chunk contains adversarial instructions | Separate system prompt from retrieved content; instruction hierarchy enforcement |

:::gotcha
The most common production failure is not a crash — it's *silent degradation*. The agent returns a confident answer built on low-quality retrieved chunks. Always emit retrieval confidence scores to your observability layer and alert when mean chunk score drops below threshold.
:::

---

## Cost levers

- **Input token cost dominates.** The Synthesiser prompt = plan + all tool outputs + history. Compress retrieved chunks (extractive summary before insertion). Trim history aggressively.
- **Cache tool results.** A vector search costs fractions of a cent, but at 10K QPS it adds up. Semantic caching (embed query, bucket by cosine similarity) gives 20–40% cache hit rates on real workloads.
- **Model routing.** Cheap model for planning + intent detection. Expensive model only for synthesis. Use eval datasets to find the cheapest model that hits your quality bar.
- **Batch non-urgent requests.** Async queue lets you batch Synthesiser calls, enabling prompt caching discounts offered by providers like Anthropic.
- **Retrieval top-K tuning.** K=10 vs K=3 is a 3x input token difference. Tune K per query type based on offline eval (precision@K vs cost).
- **Agent timeout = cost cap.** A stuck agent running a 10-second tool loop 1000 times an hour is a surprise bill. Timeouts are a cost control, not just a reliability feature.

:::why-prod
In Pune-based fintech teams I've seen, the surprise cost driver is almost always the Synthesiser prompt size — not the LLM call count. Measure tokens-per-request in your metrics dashboard before you optimise anything else.
:::

---

## Tradeoffs & alternatives

| Decision | Option A | Option B | When to pick B |
|---|---|---|---|
| Planning strategy | ReAct (interleaved reasoning + acting) | Plan-then-execute (upfront plan, no re-planning) | Latency-critical; user query is well-structured |
| Agent topology | Centralised Planner + sub-agents | Peer-to-peer multi-agent mesh | Complex, evolving domain; agents need to negotiate |
| Memory | In-context only | External memory store (Redis / vector DB) | Users expect continuity across sessions |
| Retrieval | Dense vector only | Hybrid (dense + BM25 keyword) | Queries contain exact product codes, IDs, legal terms |
| Guardrails | Post-generation check | Constrained decoding / structured output | Latency budget is tight; structured JSON response required |
| Orchestration | Custom Python loop | Framework (LangGraph, CrewAI, AutoGen) | Faster prototyping; accept framework lock-in risk |

:::key-takeaway
Agentic RAG is just RAG with a planning loop and multiple retrieval sources. The hard engineering is in the *planner loop control* (avoid infinite loops), *parallel dispatch* (latency), and *context budget management* (cost). Nail those three and the rest is wiring.
:::

---

## How to present this in an interview

Lead with the problem, not the acronym.

> "Standard RAG works when all the answer context lives in one place — a doc corpus. But real enterprise queries span structured data, live APIs, and documents simultaneously. So we add a Planner agent that decomposes the question, dispatches to the right tools in parallel, and a Synthesiser that produces a grounded answer from all the results. The key production concerns are: bounding the planning loop so it can't spin forever, caching tool results to control cost, and making retrieval quality observable so silent degradation doesn't go unnoticed."

Then offer to go deep on whichever dimension the interviewer cares about — latency (parallel dispatch, model tiering), reliability (circuit breakers, partial-result mode), or cost (caching, context compression).

**Tie-back to tracks:**
- *RAG track* — vector store, chunk sizing, hybrid retrieval, top-K tuning
- *Serving / inference track* — model tiering, streaming, latency SLOs
- *Feature store track* — structured DB lookups, feature freshness, cache TTLs
- *Monitoring track* — trace IDs, retrieval confidence metrics, per-agent span timing, alerting on quality regression
- *Data pipeline track* — document ingestion, embedding jobs, index refresh cadence
