---
id: sysdesign-genai
track: t7-systemdesign-capstones
title: "How GenAI system design differs"
badge: HOT
minutes: 9
prereqs: []
tags: [system-design, genai, llm, latency, cost, architecture]
xp: 60
hot2026: true
---

Imagine you just joined a product team. Your manager drops a ticket on your desk: "Design a customer support chat feature powered by an LLM." You open your trusty system design playbook — load balancers, databases, caches, queues. All the classics. And then you realize: this is not a classic problem. The "database" thinks. The "query" costs $0.02 per call. And the "response time" is measured in seconds, not milliseconds.

Welcome to GenAI system design. Same foundation, wildly different constraints.

## Why GenAI is a different beast

Traditional systems handle **deterministic, fast, cheap** operations. A user queries a database — you get a row back in 5 ms for a fraction of a cent. Horizontally scale, and you're done.

GenAI systems handle **probabilistic, slow, expensive** operations. A user sends a message to an LLM — you get a streamed response in 2–15 seconds, costing real money per request. And the output might be slightly different every time you call it.

This flips several design assumptions on their head.

:::why-prod
In production, a GenAI feature that "works in the demo" can silently bankrupt a team at scale, degrade to 30-second responses under load, or return inconsistent answers that erode user trust. You need to design for cost, latency, and quality from day one — not as an afterthought.
:::

## The five dimensions that change everything

:::table {title="Classic vs GenAI system design dimensions"}
| Dimension | Classic System | GenAI System |
|---|---|---|
| Latency target | 10–200 ms | 500 ms – 15 s (often streamed) |
| Cost per request | Fractions of a cent | $0.001 – $0.10+ |
| Output determinism | Fully deterministic | Probabilistic (temperature > 0) |
| Scaling unit | Compute (CPU/memory) | Tokens throughput + API rate limits |
| Failure mode | Timeout / crash | Hallucination / refusal / truncation |
:::

Let's unpack the ones that trip people up most.

## Latency is a different problem

In a typical API, you optimize for time-to-first-byte. In a GenAI system, you optimize for **time-to-first-token** (TTFT) and **tokens-per-second** (TPS). These are distinct metrics with distinct levers.

Users will tolerate a 5-second wait *if* they see words appearing immediately. A blank screen for 3 seconds then a wall of text? Feels broken. This is why streaming is not optional for user-facing GenAI features — it is table stakes.

Your architecture has to support streaming end-to-end: from the LLM API, through your backend, over the wire to the client. Server-Sent Events (SSE) or WebSockets, not plain REST.

## Cost is a first-class constraint

In a classic system, you think about cost at the infrastructure layer. In GenAI, cost is baked into every single user request, and it scales with *input length + output length*, not just request count.

A sloppy system prompt? That's 500 extra tokens on every call. A retrieval system that stuffs 10 documents into context? Costs 3× more than stuffing 3. At 10,000 daily active users, that carelessness turns into thousands of dollars per month.

```python {title="Estimate monthly LLM cost before you build" run=false}
# Quick back-of-envelope cost estimator
# Run locally with: python cost_estimate.py
# No API key needed — this is pure math

DAILY_ACTIVE_USERS = 10_000
REQUESTS_PER_USER_PER_DAY = 3

# Token counts (rough estimates)
AVG_INPUT_TOKENS = 800   # system prompt + user message + retrieved context
AVG_OUTPUT_TOKENS = 300  # model response

# Pricing (check current rates — these are illustrative)
INPUT_PRICE_PER_1K  = 0.003   # $ per 1K input tokens
OUTPUT_PRICE_PER_1K = 0.015   # $ per 1K output tokens

total_requests_per_day = DAILY_ACTIVE_USERS * REQUESTS_PER_USER_PER_DAY
daily_input_cost  = (total_requests_per_day * AVG_INPUT_TOKENS  / 1000) * INPUT_PRICE_PER_1K
daily_output_cost = (total_requests_per_day * AVG_OUTPUT_TOKENS / 1000) * OUTPUT_PRICE_PER_1K
daily_total = daily_input_cost + daily_output_cost

print(f"Daily requests:     {total_requests_per_day:,}")
print(f"Daily input cost:   ${daily_input_cost:,.2f}")
print(f"Daily output cost:  ${daily_output_cost:,.2f}")
print(f"Daily total:        ${daily_total:,.2f}")
print(f"Monthly estimate:   ${daily_total * 30:,.2f}")
```

Run this math before you write a single line of feature code. It tells you whether to use a frontier model, a cheaper model, or add semantic caching on top.

## Failure modes are qualitative

A classic server either responds or it doesn't. You can write a test for that. An LLM can respond with *something confidently wrong*. That's a failure mode your monitoring dashboard won't catch unless you design for it.

GenAI failure modes you must plan for:

- **Hallucination** — factually incorrect but fluent output
- **Refusal** — model declines to answer due to safety guardrails
- **Context overflow** — input exceeds model's context window and content is silently truncated
- **Prompt injection** — user input manipulates your system prompt
- **Latency spikes** — third-party API slowness cascades to your users

:::gotcha
Don't treat the LLM API as a fast, reliable internal service. It's a third-party, probabilistic, rate-limited dependency. Always set timeouts, implement retries with exponential backoff, and have a graceful fallback (cached response, simpler model, or a friendly error message). Teams that skip this ship a feature that goes dark during the API provider's next outage.
:::

## The new architectural primitives

GenAI systems introduce components that don't exist in classic designs:

**Prompt layer** — the managed template that wraps every user request. Versioned like code, tested like code.

**Context builder** — retrieves and ranks documents, conversation history, or tool outputs to inject into the prompt. (This is the RAG pattern.)

**LLM gateway** — a thin proxy that handles routing between models, rate-limit management, logging, and cost tracking. Don't call the LLM API directly from every microservice.

**Evaluation pipeline** — automated checks that score output quality. Runs in CI/CD and in production on sampled requests.

**Semantic cache** — stores LLM responses keyed by embedding similarity, not exact string match. Cuts costs dramatically for repeated or near-duplicate queries.

:::war-story {title="The $40k surprise invoice"}
A team launched an internal knowledge-base chatbot. It went viral inside the company — 2,000 employees started using it daily instead of the expected 200. The system prompt was verbose (1,200 tokens), the retrieval stuffed 8 documents into context, and there was zero caching. Three weeks later, the cloud bill arrived: $40k for the month. The feature was immediately taken down for "maintenance" while they added semantic caching and rewrote the prompt to 300 tokens. Monthly cost dropped to under $4k. Same feature. Better design.
:::

## How to frame your answer in an interview

When asked "Design a GenAI feature," don't jump straight to boxes and arrows. Start by surfacing the GenAI-specific constraints:

1. What's the acceptable latency? (Streaming required?)
2. What's the cost budget per request?
3. What are the quality/safety requirements?
4. Is the output user-facing or machine-consumed?

These questions signal that you understand GenAI is *not* just another CRUD service.

:::interview-line
"Before I draw the architecture, I want to nail down the token budget, latency target, and quality bar — because those three constraints drive almost every design decision in a GenAI system."
:::

:::qa {q="How is designing a GenAI system different from designing a traditional REST API?"}
GenAI systems add three new first-class constraints: cost (charged per token, not per request type), latency (seconds, not milliseconds, with streaming as the baseline UX), and output quality (probabilistic outputs require evaluation pipelines, not just unit tests). You also need a new set of components — prompt layer, context builder, LLM gateway, semantic cache — that have no direct equivalent in classic system design.
:::

:::qa {q="What is a semantic cache and when would you use it?"}
A semantic cache stores LLM responses indexed by the embedding of the input, not an exact string match. When a new query arrives, you compute its embedding and look for a sufficiently similar cached query. If found, you return the cached response and skip the LLM call entirely. Use it when your query distribution has significant repetition or paraphrasing — common in FAQ-style chatbots, search assistants, and internal knowledge tools. It can cut LLM costs by 30–70% in the right workloads.
:::

:::drill {type="mcq" q="A user-facing GenAI chat feature shows a blank screen for 4 seconds then dumps the full response. What is the most impactful fix?"}
- [ ] Switch to a faster LLM model
- [ ] Add a semantic cache layer
- [x] Enable streaming (SSE or WebSocket) so tokens render as they are generated
- [ ] Reduce the system prompt length
:::

:::drill {type="mcq" q="Your GenAI feature's daily API cost is 10× higher than projected. Which investigation step is MOST likely to find the biggest savings?"}
- [ ] Reduce the number of daily active users
- [ ] Switch to a more expensive model with better output quality
- [x] Audit token usage: measure system prompt length, retrieved context size, and average output length per request
- [ ] Add more backend servers to reduce latency
:::

:::key-takeaway
GenAI system design is classic system design plus three new constraints — token cost, streaming latency, and probabilistic output quality — and a new set of components (prompt layer, LLM gateway, semantic cache, eval pipeline) to manage them. Surface these constraints first; the architecture follows.
:::
