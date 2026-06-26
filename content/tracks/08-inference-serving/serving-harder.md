---
id: serving-harder
track: 08-inference-serving
title: "Why serving is harder than training (the tail is the product)"
badge: CORE
minutes: 8
prereqs: []
tags: [inference, serving, latency, p99, slo, production, tail-latency]
xp: 45
hot2026: false
---

Your training run finished last night. Loss curve: textbook. Eval scores: impressive. You export the model, wire up a FastAPI endpoint, and push it to staging. First ten requests look great — 130 ms average. Your manager demos it live to the client. Request eleven? Two full seconds. The client looks at the screen. Your manager looks at you.

That one slow request is the entire story of why serving is harder than training.

## Training and Serving Are Completely Different Problems

When you train, *you* control everything: batch size, data order, hardware load, when to stop. It is an offline job. If one step is slow, you don't care — the next step averages it out. The metric that matters is throughput: tokens per second across the whole GPU.

Serving flips every assumption.

- You don't control when requests arrive.
- You don't know how long the output will be.
- Multiple users hit you simultaneously.
- And the user *feels every single slow response*.

Training is a scientific experiment you run on your schedule. Serving is a live performance with a crowd you cannot see.

:::why-prod
In production, your p99 latency — the slowest 1 in 100 requests — defines the worst experience a user reliably gets. At 10,000 daily active users, that is 100 people per day hitting your slow tail. Those 100 write the one-star reviews and open the support tickets.
:::

## p50 Is a Lie. p99 Is the Product.

Here is the trap most engineers fall into: they measure average latency, see 140 ms, and ship it.

Averages hide the tail.

Imagine 99 requests complete in 120 ms. One request — because the user asked for an 800-token response instead of 50 — takes 2,400 ms. Your *average* is 144 ms. Looks healthy on the dashboard. But that one user waited 2.4 seconds.

At scale, "1 in 100" is not rare. It is constant.

The industry uses percentiles to tell the truth:

- **p50**: median — half of requests are faster, half are slower
- **p95**: 95% of requests land at or below this number
- **p99**: the worst 1% — what a typical user hits if they use the product long enough
- **p999**: the worst 0.1% — rare but real, and often catastrophic for streaming responses

SLOs (Service Level Objectives) are nearly always written as p99 targets: *"p99 latency < 500 ms"*. If you only measure average, you are flying blind.

```python {title="Revealing your latency tail" run=false}
import numpy as np

# Simulate 1000 request latencies (milliseconds).
# Most complete fast; a small fraction hit long outputs or KV-cache pressure.
np.random.seed(42)
fast = np.random.normal(loc=120, scale=20, size=950)    # typical short responses
slow = np.random.normal(loc=800, scale=100, size=50)    # long-output requests
latencies = np.concatenate([fast, slow])

p50  = np.percentile(latencies, 50)
p95  = np.percentile(latencies, 95)
p99  = np.percentile(latencies, 99)
p999 = np.percentile(latencies, 99.9)

print(f"p50  (median): {p50:.0f} ms   ← your CEO sees this in demos")
print(f"p95:           {p95:.0f} ms")
print(f"p99:           {p99:.0f} ms   ← your users feel this at scale")
print(f"p999:          {p999:.0f} ms  ← your on-call engineer gets paged for this")

# Run locally with: pip install numpy  (no GPU required)
```

## Why the Tail Spikes in LLM Serving Specifically

Training batches are uniform — you pad every sample to the same length, process them in lockstep, and iterate. Serving requests are wildly variable in every dimension.

:::table {title="Training vs. Serving: the real differences"}
| Dimension | Training | Serving |
|---|---|---|
| Request timing | You control | User controls |
| Output length | Fixed (labels) | Unknown until EOS token |
| Primary metric | Throughput (tokens/GPU/sec) | p99 latency AND throughput |
| Error impact | Bad batch → silent retry | Bad request → user sees it live |
| Concurrency | Fixed batch size | Unbounded simultaneous users |
| Memory pressure | Predictable per step | Spikes with long outputs and burst traffic |
:::

Three things drive the tail in LLM serving and nowhere else:

**1. Variable output length.** You start generating a response but you cannot know when it ends until the EOS token appears. A "write me a tagline" request might finish in 12 tokens. "Summarise this 8,000-word contract" might take 600. Same endpoint — 50× latency difference. You cannot even time-out intelligently without knowing the expected length.

**2. KV cache memory pressure.** Every token in flight occupies GPU memory in the KV cache. Long requests eat into the same pool that short requests need. When memory gets tight, new requests queue behind the giants — or the system evicts cache entries and recomputes them mid-generation. Either way, your tail explodes. (vLLM's paged attention directly attacks this — that is covered in the vLLM lesson.)

**3. Request interference inside shared batches.** Without careful scheduling, one enormous concurrent request stalls every short request sitting in the same batch. A single bad actor inflates the p99 for everyone sharing the GPU.

:::gotcha
Do not optimise for p50 and call it done. A beautiful average with a terrible p99 is the most common serving mistake. Instrument p95 and p99 from day one — not after the first incident. Add them to your monitoring before you add any features.
:::

:::war-story {title="The demo that looked fine until it didn't"}
A team built a document-QA product and tested it internally with short queries. Average latency: 200 ms, rock solid. They measured only the average. At launch, a few beta users pasted in long legal contracts. The KV cache filled up, new short requests queued behind the giants, and p99 silently ballooned to 4 seconds. The average barely moved — it crept to 240 ms. Nobody noticed until users started complaining. The fix took two days: output-length limits, a queue depth alert on p99, and request-size bucketing on the dashboard. The lesson: the average told them nothing.
:::

:::interview-line
"Training optimises for throughput on controlled, fixed-length batches. Serving optimises for p99 latency on variable, concurrent, user-controlled inputs — and the tail is what users actually experience."
:::

:::qa {q="Why is p99 latency more important than average latency for a serving system?"}
Averages hide the tail entirely. At any meaningful scale, 1-in-100 requests is a continuous stream of real users hitting a slow experience. SLOs are written in percentiles because percentiles capture what users actually feel. A system with 130 ms average but 3,000 ms p99 will generate complaints and churn even while the dashboard looks healthy.
:::

:::qa {q="What three things cause tail latency spikes specifically in LLM inference — and not in traditional web APIs or classical ML?"}
First, variable output length — autoregressive generation cannot predict token count upfront, so a single long response ties up resources unpredictably. Second, KV cache memory pressure — long requests compete for the same GPU memory pool as short ones, causing queuing or recomputation. Third, request interference in shared batches — one heavy request can stall all co-scheduled lighter ones. None of these exist in, say, an image classifier endpoint or a REST CRUD API.
:::

:::drill {type="mcq" q="Your serving dashboard shows: p50 = 110 ms, p95 = 430 ms, p99 = 2,800 ms. Your SLO is p99 < 500 ms. What is true?"}
- [ ] You are meeting your SLO — the majority of requests are fast
- [ ] Only p95 matters; p99 is too rare to worry about
- [x] You are breaching your SLO — 1% of users wait nearly 3 seconds per request
- [ ] The p50 is too high and should be investigated first
:::

:::drill {type="mcq" q="Why does output-length variability cause tail spikes in LLM serving but NOT in a traditional image-classification endpoint?"}
- [ ] Image classifiers use larger batch sizes and absorb variance better
- [ ] LLMs always produce longer outputs than image models
- [ ] Image classifiers run on CPU and avoid KV cache constraints entirely
- [x] LLM generation is autoregressive — each token is produced sequentially and the total count is unknown upfront, tying up memory for an unpredictable duration
:::

:::key-takeaway
Training is an offline throughput problem you control. Serving is a real-time tail-latency problem your users control. The p99 — not the average — is what they experience, and variable output length plus KV cache pressure are the two root causes that make LLM serving tails uniquely hard to tame.
:::
