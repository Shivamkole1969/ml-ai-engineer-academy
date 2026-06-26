---
id: tail-latency
track: 13-monitoring-cost
title: "Latency, outliers & the tail (percentiles don't average)"
badge: HOT
minutes: 9
prereqs: []
tags: [monitoring, latency, percentiles, slo, production, inference]
xp: 60
hot2026: true
---

It's 11 PM on a Thursday. Your ML inference service's dashboard is green. Average latency: 95ms. SLO says 200ms. You close your laptop.

At 2 AM, Slack erupts. A key enterprise client says their product "feels broken." Their engineers share traces: some requests hit 8 seconds. You stare at your dashboard. Still green. Average: 98ms. What happened?

The average lied to you. Your p99 was screaming the whole time — you just weren't listening.

## Averages are liars

Take 999 requests at 90ms and 1 request at 8000ms. The average? About 98ms. Looks fine. But that 1 request was a real user — maybe your most important one.

This is the **tail latency problem**: the slowest requests lurking at high percentiles are invisible to averages, yet they hit real people.

Percentiles give you the truth:

- **p50 (median):** Half of requests are faster. Your "typical" user experience.
- **p95:** 95 of 100 requests are faster. Your "usually-unhappy" threshold.
- **p99:** 99 of 100 are faster. Your SLO violation zone.
- **p99.9:** 999 of 1000 are faster. Your enterprise / high-value user zone.

If your p99 is 2 seconds and your SLO is 500ms, you are breaching that SLO for 1% of traffic. At 1000 req/s, that is 10 users getting a broken experience every single second.

:::why-prod
Enterprise SLAs are written against p99 or p99.9 — not averages. Averaging hides outliers by construction. One slow request per hundred is invisible to the mean but felt by every user who got it. You cannot debug what you are not measuring.
:::

## What causes tail latency in ML systems

ML services have unique tail-latency triggers that normal web APIs do not face:

- **Cold model loads:** The first request after a scale-up event hits a pod that has not loaded weights yet.
- **GC pauses:** Python garbage collection — or the JVM — can freeze processing for tens of milliseconds at the worst moment.
- **Batching wait:** If you batch for GPU efficiency, a small batch waits for a full window to fill. Low-traffic periods mean every request sits idle for the timeout.
- **Thundering herd:** A cached embedding expires and 50 simultaneous requests all try to recompute it at once.
- **Lock contention:** A shared tokenizer or preprocessor with a thread lock creates a queue behind it.

:::table {title="Percentile reference for ML inference services"}
| Percentile | What it means | Typical use |
|---|---|---|
| p50 | Half of requests are faster | Internal health check |
| p90 | 1 in 10 is slower | Soft internal target |
| p95 | 1 in 20 is slower | Common product SLO |
| p99 | 1 in 100 is slower | Enterprise / API SLO |
| p99.9 | 1 in 1000 is slower | Critical / payment flows |
:::

## Measuring it yourself

Do not wait for a Datadog alert to understand your distribution. A quick local sanity check goes a long way:

```python {title="Percentile check on recorded latencies" run=false}
import numpy as np

# Replace with your real latency logs (in ms)
# Simulate: 1000 requests, mostly fast, a handful of slow outliers
rng = np.random.default_rng(42)
latencies_ms = np.concatenate([
    rng.normal(loc=90, scale=10, size=980),    # typical fast requests
    rng.normal(loc=4000, scale=500, size=20),  # outlier slow ones
])

percentiles = [50, 90, 95, 99, 99.9]
for p in percentiles:
    val = np.percentile(latencies_ms, p)
    print(f"p{p:<5} -> {val:>8.1f} ms")

# Run locally: pip install numpy && python latency_check.py
# You will see: p50 looks healthy (~90ms), p99.9 is catastrophic (~4500ms)
# The average would have reported ~170ms and told you almost nothing.
```

The average here comes out around 170ms — comfortably under a 200ms SLO. The p99 tells the real story.

:::gotcha
Never set your paging alert on average latency. It is mathematically guaranteed to smooth over your worst user experiences. Set alerts on p95 or p99 — whichever percentile your SLO is written against — and page on that number. Teams that alert on the mean will sleep through real incidents.
:::

:::war-story {title="The batch timeout that owned the weekend"}
A recommendation service batched embedding requests in groups of 32 to saturate the GPU. The batch timeout was 50ms — if 32 requests did not arrive in time, the service waited the full window anyway. At low traffic (nights, weekends) only a few requests arrived per second. Every request sat idle for 50ms before anything ran. The p50 was 140ms (acceptable). The p99 ballooned to 800ms because users kept hitting that timeout wall. The team only noticed when a B2B client ran a Monday morning load test and escalated an SLO breach. The fix: make the batch timeout adaptive — if traffic is low, flush early instead of waiting for the window.
:::

:::interview-line
"We do not monitor average latency — we alert on p99. Averages hide the tail, and the tail is exactly what your SLO is measured against."
:::

:::qa {q="Why is p99 more useful than average latency for production ML services?"}
The average is dominated by the fast majority and smooths over outliers by design. p99 shows you the worst experience that 1% of users are getting — a concrete, actionable number. At any meaningful scale, 1% of traffic is a large cohort of real people, often the users most likely to complain or churn.
:::

:::qa {q="What is a common ML-specific cause of tail latency that you would not see in a typical CRUD API?"}
Batching strategies cause latency spikes at low traffic: if the batch window waits for N requests before processing, individual requests idle for the full timeout rather than being served immediately. Cold model loads after autoscaling events are another classic source — the first request to a freshly spun-up pod can be 10 to 100x slower than steady-state because model weights have not yet loaded into GPU memory.
:::

:::drill {type="mcq" q="Your inference service handles 1000 requests. 998 complete in 85ms. 2 complete in 6000ms. What does average latency report approximately?"}
- [ ] 6000 ms
- [ ] 3000 ms
- [x] 97 ms
- [ ] 500 ms
:::

:::drill {type="mcq" q="You set an SLO: p99 latency must stay below 300ms. Monitoring shows p99 = 340ms. What does this mean?"}
- [ ] 99% of requests are slower than 300ms
- [ ] Your average latency is 340ms
- [ ] 0.01% of requests are slower than 300ms
- [x] 1% of requests are exceeding 300ms — a real SLO breach affecting users at scale
:::

:::key-takeaway
Averages hide tail latency by design. Always monitor and alert on p95 or p99 — whichever your SLO uses — never the mean. The tail is where real user pain lives, and it is invisible until you measure it directly.
:::
