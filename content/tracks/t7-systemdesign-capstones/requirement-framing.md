---
id: requirement-framing
track: t7-systemdesign-capstones
title: "Requirement framing & SLAs"
badge: HOT
minutes: 9
prereqs: []
tags: [system-design, sla, requirements, latency, ml-production, interviews]
xp: 60
hot2026: true
---

Your interviewer says: "Design a fraud detection system for a payment platform." You crack your knuckles, open your mouth, and start sketching a neural network pipeline. Three boxes in, the interviewer asks: "What latency do we need?" Dead silence. You realize you've been designing an answer to the wrong question.

That moment — fumbling for numbers you never asked for — is the most common way strong engineers fail system design interviews. And it's completely avoidable.

## The Two Questions That Unlock Everything

Before you draw a single box, you need to know two things:

**What should this system DO?** (Functional requirements)
**How well must it do it?** (Non-functional requirements / SLAs)

Functional requirements are the features. Non-functional requirements — especially SLAs — are the constraints that determine your entire architecture.

For an ML system, this split looks like:

:::table {title="Functional vs. Non-Functional in ML Systems"}
| Functional | Non-Functional (SLA) |
|---|---|
| Detect fraud in real-time | Prediction latency < 100ms at p99 |
| Recommend products | 99.9% availability |
| Classify support tickets | Model freshness: retrain daily |
| Generate summaries | Throughput: 10,000 requests/second |
:::

:::why-prod
In production, nobody fires you for building the wrong feature. They fire you for building something that's too slow, goes down at 2 AM, or serves stale predictions after a distribution shift. SLAs are how the business communicates what "good enough" means — and they shape every architectural decision you make.
:::

## Decoding SLA Vocabulary

SLAs use specific language. Learn it or be lost.

**Latency percentiles: p50, p95, p99, p999**
The p99 latency is the threshold that 99% of requests fall below. If your p99 is 200ms, 1 in 100 requests is slower than that. For a fraud detection system processing 10,000 transactions/second, that's 100 slow requests every second — which might be acceptable or catastrophic depending on the business.

**Availability: usually expressed as "nines"**
- 99.9% = ~8.7 hours of downtime per year ("three nines")
- 99.99% = ~52 minutes of downtime per year ("four nines")

One extra nine costs roughly 10x in engineering effort. Know what your system actually needs before committing.

**Throughput: requests per second (RPS) or queries per second (QPS)**
Always ask peak vs. average. A system that handles 1,000 RPS on average might need to serve 5,000 RPS during a flash sale or a viral moment.

**Model-specific SLAs** (the ML layer on top of standard infra SLAs):
- **Model freshness**: How stale can predictions be? Real-time streaming? Hourly batch? Daily retrain?
- **Accuracy floor**: What's the minimum acceptable precision or recall before an alert fires?
- **Degraded-mode behavior**: What does the system serve if the ML model is unavailable?

## The Requirement Framing Checklist

In an interview — or a real design session — work through this before touching the architecture:

```python {title="Requirement framing template" run=false}
# Use this as a mental checklist or paste it into a design doc

REQUIREMENT_FRAME = {
    # Functional
    "core_features": "What are the top 3 things this system must do?",
    "out_of_scope": "What are we explicitly NOT building?",

    # Scale
    "users_or_rps": "How many users / requests per day (peak)?",
    "data_volume": "How much data do we store or process?",

    # ML-specific
    "model_freshness":     "Retrain frequency? (real-time / hourly / daily)",
    "prediction_latency":  "p99 latency budget for inference?",
    "accuracy_floor":      "What recall/precision drop triggers an incident?",
    "fallback":            "What do we serve if ML is down? (rules / cached / deny-all)",

    # Reliability
    "availability_target": "99.9% or 99.99%? (the difference is real money)",
    "consistency":         "Stale-but-available OK, or must we be strongly consistent?",

    # Constraints
    "budget_hints":  "On-prem, cloud, GPU budget?",
    "compliance":    "PII handling, data residency, audit logging?",
}

# In an interview: state these out loud and invite the interviewer to fill them in.
# Interviewers reward the habit. It signals production thinking, not tutorial thinking.
```

Run through this in the first 5 minutes of any design session. It's a conversation, not a solo performance — interviewers are actively looking to see whether you ask before you build.

## Translating Business Language to Technical SLAs

Product managers speak in outcomes. You need to translate them into numbers your architecture can target.

:::table {title="Business → Technical SLA translation"}
| Business says… | You design for… |
|---|---|
| "Users can't wait" | p99 inference latency < 200ms |
| "We can't afford downtime" | 99.99% availability, multi-region failover |
| "Recommendations must be fresh" | Model retrain every 1–4 hours |
| "Fraud loss is a top-3 KPI" | Recall ≥ 95%; alert if it drops below 90% |
| "Just make it work for now" | 99.9% availability, single-region is fine |
:::

When you hear business language, ask one follow-up question: **"What does failure look like?"** The answer almost always tells you your SLA. "Users abandon the page" → latency. "We get fined" → compliance. "Revenue drops" → recall or availability.

:::gotcha
The trap is assuming latency requirements. Many engineers default to "let's aim for sub-100ms" without asking. A batch fraud report that runs overnight has a completely different latency budget than a real-time payment gate. State your assumptions out loud and validate them — in an interview, silent assumptions are missed points.
:::

:::war-story {title="The 50ms Nobody Asked For"}
A team built a content moderation system with outstanding accuracy and an elegant pipeline. Three weeks post-launch, the mobile team discovered every image upload was stalling for 800ms — the moderation model was running synchronously on the upload path. Nobody had asked "what's the latency budget for moderation?" during the design phase. The answer turned out to be 50ms, async, fire-and-forget. The system had to be rebuilt around an async queue. Three weeks of user complaints and two engineers' weekends lost — all from one un-asked question in week one.
:::

:::interview-line
"Before I design anything, I want to nail the SLAs — because the difference between p99 100ms and p99 500ms changes my entire caching and serving strategy."
:::

:::qa {q="Why do SLAs matter more for ML systems than for traditional APIs?"}
ML systems have an extra failure mode: the model can be live and healthy but producing bad predictions — due to data drift, a stale training snapshot, or a bug in feature engineering. Traditional SLAs cover availability and latency. ML SLAs must also cover model freshness and accuracy thresholds. Without those, you can have a fully green dashboard while the system is silently losing the business money.
:::

:::qa {q="How do you decide between p99 and p999 as your latency target?"}
It depends on the cost of a slow request and your traffic volume. At 100 RPS, p999 covers roughly 0.1 slow requests per second — usually fine to ignore. At 100,000 RPS, p999 is 100 slow requests per second, which could seriously degrade user experience. Pick the percentile where a slow request has real business impact, then size your system to meet that bar. For user-facing ML inference, p99 is usually the right anchor; for financial transactions, consider p999.
:::

:::drill {type="mcq" q="Your system serves 10,000 requests/second. The p99 inference latency is 300ms but your SLA requires p99 < 200ms. Which is the correct first step?"}
- [ ] Immediately switch to a smaller model to reduce compute time
- [ ] Add more replicas behind the load balancer
- [ ] Increase model batch size to improve GPU utilization
- [x] Profile where time is actually spent — preprocessing, model inference, or postprocessing — before changing anything
:::

:::drill {type="mcq" q="A product manager says 'our recommendation model needs to stay fresh.' Which SLA best captures this requirement?"}
- [ ] p99 latency < 100ms
- [ ] Availability ≥ 99.9%
- [ ] Throughput ≥ 5,000 QPS
- [x] Model retrain frequency ≤ 4 hours, with an alert if the training pipeline falls behind
:::

:::key-takeaway
Requirement framing is the difference between designing the right system and building a beautiful answer to the wrong question. Always lock down functional scope, latency budget, availability target, and ML-specific SLAs — model freshness and accuracy floor — before drawing your first architecture box.
:::
