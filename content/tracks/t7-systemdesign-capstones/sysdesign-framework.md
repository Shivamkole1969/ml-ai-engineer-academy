---
id: sysdesign-framework
track: t7-systemdesign-capstones
title: "The answer framework (use every time)"
badge: HOT
minutes: 9
prereqs: []
tags: [system-design, interview, framework, ml-design, architecture]
xp: 60
hot2026: true
---

Imagine it's a Friday afternoon and you're 10 minutes into a system design interview. The interviewer says: "Design a real-time fraud detection system." Your heart rate ticks up. You know ML. You know distributed systems. You even built something like this at your last job. But nothing comes out, because you have no map — no place to start that feels safe.

That's exactly the problem this lesson fixes. A repeatable six-step framework you can reach for on autopilot, no matter what system the interviewer throws at you. Once it's in your muscle memory, blank moments stop happening.

## The Six Steps — Run Them Every Time

Think of this as a flight checklist. Pilots who skip steps crash. Engineers who skip steps confuse interviewers and confuse themselves.

Here's the framework: **Clarify → Scope → Design → ML Details → Scale → Failures & Trade-offs**.

You can remember it as **C-S-D-M-S-F** or just think of it as "calm, steady, disciplined minds solve fast."

### Step 1 — Clarify the Problem (2 min)

Before you draw a single box, ask questions. Not timid questions — confident, purposeful ones.

- Who are the users? Internal analysts or end customers?
- What's the input? Clicks, text, images, sensor data?
- Is this batch or real-time?
- Is there an existing system or are we starting from scratch?

You're not stalling. You're demonstrating production instinct. Senior engineers always clarify before they code.

### Step 2 — Scope It and Define Success (2 min)

Pin down the constraints and the finish line.

- Volume: how many requests per second at peak?
- Latency: is 50 ms acceptable or do we need 5 ms?
- Accuracy vs. speed: where does the business want the dial?
- Success metric: precision, recall, revenue lift, click-through rate?

Write these on the whiteboard or doc. They become the north star for every decision that follows.

### Step 3 — High-Level Architecture (4 min)

Sketch five to eight boxes. Don't worry about perfection yet. Think in layers:

- **Data layer**: where does data come from, how does it land?
- **ML pipeline**: training, evaluation, model registry
- **Serving layer**: how does the model get called?
- **Monitoring layer**: is the system healthy right now?

This sketch is your "before the details" map. The interviewer wants to see that you think holistically before diving in.

### Step 4 — ML Specifics (3 min)

Now zoom into the model itself:

- Feature engineering — what signals matter most?
- Model family — why a gradient boosted tree vs. a neural net here?
- Training cadence — retrain daily? Weekly? Online?
- Evaluation — offline test set? A/B test? Shadow mode?

This is where you show ML depth. Match your choice to the constraints from Step 2. Low latency? Maybe a simpler model wins. High accuracy on complex text? Transformers make sense.

### Step 5 — Scale and Cost (2 min)

Revisit the numbers from Step 2 and show how your design handles them:

- Caching strategies (model output cache, feature cache)
- Autoscaling policies (GPU vs. CPU inference)
- Data sharding if the dataset is large
- Rough cost estimate — this signals business awareness

### Step 6 — Failures and Trade-offs (2 min)

Every good design has failure modes. Name them before the interviewer does:

- What happens if the model is unavailable? (fallback rule engine? cached predictions?)
- What if data quality degrades? (data validation gate, alerting)
- What did you sacrifice for this design? What would you do differently with twice the budget?

Showing trade-off awareness is the difference between a mid-level and a senior answer.

:::why-prod
In production, skipping any of these steps is how teams end up with models that work in notebooks but fail under load, or accurate models that nobody can afford to run. The framework is a forcing function for completeness — use it in your daily design docs, not just interviews.
:::

:::table {title="Framework at a glance"}
| Step | What you're doing | Time (interview) |
|---|---|---|
| 1. Clarify | Ask scoping questions | 2 min |
| 2. Scope & SLAs | Define load, latency, success metric | 2 min |
| 3. Architecture | Sketch high-level boxes | 4 min |
| 4. ML Details | Features, model choice, training loop | 3 min |
| 5. Scale & Cost | Caching, autoscaling, rough $ | 2 min |
| 6. Failures & Trade-offs | What breaks, what you'd change | 2 min |
:::

```python {title="Design doc template — fill this in before you build" run=false}
# Use this as a living doc when designing real systems, not just interviews.
# Paste into Notion, Confluence, or a plain .md file.

from dataclasses import dataclass, field
from typing import List

@dataclass
class SystemDesignDoc:
    problem: str = ""                     # Step 1 — what are we solving?
    users: str = ""                       # who uses this system?
    input_output: str = ""               # what goes in, what comes out?

    # Step 2 — scope
    peak_qps: int = 0                    # requests/second at peak load
    p99_latency_ms: int = 100            # latency target in milliseconds
    success_metric: str = ""             # e.g. "F1 > 0.92 on held-out set"

    # Step 3 — architecture components
    components: List[str] = field(default_factory=list)
    # e.g. ["feature store", "model server", "monitoring dashboard"]

    # Step 4 — ML details
    model_family: str = ""               # e.g. "LightGBM", "BERT-base fine-tuned"
    features: List[str] = field(default_factory=list)
    retrain_cadence: str = ""            # e.g. "daily batch retrain"
    eval_strategy: str = ""             # e.g. "A/B test, 5% traffic"

    # Step 5 — scale
    caching_strategy: str = ""
    estimated_monthly_cost_usd: float = 0.0

    # Step 6 — failures
    failure_modes: List[str] = field(default_factory=list)
    fallback: str = ""                   # e.g. "rule-based fallback if model P99 > 200ms"
    trade_offs: str = ""                 # what you gave up for this design

# --- example: fraud detection system ---
fraud_doc = SystemDesignDoc(
    problem="Real-time fraud detection for payment transactions",
    users="Payment gateway; decisions must be invisible to end customers",
    input_output="Transaction event -> fraud probability [0,1] + decision (allow/block/review)",
    peak_qps=5000,
    p99_latency_ms=30,
    success_metric="Recall > 0.95 at precision > 0.80 on holdout (monthly refresh)",
    components=["Kafka ingest", "feature store (Redis)", "LightGBM model server",
                "shadow model (Transformer)", "Prometheus + Grafana"],
    model_family="LightGBM — low latency, explainable, handles sparse features well",
    features=["velocity_1h", "amount_zscore", "merchant_risk_tier", "device_fingerprint"],
    retrain_cadence="Weekly batch; online feature updates every 15 min",
    eval_strategy="Shadow mode first, then 2% A/B, then full rollout",
    caching_strategy="Feature cache in Redis (TTL 10 min); no output cache (fraud is stateful)",
    estimated_monthly_cost_usd=4200.0,
    failure_modes=["model pod crash", "feature store latency spike", "concept drift after promo"],
    fallback="Static rule engine blocks transactions > $5000 from new devices if model is down",
    trade_offs="Chose LightGBM over deep model: +8ms accuracy gain not worth 80ms latency cost"
)
```

:::gotcha
Most candidates spend 80% of the time on architecture and skip Step 6 entirely. Interviewers specifically wait for you to talk about trade-offs and failures — if you never get there, they mark you down for "doesn't think about production reality." Budget your time: move on even if the architecture isn't perfect.
:::

:::war-story {title="The 40-minute architecture that answered the wrong question"}
A strong candidate was asked to design a content moderation system for a social platform. They spent 40 minutes drawing a beautiful three-tier neural network pipeline with custom embedding models. It was technically impressive. The only problem: the interviewer had clarified in minute two that the system needed to process 500K posts per day in batch mode with a 4-hour SLA. The candidate, who never wrote down the constraints, kept optimizing for real-time sub-second latency that nobody asked for. The hire was declined — not for lack of knowledge, but for lack of a framework that would have forced them to re-read their own notes in Step 5.
:::

:::interview-line
"Before I start drawing, let me make sure I understand the constraints — because every architectural decision I make will flow from your latency and scale targets."
:::

:::qa {q="Why ask clarifying questions at the start of a design interview?"}
Clarifying questions surface constraints that change the entire design — batch vs. real-time, 100 QPS vs. 100K QPS, cost-sensitive vs. accuracy-first. A design without constraints isn't a design, it's a sketch. Senior engineers ask first because they've learned that building the wrong thing perfectly is still failure.
:::

:::qa {q="How do you handle trade-offs in a system design answer?"}
Name what you chose, name what you sacrificed, and name the condition under which you'd make the opposite choice. For example: "I chose LightGBM over a transformer because our P99 latency target is 30ms. If accuracy mattered more than latency, I'd move to a two-stage system — fast model first, slower model for edge cases." That structure shows that your decision was deliberate, not accidental.
:::

:::qa {q="What's the biggest mistake engineers make in ML system design interviews?"}
Jumping straight to model choice — talking about BERT vs. GPT-4 before establishing what the data looks like, what latency is required, or what success means. Model selection is Step 4 in the framework, not Step 1. Starting there signals junior thinking, because in production, the model is rarely the hard part.
:::

:::drill {type="mcq" q="You're designing a real-time recommendation engine. The interviewer says 'aim for 200ms end-to-end latency and about 50K requests per second at peak.' Where should this information live in your framework?"}
- [ ] Step 4 — it influences which model family you pick
- [x] Step 2 — these are your SLAs that constrain every later decision
- [ ] Step 6 — latency and load are failure-mode concerns
- [ ] Step 3 — you should add a box on the architecture diagram first
:::

:::drill {type="mcq" q="A candidate skips clarification and immediately proposes a transformer-based model for a fraud detection problem. What is the most likely consequence?"}
- [ ] They save time and impress the interviewer with confidence
- [ ] They get credit for knowing the latest architecture
- [x] They risk designing for the wrong constraints and miss trade-off discussion
- [ ] The interviewer will steer them back automatically with no penalty
:::

:::key-takeaway
Run the same six steps every time — Clarify, Scope, Design, ML Details, Scale, Failures — and budget your minutes deliberately. The framework is not a crutch; it's the signal that you think like someone who has shipped real systems.
:::
