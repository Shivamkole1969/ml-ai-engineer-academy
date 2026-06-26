---
id: metrics-roi
track: t7-systemdesign-capstones
title: "Business & product thinking: metrics & ROI"
badge: HOT
minutes: 9
prereqs: []
tags: [metrics, roi, product-thinking, system-design, business-value]
xp: 60
hot2026: true
---

Your team ships a fraud detection model with 95% recall. The security team pops champagne. Six months later, finance sends a grim spreadsheet: false positives have been blocking 12% of high-value legitimate transactions. Customers churned quietly. The quarterly revenue report shows a $2M hole. Your model was technically great. It just optimized for the wrong objective.

This is the metrics-ROI trap — and it ends more ML careers than bad code ever does.

## The Two Worlds Every ML Engineer Must Bridge

There are two worlds in every AI project.

**World 1: ML Metrics** — accuracy, F1, AUC-ROC, BLEU, latency P99. These live in your notebooks and dashboards.

**World 2: Business Metrics** — revenue, churn rate, cost per decision, customer lifetime value (CLV), support ticket volume. These live in the boardroom.

Your job is to translate between them — fluently, before you write a single line of model code.

In a system design interview, the first question you should ask is *not* "what's the expected QPS?" It's: **"What business problem does this solve, and how will we know it's working?"**

:::why-prod
A model with great offline metrics can still destroy business value if it optimizes a proxy metric that diverges from the real goal. Production ML is about business outcomes, not leaderboard scores. Teams that can't articulate ROI get their projects cancelled — or worse, kept alive while silently doing damage.
:::

## Mapping ML Metrics to Business Outcomes

Every ML metric should have a business metric it approximates. Here's the translation table:

:::table {title="ML metric → Business metric mapping"}
| ML Metric | What It Measures | Business Proxy |
|---|---|---|
| Precision | Of flagged items, how many are correct? | Cost of false alarms (human review cost, customer friction) |
| Recall | Of real positives, how many were caught? | Cost of misses (fraud loss, missed revenue, safety risk) |
| Latency P99 | Slowest 1% of requests | Cart abandonment rate, SLA breach penalty |
| Model drift | Distribution shift over time | Silent revenue leak, compliance exposure |
| Throughput | Requests handled per second | Infrastructure cost per transaction |
:::

When you design a system, pick the ML metric that most closely matches the business cost of each error type.

**The one question that changes everything:** "Which is more expensive — a false positive or a false negative?" The answer reshapes your entire model design: threshold, loss function, evaluation strategy.

## Calculating ROI: The Formula That Gets Approvals

ROI for ML systems is not "did accuracy improve." It is:

```
ROI = (Value Generated) − (Cost to Build + Cost to Run)
```

Make that concrete. Here is a back-of-envelope ROI calculator you would sketch in an interview or a design doc:

```python {title="ML system ROI estimator" run=false}
# Rough ROI calculator for an ML system
# Run locally with: python roi_calc.py

# --- Inputs: fill these in for your use case ---
daily_decisions   = 500_000   # how many times the model runs per day
baseline_error_rate = 0.08    # error rate WITHOUT the model (human baseline)
model_error_rate    = 0.02    # error rate WITH the model
cost_per_error_usd  = 15.0    # business cost of each mistake (fraud loss, refund, etc.)

infra_cost_per_day   = 200.0  # GPU/CPU inference cost per day
team_cost_amortized  = 150.0  # eng + data science cost amortized per day

# --- Calculation ---
errors_before    = daily_decisions * baseline_error_rate
errors_after     = daily_decisions * model_error_rate
errors_prevented = errors_before - errors_after

daily_value = errors_prevented * cost_per_error_usd
daily_cost  = infra_cost_per_day + team_cost_amortized
daily_roi   = daily_value - daily_cost

build_cost  = 500_000  # total one-time build investment

print(f"Errors prevented per day:  {errors_prevented:,.0f}")
print(f"Daily value generated:     ${daily_value:,.2f}")
print(f"Daily operating cost:      ${daily_cost:,.2f}")
print(f"Daily ROI:                 ${daily_roi:,.2f}")
print(f"Payback period (days):     {(build_cost / daily_roi):.1f}")
```

This won't be exact — real systems have messier numbers. But it forces the right conversation. Interviewers and product managers love when you can reason about tradeoffs quantitatively rather than hand-waving at "better accuracy."

## What to Track in Production

Three categories of metrics, always running in parallel:

**1. Business health** — revenue per prediction, fraud loss rate, CLV of affected customers, support ticket volume.

**2. Model health** — drift score, accuracy on a labeled validation slice, prediction distribution shift week-over-week.

**3. System health** — P50/P99 latency, error rate, throughput, cost per inference.

Set alerts on all three layers. A model can be perfectly healthy technically while the business metric bleeds silently.

:::gotcha
Engineers obsess over model health dashboards but forget to wire up business metric alerts. Your dashboard shows green — precision is 0.94, latency is 18ms — but nobody notices that revenue from the affected customer cohort has dropped 7% over three weeks. Always include at least one business metric in your alerting runbook. "The model is healthy" and "the product is working" are not the same sentence.
:::

:::war-story {title="The CTR win that tanked the quarter"}
A large e-commerce team rebuilt their recommendation engine and achieved a 14% lift in click-through rate (CTR) in A/B testing. They shipped globally. Three weeks later, finance flagged a revenue-per-session drop of 9%. The new model had learned to recommend cheaper, more clickable items — great for CTR, terrible for average order value (AOV). They had optimized the proxy metric perfectly and the business metric exactly wrong. The fix was adding AOV as a second optimization objective and rerunning the A/B test with revenue as the primary success criterion. The lesson: always A/B test your *business* metric, not just your ML metric.
:::

:::interview-line
"Before I design the model, I need to understand the cost asymmetry: what's the business cost of a false positive versus a false negative? That determines the threshold, the metric, and ultimately the architecture."
:::

:::qa {q="How do you justify the cost of building an ML system to a product team?"}
Frame it as a comparison: what's the cost of the current approach — human review, rule-based logic, or doing nothing — versus the projected cost with ML, accounting for build time, infrastructure, and ongoing maintenance? Then show the break-even point in days. A system that costs $5K/month to run but prevents $80K/month in fraud losses is a straightforward approval. Quantify both sides of the equation before the meeting, even if the numbers are rough.
:::

:::qa {q="How do you decide which ML metric to optimize when there's no single right answer?"}
Ask what's more expensive — false positives or false negatives — and in what ratio. If a false negative (missed fraud) costs 20× more than a false positive (blocking a legitimate transaction), you optimize for recall and accept more false alarms. The business cost ratio maps directly to where you set your decision threshold and which loss function you use during training. When in doubt, build a confusion-cost matrix and use it as a north star.
:::

:::drill {type="mcq" q="Your fraud model has 97% recall and 60% precision. Leadership asks if you should deploy it. What's the FIRST thing you ask?"}
- [ ] What is the model's AUC-ROC on the test set?
- [ ] How much GPU capacity do we have available?
- [x] What is the business cost of a false positive versus a false negative?
- [ ] Can we improve precision with more training data first?
:::

:::drill {type="mcq" q="Which of the following is a lagging business indicator for a content recommendation system?"}
- [ ] Model prediction latency at P99
- [ ] Click-through rate in A/B test
- [ ] Feature store cache hit rate
- [x] 30-day user retention rate
:::

:::key-takeaway
Always translate your ML metric to a business metric before designing anything. Ask: "What does a false positive cost versus a false negative?" Wire up business metric alerts in production — not just model health dashboards. The engineers who get promoted are the ones who speak both languages fluently.
:::
