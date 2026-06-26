---
id: eval-funnel
track: 05-evaluation
title: "Offline vs online & the evaluation funnel"
badge: CORE
minutes: 9
prereqs: []
tags: [evaluation, offline, online, production-ml, deployment, shadow-mode, canary]
xp: 45
hot2026: false
---

You spent three weeks training a recommendation model. Offline AUC: 0.89 — your personal best. You deploy on a Friday afternoon. Monday morning, your PM pings you on Slack: "Watch-time dropped 4% since Friday. What happened?"

Your model was great. On the dataset. In the past. For a world that no longer exists.

Welcome to the gap between offline and online evaluation — and why the **evaluation funnel** exists to save you from exactly this pain.

## What Is the Evaluation Funnel?

Think of evaluation as a narrowing funnel with three layers:

1. **Offline** — you test on held-out historical data before any user sees the model
2. **Online proxy** — the model serves real traffic; you watch short-term signals like click-through rate, latency, and error rate
3. **Business impact** — the metric that actually matters: revenue, retention, watch-time, DAU

Every model enters at the top. Most don't make it out the bottom. And that's fine — the funnel is supposed to kill bad ideas cheaply, before they hurt real users.

The tragedy isn't failing in the funnel. It's skipping stages.

## Offline Evaluation: Fast, Cheap, Imperfect

Offline evaluation means running your model against a static, labeled dataset it never trained on. You get a number — AUC, RMSE, F1 — instantly, reproducibly, at zero user risk.

That's the upside.

The downside: **your dataset is a frozen snapshot of the past.** It doesn't know that:
- Users changed their behavior after the data was collected
- Your feature pipeline has a subtle bug only triggered by live traffic patterns
- The model's own predictions change what users click on, which reshapes future data (feedback loops)

Offline evaluation tells you "this model is better than the baseline on last month's logs." It cannot tell you "this model will improve next month's revenue."

:::why-prod
You need offline evaluation because shipping every candidate model to production is reckless and slow. But you need online evaluation because offline scores regularly lie about production impact — sometimes by double-digit percentages. Both exist because neither alone is enough.
:::

:::table {title="Offline vs Online Evaluation at a Glance"}
| Dimension | Offline | Online |
|---|---|---|
| Speed | Minutes to hours | Days to weeks |
| Cost | Cheap (compute only) | Real users + engineering effort |
| Risk to users | Zero | Non-zero |
| Reflects live distribution | No | Yes |
| Catches feedback loops | No | Yes |
| Best used for | Fast iteration and filtering | Go / no-go shipping decisions |
:::

## Online Evaluation: The Moment of Truth

Online evaluation means your model is serving real requests. There are three common modes, each with a different risk/signal tradeoff:

**Shadow mode** — the new model runs silently in parallel with the old one. Users see only the old model's output. You log both outputs and compare score distributions. Zero risk to users, but no behavioral signal (users can't react to outputs they never see).

**Canary / staged rollout** — the new model serves a small slice of traffic — 1%, 5%, whatever you're comfortable with. If metrics look stable after a few hours, you expand gradually. If they tank, you roll back before most users notice. This is the most common production pattern.

**Full A/B test** — the gold standard for decisions. You randomly split users into control (old model) and treatment (new model), run until you hit statistical significance, then decide. The specific traps here — peeking, novelty effects, interference — are a whole lesson on their own.

The critical insight: online evaluation captures **counterfactual user behavior**. Your model doesn't just observe users — it influences them. A recommendation model changes what users watch, which changes their preferences, which changes future clicks. No static offline dataset captures this loop.

## The Gap Between Offline and Online

The offline-online gap is real, common, and sometimes brutal. The main culprits:

- **Distribution shift** — live traffic doesn't look like your training logs. New users, seasonal patterns, competitors launching features.
- **Feature skew** — features you computed offline differ from features served in production. A timestamp bug, a join that returns different rows in real-time vs batch — classic.
- **Position bias** — users click top results more, regardless of quality. Your labels reflect this, your model learns it, and then misbehaves when serving positions the labels never covered.
- **Cold-start** — new users and new items don't appear in historical data. Your model has never seen them, but production is full of them.

When offline and online disagree, trust online. Always.

:::gotcha
The trap: shipping a model the moment offline metrics beat the baseline. A 2% AUC lift offline can mean nothing online — or even a regression. Always define your online success criterion *before* you run the experiment. Deciding what "success" looks like after you see the numbers is p-hacking with extra steps.
:::

## Running the Funnel: Gate by Gate

Treat the funnel as a series of explicit gates, not a formality:

**Gate 1 — Offline**: Is this model statistically better than the baseline on held-out data? If not, iterate. If yes, move on. Do not linger here.

**Gate 2 — Shadow or canary**: Does production traffic surface any surprises? Latency spikes, null predictions, weird score distributions at tails? Fix before full rollout.

**Gate 3 — A/B test**: Does the model move the business metric you actually care about, with statistical confidence? This is the only gate that counts for shipping decisions.

```python {title="Offline gate — run locally with sklearn" run=false}
from sklearn.metrics import roc_auc_score

def offline_gate(
    y_true,
    y_score,
    baseline_auc: float,
    min_lift: float = 0.005,
) -> bool:
    """
    Returns True only if candidate beats baseline by at least min_lift.
    Fail loudly — never silently pass a weak model downstream.
    """
    candidate_auc = roc_auc_score(y_true, y_score)
    lift = candidate_auc - baseline_auc
    print(f"Candidate AUC : {candidate_auc:.4f}")
    print(f"Baseline AUC  : {baseline_auc:.4f}")
    print(f"Lift          : {lift:+.4f}")

    if lift < min_lift:
        raise ValueError(
            f"Offline gate FAILED — lift {lift:.4f} < required {min_lift}. "
            "Do not promote this model."
        )
    print("Offline gate PASSED. Promote to shadow / canary.")
    return True
```

The funnel forces humility. It reminds you that your job isn't to maximize an offline number — it's to improve something real for someone real.

:::interview-line
"We gate on offline metrics for speed, but we only ship on online metrics — because offline can't see how the model changes user behavior in real time."
:::

:::qa {q="What is the evaluation funnel and why does it matter?"}
The evaluation funnel is a staged process: offline testing on historical data, then online testing on live traffic, then validating business impact. It matters because each stage catches different failure modes that the previous stage misses. Offline is fast and safe for filtering bad ideas; online is truthful about real-world impact. Skipping stages is how models that look great on paper quietly destroy production metrics.
:::

:::qa {q="Why might a model with higher offline AUC perform worse online?"}
Offline AUC measures ranking quality on a historical snapshot under fixed conditions. Online, the model faces distribution shift (live traffic differs from logs), feature skew (production features computed differently than batch), position bias baked into historical labels, and feedback loops where the model's own outputs reshape future user behavior. None of these are visible in a static holdout set — which is why offline-online disagreement is the rule, not the exception.
:::

:::drill {type="mcq" q="Your new model beats the baseline by 3% AUC on the holdout set. What should you do next?"}
- [ ] Deploy it immediately — a 3% AUC lift is significant enough to ship
- [ ] Re-train with more data to widen the gap before touching production
- [x] Move to shadow mode or a canary rollout to validate on live traffic
- [ ] Skip straight to a full A/B test to save time
:::

:::drill {type="mcq" q="Which failure mode does shadow mode NOT help you catch?"}
- [ ] Feature skew between your offline pipeline and the production serving path
- [ ] Latency regressions under real traffic load and concurrency
- [x] Whether users actually prefer the new model's outputs over the old model's
- [ ] Null or out-of-range predictions on edge-case inputs in live data
:::

:::key-takeaway
Offline metrics tell you a model is worth trying; online metrics tell you whether to ship it. Build the funnel, run every gate in order, and never mistake a great AUC for a great product.
:::
