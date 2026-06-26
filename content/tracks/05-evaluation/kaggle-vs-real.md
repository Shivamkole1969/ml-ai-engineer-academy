---
id: kaggle-vs-real
track: 05-evaluation
title: "Kaggle Metrics vs Real-World Metrics"
badge: CORE
minutes: 8
prereqs: []
tags: [evaluation, metrics, production, kaggle, business-metrics, auc, precision]
xp: 45
hot2026: false
---

You just placed top 5% on a Kaggle fraud-detection competition. AUC of 0.97. You put it on your resume, you get the job at a Pune fintech, and on day 30 your manager pulls you into a meeting. The fraud operations team is drowning. Your model is firing 10,000 alerts a day. Nine thousand seven hundred of them are innocent customers. The call centre is furious. Your 0.97 AUC model is about to get switched off.

This is not a hypothetical. This is Tuesday.

## Why the Kaggle number felt real but wasn't

On Kaggle, you are optimising a single number on a fixed, clean snapshot of data. The leaderboard rewards whoever gets the highest score — period. Nobody asks what happens when you *act on* the score. Nobody asks what one wrong prediction costs.

In production, a metric is only useful if it tracks something a human actually cares about. AUC tells you "how well can this model *rank* fraud above legitimate transactions?" That is a useful property. But the operations team does not rank transactions — they *investigate* alerts. For them, precision (what fraction of alerts are real fraud?) is the number that pays the bills.

The gap between the competition metric and the operational metric is the gap between a model that wins a leaderboard and a model that ships.

:::why-prod
A model optimised purely on AUC can have terrible precision at any operationally relevant threshold. Ops teams burn out, legitimate customers get blocked, and the product dies — even though the model technically had a great score. The metric you optimise *becomes* the thing you build toward, so you have to pick the right one.
:::

## The common mismatches — a quick map

:::table {title="Kaggle vs Production: metric translation"}
| Kaggle favourite | What it actually measures | Common production replacement |
|---|---|---|
| AUC-ROC | Ranking quality across *all* thresholds | Precision@fixed threshold (ops can act on it) |
| RMSE / MAE | Average error | P90 / P99 error (tail behaviour kills SLAs) |
| Accuracy | Fraction correct | Recall on the rare positive class (fraud, cancer, churn) |
| Log-loss | Probabilistic calibration (good!) | Same — but also check calibration curves, not just the scalar |
| nDCG / MAP | Ranking quality for recs | Click-through rate, dwell time, conversion (what the user *does*) |
:::

The pattern: Kaggle metrics measure model *properties*. Business metrics measure *outcomes*. Your job is to know the mapping between them — and be suspicious when it is not obvious.

## Code: the fraud-alert trap in 20 lines

```python {title="AUC looks great; precision hurts" run=false}
from sklearn.metrics import roc_auc_score, precision_score
import numpy as np

# 10 000 transactions, 1 % fraud — realistic for many fintechs
np.random.seed(42)
n, fraud_rate = 10_000, 0.01
y_true = np.random.binomial(1, fraud_rate, n)

# Model scores: fraudulent txns skew higher, but there is huge overlap
scores = np.where(
    y_true == 1,
    np.random.beta(4, 2, n),   # fraud: higher scores
    np.random.beta(1, 5, n),   # legit: lower scores
)

auc = roc_auc_score(y_true, scores)

# Threshold that catches 80 % of actual fraud
thresh = np.percentile(scores[y_true == 1], 20)
y_pred = (scores >= thresh).astype(int)

precision = precision_score(y_true, y_pred)
total_alerts = y_pred.sum()
false_alerts = ((y_pred == 1) & (y_true == 0)).sum()

print(f"AUC:                {auc:.3f}")          # 0.93 — looks great
print(f"Total alerts/day:   {total_alerts}")     # ~1900
print(f"False alerts/day:   {false_alerts}")     # ~1820 — investigators hate you
print(f"Precision:          {precision:.3f}")    # 0.04 — 96 % noise

# The fix: talk to ops, agree on a maximum false-alert budget,
# derive the threshold from that — not from "best F1 on test set".
```

This runs locally with `pip install scikit-learn`. Run it; change `fraud_rate` to 0.001 and watch precision crater further.

## Three questions to ask before you pick a metric

**1. Who acts on this model's output, and what does one wrong prediction cost them?**
An alert that goes to a human investigator costs their time. A price prediction error that auto-executes a trade costs real money. The cost asymmetry should shape whether you weight precision vs recall.

**2. Does the test set reflect the live distribution?**
Kaggle datasets are often pre-balanced (50/50 fraud). Real fraud rates are 0.01–0.1 %. A model tuned on balanced data will be wildly over-confident on live traffic.

**3. Is the metric you are tracking the thing that changes behaviour?**
This is Goodhart's Law territory — covered in its own lesson. For now: if the metric is easy to game without actually improving the product, it will be gamed, by the model or by people.

:::gotcha
Picking a threshold based on "the F1-maximising point on the PR curve" sounds principled, but it is an arbitrary default. In production, thresholds are business decisions — they depend on ops capacity, legal requirements, and cost-of-false-positive vs cost-of-false-negative. Never let the validation set pick your threshold for you.
:::

:::interview-line
"AUC tells me how well the model ranks; precision at my operating threshold tells me whether the ops team can survive Monday morning."
:::

:::qa {q="Your model has 0.94 AUC on the test set but the product team says it's not useful. How do you diagnose this?"}
The AUC is a threshold-agnostic ranking metric — it says nothing about how the model behaves at the specific decision point in production. I would first align on what action the model's output drives and at what threshold, then measure precision and recall there. I would also check whether the test set distribution matches live traffic, particularly the positive class rate, which often differs drastically between curated datasets and production data.
:::

:::qa {q="When would you use RMSE in production vs a percentile error like P99?"}
RMSE is dominated by large errors because it squares them, but it averages them — so a model that is usually good but occasionally catastrophic can still look fine. In systems with SLAs (latency prediction, ETA, pricing) the *tail* matters more than the mean. I use P90 or P99 error when downstream systems have hard limits, and RMSE mainly during training for gradient-friendly loss computation.
:::

:::drill {type="mcq" q="A recommendation model achieves 0.88 nDCG@10 on the offline test set. Which scenario best explains why it still fails to improve click-through rate in A/B test?"}
- [ ] The model was not tuned with learning-rate warmup.
- [x] The test set contains historical clicks, so the model learns to rank items users already saw — not items they would click on fresh recommendations.
- [ ] nDCG@10 is always a bad metric for recommendations.
- [ ] The model needed a higher AUC first.
:::

:::drill {type="mcq" q="You are building a cancer-screening model. The dataset is 95 % negative (healthy). Which metric is most dangerous to rely on alone?"}
- [ ] Recall (sensitivity)
- [ ] Precision
- [x] Accuracy
- [ ] F1 score
:::

:::key-takeaway
A metric is a proxy for a goal, not the goal itself. Before choosing what to optimise, ask what *action* the model's output drives and what the cost of each type of error is — then work backwards to a metric that reflects that, not whatever the Kaggle leaderboard defaulted to.
:::
