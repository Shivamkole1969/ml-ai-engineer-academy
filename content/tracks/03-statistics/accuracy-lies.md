---
id: accuracy-lies
track: 03-statistics
title: 'Why "95% accuracy" lies: base rates, P/R/F1, ROC-AUC vs PR-AUC'
badge: HOT
minutes: 10
prereqs: []
tags: [classification, metrics, precision, recall, f1, roc-auc, pr-auc, imbalanced-data]
xp: 60
hot2026: true
---

It is Friday evening. You have just finished training a fraud detection model for a Pune-based
fintech client. The notebook says **99.1% accuracy**. You screenshot it, paste it into Slack,
and the CTO reacts with a rocket emoji. Life is good.

Monday morning: the first production report lands. Fraudsters cleared ₹40 lakh over the weekend.
Your model flagged exactly **zero** of them.

Here is what happened. Of 10,000 daily transactions, only about 100 are fraud — that is 1%.
Your model learned one beautiful trick: predict "not fraud" for everything. It gets 99% accuracy
for free. It also lets every fraudster walk right through.

This is the accuracy trap, and it is the single most common mistake junior ML engineers make
in interviews — and in production.

## The Base Rate Problem

Accuracy measures how often you are right overall. When classes are imbalanced, "overall" is
almost entirely the majority class. The minority class — the one you actually care about —
drowns.

Think of it this way. If 99 of every 100 emails are normal, a model that says "not spam" for
every email scores 99% accuracy. But your inbox would be a disaster.

The fraction of positives in your data is called the **base rate** (or class prior). The lower
it is, the more accuracy lies to you. Fraud, disease, churn, rare defects — anything with a
low base rate will fool a careless accuracy report.

:::why-prod
In production, the cost of a false negative (missing actual fraud) is often 10× the cost of
a false positive (flagging a clean transaction). Accuracy weights them equally. You need metrics
that let you dial this trade-off explicitly.
:::

## Precision, Recall, and F1

Three numbers that actually tell the truth.

**Precision** — "when you cry wolf, how often is there actually a wolf?"

> Precision = TP / (TP + FP)

Out of everything you flagged as fraud, what fraction was real fraud? Low precision = lots of
innocent transactions blocked. Angry customers.

**Recall** (also called sensitivity or TPR) — "how many wolves did you catch?"

> Recall = TP / (TP + FN)

Out of all actual fraud cases, what fraction did you catch? Low recall = fraud slips through.
Real financial loss.

The two are in constant tension. Push your threshold down (flag more things) → recall goes up,
precision goes down. Push it up → precision rises, recall falls.

**F1 score** is the harmonic mean of the two. It punishes extreme imbalance between P and R —
so a model that games recall by flagging everything still scores poorly.

> F1 = 2 · (P · R) / (P + R)

For asymmetric priorities, use **F-beta**: F2 weighs recall twice as heavily (good for fraud),
F0.5 weighs precision twice (good for content moderation where false flags hurt trust).

:::table {title="Metric quick-reference"}
| Metric | What it asks | Low value means | Use when |
|---|---|---|---|
| Accuracy | Overall correct? | You might be fooled by imbalance | Balanced classes only |
| Precision | Flags trustworthy? | Too many false alarms | FP cost is high (content mod) |
| Recall | Catch rate? | Misses the real events | FN cost is high (fraud, disease) |
| F1 | P and R balanced? | One of P/R is tanking | Quick single-number comparison |
| ROC-AUC | Rank order quality? | Model barely beats random | General discriminative ability |
| PR-AUC | Useful at high precision? | Low-precision regime useless | Imbalanced data, rare positives |
:::

## ROC-AUC vs PR-AUC — Pick the Right One

Both measure model quality across all thresholds instead of one fixed decision point.

**ROC curve** plots True Positive Rate (recall) vs False Positive Rate at every threshold.
AUC = area under that curve. A perfect model scores 1.0; random gets 0.5.

The catch: FPR = FP / (FP + TN). When negatives are massive (99,900 legit transactions),
even 200 false positives gives FPR = 0.002. Looks tiny. ROC-AUC stays high and looks
impressive — even when your model is flagging 200 clean transactions per day.

**PR curve** plots Precision vs Recall. It completely ignores true negatives. No TN term
anywhere. That means class imbalance cannot inflate it. When positives are rare, PR-AUC
exposes the real cost: to get high recall you often crater precision.

Rule of thumb: use **PR-AUC** whenever your positive rate is below ~10%. Use ROC-AUC when
classes are roughly balanced or when you want to compare models on ranking ability in general.

```python {title="Accuracy vs real metrics on imbalanced data" run=false}
# pip install scikit-learn numpy  (free, runs locally)
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
)

rng = np.random.default_rng(42)
n = 10_000

# 1% fraud rate — realistic for many fintech datasets
y_true = np.zeros(n, dtype=int)
y_true[:100] = 1

# ---- Model A: predicts "no fraud" every time ----
y_dumb   = np.zeros(n, dtype=int)
s_dumb   = np.zeros(n)          # score = 0 for everyone

# ---- Model B: catches 70 of 100 frauds, 200 false positives ----
y_real = np.zeros(n, dtype=int)
y_real[:70]    = 1              # 70 true positives
y_real[100:300] = 1             # 200 false positives

# Realistic score: fraud scores cluster higher
s_real = rng.uniform(0.0, 0.5, n)
s_real[:100] += 0.4             # fraud scores boosted
s_real = np.clip(s_real, 0, 1)

for label, y_pred, scores in [
    ("Dumb model (always 'no fraud')", y_dumb, s_dumb),
    ("Real model (70% recall)",        y_real, s_real),
]:
    print(f"\n=== {label} ===")
    print(f"  Accuracy  {accuracy_score(y_true, y_pred):.1%}")
    print(f"  Precision {precision_score(y_true, y_pred, zero_division=0):.1%}")
    print(f"  Recall    {recall_score(y_true, y_pred):.1%}")
    print(f"  F1        {f1_score(y_true, y_pred, zero_division=0):.1%}")
    print(f"  ROC-AUC   {roc_auc_score(y_true, scores):.3f}")
    print(f"  PR-AUC    {average_precision_score(y_true, scores):.3f}")

# Expected output (approx):
# Dumb model:  Accuracy 99.0%  Precision 0%  Recall 0%  F1 0%
#              ROC-AUC 0.500   PR-AUC 0.010
# Real model:  Accuracy 97.3%  Precision 26%  Recall 70%  F1 38%
#              ROC-AUC 0.847   PR-AUC 0.340
# The dumb model LOOKS better on accuracy. Every other metric tells the truth.
```

Notice that the dumb model's accuracy (99%) is *higher* than the real model's (97.3%). Anyone
who only looks at accuracy would ship the useless model.

:::gotcha
ROC-AUC above 0.9 on imbalanced data can feel like a green light to deploy. It is not.
On a dataset that is 1% positive, a model with ROC-AUC = 0.95 can still have PR-AUC below 0.3
— meaning precision collapses before you reach meaningful recall. Always plot the PR curve
alongside ROC before signing off on a model going to production.
:::

:::war-story {title="The 98% model that blocked ₹0 fraud"}
A credit-card risk team shipped a model with 98% accuracy and a proud ROC-AUC of 0.92.
No one looked at PR-AUC. In production, to hit 80% recall the team had to drop the threshold
so low that precision fell to 4% — meaning 24 legit transactions were blocked for every one
fraudulent one. Customer service was flooded. Within a week the threshold was cranked back up,
recall dropped to 11%, and fraud losses resumed. The post-mortem finding: the team had optimised
the wrong metric from day one. A quick PR-AUC check during experimentation would have caught it.
:::

:::interview-line
"Accuracy is a lagging indicator on imbalanced problems. I default to PR-AUC during
experimentation and only lock in a threshold after aligning on the business cost of a
false negative versus a false positive."
:::

:::qa {q="A recruiter asks: 'Our fraud model has 97% accuracy — is that good?'"}
Not necessarily. If fraud is 1–2% of transactions, a model predicting 'no fraud' always would
score 98–99% accuracy. The meaningful questions are: what is the recall (catch rate on actual
fraud)? What is precision at that recall? And which metric was optimised during training?
ROC-AUC and PR-AUC tell a far more honest story on imbalanced data.
:::

:::qa {q="When would you prefer ROC-AUC over PR-AUC?"}
When classes are roughly balanced — say 40/60 or 30/70 — ROC-AUC is reliable and widely
understood. It is also the right choice when you care about the model's overall ranking ability
across the full population, not just its precision at the rare-positive end. For anything below
~10% positive rate, PR-AUC is the more honest signal because it is not inflated by the large
true-negative pool.
:::

:::drill {type="mcq" q="A dataset has 2% positive rate. Your model gets 98.3% accuracy and ROC-AUC 0.91. What should you check NEXT?"}
- [ ] Retrain with more epochs — accuracy could still improve
- [ ] Nothing; ROC-AUC 0.91 is excellent and the model is ready
- [x] Plot the PR curve and compute PR-AUC — ROC-AUC can be inflated by the large negative class
- [ ] Switch to a regression model to avoid threshold decisions
:::

:::drill {type="mcq" q="You need high recall on a rare disease screen, accepting some false positives. Which F-beta score fits best?"}
- [ ] F0.5 (precision-weighted)
- [ ] F1 (equal weight)
- [x] F2 (recall-weighted)
- [ ] Accuracy, because more samples reduces noise
:::

:::key-takeaway
On any imbalanced problem, accuracy is almost meaningless. Lead with Precision, Recall, F1,
and PR-AUC — and always align the metric you optimise to the actual business cost of missing
a real positive versus flagging a false one.
:::
