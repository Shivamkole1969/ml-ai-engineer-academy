---
id: decision-driven-eval
track: 05-evaluation
title: "Decision-driven eval (work backwards from the decision)"
badge: HOT
minutes: 8
prereqs: []
tags: [evaluation, product-ml, metrics, decision-making, threshold, cost-sensitive, production]
xp: 60
hot2026: true
---

It's quarter-end at a Pune fintech. The ML team ships a churn prediction model — AUC 0.87, accuracy 82%, test loss trending nicely. The retention team runs a coupon campaign off the scores. Three months later: churn went *up*. The head of growth is furious. Leadership wants a post-mortem.

What went wrong? The model was technically solid. The eval was the problem. Nobody asked the one question that should come first:

**"What decision will this model's output drive?"**

That question — asked before you write a single line of eval code — is this entire lesson.

## Work backwards from the decision

Every ML model sits *upstream* of a human or automated decision. The churn model doesn't stop customers from leaving. A person (or an automation rule) reads the score and decides: *send a ₹200 coupon*, *assign a success manager*, *do nothing*. The model is decision-support infrastructure. Your eval should test how well it supports that specific decision — not how pretty the ROC curve looks.

The framework has four steps.

**Step 1 — Name the downstream action.**
Be concrete. Not "predict churn" — but "trigger a coupon workflow for users above score threshold T". If the action is ambiguous, the eval will be too.

**Step 2 — Name the two error types and put real costs on them.**
Every binary decision has two failure modes. Write them out:

- *False positive:* model says "churn risk", but the user was fine → you spent ₹200 on a coupon for someone loyal. Cost: ₹200 per user.
- *False negative:* model says "safe", but user churns → you lost ₹1,500 monthly recurring revenue (MRR). Cost: ₹1,500 per user.

The ratio here is **1:7.5**. False negatives hurt 7.5× more. That asymmetry is now your eval spec.

:::why-prod
Production models don't live in AUC-land. They live inside business workflows where one error type often costs an order of magnitude more than the other. A metric that ignores this asymmetry will mislead you every time you ship.
:::

**Step 3 — Pick the metric that reflects the cost ratio.**
Once you have costs, metric choice becomes almost mechanical.

:::table {title="Cost asymmetry → metric selection"}
| Cost ratio (FN:FP) | Natural metric | When to use |
|---|---|---|
| ~1:1 | F1, balanced accuracy | Rare — costs truly symmetric |
| 3:1 to 10:1 | F-beta (β > 1) or weighted cost | Churn, lead scoring, fraud |
| > 10:1 | Recall-first, then precision gate | Medical triage, safety systems |
| FP >> FN | Precision-first | Ops teams flooded by alerts |
:::

For our churn case: use **F-beta with β ≈ 2.7** (roughly √7.5) or compute expected cost directly and minimize it. An AUC of 0.87 tells you *nothing* about which setting of these metrics the model achieves at the operating threshold.

**Step 4 — Set the threshold from the decision, not from the ROC curve.**
This is where most teams go wrong. They train a model, plot the ROC curve, pick the threshold at Youden's J (maximizes TPR − FPR) or default to 0.5 — and never revisit it. The correct threshold minimises *expected cost*, not a geometric property of the curve.

```python {title="Decision-threshold optimisation" run=false}
import numpy as np

# cost_fn = cost of a false negative (missed churn)
# cost_fp = cost of a false positive (wasted coupon)
cost_fn = 1500   # ₹ MRR lost
cost_fp = 200    # ₹ coupon + ops

# y_true, y_prob from your holdout set
# (swap in your own arrays)
def best_threshold(y_true, y_prob, cost_fn, cost_fp):
    thresholds = np.linspace(0.01, 0.99, 200)
    best_cost, best_t = float("inf"), 0.5
    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        fn = ((y_hat == 0) & (y_true == 1)).sum()
        fp = ((y_hat == 1) & (y_true == 0)).sum()
        expected_cost = fn * cost_fn + fp * cost_fp
        if expected_cost < best_cost:
            best_cost, best_t = expected_cost, t
    return best_t, best_cost

# Example usage:
# t_star, cost = best_threshold(y_true, y_prob, cost_fn, cost_fp)
# print(f"Optimal threshold: {t_star:.2f}  |  Expected cost: ₹{cost:,.0f}")
```

Run this on your validation set, lock the threshold, then re-check it whenever the business cost assumptions change — because they will.

:::gotcha
Picking the threshold on the *test* set is data leakage for thresholds. Always fit the threshold on the validation set. Reserve the test set as a final sanity check only — look at it once, then move on.
:::

:::war-story {title="The coupon blast that churned loyal users"}
A growth team set the churn model's threshold at 0.5 (the training-time default). At that cutoff, precision was 0.61 — meaning 39% of coupon recipients were loyal users who would never have left. Sending unsolicited "win-back" discounts to loyal users trained them to wait for discount codes before renewing — artificially *inducing* churn behaviour over time. Expected-cost optimisation would have pushed the threshold to 0.72, cutting false positives by 55% and saving ₹14 lakh over the quarter.
:::

:::interview-line
"Before I write any eval code, I ask: what decision does this output drive, what does each error type cost, and what threshold minimises expected cost — not just maximises AUC."
:::

:::qa {q="How do you choose a classification threshold in production?"}
You set it by minimising expected cost on the validation set, not by picking 0.5 or the Youden-J point on the ROC curve. Write down the cost of a false positive and a false negative in the same units (money, time, SLA minutes), then sweep thresholds and pick the one with lowest total expected cost. Revisit whenever the business context changes.
:::

:::qa {q="Your model's AUC improved from 0.84 to 0.89. Should you ship it?"}
Not necessarily. AUC measures ranking quality across all thresholds, but in production you operate at exactly one threshold. If the new model's expected cost at the business-optimal threshold is lower, ship it. If the improvement only shows up at thresholds that don't match your decision economics, the AUC gain is misleading. Always compare models at the operating threshold, not across the whole curve.
:::

:::drill {type="mcq" q="Your fraud detection model has a false-negative cost (missed fraud) of ₹5,000 and a false-positive cost (blocked good transaction) of ₹50. Which approach is most appropriate for setting the threshold?"}
- [ ] Default to 0.5 because that is the standard practice
- [ ] Pick the threshold that maximises F1 score
- [x] Sweep thresholds and minimise (FN × 5000 + FP × 50) on the validation set
- [ ] Pick the Youden-J point on the ROC curve
:::

:::drill {type="mcq" q="A colleague says 'our model hits 94% accuracy on the test set, so it's production-ready.' What is the most important follow-up question?"}
- [ ] "What is the learning rate you used?"
- [ ] "Did you shuffle the dataset before splitting?"
- [x] "What does a false negative cost versus a false positive, and what does accuracy look like at the decision threshold?"
- [ ] "Is the test set large enough for statistical significance?"
:::

:::key-takeaway
Name the downstream decision first, cost the two error types in real units, then optimise the threshold to minimise expected cost — not to maximise a metric that ignores your business reality.
:::
