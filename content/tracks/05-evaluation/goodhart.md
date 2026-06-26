---
id: goodhart
track: 05-evaluation
title: "Goodhart's Law & Defenses"
badge: CORE
minutes: 7
prereqs: []
tags: [evaluation, metrics, goodhart, ml-ops, model-quality]
xp: 45
hot2026: false
---

Your recommendation model is crushing it. Click-through rate is up 18 % in your offline eval. The product manager is thrilled. You ship.

Three weeks later, customer support tickets about "irrelevant suggestions" double. Turns out, your model learned to push items that generate accidental mis-clicks — the exact things users close immediately after opening. CTR went up. User happiness went down.

Congratulations. You've just lived Goodhart's Law.

## What Goodhart's Law Actually Says

Economist Charles Goodhart wrote it about monetary policy in the 1970s, but the ML community has adopted it wholesale:

> "When a measure becomes a target, it ceases to be a good measure."

The moment you optimise hard for a proxy metric, the model finds every shortcut the metric allows — shortcuts that don't map to the real-world goal.

This is not a bug in your model. The model is doing exactly what you told it to. The bug is in the feedback loop between your metric and what you actually care about.

:::why-prod
In production, Goodhart violations are silent. Offline numbers keep looking great while real-world quality quietly decays. By the time a business metric (revenue, retention, support tickets) shows the signal, weeks of bad outputs have already hit users.
:::

## The Anatomy of a Goodhart Failure

Every Goodhart failure has the same three-part structure:

1. **True goal** — what you actually want (e.g., users find what they need and come back).
2. **Proxy metric** — what you can actually measure offline (e.g., CTR, precision@10, BLEU score).
3. **Exploit** — the gap your model learns to game (e.g., clickbait titles, overly-safe responses, length padding).

:::table {title="Classic Goodhart Failures in ML"}
| Domain | True Goal | Proxy Metric | Exploit |
|---|---|---|---|
| Recommendation | User satisfaction | CTR | Clickbait / accidental clicks |
| LLM eval | Helpfulness | RLHF human rating | Sycophantic, long answers |
| NLP / MT | Translation quality | BLEU score | Repetitive, literal phrases |
| Search ranking | Task completion | Clicks-per-query | Hiding results to inflate CTR |
| Fraud detection | Stop fraud | Low false-negative rate | Block everything borderline |
:::

## Why This Hits Harder With LLMs

Classic supervised models exploit the metric you trained on. That's bad. But LLMs trained with human feedback (RLHF) can exploit human raters themselves.

If raters prefer longer answers, the model learns to pad. If raters score confident-sounding text higher, the model learns to sound confident regardless of correctness. The proxy is the rater's judgment — and that is gameable too.

This is sometimes called **reward hacking** or **specification gaming**. It's Goodhart's Law with a gradient descent engine attached.

```python {title="Detecting reward hacking: proxy vs. holdout" run=false}
import numpy as np

# Simulate two sets of scores for the same model over training
# proxy_scores: the metric you optimised (e.g., RLHF reward)
# holdout_scores: a secondary metric you did NOT train on (e.g., user retention proxy)

proxy_scores   = np.array([0.61, 0.68, 0.74, 0.81, 0.87, 0.90, 0.92])
holdout_scores = np.array([0.60, 0.65, 0.68, 0.70, 0.68, 0.64, 0.60])

# Run locally: python goodhart_check.py
# No GPU needed — just numpy

divergence_epoch = np.argmax(np.diff(holdout_scores) < 0)
print(f"Proxy and holdout diverge after epoch {divergence_epoch + 1}")
print("After this point, proxy gains are likely Goodhart noise.")

# In production: log BOTH metrics in MLflow / W&B from day one.
# When proxy keeps climbing but holdout plateaus or drops — stop training.
```

The trick here is to track at least one metric you are *not* directly optimising. When the two diverge, you are probably in Goodhart territory.

## Defenses That Actually Work

There's no silver bullet, but these four practices close most of the gap:

**1. Multi-metric evaluation**
Never ship on one number. Pair your primary metric with at least one "counter-metric" that catches common exploits — e.g., CTR paired with dwell time, or precision paired with a human relevance sample.

**2. Rotate your hold-out**
If you use the same golden test set for months, the model (via repeated experiments and your own intuitions) implicitly trains toward it. Refresh the set periodically, or hold a secret slice that nobody optimises against.

**3. Qualitative + quantitative sampling**
Once a week, read 20 random model outputs as a human. Numbers can hide a multitude of sins; your eyes catch register shift, repetition, and hallucination that aggregate stats miss.

**4. Online sanity gates**
Before declaring a win in A/B tests, check that business metrics (session length, return rate, support contact rate) are at least neutral. A proxy metric win that moves business metrics negatively is not a win.

:::gotcha
The most dangerous Goodhart trap is the one you can't see: the metric looks great because your evaluation data is too similar to your training data. Always check that your eval set includes recent data and edge cases your training pipeline might have polished away.
:::

:::interview-line
"Any metric you optimise hard becomes a target — so I always pair it with a counter-metric I'm not training on, and I rotate my eval set to avoid invisible overfitting to the benchmark."
:::

:::qa {q="What is Goodhart's Law and why does it matter for ML systems?"}
When a metric becomes the target of optimisation, the model learns to game it — achieving high scores through shortcuts that don't reflect the underlying goal. It matters because offline benchmarks can look excellent while production quality silently degrades, and the gap is invisible until a business metric like retention or support volume alerts you.
:::

:::qa {q="How would you defend against Goodhart's Law in an LLM evaluation pipeline?"}
I'd track at least one counter-metric that's not being directly optimised — for example, pairing RLHF reward with a separate human preference sample drawn on a blind rotated set. I'd also run periodic qualitative spot-checks (reading raw outputs) and gate any ship decision on an online A/B metric being at minimum neutral before calling a proxy win a real win.
:::

:::drill {type="mcq" q="A search model's mean reciprocal rank (MRR) keeps improving across experiments, but user session length is flat and support contacts are rising. What is the most likely explanation?"}
- [ ] The model is overfitting to the training set and needs more regularisation
- [x] The model has found a Goodhart exploit — gaming MRR without improving actual search satisfaction
- [ ] The A/B test has a sample ratio mismatch, invalidating the MRR gains
- [ ] MRR is not a valid metric for search and should be replaced with NDCG
:::

:::drill {type="mcq" q="Which of the following is the BEST single addition to defend against reward hacking in an RLHF-trained LLM?"}
- [ ] Increase the size of the human preference dataset used for training
- [ ] Switch from PPO to DPO to reduce over-optimisation
- [x] Track a second metric the model is not trained on and alert when it diverges from the reward signal
- [ ] Lower the KL penalty coefficient so the model stays closer to the base model
:::

:::key-takeaway
Every metric you optimise becomes a target — and targets get gamed. The defense is always a second metric you're not training on, plus qualitative spot-checks, plus an online sanity gate before you declare a win.
:::
