---
id: proxy-gap
track: 04-losses
title: "Misaligned losses: the proxy gap (how bad products are born)"
badge: HOT
minutes: 9
prereqs: []
tags: [loss-design, proxy-metric, objective-alignment, production-ml, goodharts-law]
xp: 60
hot2026: true
---

You've spent six weeks training a news-feed ranking model. Offline, NDCG@10 is up four points. Your manager is thrilled. You ship it on a Friday evening. Two weeks later the app-store rating drops half a star. Support tickets flood in: "It shows me outrage and doom all day." You dig into the data. The model learned that clickbait headlines get clicked, re-read, and shared — and your loss function rewarded exactly that. Great loss. Bad product.

This is the **proxy gap**: the distance between *what your loss actually measures* and *what you genuinely want the model to do*.

## What is a proxy, and why does it slip?

Your real objective — "make users happier", "reduce customer churn", "catch more fraud without annoying good users" — is either unmeasurable in real time, ambiguous, or delayed by months. So you substitute a **proxy**: CTR, NDCG, AUC, cross-entropy on a labelled set.

The proxy works fine at the start. The problem is **Goodhart's Law**: once a measure becomes a target, it ceases to be a good measure. The model finds every statistical shortcut your proxy rewards that does *not* align with your real goal, and it exploits all of them. Not maliciously. It just optimizes what you told it to.

The gap does not shrink at scale. It grows. More data gives the model more signal to exploit the proxy.

:::why-prod
A misaligned loss ships a model that passes every offline evaluation and fails every user. The model is technically correct and practically wrong. The cost is user trust — once lost, it doesn't come back with a hotfix.
:::

## Where the gap hides — common pairings

:::table {title="Proxy vs. real objective mismatches"}
| Domain | What you optimize | What you actually want | Classic failure mode |
|---|---|---|---|
| Feed ranking | CTR, watch time | Long-term user satisfaction | Outrage loops, filter bubbles |
| Search | NDCG on annotated queries | User task completion | Good-looking results that don't answer the question |
| Fraud detection | AUC on historical labels | Catching future fraud, not annoying good users | High AUC, but threshold wrong for actual cost ratio |
| Translation / summarisation | BLEU / ROUGE | Correct, natural output | Fluent nonsense that scores well |
| Content moderation | Precision/recall on labelled set | Safe platform, low false-positive rate | Great lab numbers, PR fire in production |
| Recommendation | Immediate rating or click | Repeat purchase, subscription renewal | Recommends what users say they like, not what keeps them |
:::

## Spotting the gap before it ships

Three signals that tell you your proxy has drifted from reality:

**1. Offline–online divergence.** Offline metric goes up, but A/B test shows flat or negative real-world impact. This is the most common and most expensive signal — you only find it after shipping.

**2. Saturation without improvement.** You keep training and the proxy improves, but manual spot-checks feel worse. The model found a shortcut the proxy can't see.

**3. Distribution shift on the metric itself.** Your labelled eval set was annotated six months ago. User behaviour has changed. The proxy no longer measures what it did when you built it.

```python {title="Detect proxy drift: log both metrics in production" run=false}
# Run this alongside your serving layer.
# Replace proxy_score and real_signal with your actual signals.
# Free to run locally with dummy data — no GPU needed.

import statistics

class ProxyGapMonitor:
    """
    Track correlation between your proxy metric and the real signal.
    A dropping correlation is your first warning that the gap is widening.
    """
    def __init__(self, window: int = 1000):
        self.window = window
        self.proxy_scores: list[float] = []
        self.real_signals: list[float] = []

    def log(self, proxy_score: float, real_signal: float) -> None:
        """Call this once per request/session when you have both signals."""
        self.proxy_scores.append(proxy_score)
        self.real_signals.append(real_signal)
        # Keep a rolling window
        if len(self.proxy_scores) > self.window:
            self.proxy_scores.pop(0)
            self.real_signals.pop(0)

    def pearson_r(self) -> float | None:
        """Simple Pearson r. Drop below 0.4 → investigate."""
        n = len(self.proxy_scores)
        if n < 30:
            return None  # not enough data yet
        mean_p = statistics.mean(self.proxy_scores)
        mean_r = statistics.mean(self.real_signals)
        num = sum(
            (p - mean_p) * (r - mean_r)
            for p, r in zip(self.proxy_scores, self.real_signals)
        )
        den_p = sum((p - mean_p) ** 2 for p in self.proxy_scores) ** 0.5
        den_r = sum((r - mean_r) ** 2 for r in self.real_signals) ** 0.5
        if den_p == 0 or den_r == 0:
            return None
        return num / (den_p * den_r)

    def report(self) -> dict:
        r = self.pearson_r()
        return {
            "window_size": len(self.proxy_scores),
            "proxy_mean": statistics.mean(self.proxy_scores) if self.proxy_scores else None,
            "real_signal_mean": statistics.mean(self.real_signals) if self.real_signals else None,
            "pearson_r": r,
            "status": "OK" if r and r > 0.4 else "INVESTIGATE",
        }

# --- toy demo ---
import random
random.seed(42)
monitor = ProxyGapMonitor(window=200)

for _ in range(200):
    proxy = random.gauss(0.7, 0.1)
    # Simulate gap: real signal weakly correlated with proxy
    real = 0.3 * proxy + random.gauss(0.5, 0.2)
    monitor.log(proxy, real)

print(monitor.report())
# {'window_size': 200, 'proxy_mean': ~0.70, 'real_signal_mean': ~0.71,
#  'pearson_r': ~0.28, 'status': 'INVESTIGATE'}  ← gap is real
```

## Closing the gap (or at least shrinking it)

You will never fully close it. The goal is to make it small and visible.

**Add real-signal feedback loops.** Collect delayed signals — subscription renewal, support-ticket rate, session return rate — and fold them into training as auxiliary objectives or reward labels.

**Multi-objective loss.** Weight your proxy alongside a slower but more honest signal. Even a rough proxy-of-the-proxy that is less gameable helps: session length is easier to fake than 7-day retention, but combining them is harder to game than either alone.

**Eval on held-out human judgements.** A small sample of human ratings is worth more than a million automatic proxy scores for detecting drift.

**Red-team your proxy.** Literally ask: "how could a model score perfectly on this while being useless or harmful?" If you can answer in thirty seconds, your proxy is fragile.

:::gotcha
The most dangerous proxy gap is the one that looks fine in staging and breaks in production. Staging data was collected under your *old* model. Your new model will shift the data distribution — and with it, what the proxy actually measures. Always budget for a limited live traffic test before full rollout, even if offline metrics are convincing.
:::

:::war-story {title="The watch-time trap"}
A mid-sized video platform in India optimised purely for watch time — the obvious proxy for engagement. The recommendation model discovered that autoplay sequences of progressively intense content kept users watching longest. Watch time went up 18% in the first quarter. Then advertiser complaints started: brands didn't want their ads next to the rabbit-hole content the model was amplifying. Platform reputation took months to recover. The fix wasn't a new loss function — it was adding a "content-diversity" penalty and a 30-day advertiser-brand-safety metric to the reward signal before retraining.
:::

:::interview-line
"We never optimise the real objective — we optimise a proxy. My job as an ML engineer is to keep the proxy gap visible and small, not to pretend it doesn't exist."
:::

:::qa {q="What is the proxy gap and why does it cause production failures?"}
The proxy gap is the difference between the metric you optimise (e.g. CTR, AUC, BLEU) and the real outcome you care about (e.g. user satisfaction, revenue, safety). It causes failures because models find every statistical shortcut the proxy rewards that the real objective does not — and at scale they find all of them. Offline metrics look great; users churn.
:::

:::qa {q="A colleague says 'our AUC is 0.97, the model is great'. What would you ask next?"}
I'd ask: great at what, for whom, and measured on what data? AUC of 0.97 on a six-month-old test set tells you the model memorised historical label patterns well. It says nothing about whether the threshold is calibrated for actual cost ratios, whether the label distribution matches production, or whether a high-AUC model actually reduces business harm. I'd want to see live precision-recall at the operating threshold alongside a real downstream metric like fraud-loss-per-thousand-transactions.
:::

:::qa {q="How do you detect that your proxy has drifted from the real objective?"}
Three signals: offline–online divergence (proxy goes up, A/B shows no win), saturation without perceived improvement (manual spot-checks get worse as training continues), and stale eval data (labels annotated months ago no longer reflect current user behaviour). The cheapest fix is logging both the proxy score and a delayed real signal in production and tracking their correlation over time.
:::

:::drill {type="mcq" q="A content-ranking model achieves its highest-ever NDCG@10 after a long training run, but the 7-day user-return rate drops. What most likely explains this?"}
- [ ] The model overfit to the training set and generalised poorly
- [ ] NDCG@10 was computed incorrectly
- [x] The model exploited a proxy gap — maximising short-term relevance signals at the cost of long-term engagement
- [ ] The learning rate was too high
:::

:::drill {type="mcq" q="Which of the following is the BEST early warning that a proxy gap is widening in production?"}
- [ ] Training loss is still decreasing
- [ ] Validation AUC is above 0.9
- [x] Offline metric improves but A/B test on live traffic shows no lift in the real business metric
- [ ] The model's p99 latency is within SLA
:::

:::key-takeaway
Every loss function is a proxy. The proxy gap — between what you measure and what you want — never disappears; it only hides until it ships. Make it visible: log real signals alongside proxy scores, track their correlation, and budget for live traffic tests before full rollout.
:::
