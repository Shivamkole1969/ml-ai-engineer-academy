---
id: deployment-ladder
track: 14-production-safety
title: "Deployment: shadow → canary → progressive → rollback"
badge: HOT
minutes: 10
prereqs: []
tags: [deployment, canary, shadow-mode, progressive-rollout, rollback, production, mlops]
xp: 60
hot2026: true
---

It is 11 PM. Your new ranking model crushed every offline metric — NDCG up 4%, latency flat. Your manager says "ship it." You're tempted. Then you remember what happened to the team before yours: they pushed directly to prod, session duration dropped 9% in 90 minutes, and the on-call engineer spent the rest of the night reverting a Kubernetes rollout while revenue bled. They had no shadow data. No canary. No rollback plan.

You don't make that mistake. You use the deployment ladder.

## The Four Rungs

The ladder is a controlled, staged release. Each rung answers one question: "Is this safe for more traffic?"

**Shadow mode** — your new model runs on 100% of real traffic, but its output is *logged and discarded*. Users see the old model. You see what the new model *would have* said. Zero user risk. This is your sanity check.

**Canary** — you route a tiny slice of real traffic (1–5%) to the new model. Now real users experience it, but blast radius is small. You watch your guardrail metrics hard.

**Progressive rollout** — confidence is growing. You step up the percentage: 10% → 25% → 50% → 100%. Each step is gated: metrics must stay healthy before you advance.

**Rollback** — not a failure state, a *feature*. Traffic flips back to the old model in seconds. You define the trigger (automated or manual) before you start, not during the incident.

:::why-prod
Offline metrics (AUC, NDCG, BLEU) never fully predict production behaviour. A model that looks perfect on a test set can wreck a metric it was never evaluated on — revenue, p99 latency, abuse rate. The ladder gives you a kill switch at every rung before damage scales.
:::

## Shadow Mode in Code

```python {title="shadow_router.py — log new model outputs without serving them" run=false}
import logging, hashlib

logger = logging.getLogger(__name__)


def shadow_predict(request, stable_model, shadow_model):
    """
    Always return the STABLE model's output.
    Fire the shadow model and log the diff — never let it affect the user.
    """
    # Stable path — user sees this
    stable_result = stable_model.predict(request)

    # Shadow path — wrapped so a crash here can never hurt the user
    try:
        shadow_result = shadow_model.predict(request)
        logger.info("shadow_compare", extra={
            "stable_score": stable_result.score,
            "shadow_score": shadow_result.score,
            "delta": shadow_result.score - stable_result.score,
            "request_hash": hashlib.md5(str(request).encode()).hexdigest()[:8],
        })
    except Exception as exc:                    # noqa: BLE001
        logger.warning("Shadow model error (safe to ignore): %s", exc)

    return stable_result                        # user always gets stable


def canary_bucket(user_id: str, canary_pct: float = 5.0) -> str:
    """
    Deterministic bucketing — same user always hits same model.
    Avoids flickering UX and makes A/B analysis clean.
    """
    bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
    return "canary" if bucket < canary_pct else "stable"


# Run locally:
#   pip install -q pytest
#   pytest shadow_router.py --doctest-modules
```

## Guardrail Metrics — What to Watch at Each Rung

:::table {title="Metrics to gate each deployment rung"}
| Rung | Traffic | Primary check | Gate condition |
|---|---|---|---|
| Shadow | 0% served | Prediction distribution shift | KL divergence < threshold |
| Canary | 1–5% | Business KPIs (CTR, session length, revenue/session) | No regression vs. baseline (stat-sig) |
| Progressive (25%) | 25% | Latency p95/p99, error rate | p99 < SLO, error rate < 0.1% |
| Progressive (100%) | 100% | All of the above + abuse / safety signals | Hold 30 min, then declare stable |
:::

## Canary Is Not A/B Testing

Easy to confuse. Here is the difference.

A/B testing is an *experiment* — you want to know which variant wins. You need statistical power, which means running it long enough to get clean results.

A canary is a *safety gate* — you want to catch fires early. You pull the canary and rollback if *anything* looks wrong, without waiting for significance. Speed beats rigour at this stage.

Run your A/B tests during shadow mode, when you have 100% of traffic with zero risk.

:::gotcha
Do not gate on a single metric. Teams have shipped a model that improved CTR while crushing session length. Define a *metric bundle* — business KPI + latency + error rate + one safety signal — and fail the gate if *any* one degrades beyond threshold. One green number hiding three red ones is how incidents happen.
:::

:::war-story {title="The 5% canary that caught a ₹40L/day bug"}
A Pune-based e-commerce team pushed a new recommendation model to 5% canary on a Thursday night. Everything looked fine on CTR. But one engineer had added "revenue per session" to the dashboard the week before. It dropped 18% on canary users. Turns out the model over-indexed on low-price items because of a normalisation bug in the training pipeline — looked great on clicks, terrible on basket value. They rolled back in four minutes. If they had skipped the canary, the bug would have run all Friday, a peak sales day. Estimated loss averted: ₹40 lakh.
:::

:::interview-line
"We use a shadow → canary → progressive ladder with automated metric gates. Rollback is a one-command operation we test in staging before every release."
:::

:::qa {q="Why run shadow mode at all if you're going to canary anyway?"}
Shadow mode is free risk — 100% of traffic, zero user exposure. It lets you compare prediction distributions, catch schema mismatches, and validate logging pipelines before a single real user sees the new model. Canary then starts from a much higher confidence baseline, so your 5% blast radius is not a shot in the dark.
:::

:::qa {q="How do you decide when a canary is healthy enough to promote?"}
You define the criteria before you deploy, not during. Pick a metric bundle (business KPI + latency + error rate + safety signal), set thresholds, and specify a minimum observation window (e.g. 30 minutes with 10k requests). Promotion is automatic when all gates pass; rollback is automatic if any gate fails. Removing human judgement from the hot path means faster decisions and no "it looks fine to me" incidents at 2 AM.
:::

:::drill {type="mcq" q="Your canary model shows +3% CTR but p99 latency jumped from 120ms to 340ms. What do you do?"}
- [ ] Promote — CTR improvement outweighs the latency cost
- [ ] Wait for more data before deciding
- [x] Rollback immediately — latency SLO is a hard gate, not a tradeoff
- [ ] Increase canary to 20% to get a larger sample before deciding
:::

:::drill {type="mcq" q="Which statement best describes the difference between a canary release and an A/B test?"}
- [ ] Canary tests require statistical significance before rollback; A/B tests do not
- [ ] They are the same thing, just different names used by different teams
- [x] Canary is a safety gate (roll back on any signal); A/B is an experiment (need significance to conclude)
- [ ] A/B tests use 1–5% traffic; canaries use 50%
:::

:::key-takeaway
The deployment ladder — shadow, canary, progressive, rollback — turns "push and pray" into a controlled, reversible process. Define your metric gates and rollback trigger before you deploy. Then the ladder protects you even at 2 AM.
:::
