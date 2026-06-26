---
id: regression-tests
track: 05-evaluation
title: "Model regression tests: golden sets & per-slice gates"
badge: HOT
minutes: 9
prereqs: []
tags: [evaluation, regression-testing, golden-set, slicing, mlops, ci-cd]
xp: 60
hot2026: true
---

Picture this: it's a Tuesday afternoon in Pune. Your team ships a new version of the product-ranking model — cleaner training data, better regularisation, shiny new F1 score. Everyone's happy. Three days later, Customer Success raises a ticket: "Our female shoppers are seeing near-zero personalisation." You dig in. For that slice, your model's NDCG dropped 12 points. The aggregate metric looked fine because that subgroup is 18% of traffic. Nobody caught it before it went live.

This lesson is about the system that catches exactly that kind of failure before it reaches users — **model regression tests**.

## What is a model regression test?

Software engineers have had regression tests for decades: before you ship code, you run a suite of checks that verify nothing previously-working has broken. ML regression tests do the same thing for models.

There are two core ingredients:

**Golden set** — A fixed, frozen, hand-curated dataset you never train on and never modify (except to add new hard cases). Every time you retrain or fine-tune, you evaluate the candidate model on this set. If any metric falls below a threshold compared to the current production model, the deployment is blocked.

**Per-slice gates** — You don't just check the aggregate metric. You check it *separately* for every important subgroup (slice): by language, by product category, by user cohort, by rare entity type, by edge-case query class. A gate fails if *any* slice regresses beyond an allowed margin.

Together, they form the **eval gate** in your ML CI/CD pipeline.

:::why-prod
Aggregate metrics hide failures on minority groups and edge cases — which are often the exact cases that cause viral complaints, fairness incidents, or SLA breaches. Regression gates make silent degradation impossible to ship.
:::

## Building the golden set

Your golden set needs to be:

- **Curated, not sampled at random.** Random samples over-represent the easy majority. You want hard cases, rare categories, boundary conditions, and examples that previous models historically struggled with.
- **Labelled to production quality.** Use the same annotation guidelines as your training labels. Stale or mismatched labels defeat the purpose.
- **Versioned and append-only.** Treat it like a database migration: you can add rows, but never delete or relabel existing ones without a team review. Keep it in git-LFS or object storage with immutable object versioning.
- **Small enough to run fast.** 500–5,000 examples is typical. If your golden set takes 30 minutes to evaluate, people start skipping it.

:::gotcha
Never let the golden set "drift" into training. If examples from your golden set end up in the next training batch — through careless data joins — you've lost your clean signal. Keep a hash registry of golden-set IDs and filter them out explicitly at dataset creation time.
:::

## Defining slices that matter

A slice is any meaningful partition of your data. Common ones in Indian production systems:

- **Language/script** — Hindi, Marathi, English, Hinglish code-mix each behave very differently.
- **Cold-start vs. warm users** — New users have no history; they're fragile.
- **Low-frequency entities** — Rare brands, niche SKUs, long-tail queries.
- **Demographic proxies** — Age bucket, location tier (Tier-1 vs Tier-2/3 cities), device type.
- **Model-defined clusters** — Use an unsupervised method to find natural clusters, then name them post-hoc.

For each slice, you set a **threshold**: the maximum allowed drop in a key metric compared to the production baseline. Example: aggregate NDCG@10 may allow a 1% drop; the "new users" slice may allow only 0.5%; the "Tier-3 mobile" slice may be a strict zero-regression gate.

:::table {title="Example per-slice gate config"}
| Slice | Metric | Max allowed regression | Severity |
|---|---|---|---|
| Overall | NDCG@10 | -1.0% | Blocking |
| Hindi queries | NDCG@10 | -0.5% | Blocking |
| New users (cold) | Precision@5 | -0.5% | Blocking |
| Tier-3 mobile | Recall@10 | 0% | Blocking |
| Long-tail SKUs | NDCG@10 | -2.0% | Warning |
:::

## Wiring it into CI/CD

The eval gate lives as a step in your model training pipeline — after training, before promotion to `staging` or `prod`. Here's a minimal example you can adapt:

```python {title="eval_gate.py — slice-aware regression check" run=false}
# Run: python eval_gate.py --candidate models/v42 --baseline models/v41
# Needs: pandas, your own model.predict() wrapper, golden_set.parquet

import json
import sys
import pandas as pd

SLICES = {
    "overall":    lambda df: df,
    "hindi":      lambda df: df[df["lang"] == "hi"],
    "cold_users": lambda df: df[df["user_history_len"] == 0],
    "tier3":      lambda df: df[df["city_tier"] == 3],
}

# Max allowed drop per slice (negative = candidate can be worse by this much)
GATES = {
    "overall":    -0.010,
    "hindi":      -0.005,
    "cold_users": -0.005,
    "tier3":       0.000,
}

def ndcg_at_10(df, model):
    """Stub — swap in your real metric function."""
    preds = model.predict(df["query"].tolist(), df["items"].tolist())
    return compute_ndcg(df["labels"].tolist(), preds, k=10)

def run_gate(candidate, baseline, golden: pd.DataFrame):
    failures = []
    for slice_name, slicer in SLICES.items():
        subset = slicer(golden)
        candidate_score = ndcg_at_10(subset, candidate)
        baseline_score  = ndcg_at_10(subset, baseline)
        delta = candidate_score - baseline_score

        status = "PASS" if delta >= GATES[slice_name] else "FAIL"
        print(f"[{status}] {slice_name}: baseline={baseline_score:.4f}  "
              f"candidate={candidate_score:.4f}  delta={delta:+.4f}  "
              f"gate={GATES[slice_name]:+.4f}")
        if status == "FAIL":
            failures.append(slice_name)

    if failures:
        print(f"\nBlocking deployment — failed slices: {failures}")
        sys.exit(1)   # non-zero exit stops the CI pipeline
    else:
        print("\nAll gates passed. Safe to promote.")

if __name__ == "__main__":
    import argparse, importlib
    p = argparse.ArgumentParser()
    p.add_argument("--candidate"); p.add_argument("--baseline")
    args = p.parse_args()
    golden = pd.read_parquet("golden_set.parquet")
    # load your model wrappers here
    run_gate(load_model(args.candidate), load_model(args.baseline), golden)
```

The script exits with code `1` on any failure. Your CI system (GitHub Actions, GitLab CI, Airflow, whatever you use) treats that as a pipeline failure and blocks the model from being promoted.

:::war-story {title="The silent Hindi regression that cost a sprint"}
A mid-size Bangalore e-commerce team shipped a re-ranked recommendation model. Aggregate NDCG was up 0.8%. What they missed: the new training batch had very few Hindi-labelled examples, so the model collapsed on Hindi queries — NDCG dropped 19 points for that slice. They had no per-slice gates. The fix was discovered not by any automated check, but by a customer support rep who noticed Hindi-speaking users were getting English-only recommendations. Rolling back, re-labelling Hindi data, and re-running the full training cycle cost the team nearly two weeks. After that incident, the team added a 0% regression gate on every language slice. The next three model versions all tripped it during development — catching real issues each time — and none of them went to production broken.
:::

:::interview-line
"We gate every model release on per-slice NDCG across language, cohort, and entity-frequency slices — if any slice regresses beyond our threshold on the golden set, the pipeline blocks deployment automatically."
:::

:::qa {q="What's the difference between a golden set and the test set you use during development?"}
A test set is used iteratively during model development and hyperparameter tuning, which means it can leak information and become stale. A golden set is frozen, never touched during training or tuning, curated for hard and edge-case coverage, and used only as a deployment gate. The discipline of keeping it truly held-out is what makes it trustworthy as a regression signal.
:::

:::qa {q="How do you decide what slices to add a gate on?"}
Start with slices tied to business risk and fairness obligations — language groups, new vs. returning users, high-value cohorts. Then add slices where previous incidents happened. Finally, look at your error analysis: if a model version ever surprised you by failing a particular sub-population, that sub-population becomes a permanent slice. You're building institutional memory into the pipeline.
:::

:::qa {q="Our golden set is getting stale — months-old examples no longer reflect real queries. What do you do?"}
You extend the golden set by adding new hard cases from recent production traffic — particularly examples the current model got wrong or that represent distribution shift. You never remove or relabel old examples without a formal review (because removing them would defeat the regression protection). Treat it like append-only migrations: the history stays, the coverage grows.
:::

:::drill {type="mcq" q="A new model version improves aggregate NDCG by 1.2% on the golden set but drops NDCG by 3% for cold-start users. Your gate threshold for cold-start is -0.5%. What happens?"}
- [ ] The model is promoted because the aggregate metric improved.
- [x] The pipeline blocks deployment because the cold-start slice failed its gate.
- [ ] The result is logged as a warning and the model is promoted with a flag.
- [ ] You retrain with more cold-start data and re-run without blocking.
:::

:::drill {type="mcq" q="Which of the following would MOST undermine the integrity of your golden set?"}
- [ ] Adding 50 new hard examples from production logs every quarter.
- [ ] Storing the golden set in versioned object storage.
- [x] Including golden-set IDs in the training dataset for the next model version.
- [ ] Running the golden set evaluation on both candidate and baseline models.
:::

:::key-takeaway
Model regression tests — a frozen golden set plus per-slice metric gates wired into CI/CD — are the difference between "we think the new model is better" and "we know it didn't break anything we care about." Aggregate metrics alone will let silent failures slip through every time.
:::
