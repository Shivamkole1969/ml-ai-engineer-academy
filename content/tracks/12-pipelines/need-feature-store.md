---
id: need-feature-store
track: 12-pipelines
title: "When you actually need a feature store (and when a SQL job is plenty)"
badge: CORE
minutes: 8
prereqs: []
tags: [feature-store, mlops, pipelines, data-engineering, training-serving-skew]
xp: 45
hot2026: false
---

Your fraud model has a 0.91 AUC offline. You deploy it. A week later, fraud slips through and nobody can explain why. The data scientist digs in and finds it: training used a 30-day rolling average of transaction amount, computed once a night in Redshift. Serving pulls the same feature from a live API — but with a 24-hour window. Same column name. Two completely different numbers. The model learned on one distribution and predicted on another.

That gap has a name: **training-serving skew**. And it is the single biggest reason feature stores exist.

But your weekly churn model almost certainly does not need one.

## The only question that matters

At the exact moment your model makes a prediction — is the feature already sitting in a table you can query, or does it need to be fresh right now?

If a SQL table computed last night is fresh enough, you probably don't need a feature store. If you need a feature that reflects what happened five minutes ago, you are about to feel pain a SQL job cannot fix.

That's it. Everything else is a consequence of that answer.

:::why-prod
Training-serving skew silently destroys model performance after deployment — no code changes, no errors, just worse predictions. A feature store enforces one canonical definition used identically in training and serving, eliminating the gap by construction.
:::

## SQL job: when it's the right answer

A typical setup that works beautifully: dbt transforms raw events into feature tables in BigQuery or Redshift. An Airflow DAG runs the job every night. Your model reads from that table, generates predictions, writes scores back. Done.

This handles:
- Monthly churn scoring for a SaaS product
- Weekly recommendation refreshes
- Daily credit-risk rankings
- Any batch inference where hours-old data is acceptable

The SQL file is your feature pipeline. It lives in git. Your data team can read it. It costs almost nothing to run.

```python {title="Batch pipeline — no feature store needed" run=false}
import pandas as pd
import sqlalchemy

# Run nightly via Airflow or cron
engine = sqlalchemy.create_engine("postgresql://user:pass@host/db")

features_df = pd.read_sql("""
    SELECT
        user_id,
        COUNT(*)           AS txn_count_30d,
        AVG(amount)        AS avg_txn_amount_30d,
        SUM(amount)        AS total_spend_30d,
        MAX(created_at)    AS last_txn_at
    FROM transactions
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY user_id
""", engine)

# Straight into batch inference
scores = model.predict(features_df[FEATURE_COLS])
```

No Redis. No Kafka. No registry server. Just SQL and a scheduler.

## When the SQL job stops being enough

Three specific pain points actually justify the complexity:

**Pain 1 — Real-time serving.** A fraud model that must respond in 200ms cannot wait for a SQL batch job. It needs pre-computed features in a low-latency key-value store (Redis, DynamoDB). A feature store manages the write path into that online store and keeps it in sync with training data.

**Pain 2 — Training-serving parity at scale.** You want one function — `get_user_features(user_id, as_of)` — that returns identical values in training and in production. Without a feature store, you write two separate pipelines and promise yourself they will stay in sync. They will not. The feature store enforces the contract in both directions.

**Pain 3 — Cross-team reuse.** The fraud team and the recommendations team both compute "user spend in the last 7 days." They write it independently. The numbers diverge by 3% because of timezone handling. Now two dashboards disagree and nobody knows which one is right. A feature store enforces one canonical definition in a shared registry.

:::table {title="SQL job vs. feature store: the gut-check"}
| Signal | SQL job is fine | Consider a feature store |
|---|---|---|
| Prediction latency needed | Hours / daily batch | Sub-second, real-time |
| Feature freshness required | Yesterday is fine | Minutes or less |
| Models sharing features | 1–2 | 5+ across teams |
| Team size | Solo / small pod | Multiple ML teams |
| Training-serving skew | Not a problem yet | Already hurting metrics |
| Compliance / lineage | Minimal | Strict audit trail |
:::

Most ML engineers build 3–5 batch models before they hit any of these walls. The feature store is not the default starting point — it is the answer to a specific set of problems you will recognise when you feel them.

:::gotcha
The most expensive mistake: adopting a feature store "for future scale" when you have one model, one team, and no real-time serving requirement. The operational overhead — Redis, a registry server, backfill pipelines, SDK integration — can absorb months of engineering time. Add it when you feel the pain it solves, not before.
:::

:::interview-line
"We moved to a feature store only when training-serving skew was visibly hurting production metrics and three teams were computing the same rolling window differently — before that, dbt plus Redshift was exactly the right tool."
:::

:::qa {q="How do you decide whether a project needs a feature store?"}
Ask three questions: Do I need features fresh enough that a nightly SQL job won't cut it? Are multiple models or teams sharing the same aggregations, risking divergent definitions? Is training-serving skew already a measured problem? If all three are "no," a well-maintained SQL pipeline is not a compromise — it is the correct architecture. The feature store earns its complexity only when at least one of those answers is "yes."
:::

:::qa {q="What is training-serving skew and how does a feature store fix it?"}
Training-serving skew is when the same feature is computed differently in the training pipeline versus the production serving path. The model learns from one distribution and predicts on another, causing silent performance degradation after deployment. A feature store fixes this by providing a single definition and a single compute path — the same code that materialises features for training also serves them at inference time, so the numbers are identical by construction.
:::

:::drill {type="mcq" q="A team runs a nightly batch churn model for 80k users. Features live in BigQuery. One ML engineer owns the whole pipeline. What should they build?"}
- [ ] Feast feature store backed by Redis for low-latency serving
- [ ] A Kafka consumer to keep features fresh in real time
- [x] A dbt model plus a scheduled query that writes features to a BigQuery table each night
- [ ] A custom feature server with DynamoDB as the online store
:::

:::drill {type="mcq" q="Which situation is the strongest signal that a team needs a feature store rather than a SQL batch job?"}
- [ ] The feature table has more than 50 columns
- [ ] The team recently migrated their warehouse from MySQL to Postgres
- [ ] The model's offline AUC is above 0.90
- [x] A fraud model must score transactions in under 300ms using a rolling feature that must match exactly what was used in training
:::

:::key-takeaway
A feature store solves three specific problems: real-time feature serving, training-serving parity, and shared definitions across teams. If none of those three are hurting you today, a SQL pipeline is not technical debt — it is the right tool.
:::
