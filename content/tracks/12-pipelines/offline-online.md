---
id: offline-online
track: 12-pipelines
title: "Offline vs online data — the same value, two clocks"
badge: CORE
minutes: 8
prereqs: []
tags: [feature-engineering, data-pipelines, training-serving-skew, ml-infrastructure]
xp: 45
hot2026: false
---

You shipped a model. Offline metrics looked clean — AUC 0.84, solid. You deploy it on Friday. By Monday, the product manager pings you: "conversion rate fell 12%." You dig in. The model is getting called fine. The predictions look reasonable. But something is wrong.

Three hours later you find it: the model was trained on a **batch of yesterday's data**. In production, it's receiving **today's real-time signal** — same feature name, completely different computation path, computed at different times. The `purchase_count_last_30d` the model trained on and the `purchase_count_last_30d` it is being served are not the same number. They haven't been the same number in weeks.

Welcome to the offline/online split — the most underestimated infrastructure problem in ML.

## Two worlds, one label

Every ML system lives in two time zones.

**Offline** is the past. It's your data warehouse, your S3 lake, your BigQuery tables. Data arrives in batches — hourly, daily, sometimes weekly. You use it for training, for evaluation, for bulk analysis. It's cheap and abundant. It's also *stale by design*.

**Online** is now. It's your Redis cache, your feature server, your Kafka consumer. Data arrives in milliseconds. You use it for inference — the model needs a fresh feature *right now* to score this request. It's expensive and fast. It's also *ephemeral by design*.

The same logical feature — say, `user_order_count_last_7d` — exists in both worlds. In offline, you compute it over a historical snapshot. In online, you compute it from a live event stream or a cached materialized value. Two clocks, one name.

:::why-prod
Training-serving skew is responsible for a large fraction of silent production regressions. The model didn't change. The world did. The offline and online pipelines drifted apart — different filters, different time zones, different NULL handling — and the model is now pattern-matching on a distribution it never trained on.
:::

## The anatomy of the split

:::table {title="Offline vs Online — side by side"}
| Dimension | Offline | Online |
|---|---|---|
| Latency | Minutes to hours | Sub-10ms |
| Freshness | Stale (last batch run) | Near-real-time |
| Storage | Data warehouse / lake | Key-value store (Redis, DynamoDB) |
| Compute | Spark, dbt, BigQuery | Streaming (Flink, Kafka Streams) or precomputed cache |
| Primary use | Training, evaluation, backfill | Inference / serving |
| Cost per query | Very low | Higher (fast storage + compute) |
| Scale | Petabytes, fine | Gigabytes, must stay lean |
:::

## Why two clocks drift apart

Here's the trap: you wrote your offline feature once, in SQL. You wrote your online feature once, in Python. They look equivalent. Six months later, someone fixes a timezone bug in the SQL. Nobody updates the Python. The clocks diverge.

```python {title="Same feature, two implementations — spot the drift" run=false}
# --- OFFLINE: dbt SQL model (runs nightly in BigQuery / Postgres) ---
# SELECT
#   user_id,
#   COUNT(*) AS order_count_7d
# FROM orders
# WHERE order_ts >= CURRENT_TIMESTAMP - INTERVAL '7 days'
#   AND status = 'COMPLETED'        -- <-- filter added in dbt v2

# --- ONLINE: Python feature server (never updated after initial launch) ---
def get_order_count_7d(user_id: str, redis_client) -> int:
    # Reads a precomputed value pushed by a Flink job.
    # Problem: the Flink job filters status IN ('COMPLETED', 'PENDING')
    # The dbt model filters status = 'COMPLETED' only.
    # These are NOT the same number anymore.
    raw = redis_client.get(f"order_count_7d:{user_id}")
    return int(raw) if raw else 0

# The model trained on dbt output.
# It is served on Redis output.
# They diverged three months ago. Nobody noticed.
# Run locally: pip install redis fakeredis; use fakeredis.FakeRedis() to test.
```

This is the core problem. Two separate code paths that *should* compute the same value. Every time one changes and the other does not, skew accumulates.

:::gotcha
The most dangerous form of this is *silent* skew. The model still returns predictions. The feature server does not error. The skew only surfaces as slow decay in business metrics — and by the time anyone investigates, the drift has been compounding for weeks. Fix: log feature values at serving time and compare their distribution against the training set. A simple check on mean, p95, and null rate for each feature catches drift in minutes.
:::

## The synchronization contract

Solving this is not glamorous. It's discipline.

The goal is to ensure offline and online pipelines share:

1. **The same logic** — ideally the same code, or generated from the same spec
2. **The same filters** — NULLs, status codes, timezone handling
3. **The same window boundaries** — "last 7 days" means the same thing in both places
4. **The same staleness expectation** — the model card should document "this feature is fresh to ±1 hour"

The cleanest solution is a **feature store** — one definition, materialized to both offline (for training) and online (for serving). But that's the next lesson's territory. Here the takeaway is simpler: *know that the split exists, name it explicitly, and treat synchronization as a first-class contract — not an afterthought.*

:::war-story {title="The ±5.5 hours that cost a sprint"}
A team training a churn model used `days_since_last_login` as a key feature. Their offline dbt pipeline computed timestamps in UTC. Their online feature server computed them in IST — because the backend was written in Pune and defaulted to local time. The difference was 5 hours 30 minutes, small enough to pass spot checks. But it was large enough to shift most morning logins from `days_since_last_login=1` into `days_since_last_login=0`, systematically suppressing churn predictions for an entire cohort. One sprint to diagnose. One `pytz.utc` to fix.
:::

:::interview-line
"Offline and online aren't just different storage tiers — they're different time contracts. I treat synchronization between them as an explicit SLA, not a lucky coincidence."
:::

:::qa {q="What is training-serving skew and what causes it?"}
Training-serving skew is when the feature distribution at inference time differs from what the model saw during training. It's caused by offline and online pipelines computing the same feature with different logic, filters, or time windows — even when the column name looks identical. It's dangerous because the model returns predictions without errors; the only signal is slow decay in business metrics.
:::

:::qa {q="How would you detect that offline and online features have drifted apart?"}
Log the actual feature values being served at inference time alongside the request ID. Periodically compare their statistical distribution — mean, p95, null rate — against the distribution from the most recent offline training batch. A sudden shift in mean or a spike in null rate is a reliable early signal. Tools like Evidently or a simple KL-divergence check on sampled data can automate this check in a monitoring job.
:::

:::drill {type="mcq" q="Your model's AUC in offline eval is 0.87. After deployment, business metrics drop but the model returns predictions without errors. The most likely first thing to check is:"}
- [ ] The model binary was corrupted during deployment
- [ ] The serving infrastructure has too high latency
- [x] The features at inference time differ from what the model trained on (training-serving skew)
- [ ] The evaluation dataset had too few samples
:::

:::drill {type="mcq" q="Which statement about the offline/online split is TRUE?"}
- [ ] A feature computed in SQL for training is automatically consistent with its online counterpart
- [ ] Online storage is always cheaper per query than offline storage
- [x] The same logical feature can have different computed values offline and online if the two pipelines diverge
- [ ] Online pipelines are only needed for real-time models, not batch-inference models
:::

:::key-takeaway
Offline and online are two separate computation paths for the same logical value — and they will drift unless you actively maintain the contract between them. Name the split, document the staleness tolerance, and monitor feature distributions at serving time. These three habits prevent most silent production regressions before they reach your on-call queue.
:::
