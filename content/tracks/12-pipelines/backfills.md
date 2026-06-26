---
id: backfills
track: 12-pipelines
title: "Backfills & point-in-time correctness"
badge: HOT
minutes: 9
prereqs: []
tags: [backfills, point-in-time, temporal-leakage, feature-engineering, data-pipelines]
xp: 60
hot2026: true
---

Your churn model scores 0.91 AUC in offline evaluation. You push it to production. Two weeks later the
data science lead pings you: "Why is the model basically random on live traffic?" You dig in and find
the culprit — the backfill that built your training set used account balances from *today*, not from
the day each label was assigned. Your model secretly learned the future. This bug has a name.

## What is a backfill — and what can go wrong

A **backfill** is when you recompute a feature (or a whole feature table) for historical time periods
— usually because you're training a new model, fixing a pipeline bug, or onboarding a new data source.

Sounds straightforward. The trap is **temporal data leakage**: you accidentally use information that
wasn't available at the time the training label was assigned.

Example: you're predicting loan default. The label is "did the user default within 90 days of
applying?" You compute the feature "user's 30-day transaction count." If your backfill query says
`WHERE event_date <= label_date` using today's snapshot of the transactions table, you're fine — as
long as no transaction ever arrives late. In practice, many do.

A transaction that *happened* on Jan 8 might not appear in your warehouse until Jan 12 because of
payment-processor delays, batch ingestion lag, or late CDC events. If the label was assigned on
Jan 11, that transaction did not exist in your feature world yet. But your backfill includes it.
Training sees a count of 4; production will see a count of 3. Silent, nasty mismatch.

:::why-prod
Point-in-time bugs are impossible to catch with unit tests and hard to catch in backtests because
both use the same corrupted training data. They only surface in production where the feature
pipeline runs in real time — by which point the model is already making wrong decisions.
:::

## The two clocks you must track

Every event has (at least) two timestamps:

- **event_time** — when the thing actually happened (user clicked, payment settled).
- **processing_time** (also called *knowledge time* or *ingestion time*) — when your system
  learned about it.

A point-in-time correct backfill must ask: *"Given a label generated at time T, which events had
processing_time <= T?"* Not event_time — processing_time.

:::table {title="event_time vs processing_time — spot the difference"}
| What it is | Column name (common) | Use for |
|---|---|---|
| When the event actually happened | `event_time`, `occurred_at` | Business logic, windowing |
| When your pipeline first saw it | `processing_time`, `ingested_at` | Point-in-time correctness |
| When the row was written to the warehouse | `updated_at`, `_dbt_updated_at` | Incremental loads |
:::

The critical rule: **filter on processing_time when backfilling features for a label at time T.**

## Writing a point-in-time correct backfill

```python {title="PIT-correct backfill — transactions feature" run=false}
# Run locally: pip install pandas  (no warehouse needed for the concept)
import pandas as pd

# Simulated transactions table — note the late-arriving row
transactions = pd.DataFrame({
    "user_id":         [1, 1, 1, 1],
    "amount":          [100, 200, 50, 300],
    "event_time":      pd.to_datetime(["2024-01-01", "2024-01-05",
                                       "2024-01-08", "2024-01-10"]),
    # Jan 8 transaction only reached the warehouse on Jan 12 (late CDC)
    "processing_time": pd.to_datetime(["2024-01-01", "2024-01-05",
                                       "2024-01-12", "2024-01-10"]),
})

label_time = pd.Timestamp("2024-01-11")  # when we assigned the churn label

# ❌ Wrong: filter on event_time — looks fine, leaks the Jan-8 txn
wrong = transactions[transactions["event_time"] <= label_time]["amount"].sum()
print(f"Leaky feature sum: {wrong}")   # 650 — includes late data we couldn't have known

# ✅ Correct: filter on processing_time — only data that existed at label_time
correct = transactions[transactions["processing_time"] <= label_time]["amount"].sum()
print(f"PIT-correct feature sum: {correct}")  # 350 — Jan-8 txn not yet processed

# In a real SQL backfill (BigQuery / Snowflake):
# SELECT user_id, SUM(amount) AS txn_sum_30d
# FROM transactions
# WHERE processing_time <= :label_time          -- <-- this is the key line
#   AND processing_time >= :label_time - INTERVAL 30 DAY
# GROUP BY user_id
```

In feature stores (Feast, Tecton, Hopsworks), the equivalent concept is the **as-of join**:
you pass a DataFrame of entity keys + timestamps, and the store returns each feature's value
*as it was known at that timestamp*. This is exactly what `get_historical_features()` does.
If you're rolling your own in SQL, you replicate it manually — but the logic is identical.

:::gotcha
Many teams store only `event_time` and drop `processing_time` to save space. When you later
need to backfill PIT-correctly, you have no way to reconstruct the knowledge lag. Enforce
`processing_time` or `ingested_at` as a **non-nullable column** from day one — it costs almost
nothing to write and saves months of debugging later.
:::

:::war-story {title="The fraud model that loved the future"}
A payments startup in Pune had a fraud detection model with near-perfect offline precision.
In production, false-negative rates were 3x higher than expected. The root cause: their
backfill for the "previous 7-day failed txn count" feature used a weekly ETL snapshot —
the same snapshot that applied retroactive status corrections. A transaction flagged as
failed on Monday was sometimes corrected to "settled" by Friday. The backfill saw the
corrected value; the real-time pipeline saw the raw Monday value. Six weeks of retraining
on corrupted features, one very uncomfortable all-hands, and a strict `processing_time`
policy later, the model worked.
:::

:::interview-line
"Backfills are only correct if you filter on processing_time, not event_time — otherwise you
leak information that didn't exist at label creation, and your offline AUC becomes a lie."
:::

:::qa {q="What does point-in-time correctness mean in the context of feature backfills?"}
It means every feature value in your training set must reflect only the information that was
observable at the moment the training label was assigned. You enforce this by joining on
processing_time (when the data was known to your system), not event_time (when it happened).
Violating this leaks future data into training and inflates offline metrics.
:::

:::qa {q="How does a feature store handle point-in-time correctness for you?"}
Feature stores expose a historical retrieval API (e.g., Feast's `get_historical_features`)
that accepts an entity DataFrame with a `event_timestamp` column. Internally it performs
an as-of join: for each row it returns the latest feature value whose effective timestamp
is <= the requested timestamp. This replaces the manual `processing_time <= label_time`
filter you'd write yourself in SQL.
:::

:::drill {type="mcq" q="Your transactions table has both event_time and processing_time columns. A label is assigned at 2024-03-15. Which filter gives a point-in-time correct 30-day transaction count?"}
- [ ] WHERE event_time <= '2024-03-15'
- [x] WHERE processing_time <= '2024-03-15'
- [ ] WHERE event_time >= '2024-03-15' - INTERVAL 30 DAY
- [ ] WHERE updated_at = '2024-03-15'
:::

:::drill {type="mcq" q="You drop processing_time to save warehouse storage. Six months later you need a PIT-correct backfill. What is the best you can do?"}
- [ ] Use event_time — it's effectively the same thing for most use cases
- [ ] Re-ingest the source system's raw logs to recover arrival times
- [x] Accept that a fully PIT-correct backfill is impossible; document the leakage risk and retrain when fresh data accumulates
- [ ] Use the warehouse's updated_at column — it always matches processing_time
:::

:::key-takeaway
A backfill is only as trustworthy as the timestamp it filters on. Always write
`processing_time` to your feature tables, and join on it — not `event_time` — when
reconstructing what your system knew at any moment in the past.
:::
