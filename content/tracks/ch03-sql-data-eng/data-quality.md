---
id: data-quality
track: ch03-sql-data-eng
title: "Data quality checks that catch real bugs"
badge: CORE
minutes: 9
prereqs: []
tags: [data-quality, sql, etl, validation, production]
xp: 45
hot2026: false
---

Imagine your recommendation model has been performing beautifully for six weeks. Then one Monday morning, your manager forwards a user complaint: "Why is the app recommending items I already bought?" You dig in and find the culprit — an upstream pipeline silently started writing duplicate rows three days ago. Your model trained on dirty data and nobody noticed until users complained. This is a data quality bug. And the maddening thing is: it is almost always preventable with checks that take an afternoon to write.

## What data quality actually means

Data quality is not one thing — it is four things that can fail independently:

- **Completeness** — are all the expected rows and columns there? (Missing order IDs, NULL prices.)
- **Uniqueness** — are rows duplicated? (Two records for the same event ID.)
- **Validity** — do values fall in the expected range or format? (Age = -3, email without "@".)
- **Freshness** — is the data recent enough? (A "daily" table that hasn't updated in 36 hours.)

When any one of these breaks, your downstream models, dashboards, and pipelines inherit the corruption — quietly.

:::why-prod
Garbage-in, garbage-out is not a cliché — it is your SLA risk. A model that trains on duplicate rows develops a bias toward whatever those rows said. A freshness failure means you are serving stale predictions. Neither error throws an exception; they just silently degrade your product.
:::

## Four checks every pipeline needs

These are the bread-and-butter assertions. Run them after every load, before anything else reads the table.

:::table {title="Core data quality checks"}
| Check type | What to assert | Example SQL |
|---|---|---|
| Row count | Count > 0 (or within a band vs. yesterday) | `SELECT COUNT(*) FROM orders WHERE load_date = CURRENT_DATE` |
| Null rate | Critical columns have NULL% below threshold | `SELECT AVG(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END) FROM events` |
| Uniqueness | Primary key has zero duplicates | `SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders` |
| Freshness | Max timestamp is within expected window | `SELECT MAX(created_at) > NOW() - INTERVAL '25 hours' FROM events` |
:::

The pattern is always the same: compute a metric, compare it to a threshold, raise an alert if it fails. Simple. Boring. Life-saving.

## Writing checks that actually run

Knowing the checks is half the job. The other half is wiring them into your pipeline so they run automatically — not manually when someone remembers.

Here is a minimal Python helper you can drop into any ETL job. It runs checks as assertions and raises loudly before any downstream step can consume bad data.

```python {title="Simple data quality gate" run=false}
import pandas as pd

def run_quality_checks(df: pd.DataFrame, table_name: str) -> None:
    """
    Run core data quality checks on a DataFrame.
    Raise ValueError on the first failure so the pipeline halts.

    # To run locally: pip install pandas
    # In production, swap df for a SQL result via your warehouse connector.
    """

    # 1. Completeness — must have rows
    if len(df) == 0:
        raise ValueError(f"[{table_name}] FAILED completeness: 0 rows loaded")

    # 2. Uniqueness — no duplicate primary keys (assumes 'id' column)
    dup_count = len(df) - df["id"].nunique()
    if dup_count > 0:
        raise ValueError(f"[{table_name}] FAILED uniqueness: {dup_count} duplicate IDs")

    # 3. Null rate — critical columns must be < 5% null
    critical_cols = ["id", "user_id", "created_at"]
    for col in critical_cols:
        if col not in df.columns:
            continue
        null_pct = df[col].isna().mean()
        if null_pct > 0.05:
            raise ValueError(
                f"[{table_name}] FAILED null check: '{col}' is {null_pct:.1%} null"
            )

    # 4. Freshness — most recent record must be within 25 hours
    if "created_at" in df.columns:
        latest = pd.to_datetime(df["created_at"]).max()
        age_hours = (pd.Timestamp.utcnow() - latest.tz_localize("UTC")).seconds / 3600
        if age_hours > 25:
            raise ValueError(
                f"[{table_name}] FAILED freshness: latest record is {age_hours:.1f}h old"
            )

    print(f"[{table_name}] All quality checks passed ✓")


# Example usage
# df = pd.read_sql("SELECT * FROM orders WHERE load_date = CURRENT_DATE", conn)
# run_quality_checks(df, table_name="orders")
```

The key design decision: **fail loudly and early**. A `ValueError` that halts the job is a feature, not a bug. Silent continuation is how corruption spreads three tables deep.

:::gotcha
The most common mistake is checking row count in absolute terms: `assert count > 0`. That passes even when 90% of rows are missing. Use a relative check instead — compare today's count to a rolling 7-day average and alert if it drops more than 20%. Sudden volume drops are one of the most common signals of upstream breakage.
:::

## Where these checks live in the stack

You have three places to put quality checks, and they are not mutually exclusive:

1. **In the pipeline code** — Python/Spark assertions that stop execution on failure (as above).
2. **As SQL tests in dbt** — dbt's built-in `not_null`, `unique`, `accepted_values` tests run against your warehouse on every model build.
3. **In a dedicated quality layer** — tools like Great Expectations or Soda run a full suite of expectations and report pass/fail without stopping your pipeline, so you can monitor trends over time.

For most teams, a combination of (1) for critical gates and (2) for warehouse-layer validation covers 90% of real-world bugs.

:::interview-line
"I put quality checks at the ingestion boundary — row count, null rate, uniqueness, and freshness — so bad data never reaches downstream models."
:::

:::qa {q="How would you detect that a pipeline is loading duplicate rows?"}
Compare `COUNT(*)` to `COUNT(DISTINCT primary_key)` after each load. If the difference is greater than zero, you have duplicates. In production I wrap this as an assertion that raises an alert before downstream jobs run, so nothing trains on dirty data.
:::

:::qa {q="Why is a freshness check important for ML pipelines specifically?"}
ML models often assume that the features they receive at inference time have the same distribution as training data. If the feature pipeline goes stale — say, a daily job silently stops running — you end up serving predictions based on day-old or week-old signals. That can cause subtle score drift that is very hard to debug without a freshness gate at the data layer.
:::

:::qa {q="What is the difference between a data quality check and a schema check?"}
A schema check (covered in the Schema Drift lesson) validates structure: column names, types, required fields. A data quality check validates content: are values reasonable, complete, unique, and fresh? Both matter, but they catch different classes of bugs. Schema changes tend to throw errors immediately; quality degradation is often silent.
:::

:::drill {type="mcq" q="Your daily orders table normally loads 50,000 rows. Today it loaded 200. Which check would catch this?"}
- [ ] A NOT NULL check on order_id
- [ ] A uniqueness check on order_id
- [x] A row count anomaly check comparing today's load to the rolling average
- [ ] A freshness check on created_at
:::

:::drill {type="mcq" q="You want to ensure no order appears twice in your training dataset. Which SQL expression correctly counts duplicates?"}
- [ ] `SELECT COUNT(DISTINCT order_id) FROM orders`
- [ ] `SELECT COUNT(*) FROM orders WHERE order_id IS NOT NULL`
- [x] `SELECT COUNT(*) - COUNT(DISTINCT order_id) FROM orders`
- [ ] `SELECT order_id FROM orders GROUP BY order_id HAVING COUNT(*) > 1 LIMIT 1`
:::

:::key-takeaway
Four checks — completeness, uniqueness, validity, freshness — placed at the ingestion boundary will catch the vast majority of real-world data bugs before they corrupt your models or dashboards. Fail loudly and early; silent continuation is how corruption spreads.
:::
