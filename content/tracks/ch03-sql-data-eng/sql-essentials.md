---
id: sql-essentials
track: ch03-sql-data-eng
title: "SQL essentials every ML engineer needs"
badge: HOT
minutes: 9
prereqs: []
tags: [sql, data-engineering, feature-engineering, window-functions, joins]
xp: 60
hot2026: true
---

Your model has been live for three weeks and everything looks fine — until your data scientist notices that the training accuracy metric on the dashboard is suspiciously high. You dig in and find a window function that was accidentally leaking future data into the feature set. The model was trained on the future. Of course it looked great.

This is not a made-up cautionary tale. It happens on real teams. And the root cause is almost always a SQL query that nobody scrutinised carefully enough.

For an ML engineer, SQL is not "just for analysts." It is the language of your training data, your feature pipelines, your data quality checks, and your experiment logging. If your SQL is shaky, your model inputs are shaky — and no amount of hyperparameter tuning fixes bad inputs.

## The SQL toolkit an ML engineer actually uses

You do not need to memorise every obscure SQL function. You need five things done right.

**1. Aggregations with GROUP BY**

This is how you build per-entity features: average session length per user, total spend per customer, error count per model version. Get comfortable grouping, filtering groups with `HAVING`, and knowing the order of operations (`WHERE` filters rows before grouping; `HAVING` filters groups after).

**2. JOINs — and knowing when they silently explode**

`INNER JOIN`, `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`. The dangerous one for ML is the `LEFT JOIN` that secretly multiplies rows when the right table has duplicates on the join key. Your feature table quietly grows and you get duplicate training examples. More on this in the gotcha below.

**3. Window functions — your most powerful feature-engineering tool**

Window functions let you compute a value for each row *relative to other rows in the same group*, without collapsing the result. Classic use: rolling averages, rank within a group, lag/lead to get the previous event.

```sql {title="Rolling 7-day average spend per user" run=false}
-- Run this in DuckDB (free, local): duckdb my_data.db
-- or paste into BigQuery / Snowflake / Redshift / Postgres

SELECT
    user_id,
    event_date,
    spend,
    -- average spend over the past 7 days for this user
    AVG(spend) OVER (
        PARTITION BY user_id
        ORDER BY event_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS rolling_7d_avg_spend
FROM user_spend
ORDER BY user_id, event_date;
```

`PARTITION BY` is like `GROUP BY` but without collapsing rows. `ORDER BY` inside `OVER()` defines the order for the window. `ROWS BETWEEN` defines how wide the window is.

**4. CTEs for readable, testable query chains**

A Common Table Expression (`WITH cte_name AS (...)`) lets you break a complex query into named steps. Think of each CTE as a unit-testable function. Your feature pipeline should read like a story, not a 300-line nested subquery.

**5. Date/time arithmetic**

Features almost always have a time dimension. Know how to truncate dates (`DATE_TRUNC`), compute differences (`DATEDIFF` / interval subtraction), and convert timezones. The exact syntax varies by engine — always check your warehouse's docs.

:::why-prod
Every ML feature pipeline lives inside SQL. Window functions create temporal features; bad window specs leak labels from the future. JOINs that fan-out silently inflate training datasets. Bad date arithmetic shifts your train/test split by a day — enough to make a model look better than it is.
:::

:::table {title="SQL concepts vs. ML use-case"}
| SQL concept | ML use-case |
|---|---|
| GROUP BY + aggregation | Per-entity features (user stats, item counts) |
| Window function (rolling) | Temporal features without label leakage |
| LEFT JOIN | Attaching features to a label table |
| CTE chain | Readable multi-step feature pipeline |
| DATE_TRUNC / DATEDIFF | Train/test split boundaries, recency features |
| HAVING | Filter groups with too few samples |
:::

## Window functions and label leakage — the big trap

Label leakage means your model sees information during training that it would never have at inference time. A window function causes leakage when the `ORDER BY` clause inside `OVER()` uses a column that is derived from the future.

The classic mistake: you compute a "7-day future revenue" feature to understand user value, then accidentally leave it in your training features. The model learns it perfectly. Your offline metrics are stellar. Production metrics are terrible.

The fix: always define your feature timestamp explicitly and use `ROWS BETWEEN N PRECEDING AND CURRENT ROW` (not `RANGE BETWEEN UNBOUNDED FOLLOWING`).

:::gotcha
A `LEFT JOIN` on a non-unique key is the most common silent corruption in ML pipelines. If your right-hand table has multiple rows per join key, every match duplicates the left-hand row. Your training dataset balloons, certain examples are over-represented, and your model trains on ghost data. Always `SELECT COUNT(*) vs COUNT(DISTINCT key)` on your join tables before trusting a pipeline.
:::

:::war-story {title="The rolling average that trained on the future"}
A recommendation team built a "7-day rolling purchase count" feature for a ranking model. The window function used `ROWS BETWEEN CURRENT ROW AND 6 FOLLOWING` — looking forward, not backward. Offline NDCG looked 12 points higher than the previous model. The team celebrated and deployed. Online A/B test showed a slight *decrease* in conversions. After two days of head-scratching, someone visualised the feature distribution at inference time versus training time and noticed the values were impossible to compute at serving. The model had memorised future purchase counts it could never know in production. Fixing the window direction to `6 PRECEDING AND CURRENT ROW` brought the offline and online metrics back into alignment.
:::

## Writing queries your future self can debug

Use CTEs, name them well, and add one-line comments explaining the *why* not the *what*. A query that extracts training data is a first-class artifact — it deserves the same care as application code.

```python {title="Running SQL in Python with DuckDB (free & local)" run=false}
import duckdb  # pip install duckdb

# DuckDB can query Parquet, CSV, or Pandas DataFrames directly
con = duckdb.connect()

query = """
WITH base AS (
    -- one row per user per day
    SELECT user_id, event_date, SUM(spend) AS daily_spend
    FROM 'events.parquet'
    GROUP BY user_id, event_date
),
rolling AS (
    SELECT
        user_id,
        event_date,
        daily_spend,
        AVG(daily_spend) OVER (
            PARTITION BY user_id
            ORDER BY event_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg
    FROM base
)
SELECT * FROM rolling
WHERE event_date < '2024-01-01'  -- keep training set before cutoff
"""

df = con.execute(query).df()
print(df.head())
```

:::interview-line
"I treat every feature-extraction query like production code — CTEs for readability, explicit window frames to prevent label leakage, and a row-count check before and after every join."
:::

:::qa {q="What is a window function and why is it useful for feature engineering?"}
A window function computes a value for each row using a set of related rows (the "window") without collapsing the result like GROUP BY does. For feature engineering, this lets you compute rolling statistics — like a user's average spend over the past 7 days — where you keep one row per event and attach the aggregate as a new column. It is far cleaner and less error-prone than self-joins.
:::

:::qa {q="How do you prevent label leakage when using window functions in a training pipeline?"}
Always use a backward-looking window frame: `ROWS BETWEEN N PRECEDING AND CURRENT ROW`. Never use `FOLLOWING` for any feature that would not be available at inference time. Set a hard cutoff timestamp for your training set and make sure all feature windows use data strictly before that cutoff.
:::

:::qa {q="What goes wrong with a LEFT JOIN on a non-unique key in a training dataset?"}
If the right-hand table has multiple rows matching the join key, each match creates a copy of the left-hand row. The result has more rows than the original label table, certain training examples are duplicated, and your model over-fits to those examples. The fix is to ensure join keys are unique on the right side — or to explicitly deduplicate with a CTE before joining.
:::

:::drill {type="mcq" q="You want to compute each user's total spend over the 30 days *before* each transaction — without collapsing rows. Which SQL construct is correct?"}
- [ ] `GROUP BY user_id` with a `SUM(spend)` and a date filter in `WHERE`
- [x] A window function with `SUM(spend) OVER (PARTITION BY user_id ORDER BY txn_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW)`
- [ ] A self-join where you join the table to itself on `user_id` and `txn_date < txn_date + 30`
- [ ] A subquery using `HAVING SUM(spend) > 0`
:::

:::drill {type="mcq" q="Your training dataset has 1 million rows. After adding a LEFT JOIN to an event table, it has 1.4 million rows. What most likely happened?"}
- [ ] The JOIN introduced NULL rows for users with no events
- [ ] The WHERE clause filtered out some rows after the join
- [x] The event table had multiple rows per join key, causing row duplication
- [ ] The SELECT clause added computed columns that expanded the schema
:::

:::key-takeaway
SQL is your feature pipeline's foundation. Master window functions with explicit frame bounds to avoid label leakage, always verify join cardinality before trusting your training data, and write every feature query in readable CTEs — future-you will thank present-you at 2 AM during an incident.
:::
