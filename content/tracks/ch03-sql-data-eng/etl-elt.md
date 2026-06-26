---
id: etl-elt
track: ch03-sql-data-eng
title: "ETL vs ELT & the modern data stack"
badge: CORE
minutes: 9
prereqs: []
tags: [etl, elt, data-engineering, data-warehouse, dbt, pipeline]
xp: 45
hot2026: false
---

Imagine you've just joined a company where the data team takes three hours every night to "process" data before anyone can query it. The morning ML pipeline fails because the nightly job ran late. The data scientists are furious. The data engineers are exhausted. Sound familiar? This is the classic ETL hangover — and understanding why it happens (and what replaced it) is a superpower for any ML engineer who touches data.

## What ETL Actually Means

ETL stands for **Extract, Transform, Load** — in that order.

1. **Extract** — pull raw data from a source (a database, an API, a CSV).
2. **Transform** — clean it, join it, aggregate it — *before* it lands anywhere permanent.
3. **Load** — write the finished, ready-to-use data into the destination warehouse.

The transform step happens *outside* the warehouse, usually on a separate server with its own compute. This was fine in the 1990s when storage was expensive and warehouses were slow. You'd clean the mess before it touched the sacred warehouse.

The problem: that transformation server became a single point of failure. It was slow, hard to scale, and painful to debug. And every time a business rule changed, someone had to rewrite the transform logic.

## Enter ELT — Flip the Order, Change the Game

**ELT** (Extract, Load, Transform) reverses the last two steps:

1. **Extract** from the source.
2. **Load** raw data into the warehouse first.
3. **Transform** inside the warehouse using SQL.

Why does this work now when it didn't in 1995? Because modern cloud warehouses — Snowflake, BigQuery, Redshift, DuckDB — are massively parallel and cheap to scale. Transforming *inside* the warehouse means you use the warehouse's compute, which is already paid for and already fast.

:::why-prod
In production ML, your training data, feature pipelines, and evaluation metrics all live downstream of these transforms. If the pipeline architecture is fragile, your model retraining breaks silently. Understanding ELT means you can debug data outages yourself instead of waiting on the data team.
:::

## The Modern Data Stack in One Picture

The modern data stack usually looks like this:

:::table {title="Modern Data Stack Layers"}
| Layer | Tool Examples | What It Does |
|---|---|---|
| Ingestion | Fivetran, Airbyte, Kafka | Extract + Load raw data |
| Storage | Snowflake, BigQuery, Redshift, DuckDB | Holds raw + transformed data |
| Transformation | dbt (data build tool) | SQL-based transforms, versioned |
| Orchestration | Airflow, Prefect, Dagster | Schedules and monitors pipelines |
| BI / ML | Looker, Metabase, your feature store | Consumes clean data |
:::

The key insight: **dbt** (rhymes with "debt") is the tool that made ELT mainstream. It lets data teams write modular SQL transforms, test them, and version them in Git — just like software. If you see `dbt` in a job description or codebase, now you know what layer it sits in.

## A Minimal dbt-Style Transform

Even without a full warehouse setup, you can prototype ELT logic locally with DuckDB — it's free, runs in-process, and speaks SQL.

```python {title="Local ELT prototype with DuckDB" run=false}
import duckdb

# ELT step 1 & 2: Load raw data directly into DuckDB (the "warehouse")
con = duckdb.connect()  # in-memory, no install needed beyond: pip install duckdb

con.execute("""
    CREATE TABLE raw_events AS
    SELECT * FROM read_csv_auto('events.csv')
""")

# ELT step 3: Transform INSIDE the warehouse with SQL
con.execute("""
    CREATE TABLE daily_active_users AS
    SELECT
        DATE_TRUNC('day', event_time) AS day,
        COUNT(DISTINCT user_id)       AS dau
    FROM raw_events
    WHERE event_type = 'session_start'
    GROUP BY 1
""")

result = con.execute("SELECT * FROM daily_active_users ORDER BY day DESC LIMIT 5").df()
print(result)
# Run locally: python etl_prototype.py
# No cloud account needed — DuckDB is just a Python package
```

This is ELT in its simplest form: raw data lands first, transforms run as SQL inside the engine.

:::gotcha
The biggest ELT trap: loading *everything* raw sounds great until your warehouse bill triples. Always partition raw tables by date and add retention policies. Also, raw data often contains PII — load it into a restricted schema and transform into a sanitized layer before your ML pipelines touch it.
:::

## ETL vs ELT — Quick Decision Guide

Still wondering which to pick? Use this heuristic:

- Need to **mask PII before it ever lands in storage**? → ETL. Transform first, load clean.
- Working with a **modern cloud warehouse** and compliance isn't the constraint? → ELT. It's faster to iterate.
- **Streaming data** (Kafka, Kinesis)? → Often a hybrid: light ETL at ingestion (schema validation, deduplication), then ELT inside the warehouse for analytics.

In ML specifically, your feature engineering is essentially a transform layer. Feature stores (Feast, Tecton) sit at this intersection — they consume ELT outputs and serve low-latency features to models.

:::interview-line
"ELT pushes transformation into the warehouse because modern cloud compute is cheap and SQL is auditable — you get versioned, testable data pipelines without a separate transform server."
:::

:::qa {q="What's the main practical advantage of ELT over ETL for a data team?"}
ELT stores raw data first, so transforms are iterative and replayable — you can re-run SQL against the raw layer whenever business logic changes without re-ingesting from the source. In ETL you'd have to re-extract from upstream, which is slow and fragile.
:::

:::qa {q="Where does dbt fit in the modern data stack?"}
dbt is a transformation layer: it runs SQL models inside your warehouse, manages dependencies between models, generates documentation, and adds data tests — all version-controlled in Git. It makes the "T" in ELT a first-class software engineering practice.
:::

:::qa {q="Why might an ML engineer care about ETL/ELT architecture?"}
Because ML pipelines consume data produced by these transforms. A badly designed transform layer means stale features, schema surprises at training time, and silent data drift. Knowing the architecture helps you debug pipeline failures and advocate for data quality upstream of your models.
:::

:::drill {type="mcq" q="A company stores raw clickstream data in BigQuery, then runs SQL models with dbt to produce feature tables for their recommender system. Which pattern is this?"}
- [ ] ETL — because they extract from clickstream before transforming
- [x] ELT — raw data loads first, then transforms run inside the warehouse using SQL
- [ ] Reverse ETL — because the data flows toward the ML model
- [ ] Streaming ETL — because clickstream implies real-time
:::

:::drill {type="mcq" q="You need to strip credit card numbers from raw transaction records before they land anywhere in your data warehouse. Which approach fits best?"}
- [ ] ELT — load everything raw, then mask with a dbt model
- [ ] Use a feature store to handle masking at serving time
- [x] ETL — transform (mask PII) before loading into any storage layer
- [ ] Either works equally well for compliance use cases
:::

:::key-takeaway
ELT won because cloud warehouses are cheap and SQL is auditable. Load raw, transform inside — but always mask PII before it touches storage, and treat your transform layer like real software (version it, test it, monitor it).
:::
