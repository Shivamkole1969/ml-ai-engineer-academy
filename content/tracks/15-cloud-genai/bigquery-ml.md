---
id: bigquery-ml
track: 15-cloud-genai
title: "BigQuery ML: train/predict in SQL"
badge: CORE
minutes: 8
prereqs: []
tags: [bigquery, bqml, gcp, sql, ml, prediction, free-tier]
xp: 45
hot2026: false
---

It is 6 PM on a Friday. Your data analyst pings you: "We have three million rows of customer transactions in BigQuery. Can we predict who will churn next month?" Your ML pipeline is still two sprints away. Your data science teammate is on leave.

Here is the thing — the data is already in BigQuery. Moving it out, preprocessing it, training a model somewhere else, and wiring up predictions would take you the whole weekend. But what if you could write a `CREATE MODEL` statement the same way you write a `CREATE TABLE`? Train the model, evaluate it, and get predictions — all in SQL, all inside BigQuery, without moving a single row?

That is BigQuery ML (BQML). Let us learn it.

## What BigQuery ML actually does

BigQuery ML lets you train, evaluate, and run inference on ML models using standard SQL. The model lives in your BigQuery dataset, next to your tables. There is no separate training cluster to provision, no data export, no serialisation step.

The workflow is three SQL statements:

1. `CREATE OR REPLACE MODEL` — trains the model
2. `ML.EVALUATE` — checks accuracy metrics
3. `ML.PREDICT` — runs inference on new rows

:::why-prod
When your data already lives in BigQuery (which it often does in data-warehouse-first orgs), training elsewhere means a painful export-transform-load cycle on millions of rows. BQML removes that entirely — the model trains where the data is, predictions happen where the data is, and your BI tools can query predictions like any other table.
:::

## Model types you need to know

BQML supports a surprisingly wide range of models for a SQL tool.

:::table {title="BQML model types — the ones that matter"}
| Model type | `model_type` value | Typical use case |
|---|---|---|
| Logistic regression | `LOGISTIC_REG` | Churn, fraud, binary classification |
| Linear regression | `LINEAR_REG` | Price prediction, demand forecasting |
| Boosted trees | `BOOSTED_TREE_CLASSIFIER` / `REGRESSOR` | Tabular classification/regression (often best accuracy) |
| K-Means clustering | `KMEANS` | Customer segmentation, anomaly grouping |
| ARIMA+ (time-series) | `ARIMA_PLUS` | Forecasting metrics, sales trends |
| Imported TF/PyTorch | `TENSORFLOW` / `ONNX` | Bring your own model, serve inside BQ |
| Remote model (LLM) | `REMOTE_MODEL` | Call Gemini or a Vertex endpoint directly from SQL |
:::

The remote model type is the bridge to GenAI — you can call Gemini for text classification or summarisation right from a `SELECT` statement. That is covered in the Vertex AI and Gemini lessons; here we focus on the native BQML training flow.

## The three-statement loop

```sql {title="BQML churn model — full lifecycle in SQL" run=false}
-- Step 1: Train
-- BigQuery does the feature scaling, cross-validation, and early stopping for you.
-- The model is stored in your dataset like a table.
CREATE OR REPLACE MODEL `my_project.my_dataset.churn_model`
OPTIONS (
  model_type = 'LOGISTIC_REG',
  input_label_cols = ['churned'],   -- target column
  auto_class_weights = TRUE,        -- handles class imbalance automatically
  max_iterations = 20
) AS
SELECT
  sessions_last_30d,
  avg_order_value,
  days_since_last_order,
  support_tickets_open,
  churned                            -- 1 = churned, 0 = retained
FROM `my_project.my_dataset.customer_features`
WHERE partition_date < '2024-10-01'; -- train on historical data

-- Step 2: Evaluate (on held-out rows)
SELECT
  precision,
  recall,
  f1_score,
  roc_auc
FROM ML.EVALUATE(MODEL `my_project.my_dataset.churn_model`,
  (SELECT * FROM `my_project.my_dataset.customer_features`
   WHERE partition_date >= '2024-10-01'));

-- Step 3: Predict on current customers
SELECT
  customer_id,
  predicted_churned,
  predicted_churned_probs[OFFSET(1)].prob AS churn_probability
FROM ML.PREDICT(MODEL `my_project.my_dataset.churn_model`,
  (SELECT * FROM `my_project.my_dataset.customer_features`
   WHERE partition_date = CURRENT_DATE()))
ORDER BY churn_probability DESC
LIMIT 500;  -- top 500 highest-risk customers for outreach
```

Notice what you did not write: no Python, no scikit-learn, no Docker, no batch job. The training job runs asynchronously inside BigQuery's infrastructure. You can watch it in the GCP console or poll with `ML.TRAINING_INFO`.

## Practical notes for the free tier

BigQuery's free tier gives you 10 GB of storage and 1 TB of query processing per month. BQML training does consume query bytes — a logistic regression on 500K rows typically costs a few GB of processing, well inside the free limit for experimentation.

The `bq` CLI (part of the Google Cloud SDK) lets you kick off and monitor jobs locally. Running `bq query --use_legacy_sql=false` with your SQL file is the cheapest local workflow.

:::gotcha
BQML silently drops columns it cannot use — strings it cannot encode, NULLs above a threshold, columns that leak the label. Always run `ML.FEATURE_INFO(MODEL ...)` after training to confirm which features actually made it in. Discovering a key feature was silently excluded after you have presented results is not a fun conversation.
:::

:::interview-line
"BigQuery ML lets me train and serve predictions entirely in SQL — no data movement, no separate cluster. For tabular classification on warehouse-scale data it is often the fastest path from idea to production inference."
:::

:::qa {q="When would you choose BigQuery ML over Vertex AI AutoML or a custom training job?"}
BQML is the right choice when your data already lives in BigQuery, the team is SQL-fluent, and you need a fast turnaround — think analysts who own the feature pipeline and want to ship predictions without handing off to an ML engineer. For custom architectures, image/text models, or when you need fine-grained control over training infrastructure, Vertex AI custom training or AutoML is the better fit.
:::

:::qa {q="How does BQML handle a remote LLM model? What does the SQL look like?"}
You create a `REMOTE_MODEL` that points to a Vertex AI endpoint or a Gemini model via a connection resource. Then `ML.GENERATE_TEXT` or `ML.PREDICT` sends each row's prompt to the LLM and returns the response inline with your table data. It is powerful for batch text classification or enrichment jobs, but each row is a real API call — cost adds up fast on large tables, so always filter aggressively before calling.
:::

:::drill {type="mcq" q="You run ML.FEATURE_INFO on your trained BQML model and notice a feature column you expected to be important is missing from the output. What is the most likely cause?"}
- [ ] BQML only supports up to 20 input features
- [ ] The column name has uppercase letters which BQML does not support
- [x] BQML silently dropped the column — possibly due to too many NULLs, a data type it cannot encode, or the column leaking the label
- [ ] You need to run ML.TRAINING_INFO first to unlock feature visibility
:::

:::drill {type="mcq" q="Which BQML model_type would you pick for predicting next month's daily sales volume (a continuous number)?"}
- [ ] LOGISTIC_REG
- [ ] KMEANS
- [x] ARIMA_PLUS
- [ ] BOOSTED_TREE_CLASSIFIER
:::

:::key-takeaway
BigQuery ML brings the model to the data — train, evaluate, and predict in SQL with no data movement. It is the fastest path to a working ML feature when your warehouse is already BigQuery. Run `ML.FEATURE_INFO` after every training run so silent feature drops do not bite you.
:::
