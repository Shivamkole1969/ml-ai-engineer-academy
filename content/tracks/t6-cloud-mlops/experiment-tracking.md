---
id: experiment-tracking
track: t6-cloud-mlops
title: "MLOps: experiment tracking (MLflow)"
badge: HOT
minutes: 9
prereqs: []
tags: [mlops, mlflow, experiment-tracking, reproducibility, model-registry]
xp: 60
hot2026: true
---

Imagine you have trained 47 model variants over three weeks. Different learning rates, different feature sets, a couple of architectural tweaks. One of them crushed your evaluation metrics. You promoted it to production, your users are happy — and then, six weeks later, a new team member asks: "Which run was that, exactly? Can we retrain it?" You open your notebook. There is a cell that reads `lr = 0.0003  # maybe try 0.001`. Your gut drops. You have no idea which value you actually shipped.

This is the canonical MLOps failure that experiment tracking solves.

## What is experiment tracking?

Experiment tracking is the practice of logging — automatically, at training time — every variable that could affect a model's outcome: hyperparameters, dataset versions, metric curves, artifacts, and environment details. It turns your messy "let me just try this" workflow into a reproducible, searchable, auditable record.

**MLflow** is the most widely used open-source tool for this. It has four components:

- **Tracking** — log runs with params, metrics, and artifacts
- **Projects** — package code so anyone can reproduce a run
- **Models** — a standard format for deploying models across frameworks
- **Model Registry** — lifecycle management (staging → production → archived)

You can run MLflow entirely locally (SQLite + file system), which means zero cloud cost for experimentation.

:::why-prod
In production, reproducibility is a compliance requirement, not just a nice-to-have. If a regulator, an on-call engineer, or a customer support escalation asks "what model made that decision?", you need to point to an exact, reproducible artifact. Experiment tracking also cuts re-training costs: instead of re-running expensive GPU jobs to find a baseline, you retrieve the logged checkpoint.
:::

## Core concepts

:::table {title="MLflow tracking concepts at a glance"}
| Concept | What it is | Example |
|---|---|---|
| Experiment | A named group of related runs | `"churn-model-v2"` |
| Run | One training execution | A single `python train.py` call |
| Param | A fixed input value | `{"lr": 0.001, "epochs": 10}` |
| Metric | A numeric result (logged over time) | `{"val_auc": 0.87}` at epoch 5 |
| Artifact | Any file output | `model.pkl`, `confusion_matrix.png` |
| Tags | Free-form key-value metadata | `{"team": "growth", "data_version": "2026-Q1"}` |
| Model Registry | Versioned, staged model store | `churn-model` version 4, stage `Production` |
:::

## A minimal MLflow tracking loop

The API is intentionally simple. You wrap your training code in a `with mlflow.start_run()` block and log whatever matters.

```python {title="Minimal MLflow tracking with model registry" run=false}
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Point to your tracking server (local SQLite by default — no server needed)
# mlflow server --host 0.0.0.0 --port 5000   (run this once in a terminal)
# Then set: mlflow.set_tracking_uri("http://localhost:5000")
# For pure local dev, the default ./mlruns folder works with no server at all.

mlflow.set_experiment("churn-prediction")

hyperparams = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
}

with mlflow.start_run(run_name="gbt-baseline"):
    # 1. Log every hyperparameter upfront
    mlflow.log_params(hyperparams)

    # 2. Also log dataset info so you know what data was used
    mlflow.set_tags({
        "dataset_version": "2026-Q1",
        "feature_set": "v3-engineered",
        "author": "team-growth",
    })

    # Train (replace with your real data loading)
    model = GradientBoostingClassifier(**hyperparams)
    model.fit(X_train, y_train)       # X_train, y_train defined elsewhere

    # 3. Log metrics — call multiple times to track progress per epoch
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    mlflow.log_metric("val_auc", val_auc)

    # 4. Log the model as a reusable artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="churn-model",   # auto-registers in Model Registry
    )

    print(f"Run complete. val_auc={val_auc:.4f}")

# Load the best model later — no guessing, just fetch by name + version
# loaded = mlflow.sklearn.load_model("models:/churn-model/Production")
```

## Promoting a model to production via the registry

After several runs, you compare them in the MLflow UI (or programmatically) and promote the winner:

```python {title="Programmatic model promotion" run=false}
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition version 3 of "churn-model" from Staging to Production
client.transition_model_version_stage(
    name="churn-model",
    version=3,
    stage="Production",
    archive_existing_versions=True,   # demotes the old Production version
)
```

This creates a full audit trail: who promoted it, when, and from which run. No more "it was... the one I trained on Tuesday, I think."

:::gotcha
Logging at the end is the most common mistake. Engineers log metrics only after training completes — so if the job crashes at epoch 18 of 20, you lose all signal. Log metrics *inside* your training loop at every epoch (or every N steps). Also: forgetting to log the **data version** is the silent killer. A model trained on January data looks identical to one trained on March data until they behave differently in production. Tag your dataset version on every single run.
:::

:::war-story {title="The model that couldn't be found"}
A team at a mid-size fintech shipped a credit-scoring model that significantly reduced defaults. Three months later, the model drifted and they needed to retrain from the same baseline. Their "tracking" was a shared Google Doc with run notes — last updated six weeks before the incident. The notebook had six cells named `train_final`, `train_final_v2`, and `train_final_ACTUAL`. It took four engineers two full days to reconstruct the original feature pipeline and hyperparameters. Switching to MLflow tracking added about 12 lines of code to their training script. The next retraining took 20 minutes.
:::

## Comparing runs and picking a winner

In the MLflow UI (`mlflow ui` at the terminal), you can select multiple runs and overlay their metric curves side by side. You can also do it in code:

```python {title="Querying runs to find the best" run=false}
import mlflow

runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="metrics.val_auc > 0.85",
    order_by=["metrics.val_auc DESC"],
)

best_run_id = runs.iloc[0]["run_id"]
print(f"Best run: {best_run_id}  AUC: {runs.iloc[0]['metrics.val_auc']:.4f}")
```

:::interview-line
"MLflow tracking gives me a tamper-evident, reproducible record of every model I've trained — params, metrics, artifact, and data version — so I can promote or roll back with full confidence."
:::

:::qa {q="What is the difference between MLflow params and metrics?"}
Params are fixed inputs set before a run starts — hyperparameters, dataset paths, feature flags. Metrics are numeric outputs that can be logged repeatedly over time, such as loss or AUC at each epoch. The distinction matters because MLflow lets you plot metric history as a curve, while params are point-in-time configuration.
:::

:::qa {q="What is the MLflow Model Registry used for, and why does staging matter?"}
The registry stores versioned model artifacts with lifecycle stages: None → Staging → Production → Archived. Staging is a holding area for models that have passed automated tests but haven't been blessed for live traffic yet. This separation enforces a gate between "trained and promising" and "trusted and serving," which reduces accidental promotions and gives you a rollback target — you can flip back to the previous Production version in one API call.
:::

:::qa {q="How do you track the dataset version alongside your model in MLflow?"}
The simplest approach is to tag every run with a dataset identifier — a date string, a Git commit hash of your feature pipeline, or an object storage path with a content hash. More formally, MLflow 2.x introduced `mlflow.log_input()` to attach a `mlflow.data.Dataset` object directly to a run. Either way, the data version becomes a first-class field in your run record, not a comment in a notebook cell.
:::

:::drill {type="mcq" q="Your training job crashed at epoch 15 of 20. Which MLflow practice would have saved the most useful signal?"}
- [ ] Log all metrics in a single call after training completes
- [x] Log metrics inside the training loop at every epoch
- [ ] Use a larger instance type so the job doesn't crash
- [ ] Increase the number of registered model versions
:::

:::drill {type="mcq" q="You want to roll back your production model from version 5 to version 4. What is the correct MLflow action?"}
- [ ] Delete version 5 from the Model Registry
- [ ] Re-train version 4 from scratch with the same params
- [x] Transition version 4 back to the Production stage and archive version 5
- [ ] Update your serving code to load a local pickle file instead
:::

:::drill {type="mcq" q="Which of the following is NOT automatically captured by mlflow.sklearn.log_model()?"}
- [ ] The serialized model artifact
- [ ] The Python and scikit-learn version (in MLmodel metadata)
- [x] The training dataset used to fit the model
- [ ] A reusable model signature (input/output schema)
:::

:::key-takeaway
Log params, metrics, dataset version, and the model artifact on every training run — not just the ones you think will win. The run you almost deleted is often the one you need six months later.
:::
