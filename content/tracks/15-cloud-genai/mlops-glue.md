---
id: mlops-glue
track: 15-cloud-genai
title: "MLOps glue: MLflow, Docker, FastAPI serving, Terraform note"
badge: HOT
minutes: 11
prereqs: []
tags: [mlflow, docker, fastapi, terraform, mlops, serving, deployment]
xp: 60
hot2026: true
---

It's 11 PM on a Friday. Your fine-tuned embedding model just beat the baseline by 4 points on recall@10. You push it to S3, wire it into an endpoint, Slack the team — and go to sleep.

Monday morning a teammate runs the same eval. Different number. "Which model is actually in prod?" Nobody knows. The S3 path changed. The Docker image is tagged `latest`. There are three `model.pkl` files with no dates on them.

This is the gap MLOps glue fills. **MLflow** remembers your experiments. **Docker** makes your environment portable. **FastAPI** exposes predictions cleanly. **Terraform** makes the infra reproducible. These four tools are the connective tissue between "it works on my laptop" and "it works in prod on AWS or GCP" — and every MLOps / ML Engineer role in India's funded startups will ask about at least two of them.

## MLflow: your experiment memory

MLflow is an open-source platform for tracking ML experiments, storing models, and managing their lifecycle. Think of it as Git — but for training runs, metrics, and model artifacts.

Every time you train, you log three things:

- **Parameters** — learning rate, base model name, chunk size, number of epochs
- **Metrics** — loss, F1, recall@10, latency
- **Artifacts** — the model files, tokenizer, confusion matrix PNG

MLflow stores all of it in a tracking server. You can open a browser UI, compare 20 runs side by side, and promote the winner to a **Model Registry** with a stage label like `Staging` or `Production`. Your serving code then loads `models:/my-model/Production` — and you can swap the champion model without touching the serving code at all.

:::why-prod
Without experiment tracking, "which model is in prod?" becomes a forensic investigation every time something breaks. MLflow makes rollback trivial: you know exactly which params, data version, and artifact produced the model that is currently running.
:::

Self-hosting: a t2.micro EC2 + SQLite backend + `mlflow server` is free-tier friendly and enough for a portfolio project.

```python {title="MLflow: log a fine-tuning run and register the winner" run=false}
import mlflow

# Point to local server (or Databricks managed URL)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("embedding-finetune-v2")

with mlflow.start_run(run_name="bge-small-lr1e-4"):
    mlflow.log_param("base_model", "BAAI/bge-small-en-v1.5")
    mlflow.log_param("learning_rate", 1e-4)
    mlflow.log_param("epochs", 3)

    # --- your training loop here ---
    val_recall = 0.74   # result from eval

    mlflow.log_metric("recall@10", val_recall)
    mlflow.log_artifact("./model_output/")   # entire folder saved

    # Register to Model Registry only if it beats the baseline
    if val_recall > 0.70:
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(
            f"runs:/{run_id}/model_output",
            "embedding-model-prod"
        )

# To view: mlflow ui   →  http://localhost:5000
```

## Docker: package it once, run it everywhere

A Docker image bundles your code, Python version, CUDA version, and every pinned dependency into a single artifact. Push it to ECR (AWS) or Artifact Registry (GCP) and it runs identically on your laptop, a SageMaker endpoint, Cloud Run, or a bare EC2 — no "but it worked in my venv" surprises.

For ML: never use `FROM python:latest` as a base. Pin it — `FROM python:3.11-slim` — and pin every package in `requirements.txt`. A silent numpy or transformers upgrade can change random seeds, numerical precision, or tokenisation behaviour.

:::gotcha
Never tag your production image `latest`. Use an immutable tag — your MLflow run ID, a git commit SHA, or a timestamp. `latest` means you cannot reproduce what was running two weeks ago, and rollback becomes guesswork.
:::

## FastAPI: the inference API your team will actually call

FastAPI is the de facto standard for serving ML models as HTTP APIs in Python. It is fast (async, Starlette underneath), auto-generates OpenAPI docs at `/docs`, and validates request/response shapes via Pydantic — no glue code required.

SageMaker and Vertex AI both support bring-your-own-container. FastAPI inside Docker is the cleanest pattern.

```python {title="FastAPI inference endpoint with MLflow model" run=false}
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import uvicorn

app = FastAPI(title="Embedding API")

# Load the model ONCE at container startup, not inside the route handler
MODEL_URI = "models:/embedding-model-prod/Production"
model = mlflow.pyfunc.load_model(MODEL_URI)

class PredictRequest(BaseModel):
    texts: list[str]

class PredictResponse(BaseModel):
    embeddings: list[list[float]]

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    result = model.predict(req.texts)   # returns np.ndarray
    return PredictResponse(embeddings=result.tolist())

@app.get("/health")
async def health():
    return {"status": "ok"}

# Run locally:  uvicorn main:app --reload
# In Docker:   uvicorn main:app --host 0.0.0.0 --port 8080
```

:::why-prod
Loading the model inside the route handler means cold-loading from disk on every request — that is 2–10 seconds of latency per call. Loading once at startup keeps the model in memory. Subsequent requests run inference only, which is milliseconds. On SageMaker endpoints the startup cost is paid once when the container warms up, not per invocation.
:::

:::table {title="Tool roles at a glance"}
| Tool | What it owns | Where it lives in the cloud |
|---|---|---|
| MLflow | Experiment logs, metrics, model registry | Databricks / self-host on EC2 |
| Docker | Portable runtime (code + deps + CUDA) | ECR (AWS) / Artifact Registry (GCP) |
| FastAPI | HTTP serving layer, request validation | SageMaker BYO / Cloud Run / EC2 |
| Terraform | Infra as code — spins up and tears down everything above | All clouds |
:::

## Terraform: one paragraph, but don't skip it

Terraform lets you define your cloud infrastructure — EC2 instance, SageMaker endpoint, S3 bucket, VPC — as `.tf` files that live in Git. `terraform apply` creates it; `terraform destroy` kills it. You do not need to write it from scratch for an interview. You do need to know: resources have types, blocks have arguments, `terraform plan` shows a dry run, and state tracks what is actually deployed. Free HCP Terraform handles remote state for small side projects. Every MLOps role in India's funded startups expects you to read and modify Terraform configs — this is your résumé gap to close.

:::war-story {title="The ₹40,000 weekend bill"}
A team at a Pune fintech trained a large model on a GPU instance on Friday afternoon. No Terraform, no tagging, no automation — just an EC2 clicked through the AWS console. They forgot to shut it down. The instance ran all weekend. Monday brought a ₹40,000 AWS bill and a very uncomfortable call with the VP Engineering. A `terraform destroy` scheduled via a cron Lambda, or even a billing alert on a tagged resource, would have caught it within hours. Instead, they found out from the finance team.
:::

:::interview-line
"I track every experiment with MLflow, containerize serving in Docker with an immutable image tag, expose predictions via FastAPI with model loaded at startup, and manage infra lifecycle in Terraform so nothing runs untagged over a long weekend."
:::

:::qa {q="Why load the model at application startup instead of inside the route handler?"}
Loading inside the handler means the model deserialises from disk on every single request — expect 2–10 seconds of added latency per call. Loading once at startup keeps the model warm in memory. Requests then pay only inference cost, not I/O cost. On managed endpoints like SageMaker, this also means the startup penalty is paid once at container warm-up, not charged per API invocation.
:::

:::qa {q="What is a Model Registry and why does it matter for production deployments?"}
A model registry is a versioned catalogue of trained models with lifecycle stage labels — Staging, Production, Archived. It decouples training from serving: your serving code always loads `models:/my-model/Production`. When a better model is ready, you promote it in the registry; serving picks it up without a code redeploy. Rollback is equally simple — demote the bad version and promote the previous one. MLflow, SageMaker Model Registry, and Vertex AI Model Registry all follow this pattern.
:::

:::qa {q="How would you roll back a model that turned out to be worse in production?"}
In the MLflow Model Registry, transition the previous version back to Production and archive the bad one. Because your serving code loads by stage alias rather than a hardcoded run ID, the rollback takes effect on the next request — no container redeployment needed. If you also used immutable Docker tags tied to run IDs, you can simultaneously roll back the container image for a full reset. The key enabler is never hardcoding a model path directly in serving code.
:::

:::drill {type="mcq" q="You tag your Docker image `latest` and deploy to production. Three weeks later a silent dependency upgrade breaks inference. What was the root mistake?"}
- [ ] Using Docker at all — virtual environments are safer for ML in production
- [x] Using a mutable tag (`latest`) instead of an immutable one like a git SHA or MLflow run ID
- [ ] Not using Kubernetes — it would have prevented the dependency upgrade
- [ ] Deploying on a Friday afternoon
:::

:::drill {type="mcq" q="Which MLflow component lets your serving code always load 'the current production champion' without hardcoding an S3 path or run ID?"}
- [ ] MLflow Tracking Server — it exposes a REST API for metrics
- [ ] MLflow Autolog — it captures params and metrics automatically
- [x] MLflow Model Registry with stage aliases, e.g. `models:/my-model/Production`
- [ ] MLflow Projects — it packages training code into reproducible runs
:::

:::key-takeaway
MLflow + Docker + FastAPI is the standard trio for reproducible, observable, portable model serving on any cloud. Track every run, pin every image tag, load the model once at startup, and let Terraform own the infra lifecycle — so nothing runs untagged and unbilled over a long weekend.
:::
