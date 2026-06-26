# MLOps Deployment — Portfolio Build Guide

**Badge:** HOT · **Maps to:** AWS SageMaker / Bedrock résumé claims

---

## Scenario / Objective

Picture this: you trained a churn model last quarter. Accuracy looked great in the notebook. You Slacked the number to your manager. Three months later, nobody knows if it's still running, what version is live, or why it started misfiring after the Diwali sale spike changed the data distribution.

That's the gap MLOps fills.

In this project you'll take *any* trained model (we'll reuse the XGBoost churn model from the pipeline project, or a tiny scikit-learn one if you're starting fresh) and deploy it the way a real team would — containerized, monitored, with a canary rollout and a rollback path. By the end you'll have:

- A **Docker image** serving predictions over a FastAPI endpoint
- A **monitoring sidecar** that logs latency percentiles (p50/p99), drift scores, and (if using an LLM) token spend
- A **canary / rollback script** you can actually demo
- Cloud-optional — runs 100 % locally on Docker Compose, with clear notes on how each piece maps to SageMaker or Bedrock

:::why-prod
Real teams don't just train — they *operate* models. Monitoring and rollback are the difference between a model and a product. Interviewers who've been burned by silent model rot ask about this stuff directly.
:::

---

## Architecture

```
           ┌─────────────────────────────────────────────────┐
           │                  Client / App                    │
           └───────────────────┬─────────────────────────────┘
                               │  POST /predict
                               ▼
           ┌─────────────────────────────────────────────────┐
           │           Nginx (traffic split)                  │
           │   95 % → model-stable    5 % → model-canary     │
           └──────┬──────────────────────────┬───────────────┘
                  │                          │
      ┌───────────▼───────────┐  ┌───────────▼───────────┐
      │   FastAPI  v1 (stable) │  │   FastAPI  v2 (canary) │
      │   model: churn-v1.pkl  │  │   model: churn-v2.pkl  │
      └───────────┬───────────┘  └───────────┬────────────┘
                  │                           │
                  └──────────┬────────────────┘
                             │  prediction + metadata
                             ▼
           ┌─────────────────────────────────────────────────┐
           │          Prometheus scrape endpoint              │
           │  metrics: latency_p99, drift_score, token_spend  │
           └───────────────────┬─────────────────────────────┘
                               │
                               ▼
           ┌─────────────────────────────────────────────────┐
           │            Grafana dashboard                     │
           │   (or: CloudWatch if you're on SageMaker)        │
           └─────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────┐
  │  Reference: AWS equivalents                               │
  │  Nginx split    → SageMaker Endpoint with variant weights │
  │  FastAPI image  → SageMaker Model / Bedrock custom model  │
  │  Prometheus     → CloudWatch Model Monitor               │
  │  Grafana        → CloudWatch dashboards / QuickSight     │
  └──────────────────────────────────────────────────────────┘
```

---

## Repo Structure

```text
mlops-deployment/
├── model/
│   ├── train.py              # quick retrain to produce v1 + v2 artifacts
│   ├── churn-v1.pkl
│   └── churn-v2.pkl
│
├── serving/
│   ├── app.py                # FastAPI prediction endpoint
│   ├── schemas.py            # Pydantic request/response models
│   ├── metrics.py            # Prometheus counters + histograms
│   └── Dockerfile
│
├── monitoring/
│   ├── drift.py              # PSI / KS drift detector (no extra libs needed)
│   ├── reference_data.parquet  # baseline feature distribution
│   └── alert_rules.yml       # Prometheus alerting rules (latency, drift)
│
├── infra/
│   ├── docker-compose.yml    # spins up stable + canary + prometheus + grafana
│   ├── nginx.conf            # 95/5 upstream split
│   └── prometheus.yml
│
├── scripts/
│   ├── canary_promote.sh     # bump canary weight to 100 %, retire stable
│   ├── rollback.sh           # flip back to stable instantly
│   └── load_test.py          # locust-less load test using httpx
│
├── notebooks/
│   └── drift_analysis.ipynb  # EDA on live vs reference distributions
│
├── sagemaker/
│   ├── deploy_endpoint.py    # boto3: create SageMaker endpoint with variants
│   └── cloudwatch_monitor.py # enable Model Monitor data capture + baseline
│
└── README.md
```

---

## Milestone Checklist

### Phase 1 — Serve it (Week 1)

- [ ] Train two versions of the model (`train.py --version v1` and `--version v2`) so you have something to canary
- [ ] Write `serving/app.py`: a POST `/predict` endpoint that loads the pkl, validates input with Pydantic, returns a prediction + probability
- [ ] Add a `/health` and `/metrics` endpoint (Prometheus text format)
- [ ] Write `serving/Dockerfile` (base: `python:3.11-slim`, copy model artifact in)
- [ ] Confirm `docker build` + `docker run -p 8000:8000` returns a prediction on a curl

### Phase 2 — Monitor it (Week 2)

- [ ] Add `metrics.py`: a Prometheus `Histogram` for request latency, a `Counter` for prediction class distribution
- [ ] Implement `monitoring/drift.py`: compute PSI (Population Stability Index) between live request features and the reference baseline; expose as a Gauge metric
- [ ] Wire up `docker-compose.yml`: stable service + Prometheus + Grafana with a pre-built dashboard JSON
- [ ] Build a Grafana panel showing p99 latency, drift score, and request rate — screenshot it for the portfolio
- [ ] Write `alert_rules.yml`: fire an alert if `latency_p99 > 200ms` or `drift_score > 0.2`

### Phase 3 — Canary + Rollback (Week 3)

- [ ] Add the canary service to `docker-compose.yml` (same image, `MODEL_VERSION=v2` env var)
- [ ] Write `nginx.conf` with `upstream` blocks and `weight` directives for 95/5 split
- [ ] Write `canary_promote.sh`: edits nginx config to 50/50, then 0/100, then `docker rm` the stable container
- [ ] Write `rollback.sh`: stops canary, sets nginx upstream back to stable-only, restarts nginx
- [ ] Run `scripts/load_test.py` and capture the Grafana screenshot *during* canary promotion — this is interview gold

### Phase 4 — Cloud mapping (Week 4, optional but recommended)

- [ ] Deploy to SageMaker using `sagemaker/deploy_endpoint.py` with two `ProductionVariants` (stable + canary)
- [ ] Enable SageMaker Model Monitor via `cloudwatch_monitor.py`: data capture → baseline → monitoring schedule
- [ ] If using Bedrock: log `inputTokens` + `outputTokens` from the API response and push to CloudWatch as a custom metric
- [ ] Write one paragraph in your README: "How this maps to what SageMaker does natively" — you'll read this paragraph aloud in interviews

---

## Key Code Snippets

### 1. Prediction endpoint with built-in metrics

```python
# serving/app.py
import time, os, pickle
from fastapi import FastAPI, Request
from prometheus_client import Histogram, Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from schemas import PredictRequest, PredictResponse
from monitoring.drift import compute_psi, load_reference

app = FastAPI()

MODEL_PATH = os.getenv("MODEL_PATH", "model/churn-v1.pkl")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

reference_dist = load_reference("monitoring/reference_data.parquet")

# --- Prometheus metrics ---
LATENCY = Histogram(
    "prediction_latency_seconds",
    "End-to-end prediction latency",
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    labelnames=["version"],
)
PREDICTIONS = Counter(
    "predictions_total",
    "Count of predictions by class",
    labelnames=["version", "predicted_class"],
)

@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest):
    features = payload.to_array()          # shape: (1, n_features)

    start = time.perf_counter()
    proba = model.predict_proba(features)[0, 1]  # churn probability
    elapsed = time.perf_counter() - start

    predicted_class = "churn" if proba >= 0.5 else "no_churn"
    LATENCY.labels(version=MODEL_VERSION).observe(elapsed)
    PREDICTIONS.labels(version=MODEL_VERSION, predicted_class=predicted_class).inc()

    # background drift check — cheap, runs in-process
    drift_score = compute_psi(features, reference_dist)

    return PredictResponse(
        probability=round(float(proba), 4),
        predicted_class=predicted_class,
        model_version=MODEL_VERSION,
        drift_score=round(drift_score, 4),
    )

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": MODEL_VERSION}
```

### 2. Lightweight PSI drift detector (no extra lib needed)

```python
# monitoring/drift.py
import numpy as np
import pandas as pd

def load_reference(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def _psi_single(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index for one numeric feature."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pct = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0]
    actual_pct   = np.histogram(actual,   bins=np.percentile(expected, breakpoints))[0]

    # avoid division by zero / log(0)
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct) / len(expected)
    actual_pct   = np.where(actual_pct   == 0, 0.0001, actual_pct)   / len(actual)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))

def compute_psi(live_row: np.ndarray, reference: pd.DataFrame) -> float:
    """
    Averaged PSI across all features for a single live row vs reference dist.
    PSI < 0.1: stable | 0.1–0.2: watch | > 0.2: retrain signal
    """
    scores = []
    for i, col in enumerate(reference.columns):
        ref_vals  = reference[col].values
        live_vals = np.array([live_row[0, i]])
        # for a single row we compare against reference — good enough for a demo
        # in prod you'd batch live requests and compare distributions daily
        scores.append(_psi_single(ref_vals, np.tile(live_vals, len(ref_vals))))
    return float(np.mean(scores))
```

:::gotcha
PSI on a single row is a demo trick — the number is directionally useful but not statistically rigorous. In production you'd accumulate a batch of live requests (say, 1 hour's worth) and compare the *distribution* against your baseline. Say this out loud in an interview before they ask.
:::

### 3. Canary nginx split

```nginx
# infra/nginx.conf
upstream model_backend {
    server model-stable:8000 weight=95;
    server model-canary:8000 weight=5;
}

server {
    listen 80;
    location /predict {
        proxy_pass http://model_backend;
        proxy_set_header Host $host;
    }
    location /health {
        proxy_pass http://model_backend;
    }
}
```

### 4. Rollback script

```bash
#!/usr/bin/env bash
# scripts/rollback.sh — run this when canary goes bad

set -e

echo "Rolling back to stable..."

# Kill canary container
docker compose stop model-canary
docker compose rm -f model-canary

# Patch nginx upstream back to 100 % stable
sed -i '' 's/weight=0/weight=95/g' infra/nginx.conf   # re-enable stable
sed -i '' 's/weight=100/weight=5/g' infra/nginx.conf  # demote canary (won't matter, it's down)

# Reload nginx (zero-downtime)
docker compose exec nginx nginx -s reload

echo "Rollback complete. All traffic → model-stable."
```

### 5. Token spend tracking (Bedrock)

```python
# Paste this wherever you call Bedrock's converse / invoke_model API
import boto3, time

bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
cloudwatch = boto3.client("cloudwatch", region_name="ap-south-1")

def invoke_and_track(model_id: str, prompt: str) -> str:
    response = bedrock.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
    )
    usage = response["usage"]                # inputTokens, outputTokens, totalTokens
    text  = response["output"]["message"]["content"][0]["text"]

    # Push token spend to CloudWatch as a custom metric
    cloudwatch.put_metric_data(
        Namespace="MLOps/TokenSpend",
        MetricData=[
            {
                "MetricName": "InputTokens",
                "Value": usage["inputTokens"],
                "Unit": "Count",
                "Dimensions": [{"Name": "ModelId", "Value": model_id}],
            },
            {
                "MetricName": "OutputTokens",
                "Value": usage["outputTokens"],
                "Unit": "Count",
                "Dimensions": [{"Name": "ModelId", "Value": model_id}],
            },
        ],
    )
    return text
```

:::why-prod
Token spend is real money. A production Bedrock integration without spend tracking is a surprise AWS bill waiting to happen. Tracking it as a CloudWatch metric lets you set a billing alarm and catch runaway prompts before they hit your limit.
:::

---

## What to Show in an Interview

**The three-tab demo (keep it under 5 minutes):**

1. **Tab 1 — Terminal.** Run `docker compose up`. Show both model containers and nginx starting. Hit `/predict` with a curl. Show the `model_version` field in the JSON response flipping between `v1` and `v2` as you spam requests.

2. **Tab 2 — Grafana dashboard.** Show your pre-built dashboard: p99 latency panel, drift score panel, prediction class distribution. Run `scripts/load_test.py` and watch the charts move in real time.

3. **Tab 3 — Canary promotion.** Run `canary_promote.sh` while Grafana is visible. Talk through what's happening: "I'm shifting 5 % of traffic to v2, watching p99 stay below 200 ms, if it's clean I'll promote to 100 %." Then show `rollback.sh` — "and if drift spiked, this gets me back in 10 seconds."

:::interview-line
"Our canary split runs in nginx today. On SageMaker you'd do the same thing with ProductionVariants — same concept, AWS manages the load balancer. I've set that up too, want to see the boto3 code?"
:::

---

## Honest Talking Points

Things you genuinely understand after building this (say them confidently):

- **Why p99, not average.** Averages hide tail latency. One slow prediction in 100 tanks user experience. p99 catches it; average masks it.
- **PSI thresholds are heuristics.** < 0.1 is industry convention, not a physical law. Context matters — a PSI of 0.15 on a stable financial feature is different from 0.15 on a volatile one.
- **Canary isn't free.** You need enough traffic for the split to be statistically meaningful. 5 % of 10 requests is 0 or 1 request — not a test. In a low-traffic setting you'd use A/B testing with fixed cohorts instead.
- **SageMaker Model Monitor is opinionated.** It expects your data in a specific capture format and runs on a schedule, not per-request. Great for batch drift detection; not great for real-time alerting. For real-time you'd keep Prometheus / custom CloudWatch metrics.
- **Rollback is a config change, not a re-deploy.** That distinction matters in an incident. A re-deploy under load can make things worse. Nginx weight changes + `nginx -s reload` are near-instant with zero dropped connections.

:::key-takeaway
MLOps is mostly about *operational trust*: can you tell, right now, if your model is healthy? Can you change it without a 2 AM incident? This project gives you a live answer to both.
:::

---

## How This De-Fakes a Résumé Claim

**The claim:** "Deployed models on AWS SageMaker / Bedrock"

Without this project, that line might mean "I clicked through a SageMaker Studio tutorial and got a training job to run." That's fine to learn from, but it won't survive a 15-minute technical screen.

With this project you can say:

> "I containerized a model, set up a canary deployment with traffic splitting, and built monitoring for latency p99 and data drift using PSI. I then mapped each piece to its SageMaker equivalent — ProductionVariants for the split, Model Monitor for drift, CloudWatch for alerting. The local Docker setup let me iterate fast; the boto3 scripts show I can do the same thing on AWS when I have an account with the right IAM permissions."

That's a sentence that ends conversations — in a good way. You're not claiming to have SageMaker production ops experience you don't have. You're claiming to understand the *concepts* well enough to implement them yourself, and to know exactly how they map to managed services. That's exactly what a junior-to-mid hire is expected to know going in.

The monitoring piece — drift, latency, token spend — is what separates this from a "deployed a Flask app" story. It shows you think past launch day.
