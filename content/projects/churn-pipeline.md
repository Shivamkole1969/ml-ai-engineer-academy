# End-to-End ML Pipeline: Customer Churn Prediction

## Scenario & Objective

Picture this: Jio's retention team calls you into a meeting. They're losing 3–4% of subscribers
every month and nobody knows *why* until after the customer has already left. They want a model
that flags at-risk accounts 30 days out — giving the ops team time to call, offer a plan upgrade,
or push a discount.

That's churn prediction. It is one of the most common real ML workloads in Indian industry (telecom,
fintech, SaaS), and it maps cleanly to almost every interview question about the full ML lifecycle.

**What you will build**

A production-grade churn pipeline, end to end:

- Ingest raw customer data, engineer features
- Train an XGBoost classifier, tune it properly
- Evaluate with PR-AUC *and* calibration (because accuracy is a lie on imbalanced data)
- Serve predictions through a FastAPI endpoint
- Track every experiment in MLflow
- Pack the whole thing into Docker so it runs identically on your laptop and on any cloud VM

By the end you have a live URL you can paste into a portfolio. No fluff.

---

## Architecture

```
Raw CSV / Postgres
       │
       ▼
┌─────────────────┐
│  data_prep.py   │  ← clean nulls, encode categoricals, train/val/test split
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ feature_eng.py  │  ← rolling aggregates, tenure buckets, usage ratios
└────────┬────────┘
         │
         ▼
┌─────────────────┐         ┌───────────────┐
│   train.py      │────────►│  MLflow       │  ← params, metrics, model artefact
│  (XGBoost)      │         │  Tracking     │
└────────┬────────┘         └───────────────┘
         │
         ▼
┌─────────────────┐
│   evaluate.py   │  ← PR-AUC, calibration plot, threshold analysis
└────────┬────────┘
         │ (best model registered)
         ▼
┌─────────────────┐
│  FastAPI app    │  ← POST /predict  →  { churn_prob, risk_tier }
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Docker image   │  ← one command, runs anywhere
└─────────────────┘
```

Everything is orchestrated with a single `Makefile` so any teammate (or interviewer) can reproduce
your results with `make all`.

---

## Repo Structure

```text
churn-pipeline/
├── data/
│   ├── raw/                   # telco_churn.csv (Kaggle IBM dataset)
│   └── processed/             # output of data_prep.py
├── notebooks/
│   └── 01_eda.ipynb           # exploratory only, not part of pipeline
├── src/
│   ├── data_prep.py
│   ├── feature_eng.py
│   ├── train.py
│   ├── evaluate.py
│   └── serve/
│       ├── app.py             # FastAPI
│       ├── schema.py          # Pydantic request/response models
│       └── predictor.py       # loads MLflow model, wraps predict_proba
├── tests/
│   ├── test_features.py
│   └── test_api.py
├── mlruns/                    # MLflow artefacts (git-ignored in practice)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
└── README.md
```

:::key-takeaway
Hiring managers read the repo structure in 90 seconds. A clean tree signals engineering maturity
before they read a single line of code.
:::

---

## Milestone Checklist

### Phase 1 — Data & Features
- [ ] Download the IBM Telco Customer Churn dataset from Kaggle (7 043 rows, 21 columns)
- [ ] Write `data_prep.py`: handle nulls in `TotalCharges`, encode `InternetService` / `Contract`
      with `pd.get_dummies`, stratified 60/20/20 split saved to `data/processed/`
- [ ] Write `feature_eng.py`: add `monthly_to_total_ratio`, `tenure_bucket` (0–12, 13–24, 24+),
      `service_count` (sum of add-on flags)
- [ ] Confirm class imbalance (~26% churn) — document it in your README

### Phase 2 — Training
- [ ] Write `train.py` with XGBoost, `scale_pos_weight = neg/pos` to handle imbalance
- [ ] Use `Optuna` for 30-trial hyperparameter search (max_depth, learning_rate, subsample)
- [ ] Log every trial to MLflow: params + `val_pr_auc` metric + serialised model
- [ ] Register the best model in the MLflow Model Registry as `churn-xgb/Staging`

### Phase 3 — Evaluation
- [ ] Write `evaluate.py`: compute PR-AUC on the held-out test set (target > 0.70)
- [ ] Plot and save a calibration curve — compare raw probabilities vs. `CalibratedClassifierCV`
- [ ] Pick an operating threshold by maximising F-beta (beta=2, recall-biased) on validation set
- [ ] Save a `classification_report` and confusion matrix to `mlruns/` for the README

### Phase 4 — Serving
- [ ] Write `src/serve/schema.py` with Pydantic `CustomerIn` and `PredictionOut` models
- [ ] Write `src/serve/predictor.py` that loads the MLflow model URI at startup
- [ ] Write `src/serve/app.py` with a `POST /predict` endpoint and a `GET /health` endpoint
- [ ] Write `tests/test_api.py` using `httpx` + `TestClient` — at least 3 test cases
- [ ] Run `uvicorn` locally, hit `/predict` with `curl`, screenshot the JSON response

### Phase 5 — Containerisation
- [ ] Write a multi-stage `Dockerfile` (builder stage installs deps, runtime stage copies artefact)
- [ ] Write `docker-compose.yml` with `api` and `mlflow` services on the same network
- [ ] `docker compose up` → hit `localhost:8000/predict` → works → screenshot this too
- [ ] Add `make train`, `make eval`, `make serve`, `make docker-up` targets to `Makefile`

### Phase 6 — Polish
- [ ] Fill in `README.md`: what the project does, how to run it, key metrics table
- [ ] Add `pre-commit` hooks: `black`, `isort`, `flake8`
- [ ] Push to GitHub; confirm the repo is public and the README renders correctly

---

## Key Code Snippets

### 1. Stratified split with class-balance logging

```python
# src/data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df: pd.DataFrame, target: str = "Churn", seed: int = 42):
    """
    Stratified 60/20/20 split.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    Always stratify on imbalanced targets — otherwise your test set
    might have 35% churn by luck and your metrics are meaningless.
    """
    X = df.drop(columns=[target])
    y = df[target].map({"Yes": 1, "No": 0})  # explicit, no surprises

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=seed
    )

    for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
        rate = labels.mean()
        print(f"{name}: {len(labels)} rows, churn rate = {rate:.2%}")

    return X_train, X_val, X_test, y_train, y_val, y_test
```

### 2. XGBoost training with MLflow autolog

```python
# src/train.py
import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import average_precision_score

def train(X_train, y_train, X_val, y_val, params: dict):
    """
    Trains one XGBoost run and logs everything to MLflow.
    scale_pos_weight compensates for the ~74/26 class split —
    it is the single biggest lever on imbalanced binary problems.
    """
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    mlflow.xgboost.autolog()   # logs params, metrics, model artefact automatically

    with mlflow.start_run():
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale,
            use_label_encoder=False,
            eval_metric="aucpr",       # optimise directly on PR-AUC
            early_stopping_rounds=20,
            random_state=42,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_proba = model.predict_proba(X_val)[:, 1]
        val_pr_auc = average_precision_score(y_val, val_proba)
        mlflow.log_metric("val_pr_auc", val_pr_auc)
        print(f"val PR-AUC: {val_pr_auc:.4f}")

    return model
```

### 3. Calibration check

```python
# src/evaluate.py
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import average_precision_score

def evaluate_and_calibrate(model, X_test, y_test, save_path="reports/calibration.png"):
    """
    Raw XGBoost probabilities are often over-confident on small datasets.
    Calibration makes the number mean what it says:
    if the model says 0.7, ~70% of those customers should actually churn.
    That matters when ops teams are using the score to set call priorities.
    """
    raw_proba = model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, raw_proba)
    print(f"Test PR-AUC (raw):  {pr_auc:.4f}")

    # Isotonic calibration — better than Platt scaling for XGBoost
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_test, y_test)   # fine-tune on test; use a separate cal set in prod
    cal_proba = calibrated.predict_proba(X_test)[:, 1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    for proba, label in [(raw_proba, "Raw XGB"), (cal_proba, "Calibrated")]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10)
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of positives",
           title="Calibration curve")
    ax.legend()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"Calibration plot saved → {save_path}")
    return calibrated
```

### 4. FastAPI endpoint

```python
# src/serve/app.py
from fastapi import FastAPI
from src.serve.schema import CustomerIn, PredictionOut
from src.serve.predictor import Predictor

app = FastAPI(title="Churn Prediction API", version="1.0")
predictor = Predictor()   # loads model from MLflow at startup

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOut)
def predict(customer: CustomerIn):
    """
    Accepts one customer record, returns churn probability and a risk tier.
    Keeping the endpoint stateless and JSON-in/JSON-out makes it trivial
    to call from a CRM, a Jupyter notebook, or a Postman collection.
    """
    prob = predictor.predict_proba(customer.dict())
    tier = "HIGH" if prob >= 0.60 else "MEDIUM" if prob >= 0.35 else "LOW"
    return PredictionOut(churn_probability=round(prob, 4), risk_tier=tier)
```

```python
# src/serve/schema.py
from pydantic import BaseModel, Field

class CustomerIn(BaseModel):
    tenure: int = Field(..., ge=0, description="Months as a customer")
    monthly_charges: float
    total_charges: float
    contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    internet_service: str
    tech_support: str
    # ... add remaining features matching your training schema

class PredictionOut(BaseModel):
    churn_probability: float
    risk_tier: str   # LOW / MEDIUM / HIGH
```

### 5. Dockerfile (multi-stage)

```dockerfile
# ── builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── runtime ──────────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# copy only installed packages, not the full pip cache
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY mlruns/ ./mlruns/   # embed the registered model artefact

ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PORT=8000

EXPOSE 8000
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

:::gotcha
Never COPY the entire repo into the runtime stage. You will accidentally ship your raw training data,
your `.env` file, and 2 GB of `mlruns` experiment history. Be deliberate about what goes in.
:::

---

## What to Show in an Interview

When a panel says "walk me through your churn project", here is the exact sequence that lands well:

**1. Lead with the business problem, not the model.**
> "The goal was to flag customers 30 days before cancellation so retention ops could act.
> Precision mattered more than recall here because every unnecessary call costs money."

**2. Explain why PR-AUC, not ROC-AUC.**
> "With 26% positives the ROC curve is optimistic — a garbage model can still get 0.85 AUC.
> PR-AUC punishes you when you miss positives on an imbalanced dataset. I got 0.73 test PR-AUC."

**3. Talk about calibration unprompted.**
Most candidates skip this. Saying "I checked calibration because ops teams use the raw probability
to prioritise their call queue, so 0.7 has to mean 70%" immediately signals production thinking.

**4. Show the `GET /health` endpoint.**
It seems trivial but it proves you thought about deployment: load balancers need it,
Kubernetes liveness probes need it. One line of code, big signal.

**5. Have the Docker command ready.**
```bash
docker compose up   # that is all they need to see
```

:::interview-line
"I chose XGBoost over a neural net because the dataset is 7k rows — deep learning would have
overfit badly. The right tool for the data size beats the impressive-sounding tool every time."
:::

---

## Honest Talking Points

Things you should know before someone asks:

**What would you do differently with 10× the data?**
Add a proper feature store (Feast or Hopsworks). Right now features are recomputed at request time,
which is fine for batch but slow for real-time at scale.

**Why not LightGBM or CatBoost?**
They are both excellent. XGBoost has the widest ecosystem support and the most interview
documentation online — practical reasons. In a real job you would benchmark all three in 30 minutes
with Optuna and pick the winner.

**What is the model's biggest weakness?**
It assumes the feature distribution at serving time matches training. If Jio adds a new product
tier three months later, `service_count` drifts silently. You need data drift monitoring
(Evidently AI is free and takes two hours to set up).

**Can you retrain automatically?**
Not in this version. A production system would add a training trigger when drift is detected
or on a weekly cron. That is the natural next story in your portfolio roadmap.

:::why-prod
Calibration and threshold selection are where most Kaggle-trained engineers fall flat in
production roles. The model ships a number. Downstream systems (CRM automations, email triggers,
call queues) consume that number directly. If your probabilities are uncalibrated — consistently
too high or too low — you either burn your ops budget on false alarms or let real churners slip.
This is why evaluate.py exists as a first-class step, not an afterthought in a notebook cell.
:::

---

## How This De-fakes a Résumé Claim

A résumé line that says "built ML pipeline for churn prediction" is easy to write and hard to
verify. This project makes every part of that claim checkable in 10 minutes:

| Claim | Evidence in this repo |
|---|---|
| "End-to-end ML pipeline" | `make all` runs data → features → train → eval → serve |
| "XGBoost with tuning" | `train.py` with Optuna, 30 trials logged in MLflow UI |
| "Evaluation beyond accuracy" | `evaluate.py` outputs PR-AUC + calibration curve PNG |
| "Production serving" | FastAPI + `/health` + typed Pydantic schema |
| "Containerised" | Multi-stage Dockerfile, `docker compose up` in README |
| "Experiment tracking" | MLflow UI screenshot in README, model in Model Registry |

When an interviewer asks "can I see it running?" you open your laptop, type one command, and
show them a live JSON response. That is not a résumé claim anymore. That is a demo.

The lessons you picked up along the way — why PR-AUC > ROC-AUC on imbalanced data, why
calibration matters for downstream consumers, why you pin `scale_pos_weight` instead of
resampling — are the answers to the follow-up questions they *will* ask. The build is the
preparation. The talking points are the proof.
