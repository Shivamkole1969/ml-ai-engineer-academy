# ML Monitoring & Drift Platform

Your team shipped a churn prediction model last quarter. Accuracy on the test set: 91%. Three months later, product ops calls — "why are we flagging the wrong customers?" You check the model. Weights are unchanged. The bug isn't in the model. It's in the *world*. Users started paying differently after a festive-season promo. The feature distribution silently shifted, and nobody noticed.

This is why drift monitoring exists. Not a luxury — a survival layer for any ML system you care about in production.

---

## Architecture Overview

<svg viewBox="0 0 860 140" width="100%" role="img" aria-label="ML monitoring pipeline: prediction logs flow into a stream processor, into a stats engine, into a drift detector, then alerts and dashboards">

  <!-- Box 1: Prediction Logs -->
  <rect x="10" y="42" width="130" height="52" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="75" y="63" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Prediction</text>
  <text x="75" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Logs</text>
  <text x="75" y="89" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">features + output</text>

  <!-- Arrow 1 -->
  <line x1="140" y1="68" x2="178" y2="68" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="173,63 178,68 173,73" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 2: Stream Ingest -->
  <rect x="180" y="42" width="130" height="52" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="245" y="63" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Stream</text>
  <text x="245" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Ingest</text>
  <text x="245" y="89" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">Kafka / Kinesis</text>

  <!-- Arrow 2 -->
  <line x1="310" y1="68" x2="348" y2="68" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="343,63 348,68 343,73" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 3: Stats Engine -->
  <rect x="350" y="42" width="130" height="52" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="415" y="63" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Stats Engine</text>
  <text x="415" y="77" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">Flink / Spark</text>
  <text x="415" y="89" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">windowed metrics</text>

  <!-- Arrow 3 -->
  <line x1="480" y1="68" x2="518" y2="68" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="513,63 518,68 513,73" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 4: Drift Detector -->
  <rect x="520" y="42" width="130" height="52" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="585" y="63" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Drift</text>
  <text x="585" y="77" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Detector</text>
  <text x="585" y="89" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">PSI / KS / MMD</text>

  <!-- Arrow 4 splits up and down -->
  <line x1="650" y1="55" x2="700" y2="28" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="694,25 700,28 696,34" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <line x1="650" y1="80" x2="700" y2="107" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="694,104 700,107 696,113" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 5: Alerts -->
  <rect x="702" y="8" width="120" height="40" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="762" y="23" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Alerts</text>
  <text x="762" y="38" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">PagerDuty / Slack</text>

  <!-- Box 6: Dashboard -->
  <rect x="702" y="92" width="120" height="40" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="762" y="107" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Dashboard</text>
  <text x="762" y="122" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">Grafana / custom</text>

</svg>

---

## Components

| Component | Role |
|---|---|
| **Prediction Logger** | Sidecar or SDK call at inference time — writes `(request_id, features, prediction, timestamp)` to a durable log stream |
| **Label Joiner** | Async process that matches delayed ground-truth labels (e.g., actual churn 30 days later) back to logged predictions using `request_id` |
| **Stream Ingest (Kafka)** | Durable, replayable backbone; decouples the serving path from monitoring compute; survives spikes |
| **Stats Engine (Flink / Spark Streaming)** | Computes windowed summary statistics — mean, variance, quantiles, cardinality — per feature and per model version |
| **Reference Store** | Stores the *training distribution* (or a recent stable window) as baseline; lives in object storage (S3/GCS) or a time-series DB |
| **Drift Detector** | Compares live stats against reference using statistical tests; emits drift scores and severity flags |
| **Metadata & Model Registry** | Links model versions to their training reference distributions; needed to pick the right baseline per model |
| **Alert Engine** | Threshold + anomaly-based alerting; routes to PagerDuty, Slack, or a retraining trigger |
| **Dashboard** | Grafana or a custom UI; shows drift scores over time, per-feature heatmaps, prediction distribution shifts |
| **Retraining Trigger** | Optional — fires a training pipeline (Kubeflow / SageMaker Pipelines) when drift crosses a hard threshold |

---

## Data Flow

A single prediction's journey through the monitoring platform:

1. **Inference happens.** The model server calls the feature store, gets features, runs inference, and — via a logging SDK — publishes `{request_id, model_id, model_version, feature_vector, prediction_score, timestamp}` to a Kafka topic.

2. **Stream ingest buffers it.** Kafka retains events for 7 days. Multiple consumers can read at their own pace without affecting the serving path at all. Latency added to the user-facing request: zero (async fire-and-forget).

3. **Stats engine windows it.** A Flink job reads from Kafka, groups events into tumbling windows (e.g., 1-hour, 1-day), and computes summary stats per feature per window. These stats go into a time-series store (Prometheus + Thanos, or InfluxDB, or a columnar table in BigQuery).

4. **Drift detector compares.** On a schedule (or at end-of-window), the drift detector pulls the live stats and the reference distribution for that model version from the Reference Store. It computes:
   - **PSI** (Population Stability Index) for categorical and binned numeric features — simple, fast, widely trusted in industry
   - **KS test** (Kolmogorov-Smirnov) for continuous features — tests whether two samples come from the same distribution
   - **MMD** (Maximum Mean Discrepancy) for high-dimensional embedding drift — heavier, used for NLP/vision models
   - **Prediction drift** — shift in the output score distribution, even if individual features look fine

5. **Scores are persisted.** Each drift score tagged with `(model_id, model_version, feature_name, window_start, window_end)` lands in the metrics store. Dashboards query this.

6. **Alert engine evaluates thresholds.** PSI > 0.2 → warning. PSI > 0.25 → critical + page on-call. A separate rule watches for label distribution shift when ground truth lands.

7. **Label joins complete.** When ground truth arrives (a day, a week, or a month later), the Label Joiner matches it to the logged prediction. This computes *actual* model performance metrics (AUC, F1, precision, recall) which land back in the metrics store alongside drift scores.

8. **Retraining trigger fires (optional).** If drift has been severe for N consecutive windows, or performance has dropped below a threshold, an automated trigger calls the training pipeline with a freshened dataset.

:::why-prod
The reason you log features at inference time — not reconstruct them later — is training-serving skew. By the time you try to recompute what the model saw, the raw data may have changed (updates, deletes, privacy TTLs). Log what the model actually received. This is the canonical trick from Google's MLOps paper and every serious ML platform at scale.
:::

---

## Scaling Levers

**Throughput bottleneck: the Stats Engine**

The Flink job is stateful and proportional to the number of features × models × windows. Levers:

- **Partition Kafka by model_id** so each Flink task group handles one model — horizontal scale without cross-partition shuffles.
- **Sketch-based stats** (t-digest for quantiles, HyperLogLog for cardinality) instead of storing raw histograms. 2–5% error, 100× memory savings.
- **Sampling** at ingest for very high-QPS models — 10% sample is enough for stable drift stats at 10k RPS.
- **Push heavy tests (MMD) to batch** — run them hourly/daily out of band rather than in the hot streaming path.

**Storage bottleneck: Reference Store**

- Keep references in object storage (cheap, durable). Each model version points to its reference snapshot. Immutable blobs.
- For online comparison, cache the reference distribution in Redis — a histogram of 100 bins per feature is a few KB.

**Dashboard bottleneck**

- Pre-aggregate drift scores into daily/weekly rollups. Don't let Grafana scan raw event tables.
- Use Materialized Views in BigQuery / ClickHouse for per-model, per-feature summaries.

---

## Failure Modes

| What breaks | Symptom | Guard |
|---|---|---|
| **Kafka consumer lag grows** | Drift scores arrive hours late; alerts fire on stale data | Monitor consumer lag; autoscale Flink parallelism; alert if lag > 10 min |
| **Label joiner misses matches** | Performance metrics flatline; team thinks model is perfect forever | TTL check — if no label arrives in N days, emit a "ground truth missing" alert; investigate upstream pipeline |
| **Reference distribution is stale** | Drift detector flags real traffic as drifted because model was retrained but reference wasn't updated | Model registry must atomically update the reference pointer when a new model version is registered |
| **PSI gaming** | Feature engineer rebins a variable post-hoc; PSI drops; real drift is hidden | Freeze bin edges at training time, store them in the reference snapshot, enforce the same edges at test time |
| **Clock skew across services** | Events arrive out of order; windowed aggregations include events from wrong windows | Flink event-time processing with watermarks, not processing-time; Kafka timestamps should be producer-side |
| **Data volume spike (sale event in Pune)** | Stats engine OOMs; drift scores silent for the duration of a peak event | Sampling fallback — if buffer fills, shed load gracefully by sampling to 10%; don't drop silently |

:::gotcha
The sneakiest failure: a model gets retrained silently (automated pipeline) and the monitoring platform keeps comparing live traffic to the *old* reference. Everything looks drifted. The team panics. The fix is a model registry webhook — whenever a model version goes live, it must register its new reference distribution before traffic is switched over.
:::

---

## Cost Levers

Monitoring pipelines can silently become your second-largest data bill if you're not careful.

- **Sketch instead of raw.** A t-digest of a feature at p50/p95/p99 takes 1–2 KB. Storing raw feature values at 50k RPS for 30 days in S3 takes terabytes. Default to sketches.
- **Sample high-QPS models.** At 10,000 predictions/sec, you don't need every row to detect drift — 1% is statistically more than enough. Sample at the Kafka producer level.
- **Hot/warm/cold tiering for the metrics store.** Keep 7 days of per-window stats in Prometheus (fast queries). Archive 90 days to ClickHouse or BigQuery (cheap, analytical). Drop raw event logs after 30 days unless compliance requires more.
- **Run heavy tests (MMD, SHAP-based drift) on batch, not streaming.** Daily is fine for catching model decay that typically unfolds over weeks.
- **Consolidate models sharing a feature set** onto one drift job — if two model versions read the same features, compute feature drift once and fan it out.

---

## Tradeoffs & Alternatives

**PSI vs. KS test vs. MMD**

PSI is the workhorse of production ML monitoring — interpretable (> 0.25 = bad, everyone agrees), cheap to compute, works on binned distributions. KS test gives a proper p-value but is sensitive to sample size (large samples make everything "significant"). MMD handles high-dimensional embedding space and is statistically principled, but is expensive and harder to explain to stakeholders. Use PSI + KS for tabular, MMD for embeddings.

**Streaming vs. batch monitoring**

Streaming (Flink) gives you near-real-time drift scores (minutes). Batch (Spark job at midnight) is simpler to operate and costs less. Choose streaming when your model's error has real-time business consequences (fraud, pricing). Choose batch when drift unfolds over days anyway (churn, LTV).

**Dedicated platform vs. roll-your-own**

Managed tools like Evidently AI, WhyLabs, Arize, and Fiddler handle the drift detection, dashboards, and alerting out of the box. You trade customisability for speed of adoption. At a startup in Pune with two ML engineers, use a managed tool. At a platform team serving 20 model squads, you'll want a custom system that integrates with your feature store, model registry, and incident management.

**Concept drift vs. data drift**

Data drift: the input distribution P(X) changes. Data drift is detectable from logs alone — no labels needed. Concept drift: the relationship P(Y|X) changes. Detecting concept drift requires ground-truth labels, which arrive late. Design your platform to handle both: data drift as a leading indicator, performance metrics as the lagging ground truth.

:::key-takeaway
Tie this to your other tracks. The **feature store** is where reference distributions live (use training-time stats). The **model registry** is the coordination point for reference updates. **RAG pipelines** need drift monitoring on embedding distributions — if retrieved chunks start looking different from what the model was fine-tuned on, that's your signal. The **serving platform** provides the prediction log. Monitoring is the connective tissue across all of them.
:::

---

## How to Present This in an Interview

Keep it sharp. Say something like:

> "The platform has four concerns: log what the model saw, compute distribution statistics over time, compare those to a training-time reference, and trigger alerts or retraining. The tricky parts are making logging zero-latency for the serving path (Kafka, async), updating the reference atomically when a new model version goes live, and choosing the right drift metric — PSI for tabular features, MMD for embeddings. For scale, I'd sketch distributions rather than store raw values, sample at high QPS, and push expensive tests to batch. The biggest failure mode is a silent mismatch between the live model version and the reference it's being compared against — the fix is a hard dependency from the model registry to the monitoring config."

Then offer to drill into any layer: drift metrics, the streaming pipeline, the label joining problem, or how this integrates with the feature store and serving platform. Let them steer.

:::interview-line
"Monitoring is a first-class citizen, not an afterthought. If I can't detect when my model is wrong in production, I don't actually have a production model — I have a deployed experiment nobody is watching."
:::
