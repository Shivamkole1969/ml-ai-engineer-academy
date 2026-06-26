# Real-Time Fraud Detection — System Design Deep-Dive

Your team just got paged. In the last 90 seconds, a rogue script ran 4,000 card transactions across three countries. Each transaction cleared in under 200 ms. No one flagged it in time. Leadership wants to know: *why didn't the system stop this?*

That's the forcing function for a real-time fraud detection system. Every decision in this design flows from one hard constraint: **the fraud signal must arrive before the payment network returns an auth response** — typically a 300–500 ms window.

---

## Architecture at a Glance

<svg viewBox="0 0 840 130" width="100%" role="img" aria-label="Real-time fraud detection pipeline: Event Ingestion → Feature Enrichment → ML Scoring → Decision Engine → Action + Audit">
  <!-- Box 1: Event Ingestion -->
  <rect x="10" y="38" width="140" height="48" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="80" y="59" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Event</text>
  <text x="80" y="75" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Ingestion</text>

  <!-- Arrow 1→2 -->
  <line x1="150" y1="62" x2="192" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="188,57 194,62 188,67" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 2: Feature Store -->
  <rect x="194" y="38" width="150" height="48" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="269" y="59" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Feature</text>
  <text x="269" y="75" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Enrichment</text>

  <!-- Arrow 2→3 -->
  <line x1="344" y1="62" x2="386" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="382,57 388,62 382,67" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 3: ML Scoring -->
  <rect x="388" y="38" width="140" height="48" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="458" y="59" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">ML</text>
  <text x="458" y="75" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Scoring</text>

  <!-- Arrow 3→4 -->
  <line x1="528" y1="62" x2="570" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="566,57 572,62 566,67" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 4: Decision Engine -->
  <rect x="572" y="38" width="150" height="48" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="647" y="59" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Decision</text>
  <text x="647" y="75" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Engine</text>

  <!-- Arrow 4→5 -->
  <line x1="722" y1="62" x2="764" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="760,57 766,62 760,67" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 5: Action + Audit -->
  <rect x="766" y="38" width="62" height="48" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="797" y="59" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Action</text>
  <text x="797" y="75" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">+ Audit</text>

  <!-- Timeline label -->
  <text x="420" y="118" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">← all of this happens in &lt; 150 ms →</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **API Gateway / gRPC endpoint** | Receives the transaction event from the payment processor; validates schema; forwards to the pipeline |
| **Kafka / Pub-Sub bus** | Decouples ingestion from processing; gives replay capability for backtesting models on past events |
| **Feature Enrichment service** | Joins the raw event with pre-computed features from the Feature Store (velocity counts, historical spend patterns, device fingerprints) |
| **Online Feature Store** (Redis / DynamoDB) | Holds low-latency, pre-aggregated features — e.g., "transactions in last 1 min for this card" — refreshed by a stream processor |
| **Stream Processor** (Flink / Spark Streaming) | Maintains real-time aggregations (sliding windows) and pushes them into the Feature Store; also writes to the offline store for model training |
| **ML Scoring service** | Runs the fraud model (gradient-boosted tree or a lightweight neural net); served via TorchServe / BentoML / SageMaker Endpoint; returns a risk score 0–1 |
| **Rules Engine** | Hard-coded guardrails that fire *before* or *alongside* the ML score — "block if 10+ transactions in 60 s", "block if IP is on deny-list" |
| **Decision Engine** | Combines rule outcomes + ML score → final decision (APPROVE / DECLINE / STEP-UP challenge); configurable thresholds per merchant category |
| **Action layer** | Publishes decision back to the payment processor; triggers alerts (SMS, push); writes case to the fraud ops queue if Step-Up |
| **Audit log / Data Lake** | Append-only store of every event, feature vector, score, and decision — feeds model retraining and regulatory reporting |
| **Model Registry** | Versioned storage of trained models; supports canary rollout (route 5% of traffic to new model before full cutover) |
| **Monitoring / Alerting** | Tracks precision/recall drift, feature drift, p99 latency, false-positive rate; pages on-call if thresholds breach |

---

## Data Flow

Here is a single transaction walking through the system, start to finish.

1. **Customer taps to pay** at a Pune mall. The POS terminal sends an auth request to the payment processor (Visa/RuPay network).

2. **Payment processor calls your fraud API** (gRPC or REST) with the transaction payload: `card_id`, `merchant_id`, `amount`, `device_fingerprint`, `geo`, `timestamp`.

3. **API Gateway** validates the schema, strips PII to a pseudonymous key, and publishes the event onto **Kafka topic `txn-raw`**.

4. **Feature Enrichment service** consumes the event. For each transaction it looks up the online Feature Store and joins:
   - Velocity features: `txn_count_1m`, `txn_count_10m`, `total_spend_1h`
   - Profile features: `avg_ticket_size_30d`, `home_geo`, `usual_device_id`
   - Network features: IP reputation score, BIN (Bank Identification Number) risk tier

5. **Enriched event** goes to the **Rules Engine** first (sub-millisecond). Hard rules fire instantly:
   - Card on block-list → DECLINE immediately, skip ML.
   - Amount > 3× 30-day average AND foreign merchant → force STEP-UP.

6. If no hard block, the enriched feature vector hits the **ML Scoring service** (target: < 80 ms including network). The model returns `fraud_score = 0.91`.

7. **Decision Engine** applies threshold logic:
   - `score < 0.3` → APPROVE
   - `0.3 ≤ score < 0.7` → APPROVE + soft flag
   - `0.7 ≤ score < 0.9` → STEP-UP (OTP challenge)
   - `score ≥ 0.9` → DECLINE

8. **Action layer** returns the decision to the payment processor in the auth response. Total time: ~120–180 ms.

9. Simultaneously, the event + feature vector + score + decision is written to:
   - **Kafka `txn-decisions`** topic (for the stream processor to update velocity counters)
   - **Audit log** (append-only, S3 or BigQuery)
   - **Fraud ops queue** if flagged for manual review

10. **Stream processor** updates sliding-window aggregates in the Online Feature Store so the *next* transaction on this card already sees the updated `txn_count_1m`.

---

## Scaling Levers

**Horizontal partitioning** — Kafka topics partitioned by `card_id` hash. Feature Store lookups and ML scoring are stateless; scale them independently behind a load balancer. Stream processor parallelism matches partition count.

**Feature precomputation** — Don't compute aggregations in the hot path. Flink maintains them asynchronously and keeps them ready in Redis. The hot-path fetch is a point read: O(1), ~1 ms.

**Model compression** — Quantize or distill the fraud model. A XGBoost model with 500 trees scores in < 5 ms. A deep network may need ONNX + INT8 quantization to stay under 20 ms on CPU.

**Tiered caching** — In-process LRU cache for the most active 10K cards (covers ~40% of traffic at a large issuer). Cuts Feature Store round-trips by half.

**Read replicas for Feature Store** — Redis in cluster mode with read replicas per AZ. Writes go to primary; scoring reads any replica.

**Load shedding** — If the system is overloaded (burst during a flash sale), degrade gracefully: skip the ML call and fall back to rules-only scoring. Always prefer a slightly worse fraud decision over timing out the payment network.

---

## Failure Modes

| What breaks | Impact | Guard |
|---|---|---|
| Feature Store is down | ML features are missing → model gets zero/default values → precision drops | Fallback to rules-only mode; alert immediately; Feature Store on multi-AZ with circuit breaker |
| ML Scoring service times out | Transaction hangs past the auth window | Hard deadline of 150 ms; on timeout, fall back to rules engine; never let the payment processor wait |
| Kafka lag spikes | Velocity counters go stale; fraudsters get a window | Alert on consumer lag > 5 s; auto-scale Flink workers; tune retention to allow replay |
| Model goes stale (concept drift) | Fraud patterns shift (new attack vector); precision degrades | Champion/challenger setup — continuously evaluate a newly trained model; daily precision/recall dashboard; retrain pipeline triggered on drift detection |
| False-positive storm | Legitimate customers declined; chargeback rates drop but so do conversions | Monitor false-positive rate per merchant category; threshold can be tuned per segment without redeploying the model |
| Data poisoning | Fraudsters learn the model's decision boundary via probing | Rate-limit API callers; randomize scores slightly before returning; don't expose raw scores in response — only APPROVE/DECLINE/STEP-UP |

:::gotcha
The silent killer is **label delay**. Chargebacks arrive 30–90 days after the fraud event. If you retrain on only confirmed labels, your training set is always lagging reality. Use a combination of: immediate "obvious fraud" labels (card stolen report, chargeback filed quickly) + heuristic labels for borderline cases + active learning from analyst reviews.
:::

---

## Cost Levers

**ML inference cost** — The scoring service is the most expensive compute. Options:
- Batch micro-decisions: group events in 10 ms windows and run one batch inference call. Works for offline-adjacent flows; risky for real-time.
- Use a cheaper "fast" model (XGBoost) as first pass; escalate to a more expensive model only for borderline scores (0.4–0.7 band). ~60% of traffic never needs the expensive model.

**Feature Store cost** — Redis is expensive per GB. Age out features older than the model's lookback window. A 90-day lookback doesn't need millisecond latency; move cold features to DynamoDB or Cassandra.

**Kafka retention** — Fraud events carry PII derivatives. Don't over-retain. 7 days is usually enough for replay; archive to cold storage after that.

**Model retraining cadence** — Daily retraining is usually enough unless you're under an active attack. Weekly model + daily threshold tuning is a good split between compute cost and freshness.

**Spot / Preemptible instances** — Use spot for the stream processor and batch training. Keep the scoring service and Feature Store on on-demand/reserved for SLA predictability.

---

## Tradeoffs & Alternatives

**Rules only vs. ML scoring**

Rules are fast, interpretable, and easy to audit. They're terrible at generalising to new attack patterns. ML catches subtle correlated signals but needs labeled data and drifts over time. In production you want *both*: rules as the first wall, ML as the second.

**Synchronous vs. asynchronous scoring**

Synchronous (inline in the auth flow) gives you a hard block before money moves. Asynchronous (score after approval, trigger chargeback or card freeze) is simpler to build but lets fraud through. For card-not-present transactions, async is sometimes acceptable. For card-present, synchronous is the standard.

**Graph features**

If your fraud is network-fraud (rings of mule accounts), a Graph Neural Network (GNN) over the transaction graph can catch rings that tabular models miss. Cost: graph store (Neo4j or Neptune), more complex feature engineering, higher inference latency. Usually a Phase 2 investment.

**Vector similarity for device fingerprints**

Store device fingerprint embeddings in a vector database (Pinecone, pgvector). At scoring time, do an ANN lookup: "is this device similar to any known-fraud device?" Adds ~20 ms but catches synthetic fingerprint variations.

**Stream vs. Lambda architecture for features**

Lambda (batch + speed layer) is the classic. In 2025, most teams go full streaming (Flink or Kafka Streams for all aggregations, no batch layer). Fewer moving parts, one code path. The tradeoff: streaming aggregations are eventually consistent; batch is exact. For fraud, eventual consistency is fine — a 2 s lag on a velocity counter is irrelevant when you're blocking a 3 s burst attack.

:::why-prod
Feature Stores are not optional once you go beyond toy scale. Without one, every team recomputes the same aggregations differently — training pipeline uses one definition of "transactions in last hour", serving uses another. You end up with **training-serving skew**, which is one of the hardest bugs to find in production: the model looks great offline and mediocre live.
:::

---

## How to Present This in an Interview

Start here and say this out loud:

> "The core constraint is latency: fraud detection must fit inside the payment network's auth window — roughly 300 to 500 milliseconds. That single constraint drives every architectural choice. I'd split the system into four layers: ingestion, feature enrichment, ML scoring, and the decision engine. The hot path is synchronous and must complete in under 150 ms. Everything else — retraining, audit logging, analyst queues — is asynchronous and can tolerate higher latency."

Then walk the interviewer through a single transaction end to end (step 1 through 10 above). After that, anchor the discussion on the three hardest problems:

1. **Latency** — Feature Store, model compression, circuit breakers.
2. **Label quality** — Chargeback delay, heuristic labelling, active learning.
3. **Drift** — Champion/challenger rollout, monitoring, threshold tuning without redeployment.

If they ask "how would you scale this to 100K transactions per second?", partition Kafka by `card_id`, scale scoring stateless horizontally, and lean on the Feature Store's in-process cache for the hot-tail traffic.

:::interview-line
"The rules engine is not a legacy system — it's a first-class citizen. It handles the easy 20% of cases instantaneously and gives you a safety net when the ML model is misconfigured or under attack."
:::

:::key-takeaway
Real-time fraud detection is a latency-constrained, label-delayed, concept-drifting ML system. The engineering challenge is not the model — it's keeping features fresh, decisions fast, and training labels honest. Get those three right and the model almost takes care of itself.
:::

---

*Relevant tracks: Feature Stores (Track 04), Model Serving & Latency (Track 06), Evaluation & Label Quality (Track 05), RAG & Retrieval patterns (Track 03), Monitoring & Drift Detection (Track 07).*
