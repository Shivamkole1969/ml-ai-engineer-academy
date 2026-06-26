# Feature Store Architecture

Your team is building a fraud detection model for a fintech product. The data science team trains it every week. The serving team exposes it via an API. But here is the problem — the training pipeline computes features one way, and the serving pipeline recomputes them differently. On training day, accuracy looks great. In production, it degrades quietly. Welcome to **training-serving skew**, the most expensive silent bug in ML.

A feature store fixes exactly this. One place to compute, store, and serve features — consistently, at scale, to both training jobs and real-time inference.

---

## Architecture Overview

<svg viewBox="0 0 820 140" width="100%" role="img" aria-label="Feature store pipeline: raw data sources flow into feature pipelines, into offline and online stores, then serve training jobs and the inference API">
  <!-- Box 1: Raw Sources -->
  <rect x="10" y="48" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="70" y="67" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Raw Sources</text>
  <text x="70" y="83" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">DB / Kafka / S3</text>

  <!-- Arrow 1 -->
  <line x1="130" y1="70" x2="168" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="163,65 168,70 163,75" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 2: Feature Pipelines -->
  <rect x="170" y="48" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="235" y="67" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Feature Pipelines</text>
  <text x="235" y="83" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">batch + streaming</text>

  <!-- Arrow 2 -->
  <line x1="300" y1="70" x2="338" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="333,65 338,70 333,75" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 3: Feature Store (central) -->
  <rect x="340" y="32" width="140" height="76" rx="12" fill="none" stroke="#3ad6ff" stroke-width="2"/>
  <text x="410" y="62" fill="#3ad6ff" font-size="12" text-anchor="middle" font-family="monospace" font-weight="bold">Feature Store</text>
  <text x="410" y="78" fill="#eaf0ff" font-size="9" text-anchor="middle" font-family="monospace">Offline Store (S3/BQ)</text>
  <text x="410" y="93" fill="#eaf0ff" font-size="9" text-anchor="middle" font-family="monospace">Online Store (Redis)</text>

  <!-- Arrow 3 to Training -->
  <line x1="480" y1="55" x2="558" y2="38" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="551,36 558,38 553,44" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Arrow 4 to Serving -->
  <line x1="480" y1="85" x2="558" y2="102" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="551,99 558,102 553,108" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 4: Training Jobs -->
  <rect x="560" y="18" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="625" y="37" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Training Jobs</text>
  <text x="625" y="53" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">point-in-time join</text>

  <!-- Box 5: Inference API -->
  <rect x="560" y="88" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="625" y="107" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Inference API</text>
  <text x="625" y="123" fill="#8b7bff" font-size="9" text-anchor="middle" font-family="monospace">&lt;10ms reads</text>

  <!-- Arrow to Models -->
  <line x1="690" y1="40" x2="728" y2="40" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="723,35 728,40 723,45" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 6: Models -->
  <rect x="730" y="18" width="80" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="770" y="44" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Models</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Feature Pipelines** | Batch (Spark/dbt) and streaming (Flink/Kafka Streams) jobs that compute feature values from raw data |
| **Offline Store** | Historical feature values — S3 + Parquet or BigQuery. Feeds training and backfill jobs |
| **Online Store** | Low-latency key-value store (Redis, DynamoDB, Cassandra). Serves real-time inference |
| **Feature Registry** | Schema catalog: feature name, type, owner, lineage, freshness SLA. Think of it as the source of truth for "what does `user_txn_count_7d` actually mean" |
| **Materialization Job** | The bridge that syncs offline → online store on a schedule or triggered by pipeline success |
| **Point-in-time Join Engine** | Reconstructs what features looked like at label time — prevents data leakage in training |
| **Feature Server** | REST/gRPC API that the inference service calls. Handles batch entity lookups efficiently |
| **Monitoring Sidecar** | Tracks feature drift, staleness, and null rates against training distributions |

---

## Data Flow

A request comes in: "Is this ₹45,000 UPI transaction from Pune fraudulent?"

1. **Ingest** — Raw events land in Kafka (transactions) and PostgreSQL (user profiles).
2. **Batch pipeline runs nightly** — Spark computes `user_txn_count_7d`, `avg_txn_amount_30d`, `merchant_risk_score`, and writes them to the offline store (S3 Parquet, partitioned by date).
3. **Streaming pipeline runs continuously** — Flink reads Kafka, computes `txn_count_last_1h` in near real-time, writes to the online store (Redis).
4. **Materialization job** — After each batch run, a job pushes the latest batch features into Redis too, so online serving has the full feature set.
5. **Inference path** — The model serving layer calls the Feature Server with entity IDs (`user_id`, `merchant_id`). Feature Server does a multi-key Redis `MGET`, returns a flat vector in under 5ms.
6. **Model scores** — The fraud model gets its feature vector and returns a probability.
7. **Training path** — A training job reads from the offline store and uses point-in-time join to snap features to their values at the moment each label was created. No leakage, consistent semantics.

:::key-takeaway
The offline and online stores are not separate systems — they share the same feature definitions from the registry. That is the whole point.
:::

---

## Scaling Levers

**Fan out reads with a read replica or cache layer**
Redis Cluster with read replicas. For very high QPS, a local L1 cache (in-process, TTL ~1s) in the serving pod reduces Redis round trips dramatically.

**Partition offline storage by entity + date**
A Hive-style partition on `(entity_id % N, date)` lets training jobs scan only the slices they need. Prune aggressively.

**Pre-join for hot entities**
If 80% of requests are for the same 10,000 merchants, precompute their feature vectors and keep them warm in a dedicated Redis keyspace.

**Materialization parallelism**
Split materialization by entity range, run in parallel. Each shard pushes its slice to Redis independently. You can saturate a Redis cluster this way, so throttle with a semaphore.

**Separate compute from storage**
Pipelines (Spark, Flink) scale independently of the stores. You don't need a bigger Redis cluster just because your Spark job is slow.

---

## Failure Modes

| What breaks | Why it hurts | Guard |
|---|---|---|
| **Stale online features** | Streaming pipeline lag → model sees old data → silent accuracy drop | Freshness SLA in registry + alerting on materialization lag |
| **Training-serving skew** | Offline and online compute diverge (e.g., timezone handling, null treatment) | Canonical feature definitions with unit-tested transform functions shared across both paths |
| **Redis OOM / eviction** | Hot key eviction at peak → feature server gets cache misses → fallback to slow path or null | `maxmemory-policy allkeys-lru` with sentinel, plus fallback reads from offline store (acceptable for non-real-time features) |
| **Point-in-time join bug** | Leaks future data into training → overfit model → terrible prod metrics | Integration tests that assert no feature appears after label timestamp |
| **Pipeline failure silent null** | Upstream table missing → feature pipeline writes nulls → model degrades | Schema validation + null-rate alerts before materialization touches online store |
| **Registry drift** | Feature definition updated without versioning → old models break | Semantic versioning on feature groups, immutable past versions |

:::gotcha
The scariest failure is silent. If your fraud model's accuracy drops from 94% to 87%, you will probably blame the data distribution shift before you check whether `user_txn_count_7d` stopped updating two days ago.
:::

---

## Cost Levers

**Redis is expensive — be selective about what goes online**
Not every feature needs sub-5ms latency. Pre-screen: does the serving SLA require it? If not, keep it offline-only and accept a slightly slower fallback path.

**Compress feature vectors**
Float32 → Float16 halves your Redis memory footprint with negligible precision loss for most tabular features.

**Tiered TTL on the online store**
Frequently changing features (last-hour aggregates) get short TTLs. Slowly changing features (user demographics) get long TTLs and are refreshed daily. Reduces write amplification.

**Parquet + columnar compression in offline store**
Snappy or ZSTD compression on Parquet files cuts S3 storage and egress costs. For very large feature tables, partition pruning can reduce Spark reads by 90%.

**Spot instances for batch pipelines**
Materialization and batch feature computation are checkpointable. Run on Spot/Preemptible. Streaming pipelines need more care — use on-demand for the Flink job managers, spot for task managers with savepoints.

---

## Tradeoffs & Alternatives

**Managed vs self-hosted**
Feast (open source) is flexible and widely used but you own ops. Tecton is the managed enterprise option — faster to production, but pricey. Vertex AI Feature Store and SageMaker Feature Store are cloud-native with less operational overhead if you're already in GCP/AWS.

**Lambda vs Kappa architecture**
Lambda (separate batch + streaming paths) is operationally heavier but gives you strong consistency for batch features. Kappa (everything through the stream) is simpler but reprocessing historical data becomes expensive. Most production feature stores use Lambda for exactly this reason.

**Push vs pull materialization**
Push: pipeline writes directly to the online store. Pull: a separate job reads from offline and materializes. Push is lower latency. Pull is simpler to reason about consistency and easier to retry. Most teams start with pull, graduate to push for latency-sensitive features.

**Feature store vs RAG retrieval store**
These look similar (both are "retrieve context for a model at inference time") but differ in structure. A feature store serves structured tabular vectors. A RAG system serves unstructured embeddings via ANN search. You can run them side by side — ML models consume feature vectors, LLM pipelines consume document chunks. Some teams are building unified retrieval layers that handle both.

:::why-prod
In production, the feature store is usually the single biggest determinant of whether your ML system is debuggable six months later. Without it, every incident becomes a forensics exercise: "what exactly did the model see at 3:47am on that transaction?"
:::

---

## How to Present This in an Interview

Set up the problem fast:

> "Without a feature store, training and serving compute features independently. That creates skew — your model sees slightly different data in prod than it trained on, and you won't know until something goes wrong. A feature store is the solution: one canonical definition, two serving paths."

Then walk the architecture:

> "Offline store handles training — historical data, point-in-time correct joins, no leakage. Online store handles inference — low-latency key-value reads, under ten milliseconds. A materialization job keeps them in sync. A feature registry gives every feature a schema, an owner, and a freshness SLA."

Then show you've thought about failure:

> "The two failure modes I care most about are staleness and skew. I put freshness SLAs in the registry and alert on lag. For skew, the fix is sharing the same transform logic across both pipelines — same code path, not just same intention."

Close with scale:

> "At high QPS, I put a local in-process cache in front of Redis — reduces tail latency and Redis load. For the offline side, Parquet + partition pruning keeps training data jobs fast and cheap."

:::interview-line
"A feature store isn't just an optimization — it's the difference between having a reproducible ML system and having a model that nobody can explain or debug in production."
:::

---

*Connects to: RAG retrieval layer (unified context serving), model monitoring (feature drift detection), training pipelines (point-in-time joins), serving infrastructure (latency SLAs), cost engineering (tiered TTL, compression).*
