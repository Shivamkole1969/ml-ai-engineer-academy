# Semantic / Hybrid Search

## Scenario

Your team just shipped a product search bar for a 10-million-SKU e-commerce catalogue. Keyword search is live. Customers are typing "comfortable running shoes for rainy weather" and getting zero results — because no product has those exact words. Your PM files a P0. You need **semantic search**, probably mixed with keyword signals, by next sprint.

That's the problem. This page is the design.

---

## Architecture Overview

<svg viewBox="0 0 820 130" width="100%" role="img" aria-label="Semantic / hybrid search pipeline: Query → Embed → Vector Search + BM25 → Fusion → Re-rank → Results">
  <!-- Query -->
  <rect x="10" y="40" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="60" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">User</text>
  <text x="60" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Query</text>

  <line x1="110" y1="62" x2="145" y2="62" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Embed -->
  <rect x="145" y="40" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="200" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Embedding</text>
  <text x="200" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Service</text>

  <line x1="255" y1="62" x2="290" y2="62" stroke="#8b7bff" stroke-width="2" marker-end="url(#arr)"/>

  <!-- Vector DB -->
  <rect x="290" y="14" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="350" y="34" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Vector DB</text>
  <text x="350" y="50" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(ANN search)</text>

  <!-- BM25 -->
  <rect x="290" y="70" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="350" y="90" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">BM25 / ES</text>
  <text x="350" y="106" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(keyword)</text>

  <!-- arrows to both -->
  <line x1="255" y1="56" x2="290" y2="36" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="255" y1="68" x2="290" y2="92" stroke="#8b7bff" stroke-width="1.5"/>

  <!-- Fusion -->
  <rect x="440" y="40" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="490" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Score</text>
  <text x="490" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Fusion</text>

  <line x1="410" y1="36" x2="440" y2="56" stroke="#8b7bff" stroke-width="1.5"/>
  <line x1="410" y1="92" x2="440" y2="68" stroke="#8b7bff" stroke-width="1.5"/>

  <line x1="540" y1="62" x2="575" y2="62" stroke="#8b7bff" stroke-width="2"/>

  <!-- Re-rank -->
  <rect x="575" y="40" width="100" height="44" rx="10" fill="none" stroke="#8b7bff" stroke-width="1.5"/>
  <text x="625" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Re-ranker</text>
  <text x="625" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">(cross-enc.)</text>

  <line x1="675" y1="62" x2="710" y2="62" stroke="#8b7bff" stroke-width="2"/>

  <!-- Results -->
  <rect x="710" y="40" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="760" y="60" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Ranked</text>
  <text x="760" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Results</text>

  <defs>
    <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8b7bff"/>
    </marker>
  </defs>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Embedding Service** | Converts query (and documents at index time) into dense vectors. Typically a bi-encoder model (e.g., `text-embedding-3-small`, `bge-large`, `e5-mistral`). Latency-sensitive — keep it warm. |
| **Vector Database** | Stores doc embeddings, runs Approximate Nearest Neighbour (ANN) search (HNSW / IVF). Options: Pinecone, Qdrant, Weaviate, pgvector, OpenSearch kNN. |
| **BM25 / Inverted Index** | Classic keyword engine. Elasticsearch / OpenSearch / Solr. Excellent at exact terms, product codes, SKUs, rare proper nouns. |
| **Score Fusion Layer** | Merges the two ranked lists into one. Most common: **Reciprocal Rank Fusion (RRF)** — no need to normalise scores across two different scoring functions. |
| **Re-ranker (optional)** | Cross-encoder model reads `(query, doc)` pairs and outputs a relevance score. Expensive per-pair, so only top-K from fusion pass through (e.g., top 50 → rerank → return top 10). |
| **Feature Store** | Supplies personalisation signals (user clicks, purchase history) to the re-ranker or a downstream ML ranker. |
| **Serving API** | Orchestrates the parallel calls to vector DB + BM25, merges, reranks, and returns results within SLA (typically < 100 ms p99). |
| **Offline Indexing Pipeline** | Embeds new/updated documents, upserts to vector DB, updates inverted index. Runs as a streaming pipeline (Kafka + Flink/Spark) or batch. |
| **Monitoring / Eval** | Tracks nDCG, MRR, click-through rate, zero-result rate. Catches embedding model drift and ANN recall degradation. |

---

## Data Flow

A single search request flows like this:

1. **Query arrives** at the Serving API (`GET /search?q=comfortable+running+shoes+rainy`).
2. **Parallel fan-out:** The API fires two calls simultaneously —
   - Sends query text to the Embedding Service → gets a 768-dim (or 1536-dim) dense vector in ~10 ms.
   - Sends the same text to BM25 / Elasticsearch as a standard full-text query.
3. **Vector DB ANN search:** The dense vector is sent to the vector DB (e.g., Qdrant). It runs HNSW to return top-100 approximate nearest neighbours with cosine similarity scores, in ~5–20 ms.
4. **BM25 results arrive:** Elasticsearch returns its own top-100 hits with BM25 scores.
5. **Score Fusion:** Both ranked lists (up to 100 each) are fed into RRF. Each doc's fused score = sum of `1 / (rank_vector + k)` and `1 / (rank_bm25 + k)` where `k = 60` is a smoothing constant. Docs missing from one list get a rank penalty. Result: a single merged list of ~100 candidates.
6. **Re-ranking (if enabled):** Top-50 candidates from fusion are scored by a cross-encoder (runs inference on `(query, doc_text)` pairs). This is the accuracy multiplier — cross-encoders see both sides together and score relevance directly. Adds ~30–80 ms.
7. **Feature injection (optional):** Re-ranker or a downstream LTR (Learning to Rank) model reads from the Feature Store — user's past clicks, item popularity, recency — to personalise the final order.
8. **Top-N returned** (e.g., top 10) to the client. Total wall-clock: target < 100 ms with re-ranking, < 30 ms without.

:::key-takeaway
Parallel fan-out (step 2) is the trick that keeps latency manageable. You're not paying BM25 time + embedding time sequentially — you pay max(both).
:::

---

## Scaling Levers

**Read path (query latency)**

- **ANN index partitioning:** Shard the vector index by product category / geography so each shard is smaller and faster.
- **Quantisation:** INT8 or binary quantisation shrinks vectors 4–32x, speeds up distance computation at minor accuracy loss. Qdrant and Weaviate support this natively.
- **Embedding cache:** Same query asked 1,000 times? Cache the vector keyed on normalised query string. Redis TTL of a few minutes covers burst traffic.
- **Bi-encoder distillation:** Distil a large embedding model into a smaller one specific to your domain. Pune's catalogue is not Stack Overflow — a fine-tuned small model often beats a large generic one.
- **Async re-ranking:** Serve the fusion results immediately; stream the reranked results 50 ms later as a secondary response. Users see something fast; quality improves before they finish reading.

**Write path (indexing throughput)**

- **Streaming ingestion pipeline:** Kafka topic for product update events → consumers embed and upsert. Keeps vector index fresh within seconds of a product change.
- **Batch backfill:** For re-embedding on model upgrades, run a distributed Spark job — embed in parallel across workers, bulk-upsert in segments.
- **Dual-write with shadow index:** New embedding model's index runs in shadow mode. Compare results vs. production before cut-over.

---

## Failure Modes

| What breaks | Why it hurts | Guard |
|---|---|---|
| Embedding service is down | Entire semantic leg fails; fallback to BM25-only silently | Health check + circuit breaker; degrade gracefully to keyword-only with a metric alert |
| ANN recall degrades after index growth | HNSW `ef` parameter not tuned for new size; top-K misses relevant docs | Monitor recall@K offline with a golden query set on every index build |
| Score fusion unfairly weights one leg | Vector scores are cosine (bounded 0–1); BM25 scores are unbounded. Naive weighted sum is wrong. | Use RRF (rank-based, not score-based) or normalise each list to [0,1] before blending |
| Re-ranker latency spikes | Cross-encoder processes pairs sequentially; long docs blow up inference time | Truncate docs to 512 tokens; set a hard timeout; skip re-ranking and return fusion results |
| Stale vectors after model upgrade | Old doc embeddings + new query embeddings = space mismatch; relevance collapses | Treat model version as part of the vector index namespace; never mix versions |
| Feature Store lag | Personalisation signals are hours old after a deploy | Set TTL alerting; fall back to non-personalised ranking if staleness > threshold |

:::gotcha
The "silent degradation" failure is the nastiest one. BM25 + vector independently look fine in isolation but the fusion blending is wrong — relevance tanks and nobody notices until nDCG drops in weekly eval. Build per-leg quality metrics, not just end-to-end.
:::

---

## Cost Levers

- **Embedding API costs:** If you're calling a hosted embedding API (e.g., OpenAI), every doc upsert and every query call has a cost. Fine-tune a small open model (e.g., `bge-small-en-v1.5`) and self-host on GPU. Often 10–50x cheaper at scale.
- **Vector DB storage:** Embeddings are float32 arrays. A 1536-dim vector = 6 KB. 10 million docs = ~60 GB just for vectors. Quantise to INT8 → 15 GB. Use product quantisation (PQ) for further reduction.
- **Re-ranker GPU cost:** Cross-encoders need GPU. Run them on spot instances, batch multiple queries, or use a lighter cross-encoder (e.g., `ms-marco-MiniLM-L6-v2`) rather than a 7B model.
- **Caching:** Embedding cache + results cache (keyed on query + filters) eliminates redundant compute for head queries (the top 1% of queries drive ~40% of traffic in most systems).
- **Cold vs. hot segments:** Keep vectors for the last 90 days of active products in a hot HNSW index; archive older/discontinued products to a cheaper flat-scan or disk-based index.

---

## Tradeoffs & Alternatives

| Decision | Default | Alternative | When to switch |
|---|---|---|---|
| Dense retrieval model | Bi-encoder (fast, approximate) | Cross-encoder (accurate, slow) | Only as re-ranker, never as retriever — it doesn't scale |
| Fusion method | RRF (simple, robust) | Learned weighted sum | When you have click data and can train fusion weights per query type |
| Vector DB | Dedicated (Qdrant / Pinecone) | pgvector (Postgres extension) | pgvector is fine under ~1M docs; simpler ops; less tuning overhead |
| ANN algorithm | HNSW (fast query, high RAM) | IVF-Flat (lower RAM, slightly slower) | Memory-constrained environment |
| Keyword engine | Elasticsearch | OpenSearch, Typesense, Meilisearch | Typesense / Meilisearch are easier ops for small teams, lower cost |
| Re-ranker | Cross-encoder | ColBERT (late interaction) | ColBERT offers better quality-speed tradeoff than naive cross-encoder; worth evaluating |
| Personalisation | Feature Store + LTR | Contextual bandits | Online learning for fresh signals; more complex to deploy |

:::why-prod
In production, "hybrid" almost always beats either pure vector or pure BM25. BM25 catches exact product codes ("Nike Air Zoom Pegasus 40"), semantic search catches intent ("shoes for monsoon runs"). You need both. The ratio between them is a hyperparameter you tune with offline eval data.
:::

---

## How to Present This in an Interview

Say this, out loud, in roughly this order:

> "The core insight is that keyword search and semantic search each fail in different ways — keyword misses paraphrases, semantic misses exact terms. Hybrid search runs both in parallel and fuses the ranked lists, typically with Reciprocal Rank Fusion which is score-distribution-agnostic. The system has three main phases: offline indexing (embed docs, push to vector DB + inverted index), online retrieval (embed query, fan out to both, merge), and optional reranking (cross-encoder on top-K candidates for precision). The hardest parts are keeping the two legs' result spaces aligned after a model upgrade, and monitoring ANN recall as the index grows. The main scaling knob is quantisation for storage and latency; the main cost knob is self-hosting the embedding model. In a RAG system, this retrieval layer is exactly the R part — the quality of what you retrieve determines the ceiling of what the LLM can generate downstream."

:::interview-line
"Hybrid search is just RRF applied to two retrieval systems that fail orthogonally. The real engineering is in the monitoring — you need per-leg recall metrics, not just end-to-end nDCG, or you'll have silent degradation that takes weeks to catch."
:::

---

**Connected tracks:** RAG (retrieval is the R), Model Serving (embedding service SLA), Feature Store (personalisation signals), Monitoring & Eval (nDCG, MRR, recall@K), Data Pipelines (streaming index ingestion).
