# Recommendation System — System Design Deep-Dive

Your team just launched a new short-video feed on a Swiggy-scale app. Day one, the feed is chronological. Day two, the PM Slack message arrives: *"retention is flat, we need personalisation, how soon?"* — and now you need to architect a recommendation system that serves 50 million users, runs under 150 ms p99, and doesn't accidentally recommend the same biryani clip to a vegan.

This is the canonical ML system design question. Let's build it properly.

---

## Architecture at a glance

<svg viewBox="0 0 860 160" width="100%" role="img" aria-label="Recommendation system pipeline: User Request feeds into Feature Lookup and Candidate Generation, which feeds into Ranker, then Re-ranker, then Served Feed">
  <!-- User Request -->
  <rect x="10" y="58" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="70" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">User</text>
  <text x="70" y="92" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Request</text>

  <!-- Arrow 1 -->
  <line x1="130" y1="80" x2="162" y2="80" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="158,75 163,80 158,85" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Feature Lookup -->
  <rect x="163" y="58" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="228" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Feature</text>
  <text x="228" y="92" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Store Lookup</text>

  <!-- Arrow 2 -->
  <line x1="293" y1="80" x2="325" y2="80" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="321,75 326,80 321,85" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Candidate Gen -->
  <rect x="326" y="58" width="140" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="396" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Candidate Gen</text>
  <text x="396" y="92" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(Two-Tower ANN)</text>

  <!-- Arrow 3 -->
  <line x1="466" y1="80" x2="498" y2="80" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="494,75 499,80 494,85" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Ranker -->
  <rect x="499" y="58" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="559" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Ranker</text>
  <text x="559" y="92" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(DNN / GBDT)</text>

  <!-- Arrow 4 -->
  <line x1="619" y1="80" x2="651" y2="80" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="647,75 652,80 647,85" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Re-ranker -->
  <rect x="652" y="58" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="712" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Re-ranker</text>
  <text x="712" y="92" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(Rules + MMR)</text>

  <!-- Arrow 5 -->
  <line x1="772" y1="80" x2="804" y2="80" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="800,75 805,80 800,85" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Served Feed -->
  <rect x="805" y="58" width="50" height="44" rx="10" fill="none" stroke="#8b7bff" stroke-width="1.5"/>
  <text x="830" y="76" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Feed</text>
  <text x="830" y="92" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">✓</text>

  <!-- Feature Store label below -->
  <text x="228" y="125" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">online KV store</text>
  <line x1="228" y1="102" x2="228" y2="118" stroke="#8b7bff" stroke-width="1" stroke-dasharray="4,2"/>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Event stream (Kafka / Kinesis)** | Ingests raw user interactions — clicks, watches, skips, likes — in real time |
| **Feature store (offline + online)** | Offline (BigQuery / Hive) for batch features; online (Redis / Feast) for sub-millisecond lookup at serve time |
| **Embedding service** | Converts user + item into dense vectors; refreshed on a cadence (hourly/daily) |
| **ANN index (FAISS / ScaNN / Milvus)** | Nearest-neighbour search over the item embedding space; retrieves top-K candidates fast |
| **Two-tower model** | Separate user tower and item tower; dot-product similarity drives retrieval |
| **Ranker (DNN or GBDT)** | Scores each candidate against rich cross-features (user × item interaction terms) |
| **Re-ranker** | Post-processing for diversity (MMR), business rules, legal blocks, sponsored slots |
| **Prediction cache** | Short-TTL cache (Redis) for repeat requests — same user, same context within a session |
| **A/B / experiment layer** | Routes traffic between model variants; computes online metrics per bucket |
| **Monitoring + drift service** | Watches feature distributions, CTR shifts, position-bias proxies, and label delay |

---

## Data flow

A user opens the app. Here is what happens in the next 100 ms.

1. **Auth + context extraction** — The API gateway resolves `user_id`, device type, geo, and request timestamp.

2. **Feature store lookup** — A single batch call to the online KV store fetches pre-computed user features: embedding vector, historical CTR by category, session-level recency signal. This is the hot path — it must complete in < 5 ms.

3. **Candidate generation** — The user embedding is sent to the ANN index. FAISS/ScaNN does an approximate nearest-neighbour search over the item embedding space and returns top-500 candidates. Total time: ~10–20 ms.

4. **Item feature hydration** — For those 500 candidates, item-level features (popularity, freshness, content category, creator score) are fetched from the feature store in a single pipeline call.

5. **Ranking** — A DNN (or LightGBM) scores all 500 candidates. Input = concatenation of user features + item features + cross features (e.g., user's historical CTR on this category). Output = P(click), P(watch ≥ 30 s), P(share). A weighted combination becomes the rank score.

6. **Re-ranking** — The top-50 are post-processed: diversity injection (Maximum Marginal Relevance to avoid 10 consecutive biryani clips), business rules (no banned content, ad slots, freshness boost for items < 2 hours old).

7. **Response** — Top-20 items are returned to the client with a page token for pagination. The response payload is also written to a request log for delayed label collection.

8. **Feedback loop** — User interactions (clicks, skips, dwell time) stream back through Kafka → feature store update → daily re-training job. Ground truth arrives with lag; the monitoring service accounts for this when alerting.

---

## Scaling levers

**Retrieval is the easy part to scale.** ANN indexes are read-heavy and horizontally scalable. Shard the item index by category or a hash ring. Each shard returns its local top-K; a scatter-gather merges and re-sorts.

**Ranking is the CPU-hungry part.** Batch your 500 candidates into a single model call — don't score them one at a time. Use TorchServe / Triton with dynamic batching. If you need sub-20 ms ranking: distil the ranker, prune features, or use GBDT (LightGBM) which is faster than a DNN for tabular data.

**Item embeddings don't change by the second.** Pre-compute and cache them. Push new item embeddings to the index on a 15-minute cadence for fresh content; nightly rebuild for the full catalog.

**User embeddings can go stale quickly.** For cold sessions, fall back to a session-context vector (what the user just watched in this session). Long-term user embeddings update daily.

**Feature store is the bottleneck you don't see until prod.** Right-size your Redis cluster. Use pipeline calls, not individual GETs. Keep online features lean — move anything not latency-critical to the offline store and materialise it at training time only.

:::why-prod
Two-tower + ANN is fast but approximate. The ranking model is slow but accurate. The entire design exists to make this tradeoff work: retrieve cheaply, rank carefully, serve fast.
:::

---

## Failure modes

| What breaks | Why it hurts | The guard |
|---|---|---|
| **Feature store goes down** | Ranking degrades or 500s entirely | Circuit breaker → serve a popularity-based fallback (no personalisation, still useful) |
| **Stale item index** | New content never surfaces; fresh creators get buried | Alert on index age; enforce max index staleness SLA (e.g., 30 min); canary-check new embeddings before promotion |
| **Training-serving skew** | Offline AUC looks great, online CTR tanks | Log features at serve time; replay the same feature pipeline for training; run a weekly skew check |
| **Position bias** | Model learns "things shown at position 1 get clicked" not "good things get clicked" | Counterfactual logging (log position), IPS / DLA debiasing in training |
| **Feedback loop collapse** | Popular items get recommended → get more clicks → become more popular (filter bubble) | Inject exploration: ε-greedy, UCB bandit, or a separate diversity model; track coverage and novelty as metrics |
| **Cold start (new user)** | No history → no embedding → nothing to retrieve | Onboarding interest picker → hand-crafted seed embeddings; after 3–5 interactions, warm the user embedding |
| **Cold start (new item)** | No engagement data → low confidence score → never shown | Content-based features (title, tags, creator) drive initial embedding; boost new-item exploration weight for 24 h |
| **Ranker model rollback** | Bad model deployed → CTR drops | Shadow mode → canary (5 %) → progressive rollout; automatic rollback trigger on CTR p-value vs control |

:::gotcha
Position bias is the most under-corrected failure in production rec systems. If you don't log the position of every shown item and debias during training, your ranker is quietly learning "show me what used to be shown at the top" rather than "show me what users actually like."
:::

---

## Cost levers

Recommendation systems can burn through compute fast. Here is the priority order.

1. **Cache aggressively first.** A short-TTL (30 s) prediction cache on `(user_id, context_hash)` can absorb most of the load for users refreshing the feed rapidly. Redis costs cents; re-ranking a 500-item set costs real GPU time.

2. **Reduce candidate set early.** Pre-filter by geography, language, and content policy before the ANN search. Fewer candidates in → faster ranker → lower cost.

3. **Quantise the ranker.** INT8 quantisation cuts ranker latency and memory by 2–4× with < 1 % offline AUC drop. Do this before buying more GPUs.

4. **Distil the ranker.** Train a smaller student model on the teacher ranker's logits. Often 80 % of the quality at 20 % of the cost.

5. **Two-tier serving.** Use cheap CPUs for GBDT rankers; GPUs only for deep rankers and embedding generation. Route by model type, not by default.

6. **Embed offline, not online.** Item embedding generation is expensive. Pre-compute nightly for the full catalog; only run the embedding model for genuinely new items.

---

## Tradeoffs & alternatives

**Two-tower vs. graph-based retrieval (GNN)**
Two-tower is simple, fast, and easy to debug. GNNs (PinSage, GraphSAGE) capture higher-order signals ("users who liked X also liked Y via Z") but are expensive to train and serve. Use GNNs when your engagement graph is dense and two-tower recall has plateaued.

**DNN ranker vs. GBDT (LightGBM)**
DNNs learn feature interactions automatically and scale to huge feature counts. LightGBM is faster at inference, needs less GPU, and is easier to debug. Many teams run GBDT first, add a DNN when GBDT improvements flatline.

**Collaborative filtering vs. content-based**
CF ("users like you liked this") fails on cold start. Content-based ("this video has tags you historically click") is cold-start-friendly but misses serendipity. Production systems blend both; the blend ratio is a hyperparameter tuned on online metrics.

**Batch vs. real-time feature updates**
Real-time streaming features (last 10 items watched in this session) dramatically improve relevance but add infrastructure complexity (Flink/Spark Streaming + online store write path). Start with daily batch features, add real-time only for the highest-signal session features.

**Single-objective vs. multi-objective ranking**
Optimising only P(click) creates clickbait. Production systems rank on a weighted combo of CTR, watch time, share rate, and user satisfaction signals. Weights are tuned through online experiments and human quality raters. See also: constrained optimisation (maximise engagement subject to minimum diversity constraint).

:::key-takeaway
The hardest part of a recommendation system is not the model — it's the feedback loop. Every design decision you make (what to retrieve, how to rank, where to add exploration) shapes what data you collect tomorrow, which shapes the next model you train. Design the data flywheel first, then the model.
:::

---

## How to present this in an interview

Start with this framing — say it out loud, keep it tight:

> "I'd split this into three phases: retrieval, ranking, and re-ranking. Retrieval is about speed — two-tower embeddings over an ANN index, returns a few hundred candidates in under 20 ms. Ranking is about accuracy — a DNN or GBDT that scores each candidate using rich cross-features; this is where I spend my model capacity. Re-ranking is about constraints — diversity, freshness, business rules, sponsored slots. These three stages let me decouple scale from quality.
>
> The system's hard problems are: training-serving skew (I log features at serve time), position bias (I log positions and debias), cold start (content-based fallback + exploration budget), and the feedback loop (I monitor coverage and novelty, not just CTR).
>
> Online I track CTR, watch-through rate, and a diversity metric. Offline I track recall@K at the retrieval stage and NDCG@10 at the ranking stage. If I could add one thing in production, it'd be counterfactual logging from day one — it's nearly impossible to retrofit."

Then anchor your answer to constraints: *"What's the latency SLA? What's the catalog size? Do we have implicit-only feedback or explicit ratings too?"* These questions signal that you know the design changes dramatically based on the answers.

---

**Relevant tracks:** Feature Store architecture (online vs. offline, point-in-time correctness), ML monitoring & drift (label delay, distribution shift, position-bias proxy metrics), LLM serving platform (batching patterns, Triton — same ideas apply to DNN rankers), Production RAG (retrieval-then-rerank is the same two-stage pattern).
