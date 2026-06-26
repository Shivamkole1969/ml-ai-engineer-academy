# Production RAG System

Your team just shipped a support chatbot. It answers fine for a week, then a customer asks about a product you updated last Tuesday — and the bot confidently gives the old answer. The model's weights don't know about Tuesday. That's the moment you need a **Retrieval-Augmented Generation (RAG)** system in production: a live knowledge layer that sits between your documents and the LLM, so the model always reasons over *current* facts, not frozen training data.

---

## Architecture Overview

<svg viewBox="0 0 860 140" width="100%" role="img" aria-label="Production RAG pipeline: Document Store → Chunker/Embedder → Vector DB → Retriever → Reranker → LLM → Response">
  <!-- Box 1: Document Store -->
  <rect x="10" y="48" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="65" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Document</text>
  <text x="65" y="82" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Store</text>

  <line x1="120" y1="70" x2="148" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="142,64 150,70 142,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 2: Chunker/Embedder -->
  <rect x="150" y="48" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="210" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Chunker /</text>
  <text x="210" y="82" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Embedder</text>

  <line x1="270" y1="70" x2="298" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="292,64 300,70 292,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 3: Vector DB -->
  <rect x="300" y="48" width="110" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="355" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Vector DB</text>
  <text x="355" y="82" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(index)</text>

  <line x1="410" y1="70" x2="438" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="432,64 440,70 432,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 4: Retriever -->
  <rect x="440" y="48" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="490" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Retriever</text>
  <text x="490" y="82" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(ANN search)</text>

  <line x1="540" y1="70" x2="568" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="562,64 570,70 562,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 5: Reranker -->
  <rect x="570" y="48" width="100" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="620" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Reranker</text>
  <text x="620" y="82" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(cross-enc.)</text>

  <line x1="670" y1="70" x2="698" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="692,64 700,70 692,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 6: LLM -->
  <rect x="700" y="48" width="80" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="740" y="68" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">LLM</text>
  <text x="740" y="82" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">(generate)</text>

  <line x1="780" y1="70" x2="808" y2="70" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="802,64 810,70 802,76" fill="none" stroke="#8b7bff" stroke-width="2"/>

  <!-- Box 7: Response -->
  <rect x="810" y="48" width="40" height="44" rx="10" fill="none" stroke="#8b7bff" stroke-width="1.5"/>
  <text x="830" y="68" fill="#eaf0ff" font-size="10" text-anchor="middle" font-family="monospace">resp</text>
  <text x="830" y="82" fill="#eaf0ff" font-size="10" text-anchor="middle" font-family="monospace">onse</text>

  <!-- Ingestion label -->
  <text x="210" y="118" fill="#8b7bff" font-size="10" text-anchor="middle" font-family="monospace">── ingestion path ──</text>
  <!-- Query label -->
  <text x="600" y="118" fill="#3ad6ff" font-size="10" text-anchor="middle" font-family="monospace">── query path ──</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Document Store** | Raw source of truth — S3, Google Drive, Confluence, database tables. Everything your users care about lives here. |
| **Chunker** | Splits documents into retrieval-sized pieces (typically 256–512 tokens). Bad chunking = bad retrieval — this is the most underrated step. |
| **Embedding Model** | Converts each chunk into a dense vector that captures semantic meaning. Same model must be used at query time. |
| **Vector Database** | Stores chunk vectors with metadata. Supports Approximate Nearest Neighbour (ANN) search at scale. Popular choices: Pinecone, Weaviate, Qdrant, pgvector. |
| **Retriever** | At query time, embeds the user question and fetches the top-K nearest chunks from the vector DB. |
| **Reranker** | A lighter cross-encoder model that rescores the top-K chunks for precision. Cuts noise before the LLM sees context. |
| **Prompt Builder** | Assembles system prompt + retrieved chunks + user query into the final LLM input. Enforces context-window limits. |
| **LLM** | Reads the assembled prompt and generates the final answer. It only *reasons* — it doesn't remember facts on its own. |
| **Response Cache** | Caches LLM responses keyed on (query hash + chunk set). Eliminates redundant LLM calls for repeated questions. |
| **Observability Layer** | Logs retrieval scores, latency, chunk IDs used, LLM token counts. Powers feedback loops for tuning. |

---

## Data Flow

Two separate paths: **ingestion** (write path) and **retrieval** (read path).

### Ingestion path (runs offline or near-real-time)

1. A document lands in the Document Store (new support article, updated product spec, etc.).
2. An ingestion worker pulls the raw text.
3. The **Chunker** splits it into overlapping windows — typically 256-token chunks with a 64-token overlap so context doesn't hard-cut across sentences.
4. Each chunk is passed through the **Embedding Model** (e.g., `text-embedding-3-small`, `bge-large`, or your own fine-tuned model).
5. The resulting vector is upserted into the **Vector DB** with metadata: `doc_id`, `chunk_index`, `source_url`, `last_updated`.
6. Old chunks for the same document are deleted or soft-expired to prevent stale data leaking.

### Query path (real-time, user-facing)

1. User sends a question: *"What's the refund policy for Pro subscriptions as of this month?"*
2. The same **Embedding Model** encodes the question into a query vector.
3. The **Vector DB** runs ANN search (HNSW or IVF-based) and returns top-20 chunk candidates with scores.
4. The **Reranker** cross-encodes the question against each candidate and reorders by semantic relevance — you typically keep top-3 to top-5 chunks.
5. The **Prompt Builder** assembles: system role + retrieved chunks (with citations) + user question. It enforces the context window, trimming or compressing if needed.
6. The **LLM** generates a grounded answer, ideally citing the chunk source.
7. Response is streamed back to the user and logged to the Observability Layer.
8. If a **Response Cache** hit exists for the same (question, chunk set), step 6 is skipped entirely.

---

## Scaling Levers

**Read path (latency is king)**

- **ANN index type**: HNSW for low latency at moderate scale; IVF-PQ for billion-vector datasets where memory is tight.
- **Embedding cache**: Cache query embeddings for popular questions — the embed call is ~5–10 ms but adds up at scale.
- **Read replicas**: Vector DB read replicas for high query-per-second (QPS) workloads. Most managed offerings handle this natively.
- **Reranker as an async tier**: For non-latency-critical use cases, reranking can be moved off the critical path and results prefetched.

**Ingestion path (throughput is king)**

- **Parallel chunking workers**: Fan out ingestion across a Celery/Kafka consumer group. Each worker handles one document.
- **Batch upserts**: Vector DBs prefer bulk inserts. Batch 100–500 vectors per API call rather than one at a time.
- **Incremental updates**: Use document fingerprints (SHA-256 of content) to skip re-embedding unchanged documents.

**LLM tier**

- **Prompt caching**: If your LLM provider supports prefix caching (Claude does), put your large system prompt in the cached prefix. You pay only for new tokens on each turn.
- **Model tiering**: Route simple factual queries to a smaller/cheaper model; complex multi-doc synthesis to a larger one.

---

## Failure Modes

| What breaks | Why it happens | Guard |
|---|---|---|
| **Stale chunks in index** | Document updated in the source but ingestion lagged | Track `last_updated` metadata; expose freshness in retrieval scores; set TTL on chunks |
| **Embedding drift** | Embedding model swapped/updated — old vectors no longer align with new query vectors | Version your embedding model in chunk metadata; re-embed entire corpus on model change |
| **Retrieval misses (low recall)** | Query semantics don't match chunk language (technical jargon, abbreviations) | Add a keyword/BM25 hybrid retrieval layer alongside vector search; expand query with synonyms |
| **Context stuffing** | Too many chunks passed to LLM → it ignores middle content (lost-in-the-middle problem) | Limit to 3–5 high-score chunks; reranker helps with this |
| **Hallucination despite retrieval** | LLM ignores retrieved context and makes up an answer | Ground the prompt ("Only use the provided context. If unsure, say so."); evaluate with LLM-as-judge |
| **Vector DB latency spikes** | Index fragmentation at write-heavy periods | Separate ingestion and query clusters; trigger reindex during low-traffic windows |
| **Prompt injection via documents** | Malicious content in a document tricks the LLM | Sanitize chunk text before insertion; use a system-prompt guardrail that ignores instructions in context |

---

## Cost Levers

Production RAG has three main cost lines — embedding, vector DB, and LLM. Each one is tunable.

**Embedding costs**

- Use a smaller embedding model where quality allows. `text-embedding-3-small` is ~5x cheaper than `text-embedding-3-large` with minimal real-world quality drop for most corpora.
- Cache embeddings aggressively. If the same document chunk appears multiple times (FAQs, boilerplate), you embed once and reuse.

**Vector DB costs**

- Choose the right index: IVF-PQ compresses vectors to ~4–8 bytes (vs 1536 floats × 4 bytes = 6 KB). For 10M chunks, that's ~40 MB vs ~60 GB — the difference between a $50/month instance and a $3,000/month one.
- Archive old or low-value documents to cold storage; only hot knowledge stays in the live index.

**LLM costs**

- Fewer, shorter chunks = smaller prompts = fewer input tokens.
- Response caching: popular questions get answered from cache, zero LLM tokens consumed.
- Prompt prefix caching (if supported): system prompt + static instructions are cached; you only pay for the chunk + question tokens per request.
- Model tiering: a 7B open-source model self-hosted on a single A10G GPU can answer 80% of support queries. Reserve the frontier model for the hard 20%.

---

## Tradeoffs & Alternatives

**Chunk size: small vs large**

Small chunks (128 tokens) give precise retrieval but lose surrounding context. Large chunks (1024 tokens) preserve context but dilute the similarity signal. The sweet spot for most domains is 256–512 tokens with overlap. Alternatively, use *parent-child chunking*: retrieve on small child chunks, return the parent chunk as context.

**Dense-only vs hybrid retrieval**

Pure vector search is great for semantic similarity but fails on exact keyword matches (product codes, names, version numbers). BM25 is the opposite — exact matches, poor semantic generalisation. **Hybrid retrieval** (dense + sparse, fused with RRF or a learned combiner) is the production default for any serious deployment.

**Managed vs self-hosted vector DB**

Pinecone, Weaviate Cloud, or Qdrant Cloud give you zero-ops at a cost premium. pgvector on RDS is cheap and familiar but tops out around 1–5M vectors before query latency degrades. Self-hosted Qdrant or Weaviate on Kubernetes is the middle path: more ops work, much lower per-query cost at scale.

**RAG vs fine-tuning**

Fine-tuning bakes knowledge into model weights — fast inference, no retrieval latency, but stale the moment your docs change. RAG keeps knowledge external — always fresh, but adds ~100–300 ms latency and costs per retrieved call. Most production teams do both: fine-tune for *style and task format*, RAG for *current knowledge*.

**Agentic RAG**

For multi-hop questions ("Who approved the refund policy that was updated in Q1?"), a single retrieval pass is insufficient. Agentic RAG lets the LLM decide when to retrieve, what to retrieve next, and how to combine results across multiple calls. More powerful, significantly more expensive and harder to debug. Use it only when single-pass RAG demonstrably fails.

---

## How to Present This in an Interview

:::interview-line
"A production RAG system is fundamentally two pipelines with a shared index in the middle. The **ingestion pipeline** chunks documents, embeds them, and writes vectors into a Vector DB. The **query pipeline** embeds the user question, does ANN retrieval, reranks with a cross-encoder, builds a grounded prompt, and passes it to the LLM. The tricky production problems are: keeping the index fresh when documents update, handling embedding model drift across versions, and preventing the LLM from hallucinating despite having good retrieved context. I'd add hybrid retrieval for keyword-heavy domains, a reranker to cut retrieval noise, and a response cache to manage LLM costs. Observability on retrieval scores and chunk usage is non-negotiable — that's how you know when retrieval is quietly breaking."
:::

**What interviewers want to hear beyond the basics:**

- You know chunking strategy matters (not just "split into chunks")
- You understand embedding model versioning is a production concern
- You've thought about stale data, not just retrieval quality
- You can speak to cost: embedding cache, prompt caching, model tiering
- You know when RAG isn't enough (multi-hop → agentic, static knowledge → fine-tune)

:::key-takeaway
The LLM is the tip of the iceberg. The real engineering in a production RAG system is in the ingestion pipeline, index freshness, hybrid retrieval, and observability. Getting those right is what separates a demo from a system your Pune support team can rely on at 2 AM.
:::

---

*Relevant tracks: RAG & Retrieval · LLM Serving · Feature Stores · Embedding Models · Observability & Monitoring · Cost Optimisation*
