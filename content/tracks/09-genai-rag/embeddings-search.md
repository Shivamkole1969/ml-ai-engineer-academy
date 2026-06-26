---
id: embeddings-search
track: 09-genai-rag
title: "Embeddings & Semantic Search (Cosine Sim, ANN)"
badge: HOT
minutes: 10
prereqs: []
tags: [embeddings, semantic-search, cosine-similarity, ANN, HNSW, vector-search, GenAI]
xp: 60
hot2026: true
---

It's 11 PM. Your product manager just Slacked: "The search on our internal knowledge base is garbage — users type 'remote work policy' and get zero results because the doc says 'work from home guidelines'." Classic keyword-mismatch problem. Your Elasticsearch is doing exact word overlap. What you actually need is search that understands *meaning*, not just letters. That's semantic search — and it starts with embeddings.

## What is an embedding?

An embedding is a list of numbers (a vector) that represents the *meaning* of a piece of text.

Sounds abstract — here's the intuition. "cat" and "kitten" are completely different strings, but a well-trained model knows they're semantically close. It encodes that closeness as vectors that point in nearly the same direction in high-dimensional space.

A text embedding model takes your sentence → runs it through a neural network → outputs a fixed-length vector (say, 768 or 1536 floats). Every sentence gets its own point in this space. Sentences with similar meaning cluster together. That's the whole idea.

Popular open-source models: `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-dim), `BAAI/bge-m3` (multilingual, strong for Hindi+English), `intfloat/e5-large-v2`. For API use: OpenAI `text-embedding-3-small` is cheap and excellent.

:::why-prod
In production RAG systems, embedding quality is the single biggest lever on retrieval accuracy — before chunking, before reranking, before prompt engineering. Bad embeddings mean your LLM never even sees the right context.
:::

## Cosine similarity — how "closeness" is measured

You have two vectors. You want to know how similar they are. The most common metric: **cosine similarity**.

Cosine sim ignores vector magnitude and looks only at the *angle* between them. Two parallel vectors (angle = 0°) → similarity = 1.0. Perpendicular → 0. Pointing opposite ways → −1.

Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`

Translation: dot product of the two vectors, divided by the product of their lengths. For unit-normalized vectors (which embedding models usually output), this simplifies to just the dot product.

:::table {title="Similarity scores — what they mean"}
| Score | Meaning | Example |
|---|---|---|
| 0.95 – 1.0 | Near-identical meaning | "WFH policy" vs "work from home policy" |
| 0.80 – 0.94 | Same topic, different angle | "vacation policy" vs "leave of absence rules" |
| 0.60 – 0.79 | Loosely related | "HR docs" vs "employee handbook" |
| < 0.60 | Probably unrelated | "dog food recipes" vs "quarterly OKRs" |
:::

## The brute-force problem — why ANN exists

You have 500,000 documents in your knowledge base. A user searches. Brute-force cosine similarity means computing 500K dot products on every query. At 100 QPS, that's 50 million dot products per second. It works in a notebook. It melts in production.

Enter **Approximate Nearest Neighbor (ANN)** search. You trade a tiny bit of recall for massive speed gains. The most popular algorithm is **HNSW** (Hierarchical Navigable Small World graph) — it builds a multi-layer graph where each node links to its nearby neighbors. Queries navigate the graph instead of scanning every point.

Result: search over millions of vectors in milliseconds, not seconds. Libraries like FAISS and Chroma use HNSW under the hood. (Vector databases are covered in the next lesson — here we focus on the fundamentals.)

```python {title="Semantic search from scratch — cosine sim + ANN" run=false}
# pip install sentence-transformers faiss-cpu numpy
# Free to run locally on CPU — no GPU needed for this scale

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. Load a lightweight open-source embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 384-dimensional vectors, ~80MB download, runs fine on a laptop

# 2. A tiny corpus — your "knowledge base"
docs = [
    "Work from home guidelines and remote work policy.",
    "Office attendance requirements and hybrid schedule.",
    "How to apply for paid leave and vacation days.",
    "Expense reimbursement process for business travel.",
    "Health insurance benefits and claims procedure.",
]

# 3. Embed the corpus
doc_embeddings = model.encode(docs, normalize_embeddings=True)
# normalize_embeddings=True → unit vectors → dot product == cosine sim

# 4. Build a FAISS flat index (exact search — fine for small corpora)
dim = doc_embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dim)  # IP = Inner Product (≡ cosine sim for unit vecs)
index.add(doc_embeddings.astype(np.float32))

# 5. Query — note: DIFFERENT words, same meaning as doc[0]
query = "remote work policy"
query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

top_k = 3
distances, indices = index.search(query_vec, top_k)

print("Top results:")
for dist, idx in zip(distances[0], indices[0]):
    print(f"  [{dist:.3f}] {docs[idx]}")

# Expected output:
#   [0.712] Work from home guidelines and remote work policy.
#   [0.581] Office attendance requirements and hybrid schedule.
#   [0.423] How to apply for paid leave and vacation days.

# Scale tip: swap IndexFlatIP → IndexHNSWFlat for ANN on large corpora
# hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # 32 = number of neighbors per node
# hnsw_index.add(doc_embeddings.astype(np.float32))
```

:::gotcha
Do NOT mix embedding models across your pipeline. If you embedded your docs with `all-MiniLM-L6-v2`, you must embed queries with the exact same model. Different models live in completely different vector spaces — comparing their outputs is comparing apples to GPS coordinates.
:::

:::war-story {title="The multilingual mismatch that killed recall"}
A Pune startup built a support bot for their Hindi-English customer base. Recall was 30% — terrible. The root cause: they embedded docs with an English-only model (`all-MiniLM-L6-v2`) but users queried in Hinglish ("mera order kab aayega?"). English model → zero semantic overlap with Hindi tokens. Swapping to `BAAI/bge-m3` (genuinely multilingual) pushed recall to 78% in two hours. Model selection matters more than index tuning.
:::

:::interview-line
"Cosine similarity measures the angle between two embedding vectors — magnitude doesn't matter, only direction. For unit-normalized embeddings it's just a dot product, which is why FAISS's IndexFlatIP gives you cosine search for free."
:::

:::qa {q="Why cosine similarity instead of Euclidean distance for text embeddings?"}
Cosine similarity is magnitude-invariant — a short tweet and a long article about the same topic point in the same direction, even though the article's vector might have larger values. Euclidean distance would penalize that length difference, hurting recall for short-vs-long text comparisons. Most embedding models are also trained with cosine similarity as the objective, so it's the natural metric.
:::

:::qa {q="What's the real cost of brute-force vector search in production?"}
Brute force is O(N × D) per query — N documents, D dimensions. At a million docs and 1536 dimensions, that's 1.5 billion float multiplications per query. ANN (HNSW) reduces this to roughly O(log N) graph traversals, typically 10–100ms vs. seconds. The trade-off is ~1–5% recall loss, which is almost always worth it.
:::

:::qa {q="How do you choose an embedding model for a new RAG project?"}
Start with the MTEB leaderboard — it benchmarks models on retrieval tasks. For pure English: `text-embedding-3-small` (cheap API) or `bge-large-en-v1.5` (free, self-hosted). For Hindi/Hinglish or multilingual: `bge-m3`. For latency-sensitive on-device use: `all-MiniLM-L6-v2`. Always run a quick eval on your own data before committing — MTEB ranks are averages, your domain may differ.
:::

:::drill {type="mcq" q="You have unit-normalized embedding vectors. Which FAISS index type gives you cosine similarity search?"}
- [ ] IndexFlatL2
- [x] IndexFlatIP
- [ ] IndexPQ
- [ ] IndexIVFFlat
:::

:::drill {type="mcq" q="A teammate embeds documents with model A and queries with model B (different architecture). What happens?"}
- [ ] Search still works because both models output 768-dim vectors
- [ ] Results are slightly worse but usable
- [x] Results are meaningless — the vectors live in incompatible spaces
- [ ] FAISS throws an error automatically
:::

:::drill {type="mcq" q="HNSW trades _____ for _____ compared to brute-force search."}
- [ ] Speed for zero recall loss
- [ ] Memory for speed
- [x] A small amount of recall for dramatically faster query time
- [ ] Model accuracy for index size
:::

:::key-takeaway
Embeddings turn text meaning into geometry — cosine similarity measures how aligned two meaning-vectors are, and ANN algorithms like HNSW let you find the closest ones across millions of docs in milliseconds. Get the embedding model right first; everything else (indexes, reranking, chunking) is optimization on top.
:::
