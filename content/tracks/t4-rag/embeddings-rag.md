---
id: embeddings-rag
track: t4-rag
title: "Embeddings & similarity: cosine vs dot, Matryoshka"
badge: HOT
minutes: 9
prereqs: []
tags: [embeddings, cosine-similarity, dot-product, matryoshka, rag, vector-search, sentence-transformers]
xp: 60
hot2026: true
---

Imagine your RAG chatbot has been live for two weeks and users love it — until one day someone asks "What's the refund policy for digital goods?" The system confidently returns chunks about *physical* product returns. Wrong answer, very correct-looking retrieval score. You dig in and realize the embedding model treats "digital" and "physical" as nearly identical because both live in "purchase / return / policy" semantic space. The retrieval was *mathematically* fine. The similarity metric was wrong for the job. Understanding what embeddings actually measure — and how to tune that — is the difference between a RAG system that impresses and one that embarrasses.

## What is an embedding, really?

An embedding is a list of numbers — a vector — that encodes the *meaning* of a piece of text. A sentence-transformer model reads "refund policy for digital goods" and outputs, say, 768 floats. The geometry of that space is trained so that semantically similar texts end up *close* to each other.

Think of it like placing cities on a map. London and Paris end up near each other. Tokyo is far away. The model learns the "map" from billions of examples.

When you embed both your query and your stored documents, retrieval becomes: **find the documents whose vectors are closest to the query vector.** The question is — how do you measure "closest"?

## Cosine similarity vs dot product

These are the two workhorses of vector search, and mixing them up is a silent killer.

**Cosine similarity** measures the *angle* between two vectors. It completely ignores magnitude. Formally:

```
cosine(A, B) = (A · B) / (|A| × |B|)
```

Result is between −1 and 1. Two vectors pointing in the same direction score 1.0, regardless of how long they are.

**Dot product** (inner product) measures both angle *and* magnitude:

```
dot(A, B) = A · B = Σ(Aᵢ × Bᵢ)
```

A large vector pointing in roughly the right direction scores high. A small vector pointing in the perfect direction scores lower.

:::why-prod
Your embedding model's training objective determines which metric you should use. Using the wrong one degrades retrieval precision silently — no errors, just bad results. `text-embedding-3-small` (OpenAI) and `all-MiniLM-L6-v2` are trained for cosine similarity. FAISS's inner-product index assumes dot product. Mismatch = garbage-in-garbage-out at retrieval time.
:::

:::table {title="Cosine vs Dot Product — quick guide"}
| Property | Cosine Similarity | Dot Product |
|---|---|---|
| Measures | Angle only | Angle + magnitude |
| Range | −1 to 1 | Unbounded |
| Normalize first? | Not needed | Yes, if you want cosine behavior |
| Works best when | All docs same "length" priority | Magnitude signals confidence |
| Default for most OSS models | Yes | No (but fast on GPU) |
| Trick: cosine via dot | — | Normalize vectors to unit length first |
:::

The fastest trick in the book: **normalize your vectors to unit length before indexing**, then dot product == cosine similarity. FAISS's `IndexFlatIP` (inner product) with L2-normalized vectors gives you cosine search at near-maximum speed.

## L2 distance: the third option nobody told you about

L2 (Euclidean distance) measures the straight-line distance between two points. For text embeddings it's almost always the wrong choice — two documents can be far apart in L2 yet semantically identical (just different magnitudes). Mention it in interviews to show you know it exists; explain why you'd skip it for text.

## Matryoshka Representation Learning (MRL)

Remember Russian nesting dolls? The biggest doll contains all the others. Matryoshka embeddings work the same way.

A standard 1536-dimension embedding from `text-embedding-3-large` encodes meaning across all 1536 dimensions. With Matryoshka training, the model is explicitly trained so that **the first 256 dimensions alone already encode a good summary of meaning**. The first 512 are better. The full 1536 are best.

Why does this matter in production? Storage and speed.

A 1536-dim float32 vector costs 6 KB. At 10 million documents that's **60 GB** just for the index. Truncating to 256 dimensions cuts that to **10 GB** with surprisingly little quality loss — often less than 5% drop in NDCG on typical retrieval benchmarks.

OpenAI's `text-embedding-3-small` and `text-embedding-3-large` are Matryoshka-trained. You can legally pass `dimensions=256` in the API call and get a genuinely smaller, high-quality embedding — not a truncated one bolted on after the fact.

```python {title="Matryoshka embeddings + cosine via normalized dot product" run=false}
# pip install openai numpy faiss-cpu
# For fully local/free: pip install sentence-transformers faiss-cpu
# and use model="sentence-transformers/all-MiniLM-L6-v2" with .encode()

import numpy as np
import faiss
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY env var

DIMS = 256  # Matryoshka: shrink from 1536 to 256


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts using Matryoshka-truncated OpenAI embeddings."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        dimensions=DIMS,  # Matryoshka magic — not a naive slice
    )
    vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    # Normalize to unit length → dot product becomes cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# --- Index documents ---
docs = [
    "Refund policy for digital goods: no returns after download.",
    "Refund policy for physical goods: 30-day return window.",
    "How to reset your password via the account portal.",
]

doc_vecs = embed(docs)

# FAISS inner-product index (== cosine because vectors are normalized)
index = faiss.IndexFlatIP(DIMS)
index.add(doc_vecs)

# --- Query ---
query_vec = embed(["digital product refund"])
scores, indices = index.search(query_vec, k=2)

for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
    print(f"Rank {rank+1} (score={score:.3f}): {docs[idx]}")
# Expected: digital goods chunk ranks #1, physical goods ranks #2
```

:::gotcha
If you mix normalized and un-normalized vectors in the same FAISS index, some queries will silently return wrong results — higher scores for semantically unrelated chunks just because those chunks had higher-magnitude embeddings. Always normalize *before* `index.add()` and normalize your query vector with the same function. Write a one-line unit test: `assert abs(np.linalg.norm(vec) - 1.0) < 1e-5`.
:::

## Choosing an embedding model in practice

Not all embedding models are equal, and bigger is not always better.

The [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (Massive Text Embedding Benchmark) is the canonical reference. Check scores for the *retrieval* task column — not the overall average, which is skewed by classification tasks.

For most production RAG systems in 2025–2026, the sweet spot is:

- **Free/local**: `BAAI/bge-m3` — multilingual, 8192-token context, strong MTEB retrieval scores
- **Cheap API**: `text-embedding-3-small` at 256 dims (Matryoshka) — excellent quality-per-dollar
- **Max quality**: `text-embedding-3-large` at full or 1536 dims

:::war-story {title="The silent score inflation bug"}
A team migrated their vector DB from Chroma (cosine by default) to FAISS (IndexFlatIP, dot product by default). They forgot to re-normalize the stored vectors. Retrieval recall appeared to *improve* on their internal eval set — because a handful of high-magnitude document embeddings (long, verbose policy pages) now scored extremely high across almost every query. Their RAGAS faithfulness metric looked great. In production, users complained the chatbot always cited the 10-page Terms of Service regardless of what they asked. The root cause: dot product rewarded length, not relevance. Fix: one line of normalization before indexing, plus a regression test on query-agnostic top-k distribution.
:::

:::interview-line
"Cosine similarity strips out magnitude and measures angle — which is almost always what you want for text retrieval. When I need speed at scale I normalize vectors once and use FAISS inner-product; the math is identical but 30% faster."
:::

:::qa {q="When would you choose dot product over cosine similarity for retrieval?"}
When your embedding model is explicitly trained with a dot-product objective (some bi-encoder models fine-tuned on commercial datasets are). Also when magnitude carries meaningful signal — for example a model trained to output higher-magnitude vectors for high-confidence passages. In practice, for most off-the-shelf sentence-transformers, cosine is the right default. Always check the model card.
:::

:::qa {q="What is Matryoshka Representation Learning and why should a production engineer care?"}
MRL trains models so that the first N dimensions of a longer embedding already encode a useful, shorter representation. This means you can truncate embeddings at inference time — say from 1536 to 256 dimensions — and get 6x storage savings with only a small retrieval quality drop. In production, this can cut your vector index memory footprint dramatically, enabling you to serve more documents in RAM or reduce cloud storage costs.
:::

:::drill {type="mcq" q="You are using FAISS IndexFlatIP (inner product). Which pre-processing step makes this equivalent to cosine similarity?"}
- [ ] Subtract the mean of each vector
- [ ] Apply PCA to reduce dimensions
- [x] Normalize each vector to unit L2 length before indexing and querying
- [ ] Convert float32 vectors to float16
:::

:::drill {type="mcq" q="A colleague says: 'Our MTEB overall leaderboard score is 72, so this model is great for RAG retrieval.' What is wrong with this reasoning?"}
- [ ] MTEB does not cover English-language models
- [ ] MTEB overall score only measures classification accuracy
- [x] The overall MTEB score averages many tasks; you should check the retrieval-specific sub-scores
- [ ] MTEB scores above 70 are unreliable due to data contamination
:::

:::key-takeaway
Use cosine similarity (or normalize-then-dot-product) for text retrieval by default. If your embedding model supports Matryoshka dimensions, truncate aggressively — you get 4–8x storage savings for very little quality loss. Always check the model card and the MTEB retrieval column before committing to a model in production.
:::
