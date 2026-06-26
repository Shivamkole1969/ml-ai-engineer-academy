---
id: reranking-hybrid
track: 09-genai-rag
title: "Reranking & hybrid search (BM25 + dense + cross-encoder)"
badge: HOT
minutes: 10
prereqs: []
tags: [rag, retrieval, bm25, dense-retrieval, reranking, cross-encoder, hybrid-search, rrf]
xp: 60
hot2026: true
---

You're building a RAG chatbot for a legal-tech startup in Pune. A senior lawyer types: **"force majeure clause pandemic"**. Your embedding search returns documents about "unforeseeable events" and "acts of God" — semantically on-point. Your keyword search surfaces documents containing "force majeure" and "pandemic" word-for-word — also on-point, but different ones. Either alone misses half the picture. Together, then filtered through a reranker, you get exactly what the lawyer needs. This lesson is about that pipeline.

## Why single-mode retrieval lets you down

**BM25** (the engine behind Elasticsearch, OpenSearch, and most legacy search) is a keyword scorer. It counts how often your query terms appear in a document and how rare those terms are across the corpus. It's blazing fast and brilliant for exact terminology — medical codes, legal clause names, product SKUs. But it has zero idea that "automobile" and "car" mean the same thing.

**Dense retrieval** uses embedding vectors (from a model like `text-embedding-3-small` or BGE). It maps your query and every document into the same high-dimensional space and finds the nearest neighbours. It understands synonyms, paraphrases, and intent. But it can fumble exact, rare terms — a 7-digit product code looks almost identical to a nearby 7-digit code in vector space.

Neither is universally better. That's the setup for hybrid search.

:::why-prod
In production, user queries are unpredictable. Some are semantic ("what's our leave policy?"), some are exact ("error code E_AUTH_4032"). A single retrieval mode will silently drop the other kind — and you won't notice until recall metrics crater or a user complains.
:::

## Hybrid search: fusing two ranked lists

Run BM25 and dense retrieval in parallel. Each returns a ranked list of document IDs. You need to merge them into one list.

The standard trick is **Reciprocal Rank Fusion (RRF)**:

```
score(doc) = Σ  1 / (k + rank_i(doc))
```

Where `rank_i` is the doc's position in retrieval list `i`, and `k` is a smoothing constant (typically 60). A doc ranked #1 in both lists wins big. A doc ranked #50 in one and absent from the other barely budges.

```python {title="Hybrid search with RRF" run=false}
# pip install rank-bm25 sentence-transformers faiss-cpu
# Free to run locally; no API key needed.

from rank_bm25 import BM25Okapi
import numpy as np

def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Merge multiple ranked doc-ID lists with RRF."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Return doc IDs sorted by fused score, highest first
    return sorted(scores, key=scores.__getitem__, reverse=True)

# --- Toy example ---
corpus = [
    "force majeure clause in commercial contracts",
    "pandemic disruptions and contract law",
    "unforeseeable events legal liability",
    "payment terms and late fees",
]
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

query = "force majeure pandemic"
bm25_scores = bm25.get_scores(query.split())
bm25_ranked = list(np.argsort(bm25_scores)[::-1])  # [0, 1, 2, 3]

# Imagine dense retrieval returned a different ranking based on semantics
dense_ranked = [2, 0, 1, 3]  # "unforeseeable events" ranked first semantically

fused = reciprocal_rank_fusion([bm25_ranked, dense_ranked])
print("Fused order:", [corpus[i] for i in fused])
# Both doc 0 and doc 2 score well — neither is left behind
```

:::table {title="BM25 vs Dense vs Hybrid"}
| | BM25 | Dense | Hybrid (RRF) |
|---|---|---|---|
| Strengths | Exact match, rare terms, fast | Semantic similarity, paraphrase | Best of both |
| Weaknesses | Can't handle synonyms | Drifts on exact rare tokens | Needs both indices |
| Latency | Very low | Low (ANN) | Slightly higher |
| Typical top-K recall | ~60–70 % | ~65–75 % | ~80–90 % |
:::

## Cross-encoder reranking: the final filter

Hybrid search gives you a merged top-50 or top-100 list. It's still noisy. That's where a **cross-encoder** steps in.

A standard embedding model encodes your query and each document independently — that's called a **bi-encoder**. It's fast because you pre-compute document embeddings offline.

A **cross-encoder** takes the query and a document *together* as one input and scores their relevance as a single forward pass. It sees the query tokens interacting with document tokens via self-attention. Much more accurate — but you can't pre-compute anything, so it's 10–100× slower.

The playbook: use bi-encoder + BM25 to get top-50, then send those 50 through the cross-encoder to re-score and re-rank. Your LLM only sees the top-3 or top-5 after reranking. Small prompt, high signal.

Popular cross-encoders you can run locally: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `BAAI/bge-reranker-large` (accurate), or Cohere's hosted `/rerank` API.

:::gotcha
Don't rerank all retrieved docs — that's expensive and slow. Always narrow with hybrid search first (top-50 to top-100), *then* rerank that short list. If you send 500 docs to a cross-encoder in production, your latency budget is gone and your users are gone soon after.
:::

:::war-story {title="The time 'transformer' broke our transformer chatbot"}
A fintech team in Bangalore shipped a RAG bot for regulatory Q&A. Dense retrieval worked great in testing. In production, users kept asking about "transformer model architecture" — and the bot returned RBI transformer substation guidelines instead of ML architecture docs. Both topics lived in the same corpus. The embedding for "transformer" sat ambiguously between both clusters. They added BM25 + a cross-encoder reranker in a weekend sprint. The reranker learned from query context that "transformer architecture layers" is not about electrical transformers. Problem solved.
:::

:::interview-line
"We run BM25 and dense retrieval in parallel, fuse with RRF, then cross-encode the top-50 to give the LLM a short, high-precision context window."
:::

:::qa {q="What is Reciprocal Rank Fusion and why use it over a simple weighted average?"}
RRF merges ranked lists using `1 / (k + rank)` without needing calibrated scores from each retrieval system — BM25 and embedding scores live on completely different scales and aren't directly comparable. RRF only cares about rank position, so it's robust, tuning-free, and proven to match or beat weighted combination in most benchmarks.
:::

:::qa {q="When would you skip the cross-encoder and rely only on bi-encoder scores?"}
When latency is the hard constraint — real-time autocomplete, sub-100 ms response requirements, or very large top-K lists (thousands of candidates). Bi-encoders pre-compute document embeddings offline, so retrieval is just a vector lookup. Cross-encoders are reserved for the final re-score of a small candidate set where quality matters more than speed.
:::

:::drill {type="mcq" q="A user queries 'RFC 2616'. BM25 retrieves the exact RFC document at rank 1. Dense retrieval ranks it at position 23. After RRF fusion (k=60), what happens to this document's final score?"}
- [ ] It drops because dense retrieval ranked it poorly
- [ ] It's excluded since it wasn't in dense retrieval's top 10
- [x] It scores well because a rank-1 BM25 hit strongly dominates the RRF sum
- [ ] It ties with all other documents that appeared in only one list
:::

:::drill {type="mcq" q="You have 10,000 candidates from hybrid search. You want to send only the best 5 to your LLM. What is the correct order of operations?"}
- [ ] Cross-encode all 10,000, then pick top 5
- [ ] Embed all 10,000 with a bi-encoder, then cross-encode top 5
- [x] Hybrid search → top 50–100 → cross-encoder → top 5 → LLM
- [ ] BM25 only → top 5 → LLM (cross-encoder is optional in production)
:::

:::key-takeaway
Hybrid search (BM25 + dense + RRF) raises recall by covering both exact and semantic queries; a cross-encoder reranker then raises precision by deeply scoring only the short candidate list — giving your LLM a tight, high-quality context window.
:::
