---
id: two-tower
track: t8-extended
title: "Two-tower retrieval & ranking"
badge: HOT
minutes: 9
prereqs: []
tags: [retrieval, ranking, embeddings, recommendations, ANN, contrastive-learning]
xp: 60
hot2026: true
---

Imagine you work at a music streaming service. You have 80 million songs and 200 million users. Every time someone opens the app, you need to surface 20 tracks they will actually want to hear — in under 100 milliseconds. Running a neural network that scores every song against every user at query time is mathematically absurd. Yet the best recommendations in production do exactly that... just not the way you might think. The trick is called the **two-tower model**, and it powers recommendation, search, and retrieval at almost every major tech company.

## What is a two-tower model?

A two-tower (also called dual-encoder) model trains two independent neural networks in parallel:

- **User tower** — takes user features (history, demographics, context) and outputs a fixed-length embedding vector, say 128 dimensions.
- **Item tower** — takes item features (song genre, tempo, lyrics embedding, popularity) and outputs a vector of the same size.

During training the two towers learn to place "good matches" close together in that shared vector space. At serving time you pre-compute all item embeddings offline, then at query time you only run the user tower once and find the nearest item vectors — no scoring 80 million songs live.

:::why-prod
Pre-computing item embeddings turns an O(N) per-query scoring problem into a single forward pass plus an ANN (approximate nearest neighbor) lookup that costs microseconds. This is what makes sub-100ms personalization at scale physically possible.
:::

## The architecture in detail

Both towers share nothing except the loss function during training. A typical setup:

:::table {title="Tower anatomy"}
| Tower | Inputs | Hidden layers | Output |
|---|---|---|---|
| User | user ID embed + history sequence + context | 2–3 dense / transformer layers | 128-d L2-normalized vector |
| Item | item ID embed + content features | 2–3 dense layers | 128-d L2-normalized vector |
:::

The **similarity score** between user vector `u` and item vector `v` is just a dot product (fast, differentiable):

```
score(u, v) = u · v
```

Because both vectors are L2-normalized, this equals cosine similarity. High score = good match.

## Training with in-batch negatives

You train on (user, positive item, negative items) triples. The easiest source of negatives: the other items in the same training batch. If your batch has 512 pairs, each user sees 511 "random" negatives for free — this is called **in-batch softmax loss**.

```python {title="Minimal two-tower training loop" run=false}
import torch
import torch.nn.functional as F

def two_tower_loss(user_emb, item_emb, temperature=0.07):
    """
    user_emb: (B, D) — one embedding per user in batch
    item_emb: (B, D) — corresponding positive item embeddings
    Negatives = all other items in the batch (in-batch strategy).
    Run locally: pip install torch  (CPU-only is fine for this snippet)
    """
    # L2-normalize both towers
    u = F.normalize(user_emb, dim=-1)   # (B, D)
    v = F.normalize(item_emb, dim=-1)   # (B, D)

    # Pairwise cosine similarity matrix: (B, B)
    logits = torch.matmul(u, v.T) / temperature

    # Diagonal = positive pairs; off-diagonal = in-batch negatives
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# Dummy tensors — swap in your real embeddings
B, D = 32, 128
user_emb = torch.randn(B, D)
item_emb = torch.randn(B, D)
print(two_tower_loss(user_emb, item_emb).item())
```

Temperature controls how "peaked" the distribution is. Lower temperature = harder training signal.

## Hard negative mining

Random negatives are easy. The model quickly learns to separate a pop song from total noise. To keep improving, you need **hard negatives** — items that look plausible but aren't right for this user (same genre, similar tempo, but the user historically skips them). Common strategies:

- **Offline mining**: run the model, find items ranked in positions 100–500 for a user, use those as negatives next epoch.
- **Dynamic hard negatives**: share item embeddings across GPUs and mine from a large pool each step (used in Google's "Sampling-Bias-Corrected" two-tower paper).

:::gotcha
If you mine negatives that are **too hard** (actual positives the user clicked but weren't labeled), training collapses — loss goes down but recall tanks. Always filter known positives before using hard negatives.
:::

## Retrieval vs. ranking: the two-stage pipeline

Two-tower models are a **retrieval** (a.k.a. candidate generation) layer. They return the top-K candidates fast and roughly. A separate, heavier **ranking model** then scores only those K candidates with richer features (cross-attention between user + item, real-time context, business rules). This cascade is everywhere:

1. **Stage 1 — two-tower retrieval**: ANN search, returns top 500 candidates in ~5 ms.
2. **Stage 2 — ranker**: a pointwise or listwise model scores the 500, returns top 20. Can afford more compute per item.
3. **Stage 3 — re-ranking / business rules**: diversity injection, freshness boost, safety filters.

The ANN search itself uses libraries like **FAISS** (Meta), **ScaNN** (Google), or **Pinecone/Weaviate** in managed settings. They index item vectors into a structure (e.g., IVF + PQ compression) that makes nearest-neighbor search sub-linear.

## Keeping item embeddings fresh

Item vectors are computed offline on a schedule. If a new song is uploaded, it won't be retrievable until the next embedding refresh — which could be hours. Solutions:

- **Streaming index updates**: recompute and upsert new-item embeddings to the ANN index within minutes of upload.
- **Cold-start fallback**: serve new items through a popularity or content-based fallback until the embedding pipeline catches up.

:::war-story {title="The 'invisible new items' incident"}
A video platform launched a major content drop — 10,000 new titles added overnight. The two-tower index was refreshed nightly at 3am. By 9am the engineering team noticed engagement on the new titles was near zero. Users simply weren't seeing them. The item tower had been run, but a pipeline bug failed to upsert new embeddings into FAISS before the cache expired. New titles were literally invisible to the retrieval stage for 18 hours. The fix was dead simple: add an ANN-upsert health check and an alert on "items indexed < items in catalog." Post-mortem led to a streaming upsert job that runs every 15 minutes.
:::

:::interview-line
"Two-tower separates encode-offline from score-online — that's the trick that makes sub-100ms personalization at scale possible."
:::

:::qa {q="Why do we normalize embeddings to unit length in a two-tower model?"}
L2 normalization makes the dot product equivalent to cosine similarity, which is scale-invariant — a feature that fires 10× louder than another won't dominate just because of magnitude. More practically, ANN indexes (especially FAISS HNSW) are optimized for inner-product or cosine search and perform best on normalized vectors.
:::

:::qa {q="What's the main difference between the retrieval stage and the ranking stage in a recommendation pipeline?"}
Retrieval (two-tower + ANN) runs over millions of items in milliseconds by using pre-computed embeddings and approximate search — it's fast but coarse. Ranking runs a heavier model over only a few hundred candidates and can use features that require seeing both user and item together (cross-features, real-time signals). The cascade trades breadth for depth at each stage.
:::

:::qa {q="What is temperature in contrastive training, and why does it matter?"}
Temperature divides the logits before softmax, controlling how sharply the model distinguishes positives from negatives. A low temperature (e.g. 0.07) creates a hard distribution that pushes the model to really separate embeddings. Too low and training becomes unstable; too high and the gradient signal is too diffuse. It's one of the most sensitive hyperparameters in two-tower training.
:::

:::drill {type="mcq" q="You've deployed a two-tower model. You add 5,000 new products but users don't see any of them in recommendations for 12 hours. What's the most likely root cause?"}
- [ ] The user tower needs retraining before it can handle new items
- [ ] Temperature is set too high, deprioritizing unseen items
- [x] New item embeddings were computed but not upserted into the ANN index before serving
- [ ] In-batch negatives accidentally included the new items as hard negatives
:::

:::drill {type="mcq" q="In a two-tower model trained with in-batch softmax loss, what are the 'negatives' for a given (user, item) positive pair?"}
- [ ] Items the user explicitly disliked (thumbs-down signals)
- [ ] A fixed set of randomly sampled items from the catalog, pre-selected before training
- [x] All other items in the same training batch that are not the positive for that user
- [ ] Items from the previous mini-batch stored in a memory bank
:::

:::key-takeaway
Two-tower models encode users and items into the same vector space offline, then use ANN search to retrieve candidates in milliseconds — making personalization at scale practical. They are always followed by a heavier ranking stage that does the fine-grained scoring.
:::
