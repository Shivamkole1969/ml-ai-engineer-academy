---
id: recsys-fundamentals
track: t8-extended
title: "Recommender Systems Fundamentals"
badge: CORE
minutes: 9
prereqs: []
tags: [recommender-systems, collaborative-filtering, content-based, matrix-factorization, cold-start, evaluation]
xp: 45
hot2026: false
---

Imagine you join a new music streaming app. You've listened to exactly three songs. The app still manages to surface a playlist that feels eerily right. Now imagine you're the engineer who built that — and it has to work just as well for the user who has listened to 10,000 songs. That's the core tension in recommender systems: making good predictions with thin data and staying useful as users grow.

Recommender systems are everywhere — e-commerce, streaming, social feeds, job boards, even the "people you may know" feature on professional networks. They also account for a huge slice of revenue. Get them right and engagement climbs. Get them wrong and users churn quietly.

## The Three Core Paradigms

Every recommendation approach falls into one of three buckets — or blends them.

**Collaborative Filtering (CF)** says: find users who behave like you, then recommend what they liked. No knowledge of item content required. Pure signal from user-item interactions.

**Content-Based Filtering (CBF)** says: if you liked an action thriller from 2019, here are more films with similar genre, director, and pacing. It relies on item metadata and your past preferences, not what others did.

**Hybrid** systems combine both signals — CB to bootstrap early, CF as interaction data grows. Most production systems land here.

:::why-prod
In production, no single paradigm wins universally. Collaborative filtering degrades on sparse data (new users, new items). Content-based filtering creates filter bubbles and misses serendipity. Knowing which weakness dominates your traffic pattern determines which architecture to reach for first.
:::

## Collaborative Filtering in Practice

The classic CF setup: you have a matrix where rows are users, columns are items, and cells hold ratings or implicit signals (clicks, plays, purchases). Most cells are empty — this is called a sparse matrix. Your job is to fill in the blanks.

**Memory-based CF** computes similarity directly: user-user (find your nearest neighbours, average their ratings) or item-item (find similar items to what you liked). It's interpretable but expensive at scale.

**Model-based CF** learns a compact representation. Matrix factorization is the workhorse here. You decompose the user-item matrix into two lower-rank matrices: one for users, one for items. Each user and item gets an embedding vector. The dot product of a user's vector and an item's vector predicts the rating.

```python {title="Toy matrix factorization with SGD" run=false}
import numpy as np

# R[i][j] = rating of item j by user i (0 means unobserved)
R = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
], dtype=float)

n_users, n_items = R.shape
K = 3          # latent factors
lr = 0.01
epochs = 1000

# Random init — small values to avoid saturation
P = np.random.normal(0, 0.1, (n_users, K))   # user embeddings
Q = np.random.normal(0, 0.1, (n_items, K))   # item embeddings

for _ in range(epochs):
    for i in range(n_users):
        for j in range(n_items):
            if R[i][j] > 0:                   # only train on observed ratings
                err = R[i][j] - P[i] @ Q[j]
                P[i] += lr * err * Q[j]
                Q[j] += lr * err * P[i]

# Predict rating for user 0, item 2 (was unobserved)
predicted = P[0] @ Q[2]
print(f"Predicted rating: {predicted:.2f}")

# Run locally: pip install numpy  — no GPU needed
```

This toy version shows the mechanics. Production systems use ALS (Alternating Least Squares) for implicit feedback, or GPU-accelerated libraries like LightFM or Implicit.

## Content-Based Filtering

CBF builds an item profile (genre, tags, description embeddings) and a user profile (the weighted average of item profiles they've interacted with). Cosine similarity ranks candidates.

The big win: cold-start on items is trivial. A new movie with rich metadata can be recommended the day it's catalogued — no interaction needed.

The big risk: you recommend more of what the user already knows they like. Discovery suffers.

## The Cold Start Problem

Cold start is the recommender's original sin. Two variants:

- **New user cold start** — you know nothing about them. Fallback: trending globally, popularity-based, onboarding questionnaire to seed preferences.
- **New item cold start** — the item has zero interactions. Fallback: content-based signals, editorial boost, bandits (explore intentionally to gather signal fast).

:::table {title="Paradigm Comparison"}
| Approach | Needs item metadata | Needs interaction data | Cold-start (user) | Cold-start (item) | Serendipity |
|---|---|---|---|---|---|
| Collaborative Filtering | No | Yes (lots) | Poor | Poor | High |
| Content-Based Filtering | Yes | Minimal | Good | Good | Low |
| Hybrid | Yes | Yes | OK | Good | Medium |
:::

## How to Measure Quality

Accuracy metrics alone mislead you in production. Track these:

- **Precision@K** — of the top-K items shown, what fraction did the user actually engage with?
- **Recall@K** — of all items the user would have engaged with, how many appeared in the top K?
- **NDCG@K** — like precision@K but rewards ranking the best item higher within the list.
- **Coverage** — what fraction of the catalog gets recommended at all? A system that only recommends the top 100 popular items is useless for a long-tail catalog.
- **Diversity & novelty** — does the list introduce the user to something genuinely new?

:::gotcha
Optimizing only for click-through rate (CTR) without a longer-horizon metric is a classic trap. A clickbait thumbnail might drive CTR up but tank session length and next-week retention. Always pair an engagement metric with a satisfaction or retention signal before declaring a win.
:::

:::interview-line
"A recommender system is really a ranking problem: you're learning a scoring function over user-item pairs, not just predicting a rating."
:::

:::qa {q="What is the cold-start problem and how do you handle it in production?"}
Cold start happens when a user or item is new and has little or no interaction history. For new users, you fall back to popularity-based recommendations or a short onboarding flow to capture initial preferences. For new items, you rely on content-based signals from metadata and optionally use exploration strategies (like multi-armed bandits) to gather interaction data quickly.
:::

:::qa {q="Why is matrix factorization preferred over memory-based collaborative filtering at scale?"}
Memory-based CF requires computing similarity between every pair of users or items at query time, which becomes prohibitive with millions of users. Matrix factorization compresses each user and item into a small embedding vector offline. At serving time, a recommendation is just a dot product — cheap to compute and easy to approximate with approximate nearest neighbour (ANN) search.
:::

:::qa {q="What evaluation metrics matter most for a recommender system?"}
Offline ranking metrics like NDCG@K and Recall@K measure how well the model orders relevant items. But they don't capture business outcomes. In production, you pair these with online A/B metrics like session length, click-through rate over time, and retention rate to confirm the model is genuinely improving user experience, not just gaming a proxy.
:::

:::drill {type="mcq" q="A brand-new user signs up for your streaming platform. Which strategy is most appropriate for initial recommendations?"}
- [ ] Run matrix factorization on their empty history and serve the top predictions
- [x] Show popularity-based or editorially-curated content while gathering early interaction signals
- [ ] Skip recommendations entirely until they have at least 20 interactions
- [ ] Run a content-based model using their username as a feature
:::

:::drill {type="mcq" q="You deploy a new recommender model. Precision@10 improves by 8% in offline evaluation but session length drops 5% in an A/B test. What should you conclude?"}
- [ ] The model is better — precision@10 is the gold standard metric
- [ ] The A/B result is likely a fluke; roll out the new model
- [x] The offline metric may not align with real user satisfaction — investigate before rolling out
- [ ] Both metrics must be wrong; re-run the offline evaluation
:::

:::key-takeaway
Recommender systems are fundamentally a sparse ranking problem. Collaborative filtering leverages crowd signal but breaks on cold start; content-based filtering works without interaction history but creates filter bubbles. Production systems blend both — and measure success with business metrics, not just offline accuracy scores.
:::
