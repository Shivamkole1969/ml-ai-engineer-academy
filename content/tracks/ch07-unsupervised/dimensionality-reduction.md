---
id: dimensionality-reduction
track: ch07-unsupervised
title: "Dimensionality reduction: PCA, t-SNE, UMAP"
badge: CORE
minutes: 9
prereqs: []
tags: [pca, t-sne, umap, dimensionality-reduction, unsupervised, embeddings, visualization, scikit-learn]
xp: 45
hot2026: false
---

Your model returns 768-dimensional embeddings for every customer review. Your manager asks: "Can you show me what the different complaint types look like?" You can't plot 768 axes — no one can. You need to squeeze those dimensions down to 2 or 3 while keeping the important structure intact. That's exactly what dimensionality reduction does, and it shows up constantly in production: before training, during debugging, and whenever you need to explain clusters to a non-technical audience.

## The Problem with Too Many Dimensions

More features sounds like more information — but past a point it becomes noise. Distance calculations lose meaning. Training slows down. Models overfit. This is the **curse of dimensionality**: the more features you have, the sparser your data becomes in that space.

Dimensionality reduction finds a lower-dimensional representation that keeps the most important structure and discards the rest. You go from 768 features to, say, 32 — or even just 2 for a plot.

There are three tools you need to know: PCA, t-SNE, and UMAP. They have very different personalities.

:::why-prod
Reducing feature dimensions before training can cut memory usage and training time by 10–100x. Visualization of embeddings helps you catch data bugs, label leakage, and cluster drift before they hit production. These aren't academic exercises — they're daily debugging tools.
:::

## PCA: The Linear Compressor

PCA (Principal Component Analysis) finds the directions along which your data varies the most and projects everything onto those directions. Those directions are called **principal components**.

It's **linear**, **deterministic**, and — critically — **reusable**. Once PCA is fit on your training set, you get a projection matrix. Apply that same matrix to a brand-new sample at inference time. No re-running required. That's what makes PCA production-safe.

PCA also tells you how much variance each component explains. If the first 10 components explain 92% of variance, you've lost only 8% of information — often mostly noise — while cutting your feature count dramatically.

**Use PCA when:**
- You need to compress features before feeding them to a downstream model
- You want to remove highly correlated features
- You need to apply the same transform to new data at inference time

**Its limit:** Linear only. If your data lives on a curved or twisted surface — a "manifold" — PCA will miss that structure.

## t-SNE: The Beautiful Visualizer

t-SNE (t-distributed Stochastic Neighbor Embedding) creates stunning 2D or 3D visualizations that reveal clusters PCA can't see. It's non-linear, so it can "unfold" complex manifolds and pull apart clusters that overlap in the original space.

The catch: **t-SNE is visualization-only**.

It learns a mapping specific to the exact dataset you give it. There is no transformation matrix to apply to new points. Run it again on the same data with a different random seed — you get a different plot. The axes are meaningless; global distances aren't preserved.

Think of t-SNE as a "look at this!" tool for exploration. It is not a "let's ship this" tool for pipelines.

**The key hyperparameter:** `perplexity` (roughly, the number of neighbors each point considers). Values of 5–50 are typical. Different perplexities produce dramatically different plots — always state what you used.

## UMAP: The Modern Workhorse

UMAP (Uniform Manifold Approximation and Projection) does what t-SNE does — beautiful non-linear visualization with cluster separation — but with three big advantages:

1. **Much faster**: often 10–50x faster than t-SNE on the same data
2. **Better global structure**: UMAP tends to preserve the relative positions of clusters better, not just their internal shape
3. **Can transform new points**: fit once, apply to new samples at inference time, just like PCA

UMAP is increasingly replacing t-SNE for embedding visualization in production systems, and it's also safe for some preprocessing use cases.

:::table {title="PCA vs t-SNE vs UMAP at a glance"}
| Method | Speed | Linear? | Transforms new points? | Best for |
|---|---|---|---|---|
| PCA | Very fast | Yes | Yes | Preprocessing, feature compression |
| t-SNE | Slow | No | No | Visualization only |
| UMAP | Fast | No | Yes | Visualization + some preprocessing |
:::

## Code: All Three Side by Side

```python {title="Compare PCA, t-SNE, and UMAP on a real dataset" run=false}
# pip install scikit-learn umap-learn matplotlib
# Uses only local data — no API key or download needed

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits  # 1797 samples, 64 features, 10 classes
import umap

digits = load_digits()
X, y = digits.data, digits.target  # X.shape == (1797, 64)

# --- PCA ---
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
print(f"Variance explained by 2 PCs: {pca.explained_variance_ratio_.sum():.1%}")

# --- t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# --- UMAP ---
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)

# Plot all three
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, X_2d, label in zip(axes, [X_pca, X_tsne, X_umap], ["PCA", "t-SNE", "UMAP"]):
    sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=5, alpha=0.7)
    ax.set_title(label)
plt.colorbar(sc, ax=axes[-1], label="Digit class")
plt.tight_layout()
plt.savefig("dim_reduction_comparison.png", dpi=150)
print("Saved: dim_reduction_comparison.png")

# Only PCA and UMAP can transform a single new sample
new_point = X[:1]  # shape (1, 64)
print("PCA transform shape:  ", pca.transform(new_point).shape)      # (1, 2) — works
print("UMAP transform shape: ", reducer.transform(new_point).shape)  # (1, 2) — works
# tsne.transform(new_point)  <-- AttributeError: t-SNE has no transform()
```

:::gotcha
The t-SNE `perplexity` parameter changes the plot dramatically — clusters can appear, merge, or split depending on the value you pick. There is no single "correct" perplexity. When sharing a t-SNE visualization, always annotate the perplexity value. Two people plotting the same embeddings with different perplexities will see completely different pictures and may draw opposite conclusions.
:::

## Choosing the Right Tool for the Job

One more practical rule: for very high-dimensional data (say, 512-dim embeddings), run PCA first to get to something like 50 dimensions, then hand that to UMAP or t-SNE. This pre-compression is a common trick that speeds up the non-linear methods significantly without losing meaningful structure.

The flow looks like:

`512-dim embeddings → PCA(50) → UMAP(2) → scatter plot`

That combination is fast and produces cleaner separations than jumping straight from 512 to 2.

:::interview-line
"PCA for preprocessing and pipelines; UMAP when I need to show stakeholders how clusters look; t-SNE only for one-off exploration — never in anything that runs on new data."
:::

:::qa {q="Why can't you use t-SNE in an inference pipeline that transforms new data points?"}
t-SNE minimizes a global probabilistic objective over the entire training set. It doesn't learn a reusable function — there's no projection matrix you can apply to a new sample. PCA learns a linear projection matrix and UMAP learns a parametric mapping, so both can call `.transform()` on new data after fitting. t-SNE cannot, making it unsuitable for any pipeline that runs at inference time.
:::

:::qa {q="Your PCA explained variance plot shows that the first 15 components capture 90% of variance in a 200-feature dataset. What would you do in practice?"}
Use 15 components as the compressed representation for downstream training. Retaining 90% of variance typically discards mostly noise rather than meaningful signal, which can actually improve generalization. After training, check whether model performance drops compared to the full 200 features — if it doesn't, you've bought yourself a 13x reduction in feature dimensionality at no cost.
:::

:::drill {type="mcq" q="You're building an inference pipeline that receives 512-dim text embeddings and needs to compress them to 32 dimensions before passing them to a classifier. Which method should you use?"}
- [ ] t-SNE with perplexity=30
- [ ] UMAP in a non-parametric mode that recomputes the manifold for every batch
- [x] PCA, because it learns a reusable linear projection that instantly transforms new points
- [ ] t-SNE with a fixed random seed to ensure reproducibility
:::

:::drill {type="mcq" q="A colleague shows you a t-SNE plot where 5 distinct customer clusters appear with no overlap. They want to use cluster membership as a feature in a production model. What's the problem?"}
- [ ] t-SNE is too slow to run daily on new data
- [ ] t-SNE can't handle more than 3 clusters
- [x] t-SNE can't transform new customers — you'd need to re-run the entire algorithm each time, and results change with different hyperparameters
- [ ] t-SNE doesn't support high-dimensional data
:::

:::key-takeaway
PCA = reusable linear compression, production-safe for pipelines. UMAP = fast non-linear visualization that can also handle new points. t-SNE = beautiful for exploration, never for inference. When in doubt: PCA first, then UMAP for the plot.
:::
