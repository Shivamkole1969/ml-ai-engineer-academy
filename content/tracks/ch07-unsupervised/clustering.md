---
id: clustering
track: ch07-unsupervised
title: "Clustering: k-means, DBSCAN, hierarchical"
badge: CORE
minutes: 9
prereqs: []
tags: [clustering, k-means, dbscan, hierarchical, unsupervised, segmentation]
xp: 45
hot2026: false
---

Your product has been live for six months and the CEO asks: "Who ARE our users?" You have 2 million rows of behavioral data and zero labels. Nobody told you who's a "power user" vs. a "window shopper" — you have to figure it out yourself.

This is the clustering problem. You're not predicting anything. You're discovering structure that already exists in the data.

## What Clustering Actually Does

Clustering groups data points so that things inside a group are more similar to each other than to things in other groups. That's it. No labels needed.

Three algorithms dominate production use:

- **k-means** — fast, simple, loves spherical blobs
- **DBSCAN** — finds arbitrary shapes, flags noise as outliers
- **Hierarchical** — builds a tree of merges, lets you pick depth later

Each makes different assumptions about what a "cluster" even looks like.

:::why-prod
Clustering drives real product decisions: personalized recommendations, targeted campaigns, fraud rings, hardware fault grouping, and customer lifetime value tiers. The output is only valuable if a human can act on each segment — "Cluster 3" must mean something to the business.
:::

## k-means: The Workhorse

You tell k-means: "Find me k clusters." It places k centroids, assigns every point to its nearest centroid, moves the centroids to the mean of their assigned points, and repeats until stable.

It's fast. It scales. It's the default starting point.

The catch: you must choose k upfront, and it assumes clusters are roughly spherical and equally sized. Elongated or crescent-shaped clusters will confuse it.

```python {title="k-means on user behavior data" run=false}
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# pip install scikit-learn
# Columns: [session_length_min, pages_per_session, purchases_30d]
X = np.array([
    [2.1, 3, 0], [1.8, 2, 0], [15.0, 20, 8],
    [14.5, 18, 6], [5.0, 7, 1], [4.8, 6, 2],
])

# ALWAYS scale before clustering — k-means is distance-based
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

print(labels)  # e.g. [0, 0, 1, 1, 2, 2]
# Interpret: cluster 1 = power buyers, cluster 0 = bouncers, cluster 2 = browsers
```

How do you pick k? Use the **elbow method**: plot inertia (sum of squared distances to centroid) vs. k and look for where the curve bends. Or use **silhouette score** — higher is more separated.

## DBSCAN: Density is the Signal

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) works differently. Instead of centroids, it finds dense regions.

You set two knobs:
- `eps` — the neighborhood radius
- `min_samples` — minimum points needed to form a dense region

Points in a dense region become a cluster. Points with no dense neighborhood get labeled **-1** (noise/outlier).

This is powerful for two reasons: clusters can be any shape, and you get outlier detection for free. In fraud detection, those -1 points are exactly what you care about.

:::table {title="Algorithm Comparison"}
| Algorithm | Needs k? | Handles outliers? | Shape assumption | Speed at scale |
|---|---|---|---|---|
| k-means | Yes | No | Spherical, equal size | Fast (O(n)) |
| DBSCAN | No | Yes (flags noise) | Arbitrary | Medium (O(n log n)) |
| Hierarchical | No | No | None | Slow (O(n²)) |
:::

## Hierarchical: The Tree View

Hierarchical clustering builds a **dendrogram** — a tree where leaves are individual points and branches merge similar groups bottom-up (agglomerative) or split them top-down (divisive).

You don't pick k upfront. You cut the dendrogram at a height that makes business sense. Want 3 customer tiers? Cut at level 3. Want 10 micro-segments? Cut at level 10.

The downside: it doesn't scale. O(n²) memory means 100k+ rows will stall your laptop. Use it for exploration on a sample, then apply k-means or DBSCAN on the full dataset once you understand the structure.

:::gotcha
Never feed raw features of different units into a clustering algorithm without scaling first. A "purchase_amount" column in dollars (0–5000) will completely dominate a "session_count" column (0–20), because distance-based methods treat large numbers as more important. Always apply `StandardScaler` or `MinMaxScaler` before clustering. This is the single most common mistake in production clustering pipelines.
:::

## Evaluating Clusters (Without Labels)

This is the hard part. You have no ground truth, so traditional accuracy doesn't work.

Common internal metrics:
- **Silhouette score** (-1 to 1): how well each point fits its own cluster vs. neighbors. Above 0.5 is decent.
- **Davies-Bouldin index**: lower is better; measures cluster separation.
- **Inertia** (k-means only): total squared distance to centroids. Use for elbow plots, not comparison across k.

But the real test is business validation: can a human describe each cluster in one sentence? Can the team take a distinct action for each segment? If yes, the clustering works.

:::interview-line
"Clustering has no ground truth, so I validate with silhouette scores internally and business stakeholders externally — a segment only matters if someone can act on it."
:::

:::qa {q="When would you choose DBSCAN over k-means?"}
When you don't know how many clusters exist upfront, when your clusters might be non-spherical or irregularly shaped, or when you want outlier detection built in. DBSCAN's noise label (-1) is a free anomaly detector — in fraud or network intrusion work, those noise points are often the most interesting ones.
:::

:::qa {q="How do you pick k for k-means in practice?"}
Start with domain knowledge if you have it — "we want three customer tiers" is a valid constraint. Otherwise, plot inertia vs. k (elbow method) and silhouette score vs. k, then pick the k where both look favorable. Always sanity-check by reading a few samples from each cluster: do they make intuitive sense?
:::

:::qa {q="Why must you scale features before clustering?"}
Clustering algorithms like k-means and DBSCAN use distance metrics. If one feature has a much larger numeric range than others, it dominates the distance calculation and the other features become irrelevant. StandardScaler transforms each feature to zero mean and unit variance so all dimensions contribute equally.
:::

:::drill {type="mcq" q="You run k-means with k=4 on customer data. Cluster 2 contains 80% of all users and the other three clusters are tiny. What is the most likely cause?"}
- [ ] The silhouette score is too high
- [ ] You forgot to set random_state
- [x] Features were not scaled before clustering
- [ ] k-means cannot handle more than 3 clusters
:::

:::drill {type="mcq" q="DBSCAN labels some points as -1. What does this mean?"}
- [ ] Those points belong to the largest cluster
- [ ] The algorithm failed to converge
- [x] Those points are noise — they don't belong to any dense region
- [ ] You need to increase the number of clusters
:::

:::drill {type="mcq" q="You have 500k rows of GPS location data and want to find geographic hot-spots of any shape. Which algorithm fits best?"}
- [ ] Hierarchical clustering
- [ ] k-means with k=10
- [x] DBSCAN
- [ ] PCA followed by logistic regression
:::

:::key-takeaway
Clustering finds hidden structure without labels. k-means is fast and simple but needs k upfront and assumes spherical blobs; DBSCAN finds arbitrary shapes and flags outliers for free; hierarchical gives you a full tree but doesn't scale. Always scale your features first, and validate clusters with both a metric (silhouette score) and a human sanity check.
:::
