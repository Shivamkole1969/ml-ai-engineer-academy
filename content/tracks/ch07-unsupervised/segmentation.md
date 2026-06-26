---
id: segmentation
track: ch07-unsupervised
title: "Segmentation that a business can act on"
badge: FOUNDATION
minutes: 9
prereqs: []
tags: [segmentation, clustering, business, profiling, stability]
xp: 30
hot2026: false
---

Your ML team spent two days tuning k-means on your e-commerce customer data. The silhouette score is solid. You get five clusters: 0, 1, 2, 3, 4. You walk into the product meeting. The head of growth leans forward and asks, "Cool — so what do we *do* with cluster 2?" Silence. That is the gap this lesson closes.

Clustering algorithms group data. Segments are groups that a team can name, prioritize, and act on. The algorithm's job ends at the cluster boundary. Your job begins there.

## Why Clustering and Segmentation Are Not the Same Thing

A cluster is a mathematical region in feature space. A segment is a cluster with a story attached.

"Cluster 3" means nothing to a product manager. "High-frequency buyers who churn on price shocks" means everything. Same data, different framing — and the framing is what makes the work valuable.

The move from cluster to segment has three steps: **profile**, **name**, **size**.

:::why-prod
In production, segments drive decisions: which users get which email, which customers get a retention call, which cohort gets a price experiment. If no one can describe a segment in plain language, no one will use it. A segment that sits in a notebook is waste.
:::

## Step 1 — Profile Each Cluster

Profiling means asking: what makes this cluster different from the others?

Pull the median of each key variable per cluster, then compare clusters against each other — not against the global average. You want *relative* character, not absolute values.

```python {title="Quick segment profiler" run=false}
import pandas as pd

# df has your features + a 'cluster' column from your trained model
profile = (
    df.groupby("cluster")
    [["avg_order_value", "purchase_freq", "days_since_last_order", "tenure_days"]]
    .median()
    .round(1)
)

# Normalize 0–1 so high/low pops at a glance
profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
print(profile_norm.to_markdown())

# Run locally: pip install pandas tabulate
# Feed any DataFrame that has a 'cluster' column
```

Look for features where one cluster is clearly at the top or bottom. Those are the defining traits — the ones that make the name obvious.

## Step 2 — Name and Describe Each Segment

Once you see the profile, give each segment a name a non-technical stakeholder would recognize.

:::table {title="Example segment profiles (e-commerce)"}
| Segment name | Order value | Purchase freq | Days since last order | Strategic action |
|---|---|---|---|---|
| Champions | High | High | Low | Upsell, loyalty perks |
| At-Risk Loyalists | High | Medium | Medium | Re-engagement campaign |
| Bargain Hunters | Low | High | Low | Bundle deals, price anchoring |
| Dormant | Medium | Low | High | Win-back email or sunset |
| New Explorers | Low | Low | Low | Onboarding, first-purchase nudge |
:::

Notice the last column: **Strategic action**. That is the real test. If you cannot fill in that column for a segment, the segment is not ready. Go back and re-profile.

If three of your five clusters would get the same action, you have too many clusters. Merge them.

## Step 3 — Size and Prioritize

Not all segments deserve equal attention. Count users (or revenue) in each one and plot effort against impact.

A segment containing 0.5 % of users and 0.3 % of revenue is probably not worth a dedicated campaign this quarter. A segment containing 8 % of users but 35 % of revenue — that is your first call on Monday.

Sizing also reveals when clustering went wrong. If one cluster holds 90 % of the data, the model found one big blob, not structure. Try a different k, different features, or a different algorithm.

## Keeping Segments Stable Over Time

Here is the part most tutorials skip.

Segments trained once will drift. User behaviour changes. New cohorts enter. If you re-run clustering next month, cluster 2 might not be "At-Risk Loyalists" anymore — the label could flip because k-means initializes randomly and the data distribution shifted.

Two production patterns that help:

**Fix the model, score new users against it.** Train once on a representative snapshot. Then use `model.predict()` to assign new users to the same segments. Labels stay stable.

**Versioned segment definitions.** Save cluster centroids and the feature pipeline together. When you do a scheduled re-train, compare new centroids against old ones before publishing. If a centroid drifts too far, raise an alert before you accidentally rename your best customers.

:::gotcha
The most common trap: choosing the number of clusters to maximize a mathematical metric (elbow curve, silhouette score) without checking if the resulting segments are *distinct enough to treat differently*. Five mathematically clean clusters where three of them drive the same business action is worse than three messier clusters where every one drives a different decision. Start with the action column; work backwards to cluster count.
:::

:::interview-line
"A segment is a cluster that passed the 'so what?' test — if you can't write a different action for each one, you have too many clusters."
:::

:::qa {q="How do you validate that your customer segments are actually useful?"}
Give each segment a distinct recommended action — a different email, price, or feature offer. Then A/B test that action against a generic control. If the segment-specific treatment outperforms the generic one, the segment carries real signal. Mathematical metrics like silhouette score only tell you the clusters are geometrically tidy; they say nothing about strategic value.
:::

:::qa {q="How do you handle segment stability when you retrain your clustering model?"}
Freeze the trained model after the initial run and use `predict()` to assign new users going forward. When a scheduled re-train is warranted, compare new cluster centroids against the saved previous version and gate promotion behind a drift check. Also keep segment names decoupled from cluster integer IDs — map names to centroid descriptions so that a centroid swap does not silently relabel your most valuable cohort.
:::

:::drill {type="mcq" q="A data scientist shows you five clusters with a high silhouette score. Three of the five clusters would receive the same marketing campaign. What should you do?"}
- [ ] Keep five clusters — a better silhouette score always means a better model.
- [x] Reduce to fewer segments so every segment drives a distinct business action.
- [ ] Switch to DBSCAN — it always produces more actionable segments.
- [ ] Add more features until the three clusters diverge.
:::

:::drill {type="mcq" q="Your clustering model is retrained monthly. Next month, cluster 0 has swapped meaning with last month's cluster 2. What is the safest long-term fix?"}
- [ ] Rename clusters manually each month after retraining.
- [ ] Always set random_state=42 — it prevents cluster label swaps entirely.
- [x] Freeze the trained model and score new data with predict(); only retrain on a scheduled cycle with a centroid drift check before publishing.
- [ ] Switch to hierarchical clustering — it never changes labels after training.
:::

:::key-takeaway
Clustering gives you groups; segmentation gives you decisions. A segment earns its name only when every segment in your list maps to a *different* business action. Start with the action column, work backwards to the algorithm.
:::
