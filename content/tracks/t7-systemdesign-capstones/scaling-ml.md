---
id: scaling-ml
track: t7-systemdesign-capstones
title: "Scaling: caching, autoscaling, sharding"
badge: HOT
minutes: 9
prereqs: []
tags: [scaling, caching, autoscaling, sharding, system-design, production, inference]
xp: 60
hot2026: true
---

Picture this: your recommendation model has been quietly humming along for a month. Latency is clean, stakeholders are happy, and then — a top streamer goes live with your product and shouts it out. Traffic spikes 12× in eight minutes. Your model starts timing out. Errors cascade. The on-call engineer (maybe you) is scrambling at 2 a.m. wondering why nobody "just added more servers."

This lesson is your prep for that conversation — in the design interview and in production.

## Why three levers, not one

Most engineers reach for "just scale horizontally" the moment they see traffic. That's fine — but it's expensive and slow if it's your *only* move. Real production systems layer three complementary strategies:

1. **Caching** — serve a stored answer instead of computing it again
2. **Autoscaling** — add or remove compute based on real-time load
3. **Sharding** — split data or models so no single node is a bottleneck

Each lever attacks a different root cause. The trick is knowing which one to pull first.

:::why-prod
Model inference is expensive per call. Caching cuts the number of calls that actually hit the model. Autoscaling handles bursty arrival rates. Sharding prevents any single node from becoming the ceiling. Used together, they let you serve 100× traffic without 100× cost.
:::

## Lever 1 — Caching

The fastest inference is the one you never run.

In ML systems, caching lives at multiple layers:

- **Input-level cache**: same query or same image seen before? Return the stored result. Works great for search and recommendation where popular items repeat constantly.
- **Embedding cache**: computing a text embedding is cheaper than a full model forward pass but still costs GPU cycles. Cache the vector for recurring inputs (product descriptions, user bios) in Redis with a reasonable TTL.
- **Feature cache**: if your serving pipeline computes the same user features 50 times a second, cache them in-memory or in a fast key-value store. Stale-by-one-second is almost always fine for a recommendation.

Cache hits should appear in your metrics dashboard. A 40% hit rate on an embedding cache means 40% of your GPU budget back in your pocket.

:::table {title="Cache type vs use case"}
| Cache Layer | What you store | Good fit |
|---|---|---|
| Response cache | Final model output | Repeated identical queries |
| Embedding cache | Encoded vectors | Text / image encoders |
| Feature cache | User or item features | High-QPS personalization |
| KV cache (LLM) | Attention key-value pairs | Long-context generation |
:::

:::gotcha
**Cache invalidation bites everyone.** If you update your model weights, stale cached outputs from the old model will mix with fresh outputs from the new one. Always version your cache keys (e.g., `rec:v3:<user_id>`) and flush on model update. Mixing outputs silently degrades recommendation quality and is notoriously hard to debug.
:::

## Lever 2 — Autoscaling

Autoscaling means the infrastructure grows and shrinks with demand automatically, so you're not paying for 100 GPUs at 3 a.m. on a Tuesday.

The key insight: **scale on the right signal.** CPU utilization is a lagging indicator for GPU-heavy inference. Scale on:

- **Request queue depth** — requests waiting to be served is the most direct signal
- **GPU utilization** — relevant when the bottleneck is actual compute
- **P95 latency** — scale out before users notice slowness

Kubernetes Horizontal Pod Autoscaler (HPA) handles CPU/memory natively. For custom metrics (queue depth, GPU %) you need KEDA or a custom metrics adapter.

```python {title="Pseudocode: autoscaler logic" run=false}
# What an autoscaler is doing under the hood — simplified
import time

TARGET_QUEUE_DEPTH = 10  # desired queue depth per replica
SCALE_COOLDOWN_SEC = 60  # avoid thrashing

def recommend_replicas(current_queue_depth: int, current_replicas: int) -> int:
    # Classic proportional scaling formula (HPA uses this too)
    desired = (current_queue_depth / TARGET_QUEUE_DEPTH) * current_replicas
    desired = max(1, min(desired, MAX_REPLICAS))  # clamp
    return round(desired)

# In practice: Kubernetes HPA with a custom metrics adapter reads
# your Prometheus queue_depth metric and calls this math automatically.
# Run locally: just mock current_queue_depth and print desired replicas.
```

One production trap with autoscaling: **model cold start latency**. A new pod with a 4 GB model takes 45–90 seconds to be ready. Autoscaling adds pods *reactively* — so you're already slow by the time help arrives. Solutions: keep a warm pool of spare pods, or pre-warm on predicted spikes (e.g., sports event start times).

## Lever 3 — Sharding

Sharding splits data (or model) so that no single machine owns everything.

In ML systems you'll encounter two flavours:

**Data sharding** (feature stores and vector indexes): your embedding index might hold 500 million product vectors. It doesn't fit on one machine. Shard by consistent hash of item ID — each shard owns a slice of the key space. A router maps every lookup to the right shard. This is how Faiss clusters and Pinecone partitions under the hood.

**Model sharding** (large models): if the model itself won't fit on one GPU, you split layers across GPUs (pipeline parallelism) or split weight tensors across GPUs (tensor parallelism). This is the daily reality for serving 70B+ parameter models.

:::war-story {title="The shard that ate all the memory"}
A team sharded their vector index across 8 nodes — but forgot to shard the metadata store that held product attributes. Every recommendation request hit the single metadata node for enrichment. At 3× baseline traffic that node maxed out RAM and started swapping. Latency went from 30 ms to 4 seconds. The fix took ten minutes once identified (add read replicas), but the incident lasted four hours because everyone assumed the model was the bottleneck. Lesson: map every component in your data path, not just the GPU.
:::

:::interview-line
"Caching reduces work, autoscaling absorbs burst, sharding removes ceilings — I layer all three because each one fails alone at different traffic shapes."
:::

:::qa {q="How do you decide when to cache vs when to scale out?"}
Cache first — it's the cheapest lever and often solves 60–80% of load spikes if your traffic has any repeated patterns. Only add replicas when either the cache hit rate is low (highly unique inputs) or when latency is breaching SLA even on cache misses. Autoscaling handles the residual burst that caching can't absorb.
:::

:::qa {q="An interviewer asks you to design a vector search service that must handle 50k QPS. Where does sharding fit?"}
Start with the data size: if the embedding index is under ~10M vectors it might fit on one machine with replication for throughput. Beyond that, shard by consistent hash of item ID across N nodes, put a stateless router in front, and scale replicas per shard independently. Quote the shard count formula: total_qps / (qps_per_node * replication_factor).
:::

:::drill {type="mcq" q="Your LLM serving pod takes 80 seconds to start. Traffic is spiking. What's the best mitigation?"}
- [ ] Increase the HPA scale-up cooldown window so pods aren't wasted
- [ ] Reduce model precision to make the pod start faster
- [x] Maintain a warm pool of pre-loaded spare pods so scale-out is instant
- [ ] Switch from horizontal to vertical scaling
:::

:::drill {type="mcq" q="You cache model outputs by input hash. After a model update, users are getting a mix of old and new predictions. Root cause?"}
- [ ] The cache TTL is too short
- [ ] Autoscaling added too many replicas
- [x] Cache keys don't include the model version, so old outputs are still being served
- [ ] Sharding routed requests to the wrong node
:::

:::key-takeaway
Cache to avoid work, autoscale to absorb burst, shard to remove hard ceilings — and always version your cache keys when you ship a new model.
:::
