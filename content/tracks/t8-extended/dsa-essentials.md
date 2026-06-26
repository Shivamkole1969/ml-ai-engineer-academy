---
id: dsa-essentials
track: t8-extended
title: "DSA essentials for ML roles"
badge: CORE
minutes: 9
prereqs: []
tags: [dsa, algorithms, complexity, interviews, python, data-structures]
xp: 45
hot2026: false
---

Your recommendation engine just went live. The product team is thrilled — until they notice that fetching the top-10 products for each user is taking 400 ms and climbing as the catalog grows. You dig in and find the culprit: a `.sort()` call on 2 million items every single request. Swap it for a heap and you're done in 8 ms. The fix takes four lines. Knowing *which* four lines is the DSA part.

ML interviews are famous for whiteboard questions, but the real reason to know DSA is this: it shows up constantly in production ML — feature stores, retrieval pipelines, deduplication jobs, graph-based data loaders. You don't need competitive-programming heroics. You need the handful of patterns that appear again and again.

## The patterns ML engineers actually use

Most ML work reduces to a small set of recurring problems. Here they are, mapped to the structures that solve them.

:::why-prod
Slow algorithms compound: a function that's "fast enough" in dev will melt a production service under real load or real data volume. Understanding complexity lets you catch these before deployment, not after your on-call page at 2 a.m.
:::

:::table {title="DSA patterns mapped to ML tasks"}
| Pattern | Data Structure | ML use case |
|---|---|---|
| Top-K items | Min-heap (`heapq`) | Top-K recommendations, approximate nearest neighbours |
| Fast lookup | Hash map / dict | Feature stores, entity resolution, vocabulary look-up |
| Sliding window | Deque (`collections.deque`) | Rolling feature windows, time-series lag features |
| Order-preserving dedup | Ordered set / sort + pointer | Deduplication before training, ranking merge |
| Graph traversal (BFS/DFS) | Adjacency list | Knowledge graphs, dependency resolution in pipelines |
| Binary search | Sorted list / `bisect` | Searching buckets, quantile feature binning |
:::

### Big-O in one paragraph

Big-O describes how runtime grows as input size `n` grows — not the exact time. O(1) means constant (a dict lookup). O(log n) means it barely grows (binary search). O(n) means linear (one pass). O(n log n) means sort-level (acceptable). O(n²) means watch out (nested loops on large data). When your dataset is 10 M rows, an O(n²) step is 100 trillion operations — that's not running in this lifetime.

### The heap trick for top-K

Sorting an array of size n to get the top K items costs O(n log n). A min-heap does it in O(n log K) — much faster when K is small (say, K=10 but n=1 million). Python's `heapq.nlargest` handles this automatically.

```python {title="Top-K with a min-heap" run=false}
import heapq

# Simulate a scored candidate list (item_id, score)
# In production this might come from an ANN index or a ranker model
candidates = [(f"item_{i}", round(i * 0.01, 2)) for i in range(1_000_000)]

# O(n log K) — far cheaper than sorted(...) which is O(n log n)
top_k = heapq.nlargest(10, candidates, key=lambda x: x[1])

# Run locally: just `python` — no GPU, no installs beyond stdlib
print(top_k)
```

### The sliding window for time-series features

Feature engineering on sequences often needs a rolling statistic — average price over the last 7 days, max event count in a 5-minute window. A naive nested loop is O(n·w) where w is the window width. A deque that evicts old elements gives you O(n).

```python {title="O(n) sliding max with a deque" run=false}
from collections import deque

def sliding_max(values: list[float], window: int) -> list[float]:
    """Return the running max over `window` elements. O(n)."""
    dq: deque[int] = deque()  # stores indices, largest first
    result = []
    for i, v in enumerate(values):
        # Remove indices that are out of the window
        while dq and dq[0] < i - window + 1:
            dq.popleft()
        # Remove indices whose values are smaller than current (can never be max)
        while dq and values[dq[-1]] < v:
            dq.pop()
        dq.append(i)
        result.append(values[dq[0]])
    return result

# Run: python dsa_essentials.py — stdlib only
print(sliding_max([3, 1, 5, 2, 8, 4], window=3))
# → [3, 5, 5, 5, 8, 8]
```

### Graph traversal in ML pipelines

Data dependencies in an ML pipeline are a directed acyclic graph (DAG). If you ever build a custom feature dependency resolver or a mini orchestrator, BFS/DFS is exactly what you reach for. Tools like Airflow and Prefect do this internally — understanding the concept lets you debug scheduling issues and circular dependency errors without panic.

:::gotcha
Defaulting to a Python list as a "set" for deduplication. `if x in my_list` is O(n) — it scans every element. `if x in my_set` is O(1). On a 1 M-row dedup job, this is the difference between 2 seconds and 30 minutes. Always use `set` or `dict` for membership checks.
:::

:::interview-line
"I always ask: what's the data size and do I need top-K, membership, or ordering? That tells me heap, hash map, or sort — and then the complexity writes itself."
:::

:::qa {q="When would you use a heap instead of sorting in an ML system?"}
When you only need the top K results out of N candidates and K is much smaller than N. Sorting costs O(n log n) but a heap gives top-K in O(n log K). In a recommendation pipeline serving K=20 results from a million candidates, the heap is orders of magnitude faster and also lets you stream results without materialising the full sorted list.
:::

:::qa {q="What's the time complexity of looking up a user's features in a feature store backed by a hash map?"}
O(1) average case — a hash map (Python dict) computes a hash of the key and jumps directly to the bucket. This is why feature stores are implemented on hash-based storage like Redis or DynamoDB rather than sorted arrays. The trade-off is higher memory usage and O(n) worst-case in pathological hash collision scenarios, which managed stores avoid by design.
:::

:::drill {type="mcq" q="You need to find the 50 highest-scoring items from a list of 5 million candidates as fast as possible. Which approach has the best time complexity?"}
- [ ] Sort the full list descending and take the first 50 — O(n log n)
- [x] Use a min-heap of size 50, iterate once through candidates — O(n log 50)
- [ ] Use a hash map keyed by score — O(n) build, O(n) scan for top-50
- [ ] Binary search the list after sorting — O(log n) per query
:::

:::drill {type="mcq" q="A feature-engineering job runs a membership check (`if user_id in seen`) inside a loop over 2 million rows. `seen` is a Python list. What is the overall complexity of this loop?"}
- [ ] O(n) — one pass through the data
- [ ] O(n log n) — typical for sorting-based dedup
- [x] O(n²) — each `in` check on a list is O(n), done n times
- [ ] O(1) — Python dicts make `in` constant time
:::

:::key-takeaway
You don't need to ace LeetCode Hard to be a great ML engineer — but you do need heap for top-K, hash map for O(1) lookup, deque for sliding windows, and a working sense of Big-O so you can spot the n² trap before it reaches production.
:::
