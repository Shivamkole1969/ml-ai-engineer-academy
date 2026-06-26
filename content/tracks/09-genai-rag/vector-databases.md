---
id: vector-databases
track: 09-genai-rag
title: "Vector databases: FAISS, Chroma, pgvector, Pinecone, Weaviate/Milvus"
badge: HOT
minutes: 11
prereqs: []
tags: [vector-db, rag, faiss, chroma, pgvector, pinecone, weaviate, milvus, hnsw, ann]
xp: 60
hot2026: true
---

Your RAG demo crushed it. The intern clapped. Your manager said "ship it."

Now the requirements land in your inbox: 500 concurrent users, 5 million product docs, filters like "only from Q3 2024, category = Compliance." Oh, and it needs to stay under 200ms p95.

Your current setup: a FAISS index file living on your laptop.

This lesson is your map for picking — and using — the right vector store for each stage of the journey.

## What a vector database actually does

An embedding is just a dense float array — say, 1536 numbers — that encodes semantic meaning. A vector database stores those arrays and lets you ask: "given this query vector, find me the K most similar ones."

But a *database* also needs things a raw index can't do: survive restarts, serve concurrent reads, filter by metadata ("author = legal, date > 2024-01-01"), and scale out when a single machine isn't enough.

Not all vector stores are created equal. Some are libraries. Some are extensions. Some are managed services. Picking the wrong one early means a painful migration later.

:::why-prod
Re-indexing millions of documents (re-embedding, reloading, reindexing) takes hours and often requires downtime. The vector store you prototype with will likely be the one you're stuck migrating away from under deadline pressure. Choose based on your scale and ops budget *before* you write the first indexing script.
:::

## The five flavors — and when to use each

:::table {title="Vector Store Comparison"}
| Tool | Type | Scale sweet spot | Metadata filter | Ops burden |
|---|---|---|---|---|
| FAISS | In-process library | < 5M, offline / batch | DIY only | None — it's a library |
| Chroma | Embeddable / light server | < 1M vectors | Built-in | Very low |
| pgvector | Postgres extension | < 10M (tuned) | Full SQL | Low (if you have PG) |
| Pinecone | Managed cloud SaaS | 10M → Billions | Built-in | Near-zero |
| Weaviate / Milvus | Self-hosted OSS cluster | 100M → Billions | Built-in | High (needs K8s ops) |
:::

### FAISS — raw speed, no fuss

FAISS (Facebook AI Similarity Search) is a C++ library with a Python wrapper. It is not a database. It's a search engine you hold in RAM (or mmap from disk) and manage entirely yourself.

**Index types you'll hit:**
- **IndexFlatL2** — exact brute-force search. Use for < 100K vectors or ground-truth benchmarking.
- **IVF_FLAT** — clusters vectors into buckets first, then searches only the nearest buckets. Fast, needs a training step.
- **HNSW** — builds a hierarchical graph. Excellent recall + speed tradeoff. Default choice for most FAISS production use cases.

```python {title="FAISS HNSW — build, query, persist" run=false}
# pip install faiss-cpu        (free; swap for faiss-gpu if you have CUDA)
import faiss
import numpy as np

DIM = 1536          # e.g. text-embedding-3-small or ada-002 output dimension

# --- Build ---
index = faiss.IndexHNSWFlat(DIM, 32)   # M=32 neighbours per node (tune: 16–64)
index.hnsw.efConstruction = 200        # higher = better quality, slower build

vecs = np.random.rand(10_000, DIM).astype("float32")   # your chunk embeddings
index.add(vecs)                         # HNSW builds incrementally — safe to add in batches

# --- Query ---
query = np.random.rand(1, DIM).astype("float32")
distances, ids = index.search(query, k=5)
print(ids)   # e.g. [[4231  892 7103  56  301]]

# --- Persist (FAISS never auto-saves!) ---
faiss.write_index(index, "chunks.faiss")
loaded = faiss.read_index("chunks.faiss")
```

FAISS has no metadata filtering. You get back integer IDs — then you look up your own metadata store (a dict, SQLite, anything) with those IDs. That DIY join is the price of its speed.

### Chroma — the sensible local default

Chroma is the "works out of the box, no server required" pick for prototyping and small production deployments. It wraps HNSW under the hood, persists to disk, and gives you Python-native metadata filtering in one call. When you outgrow it (usually past 1M vectors with heavy concurrent writes), you move to something more serious.

### pgvector — underrated, if you live in Postgres

pgvector adds a `vector` column type to Postgres and an HNSW or IVF index on top. The killer feature: SQL joins. You can write `WHERE doc.date > '2024-01-01' AND doc.team = 'legal'` right inside your vector query. Zero new infra, zero new ops pattern — just an extension your DBA already trusts.

It's massively underused by teams that reflexively reach for a new service instead. Under ~5–10M vectors it's competitive on latency and far simpler operationally.

### Pinecone — managed, zero ops

Pinecone is the "we have budget and zero infra bandwidth" answer. You push vectors to a managed endpoint; they handle sharding, replication, index maintenance, and keep p99 under 10ms at 100M+ vectors. The bill grows fast at scale — but so does your engineering time if you self-host.

### Weaviate / Milvus — self-hosted at serious scale

Both are open-source, Kubernetes-native, and built for billions of vectors without a managed bill. Weaviate has a friendlier developer experience (GraphQL API, built-in hybrid search). Milvus has higher raw throughput and more tuning knobs. Both need a dedicated DevOps investment — don't pick either if nobody on your team has Helm chart experience.

## How to choose in 30 seconds

1. **Weekend POC?** → Chroma (embedded mode, `pip install chromadb`, done)
2. **Already on Postgres, < 5M vectors?** → pgvector
3. **Managed service, money not the bottleneck?** → Pinecone
4. **Self-hosted at scale, DevOps team exists?** → Milvus or Weaviate
5. **Offline batch pipeline / no serving?** → FAISS

:::gotcha
HNSW uses a lot of RAM — the index graph alone can hit 100–200+ bytes per vector dimension in edge cases. For 10M vectors at 1536 dims you can blow past 50 GB just on the index, before your actual data. At large scale, use IVF_PQ (product quantization) or Milvus's DiskANN index: you trade ~5% recall for a 5–10x memory saving. Always profile memory before sizing your instances.
:::

:::war-story {title="The 40-minute ghost chatbot"}
A Pune-based SaaS team ran their RAG support bot on a flat FAISS index holding 2M vectors. Worked fine. Then a background job kicked off a full re-index after a model upgrade — sequential rebuild, no hot-swap, 40 minutes of zero results in production. Their support chatbot returned empty responses the entire time. The fix: migrate to Milvus, which supports adding vectors while serving reads, and implement a build-then-swap index rotation. Add-while-serving is a first-class feature in every real vector database. FAISS doesn't have it.
:::

:::interview-line
"We started with pgvector — zero new infra, full SQL filtering. When we crossed 8M vectors and p95 crept past 180ms, we moved the hot namespace to Pinecone and kept the archival data in Postgres."
:::

:::qa {q="When would you choose pgvector over Pinecone?"}
pgvector shines when your payload is already in Postgres, your vector count is under a few million, and you need complex metadata filters or joins — you just use SQL. You avoid introducing a new service, and your team already knows how to operate Postgres. Pinecone wins when you need zero-ops managed scaling, have no infra team to maintain a cluster, or need to search across hundreds of millions of vectors with consistent low latency and no index tuning.
:::

:::qa {q="What is HNSW and why is it the dominant vector index in production?"}
HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest neighbour algorithm. It builds a multi-layer graph where upper layers act as "highways" for fast coarse navigation, and lower layers store fine-grained local neighbourhoods. At query time it enters at the top, descends greedily, and returns approximate nearest neighbours in sub-millisecond time with 95%+ recall. That recall-vs-speed tradeoff beats tree-based and hash-based alternatives at the scales RAG systems operate at — which is why FAISS, pgvector, Chroma, Pinecone, and Weaviate all use it.
:::

:::drill {type="mcq" q="Your team has 80M document chunks, a DevOps engineer, a cost cap, and wants no cloud vendor lock-in. What's the right call?"}
- [ ] FAISS with a large in-memory EC2 instance
- [ ] Chroma in server mode behind a load balancer
- [ ] pgvector with connection pooling
- [x] Milvus or Weaviate deployed on Kubernetes
:::

:::drill {type="mcq" q="You're building a quick prototype with 60,000 vectors and need exact (not approximate) similarity results for benchmarking. Which FAISS index type should you use?"}
- [ ] IndexHNSWFlat
- [x] IndexFlatL2
- [ ] IndexIVFPQ
- [ ] IndexIVFFlat with nlist=256
:::

:::key-takeaway
There is no universally best vector store. Start with Chroma or pgvector locally, graduate to Pinecone (managed) or Milvus/Weaviate (self-hosted) when scale demands it. The switch costs you hours of re-indexing — so pick based on your *next* 12 months of scale, not your current demo.
:::
