---
id: rag-pipeline
track: 09-genai-rag
title: "The RAG pipeline: load → chunk → embed → index → retrieve → rerank → generate → cite"
badge: HOT
minutes: 12
prereqs: []
tags: [rag, retrieval, embeddings, llm, genai, pipeline, production]
xp: 60
hot2026: true
---

It's 11 PM in Pune. Your product manager messages on Slack: "The chatbot just told a customer we charge ₹0 for premium plans. That's not in our docs anywhere." You open the logs. The model hallucinated — not because it's bad, but because nobody gave it the right facts at query time.

This is the problem Retrieval-Augmented Generation (RAG) was built to solve. Instead of hoping the LLM memorized your docs during training, you *fetch* the relevant facts at runtime and hand them to the model. No more ₹0 surprises.

This lesson walks the entire pipeline end-to-end so you understand what each stage does, why it's there, and where it breaks in production.

## The eight stages at a glance

Think of RAG as two separate workflows that share an index. The **indexing workflow** runs offline (or on a schedule). The **query workflow** runs live, per user request.

```
Indexing:   raw docs → [LOAD] → [CHUNK] → [EMBED] → [INDEX]
Query:      user question → [EMBED] → [RETRIEVE] → [RERANK] → [GENERATE] → [CITE]
```

Let's walk each step.

## Stage 1 — Load

You have PDFs, Confluence pages, Google Docs, SQL tables, Slack exports. The loader's job is to turn all of them into plain text with metadata (source URL, page number, last-modified timestamp).

Good loaders preserve structure — headings, tables, code blocks. Lossy loaders collapse a 40-column table into a blob of text and your retrieval will never find that pricing row again.

:::why-prod
Source metadata travels with every chunk all the way to the final citation. If you lose it at load time, you can never tell the user *where* the answer came from — and your compliance team will not be happy.
:::

## Stage 2 — Chunk

You split each document into bite-sized pieces. Why? Because LLMs have context limits, and embedding a 200-page PDF as one vector means that vector represents nothing specific.

Chunking strategy is a full lesson on its own (see *Chunking strategies & context optimization*). The key rule here: **a chunk should be the smallest unit of meaning you'd want to retrieve.** A paragraph is usually right. A single sentence is often too small. A full chapter is too big.

## Stage 3 — Embed

Each chunk is passed through an embedding model (e.g. `text-embedding-3-small`, `bge-m3`, `nomic-embed-text`) which produces a dense vector — a list of floats that encodes semantic meaning. Similar chunks land close together in this high-dimensional space.

This is where "semantic search" comes from: you're not matching keywords, you're matching *meaning*.

## Stage 4 — Index

You store all those vectors in a vector database or index — Chroma, FAISS, pgvector, Pinecone, etc. (covered in the Vector Databases lesson). The index lets you search millions of vectors in milliseconds using approximate nearest-neighbour (ANN) algorithms.

You also store the original chunk text and its metadata alongside the vector. The vector is the *key*; the text is the *value*.

## Stage 5 — Retrieve

At query time, you embed the user's question using the **same** embedding model used at index time. Then you ask the index: "Give me the top-K most similar chunks." Typical K is 5–20.

This is your recall step. You'd rather retrieve too much than too little — you can always throw away bad matches later.

## Stage 6 — Rerank

Raw vector similarity is fast but noisy. A cross-encoder reranker reads each (question, chunk) pair and scores how relevant the chunk actually is to the question. It's slower but much more precise.

You pass the top-K retrieved chunks through the reranker, sort by rerank score, and keep only the top-N (e.g. top 5). This is where hybrid search also lives — combining dense (vector) scores with sparse (BM25 keyword) scores before reranking. The reranking lesson goes deep on this.

## Stage 7 — Generate

You stuff the top-N chunks into a prompt and call the LLM:

```
System: You are a helpful assistant. Answer ONLY from the context below.
Context: [chunk 1] [chunk 2] ... [chunk N]
User: {user question}
```

The LLM reads the context and produces an answer grounded in your actual documents. This is the whole point.

## Stage 8 — Cite

A trustworthy RAG system doesn't just answer — it tells the user *which chunks* supported the answer. You ask the LLM to return source IDs alongside its response (structured output works well here), then you render links back to the original documents.

Without citations, users can't verify answers and support teams can't debug wrong answers.

:::table {title="Pipeline stage cheat-sheet"}
| Stage | Runs when | Main failure mode | Quick fix |
|---|---|---|---|
| Load | Offline | Lost metadata, garbled tables | Validate loader output before indexing |
| Chunk | Offline | Chunks too big or too small | Tune size + overlap, test retrieval manually |
| Embed | Offline | Wrong model, mismatch at query time | Lock model version in config |
| Index | Offline | Stale index after doc updates | Incremental upserts keyed by doc hash |
| Retrieve | Online | Low recall (right doc not in top-K) | Raise K, add query expansion |
| Rerank | Online | Slow latency | Cache reranker; reduce K before reranking |
| Generate | Online | Hallucination, prompt stuffing | Add "only use context" instruction, limit tokens |
| Cite | Online | Missing or wrong source links | Return structured JSON with source_ids |
:::

## A minimal end-to-end example

```python {title="Minimal RAG pipeline (ChromaDB + OpenAI-compatible API)" run=false}
# pip install chromadb sentence-transformers openai
# Use any OpenAI-compatible endpoint — e.g. local Ollama, Groq free tier, or OpenAI

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- INDEXING (run once) ---
docs = [
    {"id": "p1", "text": "Premium plan costs ₹999/month.", "source": "pricing.md"},
    {"id": "p2", "text": "Free plan includes 5 projects.", "source": "pricing.md"},
    {"id": "p3", "text": "Enterprise pricing is custom.", "source": "pricing.md"},
]

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")  # free, runs locally

client = chromadb.Client()
collection = client.create_collection("docs")

texts = [d["text"] for d in docs]
vectors = embedder.encode(texts).tolist()

collection.add(
    ids=[d["id"] for d in docs],
    embeddings=vectors,
    documents=texts,
    metadatas=[{"source": d["source"]} for d in docs],
)

# --- QUERY (runs per user request) ---
question = "How much does the premium plan cost?"
q_vector = embedder.encode([question]).tolist()

results = collection.query(query_embeddings=q_vector, n_results=3)
chunks = results["documents"][0]          # list of chunk texts
sources = results["metadatas"][0]         # list of metadata dicts

context = "\n\n".join(f"[{s['source']}] {c}" for c, s in zip(chunks, sources))

# Swap base_url for any OpenAI-compatible provider (Ollama, Groq, etc.)
llm = OpenAI(api_key="sk-...")
response = llm.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer ONLY from the context below.\n\n" + context},
        {"role": "user", "content": question},
    ],
)

print(response.choices[0].message.content)
# → "The premium plan costs ₹999/month."
```

:::gotcha
Never embed the user's question with a different model than you used at index time. A mismatch means your query vector and your chunk vectors live in completely different spaces — similarity scores become meaningless and retrieval silently fails. Lock the embedding model name in a single config constant and read it in both the indexing job and the query handler.
:::

:::war-story {title="The invisible update that poisoned the index"}
A Pune-based fintech team updated their fee schedule PDF on a Friday night. The RAG indexer only ran weekly. All weekend, customer-facing answers quoted the old ₹499 rate while the actual product charged ₹699. Support got 140 tickets before someone noticed the index was stale. The fix was an event-driven re-index triggered by any document upload — not a weekly cron. Cost of lesson: one very bad Monday.
:::

:::interview-line
"RAG is a two-workflow system: an offline indexing pipeline that builds a searchable chunk store, and an online query pipeline that retrieves, reranks, and grounds the LLM's output in your actual documents at runtime."
:::

:::qa {q="Why do we need a reranking step if vector retrieval already gives us semantic similarity?"}
Vector retrieval is an approximation — it finds chunks that are *topically* close to the query, but it can't deeply reason about whether a chunk actually *answers* the question. A cross-encoder reranker reads the full (question, chunk) pair together and produces a much more precise relevance score. The tradeoff is latency, so you rerank only the top-K candidates, not the entire index.
:::

:::qa {q="What happens if the answer to a user's question isn't in any indexed document?"}
Without relevant chunks, the LLM either hallucinates or says "I don't know" — which one depends on your system prompt. You should always instruct the model to say "I couldn't find this in the provided documents" when context is insufficient. A confidence threshold on the reranker score (e.g. discard anything below 0.4) helps detect the no-answer case before you even call the LLM, saving cost and preventing hallucinations.
:::

:::drill {type="mcq" q="At query time, which model should you use to embed the user's question?"}
- [ ] Any high-quality embedding model available at that moment
- [ ] The model with the highest MTEB benchmark score
- [x] The same embedding model used to index the chunks
- [ ] A larger model than the one used at index time for better accuracy
:::

:::drill {type="mcq" q="Your RAG system retrieves 20 chunks (K=20) but you can only fit 5 in the LLM's prompt. What is the BEST next step?"}
- [ ] Summarize all 20 chunks into one before sending to the LLM
- [ ] Randomly pick 5 from the 20 to keep latency low
- [x] Run a reranker over the 20 chunks and pass the top 5 to the LLM
- [ ] Increase the LLM context window until all 20 fit
:::

:::drill {type="mcq" q="A document is updated in your knowledge base. Which RAG component must be refreshed to reflect the change?"}
- [ ] The embedding model weights
- [ ] The LLM system prompt
- [x] The vector index (re-chunk, re-embed, and upsert the updated document)
- [ ] The reranker threshold
:::

:::key-takeaway
RAG grounds LLM answers in your own documents by fetching relevant chunks at query time — the full pipeline is load → chunk → embed → index (offline) then embed → retrieve → rerank → generate → cite (online). Keeping the embedding model consistent across both halves and keeping the index fresh are the two highest-leverage production concerns.
:::
