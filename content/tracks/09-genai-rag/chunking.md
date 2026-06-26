---
id: chunking
track: 09-genai-rag
title: "Chunking strategies & context optimization"
badge: HOT
minutes: 10
prereqs: []
tags: [rag, chunking, embeddings, context-window, llm, retrieval]
xp: 60
hot2026: true
---

Your RAG pipeline is live. Docs loaded, vectors indexed, everything looks clean. A user asks: "What's the refund policy for premium subscribers?" — and the bot confidently explains the policy for *free* subscribers. Both policies were on the same page. Your chunker split them at token 512. The retriever grabbed the wrong half.

That's a chunking bug. And it's the number-one silent killer of RAG quality in production.

## What chunking actually does

Every document you index gets sliced into smaller pieces — chunks. Each chunk becomes one vector. When a query arrives, the retriever fetches the chunks with the highest embedding similarity. The LLM only ever sees those chunks. If the right information didn't land in the right chunk, the LLM literally cannot answer correctly, no matter how good your prompt is.

The core tension: LLMs want **large context** (more information = better answers). Embedding models want **small, focused text** (specificity = better retrieval). Chunking is where you navigate that tradeoff.

:::why-prod
Bad chunking cascades. Wrong chunks in → irrelevant context to the LLM → hallucinated or incomplete answers. No amount of reranking, prompt engineering, or model upgrades fully compensates for a broken chunking strategy. Fix the root cause first.
:::

## The main strategies

:::table {title="Chunking strategy comparison"}
| Strategy | How it works | Best for | Watch out for |
|---|---|---|---|
| Fixed-size (token) | Split every N tokens, optional overlap | Quick prototypes, homogeneous prose | Splits mid-sentence, mid-table, mid-code |
| Recursive character | Try `\n\n` → `\n` → `.` → ` ` in order | General docs, README files | Still ignores document-level semantics |
| Structure-aware | Split on Markdown headers, HTML tags, code blocks | Wikis, API docs, Notion exports | Only works if the source format is clean |
| Sentence-based | NLP boundary detection (spaCy, NLTK) | Dense paragraphs, legal text | Tiny chunks = low context for the LLM |
| Semantic chunking | Embed sentences; cut when cosine similarity drops | Mixed-topic docs, research papers | Slower; needs an embedding model at index time |
| Parent-child | Small child chunks for retrieval; parent section for LLM | Policy docs, contracts, long manuals | More complex index and retrieval plumbing |
:::

## Fixed-size chunking — the default trap

Every quickstart tutorial defaults to this: split at 512 tokens, overlap 50. Fast to implement, and it works until it doesn't.

The problem: your documents don't care about your token budget. A JSON schema, a database migration, a pricing table — these have natural boundaries. A fixed-size split blows straight through them. Overlap (repeating N tokens between adjacent chunks) helps continuity but inflates your index and your embedding cost. Keep overlap below 20% of chunk size.

## Recursive splitting — the safe production default

LangChain's `RecursiveCharacterTextSplitter` tries a priority list of separators. It starts with double newlines (paragraph breaks), falls back to single newlines, then sentences, then spaces. It respects natural structure whenever the text has it and degrades gracefully when it doesn't.

```python {title="Recursive splitting + token-aware sizing" run=false}
# pip install langchain-text-splitters tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # same encoder as GPT-4 / text-embedding-3

def token_len(text: str) -> int:
    return len(enc.encode(text))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,          # tokens, not characters
    chunk_overlap=40,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

with open("policy.md") as f:
    raw = f.read()

chunks = splitter.create_documents([raw])
print(f"{len(chunks)} chunks | avg size: {sum(token_len(c.page_content) for c in chunks) // len(chunks)} tokens")
```

One critical detail above: `length_function=token_len`. The default uses `len()` (character count). A single line of Hindi, Chinese, or dense code can be 3–4 tokens per character. Character counts lie — always measure tokens when your budget is a token limit.

## Semantic chunking — when topic drift matters

Instead of splitting on structure, you embed each sentence and watch for a **cosine similarity drop** between adjacent sentences. When the topic shifts, you cut there.

This produces semantically coherent chunks — invaluable when a single document mixes multiple topics (product docs, medical reports, legal contracts). The cost: you run the embedding model twice at index time — once per sentence during chunking, then once per final chunk for storage.

Worth it for mixed-topic corpora. Overkill for clean, structured Markdown.

## Parent-child retrieval — the highest-ROI trick

Small chunks rank well in retrieval because their embedding is precise and uncluttered. But they give the LLM too little context to actually answer well. The fix: store small **child** chunks in the vector index, but return the full **parent** section to the LLM when a child is matched.

LlamaIndex calls this `ParentDocumentRetriever`. You get the precision of small-chunk retrieval plus the context of larger sections. It's one of the biggest quality improvements you can add to an existing RAG system with minimal refactoring.

:::gotcha
Setting `chunk_size=512` and assuming that's 512 tokens is a silent bug. LangChain's default `length_function=len` counts **characters**, not tokens. A 512-character chunk of code or a non-Latin script can easily exceed your embedding model's token limit and get silently truncated. Always pass a real token-counting function.
:::

:::war-story {title="The API schema that got sliced in half"}
A team building an internal developer assistant had chunked their REST API documentation with a naive 512-token fixed splitter. One endpoint's JSON schema happened to fall across a chunk boundary — the field names ended up in chunk 47 and the type definitions started in chunk 48. Depending on how a developer phrased their query, the retriever returned one half or the other. The LLM would invent plausible-sounding field types that simply didn't exist. It passed fluency evals with flying colours. It took three days and a frustrated developer asking "why did it tell me `amount` is a string?" to trace the bug back to the chunker. Switching to Markdown-aware splitting — the docs were already `.md` files — fixed it in one afternoon.
:::

:::interview-line
"Chunking isn't a preprocessing detail — it's a retrieval architecture decision that controls what context your LLM ever gets to see."
:::

:::qa {q="Why does chunk size affect RAG answer quality?"}
Small chunks are precise but low-context — the LLM gets a fragment without enough surrounding information to reason well. Large chunks give context but dilute the embedding signal, hurting retrieval precision. The right size depends on document structure and query type. A good starting point is 300–500 tokens, tuned against evals, not guesswork.
:::

:::qa {q="What is parent-child retrieval and when should you use it?"}
You store small child chunks in the vector index for high-precision retrieval, but when a child chunk matches a query you return its full parent section to the LLM. Use it when your docs have clear section boundaries — policy manuals, contracts, API references — and when a single small chunk consistently lacks enough context for the LLM to give a complete answer.
:::

:::qa {q="When would you choose semantic chunking over recursive character splitting?"}
When your documents mix multiple topics within the same page or section — research papers, medical notes, mixed-content wikis. Recursive splitting cuts at structural boundaries (paragraphs, newlines) which may not align with topic boundaries. Semantic chunking detects topic drift directly via embedding similarity, producing more coherent chunks at the cost of extra compute at index time.
:::

:::drill {type="mcq" q="A RAG bot returns the wrong subscriber tier information even though both tiers are documented in the same file. What is the most likely root cause?"}
- [ ] The embedding model's vector dimension is too small
- [ ] The LLM temperature is set too high
- [x] A fixed-size chunker split the two tier descriptions across separate chunks, so retrieval picks one or the other based on phrasing
- [ ] The vector database index is stale and needs rebuilding
:::

:::drill {type="mcq" q="You are chunking a 300-page technical manual that is already formatted as clean Markdown with H1/H2/H3 heading hierarchy. Which strategy gives the best retrieval quality for the least complexity?"}
- [ ] Fixed 512-token split with 50-token overlap
- [ ] Semantic chunking with cosine similarity thresholds
- [x] Structure-aware splitting on Markdown heading delimiters
- [ ] Sentence-boundary splitting with spaCy
:::

:::key-takeaway
Your chunking strategy IS your retrieval strategy. Start with recursive splitting at 300–500 tokens (measured in real tokens, not chars), upgrade to structure-aware splitting when your docs have clear section boundaries, and add parent-child retrieval when the LLM consistently needs more context than a single chunk provides.
:::
