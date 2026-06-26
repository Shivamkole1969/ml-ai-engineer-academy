---
id: advanced-rag
track: 09-genai-rag
title: "Advanced RAG: Query Rewriting, CRAG, Self-RAG, Agentic RAG"
badge: HOT
minutes: 12
prereqs: []
tags: [rag, query-rewriting, crag, self-rag, agentic-rag, genai, llm]
xp: 60
hot2026: true
---

Your RAG chatbot is live and the PM is happy. Then at 11 PM you get a Slack ping: "The bot told a customer they don't need to pay GST on our SaaS invoice. Is that right?" You pull the logs. The user typed: "do I pay tax on your product?" Your retriever fetched a 2018 FAQ chunk — written before GST rollout — and the LLM trusted it completely. Confident, fluent, wrong.

Basic RAG doesn't know what it doesn't know. Advanced RAG adds that self-awareness.

## Why Naive RAG Breaks in Production

The pipeline from the last lesson — chunk → embed → retrieve → generate — is a solid foundation. But it has three structural blind spots:

1. **Fuzzy queries.** "tax on product" matches chunks about tax *avoidance*, *product returns*, and *tax law history* — not the one you want.
2. **Silent retrieval failure.** If the top-3 chunks are off-topic, the LLM doesn't say "I couldn't find this." It hallucinates from training memory and sounds just as confident.
3. **No verification loop.** Once the model generates, that's it. Nothing checks whether the output actually lines up with what was retrieved.

Each advanced RAG pattern plugs one of these holes.

:::why-prod
In a compliance assistant, support bot, or internal knowledge tool, a confidently wrong answer is worse than "I don't know." These patterns add just enough intelligence to catch and correct failures before they reach the user — without exploding latency or cost.
:::

## The Four Upgrades

### 1. Query Rewriting

Before hitting the vector store, rewrite the user's raw question into better search queries. Two flavors that actually work:

**Multi-query:** Ask an LLM to generate 3–5 rephrasings. Run all of them against your vector store. Union the results, deduplicate by chunk ID. Recall goes up significantly for the cost of one extra LLM call.

**HyDE (Hypothetical Document Embeddings):** Ask the LLM: "What would a document that answers this question look like?" Embed that hypothetical answer. In vector space, a good hypothetical answer sits closer to the real document than the user's question does. Sounds weird — works surprisingly well.

```python {title="Multi-query rewriting — drop-in function" run=false}
import anthropic

# pip install anthropic
# export ANTHROPIC_API_KEY=sk-ant-...
client = anthropic.Anthropic()

def rewrite_query(user_question: str, n: int = 4) -> list[str]:
    """
    Generate n alternative search queries from a raw user question.
    Returns the original + rewrites so you always retrieve on the original too.
    """
    prompt = f"""You are a retrieval optimizer for a search system.
Given the user question below, write {n} distinct search queries that
would help retrieve relevant documents. Vary phrasing and specificity.
Return only the queries, one per line — no numbering, no explanations.

User question: {user_question}"""

    resp = client.messages.create(
        model="claude-opus-4-5",   # check docs.anthropic.com for latest model IDs
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    rewrites = [q.strip() for q in resp.content[0].text.strip().splitlines() if q.strip()]
    return [user_question] + rewrites[:n]   # original first, then rewrites


# Example
queries = rewrite_query("do I pay tax on your product?")
# → [
#     "do I pay tax on your product?",          # original
#     "GST applicability on SaaS subscription India",
#     "indirect tax software service B2B invoice 2024",
#     "is cloud software taxable under GST India",
#     "GST rate for online software services India"
# ]
# Now run each query through your vector store, union chunks, deduplicate by ID.
# Cap at 8–12 total chunks before passing to the LLM.
```

### 2. CRAG — Corrective RAG

After retrieval, score each chunk for relevance using a lightweight evaluator — an LLM judge or a cross-encoder. If every chunk scores below a threshold (say, 0.5), don't force the LLM to generate from garbage context.

Instead: fall back to a web search, widen the search scope, or return a controlled "I couldn't find a reliable answer" message. The key insight is that the retrieval step can fail silently — CRAG makes that failure visible and actionable.

### 3. Self-RAG

Self-RAG teaches the LLM to emit **reflection tokens** during generation:

- `[Retrieve]` or `[No Retrieve]` — "do I even need to look something up here?"
- `[Relevant]` or `[Irrelevant]` — "is this retrieved chunk actually useful?"
- `[Supported]` or `[Contradicted]` — "does my answer align with what I retrieved?"

You won't retrain a model for most projects. But you can implement the same logic in your orchestration layer: a judge LLM call that checks each chunk before generation, and another that verifies the output is grounded. Same philosophy, no finetuning needed.

### 4. Agentic RAG

Instead of a fixed retrieve → generate step, an agent loop runs:

1. **Plan** — what exactly do I need to find?
2. **Retrieve** — call the vector store (or multiple tools)
3. **Evaluate** — is the context sufficient and accurate?
4. **Re-retrieve** — if not, refine the query and try again
5. **Synthesize** — generate with verified, complete context

This is the pattern behind LangGraph's agent modules. The LLM orchestrates its own research. Latency climbs, but for multi-hop questions — "compare our Q1 and Q3 policy docs and flag contradictions" — it's the only pattern that structurally works.

:::table {title="Which pattern to reach for"}
| Pattern | Core Problem Solved | When to Use | Latency Overhead |
|---|---|---|---|
| Query Rewriting | Fuzzy / ambiguous queries | Almost always — low cost | +1 LLM call |
| CRAG | Stale or irrelevant retrieved chunks | Compliance, legal, support bots | +1 judge call |
| Self-RAG | No verification of generated output | High-stakes, factual generation | +2–3 LLM calls |
| Agentic RAG | Multi-hop, complex research questions | Analyst tools, async pipelines | +3–10x total |
:::

:::gotcha
Multi-query rewriting multiplies your vector store hits — 5 query variants at top-5 each gives you 25 chunks before deduplication. Without a hard cap, you'll blow up your context window and your invoice. Always deduplicate by chunk ID first, then cap at 8–12 chunks regardless of how many queries ran. More context is not always better; irrelevant padding hurts generation quality.
:::

:::war-story {title="The GST incident that killed a fintech demo"}
A Pune-based fintech had a polished RAG bot over their loan and tax docs. During a live investor demo, the founder asked: "Is prepayment allowed on personal loans?" The retriever fetched a home-loan prepayment chunk — similar embedding neighbourhood, completely wrong document type. The LLM generated a fluent, confident wrong answer. Silence in the room. The fix was a CRAG relevance filter: score each chunk against the original query with a lightweight judge, drop anything below 0.6, show a fallback message otherwise. Two hours to ship. The check should have been in v1.
:::

:::interview-line
"We stack advanced RAG in priority order: query rewriting is always on, CRAG kicks in when retrieval confidence is low, and we only reach for agentic loops on multi-hop questions where single-shot retrieval structurally can't work."
:::

:::qa {q="What is the difference between CRAG and Self-RAG?"}
CRAG is a pipeline-level pattern: after retrieval, a separate evaluator scores chunk relevance and triggers a fallback if context quality is too low. Self-RAG is a model-level pattern: the LLM itself emits reflection tokens to decide whether to retrieve, whether a chunk is relevant, and whether its own output is grounded. CRAG is easy to add to any pipeline; Self-RAG requires a specially trained model, though you can approximate its behaviour with orchestration-layer judge calls.
:::

:::qa {q="When should you NOT use Agentic RAG?"}
When latency and cost matter more than completeness. An agentic loop can run 5–10 LLM calls for a single user query — acceptable for an async research pipeline, unacceptable for a real-time support chat targeting sub-2-second responses. Start with query rewriting, add a CRAG relevance gate, and only reach for agentic patterns when the question type genuinely requires multi-hop reasoning that single-shot retrieval cannot resolve.
:::

:::drill {type="mcq" q="A user asks your RAG bot: 'What were the major regulatory changes in our sector between 2022 and 2024?' Retrieval returns three chunks — all from a 2019 policy document. Which pattern is BEST suited to detect and correct this?"}
- [ ] HyDE query rewriting, because a hypothetical answer would embed closer to 2022–2024 documents
- [x] CRAG, because a relevance evaluator would score these chunks as low-confidence and trigger a fallback or broader retrieval
- [ ] Self-RAG, because the LLM would emit a [Contradicted] token on its own output
- [ ] Multi-query rewriting, because more queries always guarantee higher recall
:::

:::drill {type="mcq" q="You enable multi-query rewriting: the original query plus 4 rewrites, each fetching top-5 chunks from the vector store. How many chunks could you be passing to the LLM before deduplication?"}
- [ ] 5
- [ ] 10
- [ ] 20
- [x] 25
:::

:::key-takeaway
Advanced RAG is a stack, not a single choice: query rewriting improves what you ask for, CRAG filters what you get back, Self-RAG verifies what you generate, and Agentic RAG loops until the answer is actually grounded. Add them in that order — each one compounds the one before.
:::
