## ⏱️ The 60-minute battle plan

You have one hour before the interview. Don't read everything — read **this**. It's the 80/20 of
what Pune / remote-India GenAI interviews actually probe. Work top to bottom.

:::table {title="Your hour, blocked out"}
| Time | Focus | If you remember nothing else |
|---|---|---|
| 0–8 min | The numbers you can recite | 16·N, KV cache, 0.95ⁿ, O(n²) |
| 8–20 min | LLM + serving | KV cache gates concurrency; continuous batching; TTFT vs TPOT |
| 20–35 min | RAG (the headline) | load→chunk→embed→retrieve→rerank→generate→cite |
| 35–45 min | Fine-tuning | knowledge→RAG, behavior→LoRA |
| 45–53 min | Agents & MCP | bound the loop; errors compound; MCP = USB-C for tools |
| 53–60 min | Production, cost, the one-liners | architecture beats wording; cache before you quantize |
:::

---

## 1 · The numbers you must be able to recite

These come up *constantly*. Memorize the four.

:::table {title="Cheat numbers"}
| Number | What it means | The line |
|---|---|---|
| **16 · N bytes** | Full training VRAM (Adam, mixed precision) | "7B full train ≈ 112GB — doesn't fit one 80GB GPU" |
| **2 · L · H · d · seq · batch · dtype** | KV cache size | "The cache, not the weights, gates concurrency" |
| **0.95¹⁰ ≈ 60%** | Agent compounding error | "Ten 95%-reliable steps → a coin flip" |
| **O(n²)** | Attention cost in sequence length | "Double the context, 4× the attention matrix" |
| **~4 chars/token** | Rough token estimate (English) | "Code and non-English run denser" |
:::

> Inference is cheap: fp16 ≈ **2N**, int8 ≈ **N**, int4 ≈ **0.5N** (+ KV cache on top).

---

## 2 · LLM foundations & serving (the engine room)

:::interview-line
"Weights are a fixed cost; the KV cache grows with sequence length × concurrency — that's why GQA,
PagedAttention, and KV-quant exist."
:::

- **Tokenization** — the model sees tokens, not characters. Weird bugs (counting, math, JSON) often start here.
- **KV cache** — cut it with **GQA** (share K/V across heads), **PagedAttention** (vLLM, no wasted memory), **KV-quant**, **prefix caching** (shared system prompts).
- **Two latencies**: **TTFT** (time to first token — prefill, parallel) vs **TPOT** (time per output token — decode, sequential). Streaming hides TTFT; TPOT sets the felt speed.
- **Batching**: static → dynamic → **continuous (in-flight)**. Continuous batching refills finished slots → near-100% GPU use. *This is why vLLM wins.*
- **Quantization**: PTQ (fast, post-hoc — GPTQ/AWQ) vs QAT (train-aware). INT8 safe, INT4 needs eval. **Re-run your real eval, not perplexity.**

---

## 3 · RAG — the headline skill

The pipeline, in one breath:

:::table {title="The RAG pipeline"}
| Stage | What & the gotcha |
|---|---|
| **Load** | Parse docs; PDFs/tables are where quality dies |
| **Chunk** | size + overlap; semantic / parent-doc beats naive splits |
| **Embed** | embedding model choice is your #1 retrieval lever |
| **Index** | vector DB: FAISS/Chroma (local, free) · pgvector · Pinecone/Weaviate/Milvus (scale) |
| **Retrieve** | top-k by cosine / ANN (HNSW, IVF-PQ) |
| **Rerank** | hybrid: BM25 + dense, then a cross-encoder rerank |
| **Generate** | stuff context, ask for **citations** |
| **Evaluate** | RAGAS: faithfulness, answer relevancy, context precision/recall |
:::

**Advanced (name-drop these):** query rewriting · **CRAG** (grade retrieved docs, fall back to web) · **Self-RAG** (model critiques its own retrieval) · **Agentic RAG** (a planner decides what to retrieve) · **GraphRAG** (Neo4j/Neptune, multi-hop).

```python {title="Free + local RAG in 6 lines (FAISS)" run=false}
# pip install langchain langchain-community faiss-cpu sentence-transformers
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_text(big_text)
store = FAISS.from_texts(chunks, emb)
docs = store.similarity_search("How does the KV cache grow?", k=4)  # feed as grounded context + ask for citations
```

:::interview-line
"RAG fixes a knowledge gap and stays fresh; fine-tuning fixes a behavior gap. The best systems do both."
:::

### Mini system design — production RAG

<svg viewBox="0 0 820 90" width="100%" role="img" aria-label="RAG flow: query to retriever to reranker to LLM to cited answer">
  <rect x="6" y="28" width="96" height="40" rx="9" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="54" y="52" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Query</text>
  <line x1="102" y1="48" x2="132" y2="48" stroke="#8b7bff" stroke-width="2"/><polyline points="126,42 134,48 126,54" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <rect x="134" y="28" width="120" height="40" rx="9" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="194" y="52" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Vector DB</text>
  <line x1="254" y1="48" x2="284" y2="48" stroke="#8b7bff" stroke-width="2"/><polyline points="278,42 286,48 278,54" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <rect x="286" y="28" width="120" height="40" rx="9" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="346" y="52" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Reranker</text>
  <line x1="406" y1="48" x2="436" y2="48" stroke="#8b7bff" stroke-width="2"/><polyline points="430,42 438,48 430,54" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <rect x="438" y="28" width="120" height="40" rx="9" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="498" y="52" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">LLM</text>
  <line x1="558" y1="48" x2="588" y2="48" stroke="#8b7bff" stroke-width="2"/><polyline points="582,42 590,48 582,54" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <rect x="590" y="28" width="150" height="40" rx="9" fill="none" stroke="#48e0a0" stroke-width="1.5"/>
  <text x="665" y="52" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Cited answer</text>
</svg>

---

## 4 · Fine-tuning — one decision, then LoRA

:::table {title="Prompt vs RAG vs Fine-tune"}
| Symptom | Fix |
|---|---|
| "I just haven't asked clearly" | **Prompt** (always try first) |
| "It doesn't know our facts / they change" | **RAG** |
| "Wrong format/tone/skill, can't prompt around it" | **Fine-tune (LoRA)** |
:::

- **LoRA** = freeze base, train tiny low-rank adapters (<1% params). Swappable per task.
- **QLoRA** = 4-bit frozen base + LoRA → fine-tune a 7B on a free Colab/Kaggle T4.
- **Don't fine-tune in facts** — it learns *style*, not reliable *knowledge*. Knowledge → RAG.

```python {title="QLoRA skeleton" run=false}
# pip install transformers peft trl bitsandbytes accelerate datasets
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
SFTTrainer(model=model, train_dataset=ds, peft_config=lora, dataset_text_field="text", max_seq_length=1024).train()
```

---

## 5 · Agents & MCP

:::interview-line
"Errors compound: ten 95%-reliable steps give ~60% end-to-end success. The fix isn't a smarter model
per step — it's fewer steps that must all go right, plus validation and retries between them."
:::

- **The loop**: reason → call tool → observe → loop or stop. Each part breaks differently.
- **Survive production**: bound the loop, make tools idempotent, validate tool output, human-in-loop for irreversible actions, trace everything.
- **Frameworks**: **LangGraph** (stateful graphs, control) · **CrewAI** (role-based crews) · **AutoGen** (conversational multi-agent).
- **MCP (Model Context Protocol)** — the open standard for connecting tools/resources to models: clients, servers, tools/resources. Think "USB-C for LLM tools" — write a tool server once, any MCP client can use it.

```python {title="Minimal LangGraph agent (bounded)" run=false}
# pip install langgraph
from langgraph.graph import StateGraph, END
def reason(s): ...   # decide: call a tool or finish
def act(s): ...      # run the tool, append observation
g = StateGraph(dict)
g.add_node("reason", reason); g.add_node("act", act)
g.add_conditional_edges("reason", lambda s: "act" if s["steps"] < 6 else END)  # <-- cap the loop
g.add_edge("act", "reason"); g.set_entry_point("reason")
agent = g.compile()
```

---

## 6 · Production, monitoring & cost

- **Security principle**: *architecture beats wording.* Least privilege, isolate untrusted input, gate irreversible actions. You can't prompt your way out of prompt injection.
- **Deployment ladder**: shadow → canary → progressive → rollback. Model rollback is harder than code (data + weights).
- **Never alert on averages** — percentiles don't average. Watch p99; the tail is the product.
- **Cost order of operations**: **measure → cache → compress → route → quantize → distill.** Caching (exact / semantic / prefix) is the cheapest big win.

:::interview-line
"I measure first, then cache, then route small-vs-big models, and only then reach for quantization or
distillation — cheapest, safest wins first."
:::

---

## 7 · The lines that land (steal these)

- "The KV cache, not the weights, gates concurrency."
- "Knowledge gap → RAG; behavior gap → fine-tune; try a better prompt first."
- "Continuous batching is why vLLM gets the throughput."
- "Errors compound — reduce the number of steps that must all go right."
- "Architecture beats wording for LLM security."
- "Don't trust perplexity after quantization — re-run the real eval."
- "Leakage is anything at train time you won't have at predict time."
- "Calibration matters when the *score* drives the decision, not just the rank."

---

## 8 · Pune / remote-India reality check

- **PyTorch ≫ TensorFlow** at product firms. **SQL is table-stakes** — expect a live query.
- Talk about **shipping & owning** a system, not notebooks. Bring **3 deployed projects** (pipeline, RAG app, MLOps).
- GenAI is in **40%+ of ML JDs** and carries a **20–40% premium**. ~30% of roles are remote-friendly.

:::key-takeaway
If you only nail four things: the **four numbers**, the **RAG pipeline**, the **RAG-vs-fine-tune decision**,
and **why agent errors compound** — you'll cover the bulk of what a 2026 GenAI interview throws at you.
:::
