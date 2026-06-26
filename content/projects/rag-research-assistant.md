# Production RAG App — Equity Research Q&A with Citations

## Scenario / Objective

It is 10 PM in Pune. An analyst at a brokerage has 40 PDF initiations-of-coverage sitting in her inbox. Her manager wants a sector view by 9 AM. She opens your app, uploads the PDFs, types _"What is the bear case for Reliance Jio according to Kotak and CLSA?"_ — and gets a cited, grounded answer in 8 seconds, with source page numbers she can verify.

That is the product you are going to build.

**What it does:**

- Ingests equity research PDFs (broker reports, earnings transcripts, sector notes)
- Chunks them intelligently, embeds them, stores in a vector DB
- At query time, retrieves relevant chunks, re-ranks them, and generates a grounded answer with `[Source: Kotak_Jio_2024.pdf, p.12]` citations
- If retrieval quality is low, the system triggers a **Corrective RAG (CRAG)** loop — it rewrites the query and tries again, or optionally falls back to a web search
- Ships as a Streamlit UI, deployed on HuggingFace Spaces (free)
- Evaluated rigorously with **RAGAS** so you can quote real numbers in an interview

**Résumé lines this directly backs:**
- "Financial Analysis Assistant (Agentic + Corrective RAG)"
- "Estimates Extractor"

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                      │
│  PDFs ──► PyMuPDF ──► Recursive Chunker ──► Embedding Model   │
│                                               (BGE / ada-002)  │
│                                                     │           │
│                                               FAISS / Chroma   │
│                                               (Vector Store)   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                        QUERY PIPELINE (CRAG Loop)              │
│                                                                │
│  User Query                                                    │
│      │                                                         │
│      ▼                                                         │
│  [1] Retrieve top-K chunks (FAISS ANN search)                  │
│      │                                                         │
│      ▼                                                         │
│  [2] Relevance Grader (LLM-as-judge, structured output)        │
│      │                                                         │
│      ├── PASS ──► [3] Reranker (cross-encoder / Cohere)        │
│      │                 │                                       │
│      │                 ▼                                       │
│      │           [4] Generator (GPT-4o / Mistral)             │
│      │                 │                                       │
│      │                 ▼                                       │
│      │           Answer + Citations                            │
│      │                                                         │
│      └── FAIL ──► [5] Query Rewriter  ──► loop back to [1]    │
│                    (or web-search fallback via Tavily)         │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              Streamlit UI  ──►  HuggingFace Spaces
              (upload, chat, source viewer)
```

The CRAG loop is the differentiator. Vanilla RAG retrieves once and hopes. CRAG asks "was that any good?" before generating — and retries if not. This is what makes the résumé line believable.

---

## Repo Structure

```text
rag-research-assistant/
├── app/
│   ├── main.py                  # Streamlit entry point
│   ├── chat.py                  # Chat interface + session state
│   └── source_viewer.py         # Side-panel PDF page viewer
├── rag/
│   ├── ingest.py                # PDF → chunks → embeddings → vector store
│   ├── retriever.py             # FAISS / Chroma retrieval wrapper
│   ├── reranker.py              # Cross-encoder reranking (or Cohere API)
│   ├── grader.py                # LLM relevance grader (structured output)
│   ├── rewriter.py              # Query rewriter for CRAG fallback
│   ├── generator.py             # Final answer generation with citations
│   └── crag_pipeline.py         # Orchestrates the full CRAG loop
├── eval/
│   ├── test_questions.json      # Golden Q&A pairs for RAGAS
│   ├── ragas_eval.py            # RAGAS evaluation script
│   └── results/                 # CSV / JSON eval outputs
├── data/
│   └── sample_reports/          # 2-3 sample PDFs (public broker notes)
├── scripts/
│   └── build_index.py           # One-shot index build CLI
├── tests/
│   ├── test_ingest.py
│   ├── test_retriever.py
│   └── test_crag_pipeline.py
├── .env.example
├── requirements.txt
├── Dockerfile                   # Optional — for local reproducibility
└── README.md
```

---

## Milestone Checklist

### M1 — Ingestion (Day 1–2)
- [ ] Set up project with `uv` / `poetry`, commit `requirements.txt`
- [ ] Parse PDFs with `PyMuPDF` (fitz), preserve page numbers in metadata
- [ ] Implement `RecursiveCharacterTextSplitter` with `chunk_size=800, overlap=150`
- [ ] Generate embeddings with `BAAI/bge-small-en-v1.5` (free, runs locally)
- [ ] Persist FAISS index to disk; write `build_index.py` CLI
- [ ] Unit test: ingest 1 PDF, assert chunk count and metadata keys present

### M2 — Basic Retrieval + Generation (Day 3–4)
- [ ] FAISS similarity search returning top-5 chunks with scores
- [ ] Prompt template: system prompt + context block with inline citations (`[1]`, `[2]`…)
- [ ] Generator calls OpenAI / Groq / Mistral, returns answer + sources list
- [ ] Streamlit MVP: upload sidebar + single-turn Q&A working locally
- [ ] Unit test: query returns answer containing at least one citation tag

### M3 — Reranking (Day 5)
- [ ] Integrate `cross-encoder/ms-marco-MiniLM-L-6-v2` for local reranking
- [ ] OR swap in Cohere Rerank API (free tier is generous)
- [ ] Compare MRR@5 before vs after reranking on your sample questions
- [ ] Add reranker toggle to Streamlit sidebar

### M4 — CRAG / Agentic Loop (Day 6–8)
- [ ] Build `grader.py`: LLM call that returns `{"relevant": true/false, "reason": "..."}` using structured output / JSON mode
- [ ] Build `rewriter.py`: takes original query + grader reason, returns improved query
- [ ] Wire `crag_pipeline.py`: retrieve → grade → (rewrite & retry OR proceed to generate)
- [ ] Set max retries = 2 to avoid infinite loops
- [ ] Add optional Tavily web-search fallback when both retrieval attempts fail
- [ ] Log each loop iteration (query, grade, action) for debugging
- [ ] Unit test: seed a deliberately bad query, assert pipeline retried at least once

### M5 — RAGAS Evaluation (Day 9–10)
- [ ] Write 20 golden Q&A pairs in `eval/test_questions.json`
- [ ] Run RAGAS: `faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`
- [ ] Document scores in `eval/results/baseline.json`
- [ ] After reranking: re-run, compare delta — you should see +5–10 pp on context metrics
- [ ] Add a one-line eval summary to your README (these numbers go in your interview)

### M6 — UI Polish + Deployment (Day 11–12)
- [ ] Multi-turn chat with `st.session_state` message history
- [ ] Source panel: clicking a citation shows the PDF page snippet
- [ ] Upload multiple PDFs, merge indexes on the fly
- [ ] Add `requirements.txt` HF-compatible (no CUDA-only deps)
- [ ] Push to HuggingFace Space (`gradio` or `streamlit` SDK)
- [ ] Set secrets (API keys) in HF Space settings, not in code

### M7 — Estimates Extractor (Bonus, Day 13–14)
- [ ] Structured extraction: given a page, extract `{ticker, target_price, recommendation, EPS_FY25, EPS_FY26}`
- [ ] Use LLM structured output / Pydantic model for extraction
- [ ] Build a comparison table view in Streamlit: multiple broker estimates side by side
- [ ] This directly maps to the "Estimates Extractor" résumé bullet

---

## Key Code Snippets

### 1. Ingestion with page-level metadata

```python
# rag/ingest.py
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_chunk(pdf_path: str) -> list[dict]:
    """Extract text from each page, keep page number in metadata."""
    doc = fitz.open(pdf_path)
    chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": pdf_path.split("/")[-1],
                    "page": page_num,
                },
            })
    return chunks


def build_index(pdf_paths: list[str], index_dir: str = "faiss_index"):
    """Build and persist FAISS index from a list of PDFs."""
    all_chunks = []
    for path in pdf_paths:
        all_chunks.extend(load_and_chunk(path))

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(index_dir)
    print(f"Index built: {len(texts)} chunks saved to {index_dir}/")
```

:::why-prod
Page numbers in metadata are not optional — they are what make citations verifiable. Without them your "RAG with citations" claim is hollow. An interviewer will ask "can I click through to the source?" and the answer must be yes.
:::

---

### 2. Relevance Grader (the CRAG brain)

```python
# rag/grader.py
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()  # reads OPENAI_API_KEY from env

class GradeResult(BaseModel):
    relevant: bool
    reason: str   # surfaced in logs — crucial for debugging

GRADER_PROMPT = """\
You are grading whether retrieved context actually answers the question.
Be strict — if the context is vaguely related but does not contain the answer, mark it not relevant.

Question: {question}

Retrieved context:
{context}

Respond with JSON: {{"relevant": true/false, "reason": "..."}}
"""

def grade_retrieval(question: str, chunks: list[dict]) -> GradeResult:
    context = "\n\n---\n\n".join(
        f"[{i+1}] ({c['metadata']['source']}, p.{c['metadata']['page']})\n{c['page_content']}"
        for i, c in enumerate(chunks)
    )
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",          # cheap, fast — use for grading
        messages=[
            {"role": "user", "content": GRADER_PROMPT.format(
                question=question, context=context
            )}
        ],
        response_format=GradeResult,
    )
    return response.choices[0].message.parsed
```

:::gotcha
Use `gpt-4o-mini` (or Groq llama-3.1-8b) for grading — not your expensive generation model. Grading is a yes/no classification task. You do not need GPT-4o for it. This keeps costs under $1 for a full eval run.
:::

---

### 3. CRAG Pipeline

```python
# rag/crag_pipeline.py
import logging
from rag.retriever import retrieve
from rag.grader import grade_retrieval
from rag.rewriter import rewrite_query
from rag.reranker import rerank
from rag.generator import generate_answer

logger = logging.getLogger(__name__)
MAX_RETRIES = 2

def run_crag(question: str, vectorstore, k: int = 8) -> dict:
    """
    Corrective RAG loop:
      1. Retrieve
      2. Grade
      3. If bad → rewrite query and retry (up to MAX_RETRIES)
      4. Rerank
      5. Generate
    Returns: {"answer": str, "sources": list[dict], "iterations": int}
    """
    current_query = question

    for attempt in range(1, MAX_RETRIES + 2):   # +2 so last attempt still runs
        logger.info(f"CRAG attempt {attempt} | query: {current_query!r}")

        chunks = retrieve(vectorstore, current_query, k=k)
        grade = grade_retrieval(question, chunks)  # always grade against ORIGINAL question

        logger.info(f"Grade: relevant={grade.relevant} | {grade.reason}")

        if grade.relevant or attempt > MAX_RETRIES:
            # Proceed even on last attempt — best effort
            break

        # Rewrite and loop
        current_query = rewrite_query(
            original_query=question,
            failed_query=current_query,
            reason=grade.reason,
        )

    top_chunks = rerank(question, chunks, top_n=4)
    answer, sources = generate_answer(question, top_chunks)

    return {
        "answer": answer,
        "sources": sources,
        "iterations": attempt,
        "final_query": current_query,
    }
```

---

### 4. Generator with inline citations

```python
# rag/generator.py
from openai import OpenAI

client = OpenAI()

SYSTEM_PROMPT = """\
You are a financial research assistant. Answer questions using ONLY the provided context.
Cite every factual claim with [N] where N is the source number.
If the context does not contain enough information, say so explicitly — do not hallucinate.
"""

def generate_answer(question: str, chunks: list[dict]) -> tuple[str, list[dict]]:
    context_block = ""
    sources = []
    for i, chunk in enumerate(chunks, start=1):
        src = chunk["metadata"]
        label = f"{src['source']}, p.{src['page']}"
        context_block += f"\n[{i}] ({label})\n{chunk['page_content']}\n"
        sources.append({"id": i, "file": src["source"], "page": src["page"]})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,   # low temp for factual finance Q&A
    )
    return response.choices[0].message.content, sources
```

---

### 5. RAGAS Evaluation

```python
# eval/ragas_eval.py
import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from rag.crag_pipeline import run_crag
from rag.retriever import load_vectorstore

def run_eval(questions_path: str = "eval/test_questions.json"):
    with open(questions_path) as f:
        qas = json.load(f)  # [{question, ground_truth}, ...]

    vectorstore = load_vectorstore("faiss_index")
    rows = []

    for qa in qas:
        result = run_crag(qa["question"], vectorstore)
        rows.append({
            "question": qa["question"],
            "answer": result["answer"],
            "contexts": [c["page_content"] for c in result.get("top_chunks", [])],
            "ground_truth": qa["ground_truth"],
        })

    dataset = Dataset.from_list(rows)
    scores = evaluate(dataset, metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    ])
    print(scores)
    scores.to_pandas().to_csv("eval/results/ragas_scores.csv", index=False)

if __name__ == "__main__":
    run_eval()
```

:::key-takeaway
RAGAS gives you four numbers. Write them down. In your interview you will say: "Faithfulness was 0.87, context precision was 0.74 — after adding reranking, precision went to 0.82. Here is the commit." That is the difference between a résumé claim and a portfolio claim.
:::

---

### 6. Estimates Extractor (Structured Output)

```python
# rag/estimates_extractor.py
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Optional

client = OpenAI()

class BrokerEstimate(BaseModel):
    broker: str
    ticker: str
    recommendation: str                   # BUY / HOLD / SELL
    target_price: Optional[float]
    eps_fy25: Optional[float] = Field(None, description="EPS estimate for FY2025")
    eps_fy26: Optional[float] = Field(None, description="EPS estimate for FY2026")
    upside_pct: Optional[float]

EXTRACT_PROMPT = """\
Extract broker estimates from this text. Return structured data only.
If a field is not mentioned, return null.

Text:
{text}
"""

def extract_estimates(page_text: str, broker_name: str) -> BrokerEstimate:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": EXTRACT_PROMPT.format(text=page_text)}
        ],
        response_format=BrokerEstimate,
    )
    result = response.choices[0].message.parsed
    result.broker = broker_name  # inject from filename
    return result
```

---

## What to Show in an Interview

When a hiring manager or a senior engineer says "walk me through your RAG project", hit these beats in order. Ten minutes, no fumbling.

**1. The problem first, not the tech.**
"Analysts spend 2 hours manually skimming 15 PDFs for a sector comparison. This app does it in under 10 seconds with citations."

**2. The CRAG loop on a whiteboard.**
Draw retrieve → grade → (rewrite → retry) → rerank → generate. Explain why vanilla RAG fails on financial documents (tables, dense numbers, cross-report comparisons that span documents).

**3. Your RAGAS numbers.**
"Faithfulness is 0.87 — which means the model almost never fabricates. Context precision improved from 0.63 to 0.82 after adding the cross-encoder reranker." Quote the delta, not just the score.

**4. A live demo.**
Upload the included sample PDF. Ask: "What is the 12-month target price for Infosys according to this report?" Show the citation panel. Show the source page.

**5. One thing that broke and how you fixed it.**
"Chunks from financial tables were garbage — the splitter cut across table rows. I added a table-detection heuristic in `ingest.py` that pads table regions with extra overlap." Real problems make the project real.

**6. Cost and latency.**
"Ingestion is a one-time cost. Per query: 1 embedding call + grader call (4o-mini) + reranker + generator. End-to-end ~3s, ~$0.004 per query."

---

## Honest Talking Points

These are the things that will trip you up if you skip them, and the things that will impress people if you have thought about them.

**What works well:**
- CRAG meaningfully reduces hallucination on straightforward factual questions (target prices, recommendations, dates)
- Reranking with a cross-encoder is a cheap, high-ROI improvement — one import, measurable lift
- RAGAS lets you compare pipeline variants with real metrics, not vibes

**What does not work well (yet):**
- Financial tables inside PDFs are a genuine unsolved problem. PyMuPDF extracts them as prose; the numbers come out garbled. The proper fix is a table-extraction model (Camelot, or a vision model for scanned PDFs).
- Multi-document reasoning ("how does Kotak's view differ from CLSA's?") is hard. The retriever pulls chunks, not cross-document summaries. You need a second-stage synthesis step.
- The CRAG grader itself can hallucinate false negatives — it marks good context as irrelevant. It needs its own eval.
- HF Spaces free tier spins down. Cold starts take ~30s. Warn the demo audience.

**Honest cost estimate:**
- OpenAI: ~$2–5 to build and eval the whole thing if you use 4o-mini for grading/extraction
- Cohere free tier covers 1,000 rerank calls/month
- HF Spaces: free, but no GPU for embeddings — stick to `bge-small-en-v1.5` (runs on CPU in ~200ms/chunk)

---

## How This De-Fakes a Résumé Claim

The industry is full of "built a RAG chatbot" bullets that mean "I ran LangChain's default example on Wikipedia". Interviewers know. Here is how this build separates you.

| Résumé claim | What the interviewer probes | What your build gives you |
|---|---|---|
| "Agentic RAG" | "What does the agent actually decide?" | Grader → rewriter loop with logged iterations |
| "Corrective RAG" | "How do you detect bad retrieval?" | LLM grader with structured output + reason field |
| "Citations" | "Can I trace back to the source page?" | Metadata-anchored chunks, source panel in UI |
| "Reranking" | "Why not just take top-5 from vector search?" | Cross-encoder MRR numbers before vs after |
| "Evaluated with RAGAS" | "What were your scores?" | `eval/results/ragas_scores.csv` in the repo |
| "Estimates Extractor" | "What does the schema look like?" | `BrokerEstimate` Pydantic model, live extraction demo |

The differentiator is not the code — it is that you can explain why each component exists and quote numbers from a real eval run. That is what a senior engineer at a brokerage or a fintech will care about. The code is just the proof.

:::interview-line
"I didn't just add reranking because tutorials say to. I measured context precision at 0.63 without it, added the cross-encoder, and it went to 0.82. That delta is in the eval results folder — here is the commit."
:::
