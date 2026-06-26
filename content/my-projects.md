Your own projects, in interview-ready shape. Each one: a **30-second snapshot**, a **STAR story**
to tell out loud, and the **questions interviewers actually ask** with crisp answers. Skim before any
interview — this is what turns "I built a thing" into "I owned a system."

---

## 1 · Financial Analyst + SQL Chatbot

> 🔗 [Live on HF](https://huggingface.co/spaces/shivamkole1969/Financial-Analyst) · natural-language → SQL over financial databases.

:::table {title="30-second snapshot"}
| | |
|---|---|
| **Use case** | Let non-technical analysts query MS SQL Server financial data in plain English — no T-SQL needed |
| **Stack** | Node.js / Express · Vanilla JS · `mssql` (MS SQL Server) · Groq + OpenAI LLMs · localStorage |
| **Your role** | Designed & built end-to-end: schema introspection, prompt design, query execution, UI |
| **The hard part** | Making the LLM produce *correct, executable* T-SQL across many databases without hallucinating |
:::

**Situation** — Analysts needed answers from financial SQL Server databases but couldn't write T-SQL, so every data request bottlenecked on an engineer.

**Task** — Build a chatbot that turns plain-English questions into correct, runnable T-SQL across multiple databases — safely and resiliently.

**Action** — Express backend introspects the schema across **all** databases via `INFORMATION_SCHEMA.COLUMNS` using `Promise.allSettled` (so a permission error on one DB doesn't break the rest), builds a compact global schema map, and injects it into the LLM system prompt. The model is constrained to return **only** a T-SQL block; a regex extracts it, the backend auto-runs it and renders results as a table with the query tucked under an accordion. Added **multi-LLM key rotation** (Groq → OpenAI) with automatic failover on rate limits; raw `SELECT`/`WITH` typed by power users bypasses the LLM entirely.

**Result** — Non-technical users self-serve analytics in plain English; the system stays up through provider rate limits; shipped as a portable desktop app.

:::interview-line
"I grounded the model in the real schema and forced it to emit only T-SQL — hallucination drops because it can only reference columns that actually exist, and bad guesses fail loudly at execution."
:::

:::qa {q="How do you stop the LLM from hallucinating table/column names?"}
Schema grounding: I inject the real `INFORMATION_SCHEMA` into the prompt so it can only reference columns that exist, and I constrain it to return *only* T-SQL (no prose). Anything wrong fails at execution rather than returning fake data.
:::

:::qa {q="How do you handle multiple databases and permission errors?"}
`Promise.allSettled` queries every database's schema concurrently — if one DB throws a permissions error, the others still succeed, and I merge them into one global schema map. No single failure blocks the whole connection.
:::

:::qa {q="Isn't auto-running generated SQL a security risk? How would you harden it?"}
Yes — it's the honest weak point. For production I'd connect with a **read-only** account, allow-list statement types (only `SELECT`/`WITH`), validate/parse the query before running, add row limits and timeouts, and log every executed statement. The architecture already separates generation from execution, which makes adding that gate clean.
:::

:::qa {q="Why the multi-key rotation?"}
A single provider's rate limit shouldn't take the app down. Keys rotate Groq → OpenAI automatically on a 429; the backend stays stateless (keys travel in the request body, nothing stored server-side).
:::

---

## 2 · Estimates Data Extractor

> 🔗 [Live on HF](https://huggingface.co/spaces/shivamkole1969/Data_Extractor) · broker PDFs → standardized Excel.

:::table {title="30-second snapshot"}
| | |
|---|---|
| **Use case** | Extract financial estimates from broker research reports (PDF/Excel) into one standardized Excel |
| **Brokers** | EGR · TAS · HAY · RJ (Raymond James) · UBS Global — each with its own layout |
| **Stack** | Python / Flask · PyMuPDF (fitz) · pdfplumber · Pandas · OpenPyXL · Docker on HF |
| **The hard part** | Every broker formats reports differently; tables in PDFs are messy and inconsistent |
:::

**Situation** — Analysts manually copied estimates out of dozens of broker PDFs into Excel — slow, error-prone, and every broker laid out their numbers differently.

**Task** — Automate extraction into a single standardized Excel, handling broker-specific layouts, from drag-dropped files or a URL.

**Action** — Flask app parses PDFs with **PyMuPDF** (text) and **pdfplumber** (tables); **broker-specific mapping modules** normalize each format to a common schema; **Pandas + OpenPyXL** assemble the standardized Excel. Added drag-drop + URL ingestion with real-time progress; Dockerized and deployed on HF Spaces.

**Result** — Cuts manual extraction from minutes-per-report to seconds, with consistent output across all five brokers and far fewer transcription errors.

:::interview-line
"I kept extraction deterministic — broker-specific parsers, not an LLM — because the data is structured tables where I need cheap, fast, auditable, repeatable results, not a model that might silently change a number."
:::

:::qa {q="PDFs are messy — how do you handle different broker layouts?"}
A parsing layer (PyMuPDF for text, pdfplumber for tables) separated from **per-broker mapping modules** that know each report's structure and map it to a common schema. Adding a new broker = adding a mapper, not touching the core.
:::

:::qa {q="Why not just use an LLM to extract everything?"}
For structured tables, deterministic parsing is cheaper, faster, fully auditable, and can't hallucinate a number — which matters a lot for financial data. I'd only bring an LLM in as a fallback for genuinely unstructured prose sections.
:::

:::qa {q="How do you know the extraction is correct?"}
Field/schema validation against the expected estimate fields per broker, flagging missing or out-of-range values, and spot-checking against the source PDF. The standardized schema makes mismatches obvious.
:::

---

## 3 · Support Desk AI — Hybrid MLOps/LLMOps

> 🔗 [Live on HF](https://huggingface.co/spaces/shivamkole1969/support-desk-ai) · cheap local model + LLM, fully observed. *Your strongest "can you ship & operate it" story.*

:::table {title="30-second snapshot"}
| | |
|---|---|
| **Use case** | Customer-support replies: classify intent locally, refine with an LLM, monitor everything |
| **Stack** | FastAPI · scikit-learn (SVM/TF-IDF) · Gemini 2.5 Flash · Langfuse · Evidently AI · DVC · MLflow · Docker |
| **Headline result** | **~70% fewer LLM tokens** by handling known intents with a cheap local classifier |
| **The hard part** | Operating it: cost control, drift monitoring, retraining — not just the model |
:::

**Situation** — An LLM-for-everything support bot is expensive and a black box; most tickets are known intents that don't need a frontier model.

**Task** — Build a cost-efficient, observable hybrid pipeline that's production-operable.

**Action** — A local **SVM/TF-IDF** classifier handles known intents; **Gemini 2.5 Flash** refines the wording only when needed. **Langfuse** traces cost/tokens/prompts; **Evidently AI** monitors data & target drift (JSD/KS); **DVC + MLflow** version data and retrain the classifier; all served from a **Dockerized FastAPI** on HF.

**Result** — ~70% token-cost reduction on known intents, real-time observability, drift alerts, and a one-click retrain path.

:::interview-line
"The model was the easy part — the value was operating it: route cheap-vs-expensive by intent, trace every token with Langfuse, and watch for drift with Evidently so it doesn't rot silently."
:::

:::qa {q="Why a hybrid local + LLM design?"}
Cost and latency. A tiny SVM/TF-IDF classifier answers known intents for almost nothing; the LLM is reserved for cases that actually need generation. That routing is where the ~70% token saving comes from.
:::

:::qa {q="How do you monitor an LLM system in production?"}
Three layers: **Langfuse** for prompt traces, token spend and latency; **Evidently AI** for data/target **drift** (JSD/KS statistics); and **MLflow** for experiment tracking and model versioning. Alerts fire when drift crosses a threshold.
:::

:::qa {q="When and how do you retrain?"}
When Evidently flags drift or enough new labeled tickets accumulate. DVC versions the dataset, MLflow tracks the run, and the classifier is retrained and promoted — so retraining is reproducible, not a manual notebook.
:::

:::key-takeaway
For every project, lead with the **problem and the result**, then the **one hard decision** you made (schema-grounding the SQL, deterministic parsing over LLM, hybrid routing for cost) — that's the senior signal interviewers listen for.
:::
