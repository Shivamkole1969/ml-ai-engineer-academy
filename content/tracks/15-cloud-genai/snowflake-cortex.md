---
id: snowflake-cortex
track: 15-cloud-genai
title: "Snowflake Cortex: LLM functions, Cortex Search, Cortex Analyst, Snowpark"
badge: CORE
minutes: 9
prereqs: []
tags: [snowflake, cortex, llm, sql, rag, cortex-search, cortex-analyst, snowpark]
xp: 45
hot2026: false
---

Picture this: your company's entire customer data — purchase history, support tickets, product reviews — already lives in Snowflake. The analytics team asks you to add an AI feature: "Can we automatically tag every support ticket with a sentiment score and a one-line summary?" The old path was painful — export the data, spin up a Python service, call an LLM API, write the results back. Data egress costs, latency, infra to babysit, security reviews. Three weeks of work.

With Snowflake Cortex, you write one SQL query. Done in an afternoon.

## What Is Snowflake Cortex?

Cortex is Snowflake's managed AI layer — a set of capabilities that bring LLM inference, semantic search, and text-to-SQL *directly inside* the Snowflake warehouse. Your data never leaves Snowflake. The models run on Snowflake's compute, billed in Snowflake credits.

Four things to know:

1. **LLM Functions** — call hosted LLMs from SQL
2. **Cortex Search** — hybrid semantic + keyword search over your Snowflake tables
3. **Cortex Analyst** — natural language → SQL → result (text-to-SQL with a semantic layer)
4. **Snowpark** — run Python/Scala/Java and ML workloads natively on Snowflake compute

:::why-prod
If your data already lives in Snowflake, using Cortex means zero data movement, no separate vector store to provision, no extra API keys to rotate, and one bill. In regulated industries (BFSI, healthcare), keeping data inside your existing Snowflake trust boundary is often the only acceptable path.
:::

## 1. LLM Functions — AI in a SELECT Statement

Cortex ships a handful of SQL functions that wrap hosted LLMs. You call them like any built-in function.

```python {title="Cortex LLM functions in SQL (run in Snowflake worksheets — free trial at snowflake.com)" run=false}
-- Sentiment: returns a float between -1 (negative) and 1 (positive)
SELECT
    ticket_id,
    SNOWFLAKE.CORTEX.SENTIMENT(ticket_text) AS sentiment_score
FROM support_tickets;

-- Summarise long text to a few sentences
SELECT
    ticket_id,
    SNOWFLAKE.CORTEX.SUMMARIZE(ticket_text) AS short_summary
FROM support_tickets;

-- Call any hosted model with a custom prompt (mistral-large, llama3-70b, etc.)
SELECT
    product_id,
    SNOWFLAKE.CORTEX.COMPLETE(
        'mistral-large2',
        CONCAT('Classify this review as BUG, FEATURE_REQUEST, or PRAISE. Reply with only one word.\n\n', review_text)
    ) AS label
FROM product_reviews;

-- Extract a specific answer from context (great for RAG-lite)
SELECT
    SNOWFLAKE.CORTEX.EXTRACT_ANSWER(
        'What is the refund policy?',
        policy_text
    ) AS answer
FROM company_policies;
```

The models available include `snowflake-arctic`, `llama3.1-70b`, `mistral-large2`, and others — the list grows with every Snowflake release. You pick the model per call, so you can use a cheap small model for classification and a bigger one for summarisation.

:::table {title="Cortex LLM functions at a glance"}
| Function | What it does | Typical use |
|---|---|---|
| `COMPLETE(model, prompt)` | Full chat completion, any prompt | Classification, extraction, generation |
| `SUMMARIZE(text)` | Returns a concise summary | Ticket/doc summarisation |
| `SENTIMENT(text)` | Float -1 to 1 | Review scoring, NPS signals |
| `TRANSLATE(text, src, tgt)` | Language translation | Multilingual support pipelines |
| `EXTRACT_ANSWER(q, ctx)` | Extracts answer span from context | Simple QA over docs |
:::

## 2. Cortex Search — Semantic Search Over Your Tables

Cortex Search is a managed search service you create on top of a Snowflake table column. It indexes the column using both dense vector embeddings and BM25 keyword scoring (hybrid search), so "return policy" and "can I get my money back" both return the same policy document.

You create it once with a DDL statement:

```sql {title="Create a Cortex Search service" run=false}
CREATE OR REPLACE CORTEX SEARCH SERVICE support_docs_search
  ON body_text                     -- column to search
  ATTRIBUTES category, doc_date   -- filterable metadata columns
  WAREHOUSE = my_wh
  TARGET_LAG = '1 hour'           -- how fresh the index stays
  AS (
    SELECT body_text, category, doc_date FROM support_documents
  );
```

Then query it in SQL or via the REST API:

```sql {title="Query the search service" run=false}
SELECT PARSE_JSON(
    SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
        'support_docs_search',
        '{"query": "how do I cancel my subscription", "limit": 5}'
    )
) AS results;
```

This is Snowflake's answer to Pinecone or pgvector — no separate vector DB to manage.

## 3. Cortex Analyst — "Chat With Your Data" for the Non-Technical Stakeholder

Cortex Analyst turns a plain-English question into a SQL query against your tables — and it actually works reliably because you give it a **semantic model**: a YAML file describing your tables, columns, metrics, and business logic (e.g., "revenue means SUM of order_amount where status = 'completed'").

The flow:

1. You write a `semantic_model.yaml` describing your schema in business terms.
2. Cortex Analyst gets the user's question + your semantic model.
3. It returns SQL + the result set.

Your BI team can then embed this in a Streamlit app or Slack bot. No rogue SQL. No hallucinated column names. The semantic model is the guardrail.

## 4. Snowpark — Python Inside Snowflake

Snowpark lets you write Python (or Scala, or Java) that runs *on Snowflake's compute nodes*, not on your laptop or a separate server. This matters for ML workloads where the alternative is exporting millions of rows.

Key capabilities:

- **Snowpark DataFrames** — a PySpark-like API that pushes computation into Snowflake SQL (lazy evaluation, no data egress)
- **Python UDFs / UDTFs** — package a scikit-learn model or any Python function and call it from SQL
- **Snowpark ML** — a sklearn-compatible API with preprocessing, feature engineering, and model registry baked in
- **Snowpark Container Services** — run arbitrary Docker containers (e.g., a FastAPI inference server) inside Snowflake's network boundary

```python {title="Snowpark DataFrame — pushdown computation, no data leaves Snowflake" run=false}
# pip install snowflake-snowpark-python
from snowflake.snowpark import Session

session = Session.builder.configs({
    "account": "your_account",
    "user": "your_user",
    "password": "your_password",
    "warehouse": "MY_WH",
    "database": "MY_DB",
    "schema": "PUBLIC",
}).create()

df = session.table("support_tickets")

# Filter and aggregate — translated to SQL, runs in Snowflake
result = (
    df.filter(df["status"] == "open")
      .group_by("category")
      .agg({"ticket_id": "count"})
      .sort("count(ticket_id)", ascending=False)
)

result.show()  # Only NOW does data come back to Python
```

:::gotcha
Cortex LLM functions are charged in **Snowflake credits per 1M tokens**, not in separate API costs. On the free trial (30-day, $400 credits) you can experiment freely — but in production, a large COMPLETE() call over millions of rows can drain credits fast. Always test on a LIMIT 100 sample first, and consider running batch Cortex jobs during off-peak to benefit from lower warehouse costs.
:::

:::interview-line
"With Snowflake Cortex I can run LLM inference, hybrid semantic search, and text-to-SQL entirely inside the warehouse — data never leaves Snowflake, so there's no egress cost, no extra infra, and no new trust boundary to secure."
:::

:::qa {q="When would you choose Cortex Search over a standalone vector database like Pinecone?"}
If your source data already lives in Snowflake and your team is Snowflake-native, Cortex Search wins on simplicity: no ETL pipeline to keep a separate DB in sync, no additional vendor contract, no new auth layer. The trade-off is that Cortex Search is tightly coupled to Snowflake — if you need cross-platform or ultra-low-latency real-time indexing (sub-second freshness), a dedicated vector DB gives you more control.
:::

:::qa {q="What is the semantic model in Cortex Analyst and why does it exist?"}
The semantic model is a YAML file that maps business terms — like 'revenue', 'active users', 'churn' — to the exact SQL expressions they represent in your schema. Without it, an LLM generating SQL would guess at column names and business logic, producing wrong or inconsistent queries. The semantic model is the single source of truth that makes Cortex Analyst reliable enough to expose to non-technical stakeholders.
:::

:::qa {q="How does Snowpark differ from just running Python locally against Snowflake with a connector?"}
With a regular connector, your Python code fetches data to your machine, processes it locally, and writes results back — large data transfers, local compute limits, data egress. Snowpark pushes the computation into Snowflake: the DataFrame operations are compiled to SQL and run on the warehouse nodes, and only the final (small) result comes to your Python process. It's the same pattern as Spark's lazy evaluation, but running on Snowflake's managed infrastructure.
:::

:::drill {type="mcq" q="You call SNOWFLAKE.CORTEX.SENTIMENT() on 10 million rows. What is the biggest risk to watch for?"}
- [ ] The function does not support batch processing and will time out
- [x] Credit consumption could spike unexpectedly — always test on a sample first
- [ ] Sentiment analysis is not available on text longer than 512 characters
- [ ] The result is cached indefinitely, so fresh data won't be scored
:::

:::drill {type="mcq" q="Which Snowflake Cortex feature turns a natural-language question into a SQL query against YOUR schema reliably?"}
- [ ] Cortex Search, because it uses hybrid vector + keyword retrieval
- [ ] SNOWFLAKE.CORTEX.COMPLETE() with a carefully engineered system prompt
- [x] Cortex Analyst, because it uses a user-supplied semantic model as a guardrail
- [ ] Snowpark UDFs, because they can execute arbitrary Python logic
:::

:::drill {type="mcq" q="A teammate says 'Snowpark is just another SQL client.' What is the key thing they are missing?"}
- [ ] Snowpark supports more SQL dialects than a regular connector
- [ ] Snowpark runs on your local GPU, so it is faster than a SQL client
- [x] Snowpark DataFrames push computation into Snowflake — no bulk data egress to Python
- [ ] Snowpark is only for Scala and Java, not Python
:::

:::key-takeaway
Snowflake Cortex turns your warehouse into an AI platform: LLM functions for in-SQL inference, Cortex Search for semantic retrieval, Cortex Analyst for safe text-to-SQL, and Snowpark for Python ML — all without moving data or managing separate infrastructure.
:::
