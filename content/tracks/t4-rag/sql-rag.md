---
id: sql-rag
track: t4-rag
title: "SQL + RAG: text-to-SQL over your warehouse"
badge: HOT
minutes: 9
prereqs: []
tags: [sql, rag, text-to-sql, llm, warehouse, schema-injection, production]
xp: 60
hot2026: true
---

Your product manager messages you at 3 pm: "Can you pull Q1 revenue by region, broken down by channel?" Normally that's a Jira ticket, a two-day wait, and a CSV with the wrong date range attached. With text-to-SQL RAG, the PM types the same question into a chat interface and gets an answer straight from the warehouse — no analyst in the loop, no ticket, no waiting.

That's the promise. Let's make it real — and safe.

## What Is text-to-SQL RAG?

Classic RAG retrieves *document chunks* from a vector store and stuffs them into a prompt. Text-to-SQL skips the vector store entirely for your data. Instead, it translates the user's natural-language question into a SQL query, runs it against the database, and feeds the *result rows* back to the LLM so it can write a plain-English summary.

The pipeline is:

**User question → schema injection → LLM writes SQL → execute query → LLM summarizes result rows → answer**

The "retrieval" step is SQL execution — the database engine does the search. No embeddings on your actual rows needed. (Embeddings can help you *select which tables* to inject for large schemas, but that's a separate step.)

:::why-prod
Analysts can't scale to every product question. Text-to-SQL lets the LLM act as a self-service analytics layer — PMs, ops teams, and customer success reps query the warehouse in plain English. Done right, it cuts turnaround from days to seconds without any new dashboards.
:::

## The Three Hard Parts

### 1. Schema injection — giving the LLM a map

The LLM writes SQL blind unless you show it the schema. For a small warehouse, inject the full DDL. For large ones (hundreds of tables), you first embed table descriptions and retrieve only the relevant subset — RAG feeding RAG.

A good schema prompt includes table names, column names with types, primary and foreign keys, and a one-line business definition for any ambiguous column — for example: `revenue_usd — net revenue after refunds, in USD, not including tax`.

### 2. SQL hallucinations — confident and wrong

LLMs hallucinate column names that almost-but-don't-exist. They'll JOIN on the wrong key. They'll write `SUM(order_id)` when you needed `COUNT(DISTINCT order_id)`. The output *looks* correct until someone compares it to the real numbers.

Defense layers that actually work in production:

- **Parse before executing.** Run the generated SQL through a dialect parser (sqlglot works great) to catch syntax errors without touching the DB.
- **Few-shot examples.** Show 3–5 example question/SQL pairs for your specific schema. This dramatically reduces dialect mistakes and wrong column references.
- **Retry with error feedback.** When a query fails, send the error message back to the LLM and ask it to self-correct. One retry loop catches roughly 80 % of mistakes.

### 3. Safety — a write query ruins your afternoon

Always connect text-to-SQL with a **read-only database user**. Non-negotiable. Also add a row-limit guard and a query timeout. One unconstrained `SELECT *` on a billion-row events table — or a generated `DELETE` if the user somehow has write access — will make for a memorable incident review.

:::table {title="Retrieval RAG vs. text-to-SQL RAG"}
| Dimension | Retrieval RAG | Text-to-SQL RAG |
|---|---|---|
| Data source | Documents, PDFs, web pages | Structured tables / warehouse |
| Retrieval mechanism | Vector similarity search | SQL execution |
| Data freshness | Refreshed on re-index | Real-time — query runs now |
| Main hallucination risk | Factual content | Wrong column names / bad SQL logic |
| Setup complexity | Chunking + embedding pipeline | Schema prompt + read-only DB user |
| Best for | Unstructured knowledge | Aggregations, filters, counts |
:::

```python {title="Text-to-SQL with LangChain SQLAgent" run=false}
# pip install langchain langchain-community langchain-openai sqlalchemy psycopg2-binary
# Swap ChatOpenAI for ChatAnthropic, ChatOllama, or any LangChain-compatible LLM.
# For a free local run: use ChatOllama with llama3 or mistral.

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI  # replace with your provider

# ALWAYS use a read-only DB user — this is your primary safety layer
db = SQLDatabase.from_uri(
    "postgresql://readonly:secret@warehouse-host/analytics",
    sample_rows_in_table_info=2,          # shows the LLM 2 example rows per table
    include_tables=["orders", "customers", "products"],  # allow-list; never expose all tables
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # low temp = more deterministic SQL

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True,       # logs the generated SQL — essential for debugging early on
    max_iterations=5,   # retry ceiling; prevents runaway loops on tricky questions
)

response = agent.invoke(
    {"input": "Which product category had the highest net revenue in Q1 2025?"}
)
print(response["output"])
# -> "Electronics led Q1 2025 with $4.2M in net revenue, followed by Apparel at $2.8M."
```

:::gotcha
Injecting your entire schema — all 300 tables, every column — blows the context window and actively confuses the model. Always use an allow-list of relevant tables, or add a schema-retrieval step that picks tables by embedding similarity to the user's question before building the prompt.
:::

:::war-story {title="The query that queried everything"}
A startup shipped a text-to-SQL assistant for their ops team with no row limit, no query timeout, and a DB user with write access because "the ORM needed it." On day two, a sales rep asked "show me all raw events for every customer." The LLM wrote `SELECT * FROM events` — 900 million rows. The query ran for 11 minutes before the DBA killed it, taking down a read replica in the process. They added `LIMIT 50000`, a 30-second timeout, and a strict read-only user before relaunching. The read replica never forgot.
:::

:::interview-line
"Text-to-SQL is RAG where retrieval is a SQL query — the key production controls are a read-only DB user, a dialect parser that validates SQL before execution, and a self-correction retry loop that feeds errors back to the model."
:::

:::qa {q="How do you prevent the LLM from writing dangerous SQL like DELETE or DROP TABLE?"}
The primary control is using a database user that has only SELECT permission — dangerous statements fail at the DB layer before any damage can occur. As a secondary layer, parse the generated SQL and reject anything that is not a SELECT statement before it even hits the database. Defense-in-depth beats prompt-based promises because prompts can be overridden by sufficiently creative questions.
:::

:::qa {q="What do you do when the generated SQL is syntactically valid but returns the wrong business answer?"}
This is the hardest problem in text-to-SQL. The main levers are: richer schema annotations (add business definitions to column names), few-shot examples that cover similar query patterns, and an evaluation set of golden question-SQL pairs you run in CI to catch regressions. For mission-critical use cases, add a human review step or a separate LLM that checks whether the SQL logically matches the question before execution.
:::

:::drill {type="mcq" q="A user asks 'How many unique customers placed more than one order in 2024?' The LLM generates: SELECT COUNT(customer_id) FROM orders WHERE year=2024 AND order_count > 1. What is the primary bug?"}
- [ ] The LLM used the wrong table name
- [x] COUNT(customer_id) counts rows not distinct customers, and order_count is likely a hallucinated column
- [ ] The WHERE clause should use BETWEEN instead of year=
- [ ] There is no bug; this SQL is correct
:::

:::drill {type="mcq" q="In a production text-to-SQL system, which control MOST directly prevents a generated DELETE FROM orders statement from executing?"}
- [ ] Telling the LLM in the system prompt to never write DELETE statements
- [ ] Wrapping the query execution in a try/except block
- [x] Connecting to the database with a user that has only SELECT privileges
- [ ] Setting verbose=True on the agent so you can review the query
:::

:::key-takeaway
Text-to-SQL is RAG where the retrieval step is a SQL query instead of a vector search. The three production essentials: inject a curated schema subset (not your whole warehouse), validate SQL syntax before executing, and always run on a read-only database user.
:::
