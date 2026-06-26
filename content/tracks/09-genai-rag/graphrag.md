---
id: graphrag
track: 09-genai-rag
title: "GraphRAG & knowledge graphs (Neo4j / Neptune; multi-hop)"
badge: HOT
minutes: 11
prereqs: []
tags: [graphrag, neo4j, neptune, knowledge-graph, multi-hop, cypher, rag]
xp: 60
hot2026: true
---

A compliance officer at a Pune fintech firm fires up your shiny new RAG chatbot and types:

> "Which of our counterparties share a board member with any company flagged in the last SEBI enforcement order?"

Your vector search retrieves three paragraphs about enforcement orders and two about board meetings. None connect the dots. The LLM apologises politely. The compliance officer files a Jira ticket titled "chatbot is useless."

That's not a retrieval quality problem. It's a *data model* problem. Flat vector search retrieves passages. It has zero concept of the relationships between entities. That gap is exactly what **GraphRAG** fills.

## The fundamental limit of passage retrieval

When a question requires you to traverse two or more connections — "find the X that links Y to Z" — you're doing **multi-hop reasoning**. A cosine-similarity index returns the most semantically similar chunks. It cannot tell you that "Reliance Infrastructure" and "Reliance Capital" share a chairman, or that an audit finding in one PDF connects to a penalty in another.

A **knowledge graph** stores facts as structured triples:

```
(Company: Reliance Capital) -[:AUDITED_BY]-> (Firm: Deloitte India)
(Person: Amit Shah) -[:DIRECTOR_OF]-> (Company: Reliance Capital)
```

These are **nodes** (entities) and **edges** (relationships). Once your facts live in a graph, you can traverse them in a single query instead of hoping seven retrieved chunks happen to mention the same names.

:::why-prod
Multi-hop reasoning is the core gap between RAG demos and production enterprise assistants. Legal, compliance, finance, and supply-chain use cases almost always need it. Companies building internal knowledge tools hit this wall within weeks of go-live — and the teams that designed for graphs from day one don't have to do expensive retrofits.
:::

## Knowledge graph fundamentals

Three building blocks, nothing more:

- **Node** — an entity with a label and properties: `(:Company {name: "Infosys", cin: "L72200MH1981PLC028216"})`.
- **Edge** — a directed, labelled relationship: `-[:PARTNER_OF {since: 2019}]->`.
- **Property** — metadata on either: `{amount: 45000000, currency: "INR"}`.

A knowledge graph is a large collection of these triples. Your application queries them with a traversal instead of a keyword or embedding lookup.

:::table {title="Neo4j vs AWS Neptune — when to pick which"}
| | Neo4j | AWS Neptune |
|---|---|---|
| Query language | Cypher (SQL-like, easy to learn) | Gremlin / SPARQL / openCypher |
| Hosting | Self-hosted or AuraDB (managed) | Fully managed on AWS |
| Free tier | AuraDB Free (50k nodes) | No free tier |
| Python driver | `neo4j`, LangChain built-in | `gremlinpython`, `boto3` |
| Best fit | Fast prototyping, LLM integrations | AWS-native production stacks |
| On-prem option | Yes (Community / Enterprise) | No |
:::

For a portfolio project or early-stage product in India: **start with Neo4j AuraDB Free**. For a company already on AWS: Neptune is the natural fit.

## Two GraphRAG flavours

**Flavour 1 — Microsoft GraphRAG** (the 2024 paper that made noise). It chunks your corpus, detects communities of co-occurring entities using the Leiden algorithm, and pre-generates *community summaries*. At query time, relevant summaries are retrieved rather than raw chunks. Excellent for large unstructured corpora when you need global reasoning — "what are all the risk themes across 10,000 documents?"

**Flavour 2 — Classic KG + RAG hybrid** (the one you'll actually build for enterprises). You extract entities and relationships from your documents, load them into Neo4j, and then at query time you: (1) identify entities in the question, (2) run a Cypher traversal to pull the relevant subgraph, (3) serialize that subgraph as context, and (4) pass it to the LLM alongside your vector-retrieved chunks.

Most production enterprise projects in India are Flavour 2. You have a defined domain with known entity types — companies, people, contracts, transactions — and you need precise, auditable answers.

## Building the pipeline

Three stages: **extract → load → retrieve**.

```python {title="GraphRAG skeleton — entity extraction + Neo4j load + multi-hop query" run=false}
# pip install neo4j openai
# Free local graph DB:
#   docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:latest
# Or use Neo4j AuraDB Free: https://neo4j.com/cloud/platform/aura-graph-database/

import json, os
from neo4j import GraphDatabase

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASS = os.getenv("NEO4J_PASS", "test")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# ── STAGE 1: Extract triples from a text chunk via LLM ──────────────────────
def extract_triples(chunk: str, llm_fn) -> list[dict]:
    """
    llm_fn: any callable(prompt: str) -> str
    Returns list of {"subject": ..., "relation": ..., "object": ...}
    """
    prompt = f"""Extract named entities and relationships from the text.
Return ONLY a JSON array of objects with keys: subject, relation, object.
Example: [{{"subject": "Infosys", "relation": "AUDITED_BY", "object": "Deloitte"}}]

Text:
{chunk}"""
    raw = llm_fn(prompt)
    return json.loads(raw)

# ── STAGE 2: Load triples into Neo4j (MERGE avoids duplicates) ───────────────
def load_triples(triples: list[dict]):
    with driver.session() as session:
        for t in triples:
            session.run(
                """
                MERGE (s:Entity {name: $subject})
                MERGE (o:Entity {name: $object})
                MERGE (s)-[r:RELATES {type: $relation}]->(o)
                """,
                subject=t["subject"],
                object=t["object"],
                relation=t["relation"],
            )

# ── STAGE 3: Multi-hop Cypher retrieval (2 hops) ────────────────────────────
def multi_hop_context(entity_name: str, hops: int = 2) -> list[dict]:
    """
    Walks up to `hops` edges from the starting entity.
    Returns a list of {chain: [...node names], rels: [...relation types]}.
    Serialize this as text and inject into your LLM prompt.
    """
    cypher = f"""
    MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(connected)
    RETURN [n IN nodes(path)  | n.name] AS chain,
           [r IN relationships(path) | r.type]  AS rels
    LIMIT 25
    """
    with driver.session() as session:
        results = session.run(cypher, name=entity_name)
        return [dict(r) for r in results]

# Example usage (plug in your LLM):
# triples = extract_triples(chunk_text, my_llm)
# load_triples(triples)
# context = multi_hop_context("Reliance Capital", hops=2)
# → serialize context → send to LLM with user question
```

The retrieved subgraph is serialized as structured text — "Reliance Capital is AUDITED_BY Deloitte; Amit Shah is DIRECTOR_OF Reliance Capital; Reliance Capital is SUBSIDIARY_OF Reliance ADAG" — and injected into the LLM's context window alongside your vector chunks. The LLM now has the relationship chain it needs to answer the question.

:::gotcha
LLM-extracted entities arrive with inconsistent spellings: "Infosys", "Infosys Ltd.", "INFY". If you `MERGE` on raw name strings, your graph splinters into disconnected islands and multi-hop queries return nothing. Always run **entity canonicalization** before loading — fuzzy match against a master entity list (use RapidFuzz), or use CIN numbers for companies and PAN for individuals as your canonical key. MERGE on the canonical ID, store the raw alias as a property.
:::

:::war-story {title="The compliance chatbot that couldn't cross two documents"}
A team at a Pune financial services firm built a RAG chatbot over 50,000 regulatory documents. The demo was slick. Three weeks after launch, a compliance officer asked: "Which of our active counterparties have been penalised by SEBI in the last six months?" Vector search returned penalty notices just fine. But it couldn't link the counterparty names in internal *transaction records* to the penalised entities in *SEBI orders* — they lived in different documents with no shared text overlap. Two engineers spent two sprints retrofitting a Neo4j entity-linking layer. The lead architect said afterward: "If the domain has entities that span document types, graph is not optional — it's the architecture."
:::

:::interview-line
"For single-document Q&A I rely on vector retrieval; the moment a question needs me to traverse relationships across entities — two hops or more — I add a Neo4j layer, run a Cypher subgraph query, and feed structured relationship context to the LLM alongside the vector chunks."
:::

:::qa {q="What problem does GraphRAG solve that standard vector RAG cannot?"}
Standard RAG retrieves passages by semantic similarity — it has no model of how entities relate to each other. GraphRAG stores facts as nodes and edges, so you can traverse connections in a single structured query. The result is multi-hop reasoning: answering questions that require linking entities which never appear together in any single chunk.
:::

:::qa {q="When would you prefer Microsoft GraphRAG over a classic KG + RAG hybrid?"}
Microsoft GraphRAG suits large, unstructured corpora where you need global summarization — "what are the main themes across these 10,000 filings?" A classic KG hybrid is better when you have a well-defined entity schema (companies, people, contracts) and need precise, traversal-based answers. Enterprise domains with known entity types almost always warrant the hybrid approach because it gives you auditable, exact relationship chains.
:::

:::qa {q="How do you prevent entity fragmentation when loading extracted triples into Neo4j?"}
Canonicalize entity names before writing to the graph. Common approaches: fuzzy string matching with RapidFuzz to resolve spelling variants, a dedicated NER + entity-linking model, or using domain-specific identifiers like CIN numbers or PAN as the MERGE key. The rule is: MERGE on a canonical stable ID, not on the raw extracted string, and store the original variant as an alias property for debugging.
:::

:::drill {type="mcq" q="A user asks: 'Which of our suppliers share a logistics partner with a vendor blacklisted last month?' Why does flat vector RAG fail on this?"}
- [ ] Embeddings are too slow to handle high query volume
- [ ] The question is too short to produce a meaningful embedding
- [x] Answering requires traversing two relationship hops across entities split across separate documents
- [ ] Vector databases cannot store vendor or supplier data
:::

:::drill {type="mcq" q="You extract the triple ('Infosys', 'AUDITED_BY', 'Deloitte') from one doc and ('Infosys Ltd.', 'PARTNER_OF', 'Microsoft') from another. What must you do before running MERGE in Neo4j?"}
- [ ] Load both as-is; Neo4j automatically deduplicates entity names
- [ ] Drop one of the Infosys nodes to avoid ambiguity in traversals
- [x] Canonicalize 'Infosys Ltd.' to 'Infosys' via entity linking so both edges attach to the same node
- [ ] Switch from Cypher to SPARQL, which handles name variants natively
:::

:::drill {type="mcq" q="Microsoft GraphRAG applies the Leiden algorithm during indexing. What is it doing?"}
- [ ] Detecting hallucinations in LLM-generated summaries
- [ ] Grouping API calls into batches to reduce cost
- [x] Clustering co-occurring entities into topic communities so each community can be pre-summarized for retrieval
- [ ] Deduplicating chunks that have high cosine similarity
:::

:::key-takeaway
GraphRAG adds a relationship layer on top of vector retrieval. Store entities and edges in Neo4j or Neptune, traverse them with Cypher for multi-hop questions, and inject the resulting subgraph as structured context for your LLM. It is the difference between a chatbot that retrieves passages and one that actually connects the dots across your entire knowledge base.
:::
