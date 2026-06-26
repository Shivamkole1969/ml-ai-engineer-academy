---
id: aws-bedrock
track: 15-cloud-genai
title: "AWS Bedrock: model access, Knowledge Bases, Agents, Guardrails"
badge: HOT
minutes: 11
prereqs: []
tags: [aws, bedrock, rag, agents, guardrails, genai, cloud]
xp: 60
hot2026: true
---

It is 11 PM on a Thursday. Your product manager messages: "The new AI chatbot is live. Can we add it to the AWS account without spinning up any GPUs?" Your first thought is to Google "run LLMs on AWS". Your second thought — after reading this lesson — will be "Bedrock, obviously."

AWS Bedrock is Amazon's managed GenAI platform. You call an API, a frontier model answers. No servers, no CUDA drivers, no GPU bills for idle time. You pay per token. That is the whole deal, and it is genuinely a big deal in production.

## What Bedrock Actually Is

Bedrock gives you a single AWS-native surface for four things:

1. **Model access** — call Claude, Llama, Mistral, Amazon Titan, and others via one unified API.
2. **Knowledge Bases** — managed RAG: point it at S3, it handles chunking, embedding, vector storage, and retrieval.
3. **Agents** — multi-step reasoning loops that can call your APIs, run code, and retrieve from Knowledge Bases.
4. **Guardrails** — content filters, PII redaction, denied-topic enforcement, and hallucination detection layered over any of the above.

These four fit together like Lego. You can use each one independently, or stack them.

:::why-prod
In production you don't want to manage embedding pipelines, vector DBs, and model servers separately — Bedrock glues them into one IAM-secured, VPC-friendly surface. That means one audit trail, one cost dashboard, and one permission boundary for your entire GenAI stack.
:::

## Model Access: The Converse API

There are two ways to call models on Bedrock. The old way is `InvokeModel` — raw, model-specific JSON payloads. The right way (as of 2024) is the **Converse API**. Same code, any model. Switch from Claude to Llama by changing one string.

```python {title="Bedrock Converse API — swap model_id to change LLM" run=false}
import boto3

# Run locally: configure AWS CLI with `aws configure`
# Free-tier note: Bedrock is pay-per-token. ~$0.001 for this snippet with Claude Haiku.
client = boto3.client("bedrock-runtime", region_name="us-east-1")

response = client.converse(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",  # swap this to change model
    messages=[
        {"role": "user", "content": [{"text": "Explain RAG in two sentences."}]}
    ],
    inferenceConfig={"maxTokens": 256, "temperature": 0.3},
)

reply = response["output"]["message"]["content"][0]["text"]
print(reply)

# To use Claude 3.5 Sonnet instead, change modelId to:
# "anthropic.claude-3-5-sonnet-20241022-v2:0"
# The rest of the code is identical — that's the point of Converse.
```

First you enable a model in the Bedrock console under **Model Access** (it is a one-click approval, not a procurement process). Then you call it. That is the entire setup.

:::table {title="Popular models on Bedrock (mid-2025)"}
| Model | Provider | Best for |
|---|---|---|
| Claude 3.5 Sonnet / Haiku | Anthropic | Reasoning, code, long context |
| Llama 3.1 70B / 405B | Meta | Open-weights, cost-sensitive workloads |
| Mistral Large | Mistral AI | European data-residency, instruction follow |
| Amazon Titan Text / Embeddings | Amazon | Embedding generation, cost baseline |
| Nova Pro / Nova Lite | Amazon | Multimodal, low latency |
:::

## Knowledge Bases: Managed RAG

Rolling your own RAG means you own the chunker, the embedding job, the vector DB, the retrieval logic, and the prompt assembly. Every piece can fail silently. Bedrock Knowledge Bases collapses that to three steps:

1. **Point at S3** (or Confluence, SharePoint, web URLs, etc.).
2. **Pick an embedding model** (Titan Embeddings v2 is the sensible default).
3. **Pick a vector store** (OpenSearch Serverless is the managed default; you can also use Aurora PostgreSQL, Pinecone, or Mongo Atlas).

Bedrock handles chunking, embedding, and indexing. You call `RetrieveAndGenerate` and get a grounded answer with source citations. Under the hood it runs a similarity search, stuffs the top-K chunks into the prompt, and calls the model — all in one SDK call.

When your document set changes, you run a **sync** (manual or scheduled). Bedrock re-indexes only the changed files. That matters at scale when you have thousands of PDFs.

## Agents: Multi-Step Reasoning

A Bedrock Agent is a reasoning loop that can:

- Call **Action Groups** — your Lambda functions or OpenAPI endpoints, auto-described to the model.
- Query **Knowledge Bases** — RAG mid-loop.
- Run a **Code Interpreter** sandbox — the model writes Python, runs it, sees the output, iterates.

You describe your tools in plain text or OpenAPI spec. Bedrock handles the ReAct loop (reason → act → observe → reason…). You do not write the orchestration code.

This matters because the painful part of building agents is not the model call — it is reliable tool dispatch, error handling, and session memory. Bedrock Agents abstract that. Your Lambda function just does its job; the agent figures out when to call it.

## Guardrails: The Safety Layer You Actually Ship

A Guardrail is a policy object you attach to any model call, Knowledge Base, or Agent. It enforces:

- **Content filters** — hate, violence, sexual content, and self-harm, each tunable from LOW to HIGH.
- **Denied topics** — plain-English rules like "never discuss competitor pricing."
- **Word filters** — block specific strings (brand names, profanity lists).
- **PII redaction** — detect and mask names, emails, phone numbers, Aadhaar-style IDs.
- **Grounding check** — score whether the model's answer is supported by the retrieved context (fights hallucination).

A Guardrail applies to both the **input** (user prompt) and the **output** (model response). You get an `action` field back — `NONE`, `INTERVENED`, or `BLOCKED` — so your app can log and act accordingly.

:::gotcha
The grounding check in Guardrails is not magic — it is a second model call that scores faithfulness. This adds ~200–400 ms and costs extra tokens. Do not enable it on every call by default. Enable it for high-stakes responses (medical, legal, financial) and skip it for low-risk ones like autocomplete. Profile first, guard second.
:::

:::war-story {title="The 2 AM jailbreak that Guardrails caught"}
A fintech startup launched a Bedrock-powered loan FAQ bot. Three hours after go-live, a user sent a carefully crafted prompt to extract the system prompt and then asked the model to roleplay as "a lenient loan officer ignoring credit policy." The denied-topics Guardrail — configured with "do not override credit assessment rules" — flagged both turns, returned a polite deflection, and wrote an entry to CloudWatch Logs. The security team saw it the next morning. No escalation, no breach, no weekend fire drill. The guardrail cost $0.002 per 1000 tokens to run. The incident response it avoided would have cost days.
:::

:::interview-line
"On AWS I use Bedrock's Converse API for model-agnostic inference, Knowledge Bases for managed RAG, and Guardrails to enforce content and PII policy — all behind one IAM boundary."
:::

:::qa {q="How does Bedrock Knowledge Bases differ from building RAG yourself?"}
Bedrock manages the full pipeline: chunking, embedding, vector storage, sync on document change, and retrieval — you supply only an S3 bucket and a question. Self-built RAG gives you more control over chunking strategy and retrieval logic, but you own every failure mode. Use Bedrock KB to move fast; self-build when you need custom hybrid search or domain-specific chunkers.
:::

:::qa {q="What is a Bedrock Agent's Action Group?"}
An Action Group is a set of tools — defined as Lambda functions or an OpenAPI schema — that the agent can call during its reasoning loop. You describe what each function does in plain language; the model decides when and how to call it. It is the equivalent of "tool use" or "function calling" in the Converse API, but wired into the managed agent orchestration loop with built-in retry and session state.
:::

:::qa {q="When would you NOT use Bedrock Guardrails for hallucination detection?"}
When latency or cost is the primary constraint. The grounding check is a second model call that adds 200–400 ms and tokens. For real-time autocomplete or classification tasks where the model output is not sourced from a Knowledge Base, the grounding check adds cost with no benefit. Use it for high-stakes factual answers where a wrong answer has real consequences.
:::

:::drill {type="mcq" q="Your team wants to switch from Claude 3 Haiku to Llama 3 70B without changing application code. What should you use?"}
- [ ] InvokeModel with a model-specific request body
- [x] The Converse API — only the modelId string changes
- [ ] SageMaker real-time endpoints
- [ ] Bedrock Agents with a code interpreter action group
:::

:::drill {type="mcq" q="A Bedrock Knowledge Base sync is triggered. What does Bedrock re-process?"}
- [ ] All documents in the S3 bucket, every time
- [ ] Only documents that were added since the last sync
- [x] Only documents that were added, modified, or deleted since the last sync
- [ ] Only the most recently uploaded document
:::

:::drill {type="mcq" q="You want to prevent your Bedrock chatbot from discussing a competitor's product. Which Guardrail feature handles this?"}
- [ ] Content filters set to HIGH severity
- [ ] PII redaction with a custom entity type
- [x] Denied topics, defined in plain English
- [ ] Word filters with the competitor name blocked
:::

:::key-takeaway
AWS Bedrock gives you model access, managed RAG (Knowledge Bases), multi-step agents, and safety enforcement (Guardrails) under one IAM-controlled surface — use the Converse API so switching models costs you one line of code, not a refactor.
:::
