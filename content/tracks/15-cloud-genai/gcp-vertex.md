---
id: gcp-vertex
track: 15-cloud-genai
title: "GCP Vertex AI: training, endpoints, RAG Engine, Agent Builder"
badge: HOT
minutes: 10
prereqs: []
tags: [gcp, vertex-ai, rag, agents, cloud, deployment, endpoints]
xp: 60
hot2026: true
---

It's Friday evening. Your team just landed a pilot with a Pune fintech. They want a document Q&A bot for loan agreements — built, deployed, and demoed by Monday morning. You have 48 hours, a GCP free-trial account, and zero desire to configure Kubernetes. This is exactly the scenario Vertex AI was built for.

Vertex AI is Google Cloud's unified ML platform. One place for training custom models, hosting prediction endpoints, building RAG pipelines, and wiring up conversational agents — all managed, all observable, none of the infra babysitting.

## The Four Pillars You Actually Use

Vertex AI is large. But four services cover 90% of real work.

**Custom Training** — Submit a Python training script and a Docker image (or use a GCP pre-built container). GCP provisions the machines, handles spot-instance preemption retries, and streams logs to Cloud Logging. Think of it as a managed `sbatch` with a Python SDK.

**Model Endpoints** — After training (or after importing a model artifact), you deploy to an endpoint. GCP handles autoscaling, health checks, and traffic splits between model versions. Online prediction gives you a low-latency REST API. Batch prediction is async, cheap, and designed for bulk inference jobs.

**Vertex AI RAG Engine** — Managed RAG (Retrieval-Augmented Generation). You upload documents → Vertex chunks them, embeds them, and stores vectors in a managed index → you query with Gemini and get grounded answers with citations. No Pinecone account, no custom chunking code, no embedding pipeline to maintain. Went GA in late 2024 and shows up in almost every GCP-track interview now.

**Agent Builder** — Build multi-turn conversational agents backed by Gemini. Give it DataStores (your RAG corpora), Tools (API endpoints), and a system prompt. It wraps into a Dialogflow CX agent under the hood. Great for that loan-agreement chatbot — and for internal copilots that compliance teams actually approve of.

:::why-prod
Managed services let your team ship features instead of babysitting infra. RAG Engine removes the embedding-pipeline maintenance burden entirely. Agent Builder produces structured session logs that compliance teams can audit — critical in BFSI and healthcare clients, which dominate Pune's enterprise market.
:::

:::table {title="Vertex AI: Four Pillars at a Glance"}
| Pillar | You bring | GCP manages | Best for |
|---|---|---|---|
| Custom Training | Python script + container | Machines, spot retries, logs | Fine-tuning, custom PyTorch/TF |
| Endpoints | Saved model artifact | Autoscaling, health checks, A/B splits | Online + batch inference |
| RAG Engine | Documents / PDFs | Chunking, embedding, vector store, retrieval | Grounded Q&A on private data |
| Agent Builder | Prompt + DataStore + Tools | Orchestration, sessions, audit logs | Conversational agents, copilots |
:::

## Deploying a Model Endpoint

The flow: upload artifact → deploy → call it. This works on the GCP free trial.

```python {title="Upload and deploy a model to Vertex AI endpoint" run=false}
# pip install google-cloud-aiplatform
# gcloud auth application-default login   (free, local credentials)

from google.cloud import aiplatform

PROJECT = "my-gcp-project"        # replace with yours
REGION  = "us-central1"           # best free-tier quota region
ARTIFACT_URI = "gs://my-bucket/models/loan_risk_v1/"  # your saved model

aiplatform.init(project=PROJECT, location=REGION)

# Step 1: Register model in Model Registry
model = aiplatform.Model.upload(
    display_name="loan-risk-v1",
    artifact_uri=ARTIFACT_URI,
    serving_container_image_uri=(
        "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    ),
)

# Step 2: Deploy to endpoint
# min_replica_count=0 → scale-to-zero when idle (free-tier friendly)
endpoint = model.deploy(
    machine_type="n1-standard-2",
    min_replica_count=0,   # ⚠ causes cold starts — see Gotcha below
    max_replica_count=2,
    traffic_split={"0": 100},
)

# Step 3: Predict
response = endpoint.predict(instances=[{"tenure": 24, "amount": 500000}])
print(response.predictions)

# Always clean up on free tier to avoid charges:
# endpoint.undeploy_all(); endpoint.delete(); model.delete()
```

## RAG Engine: Upload Docs, Query with Gemini

```python {title="Vertex AI RAG Engine: corpus creation and grounded query" run=false}
# pip install google-cloud-aiplatform[preview]
import vertexai
from vertexai.preview import rag
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview.generative_models import grounding

vertexai.init(project="my-gcp-project", location="us-central1")

# 1. Create managed vector corpus
corpus = rag.create_corpus(display_name="loan-docs-corpus")

# 2. Import PDFs — Vertex handles chunking + embedding (Gecko model by default)
rag.import_files(
    corpus.name,
    paths=["gs://my-bucket/loan-agreements/"],
    chunk_size=512,
    chunk_overlap=100,
)

# 3. Query: Vertex retrieves relevant chunks, Gemini uses them as context
retrieval_tool = Tool.from_retrieval(
    retrieval=grounding.Retrieval(
        source=grounding.VertexRagStore(rag_corpora=[corpus.name]),
    )
)
model = GenerativeModel("gemini-1.5-pro-002", tools=[retrieval_tool])
chat  = model.start_chat()
resp  = chat.send_message("What is the prepayment penalty in the 2024 agreement?")
print(resp.text)   # includes citations to source document + page
```

## Agent Builder: the Last Mile

Agent Builder lives in the GCP Console under **Vertex AI → Agent Builder**. For a 48-hour pilot, the UI is faster than the SDK:

1. Create a **DataStore** — point it at your RAG corpus or a GCS bucket of documents.
2. Create an **App** → choose "Conversational Agent" → pick your Gemini model.
3. Add the DataStore as a grounding source and write your system prompt.
4. Test in the built-in chat UI, then grab the REST endpoint URL for your frontend.

For programmatic access, use `google-cloud-discoveryengine` or the Dialogflow CX Python SDK. For demos and MVPs, the console gets you live faster.

:::gotcha
`min_replica_count=0` (scale-to-zero) is tempting on free tier — but the first request after the endpoint sits idle triggers a 60–90 second cold start while GCP provisions a VM. If your client sees a blank screen for 90 seconds and refreshes, the demo is already in trouble. Set `min_replica_count=1` for anything demo-facing. Budget roughly ₹3,500–5,000/month per always-warm `n1-standard-2` node — less than one hour of a consultant's time.
:::

:::war-story {title="The 10 AM Demo That Went Silent"}
A Bengaluru team deployed their RAG chatbot to Vertex AI the night before a client review. Scale-to-zero kept their overnight costs at ₹0. The client opened the app at 10 AM after 8 hours of idle. First question: 85 seconds of blank screen. The client refreshed three times assuming it was broken. The second question responded in 200ms (warm node). The demo recovered — but the first impression was gone. One always-warm replica would have cost ₹4,000 that month. The lost deal cost far more.
:::

:::interview-line
"On Vertex AI I use RAG Engine for grounded Q&A on private documents — it manages chunking, embedding, and retrieval — and Agent Builder to expose it as a conversational interface, so we skip the infra and ship the product."
:::

:::qa {q="What's the difference between Vertex AI RAG Engine and self-hosting a vector DB like Pinecone?"}
RAG Engine is fully managed — GCP handles chunking strategy, embedding (Gecko by default), indexing, and retrieval scoring. You pay per query and per document stored; no cluster to maintain. The trade-off is less control: you can't swap embedding models freely or tune index parameters deeply. Pinecone gives full control; RAG Engine gives zero ops overhead. For teams without a dedicated ML platform engineer, RAG Engine typically wins.
:::

:::qa {q="How do you safely roll out a new model version on a Vertex AI endpoint without downtime?"}
Deploy the new version as a second deployment on the same endpoint, then use a traffic split — start at 90/10 old/new, monitor latency and error rate in Cloud Monitoring, then shift gradually to 0/100. Vertex routes traffic in-flight with no downtime. To roll back, flip the split back to the previous deployment instantly. Never delete the old deployment until you're confident in the new one.
:::

:::drill {type="mcq" q="Your Vertex AI endpoint has min_replica_count=0. After 3 hours of no traffic, the next request takes 85 seconds. What is the root cause?"}
- [ ] The model artifact in GCS has become corrupted and needs re-download
- [ ] Vertex AI rate-limits the first request after idle to prevent abuse
- [x] Scale-to-zero requires provisioning a new VM on the first request (cold start)
- [ ] GCP applies a mandatory warm-up delay on all Gemini-grounded endpoints
:::

:::drill {type="mcq" q="Which Vertex AI service automatically handles document chunking, embedding, and vector storage when you upload PDFs from GCS?"}
- [ ] Vertex Matching Engine (standalone)
- [ ] Vertex Feature Store
- [x] Vertex AI RAG Engine
- [ ] Vertex AI Pipelines
:::

:::key-takeaway
Vertex AI's four pillars — Custom Training, Endpoints, RAG Engine, Agent Builder — map directly onto the GenAI product lifecycle. For most teams, RAG Engine + Agent Builder get you to a working grounded chatbot in a weekend; Custom Training and Endpoints come into play when you need fine-tuned models in production.
:::
