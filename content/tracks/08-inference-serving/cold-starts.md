---
id: cold-starts
track: 08-inference-serving
title: "Cold starts & the scale-to-zero trap"
badge: CORE
minutes: 7
prereqs: []
tags: [inference, serving, latency, kubernetes, scale-to-zero, cold-start, deployment]
xp: 45
hot2026: false
---

It's Friday afternoon. You demo your new LLM-powered search API to your PM. Snappy, clean, impressive — 800 ms per request. Everyone leaves happy.

Monday morning. A user hits the endpoint at 9:01 AM. The app spins. Three minutes later — finally — a response crawls back. Slack lights up: "Is the search down?"

Nothing broke. The model just went **cold**. Welcome to scale-to-zero.

## What actually happens during a cold start

When a container is spun down and a fresh request arrives, three things must happen *in sequence* before your model can respond:

1. **Container boot** — the runtime (Docker, containerd) starts the image. Roughly 10–30 seconds.
2. **Weight download** — the model is fetched from S3 / GCS / Azure Blob. A 7B model at fp16 is ~14 GB. On a decent 1 Gbps uplink, that's nearly two minutes.
3. **GPU load** — CUDA allocates VRAM, weights are transferred from host RAM to GPU memory, and the KV-cache is sized. Another 30–60 seconds depending on model and driver.

Add it up: **90 seconds to 5 minutes** before the first token leaves the box. For a user staring at a spinner, that's indistinguishable from a crash.

## Why scale-to-zero is tempting (and when it backfires)

Scale-to-zero means your infra drops to 0 replicas during idle periods. Kubernetes with KEDA, Google Cloud Run, AWS Lambda, Knative — all support it. The pitch is compelling: *pay only when traffic exists*.

For stateless microservices — a currency converter, an image resizer — it's excellent. Cold starts there are 200–500 ms. Barely a blink.

For an LLM? It's a trap. The weight-loading cost that's trivial at 2 MB becomes catastrophic at 14 GB. And the worst part: **you will never see this in your load test**, because load tests keep replicas alive. The problem lands squarely on real users at 9 AM after a quiet night.

:::why-prod
Cold starts are invisible during testing but hit users at the worst moments — first thing Monday morning, after lunch, post-weekend. Your p99 latency isn't 1.2 seconds; it's 3 minutes. The tail IS the product, and cold starts are pure tail.
:::

:::table {title="Rough cold-start times by model size (fp16, A100 80GB, 1 Gbps network)"}
| Model Size | Weight File | Download | GPU Load | Typical Total |
|---|---|---|---|---|
| 1B params | ~2 GB | ~16 s | ~10 s | ~40 s |
| 7B params | ~14 GB | ~112 s | ~30 s | ~2.5 min |
| 13B params | ~26 GB | ~210 s | ~50 s | ~4.5 min |
| 70B params | ~140 GB | 18+ min | ~3 min | 20+ min |
:::

These are ballpark numbers. Your actual storage tier (S3 Standard vs instance-local NVMe), network speed, and serving framework (vLLM vs Triton vs plain HuggingFace) all shift the curve. The shape is always the same though: bigger model, brutal cold start.

## Your four levers

**1. Never let LLM pods hit zero.**
Set `minReplicas: 1`. You pay for one idle GPU between requests — roughly ₹15,000–30,000/month at cloud on-demand rates for an A10G. That is the cost of a production SLA. Budget it in from day one.

**2. Cache weights close to the GPU.**
Mount the model from an instance-local NVMe volume (EBS gp3, GCP Hyperdisk, a hostPath volume on a dedicated node) instead of downloading from blob storage on every start. Download time drops to near zero. This single change often cuts cold-start from 2–3 minutes to under 30 seconds.

**3. Use provisioned concurrency / minimum instances.**
SageMaker Serverless, Cloud Run with minimum instances, and Lambda with provisioned concurrency all keep containers pre-warmed at the platform level. Weights are loaded; only inference runs. Cleaner than managing Kubernetes yourself if you're early-stage.

**4. Quantize to shrink the payload.**
An INT4-quantized 7B model is ~4 GB instead of 14 GB. Download time drops 3.5×. Cold start goes from 2.5 min to under a minute. Quantization belongs in the sibling lesson; just know that smaller weights are a free cold-start improvement on top of accuracy-cost tradeoffs.

```python {title="Kubernetes — floor at 1 replica, never 0 for LLMs" run=false}
# deploy.yaml — apply with: kubectl apply -f deploy.yaml
# Assumes vLLM image and a node with 1x A10G or similar

apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-inference
  template:
    metadata:
      labels:
        app: llm-inference
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - "--model"
            - "mistralai/Mistral-7B-Instruct-v0.2"
            - "--max-model-len"
            - "4096"
          volumeMounts:
            - name: model-cache
              mountPath: /root/.cache/huggingface  # weights live here on the node
          resources:
            limits:
              nvidia.com/gpu: "1"
      volumes:
        - name: model-cache
          hostPath:
            path: /mnt/nvme/models  # pre-pulled onto node's local NVMe
            type: DirectoryOrCreate
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-inference-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-inference
  minReplicas: 1   # <--- THE KEY LINE. For LLMs, this is never 0.
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

:::gotcha
Teams new to LLM serving copy a microservice HPA config and set `minReplicas: 0` because "free scaling." It looks fine in staging — the test runner keeps the pod alive. It bites users on the very first real request after any idle window. For LLMs, `minReplicas: 0` is not a configuration choice; it's a latency time bomb.
:::

:::interview-line
"Scale-to-zero is a microservice pattern — for LLMs it trades billing savings for catastrophic tail latency. I keep minReplicas at 1 and cache weights on local NVMe so cold-start is seconds, not minutes."
:::

:::qa {q="Why are cold starts so much worse for LLMs than for a typical REST microservice?"}
A microservice's cold start is mostly container boot — 200–500 ms. An LLM adds two more sequential costs: downloading gigabytes of weights from remote storage and then loading them into GPU VRAM through CUDA. Each step adds tens of seconds to minutes. Stack all three and even a 7B model can take 2–3 minutes, which is an eternity for a user-facing product.
:::

:::qa {q="How would you eliminate cold starts in a cost-sensitive production setup?"}
Keep `minReplicas: 1` so at least one pod stays warm at all times — the idle GPU cost is just the price of an SLA. Mount model weights from instance-local NVMe rather than fetching from blob storage on every restart to cut the biggest chunk of cold-start time. If the team is small, use a managed option like Cloud Run minimum instances or SageMaker provisioned concurrency rather than operating Kubernetes yourself. Apply quantization to shrink the weight file as a secondary win.
:::

:::drill {type="mcq" q="Your LLM API has scale-to-zero enabled. Traffic peaks 10 AM–6 PM on weekdays. When will users most likely hit cold-start latency?"}
- [ ] During peak load at 2 PM when requests queue up
- [x] At 10 AM Monday after the weekend idle window
- [ ] During a rolling deployment when pods restart
- [ ] When GPU memory is fragmented by long prompts
:::

:::drill {type="mcq" q="Which single change gives the most direct reduction in LLM cold-start time?"}
- [ ] Switching the load balancer from round-robin to least-connections
- [ ] Increasing maxReplicas from 4 to 8
- [x] Mounting model weights from instance-local NVMe instead of downloading from S3 on each start
- [ ] Adding a Redis layer to cache inference outputs
:::

:::key-takeaway
Scale-to-zero is a trap for LLMs: weight-loading turns a cold start into a multi-minute user-facing outage. Set `minReplicas: 1` and cache weights on local storage — that one config line is the difference between a reliable product and a broken one at 9 AM Monday.
:::
