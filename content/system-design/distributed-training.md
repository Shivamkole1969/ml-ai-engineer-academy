# Distributed Training Platform

Your team needs to train a 7B-parameter LLM on 2 TB of curated text — and the single A100 you have will finish in approximately "never". You need a platform that splits the work across dozens of GPUs, recovers from node crashes without losing progress, and bills you fairly for what you actually used. That's a distributed training platform.

---

<svg viewBox="0 0 860 130" width="100%" role="img" aria-label="Distributed training pipeline: Data Store → Data Loader Workers → Job Scheduler → Training Cluster → Checkpoint Store → Experiment Tracker">
  <!-- Box 1: Data Store -->
  <rect x="10" y="40" width="120" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="70" y="58" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Data</text>
  <text x="70" y="74" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Store</text>
  <!-- Arrow -->
  <line x1="130" y1="62" x2="168" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="162,57 168,62 162,67" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <!-- Box 2: Data Loader -->
  <rect x="170" y="40" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="235" y="58" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Data Loader</text>
  <text x="235" y="74" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Workers</text>
  <!-- Arrow -->
  <line x1="300" y1="62" x2="338" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="332,57 338,62 332,67" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <!-- Box 3: Job Scheduler -->
  <rect x="340" y="40" width="130" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="405" y="58" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Job</text>
  <text x="405" y="74" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Scheduler</text>
  <!-- Arrow -->
  <line x1="470" y1="62" x2="508" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="502,57 508,62 502,67" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <!-- Box 4: Training Cluster -->
  <rect x="510" y="40" width="135" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="577" y="58" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Training</text>
  <text x="577" y="74" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Cluster (GPUs)</text>
  <!-- Arrow -->
  <line x1="645" y1="62" x2="683" y2="62" stroke="#8b7bff" stroke-width="2"/>
  <polyline points="677,57 683,62 677,67" fill="none" stroke="#8b7bff" stroke-width="2"/>
  <!-- Box 5: Checkpoint + Tracker -->
  <rect x="685" y="40" width="160" height="44" rx="10" fill="none" stroke="#3ad6ff" stroke-width="1.5"/>
  <text x="765" y="58" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Checkpoint +</text>
  <text x="765" y="74" fill="#eaf0ff" font-size="11" text-anchor="middle" font-family="monospace">Experiment Tracker</text>
</svg>

---

## Components

| Component | Role |
|---|---|
| **Distributed Object Store** (S3/GCS/Azure Blob) | Holds raw training data, tokenized datasets, model artifacts, and checkpoints. Single source of truth. |
| **Data Loader Workers** | Pre-fetch, shuffle, tokenize, and batch data on CPU nodes. Feed GPUs without starving them. |
| **Job Scheduler** (SLURM / Ray / Kubernetes + Volcano) | Allocates GPU nodes, enforces resource quotas, queues jobs by priority, handles preemption. |
| **Parameter Server / AllReduce Ring** (NCCL, Gloo) | Aggregates gradients across workers. AllReduce is dominant for dense models; PS pattern lives on for sparse embeddings. |
| **Training Workers (GPU Nodes)** | Execute the forward + backward pass. Each holds a model shard (tensor/pipeline/data parallel) and owns a slice of the minibatch. |
| **Gradient Compression / Quantization Layer** | Reduces inter-node bandwidth by sending FP16 or INT8 gradients instead of FP32. |
| **Checkpoint Store** | Periodic snapshots of model weights + optimizer state to durable storage. Enables restart without full replay. |
| **Experiment Tracker** (MLflow / W&B / custom) | Logs hyperparams, loss curves, throughput (tokens/sec), GPU utilization. Connects to the monitoring stack. |
| **Feature Store** (optional) | For use cases where training on precomputed embeddings (e.g., retrieval features) is cheaper than recomputing them every epoch. |
| **Monitoring + Alerting** | GPU/CPU utilization, loss divergence, throughput drop, OOM rates. Pages on-call if training stalls. |

---

## Data Flow

A single training iteration, traced from disk to gradient update:

1. **Job submission** — A user submits a config (model arch, dataset path, hyperparams). The scheduler validates quota, queues the job, and allocates `N` GPU nodes.

2. **Worker startup** — Each worker pulls the latest checkpoint from the Checkpoint Store (or starts fresh). Workers form an AllReduce communication group via NCCL.

3. **Data prefetch** — Data Loader Workers read shards from the Object Store, tokenize/pad them, and push micro-batches into a shared queue. GPU workers pull from this queue so they're never waiting on I/O.

4. **Forward pass** — Each worker runs the forward pass on its local batch slice. With tensor parallelism, individual layers are split across GPUs within a node; with pipeline parallelism, layers are split across nodes.

5. **Loss computation** — Each worker computes its local loss. In data-parallel mode, losses are averaged across the ring in the AllReduce step.

6. **Backward pass + gradient sync** — Workers compute local gradients, then AllReduce aggregates them. Every worker ends up with the same averaged gradient — no parameter server needed.

7. **Optimizer step** — Each worker applies the gradient update locally (e.g., AdamW). Model weights are now identical across all workers.

8. **Checkpoint emit** — Every K steps, the rank-0 worker (or a dedicated checkpointing process) writes weights + optimizer state to the Checkpoint Store.

9. **Metrics emit** — Loss, throughput, LR, and gradient norms are pushed to the Experiment Tracker and Monitoring service after each step.

---

## Scaling Levers

**Data parallelism** is the first dial you turn. Replicate the full model across N workers; each processes a different batch slice. Works great until the model doesn't fit on one GPU.

**Tensor parallelism** splits individual weight matrices across GPUs in the same node. Keeps intra-node traffic on NVLink (fast). Megatron-LM pioneered this.

**Pipeline parallelism** assigns consecutive layers to consecutive nodes. You send activations forward and gradients backward over the network. Good for very deep models; introduces pipeline bubbles that hurt efficiency.

**Gradient accumulation** lets small clusters simulate large batch sizes without increasing memory. Run multiple micro-batches, accumulate gradients, then sync — fewer AllReduce calls.

**Mixed precision (BF16/FP16 + FP32 master weights)** roughly halves memory and speeds up matrix ops on modern GPUs.

**Spot / preemptible instances** (AWS Spot, GCP Preemptible) slash compute cost by 60-80% — but only if your checkpoint frequency is tight enough to survive eviction.

---

## Failure Modes

| What breaks | Symptom | Guard |
|---|---|---|
| **Node crash mid-run** | Training stalls; AllReduce hangs waiting for a dead worker | Elastic training (PyTorch Elastic / Ray Train) — detect dead workers and restart from last checkpoint with N-1 nodes |
| **Checkpoint corruption** | Restart loads bad weights; loss spikes or NaNs | Write checkpoints atomically (temp file + rename); keep last 3 checkpoints; validate checksum on load |
| **Gradient explosion / NaN** | Loss becomes `inf` or `nan`; training diverges | Gradient clipping (`max_norm`); loss scaling for FP16; halt-on-NaN callback that pages on-call |
| **Slow node (straggler)** | AllReduce latency dominated by the slowest GPU | Straggler detection + node replacement; profile per-worker throughput; async gradient updates as fallback |
| **Data pipeline starvation** | GPU utilization drops below 50%; training throughput collapses | Pre-fetch queue depth monitoring; auto-scale data loader workers; profile with `nvidia-smi` + DCGM |
| **Checkpoint store full** | Checkpoint writes fail silently; next crash is unrecoverable | Storage quota alerts at 70%; auto-evict checkpoints older than 48 h (keep epoch boundaries) |
| **Network partition** | AllReduce hangs indefinitely | NCCL timeout + watchdog thread; job requeued from last checkpoint |

---

## Cost Levers

**Right-size your instances.** A100 80 GB is expensive; for models that fit in 40 GB, the 40 GB variant is 30% cheaper. Don't auto-scale to the premium tier.

**Use spot/preemptible nodes for the bulk of training.** Reserve 1-2 on-demand nodes as "stable" coordinators; run the rest on spot. With frequent checkpoints (every 10-15 min), an eviction costs minutes, not hours.

**Reduce checkpoint I/O costs.** Writing 13 GB of weights every 100 steps to S3 adds up. Tune checkpoint frequency to the failure rate of your cluster, not to a fixed timer.

**Compress gradients.** PowerSGD or 1-bit Adam cuts AllReduce bandwidth by 4-10x — directly reducing inter-node transfer costs in cloud environments where egress is billed.

**Pack jobs with gang scheduling.** A scheduler that packs small jobs into the same node cluster reduces fragmentation and idle GPU time. Kubernetes + Volcano or SLURM fair-share does this well.

**Profile before scaling.** Doubling GPUs rarely doubles throughput. Profile step time, data loader wait, and AllReduce time first — the bottleneck is usually data I/O or a single fat layer, not raw compute.

---

## Tradeoffs & Alternatives

**PyTorch DDP vs. FSDP vs. Megatron**

- DDP (data parallel) is simple and works well for models that fit on one GPU. Gradients are synced; weights stay replicated. Easy to debug.
- FSDP shards model parameters, gradients, and optimizer state across workers. Lets you train models larger than single-GPU memory without tensor parallelism. Good default for 3B–70B range.
- Megatron-LM adds tensor + pipeline parallelism on top. Built for 100B+ models on tightly coupled GPU clusters. Harder to set up, but necessary at that scale.

**Ray Train vs. SLURM**

- SLURM is the classic HPC scheduler. Excellent for batch jobs, bad for elastic or heterogeneous workloads. Most academic clusters run it.
- Ray Train runs on Kubernetes, handles elastic workers natively, and integrates with Ray Tune for hyperparameter search. Better for cloud-native shops.

**Synchronous vs. asynchronous training**

- Synchronous AllReduce (dominant) guarantees consistent gradients. Slower if you have straggler nodes.
- Asynchronous PS-style updates allow stale gradients. Converges faster in wall-clock time on heterogeneous clusters, but often needs a lower learning rate to stay stable. Rarely used for LLM training today.

**Ties to other tracks**

- **Feature Store**: If you're training on precomputed embeddings (e.g., dense retrieval features for RAG), the feature store feeds the data pipeline — you avoid recomputing at every epoch.
- **Serving / Inference**: The checkpoint produced here is the artifact your model serving stack (vLLM, TGI, Triton) loads. Your checkpoint format (safetensors vs. PyTorch bin) should be agreed on between training and serving teams upfront.
- **Monitoring**: Training metrics (loss, throughput, GPU util) feed the same observability stack as production inference. Unified dashboards help you compare training compute cost vs. inference serving cost per token.
- **Evaluation (Track 05)**: After each epoch, you may want to run offline evals on a held-out benchmark set. Hook your experiment tracker to trigger eval jobs automatically — treat it like CI for model quality.

---

## How to present this in an interview

> "I'd start by clarifying the model size and cluster size, because those drive which parallelism strategy to use. For a model that fits on one GPU, DDP is the right answer — simple and fast. Once it outgrows a single GPU, I'd reach for FSDP or Megatron depending on whether we're on 7B or 70B+. The job scheduler owns resource allocation and preemption. Checkpoints go to durable object storage on a configurable cadence, and we design for restart from checkpoint as the default recovery path — not as an afterthought. The hardest operational problem is usually straggler nodes slowing AllReduce, so I'd add per-worker throughput monitoring with automatic node replacement. On cost, I'd default to spot instances for training workers with a small on-demand coordinator, and I'd tune checkpoint frequency to match the spot eviction rate. Finally, the output of training feeds directly into the serving stack, so I'd align on checkpoint format and model registry schema with the serving team before we write a single line of training code."

:::key-takeaway
Distributed training is a reliability + efficiency problem as much as it is an ML problem. The model code is 10% of the work; checkpoint hygiene, straggler handling, and cost-aware scheduling are the other 90%.
:::

:::why-prod
In production at Pune-based AI shops, the most common failure isn't a GPU crash — it's a data pipeline that can't keep GPUs fed. Profile GPU utilization before blaming compute capacity.
:::

:::gotcha
AllReduce hangs silently when a worker dies. Always set a NCCL timeout and run a watchdog thread that checkpoints and exits cleanly instead of hanging the whole job for hours.
:::

:::interview-line
"We treat checkpoint-and-restart as the primary recovery primitive. Any design that assumes nodes never fail is a design that has never run a long job."
:::
