---
id: aws-sagemaker
track: 15-cloud-genai
title: "AWS SageMaker: Training Jobs, Endpoints, and Pipelines"
badge: CORE
minutes: 10
prereqs: []
tags: [aws, sagemaker, mlops, training-jobs, endpoints, pipelines, s3, ec2]
xp: 45
hot2026: false
---

It's Monday morning. Your manager has just handed you a Jupyter notebook that trains a sentiment model on 10 GB of customer reviews. It works fine on your laptop — with 2 GB of data, after 40 minutes. He wants the full run done by EOD and the model behind an API by tomorrow.

Your laptop says no. AWS SageMaker says yes.

SageMaker is AWS's managed ML platform. It handles the annoying parts: provisioning GPU boxes, shutting them down when done, storing artefacts, and wrapping your model in a REST endpoint — all without you SSHing into anything.

Let's break it into the pieces that actually matter for interviews and day-to-day work.

## S3: Where Everything Lives

Before SageMaker touches your data, it needs to be in S3 (Simple Storage Service). Think of S3 as a giant, durable, cheap hard drive in the cloud — organised into **buckets** (like top-level folders) and **objects** (the files inside).

SageMaker reads training data from S3, writes model artefacts back to S3, and pulls them again at endpoint deploy time. You can't skip this step.

```python {title="Upload training data to S3" run=false}
import boto3

s3 = boto3.client("s3", region_name="ap-south-1")  # Mumbai region — lowest latency from Pune

# Create bucket once
s3.create_bucket(
    Bucket="my-ml-project-data",
    CreateBucketConfiguration={"LocationConstraint": "ap-south-1"},
)

# Upload your CSV / parquet / whatever
s3.upload_file(
    Filename="data/reviews_train.csv",
    Bucket="my-ml-project-data",
    Key="sentiment/train/reviews_train.csv",
)

print("Data is now at s3://my-ml-project-data/sentiment/train/reviews_train.csv")
```

:::why-prod
Every SageMaker component — training jobs, endpoints, pipelines — uses S3 as its source of truth. Getting your S3 paths wrong is the #1 cause of "job failed immediately" incidents. Keep a consistent naming convention like `s3://<project>/<model-name>/<split>/` from day one.
:::

## EC2 in One Sentence

EC2 is AWS's raw virtual machine service. SageMaker runs on EC2 under the hood, but you almost never manage it directly. You just tell SageMaker *what instance type* you want (e.g. `ml.g4dn.xlarge` for GPU, `ml.m5.large` for CPU-only) and it handles provisioning and teardown. The "EC2 basics" you need: understand that instance types encode compute class (`m` = general, `c` = compute, `g`/`p` = GPU) and size (`large`, `xlarge`, `2xlarge`…). Bigger = more expensive per hour.

:::table {title="Common SageMaker instance types"}
| Instance | vCPU | RAM | GPU | Use case | On-demand cost (ap-south-1) |
|---|---|---|---|---|---|
| ml.t3.medium | 2 | 4 GB | — | Studio notebooks, light dev | ~$0.05 / hr |
| ml.m5.xlarge | 4 | 16 GB | — | CPU training, batch transform | ~$0.23 / hr |
| ml.g4dn.xlarge | 4 | 16 GB | 1× T4 | Fine-tuning, inference | ~$0.74 / hr |
| ml.p3.2xlarge | 8 | 61 GB | 1× V100 | Heavy training | ~$3.83 / hr |
:::

## Training Jobs: Managed, Ephemeral, Auditable

A **Training Job** is the SageMaker primitive for running your training script. You point it at your code, pick an instance type, and SageMaker:
1. Spins up the instance.
2. Pulls your container image (built-in or custom).
3. Copies S3 data onto the box.
4. Runs your script.
5. Copies the `model.tar.gz` output back to S3.
6. Terminates the instance.

You pay only for the time the job runs. No instance left idling overnight.

```python {title="Launch a SageMaker Training Job (PyTorch)" run=false}
import sagemaker
from sagemaker.pytorch import PyTorch

role = "arn:aws:iam::123456789012:role/SageMakerExecutionRole"  # set up once in IAM

estimator = PyTorch(
    entry_point="train.py",           # your training script
    source_dir="./src",               # folder with train.py + requirements.txt
    role=role,
    instance_type="ml.g4dn.xlarge",   # 1 GPU; cheapest GPU option
    instance_count=1,
    framework_version="2.1",
    py_version="py310",
    output_path="s3://my-ml-project-data/sentiment/model-artefacts/",
    hyperparameters={"epochs": 3, "batch-size": 32},
)

estimator.fit(
    {"train": "s3://my-ml-project-data/sentiment/train/",
     "val":   "s3://my-ml-project-data/sentiment/val/"}
)
# Blocks until done. Logs stream to your terminal in real time.
```

## Endpoints: Your Model as a REST API

After training, `estimator.deploy()` creates a **Real-Time Endpoint** — a load-balanced, auto-scaling REST API running your model. SageMaker manages the container, health checks, and TLS.

```python {title="Deploy and call a SageMaker endpoint" run=false}
# Deploy — takes ~5 minutes to come live
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",       # CPU fine for inference if batch < 100ms SLA
    endpoint_name="sentiment-v1",
)

# Call it — SageMaker wraps request/response serialisation
result = predictor.predict({"text": "Delivery was late but the product is great!"})
print(result)  # {"label": "POSITIVE", "score": 0.91}

# IMPORTANT: delete when done or you'll pay 24×7
predictor.delete_endpoint()
```

:::gotcha
Endpoints keep billing even when no one is calling them. A `ml.g4dn.xlarge` endpoint left running over a weekend costs ~$53. Always delete after demos. In staging, use **Serverless Inference** (pay-per-request, cold-start ~1 s) instead of always-on instances.
:::

## Pipelines: Repeatable, Auditable ML Workflows

A **SageMaker Pipeline** chains steps — data processing, training, evaluation, conditional model registration — into a DAG you can trigger on demand or on a schedule. Think of it as your CI/CD for ML.

Key steps you'll compose:

- `ProcessingStep` — run a script to clean / split data (SageMaker Processing job under the hood)
- `TrainingStep` — the training job from above
- `ConditionStep` — "only register the model if validation F1 > 0.85"
- `ModelStep` — register to the Model Registry
- `TransformStep` — batch scoring

Pipelines log every run with parameters, metrics, and artefact lineage. That's what interviewers mean when they ask "how do you ensure reproducibility?"

:::interview-line
"We run everything through SageMaker Pipelines — each run is immutable: same code, same data hash, same hyperparameters — so if a model regresses in prod we can one-click reproduce the exact training run that built it."
:::

:::qa {q="What is the difference between a SageMaker Training Job and a SageMaker Endpoint?"}
A Training Job is an ephemeral compute run — it starts, trains, saves artefacts to S3, and terminates. You pay only while it runs. An Endpoint is a persistent, always-on REST service that loads the artefacts and serves predictions; billing continues until you delete it. The split keeps compute costs low and inference paths flexible.
:::

:::qa {q="When would you choose Serverless Inference over a Real-Time Endpoint?"}
Serverless Inference suits low or spiky traffic where you can tolerate a ~1 second cold start. You pay per invocation instead of per hour, so the cost is near zero when idle. A Real-Time Endpoint is better when you have consistent traffic and need sub-100 ms P99 latency, because the container is always warm.
:::

:::qa {q="What role does S3 play in a typical SageMaker workflow?"}
S3 is the glue. Training data goes in, model artefacts come out, and Pipelines read / write intermediate artefacts between steps. Everything is referenced by S3 URI, which also gives you a free audit trail and makes it trivial to version datasets by date or experiment prefix.
:::

:::drill {type="mcq" q="A SageMaker Training Job has finished. Where is the trained model artefact saved by default?"}
- [ ] Inside the training container image
- [ ] In the SageMaker Model Registry automatically
- [x] In the S3 output_path you specified when creating the estimator
- [ ] On the EC2 instance that ran the job
:::

:::drill {type="mcq" q="You deploy a SageMaker Real-Time Endpoint on Friday for a demo and forget about it over the weekend. What happens?"}
- [ ] SageMaker auto-deletes idle endpoints after 24 hours
- [ ] The endpoint pauses and you are not billed
- [x] The endpoint keeps running and you are billed for every hour it exists
- [ ] AWS sends an alert and shuts it down after 48 hours
:::

:::drill {type="mcq" q="Which SageMaker Pipeline step would you use to enforce 'only register the model if F1 > 0.85'?"}
- [ ] TrainingStep with an early-stopping callback
- [ ] ModelStep with a quality gate parameter
- [x] ConditionStep that branches to ModelStep only when the metric condition is met
- [ ] ProcessingStep that raises an exception on low F1
:::

:::key-takeaway
SageMaker = S3 for storage + ephemeral Training Jobs for compute + always-on Endpoints for serving + Pipelines for reproducibility. Master this four-part mental model and you can navigate any AWS ML interview or on-call incident.
:::
