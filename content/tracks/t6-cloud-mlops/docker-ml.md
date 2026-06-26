---
id: docker-ml
track: t6-cloud-mlops
title: "Containers: Docker for ML services"
badge: HOT
minutes: 9
prereqs: []
tags: [docker, containers, mlops, deployment, reproducibility, cuda, gpu]
xp: 60
hot2026: true
---

Your model scores beautifully in the notebook. You hand the weights to the platform team and two hours later they message you: *"It crashes immediately."* You compare versions. Their Python is 3.10, yours is 3.11. Their NumPy is 1.24, yours is 1.26. Their machine has CUDA 11.8; the cloud instance has 12.1. Nothing is aligned. The model works, but the *environment* is a disaster. That is the exact problem Docker was built to solve — and it matters even more for ML, where a single mismatched CUDA minor version can silently corrupt your GPU output rather than crashing cleanly.

## What a container actually is

Think of a container as a lightweight, hermetically sealed box. Inside the box lives your code, your Python version, your packages, your system libraries — everything the model needs to run — frozen at the exact state you tested it. You ship the box instead of shipping instructions about how to rebuild the box on someone else's machine.

The key parts:

- **Image** — the read-only snapshot (the blueprint of the box).
- **Container** — a live running instance of that image.
- **Dockerfile** — the recipe you write to build the image.
- **Registry** — a remote store (Docker Hub, AWS ECR, GCP Artifact Registry) where you push images so teammates and cloud services can pull them.

:::why-prod
In production, "works on my machine" is a zero-value claim. Containers make the environment itself part of the artifact you ship, version, and roll back. Every inference server, every batch job, every CI pipeline runs the same image that passed your tests — no surprises at 3 a.m.
:::

## Writing a Dockerfile for an ML service

An ML Dockerfile has a few concerns a typical web app doesn't:

1. **Base image** — pick one with the right CUDA toolkit baked in if you use a GPU.
2. **Layer caching** — install dependencies before copying your code, or every code change rebuilds the whole package layer.
3. **Model weights** — almost always loaded at runtime, not baked into the image (more on this in the gotcha below).
4. **Non-root user** — security baseline for production.

```python {title="Dockerfile for a PyTorch inference service" run=false}
# Base image: CUDA 12.1 + Python 3.11 (NVIDIA provides these on Docker Hub)
# For CPU-only: use python:3.11-slim instead — much smaller
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Copy requirements FIRST so this layer is cached ----
# Docker re-uses this layer if requirements.txt didn't change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy the rest of your code AFTER packages ----
COPY . .

# Run as non-root (security)
RUN useradd -m appuser
USER appuser

# Model weights are NOT baked in — they're mounted or downloaded at startup
# See MODEL_PATH env var in your entrypoint script
EXPOSE 8000
CMD ["python3", "serve.py"]
```

```python {title="Build, run, and push — the three commands you'll use daily" run=false}
# Build (run from the directory with your Dockerfile)
docker build -t my-model-service:v1.0 .

# Run locally on GPU (remove --gpus if CPU-only)
docker run --gpus all -p 8000:8000 \
  -e MODEL_PATH=/models/my_model.pt \
  -v /local/models:/models \
  my-model-service:v1.0

# Push to a registry (AWS ECR example — swap for Docker Hub / GCR)
docker tag my-model-service:v1.0 123456789.dkr.ecr.us-east-1.amazonaws.com/my-model-service:v1.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/my-model-service:v1.0
```

## Layer caching: your best friend for fast iteration

Docker builds images as a stack of layers. Each `RUN`, `COPY`, and `ADD` instruction is one layer. If a layer's input hasn't changed, Docker reuses the cached version — skipping it entirely.

:::table {title="Layer caching rules of thumb"}
| What changed | Layers rebuilt |
|---|---|
| Only your Python code | Code layer onward (fast) |
| `requirements.txt` | Pip install layer + code (medium) |
| Base image version | Everything (slow — plan upgrades) |
| `COPY . .` before pip install | Always rebuilds packages (the classic mistake) |
:::

The rule: put things that change rarely at the **top** of the Dockerfile, things that change often at the **bottom**.

## Multi-stage builds: keeping images lean

CUDA base images can be 5–8 GB. You don't need the full compiler toolkit at inference time — only the runtime libraries. Multi-stage builds let you build heavy things in one stage and copy only the outputs into a lighter final image.

```python {title="Multi-stage: slim runtime image" run=false}
# Stage 1: builder — has compilers, build tools
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# Stage 2: runtime — lean, no compilers
FROM python:3.11-slim
COPY --from=builder /install /usr/local
WORKDIR /app
COPY . .
USER nobody
CMD ["python", "serve.py"]

# Result: often 60–80% smaller than a naive single-stage build
```

:::gotcha
Never bake model weights into the image. A 7B-parameter model is 14 GB. Your image becomes enormous, every push/pull takes forever, and changing the model forces a full image rebuild and re-push. Instead, pass the weights as a mounted volume (`-v`) or download them from object storage (S3, GCS) at container startup using an environment variable for the path. The image is the *code*; the weights are *data*.
:::

:::war-story {title="The silent CUDA minor-version massacre"}
A team shipped an image based on CUDA 11.7 to staging and it passed all tests — then re-tagged the same image for production, where the hosts had CUDA 12.0 drivers. PyTorch loaded without a single error. Latency looked normal. But the P99 numbers for a specific matrix multiplication path were 40% higher than expected. Three days of profiling later, a CUDA operations researcher noticed a cublas kernel was silently falling back to a slower compatibility path because the minor version mismatch triggered a runtime fallback. The fix: pin the exact CUDA version in the Dockerfile and test the same image on hardware that matches production — no re-tagging, no "close enough" substitutions.
:::

## Image tagging strategy

In a real pipeline you want three types of tags on the same image:

- `latest` — dangerous in production, fine for local dev
- `git-sha` (e.g. `abc1234`) — immutable, traceable to exact code
- `semver` (e.g. `v1.3.0`) — for human-readable release notes

In CI, build once, push all three tags pointing to the same image digest. Your deployment manifests should reference the `git-sha` tag, never `latest`.

:::interview-line
"A container packages the entire runtime environment, so the environment itself becomes a versioned artifact — not just the code. That's why we can roll back a deployment and get identical behavior."
:::

:::qa {q="Why not just use a virtual environment instead of Docker for ML deployments?"}
A virtualenv isolates Python packages but shares the host OS, system libraries, and hardware drivers. If staging and production have different CUDA versions, libc versions, or even kernel-level library paths, a virtualenv won't protect you. Docker packages everything from the Python interpreter down to the system libs into one portable artifact, so behavior is consistent across any host running the Docker runtime.
:::

:::qa {q="How do you handle GPU access inside a Docker container?"}
You install the NVIDIA Container Toolkit on the host, which hooks into Docker's runtime. Then you pass `--gpus all` (or a specific GPU index) when running the container. The container sees the GPU via the host driver — so the host's CUDA driver version must be equal to or newer than the CUDA version your image was built against. The image carries the CUDA libraries; the host carries the driver. Mixing these up is the most common GPU container failure.
:::

:::qa {q="How do you keep ML Docker images from becoming huge?"}
Three levers: use the `-slim` or `-runtime` base images instead of `-devel` variants; use multi-stage builds to discard compilers and build artifacts; and never bake model weights into the image — load them from object storage at startup. A clean PyTorch inference image should sit under 3 GB in most cases; a naive approach can easily reach 15 GB+.
:::

:::drill {type="mcq" q="You updated only your FastAPI route handler — no new packages. Where does a Docker rebuild pick up from (assuming a well-written Dockerfile)?"}
- [ ] From the base image (rebuilds everything)
- [ ] From the `pip install` step (reinstalls all packages)
- [x] From the `COPY . .` step (reuses the cached package layer)
- [ ] Docker always does a full rebuild unless you pass --cache-from
:::

:::drill {type="mcq" q="Your production instance has an NVIDIA driver supporting CUDA 12.2. Your Docker image was built with the CUDA 12.4 runtime base. What happens?"}
- [ ] Everything works — CUDA is always backward compatible
- [x] The container will fail to start; host driver version must be >= image CUDA version
- [ ] PyTorch auto-downgrades CUDA at import time
- [ ] GPU is invisible but CPU inference still works
:::

:::key-takeaway
A Docker image is not just deployment packaging — it IS the environment. Pin your base image version (especially CUDA), copy requirements before code, load weights at runtime not build time, and always test the exact same image digest that you push to production.
:::
