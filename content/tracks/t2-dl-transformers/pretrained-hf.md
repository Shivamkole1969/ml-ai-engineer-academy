---
id: pretrained-hf
track: t2-dl-transformers
title: "Using pretrained models (Hugging Face)"
badge: HOT
minutes: 9
prereqs: []
tags: [huggingface, transformers, pipeline, inference, pretrained, nlp, production]
xp: 60
hot2026: true
---

Imagine your product manager walks in on a Monday morning and says: "We need sentiment analysis on customer reviews by Thursday." Training a transformer from scratch would take weeks of data labeling, GPU time, and tuning. Instead, you open a browser, type three lines of Python, and you're done before lunch. That's the Hugging Face ecosystem in a nutshell — and it's why every AI team in the world uses it.

## What the Hub actually is

The [Hugging Face Hub](https://huggingface.co/models) is a public registry with over 800,000 model checkpoints (and growing). Think of it as npm for neural networks.

Each model on the Hub comes with a **model card** — a description of what the model does, what data it was trained on, its license, and its known limitations. Before you ship anything to production, read the model card. It's the single most important document for that model.

Three things to check on every model card:

- **License** — MIT and Apache-2.0 are commercially safe. `non-commercial` is not.
- **Training data** — biases in the training data become biases in your product.
- **Benchmark scores** — on what dataset? Is that dataset close to your domain?

:::why-prod
In production you almost never train a model from zero. Pretrained models compress years of compute and terabytes of data into a file you can download in minutes. The real skill is choosing the right model and wiring it into your system correctly — not rebuilding transformers from scratch.
:::

## The Pipeline API: zero-to-prediction in three lines

The `pipeline` function from the `transformers` library is the fastest on-ramp. You name a task, optionally name a model, and you get a callable that handles tokenization, inference, and decoding for you.

```python {title="Sentiment analysis with pipeline" run=false}
# pip install transformers torch   (or: pip install transformers tf for TensorFlow)
from transformers import pipeline

# Hugging Face downloads and caches the model on first run (~500 MB for distilbert)
classifier = pipeline(
    task="sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

results = classifier([
    "The model deployment went flawlessly.",
    "Three hours of downtime on a Friday. Not great.",
])

for r in results:
    print(r["label"], f"{r['score']:.2%}")
# POSITIVE 99.95%
# NEGATIVE 99.82%
```

The first call downloads the model weights and caches them in `~/.cache/huggingface/hub`. Every subsequent call loads from disk — no re-download.

:::table {title="Common pipeline tasks"}
| Task string | What it does | Popular model |
|---|---|---|
| `sentiment-analysis` | Positive / Negative label + score | distilbert-sst-2 |
| `text-generation` | Autocomplete / generation | gpt2, mistral |
| `summarization` | Condense long text | facebook/bart-large-cnn |
| `translation_en_to_fr` | Translate English → French | Helsinki-NLP/opus-mt |
| `ner` | Named entity recognition | dslim/bert-base-NER |
| `zero-shot-classification` | Classify without fine-tuning | facebook/bart-large-mnli |
| `feature-extraction` | Raw embeddings | sentence-transformers/... |
:::

## AutoModel and AutoTokenizer: when you need control

The `pipeline` API is magical for prototyping but hides everything. In production you often need to:

- Run on a GPU you control
- Batch requests manually for throughput
- Get raw logits instead of decoded labels
- Keep the model in memory across many requests (don't re-instantiate per call)

That's when you drop to `AutoTokenizer` and `AutoModelFor*`:

```python {title="AutoModel for batch inference on GPU" run=false}
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Load once, at startup — not inside a request handler
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def predict_batch(texts: list[str]) -> list[dict]:
    # Tokenize: padding=True pads short sequences, truncation=True clips long ones
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():                  # no gradients needed at inference time
        logits = model(**enc).logits       # shape: (batch_size, num_labels)

    probs = logits.softmax(dim=-1).cpu()
    labels = model.config.id2label        # {0: "NEGATIVE", 1: "POSITIVE"}

    return [
        {"label": labels[p.argmax().item()], "score": p.max().item()}
        for p in probs
    ]

print(predict_batch(["Great product!", "Terrible support."]))
```

The key production habit: **load the model once at startup**, not on every API request. Loading weights from disk takes 1–5 seconds and defeats the purpose of a fast endpoint.

:::gotcha
`pipeline("sentiment-analysis")` with no model argument downloads the default model silently — and that default can change across library versions. Pin the model name explicitly in production code (e.g., `model="distilbert-base-uncased-finetuned-sst-2-english"`) so a `pip upgrade` can't swap your model without you noticing.
:::

## How fine-tuned vs. base models differ

The Hub has two flavours of checkpoint:

**Base model** (e.g., `bert-base-uncased`) — a general language model. It knows language but can't classify sentiment. Use as a starting point for fine-tuning.

**Fine-tuned model** (e.g., `distilbert-base-uncased-finetuned-sst-2-english`) — the same architecture, but its head was trained on a specific task. Download and use directly.

If you find a fine-tuned model close to your domain, start there. Fine-tuning it further on your own labeled data (even 500–1000 examples) usually beats starting from the base model.

:::war-story {title="The model that evaporated overnight"}
A team's internal search feature depended on a third-party model they fetched at runtime with no version pin — just `from_pretrained("some-org/great-model")`. One morning, the model owner deleted the repo to comply with a licensing request. The service started crashing at 3 AM with a 404 on the weights download. The fix was a two-hour scramble to swap in a cached backup. The lesson: in production, always snapshot model weights to your own artifact store (S3, GCS, a private Hub repo) and never depend on external repos staying up.
:::

:::interview-line
"We never pull models from the Hub at runtime in production — we snapshot weights to our own artifact store at deploy time so a deleted repo can't take down our service."
:::

:::qa {q="What is the difference between pipeline() and AutoModel in Hugging Face?"}
`pipeline()` is a high-level wrapper that bundles tokenization, model inference, and post-processing into one callable — great for prototyping. `AutoModel` (paired with `AutoTokenizer`) gives you direct access to logits, lets you control batching and device placement, and is what you use when building production inference services. Both ultimately run the same model weights.
:::

:::qa {q="How do you pick a pretrained model for a new task on the Hub?"}
Start by filtering by task type, then check the license for commercial use. Read the model card's training data section to gauge domain match, look at benchmark scores on datasets close to your distribution, and check the download count and recency as a proxy for community health. Then run a quick offline evaluation on a sample of your own data before committing to the model.
:::

:::drill {type="mcq" q="You load a Hugging Face model inside your FastAPI request handler (not at startup). What is the most likely production problem?"}
- [ ] The model will use fp16 instead of fp32 automatically
- [ ] The tokenizer will fail to pad short sequences
- [x] Each request will take several extra seconds to load weights from disk, causing high latency
- [ ] The Hub rate-limits model downloads after 100 requests per day
:::

:::drill {type="mcq" q="A colleague pins the model as pipeline('text-classification') with no model argument. What is the key risk?"}
- [ ] The pipeline will throw a DeprecationWarning in newer versions
- [ ] The default model does not support GPU acceleration
- [x] A library upgrade could silently swap the default model, changing predictions in production
- [ ] The model card cannot be read without specifying a model name
:::

:::key-takeaway
The Hugging Face Hub gives you production-grade transformer models in three lines — but using them correctly means reading the model card for license and bias, pinning the exact model name, loading weights once at startup (not per request), and snapshotting weights to your own storage so external deletions can't break your service.
:::
