---
id: vision-transformers
track: t8-extended
title: "Vision Transformers & CLIP"
badge: CORE
minutes: 9
prereqs: []
tags: [vision, transformers, CLIP, multimodal, embeddings, zero-shot]
xp: 45
hot2026: false
---

A user types "red floral dress for summer wedding" into your e-commerce search bar. No image, just words. Your CNN-based image classifier stares back blankly — it only knows the 200 labels you trained it on. "Dress" maybe. "Floral"? Not a chance. The user bounces.

This is the exact problem that made the ML world rethink how computers understand images. Two ideas cracked it open: **Vision Transformers (ViT)**, which brought the Transformer architecture from NLP into vision, and **CLIP**, which taught a model to connect images and text in a shared space. Together, they power modern image search, zero-shot classification, and multimodal AI — the backbone of most serious vision systems built today.

## Why CNNs Hit a Ceiling

Convolutional Neural Networks (CNNs) are brilliant at what they do. They scan images patch-by-patch, building up from edges to shapes to objects. That built-in "look locally first" behaviour is called an **inductive bias**, and it works well when you have limited data.

But inductive biases are also a ceiling. CNNs struggle to model relationships between distant parts of an image — the dog's collar and its owner standing across the frame, for instance. Transformers, which can attend to any part of the input from the very first layer, don't have that ceiling. They scale better, and at large dataset sizes, they outperform CNNs.

The trade-off: Transformers need a lot more data to learn what CNNs get for free. When you have millions of images (or can access a pretrained checkpoint), ViT wins. When you're training from scratch on 10,000 images, a CNN is still your friend.

## The Patch Game: How ViT Sees Images

A standard Transformer takes a sequence of tokens as input. So how do you feed it an image?

You chop the image into fixed-size square patches — typically 16×16 pixels. Each patch is flattened into a vector, linearly projected to a fixed dimension, and treated as one token. A 224×224 image becomes 196 patches (14×14 grid). Add a special `[CLS]` token at the front, attach **learnable positional embeddings** so the model knows where each patch came from, and you have your sequence.

From there, it's a standard Transformer encoder: multi-head self-attention, feed-forward layers, repeat N times. The `[CLS]` token's final representation is used for classification.

That's it. No convolutions. No sliding windows. Just attention.

:::why-prod
ViT checkpoints (especially `ViT-L/16` and `ViT-H/14`) are the backbone of most modern vision APIs. If you're calling a cloud vision service in production, there's a good chance a ViT is doing the heavy lifting. Understanding its behavior — especially its data-hunger and patch size sensitivity — lets you debug failures faster and choose the right pretrained model for your use case.
:::

:::table {title="ViT vs CNN: when to reach for each"}
| Dimension | CNN (ResNet, EfficientNet) | Vision Transformer (ViT) |
|---|---|---|
| Data needed | Works with ~10K images | Shines at 1M+ (or pretrained) |
| Local patterns | Built-in via convolution | Learned via attention |
| Long-range context | Weak | Strong |
| Interpretability | Grad-CAM works well | Attention maps available |
| Serving latency | Faster on edge devices | Faster on GPU/TPU at scale |
| Best starting point | Fine-tuning on small datasets | Transfer learning from large checkpoints |
:::

## CLIP: One Space for Images and Words

CLIP (Contrastive Language-Image Pre-training, from OpenAI, 2021) does something elegant. It trains two encoders — one for images, one for text — and pulls them into a **shared embedding space**. If an image and a caption go together, their embeddings should be close. If they don't belong together, they should be far apart.

The training signal: 400 million (image, caption) pairs scraped from the web. For each batch of N pairs, it maximizes similarity for the N correct pairings and minimizes it for the N²−N wrong ones. This is called **contrastive learning**.

What comes out is remarkable: you can embed any text string and any image into the same space, then compare them with a dot product. No task-specific fine-tuning needed.

**Zero-shot classification** falls out naturally. To classify an image as "cat" or "dog," you embed both strings as `"a photo of a cat"` and `"a photo of a dog"`, embed the image, and pick the closest text. On ImageNet, CLIP with zero labeled examples matches a ResNet-50 trained on the full dataset.

```python {title="Zero-shot image classification with CLIP" run=false}
# pip install transformers torch Pillow
# Free to run locally on CPU (slow) or any GPU

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests, torch

model = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load any image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Define your candidate labels — no training needed
labels = ["a photo of a dog", "a photo of a cat", "a photo of a car"]

inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# Softmax over label similarities → probability distribution
probs = outputs.logits_per_image.softmax(dim=1)
for label, prob in zip(labels, probs[0]):
    print(f"{label}: {prob:.2%}")

# Output (roughly):
# a photo of a dog: 97.3%
# a photo of a cat:  2.1%
# a photo of a car:  0.6%
```

:::gotcha
CLIP's text encoder was trained on English-heavy data. Zero-shot accuracy on multilingual captions drops significantly. If your user base writes queries in other languages, either translate first or reach for a multilingual CLIP variant (like `mCLIP`). Assuming "it handles all languages" is a common production mistake.
:::

## What This Enables in Production

CLIP embeddings unlock a family of powerful features without labeling a single image:

- **Semantic image search**: store image embeddings in a vector DB, query with text at runtime.
- **Content moderation**: embed banned categories as text, flag images that land close in the shared space.
- **Product tagging**: autogenerate descriptive labels from a candidate list — no annotation pipeline.
- **Multimodal RAG**: retrieve relevant images and text together in a single retrieval step.

ViT alone powers classification, detection backbones, and image-to-image retrieval. CLIP unlocks the cross-modal use cases. In practice, modern vision stacks often use ViT inside CLIP — you get both.

:::interview-line
"ViT treats image patches as tokens and runs a standard Transformer encoder over them; CLIP adds a text encoder and trains both with contrastive loss so images and captions land close in a shared embedding space — enabling zero-shot classification and semantic image search without any labeled data."
:::

:::qa {q="What is the role of the [CLS] token in a Vision Transformer?"}
The `[CLS]` (classification) token is a learnable token prepended to the patch sequence before the Transformer encoder. Because all tokens attend to each other through self-attention, the `[CLS]` token aggregates information from every patch by the final layer. Its output representation is then used as the image-level feature vector for downstream tasks like classification.
:::

:::qa {q="How does CLIP achieve zero-shot classification without any task-specific training?"}
CLIP trains image and text encoders jointly on hundreds of millions of image-caption pairs using contrastive loss, placing matching pairs close together in a shared embedding space. At inference time you describe each candidate class as a natural language string (e.g., "a photo of a cat"), embed both the image and all class strings, and pick the class whose embedding is nearest the image embedding. No labeled examples or fine-tuning are required.
:::

:::drill {type="mcq" q="A ViT-B/16 model processes a 224×224 image. How many patch tokens does it feed into the Transformer (excluding the [CLS] token)?"}
- [ ] 16
- [ ] 64
- [x] 196
- [ ] 256
:::

:::drill {type="mcq" q="You have 5,000 labeled product images and need to ship an image classifier next week. Which approach is most pragmatic?"}
- [ ] Train ViT-L/16 from scratch on your 5,000 images
- [x] Fine-tune a pretrained CNN (e.g., EfficientNet) on your 5,000 images
- [ ] Use raw CLIP embeddings without any fine-tuning
- [ ] Collect 1M more images before starting
:::

:::key-takeaway
ViT replaced hand-crafted convolutions with pure attention over image patches and scales better than CNNs given enough data. CLIP extended this into a shared image-text embedding space, making zero-shot classification and semantic image search practical. Together they define the modern production vision stack.
:::
