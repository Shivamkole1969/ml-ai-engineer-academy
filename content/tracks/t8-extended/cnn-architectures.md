---
id: cnn-architectures
track: t8-extended
title: "CNN architectures: ResNet → EfficientNet"
badge: CORE
minutes: 9
prereqs: []
tags: [cnn, resnet, efficientnet, computer-vision, transfer-learning, production]
xp: 45
hot2026: false
---

Imagine you join a team that just shipped a product-image classifier for a marketplace. It works fine in the demo. Then your PM asks: "Can we run this on-device for 200ms latency?" You open the model card and realize it's a vanilla VGG-16 — 138 million parameters, 500 MB on disk, no way it fits on a phone. You need to swap it for something smarter. This lesson is the map for that decision.

## Why CNN architecture choice matters in production

Picking the right backbone is not an academic exercise. It affects inference latency, memory footprint, serving cost, accuracy, and whether your model even fits in a container. Most teams default to "whatever the tutorial used" and regret it later.

:::why-prod
In production, a 2× model size difference can mean 4× memory, doubled GPU cost, or a failed real-time SLA. Architecture choice is a first-class engineering decision, not an afterthought.
:::

## The 30-second history that explains everything

Early CNNs (AlexNet, VGG) just stacked convolution layers deeper. Problem: deeper = vanishing gradients = training hell. By the time you hit 20+ layers, gradients from the loss barely reach the early weights.

ResNet (2015) cracked this with one elegant idea: **residual connections** (also called skip connections). Instead of learning the full transformation, a block learns only the *residual* — the difference from the input.

```
output = F(x) + x     # x is the shortcut; F(x) is what the block learns
```

If the block does not need to change anything, it can just set F(x) → 0 and pass the input through unchanged. Gradients now have a highway straight to early layers. This let teams train networks 50, 100, even 152 layers deep — and win ImageNet.

## ResNet: the architecture that still ships

ResNet-50 is the "default backbone" of the industry. You will see it everywhere: object detection heads, embedding models, medical imaging pipelines.

:::table {title="ResNet variants at a glance"}
| Variant | Params | Top-1 (ImageNet) | When to pick it |
|---|---|---|---|
| ResNet-18 | 11 M | ~70% | Tiny dataset, edge device |
| ResNet-50 | 25 M | ~76% | General-purpose default |
| ResNet-101 | 45 M | ~77.5% | When you have GPU budget |
| ResNet-152 | 60 M | ~78.3% | Rarely worth it over 101 |
:::

The jump from ResNet-18 to ResNet-50 is often worth it. The jump from ResNet-101 to ResNet-152 usually is not — you're paying 33% more parameters for half a percent of accuracy.

## EfficientNet: the smart way to scale

In 2019, Google Brain asked a sharper question: instead of just making networks deeper, what if we scale depth, width, AND resolution together in a principled way?

They found the optimal scaling ratio using neural architecture search and called it **compound scaling**. The result — EfficientNet — beat every prior model at every parameter budget.

EfficientNet-B0 is the baseline. B1 through B7 scale it up by the compound ratio. B4 is often the sweet spot for server-side tasks; B0 or B1 for edge/mobile.

```python {title="Transfer learning: swap ResNet-50 for EfficientNet-B4" run=false}
import torchvision.models as models
import torch.nn as nn

# --- Option A: ResNet-50 (classic default) ---
backbone_resnet = models.resnet50(weights="IMAGENET1K_V2")
# Freeze all base layers, only train the head
for param in backbone_resnet.parameters():
    param.requires_grad = False
num_classes = 10
backbone_resnet.fc = nn.Linear(backbone_resnet.fc.in_features, num_classes)

# --- Option B: EfficientNet-B4 (better accuracy/param tradeoff) ---
backbone_eff = models.efficientnet_b4(weights="IMAGENET1K_V1")
for param in backbone_eff.parameters():
    param.requires_grad = False
backbone_eff.classifier[1] = nn.Linear(
    backbone_eff.classifier[1].in_features, num_classes
)

# Run locally: pip install torch torchvision
# Free GPU: Google Colab T4 (free tier)
```

## Bottleneck blocks and depthwise separable convolutions

Two efficiency tricks you will see constantly:

**Bottleneck block (ResNet-50+):** Instead of two 3×3 convolutions, use 1×1 → 3×3 → 1×1. The 1×1 layers compress and expand channels cheaply. Cuts compute by ~3×.

**Depthwise separable convolution (MobileNet, EfficientNet):** Split a standard convolution into a per-channel spatial filter (depthwise) followed by a 1×1 pointwise mix. This cuts multiply-adds by ~8–9× with minimal accuracy loss. It is why MobileNet runs on a phone.

EfficientNet uses **MBConv blocks** — mobile inverted bottlenecks with squeeze-and-excitation attention on top. This is why it beats ResNet at the same parameter count: every operation is efficient by design.

:::gotcha
Freezing the entire backbone and only training the head is safe but often suboptimal. A better practice is **gradual unfreezing**: train the head for a few epochs, then unfreeze the last few backbone blocks with a small learning rate (1e-5 or less). Unfreezing too early with a large LR destroys the pretrained weights.
:::

## How to pick in practice

The decision tree is simple:

1. **Edge / mobile / on-device** → EfficientNet-B0, B1, or MobileNetV3
2. **Server, latency < 50 ms** → EfficientNet-B4 or ResNet-50 (both are well-supported in TorchServe / TF Serving)
3. **Server, accuracy is king** → EfficientNet-B7 or a ConvNeXt variant
4. **You need pretrained embeddings for retrieval** → ResNet-50 or ViT (covered in the Vision Transformers lesson)

Do not over-engineer. If ResNet-50 already hits your accuracy target, stop there.

:::war-story {title="The $4,000/month backbone mistake"}
A team built a fashion similarity search using EfficientNet-B7 fine-tuned on 50,000 product images. Accuracy was great. Then they autoscaled and the GPU bill tripled — B7 needs nearly 2× the FLOPs of B4 at inference. Swapping to B4 dropped accuracy by 0.3% (users did not notice) and cut inference cost by 40%. The lesson: benchmark on your *real query distribution* before committing to a backbone in production.
:::

:::interview-line
"I choose a CNN backbone based on three axes: parameter budget, latency SLA, and dataset size — EfficientNet-B4 is my go-to server-side default because it wins the accuracy-per-FLOP tradeoff across most benchmarks."
:::

:::qa {q="What problem did ResNet solve that earlier deep CNNs couldn't?"}
Vanishing gradients. As networks go deeper, gradients shrink to near-zero before reaching early layers, so those layers stop learning. ResNet's skip connections give gradients a direct path back through the network, enabling stable training at 50–150+ layers.
:::

:::qa {q="Why does EfficientNet outperform ResNet at the same parameter count?"}
EfficientNet scales depth, width, and input resolution simultaneously using a compound coefficient derived from neural architecture search. It also uses MBConv blocks with depthwise separable convolutions and squeeze-and-excitation, making each parameter do more work than a plain conv layer in ResNet.
:::

:::qa {q="When would you NOT choose EfficientNet for a production vision model?"}
When you need maximum ecosystem support and simplicity: ResNet is more widely supported in older deployment stacks, easier to export to older ONNX versions, and better understood by most ML infrastructure teams. For a fast-moving team with standard infra, EfficientNet is fine — but know your stack.
:::

:::drill {type="mcq" q="Your model needs to run on a mobile device with < 20 MB memory budget. Which backbone is the best starting point?"}
- [ ] ResNet-152 — deepest, highest accuracy
- [ ] EfficientNet-B7 — best overall benchmark
- [x] EfficientNet-B0 or MobileNetV3 — designed for edge, tiny footprint
- [ ] ResNet-50 — the safe default
:::

:::drill {type="mcq" q="You fine-tune a ResNet-50 on 5,000 images and the model converges fast but validation accuracy stops improving early, while training accuracy keeps rising. What is the most likely cause?"}
- [ ] The backbone's skip connections are interfering with the head
- [x] Overfitting — the model memorizes the small dataset; use data augmentation, dropout, or stronger weight decay
- [ ] EfficientNet would not have this problem
- [ ] The bottleneck blocks need to be unfrozen first
:::

:::key-takeaway
ResNet gave us deep networks that actually train (skip connections fix vanishing gradients). EfficientNet gave us the best accuracy-per-parameter ratio (compound scaling + MBConv). For most production tasks, start with EfficientNet-B4 server-side or B0/B1 on-device, and always benchmark latency on your actual hardware before committing.
:::
