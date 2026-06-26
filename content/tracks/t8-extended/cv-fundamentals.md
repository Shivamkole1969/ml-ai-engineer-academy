---
id: cv-fundamentals
track: t8-extended
title: "Computer Vision fundamentals"
badge: CORE
minutes: 9
prereqs: []
tags: [computer-vision, image-processing, convolution, deep-learning, cnn, production-ml]
xp: 45
hot2026: false
---

Imagine you work at an e-commerce company and sellers upload product photos. Within weeks of launch, your recommendation engine starts surfacing blurry thumbnails, misrotated items, and the occasional photo of someone's lunch. The culprit? Nobody validated image quality before features were extracted. A few engineers who understood computer vision basics could have caught this on day one. That is what this lesson gives you.

## What a Machine Sees When It Looks at an Image

To you, a photo is a cat or a sunset. To a model, it is a 3-dimensional array of numbers — a **tensor** shaped `(height, width, channels)`. A typical RGB image that is 224 × 224 pixels becomes a tensor of shape `(224, 224, 3)` where the three channels hold red, green, and blue intensities, each ranging from 0 to 255.

That grid of numbers is the only thing your model ever touches. Every CV technique — from edge detection to object recognition — is ultimately a set of mathematical operations on that array.

Two things trip up beginners here:

- **Channel order**: PyTorch expects `(C, H, W)` — channels first. OpenCV loads images as `(H, W, C)` — channels last — and in BGR order, not RGB. Mixing these up silently corrupts every image your model ever sees.
- **Value range**: Most pretrained models expect pixel values normalized to roughly `[0, 1]` or `[-1, 1]`, not the raw `[0, 255]` range. Forgetting to normalize gives the model wildly out-of-distribution inputs.

:::why-prod
Production CV pipelines break in silent, hard-to-debug ways. A wrong channel swap looks like a "bad model" until you squint at a prediction on a bright-red image and notice the model thinks it's blue. Getting the tensor format and normalization right at the data loading step saves weeks of debugging downstream.
:::

## The Four Core CV Tasks

Before you pick an architecture or a loss function, you need to know which CV task you are solving. Each task has a different output format, evaluation metric, and training strategy.

:::table {title="Core computer vision tasks"}
| Task | What the model outputs | Example | Common metric |
|---|---|---|---|
| Classification | A single class label per image | "This X-ray shows pneumonia" | Accuracy, AUC |
| Object detection | Bounding boxes + class labels | "Car at [x1,y1,x2,y2], confidence 0.92" | mAP (mean Average Precision) |
| Semantic segmentation | A class label per pixel | Colour every road pixel orange | mIoU |
| Instance segmentation | Per-pixel + per-object identity | Separate masks for each car | mAP mask |
:::

In production you almost never build these from scratch — you fine-tune a pretrained backbone. But you must know which task you are solving before you choose a loss, because a classification cross-entropy head will not give you bounding boxes.

## Convolution: the Operation That Makes It Work

A **convolution** slides a small filter (say, 3 × 3 numbers) across the image, computing a dot product at each position. The result is a **feature map** — a new array that highlights whatever pattern the filter detected (edges, corners, textures).

What makes neural networks powerful is that these filters are *learned*, not hand-crafted. Early layers learn low-level patterns like horizontal edges. Deeper layers combine those into eyes, wheels, or logos. Stacking convolution layers is the idea behind **CNNs (Convolutional Neural Networks)** — the backbone of nearly every CV system built before 2021.

Two other building blocks appear in every CNN:

- **Pooling** (usually max pooling): shrinks the spatial dimensions, making the network less sensitive to exact pixel location and reducing computation.
- **Activation functions** (usually ReLU): add non-linearity after each conv layer so the network can learn complex patterns, not just linear transforms.

```python {title="Minimal image tensor prep with PyTorch" run=false}
# pip install torch torchvision pillow
from PIL import Image
import torchvision.transforms as T

# Standard ImageNet preprocessing — works for most pretrained models
transform = T.Compose([
    T.Resize((224, 224)),       # resize to model's expected input
    T.ToTensor(),               # HxWxC uint8 [0,255] -> CxHxW float [0,1]
    T.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet channel means
        std=[0.229, 0.224, 0.225],    # ImageNet channel stds
    ),
])

img = Image.open("photo.jpg").convert("RGB")   # force RGB, not RGBA or BGR
tensor = transform(img)   # shape: (3, 224, 224)
batch = tensor.unsqueeze(0)  # add batch dim -> (1, 3, 224, 224)

print(batch.shape, batch.min().item(), batch.max().item())
# torch.Size([1, 3, 224, 224])  ~-2.1  ~2.6  (normalized, not [0,1])
```

## Data Augmentation: Free Performance Gains

A model that only ever sees perfectly framed, well-lit photos will fail on real user uploads. **Data augmentation** randomly transforms training images — flipping, rotating, adjusting brightness, cropping — so the model learns that the same object looks different from different angles and lighting conditions.

This is essentially free: you get a more robust model with no extra labelled data and very little extra cost. In production, augmentation happens only at training time. At inference, you apply the same normalization but skip the random transforms.

:::gotcha
Applying augmentation at inference time by accident is a surprisingly common bug in copy-pasted training code. Define two separate transform pipelines — one for training (with random flips, crops, colour jitter) and one for validation/production (resize + normalize only). Never share the same `transform` object between training and inference.
:::

:::interview-line
"CV fundamentals come down to tensors, channel order, normalization, and knowing whether you need a classification head or a detection head — get those four right and most other bugs find themselves."
:::

:::qa {q="Why do pretrained CV models require a specific normalization (mean/std)?"}
Pretrained models like ResNet were trained on ImageNet with specific pixel statistics. Feeding them un-normalized or differently normalized inputs shifts the distribution the model was optimized for, degrading performance even if the architecture is identical. Always use the normalization values that match the pretraining dataset, not arbitrary ones.
:::

:::qa {q="What is the difference between semantic segmentation and instance segmentation?"}
Semantic segmentation assigns a class label to every pixel but does not distinguish between individual objects — all cars are "car." Instance segmentation additionally separates each distinct object, so you get a separate mask for car #1, car #2, and so on. Instance segmentation is harder and computationally heavier, so choose semantic segmentation when you only need to know what is where, not which individual thing it is.
:::

:::drill {type="mcq" q="A PyTorch model expects input shape (B, C, H, W). You load an image with OpenCV and get an array of shape (480, 640, 3). What two things must you fix before passing it to the model?"}
- [ ] Resize to 224 × 224 and convert to float32
- [ ] Flip horizontal and add a batch dimension
- [x] Convert BGR → RGB and transpose to (C, H, W), then add a batch dimension
- [ ] Normalize to [0, 255] and add a channel dimension
:::

:::drill {type="mcq" q="You are building a system that counts the number of people in a crowd image and draws a bounding box around each individual. Which CV task is this?"}
- [ ] Semantic segmentation
- [ ] Image classification
- [ ] Keypoint estimation
- [x] Object detection
:::

:::key-takeaway
Before anything else in CV — before choosing an architecture, a loss, or a training recipe — make sure your tensors have the right shape, the right channel order, and the right normalization range. Every subsequent technique (CNN, ViT, detection head) builds on that foundation.
:::
