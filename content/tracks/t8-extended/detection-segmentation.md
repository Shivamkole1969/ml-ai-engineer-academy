---
id: detection-segmentation
track: t8-extended
title: "Object detection & segmentation"
badge: CORE
minutes: 9
prereqs: []
tags: [computer-vision, object-detection, segmentation, yolo, mAP, instance-segmentation]
xp: 45
hot2026: false
---

Imagine your product team wants to ship a smart-shelf system for a supermarket chain. Ceiling cameras scan every aisle 24/7. The business goal: detect when a product slot goes empty, count remaining items, and alert staff in near real time. You think: "CNN image classifier — done." Then you realize a classifier only tells you *what* is in the frame, not *where* or *how many*. You need the model to draw boxes. And if the business later asks for pixel-perfect outlines — say, for a surgical-robotics spinoff — you need masks too. That progression is the story of object detection and segmentation.

## Classification, detection, segmentation — the spectrum

These three tasks share the same backbone technology but ask increasingly harder spatial questions.

:::table {title="Vision task comparison"}
| Task | Output | Real-world use |
|---|---|---|
| Classification | One label per image | "Is there a defect?" |
| Object detection | Bounding boxes + labels + confidence | Counting items, ADAS, face detection |
| Semantic segmentation | Per-pixel class label (no object distinction) | Road vs sidewalk vs sky |
| Instance segmentation | Per-pixel mask, one per object instance | Surgery robots, retail shelf fills |
| Panoptic segmentation | Semantic + instance merged | Self-driving full-scene understanding |
:::

:::why-prod
Production systems live in the detection/segmentation tier because bounding boxes and masks are what downstream code actually uses: crop an ROI, measure dimensions, reject defective parts, trigger alerts. A label alone rarely drives an action.
:::

## The two big detection families

**Two-stage detectors** (Faster R-CNN, R-FCN): pass the image through a Region Proposal Network (RPN) that shortlists candidate regions, then classify each region. More accurate, but slower — often 5–15 FPS on a GPU.

**One-stage detectors** (YOLO family, SSD, RetinaNet): skip the proposal step entirely. The network predicts boxes and class scores in one forward pass. Easily 30–200+ FPS. This is the architecture family you'll reach for in any real-time scenario.

YOLO (You Only Look Once) has become the industry default for real-time detection. YOLOv8 and YOLOv9 from Ultralytics are the current goto baselines — well-documented, pre-trained on COCO, and easy to fine-tune on custom data in a few hours.

## How detection heads work (the fast version)

The backbone (a CNN or a Vision Transformer) extracts feature maps at multiple scales. The detection head then predicts, for each cell of a grid:

1. **Bounding box coordinates** — typically expressed as offsets from anchor boxes (predefined box shapes at each grid cell).
2. **Objectness score** — how confident the model is that any object lives here.
3. **Class probabilities** — which class (among the N classes) the object belongs to.

At inference time, you run Non-Maximum Suppression (NMS) to remove duplicate boxes that overlap the same object. The box with the highest confidence wins; all others above an IoU threshold get discarded.

## Segmentation architectures in brief

For **semantic segmentation**, the backbone extracts features and a decoder upsamples them back to full image resolution, labeling every pixel. U-Net (encoder + skip connections + decoder) is the workhorse for medical imaging. DeepLab v3+ with atrous convolutions handles multi-scale objects well.

For **instance segmentation**, Mask R-CNN adds a small fully-convolutional mask branch on top of Faster R-CNN. For each detected box, it predicts a binary mask. YOLOv8-seg brings this capability into the one-stage world — close to real-time instance masks.

## Measuring model quality: mAP and mIoU

**IoU (Intersection over Union)** measures overlap between predicted box and ground-truth box. IoU = 1 means perfect overlap; IoU = 0 means no overlap.

A prediction is a **true positive** if IoU ≥ a threshold and the class matches. The standard threshold is 0.5 (written AP@50). The COCO benchmark uses AP averaged across IoU thresholds from 0.5 to 0.95 in steps of 0.05 — written mAP@[0.5:0.95].

**mAP** (mean Average Precision) averages AP across all object classes. It rewards both precision (few false detections) and recall (few missed objects).

For segmentation quality, **mIoU** (mean IoU over all classes) is the standard metric for semantic segmentation. Mask AP is used for instance segmentation — same idea as bounding-box AP but evaluated on pixel masks.

## A minimal YOLOv8 detection run

```python {title="YOLOv8 inference — install ultralytics, free to run" run=false}
# pip install ultralytics
from ultralytics import YOLO

# Load pretrained YOLOv8 nano — the fastest/smallest variant (~6 MB)
model = YOLO("yolov8n.pt")  # auto-downloads on first run

# Run detection on a local image; conf = min confidence threshold
results = model("your_image.jpg", conf=0.4)

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]          # e.g. "person", "car"
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"{label} ({conf:.2f}) → [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

# Switch to instance segmentation: just change the model file
# model = YOLO("yolov8n-seg.pt")
# Then r.masks gives per-object pixel masks
```

## Production realities: latency vs accuracy

In production, mAP is only half the story. A 55-mAP model running at 2 FPS is useless for a checkout camera but fine for nightly batch analysis of satellite imagery. Match your model to the latency budget first.

Common levers to hit latency targets:
- **Quantization** — INT8 via TensorRT or ONNX Runtime, often 2–4x speedup with small accuracy loss.
- **Smaller variants** — YOLOv8 comes in nano/small/medium/large/extra-large. Start nano, grow as needed.
- **Lower input resolution** — 640 px is YOLO's default; 416 px cuts compute ~35% with modest accuracy loss.
- **Export to ONNX/TensorRT** — move off PyTorch for inference; this alone halves latency in many setups.

Data quality beats model complexity almost every time. 2,000 cleanly annotated images of your specific domain (labeled in COCO JSON or YOLO format) will outperform a generic pretrained model. Budget for annotation tooling (Label Studio, Roboflow, CVAT — all have free tiers).

:::gotcha
Never trust mAP from COCO benchmarks as a proxy for your domain. COCO images are natural photos. If your camera is a top-down warehouse ceiling cam at night, the visual distribution is completely different. Always evaluate on a held-out set from your own data before declaring a model "production-ready."
:::

:::interview-line
"Detection is a regression-plus-classification problem: the model simultaneously predicts where an object is and what it is, with mAP@[0.5:0.95] as the production-grade quality bar."
:::

:::qa {q="What is the difference between semantic and instance segmentation?"}
Semantic segmentation assigns a class label to every pixel but doesn't distinguish between separate objects of the same class — all cars are "car." Instance segmentation assigns a separate mask to every individual object instance, so car #1 and car #2 each get their own mask. Instance segmentation is strictly harder and slower, but necessary whenever you need to count or individually track objects.
:::

:::qa {q="Why does NMS (Non-Maximum Suppression) matter and what happens if you skip it?"}
Without NMS, the same physical object gets dozens of overlapping boxes — the detection head fires at every grid cell near the object. NMS keeps only the highest-confidence box in each overlapping cluster by suppressing any box that overlaps it beyond an IoU threshold. Skip it and your downstream system sees 20 "detections" of the single same car.
:::

:::qa {q="When would you choose Faster R-CNN over YOLOv8?"}
When accuracy matters more than latency — medical imaging, satellite analysis, or any offline batch pipeline where you have seconds per image. Faster R-CNN's two-stage design (region proposals, then classification) is slower but often more precise, especially on small or densely packed objects. YOLOv8 wins for real-time inference.
:::

:::drill {type="mcq" q="A model achieves mAP@50 of 0.72 on COCO. Your team deploys it to inspect printed circuit boards under UV light and it detects almost nothing. The most likely reason is:"}
- [ ] The model is quantized to INT8, which strips spatial precision
- [ ] mAP@50 is too loose a threshold; you need mAP@[0.5:0.95]
- [x] The visual distribution of UV-lit PCBs is completely unlike COCO natural photos, so pretrained features don't transfer
- [ ] Faster R-CNN should have been used instead of YOLO
:::

:::drill {type="mcq" q="You need pixel-level masks for each individual car in a parking-lot image — not just which pixels are 'car' in general. Which approach is correct?"}
- [ ] Semantic segmentation with a U-Net backbone
- [ ] Image classification with a softmax head
- [x] Instance segmentation (e.g., Mask R-CNN or YOLOv8-seg)
- [ ] Panoptic segmentation with only the 'stuff' classes enabled
:::

:::key-takeaway
Object detection adds bounding boxes to classification; segmentation adds pixel masks. YOLO-family one-stage detectors dominate real-time production, Mask R-CNN for accurate instance masks. Always evaluate mAP on your own domain data — COCO scores do not transfer to custom environments.
:::
