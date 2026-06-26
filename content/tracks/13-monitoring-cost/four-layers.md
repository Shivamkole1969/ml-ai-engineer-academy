---
id: four-layers
track: 13-monitoring-cost
title: "The four monitoring layers & ground-truth lag"
badge: CORE
minutes: 9
prereqs: []
tags: [monitoring, drift, production-ml, mlops, ground-truth, observability]
xp: 45
hot2026: false
---

Your credit-risk model shipped three weeks ago. Grafana is green — latency fine, error rate zero, GPU humming along. Then the product manager pings you: "Loan approvals are down 40% and nobody flagged it." The model wasn't crashing. It was silently returning bad predictions. No alert ever fired because you were only watching the infrastructure. That's a Layer 1 blindspot, and it happens on production ML systems constantly.

## The four-layer stack

Think of ML monitoring as four floors of a building. Instrument only the ground floor and you'll miss everything happening upstairs.

:::why-prod
Most ML failures in production are silent — no exception, no 5xx, no pager page. The model just returns wrong predictions confidently. Without multi-layer monitoring, you won't find out until business metrics crater, which is always the worst time.
:::

:::table {title="The four monitoring layers"}
| Layer | What you watch | Example signals | Time to alert |
|---|---|---|---|
| 1 — Infrastructure | System health | CPU/GPU utilisation, memory, p99 latency, error rate | Seconds |
| 2 — Input (data) | What enters the model | Feature distribution shift, missing values, schema violations | Minutes–hours |
| 3 — Output (model) | What the model returns | Prediction score distribution, confidence percentiles, class-label frequency | Minutes–hours |
| 4 — Outcome (business) | Was the model actually right? | Accuracy, F1, precision@K, downstream revenue/KPI impact | Hours–months |
:::

**Layer 1 — Infrastructure** is the floor everyone already monitors. If your inference server runs out of GPU memory, you know in seconds. Necessary, but nowhere near sufficient.

**Layer 2 — Input monitoring** watches your features before the model sees them. If today's borrowers are 30% older than your training distribution, the model will silently extrapolate into territory it has never seen. No error is thrown — the predictions just drift off a cliff.

**Layer 3 — Output monitoring** watches what comes out of the model. If your fraud detector suddenly flags 0.1% of transactions instead of the usual 2.3%, something has changed upstream — even if you can't prove it yet. Prediction drift is a *leading indicator*: it fires before you accumulate enough ground truth to confirm the problem.

**Layer 4 — Outcome monitoring** is the only layer that tells you the model is *actually wrong*. It requires labelled outcomes. Which brings us to the hard part.

## Ground-truth lag: the silent gap

Ground truth rarely arrives instantly. How long you wait depends entirely on your domain:

- **Recommendation click** — 10 minutes
- **Ad conversion** — 24–72 hours
- **Fraud chargeback** — 30–90 days
- **Loan default** — 6–24 months

This delay is called **ground-truth lag** (also "label delay"). It creates a monitoring gap between Layers 3 and 4. Your model could be systematically wrong for weeks before Layer 4 can confirm anything.

The practical implication: you cannot rely solely on Layer 4. Layers 2 and 3 are your early-warning system *while* ground-truth labels accumulate slowly in the background. A well-run team runs all four layers in parallel, calibrates each layer's alert threshold to its expected lag, and never mistakes "Layer 1 is green" for "the model is correct."

:::gotcha
Treating a non-crashing model as a healthy model. If your model returns predictions without throwing errors, it will pass every infrastructure check perfectly. You need prediction distribution monitoring (Layer 3) to catch the moment outputs go wrong — well before ground truth arrives to confirm it. Teams that skip Layer 3 are flying blind for exactly that lag window.
:::

:::interview-line
"We monitor four layers — infra, input, output, and outcome — because the layer that tells you the model is wrong is also the one with the most lag. The other three act as early-warning proxies while ground truth catches up."
:::

:::qa {q="Why can't you just monitor model accuracy in production and call it done?"}
Accuracy needs ground-truth labels, which often arrive days to months after the prediction. By the time your accuracy metric dips, the model may have made thousands of wrong decisions. Layers 2 and 3 — input and prediction distributions — fire much earlier and act as proxies that give you time to investigate before confirmed labels appear.
:::

:::qa {q="What is ground-truth lag and how do production teams handle it?"}
Ground-truth lag is the delay between a model making a prediction and the system learning the correct outcome. Teams handle it by running continuous join pipelines that match predictions to labels as they trickle in, setting up proxy alerting on Layer 2 and 3 signals for the interim window, and explicitly designing SLAs around the expected lag so the on-call team knows which layer to trust at each time horizon.
:::

:::drill {type="mcq" q="A fraud model's Layer 1 dashboard shows perfect health: p99 latency 44 ms, error rate 0.0%, GPU utilisation 61%. Yet the fraud operations team reports a sharp rise in missed-fraud complaints over the past week. Which layer most likely failed to alert?"}
- [ ] Layer 1 — Infrastructure
- [ ] Layer 2 — Input (data) monitoring
- [x] Layer 3 — Output monitoring (prediction distribution drift went unnoticed)
- [ ] This scenario is impossible if Layer 1 is healthy
:::

:::drill {type="mcq" q="Which statement about ground-truth lag is correct?"}
- [ ] Ground-truth lag only affects regression tasks, not classification
- [ ] If the Layer 3 prediction distribution is stable, ground-truth lag stops mattering
- [x] Ground-truth lag means Layer 4 accuracy can lag the actual onset of model degradation by days or months, making earlier-layer proxies essential
- [ ] Ground-truth lag is fully solved by running A/B tests alongside the model
:::

:::key-takeaway
Instrument all four layers — infrastructure, input, output, and outcome — because each catches a different class of failure. Ground-truth lag guarantees Layer 4 always arrives late, so treat Layers 2 and 3 as your real-time early-warning system.
:::
