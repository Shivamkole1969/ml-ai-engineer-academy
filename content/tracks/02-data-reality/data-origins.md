---
id: data-origins
track: 02-data-reality
title: "Where data comes from & its built-in distortion"
badge: CORE
minutes: 8
prereqs: []
tags: [data-quality, bias, data-collection, ml-fundamentals, production]
xp: 45
hot2026: false
---

You join a fintech startup. Someone hands you a CSV: "Here's our last 18 months of transaction data — go build the fraud model." You spend three weeks doing everything right. Clean splits. Careful features. Solid ROC-AUC on validation. You ship it.

Two weeks in, the model is missing fraud that any human analyst would catch. The post-mortem reveals the problem: every row in that CSV was a transaction the *old rule engine had already approved*. Anything suspicious was blocked before it could be logged. You trained on a universe of "things we didn't think were fraud," and you're surprised the model can't spot fraud.

The dataset wasn't corrupted. It was perfectly faithful to reality — just a much narrower slice of reality than you assumed.

## Data is a record of what was measured, not what happened

Here is the most important sentence in this track: **A row only exists in your dataset if a chain of decisions allowed it to get there.**

Someone chose what to instrument. A system decided what to log. A pipeline decided what to keep. A labeler decided what to annotate. Each decision is a filter. Filters introduce distortions. Those distortions flow straight into your model, silently, and show up only after you've deployed.

This lesson is about recognising *where data comes from* so you know *what distortions come bundled with it*.

:::why-prod
Every model inherits the collection process's blind spots. Knowing the origin of your data tells you which failure modes to look for before you ever train a single epoch. This is why senior ML engineers ask "how was this collected?" before they look at a single feature.
:::

## The five main origins — and the distortion each one ships with

Data in ML systems almost always comes from one of five places. Each has a characteristic way of lying to you.

:::table {title="Data origins and their built-in distortions"}
| Origin | Examples | Built-in distortion |
|---|---|---|
| **System logs / events** | Clickstream, API calls, app events | Only records events the system was *built to record*. Silent failures, dropped packets, and dark-pattern interactions leave no trace. |
| **Transactional records** | Orders, payments, support tickets | Captures what *completed*. Abandoned carts, rejected applications, and blocked transactions are systematically absent or underrepresented. |
| **Sensor / IoT streams** | Temperature sensors, GPS pings, wearables | Hardware has ranges, dead zones, and maintenance windows. Missing data is never random — sensors fail in predictable conditions. |
| **Scraped / third-party** | Web crawls, public APIs, purchased datasets | Reflects whoever made it onto the internet, into the source system, or into the vendor's collection period. Urban, English-speaking, and tech-engaged populations are massively overrepresented. |
| **Human-annotated labels** | Image labels, NLP sentiment, moderation flags | Reflects the labeler pool, the guidelines at *annotation time*, and whoever had the patience (or was paid) to label. Next track covers this in depth. |
:::

## The one question that unlocks the diagnosis

You don't need to memorise a taxonomy of biases. You need one habit: when you first see a dataset, ask yourself out loud —

> **"What had to happen for this row to exist?"**

Work backwards. A row in a customer churn dataset: the customer had to sign up, stay long enough to generate enough events, and *not* delete their account before the pipeline ran. Who's missing? People who churned in the first week. People who never fully onboarded. Your "churned" class is therefore only the *slow* churners.

A row in a medical imaging dataset: a patient had to get a scan, the hospital had to digitise it, the radiology team had to annotate it, and the institution had to share it. Who's missing? Patients from lower-income clinics. Conditions that don't warrant imaging. Edge cases that confuse annotators.

Ask the question. Trace the chain. The distortions reveal themselves.

## Instrumentation is not neutral — it changes over time

One more thing that catches engineers off guard: the collection process itself evolves.

Your company rolls out a new mobile app in Q3. Suddenly the event schema changes, new event types appear, and old ones go silent. Your model's training window now contains two distinct data-generating processes — stitched together with no label to tell them apart.

This is why your model can be perfectly trained and still degrade after a seemingly unrelated product release. The features haven't drifted in the wild; the *definition* of the feature changed in the logs. Drift has a whole lesson of its own in this track, but the seed of it is here: data is a living artifact, not a static table.

:::gotcha
Joining two tables and assuming the join makes the data "complete" is a classic mistake. A LEFT JOIN that fills missing values with zero is silently telling your model "this thing didn't happen" when the truth is "this thing wasn't recorded." Always trace *why* a value is missing before imputing it.
:::

:::interview-line
"I always start by asking what had to happen for a row to exist — every data source has structural gaps, and those gaps become the model's blind spots."
:::

:::qa {q="You're handed a dataset from a production system. What's the first thing you check?"}
I ask how it was collected: what system generated it, what events it captures, and whether there are implicit filters — like only logging successful transactions. Understanding the collection process tells me which populations and outcomes are structurally underrepresented before I ever look at a feature distribution.
:::

:::qa {q="A model works great in offline eval but underperforms on a specific user segment after launch. Where do you look first?"}
I check whether that segment was well-represented in the training data at all — not just by count, but by whether the collection process was equally likely to capture events from them. Underperforming segments are often segments the data pipeline captured poorly: different devices, different regions, or users whose events hit a different code path.
:::

:::drill {type="mcq" q="You're training a loan default model. The dataset contains only loans that were *approved* by the previous model. What is the most direct consequence for training?"}
- [ ] The features will have higher variance than expected
- [ ] The model will be slower to train because approved loans are complex
- [x] The model never sees the distribution of applicants who were rejected, so it learns to score within an already-filtered population
- [ ] This is fine — rejected loans wouldn't have default labels anyway
:::

:::drill {type="mcq" q="An IoT temperature sensor stops reporting values when ambient temperature drops below 5°C (it freezes). Your dataset shows no readings below 5°C. What kind of missingness is this?"}
- [ ] Missing completely at random (MCAR)
- [ ] Missing at random (MAR), conditioned on other variables
- [x] Missing not at random (MNAR) — the missingness is caused by the value itself
- [ ] Structural missingness introduced by the labeling process
:::

:::key-takeaway
Every dataset is shaped by the process that created it. Before modelling anything, trace the chain: what had to happen for each row to exist? The gaps in that chain are the model's future failure modes.
:::
