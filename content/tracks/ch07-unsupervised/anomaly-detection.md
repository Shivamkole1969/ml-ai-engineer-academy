---
id: anomaly-detection
track: ch07-unsupervised
title: "Anomaly detection in production"
badge: HOT
minutes: 9
prereqs: []
tags: [anomaly-detection, isolation-forest, autoencoder, monitoring, fraud-detection, unsupervised]
xp: 60
hot2026: true
---

Imagine your payment API has been running smoothly for six weeks. Then, at 2 AM on a Tuesday, a wave of transactions hits — each one just *slightly* off. The amounts look normal. The users exist. But the device fingerprints, the timing gaps, and the merchant categories form a pattern nobody hard-coded a rule for. Your rule-based fraud system misses every single one.

Anomaly detection is the branch of ML that catches *that* — deviations from normal, without you ever labeling what "abnormal" looks like in advance.

## What Is an Anomaly?

Three types appear constantly in production:

- **Point anomaly** — one data point is weird in isolation (a transaction for $50,000 when the user typically spends $50).
- **Contextual anomaly** — the value is reasonable in another context, but wrong here (a login at 3 AM from a new country, right after a password reset).
- **Collective anomaly** — no single point is odd, but the *sequence* is (small charges every 47 seconds for two hours straight).

Most real incidents are contextual or collective. That is exactly why simple threshold rules keep failing.

:::why-prod
In live systems, you rarely have labeled examples of every failure mode. Anomaly detection gives you unsupervised coverage — catching fraud, server faults, data quality issues, and model drift — before you have the labels to train a classifier.
:::

## The Core Algorithms

You don't need all of them. Pick based on your data shape and latency budget.

:::table {title="Anomaly detection algorithms at a glance"}
| Algorithm | Best for | Scales to | Watch out for |
|---|---|---|---|
| Isolation Forest | Tabular, high-dimensional | Millions of rows | Struggles with dense clusters |
| Local Outlier Factor (LOF) | Low-to-medium dimensional | ~100k rows | Slow at inference |
| One-Class SVM | Well-behaved numeric features | ~10k rows | Very slow to train |
| Z-score / IQR | Single numeric signals | Any size | Can't model interactions |
| Autoencoder | Images, logs, time-series windows | Any, GPU helps | Needs tuning; harder to explain |
:::

For most tabular production workloads, **Isolation Forest** is the right starting point — fast, reasonably accurate, and it produces a continuous score you can threshold.

## Isolation Forest in 20 Lines

The idea is elegant: anomalies are *easier to isolate* than normal points. Build random trees that partition the feature space. Weird points get isolated in fewer splits than normal ones do.

```python {title="Isolation Forest on simulated transaction data" run=false}
import numpy as np
from sklearn.ensemble import IsolationForest

# Run locally: pip install scikit-learn numpy
rng = np.random.default_rng(42)

# 1 000 normal transactions (amount, hour-of-day)
normal = rng.normal(loc=[100, 14], scale=[20, 3], size=(1000, 2))
# 10 fraudulent transactions — high amounts, odd hours
fraud  = rng.normal(loc=[800, 3],  scale=[50, 0.5], size=(10, 2))

X = np.vstack([normal, fraud])

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,   # expected fraction of anomalies — tune this carefully
    random_state=42,
)
model.fit(X)

scores = model.decision_function(X)   # lower score = more anomalous
labels = model.predict(X)             # -1 = anomaly, 1 = normal

flagged = np.where(labels == -1)[0]
print(f"Flagged {len(flagged)} anomalies out of {len(X)} points")
# Expected output: ~10 flagged — the fraud rows
```

The `contamination` parameter is the one you'll wrestle with in production. Set it too high and you flood your ops team with false positives. Too low and real events slip through unnoticed.

:::gotcha
**The contamination trap.** Setting `contamination=0.01` tells the model "I expect 1% of data to be anomalies." If your actual positive rate is 5%, the model under-flags everything. Worse, if you retrain on data where flagged records were already removed, anomalies quietly leak back into your training set over time. Always measure your *actual* positive rate before setting this number — and revisit it every quarter.
:::

## Autoencoders When Your Data Isn't a Table

When your data is a log line, a time-series window, or an image, autoencoders shine. You train the network to compress and then reconstruct *normal* data only. At inference time, reconstruction error becomes your anomaly score: if the model can't rebuild an input accurately, something unusual is going on.

The catch? You have to decide what counts as "high" reconstruction error. That threshold lives in your business logic, not inside the model. Own it.

:::war-story {title="The silent mobile revenue dip"}
An e-commerce team trained an Isolation Forest on clickstream features. It worked great in testing. Six months later, a new mobile app version changed how session lengths were recorded — artificially doubling the value for all mobile users. The model quietly started flagging every mobile checkout as anomalous and suppressing those sessions from a downstream recommender. Nobody noticed for three weeks because the suppression was soft (a score, not a hard block). Mobile revenue dipped 8% before an on-call engineer pulled the anomaly score distributions and spotted the shift. Lesson: anomaly detector *outputs* need their own monitoring dashboards, not just the model inputs.
:::

## Serving Anomaly Detection in Production

A few things that reliably bite teams at scale:

1. **Threshold drift** — the score boundary that worked at launch stops working after six months as normal behavior evolves. Log scores continuously and alert when the distribution shifts.
2. **Feedback loops** — if flagged records are removed from retraining data, the model never learns the new normal.
3. **Latency** — LOF and One-Class SVM are too slow for real-time APIs. Use Isolation Forest or a lightweight autoencoder with offline batch pre-scoring.
4. **Explainability** — ops teams need to know *why* something was flagged. Log which features most influenced the score, even for unsupervised models. SHAP works on Isolation Forest.

:::interview-line
"Anomaly detection learns the pattern of normal, not the pattern of fraud — so it catches things you've never seen before, which is exactly why rule engines can't."
:::

:::qa {q="How do you pick a threshold for your anomaly score in production?"}
Start from a business constraint — how many alerts can your ops team actually review per day? Work backwards from that capacity to a false-positive budget. Set the threshold to match that budget on a holdout validation period. Revisit it monthly as the data distribution shifts and ops capacity changes.
:::

:::qa {q="Why can't you just train a supervised classifier for every fraud or fault detection problem?"}
Supervised classifiers need labeled examples of every attack or failure pattern you want to catch. Novel failure modes, by definition, have no labels yet. An anomaly detector trained only on normal data will still flag them because they deviate from the learned normal — no labels required.
:::

:::qa {q="What is reconstruction error and how does an autoencoder use it as an anomaly score?"}
An autoencoder learns to compress and reconstruct its training data, which represents the normal class. When it sees an unusual sample at inference time, it cannot reconstruct it accurately. The difference between the original input and the model's output — the reconstruction error — is large. You use that magnitude directly as an anomaly score and set a threshold above which you flag the sample.
:::

:::drill {type="mcq" q="Your Isolation Forest flags 15% of production traffic as anomalous, but your ops team can only review 1% of traffic per day. What is the FIRST thing you should change?"}
- [ ] Switch to a more powerful algorithm like One-Class SVM
- [ ] Retrain the model with twice as much data
- [x] Recalibrate the contamination parameter and score threshold to match ops review capacity
- [ ] Remove the anomaly detector and replace it with hand-written rules
:::

:::drill {type="mcq" q="A fraud pattern emerges: many small transactions spread across different merchants within 10 minutes — each one looks normal on its own. What type of anomaly is this?"}
- [ ] Point anomaly
- [ ] Contextual anomaly
- [x] Collective anomaly
- [ ] Structural anomaly
:::

:::drill {type="mcq" q="You retrain your anomaly detector monthly by removing all flagged records from the training set first. What long-term risk does this create?"}
- [ ] The model will overfit to the normal class
- [ ] Training will become slower over time
- [x] Anomalous patterns leak back into the training distribution, making the model blind to them
- [ ] The contamination parameter becomes invalid after the first retrain
:::

:::key-takeaway
Anomaly detection catches what you haven't labeled yet — but it only stays useful if you monitor the score distributions continuously, recalibrate thresholds as behavior evolves, and give your ops team enough context to act on every flag.
:::
