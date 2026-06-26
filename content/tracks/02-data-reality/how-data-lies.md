---
id: how-data-lies
track: 02-data-reality
title: "How data lies: selection, survivorship, feedback, Simpson's, MNAR"
badge: HOT
minutes: 10
prereqs: []
tags: [data-quality, bias, simpsons-paradox, mnar, feedback-loops, survivorship-bias]
xp: 60
hot2026: true
---

It's 11 PM. Your loan-approval model has been live for three months and the business team is delighted — validation AUC was 0.91. Then the credit-risk analyst pulls a cohort report. Defaults are three times higher than the model predicted. Your first instinct: model bug. But the model is fine. The data was lying the whole time.

This lesson names the five liars inside every dataset, shows you how to catch them, and gives you the vocabulary to walk into any interview or design review and sound like someone who has been burned before.

## The five ways data lies to you

Every dataset is a *sample* of reality, not reality itself. The gap between the sample and the real world is where bugs hide. Here are the five classic gaps.

**1. Selection bias** — your training data was collected by a process that excluded part of the population your model will score. Loan data only contains people who *applied* for a loan and got *approved*. Everyone else is invisible to you.

**2. Survivorship bias** — a special case of selection bias: you only see the things that survived a filter. Historical startup data on Crunchbase is full of successful exits; the thousands of companies that died quietly were never updated.

**3. Feedback loops** — your model's output changes the world, which changes the next batch of training data, which changes the model. The model becomes its own weather. Credit scoring is the textbook case: if your model denies credit to a segment, that segment never gets to repay (or default), so future retraining sees even less data about them.

**4. Simpson's paradox** — an aggregate trend that *reverses* inside every sub-group. The company-level conversion rate went up, but it went down in every individual city. The aggregate masks the real story because group sizes shifted.

**5. MNAR (Missing Not At Random)** — data is not missing by chance; the missingness is correlated with the value itself. Blood-pressure readings are more likely to be skipped for patients who are too ill to sit up. The absence of the value *is* a signal — and naively imputing the mean is wrong.

:::why-prod
Models trained on biased data pass offline evals and fail in production. The model isn't broken — the *evaluation* was done on the same biased distribution as training. You will not catch these bugs with a confusion matrix.
:::

:::table {title="The five liars at a glance"}
| Liar | Root cause | Danger signal | Quick mitigation |
|---|---|---|---|
| Selection bias | Who was eligible to generate data | Offline AUC great, live precision collapses | Importance weighting; calibration on unbiased sample |
| Survivorship bias | Losers dropped from history | Training on "good examples" only | Deliberately include failure cases |
| Feedback loop | Model output → future training labels | Drift accelerates post-deployment | Hold out a random-serve arm for uncontaminated data |
| Simpson's paradox | Confound shifts group sizes | Aggregate metric improves, segment metrics fall | Always slice metrics by key sub-groups before trusting aggregates |
| MNAR | Missingness ↔ value | High `null` rate on a sensitive feature | Model missingness indicator as a feature; don't impute blindly |
:::

## Simpson's paradox in six lines

This one trips people in interviews because it sounds impossible. Here it is in code you can run right now.

```python {title="Simpson's paradox — aggregate vs per-group trend" run=false}
# pip install pandas  (standard in any ML environment)
import pandas as pd

# Treatment A tested mostly on mild cases; Treatment B on severe cases.
# Within each severity group, A beats B. But aggregate flips.
data = pd.DataFrame({
    "treatment": ["A","A","B","B"],
    "severity":  ["mild","severe","mild","severe"],
    "n_patients": [700, 300, 100, 900],
    "n_recovered": [560, 120, 90, 630],
})
data["rate"] = data["n_recovered"] / data["n_patients"]

agg = data.groupby("treatment")[["n_recovered","n_patients"]].sum()
agg["rate"] = agg["n_recovered"] / agg["n_patients"]

print(data[["treatment","severity","rate"]])
# A outperforms B in BOTH severity groups
print("\nAggregate:")
print(agg["rate"])
# Yet B looks better in aggregate because it treated more severe patients
# who have higher base recovery numbers
```

Run it. The per-group rates and the aggregate rate disagree. This is exactly what happens when you report a model's accuracy without slicing by user cohort, geography, or product line.

:::gotcha
MNAR is the hardest to detect because the data *looks* complete after you drop nulls. Always ask: "Why is this field null for *these* rows specifically?" If the answer is correlated with the target, you have MNAR. Dropping rows or mean-imputing destroys the signal hiding in the missingness pattern.
:::

:::war-story {title="The disappearing segment"}
A fintech in Pune deployed a credit model that looked great on validation. Six months in, the team noticed one geographic cluster had a suspiciously low approval rate. Digging in: the sales team had manually re-routed applications from that cluster to a different product line during the data-collection period. Those customers never entered the training set. The model had never seen them — and when it finally did, it had no idea what to do. Classic selection bias, discovered only after real money was lost.
:::

:::interview-line
"Before I trust any metric, I ask: what data *couldn't* enter this training set, and is the model's deployment population the same as the population that generated the labels?"
:::

:::qa {q="What is MNAR and why is it worse than MAR?"}
MAR (Missing At Random) means the probability of a value being missing depends only on *other observed* variables — you can model it away. MNAR means the probability of missingness depends on the missing value itself, so no amount of imputation from observed data can fix the bias. The only real fix is to either collect the missing data or explicitly model the indicator of missingness as a feature.
:::

:::qa {q="How do feedback loops corrupt model retraining?"}
When your model's decisions determine which outcomes get recorded, the next training batch is no longer a random sample of reality — it is a sample of reality filtered through your previous model's beliefs. The model reinforces its own blind spots with each retraining cycle. The defence is a random-holdout or explore arm: a small fraction of traffic where you override the model and record unbiased outcomes.
:::

:::drill {type="mcq" q="A model predicts employee attrition using HR records. Only employees who stayed more than six months are in the dataset. Which bias is most directly at play?"}
- [ ] MNAR — the attrition values are missing non-randomly
- [x] Survivorship bias — employees who left quickly were excluded, so the model never learns from early leavers
- [ ] Simpson's paradox — sub-group trends reverse the aggregate
- [ ] Feedback loop — model output changes who gets hired
:::

:::drill {type="mcq" q="You slice an overall-improving conversion metric by region and find it is *down* in every single region. What is the most likely explanation?"}
- [ ] Label noise — annotators disagreed on conversion events
- [ ] MNAR — conversion data is missing for some regions
- [ ] Survivorship bias — low-converting regions were removed from the dataset
- [x] Simpson's paradox — the mix of regions in the aggregate shifted, masking per-region decline
:::

:::key-takeaway
Data lies through what it excludes, what it includes selectively, what it makes missing on purpose, and what it aggregates too eagerly. Name the liar before you trust the metric.
:::
