---
id: eda-workflow
track: ch05-eda-features
title: "An EDA workflow that finds the story"
badge: CORE
minutes: 9
prereqs: []
tags: [eda, pandas, data-quality, feature-engineering, workflow]
xp: 45
hot2026: false
---

Imagine you just got handed a dataset for a new fraud-detection model. It has 47 columns, a README that says "self-explanatory", and a deadline in two weeks. Your instinct is to immediately run `model.fit()`. Don't. Almost every painful model debugging session — the ones that cost you a weekend — traces back to something you would have caught in the first 30 minutes of proper Exploratory Data Analysis.

EDA is not a box to tick. It is the part of the job where you find the story hiding in the numbers before the model silently learns the wrong story for you.

## What EDA actually is

EDA (Exploratory Data Analysis) is the process of systematically getting to know your dataset before you model it. You are answering three questions:

1. **What is in here?** — shape, types, missing values, duplicates
2. **Does it make sense?** — distributions, outliers, impossible values
3. **What relationships exist?** — correlations, target associations, suspicious patterns

Think of it as interviewing the data. You are not trusting it blindly. You are pressure-testing every column.

:::why-prod
In production, bad data costs you silently. A model trained on a column with 40% nulls that were backfilled with zeros will look fine in offline metrics — until real traffic arrives and the null pattern is different. EDA is your first and cheapest line of defense against this category of failure.
:::

## A repeatable EDA workflow

Good EDA is not random clicking around. It follows a rhythm. Here is one that works reliably:

:::table {title="EDA workflow — five phases in order"}
| Phase | What you do | Why it matters |
|---|---|---|
| 1. Shape & types | `.shape`, `.dtypes`, `.info()` | Spot wrong types (numbers stored as strings, dates as objects) immediately |
| 2. Missing values | Count & pattern-mine nulls per column | Nulls that cluster by time or user signal data pipeline issues, not randomness |
| 3. Distributions | Histograms, `.describe()`, value counts | Skew, impossible values, and suspiciously round numbers all show up here |
| 4. Target analysis | Class balance, target vs. each feature | A heavily imbalanced target will fool your default metrics |
| 5. Correlations & leakage check | Heatmap + pairplot on top features | Columns that correlate too perfectly with the target are often leaking future info |
:::

## Phase 1 in code

Here is a minimal but real starter that works on any tabular dataset. Run it the moment you receive a new file.

```python {title="EDA starter — run this first on any new dataset" run=false}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data (adjust path / format as needed)
df = pd.read_csv("your_dataset.csv")

# --- Phase 1: shape & types ---
print(f"Shape: {df.shape}")
print(df.dtypes.value_counts())  # how many of each type

# --- Phase 2: missing values ---
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])  # only columns that have nulls

# --- Phase 3: distributions (numeric columns) ---
df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T

# --- Phase 4: target analysis (replace 'label' with your target column) ---
print(df["label"].value_counts(normalize=True))

# --- Phase 5: quick correlation heatmap ---
numeric_cols = df.select_dtypes("number").columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, fmt=".1f", annot=len(numeric_cols) < 15)
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
# Free to run locally: pip install pandas matplotlib seaborn
```

## How to read what you find

Numbers alone don't teach you anything. The skill is in *interpreting* them.

**A column with 60% nulls** — ask whether it was always this way or changed over time. If it changed, your training and serving distributions are different. That is a feature you should probably drop or engineer carefully.

**A perfectly uniform distribution** — suspiciously flat histograms often mean bucketed or capped values upstream. Know where those caps are before you model.

**A feature that correlates 0.98 with the target** — almost always leakage (see the dedicated lesson). Stop and investigate before going further.

**Skewed numerics** — a column where 95% of rows are 0 and a handful are in the millions. Linear models will be dominated by those outliers. Tree models will be fine. Know your model family before you decide whether to transform.

:::gotcha
Do not skip EDA when a dataset "looks clean" or came from an internal data warehouse. Internal datasets often have silent conventions — nulls that mean "zero", timestamps in local timezone without a flag, categorical columns that gained new values after your cutoff. Tidy-looking data is where the most subtle bugs hide.
:::

## The story, not just the stats

Here is what separates engineers who do EDA from engineers who do good EDA: they write down what they learn as they go. A short markdown cell or comment per phase — "30% of `txn_amount` is null before 2023, then drops to 2%" — is worth more than ten plots with no notes.

By the end of a good EDA pass you should be able to say one sentence to your team: "The dataset has X rows spanning Y time period, the target is Z% positive, the two most predictive raw features appear to be A and B, and there is a null pattern in column C we need to handle."

That sentence is the story. If you cannot say it, you are not done yet.

:::interview-line
"Before touching a model I run a five-phase EDA: shape/types, null patterns, distributions, target balance, and a leakage check — because every debugging session I have skipped that has cost me later."
:::

:::qa {q="What is the difference between checking null counts vs. null patterns?"}
Null counts tell you how much data is missing. Null patterns tell you why. A column that is null only before a certain date, or only for a specific user segment, indicates a pipeline change or a population difference — and that means your training distribution will differ from your serving distribution, which is a serious production risk.
:::

:::qa {q="When should you stop EDA and move to modeling?"}
When you can articulate the story: you understand the shape and quality of the data, you know which features look promising, you have handled or noted all major null and outlier patterns, and you have done at least a quick leakage check. EDA is iterative — you will come back after modeling — but you need that first complete pass before fitting anything.
:::

:::drill {type="mcq" q="A numeric feature in your dataset has a 0.97 Pearson correlation with your target variable. What is the most important thing to check first?"}
- [ ] Apply a log transform to reduce the correlation
- [ ] Drop the column because high correlation indicates multicollinearity
- [x] Investigate whether the feature leaks future information about the target
- [ ] Confirm that the distribution is Gaussian before using it
:::

:::drill {type="mcq" q="You receive a dataset where 55% of values in column `days_since_last_login` are null. What is the safest next step?"}
- [ ] Impute all nulls with the column median and proceed
- [ ] Drop the column immediately
- [ ] Convert nulls to -1 and treat it as a numeric feature
- [x] Investigate whether nulls cluster by time, user segment, or data source before deciding how to handle them
:::

:::key-takeaway
EDA is not setup — it is the investigation. Run it in five phases (shape, nulls, distributions, target, correlations), write down what you find, and you will catch 80% of production data bugs before they reach your model.
:::
