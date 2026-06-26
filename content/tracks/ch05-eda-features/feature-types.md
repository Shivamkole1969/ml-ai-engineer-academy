---
id: feature-types
track: ch05-eda-features
title: "Feature types & transformations"
badge: CORE
minutes: 9
prereqs: []
tags: [features, transformations, preprocessing, encoding, scaling, one-hot, target-encoding]
xp: 45
hot2026: false
---

Imagine you train a salary prediction model on your company's HR data. It nails the dev and test sets. You push it to production. Predictions go haywire within a week. Three days of debugging later, a data engineer finds the culprit: the raw income column had a handful of executive outliers, nobody log-transformed it before training, and the scaling parameters baked into the model were meaningless in a new payroll system that reports salaries in a different currency unit.

Feature types and transformations are not busywork. They are the bridge between raw messy data and a model that actually learns — and keeps learning correctly in production.

## The Four Families of Features

Every column you feed a model belongs to one of these families, and each one needs its own treatment.

**Numerical — continuous**: height, revenue, temperature. Can take any value in a range.

**Numerical — discrete**: purchase count, star ratings (1–5), page views. Whole numbers, usually bounded.

**Categorical — nominal**: country, product category, browser type. No natural order. "France" is not greater than "Germany."

**Categorical — ordinal**: T-shirt size (S < M < L), customer tier (Bronze < Silver < Gold). There IS a meaningful order — and preserving it matters.

**Binary**: is_fraud (0/1), has_subscription (True/False). A special case of categorical, usually fine as-is.

**Temporal**: dates and timestamps. Not useful raw — but rich once you extract parts: day of week, hour of day, days since last event.

Misidentifying a type causes quiet, hard-to-spot bugs. Treating a nominal "country" column as an integer (1=US, 2=France, 3=Germany…) tells your linear model that Germany is three times the US — pure nonsense. Treating an ordinal T-shirt size as plain text throws away the ordering entirely.

:::why-prod
Mis-typed features rarely crash a pipeline. They silently corrupt model weights. These bugs surface weeks after deployment as mysteriously drifting accuracy — with no stack trace to follow.
:::

## Transformations That Actually Matter

Raw features almost never enter a model unchanged. Here is the standard playbook.

:::table {title="Common transformations by feature type"}
| Feature type | Common transforms | Why |
|---|---|---|
| Continuous, right-skewed | Log, sqrt, Box-Cox | Compresses outliers; helps linear models |
| Continuous, arbitrary scale | StandardScaler, MinMaxScaler | Gradient descent converges faster |
| Nominal categorical | One-hot encoding | Turns labels into numbers without imposing fake order |
| Ordinal categorical | Ordinal / label encoding | Preserves the real rank |
| High-cardinality nominal | Target encoding, embeddings | One-hot creates hundreds of columns |
| Temporal | Extract hour, day-of-week, time-since | Models cannot do date arithmetic natively |
| Binary | Leave as 0/1 | No transform needed |
:::

A few worth anchoring in detail:

**Log transform** is your friend for income, price, and any count column with a long right tail. Use `np.log1p` (adds 1 before taking the log) to handle zeros gracefully.

**StandardScaler** subtracts the mean and divides by standard deviation. It is the default for most algorithms. **MinMaxScaler** squeezes values to 0–1 — preferred when you need a strict bounded range, like inputs to a sigmoid layer.

**One-hot encoding** creates one binary column per category. Clean and interpretable, but explodes your feature space. A country column with 200 unique values becomes 200 columns — manageable for tree models, potentially expensive for neural nets.

**Target encoding** replaces each category with the mean of the target for that group. Compact and powerful, but must be computed on training data only. Otherwise it leaks the answer straight into your features.

```python {title="Core transforms with sklearn" run=false}
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Sample dataset
df = pd.DataFrame({
    "revenue": [500, 1200, 45000, 300, 8900],   # skewed — needs log
    "country": ["US", "IN", "US", "DE", "IN"],  # nominal — needs one-hot
    "days_since_signup": [10, 200, 5, 90, 30],  # continuous — standard scale
})

# Log-transform the skewed column first
df["revenue_log"] = np.log1p(df["revenue"])

# ColumnTransformer applies the right transform to each column type
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), ["revenue_log", "days_since_signup"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["country"]),
])

X_transformed = preprocessor.fit_transform(df)
print(X_transformed.shape)  # (5, 2 numeric + n_country_cols)

# Run locally:  pip install scikit-learn pandas numpy
```

:::gotcha
Fitting your scaler or encoder on the full dataset — before the train/test split — is textbook **data leakage**. Test-set statistics (mean, standard deviation, category list) bleed into the training process. The model looks better than it is, and then fails in production where those statistics differ. Always fit transformers on training data only, then call `.transform()` on everything else. Wrapping steps in a sklearn `Pipeline` enforces this automatically.
:::

## Derived and Interaction Features

Sometimes the raw columns are not the signal — combinations are.

**Interaction features**: multiply two columns (`price × quantity = total_spend`). Captures joint effects that each column alone cannot express.

**Ratio features**: `clicks / impressions = CTR`. Division often reveals the real story better than either number alone.

**Lag features**: yesterday's value, last week's rolling average. Essential for time-series and sequential models.

**Bucketed features**: age → age band (0–18, 18–35, 35–60, 60+). Useful when a threshold matters more than the exact value — a 35-year-old and a 36-year-old may belong to meaningfully different customer segments.

Domain knowledge makes these obvious. A model can technically discover interactions on its own, but it needs far more data and training time to do so. Handing it the obvious derived features is almost always worth the ten lines of code.

:::interview-line
"Every feature type has a natural transformation — the trick is knowing which one, fitting it only on training data, and packaging it in a pipeline so nothing can drift between training and production."
:::

:::qa {q="Why does feature scaling matter for some models but not others?"}
Algorithms that rely on distances or gradient descent — k-NN, SVMs, logistic regression, neural nets — are sensitive to scale. A revenue column in the millions can swamp a click-rate column that ranges 0–1, forcing the optimizer to fight the imbalance instead of learning patterns. Tree-based models like Random Forest and XGBoost split on thresholds and are scale-invariant, so they do not need scaling. Applying it anyway is harmless but unnecessary for trees.
:::

:::qa {q="When would you choose target encoding over one-hot encoding for a categorical column?"}
Target encoding wins when a categorical column has high cardinality — hundreds or thousands of unique values. One-hot would create an unmanageable number of sparse columns. The key risk is leakage: category means must be computed on training data only and applied to validation and test sets. In practice, use cross-fold target encoding or add a smoothing term to stabilize categories that appear rarely in the training set.
:::

:::drill {type="mcq" q="A 'city' column has 900 unique values. Which encoding strategy is most practical?"}
- [ ] One-hot encoding — always the safest and most interpretable choice
- [ ] Label encoding — assign integers 0 to 899 to preserve some structure
- [x] Target encoding or learned embeddings — one-hot creates 900 sparse columns and label encoding implies a false numeric order
- [ ] Drop the column — high-cardinality categoricals never add signal
:::

:::drill {type="mcq" q="You fit a StandardScaler on your full dataset before splitting into train and test sets. What goes wrong?"}
- [ ] Nothing — scaling is deterministic and has no effect on model learning
- [ ] The test set may have column names that do not match training
- [x] Test statistics (mean, std) influence the scaler, giving the model indirect knowledge of test data during training
- [ ] StandardScaler does not support being called on a test set at all
:::

:::key-takeaway
Know your feature type, apply the right transform, fit it only on training data, and wrap it in a pipeline. That four-step habit is the difference between a model that scores well in a notebook and one that behaves correctly in production six months from now.
:::
