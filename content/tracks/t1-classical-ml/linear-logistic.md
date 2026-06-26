---
id: linear-logistic
track: t1-classical-ml
title: "Linear & logistic regression, done right"
badge: CORE
minutes: 9
prereqs: []
tags: [regression, classification, regularization, sklearn, interpretability, coefficients]
xp: 45
hot2026: false
---

Imagine you've spent three weeks building a churn-prediction model. A VP walks in and asks: "Why did the model flag this customer?" With a neural network you'd say "uh, the gradients…" and hope for the best. With logistic regression, you point to a screen and say: "Their usage dropped 40% and they missed two payments — those two features drive 80% of the score." Meeting over. In production, *explainability is a feature*, and regression models ship it for free.

## Linear regression: the honest workhorse

Linear regression fits a straight-line relationship between features and a continuous output.

```
ŷ = w₀ + w₁x₁ + w₂x₂ + … + wₙxₙ
```

Each `w` (weight or coefficient) tells you: "holding everything else equal, how much does `ŷ` change per unit of this feature?" That sentence alone makes linear regression powerful — it's auditable, debuggable, and fast.

The model minimises **Mean Squared Error (MSE)** — the average squared gap between your predictions and reality. Squaring the errors punishes big mistakes harder than small ones, which is usually what you want.

:::why-prod
In regulated industries (finance, healthcare, insurance) models are often *required* to be explainable. Linear models pass compliance reviews that black-box models don't. Even when you later graduate to gradient boosting, a linear baseline is your sanity check — if XGBoost can't beat plain regression by a meaningful margin, the fancier model may just be overfitting.
:::

## The regularisation add-on you always need

Raw linear regression will happily overfit when features outnumber rows, or when features are correlated. The fix is regularisation — a penalty that shrinks coefficients toward zero.

:::table {title="Ridge vs Lasso at a glance"}
| | Ridge (L2) | Lasso (L1) |
|---|---|---|
| Penalty | sum of squared weights | sum of absolute weights |
| Effect | shrinks all weights | drives some to exactly 0 |
| Use when | all features probably matter | you want automatic feature selection |
| sklearn param | `Ridge(alpha=1.0)` | `Lasso(alpha=0.1)` |
:::

**Alpha** is the strength knob. Higher alpha = more shrinkage. Always tune it via cross-validation — never eyeball it.

```python {title="Ridge regression with scaled features" run=false}
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Always scale before Ridge/Lasso — penalties are scale-sensitive
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge()),
])

param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
cv = GridSearchCV(pipe, param_grid, scoring="neg_mean_squared_error", cv=5)
cv.fit(X_train, y_train)

print("Best alpha:", cv.best_params_)
print("Test R²:", cv.best_estimator_.score(X_test, y_test))
# Inspect coefficients after fitting
coefs = cv.best_estimator_.named_steps["model"].coef_
```

:::gotcha
Not scaling before regularisation is the most common beginner mistake. A feature measured in millions (salary) will get a tiny raw coefficient; one measured in fractions (ratio) will get a huge one. The L1/L2 penalty then unfairly crushes the large-coefficient feature. Always wrap your regression in a `Pipeline` with `StandardScaler` first.
:::

## Logistic regression: regression in disguise

Despite the name, logistic regression is a **classification** model. It predicts the *probability* that a sample belongs to the positive class.

The trick: pass the linear combination through a **sigmoid function** that squashes any real number into [0, 1].

```
P(y=1) = 1 / (1 + e^(−z))   where z = w₀ + w₁x₁ + …
```

A predicted probability above some threshold (default 0.5) is labelled class 1. Crucially, you get a *probability*, not just a label — which means you can tune the threshold to trade precision for recall based on business cost.

:::why-prod
Downstream systems almost always need a score, not a binary label. A fraud-detection pipeline might block transactions above 0.95, flag for review between 0.70–0.95, and let through everything below. Logistic regression gives you that score natively. Most tree-based models need calibration post-hoc to produce reliable probabilities.
:::

## Reading the coefficients

After fitting, each coefficient represents the change in **log-odds** per unit change in that feature. Log-odds aren't intuitive, so exponentiate them to get **odds ratios**.

If `exp(coef)` for "missed payment" = 2.3, that means missing a payment multiplies the odds of churn by 2.3× — a concrete business number your stakeholders can act on.

:::gotcha
Logistic regression is not magic. It assumes a *linear decision boundary*. If the true boundary is curved (e.g., churn is high both for very new AND very old customers), logistic regression will miss it. Add polynomial features or switch to a tree model for those cases.
:::

## When to use which

- **Continuous target** (price, temperature, revenue) → linear regression (with Ridge/Lasso)
- **Binary target** (churn yes/no, fraud yes/no) → logistic regression
- **Multiclass** (product category A/B/C) → logistic regression with `multi_class='multinomial'`
- **Need probability calibration out of the box** → logistic regression over tree models

:::interview-line
"I start every project with a regularised linear or logistic baseline — it's the fastest sanity check, passes compliance reviews, and gives me interpretable coefficients before I touch gradient boosting."
:::

:::qa {q="Why do you always scale features before running logistic or linear regression?"}
Regularisation penalties are magnitude-sensitive. A feature with values in the thousands will naturally have a tiny coefficient, while one in the 0–1 range will have a large one. Without scaling, the penalty unfairly crushes high-magnitude features. StandardScaler brings all features to mean=0, std=1 so the penalty treats them equally.
:::

:::qa {q="What does the threshold in logistic regression control, and why would you change it?"}
The threshold converts a predicted probability into a class label. At 0.5, anything above is predicted positive. Lowering the threshold (e.g., to 0.3) catches more true positives but increases false positives — useful when missing a positive is costly (medical screening). Raising it (e.g., 0.8) reduces false positives at the cost of missing some positives — useful when false alarms are expensive (fraud blocks on a good transaction). Always tune the threshold against a business cost function, not just accuracy.
:::

:::qa {q="A colleague says Lasso 'performed feature selection automatically.' What does that mean?"}
L1 regularisation (Lasso) adds a penalty proportional to the absolute value of each coefficient. Mathematically, this drives less-informative feature coefficients exactly to zero — those features are removed from the model. Ridge, by contrast, only shrinks coefficients toward zero without zeroing them out. So Lasso is useful when you suspect only a subset of your features are actually predictive.
:::

:::drill {type="mcq" q="Your model predicts house prices (in dollars). You add 'distance to school in km' as a feature. You train a Ridge regression WITHOUT scaling. What goes wrong?"}
- [ ] The model will refuse to converge with mixed units
- [x] The regularisation penalty will disproportionately shrink the distance coefficient because its values are small, even if it's an important feature
- [ ] Ridge regression requires all features to be binary
- [ ] Nothing — Ridge handles unscaled features perfectly
:::

:::drill {type="mcq" q="Your logistic regression achieves 97% accuracy on a fraud dataset where 97% of samples are non-fraud. What is the most likely problem?"}
- [ ] The model is overfitting due to too many features
- [ ] The learning rate is too high
- [x] The model is predicting 'not fraud' for every sample — accuracy is a misleading metric on imbalanced data
- [ ] Logistic regression cannot handle binary targets
:::

:::key-takeaway
Always regularise (Ridge or Lasso), always scale first, and always read your coefficients — they're not just model internals, they're business insights your stakeholders can act on.
:::
