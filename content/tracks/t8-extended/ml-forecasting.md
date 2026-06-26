---
id: ml-forecasting
track: t8-extended
title: "ML for forecasting & leakage traps"
badge: CORE
minutes: 9
prereqs: []
tags: [forecasting, time-series, leakage, gradient-boosting, feature-engineering, walk-forward-validation]
xp: 45
hot2026: false
---

Picture this: you spend two weeks building a demand-forecasting model for an e-commerce client. XGBoost, 200 features, cross-validated R² of 0.97. You deploy it on a Friday. By Monday, operations is calling — the model is wildly off. After a frantic weekend of log-diving, you find the culprit: one innocent-looking column called `weekly_avg_sales` was computed on the *entire* dataset, including future weeks. Your "powerful" model had memorized tomorrow's answer. Welcome to data leakage — the silent killer of time-series ML.

## Why go beyond ARIMA and Prophet?

Classical methods (ARIMA, Exponential Smoothing, Prophet) are great at capturing trend and seasonality in a single numeric series. But real forecasting problems are richer:

- You have *dozens of external signals* — promotions, weather, competitor prices, holidays.
- The series has *non-linear patterns* that stats models can't bend to fit.
- You need a *single model* that generalizes across hundreds of products or store locations.

Gradient-boosted trees (XGBoost, LightGBM, CatBoost) and neural nets handle all of this. They treat forecasting as a standard supervised regression problem — you engineer features from the past, predict the future. Simple framing, powerful results.

:::why-prod
Production forecasting pipelines almost always run on tabular ML (LightGBM is the industry workhorse) rather than pure stats models, because the feature matrix can absorb anything — lag values, categorical IDs, external regressors — without bespoke per-series modeling. Faster to retrain, easier to monitor, and scales to millions of series.
:::

## Turning a time series into a feature matrix

The core transformation: for each row (one point in time), look *backward* to build features, then predict what comes *next*.

:::table {title="Common time-series features"}
| Feature type | Example | What it captures |
|---|---|---|
| Lag | `sales_lag_7` (value 7 days ago) | Autocorrelation, same-day-last-week |
| Rolling stat | `sales_roll_mean_28` (28-day avg) | Smooth trend |
| Rolling std | `sales_roll_std_7` | Volatility / uncertainty |
| Diff | `sales_diff_1` (change from yesterday) | Momentum |
| Calendar | `day_of_week`, `is_holiday` | Seasonality |
| External | `promo_flag`, `temperature` | Exogenous signals |
:::

The key rule is deceptively simple: **every feature must be knowable at prediction time.** If you're forecasting next week's sales, a feature must be available right now — or be a known future value (like a scheduled promotion).

## The leakage trap — the most important concept in this lesson

Data leakage means your training data contains information that wouldn't exist in the real world at prediction time. In time series, it wears three disguises:

**1. Future data smuggled into features**
Computing `rolling_mean` over a window that includes future rows. Sounds obvious; harder to spot when it happens inside a library call or a join.

**2. Target-derived features**
Using a derived column (like `7-day average of the target`) that was computed across the full dataset, not just the historical window. Classic silent killer.

**3. Wrong train/test split**
Using `sklearn`'s default `train_test_split` on time series data. It samples randomly, so your training set contains rows *from the future* relative to your test set. The model learns future context it should never have.

:::gotcha
Never use `train_test_split(shuffle=True)` on time series data. Random shuffling mixes future rows into the training set. Always split by a cutoff date or use walk-forward (rolling-origin) validation.
:::

## Walk-forward validation — the right way to evaluate

Standard k-fold cross-validation assumes rows are exchangeable. Time series rows are *not* — order matters. The correct pattern is **walk-forward validation** (also called rolling-origin or expanding-window cross-validation):

1. Train on rows 1–T₁, validate on rows T₁+1 to T₁+H (H = forecast horizon).
2. Expand training window to 1–T₂, validate on T₂+1 to T₂+H.
3. Repeat, always predicting only into the future relative to the training window.

This mimics how the model will actually be used: trained on the past, predicting the near future. Metrics from walk-forward splits are honest estimates of production accuracy.

```python {title="Walk-forward validation with LightGBM" run=false}
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

# df is sorted by date; 'target' is the value to forecast
# Assumes lag features were built BEFORE this function (leak-free)

def walk_forward_validate(df, date_col, target_col, feature_cols,
                          n_splits=5, horizon=7):
    """
    Expanding-window walk-forward validation.
    Returns mean MAE across all folds.
    """
    dates = df[date_col].sort_values().unique()
    split_size = len(dates) // (n_splits + 1)
    maes = []

    for fold in range(1, n_splits + 1):
        cutoff = dates[fold * split_size]
        val_end = dates[min((fold + 1) * split_size, len(dates) - 1)]

        train = df[df[date_col] < cutoff]
        val   = df[(df[date_col] >= cutoff) & (df[date_col] < val_end)]

        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                   num_leaves=64, random_state=42)
        model.fit(train[feature_cols], train[target_col])

        preds = model.predict(val[feature_cols])
        mae   = mean_absolute_error(val[target_col], preds)
        maes.append(mae)
        print(f"Fold {fold} | cutoff={cutoff} | MAE={mae:.4f}")

    print(f"\nMean MAE: {np.mean(maes):.4f}")
    return maes

# Run locally (pip install lightgbm scikit-learn pandas):
# maes = walk_forward_validate(df, 'date', 'sales', feature_cols)
```

## Building lag features without leaking

The safest pattern: compute all lag and rolling features using only a `groupby + shift`, never an unshifted aggregate over the full frame.

```python {title="Leak-free lag and rolling feature builder" run=false}
import pandas as pd

def add_lag_features(df, target_col, lags, windows, group_col=None):
    """
    Adds lag and rolling-mean features.
    Uses shift(1) to ensure t-1 is the earliest information included.
    """
    df = df.sort_values("date").copy()
    base = df.groupby(group_col)[target_col] if group_col else df[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = base.shift(lag)

    for w in windows:
        # shift(1) first — rolling never sees the current row
        df[f"{target_col}_roll_mean_{w}"] = (
            base.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{w}"] = (
            base.shift(1).rolling(window=w, min_periods=1).std()
        )

    return df.dropna()   # drop rows with insufficient history

# Usage:
# df = add_lag_features(df, "sales", lags=[1,7,14], windows=[7,28],
#                        group_col="product_id")
```

The critical detail: `.shift(1)` before `.rolling()` ensures the rolling window never peeks at the current row. One missing `.shift()` and you have leakage.

## Production checklist

Before you ship a forecasting model, run through these:

- Did every feature use only data available *before* the prediction date?
- Did you validate with walk-forward splits, not random shuffle?
- Did you refit lag statistics on the training window only, then apply to test?
- Is your retrain cadence aligned with how stale lag features become?
- Do you have monitoring for distribution shift in your feature inputs?

:::interview-line
"I always build lag features with an explicit `.shift()`, split by cutoff date, and validate with walk-forward CV — random-shuffle CV on time series is measuring leakage, not model quality."
:::

:::qa {q="What is data leakage in time-series forecasting and why is it so dangerous?"}
Leakage means your model trains on information that won't be available at real prediction time — typically future target values smuggled into feature computations or a random train/test split that mixes future rows into training data. It's dangerous because validation metrics look excellent while production performance is terrible. The model has effectively memorized the answer.
:::

:::qa {q="Why can't you use standard k-fold cross-validation for time series?"}
Standard k-fold shuffles rows randomly, so training folds can contain data points that occurred *after* validation fold points. The model learns future context it should never have, inflating metrics. Walk-forward (rolling-origin) validation always trains on the past and tests on a strictly future window, mimicking real deployment conditions.
:::

:::qa {q="What's the advantage of LightGBM over Prophet for a multi-product demand forecast?"}
LightGBM trains a single model across all products, incorporating external features like promotions, weather, and competitor prices naturally through the feature matrix. Prophet fits one model per series and handles exogenous regressors less flexibly. LightGBM also scales better — retraining one model on millions of rows is faster than fitting thousands of individual series models.
:::

:::drill {type="mcq" q="You compute a `7-day rolling average of sales` feature for each row in your dataset using the entire dataframe, then train on the first 80% of rows by date and test on the last 20%. What is wrong?"}
- [ ] Nothing — rolling averages are always safe to compute over the full dataset.
- [x] The rolling average for rows in the training set includes future values from the test period, leaking target information.
- [ ] You should use a 14-day window instead of 7.
- [ ] The 80/20 split ratio is incorrect for time series.
:::

:::drill {type="mcq" q="Which of the following correctly prevents leakage when building a 7-day rolling mean feature for forecasting?"}
- [ ] `df['roll_7'] = df['sales'].rolling(7).mean()`
- [ ] `df['roll_7'] = df['sales'].rolling(7).mean().shift(-1)`
- [x] `df['roll_7'] = df['sales'].shift(1).rolling(7).mean()`
- [ ] `df['roll_7'] = df['sales'].expanding().mean()`
:::

:::drill {type="mcq" q="A colleague says your walk-forward CV MAE is 120 units, but after deployment the model's MAE is 310 units. What is the most likely explanation?"}
- [ ] LightGBM is poorly calibrated by default.
- [ ] The horizon length was set too short.
- [x] There was leakage in feature construction — validation metrics were optimistic.
- [ ] The model needs more training data.
:::

:::key-takeaway
ML models can massively outperform classical stats models on forecasting tasks — but only if you build features with strict look-back windows and validate with walk-forward splits. One accidentally unshifted rolling mean can make your model look brilliant in training and useless in production.
:::
