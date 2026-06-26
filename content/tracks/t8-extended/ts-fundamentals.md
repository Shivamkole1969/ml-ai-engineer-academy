---
id: ts-fundamentals
track: t8-extended
title: "Time series fundamentals"
badge: CORE
minutes: 9
prereqs: []
tags: [time-series, stationarity, seasonality, autocorrelation, decomposition, forecasting]
xp: 45
hot2026: false
---

Imagine you're the ML engineer at a food delivery startup. Your demand model aces every offline test. You ship it. Three weeks later, the ops team is furious — the model keeps over-ordering ingredients every Tuesday, even when nothing unusual happens. What went wrong?

The model saw last Tuesday's dinner rush as a signal that "demand is permanently rising." It had no concept of weekly rhythm. Nobody told it that time has *structure* — that what happened at 7 pm last Tuesday is deeply connected to what will happen at 7 pm this Tuesday.

That's the core challenge of time series data. Standard ML treats rows as independent. Time series rows are anything but.

## What Makes Time Series Data Different

A time series is a sequence of observations collected at regular time intervals — hourly sensor readings, daily sales figures, monthly active users, weekly churn rates. The defining feature: **order matters**. Shuffle the rows and you destroy the signal.

This breaks the fundamental assumption baked into most ML algorithms: that observations are *independent and identically distributed* (IID). In time series, observation at time *t* is almost always correlated with observation at time *t-1*, *t-2*, and beyond. That correlation is not noise — it's the signal you want to exploit.

:::why-prod
Most real-world business metrics are time series: revenue, latency, error rates, user counts. Getting forecasts wrong in production means over-staffing, under-buying inventory, or missing SLA breaches before they happen. Understanding the underlying structure is what separates a model that works from one that looks good in a notebook and fails on Monday morning.
:::

## The Four Components

Every time series can be decomposed into four parts. Learn to see these and you'll diagnose most problems fast.

:::table {title="Time Series Components"}
| Component | What it is | Example |
|---|---|---|
| **Trend** | Long-run direction of the series (up, down, flat) | Monthly revenue climbing 5% YoY |
| **Seasonality** | Repeating pattern at a fixed, known period | Spike every December, dip every January |
| **Cyclicality** | Repeating pattern with *no fixed period* | Economic boom-and-bust cycles |
| **Residual (noise)** | What's left after removing the above three | Random daily fluctuations |
:::

Decomposition matters because different algorithms handle different components well. Prophet is great at trend + seasonality. ARIMA assumes you've already removed the trend (made the series *stationary*). Neural nets can in principle learn all four — but they need enough data.

## Stationarity: The Property Everything Else Depends On

A series is **stationary** when its statistical properties — mean, variance, autocorrelation — don't change over time.

Why does this matter? Most classical forecasting methods (ARIMA and its cousins) are mathematically derived under the assumption of stationarity. Feed them a drifting series and their parameter estimates are garbage.

A non-stationary series has a visible trend, a shifting mean, or growing variance. The fix is usually **differencing**: subtract each observation from the previous one. One round of differencing removes a linear trend. Two rounds removes a quadratic trend. That's what the "I" in ARIMA stands for — *Integrated*, meaning you difference until stationary.

```python {title="Check stationarity with the ADF test" run=false}
# pip install statsmodels
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Load your series (e.g., daily sales)
series = pd.read_csv("sales.csv", parse_dates=["date"], index_col="date")["revenue"]

result = adfuller(series)
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")

# Rule of thumb: p < 0.05 → stationary (reject the null of a unit root)
# p >= 0.05 → non-stationary → try differencing
if result[1] >= 0.05:
    series_diff = series.diff().dropna()
    result2 = adfuller(series_diff)
    print(f"\nAfter 1st differencing → p-value: {result2[1]:.4f}")
```

## Autocorrelation: The Hidden Signal

**Autocorrelation** measures how correlated a series is with a lagged version of itself. High autocorrelation at lag 7 on daily data? Your series has a weekly pattern. High autocorrelation at lag 1? Yesterday's value is a strong predictor of today's.

Two plots are your best friends here:

- **ACF** (Autocorrelation Function): correlation at each lag. Cuts off after lag *q* → suggests an MA model.
- **PACF** (Partial Autocorrelation Function): correlation at each lag *after* removing the effect of shorter lags. Cuts off after lag *p* → suggests an AR model.

Learning to read ACF/PACF plots is how practitioners chose ARIMA orders before auto-ARIMA tools existed — and it's still how you sanity-check what those tools find.

:::gotcha
Never shuffle time series data before splitting into train/test. Standard k-fold cross-validation leaks future information into training. Always split chronologically: train on the past, test on the future. For cross-validation, use **time-series split** (walk-forward validation) where each fold's test window is strictly after its training window. Getting this wrong is one of the most common reasons a "great" notebook model fails in production.
:::

:::interview-line
"Stationarity isn't an assumption you hope holds — it's a property you *engineer* through differencing, log transforms, or detrending before fitting classical models."
:::

:::qa {q="What is stationarity and why does it matter for forecasting?"}
A stationary series has a constant mean, variance, and autocorrelation structure over time. It matters because most classical forecasting models (ARIMA family) assume stationarity — their parameters are estimated under that assumption. A non-stationary input produces unreliable forecasts. You achieve stationarity by differencing, log-transforming, or seasonally adjusting the series before fitting.
:::

:::qa {q="What does autocorrelation tell you, and how do you use ACF/PACF plots?"}
Autocorrelation tells you how strongly a time series predicts itself at past lags. The ACF plot shows the raw correlation at each lag; a sharp cutoff at lag *q* suggests a Moving Average (MA) component. The PACF shows the correlation after removing shorter-lag effects; a sharp cutoff at lag *p* suggests an Autoregressive (AR) component. Together, they guide you to the right ARIMA(p, d, q) order rather than blindly guessing.
:::

:::qa {q="Why can't you use standard k-fold cross-validation on time series?"}
Standard k-fold randomly shuffles data across folds, which means future observations can appear in the training set. This is data leakage — the model learns from information it could never have had in production. For time series you must use walk-forward (time-series split) cross-validation: each fold trains on all data up to a cutoff and tests on the window immediately after it, preserving the arrow of time.
:::

:::drill {type="mcq" q="A daily retail sales series has a p-value of 0.32 from the ADF test. What should you do next?"}
- [ ] Fit an ARIMA model directly — the series is stationary
- [x] Apply first-order differencing and re-test with ADF
- [ ] Switch to a neural network, which doesn't need stationarity
- [ ] Collect more data until the p-value drops below 0.05
:::

:::drill {type="mcq" q="Your ACF plot shows significant spikes at lags 7, 14, and 21 on daily data. What does this most likely indicate?"}
- [ ] The series has a linear upward trend
- [ ] The series is stationary and ready to model
- [x] The series has a weekly seasonal pattern
- [ ] The model is overfitting to recent data
:::

:::drill {type="mcq" q="You're asked to cross-validate a sales forecasting model. Which approach is correct?"}
- [ ] Stratified k-fold to balance sales volumes across folds
- [ ] Random 80/20 split repeated 5 times
- [ ] Leave-one-out cross-validation on individual days
- [x] Walk-forward validation: train up to month N, test on month N+1, repeat
:::

:::key-takeaway
Time series data has memory — order matters, structure exists, and the future leaks into the past if you're not careful with splits. Master the four components (trend, seasonality, cyclicality, noise) and stationarity, and every forecasting method you meet afterward will make intuitive sense.
:::
