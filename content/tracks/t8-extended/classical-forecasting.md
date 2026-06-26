---
id: classical-forecasting
track: t8-extended
title: "Forecasting: ARIMA, Prophet"
badge: CORE
minutes: 9
prereqs: []
tags: [time-series, arima, prophet, forecasting, statistics, seasonality]
xp: 45
hot2026: false
---

Imagine you run the supply chain team at a mid-sized e-commerce company. Black Friday is six weeks out. Your warehouse manager walks in and asks: "How many units of the top 50 SKUs should we stock?" You can't say "I don't know." You need a number, and ideally a confidence interval around it. That is the forecasting problem — and two tools have earned their place in almost every production pipeline that solves it: **ARIMA** and **Prophet**.

## Why Classical Forecasting Still Matters

Before transformer-based time series models became popular, ARIMA and Prophet were (and often still are) the go-to choices in production. They are fast to fit, require no GPU, are interpretable, and perform surprisingly well when your series has clean structure. Large companies use Prophet in production for thousands of series at once. ARIMA still wins in many competition benchmarks on short, stationary series.

:::why-prod
In production you often need to forecast hundreds or thousands of series (one per SKU, one per store, one per user segment). Classical models fit in milliseconds each and run without a GPU cluster. Explainability matters too — when the CFO asks "why do we expect a dip in week 3?", Prophet can show you the seasonal component.
:::

## ARIMA: Three Letters, One Powerful Idea

ARIMA stands for **AutoRegressive Integrated Moving Average**. That sounds academic, but each word maps to something concrete.

**AR (AutoRegressive)** — The model predicts the next value using a weighted sum of the last *p* values. Think of it as: "tomorrow's sales look a lot like the last three days."

**I (Integrated)** — Before fitting, we "difference" the series *d* times to remove trends and make it stationary. A stationary series has a stable mean and variance over time. ARIMA needs this. Without it, the model fights the trend instead of learning from it.

**MA (Moving Average)** — The model also uses the last *q* forecast errors as predictors. This corrects for recent surprises: "I underestimated by 50 units yesterday, so let me adjust."

You write this as **ARIMA(p, d, q)** — three integers. For example, ARIMA(1, 1, 1) uses one lag, differences once, and uses one error term.

Seasonal patterns get their own extension: **SARIMA(p, d, q)(P, D, Q, m)**, where *m* is the season length (12 for monthly data with annual cycles).

:::table {title="ARIMA parameter cheat-sheet"}
| Parameter | What it controls | How to choose |
|---|---|---|
| p | Number of autoregressive lags | ACF / PACF plots, or auto_arima() |
| d | Differencing order | ADF test for stationarity |
| q | Moving-average window | ACF / PACF plots, or auto_arima() |
| m | Seasonal period | Domain knowledge (7=weekly, 12=monthly) |
:::

In practice, most people skip the manual ACF/PACF analysis and use `auto_arima()` from the `pmdarima` library, which searches the parameter space and returns the best fit by AIC.

:::gotcha
ARIMA requires a **stationary** series. If your data has a strong upward trend or obvious seasonality you haven't removed (differenced), the model will produce garbage forecasts with misleadingly tight confidence intervals. Always run an ADF test or plot the rolling mean before trusting an ARIMA output.
:::

## Prophet: Forecasting for the Rest of Us

Prophet (open-sourced by Meta/Facebook) was designed for analysts who are experts in their domain but not in time series statistics. Its interface is two lines of code. Its model is an **additive decomposition**:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + noise
```

You feed it a dataframe with a `ds` (datestamp) column and a `y` (value) column. That's it.

```python {title="Prophet quickstart — pip install prophet" run=false}
from prophet import Prophet
import pandas as pd

# df must have columns: 'ds' (datetime) and 'y' (numeric target)
df = pd.read_csv("daily_sales.csv", parse_dates=["ds"])

model = Prophet(
    seasonality_mode="multiplicative",  # use 'additive' if peaks scale uniformly
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,            # only useful for sub-daily data
)

# Add custom holiday effects (company sale events, public holidays, etc.)
model.add_country_holidays(country_name="US")

model.fit(df)

# Forecast 30 days forward
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Inspect components — great for explaining to stakeholders
fig = model.plot_components(forecast)
fig.savefig("components.png")

# Key columns in forecast:
# yhat        → point forecast
# yhat_lower  → lower 80% interval
# yhat_upper  → upper 80% interval
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(30))
```

Prophet handles **missing data** gracefully, detects **changepoints** (moments when the trend bends), and lets you inject domain knowledge — like a product launch date or a promotional event — as a regressor or holiday. That last feature is huge in business forecasting.

## Choosing Between ARIMA and Prophet

Neither is universally better. Use both, compare validation errors, and ship whichever wins on your series.

**Lean toward ARIMA when:**
- Your series is short and stationary (or easily made so).
- You need maximum statistical rigor and well-understood confidence intervals.
- You have univariate data without complex seasonality.

**Lean toward Prophet when:**
- You have strong weekly + annual seasonality together.
- You need to explain the forecast to non-technical stakeholders.
- You want to inject holidays or known future events.
- You're fitting hundreds of series automatically with minimal tuning.

:::interview-line
"ARIMA is my first call for stationary, short series where I want tight statistical guarantees; Prophet is my first call when I have rich seasonality, known holidays, and need stakeholders to trust the output."
:::

:::qa {q="What does the 'I' in ARIMA do and why does it matter?"}
The 'I' stands for Integrated — it means we difference the series *d* times before modelling. Differencing removes trends and makes the series stationary, which ARIMA mathematically requires. Without stationarity, the parameter estimates are unreliable and the confidence intervals lie.
:::

:::qa {q="When would you choose Prophet over ARIMA in a production system?"}
Prophet shines when the series has multiple overlapping seasonalities (say, weekly and yearly), when you want to add known future events like holidays or promotional campaigns as regressors, or when you need to auto-forecast a large number of series with minimal manual tuning. ARIMA would need a separate SARIMA specification per series, which is tedious at scale.
:::

:::qa {q="How do you evaluate a forecasting model before deploying it?"}
Use a time-aware cross-validation strategy — often called **walk-forward validation** or **backtesting**. You train on history up to time T, forecast the next N steps, measure error (MAE, RMSE, or MAPE), then slide the window forward and repeat. Never use random k-fold splits on time series data — that leaks future information into training.
:::

:::drill {type="mcq" q="You fit an ARIMA(1,0,1) model and your residuals show a clear spike at lag 12 in the ACF plot. What is the most likely fix?"}
- [ ] Increase the MA order q from 1 to 2
- [ ] Apply first differencing (set d=1)
- [x] Add a seasonal component — switch to SARIMA(1,0,1)(1,0,1,12)
- [ ] Remove the AR term entirely
:::

:::drill {type="mcq" q="Prophet's `seasonality_mode='multiplicative'` vs `'additive'` — when should you use multiplicative?"}
- [ ] When the series has no trend
- [ ] When the holiday effect is constant across years
- [x] When seasonal swings grow proportionally as the trend level rises
- [ ] When you have fewer than 100 observations
:::

:::key-takeaway
ARIMA is the rigorous statistical workhorse for stationary, short series; Prophet is the practical, explainable choice when you have rich seasonality, known events, and need to forecast at scale. In production, always validate both with walk-forward backtesting before choosing.
:::
