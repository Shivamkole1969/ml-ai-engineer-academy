---
id: logging-errors
track: t0-foundations
title: "Logging & error handling for ML services"
badge: CORE
minutes: 9
prereqs: []
tags: [logging, error-handling, python, production, observability]
xp: 45
hot2026: false
---

Imagine your sentiment model has been live for three weeks. Everything looks fine on your dashboard — no 5xx errors, response times are healthy. Then a product manager messages you: "Hey, the model seems to be returning neutral for *everything* since Tuesday." You dig in. Turns out a data pipeline change silently started sending empty strings to your endpoint. Your model dutifully ran inference on empty strings and returned a confident neutral. No crash. No alert. Just wrong answers, quietly, for three days.

That is a logging and error-handling problem. And it is far more common than model accuracy problems.

## What "good" logging actually looks like

Logging is not just `print()` with a timestamp. In a production ML service, logging is your remote eyes. When you are not watching, logs tell you what the model saw, what it decided, and — critically — what went wrong.

Python's built-in `logging` module is all you need to start. The key shift is moving from *event logging* ("something happened") to *structured logging* ("here is a JSON object describing exactly what happened and with what values").

```python {title="Structured logging for an ML inference service" run=false}
import logging
import json
import traceback
from datetime import datetime, timezone

# Configure once at app startup — not inside a function
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",   # we emit JSON ourselves
)
logger = logging.getLogger("inference-service")

def log_event(level: str, event: str, **kwargs):
    """Emit a single structured log line as JSON."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "event": event,
        **kwargs,
    }
    getattr(logger, level)(json.dumps(record))

# --- example: wrap your predict function ---
def predict(text: str, model, threshold: float = 0.5):
    # Guard at the boundary — validate BEFORE inference
    if not text or not text.strip():
        log_event("warning", "empty_input_received", text_repr=repr(text))
        raise ValueError("Input text must be non-empty.")

    try:
        score = model.predict([text])[0]          # your model call here
        label = "positive" if score >= threshold else "negative"
        log_event("info", "prediction_ok", score=round(score, 4), label=label,
                  input_length=len(text))
        return {"label": label, "score": score}

    except Exception as exc:
        log_event("error", "prediction_failed",
                  error_type=type(exc).__name__,
                  error_msg=str(exc),
                  traceback=traceback.format_exc())
        raise   # re-raise so the API layer returns a proper 500
```

A few things to notice. The logger is created once at module level, not inside every function. Every log line is a valid JSON object, which means log-aggregation tools (Datadog, Cloud Logging, Loki) can parse and filter it without regex hacks. And the `predict` function validates its input *before* calling the model — catching the silent-wrong-answer class of bugs at the boundary.

:::why-prod
In production, your model runs unsupervised. Structured logs are the only way to reconstruct *what the model saw* after a problem is reported. Without them, debugging a three-day-old incident becomes a guessing game.
:::

## Logging levels: use them deliberately

Most engineers only use `INFO` and `ERROR`. That is leaving money on the table.

:::table {title="Logging levels and when to use them in ML services"}
| Level | Use it for |
|---|---|
| `DEBUG` | Raw inputs/outputs during local dev — turn off in prod |
| `INFO` | Every prediction served, batch job start/finish, model loaded |
| `WARNING` | Recoverable anomalies: empty input, schema mismatch, feature fallback |
| `ERROR` | Exceptions your service caught — model call failed, downstream timeout |
| `CRITICAL` | Service is about to die — OOM, model file missing on startup |
:::

In practice: set `INFO` in staging and production. Use `WARNING` for anything that *could* be a data quality issue — your future self will thank you when you filter `level=warning` and immediately see 8,000 empty-string inputs on Tuesday.

## Error handling: the two jobs

Error handling in an ML service has two jobs that people often confuse:

**Job 1 — protect the caller.** Your API should never return an unhandled stack trace to a client. Catch exceptions at the boundary, log the full traceback internally, and return a clean error response (e.g., HTTP 422 for bad input, 500 for unexpected failures).

**Job 2 — protect the data.** Silent failures are worse than loud ones. A model that returns a default value when something goes wrong is dangerous unless that fallback is *logged* and *tracked*. If you catch an error and swallow it silently, you have just created Tuesday's incident.

The pattern that works: catch broadly at the outermost layer, log everything, and let specific exceptions (like `ValueError` for bad input) propagate to the caller as structured error responses.

:::gotcha
Never use a bare `except:` or `except Exception: pass`. Swallowing exceptions hides bugs and makes incidents nearly impossible to diagnose. Always log the exception — including the traceback — before deciding what to do next. Even if you return a fallback value, the swallowed exception must appear in your logs.
:::

## What to log per prediction (the minimum viable log)

You do not need to log every token of a long prompt in production. You do need:

- Timestamp (UTC, ISO 8601)
- A unique request or trace ID (so you can join logs across services)
- Input shape or length (not always the raw content — think privacy)
- Model name and version
- Prediction result and confidence score
- Latency in milliseconds
- Any warnings or errors

This gives you enough to debug the "Tuesday neutral" incident in five minutes instead of three days.

:::interview-line
"I treat logs as the contract between my model and the humans who have to debug it at 2 a.m. — structured, complete, and searchable."
:::

:::qa {q="What is the difference between logging and monitoring in an ML service?"}
Logging records discrete events — individual predictions, errors, warnings — as a time-ordered stream of structured records. Monitoring aggregates those events into metrics and alerts (e.g., error rate, p99 latency, prediction distribution). Logs let you debug *what* happened in a specific request; monitoring tells you *when* something is trending wrong across all requests. Both are necessary; neither replaces the other.
:::

:::qa {q="A model endpoint starts returning HTTP 200 responses but with wrong predictions. How would your logging setup catch this?"}
A well-structured setup would log each prediction with the input characteristics, output label, and confidence score. You would then set up a monitoring check on confidence score distribution or label distribution over a rolling window. A sudden shift — like everything collapsing to "neutral" — would trigger an alert even though HTTP status codes look healthy. Input-validation warnings (e.g., empty strings) would also surface in `WARNING`-level logs, pointing directly at the root cause.
:::

:::drill {type="mcq" q="You catch an exception in your inference function, return a safe default prediction, and want to ensure the problem is visible. What is the correct approach?"}
- [ ] Use `except Exception: pass` and increment a counter variable in memory
- [ ] Re-raise the exception so the API returns a 500 to the client every time
- [x] Log the full exception and traceback at ERROR level, then return the safe default
- [ ] Write the error to a local file and check it manually each morning
:::

:::drill {type="mcq" q="Which log level is most appropriate when your feature pipeline falls back to a default value because a feature is missing for a specific user?"}
- [ ] DEBUG — it is a minor internal detail
- [ ] INFO — it is a normal part of the prediction flow
- [x] WARNING — it is recoverable but signals a potential data quality issue worth tracking
- [ ] CRITICAL — any missing feature could break the model
:::

:::key-takeaway
Structure your logs as JSON from day one, validate inputs at the service boundary before inference, and never swallow exceptions silently. A model that fails loudly is far easier to fix than one that fails quietly.
:::
