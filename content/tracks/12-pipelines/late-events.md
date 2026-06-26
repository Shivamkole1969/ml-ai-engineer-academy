---
id: late-events
track: 12-pipelines
title: "Late-arriving events, watermarks, windowing"
badge: CORE
minutes: 8
prereqs: []
tags: [streaming, watermarks, windowing, spark-streaming, flink, real-time, pipelines]
xp: 45
hot2026: false
---

It's 11:52pm. You're on-call. The real-time dashboard for your food-delivery platform shows order counts crashing — a sudden 40% drop since 11:45. You page the backend team. They stare at their logs: requests are fine, payments are going through, riders are being assigned. Nothing is broken.

Then at 12:03am, the dashboard lurches back up. The "missing" orders appear, all timestamped 11:45–11:50. Mystery solved: the events weren't lost. They were *late*.

This is the central problem of stream processing. The clock on your pipeline and the clock on the events arriving into it are two very different things.

## Processing time vs. event time

Every event has two timestamps:

- **Event time** — when the thing *actually happened* (the user tapped "Order Now" at 11:47pm).
- **Processing time** — when your pipeline *saw* the event (it arrived at Kafka at 11:52pm because the user's app was briefly offline).

When you aggregate by processing time, you get fast, simple answers that are sometimes wrong. When you aggregate by event time, you get correct answers — but you have to decide *how long to wait* for stragglers.

Late events happen for real, boring reasons: mobile apps batch events when offline, IoT sensors have flaky connectivity, microservices retry with backoff, Kafka consumers rebalance. Assume lateness is normal, not exceptional.

:::why-prod
Aggregating by processing time silently under-counts metrics during network hiccups or mobile reconnects — exactly the moments your business cares most about. Watermark-based event-time windowing is what separates a dashboard your team trusts from one they ignore.
:::

## The three window shapes

Before we fix lateness, a quick map of how streams get bucketed in the first place.

:::table {title="Window types at a glance"}
| Window | Shape | Classic use case | Closes when |
|---|---|---|---|
| Tumbling | Fixed size, no overlap | Hourly order count per city | Timer fires every N minutes |
| Sliding | Fixed size, overlapping | Rolling 5-min avg latency, updated every 1 min | Timer fires on the slide interval |
| Session | Variable size, gap-based | Group a user's clicks until 30s of silence | Inactivity gap expires |
:::

Tumbling is the simplest and most common. Session windows are powerful for user-journey analysis but expensive to maintain in state. Sliding windows sit in between.

## Watermarks: your pipeline's promise to itself

A watermark is a statement: *"I'm confident no event with a timestamp earlier than T will ever arrive."*

Once the watermark passes time T, your system can safely close any window that ended at T and emit its result — without waiting forever. The formula is almost always:

```
watermark = max(event_time_seen_so_far) - allowed_lateness
```

If the latest event you've processed has event_time of 11:52pm, and you allow 5 minutes of lateness, your watermark sits at 11:47pm. Events timestamped before 11:47pm are either dropped or routed to a side output for special handling (dead-letter, correction jobs, etc.).

Here's what this looks like in PySpark Structured Streaming — the most common choice for Python-first ML teams in India:

```python {title="Tumbling window + watermark in PySpark Structured Streaming" run=false}
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, col, count

spark = SparkSession.builder.appName("late-events-demo").getOrCreate()

# In prod: swap "rate" for kafka or socket source
# spark-submit --packages org.apache.spark:spark-sql-kafka-*  (free, runs locally)
events = (
    spark.readStream
    .format("rate")          # generates rows with a 'timestamp' column
    .option("rowsPerSecond", 100)
    .load()
)

# Key line: withWatermark tells Spark how late is "too late"
result = (
    events
    .withWatermark("timestamp", "5 minutes")   # drop events > 5 min late
    .groupBy(
        window(col("timestamp"), "1 minute")   # tumbling 1-min windows
    )
    .agg(count("*").alias("event_count"))
)

query = (
    result.writeStream
    .outputMode("append")   # only emit a window AFTER its watermark passes
    .format("console")
    .start()
)
query.awaitTermination()
```

The critical detail: `outputMode("append")` only emits a window's result *after* the watermark has advanced past that window. You get correct, complete counts — just slightly delayed. Switch to `"update"` if you need partial results sooner (useful for dashboards, costs more state).

:::gotcha
Setting `allowed_lateness` too tight (e.g., 0 seconds) drops late events silently. Setting it too loose (e.g., 24 hours) holds huge amounts of in-memory state and delays every downstream consumer. Profile your actual p95 event latency from Kafka lag metrics before picking a number — 5–10 minutes covers most mobile reconnect scenarios without blowing up your state store.
:::

## What happens to truly late events?

Events that arrive *after* the watermark has moved past their window are called **dropped late data** in Spark. Apache Flink gives you more control: you can route them to a side output stream and apply correction logic (subtract old count, add corrected count). In practice, most teams either accept the small loss or run a nightly batch reconciliation job that replays Kafka from the raw log and overwrites the day's metrics. Both are valid trade-offs.

:::interview-line
"We use event-time windowing with a watermark tuned to our p95 Kafka consumer lag — that way we catch late mobile events without holding unbounded state."
:::

:::qa {q="What is a watermark in stream processing?"}
A watermark is a moving threshold in event time that represents the pipeline's confidence about completeness. It advances as new events arrive, and once it passes a window's end time, that window is closed and its result emitted. Events arriving after the watermark passes their window are considered late and are either dropped or handled separately.
:::

:::qa {q="Why use event time instead of processing time for ML feature pipelines?"}
ML features often describe user behaviour — sessions, click rates, order frequency. Those behaviours happened at a real moment in time, not when your Kafka consumer got around to reading them. Training on processing time creates a subtle label-leakage risk: your model sees patterns shifted by pipeline lag that won't exist the same way at inference time. Event time keeps training and serving on the same clock.
:::

:::drill {type="mcq" q="A user's mobile app was offline for 8 minutes and reconnects, sending 40 queued events. Your pipeline uses a 5-minute watermark and tumbling 1-minute windows. What happens to those events?"}
- [ ] All 40 events are processed normally into their original windows
- [ ] All 40 events are queued in Kafka until the watermark resets
- [x] Events older than 5 minutes relative to the current watermark are dropped (or sent to side output)
- [ ] The pipeline pauses until all late events are consumed
:::

:::drill {type="mcq" q="You're building a real-time feature: 'number of orders placed by this user in the last 10 minutes'. Which window type is the natural fit?"}
- [ ] Tumbling window (10-minute buckets)
- [x] Sliding window (10-minute window, sliding every minute)
- [ ] Session window (10-minute inactivity gap)
- [ ] No window needed — just filter by timestamp
:::

:::key-takeaway
Late events are normal in production. Event-time windowing with a watermark — set to your actual p95 latency, not zero — is how you get correct streaming aggregations without holding infinite state.
:::
