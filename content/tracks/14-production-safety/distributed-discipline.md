---
id: distributed-discipline
track: 14-production-safety
title: "Distributed-systems discipline: queues, retries, idempotency, backpressure"
badge: HOT
minutes: 10
prereqs: []
tags: [distributed-systems, queues, retries, idempotency, backpressure, mlops, production]
xp: 60
hot2026: true
---

It's 2 am. You're on call. Your ML inference service just fell over because a downstream LLM API timed out. Your retry loop hammered the API 10 000 times in three seconds — making things worse. The on-call engineer at the API provider just got paged too. You've created a thundering herd, and tomorrow morning your manager wants a root-cause document.

This lesson is how you never do that again.

## The four ideas, in thirty seconds

Distributed systems break in predictable ways. Four patterns keep ML pipelines standing:

1. **Queues** — absorb traffic spikes. Instead of your API saying "no" during a burst, it says "I'll get to you." Work waits in a queue; workers drain it at their own pace.

2. **Retries** — upstream things fail transiently. Retry them — but *carefully*, or you amplify the problem.

3. **Idempotency** — if you retry, you might run the same operation twice. An idempotent operation is safe to repeat: the second run produces the same result, not a duplicate side-effect.

4. **Backpressure** — when the queue is full, stop accepting new work rather than silently dropping it or crashing.

These four are a single idea broken into parts. Skip any one and the others stop protecting you.

:::why-prod
LLM-backed services are I/O heavy and call multiple external APIs. Any single hop can fail or slow down. Without these four patterns, one flaky endpoint cascades into a full outage — and you get paged at 2 am from Pune at the worst possible time.
:::

## Retries done right

A naive retry — `for _ in range(3): call()` — sends all three requests the moment something goes wrong. If 1 000 clients do this simultaneously, the downstream service receives a wall of 3 000 requests at once. That is the **thundering herd**.

The fix is **exponential backoff with jitter**:

- Wait longer after each failure: 0.5 s → 1 s → 2 s → give up.
- Add *randomness* (jitter) so all clients don't retry at the exact same millisecond.

```python {title="Retry decorator with exponential backoff + full jitter" run=false}
import random, time
from functools import wraps

def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Retries with exponential backoff and full jitter.
    Run locally: decorate any function that raises TimeoutError and call it.
    pip install nothing — pure stdlib.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except (TimeoutError, ConnectionError) as exc:
                    if attempt == max_retries:
                        raise  # out of retries — let it propagate
                    cap = base_delay * (2 ** attempt)   # 1 s, 2 s, 4 s ...
                    sleep = random.uniform(0, cap)       # full jitter
                    print(f"attempt {attempt + 1} failed ({exc}); "
                          f"sleeping {sleep:.2f}s before retry")
                    time.sleep(sleep)
        return wrapper
    return decorator


@retry_with_backoff(max_retries=3, base_delay=0.5)
def call_embedding_api(text: str) -> list[float]:
    # swap in your real HTTP call here
    raise TimeoutError("upstream slow")  # simulates a transient failure
```

## Idempotency — the overlooked piece

Retrying is only safe if the operation is **idempotent**: running it twice is the same as running it once. Writing an embedding to a vector DB with `upsert` is idempotent. Triggering a payment, sending an email, or firing a webhook twice — not so much.

The standard fix is an **idempotency key**: a stable ID you generate from the request content. Send it to the upstream service; the server uses it to detect duplicates and return the cached response rather than executing the work again.

:::table {title="Idempotent vs. not — common ML pipeline operations"}
| Operation | Idempotent? | Fix when it isn't |
|---|---|---|
| Upsert embedding to Pinecone/pgvector | Yes | — |
| `INSERT` row to Postgres | No | `INSERT … ON CONFLICT DO NOTHING` |
| POST to payment / notification API | No | Send idempotency-key header |
| Enqueue a Celery / RQ task | No | Deduplicate by task ID before enqueue |
| Trigger a model training run | No | Check for existing run with same config hash |
:::

## Backpressure — learning to say "not right now"

When a traffic spike arrives, your queue fills. Three choices:

- **Reject with 429 Too Many Requests** — honest, lets callers back off. Best for synchronous HTTP APIs.
- **Block the producer** — natural in synchronous queue clients; good for internal batch pipelines.
- **Drop** — only acceptable for truly ephemeral data like low-priority telemetry.

The wrong move is silently accepting work you cannot process. The queue grows without limit, memory climbs, and everything eventually crashes. Backpressure turns a crash into a polite "try again in 10 seconds."

:::gotcha
Retrying at *every* layer is a trap. If your HTTP client retries, your queue retries, and your worker retries — a single transient failure multiplies into thousands of duplicate requests. Decide which layer owns the retry and disable retrying everywhere else. A common split: one retry at the call site with backoff, dead-letter queue (DLQ) at the worker for persistent failures.
:::

:::war-story {title="The retry storm that billed Rs 80k in overages by 4am"}
A Bengaluru fintech ran nightly batch inference — 50 000 records sent to an external LLM API. The API hiccupped for 30 seconds. Their job had a bare `while True: retry()` loop with no backoff and no ceiling. All 50k jobs retried simultaneously, hit rate limits, failed again, and retried harder. The loop spun for six hours instead of forty minutes, exhausted their monthly token quota, and triggered auto-billing overages of Rs 80k before anyone woke up. Fix: exponential backoff, max 5 retries, dead-letter queue for unrecoverable failures, and a cost alert at 50% of monthly budget.
:::

:::interview-line
"In any distributed ML pipeline I ask three questions first: is this operation idempotent, which layer owns the retry, and what happens when the queue is full — those three questions prevent most production fires."
:::

:::qa {q="What is the thundering herd problem and how do you prevent it in a retry loop?"}
When many clients retry a failed service at the same instant, their simultaneous requests hit the recovering service in a burst — often re-crashing it. The prevention is exponential backoff with jitter: each client waits a random amount drawn from an exponentially growing window. This spreads retries over time so the service can recover rather than absorb a second spike.
:::

:::qa {q="What is an idempotency key and when would you use one in an LLM pipeline?"}
An idempotency key is a stable, client-generated identifier attached to a request. The server stores the result keyed by it; if the same key arrives again it returns the cached result instead of re-executing the operation. You'd use one any time you retry a non-idempotent call — an LLM API call that triggers a charge, a webhook dispatch, or any write that must not happen twice.
:::

:::drill {type="mcq" q="You need to retry a failed POST to an external payment API. Which approach is safest?"}
- [ ] Retry immediately up to 10 times with a fixed 1-second delay
- [x] Retry with exponential backoff, full jitter, and an idempotency key in the request header
- [ ] Retry with exponential backoff but no jitter and no idempotency key
- [ ] Catch all exceptions, log them, and silently skip failures
:::

:::drill {type="mcq" q="Your async worker queue is at 95% capacity during a traffic spike. What is the correct backpressure response from your HTTP API?"}
- [ ] Accept all requests and let the queue overflow — workers will catch up eventually
- [ ] Log a warning and continue accepting to avoid dropping requests
- [x] Return HTTP 429 Too Many Requests so callers can back off and retry later
- [ ] Spawn one new worker thread per incoming request to drain the queue faster
:::

:::key-takeaway
Queues, retries, idempotency, and backpressure are a matched set — skip one and the others can't protect you. In LLM pipelines especially, always add jitter to retries and always ask "is it safe to run this twice?" before you retry anything.
:::
