---
id: sql-coding-round
track: t8-extended
title: "The SQL coding round"
badge: HOT
minutes: 9
prereqs: []
tags: [sql, interviews, window-functions, data-engineering, feature-engineering]
xp: 60
hot2026: true
---

You've aced the ML theory questions. The interviewer says "great, let's try a quick data problem" — and pastes a schema with three tables. Your job: write a query. This is the SQL coding round, and it trips up more ML candidates than any algorithm question. Not because SQL is hard, but because most ML engineers only use it for simple `SELECT *` exploration. Production SQL — the kind that impresses — looks very different.

## Why ML engineers get SQL questions

You might be wondering: "I'm applying for an ML role, not a data engineering role. Why SQL?"

Because almost every real ML project starts with data that lives in a database or warehouse. Feature engineering, cohort analysis, training-set construction, experiment evaluation — all SQL. At most companies, your first day involves writing queries to understand the data before you write a single line of Python.

The coding round tests whether you can do *that* work efficiently, not just train models.

:::why-prod
Feature pipelines in production almost always have a SQL layer. If you can write clean, correct SQL, you can debug data issues, create features at scale, and collaborate with data engineers — all of which make you dramatically more productive than an ML engineer who hands everything off and waits.
:::

## The four SQL patterns that appear over and over

Interviewers recycle the same underlying patterns. Once you see them, you can decompose any problem.

:::table {title="The SQL interview pattern map"}
| Pattern | What it does | Common question type |
|---|---|---|
| Window functions | Compute over a "window" of rows without collapsing them | Ranking, running totals, row-over-row deltas |
| CTE chains | Break a hairy query into named steps | Any multi-step logic |
| Self-join | Join a table to itself | Consecutive events, friend-of-friend |
| Conditional aggregation | `SUM(CASE WHEN … THEN 1 ELSE 0 END)` | Pivot-style breakdowns |
:::

Most SQL interview questions are one or two of these patterns combined. Recognise the pattern, apply the template, adjust column names.

## Window functions: the most tested skill

Window functions are the gap between "I know SQL" and "I'm good at SQL." They let you compute a value for each row while still seeing *all* the rows — no GROUP BY collapsing.

Here is the core syntax to burn into memory:

```sql {title="Window function cheat sheet" run=false}
-- Pattern: function() OVER (PARTITION BY ... ORDER BY ... ROWS/RANGE ...)

-- 1. Rank users by revenue inside each country
SELECT
    user_id,
    country,
    revenue,
    RANK() OVER (PARTITION BY country ORDER BY revenue DESC) AS country_rank
FROM orders;

-- 2. Day-over-day delta (LAG fetches the previous row's value)
SELECT
    event_date,
    daily_active_users,
    daily_active_users
        - LAG(daily_active_users, 1) OVER (ORDER BY event_date) AS dau_delta
FROM daily_metrics;

-- 3. Running total — useful for cumulative revenue / experiment spend
SELECT
    event_date,
    revenue,
    SUM(revenue) OVER (ORDER BY event_date
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM daily_revenue;

-- 4. Deduplicate: keep only the most recent record per user
WITH ranked AS (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY updated_at DESC) AS rn
    FROM user_events
)
SELECT * FROM ranked WHERE rn = 1;
```

Pay attention to the deduplication pattern (snippet 4). It appears in nearly every interview at companies with messy event logs.

## The CTE mindset: write queries like code

A Common Table Expression (`WITH cte AS (...)`) is SQL's version of a named variable. Instead of nesting six subqueries until nothing is readable, you build up the answer in named steps.

Think of it as a mini pipeline: each CTE is a transformation stage, just like a step in a Pandas chain or a scikit-learn pipeline. Interviewers *notice* when you use CTEs well — it signals you think in data transformations, not raw SQL syntax.

:::gotcha
Candidates often compute a window function and then try to filter on it in the same `WHERE` clause — `WHERE RANK() OVER (...) = 1`. SQL evaluates `WHERE` before window functions, so this always fails with a syntax error. Always wrap the windowed query in a CTE or subquery first, then filter in the outer query.
:::

## Reading the problem: the interview workflow

When the problem lands in front of you, do this in order:

1. Read the schema and say out loud what each table represents.
2. Ask for a sample row if none is given (shows you think about data, not just queries).
3. Identify the grain — what does one row mean after transformation?
4. Map to a pattern: ranking? Running total? Consecutive events? Pivot?
5. Write the innermost piece first, confirm it looks right, then wrap it.

This approach eliminates most bugs before you type.

:::war-story {title="The silent GROUP BY trap"}
A candidate at a product interview was asked: "Find users who made more than 3 purchases in a 30-day window." They wrote a clean GROUP BY with a HAVING clause — but grouped by calendar month, not a rolling 30-day window. The query ran without error and returned plausible-looking numbers. Nobody caught it until the interviewer asked them to trace through a user who made purchases on Jan 29th, Feb 1st, and Feb 3rd. The candidate's query missed that burst entirely because it split on month boundaries. The correct answer needed a self-join or a window frame, not GROUP BY MONTH. The bug was invisible without thinking through edge cases out loud.
:::

## Handling NULLs and edge cases

SQL NULLs behave differently from Python `None`. A `NULL` in a `SUM` is ignored. A `NULL` in a comparison (`NULL = NULL`) returns `NULL`, not `True`. Interviews often hide a NULL edge case to see if you notice.

Safe habits:
- Use `COALESCE(column, 0)` or `COALESCE(column, 'unknown')` for default values.
- Use `IS NULL` / `IS NOT NULL`, never `= NULL`.
- When joining, consider whether you need `LEFT JOIN` to keep unmatched rows.

:::interview-line
"I always start by identifying the grain and the output shape — once I know what one row of the result means, the query almost writes itself."
:::

:::qa {q="What is the difference between RANK(), DENSE_RANK(), and ROW_NUMBER()?"}
All three number rows within a partition. `ROW_NUMBER()` assigns a unique sequential number — no ties. `RANK()` gives tied rows the same number but skips the next rank (1, 1, 3). `DENSE_RANK()` gives tied rows the same number without skipping (1, 1, 2). For deduplication use `ROW_NUMBER()`; for leaderboard-style "top N per group" use `RANK()` or `DENSE_RANK()` depending on whether ties should both appear.
:::

:::qa {q="How would you find sessions from a raw event log that has only user_id and event_timestamp?"}
Define a session boundary — typically 30 minutes of inactivity. Use `LAG(event_timestamp) OVER (PARTITION BY user_id ORDER BY event_timestamp)` to get the previous event's time, then flag rows where the gap exceeds the threshold as session starts. Assign a session ID by doing a running `SUM()` of those flags. This is a classic window-function chain: lag → gap flag → cumulative sum as ID.
:::

:::drill {type="mcq" q="You want the top-3 revenue users per country. Which function handles ties correctly by including all tied users even if that means more than 3 per country?"}
- [ ] ROW_NUMBER()
- [ ] RANK()
- [x] DENSE_RANK() with WHERE dense_rank <= 3
- [ ] NTILE(3)
:::

:::drill {type="mcq" q="A query uses WHERE rank <= 3 after a window function in the same SELECT. What happens?"}
- [ ] It works — WHERE is evaluated last
- [ ] It returns an empty result set
- [x] It throws an error — window functions cannot be referenced in WHERE
- [ ] It silently ignores the filter and returns all rows
:::

:::key-takeaway
Master four patterns — window functions, CTE chains, self-joins, and conditional aggregation — and you can handle virtually every SQL interview question thrown at ML candidates. The deduplication CTE (`ROW_NUMBER() + WHERE rn = 1`) and the session-boundary lag pattern alone appear in the majority of real interviews.
:::
