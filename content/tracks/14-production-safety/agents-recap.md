---
id: agents-recap
track: 14-production-safety
title: "Agents recap as a production system (links to Track 11)"
badge: CORE
minutes: 7
prereqs: []
tags: [agents, production, reliability, observability, blast-radius, idempotency]
xp: 45
hot2026: false
---

It's 10:30 AM on a Monday. Your team shipped an agent over the weekend — it's supposed to book conference rooms based on a Slack message. You come in, open your laptop, and see 47 calendar invites from the CEO's account, all for "Team Sync" in Room 3C, all on the same day, all sent in the span of four minutes.

The agent hit a transient API error on step 3. It retried. The tool wasn't idempotent. Each retry booked a new room. Nobody had a kill switch.

This is the exact gap between "it works in the notebook" and "it works in production."

## What an agent actually is (the fast version)

Track 11 goes deep on agents — loop structure, frameworks, MCP, memory. This lesson doesn't repeat that. Here's the one-sentence recap:

> An agent is a loop: the LLM decides an action, a tool executes it, the result comes back, the LLM decides the next action — until the task is done.

That loop is what makes agents powerful. It's also what makes them dangerous in production. Every iteration is a network call, a side effect, and a potential failure point. Errors don't just fail — they compound. Step 4 is only wrong because step 2 was wrong, and by the time you notice, you've already taken 12 irreversible actions.

The shift this lesson is about: stop thinking of an agent as "a smart prompt" and start thinking of it as **a distributed system you're responsible for**.

:::why-prod
Agents turn a single user request into a chain of real-world actions — API calls, database writes, emails, bookings. A bug in a stateless API returns a bad response. A bug in an agent sends 47 calendar invites before your PagerDuty alert fires. The production bar is much higher.
:::

## Four questions to ask before you ship any agent

These questions apply whether you're building a customer support bot, a data pipeline agent, or a code review assistant.

**1. What is the blast radius?**
If this agent goes wrong — retries too aggressively, misunderstands the task, gets fed a malicious prompt — what's the worst it can do? Delete files? Send emails to customers? Make paid API calls in a loop? Map the blast radius first.

**2. Are the tools idempotent?**
Idempotent means: calling the tool twice with the same input has the same effect as calling it once. Reads are always idempotent. Writes usually aren't — unless you make them. If a tool books a room, sending the same booking request twice should either create one booking or return the existing one. This is your first line of defense against retry disasters.

**3. Can you observe what it's doing right now?**
A trace per step — what action the LLM chose, what the tool returned, how long it took — is not a nice-to-have. It's how you debug a four-step failure at 10:30 AM on a Monday.

**4. Can you stop it?**
Every agent that takes real-world actions needs a kill switch: a flag, a budget cap on tool calls, or a human-in-the-loop checkpoint before any destructive action. If the answer is "we'd have to redeploy," that's not a kill switch, that's a prayer.

:::table {title="Agent components and their production concerns"}
| Component | Role | Production concern |
|---|---|---|
| Orchestrator (LLM) | Chooses next action | Non-determinism, cost drift, prompt injection |
| Tools | Execute the action | Idempotency, auth, rate limits, blast radius |
| Memory / context | Carries state across steps | Context window limits, stale data, drift |
| Environment | State the agent acts on | Rollback difficulty, side effects |
| Loop controller | Decides when to stop | Infinite loops, retry storms, budget overrun |
:::

## The mental model shift

Here's the core reframe for this track.

A stateless API endpoint is bounded. One request in, one response out, no side effects, no memory. If it fails, the user retries and maybe gets an error page.

An agent is **stateful, sequential, and action-taking**. Each step narrows the space of "what can still go right" and widens the space of "what's already been done and can't be undone." This is why the distributed-systems discipline you'll see in the next lessons — queues, retries, idempotency, backpressure — isn't just optional polish. It's the foundation.

:::gotcha
The most common mistake: testing agents only on happy-path inputs. An agent that books rooms correctly every time in your test suite will still book 47 rooms on the day a downstream calendar API returns a 503 and your retry logic runs unchecked. Test adversarially: what happens on timeout? On partial failure? On a malformed tool response? These are the failure modes that matter in production.
:::

:::interview-line
"An agent isn't just a smart prompt — it's a distributed system. I treat it like one: map the blast radius, make tools idempotent, add per-step traces, and ship with a kill switch."
:::

:::qa {q="How is an agent different from a regular API call in production?"}
A regular API call is stateless and bounded — one input, one output, easy to retry safely. An agent runs a loop where each step depends on the last, and each step may take a real-world action. Failures compound: a wrong step 2 corrupts step 3, 4, and 5. This makes blast radius and observability critical in ways that don't apply to stateless services.
:::

:::qa {q="What does 'idempotency' mean for agent tools, and why does it matter?"}
A tool is idempotent if calling it twice with the same arguments produces the same outcome as calling it once — no duplicates, no double-charges, no extra bookings. It matters for agents because agents retry on failure. If a tool isn't idempotent and the agent retries, you get the 47-calendar-invite problem. Making tools idempotent is the cheapest blast-radius reduction you have.
:::

:::qa {q="What is the first guardrail you add to an agent that can send emails or modify data?"}
A hard cap on the number of tool calls per run — a simple counter that terminates the loop if it exceeds, say, 20 steps. This prevents runaway loops from causing unbounded damage. The second guardrail is a human-approval checkpoint before any irreversible action (sending, deleting, paying). Together these limit blast radius without needing to predict every failure mode in advance.
:::

:::drill {type="mcq" q="Your agent hits a transient 503 from a booking API and retries three times. Each retry creates a new booking. What property of the tool would have prevented this?"}
- [ ] The tool should have a shorter timeout
- [ ] The LLM should use a lower temperature
- [x] The tool should be idempotent — same request, same outcome, no duplicate bookings
- [ ] The agent should have more context in its system prompt
:::

:::drill {type="mcq" q="Which of these is the BEST early warning signal that an agent is going wrong during a live run?"}
- [ ] A user reports the final output is incorrect
- [ ] The LLM returns an unusually long response
- [x] A per-step trace shows the agent invoking the same tool with the same arguments three times in a row
- [ ] The agent's total token usage exceeds the daily average
:::

:::key-takeaway
An agent is a distributed system disguised as a prompt. Treat it as one: know the blast radius, make tools idempotent, trace every step, and always have a kill switch before you ship to production.
:::
