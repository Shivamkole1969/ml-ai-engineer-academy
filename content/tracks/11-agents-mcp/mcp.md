---
id: mcp
track: 11-agents-mcp
title: "Model Context Protocol (MCP): Clients, Servers, Tools & Resources"
badge: HOT
minutes: 10
prereqs: []
tags: [mcp, agents, tools, protocols, production, anthropic]
xp: 60
hot2026: true
---

It's your first week at a Pune-based AI startup. Your agent needs to search Jira tickets, pull files from S3, run a SQL query on the analytics DB, and ping an internal REST API. Four tools. Four integration patterns your predecessor invented from scratch. Four auth flows to debug at 1am.

Six months later, the CTO says: "Let's try switching to Gemini." You smile politely. Inside, you're quietly spiralling — every integration is Claude-specific. You'd have to rewrite all four from scratch.

**MCP exists so that never happens to you.**

## What Is MCP?

Model Context Protocol (MCP) is an open standard — published by Anthropic in late 2024 — that defines a single, consistent language for AI models to talk to external tools and data sources.

Think USB-C for AI. Before USB-C, every laptop had different ports. Before MCP, every agent framework had its own bespoke tool format. MCP is the universal socket.

Three roles make it work:

**Client** — the AI application. Claude Desktop, Cursor, your custom LangGraph agent. The client sends requests: "call this tool", "give me this resource". The client holds the LLM; it drives the conversation.

**Server** — a lightweight process (local or remote) that exposes capabilities. One server might wrap your Jira API. Another wraps a local filesystem. A third talks to your Postgres DB. Servers don't know or care which client calls them.

**Protocol** — JSON-RPC 2.0 messages over a transport (stdio for local processes, HTTP + Server-Sent Events for remote). Clients and servers stay decoupled; they only share the message format.

Within a server, you can expose three kinds of things:

- **Tools** — callable functions the LLM actively invokes (`search_jira`, `run_sql`, `send_slack_message`). The model decides *when* to call them based on context.
- **Resources** — readable data blobs the client can fetch and inject into context (a file, a DB row, an API response). Think "context you pull" rather than "actions you take".
- **Prompts** — reusable, parameterised prompt templates a server can offer. Less common day-to-day, but useful for encoding domain-specific instructions once and sharing them across agents.

:::why-prod
Before MCP, swapping your LLM or your agent framework meant rewriting every tool integration. With MCP, your tool servers are framework-agnostic — build once, plug into Claude, Cursor, or any future client without touching the server code. That's the kind of leverage that matters when production is already running.
:::

## How a Tool Call Actually Flows

Here's the lifecycle when an agent calls an MCP tool:

1. Client (your agent) sends a `tools/call` JSON-RPC request to the server.
2. Server executes the function, returns a `CallToolResult` — either a text response, an image, or structured data.
3. Client receives the result, appends it to the conversation context, and hands it back to the LLM.
4. LLM decides: call another tool, or produce a final answer.

The server never talks to the LLM directly. It just receives a call, does work, returns a result. Clean and testable in isolation.

:::table {title="MCP primitives at a glance"}
| Primitive | Who triggers it | What it returns | Typical use |
|---|---|---|---|
| Tool | LLM (via client) | Text / data / error | Search, write, compute |
| Resource | Client or LLM | File / blob / rows | Load context before reasoning |
| Prompt | User / client UI | Pre-filled messages | Standardise domain interactions |
:::

## Writing a Minimal MCP Server

The official Python SDK makes this straightforward. Here's a server that exposes one tool — a SQL runner against a local DuckDB file. Zero cloud cost to try this yourself.

```python {title="minimal_mcp_server.py" run=false}
# pip install mcp duckdb
# Run locally: python minimal_mcp_server.py
# Then connect from any MCP client (Claude Desktop, your agent)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import duckdb, json

app = Server("analytics-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="run_sql",
            description="Run a read-only SQL query on the analytics DuckDB.",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "run_sql":
        raise ValueError(f"Unknown tool: {name}")

    # Safety: only allow SELECT
    query = arguments["query"].strip()
    if not query.upper().startswith("SELECT"):
        return [TextContent(type="text", text="Error: only SELECT queries are allowed.")]

    con = duckdb.connect("analytics.duckdb", read_only=True)
    rows = con.execute(query).fetchall()
    cols = [d[0] for d in con.description]
    result = [dict(zip(cols, row)) for row in rows]
    return [TextContent(type="text", text=json.dumps(result, indent=2))]

if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(app))
```

The LLM sees this server's `run_sql` tool in its context. It decides on its own when a question warrants a SQL lookup. You never hardcoded that logic in the agent.

:::gotcha
MCP servers are processes — they can crash, time out, or hang. If your client doesn't enforce a tool-call timeout and the server stalls, your agent hangs indefinitely. Always set a timeout on the client side (most SDKs have `timeout_seconds`) and return structured error responses from the server so the LLM can recover gracefully instead of retrying forever.
:::

:::war-story {title="The phantom Jira tool"}
A team built an MCP server wrapping Jira. It worked perfectly in local tests. In production, the server started via stdio but the parent process had a different working directory, so relative paths to a config file silently failed. The server started — but `list_tools` returned an empty list. The LLM kept saying "I don't have access to Jira" with no error surfaced anywhere. Three hours of debugging later: one absolute path fixed it. MCP servers need the same environment hygiene you'd give any production microservice.
:::

:::interview-line
"MCP decouples what an agent *can* do from how the LLM is implemented — I write the tool server once, and any MCP-compliant client can call it without knowing anything about the underlying model."
:::

:::qa {q="What is the difference between an MCP Tool and an MCP Resource?"}
A Tool is an action the LLM actively invokes at runtime — it runs code, calls APIs, modifies state. A Resource is readable data the client fetches and injects into context — more like a file the model can consult. Tools are dynamic and side-effectful; Resources are passive and read-only.
:::

:::qa {q="Why would you use MCP instead of just implementing tool calling directly in your agent?"}
Direct tool implementations are framework-specific — tied to LangChain, LlamaIndex, or your own glue code. MCP servers are framework and model agnostic. You write the server once, and any compliant client — Claude Desktop, a LangGraph agent, or a future framework you haven't chosen yet — can use it without modification. This matters in production where stacks evolve.
:::

:::qa {q="How does an MCP server tell the client what tools are available?"}
The client sends a `tools/list` JSON-RPC request to the server at startup (or on demand). The server responds with a list of Tool objects — each with a name, description, and JSON Schema for its input parameters. The client then injects those tool definitions into the LLM's context so the model knows what it can call.
:::

:::drill {type="mcq" q="Your MCP server's `list_tools` returns an empty list in production but works locally. What is the most likely first thing to check?"}
- [ ] The LLM's temperature is set too high, causing it to skip tool calls
- [ ] The MCP protocol version is mismatched between client and server
- [x] The server process is starting but failing silently during initialisation (missing env vars, wrong working directory, config file not found)
- [ ] Resources must be listed before tools in the server registration order
:::

:::drill {type="mcq" q="Which MCP primitive would you use to inject the last 50 rows of an error log into the LLM's context before it starts reasoning about an incident?"}
- [ ] Tool — call a `read_log` function and let the LLM trigger it
- [x] Resource — expose the log as a readable blob the client fetches upfront
- [ ] Prompt — encode the log as a reusable prompt template
- [ ] None — just hardcode it in the system prompt
:::

:::key-takeaway
MCP is the universal connector between LLMs and the outside world: clients (agents/apps) call servers (your tool wrappers) using a shared JSON-RPC protocol, and your tool servers stay completely reusable across models and frameworks.
:::
