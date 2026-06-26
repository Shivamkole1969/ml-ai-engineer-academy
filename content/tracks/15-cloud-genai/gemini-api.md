---
id: gemini-api
track: 15-cloud-genai
title: "Gemini API & Gemini Enterprise: grounding, function calling, search"
badge: HOT
minutes: 10
prereqs: []
tags: [gemini, google, grounding, function-calling, genai, vertex-ai]
xp: 60
hot2026: true
---

It's Friday evening. Your manager demos the new internal chatbot to the VP. First question: "What is our Q3 revenue?" The bot answers confidently — and rattles off a number from last year's earnings call. Wrong quarter, wrong data, wrong impression. The room goes quiet.

You could have prevented this. That's exactly what **grounding** solves.

## Two doors into Gemini

Google gives you two ways in:

- **Google AI Studio + the Gemini API** — free tier, one API key, zero GCP setup. Perfect for prototyping and your portfolio.
- **Vertex AI (Gemini Enterprise)** — GCP project, IAM roles, audit logs, SLAs, higher quotas. This is where production systems live.

Same underlying models, different control planes. For interviews: know both, know *why* you'd graduate from one to the other.

:::why-prod
Free-tier Gemini lets you build real projects for ₹0 — crucial for a Pune portfolio. Vertex AI Gemini is what enterprises deploy for data residency, audit trails, and reliability. The Python SDK patterns transfer directly, so learning one makes the other trivial.
:::

:::table {title="AI Studio vs Vertex AI Gemini"}
| Dimension | Google AI Studio (free) | Vertex AI / Enterprise |
|---|---|---|
| Auth | API key | Service account / ADC |
| Rate limits | ~60 RPM free tier | Configurable, usage-billed |
| Grounding source | Google Search | Search + your own corpus |
| Audit logs / VPC | No | Yes |
| Context caching | Limited | Full support, big cost win |
| SLA | No | Yes |
:::

## Grounding: give the model a reality check

A model's weights are a snapshot frozen at training time. Ask it about today's RBI repo rate or your internal wiki — it will either say "I don't know" or (worse) invent something that sounds completely reasonable.

**Grounding** fixes this by letting the model retrieve fresh context *before* it generates. Two flavours:

1. **Google Search grounding** — the model auto-queries Google, reads top results, cites them. One line of config on the free tier.
2. **Vertex AI Search grounding** — point it at your own private corpus (uploaded docs, BigQuery tables). Enterprise RAG without you building a vector DB from scratch.

```python {title="Grounding with Google Search — free tier" run=false}
import google.generativeai as genai

# pip install google-generativeai
# Get your free key at aistudio.google.com

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",          # or gemini-1.5-pro
    tools="google_search_retrieval",         # ← one line enables grounding
)

response = model.generate_content(
    "What is the current Repo Rate set by RBI?"
)

print(response.text)   # grounded answer with cited sources

# Inspect which sources the model used
for chunk in response.candidates[0].grounding_metadata.grounding_chunks:
    print(chunk.web.uri)
```

:::gotcha
Enabling `google_search_retrieval` does NOT mean the model searches your private data. It only reads the public web. If the ground truth lives in your Confluence, Notion, or internal DB, you need Vertex AI Search grounding — or your own RAG pipeline. Mixing these up is a very common production mistake.
:::

## Function Calling: the model as an orchestrator

Sometimes you don't want an answer — you want the model to *decide what to call*. That is function calling (also called tool use).

Here is the flow:

1. You declare your functions as a schema (name, description, parameter types).
2. The user asks something. The model reads it, picks the right function, fills in the arguments as structured JSON, and returns a `FunctionCall` object.
3. **Your code** executes the actual function.
4. You send the result back. The model composes the final answer.

The model never runs your code. It just fills in the form. You stay in control of side effects.

```python {title="Function calling — Gemini picks the tool, you run it" run=false}
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# 1. Declare your tool schema
get_stock_price = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_stock_price",
            description="Returns the latest price for an NSE stock ticker.",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "ticker": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="NSE ticker symbol, e.g. INFY",
                    )
                },
                required=["ticker"],
            ),
        )
    ]
)

model = genai.GenerativeModel("gemini-1.5-flash", tools=[get_stock_price])
chat = model.start_chat()

response = chat.send_message("What is the current price of Infosys?")

# 2. Model returns a structured function call — not text
call = response.candidates[0].content.parts[0].function_call
print(call.name, dict(call.args))  # get_stock_price {'ticker': 'INFY'}

# 3. You call your actual API, then send the result back
result = {"price": 1523.40, "currency": "INR"}   # your real API call here

response2 = chat.send_message(
    genai.protos.Content(
        parts=[genai.protos.Part(
            function_response=genai.protos.FunctionResponse(
                name=call.name,
                response={"result": result},
            )
        )]
    )
)

print(response2.text)
# → "The current price of Infosys (INFY) is ₹1,523.40."
```

:::war-story {title="The model that booked the wrong flight"}
A travel startup wired Gemini function calling directly to their booking API with no confirmation step. The model, given "cheapest flight to Bangalore next week," called `book_flight()` immediately. A user who was just browsing ended up with a confirmed non-refundable ticket. The fix was splitting it into two tools: `preview_booking` (safe, always allowed) and `confirm_booking` (requires explicit user approval before the model can invoke it). Lesson: never expose irreversible actions as a single tool with no gate.
:::

## Gemini Enterprise extras worth knowing

On Vertex AI, Gemini gets a few more levers that matter in production:

- **System instructions at the API level** — set a persistent persona or role without stuffing it into every prompt payload.
- **Context caching** — cache a large document (say, your 400-page product manual) once and pay per query, not per token on that document every single call. Can cut costs by 60–75% on doc-heavy workflows.
- **Provisioned Throughput** — reserved capacity so prod latency doesn't spike when your app goes viral.
- **Data residency** — keep tokens inside a GCP region. Required for BFSI and healthcare customers in India.

For your portfolio: prototype on AI Studio. For the interview: be ready to say "in production I'd use Vertex AI for IAM, audit logs, and context caching."

:::interview-line
"Gemini's function calling turns the model into an orchestrator — it decides which tool to invoke and with what arguments, but my code does the actual execution and returns the result. That separation keeps the model away from direct access to side-effecting systems."
:::

:::qa {q="What is grounding in Gemini and when would you choose Google Search grounding over Vertex AI Search grounding?"}
Grounding lets the model retrieve live context before generating, preventing hallucinations on facts it wasn't trained on. Google Search grounding is the right choice for public, real-time information — news, stock prices, current events. You switch to Vertex AI Search grounding when the facts live in your private corpus: internal wikis, product docs, enterprise data. The public web has no visibility into those, so Search grounding would just hallucinate.
:::

:::qa {q="How is Gemini function calling different from simply prompting the model to output JSON?"}
With plain JSON prompting, the model guesses at structure and can hallucinate field names or invent plausible-sounding values. Function calling uses a declared schema — the model is constrained to produce valid arguments against that schema, and you get a structured `FunctionCall` object back, not a string you have to parse and validate yourself. It is the difference between a typed API contract and a gentlemen's agreement.
:::

:::drill {type="mcq" q="You send Gemini a message with function calling enabled and it returns a FunctionCall object instead of a text response. What should your code do next?"}
- [ ] Treat it as an error — the model failed to answer
- [ ] Re-send the same message with a clearer prompt
- [x] Execute the named function with the provided arguments, then send the result back to the model
- [ ] Enable grounding so the model can answer directly
:::

:::drill {type="mcq" q="A fintech startup builds an internal assistant on the free Gemini API. They want to ground answers in their private RBI compliance documents stored in Google Drive. What is the correct approach?"}
- [ ] Enable `google_search_retrieval` — it auto-indexes Drive files linked to the account
- [ ] Use Gemini 1.5 Pro; Pro tier can access private Google Drive
- [x] Move to Vertex AI and configure Vertex AI Search grounding pointed at an ingested copy of the documents
- [ ] Paste the document text into the system prompt on each request
:::

:::key-takeaway
Grounding stops hallucinations by retrieving live context before generation. Function calling turns Gemini into an orchestrator that decides *what* to call while your code does the actual execution. Together, they are the backbone of any production-grade Gemini integration.
:::
