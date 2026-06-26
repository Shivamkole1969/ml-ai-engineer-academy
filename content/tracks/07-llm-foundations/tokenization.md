---
id: tokenization
track: 07-llm-foundations
title: "Tokenization — the layer everyone skips then debugs"
badge: HOT
minutes: 10
prereqs: []
tags: [tokenization, llm, context-window, cost, tiktoken, bpe]
xp: 60
hot2026: true
---

Your company just launched a customer-support chatbot. English beta went fine. On launch day you add Hindi support — and two things happen within the hour: context windows start filling up at 3× the rate of English, and your OpenAI bill looks like it has a rounding error. No code changed. No extra features. Just different text.

Welcome to your first real tokenization bug.

## What a token actually is

A token is not a word. It's not a character either. It's a **subword chunk** that the model's vocabulary recognises as a single unit.

Modern LLMs use an algorithm called **Byte-Pair Encoding (BPE)**. The short version: start with individual characters, then iteratively merge the pairs that appear most frequently in the training corpus. Do this enough times and you end up with a vocabulary of ~50,000–100,000 subwords. Common English words (`the`, `model`, `return`) become a single token. Rare or non-English text gets split into many small pieces.

A few things that will surprise you:

- `tokenization` → 2 tokens (`token` + `ization`)
- `Tokenization` (capital T) → also 2, but **different** tokens
- `2024-01-15` → 5+ tokens (each digit and dash is often separate)
- A single Hindi word like `टोकनीकरण` can be 6–10 tokens
- Python `f"..."` strings: the quote characters are their own tokens

:::why-prod
Every API call is billed by the token. Every context window is sized in tokens. If your prompt secretly expands to 3× the token count you estimated (because of Hindi, JSON, code, or numbers), you hit the limit faster, pay more, and your downstream logic breaks silently — often at 2 AM.
:::

## Counting tokens before you pay for them

Always count tokens **before** sending. The `tiktoken` library is the reference tool for OpenAI-family models; Hugging Face tokenizers work the same way for open-source models.

```python {title="Count tokens before the API call" run=false}
import tiktoken

# cl100k_base is used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

english = "The quick brown fox jumps over the lazy dog."
hindi   = "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूदती है।"

print(count_tokens(english))   # ~10 tokens
print(count_tokens(hindi))     # ~30–40 tokens — same sentence, 3x the cost

# For Mistral / LLaMA via Hugging Face:
# from transformers import AutoTokenizer
# tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
# print(len(tok.encode(text)))
```

:::widget {name="tokencost"}
:::

:::table {title="Token counts for the same concept across text types"}
| Text sample | Characters | Approx tokens | Why it matters |
|---|---|---|---|
| `"hello world"` | 11 | 2 | English common words: 1:1 |
| `"नमस्ते दुनिया"` (Hindi) | 14 | 8–10 | Non-Latin script → more fragments |
| `2024-01-15T08:30:00Z` | 21 | 9–11 | Dates/times fragment badly |
| `{"key": "val"}` | 14 | 6 | JSON punctuation is per-token |
| Python function (20 lines) | ~300 | ~100–130 | Code is denser than prose |
:::

## The leading-space trap and other gotchas

:::gotcha
A space before a word often produces a **different token** than the same word without a space. `" model"` and `"model"` are different token IDs. This matters when you're doing token-level logprob analysis or manual prompt surgery. Also: never assume 1 word = 1 token in any cost estimate. Assume 1 word ≈ 1.3–1.5 tokens for English, 2–4 tokens for Hindi/Tamil/other Indic scripts, and 3–5 tokens for structured data (JSON, CSV).
:::

:::war-story {title="The Hindi support launch that doubled the bill overnight"}
A Pune-based SaaS team built a 16k-context RAG chatbot. English support queries consumed about 1,200 tokens per turn including retrieval context. When they flipped the switch on Hindi, the same semantic content consumed 3,400 tokens per turn. Within 6 hours the product hit rate limits. The fix was a second retrieval path that aggressively compressed Hindi context and pre-counted tokens before stuffing the prompt. The root cause: nobody had run `count_tokens()` on the Hindi corpus during planning.
:::

## Practical rules for production

1. **Budget tokens, not words.** Store token counts in your DB alongside raw text for any content you'll stuff into prompts.
2. **Test your non-English inputs explicitly.** Count tokens for Hindi, Tamil, Kannada, or whatever your users actually type.
3. **Structured output bloats context.** If you're asking the model to return JSON, factor in the brackets, quotes, and commas — they all cost tokens.
4. **Use the model's native tokenizer.** Don't estimate for GPT-4 using a LLaMA tokenizer. They differ, sometimes significantly.

:::interview-line
"Tokens are the real unit of cost and context — I always count them before the API call, because the same information in Hindi or JSON can be 3× the tokens of plain English prose."
:::

:::qa {q="Why does the same sentence cost more tokens in Hindi than in English?"}
BPE vocabularies are built from training corpora. English dominates most LLM training data, so common English words earn single-token slots. Indic scripts have fewer high-frequency subword sequences in the vocabulary, so the tokenizer falls back to smaller fragments — sometimes individual Unicode bytes. More fragments = more tokens = higher cost and faster context consumption.
:::

:::qa {q="How would you prevent a prompt from exceeding the context limit at runtime?"}
Count tokens before constructing the final prompt using the model's tokenizer (e.g. tiktoken for OpenAI, AutoTokenizer for Hugging Face). Set a hard budget for each slot — system prompt, retrieved context, conversation history, output reservation — and truncate or summarise the lowest-priority slot if the total exceeds the limit. Never rely on character counts; always use actual token counts.
:::

:::drill {type="mcq" q="A Hindi sentence has 15 characters. Roughly how many tokens should you expect compared to a similar 15-character English sentence?"}
- [ ] About the same — tokenizers are language-agnostic
- [ ] Slightly fewer — Hindi is more compact
- [x] Roughly 2–4× more — Indic scripts fragment into smaller subword pieces
- [ ] Exactly 15 — one character equals one token
:::

:::drill {type="mcq" q="You need to count tokens for a prompt before sending it to gpt-4o. Which approach is most accurate?"}
- [ ] Count whitespace-separated words and multiply by 1.3
- [ ] Use len(text) // 4 as a rule of thumb
- [ ] Use the LLaMA tokenizer — all transformers share a vocabulary
- [x] Use tiktoken with the cl100k_base encoding, which matches gpt-4/gpt-4o
:::

:::key-takeaway
Tokens are the true unit of cost, context, and correctness for every LLM system. Count them with the model's actual tokenizer before the API call — and always test with the real scripts your users write in.
:::
