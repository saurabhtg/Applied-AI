# Part 14 — The Token Accountant (Working Memory)

> *How to stop an agent from thinking its way into bankruptcy.*

## What's here

| File | What it does |
|------|--------------|
| [`working_memory.py`](./working_memory.py) | A bounded, token-aware conversation buffer. When the budget is crossed it sweeps the oldest unpinned entries out in one go (dropped, or summarized via a callback), down to a low-water mark. Token estimates calibrate themselves from `usage.input_tokens`. |
| [`examples/customer_support.py`](./examples/customer_support.py) | A ten-turn Jio Fiber support session on a deliberately tiny budget, so you can watch a sweep happen. |
| [`diagrams/`](./diagrams/) | The article's four diagrams. Each one is an SVG source plus a 2x PNG ready to upload to Substack or LinkedIn. |

## Use as a snippet

```python
from vol3_agents.part14_working_memory.working_memory import WorkingMemory

mem = WorkingMemory(
    max_tokens=64_000,       # your working budget, not the model's window
    reserve_tokens=16_000,   # headroom for the next reply, thinking included
    compact_to=0.5,          # sweep down to half the budget
)

# Pin the system message so it never gets swept.
mem.add("system", system_prompt, pinned=True)
mem.add("user", user_message)

response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=4_000,
    system=mem.system(),
    messages=mem.render(),
)
mem.observe_usage(response.usage.input_tokens)   # calibrate before adding the reply
mem.add("assistant", reply_text)
```

## Why this matters

A 1M-token window doesn't mean you should fill it. Models get worse as
context grows, long before the window is full, and every turn re-reads
the whole conversation, so cost grows with the square of its length.

The working memory module makes the policy explicit:
- **A working budget, well below the window.** When it's crossed, the
  oldest unpinned entries go in one sweep (dropped, or summarized via a
  callback) down to a low-water mark.
- **Sweep rarely, in bulk.** A rolling window changes the start of the
  prompt every turn and misses the prompt cache every turn. A sweep
  misses once, then the history is append-only until the next one.
- **Pinned entries are sticky.** The system prompt and the facts you pin
  survive every sweep.
- **Real token counts.** Not tiktoken (OpenAI's tokenizer undercounts
  Claude). Estimates are calibrated from `usage.input_tokens`; use
  `client.messages.count_tokens()` when you need an exact number up front.
- **Reserve tokens for the response.** Thinking counts toward
  `max_tokens`, so leave room for it.

## Production wiring

For long-running customer-service or troubleshooting sessions:
1. Put rules in the pinned system prompt. Summaries drop rules first.
2. Pin facts (account number, plan, prior tickets), not whole turns.
3. Set a `summarize_fn` that says what must survive: IDs, dates, promises.
   Or use server-side compaction (`compact_20260112`, with `instructions`).
4. Monitor `mem.available_tokens`, `mem.compactions` and
   `cache_read_input_tokens` in your tracing.
5. Anything that should outlive the session goes in long-term memory
   (the memory tool, or Managed Agents memory stores), never secrets.
