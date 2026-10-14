# Part 13 — The Plan vs. The Reality (ReAct)

> *Building ReAct patterns that don't crumble the moment an API returns
> a 404.*

## What's here

| File | What it does |
|------|--------------|
| [`react_loop.py`](./react_loop.py) | A ReAct (Reason + Act) loop on native `tool_use` blocks, with bounded rounds, a token budget, and errors-as-observations. |
| [`diagrams/`](./diagrams/) | The article's five diagrams. Each one is an SVG source plus a 2x PNG ready to upload to Substack or LinkedIn. |

## The loop

```
Thought:        model reasons about what to do next
Action:         model picks a tool and arguments
Observation:    tool runs, returns a result
    ↓
    (loop until final answer, max steps, or token budget hit)
```

## Use as a snippet

```python
from vol3_agents.part13_react_patterns.react_loop import ReActAgent, Tool

def lookup_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}

agent = ReActAgent(
    tools=[Tool(name="lookup_order", description="Look up an order by ID", fn=lookup_order)],
    max_steps=10,
    max_tokens_total=50_000,
)

result = agent.run("What's the status of order ORD-12345?")
print(result.answer)
```

## What this implementation gets right that toy examples don't

- **Native `tool_use` blocks, no regex.** The model emits typed tool calls
  with validated arguments. No `Thought:/Action:` parser to drift away from
  the prompt, no silent parse failure that turns into a wrong answer.
- **Parallel tool calls.** Independent lookups go out together, and every
  result comes back in a single user message. Splitting them teaches the
  model to stop batching.
- **Schemas derived from the function signature.** `Tool` needs a name, a
  description and a callable. The JSON schema comes from `inspect`, and is
  marked `strict` when every parameter maps cleanly to a JSON type.
- **Hard cap on rounds.** An agent will run forever if you let it.
  `max_steps=10` keeps the pathological case bounded.
- **Token budget tracking.** When you cross `max_tokens_total`, the agent
  stops, even mid-loop. No surprise ₹40 invoices.
- **Errors are observations, not exceptions.** A raising tool comes back as
  a `tool_result` with `is_error=True`. The model reads the failure and
  picks another route instead of the process dying.
- **Append-only history.** Thinking blocks are signed against everything
  before them, so the loop never edits an earlier turn. Trimming is left to
  server-side context editing.
- **Refusals fall back.** A safety-classifier decline is rerun on another
  model server-side (`fallbacks: "default"`), so a false positive doesn't
  end the run.
- **Every stop is explained.** `stopped_reason` is one of `final_answer`,
  `token_budget`, `max_steps`, `refusal`, `max_tokens`. Log it. Alert on it.
- **Every round and tool call emits a tracing span.** You can reconstruct
  what the agent did, in what order, with what arguments.

## What's coming in later parts

- Part 14: a token accountant for working memory so multi-turn agents
  don't blow their context window.
- Part 15: reflexion — the agent grades its own output and retries if
  it judges itself wrong.
- Part 17: the orchestrator pattern that routes between specialist
  agents instead of having one giant generalist.
