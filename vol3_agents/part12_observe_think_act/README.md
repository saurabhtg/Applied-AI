# Part 12 — Observe, Think, Act

> *The mental model. No code yet — that's Part 13.*

## What's here

| File | What it does |
|------|--------------|
| [`article.md`](./article.md) | The conceptual foundation for Volume III. Why the observe-think-act loop is the right shape for an agent, and why a chatbot can never get there. |

## The loop at a glance

```
         Goal arrives
              │
              ▼
         ┌─────────┐
    ┌───►│ OBSERVE │◄──── tool result / initial goal
    │    └─────────┘
    │         │
    │         ▼
    │    ┌─────────┐
    │    │  THINK  │──── "I have enough to answer" ──► Final Answer
    │    └─────────┘
    │         │ "I need more data"
    │         ▼
    │    ┌─────────┐
    │    │   ACT   │──── executes a tool, queries a system
    │    └─────────┘
    │         │
    └─────────┘  tool result comes back → OBSERVE
```

## What's next

- **Part 13:** A working ReAct implementation with bounded steps, token budget, and errors-as-observations.
- **Part 14:** Working memory so the loop doesn't eat the entire context window.
- **Part 15:** Reflexion — the agent that evaluates its own output and retries when it falls short.
