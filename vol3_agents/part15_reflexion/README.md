# Part 15 — The Socratic Agent (Reflexion)

> *The agent that grades its own answers and tries again when they fall short.*

## What's here

| File | What it does |
|------|--------------|
| [`reflexion.py`](./reflexion.py) | Core `ReflexionAgent` — wraps any `ReActAgent` with self-evaluation and retry. The evaluator scores answers 1–5; the reflector diagnoses what failed; the retry carries that diagnosis into the next run. |
| [`examples/complaint_resolver.py`](./examples/complaint_resolver.py) | D2C brand (Boult Audio) complaint resolution agent. Shows Reflexion catching a policy miss: agent refuses on expired return window, reflection points to unchecked warranty policy, second attempt gets it right. |
| [`diagrams/`](./diagrams/) | The article's three diagrams. Each one is an SVG source plus a 2x PNG ready to upload to Substack or LinkedIn. |

## The loop

```
      Goal
        │
        ▼
   ┌──────────┐
   │ ReAct    │──── answer ──────────────────────┐
   │ Agent    │                                  ▼
   └──────────┘                          ┌────────────────┐
        ▲                                │   Evaluate     │
        │                                │   (score 1–5)  │
        │                                └────────────────┘
        │                                        │
        │                               score ≥ threshold?
        │                                  No        Yes
        │                                  │          │
        │                          ┌───────────┐    Return answer
        │                          │  Reflect  │
        └──────────────────────────│  (why?)   │
            (retry with reflection)└───────────┘
```

## Use as a snippet

```python
from vol3_agents.part13_react_patterns.react_loop import ReActAgent, Tool
from vol3_agents.part15_reflexion.reflexion import ReflexionAgent

base = ReActAgent(tools=TOOLS, max_steps=10, max_tokens_total=30_000)

agent = ReflexionAgent(
    base_agent=base,
    evaluation_criteria="""
    1. Did the agent check all applicable policies?
    2. Is the resolution correct and supported by the policy?
    3. Was the resolution logged with process_resolution?
    """,
    max_attempts=3,
    min_score=4.0,
)

result = agent.run("Resolve complaint for order BA-55421 — defective earbuds.")
print(result.final_answer)
print(f"Attempts: {len(result.attempts)} | Succeeded: {result.succeeded}")
```

## What this implementation gets right

- **Specific evaluation criteria, not vague quality scores.** "Did the agent check both return and warranty policies?" produces a useful critique. "Was this good?" doesn't.
- **The grader sees evidence.** It gets the agent's tool trace, not just the final answer, and grades each criterion against it in a fresh context.
- **Structured verdict.** One met/not-met per criterion plus a score, validated against a JSON schema. An attempt passes only if the score clears `min_score` and every criterion is met. A grader that can't decide fails the attempt with the reason, never with an invented score.
- **Lessons accumulate.** Each retry carries the last three reflections, not just the latest one.
- **Reflection names what failed.** Not "be more thorough" — "the agent never called `check_warranty_policy` despite a defect claim."
- **Returns all attempts.** You can see the improvement arc, not just the final answer.
- **Returns best attempt if max_attempts exceeded.** Doesn't return the last attempt, which may be confused by accumulated context. Returns the highest-scored.

## When Reflexion earns its keep

Use it when tasks have multiple applicable pathways the agent might miss, correctness matters more than speed, and you can write explicit evaluation criteria.

Skip it for simple lookups, real-time flows where latency matters, or tasks where you can't write specific criteria.

## Run it

```bash
python -m vol3_agents.part15_reflexion.examples.complaint_resolver
```

Cost per run: under ₹10. Three complaints, up to 3 attempts each.
