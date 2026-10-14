# Volume III — The Nervous System (Agents)

> Moving from "chatting" to "doing." This is where the abstractions
> get leaky.

## Parts

| Part | Title | Status | Folder |
|------|-------|--------|--------|
| 12 | Observe, Think, Act | Built N Runnable | [`part12_observe_think_act/`](./part12_observe_think_act/) |
| 13 | The Plan vs. The Reality — ReAct Patterns | Built N Runnable | [`part13_react_patterns/`](./part13_react_patterns/) |
| 14 | The Token Accountant — Working Memory | Built N Runnable | [`part14_working_memory/`](./part14_working_memory/) |
| 15 | The Socratic Agent — Reflexion | Built N Runnable | [`part15_reflexion/`](./part15_reflexion/) |

## How the parts hang together

Part 12 explains why the observe-think-act loop is the right shape for
an agent, and why a chatbot can never close that loop. Part 13 hands you
a working implementation with the production discipline most toy examples
skip — bounded steps, errors as observations, every step traced. Part 14
gives the agent a token accountant so it can't think its way into bankruptcy.
Part 15 teaches the agent to grade its own work and try again when it
falls short.

The order matters. An agent without Part 14's bounded memory will run
forever and overspend. An agent without Part 13's discipline will run
forever and accomplish nothing. An agent without Part 15 accepts its first
answer even when it's wrong.

## What's in the box

```
vol3_agents/
├── part12_observe_think_act/
│   ├── article.md              # The mental model — Observe, Think, Act
│   └── README.md
│
├── part13_react_patterns/
│   ├── react_loop.py           # ReActAgent: bounded steps, token budget,
│   │                           #   errors-as-observations, tracing
│   ├── article.md              # Deepak's Meesho return, ReAct explained
│   ├── README.md
│   └── examples/
│       └── delivery_tracker.py  # Meesho shipment tracking + return agent
│
├── part14_working_memory/
│   ├── working_memory.py       # WorkingMemory: bounded buffer, pinning,
│   │                           #   optional summarize callback
│   ├── article.md              # Vikram's Jio Fiber session that forgot him
│   ├── README.md
│   └── examples/
│       └── customer_support.py  # Multi-turn Jio Fiber support session
│
└── part15_reflexion/
    ├── reflexion.py            # ReflexionAgent: evaluate → reflect → retry
    ├── article.md              # Priya vs Ankur, UPSC method for agents
    ├── README.md
    └── examples/
        └── complaint_resolver.py  # Boult Audio complaints, warranty vs return
```

## Quick start

```bash
# Delivery tracking (ReAct loop):
python -m vol3_agents.part13_react_patterns.examples.delivery_tracker

# Multi-turn support with working memory:
python -m vol3_agents.part14_working_memory.examples.customer_support

# Complaint resolution with Reflexion:
python -m vol3_agents.part15_reflexion.examples.complaint_resolver
```

Each example makes real API calls to Anthropic. Cost per run is under ₹10.
