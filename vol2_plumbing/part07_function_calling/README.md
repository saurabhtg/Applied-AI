# Part 7 — Giving the Machine a Wrench: Function Calling

> *The model knows a lot. Without tools, it can't do anything.*

---

A bot that can only talk about account details — but never look them up —
is a very well-spoken dead end. Function calling is how you give the model
the ability to act: fetch a transaction, query a database, call an API.

The core idea is a loop. The model says "I need this data." You fetch it.
You tell the model what you found. It either asks for more data or gives
you the final answer.

That loop is what this part is about.

---

## What's here

| File | What it does |
|------|--------------|
| [`tool_caller.py`](./tool_caller.py) | `ToolCaller` — the agentic loop. Sends messages, handles `tool_use` responses, executes tools, feeds results back, repeats until done. |

## Example

| File | Pattern | Story |
|------|---------|-------|
| [`examples/upi_assistant.py`](./examples/upi_assistant.py) | Single-turn and multi-tool function calling | A UPI customer service bot that can actually look up your transaction instead of asking you to "contact the branch" |

## Run

```bash
python -m vol2_plumbing.part07_function_calling.examples.upi_assistant
```

Makes real Anthropic API calls. Tools are mocked. Cost per run: under ₹2.

---

## The loop

The thing most tutorials skip over is that function calling isn't a single
request — it's a loop:

```
User question
     │
     ▼
[Model call 1]
  stop_reason = "tool_use"
  content: [tool_use: get_transaction_status(txn_id="TXN9834561")]
     │
     ▼
  execute get_transaction_status("TXN9834561")
  → {"status": "SUCCESS", "amount_inr": 1500, ...}
     │
     ▼
[Model call 2]  ← send tool result back
  stop_reason = "end_turn"
  content: [text: "Your transaction of ₹1,500 to Suresh Kumar was
            successful on 27 May at 2:23 PM."]
     │
     ▼
Done. Return final_text.
```

Two API calls for one user question. That's normal. A complex question
might take three or four rounds. The `max_rounds` parameter is your
circuit breaker.

## Three rules for tool calling

**1. All tool results go back in one user message.**

If the model asks for two tools in one response, you execute both and
return both results in a single `{"role": "user", "content": [...]}`.
Not two separate user messages. One.

**2. Load the old scores before running the new eval.**

Actually, that's a Part 6 rule. The function-calling rule is: never
start the next model call until you have results for every `tool_use`
block in the previous response.

**3. The model decides when it's done.**

Don't stop the loop just because the model got some data. Let it decide
when it has enough to answer. It will set `stop_reason = "end_turn"`.

---

## How `ToolDefinition` works

```python
ToolDefinition(
    name="get_transaction_status",
    description="Look up a UPI transaction by ID. Returns status, amount, recipient.",
    input_schema={
        "type": "object",
        "properties": {
            "txn_id": {"type": "string", "description": "UPI transaction ID"},
        },
        "required": ["txn_id"],
    },
    fn=get_transaction_status,   # your Python function
)
```

The `description` is what the model reads to decide whether to call
this tool. Write it like documentation for a junior engineer: clear,
specific, no jargon.

The `input_schema` is JSON Schema. The model uses it to figure out what
arguments to pass. Keep it precise — vague schemas produce bad inputs.

---

## What's next

- Part 8 ([`../part08_parallel_tool_calls/`](../part08_parallel_tool_calls/)): the model
  often asks for multiple tools at once. Run them in parallel — not one after the other.
- Part 9 ([`../part09_rag_retrieval/`](../part09_rag_retrieval/)): when you can't fit
  the knowledge into the prompt, retrieve it first and pass it as context.
