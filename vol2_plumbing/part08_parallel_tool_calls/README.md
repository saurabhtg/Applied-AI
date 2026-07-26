# Part 8 — The Multitasking Trap: Parallel Tool Calls

> *The model asked for three things at once. You did them one at a time.*

---

Part 7 built the tool-calling loop. This part makes it faster by doing the
obvious thing: when the model requests multiple tools in a single response,
run them in parallel instead of waiting for each one to finish before
starting the next.

The model doesn't care. It just wants all the results back before it
continues. Serial or parallel is entirely up to you.

---

## What's here

| File | What it does |
|------|--------------|
| [`parallel_runner.py`](./parallel_runner.py) | `ParallelToolRunner` — same loop as `ToolCaller`, but uses `ThreadPoolExecutor` to fire multiple tool calls concurrently. Reports timing savings per batch. |

## Example

| File | Pattern | Story |
|------|---------|-------|
| [`examples/loan_verifier.py`](./examples/loan_verifier.py) | Three independent verifications (CIBIL, ITR income, existing EMIs) run in parallel for each loan applicant | The serial version takes ~1,050ms. The parallel version takes ~420ms — the slowest single check. |

## Run

```bash
python -m vol2_plumbing.part08_parallel_tool_calls.examples.loan_verifier
```

Makes real Anthropic API calls. Tool implementations are mocked with
realistic latency (`time.sleep`). Cost per run: under ₹3.

---

## Serial vs parallel

When the model returns two or more `tool_use` blocks in one response,
the serial approach looks like this:

```
[Model response: 3 tool_use blocks]
        │
        ▼
  CIBIL check ── 350ms ──► result 1
        │
        ▼
  ITR check ─── 420ms ──► result 2
        │
        ▼
  EMI check ─── 280ms ──► result 3
        │
        ▼
  [send all 3 results]
  Total wait: 1,050ms
```

The parallel approach:

```
[Model response: 3 tool_use blocks]
        │
        ├── CIBIL check ── 350ms ──┐
        │                          │
        ├── ITR check ──── 420ms ──┤ (all running at once)
        │                          │
        └── EMI check ──── 280ms ──┘
                                   │
                      [send all 3 results]
                      Total wait: 420ms (the slowest one)
```

The savings compound. For a pipeline that hits five APIs, serial is
~5× slower than parallel. With fast APIs (< 50ms each), the difference
is negligible. With external APIs that take 300–500ms each, it matters.

---

## The rule that trips people up

The Anthropic API requires that all `tool_result` messages for a given
assistant turn go back in a SINGLE user message:

```python
# CORRECT: one user message with all results
messages.append({
    "role": "user",
    "content": [
        {"type": "tool_result", "tool_use_id": "id_1", "content": "..."},
        {"type": "tool_result", "tool_use_id": "id_2", "content": "..."},
        {"type": "tool_result", "tool_use_id": "id_3", "content": "..."},
    ]
})

# WRONG: three separate user messages
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id_1", ...}]})
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id_2", ...}]})
messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "id_3", ...}]})
```

The parallel execution doesn't change this. You still send one message.
You just fill that message with results that were computed concurrently.

---

## When parallel doesn't help

- **Only one tool requested**: no batch, no speedup. `ParallelToolRunner`
  handles this inline, no thread overhead.
- **Tools with dependencies**: if tool B needs the output of tool A, you
  can't run them in parallel. The model usually knows this and won't ask
  for them in the same response.
- **CPU-bound tools in CPython**: `ThreadPoolExecutor` doesn't help with
  CPU work due to the GIL. For CPU-heavy tools, use `ProcessPoolExecutor`.
  For I/O-bound tools (API calls, DB queries), threads are fine.

---

## What's next

- Part 9 ([`../part09_rag_retrieval/`](../part09_rag_retrieval/)): your
  tools often need to retrieve documents to answer correctly. That's the
  retrieval problem, and it's deeper than it looks.
