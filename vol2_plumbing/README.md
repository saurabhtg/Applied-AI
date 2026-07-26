# Volume II — The Plumbing (Tools & RAG)

> LLMs are blind without data and toothless without tools. Here is how
> to give them a wrench without them breaking the pipes.

## Parts

| Part | Title | Status | Folder |
|------|-------|--------|--------|
| 7 | Giving the Machine a Wrench — Function Calling | **Built** | [`part07_function_calling/`](./part07_function_calling/) |
| 8 | The Multitasking Trap — Parallel Tool Calls | **Built** | [`part08_parallel_tool_calls/`](./part08_parallel_tool_calls/)|
| 9 | RAG is a Search Problem | **Built** | [`part09_rag_retrieval/`](./part09_rag_retrieval/)|
| 10 | Measuring the Truth — RAGAS | **Built** | [`part10_ragas/`](./part10_ragas/)|
| 11 | Thanks for the (Semantic) Memory — Caching | **Built** | [`part11_semantic_cache/`](./part11_semantic_cache/)|

## How the parts hang together

Parts 7 and 8 are about tools — giving the model the ability to act on
the world. Parts 9-11 are about retrieval — giving the model the
context it needs to act *correctly*. The two halves are usually wired
together in production: you retrieve relevant policies, then call a
tool that respects them.

The retrieval lessons compound: Part 9 says "vectors alone aren't
enough, you need hybrid retrieval." Part 10 says "but how do you know
your hybrid retrieval is working — measure faithfulness and recall."
Part 11 says "the same query gets asked fifty different ways; cache it
semantically." Skipping any of these is how teams ship RAG systems
that look right and are actually wrong.
