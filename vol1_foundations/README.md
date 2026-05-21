# Volume I — The Mental Model (Foundations)

> You can't build a skyscraper on a swamp.

These parts are about understanding the non-deterministic chaos of the
silicon brain — what the model is actually doing, where it breaks, and
the discipline that separates a demo from a production system you can
sleep through the night with.

## Parts

| Part | Title | Status | Folder |
|------|-------|--------|--------|
| 1 | Stop Treating LLMs Like APIs | Article only | — |
| 2 | The Power of the Good Example | **Built** | [`part02_few_shot/`](./part02_few_shot/) |
| 3 | The 'Yes-Man' Bug — Sycophancy | **Built** | [`part03_sycophancy/`](./part03_sycophancy/) |
| 4 | Prompts are Production Code | **Built** | [`part04_prompts_as_code/`](./part04_prompts_as_code/) |
| 5 | The JSON Straitjacket | **Built**  | [`part05_structured_outputs/`](./part05_structured_outputs/) |
| 6 | The Architect's Safety Net — Regression Harness | Article-level; eval runner lives in Part 4 | — |

## How the parts hang together

Part 1 sets up the mental model: LLMs are samplers, not functions. Part 2
gives you the three techniques that take a system from "demo" to
"production-ish" (few-shot, self-consistency, prompt voting). Part 3
makes you fluent in the failure modes you'll spend the rest of your
career detecting (sycophancy, drift, collapse). Part 4 nails down the
engineering discipline — prompts in Git, eval sets in CI, version-
stamped every call — that makes the techniques in Parts 2-3 actually
hold up over time.

Read them in order. Each one builds on the previous.
