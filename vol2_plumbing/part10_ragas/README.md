# Part 10 — Measuring the Truth (RAGAS)

> *Moving beyond "it looks right to me" toward mathematically provable
> retrieval quality.*

## What's here

| File | What it does |
|------|--------------|
| [`ragas_runner.py`](./ragas_runner.py) | Compute the four core RAGAS-style metrics for a (question, contexts, answer, ground-truth) tuple. |

## The four metrics

- **Faithfulness** — did the answer stay grounded in the retrieved
  context, or did the model hallucinate?
- **Answer relevance** — did the answer actually address the question,
  or did the model go off on a tangent?
- **Context precision** — how much of the retrieved context was
  useful? (Tells you when your retriever returns too much noise.)
- **Context recall** — did retrieval find the information the answer
  needed? (Requires a ground-truth answer to compute.)

Each metric is computed by asking a strong judge model a structured
question — "for each claim in this answer, is it supported by the
context?" — and parsing the verdict. The implementation is in code you
can read and modify.

## Use as a snippet

```python
from vol2_plumbing.part10_ragas.ragas_runner import compute_ragas_metrics

metrics = compute_ragas_metrics(
    question="What is the threshold for filing an STR?",
    answer="An STR must be filed with FIU-IND within 7 days of confirming suspicion.",
    contexts=["...passages retrieved by your RAG pipeline..."],
    ground_truth="STRs are filed with FIU-IND within seven days of confirmation.",
)
print(metrics.as_dict())
```

## When to run

- **On every retrieval pipeline change.** Tweaking your alpha,
  swapping embedding models, changing chunk size — all of these are
  changes that need RAGAS scores before and after.
- **On a held-out eval set.** Roughly 30-50 (question, ground-truth)
  pairs is enough to detect meaningful regressions. More is better but
  costs more to run.

The official RAGAS library has more metrics and a turn-key
implementation. This module ships the four that matter most, in code
you can fork.
