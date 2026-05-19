# Part 2 — The Power of the Good Example

> *Why three hand-picked "golden" examples beat a 03-page system prompt
> every time.*

## What's here

| File | What it does |
|------|--------------|
| [`few_shot.py`](./few_shot.py) | Build a few-shot prompt from a list of input/output examples. Includes a diversity heuristic so you can spot collapse-prone example sets before shipping. |
| [`self_consistency.py`](./self_consistency.py) | Run the same prompt N times at temperature > 0, take the majority. The classic accuracy lift on hard cases. |
| [`prompt_voting.py`](./prompt_voting.py) | Run multiple *phrasings* of the same question. For defeating prompt-specific blind spots — the cases where same prompt re-run five times gives the same wrong answer five times. |

## Examples

| File | Pattern | Story |
|------|---------|-------|
| [`examples/upi_fraud_classifier.py`](./examples/upi_fraud_classifier.py) | Few-shot + lazy self-consistency | UPI fraud team cut false-block rate by a third |
| [`examples/contract_review.py`](./examples/contract_review.py) | Prompt voting | Beating the model's American-bias on Indian Section 27 non-compete law |

## Run

```bash
python3 -m vol1_foundations.part02_few_shot.examples.upi_fraud_classifier
python3 -m vol1_foundations.part02_few_shot.examples.contract_review
```

## When to use which

- **Few-shot**: any time you're explaining a format. Three good examples
  beat ten bullet points of instructions.
- **Self-consistency**: classification or short-answer where the model
  is *mostly* right and *occasionally* flips. Same prompt, N runs,
  majority wins.
- **Prompt voting**: the model is *consistently* wrong on a specific
  framing. Don't re-roll — re-phrase. Multiple structurally different
  prompts, all votes count.

If you can't tell which problem you have, run prompt voting first. The
per-phrasing breakdown will tell you whether you have noise (where
self-consistency is sufficient) or a blind spot (where you needed
prompt voting).
