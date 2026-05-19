# Part 3 — The 'Yes-Man' Bug

> *Your model is lying to you just to be polite; here is how to stop it.*

Three failure modes that production teaches you the hard way. Each one
gets its own detector in this part.

## What's here

| File | Failure mode | What it does |
|------|--------------|--------------|
| [`sycophancy.py`](./sycophancy.py) | Model agrees with user when user is wrong | Scan a conversation for the assistant reversing without new info |
| [`drift.py`](./drift.py) | System-prompt rules fade in long conversations | Scan assistant turns past N for explicit rule violations |
| [`collapse.py`](./collapse.py) | Outputs become templated and repetitive | Statistical check on opening/closing repetition + n-gram diversity across a batch |

## Examples

| File | Pattern | Story |
|------|---------|-------|
| [`examples/medical_triage.py`](./examples/medical_triage.py) | Sycophancy regression suite | Telemedicine bot capitulating under "I had rajma chawal, it's probably gas" pushback |

## Run

```bash
python3 -m vol1_foundations.part03_sycophancy.examples.medical_triage
```

## When this matters

The three failure modes have wildly different production consequences:

- **Sycophancy** in a triage or compliance system can be the bug that
  kills a patient or signs off on fraud.
- **Drift** is how customer-service bots that promised they'd never
  authorize refunds end up authorizing refunds on turn 42.
- **Collapse** is how an e-commerce site's product descriptions all
  start with "Introducing the..." and get downranked by Google.

The detectors are heuristics — they'll have false positives. Tune the
thresholds to your tolerance. The point is to have *some* automatic
check, not to replace human review.
