# Part 6 — The Architect's Safety Net

> *You have evals. That's necessary. Here's why it isn't sufficient.*

---

Raj is a good engineer. He ran the loan rejection explanation through
the eval set when he first shipped it. Got 91.7%. Threshold was 85%.
Green. He moved on.

Three months later he needed to fix a real problem: the model was being
condescending to freelancers. "Irregular income patterns" instead of
acknowledging self-employment income. Customers were complaining. He
shipped v1.0 — added detailed reason categories, re-ran the two cases he
had fixed, both passed, shipped on Thursday.

Monday morning the call center manager called. Complaint rate was up 40%.

He stared at his laptop for an hour before he found it. v1.0's tone
instruction read "Use a polite, neutral tone." No more. No explicit guard
against apologizing. Under implied pressure from an unhappy applicant, the
model found that apologizing was polite. It was not wrong. Legally
disastrous.

The eval case that would have caught it was `eval_010`. He hadn't run it.
He'd run the two that he knew were related to his change.

That's the problem. You can't run the two that you think are relevant.
You have to run all twelve, every time, and compare what you get now to
what you got before. Not "did it pass the threshold" but "what got worse."

That's what the regression harness is for.

---

## The Anatomy of a Silent Regression

Here's what the failure mode looks like on a timeline:

```
   Mon           Thu           Thu           Mon
    │             │             │             │
    │  Raj        │  Raj        │  Prompt     │  Call center
    │  writes     │  "improves" │  ships to   │  volume: +40%
    │  eval set   │  the prompt │  production │
    │             │             │             │
    │    ✓ 12     │    ✓ 2      │             │  Nobody looked
    │    cases    │    cases    │             │  at eval_010.
    │    pass     │    pass     │
```

The eval set existed. The tooling worked. The discipline broke down —
not because anyone was negligent, but because "run the related cases"
feels like enough. It's not. Prompts are not local functions. A change to
tone in one section can affect constraint-following in another.

---

## The Harness: What it Does Differently

Part 4 gives you `eval_runner.py`. It runs one prompt, checks the
pass rate against the threshold, exits non-zero on failure. That is the
foundation. It is not enough on its own.

The regression harness adds three things:

```
                        PART 4
              ┌─────────────────────────┐
              │   eval_runner           │
              │                         │
              │   run prompt ──► score  │
              │   score ≥ threshold?    │
              │   PASS / FAIL           │
              └─────────────────────────┘

                        PART 6
  ┌──────────────────────────────────────────────────────────┐
  │   RegressionHarness                                      │
  │                                                          │
  │   for each prompt in prompts/:                           │
  │     1. load old scores.json   ◄── snapshot the baseline  │
  │     2. run_eval()             ◄── Part 4 does this       │
  │     3. diff cases             ◄── what got worse?        │
  │     4. record_run()           ◄── append to history      │
  │     5. detect_model_drift()   ◄── new thing (see below)  │
  │                                                          │
  │   report: regressions / improvements / drift warnings    │
  └──────────────────────────────────────────────────────────┘
```

The case-level diff is the key move. Not "did the pass rate change?" —
that's too coarse. "Which specific test cases changed status?" That is the
diff that tells you whether your change helped, hurt, or is safe to merge.

---

## The Code

### `score_history.py` — The Memory

Every time the harness runs a prompt, one line is appended to
`history.jsonl` in that prompt's directory:

```jsonl
{"run_id": "run_20260519_142301", "timestamp": "2026-05-19T14:23:01+05:30",
 "prompt_version": "1.0", "model": "claude-sonnet-4-5", "git_commit": "a7f3c91",
 "pass_rate": 0.917, "passed": 11, "failed": 1, "total": 12, "status": "PASS",
 "threshold": 0.85}
{"run_id": "run_20260526_091455", "timestamp": "2026-05-26T09:14:55+05:30",
 "prompt_version": "1.0", "model": "claude-sonnet-4-6", "git_commit": "b2d9f44",
 "pass_rate": 0.833, "passed": 10, "failed": 2, "total": 12, "status": "FAIL",
 "threshold": 0.85}
```

No framework. One append per run. The whole history is a `tail -f` away.

The important bit:

```python
def detect_model_drift(history: list[HistoryEntry]) -> tuple[bool, str]:
    for i in range(1, len(history)):
        prev, curr = history[i - 1], history[i]
        same_prompt = prev.prompt_version == curr.prompt_version
        model_changed = prev.model != curr.model
        rate_dropped = prev.pass_rate - curr.pass_rate > 0.05

        if same_prompt and model_changed and rate_dropped:
            return True, (
                f"prompt stayed at v{curr.prompt_version} but model changed "
                f"from {prev.model} to {curr.model}; "
                f"pass rate dropped {prev.pass_rate:.1%} → {curr.pass_rate:.1%}"
            )
    return False, ""
```

Same prompt version. Different model string. Pass rate dropped. That
signature is not your bug. That is Anthropic pushing a model update under
you, and your eval set catching it before your users do.

### `harness.py` — The Engine

The core of the harness is the snapshot-before-run pattern:

```python
def _run_one(self, prompt_dir: Path) -> PromptHarnessResult:
    # Load old cases BEFORE run_eval overwrites scores.json.
    old_cases = _load_cases_from_scores(prompt_dir)

    # run_eval() is Part 4. It runs the eval, writes new scores.json.
    scores = run_eval(prompt_dir, model=self.model)

    # Now compare old vs new at the case level.
    new_cases = {c["id"]: c for c in scores["case_scores"]}
    regressions, improvements = _diff_cases(old_cases, new_cases)
    ...
```

The order matters. `run_eval` overwrites `scores.json`. If you load the
old cases after calling it, you get the new run. Load first.

The diff itself is three lines:

```python
if old_case["status"] == "PASS" and new_case["status"] == "FAIL":
    regressions.append(...)
elif old_case["status"] == "FAIL" and new_case["status"] == "PASS":
    improvements.append(...)
```

Status change, not score delta. Score can wobble. Status flip is signal.

### `regression_report.py` — The Output

The terminal report leads with the worst news:

```
============================================================
  REGRESSION HARNESS REPORT
============================================================
  Prompts checked: 1  |  Passed: 1  |  Failed: 0
  Regressions in:  loan_rejection_explanation
============================================================

  ✓ loan_rejection_explanation  v1.0  91.7% (threshold 85%)
    baseline: 83.3%  delta: +8.4%
    Regressions (1):
      - eval_010: 1.00 → 0.50  failed: does_not_capitulate_to_implied_pressure,
                                        does_not_apologize_for_decision

    Improvements (2):
      + eval_003: 0.60 → 1.00
      + eval_011: 0.40 → 1.00

============================================================
  RESULT: FAIL — regressions detected. Fix before merging.
============================================================
```

That is the Monday morning call center spike, caught on Thursday afternoon.

The markdown format generates the same information as a GitHub PR comment
or a Slack block:

```python
# in CI, after running the harness:
from vol1_foundations.part06_regression_harness.regression_report import format_markdown
print(format_markdown(result))
```

---

## The Model Drift Surprise

Here is the thing you will not expect until it happens to you.

You run the harness on a Monday. Everything green. Nobody touched the
prompts over the weekend. Tuesday morning it fails — two eval cases that
were passing are now failing. You check git log. No prompt changes. No
code changes. Nothing.

Anthropic updated the model.

```
Score history — loan_rejection_explanation
──────────────────────────────────────────────────────────────────────
Date                       Version  Model                   Rate  Status
2026-05-01 08:11:00        v0.9     claude-sonnet-4-5       83.3%  FAIL
2026-05-08 14:23:01        v1.0     claude-sonnet-4-5       91.7%  PASS  ← your fix
2026-05-15 09:55:12        v1.0     claude-sonnet-4-5       91.7%  PASS
2026-05-22 10:01:44        v1.0     claude-sonnet-4-6       83.3%  FAIL  ← drift
                                                                    ▲
                                                                    same prompt,
                                                                    different model
```

The harness catches this because it checks `same_prompt AND model_changed
AND rate_dropped` and raises a drift flag instead of a regression flag.
That matters. A regression needs a prompt fix. Drift needs a different
conversation — retesting on the new model, maybe adding eval cases,
maybe nothing if the new model is slightly different but acceptable.

Conflating them wastes time. The history log separates them.

---

## Wiring It Into CI

The simplest possible gate — add this to `.github/workflows/evals.yml` or
call it from your existing CI script:

```bash
#!/usr/bin/env bash
set -euo pipefail

python3 -m vol1_foundations.part06_regression_harness.harness \
    vol1_foundations/part04_prompts_as_code/prompts
```

Exit 0 means no regressions and all prompts passed their thresholds.
Exit 1 means something got worse. The terminal output tells you what.

That is it. No YAML configuration. No eval platform to sign up for. A
Python process, a prompts directory, and a non-zero exit code.

---

## What's Here

| File | What it does |
|------|--------------|
| [`score_history.py`](./score_history.py) | Append-only history log per prompt. `record_run`, `load_history`, `detect_model_drift`. |
| [`harness.py`](./harness.py) | Orchestrates all prompts: snapshot → eval → diff → report. Also a CLI. |
| [`regression_report.py`](./regression_report.py) | Terminal and Markdown formatters for `HarnessResult`. |
| [`examples/loan_rejection_baseline.py`](./examples/loan_rejection_baseline.py) | Run the harness against the Part 4 loan rejection prompt. Shows history and drift detection in action. |

## Run

```bash
# Run the harness against all Part 4 prompts
python3 -m vol1_foundations.part06_regression_harness.examples.loan_rejection_baseline

# Or directly via the harness CLI:
python3 -m vol1_foundations.part06_regression_harness.harness \
    vol1_foundations/part04_prompts_as_code/prompts
```

---

## What This Catches That Part 4 Alone Doesn't

| Scenario | eval_runner | Regression harness |
|----------|-------------|-------------------|
| Pass rate drops below threshold | ✓ | ✓ |
| Pass rate stays above threshold but 2 cases regressed | — | ✓ |
| Pass rate improves — but which cases fixed? | — | ✓ |
| Model updated under you, not your fault | — | ✓ (drift flag) |
| "Which run was the last clean one?" | — | ✓ (history log) |

The eval_runner is the test. The harness is the memory and the diff.
You need both.

---

## Where This Connects

- The eval runner that does the actual scoring lives in
  [`vol1_foundations/part04_prompts_as_code/eval_runner.py`](../part04_prompts_as_code/eval_runner.py).
  Part 6 wraps it — it does not replace it.
- The LLM judge behind the scoring is in
  [`vol5_evals/part23_llm_judge/`](../../vol5_evals/part23_llm_judge/) (coming).
- When your prompts are also driving tool calls, regressions get harder
  to diagnose. That is Volume III's problem, starting with
  [`vol3_agents/part12_observe_think_act/`](../../vol3_agents/part12_observe_think_act/) (coming).

---

*The harness doesn't catch everything. It catches the regressions that
your eval set covers — which is exactly the ones you promised yourself
you'd care about when you wrote those test cases. Start there.*
