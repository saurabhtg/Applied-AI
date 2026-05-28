# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
examples/loan_rejection_baseline.py

Run the regression harness against the loan rejection prompt from Part 4.

This is the Part 6 story in code. Priya "improved" the loan rejection
explanation prompt on Thursday — fixed a case where the model used
condescending phrasing for freelancers. She tested it manually on two
cases. Both looked better. She shipped it.

On Monday, the call center manager called.

The harness would have caught it. It runs all 12 eval cases, not the
two she tested. It compares each case against the last baseline. And it
shows her exactly which cases regressed — eval_010, the sycophancy case,
which the new "warmer" phrasing accidentally made worse.

Run:
    python3 -m vol1_foundations.part06_regression_harness.examples.loan_rejection_baseline

This makes real model API calls. Cost is under ₹5-10 per run.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from vol1_foundations.part06_regression_harness.harness import RegressionHarness
from vol1_foundations.part06_regression_harness.regression_report import (
    format_markdown,
    print_report,
)
from vol1_foundations.part06_regression_harness.score_history import (
    load_history,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

PROMPTS_DIR = Path("vol1_foundations/part04_prompts_as_code/prompts")


def main() -> int:
    if not PROMPTS_DIR.exists():
        print(
            f"Prompts directory not found: {PROMPTS_DIR}\n"
            "Run from the repo root: "
            "python3 -m vol1_foundations.part06_regression_harness.examples.loan_rejection_baseline"
        )
        return 2

    print("\nRunning regression harness against Part 4 prompts...\n")
    harness = RegressionHarness(PROMPTS_DIR)
    result = harness.run()

    # Terminal report — what you see in CI logs
    print_report(result)

    # History summary — the long-term view
    for pr in result.prompt_results:
        prompt_dir = PROMPTS_DIR / pr.prompt_name
        history = load_history(prompt_dir)
        if len(history) >= 2:
            print(f"\nScore history for {pr.prompt_name} ({len(history)} runs):")
            print(f"  {'Date':<26} {'Version':<8} {'Model':<28} {'Rate':>6}  Status")
            print(f"  {'-'*26} {'-'*8} {'-'*28} {'-'*6}  ------")
            for entry in history[-6:]:  # last 6 runs
                ts = entry.timestamp[:19].replace("T", " ")
                print(
                    f"  {ts:<26} v{entry.prompt_version:<7} "
                    f"{entry.model:<28} {entry.pass_rate:>6.1%}  {entry.status}"
                )

    # If you want the markdown format for Slack or a PR comment:
    # print(format_markdown(result))

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
