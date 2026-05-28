# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part06_regression_harness/harness.py

The regression harness. Orchestrates eval runs across every prompt in a
directory, compares each run against the last known-good baseline, and
tells you exactly what got worse.

Part 4's eval_runner answers "did this prompt pass its threshold?" That
is necessary. It is not sufficient. This harness answers the better
question: "did anything GET WORSE since the last time everything was
fine?" Those are different questions. The first one can pass while the
second one fails, and when it does, call center volume goes up on Monday.

Usage (programmatic):
    from pathlib import Path
    from vol1_foundations.part06_regression_harness.harness import RegressionHarness

    result = RegressionHarness(Path("prompts")).run()
    if not result.all_passed:
        sys.exit(1)

Usage (CLI):
    python3 -m vol1_foundations.part06_regression_harness.harness \\
        vol1_foundations/part04_prompts_as_code/prompts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from vol1_foundations.part04_prompts_as_code.eval_runner import run_eval
from vol1_foundations.part06_regression_harness.score_history import (
    detect_model_drift,
    last_passing_entry,
    load_history,
    record_run,
)

logger = logging.getLogger(__name__)


@dataclass
class CaseDelta:
    """One test case that changed status between runs."""

    case_id: str
    old_status: str
    new_status: str
    old_score: float
    new_score: float
    failed_properties: list[str] = field(default_factory=list)

    @property
    def is_regression(self) -> bool:
        return self.old_status == "PASS" and self.new_status == "FAIL"

    @property
    def is_improvement(self) -> bool:
        return self.old_status == "FAIL" and self.new_status == "PASS"


@dataclass
class PromptHarnessResult:
    """Everything the harness knows about one prompt after a run."""

    prompt_name: str
    prompt_version: str
    pass_rate: float
    threshold: float
    passed: bool
    model_drift_suspected: bool
    drift_reason: str
    regressions: list[CaseDelta] = field(default_factory=list)
    improvements: list[CaseDelta] = field(default_factory=list)
    baseline_version: str | None = None
    baseline_pass_rate: float | None = None


@dataclass
class HarnessResult:
    """The top-level result of running the harness across all prompts."""

    prompt_results: list[PromptHarnessResult]
    all_passed: bool
    total_prompts: int
    failed_prompts: list[str]
    regressed_prompts: list[str]
    improved_prompts: list[str]
    drift_prompts: list[str]


class RegressionHarness:
    """
    Run all prompts in a directory, compare each to its baseline, report
    what regressed.

    The comparison is case-level: not just "did the pass rate drop?" but
    "which specific test cases changed status?" That is the diff that
    tells you whether your prompt change helped, hurt, or is fine to merge.
    """

    def __init__(self, prompts_root: Path, model: str | None = None):
        self.prompts_root = Path(prompts_root)
        self.model = model

    def run(self) -> HarnessResult:
        prompt_dirs = sorted(
            d for d in self.prompts_root.iterdir()
            if d.is_dir()
            and (d / "prompt.txt").exists()
            and (d / "eval.jsonl").exists()
        )
        if not prompt_dirs:
            raise ValueError(
                f"No runnable prompt directories found under {self.prompts_root}. "
                f"Each directory needs prompt.txt and eval.jsonl."
            )

        results: list[PromptHarnessResult] = []
        for prompt_dir in prompt_dirs:
            logger.info("Harness: running %s", prompt_dir.name)
            result = self._run_one(prompt_dir)
            results.append(result)

        return HarnessResult(
            prompt_results=results,
            all_passed=all(r.passed for r in results),
            total_prompts=len(results),
            failed_prompts=[r.prompt_name for r in results if not r.passed],
            regressed_prompts=[r.prompt_name for r in results if r.regressions],
            improved_prompts=[r.prompt_name for r in results if r.improvements],
            drift_prompts=[r.prompt_name for r in results if r.model_drift_suspected],
        )

    def _run_one(self, prompt_dir: Path) -> PromptHarnessResult:
        # Load old cases BEFORE run_eval overwrites scores.json.
        old_cases = _load_cases_from_scores(prompt_dir)

        # Also grab the last passing history entry for the baseline label.
        old_history = load_history(prompt_dir)
        baseline = last_passing_entry(old_history)

        # Run evals — this overwrites scores.json with the new results.
        scores = run_eval(prompt_dir, model=self.model)

        # Append to history AFTER we have the new scores.
        record_run(prompt_dir, scores)

        # Re-read history (now includes the new entry) for drift detection.
        history = load_history(prompt_dir)
        drift, drift_reason = detect_model_drift(history)

        new_cases = {c["id"]: c for c in scores["case_scores"]}
        regressions, improvements = _diff_cases(old_cases, new_cases)

        summary = scores["summary"]
        return PromptHarnessResult(
            prompt_name=prompt_dir.name,
            prompt_version=scores["prompt_version"],
            pass_rate=summary["pass_rate"],
            threshold=summary["pass_threshold"],
            passed=summary["status"] == "PASS",
            model_drift_suspected=drift,
            drift_reason=drift_reason,
            regressions=regressions,
            improvements=improvements,
            baseline_version=baseline.prompt_version if baseline else None,
            baseline_pass_rate=baseline.pass_rate if baseline else None,
        )


def _load_cases_from_scores(prompt_dir: Path) -> dict[str, dict]:
    """
    Load the case results from scores.json before we overwrite it.

    Returns an empty dict if the file doesn't exist or is malformed —
    that just means we have no baseline to compare against, which is
    fine for a first run.
    """
    scores_path = Path(prompt_dir) / "scores.json"
    if not scores_path.exists():
        return {}
    try:
        data = json.loads(scores_path.read_text(encoding="utf-8"))
        return {c["id"]: c for c in data.get("case_scores", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _diff_cases(
    old: dict[str, dict],
    new: dict[str, dict],
) -> tuple[list[CaseDelta], list[CaseDelta]]:
    """
    Compute which cases regressed and which improved.

    Both dicts are keyed by case ID. Cases that only exist in one dict
    (added or removed) are ignored — they are not regressions.
    """
    regressions: list[CaseDelta] = []
    improvements: list[CaseDelta] = []

    for case_id, new_case in new.items():
        if case_id not in old:
            continue
        old_case = old[case_id]
        if old_case["status"] == new_case["status"]:
            continue

        delta = CaseDelta(
            case_id=case_id,
            old_status=old_case["status"],
            new_status=new_case["status"],
            old_score=old_case.get("score", 0.0),
            new_score=new_case.get("score", 0.0),
            failed_properties=new_case.get("failed_properties", []),
        )
        if delta.is_regression:
            regressions.append(delta)
        elif delta.is_improvement:
            improvements.append(delta)

    return regressions, improvements


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the regression harness across all prompts in a directory."
    )
    parser.add_argument(
        "prompts_dir",
        help="Root directory containing prompt subdirectories.",
    )
    parser.add_argument(
        "--model",
        help="Override the default model for this run.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    from vol1_foundations.part06_regression_harness.regression_report import print_report

    harness = RegressionHarness(Path(args.prompts_dir), model=args.model)
    result = harness.run()
    print_report(result)
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
