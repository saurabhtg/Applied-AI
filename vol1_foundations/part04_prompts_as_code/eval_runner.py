# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part04_prompts_as_code/eval_runner.py

Run an eval set against a prompt and write scores.json.

Usage:
    python3 -m vol1_foundations.part04_prompts_as_code.eval_runner vol1_foundations/part04_prompts_as_code/prompts/loan_rejection_explanation

What it does:
    1. Loads prompt.txt (parsing the version from its header).
    2. Loads eval.jsonl (one test case per line).
    3. Runs each test case through the model.
    4. Scores each case against its expected_properties using an LLM judge.
    5. Writes scores.json next to the prompt.
    6. Exits 0 on PASS, 1 on FAIL — so CI can gate merges on it.

The shape of an eval case (in eval.jsonl):
    {
        "id": "eval_001",
        "input": { ... },                  # passed to prompt.format()
        "expected_properties": { ... },     # checked by the judge
        "notes": "what this case exists to catch"
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from vol5_evals.part23_llm_judge.llm_judge import judge_response
from common.llm_client import LLMClient
from vol1_foundations.part04_prompts_as_code.prompt_loader import load_prompt
from common.tracing import span

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


@dataclass
class CaseResult:
    id: str
    status: str  # "PASS" or "FAIL"
    score: float
    properties_passed: int
    properties_failed: int
    failed_properties: list[str]
    output: str
    notes: str = ""


def _git_commit() -> str:
    """Return the current Git commit short hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _load_eval_cases(eval_path: Path) -> list[dict[str, Any]]:
    """Parse JSONL file, one case per line."""
    cases = []
    with eval_path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {eval_path}: {exc}"
                ) from exc
    return cases


def run_eval(prompt_dir: Path, model: str | None = None) -> dict[str, Any]:
    """
    Run the eval set for one prompt directory.

    Args:
        prompt_dir: Directory containing prompt.txt and eval.jsonl.
        model: Optional model override.

    Returns:
        The full scores dictionary that gets written to scores.json.
    """
    prompt_dir = Path(prompt_dir)
    prompt_name = prompt_dir.name
    eval_path = prompt_dir / "eval.jsonl"
    scores_path = prompt_dir / "scores.json"

    prompt = load_prompt(prompt_name, prompts_dir=prompt_dir.parent)
    cases = _load_eval_cases(eval_path)

    if not cases:
        raise ValueError(f"No eval cases found in {eval_path}")

    client = LLMClient()
    results: list[CaseResult] = []

    with span(
        "eval.run",
        attributes={
            "eval.prompt_name": prompt_name,
            "eval.prompt_version": prompt.version,
            "eval.case_count": len(cases),
        },
    ):
        for case in cases:
            result = _run_one_case(case, prompt, client, model)
            results.append(result)
            logger.info("case %s: %s (score=%.2f)", result.id, result.status, result.score)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = len(results) - passed
    pass_rate = passed / len(results)
    status = "PASS" if pass_rate >= prompt.eval_pass_threshold else "FAIL"

    scores = {
        "prompt_name": prompt_name,
        "prompt_version": prompt.version,
        "model": model or client.default_model,
        "eval_run_id": f"run_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}",
        "eval_run_timestamp": datetime.now(IST).isoformat(),
        "git_commit": _git_commit(),
        "summary": {
            "total_cases": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 3),
            "pass_threshold": prompt.eval_pass_threshold,
            "status": status,
        },
        "case_scores": [asdict(r) for r in results],
    }

    scores_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False))
    return scores


def _run_one_case(
    case: dict[str, Any],
    prompt: Any,
    client: LLMClient,
    model: str | None,
) -> CaseResult:
    """Run one eval case and score it."""
    case_id = case["id"]
    case_input = case["input"]
    expected = case.get("expected_properties", {})
    notes = case.get("notes", "")

    # Most prompts take a {input} placeholder. If yours takes named fields,
    # the prompt template should reference them directly; we pass the raw
    # dict in either case.
    if "{input}" in prompt.text:
        formatted = prompt.text.replace("{input}", json.dumps(case_input, ensure_ascii=False))
    else:
        try:
            formatted = prompt.text.format(**case_input)
        except KeyError:
            formatted = prompt.text + "\n\nInput: " + json.dumps(case_input, ensure_ascii=False)

    response = client.complete(
        prompt=formatted,
        model=model,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        temperature=0.3,
    )

    judgment = judge_response(
        output=response.text,
        expected_properties=expected,
        case_input=case_input,
    )

    failed_props = [k for k, v in judgment.property_results.items() if not v]
    passed_count = len(judgment.property_results) - len(failed_props)

    return CaseResult(
        id=case_id,
        status="PASS" if judgment.score >= 0.85 else "FAIL",
        score=round(judgment.score, 2),
        properties_passed=passed_count,
        properties_failed=len(failed_props),
        failed_properties=failed_props,
        output=response.text,
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run evals for a prompt.")
    parser.add_argument("prompt_dir", help="Path to the prompt directory.")
    parser.add_argument("--model", help="Override the default model.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    scores = run_eval(Path(args.prompt_dir), model=args.model)
    summary = scores["summary"]

    print(f"\nRunning eval set: {scores['prompt_name']} v{scores['prompt_version']}")
    print(
        f"Cases: {summary['total_cases']} | "
        f"Passed: {summary['passed']} | "
        f"Failed: {summary['failed']} | "
        f"Pass rate: {summary['pass_rate']:.1%}"
    )
    print(f"Status: {summary['status']} (threshold: {summary['pass_threshold']:.0%})")
    if summary["failed"] > 0:
        failed_ids = [
            c["id"] for c in scores["case_scores"] if c["status"] == "FAIL"
        ]
        print(f"Failures: {', '.join(failed_ids)}")
    print(f"Full scores written to: {Path(args.prompt_dir) / 'scores.json'}\n")

    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
