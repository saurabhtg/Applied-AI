# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
ci/compare_versions.py

Compare two scores.json snapshots and report regressions, improvements,
and unchanged cases.

Two ways to use:

    # Compare against main branch
    python ci/compare_versions.py prompts/loan_rejection_explanation/scores.json \\
                                  origin/main:prompts/loan_rejection_explanation/scores.json

    # Compare two arbitrary files
    python ci/compare_versions.py path/to/v10_scores.json path/to/v11_scores.json

Exit codes:
    0 — no regressions
    1 — regressions detected
    2 — error reading files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load_scores(path_or_ref: str) -> dict:
    """Load scores from a local file or Git ref (e.g. 'origin/main:path/to/file')."""
    if ":" in path_or_ref and not Path(path_or_ref).exists():
        # Treat as Git ref.
        ref, file_path = path_or_ref.split(":", 1)
        try:
            result = subprocess.run(
                ["git", "show", f"{ref}:{file_path}"],
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as exc:
            print(f"Error reading {path_or_ref}: {exc}", file=sys.stderr)
            sys.exit(2)
    else:
        path = Path(path_or_ref)
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(2)
        return json.loads(path.read_text())


def _index_cases(scores: dict) -> dict[str, dict]:
    return {c["id"]: c for c in scores.get("case_scores", [])}


def compare(new_scores: dict, old_scores: dict) -> dict:
    """Compute the diff between two score snapshots."""
    new_cases = _index_cases(new_scores)
    old_cases = _index_cases(old_scores)

    regressions: list[dict] = []
    improvements: list[dict] = []
    unchanged: list[dict] = []
    added: list[dict] = []
    removed: list[dict] = []

    for case_id, new_case in new_cases.items():
        if case_id not in old_cases:
            added.append(new_case)
            continue
        old_case = old_cases[case_id]
        if new_case["status"] == "FAIL" and old_case["status"] == "PASS":
            regressions.append({
                "id": case_id,
                "old_score": old_case["score"],
                "new_score": new_case["score"],
                "failed_properties": new_case.get("failed_properties", []),
            })
        elif new_case["status"] == "PASS" and old_case["status"] == "FAIL":
            improvements.append({
                "id": case_id,
                "old_score": old_case["score"],
                "new_score": new_case["score"],
            })
        else:
            unchanged.append({
                "id": case_id,
                "score_delta": round(new_case["score"] - old_case["score"], 2),
            })

    for case_id, old_case in old_cases.items():
        if case_id not in new_cases:
            removed.append(old_case)

    return {
        "old_version": old_scores.get("prompt_version", "unknown"),
        "new_version": new_scores.get("prompt_version", "unknown"),
        "old_pass_rate": old_scores.get("summary", {}).get("pass_rate", 0),
        "new_pass_rate": new_scores.get("summary", {}).get("pass_rate", 0),
        "regressions": regressions,
        "improvements": improvements,
        "added": [c["id"] for c in added],
        "removed": [c["id"] for c in removed],
        "unchanged_count": len(unchanged),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two prompt eval snapshots.")
    parser.add_argument("new", help="Path or Git ref to the new scores.json")
    parser.add_argument("old", help="Path or Git ref to the old scores.json")
    args = parser.parse_args()

    new_scores = _load_scores(args.new)
    old_scores = _load_scores(args.old)
    diff = compare(new_scores, old_scores)

    print(
        f"\nComparing v{diff['old_version']} → v{diff['new_version']}"
    )
    print(
        f"Pass rate: {diff['old_pass_rate']:.1%} → {diff['new_pass_rate']:.1%}"
    )

    if diff["regressions"]:
        print(f"\n✗ Regressions ({len(diff['regressions'])}):")
        for r in diff["regressions"]:
            print(
                f"  - {r['id']}: {r['old_score']:.2f} → {r['new_score']:.2f}"
                f"  failed: {', '.join(r['failed_properties'])}"
            )

    if diff["improvements"]:
        print(f"\n✓ Improvements ({len(diff['improvements'])}):")
        for i in diff["improvements"]:
            print(f"  - {i['id']}: {i['old_score']:.2f} → {i['new_score']:.2f}")

    if diff["added"]:
        print(f"\n+ New cases ({len(diff['added'])}): {', '.join(diff['added'])}")
    if diff["removed"]:
        print(f"\n- Removed cases ({len(diff['removed'])}): {', '.join(diff['removed'])}")

    print(f"\nUnchanged: {diff['unchanged_count']} cases\n")

    if diff["regressions"]:
        print("REGRESSIONS DETECTED — fix before merging.")
        return 1
    print("No regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
