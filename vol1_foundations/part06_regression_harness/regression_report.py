# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part06_regression_harness/regression_report.py

Format a HarnessResult for humans.

Two formats:
  - Terminal (ANSI-free, works in CI logs and local shells)
  - Markdown (for Slack notifications, GitHub PR comments, email)

Both formats lead with the most alarming thing first: regressions, then
failures, then drift warnings, then the good news. If a build is broken,
the engineer scanning the log should see why within the first three lines.
"""

from __future__ import annotations

from vol1_foundations.part06_regression_harness.harness import (
    HarnessResult,
    PromptHarnessResult,
)


# ──────────────────────────────────────────────────────────────────────────────
# Terminal report
# ──────────────────────────────────────────────────────────────────────────────

def format_terminal(result: HarnessResult) -> str:
    lines: list[str] = []
    _header(lines, result)
    for pr in result.prompt_results:
        _prompt_block_terminal(lines, pr)
    _footer_terminal(lines, result)
    return "\n".join(lines)


def print_report(result: HarnessResult) -> None:
    print(format_terminal(result))


def _header(lines: list[str], result: HarnessResult) -> None:
    lines.append("")
    lines.append("=" * 60)
    lines.append("  REGRESSION HARNESS REPORT")
    lines.append("=" * 60)
    lines.append(
        f"  Prompts checked: {result.total_prompts}"
        f"  |  Passed: {result.total_prompts - len(result.failed_prompts)}"
        f"  |  Failed: {len(result.failed_prompts)}"
    )
    if result.regressed_prompts:
        lines.append(f"  Regressions in:  {', '.join(result.regressed_prompts)}")
    if result.drift_prompts:
        lines.append(f"  Drift suspected: {', '.join(result.drift_prompts)}")
    lines.append("=" * 60)


def _prompt_block_terminal(lines: list[str], pr: PromptHarnessResult) -> None:
    status_icon = "✓" if pr.passed else "✗"
    lines.append("")
    lines.append(
        f"  {status_icon} {pr.prompt_name}  "
        f"v{pr.prompt_version}  "
        f"{pr.pass_rate:.1%} "
        f"(threshold {pr.threshold:.0%})"
    )

    if pr.baseline_pass_rate is not None:
        delta = pr.pass_rate - pr.baseline_pass_rate
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"    baseline v{pr.baseline_version}: {pr.baseline_pass_rate:.1%}  "
            f"delta: {sign}{delta:.1%}"
        )

    if pr.regressions:
        lines.append(f"    Regressions ({len(pr.regressions)}):")
        for r in pr.regressions:
            props = ", ".join(r.failed_properties) if r.failed_properties else "—"
            lines.append(
                f"      - {r.case_id}: "
                f"{r.old_score:.2f} → {r.new_score:.2f}  "
                f"failed: {props}"
            )

    if pr.improvements:
        lines.append(f"    Improvements ({len(pr.improvements)}):")
        for i in pr.improvements:
            lines.append(
                f"      + {i.case_id}: "
                f"{i.old_score:.2f} → {i.new_score:.2f}"
            )

    if pr.model_drift_suspected:
        lines.append(f"    ⚠  Model drift: {pr.drift_reason}")


def _footer_terminal(lines: list[str], result: HarnessResult) -> None:
    lines.append("")
    lines.append("=" * 60)
    if result.all_passed and not result.regressed_prompts:
        lines.append("  RESULT: PASS — no regressions. Safe to merge.")
    elif result.regressed_prompts:
        lines.append("  RESULT: FAIL — regressions detected. Fix before merging.")
    else:
        lines.append("  RESULT: FAIL — eval threshold not met. Check scores above.")
    lines.append("=" * 60)
    lines.append("")


# ──────────────────────────────────────────────────────────────────────────────
# Markdown report (for Slack, GitHub PR comments, email)
# ──────────────────────────────────────────────────────────────────────────────

def format_markdown(result: HarnessResult) -> str:
    lines: list[str] = []
    overall = "PASS" if result.all_passed and not result.regressed_prompts else "FAIL"
    icon = "✅" if overall == "PASS" else "❌"

    lines.append(f"## {icon} Regression Harness — {overall}")
    lines.append("")
    lines.append(
        f"| Prompts | Passed | Failed | Regressions | Drift |"
    )
    lines.append("|---------|--------|--------|-------------|-------|")
    lines.append(
        f"| {result.total_prompts} "
        f"| {result.total_prompts - len(result.failed_prompts)} "
        f"| {len(result.failed_prompts)} "
        f"| {len(result.regressed_prompts)} "
        f"| {len(result.drift_prompts)} |"
    )
    lines.append("")

    for pr in result.prompt_results:
        _prompt_block_markdown(lines, pr)

    return "\n".join(lines)


def _prompt_block_markdown(lines: list[str], pr: PromptHarnessResult) -> None:
    icon = "✅" if pr.passed else "❌"
    lines.append(
        f"### {icon} `{pr.prompt_name}` — v{pr.prompt_version} — "
        f"{pr.pass_rate:.1%}"
    )

    if pr.baseline_pass_rate is not None:
        delta = pr.pass_rate - pr.baseline_pass_rate
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"Baseline (v{pr.baseline_version}): {pr.baseline_pass_rate:.1%} "
            f"→ {pr.pass_rate:.1%} ({sign}{delta:.1%})"
        )

    if pr.regressions:
        lines.append("")
        lines.append(f"**Regressions ({len(pr.regressions)})**")
        for r in pr.regressions:
            props = ", ".join(r.failed_properties) if r.failed_properties else "—"
            lines.append(
                f"- `{r.case_id}`: score {r.old_score:.2f} → {r.new_score:.2f} "
                f"| failed: _{props}_"
            )

    if pr.improvements:
        lines.append("")
        lines.append(f"**Improvements ({len(pr.improvements)})**")
        for i in pr.improvements:
            lines.append(
                f"- `{i.case_id}`: {i.old_score:.2f} → {i.new_score:.2f}"
            )

    if pr.model_drift_suspected:
        lines.append("")
        lines.append(f"> ⚠️ **Model drift suspected:** {pr.drift_reason}")

    lines.append("")
