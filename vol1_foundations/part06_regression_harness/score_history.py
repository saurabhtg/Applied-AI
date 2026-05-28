# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part06_regression_harness/score_history.py

Append-only run history for each prompt.

Every time the harness runs a prompt, one line gets appended to
`history.jsonl` next to the prompt's `scores.json`. That log is what
lets you answer two questions that will otherwise keep you up at night:

  1. "When did this regression appear?" — scan for the last PASS entry.
  2. "Did my prompt change cause this, or did the model drift?"
     If the prompt version is the same but the model string changed
     and pass rate dropped, that is model drift. Anthropic pushed an
     update. Not your bug. But still your problem at 2 a.m.

The file stays small: we record only the summary, not full per-case
output. Case detail lives in scores.json, which is always the latest run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class HistoryEntry:
    run_id: str
    timestamp: str
    prompt_version: str
    model: str
    git_commit: str
    pass_rate: float
    passed: int
    failed: int
    total: int
    status: str        # "PASS" | "FAIL"
    threshold: float


def record_run(prompt_dir: Path, scores: dict) -> HistoryEntry:
    """
    Append a summary of this eval run to `<prompt_dir>/history.jsonl`.

    Creates the file if it does not exist. Returns the new entry so the
    caller can use it without re-reading the file.
    """
    summary = scores["summary"]
    entry = HistoryEntry(
        run_id=scores.get("eval_run_id", ""),
        timestamp=scores.get("eval_run_timestamp", ""),
        prompt_version=scores.get("prompt_version", "unknown"),
        model=scores.get("model", "unknown"),
        git_commit=scores.get("git_commit", "unknown"),
        pass_rate=summary["pass_rate"],
        passed=summary["passed"],
        failed=summary["failed"],
        total=summary["total_cases"],
        status=summary["status"],
        threshold=summary["pass_threshold"],
    )
    history_path = Path(prompt_dir) / "history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
    return entry


def load_history(prompt_dir: Path) -> list[HistoryEntry]:
    """Load all history entries for a prompt, oldest first."""
    history_path = Path(prompt_dir) / "history.jsonl"
    if not history_path.exists():
        return []
    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(HistoryEntry(**json.loads(line)))
            except (json.JSONDecodeError, TypeError):
                continue
    return entries


def last_passing_entry(history: list[HistoryEntry]) -> HistoryEntry | None:
    """Return the most recent PASS entry, or None if there are none."""
    for entry in reversed(history):
        if entry.status == "PASS":
            return entry
    return None


def detect_model_drift(history: list[HistoryEntry]) -> tuple[bool, str]:
    """
    Return (True, reason) if pass_rate dropped after a model change
    with no corresponding prompt version change.

    Logic: look for back-to-back entries where the model string changed
    but the prompt version stayed the same, and pass_rate dropped by
    more than 5 percentage points. That signature is model drift.

    Returns (False, "") if no drift is detected or history is too short.
    """
    if len(history) < 2:
        return False, ""

    for i in range(1, len(history)):
        prev, curr = history[i - 1], history[i]
        same_prompt = prev.prompt_version == curr.prompt_version
        model_changed = prev.model != curr.model
        rate_dropped = prev.pass_rate - curr.pass_rate > 0.05

        if same_prompt and model_changed and rate_dropped:
            reason = (
                f"prompt stayed at v{curr.prompt_version} but model changed "
                f"from {prev.model} to {curr.model}; "
                f"pass rate dropped {prev.pass_rate:.1%} → {curr.pass_rate:.1%}"
            )
            return True, reason

    return False, ""
