# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part04_prompts_as_code/prompt_loader.py

Load prompts from disk with version metadata parsed out of the header.

The pattern: every prompt is a .txt file with comment-style metadata at the top,
followed by the actual prompt body. The version lives in the file, not the
filename, so PRs show real word-level diffs in Git.

Example file format:

    # version: 1.0
    # updated: 2026-05-19
    # author: saurabhg@thinknbuild.co.in
    # owner: lending-platform-team
    # eval_pass_threshold: 0.85

    You are a customer service writer for a digital lending platform...

This is twenty lines of code that will save you a month of debugging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Prompt:
    """A loaded prompt with its metadata."""

    name: str
    text: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def format(self, **kwargs: Any) -> str:
        """Format the prompt body with the given keyword arguments."""
        return self.text.format(**kwargs)

    @property
    def eval_pass_threshold(self) -> float:
        """Threshold below which the eval is considered a failure. Default: 0.85."""
        return float(self.metadata.get("eval_pass_threshold", 0.85))


_HEADER_LINE_RE = re.compile(r"^\s*#\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+?)\s*$")


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """
    Parse comment-style metadata from the top of a prompt file.

    Returns (metadata_dict, line_index_where_body_starts).
    Stops at the first non-comment, non-blank line.
    """
    metadata: dict[str, str] = {}
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # Blank line — could be inside header or right before body. Keep scanning.
            body_start = i + 1
            continue

        match = _HEADER_LINE_RE.match(line)
        if match:
            key, value = match.group(1), match.group(2)
            metadata[key] = value
            body_start = i + 1
            continue

        # First real (non-blank, non-header) line — body starts here.
        body_start = i
        break

    return metadata, body_start


def load_prompt(name: str, prompts_dir: str | Path = "prompts") -> Prompt:
    """
    Load a prompt by name.

    Looks for `<prompts_dir>/<name>/prompt.txt`. The version and other metadata
    are parsed from the file header. The remaining content is the prompt body.

    Args:
        name: The prompt's directory name (e.g. "loan_rejection_explanation").
        prompts_dir: Root directory containing prompt subdirectories.

    Returns:
        A Prompt object with .text, .version, and .metadata.

    Raises:
        FileNotFoundError: If the prompt file does not exist.
        ValueError: If the file has no version in its header.
    """
    prompts_dir = Path(prompts_dir)
    prompt_path = prompts_dir / name / "prompt.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}. "
            f"Expected directory structure: {prompts_dir}/<name>/prompt.txt"
        )

    raw = prompt_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    metadata, body_start = _parse_header(lines)

    if "version" not in metadata:
        raise ValueError(
            f"Prompt {prompt_path} has no '# version: X.Y' header. "
            f"Every prompt must declare its version. "
            f"See Part 4 of the Applied AI course for why."
        )

    body = "\n".join(lines[body_start:]).strip()

    return Prompt(
        name=name,
        text=body,
        version=metadata["version"],
        metadata=metadata,
        source_path=prompt_path,
    )


def list_prompts(prompts_dir: str | Path = "prompts") -> list[str]:
    """List all prompt names available in the given directory."""
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.exists():
        return []
    return sorted(
        d.name for d in prompts_dir.iterdir()
        if d.is_dir() and (d / "prompt.txt").exists()
    )
