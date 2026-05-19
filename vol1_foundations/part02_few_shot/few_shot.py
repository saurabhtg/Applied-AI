# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part02_few_shot/few_shot.py

Build few-shot prompts.

The lesson from Part 2: when you find yourself adding a sixth bullet
point of instructions, stop and write an example instead. Models pattern-
match better than they follow rules.

The e-commerce extraction case in the article went from 70% accuracy
with a long paragraph of rules to 91% accuracy with three examples and
one line of instruction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class FewShotExample:
    """One input/output example for few-shot prompting."""

    input: Any
    output: Any
    label: str | None = None  # Optional caption for the example.


def build_few_shot_prompt(
    instruction: str,
    examples: list[FewShotExample],
    query: Any,
    *,
    input_label: str = "Input",
    output_label: str = "Output",
    use_json_formatting: bool = True,
) -> str:
    """
    Compose a few-shot prompt from an instruction and a list of examples.

    Args:
        instruction: One-line task description. Keep it short.
        examples: 2-5 input/output pairs that demonstrate the task.
        query: The actual input you want the model to process.
        input_label: Label preceding each input. Default "Input".
        output_label: Label preceding each output. Default "Output".
        use_json_formatting: If True, dict inputs/outputs are JSON-serialized.

    Returns:
        A formatted prompt string ready to send to the model.

    Design notes:
        - 3-5 examples is the sweet spot. Two is often enough but four is
          safer for harder formats.
        - Vary the *structure* of your examples, not just the content.
          If all your examples have the same shape, the model will lock
          onto that shape (see collapse failure mode).
        - For extraction tasks, include at least one example where a
          field is null/missing. Otherwise the model thinks every field
          is always present.
    """
    if not examples:
        raise ValueError("few-shot needs at least one example. Use 3-5 in practice.")
    if len(examples) > 8:
        # Not an error, but you're probably overdoing it.
        # More examples = more tokens = more cost, with diminishing returns.
        pass

    def fmt(value: Any) -> str:
        if isinstance(value, (dict, list)) and use_json_formatting:
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    parts: list[str] = [instruction.strip(), ""]
    parts.append("Examples:")
    parts.append("")

    for i, ex in enumerate(examples, start=1):
        if ex.label:
            parts.append(f"# {ex.label}")
        parts.append(f"{input_label}: {fmt(ex.input)}")
        parts.append(f"{output_label}: {fmt(ex.output)}")
        parts.append("")

    parts.append(f"Now process this:")
    parts.append(f"{input_label}: {fmt(query)}")
    parts.append(f"{output_label}:")

    return "\n".join(parts)


def diversity_score(examples: list[FewShotExample]) -> float:
    """
    Rough diversity heuristic for a set of examples.

    Returns 0.0-1.0 where 1.0 means all examples have distinct
    opening tokens in their outputs. Low scores predict collapse.

    Use this as a smoke check on your example set before shipping.
    """
    if len(examples) < 2:
        return 1.0

    openings = []
    for ex in examples:
        out_str = ex.output if isinstance(ex.output, str) else json.dumps(ex.output)
        first_words = out_str.strip().split()[:2]
        openings.append(" ".join(first_words).lower())

    return len(set(openings)) / len(openings)
