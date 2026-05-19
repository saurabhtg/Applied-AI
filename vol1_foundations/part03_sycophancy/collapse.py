# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part03_sycophancy/collapse.py

Detect collapse: outputs become weirdly templated, repetitive, or generic.

The classic production story: a Myntra-style retailer's product
description generator started every single description with
"Introducing the..." and ended every one with "...you'll love it."
Three thousand products, same skeleton. The marketing team only noticed
when SEO traffic dropped — Google had figured out the pages were
templated and started downranking them.

How collapse is detected here:
    Across a batch of outputs, measure:
        1. Opening-token similarity: how often outputs start the same way.
        2. Closing-token similarity: how often outputs end the same way.
        3. N-gram repetition: how many distinct n-grams across the batch.

    If any of these cross a threshold, the batch shows collapse.

This is a statistical signal, not a per-output check. Collapse only
makes sense across a population.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class CollapseSignal:
    detected: bool
    opening_repetition_rate: float
    closing_repetition_rate: float
    distinct_ngram_ratio: float
    explanation: str
    examples: list[str] | None = None


def _tokenize(text: str) -> list[str]:
    """Cheap word-level tokenization."""
    return re.findall(r"\b\w+\b", text.lower())


def _opening(text: str, n_words: int = 3) -> str:
    """First N words, normalized."""
    tokens = _tokenize(text)
    return " ".join(tokens[:n_words])


def _closing(text: str, n_words: int = 3) -> str:
    """Last N words, normalized."""
    tokens = _tokenize(text)
    return " ".join(tokens[-n_words:]) if tokens else ""


def _ngrams(text: str, n: int = 3) -> list[str]:
    """Word n-grams."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return [" ".join(tokens)]
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def detect_collapse(
    outputs: list[str],
    *,
    opening_threshold: float = 0.5,
    closing_threshold: float = 0.5,
    distinct_ngram_threshold: float = 0.4,
    opening_words: int = 3,
    closing_words: int = 3,
) -> CollapseSignal:
    """
    Look for collapse across a batch of generated outputs.

    Args:
        outputs: A batch of generated texts (>= 5 recommended).
        opening_threshold: Fraction of outputs sharing the same opening
            above which collapse is flagged. Default 0.5.
        closing_threshold: Same, for closings.
        distinct_ngram_threshold: Fraction of distinct n-grams below which
            collapse is flagged. Lower = more repetition.
        opening_words: How many opening words to consider.
        closing_words: How many closing words to consider.

    Returns:
        A CollapseSignal with metrics and examples.
    """
    if len(outputs) < 5:
        return CollapseSignal(
            detected=False,
            opening_repetition_rate=0.0,
            closing_repetition_rate=0.0,
            distinct_ngram_ratio=1.0,
            explanation="Need at least 5 outputs to detect collapse.",
        )

    openings = [_opening(o, opening_words) for o in outputs]
    closings = [_closing(o, closing_words) for o in outputs]

    opening_counts = Counter(openings)
    closing_counts = Counter(closings)

    top_opening, top_opening_count = opening_counts.most_common(1)[0]
    top_closing, top_closing_count = closing_counts.most_common(1)[0]

    opening_rate = top_opening_count / len(outputs)
    closing_rate = top_closing_count / len(outputs)

    # N-gram diversity across the batch.
    all_ngrams: list[str] = []
    for output in outputs:
        all_ngrams.extend(_ngrams(output, n=3))
    distinct_ratio = (
        len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 1.0
    )

    reasons: list[str] = []
    if opening_rate >= opening_threshold:
        reasons.append(
            f"{opening_rate:.0%} of outputs start with \"{top_opening}\""
        )
    if closing_rate >= closing_threshold:
        reasons.append(
            f"{closing_rate:.0%} of outputs end with \"{top_closing}\""
        )
    if distinct_ratio <= distinct_ngram_threshold:
        reasons.append(
            f"only {distinct_ratio:.0%} distinct 3-grams (low diversity)"
        )

    detected = bool(reasons)
    explanation = (
        "Collapse detected: " + "; ".join(reasons)
        if detected
        else "No collapse signal."
    )

    examples = None
    if detected:
        examples = [
            o for o, op in zip(outputs, openings) if op == top_opening
        ][:3]

    return CollapseSignal(
        detected=detected,
        opening_repetition_rate=round(opening_rate, 3),
        closing_repetition_rate=round(closing_rate, 3),
        distinct_ngram_ratio=round(distinct_ratio, 3),
        explanation=explanation,
        examples=examples,
    )
