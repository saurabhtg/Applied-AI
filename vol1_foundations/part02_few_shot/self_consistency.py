# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part02_few_shot/self_consistency.py

Self-consistency: run the same prompt N times at temperature > 0, take
the majority answer.

The math from Part 2: if a single call is 75% accurate on hard cases,
five calls with majority vote pushes you to about 90%. The UPI fraud
team used this to cut their false-block rate by a third.

When to use:
    - Classification or short-answer tasks where the answer is one of
      a small set of categories.
    - Hard inputs where the model is mostly right but occasionally flips.

When NOT to use:
    - Generation tasks (summaries, descriptions, anything long-form).
      There's no "majority" of free text.
    - Anything where the cost of N calls outweighs the accuracy gain.

A common pattern: run one call first. If the model returns a low-
confidence category (or specifically returns "review_needed"), then
run the remaining N-1 calls. Most cases only cost one call.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from common.llm_client import LLMClient


@dataclass
class ConsistencyResult:
    answer: str
    vote_count: int
    total_runs: int
    agreement_rate: float
    all_answers: list[str]


def self_consistent_answer(
    prompt: str,
    *,
    n: int = 5,
    temperature: float = 0.7,
    extract_answer: Callable[[str], str] | None = None,
    client: LLMClient | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> ConsistencyResult:
    """
    Run the prompt N times, return the majority answer.

    Args:
        prompt: The prompt to run.
        n: Number of runs. 5 is the typical sweet spot.
        temperature: Must be > 0 — otherwise all runs return the same answer.
        extract_answer: Optional function to extract the answer from each
            raw response. Defaults to using the whole stripped response.
            Useful when the model returns reasoning + an answer and you
            only want to vote on the answer.
        client: Optional LLM client.
        prompt_name: For logging.
        prompt_version: For logging.

    Returns:
        A ConsistencyResult with the winning answer, vote count, and all answers.
    """
    if temperature <= 0:
        raise ValueError(
            "self-consistency needs temperature > 0. Otherwise every run "
            "returns the same answer and you've just paid 5x for one call."
        )
    if n < 3:
        raise ValueError("self-consistency needs at least 3 runs (5 is typical).")

    client = client or LLMClient()
    extract = extract_answer or (lambda s: s.strip())

    answers: list[str] = []
    for _ in range(n):
        response = client.complete(
            prompt=prompt,
            temperature=temperature,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
        )
        answers.append(extract(response.text))

    counter = Counter(answers)
    winner, vote_count = counter.most_common(1)[0]

    return ConsistencyResult(
        answer=winner,
        vote_count=vote_count,
        total_runs=n,
        agreement_rate=round(vote_count / n, 3),
        all_answers=answers,
    )
