# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part02_few_shot/prompt_voting.py

Majority voting across structurally different prompts.

The legal-tech case from Part 2: a contract-review tool kept saying a
non-compete clause was enforceable, when in India, Section 27 of the
Indian Contract Act, 1872 makes it void. Self-consistency didn't help —
the model was consistently wrong on that specific phrasing. The fix was
to add two more phrasings:

    - "Under Section 27 of the Indian Contract Act, 1872, would this
       clause hold up in an Indian court?"
    - "If this dispute went to a court in Bengaluru, what would the
       likely ruling be?"

Three prompts, each run twice, majority vote across six answers. The
wrong answer stopped winning.

Use this when you suspect prompt-specific bias, not just sampling noise.
If the model is confidently wrong no matter how many times you re-roll,
re-rolling won't help. Change the question.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from common.llm_client import LLMClient


@dataclass
class PromptVoteResult:
    answer: str
    vote_count: int
    total_votes: int
    agreement_rate: float
    per_prompt_answers: dict[str, list[str]]


def prompt_vote(
    prompts: list[str],
    *,
    runs_per_prompt: int = 2,
    temperature: float = 0.5,
    extract_answer: Callable[[str], str] | None = None,
    client: LLMClient | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> PromptVoteResult:
    """
    Run multiple phrasings of the same question, take the majority answer.

    Args:
        prompts: 2-5 different phrasings of the same question. They should
            ask for the same answer in genuinely different ways — not the
            same prompt with synonyms.
        runs_per_prompt: How many times to run each phrasing. 2 is typical.
        temperature: Sampling temperature. > 0 recommended.
        extract_answer: Optional answer extractor.
        client: Optional LLM client.
        prompt_name: For logging.
        prompt_version: For logging.

    Returns:
        A PromptVoteResult with the winning answer and per-prompt breakdown.

    Notes:
        If you only have one phrasing, this is just self-consistency.
        Use self_consistent_answer() instead.
    """
    if len(prompts) < 2:
        raise ValueError(
            "prompt voting needs at least 2 different phrasings. "
            "If you only have one, use self_consistent_answer instead."
        )

    client = client or LLMClient()
    extract = extract_answer or (lambda s: s.strip())

    all_answers: list[str] = []
    per_prompt: dict[str, list[str]] = {}

    for idx, prompt in enumerate(prompts):
        prompt_key = f"phrasing_{idx + 1}"
        per_prompt[prompt_key] = []
        for _ in range(runs_per_prompt):
            response = client.complete(
                prompt=prompt,
                temperature=temperature,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
            )
            answer = extract(response.text)
            per_prompt[prompt_key].append(answer)
            all_answers.append(answer)

    counter = Counter(all_answers)
    winner, vote_count = counter.most_common(1)[0]

    return PromptVoteResult(
        answer=winner,
        vote_count=vote_count,
        total_votes=len(all_answers),
        agreement_rate=round(vote_count / len(all_answers), 3),
        per_prompt_answers=per_prompt,
    )
