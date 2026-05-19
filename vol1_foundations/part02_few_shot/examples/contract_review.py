# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
examples/contract_review/review.py

Review a non-compete clause for enforceability under Indian law.

This is the example from Part 2. A contract-review tool was telling
Bengaluru-based startups that their employee non-compete clauses were
enforceable. Under Indian law — specifically Section 27 of the Indian
Contract Act, 1872 — post-employment non-compete restrictions are void.
The model had picked up the American framing of the question (where
non-competes are often enforceable, jurisdiction permitting) and never
considered that the answer might be different in India.

The fix: prompt voting. Three structurally different phrasings of the
same question, two runs each, majority vote across six answers.

Phrasing 1 is neutral.
Phrasing 2 invokes Section 27 explicitly.
Phrasing 3 frames it as a court case in an Indian city.

The wrong answer (under the American framing) stopped winning the vote.

Run:
    python3 -m vol1_foundations.part02_few_shot.examples.contract_review
"""

from __future__ import annotations

import json
import re

from common.llm_client import LLMClient
from vol1_foundations.part02_few_shot.prompt_voting import prompt_vote


CLAUSE = """\
The Employee agrees that for a period of 24 months following termination
of employment with the Company, the Employee shall not, directly or
indirectly, engage in any business that competes with the Company within
the territory of India. The Employee further agrees not to solicit any
customers, clients, or employees of the Company during this period.
"""


def _phrasing_neutral(clause: str) -> str:
    return (
        f"Review the following non-compete clause and assess whether it would be "
        f"enforceable. Respond with ONE word only: ENFORCEABLE or VOID.\n\n"
        f"Clause:\n{clause}"
    )


def _phrasing_section_27(clause: str) -> str:
    return (
        f"Under Section 27 of the Indian Contract Act, 1872, would the following "
        f"post-employment restriction hold up against challenge in an Indian court? "
        f"Respond with ONE word only: ENFORCEABLE or VOID.\n\n"
        f"Clause:\n{clause}"
    )


def _phrasing_court(clause: str) -> str:
    return (
        f"You are advising a Bengaluru-based employee who has been told to sign "
        f"this clause. If the employee leaves and the employer attempts to enforce "
        f"this in a Karnataka High Court, what is the likely ruling? Respond with "
        f"ONE word only: ENFORCEABLE or VOID.\n\n"
        f"Clause:\n{clause}"
    )


def _extract_verdict(text: str) -> str:
    """Pull ENFORCEABLE or VOID out of the model output, robust to extra wording."""
    upper = text.upper()
    if "VOID" in upper:
        return "VOID"
    if "ENFORCEABLE" in upper or "UNENFORCEABLE" in upper:
        # If both appear, prefer the one that appears first.
        m = re.search(r"\b(UN)?ENFORCEABLE\b", upper)
        if m and m.group(0) == "UNENFORCEABLE":
            return "VOID"
        return "ENFORCEABLE"
    return "UNCLEAR"


def review_clause(clause: str = CLAUSE) -> dict:
    """Run prompt voting on the clause and return the verdict."""
    client = LLMClient()

    phrasings = [
        _phrasing_neutral(clause),
        _phrasing_section_27(clause),
        _phrasing_court(clause),
    ]

    result = prompt_vote(
        prompts=phrasings,
        runs_per_prompt=2,
        temperature=0.5,
        extract_answer=_extract_verdict,
        client=client,
        prompt_name="contract_review_noncompete",
        prompt_version="1.0",
    )

    return {
        "clause": clause.strip(),
        "verdict": result.answer,
        "agreement_rate": result.agreement_rate,
        "votes_for_winner": result.vote_count,
        "total_votes": result.total_votes,
        "per_phrasing": result.per_prompt_answers,
        "legal_basis": (
            "Section 27 of the Indian Contract Act, 1872 declares void any "
            "agreement that restrains a person from exercising a lawful "
            "profession, trade, or business. The Supreme Court has repeatedly "
            "held that post-employment non-compete restrictions are void in "
            "India, with narrow exceptions for trade-secret protection."
        ) if result.answer == "VOID" else None,
    }


def main() -> None:
    result = review_clause()
    print("\nContract Review: Post-Employment Non-Compete\n")
    print(f"Verdict: {result['verdict']}")
    print(f"Agreement: {result['votes_for_winner']}/{result['total_votes']} votes ({result['agreement_rate']:.0%})")
    print("\nPer-phrasing breakdown:")
    for phrasing, answers in result["per_phrasing"].items():
        print(f"  {phrasing}: {answers}")
    if result["legal_basis"]:
        print(f"\nLegal basis:\n{result['legal_basis']}")
    print()


if __name__ == "__main__":
    main()
