# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
examples/upi_fraud_classifier.py

UPI transaction fraud classifier using self-consistency.

The pattern from Part 2: most transactions are easy (a ₹120 Swiggy order
from a phone that's done it fifty times). The 5% that are ambiguous get
5 model calls and a majority vote. Cost goes up ~8% total. False-block
rate drops by a third.

Run:
    python -m vol1_foundations.part02_few_shot.examples.upi_fraud_classifier
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from common.llm_client import LLMClient
from vol1_foundations.part02_few_shot.few_shot import FewShotExample, build_few_shot_prompt
from vol1_foundations.part02_few_shot.self_consistency import self_consistent_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


Decision = Literal["legitimate", "review_needed", "block"]


# Few-shot examples covering the three categories.
# Note the deliberate diversity: weekday/weekend, different merchants,
# different amounts, different payee histories. Same-shape examples
# cause collapse.
EXAMPLES = [
    FewShotExample(
        input={
            "amount_inr": 120,
            "merchant": "Swiggy",
            "merchant_category": "food_delivery",
            "payee_known_for_days": 412,
            "hour_ist": 13,
            "amount_z_score_vs_user_history": 0.2,
        },
        output={"decision": "legitimate", "reason": "Routine small food order, established payee"},
    ),
    FewShotExample(
        input={
            "amount_inr": 49999,
            "merchant": "Unknown Individual",
            "merchant_category": "p2p",
            "payee_known_for_days": 0,
            "hour_ist": 2,
            "amount_z_score_vs_user_history": 4.8,
        },
        output={"decision": "block", "reason": "Round-figure amount under reporting threshold, new payee, odd hour"},
    ),
    FewShotExample(
        input={
            "amount_inr": 12500,
            "merchant": "Apollo Pharmacy",
            "merchant_category": "pharmacy",
            "payee_known_for_days": 0,
            "hour_ist": 21,
            "amount_z_score_vs_user_history": 2.1,
        },
        output={"decision": "review_needed", "reason": "Larger than usual but legitimate merchant category, evening hour"},
    ),
]


INSTRUCTION = """\
You are a fraud-detection classifier for UPI transactions in India.
Given a transaction with its features, decide if it is legitimate,
needs human review, or should be blocked outright.

Respond with valid JSON only, in this format:
{"decision": "legitimate" | "review_needed" | "block", "reason": "<one short sentence>"}
"""


def classify_transaction(transaction: dict) -> dict:
    """
    Classify one UPI transaction.

    Cheap path: one model call. If the model returns review_needed or
    block, fall back to self-consistency with 5 calls and a majority vote.
    """
    client = LLMClient()
    prompt = build_few_shot_prompt(
        instruction=INSTRUCTION,
        examples=EXAMPLES,
        query=transaction,
    )

    # First pass: single call.
    response = client.complete(
        prompt=prompt,
        max_tokens=200,
        temperature=0.2,
        prompt_name="upi_fraud_classifier",
        prompt_version="1.0",
    )
    first = _parse_decision(response.text)

    if first["decision"] == "legitimate":
        logger.info("single-call decision: legitimate")
        return {**first, "calls_used": 1}

    # Second pass: self-consistency for the ambiguous cases.
    logger.info("ambiguous case, running self-consistency")
    result = self_consistent_answer(
        prompt=prompt,
        n=5,
        temperature=0.7,
        extract_answer=lambda text: _parse_decision(text)["decision"],
        client=client,
        prompt_name="upi_fraud_classifier",
        prompt_version="1.0",
    )

    return {
        "decision": result.answer,
        "reason": first["reason"],  # Keep the first-call reason for context.
        "calls_used": 1 + result.total_runs,
        "agreement_rate": result.agreement_rate,
        "vote_breakdown": result.all_answers,
    }


def _parse_decision(text: str) -> dict:
    """Parse the model output. Defensive against extra text around the JSON."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return {"decision": "review_needed", "reason": "Could not parse model output"}


def main() -> None:
    samples = [
        {
            "amount_inr": 85,
            "merchant": "Big Basket",
            "merchant_category": "grocery",
            "payee_known_for_days": 230,
            "hour_ist": 19,
            "amount_z_score_vs_user_history": -0.1,
        },
        {
            "amount_inr": 49999,
            "merchant": "Unknown",
            "merchant_category": "p2p",
            "payee_known_for_days": 1,
            "hour_ist": 3,
            "amount_z_score_vs_user_history": 5.2,
        },
        {
            "amount_inr": 8500,
            "merchant": "BookMyShow",
            "merchant_category": "entertainment",
            "payee_known_for_days": 0,
            "hour_ist": 22,
            "amount_z_score_vs_user_history": 1.8,
        },
    ]

    for tx in samples:
        print(f"\nTransaction: ₹{tx['amount_inr']:,} to {tx['merchant']}")
        result = classify_transaction(tx)
        print(f"  Decision: {result['decision']}")
        print(f"  Calls used: {result['calls_used']}")
        if "agreement_rate" in result:
            print(f"  Agreement rate: {result['agreement_rate']:.0%}")


if __name__ == "__main__":
    main()
