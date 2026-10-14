# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Example: D2C brand complaint resolution agent with Reflexion.

A customer returns a pair of Boult Audio earbuds claiming they stopped
working after 38 days. The standard return window is 30 days. But there's
also a 6-month warranty clause — and the agent needs to catch both.

On the first attempt, the agent checks the return window, sees it's expired,
and refuses. Score: 2/5. The reflection identifies: "The agent only checked
the return window. It didn't check warranty eligibility, which is a separate
policy with a 6-month window for defective products."

On the second attempt, the agent checks both policies, correctly identifies
the warranty claim path, and offers a replacement. Score: 5/5.

This is the scenario where Reflexion earns its keep: the first answer is
defensible (return window IS expired), but it's wrong because it ignores
a second applicable policy. The evaluation catches it. The reflection
fixes it.

All tools are mocked. Real API calls to Anthropic.
Cost per run is under ₹10.

Run:
    python -m vol3_agents.part15_reflexion.examples.complaint_resolver
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from vol3_agents.part13_react_patterns.react_loop import ReActAgent, Tool
from vol3_agents.part15_reflexion.reflexion import ReflexionAgent, ReflexionResult

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_ORDERS = {
    "BA-55421": {
        "customer_id": "CUST-8821",
        "customer_name": "Ananya Sharma",
        "item": "Boult Audio Z40 TWS Earbuds",
        "amount_inr": 1299,
        "order_date": "2026-04-18",
        "delivery_date": "2026-04-20",
        "payment": "UPI (Paytm)",
        "category": "Electronics",
    },
    "BA-71093": {
        "customer_id": "CUST-4412",
        "customer_name": "Rohit Mehta",
        "item": "Boult Audio Curve Neckband",
        "amount_inr": 699,
        "order_date": "2026-05-01",
        "delivery_date": "2026-05-04",
        "payment": "COD",
        "category": "Electronics",
    },
    "BA-38902": {
        "customer_id": "CUST-6630",
        "customer_name": "Sunita Joshi",
        "item": "Boult Audio BassBuds X1 (Blue)",
        "amount_inr": 449,
        "order_date": "2026-05-20",
        "delivery_date": "2026-05-22",
        "payment": "Debit card (SBI)",
        "category": "Electronics",
    },
}

_CUSTOMER_HISTORY = {
    "CUST-8821": {
        "total_orders": 4,
        "returns_requested": 0,
        "is_premium": False,
        "account_since": "2024-11",
    },
    "CUST-4412": {
        "total_orders": 1,
        "returns_requested": 0,
        "is_premium": False,
        "account_since": "2026-05",
    },
    "CUST-6630": {
        "total_orders": 12,
        "returns_requested": 1,
        "is_premium": True,
        "account_since": "2023-08",
    },
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def get_order_details(order_id: str) -> str:
    data = _ORDERS.get(order_id.upper())
    if data is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    return json.dumps(data)


def check_return_policy(order_id: str) -> str:
    """Check standard return window eligibility (not warranty)."""
    from datetime import datetime

    order = _ORDERS.get(order_id.upper())
    if order is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    delivery_date = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
    today = datetime(2026, 5, 28)
    days_since_delivery = (today - delivery_date).days
    return_window = 30

    if days_since_delivery <= return_window:
        return json.dumps({
            "policy": "standard_return",
            "eligible": True,
            "days_since_delivery": days_since_delivery,
            "window_days": return_window,
            "days_remaining": return_window - days_since_delivery,
            "resolution": "Full refund to original payment method",
        })
    return json.dumps({
        "policy": "standard_return",
        "eligible": False,
        "days_since_delivery": days_since_delivery,
        "window_days": return_window,
        "reason": f"Standard 30-day return window expired {days_since_delivery - return_window} day(s) ago.",
    })


def check_warranty_policy(order_id: str, complaint: str) -> str:
    """Check warranty eligibility for defective products (separate from return window)."""
    from datetime import datetime

    order = _ORDERS.get(order_id.upper())
    if order is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    delivery_date = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
    today = datetime(2026, 5, 28)
    days_since_delivery = (today - delivery_date).days
    warranty_days = 180  # 6 months

    defect_keywords = ["defective", "not working", "stopped", "broken",
                       "malfunction", "dead", "no sound", "crackling"]
    is_defect_claim = any(kw in complaint.lower() for kw in defect_keywords)

    if not is_defect_claim:
        return json.dumps({
            "policy": "warranty",
            "eligible": False,
            "reason": "Warranty covers manufacturing defects only. "
                      "The complaint does not appear to describe a defect.",
        })

    if days_since_delivery <= warranty_days:
        return json.dumps({
            "policy": "warranty",
            "eligible": True,
            "days_since_delivery": days_since_delivery,
            "warranty_days": warranty_days,
            "days_remaining": warranty_days - days_since_delivery,
            "resolution": "Replacement unit (same model) dispatched within 5 business days. "
                          "Customer keeps defective unit OR ships it back with prepaid label.",
            "note": "Warranty replacement does not require the standard 30-day return window.",
        })
    return json.dumps({
        "policy": "warranty",
        "eligible": False,
        "days_since_delivery": days_since_delivery,
        "warranty_days": warranty_days,
        "reason": "6-month warranty period has expired.",
    })


def get_customer_history(customer_id: str) -> str:
    data = _CUSTOMER_HISTORY.get(customer_id)
    if data is None:
        return json.dumps({"error": f"Customer '{customer_id}' not found."})
    return json.dumps(data)


def process_resolution(order_id: str, resolution_type: str, notes: str) -> str:
    """Record and process the final resolution decision."""
    valid_types = ["full_refund", "warranty_replacement", "partial_refund",
                   "reject", "escalate_to_manager"]
    if resolution_type not in valid_types:
        return json.dumps({
            "error": f"Invalid resolution type. Must be one of: {valid_types}"
        })
    return json.dumps({
        "success": True,
        "order_id": order_id,
        "resolution": resolution_type,
        "ticket_id": f"CR-{hash(order_id + resolution_type) % 100000:05d}",
        "notes": notes,
        "customer_communication": "Automated email + SMS will be sent within 10 minutes.",
    })


# ---------------------------------------------------------------------------
# Agent configuration
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="get_order_details",
        description=(
            "Fetch order details — item name, customer, amount, delivery date, "
            "and product category — using an order ID (e.g. BA-55421)."
        ),
        fn=get_order_details,
    ),
    Tool(
        name="check_return_policy",
        description=(
            "Check if the order is within the standard 30-day return window. "
            "Use this for 'change of mind', 'wrong item', or 'size issue' complaints. "
            "Does NOT cover defective product claims."
        ),
        fn=check_return_policy,
    ),
    Tool(
        name="check_warranty_policy",
        description=(
            "Check if the order is eligible for warranty replacement. "
            "Use this when the customer reports a defect, malfunction, or product "
            "stopping working. The warranty window is 6 months from delivery, "
            "separate from the 30-day return window."
        ),
        fn=check_warranty_policy,
    ),
    Tool(
        name="get_customer_history",
        description=(
            "Get the customer's order and return history, and whether they are "
            "a premium member. Use the customer_id from order details."
        ),
        fn=get_customer_history,
    ),
    Tool(
        name="process_resolution",
        description=(
            "Record and process the final resolution decision for a complaint. "
            "resolution_type must be one of: full_refund, warranty_replacement, "
            "partial_refund, reject, escalate_to_manager. Include a brief note "
            "explaining the decision."
        ),
        fn=process_resolution,
    ),
]

AGENT_SYSTEM = """\
You are a customer support resolution agent for Boult Audio, an Indian audio brand.

Your job: investigate customer complaints, apply the correct policy, and resolve them.

Policies:
1. Standard return: 30 days from delivery. Any reason. Full refund.
2. Warranty: 6 months from delivery. Defective products only. Replacement (not refund).
3. Premium customers: escalate to manager if policy doesn't cover their case.
4. COD orders with no prior returns: no restocking fee on first return.

Always check BOTH return and warranty eligibility for defective product complaints.
One policy might not apply, but the other might. Don't stop at the first "no".

When you have a clear decision, call process_resolution to log it.
"""

EVALUATION_CRITERIA = """\
1. Did the agent check ALL applicable policies (return window AND warranty)?
2. Is the resolution correct based on the applicable policy?
3. Did the agent check customer history if needed (premium status)?
4. Was the resolution logged via process_resolution?
5. Is the explanation clear enough for the customer to understand?
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_reflexion_result(complaint: str, result: ReflexionResult) -> None:
    print(f"\n[Complaint] {complaint}")
    print()

    for attempt in result.attempts:
        status_icon = "PASS" if attempt.passed else "FAIL"
        print(f"  Attempt {attempt.attempt_number} [{status_icon}]  "
              f"Score: {attempt.score:.0f}/5  |  "
              f"Steps: {attempt.agent_steps}  |  "
              f"Tokens: {attempt.agent_tokens:,}")
        for c in attempt.criteria:
            mark = "x" if c["met"] else " "
            print(f"    [{mark}] {c['criterion'][:70]}")
        print(f"  Critique: {attempt.critique}")
        if attempt.reflection:
            print(f"  Reflection: {attempt.reflection[:120]}...")
        print()

    final_label = "SUCCEEDED" if result.succeeded else "BEST ATTEMPT"
    print(f"  [{final_label}] Final answer:")
    for line in result.final_answer.split("\n"):
        if line.strip():
            print(f"  {line}")
    print(f"\n  Total tokens: {result.total_tokens:,}")
    print("-" * 70)


def main() -> None:
    base_agent = ReActAgent(
        tools=TOOLS,
        max_steps=10,
        max_tokens_total=30_000,
    )
    agent = ReflexionAgent(
        base_agent=base_agent,
        evaluation_criteria=EVALUATION_CRITERIA,
        max_attempts=3,
        min_score=4.0,
    )

    print("=" * 70)
    print("  Boult Audio Complaint Resolver — Part 15: Reflexion")
    print("=" * 70)

    complaints = [
        # Warranty case — agent must check both policies
        (
            "Resolve complaint for order BA-55421. Customer says: "
            "My Z40 earbuds completely stopped working after 38 days. "
            "No sound from the right earbud at all. Defective product."
        ),
        # Clean return — should be straightforward
        (
            "Resolve complaint for order BA-38902. Customer says: "
            "I received the wrong colour. I ordered blue but got black. "
            "Customer is Sunita Joshi, premium member."
        ),
        # Borderline case — within return window, no defect claim
        (
            "Resolve complaint for order BA-71093. Customer says: "
            "I changed my mind. I don't like the sound quality. "
            "I want a full refund."
        ),
    ]

    for complaint in complaints:
        result = agent.run(complaint)
        _print_reflexion_result(complaint, result)


if __name__ == "__main__":
    main()
