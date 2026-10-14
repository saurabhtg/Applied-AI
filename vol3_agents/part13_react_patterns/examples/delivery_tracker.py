# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Example: Meesho delivery and return tracking agent.

A customer support bot for a D2C e-commerce platform. It can track
deliveries, check return status, and request redelivery — using a
ReActAgent with four tools over mock logistics data.

The point isn't the tools. It's watching how the ReAct loop reasons
across multiple steps: first tracking the shipment, then reading the
result, then deciding whether that's enough or whether it needs more
data before answering.

All tool data is mocked. The only real API call is to Anthropic.
Cost per run is under ₹5.

Run:
    python -m vol3_agents.part13_react_patterns.examples.delivery_tracker
"""

from __future__ import annotations

import json
from datetime import date

from vol3_agents.part13_react_patterns.react_loop import AgentResult, ReActAgent, Tool

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_SHIPMENTS = {
    "SHP-4521": {
        "order_id": "ORD-887654",
        "status": "Out for Delivery",
        "carrier": "Delhivery",
        "current_location": "Malad West Hub, Mumbai",
        "estimated_delivery": "Today by 7 PM",
        "last_update": "10:34 AM",
        "attempts": 1,
    },
    "SHP-9982": {
        "order_id": "ORD-662310",
        "status": "Delivery Attempted",
        "carrier": "Ekart Logistics",
        "current_location": "Koramangala Facility, Bengaluru",
        "last_update": "Yesterday, 3:15 PM",
        "attempt_note": "Recipient not available. Door locked.",
        "attempts": 2,
        "next_attempt": "Tomorrow",
    },
    "SHP-1177": {
        "order_id": "ORD-551922",
        "status": "Delivered",
        "carrier": "Shadowfax",
        "delivered_at": "26 May 2026, 2:45 PM",
        "signed_by": "Ramesh (neighbour, Flat 402)",
        "attempts": 1,
    },
    "RSN-4521": {
        "order_id": "ORD-443251",
        "type": "RETURN",
        "status": "Pickup Scheduled",
        "carrier": "Delhivery",
        "pickup_slot": "28 May 2026, 10 AM – 1 PM",
        "item": "Cotton kurta (size M)",
        "reason": "Size too large",
        "refund_status": "Pending pickup",
    },
    "RSN-7730": {
        "order_id": "ORD-228843",
        "type": "RETURN",
        "status": "Return Received at Warehouse",
        "carrier": "Ekart",
        "received_at": "25 May 2026",
        "item": "Wireless earbuds",
        "reason": "Defective product",
        "refund_status": "Refund initiated — ₹849 to original payment method",
        "refund_eta": "3–5 business days",
    },
}

_REDELIVERY_REQUESTS = {}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def track_shipment(tracking_id: str) -> str:
    """Look up current shipment status by tracking ID."""
    data = _SHIPMENTS.get(tracking_id.upper())
    if data is None:
        return json.dumps({
            "error": f"Tracking ID '{tracking_id}' not found. Please verify "
                     f"the ID from your order confirmation SMS or email."
        })
    return json.dumps(data)


def get_order_details(order_id: str) -> str:
    """Fetch order details — item, seller, amount, payment method."""
    order_map = {
        "ORD-887654": {
            "item": "Floral print saree (Beige)",
            "seller": "Rajlaxmi Textiles, Surat",
            "amount_inr": 649,
            "payment": "UPI (Google Pay)",
            "order_date": "23 May 2026",
        },
        "ORD-662310": {
            "item": "Men's denim jacket (XL)",
            "seller": "Style Hub, Delhi",
            "amount_inr": 1299,
            "payment": "COD",
            "order_date": "21 May 2026",
        },
        "ORD-551922": {
            "item": "Kids story books set (6 books)",
            "seller": "Read & Grow, Chennai",
            "amount_inr": 450,
            "payment": "Debit card (SBI)",
            "order_date": "20 May 2026",
        },
        "ORD-443251": {
            "item": "Cotton kurta (size M)",
            "seller": "Ethnix Fashion, Jaipur",
            "amount_inr": 399,
            "payment": "UPI (PhonePe)",
            "order_date": "14 May 2026",
        },
        "ORD-228843": {
            "item": "Wireless earbuds",
            "seller": "TechGadgets, Mumbai",
            "amount_inr": 849,
            "payment": "Credit card (HDFC)",
            "order_date": "10 May 2026",
        },
    }
    data = order_map.get(order_id.upper())
    if data is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})
    return json.dumps(data)


def request_redelivery(tracking_id: str, preferred_date: str) -> str:
    """Request redelivery for a failed delivery attempt."""
    shipment = _SHIPMENTS.get(tracking_id.upper())
    if shipment is None:
        return json.dumps({"error": f"Tracking ID '{tracking_id}' not found."})
    if shipment.get("status") != "Delivery Attempted":
        return json.dumps({
            "error": f"Redelivery can only be requested for shipments with "
                     f"'Delivery Attempted' status. Current status: {shipment['status']}"
        })
    if shipment.get("attempts", 0) >= 3:
        return json.dumps({
            "error": "Maximum delivery attempts (3) reached. Package will be "
                     "returned to seller. Please contact support for a refund."
        })
    _redelivery_requests = _REDELIVERY_REQUESTS
    _redelivery_requests[tracking_id] = preferred_date
    return json.dumps({
        "success": True,
        "message": f"Redelivery requested for {preferred_date}.",
        "confirmation_id": f"REDEL-{hash(tracking_id) % 10000:04d}",
        "note": "You will receive an SMS confirmation within 2 hours.",
    })


def check_return_eligibility(order_id: str) -> str:
    """Check if an order is eligible for return and what the refund timeline is."""
    from datetime import datetime, timedelta

    order_dates = {
        "ORD-887654": "2026-05-23",
        "ORD-662310": "2026-05-21",
        "ORD-551922": "2026-05-20",
        "ORD-443251": "2026-05-14",
    }
    order_date_str = order_dates.get(order_id.upper())
    if order_date_str is None:
        return json.dumps({"error": f"Order '{order_id}' not found."})

    order_date = datetime.strptime(order_date_str, "%Y-%m-%d")
    today = datetime(2026, 5, 28)
    days_since_order = (today - order_date).days
    return_window = 7

    if days_since_order <= return_window:
        return json.dumps({
            "eligible": True,
            "days_since_order": days_since_order,
            "return_window_days": return_window,
            "days_left": return_window - days_since_order,
            "refund_method": "Original payment method",
            "refund_eta": "5–7 business days after pickup",
        })
    return json.dumps({
        "eligible": False,
        "days_since_order": days_since_order,
        "return_window_days": return_window,
        "reason": f"Return window of {return_window} days has expired.",
        "alternative": "If the item is defective, a warranty claim may still be possible.",
    })


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

TOOLS = [
    Tool(
        name="track_shipment",
        description=(
            "Track a shipment or return by its tracking ID "
            "(e.g. SHP-4521 for deliveries, RSN-4521 for returns). "
            "Returns current status, location, carrier, and ETA."
        ),
        fn=track_shipment,
    ),
    Tool(
        name="get_order_details",
        description=(
            "Fetch order details — item name, seller, amount paid, "
            "payment method, and order date — using an order ID (e.g. ORD-887654)."
        ),
        fn=get_order_details,
    ),
    Tool(
        name="request_redelivery",
        description=(
            "Request a new delivery attempt for a shipment whose status is "
            "'Delivery Attempted'. Requires the tracking ID and a preferred "
            "date for redelivery (e.g. '29 May 2026')."
        ),
        fn=request_redelivery,
    ),
    Tool(
        name="check_return_eligibility",
        description=(
            "Check if an order is still within the return window and eligible "
            "for a refund. Returns eligibility status, days remaining, and "
            "refund timeline."
        ),
        fn=check_return_eligibility,
    ),
]

SYSTEM = """\
You are a helpful delivery support agent for Meesho, an Indian e-commerce platform.

You help customers track their orders, understand delivery issues, and manage returns.
Use the available tools to look up real shipment data before answering.

Guidelines:
- Always verify status using the tools. Don't guess or assume.
- Use ₹ for rupee amounts.
- Be direct and specific — customers want facts, not explanations of how logistics works.
- If a delivery failed, tell the customer the exact reason and what their options are.
- For returns, always check the actual refund status, not just the pickup status.
- Keep replies short. One or two paragraphs at most.
"""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _print_result(scenario: str, result: AgentResult) -> None:
    print(f"\n[Customer] {scenario}")
    print(f"\n[Agent]    {result.answer}")
    print(f"\n  Steps: {len(result.steps)}  |  Tokens: {result.total_tokens}  "
          f"|  Stopped: {result.stopped_reason}")
    for step in result.steps:
        if step.action:
            status = "ERR" if step.is_error else "OK"
            args_str = str(step.action_input or {})
            print(f"  [{status}] {step.action}({args_str})")
    print("-" * 70)


def main() -> None:
    agent = ReActAgent(
        tools=TOOLS,
        system=SYSTEM,
        max_steps=8,
        max_tokens_total=40_000,
    )

    print("=" * 70)
    print("  Meesho Delivery Tracker — Part 13: ReAct Patterns")
    print("=" * 70)

    scenarios = [
        "Where is my package? Tracking ID is SHP-4521.",
        "The delivery for SHP-9982 failed yesterday. Can you reschedule it "
        "for 30 May 2026? Also tell me what was in the order.",
        "I have a return shipment RSN-4521. When is the pickup? And when will "
        "I get my refund?",
        "My tracking shows SHP-1177 as delivered but I never received it. "
        "It was signed by someone called Ramesh. What should I do?",
    ]

    for scenario in scenarios:
        result = agent.run(scenario)
        _print_result(scenario, result)


if __name__ == "__main__":
    main()
