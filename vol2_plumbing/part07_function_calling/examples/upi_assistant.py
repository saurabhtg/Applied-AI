# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Example: UPI customer service assistant.

A customer support bot for a fintech app. It can look up transaction
status, list recent transactions, and check daily UPI limits.

All three tools are mocked — no real payment API is called. The only
real API call is to Anthropic. Cost per run is under ₹2.

Run:
    python -m vol2_plumbing.part07_function_calling.examples.upi_assistant
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from vol2_plumbing.part07_function_calling.tool_caller import AgentRun, ToolCaller, ToolDefinition


# ---------------------------------------------------------------------------
# Mock tool implementations
# ---------------------------------------------------------------------------

_KNOWN_TRANSACTIONS = {
    "TXN9834561": {
        "status": "SUCCESS",
        "amount_inr": 1500,
        "to_name": "Suresh Kumar",
        "to_upi": "suresh@okaxis",
        "timestamp": "2026-05-27 14:23",
        "description": "Rent May 2026",
    },
    "TXN4521987": {
        "status": "FAILED",
        "amount_inr": 500,
        "to_name": "Meena Stores",
        "to_upi": "meena@upi",
        "timestamp": "2026-05-27 11:05",
        "failure_reason": "Insufficient funds in source account",
    },
    "TXN7761234": {
        "status": "PENDING",
        "amount_inr": 2000,
        "to_name": "Flipkart",
        "to_upi": "flipkart@okicici",
        "timestamp": "2026-05-28 09:15",
        "description": "Order #FL887234",
    },
}


def get_transaction_status(txn_id: str) -> str:
    txn = _KNOWN_TRANSACTIONS.get(txn_id.upper())
    if txn is None:
        return json.dumps({"error": f"Transaction '{txn_id}' not found. Check the ID and try again."})
    return json.dumps(txn)


def get_recent_transactions(phone: str, days: int = 7) -> str:
    random.seed(phone)
    transactions = []
    names = ["Bigbasket", "Swiggy", "Zepto", "Rahul Verma", "BESCOM (Electricity)", "Amazon Pay"]
    for _ in range(min(days * 2, 8)):
        days_ago = random.randint(0, days - 1)
        transactions.append({
            "txn_id": f"TXN{random.randint(1000000, 9999999)}",
            "amount_inr": random.choice([149, 350, 500, 1200, 1800, 3500]),
            "to": random.choice(names),
            "status": random.choices(["SUCCESS", "FAILED"], weights=[85, 15])[0],
            "date": (datetime.now() - timedelta(days=days_ago)).strftime("%d %b %Y"),
        })
    transactions.sort(key=lambda x: x["date"], reverse=True)
    return json.dumps({"phone": phone, "period_days": days, "transactions": transactions[:6]})


def check_upi_limit(phone: str) -> str:
    random.seed(phone + "_limit")
    used = random.choice([0, 750, 2400, 12000, 28000])
    daily_limit = 100000
    return json.dumps({
        "phone": phone,
        "daily_limit_inr": daily_limit,
        "used_today_inr": used,
        "remaining_inr": daily_limit - used,
        "reset_time": "midnight IST",
    })


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_transaction_status",
        description=(
            "Look up the status and details of a specific UPI transaction "
            "using its transaction ID (format: TXNxxxxxxx). Returns status, "
            "amount, recipient, timestamp, and failure reason if applicable."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "txn_id": {
                    "type": "string",
                    "description": "The UPI transaction ID, e.g. TXN9834561",
                },
            },
            "required": ["txn_id"],
        },
        fn=get_transaction_status,
    ),
    ToolDefinition(
        name="get_recent_transactions",
        description=(
            "Get recent UPI transactions for a customer's phone number. "
            "Returns up to 6 recent transactions with amounts and status."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "10-digit mobile number linked to UPI",
                },
                "days": {
                    "type": "integer",
                    "description": "How many days back to look. Default 7, max 30.",
                },
            },
            "required": ["phone"],
        },
        fn=get_recent_transactions,
    ),
    ToolDefinition(
        name="check_upi_limit",
        description=(
            "Check the daily UPI transaction limit and how much of it has "
            "been used today for a given phone number."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "phone": {
                    "type": "string",
                    "description": "10-digit mobile number linked to UPI",
                },
            },
            "required": ["phone"],
        },
        fn=check_upi_limit,
    ),
]

SYSTEM = """\
You are a helpful UPI support assistant for PayFast, a popular fintech app.

You have tools to look up transaction status, recent transactions, and daily limits.

Guidelines:
- Use ₹ for rupee amounts.
- Keep replies short and direct — customers are usually on their phones.
- If a transaction failed, always mention the reason.
- For pending transactions, tell the customer it can take up to 30 minutes to settle.
- Never make up transaction details. Always call the appropriate tool.
"""


def _print_run(query: str, result: AgentRun) -> None:
    print(f"\n[User] {query}")
    print(f"\n[Assistant] {result.final_text}")
    print(f"\n  Rounds: {result.rounds}  |  Tool calls: {len(result.tool_calls)}")
    for call in result.tool_calls:
        status = "ERROR" if call.is_error else "OK"
        print(f"  → [{status}] {call.tool_name}({call.inputs})")
    print("-" * 68)


def main() -> None:
    caller = ToolCaller(tools=TOOLS)

    print("=" * 68)
    print("  UPI Assistant — Part 7: Function Calling")
    print("=" * 68)

    scenarios = [
        "What happened with my transaction TXN9834561?",
        "My UPI is 9876543210. Show me what I spent last week and also check how much limit I have left today.",
        "Transaction TXN4521987 failed. Why? My number is 9876543210 — can I try again or is my limit exhausted?",
        "I sent ₹2000 to Flipkart — transaction ID TXN7761234. Has it gone through?",
    ]

    for scenario in scenarios:
        result = caller.run(scenario, system=SYSTEM)
        _print_run(scenario, result)


if __name__ == "__main__":
    main()
