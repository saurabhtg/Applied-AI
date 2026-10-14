# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Example: Jio Fiber multi-turn customer support agent with working memory.

Simulates a long troubleshooting session — the kind where the customer
explains their problem, gets asked to try three things that don't work,
then gets escalated to a field visit. By the time the session is 15 turns
long, a naive implementation has forgotten the account number and is
giving generic advice again.

This example shows WorkingMemory keeping the conversation bounded and the
system prompt (pinned) staying intact through compaction.

Key things to watch:
    - Token count printed after each turn.
    - "COMPACTED" when the buffer sweeps old entries out in one go.
    - The system prompt (pinned) is in mem.system() after every sweep.
    - The agent never forgets the customer's account number.

All tool data is mocked. Real API calls to Anthropic for the LLM.
Cost per run is under ₹8.

Run:
    python -m vol3_agents.part14_working_memory.examples.customer_support
"""

from __future__ import annotations

import json
import logging
import os

from anthropic import Anthropic

from vol3_agents.part14_working_memory.working_memory import WorkingMemory

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Mock support tools (called manually — this isn't using ReActAgent)
# ---------------------------------------------------------------------------

_ACCOUNTS = {
    "JF-9921443": {
        "customer": "Vikram Nair",
        "plan": "Jio Fiber Gold (300 Mbps)",
        "status": "Active",
        "address": "Flat 7B, Sunrise Towers, Powai, Mumbai",
        "last_bill_paid": "10 May 2026",
        "outstanding_inr": 0,
    }
}

_TICKET_HISTORY = {
    "JF-9921443": [
        {"date": "12 Apr 2026", "issue": "Intermittent drops in evening hours", "status": "Resolved"},
        {"date": "1 Mar 2026", "issue": "Router replacement", "status": "Resolved"},
    ]
}


def get_account(account_id: str) -> dict:
    return _ACCOUNTS.get(account_id, {"error": "Account not found"})


def get_ticket_history(account_id: str) -> list:
    return _TICKET_HISTORY.get(account_id, [])


def check_line_status(account_id: str) -> dict:
    if account_id == "JF-9921443":
        return {
            "signal_strength": "Good",
            "line_speed_mbps": 12,  # degraded from 300 — hint for the agent
            "line_errors": 847,
            "last_sync": "28 May 2026, 09:41 AM",
            "note": "High error count on line. Possible signal attenuation.",
        }
    return {"error": "Cannot check line for this account."}


def create_field_visit(account_id: str, issue_summary: str) -> dict:
    return {
        "ticket_id": f"FV-{hash(account_id) % 100000:05d}",
        "scheduled": "30 May 2026, 10 AM – 1 PM",
        "technician": "Will be assigned 24 hours before visit",
        "contact": "Customer will receive SMS with technician details",
        "issue_logged": issue_summary,
    }


# ---------------------------------------------------------------------------
# Conversation turns — a realistic troubleshooting session
# ---------------------------------------------------------------------------

CONVERSATION_SCRIPT = [
    # (role, content)
    ("user", "Hi, my internet has been really slow since yesterday evening. "
             "My account ID is JF-9921443."),
    ("user", "I already tried restarting the router twice. Still slow."),
    ("user", "The speed test is showing only 12 Mbps. I'm on the 300 Mbps plan."),
    ("user", "No the router lights are all green. ONT light is also green."),
    ("user", "I checked the cable. It's fine. This is frustrating — I work "
             "from home and this has been happening for almost 2 days now."),
    ("user", "Is there something wrong at my area? My neighbour on the same "
             "floor also has Jio Fiber and his is working fine."),
    ("user", "OK I tried factory reset as you suggested. Still 12 Mbps. "
             "Nothing changed."),
    ("user", "I can't do a test without the router — my laptop doesn't have "
             "an ethernet port. But I don't think it's the router because the "
             "line itself is the issue right?"),
    ("user", "I've been a customer for 3 years. This is unacceptable. Can I "
             "get a technician visit? I need this sorted."),
    ("user", "Yes please escalate. The visit should be before Friday. "
             "I'm available in the morning."),
]

SYSTEM_PROMPT = """\
You are a Jio Fiber customer support agent.

You have access to account data, line diagnostics, and can create field visit tickets.
Your job is to troubleshoot connectivity issues, escalate when remote fixes don't work,
and keep the customer updated.

Account data you've already fetched for this session:
- Customer: Vikram Nair | Account: JF-9921443
- Plan: Jio Fiber Gold (300 Mbps) | Status: Active
- No outstanding dues
- Line diagnostic: Speed 12 Mbps (expected 300), 847 line errors — possible signal attenuation

Past tickets: Two previous issues (April and March 2026), both resolved.

Guidelines:
- Be direct. Don't make the customer repeat information they've already given.
- Troubleshooting steps before escalating: (1) router restart, (2) check lights,
  (3) check cables, (4) factory reset, (5) wired connection test.
- If remote steps fail, offer a field visit. Don't make them wait more than 3 business days.
- Use the customer's name. Keep responses under 3 sentences when possible.
"""


# ---------------------------------------------------------------------------
# Main: run the multi-turn session with WorkingMemory
# ---------------------------------------------------------------------------


def run_session() -> None:
    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    model = "claude-opus-5-5"

    mem = WorkingMemory(
        max_tokens=2_000,   # absurdly tight, so a ten-turn chat shows a sweep
        reserve_tokens=500,
    )

    mem.add("system", SYSTEM_PROMPT, pinned=True)

    print("=" * 70)
    print("  Jio Fiber Support — Part 14: Working Memory")
    print("=" * 70)
    print(f"\n  Token budget: {mem.max_tokens} | Reserve: {mem.reserve_tokens}")
    print(f"  System prompt tokens (pinned): {mem.entries[0].tokens}")
    print()

    for turn_num, (role, content) in enumerate(CONVERSATION_SCRIPT, start=1):
        compactions_before = mem.compactions
        mem.add(role, content)

        response = client.messages.create(
            model=model,
            # Thinking is always on and counts toward max_tokens, so leave
            # room for it as well as the short reply.
            max_tokens=4_000,
            system=mem.system(),
            messages=mem.render(),
            # A support chat doesn't need deep thought on every turn.
            output_config={"effort": "low"},
        )
        # The API's own count beats any estimate. Calibrate before adding the reply.
        mem.observe_usage(response.usage.input_tokens)
        # Read blocks by type: a response can open with a thinking block.
        assistant_text = "".join(b.text for b in response.content if b.type == "text").strip()
        mem.add("assistant", assistant_text)

        marker = "  <- COMPACTED" if mem.compactions > compactions_before else ""
        print(f"Turn {turn_num:2d} | tokens in buffer: {mem.total_tokens:,} "
              f"| entries: {len(mem.entries)} "
              f"| available: {mem.available_tokens:,}{marker}")
        print(f"  [Customer] {content[:80]}{'...' if len(content) > 80 else ''}")
        print(f"  [Agent]    {assistant_text[:120]}{'...' if len(assistant_text) > 120 else ''}")
        print()

    print("=" * 70)
    print("Session complete.")
    print(f"Final buffer: {mem.total_tokens} tokens across {len(mem.entries)} entries")
    print(f"System prompt still pinned: {0 in mem.pinned_indices}")

    print(f"Compactions: {mem.compactions}")
    print("=" * 70)


if __name__ == "__main__":
    run_session()
