# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Example: Loan eligibility verification with parallel tool calls.

A loan officer assistant at Shikhar Finance that verifies three things
before recommending approval on a ₹15L personal loan:
  1. CIBIL score (350ms simulated latency)
  2. ITR income verification (420ms)
  3. Existing EMI obligations (280ms)

All three are independent. With serial execution: ~1,050ms.
With parallel execution: ~420ms (the slowest of the three).

Run:
    python -m vol2_plumbing.part08_parallel_tool_calls.examples.loan_verifier
"""

from __future__ import annotations

import json
import time

from vol2_plumbing.part07_function_calling.tool_caller import ToolCaller, ToolDefinition
from vol2_plumbing.part08_parallel_tool_calls.parallel_runner import (
    ParallelAgentRun,
    ParallelToolRunner,
)


# ---------------------------------------------------------------------------
# Mock tool implementations — simulate real latency from external APIs
# ---------------------------------------------------------------------------

_CIBIL_DATA = {
    "ABCDE1234F": {"score": 742, "rating": "Good", "history_years": 8, "defaults": 0},
    "XYZAB9876K": {"score": 612, "rating": "Fair", "history_years": 3, "defaults": 1},
    "LMNOP5678Q": {"score": 803, "rating": "Excellent", "history_years": 14, "defaults": 0},
}

_ITR_DATA = {
    "ABCDE1234F": {"gross_income_inr": 1_450_000, "filing_status": "FILED", "ay": "2024-25"},
    "XYZAB9876K": {"gross_income_inr": 480_000, "filing_status": "FILED", "ay": "2024-25"},
    "LMNOP5678Q": {"gross_income_inr": 2_850_000, "filing_status": "FILED", "ay": "2024-25"},
}

_EMI_DATA = {
    "ABCDE1234F": {
        "total_monthly_emi_inr": 22_000,
        "loans": [
            {"type": "Home Loan", "emi_inr": 18_000, "remaining_months": 84},
            {"type": "Car Loan", "emi_inr": 4_000, "remaining_months": 12},
        ],
    },
    "XYZAB9876K": {
        "total_monthly_emi_inr": 8_500,
        "loans": [{"type": "Personal Loan", "emi_inr": 8_500, "remaining_months": 6}],
    },
    "LMNOP5678Q": {"total_monthly_emi_inr": 0, "loans": []},
}


def get_cibil_score(pan: str) -> str:
    """Fetch CIBIL credit score. Real call takes ~350ms."""
    time.sleep(0.35)
    data = _CIBIL_DATA.get(pan.upper(), {"score": 680, "rating": "Good", "history_years": 5, "defaults": 0})
    return json.dumps({"pan": pan, **data})


def verify_income_itr(pan: str, assessment_year: str = "2024-25") -> str:
    """Verify income from ITR. Real call takes ~420ms."""
    time.sleep(0.42)
    data = _ITR_DATA.get(pan.upper(), {"gross_income_inr": 720_000, "filing_status": "FILED", "ay": assessment_year})
    return json.dumps({"pan": pan, **data})


def get_existing_emis(pan: str) -> str:
    """Check existing loan EMI obligations. Real call takes ~280ms."""
    time.sleep(0.28)
    data = _EMI_DATA.get(pan.upper(), {"total_monthly_emi_inr": 5_000, "loans": []})
    return json.dumps({"pan": pan, **data})


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[ToolDefinition] = [
    ToolDefinition(
        name="get_cibil_score",
        description=(
            "Fetch the CIBIL credit score and credit history for a loan applicant. "
            "Returns score (300–900), rating, years of credit history, and defaults count."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pan": {"type": "string", "description": "Applicant's PAN number, e.g. ABCDE1234F"},
            },
            "required": ["pan"],
        },
        fn=get_cibil_score,
    ),
    ToolDefinition(
        name="verify_income_itr",
        description=(
            "Verify the applicant's annual income from their Income Tax Return filing. "
            "Returns gross income, filing status, and assessment year."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pan": {"type": "string", "description": "Applicant's PAN number"},
                "assessment_year": {
                    "type": "string",
                    "description": "ITR assessment year, e.g. 2024-25. Defaults to latest.",
                },
            },
            "required": ["pan"],
        },
        fn=verify_income_itr,
    ),
    ToolDefinition(
        name="get_existing_emis",
        description=(
            "Fetch the applicant's existing loan EMI obligations. Returns total monthly "
            "EMI and a breakdown by loan type."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "pan": {"type": "string", "description": "Applicant's PAN number"},
            },
            "required": ["pan"],
        },
        fn=get_existing_emis,
    ),
]

SYSTEM = """\
You are a loan officer assistant at Shikhar Finance, an NBFC serving salaried professionals.

Loan requested: ₹15,00,000 (15 lakhs) personal loan over 5 years.
Approximate EMI at 10.5% for 60 months = ₹32,238 per month.

Eligibility rules:
- CIBIL score >= 700: Approved. 650–699: Conditional approval (co-applicant or higher margin).
  Below 650: Declined.
- Monthly income: Total EMI burden (existing + new) must not exceed 50% of gross monthly income.
- ITR must be filed for the latest assessment year.

For each applicant, call all three verification tools, then give a clear APPROVE / CONDITIONAL /
DECLINE recommendation with your reasoning. Use ₹ for amounts and lakh/crore notation.
Keep your response to 4–6 lines.
"""


def _fmt_inr(amount: int) -> str:
    if amount >= 10_00_000:
        return f"₹{amount / 10_00_000:.1f}Cr"
    if amount >= 1_00_000:
        return f"₹{amount / 1_00_000:.1f}L"
    return f"₹{amount:,}"


def main() -> None:
    applicants = [
        ("ABCDE1234F", "Rajesh Kumar", "Senior Software Engineer, Bengaluru"),
        ("XYZAB9876K", "Priti Sharma", "Freelance UX Consultant, Pune"),
        ("LMNOP5678Q", "Vikram Nair", "VP Operations, Gurugram"),
    ]

    parallel_runner = ParallelToolRunner(tools=TOOLS)
    serial_runner = ToolCaller(tools=TOOLS)

    print("=" * 70)
    print("  Loan Verification — Part 8: Parallel Tool Calls")
    print("  Loan: ₹15,00,000 | 5 years | 10.5% p.a. | EMI: ₹32,238/month")
    print("=" * 70)

    for pan, name, profile in applicants:
        print(f"\n{'─' * 70}")
        print(f"  Applicant : {name}")
        print(f"  Profile   : {profile}")
        print(f"  PAN       : {pan}")
        print(f"{'─' * 70}")

        query = (
            f"Assess loan eligibility for PAN {pan}. "
            "Fetch CIBIL score, income from ITR, and existing EMIs in parallel, "
            "then give your recommendation."
        )

        # Parallel run
        t0 = time.perf_counter()
        result: ParallelAgentRun = parallel_runner.run(query, system=SYSTEM)
        wall_ms = (time.perf_counter() - t0) * 1000

        print(f"\n[Decision]\n{result.final_text}")

        print(f"\n[Timing]")
        print(f"  Wall time        : {wall_ms:.0f}ms")
        if result.serial_equivalent_ms > 0:
            print(f"  Serial equivalent: {result.serial_equivalent_ms:.0f}ms")
            print(f"  Time saved       : {result.time_saved_ms:.0f}ms ({result.speedup:.1f}x faster)")

        print(f"\n[Tool calls]")
        for call in result.tool_calls:
            status = "ERR" if call.is_error else " OK"
            print(f"  [{status}] {call.tool_name}(pan={pan})")


if __name__ == "__main__":
    main()
