# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part05_structured_outputs/schema.py

Schemas for structured extraction, plus the unglamorous repair code that
makes model-generated JSON actually parse.

Here's the thing nobody tells you when you start: the model will get the
JSON *almost* right, almost every time. It will wrap it in a markdown
fence. It will add a chatty preamble ("Sure! Here's the JSON you asked
for:"). It will use single quotes because it saw Python somewhere in its
training data. It will add a trailing comma. It will write the literal
word None instead of null because, again, Python.

None of these are model failures exactly. They're the model being helpful
in a way that breaks json.loads(). So before you reach for a bigger model
or a cleverer prompt, reach for the thirty lines of repair code in this
file. They fix the boring 95% of breakages for free.

The schemas themselves use Pydantic, which gives you validation for free:
type coercion, required-field checks, range constraints, and a readable
error when the model hands you something that doesn't fit. When validation
fails, you feed the error back to the model and let it try again — that
loop lives in structured_client.py.
"""

from __future__ import annotations

import json
import re
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# -----------------------------------------------------------------------------
# Example schemas. These are deliberately concrete — invoice line items and
# a hospital patient intake — because abstract "Item" schemas teach you
# nothing.
# -----------------------------------------------------------------------------

class Currency(str, Enum):
    INR = "INR"
    USD = "USD"
    EUR = "EUR"


class InvoiceLineItem(BaseModel):
    """One line on a vendor invoice."""

    description: str
    quantity: int = Field(ge=1)
    unit_price: float = Field(ge=0)
    line_total: float = Field(ge=0)


class Invoice(BaseModel):
    """A vendor invoice extracted from a PDF or email body."""

    vendor_name: str
    invoice_number: str
    invoice_date: date | None = None
    currency: Currency = Currency.INR
    line_items: list[InvoiceLineItem] = Field(default_factory=list)
    subtotal: float = Field(ge=0)
    gst_amount: float = Field(ge=0, default=0.0)
    total: float = Field(ge=0)

    @field_validator("invoice_number")
    @classmethod
    def _strip_invoice_number(cls, v: str) -> str:
        return v.strip()


class WardType(str, Enum):
    GENERAL = "general"
    ICU = "icu"
    MATERNITY = "maternity"
    PEDIATRIC = "pediatric"


class PatientIntake(BaseModel):
    """
    Structured fields pulled from a hospital admissions intake form.

    This is the schema behind the Part 5 story: a hospital photographs the
    handwritten intake form, a model reads it, and these fields drop into
    the patient system. The model reads the form correctly; the danger is
    the JSON it wraps the answer in — and, worse, plausible-but-wrong values
    that pass JSON parsing but make no clinical sense.

    Note the `age` field. The schema allows a wide 0–120 range so paediatric
    and maternity records validate. The narrow, domain-specific check — "an
    adult-ward patient's age should be 18–110" — is the one dumb, vital
    guardrail that catches the model filing a registration number (38) where
    the age (54) should have gone. JSON validity will never catch that. A
    range check will. That check lives in the example (patient_intake.py),
    not here, because different wards have different plausible ranges and you
    don't bake one ward's rule into the shared model.
    """

    patient_name: str
    age: int = Field(ge=0, le=120)
    sex: str | None = None  # M | F | other | unspecified
    symptoms: list[str] = Field(default_factory=list)
    ward_type: WardType | None = None
    referring_doctor: str | None = None
    admission_date: date | None = None

    @field_validator("patient_name")
    @classmethod
    def _name_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("patient_name must not be empty")
        return v.strip()


# -----------------------------------------------------------------------------
# The repair code. This is the part that earns its keep.
# -----------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def extract_json(text: str) -> str:
    """
    Pull the JSON substring out of a model response.

    Handles the three most common wrappers:
        1. Markdown fences: ```json ... ```
        2. Chatty preamble before the JSON.
        3. Trailing chatter after the JSON.

    Returns the best-guess JSON string. Does NOT parse it — that's the
    caller's job, so they can decide what to do on failure.
    """
    # 1. If there's a fenced block, that's almost certainly the payload.
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        return fence_match.group(1).strip()

    # 2. No fence — find the outermost {...} or [...] span.
    #    We scan for the first opening bracket and the last matching close.
    start = _first_bracket(text)
    if start == -1:
        return text.strip()

    open_char = text[start]
    close_char = "}" if open_char == "{" else "]"
    end = text.rfind(close_char)
    if end == -1 or end < start:
        return text[start:].strip()

    return text[start:end + 1].strip()


def _first_bracket(text: str) -> int:
    """Index of the first { or [ in the text, or -1."""
    candidates = [i for i in (text.find("{"), text.find("[")) if i != -1]
    return min(candidates) if candidates else -1


def repair_json(raw: str) -> str:
    """
    Apply cheap, deterministic fixes to almost-valid JSON.

    These are the breakages we see in the wild, in rough order of
    frequency:
        - Trailing commas before } or ]
        - Python literals (None / True / False) instead of JSON ones
        - Smart quotes that should be straight quotes

    This is intentionally NOT a full JSON parser-and-fixer. It handles the
    boring common cases. Anything weirder than this should fail loudly and
    go back to the model via the retry loop, not get silently mangled here.
    """
    fixed = raw

    # Smart quotes → straight quotes. The model picks these up from prose.
    fixed = (
        fixed.replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2018", "'").replace("\u2019", "'")
    )

    # Python literals → JSON literals. Word-boundary so we don't touch
    # "None" inside a string value like "NoneSuch Pvt Ltd".
    fixed = re.sub(r"\bNone\b", "null", fixed)
    fixed = re.sub(r"\bTrue\b", "true", fixed)
    fixed = re.sub(r"\bFalse\b", "false", fixed)

    # Trailing commas before a close bracket.
    fixed = _TRAILING_COMMA_RE.sub(r"\1", fixed)

    return fixed


def parse_json_lenient(text: str) -> Any:
    """
    Extract, then parse, then (if that fails) repair-and-parse.

    Raises json.JSONDecodeError if even the repaired version won't parse.
    The caller is expected to catch that and either retry against the
    model or give up gracefully.
    """
    candidate = extract_json(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # One repair attempt, then let it raise.
        return json.loads(repair_json(candidate))
