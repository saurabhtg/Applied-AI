# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
examples/patient_intake.py

Extract structured patient data from a messy hospital admissions form.

This is the example behind the Part 5 story. A hospital photographs a
handwritten intake form, a model reads it, and the fields drop into the
patient system. The model reads the form fine — the danger is the JSON it
wraps the answer in, and the plausible-but-wrong values that sail through
JSON validation but make no clinical sense.

The input below is deliberately the way a real transcribed form reads:
half-sentences, the doctor's shorthand, an age written next to a
registration number, symptoms in prose rather than a tidy list. The
structured client's job is to turn that into a PatientIntake object that
passes the schema, retrying against the validation error if the first
pass is malformed.

Then — and this is the whole point of the story — we run one extra check
the schema can't: does the extracted age even make sense for an adult
ward? That's the line nobody writes until the day they wish they had.

Run:
    python -m vol1_foundations.part05_structured_outputs.examples.patient_intake
"""

from __future__ import annotations

import logging

from vol1_foundations.part05_structured_outputs.schema import PatientIntake
from vol1_foundations.part05_structured_outputs.structured_client import extract_structured

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# A transcribed admissions form, the way they actually arrive — not a clean
# table. Note the registration number sitting right next to the age, which
# is exactly the trap that produces a valid-but-wrong record.
MESSY_INTAKE_FORM = """\
ADMISSIONS — GENERAL WARD

Patient: Viral Melody
Reg no: 38    Age: 44 / F

Complaints: c/o chest discomfort since morning, mild breathlessness on
exertion, no fever. Known hypertensive.

Ref by: Dr. Anand Kulkarni (Cardiology)
Adm date: 19/05/2026
"""


# Domain guardrail. The schema allows ages 0–120 so paediatric and
# maternity records validate. This adult-ward check is narrower and catches
# the dangerous case: the model filing the registration number (38) where
# the age (54) should have gone. A perfectly valid integer. Clinically wrong.
ADULT_WARD_MIN_AGE = 18
ADULT_WARD_MAX_AGE = 110


def looks_clinically_sane(intake: PatientIntake) -> tuple[bool, str | None]:
    """Cheap domain checks the JSON schema can't express. Returns (ok, reason)."""
    if not (ADULT_WARD_MIN_AGE <= intake.age <= ADULT_WARD_MAX_AGE):
        return False, (
            f"age {intake.age} is outside the plausible adult-ward range "
            f"({ADULT_WARD_MIN_AGE}-{ADULT_WARD_MAX_AGE}) — likely a "
            f"misread field. Flag for human review."
        )
    if not intake.symptoms:
        return False, "no symptoms captured — an admission with no complaint is suspicious."
    return True, None


def main() -> None:
    print("\nExtracting structured patient intake from a messy form...\n")

    result = extract_structured(
        prompt=(
            "Extract the patient admission details from the following intake "
            "form. 'Reg no' is the registration number, NOT the age — do not "
            "confuse them. Capture each complaint as a separate symptom in the "
            "symptoms list.\n\n"
            f"Form:\n{MESSY_INTAKE_FORM}"
        ),
        model_cls=PatientIntake,
        max_attempts=3,
        prompt_name="patient_intake",
        prompt_version="1.0",
    )

    if not result.succeeded:
        print(f"FAILED after {result.attempts} attempts.")
        print(f"Last error:\n{result.last_error}")
        print(f"Last raw response:\n{result.raw_response}")
        return

    intake: PatientIntake = result.value  # type: ignore[assignment]
    print(f"Parsed and schema-valid on attempt {result.attempts}.\n")
    print(f"Patient:        {intake.patient_name}")
    print(f"Age:            {intake.age}")
    print(f"Sex:            {intake.sex}")
    print(f"Ward:           {intake.ward_type.value if intake.ward_type else '—'}")
    print(f"Referring doc:  {intake.referring_doctor}")
    print(f"Admission date: {intake.admission_date}")
    print(f"Symptoms:")
    for s in intake.symptoms:
        print(f"  - {s}")

    # The check the schema can't do. Valid JSON is not the same as a sane
    # record. This is the move that separates people who've been burned from
    # people who are about to be.
    ok, reason = looks_clinically_sane(intake)
    if ok:
        print(f"\n✓ Clinical sanity check passed: age {intake.age} is plausible for an adult ward.")
    else:
        print(f"\n⚠️  Clinical sanity check FAILED: {reason}")


if __name__ == "__main__":
    main()
