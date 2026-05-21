# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
Part 5 — The JSON Straitjacket

Forcing a model to speak structured data without it losing its mind
(or its schema). This part ships:
    - schema.py: Pydantic models (PatientIntake, Invoice) plus the repair
      helpers (extract_json, repair_json, parse_json_lenient) that fix the
      common ways model JSON arrives broken.
    - structured_client.py: a wrapper that asks for JSON, parses it,
      validates it, and retries with the validation error fed back in.
    - extractors.py: prefill and stop-sequence extraction helpers.
"""
