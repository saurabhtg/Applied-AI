# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part05_structured_outputs/structured_client.py

Ask the model for JSON, parse it, validate it against a schema, and — when
validation fails — feed the error back and let the model fix its own
mistake. Bounded retries, because an infinite loop is just a slower way to
go bankrupt.

This is the single most useful 80 lines in this whole repo for anyone doing
extraction or classification in production. The pattern:

    1. Send the prompt, with the JSON schema embedded so the model knows
       the shape you want.
    2. Parse the response leniently (strip fences, repair common breakage).
    3. Validate against a Pydantic model.
    4. If validation fails, send the validation error back to the model
       and ask it to correct exactly those fields. Repeat up to N times.
    5. If it still fails after N tries, raise — don't silently return
       garbage.

Most calls succeed on attempt 1. The retry exists for the long tail, and
the long tail is exactly what wakes you up at 2 a.m. if you don't handle
it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

from common.llm_client import LLMClient
from common.tracing import span
from vol1_foundations.part05_structured_outputs.schema import parse_json_lenient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    """The outcome of a structured-extraction call."""

    value: BaseModel | None
    attempts: int
    succeeded: bool
    last_error: str | None = None
    raw_response: str | None = None


def _schema_instructions(model_cls: Type[BaseModel]) -> str:
    """
    Build the 'respond in this exact JSON shape' instruction from a
    Pydantic model's JSON schema.

    We hand the model the actual JSON schema rather than describing the
    fields in prose. The model is much better at following a schema than
    a paragraph, and there's no gap between what you asked for and what
    you'll validate against — they're literally the same object.
    """
    schema = model_cls.model_json_schema()
    return (
        "Respond with a single valid JSON object and nothing else. "
        "No markdown fences, no commentary, no preamble. "
        "The JSON must conform to this schema:\n\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "If a value is genuinely unknown or not present in the input, use "
        "null rather than guessing."
    )


def extract_structured(
    prompt: str,
    model_cls: Type[T],
    *,
    client: LLMClient | None = None,
    system: str | None = None,
    max_attempts: int = 3,
    model: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> StructuredResult:
    """
    Run the ask → parse → validate → retry loop.

    Args:
        prompt: The task prompt (what to extract, plus the input text).
            The schema instructions are appended automatically.
        model_cls: A Pydantic model class. The model's output is validated
            against this.
        client: Optional LLM client.
        system: Optional system prompt.
        max_attempts: How many times to try before giving up. 3 is plenty;
            if it can't produce valid JSON in 3 tries against a fed-back
            error, a 4th won't save you.
        model: Optional model override.
        prompt_name / prompt_version: For logging, as everywhere else.

    Returns:
        A StructuredResult. Check .succeeded before trusting .value.
    """
    client = client or LLMClient()
    schema_block = _schema_instructions(model_cls)

    # The conversation grows by one correction turn per failed attempt.
    conversation = f"{prompt}\n\n{schema_block}"
    last_error: str | None = None
    last_raw: str | None = None

    with span(
        "structured.extract",
        attributes={
            "structured.model_cls": model_cls.__name__,
            "structured.max_attempts": max_attempts,
        },
    ) as s:
        for attempt in range(1, max_attempts + 1):
            response = client.complete(
                prompt=conversation,
                system=system,
                model=model,
                temperature=0.0,  # extraction wants determinism, not creativity
                prompt_name=prompt_name or f"structured_{model_cls.__name__}",
                prompt_version=prompt_version or "1.0",
            )
            last_raw = response.text

            try:
                data = parse_json_lenient(response.text)
            except json.JSONDecodeError as exc:
                last_error = f"Response was not valid JSON: {exc}"
                logger.warning("attempt %d: JSON parse failed: %s", attempt, exc)
                conversation = _correction_turn(conversation, response.text, last_error)
                continue

            try:
                value = model_cls.model_validate(data)
                s.set_attribute("structured.attempts", attempt)
                s.set_attribute("structured.succeeded", True)
                return StructuredResult(
                    value=value,
                    attempts=attempt,
                    succeeded=True,
                    raw_response=response.text,
                )
            except ValidationError as exc:
                last_error = _format_validation_error(exc)
                logger.warning("attempt %d: schema validation failed:\n%s", attempt, last_error)
                conversation = _correction_turn(conversation, response.text, last_error)

        s.set_attribute("structured.attempts", max_attempts)
        s.set_attribute("structured.succeeded", False)
        return StructuredResult(
            value=None,
            attempts=max_attempts,
            succeeded=False,
            last_error=last_error,
            raw_response=last_raw,
        )


def _correction_turn(prior_prompt: str, bad_response: str, error: str) -> str:
    """
    Build the next prompt by appending the model's wrong answer and the
    specific error, asking it to fix exactly that.

    Telling the model *what* was wrong is the whole trick. "Try again"
    gets you the same mistake. "Field 'total' must be >= 0, you returned
    -50" gets you a fix.
    """
    return (
        f"{prior_prompt}\n\n"
        f"--- Your previous response ---\n{bad_response}\n\n"
        f"--- The problem ---\n{error}\n\n"
        f"Correct ONLY the problem above and return the full valid JSON object again."
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic ValidationError into a short, model-readable message."""
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        lines.append(f"- field '{loc}': {err['msg']}")
    return "Schema validation failed:\n" + "\n".join(lines)
