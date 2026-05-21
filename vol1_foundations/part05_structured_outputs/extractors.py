# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol1_foundations/part05_structured_outputs/extractors.py

Two cheap tricks that get you clean JSON on the first try more often, so
the retry loop in structured_client.py fires less.

Trick 1 — Prefill the opening brace.
    If you start the assistant's turn with "{", the model has no room to
    write "Sure! Here's the JSON:". It's already inside the object. This
    is the single highest-leverage line of code in structured extraction
    and it costs nothing.

Trick 2 — Stop sequence on the closing brace.
    If you tell the model to stop generating at "}", you cut off any
    trailing chatter before it starts. Combined with prefill, you get a
    response that is *only* the JSON object, no wrappers to strip.

Both are wrappers around the same LLMClient.complete you've used
everywhere else. Nothing magic. The magic is knowing they exist before
you waste a week blaming the model.
"""

from __future__ import annotations

import json
from typing import Any

from common.llm_client import LLMClient
from vol1_foundations.part05_structured_outputs.schema import parse_json_lenient


def extract_with_prefill(
    prompt: str,
    *,
    client: LLMClient | None = None,
    system: str | None = None,
    model: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> Any:
    """
    Extract JSON using the prefill trick.

    We send the prompt, then tell the client we want a JSON object back.
    The thin LLMClient in this repo is single-turn and doesn't expose an
    assistant-prefill parameter directly, so we simulate the effect by
    instructing hard and parsing leniently. In your own client, prefer a
    true assistant prefill of "{" — it's stronger.

    Returns the parsed JSON (dict/list). Raises json.JSONDecodeError if
    the response can't be parsed even after repair.
    """
    client = client or LLMClient()
    hardened = (
        f"{prompt}\n\n"
        "Output ONLY a JSON object. Begin your response with the character "
        "{ and end it with }. No other text."
    )
    response = client.complete(
        prompt=hardened,
        system=system,
        model=model,
        temperature=0.0,
        prompt_name=prompt_name or "extract_prefill",
        prompt_version=prompt_version or "1.0",
    )
    return parse_json_lenient(response.text)


def extract_with_stop_sequence(
    prompt: str,
    *,
    client: LLMClient | None = None,
    system: str | None = None,
    model: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
) -> Any:
    """
    Extract JSON using a stop sequence on the closing brace.

    By stopping generation at the first "}", we cut off trailing chatter.
    We then re-append the "}" we stopped before (the stop sequence is not
    included in the output) and parse.

    Caveat: this only works for a single flat object with no nested
    objects, because the first "}" would end a nested object too early.
    For nested schemas, use extract_structured (validate-and-retry) or a
    true tool/function-calling interface instead. Know the limits of your
    tricks.
    """
    client = client or LLMClient()
    response = client.complete(
        prompt=f"{prompt}\n\nRespond with only a flat JSON object.",
        system=system,
        model=model,
        temperature=0.0,
        stop_sequences=["}"],
        prompt_name=prompt_name or "extract_stop_seq",
        prompt_version=prompt_version or "1.0",
    )
    # The stop sequence isn't in the output, so add the brace back.
    text = response.text
    if "{" in text and "}" not in text:
        text = text + "}"
    return parse_json_lenient(text)
