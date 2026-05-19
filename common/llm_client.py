# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
common/llm_client.py

A thin wrapper around the Anthropic API. Modify it for other LLM API 
provider(s)

The point is not to abstract the model. The point is to make sure every
single call to the model is logged with the prompt version that ran it,
so when production breaks at 2 a.m. you can answer "which prompt was
running?" in thirty seconds.

If you ever find yourself adding a sixth helper method here, stop and
ask whether you're building a framework. You're probably building a
framework. Don't.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic, APIError, APIStatusError

from common.tracing import span

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 60.0

# Models that don't support the temperature parameter
_MODELS_WITHOUT_TEMPERATURE = {
    "claude-opus-4-7",
    "claude-opus-4-5",
    # Add future models here as needed
}

@dataclass
class LLMResponse:
    """A single response from the model, with metadata for logging and tracing."""

    text: str
    model: str
    prompt_version: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str | None = None
    raw: Any = field(default=None, repr=False)


class LLMClient:
    """
    Thin wrapper around Anthropic's Messages API.

    Every call logs the prompt version and emits a tracing span. Retries
    on transient errors with exponential backoff. Nothing fancy.
    """

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str = DEFAULT_MODEL,
        max_retries: int = 3,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.client = Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        self.default_model = default_model
        self.max_retries = max_retries

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        prompt_version: str | None = None,
        prompt_name: str | None = None,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """
        Send a single-turn message to the model.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            model: Model identifier. Defaults to claude-sonnet-4-6.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. 0.0-1.0.
            prompt_version: Version string for logging. Pull this from
                load_prompt() and pass it through. Critical for debugging.
            prompt_name: Prompt's logical name, also for logging.
            stop_sequences: Optional stop strings.

        Returns:
            An LLMResponse with text, token counts, and latency.
        """
        model = model or self.default_model

        with span(
            "llm.complete",
            attributes={
                "llm.model": model,
                "llm.prompt_name": prompt_name or "unknown",
                "llm.prompt_version": prompt_version or "unversioned",
                "llm.temperature": temperature,
                "llm.max_tokens": max_tokens,
            },
        ) as current_span:
            messages = [{"role": "user", "content": prompt}]
            kwargs: dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            
            # Only include temperature if the model supports it
            if model not in _MODELS_WITHOUT_TEMPERATURE:
                kwargs["temperature"] = temperature
                
            if system:
                kwargs["system"] = system
            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            start = time.perf_counter()
            response = self._call_with_retry(**kwargs)
            latency_ms = (time.perf_counter() - start) * 1000

            text = "".join(
                block.text for block in response.content if block.type == "text"
            )

            current_span.set_attribute("llm.input_tokens", response.usage.input_tokens)
            current_span.set_attribute("llm.output_tokens", response.usage.output_tokens)
            current_span.set_attribute("llm.latency_ms", latency_ms)

            logger.info(
                "llm_call prompt=%s version=%s model=%s in_tokens=%d out_tokens=%d latency_ms=%.1f",
                prompt_name or "unknown",
                prompt_version or "unversioned",
                model,
                response.usage.input_tokens,
                response.usage.output_tokens,
                latency_ms,
            )

            return LLMResponse(
                text=text,
                model=model,
                prompt_version=prompt_version,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_ms=latency_ms,
                stop_reason=response.stop_reason,
                raw=response,
            )

    def _call_with_retry(self, **kwargs: Any) -> Any:
        """Call the API with exponential backoff on transient errors."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(**kwargs)
            except APIStatusError as exc:
                # 4xx errors (except 429) are not transient — fail fast.
                if 400 <= exc.status_code < 500 and exc.status_code != 429:
                    raise
                last_exc = exc
            except APIError as exc:
                last_exc = exc

            wait = 2 ** attempt
            logger.warning(
                "llm_call_retry attempt=%d wait=%ds error=%s",
                attempt + 1,
                wait,
                last_exc,
            )
            time.sleep(wait)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_exc
