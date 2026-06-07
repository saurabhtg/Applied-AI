# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
part07_function_calling/tool_caller.py

The agentic loop for function calling.

The model responds with tool_use blocks. You execute the tools. You
tell the model what happened. It decides what to do next.
Keep going until stop_reason == "end_turn".

The pattern:
    1. Send message + tool definitions to the API.
    2. Model responds with tool_use blocks (+ optional text).
    3. Execute every requested tool.
    4. Send all tool results back in one user message.
    5. Repeat until the model is done.

Nothing here is novel. The discipline is in the loop.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from anthropic import Anthropic

from common.tracing import span

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ROUNDS = 10


@dataclass
class ToolDefinition:
    """A callable with the metadata the model needs to use it."""

    name: str
    description: str
    input_schema: dict  # JSON Schema: {"type": "object", "properties": {...}, "required": [...]}
    fn: Callable[..., Any]

    def to_api_spec(self) -> dict:
        """Format this tool for the Anthropic tools array."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolCallResult:
    """The outcome of one tool execution."""

    tool_use_id: str
    tool_name: str
    inputs: dict
    output: str
    is_error: bool = False


@dataclass
class AgentRun:
    """The complete result of a tool-calling conversation."""

    final_text: str
    rounds: int  # number of LLM calls made
    tool_calls: list[ToolCallResult] = field(default_factory=list)


class ToolCaller:
    """
    Runs the tool-calling loop against the Anthropic Messages API.

    One "round" is one model call. A round ends when the model either
    stops (stop_reason == "end_turn") or asks for tool(s) (stop_reason
    == "tool_use"). If it asks for tools, execute them and loop.

    Usage:
        caller = ToolCaller(tools=[my_tool_1, my_tool_2])
        result = caller.run("What is the EMI for ₹5L at 10% for 3 years?")
        print(result.final_text)
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        api_key: str | None = None,
    ):
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._tools = {t.name: t for t in tools}
        self._api_specs = [t.to_api_spec() for t in tools]
        self.model = model
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds

    def run(
        self,
        user_message: str,
        *,
        system: str | None = None,
    ) -> AgentRun:
        """
        Run the tool-calling loop.

        Returns an AgentRun with the final text and a log of every tool
        call made along the way.
        """
        with span("tool_caller.run", attributes={"tool_caller.model": self.model}) as s:
            messages: list[dict] = [{"role": "user", "content": user_message}]
            all_tool_calls: list[ToolCallResult] = []

            for round_num in range(self.max_rounds):
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "tools": self._api_specs,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = self._client.messages.create(**kwargs)
                logger.info(
                    "tool_caller_round round=%d stop_reason=%s in=%d out=%d",
                    round_num + 1,
                    response.stop_reason,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                )

                # Append the assistant turn so the model has memory of it.
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    text = _extract_text(response.content)
                    s.set_attribute("tool_caller.rounds", round_num + 1)
                    s.set_attribute("tool_caller.tool_calls", len(all_tool_calls))
                    return AgentRun(
                        final_text=text,
                        rounds=round_num + 1,
                        tool_calls=all_tool_calls,
                    )

                if response.stop_reason != "tool_use":
                    raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")

                # Execute each requested tool and collect results.
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_result_messages = []

                for block in tool_use_blocks:
                    result = self._execute_tool(block)
                    all_tool_calls.append(result)
                    tool_result_messages.append(_to_tool_result_msg(result))

                # All results go back in a SINGLE user turn. This is required.
                messages.append({"role": "user", "content": tool_result_messages})

            raise RuntimeError(
                f"Tool-calling loop did not finish within {self.max_rounds} rounds. "
                "Either the task is too complex or the model is looping. "
                "Raise max_rounds or simplify the task."
            )

    def _execute_tool(self, block: Any) -> ToolCallResult:
        """Call the tool function and wrap the result."""
        tool = self._tools.get(block.name)
        if tool is None:
            logger.warning("unknown_tool tool=%s", block.name)
            return ToolCallResult(
                tool_use_id=block.id,
                tool_name=block.name,
                inputs=block.input,
                output=f"Unknown tool: {block.name!r}. Available: {list(self._tools)}",
                is_error=True,
            )

        with span(f"tool.{block.name}", attributes={"tool.name": block.name}):
            try:
                raw = tool.fn(**block.input)
                output = raw if isinstance(raw, str) else str(raw)
                is_error = False
            except Exception as exc:
                output = f"Tool raised an error: {exc}"
                is_error = True
                logger.warning("tool_error tool=%s error=%s", block.name, exc)

        logger.info(
            "tool_call tool=%s inputs=%s is_error=%s", block.name, block.input, is_error
        )
        return ToolCallResult(
            tool_use_id=block.id,
            tool_name=block.name,
            inputs=block.input,
            output=output,
            is_error=is_error,
        )


def _extract_text(content: list) -> str:
    """Pull all text blocks from a message content list."""
    return "".join(block.text for block in content if hasattr(block, "text"))


def _to_tool_result_msg(result: ToolCallResult) -> dict:
    """Format a ToolCallResult as an Anthropic tool_result message."""
    msg: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": result.tool_use_id,
        "content": result.output,
    }
    if result.is_error:
        msg["is_error"] = True
    return msg
