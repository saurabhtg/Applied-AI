# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol3_agents/part13_react_patterns/react_loop.py

A ReAct loop, written the way you'd write one in 2026.

The pattern is unchanged since the 2022 paper:
    Thought      -> the model reasons about what to do next
    Action       -> the model picks a tool and its arguments
    Observation  -> the tool runs, the result goes back in

What changed is who does the formatting. The old implementations asked
the model to *print* "Thought:/Action:/Action Input:" and then parsed
that with regex. Current models emit thinking blocks and typed tool_use
blocks natively, so the parser is gone. Your code dispatches; it no
longer interprets prose.

What this implementation gets right that toy examples don't:
    - Native tool_use blocks. No regex, no format drift, no silent
      "the model wrote Actoin: instead of Action:" failures.
    - Parallel tool calls in a single turn, with every result returned
      in ONE user message (splitting them trains the model to stop
      calling tools in parallel).
    - Tool errors come back as tool_result blocks with is_error=True.
      The model reads the failure and picks another route.
    - Hard cap on rounds AND a token budget. Both stop cleanly and say
      why in stopped_reason.
    - The full assistant content list is echoed back, thinking blocks
      included. Flatten it to a string and the model forgets its own
      reasoning between turns.
    - History is append-only. Thinking blocks are signed against
      everything before them, so editing an earlier turn (or rebuilding
      the system prompt) invalidates them. Nothing here rewrites messages.
    - Safety-classifier refusals fall back to another model server-side,
      so a false positive doesn't end the run.
    - Every round and every tool call emits a tracing span.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from anthropic import Anthropic

from common.tracing import span

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
# Thinking counts toward max_tokens even when its text isn't returned, so
# leave room for it as well as the reply.
DEFAULT_MAX_TOKENS = 16_000

# Server-side refusal fallback: on a classifier decline the API reruns the
# same request on the model Anthropic recommends for that category.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Python annotation -> JSON Schema type. Anything outside this map means we
# can't guarantee the schema, so we don't claim strict validation.
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

DEFAULT_SYSTEM = """\
You are a task-solving agent with access to tools.

Rules:
- Look things up with the tools before answering. Never guess a value
  a tool could have told you.
- Call tools in parallel when they don't depend on each other.
- If a tool returns an error, read it and try another route. Say so
  plainly if there is no route left.
- Stop calling tools the moment you can answer. Then answer.
"""


@dataclass
class Tool:
    """A function the agent can call.

    input_schema is optional. Leave it out and it gets derived from the
    function signature, which is the only schema that can't drift away
    from the code it describes.
    """

    name: str
    description: str
    fn: Callable[..., Any]
    input_schema: dict | None = None

    def to_api_spec(self) -> dict:
        schema, exact = (self.input_schema, False) if self.input_schema else _schema_from_signature(self.fn)
        spec: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "input_schema": schema,
        }
        # strict=True makes the API guarantee the arguments validate against
        # the schema. Only claim it when we derived every type cleanly.
        if exact:
            spec["strict"] = True
        return spec


@dataclass
class Step:
    """One thought/action/observation triplet.

    A single model turn can produce several tool calls. Each one becomes
    its own Step; the model's reasoning text is attached to the first.
    """

    thought: str
    action: str | None
    action_input: dict | None
    observation: str | None
    is_final: bool = False
    final_answer: str | None = None
    is_error: bool = False
    tool_use_id: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[Step]
    total_tokens: int
    # "final_answer" | "max_steps" | "token_budget" | "refusal" | "max_tokens"
    stopped_reason: str
    rounds: int = 0
    cache_read_tokens: int = 0


class ReActAgent:
    """
    A ReAct agent with bounded rounds and a token budget.

    Build it with a set of tools, then call .run(goal) for an answer:

        agent = ReActAgent(tools=[Tool(name="lookup", description="...", fn=lookup)])
        result = agent.run("Where is order ORD-12345?")
        print(result.answer, result.stopped_reason)
    """

    def __init__(
        self,
        tools: list[Tool],
        *,
        client: Anthropic | None = None,
        max_steps: int = 10,
        max_tokens_total: int = 50_000,
        model: str | None = None,
        system: str | None = None,
        effort: str = "high",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
    ):
        if not tools:
            raise ValueError("A ReAct agent needs at least one tool.")
        if max_steps > 30:
            logger.warning(
                "max_steps=%d is high. Agents that don't converge in 10-15 "
                "rounds usually never do.",
                max_steps,
            )

        self.tools = {t.name: t for t in tools}
        self._api_specs = [t.to_api_spec() for t in tools]
        self.client = client or Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.max_steps = max_steps
        self.max_tokens_total = max_tokens_total
        self.model = model or DEFAULT_MODEL
        self.system = system or DEFAULT_SYSTEM
        # low | medium | high | xhigh | max. Opus 5.5 defaults to medium if
        # you say nothing, so we always say something. Keep xhigh for routes
        # where you've measured a gain; use low for cheap sub-agents.
        self.effort = effort
        self.max_tokens = max_tokens

    def run(self, goal: str) -> AgentResult:
        """Execute the loop until the model is done, capped, or out of budget."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": goal}]
        steps: list[Step] = []
        total_tokens = 0
        cache_read = 0

        with span(
            "agent.run",
            attributes={"agent.goal": goal, "agent.max_steps": self.max_steps, "agent.model": self.model},
        ) as agent_span:
            for round_idx in range(self.max_steps):
                if total_tokens >= self.max_tokens_total:
                    return self._stop(
                        agent_span, "token_budget",
                        "(token budget exhausted before final answer)",
                        steps, total_tokens, round_idx, cache_read,
                    )

                response = self.client.beta.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=self.system,
                    tools=self._api_specs,
                    messages=messages,
                    # Thinking is always on for Opus 5.5; "disabled" and
                    # budget_tokens both return a 400, and so does temperature.
                    # Effort is the only dial.
                    thinking={"type": "adaptive"},
                    output_config={"effort": self.effort},
                    # Rolling prefix cache. The system prompt and tool specs are
                    # byte-identical every round, so the prefix stays warm.
                    cache_control={"type": "ephemeral"},
                    betas=[FALLBACK_BETA],
                    extra_body={"fallbacks": "default"},
                )

                total_tokens += response.usage.input_tokens + response.usage.output_tokens
                cache_read += getattr(response.usage, "cache_read_input_tokens", 0) or 0
                logger.info(
                    "round=%d stop_reason=%s in=%d out=%d cache_read=%d",
                    round_idx,
                    response.stop_reason,
                    response.usage.input_tokens,
                    response.usage.output_tokens,
                    getattr(response.usage, "cache_read_input_tokens", 0) or 0,
                )

                # Echo the whole content list back, thinking blocks and all.
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "pause_turn":
                    # A long-running server tool paused mid-turn. Send it back
                    # untouched and let the model carry on.
                    continue

                if response.stop_reason == "refusal":
                    # Only reached when the fallback model declined as well.
                    detail = getattr(response, "stop_details", None)
                    category = getattr(detail, "category", None)
                    return self._stop(
                        agent_span, "refusal",
                        f"(model declined the request: {category})",
                        steps, total_tokens, round_idx + 1, cache_read,
                    )

                if response.stop_reason != "tool_use":
                    # end_turn, or max_tokens if we clipped it.
                    text = _text_of(response.content)
                    reason = "max_tokens" if response.stop_reason == "max_tokens" else "final_answer"
                    steps.append(Step(
                        thought=text, action=None, action_input=None,
                        observation=None, is_final=True, final_answer=text,
                    ))
                    return self._stop(
                        agent_span, reason, text, steps, total_tokens, round_idx + 1, cache_read,
                    )

                thought = _text_of(response.content)
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                results: list[dict[str, Any]] = []

                for i, block in enumerate(tool_uses):
                    observation, is_error = self._execute_tool(block.name, dict(block.input))
                    steps.append(Step(
                        thought=thought if i == 0 else "",
                        action=block.name,
                        action_input=dict(block.input),
                        observation=observation,
                        is_error=is_error,
                        tool_use_id=block.id,
                    ))
                    result_block: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": observation,
                    }
                    if is_error:
                        result_block["is_error"] = True
                    results.append(result_block)

                # Every result, one user message. Splitting these across several
                # messages quietly teaches the model to stop batching calls.
                messages.append({"role": "user", "content": results})

            return self._stop(
                agent_span, "max_steps",
                "(max rounds reached without a final answer)",
                steps, total_tokens, self.max_steps, cache_read,
            )

    def _execute_tool(self, name: str, args: dict) -> tuple[str, bool]:
        """Run a tool. Returns (observation, is_error). Never raises."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Unknown tool '{name}'. Available: {list(self.tools)}", True

        with span("agent.tool", attributes={"tool.name": name}) as s:
            try:
                result = tool.fn(**args)
                output = result if isinstance(result, str) else json.dumps(result, default=str)
                s.set_attribute("tool.success", True)
                return output, False
            except Exception as exc:
                s.set_attribute("tool.success", False)
                logger.warning("tool_error tool=%s args=%s error=%s", name, args, exc)
                # The failure is an observation, not an exception. The model
                # gets to decide what happens next.
                return f"Error from {name}: {exc}", True

    @staticmethod
    def _stop(
        agent_span: Any,
        reason: str,
        answer: str,
        steps: list[Step],
        total_tokens: int,
        rounds: int,
        cache_read: int,
    ) -> AgentResult:
        agent_span.set_attribute("agent.stopped", reason)
        agent_span.set_attribute("agent.total_steps", len(steps))
        agent_span.set_attribute("agent.total_tokens", total_tokens)
        return AgentResult(
            answer=answer,
            steps=steps,
            total_tokens=total_tokens,
            stopped_reason=reason,
            rounds=rounds,
            cache_read_tokens=cache_read,
        )


def _text_of(content: list) -> str:
    """Visible text from a response. Thinking blocks are deliberately skipped."""
    return "".join(b.text for b in content if b.type == "text").strip()


def _schema_from_signature(fn: Callable[..., Any]) -> tuple[dict, bool]:
    """Derive a JSON Schema from a function signature.

    Returns (schema, exact). exact is False when an annotation is missing or
    isn't a type we can map, which is the signal not to claim strict mode.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    exact = True

    for param in inspect.signature(fn).parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            exact = False
            continue
        json_type = _JSON_TYPES.get(param.annotation)
        if json_type is None:
            json_type, exact = "string", False
        properties[param.name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param.name)

    return (
        {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
        exact,
    )
