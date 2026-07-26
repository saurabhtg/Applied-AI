# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
part08_parallel_tool_calls/parallel_runner.py

Parallel tool execution.

When the model requests multiple tools in a single response, run them
concurrently. Serial execution works; parallel execution is faster
by: sum(individual_latencies) - max(individual_latencies).

The invariant that must never break: ALL tool_use blocks from a
single response must be executed and returned in ONE user message
before the next model call. Serial or parallel is your choice.
Always parallel.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from anthropic import Anthropic

from common.tracing import span
from vol2_plumbing.part07_function_calling.tool_caller import (
    ToolCallResult,
    ToolDefinition,
    _to_tool_result_msg,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ROUNDS = 10


@dataclass
class ParallelAgentRun:
    """Result of a parallel tool-calling run, with timing breakdowns."""

    final_text: str
    rounds: int
    tool_calls: list[ToolCallResult] = field(default_factory=list)

    # Timing across ALL parallel batches in this run.
    # serial_equivalent_ms: what it would have cost running each tool one after another.
    # actual_ms: what it actually cost running them in parallel.
    serial_equivalent_ms: float = 0.0
    actual_ms: float = 0.0

    @property
    def time_saved_ms(self) -> float:
        return max(0.0, self.serial_equivalent_ms - self.actual_ms)

    @property
    def speedup(self) -> float:
        if self.actual_ms < 1.0:
            return 1.0
        return self.serial_equivalent_ms / self.actual_ms


class ParallelToolRunner:
    """
    Like ToolCaller, but fires multiple tool requests concurrently.

    When the model returns N tool_use blocks in one response, this runner
    submits all N to a thread pool, collects all results, then sends them
    back in a single user message. Single-tool rounds are executed inline
    (no thread overhead for a batch of one).

    Usage:
        runner = ParallelToolRunner(tools=[cibil_tool, itr_tool, emi_tool])
        result = runner.run("Is Rajesh Kumar eligible for a ₹15L loan?")
        print(f"Answer: {result.final_text}")
        print(f"Saved {result.time_saved_ms:.0f}ms vs serial execution")
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        max_workers: int = 10,
        api_key: str | None = None,
    ):
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._tools = {t.name: t for t in tools}
        self._api_specs = [t.to_api_spec() for t in tools]
        self.model = model
        self.max_tokens = max_tokens
        self.max_rounds = max_rounds
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(
        self,
        user_message: str,
        *,
        system: str | None = None,
    ) -> ParallelAgentRun:
        """Run the tool-calling loop with parallel tool execution."""
        with span("parallel_runner.run", attributes={"parallel_runner.model": self.model}) as s:
            messages: list[dict] = [{"role": "user", "content": user_message}]
            all_tool_calls: list[ToolCallResult] = []
            total_serial_ms = 0.0
            total_actual_ms = 0.0

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
                    "parallel_runner_round round=%d stop_reason=%s tools_requested=%d",
                    round_num + 1,
                    response.stop_reason,
                    sum(1 for b in response.content if b.type == "tool_use"),
                )
                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    text = _extract_text(response.content)
                    s.set_attribute("parallel_runner.rounds", round_num + 1)
                    s.set_attribute("parallel_runner.total_tool_calls", len(all_tool_calls))
                    return ParallelAgentRun(
                        final_text=text,
                        rounds=round_num + 1,
                        tool_calls=all_tool_calls,
                        serial_equivalent_ms=total_serial_ms,
                        actual_ms=total_actual_ms,
                    )

                if response.stop_reason != "tool_use":
                    raise RuntimeError(f"Unexpected stop_reason: {response.stop_reason!r}")

                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                if len(tool_use_blocks) == 1:
                    result = self._execute_one(tool_use_blocks[0])
                    all_tool_calls.append(result)
                    tool_result_messages = [_to_tool_result_msg(result)]
                else:
                    results, actual_ms, serial_ms = self._execute_parallel(tool_use_blocks)
                    all_tool_calls.extend(results)
                    tool_result_messages = [_to_tool_result_msg(r) for r in results]
                    total_actual_ms += actual_ms
                    total_serial_ms += serial_ms
                    logger.info(
                        "parallel_batch n=%d serial_ms=%.0f actual_ms=%.0f saved_ms=%.0f",
                        len(tool_use_blocks),
                        serial_ms,
                        actual_ms,
                        serial_ms - actual_ms,
                    )

                # All results in a single user turn — this is non-negotiable.
                messages.append({"role": "user", "content": tool_result_messages})

            raise RuntimeError(
                f"Agent did not finish within {self.max_rounds} rounds. "
                "Raise max_rounds or simplify the task."
            )

    def _execute_parallel(
        self,
        blocks: list[Any],
    ) -> tuple[list[ToolCallResult], float, float]:
        """
        Execute all tool blocks concurrently.

        Returns: (results_in_original_order, actual_wall_ms, serial_equivalent_ms)

        Results are returned in the same order as `blocks`. The API doesn't
        strictly require this, but it makes logs easier to follow.
        """
        futures = {self._executor.submit(self._execute_one_timed, b): b for b in blocks}
        results_by_id: dict[str, ToolCallResult] = {}
        durations_by_id: dict[str, float] = {}

        wall_start = time.perf_counter()
        for future in as_completed(futures):
            result, duration_ms = future.result()
            results_by_id[result.tool_use_id] = result
            durations_by_id[result.tool_use_id] = duration_ms
        actual_ms = (time.perf_counter() - wall_start) * 1000

        serial_ms = sum(durations_by_id.values())
        ordered = [results_by_id[b.id] for b in blocks]
        return ordered, actual_ms, serial_ms

    def _execute_one_timed(self, block: Any) -> tuple[ToolCallResult, float]:
        t0 = time.perf_counter()
        result = self._execute_one(block)
        return result, (time.perf_counter() - t0) * 1000

    def _execute_one(self, block: Any) -> ToolCallResult:
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

        logger.info("tool_call tool=%s inputs=%s is_error=%s", block.name, block.input, is_error)
        return ToolCallResult(
            tool_use_id=block.id,
            tool_name=block.name,
            inputs=block.input,
            output=output,
            is_error=is_error,
        )


def _extract_text(content: list) -> str:
    return "".join(block.text for block in content if hasattr(block, "text"))
