# Applied-AI Series - Think and Build with SaurabhG
#
# MIT License
#
# Copyright (c) 2026 Saurabh Gupta

"""
vol3_agents/part15_reflexion/reflexion.py

The Socratic Agent — Reflexion.

Instead of accepting its first answer, the agent has its work graded and
retries if the answer falls short. Like a student who writes an answer,
checks it against the marking scheme, spots what they missed, and tries
again.

The pattern:
    1. Run the base ReActAgent. Get an answer and a trace of tool calls.
    2. A grader, in a fresh context, checks each criterion against the
       trace and the answer, then scores the whole thing 1–5.
    3. If it falls short: generate a reflection — a specific diagnosis of
       what went wrong and what to try differently.
    4. Run the base agent again with every reflection so far attached.
    5. Repeat up to max_attempts.

What this gets right that simpler implementations don't:
    - The grader gets evidence, not just prose. It sees which tools ran
      and what came back, so "did it check the warranty?" is a fact it
      can look up, not a guess about the wording of the answer.
    - The verdict is structured output: one met/not-met per criterion plus
      a score, validated against a JSON schema. No regex, and no silent
      default score when the grader says something unexpected.
    - Reflections accumulate, like the episodic memory in the original
      paper. Attempt 3 knows what went wrong in attempts 1 and 2.
    - All attempts are returned, not just the last one. You can see the
      improvement arc.
    - If max_attempts is exhausted without passing, we return the best attempt
      by score — not the last one, which might be confused by excessive context.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from anthropic import Anthropic

from common.tracing import span
from vol3_agents.part13_react_patterns.react_loop import (
    FALLBACK_BETA,
    AgentResult,
    ReActAgent,
)

logger = logging.getLogger(__name__)

# The grader defaults to the same model as the base agent. Graders favour
# their own family's work, so if you have a second model you trust, pass it
# as `model` and the grading gets more honest.
DEFAULT_JUDGE_MODEL = "claude-opus-5-5"

# How many past reflections ride along with the next attempt.
MAX_REFLECTIONS_CARRIED = 3

_EVALUATOR_SYSTEM = """\
You are a strict grader for an AI agent's work. You receive the task, a
trace of the tool calls the agent made, the agent's final answer, and a
list of criteria.

Grade each criterion on its own, using the trace as evidence. If a
criterion doesn't apply to this task, mark it met and say why. Then score
the whole answer:

  1 = Wrong or missing key information
  2 = Partly correct with significant gaps
  3 = Mostly correct, minor errors or omissions
  4 = Correct, small improvements possible
  5 = Complete and accurate

The critique is one or two sentences about what fell short, pointing at
the evidence. If nothing fell short, say so.
"""

_EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "met": {"type": "boolean"},
                    "evidence": {"type": "string"},
                },
                "required": ["criterion", "met", "evidence"],
                "additionalProperties": False,
            },
        },
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "critique": {"type": "string"},
    },
    "required": ["criteria", "score", "critique"],
    "additionalProperties": False,
}

_EVALUATOR_USER_TEMPLATE = """\
Task: {goal}

Tool calls the agent made:
{trace}

Agent's answer:
{answer}

Criteria:
{criteria}
"""

_REFLECTION_SYSTEM = """\
You are coaching an AI agent that gave a poor answer. Write a reflection
that tells the agent specifically what went wrong and what it should do
differently. Be concrete — name the missing step, the wrong assumption,
or the overlooked data. One paragraph, no fluff.
"""

_REFLECTION_USER_TEMPLATE = """\
Task: {goal}

Tool calls the agent made:
{trace}

Agent's answer:
{answer}

Grader's critique: {critique}
"""


@dataclass
class ReflexionAttempt:
    """One attempt: the agent ran, we evaluated it."""

    attempt_number: int
    answer: str
    agent_steps: int
    agent_tokens: int
    score: float
    critique: str
    reflection: str | None  # None if this attempt succeeded or was last
    passed: bool = False
    # One {"criterion", "met", "evidence"} dict per criterion, from the grader.
    criteria: list[dict] = field(default_factory=list)


@dataclass
class ReflexionResult:
    """All attempts, plus the final answer."""

    final_answer: str
    attempts: list[ReflexionAttempt] = field(default_factory=list)
    succeeded: bool = False
    total_tokens: int = 0


class ReflexionAgent:
    """
    Wraps a ReActAgent with self-evaluation and retry.

    On each failed attempt, the agent generates a reflection — a diagnosis
    of what went wrong — and passes it forward as context for the next run.
    """

    def __init__(
        self,
        base_agent: ReActAgent,
        evaluation_criteria: str,
        *,
        client: Anthropic | None = None,
        max_attempts: int = 3,
        min_score: float = 4.0,
        model: str | None = None,
    ):
        if max_attempts > 5:
            logger.warning(
                "max_attempts=%d is high. If the agent doesn't improve within "
                "3 attempts, the issue is usually the goal definition or missing "
                "tools, not the retry count.",
                max_attempts,
            )
        self.base_agent = base_agent
        self.evaluation_criteria = evaluation_criteria
        self.client = client or Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.max_attempts = max_attempts
        self.min_score = min_score
        # Model for the grader and the reflector.
        self.model = model or DEFAULT_JUDGE_MODEL

    def run(self, goal: str) -> ReflexionResult:
        """Run with self-evaluation and retry. Returns all attempts."""
        attempts: list[ReflexionAttempt] = []
        total_tokens = 0
        reflections: list[str] = []

        with span("reflexion.run", attributes={"reflexion.goal": goal[:120]}):
            for attempt_num in range(1, self.max_attempts + 1):
                augmented_goal = _with_reflections(goal, reflections)

                with span(
                    "reflexion.attempt",
                    attributes={"reflexion.attempt_number": attempt_num},
                ):
                    agent_result: AgentResult = self.base_agent.run(augmented_goal)
                    total_tokens += agent_result.total_tokens
                    trace = _trace_of(agent_result)

                    verdict, eval_tokens = self._evaluate(goal, trace, agent_result.answer)
                    total_tokens += eval_tokens

                    unmet = [c for c in verdict["criteria"] if not c["met"]]
                    succeeded = verdict["score"] >= self.min_score and not unmet

                    reflection_text = None
                    if not succeeded and attempt_num < self.max_attempts:
                        reflection_text, reflect_tokens = self._reflect(
                            goal, trace, agent_result.answer, verdict["critique"]
                        )
                        total_tokens += reflect_tokens
                        reflections.append(reflection_text)

                    attempt = ReflexionAttempt(
                        attempt_number=attempt_num,
                        answer=agent_result.answer,
                        agent_steps=len(agent_result.steps),
                        agent_tokens=agent_result.total_tokens,
                        score=float(verdict["score"]),
                        critique=verdict["critique"],
                        reflection=reflection_text,
                        passed=succeeded,
                        criteria=verdict["criteria"],
                    )
                    attempts.append(attempt)

                    logger.info(
                        "reflexion attempt=%d score=%d unmet=%d succeeded=%s",
                        attempt_num,
                        verdict["score"],
                        len(unmet),
                        succeeded,
                    )

                    if succeeded:
                        return ReflexionResult(
                            final_answer=agent_result.answer,
                            attempts=attempts,
                            succeeded=True,
                            total_tokens=total_tokens,
                        )

        # Exhausted attempts without passing. Return best by score.
        best = max(attempts, key=lambda a: a.score)
        return ReflexionResult(
            final_answer=best.answer,
            attempts=attempts,
            succeeded=False,
            total_tokens=total_tokens,
        )

    def _evaluate(self, goal: str, trace: str, answer: str) -> tuple[dict, int]:
        """Grade the attempt. Returns (verdict, tokens_used).

        A grader that can't produce a verdict is reported as a failed
        attempt with the reason in the critique, never as a made-up score.
        """
        user_msg = _EVALUATOR_USER_TEMPLATE.format(
            goal=goal,
            trace=trace,
            answer=answer,
            criteria=self.evaluation_criteria,
        )
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=8000,
            system=_EVALUATOR_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": _EVALUATION_SCHEMA},
            },
            betas=[FALLBACK_BETA],
            extra_body={"fallbacks": "default"},
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens

        if response.stop_reason != "end_turn":
            logger.warning("grader stopped early: stop_reason=%s", response.stop_reason)
            return _no_verdict(f"Grader returned no verdict (stop_reason={response.stop_reason})."), tokens

        return json.loads(_text_of(response.content)), tokens

    def _reflect(self, goal: str, trace: str, answer: str, critique: str) -> tuple[str, int]:
        """Generate a reflection. Returns (reflection_text, tokens_used)."""
        user_msg = _REFLECTION_USER_TEMPLATE.format(
            goal=goal,
            trace=trace,
            answer=answer,
            critique=critique,
        )
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=8000,
            system=_REFLECTION_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            extra_body={"fallbacks": "default"},
        )
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return _text_of(response.content), tokens


def _with_reflections(goal: str, reflections: list[str]) -> str:
    """Attach the most recent reflections to the goal, oldest first."""
    if not reflections:
        return goal
    recent = reflections[-MAX_REFLECTIONS_CARRIED:]
    notes = "\n".join(f"{i}. {r}" for i, r in enumerate(recent, start=1))
    return (
        f"{goal}\n\n"
        f"[Lessons from previous attempts at this task. Read them and do not "
        f"repeat these mistakes]:\n{notes}"
    )


def _trace_of(result: AgentResult, max_obs_chars: int = 400) -> str:
    """Render the agent's tool calls as evidence for the grader."""
    lines = []
    for step in result.steps:
        if step.action is None:
            continue
        status = "ERROR" if step.is_error else "ok"
        obs = (step.observation or "")[:max_obs_chars]
        lines.append(f"- [{status}] {step.action}({json.dumps(step.action_input)}) -> {obs}")
    return "\n".join(lines) or "(no tool calls)"


def _no_verdict(reason: str) -> dict:
    return {"criteria": [], "score": 1, "critique": reason}


def _text_of(content: list) -> str:
    """Visible text from a response. Thinking blocks are skipped."""
    return "".join(b.text for b in content if b.type == "text").strip()
