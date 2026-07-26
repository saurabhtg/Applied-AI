"""
core/evals/ragas_runner.py

Compute RAGAS-style metrics for RAG pipelines.

The four metrics worth tracking:
    - faithfulness: did the answer stay grounded in the retrieved context?
    - answer_relevance: did the answer actually address the question?
    - context_precision: how much of the retrieved context was useful?
    - context_recall: did retrieval find the information the answer needed?

These can be computed with a strong judge model. The official RAGAS
library exists if you want more metrics and a turn-key implementation;
this module gives you the four that matter most, implemented in code
you can read and modify.

Usage:
    from vol2_plumbing.part10_ragas.ragas_runner import compute_ragas_metrics
    metrics = compute_ragas_metrics(
        question="...",
        answer="...",
        contexts=["...", "..."],
        ground_truth="...",  # optional, needed for context_recall
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from common.llm_client import LLMClient


@dataclass
class RagasMetrics:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float | None  # None if ground_truth not provided.

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3) if self.context_recall is not None else None,
        }


_FAITHFULNESS_PROMPT = """\
You are checking whether an answer is grounded in the provided context.

Context:
{context}

Answer:
{answer}

Break the answer into individual factual claims. For each claim, decide
whether it is SUPPORTED by the context.

Output one line per claim in this format:
CLAIM: <claim> | SUPPORTED: yes | no

After all claims, output:
TOTAL_CLAIMS: <integer>
SUPPORTED_CLAIMS: <integer>
"""

_RELEVANCE_PROMPT = """\
Given an answer, infer what question it would best answer.

Answer:
{answer}

Output: a single question that this answer addresses. No preamble.
"""

_RECALL_PROMPT = """\
Given a ground truth answer, decide how much of it is covered by the
provided context.

Ground truth:
{ground_truth}

Context:
{context}

Break the ground truth into individual factual claims. For each one,
decide whether it is COVERED by the context.

Output one line per claim:
CLAIM: <claim> | COVERED: yes | no

After all claims:
TOTAL_CLAIMS: <integer>
COVERED_CLAIMS: <integer>
"""


def compute_ragas_metrics(
    question: str,
    answer: str,
    contexts: list[str],
    *,
    ground_truth: str | None = None,
    client: LLMClient | None = None,
) -> RagasMetrics:
    """
    Compute the four core RAGAS metrics.

    Args:
        question: The user's question.
        answer: The RAG pipeline's answer.
        contexts: List of retrieved context passages.
        ground_truth: Optional reference answer. Required for context_recall.
        client: Optional LLM client.

    Returns:
        A RagasMetrics object.
    """
    client = client or LLMClient()
    joined_context = "\n\n---\n\n".join(contexts)

    faithfulness = _measure_faithfulness(answer, joined_context, client)
    answer_relevance = _measure_relevance(question, answer, client)
    context_precision = _measure_precision(question, contexts, client)
    context_recall = (
        _measure_recall(ground_truth, joined_context, client)
        if ground_truth
        else None
    )

    return RagasMetrics(
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        context_precision=context_precision,
        context_recall=context_recall,
    )


def _measure_faithfulness(answer: str, context: str, client: LLMClient) -> float:
    """Fraction of claims in the answer supported by context."""
    response = client.complete(
        prompt=_FAITHFULNESS_PROMPT.format(context=context, answer=answer),
        max_tokens=800,
        temperature=0.0,
        prompt_name="ragas_faithfulness",
        prompt_version="1.0",
    )
    return _parse_supported_ratio(response.text, "SUPPORTED_CLAIMS")


def _measure_relevance(question: str, answer: str, client: LLMClient) -> float:
    """
    Given the answer, infer what question it answers. Then ask the judge
    how similar the inferred question is to the original.

    This is a rough proxy for whether the answer is on-topic.
    """
    response = client.complete(
        prompt=_RELEVANCE_PROMPT.format(answer=answer),
        max_tokens=100,
        temperature=0.0,
        prompt_name="ragas_relevance_infer",
        prompt_version="1.0",
    )
    inferred = response.text.strip()
    return _semantic_overlap(question, inferred)


def _measure_precision(question: str, contexts: list[str], client: LLMClient) -> float:
    """
    Fraction of retrieved contexts that contain information relevant to the
    question. Cheap heuristic: ask the judge yes/no per context.
    """
    if not contexts:
        return 0.0
    relevant = 0
    for ctx in contexts:
        prompt = (
            f"Question: {question}\n\nContext passage:\n{ctx}\n\n"
            f"Is this passage relevant to answering the question? "
            f"Respond with only YES or NO."
        )
        response = client.complete(
            prompt=prompt,
            max_tokens=10,
            temperature=0.0,
            prompt_name="ragas_precision",
            prompt_version="1.0",
        )
        if "YES" in response.text.upper():
            relevant += 1
    return relevant / len(contexts)


def _measure_recall(ground_truth: str, context: str, client: LLMClient) -> float:
    """Fraction of ground truth claims covered by the context."""
    response = client.complete(
        prompt=_RECALL_PROMPT.format(ground_truth=ground_truth, context=context),
        max_tokens=800,
        temperature=0.0,
        prompt_name="ragas_recall",
        prompt_version="1.0",
    )
    return _parse_supported_ratio(response.text, "COVERED_CLAIMS")


def _parse_supported_ratio(text: str, supported_key: str) -> float:
    """Parse total/supported counts from the judge output."""
    total = 0
    supported = 0
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("TOTAL_CLAIMS:"):
            try:
                total = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.upper().startswith(supported_key.upper() + ":"):
            try:
                supported = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
    return supported / total if total > 0 else 0.0


def _semantic_overlap(a: str, b: str) -> float:
    """Crude Jaccard overlap on word sets. Replace with embeddings for production."""
    a_words = set(re.findall(r"\b\w+\b", a.lower()))
    b_words = set(re.findall(r"\b\w+\b", b.lower()))
    if not a_words or not b_words:
        return 0.0
    intersect = a_words & b_words
    union = a_words | b_words
    return len(intersect) / len(union)
