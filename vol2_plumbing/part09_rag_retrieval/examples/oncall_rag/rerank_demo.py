"""
examples/oncall_rag/rerank_demo.py

The full retrieve -> rerank -> answer pipeline, end to end.

Part 9's qa.py stops after retrieval: it takes the hybrid retriever's
top-k and hands them straight to the model. This script adds the step in
between. It retrieves a wider net of candidates, rescoring each one with a
cross-encoder, and only then sends the survivors to the model.

Two retrievers, two jobs:

    HybridRetriever (BM25 + dense)   fast, approximate, casts a wide net.
                                     Used to go from 15 docs -> top 10.

    CrossEncoderReranker             slow, accurate, reads query + passage
                                     together. Used to go from 10 -> top 4.

The cross-encoder:

    Preferred: sentence-transformers' "cross-encoder/ms-marco-MiniLM-L-6-v2".
    A small, real, MS-MARCO-trained reranker. First run downloads ~80 MB.

    Fallback: if sentence-transformers isn't installed (or the model can't
    be downloaded), we drop to a crude token-overlap scorer so the pipeline
    still runs and you can see the mechanics. The rankings will be worse
    than a real cross-encoder's -- which is the whole argument for installing
    one. Watch the [reranker=...] line in the output to see which you got.

    To get the real thing:
        pip install sentence-transformers

Run:
    python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.rerank_demo
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from common.llm_client import LLMClient
from vol2_plumbing.part09_rag_retrieval.reranker import (
    CrossEncoderReranker,
    RerankedResult,
)
from vol2_plumbing.part09_rag_retrieval.retriever import RetrievalResult
from .qa import ANSWER_PROMPT, build_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# --------------------------------------------------------------------------- #
# Loading a cross-encoder, with a graceful fallback.
# --------------------------------------------------------------------------- #


def _tokens(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


class _LexicalCrossEncoder:
    """
    Fallback used when sentence-transformers isn't available.

    A real cross-encoder reads the query and the passage jointly through a
    transformer and outputs a learned relevance score. We can't do that
    without the model. This approximates relevance with query-token overlap
    -- a weak proxy, but enough to make the pipeline run and the rank
    movements visible. Do not ship this; install the real model instead.
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, doc in pairs:
            q = set(_tokens(query))
            if not q:
                scores.append(0.0)
                continue
            d = set(_tokens(doc))
            scores.append(len(q & d) / len(q))
        return scores


def load_reranker() -> tuple[CrossEncoderReranker, str]:
    """
    Build a CrossEncoderReranker.

    Returns the reranker and a label naming the model actually loaded, so
    the caller can tell whether it got the real cross-encoder or the stub.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        logger.warning(
            "sentence-transformers not installed; using lexical stub. "
            "Run `pip install sentence-transformers` for real reranking."
        )
        return CrossEncoderReranker(_LexicalCrossEncoder()), "lexical-stub"

    try:
        # CrossEncoder.predict(pairs) already matches the CrossEncoderModel
        # protocol -- it takes (query, doc) pairs and returns scores -- so we
        # can hand it straight to the reranker, no adapter needed.
        model = CrossEncoder(_MODEL_NAME)
    except Exception as exc:  # download failure, offline, disk, etc.
        logger.warning(
            "Could not load %s (%s); using lexical stub instead.",
            _MODEL_NAME,
            exc,
        )
        return CrossEncoderReranker(_LexicalCrossEncoder()), "lexical-stub"

    return CrossEncoderReranker(model), _MODEL_NAME


# --------------------------------------------------------------------------- #
# The pipeline.
# --------------------------------------------------------------------------- #


def _delta_arrow(delta: int) -> str:
    """Render a rank delta: positive means the doc moved up after reranking."""
    if delta > 0:
        return f"up {delta}"
    if delta < 0:
        return f"down {-delta}"
    return "--"


def show_rerank(
    question: str,
    candidates: Sequence[RetrievalResult],
    reranked: Sequence[RerankedResult],
) -> None:
    """Print the retriever's order next to the cross-encoder's order."""
    print(f"\nQ: {question}")
    print(f"  retriever top {len(candidates)} (BM25 + dense):")
    for rank, r in enumerate(candidates):
        print(f"    {rank:>2}. {r.document.id}   retr={r.score:.3f}")
    print(f"  after cross-encoder rerank -> top {len(reranked)}:")
    for new_rank, rr in enumerate(reranked):
        was = new_rank + rr.rank_delta
        print(
            f"    {new_rank:>2}. {rr.document_id}   "
            f"ce={rr.rerank_score:+.3f}   (was #{was}, {_delta_arrow(rr.rank_delta)})"
        )


def answer_with_rerank(
    question: str,
    reranker: CrossEncoderReranker,
    *,
    candidate_k: int = 10,
    top_k: int = 4,
    show: bool = True,
) -> dict:
    """
    Full pipeline for one question: retrieve wide, rerank, answer.

    Args:
        question: The user's question.
        reranker: A loaded CrossEncoderReranker.
        candidate_k: How many candidates the retriever returns (the wide net).
        top_k: How many survive the rerank and reach the model.
        show: Print the before/after ranking comparison.
    """
    retriever = build_retriever()
    candidates = retriever.retrieve(question, k=candidate_k)
    if not candidates:
        return {"question": question, "answer": "No relevant context found.", "sources": []}

    reranked = reranker.rerank(question, candidates, top_k=top_k)
    if show:
        show_rerank(question, candidates, reranked)

    context_block = "\n\n".join(
        f"[{rr.document_id}] (rerank={rr.rerank_score:.2f})\n{rr.text}"
        for rr in reranked
    )
    prompt = ANSWER_PROMPT.format(context=context_block, question=question)

    client = LLMClient()
    response = client.complete(
        prompt=prompt,
        max_tokens=500,
        temperature=0.0,
        prompt_name="oncall_rag_rerank",
        prompt_version="1.0",
    )
    return {
        "question": question,
        "answer": response.text,
        "sources": [rr.document_id for rr in reranked],
    }


def main() -> None:
    reranker, label = load_reranker()
    print(f"[reranker={label}]")

    questions = [
        "How do I roll back a bad deployment?",
        "What does ERR_5521 mean?",
        "A payments-api pod keeps restarting on its own — what should I check?",
    ]

    for q in questions:
        result = answer_with_rerank(q, reranker)
        print(f"  A: {result['answer']}")
        print(f"  sources: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
