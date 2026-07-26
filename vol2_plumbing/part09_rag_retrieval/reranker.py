"""
core/rag/reranker.py

Re-rank retrieval results with a cross-encoder.

The pattern:
    1. Retrieve top-50 candidates from your hybrid retriever (cheap).
    2. Score each (query, doc) pair with a cross-encoder (more expensive
       but much more accurate than dual-encoder cosine similarity).
    3. Return the top-K after re-ranking.

This adds latency (each pair is a forward pass), so don't re-rank 1000
documents. 20-50 is the sweet spot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from vol2_plumbing.part09_rag_retrieval.retriever import RetrievalResult
from common.tracing import span


class CrossEncoderModel(Protocol):
    """Minimal interface for a cross-encoder model.

    Implementations: sentence-transformers CrossEncoder, Cohere Rerank API,
    or your own model. Anything that scores (query, document) pairs.
    """

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        ...


@dataclass
class RerankedResult:
    """A retrieval result after re-ranking."""

    document_id: str
    text: str
    original_score: float
    rerank_score: float
    rank_delta: int  # original_rank - new_rank. Positive = moved up.


class CrossEncoderReranker:
    """
    Re-rank a list of retrieval results using a cross-encoder.

    Example:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoderReranker(model)
        top_results = reranker.rerank(query, retrieval_results, top_k=5)
    """

    def __init__(self, model: CrossEncoderModel):
        self.model = model

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Re-score and re-order retrieval results.

        Args:
            query: The original query.
            results: Output of HybridRetriever.retrieve().
            top_k: How many results to return after re-ranking.

        Returns:
            Top-K results sorted by re-rank score, with rank deltas
            showing how much each document moved.
        """
        if not results:
            return []

        with span(
            "rag.rerank",
            attributes={"rag.candidates": len(results), "rag.top_k": top_k},
        ) as s:
            pairs = [(query, r.document.text) for r in results]
            scores = list(self.model.predict(pairs))

            scored = list(zip(range(len(results)), results, scores))
            scored.sort(key=lambda x: x[2], reverse=True)

            top_indices = scored[:top_k]
            new_rank_by_id = {
                r.document.id: new_rank
                for new_rank, (_, r, _) in enumerate(top_indices)
            }

            reranked: list[RerankedResult] = []
            for new_rank, (orig_idx, result, score) in enumerate(top_indices):
                reranked.append(
                    RerankedResult(
                        document_id=result.document.id,
                        text=result.document.text,
                        original_score=result.score,
                        rerank_score=float(score),
                        rank_delta=orig_idx - new_rank,
                    )
                )

            s.set_attribute("rag.reranked_count", len(reranked))
            return reranked
