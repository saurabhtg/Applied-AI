"""
core/rag/retriever.py

Hybrid retrieval: BM25 (lexical) + dense embeddings (semantic).

Why hybrid: pure vector search misses exact-match queries (acronyms,
product codes, names). Pure BM25 misses semantic paraphrases. Together
they cover each other's blind spots.

The fix is rarely a different vector DB. The fix is usually:
    1. Add BM25 alongside the vector search.
    2. Fuse the two result sets. Either blend normalized scores
       (fusion="weighted") or fuse the rankings (fusion="rrf").
    3. Take the top-K, deduplicate, re-rank.

This module gives you that, in a form you can fork and modify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from common.tracing import span


@dataclass
class Document:
    """A retrievable document."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A retrieved document with its score.

    ``source`` names the fusion mode, but two docs can both be "hybrid"
    hits while one was dragged in by BM25 and the other by dense — the
    label alone can't tell you which. ``bm25_rank``/``dense_rank`` (and
    the matching raw scores) are each retriever's *own* opinion of this
    doc, independent of fusion, so you can log per-chunk provenance
    instead of guessing which half of the hybrid pipeline is starving.
    """

    document: Document
    score: float
    source: str  # "bm25" | "hybrid" | "hybrid_rrf"
    bm25_rank: int | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    dense_score: float | None = None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization. Replace if you need more."""
    return text.lower().split()


def _minmax_normalize(scores: list[float]) -> list[float]:
    """Min-max scale scores into [0, 1]."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of document IDs by *rank*, not by score.

    Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009) scores each
    document by the sum of ``1 / (k + rank)`` across every list it appears
    in, where ``rank`` is its 0-based position in that list::

        rrf(doc) = Σ  1 / (k + rank_in_list_i(doc))

    Why bother when we already have alpha-weighted score blending?
    Min-max normalization assumes both retrievers' score distributions are
    well behaved. They often aren't: BM25 scores spike hard on an exact
    acronym match while cosine similarities sit in a narrow band. After
    min-max scaling, one outlier flattens everything else, and your alpha
    stops meaning anything. RRF never looks at the raw scores — only the
    order — so it is immune to that. It is also what most production engines
    (Elasticsearch, Qdrant, Weaviate) ship as their default fusion.

    The constant ``k`` damps the influence of top ranks: a larger ``k``
    makes the contribution of rank 0 vs rank 5 more similar, so agreement
    across lists matters more than being #1 in any single list. ``k = 60``
    is the value from the original paper and a sane default.

    Args:
        ranked_lists: One ordered sequence of document IDs per retriever,
            best match first. Lists may differ in length and contents.
        k: Rank-damping constant. Defaults to 60.

    Returns:
        ``(document_id, rrf_score)`` pairs sorted by score descending.
    """
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


class HybridRetriever:
    """
    Combines BM25 lexical search and dense semantic search.

    Build it once with a corpus, then call .retrieve(query, k) to get
    the top-K results. Bring your own embedding function.
    """

    def __init__(
        self,
        documents: Sequence[Document],
        embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
        alpha: float = 0.5,
        fusion: str = "weighted",
        rrf_k: int = 60,
    ):
        """
        Args:
            documents: The corpus to index.
            embed_fn: Function that takes a list of strings, returns an
                (N, D) numpy array of embeddings. If None, dense search
                is disabled and only BM25 is used.
            alpha: Weight for dense scores in the hybrid combination.
                0.0 = pure BM25, 1.0 = pure dense, 0.5 = balanced.
                Only used when ``fusion="weighted"``.
            fusion: How to combine BM25 and dense results.
                "weighted" = min-max normalize each, then alpha-blend the
                scores. "rrf" = reciprocal rank fusion, which combines the
                two *rankings* and ignores raw scores (more robust when the
                score distributions are skewed). See
                :func:`reciprocal_rank_fusion`.
            rrf_k: Rank-damping constant for RRF. Only used when
                ``fusion="rrf"``. Defaults to 60.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if fusion not in ("weighted", "rrf"):
            raise ValueError("fusion must be 'weighted' or 'rrf'")

        self.documents = list(documents)
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.embed_fn = embed_fn

        # BM25 index.
        self._tokenized = [_tokenize(d.text) for d in self.documents]
        self._bm25 = BM25Okapi(self._tokenized)

        # Dense index, if embedder provided.
        self._doc_embeddings: np.ndarray | None = None
        if embed_fn is not None:
            self._doc_embeddings = embed_fn([d.text for d in self.documents])

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """
        Retrieve the top-K documents for the query.

        If an embed_fn was provided, returns hybrid results.
        Otherwise returns pure BM25 results.
        """
        with span(
            "rag.retrieve",
            attributes={"rag.query": query, "rag.k": k, "rag.alpha": self.alpha},
        ) as s:
            bm25_scores = self._bm25.get_scores(_tokenize(query))
            bm25_norm = _minmax_normalize(bm25_scores.tolist())
            # Rank of every doc in BM25's own ordering, independent of fusion.
            bm25_rank_by_idx = {
                int(i): rank for rank, i in enumerate(np.argsort(bm25_scores)[::-1])
            }

            if self.embed_fn is None or self._doc_embeddings is None:
                # Pure BM25.
                indexed = sorted(
                    enumerate(bm25_norm), key=lambda x: x[1], reverse=True
                )[:k]
                results = [
                    RetrievalResult(
                        self.documents[i],
                        score,
                        "bm25",
                        bm25_rank=bm25_rank_by_idx[i],
                        bm25_score=float(bm25_scores[i]),
                    )
                    for i, score in indexed
                ]
                s.set_attribute("rag.mode", "bm25_only")
                s.set_attribute("rag.results_count", len(results))
                self._log_contributors(s, results)
                return results

            # Hybrid: combine BM25 and dense.
            query_vec = self.embed_fn([query])[0]
            dense_scores = self._cosine_sim(query_vec, self._doc_embeddings)
            # Rank of every doc in dense's own ordering, independent of fusion.
            dense_rank_by_idx = {
                int(i): rank for rank, i in enumerate(np.argsort(dense_scores)[::-1])
            }

            if self.fusion == "rrf":
                # Rank-based fusion: turn each retriever's scores into a
                # ranking of document IDs, then fuse the rankings. Raw
                # score scales never enter the calculation.
                bm25_ranking = [
                    self.documents[i].id
                    for i in np.argsort(bm25_scores)[::-1]
                ]
                dense_ranking = [
                    self.documents[i].id
                    for i in np.argsort(dense_scores)[::-1]
                ]
                fused = reciprocal_rank_fusion(
                    [bm25_ranking, dense_ranking], k=self.rrf_k
                )[:k]
                by_id = {d.id: d for d in self.documents}
                idx_by_id = {d.id: i for i, d in enumerate(self.documents)}
                results = [
                    RetrievalResult(
                        by_id[doc_id],
                        score,
                        "hybrid_rrf",
                        bm25_rank=bm25_rank_by_idx.get(idx_by_id[doc_id]),
                        dense_rank=dense_rank_by_idx.get(idx_by_id[doc_id]),
                        bm25_score=float(bm25_scores[idx_by_id[doc_id]]),
                        dense_score=float(dense_scores[idx_by_id[doc_id]]),
                    )
                    for doc_id, score in fused
                ]
                s.set_attribute("rag.mode", "hybrid_rrf")
                s.set_attribute("rag.results_count", len(results))
                self._log_contributors(s, results)
                return results

            dense_norm = _minmax_normalize(dense_scores.tolist())
            combined = [
                (1 - self.alpha) * b + self.alpha * d
                for b, d in zip(bm25_norm, dense_norm)
            ]
            indexed = sorted(
                enumerate(combined), key=lambda x: x[1], reverse=True
            )[:k]
            results = [
                RetrievalResult(
                    self.documents[i],
                    score,
                    "hybrid",
                    bm25_rank=bm25_rank_by_idx[i],
                    dense_rank=dense_rank_by_idx[i],
                    bm25_score=float(bm25_scores[i]),
                    dense_score=float(dense_scores[i]),
                )
                for i, score in indexed
            ]
            s.set_attribute("rag.mode", "hybrid")
            s.set_attribute("rag.results_count", len(results))
            self._log_contributors(s, results)
            return results

    @staticmethod
    def _log_contributors(s: Any, results: list[RetrievalResult]) -> None:
        """Emit one span event per result recording which retriever(s)
        actually ranked it and where — so "why is this chunk here" is a
        trace lookup, not a guess from the aggregate ``source`` label."""
        for r in results:
            s.add_event(
                "rag.result",
                attributes={
                    "doc.id": r.document.id,
                    "source": r.source,
                    "bm25_rank": r.bm25_rank if r.bm25_rank is not None else -1,
                    "dense_rank": r.dense_rank if r.dense_rank is not None else -1,
                },
            )

    @staticmethod
    def _cosine_sim(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a query vector and a matrix of docs."""
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        d_norms = doc_matrix / (
            np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-9
        )
        return d_norms @ q_norm
