"""
examples/oncall_rag/fusion_compare.py

Two ways to fuse BM25 + dense, side by side.

Part 9's HybridRetriever can combine its two retrievers in two ways:

    1. "weighted" — min-max normalize each retriever's scores onto [0, 1],
       then blend: (1 - alpha) * bm25 + alpha * dense.
    2. "rrf" — reciprocal rank fusion. Ignore the raw scores entirely;
       combine the two *rankings* by summing 1 / (k + rank).

This script runs the same exact-token query through both and prints the
rankings next to each other, so you can see where they disagree.

Why the disagreement matters: on a query like "ERR_5521", BM25 gives the
one exact-match passage a huge score and everything else ~0. After
min-max normalization that single spike flattens the rest of the BM25
signal, so the blended score leans almost entirely on the dense side —
which has no idea what "ERR_5521" means. RRF only sees that the exact-match
doc is ranked #1 by BM25, so it keeps it near the top regardless of how
extreme the raw score was.

Run:
    python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.fusion_compare
"""

from __future__ import annotations

from vol2_plumbing.part09_rag_retrieval.retriever import Document, HybridRetriever
from .corpus import ONCALL_CORPUS
from .qa import _embed_stub


def _build(fusion: str) -> HybridRetriever:
    """Build a retriever over the on-call corpus with the given fusion mode."""
    documents = [
        Document(id=d.id, text=d.text, metadata={"source": d.source})
        for d in ONCALL_CORPUS
    ]
    return HybridRetriever(
        documents=documents,
        embed_fn=_embed_stub,
        alpha=0.4,  # same BM25-leaning weight the qa.py example uses
        fusion=fusion,
    )


def compare(query: str, k: int = 5) -> None:
    """Print weighted-fusion vs RRF rankings for one query, side by side."""
    weighted = _build("weighted").retrieve(query, k=k)
    rrf = _build("rrf").retrieve(query, k=k)

    print(f"\nQuery: {query!r}")
    print("-" * 64)
    print(f"{'rank':<5}{'weighted (alpha=0.4)':<30}{'rrf (k=60)':<30}")
    print(f"{'':<5}{'doc      score':<30}{'doc      score':<30}")
    print("-" * 64)
    for rank in range(max(len(weighted), len(rrf))):
        w = weighted[rank] if rank < len(weighted) else None
        r = rrf[rank] if rank < len(rrf) else None
        w_cell = f"{w.document.id:<9}{w.score:.3f}" if w else ""
        r_cell = f"{r.document.id:<9}{r.score:.3f}" if r else ""
        # Flag where the two strategies put a different doc at this rank.
        flag = ""
        if w and r and w.document.id != r.document.id:
            flag = "  <- differs"
        print(f"{rank:<5}{w_cell:<30}{r_cell:<30}{flag}")


def main() -> None:
    # Exact-token queries are where the two strategies diverge most.
    queries = [
        "What does ERR_5521 mean?",
        "OOMKilled on checkout-worker",
        "How do I roll back a bad deploy?",
    ]
    for q in queries:
        compare(q)
    print()


if __name__ == "__main__":
    main()
