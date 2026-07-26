"""
examples/oncall_rag/qa.py

Q&A over a small runbook/postmortem corpus using hybrid retrieval.

The realistic version of this in production:
    - Thousands of pages of runbooks, postmortems, architecture docs,
      and stale wiki pages nobody has touched in two years.
    - Updates constantly. Old runbooks superseded but still in the index
      (see rb_010 vs rb_011 in corpus.py).
    - Queries are whatever a stressed engineer types at 3 a.m. — "what
      does ERR_5521 mean" right next to "why is the site slow under load".
    - A wrong answer during an incident makes the incident worse.

This example uses the small fixture corpus in corpus.py. The patterns
are the same as for the production version:
    1. Hybrid retrieval (BM25 + dense). Pure vectors miss "ERR_5521",
       "OOMKilled", "CrashLoopBackOff", "payments-api" — exact tokens
       with no semantic neighbours.
    2. Optional re-ranker on top.
    3. Strict instruction to answer from context only; flag insufficient
       context rather than guessing a fix that could deepen an outage.

This script runs without a real embedding model — it falls back to a
hashed bag-of-words stub so you can see the BM25 component at work. To
enable true dense retrieval, plug in sentence-transformers or your
embedding API of choice (the embed_fn signature is in retriever.py).

Run:
    python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.qa
"""

from __future__ import annotations

import logging
import zlib

import numpy as np

from common.llm_client import LLMClient
from vol2_plumbing.part09_rag_retrieval.retriever import Document, HybridRetriever
from .corpus import ONCALL_CORPUS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


ANSWER_PROMPT = """\
You are an on-call assistant answering an engineer's question from the
team's internal runbooks, postmortems, and architecture docs.

Use ONLY the context passages below. If the context does not contain
sufficient information to answer, say so plainly — do not guess or use
general knowledge. A wrong fix during an incident makes the incident worse.

When two passages conflict, prefer the more recent one (a postmortem that
supersedes an old runbook) and say which you used and why.

Always cite the runbook IDs you used in this format at the end of your
answer: [Sources: rb_001, rb_004]

Context:
{context}

Question: {question}

Answer:
"""


def _embed_stub(texts: list[str]) -> np.ndarray:
    """
    Placeholder embedder using hashed bag-of-words. Replaceable with a
    real embedding model. The retriever still works; semantic recall
    will be poor compared to a proper encoder.

    Note: we hash words with zlib.crc32 rather than the built-in hash(),
    which Python randomizes per process. crc32 is deterministic, so this
    stub gives the same rankings on every run — handy for a demo whose
    output you want to reproduce.
    """
    dim = 256
    rng = np.random.RandomState(42)
    vectors = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        for word in text.lower().split():
            idx = zlib.crc32(word.encode("utf-8")) % dim
            vectors[i, idx] += 1.0
        # Add a tiny bit of random noise for variation.
        vectors[i] += rng.randn(dim) * 0.01
    # L2 normalize.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-9)


def build_retriever() -> HybridRetriever:
    """Build a hybrid retriever over the on-call corpus."""
    documents = [
        Document(
            id=d.id,
            text=d.text,
            metadata={"source": d.source},
        )
        for d in ONCALL_CORPUS
    ]
    return HybridRetriever(
        documents=documents,
        embed_fn=_embed_stub,
        alpha=0.4,  # BM25-leaning. Exact tokens (error codes, k8s) matter here.
    )


def answer_question(question: str, top_k: int = 4) -> dict:
    """Run the full retrieve → answer pipeline for one question."""
    retriever = build_retriever()
    client = LLMClient()

    results = retriever.retrieve(question, k=top_k)
    if not results:
        return {
            "question": question,
            "answer": "No relevant context found.",
            "sources": [],
        }

    context_block = "\n\n".join(
        f"[{r.document.id}] (score={r.score:.2f})\n{r.document.text}"
        for r in results
    )

    prompt = ANSWER_PROMPT.format(context=context_block, question=question)
    response = client.complete(
        prompt=prompt,
        max_tokens=500,
        temperature=0.0,
        prompt_name="oncall_rag_qa",
        prompt_version="1.0",
    )

    return {
        "question": question,
        "answer": response.text,
        "sources": [r.document.id for r in results],
        "scores": [round(r.score, 3) for r in results],
    }


def main() -> None:
    questions = [
        "How do I roll back a bad deployment?",
        "What does ERR_5521 mean?",
        "What's the process when we declare a Sev-1 incident?",
        "A payments-api pod keeps restarting on its own — what should I check?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        result = answer_question(q)
        print(f"A: {result['answer']}")
        print(f"Retrieved: {', '.join(result['sources'])}")


if __name__ == "__main__":
    main()
