# Part 9 — RAG is a Search Problem

> *The sobering reality: our expensive vector database won't save a
> fundamentally broken retrieval strategy.*

## The article (two parts)

| File | Covers |
|------|--------|
| [`article_part1.md`](./article_part1.md) | **The foundations.** How retrieval works (embeddings, cosine, BM25), why naive RAG breaks, and the load-bearing spine: hybrid retrieve → fuse (RRF) → rerank. |
| [`article_part2.md`](./article_part2.md) | **The modern stack (2024–2026).** Chunking & Contextual Retrieval, late interaction (ColBERT/MUVERA/BGE-M3/ColPali), query rewriting, agentic/CRAG, GraphRAG/RAPTOR, long-context, and the stale-runbook trap. |

## What's here

| File | What it does |
|------|--------------|
| [`retriever.py`](./retriever.py) | Hybrid BM25 + dense retrieval. Two ways to fuse: `fusion="weighted"` (min-max normalize, then alpha-blend the scores) or `fusion="rrf"` (reciprocal rank fusion — combine the *rankings*, ignore raw scores). Bring your own embedding function. |
| [`reranker.py`](./reranker.py) | Cross-encoder re-ranker. Score the top-50 from your retriever with a more expensive but more accurate model, return the top-K. |

## Example

| File | Pattern | Story |
|------|---------|-------|
| [`examples/oncall_rag/`](./examples/oncall_rag/) | Hybrid retrieval over a runbook/postmortem fixture corpus | Pure-vector RAG failed on an on-call corpus full of error codes and kubectl states; hybrid fixed it |
| [`examples/oncall_rag/fusion_compare.py`](./examples/oncall_rag/fusion_compare.py) | Weighted blending vs RRF, side by side | On exact-token queries like `ERR_5521`, BM25's lopsided scores break min-max normalization; RRF counts ranks instead and keeps the exact match pinned at #1 |
| [`examples/oncall_rag/rerank_demo.py`](./examples/oncall_rag/rerank_demo.py) | Full retrieve → rerank → answer, with a real cross-encoder | Retriever casts a wide net (top 10); a cross-encoder rereads each query+passage pair and promotes the genuinely relevant ones to the top 4 |

## Run

```bash
# Hybrid retrieval Q&A over the on-call runbook corpus
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.qa

# Watch weighted fusion and RRF disagree on the same exact-token queries
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.fusion_compare

# Retrieve wide, then rerank with a cross-encoder before answering
# (uses sentence-transformers if installed; falls back to a stub otherwise)
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.rerank_demo
```

## Why hybrid

Pure vector search misses exact-match queries — error codes, kubectl
states, service names, env vars. An on-call corpus full of `ERR_5521`,
`OOMKilled`, `CrashLoopBackOff`, `payments-api` will frustrate any
embedding-only retriever. BM25 nails these.

Pure BM25 misses semantic paraphrases. "Why is the site slow under load?"
won't match a passage about "p99 latency under traffic". Dense vectors
nail these.

Together they cover each other's blind spots. Min-max normalize each
retriever's scores (since they live on different scales), then combine
with a tunable alpha — or skip the scores entirely and use RRF. For
token-heavy corpora, tilt toward BM25 (`alpha < 0.5`). For
paraphrase-heavy corpora, tilt toward dense.

## What to add next

- Part 10 ([`../part10_ragas/`](../part10_ragas/)): measure how well
  this retrieval pipeline is *actually* working. Faithfulness, answer
  relevance, context precision, context recall.
- Part 11 ([`../part11_semantic_cache/`](../part11_semantic_cache/)):
  cache the same question asked fifty different ways.
