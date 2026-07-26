"""
core/rag/semantic_cache.py

Semantic caching. Save latency budget and API spend by remembering past
answers — and treating "close enough" queries as cache hits.

Why semantic (not key-value): users phrase the same question fifty
different ways. "How do I reset my UPI PIN?" and "Forgot UPI PIN, how
to change?" should hit the same cache entry, but a key-value lookup
on the query string never will.

Tradeoffs:
    - You will get false hits if your threshold is too loose.
    - You will get few hits if your threshold is too strict.
    - 0.92 cosine similarity is a reasonable starting point. Tune for
      your corpus and acceptable false-hit rate.

This is an in-process cache. For production, back it with Redis or
a vector DB and add TTLs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from common.tracing import span


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    response: str
    created_at: float
    hit_count: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class CacheLookup:
    hit: bool
    response: str | None
    matched_query: str | None
    similarity: float
    age_seconds: float | None


class SemanticCache:
    """
    In-memory semantic cache backed by cosine similarity.

    For production, replace the internal numpy storage with a real
    vector DB (Pinecone, Weaviate, pgvector, etc.) and add TTLs.
    """

    def __init__(
        self,
        embed_fn: Callable[[Sequence[str]], np.ndarray],
        *,
        threshold: float = 0.92,
        ttl_seconds: float | None = 24 * 3600,
        max_entries: int = 10_000,
    ):
        """
        Args:
            embed_fn: Function that embeds a list of strings.
            threshold: Cosine similarity above which a query is a cache hit.
                0.92 is a reasonable default. Lower = more hits but more
                false hits. Higher = fewer false hits but lower hit rate.
            ttl_seconds: Entries older than this are evicted. None = no TTL.
            max_entries: When the cache exceeds this, oldest entries are
                evicted first (FIFO).
        """
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")

        self.embed_fn = embed_fn
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: list[CacheEntry] = []

    def lookup(self, query: str) -> CacheLookup:
        """Try to find a cache hit for the query."""
        with span("cache.lookup", attributes={"cache.query": query}) as s:
            self._evict_expired()
            if not self._entries:
                s.set_attribute("cache.hit", False)
                return CacheLookup(False, None, None, 0.0, None)

            query_vec = self.embed_fn([query])[0]
            similarities = self._cosine_sim_all(query_vec)
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            s.set_attribute("cache.best_similarity", best_sim)

            if best_sim >= self.threshold:
                entry = self._entries[best_idx]
                entry.hit_count += 1
                s.set_attribute("cache.hit", True)
                s.set_attribute("cache.matched_query", entry.query)
                return CacheLookup(
                    hit=True,
                    response=entry.response,
                    matched_query=entry.query,
                    similarity=best_sim,
                    age_seconds=time.time() - entry.created_at,
                )

            s.set_attribute("cache.hit", False)
            return CacheLookup(False, None, None, best_sim, None)

    def store(self, query: str, response: str, metadata: dict | None = None) -> None:
        """Store a query/response pair in the cache."""
        embedding = self.embed_fn([query])[0]
        entry = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._evict_overflow()

    def stats(self) -> dict:
        """Return cache stats for monitoring."""
        total_hits = sum(e.hit_count for e in self._entries)
        return {
            "entries": len(self._entries),
            "total_hits": total_hits,
            "max_entries": self.max_entries,
            "threshold": self.threshold,
        }

    def clear(self) -> None:
        self._entries.clear()

    def _cosine_sim_all(self, query_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity between query vector and all cached embeddings."""
        if not self._entries:
            return np.array([])
        matrix = np.array([e.embedding for e in self._entries])
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        m_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        return m_norms @ q_norm

    def _evict_expired(self) -> None:
        if self.ttl_seconds is None:
            return
        cutoff = time.time() - self.ttl_seconds
        self._entries = [e for e in self._entries if e.created_at >= cutoff]

    def _evict_overflow(self) -> None:
        if len(self._entries) > self.max_entries:
            # Drop oldest entries first.
            self._entries.sort(key=lambda e: e.created_at)
            self._entries = self._entries[-self.max_entries:]
