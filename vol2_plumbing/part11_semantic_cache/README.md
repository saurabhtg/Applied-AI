# Part 11 — Thanks for the (Semantic) Memory

> *Saving our latency budget (and our wallet) by remembering the past.*

## What's here

| File | What it does |
|------|--------------|
| [`semantic_cache.py`](./semantic_cache.py) | Embedding-based cache. A new query is a hit if its cosine similarity to a stored query exceeds the threshold. In-memory storage; TTL and max-entries eviction. |

## Why semantic, not key-value

Users phrase the same question fifty different ways:

- "How do I reset my UPI PIN?"
- "Forgot UPI PIN, how to change?"
- "PIN bhool gaya, kya karu"
- "Change UPI PIN steps"

A key-value lookup on the query string hits zero of these on a repeat
ask. A semantic cache hits all four — they're all asking the same
thing.

## Use as a snippet

```python
from vol2_plumbing.part11_semantic_cache.semantic_cache import SemanticCache

cache = SemanticCache(
    embed_fn=your_embedder,
    threshold=0.92,
    ttl_seconds=24 * 3600,
)

# Try the cache first.
hit = cache.lookup(query)
if hit.hit:
    return hit.response

# Cache miss — call the model.
response = client.complete(query)
cache.store(query, response)
return response
```

## Tradeoffs

- **Threshold too loose** → false hits (you return a stale answer to a
  similar-but-different question). For lending or compliance Q&A, false
  hits are dangerous; raise the threshold to 0.95.
- **Threshold too strict** → low hit rate; you barely save anything.
  For casual product FAQ, 0.88 is often fine.
- **No semantic match for "what is X today"** → user asks "what's the
  RBI repo rate", you return last week's cached answer. Time-sensitive
  questions should bypass the cache. Tag them in your query
  pre-processing.

For production, back the in-memory storage with Redis or a vector DB.
The interface stays the same.
