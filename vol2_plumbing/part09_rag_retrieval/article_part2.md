# RAG is a Search Problem — Part 2: Making the Search Actually Trustworthy

*Applied AI, Volume II — Part 9 (2 of 2)*

---

> **TL;DR — Part 2, the modern stack**
>
> [Part 1](./article_part1.md) built the spine: **hybrid retrieve → rerank →
> generate.** It's enough to beat most RAG in production — and still not enough
> to *trust* at 3 a.m. Here's what closes the gap, roughly in build order:
>
> - **Chunk so each piece stands alone.** Fixed-size cuts orphan facts
>   ("set timeout to 30s" — for *what?*). Fix with **semantic** / **late
>   chunking** or **Contextual Retrieval** (prepend a one-line "where this came
>   from" — cheap thanks to prompt caching).
> - **Go beyond one-vector-per-chunk** when precision matters: **late
>   interaction** (ColBERT → MUVERA, BGE-M3, ColPali for scanned pages).
> - **Fix the question, not just the index:** **HyDE**, decompose, step-back.
> - **Let retrieval check itself:** **CRAG** grades the fetch, **Self-RAG**
>   reflects, **agentic RAG** loops and re-queries instead of guessing once.
> - **For "what keeps breaking *everywhere*?"** no chunk retriever helps —
>   you need **GraphRAG / RAPTOR** structure built at index time.
> - **Long context didn't kill RAG**, it changed its job: retrieve to narrow
>   scope, let the window do the reasoning.
> - **The trap that bites hardest:** retrieval cheerfully serves the
>   *deprecated* runbook over the postmortem that replaced it. Fix is
>   **metadata + recency**, not a better embedding.
>
> Build in this order, stop when your eval says you're good enough. Most teams
> over-build the late stages and starve the early ones. **The wins are early.**

---

## Where we left off

In Part 1, Tanvi's on-call bot went from "returns garbage on `ERR_5521`" to a
solid two-stage pipeline: hybrid retrieval (BM25 + dense, fused with RRF) feeding
a cross-encoder reranker. That spine fixes the headline failure — exact tokens
the embedding model never learned.

But the field didn't stop in 2023, and a bot that's merely *good* isn't yet
*trustworthy*. The failures left are subtler and, frankly, scarier: a retrieved
chunk that's useless because it got sliced away from its heading; a question
phrased so badly nothing matches; a confidently wrong answer built on a chunk
the retriever should never have trusted; and the one that keeps me up — a
*deprecated* runbook outranking the postmortem that exists to stop you running
it. This is the part of RAG that 2024–2026 research actually moved.

---

## The chunking problem nobody puts in the tutorial

Back up to step 1 of the naive pipeline from Part 1: *"chop the docs into
chunks."* Everyone breezes past it. In practice it's where most RAG systems
quietly bleed quality.

The default move is **fixed-size chunking**: every 500 tokens, cut. Simple, and a
small catastrophe, because docs don't come in 500-token units of meaning. The cut
lands mid-sentence, mid-table, mid-command. Worse, it severs context. Here's a
real example from Tanvi's corpus:

```
Chunk 31: "...set the connection timeout to 30s and retry up to 3 times."
```

A 30-second timeout on *what*? The line that named the subject — "For the
connection from **payments-api to its primary database**..." — got sliced off
into chunk 30. Now chunk 31 is a free-floating config tip. The embedding model
encodes "set the timeout to 30s, retry 3 times" with no clue it's about
payments-api's database. So when someone asks "what's the DB timeout for
payments-api?", chunk 31 — *the literal answer* — doesn't match, because in
isolation it isn't about payments-api at all.

<svg viewBox="0 0 900 280" width="100%" role="img" aria-label="Chunking strategies compared" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="280" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">How you cut the doc decides what you can find</text>
  <text x="30" y="62" font-size="12" fill="#dc2626" font-weight="600">① fixed-size (naive)</text>
  <rect x="30" y="72" width="120" height="34" fill="#fee2e2" stroke="#dc2626"/>
  <rect x="150" y="72" width="120" height="34" fill="#fecaca" stroke="#dc2626"/>
  <rect x="270" y="72" width="120" height="34" fill="#fee2e2" stroke="#dc2626"/>
  <line x1="150" y1="68" x2="150" y2="110" stroke="#991b1b" stroke-width="2" stroke-dasharray="3 2"/>
  <text x="410" y="94" font-size="11" fill="#991b1b">cuts mid-sentence, severs context</text>
  <text x="30" y="142" font-size="12" fill="#d97706" font-weight="600">② semantic</text>
  <rect x="30" y="152" width="150" height="34" fill="#fef3c7" stroke="#d97706"/>
  <rect x="180" y="152" width="90" height="34" fill="#fde68a" stroke="#d97706"/>
  <rect x="270" y="152" width="170" height="34" fill="#fef3c7" stroke="#d97706"/>
  <text x="460" y="174" font-size="11" fill="#92400e">cuts at topic boundaries — keeps ideas whole</text>
  <text x="30" y="222" font-size="12" fill="#16a34a" font-weight="600">③ contextual</text>
  <rect x="30" y="232" width="120" height="34" fill="#dcfce7" stroke="#16a34a"/>
  <rect x="155" y="232" width="20" height="34" fill="#86efac" stroke="#16a34a"/>
  <rect x="180" y="232" width="120" height="34" fill="#dcfce7" stroke="#16a34a"/>
  <rect x="305" y="232" width="20" height="34" fill="#86efac" stroke="#16a34a"/>
  <text x="340" y="248" font-size="11" fill="#166534">each chunk gets a prepended context blurb</text>
  <text x="340" y="263" font-size="11" fill="#166534">(green) so it stands on its own</text>
</svg>

**Semantic chunking** cuts where the topic turns instead of every N tokens: embed
consecutive sentences, split where neighbours stop being similar. Chunks line up
with ideas. Better — though the orphaned-timeout problem can still bite at a real
topic boundary.

**Late chunking** (Jina AI, 2024) flips the order of operations, and the flip is
the whole trick. Naively you *chunk then embed*, so each chunk is encoded in
isolation. Late chunking *embeds then chunks*: run the whole document through a
long-context model first, get one contextual vector per token (every token vector
has already "seen" the whole doc through attention), and only *then* pool those
token vectors into chunks.

<svg viewBox="0 0 900 300" width="100%" role="img" aria-label="Late chunking order of operations" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p2a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="300" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Same chunks, opposite order — and it changes everything</text>
  <rect x="30" y="48" width="410" height="232" rx="8" fill="#ffffff" stroke="#dc2626"/>
  <text x="235" y="72" text-anchor="middle" font-size="12" fill="#991b1b" font-weight="700">naive: chunk, THEN embed</text>
  <rect x="110" y="86" width="250" height="32" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="235" y="107" text-anchor="middle" font-size="11" fill="#991b1b">1. cut the doc into chunks</text>
  <line x1="235" y1="118" x2="235" y2="136" stroke="#475569" stroke-width="2" marker-end="url(#p2a)"/>
  <rect x="110" y="138" width="250" height="32" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="235" y="159" text-anchor="middle" font-size="11" fill="#991b1b">2. embed each chunk alone</text>
  <line x1="235" y1="170" x2="235" y2="188" stroke="#475569" stroke-width="2" marker-end="url(#p2a)"/>
  <rect x="80" y="190" width="310" height="60" rx="6" fill="#fff5f5" stroke="#dc2626"/><text x="235" y="214" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="600">chunk 31 was encoded with no idea</text><text x="235" y="232" text-anchor="middle" font-size="11" fill="#991b1b">it belongs to payments-api ✗</text>
  <rect x="460" y="48" width="410" height="232" rx="8" fill="#ffffff" stroke="#16a34a"/>
  <text x="665" y="72" text-anchor="middle" font-size="12" fill="#166534" font-weight="700">late: embed, THEN chunk</text>
  <rect x="540" y="86" width="250" height="32" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="665" y="107" text-anchor="middle" font-size="11" fill="#166534">1. embed the WHOLE doc → token vectors</text>
  <line x1="665" y1="118" x2="665" y2="136" stroke="#475569" stroke-width="2" marker-end="url(#p2a)"/>
  <rect x="540" y="138" width="250" height="32" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="665" y="159" text-anchor="middle" font-size="11" fill="#166534">2. pool token vectors into chunks</text>
  <line x1="665" y1="170" x2="665" y2="188" stroke="#475569" stroke-width="2" marker-end="url(#p2a)"/>
  <rect x="510" y="190" width="310" height="60" rx="6" fill="#f0fdf4" stroke="#16a34a"/><text x="665" y="214" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">every token already saw the whole doc,</text><text x="665" y="232" text-anchor="middle" font-size="11" fill="#166534">so chunk 31 carries the context ✓</text>
</svg>

**Contextual Retrieval** (Anthropic, late 2024) is the most practical of the
three, and it earns its own section.

---

## Contextual Retrieval: spend a few cents to fix recall

This one's from Anthropic, and it's almost insultingly simple — which is exactly
why it works. Before you embed a chunk, you ask a cheap LLM to write a sentence
or two saying where this chunk sits in its document, and you **prepend that to
the chunk.** Then you embed *and* BM25-index the combined text.

<svg viewBox="0 0 900 300" width="100%" role="img" aria-label="Contextual Retrieval" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar3" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="300" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Contextual Retrieval (Anthropic): teach each chunk where it lives</text>
  <rect x="40" y="60" width="200" height="90" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="140" y="82" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="600">raw chunk</text>
  <text x="140" y="106" text-anchor="middle" font-size="11" fill="#991b1b">"set the timeout</text>
  <text x="140" y="122" text-anchor="middle" font-size="11" fill="#991b1b">to 30s, retry 3×"</text>
  <text x="140" y="142" text-anchor="middle" font-size="10" fill="#dc2626">30s for WHICH service?</text>
  <line x1="240" y1="105" x2="290" y2="105" stroke="#475569" stroke-width="2" marker-end="url(#ar3)"/>
  <rect x="295" y="70" width="160" height="70" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="375" y="95" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">cheap LLM writes</text>
  <text x="375" y="113" text-anchor="middle" font-size="11" fill="#5b21b6">a context blurb</text>
  <text x="375" y="131" text-anchor="middle" font-size="10" fill="#7c3aed">(prompt-cache the doc!)</text>
  <line x1="455" y1="105" x2="505" y2="105" stroke="#475569" stroke-width="2" marker-end="url(#ar3)"/>
  <rect x="510" y="55" width="350" height="100" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="685" y="78" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">contextualized chunk (what you index)</text>
  <text x="525" y="100" font-size="10.5" fill="#166534">"From the payments-api database runbook,</text>
  <text x="525" y="116" font-size="10.5" fill="#166534">on the primary DB connection: set the</text>
  <text x="525" y="132" font-size="10.5" fill="#166534">connection timeout to 30s, retry 3×."</text>
  <rect x="180" y="190" width="540" height="90" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="450" y="214" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">Anthropic's reported recall lift (top-20 retrieval failure rate)</text>
  <text x="200" y="240" font-size="11" fill="#475569">contextual embeddings ............ −35%  failures</text>
  <text x="200" y="260" font-size="11" fill="#475569">+ contextual BM25 ................ −49%  failures</text>
  <text x="560" y="250" font-size="11" fill="#16a34a" font-weight="600">+ reranking → −67%</text>
</svg>

Embed the contextualized chunk and its vector lands near "payments-api DB
timeout." BM25 indexes "payments-api," "database," "connection," "timeout," "30s"
— all of it. The orphan tip is now a fully-addressed citizen of the index.

The reason it's *shippable* and not just clever is the indexing pipeline, and one
trick inside it:

<svg viewBox="0 0 900 250" width="100%" role="img" aria-label="Contextual Retrieval indexing pipeline" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p2b" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="250" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Index-time pipeline (done once, offline)</text>
  <rect x="30" y="95" width="140" height="70" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="100" y="122" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">whole doc</text>
  <text x="100" y="140" text-anchor="middle" font-size="9.5" fill="#92400e">cached ONCE</text>
  <text x="100" y="154" text-anchor="middle" font-size="9.5" fill="#92400e">≈ $1 / 1M tokens</text>
  <line x1="170" y1="130" x2="205" y2="130" stroke="#475569" stroke-width="2" marker-end="url(#p2b)"/>
  <rect x="210" y="95" width="160" height="70" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="290" y="120" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">per chunk: LLM</text>
  <text x="290" y="138" text-anchor="middle" font-size="10" fill="#5b21b6">writes 1-line context</text>
  <text x="290" y="154" text-anchor="middle" font-size="9.5" fill="#7c3aed">(doc already cached)</text>
  <line x1="370" y1="130" x2="405" y2="130" stroke="#475569" stroke-width="2" marker-end="url(#p2b)"/>
  <rect x="410" y="95" width="170" height="70" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="495" y="125" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">contextualized</text>
  <text x="495" y="143" text-anchor="middle" font-size="11" fill="#166534">chunk</text>
  <line x1="580" y1="120" x2="615" y2="95" stroke="#475569" stroke-width="2" marker-end="url(#p2b)"/>
  <line x1="580" y1="140" x2="615" y2="165" stroke="#475569" stroke-width="2" marker-end="url(#p2b)"/>
  <rect x="620" y="68" width="240" height="44" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="740" y="95" text-anchor="middle" font-size="11" fill="#1e3a8a" font-weight="600">dense index (embeddings)</text>
  <rect x="620" y="148" width="240" height="44" rx="8" fill="#fff7ed" stroke="#d97706"/>
  <text x="740" y="175" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">BM25 index (keywords)</text>
  <text x="450" y="225" text-anchor="middle" font-size="11" fill="#475569">Prompt-cache the document so you read it once, not once per chunk — that's what makes it cost a coffee.</text>
</svg>

Two notes that matter in practice:

- **Prompt caching makes it cheap.** Naively, contextualizing every chunk means
  re-sending the whole document to the LLM once per chunk — brutal. Cache the
  document once and only vary the chunk, and you pay the document-read cost a
  single time. Anthropic's writeup pegs this at roughly a dollar per million
  document tokens. For Tanvi's entire runbook corpus that's a coffee, paid once.
- **Contextual BM25 is half the win.** The biggest single jump in the recall
  numbers comes from also feeding the contextualized text to *BM25*, not just the
  embeddings. Same hybrid lesson from Part 1 — lexical plus semantic — except now
  both halves operate on chunks that finally carry their context.

> **War Story**
>
> Tanvi re-indexed over a weekend with contextual retrieval. The whole "30s
> timeout for *what*?" class of bug — chunks that technically got retrieved but
> were useless ripped out of their parent section — basically fell off her error
> dashboard. The single most cost-effective change she made all quarter, and it
> touched zero lines of her query-time code.

---

## Beyond one-vector-per-chunk: late interaction

Step back and notice an assumption we've dragged along since Part 1: each chunk
becomes *one* vector. That's the bi-encoder bargain — crush a whole passage into a
single point so search stays fast. We already paid for it: nuance gets flattened
before the query ever arrives.

A different lineage refuses the crush. **ColBERT** (and ColBERTv2) keeps *one
vector per token* and compares the query against the passage token by token, with
an operator called **MaxSim**: for each query token, find its best-matching
passage token, then sum those bests. This is **late interaction** — the
fine-grained, cross-encoder-style comparison happens late, at query time, but on
precomputed token vectors, so you keep most of the bi-encoder's speed and buy
back much of the cross-encoder's sharpness.

<svg viewBox="0 0 900 280" width="100%" role="img" aria-label="Late interaction MaxSim" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="280" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Late interaction: match token-to-token, not blob-to-blob</text>
  <text x="120" y="62" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">query tokens</text>
  <rect x="60" y="75" width="120" height="28" rx="5" fill="#dbeafe" stroke="#2563eb"/><text x="120" y="94" text-anchor="middle" font-size="11" fill="#1e3a8a">ERR_5521</text>
  <rect x="60" y="113" width="120" height="28" rx="5" fill="#dbeafe" stroke="#2563eb"/><text x="120" y="132" text-anchor="middle" font-size="11" fill="#1e3a8a">gateway</text>
  <rect x="60" y="151" width="120" height="28" rx="5" fill="#dbeafe" stroke="#2563eb"/><text x="120" y="170" text-anchor="middle" font-size="11" fill="#1e3a8a">timeout</text>
  <text x="700" y="62" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">passage tokens</text>
  <rect x="620" y="70" width="160" height="26" rx="5" fill="#fef3c7" stroke="#d97706"/><text x="700" y="88" text-anchor="middle" font-size="10.5" fill="#92400e">upstream</text>
  <rect x="620" y="102" width="160" height="26" rx="5" fill="#fde68a" stroke="#d97706"/><text x="700" y="120" text-anchor="middle" font-size="10.5" fill="#92400e">ERR_5521</text>
  <rect x="620" y="134" width="160" height="26" rx="5" fill="#fde68a" stroke="#d97706"/><text x="700" y="152" text-anchor="middle" font-size="10.5" fill="#92400e">gateway</text>
  <rect x="620" y="166" width="160" height="26" rx="5" fill="#fef3c7" stroke="#d97706"/><text x="700" y="184" text-anchor="middle" font-size="10.5" fill="#92400e">payments-api</text>
  <line x1="180" y1="89" x2="620" y2="115" stroke="#16a34a" stroke-width="2"/>
  <line x1="180" y1="127" x2="620" y2="147" stroke="#16a34a" stroke-width="2"/>
  <line x1="180" y1="165" x2="620" y2="115" stroke="#94a3b8" stroke-width="1.5" stroke-dasharray="3 2"/>
  <text x="400" y="230" text-anchor="middle" font-size="11" fill="#166534">each query token finds its best passage token (MaxSim), then sum</text>
  <text x="400" y="250" text-anchor="middle" font-size="11" fill="#475569">"ERR_5521" finds "ERR_5521" exactly — the single-vector crush could never do this</text>
</svg>

Notice what late interaction hands you for free: because `ERR_5521` the query
token matches `ERR_5521` the passage token directly, it claws back some of
BM25's exact-match superpower *inside a neural retriever.* The price is storage —
one vector per token means a fatter index — and the 2024–2025 work is all about
paying that price down:

<svg viewBox="0 0 900 270" width="100%" role="img" aria-label="Single vs multi vector storage" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p2c" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="270" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">One vector, many vectors, or the best of both</text>
  <rect x="30" y="50" width="270" height="200" rx="8" fill="#ffffff" stroke="#2563eb"/>
  <text x="165" y="74" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="700">bi-encoder</text>
  <rect x="120" y="90" width="90" height="26" rx="5" fill="#dbeafe" stroke="#2563eb"/><text x="165" y="108" text-anchor="middle" font-size="10" fill="#1e3a8a">chunk</text>
  <line x1="165" y1="116" x2="165" y2="134" stroke="#475569" stroke-width="2" marker-end="url(#p2c)"/>
  <rect x="135" y="136" width="60" height="22" rx="4" fill="#93c5fd"/><text x="165" y="152" text-anchor="middle" font-size="10" fill="#1e3a8a">1 vector</text>
  <text x="165" y="190" text-anchor="middle" font-size="10.5" fill="#16a34a">small index 👍</text>
  <text x="165" y="210" text-anchor="middle" font-size="10.5" fill="#dc2626">blunt 👎</text>
  <rect x="315" y="50" width="270" height="200" rx="8" fill="#ffffff" stroke="#7c3aed"/>
  <text x="450" y="74" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="700">ColBERT (late interaction)</text>
  <rect x="405" y="90" width="90" height="26" rx="5" fill="#ede9fe" stroke="#7c3aed"/><text x="450" y="108" text-anchor="middle" font-size="10" fill="#5b21b6">chunk</text>
  <line x1="450" y1="116" x2="450" y2="134" stroke="#475569" stroke-width="2" marker-end="url(#p2c)"/>
  <rect x="378" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="402" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="426" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="450" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="474" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="498" y="136" width="20" height="22" rx="3" fill="#c4b5fd"/>
  <text x="450" y="190" text-anchor="middle" font-size="10.5" fill="#16a34a">sharp 👍</text>
  <text x="450" y="210" text-anchor="middle" font-size="10.5" fill="#dc2626">one vector per token → big index 👎</text>
  <rect x="600" y="50" width="270" height="200" rx="8" fill="#ffffff" stroke="#16a34a"/>
  <text x="735" y="74" text-anchor="middle" font-size="12" fill="#166534" font-weight="700">MUVERA (2025)</text>
  <rect x="663" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="687" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="711" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="735" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="759" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/><rect x="783" y="90" width="20" height="22" rx="3" fill="#c4b5fd"/>
  <line x1="735" y1="116" x2="735" y2="134" stroke="#475569" stroke-width="2" marker-end="url(#p2c)"/>
  <rect x="705" y="136" width="60" height="22" rx="4" fill="#86efac"/><text x="735" y="152" text-anchor="middle" font-size="10" fill="#166534">1 fixed vector</text>
  <text x="735" y="190" text-anchor="middle" font-size="10.5" fill="#16a34a">squash N→1, keep most sharpness</text>
  <text x="735" y="210" text-anchor="middle" font-size="10.5" fill="#16a34a">serve on a normal ANN index 👍</text>
</svg>

- **ColBERTv2** compresses those token vectors hard (residual quantization) so
  the index is practical.
- **MUVERA** (Google, 2025) maps the whole multi-vector set into a *single*
  fixed-dimensional vector that approximates MaxSim, so you can serve late
  interaction on ordinary single-vector infrastructure and get most of the
  quality back.
- **BGE-M3** (2024) is the swiss-army embedding model: from one forward pass it
  spits out a dense vector, a sparse/lexical vector, *and* ColBERT-style
  multi-vectors. Dense + sparse + late-interaction hybrid, from a single model.

And the same idea jumped to documents-as-images. **ColPali / ColQwen** (2024–2025)
run late interaction over the *visual patches* of a rendered page — no OCR, no
layout parsing. That matters more than it sounds: half of Tanvi's "docs" are
screenshots of dashboards and architecture diagrams pasted into Confluence, and
OCR turns those into word salad. Retrieving over the page image sidesteps the
whole brittle text-extraction mess.

> **Handy Heuristic**
>
> Single-vector dense + BM25 + a reranker (Part 1) covers most corpora. Reach for
> late interaction (the ColBERT family) when exact-token precision *and* semantic
> matching both matter and a reranker isn't enough — long technical docs, code
> search, heavy-jargon domains. Reach for ColPali-style visual retrieval when
> your "documents" are really screenshots, tables, and diagrams.

---

## Fix the question, not just the index: query-side tricks

Everything so far improved the *index*. But half your retrieval misses are the
query's fault — the engineer's panicked half-sentence simply doesn't look like
the runbook that answers it. A cluster of techniques rewrites the query *before*
it hits the retriever.

**HyDE (Hypothetical Document Embeddings).** Delightfully backwards: instead of
embedding the *question*, you ask an LLM to hallucinate a *plausible answer*,
then embed that. Why? Because answers look like answers. A made-up paragraph
about an error code — even if the details are wrong — sits far closer in
embedding space to the real runbook than the terse question does.

<svg viewBox="0 0 900 240" width="100%" role="img" aria-label="HyDE flow" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p2d" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="240" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">HyDE: embed a fake answer, not the question</text>
  <rect x="30" y="100" width="150" height="50" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="105" y="122" text-anchor="middle" font-size="11" fill="#1e3a8a">"payments-api</text>
  <text x="105" y="138" text-anchor="middle" font-size="11" fill="#1e3a8a">throwing 5521s?"</text>
  <line x1="180" y1="80" x2="230" y2="60" stroke="#dc2626" stroke-width="2" stroke-dasharray="4 3" marker-end="url(#p2d)"/>
  <text x="300" y="52" text-anchor="middle" font-size="10.5" fill="#991b1b">embed the question directly →</text>
  <rect x="430" y="36" width="150" height="34" rx="6" fill="#fee2e2" stroke="#dc2626"/><text x="505" y="58" text-anchor="middle" font-size="10.5" fill="#991b1b">often misses ✗</text>
  <line x1="180" y1="135" x2="230" y2="135" stroke="#475569" stroke-width="2" marker-end="url(#p2d)"/>
  <rect x="235" y="108" width="160" height="54" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="315" y="130" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">LLM writes a</text>
  <text x="315" y="147" text-anchor="middle" font-size="11" fill="#5b21b6">plausible answer</text>
  <line x1="395" y1="135" x2="430" y2="135" stroke="#475569" stroke-width="2" marker-end="url(#p2d)"/>
  <rect x="435" y="100" width="200" height="70" rx="8" fill="#fff7ed" stroke="#d97706"/>
  <text x="535" y="122" text-anchor="middle" font-size="10" fill="#92400e">"ERR_5521 = upstream gateway</text>
  <text x="535" y="138" text-anchor="middle" font-size="10" fill="#92400e">timed out; check the status page…"</text>
  <text x="535" y="156" text-anchor="middle" font-size="9.5" fill="#d97706">(reads like a real runbook)</text>
  <line x1="635" y1="135" x2="670" y2="135" stroke="#475569" stroke-width="2" marker-end="url(#p2d)"/>
  <rect x="675" y="108" width="180" height="54" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="765" y="130" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">embed THIS → retrieve</text>
  <text x="765" y="147" text-anchor="middle" font-size="11" fill="#166534">the real runbook ✓</text>
</svg>

**Multi-query / decomposition.** One question, several retrievals. "The deploy is
bad *and* a payments-api pod is CrashLooping — what do I do?" retrieves a muddle
as a single query. Split it: "roll back a bad deploy" and "fix a CrashLoopBackOff
pod," retrieve each, union the results.

**Step-back prompting.** Before the specific, retrieve the general. "Is
checkout-worker affected by the 30s DB timeout?" → step back to "what are
payments-api's database timeout and pool settings?" → pull the broad config, then
answer the specific case from it.

<svg viewBox="0 0 900 230" width="100%" role="img" aria-label="Query transformations" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar4" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="230" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Rewrite the question so it looks like the answer</text>
  <rect x="40" y="55" width="160" height="130" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="120" y="80" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">raw question</text>
  <text x="120" y="110" text-anchor="middle" font-size="11" fill="#1e3a8a">often short,</text>
  <text x="120" y="128" text-anchor="middle" font-size="11" fill="#1e3a8a">vague, or</text>
  <text x="120" y="146" text-anchor="middle" font-size="11" fill="#1e3a8a">compound</text>
  <line x1="200" y1="120" x2="250" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#ar4)"/>
  <rect x="255" y="45" width="250" height="46" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="380" y="66" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">HyDE</text>
  <text x="380" y="83" text-anchor="middle" font-size="10.5" fill="#5b21b6">hallucinate an answer, embed it</text>
  <rect x="255" y="97" width="250" height="46" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="380" y="118" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">decompose</text>
  <text x="380" y="135" text-anchor="middle" font-size="10.5" fill="#5b21b6">split into sub-questions</text>
  <rect x="255" y="149" width="250" height="46" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="380" y="170" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">step-back</text>
  <text x="380" y="187" text-anchor="middle" font-size="10.5" fill="#5b21b6">generalize, then specialize</text>
  <line x1="505" y1="120" x2="555" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#ar4)"/>
  <rect x="560" y="80" width="150" height="80" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="635" y="115" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">retriever</text>
  <text x="635" y="135" text-anchor="middle" font-size="11" fill="#92400e">(now hits)</text>
  <line x1="710" y1="120" x2="760" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#ar4)"/>
  <rect x="765" y="90" width="110" height="60" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="820" y="125" text-anchor="middle" font-size="12" fill="#166534" font-weight="600">better recall</text>
</svg>

---

## When retrieval should argue with itself: self-correcting & agentic RAG

So far retrieval is a one-shot: fetch, then generate, and *hope* the fetch was
good. The honest problem is that sometimes it wasn't, and naive RAG has no idea —
it cheerfully feeds the model irrelevant runbooks, and the model cheerfully
writes a confident fix on top of them. Garbage in, fluent garbage out, shipped to
an engineer who's about to run it on production.

The 2024 fix is to put a feedback loop in the middle. Think of it as a ladder of
increasing self-awareness:

<svg viewBox="0 0 900 320" width="100%" role="img" aria-label="RAG maturity ladder" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="320" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">The RAG maturity ladder: from "fetch once" to "decide for itself"</text>
  <rect x="40" y="250" width="150" height="44" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="115" y="270" text-anchor="middle" font-size="11" fill="#991b1b" font-weight="600">naive RAG</text>
  <text x="115" y="286" text-anchor="middle" font-size="9.5" fill="#991b1b">fetch → stuff → answer</text>
  <rect x="210" y="206" width="150" height="44" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="285" y="226" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">+ rerank (Part 1)</text>
  <text x="285" y="242" text-anchor="middle" font-size="9.5" fill="#92400e">sharpen the top-k</text>
  <rect x="380" y="162" width="150" height="44" rx="6" fill="#fef9c3" stroke="#ca8a04"/>
  <text x="455" y="182" text-anchor="middle" font-size="11" fill="#854d0e" font-weight="600">CRAG</text>
  <text x="455" y="198" text-anchor="middle" font-size="9.5" fill="#854d0e">grade fetch, drop junk</text>
  <rect x="550" y="118" width="150" height="44" rx="6" fill="#ecfccb" stroke="#65a30d"/>
  <text x="625" y="138" text-anchor="middle" font-size="11" fill="#3f6212" font-weight="600">Self-RAG</text>
  <text x="625" y="154" text-anchor="middle" font-size="9.5" fill="#3f6212">reflect on its own draft</text>
  <rect x="720" y="74" width="150" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="795" y="94" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">agentic RAG</text>
  <text x="795" y="110" text-anchor="middle" font-size="9.5" fill="#166534">loop, re-query, use tools</text>
  <line x1="190" y1="252" x2="210" y2="240" stroke="#94a3b8"/>
  <line x1="360" y1="208" x2="380" y2="196" stroke="#94a3b8"/>
  <line x1="530" y1="164" x2="550" y2="152" stroke="#94a3b8"/>
  <line x1="700" y1="120" x2="720" y2="108" stroke="#94a3b8"/>
  <text x="450" y="312" text-anchor="middle" font-size="11" fill="#475569">each rung adds self-awareness — and latency. Climb only as high as the question demands.</text>
</svg>

**CRAG (Corrective RAG)** adds a lightweight grader. After retrieving, a small
model scores: are these passages actually relevant? If yes, proceed. If they're
junk, *don't* answer from them — trigger a corrective action: a web search, a
query rewrite, a fallback. The system *notices* it pulled garbage.

**Self-RAG** trains the generator itself to emit "reflection tokens": decide
*whether* retrieval is even needed, critique each retrieved passage for
relevance, then critique its own draft for whether the passages actually support
it. Retrieval, generation, and self-criticism interleave.

**Agentic RAG** is the general shape, and clearly where the field is heading in
2025–2026. Retrieval stops being a fixed pipeline stage and becomes a *tool the
model calls* — as many times as it needs, with the queries it chooses, deciding
for itself when it has enough to answer. (Yes, this is exactly the
function-calling and tool-use machinery from Parts 7 and 8, pointed at a
retriever.)

<svg viewBox="0 0 900 320" width="100%" role="img" aria-label="Agentic RAG loop" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar5" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="320" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Agentic RAG: retrieve, grade, decide, loop</text>
  <rect x="360" y="55" width="180" height="46" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="450" y="83" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">question</text>
  <line x1="450" y1="101" x2="450" y2="121" stroke="#475569" stroke-width="2" marker-end="url(#ar5)"/>
  <rect x="350" y="123" width="200" height="46" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="450" y="151" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="600">LLM plans a query</text>
  <line x1="550" y1="146" x2="640" y2="146" stroke="#475569" stroke-width="2" marker-end="url(#ar5)"/>
  <rect x="645" y="123" width="180" height="46" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="735" y="151" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">retrieve</text>
  <line x1="735" y1="169" x2="735" y2="200" stroke="#475569" stroke-width="2" marker-end="url(#ar5)"/>
  <rect x="645" y="202" width="180" height="46" rx="8" fill="#fee2e2" stroke="#dc2626"/>
  <text x="735" y="223" text-anchor="middle" font-size="12" fill="#991b1b" font-weight="600">grade: relevant?</text>
  <text x="735" y="240" text-anchor="middle" font-size="10.5" fill="#991b1b">enough to answer?</text>
  <path d="M645,225 H460 V169" fill="none" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 3" marker-end="url(#ar5)"/>
  <text x="500" y="200" font-size="10.5" fill="#dc2626">NO → rewrite / re-query / web search</text>
  <line x1="735" y1="248" x2="735" y2="275" stroke="#16a34a" stroke-width="2" marker-end="url(#ar5)"/>
  <rect x="620" y="277" width="230" height="36" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="735" y="300" text-anchor="middle" font-size="11.5" fill="#166534" font-weight="600">YES → generate grounded answer</text>
  <rect x="60" y="180" width="250" height="110" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="185" y="203" text-anchor="middle" font-size="11" fill="#0f172a" font-weight="700">vs. naive RAG</text>
  <text x="78" y="226" font-size="10.5" fill="#475569">fetch once → stuff → generate</text>
  <text x="78" y="246" font-size="10.5" fill="#475569">no grading, no retry,</text>
  <text x="78" y="266" font-size="10.5" fill="#dc2626">bad fetch = confident wrong fix</text>
</svg>

The trade-off is real and you should say it out loud: every loop costs latency
and tokens. You don't make a simple "what does ERR_5521 mean" lookup agentic. You
reserve it for the *hard, multi-hop, high-stakes* questions — "three services are
degraded, which postmortem matches this pattern and what was the fix?" — where
one-shot retrieval was never going to cut it anyway.

---

## When the question is global, not local: GraphRAG & RAPTOR

There's a class of question *no* amount of chunk retrieval can answer, and it's
worth knowing why before you waste a week on it. Ask:

> "What are the recurring root causes across all our payments-api postmortems
> this year?"

No single chunk holds that answer. It's *distributed* across the whole corpus —
a property of the pile, not of any one passage. Top-k retrieval physically can't
see it: it returns ten chunks, the model summarizes ten chunks, and the actual
pattern across a hundred postmortems sails right past.

**GraphRAG** (Microsoft, 2024) attacks this at index time. An LLM extracts
entities and relationships from every chunk, builds a knowledge graph, clusters
it into communities, and pre-writes a summary of each community. A *global*
question gets answered from the community summaries; a *local* one still uses
ordinary retrieval. You trade expensive index-time preprocessing for the ability
to answer questions that span the entire corpus.

**RAPTOR** (2024) takes a gentler route: recursively cluster and summarize chunks
into a tree — leaves are raw chunks, parents summarize their children, on up to a
root summary of the whole corpus. At query time you retrieve at whatever altitude
the question needs: a fine detail pulls a leaf, a "what's the overall pattern"
pulls a high node.

<svg viewBox="0 0 900 270" width="100%" role="img" aria-label="GraphRAG and RAPTOR" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="270" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Global questions need structure, not just top-k</text>
  <text x="220" y="58" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="600">GraphRAG: entities → graph → community summaries</text>
  <circle cx="120" cy="110" r="20" fill="#ede9fe" stroke="#7c3aed"/><text x="120" y="114" text-anchor="middle" font-size="8" fill="#5b21b6">pay-api</text>
  <circle cx="205" cy="95" r="20" fill="#ede9fe" stroke="#7c3aed"/><text x="205" y="99" text-anchor="middle" font-size="8" fill="#5b21b6">gateway</text>
  <circle cx="190" cy="165" r="20" fill="#ede9fe" stroke="#7c3aed"/><text x="190" y="169" text-anchor="middle" font-size="7.5" fill="#5b21b6">OOMKill</text>
  <circle cx="285" cy="140" r="20" fill="#ede9fe" stroke="#7c3aed"/><text x="285" y="144" text-anchor="middle" font-size="8" fill="#5b21b6">INC-4821</text>
  <line x1="120" y1="110" x2="205" y2="95" stroke="#a78bfa"/><line x1="205" y1="95" x2="285" y2="140" stroke="#a78bfa"/>
  <line x1="120" y1="110" x2="190" y2="165" stroke="#a78bfa"/><line x1="190" y1="165" x2="285" y2="140" stroke="#a78bfa"/>
  <rect x="100" y="200" width="220" height="34" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="210" y="222" text-anchor="middle" font-size="10.5" fill="#166534">community summary answers global Qs</text>
  <line x1="460" y1="50" x2="460" y2="250" stroke="#cbd5e1" stroke-dasharray="4 4"/>
  <text x="680" y="58" text-anchor="middle" font-size="12" fill="#d97706" font-weight="600">RAPTOR: summarize chunks into a tree</text>
  <rect x="640" y="78" width="80" height="26" rx="5" fill="#fde68a" stroke="#d97706"/><text x="680" y="96" text-anchor="middle" font-size="10" fill="#92400e">root</text>
  <rect x="590" y="135" width="80" height="26" rx="5" fill="#fef3c7" stroke="#d97706"/><text x="630" y="153" text-anchor="middle" font-size="10" fill="#92400e">summary</text>
  <rect x="700" y="135" width="80" height="26" rx="5" fill="#fef3c7" stroke="#d97706"/><text x="740" y="153" text-anchor="middle" font-size="10" fill="#92400e">summary</text>
  <rect x="560" y="195" width="58" height="24" rx="5" fill="#ffffff" stroke="#94a3b8"/><text x="589" y="211" text-anchor="middle" font-size="9" fill="#475569">chunk</text>
  <rect x="624" y="195" width="58" height="24" rx="5" fill="#ffffff" stroke="#94a3b8"/><text x="653" y="211" text-anchor="middle" font-size="9" fill="#475569">chunk</text>
  <rect x="700" y="195" width="58" height="24" rx="5" fill="#ffffff" stroke="#94a3b8"/><text x="729" y="211" text-anchor="middle" font-size="9" fill="#475569">chunk</text>
  <rect x="764" y="195" width="58" height="24" rx="5" fill="#ffffff" stroke="#94a3b8"/><text x="793" y="211" text-anchor="middle" font-size="9" fill="#475569">chunk</text>
  <line x1="680" y1="104" x2="630" y2="135" stroke="#d97706"/><line x1="680" y1="104" x2="740" y2="135" stroke="#d97706"/>
  <line x1="630" y1="161" x2="589" y2="195" stroke="#d97706"/><line x1="630" y1="161" x2="653" y2="195" stroke="#d97706"/>
  <line x1="740" y1="161" x2="729" y2="195" stroke="#d97706"/><line x1="740" y1="161" x2="793" y2="195" stroke="#d97706"/>
</svg>

> **Handy Heuristic**
>
> If your engineers ask "find me the runbook for X," you need good chunk
> retrieval — hybrid + rerank. If they ask "what keeps breaking / what are the
> recurring causes across everything," no chunk retriever saves you; you need
> GraphRAG or RAPTOR-style structure built at index time. Most real systems get
> both kinds of question. Most teams build only for the first and are mystified
> by the second.

---

## "Is RAG dead?" — the long-context question, answered honestly

Every six months since context windows blew past a million tokens, somebody
declares RAG obsolete: *just paste the whole wiki in the prompt and let the model
sort it out.* It's a fair question and it deserves a straight answer, not tribal
defensiveness.

<svg viewBox="0 0 900 230" width="100%" role="img" aria-label="RAG vs long context" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="230" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">They're not rivals. They compose.</text>
  <rect x="40" y="50" width="260" height="160" rx="8" fill="#fff7ed" stroke="#d97706"/>
  <text x="170" y="74" text-anchor="middle" font-size="12" fill="#92400e" font-weight="700">stuff everything in context</text>
  <text x="58" y="100" font-size="10.5" fill="#16a34a">+ no retrieval to get wrong</text>
  <text x="58" y="120" font-size="10.5" fill="#dc2626">– $$ per token, every single call</text>
  <text x="58" y="140" font-size="10.5" fill="#dc2626">– slower, higher latency</text>
  <text x="58" y="160" font-size="10.5" fill="#dc2626">– "lost in the middle" accuracy dip</text>
  <text x="58" y="180" font-size="10.5" fill="#dc2626">– corpus &gt; window? doesn't fit at all</text>
  <text x="58" y="200" font-size="10.5" fill="#dc2626">– no citations / audit trail</text>
  <rect x="320" y="50" width="260" height="160" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="450" y="74" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="700">RAG (retrieve first)</text>
  <text x="338" y="100" font-size="10.5" fill="#16a34a">+ cheap: model reads ~4 passages</text>
  <text x="338" y="120" font-size="10.5" fill="#16a34a">+ scales past any window size</text>
  <text x="338" y="140" font-size="10.5" fill="#16a34a">+ citable, auditable</text>
  <text x="338" y="160" font-size="10.5" fill="#16a34a">+ update corpus, not the model</text>
  <text x="338" y="180" font-size="10.5" fill="#dc2626">– retrieval can miss</text>
  <text x="338" y="200" font-size="10.5" fill="#dc2626">– more moving parts</text>
  <rect x="600" y="50" width="270" height="160" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="735" y="74" text-anchor="middle" font-size="12" fill="#166534" font-weight="700">the 2026 answer: both</text>
  <text x="618" y="100" font-size="10.5" fill="#166534">retrieve to NARROW 10M docs</text>
  <text x="618" y="120" font-size="10.5" fill="#166534">to the relevant few hundred,</text>
  <text x="618" y="140" font-size="10.5" fill="#166534">then use a long window to reason</text>
  <text x="618" y="160" font-size="10.5" fill="#166534">over ALL of them at once.</text>
  <text x="618" y="186" font-size="10.5" fill="#166534" font-weight="600">retrieval = scope control;</text>
  <text x="618" y="204" font-size="10.5" fill="#166534" font-weight="600">long context = reasoning room.</text>
</svg>

The honest answer in 2026: **long context didn't kill RAG, it changed RAG's
job.** A million-token window is wonderful, but Tanvi's wiki plus years of Slack
plus every postmortem is bigger than that, costs real money to re-read on every
question, gets *less* accurate as the one relevant runbook hides in the noise,
and gives answers with no link to the doc they came from. Retrieval still earns
its keep — it just gets to be looser. Instead of sweating to nail the perfect
top-4, retrieve the top few-hundred and let a long window sort them out.
Retrieval controls *scope* and *cost*; the window provides *reasoning room*. They
compose. Anyone selling you one as the death of the other is selling something.

---

## The 2026 stack, in one picture

Put it together and here's what a state-of-the-art RAG pipeline looks like now.
Tanvi's bot, fully grown up:

<svg viewBox="0 0 900 360" width="100%" role="img" aria-label="The modern RAG stack" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar6" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="360" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">The modern RAG stack (mid-2026)</text>
  <rect x="30" y="50" width="300" height="290" rx="8" fill="#fef9c3" stroke="#ca8a04"/>
  <text x="180" y="72" text-anchor="middle" font-size="12" fill="#854d0e" font-weight="700">INDEX TIME (offline, once)</text>
  <rect x="55" y="86" width="250" height="34" rx="6" fill="#fff" stroke="#ca8a04"/><text x="180" y="108" text-anchor="middle" font-size="10.5" fill="#854d0e">semantic / late chunking</text>
  <rect x="55" y="128" width="250" height="34" rx="6" fill="#fff" stroke="#ca8a04"/><text x="180" y="150" text-anchor="middle" font-size="10.5" fill="#854d0e">contextual blurb per chunk (cached LLM)</text>
  <rect x="55" y="170" width="250" height="34" rx="6" fill="#fff" stroke="#ca8a04"/><text x="180" y="192" text-anchor="middle" font-size="10.5" fill="#854d0e">embed (dense) + index (BM25/sparse)</text>
  <rect x="55" y="212" width="250" height="34" rx="6" fill="#fff" stroke="#ca8a04"/><text x="180" y="234" text-anchor="middle" font-size="10.5" fill="#854d0e">+ metadata: source, date, deprecated?</text>
  <rect x="55" y="254" width="250" height="34" rx="6" fill="#fff" stroke="#ca8a04"/><text x="180" y="276" text-anchor="middle" font-size="10.5" fill="#854d0e">optional: GraphRAG / RAPTOR structure</text>
  <text x="180" y="312" text-anchor="middle" font-size="10" fill="#854d0e">do the expensive work once, not per query</text>
  <line x1="330" y1="195" x2="375" y2="195" stroke="#475569" stroke-width="2" marker-end="url(#ar6)"/>
  <rect x="380" y="50" width="490" height="290" rx="8" fill="#eef2ff" stroke="#4f46e5"/>
  <text x="625" y="72" text-anchor="middle" font-size="12" fill="#3730a3" font-weight="700">QUERY TIME (online, per question)</text>
  <rect x="405" y="86" width="440" height="32" rx="6" fill="#fff" stroke="#4f46e5"/><text x="625" y="107" text-anchor="middle" font-size="10.5" fill="#3730a3">① query transform (HyDE / decompose / step-back)</text>
  <rect x="405" y="126" width="440" height="32" rx="6" fill="#fff" stroke="#4f46e5"/><text x="625" y="147" text-anchor="middle" font-size="10.5" fill="#3730a3">② hybrid retrieve: dense + BM25 → fuse (RRF)</text>
  <rect x="405" y="166" width="440" height="32" rx="6" fill="#fff" stroke="#4f46e5"/><text x="625" y="187" text-anchor="middle" font-size="10.5" fill="#3730a3">③ rerank top-50 with a cross-encoder → top-k</text>
  <rect x="405" y="206" width="440" height="32" rx="6" fill="#fff" stroke="#4f46e5"/><text x="625" y="227" text-anchor="middle" font-size="10.5" fill="#3730a3">④ grade relevance (CRAG) — junk? loop back to ①</text>
  <rect x="405" y="246" width="440" height="32" rx="6" fill="#fff" stroke="#4f46e5"/><text x="625" y="267" text-anchor="middle" font-size="10.5" fill="#3730a3">⑤ generate: grounded, "use ONLY context", cite IDs</text>
  <rect x="405" y="286" width="440" height="32" rx="6" fill="#dcfce7" stroke="#16a34a"/><text x="625" y="307" text-anchor="middle" font-size="10.5" fill="#166534">⑥ evaluate (next part: RAGAS) — close the loop</text>
</svg>

Notice the shape. The clever, expensive work — contextualizing, building
structure — happens **once, offline.** The query-time path is the same
hybrid-retrieve → rerank → generate spine from Part 1, with a query transform
bolted on the front and a relevance grader on the back. You do *not* need all of
this on day one. You need the spine. You add the rest where your errors tell you
to, and not one step sooner.

> **Handy Heuristic**
>
> Build in this order, and stop the moment your eval (Part 10) says you're good
> enough: (1) hybrid retrieval, (2) reranking, (3) better chunking + contextual
> retrieval, (4) query transforms, (5) agentic/corrective loops, (6) GraphRAG —
> *only* if you genuinely have global questions. Most teams over-build the late
> stages and starve the early ones. The early ones are where the wins live.

---

## The stale-runbook problem (where this gets genuinely dangerous)

One more failure, and it's the one that actually scares me, because it's the one
where good retrieval hands you a confident *wrong* answer instead of no answer.

Tanvi's corpus has two docs about recovering a stuck `payments-api` pod. One is
an old runbook (`rb_010`): *SSH into the node and `docker restart` the
container.* The other is a postmortem from a few months ago (`rb_011`): *NEVER do
that — a hard restart killed an in-flight write, corrupted the write-ahead log,
and turned a 2-minute blip into a 40-minute outage; drain the pod first.* The
postmortem supersedes the runbook. One of them is now actively dangerous advice.

Watch what the hybrid retriever actually does with "a payments-api pod keeps
restarting on its own — what should I check?":

```bash
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.qa
```

```
Retrieved: rb_004, rb_010, rb_003, rb_013
```

Read that top-4 carefully. `rb_010` — the dangerous, deprecated *"just docker
restart it"* runbook — made the cut. `rb_011`, the postmortem that exists
*specifically to stop you doing that*, didn't even crack the top four. Both docs
are about the same topic, they read almost identically to an embedding model, and
nothing in pure vector-or-keyword similarity knows which one is current. This is
not hypothetical; it's the real output of the example you can run right now.

The fix isn't a better embedding. It's **metadata.** Tag every chunk with its
source and date, mark superseded docs as deprecated, and apply a recency boost —
or a hard filter — so the postmortem outranks the runbook it killed.

<svg viewBox="0 0 900 250" width="100%" role="img" aria-label="Metadata and recency rescue" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p2g" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="250" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Vectors alone can't tell current from deprecated — metadata can</text>
  <text x="160" y="58" text-anchor="middle" font-size="12" fill="#991b1b" font-weight="600">pure similarity ranking</text>
  <rect x="40" y="70" width="240" height="44" rx="6" fill="#fee2e2" stroke="#dc2626"/>
  <text x="160" y="89" text-anchor="middle" font-size="10.5" fill="#991b1b" font-weight="600">1. rb_010  "docker restart it"</text>
  <text x="160" y="105" text-anchor="middle" font-size="9.5" fill="#991b1b">DEPRECATED 2024 — dangerous ✗</text>
  <rect x="40" y="120" width="240" height="44" rx="6" fill="#fff" stroke="#cbd5e1"/>
  <text x="160" y="139" text-anchor="middle" font-size="10.5" fill="#475569">2. rb_011  "NEVER restart; drain"</text>
  <text x="160" y="155" text-anchor="middle" font-size="9.5" fill="#475569">postmortem 2026 — correct, buried</text>
  <line x1="290" y1="120" x2="350" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#p2g)"/>
  <rect x="355" y="92" width="170" height="56" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="440" y="116" text-anchor="middle" font-size="10.5" fill="#5b21b6" font-weight="600">recency boost +</text>
  <text x="440" y="132" text-anchor="middle" font-size="10.5" fill="#5b21b6">deprecated filter</text>
  <line x1="525" y1="120" x2="585" y2="120" stroke="#475569" stroke-width="2" marker-end="url(#p2g)"/>
  <text x="730" y="58" text-anchor="middle" font-size="12" fill="#166534" font-weight="600">metadata-aware ranking</text>
  <rect x="600" y="70" width="260" height="44" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="730" y="89" text-anchor="middle" font-size="10.5" fill="#166534" font-weight="600">1. rb_011  "NEVER restart; drain" ✓</text>
  <text x="730" y="105" text-anchor="middle" font-size="9.5" fill="#166534">newest wins</text>
  <rect x="600" y="120" width="260" height="44" rx="6" fill="#fff" stroke="#cbd5e1"/>
  <text x="730" y="139" text-anchor="middle" font-size="10.5" fill="#94a3b8">rb_010 — filtered out / demoted</text>
  <text x="730" y="155" text-anchor="middle" font-size="9.5" fill="#94a3b8">deprecated, never served</text>
  <text x="450" y="198" text-anchor="middle" font-size="11" fill="#475569">Retrieval quality isn't only about the vectors — it's about the metadata you index alongside them.</text>
  <text x="450" y="222" text-anchor="middle" font-size="11" fill="#0f172a" font-weight="600">Good search uses every signal you've got, not just the trendy one.</text>
</svg>

Pure similarity search for an evolving knowledge base isn't *wrong*, it's
*incomplete* — and "incomplete" during an incident is how a blip becomes an
outage. Which is the whole thesis again, said one more way: **getting the right
doc is a search problem,** and good search uses every signal you've got.

---

## The answer template (retrieval is necessary, not sufficient)

Retrieval hands you the right runbooks. You still have to make the model answer
*from them and only them.* The prompt earns its keep:

```
You are an on-call assistant answering from the team's runbooks,
postmortems, and architecture docs.

Use ONLY the context passages below. If the context does not contain
sufficient information, say so plainly — do not guess. A wrong fix during
an incident makes the incident worse.

When two passages conflict, prefer the more recent one (a postmortem that
supersedes an old runbook) and say which you used and why.

Always cite the runbook IDs you used: [Sources: rb_002, rb_011]

Context:
{context}

Question: {question}
```

Four instructions in there that everyone regrets leaving out:

- **"Use ONLY the context."** Without it, the model falls back on training
  knowledge when the retrieved context is thin. Sometimes that's right. Sometimes
  it's a generic Stack Overflow answer that doesn't match your infra. You can't
  tell which, and neither can the engineer running the command.
- **"Say so plainly if insufficient."** Without it, the model papers over gaps
  with confident-sounding fiction. During an incident, a confident wrong fix is
  strictly worse than "I couldn't find a runbook for this."
- **"Prefer the more recent passage."** This is the stale-runbook fix at the
  generation layer — a backstop for when retrieval hands the model both the old
  runbook and the postmortem that killed it.
- **"Cite the sources."** When an answer cites `rb_011` and you can open it, you
  catch a bad answer in seconds — instead of finding out by running it on
  production.

---

## Run it

```bash
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.qa
```

Four on-call questions over a fifteen-passage fixture corpus. BM25-leaning hybrid
at `alpha=0.4`. Answers with source citations.

```
Q: What does ERR_5521 mean?
Retrieved: rb_002, rb_009, rb_011, rb_005
A: ERR_5521 is raised by payments-api when the upstream payment gateway
   doesn't respond within the configured timeout. It's an upstream timeout,
   not a bug in our code — check the gateway's status page first, then
   whether it's hitting all merchants or just one. It usually clears once
   the gateway recovers; if it persists, escalate to the Payments team.
   [Sources: rb_002]
```

Grounded, cited, correct — and `rb_002` got there because BM25 matched the exact
token `ERR_5521` that pure vectors threw on the floor. That one retrieval is the
whole two-part article in miniature.

---

Tanvi's original bot was fine for natural language. It fell over on the
vocabulary her engineers actually type at 3 a.m. — error codes, kubectl states,
service names, the deprecated runbook that should never have surfaced. None of
that is a model problem. All of it is a *search* problem. And the road from
"broken vector search" to "trustworthy on-call answer" was nothing but a stack of
honest, individually-simple search fixes:

> BM25 handles the exact tokens. Dense vectors handle the paraphrases. Fusion
> settles their fights. The reranker sharpens what survives. Contextual chunking
> stops chunks from getting orphaned. Query transforms fix the question itself.
> Corrective loops notice when the fetch was garbage. Metadata stops you running
> the dangerous old runbook. And structure — GraphRAG, RAPTOR — answers the
> questions no single chunk ever could.

None of it is magic. All of it is search. **RAG is a search problem** — and once
you actually believe that, you stop blaming the model and start fixing the thing
that's broken.

But notice we've been calling answers "good," "grounded," "correct" by reading
them and nodding. *"It looks right"* is not a metric. Next, we measure.

---

*Back to Part 1 → `article_part1.md`*

*Part 10: Measuring the Truth — RAGAS. Faithfulness, answer relevance, context
precision, context recall. Because "it looks right" is not a retrieval quality
metric.*

*Code: [github.com/saurabhtg/Applied-AI](https://github.com/saurabhtg/Applied-AI)*

*All thirty parts: [Applied AI Series](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/)*
