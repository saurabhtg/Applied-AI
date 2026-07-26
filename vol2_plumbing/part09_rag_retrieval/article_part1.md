# RAG is a Search Problem — Part 1: The Search Engine Hiding Under Your LLM

*Applied AI, Volume II — Part 9 (1 of 2)*

---

> **TL;DR — Part 1, the foundations**
>
> It's 3 a.m., something's on fire, and your shiny RAG bot just handed the
> on-call engineer the *wrong* runbook. The model is fine. The **search** is
> broken. RAG is a search engine with an LLM on top — not an LLM with a little
> search bolted on. Get that backwards and nothing else helps.
>
> - **Vector search finds meaning** ("site is slow" ≈ "p99 latency") but goes
>   *stone blind* on exact strings it never learned: `ERR_5521`, `OOMKilled`,
>   `payments-api`.
> - **Keyword search (BM25) finds exact words** but whiffs every paraphrase.
> - They have **mirror-image blind spots** → **run both** (hybrid retrieval).
> - **Mix the two result lists.** Blend scores with an `alpha` knob, *or* —
>   safer — use **RRF**: go by *rank*, not raw score, so one exact-match spike
>   can't wreck the blend.
> - **Then sharpen:** a **reranker** (cross-encoder) rereads the question and
>   each passage *together* and keeps the best few. Search *finds*; the reranker
>   *chooses*.
>
> That's the load-bearing spine: **hybrid retrieve → rerank → generate.** Build
> only this and you're ahead of most RAG in production. **Part 2** covers
> everything that makes it *trustworthy* — chunking, Contextual Retrieval, late
> interaction, query rewriting, agentic/CRAG, GraphRAG, and the stale-runbook
> trap.

---

## The 3 a.m. story

Tanvi builds an on-call assistant. The pitch writes itself: the team has years
of runbooks, postmortems, and architecture docs rotting in a wiki nobody can
search, and every Sev-1 starts with some bleary engineer at 3 a.m. grepping
Confluence with one eye open. So she wires up a bot. Ask it a question, it digs
up the right runbook, an LLM turns that into a straight answer. Good idea.
Genuinely useful. The kind of thing that gets a demo slot at the all-hands.

She sets up a vector database, embeds every doc, ships it. Her manager — the one
with the demo slot — tries the first question: *"How do I roll back a bad
deployment?"*

It works. Pulls up the deploy runbook, answers cleanly. Tanvi exhales.

The next morning an actual on-call engineer, mid-incident, types: *"What does
`ERR_5521` mean?"*

The bot returns a passage about "handling upstream errors gracefully" and
another on "best practices for error logging." Neither one mentions `ERR_5521`.
The runbook that defines it — *payments-api, upstream gateway timeout, check the
gateway status page first* — is right there in the corpus. The vector search
just couldn't find it, because `ERR_5521` has no semantic neighbours in
embedding space. It's a string. It means nothing to the model.

Then: *"`OOMKilled` on checkout-worker."* The bot helpfully surfaces a doc
titled "Diagnosing Service Crashes," in general, philosophically. The one
runbook that says *the kernel killed your pod for blowing its memory limit,
here's the kubectl command* — it never shows up.

This is the failure nobody warns you about, and here's the part that took Tanvi
a week and one bad incident to swallow: **it is not a vector database problem.**
The vector database is doing precisely what it was built to do, flawlessly, at
scale. It's a *search* problem. And the most expensive vector DB on the market
will index a broken search strategy beautifully, with five-nines uptime, and
hand your on-call engineer garbage at 3 a.m. all the same.

This two-part article is the whole arc of fixing it — from the dumbest baseline
that every tutorial hands you, up to what good teams actually ship in 2026. Part
1 (this one) builds the foundation: how retrieval actually works, why the naive
version breaks, and the two-stage spine that fixes most of it.

---

## First, what RAG even is (and why you can't skip it)

Cut through the acronym and Retrieval-Augmented Generation is one sentence:

> **Before you ask the model a question, go find the relevant facts and paste
> them into the prompt.**

That's the whole trick. The "retrieval" is the finding. The "augmented" is the
pasting. The "generation" is the model writing an answer out of what you pasted.

<svg viewBox="0 0 900 230" width="100%" role="img" aria-label="RAG pipeline overview" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar1" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="230" fill="#f8fafc" rx="10"/>
  <rect x="20" y="90" width="130" height="50" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="85" y="112" text-anchor="middle" font-size="13" fill="#1e3a8a" font-weight="600">Engineer asks</text>
  <text x="85" y="130" text-anchor="middle" font-size="11" fill="#1e3a8a">"what is ERR_5521?"</text>
  <line x1="150" y1="115" x2="195" y2="115" stroke="#475569" stroke-width="2" marker-end="url(#ar1)"/>
  <rect x="200" y="70" width="150" height="90" rx="8" fill="#fef3c7" stroke="#d97706"/>
  <text x="275" y="100" text-anchor="middle" font-size="13" fill="#92400e" font-weight="600">RETRIEVER</text>
  <text x="275" y="120" text-anchor="middle" font-size="11" fill="#92400e">searches the corpus,</text>
  <text x="275" y="135" text-anchor="middle" font-size="11" fill="#92400e">returns top passages</text>
  <rect x="200" y="175" width="150" height="40" rx="6" fill="#fff7ed" stroke="#d97706" stroke-dasharray="4 3"/>
  <text x="275" y="199" text-anchor="middle" font-size="11" fill="#92400e">📚 runbooks + postmortems</text>
  <line x1="275" y1="160" x2="275" y2="173" stroke="#d97706" stroke-width="1.5"/>
  <line x1="350" y1="115" x2="395" y2="115" stroke="#475569" stroke-width="2" marker-end="url(#ar1)"/>
  <rect x="400" y="80" width="140" height="70" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="470" y="108" text-anchor="middle" font-size="12" fill="#166534" font-weight="600">Top 4 passages</text>
  <text x="470" y="126" text-anchor="middle" font-size="11" fill="#166534">the relevant facts</text>
  <line x1="540" y1="115" x2="585" y2="115" stroke="#475569" stroke-width="2" marker-end="url(#ar1)"/>
  <rect x="590" y="70" width="150" height="90" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="665" y="100" text-anchor="middle" font-size="13" fill="#5b21b6" font-weight="600">LLM</text>
  <text x="665" y="120" text-anchor="middle" font-size="11" fill="#5b21b6">writes answer using</text>
  <text x="665" y="135" text-anchor="middle" font-size="11" fill="#5b21b6">ONLY those passages</text>
  <line x1="740" y1="115" x2="785" y2="115" stroke="#475569" stroke-width="2" marker-end="url(#ar1)"/>
  <rect x="790" y="90" width="100" height="50" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="840" y="112" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">Grounded</text>
  <text x="840" y="129" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">+ cited answer</text>
  <text x="450" y="30" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">The RAG pipeline: find the facts, then answer from them</text>
</svg>

Why bother? Because the three obvious alternatives are all worse for Tanvi:

- **Fine-tune the model on every runbook.** Expensive, slow, and the day someone
  updates a runbook your model is confidently quoting the old one with no way to
  cite where it got that. Fine-tuning teaches the model *how to behave*. It is a
  miserable way to teach it *facts that change every sprint*.
- **Paste the entire wiki into the prompt.** Even with a giant context window —
  and we'll get to why that's not the slam dunk people think — you're paying to
  re-read the whole library on every question, and the model gets *less* accurate
  as the one relevant runbook drowns under fifty irrelevant ones. That's the
  "lost in the middle" effect, and it is real.
- **Let the model wing it from training data.** Congratulations, you've built a
  machine that invents a plausible, fluent, *wrong* fix and recommends it during
  an outage. There is no faster way to turn a two-minute blip into a postmortem.

RAG threads the needle. The facts live in a corpus you own and update whenever
you like, and the model only ever sees the handful of passages that matter for
*this* question. Cheap to run, always current, and every answer traces back to a
doc you can open.

> **Software Dogma**
>
> *RAG is not a model feature. It's a search system with an LLM stapled to the
> end.* Ninety percent of your answer quality comes from the search half — the
> part that has nothing to do with the LLM. Teams burn weeks tuning the prompt
> and can't work out why the answers are still wrong. The answers are wrong
> because the search handed the model the wrong runbook.

---

## The baseline everybody ships first (and why it cracks)

Here's the naive RAG pipeline. I'd bet a week's pay it's the one you'd build
this afternoon if I handed you the wiki, because it's the one every tutorial,
every framework quickstart, every Medium post shows:

1. Chop the docs into chunks — say, 500 tokens each.
2. Run each chunk through an embedding model. One vector per chunk.
3. Drop the vectors in a vector database.
4. At query time, embed the question, grab the nearest chunk vectors by cosine
   similarity, take the top few.
5. Paste those chunks into the prompt. Generate.

This is **dense retrieval**, and to use it well you need to know what step 2
actually does. An **embedding model** is a neural network that reads a piece of
text and squeezes it down to a single list of a few hundred to a few thousand
numbers — a *vector*. The magic is what those numbers encode: the model is
trained so that texts which *mean* similar things get similar number-lists.

<svg viewBox="0 0 900 210" width="100%" role="img" aria-label="How text becomes a vector" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p1a" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker></defs>
  <rect x="0" y="0" width="900" height="210" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">How an embedding model turns text into a point in space</text>
  <rect x="20" y="80" width="150" height="60" rx="8" fill="#dbeafe" stroke="#2563eb"/>
  <text x="95" y="105" text-anchor="middle" font-size="11" fill="#1e3a8a">"OOMKilled on</text>
  <text x="95" y="123" text-anchor="middle" font-size="11" fill="#1e3a8a">checkout-worker"</text>
  <line x1="170" y1="110" x2="205" y2="110" stroke="#475569" stroke-width="2" marker-end="url(#p1a)"/>
  <rect x="210" y="80" width="195" height="60" rx="8" fill="#eef2ff" stroke="#6366f1"/>
  <text x="307" y="103" text-anchor="middle" font-size="11" fill="#3730a3" font-weight="600">tokenize</text>
  <text x="307" y="123" text-anchor="middle" font-size="9.5" fill="#3730a3">[OOM][Killed][on][check]…</text>
  <line x1="405" y1="110" x2="440" y2="110" stroke="#475569" stroke-width="2" marker-end="url(#p1a)"/>
  <rect x="445" y="80" width="160" height="60" rx="8" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="525" y="105" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">embedding model</text>
  <text x="525" y="123" text-anchor="middle" font-size="10" fill="#5b21b6">(a transformer)</text>
  <line x1="605" y1="110" x2="640" y2="110" stroke="#475569" stroke-width="2" marker-end="url(#p1a)"/>
  <rect x="645" y="80" width="235" height="60" rx="8" fill="#dcfce7" stroke="#16a34a"/>
  <text x="762" y="104" text-anchor="middle" font-size="10.5" fill="#166534" font-weight="600">[0.12, -0.85, 0.33, … ]</text>
  <text x="762" y="123" text-anchor="middle" font-size="10" fill="#166534">one vector ≈ hundreds of numbers</text>
  <text x="450" y="178" text-anchor="middle" font-size="11" fill="#475569">Similar meanings get similar number-lists — so "close in meaning" becomes "close in space."</text>
</svg>

Do that for every chunk in the corpus *once*, up front, and store the vectors.
At query time you embed the question the same way and ask: which stored vectors
are *closest* to the question's vector? "Closest" is measured by **cosine
similarity** — the cosine of the angle between two vectors. Point in the same
direction, cosine is `1.0` (identical meaning). At right angles, `0.0`
(unrelated). It's a single number between 0 and 1 that says "how aligned are
these two meanings?"

<svg viewBox="0 0 900 280" width="100%" role="img" aria-label="Cosine similarity" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p1b" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="#334155"/></marker></defs>
  <rect x="0" y="0" width="900" height="280" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Cosine similarity: rank by the angle between meanings</text>
  <line x1="140" y1="250" x2="500" y2="250" stroke="#cbd5e1"/>
  <line x1="140" y1="250" x2="140" y2="60" stroke="#cbd5e1"/>
  <line x1="140" y1="250" x2="380" y2="80" stroke="#2563eb" stroke-width="2.5" marker-end="url(#p1b)"/>
  <text x="388" y="78" font-size="11" fill="#1e3a8a" font-weight="600">query: "site is slow"</text>
  <line x1="140" y1="250" x2="400" y2="120" stroke="#16a34a" stroke-width="2.5" marker-end="url(#p1b)"/>
  <text x="408" y="122" font-size="11" fill="#166534">doc: "p99 latency high"  → cos ≈ 0.97 ✓</text>
  <line x1="140" y1="250" x2="430" y2="232" stroke="#ef4444" stroke-width="2.5" marker-end="url(#p1b)"/>
  <text x="438" y="234" font-size="11" fill="#991b1b">doc: "lunch menu"  → cos ≈ 0.15 ✗</text>
  <path d="M 200 224 A 64 64 0 0 1 214 210" fill="none" stroke="#475569"/>
  <text x="210" y="205" font-size="10" fill="#475569">small angle</text>
  <rect x="560" y="70" width="320" height="160" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="720" y="94" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">read it like this</text>
  <text x="578" y="122" font-size="11" fill="#475569">cos 0°  = 1.00  → same direction (same meaning)</text>
  <text x="578" y="146" font-size="11" fill="#475569">cos 90° = 0.00  → unrelated</text>
  <text x="578" y="170" font-size="11" fill="#475569">small angle = high score = good match</text>
  <text x="578" y="200" font-size="11" fill="#166534" font-weight="600">retrieval = "give me the chunks whose</text>
  <text x="578" y="216" font-size="11" fill="#166534" font-weight="600">vectors point most like the question's."</text>
</svg>

When it works it feels like magic. *"How do I roll back a bad deploy?"* and
*"reverting a broken release"* end up neighbours even though they barely share a
word — because the model learned they *mean* the same thing.

<svg viewBox="0 0 900 300" width="100%" role="img" aria-label="Embedding space neighbours" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="300" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Embedding space: meaning becomes geometry</text>
  <rect x="40" y="50" width="380" height="230" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="230" y="72" text-anchor="middle" font-size="12" fill="#16a34a" font-weight="600">✓ where dense retrieval shines</text>
  <circle cx="140" cy="130" r="6" fill="#2563eb"/>
  <text x="152" y="134" font-size="11" fill="#1e3a8a">"roll back a bad deploy"</text>
  <circle cx="175" cy="160" r="6" fill="#2563eb"/>
  <text x="187" y="164" font-size="11" fill="#1e3a8a">"revert a broken release"</text>
  <circle cx="150" cy="190" r="6" fill="#2563eb"/>
  <text x="162" y="194" font-size="11" fill="#1e3a8a">"undo the last deployment"</text>
  <ellipse cx="165" cy="160" rx="80" ry="55" fill="none" stroke="#16a34a" stroke-dasharray="4 3"/>
  <circle cx="340" cy="240" r="6" fill="#94a3b8"/>
  <text x="250" y="244" font-size="11" fill="#64748b">"lunch menu" (far away)</text>
  <rect x="480" y="50" width="380" height="230" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="670" y="72" text-anchor="middle" font-size="12" fill="#dc2626" font-weight="600">✗ where it falls apart</text>
  <circle cx="600" cy="135" r="6" fill="#ef4444"/>
  <text x="612" y="139" font-size="11" fill="#991b1b">"ERR_5521" — floats alone, no neighbours</text>
  <circle cx="640" cy="180" r="6" fill="#ef4444"/>
  <text x="525" y="184" font-size="11" fill="#991b1b">"OOMKilled"  "CrashLoopBackOff"</text>
  <text x="525" y="225" font-size="11" fill="#64748b">The model never learned these tokens.</text>
  <text x="525" y="242" font-size="11" fill="#64748b">In embedding space they're just noise —</text>
  <text x="525" y="259" font-size="11" fill="#64748b">nothing meaningful sits nearby.</text>
</svg>

So why did Tanvi's bot faceplant on `ERR_5521` and `OOMKilled`? Because dense
retrieval works on *learned meaning*, and the embedding model never learned what
those strings mean. They barely appear in its training data. `ERR_5521`,
`CrashLoopBackOff`, `payments-api`, `MAX_POOL_SIZE` — to the model these are
near-random tokens with no good neighbours. It shrugs and returns whatever was
vaguely nearby and vaguely about "errors."

That's **failure mode #1: dense retrieval is blind to exact tokens it never
learned to embed.** Error codes, Kubernetes states, service names, env vars,
flag names, SKUs, ticket IDs — the whole family of strings that mean something
precise to an engineer and nothing to a general embedding model. Which, if you
look at how your team actually talks during an incident, is *most of the words
that matter.*

---

## The dumb-but-honest baseline: keyword search

Long before embeddings, search ran on **BM25** — the grown-up version of
TF-IDF that your grandfather's search box used. Two simple instincts power it,
and both are worth understanding because they explain exactly why it succeeds
where vectors fail:

1. **Rare words carry the signal (inverse document frequency).** If a word shows
   up in every doc — "the," "error," "service" — matching it tells you almost
   nothing. If a word is rare — `ERR_5521`, `OOMKilled` — matching it tells you
   almost everything. BM25 weights rare words way up and common words way down.
2. **Repetition has diminishing returns (saturation).** A doc that says
   `ERR_5521` once is relevant. A doc that says it forty times isn't forty times
   more relevant. BM25 lets the score rise with term frequency, then flattens.

<svg viewBox="0 0 900 290" width="100%" role="img" aria-label="BM25 intuition" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="290" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Why BM25 nails the exact token vectors miss</text>
  <rect x="30" y="48" width="410" height="222" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="235" y="70" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">① rare words shout, common words whisper</text>
  <line x1="70" y1="240" x2="410" y2="240" stroke="#94a3b8"/>
  <line x1="70" y1="90" x2="70" y2="240" stroke="#94a3b8"/>
  <rect x="95" y="228" width="44" height="12" fill="#cbd5e1"/><text x="117" y="256" text-anchor="middle" font-size="9.5" fill="#475569">"the"</text>
  <rect x="165" y="222" width="44" height="18" fill="#cbd5e1"/><text x="187" y="256" text-anchor="middle" font-size="9.5" fill="#475569">"error"</text>
  <rect x="245" y="120" width="44" height="120" fill="#d97706"/><text x="267" y="256" text-anchor="middle" font-size="9.5" fill="#92400e">ERR_5521</text>
  <rect x="320" y="135" width="44" height="105" fill="#d97706"/><text x="342" y="256" text-anchor="middle" font-size="9.5" fill="#92400e">OOMKilled</text>
  <text x="78" y="100" font-size="9.5" fill="#475569">weight (IDF)</text>
  <rect x="460" y="48" width="410" height="222" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="665" y="70" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">② repeating a word saturates</text>
  <line x1="500" y1="240" x2="840" y2="240" stroke="#94a3b8"/>
  <line x1="500" y1="90" x2="500" y2="240" stroke="#94a3b8"/>
  <path d="M 500 240 Q 560 110 700 100 T 840 95" fill="none" stroke="#16a34a" stroke-width="2.5"/>
  <text x="700" y="135" font-size="10" fill="#166534">score rises, then flattens</text>
  <text x="670" y="258" text-anchor="middle" font-size="9.5" fill="#475569">how many times the word appears →</text>
  <text x="508" y="100" font-size="9.5" fill="#475569">score</text>
</svg>

BM25 has the exact opposite personality from dense retrieval. It hasn't the
faintest idea what words *mean*. `ERR_5521` in the query matches `ERR_5521` in
the passage — literal, exact, done. It does not know "documents" and
"documentation" are cousins, and it certainly does not know `OOMKilled` is about
memory. Watch the two retrievers take the same two queries:

<svg viewBox="0 0 900 250" width="100%" role="img" aria-label="BM25 vs dense failure table" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="250" fill="#f8fafc" rx="10"/>
  <text x="450" y="30" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">They fail on opposite queries</text>
  <rect x="40" y="50" width="300" height="40" fill="#1e293b" rx="6"/>
  <text x="190" y="75" text-anchor="middle" font-size="12" fill="#fff" font-weight="600">the query</text>
  <rect x="350" y="50" width="250" height="40" fill="#d97706" rx="6"/>
  <text x="475" y="75" text-anchor="middle" font-size="12" fill="#fff" font-weight="600">BM25 (lexical)</text>
  <rect x="610" y="50" width="250" height="40" fill="#2563eb" rx="6"/>
  <text x="735" y="75" text-anchor="middle" font-size="12" fill="#fff" font-weight="600">Dense (semantic)</text>
  <rect x="40" y="95" width="300" height="65" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="55" y="132" font-size="12" fill="#0f172a">"what is ERR_5521?"</text>
  <rect x="350" y="95" width="250" height="65" fill="#dcfce7" stroke="#16a34a"/>
  <text x="475" y="125" text-anchor="middle" font-size="11" fill="#166534">ERR_5521 → exact match ✓</text>
  <text x="475" y="143" text-anchor="middle" font-size="11" fill="#166534">nails it</text>
  <rect x="610" y="95" width="250" height="65" fill="#fee2e2" stroke="#dc2626"/>
  <text x="735" y="125" text-anchor="middle" font-size="11" fill="#991b1b">no neighbours for the token ✗</text>
  <text x="735" y="143" text-anchor="middle" font-size="11" fill="#991b1b">returns generic mush</text>
  <rect x="40" y="165" width="300" height="65" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="55" y="195" font-size="12" fill="#0f172a">"why is the site slow</text>
  <text x="55" y="212" font-size="12" fill="#0f172a">under load?"</text>
  <rect x="350" y="165" width="250" height="65" fill="#fee2e2" stroke="#dc2626"/>
  <text x="475" y="195" text-anchor="middle" font-size="11" fill="#991b1b">"slow" is everywhere,</text>
  <text x="475" y="213" text-anchor="middle" font-size="11" fill="#991b1b">no signal ✗</text>
  <rect x="610" y="165" width="250" height="65" fill="#dcfce7" stroke="#16a34a"/>
  <text x="735" y="195" text-anchor="middle" font-size="11" fill="#166534">embeds to "p99 latency</text>
  <text x="735" y="213" text-anchor="middle" font-size="11" fill="#166534">under traffic" → match ✓</text>
</svg>

Sit with that table a second. It isn't that one retriever is better. They have
**mirror-image blind spots.** BM25 owns the exact-token query and whiffs the
paraphrase. Dense owns the paraphrase and whiffs the exact token. Put them in
the same room and each one plugs exactly the hole the other leaves wide open.

That observation — embarrassingly simple, ferociously effective — is the whole
foundation of modern retrieval. And notice what it really says: getting the
right runbook is a *search* problem with two halves, and you were only running
one of them.

---

## Hybrid retrieval: run both, fuse the results

Run BM25 *and* dense on every query, then merge the two lists. The merge is the
only interesting part, and there are two ways to do it.

### Way one: weighted score blending

Each retriever hands you a score per doc. Normalize both onto a 0–1 scale, then
blend with a tunable knob, `alpha`:

```
combined_score = (1 - alpha) × BM25_score + alpha × dense_score
```

`alpha = 0.0` is pure BM25. `alpha = 1.0` is pure dense. `alpha = 0.5` splits
the vote. For an on-call corpus stuffed with error codes and kubectl states, you
lean toward BM25. Tanvi runs hers at `alpha = 0.4`. Here it is worked out on three
docs for the query "what is ERR_5521?":

<svg viewBox="0 0 900 270" width="100%" role="img" aria-label="Weighted fusion worked example" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="270" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Weighted blend, worked: (0.6 × BM25) + (0.4 × dense), alpha = 0.4</text>
  <rect x="60" y="50" width="160" height="34" fill="#1e293b"/><text x="140" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="600">doc</text>
  <rect x="220" y="50" width="180" height="34" fill="#d97706"/><text x="310" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="600">BM25 (norm)</text>
  <rect x="400" y="50" width="180" height="34" fill="#2563eb"/><text x="490" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="600">dense (norm)</text>
  <rect x="580" y="50" width="260" height="34" fill="#16a34a"/><text x="710" y="72" text-anchor="middle" font-size="11" fill="#fff" font-weight="600">blend</text>
  <rect x="60" y="86" width="160" height="44" fill="#dcfce7" stroke="#16a34a"/><text x="140" y="113" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">rb_002 (ERR_5521)</text>
  <rect x="220" y="86" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="310" y="113" text-anchor="middle" font-size="11" fill="#0f172a">1.00  (exact match)</text>
  <rect x="400" y="86" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="490" y="113" text-anchor="middle" font-size="11" fill="#0f172a">0.20</text>
  <rect x="580" y="86" width="260" height="44" fill="#dcfce7" stroke="#16a34a"/><text x="710" y="113" text-anchor="middle" font-size="11" fill="#166534" font-weight="700">0.6(1.00)+0.4(0.20) = 0.68 ← wins</text>
  <rect x="60" y="132" width="160" height="44" fill="#fff" stroke="#cbd5e1"/><text x="140" y="159" text-anchor="middle" font-size="11" fill="#0f172a">rb_009 (escalation)</text>
  <rect x="220" y="132" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="310" y="159" text-anchor="middle" font-size="11" fill="#0f172a">0.10</text>
  <rect x="400" y="132" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="490" y="159" text-anchor="middle" font-size="11" fill="#0f172a">0.55</text>
  <rect x="580" y="132" width="260" height="44" fill="#fff" stroke="#cbd5e1"/><text x="710" y="159" text-anchor="middle" font-size="11" fill="#0f172a">0.6(0.10)+0.4(0.55) = 0.28</text>
  <rect x="60" y="178" width="160" height="44" fill="#fff" stroke="#cbd5e1"/><text x="140" y="205" text-anchor="middle" font-size="11" fill="#0f172a">rb_015 (TLS cert)</text>
  <rect x="220" y="178" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="310" y="205" text-anchor="middle" font-size="11" fill="#0f172a">0.00</text>
  <rect x="400" y="178" width="180" height="44" fill="#fff" stroke="#cbd5e1"/><text x="490" y="205" text-anchor="middle" font-size="11" fill="#0f172a">0.70</text>
  <rect x="580" y="178" width="260" height="44" fill="#fff" stroke="#cbd5e1"/><text x="710" y="205" text-anchor="middle" font-size="11" fill="#0f172a">0.6(0.00)+0.4(0.70) = 0.28</text>
  <text x="450" y="248" text-anchor="middle" font-size="11" fill="#475569">BM25 matches ERR_5521 exactly; dense drags in gateway-timeout passages by meaning. The blend lets the right doc win.</text>
</svg>

```python
retriever = HybridRetriever(
    documents=documents,
    embed_fn=your_embed_fn,
    alpha=0.4,        # lean BM25 — exact tokens carry this corpus
)
results = retriever.retrieve("What does ERR_5521 mean?", k=4)
```

That's the punchline from the opening: Tanvi never needed a fancier database. She
needed a second retriever running next to the first.

### The normalization step (do not skip this)

You can't add a BM25 score of `12.4` to a cosine similarity of `0.87`. Different
universes. Min-max normalization drags both into [0, 1]:

```
normalized = (score - min) / (max - min)
```

Do it **per query**, separately for the BM25 list and the dense list, *before*
you blend. Skip it and the bigger BM25 numbers bulldoze everything, your `alpha`
knob does nothing, and you'll lose an afternoon wondering why `alpha=0.4`
behaves exactly like `alpha=0.9`.

---

## The day normalization lied to Tanvi

Tanvi ships the weighted hybrid. `alpha=0.4`, BM25-leaning, exactly as the
error-code problem demanded. She types "What does ERR_5521 mean?" to enjoy the
win.

Rank 1 is the right runbook. Good. But rank 2 is the on-call escalation policy,
and rank 3 is the TLS cert runbook. Neither says `ERR_5521`. Strange. She tuned
`alpha` to give BM25 most of the vote, so why is everything below the top hit
full of passages BM25 never cared about?

Here's what happened, and it's sneaky. `ERR_5521` appears in exactly **one**
passage. BM25 gives that passage a fat score and every other passage essentially
zero. Then min-max normalization does its job a touch too well: the one match
becomes `1.0` and everything else gets crushed down to `~0.0`. There's no longer
any daylight between BM25's 2nd-favourite passage and its 50th. They all
flatlined together at the bottom.

<svg viewBox="0 0 900 290" width="100%" role="img" aria-label="The normalization squash" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="290" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">One outlier and min-max squashes the rest to zero</text>
  <text x="220" y="58" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">raw BM25 scores</text>
  <line x1="80" y1="240" x2="380" y2="240" stroke="#94a3b8"/>
  <line x1="80" y1="80" x2="80" y2="240" stroke="#94a3b8"/>
  <rect x="100" y="95" width="34" height="145" fill="#d97706"/>
  <text x="117" y="255" text-anchor="middle" font-size="10" fill="#475569">ERR_5521</text>
  <rect x="160" y="205" width="34" height="35" fill="#fbbf24"/>
  <rect x="220" y="212" width="34" height="28" fill="#fbbf24"/>
  <rect x="280" y="218" width="34" height="22" fill="#fbbf24"/>
  <text x="237" y="255" text-anchor="middle" font-size="10" fill="#475569">the rest (small but distinct)</text>
  <text x="660" y="58" text-anchor="middle" font-size="12" fill="#475569" font-weight="600">after min-max → [0,1]</text>
  <line x1="520" y1="240" x2="820" y2="240" stroke="#94a3b8"/>
  <line x1="520" y1="80" x2="520" y2="240" stroke="#94a3b8"/>
  <rect x="540" y="95" width="34" height="145" fill="#d97706"/>
  <text x="557" y="255" text-anchor="middle" font-size="10" fill="#475569">→ 1.0</text>
  <rect x="600" y="238" width="34" height="2" fill="#ef4444"/>
  <rect x="660" y="239" width="34" height="1" fill="#ef4444"/>
  <rect x="720" y="239" width="34" height="1" fill="#ef4444"/>
  <text x="690" y="255" text-anchor="middle" font-size="10" fill="#dc2626">all ≈ 0.0 — flatlined together</text>
  <text x="660" y="278" text-anchor="middle" font-size="11" fill="#991b1b">below rank 1, BM25 now contributes nothing</text>
</svg>

So below rank 1, BM25 has nothing left to say. The `alpha=0.4` knob keeps
turning, dutifully blending in 60% of a number that is now zero for every
remaining passage. Which means the *dense* retriever — the one that has no idea
what `ERR_5521` is — quietly picks the entire rest of the list. The knob Tanvi
trusted got quietly overruled by a rounding artifact.

That's the dirty secret of weighted blending: it assumes your scores are spread
out nicely. One lopsided query and that assumption is a flat-out liar. And
exact-token queries — the ones that matter most on call — are *always* lopsided.

### Way two: count places, not points (Reciprocal Rank Fusion)

There's an older, dumber, sturdier idea. Don't trust the scores at all. Trust
the *order*.

It's called **Reciprocal Rank Fusion** (Cormack, Clarke & Buettcher, 2009) and
the whole thing fits on a napkin. Each retriever hands you a ranked list. For
every doc, add up one over its position in each list:

```
rrf(doc) = Σ  1 / (k + rank in list i)
            i
```

A passage BM25 ranks first contributes `1/(k+0)`. Ranked second, `1/(k+1)`.
That's the entire algorithm. Nobody asks whether BM25's top score was a polite
`8.0` or a deranged `80.0` — first place is first place. The outlier that wrecked
min-max can't lay a finger on you here, because RRF never looks at the number,
only at the seat it's sitting in.

<svg viewBox="0 0 900 250" width="100%" role="img" aria-label="RRF as an election" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="250" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">RRF: an election, not a shouting match</text>
  <rect x="60" y="55" width="160" height="170" rx="8" fill="#fff7ed" stroke="#d97706"/>
  <text x="140" y="78" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">BM25 ballot</text>
  <text x="140" y="103" text-anchor="middle" font-size="11" fill="#92400e">1. doc-A</text>
  <text x="140" y="125" text-anchor="middle" font-size="11" fill="#92400e">2. doc-C</text>
  <text x="140" y="147" text-anchor="middle" font-size="11" fill="#92400e">3. doc-B</text>
  <rect x="260" y="55" width="160" height="170" rx="8" fill="#eff6ff" stroke="#2563eb"/>
  <text x="340" y="78" text-anchor="middle" font-size="12" fill="#1e3a8a" font-weight="600">Dense ballot</text>
  <text x="340" y="103" text-anchor="middle" font-size="11" fill="#1e3a8a">1. doc-C</text>
  <text x="340" y="125" text-anchor="middle" font-size="11" fill="#1e3a8a">2. doc-A</text>
  <text x="340" y="147" text-anchor="middle" font-size="11" fill="#1e3a8a">3. doc-D</text>
  <text x="470" y="145" font-size="22" fill="#475569">→</text>
  <rect x="520" y="55" width="320" height="170" rx="8" fill="#ffffff" stroke="#16a34a"/>
  <text x="680" y="78" text-anchor="middle" font-size="12" fill="#166534" font-weight="600">fused (k=60)</text>
  <text x="540" y="103" font-size="11" fill="#166534">doc-A:  1/60 + 1/61 = 0.0331  ← both rank it high</text>
  <text x="540" y="125" font-size="11" fill="#166534">doc-C:  1/61 + 1/60 = 0.0331  ← both rank it high</text>
  <text x="540" y="147" font-size="11" fill="#64748b">doc-B:  1/62 + 0      = 0.0161  ← only one liked it</text>
  <text x="540" y="169" font-size="11" fill="#64748b">doc-D:  0      + 1/62 = 0.0161  ← only one liked it</text>
  <text x="540" y="200" font-size="11" fill="#166534" font-weight="600">winners = the docs BOTH retrievers agreed on</text>
</svg>

Think of it as an election. Score-blending lets one wildly over-caffeinated
voter — BM25, screaming about its one exact match — shout down the whole room.
RRF makes everyone rank their picks and counts ballots. A passage that *both*
retrievers put near the top wins. A passage only one retriever loves does not.

That little `k` is the cynicism dial. It sets how much being #1 is worth over
being #5. At `k=60` — the value from the original paper, and a perfectly good
default — first place scores `1/60 ≈ 0.0167` and sixth place `1/65 ≈ 0.0154`.
Barely a gap. So RRF leans toward *agreement between retrievers* instead of
crowning whoever yelled loudest in one list. Want the top spot to dominate more?
Drop `k`. Want a flatter, consensus list? Raise it.

Switching Tanvi over is one argument:

```python
retriever = HybridRetriever(
    documents=documents,
    embed_fn=your_embed_fn,
    fusion="rrf",     # count places, not points
)
results = retriever.retrieve("What does ERR_5521 mean?", k=4)
```

Run the two methods side by side on the fixture corpus and watch them split:

```bash
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.fusion_compare
```

```
Query: 'What does ERR_5521 mean?'
rank   weighted (alpha=0.4)     rrf (k=60)
0      rb_002   1.000           rb_002   0.033
1      rb_009   0.411           rb_009   0.033
2      rb_011   0.134           rb_011   0.031
3      rb_005   0.123           rb_014   0.031   <- differs
4      rb_001   0.109           rb_013   0.030   <- differs
```

Same retrievers, same query, different lists below the top hit — because the two
methods flat-out disagree about how to treat BM25's lopsided scores. Notice the
thing that *doesn't* move: `rb_002`, the exact match, holds rank 1 under both.
RRF's whole job is to make sure the exact-match runbook can never get washed out
by a normalization artifact, and there it is, refusing to budge.

> **Handy Heuristic**
>
> Reach for **weighted blending** when you genuinely want a knob — when you
> *know* one retriever should outweigh the other and you'll actually tune it.
> Reach for **RRF** when you want it to just work and you don't trust the scores
> to be comparable. That's why Elasticsearch, OpenSearch, Qdrant, and Weaviate
> all ship RRF as the *default* fusion: there's no per-corpus dial to get wrong
> at 3 a.m.

---

## The reranker: a fast scout and a slow expert

Hybrid retrieval gets the right runbook *into the pile*. The last job is getting
it to the *top* of the pile. That's the reranker.

The retriever is fast but approximate. Cheap signals — term overlap, embedding
dot products — let it sweep a huge corpus down to maybe 50 candidates in
milliseconds. It's a scout: covers a lot of ground in a sprint, then yells "the
answer is *probably* in this pile."

The reranker is slow but accurate. It takes those 50 and re-scores each one with
a **cross-encoder** — a model that reads the *query and the passage together*
and emits one relevance number. It's the expert who actually reads each
candidate and ranks them properly.

<svg viewBox="0 0 900 300" width="100%" role="img" aria-label="Retrieve then rerank funnel" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs>
    <marker id="ar2" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#475569"/></marker>
  </defs>
  <rect x="0" y="0" width="900" height="300" fill="#f8fafc" rx="10"/>
  <text x="450" y="30" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Two-stage retrieval: cast wide, then sharpen</text>
  <polygon points="120,60 600,60 470,120 250,120" fill="#fde68a" stroke="#d97706"/>
  <text x="360" y="95" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">Corpus: every runbook + postmortem (10k+ docs)</text>
  <line x1="360" y1="120" x2="360" y2="140" stroke="#475569" stroke-width="2" marker-end="url(#ar2)"/>
  <rect x="240" y="145" width="240" height="40" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="360" y="170" text-anchor="middle" font-size="12" fill="#92400e" font-weight="600">RETRIEVER (bi-encoder) → top 50</text>
  <line x1="360" y1="185" x2="360" y2="205" stroke="#475569" stroke-width="2" marker-end="url(#ar2)"/>
  <rect x="280" y="210" width="160" height="40" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="360" y="235" text-anchor="middle" font-size="12" fill="#5b21b6" font-weight="600">RERANKER → top 4</text>
  <line x1="360" y1="250" x2="360" y2="270" stroke="#475569" stroke-width="2" marker-end="url(#ar2)"/>
  <rect x="300" y="272" width="120" height="24" rx="6" fill="#dcfce7" stroke="#16a34a"/>
  <text x="360" y="289" text-anchor="middle" font-size="11" fill="#166534" font-weight="600">LLM gets these</text>
  <rect x="600" y="120" width="280" height="150" rx="8" fill="#ffffff" stroke="#cbd5e1"/>
  <text x="740" y="145" text-anchor="middle" font-size="12" fill="#0f172a" font-weight="700">recall vs precision</text>
  <text x="618" y="172" font-size="11" fill="#d97706" font-weight="600">RETRIEVER's job: recall</text>
  <text x="618" y="190" font-size="11" fill="#475569">drag the right doc into the top 50</text>
  <text x="618" y="208" font-size="11" fill="#475569">somewhere, anywhere.</text>
  <text x="618" y="234" font-size="11" fill="#7c3aed" font-weight="600">RERANKER's job: precision</text>
  <text x="618" y="252" font-size="11" fill="#475569">of those 50, float the real 4 up.</text>
</svg>

### Bi-encoder vs cross-encoder: why you need two models

Both your retriever and your reranker answer "how well does this passage match
the query?" They answer it in completely different ways, and the difference is
the whole game.

<svg viewBox="0 0 900 320" width="100%" role="img" aria-label="Bi-encoder vs cross-encoder" font-family="ui-sans-serif, system-ui, sans-serif">
  <rect x="0" y="0" width="900" height="320" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Two models, two architectures</text>
  <rect x="30" y="50" width="410" height="250" rx="8" fill="#ffffff" stroke="#d97706"/>
  <text x="235" y="74" text-anchor="middle" font-size="13" fill="#92400e" font-weight="700">BI-ENCODER (retriever)</text>
  <rect x="60" y="95" width="120" height="34" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="120" y="117" text-anchor="middle" font-size="11" fill="#92400e">query</text>
  <rect x="290" y="95" width="120" height="34" rx="6" fill="#fef3c7" stroke="#d97706"/>
  <text x="350" y="117" text-anchor="middle" font-size="11" fill="#92400e">passage</text>
  <text x="120" y="155" text-anchor="middle" font-size="20" fill="#d97706">↓</text>
  <text x="350" y="155" text-anchor="middle" font-size="20" fill="#d97706">↓</text>
  <rect x="75" y="165" width="90" height="28" rx="6" fill="#fde68a"/>
  <text x="120" y="184" text-anchor="middle" font-size="11" fill="#92400e">vector</text>
  <rect x="305" y="165" width="90" height="28" rx="6" fill="#fde68a"/>
  <text x="350" y="184" text-anchor="middle" font-size="11" fill="#92400e">vector</text>
  <text x="235" y="225" text-anchor="middle" font-size="12" fill="#475569">compare the two vectors (dot product)</text>
  <text x="235" y="255" text-anchor="middle" font-size="11" fill="#16a34a" font-weight="600">FAST: embed corpus once, up front.</text>
  <text x="235" y="275" text-anchor="middle" font-size="11" fill="#dc2626">BLUNT: query never meets the passage.</text>
  <rect x="460" y="50" width="410" height="250" rx="8" fill="#ffffff" stroke="#7c3aed"/>
  <text x="665" y="74" text-anchor="middle" font-size="13" fill="#5b21b6" font-weight="700">CROSS-ENCODER (reranker)</text>
  <rect x="540" y="95" width="250" height="34" rx="6" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="665" y="117" text-anchor="middle" font-size="11" fill="#5b21b6">[query  +  passage]  glued together</text>
  <text x="665" y="155" text-anchor="middle" font-size="20" fill="#7c3aed">↓</text>
  <rect x="580" y="165" width="170" height="34" rx="6" fill="#ddd6fe" stroke="#7c3aed"/>
  <text x="665" y="187" text-anchor="middle" font-size="11" fill="#5b21b6">one model reads both at once</text>
  <text x="665" y="225" text-anchor="middle" font-size="12" fill="#475569">outputs ONE relevance number</text>
  <text x="665" y="255" text-anchor="middle" font-size="11" fill="#dc2626">SLOW: no precompute, one pair at a time.</text>
  <text x="665" y="275" text-anchor="middle" font-size="11" fill="#16a34a" font-weight="600">SHARP: sees the actual words side by side.</text>
</svg>

The **bi-encoder** reads the query and turns it into a vector. Separately —
usually months earlier, at indexing time — it read each passage and turned *it*
into a vector. At query time it just compares the two vectors. The query and the
passage never actually meet; they're introduced by their coordinates, like two
people set up by a dating app that only ever saw their profiles.

That's why it's fast. Embed the corpus once, and a query is one embedding plus a
batch of dot products. Ten million docs, no sweat.

It's also why it's blunt. Take two runbooks: "to recover a stuck pod, restart it
directly" and "*never* manually restart the pod." In vector space they sit
practically on top of each other — both about stuck pods and restarting. The
bi-encoder squashed each one into a single point *before it ever saw your
question*, so the part that flips the meaning — that one "never" — is gone by the
time you ask. It can't go back and reread.

The **cross-encoder** takes the query and one passage, glues them into a single
input, and reads them together. Now "restart" and "never restart" aren't nearby
points, they're tokens the model weighs against your actual question. That joint
read is what makes it sharp. It's also why it can't precompute anything: the
passage has to be scored against *your specific query*, so the work only happens
at query time, one pair at a time. That's the whole reason you run two stages —
cheap-and-blunt to narrow the field, pricey-and-sharp to pick the winners:

<svg viewBox="0 0 900 300" width="100%" role="img" aria-label="Latency vs accuracy two stages" font-family="ui-sans-serif, system-ui, sans-serif">
  <defs><marker id="p1f" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#7c3aed"/></marker></defs>
  <rect x="0" y="0" width="900" height="300" fill="#f8fafc" rx="10"/>
  <text x="450" y="28" text-anchor="middle" font-size="14" fill="#0f172a" font-weight="700">Why two stages: trade speed for sharpness, in that order</text>
  <line x1="110" y1="250" x2="780" y2="250" stroke="#94a3b8"/>
  <line x1="110" y1="60" x2="110" y2="250" stroke="#94a3b8"/>
  <text x="445" y="282" text-anchor="middle" font-size="11" fill="#475569">cost / latency per document  →</text>
  <text x="92" y="150" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 92 150)">accuracy  →</text>
  <circle cx="200" cy="210" r="12" fill="#fde68a" stroke="#d97706"/>
  <text x="200" y="194" text-anchor="middle" font-size="11" fill="#92400e" font-weight="600">bi-encoder</text>
  <text x="200" y="234" text-anchor="middle" font-size="10" fill="#92400e">cheap, blunt</text>
  <circle cx="690" cy="100" r="12" fill="#ddd6fe" stroke="#7c3aed"/>
  <text x="690" y="84" text-anchor="middle" font-size="11" fill="#5b21b6" font-weight="600">cross-encoder</text>
  <text x="690" y="124" text-anchor="middle" font-size="10" fill="#5b21b6">pricey, sharp</text>
  <path d="M 212 206 C 420 200, 520 150, 678 108" fill="none" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#p1f)"/>
  <text x="430" y="150" text-anchor="middle" font-size="11" fill="#5b21b6">run bi-encoder on 10M docs → 50,</text>
  <text x="430" y="167" text-anchor="middle" font-size="11" fill="#5b21b6">then cross-encoder on just those 50 → 4</text>
</svg>

```
bi-encoder    (retriever):  10,000,000 docs  →  top 50     cheap, blunt
cross-encoder (reranker):   top 50           →  top 4      pricey, sharp
```

One catch that bites everyone exactly once: **the reranker can only promote what
the retriever already found.** If the right runbook isn't in the top 50, no
amount of reranking conjures it — you're just re-sorting a list that doesn't
contain the answer. When *recall* is your problem, fix the retriever. The
reranker is for precision, full stop.

`rerank_demo.py` runs the whole chain end to end — hybrid retrieval down to 10
candidates, a real cross-encoder (`ms-marco-MiniLM-L-6-v2`) down to 4, then the
model — and prints the before-and-after so you can watch passages climb:

```bash
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.rerank_demo
```

> **Handy Heuristic**
>
> Rerank 20–50 candidates, not 1,000. Below ~10 there's nothing to sort; above
> ~50 you're paying cross-encoder latency on docs the retriever already knows
> are junk. If you catch yourself reranking hundreds, the bug is in your
> retriever's recall, not your reranker's depth.

---

## See it work (Part 1, runnable with zero setup)

The example ships with a bag-of-words stub embedder, so the whole hybrid + RRF
machinery runs without an API key or a model download:

```bash
# watch weighted blending and RRF disagree on an exact-token query
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.fusion_compare

# retrieve wide, then rerank with a cross-encoder
python -m vol2_plumbing.part09_rag_retrieval.examples.oncall_rag.rerank_demo
```

The BM25 half works correctly. The dense half uses hashed word vectors — better
than random noise, far worse than a real encoder. Swap in `sentence-transformers`
(or any embedding API) via the `embed_fn` argument and the dense half starts
pulling its real weight.

```python
from vol2_plumbing.part09_rag_retrieval.retriever import Document, HybridRetriever

documents = [Document(id="rb_002", text="...", metadata={"source": "Error Codes"})]
retriever = HybridRetriever(documents=documents, embed_fn=my_embedder, alpha=0.4)

for r in retriever.retrieve("What does ERR_5521 mean?", k=4):
    print(f"{r.document.id}  score={r.score:.2f}  source={r.source}  "
          f"bm25_rank={r.bm25_rank}  dense_rank={r.dense_rank}")
```

`embed_fn` takes a list of strings and returns an `(N, dim)` numpy array —
`sentence-transformers`, Voyage AI (Anthropic's recommended embedding provider),
Cohere `embed-v3`, OpenAI `text-embedding-3`, anything. The retriever doesn't
care which.

### Debugging: which retriever actually put this chunk here?

`source` only tells you the *fusion mode* — every doc in a hybrid result set
says `"hybrid"`, whether BM25 found it, dense found it, or both agreed. That's
not enough to debug a bad answer. So `RetrievalResult` also carries each
retriever's *own* opinion: `bm25_rank`, `dense_rank`, `bm25_score`,
`dense_score` — independent of whichever one won the fusion vote. Run the
top 4 of the `ERR_5521` query above and it reads:

```
rb_002  source=hybrid  bm25_rank=0   dense_rank=0   bm25_score=5.38  dense_score=0.16
rb_009  source=hybrid  bm25_rank=1   dense_rank=1   bm25_score=1.69  dense_score=0.09
rb_011  source=hybrid  bm25_rank=6   dense_rank=2   bm25_score=0.00  dense_score=0.05
rb_005  source=hybrid  bm25_rank=11  dense_rank=3   bm25_score=0.00  dense_score=0.05
```

`bm25_score=0.00` on `rb_011` and `rb_005` is the smoking gun: BM25 had
nothing to say about either one, so dense alone dragged them in — the exact
squash from the section above, now visible per chunk instead of guessed at.
`HybridRetriever.retrieve()` logs this as a `rag.result` span event per
result (see `common/tracing.py`), so in production it's a trace query —
"show me every `hybrid` chunk where `bm25_score` rounds to zero" — not a
guess at 3 a.m.

---

## Where Part 1 leaves you

You now have the load-bearing spine of every serious RAG system:

> **Hybrid retrieve → rerank → generate.** BM25 handles the exact tokens. Dense
> handles the paraphrases. Fusion (RRF) settles their fights without trusting
> shaky scores. The cross-encoder reranker sharpens the survivors. And the whole
> thing rests on one idea you can't un-see: **RAG is a search problem.**

Build only this and you're already ahead of most RAG shipping today. But the
field didn't stop in 2023, and Tanvi's bot is about to hit a fresh batch of
failures — chunks that lose their context, questions phrased so badly nothing
matches, a *deprecated runbook* that outranks the postmortem warning against it,
and questions no single chunk can ever answer.

That's **Part 2: Making the Search Actually Trustworthy** — chunking and
Contextual Retrieval, late interaction (ColBERT / MUVERA / BGE-M3 / ColPali),
query rewriting (HyDE, decompose, step-back), self-correcting and agentic RAG
(CRAG, Self-RAG), GraphRAG and RAPTOR for "what keeps breaking everywhere," the
honest take on long context, and the full 2026 stack in one picture.

---

*Continue to Part 2 → `article_part2.md`*

*Code: [github.com/saurabhtg/Applied-AI](https://github.com/saurabhtg/Applied-AI)*

*All thirty parts: [Applied AI Series](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/)*
