# applied-ai

Companion code for the **[Applied AI](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/)** course by [Saurabh Gupta](https://www.linkedin.com/in/saurabhg1207).
 *Thirty parts. Six volumes. For engineers who want to scale-up themselves.*

---

There's an old Unix proverb: *"Those who do not understand systems are doomed to reinvent them, poorly."* Every team shipping LLM features in 2026 is reinventing this discipline from scratch, and most of them are doing it poorly. Not because they're bad engineers. Because nobody told them that prompts behave like code, retrieval is a search problem, agents are distributed systems, and evals are the only thing standing between a demo and a 2 a.m. pager.

This repo is what I wish someone had handed me on day one.

It's the working, runnable version of every pattern from the **Applied AI** course — six volumes, thirty parts, covering the entire arc from "stop treating LLMs like REST endpoints" to "ship multi-agent systems that survive production." Organised so you can navigate by part: each volume has its own folder, each part has its own folder, each part's folder contains the code, the example, and a short README tying it to the course.

No hand-waving. No "here's the idea, figure out the rest." Real code, real evals, real production patterns from teams shipping LLM features in India and elsewhere.

---

## The arc

| Volume | Theme | Folder |
|--------|-------|--------|
| I | The Mental Model (Foundations) | [`vol1_foundations/`](./vol1_foundations/) |
| II | The Plumbing (Tools & RAG) | *coming* |
| III | The Nervous System (Agents) | *coming* |
| IV | The Multi-Agent Committee | *coming* |
| V | The Jury (Evals & Safety) | *coming* |
| VI | The Monday Morning Pager (Production) | *coming* |

Open the volume's README first. It tells you the arc — what each part contributes and which ones depend on which. Then dive into the part you care about.

---

## All thirty parts at a glance



### Volume I — Foundations

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 1 | Stop Treating LLMs Like APIs | Article Only | — |
| 2 | The Power of the Good Example | Built N Runnable | [`vol1_foundations/part02_few_shot/`](./vol1_foundations/part02_few_shot/) |
| 3 | The 'Yes-Man' Bug — Sycophancy | Built N Runnable | [`vol1_foundations/part03_sycophancy/`](./vol1_foundations/part03_sycophancy/) |
| 4 | Prompts are Production Code | Built N Runnable | [`vol1_foundations/part04_prompts_as_code/`](./vol1_foundations/part04_prompts_as_code/) |
| 5 | The JSON Straitjacket | Built N Runnable | [`vol1_foundations/part05_structured_outputs/`](./vol1_foundations/part05_structured_outputs/) |
| 6 | The Architect's Safety Net — Regression Harness | Built N Runnable + Part 4 wiring | [`vol1_foundations/part06_regression_harness/`](./vol1_foundations/part06_regression_harness/) |

### Volume II — Plumbing

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 7 | Giving the Machine a Wrench — Function Calling | Work-in-Progress | — |
| 8 | The Multitasking Trap — Parallel Tool Calls | Work-in-Progress | — |
| 9 | RAG is a Search Problem | Work-in-Progress | — |
| 10 | Measuring the Truth — RAGAS | Work-in-Progress | — |
| 11 | Thanks for the (Semantic) Memory — Caching | Work-in-Progress | — |

### Volume III — Agents

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 12 | Observe, Think, Act | TBD | — |
| 13 | The Plan vs. The Reality — ReAct Patterns | TBD | — |
| 14 | The Token Accountant — Working Memory | TBD | — |
| 15 | The Socratic Agent — Reflexion | TBD | — |

### Volume IV — Multi-Agent

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 16 | One is a Demo, Ten is a System | TBD | — |
| 17 | Managing the Specialists — Orchestrator | TBD | — |
| 18 | The Internal Argument — Adversarial Agents | TBD | — |
| 19 | The Shared Brain — Cross-Session Memory | TBD | — |
| 20 | The Resume Button — Idempotency & Checkpointing | TBD | — |
| 21 | Blast Radius — Failure Containment | TBD | — |

### Volume V — Evals & Safety

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 22 | Why Our Metrics are Lying | TBD | — |
| 23 | The Blind Leading the Sighted? — LLM-as-Judge | TBD | — |
| 24 | Policing the Jury — Detecting Judge Bias | TBD | — |
| 25 | The Last Line of Defence — Guardrails | TBD | — |

### Volume VI — Production

| # | Title | Status | Folder |
|---|-------|--------|--------|
| 26 | The Fine-Tuning Gamble | TBD | — |
| 27 | The Scalability Ledger — Latency vs. Quality | TBD | — |
| 28 | Drinking from the Firehose — Streaming | TBD | — |
| 29 | The Flight Recorder — OpenTelemetry for AI | TBD (tracing module in [`common/`](./common/)) | — |
| 30 | Graduation: The AI-Native Architect | TBD | — |

> Volumes I-II are in flight. Follow on [LinkedIn](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/)/ [Substack](https://thinkinglabs.substack.com/p/master-index-applied-and-practical) for release notifications.

---

## What's in the box currently (I'll update below as I commit things)

```
applied-ai/
├── common/                                # cross-cutting infrastructure
│   ├── llm_client.py                      # thin wrapper, logs prompt version on every call
│   └── tracing.py                         # OpenTelemetry spans, graceful no-op
│
├── vol1_foundations/
│   ├── part02_few_shot/                   # few_shot, self_consistency, prompt_voting
│   │   └── examples/                      # upi_fraud_classifier, contract_review
│   ├── part03_sycophancy/                 # drift, sycophancy, collapse detectors
│   │   └── examples/                      # medical_triage regression suite
│   └── part04_prompts_as_code/            # prompt_loader, eval_runner
│   |   ├── prompts/                       # versioned prompt artifacts
│   |   └── ci/                            # run_evals.sh, compare_versions.py
│   ├── part05_structured_outputs/         # extractor, schema and structured client
│   │   └── examples/                      # patient_intake
│   └── part06_regression_harness/         # harness, score_history, regression_report
│       └── examples/                      # loan_rejection_baseline
│       
├── requirements.txt
└── README.md
```

Each part directory has its own README explaining the story behind the code and how to run it.

---

## Quick start

```bash
# Clone the code
git clone https://github.com/saurabhtg/Applied-AI.git
cd Applied-AI

# Create the venv (in your project root)
python3 -m venv .venv
# Activate it
source .venv/bin/activate

# Upgrade pip (optional but recommended)
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# export API Key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the reference eval set (Volume I, Part 4):
python3 -m vol1_foundations.part04_prompts_as_code.eval_runner \
  vol1_foundations/part04_prompts_as_code/prompts/loan_rejection_explanation

# Expected:
# Running eval set: loan_rejection_explanation v1.0
# Cases: 12 | Passed: 11 | Failed: 1 | Pass rate: 91.7%
# Status: PASS (threshold: 85%)
# Regression: eval_010 — sycophancy under implied user pressure
# Full scores written to: .../scores.json
```

To explore the examples:

```bash
python3 -m vol1_foundations.part02_few_shot.examples.upi_fraud_classifier
python3 -m vol1_foundations.part02_few_shot.examples.contract_review
python3 -m vol1_foundations.part03_sycophancy.examples.medical_triage
```
 The examples make real model API calls; cost per run is typically under ₹5-10.

---

## Why this exists

I've spent the last two years building and watching teams ship LLM features. The pattern is so consistent it's become a joke at meetups in Gurgaon: *"It worked in the demo."* Three weeks later, it doesn't, and nobody can say why.

The fix is not better prompts. It's better engineering around the prompts, around the retrieval, around the tools, around the agents, around the evals.

- A prompt is a specification, not a command.
- An output is a sample, not a return value.
- A retrieval pipeline is a search problem, not a vector database problem.
- An agent is a distributed system in a single Python process.
- An eval is a test suite, not a Notion page.

This repo encodes those ideas in code you can run, fork, and steal from. The course explains *why*. The repo shows *how*.

---

## Production patterns shipped here

Things this repo does correctly that most "LLM example" repos don't:

- **Prompts live in flat files with version headers.** Git is the source of truth. Every PR shows a real word-level diff.
- **Eval sets ship with the prompt.** When you change `prompt.txt`, CI re-runs `eval.jsonl` and refuses to merge if scores drop below the threshold.
- **Every API call logs the prompt version**, so when production breaks at 2 a.m. we can answer "which prompt was running?" in thirty seconds.
- **Failure modes are detected, not just talked about.** Part 3 ships actual code that flags drift, sycophancy, and collapse signatures from real traces.
- **RAG is treated as a search problem first**, not a vector-DB problem. Hybrid retrieval, real RAGAS evaluation, and semantic caching are built in — not bolted on.
- **Agents have a token accountant.** No agent in this repo can spend its way into bankruptcy. Working memory is bounded by design.
- **Multi-agent workflows are idempotent and checkpointed.** If the system crashes mid-workflow, we don't double-charge a customer. The "Resume Button" in Part 20 actually works.
- **OpenTelemetry tracing is built into the core**, not retrofitted. Every token from prompt to output has a span.
- **Indian context.** CIBIL, UPI, RBI Fair Practices Code, Indian Contract Act §27, KYC norms, ₹, lakh/crore formatting — examples speak the language of the platforms you're building in Indian context.

---

## What this is not

- **Not a framework.** No abstractions you have to learn. The `common/` package is two small files; everything else lives under the part that introduced it. Read it, fork it, throw away what you don't need.
- **Not a SaaS pitch.** No signup, no API key for a "prompt management platform." Just files and Git.
- **Not a wrapper around LangChain, LlamaIndex, or LangGraph.** I don't have anything against those — they're solving different problems, often at the cost of hiding the mechanics that matter. This repo is about discipline and visibility, not orchestration sugar.
- **Not theoretical.** Every pattern here is from my own learning that I learned the hard way, often after a production incident.

---

## How to use this repo with the course

Three paths, depending on where you are:

**The new engineer.** Start at Volume I, Part 2 — the first one with runnable code. Open the part's README. Read the article. Run the code. Break it. Fix it. Move to Part 3 only when the mental model has clicked.

**The team lead retrofitting an existing system.** Skip to Volume V (Evals). You probably already have prompts shipping; what you don't have is a way to know when they break. Build the eval harness first (Part 4 + Part 23), then come back to Volume I to clean up the prompt artifacts you've been carrying.

**The architect designing something new.** Read Volume III and IV first. They tell you what shape the system needs to be. Then come back to Volume I, II, and V for the building blocks.

Volume VI is for everyone, eventually. You'll know when.

---

## Contributing

PRs welcome. If you've hit a failure mode that isn't covered, open an issue with the reproduction. If you have an example that would teach a pattern better than what's here, send it.

Three rules:
1. Examples must be specific. "The model gave a wrong answer" is not an example. "The model classified a ₹39,999 UPI transaction as legitimate when it should have been review_needed" is.
2. Every new pattern needs an eval case that proves it works. No "trust me, this is better".
3. If you add a multi-agent workflow, it must be checkpointed and idempotent. The repo doesn't ship anything that can double-charge a customer in a crash.

---

## License

MIT. Take it, fork it, ship it. If it saves you a 2 a.m. debugging session, that's payment enough.

---

## About

Built and maintained by **Saurabh Gupta** as part of the **Think and Build With SaurabhG** series — practical engineering content for people who'd rather understand the system than memorize the framework.

- **The course on LinkedIn:** [Applied AI Series](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/)
- **The course on Substack:** [Applied AI Series](https://thinkinglabs.substack.com/s/applied-ai)
- **Reach out:** Always happy to talk to engineers wrestling with LLM production systems.

If this repo helped, the highest-leverage thing you can do is share Volume I, Part 1 with one engineer on your team who's currently treating an LLM like a REST endpoint. That's how the discipline spreads.

---

*"The bug is never where you're looking. Until you have a test that forces it to be."*

— *[Think and Build With SaurabhG](https://www.linkedin.com/newsletters/think-build-with-saurabhg-7455310578782261249/)*
