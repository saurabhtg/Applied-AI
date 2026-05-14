# applied-ai

> Companion code for the **Applied AI** course by [Saurabh Gupta](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/?trackingId=t0XsSFox8FEIxb6PGaCnYg%3D%3D).
> *Thirty parts. Six volumes. For engineers.*

---

There's an old Unix proverb: "Those who do not understand systems are doomed to reinvent them, poorly". Every team shipping LLM features in 2026 is reinventing this discipline from scratch, and most of them are doing it poorly. Not because they're bad engineers. Because nobody told them that prompts behave like code, retrieval is a search problem, agents are distributed systems, and evals are the only thing standing between a demo and a 2 a.m. pager.

This repo is what I wish someone had handed me on day one.

It's the working, runnable version of every pattern from the **Applied AI** course — six volumes, thirty parts, covering the entire arc from "stop treating LLMs like REST endpoints" to "ship multi-agent systems that survive production".

No hand-waving. No "here's the idea, figure out the rest". Real code, real evals, real production patterns from teams shipping LLM features.

---

## The arc

Six volumes, each building on the last. You don't have to read them in order — but skipping volumes is how the leaky abstractions catch up with you later.

### Volume I — The Mental Model (Foundations)
*You can't build a skyscraper on a swamp.*

| Part | Title | Folder |
|------|-------|--------|
| 1 | Stop Treating LLMs Like APIs | No code |
| 2 | The Power of the Good Example | No code |
| 3 | The 'Yes-Man' Bug — Sycophancy | No code |
| 4 | Prompts are Production Code | `vol1_foundations/part04_prompts_as_code/` |
| 5 | The JSON Straitjacket | `vol1_foundations/part05_structured_outputs/` |
| 6 | The Architect's Safety Net — Regression Harness | `vol1_foundations/part06_regression_harness/` |

### Volume II — The Plumbing (Tools & RAG)
*LLMs are blind without data and toothless without tools.*

| Part | Title | Folder |
|------|-------|--------|
| 7 | Giving the Machine a Wrench — Function Calling | `vol2_plumbing/part07_function_calling/` |
| 8 | The Multitasking Trap — Parallel Tool Calls | `vol2_plumbing/part08_parallel_tools/` |
| 9 | RAG is a Search Problem | `vol2_plumbing/part09_rag_retrieval/` |
| 10 | Measuring the Truth — RAGAS | `vol2_plumbing/part10_ragas/` |
| 11 | Thanks for the (Semantic) Memory — Caching | `vol2_plumbing/part11_semantic_cache/` |

### Volume III — The Nervous System (Agents)
*Moving from "chatting" to "doing." This is where the abstractions get leaky.*

| Part | Title | Folder |
|------|-------|--------|
| 12 | Observe, Think, Act | `vol3_agents/part12_observe_think_act/` |
| 13 | The Plan vs. The Reality — ReAct | `vol3_agents/part13_react_patterns/` |
| 14 | The Token Accountant — Working Memory | `vol3_agents/part14_working_memory/` |
| 15 | The Socratic Agent — Reflexion | `vol3_agents/part15_reflexion/` |

### Volume IV — The Multi-Agent Committee
*One agent is a hobby; ten agents is a distributed systems nightmare.*

| Part | Title | Folder |
|------|-------|--------|
| 16 | One is a Demo, Ten is a System | `vol4_multi_agent/part16_specialists/` |
| 17 | Managing the Specialists — Orchestrator | `vol4_multi_agent/part17_orchestrator/` |
| 18 | The Internal Argument — Adversarial Agents | `vol4_multi_agent/part18_adversarial/` |
| 19 | The Shared Brain — Cross-Session Memory | `vol4_multi_agent/part19_shared_memory/` |
| 20 | The Resume Button — Idempotency & Checkpointing | `vol4_multi_agent/part20_checkpointing/` |
| 21 | Blast Radius — Failure Containment | `vol4_multi_agent/part21_blast_radius/` |

### Volume V — The Jury (Evals & Safety)
*In the world of LLMs, unit tests are necessary but insufficient.*

| Part | Title | Folder |
|------|-------|--------|
| 22 | Why Our Metrics are Lying — Beyond ROUGE/BLEU | `vol5_evals/part22_metrics/` |
| 23 | The Blind Leading the Sighted? — LLM-as-Judge | `vol5_evals/part23_llm_judge/` |
| 24 | Policing the Jury — Detecting Judge Bias | `vol5_evals/part24_judge_bias/` |
| 25 | The Last Line of Defence — Guardrails | `vol5_evals/part25_guardrails/` |

### Volume VI — The Monday Morning Pager (Production)
*Shipping code that lets us sleep through the night.*

| Part | Title | Folder |
|------|-------|--------|
| 26 | The Fine-Tuning Gamble | `vol6_production/part26_fine_tuning/` |
| 27 | The Scalability Ledger — Latency vs. Quality | `vol6_production/part27_latency_budget/` |
| 28 | Drinking from the Firehose — Streaming | `vol6_production/part28_streaming/` |
| 29 | The Flight Recorder — OpenTelemetry for AI | `vol6_production/part29_observability/` |
| 30 | Graduation: The AI-Native Architect | `vol6_production/part30_graduation/` |

> Volumes I and II are shipping now. III through VI roll out alongside the course. Follow on [LinkedIn](https://www.linkedin.com/newsletters/think-build-with-saurabhg-7455310578782261249/) for release notifications.

---

## What's in the box

A small framework for treating LLM systems as engineering artifacts, not Slack experiments.

```
applied-ai/
├── vol1_foundations/           # prompts, evals, structure
├── vol2_plumbing/              # tools, RAG, caching
├── vol3_agents/                # single-agent loops
├── vol4_multi_agent/           # orchestration, memory, blast radius
├── vol5_evals/                 # judges, guardrails, bias detection
├── vol6_production/            # observability, fine-tuning, streaming
│
├── core/                       # shared infrastructure used across volumes
    ├── prompt_loader.py        # load_prompt() with header parsing
    ├── eval_runner.py          # run an eval set, write scores.json
    ├── llm_client.py           # thin wrapper around the model API
    ├── tracing.py              # OpenTelemetry spans for every call
    ├── failure_detectors/
    │   ├── drift.py            # system-prompt drift detection
    │   ├── sycophancy.py       # agreement-with-user-when-wrong
    │   └── collapse.py         # repetitive/templated output
    ├── techniques/
    │   ├── few_shot.py
    │   ├── self_consistency.py
    │   └── prompt_voting.py
    ├── rag/
    │   ├── retriever.py        # hybrid BM25 + vector
    │   ├── reranker.py
    │   └── semantic_cache.py
    ├── agents/
    │   ├── react_loop.py       # the canonical ReAct implementation
    │   ├── orchestrator.py     # multi-agent coordination
    │   ├── checkpoint.py       # idempotency + resume
    │   └── working_memory.py   # token accountant
    └── evals/
        ├── llm_judge.py
        ├── ragas_runner.py
        └── guardrails.py
```

Every example is grounded in a real production context — UPI fraud classification, loan rejection explanations with RBI Fair Practices Code compliance, contract review that knows the difference between US and Indian law, telemedicine triage that doesn't fold when a patient pushes back, RAG over KYC compliance documents.

---

## Quick start

```bash
# Clone
git clone https://github.com/saurabhtg/Applied-AI.git
cd applied-ai

# Install
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run an eval against a sample prompt (Volume I, Part 4)
python -m core.eval_runner vol1_foundations/part04_prompts_as_code/prompts/loan_rejection_explanation

# Output:
# Running eval set: loan_rejection_explanation v3.7
# Cases: 12 | Passed: 11 | Failed: 1 | Pass rate: 91.7%
# Status: PASS (threshold: 85%)
# Regression: eval_010 — sycophancy under implied user pressure
# Full scores written to: prompts/loan_rejection_explanation/scores.json
```

That's it. Now you have versioned prompts, runnable evals, and a score file that goes into Git alongside your prompt. The same `core/` modules power everything from Volume II RAG to Volume IV multi-agent orchestration.

No SaaS. No API keys for a fourth-party tool. No monthly bill.

---

## Why this exists

I've spent the last two years watching teams ship LLM features. The pattern is so consistent it's become a joke at meetups in Bengaluru: "It worked in the demo". Three weeks later, it doesn't, and nobody can say why.

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
- **Every API call logs the prompt version**, so when production breaks at 2 a.m. you can answer "which prompt was running?" in thirty seconds.
- **Failure modes are detected, not just talked about.** `core/failure_detectors/` has actual code that flags drift, sycophancy, and collapse signatures from real traces.
- **RAG is treated as a search problem first**, not a vector-DB problem. Hybrid retrieval, real RAGAS evaluation, and semantic caching are built in — not bolted on.
- **Agents have a token accountant.** No agent in this repo can spend its way into bankruptcy. Working memory is bounded by design.
- **Multi-agent workflows are idempotent and checkpointed.** If the system crashes mid-workflow, you don't double-charge a customer. The "Resume Button" in Part 20 actually works.
- **OpenTelemetry tracing is built into the core**, not retrofitted. Every token from prompt to output has a span.

---

## What this is not

- **Not a framework.** No abstractions you have to learn. The whole `core/` directory aims for under 2,000 lines of Python by the time all six volumes ship. Read it, fork it, throw away what you don't need.
- **Not a SaaS pitch**. No signup, no API key for a "prompt management platform". Just files and Git.
- **Not a wrapper around LangChain, LlamaIndex, or LangGraph**. I don't have anything against those — they're solving different problems, often at the cost of hiding the mechanics that matter. This repo is about discipline and visibility, not orchestration sugar.
- **Not theoretical.** Every pattern here is from a team that learned it the hard way, often after a production incident with a real customer-facing consequence.

---

## How to use this repo with the course

Three paths, depending on where you are:

**The new engineer.** Start at Volume I, Part 1 to 4. Read the article, then open `vol1_foundations/part04_prompts_as_code/`. Run the code. Break it. Fix it. Move to Part 5 only when the mental model has clicked.

**The team lead retrofitting an existing system.** Skip to Volume V (Evals). You probably already have prompts shipping; what you don't have is a way to know when they break. Build the eval harness first, then come back to Volume I to clean up the prompt artifacts you've been carrying.

**The architect designing something new.** Read Volume III and IV first. They tell you what shape the system needs to be. Then come back to Volume I, II, and V for the building blocks.

Volume VI is for everyone, eventually. You'll know when.

---

## Contributing

PRs welcome. If you've hit a failure mode that isn't covered, open an issue with the reproduction. If you have an Indian-context example that would teach a pattern better than what's here, send it.

Three rules:
1. Examples must be specific. "The model gave a wrong answer" is not an example. "The model classified a ₹49,999 UPI transaction as legitimate when it should have been review_needed" is.
2. Every new pattern needs an eval case that proves it works. No "trust me, this is better".
3. If you add a multi-agent workflow, it must be checkpointed and idempotent. The repo doesn't ship anything that can double-charge a customer in a crash.

---

## License

MIT. Take it, fork it, ship it. If it saves you a 2 a.m. debugging session, that's payment enough.

---

## About

Built and maintained by **Saurabh Gupta** as part of the **Think and Build With SaurabhG** series — practical engineering content for people who'd rather understand the system than memorize the framework.

-  **The course on LinkedIn:** [Applied AI Series](https://www.linkedin.com/pulse/master-index-applied-practical-ai-saurabh-gupta-f8htc/?trackingId=t0XsSFox8FEIxb6PGaCnYg%3D%3D)
-  **Reach out:** Always happy to talk to engineers wrestling with LLM production systems.

If this repo helped, the highest-leverage thing you can do is share Volume I, Part 1 with one engineer on your team who's currently treating an LLM like a REST endpoint. That's how the discipline spreads.

---

*"The bug is never where you're looking. Until you have a test that forces it to be."*

— *Think and Build With SaurabhG*
