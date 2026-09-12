# Build an Enterprise Knowledge Assistant — the course

A follow-along course that teaches you to build a **production-style GenAI system** from
scratch: an assistant that answers questions from a company's documents, cites its sources,
follows relationships across documents (a knowledge graph), refuses unsafe requests, caches
answers, records every step, and runs on the public internet.

Written for **complete beginners**. Every technical term is explained the first time it
appears, and every meaningful piece of code is shown as a real excerpt from this repository
and explained in plain English — what it does *and* why it is there.

> **Live demo:** https://nitishgalat-enterprise-knowledge-assistant.hf.space
> **Code:** `github.com/niti007/knowledge-graph-version-2`

One thing makes this course different from a tutorial: the numbers are real. Every figure you
read (chunk counts, graph sizes, eval scores, latencies, attack results) came from running the
system in this repository, and where the honest number was worse than the flattering one, the
honest one is the one printed.

---

## How to use this course

- Read the chapters **in order**. Each one builds on the last, and the order is also the
  order the system is built and started in.
- Every chapter opens with **Why this / what's the need** — the problem in plain language,
  before any code.
- Watch for **🔑 New word** callouts. They define a term in one sentence.
- Every chapter ends with **✅ You just learned**, **▶️ Run this now**, and **🧠 Check
  yourself**. Do the "Run this now" step — reading code and running code teach different things.
- Teaching a class? Go straight to **[Chapter 15 — Teacher Notes](15-teacher-notes.md)** for
  a schedule, demo tips, common student mistakes, and exercises.

## What you'll need (details in Chapter 01)

A computer with Python 3.11, Docker, and Node.js; free accounts on OpenRouter (the AI
model), Qdrant Cloud (vector database), Langfuse Cloud (tracing), and Tavily (web search).
Budget: the whole build including every evaluation run cost **$3.18** of OpenRouter credit.

---

## Table of contents

### Part 1 — Understand and set up
- **[00 — What we're building and why](00-what-and-why.md)** — the business problem, the
  architecture, and why each technology was chosen (no coding).
- **[01 — Tools and accounts](01-tools-and-accounts.md)** — Python via `uv`, Docker, Node,
  and exactly where to get each API key.
- **[02 — Project setup](02-project-setup.md)** — `make install`, `.env`, `make up`, and the
  Phase 0 credential gate that nothing proceeds without.
- **[03 — Settings](03-settings.md)** — `app/config.py`, the one place every knob lives, and
  why several defaults carry a comment explaining a measured number.

### Part 2 — Build the pieces (real code, line by line)
- **[04 — The corpus and ingestion](04-corpus-and-ingestion.md)** — loading PDFs, Markdown
  and CSVs; the `Auth-DB` vs `Auth-Db` canonicalization story; rendering spreadsheets as prose.
- **[05 — Chunking and the vector index](05-chunking-and-vector-index.md)** — the
  512-token truncation bug, the real-tokenizer fix, BGE's query/passage asymmetry, and
  idempotent upserts with an orphan purge.
- **[06 — The knowledge graph](06-knowledge-graph.md)** — a deterministic 98-node / 236-edge
  Neo4j graph, why `DEPENDS_ON` is 8 edges not 15, and the 8 parameterized Cypher templates.
- **[07 — Hybrid retrieval](07-hybrid-retrieval.md)** — Reciprocal Rank Fusion, cross-encoder
  re-ranking, uncued-fact demotion, and the finding that "relevance is not answerability".
- **[08 — The agent](08-the-agent.md)** — a LangGraph loop with three tools, a structural
  4-iteration bound, and citations that the model cannot invent.
- **[09 — Guardrails](09-guardrails.md)** — NeMo Guardrails Colang rails, Presidio PII masking
  (and the `.internal` email recognizer), the three-signal grounding rail, and the
  two-predicate citation policy.
- **[10 — The API and UI](10-api-and-ui.md)** — FastAPI, in-memory sessions, `/metrics`, the
  system-card meta router, and a thin Streamlit client.
- **[11 — Caching, tiering, tracing](11-cache-tiering-tracing.md)** — why a 0.95 cosine
  cache serves the wrong answer on this corpus, the structural guard, two model tiers, and
  Langfuse traces with real cost.

### Part 3 — Run it, prove it, ship it
- **[12 — Run it end to end](12-run-it.md)** — `make up` / `ingest` / `serve` / `ui` and a
  guided first session with the multi-hop question.
- **[13 — Testing and evaluation](13-testing-and-evaluation.md)** — 473 pytest tests, the
  Promptfoo suite (35/37), the RAGAS ablation table, the safety scorecard (two successful
  attacks, reported honestly), and the load test.
- **[14 — Deployment](14-deployment.md)** — one Docker image with Neo4j embedded, the
  four-step `start.sh`, the stale-secret incident, and the live Hugging Face Space.

### For instructors
- **[15 — Teacher notes](15-teacher-notes.md)** — schedule, demo tips, the debugging stories
  students remember, common mistakes, and exercises.

---

## The finished system, in one picture

```
Streamlit UI ──► FastAPI POST /chat
                     │
               system card?  ── yes ──► canned answer (~1 ms, no model)
                     │ no
               NeMo input rails      mask PII · jailbreak check · topic check
                     │
               semantic cache        Qdrant `acme_cache`, cosine ≥ 0.97 + structural guard
                     │ miss
               LangGraph agent       knowledge_search · graph_query · web_search  (≤ 4 rounds)
                     │
               NeMo output rails     output policy · grounding · PII · citations
                     │
               answer + citations + trace_id  ──► every step recorded in Langfuse
```

Start here → **[00 — What we're building and why](00-what-and-why.md)**

---

## Companion documents

- [../../README.md](../../README.md) — the repository README: quickstart, deliverables map,
  deliberate omissions, known issues.
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — components, data flow, and the key design decisions.
- [../WALKTHROUGH.md](../WALKTHROUGH.md) — the eight-section project walkthrough with all
  the evaluation numbers in one place.
- [../screenshots/README.md](../screenshots/README.md) — which Langfuse screenshots to capture.
