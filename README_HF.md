---
title: Enterprise Knowledge Assistant
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Hybrid vector+graph RAG with guardrails over enterprise docs
---

# Enterprise Knowledge Assistant

An agentic RAG assistant over a synthetic enterprise corpus (policies, SOPs,
incident reports, system manuals, FAQ). Retrieval is hybrid: dense vectors in
Qdrant Cloud fused with a Neo4j knowledge graph (systems, teams, leads,
dependencies), re-ranked, then answered by an LLM through OpenRouter. Every
turn passes NeMo Guardrails input/output rails (PII masking, jailbreak and
topic checks, a structural grounding check, citation verification) and is
traced to Langfuse.

Try a multi-hop question the graph is needed for:

> Which team owns the system Payment-Service depends on for authentication, and who leads that team?

## What runs in this Space

One container, four steps (`start.sh`), each timed in the Space logs:

1. **Neo4j Community 5.26** starts *inside* the container, loopback only, no auth
   (nothing outside the container can reach it).
2. The **knowledge graph** (98 nodes / 236 relationships) is rebuilt from the
   bundled corpus. It is deterministic, so ephemeral storage is fine.
3. **FastAPI** starts on an internal port and warms the embedding, reranker and
   guardrail models (pre-downloaded into the image; no Hub traffic on boot).
4. **Streamlit** serves on port 7860 and talks to the API over localhost.

Qdrant (vectors + semantic cache), Langfuse (traces), OpenRouter (LLM) and
Tavily (web search) are cloud services, reached with the Space's secrets.

## Honest operational notes

- **Cold start is about 35-40 seconds** on the free CPU tier (measured locally
  on a 2-core container: Neo4j ~8s, graph rebuild ~14s, API warmup ~12s). Until
  then the UI may show "Not ready". Refresh once the logs print `[4/4]`.
- **Free Spaces sleep after 48 hours without traffic.** The first visit after
  that pays the cold start again; the graph is rebuilt, the vector index is not
  (it lives in Qdrant Cloud and is unaffected).
- Each answer costs real LLM tokens on the owner's OpenRouter account; the
  semantic cache (`cached: true` in the API response) absorbs exact and
  paraphrased repeats.
- Neo4j data is not persisted across restarts by design; nothing is lost
  because the graph is derived entirely from `data/raw`.

## Required secrets (Settings -> Variables and secrets)

| Secret | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | LLM calls (answers and the rails' own classifier) |
| `QDRANT_URL`, `QDRANT_API_KEY` | vector index + semantic cache (must already hold the `acme_docs` collection) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | tracing (optional; set `LANGFUSE_ENABLED=false` to disable) |
| `TAVILY_API_KEY` | web-search tool (optional; the tool reports itself unavailable without it) |

No secret is baked into the image; the build fails if a `.env` file is present.
