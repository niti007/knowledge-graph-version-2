# Chapter 15 — Teacher Notes

This chapter is for the instructor: a schedule, the questions students always ask, demo
tips, the debugging stories that make the project memorable, the mistakes students make,
and exercises with answers.

---

## Suggested schedule

Each chapter is roughly one session. Sessions 4–9 are the technical core.

| Session | Chapters | Focus |
|---|---|---|
| 1 | 00 | The problem and the architecture. Show the live demo first. |
| 2 | 01, 02 | Everyone installs tools, gets keys, and passes `make check`. |
| 3 | 03, 04 | Settings; loading and canonicalizing the corpus. |
| 4 | 05 | Chunking and embeddings — the truncation bug is the lesson. |
| 5 | 06 | The knowledge graph and the eight templates. |
| 6 | 07 | Hybrid retrieval, RRF, and "relevance is not answerability". |
| 7 | 08 | The agent — structural bound and code-built citations. |
| 8 | 09 | Guardrails — the most content-dense session; allow extra time. |
| 9 | 10, 11 | API, UI, cache, tiering, tracing. |
| 10 | 12 | Everyone runs the full stack and the seven-question session. |
| 11 | 13 | Evaluation — read the scorecard together, including the two successes. |
| 12 | 14 | Deployment and wrap-up. |

> Tip: end Session 1 by asking the live Space the multi-hop question and opening its
> Langfuse trace. Students learn faster when they have seen where they are going.

---

## Questions students always ask

**"Why not just use ChatGPT?"**
It does not know ACME's documents and will invent a team lead. This system answers only
from the corpus, cites the chunk or graph path, and says "I don't know" when the corpus
does not contain the answer — and there are 60 adversarial probes proving where that holds
and where it does not.

**"Why both a vector database and a graph?"**
Vector search finds text *about* something; the graph follows *links* and can say
*nothing*. The RAGAS table (Chapter 13) is the evidence: hybrid + re-rank beats vector-only
by +9.2pp precision. Fusion *without* the re-ranker is worse than vector-only — a useful
surprise to put on the board.

**"Why is the first answer so slow?"**
Locally, warmup runs at startup so the first request is *not* slow — that was measured
(19.6 s vs 4.5 s) and fixed in Chapter 10. On the Space after 48 idle hours, the ~41 s cold
start is Neo4j + graph rebuild + model warmup.

**"Why does a cache hit still take 3 seconds?"**
The cache sits after the input rails and before the agent. It skips the agent (~2.4 s) but
still pays three rail LLM calls (~2.3 s). That is the architectural floor of caching behind
the rails.

**"Is it safe?"**
Seven rails, 47/60 attacks blocked at the door, 0 of 46 employee emails leaked across 102
responses, 5/5 lookalikes allowed. And two attacks succeeded — one produced a
confidently-cited false claim. Show students the scorecard's explanation of why every rail
passed it. That paragraph is the most educational thing in the repository.

**"Why is PERSON not masked?"**
Because "who leads Infrastructure?" is answered by a name. Read the scope statement in
`actions.py` together — it is a worked example of a documented risk acceptance with
compensating controls.

**"Why are two Promptfoo tests failing?"**
Because they are real defects with one root cause (empty `graph_query` result ends the
search instead of falling back). Fixing the tests would hide the defect. This is the single
best discussion prompt in the course: what is the difference between fixing a test and
loosening it?

---

## Live-demo tips

- Warm the API before class (`make serve`, wait for `warmup finished`).
- Ask a simple question first ("Who owns Payment-Service?"), then the multi-hop one, and
  point at `tools_used` and the graph citations.
- Open the Langfuse trace and expand `chat > guardrails > agent`. Point at the per-call
  cost on each `generation`.
- Run the attack and its lookalike back to back (Chapter 12, questions 2 and 3). The
  "ignore the *runbook*" vs "ignore *your instructions*" distinction lands every time.
- Ask the same question twice and show `cached: true` and the identical citations.
- Ask "Which systems does Auth-DB depend on?" then "Which systems depend on Auth-DB?" and
  show `guard: argument_order_differs` in the second trace's `cache.lookup`.
- Open Neo4j Browser (http://localhost:7475) and run
  `MATCH (n)-[r]->(m) RETURN n,r,m LIMIT 100` to show the graph as a picture.
- Have `evals/safety/scorecard.md` open in a tab. Read `deep-gnd-05` aloud.

---

## The debugging stories (students remember these)

1. **The silent truncation.** A word-count estimate undershot by 5.75× on URL-bearing
   lines; nine chunks exceeded 512 tokens; the embedder truncated silently; the Security
   roster and INC-204's root cause were never indexed while the pipeline printed "Done".
   *Lesson:* measure with the real instrument, and make silent failures loud.
2. **The wrong-direction template.** `dependency_cascade` returned Auth-DB for "what does
   Payment-Service depend on" — right by coincidence, because the two depend on each other.
   Notification-Service was silently dropped. *Lesson:* a plausible wrong answer is
   invisible to every check; you need a query that answers the question actually asked.
3. **The ledger in a module dict.** 299 tests passed. Two concurrent requests, and one was
   served the other's citations. *Lesson:* tests that never overlap cannot find
   concurrency bugs; per-request state belongs in a `ContextVar`.
4. **The invisible email addresses.** Presidio's email recognizer requires a public TLD, so
   all 46 `@acme.internal` addresses were invisible at any threshold. The "defence in
   depth" was one layer. *Lesson:* test the guardrail on the data it is supposed to protect.
5. **The 187-character answer.** A 200-character short-circuit in the refusal detector
   stripped citations from the canonical correct answer and kept them for the same answer
   plus 14 characters. *Lesson:* a boundary documented in a test suite but never tested
   below is not a rule.
6. **The cache that scored opposites at 0.9929.** *Lesson:* a bi-encoder measures
   similarity, not equivalence; the fix was structural, and its cost was written down.
7. **The stale secret.** A reused Space carried an Aura URL that overrode the Dockerfile.
   *Lesson:* own your invariants in the container; log when you override.

---

## Common student mistakes

| Mistake | What they see | Fix |
|---|---|---|
| Changing `NEO4J_PASSWORD` after first `make up` | "unauthorized" in `make check` | `docker compose down -v && make up` |
| Running `make serve` before `make ingest` | `/health` 503 `degraded`, "collection missing" | run `make ingest` |
| Using `embed_passages` for a query in an exercise | recall silently worse, no error | use `embed_query`; point at `test_cache_embeds_with_embed_query_not_passages` |
| Adding a system alias with a substring match | "gateway drug" resolves to APIGateway | whole-token matching (`_contains_phrase`) |
| Calling `Guardrails.run()` inside FastAPI | `RuntimeError` (by design) | `await arun()` |
| Committing `.env` | (hopefully) nothing, because `.gitignore` | rotate the key anyway |
| Editing a Promptfoo assertion until it passes | 37/37 | ask: was the test wrong, or the system? Show the G4 note in `evals/promptfoo/README.md` |
| Expecting `agent_ran` to mean "not cached" | `agent_ran: true` on hits | use `cached`; the load report notes this |

---

## Exercises

**E1 — Add a Cypher template.** Chapter 13's failing case M7 needs an `incident_system`
template (incident → systems). Add the Cypher to `GraphQueries`, the name to `TEMPLATES`,
`GRAPH_TEMPLATES` and the `TemplateName` literal, and a renderer in `retrieval/graph.py`.
`test_template_registry_matches_the_methods_that_exist` will tell you what you missed.
Re-run the Promptfoo suite. Does M7 pass now? (Answer: only if the agent *uses* it — this
is a good moment to discuss the fallback root cause.)

**E2 — The agent fallback.** In `agent/tools.py`, when `graph_query` returns
`NO_RESULTS` for a *resolved* entity, what should the prompt or the loop do differently?
Prototype a prompt-level fix first, then discuss why a prompt is weaker than a structural
one.

**E3 — Add a document.** Drop a new Markdown incident (`INC-206.md`) into `data/raw/` with a
row in `doc_metadata.csv`, run `make ingest`, and ask about it. Then *edit* it and re-ingest:
verify the old chunk was purged (`purged 1 orphaned points`).

**E4 — Break the chunker on purpose.** Set `CHUNK_SIZE=600` in `.env` and run ingest.
Which error do you get, and from which line? Then set it to `300` and explain why 700 was
never safe for this model.

**E5 — A benign lookalike.** Write a question containing "disregard" that should *pass* the
jailbreak rail, and one that should *block*. Add both to `evals/safety/benign_probes.py` /
`attacks.py` and re-run `make eval`.

**E6 — The cache guard.** Find a genuine paraphrase pair that `guard_verdict` wrongly
refuses (`argument_order_differs`) and one inversion it correctly refuses. Propose a
refinement and write the test that would catch the regression.

**E7 — The entailment check.** Design (do not necessarily build) the claim-level
entailment rail that would have caught `deep-gnd-05`. Where in `output.co` does it go,
which tier does it run on, and what does it cost per request?

**E8 — Swap the re-ranker.** Set `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
and re-run `evals/retrieval_probe.py`. Compare latency and top-1. Then read the RAGAS
report's re-ranker section and argue for or against switching.

---

## Where to go deeper

- `docs/WALKTHROUGH.md` — the eight-section walkthrough with every number in one place.
- `docs/ARCHITECTURE.md` — the diagram and the design-decision table.
- `evals/safety/scorecard.md`, `evals/ragas/report.md`, `evals/load/report.md`,
  `evals/promptfoo/README.md` — the four evaluation reports.
- `git log` — every commit message is a phase report with its numbers and reasoning.

## Wrap-up

Students who finish this course have built a hybrid RAG system with a knowledge graph, a
bounded agent, seven guardrails, a measured cache, tracing, five evaluation instruments,
and a live deployment — and, more unusually, they have seen what it looks like to report a
result honestly when the honest result is worse.

← Back to the course index: [README.md](README.md)
