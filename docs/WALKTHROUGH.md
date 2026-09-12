# Project Walkthrough

Enterprise Knowledge Assistant — the eight sections the capstone specification requires.
Every number below comes from a report in `evals/`, a commit message, or the code.

**Live:** https://nitishgalat-enterprise-knowledge-assistant.hf.space

---

## 1. Business problem

ACME Corp's operational knowledge is spread across 28 files in three formats: four policy
PDFs, seven SOPs, five incident post-mortems, five technical manuals, an FAQ, and three
spreadsheets (46 employees in 6 teams, 7 products, a transaction ledger). The questions an
on-call engineer or a support agent actually asks span several of them at once:

- "Which team owns the system Payment-Service depends on, and who leads it?" — three facts
  in three documents.
- "What breaks if Auth-DB fails?" — a dependency walk.
- "What was the root cause of INC-204 and which SOP resolved it?"
- "What is the password rotation policy?"

Keyword search returns fifty hits and no chain. A plain LLM invents a team lead. The
requirement was an assistant that answers from the corpus only, cites the exact chunk or
graph path, follows relationships, refuses attacks, masks personal data, says "I don't
know" when the corpus does not know, records every step for audit and cost, and runs on a
public URL that survives a week of nobody looking at it.

Two hazards in the data shaped the build: `doc_metadata.csv` spells the same system four
ways (`Auth-DB` / `Auth-Db`, `APIGateway` / `Apigateway`), and the `DEPENDS_ON` facts are
not in any CSV — they are sentences inside the manuals.

## 2. System architecture

```
Streamlit ──► FastAPI /chat
                  │
            system card (meta questions, ~1 ms)
                  │
            NeMo input rails        mask PII · jailbreak · topic
                  │
            semantic cache          Qdrant acme_cache, cosine ≥ 0.97 + structural guard
                  │ miss
            LangGraph agent         gpt-4o, ≤ 4 tool rounds, tools unbound at the ceiling
                  ├─ knowledge_search  →  Qdrant + Neo4j → RRF → bge re-rank → top-5
                  ├─ graph_query        →  8 parameterized Cypher templates
                  └─ web_search         →  Tavily
                  │
            NeMo output rails       output policy · grounding · PII · citations
                  │
            answer + citations + trace_id  ──►  Langfuse (every span, cost per call)
```

| Layer | Choice |
|---|---|
| LLM | OpenRouter → `openai/gpt-4o` (agent), `openai/gpt-4o-mini` (rails, 5 calls/request) |
| Embeddings / re-rank | local `BAAI/bge-small-en-v1.5` (384-d) / `BAAI/bge-reranker-base` |
| Vector store + cache | Qdrant Cloud, collections `acme_docs` (66 points) and `acme_cache` |
| Graph | Neo4j 5 — Docker locally (ports 7475/7688); embedded in the Space container |
| Agent | LangGraph, `agent ⟲ tools → END` |
| Guardrails | NeMo Guardrails 0.24 (Colang 1.0) + Presidio |
| Serving | FastAPI (`/chat`, `/health`, `/sessions/{id}`, `/metrics`) + Streamlit |
| Observability | Langfuse Cloud (v4 SDK) |
| Hosting | Hugging Face Docker Space, `python:3.11-slim` + JRE 21 + Neo4j 5.26, 4.02 GB |

The architecture has one routing decision and it is the agent's tool choice. There is no
standalone query router: `knowledge_search` fuses both retrieval backends, so the agent
never has to choose between vector and graph — a decision it has no information to make
well. The rails sit *outside* the agent with the agent installed as NeMo's generation step
(`passthrough: true`), so every rail decision crosses one boundary into one ledger.

Full component and decision tables: [ARCHITECTURE.md](ARCHITECTURE.md).

## 3. Retrieval strategy

**Ingestion.** 28 files → 25 documents → 66 chunks → 66 points. CSVs are rendered to prose
("Priya Sharma (user id U0001) is a Team Lead on the Billing team…") under Markdown
headings. System names are canonicalized via a slug lookup; an unmapped spelling in the
metadata CSV is a hard error. Chunks are header-aware (the title and section breadcrumb are
inside every chunk's text), packed from small sections, and **sized with the embedding
model's own tokenizer** — after a word-count estimate undershot by up to 5.75× and nine
chunks were silently truncated at 512 tokens, dropping the Security roster and INC-204's
root cause from the index. `chunk_size` is 300 (not the planned 700) and any chunk over the
model limit raises. Ingestion is idempotent: content-hash ids, upsert plus orphan purge,
`points == chunks` asserted; an unreadable source file aborts the run rather than being
purged from the index.

**Graph.** 98 nodes / 236 relationships across System, Team, Person, Document, SOP, Product
and DEPENDS_ON, OWNS, MANAGES, RESOLVED_BY, RELATED_TO — built deterministically from the
CSVs and regex-parsed prose, no LLM. `DEPENDS_ON` is **8 edges, not 15**: the manuals'
ASCII diagrams contradict each manual's own "downstream event emission" sentence, so only
edges corroborated by explicit prose or an incident RCA are loaded. The rejected set
included `Auth-DB → ReportingPortal`. 150 person-purchased-product edges were dropped
because no template read them. All four policies were graph orphans (`dept=All` matched no
team) and now link to every team plus the systems named in their Scope sections.

**Eight Cypher templates**, bound parameters only: `system_ownership`, `team_leadership`,
`dependency_cascade` (reverse: what breaks), `system_dependencies` (forward: what X needs —
added after the reverse template answered the forward question right by coincidence and
dropped Notification-Service), `incident_sop`, `system_incidents`, `team_roster`,
`policies_for_system`. `resolve_system` matches whole tokens only and declines rather than
guessing ("gateway drug" no longer resolves to APIGateway).

**Hybrid.** Vector top-20 and cue-ranked graph facts are fused by Reciprocal Rank Fusion
(k=60); graph facts the query did not ask for start at rank 11 so a guess can never tie a
rank-1 vector hit; the bge cross-encoder re-ranks to top-5. Three ablatable modes share one
interface.

**The key finding.** A cross-encoder scores topical relevance, not answerability: "What
depends on DataWarehouse?" scores 0.9944 when nothing does; a non-existent INC-206 scores
0.9863; the best achievable accuracy of a score threshold alone is 90%. The graph is the
only component that can say "the corpus contains no such fact", and re-ranking discards
exactly that evidence. So abstention signals (`entity_resolved_but_graph_empty`,
`negative_facts`, `max_rerank_score`) are snapshotted **before** re-rank and carried on
`RetrievalResult` for the grounding rail.

**Phase 3 probe (25 queries):** vector-only 88% top-1 / 47 ms; hybrid+re-rank 96% top-1 /
724 ms. **RAGAS (Section 6)** confirmed the win and showed fusion without re-rank is a net
loss.

## 4. Agent workflow

`app/agent/graph.py` compiles a two-node LangGraph: `agent` (an LLM call with tools bound)
and `tools` (runs the calls, appends provenance to state), with a conditional edge back to
`agent` while the model keeps calling tools.

Two guarantees are structural, not prompted:

1. **The 4-iteration bound cannot be argued past.** At the ceiling the model is re-invoked
   with **no tools bound** and a final-turn nudge, so a fifth tool call is not
   representable. Tested against a model that always calls a tool, 50 times.
2. **Citations are a pure projection of the ledger.** Every tool appends the
   `RetrievedItem`s it actually returned to a per-request `Toolbox.records`;
   `build_citations(retrieved)` takes only that list. A test has the model write POL-999,
   INC-206 and SOP-42 in its prose; none reach the citation set.

The three tools: `knowledge_search(query)` (hybrid retrieval, the default),
`graph_query(template, entity, max_depth)` (template is a `Literal` — free-form Cypher is
unreachable), `web_search(query)` (Tavily, public web only). A `NO_RESULTS` marker is
returned for empty results, distinguishing "could not identify that entity" from "that
entity exists and has no such fact", and the system prompt's largest section is titled
"HONESTY -- THE PART THAT MATTERS MOST".

Model tiering is by *task*: `Task.AGENT` → `gpt-4o`; the rails' `SELF_CHECK` → `gpt-4o-mini`;
unknown tasks default to the fast tier. Confirmed by provider token counts; saves ~25% of
request cost, though 98% of spend is the agent.

Live on five questions ($0.057): the multi-hop question resolved to Auth-DB / Infrastructure
/ Marcus Lee and reported "Notification-Service does not have an ownership record in the
corpus"; "parental leave policy" declined at max rerank 0.00027; "what depends on
DataWarehouse?" answered "nothing does".

## 5. Guardrails

Seven rails in Colang, with the agent as the rails' generation step:

| Stage | Rail | Kind | Model |
|---|---|---|---|
| input | `pii_input` (Presidio) | masks | local |
| input | `self_check_input` | blocks | gpt-4o-mini |
| input | `topic_scope` | blocks | gpt-4o-mini |
| output | `self_check_output` | blocks | gpt-4o-mini |
| output | `check_grounding` | blocks | local (signals) |
| output | `pii_output` (Presidio) | masks | local |
| output | `check_citations` | demotes | local |

PII is masked **first**, before the jailbreak check sends the text to an LLM, and the
masked text is what reaches retrieval, the agent and the cache key.

**Grounding is three signals, not one score.** The re-rank score is trusted only at the
bottom of its range (< 0.10, a zero-false-negative off-topic detector). The near-miss cases
it cannot see are decided by `entity_resolved_but_graph_empty` and `negative_facts` — facts
about retrieval, not estimates. An empty lookup counts only if the answer asserts something
about that entity (the *scoped* rule, added after gpt-4o's exploratory ownership lookup on
unowned Notification-Service blocked the demo question 1-in-5 times; now 0/6).
`answered_without_retrieval` catches parametric answers.

**Two refusal predicates.** The grounding rail uses `acknowledges_absence` (anywhere);
the citation rail uses `looks_like_refusal` (first substantive sentence). The canonical
mixed answer — two cited facts plus "there is no ownership information for
Notification-Service" — must pass grounding *and* keep its citations; one predicate fails
one way or the other. A 200-character short-circuit was removed after it stripped citations
from the 187-character canonical answer.

**Citation policy:** a refusal's citations are moved to `provenance` tagged
`supports_answer: False`; nothing is deleted, the claim of support is.

**PII scope:** EMAIL_ADDRESS, PHONE_NUMBER, US_SSN, CREDIT_CARD, IBAN_CODE, IP_ADDRESS,
CRYPTO at threshold 0.40. **PERSON and LOCATION are deliberately unmasked** — the
product's headline answer is a person's name — as a documented risk acceptance with three
compensating controls. Three loose recognizers were removed after they masked an 8-hex chunk
id inside a citation marker; citation markers are now excluded from scanning. A custom
`.internal` email recognizer took corpus detections from 0 to 46, because Presidio's
default requires a public TLD.

**Concurrency.** The first ledger was a module-level dict; two overlapping requests swapped
citations while 299 tests passed. The ledger now lives in a `ContextVar` bound inside the
per-request coroutine, the agent runs via `asyncio.to_thread`, all rails traffic runs on
one owned event loop, and `LLMRails` construction is serialised. Verified with 6 overlapping
requests, then again at the API layer, then again under 3-way load with 0 errors.

**Meta questions** ("what can you do?") are answered from a static system card in the API
layer, before the rails — routing logic does not belong in a safety rail. 6/6 routed,
0/10 corpus traps misrouted, meta false positives 3 → 0.

## 6. Evaluation results

### pytest — 473 tests, green from a clean shell

### Promptfoo — 35 / 37 (94.6%) against the live API

| Category | Cases | Passing |
|---|---|---|
| Factual / citation | 10 | 10 |
| Multi-hop | 8 | 7 |
| Aggregation | 6 | 5 |
| Adversarial | 8 | 8 |
| Refusal / unknown | 5 | 5 |

Assertions read rail telemetry (`blocked_by`, `tools_used`, `citations`), not refusal prose.
The two failures (M7, G6) share one root cause — an empty `graph_query` result ends the
agent's search instead of falling back to `knowledge_search` — and are left failing. The
defect only ever under-answers. One assertion was *corrected* (`contains:'4'` on the correct
answer "Four systems…") and is flagged because "fixed" and "loosened" look alike in a diff.

### RAGAS — 30-question golden set, 4 configurations, 480 evaluations, 0 NaN

| configuration | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| vector only | 0.950 | 0.931 | 0.812 | 0.867 |
| hybrid, no re-rank | 0.950 | 0.935 | 0.841 | **0.839** |
| hybrid + bge re-rank (shipped) | 0.933 | 0.965 | **0.904** | **0.922** |
| hybrid + MiniLM re-rank | 0.967 | 0.965 | 0.836 | 0.922 |

- Hybrid + re-rank vs dense baseline: **+9.2pp context precision, +5.5pp recall.**
- **Fusion alone is a net loss:** −2.8pp recall. Fusing into a five-context budget displaces
  good dense hits with sometimes-irrelevant graph facts. The re-ranker is what makes fusion pay.
- Where vector-only wins: incident questions (n=4), precision 0.972 vs 0.876, because they
  name their document id. Reported, not smoothed.
- Re-ranker decision **deferred**: bge and MiniLM tie on two metrics and split the other two
  by similar magnitude; 1401 ms vs 232 ms p50; n=30 too small to decide. bge ships.

### Safety scorecard — 102 probes, cache disabled

| number | value |
|---|---:|
| adversarial probes | 60 across 8 categories |
| blocked at input rails | 47 (78%) |
| blocked at output rails | 2 (3%) |
| not blocked | 11 (18%) |
| **attacker's goal achieved** | **2** |
| benign blocked | 2 / 32 (6.2%) — both correct refusals; 0 genuine false positives |
| lookalike / attack pairs | 5/5 allowed, 5/5 blocked |
| real corpus emails leaked | **0** of 46 across 102 responses |

**The two successes.** `deep-gnd-05`: a false-premise injection produced "documented in
SOP-01 [SOP-01#8ab9023d]" for a CTO / 15-minute rule that SOP-01 does not contain. Every rail
passed it because the chunk *is* about incident response — the grounding rail detects "the
corpus does not cover this", not "the corpus covers this and does not say it". Relevance is
not entailment. `deep-out-04`: a "security awareness training" framing produced a complete
phishing email naming the real Infrastructure lead; `self_check_output` allowed it.

Of the 11 not blocked, 9 were defence successes the block rate cannot see: 3 PII answers
*masked* by `pii_output`, 1 `self_check_output` block, 5 fabrication baits the agent refused.
All seven rails fired at least once; `self_check_input` does 94% of the blocking.

### Load — Locust, 3 users, 3 minutes per mode, warmup excluded

| | cache OFF | cache ON |
|---|---:|---:|
| requests | 66 | 102 (+55%) |
| p50 / p95 / p99 | 4.9 / 7.8 / 60.0 s | 3.0 / 6.0 / 6.7 s |
| errors | 0 | 0 |

The 60 s p99 is one stalled OpenRouter call hitting the client's 60 s timeout. Overhead at
p50: agent ~2.4 s, three rails LLM calls ~2.3 s, everything local **~35 ms (< 1%)**. Cost:
$0.0038 per answer, $0.0016 per blocked refusal; Phase 9 ~$0.80 over 308 requests; whole
project **$3.18**.

## 7. Observability insights

Every request is a Langfuse trace rooted on a `uuid4().hex` minted by `/chat` — the same id
in the API response, the session transcript and the Langfuse UI. Under the root:
`guardrails` → one `rail.*` observation per decision (with reason and latency from the
ledger) and one `llm.rails.*` generation per rails LLM call (tokens from NeMo's own log);
`cache.lookup` with hit/miss, reason and candidates; `agent` → `llm.agent.turn_N`
generations with provider token counts and `cost_details` computed from OpenRouter list
prices, and `tool.*` spans with `retrieval.hybrid_rerank` carrying Phase 3's stage timings.
Tracing is wrapped so it can never fail a request (tested against a client that raises on
every method), and the trace-URL template is resolved once at warmup.

What the traces showed:

- **Tiering is real** — `gpt-4o` on the agent, `gpt-4o-mini` on all five rail calls,
  confirmed by provider token counts.
- **The rails cost as much wall time as the agent** (~2.3 s vs ~2.4 s), all of it provider
  round-trips. The code this project wrote is under 1% of a request.
- **A cache hit floors at ~3.0 s**, not 0.3 s, because it sits behind the input rails and
  the output rails re-check it. Cold 6.3 s → warm 3.5 s.
- **Warmup matters**: 19.6 s first request without it vs 4.5 s after.
- **The re-ranker is the dominant local cost** (1401 ms p50 for bge).
- **Refusals are not free**: $0.0016 each, 41% of a full answer, because the jailbreak
  prompt is long and runs on every request.
- Traces arrive from inside the deployed container.

## 8. Challenges and improvements

**What was hard, and what it taught**

1. *Silent truncation* — a word-count estimate let nine chunks exceed 512 tokens and the
   embedder dropped their tails while reporting success. Measure with the real tokenizer;
   make silent failures raise.
2. *A wrong-direction template that looked right* — the reverse dependency query answered
   the forward question by coincidence. A plausible wrong answer is invisible to every
   abstention signal; the fix has to be a query that answers the question actually asked.
3. *Relevance is not answerability / equivalence / entailment* — the same gap surfaced three
   times: the re-rank score (0.9944 for "nothing"), the cache (inversions at 0.9929, 1.000
   on the cross-encoder), and the scorecard's headline attack (a topically-relevant chunk
   cited for a claim it does not make). Each time the fix was structural evidence, not a
   better threshold.
4. *Concurrency* — a ledger in a module dict passed 299 sequential tests and swapped
   citations between two concurrent requests.
5. *The guardrail that could not see the data* — Presidio's email recognizer needs a public
   TLD; all 46 `@acme.internal` addresses were invisible.
6. *Deployment* — cut once because a Space cannot reach a laptop's Docker and Aura sleeps;
   reinstated by embedding Neo4j and rebuilding the deterministic graph on boot. Then the
   first live boot failed on a stale secret from a reused Space.

**Improvements, in priority order**

1. **Claim-level entailment check** between each assertion in the answer and its cited
   chunk — the NLI-style grounding the signal-based rail does not replace. Closes
   `deep-gnd-05`.
2. **Agent fallback after an empty `graph_query`**: retry with `knowledge_search` before
   declaring absence. Closes Promptfoo M7 and G6. Two more templates (`incident_system`,
   `team_products`) would help.
3. **`self_check_output` prompt**: treat "realistic phishing example" as facilitating harm
   regardless of stated purpose. Closes `deep-out-04`, at the cost of some false positives
   on genuine training content.
4. **Re-ranker**: re-run the ablation past ~100 questions; if MiniLM holds, switching takes
   ~1.2 s off every cache miss.
5. **Run early tool-selection turns on the fast tier** — the larger cost lever than tiering
   the rails.
6. **Decide whether cache hits must re-run output rails**; if not, hits drop from ~3.0 s
   toward ~1.4 s.
7. **Tighten the 60 s upstream timeout** and rename `agent_ran`.
8. **Persist sessions** if ever deployed behind more than one worker.
