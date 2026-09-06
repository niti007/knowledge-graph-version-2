# RAGAS — 30-question golden set and the retrieval ablation

The point of this directory is the **ablation table**. RAGAS is the instrument;
the question it answers is whether hybrid retrieval and cross-encoder re-ranking
actually earn their place over a plain dense baseline, and which cross-encoder
to keep.

## Headline

30 questions, judge `openai/gpt-4o-mini` via OpenRouter, four configurations.
**Zero NaN cells in 480 metric evaluations** — every number below is a real
judged score, not a hole averaged over.

| configuration | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| `vector_only` | 0.950 | 0.931 | 0.812 | 0.867 |
| `hybrid_no_rerank` | 0.950 | 0.935 | 0.841 | 0.839 |
| **`hybrid_rerank@bge`** (shipped) | 0.933 | **0.965** | **0.904** | **0.922** |
| `hybrid_rerank@minilm` | **0.967** | **0.965** | 0.836 | **0.922** |

Delta vs `vector_only`, in percentage points:

| configuration | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| `hybrid_no_rerank` | +0.0 | +0.4 | +2.9 | **−2.8** |
| `hybrid_rerank@bge` | −1.7 | +3.4 | **+9.2** | **+5.5** |
| `hybrid_rerank@minilm` | +1.7 | +3.4 | +2.4 | **+5.5** |

### What the ablation actually shows

**Hybrid + re-rank beats vector-only.** +9.2pp context precision and +5.5pp
context recall over the dense baseline, with answer relevancy up 3.4pp. The
target for this phase was faithfulness ≥ 0.85 and hybrid > vector-only; both
hold (0.933, and hybrid ahead on 3 of 4 metrics).

**Fusion alone does not.** `hybrid_no_rerank` is the interesting row: adding the
graph branch and RRF-fusing it moves context precision +2.9pp but moves context
recall **−2.8pp** — worse than doing nothing. Fusing two branches into a fixed
budget of five contexts displaces good dense hits with graph facts that are only
sometimes relevant; without a re-ranker to arbitrate, that is a net loss. **The
re-ranker is not a polish step on top of fusion — it is what makes fusion pay.**
That is the single most useful thing in this table, and it is the opposite of
the "just add a graph branch" intuition.

**Faithfulness is flat, and that is expected.** All four configurations sit
between 0.93 and 0.97. Faithfulness asks whether the answer is entailed by the
context it was given, so a better retriever changes *which* context, not whether
the generator hallucinated against it. Retrieval quality shows up in the two
context metrics, and it does.

### Per-category (n in parentheses)

| category | vector_only ctx-prec | hybrid_rerank@bge ctx-prec | vector_only ctx-recall | hybrid_rerank@bge ctx-recall |
|---|---|---|---|---|
| factual (9) | 0.763 | **0.915** | 0.889 | **0.963** |
| policy (7) | 0.941 | **0.972** | 0.857 | **1.000** |
| sop (5) | 0.929 | **1.000** | 1.000 | 1.000 |
| incident (4) | **0.972** | 0.876 | 1.000 | 1.000 |
| multi_hop (4) | 0.342 | **0.646** | 0.500 | 0.500 |
| aggregation (1) | 1.000 | 1.000 | 1.000 | 1.000 |

Multi-hop is where the baseline collapses — 0.342 context precision — and where
the full system nearly doubles it. It is also where the system is weakest in
absolute terms, which matches the Promptfoo result: the multi-hop path is the
one that still has a real defect in it (see `../promptfoo/README.md`, M7).

`incident` is the one category where vector-only wins on context precision
(0.972 vs 0.876). Incident questions name their document id in the question
("What caused INC-205?"), so the dense retriever lands on the right chunk
immediately, and the graph branch adds ownership facts that RAGAS correctly
scores as not-useful-for-this-question. It is a real, explainable regression on
one category of four questions, not noise to wave at.

---

## Settling the deferred re-ranker decision

Phase 3 compared `BAAI/bge-reranker-base` (96% top-1, 664ms) against
`cross-encoder/ms-marco-MiniLM-L-6-v2` (92% top-1, 131ms) on 25 probes, called a
one-probe difference noise, and deferred. Here is the larger sample.

| | bge-reranker-base | ms-marco-MiniLM-L-6-v2 |
|---|---|---|
| faithfulness | 0.933 | **0.967** |
| answer relevancy | 0.965 | 0.965 |
| **context precision** | **0.904** | 0.836 |
| **context recall** | 0.922 | 0.922 |
| retrieval p50 (this box) | 1401 ms | **232 ms** |

Multi-hop subset only (**n = 4** — read with that in mind):

| | bge | minilm |
|---|---|---|
| context precision | **0.646** | 0.414 |
| context recall | 0.500 | **0.750** |
| faithfulness | 0.750 | **0.875** |

Per question, on the four multi-hop items, the two models trade wins rather than
one dominating: bge takes context precision on Q27 (1.000 vs 0.700) and Q28
(1.000 vs 0.756); minilm takes context recall on Q27 (1.000 vs 0.000) — Q27 is
"which systems break if Auth-DB goes down", the cascade question, and minilm is
the model that surfaced the graph cascade fact. Q29 defeats both (recall 0.000
each): "besides Auth-DB, which system does DataWarehouse depend on" needs the
`DEPENDS_ON(DataWarehouse, ReportingPortal)` edge and neither re-ranker floats
it into the top five.

### Recommendation: switch to `cross-encoder/ms-marco-MiniLM-L-6-v2`

The honest reading of this data is that **the two re-rankers are not
distinguishable on answer quality at n=30**, so the decision falls to latency.

- They tie exactly on context recall (0.922) and answer relevancy (0.965).
- They split the remaining two in opposite directions and by comparable
  magnitudes: bge +6.8pp context precision, minilm +3.4pp faithfulness.
- Crucially, bge's context-precision lead **does not propagate**. If those
  contexts were meaningfully better, the answers built from them would be more
  faithful and more relevant. They are not — minilm's are marginally more
  faithful. Context precision here is measuring a re-ordering the generator
  turns out to be insensitive to at k=5.
- On multi-hop, the category Phase 3 was actually worried about, the sample is
  four questions and the two models trade wins. That does not resolve anything
  in bge's favour; it confirms Phase 3's instinct that a one-probe difference
  was noise.
- Latency is not noise. 1401 ms vs 232 ms p50, measured over 30 queries on the
  same box in the same process. The re-ranker is the largest single item in the
  retrieval path, and this removes ~1.17 s from every cache-cold request.

So: no measurable answer-quality cost, a ~6x faster re-ranker, and `reranker_model`
is already a setting — the change is one line in `app/config.py`.

**The caveats, stated plainly.** n=30 overall and n=4 on multi-hop. A 6.8pp
context-precision difference on 30 questions is not a result I would defend as
significant, and neither is a 3.4pp faithfulness difference. What this run
supports is the *negative* claim — that no quality difference large enough to
justify 1.17 s per query has been demonstrated — and that is enough to decide,
because the burden is on the slower model. If the golden set later grows past
~100 questions with a proper multi-hop block, this is worth re-running before
treating it as settled.

**This recommendation is not applied.** `app/config.py` still ships
`BAAI/bge-reranker-base`, and every number above was produced with the
configuration as it stands. Phase 8 measures the system; changing it is a
separate, reviewable decision.

---

## How to reproduce

Two venvs, because RAGAS 0.2.15 pins the langchain 0.3 line while the app runs
on langchain-core 1.6.1 / langgraph 1.2.11. Installing RAGAS into `.venv`
silently downgrades the agent's runtime, so it gets its own.

```bash
# one-time: the isolated RAGAS environment
PATH="$HOME/.local/bin:$PATH" uv venv .venv-ragas --python 3.11
PATH="$HOME/.local/bin:$PATH" VIRTUAL_ENV=$PWD/.venv-ragas uv pip install \
  "ragas==0.2.15" "langchain-openai<0.4" "langchain-community<0.4" \
  "datasets<4" langchain-huggingface sentence-transformers

# step 1 - collect contexts + answers, MAIN venv (needs the app's retrieval stack)
PYTHONPATH=. .venv/bin/python -m evals.ragas.collect          # ~3 min, writes evals/ragas/data/*.jsonl

# step 2 - score them, RAGAS venv
cd evals/ragas
set -a && . ../../.env && set +a
../../.venv-ragas/bin/python run_ragas.py                     # ~12 min, writes results/
../../.venv-ragas/bin/python report.py                        # re-render tables, no LLM calls
```

Neo4j and Qdrant must be up for step 1; step 2 touches neither.

### Design notes

**Why a single pinned generator, not the agent.** `collect.py` answers every
question with `gpt-4o-mini` from the retrieved context alone, using one prompt,
across all four configurations. The only variable between configurations is what
retrieval put in front of the model — which is what makes this an ablation. Run
through the full LangGraph agent instead and tool-choice, multi-turn reasoning
and the guardrails would paper over a weak retriever; the table would measure
the agent, not the retrieval mode. `pick_model(SYNTHESIS)` would also route to
gpt-4o, so the generator is pinned explicitly rather than inherited.

**Only answerable questions.** All four RAGAS metrics assume a retrievable
answer exists — context recall against an empty gold context is undefined, not
zero. Abstention is measured in the Promptfoo refusal category (5/5) and in the
Phase 9 scorecard, not here.

**Ground truths.** Every `ground_truth` in `golden_set.py` carries an `evidence`
field naming the `doc_id` or the Cypher relation it came from, so a disputed
answer key is one grep away from being re-checked.

### RAGAS and OpenRouter

RAGAS did **not** fight the OpenAI-compatible base URL. It accepts any LangChain
`BaseChatModel` through `LangchainLLMWrapper`, so a `ChatOpenAI` pointed at
`https://openrouter.ai/api/v1` works with no patching. Two things did need
handling, both reported rather than worked around:

1. **Version pinning.** `ragas` resolves to 0.4.3 by default, which imports
   `langchain_community.chat_models.vertexai` — a module removed in
   langchain-community 0.4. Pinning `ragas==0.2.15` with `langchain-community<0.4`
   fixes it. The exact pins are in the install command above.
2. **Embeddings.** `answer_relevancy` needs an embedding model and OpenRouter
   serves no embeddings endpoint. Rather than add a second provider, this uses
   the **local `BAAI/bge-small-en-v1.5`** — the same model the corpus was
   indexed with, so relevancy is measured in the same vector space the retriever
   operates in. Note this for comparability: these `answer_relevancy` numbers are
   not directly comparable to published figures computed with OpenAI embeddings.

`RunConfig(max_workers=4, timeout=180, max_retries=5)` is deliberate. OpenRouter
rate-limits harder than the OpenAI endpoint RAGAS' defaults assume, and a
throttled judge returns NaN — which pandas averages away, quietly making a
broken run look like a good one. `run_ragas.py` reports `nan_cells` per
configuration for exactly that reason. It was 0 for all four.

## Runtime and cost

| stage | wall clock |
|---|---|
| `collect.py`, all four configurations (120 generations) | ~3 min |
| `run_ragas.py`, 480 judged metric evaluations | **11 min 56 s** |
| per configuration | 101 s – 243 s |

**Cost, estimated:** the run was not metered against a before/after balance
snapshot, so this is an estimate from payload size, not a billing figure. The
120 collected rows carry ~126k tokens of context, answer and reference text.
RAGAS re-reads that per metric, and `context_precision` issues one call per
context (5 per row), which dominates: roughly 1.0–1.1M input and ~150k output
tokens across the four configurations. At gpt-4o-mini's $0.15/1M input and
$0.60/1M output that is **≈ $0.25**, plus **≈ $0.07** for the 120 generations —
call it **$0.30–0.35 for the whole RAGAS deliverable**. The Promptfoo suite adds
under $0.10 (see its README). Phase 8 total is comfortably under $0.50.
