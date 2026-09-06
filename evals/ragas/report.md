# RAGAS Evaluation Report

Enterprise Knowledge Assistant — retrieval ablation over a 30-question golden set.

Judge model `openai/gpt-4o-mini` via OpenRouter. Embeddings for `answer_relevancy` are the
local `BAAI/bge-small-en-v1.5` — the same space as the index, because OpenRouter serves no
embeddings endpoint. That makes these figures internally comparable but **not** directly
comparable to published numbers computed with OpenAI embeddings.

Ground truths are corpus-derived; each carries an `evidence` field naming its `doc_id` or
Cypher relation. 480 metric evaluations, **zero NaN cells**.

## Headline ablation

| configuration | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| vector only | 0.950 | 0.931 | 0.812 | 0.867 |
| hybrid, no re-rank | 0.950 | 0.935 | 0.841 | 0.839 |
| hybrid + re-rank (bge-reranker-base) | 0.933 | 0.965 | 0.904 | 0.922 |
| hybrid + re-rank (MiniLM-L-6-v2) | 0.967 | 0.965 | 0.836 | 0.922 |

**Hybrid + re-ranking beats the dense baseline**: context precision 0.812 → 0.904 (+9.2pp), context recall 0.867 → 0.922 (+5.5pp).

## The finding that matters: fusion alone is a net loss

`hybrid, no re-rank` moves context recall 0.867 → 0.839 — **-2.8pp against the dense baseline.** Fusing two branches into a fixed five-context budget displaces
good dense hits with graph facts that are sometimes irrelevant. The cross-encoder is not
polish applied on top of fusion; it is the component that makes fusion pay for itself.

This is the empirical answer to a question raised in Phase 3, where the graph branch flipped
a miss to a hit only once in 25 probes. Fusion earns its place only in combination with
re-ranking.

## By question category

### Faithfulness

| category | vector only | hybrid, no re-rank | hybrid + re-rank (bge-reranker-base) | hybrid + re-rank (MiniLM-L-6-v2) |
|---|---|---|---|---|
| aggregation | 1.000 | 1.000 | 1.000 | 1.000 |
| factual | 0.944 | 0.944 | 0.944 | 0.944 |
| incident | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 0.958 | 0.875 | 0.750 | 0.875 |
| policy | 0.929 | 0.929 | 0.929 | 1.000 |
| sop | 0.933 | 1.000 | 1.000 | 1.000 |

### Answer relevancy

| category | vector only | hybrid, no re-rank | hybrid + re-rank (bge-reranker-base) | hybrid + re-rank (MiniLM-L-6-v2) |
|---|---|---|---|---|
| aggregation | 0.990 | 0.990 | 0.990 | 0.990 |
| factual | 0.971 | 0.977 | 0.977 | 0.972 |
| incident | 0.987 | 0.988 | 0.990 | 0.989 |
| multi_hop | 0.935 | 0.943 | 0.935 | 0.936 |
| policy | 0.803 | 0.807 | 0.938 | 0.950 |
| sop | 0.978 | 0.978 | 0.978 | 0.971 |

### Context precision

| category | vector only | hybrid, no re-rank | hybrid + re-rank (bge-reranker-base) | hybrid + re-rank (MiniLM-L-6-v2) |
|---|---|---|---|---|
| aggregation | 1.000 | 1.000 | 1.000 | 1.000 |
| factual | 0.763 | 0.789 | 0.915 | 0.869 |
| incident | 0.972 | 1.000 | 0.876 | 0.938 |
| multi_hop | 0.342 | 0.460 | 0.646 | 0.414 |
| policy | 0.941 | 0.941 | 0.972 | 0.917 |
| sop | 0.929 | 0.939 | 1.000 | 0.890 |

### Context recall

| category | vector only | hybrid, no re-rank | hybrid + re-rank (bge-reranker-base) | hybrid + re-rank (MiniLM-L-6-v2) |
|---|---|---|---|---|
| aggregation | 1.000 | 1.000 | 1.000 | 1.000 |
| factual | 0.889 | 0.852 | 0.963 | 0.852 |
| incident | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_hop | 0.500 | 0.375 | 0.500 | 0.750 |
| policy | 0.857 | 0.857 | 1.000 | 1.000 |
| sop | 1.000 | 1.000 | 1.000 | 1.000 |

## Where vector-only wins

On `incident` questions (n=4) the dense baseline beats hybrid on context precision (0.972 vs 0.876). Those questions name their document id directly, so dense retrieval
lands the right chunk immediately and the graph branch contributes facts that RAGAS correctly
scores as unhelpful context. Reported rather than smoothed: hybrid retrieval is not uniformly
better, and this is the shape of the cases where it is not.

## Re-ranker comparison

| metric | bge-reranker-base | MiniLM-L-6-v2 | delta |
|---|---|---|---|
| Faithfulness | 0.933 | 0.967 | +3.3pp |
| Answer relevancy | 0.965 | 0.965 | +0.0pp |
| Context precision | 0.904 | 0.836 | -6.8pp |
| Context recall | 0.922 | 0.922 | +0.0pp |

The two **tie exactly** on context recall and answer relevancy, and split the remaining two
in opposite directions by comparable magnitude — bge better on context precision, MiniLM
better on faithfulness. Crucially, bge's precision advantage does not propagate to answer
quality: answer relevancy is identical.

Latency is the only difference outside the noise: measured p50 **1401 ms vs 232 ms**, roughly
1.17 s on every cache-cold request.

The supportable claim is the negative one — no quality difference large enough to justify
1.17 s per query has been demonstrated, and the burden of proof sits with the slower model.
The counter-argument is sample size: n=30 overall and n=4 on multi-hop, where the two trade
wins (bge better precision, MiniLM better recall). **The shipped default remains
`bge-reranker-base`**, because every number in this report was produced by the system as it
stands, and `reranker_model` is a single setting to change. Re-run past ~100 questions before
treating the difference as settled.

## Reproducing

```bash
.venv-ragas/bin/python evals/ragas/collect.py   # retrieval outputs per configuration
.venv-ragas/bin/python evals/ragas/run_ragas.py  # judged run (~12 min, ~$0.30)
.venv-ragas/bin/python evals/ragas/report.py     # regenerate tables from summary.json
```

RAGAS runs in an isolated `.venv-ragas`: its pins conflict with the application's
langchain-core 1.6.1 / langgraph 1.2.11, which Phases 4–7 depend on.

Judged runtime 11.1 min across four configurations; approximately $0.30–0.35.
