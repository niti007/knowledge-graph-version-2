# Chapter 07 — Hybrid Retrieval

## Why this / what's the need

We now have two ways to find things: vector search (Chapter 05) returns chunks that are
*about* the question; the graph (Chapter 06) returns precise facts about named entities.
Each is weak where the other is strong. Hybrid retrieval runs both, merges the two ranked
lists, and hands the best five pieces of context to the agent.

Three things in this chapter are non-obvious and were each learned by measurement:

1. Merging two lists needs a rule that does not depend on the lists' scores being
   comparable (a cosine similarity and "this came from a graph query" are not the same
   kind of number). That rule is **Reciprocal Rank Fusion**.
2. A graph fact that was produced *only because the entity was recognised*, not because the
   question asked for it, is a guess, and a guess must never be the top result.
3. The re-ranker measures **relevance, not answerability**. It rates "What depends on
   DataWarehouse?" at 0.9944 when the true answer is *nothing*. So the signal that the
   corpus *cannot* answer has to be captured *before* re-ranking discards it.

> 🔑 **New word — Reciprocal Rank Fusion (RRF):** merge ranked lists by giving each item
> `1 / (k + rank)` from every list it appears in and summing. Only positions matter, never
> raw scores.

> 🔑 **New word — ablation:** switching one component off to measure what it contributed.
> This module has three modes precisely so Chapter 13 can ablate it.

---

## Three modes, one interface — `app/retrieval/hybrid.py`

```python
class Mode(str, Enum):
    VECTOR_ONLY = "vector_only"
    HYBRID_NO_RERANK = "hybrid_no_rerank"
    HYBRID_RERANK = "hybrid_rerank"
```

```python
    VECTOR_ONLY       dense top-k, no graph, no re-rank        (the baseline)
    HYBRID_NO_RERANK  vector + graph, RRF fused                (isolates fusion)
    HYBRID_RERANK     the above, re-ranked to rerank_top_n     (the full system)
```

Every returned item carries its provenance:

```python
@dataclass
class RetrievedItem:
    text: str
    branch: str                     # "vector" | "graph"
    source_type: str                # "chunk" | "graph_fact"
    doc_id: str | None = None
    chunk_id: str | None = None
    graph_path: str | None = None
    graph_template: str | None = None
    branch_rank: int = 0            # 1-based rank within its own branch
    branch_score: float = 0.0       # cosine similarity, or graph cue rank score
    rrf_score: float = 0.0
    rerank_score: float | None = None   # None means "not re-ranked", never 0.0
    final_rank: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        """A citation string that names something that actually exists."""
        if self.source_type == "chunk":
            return f"{self.doc_id}#{self.chunk_id[:8]}" if self.chunk_id else str(self.doc_id)
        return f"graph:{self.graph_template}({self.graph_path})"
```

- `rerank_score: float | None` — `None` means the number was never computed. In
  `vector_only` mode it stays `None` rather than a plausible-looking 0.0. Chapter 08 builds
  citations from these fields, so honesty here is honesty in the citation.
- `citation` is the string a user sees: `SOP-01#8ab9023d` for a chunk,
  `graph:system_ownership(Auth-DB)` for a graph fact.

## RRF

```python
def reciprocal_rank_fusion(branches: dict[str, list[RetrievedItem]],
                           k: int) -> list[RetrievedItem]:
    """Standard RRF: score(d) = sum over branches of 1 / (k + rank_in_branch)."""
    fused: dict[str, RetrievedItem] = {}
    for items in branches.values():
        for item in items:
            key = item.citation
            contribution = 1.0 / (k + item.branch_rank)
            if key in fused:
                fused[key].rrf_score += contribution
            else:
                item.rrf_score = contribution
                fused[key] = item
    ranked = sorted(fused.values(), key=lambda i: (-i.rrf_score, i.citation))
    for pos, item in enumerate(ranked, start=1):
        item.final_rank = pos
    return ranked
```

- `k` is `rrf_k = 60` from the settings (the value from the original RRF paper). Rank 1
  contributes 1/61, rank 2 contributes 1/62 — a gentle curve, so an item ranked highly in
  both branches beats an item ranked first in one.
- Items are keyed by `citation`, so the same chunk appearing in both branches accumulates
  both contributions instead of appearing twice.
- The tie-break `i.citation` makes the order deterministic.

## The graph branch — `app/retrieval/graph.py`

The graph branch resolves entities in the query (using `resolve_system` and `resolve_team`
from Chapter 06, plus an `INC-nnn` regex) and renders 1–2 hop neighbourhoods as short English
facts. Which templates run depends on **cue words**:

```python
TEMPLATE_CUES: dict[str, tuple[str, ...]] = {
    "dependency_cascade": ("depend", "depends", "dependency", "dependencies",
                           "fails", "fail", "failure", "breaks", "break",
                           "outage", "down", "cascade", "impact", "affected"),
    "system_ownership": ("own", "owns", "owner", "owned", "responsible",
                         "maintains", "team"),
    # "inc" is deliberately NOT a cue: it matches the company suffix as a whole
    # word ("ACME Inc uses Auth-DB"), and a false cue sets cued=True, which is
    # precisely the flag that bypasses graph_uncued_rank_offset -- so a
    # speculative fact would reach rank 1 by the one route the demotion exists
    # to close. INC-nnn identifiers are matched separately by _INCIDENT_RE.
    "system_incidents": ("incident", "incidents", "outage", "postmortem",
                         "post-mortem"),
    ...
}

# Run these even with no keyword cue, so a bare entity mention is still useful.
DEFAULT_SYSTEM_TEMPLATES = ("system_ownership", "dependency_cascade")
```

- Cue matching is whole-word only (`_cue_score`), after "circuit **break**er threshold"
  fired the cascade template on a configuration question.
- A template can run for one of two reasons: the query *asked* for it (cued), or it is a
  default for a recognised entity (uncued). That distinction is carried on every fact:

```python
@dataclass
class GraphFact:
    text: str
    template: str
    path: str
    entity: str
    # True when the query actually asked for this template (a cue word matched).
    # False means it ran only because it is a default for the resolved entity,
    # which makes it speculative context rather than an answer.
    cued: bool = True
    evidence_docs: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    # True when this fact asserts an ABSENCE ("nothing depends on X"). ...
    negative: bool = False
```

And the cascade renderer says "nothing" out loud when the graph returns no rows:

```python
def _render_cascade(system: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        # Silence is meaningful here and the corpus has real cases of it
        # (nothing depends on DataWarehouse or UserProfile-API), so say so
        # rather than emitting nothing and letting the model assume.
        return GraphFact(
            f"No system in the corpus is recorded as depending on {system}, so "
            f"an outage of {system} has no recorded downstream cascade.",
            "dependency_cascade", f"(System)-[:DEPENDS_ON*]->({system})", system,
            negative=True)
```

## Uncued facts are demoted

Back in `hybrid.py`:

```python
    def _graph_items(self, query: str) -> tuple[list[RetrievedItem], GraphContext]:
        ctx = retrieve_graph_context(query, self.graph_queries, self.settings)
        offset = self.settings.graph_uncued_rank_offset
        items: list[RetrievedItem] = []
        n_cued = 0
        for fact in ctx.facts:
            if fact.cued:
                n_cued += 1
                rank = n_cued
            else:
                rank = offset + (len(items) - n_cued) + 1
            items.append(RetrievedItem(
                text=fact.text, branch="graph", source_type="graph_fact",
                graph_path=fact.path, graph_template=fact.template,
                branch_rank=rank,
                # The graph has no similarity score; use a decaying rank score so
                # the field is populated honestly rather than faked as a cosine.
                branch_score=1.0 / rank,
                ...
```

- A cued fact gets branch rank 1, 2, 3… An uncued fact starts at rank `offset + 1 = 11`.
  Its RRF contribution (1/71) can therefore never tie a vector hit at rank 1 (1/61). The
  docstring records the case that motivated it: an *ownership* fact was tying the best
  vector hit on "what is the connection pool ceiling for Payment-Service" — a question
  that never mentioned owners — and winning the tie-break.
- Uncued facts still enter fusion, so if both branches corroborate them they can rise.

## Re-ranking

```python
    def _rerank(self, query: str, items: list[RetrievedItem],
                top_n: int) -> list[RetrievedItem]:
        if not items:
            return items
        from app.ingestion.embedding_model import get_reranker

        model = get_reranker(self.settings)
        scores = model.predict([(query, i.text) for i in items])
        for item, score in zip(items, scores):
            item.rerank_score = float(score)
        ranked = sorted(items, key=lambda i: -i.rerank_score)[:top_n]
```

- The cross-encoder (`BAAI/bge-reranker-base`) reads `(query, text)` pairs together and
  scores each. The top `rerank_top_n = 5` survive.
- Measured on 25 probes in Phase 3: vector-only 88% top-1 at 47 ms; hybrid + re-rank 96%
  top-1 at 724 ms, of which the re-rank was 664 ms. The win was real but narrow — the graph
  branch flipped a miss to a hit once in 25 probes. It was kept for the reason below.

## Abstention signals: captured *before* re-rank

This is the chapter's key finding. The re-ranker scores topical relevance, and a correct
negative fact ("nothing depends on DataWarehouse") is *less* topically rich than an
on-topic chunk that does not answer the question, so re-ranking discards exactly the
evidence that says "the corpus cannot answer this". Phase 3 measured that the best
achievable accuracy of a threshold over the re-rank score alone was 90%, and that a
fabricated `INC-206` scored 0.9863.

So `retrieve()` snapshots the structural evidence first:

```python
        # Snapshot abstention evidence while the negative facts are still here.
        negatives = [i.text for i in items if i.metadata.get("negative")]
        entity_resolved = any(v for v in (ctx.resolved_system, ctx.resolved_team,
                                          ctx.resolved_incident))
        abstention = {
            # An entity WAS named and resolved, yet the graph returned nothing:
            # the corpus knows the thing and has no fact for the question asked.
            "entity_resolved_but_graph_empty": bool(ctx.declined and entity_resolved),
            "negative_facts": negatives,
            "graph_declined": ctx.declined,
        }

        if mode is Mode.HYBRID_RERANK:
            t0 = time.perf_counter()
            items = self._rerank(query, items, top_n)
            timings["rerank_ms"] = (time.perf_counter() - t0) * 1000
            abstention["negative_fact_survived_rerank"] = any(
                i.metadata.get("negative") for i in items)
        abstention["max_rerank_score"] = max(
            (i.rerank_score for i in items if i.rerank_score is not None), default=None)
```

- `entity_resolved_but_graph_empty` — the query named a real system/team/incident, the graph
  had nothing for the question. That is a fact about retrieval, not an estimate.
- `negative_facts` — explicit "none" statements from the graph.
- `max_rerank_score` — still reported, because at the *bottom* of its range it is a reliable
  off-topic detector (legitimate queries scored as low as 0.12; off-topic ceiling 0.072).

These three signals are what Chapter 09's grounding rail reads. The commit message states
the conclusion: *the graph is the only component that can say "the corpus does not contain
this", and re-ranking was discarding exactly that evidence. That is why the graph branch
stays in retrieval rather than moving behind the agent's tool.*

## Did it pay off? (preview of Chapter 13)

The RAGAS ablation on 30 questions:

| configuration | faithfulness | answer relevancy | context precision | context recall |
|---|---|---|---|---|
| vector only | 0.950 | 0.931 | 0.812 | 0.867 |
| hybrid, no re-rank | 0.950 | 0.935 | 0.841 | **0.839** |
| hybrid + bge re-rank | 0.933 | 0.965 | **0.904** | **0.922** |

Hybrid + re-rank beats the dense baseline by +9.2pp context precision and +5.5pp recall.
**Fusion without re-ranking loses 2.8pp of recall** — the graph facts displace good dense
hits in a fixed five-item budget. The re-ranker is not polish on top of fusion; it is what
makes fusion pay.

---

## ✅ You just learned
- Three ablatable modes over one interface, with honest `None` where nothing was computed.
- RRF: merge by rank, keyed by citation, `k = 60`.
- Cued vs uncued graph facts, and why an uncued fact starts at rank 11.
- Why abstention signals are captured before re-ranking, and what "relevance is not
  answerability" means in numbers.

## ▶️ Run this now
```bash
.venv/bin/python -c "
from app.retrieval.hybrid import HybridRetriever
with HybridRetriever() as r:
    for mode, res in r.ablate('What depends on DataWarehouse?').items():
        print(mode, res.abstention.get('negative_facts'), [i.citation for i in res.items][:3])
"
```
Note that the negative fact is present in `abstention` in both hybrid modes even when it
does not survive re-ranking into `items`.

Also run the Phase 3 probe harness: `.venv/bin/python -m evals.retrieval_probe`.

## 🧠 Check yourself
1. Why does RRF use ranks rather than the branches' own scores?
2. A query mentions Auth-DB but asks nothing about ownership. What branch rank does the
   ownership fact get, and why?
3. Why is a high re-rank score not evidence that the corpus can answer the question?
4. What single number in the RAGAS table justifies keeping the re-ranker?

---

Next: the agent that decides which tool to call →
[08-the-agent.md](08-the-agent.md)
