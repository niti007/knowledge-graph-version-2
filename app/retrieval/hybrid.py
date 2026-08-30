"""Hybrid retrieval: vector + graph, fused by RRF, re-ranked by a cross-encoder.

Three modes share one interface and one return type so Phase 8 can A/B them
from an eval script without special-casing:

    VECTOR_ONLY       dense top-k, no graph, no re-rank        (the baseline)
    HYBRID_NO_RERANK  vector + graph, RRF fused                (isolates fusion)
    HYBRID_RERANK     the above, re-ranked to rerank_top_n     (the full system)

Every returned item carries honest provenance -- which branch produced it, its
rank inside that branch, its fused score, and its re-rank score if one was
computed. Phase 4 builds citations from these fields, so a value that was never
computed is None rather than a plausible-looking number.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.retrieval import vector as vector_branch
from app.retrieval.graph import GraphContext, retrieve_graph_context
from app.retrieval.graph_queries import GraphQueries


class Mode(str, Enum):
    VECTOR_ONLY = "vector_only"
    HYBRID_NO_RERANK = "hybrid_no_rerank"
    HYBRID_RERANK = "hybrid_rerank"


ALL_MODES = (Mode.VECTOR_ONLY, Mode.HYBRID_NO_RERANK, Mode.HYBRID_RERANK)


@dataclass
class RetrievedItem:
    """One retrieved unit of context, with full provenance."""

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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citation"] = self.citation
        return d


@dataclass
class RetrievalResult:
    query: str
    mode: str
    items: list[RetrievedItem]
    timings_ms: dict[str, float] = field(default_factory=dict)
    graph_declined: bool = False
    graph_reason: str = ""
    resolved_entities: dict = field(default_factory=dict)
    n_vector: int = 0
    n_graph: int = 0
    # Structured abstention evidence, captured BEFORE re-ranking. The
    # cross-encoder scores topical relevance, not answerability, and will drop a
    # correct negative graph fact in favour of an on-topic non-answer -- it
    # scores "what depends on DataWarehouse" at 0.9944 when nothing does. These
    # signals are what Phase 5's grounding rail needs and rerank_score alone
    # cannot provide: no threshold over a relevance score recovers answerability.
    abstention: dict = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        """Ordered, de-duplicated source documents backing this result."""
        out: list[str] = []
        for it in self.items:
            for d in ([it.doc_id] if it.doc_id else it.metadata.get("evidence_docs", [])):
                if d and d not in out:
                    out.append(d)
        return out

    def context_text(self) -> str:
        return "\n\n".join(f"[{i.citation}]\n{i.text}" for i in self.items)


# --------------------------------------------------------------------- RRF

def reciprocal_rank_fusion(branches: dict[str, list[RetrievedItem]],
                           k: int) -> list[RetrievedItem]:
    """Standard RRF: score(d) = sum over branches of 1 / (k + rank_in_branch).

    Ranks are 1-based. Items are identified across branches by their citation,
    so the same chunk surfacing in two branches accumulates both contributions
    instead of appearing twice.
    """
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


# ------------------------------------------------------------------ engine

class HybridRetriever:
    """Reusable retriever. Holds the Qdrant client, Neo4j driver and models."""

    def __init__(self, settings: Settings | None = None,
                 client: QdrantClient | None = None,
                 graph_queries: GraphQueries | None = None):
        self.settings = settings or get_settings()
        self._client = client
        self._gq = graph_queries
        self._owns_client = client is None
        self._owns_gq = graph_queries is None

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            from app.ingestion.vector_index import get_client
            self._client = get_client(self.settings)
        return self._client

    @property
    def graph_queries(self) -> GraphQueries:
        if self._gq is None:
            self._gq = GraphQueries(settings=self.settings)
        return self._gq

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None
        if self._gq is not None and self._owns_gq:
            self._gq.close()
            self._gq = None

    def __enter__(self) -> "HybridRetriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ branches
    def _vector_items(self, query: str, top_k: int) -> list[RetrievedItem]:
        hits = vector_branch.search(query, top_k, settings=self.settings,
                                    client=self.client)
        return [
            RetrievedItem(
                text=h.text, branch="vector", source_type="chunk",
                doc_id=h.doc_id, chunk_id=h.chunk_id,
                branch_rank=rank, branch_score=h.score,
                metadata={k: v for k, v in h.payload.items() if k != "text"},
            )
            for rank, h in enumerate(hits, start=1)
        ]

    def _graph_items(self, query: str) -> tuple[list[RetrievedItem], GraphContext]:
        """Render the graph branch, demoting facts the query never asked for.

        A fact produced only because it is a default for the resolved entity is
        a guess. RRF rank 1 is the strongest endorsement the fuser can give, so
        a guess must not land there: an ownership fact was tying the best vector
        hit on "what is the connection pool ceiling for Payment-Service", a
        question about configuration that mentions no owner at all, and winning
        the tie-break. Uncued facts still enter fusion -- they can rise if both
        branches corroborate them -- but they start below `graph_uncued_rank_offset`
        so their solo RRF contribution can never reach a cued vector hit's.
        """
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
                metadata={"entity": fact.entity,
                          "evidence_docs": fact.evidence_docs,
                          "cued": fact.cued, "rows": fact.rows,
                          "negative": fact.negative},
            ))
        return items, ctx

    # -------------------------------------------------------------- rerank
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
        for pos, item in enumerate(ranked, start=1):
            item.final_rank = pos
        return ranked

    # ------------------------------------------------------------ retrieve
    def retrieve(self, query: str, mode: Mode | str = Mode.HYBRID_RERANK,
                 top_k: int | None = None, top_n: int | None = None
                 ) -> RetrievalResult:
        mode = Mode(mode)
        s = self.settings
        top_k = top_k or s.retrieval_top_k
        top_n = top_n or s.rerank_top_n
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        vector_items = self._vector_items(query, top_k)
        timings["vector_ms"] = (time.perf_counter() - t0) * 1000

        graph_items: list[RetrievedItem] = []
        ctx = GraphContext([], declined=True, reason="graph branch disabled in "
                                                     "vector_only mode")
        if mode is not Mode.VECTOR_ONLY:
            t0 = time.perf_counter()
            graph_items, ctx = self._graph_items(query)
            timings["graph_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if mode is Mode.VECTOR_ONLY:
            # No fusion: keep dense order, but populate rrf_score with the
            # single-branch RRF value so the field means the same thing in
            # every mode.
            items = vector_items
            for it in items:
                it.rrf_score = 1.0 / (s.rrf_k + it.branch_rank)
                it.final_rank = it.branch_rank
            items = items[:top_n]
        else:
            items = reciprocal_rank_fusion(
                {"vector": vector_items, "graph": graph_items}, s.rrf_k)
            if mode is Mode.HYBRID_NO_RERANK:
                items = items[:top_n]
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000

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

        timings["total_ms"] = (time.perf_counter() - t_start) * 1000
        return RetrievalResult(
            query=query, mode=mode.value, items=items, timings_ms=timings,
            graph_declined=ctx.declined, graph_reason=ctx.reason,
            resolved_entities={"system": ctx.resolved_system,
                               "team": ctx.resolved_team,
                               "incident": ctx.resolved_incident},
            n_vector=len(vector_items), n_graph=len(graph_items),
            abstention=abstention,
        )

    def ablate(self, query: str, modes: Iterable[Mode | str] = ALL_MODES
               ) -> dict[str, RetrievalResult]:
        """Run the same query through several modes. One call, same return type."""
        return {Mode(m).value: self.retrieve(query, m) for m in modes}


def retrieve(query: str, mode: Mode | str = Mode.HYBRID_RERANK,
             settings: Settings | None = None) -> RetrievalResult:
    """One-shot convenience wrapper. Prefer HybridRetriever to reuse clients."""
    with HybridRetriever(settings) as r:
        return r.retrieve(query, mode)
