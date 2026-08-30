"""Phase 3 retrieval tests: prefix correctness, RRF math, degradation, re-rank.

Vector tests run against an in-memory Qdrant so they need no network. Tests
that need the real graph skip when Neo4j is unreachable.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from qdrant_client import QdrantClient

from app.config import Settings, get_settings
from app.ingestion.chunker import Chunk, make_chunk_id
from app.ingestion.embedding_model import count_tokens
from app.ingestion import vector_index as vi
from app.retrieval import vector as vector_branch
from app.retrieval.graph import GraphContext, GraphFact, retrieve_graph_context
from app.retrieval.graph_queries import GraphQueries
from app.retrieval.hybrid import (
    ALL_MODES,
    HybridRetriever,
    Mode,
    RetrievedItem,
    reciprocal_rank_fusion,
    retrieve,
)

SETTINGS = get_settings()

CORPUS = [
    "Payment-Service is owned by the Billing team, led by Priya Sharma.",
    "Auth-DB is owned by the Infrastructure team, led by Marcus Lee.",
    "SOP-01 defines the incident response procedure for all engineers.",
    "The data retention policy requires records be kept for seven years.",
    "APIGateway routes all external traffic and depends on Auth-DB.",
]


@pytest.fixture(scope="module")
def mem():
    """In-memory Qdrant seeded with a tiny corpus, using the real embedder."""
    settings = Settings(qdrant_collection="test_retrieval")
    client = QdrantClient(":memory:")
    chunks = [
        Chunk(chunk_id=make_chunk_id("D", t), doc_id=f"doc{i}", text=t,
              heading="H", heading_path=["H"], section_headings=["H"],
              chunk_index=i, token_count=count_tokens(t),
              metadata={"doc_type": "sop", "dept": "All", "filename": f"doc{i}.md",
                        "system_refs": [], "sop_refs": [], "title": f"doc{i}"})
        for i, t in enumerate(CORPUS)
    ]
    vi.sync_chunks(chunks, settings, client)
    try:
        yield client, settings
    finally:
        client.close()


@pytest.fixture(scope="module")
def gq():
    try:
        q = GraphQueries(settings=SETTINGS)
        q.driver.verify_connectivity()
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Neo4j unavailable: {exc}")
    yield q
    q.close()


# ------------------------------------------------- BGE query-prefix correctness

def test_vector_search_uses_embed_query_not_embed_passages(mem):
    """MUTATION GUARD. BGE is asymmetric: passages take no prefix, queries take
    "Represent this sentence...". Swapping the two calls degrades recall
    silently -- nothing raises, results just get worse. Asserting the constants
    cannot catch that; asserting which function is CALLED can."""
    client, settings = mem
    with patch.object(vector_branch, "embed_query",
                      wraps=vector_branch.embed_query) as spy_q, \
         patch.object(vi, "embed_passages") as spy_p:
        vector_branch.search("who owns Auth-DB", 3, settings=settings, client=client)
    assert spy_q.call_count == 1, "the query must be embedded with embed_query"
    assert spy_p.call_count == 0, "a query must never be embedded as a passage"
    assert spy_q.call_args[0][0] == "who owns Auth-DB"


def test_query_and_passage_embeddings_differ_for_the_same_text():
    text = "Auth-DB credential rotation"
    q = np.array(vi.embed_query(text))
    p = np.array(vi.embed_passages([text])[0])
    assert not np.allclose(q, p, atol=1e-4), (
        "query and passage embeddings are identical -- the prefix is not applied")


def test_search_returns_ranked_hits(mem):
    client, settings = mem
    hits = vector_branch.search("who leads Infrastructure", 3,
                                settings=settings, client=client)
    assert len(hits) == 3
    assert hits == sorted(hits, key=lambda h: -h.score)
    assert "Marcus Lee" in hits[0].text


def test_search_respects_payload_filter(mem):
    client, settings = mem
    assert vector_branch.search("anything", 5, settings=settings, client=client,
                                doc_type="policy") == []
    assert vector_branch.search("anything", 5, settings=settings, client=client,
                                doc_type="sop")


# ------------------------------------------------------------------ RRF math

def _item(branch: str, rank: int, cid: str) -> RetrievedItem:
    return RetrievedItem(text=cid, branch=branch, source_type="chunk",
                         doc_id=cid, chunk_id=cid, branch_rank=rank)


def test_rrf_scores_match_the_formula():
    k = 60
    fused = reciprocal_rank_fusion({"vector": [_item("vector", 1, "a"),
                                               _item("vector", 2, "b")]}, k)
    assert fused[0].chunk_id == "a"
    assert fused[0].rrf_score == pytest.approx(1 / 61)
    assert fused[1].rrf_score == pytest.approx(1 / 62)


def test_rrf_sums_contributions_across_branches():
    """An item found by both branches must accumulate both, and appear once."""
    k = 60
    fused = reciprocal_rank_fusion({
        "vector": [_item("vector", 1, "a"), _item("vector", 2, "shared")],
        "graph": [_item("graph", 1, "shared")],
    }, k)
    ids = [f.chunk_id for f in fused]
    assert ids.count("shared") == 1
    shared = next(f for f in fused if f.chunk_id == "shared")
    assert shared.rrf_score == pytest.approx(1 / 62 + 1 / 61)
    # 1/62 + 1/61 > 1/61, so the doubly-supported item outranks the singleton
    assert ids[0] == "shared"


def test_rrf_k_flattens_rank_advantage():
    top = reciprocal_rank_fusion({"v": [_item("v", 1, "a")]}, 1)[0].rrf_score
    flat = reciprocal_rank_fusion({"v": [_item("v", 1, "a")]}, 1000)[0].rrf_score
    assert top > flat


def test_rrf_k_is_a_setting_not_a_literal():
    import inspect

    from app.retrieval import hybrid

    assert "rrf_k" in inspect.getsource(hybrid.HybridRetriever.retrieve)
    assert get_settings().rrf_k == 60


def test_rrf_assigns_dense_final_ranks():
    fused = reciprocal_rank_fusion(
        {"v": [_item("v", i, f"c{i}") for i in range(1, 5)]}, 60)
    assert [f.final_rank for f in fused] == [1, 2, 3, 4]


# --------------------------------------------------- graceful graph degradation

def test_graph_branch_declines_on_ambiguous_query(gq):
    """The resolver declines by design; the branch must return empty, not raise
    and not guess."""
    ctx = retrieve_graph_context("our reporting requirements under POL-002", gq)
    assert ctx.facts == []
    assert ctx.declined is True
    assert ctx.reason
    assert ctx.resolved_system is None


@pytest.mark.parametrize("query", [
    "what is the parental leave policy",
    "gateway drug rehabilitation",
    "",
    "   ",
    "how many transactions failed last quarter",
])
def test_graph_branch_never_raises(gq, query):
    ctx = retrieve_graph_context(query, gq)
    assert isinstance(ctx, GraphContext)
    assert isinstance(ctx.facts, list)


def test_graph_branch_resolves_and_renders(gq):
    ctx = retrieve_graph_context("What breaks if Auth-DB goes down?", gq)
    assert ctx.resolved_system == "Auth-DB"
    assert not ctx.declined
    blob = " ".join(f.text for f in ctx.facts)
    assert "Payment-Service" in blob
    assert all(isinstance(f, GraphFact) and f.path and f.template for f in ctx.facts)


def test_empty_cascade_is_stated_not_silent(gq):
    """Nothing depends on DataWarehouse or UserProfile-API. Emitting nothing
    would let a model assume the query failed rather than that the answer is
    'no downstream systems'."""
    ctx = retrieve_graph_context("what breaks if UserProfile-API fails?", gq)
    cascade = [f for f in ctx.facts if f.template == "dependency_cascade"]
    assert cascade
    assert "no recorded downstream cascade" in cascade[0].text


def test_hybrid_degrades_to_vector_when_graph_declines(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("what is the parental leave policy",
                         Mode.HYBRID_RERANK)
    assert res.graph_declined is True
    assert res.n_graph == 0
    assert res.items, "vector branch must still answer"
    assert all(i.branch == "vector" for i in res.items)


# ------------------------------------------------------------------- re-rank

def test_reranker_actually_reorders(mem, gq):
    """A re-ranker that returns its input order is a no-op and worthless.

    Fed a deliberately inverted candidate list -- the correct answer last --
    it must pull the relevant item to the top.
    """
    client, settings = mem
    candidates = [
        RetrievedItem(text=t, branch="vector", source_type="chunk",
                      doc_id=f"doc{i}", chunk_id=f"c{i}", branch_rank=i + 1)
        for i, t in enumerate([
            "The data retention policy requires records be kept for seven years.",
            "SOP-01 defines the incident response procedure for all engineers.",
            "Payment-Service is owned by the Billing team, led by Priya Sharma.",
            "Auth-DB is owned by the Infrastructure team, led by Marcus Lee.",
        ])
    ]
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        ranked = r._rerank("who leads the Infrastructure team?", candidates, 4)

    assert [i.doc_id for i in ranked] != ["doc0", "doc1", "doc2", "doc3"], (
        "re-ranker returned its input order -- it is a no-op")
    assert ranked[0].doc_id == "doc3", (
        f"expected the Infrastructure chunk first, got {ranked[0].text!r}")
    assert ranked[0].rerank_score > ranked[-1].rerank_score
    assert [i.final_rank for i in ranked] == [1, 2, 3, 4]


def test_reranker_changes_order_on_a_real_query(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        fused = r.retrieve("who leads Infrastructure", Mode.HYBRID_NO_RERANK,
                           top_n=5).items
        reranked = r.retrieve("who leads Infrastructure", Mode.HYBRID_RERANK,
                              top_n=5).items
    assert all(i.rerank_score is not None for i in reranked)
    assert [i.citation for i in fused] != [i.citation for i in reranked] or \
        len(fused) <= 1, "fusion and re-rank produced identical orderings"


def test_reranker_scores_are_discriminating(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("who leads Infrastructure", Mode.HYBRID_RERANK)
    scores = [i.rerank_score for i in res.items]
    assert len(set(round(s, 4) for s in scores)) > 1, "all scores identical"
    assert scores == sorted(scores, reverse=True)


def test_rerank_latency_is_reported_separately(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("incident response", Mode.HYBRID_RERANK)
        no_rr = r.retrieve("incident response", Mode.HYBRID_NO_RERANK)
    assert "rerank_ms" in res.timings_ms
    assert res.timings_ms["rerank_ms"] > 0
    assert "rerank_ms" not in no_rr.timings_ms


# ---------------------------------------------------------------- provenance

@pytest.mark.parametrize("mode", ALL_MODES)
def test_provenance_is_complete_in_every_mode(mem, gq, mode):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("who owns Auth-DB", mode)
    assert res.items
    for item in res.items:
        assert item.branch in {"vector", "graph"}
        assert item.source_type in {"chunk", "graph_fact"}
        assert item.branch_rank >= 1
        assert item.final_rank >= 1
        assert item.rrf_score > 0
        assert item.citation
        if item.source_type == "chunk":
            assert item.doc_id and item.chunk_id
        else:
            assert item.graph_path and item.graph_template


def test_rerank_score_is_none_when_not_reranked(mem, gq):
    """None, never 0.0: a citation must not carry a score that was never
    computed."""
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        for mode in (Mode.VECTOR_ONLY, Mode.HYBRID_NO_RERANK):
            res = r.retrieve("who owns Auth-DB", mode)
            assert all(i.rerank_score is None for i in res.items)
        res = r.retrieve("who owns Auth-DB", Mode.HYBRID_RERANK)
        assert all(i.rerank_score is not None for i in res.items)


def test_citations_name_real_retrieved_things(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("who owns Auth-DB", Mode.HYBRID_RERANK)
    seeded = {c for c in (f"doc{i}" for i in range(len(CORPUS)))}
    for item in res.items:
        if item.source_type == "chunk":
            assert item.doc_id in seeded, f"invented doc_id {item.doc_id}"
        else:
            assert item.graph_template


def test_result_doc_ids_are_ordered_and_deduped(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("who owns Auth-DB", Mode.HYBRID_RERANK)
    assert len(res.doc_ids) == len(set(res.doc_ids))


# ------------------------------------------------------------------ ablation

def test_ablate_returns_all_modes_with_one_interface(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        out = r.ablate("who owns Auth-DB")
    assert set(out) == {m.value for m in ALL_MODES}
    types = {type(v) for v in out.values()}
    assert len(types) == 1, "all modes must share one return type"
    for res in out.values():
        assert res.items and res.timings_ms["total_ms"] > 0


def test_vector_only_mode_never_touches_the_graph(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("What breaks if Auth-DB goes down?", Mode.VECTOR_ONLY)
    assert res.n_graph == 0
    assert all(i.branch == "vector" for i in res.items)
    assert "graph_ms" not in res.timings_ms


def test_modes_respect_top_n(mem, gq):
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        for mode in ALL_MODES:
            assert len(r.retrieve("incident", mode, top_n=3).items) <= 3


# ------------------------------------------- speculative graph facts are demoted

def test_uncued_graph_facts_are_marked_not_cued(gq):
    """A template that ran only because it is a default for the resolved entity
    is speculative -- the query never asked for it."""
    ctx = retrieve_graph_context(
        "What is the connection pool ceiling for Payment-Service?", gq)
    assert ctx.resolved_system == "Payment-Service"
    assert ctx.facts, "the entity resolved, so defaults should still run"
    assert all(f.cued is False for f in ctx.facts), [f.template for f in ctx.facts]

    cued = retrieve_graph_context("What breaks if Auth-DB goes down?", gq)
    assert any(f.cued for f in cued.facts)
    assert cued.facts[0].cued is True, "a cued fact must sort ahead of defaults"


def test_uncued_rank_offset_is_a_real_demotion():
    """MUTATION GUARD (math). With the offset at 0 an uncued fact ties a rank-1
    vector hit, which is exactly the bug: RRF rank 1 is the strongest possible
    endorsement and a guess must never land there."""
    s = get_settings()
    assert s.graph_uncued_rank_offset >= 1
    vector_top = 1.0 / (s.rrf_k + 1)
    uncued_top = 1.0 / (s.rrf_k + s.graph_uncued_rank_offset + 1)
    assert uncued_top < vector_top


def test_uncued_graph_fact_cannot_outrank_a_vector_hit(mem, gq):
    """MUTATION GUARD (integration). Reverting the demotion makes the uncued
    fact's RRF score equal the top vector hit's, and this fails.

    The query mentions Payment-Service, so the entity resolves and the default
    templates run -- but it asks about configuration, not ownership or
    dependencies, so nothing the graph produced was actually requested.
    """
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("What is the connection pool ceiling for Payment-Service?",
                         Mode.HYBRID_NO_RERANK, top_n=5)

    graph_items = [i for i in res.items if i.branch == "graph"]
    vector_items = [i for i in res.items if i.branch == "vector"]
    assert vector_items, "the vector branch must still contribute"
    assert res.items[0].branch == "vector", (
        f"an uncued graph fact reached rank 1: {res.items[0].citation}")

    top_vector_rrf = max(i.rrf_score for i in vector_items)
    for item in graph_items:
        assert item.metadata.get("cued") is False
        assert item.branch_rank > settings.graph_uncued_rank_offset
        assert item.rrf_score < top_vector_rrf, (
            "an uncued graph fact tied or beat the best vector hit")


def test_cued_graph_fact_may_still_win_rank_one(mem, gq):
    """The demotion must not silence the graph. When the query genuinely asks a
    graph question, the fact is allowed to lead."""
    client, settings = mem
    with HybridRetriever(settings, client=client, graph_queries=gq) as r:
        res = r.retrieve("What breaks if Auth-DB goes down?",
                         Mode.HYBRID_NO_RERANK, top_n=5)
    graph_items = [i for i in res.items if i.branch == "graph"]
    assert graph_items, "a cued graph question must produce graph context"
    assert any(i.metadata.get("cued") for i in graph_items)
    assert min(i.branch_rank for i in graph_items) == 1


def test_cue_matching_requires_both_word_boundaries():
    """REGRESSION. "circuit breaker" must not fire the "break" cue. With only a
    leading boundary it did, and a pure configuration question was answered
    with a dependency cascade -- the same prefix-matching failure the entity
    resolver was fixed for in Phase 2."""
    from app.retrieval.graph import TEMPLATE_CUES, _cue_score

    cascade = TEMPLATE_CUES["dependency_cascade"]
    assert _cue_score("What is the circuit breaker threshold for Auth-DB?",
                      cascade) == 0
    assert _cue_score("What breaks if Auth-DB goes down?", cascade) > 0
    # a few more prefix traps
    assert _cue_score("who is the team lead", TEMPLATE_CUES["team_leadership"]) > 0
    assert _cue_score("leadership training budget",
                      TEMPLATE_CUES["team_leadership"]) == 0


def test_config_question_gets_no_cued_graph_fact(gq):
    ctx = retrieve_graph_context(
        "What is the circuit breaker threshold for Auth-DB?", gq)
    assert all(f.cued is False for f in ctx.facts), \
        [f.template for f in ctx.facts]


# ---------------------------------------------------------------------------
# Regressions from Phase 3 validation.
# ---------------------------------------------------------------------------

def test_company_suffix_inc_does_not_cue_the_incidents_template():
    """'inc' as a cue matched the company suffix as a whole word, and a false
    cue sets cued=True -- the one flag that BYPASSES graph_uncued_rank_offset.
    A speculative fact would reach graph rank 1 by the exact route the fusion
    demotion exists to close."""
    from app.retrieval.graph import TEMPLATE_CUES, _cue_score
    cues = TEMPLATE_CUES["system_incidents"]
    assert _cue_score("ACME Inc uses Auth-DB for authentication", cues) == 0
    assert _cue_score("what incidents hit Auth-DB", cues) > 0
    assert "inc" not in cues


def test_incident_ids_still_route_without_the_inc_cue():
    """INC-nnn is matched by _INCIDENT_RE, not by the cue list, so dropping the
    'inc' cue must not cost incident routing."""
    r = retrieve("What was the root cause of INC-204?", mode=Mode.HYBRID_NO_RERANK)
    assert r.resolved_entities["incident"] == "INC-204"


@pytest.mark.parametrize("query, key", [
    # Nothing depends on DataWarehouse: the graph emits a correct negative fact
    # that the cross-encoder then drops for an on-topic non-answer (0.9944).
    ("What depends on DataWarehouse?", "negative_facts"),
    # INC-206 does not exist: the entity resolves, the graph returns nothing.
    ("What was the root cause of INC-206?", "entity_resolved_but_graph_empty"),
])
def test_abstention_signals_survive_reranking(query, key):
    """rerank_score alone cannot ground these -- both score >0.98 while being
    unanswerable, because a cross-encoder scores topical relevance, not
    answerability. The signal must be captured before rerank discards it."""
    r = retrieve(query, mode=Mode.HYBRID_RERANK)
    assert r.abstention[key], f"{key} lost for {query!r}"
    assert r.abstention["max_rerank_score"] > 0.9, "premise: rerank is confident here"


def test_answerable_query_raises_no_abstention_signal():
    """The negative case: signals must not fire on a question the corpus answers."""
    r = retrieve("Who owns Payment-Service?", mode=Mode.HYBRID_RERANK)
    assert not r.abstention["entity_resolved_but_graph_empty"]
    assert not r.abstention["negative_facts"]
