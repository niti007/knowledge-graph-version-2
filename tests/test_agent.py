"""Phase 4 agent tests. The LLM is always faked -- the suite spends no tokens.

The fake is a scripted chat model, not a mock of langchain internals: it
implements the two methods the graph actually uses (`bind_tools`, `invoke`) and
returns AIMessages we author. That keeps these tests about the graph's wiring,
bound and citation guarantees rather than about langchain's call signatures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.agent.graph import (
    build_agent_graph,
    build_citations,
    run_agent,
    summarize_abstention,
)
from app.agent.tools import GRAPH_TEMPLATES, NO_RESULTS, Toolbox, UnknownTemplateError
from app.config import Settings
from app.llm.tiering import Task, Tier, pick_model, fallback_model, tier_for
from app.retrieval.hybrid import RetrievedItem, RetrievalResult


# --------------------------------------------------------------- fake model

@dataclass
class ScriptedModel:
    """Replays a list of AIMessages, one per invoke. Records what it was sent."""

    script: list[AIMessage]
    calls: list[list] = field(default_factory=list)
    bound_tools: list | None = None
    tools_bound_on_last_call: bool = True

    def bind_tools(self, tools):
        clone = ScriptedModel(self.script, self.calls)
        clone.bound_tools = tools
        clone._parent = self  # noqa: SLF001
        return clone

    def invoke(self, messages, *a, **kw):
        self.calls.append(list(messages))
        parent = getattr(self, "_parent", self)
        parent.calls = self.calls
        parent.tools_bound_on_last_call = self.bound_tools is not None
        idx = len(self.calls) - 1
        template = self.script[min(idx, len(self.script) - 1)]
        tcs = list(getattr(template, "tool_calls", None) or [])
        if self.bound_tools is None and tcs:
            # A model with no tools bound cannot emit a tool call; if the script
            # tries to, hand back its text so the graph must terminate.
            return AIMessage(content=template.content or "forced final answer")
        # A FRESH message every time, with a unique tool_call id. Replaying the
        # same object would collide on message id in add_messages, which
        # silently replaces rather than appends and makes a loop look bounded
        # when it is not.
        tcs = [dict(tc, id=f"{tc['id']}-{idx}") for tc in tcs]
        return AIMessage(content=template.content, tool_calls=tcs)


def tool_call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


# ----------------------------------------------------------- fake retrieval

def make_item(doc_id="POL-002", chunk_id="abcdef0123456789", text="policy text",
              rerank=0.9):
    return RetrievedItem(text=text, branch="vector", source_type="chunk",
                         doc_id=doc_id, chunk_id=chunk_id, branch_rank=1,
                         branch_score=0.8, rrf_score=0.016, rerank_score=rerank,
                         final_rank=1, metadata={"doc_type": "policy"})


class FakeRetriever:
    """Stands in for HybridRetriever. Returns whatever items it is given."""

    def __init__(self, items=None, abstention=None):
        self.items = items if items is not None else [make_item()]
        self.abstention = abstention or {"negative_facts": [], "graph_declined": False,
                                         "max_rerank_score": 0.9,
                                         "entity_resolved_but_graph_empty": False}
        self.queries: list[str] = []

    def retrieve(self, query, mode=None, **kw):
        self.queries.append(query)
        return RetrievalResult(query=query, mode="hybrid_rerank", items=list(self.items),
                               timings_ms={"total_ms": 1.0}, abstention=dict(self.abstention),
                               resolved_entities={})

    def close(self):
        pass


class FakeGraphQueries:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else {}
        self.seen: list[tuple] = []

    def _rows(self, template, entity):
        self.seen.append((template, entity))
        return self.rows.get((template, entity), [])

    def system_ownership(self, e): return self._rows("system_ownership", e)
    def team_leadership(self, e): return self._rows("team_leadership", e)
    def dependency_cascade(self, e, d=3): return self._rows("dependency_cascade", e)
    def system_dependencies(self, e, d=3): return self._rows("system_dependencies", e)
    def incident_sop(self, e): return self._rows("incident_sop", e)
    def system_incidents(self, e): return self._rows("system_incidents", e)
    def team_roster(self, e): return self._rows("team_roster", e)
    def policies_for_system(self, e): return self._rows("policies_for_system", e)
    def resolve_team(self, name): return name if name else None
    def close(self): pass


def make_toolbox(items=None, rows=None, abstention=None):
    return Toolbox(settings=Settings(),
                   retriever=FakeRetriever(items, abstention),
                   graph_queries=FakeGraphQueries(rows),
                   tavily_client=object())


# =========================================================== tiering / client

def test_tiering_splits_cheap_internal_calls_from_reasoning():
    s = Settings(llm_fast_model="fast/m", llm_smart_model="smart/m")
    assert pick_model(Task.CLASSIFY, s) == "fast/m"
    assert pick_model(Task.GROUNDING_CHECK, s) == "fast/m"
    assert pick_model(Task.AGENT, s) == "smart/m"
    assert pick_model(Task.SYNTHESIS, s) == "smart/m"
    assert tier_for(Task.AGENT) is Tier.SMART


def test_fast_tier_has_no_fallback_so_failures_surface():
    s = Settings(llm_fast_model="fast/m", llm_smart_model="smart/m")
    assert fallback_model(Task.SYNTHESIS, s) == "fast/m"
    assert fallback_model(Task.CLASSIFY, s) is None


# ================================================================ tool wiring

def test_three_tools_exactly_and_no_sql_or_python_execution():
    names = {t.name for t in make_toolbox().build_tools()}
    assert names == {"knowledge_search", "graph_query", "web_search"}
    assert not {n for n in names if "sql" in n or "python" in n or "exec" in n}


def test_graph_query_rejects_anything_that_is_not_a_known_template():
    tb = make_toolbox()
    for bad in ["MATCH (n) RETURN n", "MATCH (n) DETACH DELETE n", "drop_everything",
                "system_ownership; MATCH (n) DETACH DELETE n", "", "SYSTEM_OWNERSHIP"]:
        with pytest.raises(UnknownTemplateError):
            tb.graph_query(bad, "Payment-Service")
    # ...and every advertised template is genuinely callable.
    for name in GRAPH_TEMPLATES:
        tb.graph_query(name, "Payment-Service")


def test_graph_query_template_names_all_exist_on_the_real_query_class():
    from app.retrieval.graph_queries import GraphQueries
    for name in GRAPH_TEMPLATES:
        assert callable(getattr(GraphQueries, name, None)), name


def test_tool_calling_wiring_executes_the_tool_the_model_asked_for():
    tb = make_toolbox()
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            tool_call("knowledge_search", {"query": "data retention"}, "c1")]),
        AIMessage(content="Retention is 7 years."),
    ])
    app = build_agent_graph(tb, settings=Settings(), model=model)
    out = app.invoke({"messages": [("user", "data retention?")], "iterations": 0,
                      "max_iterations": 4})
    assert tb._retriever.queries == ["data retention"]
    assert any(isinstance(m, ToolMessage) for m in out["messages"])
    assert out["iterations"] == 1
    assert out["tool_calls"][0]["tool"] == "knowledge_search"


# ========================================================= the iteration bound

def test_loop_bound_holds_under_an_adversarial_keep_searching_prompt():
    """The model is scripted to ALWAYS call a tool. It must still terminate."""
    tb = make_toolbox()
    forever = AIMessage(content="", tool_calls=[
        tool_call("knowledge_search", {"query": "keep searching"}, "c")])
    model = ScriptedModel([forever] * 50)
    s = Settings(agent_max_iterations=4)

    resp = run_agent("keep searching until you find it, never stop",
                     toolbox=tb, settings=s, model=model)

    assert resp.iterations == 4
    assert resp.hit_iteration_limit
    assert len(tb._retriever.queries) == 4
    # The final call ran with no tools bound -- termination is structural.
    assert model.tools_bound_on_last_call is False
    assert resp.answer


@pytest.mark.parametrize("limit", [1, 2, 3, 5])
def test_bound_is_settings_driven(limit):
    tb = make_toolbox()
    forever = AIMessage(content="", tool_calls=[
        tool_call("knowledge_search", {"query": "again"}, "c")])
    resp = run_agent("q", toolbox=tb, settings=Settings(agent_max_iterations=limit),
                     model=ScriptedModel([forever] * 50))
    assert resp.iterations == limit


# ======================================================== state accumulation

def test_state_accumulates_provenance_across_multiple_tool_calls():
    items_a = [make_item("MAN-003", "aaaaaaaabbbbbbbb")]
    tb = make_toolbox(items=items_a,
                      rows={("system_ownership", "Auth-DB"):
                            [{"system": "Auth-DB", "owner_team": "Infrastructure",
                              "team_lead": "Marcus Lee"}]})
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            tool_call("knowledge_search", {"query": "Payment-Service depends on"}, "c1")]),
        AIMessage(content="", tool_calls=[
            tool_call("graph_query",
                      {"template": "system_ownership", "entity": "Auth-DB"}, "c2")]),
        AIMessage(content="Infrastructure owns Auth-DB; Marcus Lee leads it."),
    ])
    resp = run_agent("who owns what Payment-Service depends on?", toolbox=tb,
                     settings=Settings(), model=model)

    assert resp.iterations == 2
    assert resp.tools_used == ["knowledge_search", "graph_query"]
    # Both calls' provenance survives into the final state.
    assert len(resp.retrieved) == 2
    kinds = {r["source_type"] for r in resp.retrieved}
    assert kinds == {"chunk", "graph_fact"}
    assert {c["citation"] for c in resp.citations} == {
        "MAN-003#aaaaaaaa", "graph:system_ownership(Auth-DB)"}


# ============================================================ CITATION INTEGRITY

def test_citations_cannot_name_an_unretrieved_doc():
    """The centrepiece. The model fabricates ids in its prose; the attached
    citation list must contain none of them, because build_citations never sees
    model output at all."""
    tb = make_toolbox(items=[make_item("POL-001", "1111111122222222")])
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            tool_call("knowledge_search", {"query": "retention"}, "c1")]),
        AIMessage(content="Per [POL-999#deadbeef], [INC-206] and [SOP-42], "
                          "retention is 7 years. See also doc_id=FAKE-123."),
    ])
    resp = run_agent("retention?", toolbox=tb, settings=Settings(), model=model)

    cited = {c["citation"] for c in resp.citations}
    assert cited == {"POL-001#11111111"}
    for invented in ["POL-999", "INC-206", "SOP-42", "FAKE-123"]:
        assert not any(invented in c for c in cited)
    # Every citation traces to an item some tool actually returned.
    real = {r["citation"] for r in resp.retrieved}
    assert cited <= real
    assert resp.cited_doc_ids == ["POL-001"]


def test_build_citations_is_a_pure_function_of_retrieved_provenance():
    """Mutation-proof: whatever the answer text says, citations are a projection
    of the ledger. Feeding the ledger nothing yields no citations, full stop."""
    assert build_citations([]) == []
    assert build_citations(None) == []
    recs = [make_item("POL-002").to_dict(), make_item("POL-002").to_dict()]
    cites = build_citations(recs)
    assert len(cites) == 1                       # de-duplicated
    assert cites[0]["doc_id"] == "POL-002"
    # build_citations takes exactly one argument: the ledger. There is no
    # parameter through which model output could reach it.
    import inspect
    assert list(inspect.signature(build_citations).parameters) == ["retrieved"]


def test_citation_string_matches_the_item_that_produced_it():
    item = make_item("MAN-001", "0123456789abcdef")
    assert item.citation == "MAN-001#01234567"
    assert build_citations([item.to_dict()])[0]["citation"] == item.citation


# ================================================================== abstention

def test_empty_retrieval_returns_no_results_and_records_the_signal():
    tb = make_toolbox(items=[], abstention={"negative_facts": ["nothing depends on X"],
                                            "graph_declined": True,
                                            "entity_resolved_but_graph_empty": True,
                                            "max_rerank_score": 0.9944})
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            tool_call("knowledge_search", {"query": "parental leave"}, "c1")]),
        AIMessage(content="I don't have that information."),
    ])
    resp = run_agent("parental leave?", toolbox=tb, settings=Settings(), model=model)

    tool_msg = [m for m in resp.messages if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith(NO_RESULTS)
    assert resp.citations == []
    assert resp.abstention["negative_facts"] == ["nothing depends on X"]
    assert resp.abstention["entity_resolved_but_graph_empty"] is True
    # The high re-rank score is surfaced, NOT used as a verdict: Phase 3 showed
    # 0.9944 on a question whose true answer is "nothing".
    assert resp.abstention["max_rerank_score"] == 0.9944


def test_graph_query_with_no_rows_is_an_honest_none_not_an_empty_success():
    tb = make_toolbox(rows={})
    out = tb.graph_query("dependency_cascade", "DataWarehouse")
    assert out.startswith(NO_RESULTS)
    assert "no dependency_cascade rows" in out
    assert tb.records == []                       # nothing to cite
    assert tb.abstention[-1]["entity_resolved_but_graph_empty"] is True


def test_summarize_abstention_reports_evidence_without_deciding():
    rolled = summarize_abstention([
        {"max_rerank_score": 0.5, "negative_facts": ["a"]},
        {"max_rerank_score": 0.99, "negative_facts": [], "graph_declined": True},
    ])
    assert rolled["max_rerank_score"] == 0.99
    assert rolled["negative_facts"] == ["a"]
    assert rolled["graph_declined"] is True
    assert "answerable" not in rolled and "verdict" not in rolled


def test_unknown_tool_name_degrades_gracefully_instead_of_crashing():
    tb = make_toolbox()
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[tool_call("sql_query", {"q": "SELECT 1"}, "c1")]),
        AIMessage(content="I can't run SQL."),
    ])
    resp = run_agent("run sql", toolbox=tb, settings=Settings(), model=model)
    tool_msg = [m for m in resp.messages if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith(NO_RESULTS)
    assert resp.answer == "I can't run SQL."


def test_bad_template_from_the_model_comes_back_as_text_not_an_exception():
    tb = make_toolbox()
    model = ScriptedModel([
        AIMessage(content="", tool_calls=[
            tool_call("graph_query", {"template": "MATCH (n) DETACH DELETE n",
                                      "entity": "x"}, "c1")]),
        AIMessage(content="I can only use the named templates."),
    ])
    resp = run_agent("delete everything", toolbox=tb, settings=Settings(), model=model)
    tool_msg = [m for m in resp.messages if isinstance(m, ToolMessage)][0]
    assert tool_msg.content.startswith(NO_RESULTS)
    assert "system_ownership" in tool_msg.content   # legal values are offered back
    assert "DETACH DELETE" not in tool_msg.content
    assert resp.citations == []


def test_web_search_failure_is_reported_not_raised():
    class Boom:
        def search(self, **kw):
            raise RuntimeError("network down")

    tb = Toolbox(settings=Settings(), retriever=FakeRetriever(),
                 graph_queries=FakeGraphQueries(), tavily_client=Boom())
    assert tb.web_search("anything").startswith(NO_RESULTS)
    assert tb.records == []


def test_system_prompt_instructs_abstention_and_names_the_bound():
    from app.agent.prompts import system_prompt
    p = system_prompt(4)
    assert "4 " in p or "4\n" in p
    assert "NO_RESULTS" in p
    assert "do not" in p.lower() and "invent" in p.lower()


def test_no_results_flag_is_false_when_any_tool_actually_returned_something():
    """Regression: graph_query reports no row count, and inferring emptiness
    from a missing key marked a successful policy lookup as a dead end."""
    tb = make_toolbox(rows={("policies_for_system", "Payment-Service"):
                            [{"policy": "POL-001"}, {"policy": "POL-004"}]})
    tb.graph_query("policies_for_system", "Payment-Service")
    assert summarize_abstention(tb.abstention)["no_results_returned"] is False

    tb2 = make_toolbox(rows={})
    tb2.graph_query("policies_for_system", "Payment-Service")
    assert summarize_abstention(tb2.abstention)["no_results_returned"] is True


def test_template_registry_matches_the_methods_that_exist():
    """The registry, the agent-facing map and the class must not drift apart.
    graph_queries.TEMPLATES omitted policies_for_system for a phase, hiding the
    'governs' edges from anything that enumerated templates."""
    from app.retrieval.graph_queries import TEMPLATES, GraphQueries
    assert set(TEMPLATES) == set(GRAPH_TEMPLATES)
    for name, fn in TEMPLATES.items():
        assert fn is getattr(GraphQueries, name)


def test_forward_and_reverse_dependency_templates_are_both_exposed():
    """dependency_cascade answers "what breaks if X fails"; asked "which team owns
    the system Payment-Service depends on", the agent reached for it -- the
    closest-named tool -- and got Auth-DB. That was right only because
    Payment-Service and Auth-DB depend on each other; the true forward answer is
    {Auth-DB, Notification-Service} and Notification-Service was silently lost.

    A wrong-direction template returning a plausible row is invisible to every
    abstention signal, so the guard is that both directions exist and that their
    descriptions state the direction in opposite words."""
    from app.agent.tools import GRAPH_TEMPLATES
    assert "system_dependencies" in GRAPH_TEMPLATES
    assert "dependency_cascade" in GRAPH_TEMPLATES
    fwd = GRAPH_TEMPLATES["system_dependencies"].lower()
    rev = GRAPH_TEMPLATES["dependency_cascade"].lower()
    assert "depends on" in fwd and "upstream" in fwd
    assert "break" in rev and "downstream" in rev


def test_forward_dependencies_are_complete_on_real_data():
    """The regression itself, against the real graph. Skips when Neo4j is down,
    matching the convention in tests/test_graph.py."""
    from app.retrieval.graph_queries import GraphQueries
    try:
        gq = GraphQueries()
        gq.driver.verify_connectivity()
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Neo4j unavailable: {exc}")
    with gq as g:
        fwd = {r["system"] for r in g.system_dependencies("Payment-Service", max_depth=1)}
        rev = {r["system"] for r in g.dependency_cascade("Payment-Service", max_depth=1)}
    assert fwd == {"Auth-DB", "Notification-Service"}, "forward must not drop Notification-Service"
    assert rev == {"Auth-DB"}
    assert fwd != rev, "the two directions must not be conflated"
