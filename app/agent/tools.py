"""The three tools the agent may call, and the provenance ledger behind them.

Three tools, one routing decision. `knowledge_search` already fuses vector and
graph, so the agent never has to choose a retrieval backend -- a choice it has
no information to make well. `graph_query` exists for the precise structural
questions fusion blurs ("who leads Infrastructure"), and `web_search` for what
the corpus provably does not contain.

Deliberately absent (see the plan's "Deliberate omissions"):

  * **sql_query** -- users.csv and products.csv *are* the graph. A SQL tool
    would query a duplicate copy of the same facts and let the agent answer one
    question two ways, with no way to adjudicate which answer is right.
  * **python_exec** -- excluded by instruction. `Toolbox.tools` is a plain list,
    so adding a sandboxed executor later is one entry, not a refactor.

The ledger is the important part. Every tool records the *actual* RetrievedItem
objects it returned into `Toolbox.records`. Citations are assembled in Phase 4's
graph from that ledger and never from model output, so a citation can only name
a document some tool really retrieved.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from langchain_core.tools import StructuredTool

from app.config import Settings, get_settings
from app.retrieval.graph_queries import GraphQueries, resolve_system
from app.retrieval.hybrid import HybridRetriever, Mode, RetrievedItem

log = logging.getLogger(__name__)

# The only graph queries reachable from a model. Free-form Cypher is not
# supported anywhere in this system: the model picks a NAME from this map and
# supplies an entity string, which arrives at Neo4j as a bound parameter.
GRAPH_TEMPLATES: dict[str, str] = {
    "system_ownership": "Which team (and person) owns a system, plus that team's lead.",
    "team_leadership": "Who leads a team.",
    # Direction is stated twice, in opposite words, because these two are the
    # pair a model confuses -- and a wrong-direction answer looks plausible.
    "dependency_cascade": ("DOWNSTREAM / reverse direction: which systems BREAK IF "
                           "the named system fails (systems that depend ON it). "
                           "Use for blast-radius and outage-impact questions."),
    "system_dependencies": ("UPSTREAM / forward direction: which systems the named "
                            "system DEPENDS ON (its own dependencies). Use for "
                            "'what does X depend on' and 'the system that X "
                            "depends on'."),
    "incident_sop": "Which SOP resolved an incident, e.g. INC-204.",
    "system_incidents": "Which incidents involved a system.",
    "team_roster": "Who is on a team, lead first.",
    "policies_for_system": "Which policies govern a system.",
}

TemplateName = Literal[
    "system_ownership",
    "team_leadership",
    "dependency_cascade",
    "system_dependencies",
    "incident_sop",
    "system_incidents",
    "team_roster",
    "policies_for_system",
]

# Templates keyed by an incident id rather than a system or team name.
_INCIDENT_TEMPLATES = {"incident_sop"}
_TEAM_TEMPLATES = {"team_leadership", "team_roster"}

NO_RESULTS = "NO_RESULTS"


class UnknownTemplateError(ValueError):
    """Raised when a caller names a graph template that does not exist."""


def _render_rows(template: str, entity: str, rows: list[dict]) -> str:
    lines = [f"graph:{template}({entity}) -> {len(rows)} row(s)"]
    for row in rows:
        lines.append("  " + "; ".join(f"{k}={v}" for k, v in row.items() if v is not None))
    return "\n".join(lines)


class Toolbox:
    """Holds the live clients and the provenance ledger for one agent run.

    Construct one per request. `records` accumulates every RetrievedItem the
    tools actually returned, in call order; `abstention` accumulates the
    structured signals Phase 3 established (negative facts, resolved-but-empty
    entities, max re-rank score) so Phase 5's grounding rail reads evidence
    rather than re-deriving it from prose.
    """

    def __init__(self, settings: Settings | None = None,
                 retriever: HybridRetriever | None = None,
                 graph_queries: GraphQueries | None = None,
                 tavily_client: Any | None = None):
        self.settings = settings or get_settings()
        self._retriever = retriever
        self._owns_retriever = retriever is None
        self._gq = graph_queries
        self._owns_gq = graph_queries is None
        self._tavily = tavily_client

        self.records: list[RetrievedItem] = []
        self.abstention: list[dict] = []
        self.calls: list[dict] = []

    # ------------------------------------------------------------- clients
    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self.settings)
        return self._retriever

    @property
    def graph_queries(self) -> GraphQueries:
        if self._gq is None:
            self._gq = self.retriever.graph_queries
            self._owns_gq = False
        return self._gq

    @property
    def tavily(self):
        if self._tavily is None:
            from tavily import TavilyClient
            self._tavily = TavilyClient(api_key=self.settings.tavily_api_key)
        return self._tavily

    def close(self) -> None:
        if self._retriever is not None and self._owns_retriever:
            self._retriever.close()
            self._retriever = None
        if self._gq is not None and self._owns_gq:
            self._gq.close()
            self._gq = None

    def __enter__(self) -> "Toolbox":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- tools
    def knowledge_search(self, query: str) -> str:
        """Hybrid vector+graph retrieval. Records provenance, returns context text."""
        from app.observability.langfuse_client import RETRIEVER, get_tracing

        # A `retriever`-typed span, nested under the tool span the agent opened.
        # Phase 3's own per-stage timings are attached as metadata rather than
        # re-measured, so the trace and the ablation harness report the same
        # numbers instead of two slightly different ones.
        with get_tracing().span("retrieval.hybrid_rerank", as_type=RETRIEVER,
                                input=query) as sp:
            result = self.retriever.retrieve(query, Mode.HYBRID_RERANK)
            sp.update(output={"n_items": len(result.items),
                              "citations": [i.citation for i in result.items][:10]},
                      metadata={"mode": Mode.HYBRID_RERANK.value
                                if hasattr(Mode.HYBRID_RERANK, "value")
                                else str(Mode.HYBRID_RERANK),
                                "timings_ms": result.timings_ms,
                                "resolved_entities": result.resolved_entities})
        self.records.extend(result.items)
        self.abstention.append({
            "tool": "knowledge_search", "query": query,
            **result.abstention,
            "resolved_entities": result.resolved_entities,
            "n_items": len(result.items),
            "returned_nothing": not result.items,
        })
        self.calls.append({"tool": "knowledge_search", "args": {"query": query},
                           "n_results": len(result.items),
                           "latency_ms": result.timings_ms.get("total_ms")})
        if not result.items:
            return (f"{NO_RESULTS}: the corpus returned nothing for {query!r}. "
                    "Do not invent an answer.")
        return result.context_text()

    def graph_query(self, template: str, entity: str, max_depth: int = 3) -> str:
        """Run ONE named Cypher template with `entity` as a bound parameter."""
        if template not in GRAPH_TEMPLATES:
            self.calls.append({"tool": "graph_query", "args": {"template": template,
                                                              "entity": entity},
                               "error": "unknown_template"})
            raise UnknownTemplateError(
                f"unknown template {template!r}; choose one of "
                f"{sorted(GRAPH_TEMPLATES)}"
            )

        gq = self.graph_queries
        method = getattr(gq, template)
        rows = method(entity, max_depth) if template == "dependency_cascade" \
            else method(entity)

        # Why the entity did not resolve matters to the abstention signal: a
        # declined resolution is "I could not identify that system", which is a
        # different answer from "that system exists and has no such fact".
        resolved = None
        if template in _TEAM_TEMPLATES:
            resolved = gq.resolve_team(entity)
        elif template not in _INCIDENT_TEMPLATES:
            resolved = resolve_system(entity).value
        else:
            resolved = entity.strip().upper()

        self.abstention.append({
            "tool": "graph_query", "template": template, "entity": entity,
            "resolved_entity": resolved,
            "entity_resolved_but_graph_empty": bool(resolved and not rows),
            "entity_unresolved": resolved is None,
            "graph_declined": resolved is None,
            "returned_nothing": not rows,
            "negative_facts": ([] if rows else
                               [f"{template}({resolved or entity}) returned no rows"]),
        })
        self.calls.append({"tool": "graph_query",
                           "args": {"template": template, "entity": entity},
                           "n_results": len(rows)})

        if not rows:
            if resolved is None:
                return (f"{NO_RESULTS}: could not identify {entity!r} as a known "
                        f"entity for template {template}. Say so rather than guessing.")
            return (f"{NO_RESULTS}: {resolved} is known, but the graph holds no "
                    f"{template} rows for it. The truthful answer is that there "
                    f"are none -- do not substitute a plausible one.")

        text = _render_rows(template, resolved or entity, rows)
        item = RetrievedItem(
            text=text, branch="graph", source_type="graph_fact",
            graph_path=str(resolved or entity), graph_template=template,
            branch_rank=1, branch_score=1.0,
            metadata={"rows": rows, "entity": resolved or entity, "cued": True,
                      "negative": False, "tool": "graph_query"},
        )
        self.records.append(item)
        return text

    def web_search(self, query: str, max_results: int = 3) -> str:
        """Tavily. For questions the internal corpus cannot answer."""
        try:
            resp = self.tavily.search(query=query, max_results=max_results,
                                      search_depth="basic")
            results = resp.get("results", []) if isinstance(resp, dict) else []
        except Exception as exc:  # noqa: BLE001 - network/provider errors vary
            log.warning("web_search failed: %s", exc)
            self.calls.append({"tool": "web_search", "args": {"query": query},
                               "error": type(exc).__name__})
            return f"{NO_RESULTS}: web search failed ({type(exc).__name__})."

        self.calls.append({"tool": "web_search", "args": {"query": query},
                           "n_results": len(results)})
        if not results:
            return f"{NO_RESULTS}: web search returned nothing for {query!r}."

        blocks = []
        for rank, r in enumerate(results, start=1):
            url = r.get("url", "")
            text = (r.get("content") or "")[:1200]
            blocks.append(f"[web:{url}]\n{text}")
            self.records.append(RetrievedItem(
                text=text, branch="web", source_type="web",
                doc_id=None, graph_path=url, graph_template="web_search",
                branch_rank=rank, branch_score=float(r.get("score") or 0.0),
                metadata={"url": url, "title": r.get("title"), "external": True},
            ))
        return "\n\n".join(blocks)

    # -------------------------------------------------------- LC tool objs
    def build_tools(self) -> list[StructuredTool]:
        """LangChain tool objects bound to THIS toolbox, so every call is ledgered."""

        def knowledge_search(
            query: Annotated[str, "A natural-language question or search phrase. "
                                  "Pass the user's wording; do not translate it "
                                  "into keywords."],
        ) -> str:
            """Search ACME's internal knowledge base (policies, incident reports,
            SOPs, technical manuals, FAQ) using hybrid vector + knowledge-graph
            retrieval with cross-encoder re-ranking. This is the default tool:
            use it first for almost every question about ACME."""
            return self.knowledge_search(query)

        def graph_query(
            template: Annotated[TemplateName,
                                "Which pre-written graph query to run. One of: "
                                + "; ".join(f"{k}: {v}" for k, v in
                                            GRAPH_TEMPLATES.items())],
            entity: Annotated[str, "The system name (e.g. 'Payment-Service', "
                                   "'Auth-DB'), team name (e.g. 'Infrastructure') "
                                   "or incident id (e.g. 'INC-204') to run it on."],
            max_depth: Annotated[int, "dependency_cascade only: hop limit, 1-5."] = 3,
        ) -> str:
            """Run one pre-written, parameterized query against ACME's knowledge
            graph for precise structural facts: ownership, team leadership,
            dependency cascades, incident-to-SOP links, a system's incidents, a
            team roster, or the policies governing a system. You choose a
            template name and an entity -- you cannot write Cypher."""
            return self.graph_query(template, entity, max_depth)

        def web_search(
            query: Annotated[str, "The public-web search query."],
        ) -> str:
            """Search the public web. Use ONLY for general or external knowledge
            that ACME's internal corpus would not contain. Never use it to
            answer a question about ACME's own systems, teams, policies or
            incidents -- if the internal corpus lacks that, the answer is that
            we do not have it."""
            return self.web_search(query)

        return [
            StructuredTool.from_function(knowledge_search),
            StructuredTool.from_function(graph_query),
            StructuredTool.from_function(web_search),
        ]


def build_toolbox(settings: Settings | None = None, **kwargs) -> Toolbox:
    return Toolbox(settings=settings, **kwargs)
