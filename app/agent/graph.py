"""The LangGraph agent: `agent ⟲ tools → end`, bounded, with citations from code.

Two decisions carry the phase.

**The loop bound is structural, not requested.** At `agent_max_iterations` the
agent node re-invokes the model with *no tools bound at all*. There is no fifth
round to grant, so an adversarial "keep searching until you find it" prompt
cannot extend the loop -- the model is simply no longer able to emit a tool
call. A prompt asking it to stop would be a request; unbinding the tools is a
fact.

**Citations are assembled from the ledger, never from the model.** Every tool
call appends the RetrievedItem objects it actually returned to `state.retrieved`.
`build_citations` reads only that list. The model's prose may reference sources,
but the citation list attached to the response is derived in code, so a citation
naming a document no tool retrieved is unreachable by construction rather than
unlikely. See tests/test_agent.py::test_citations_cannot_name_an_unretrieved_doc.

Phase 5 (guardrails), 6 (API) and 7 (cache/tracing) sit outside this module:
`run_agent` is a plain function over a plain response object, so rails wrap it
without reaching inside the loop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agent.prompts import FINAL_TURN_NUDGE, system_prompt
from pydantic import ValidationError

from app.agent.tools import (
    GRAPH_TEMPLATES,
    NO_RESULTS,
    Toolbox,
    UnknownTemplateError,
)
from app.config import Settings, get_settings
from app.llm.client import get_chat_model
from app.llm.tiering import Task, pick_model, tier_for
from app.observability.langfuse_client import AGENT, TOOL, get_tracing

log = logging.getLogger(__name__)


def _extend(left: list, right: list) -> list:
    """Reducer: accumulate across nodes instead of overwriting."""
    return (left or []) + (right or [])


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # Provenance accumulated across every tool call in the run. This is the
    # citation source of truth.
    retrieved: Annotated[list[dict], _extend]
    # Phase 3's structured abstention evidence, per tool call. Phase 5's
    # grounding rail reads this; nothing here enforces anything.
    abstention: Annotated[list[dict], _extend]
    tool_calls: Annotated[list[dict], _extend]
    iterations: int
    max_iterations: int


@dataclass
class AgentResponse:
    answer: str
    citations: list[dict] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    abstention: dict = field(default_factory=dict)
    iterations: int = 0
    hit_iteration_limit: bool = False
    latency_ms: float = 0.0
    messages: list[BaseMessage] = field(default_factory=list)
    # Per-LLM-call usage for THIS run, in call order: task, tier, model and the
    # provider's own token counts. Request-local by construction (the list is
    # created in `run_agent` and passed down), which is what lets the Phase 7
    # tiering table be a measurement rather than a claim.
    llm_usage: list[dict] = field(default_factory=list)

    @property
    def cited_doc_ids(self) -> list[str]:
        return [c["doc_id"] for c in self.citations if c.get("doc_id")]


# ------------------------------------------------------------------ citations

def build_citations(retrieved: list[dict]) -> list[dict]:
    """Assemble the citation list from retrieved provenance ONLY.

    Input is the ledger of items tools actually returned. The model's output is
    not an argument to this function and cannot be: that is the whole guarantee.
    De-duplicated by citation string, ordered by first appearance.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for rec in retrieved or []:
        cite = rec.get("citation")
        if not cite or cite in seen:
            continue
        seen.add(cite)
        out.append({
            "citation": cite,
            "source_type": rec.get("source_type"),
            "branch": rec.get("branch"),
            "doc_id": rec.get("doc_id"),
            "chunk_id": rec.get("chunk_id"),
            "graph_template": rec.get("graph_template"),
            "graph_path": rec.get("graph_path"),
            "title": (rec.get("metadata") or {}).get("title"),
            "url": (rec.get("metadata") or {}).get("url"),
            "rerank_score": rec.get("rerank_score"),
            "final_rank": rec.get("final_rank"),
        })
    return out


def summarize_abstention(signals: list[dict]) -> dict:
    """Roll per-call abstention evidence into one dict for Phase 5.

    Deliberately no threshold and no verdict. Phase 3 showed max_rerank_score is
    a poor answerability signal on its own (0.9944 for a question whose true
    answer is "nothing"), so this reports the evidence and leaves the judgement
    to the rail that owns it.
    """
    signals = signals or []
    scores = [s.get("max_rerank_score") for s in signals
              if s.get("max_rerank_score") is not None]
    negatives: list[str] = []
    for s in signals:
        negatives.extend(s.get("negative_facts") or [])
    return {
        "any_tool_called": bool(signals),
        "negative_facts": negatives,
        "entity_resolved_but_graph_empty": any(
            s.get("entity_resolved_but_graph_empty") for s in signals),
        "entity_unresolved": any(s.get("entity_unresolved") for s in signals),
        "graph_declined": any(s.get("graph_declined") for s in signals),
        "max_rerank_score": max(scores) if scores else None,
        # True only when EVERY tool call came back empty. Each tool sets
        # `returned_nothing` itself; inferring it here from a row count that
        # graph_query does not report made a successful policy lookup look like
        # a dead end.
        "no_results_returned": bool(signals) and all(
            s.get("returned_nothing") for s in signals),
        "per_call": signals,
    }


# ----------------------------------------------------------------- the graph

def build_agent_graph(toolbox: Toolbox, settings: Settings | None = None,
                      model: Any | None = None,
                      usage_sink: list[dict] | None = None):
    """Compile `agent ⟲ tools → end` over this toolbox.

    `model` is an injection seam for tests: pass a fake chat model and the whole
    loop runs deterministically with no provider call. `usage_sink` is the
    caller's own list; token counts are appended to it rather than accumulated
    in a module global, so two concurrent runs cannot merge their accounting.
    """
    s = settings or get_settings()
    max_iterations = s.agent_max_iterations
    tools = toolbox.build_tools()
    by_name = {t.name: t for t in tools}

    base = model if model is not None else get_chat_model(
        Task.AGENT, temperature=s.agent_temperature, settings=settings)
    with_tools = base.bind_tools(tools)

    # The model id the tier resolved to, for the trace. Read off the injected
    # model when a test supplies one, so a fake never reports a real model name.
    model_id = (getattr(base, "model_name", None) or getattr(base, "model", None)
                or pick_model(Task.AGENT, settings))
    tier_name = tier_for(Task.AGENT).value

    def _traced_invoke(llm, msgs, *, name: str):
        """One LLM round-trip, as a Langfuse `generation` with usage and cost.

        Token counts come off `usage_metadata`, which langchain-openai fills
        from the provider response; OpenRouter returns real usage, so these are
        measured rather than estimated. Cost is computed in the observability
        module because Langfuse cannot price OpenRouter's `openai/` ids.
        """
        tracing = get_tracing()
        with tracing.generation(
                name, model=str(model_id),
                input=[getattr(m, "content", str(m)) for m in msgs][-2:],
                model_parameters={"temperature": s.agent_temperature}) as sp:
            reply = llm.invoke(msgs)
            usage = getattr(reply, "usage_metadata", None) or {}
            if usage_sink is not None:
                usage_sink.append({
                    "task": Task.AGENT.value, "tier": tier_name,
                    "model": str(model_id), "span": name,
                    "input": usage.get("input_tokens") or 0,
                    "output": usage.get("output_tokens") or 0})
            tracing.record_generation(
                sp, model=str(model_id),
                prompt_tokens=usage.get("input_tokens"),
                completion_tokens=usage.get("output_tokens"),
                output=getattr(reply, "content", "") or "",
                task=Task.AGENT.value, tier=tier_name,
                extra={"n_tool_calls": len(getattr(reply, "tool_calls", None) or [])})
        return reply

    def agent_node(state: AgentState) -> dict:
        iterations = state.get("iterations", 0)
        limit = state.get("max_iterations", max_iterations)
        messages = list(state["messages"])

        if iterations >= limit:
            # Ceiling: no tools bound, so no tool call is representable.
            reply = _traced_invoke(
                base, messages + [HumanMessage(content=FINAL_TURN_NUDGE)],
                name="llm.agent.final_turn")
            if getattr(reply, "tool_calls", None):
                reply = AIMessage(content=reply.content or "")
            return {"messages": [reply]}

        return {"messages": [_traced_invoke(with_tools, messages,
                                            name=f"llm.agent.turn_{iterations + 1}")]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out_messages: list[ToolMessage] = []
        before_records = len(toolbox.records)
        before_signals = len(toolbox.abstention)
        before_calls = len(toolbox.calls)

        for call in getattr(last, "tool_calls", []) or []:
            name = call.get("name")
            args = call.get("args") or {}
            tool = by_name.get(name)
            if tool is None:
                content = (f"{NO_RESULTS}: no tool named {name!r}. Available: "
                           f"{sorted(by_name)}")
            else:
                # The span wraps the WHOLE call including its failure handling,
                # so a tool that raised shows up as a span carrying an error
                # rather than as a gap where a tool call should have been.
                mark = len(toolbox.records)
                with get_tracing().span(f"tool.{name}", as_type=TOOL,
                                        input=args) as tool_span:
                    try:
                        content = tool.invoke(args)
                    except UnknownTemplateError as exc:
                        content = f"{NO_RESULTS}: {exc}"
                    except ValidationError as exc:
                        # The tool schema rejected the arguments before the tool
                        # ran -- graph_query's `template` is a Literal, so an
                        # injected Cypher string never reaches Neo4j. Tell the
                        # model what the legal values are so it can correct
                        # itself in one round.
                        content = (f"{NO_RESULTS}: invalid arguments for {name}: "
                                   f"{exc.error_count()} validation error(s). "
                                   f"For graph_query, `template` must be one of "
                                   f"{sorted(GRAPH_TEMPLATES)}.")
                    except Exception as exc:  # noqa: BLE001 - tools touch the network
                        log.warning("tool %s failed: %s", name, exc)
                        content = f"{NO_RESULTS}: tool {name} failed ({type(exc).__name__})."
                    tool_span.update(
                        output=content if isinstance(content, str) else str(content),
                        metadata={"n_new_records": len(toolbox.records) - mark,
                                  "no_results": isinstance(content, str)
                                  and content.startswith(NO_RESULTS)})
            out_messages.append(ToolMessage(
                content=content if isinstance(content, str) else json.dumps(content),
                tool_call_id=call.get("id", ""), name=name or "unknown"))

        return {
            "messages": out_messages,
            "retrieved": [r.to_dict() for r in toolbox.records[before_records:]],
            "abstention": toolbox.abstention[before_signals:],
            "tool_calls": toolbox.calls[before_calls:],
            "iterations": state.get("iterations", 0) + 1,
        }

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()


def run_agent(question: str, toolbox: Toolbox | None = None,
              settings: Settings | None = None, model: Any | None = None,
              history: list[BaseMessage] | None = None) -> AgentResponse:
    """Answer one question. Owns a Toolbox unless one is supplied."""
    s = settings or get_settings()
    owns = toolbox is None
    tb = toolbox or Toolbox(settings=s)
    t0 = time.perf_counter()
    tracing = get_tracing()
    try:
        usage: list[dict] = []
        app = build_agent_graph(tb, settings=s, model=model, usage_sink=usage)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt(s.agent_max_iterations))]
        messages.extend(history or [])
        messages.append(HumanMessage(content=question))

        # `agent`-typed so the LLM turns and tool calls below nest under one
        # node in the Langfuse tree instead of hanging off the request root.
        # run_agent is called via asyncio.to_thread, which copies the calling
        # context, so this span still attaches to the right request's trace.
        with tracing.span("agent", as_type=AGENT, input=question) as agent_span:
            final = app.invoke(
                {"messages": messages, "iterations": 0,
                 "max_iterations": s.agent_max_iterations},
                # One super-step per node visit; the real bound is the unbinding
                # in agent_node. This only stops a pathological graph spinning.
                {"recursion_limit": 2 * s.agent_max_iterations + 4},
            )
            agent_span.update(
                output={"iterations": final.get("iterations", 0),
                        "n_retrieved": len(final.get("retrieved", []) or [])},
                metadata={"tools_used": list(dict.fromkeys(
                    c["tool"] for c in final.get("tool_calls", []) or []))})
    finally:
        if owns:
            tb.close()

    retrieved = final.get("retrieved", [])
    answer = ""
    for m in reversed(final["messages"]):
        if isinstance(m, AIMessage) and m.content:
            answer = m.content if isinstance(m.content, str) else str(m.content)
            break

    iterations = final.get("iterations", 0)
    return AgentResponse(
        answer=answer,
        citations=build_citations(retrieved),
        tools_used=list(dict.fromkeys(c["tool"] for c in final.get("tool_calls", []))),
        tool_calls=final.get("tool_calls", []),
        retrieved=retrieved,
        abstention=summarize_abstention(final.get("abstention", [])),
        iterations=iterations,
        hit_iteration_limit=iterations >= s.agent_max_iterations,
        latency_ms=(time.perf_counter() - t0) * 1000,
        messages=final["messages"],
        llm_usage=usage,
    )
