# Chapter 08 — The Agent

## Why this / what's the need

A single retrieve-then-answer step cannot handle "which team owns the system Payment-Service
depends on, and who leads it?" — the second lookup depends on the result of the first. An
**agent** is an LLM in a loop: look at the question, choose a tool, read the result, choose
again, and eventually write the answer.

Two things can go wrong with agents, and this chapter makes both *structurally* impossible
rather than merely discouraged:

1. **The loop never ends.** A prompt saying "stop after four rounds" is a request the model
   may decline — an attacker can say "keep searching until you find it". Here, at the fourth
   round the model is re-invoked **with no tools bound**, so a fifth tool call is not
   representable.
2. **The model invents citations.** If citations came from the model's text, it could cite
   `POL-999`. Here the citation list is a *pure function of what the tools actually
   returned*; the model's output is not an argument to that function.

> 🔑 **New word — tool (function calling):** a function the LLM can ask to run by name with
> arguments. The model does not run it — the code does — and the result is fed back as a
> message.

> 🔑 **New word — LangGraph:** a library for writing LLM applications as a graph of nodes
> and edges over a shared state. Here it is two nodes and a loop.

---

## The three tools — `app/agent/tools.py`

```python
Three tools, one routing decision. `knowledge_search` already fuses vector and
graph, so the agent never has to choose a retrieval backend -- a choice it has
no information to make well. `graph_query` exists for the precise structural
questions fusion blurs ("who leads Infrastructure"), and `web_search` for what
the corpus provably does not contain.
```

| Tool | What it does | What it records |
|---|---|---|
| `knowledge_search(query)` | hybrid retrieval, `HYBRID_RERANK` mode | the 5 retrieved items + abstention signals |
| `graph_query(template, entity, max_depth)` | one of the 8 Cypher templates | the rendered rows as a graph fact |
| `web_search(query)` | Tavily, public web only | each result with its URL |

Deliberately absent, and the file says so:

```python
  * **sql_query** -- users.csv and products.csv *are* the graph. A SQL tool
    would query a duplicate copy of the same facts and let the agent answer one
    question two ways, with no way to adjudicate which answer is right.
  * **python_exec** -- excluded by instruction. `Toolbox.tools` is a plain list,
    so adding a sandboxed executor later is one entry, not a refactor.
```

### The ledger

```python
class Toolbox:
    def __init__(self, settings=None, retriever=None, graph_queries=None, tavily_client=None):
        ...
        self.records: list[RetrievedItem] = []
        self.abstention: list[dict] = []
        self.calls: list[dict] = []
```

- One `Toolbox` per request. `records` accumulates every `RetrievedItem` a tool actually
  returned; `abstention` accumulates the structural signals from Chapter 07; `calls` is the
  audit trail. Citations come from `records` and nowhere else.

### `graph_query` cannot run arbitrary Cypher

```python
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
```

- The tool's `template` argument is a `Literal` type. The model picks a *name*; the entity
  string reaches Neo4j as a bound parameter. If the model emits anything else, pydantic
  rejects the arguments before the tool runs, and the agent is told the legal values.

The two dependency templates describe their direction in opposite words, on purpose:

```python
    "dependency_cascade": ("DOWNSTREAM / reverse direction: which systems BREAK IF "
                           "the named system fails (systems that depend ON it). "
                           "Use for blast-radius and outage-impact questions."),
    "system_dependencies": ("UPSTREAM / forward direction: which systems the named "
                            "system DEPENDS ON (its own dependencies). Use for "
                            "'what does X depend on' and 'the system that X "
                            "depends on'."),
```

### A `NO_RESULTS` marker the model is taught to respect

```python
        if not rows:
            if resolved is None:
                return (f"{NO_RESULTS}: could not identify {entity!r} as a known "
                        f"entity for template {template}. Say so rather than guessing.")
            return (f"{NO_RESULTS}: {resolved} is known, but the graph holds no "
                    f"{template} rows for it. The truthful answer is that there "
                    f"are none -- do not substitute a plausible one.")
```

- Two different empties: "I could not identify that system" versus "that system exists and
  has no such fact". Both are recorded in `abstention` with different flags
  (`entity_unresolved` vs `entity_resolved_but_graph_empty`) because the grounding rail
  treats them differently.

## The system prompt — `app/agent/prompts.py`

The prompt is short and its most important section is titled honestly:

```python
HONESTY -- THE PART THAT MATTERS MOST
A tool result beginning with NO_RESULTS means the corpus genuinely does not \
contain that information. It is NOT an invitation to try harder with a \
rephrasing, and it is NOT a licence to fill the gap from your own background \
knowledge.
...
- Never invent document ids, incident numbers, SOP numbers, people or dates. \
If you did not see it in a tool result, it does not exist.
```

The module docstring is explicit that this is a *prompt*, not a guardrail: enforcement is
Chapter 09's job; the prompt just tells the model what a `NO_RESULTS` marker means.

## The graph — `app/agent/graph.py`

### State

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: Annotated[list[dict], _extend]
    abstention: Annotated[list[dict], _extend]
    tool_calls: Annotated[list[dict], _extend]
    iterations: int
    max_iterations: int
```

- The `Annotated[..., add_messages]` and `Annotated[..., _extend]` parts are **reducers**:
  when a node returns `{"messages": [x]}`, LangGraph *appends* rather than overwrites.
  Without a reducer, each step would replace the conversation and a tool's reply would have
  no matching call.

### Two nodes and a conditional edge

```python
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile()
```

```python
    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END
```

- If the model's last message contains tool calls, run them; otherwise we are done.

### The structural bound

```python
    base = model if model is not None else get_chat_model(
        Task.AGENT, temperature=s.agent_temperature, settings=settings)
    with_tools = base.bind_tools(tools)
    ...
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
```

- Two model handles: `base` (no tools) and `with_tools`. Below the limit the agent uses
  `with_tools`; at the limit it uses `base` and appends `FINAL_TURN_NUDGE` ("answer now
  using only the evidence already retrieved"). The `if getattr(reply, "tool_calls")` line
  is belt-and-braces: even if a provider somehow returned a tool call, it is stripped.
- Tested against a fake model that *always* calls a tool, 50 times: the loop ends at 4.

### The tool node ledgers everything

```python
    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out_messages: list[ToolMessage] = []
        before_records = len(toolbox.records)
        before_signals = len(toolbox.abstention)
        before_calls = len(toolbox.calls)

        for call in getattr(last, "tool_calls", []) or []:
            ...
        return {
            "messages": out_messages,
            "retrieved": [r.to_dict() for r in toolbox.records[before_records:]],
            "abstention": toolbox.abstention[before_signals:],
            "tool_calls": toolbox.calls[before_calls:],
            "iterations": state.get("iterations", 0) + 1,
        }
```

- It notes the ledger lengths before running the calls, then returns only the *new* slice.
  With the `_extend` reducer, state accumulates every tool call's provenance in order.

### Citations are a projection of the ledger

```python
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
            "doc_id": rec.get("doc_id"),
            "chunk_id": rec.get("chunk_id"),
            "graph_template": rec.get("graph_template"),
            "graph_path": rec.get("graph_path"),
            ...
        })
    return out
```

- Look at the signature: `build_citations(retrieved)`. There is no parameter through which
  the model's text could arrive. `tests/test_agent.py::test_citations_cannot_name_an_unretrieved_doc`
  has the model write `POL-999`, `INC-206` and `SOP-42` in its prose and asserts that the
  citation list contains only what was retrieved.

### Rolling up abstention for the rails

```python
def summarize_abstention(signals: list[dict]) -> dict:
    """Roll per-call abstention evidence into one dict for Phase 5.

    Deliberately no threshold and no verdict. ...
    """
    ...
    return {
        "any_tool_called": bool(signals),
        "negative_facts": negatives,
        "entity_resolved_but_graph_empty": any(
            s.get("entity_resolved_but_graph_empty") for s in signals),
        "empty_entities": sorted({...}),
        "entity_unresolved": any(s.get("entity_unresolved") for s in signals),
        "graph_declined": any(s.get("graph_declined") for s in signals),
        "max_rerank_score": max(scores) if scores else None,
        "no_results_returned": bool(signals) and all(
            s.get("returned_nothing") for s in signals),
        "per_call": signals,
    }
```

- `empty_entities` names *which* entities came back empty. Chapter 09 needs this to ask the
  only question that matters: does the answer assert something about one of them?

## Live behaviour (Phase 4, five questions, $0.057)

- The multi-hop question resolved correctly *and* reported "Notification-Service does not
  have an ownership record in the corpus" — Chapter 06's decision to return `[]` surfacing
  as an honest absence.
- "Parental leave policy" declined cleanly rather than confabulating from POL-004
  (`max_rerank_score` 0.00027).
- "What depends on DataWarehouse?" answered "nothing does".

One gap was recorded for the next chapter: a *decline* still carried citations, because the
ledger honestly records what was retrieved. Accurate as provenance, wrong as support.

## Model tiering — `app/llm/tiering.py`

```python
class Task(str, Enum):
    # --- fast tier: internal, structured, not read by a human ---
    CLASSIFY = "classify"
    ROUTE = "route"
    GROUNDING_CHECK = "grounding_check"   # Phase 5 seam
    SELF_CHECK = "self_check"             # Phase 5 seam
    SUMMARIZE_TOOL_OUTPUT = "summarize_tool_output"

    # --- smart tier: reasoning and user-visible prose ---
    AGENT = "agent"
    SYNTHESIS = "synthesis"
```

```python
def pick_model(task: Task | str, settings: Settings | None = None) -> str:
    """Return the OpenRouter model id for this task."""
    s = settings or get_settings()
    return s.llm_smart_model if tier_for(task) is Tier.SMART else s.llm_fast_model
```

- Callers name a *task*, never a model. The agent uses `Task.AGENT` → `gpt-4o`; the rails'
  yes/no checks use `Task.SELF_CHECK` → `gpt-4o-mini`. Unknown task strings resolve to
  the fast tier deliberately: an unregistered call site costing 15× more is a worse failure
  than one being slightly dumber.

`app/llm/client.py` builds the model with `streaming=False` — an output rail that inspects
a grounding claim cannot inspect tokens the user has already read.

---

## ✅ You just learned
- Three tools, a per-request ledger, and a `Literal`-typed template name that keeps Cypher
  out of the model's hands.
- The `agent ⟲ tools → END` graph, reducers, and the structural 4-iteration bound.
- Citations as a projection of the ledger — and the test that proves it.
- Task-based model tiering.

## ▶️ Run this now
```bash
.venv/bin/python -c "
from app.agent.graph import run_agent
r = run_agent('Which team owns the system Payment-Service depends on, and who leads it?')
print(r.answer); print(r.tools_used, r.iterations); print([c['citation'] for c in r.citations])
"
```
This costs a few cents. Expect ≥ 2 tool calls, Infrastructure / Marcus Lee, and citations
that name real chunk ids and graph templates.

## 🧠 Check yourself
1. Why is "unbind the tools" a stronger guarantee than "tell the model to stop"?
2. What is the only input to `build_citations`, and why does that matter?
3. What is the difference between `entity_unresolved` and `entity_resolved_but_graph_empty`?
4. Why does `tier_for` send an unknown task to the *fast* tier?

---

Next: the rails around the agent →
[09-guardrails.md](09-guardrails.md)
