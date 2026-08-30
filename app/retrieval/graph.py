"""Graph branch: resolve entities in a query, then render neighbourhood facts.

Two design points worth stating.

First, entity resolution DECLINES on ambiguous input by design (Phase 2). A
query like "our reporting requirements under POL-002" must not resolve to
ReportingPortal, so this branch returns nothing for it. That is the intended
outcome, not a failure: the vector branch still answers and fusion degrades to
vector-only. Nothing here raises or guesses.

Second, the graph is queried only through the seven parameterized templates.
No Cypher is ever assembled from query text, so a hostile query is at worst an
entity name that resolves to nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.retrieval.graph_queries import GraphQueries, resolve_system

# Which templates a query is asking for. Matched on whole words against the raw
# query; a template with no keyword hit still runs if it is a default, so an
# entity-only query ("Auth-DB") still returns its core facts.
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
    "policies_for_system": ("policy", "policies", "compliance", "governs",
                            "governing", "retention", "security", "acceptable"),
    "team_leadership": ("lead", "leads", "leader", "manager", "manages",
                        "head", "runs"),
    "team_roster": ("roster", "members", "member", "who is on", "team",
                    "people", "staff", "engineers"),
}

# Run these even with no keyword cue, so a bare entity mention is still useful.
DEFAULT_SYSTEM_TEMPLATES = ("system_ownership", "dependency_cascade")
DEFAULT_TEAM_TEMPLATES = ("team_leadership",)

_INCIDENT_RE = re.compile(r"\bINC[- ]?(\d{3})\b", re.I)
_SOP_RE = re.compile(r"\bSOP[- ]?(\d{2})\b", re.I)


@dataclass
class GraphFact:
    """One rendered fact block, ready to sit beside a document chunk."""

    text: str
    template: str
    path: str                     # human-readable provenance, e.g. "Auth-DB -[OWNS]- Team"
    entity: str
    # True when the query actually asked for this template (a cue word matched).
    # False means it ran only because it is a default for the resolved entity,
    # which makes it speculative context rather than an answer.
    cued: bool = True
    evidence_docs: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    # True when this fact asserts an ABSENCE ("nothing depends on X"). The graph
    # is the only component that can say the corpus does not contain something;
    # a cross-encoder scores topical relevance and will happily rank an on-topic
    # non-answer above a correct negative. Flagged so the signal survives rerank.
    negative: bool = False


@dataclass
class GraphContext:
    facts: list[GraphFact]
    resolved_system: str | None = None
    resolved_team: str | None = None
    resolved_incident: str | None = None
    declined: bool = False
    reason: str = ""


def _cue_score(query: str, cues: tuple[str, ...]) -> int:
    """How many distinct cue words the query hits. Whole-word matching only.

    Both boundaries are required. With only a leading boundary, "circuit
    breaker threshold" matched the cue "break" and fired the dependency-cascade
    template on a pure configuration question -- the same prefix-matching
    failure that the entity resolver was fixed for in Phase 2. Cue inflections
    are listed explicitly instead ("break", "breaks", "fail", "fails").
    """
    low = query.lower()
    return sum(1 for c in cues
               if re.search(rf"(?<![\w-]){re.escape(c)}(?![\w-])", low))


def _has_cue(query: str, cues: tuple[str, ...]) -> bool:
    return _cue_score(query, cues) > 0


def _rank_templates(query: str, names: tuple[str, ...],
                    defaults: tuple[str, ...]) -> list[tuple[str, bool]]:
    """Order templates by how strongly the query asks for them.

    Returns (template_name, cued) pairs. Branch rank feeds RRF, so this
    ordering is not cosmetic: putting the incident list above the dependency
    cascade for "what incidents affected Payment-Service" is what makes fusion
    rank the right fact first. The `cued` flag separates "the query asked for
    this" from "this is a default for the resolved entity" -- only the former
    is allowed to compete for the top of the fused list.
    """
    scored = [(name, _cue_score(query, TEMPLATE_CUES[name])) for name in names]
    ordered: list[tuple[str, bool]] = [
        (n, True) for n, sc in sorted(scored, key=lambda kv: -kv[1]) if sc > 0]
    chosen = {n for n, _ in ordered}
    for name in defaults:
        if name not in chosen:
            ordered.append((name, False))
    return ordered


def _fmt_people(rows: list[dict]) -> str:
    return "; ".join(
        f"{r['name']} ({r.get('role', '?')})" for r in rows if r.get("name"))


# --------------------------------------------------------------- renderers

def _render_ownership(system: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    r = rows[0]
    parts = [f"{r['system']} is owned by the {r['owner_team']} team."]
    if r.get("system_owner"):
        parts.append(f"Its named system owner is {r['system_owner']}.")
    if r.get("team_lead") and r["team_lead"] != r.get("system_owner"):
        parts.append(f"The {r['owner_team']} team is led by {r['team_lead']}.")
    return GraphFact(" ".join(parts), "system_ownership",
                     f"({system})<-[:OWNS]-(Team)-[:MANAGES]-(Person)", system,
                     rows=rows)


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
    by_hop: dict[int, list[str]] = {}
    for r in rows:
        by_hop.setdefault(r["hops"], []).append(r["system"])
    lines = [f"If {system} fails, these systems are affected "
             f"(they depend on it, directly or transitively):"]
    for hop in sorted(by_hop):
        lines.append(f"  - {hop} hop(s): {', '.join(sorted(by_hop[hop]))}")
    return GraphFact("\n".join(lines), "dependency_cascade",
                     f"(System)-[:DEPENDS_ON*1..n]->({system})", system, rows=rows)


def _render_incidents(system: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    lines = [f"Incidents recorded against {system}:"]
    for r in rows:
        sops = ", ".join(r.get("resolving_sops") or []) or "no SOP recorded"
        lines.append(f"  - {r['incident']} ({r['date']}, {r['dept']} team, "
                     f"reported by {r['author']}) resolved via {sops}")
    return GraphFact("\n".join(lines), "system_incidents",
                     f"(Document)-[:RELATED_TO]->({system})", system,
                     evidence_docs=[r["incident"] for r in rows], rows=rows)


def _render_policies(system: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    listed = "; ".join(f"{r['policy']} ({r['title']})" for r in rows)
    return GraphFact(
        f"Policies that govern {system}: {listed}.", "policies_for_system",
        f"(Document:policy)-[:RELATED_TO{{governs}}]->({system})", system,
        evidence_docs=[r["policy"] for r in rows], rows=rows)


def _render_leadership(team: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    r = rows[0]
    return GraphFact(
        f"The {r['team']} team is led by {r['lead']} ({r['role']}, {r['email']}).",
        "team_leadership", f"(Person)-[:MANAGES]->({team})", team, rows=rows)


def _render_roster(team: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    lead = [r for r in rows if r.get("role") == "Team Lead"]
    members = [r for r in rows if r.get("role") != "Team Lead"]
    lines = [f"The {team} team has {len(rows)} members."]
    if lead:
        lines.append(f"Team lead: {lead[0]['name']}.")
    if members:
        lines.append(f"Members: {_fmt_people(members)}.")
    return GraphFact(" ".join(lines), "team_roster",
                     f"(Person)<-[:MANAGES]-(Person)-[:MANAGES]->({team})", team,
                     rows=rows)


def _render_incident_sop(incident: str, rows: list[dict]) -> GraphFact | None:
    if not rows:
        return None
    r = rows[0]
    return GraphFact(
        f"{r['incident']} ({r['date']}, reported by {r['author']}) was resolved "
        f"by {r['sop']}: {r['sop_title']}.",
        "incident_sop", f"({incident})-[:RESOLVED_BY]->(SOP)", incident,
        evidence_docs=[r["incident"]], rows=rows)


# ------------------------------------------------------------------ branch

def retrieve_graph_context(query: str, gq: GraphQueries,
                           settings: Settings | None = None) -> GraphContext:
    """Resolve entities in `query` and render their 1-2 hop neighbourhood."""
    s = settings or get_settings()
    facts: list[GraphFact] = []

    sys_res = resolve_system(query)
    system = sys_res.value
    team = gq.resolve_team(query)
    incident_match = _INCIDENT_RE.search(query)
    incident = f"INC-{incident_match.group(1)}" if incident_match else None

    if not (system or team or incident):
        return GraphContext([], declined=True,
                            reason="no system, team or incident resolved from the "
                                   "query; entity resolution declined rather than "
                                   "guessing")

    def add(fact: GraphFact | None, cued: bool) -> None:
        if fact is not None:
            fact.cued = cued
            facts.append(fact)

    if incident:
        # An explicit INC-nnn in the query is always a direct ask.
        add(_render_incident_sop(incident, gq.incident_sop(incident)), True)

    if system:
        wanted = _rank_templates(
            query,
            ("policies_for_system", "system_incidents", "dependency_cascade",
             "system_ownership"),
            DEFAULT_SYSTEM_TEMPLATES)
        for name, cued in wanted:
            if name == "system_ownership":
                add(_render_ownership(system, gq.system_ownership(system)), cued)
            elif name == "dependency_cascade":
                add(_render_cascade(system, gq.dependency_cascade(system)), cued)
            elif name == "system_incidents":
                add(_render_incidents(system, gq.system_incidents(system)), cued)
            elif name == "policies_for_system":
                add(_render_policies(system, gq.policies_for_system(system)), cued)

    if team:
        wanted = _rank_templates(query, ("team_roster", "team_leadership"),
                                 DEFAULT_TEAM_TEMPLATES)
        for name, cued in wanted:
            if name == "team_leadership":
                add(_render_leadership(team, gq.team_leadership(team)), cued)
            elif name == "team_roster":
                add(_render_roster(team, gq.team_roster(team)), cued)

    # Drop renderers that had nothing to say, de-duplicate, and cap.
    seen: set[str] = set()
    kept: list[GraphFact] = []
    for f in facts:
        if f.template in seen:
            continue
        seen.add(f.template)
        kept.append(f)

    kept.sort(key=lambda f: not f.cued)   # stable: cued facts keep their order
    return GraphContext(kept[: s.graph_max_facts], resolved_system=system,
                        resolved_team=team, resolved_incident=incident,
                        declined=not kept,
                        reason="" if kept else "entities resolved but no template "
                                               "returned rows")
