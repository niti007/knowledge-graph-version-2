"""The six parameterized Cypher templates, plus entity resolution.

These are the ONLY queries the agent is permitted to run. Nothing here ever
interpolates model output into Cypher: every template takes bound parameters,
so a prompt-injected "; MATCH (n) DETACH DELETE n" arrives as a string that
matches no node instead of as syntax. Free-form LLM-generated Cypher is
deliberately not supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from neo4j import Driver

from app.config import Settings, get_settings
from app.ingestion.graph_builder import get_driver
from app.ingestion.normalize import CANONICAL_SYSTEMS, canonicalize_system

# Depth ceiling for the cascade walk. Bounded because the real dependency graph
# contains cycles (see resolve/cascade notes), so an unbounded walk would only
# re-derive the same systems through longer paths.
MAX_CASCADE_DEPTH = 5

# Words that carry no identity: "the payments API" and "payments" name the same
# system, and "team"/"the" appear in most natural phrasings.
# Interrogatives and framing verbs are stripped too, so an ordinary question
# ("who owns the reporting portal") reduces to its entity and resolves, while a
# sentence with real content words beside the alias ("our reporting requirements
# under POL-002") keeps them and is correctly declined by the length guard.
_NOISE_WORDS = {"the", "a", "an", "our", "system", "service", "svc", "api",
                "db", "database", "team", "platform", "please", "of",
                "who", "what", "which", "whom", "whose", "when", "where", "why",
                "how", "is", "are", "was", "were", "do", "does", "did", "can",
                "tell", "me", "about", "owns", "own", "leads", "lead", "manages",
                "runs", "for", "on", "in", "to"}

# Hand-written aliases for phrasings that stemming alone will not reach.
SYSTEM_ALIASES: dict[str, str] = {
    "payments": "Payment-Service",
    "payment": "Payment-Service",
    "billing system": "Payment-Service",
    "auth": "Auth-DB",
    "authentication": "Auth-DB",
    "authdb": "Auth-DB",
    "login": "Auth-DB",
    "user profile": "UserProfile-API",
    "userprofile": "UserProfile-API",
    "profiles": "UserProfile-API",
    "gateway": "APIGateway",
    "warehouse": "DataWarehouse",
    "analytics warehouse": "DataWarehouse",
    "notifications": "Notification-Service",
    "notification": "Notification-Service",
    "reporting": "ReportingPortal",
    "reports": "ReportingPortal",
    "portal": "ReportingPortal",
}


def _key(text: str) -> str:
    """Lowercase, drop punctuation and noise words, collapse spaces."""
    words = re.sub(r"[^a-z0-9]+", " ", text.lower()).split()
    kept = [w for w in words if w not in _NOISE_WORDS]
    return " ".join(kept or words)


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


@dataclass(frozen=True)
class Resolution:
    """A resolution attempt. `value is None` means "decline", not "no match" --
    callers are expected to say they could not identify the entity rather than
    query a guess."""

    value: str | None
    matched_by: str          # canonical | alias | token | none
    confidence: float = 0.0

    def __bool__(self) -> bool:
        return self.value is not None


def _tokens(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def _contains_phrase(haystack: list[str], needle: list[str]) -> bool:
    """Whole-token subsequence match: ['the','gateway'] contains ['gateway'],
    but ['gateway','drug'] does NOT contain ['gateway'] as an entity mention...
    that distinction is made by the caller; this is strict token equality so
    that 'reporting' never matches inside 'requirements'."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(haystack[i:i + len(needle)] == needle
               for i in range(len(haystack) - len(needle) + 1))


def resolve_system(name: str) -> Resolution:
    """Resolve free text to one of the 7 canonical System names, or decline.

    Matching is strictly on whole tokens. An earlier version tested
    `if alias in key` on raw substrings, which is confidently wrong on ordinary
    phrasings the agent will pass through verbatim in Phase 4:

        "our reporting requirements under POL-002"  -> ReportingPortal
        "what is the login policy for contractors"  -> Auth-DB
        "gateway drug"                              -> APIGateway

    Those now resolve to None. An alias only counts when it accounts for
    essentially the whole phrase: a generic token ("gateway", "login") or even a
    short alias phrase ("billing system") buried in a longer sentence yields
    Resolution(None, "none", 0.2) -- the candidate is deliberately DISCARDED,
    not returned, so a caller that reads only .value cannot act on a guess.
    An exact canonical name is 1.0.
    """
    if not name or not name.strip():
        return Resolution(None, "none")

    try:
        return Resolution(canonicalize_system(name, strict=True), "canonical", 1.0)
    except Exception:
        pass

    tokens = _tokens(name)
    content = [t for t in tokens if t not in _NOISE_WORDS] or tokens

    # Exact match on the content tokens, e.g. "the payments API" -> ["payments"].
    joined = " ".join(content)
    if joined in SYSTEM_ALIASES:
        return Resolution(SYSTEM_ALIASES[joined], "alias", 0.9)

    # Squashed whole-string equality, e.g. "userprofileapi".
    # Squash the noise-stripped content, not the raw string: _squash("the
    # reporting portal") is "thereportingportal", which matches nothing, so
    # the most natural phrasing of two of the seven systems was declining.
    squashed = _squash(joined)
    for canonical in CANONICAL_SYSTEMS:
        if squashed == _squash(canonical):
            return Resolution(canonical, "canonical", 1.0)

    # A canonical name appearing as a whole token run inside a longer phrase,
    # e.g. "restart the Payment-Service now".
    for canonical in CANONICAL_SYSTEMS:
        if _contains_phrase(tokens, _tokens(canonical)):
            return Resolution(canonical, "token", 0.8)

    # An alias phrase appearing as a whole token run. Multi-word alias phrases
    # ("user profile", "reporting portal") are specific enough to trust; a bare
    # single generic word ("reporting", "login", "gateway") inside a longer
    # sentence is NOT, and is reported at low confidence.
    for alias, canonical in sorted(SYSTEM_ALIASES.items(),
                                   key=lambda kv: -len(kv[0].split())):
        alias_tokens = _tokens(alias)
        if not _contains_phrase(tokens, alias_tokens):
            continue
        # Trust the alias only when it accounts for essentially the whole
        # phrase. A 2-token alias buried in a 12-token sentence is no more an
        # entity mention than a 1-token one -- "the billing system upgrade was
        # discussed at the all-hands" is not a question about Payment-Service.
        if len(content) <= len(alias_tokens):
            return Resolution(canonical, "alias", 0.9 if len(alias_tokens) > 1 else 0.75)
        return Resolution(None, "none", 0.2)

    return Resolution(None, "none")


class GraphQueries:
    """The six templates. Each returns a list of plain dicts."""

    def __init__(self, driver: Driver | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._driver = driver
        self._owns_driver = driver is None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = get_driver(self.settings)
        return self._driver

    def close(self) -> None:
        if self._driver is not None and self._owns_driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "GraphQueries":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _run(self, cypher: str, **params) -> list[dict]:
        with self.driver.session() as session:
            return [dict(r) for r in session.run(cypher, **params)]

    # ---------------------------------------------------------------- 1
    SYSTEM_OWNERSHIP = """
        MATCH (t:Team)-[:OWNS]->(s:System {name: $system})
        OPTIONAL MATCH (p:Person)-[:OWNS]->(s)
        OPTIONAL MATCH (lead:Person)-[:MANAGES]->(t)
        RETURN s.name AS system, t.name AS owner_team,
               p.name AS system_owner, lead.name AS team_lead
    """

    def system_ownership(self, system: str) -> list[dict]:
        """Which team owns system X (and who is its named owner / team lead).

        Returns [] when the corpus names no owner. It previously returned one
        row with every field null, which reads as an authoritative "nobody owns
        this" -- and for ReportingPortal that directly contradicted the FAQ
        chunk in the vector index, handing Phase 3 fusion two opposing contexts.
        An empty result lets the caller fall back to retrieval instead.
        """
        r = resolve_system(system)
        if not r:
            return []
        return self._run(self.SYSTEM_OWNERSHIP, system=r.value)

    # ---------------------------------------------------------------- 2
    TEAM_LEADERSHIP = """
        MATCH (lead:Person)-[:MANAGES]->(t:Team {name: $team})
        RETURN t.name AS team, lead.name AS lead, lead.email AS email,
               lead.user_id AS user_id, lead.role AS role
    """

    def team_leadership(self, team: str) -> list[dict]:
        """Who leads team X."""
        resolved = self.resolve_team(team)
        return self._run(self.TEAM_LEADERSHIP, team=resolved) if resolved else []

    # ---------------------------------------------------------------- 3
    # Neo4j will not bind a variable-length bound to a parameter, so the ceiling
    # is interpolated ONCE from the module constant -- never from user input --
    # and the per-call depth stays a bound parameter.
    DEPENDENCY_CASCADE = """
        MATCH path = (dependent:System)-[:DEPENDS_ON*1..%(max)d]->(failed:System {name: $system})
        WHERE length(path) <= $max_depth
          AND ALL(n IN nodes(path) WHERE size([m IN nodes(path) WHERE m = n]) = 1)
        WITH dependent, min(length(path)) AS hops
        RETURN dependent.name AS system, hops
        ORDER BY hops, system
    """ % {"max": MAX_CASCADE_DEPTH}

    # ---------------------------------------------------------------- 8
    SYSTEM_DEPENDENCIES = """
        MATCH path = (s:System {name: $system})-[:DEPENDS_ON*1..%(max)d]->(dep:System)
        WHERE length(path) <= $max_depth
          AND ALL(n IN nodes(path) WHERE size([m IN nodes(path) WHERE m = n]) = 1)
        RETURN DISTINCT dep.name AS system, min(length(path)) AS hops
        ORDER BY hops, system
    """ % {"max": MAX_CASCADE_DEPTH}

    def system_dependencies(self, system: str, max_depth: int = 3) -> list[dict]:
        """What system X depends ON -- the forward direction.

        This is the inverse of dependency_cascade and exists because its absence
        was a real correctness hole. Asked "which team owns the system that
        Payment-Service depends on", the agent reached for dependency_cascade --
        the closest-named tool -- and got Auth-DB back. That looked correct only
        because Payment-Service and Auth-DB happen to depend on each other; the
        true answer is {Auth-DB, Notification-Service}, and Notification-Service
        was silently dropped. A wrong-direction template that returns a
        plausible row is invisible to every abstention signal we have, so the
        fix has to be a template that actually answers the question.
        """
        r = resolve_system(system)
        if not r:
            return []
        depth = max(1, min(int(max_depth), MAX_CASCADE_DEPTH))
        return self._run(self.SYSTEM_DEPENDENCIES, system=r.value, max_depth=depth)

    def dependency_cascade(self, system: str, max_depth: int = 3) -> list[dict]:
        """What breaks if system X fails.

        Walks DEPENDS_ON *backwards*: A DEPENDS_ON B means A breaks when B does.
        The path is required to have no repeated node because the real graph is
        cyclic (Payment-Service and Auth-DB each depend on the other, per the
        manuals and INC-204 respectively), and without that guard a cycle
        reports a system as a casualty of its own outage.
        """
        r = resolve_system(system)
        if not r:
            return []
        depth = max(1, min(int(max_depth), MAX_CASCADE_DEPTH))
        return self._run(self.DEPENDENCY_CASCADE, system=r.value, max_depth=depth)

    # ---------------------------------------------------------------- 4
    INCIDENT_SOP = """
        MATCH (d:Document {doc_id: $incident})-[:RESOLVED_BY]->(s:SOP)
        OPTIONAL MATCH (sop_doc:Document {doc_id: s.sop_id})
        RETURN d.doc_id AS incident, d.date AS date, d.author AS author,
               s.sop_id AS sop, coalesce(s.title, sop_doc.title) AS sop_title
    """

    def incident_sop(self, incident_id: str) -> list[dict]:
        """Which SOP resolved incident X."""
        return self._run(self.INCIDENT_SOP, incident=incident_id.strip().upper())

    # ---------------------------------------------------------------- 5
    SYSTEM_INCIDENTS = """
        MATCH (d:Document)-[:RELATED_TO {kind: 'system'}]->(s:System {name: $system})
        WHERE d.doc_type = 'incident_report'
        OPTIONAL MATCH (d)-[:RESOLVED_BY]->(sop:SOP)
        RETURN d.doc_id AS incident, d.date AS date, d.dept AS dept,
               d.author AS author, collect(DISTINCT sop.sop_id) AS resolving_sops
        ORDER BY date
    """

    def system_incidents(self, system: str) -> list[dict]:
        """Which incidents involved system X."""
        r = resolve_system(system)
        return self._run(self.SYSTEM_INCIDENTS, system=r.value) if r else []

    # ---------------------------------------------------------------- 7
    POLICIES_FOR_SYSTEM = """
        MATCH (d:Document)-[:RELATED_TO {kind: 'governs'}]->(s:System {name: $system})
        RETURN d.doc_id AS policy, d.title AS title, d.doc_type AS doc_type,
               d.dept AS dept, d.date AS date
        ORDER BY policy
    """

    def policies_for_system(self, system: str) -> list[dict]:
        """Which policies govern system X.

        Without this the 16 'governs' edges are unreachable through the template
        API, which is the same criticism that retired the purchase edges: graph
        content no query reads. Compliance questions route here.
        """
        r = resolve_system(system)
        return self._run(self.POLICIES_FOR_SYSTEM, system=r.value) if r else []

    # ---------------------------------------------------------------- 6
    TEAM_ROSTER = """
        MATCH (lead:Person)-[:MANAGES]->(t:Team {name: $team})
        OPTIONAL MATCH (lead)-[:MANAGES]->(member:Person)
        WITH t, lead, collect(DISTINCT member) AS members
        UNWIND ([lead] + members) AS person
        RETURN t.name AS team, person.name AS name, person.role AS role,
               person.email AS email, person.user_id AS user_id,
               person.active AS active
        ORDER BY CASE WHEN role = 'Team Lead' THEN 0 ELSE 1 END, name
    """

    def team_roster(self, team: str) -> list[dict]:
        """Who is on team X, lead first."""
        resolved = self.resolve_team(team)
        return self._run(self.TEAM_ROSTER, team=resolved) if resolved else []

    # ------------------------------------------------------------ helpers
    def team_names(self) -> list[str]:
        return [r["name"] for r in
                self._run("MATCH (t:Team) RETURN t.name AS name ORDER BY name")]

    def resolve_team(self, name: str) -> str | None:
        """Resolve free text to a Team node name, against the live graph."""
        if not name or not name.strip():
            return None
        teams = self.team_names()
        key = _squash(name)
        for t in teams:
            if _squash(t) == key:
                return t
        wanted = _key(name)
        for t in teams:
            if _key(t) == wanted:
                return t
        for t in teams:
            ts = _squash(t)
            if key and (key in ts or ts in key):
                return t
        return None


# The complete template registry. Every entry here is reachable from the agent
# (app.agent.tools.GRAPH_TEMPLATES mirrors these names) and every name is
# asserted to be a real method by
# tests/test_agent.py::test_template_registry_matches_the_methods_that_exist --
# this dict silently omitted policies_for_system for a phase, which made the 16
# 'governs' edges look unreachable to anything reading the registry.
TEMPLATES = {
    "system_ownership": GraphQueries.system_ownership,
    "team_leadership": GraphQueries.team_leadership,
    "dependency_cascade": GraphQueries.dependency_cascade,
    "system_dependencies": GraphQueries.system_dependencies,
    "incident_sop": GraphQueries.incident_sop,
    "system_incidents": GraphQueries.system_incidents,
    "team_roster": GraphQueries.team_roster,
    "policies_for_system": GraphQueries.policies_for_system,
}
