"""Neo4j knowledge graph, built deterministically from the corpus.

Node labels : System, Team, Person, Document, SOP, Product
Relationships: DEPENDS_ON, OWNS, MANAGES, RESOLVED_BY, RELATED_TO

Everything here is derived from structured sources (users.csv, products.csv,
transactions.csv, doc_metadata.csv) or from *structurally parsed* prose. No LLM
is involved, so the graph is byte-identical across runs -- see
`extract_dependencies` for why the DEPENDS_ON prose did not need one.

Every system name passes through `normalize.canonicalize_system`. The graph is
where name fragmentation does the most damage: "Auth-DB" and "Auth-Db" as two
nodes would silently break every multi-hop query rather than raise an error.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from neo4j import Driver, GraphDatabase

from app.config import Settings, get_settings
from app.ingestion.loaders import load_corpus
from app.ingestion.normalize import (
    CANONICAL_SYSTEMS,
    UnknownSystemError,
    build_documents,
    canonicalize_system,
)

log = logging.getLogger(__name__)

NODE_LABELS = ("System", "Team", "Person", "Document", "SOP", "Product")
REL_TYPES = ("DEPENDS_ON", "OWNS", "MANAGES", "RESOLVED_BY", "RELATED_TO")

# One unique key per label -- these are what MERGE keys on, so they are what
# makes a rebuild idempotent rather than duplicating the graph.
CONSTRAINTS = {
    "System": "name",
    "Team": "name",
    "Person": "user_id",
    "Document": "doc_id",
    "SOP": "sop_id",
    "Product": "product_id",
}

LEAD_ROLE = "Team Lead"


# ---------------------------------------------------------------------------
# DEPENDS_ON extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DependencyEdge:
    source: str          # the system that depends
    target: str          # the system depended upon
    origin: str          # which parser produced it
    doc_id: str
    quote: str

    def key(self) -> tuple[str, str]:
        return (self.source, self.target)


# Four independent prose patterns. Each is anchored on an explicit dependency
# verb, never on mere co-occurrence.
_DEP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # manual_*.md, Architecture section:
    #   "Payment-Service depends on **Auth-DB** for all authentication ..."
    ("manual_prose", re.compile(
        r"\*{0,2}([\w.-]+)\*{0,2}\s+depends\s+on\s+\*{0,2}([\w.-]+)\*{0,2}", re.I)),
    # INC-*.md Summary:
    #   "cascading failures in **UserProfile-API** which depends\non Auth-DB for ..."
    ("incident_summary", re.compile(
        r"cascading failures in\s+\*{0,2}([\w.-]+)\*{0,2}\s+which depends\s+on\s+"
        r"\*{0,2}([\w.-]+)\*{0,2}", re.I | re.S)),
    # INC-*.md Root Cause:
    #   "UserProfile-API has a hard\ndependency on Auth-DB;"
    ("incident_rca", re.compile(
        r"\*{0,2}([\w.-]+)\*{0,2}\s+has a hard\s+dependency on\s+"
        r"\*{0,2}([\w.-]+)\*{0,2}", re.I | re.S)),
)

# manual_*.md Architecture diagram:
#     [Clients] -> [APIGateway] -> [Payment-Service] -> [ReportingPortal]
#                                           \-> [Auth-DB]
# An arrow means "calls / sends to", and calling a service is depending on it,
# so "A -> B" is "A DEPENDS_ON B". The opposite reading was tried first and is
# demonstrably wrong: it produced "Auth-DB depends on DataWarehouse" while the
# same file's prose says "DataWarehouse depends on **Auth-DB**". Reading the
# arrow as a call makes the diagram agree with the prose in all five manuals.
#
# CAVEAT, and the reason diagram-only edges are not trusted on their own: each
# manual's Overview also says the subject "provides ... downstream event
# emission to services including X and Y", where X and Y are exactly its
# diagram successors. Emission implies the OPPOSITE direction. The corpus is
# internally contradictory here, so the arrow reading is a defensible
# interpretation, not a settled fact. Only diagram edges corroborated by an
# explicit prose dependency statement are loaded (see LOADED_ORIGINS).
_DIAGRAM_CHAIN = re.compile(r"\[([^\]]+)\]\s*(?:->|→)")
_DIAGRAM_BOX = re.compile(r"\[([^\]]+)\]")
# A continuation line ("  \-> [Auth-DB]") branches off the subject system of
# the manual it appears in, since it is drawn hanging beneath it.
_DIAGRAM_BRANCH = re.compile(r"^\s*(?:↘|↗|\\->|\\)\s*\[([^\]]+)\]")

# Relative pronouns and articles that the loose "X depends on Y" pattern can
# capture from a wrapped sentence. Never system names; excluded so that
# `unknown_prose_names` stays a real signal instead of parser noise.
_NOT_A_SYSTEM = {"which", "that", "it", "they", "this", "these", "and", "the", "a"}

# FAQ.md: "Q: Which systems depend on Auth-DB? ... Dependent systems include:
#          - Payment-Service (Billing team - Priya Sharma)"
_FAQ_DEPENDENTS = re.compile(
    r"Which systems depend on\s+\*{0,2}([\w.-]+)\*{0,2}\s*\?(.*?)(?=\n\*\*Q:|\n## |\Z)",
    re.I | re.S)
_FAQ_BULLET = re.compile(r"^\s*-\s+\*{0,2}([\w.-]+?)\*{0,2}\s*(?:\(|$)", re.M)


def mentioned_systems(text: str) -> list[str]:
    """Canonical systems named verbatim in a body of text, whole-word only."""
    return [name for name in CANONICAL_SYSTEMS
            if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)]


def _try_canonical(name: str, doc_id: str, unknown: Counter) -> str | None:
    """Canonicalize a name found in prose, or return None if it is not a system.

    Prose is not authoritative the way doc_metadata.csv is -- it mentions
    "Clients", "InventoryEngine", and other non-systems -- so an unmapped name
    here is skipped and counted, not raised.
    """
    if name.strip().lower() in _NOT_A_SYSTEM:
        return None
    try:
        return canonicalize_system(name, strict=True)
    except UnknownSystemError:
        unknown[(doc_id, name.strip())] += 1
        return None


def extract_dependencies(documents: list[dict]) -> tuple[list[DependencyEdge], Counter]:
    """Parse DEPENDS_ON edges out of prose. Structural regex only, no LLM.

    An LLM fallback was scoped for whatever the parser could not reach, but
    every dependency statement in this corpus is templated and machine-regular:
    four fixed sentence forms plus an ASCII diagram. A model would add cost,
    latency and run-to-run nondeterminism to re-derive what a regex gets
    exactly, so none is used. `origin` records which parser produced each edge.
    """
    edges: list[DependencyEdge] = []
    unknown: Counter = Counter()

    for doc in documents:
        doc_id, text = doc["doc_id"], doc["text"]

        for origin, pattern in _DEP_PATTERNS:
            for match in pattern.finditer(text):
                src = _try_canonical(match.group(1), doc_id, unknown)
                tgt = _try_canonical(match.group(2), doc_id, unknown)
                if src and tgt:
                    edges.append(DependencyEdge(
                        src, tgt, origin, doc_id,
                        " ".join(match.group(0).split())[:200]))

        # Architecture diagram. The subject system anchors branch lines.
        subject = (doc.get("system_refs") or [None])[0]
        for line in text.split("\n"):
            branch = _DIAGRAM_BRANCH.match(line)
            if branch and subject:
                tgt = _try_canonical(branch.group(1), doc_id, unknown)
                if tgt:
                    edges.append(DependencyEdge(
                        subject, tgt, "manual_diagram_branch", doc_id,
                        line.strip()[:200]))
                continue
            if not _DIAGRAM_CHAIN.search(line):
                continue
            boxes = _DIAGRAM_BOX.findall(line)
            for left, right in zip(boxes, boxes[1:]):
                src = _try_canonical(left, doc_id, unknown)
                tgt = _try_canonical(right, doc_id, unknown)
                if src and tgt:
                    edges.append(DependencyEdge(
                        src, tgt, "manual_diagram_chain", doc_id,
                        line.strip()[:200]))

        # FAQ list of dependents (direction is inverted: the bullets depend on
        # the system named in the question).
        for match in _FAQ_DEPENDENTS.finditer(text):
            tgt = _try_canonical(match.group(1), doc_id, unknown)
            if not tgt:
                continue
            for bullet in _FAQ_BULLET.findall(match.group(2)):
                src = _try_canonical(bullet, doc_id, unknown)
                if src:
                    edges.append(DependencyEdge(
                        src, tgt, "faq_list", doc_id, f"{src} depends on {tgt}"))

    return edges, unknown


# Origins that assert a dependency in words. An edge supported ONLY by the
# ASCII diagram is a reading of an ambiguous picture (see the caveat above), so
# it is not loaded on its own.
EXPLICIT_ORIGINS = frozenset(
    {"manual_prose", "incident_summary", "incident_rca", "faq_list"})


def _confidence(origins: set[str]) -> str:
    explicit = origins & EXPLICIT_ORIGINS
    if len(explicit) >= 2:
        return "high"       # independently stated in two or more documents
    if explicit:
        return "medium"     # one explicit statement, possibly plus the diagram
    return "diagram_only"   # not loaded


def merge_dependency_edges(
    edges: list[DependencyEdge],
) -> tuple[list[dict], list[DependencyEdge], list[dict]]:
    """Collapse duplicate (source, target) pairs and drop self-loops.

    Returns (loaded_rows, self_loops, rejected_rows). A pair supported only by
    the Architecture diagram lands in `rejected`: the diagram contradicts each
    manual's own "downstream event emission" sentence, and trusting it alone
    produced "Auth-DB depends on ReportingPortal" -- the auth database
    depending on the reporting portal, while the FAQ calls Auth-DB "a
    foundational service".

    Self-loops are a real defect in this corpus, not a parsing artifact:
    manual_auth_db.md literally reads "Auth-DB depends on **Auth-DB**", and the
    APIGateway diagram reads "[APIGateway] -> [APIGateway]". Left in, they make
    the dependency-cascade query return the failed system as its own casualty.
    """
    self_loops = [e for e in edges if e.source == e.target]
    grouped: dict[tuple[str, str], list[DependencyEdge]] = {}
    for e in edges:
        if e.source == e.target:
            continue
        grouped.setdefault(e.key(), []).append(e)

    rows, rejected = [], []
    for (src, tgt), group in sorted(grouped.items()):
        origins = {e.origin for e in group}
        row = {
            "source": src,
            "target": tgt,
            "origins": sorted(origins),
            "evidence_docs": sorted({e.doc_id for e in group}),
            "quote": group[0].quote,
            "confidence": _confidence(origins),
        }
        (rows if origins & EXPLICIT_ORIGINS else rejected).append(row)
    return rows, self_loops, rejected


# ---------------------------------------------------------------------------
# Structured extraction
# ---------------------------------------------------------------------------

_MANUAL_HEADER = re.compile(r"\*\*(Owner Team|System Owner):\*\*\s*(.+)")

# FAQ.md: "ReportingPortal failures are usually caused by DataWarehouse schema
# drift (see INC-203). ... The Data Engineering team (Chen Wei) owns both
# systems." -- as templated as everything else the parser handles, and the only
# ownership statement in the corpus for ReportingPortal.
_FAQ_OWNS_BOTH = re.compile(
    r"\*{0,2}([\w.-]+)\*{0,2}\s+failures[^?]*?\bcaused by\s+\*{0,2}([\w.-]+)\*{0,2}"
    r"[^?]*?The\s+\*{0,2}([\w .&-]+?)\*{0,2}\s+team\s*\(\s*\*{0,2}([\w .-]+?)\*{0,2}\s*\)"
    r"\s+owns both systems",
    re.I | re.S)


def extract_system_ownership(documents: list[dict]) -> list[dict]:
    """Owner Team / System Owner, from manual headers and the FAQ statement."""
    out = []
    for doc in documents:
        if doc.get("doc_type") != "technical_manual" or not doc["system_refs"]:
            continue
        header = dict(_MANUAL_HEADER.findall(doc["text"].split("## Overview")[0]))
        out.append({
            "system": doc["system_refs"][0],
            "owner_team": (header.get("Owner Team") or "").strip() or None,
            "owner_person": (header.get("System Owner") or "").strip() or None,
            "doc_id": doc["doc_id"],
            "origin": "manual_header",
        })

    # Only ownership statement in the corpus for ReportingPortal.
    unknown: Counter = Counter()
    for doc in documents:
        for match in _FAQ_OWNS_BOTH.finditer(doc["text"]):
            team = match.group(3).strip()
            person = match.group(4).strip()
            for raw_name in (match.group(1), match.group(2)):
                system = _try_canonical(raw_name, doc["doc_id"], unknown)
                if system:
                    out.append({"system": system, "owner_team": team,
                                "owner_person": person, "doc_id": doc["doc_id"],
                                "origin": "faq_prose"})
    # First statement wins; manual headers are listed first and are the more
    # authoritative source, so they take precedence over the FAQ.
    deduped: dict[str, dict] = {}
    for row in out:
        deduped.setdefault(row["system"], row)
    return list(deduped.values())


@dataclass
class GraphStats:
    nodes: dict[str, int] = field(default_factory=dict)
    relationships: dict[str, int] = field(default_factory=dict)
    dependency_rows: list[dict] = field(default_factory=list)
    rejected_dependency_rows: list[dict] = field(default_factory=list)
    self_loops: list[DependencyEdge] = field(default_factory=list)
    unmatched_departments: Counter = field(default_factory=Counter)
    orphan_documents: list[str] = field(default_factory=list)
    unknown_prose_names: Counter = field(default_factory=Counter)
    systems_without_owner: list[str] = field(default_factory=list)

    @property
    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    @property
    def total_relationships(self) -> int:
        return sum(self.relationships.values())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def get_driver(settings: Settings | None = None) -> Driver:
    s = settings or get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def create_constraints(session) -> None:
    for label, key in CONSTRAINTS.items():
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_{key}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )


def reset_graph(session) -> int:
    """Delete every node this builder owns, returning how many it deleted.

    Scoped to NODE_LABELS, and the count is scoped the same way -- reporting a
    database-wide total while deleting only our own labels would overstate the
    teardown in any database that also holds something else.
    """
    labels = "|".join(NODE_LABELS)
    owned = session.run(
        f"MATCH (n:{labels}) RETURN count(n) AS c").single()["c"]
    session.run(f"MATCH (n:{labels}) DETACH DELETE n")
    return owned


def _count_nodes(session) -> dict[str, int]:
    return {
        label: session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        for label in NODE_LABELS
    }


def _count_relationships(session) -> dict[str, int]:
    return {
        rel: session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()["c"]
        for rel in REL_TYPES
    }


def build_graph(settings: Settings | None = None, driver: Driver | None = None,
                reset: bool = False) -> GraphStats:
    """Build the whole graph. Idempotent: every write is a MERGE on a unique key."""
    s = settings or get_settings()
    raw_dir: Path = s.raw_dir
    owns_driver = driver is None
    driver = driver or get_driver(s)

    documents = build_documents(load_corpus(raw_dir, strict=True),
                                raw_dir / "doc_metadata.csv")
    users = pd.read_csv(raw_dir / "users.csv")
    products = pd.read_csv(raw_dir / "products.csv")

    dep_edges, unknown = extract_dependencies(documents)
    dep_rows, self_loops, rejected_deps = merge_dependency_edges(dep_edges)
    ownership = extract_system_ownership(documents)

    stats = GraphStats(dependency_rows=dep_rows, self_loops=self_loops,
                       rejected_dependency_rows=rejected_deps,
                       unknown_prose_names=unknown)

    try:
        with driver.session() as session:
            create_constraints(session)
            if reset:
                reset_graph(session)

            # --- Systems (all 7 canonical, whether or not prose mentions them)
            session.run(
                "UNWIND $rows AS r MERGE (s:System {name: r.name})",
                rows=[{"name": n} for n in CANONICAL_SYSTEMS])

            # --- Teams and People
            session.run(
                "UNWIND $rows AS r MERGE (t:Team {name: r.name})",
                rows=[{"name": t} for t in sorted(users["team"].unique())])
            session.run("""
                UNWIND $rows AS r
                MERGE (p:Person {user_id: r.user_id})
                SET p.name = r.name, p.email = r.email, p.team = r.team,
                    p.role = r.role, p.active = r.active, p.created_date = r.created_date
            """, rows=[{
                "user_id": r["user_id"], "name": r["name"], "email": r["email"],
                "team": r["team"], "role": r["role"],
                "active": str(r["active"]).strip().lower() == "true",
                "created_date": str(r["created_date"]),
            } for r in users.to_dict("records")])

            # --- MANAGES: lead -> their team, and lead -> each team member
            leads = users[users["role"] == LEAD_ROLE]
            session.run("""
                UNWIND $rows AS r
                MATCH (p:Person {user_id: r.user_id}), (t:Team {name: r.team})
                MERGE (p)-[m:MANAGES]->(t)
                SET m.role = 'Team Lead'
            """, rows=leads[["user_id", "team"]].to_dict("records"))

            lead_by_team = dict(zip(leads["team"], leads["user_id"]))
            member_rows = [
                {"lead_id": lead_by_team[r["team"]], "member_id": r["user_id"]}
                for r in users.to_dict("records")
                if r["role"] != LEAD_ROLE and r["team"] in lead_by_team
            ]
            session.run("""
                UNWIND $rows AS r
                MATCH (l:Person {user_id: r.lead_id}), (m:Person {user_id: r.member_id})
                MERGE (l)-[:MANAGES]->(m)
            """, rows=member_rows)

            # --- Products, and the teams that own them
            session.run("""
                UNWIND $rows AS r
                MERGE (p:Product {product_id: r.product_id})
                SET p.name = r.name, p.category = r.category, p.price_usd = r.price_usd
                WITH p, r
                MATCH (t:Team {name: r.owner_team})
                MERGE (t)-[:OWNS]->(p)
            """, rows=products.to_dict("records"))

            # transactions.csv is deliberately NOT loaded. A Person-purchased->
            # Product edge per (user, product) pair would add 150 edges that no
            # query template reads, and the groupby that produced them discarded
            # `status` and `payment_system` -- the two fields behind the only
            # transaction question an ops assistant is actually asked ("how many
            # transactions failed on Payment-Service?"), which the vector index
            # already answers from the rendered ledger summary.

            # --- System ownership, from the manual header blocks
            # MATCH, never MERGE, the team: a typo in a manual header must not
            # invent a person-less Team node that then shows up in rosters.
            owner_rows = [r for r in ownership if r["owner_team"]]
            session.run("""
                UNWIND $rows AS r
                MATCH (s:System {name: r.system}), (t:Team {name: r.owner_team})
                MERGE (t)-[:OWNS]->(s)
            """, rows=owner_rows)
            known_teams = {rec["n"] for rec in
                           session.run("MATCH (t:Team) RETURN t.name AS n")}
            for row in owner_rows:
                if row["owner_team"] not in known_teams:
                    log.warning("manual %s names owner team %r, which is not in "
                                "users.csv -- OWNS edge skipped",
                                row["doc_id"], row["owner_team"])
            session.run("""
                UNWIND $rows AS r
                MATCH (s:System {name: r.system}), (p:Person {name: r.owner_person})
                MERGE (p)-[:OWNS]->(s)
            """, rows=[r for r in ownership if r["owner_person"]])

            # --- Documents and SOPs
            session.run("""
                UNWIND $rows AS r
                MERGE (d:Document {doc_id: r.doc_id})
                SET d.filename = r.filename, d.doc_type = r.doc_type, d.dept = r.dept,
                    d.date = r.date, d.author = r.author, d.title = r.title
            """, rows=[{k: doc[k] for k in
                        ("doc_id", "filename", "doc_type", "dept", "date", "author", "title")}
                       for doc in documents])

            sop_ids = sorted({sop for doc in documents for sop in doc["sop_refs"]})
            session.run("UNWIND $rows AS r MERGE (s:SOP {sop_id: r.sop_id})",
                        rows=[{"sop_id": x} for x in sop_ids])
            # An SOP's own document carries its title.
            session.run("""
                UNWIND $rows AS r
                MATCH (s:SOP {sop_id: r.sop_id})
                SET s.title = r.title, s.doc_id = r.doc_id
            """, rows=[{"sop_id": doc["doc_id"], "title": doc["title"],
                        "doc_id": doc["doc_id"]}
                       for doc in documents if doc["doc_type"] == "sop"])

            # Document -> System, and Document -> its department team
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (s:System {name: r.system})
                MERGE (d)-[:RELATED_TO {kind: 'system'}]->(s)
            """, rows=[{"doc_id": doc["doc_id"], "system": sys}
                       for doc in documents for sys in doc["system_refs"]])
            # doc_metadata.csv's dept is either one of the 6 team names or the
            # literal "All". "All" matched no Team, so this MATCH silently
            # dropped 15 of 25 documents -- every policy and SOP -- leaving the
            # compliance backbone unreachable from the graph. Corpus-wide
            # documents now attach to every team as 'applies_to', and any dept
            # that matches nothing is counted and reported rather than lost.
            team_names = {rec["n"] for rec in
                          session.run("MATCH (t:Team) RETURN t.name AS n")}
            dept_rows, applies_rows = [], []
            for doc in documents:
                dept = (doc.get("dept") or "").strip()
                if dept in team_names:
                    dept_rows.append({"doc_id": doc["doc_id"], "dept": dept})
                elif dept.lower() in {"all", "company-wide", "companywide"}:
                    applies_rows += [{"doc_id": doc["doc_id"], "team": t}
                                     for t in sorted(team_names)]
                else:
                    stats.unmatched_departments[dept or "(blank)"] += 1
                    log.warning("document %s has dept %r which matches no Team",
                                doc["doc_id"], dept)
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (t:Team {name: r.dept})
                MERGE (d)-[:RELATED_TO {kind: 'department'}]->(t)
            """, rows=dept_rows)
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (t:Team {name: r.team})
                MERGE (d)-[:RELATED_TO {kind: 'applies_to'}]->(t)
            """, rows=applies_rows)

            # Policies name the systems they cover in their Scope section, so a
            # policy is reachable from a system ("which policy governs access to
            # Auth-DB?"). doc_metadata.csv leaves system_refs blank for all four.
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (s:System {name: r.system})
                MERGE (d)-[:RELATED_TO {kind: 'governs'}]->(s)
            """, rows=[{"doc_id": doc["doc_id"], "system": sys}
                       for doc in documents if doc["doc_type"] == "policy"
                       for sys in mentioned_systems(doc["text"])])

            # RESOLVED_BY is specifically incident -> the SOP that resolved it.
            # Every other doc/SOP link is a reference, not a resolution.
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (s:SOP {sop_id: r.sop_id})
                MERGE (d)-[:RESOLVED_BY]->(s)
            """, rows=[{"doc_id": doc["doc_id"], "sop_id": sop}
                       for doc in documents if doc["doc_type"] == "incident_report"
                       for sop in doc["sop_refs"]])
            session.run("""
                UNWIND $rows AS r
                MATCH (d:Document {doc_id: r.doc_id}), (s:SOP {sop_id: r.sop_id})
                MERGE (d)-[:RELATED_TO {kind: 'sop'}]->(s)
            """, rows=[{"doc_id": doc["doc_id"], "sop_id": sop}
                       for doc in documents if doc["doc_type"] != "incident_report"
                       for sop in doc["sop_refs"]])

            # Authorship, where the author is a real person in users.csv
            session.run("""
                UNWIND $rows AS r
                MATCH (p:Person {name: r.author}), (d:Document {doc_id: r.doc_id})
                MERGE (p)-[:OWNS]->(d)
            """, rows=[{"author": doc["author"], "doc_id": doc["doc_id"]}
                       for doc in documents if doc["author"]])

            # --- DEPENDS_ON, from the prose parsers
            session.run("""
                UNWIND $rows AS r
                MATCH (a:System {name: r.source}), (b:System {name: r.target})
                MERGE (a)-[d:DEPENDS_ON]->(b)
                SET d.origins = r.origins, d.evidence_docs = r.evidence_docs,
                    d.quote = r.quote, d.extraction = 'parser',
                    d.confidence = r.confidence
            """, rows=dep_rows)

            stats.nodes = _count_nodes(session)
            stats.relationships = _count_relationships(session)
            stats.orphan_documents = [
                rec["id"] for rec in session.run(
                    "MATCH (d:Document) WHERE NOT (d)--() "
                    "RETURN d.doc_id AS id ORDER BY id")
            ]
            stats.systems_without_owner = [
                rec["name"] for rec in session.run(
                    "MATCH (s:System) WHERE NOT (:Team)-[:OWNS]->(s) "
                    "RETURN s.name AS name ORDER BY name")
            ]
    finally:
        if owns_driver:
            driver.close()

    return stats
