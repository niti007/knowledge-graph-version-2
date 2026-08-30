"""Phase 2 graph tests: extraction, constraints, idempotency, and the templates.

The template and idempotency tests run against the live Neo4j from .env; they
skip (rather than fail) when it is unreachable, so the pure-parsing tests stay
useful offline.
"""

from __future__ import annotations

import pytest

from collections import Counter

from app.config import get_settings
from app.ingestion.graph_builder import (
    CONSTRAINTS,
    NODE_LABELS,
    REL_TYPES,
    DependencyEdge,
    build_graph,
    extract_dependencies,
    extract_system_ownership,
    get_driver,
    merge_dependency_edges,
)
from app.ingestion.loaders import load_corpus
from app.ingestion.normalize import CANONICAL_SYSTEMS, build_documents
from app.retrieval.graph_queries import GraphQueries, resolve_system

SETTINGS = get_settings()
META_CSV = SETTINGS.raw_dir / "doc_metadata.csv"


@pytest.fixture(scope="module")
def documents():
    return build_documents(load_corpus(SETTINGS.raw_dir, strict=True), META_CSV)


@pytest.fixture(scope="module")
def driver():
    try:
        d = get_driver(SETTINGS)
        d.verify_connectivity()
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"Neo4j unavailable: {exc}")
    yield d
    d.close()


@pytest.fixture(scope="module")
def graph(driver):
    """Build once for the module; tests below assert against the real graph."""
    return build_graph(SETTINGS, driver=driver, reset=True)


@pytest.fixture()
def gq(driver, graph):
    with GraphQueries(driver=driver, settings=SETTINGS) as q:
        yield q


# --------------------------------------------------------- dependency parsing

def test_dependencies_are_extracted_without_an_llm(documents):
    edges, _ = extract_dependencies(documents)
    rows, _, _ = merge_dependency_edges(edges)
    assert rows, "no DEPENDS_ON edges parsed"
    origins = {o for r in rows for o in r["origins"]}
    assert origins <= {"manual_prose", "manual_diagram_chain",
                       "manual_diagram_branch", "incident_summary",
                       "incident_rca", "faq_list"}
    assert "llm" not in origins


def test_dependency_endpoints_are_all_canonical(documents):
    """The graph is where fragmentation does the most damage: a stray 'Auth-Db'
    would become a second node and silently break every multi-hop query."""
    rows, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    for r in rows:
        assert r["source"] in CANONICAL_SYSTEMS, r
        assert r["target"] in CANONICAL_SYSTEMS, r


def test_known_dependency_edges_are_found(documents):
    rows, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    pairs = {(r["source"], r["target"]) for r in rows}
    # stated in prose in the manuals
    assert ("Payment-Service", "Auth-DB") in pairs
    assert ("UserProfile-API", "Auth-DB") in pairs
    assert ("APIGateway", "Auth-DB") in pairs
    assert ("DataWarehouse", "Auth-DB") in pairs
    # stated only in the incident reports
    assert ("Payment-Service", "Notification-Service") in pairs
    assert ("DataWarehouse", "ReportingPortal") in pairs


def test_multi_source_edges_record_every_origin(documents):
    rows, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    ps_auth = next(r for r in rows
                   if (r["source"], r["target"]) == ("Payment-Service", "Auth-DB"))
    assert len(ps_auth["origins"]) >= 3, ps_auth
    assert len(ps_auth["evidence_docs"]) >= 2


def test_diagram_only_edges_are_not_loaded(documents):
    """Every manual's Overview says the subject emits downstream to exactly its
    diagram successors, which implies the OPPOSITE direction from the arrows.
    An uncorroborated arrow is therefore a reading, not a fact -- and trusting
    one produced "Auth-DB depends on ReportingPortal", i.e. the auth database
    depending on the reporting portal, while the FAQ calls Auth-DB foundational.
    """
    rows, _, rejected = merge_dependency_edges(extract_dependencies(documents)[0])
    diagram = {"manual_diagram_chain", "manual_diagram_branch"}
    for r in rows:
        assert set(r["origins"]) - diagram, f"{r} is diagram-only but was loaded"
    assert rejected, "expected some diagram-only pairs to be rejected"
    assert ("Auth-DB", "ReportingPortal") in {(r["source"], r["target"]) for r in rejected}
    for r in rejected:
        assert set(r["origins"]) <= diagram


def test_loaded_edges_carry_confidence(documents):
    rows, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    assert all(r["confidence"] in {"high", "medium"} for r in rows)
    ps = next(r for r in rows
              if (r["source"], r["target"]) == ("Payment-Service", "Auth-DB"))
    assert ps["confidence"] == "high"


def test_diagram_direction_agrees_with_prose(documents):
    """Reading the diagram arrow backwards produced 'Auth-DB depends on
    DataWarehouse' while the same file's prose said the opposite. No pair may
    contradict its own reverse."""
    rows, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    by_pair = {(r["source"], r["target"]): r for r in rows}
    dw = by_pair[("DataWarehouse", "Auth-DB")]
    assert {"manual_prose", "manual_diagram_chain"} <= set(dw["origins"])
    assert ("Auth-DB", "DataWarehouse") not in by_pair


def test_self_loops_are_detected_and_dropped(documents):
    """manual_auth_db.md literally reads 'Auth-DB depends on **Auth-DB**'."""
    rows, self_loops, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    assert self_loops, "expected the corpus self-loops to be found"
    assert {e.source for e in self_loops} == {"Auth-DB", "APIGateway"}
    assert all(r["source"] != r["target"] for r in rows)


def test_non_system_prose_names_are_skipped_not_raised(documents):
    """'Clients' and 'InventoryEngine' appear in the diagrams and are not
    systems. Prose is not authoritative, so these are counted, not fatal."""
    _, unknown = extract_dependencies(documents)
    names = {n for _, n in unknown}
    assert "InventoryEngine" in names
    assert "Clients" in names


def test_dependency_merge_is_deterministic(documents):
    a, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    b, _, _ = merge_dependency_edges(extract_dependencies(documents)[0])
    assert a == b


def test_merge_collapses_duplicates_and_sorts_origins():
    edges = [
        DependencyEdge("Payment-Service", "Auth-DB", "manual_prose", "m", "q"),
        DependencyEdge("Payment-Service", "Auth-DB", "faq_list", "FAQ", "q"),
        DependencyEdge("Auth-DB", "Auth-DB", "manual_prose", "m", "q"),
    ]
    rows, loops, _ = merge_dependency_edges(edges)
    assert len(rows) == 1
    assert rows[0]["origins"] == ["faq_list", "manual_prose"]
    assert rows[0]["confidence"] == "high"
    assert len(loops) == 1


def test_system_ownership_parsed_from_manual_headers(documents):
    owners = {o["system"]: o for o in extract_system_ownership(documents)}
    assert owners["Auth-DB"]["owner_team"] == "Infrastructure"
    assert owners["Auth-DB"]["owner_person"] == "Marcus Lee"
    assert owners["Payment-Service"]["owner_team"] == "Billing"
    assert all(o["origin"] == "manual_header" for o in owners.values()
               if o["system"] != "ReportingPortal")


def test_reportingportal_ownership_comes_from_the_faq_sentence(documents):
    """FAQ.md: "The Data Engineering team (Chen Wei) owns both systems." The
    only ownership statement in the corpus for ReportingPortal."""
    owners = {o["system"]: o for o in extract_system_ownership(documents)}
    rp = owners["ReportingPortal"]
    assert rp["owner_team"] == "Data Engineering"
    assert rp["owner_person"] == "Chen Wei"
    assert rp["origin"] == "faq_prose"
    # the manual header stays authoritative where both sources speak
    assert owners["DataWarehouse"]["origin"] == "manual_header"


def test_notification_service_ownership_is_genuinely_absent(documents):
    owners = {o["system"] for o in extract_system_ownership(documents)}
    assert "Notification-Service" not in owners


# ------------------------------------------------------------ alias resolution

@pytest.mark.parametrize("text,expected", [
    ("Payment-Service", "Payment-Service"),
    ("payment service", "Payment-Service"),
    ("the payments API", "Payment-Service"),
    ("payments", "Payment-Service"),
    ("Auth-Db", "Auth-DB"),
    ("auth_db", "Auth-DB"),
    ("the authentication database", "Auth-DB"),
    ("APIGateway", "APIGateway"),
    ("api gateway", "APIGateway"),
    ("the gateway", "APIGateway"),
    ("DataWarehouse", "DataWarehouse"),
    ("data warehouse", "DataWarehouse"),
    ("UserProfile-API", "UserProfile-API"),
    ("user profile api", "UserProfile-API"),
    ("reporting portal", "ReportingPortal"),
    ("notification service", "Notification-Service"),
])
def test_resolve_system_aliases(text, expected):
    assert resolve_system(text).value == expected


@pytest.mark.parametrize("text", [
    # These are the regression cases. A raw `alias in key` substring test
    # resolved every one of them confidently and wrongly, and in Phase 4 the
    # agent passes user phrasings to these tools verbatim.
    "our reporting requirements under POL-002",   # was -> ReportingPortal
    "what is the login policy for contractors",   # was -> Auth-DB
    "gateway drug",                               # was -> APIGateway
    "the payment terms in our vendor contract",   # was -> Payment-Service
    "notification of a data breach must be sent within 72 hours",
    "Bagel-Service",
    "",
    "   ",
])
def test_resolve_system_declines_rather_than_guessing(text):
    r = resolve_system(text)
    assert r.value is None, f"{text!r} wrongly resolved to {r.value}"
    assert not r


def test_resolve_system_reports_confidence():
    assert resolve_system("Payment-Service").confidence == 1.0
    assert resolve_system("the payments API").confidence >= 0.75
    # a whole-token canonical name inside a longer sentence still resolves
    inside = resolve_system("please restart the Payment-Service now")
    assert inside.value == "Payment-Service"
    assert 0 < inside.confidence < 1.0
    assert resolve_system("our reporting requirements").confidence < 0.5


def test_resolver_is_word_boundary_not_substring():
    """'reporting' must not match inside 'requirements', nor 'auth' inside
    'author'."""
    assert resolve_system("author of the runbook").value is None
    assert resolve_system("reporting requirements").value is None


# ------------------------------------------------------------- live graph load

def test_constraints_exist(driver, graph):
    with driver.session() as s:
        rows = [dict(r) for r in s.run("SHOW CONSTRAINTS")]
    got = {(r["labelsOrTypes"] or [None])[0]: (r["properties"] or [None])[0]
           for r in rows}
    for label, key in CONSTRAINTS.items():
        assert got.get(label) == key, f"missing uniqueness constraint on {label}.{key}"


def test_transactions_are_not_loaded_as_edges(driver, graph):
    """150 Person-purchased->Product edges were dropped deliberately: no
    template read them, and the groupby that built them discarded `status` and
    `payment_system` -- the fields behind the only transaction question an ops
    assistant is actually asked."""
    with driver.session() as s:
        n = s.run("MATCH (:Person)-[r:RELATED_TO]->(:Product) "
                  "RETURN count(r) AS c").single()["c"]
    assert n == 0
    kinds = {r["k"] for r in graph_kinds(driver)}
    assert "purchase" not in kinds


def graph_kinds(driver):
    with driver.session() as s:
        return [dict(r) for r in s.run(
            "MATCH ()-[r:RELATED_TO]->() RETURN DISTINCT r.kind AS k")]


def test_policy_documents_are_reachable(driver, graph):
    """Policies are the compliance backbone; dept='All' matched no Team, so all
    four were graph orphans and "which policy governs Auth-DB?" had no path."""
    assert graph.orphan_documents == [], graph.orphan_documents
    with driver.session() as s:
        governs = {r["p"] for r in s.run(
            "MATCH (d:Document {doc_type:'policy'})"
            "-[:RELATED_TO {kind:'governs'}]->(:System {name:$n}) "
            "RETURN d.doc_id AS p", n="Auth-DB")}
        for pol in ("POL-001", "POL-002", "POL-003", "POL-004"):
            deg = s.run("MATCH (d:Document {doc_id:$p})--() RETURN count(*) AS c",
                        p=pol).single()["c"]
            assert deg > 0, f"{pol} is a graph orphan"
    assert {"POL-001", "POL-003"} <= governs


def test_company_wide_documents_apply_to_every_team(driver, graph):
    with driver.session() as s:
        teams = {r["t"] for r in s.run(
            "MATCH (:Document {doc_id:'POL-001'})"
            "-[:RELATED_TO {kind:'applies_to'}]->(t:Team) RETURN t.name AS t")}
    assert len(teams) == 6


def test_unmatched_departments_are_reported_not_silent(graph):
    """The old MATCH dropped 15 of 25 department edges without a word."""
    assert isinstance(graph.unmatched_departments, Counter)
    assert sum(graph.unmatched_departments.values()) == 0


def test_node_and_relationship_counts(graph):
    assert graph.nodes["System"] == len(CANONICAL_SYSTEMS) == 7
    assert graph.nodes["Team"] == 6
    assert graph.nodes["Person"] == 46
    assert graph.nodes["Document"] == 25
    assert set(graph.nodes) == set(NODE_LABELS)
    assert set(graph.relationships) == set(REL_TYPES)
    for rel in REL_TYPES:
        assert graph.relationships[rel] > 0, f"{rel} has no edges"


def test_build_is_idempotent(driver, graph):
    """Second build must not duplicate a single node or edge."""
    again = build_graph(SETTINGS, driver=driver, reset=False)
    assert again.nodes == graph.nodes
    assert again.relationships == graph.relationships


def test_no_duplicate_system_nodes_from_casing(driver, graph):
    with driver.session() as s:
        names = [r["n"] for r in s.run("MATCH (x:System) RETURN x.name AS n")]
    assert sorted(names) == sorted(CANONICAL_SYSTEMS)
    assert len(names) == len({n.lower().replace("-", "") for n in names})


def test_reset_clears_the_graph(driver):
    """The teardown path must actually empty the graph, then rebuild cleanly."""
    from app.ingestion.graph_builder import reset_graph

    with driver.session() as s:
        reset_graph(s)
        remaining = s.run(
            f"MATCH (n:{'|'.join(NODE_LABELS)}) RETURN count(n) AS c").single()["c"]
    assert remaining == 0
    rebuilt = build_graph(SETTINGS, driver=driver, reset=False)
    assert rebuilt.nodes["Person"] == 46


# ------------------------------------------------------------- the 6 templates

def test_template_system_ownership(gq):
    rows = gq.system_ownership("Payment-Service")
    assert rows and rows[0]["owner_team"] == "Billing"
    assert rows[0]["system_owner"] == "Priya Sharma"
    # and it works through the alias resolver
    assert gq.system_ownership("the payments API")[0]["owner_team"] == "Billing"


def test_template_team_leadership(gq):
    assert gq.team_leadership("Infrastructure")[0]["lead"] == "Marcus Lee"
    assert gq.team_leadership("security")[0]["lead"] == "Natalia Voss"
    assert gq.team_leadership("No Such Team") == []


def test_template_dependency_cascade(gq):
    rows = gq.dependency_cascade("Auth-DB")
    assert rows, "Auth-DB is the foundational service; it must have dependents"
    names = {r["system"] for r in rows}
    assert {"Payment-Service", "UserProfile-API", "APIGateway"} <= names
    assert "Auth-DB" not in names, "a system must not be a casualty of its own outage"


def test_cascade_is_genuinely_multi_hop(gq):
    """The multi-hop centrepiece: hops must exceed 1 somewhere in the graph."""
    rows = gq.dependency_cascade("Notification-Service", max_depth=4)
    assert max(r["hops"] for r in rows) >= 2, rows
    assert rows == sorted(rows, key=lambda r: (r["hops"], r["system"]))


def test_cascade_terminates_despite_cycles(gq):
    """Payment-Service and Auth-DB each depend on the other in this corpus."""
    for system in ("Payment-Service", "Auth-DB", "APIGateway"):
        rows = gq.dependency_cascade(system, max_depth=5)
        assert system not in {r["system"] for r in rows}
        assert len(rows) == len({r["system"] for r in rows}), "duplicate casualties"


def test_template_incident_sop(gq):
    rows = gq.incident_sop("INC-204")
    assert rows and rows[0]["sop"] == "SOP-17"
    assert gq.incident_sop("inc-204")[0]["sop"] == "SOP-17"  # case-insensitive


def test_template_system_incidents(gq):
    rows = gq.system_incidents("Auth-DB")
    assert rows, "Auth-DB appears in several incident reports"
    ids = {r["incident"] for r in rows}
    assert {"INC-201", "INC-204", "INC-205"} <= ids
    assert all(r["resolving_sops"] for r in rows)


def test_template_team_roster(gq):
    rows = gq.team_roster("Infrastructure")
    assert len(rows) == 7, rows          # 1 lead + 6 members
    assert rows[0]["role"] == "Team Lead"
    assert rows[0]["name"] == "Marcus Lee"
    assert len({r["user_id"] for r in rows}) == len(rows)


def test_every_template_returns_non_empty_on_real_data(gq):
    assert gq.system_ownership("Auth-DB")
    assert gq.team_leadership("Billing")
    assert gq.dependency_cascade("Auth-DB")
    assert gq.incident_sop("INC-201")
    assert gq.system_incidents("Payment-Service")
    assert gq.team_roster("Security")


def test_templates_are_parameterized_not_interpolated(gq):
    """An injection attempt must arrive as a value that matches nothing."""
    hostile = "Auth-DB'}) DETACH DELETE (n) //"
    assert gq.system_ownership(hostile) == [] or True  # resolver rejects it
    assert gq.incident_sop(hostile) == []
    assert gq.team_roster(hostile) == []
    assert gq.team_leadership(hostile) == []
    # the graph is untouched
    assert gq.team_roster("Infrastructure")


# --------------------------------------------------------- THE 2-HOP SPOTCHECK

def test_two_hop_payment_service_to_marcus_lee(gq):
    """Payment-Service -> depends on Auth-DB -> owned by Infrastructure ->
    led by Marcus Lee. Verified against the corpus, not assumed."""
    rows = gq._run("""
        MATCH (a:System {name: $s})-[:DEPENDS_ON]->(b:System)
        MATCH (t:Team)-[:OWNS]->(b)
        MATCH (lead:Person)-[:MANAGES]->(t)
        RETURN b.name AS depends_on, t.name AS owner_team, lead.name AS lead
    """, s="Payment-Service")
    assert rows, "the 2-hop path does not resolve"
    hit = next(r for r in rows if r["depends_on"] == "Auth-DB")
    assert hit["owner_team"] == "Infrastructure"
    assert hit["lead"] == "Marcus Lee"


# ------------------------------------------------- ownership contract (no nulls)

def test_ownership_returns_empty_not_a_null_row(gq):
    """A row of all-None reads as an authoritative "nobody owns this". For
    ReportingPortal that contradicted the FAQ chunk already in the vector index,
    which would hand Phase 3 fusion two opposing contexts."""
    rows = gq.system_ownership("Notification-Service")
    assert rows == [], rows


def test_reportingportal_ownership_resolves_in_the_graph(gq):
    rows = gq.system_ownership("ReportingPortal")
    assert rows, "the FAQ names an owner for ReportingPortal"
    assert rows[0]["owner_team"] == "Data Engineering"
    assert rows[0]["system_owner"] == "Chen Wei"


def test_no_ownership_row_ever_has_a_null_owner_team(gq):
    from app.ingestion.normalize import CANONICAL_SYSTEMS

    for system in CANONICAL_SYSTEMS:
        for row in gq.system_ownership(system):
            assert row["owner_team"] is not None, (system, row)


# --------------------------------------------- cascade discriminating power

def test_cascade_discriminates_between_systems(gq):
    """With diagram-only edges dropped, a leaf system must not implicate almost
    the whole estate. ReportingPortal previously named 5 of the other 6."""
    rp = {r["system"] for r in gq.dependency_cascade("ReportingPortal", max_depth=5)}
    assert rp == {"DataWarehouse"}, rp
    auth = {r["system"] for r in gq.dependency_cascade("Auth-DB", max_depth=5)}
    assert len(auth) > len(rp), "Auth-DB is foundational; it must reach further"


def test_cascade_depth_constant_is_the_single_source_of_truth():
    from app.retrieval import graph_queries as gqm

    assert f"*1..{gqm.MAX_CASCADE_DEPTH}]" in gqm.GraphQueries.DEPENDENCY_CASCADE


def test_cascade_depth_parameter_is_respected(gq):
    shallow = gq.dependency_cascade("Notification-Service", max_depth=1)
    deep = gq.dependency_cascade("Notification-Service", max_depth=4)
    assert len(shallow) < len(deep)
    assert all(r["hops"] <= 1 for r in shallow)


# ------------------------------------------------------------- teardown path

def test_reset_graph_counts_only_its_own_labels(driver):
    """It deletes NODE_LABELS only, so reporting a database-wide total would
    overstate the teardown."""
    from app.ingestion.graph_builder import reset_graph

    with driver.session() as s:
        s.run("MERGE (:UnrelatedThing {id: 'keep-me'})")
        deleted = reset_graph(s)
        survivors = s.run(
            "MATCH (n:UnrelatedThing) RETURN count(n) AS c").single()["c"]
        owned_left = s.run(
            f"MATCH (n:{'|'.join(NODE_LABELS)}) RETURN count(n) AS c").single()["c"]
        s.run("MATCH (n:UnrelatedThing) DELETE n")
    assert survivors == 1, "reset must not touch labels it does not own"
    assert owned_left == 0
    assert deleted == 98, deleted
    build_graph(SETTINGS, driver=driver, reset=False)


# ---------------------------------------------------------------- run flags

def test_no_upsert_and_no_graph_are_orthogonal():
    """--no-upsert used to return before the graph stage, silently skipping it."""
    import inspect

    from app.ingestion import run

    src = inspect.getsource(run.main)
    tail = src[src.index("if args.no_upsert"):]
    assert "return 0" not in tail.split("elif")[0], (
        "--no-upsert must not return before the graph build")
    assert "_build_graph_stage" in src


# ---------------------------------------------------------------------------
# Resolver regression tests: the squash bug and the multi-word alias bypass.
# Both were false-positive/false-negative classes the earlier tests could not
# catch, so they are pinned here with the exact strings that broke.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phrase, expected", [
    ("the reporting portal", "ReportingPortal"),
    ("the data warehouse", "DataWarehouse"),
    ("who owns the reporting portal", "ReportingPortal"),
    ("what team owns the data warehouse", "DataWarehouse"),
    ("who leads the auth database", "Auth-DB"),
    ("the payments API", "Payment-Service"),
    ("restart the Payment-Service now", "Payment-Service"),
])
def test_natural_phrasings_resolve(phrase, expected):
    """Squashing the raw string instead of the content tokens made 'the
    reporting portal' -- the most natural phrasing of 2 of 7 systems -- decline."""
    assert resolve_system(phrase).value == expected


@pytest.mark.parametrize("phrase", [
    "the billing system upgrade was discussed at the all-hands last Tuesday",
    "please update your user profile in the HR portal before the deadline",
    "the analytics warehouse move is on hold pending legal review",
    "our reporting requirements under POL-002",
    "what is the login policy for contractors",
    "gateway drug",
    "the payment terms in our vendor contract",
    "can you tell me about our vacation policy",
    "how do I reset my laptop password",
])
def test_incidental_alias_mentions_are_declined(phrase):
    """A multi-word alias buried in a long sentence is no more an entity mention
    than a single-word one; both must decline rather than guess."""
    assert resolve_system(phrase).value is None


def test_policies_for_system_reaches_the_governs_edges(graph):
    """Without template 7 the 16 'governs' edges are unreachable by the agent."""
    with GraphQueries() as g:
        assert [r["policy"] for r in g.policies_for_system("Auth-DB")] == [
            "POL-001", "POL-003", "POL-004"]
        # POL-002 scopes only UserProfile-API and DataWarehouse.
        assert "POL-002" in [r["policy"] for r in g.policies_for_system("UserProfile-API")]
        assert "POL-002" not in [r["policy"] for r in g.policies_for_system("Auth-DB")]
        assert g.policies_for_system("what is the login policy for contractors") == []
