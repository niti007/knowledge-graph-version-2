# Chapter 06 — The Knowledge Graph

## Why this / what's the need

Vector search finds text that is *about* something. It cannot follow a chain. "Which team
owns the system Payment-Service depends on, and who leads it?" is three facts in three
documents: the dependency (the Payment-Service manual), the owner (the Auth-DB manual), and
the lead (the employee directory). A graph stores exactly those links, so a query can walk
from one fact to the next.

A graph also does something no similarity score can: it can say **no**. "What depends on
DataWarehouse?" has the answer *nothing*, and a graph query that returns zero rows is a
fact, not a failure. That negative answer becomes the most important safety signal in
Chapter 09.

> 🔑 **New word — node / edge:** a node is a thing (a System, a Team, a Person); an edge is
> a directed relationship between two nodes (`Billing -[OWNS]-> Payment-Service`).

> 🔑 **New word — Cypher:** Neo4j's query language. `MATCH (t:Team)-[:OWNS]->(s:System)`
> reads as "find a Team node with an OWNS edge to a System node".

The graph this project builds: **98 nodes / 236 relationships**, six node labels
(`System`, `Team`, `Person`, `Document`, `SOP`, `Product`) and five relationship types
(`DEPENDS_ON`, `OWNS`, `MANAGES`, `RESOLVED_BY`, `RELATED_TO`). It is built **with no LLM**,
so it is byte-identical on every run and costs nothing to rebuild.

---

## `app/ingestion/graph_builder.py`

### Uniqueness constraints make rebuilds idempotent

```python
CONSTRAINTS = {
    "System": "name",
    "Team": "name",
    "Person": "user_id",
    "Document": "doc_id",
    "SOP": "sop_id",
    "Product": "product_id",
}
```

```python
def create_constraints(session) -> None:
    for label, key in CONSTRAINTS.items():
        session.run(
            f"CREATE CONSTRAINT {label.lower()}_{key}_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{key} IS UNIQUE"
        )
```

- Every node label has one unique key, and every write is a `MERGE` on that key. `MERGE`
  means "create if absent, otherwise match" — so running the builder twice produces the
  same 98 nodes, not 196.

### People and teams from `users.csv`

```python
            # --- MANAGES: lead -> their team, and lead -> each team member
            leads = users[users["role"] == LEAD_ROLE]
            session.run("""
                UNWIND $rows AS r
                MATCH (p:Person {user_id: r.user_id}), (t:Team {name: r.team})
                MERGE (p)-[m:MANAGES]->(t)
                SET m.role = 'Team Lead'
            """, rows=leads[["user_id", "team"]].to_dict("records"))
```

- `UNWIND $rows AS r` — Cypher's way of looping over a list parameter. All 46 people go in
  one round trip instead of 46.
- `$rows` is a **bound parameter**. Nothing from the data is ever pasted into the query
  string. That matters more in `graph_queries.py`, where the input comes from the model.

### System ownership from the manual headers

Each manual opens with `**Owner Team:** Infrastructure` / `**System Owner:** Marcus Lee`.
A regex reads both:

```python
_MANUAL_HEADER = re.compile(r"\*\*(Owner Team|System Owner):\*\*\s*(.+)")
```

And the OWNS edge is created with `MATCH`, not `MERGE`, on the team:

```python
            # --- System ownership, from the manual header blocks
            # MATCH, never MERGE, the team: a typo in a manual header must not
            # invent a person-less Team node that then shows up in rosters.
            owner_rows = [r for r in ownership if r["owner_team"]]
            session.run("""
                UNWIND $rows AS r
                MATCH (s:System {name: r.system}), (t:Team {name: r.owner_team})
                MERGE (t)-[:OWNS]->(s)
            """, rows=owner_rows)
```

- If a manual said `Owner Team: Infrastucture` (typo), `MERGE` would create a phantom team.
  `MATCH` finds nothing, the edge is skipped, and a warning names the manual.

Only five of seven systems have a manual. ReportingPortal's ownership is stated once, in the
FAQ ("The Data Engineering team (Chen Wei) owns both systems"), and a second regex catches
exactly that sentence. **Notification-Service has no owner anywhere in the corpus**, and the
builder says so at the end of every run:

```python
            stats.systems_without_owner = [
                rec["name"] for rec in session.run(
                    "MATCH (s:System) WHERE NOT (:Team)-[:OWNS]->(s) "
                    "RETURN s.name AS name ORDER BY name")
            ]
```

Remember Notification-Service. It comes back in Chapters 08 and 09.

### `DEPENDS_ON` — parsed from prose, and why it is 8 edges not 15

The dependency edges are not in any CSV; they live in sentences inside the manuals,
incident reports and FAQ. Four regexes, each anchored on an explicit dependency verb:

```python
_DEP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # manual_*.md, Architecture section:
    #   "Payment-Service depends on **Auth-DB** for all authentication ..."
    ("manual_prose", re.compile(
        r"\*{0,2}([\w.-]+)\*{0,2}\s+depends\s+on\s+\*{0,2}([\w.-]+)\*{0,2}", re.I)),
    # INC-*.md Summary:
    #   "cascading failures in **UserProfile-API** which depends\non Auth-DB for ..."
    ("incident_summary", re.compile(...)),
    # INC-*.md Root Cause:
    #   "UserProfile-API has a hard\ndependency on Auth-DB;"
    ("incident_rca", re.compile(...)),
)
```

- `\*{0,2}` tolerates Markdown bold markers around a name. Each edge records its `origin`
  (which parser found it), the document, and a quote.

The plan allowed an LLM fallback for prose the regexes could not reach. It was never
needed:

```python
    An LLM fallback was scoped for whatever the parser could not reach, but
    every dependency statement in this corpus is templated and machine-regular:
    four fixed sentence forms plus an ASCII diagram. A model would add cost,
    latency and run-to-run nondeterminism to re-derive what a regex gets
    exactly, so none is used.
```

The manuals also contain an ASCII architecture diagram
(`[APIGateway] -> [Payment-Service] -> [ReportingPortal]`), and this is where honesty cost
seven edges. The comment explains:

```python
# CAVEAT, and the reason diagram-only edges are not trusted on their own: each
# manual's Overview also says the subject "provides ... downstream event
# emission to services including X and Y", where X and Y are exactly its
# diagram successors. Emission implies the OPPOSITE direction. The corpus is
# internally contradictory here, so the arrow reading is a defensible
# interpretation, not a settled fact. Only diagram edges corroborated by an
# explicit prose dependency statement are loaded (see LOADED_ORIGINS).
```

```python
EXPLICIT_ORIGINS = frozenset(
    {"manual_prose", "incident_summary", "incident_rca", "faq_list"})


def _confidence(origins: set[str]) -> str:
    explicit = origins & EXPLICIT_ORIGINS
    if len(explicit) >= 2:
        return "high"       # independently stated in two or more documents
    if explicit:
        return "medium"     # one explicit statement, possibly plus the diagram
    return "diagram_only"   # not loaded
```

- An edge supported only by the diagram is *rejected*. The rejected set included
  `Auth-DB -> ReportingPortal` — the authentication database depending on the reporting
  portal — while the FAQ calls Auth-DB "foundational". Loading it would have made
  "what breaks if ReportingPortal fails?" answer *five of six systems*. With the 8
  corroborated edges, the answer is DataWarehouse alone.
- Self-loops are dropped too; the corpus literally contains "Auth-DB depends on
  **Auth-DB**".

### Why purchase edges were dropped

```python
            # transactions.csv is deliberately NOT loaded. A Person-purchased->
            # Product edge per (user, product) pair would add 150 edges that no
            # query template reads, and the groupby that produced them discarded
            # `status` and `payment_system` -- the two fields behind the only
            # transaction question an ops assistant is actually asked ("how many
            # transactions failed on Payment-Service?"), which the vector index
            # already answers from the rendered ledger summary.
```

- 150 edges would have pushed the count from 236 to ~285 and made the graph look bigger.
  Nothing would have read them. The commit message puts it plainly: *an honest 236 beats a
  padded 285.*

### Policies were orphans

`doc_metadata.csv` marks all four policies as `dept=All`. No team is called "All", so the
`MATCH` on the department found nothing and the policies — and every SOP — had no edges.
The fix attaches company-wide documents to every team, and additionally parses each
policy's Scope section for the systems it names:

```python
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
```

The builder reports orphans on every run so this class of bug cannot return silently:

```python
            stats.orphan_documents = [
                rec["id"] for rec in session.run(
                    "MATCH (d:Document) WHERE NOT (d)--() "
                    "RETURN d.doc_id AS id ORDER BY id")
            ]
```

---

## `app/retrieval/graph_queries.py` — the 8 templates

The module docstring states the security rule:

```python
"""The six parameterized Cypher templates, plus entity resolution.

These are the ONLY queries the agent is permitted to run. Nothing here ever
interpolates model output into Cypher: every template takes bound parameters,
so a prompt-injected "; MATCH (n) DETACH DELETE n" arrives as a string that
matches no node instead of as syntax. Free-form LLM-generated Cypher is
deliberately not supported.
"""
```

(The docstring says six; the plan asked for six, and two were added during the build.
The registry at the bottom of the file has eight, and a test keeps it in sync with the
methods.)

| Template | Question it answers |
|---|---|
| `system_ownership` | which team (and person) owns system X, and who leads that team |
| `team_leadership` | who leads team X |
| `dependency_cascade` | what **breaks if** X fails (walks `DEPENDS_ON` backwards) |
| `system_dependencies` | what X **depends on** (forward) |
| `incident_sop` | which SOP resolved INC-nnn |
| `system_incidents` | which incidents involved X |
| `team_roster` | who is on team X, lead first |
| `policies_for_system` | which policies govern X |

### The forward/reverse pair

Only `dependency_cascade` existed at first. Chapter 08's agent, asked "which team owns the
system that Payment-Service depends on", reached for it and got `Auth-DB` — which was
*right by coincidence*, because Payment-Service and Auth-DB happen to depend on each other.
The true forward answer is `{Auth-DB, Notification-Service}`, and Notification-Service was
silently dropped. The docstring of the fix:

```python
    def system_dependencies(self, system: str, max_depth: int = 3) -> list[dict]:
        """What system X depends ON -- the forward direction.

        This is the inverse of dependency_cascade and exists because its absence
        was a real correctness hole. ... A wrong-direction template that
        returns a plausible row is invisible to every abstention signal we have,
        so the fix has to be a template that actually answers the question.
        """
```

The cascade query itself guards against cycles:

```python
    DEPENDENCY_CASCADE = """
        MATCH path = (dependent:System)-[:DEPENDS_ON*1..%(max)d]->(failed:System {name: $system})
        WHERE length(path) <= $max_depth
          AND ALL(n IN nodes(path) WHERE size([m IN nodes(path) WHERE m = n]) = 1)
        WITH dependent, min(length(path)) AS hops
        RETURN dependent.name AS system, hops
        ORDER BY hops, system
    """ % {"max": MAX_CASCADE_DEPTH}
```

- `[:DEPENDS_ON*1..5]` — a variable-length path, up to five hops. Neo4j will not bind the
  upper bound to a parameter, so it is interpolated **once from a module constant** — never
  from user input — while `$system` and `$max_depth` stay bound parameters.
- The `ALL(...)` clause requires no repeated node on the path. Without it, the real cycle
  (Payment-Service ↔ Auth-DB) reports a system as a casualty of its own outage.

### Empty means empty

```python
    def system_ownership(self, system: str) -> list[dict]:
        """Which team owns system X (and who is its named owner / team lead).

        Returns [] when the corpus names no owner. It previously returned one
        row with every field null, which reads as an authoritative "nobody owns
        this" -- and for ReportingPortal that directly contradicted the FAQ
        chunk in the vector index, handing Phase 3 fusion two opposing contexts.
        An empty result lets the caller fall back to retrieval instead.
        """
```

### `resolve_system` — whole tokens only

The agent will pass user phrasings straight through. The resolver maps free text to one of
the seven canonical names, or **declines**:

```python
    Matching is strictly on whole tokens. An earlier version tested
    `if alias in key` on raw substrings, which is confidently wrong on ordinary
    phrasings the agent will pass through verbatim in Phase 4:

        "our reporting requirements under POL-002"  -> ReportingPortal
        "what is the login policy for contractors"  -> Auth-DB
        "gateway drug"                              -> APIGateway

    Those now resolve to None.
```

```python
@dataclass(frozen=True)
class Resolution:
    """A resolution attempt. `value is None` means "decline", not "no match" --
    callers are expected to say they could not identify the entity rather than
    query a guess."""

    value: str | None
    matched_by: str          # canonical | alias | token | none
    confidence: float = 0.0
```

- "reporting" inside "requirements" is not a mention of ReportingPortal. Matching on whole
  tokens (`_contains_phrase`) fixes that, and an alias buried in a long sentence is
  *discarded* rather than returned at low confidence, so a caller reading only `.value`
  cannot act on a guess.

---

## ✅ You just learned
- Six labels, five relationship types, 98/236, built deterministically with `MERGE` on
  unique keys.
- Why `DEPENDS_ON` is 8 edges (corroborated) not 15 (diagram-only), why purchase edges were
  dropped, and how policies stopped being orphans.
- The eight parameterized templates, why there is a forward *and* a reverse dependency
  query, and why `[]` is the right answer for an unowned system.
- Entity resolution that declines rather than guesses.

## ▶️ Run this now
```bash
.venv/bin/python -m app.ingestion.run --no-upsert --reset-graph
```
Read the `[5/5]` block: 98 nodes, 236 relationships, `DEPENDS_ON edges: 8 loaded`, the
rejected diagram-only edges, and the warning that Notification-Service has no owner.

Then open http://localhost:7475 (user `neo4j`, your password) and run:
```cypher
MATCH (s:System {name:'Payment-Service'})-[:DEPENDS_ON]->(d:System)<-[:OWNS]-(t:Team)<-[:MANAGES]-(p:Person)
RETURN s.name, d.name, t.name, p.name
```

## 🧠 Check yourself
1. Why does the builder use `MATCH` (not `MERGE`) for the owning team?
2. What would `dependency_cascade('ReportingPortal')` return if diagram-only edges were
   loaded, and why is that wrong?
3. Why must `resolve_system("what is the login policy for contractors")` return `None`?
4. What stops a prompt-injected string from reaching Neo4j as Cypher?

---

Next: fusing the two search branches →
[07-hybrid-retrieval.md](07-hybrid-retrieval.md)
