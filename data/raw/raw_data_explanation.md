# Raw Data — What's Here and How It Connects

This folder holds the **dummy enterprise dataset** the assistant answers questions from.
It's fake data about a made-up company (**ACME Corp**), but everything is **cross-linked**
on purpose — the same people, teams, systems, and procedures appear across many files.
That linking is what lets the assistant answer *multi-step* questions like
"if Payment-Service goes down, what else breaks and who do I call?"

The data is loaded into two places:
- **ChromaDB (vector search)** — the text of every document, for "what does it say?"
- **Neo4j (graph)** — the entities and their links, for "how is this connected?"

---

## 1. The files at a glance (26 files)

| Group | Files | Format | What it is |
|---|---|---|---|
| **Policies** | POL-001 … POL-004 | PDF | Company-wide policy documents (Legal/Compliance) |
| **Incident reports** | INC-201 … INC-205 | Markdown | Write-ups of past outages (cause, impact, fix) |
| **SOPs** | SOP-01 … SOP-05, SOP-17, SOP-22 | Markdown | Standard Operating Procedures (step-by-step guides) |
| **Technical manuals** | manual_* (5 systems) | Markdown | How each system works |
| **FAQ** | FAQ.md | Markdown | Common questions and answers |
| **People** | users.csv | CSV | 46 employees: name, email, team, role |
| **Products** | products.csv | CSV | 7 products and the team that owns each |
| **Transactions** | transactions.csv | CSV | Sample transaction records |
| **Metadata map** | doc_metadata.csv | CSV | A table linking each document to the systems + SOPs it references |

---

## 1b. What each file actually looks like (real examples)

### Incident report — e.g. `INC-204.md`
A structured write-up of a past outage. Header fields + sections for summary, impact,
root cause, resolution, and action items. Real excerpt:

```markdown
# Incident Report — INC-204
**Date:** 2026-03-12   **Severity:** P1   **System Affected:** Payment-Service
**Owner Team:** Billing   **Incident Lead:** Priya Sharma   **Referenced SOP:** SOP-17

## Root Cause Analysis
Auth-DB hitting connection limits ... Payment-Service saturated its connection pool,
all downstream callers including Auth-DB began failing.

## Resolution
Resolved following SOP-17 (Payment Service Restoration).

## Action Items
| # | Action | Owner | Due |
| 2 | Add circuit breaker between Auth-DB and Payment-Service | Marcus Lee | +14 days |
```
→ Gives the assistant: what broke, why, who fixed it, and which SOP was used.

### SOP (Standard Operating Procedure) — e.g. `SOP-17.md`
A numbered how-to for handling a situation. Real excerpt:

```markdown
# SOP-17 — Payment Service Restoration
**Owner Team:** Billing   **Procedure Owner:** Priya Sharma

## Procedure
1. Confirm Payment-Service is returning 5xx errors via APIGateway logs.
2. Check Auth-DB connection pool metrics on Grafana (primary dependency).
3. If Auth-DB is saturated: page Marcus Lee (Infrastructure) immediately.
...
## Escalation
Primary: Priya Sharma (Billing Lead) → Secondary: Marcus Lee → Executive: VP Engineering
```
→ Gives the assistant: the exact steps and escalation path to fix a problem.

### Technical manual — e.g. `manual_payment_service.md`
Explains how one system works: overview, architecture diagram, config, operations. Excerpt:

```markdown
# Technical Manual — Payment-Service
**Owner Team:** Billing   **System Owner:** Priya Sharma

Payment-Service depends on Auth-DB for all authentication and session validation.
Any outage in Auth-DB will cause Payment-Service to return 503 errors. See INC-204.

| Parameter | Default | Description |
| MAX_CONNECTIONS | 100 | Connection pool ceiling |
```
→ Gives the assistant: how a system is built, its settings, and its dependencies.

### FAQ — `FAQ.md`
Question-and-answer pairs covering common topics. Excerpt:

```markdown
**Q: Who owns the Payment-Service?**
A: Payment-Service is owned by the Billing team, led by Priya Sharma...

**Q: Which systems depend on Auth-DB?**
A: Payment-Service (Billing), UserProfile-API (Customer Success), APIGateway (Security)...
```
→ Quick, direct answers already written in plain language.

### Policy PDFs — `POL-001.pdf` … `POL-004.pdf`
Company-wide rules (Legal/Compliance), stored as PDF to prove the app can read PDFs too.
Examples referenced throughout: POL-002 (Data Retention), POL-003 (Information Security).

### CSV — people: `users.csv`
One row per employee. Columns: `user_id, name, email, team, role, active, created_date`

```csv
U0001,Priya Sharma,priya.sharma@acme.internal,Billing,Team Lead,True,2023-01-15
U0002,Marcus Lee,marcus.lee@acme.internal,Infrastructure,Team Lead,True,2023-01-15
U0004,Chen Wei,chen.wei@acme.internal,Data Engineering,Team Lead,True,2023-01-15
```
→ 46 people across 6 teams. Answers "who is on Billing?", "how many managers?" (SQL tool).

### CSV — products: `products.csv`
Columns: `product_id, name, owner_team, category, price_usd`

```csv
PRD-001,Analytics Suite,Product,SaaS,299.0
PRD-005,Billing Automation,Billing,SaaS,199.0
```
→ 7 products, each linked to the team that owns it.

### CSV — transactions: `transactions.csv`
Columns: `txn_id, date, user_id, product_id, amount_usd, status, payment_system`

```csv
TXN-00001,2025-11-06,U0009,PRD-006,955.81,completed,Payment-Service
TXN-00002,2026-05-20,U0017,PRD-005,1746.14,completed,Payment-Service
```
→ Sample sales records. Answers "total sales?", "how many completed transactions?" (SQL tool).
Note each row links a **user** (U0009) to a **product** (PRD-006) via the **Payment-Service**.

### CSV — the master map: `doc_metadata.csv`
The glue file. Columns: `filename, doc_type, dept, date, author, system_refs, sop_refs`

```csv
INC-204.md,incident_report,Billing,2026-03-12,Priya Sharma,"Payment-Service,Auth-DB",SOP-17
manual_payment_service.md,technical_manual,Infrastructure,2026-06-30,Marcus Lee,Payment-Service,"SOP-01,SOP-04"
```
→ For every document it records which **systems** and **SOPs** it references — this is how
we know the files are connected, and it drives filtering and graph building.

---

## 2. The building blocks (entities)

**6 Teams:** Billing · Infrastructure · Security · Data Engineering · Customer Success · Product

**6 Team Leads (from `users.csv`):**
| Person | Leads team |
|---|---|
| Priya Sharma | Billing |
| Marcus Lee | Infrastructure |
| Natalia Voss | Security |
| Chen Wei | Data Engineering |
| Amara Okafor | Customer Success |
| Diego Reyes | Product |

**Core systems:** Payment-Service · Auth-DB · UserProfile-API · DataWarehouse ·
APIGateway · Notification-Service · ReportingPortal

**Procedures:** SOP-01 … SOP-05, SOP-17, SOP-22 (each incident points to the SOP used to fix it)

---

## 3. How everything connects (the important part)

The documents deliberately reference each other. Example — the incident **INC-204** ties
together a system, a dependency, a team, a person, and a procedure all at once:

> "A P1 incident hit **Payment-Service** because **Auth-DB** hit connection limits.
> The **Billing** team, led by **Priya Sharma**, owns Payment-Service and resolved it
> using **SOP-17**. Follow-up action assigned to **Marcus Lee** (Infrastructure)."

From that one file we learn these links:

```
Payment-Service  —DEPENDS_ON→  Auth-DB
Billing          —OWNS→        Payment-Service
Priya Sharma     —MANAGES→     Billing
INC-204          —RESOLVED_BY→ SOP-17
```

The **`doc_metadata.csv`** file is the master map — every document row lists the systems
(`system_refs`) and procedures (`sop_refs`) it touches. For example:

| filename | systems referenced | SOP referenced |
|---|---|---|
| INC-201 | Auth-DB, UserProfile-API | SOP-01 |
| INC-202 | Notification-Service, Payment-Service | SOP-05 |
| INC-203 | ReportingPortal, DataWarehouse | SOP-02 |
| INC-204 | Payment-Service, Auth-DB | SOP-17 |
| INC-205 | APIGateway, Auth-DB | SOP-22 |

Because these references overlap (Auth-DB shows up in INC-201, INC-204, and INC-205),
the systems form a connected web — which is exactly what the knowledge graph captures.

---

## 4. Relationship types in the graph

When we build the Neo4j graph, every link is one of five types:

| Link | Meaning | Example |
|---|---|---|
| `DEPENDS_ON` | System A needs System B to work | Payment-Service → Auth-DB |
| `OWNS` | A team owns a system/product | Billing → Payment-Service |
| `MANAGES` | A person leads a team | Priya Sharma → Billing |
| `RESOLVED_BY` | An incident was fixed using an SOP | INC-204 → SOP-17 |
| `RELATED_TO` | Any other meaningful connection | Payment-Service → INC-204 |

Result: ~120 entities (dots) and ~280 links (arrows).

---

## 5. What kind of question each source answers

| You ask… | Answered from… |
|---|---|
| "What's the refund policy?" | Policy PDFs (text → vector search) |
| "How do I restore Payment-Service?" | SOP-17 (text → vector search) |
| "What caused the March 2026 outage?" | INC-204 (text → vector search) |
| "Who owns Payment-Service and who leads that team?" | Graph (OWNS + MANAGES links) |
| "If Auth-DB fails, what else is affected?" | Graph (DEPENDS_ON links, multi-hop) |
| "How many people are in the Billing team?" | users.csv (SQL query tool) |

---

## 6. How to (re)build from this data

```powershell
python -m app.ingestion.chunker         # text → ChromaDB vector index
python -m app.ingestion.graph_builder   # entities/links → Neo4j graph
```

To regenerate a **fresh** random version of this whole dataset:
```powershell
python -m app.ingestion.generate_fake_data
```

> Note: the graph currently in Neo4j was built from *this exact* copy of the data. If you
> regenerate the dataset, rebuild the graph too so the names stay in sync.
