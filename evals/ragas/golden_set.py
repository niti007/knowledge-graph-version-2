"""30-question RAGAS golden set, with ground truths derived from the corpus.

Every `ground_truth` below was read out of `data/processed/documents.jsonl` or
out of Neo4j directly (98 nodes / 236 relationships) before it was written. The
`evidence` field names where, so a disputed answer key can be re-checked in one
grep rather than re-litigated from memory.

Only answerable questions are here on purpose. RAGAS' four metrics all assume a
retrievable answer exists -- context_recall against an empty gold context is
undefined, not zero -- so abstention behaviour is measured by the Promptfoo
refusal category and the Phase 9 scorecard, not here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenQuestion:
    qid: str
    category: str           # factual | policy | incident | sop | multi_hop | aggregation
    question: str
    ground_truth: str
    evidence: str           # doc_id(s) or the Cypher relation the answer comes from


GOLDEN_SET: list[GoldenQuestion] = [
    # ------------------------------------------------------- factual / config
    GoldenQuestion(
        "Q01", "factual",
        "Who owns the Payment-Service?",
        "Payment-Service is owned by the Billing team. Priya Sharma is the system "
        "owner and the Billing team lead.",
        "FAQ; manual_payment_service; (:Team {name:'Billing'})-[:OWNS]->(:System {name:'Payment-Service'})"),
    GoldenQuestion(
        "Q02", "factual",
        "What is the default MAX_CONNECTIONS value for Auth-DB?",
        "The default MAX_CONNECTIONS for Auth-DB is 100. It is the connection pool ceiling.",
        "manual_auth_db (Configuration table)"),
    GoldenQuestion(
        "Q03", "factual",
        "What is the default CIRCUIT_BREAKER_THRESHOLD for APIGateway?",
        "The default CIRCUIT_BREAKER_THRESHOLD for APIGateway is 0.5, the error-rate "
        "threshold at which the circuit opens.",
        "manual_apigateway (Configuration table)"),
    GoldenQuestion(
        "Q04", "factual",
        "What is the default request timeout for Payment-Service?",
        "TIMEOUT_MS defaults to 5000 milliseconds for Payment-Service.",
        "manual_payment_service (Configuration table)"),
    GoldenQuestion(
        "Q05", "factual",
        "What does the Payment-Service health check endpoint return?",
        "GET /health returns {\"status\": \"ok\", \"version\": \"3.4\"}.",
        "manual_payment_service (Operations / Health Check)"),

    # ------------------------------------------------------------- policy
    GoldenQuestion(
        "Q06", "policy",
        "How long does ACME retain transaction records?",
        "Transaction records are retained for 7 years, which is a regulatory requirement "
        "under POL-002, the Data Retention Policy.",
        "POL-002; FAQ"),
    GoldenQuestion(
        "Q07", "policy",
        "After how many days is customer PII anonymised?",
        "Customer PII held in UserProfile-API and DataWarehouse is anonymised after "
        "90 days, per GDPR requirements in POL-002.",
        "POL-002; FAQ"),
    GoldenQuestion(
        "Q08", "policy",
        "How long are incident reports retained?",
        "Incident reports, such as INC-201 through INC-205, are retained for 3 years "
        "under POL-002.",
        "POL-002 (Retention Schedules)"),
    GoldenQuestion(
        "Q09", "policy",
        "What encryption does the Information Security Policy require in transit and at rest?",
        "POL-003 requires TLS 1.3 as a minimum for data in transit and AES-256 for data "
        "at rest. APIGateway enforces TLS for all inbound requests.",
        "POL-003 (Encryption)"),
    GoldenQuestion(
        "Q10", "policy",
        "Which activities does the Acceptable Use Policy prohibit?",
        "POL-001 prohibits sharing credentials, bypassing Auth-DB authentication, "
        "exfiltrating customer PII from UserProfile-API or DataWarehouse, and running "
        "unauthorised scripts on production systems without Change Management approval "
        "under SOP-04.",
        "POL-001 (Prohibited Activities)"),
    GoldenQuestion(
        "Q11", "policy",
        "What equipment are remote workers required to use?",
        "POL-004 requires company-issued laptops only. Personal devices require MDM "
        "enrollment approved by the Security team, and disk encryption with FileVault "
        "or BitLocker is mandatory.",
        "POL-004 (Equipment)"),
    GoldenQuestion(
        "Q12", "policy",
        "Which systems require multi-factor authentication?",
        "POL-003 makes multi-factor authentication mandatory for Auth-DB, "
        "UserProfile-API, DataWarehouse and APIGateway.",
        "POL-003 (Access Control)"),

    # ------------------------------------------------------------ people / org
    GoldenQuestion(
        "Q13", "factual",
        "Who is the Security team lead?",
        "Natalia Voss is the Security team lead.",
        "users; (:Person {name:'Natalia Voss'})-[:MANAGES]->(:Team {name:'Security'})"),
    GoldenQuestion(
        "Q14", "factual",
        "Who leads the Data Engineering team?",
        "Chen Wei is the Data Engineering team lead.",
        "users; (:Person {name:'Chen Wei'})-[:MANAGES]->(:Team {name:'Data Engineering'})"),
    GoldenQuestion(
        "Q15", "factual",
        "Which team owns UserProfile-API and who leads that team?",
        "UserProfile-API is owned by the Customer Success team, led by Amara Okafor, "
        "who is also the system owner.",
        "manual_userprofile_api; (:Team {name:'Customer Success'})-[:OWNS]->(:System {name:'UserProfile-API'})"),
    GoldenQuestion(
        "Q16", "factual",
        "Which team owns APIGateway and who is its system owner?",
        "APIGateway is owned by the Security team; Natalia Voss is the system owner "
        "and the Security team lead.",
        "manual_apigateway; (:Team {name:'Security'})-[:OWNS]->(:System {name:'APIGateway'})"),

    # ---------------------------------------------------------------- SOP
    GoldenQuestion(
        "Q17", "sop",
        "Which SOP covers security breach containment and who owns it?",
        "SOP-22, Security Breach Containment, owned by the Security team with "
        "Natalia Voss as procedure owner.",
        "SOP-22 (header)"),
    GoldenQuestion(
        "Q18", "sop",
        "Which SOP covers payment service restoration and who owns it?",
        "SOP-17, Payment Service Restoration, owned by the Billing team with "
        "Priya Sharma as procedure owner.",
        "SOP-17 (header)"),
    GoldenQuestion(
        "Q19", "sop",
        "What is the first step of the incident response procedure?",
        "Step 1 of SOP-01 is to acknowledge the PagerDuty alert within 5 minutes.",
        "SOP-01 (Procedure, step 1)"),
    GoldenQuestion(
        "Q20", "sop",
        "What is the team-lead escalation path in the on-call escalation SOP?",
        "SOP-05 gives the team lead escalation path as Marcus Lee, then Natalia Voss, "
        "then VP Engineering. Escalation happens if the issue is unresolved after "
        "20 minutes.",
        "SOP-05 (Procedure, step 4)"),
    GoldenQuestion(
        "Q21", "sop",
        "Within how many minutes must a security breach be reported, and to whom?",
        "SOP-22 and POL-003 require alerting Natalia Voss, the Security Lead, within "
        "15 minutes of detection.",
        "SOP-22 (step 1); POL-003 (Incident Response); FAQ"),

    # ------------------------------------------------------------- incident
    GoldenQuestion(
        "Q22", "incident",
        "What caused the March 2026 Payment-Service outage?",
        "INC-204, on 2026-03-12, was a P1 incident on Payment-Service caused by Auth-DB "
        "hitting its connection limits. Priya Sharma led it and it was resolved per "
        "SOP-17.",
        "INC-204; FAQ"),
    GoldenQuestion(
        "Q23", "incident",
        "What caused incident INC-205?",
        "INC-205, on 2026-04-07, was a P1 incident on APIGateway caused by TLS "
        "certificate expiry. Natalia Voss led it and it was resolved per SOP-22.",
        "INC-205"),
    GoldenQuestion(
        "Q24", "incident",
        "What caused incident INC-203 and who led it?",
        "INC-203, on 2026-02-28, was a P2 incident on ReportingPortal caused by "
        "DataWarehouse schema drift. Chen Wei led it and it was resolved per SOP-02.",
        "INC-203"),
    GoldenQuestion(
        "Q25", "incident",
        "Which incident affected Auth-DB in January 2026, and which SOP resolved it?",
        "INC-201, on 2026-01-15, a P2 connection pool exhaustion on Auth-DB, resolved "
        "per SOP-01 and led by Marcus Lee.",
        "INC-201; (:Document {doc_id:'INC-201'})-[:RESOLVED_BY]->(:SOP {sop_id:'SOP-01'})"),

    # ------------------------------------------------------------- multi-hop
    GoldenQuestion(
        "Q26", "multi_hop",
        "Which team owns the system Payment-Service depends on for authentication, "
        "and who leads that team?",
        "Payment-Service depends on Auth-DB for authentication. Auth-DB is owned by the "
        "Infrastructure team, which is led by Marcus Lee.",
        "DEPENDS_ON(Payment-Service, Auth-DB); OWNS(Infrastructure, Auth-DB); MANAGES(Marcus Lee, Infrastructure)"),
    GoldenQuestion(
        "Q27", "multi_hop",
        "Which systems break if Auth-DB goes down?",
        "Four systems depend on Auth-DB and fail with it: Payment-Service, "
        "UserProfile-API, APIGateway and DataWarehouse.",
        "dependency_cascade('Auth-DB') -> 4 systems at 1 hop; FAQ"),
    GoldenQuestion(
        "Q28", "multi_hop",
        "Who should be contacted if UserProfile-API is failing because of its "
        "authentication dependency?",
        "Marcus Lee. UserProfile-API depends on Auth-DB, which the Infrastructure team "
        "owns and Marcus Lee leads.",
        "DEPENDS_ON(UserProfile-API, Auth-DB); OWNS(Infrastructure, Auth-DB); FAQ"),
    GoldenQuestion(
        "Q29", "multi_hop",
        "Besides Auth-DB, which system does DataWarehouse depend on, and who owns it?",
        "DataWarehouse also depends on ReportingPortal, which is owned by the Data "
        "Engineering team and led by Chen Wei.",
        "DEPENDS_ON(DataWarehouse, ReportingPortal); OWNS(Data Engineering, ReportingPortal)"),

    # ------------------------------------------------------------ aggregation
    GoldenQuestion(
        "Q30", "aggregation",
        "How many teams does ACME have and what are they?",
        "ACME has 6 teams: Billing, Customer Success, Data Engineering, Infrastructure, "
        "Product and Security.",
        "users; MATCH (t:Team) RETURN count(t) -> 6"),
]

assert len(GOLDEN_SET) == 30, len(GOLDEN_SET)
assert len({q.qid for q in GOLDEN_SET}) == 30
