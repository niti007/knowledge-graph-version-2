# Enterprise Knowledge Assistant — FAQ

## General

**Q: Who owns the Payment-Service?**
A: Payment-Service is owned by the **Billing** team, led by **Priya Sharma**. For escalations,
contact Priya directly or page the Billing on-call via PagerDuty.

**Q: What do I do during a P1 incident?**
A: Follow SOP-01 (Incident Response Procedure). Immediately page the on-call engineer via
PagerDuty. If the incident involves Auth-DB or APIGateway, also loop in **Marcus Lee**
(Infrastructure). For security-related incidents, contact **Natalia Voss** (Security).

**Q: Which systems depend on Auth-DB?**
A: Auth-DB is a foundational service. Dependent systems include:
- Payment-Service (Billing team — Priya Sharma)
- UserProfile-API (Customer Success — Amara Okafor)
- APIGateway (Security — Natalia Voss)

Any Auth-DB outage triggers cascading failures across these systems. See INC-204 for the
March 2026 example.

**Q: How do I request access to a system?**
A: Submit an access request per SOP-03 (Access Provisioning). Access is governed by
POL-003 (Information Security Policy). All requests require manager approval.

**Q: What is the data retention period for transaction records?**
A: Per POL-002 (Data Retention Policy), transaction records are retained for 7 years.
PII data is anonymised after 90 days per GDPR requirements. Contact **Chen Wei**
(Data Engineering) for data queries.

**Q: Who is the on-call lead for Infrastructure incidents?**
A: **Marcus Lee** is the primary on-call lead for Infrastructure. The escalation path is:
Marcus Lee → VP Engineering → CTO. Follow SOP-05 for escalation steps.

**Q: How does the model router work?**
A: The AI assistant routes simple factual queries to gpt-4o-mini (faster, cheaper) and
complex multi-hop or reasoning queries to gpt-4o. Routing is based on query complexity
signals: length, presence of multi-entity references, and historical latency.

## Data & Compliance

**Q: Where is customer PII stored?**
A: Customer PII resides in UserProfile-API's backing store (managed by the Customer Success
team under Amara Okafor) and the DataWarehouse (Data Engineering, Chen Wei). Both are
governed by POL-002 and POL-003.

**Q: What happens if a security breach is detected?**
A: Follow SOP-22 (Security Breach Containment) immediately. Contact **Natalia Voss**
(Security Lead) within 15 minutes. The Security team will invoke POL-003 procedures and
notify relevant stakeholders. Do NOT attempt containment without Natalia's sign-off.

## Systems

**Q: What caused the March 2026 Payment-Service outage?**
A: INC-204 (2026-03-12, P1): Auth-DB hit its connection pool limit, causing Payment-Service
to fail. Payment-Service has a hard dependency on Auth-DB. Billing team lead Priya Sharma
managed recovery per SOP-17 (Payment Service Restoration).

**Q: Is there a runbook for ReportingPortal failures?**
A: Yes. ReportingPortal failures are usually caused by DataWarehouse schema drift (see
INC-203). Follow SOP-02 (Data Backup and Recovery). The Data Engineering team (Chen Wei)
owns both systems.
