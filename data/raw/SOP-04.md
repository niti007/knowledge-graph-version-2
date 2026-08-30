# SOP-04 — Change Management

**Owner Team:** Infrastructure
**Procedure Owner:** Marcus Lee
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Infrastructure** team must follow when
executing: **Change Management**.

## Scope

Applies to all engineers in the **Infrastructure** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Marcus Lee.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Submit change request in the ITSM tool.
2. Obtain approval from the owning team lead.
3. Schedule the change during approved maintenance windows.
4. Run automated tests in staging (CI/CD pipeline).
5. Deploy to production with rollback plan ready.
6. Monitor Grafana dashboards for 30 minutes post-deployment.
7. Close the change request and document outcomes.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Marcus Lee** (Infrastructure Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
