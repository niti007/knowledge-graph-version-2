# SOP-01 — Incident Response Procedure

**Owner Team:** Infrastructure
**Procedure Owner:** Marcus Lee
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Infrastructure** team must follow when
executing: **Incident Response Procedure**.

## Scope

Applies to all engineers in the **Infrastructure** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Marcus Lee.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Acknowledge the PagerDuty alert within 5 minutes.
2. Assess severity (P1–P4) using the incident matrix in the runbook.
3. Page the owning team lead (see PEOPLE directory in FAQ.md).
4. Open a war-room Slack channel: #incident-<INC-ID>.
5. Identify root cause using Grafana dashboards and Datadog logs.
6. Apply remediation steps from the system-specific runbook.
7. Validate recovery via health-check endpoints.
8. Write post-mortem within 48 hours and update this SOP if gaps found.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Marcus Lee** (Infrastructure Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
