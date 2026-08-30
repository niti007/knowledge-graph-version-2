# SOP-05 — On-Call Escalation

**Owner Team:** Infrastructure
**Procedure Owner:** Marcus Lee
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Infrastructure** team must follow when
executing: **On-Call Escalation**.

## Scope

Applies to all engineers in the **Infrastructure** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Marcus Lee.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Receive PagerDuty alert (auto-paged on P1/P2).
2. Acknowledge within 5 minutes to prevent escalation.
3. If unable to resolve in 20 minutes: escalate to team lead.
4. Team lead escalation path: Marcus Lee → Natalia Voss → VP Engineering.
5. For cross-team incidents, coordinate in #incident-war-room Slack channel.
6. Document all actions in the incident ticket.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Marcus Lee** (Infrastructure Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
