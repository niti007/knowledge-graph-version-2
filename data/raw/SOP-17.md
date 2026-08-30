# SOP-17 — Payment Service Restoration

**Owner Team:** Billing
**Procedure Owner:** Priya Sharma
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Billing** team must follow when
executing: **Payment Service Restoration**.

## Scope

Applies to all engineers in the **Billing** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Priya Sharma.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Confirm Payment-Service is returning 5xx errors via APIGateway logs.
2. Check Auth-DB connection pool metrics on Grafana (primary dependency).
3. If Auth-DB is saturated: page Marcus Lee (Infrastructure) immediately.
4. Apply Auth-DB connection pool increase per the runbook parameters.
5. Enable Payment-Service circuit breaker to shed load.
6. Verify Payment-Service health-check returns 200.
7. Confirm transaction processing resumes (check Billing dashboard).
8. Notify Priya Sharma (Billing Lead) of resolution.
9. File INC report and update this SOP if new failure mode discovered.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Priya Sharma** (Billing Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
