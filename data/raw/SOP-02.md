# SOP-02 — Data Backup and Recovery

**Owner Team:** Data Engineering
**Procedure Owner:** Chen Wei
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Data Engineering** team must follow when
executing: **Data Backup and Recovery**.

## Scope

Applies to all engineers in the **Data Engineering** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Chen Wei.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Verify backup status in the DataWarehouse backup dashboard.
2. Identify the last clean snapshot (use the data catalog managed by Chen Wei).
3. Notify stakeholders: Billing (Priya Sharma) if transaction data affected.
4. Restore from snapshot to staging environment first.
5. Run data integrity checks (row counts, checksums).
6. Promote to production after sign-off from Chen Wei.
7. Update the backup log and notify affected teams.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Chen Wei** (Data Engineering Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
