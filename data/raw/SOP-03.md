# SOP-03 — Access Provisioning

**Owner Team:** Security
**Procedure Owner:** Natalia Voss
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Security** team must follow when
executing: **Access Provisioning**.

## Scope

Applies to all engineers in the **Security** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Natalia Voss.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Receive access request ticket with manager approval.
2. Verify requester identity via Auth-DB MFA.
3. Check POL-003 to confirm access level is permitted.
4. Grant access in the identity management system.
5. Log the provisioning event in the audit trail.
6. Notify the requester and their manager.
7. Schedule access review for 90 days.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Natalia Voss** (Security Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
