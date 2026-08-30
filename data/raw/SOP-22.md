# SOP-22 — Security Breach Containment

**Owner Team:** Security
**Procedure Owner:** Natalia Voss
**Last Reviewed:** 2026-06-30
**Version:** 2.1

## Purpose

This Standard Operating Procedure defines the steps the **Security** team must follow when
executing: **Security Breach Containment**.

## Scope

Applies to all engineers in the **Security** team and any on-call responder triggered via
PagerDuty escalation. Cross-team coordination requires approval from Natalia Voss.

## Prerequisites

- Access to internal monitoring (Grafana, Datadog)
- Write access to affected systems (granted per POL-003 — Information Security Policy)
- Familiarity with POL-002 (Data Retention Policy) if data recovery is involved

## Procedure

1. Alert Natalia Voss (Security Lead) immediately — within 15 minutes of detection.
2. Isolate affected systems from the network (coordinate with Marcus Lee).
3. Revoke compromised credentials via Auth-DB admin interface.
4. Preserve logs (do NOT delete — legal requirement per POL-002).
5. Notify legal/compliance team if PII may be exposed.
6. Conduct forensic analysis with Security team.
7. Document breach timeline and apply fixes.
8. Issue post-breach security review and update POL-003 if needed.

## Escalation

If the procedure cannot be completed within 30 minutes, escalate to:
- Primary: **Natalia Voss** (Security Lead)
- Secondary: **Marcus Lee** (Infrastructure)
- Executive: VP of Engineering

## Related Documents

- POL-003: Information Security Policy
- SOP-05: On-Call Escalation
- Contact directory: https://wiki.internal/contacts
