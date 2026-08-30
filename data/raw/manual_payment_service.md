# Technical Manual — Payment-Service

**Owner Team:** Billing
**System Owner:** Priya Sharma
**Version:** 3.4
**Last Updated:** 2026-06-30

## Overview

**Payment-Service** is a core component of the enterprise platform, managed by the **Billing** team
under the technical leadership of **Priya Sharma**.

It provides: real-time processing, data validation, and downstream event emission to services
including **ReportingPortal** and **Auth-DB**.

## Architecture

```
[Clients] → [APIGateway] → [Payment-Service] → [ReportingPortal]
                                      ↘ [Auth-DB]
```

Payment-Service depends on **Auth-DB** for all authentication and session validation. Any outage in
Auth-DB will cause Payment-Service to return 503 errors to callers. See INC-204 for a historical
example of this failure mode.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_CONNECTIONS` | 100 | Connection pool ceiling |
| `TIMEOUT_MS` | 5000 | Request timeout in milliseconds |
| `RETRY_ATTEMPTS` | 3 | Retry count on transient failure |
| `CIRCUIT_BREAKER_THRESHOLD` | 0.5 | Error-rate threshold to open circuit |

## Operations

### Health Check
```
GET /health  →  { "status": "ok", "version": "3.4" }
```

### Deployment
Deployments follow **SOP-04** (Change Management). All changes require approval from **Priya Sharma**
and a successful run of the test suite in staging before production promotion.

### Incident Response
On failure, follow **SOP-01** (Incident Response Procedure). Page the **Billing** team
on-call channel. If Auth-DB is involved, coordinate with **Marcus Lee**.

## Monitoring

- Grafana dashboard: https://monitoring.internal/payment-service/overview
- Alert rules: PagerDuty policy `billing-payment-service`
- Log stream: Datadog service `payment-service`

## Related Documents

- SOP-01: Incident Response
- SOP-04: Change Management
- SOP-17: Payment Service Restoration (if Payment-Service = Payment-Service)
- POL-003: Information Security Policy
