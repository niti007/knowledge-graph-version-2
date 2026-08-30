# Technical Manual — Auth-DB

**Owner Team:** Infrastructure
**System Owner:** Marcus Lee
**Version:** 3.4
**Last Updated:** 2026-06-30

## Overview

**Auth-DB** is a core component of the enterprise platform, managed by the **Infrastructure** team
under the technical leadership of **Marcus Lee**.

It provides: real-time processing, data validation, and downstream event emission to services
including **Payment-Service** and **ReportingPortal**.

## Architecture

```
[Clients] → [APIGateway] → [Auth-DB] → [Payment-Service]
                                      ↘ [ReportingPortal]
```

Auth-DB depends on **Auth-DB** for all authentication and session validation. Any outage in
Auth-DB will cause Auth-DB to return 503 errors to callers. See INC-204 for a historical
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
Deployments follow **SOP-04** (Change Management). All changes require approval from **Marcus Lee**
and a successful run of the test suite in staging before production promotion.

### Incident Response
On failure, follow **SOP-01** (Incident Response Procedure). Page the **Infrastructure** team
on-call channel. If Auth-DB is involved, coordinate with **Marcus Lee**.

## Monitoring

- Grafana dashboard: https://monitoring.internal/auth-db/overview
- Alert rules: PagerDuty policy `infrastructure-auth-db`
- Log stream: Datadog service `auth-db`

## Related Documents

- SOP-01: Incident Response
- SOP-04: Change Management
- SOP-17: Payment Service Restoration (if Auth-DB = Payment-Service)
- POL-003: Information Security Policy
