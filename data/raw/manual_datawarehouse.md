# Technical Manual — DataWarehouse

**Owner Team:** Data Engineering
**System Owner:** Chen Wei
**Version:** 3.4
**Last Updated:** 2026-06-30

## Overview

**DataWarehouse** is a core component of the enterprise platform, managed by the **Data Engineering** team
under the technical leadership of **Chen Wei**.

It provides: real-time processing, data validation, and downstream event emission to services
including **Auth-DB** and **APIGateway**.

## Architecture

```
[Clients] → [APIGateway] → [DataWarehouse] → [Auth-DB]
                                      ↘ [APIGateway]
```

DataWarehouse depends on **Auth-DB** for all authentication and session validation. Any outage in
Auth-DB will cause DataWarehouse to return 503 errors to callers. See INC-204 for a historical
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
Deployments follow **SOP-04** (Change Management). All changes require approval from **Chen Wei**
and a successful run of the test suite in staging before production promotion.

### Incident Response
On failure, follow **SOP-01** (Incident Response Procedure). Page the **Data Engineering** team
on-call channel. If Auth-DB is involved, coordinate with **Marcus Lee**.

## Monitoring

- Grafana dashboard: https://monitoring.internal/datawarehouse/overview
- Alert rules: PagerDuty policy `data engineering-datawarehouse`
- Log stream: Datadog service `datawarehouse`

## Related Documents

- SOP-01: Incident Response
- SOP-04: Change Management
- SOP-17: Payment Service Restoration (if DataWarehouse = Payment-Service)
- POL-003: Information Security Policy
