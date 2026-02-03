# Monitoring & Observability

## Prometheus Metrics

Exposed at `/metrics`:

```
# Inference metrics
inference_requests_total{status="success|error"}
inference_duration_seconds

# System metrics
process_cpu_seconds_total
process_resident_memory_bytes
```

## Starting Monitoring Stack

```bash
docker-compose up prometheus grafana alertmanager

# Access
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
# Alertmanager: http://localhost:9093
```

## Dashboards

Pre-configured Grafana dashboards:
- Request rate and latency
- Error rates
- Model inference performance
- System resource usage

## Alerts

Configured in `monitoring/prometheus-alerts.yml`:

| Alert | Condition | Severity |
|-------|-----------|----------|
| HighErrorRate | >5% errors over 5min | critical |
| HighLatency | p95 >1s | warning |
| ServiceDown | No metrics for 1min | critical |
| HighInferenceLatency | >500ms | warning |
| HighMemoryUsage | >80% | warning |

## Structured Logging

JSON format with:
```json
{
  "timestamp": "2026-02-03T18:00:00.000Z",
  "level": "INFO",
  "logger": "backend.app.routes.inference",
  "message": "Prediction complete",
  "request_id": "abc123",
  "inference_time": 0.045
}
```

## Viewing Logs

```bash
# Docker logs
docker-compose logs -f backend

# Filter by level
docker-compose logs backend | grep ERROR

# Follow specific service
docker-compose logs -f prometheus
```

## Production Recommendations

Replace with managed services:
- CloudWatch (AWS)
- Datadog
- New Relic
- Grafana Cloud

Self-hosted monitoring adds operational overhead.
