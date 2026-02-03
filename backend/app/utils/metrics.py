"""Prometheus metrics for monitoring"""

from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response

# Metrics
request_count = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
request_duration = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])
inference_count = Counter("inference_requests_total", "Total inference requests", ["status"])
inference_duration = Histogram("inference_duration_seconds", "Inference duration")


def metrics_endpoint():
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")
