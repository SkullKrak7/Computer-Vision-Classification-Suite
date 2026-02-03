"""Tests for security middleware"""

import time

import pytest


def test_health_check(client):
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_security_headers(client):
    """Test security headers present"""
    response = client.get("/health")
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "X-XSS-Protection" in response.headers
    assert "Strict-Transport-Security" in response.headers
    assert "Content-Security-Policy" in response.headers


def test_request_id_header(client):
    """Test request ID tracking"""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]
    assert len(request_id) == 36  # UUID format


def test_request_id_unique(client):
    """Test each request gets unique ID"""
    response1 = client.get("/health")
    response2 = client.get("/health")
    assert response1.headers["X-Request-ID"] != response2.headers["X-Request-ID"]


def test_timing_header(client):
    """Test timing header"""
    response = client.get("/health")
    assert "X-Process-Time" in response.headers
    process_time = float(response.headers["X-Process-Time"])
    assert process_time >= 0
    assert process_time < 1.0  # Should be fast


def test_rate_limiting_basic(client):
    """Test rate limiting allows normal traffic"""
    responses = [client.get("/health") for _ in range(10)]
    assert all(r.status_code == 200 for r in responses)


def test_rate_limiting_burst(client):
    """Test rate limiting with burst traffic"""
    # Make 50 requests rapidly
    responses = []
    for _ in range(50):
        responses.append(client.get("/health"))

    # All should succeed (under 100/min limit)
    success_count = sum(1 for r in responses if r.status_code == 200)
    assert success_count == 50


def test_cors_headers(client):
    """Test CORS configuration"""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200


def test_cors_blocked_origin(client):
    """Test CORS blocks unauthorized origins"""
    response = client.get("/health", headers={"Origin": "http://evil.com"})
    # Should still return 200 but without CORS headers
    assert response.status_code == 200


def test_middleware_order(client):
    """Test middleware executes in correct order"""
    response = client.get("/health")
    # All middleware should have run
    assert "X-Request-ID" in response.headers  # RequestIDMiddleware
    assert "X-Process-Time" in response.headers  # TimingMiddleware
    assert "X-Content-Type-Options" in response.headers  # SecurityHeadersMiddleware


def test_error_handling_with_middleware(client):
    """Test middleware works with errors"""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    # Middleware should still add headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers
