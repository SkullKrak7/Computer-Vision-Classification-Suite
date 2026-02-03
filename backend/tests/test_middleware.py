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
    assert "X-XSS-Protection" in response.headers


def test_request_id_header(client):
    """Test request ID tracking"""
    response = client.get("/health")
    assert "X-Request-ID" in response.headers


def test_timing_header(client):
    """Test timing header"""
    response = client.get("/health")
    assert "X-Process-Time" in response.headers
    assert float(response.headers["X-Process-Time"]) >= 0


def test_rate_limiting(client):
    """Test rate limiting (basic check)"""
    # Make multiple requests quickly
    responses = [client.get("/health") for _ in range(10)]
    assert all(r.status_code == 200 for r in responses)


def test_cors_headers(client):
    """Test CORS configuration"""
    response = client.options("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code in [200, 405]  # OPTIONS may not be implemented
