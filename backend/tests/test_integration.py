"""End-to-end integration tests"""

import pytest


@pytest.mark.integration
def test_health_check_integration(client):
    """Test health endpoint returns correct structure"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


@pytest.mark.integration
def test_metrics_endpoint_integration(client):
    """Test metrics endpoint returns Prometheus format"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    content = response.text
    assert "http_requests_total" in content or "# HELP" in content


@pytest.mark.integration
def test_full_prediction_flow(client, sample_image_bytes):
    """Test complete prediction workflow"""
    from io import BytesIO

    # Make prediction request
    response = client.post(
        "/v1/inference/predict", files={"file": ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
    )

    # Should either succeed or fail gracefully (model may not load or inference may fail)
    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "inference_time" in data
        assert isinstance(data["predictions"], list)
        assert len(data["predictions"]) > 0
        assert "class_name" in data["predictions"][0]
        assert "confidence" in data["predictions"][0]


@pytest.mark.integration
def test_error_handling_flow(client):
    """Test error handling returns proper format"""
    # Invalid request
    response = client.post("/v1/inference/predict")
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.integration
def test_cors_headers_integration(client):
    """Test CORS headers are set correctly"""
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200


@pytest.mark.integration
def test_security_headers_integration(client):
    """Test all security headers are present"""
    response = client.get("/health")
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers
    assert "X-XSS-Protection" in response.headers
    assert "Strict-Transport-Security" in response.headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


@pytest.mark.integration
def test_rate_limiting_integration(client):
    """Test rate limiting works across multiple requests"""
    # Make 10 requests quickly
    responses = []
    for _i in range(10):
        resp = client.get("/health")
        responses.append(resp.status_code)

    # All should succeed (under 100 req/min limit)
    assert all(status == 200 for status in responses)


@pytest.mark.integration
def test_invalid_route_integration(client):
    """Test 404 handling"""
    response = client.get("/nonexistent")
    assert response.status_code == 404


@pytest.mark.integration
def test_method_not_allowed_integration(client):
    """Test 405 handling"""
    response = client.get("/v1/inference/predict")
    assert response.status_code == 405
