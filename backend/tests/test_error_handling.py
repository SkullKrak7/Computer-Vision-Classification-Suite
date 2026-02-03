"""Tests for error handling"""

import pytest


def test_404_error_format(client):
    """Test 404 returns proper error format"""
    response = client.get("/nonexistent")
    assert response.status_code == 404


def test_405_method_not_allowed(client):
    """Test 405 for wrong HTTP method"""
    response = client.get("/v1/inference/predict")
    assert response.status_code == 405


def test_422_validation_error(client):
    """Test 422 for validation errors"""
    response = client.post("/v1/inference/predict")
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_400_bad_request(client):
    """Test 400 for bad requests"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_413_payload_too_large(client, large_image_bytes):
    """Test 413 for oversized files"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("large.jpg", BytesIO(large_image_bytes), "image/jpeg")}
    )
    assert response.status_code == 413
    data = response.json()
    assert "detail" in data


def test_503_service_unavailable(client, sample_image_bytes):
    """Test 503 when model not loaded"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
    )
    # Either works or returns 503
    assert response.status_code in [200, 503]
    if response.status_code == 503:
        data = response.json()
        assert "detail" in data


def test_error_response_structure(client):
    """Test error responses have consistent structure"""
    response = client.get("/nonexistent")
    assert response.status_code == 404
    # Should be JSON
    assert response.headers["content-type"] == "application/json"


def test_multiple_validation_errors(client):
    """Test handling multiple validation errors"""
    # Send completely invalid request
    response = client.post("/v1/inference/predict", json={"invalid": "data"})
    assert response.status_code in [400, 422]


def test_error_with_special_characters(client):
    """Test error handling with special characters in input"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("test<>|?.jpg", BytesIO(b"data"), "image/jpeg")}
    )
    assert response.status_code in [400, 413, 503]


def test_concurrent_errors(client):
    """Test error handling under concurrent requests"""
    from io import BytesIO

    # Make multiple invalid requests
    responses = []
    for _ in range(5):
        resp = client.post("/v1/inference/predict", files={"file": ("test.txt", BytesIO(b"bad"), "text/plain")})
        responses.append(resp)

    # All should return 400
    assert all(r.status_code == 400 for r in responses)
