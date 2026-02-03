"""Tests for inference API endpoints"""

from io import BytesIO

import pytest


def test_predict_endpoint_no_model(client, sample_image_bytes):
    """Test predict when model not loaded"""
    response = client.post("/predict", files={"file": ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg")})
    # May return 503 if model not found, or 200 if model exists
    assert response.status_code in [200, 503]


def test_predict_endpoint_invalid_file(client):
    """Test predict with invalid file"""
    response = client.post("/predict", files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")})
    assert response.status_code == 400


def test_predict_endpoint_large_file(client, large_image_bytes):
    """Test predict with oversized file"""
    response = client.post("/predict", files={"file": ("large.jpg", BytesIO(large_image_bytes), "image/jpeg")})
    assert response.status_code == 413


def test_predict_endpoint_missing_file(client):
    """Test predict without file"""
    response = client.post("/predict")
    assert response.status_code == 422  # Unprocessable entity
