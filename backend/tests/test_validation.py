"""Tests for image validation utilities"""

import pytest
from fastapi import HTTPException

from backend.app.utils.validation import sanitize_filename


def test_validate_image_valid_jpeg(client, sample_image_bytes):
    """Test valid JPEG upload via API"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg")}
    )
    assert response.status_code in [200, 503]


def test_validate_image_invalid_size(client, large_image_bytes):
    """Test file size limit via API"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("large.jpg", BytesIO(large_image_bytes), "image/jpeg")}
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


def test_validate_image_invalid_mime(client):
    """Test invalid MIME type via API"""
    from io import BytesIO

    response = client.post(
        "/v1/inference/predict", files={"file": ("test.txt", BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400
    assert "mime" in response.json()["detail"].lower() or "invalid" in response.json()["detail"].lower()


def test_sanitize_filename_path_traversal():
    """Test path traversal prevention"""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("../../../secret.txt") == "secret.txt"
    assert sanitize_filename("/etc/passwd") == "passwd"


def test_sanitize_filename_special_chars():
    """Test special character removal"""
    result = sanitize_filename("file<>name.jpg")
    assert "<" not in result and ">" not in result
    result = sanitize_filename("test|file?.png")
    assert "|" not in result and "?" not in result


def test_sanitize_filename_null_bytes():
    """Test null byte removal"""
    result = sanitize_filename("file\x00name.jpg")
    assert "\x00" not in result


def test_sanitize_filename_unicode():
    """Test unicode handling"""
    result = sanitize_filename("файл.jpg")
    assert len(result) > 0
