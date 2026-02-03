"""Tests for image validation utilities"""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.utils.validation import sanitize_filename, validate_image_upload


@pytest.mark.asyncio
async def test_validate_image_valid_jpeg(sample_image_bytes):
    """Test valid JPEG upload"""
    file = UploadFile(filename="test.jpg", file=BytesIO(sample_image_bytes))
    file.content_type = "image/jpeg"
    image, filename = await validate_image_upload(file)
    assert image.size == (100, 100)
    assert filename == "test.jpg"


@pytest.mark.asyncio
async def test_validate_image_invalid_size(large_image_bytes):
    """Test file size limit"""
    file = UploadFile(filename="large.jpg", file=BytesIO(large_image_bytes))
    file.content_type = "image/jpeg"
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_validate_image_invalid_mime():
    """Test invalid MIME type"""
    file = UploadFile(filename="test.txt", file=BytesIO(b"not an image"))
    file.content_type = "text/plain"
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    assert exc.value.status_code == 400


def test_sanitize_filename_path_traversal():
    """Test path traversal prevention"""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("../../../secret.txt") == "secret.txt"


def test_sanitize_filename_special_chars():
    """Test special character removal"""
    assert sanitize_filename("file<>name.jpg") == "filename.jpg"
    assert sanitize_filename("test|file?.png") == "testfile.png"
