"""Async tests for validation utilities"""

from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from backend.app.utils.validation import validate_image_upload


@pytest.mark.asyncio
async def test_validate_image_upload_valid():
    """Test async validation with valid image"""
    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class MockUploadFile:
        def __init__(self, filename, file, content_type):
            self.filename = filename
            self.file = file
            self.content_type = content_type
        
        async def read(self):
            return self.file.read()

    file = MockUploadFile(filename="test.jpg", file=buf, content_type="image/jpeg")
    image, filename = await validate_image_upload(file)
    
    assert image.size == (100, 100)
    assert filename == "test.jpg"


@pytest.mark.asyncio
async def test_validate_image_upload_too_large():
    """Test async validation with oversized file"""
    # Create 15MB of data
    large_data = b"x" * (15 * 1024 * 1024)
    
    class MockUploadFile:
        def __init__(self, filename, file, content_type):
            self.filename = filename
            self.file = file
            self.content_type = content_type
        
        async def read(self):
            return self.file.read()
    
    file = MockUploadFile(filename="large.jpg", file=BytesIO(large_data), content_type="image/jpeg")
    
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_validate_image_upload_invalid_format():
    """Test async validation with invalid format"""
    class MockUploadFile:
        def __init__(self, filename, file, content_type):
            self.filename = filename
            self.file = file
            self.content_type = content_type
        
        async def read(self):
            return self.file.read()
    
    file = MockUploadFile(filename="test.txt", file=BytesIO(b"not an image"), content_type="text/plain")
    
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_validate_image_upload_dimensions_too_large():
    """Test async validation with oversized dimensions"""
    from PIL import Image

    img = Image.new("RGB", (5000, 5000), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class MockUploadFile:
        def __init__(self, filename, file, content_type):
            self.filename = filename
            self.file = file
            self.content_type = content_type
        
        async def read(self):
            return self.file.read()

    file = MockUploadFile(filename="huge.jpg", file=buf, content_type="image/jpeg")
    
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    
    assert exc.value.status_code == 400
    assert "dimension" in str(exc.value.detail).lower()


@pytest.mark.asyncio
async def test_validate_image_upload_dimensions_too_small():
    """Test async validation with undersized dimensions"""
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="green")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    class MockUploadFile:
        def __init__(self, filename, file, content_type):
            self.filename = filename
            self.file = file
            self.content_type = content_type
        
        async def read(self):
            return self.file.read()

    file = MockUploadFile(filename="tiny.jpg", file=buf, content_type="image/jpeg")
    
    with pytest.raises(HTTPException) as exc:
        await validate_image_upload(file)
    
    assert exc.value.status_code == 400
    assert "dimension" in str(exc.value.detail).lower()
