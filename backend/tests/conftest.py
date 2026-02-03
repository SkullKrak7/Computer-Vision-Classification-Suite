"""Pytest fixtures for backend tests"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Generate minimal valid JPEG bytes"""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def large_image_bytes():
    """Generate image exceeding size limit (>10MB)"""
    from io import BytesIO

    from PIL import Image

    img = Image.new("RGB", (5000, 5000), color="blue")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=100)
    buf.seek(0)
    return buf.getvalue()
