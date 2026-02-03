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
    # Create 11MB of random data to exceed 10MB limit
    return b"fake_image_data" * (1024 * 1024)  # 15MB of data
