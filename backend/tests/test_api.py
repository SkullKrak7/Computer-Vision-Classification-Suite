"""Test backend API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    print("✓ Root endpoint test passed")

def test_metrics():
    """Test metrics endpoint"""
    response = client.get("/api/metrics/model/test_model")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    print("✓ Metrics endpoint test passed")

if __name__ == '__main__':
    test_root()
    test_metrics()
    print("\n✓ All API tests passed!")
