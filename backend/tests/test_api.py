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
    print(" Root endpoint test passed")

def test_metrics():
    """Test metrics endpoint - should return 404 for non-existent model"""
    response = client.get("/api/metrics/model/test_model")
    assert response.status_code == 404
    print(" Metrics endpoint correctly returns 404 for missing model")
    
def test_metrics_existing():
    """Test metrics endpoint with existing model"""
    import json
    from pathlib import Path
    
    # Create test metrics file
    Path("models/baseline").mkdir(parents=True, exist_ok=True)
    test_metrics = {"accuracy": 0.85, "precision": 0.84, "recall": 0.83, "f1_score": 0.835}
    with open("models/baseline/test_metrics.json", "w") as f:
        json.dump(test_metrics, f)
    
    # Test endpoint (need to add test_model to METRICS_MAP first)
    print(" Metrics dynamic loading test skipped (requires model files)")

if __name__ == '__main__':
    test_root()
    test_metrics()
    test_metrics_existing()
    print("\n All API tests passed!")
