"""Tests for metrics routes"""

from unittest.mock import mock_open, patch

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_get_metrics_success():
    """Test getting metrics for valid model"""
    mock_data = '{"accuracy": 0.95, "precision": 0.93, "recall": 0.94, "f1_score": 0.935}'

    with patch("pathlib.Path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data=mock_data)):
            response = client.get("/v1/metrics/model/knn")

    assert response.status_code == 200
    data = response.json()
    assert data["accuracy"] == 0.95
    assert data["precision"] == 0.93


def test_get_metrics_model_not_found():
    """Test getting metrics for unknown model"""
    response = client.get("/v1/metrics/model/unknown_model")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_metrics_file_not_exists():
    """Test getting metrics when file doesn't exist"""
    with patch("pathlib.Path.exists", return_value=False):
        response = client.get("/v1/metrics/model/knn")

    assert response.status_code == 404
