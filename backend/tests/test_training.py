"""Tests for training routes"""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_start_training():
    """Test starting a training job"""
    response = client.post("/v1/training/start", json={"model_type": "resnet50", "dataset_path": "/data/train"})

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "training_started"
    assert "job_id" in data


def test_get_status_not_found():
    """Test getting status for non-existent job"""
    response = client.get("/v1/training/status/fake-job-id")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "not_found"
    assert data["progress"] == 0.0


def test_get_status_running():
    """Test getting status for running job"""
    from backend.app.routes.training import training_jobs

    training_jobs["test-job-123"] = {"status": "running", "progress": 0.5, "model": "resnet50"}

    response = client.get("/v1/training/status/test-job-123")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert data["progress"] == 0.5

    training_jobs.clear()


def test_get_status_completed():
    """Test getting status for completed job"""
    from backend.app.routes.training import training_jobs

    training_jobs["test-job-456"] = {"status": "completed", "progress": 1.0, "model": "vgg16"}

    response = client.get("/v1/training/status/test-job-456")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["progress"] == 1.0

    training_jobs.clear()
