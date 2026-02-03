"""Tests for Pydantic models"""

import pytest
from pydantic import ValidationError

from backend.app.models import DatasetConfig, InferenceResponse, TrainingConfig


def test_training_config_valid():
    """Test valid training config"""
    config = TrainingConfig(
        model_type="pytorch",
        dataset_path="/tmp/data",
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
    )
    assert config.epochs == 10
    assert config.batch_size == 32


def test_training_config_invalid_epochs():
    """Test epochs validation"""
    with pytest.raises(ValidationError):
        TrainingConfig(model_type="pytorch", dataset_path="/tmp/data", epochs=0)


def test_training_config_invalid_batch_size():
    """Test batch size validation"""
    with pytest.raises(ValidationError):
        TrainingConfig(model_type="pytorch", dataset_path="/tmp/data", batch_size=0)


def test_training_config_invalid_model_type():
    """Test model type validation"""
    with pytest.raises(ValidationError):
        TrainingConfig(model_type="invalid", dataset_path="/tmp/data")


def test_dataset_config_valid():
    """Test valid dataset config"""
    config = DatasetConfig(dataset_path="/tmp/data", train_split=0.8, val_split=0.1, test_split=0.1)
    assert config.train_split == 0.8


def test_dataset_config_splits_sum():
    """Test splits sum to 1.0"""
    with pytest.raises(ValidationError):
        DatasetConfig(dataset_path="/tmp/data", train_split=0.5, val_split=0.3, test_split=0.3)


def test_inference_response_valid():
    """Test inference response"""
    response = InferenceResponse(
        predictions=[{"class_id": 0, "class_name": "cat", "confidence": 0.95}], inference_time=0.1
    )
    assert len(response.predictions) == 1
    assert response.inference_time == 0.1
