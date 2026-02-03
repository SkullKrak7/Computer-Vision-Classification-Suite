"""Tests for database models and operations"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base, ModelMetrics, PredictionLog


@pytest.fixture
def db_session():
    """Create in-memory SQLite database for testing"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_prediction_log_creation(db_session):
    """Test creating a prediction log entry"""
    log = PredictionLog(filename="test.jpg", predicted_class="cat", confidence=0.95, inference_time=45.2)
    db_session.add(log)
    db_session.commit()

    result = db_session.query(PredictionLog).first()
    assert result.filename == "test.jpg"
    assert result.predicted_class == "cat"
    assert result.confidence == 0.95
    assert result.inference_time == 45.2


def test_model_metrics_creation(db_session):
    """Test creating model metrics entry"""
    metrics = ModelMetrics(model_name="resnet50", accuracy=0.92, precision=0.89, recall=0.91, f1_score=0.90)
    db_session.add(metrics)
    db_session.commit()

    result = db_session.query(ModelMetrics).first()
    assert result.model_name == "resnet50"
    assert result.accuracy == 0.92
    assert result.precision == 0.89


def test_prediction_log_timestamp(db_session):
    """Test that timestamp is auto-generated"""
    log = PredictionLog(filename="test.jpg", predicted_class="dog", confidence=0.88, inference_time=50.0)
    db_session.add(log)
    db_session.commit()

    result = db_session.query(PredictionLog).first()
    assert result.created_at is not None
    assert isinstance(result.created_at, datetime)


def test_model_metrics_timestamp(db_session):
    """Test that created_at is auto-generated"""
    metrics = ModelMetrics(model_name="vgg16", accuracy=0.85, precision=0.83, recall=0.84, f1_score=0.835)
    db_session.add(metrics)
    db_session.commit()

    result = db_session.query(ModelMetrics).first()
    assert result.created_at is not None
    assert isinstance(result.created_at, datetime)
