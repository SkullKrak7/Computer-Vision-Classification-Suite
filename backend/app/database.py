"""Database models"""

from typing import Any

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base: Any = declarative_base()


class PredictionLog(Base):
    """Log of inference predictions"""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    predicted_class = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    inference_time = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelMetrics(Base):
    """Model performance metrics"""

    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
