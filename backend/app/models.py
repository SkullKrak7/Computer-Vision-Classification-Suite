"""Pydantic models for API"""

from pydantic import BaseModel
from typing import List, Optional

class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float

class InferenceResponse(BaseModel):
    predictions: List[PredictionResponse]
    inference_time: float

class TrainingRequest(BaseModel):
    dataset_path: str
    model_type: str
    epochs: int = 20
    batch_size: int = 32

class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
