"""Pydantic models for API"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictionResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float


class InferenceResponse(BaseModel):
    predictions: list[PredictionResponse]
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


class ErrorResponse(BaseModel):
    """Error response model"""

    detail: str


class TrainingConfig(BaseModel):
    """Training configuration with validation"""

    model_type: Literal["pytorch", "tensorflow", "keras"]
    dataset_path: str
    epochs: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=512)
    learning_rate: float = Field(default=0.001, gt=0, le=1)


class DatasetConfig(BaseModel):
    """Dataset configuration with validation"""

    dataset_path: str
    train_split: float = Field(default=0.8, ge=0, le=1)
    val_split: float = Field(default=0.1, ge=0, le=1)
    test_split: float = Field(default=0.1, ge=0, le=1)

    @field_validator("test_split")
    @classmethod
    def validate_splits_sum(cls, v, info):
        """Ensure splits sum to 1.0"""
        train = info.data.get("train_split", 0.8)
        val = info.data.get("val_split", 0.1)
        if abs(train + val + v - 1.0) > 0.01:
            raise ValueError("train_split + val_split + test_split must equal 1.0")
        return v


class ImageUploadRequest(BaseModel):
    """Image upload request"""

    filename: str
    content_type: str
