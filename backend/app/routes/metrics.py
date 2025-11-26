"""Metrics API routes"""

from fastapi import APIRouter, HTTPException
from ..models import MetricsResponse
import json
from pathlib import Path

router = APIRouter()

METRICS_MAP = {
    "knn": "models/knn_metadata.json",
    "svm": "models/svm_metadata.json",
    "pytorch_cnn": "models/pytorch_cnn_tuned_metadata.json",
    "tensorflow_mobilenet": "models/tensorflow_mobilenet_tuned_metadata.json",
}

@router.get("/model/{model_id}", response_model=MetricsResponse)
async def get_metrics(model_id: str):
    """Get model metrics from saved JSON files"""
    metrics_path = METRICS_MAP.get(model_id)
    if not metrics_path or not Path(metrics_path).exists():
        raise HTTPException(status_code=404, detail=f"Metrics not found for {model_id}")
    
    with open(metrics_path) as f:
        data = json.load(f)
    
    return MetricsResponse(
        accuracy=data.get('accuracy', 0.0),
        precision=data.get('precision', 0.0),
        recall=data.get('recall', 0.0),
        f1_score=data.get('f1_score', 0.0)
    )
