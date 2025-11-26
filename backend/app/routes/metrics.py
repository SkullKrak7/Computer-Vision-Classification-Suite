"""Metrics API routes"""

from fastapi import APIRouter
from ..models import MetricsResponse

router = APIRouter()

@router.get("/model/{model_id}", response_model=MetricsResponse)
async def get_metrics(model_id: str):
    """Get model metrics"""
    return MetricsResponse(
        accuracy=0.95,
        precision=0.94,
        recall=0.93,
        f1_score=0.935
    )
