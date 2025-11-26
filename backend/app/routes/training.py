"""Training API routes"""

from fastapi import APIRouter
from ..models import TrainingRequest

router = APIRouter()

@router.post("/start")
async def start_training(request: TrainingRequest):
    """Start model training"""
    return {"status": "training_started", "job_id": "123"}

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get training status"""
    return {"job_id": job_id, "status": "running", "progress": 0.5}
