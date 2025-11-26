"""Training API routes"""

import time
import uuid

from fastapi import APIRouter, BackgroundTasks

from ..models import TrainingRequest

router = APIRouter()

# In-memory job storage (use Redis/DB in production)
training_jobs = {}


def run_training_job(job_id: str, model_type: str):
    """Simulate training job"""
    training_jobs[job_id] = {"status": "running", "progress": 0.0, "model": model_type}

    # Simulate training progress
    for i in range(1, 11):
        time.sleep(1)
        training_jobs[job_id]["progress"] = i / 10.0

    training_jobs[job_id]["status"] = "completed"
    training_jobs[job_id]["progress"] = 1.0


@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training"""
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_training_job, job_id, request.model_type)
    return {"status": "training_started", "job_id": job_id}


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get training status"""
    if job_id not in training_jobs:
        return {"job_id": job_id, "status": "not_found", "progress": 0.0}

    job = training_jobs[job_id]
    return {"job_id": job_id, "status": job["status"], "progress": job["progress"]}
