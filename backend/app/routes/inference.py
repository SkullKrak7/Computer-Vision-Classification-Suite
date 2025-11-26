"""Inference API routes"""

from fastapi import APIRouter, UploadFile, File
from ..models import InferenceResponse
import time

router = APIRouter()

@router.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """Run inference on uploaded image"""
    start = time.time()
    # TODO: Implement actual inference
    return InferenceResponse(
        predictions=[],
        inference_time=time.time() - start
    )
