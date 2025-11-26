"""FastAPI backend for CV classification"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import inference, training, metrics

app = FastAPI(title="CV Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["metrics"])

@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}
