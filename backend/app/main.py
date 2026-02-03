"""FastAPI backend with security, logging, and observability"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    TimingMiddleware,
)
from .models import ErrorResponse
from .routes import inference, metrics, training
from .utils.logging import get_logger
from .utils.metrics import metrics_endpoint

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info(f"Starting {settings.api_title} v{settings.api_version}")
    yield
    logger.info("Shutting down CV Classification API")


app = FastAPI(title=settings.api_title, version=settings.api_version, lifespan=lifespan)

# Security middleware
app.add_middleware(TimingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Routes with versioning
app.include_router(inference.router, prefix="/v1/inference", tags=["inference"])
app.include_router(training.router, prefix="/v1/training", tags=["training"])
app.include_router(metrics.router, prefix="/v1/metrics", tags=["metrics"])


# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(detail=exc.detail).model_dump())


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content=ErrorResponse(detail="Internal server error").model_dump())


@app.get("/")
def root():
    """Root endpoint"""
    logger.info("Root endpoint accessed")
    return {"status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return metrics_endpoint()
