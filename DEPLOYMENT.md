# Gold Standard Compliance - Phase 1 Complete

## Overview
Computer Vision Classification Suite upgraded to Gold Standard compliance with comprehensive security, testing, observability, and best practices.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Security Middleware Stack                            │  │
│  │  - Rate Limiting (100 req/min)                        │  │
│  │  - Request ID Tracking                                │  │
│  │  - Security Headers (OWASP)                           │  │
│  │  - Timing Metrics                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  API Routes (Versioned /v1/*)                         │  │
│  │  - /v1/inference/predict                              │  │
│  │  - /v1/training/*                                     │  │
│  │  - /v1/metrics/*                                      │  │
│  │  - /health                                            │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ML Pipeline                                          │  │
│  │  - PyTorch CNN (64x64 images)                         │  │
│  │  - TensorFlow/Keras models                            │  │
│  │  - ONNX export support                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Deployment

### Prerequisites
- Python 3.12+
- Docker (optional)
- 4GB RAM minimum

### Local Development
```bash
# Clone repository
git clone https://github.com/SkullKrak7/Computer-Vision-Classification-Suite
cd Computer-Vision-Classification-Suite

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run backend
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest --cov=backend/app --cov=python/src
```

### Docker Deployment
```bash
# Build image
docker build -f docker/Dockerfile.python -t cv-classification:latest .

# Run container
docker run -p 8000:8000 -e LOG_LEVEL=INFO cv-classification:latest
```

## Rollback Procedures

### Application Rollback
```bash
# Rollback to previous commit
git revert HEAD
git push origin main

# Redeploy
docker pull cv-classification:previous-tag
docker stop cv-classification
docker run -d --name cv-classification cv-classification:previous-tag
```

### Configuration Rollback
```bash
# Restore previous .env
cp .env.backup .env
# Restart application
systemctl restart cv-classification
```

## Monitoring

### Health Checks
- **Endpoint**: `GET /health`
- **Expected**: `{"status": "healthy"}`
- **Frequency**: Every 30s

### Logs
- **Format**: Structured JSON (timestamp, level, message)
- **Location**: stdout (captured by Docker/systemd)
- **Levels**: DEBUG, INFO, WARNING, ERROR

### Metrics (Future)
- Request count by endpoint
- Response time percentiles (p50, p95, p99)
- Error rate
- Model inference latency

## Security

### Input Validation
- File size limit: 10MB
- Allowed MIME types: image/jpeg, image/png
- Dimension limits: 32-4096px
- Path traversal prevention

### Rate Limiting
- 100 requests/minute per IP
- Configurable via `RATE_LIMIT_PER_MINUTE`

### Headers
- X-Content-Type-Options: nosniff
- X-Frame-Options: DENY
- X-XSS-Protection: 1; mode=block
- Strict-Transport-Security: max-age=31536000

## Testing

### Coverage
- **Target**: 80%
- **Current**: 80%+ (Phase 2A complete)

### Run Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=backend/app --cov=python/src --cov-report=html

# Specific test file
pytest backend/tests/test_models.py -v
```

## API Documentation

### Swagger UI
- **URL**: http://localhost:8000/docs
- **Available in**: Development only

### Endpoints

#### POST /v1/inference/predict
Upload image for classification.

**Request**:
```bash
curl -X POST http://localhost:8000/v1/inference/predict \
  -F "file=@image.jpg"
```

**Response**:
```json
{
  "predictions": [
    {"class_id": 0, "class_name": "buildings", "confidence": 0.95},
    {"class_id": 1, "class_name": "forest", "confidence": 0.03}
  ],
  "inference_time": 0.123
}
```

#### GET /health
Health check endpoint.

**Response**:
```json
{"status": "healthy"}
```

## Troubleshooting

### Model Not Loading
**Error**: `503 Model not loaded`
**Solution**: Ensure `models/pytorch_cnn_tuned.pth` exists

### Rate Limit Exceeded
**Error**: `429 Too Many Requests`
**Solution**: Wait 1 minute or increase `RATE_LIMIT_PER_MINUTE`

### File Too Large
**Error**: `413 Payload Too Large`
**Solution**: Reduce image size or increase `MAX_FILE_SIZE_MB`

## Contributing
See CONTRIBUTING.md for development guidelines.

## License
MIT License - see LICENSE file.
