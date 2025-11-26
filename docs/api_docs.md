# API Documentation

## Backend API

Base URL: `http://localhost:8000`

### Inference

**POST** `/api/inference/predict`
- Upload image for classification
- Returns: predictions with confidence scores

### Training

**POST** `/api/training/start`
- Start model training job
- Body: `{dataset_path, model_type, epochs, batch_size}`

**GET** `/api/training/status/{job_id}`
- Get training progress

### Metrics

**GET** `/api/metrics/model/{model_id}`
- Get model performance metrics

## Interactive Docs

Visit `http://localhost:8000/docs` for Swagger UI

## Example Usage

```bash
# Start backend
uvicorn backend.app.main:app --reload

# Predict
curl -X POST "http://localhost:8000/api/inference/predict" \
  -F "file=@image.jpg"
```
