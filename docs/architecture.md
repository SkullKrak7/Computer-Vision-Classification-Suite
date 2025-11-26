# Architecture Overview

**Built with Kiro CLI** - System architecture designed and implemented using Amazon's AI-powered development assistant.

## Project Structure

```
├── python/          # Python ML implementation
│   ├── src/        # Source modules
│   └── scripts/    # Training/inference scripts
├── backend/        # FastAPI REST API
├── cpp/            # C++ inference engine
├── frontend/       # React web interface
├── models/         # Trained models
├── datasets/       # Training data
└── configs/        # Configuration files
```

## Components

### Python ML Pipeline
- Data loading and augmentation
- Model training (PyTorch/TensorFlow)
- Evaluation and benchmarking
- ONNX export

### Backend API
- FastAPI REST endpoints
- Model serving
- Training job management

### C++ Inference
- ONNX Runtime integration
- High-performance inference
- Minimal dependencies

### Frontend
- React web interface
- Real-time visualization
- Model comparison

## Data Flow

1. Load dataset → Preprocess → Augment
2. Train model → Evaluate → Export ONNX
3. Deploy via API or C++ engine
4. Monitor metrics via frontend
