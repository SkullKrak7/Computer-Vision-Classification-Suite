# Computer Vision Classification Suite

Production-ready ML system with Python training, C++ inference, React frontend, and FastAPI backend.

[![Tests](https://img.shields.io/badge/tests-51%20passing-success)](docs/TESTING.md)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](docs/TESTING.md)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](python/)
[![C++](https://img.shields.io/badge/C++-17-orange)](cpp/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Quick Start

```bash
# Setup
make setup && source venv/bin/activate

# Train models
make train

# Run tests (51 tests, 93% coverage)
make test

# Start all services
docker-compose up
```

**Access**: [Frontend](http://localhost:3000) | [API](http://localhost:8000) | [Docs](http://localhost:8000/docs)

## Features

[x] **93% Test Coverage** - Comprehensive test suite  
[x] **GPU Accelerated** - 15x speedup with PyTorch AMP  
[x] **Production Ready** - Monitoring, security, CI/CD  
[x] **Multi-Framework** - PyTorch, TensorFlow, SVM, KNN  
[x] **C++ Inference** - 2-3x faster than Python  
[x] **Real-time API** - FastAPI with WebSocket  

## Performance

| Model | Accuracy | Speed |
|-------|----------|-------|
| TensorFlow MobileNetV2 | 88.92% | 25 img/s |
| PyTorch CNN | 87.28% | 70 img/s |
| C++ ONNX | - | 50-65 img/s |

## Documentation

 [Testing Guide](docs/TESTING.md) - 51 tests, 93% coverage  
 [Monitoring](docs/MONITORING.md) - Prometheus, Grafana, alerts  
 [Security](docs/SECURITY.md) - Scanning, best practices  
 [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues  
 [Deployment](docs/DEPLOYMENT.md) - Production guide  
📝 [Changelog](CHANGELOG.md) - Release history  

## Project Structure

```
├── python/         # ML training (17 modules, OOP design)
├── cpp/            # C++ inference engine (ONNX Runtime)
├── frontend/       # React UI with live inference
├── backend/        # FastAPI server (51 tests, 93% coverage)
├── docs/           # Comprehensive documentation
├── monitoring/     # Prometheus + Grafana + Alertmanager
└── tests/          # Test suite with CI/CD
```

## Installation

### Docker (Recommended)
```bash
docker-compose up --build
```

### Manual Setup
```bash
# Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train models
python python/scripts/auto_tune.py

# Start backend
uvicorn backend.app.main:app --reload

# Start frontend (new terminal)
cd frontend && npm install && npm run dev
```

### C++ Inference
```bash
cd cpp && mkdir build && cd build
cmake .. && make -j$(nproc)
./cv_inference ../models/onnx/model.onnx image.jpg
```

## Usage

### Training
```python
from src.models.deep_learning import PyTorchCNNClassifier

model = PyTorchCNNClassifier(num_classes=6, use_amp=True)
model.train(X_train, y_train, label_map, epochs=20)
model.save("models/pytorch/model.pth")
```

### Inference API
```bash
curl -X POST http://localhost:8000/v1/inference/predict \
  -F "file=@image.jpg"
```

### Monitoring
```bash
# Start monitoring stack
docker-compose up prometheus grafana

# Access dashboards
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

## API Documentation

Interactive API docs: http://localhost:8000/docs

**Endpoints**:
- `POST /v1/inference/predict` - Image classification
- `GET /v1/metrics/model/{id}` - Model metrics
- `POST /v1/training/start` - Start training job
- `GET /v1/training/status/{id}` - Training status
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Testing

```bash
# All tests
make test

# Specific tests
pytest backend/tests/test_inference.py -v

# With coverage
pytest backend/tests/ --cov=backend/app --cov-report=term-missing

# C++ tests
cd cpp/build && ctest
```

**Test Suite**: 51 tests, 93% coverage  
**CI/CD**: Automated testing, linting, security scanning  
**Quality**: Black, isort, ruff, mypy, bandit  

## Requirements

- Python 3.12+
- PyTorch 2.0+ with CUDA
- TensorFlow 2.18+ with GPU
- Node.js 18+
- C++17 compiler
- CMake 3.15+
- Docker & Docker Compose

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Run tests (`make test`)
4. Commit changes (`git commit -m 'Add feature'`)
5. Push to branch (`git push origin feature/amazing`)
6. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file

## Contact

GitHub: [@SkullKrak7](https://github.com/SkullKrak7)

---

**Status**: Production Ready | 93% Coverage | GPU Optimized | Docker Ready
