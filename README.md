# Computer Vision Classification Suite

Multi-language ML project combining Python training, C++ inference, and JavaScript visualization for image classification tasks.

## Project Structure

Computer-Vision-Classification-Suite/
├── python/ # ML training pipeline
├── cpp/ # C++ real-time inference
├── frontend/ # React dashboard
├── backend/ # FastAPI server
├── datasets/ # Training data
└── models/ # Trained models

## Features

- Modern dataset loader with auto-format detection
- Multiple model architectures (KNN, SVM, PyTorch CNN, TensorFlow MobileNetV2)
- Interactive webcam capture tool
- ONNX export for cross-platform deployment
- Comprehensive benchmarking
- Real-time inference

## Quick Start

### 1. Setup Environment

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r python/requirements.txt


### 2. Capture Data

python python/src/data/capture.py


### 3. Train Models

Train baseline models
python python/scripts/train_baseline.py

Train deep learning models
python python/scripts/train_cnn.py --model pytorch
python python/scripts/train_cnn.py --model tensorflow

### 4. Benchmark

python python/scripts/benchmark_all.py

### 5. Export to ONNX

python python/src/export/onnx_exporter.py models/pytorch/cnn_model.pth

## Requirements

- Python 3.10+
- PyTorch 2.0+
- TensorFlow 2.13+
- OpenCV 4.8+
- scikit-learn 1.3+

## Model Performance

Results on custom dataset:

| Model | Accuracy | Inference Time |
|-------|----------|----------------|
| TensorFlow MobileNetV2 | TBD | TBD ms/sample |
| PyTorch CNN | TBD | TBD ms/sample |
| SVM | TBD | TBD ms/sample |
| KNN | TBD | TBD ms/sample |

## License

MIT License