# Python Implementation Guide

## Quick Start

```bash
# Setup
make setup
source venv/bin/activate

# Train models
python python/scripts/train_cnn.py

# Test
make test
```

## Modules

### Data
- `DatasetLoader`: Load datasets from folders/CSV
- `augment_image`: Apply data augmentation

### Models
- `PyTorchCNNClassifier`: PyTorch CNN with GPU support
- `TFMobileNetClassifier`: TensorFlow MobileNetV2

### Training
- `Trainer`: Generic model trainer
- `TrainingConfig`: Hyperparameter configuration

### Evaluation
- `compute_metrics`: Calculate accuracy, precision, recall, F1
- `benchmark_inference`: Measure inference speed

### Export
- `export_model`: Export to ONNX format

## GPU Acceleration

See `GPU_OPTIMIZATION.md` for detailed GPU setup and optimization guide.

## Examples

### Train PyTorch Model
```python
from models.deep_learning import PyTorchCNNClassifier
from data import DatasetLoader

loader = DatasetLoader('datasets/intel_images')
(X_train, X_test, y_train, y_test), label_map = loader.load()

model = PyTorchCNNClassifier(num_classes=len(label_map), use_amp=True)
model.train(X_train, y_train, label_map, epochs=20)
model.save('models/pytorch/my_model.pth')
```

### Export to ONNX
```python
from export import export_model
export_model('models/pytorch/my_model.pth', 'models/onnx')
```
